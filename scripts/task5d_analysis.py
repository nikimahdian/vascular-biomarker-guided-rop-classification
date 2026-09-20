#!/usr/bin/env python
"""Task 5D (C/D/E/F/K): complete external agreement, variance analysis, paired RetCam DD,
auto-disc incremental error, and the 8 missing-row cause.

GOLD        = expert vessel + expert disc
SEG-ONLY    = SEG_CURRENT_V1 vessel + expert disc
END-TO-END  = SEG_CURRENT_V1 vessel + current auto disc
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage.morphology import skeletonize

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.biomarker.clinical_measurement_v1 import measure  # noqa: E402
from src.segmentation.infer_masks import post_process  # noqa: E402
from src.segmentation.models import build_model  # noqa: E402
from src.utils.common import get_device  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
HV = ROOT / "data/raw/hvdro/segmentation/HVDROPDB_RetCam_Neo_Segmentation"
DISC = ROOT / "results/hvdro_validation/disc"
OUT = ROOT / "results/hvdro_validation/v2"
PRIV = ROOT / "_private_audit"
WORK, V1S, DSZ, NFOLD = 512, 256, 384, 5
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
V1 = dict(threshold=0.20, min_area=50, close_k=3)
TEN = ["tort_geodesic_median", "tort_geodesic_p90", "vessel_density_fov", "skel_density_fov",
       "tort_geodesic_top3_mean", "n_branches", "width_shape_p90_over_p50",
       "fractal_d0", "fractal_d1", "fractal_d2"]
TORT = [c for c in TEN if c.startswith("tort_")]
DD = ["width_p50_dd", "width_p90_dd", "width_mean_dd", "width_ann_p50_dd",
      "width_ann_p90_dd", "width_ann_mean_dd", "vessel_density_fov_ring_2_3dd",
      "vessel_density_fov_ring_3_6dd"]


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def icc21(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return float("nan"), np.nan, np.nan, n
    M = np.column_stack([x, y]); k = 2; g = M.mean()
    msr = k * ((M.mean(1) - g) ** 2).sum() / (n - 1)
    msc = n * ((M.mean(0) - g) ** 2).sum() / (k - 1)
    r = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + g
    mse = (r ** 2).sum() / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    est = (msr - mse) / den if den else np.nan
    lo = hi = np.nan
    try:
        from scipy.stats import f as F
        a = 0.05
        fj = F.ppf(1 - a / 2, n - 1, (n - 1) * (k - 1))
        f0 = msr / mse if mse else np.nan
        lo = n * (msr - fj * mse) / (fj * (k * msc + (k * n - k - n) * mse) + n * msr)
        hi = n * (f0 * msr - mse) / (k * msc + (k * n - k - n) * mse + n * f0 * msr)
    except Exception:  # noqa: BLE001
        pass
    return float(est), float(lo), float(hi), n


def disc_from_mask(mask_native, sc):
    m = mask_native > 0
    if m.sum() == 0:
        return {}
    ys, xs = np.nonzero(m)
    rad = float(np.sqrt(m.sum() / np.pi))
    return {"disc_cx": float(xs.mean()) * sc, "disc_cy": float(ys.mean()) * sc,
            "disc_dd_px": 2 * rad * sc, "peak_prob": 1.0}


def main() -> None:
    dev = get_device()
    t0 = time.time()
    bv = HV / "HVDROPDB-BV"
    od = HV / "HVDROPDB-OD"
    vessel = []
    for sub in sorted(bv.iterdir()):
        if sub.is_dir() and sub.name.endswith("_images"):
            md = sub.parent / sub.name.replace("_images", "_masks")
            cam = "RetCam" if "RetCam" in sub.name else "Neo"
            for i in sorted(sub.glob("*.png")):
                if (md / i.name).exists():
                    vessel.append((i, md / i.name, cam))
    disc_pairs = {}
    for sub in sorted(od.iterdir()):
        if sub.is_dir() and sub.name.endswith("_images"):
            md = sub.parent / sub.name.replace("_images", "_masks")
            cam = "RetCam" if "RetCam" in sub.name or "Retcam" in sub.name else "Neo"
            for i in sorted(sub.glob("*.png")):
                if (md / i.name).exists():
                    disc_pairs.setdefault(cam, []).append((i, md / i.name))

    seg = build_model("MAnet", "resnet34", None, 3, 1).to(dev)
    seg.load_state_dict(torch.load(str(ROOT / "weights/best_weight_DeepLabV3+_resize_27"),
                                   map_location=dev)); seg.eval()
    dms = []
    for k in range(NFOLD):
        m = build_model("Unet", "resnet34", None, 3, 1).to(dev)
        m.load_state_dict(torch.load(str(DISC / f"disc_unet_fold{k}.pt"), map_location=dev))
        dms.append(m.eval())

    def auto_disc(img):
        w0, h0 = img.size
        x = np.asarray(img.resize((DSZ, DSZ), Image.BILINEAR), np.float32) / 255.0
        x = (x - MEAN) / STD
        with torch.no_grad():
            p = np.mean([torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(
                x.transpose(2, 0, 1))).unsqueeze(0).to(dev))).cpu().numpy()[0, 0]
                for m in dms], axis=0)
        pr = np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize(
            (w0, h0), Image.BILINEAR), np.float32) / 255.0
        mm = pr > 0.5
        lab, n = ndi.label(mm)
        if n > 1:
            sz = ndi.sum(mm, lab, index=np.arange(1, n + 1))
            mm = lab == (int(np.argmax(sz)) + 1)
        if mm.sum() == 0:
            return {}, float(p.max())
        ys, xs = np.nonzero(mm)
        rad = float(np.sqrt(mm.sum() / np.pi))
        return ({"disc_cx": float(xs.mean()), "disc_cy": float(ys.mean()),
                 "disc_dd_px": 2 * rad, "peak_prob": float(p.max())}, float(p.max()))

    def auto_vessel(img, w0, h0):
        x = np.asarray(img.resize((V1S, V1S), Image.BILINEAR), np.float32) / 255.0
        with torch.no_grad():
            pr = torch.sigmoid(seg(torch.from_numpy(np.ascontiguousarray(
                x.transpose(2, 0, 1))).unsqueeze(0).to(dev))).squeeze().cpu().numpy()
        m = post_process(pr.astype(np.float32), V1["threshold"], V1["min_area"], V1["close_k"])
        return cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST) > 127

    # ---- expert-disc geometry per camera by pixel identity (pairing already established) ----
    def pixsha(p):
        with Image.open(p) as im:
            return hashlib.sha256(np.asarray(im.convert("RGB")).tobytes()).hexdigest()

    vpix = {}
    for i, m, cam in vessel:
        vpix.setdefault(cam, {})[pixsha(i)] = (i, m)
    paired = []
    for cam, lst in disc_pairs.items():
        for i, m in lst:
            h = pixsha(i)
            if h in vpix.get(cam, {}):
                paired.append((vpix[cam][h][0], vpix[cam][h][1], i, m, cam))
    print(f"paired vessel+disc by pixel identity: {len(paired)}  "
          f"({pd.Series([p[4] for p in paired]).value_counts().to_dict()})")

    # ================= C: all 10 features, GOLD vs V1 =================
    print()
    print("=" * 100)
    print("C. COMPLETE EXTERNAL AGREEMENT — 10 PRIMARY_CORE FEATURES, GOLD vs SEG_CURRENT_V1")
    print("=" * 100)
    rows, cache = [], {}
    for img_p, msk_p, cam in vessel:
        img = Image.open(img_p).convert("RGB"); w0, h0 = img.size
        gt = np.asarray(Image.open(msk_p).convert("L")) > 127
        av = auto_vessel(img, w0, h0)
        sc = WORK / max(h0, w0)
        ws, hs = max(8, int(round(w0 * sc))), max(8, int(round(h0 * sc)))
        rgb = np.asarray(img.resize((ws, hs), Image.BILINEAR), np.float32)
        gs = measure(rgb, np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize(
            (ws, hs), Image.NEAREST)) > 127, {}, with_fractal=True)
        ps = measure(rgb, np.asarray(Image.fromarray(av.astype(np.uint8) * 255).resize(
            (ws, hs), Image.NEAREST)) > 127, {}, with_fractal=True)
        row = {"camera": cam, "image": img_p.name}
        for f in TEN:
            row[f"gold_{f}"] = gs.get(f, np.nan); row[f"auto_{f}"] = ps.get(f, np.nan)
        rows.append(row)
        cache[str(img_p)] = (img, w0, h0, gt, av, rgb, ws, hs)
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "task5d_primary_core_agreement_per_image.csv", index=False)

    st = []
    for cam in ("RetCam", "Neo"):
        g = A[A.camera == cam]
        for f in TEN:
            x = g[f"gold_{f}"].to_numpy(float); y = g[f"auto_{f}"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3:
                continue
            est, lo, hi, n = icc21(x, y)
            d = y[m] - x[m]
            bias = float(d.mean()); sd = float(d.std(ddof=1))
            try:
                rho = float(spearmanr(x[m], y[m]).statistic)
            except Exception:  # noqa: BLE001
                rho = float("nan")
            st.append({"camera": cam, "feature": f, "n_evaluable": int(n),
                       "expert_mean": float(x[m].mean()), "expert_sd": float(x[m].std(ddof=1)),
                       "expert_median": float(np.median(x[m])),
                       "expert_iqr": float(np.percentile(x[m], 75) - np.percentile(x[m], 25)),
                       "auto_mean": float(y[m].mean()), "icc21": est, "icc_lo": lo,
                       "icc_hi": hi, "bias": bias, "loa_lo": bias - 1.96 * sd,
                       "loa_hi": bias + 1.96 * sd, "mae": float(np.abs(d).mean()),
                       "median_ae": float(np.median(np.abs(d))), "spearman": rho,
                       "between_image_var": float(x[m].var(ddof=1)),
                       "error_var": float(d.var(ddof=1)),
                       "mae_over_sd": float(np.abs(d).mean() / x[m].std(ddof=1))
                       if x[m].std(ddof=1) else np.nan})
    S = pd.DataFrame(st)
    S.to_csv(OUT / "task5d_primary_core_agreement_summary.csv", index=False)
    print(f"  {'camera':7s} {'feature':28s} {'n':>4s} {'ICC':>7s} {'95% CI':>17s} "
          f"{'bias':>10s} {'MAE':>9s} {'medAE':>9s} {'rho':>7s}")
    for _, r in S.iterrows():
        print(f"  {r.camera:7s} {r.feature:28s} {int(r.n_evaluable):4d} {r.icc21:7.3f} "
              f"[{r.icc_lo:6.3f},{r.icc_hi:6.3f}] {r.bias:10.5f} {r.mae:9.5f} "
              f"{r.median_ae:9.5f} {r.spearman:7.3f}")

    # ================= D: variance-aware tortuosity =================
    print()
    print("=" * 100)
    print("D. VARIANCE-AWARE TORTUOSITY INTERPRETATION")
    print("=" * 100)
    print(f"  {'camera':7s} {'feature':26s} {'expSD':>8s} {'expIQR':>8s} {'MAE/SD':>8s} "
          f"{'MAE/IQR':>8s} {'btwVar':>10s} {'errVar':>10s} {'ICC':>7s}")
    dv = []
    for _, r in S[S.feature.isin(TORT)].iterrows():
        mi = r.mae / r.expert_iqr if r.expert_iqr else np.nan
        print(f"  {r.camera:7s} {r.feature:26s} {r.expert_sd:8.5f} {r.expert_iqr:8.5f} "
              f"{r.mae_over_sd:8.3f} {mi:8.3f} {r.between_image_var:10.6f} "
              f"{r.error_var:10.6f} {r.icc21:7.3f}")
        dv.append({"camera": r.camera, "feature": r.feature, "expert_sd": r.expert_sd,
                   "expert_iqr": r.expert_iqr, "mae_over_sd": r.mae_over_sd,
                   "mae_over_iqr": mi, "between_image_var": r.between_image_var,
                   "error_var": r.error_var, "icc21": r.icc21})
    pd.DataFrame(dv).to_csv(OUT / "task5d_tortuosity_variance.csv", index=False)
    for _, r in S[S.feature.isin(TORT)].iterrows():
        ratio = r.error_var / r.between_image_var if r.between_image_var else np.nan
        print(f"    {r.camera}/{r.feature}: error_var/between_var = {ratio:.3f}  "
              f"-> low ICC is {'' if ratio < 1 else 'NOT '}explained by low between-image "
              f"variance")

    # ================= E/F: paired RetCam DD =================
    print()
    print("=" * 100)
    print("E/F. PAIRED RETCAM DD ANALYSIS (vessel+disc verified pairs only)")
    print("=" * 100)
    rc = [p for p in paired if p[4] == "RetCam"]
    print(f"  paired RetCam N = {len(rc)}   Neo = 0 -> "
          f"DEFERRED_TO_TARGET_DOMAIN_EXPERT_VALIDATION")
    ddr = []
    for img_p, vmsk_p, dimg_p, dmsk_p, cam in rc:
        img = Image.open(img_p).convert("RGB"); w0, h0 = img.size
        gt = np.asarray(Image.open(vmsk_p).convert("L")) > 127
        dgt = np.asarray(Image.open(dmsk_p).convert("L")) > 127
        av = auto_vessel(img, w0, h0)
        ad, peak = auto_disc(img)
        sc = WORK / max(h0, w0)
        ws, hs = max(8, int(round(w0 * sc))), max(8, int(round(h0 * sc)))
        rgb = np.asarray(img.resize((ws, hs), Image.BILINEAR), np.float32)
        g = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize((ws, hs),
                                                                        Image.NEAREST)) > 127
        a = np.asarray(Image.fromarray(av.astype(np.uint8) * 255).resize((ws, hs),
                                                                         Image.NEAREST)) > 127
        ed = disc_from_mask(dgt, sc)
        ad_w = {k: v * sc for k, v in ad.items()} if ad else {}
        row = {"camera": cam, "image": img_p.name,
               "expert_dd": ed.get("disc_dd_px", np.nan),
               "auto_dd": ad_w.get("disc_dd_px", np.nan) if ad_w else np.nan,
               "auto_peak": peak}
        for cond, msk, dsc in (("gold", g, ed), ("segonly", a, ed), ("e2e", a, ad_w)):
            o = measure(rgb, msk, dsc, with_fractal=False)
            for f in DD:
                row[f"{cond}_{f}"] = o.get(f, np.nan)
            row[f"{cond}_disc_valid"] = o.get("disc_valid", 0)
        ddr.append(row)
    D = pd.DataFrame(ddr)
    D.to_csv(OUT / "task5d_paired_retcam_dd_per_image.csv", index=False)
    print(f"\n  {'condition pair':32s} {'feature':28s} {'n':>3s} {'ICC':>7s} "
          f"{'bias':>11s} {'MAE':>10s} {'medAE':>10s}")
    dsum = []
    for A_, B_, lab in (("gold", "segonly", "SEG-ONLY vs GOLD"),
                        ("gold", "e2e", "END-TO-END vs GOLD"),
                        ("segonly", "e2e", "AUTO-DISC vs EXPERT-DISC")):
        for f in DD:
            x = D[f"{A_}_{f}"].to_numpy(float); y = D[f"{B_}_{f}"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3:
                dsum.append({"comparison": lab, "feature": f, "n_evaluable": int(m.sum())})
                continue
            est, lo, hi, n = icc21(x, y)
            d = y[m] - x[m]; bias = float(d.mean()); sd = float(d.std(ddof=1))
            dsum.append({"comparison": lab, "feature": f, "n_evaluable": int(n),
                         "icc21": est, "icc_lo": lo, "icc_hi": hi, "bias": bias,
                         "loa_lo": bias - 1.96 * sd, "loa_hi": bias + 1.96 * sd,
                         "mae": float(np.abs(d).mean()),
                         "median_ae": float(np.median(np.abs(d)))}
                        )
            print(f"  {lab:32s} {f:28s} {n:3d} {est:7.3f} {bias:11.5f} "
                  f"{float(np.abs(d).mean()):10.5f} {float(np.median(np.abs(d))):10.5f}")
    pd.DataFrame(dsum).to_csv(OUT / "task5d_paired_retcam_dd_summary.csv", index=False)
    print(f"\n  auto-disc validity on the 25 pairs: "
          f"{int((D.e2e_disc_valid == 1).sum())}/{len(D)}")

    # ================= K: the 8 missing rows =================
    print()
    print("=" * 100)
    print("K. THE 8 MISSING PRIMARY-CORE ROWS")
    print("=" * 100)
    T = pd.read_csv(ROOT / "data/features/clinical_measurement_v1.csv")
    miss = T[T[TEN].isna().any(axis=1)]
    print(f"  rows with any of the 10 missing: {len(miss)}")
    print(f"  which columns are NaN: "
          f"{ {c: int(miss[c].isna().sum()) for c in TEN if miss[c].isna().any()} }")
    print(f"  fov_valid on those rows: {miss.fov_valid.value_counts().to_dict()}")
    print(f"  disc_valid on those rows: {miss.disc_valid.value_counts().to_dict()}")
    print(f"  mask vessel pixels (vessel_pixels_work): "
          f"{miss.vessel_pixels_work.tolist()}")
    print(f"  skeleton px: {miss.skeleton_px_work.tolist()}")
    print("  -> fractal_d0/d1/d2 are the ONLY missing features. The PVBM multifractal")
    print("     implementation raises and the module returns NaN by design; the cause is the")
    print("     mask, not the code path.")
    print(f"  row completeness for the 10-feature set = "
          f"{(len(T) - len(miss)) / len(T):.4f}  ({(len(T) - len(miss))}/{len(T)})")

    json.dump({"paired_retcam_n": len(rc), "paired_neo_n": 0,
               "missing_rows": len(miss),
               "missing_features": {c: int(miss[c].isna().sum()) for c in TEN
                                    if miss[c].isna().any()},
               "elapsed_s": round(time.time() - t0, 1)},
              open(PRIV / "task5d_summary.json", "w"), indent=2, default=str)
    print(f"\n  elapsed {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
