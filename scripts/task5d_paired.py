#!/usr/bin/env python
"""Task 5D (E/F) rerun with the peak_prob scaling bug fixed.

Bug: the E2E condition scaled the whole auto-disc dict by the working-grid factor, including
peak_prob, so 0.988 became 0.790 and the frozen disc_valid gate (peak > 0.90) rejected all 25
images. Only geometry keys may be scaled.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi

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
WORK, V1S, DSZ, NFOLD = 512, 256, 384, 5
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
V1 = dict(threshold=0.20, min_area=50, close_k=3)
DD = ["width_p50_dd", "width_p90_dd", "width_mean_dd", "width_ann_p50_dd",
      "width_ann_p90_dd", "width_ann_mean_dd", "vessel_density_fov_ring_2_3dd",
      "vessel_density_fov_ring_3_6dd"]
SCALE_KEYS = ("disc_cx", "disc_cy", "disc_dd_px")   # peak_prob must NOT be scaled


def icc21(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, n
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


def pixsha(p):
    with Image.open(p) as im:
        return hashlib.sha256(np.asarray(im.convert("RGB")).tobytes()).hexdigest()


def main() -> None:
    dev = get_device()
    bv, od = HV / "HVDROPDB-BV", HV / "HVDROPDB-OD"
    v, d = {}, {}
    for base, store in ((bv, v), (od, d)):
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and sub.name.endswith("_images"):
                md = sub.parent / sub.name.replace("_images", "_masks")
                for i in sorted(sub.glob("*.png")):
                    if (md / i.name).exists():
                        store[pixsha(i)] = (i, md / i.name)
    pairs = [(v[h][0], v[h][1], d[h][1]) for h in v if h in d]
    print(f"paired by pixel identity: {len(pairs)}")

    seg = build_model("MAnet", "resnet34", None, 3, 1).to(dev)
    seg.load_state_dict(torch.load(str(ROOT / "weights/best_weight_DeepLabV3+_resize_27"),
                                   map_location=dev)); seg.eval()
    dms = []
    for k in range(NFOLD):
        m = build_model("Unet", "resnet34", None, 3, 1).to(dev)
        m.load_state_dict(torch.load(str(DISC / f"disc_unet_fold{k}.pt"), map_location=dev))
        dms.append(m.eval())

    rows = []
    for img_p, vmsk_p, dmsk_p in pairs:
        img = Image.open(img_p).convert("RGB"); w0, h0 = img.size
        gt = np.asarray(Image.open(vmsk_p).convert("L")) > 127
        dgt = np.asarray(Image.open(dmsk_p).convert("L")) > 127
        x = np.asarray(img.resize((V1S, V1S), Image.BILINEAR), np.float32) / 255.0
        with torch.no_grad():
            pr = torch.sigmoid(seg(torch.from_numpy(np.ascontiguousarray(
                x.transpose(2, 0, 1))).unsqueeze(0).to(dev))).squeeze().cpu().numpy()
        av = cv2.resize(post_process(pr.astype(np.float32), V1["threshold"], V1["min_area"],
                                     V1["close_k"]), (w0, h0),
                        interpolation=cv2.INTER_NEAREST) > 127
        xd = np.asarray(img.resize((DSZ, DSZ), Image.BILINEAR), np.float32) / 255.0
        xd = (xd - MEAN) / STD
        with torch.no_grad():
            p3 = np.mean([torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(
                xd.transpose(2, 0, 1))).unsqueeze(0).to(dev))).cpu().numpy()[0, 0]
                for m in dms], axis=0)
        pw = np.asarray(Image.fromarray((p3 * 255).astype(np.uint8)).resize(
            (w0, h0), Image.BILINEAR), np.float32) / 255.0
        mm = pw > 0.5
        lab, n = ndi.label(mm)
        if n > 1:
            sz = ndi.sum(mm, lab, index=np.arange(1, n + 1))
            mm = lab == (int(np.argmax(sz)) + 1)
        ad = {}
        if mm.sum():
            ys, xs = np.nonzero(mm)
            ad = {"disc_cx": float(xs.mean()), "disc_cy": float(ys.mean()),
                  "disc_dd_px": 2 * float(np.sqrt(mm.sum() / np.pi)),
                  "peak_prob": float(p3.max())}
        sc = WORK / max(h0, w0)
        ws, hs = max(8, int(round(w0 * sc))), max(8, int(round(h0 * sc)))
        rgb = np.asarray(img.resize((ws, hs), Image.BILINEAR), np.float32)
        g = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize(
            (ws, hs), Image.NEAREST)) > 127
        a = np.asarray(Image.fromarray(av.astype(np.uint8) * 255).resize(
            (ws, hs), Image.NEAREST)) > 127
        ys, xs = np.nonzero(dgt)
        ed = {"disc_cx": float(xs.mean()) * sc, "disc_cy": float(ys.mean()) * sc,
              "disc_dd_px": 2 * float(np.sqrt(dgt.sum() / np.pi)) * sc, "peak_prob": 1.0}
        adw = {k: (v_ * sc if k in SCALE_KEYS else v_) for k, v_ in ad.items()}
        row = {"image": img_p.name, "auto_peak": float(p3.max()), "expert_dd": ed["disc_dd_px"],
               "auto_dd": adw.get("disc_dd_px", np.nan)}
        for cond, msk, dsc in (("gold", g, ed), ("segonly", a, ed), ("e2e", a, adw)):
            o = measure(rgb, msk, dsc, with_fractal=False)
            for f in DD:
                row[f"{cond}_{f}"] = o.get(f, np.nan)
            row[f"{cond}_disc_valid"] = int(o.get("disc_valid", 0))
        rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "task5d_paired_retcam_dd_per_image.csv", index=False)
    print(f"  disc_valid: gold={int((D.gold_disc_valid == 1).sum())}/{len(D)}  "
          f"segonly={int((D.segonly_disc_valid == 1).sum())}/{len(D)}  "
          f"e2e={int((D.e2e_disc_valid == 1).sum())}/{len(D)}")
    print()
    print(f"  {'comparison':28s} {'feature':30s} {'n':>3s} {'ICC':>7s} {'95% CI':>17s} "
          f"{'bias':>11s} {'MAE':>10s} {'medAE':>10s}")
    out = []
    for A_, B_, lab in (("gold", "segonly", "SEG-ONLY vs GOLD"),
                        ("gold", "e2e", "END-TO-END vs GOLD"),
                        ("segonly", "e2e", "AUTO-DISC vs EXPERT-DISC")):
        for f in DD:
            x = D[f"{A_}_{f}"].to_numpy(float); y = D[f"{B_}_{f}"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3:
                out.append({"comparison": lab, "feature": f, "n_evaluable": int(m.sum())})
                print(f"  {lab:28s} {f:30s} {int(m.sum()):3d}   (not evaluable)")
                continue
            est, lo, hi, n = icc21(x, y)
            dd = y[m] - x[m]; bias = float(dd.mean()); sd = float(dd.std(ddof=1))
            out.append({"comparison": lab, "feature": f, "n_evaluable": int(n), "icc21": est,
                        "icc_lo": lo, "icc_hi": hi, "bias": bias, "loa_lo": bias - 1.96 * sd,
                        "loa_hi": bias + 1.96 * sd, "mae": float(np.abs(dd).mean()),
                        "median_ae": float(np.median(np.abs(dd)))})
            print(f"  {lab:28s} {f:30s} {n:3d} {est:7.3f} [{lo:6.3f},{hi:6.3f}] "
                  f"{bias:11.5f} {float(np.abs(dd).mean()):10.5f} "
                  f"{float(np.median(np.abs(dd))):10.5f}")
    pd.DataFrame(out).to_csv(OUT / "task5d_paired_retcam_dd_summary.csv", index=False)
    O = pd.DataFrame(out)
    inc = O[(O.comparison == "AUTO-DISC vs EXPERT-DISC") & O.mae.notna()]
    if len(inc):
        print()
        print("  F. AUTO-DISC INCREMENTAL ERROR (median MAE over the evaluable DD features):")
        print(f"    AUTO vessel + AUTO disc vs AUTO vessel + EXPERT disc: MAE median "
              f"{inc.mae.median():.5f}, bias median {inc.bias.median():.5f}")
    json.dump({"paired_n": len(D), "e2e_valid": int((D.e2e_disc_valid == 1).sum()),
               "incremental_mae_median": float(inc.mae.median()) if len(inc) else None},
              open(ROOT / "_private_audit/task5d_paired_dd.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
