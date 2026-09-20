#!/usr/bin/env python
"""Task 5C (Y3, Q1, Q2, R): the V1/HVDROPDB/current-disc work required regardless of V2.

Runs SEG_CURRENT_V1 on the 100 expert-vessel images, the current 5-fold disc ensemble on the
100 expert-disc images, and CLINICAL_MEASUREMENT_V1 disc-independent biomarkers on expert vs
automatic vessel masks. RetCam and Neo are always reported separately.
"""
from __future__ import annotations

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
OUT.mkdir(parents=True, exist_ok=True)
V1 = dict(size=256, threshold=0.20, min_area=50, close_k=3)
DISC_SIZE, NFOLD = 384, 5
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
WORK = 512

DISK_INDEP = ["vessel_density_fov", "skel_density_fov", "tort_geodesic_median",
              "tort_geodesic_p90", "tort_geodesic_top3_mean", "width_shape_p90_over_p50",
              "n_branches"]


def camera_of(d: Path) -> str:
    n = d.name
    return "RetCam" if "RetCam" in n or "Retcam" in n else "Neo"


def pairs(kind: str):
    """(image_path, expert_mask_path, camera) for HVDROPDB-BV or HVDROPDB-OD."""
    base = HV / ("HVDROPDB-BV" if kind == "bv" else "HVDROPDB-OD")
    out = []
    for sub in sorted(base.iterdir()):
        if not sub.is_dir() or not sub.name.endswith("_images"):
            continue
        cam = camera_of(sub)
        mdir = sub.parent / sub.name.replace("_images", "_masks")
        for img in sorted(sub.glob("*.png")):
            m = mdir / img.name
            if m.exists():
                out.append((img, m, cam, sub.name))
    return out


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else float("nan")


def cldice(pred, gt):
    sp, sg = skeletonize(pred), skeletonize(gt)
    if sp.sum() == 0 or sg.sum() == 0:
        return float("nan")
    tprec = np.logical_and(sp, gt).sum() / sp.sum()
    tsens = np.logical_and(sg, pred).sum() / sg.sum()
    return float(2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) else float("nan")


def icc21(x, y):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    M = np.column_stack([x, y])
    k = 2
    grand = M.mean()
    ms_rows = k * ((M.mean(1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((M.mean(0) - grand) ** 2).sum() / (k - 1)
    resid = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    est = (ms_rows - ms_err) / denom if denom else float("nan")
    # approximate 95% CI via the F distribution
    try:
        from scipy.stats import f as fdist

        a = 0.05
        fj = fdist.ppf(1 - a / 2, n - 1, (n - 1) * (k - 1))
        fl = fdist.ppf(1 - a / 2, (n - 1) * (k - 1), n - 1)
        f0 = ms_rows / ms_err if ms_err else np.nan
        lo = n * (ms_rows - fj * ms_err) / (fj * (k * ms_cols + (k * n - k - n) * ms_err) + n * ms_rows)
        hi = n * (f0 * ms_rows - ms_err) / (k * ms_cols + (k * n - k - n) * ms_err + n * f0 * ms_rows)
    except Exception:  # noqa: BLE001
        lo = hi = float("nan")
    return float(est), (float(lo), float(hi)), n


def main() -> None:
    dev = get_device()
    t0 = time.time()
    print(f"device={dev}")

    # ---------------- V1 vessel segmentation on 100 expert-vessel images ----------------
    print()
    print("=" * 100)
    print("V1 SEGMENTATION BENCHMARK — 100 HVDROPDB expert-vessel images")
    print("=" * 100)
    vessel = pairs("bv")
    print(f"  cohort n={len(vessel)}  "
          f"RetCam={sum(1 for v in vessel if v[2] == 'RetCam')}  "
          f"Neo={sum(1 for v in vessel if v[2] == 'Neo')}")
    seg = build_model("MAnet", "resnet34", None, 3, 1).to(dev)
    seg.load_state_dict(torch.load(str(ROOT / "weights/best_weight_DeepLabV3+_resize_27"),
                                   map_location=dev))
    seg.eval()
    vrows = []
    cache = {}
    for img_p, msk_p, cam, subname in vessel:
        img = Image.open(img_p).convert("RGB")
        w0, h0 = img.size
        gt = np.asarray(Image.open(msk_p).convert("L")) > 127
        x = np.asarray(img.resize((V1["size"], V1["size"]), Image.BILINEAR), np.float32) / 255.0
        with torch.no_grad():
            pr = torch.sigmoid(seg(torch.from_numpy(
                np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).to(dev)))
            pr = pr.squeeze().cpu().numpy().astype(np.float32)
        m256 = post_process(pr, V1["threshold"], V1["min_area"], V1["close_k"])
        pred = cv2.resize(m256, (w0, h0), interpolation=cv2.INTER_NEAREST) > 127
        tp = int(np.logical_and(pred, gt).sum())
        prec = tp / max(int(pred.sum()), 1)
        rec = tp / max(int(gt.sum()), 1)
        vrows.append({"camera": cam, "subset": subname, "image": img_p.name,
                      "dice": dice(pred, gt), "cldice": cldice(pred, gt),
                      "precision": prec, "recall": rec,
                      "gt_frac": float(gt.mean()), "pred_frac": float(pred.mean())})
        cache[str(img_p)] = (img, w0, h0, gt, pred)
    V = pd.DataFrame(vrows)
    V.to_csv(OUT / "v1_segmentation_per_image.csv", index=False)
    print(f"  {'camera':8s} {'n':>4s} {'Dice':>8s} {'clDice':>8s} {'precision':>10s} "
          f"{'recall':>8s}")
    segsum = []
    for cam, g in V.groupby("camera"):
        row = {"camera": cam, "n": len(g), "dice": g.dice.mean(), "cldice": g.cldice.mean(),
               "precision": g.precision.mean(), "recall": g.recall.mean()}
        segsum.append(row)
        print(f"  {cam:8s} {len(g):4d} {row['dice']:8.4f} {row['cldice']:8.4f} "
              f"{row['precision']:10.4f} {row['recall']:8.4f}")
    pooled = {"camera": "pooled", "n": len(V), "dice": V.dice.mean(), "cldice": V.cldice.mean(),
              "precision": V.precision.mean(), "recall": V.recall.mean()}
    print(f"  {'pooled':8s} {len(V):4d} {pooled['dice']:8.4f} {pooled['cldice']:8.4f} "
          f"{pooled['precision']:10.4f} {pooled['recall']:8.4f}   (secondary)")
    pd.DataFrame(segsum + [pooled]).to_csv(OUT / "v1_segmentation_summary.csv", index=False)

    # ---------------- current disc detector on 100 expert-disc images ----------------
    print()
    print("=" * 100)
    print("CURRENT DISC DETECTOR BENCHMARK — 100 HVDROPDB expert-disc images")
    print("=" * 100)
    dm = []
    for k in range(NFOLD):
        m = build_model("Unet", "resnet34", None, 3, 1).to(dev)
        m.load_state_dict(torch.load(str(DISC / f"disc_unet_fold{k}.pt"), map_location=dev))
        dm.append(m.eval())
    drows = []
    for img_p, msk_p, cam, subname in pairs("od"):
        img = Image.open(img_p).convert("RGB")
        w0, h0 = img.size
        g = np.asarray(Image.open(msk_p).convert("L"))
        gt = g > 127
        x = np.asarray(img.resize((DISC_SIZE, DISC_SIZE), Image.BILINEAR), np.float32) / 255.0
        x = (x - MEAN) / STD
        with torch.no_grad():
            p3 = np.mean([torch.sigmoid(m(torch.from_numpy(
                np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).to(dev))
            ).cpu().numpy()[0, 0] for m in dm], axis=0)
        pr = np.asarray(Image.fromarray((p3 * 255).astype(np.uint8)).resize(
            (w0, h0), Image.BILINEAR), np.float32) / 255.0
        pred = pr > 0.5
        lab, n = ndi.label(pred)
        if n > 1:
            sizes = ndi.sum(pred, lab, index=np.arange(1, n + 1))
            pred = lab == (int(np.argmax(sizes)) + 1)
        peak = float(p3.max())
        if pred.sum() == 0:
            drow = {"camera": cam, "image": img_p.name, "valid": 0, "peak_prob": peak,
                    "dd_dice": float("nan"), "centre_err_dd": float("nan"),
                    "diam_ratio": float("nan"), "expert_dd_px": float("nan"),
                    "auto_dd_px": float("nan")}
        else:
            ys, xs = np.nonzero(gt)
            true_c = (xs.mean(), ys.mean())
            true_r = float(np.sqrt(gt.sum() / np.pi))
            true_dd = 2 * true_r
            py, px = np.nonzero(pred)
            auto_dd = 2 * float(np.sqrt(pred.sum() / np.pi))
            drow = {"camera": cam, "image": img_p.name, "valid": 1, "peak_prob": peak,
                    "dd_dice": dice(pred, gt),
                    "centre_err_dd": float(np.hypot(px.mean() - true_c[0],
                                                    py.mean() - true_c[1]) / true_dd),
                    "diam_ratio": auto_dd / true_dd if true_dd else float("nan"),
                    "expert_dd_px": true_dd, "auto_dd_px": auto_dd}
        drow["legit"] = int(peak > 0.90 and drow.get("valid", 0) == 1)
        drows.append(drow)
    D = pd.DataFrame(drows)
    D.to_csv(OUT / "disc_benchmark_per_image.csv", index=False)
    print(f"  {'camera':8s} {'n':>4s} {'validity':>9s} {'Dice':>8s} {'centre err':>11s} "
          f"{'diam ratio':>11s} {'dice(legit)':>12s}")
    dsum = []
    for cam, g in D.groupby("camera"):
        lg = g[g.legit == 1]
        row = {"camera": cam, "n": len(g), "validity_rate": g.valid.mean(),
               "dd_dice": g.dd_dice.mean(), "centre_err_dd": g.centre_err_dd.mean(),
               "diam_ratio": g.diam_ratio.mean(),
               "dd_dice_legit_n": len(lg), "dd_dice_legit": lg.dd_dice.mean() if len(lg) else np.nan}
        dsum.append(row)
        print(f"  {cam:8s} {len(g):4d} {row['validity_rate']:9.4f} {row['dd_dice']:8.4f} "
              f"{row['centre_err_dd']:11.4f} {row['diam_ratio']:11.4f} "
              f"{row['dd_dice_legit']:12.4f}")
    pd.DataFrame(dsum).to_csv(OUT / "disc_benchmark_summary.csv", index=False)

    # ---------------- CLINICAL_MEASUREMENT_V1 agreement, GOLD vs V1 ----------------
    print()
    print("=" * 100)
    print("CLINICAL_MEASUREMENT_V1 DISC-INDEPENDENT AGREEMENT — GOLD vs SEG_CURRENT_V1")
    print("=" * 100)
    arows = []
    for img_p, msk_p, cam, subname in vessel:
        img, w0, h0, gt, pred = cache[str(img_p)]
        sc = WORK / max(h0, w0)
        ws, hs = max(8, int(round(w0 * sc))), max(8, int(round(h0 * sc)))
        rgb = np.asarray(img.resize((ws, hs), Image.BILINEAR), np.float32)
        gs = measure(rgb, np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize(
            (ws, hs), Image.NEAREST)) > 127, {}, with_fractal=False)
        ps = measure(rgb, np.asarray(Image.fromarray(pred.astype(np.uint8) * 255).resize(
            (ws, hs), Image.NEAREST)) > 127, {}, with_fractal=False)
        row = {"camera": cam, "image": img_p.name}
        for f in DISK_INDEP:
            row[f"gold_{f}"] = gs.get(f, np.nan)
            row[f"auto_{f}"] = ps.get(f, np.nan)
        arows.append(row)
    A = pd.DataFrame(arows)
    A.to_csv(OUT / "v1_biomarker_agreement_per_image.csv", index=False)
    print(f"  {'camera':8s} {'feature':28s} {'n':>4s} {'ICC(2,1)':>9s} "
          f"{'95% CI':>18s} {'bias':>10s} {'MAE':>9s} {'medAE':>9s} {'LoA':>18s}")
    st = []
    for cam in ("RetCam", "Neo", "pooled"):
        g = A if cam == "pooled" else A[A.camera == cam]
        for f in DISK_INDEP:
            x = g[f"gold_{f}"].to_numpy(float)
            y = g[f"auto_{f}"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3:
                continue
            est, ci, n = icc21(x, y)
            d = y[m] - x[m]
            bias, sd = float(d.mean()), float(d.std(ddof=1))
            row = {"camera": cam, "feature": f, "n_evaluable": int(n),
                   "icc21": est, "icc_lo": ci[0], "icc_hi": ci[1],
                   "bias": bias, "loa_lo": bias - 1.96 * sd, "loa_hi": bias + 1.96 * sd,
                   "mae": float(np.abs(d).mean()), "median_ae": float(np.median(np.abs(d))),
                   "expert_mean": float(x[m].mean())}
            st.append(row)
            print(f"  {cam:8s} {f:28s} {n:4d} {est:9.4f} "
                  f"[{ci[0]:7.4f},{ci[1]:7.4f}] {bias:10.5f} {row['mae']:9.5f} "
                  f"{row['median_ae']:9.5f} [{row['loa_lo']:8.5f},{row['loa_hi']:8.5f}]")
    S = pd.DataFrame(st)
    S.to_csv(OUT / "v1_biomarker_agreement_summary.csv", index=False)
    print(f"\n  elapsed {time.time() - t0:.0f}s")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
