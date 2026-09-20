#!/usr/bin/env python
"""Blok 1 gate -- can vessel width be measured on this data?

Evaluated on the 100 HVDROPDB-BV expert masks, at the deployed inference setting
(img_size 256, threshold 0.20, min_area 50, close_k 3).

Three estimators, all evaluated on the *expert* skeleton (so they share one grid):
  C  expert   : 2 * EDT(expert mask)              -- the reference / ceiling
  A  deployed : 2 * EDT(post_process(prob,0.20))  -- what the project would do today
  B  fwhm     : full width at half maximum of the model probability map, sampled
                perpendicular to the model's local vessel orientation

Reports per-pixel agreement and per-image (feature-level) agreement, plus the single
global calibration factor k that best maps A onto C.

Read-only on the repo; writes to results/hvdro_validation/width/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from torchvision import transforms

sys.path.insert(0, "/Users/moniaz/niki")
from src.segmentation.infer_masks import post_process  # noqa: E402
from src.segmentation.models import build_model  # noqa: E402
from src.utils.common import get_device, load_config  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
SEG = ROOT / "data/raw/hvdro/segmentation/HVDROPDB_RetCam_Neo_Segmentation"
OUT = ROOT / "results/hvdro_validation/width"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 256          # deployed inference resolution (best Dice in the earlier sweep)
MAX_HALF = 12.0     # perpendicular search half-length, native px
STEP = 0.5          # perpendicular sampling step, native px
MAX_SKEL = 3000     # subsample of expert skeleton pixels per image
SEED = 20260101

cfg = load_config()
sc = cfg["segmentation"]
THR, MIN_AREA, CLOSE_K = float(sc["threshold"]), int(sc["min_area"]), int(sc["morph_close_kernel"])
device = get_device()
print(f"device={device} arch={sc['arch']} encoder={sc['encoder']} thr={THR} size={SIZE}")

ckpt = ROOT / sc["weight_path"]
model = build_model(sc["arch"], sc["encoder"], None, int(sc["in_channels"]), 1).to(device).eval()
model.load_state_dict(torch.load(str(ckpt), map_location=device))
print("model loaded:", ckpt)

tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])


def predict(img: Image.Image) -> np.ndarray:
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(x)).squeeze().cpu().numpy()


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = x - x0, y - y0
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def orientation(prob: np.ndarray, sigma: float = 2.0):
    """Principal gradient direction (across the vessel) from the structure tensor."""
    p = cv2.GaussianBlur(prob.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)   # direction of the dominant gradient
    return np.cos(theta), np.sin(theta)            # unit vector across the vessel


def fwhm_at(prob: np.ndarray, nx: np.ndarray, ny: np.ndarray, ox: np.ndarray, oy: np.ndarray,
            frac: float = 0.5) -> np.ndarray:
    """Full width at half maximum of `prob` sampled perpendicular at (nx,ny) along (ox,oy)."""
    t = np.arange(-MAX_HALF, MAX_HALF + 1e-9, STEP)
    prof = np.empty((len(nx), len(t)), dtype=np.float32)
    for i, dt in enumerate(t):
        prof[:, i] = bilinear(prob, nx + dt * ox, ny + dt * oy)
    c = int(np.argmin(np.abs(t)))
    peak = prof.max(axis=1)
    thr = frac * peak
    left_mask = prof[:, : c + 1] < thr[:, None]
    right_mask = prof[:, c:] < thr[:, None]
    has_l = left_mask.any(axis=1)
    has_r = right_mask.any(axis=1)
    li = c - np.argmax(left_mask[:, ::-1], axis=1)
    ri = c + np.argmax(right_mask, axis=1)
    width = (ri - li) * STEP
    width[~has_l | ~has_r] = np.nan          # never dropped below half maximum -> not a vessel
    width[peak < 0.25] = np.nan              # no vessel here at all
    return width


def edt_half(binary: np.ndarray):
    """(skeleton, width map) where width = 2 * EDT, propagated to the nearest skeleton pixel."""
    if binary.sum() == 0:
        return None, None
    skel = skeletonize(binary)
    if skel.sum() == 0:
        return None, None
    w = 2.0 * ndi.distance_transform_edt(binary)
    _, idx = ndi.distance_transform_edt(~skel, return_indices=True)
    return skel, w[tuple(idx)]


def icc21(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return float("nan")
    data = np.stack([x, y], axis=1).astype(float)
    k = 2
    gm = data.mean()
    msr = k * ((data.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((data.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    sse = ((data - data.mean(axis=1, keepdims=True) - data.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den else float("nan")


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


rng = np.random.default_rng(SEED)
rows, pix_rows = [], []
cal_num = cal_den = 0.0

for cam in ("RetCam_Vessels", "Neo_Vessels"):
    idir = SEG / "HVDROPDB-BV" / f"{cam}_images"
    mdir = SEG / "HVDROPDB-BV" / f"{cam}_masks"
    stems = sorted(p.stem for p in idir.glob("*.png"))
    print(f"\n=== {cam}: {len(stems)} images ===", flush=True)
    for i, stem in enumerate(stems):
        img = Image.open(idir / f"{stem}.png").convert("RGB")
        w0, h0 = img.size
        gt = np.array(Image.open(mdir / f"{stem}.png").convert("L")) > 127
        prob = predict(img)
        prob_n = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)
        dep = post_process(prob, THR, MIN_AREA, CLOSE_K)
        dep_n = cv2.resize(dep, (w0, h0), interpolation=cv2.INTER_NEAREST) > 0

        skel_gt = skeletonize(gt)
        if skel_gt.sum() == 0:
            continue
        ys, xs = np.nonzero(skel_gt)
        if len(ys) > MAX_SKEL:
            sel = rng.choice(len(ys), MAX_SKEL, replace=False)
            ys, xs = ys[sel], xs[sel]

        wC = 2.0 * ndi.distance_transform_edt(gt)[ys, xs]
        wA = 2.0 * ndi.distance_transform_edt(dep_n)[ys, xs]
        wA = np.where(dep_n[ys, xs], wA, np.nan)          # vessel absent in the model mask
        ox, oy = orientation(prob_n)
        wB = fwhm_at(prob_n, xs.astype(np.float64), ys.astype(np.float64),
                     ox[ys, xs].astype(np.float64), oy[ys, xs].astype(np.float64))

        cover = float(dep_n[ys, xs].mean())
        m = np.isfinite(wA) & np.isfinite(wC)
        cal_num += float(np.nansum(wA[m] * wC[m]))
        cal_den += float(np.nansum(wA[m] * wA[m]))

        rows.append({
            "camera": cam, "stem": stem, "native_w": w0, "native_h": h0,
            "n_skel_eval": int(len(ys)), "coverage_model_mask": cover,
            "median_A_deployed": float(np.nanmedian(wA)), "median_B_fwhm": float(np.nanmedian(wB)),
            "median_C_expert": float(np.nanmedian(wC)),
            "mean_A_deployed": float(np.nanmean(wA)), "mean_B_fwhm": float(np.nanmean(wB)),
            "mean_C_expert": float(np.nanmean(wC)),
            "expert_vessel_frac": float(gt.mean()),
            "expert_skeleton_px": int(skel_gt.sum()),
        })
        pix_rows.append(pd.DataFrame({
            "camera": cam, "stem": stem, "wA": wA, "wB": wB, "wC": wC,
        }))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(stems)}", flush=True)

pdf = pd.DataFrame(rows)
pix = pd.concat(pix_rows, ignore_index=True)
pdf.to_csv(OUT / "width_per_image.csv", index=False)
pix.to_csv(OUT / "width_per_pixel.csv.gz", index=False, compression="gzip")

K = cal_num / cal_den if cal_den else float("nan")
print(f"\nglobal calibration factor k (best A->C scale): {K:.4f}")

print("\n" + "=" * 100)
print("PER-IMAGE (feature-level) agreement -- median width per image, n=50 per camera")
print("=" * 100)
hdr = f"{'camera':16s} {'est':10s} {'median_px':>10s} {'bias':>8s} {'ratio':>7s} {'r':>7s} {'rho':>7s} {'ICC21':>7s} {'scaled_bias':>12s}"
print(hdr)
summary = []
for cam, g in pdf.groupby("camera"):
    C = g["median_C_expert"].to_numpy()
    for est, col in (("A_deployed", "median_A_deployed"), ("B_fwhm", "median_B_fwhm"),
                     ("C_expert", "median_C_expert")):
        E = g[col].to_numpy()
        m = np.isfinite(E) & np.isfinite(C)
        bias = float(np.nanmean(E[m] - C[m]))
        ratio = float(np.nanmedian(E[m] / C[m]))
        r = float(np.corrcoef(E[m], C[m])[0, 1]) if m.sum() > 2 else float("nan")
        icc = icc21(E, C)
        if est == "C_expert":
            srow = (cam, est, float(np.nanmedian(C)), 0.0, 1.0, 1.0, 1.0, 1.0, 0.0)
        else:
            sc_bias = float(np.nanmean(K * E[m] - C[m]))
            srow = (cam, est, float(np.nanmedian(E[m])), bias, ratio, r, spearman(E, C), icc, sc_bias)
        summary.append(dict(camera=srow[0], estimator=srow[1], median_px=srow[2], bias_px=srow[3],
                            ratio=srow[4], pearson_r=srow[5], spearman_rho=srow[6], icc21=srow[7],
                            scaled_bias_px=srow[8]))
        print(f"{srow[0]:16s} {srow[1]:10s} {srow[2]:10.3f} {srow[3]:8.3f} {srow[4]:7.3f} "
              f"{srow[5]:7.3f} {srow[6]:7.3f} {srow[7]:7.3f} {srow[8]:12.3f}")

print("\n" + "=" * 100)
print("PER-PIXEL agreement on the expert skeleton (pooled per camera)")
print("=" * 100)
print(f"{'camera':16s} {'est':10s} {'n_px':>10s} {'bias':>8s} {'MAE':>8s} {'ratio':>7s} {'rho':>7s} {'LoA_lo':>8s} {'LoA_hi':>8s}")
for cam, g in pix.groupby("camera"):
    C = g["wC"].to_numpy()
    for est, col in (("A_deployed", "wA"), ("B_fwhm", "wB")):
        E = g[col].to_numpy()
        m = np.isfinite(E) & np.isfinite(C)
        d = E[m] - C[m]
        print(f"{cam:16s} {est:10s} {m.sum():10d} {d.mean():8.3f} {np.abs(d).mean():8.3f} "
              f"{np.median(E[m] / C[m]):7.3f} {spearman(E, C):7.3f} "
              f"{d.mean() - 1.96 * d.std():8.3f} {d.mean() + 1.96 * d.std():8.3f}")

sumdf = pd.DataFrame(summary)
sumdf.to_csv(OUT / "width_summary.csv", index=False)
json.dump({
    "inference_size": SIZE, "threshold": THR, "min_area": MIN_AREA, "close_k": CLOSE_K,
    "max_half_px": MAX_HALF, "step_px": STEP, "max_skeleton_px_per_image": MAX_SKEL, "seed": SEED,
    "calibration_factor_k": K,
    "estimators": {
        "C_expert": "2*EDT(expert mask) at expert skeleton -- reference",
        "A_deployed": "2*EDT(post_process(prob,0.20,50,3)) at expert skeleton, NaN where model mask absent",
        "B_fwhm": "FWHM(frac=0.5) of the model probability map, perpendicular to the model orientation",
    },
    "note": ("RetCam GT is clean binary; Neo GT carries JPEG compression noise (256 distinct values). "
             "BV and OD images are NOT paired, so no DD normalisation in this file."),
}, open(OUT / "manifest.json", "w"), indent=2)
print(f"\n[done] -> {OUT}")
