#!/usr/bin/env python
"""Blok 1 re-run with the two fixes diagnosed from the first pass.

Pass 1 failed the gate (RetCam per-image ICC 0.576, Neo -0.157) for four reasons. This script
addresses three of them:

  (1) No anatomical region -- the first pass took the median over the WHOLE vessel tree, which is
      dominated by how much periphery the camera frames (corr(skeleton_px, median_width) = -0.42
      on Neo). Width is now measured inside a peripapillary annulus in disc diameters, using the
      disc predicted by the HVDROPDB-trained ensemble.
  (2) Integer quantisation on 640x480 RetCam. The reference is now the FWHM of the Gaussian-
      blurred expert mask, which is sub-pixel: for a blurred binary strip the 0.5 level set sits
      on the true boundary, so FWHM recovers width continuously instead of in 2-px steps.
  (3) The FWHM sampler anchored its crossing search at the skeleton point, so whenever the model
      centreline was offset the profile was already below threshold and the width collapsed to 0
      (that is the median_B = 0.00 seen on Neo). The search is now anchored at the profile peak.

Estimators, all evaluated on the expert skeleton inside the annulus:
  C_edt   2*EDT(expert mask)                        integer, pass-1 style
  C_fwhm  FWHM of blurred expert mask, absolute 0.5 sub-pixel reference
  A_edt   2*EDT(deployed model mask)
  A_fwhm  FWHM of blurred deployed model mask, absolute 0.5
  B_fwhm  FWHM of the model probability map, relative 0.5*peak
Widths are reported both in native pixels and in disc diameters.
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
DISC = ROOT / "results/hvdro_validation/disc"
OUT = ROOT / "results/hvdro_validation/width2"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 256
MAX_HALF = 14.0
STEP = 0.25
BLUR_SIGMA = 1.0
MAX_SHIFT = 3.0
MAX_SKEL = 4000
ANN = (0.5, 2.0)
SEED = 20260101

cfg = load_config()
sc = cfg["segmentation"]
THR, MIN_AREA, CLOSE_K = float(sc["threshold"]), int(sc["min_area"]), int(sc["morph_close_kernel"])
device = get_device()
print(f"device={device} thr={THR} size={SIZE} annulus={ANN} DD")

model = build_model(sc["arch"], sc["encoder"], None, int(sc["in_channels"]), 1).to(device).eval()
model.load_state_dict(torch.load(str(ROOT / sc["weight_path"]), map_location=device))
tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])

disc = pd.read_csv(DISC / "bv_disc_predictions.csv").set_index("stem")
print("disc predictions:", disc.shape)


def predict(img):
    with torch.no_grad():
        return torch.sigmoid(model(tf(img).unsqueeze(0).to(device))).squeeze().cpu().numpy()


def bilinear(img, x, y):
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = x - x0, y - y0
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
            + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def orientation(prob, sigma=2.0):
    p = cv2.GaussianBlur(prob.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    th = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    return np.cos(th), np.sin(th)


def fwhm(mp, nx, ny, ox, oy, mode="relative"):
    """Width across (ox,oy) at (nx,ny). mode='relative' -> 0.5*peak, 'absolute' -> 0.5."""
    t = np.arange(-MAX_HALF, MAX_HALF + 1e-9, STEP)
    prof = np.empty((len(nx), len(t)), dtype=np.float32)
    for i, dt in enumerate(t):
        prof[:, i] = bilinear(mp, nx + dt * ox, ny + dt * oy)
    c = int(np.argmin(np.abs(t)))
    if mode == "relative":
        thr = 0.5 * prof.max(axis=1)
        ok_peak = prof.max(axis=1) >= 0.25
    else:
        thr = np.full(len(nx), 0.5, dtype=np.float32)
        ok_peak = prof.max(axis=1) >= 0.5
    ctr = prof.argmax(axis=1)                      # anchor at the peak, not at the skeleton point
    ar = np.arange(len(t))[None, :]
    lm = (prof < thr[:, None]) & (ar <= ctr[:, None])
    rm = (prof < thr[:, None]) & (ar >= ctr[:, None])
    has_l, has_r = lm.any(axis=1), rm.any(axis=1)
    li = np.where(has_l, ctr - np.argmax(lm[:, ::-1], axis=1), ctr)
    ri = np.where(has_r, ctr + np.argmax(rm, axis=1), ctr)
    w = (ri - li) * STEP
    shift = np.abs(ctr - c) * STEP
    bad = (~has_l) | (~has_r) | (~ok_peak) | (shift > MAX_SHIFT)
    w = w.astype(np.float64)
    w[bad] = np.nan
    return w


def icc21(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return np.nan
    d = np.stack([x, y], 1).astype(float)
    k, gm = 2, d.mean()
    msr = k * ((d.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((d.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den else np.nan


def spear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(pd.Series(a[m]).rank(), pd.Series(b[m]).rank())[0, 1])


rng = np.random.default_rng(SEED)
rows, pix = [], []
for cam in ("RetCam_Vessels", "Neo_Vessels"):
    idir, mdir = SEG / "HVDROPDB-BV" / f"{cam}_images", SEG / "HVDROPDB-BV" / f"{cam}_masks"
    stems = sorted(p.stem for p in idir.glob("*.png"))
    print(f"\n=== {cam} ({len(stems)}) ===", flush=True)
    for i, stem in enumerate(stems):
        img = Image.open(idir / f"{stem}.png").convert("RGB")
        w0, h0 = img.size
        gt = np.array(Image.open(mdir / f"{stem}.png").convert("L")) > 127
        if stem not in disc.index:
            print("  no disc prediction for", stem)
            continue
        d = disc.loc[stem]
        dd = float(d["pred_dd_px"])
        xc, yc = float(d["pred_cx"]), float(d["pred_cy"])
        if not (np.isfinite(dd) and dd > 4):
            print("  bad disc for", stem)
            continue

        prob = predict(img)
        prob_n = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)
        dep = cv2.resize(post_process(prob, THR, MIN_AREA, CLOSE_K), (w0, h0),
                         interpolation=cv2.INTER_NEAREST) > 0
        gtb = cv2.GaussianBlur(gt.astype(np.float32), (0, 0), BLUR_SIGMA)
        depb = cv2.GaussianBlur(dep.astype(np.float32), (0, 0), BLUR_SIGMA)

        skel = skeletonize(gt)
        ys, xs = np.nonzero(skel)
        if len(ys) == 0:
            continue
        yy, xx = np.mgrid[0:h0, 0:w0]
        rdd = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2) / dd
        ann = (rdd >= ANN[0]) & (rdd < ANN[1])
        inside = ann[ys, xs]
        if inside.sum() < 50:
            print(f"  {stem}: only {inside.sum()} skeleton px in annulus")
            continue
        ys, xs = ys[inside], xs[inside]
        if len(ys) > MAX_SKEL:
            s = rng.choice(len(ys), MAX_SKEL, replace=False)
            ys, xs = ys[s], xs[s]

        edt_gt = 2.0 * ndi.distance_transform_edt(gt)[ys, xs]
        edt_dep = 2.0 * ndi.distance_transform_edt(dep)[ys, xs]
        edt_dep = np.where(dep[ys, xs], edt_dep, np.nan)
        ogx, ogy = orientation(gtb)
        odx, ody = orientation(depb)
        opx, opy = orientation(prob_n)
        wC = fwhm(gtb, xs.astype(float), ys.astype(float), ogx[ys, xs], ogy[ys, xs], "absolute")
        wA = fwhm(depb, xs.astype(float), ys.astype(float), odx[ys, xs], ody[ys, xs], "absolute")
        wB = fwhm(prob_n, xs.astype(float), ys.astype(float), opx[ys, xs], opy[ys, xs], "relative")

        rows.append({
            "camera": cam, "stem": stem, "dd_px": dd, "n_skel_ann": int(len(ys)),
            "median_C_edt_px": float(np.median(edt_gt)), "median_C_fwhm_px": float(np.nanmedian(wC)),
            "median_A_edt_px": float(np.nanmedian(edt_dep)), "median_A_fwhm_px": float(np.nanmedian(wA)),
            "median_B_fwhm_px": float(np.nanmedian(wB)),
            "median_C_fwhm_dd": float(np.nanmedian(wC) / dd),
            "median_A_fwhm_dd": float(np.nanmedian(wA) / dd),
            "median_B_fwhm_dd": float(np.nanmedian(wB) / dd),
        })
        pix.append(pd.DataFrame({"camera": cam, "stem": stem, "wA": wA, "wB": wB, "wC": wC}))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(stems)}", flush=True)

pdf = pd.DataFrame(rows)
px = pd.concat(pix, ignore_index=True)
pdf.to_csv(OUT / "width2_per_image.csv", index=False)
px.to_csv(OUT / "width2_per_pixel.csv.gz", index=False, compression="gzip")

print("\n" + "=" * 104)
print("PER-IMAGE agreement inside the 0.5-2.0 DD annulus (n=50 per camera)")
print("=" * 104)
print(f"{'camera':10s} {'quantity':22s} {'median':>9s} {'bias':>8s} {'ratio':>7s} {'r':>7s} {'rho':>7s} {'ICC21':>7s}")
summ = []
pairs = [("A_edt", "median_A_edt_px", "median_C_edt_px"), ("C_edt", "median_C_edt_px", "median_C_edt_px"),
         ("A_fwhm", "median_A_fwhm_px", "median_C_fwhm_px"), ("B_fwhm", "median_B_fwhm_px", "median_C_fwhm_px"),
         ("C_fwhm", "median_C_fwhm_px", "median_C_fwhm_px"),
         ("A_fwhm_dd", "median_A_fwhm_dd", "median_C_fwhm_dd"),
         ("B_fwhm_dd", "median_B_fwhm_dd", "median_C_fwhm_dd")]
for cam, g in pdf.groupby("camera"):
    for name, ec, cc in pairs:
        E, C = g[ec].to_numpy(), g[cc].to_numpy()
        m = np.isfinite(E) & np.isfinite(C)
        bias = float(np.mean(E[m] - C[m]))
        ratio = float(np.median(E[m] / C[m]))
        r = float(np.corrcoef(E[m], C[m])[0, 1]) if m.sum() > 2 else np.nan
        icc = icc21(E, C)
        print(f"{cam:10s} {name:22s} {np.median(E[m]):9.3f} {bias:8.3f} {ratio:7.3f} {r:7.3f} "
              f"{spear(E, C):7.3f} {icc:7.3f}")
        summ.append(dict(camera=cam, quantity=name, median=float(np.median(E[m])), bias_px=bias,
                         ratio=ratio, pearson_r=r, spearman=spear(E, C), icc21=icc))
pd.DataFrame(summ).to_csv(OUT / "width2_summary.csv", index=False)

print("\n" + "=" * 104)
print("PER-PIXEL agreement inside the annulus, widths in disc diameters")
print("=" * 104)
for cam, g in px.groupby("camera"):
    ddp = pdf[pdf.camera == cam].set_index("stem")["dd_px"]
    dd = g["stem"].map(ddp).to_numpy()
    for name, col in (("A_fwhm", "wA"), ("B_fwhm", "wB")):
        e = g[col].to_numpy() / dd
        c = g["wC"].to_numpy() / dd
        m = np.isfinite(e) & np.isfinite(c)
        d = e[m] - c[m]
        print(f"{cam:10s} {name:8s} n={m.sum():7d} bias={d.mean():+.4f} MAE={np.abs(d).mean():.4f} "
              f"ratio={np.median(e[m] / c[m]):.3f} rho={spear(e, c):+.3f} "
              f"LoA=[{d.mean() - 1.96 * d.std():+.4f},{d.mean() + 1.96 * d.std():+.4f}]")

json.dump({"inference_size": SIZE, "annulus_dd": list(ANN), "blur_sigma": BLUR_SIGMA,
           "max_shift_px": MAX_SHIFT, "max_skeleton_px": MAX_SKEL, "seed": SEED,
           "disc_source": "HVDROPDB-trained Unet resnet34 5-fold ensemble",
           "estimators": {"C_edt": "2*EDT(expert), integer", "C_fwhm": "FWHM blurred expert, abs 0.5",
                          "A_fwhm": "FWHM blurred deployed mask, abs 0.5",
                          "B_fwhm": "FWHM model probability, rel 0.5*peak"}},
          open(OUT / "manifest.json", "w"), indent=2)
print(f"\n[done] -> {OUT}")
