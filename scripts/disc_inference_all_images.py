#!/usr/bin/env python
"""Run the 5-fold HVDROPDB disc ensemble over every image in biomarker_features.csv.

Records the PEAK PROBABILITY as well as the geometry, because on Farabi the detector is bimodal:
53% of images fire confidently (peak>0.9) and 28% do not fire at all (peak<0.3) -- a genuine
domain-transfer failure, verified visually (the disc is present in the failures).
Downstream the peak probability becomes a measurability flag rather than a silent drop.

Writes results/hvdro_validation/disc/disc_predictions_all.csv. Batched, resumable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, "/Users/moniaz/niki")
from src.segmentation.models import build_model  # noqa: E402
from src.utils.common import get_device  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
DISC = ROOT / "results/hvdro_validation/disc"
OUT = DISC / "disc_predictions_all.csv"
PART = DISC / "disc_predictions_all.partial.csv"
SIZE, NFOLD, BATCH = 384, 5, 8
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

device = get_device()
print("device:", device)

meta = pd.read_csv(ROOT / "data/features/biomarker_features.csv",
                   usecols=["image_path", "source", "label"])
print("rows in feature table:", len(meta))
done = set()
if PART.exists():
    done = set(pd.read_csv(PART, usecols=["image_path"])["image_path"])
    print("resume, already done:", len(done))
todo = meta[~meta["image_path"].isin(done)].reset_index(drop=True)
print("todo:", len(todo))
if not len(todo):
    pd.read_csv(PART).to_csv(OUT, index=False)
    PART.unlink(missing_ok=True)
    print("nothing to do ->", OUT)
    sys.exit(0)

models = []
for k in range(NFOLD):
    m = build_model("Unet", "resnet34", None, 3, 1).to(device)
    m.load_state_dict(torch.load(str(DISC / f"disc_unet_fold{k}.pt"), map_location=device))
    models.append(m.eval())
print("ensemble loaded:", len(models))


def biggest(m):
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    return lab == (int(np.argmax(ndi.sum(m, lab, index=np.arange(1, n + 1)))) + 1)


rows, t0, buf = [], time.time(), []
for i in range(0, len(todo), BATCH):
    chunk = todo.iloc[i:i + BATCH]
    tens, sizes, keep = [], [], []
    for _, r in chunk.iterrows():
        p = r["image_path"]
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            buf.append(dict(image_path=p, source=r["source"], label=r["label"], ok=False,
                            err=type(e).__name__))
            continue
        sizes.append(img.size)
        keep.append(r)
        x = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        tens.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
    if not tens:
        continue
    xb = torch.stack(tens).to(device)
    with torch.no_grad():
        p384 = np.mean([torch.sigmoid(m(xb)).cpu().numpy()[:, 0] for m in models], axis=0)
    for j, (w0, h0) in enumerate(sizes):
        r = keep[j]
        pk = float(p384[j].max())
        row = dict(image_path=r["image_path"], source=r["source"], label=r["label"], w=w0, h=h0,
                   peak_prob=pk, area_at_50=float((p384[j] > 0.5).mean()),
                   area_at_30=float((p384[j] > 0.3).mean()),
                   argmax_x=float(np.unravel_index(p384[j].argmax(), p384[j].shape)[1] / SIZE),
                   argmax_y=float(np.unravel_index(p384[j].argmax(), p384[j].shape)[0] / SIZE))
        pr = np.asarray(Image.fromarray((p384[j] * 255).astype(np.uint8)).resize((w0, h0),
                                                                                 Image.BILINEAR),
                        dtype=np.float32) / 255.0
        m = biggest(pr > 0.5)
        if m.sum() == 0:
            row.update(ok=False, err="empty", disc_area_frac=0.0, disc_radius_px=np.nan,
                       disc_dd_px=np.nan, disc_cx=np.nan, disc_cy=np.nan,
                       dd_over_min_side=np.nan, centre_offset_px=np.nan)
        else:
            ys, xs = np.nonzero(m)
            rad = float(np.sqrt(m.sum() / np.pi))
            row.update(ok=True, err="", disc_area_frac=float(m.mean()), disc_radius_px=rad,
                       disc_dd_px=2 * rad, disc_cx=float(xs.mean()), disc_cy=float(ys.mean()),
                       dd_over_min_side=float(2 * rad / min(h0, w0)),
                       centre_offset_px=float(np.hypot(xs.mean() - w0 / 2, ys.mean() - h0 / 2)))
        buf.append(row)
    if len(buf) >= 200:
        pd.DataFrame(buf).to_csv(PART, mode="a", header=not PART.exists(), index=False)
        buf = []
        el = time.time() - t0
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)}  {el:.0f}s  "
              f"eta {el / max(i + BATCH, 1) * (len(todo) - i - BATCH) / 60:.1f}min", flush=True)
if buf:
    pd.DataFrame(buf).to_csv(PART, mode="a", header=not PART.exists(), index=False)

df = pd.read_csv(PART)
df.to_csv(OUT, index=False)
PART.unlink(missing_ok=True)
print(f"[done] {OUT} n={len(df)}  total {(time.time() - t0) / 60:.1f}min")
print("\ncoverage by source (peak_prob):")
g = df.groupby("source")["peak_prob"]
print(g.agg(n="count", median="median",
            frac_gt_0p9=lambda s: float((s > 0.9).mean()),
            frac_lt_0p5=lambda s: float((s < 0.5).mean()),
            frac_lt_0p3=lambda s: float((s < 0.3).mean())).round(4).to_string())
print("\nDD/min_side where ok, by source:")
print(df[df.ok].groupby("source")["dd_over_min_side"].describe().round(4).to_string())
