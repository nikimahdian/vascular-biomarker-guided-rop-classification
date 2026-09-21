#!/usr/bin/env python
"""Task 5B-H3 section L: deterministic visual QC for the locked evaluation.

Reads `_private_audit/task5b_h3_rows.csv` + `task5b_h3_baseline.csv` (written by
`scripts/task5b_h3_eval.py`) and renders private montages for:

  * the strongest V1 border failures;
  * the strongest V2 residual failures;
  * the largest native V1 vs V2 disagreements;
  * one median case per source and per geometry.

Each montage shows the baseline image with BOTH FOV contours (V1 green, V2 cyan), the perturbed
image with both FOV contours (V1 red, V2 orange), the fixed SEG_CURRENT_V1 vessel mask, and the
baseline/perturbed values of all five FINAL_PRIMARY features for both versions.

No disease label is read or drawn. No measurement path is modified; the montages call the frozen
production `retinal_fov` and the predeclared candidate's `retinal_fov_v2` only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.task5b_h_fov_stress import apply_border, apply_brightness, pad_specs  # noqa: E402
from scripts.task5b_h3_eval import novel_specs  # noqa: E402
from scripts.task5b_n2 import load  # noqa: E402
from src.biomarker import clinical_measurement_v2_candidate as v2  # noqa: E402
from src.biomarker.clinical_measurement_v1 import retinal_fov  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]


def contour(m):
    return m & ~ndi.binary_erosion(m, np.ones((3, 3), bool))


def panel(rgb, masks):
    img = np.clip(rgb, 0, 255).astype(np.uint8).copy()
    for m, col in masks:
        if m is None:
            continue
        img[contour(m)] = col
    return img


def to_png(arr, h):
    im = Image.fromarray(arr)
    w = max(1, int(round(im.width * h / im.height)))
    return im.resize((w, h), Image.BILINEAR)


def render(image_path, mask_path, cond, rows, base, path):
    rgb, msk = load(image_path, mask_path)
    h0, w0 = msk.shape
    f1b, _ = retinal_fov(rgb[:, :, 1])
    _o2b, f2b, _q2b = v2.measure_v2(rgb, msk, {})
    if cond == "NATIVE":
        prgb, pmsk, pad = rgb, msk, (0, 0, 0, 0)
    elif cond.startswith("x"):
        prgb, pmsk, pad = apply_brightness(rgb, msk, float(cond[1:]))
    else:
        spec = [s for s in (pad_specs(h0, w0) + novel_specs(h0, w0)) if s[1] == cond]
        if not spec:
            return False
        prgb, pmsk, pad = apply_border(rgb, msk, spec[0])
    t, b, l, r = pad
    f1p, _ = retinal_fov(prgb[:, :, 1])
    _o2p, f2p, _q2p = v2.measure_v2(prgb, pmsk, {})
    c1, c2 = f1p[t:t + h0, l:l + w0], f2p[t:t + h0, l:l + w0]

    height = 420
    a = to_png(panel(rgb, [(f1b, (0, 255, 0)), (f2b, (0, 255, 255))]), height)
    bb = to_png(panel(prgb, [(f1p, (255, 0, 0)), (f2p, (255, 165, 0))]), height)
    mm = Image.fromarray(
        (msk.astype(bool)[:, :, None] * np.array([255, 255, 255], np.uint8)).astype(np.uint8)
    ).resize((height, height), Image.NEAREST)
    canvas = Image.new("RGB", (a.width + bb.width + mm.width + 8, height + 62), (0, 0, 0))
    canvas.paste(a, (0, 0))
    canvas.paste(bb, (a.width + 4, 0))
    canvas.paste(mm, (a.width + bb.width + 8, 0))
    dr = ImageDraw.Draw(canvas)
    dr.text((4, height + 2), f"{image_path.split('/')[-1][:52]}  |  {cond}", fill=(255, 255, 0))
    dr.text((4, height + 14), "left: baseline, V1 green / V2 cyan    mid: perturbed, V1 red / V2 orange"
                             "    right: fixed SEG_CURRENT_V1 mask", fill=(200, 200, 200))
    y = height + 26
    for i, f in enumerate(FEATS):
        bv1, bv2 = base.get(f"v1_{f}", np.nan), base.get(f"v2_{f}", np.nan)
        pv1, pv2 = rows.get(f"v1_{f}", np.nan), rows.get(f"v2_{f}", np.nan)
        dr.text((4, y + (i % 3) * 12),
                f"{f:20s} V1 {bv1:.5f} -> {pv1:.5f}   V2 {bv2:.5f} -> {pv2:.5f}",
                fill=(255, 255, 255))
    canvas.save(path)
    return True


def main() -> None:
    C = pd.read_csv(OUT / "task5b_h3_rows.csv")
    B = pd.read_csv(OUT / "task5b_h3_baseline.csv")
    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                       usecols=["image_path", "mask_path"])
    mpath = dict(zip(meta.image_path, meta.mask_path))
    Bm = B.set_index("image_path")
    cn = C[C.family == "border"]

    picks = []
    for r in cn.sort_values("v1_dice").head(3).itertuples():
        picks.append((r.image_path, r.condition, "worstV1"))
    res = cn.assign(v2err=cn["v2_vessel_density_fov_rel"].abs()).sort_values("v2err", ascending=False)
    for r in res.head(3).itertuples():
        picks.append((r.image_path, r.condition, "worstV2"))
    for r in B.sort_values("drift_fov_dice").head(3).itertuples():
        picks.append((r.image_path, "NATIVE", "nativeDrift"))
    for key in ("source", "geom"):
        for k, g in cn.groupby(key):
            med = g.v1_dice.median()
            r = g.iloc[(g.v1_dice - med).abs().argsort()[:1]].iloc[0]
            picks.append((r.image_path, r.condition, f"{key}_{k}"))

    seen, uniq = set(), []
    for p in picks:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            uniq.append(p)
    n = 0
    for image_path, cond, tag in uniq:
        if cond == "NATIVE":
            rows = {}
        else:
            m = C[(C.image_path == image_path) & (C.condition == cond)]
            if not len(m):
                continue
            rows = m.iloc[0].to_dict()
        base = Bm.loc[image_path].to_dict()
        name = f"task5b_h3_qc_{tag}__{cond}__{image_path.split('/')[-1][:12]}.png".replace("/", "_")
        try:
            if render(image_path, mpath[image_path], cond, rows, base, OUT / name):
                n += 1
                print(f"  {name}")
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {name}: {type(e).__name__}: {e}")
    pd.DataFrame(uniq, columns=["image_path", "condition", "tag"]).to_csv(
        OUT / "task5b_h3_qc_cases.csv", index=False)
    print(f"  montages written: {n}")
    print(f"  case list: _private_audit/task5b_h3_qc_cases.csv")


if __name__ == "__main__":
    main()
