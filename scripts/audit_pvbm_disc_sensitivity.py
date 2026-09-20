#!/usr/bin/env python
"""Section J: how much do the PVBM features actually depend on the optic-disc parameters?

In production the disc is fabricated: centre = image centre, radius = max(8, min(h,w)//8). Those
two numbers are passed into PVBM's compute_geomVBMs, where they control the starting-point
selection ("distance from centre < 100 + radius") and therefore the whole graph traversal.

This perturbs the centre and the radius and measures what moves.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from skimage.morphology import skeletonize

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/moniaz/niki")
from PVBM.GeometryAnalysis import GeometricalVBMs  # noqa: E402

ROOT = "/Users/moniaz/niki"
GEOM = ["area", "tortuosity_index", "median_tortuosity", "overall_length",
        "median_branching_angle", "n_startpoints", "n_endpoints", "n_intersections"]

feats = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv",
                    usecols=["image_path", "mask_path", "source"])
disc = pd.read_csv(f"{ROOT}/results/hvdro_validation/disc/disc_predictions_all.csv")
disc = disc.set_index("image_path")
feats["min_side"] = [min(Image.open(p).size[::-1]) if __import__("os").path.exists(p) else np.nan
                     for p in feats.image_path]
feats = feats.dropna(subset=["min_side"])
sample = (feats.groupby("min_side", group_keys=False)
          .apply(lambda g: g.sample(min(4, len(g)), random_state=0))
          .reset_index(drop=True))
print(f"sampled {len(sample)} masks across {sorted(sample.min_side.unique())}")


def run(mask, xc, yc, radius):
    seg = mask.astype(float)
    skel = skeletonize(seg > 0).astype(int)
    geom = GeometricalVBMs()
    prev = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    try:
        vbms, _ = geom.compute_geomVBMs(blood_vessel=seg, skeleton=skel,
                                        xc=int(xc), yc=int(yc), radius=int(radius))
        return dict(zip(GEOM, [float(v) for v in vbms]))
    except Exception as e:  # noqa: BLE001
        return {k: np.nan for k in GEOM} | {"_err": type(e).__name__}
    finally:
        sys.setrecursionlimit(prev)


rows = []
t0 = time.time()
for _, r in sample.iterrows():
    try:
        m = (np.array(Image.open(r.mask_path).convert("L")) > 127).astype(np.uint8)
    except Exception:  # noqa: BLE001
        continue
    h, w = m.shape
    base_radius = max(8, min(h, w) // 8)
    conditions = {
        "production_centre_fabricated_radius": (w // 2, h // 2, base_radius),
        "centre_shift_+10%_x": (int(w * 0.6), h // 2, base_radius),
        "centre_shift_-10%_x": (int(w * 0.4), h // 2, base_radius),
        "centre_shift_+10%_y": (w // 2, int(h * 0.6), base_radius),
        "centre_shift_-25%_y": (w // 2, int(h * 0.25), base_radius),
        "radius_half": (w // 2, h // 2, max(8, base_radius // 2)),
        "radius_double": (w // 2, h // 2, base_radius * 2),
        "radius_very_large": (w // 2, h // 2, max(h, w)),
    }
    if r.image_path in disc.index:
        d = disc.loc[r.image_path]
        if np.isfinite(d.get("disc_dd_px", np.nan)) and d["disc_dd_px"] > 2:
            conditions["measured_disc_centre_and_radius"] = (
                int(d["disc_cx"]), int(d["disc_cy"]), int(d["disc_dd_px"] / 2))

    got = {}
    for name, (xc, yc, rad) in conditions.items():
        got[name] = run(m, xc, yc, rad)
    base = got["production_centre_fabricated_radius"]
    for name, v in got.items():
        rec = {"image_path": r.image_path, "source": r.source, "min_side": int(r.min_side),
               "condition": name}
        for k in GEOM:
            rec[k] = v.get(k, np.nan)
            b = base.get(k, np.nan)
            rec[f"rel_{k}"] = (abs(v.get(k, np.nan) - b) / abs(b)
                               if np.isfinite(b) and b != 0 and np.isfinite(v.get(k, np.nan))
                               else np.nan)
        rows.append(rec)
    print(f"  {r.min_side:>5.0f}  {r.source:10s}  done  ({time.time() - t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{ROOT}/results/audit_pvbm_disc_sensitivity.csv", index=False)
print(f"\nrows: {len(df)}  masks: {df.image_path.nunique()}")

print("\n" + "=" * 108)
print("Relative change in each PVBM feature when the disc parameters change")
print("(median over masks; the baseline is the production fabricated disc)")
print("=" * 108)
for cond in df.condition.unique():
    if cond == "production_centre_fabricated_radius":
        continue
    sub = df[df.condition == cond]
    line = f"  {cond:38s}"
    for k in GEOM:
        v = sub[f"rel_{k}"].median()
        line += f" {k[:9]:>10s}={v:7.3f}" if np.isfinite(v) else f" {k[:9]:>10s}={'-':>7s}"
    print(line)

print("\n  features whose median relative change exceeds 5% under any perturbation:")
worst = {}
for k in GEOM:
    worst[k] = df[df.condition != "production_centre_fabricated_radius"][f"rel_{k}"].max()
for k, v in sorted(worst.items(), key=lambda kv: -kv[1]):
    flag = "  <-- SENSITIVE" if v > 0.05 else ""
    print(f"    {k:26s} max rel change = {v:8.4f}{flag}")
