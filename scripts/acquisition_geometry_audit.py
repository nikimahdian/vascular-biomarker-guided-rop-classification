#!/usr/bin/env python
"""Independent verification: is Plus prevalence really geometry-dependent and split-inverted?

Reads the ORIGINAL feature table and the mask manifest (not any derived table), so the result does
not depend on my renormalisation work. Also counts how many patient groups contribute to each
geometry x split cell, because with few groups the inversion could be a small-sample accident.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROOT = "/Users/moniaz/niki"
f = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv")
print("feature table:", f.shape, "columns:", [c for c in f.columns][:14])

man = pd.read_csv(f"{ROOT}/data/masks/mask_manifest.csv")
print("mask manifest:", man.shape, list(man.columns)[:14])

# find width/height columns in the manifest
wc = next((c for c in man.columns if c.lower() in ("w", "width")), None)
hc = next((c for c in man.columns if c.lower() in ("h", "height")), None)
print("manifest size cols:", wc, hc)
if wc is None or hc is None:
    print("no size columns; falling back to disc_predictions_all.csv")
    d = pd.read_csv(f"{ROOT}/results/hvdro_validation/disc/disc_predictions_all.csv")
else:
    key = next((c for c in man.columns if "mask" in c.lower() and "path" in c.lower()), None)
    print("join key:", key)
    d = man[[key, wc, hc]].rename(columns={key: "mask_path", wc: "w", hc: "h"})

m = f.merge(d[["mask_path", "w", "h"]] if "mask_path" in d.columns else d,
            on="mask_path", how="left") if "mask_path" in d.columns else None
if m is None:
    # no usable manifest join; use image_path against the disc inference file
    dd = pd.read_csv(f"{ROOT}/results/hvdro_validation/disc/disc_predictions_all.csv")
    m = f.merge(dd[["image_path", "w", "h"]], on="image_path", how="left")
m["min_side"] = np.minimum(m.w, m.h)
m["y"] = (m.label == 2).astype(int)
print("\njoined n:", len(m), " missing size:", int(m.min_side.isna().sum()))

print("\n" + "=" * 100)
print("Plus counts and prevalence by geometry x split  (ORIGINAL table)")
print("=" * 100)
t = m.pivot_table(index="min_side", columns="split", values="y", aggfunc=["sum", "size", "mean"])
print(t.round(4).to_string())

print("\n" + "=" * 100)
print("how many distinct groups per geometry x split?")
print("=" * 100)
g = m.pivot_table(index="min_side", columns="split", values="group_id", aggfunc="nunique")
print(g.to_string())

print("\n" + "=" * 100)
print("groups that contain Plus, by geometry x split")
print("=" * 100)
gp = m[m.y == 1].pivot_table(index="min_side", columns="split", values="group_id",
                             aggfunc="nunique").fillna(0).astype(int)
print(gp.to_string())

print("\n" + "=" * 100)
print("source x geometry cross-tab (images)")
print("=" * 100)
print(pd.crosstab(m.source, m.min_side).to_string())

print("\n" + "=" * 100)
print("label x geometry, whole dataset")
print("=" * 100)
ct = pd.crosstab(m.min_side, m.label)
ct["n"] = ct.sum(1)
ct["plus_frac"] = (ct.get(2, 0) / ct["n"]).round(4)
print(ct.to_string())
