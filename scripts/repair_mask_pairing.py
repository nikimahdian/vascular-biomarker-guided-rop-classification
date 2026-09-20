#!/usr/bin/env python
"""Repair the image/mask prefix-collision bug, without touching any historical artefact.

The bug: infer_masks.find_existing_mask globs f"{stem}_*.png". Because image stems such as
"..._S01_1" are a prefix of the sibling "..._S01_10", the glob also matches the sibling's mask, and
when exactly one file matches it is returned. 610 of 8870 images therefore carry a sibling's mask.

The repair:
  1. rebuild the manifest pairing each image with the mask whose parsed stem is EXACTLY the image
     stem (every affected image has exactly one such file, so no re-segmentation is needed)
  2. recompute the PVBM features for the corrected rows only, as a NEW table with a NEW version
  3. leave biomarker_features.csv, the splits and every historical output untouched
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = "/Users/moniaz/niki"
sys.path.insert(0, ROOT)

from src.biomarker.extract_pvbm import (density_features, geom_features,  # noqa: E402
                                        load_binary_mask, GEOM_COLUMNS, META_COLS)

OUT_MANIFEST = f"{ROOT}/data/masks/mask_manifest_v2.csv"
OUT_FEATURES = f"{ROOT}/data/features/biomarker_features_v4_audit.csv"
REPORT = f"{ROOT}/results/audit_mask_repair.json"
PAT = re.compile(r"^(?P<stem>.+)_(?P<h>[0-9a-f]{8})\.png$")

feats = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv")
masks_dir = os.path.dirname(feats.mask_path.iloc[0])

by_stem: dict[str, list[str]] = defaultdict(list)
for name in os.listdir(masks_dir):
    m = PAT.match(name)
    if m:
        by_stem[m.group("stem")].append(name)

fixed_path, changed = [], []
for _, r in feats.iterrows():
    stem = os.path.splitext(os.path.basename(r.image_path))[0]
    cur = os.path.basename(r.mask_path)
    parsed = PAT.match(cur)
    if parsed and parsed.group("stem") == stem:
        fixed_path.append(r.mask_path)
        continue
    cands = by_stem.get(stem, [])
    if len(cands) == 1:
        fixed_path.append(os.path.join(masks_dir, cands[0]))
        changed.append({"image_path": r.image_path, "old_mask": r.mask_path,
                        "new_mask": os.path.join(masks_dir, cands[0])})
    else:
        fixed_path.append(r.mask_path)
        print(f"  [warn] {stem}: {len(cands)} candidates, left unchanged")

feats["mask_path_v2"] = fixed_path
print(f"repairable rows: {len(changed)}  (expected 610)")
assert len(changed) == 610, f"expected 610 repairs, got {len(changed)}"

pd.DataFrame(changed).to_csv(f"{ROOT}/results/audit_mask_repair_pairs.csv", index=False)

man = pd.read_csv(f"{ROOT}/data/masks/mask_manifest.csv")
fix = {c["image_path"]: c["new_mask"] for c in changed}
man["mask_path"] = [fix.get(p, m) for p, m in zip(man.image_path, man.mask_path)]
man.to_csv(OUT_MANIFEST, index=False)
print(f"corrected manifest -> {OUT_MANIFEST}  rows={len(man)}")
print(f"  duplicate mask paths now: {len(man) - man.mask_path.nunique()}")

# ---- recompute features for the corrected rows only, into a NEW table -------------
out = feats.drop(columns=["mask_path_v2"]).copy()
patched = out.set_index("image_path")
todo = [c for c in changed]
print(f"\nrecomputing PVBM features for {len(todo)} corrected rows ...", flush=True)
t0 = time.time()
for k, c in enumerate(todo, 1):
    mask = load_binary_mask(c["new_mask"])
    row = {"mask_path": c["new_mask"]}
    row.update(density_features(mask))
    row.update(geom_features(mask, "whole"))
    for col, val in row.items():
        patched.loc[c["image_path"], col] = val
    if k % 100 == 0:
        print(f"  {k}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)

out = patched.reset_index()
out["feature_version"] = "v4_audit_maskfix"
out.to_csv(OUT_FEATURES, index=False)
print(f"\ncustom feature table -> {OUT_FEATURES}  shape={out.shape}")

# ---- how much did the features move for the affected rows? ------------------------
cmp_rows = []
for c in changed:
    old = feats[feats.image_path == c["image_path"]].iloc[0]
    new = out[out.image_path == c["image_path"]].iloc[0]
    for col in GEOM_COLUMNS + ["vessel_density", "vessel_pixels"]:
        a, b = old.get(col, np.nan), new.get(col, np.nan)
        if np.isfinite(a) and np.isfinite(b) and a != 0:
            cmp_rows.append({"feature": col, "rel_change": abs(b - a) / abs(a)})
cd = pd.DataFrame(cmp_rows)
summary = (cd.groupby("feature").rel_change
           .agg(["median", "mean", "max"]).sort_values("median", ascending=False).round(4))
print("\n=== relative change in the corrected rows (old vs new mask) ===")
print(summary.to_string())

json.dump({
    "repaired_rows": len(changed),
    "feature_table": OUT_FEATURES,
    "manifest": OUT_MANIFEST,
    "by_source": feats[feats.image_path.isin([c["image_path"] for c in changed])]
                 .source.value_counts().to_dict(),
    "by_split": feats[feats.image_path.isin([c["image_path"] for c in changed])]
                .split.value_counts().to_dict(),
    "feature_change": summary.reset_index().to_dict("records"),
    "historical_artifacts_untouched": [
        "data/features/biomarker_features.csv", "data/splits/*",
        "results/branch_a_results.json", "results/branch_c_results.json",
        "results/loo*", "results/hvdro_validation/*",
    ],
}, open(REPORT, "w"), indent=2, default=str)
print(f"\n[done] -> {REPORT}")
