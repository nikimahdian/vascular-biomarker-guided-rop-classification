#!/usr/bin/env python
"""Quantify the image/mask prefix-collision bug exactly.

A correct mask filename is "<image_stem>_<8 hex>.png". The buggy fallback in
infer_masks.find_existing_mask globs "<image_stem>_*.png", which also matches a sibling image whose
filename is the image stem plus a digit, e.g. stem "..._S01_1" matches "..._S01_10_<hash>.png".
"""
from __future__ import annotations

import os
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"

f = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv",
                usecols=["image_path", "mask_path", "split", "source", "label", "group_id"])
pat = re.compile(r"^(?P<stem>.+)_(?P<h>[0-9a-f]{8})\.png$")

rows = []
for _, r in f.iterrows():
    img_stem = os.path.splitext(os.path.basename(r.image_path))[0]
    m = pat.match(os.path.basename(r.mask_path))
    mask_stem = m.group("stem") if m else None
    rows.append({"image_stem": img_stem, "mask_stem": mask_stem,
                 "ok": mask_stem == img_stem})
d = pd.DataFrame(rows)
f = pd.concat([f, d], axis=1)

print(f"rows                          : {len(f)}")
print(f"  mask filename unparseable   : {int(f.mask_stem.isna().sum())}")
print(f"  CORRECT pairing             : {int(f.ok.sum())}")
print(f"  WRONG pairing (prefix bug)  : {int((~f.ok & f.mask_stem.notna()).sum())}")

wrong = f[~f.ok & f.mask_stem.notna()].copy()
print(f"\n=== the wrong pairings ===")
print(f"  distinct images affected    : {wrong.image_path.nunique()}")
print(f"  by source                   : {wrong.source.value_counts().to_dict()}")
print(f"  by split                    : {wrong.split.value_counts().to_dict()}")
print(f"  by label                    : {wrong.label.value_counts().to_dict()}")

# is the borrowed mask always from the same group?
key = f.set_index("image_path")["group_id"].to_dict()
same_group = 0
for _, r in wrong.iterrows():
    donor = r.mask_stem
    # the donor image is the one whose stem equals the mask stem
    for cand, g in key.items():
        if os.path.splitext(os.path.basename(cand))[0] == donor:
            if g == r.group_id:
                same_group += 1
            break
print(f"  donor in the SAME group     : {same_group} / {len(wrong)}")

print("\n  examples:")
for _, r in wrong.head(6).iterrows():
    print(f"    image stem: {r.image_stem}")
    print(f"    mask stem : {r.mask_stem}")

print(f"\n=== does a correct mask exist on disk for the affected images? ===")
# if the correct file exists but was not used, the bug is a selection bug only
import glob
have_own = 0
for _, r in wrong.head(200).iterrows():
    if glob.glob(os.path.join(os.path.dirname(r.mask_path), f"{r.image_stem}_*.png")):
        have_own += 1
print(f"  among the first 200 affected images, a file named <correct stem>_*.png exists for "
      f"{have_own} of them")
print("  (a file named <correct stem>_*.png may still be the wrong sibling's mask, because the")
print("   original naming used the path hash of another machine)")
