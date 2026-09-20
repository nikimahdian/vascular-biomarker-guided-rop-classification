#!/usr/bin/env python
"""Section B: explain mask_manifest.csv = 17740 rows for 8870 images.

Read-only. Every claim about the two rows per image is computed, not assumed.
"""
from __future__ import annotations

import hashlib
import os
import sys
from collections import Counter

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
MAN = f"{ROOT}/data/masks/mask_manifest.csv"


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


print("=" * 100)
print("B1. FILE LEVEL FACTS")
print("=" * 100)
print(f"mask_manifest.csv sha256        : {sha(MAN)}")
print(f"mask_manifest.csv size          : {os.path.getsize(MAN):,} bytes")
man = pd.read_csv(MAN)
print(f"rows                            : {len(man)}")
print(f"columns                         : {list(man.columns)}")
print(f"unique image_path               : {man.image_path.nunique()}")
print(f"unique mask_path                : {man.mask_path.nunique()}")
print(f"ratio rows/images               : {len(man) / man.image_path.nunique():.6f}")
print(f"unique label values             : {sorted(man.label.unique())}")
print(f"unique split values             : {sorted(man.split.astype(str).unique())}")
print(f"unique source values            : {sorted(man.source.astype(str).unique())}")

print()
print("=" * 100)
print("B2. ROWS PER IMAGE")
print("=" * 100)
c = man.groupby("image_path").size().value_counts().sort_index()
print("n images with k rows:")
for k, v in c.items():
    print(f"   k={k}: {v} images")
print(f"exactly 2 rows for every image: {bool((man.groupby('image_path').size() == 2).all())}")

print()
print("=" * 100)
print("B3. WHAT DISTINGUISHES THE TWO ROWS OF AN IMAGE?")
print("=" * 100)
g = man.groupby("image_path")
dup2 = g.filter(lambda d: len(d) == 2)
print(f"rows in two-row images: {len(dup2)}")

same_mask = g.filter(lambda d: len(d) == 2 and d.mask_path.nunique() == 1)
diff_mask = g.filter(lambda d: len(d) == 2 and d.mask_path.nunique() == 2)
print(f"  images whose TWO rows have the SAME mask_path : {same_mask.image_path.nunique()}")
print(f"  images whose TWO rows have DIFFERENT mask_path: {diff_mask.image_path.nunique()}")

# is the row order within each pair meaningful?
first = dup2.groupby("image_path", sort=False).nth(0)
second = dup2.groupby("image_path", sort=False).nth(1)
for col in [x for x in man.columns if x not in ("image_path", "mask_path")]:
    neq = int((first[col].astype(str).values != second[col].astype(str).values).sum())
    print(f"  rows differ in '{col}': {neq} of {len(first)}")

print()
print("=" * 100)
print("B4. FOR THE IMAGES WITH TWO DIFFERENT MASKS, WHICH IS WHICH?")
print("=" * 100)
if diff_mask.image_path.nunique():
    ex = diff_mask.image_path.drop_duplicates().head(10)
    for ip in ex:
        d = man[man.image_path == ip]
        print(f"\n  image: {os.path.basename(ip)}")
        for k, (_, r) in enumerate(d.iterrows()):
            mp = r.mask_path
            exists = os.path.exists(mp)
            print(f"    row {k}: {os.path.basename(mp)}  exists={exists}"
                  f"  sha={(sha(mp)[:16] if exists else '-')}")

print()
print("=" * 100)
print("B5. DUPLICATE IDENTICAL ROWS (same image AND same mask)")
print("=" * 100)
key = man.duplicated(["image_path", "mask_path"], keep=False).sum()
print(f"rows that are a duplicate of (image_path, mask_path): {key}")
exact_dup_rows = man.duplicated(keep=False).sum()
print(f"rows that are a FULL duplicate row                 : {exact_dup_rows}")
print(f"distinct full rows                                 : {man.drop_duplicates().shape[0]}")

print()
print("=" * 100)
print("B6. DOES THE FEATURE TABLE MATCH THE FIRST OR THE SECOND ROW?")
print("=" * 100)
feat = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv",
                   usecols=["image_path", "mask_path"])
fm = dict(zip(feat.image_path, feat.mask_path))
first_m = dict(zip(first.index, first.mask_path))
second_m = dict(zip(second.index, second.mask_path))
n1 = n2 = nneither = 0
for ip, m in fm.items():
    if first_m.get(ip) == m:
        n1 += 1
    elif second_m.get(ip) == m:
        n2 += 1
    else:
        nneither += 1
print(f"  feature-table mask == manifest row 0 : {n1}")
print(f"  feature-table mask == manifest row 1 : {n2}")
print(f"  neither                              : {nneither}")
