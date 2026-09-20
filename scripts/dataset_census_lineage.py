#!/usr/bin/env python
"""Task 2 part 2: find the exact LOSO population."""
from __future__ import annotations

import glob
import hashlib
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"


def sha(p):
    try:
        with open(p, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:  # noqa: BLE001
        return None


print("=" * 100)
print("A. EVERY manifest / feature / split artifact on disk with its date")
print("=" * 100)
pats = [f"{ROOT}/data/masks/*.csv", f"{ROOT}/data/features/*.csv", f"{ROOT}/data/splits/*.csv",
        f"{ROOT}/data/splits_candidate*/*.csv", f"{ROOT}/data/*.csv"]
for pat in pats:
    for p in sorted(glob.glob(pat)):
        try:
            d = pd.read_csv(p)
            src = d.source.value_counts().to_dict() if "source" in d.columns else "-"
            print(f"  {os.path.getmtime(p):.0f}  rows={len(d):6d}  {os.path.basename(p):58s} "
                  f"src={src}")
        except Exception as e:  # noqa: BLE001
            print(f"  {os.path.getmtime(p):.0f}  ERR {os.path.basename(p)}: {type(e).__name__}")

print()
print("=" * 100)
print("B. THE LEGACY MANIFEST (dated 2026-08-26, the LOSO date)")
print("=" * 100)
p = f"{ROOT}/data/masks/mask_manifest_legacy_image_level_20260826.csv"
if os.path.exists(p):
    d = pd.read_csv(p)
    print(f"  rows={len(d)}  cols={list(d.columns)}  sha256={sha(p)}")
    print(f"  unique image_path = {d.image_path.nunique()}")
    print(f"  unique mask_path  = {d.mask_path.nunique() if 'mask_path' in d.columns else '-'}")
    print(f"  source counts     = {d.source.value_counts().to_dict()}")
    print(f"  label counts      = {d.label.value_counts().to_dict()}")
    print(f"  -> reproduces 8947 = {len(d) == 8947}")
    print(f"  -> reproduces plus 6004 / farfum 1533 / farabi 1410 = "
          f"{d.source.value_counts().to_dict() == {'plus': 6004, 'farfum_rop': 1533, 'farabi': 1410}}")
    if "mask_path" in d.columns:
        miss = sum(1 for m in d.mask_path if not os.path.exists(str(m)))
        print(f"  mask_path entries that do not exist on disk: {miss} / {len(d)}")
    if "split" in d.columns:
        print(f"  split counts      = {d.split.value_counts().to_dict()}")
    can = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv", usecols=["image_path"])
    cs, ls = set(can.image_path), set(d.image_path)
    print(f"\n  LOSO_ONLY (legacy - canonical) = {len(ls - cs)}")
    print(f"  CANONICAL_ONLY                 = {len(cs - ls)}")
    print(f"  intersection                   = {len(ls & cs)}")
    loso_only = d[d.image_path.isin(ls - cs)]
    print(f"  LOSO-only by source            = {loso_only.source.value_counts().to_dict()}")
    print(f"  LOSO-only by label             = {loso_only.label.value_counts().to_dict()}")
    loso_only.head(200).to_csv(f"{ROOT}/_private_audit/loso_only_rows.csv", index=False)
    # are they in the exclusion tables?
    for t in ("excluded_ambiguous_exact_duplicates", "missing_masks", "rerun_masks"):
        tp = f"{ROOT}/data/splits/{t}.csv"
        if os.path.exists(tp):
            tt = pd.read_csv(tp)
            inx = len(set(loso_only.image_path) & set(tt.image_path))
            print(f"  LOSO-only rows present in {t}: {inx}")
else:
    print("  MISSING")
