#!/usr/bin/env python
"""Task 2: reconcile the canonical 8870 cohort against the historical LOSO population."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
OUT = f"{ROOT}/_private_audit"
os.makedirs(OUT, exist_ok=True)


def sha(p):
    try:
        with open(p, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:  # noqa: BLE001
        return None


S = f"{ROOT}/data/splits"
feat = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv")
print("=" * 100)
print("1. CANONICAL COHORT, RECONSTRUCTED FROM FILES")
print("=" * 100)
print(f"  biomarker_features.csv rows = {len(feat)}  sha256={sha(f'{ROOT}/data/features/biomarker_features.csv')}")
print(f"  source counts               = {feat.source.value_counts().to_dict()}")
allsp = pd.read_csv(f"{S}/all.csv")
print(f"  data/splits/all.csv rows    = {len(allsp)}  sha256={sha(f'{S}/all.csv')}")
print(f"    (sha256 matches the locked split fingerprint: "
      f"{sha(f'{S}/all.csv') == '0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8'})")
print(f"  split counts                = {allsp.split.value_counts().to_dict()}")
print(f"  groups                      = {allsp.group_id.nunique()}")
print(f"  source counts in all.csv    = {allsp.source.value_counts().to_dict()}")

print()
print("=" * 100)
print("2. HISTORICAL LOSO ARTIFACTS (what they claim)")
print("=" * 100)
for f in ("loo_partial.json", "loo_results_partial.json", "loo_results_full.json"):
    p = f"{ROOT}/results/{f}"
    if not os.path.exists(p):
        print(f"  {f}: MISSING")
        continue
    print(f"  {f}:  sha256={sha(p)}")
    d = json.load(open(p))
    res = d.get("results", d) if isinstance(d, dict) else d
    agg = {}
    for r in res:
        if r.get("branch") == "A":
            agg[r["holdout"]] = r.get("n_test")
    print(f"    n_test by holdout (branch A): {agg}   total={sum(v for v in agg.values() if v)}")

print()
print("=" * 100)
print("3. THE EXCLUSION ARTIFACTS ON DISK")
print("=" * 100)
for f in ("excluded_ambiguous_exact_duplicates.csv", "missing_masks.csv", "rerun_masks.csv",
          "split_counts.csv", "group_assignments.csv"):
    p = f"{S}/{f}"
    if not os.path.exists(p):
        print(f"  {f}: MISSING")
        continue
    d = pd.read_csv(p)
    print(f"  {f}: rows={len(d)}  cols={list(d.columns)[:8]}  sha256={sha(p)[:16]}")
    if "source" in d.columns:
        print(f"      source counts: {d.source.value_counts().to_dict()}")

print()
print("=" * 100)
print("4. DECISIVE TEST: 8870 + excluded == 8947 ?")
print("=" * 100)
exc = pd.read_csv(f"{S}/excluded_ambiguous_exact_duplicates.csv")
print(f"  excluded rows                 : {len(exc)}")
print(f"  canonical rows                : {len(feat)}")
print(f"  canonical + excluded          : {len(feat) + len(exc)}")
print(f"  historical LOSO population    : 6004 + 1533 + 1410 = {6004 + 1533 + 1410}")
print(f"  MATCH                         : {len(feat) + len(exc) == 8947}")
if "source" in exc.columns:
    print(f"  excluded by source            : {exc.source.value_counts().to_dict()}")
    print(f"  needed extra by source        : plus +73, farfum_rop +4, farabi +0")
    can = feat.source.value_counts().to_dict()
    tot = pd.concat([feat[["source"]], exc[["source"]]]).source.value_counts().to_dict()
    print(f"  canonical+excluded by source  : {tot}")
    print(f"  -> would reproduce LOSO 6004/1533/1410: "
          f"{tot.get('plus') == 6004 and tot.get('farfum_rop') == 1533 and tot.get('farabi') == 1410}")

print()
print("=" * 100)
print("5. IDENTITY OF THE EXCLUDED ROWS vs THE CANONICAL SET")
print("=" * 100)
if "image_path" in exc.columns:
    can_paths = set(feat.image_path)
    ex_paths = set(exc.image_path)
    print(f"  excluded paths                       : {len(ex_paths)}")
    print(f"  excluded paths ALSO in canonical     : {len(ex_paths & can_paths)}")
    print(f"  excluded paths NOT in canonical      : {len(ex_paths - can_paths)}")
    print(f"  missing from canonical              : {len(can_paths - ex_paths) - (len(can_paths) - len(ex_paths & can_paths))}")
    if "exclusion_reason" in exc.columns:
        print(f"  exclusion reasons: {exc.exclusion_reason.value_counts().to_dict()}")
    if "sha256" in exc.columns:
        print(f"  excluded rows carry pixel sha256: {exc.sha256.notna().sum()}")
    print("\n  first 5 excluded rows (paths abbreviated):")
    for p in list(ex_paths - can_paths)[:5]:
        print(f"    {p[:60]}...{p[-30:]}")
