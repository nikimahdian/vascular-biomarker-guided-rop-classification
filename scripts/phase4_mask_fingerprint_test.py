#!/usr/bin/env python
"""Task 4B: dated fingerprints of the HISTORICAL mask generation.

results/phase4_mask_review_manifest.csv (mtime 2026-08-27 02:04, ~6 minutes after the
historical feature table at 01:58) records 32 (image_path, mask_path, vessel_density)
triples. results/phase4_feature_quality.csv records per-feature min/median/max for the
whole historical table. Recomputing vessel_density from the mask file at the recorded
mask_path tests, with a dated fingerprint, whether that file's content has changed.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
RES = f"{ROOT}/results"


def dens(p: str):
    try:
        a = np.array(Image.open(p).convert("L"))
        m = (a > 127).astype(np.uint8)
        return float(m.sum()) / float(m.shape[0] * m.shape[1]), int(m.sum())
    except Exception:  # noqa: BLE001
        return np.nan, -1


print("=" * 100)
print("1. DATED FINGERPRINT TEST: 2026-08-27 RECORDED vessel_density VS FILE TODAY")
print("=" * 100)
q = pd.read_csv(f"{RES}/phase4_mask_review_manifest.csv")
print(f"  phase4_mask_review_manifest.csv rows={len(q)}  "
      f"mtime={time.ctime(os.path.getmtime(f'{RES}/phase4_mask_review_manifest.csv'))}")
print(f"  historical feature table mtime = "
      f"{time.ctime(os.path.getmtime(f'{ROOT}/data/features/biomarker_features.csv'))}")
print()
rows = []
for _, r in q.iterrows():
    d_now, vp_now = dens(r.mask_path)
    rows.append({
        "image_path": r.image_path, "mask_path": r.mask_path,
        "recorded_vessel_density": float(r.vessel_density),
        "density_from_file_now": d_now, "vp_now": vp_now,
        "match": bool(np.isfinite(d_now) and abs(d_now - float(r.vessel_density)) < 1e-12),
        "mask_mtime": time.ctime(os.path.getmtime(r.mask_path)) if os.path.exists(r.mask_path) else "MISSING",
    })
M = pd.DataFrame(rows)
print(f"  files present        : {int((M.mask_mtime != 'MISSING').sum())} / {len(M)}")
print(f"  EXACT density match  : {int(M.match.sum())} / {len(M)}")
print()
for _, r in M.iterrows():
    flag = "OK " if r.match else "DIFF"
    print(f"  {flag} rec={r.recorded_vessel_density:.12f} now={r.density_from_file_now:.12f} "
          f"vp={r.vp_now:7d} {os.path.basename(r.mask_path)}")
M.to_csv(f"{ROOT}/_private_audit/phase4_fingerprint_test.csv", index=False)

print()
print("=" * 100)
print("2. WHOLE-HISTORICAL-TABLE FINGERPRINT (phase4_feature_quality.csv)")
print("=" * 100)
fq = pd.read_csv(f"{RES}/phase4_feature_quality.csv")
hist = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv")
corr = pd.read_csv(f"{ROOT}/data/features/biomarker_features_historical_equivalent_corrected_v1.csv")
print(f"  {'feature':24s} {'hist_min':>14s} {'corr_min':>14s} {'hist_med':>16s} "
      f"{'corr_med':>16s} {'hist_max':>14s} {'corr_max':>14s}")
for _, r in fq.iterrows():
    f = r.feature
    if f not in hist.columns:
        continue
    print(f"  {f:24s} {r['min']:14.8g} {corr[f].min():14.8g} {r['median']:16.10g} "
          f"{corr[f].median():16.10g} {r['max']:14.8g} {corr[f].max():14.8g}")
print()
print("  the phase4 file describes the HISTORICAL table; the corrected column is built")
print("  from CURRENT masks. Agreement of min/median/max indicates the mask generation")
print("  drift is small in distributional terms even though it is exact-match fatal.")

print()
print("=" * 100)
print("3. MASK DIRECTORY AND MANIFEST TIMELINE")
print("=" * 100)
MD = f"{ROOT}/data/masks"
for f in ("mask_manifest.csv", "mask_manifest_v2.csv", "mask_manifest_canonical_v1.csv",
          "mask_manifest_legacy_image_level_20260826.csv"):
    p = f"{MD}/{f}"
    if os.path.exists(p):
        print(f"  {f:52s} {time.ctime(os.path.getmtime(p))}")
print(f"  {'data/masks directory':52s} {time.ctime(os.path.getmtime(MD))}")
print(f"  {'data/features/biomarker_features.csv':52s} "
      f"{time.ctime(os.path.getmtime(f'{ROOT}/data/features/biomarker_features.csv'))}")
print()
print("  mask_manifest_v2.csv head:")
v2 = pd.read_csv(f"{MD}/mask_manifest_v2.csv")
print(f"    rows={len(v2)} cols={list(v2.columns)}")
print(v2.head(3).to_string())
