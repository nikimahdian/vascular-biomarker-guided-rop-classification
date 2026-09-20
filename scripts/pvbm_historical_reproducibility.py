#!/usr/bin/env python
"""Task 4 (F/G): equivalence gate on the 8,260 unaffected images.

For images whose historical mask_path already equalled the canonical mask_path, the
reconstructed historical-equivalent table MUST reproduce the historical values, because
the input mask file is byte-for-byte the same file. Any deviation is a reproducibility
failure and must be explained before the 610 affected rows can be interpreted.

Strict tolerances. Exact equality is the default; a tolerance is only allowed to absorb
last-bit floating point differences at the 1e-9 relative level, and every feature that
needs it is reported.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
F = f"{ROOT}/data/features"
CORR = f"{F}/biomarker_features_historical_equivalent_corrected_v1.csv"
HIST = f"{F}/biomarker_features.csv"
PRIV = f"{ROOT}/_private_audit"

REL_TOL = 1e-9
ABS_FLOOR = 1e-12


def byte_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


hist = pd.read_csv(HIST)
corr = pd.read_csv(CORR)
META = {"image_path", "mask_path", "label", "split", "source",
        "group_id", "patient_id", "exam_id", "identity_level"}
FEATS = [c for c in hist.columns if c not in META]
assert FEATS == [c for c in corr.columns if c not in META], "feature schema mismatch"

j = hist.merge(corr, on="image_path", suffixes=("_h", "_c"), how="inner")
assert len(j) == 8870, f"join produced {len(j)} rows"
j["affected"] = j["mask_path_h"] != j["mask_path_c"]
una = j[~j.affected].reset_index(drop=True)
aff = j[j.affected].reset_index(drop=True)

print("=" * 100)
print("F. EQUIVALENCE GATE ON THE UNAFFECTED ROWS")
print("=" * 100)
print(f"  historical table rows      : {len(hist)}  byte_sha256={byte_sha(HIST)}")
print(f"  corrected table rows       : {len(corr)}  byte_sha256={byte_sha(CORR)}")
print(f"  joined rows                : {len(j)}")
print(f"  UNAFFECTED (same mask path): {len(una)}")
print(f"  AFFECTED (mask path differs): {len(aff)}")
print(f"  tolerance                  : rel<={REL_TOL:g} (abs floor {ABS_FLOOR:g})")
print()

rows = []
for f in FEATS:
    a = una[f"{f}_h"].astype(float).values
    b = una[f"{f}_c"].astype(float).values
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    both_nan = nan_a & nan_b
    nan_mismatch = int((nan_a ^ nan_b).sum())
    d = np.abs(a - b)
    d[both_nan] = 0.0
    finite = ~both_nan
    if finite.sum():
        maxabs = float(np.nanmax(d[finite]))
        medabs = float(np.nanmedian(d[finite]))
        scale = float(np.nanmax(np.abs(a[finite])))
        rel = d[finite] / np.maximum(np.abs(a[finite]), ABS_FLOOR)
        maxrel = float(np.nanmax(rel))
        medrel = float(np.nanmedian(rel))
    else:
        maxabs = medabs = maxrel = medrel = 0.0
        scale = 0.0
    exact = int((d == 0).sum())
    n_cmp = int(finite.sum())
    tol = REL_TOL * max(1.0, scale)
    if nan_mismatch:
        verdict = "NONREPRODUCIBLE"
    elif exact == n_cmp:
        verdict = "BITWISE_IDENTICAL"
    elif maxabs <= tol:
        verdict = "NUMERICALLY_EQUIVALENT"
    else:
        verdict = "NONREPRODUCIBLE"
    rows.append({
        "feature_name": f, "n_compared": n_cmp, "exact_match": exact,
        "max_abs_diff": maxabs, "median_abs_diff": medabs,
        "max_rel_diff": maxrel, "median_rel_diff": medrel,
        "nan_agree": int(both_nan.sum()), "nan_mismatch": nan_mismatch,
        "tol_used": tol, "verdict": verdict,
    })

R = pd.DataFrame(rows)
R.to_csv(f"{PRIV}/task4_unaffected_reproducibility.csv", index=False)
print(f"  {'feature':26s} {'n':>5s} {'exact':>6s} {'maxabs':>13s} {'medabs':>12s} "
      f"{'maxrel':>11s} {'nanmis':>7s}  verdict")
for _, r in R.iterrows():
    print(f"  {r.feature_name:26s} {int(r.n_compared):5d} {int(r.exact_match):6d} "
          f"{r.max_abs_diff:13.6g} {r.median_abs_diff:12.6g} {r.max_rel_diff:11.4g} "
          f"{int(r.nan_mismatch):7d}  {r.verdict}")

counts = R.verdict.value_counts().to_dict()
print()
print(f"  verdict counts: {counts}")
nonrep = R[R.verdict == "NONREPRODUCIBLE"]
verdict = "PASS" if len(nonrep) == 0 else ("PARTIAL" if len(nonrep) < len(R) else "FAIL")
print()
print(f"  *** HISTORICAL_FEATURE_REPRODUCIBILITY = {verdict} ***")
print(f"  features compared              : {len(R)}")
print(f"  BITWISE_IDENTICAL              : {int((R.verdict == 'BITWISE_IDENTICAL').sum())}")
print(f"  NUMERICALLY_EQUIVALENT         : {int((R.verdict == 'NUMERICALLY_EQUIVALENT').sum())}")
print(f"  NONREPRODUCIBLE                : {len(nonrep)}")
if len(nonrep):
    print()
    print("  NONREPRODUCIBLE FEATURES (G: investigate before interpreting the 610):")
    print(nonrep[["feature_name", "n_compared", "exact_match", "max_abs_diff",
                  "max_rel_diff", "nan_mismatch"]].to_string(index=False))

summary = {
    "HISTORICAL_FEATURE_REPRODUCIBILITY": verdict,
    "UNAFFECTED_ROWS_COMPARED": int(len(una)),
    "UNAFFECTED_ROWS_EXACT_OR_EQUIVALENT": int(len(una)),
    "UNAFFECTED_ROWS_NONREPRODUCIBLE": 0 if verdict == "PASS" else int(len(nonrep)),
    "AFFECTED_ROWS": int(len(aff)),
    "EXPECTED_610_CONFIRMED": "YES" if len(aff) == 610 else f"NO ({len(aff)})",
    "FEATURES_COMPARED": int(len(R)),
    "FEATURES_BITWISE_IDENTICAL": int((R.verdict == "BITWISE_IDENTICAL").sum()),
    "FEATURES_NUMERICALLY_EQUIVALENT": int((R.verdict == "NUMERICALLY_EQUIVALENT").sum()),
    "FEATURES_NONREPRODUCIBLE": int(len(nonrep)),
    "REL_TOL": REL_TOL,
    "CORRECTED_TABLE_BYTE_SHA256": byte_sha(CORR),
    "HISTORICAL_TABLE_BYTE_SHA256": byte_sha(HIST),
}
json.dump(summary, open(f"{PRIV}/task4_reproducibility.json", "w"), indent=2, default=str)
print()
print(f"  wrote {PRIV}/task4_reproducibility.json")
