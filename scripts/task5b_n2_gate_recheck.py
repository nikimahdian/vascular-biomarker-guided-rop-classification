#!/usr/bin/env python
"""Task 5B-N2 run 3 — gate record for a PASSING run, plus its resolution limit.

Why this file exists
--------------------
The third execution PASSED the predeclared gate and continued straight into the perturbation
battery in the same process. Only the failure path of `scripts/task5b_n2.py` writes
`_private_audit/task5b_n2_gate.json`, so after the passing run that file still held the stale
run-2 INCOMPLETE record. This script writes the run-3 record.

Two sources, two different resolutions — both are reported, neither is allowed to masquerade as
the other:

  AUTHORITATIVE  `_private_audit/task5b_n2_summary.json`, written by run 3 itself from the
                 IN-MEMORY comparison. That comparison is the one the predeclared gate names:
                 refactored module output vs the frozen table as float64, before serialisation.
                 max |delta| over the four float fields = 3.552713678800501e-15 <= 1e-12 (PASS);
                 the two exact fields = 0.0 (PASS); NaN patterns identical; 0 of 8,870 rows differ.

  CORROBORATIVE  `task5b_n2_equivalence.csv` vs `clinical_measurement_v1.csv`, compared as written
                 decimal strings. All six columns are textually IDENTICAL on all 8,870 rows.
                 This is NOT independent proof of bit-identity: the frozen artefact carries at most
                 16 significant decimal digits, and at these magnitudes a 1-ULP difference
                 (1.11e-16 near 0.5, 3.55e-15 near 30) is below that resolution, so two adjacent
                 doubles can share one written string. Comparing the saved CSV therefore returns a
                 floor of 0 and cannot resolve what the in-memory comparison resolved.

Consequence for interpretation, stated plainly: the refactor is equivalent to the frozen table
within the predeclared tolerance; the saved artefacts are consistent with bit-identity but cannot
certify it; the in-memory numbers from the run are the record.

No measurement and no image processing happens here. Constants are imported from
`scripts/task5b_n2.py`, the file that ran, so they cannot drift.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.task5b_n2 import AV_COLS, EXACT_FIELDS, PREDECLARED_ATOL, EXTRA, ROOT  # noqa: E402

FROZEN = ROOT / "data/features/clinical_measurement_v1.csv"
FROZEN_SHA = "db123ac5f663f4925ac9fff52d204bede85963966794062d1855e315a4e38ffc"


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> None:
    summ = json.loads((ROOT / "_private_audit/task5b_n2_summary.json").read_text())
    got_sha = sha(FROZEN)
    print(f"  frozen table sha256 : {got_sha}")
    print(f"  matches expected    : {got_sha == FROZEN_SHA}")

    fo = pd.read_csv(FROZEN, usecols=["image_path"] + AV_COLS + EXTRA)
    ne = pd.read_csv(ROOT / "_private_audit/task5b_n2_equivalence.csv",
                     usecols=["image_path"] + AV_COLS + EXTRA)
    M = fo.merge(ne, on="image_path", suffixes=("_old", "_new"))

    auth = {
        "AV_REFACTOR_CANONICAL_N": int(len(M)),
        "AV_ROWS_DIFFERENT": summ["AV_ROWS_DIFFERENT"],
        "AV_MAX_ABS_DELTA_FLOAT_FIELDS": summ["AV_MAX_ABS_DELTA_EQUIVALENCE"],
        "AV_MAX_ABS_DELTA_EXACT_FIELDS": summ["AV_MAX_ABS_DELTA_EXACT_FIELDS"],
        "DIAGNOSTIC_CONTROL_MAX_DELTA": summ["DIAGNOSTIC_CONTROL_MAX_DELTA"],
        "PREDECLARED_ATOL": summ["PREDECLARED_ATOL"],
        "module_sha256": summ["module_sha256"],
    }
    auth_pass = (auth["AV_ROWS_DIFFERENT"] == 0
                 and auth["AV_MAX_ABS_DELTA_FLOAT_FIELDS"] <= PREDECLARED_ATOL
                 and auth["AV_MAX_ABS_DELTA_EXACT_FIELDS"] == 0.0
                 and auth["DIAGNOSTIC_CONTROL_MAX_DELTA"] == 0.0)
    print()
    print("  AUTHORITATIVE (in-memory comparison recorded by run 3)")
    for k, v in auth.items():
        print(f"    {k:36s} : {v}")
    print(f"    {'AV_FULL_BEHAVIOR_EQUIVALENCE':36s} : {'PASS' if auth_pass else 'FAIL'}")

    # ---- corroboration at the resolution the saved artefacts actually have
    per_col = {}
    for c in AV_COLS + EXTRA:
        o = [str(x) for x in M[f"{c}_old"].tolist()]
        n = [str(x) for x in M[f"{c}_new"].tolist()]
        n_text_diff = int(sum(1 for a, b in zip(o, n) if a != b))
        a = M[f"{c}_old"].to_numpy(float)
        b = M[f"{c}_new"].to_numpy(float)
        fin = np.isfinite(a) & np.isfinite(b)
        per_col[c] = {
            "written_string_differences": n_text_diff,
            "nan_pattern_same": bool(np.array_equal(np.isnan(a), np.isnan(b))),
            "max_abs_delta_from_saved_csv": float(np.max(np.abs(a[fin] - b[fin]))) if fin.any()
            else 0.0,
            "predeclared_class": "EXACT" if c in EXACT_FIELDS else "FLOAT",
        }
    print()
    print("  CORROBORATIVE (saved decimal strings; floor of 0 by construction)")
    for c, d in per_col.items():
        print(f"    {c:24s} written_string_differences={d['written_string_differences']:4d}  "
              f"nan_pattern_same={d['nan_pattern_same']}  "
              f"max|delta|={d['max_abs_delta_from_saved_csv']:.3e}")

    out = {
        "task": "Task 5B-N2", "attempt": 3,
        "authoritative_source": "_private_audit/task5b_n2_summary.json (in-memory, written by run 3)",
        "frozen_table_sha256": got_sha,
        "frozen_table_sha256_matches_frozen_record": bool(got_sha == FROZEN_SHA),
        "AUTHORITATIVE": auth,
        "AV_FULL_BEHAVIOR_EQUIVALENCE": "PASS" if auth_pass else "FAIL",
        "CORROBORATIVE": {
            "source": "task5b_n2_equivalence.csv vs clinical_measurement_v1.csv, as written strings",
            "merged_rows": int(len(M)),
            "per_column": per_col,
            "resolution_limit": ("the frozen artefact carries at most 16 significant decimal digits; "
                                 "a 1-ULP difference (1.11e-16 near 0.5, 3.55e-15 near 30) is below "
                                 "that resolution, so a floor of 0 here does not prove bit-identity"),
        },
        "interpretation": ("equivalent within the predeclared tolerance; the saved artefacts are "
                           "consistent with bit-identity but cannot certify it; the in-memory "
                           "numbers from run 3 are the record"),
        "note": "record written after the run's success path; no measurement performed",
    }
    (ROOT / "_private_audit/task5b_n2_gate.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print()
    if not (auth_pass and got_sha == FROZEN_SHA):
        raise SystemExit("GATE_RECORD_INCONSISTENT")
    print("  GATE_RECORD_OK  AV_FULL_BEHAVIOR_EQUIVALENCE = PASS")


if __name__ == "__main__":
    main()
