#!/usr/bin/env python3
"""Verify Phase 0-4 acceptance gates on remote."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

root = Path("/Users/moniaz/niki")
checks: list[tuple[str, bool, str]] = []

backup = Path("/Users/moniaz/niki_before_patient_fix_20260826.tgz")
checks.append(("P0 backup exists", backup.exists(), str(backup)))
checks.append(
    (
        "P0 provisional dir",
        (root / "results/current_provisional_20260826").exists(),
        "results/current_provisional_20260826",
    )
)

idmap = root / "data/metadata/identity_map.csv"
checks.append(("P1 identity_map", idmap.exists(), f"rows={sum(1 for _ in open(idmap)) - 1}"))

splits = pd.read_csv(root / "data/splits/all.csv")
for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
    overlap = set(splits[splits.split == left].group_id) & set(
        splits[splits.split == right].group_id
    )
    checks.append((f"P2 zero overlap {left}/{right}", len(overlap) == 0, str(len(overlap))))
checks.append(("P2 retained images", len(splits) == 8870, str(len(splits))))

proc = subprocess.run(
    [
        str(root / ".venv/bin/python"),
        "-m",
        "pytest",
        "tests/test_prepare_split.py",
        "tests/test_phase3_regressions.py",
        "-q",
        "--tb=no",
    ],
    cwd=root,
    capture_output=True,
    text=True,
)
checks.append(("P3 regression tests", proc.returncode == 0, proc.stdout.strip().split("\n")[-1]))

report = json.loads((root / "results/phase4_quality_gates.json").read_text())
checks.append(("P4 automated gates", report["status"] == "AUTOMATED_GATES_PASS", report["status"]))
checks.append(
    (
        "P4 review queue complete",
        report["mask_review_queue"]["unresolved"] == 0,
        str(report["mask_review_queue"]),
    )
)
checks.append(("P4 aligned rows", report["rows"] == len(splits), f"{report['rows']} vs {len(splits)}"))

scope = root / "results/phase4_clinical_scope.md"
checks.append(("P4 clinical scope doc", scope.exists(), str(scope)))

all_pass = all(item[1] for item in checks)
for name, ok, detail in checks:
    status = "PASS" if ok else "FAIL"
    print(f"{status} | {name} | {detail}")
print("ALL_PASS" if all_pass else "BLOCKED", all_pass)
raise SystemExit(0 if all_pass else 1)
