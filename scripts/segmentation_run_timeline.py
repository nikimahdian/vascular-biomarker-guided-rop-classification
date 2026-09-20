#!/usr/bin/env python
"""Task 4B finalisation: freeze inventory, checkpoint candidate table, reclassification."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
PRIV = f"{ROOT}/_private_audit"
os.makedirs(PRIV, exist_ok=True)


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


print("=" * 100)
print("B. FREEZE INVENTORY OF THE CURRENT STATE")
print("=" * 100)
files = [
    f"{ROOT}/src/segmentation/infer_masks.py",
    f"{ROOT}/src/segmentation/models.py",
    f"{ROOT}/src/segmentation/train.py",
    f"{ROOT}/src/segmentation/dataset.py",
    f"{ROOT}/configs/config.yaml",
    f"{ROOT}/requirements-lock.txt",
    f"{ROOT}/weights/best_weight_DeepLabV3+_resize_27",
    f"{ROOT}/data/masks/mask_manifest.csv",
    f"{ROOT}/data/masks/mask_manifest_v2.csv",
    f"{ROOT}/data/masks/mask_manifest_canonical_v1.csv",
    f"{ROOT}/data/masks/mask_manifest_legacy_image_level_20260826.csv",
    f"{ROOT}/data/features/biomarker_features.csv",
    f"{ROOT}/data/features/biomarker_features_historical_equivalent_corrected_v1.csv",
    f"{ROOT}/data/manifests/vessel_prob_v1.json",
]
rows = []
for p in files:
    if not os.path.exists(p):
        rows.append({"path": p, "size": -1, "mtime": "MISSING", "sha256": ""})
        continue
    rows.append({"path": p, "size": os.path.getsize(p),
                 "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime(os.path.getmtime(p))),
                 "sha256": sha(p)})
F = pd.DataFrame(rows)
F.to_csv(f"{PRIV}/task4b_freeze_inventory.csv", index=False)
print(F[["path", "size", "mtime"]].to_string(index=False))

print()
pngs = glob.glob(f"{ROOT}/data/masks/*.png")
print(f"data/masks PNG count: {len(pngs)}")
tot = sum(os.path.getsize(p) for p in pngs[:500])
print(f"  (first 500 masks total {tot:,} bytes)")

print()
print("=" * 100)
print("E. CHECKPOINT CANDIDATE TABLE")
print("=" * 100)
cand = json.load(open(f"{PRIV}/task4b_diffonly_candidates.json"))
surv = f"{ROOT}/weights/best_weight_DeepLabV3+_resize_27"
surv_sha = sha(surv)
rows = []
for fn, e in cand.items():
    rows.append({
        "checkpoint_path": f"{ROOT}/_private/historical_seg_candidates/{fn}",
        "filename": fn, "size": e.get("size"),
        "sha256": e.get("sha256"),
        "architecture": e.get("architecture"),
        "encoder": e.get("encoder"),
        "best_threshold": e.get("best_threshold"),
        "best_area_exact_of_300": e.get("best_area_exact"),
        "best_median_abs_delta": e.get("best_median_abs_delta"),
        "identical_to_surviving_checkpoint": e.get("sha256") == surv_sha,
        "origin": "configs/config.yaml pretrained_weight_ids (Google Drive)",
    })
rows.append({
    "checkpoint_path": surv, "filename": os.path.basename(surv),
    "size": os.path.getsize(surv), "sha256": surv_sha,
    "architecture": "MAnet", "encoder": "resnet34",
    "best_threshold": 0.20, "best_area_exact_of_300": 0,
    "best_median_abs_delta": 153.5,
    "identical_to_surviving_checkpoint": True,
    "origin": "in-project weights/ (mtime 2026-08-22)",
})
C = pd.DataFrame(rows)
C.to_csv(f"{PRIV}/historical_segmentation_checkpoint_candidates.csv", index=False)
print(C[["filename", "architecture", "encoder", "best_threshold",
         "best_area_exact_of_300", "identical_to_surviving_checkpoint"]].to_string(index=False))

print()
print("=" * 100)
print("S. RECLASSIFICATION OF THE TASK-4 TABLE")
print("=" * 100)
rec = {
    "table_path": f"{ROOT}/data/features/biomarker_features_historical_equivalent_corrected_v1.csv",
    "sha256": sha(f"{ROOT}/data/features/biomarker_features_historical_equivalent_corrected_v1.csv"),
    "PREVIOUS_CLASSIFICATION": "historical-equivalent corrected feature table",
    "NEW_CLASSIFICATION": "CURRENT_MASK_HISTORICAL_DEFINITION_BASELINE",
    "RATIONALE": (
        "Its PVBM feature definitions are historical and unchanged, but its segmentation "
        "measurement generation is the CURRENT one (checkpoint c373f538..., threshold 0.20, "
        "masks of 2026-08-30), not the generation that produced the historical feature table. "
        "It therefore cannot be called 'historical-equivalent'."
    ),
    "FILE_RETAINED": True,
    "FILE_RENAMED": False,
    "HISTORICAL_TABLE_CLASSIFICATION":
        "HISTORICAL_CONTAMINATED and NON_REPRODUCIBLE (segmentation generation unrecoverable)",
    "CORRECTED_MEASUREMENT_LAYER": "reserved for later clinical-v4-type definitions; not used",
}
json.dump(rec, open(f"{PRIV}/task4b_reclassification.json", "w"), indent=2, default=str)
for k, v in rec.items():
    print(f"  {k}: {v}")

sidecar = {
    "table": "biomarker_features_historical_equivalent_corrected_v1.csv",
    "sha256": rec["sha256"],
    "classification": "CURRENT_MASK_HISTORICAL_DEFINITION_BASELINE",
    "NOT": "HISTORICAL_EQUIVALENT_CORRECTED",
    "reclassified_by": "Task 4B",
    "reclassification_reason": rec["RATIONALE"],
    "reclassified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "historical_segmentation_state": "IRRECOVERABLE",
}
json.dump(sidecar, open(f"{ROOT}/artifacts/task4_table_classification.json", "w"), indent=2)
print(f"\n  sidecar written -> {ROOT}/artifacts/task4_table_classification.json")

print()
print("=" * 100)
print("FINAL RESULT STATUS")
print("=" * 100)
final = {
    "HISTORICAL_MASK_ARCHIVE_FOUND": "NO",
    "HISTORICAL_CHECKPOINT_CANDIDATES_FOUND": int(len(C)),
    "HISTORICAL_MASK_GENERATION_RECOVERED": "NO",
    "RECOVERY_METHOD": "NONE",
    "HISTORICAL_SEGMENTATION_STATE_IRRECOVERABLE": "YES",
    "CURRENT_VS_HISTORICAL_DIFFERENCE_CAUSE": "UNRESOLVED",
    "SUBSET_REPRODUCTION_N": 300,
    "SUBSET_AREA_EXACT_MATCH_N": 0,
    "PAIRING_BUG_EFFECT_ISOLATABLE": "NO",
    "TASK4_CAN_BE_REOPENED": "NO",
    "TASK4B_STATUS": "COMPLETE",
}
json.dump(final, open(f"{PRIV}/task4b_final.json", "w"), indent=2)
for k, v in final.items():
    print(f"  {k}: {v}")
