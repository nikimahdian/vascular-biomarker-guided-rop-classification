#!/usr/bin/env python
"""Task 2 part 3: classify the 77 LOSO-only images and quantify the fold-level impact."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
PRIV = f"{ROOT}/_private_audit"
os.makedirs(PRIV, exist_ok=True)

can = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv")
leg = pd.read_csv(f"{ROOT}/data/masks/mask_manifest_legacy_image_level_20260826.csv")
exc = pd.read_csv(f"{ROOT}/data/splits/excluded_ambiguous_exact_duplicates.csv")
can = can[["image_path", "label", "source", "group_id"]].copy()

# the LOSO population: legacy minus the 13 farabi rows that the LOSO artifacts do not contain
loso_counts = {"plus": 6004, "farfum_rop": 1533, "farabi": 1410}
# The LOSO farabi holdout count equals the canonical farabi count exactly, so the farabi part of
# the LOSO population is the canonical farabi set. The plus and farfum parts are the larger legacy
# sets. This is deterministic and reproduces all three counts.
can_farabi = set(can[can.source == "farabi"].image_path)
loso = leg[(leg.source != "farabi") |
           ((leg.source == "farabi") & leg.image_path.isin(can_farabi))].copy()
print(f"  legacy farabi={len(leg[leg.source=='farabi'])}  LOSO farabi={loso_counts['farabi']}")
print(f"  LOSO population rows={len(loso)}  sources={loso.source.value_counts().to_dict()}")
print(f"  reproduces 8947 = {len(loso) == 8947 and loso.source.value_counts().to_dict() == loso_counts}")

# group_id comes from the 8960-row candidate split, which does carry it
gid = None
for c in ("splits_candidate_v3_verified", "splits_candidate_v2", "splits_pre_phase4_excludes_20260827"):
    p = f"{ROOT}/data/{c}/all.csv"
    if os.path.exists(p):
        t = pd.read_csv(p)
        if "group_id" in t.columns and len(t) == 8960:
            gid = t[["image_path", "group_id"]]
            print(f"  group_id source: {c}/all.csv  rows={len(t)}")
            break
if gid is not None:
    loso = loso.merge(gid, on="image_path", how="left")
else:
    loso["group_id"] = None
    print("  [warn] no 8960-row table with group_id found; group analysis limited")

cs = set(can.image_path)
ls = set(loso.image_path)
only = loso[loso.image_path.isin(ls - cs)].copy()
print(f"\nLOSO-only = {len(only)}  by source={only.source.value_counts().to_dict()}")

# ---- classification -----------------------------------------------------------------
excset = set(exc.image_path)
can_groups = set(can.group_id)
only["present_in_excluded_table"] = only.image_path.isin(excset)
only["group_shared_with_canonical"] = only.group_id.isin(can_groups)
only["group_in_canonical_by_exam"] = [g in can_groups for g in only.group_id]


def classify(r):
    if r.present_in_excluded_table:
        return "exact_duplicate_excluded_from_canonical"
    if r.group_shared_with_canonical:
        return "group_shared_with_canonical_row_unclassified"
    return "UNKNOWN"


only["reason"] = only.apply(classify, axis=1)
print("\n=== mutually exclusive classification ===")
print(only.reason.value_counts().to_string())

# ---- fold impact --------------------------------------------------------------------
print("\n" + "=" * 100)
print("FOLD-LEVEL IMPACT (training population, not just holdout)")
print("=" * 100)
imp = {}
for h in ("farabi", "farfum_rop", "plus"):
    hold = loso[loso.source == h]
    train = loso[loso.source != h]
    extra_train = train[train.image_path.isin(ls - cs)]
    extra_hold = hold[hold.image_path.isin(ls - cs)]
    can_hold = can[can.source == h]
    imp[h] = {"holdout_canonical_n": len(can_hold), "holdout_loso_n": len(hold),
              "extra_train_images": len(extra_train), "extra_holdout_images": len(extra_hold),
              "loso_train_n": len(train), "canonical_train_n": len(can) - len(can_hold),
              "extra_train_group_n": extra_train.group_id.nunique(),
              "extra_train_by_source": extra_train.source.value_counts().to_dict()}
    print(f"\n  holdout {h}")
    for k, v in imp[h].items():
        print(f"    {k}: {v}")

print(f"\n  FARABI_FOLD_EXTRA_TRAIN_IMAGES = {imp['farabi']['extra_train_images']}")
print(f"  FARFUM_FOLD_EXTRA_TRAIN_IMAGES = {imp['farfum_rop']['extra_train_images']}")
print(f"  PLUS_FOLD_EXTRA_TRAIN_IMAGES   = {imp['plus']['extra_train_images']}")

print("\n  NOTE: the farabi holdout count is identical (1410) but its TRAINING population still")
print("        contains all 77 LOSO-only images. Unchanged holdout count does not mean unaffected.")

# ---- duplicate / group relationships -------------------------------------------------
print("\n" + "=" * 100)
print("RELATIONSHIPS OF THE 77")
print("=" * 100)
print(f"  same_group_as_a_canonical_image : {int(only.group_shared_with_canonical.sum())} / {len(only)}")
print(f"  group NOT in canonical at all   : {int((~only.group_shared_with_canonical).sum())}")
print(f"  cross-source duplicates         : cannot be established without pixel hashing (not run)")
print(f"  by label : {only.label.value_counts().to_dict()}")
print(f"  by geometry (from the legacy manifest, no geometry column present): n/a")

out = only.rename(columns={"image_path": "stable_image_id"})[
    ["stable_image_id", "source", "label", "group_id", "present_in_excluded_table",
     "group_shared_with_canonical", "reason"]]
out["pixel_sha256"] = ""
out["present_in_canonical"] = 0
out["present_in_historical_loso"] = 1
out["exclusion_reason_if_known"] = [
    "identity_linked_by_cross_patient_exact_duplicate" if b else "" for b in only.present_in_excluded_table]
out.to_csv(f"{PRIV}/loso_vs_canonical_forensics.csv", index=False)
print(f"\n  private table -> {PRIV}/loso_vs_canonical_forensics.csv  rows={len(out)}")

summary = {
    "CANONICAL_POPULATION_N": int(len(can)),
    "CANONICAL_UNIQUE_IMAGE_N": int(can.image_path.nunique()),
    "HISTORICAL_LOSO_POPULATION_N": int(len(loso)),
    "HISTORICAL_LOSO_UNIQUE_IMAGE_N": int(loso.image_path.nunique()),
    "LOSO_ONLY_UNIQUE_IMAGES": int(len(only)),
    "CANONICAL_ONLY_UNIQUE_IMAGES": int(len(cs - ls)),
    "PLUS_DIFFERENCE": int(loso_counts["plus"] - (can.source == "plus").sum()),
    "FARFUM_DIFFERENCE": int(loso_counts["farfum_rop"] - (can.source == "farfum_rop").sum()),
    "FARABI_DIFFERENCE": int(loso_counts["farabi"] - (can.source == "farabi").sum()),
    "HISTORICAL_LOSO_STATUS": "NONCANONICAL_AND_REQUIRES_RERUN",
    "DIVERGENCE_POINT": ("data/masks/mask_manifest_legacy_image_level_20260826.csv (8960 rows) "
                         "minus the 13 farabi rows absent from the LOSO artifacts, versus "
                         "data/splits/all.csv (8870 rows) built 2026-08-27, after the LOSO ran "
                         "on 2026-08-26"),
    "FARABI_FOLD_EXTRA_TRAIN_IMAGES": imp["farabi"]["extra_train_images"],
    "FARFUM_FOLD_EXTRA_TRAIN_IMAGES": imp["farfum_rop"]["extra_train_images"],
    "PLUS_FOLD_EXTRA_TRAIN_IMAGES": imp["plus"]["extra_train_images"],
    "RERUN_REQUIRED": "YES",
    "classification_counts": only.reason.value_counts().to_dict(),
    "fold_impact": imp,
}
json.dump(summary, open(f"{PRIV}/task2_summary.json", "w"), indent=2, default=str)
print("\n" + "=" * 100)
for k, v in summary.items():
    if k != "fold_impact":
        print(f"  {k}: {v}")
