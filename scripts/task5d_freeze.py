#!/usr/bin/env python
"""Task 5D (J/K/L/M): final feature sets, governed missingness, frozen table, integrity."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path("/Users/moniaz/niki")
SRC = ROOT / "data/features/clinical_measurement_v1.csv"
DST = ROOT / "data/features/final_biomarkers_v1.csv"

PRIMARY = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
SECONDARY = ["tort_geodesic_median", "tort_geodesic_p90", "tort_geodesic_top3_mean"]
DISC_COND = ["width_p50_dd", "width_p90_dd", "width_mean_dd", "width_ann_p50_dd",
             "width_ann_p90_dd", "width_ann_mean_dd", "vessel_density_fov_ring_2_3dd",
             "vessel_density_fov_ring_3_6dd"]
EXPLORATORY = ["n_branches"]
FORBIDDEN = ["width_shape_p90_over_p50"]
TEN = PRIMARY + SECONDARY + EXPLORATORY + FORBIDDEN


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> None:
    # ---- verify the 10 names from the YAML, do not copy blindly ----
    core = yaml.safe_load(open(ROOT / "configs/primary_core_v1_features.yaml", encoding="utf-8"))
    yaml_list = core["features"]
    print("=" * 100)
    print("VERIFY PRIMARY_CORE_V1 NAMES AGAINST THE YAML")
    print("=" * 100)
    print(f"  yaml n = {len(yaml_list)}")
    print(f"  matches the Task 5D list exactly: {sorted(yaml_list) == sorted(TEN)}")
    if sorted(yaml_list) != sorted(TEN):
        print(f"  yaml: {sorted(yaml_list)}")
        print(f"  task: {sorted(TEN)}")
        raise SystemExit("name mismatch")

    # ---- final feature-set contracts ----
    sets = {
        "final_primary_core_v1_features.yaml": {
            "feature_set": "FINAL_PRIMARY_CORE_V1",
            "status": "FROZEN before any disease-model performance was seen",
            "measurement_version": "CLINICAL_MEASUREMENT_V1",
            "segmentation_generation": "SEG_CURRENT_V1",
            "purpose": "the feature set for the main Branch A and Branch C biomarker experiment",
            "n_features": len(PRIMARY),
            "features": PRIMARY,
            "selection_rule": (
                "disc-independent, no QC, no metadata, no exploratory and no forbidden feature. "
                "Every member has small or negligible systematic bias in BOTH cameras and "
                "ICC(2,1) >= 0.56 in the weaker camera. Selection used measurement evidence "
                "only; no label, AUC or feature importance was consulted."
            ),
            "external_evidence": {
                "vessel_density_fov": "RetCam ICC 0.872 [0.789,0.995] bias +0.0158 MAE 0.0192 rho 0.942; Neo ICC 0.774 bias +0.0001 MAE 0.0129 rho 0.751",
                "skel_density_fov": "RetCam ICC 0.933 [0.886,0.998] bias -0.0016 MAE 0.0016 rho 0.951; Neo ICC 0.547 bias -0.0050 MAE 0.0051 rho 0.816",
                "fractal_d0": "RetCam ICC 0.940 [0.897,0.999] bias -0.0183 MAE 0.0202 rho 0.932; Neo ICC 0.569 bias -0.0779 MAE 0.0786 rho 0.820",
                "fractal_d1": "RetCam ICC 0.923 [0.870,0.998] bias -0.0246 MAE 0.0260 rho 0.927; Neo ICC 0.578 bias -0.0817 MAE 0.0821 rho 0.830",
                "fractal_d2": "RetCam ICC 0.911 [0.851,0.997] bias -0.0273 MAE 0.0291 rho 0.924; Neo ICC 0.566 bias -0.0817 MAE 0.0825 rho 0.815",
            },
            "known_limitations": [
                "the fractal family is markedly weaker on Neo (ICC 0.57 vs 0.91-0.94) with a -0.08 bias",
                "all members inherit the 256x256 segmentation bottleneck and the geometry-dependent shape bias",
                "no clinical validation: HVDROPDB is development/benchmark data",
            ],
        },
        "final_secondary_core_v1_features.yaml": {
            "feature_set": "FINAL_SECONDARY_CORE_V1",
            "status": "FROZEN before any disease-model performance was seen",
            "measurement_version": "CLINICAL_MEASUREMENT_V1",
            "segmentation_generation": "SEG_CURRENT_V1",
            "purpose": (
                "predeclared sensitivity-analysis set. It adds technically valid, disc-independent "
                "features whose external agreement is VARIANCE_LIMITED. It must NOT be used to "
                "replace the primary result after inspecting test performance."
            ),
            "n_features": len(SECONDARY),
            "features": SECONDARY,
            "reason_not_primary": (
                "tortuosity has near-zero bias and the smallest absolute error in the whole "
                "panel, but its error variance equals or exceeds the between-image expert "
                "variance, so MAE/expert SD is 0.46-1.23. The feature is not wrong; it is noisy "
                "relative to its own spread, which makes it unsuitable as a primary predictor."
            ),
            "external_evidence": {
                "tort_geodesic_median": "RetCam ICC 0.258 bias +0.0012 MAE 0.0063 MAE/SD 0.594 err/between var 1.020; Neo ICC 0.218 bias +0.0051 MAE 0.0059 MAE/SD 1.232 err/between 1.743",
                "tort_geodesic_p90": "RetCam ICC 0.309 bias -0.0060 MAE 0.0271 MAE/SD 0.633 err/between 0.820; Neo ICC 0.248 bias +0.0131 MAE 0.0263 MAE/SD 0.796 err/between 1.190",
                "tort_geodesic_top3_mean": "RetCam ICC 0.207 bias -0.1157 MAE 0.1617 MAE/SD 0.458 err/between 0.817; Neo ICC 0.248 bias -0.0201 MAE 0.0680 MAE/SD 0.716 err/between 0.959",
            },
        },
        "disc_conditional_v1_features.yaml": {
            "feature_set": "DISC_CONDITIONAL_V1",
            "status": "FROZEN before any disease-model performance was seen",
            "measurement_version": "CLINICAL_MEASUREMENT_V1",
            "segmentation_generation": "SEG_CURRENT_V1",
            "purpose": (
                "the eight disc-dependent measurements, used only in a separately labelled "
                "conditional analysis."
            ),
            "n_features": len(DISC_COND),
            "features": DISC_COND,
            "valid_only_when": "disc_valid = 1 (3772 of 8870 project images, 42.5%)",
            "population_warning": (
                "do not compare this population directly with full-cohort primary performance "
                "without reporting the population difference: disc validity is source-dependent "
                "(plus 38.2%, farfum_rop 53.3%, farabi 48.9%)."
            ),
            "external_evidence_paired_retcam_n25": {
                "width_p50_dd": "SEG-ONLY vs GOLD ICC 0.341 bias +0.0328 MAE 0.0328",
                "width_p90_dd": "ICC 0.125 bias +0.1116 MAE 0.1116",
                "width_mean_dd": "ICC 0.278 bias +0.0455",
                "width_ann_p50_dd": "ICC 0.321 bias +0.0461",
                "width_ann_p90_dd": "ICC 0.047 bias +0.1705",
                "width_ann_mean_dd": "ICC 0.174 bias +0.0699",
                "vessel_density_fov_ring_2_3dd": "N=21 ICC 0.821 bias +0.0312 MAE 0.0317",
                "vessel_density_fov_ring_3_6dd": "N=21 ICC 0.773 bias +0.0286",
            },
            "external_evidence_note": (
                "25 verified paired RetCam images only. Every caliber feature carries a "
                "POSITIVE systematic bias of +0.03 to +0.17 DD: the automatic vessel mask "
                "over-measures caliber relative to expert. Neo has no paired reference."
            ),
        },
    }
    for name, payload in sets.items():
        (ROOT / "configs" / name).write_text(
            yaml.safe_dump(payload, sort_keys=False, width=100), encoding="utf-8")
        print(f"  wrote configs/{name}")

    print()
    print("=" * 100)
    print("H/I. FINAL MODEL-ADMISSION STATUS")
    print("=" * 100)
    for n in PRIMARY:
        print(f"  FINAL_PRIMARY    {n}")
    for n in SECONDARY:
        print(f"  FINAL_SECONDARY  {n}")
    for n in EXPLORATORY:
        print(f"  EXPLORATORY_ONLY {n}   (demoted: camera-signed bias +10.0 RetCam / -15.5 Neo)")
    for n in FORBIDDEN:
        print(f"  FORBIDDEN        {n}   (demoted: +0.334 bias in BOTH cameras, ICC 0.05/0.14)")

    # ---- L: final table ----
    T = pd.read_csv(SRC)
    T["final_table_version"] = "FINAL_BIOMARKERS_V1"
    T["final_segmentation_generation"] = "SEG_CURRENT_V1"
    T.to_csv(DST, index=False)
    tsha = sha(DST)
    print()
    print("=" * 100)
    print("L/M. FINAL TABLE AND PRE-TRAINING INTEGRITY")
    print("=" * 100)
    print(f"  {DST.name}: rows={len(T)} cols={T.shape[1]}  sha256={tsha}")
    splits = pd.read_csv(ROOT / "data/splits/all.csv")
    F = pd.read_csv(DST)
    checks = {
        "rows_8870": len(F) == 8870,
        "stable_ids_unique": F.image_path.nunique() == 8870,
        "split_membership_matches_canonical": set(F.image_path) == set(splits.image_path),
        "split_counts_match": F.split.value_counts().to_dict() == splits.split.value_counts().to_dict(),
        "source_counts_match": F.source.value_counts().to_dict() == splits.source.value_counts().to_dict(),
        "group_linkage_complete": bool(F.group_id.notna().all()),
        "no_duplicate_identity": F.image_path.duplicated().sum() == 0,
        "no_inf_predictors": not np.isinf(F[TEN].to_numpy(float)).any(),
        "all_FINAL_PRIMARY_present": set(PRIMARY) <= set(F.columns),
        "no_metadata_in_primary": not (set(PRIMARY) & {"image_path", "mask_path", "label",
                                                       "split", "source", "group_id",
                                                       "patient_id", "exam_id",
                                                       "identity_level"}),
        "no_qc_in_primary": not (set(PRIMARY) & {c for c in F.columns
                                                 if c.startswith(("disc_", "fov_", "roi_"))
                                                 or c in ("measurement_version",
                                                          "work_resolution")}),
        "no_disc_dependency_in_primary": all(
            not yaml.safe_load(open(ROOT / "configs/clinical_measurement_v1.yaml",
                                    encoding="utf-8"))["features"][n].get("depends_on_disc",
                                                                          False)
            for n in PRIMARY),
        "measurement_version_present": set(F.measurement_version.unique()) ==
        {"CLINICAL_MEASUREMENT_V1"},
    }
    for k, v in checks.items():
        print(f"  {k:42s} : {'PASS' if v else 'FAIL'}")
    prim_complete = F[PRIMARY].notna().all(axis=1)
    print()
    print(f"  FINAL_PRIMARY row completeness : {prim_complete.mean():.4f} "
          f"({int(prim_complete.sum())}/{len(F)})")
    print(f"  missing rows                   : {int((~prim_complete).sum())}")
    miss = F[~prim_complete]
    print(f"  NaN columns on those rows       : "
          f"{ {c: int(miss[c].isna().sum()) for c in PRIMARY if miss[c].isna().any()} }")
    print()
    print("  PREDECLARED MISSINGNESS POLICY (option 2 of section K):")
    print("    cause: PVBM MultifractalVBMs returns NaN for 8 masks; the module is correct and")
    print("    the values are genuinely undefined for those inputs, so no defect is fixed.")
    print("    policy: COMPLETE-CASE evaluation. Any analysis that includes a fractal feature")
    print("    has N = 8862 and must report 8862. No imputation, and the policy is not chosen")
    print("    by disease-model performance.")
    gate = "PASS" if all(checks.values()) and len(F) == 8870 else "FAIL"
    print()
    print(f"  PRETRAINING_FEATURE_FREEZE = {gate}")

    json.dump({"FINAL_SEGMENTATION_GENERATION": "SEG_CURRENT_V1",
               "FINAL_PRIMARY_FEATURE_N": len(PRIMARY), "FINAL_PRIMARY_FEATURES": PRIMARY,
               "FINAL_SECONDARY_FEATURE_N": len(SECONDARY),
               "FINAL_SECONDARY_FEATURES": SECONDARY,
               "DISC_CONDITIONAL_FEATURE_N": len(DISC_COND),
               "FINAL_PRIMARY_ROW_COMPLETENESS": float(prim_complete.mean()),
               "final_table_sha256": tsha, "integrity": checks, "gate": gate},
              open(ROOT / "_private_audit/task5d_freeze.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
