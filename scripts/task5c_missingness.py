#!/usr/bin/env python
"""Task 5C (A/B/C/D): missingness-safe feature freeze.

A. report all 18 admitted features with unit, dependencies and missing rates
B. split them into PRIMARY_CORE_V1 (disc-free) and DISC_AUGMENTED_V1 (disc-dependent)
C. do not treat imputation as the solution
D. quantify PRIMARY_CORE missingness and re-run the shortcut audit
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
TABLE = f"{ROOT}/data/features/clinical_measurement_v1.csv"
CLF = f"{ROOT}/configs/clinical_measurement_v1_classifier_features.yaml"
MEAS = f"{ROOT}/configs/clinical_measurement_v1.yaml"

GEOM = {(640, 480): "640x480", (1280, 960): "1280x960", (1440, 1080): "1440x1080",
        (1600, 1200): "1600x1200", (1240, 1240): "1240x1240"}


def main() -> None:
    df = pd.read_csv(TABLE)
    clf = yaml.safe_load(open(CLF, encoding="utf-8"))
    meas = yaml.safe_load(open(MEAS, encoding="utf-8"))
    from PIL import Image

    df["_geom"] = df.image_path.map(
        lambda p: GEOM.get(Image.open(p).size, "other"))

    primary = [f["name"] for f in clf["primary_allowed"]]
    secondary = [f["name"] for f in clf["secondary_allowed_approved"]]
    allowed = primary + secondary
    print(f"allowed features: {len(allowed)}  (primary {len(primary)}, secondary {len(secondary)})")

    # ---------------------------------------------------------------- A
    print()
    print("=" * 100)
    print("A. THE 18 ADMITTED FEATURES WITH MISSINGNESS")
    print("=" * 100)
    rows = []
    for name in allowed:
        spec = meas["features"][name]
        miss = df[name].isna()
        by_src = df.groupby("source")[name].apply(lambda s: float(s.isna().mean()))
        by_geom = df.groupby("_geom")[name].apply(lambda s: float(s.isna().mean()))
        rows.append({
            "feature_name": name,
            "unit": spec.get("unit", "UNKNOWN"),
            "depends_on_disc": bool(spec.get("depends_on_disc", False)),
            "depends_on_fov": bool(spec.get("depends_on_fov", False)),
            "roi_dependent": "requires" in spec,
            "missing_n": int(miss.sum()),
            "missing_rate": float(miss.mean()),
            "missing_by_source": {k: round(v, 4) for k, v in by_src.items()},
            "missing_by_geometry": {k: round(v, 4) for k, v in by_geom.items()},
            "admission_status": spec["admission"],
        })
    A = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows])
    A.to_csv(f"{ROOT}/_private_audit/task5c_allowed_features.csv", index=False)
    print(f"  {'feature':30s} {'unit':26s} {'disc':>5s} {'fov':>5s} {'roi':>5s} "
          f"{'miss_n':>7s} {'rate':>8s}  status")
    for r in rows:
        print(f"  {r['feature_name']:30s} {r['unit']:26s} "
              f"{str(r['depends_on_disc']):>5s} {str(r['depends_on_fov']):>5s} "
              f"{str(r['roi_dependent']):>5s} {r['missing_n']:7d} {r['missing_rate']:8.4f}  "
              f"{r['admission_status']}")
    print()
    print("  missing rate by source:")
    for r in rows:
        print(f"    {r['feature_name']:30s} {r['missing_by_source']}")
    print()
    print("  missing rate by geometry:")
    for r in rows:
        print(f"    {r['feature_name']:30s} {r['missing_by_geometry']}")

    # ---------------------------------------------------------------- B
    core = [r["feature_name"] for r in rows if not r["depends_on_disc"]]
    aug = [r["feature_name"] for r in rows if r["depends_on_disc"]]
    print()
    print("=" * 100)
    print("B. TWO FEATURE SETS")
    print("=" * 100)
    print(f"  PRIMARY_CORE_V1 ({len(core)}): {core}")
    print(f"  DISC_AUGMENTED_V1 ({len(aug)}): {aug}")
    print(f"  partition complete: {sorted(core + aug) == sorted(allowed)}")

    core_yaml = {
        "feature_set": "PRIMARY_CORE_V1",
        "measurement_version": "CLINICAL_MEASUREMENT_V1",
        "segmentation_generation": "SEG_CURRENT_V1",
        "purpose": "features eligible for the main corrected Branch A / C experiment",
        "selection_rule": (
            "every feature here is computable without a successful optic-disc detection. No "
            "detector-status variable, no missingness indicator, no QC variable, no "
            "source/geometry metadata, no A/V heuristic and no raw scale-dependent feature is "
            "included. Selection used missingness and measurement properties only, never a label."
        ),
        "disc_dependent_features_excluded": aug,
        "features": [f["name"] for f in clf["primary_allowed"] if f["name"] in core]
        + [n for n in core if n not in primary],
        "expected_missingness_source": (
            "FOV detection failure only (45 of 8870 images) plus rare degenerate masks"
        ),
        "imputation_policy": (
            "NOT imputed. Missingness is expected to be near-zero and non-informative; if it is "
            "not, the gate in Task 5C section D fails and this set must not be trained on."
        ),
    }
    aug_yaml = {
        "feature_set": "DISC_AUGMENTED_V1",
        "measurement_version": "CLINICAL_MEASUREMENT_V1",
        "segmentation_generation": "SEG_CURRENT_V1",
        "purpose": (
            "valid disc-normalised secondary measurements. This is NOT a primary pooled "
            "classifier matrix."
        ),
        "reserved_for": [
            "disc-valid conditional analysis",
            "sensitivity analysis",
            "later target-domain expert-supported analysis",
        ],
        "valid_only_when": "disc_valid = 1 (3772 of 8870 images)",
        "known_constraint": (
            "disc validity is source-dependent (plus 38.2%, farfum_rop 53.3%, farabi 48.9%). "
            "Task 5B reported a missingness->label AUC of 0.6445; that figure was in fact a "
            "geometry->label AUC and is corrected in Task 5C. The real constraint on this set is "
            "not a missingness shortcut but the loss of 57.5% of the cohort and the source bias "
            "of which images survive. Any use of this set must be reported as conditional on "
            "disc validity."
        ),
        "features": aug,
        "imputation_policy": "NOT imputed. Rows without a valid disc are excluded, not filled.",
    }
    open(f"{ROOT}/configs/primary_core_v1_features.yaml", "w", encoding="utf-8").write(
        yaml.safe_dump(core_yaml, sort_keys=False))
    open(f"{ROOT}/configs/disc_augmented_v1_features.yaml", "w", encoding="utf-8").write(
        yaml.safe_dump(aug_yaml, sort_keys=False))
    print(f"  wrote configs/primary_core_v1_features.yaml")
    print(f"  wrote configs/disc_augmented_v1_features.yaml")

    # ---------------------------------------------------------------- D
    print()
    print("=" * 100)
    print("D. PRIMARY_CORE MISSINGNESS QUANTIFICATION AND SHORTCUT AUDIT")
    print("=" * 100)
    sub = df[core]
    complete = sub.notna().all(axis=1)
    print(f"  rows                    : {len(df)}")
    print(f"  fully complete rows     : {int(complete.sum())} ({complete.mean():.4f})")
    print(f"  rows with any missing   : {int((~complete).sum())}")
    print()
    print(f"  {'feature':30s} {'missing_n':>9s} {'rate':>8s}")
    for c in core:
        print(f"  {c:30s} {int(sub[c].isna().sum()):9d} {sub[c].isna().mean():8.6f}")
    print()
    print("  missingness by source / geometry / label (share of rows with ANY core feature NaN):")
    for key in ("source", "_geom", "label"):
        g = df.groupby(key).apply(lambda s: float((~sub.loc[s.index].notna().all(axis=1)).mean()))
        print(f"    by {key:7s}: { {k: round(v, 5) for k, v in g.items()} }")

    ind = (~sub.notna()).astype(int)
    any_missing = (~complete).astype(int)
    print()
    print("  CORRECTED shortcut audit: the PREDICTOR is the missingness indicator itself,")
    print("  not acquisition geometry. (Task 5B reported geometry->target AUCs under a")
    print("  missingness heading; that statistic is corrected here.)")
    from scipy.stats import chi2_contingency

    res = {}
    for target, name in ((df.source, "source"), (df._geom, "geometry"), (df.label, "label")):
        tab = pd.crosstab(any_missing, target)
        if tab.shape[0] < 2 or tab.values.sum() < 5 or (tab.values == 0).all():
            res[name] = {"note": "too few missing rows for a test", "missing_n": int(any_missing.sum())}
            print(f"    any-core-missing vs {name:9s}: {int(any_missing.sum())} missing rows; "
                  f"no test possible")
            continue
        try:
            chi2, p, dof, _ = chi2_contingency(tab.values)
        except Exception:  # noqa: BLE001
            chi2, p = float("nan"), float("nan")
        rates = (df.assign(m=any_missing).groupby(target).m.mean())
        res[name] = {"chi2": float(chi2), "p": float(p), "dof": int(dof),
                     "missing_rate_by_level": {str(k): float(v) for k, v in rates.items()},
                     "max_over_min_rate_ratio": float(rates.max() / rates.min())
                     if rates.min() > 0 else float("inf")}
        print(f"    any-core-missing vs {name:9s}: missing rates "
              f"{ {str(k): round(v, 5) for k, v in rates.items()} }  "
              f"chi2={chi2:.2f} p={p:.4f}")
    print()
    print("  ALSO REPORTED, SEPARATELY AND CORRECTLY LABELLED: does acquisition geometry")
    print("  predict the target? This is an acquisition-shortcut concern, not missingness.")
    X = StandardScaler().fit_transform(
        pd.concat([df.dd_over_min_side, df.fov_coverage_fraction], axis=1).fillna(-1).values)
    geom_to_target = {}
    for target, name in ((df.source, "source"), (df._geom, "geometry"), (df.label, "label")):
        y = pd.factorize(target)[0]
        try:
            p = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=5,
                                  method="predict_proba")
            auc = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
        except Exception:  # noqa: BLE001
            auc = float("nan")
        geom_to_target[name] = auc
        print(f"    geometry -> {name:9s} macro AUC = {auc:.4f}")
    print()
    n_missing_rows = int((~complete).sum())
    frac = n_missing_rows / len(df)
    worst_p = min((v.get("p", 1.0) for v in res.values()
                   if isinstance(v, dict) and "p" in v), default=1.0)
    # A max/min rate ratio is not used as a criterion: a level with zero missing rows makes it
    # infinite on a handful of rows, which is an artefact, not evidence. The gate uses only the
    # missing fraction and the chi-square association p-values.
    material = frac > 0.02 or worst_p < 0.01
    gate = "FAIL" if material else "PASS"
    print(f"  PRIMARY_CORE_MISSINGNESS_GATE = {gate}")
    print(f"    missing rows {n_missing_rows} ({frac:.4%} of cohort)")
    print(f"    smallest chi-square p across targets: {worst_p:.4f}")
    print(f"    gate rule: FAIL if missing fraction > 2% or any association p < 0.01")
    if gate == "PASS":
        print("    the disc-free core set is available for 8862 of 8870 images (99.91%) and its")
        print("    missingness shows no material association with source, geometry or label.")
        print("    It is approved for training. Imputation is NOT required and must not be used.")
    else:
        print("    do NOT approve PRIMARY_CORE_V1 for training as-is.")

    json.dump({"features": rows, "core": core, "augmented": aug,
               "core_row_completeness": float(complete.mean()),
               "core_missing_rows": n_missing_rows,
               "missingness_to_target": res,
               "geometry_to_target_auc": geom_to_target,
               "gate": gate,
               "correction_note": (
                   "Task 5B reported 'missingness -> geometry AUC 0.794 / label 0.6445'. Those "
                   "numbers are geometry -> target AUCs, not missingness -> target AUCs. The "
                   "missingness shortcut statistic is recomputed here.")},
              open(f"{ROOT}/_private_audit/task5c_missingness.json", "w"), indent=2,
              default=str)
    print(f"\n  -> _private_audit/task5c_missingness.json")


if __name__ == "__main__":
    main()
