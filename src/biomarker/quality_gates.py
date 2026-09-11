"""Automated Phase 4 segmentation and biomarker quality gates.

Writes normalized, resolution-safe biomarker features plus machine-readable
quality reports. Clinical segmentation validity still requires expert masks or
blinded clinician review and is never inferred from these automated checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.classify.branch_a_tabular import META_COLS
from src.utils.common import (
    build_provenance,
    ensure_dirs,
    load_config,
    sha256_file,
)


PIXEL_AREA_COLUMNS = ["vessel_pixels", "area"]
PIXEL_LENGTH_COLUMNS = ["overall_length", "singularity_length"]
COUNT_COLUMNS = ["n_startpoints", "n_endpoints", "n_intersections"]
RELEASE_DROP_COLUMNS = [
    "vessel_pixels",
    "area",
    "overall_length",
    "n_startpoints",
    "n_endpoints",
    "n_intersections",
]


def _source_predictability(
    frame: pd.DataFrame,
    feature_columns: list[str],
    groups: np.ndarray,
    seed: int,
) -> dict[str, float]:
    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = frame["source"].astype(str).values
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    predictions = cross_val_predict(model, X, y, groups=groups, cv=folds)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "chance_accuracy": float(pd.Series(y).value_counts(normalize=True).max()),
        "n_features": len(feature_columns),
    }


def _source_macro_auc(
    frame: pd.DataFrame,
    feature_columns: list[str],
    groups: np.ndarray,
    seed: int,
) -> float:
    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["source"].astype(str))
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    )
    probabilities = cross_val_predict(
        model, X, y, groups=groups, cv=folds, method="predict_proba"
    )
    return float(
        roc_auc_score(
            y,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=np.arange(len(encoder.classes_)),
        )
    )


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-empty-masks",
        action="store_true",
        help="report but do not fail when a mask contains zero vessel pixels",
    )
    args = parser.parse_args()

    splits = pd.read_csv(cfg["paths"]["splits_dir"] / "all.csv")
    features_path = cfg["paths"]["features_dir"] / "biomarker_features.csv"
    masks_path = cfg["paths"]["masks_dir"] / "mask_manifest.csv"
    features = pd.read_csv(features_path)
    masks = pd.read_csv(masks_path)
    if not (
        set(splits["image_path"])
        == set(features["image_path"])
        == set(masks["image_path"])
    ):
        raise SystemExit("Split, feature, and mask image sets differ.")
    if any(frame["image_path"].duplicated().any() for frame in [splits, features, masks]):
        raise SystemExit("Duplicate image_path found in Phase 4 inputs.")

    dimensions = []
    invalid_masks = []
    for row in masks.itertuples(index=False):
        image_path, mask_path = Path(row.image_path), Path(row.mask_path)
        if not image_path.exists() or not mask_path.exists():
            invalid_masks.append(
                {"image_path": str(image_path), "mask_path": str(mask_path), "issue": "missing"}
            )
            continue
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            mask_array = np.asarray(mask.convert("L"))
            values = set(np.unique(mask_array).tolist())
            if image.size != mask.size:
                invalid_masks.append(
                    {
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "issue": "dimension_mismatch",
                    }
                )
            if not values.issubset({0, 255}):
                invalid_masks.append(
                    {
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "issue": f"non_binary_values:{sorted(values)[:10]}",
                    }
                )
            vessel_pixels = int((mask_array > 0).sum())
            dimensions.append(
                {
                    "image_path": str(image_path),
                    "width": image.width,
                    "height": image.height,
                    "image_area": image.width * image.height,
                    "image_diagonal": float(np.hypot(image.width, image.height)),
                    "mask_vessel_pixels": vessel_pixels,
                    "mask_density_recomputed": vessel_pixels / (image.width * image.height),
                }
            )

    dim = pd.DataFrame(dimensions)
    frame = features.merge(dim, on="image_path", how="inner", validate="one_to_one")
    frame["vessel_pixels_fraction"] = frame["vessel_pixels"] / frame["image_area"]
    frame["area_fraction"] = frame["area"] / frame["image_area"]
    for column in PIXEL_LENGTH_COLUMNS:
        frame[f"{column}_per_diagonal"] = frame[column] / frame["image_diagonal"]
    for column in COUNT_COLUMNS:
        frame[f"{column}_per_megapixel"] = frame[column] / (
            frame["image_area"] / 1_000_000.0
        )

    normalized_path = cfg["paths"]["features_dir"] / "biomarker_features_normalized.csv"
    helper_columns = {
        "width",
        "height",
        "image_area",
        "image_diagonal",
        "mask_vessel_pixels",
        "mask_density_recomputed",
    }
    normalized_columns = [
        column
        for column in frame.columns
        if column not in set(PIXEL_AREA_COLUMNS + PIXEL_LENGTH_COLUMNS + COUNT_COLUMNS)
        and column not in helper_columns
    ]
    frame[normalized_columns].to_csv(normalized_path, index=False)
    release_safe_path = (
        cfg["paths"]["features_dir"] / "biomarker_features_phase4.csv"
    )
    release_safe_columns = [
        column for column in features.columns if column not in RELEASE_DROP_COLUMNS
    ]
    features[release_safe_columns].to_csv(release_safe_path, index=False)

    feature_columns = [column for column in features.columns if column not in META_COLS]
    normalized_feature_columns = [
        column
        for column in normalized_columns
        if column not in META_COLS
    ]
    release_safe_feature_columns = [
        column
        for column in release_safe_columns
        if column not in META_COLS
    ]
    numeric = features[feature_columns].replace([np.inf, -np.inf], np.nan)
    missing = numeric.isna().sum()
    outlier_rows = []
    for column in feature_columns:
        series = numeric[column].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 3 * iqr, q3 + 3 * iqr
        outlier_rows.append(
            {
                "feature": column,
                "missing": int(missing[column]),
                "non_finite": int((~np.isfinite(numeric[column].fillna(0))).sum()),
                "extreme_outliers_3iqr": int(((series < low) | (series > high)).sum()),
                "min": float(series.min()),
                "median": float(series.median()),
                "max": float(series.max()),
            }
        )
    results_dir = cfg["paths"]["results_dir"]
    pd.DataFrame(outlier_rows).to_csv(
        results_dir / "phase4_feature_quality.csv", index=False
    )
    frame.groupby(["source", "label"])[
        ["vessel_density", "tortuosity_index", "fractal_d0", "area_fraction"]
    ].agg(["count", "mean", "std", "min", "median", "max"]).to_csv(
        results_dir / "phase4_source_class_summary.csv"
    )
    review_rows = []
    for (source, label), group in frame.groupby(["source", "label"]):
        for quantile, name in [(0.05, "low"), (0.50, "median"), (0.95, "high")]:
            target = group["vessel_density"].quantile(quantile)
            row = group.loc[(group["vessel_density"] - target).abs().idxmin()]
            review_rows.append(
                {
                    "source": source,
                    "label": int(label),
                    "stratum": name,
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "vessel_density": row["vessel_density"],
                }
            )
        random_row = group.sample(1, random_state=cfg["seed"]).iloc[0]
        review_rows.append(
            {
                "source": source,
                "label": int(label),
                "stratum": "random",
                "image_path": random_row["image_path"],
                "mask_path": random_row["mask_path"],
                "vessel_density": random_row["vessel_density"],
            }
        )
    pd.DataFrame(review_rows).drop_duplicates("image_path").to_csv(
        results_dir / "phase4_mask_review_manifest.csv", index=False
    )
    review_queue = []
    flagged = frame[
        (frame["vessel_density"] < 0.01) | (frame["vessel_density"] > 0.15)
    ]
    for row in flagged.itertuples(index=False):
        review_queue.append(
            {
                "image_path": row.image_path,
                "mask_path": row.mask_path,
                "source": row.source,
                "label": int(row.label),
                "reason": (
                    "coverage_below_1pct"
                    if row.vessel_density < 0.01
                    else "coverage_above_15pct"
                ),
                "vessel_density": row.vessel_density,
                "decision": "",
                "reviewer": "",
                "notes": "",
            }
        )
    for (source, label), group in frame.groupby(["source", "label"]):
        ranked = group["vessel_density"].rank(method="first")
        terciles = pd.qcut(ranked, 3, labels=["low", "mid", "high"])
        for tercile in ["low", "mid", "high"]:
            stratum = group[terciles == tercile]
            for row in stratum.sample(
                min(5, len(stratum)), random_state=cfg["seed"]
            ).itertuples(index=False):
                review_queue.append(
                    {
                        "image_path": row.image_path,
                        "mask_path": row.mask_path,
                        "source": source,
                        "label": int(label),
                        "reason": f"stratified_{tercile}_coverage",
                        "vessel_density": row.vessel_density,
                        "decision": "",
                        "reviewer": "",
                        "notes": "",
                    }
                )
    review_queue_frame = pd.DataFrame(review_queue).drop_duplicates("image_path")
    review_queue_path = results_dir / "phase4_mask_review_queue.csv"
    if review_queue_path.exists():
        previous = pd.read_csv(review_queue_path).set_index("image_path")
        for column in ["decision", "reviewer", "notes"]:
            if column in previous.columns:
                review_queue_frame[column] = review_queue_frame["image_path"].map(
                    previous[column]
                ).fillna(review_queue_frame[column])
    review_queue_frame.to_csv(
        review_queue_path, index=False
    )

    empty = frame[frame["mask_vessel_pixels"] == 0][
        ["image_path", "mask_path", "source", "label", "split", "group_id"]
    ]
    empty.to_csv(results_dir / "phase4_empty_masks.csv", index=False)
    source_raw = _source_predictability(
        frame, feature_columns, frame["group_id"].values, cfg["seed"]
    )
    source_normalized = _source_predictability(
        frame, normalized_feature_columns, frame["group_id"].values, cfg["seed"]
    )
    source_auc_raw = _source_macro_auc(
        frame, feature_columns, frame["group_id"].values, cfg["seed"]
    )
    source_auc_release_safe = _source_macro_auc(
        frame, release_safe_feature_columns, frame["group_id"].values, cfg["seed"]
    )
    source_auc_reduction = source_auc_raw - source_auc_release_safe
    source_reduction_pass = source_auc_reduction >= 0.05
    valid_decisions = {"accept", "exclude", "rerun"}
    unresolved_reviews = int(
        (~review_queue_frame["decision"].fillna("").isin(valid_decisions)).sum()
    )
    density_error = float(
        np.max(
            np.abs(
                frame["vessel_density"].values
                - frame["mask_density_recomputed"].values
            )
        )
    )
    report = {
        "status": (
            "AUTOMATED_GATES_PASS"
            if (
                not invalid_masks
                and (args.allow_empty_masks or empty.empty)
                and source_reduction_pass
                and unresolved_reviews == 0
            )
            else "AUTOMATED_GATES_FAIL"
        ),
        "rows": len(frame),
        "invalid_mask_records": len(invalid_masks),
        "empty_masks": len(empty),
        "max_vessel_density_recompute_error": density_error,
        "feature_columns_raw": len(feature_columns),
        "feature_columns_resolution_safe": len(normalized_feature_columns),
        "feature_columns_release_safe": len(release_safe_feature_columns),
        "source_predictability_raw": source_raw,
        "source_predictability_resolution_safe": source_normalized,
        "source_confounding_gate": {
            "pass": source_reduction_pass,
            "raw_macro_ovr_auc": source_auc_raw,
            "release_safe_macro_ovr_auc": source_auc_release_safe,
            "minimum_required_auc_reduction": 0.05,
            "observed_auc_reduction": source_auc_reduction,
            "residual_auc_warning": source_auc_release_safe > 0.80,
        },
        "normalized_features_path": str(normalized_path),
        "release_safe_features_path": str(release_safe_path),
        "mask_review_queue": {
            "rows": len(review_queue_frame),
            "unresolved": unresolved_reviews,
            "allowed_decisions": sorted(valid_decisions),
        },
        "clinical_validation": {
            "status": "PENDING",
            "reason": (
                "No target-domain expert vessel masks or completed blinded "
                "clinician review were available; Dice/IoU cannot be claimed."
            ),
        },
        "provenance": {
            "current": build_provenance(cfg),
            "segmentation_checkpoint_sha256": sha256_file(
                Path(cfg["segmentation"]["weight_path"])
                if Path(cfg["segmentation"]["weight_path"]).is_absolute()
                else cfg["_root"] / cfg["segmentation"]["weight_path"]
            ),
            "feature_table_sha256": sha256_file(features_path),
            "mask_manifest_sha256": sha256_file(masks_path),
            "requirements_lock_sha256": sha256_file(
                cfg["_root"] / "requirements-lock.txt"
            ),
            "generation_time_binding": "UNAVAILABLE_RETROSPECTIVE_HASHES_ONLY",
        },
        "invalid_masks": invalid_masks,
    }
    with (results_dir / "phase4_quality_gates.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if report["status"] != "AUTOMATED_GATES_PASS":
        raise SystemExit("Phase 4 automated quality gates failed; inspect reports.")


if __name__ == "__main__":
    main()
