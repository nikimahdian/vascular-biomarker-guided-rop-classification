"""Grouped cross-validation stability report for Branch A.

Uses only canonical train+validation rows. Final test remains untouched.
Every fold fits feature selection, imputation, scaling, and model from its
training groups only. Thresholded F1 uses fixed 0.5; AUC is primary.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.classify.branch_a_tabular import META_COLS, build_models
from src.utils.branch_eval import scores_from_classifier
from src.utils.common import build_provenance, ensure_dirs, load_config, plus_class_index


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folds", type=int, default=cfg["classify_a"].get("cv_folds", 5)
    )
    parser.add_argument("--models", nargs="*", default=cfg["classify_a"]["models"])
    args = parser.parse_args()

    frame = pd.read_csv(
        cfg["paths"]["features_dir"] / "biomarker_features_normalized.csv"
    )
    frame = frame[frame["split"].isin(["train", "val"])].reset_index(drop=True)
    if "group_id" not in frame or frame["group_id"].isna().any():
        raise SystemExit("Grouped CV requires complete group_id.")
    y = frame["label"].astype(int).values
    plus_idx = plus_class_index(cfg)
    groups = frame["group_id"].values
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=cfg["seed"]
    )
    candidate_columns = [
        column for column in frame.columns if column not in META_COLS
    ]
    fold_rows = []
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(frame, y, groups), start=1
    ):
        assert not set(groups[train_index]) & set(groups[valid_index])
        numeric = frame[candidate_columns].apply(pd.to_numeric, errors="coerce")
        training = numeric.iloc[train_index]
        columns = [
            column
            for column in candidate_columns
            if not training[column].isna().all()
            and float(training[column].std(skipna=True) or 0.0) > 0.0
        ]
        medians = training[columns].median().fillna(0.0)
        scaler = StandardScaler().fit(training[columns].fillna(medians))
        X_train = scaler.transform(training[columns].fillna(medians))
        X_valid = scaler.transform(numeric.iloc[valid_index][columns].fillna(medians))
        for name, model in build_models(cfg["seed"] + fold, args.models).items():
            model.fit(X_train, y[train_index])
            scores, _ = scores_from_classifier(model, X_valid, plus_idx)
            binary = (y[valid_index] == plus_idx).astype(int)
            fold_rows.append(
                {
                    "fold": fold,
                    "model": name,
                    "auc_plus": float(roc_auc_score(binary, scores)),
                    "f1_plus_at_0_5": float(
                        f1_score(binary, scores >= 0.5, zero_division=0)
                    ),
                    "n_train": len(train_index),
                    "n_valid": len(valid_index),
                    "groups_train": len(set(groups[train_index])),
                    "groups_valid": len(set(groups[valid_index])),
                    "n_features": len(columns),
                }
            )

    folds = pd.DataFrame(fold_rows)
    results_dir = cfg["paths"]["results_dir"]
    folds.to_csv(results_dir / "branch_a_grouped_cv_folds.csv", index=False)
    summary = (
        folds.groupby("model")
        .agg(
            auc_mean=("auc_plus", "mean"),
            auc_sd=("auc_plus", "std"),
            f1_mean=("f1_plus_at_0_5", "mean"),
            f1_sd=("f1_plus_at_0_5", "std"),
        )
        .reset_index()
    )
    summary.to_csv(results_dir / "branch_a_grouped_cv_summary.csv", index=False)
    payload = {
        "protocol": (
            "StratifiedGroupKFold on train+validation only; fixed 0.5 F1; "
            "final test untouched; Farabi groups are exams"
        ),
        "folds": args.folds,
        "provenance": build_provenance(cfg),
        "summary": summary.to_dict("records"),
    }
    (results_dir / "branch_a_grouped_cv.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
