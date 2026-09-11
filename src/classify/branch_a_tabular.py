"""Branch A: train tabular classifiers on the PVBM biomarker table.

Train on train split; pick model + Plus threshold on val; report test with
bootstrap CI and 3-class metrics. Best model chosen by val F1 (Plus-OvR), not
test AUC — avoids RF-high-AUC / zero-sensitivity trap.

Usage:
    python -m src.classify.branch_a_tabular
    python -m src.classify.branch_a_tabular --force
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.branch_eval import scores_from_classifier, train_select_eval_tabular
from src.utils.common import (
    build_provenance,
    ensure_dirs,
    load_config,
    plus_class_index,
    provenance_matches_current,
    set_seed,
)

META_COLS = {
    "image_path",
    "mask_path",
    "label",
    "split",
    "source",
    "group_id",
    "patient_id",
    "exam_id",
    "identity_level",
}


def build_models(seed: int, names: list[str]) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    models: dict = {}
    if "logreg" in names:
        models["logreg"] = LogisticRegression(max_iter=1000, class_weight="balanced")
    if "random_forest" in names:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=seed
        )
    if "mlp" in names:
        models["mlp"] = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed)
    if "lightgbm" in names:
        try:
            from lightgbm import LGBMClassifier

            models["lightgbm"] = LGBMClassifier(
                n_estimators=500, learning_rate=0.03, class_weight="balanced", random_state=seed
            )
        except ImportError:
            print("[warn] lightgbm not installed, skipping")
    if "xgboost" in names:
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                eval_metric="logloss",
                random_state=seed,
            )
        except ImportError:
            print("[warn] xgboost not installed, skipping")
    return models


def prepare_matrix(df: pd.DataFrame, split: pd.Series | None = None):
    """Feature matrix with NaNs imputed using train-split medians only (no test peeking)."""
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.std(numeric_only=True).fillna(0) > 0]
    if split is None:
        X = X.fillna(X.median(numeric_only=True))
    else:
        med = X[split.values == "train"].median(numeric_only=True)
        med = med.fillna(X.median(numeric_only=True))
        X = X.fillna(med)
    return X, list(X.columns)


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg["seed"])
    ev = cfg.get("eval", {})

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if results exist")
    args = ap.parse_args()

    res_dir = cfg["paths"]["results_dir"]
    out_json = res_dir / "branch_a_results.json"
    if out_json.exists() and not args.force:
        existing = json.loads(out_json.read_text())
        if provenance_matches_current(existing, cfg):
            print(f"[skip] current Branch A results exist -> {out_json}")
            return
        print(f"[stale] ignoring Branch A results from different split/config: {out_json}")

    feats_path = cfg["paths"]["features_dir"] / "biomarker_features.csv"
    if not feats_path.exists():
        raise SystemExit(f"Feature table not found: {feats_path}. Run src.biomarker.extract_pvbm first.")
    df = pd.read_csv(feats_path)

    y = df["label"].astype(int).values
    split = df["split"].values

    candidate_cols = [column for column in df.columns if column not in META_COLS]
    numeric = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    train_mask = split == "train"
    train_numeric = numeric.loc[train_mask]
    feat_cols = [
        column
        for column in numeric.columns
        if not train_numeric[column].isna().all()
        and float(train_numeric[column].std(skipna=True) or 0.0) > 0.0
    ]
    medians = train_numeric[feat_cols].median().fillna(0.0)
    X = numeric[feat_cols].fillna(medians)

    from sklearn.preprocessing import StandardScaler

    val_mask = split == "val"
    test_mask = split == "test"
    scaler = StandardScaler().fit(X.values[train_mask])
    X_train = scaler.transform(X.values[train_mask])
    X_val = scaler.transform(X.values[val_mask])
    X_test = scaler.transform(X.values[test_mask])
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    metadata_columns = [
        column
        for column in [
            "image_path",
            "label",
            "source",
            "group_id",
            "patient_id",
            "exam_id",
            "identity_level",
        ]
        if column in df.columns
    ]
    test_meta = df.loc[test_mask, metadata_columns]

    plus_idx = plus_class_index(cfg)
    class_names = cfg["data"]["class_names"]
    models = build_models(cfg["seed"], cfg["classify_a"]["models"])

    summary, best_name, best_scores, best_model = train_select_eval_tabular(
        models,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        test_meta,
        plus_idx,
        class_names,
        threshold_method=ev.get("threshold_method", "youden"),
        model_select=ev.get("model_select", "val_f1_plus"),
        n_boot=ev.get("bootstrap_n", 1000),
        seed=cfg["seed"],
    )
    summary["n_features"] = len(feat_cols)
    summary["preprocessing"] = {
        "feature_selection": "training_split_only",
        "imputation": "training_median",
        "scaling": "StandardScaler fit on training only",
    }
    summary["provenance"] = build_provenance(cfg)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if best_scores is not None:
        pred_df = df.loc[test_mask, metadata_columns].copy()
        pred_df["p_plus_branch_a"] = best_scores
        pred_df.to_csv(res_dir / "branch_a_test_preds.csv", index=False)

    if best_model is not None:
        import joblib

        joblib.dump(
            {
                "model": best_model,
                "feature_columns": feat_cols,
                "training_medians": medians.to_dict(),
                "scaler": scaler,
                "plus_idx": plus_idx,
            },
            res_dir / "branch_a_model.joblib",
        )
        val_scores_best, _ = scores_from_classifier(best_model, X_val, plus_idx)
        val_cols = metadata_columns
        val_df = df.loc[val_mask, val_cols].copy()
        val_df["p_plus_branch_a"] = val_scores_best
        val_df.to_csv(res_dir / "branch_a_val_preds.csv", index=False)

    _shap_plot(models, best_name, X_train, feat_cols, plus_idx, res_dir)
    best_test = summary["metrics"][best_name]["test"]["plus_ovr"]
    print(f"[done] best={best_name} test_AUC={best_test['auc']:.3f} sens={best_test['sensitivity']:.3f} -> {res_dir}")


def _shap_plot(models, best_name, X_tr, feat_cols, plus_idx: int, res_dir: Path) -> None:
    if best_name not in {"random_forest", "lightgbm", "xgboost"}:
        return
    output_path = res_dir / "branch_a_shap.png"
    output_path.unlink(missing_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        explainer = shap.TreeExplainer(models[best_name])
        sv = explainer.shap_values(X_tr)
        classes = np.asarray(models[best_name].classes_)
        positions = np.flatnonzero(classes == plus_idx)
        if len(positions) != 1:
            raise ValueError(f"Plus class {plus_idx} missing from {classes.tolist()}")
        class_position = int(positions[0])
        if isinstance(sv, list):
            sv = sv[class_position]
        elif getattr(sv, "ndim", 0) == 3:
            sv = sv[:, :, class_position]
        if np.shape(sv) != (len(X_tr), len(feat_cols)):
            raise ValueError(
                f"Unexpected Plus SHAP shape {np.shape(sv)}; "
                f"expected {(len(X_tr), len(feat_cols))}"
            )
        shap.summary_plot(sv, features=X_tr, feature_names=feat_cols, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[shap] saved Plus-class explanation -> {output_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] SHAP plot skipped: {e}")


if __name__ == "__main__":
    main()
