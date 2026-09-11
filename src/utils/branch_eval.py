"""Shared train/val/test evaluation for tabular branches (A and C)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.common import (
    evaluate_plus_ovr_bundle,
    plus_ovr_metrics,
    tune_plus_threshold,
)


def scores_from_classifier(clf, X, plus_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (P(Plus), multiclass argmax predictions)."""
    if hasattr(clf, "predict_proba"):
        proba = np.asarray(clf.predict_proba(X))
        classes = np.asarray(getattr(clf, "classes_", np.arange(proba.shape[1])))
        matches = np.flatnonzero(classes == plus_idx)
        if len(matches) != 1:
            raise ValueError(
                f"Plus class {plus_idx} not uniquely present in classifier classes "
                f"{classes.tolist()}."
            )
        predictions = classes[proba.argmax(axis=1)].astype(int)
        return proba[:, matches[0]], predictions
    dec = clf.decision_function(X)
    classes = np.asarray(getattr(clf, "classes_", []))
    if np.ndim(dec) != 1 or len(classes) != 2:
        raise ValueError("Classifier without predict_proba must expose binary decision scores.")
    matches = np.flatnonzero(classes == plus_idx)
    if len(matches) != 1:
        raise ValueError(f"Plus class {plus_idx} not present in {classes.tolist()}.")
    plus_scores = np.asarray(dec) if matches[0] == 1 else -np.asarray(dec)
    predictions = np.where(np.asarray(dec) >= 0, classes[1], classes[0]).astype(int)
    return plus_scores, predictions


def per_source_plus_metrics(
    meta_df: pd.DataFrame,
    scores: np.ndarray,
    plus_idx: int,
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Plus-OvR metrics broken down by data source on the test set."""
    out: dict[str, dict[str, float]] = {}
    if "source" not in meta_df.columns:
        return out
    for src in sorted(meta_df["source"].unique()):
        mask = meta_df["source"].values == src
        if mask.sum() == 0:
            continue
        m = plus_ovr_metrics(meta_df["label"].values[mask], scores[mask], plus_idx, threshold)
        out[str(src)] = {k: m[k] for k in ("auc", "sensitivity", "specificity", "f1", "accuracy") if k in m}
    return out


def train_select_eval_tabular(
    models: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    test_meta: pd.DataFrame,
    plus_idx: int,
    class_names: list[str],
    *,
    threshold_method: str = "youden",
    model_select: str = "val_f1_plus",
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[dict[str, Any], str | None, np.ndarray | None, Any]:
    """Fit on train, tune threshold on val, report test with bootstrap CI."""
    results: dict[str, Any] = {}
    best_name, best_select, best_scores, best_model, best_thr = (
        None,
        -1.0,
        None,
        None,
        0.5,
    )

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        val_scores, _ = scores_from_classifier(clf, X_val, plus_idx)
        thr = tune_plus_threshold(y_val, val_scores, plus_idx, method=threshold_method)
        val_m = plus_ovr_metrics(y_val, val_scores, plus_idx, threshold=thr)
        select = val_m["f1"] if model_select == "val_f1_plus" else val_m["auc"]

        results[name] = {
            "val_threshold": float(thr),
            "val": val_m,
        }
        print(
            f"{name:14s} val_f1={val_m['f1']:.3f} val_auc={val_m['auc']:.3f} "
            f"thr={thr:.3f}"
        )
        if not np.isnan(select) and select > best_select:
            best_select, best_name, best_model, best_thr = (
                select,
                name,
                clf,
                thr,
            )

    if best_model is not None and best_name is not None:
        best_scores, test_pred = scores_from_classifier(best_model, X_test, plus_idx)
        test_bundle = evaluate_plus_ovr_bundle(
            y_test,
            best_scores,
            test_pred,
            plus_idx,
            best_thr,
            class_names,
            n_boot=n_boot,
            seed=seed,
            groups=(
                test_meta["group_id"].values
                if "group_id" in test_meta.columns
                else None
            ),
        )
        results[best_name]["test"] = test_bundle
        results[best_name]["per_source_test"] = per_source_plus_metrics(
            test_meta, best_scores, plus_idx, best_thr
        )
        print(
            f"[selected:{best_name}] test_auc="
            f"{test_bundle['plus_ovr']['auc']:.3f} test_sens="
            f"{test_bundle['plus_ovr']['sensitivity']:.3f}"
        )

    summary = {
        "best_model": best_name,
        "best_val_threshold": float(best_thr) if best_thr is not None else 0.5,
        "model_select": model_select,
        "threshold_method": threshold_method,
        "metrics": results,
    }
    return summary, best_name, best_scores, best_model
