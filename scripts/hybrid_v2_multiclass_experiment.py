#!/usr/bin/env python3
"""Multiclass linear-head trial for frozen B embeddings and safe hybrid fusion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "results" / "hybrid_v2_multiclass"
META = {
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
C_VALUES = [0.0001, 0.001, 0.01, 0.1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def youden(y: np.ndarray, score: np.ndarray) -> float:
    binary = (y == 2).astype(int)
    best_threshold, best_j = 0.5, -np.inf
    for threshold in np.unique(score):
        pred = (score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            binary, pred, labels=[0, 1]
        ).ravel()
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > best_j:
            best_j = j
            best_threshold = float(threshold)
    return best_threshold


def evaluate(
    y: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict:
    plus = probabilities[:, 2]
    binary = (y == 2).astype(int)
    binary_pred = (plus >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        binary, binary_pred, labels=[0, 1]
    ).ravel()
    multiclass_pred = probabilities.argmax(axis=1)
    return {
        "plus_auc": float(roc_auc_score(binary, plus)),
        "threshold": float(threshold),
        "plus_sensitivity": float(tp / max(tp + fn, 1)),
        "plus_specificity": float(tn / max(tn + fp, 1)),
        "plus_f1": float(f1_score(binary, binary_pred, zero_division=0)),
        "multiclass_accuracy": float(accuracy_score(y, multiclass_pred)),
        "multiclass_macro_f1": float(
            f1_score(y, multiclass_pred, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y, multiclass_pred, labels=[0, 1, 2]
        ).tolist(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "data/features/biomarker_features.csv")
    safe = pd.read_csv(ROOT / "data/features/biomarker_features_phase4.csv")
    safe_cols = [
        col
        for col in safe.columns
        if col not in META
        and col != "singularity_length"
        and pd.api.types.is_numeric_dtype(safe[col])
    ]
    safe = safe[["image_path"] + safe_cols]
    frame = raw[list(META & set(raw.columns))].merge(
        safe, on="image_path", how="inner", validate="one_to_one"
    )
    frame["_order"] = frame["split"].map({"train": 0, "val": 1, "test": 2})
    frame = frame.sort_values(["_order", "image_path"]).drop(
        columns="_order"
    ).reset_index(drop=True)

    cache = np.load(ROOT / "results/hybrid_v2/embeddings_current.npz")
    embeddings = cache["embeddings"]
    cache_meta = json.loads(
        (ROOT / "results/hybrid_v2/embeddings_current.json").read_text()
    )
    order_sha = hashlib.sha256(
        "\n".join(frame["image_path"]).encode()
    ).hexdigest()
    if cache_meta["image_order_sha256"] != order_sha:
        raise SystemExit("Embedding cache order mismatch.")
    if len(embeddings) != len(frame):
        raise SystemExit("Embedding cache row mismatch.")

    train = frame["split"].values == "train"
    val = frame["split"].values == "val"
    test = frame["split"].values == "test"
    y = frame["label"].to_numpy(dtype=int)
    medians = frame.loc[train, safe_cols].median().fillna(0.0)
    bio = frame[safe_cols].fillna(medians).to_numpy(dtype=float)

    candidates = []
    fitted = {}
    for mode in ["embedding_only", "fusion_safe16"]:
        X = embeddings if mode == "embedding_only" else np.hstack(
            [embeddings, bio]
        )
        for c_value in C_VALUES:
            scaler = StandardScaler().fit(X[train])
            model = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
                solver="lbfgs",
            )
            model.fit(scaler.transform(X[train]), y[train])
            val_prob = model.predict_proba(scaler.transform(X[val]))
            threshold = youden(y[val], val_prob[:, 2])
            result = {
                "candidate": f"{mode}_c{c_value}",
                "mode": mode,
                "C": c_value,
                "validation": evaluate(y[val], val_prob, threshold),
            }
            candidates.append(result)
            fitted[result["candidate"]] = (model, scaler, X)
            print(
                f"[val] {result['candidate']}: "
                f"AUC={result['validation']['plus_auc']:.4f} "
                f"macroF1={result['validation']['multiclass_macro_f1']:.4f}"
            )

    # Highest validation Plus AUC; macro-F1 tie-break. Test unseen.
    locked = max(
        candidates,
        key=lambda row: (
            row["validation"]["plus_auc"],
            row["validation"]["multiclass_macro_f1"],
        ),
    )
    model, scaler, X = fitted[locked["candidate"]]
    test_prob = model.predict_proba(scaler.transform(X[test]))
    test_result = evaluate(
        y[test], test_prob, locked["validation"]["threshold"]
    )
    print(
        f"[LOCKED TEST] {locked['candidate']}: "
        f"AUC={test_result['plus_auc']:.4f} "
        f"macroF1={test_result['multiclass_macro_f1']:.4f} "
        f"accuracy={test_result['multiclass_accuracy']:.4f}"
    )
    payload = {
        "protocol": (
            "predeclared embedding-only and safe16-fusion multiclass logistic "
            "candidates; select by validation Plus AUC; test locked winner once"
        ),
        "split_sha256": sha(ROOT / "data/splits/all.csv"),
        "embedding_cache_meta": cache_meta,
        "safe_features": safe_cols,
        "candidates": candidates,
        "locked_candidate": locked["candidate"],
        "test": test_result,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    pred = frame.loc[
        test,
        [
            "image_path",
            "label",
            "source",
            "group_id",
            "patient_id",
            "exam_id",
            "identity_level",
        ],
    ].copy()
    for class_index in range(3):
        pred[f"p_class_{class_index}"] = test_prob[:, class_index]
    pred.to_csv(OUT / "test_predictions.csv", index=False)
    print(f"[saved] {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
