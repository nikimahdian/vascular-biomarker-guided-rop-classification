#!/usr/bin/env python3
"""True hybrid meta-fusion: Branch B probability + safe biomarkers.

The meta-model is selected with grouped OOF predictions inside validation,
then fitted on full validation and evaluated once on test. This is exploratory
because Branch B checkpoint selection already used validation labels.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "results" / "hybrid_meta_fusion"
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


@dataclass(frozen=True)
class Candidate:
    name: str
    bio_profile: str
    c: float


CANDIDATES = [
    Candidate("B_only_logit", "none", 1.0),
    Candidate("B_plus_safe16_c0.001", "safe16", 0.001),
    Candidate("B_plus_safe16_c0.01", "safe16", 0.01),
    Candidate("B_plus_safe16_c0.1", "safe16", 0.1),
    Candidate("B_plus_safe16_c1", "safe16", 1.0),
    Candidate("B_plus_normalized_c0.001", "normalized", 0.001),
    Candidate("B_plus_normalized_c0.01", "normalized", 0.01),
    Candidate("B_plus_normalized_c0.1", "normalized", 0.1),
    Candidate("B_plus_normalized_c1", "normalized", 1.0),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def youden(y: np.ndarray, scores: np.ndarray) -> float:
    best_threshold, best_j = 0.5, -np.inf
    for threshold in np.unique(scores):
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        j = sensitivity + specificity - 1
        if j > best_j:
            best_j = j
            best_threshold = float(threshold)
    return best_threshold


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, scores)),
        "threshold": float(threshold),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def load_profile(path: Path, drop: set[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    drop = drop or set()
    columns = [
        col
        for col in frame.columns
        if col not in META
        and col not in drop
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    return frame[["image_path"] + columns], columns


def build_frame(split: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    b = pd.read_csv(ROOT / f"results/branch_b_{split}_preds.csv")
    b_col = next(col for col in b.columns if col.startswith("p_plus"))
    safe, safe_cols = load_profile(
        ROOT / "data/features/biomarker_features_phase4.csv",
        drop={"singularity_length"},
    )
    normalized, normalized_cols = load_profile(
        ROOT / "data/features/biomarker_features_normalized.csv",
        drop={"singularity_length"},
    )
    normalized = normalized.rename(
        columns={col: f"norm__{col}" for col in normalized_cols}
    )
    normalized_cols = [f"norm__{col}" for col in normalized_cols]
    out = b.merge(safe, on="image_path", how="inner", validate="one_to_one")
    out = out.merge(
        normalized, on="image_path", how="inner", validate="one_to_one"
    )
    if len(out) != len(b):
        raise SystemExit(f"{split}: feature/prediction mismatch {len(out)} != {len(b)}")
    out["b_logit"] = logit(out[b_col].values)
    return out, {"safe16": safe_cols, "normalized": normalized_cols}


def matrix(
    frame: pd.DataFrame, candidate: Candidate, profiles: dict[str, list[str]]
) -> np.ndarray:
    columns = ["b_logit"]
    if candidate.bio_profile != "none":
        columns += profiles[candidate.bio_profile]
    data = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    medians = data.median().fillna(0.0)
    return data.fillna(medians).to_numpy(dtype=float)


def source_aucs(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    output = {}
    for source in sorted(frame["source"].unique()):
        mask = frame["source"].values == source
        y = (frame.loc[mask, "label"].values == 2).astype(int)
        output[str(source)] = float(roc_auc_score(y, scores[mask]))
    return output


def oof(
    frame: pd.DataFrame,
    X: np.ndarray,
    candidate: Candidate,
    seed: int,
) -> tuple[np.ndarray, dict]:
    y = (frame["label"].values == 2).astype(int)
    groups = frame["group_id"].astype(str).values
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.full(len(frame), np.nan)
    fold_aucs = []
    for fold, (fit_idx, hold_idx) in enumerate(
        folds.split(X, y, groups), start=1
    ):
        scaler = StandardScaler().fit(X[fit_idx])
        model = LogisticRegression(
            C=candidate.c,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed + fold,
            solver="lbfgs",
        )
        model.fit(scaler.transform(X[fit_idx]), y[fit_idx])
        scores[hold_idx] = model.predict_proba(
            scaler.transform(X[hold_idx])
        )[:, 1]
        fold_aucs.append(float(roc_auc_score(y[hold_idx], scores[hold_idx])))
    threshold = youden(y, scores)
    result = {
        **metrics(y, scores, threshold),
        "fold_aucs": fold_aucs,
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_sd": float(np.std(fold_aucs, ddof=1)),
        "source_aucs": source_aucs(frame, scores),
    }
    result["source_macro_auc"] = float(
        np.mean(list(result["source_aucs"].values()))
    )
    return scores, result


def fit_predict(
    val: pd.DataFrame,
    test: pd.DataFrame,
    X_val: np.ndarray,
    X_test: np.ndarray,
    candidate: Candidate,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = (val["label"].values == 2).astype(int)
    scaler = StandardScaler().fit(X_val)
    model = LogisticRegression(
        C=candidate.c,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(scaler.transform(X_val), y)
    return (
        model.predict_proba(scaler.transform(X_val))[:, 1],
        model.predict_proba(scaler.transform(X_test))[:, 1],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    val, val_profiles = build_frame("val")
    test, test_profiles = build_frame("test")
    if val_profiles != test_profiles:
        raise SystemExit("Val/test biomarker profiles differ.")
    results = []
    matrices = {}
    for candidate in CANDIDATES:
        X = matrix(val, candidate, val_profiles)
        matrices[candidate.name] = X
        _, result = oof(val, X, candidate, seed=42)
        result.update(
            {
                "candidate": candidate.name,
                "bio_profile": candidate.bio_profile,
                "C": candidate.c,
                "n_features": int(X.shape[1]),
            }
        )
        results.append(result)
        print(
            f"[OOF] {candidate.name}: pooled={result['auc']:.4f} "
            f"fold={result['fold_auc_mean']:.4f}±{result['fold_auc_sd']:.4f} "
            f"macro={result['source_macro_auc']:.4f}"
        )

    # Lock by fold-mean AUC, tie-break source-macro AUC. No test data consulted.
    locked_result = max(
        results,
        key=lambda row: (row["fold_auc_mean"], row["source_macro_auc"]),
    )
    locked = next(
        item for item in CANDIDATES
        if item.name == locked_result["candidate"]
    )
    X_val = matrices[locked.name]
    X_test = matrix(test, locked, test_profiles)
    _, test_scores = fit_predict(
        val, test, X_val, X_test, locked, seed=42
    )
    y_test = (test["label"].values == 2).astype(int)
    test_result = metrics(
        y_test, test_scores, locked_result["threshold"]
    )
    test_result["source_aucs"] = source_aucs(test, test_scores)
    print(
        f"[LOCKED TEST] {locked.name}: AUC={test_result['auc']:.4f} "
        f"sens={test_result['sensitivity']:.4f} "
        f"spec={test_result['specificity']:.4f} "
        f"f1={test_result['f1']:.4f}"
    )

    b_test = pd.read_csv(ROOT / "results/branch_b_test_preds.csv")
    b_col = next(col for col in b_test.columns if col.startswith("p_plus"))
    b_auc = float(
        roc_auc_score(
            (b_test["label"].values == 2).astype(int),
            b_test[b_col].values,
        )
    )
    payload = {
        "protocol": (
            "5-fold grouped OOF candidate selection on validation; "
            "fit locked meta-model on full validation; evaluate test once"
        ),
        "exploratory_limitation": (
            "Branch B checkpoint selection already used validation labels."
        ),
        "split_sha256": sha(ROOT / "data/splits/all.csv"),
        "candidate_results": results,
        "locked_candidate": locked.name,
        "test_result": test_result,
        "branch_b_test_auc": b_auc,
        "delta_auc_vs_b": float(test_result["auc"] - b_auc),
        "safe_profile_note": (
            "singularity_length removed from phase4 profile because it is "
            "resolution-dependent; normalized profile used normalized variants."
        ),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    pred = test[
        [
            "image_path",
            "label",
            "source",
            "group_id",
            "patient_id",
            "exam_id",
            "identity_level",
        ]
    ].copy()
    pred["p_plus_hybrid_meta"] = test_scores
    pred.to_csv(OUT / "test_predictions.csv", index=False)
    print(f"[saved] {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
