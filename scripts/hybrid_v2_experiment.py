#!/usr/bin/env python3
"""Leakage-aware Branch C v2 experiment.

Protocol:
1. Extract/cache frozen Branch B embeddings with checkpoint/split binding.
2. Rank predeclared binary Plus classifiers by grouped OOF AUC on train only.
3. Confirm top candidates on validation, lock winner by validation AUC.
4. Evaluate only locked winner on test.

The CNN encoder was trained on the full training split, so downstream grouped
OOF estimates model-head stability, not fully nested encoder generalization.
Validation and test remain unseen by the downstream heads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.classify.branch_b_cnn import BackboneClassifier
from src.classify.branch_c_hybrid import extract_embeddings
from src.utils.common import get_device, load_config, plus_class_index, sha256_file


ROOT = Path("/Users/moniaz/niki")
OUT_DIR = ROOT / "results" / "hybrid_v2"
CACHE_PATH = OUT_DIR / "embeddings_current.npz"
CACHE_META_PATH = OUT_DIR / "embeddings_current.json"
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


@dataclass(frozen=True)
class Candidate:
    name: str
    inputs: str
    reduction: str
    n_components: int | None
    c: float


CANDIDATES = [
    Candidate("bio_safe_logit_c0.1", "bio", "none", None, 0.1),
    Candidate("bio_safe_logit_c1", "bio", "none", None, 1.0),
    Candidate("emb_logit_c0.001", "emb", "none", None, 0.001),
    Candidate("emb_logit_c0.01", "emb", "none", None, 0.01),
    Candidate("emb_logit_c0.1", "emb", "none", None, 0.1),
    Candidate("emb_logit_c1", "emb", "none", None, 1.0),
    Candidate("fusion_safe_logit_c0.001", "fusion", "none", None, 0.001),
    Candidate("fusion_safe_logit_c0.01", "fusion", "none", None, 0.01),
    Candidate("fusion_safe_logit_c0.1", "fusion", "none", None, 0.1),
    Candidate("fusion_safe_pca64_c0.1", "fusion", "pca", 64, 0.1),
    Candidate("fusion_safe_pca128_c0.1", "fusion", "pca", 128, 0.1),
    Candidate("fusion_safe_select256_c0.1", "fusion", "select", 256, 0.1),
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binary(labels: np.ndarray, plus_idx: int) -> np.ndarray:
    return (np.asarray(labels, dtype=int) == plus_idx).astype(int)


def _metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, score)),
        "threshold": float(threshold),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _youden_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(score)
    best_threshold, best_j = 0.5, -np.inf
    for threshold in candidates:
        pred = (score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, pred, labels=[0, 1]
        ).ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        j = sensitivity + specificity - 1.0
        if j > best_j:
            best_j = j
            best_threshold = float(threshold)
    return best_threshold


def _source_aucs(meta: pd.DataFrame, score: np.ndarray, plus_idx: int) -> dict:
    output = {}
    y = _binary(meta["label"].values, plus_idx)
    for source in sorted(meta["source"].unique()):
        mask = meta["source"].values == source
        output[str(source)] = float(roc_auc_score(y[mask], score[mask]))
    return output


class Head:
    def __init__(self, candidate: Candidate, seed: int):
        self.candidate = candidate
        self.seed = seed
        self.emb_scaler = None
        self.bio_scaler = None
        self.reducer = None
        self.model = LogisticRegression(
            C=candidate.c,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        )

    def _fit_transform(
        self, emb: np.ndarray, bio: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        pieces = []
        if self.candidate.inputs in {"emb", "fusion"}:
            self.emb_scaler = StandardScaler()
            emb_scaled = self.emb_scaler.fit_transform(emb)
            if self.candidate.reduction == "pca":
                self.reducer = PCA(
                    n_components=self.candidate.n_components,
                    svd_solver="randomized",
                    random_state=self.seed,
                )
                emb_scaled = self.reducer.fit_transform(emb_scaled)
            elif self.candidate.reduction == "select":
                self.reducer = SelectKBest(
                    f_classif, k=min(self.candidate.n_components, emb.shape[1])
                )
                emb_scaled = self.reducer.fit_transform(emb_scaled, y)
            pieces.append(emb_scaled)
        if self.candidate.inputs in {"bio", "fusion"}:
            self.bio_scaler = StandardScaler()
            pieces.append(self.bio_scaler.fit_transform(bio))
        return np.hstack(pieces)

    def _transform(self, emb: np.ndarray, bio: np.ndarray) -> np.ndarray:
        pieces = []
        if self.candidate.inputs in {"emb", "fusion"}:
            transformed = self.emb_scaler.transform(emb)
            if self.reducer is not None:
                transformed = self.reducer.transform(transformed)
            pieces.append(transformed)
        if self.candidate.inputs in {"bio", "fusion"}:
            pieces.append(self.bio_scaler.transform(bio))
        return np.hstack(pieces)

    def fit(
        self, emb: np.ndarray, bio: np.ndarray, y: np.ndarray
    ) -> "Head":
        x = self._fit_transform(emb, bio, y)
        self.model.fit(x, y)
        return self

    def predict(self, emb: np.ndarray, bio: np.ndarray) -> np.ndarray:
        x = self._transform(emb, bio)
        return self.model.predict_proba(x)[:, 1]


def _load_tables() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    raw = pd.read_csv(ROOT / "data/features/biomarker_features.csv")
    safe = pd.read_csv(ROOT / "data/features/biomarker_features_phase4.csv")
    if set(raw["image_path"]) != set(safe["image_path"]):
        raise SystemExit("Raw and release-safe image sets differ.")
    safe_cols = [
        column
        for column in safe.columns
        if column not in META_COLS
        and pd.api.types.is_numeric_dtype(safe[column])
    ]
    safe_numeric = safe[["image_path"] + safe_cols].copy()
    frame = raw[list(META_COLS & set(raw.columns))].merge(
        safe_numeric, on="image_path", how="inner", validate="one_to_one"
    )
    split_order = {"train": 0, "val": 1, "test": 2}
    frame["_split_order"] = frame["split"].map(split_order)
    frame = frame.sort_values(["_split_order", "image_path"]).drop(
        columns="_split_order"
    )
    train_mask = frame["split"].values == "train"
    medians = frame.loc[train_mask, safe_cols].median().fillna(0.0)
    bio = frame[safe_cols].fillna(medians).to_numpy(dtype=np.float32)
    if not np.isfinite(bio).all():
        raise SystemExit("Non-finite release-safe biomarkers.")
    return frame.reset_index(drop=True), bio, safe_cols


def _checkpoint_path() -> tuple[Path, dict]:
    results = json.loads((ROOT / "results/branch_b_results.json").read_text())
    checkpoint = Path(results["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    return checkpoint, results


def _load_or_extract_embeddings(frame: pd.DataFrame) -> np.ndarray:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint, branch_b = _checkpoint_path()
    split_sha = _sha(ROOT / "data/splits/all.csv")
    checkpoint_sha = _sha(checkpoint)
    expected = {
        "split_sha256": split_sha,
        "checkpoint_sha256": checkpoint_sha,
        "n_rows": len(frame),
        "image_order_sha256": hashlib.sha256(
            "\n".join(frame["image_path"]).encode()
        ).hexdigest(),
    }
    if CACHE_PATH.exists() and CACHE_META_PATH.exists():
        meta = json.loads(CACHE_META_PATH.read_text())
        if all(meta.get(key) == value for key, value in expected.items()):
            cached = np.load(CACHE_PATH)
            embeddings = cached["embeddings"]
            if len(embeddings) == len(frame):
                print(f"[cache] loaded {embeddings.shape} from {CACHE_PATH}")
                return embeddings

    state = torch.load(str(checkpoint), map_location=get_device())
    if state.get("split_sha256") != split_sha:
        raise SystemExit("Branch B checkpoint split hash mismatch.")
    backbone = state.get("backbone", branch_b["backbone"])
    model = BackboneClassifier(
        backbone, num_classes=int(state["state_dict"]["head.weight"].shape[0])
    ).to(get_device())
    model.load_state_dict(state["state_dict"])
    cfg = load_config()
    embeddings = extract_embeddings(
        model,
        frame["image_path"].tolist(),
        cfg["classify_b"]["img_size"],
        get_device(),
    ).astype(np.float32)
    np.savez_compressed(CACHE_PATH, embeddings=embeddings)
    CACHE_META_PATH.write_text(
        json.dumps(
            {
                **expected,
                "shape": list(embeddings.shape),
                "backbone": backbone,
            },
            indent=2,
        )
    )
    print(f"[cache] wrote {embeddings.shape} -> {CACHE_PATH}")
    return embeddings


def _oof_candidate(
    candidate: Candidate,
    emb: np.ndarray,
    bio: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    source: np.ndarray,
    seed: int,
) -> dict:
    folds = StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=seed
    )
    scores = np.full(len(y), np.nan, dtype=float)
    fold_aucs = []
    for fold, (fit_idx, hold_idx) in enumerate(
        folds.split(np.zeros(len(y)), y, groups), start=1
    ):
        head = Head(candidate, seed + fold)
        head.fit(emb[fit_idx], bio[fit_idx], y[fit_idx])
        scores[hold_idx] = head.predict(emb[hold_idx], bio[hold_idx])
        fold_aucs.append(float(roc_auc_score(y[hold_idx], scores[hold_idx])))
    if not np.isfinite(scores).all():
        raise RuntimeError(f"Missing OOF predictions for {candidate.name}")
    meta = pd.DataFrame({"label": np.where(y == 1, 2, 0), "source": source})
    source_aucs = _source_aucs(meta, scores, plus_idx=2)
    return {
        "candidate": candidate.name,
        "inputs": candidate.inputs,
        "reduction": candidate.reduction,
        "n_components": candidate.n_components,
        "C": candidate.c,
        "oof_auc": float(roc_auc_score(y, scores)),
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_sd": float(np.std(fold_aucs, ddof=1)),
        "fold_aucs": fold_aucs,
        "source_aucs": source_aucs,
        "source_macro_auc": float(np.mean(list(source_aucs.values()))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate only the candidate locked by train OOF + validation",
    )
    args = parser.parse_args()
    cfg = load_config()
    seed = int(cfg["seed"])
    plus_idx = plus_class_index(cfg)
    frame, bio, safe_cols = _load_tables()
    emb = _load_or_extract_embeddings(frame)
    if emb.shape != (len(frame), 2048):
        raise SystemExit(f"Unexpected embedding shape {emb.shape}")

    labels = frame["label"].to_numpy(dtype=int)
    y = _binary(labels, plus_idx)
    groups = frame["group_id"].astype(str).to_numpy()
    source = frame["source"].astype(str).to_numpy()
    train_mask = frame["split"].values == "train"
    val_mask = frame["split"].values == "val"
    test_mask = frame["split"].values == "test"

    print(
        f"[data] train={train_mask.sum()} val={val_mask.sum()} "
        f"test={test_mask.sum()} emb={emb.shape[1]} bio_safe={bio.shape[1]}"
    )
    cv_rows = []
    for index, candidate in enumerate(CANDIDATES, start=1):
        print(f"[cv {index}/{len(CANDIDATES)}] {candidate.name}")
        result = _oof_candidate(
            candidate,
            emb[train_mask],
            bio[train_mask],
            y[train_mask],
            groups[train_mask],
            source[train_mask],
            seed,
        )
        cv_rows.append(result)
        print(
            f"  OOF={result['oof_auc']:.4f} "
            f"fold={result['fold_auc_mean']:.4f}±{result['fold_auc_sd']:.4f} "
            f"source_macro={result['source_macro_auc']:.4f}"
        )

    cv_frame = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key not in {"fold_aucs", "source_aucs"}}
            for row in cv_rows
        ]
    ).sort_values(["oof_auc", "source_macro_auc"], ascending=False)
    cv_frame.to_csv(OUT_DIR / "train_grouped_oof_candidates.csv", index=False)
    (OUT_DIR / "train_grouped_oof_details.json").write_text(
        json.dumps(cv_rows, indent=2)
    )

    # Validation confirmation is restricted to top three train-OOF candidates.
    top_names = cv_frame.head(3)["candidate"].tolist()
    val_rows = []
    fitted = {}
    for name in top_names:
        candidate = next(item for item in CANDIDATES if item.name == name)
        head = Head(candidate, seed).fit(
            emb[train_mask], bio[train_mask], y[train_mask]
        )
        val_score = head.predict(emb[val_mask], bio[val_mask])
        threshold = _youden_threshold(y[val_mask], val_score)
        row = {
            "candidate": name,
            **_metrics(y[val_mask], val_score, threshold),
            "source_aucs": _source_aucs(
                frame.loc[val_mask, ["label", "source"]],
                val_score,
                plus_idx,
            ),
        }
        val_rows.append(row)
        fitted[name] = head
        print(
            f"[val] {name}: AUC={row['auc']:.4f} "
            f"sens={row['sensitivity']:.4f} spec={row['specificity']:.4f}"
        )

    # Predeclared lock: highest validation AUC among top-3 OOF candidates.
    locked = max(val_rows, key=lambda row: row["auc"])
    lock = {
        "locked_candidate": locked["candidate"],
        "selection_rule": (
            "top 3 by grouped train OOF AUC; highest validation AUC; "
            "test never used for selection"
        ),
        "train_grouped_oof_top3": top_names,
        "validation_results": val_rows,
        "split_sha256": _sha(ROOT / "data/splits/all.csv"),
        "branch_b_checkpoint_sha256": _sha(_checkpoint_path()[0]),
        "release_safe_features": safe_cols,
        "test_evaluated": bool(args.evaluate_test),
        "nested_encoder_limitation": (
            "Branch B encoder saw the full training split; downstream grouped "
            "OOF measures head stability, not fully nested encoder performance."
        ),
    }

    if args.evaluate_test:
        candidate = next(
            item for item in CANDIDATES
            if item.name == locked["candidate"]
        )
        head = fitted[locked["candidate"]]
        test_score = head.predict(emb[test_mask], bio[test_mask])
        test_result = _metrics(
            y[test_mask], test_score, locked["threshold"]
        )
        test_result["source_aucs"] = _source_aucs(
            frame.loc[test_mask, ["label", "source"]],
            test_score,
            plus_idx,
        )
        lock["test_result"] = test_result
        pred = frame.loc[
            test_mask,
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
        pred["p_plus_branch_c_v2"] = test_score
        pred.to_csv(OUT_DIR / "locked_test_predictions.csv", index=False)
        print(
            f"[test locked] {candidate.name}: "
            f"AUC={test_result['auc']:.4f} "
            f"sens={test_result['sensitivity']:.4f} "
            f"spec={test_result['specificity']:.4f} "
            f"f1={test_result['f1']:.4f}"
        )

    (OUT_DIR / "locked_experiment.json").write_text(json.dumps(lock, indent=2))
    print(
        f"[locked] {locked['candidate']} val_AUC={locked['auc']:.4f} "
        f"test_evaluated={args.evaluate_test}"
    )


if __name__ == "__main__":
    main()
