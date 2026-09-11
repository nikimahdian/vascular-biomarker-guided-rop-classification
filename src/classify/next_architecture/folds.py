"""Grouped fold manifests. Never silently regenerate a hashed assignment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.common import sha256_file


def _stratify_key(frame: pd.DataFrame) -> np.ndarray:
    """Composite label for StratifiedGroupKFold: class, with source mixed in when possible."""
    labels = frame["label"].astype(int).astype(str)
    if "source" in frame.columns:
        return (labels + "::" + frame["source"].astype(str)).values
    return labels.values


def build_grouped_folds(
    frame: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> dict[str, Any]:
    if "group_id" not in frame.columns or frame["group_id"].isna().any():
        raise ValueError("Fold construction requires complete group_id.")
    development = frame.reset_index(drop=True).copy()
    y = _stratify_key(development)
    groups = development["group_id"].values
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(development, y, groups), start=1):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        if train_groups & val_groups:
            raise RuntimeError(f"Group leakage in fold {fold}.")
        folds.append(
            {
                "fold": fold,
                "train_index": [int(i) for i in train_idx],
                "val_index": [int(i) for i in val_idx],
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "train_groups": sorted(map(str, train_groups)),
                "val_groups": sorted(map(str, val_groups)),
            }
        )
    payload = {
        "protocol": "StratifiedGroupKFold on provided development rows; groups never split",
        "n_folds": n_folds,
        "seed": seed,
        "n_rows": int(len(development)),
        "farabi_grouping": "exam_level_not_proven_patient",
        "image_paths": development["image_path"].astype(str).tolist(),
        "folds": folds,
    }
    return payload


def manifest_sha256(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def save_fold_manifest(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["manifest_sha256"] = manifest_sha256({k: v for k, v in payload.items() if k != "manifest_sha256"})
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)
    return payload["manifest_sha256"]


def load_or_create_folds(
    path: Path,
    frame: pd.DataFrame,
    n_folds: int,
    seed: int,
    *,
    regenerate: bool = False,
) -> dict[str, Any]:
    if path.exists() and not regenerate:
        payload = json.loads(path.read_text())
        stored = payload.get("manifest_sha256")
        recomputed = manifest_sha256({k: v for k, v in payload.items() if k != "manifest_sha256"})
        if stored != recomputed:
            raise SystemExit(f"Fold manifest hash mismatch: {path}")
        if payload.get("n_folds") != n_folds or payload.get("seed") != seed:
            raise SystemExit(
                f"Existing fold manifest {path} has n_folds={payload.get('n_folds')} "
                f"seed={payload.get('seed')}; refusing silent regeneration. Pass --regenerate-folds."
            )
        if payload.get("image_paths") != frame["image_path"].astype(str).tolist():
            raise SystemExit("Fold manifest image order does not match development frame.")
        return payload
    payload = build_grouped_folds(frame, n_folds, seed)
    save_fold_manifest(path, payload)
    return json.loads(path.read_text())


def official_reduced_fold(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, Any]:
    """Single fold = official train/val. Explicitly NOT nested CV."""
    return {
        "protocol": "official_train_val_not_nested_cv",
        "n_folds": 1,
        "farabi_grouping": "exam_level_not_proven_patient",
        "folds": [
            {
                "fold": "official",
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "train_groups": sorted(train_df["group_id"].astype(str).unique()),
                "val_groups": sorted(val_df["group_id"].astype(str).unique()),
            }
        ],
        "split_files": {"train": "train.csv", "val": "val.csv"},
    }


def assert_no_group_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    overlap = set(train_df["group_id"]) & set(val_df["group_id"])
    if overlap:
        raise RuntimeError(f"Group overlap between train and val: {sorted(overlap)[:5]}")


def assert_no_test_images(dev_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    overlap = set(dev_df["image_path"]) & set(test_df["image_path"])
    if overlap:
        raise RuntimeError("Development frame contains canonical test images.")
