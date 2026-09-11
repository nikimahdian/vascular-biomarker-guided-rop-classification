"""Shared helpers: config loading, seeding, paths, metrics."""
from __future__ import annotations

import os
import random
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Project root = two levels up from this file (src/utils/common.py -> project root).
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    """Current data/config/code identity for every new model artifact."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": cfg["seed"],
        "split_sha256": sha256_file(cfg["paths"]["splits_dir"] / "all.csv"),
        "config_sha256": sha256_file(ROOT / "configs" / "config.yaml"),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def provenance_matches_current(payload: dict[str, Any], cfg: dict[str, Any]) -> bool:
    provenance = payload.get("provenance", payload)
    return (
        provenance.get("split_sha256")
        == sha256_file(cfg["paths"]["splits_dir"] / "all.csv")
        and provenance.get("config_sha256")
        == sha256_file(ROOT / "configs" / "config.yaml")
    )


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the YAML config and resolve every path under `paths` to an absolute Path."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    for key, rel in cfg.get("paths", {}).items():
        cfg["paths"][key] = (ROOT / rel).resolve()
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for p in cfg.get("paths", {}).values():
        Path(p).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def get_device(prefer: str = "cuda") -> str:
    try:
        import torch

        if prefer == "cuda" and torch.cuda.is_available():
            return "cuda"
        if prefer in ("cuda", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def plus_class_index(cfg: dict[str, Any]) -> int:
    names = cfg.get("data", {}).get("class_names", ["No_Plus", "Plus"])
    if "Plus" in names:
        return names.index("Plus")
    return 1


def num_classes(cfg: dict[str, Any]) -> int:
    return len(cfg.get("data", {}).get("class_names", ["No_Plus", "Plus"]))


def plus_ovr_metrics(y_true, y_score, plus_idx: int = 2, threshold: float = 0.5) -> dict[str, float]:
    """Metrics for detecting Plus: binary OvR using P(Plus) vs label==plus_idx."""
    y_true = np.asarray(y_true).astype(int)
    if y_score.ndim == 2:
        y_score = y_score[:, plus_idx]
    y_bin = (y_true == plus_idx).astype(int)
    return binary_metrics(y_bin, y_score, threshold=threshold)


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    """Clinical metrics for a binary classifier. y_score = P(positive)."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    out: dict[str, float] = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["auc"] = float("nan")  # single-class test fold
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")  # recall for Plus
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    out["tp"], out["fp"], out["fn"], out["tn"] = map(float, (tp, fp, fn, tn))
    return out


def tune_binary_threshold(y_true, y_score, method: str = "youden") -> float:
    """Tune a threshold for labels already encoded as binary 0/1."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("Binary threshold tuning requires equal-length 1D labels and scores.")
    if len(y_true) == 0:
        raise ValueError("Binary threshold tuning requires at least one row.")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Binary threshold labels must contain only 0 and 1.")
    if not np.isfinite(y_score).all():
        raise ValueError("Threshold scores must all be finite.")
    if method not in {"f1", "youden"}:
        raise ValueError(f"Unknown threshold method: {method}")
    if len(np.unique(y_true)) < 2 and method == "youden":
        return 0.5
    unique = np.unique(y_score)
    candidates = np.concatenate(
        [
            [np.nextafter(unique.max(), np.inf)],
            unique[::-1],
            [np.nextafter(unique.min(), -np.inf)],
        ]
    )
    best_threshold, best_value = 0.5, -np.inf
    for threshold in candidates:
        prediction = (y_score >= threshold).astype(int)
        if method == "f1":
            value = f1_score(y_true, prediction, zero_division=0)
        else:
            positive = y_true == 1
            negative = ~positive
            sensitivity = prediction[positive].mean()
            specificity = (prediction[negative] == 0).mean()
            value = sensitivity + specificity - 1.0
        if value > best_value:
            best_value, best_threshold = value, float(threshold)
    return best_threshold


def tune_plus_threshold(
    y_true,
    y_score,
    plus_idx: int = 2,
    method: str = "youden",
) -> float:
    """Pick Plus-OvR threshold from multiclass validation labels and scores."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    unique_labels = set(np.unique(y_true))
    if plus_idx not in unique_labels and unique_labels == {0, 1} and plus_idx != 1:
        raise ValueError(
            "Labels look binary but tune_plus_threshold expects multiclass labels. "
            "Use tune_binary_threshold instead."
        )
    if y_score.ndim == 2:
        if plus_idx >= y_score.shape[1]:
            raise ValueError(
                f"plus_idx={plus_idx} outside score matrix with {y_score.shape[1]} columns."
            )
        y_score = y_score[:, plus_idx]
    return tune_binary_threshold((y_true == plus_idx).astype(int), y_score, method)


def multiclass_metrics(y_true, y_pred, class_names: list[str] | None = None) -> dict[str, float | dict]:
    """Standard 3-class report: accuracy, macro/weighted F1, per-class recall."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        recall_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    names = class_names or [str(i) for i in range(int(max(y_true.max(), y_pred.max())) + 1)]
    out: dict[str, float | dict] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    per_class: dict[str, dict[str, float]] = {}
    recalls = recall_score(y_true, y_pred, labels=list(range(len(names))), average=None, zero_division=0)
    for i, name in enumerate(names):
        per_class[name] = {"recall": float(recalls[i]) if i < len(recalls) else 0.0}
    out["per_class"] = per_class
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    out["confusion_matrix"] = cm.tolist()
    return out


def bootstrap_plus_ovr_ci(
    y_true,
    y_score,
    plus_idx: int = 2,
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    groups=None,
) -> dict[str, dict[str, float]]:
    """Cluster bootstrap CI; falls back to row bootstrap when groups are absent."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim == 2:
        y_score = y_score[:, plus_idx]
    n = len(y_true)
    if n == 0:
        return {}

    rng = np.random.default_rng(seed)
    keys = ["auc", "sensitivity", "specificity", "f1"]
    samples = {k: [] for k in keys}
    idx = np.arange(n)
    group_values = None if groups is None else np.asarray(groups)
    if group_values is not None and len(group_values) != n:
        raise ValueError("groups must have one value per row.")
    unique_groups = None if group_values is None else np.unique(group_values)
    for _ in range(n_boot):
        if unique_groups is None:
            draw = rng.choice(idx, size=n, replace=True)
        else:
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            draw = np.concatenate([idx[group_values == group] for group in sampled])
        yt, ys = y_true[draw], y_score[draw]
        if len(np.unique((yt == plus_idx).astype(int))) < 2:
            continue
        m = plus_ovr_metrics(yt, ys, plus_idx, threshold=threshold)
        for k in keys:
            if not np.isnan(m.get(k, float("nan"))):
                samples[k].append(m[k])

    point = plus_ovr_metrics(y_true, y_score, plus_idx, threshold=threshold)
    ci: dict[str, dict[str, float]] = {}
    lo = 100 * (alpha / 2)
    hi = 100 * (1 - alpha / 2)
    for k in keys:
        arr = samples[k]
        ci[k] = {
            "point": float(point.get(k, float("nan"))),
            "ci_low": float(np.percentile(arr, lo)) if arr else float("nan"),
            "ci_high": float(np.percentile(arr, hi)) if arr else float("nan"),
        }
    return ci


def evaluate_plus_ovr_bundle(
    y_true,
    y_score,
    y_pred_multiclass,
    plus_idx: int,
    threshold: float,
    class_names: list[str],
    n_boot: int = 1000,
    seed: int = 42,
    groups=None,
) -> dict[str, Any]:
    """Test-set bundle: Plus-OvR + 3-class metrics + bootstrap CIs."""
    plus = plus_ovr_metrics(y_true, y_score, plus_idx, threshold=threshold)
    plus["threshold"] = float(threshold)
    multi = multiclass_metrics(y_true, y_pred_multiclass, class_names)
    ci = bootstrap_plus_ovr_ci(
        y_true,
        y_score,
        plus_idx,
        threshold,
        n_boot=n_boot,
        seed=seed,
        groups=groups,
    )
    return {
        "plus_ovr": plus,
        "multiclass": multi,
        "bootstrap_ci": ci,
        "bootstrap_unit": "group" if groups is not None else "image",
    }


def append_csv_rows(path: Path, rows: list[dict]) -> None:
    """Append rows to a CSV, writing the header only when the file is new."""
    import pandas as pd

    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def validate_prediction_frame(
    predictions,
    split_name: str,
    cfg: dict[str, Any],
) -> None:
    """Reject stale predictions not matching the current canonical split."""
    import pandas as pd

    expected_path = cfg["paths"]["splits_dir"] / f"{split_name}.csv"
    expected = pd.read_csv(expected_path)[["image_path", "label"]]
    required = {"image_path", "label"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Prediction file missing columns: {sorted(required - set(predictions))}")
    if predictions["image_path"].duplicated().any():
        raise ValueError("Prediction file contains duplicate image_path rows.")
    if set(predictions["image_path"]) != set(expected["image_path"]):
        raise ValueError(
            f"Stale {split_name} predictions: image set does not match "
            f"{expected_path}."
        )
    observed = predictions.set_index("image_path")["label"].astype(int)
    truth = expected.set_index("image_path")["label"].astype(int)
    if not observed.loc[truth.index].equals(truth):
        raise ValueError(f"Prediction labels do not match current {split_name} split.")
