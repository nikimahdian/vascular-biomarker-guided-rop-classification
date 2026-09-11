"""Development metrics, calibration, and grouped/paired bootstrap."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src.utils.common import binary_metrics, tune_binary_threshold


def plus_binary(y: np.ndarray, plus_idx: int = 2) -> np.ndarray:
    return (np.asarray(y).astype(int) == plus_idx).astype(int)


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(((y_prob - y_true) ** 2).mean())


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    ece, total = 0.0, len(y_true)
    if total == 0:
        return float("nan")
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def high_sensitivity_threshold(
    y_true: np.ndarray, y_score: np.ndarray, target_sens: float = 0.95
) -> float:
    """Lowest threshold on *validation* that reaches target sensitivity; else min score."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    yt = y_true[order]
    ys = y_score[order]
    tp = 0
    n_pos = max(int(y_true.sum()), 1)
    chosen = float(ys[-1]) if len(ys) else 0.5
    for i, label in enumerate(yt):
        tp += int(label == 1)
        if tp / n_pos >= target_sens:
            chosen = float(ys[i])
            break
    return chosen


def discrimination_bundle(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    *,
    n_bins_ece: int = 15,
) -> dict[str, Any]:
    metrics = binary_metrics(y_true_bin, y_score, threshold=threshold)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true_bin, pred, labels=[0, 1])
    metrics["auprc"] = auprc(y_true_bin, y_score)
    metrics["brier"] = brier_score(y_true_bin, y_score)
    metrics["ece"] = expected_calibration_error(y_true_bin, y_score, n_bins=n_bins_ece)
    metrics["threshold"] = float(threshold)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def per_source_bundle(
    sources: np.ndarray,
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    aucs = []
    for source in sorted(set(map(str, sources))):
        mask = np.asarray(sources).astype(str) == source
        if mask.sum() == 0:
            continue
        sub = discrimination_bundle(y_true_bin[mask], y_score[mask], threshold)
        sub["n"] = int(mask.sum())
        sub["n_plus"] = int(y_true_bin[mask].sum())
        out[source] = sub
        if not np.isnan(sub["auc"]):
            aucs.append(sub["auc"])
    out["_worst_source_auc"] = float(min(aucs)) if aucs else float("nan")
    out["_n_sources_with_auc"] = len(aucs)
    return out


def fit_temperature(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    y = np.asarray(y_true_bin, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration validation labels must contain both classes.")
    z = np.log(p / (1 - p))

    def nll(log_t: float) -> float:
        q = 1.0 / (1.0 + np.exp(-z / np.exp(log_t)))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    result = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def apply_temperature(y_prob: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) / float(temperature)
    return 1.0 / (1.0 + np.exp(-z))


def grouped_bootstrap_auc(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    idx = np.arange(len(y_true_bin))
    samples = []
    skipped = 0
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draw = np.concatenate([idx[groups == g] for g in sampled])
        yt, ys = y_true_bin[draw], y_score[draw]
        if len(np.unique(yt)) < 2:
            skipped += 1
            continue
        samples.append(float(roc_auc_score(yt, ys)))
    point = float(roc_auc_score(y_true_bin, y_score)) if len(np.unique(y_true_bin)) > 1 else float("nan")
    if samples:
        lo, hi = np.percentile(samples, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "n_valid": len(samples),
        "n_skipped_single_class": skipped,
        "unit": "group_id",
    }


def paired_grouped_delta_auc(
    y_true_bin: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Delta = AUC(B) - AUC(A) with cluster bootstrap on identical samples."""
    y_true_bin = np.asarray(y_true_bin).astype(int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    groups = np.asarray(groups)
    if not (len(y_true_bin) == len(score_a) == len(score_b) == len(groups)):
        raise ValueError("Paired comparison requires aligned arrays.")
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    idx = np.arange(len(y_true_bin))
    deltas = []
    skipped = 0
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draw = np.concatenate([idx[groups == g] for g in sampled])
        yt = y_true_bin[draw]
        if len(np.unique(yt)) < 2:
            skipped += 1
            continue
        deltas.append(float(roc_auc_score(yt, score_b[draw]) - roc_auc_score(yt, score_a[draw])))
    point = float(roc_auc_score(y_true_bin, score_b) - roc_auc_score(y_true_bin, score_a))
    if deltas:
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p = 2 * min((np.array(deltas) > 0).mean(), (np.array(deltas) < 0).mean())
    else:
        lo = hi = p = float("nan")
    return {
        "delta": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_two_sided": float(p),
        "n_boot": n_boot,
        "n_valid": len(deltas),
        "n_skipped_single_class": skipped,
        "unit": "group_id",
    }


def youden_threshold(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    return float(tune_binary_threshold(y_true_bin, y_score, method="youden"))
