"""Final head-to-head: Biomarker (A) vs image CNN (B) vs HYBRID (C).

Reads rigorous eval outputs (val-tuned threshold, bootstrap CI when present).
Writes:
    results/comparison.csv
    results/comparison_bar.png
    results/roc_comparison.png

Usage:
    python -m src.compare.compare
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.common import (
    binary_metrics,
    ensure_dirs,
    load_config,
    plus_class_index,
    tune_binary_threshold,
    validate_prediction_frame,
)

KEYS = ["auc", "sensitivity", "specificity", "f1", "accuracy"]


def _load_json(p: Path):
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _branch_metrics(payload: dict | None, tag: str) -> dict | None:
    if payload is None:
        return None
    if tag in {"a", "c"}:
        best = payload.get("best_model")
        node = payload.get("metrics", {}).get(best, {})
        if "test" in node and "plus_ovr" in node["test"]:
            return node["test"]["plus_ovr"]
        if isinstance(node, dict) and "auc" in node:
            return node
    if tag == "b":
        if "test" in payload and "plus_ovr" in payload["test"]:
            return payload["test"]["plus_ovr"]
        return payload.get("metrics")
    return None


def _branch_label(payload: dict | None, tag: str, default: str) -> str:
    if payload is None:
        return default
    if tag == "a":
        return f"Branch A (biomarker: {payload.get('best_model', '?')})"
    if tag == "b":
        return f"Branch B (CNN: {payload.get('backbone', '?')})"
    if tag == "c":
        return f"Branch C (hybrid: {payload.get('best_model', '?')})"
    return default


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    res = cfg["paths"]["results_dir"]
    plus_idx = plus_class_index(cfg)

    a = _load_json(res / "branch_a_results.json")
    b = _load_json(res / "branch_b_results.json")
    c = _load_json(res / "branch_c_results.json")

    columns: dict[str, dict] = {}
    for payload, tag in [(a, "a"), (b, "b"), (c, "c")]:
        m = _branch_metrics(payload, tag)
        if m:
            columns[_branch_label(payload, tag, tag)] = m

    if len(columns) < 2:
        raise SystemExit("Need at least two branch result files. Run the branches first.")

    table = pd.DataFrame({name: {k: m.get(k) for k in KEYS} for name, m in columns.items()}).T
    table.to_csv(res / "comparison.csv")
    print(table.round(4).to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(KEYS))
    names = list(columns)
    n = len(names)
    w = 0.8 / n
    for i, name in enumerate(names):
        offset = (i - (n - 1) / 2) * w
        ax.bar(x + offset, [columns[name].get(k, np.nan) for k in KEYS], w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(KEYS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Biomarker vs CNN vs Hybrid (test, val-tuned threshold)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(res / "comparison_bar.png", dpi=150)
    plt.close()

    _roc_and_fusion(res, plus_idx, cfg)
    print(f"[done] comparison written to {res}")


def _roc_and_fusion(res: Path, plus_idx: int, cfg: dict) -> None:
    from sklearn.metrics import roc_curve

    branches = [
        ("branch_a_test_preds.csv", "p_plus_branch_a", "Branch A (biomarker)"),
        ("branch_b_test_preds.csv", "p_plus_branch_b", "Branch B (CNN)"),
        ("branch_c_test_preds.csv", "p_plus_branch_c", "Branch C (hybrid)"),
    ]

    plt.figure(figsize=(6, 6))
    plotted = 0
    preds: dict[str, pd.DataFrame] = {}
    for fname, col, label in branches:
        p = res / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        validate_prediction_frame(d, "test", cfg)
        preds[col] = d
        y_bin = (d["label"].astype(int) == plus_idx).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, d[col])
        m = binary_metrics(y_bin, d[col])
        plt.plot(fpr, tpr, label=f"{label} (AUC={m['auc']:.3f})")
        plotted += 1

    if plotted:
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("1 - specificity")
        plt.ylabel("sensitivity")
        plt.title("ROC: Plus vs rest (test set)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(res / "roc_comparison.png", dpi=150)
    plt.close()

    if "p_plus_branch_a" in preds and "p_plus_branch_b" in preds:
        va_p, vb_p = res / "branch_a_val_preds.csv", res / "branch_b_val_preds.csv"
        merged = preds["p_plus_branch_a"].merge(
            preds["p_plus_branch_b"], on="image_path", suffixes=("_a", "_b")
        )
        if not (va_p.exists() and vb_p.exists()):
            print("[late-fusion A+B] skipped: branch_{a,b}_val_preds.csv missing; cannot tune threshold on val")
        elif merged.empty:
            print("[late-fusion A+B] skipped: empty test merge")
        else:
            va = pd.read_csv(va_p)
            vb = pd.read_csv(vb_p)
            validate_prediction_frame(va, "val", cfg)
            validate_prediction_frame(vb, "val", cfg)
            mval = va.merge(vb, on="image_path", suffixes=("_a", "_b"))
            y_val_bin = (mval["label_a"].astype(int) == plus_idx).astype(int)
            s_val = (mval["p_plus_branch_a"] + mval["p_plus_branch_b"]) / 2.0
            thr = tune_binary_threshold(y_val_bin, s_val, method="youden")
            y = (merged["label_a"].astype(int) == plus_idx).astype(int)
            fused = (merged["p_plus_branch_a"] + merged["p_plus_branch_b"]) / 2.0
            fm = binary_metrics(y, fused, threshold=thr)
            fm["threshold"] = float(thr)
            with open(res / "fusion_results.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"method": "mean_probability_A_B", "threshold_tuned_on": "val_youden", "metrics": fm},
                    f,
                    indent=2,
                )
            print(
                f"[late-fusion A+B] thr={thr:.3f} AUC={fm['auc']:.3f} "
                f"sens={fm['sensitivity']:.3f} spec={fm['specificity']:.3f}"
            )


if __name__ == "__main__":
    main()
