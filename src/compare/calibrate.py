"""Post-hoc calibration (temperature scaling) for branch Plus-OvR probabilities.

Fits one temperature per branch on VALIDATION scores by minimizing NLL of the
binary Plus-vs-rest problem, applies it to test scores, and reports ECE /
Brier / NLL before and after. Temperature scaling is monotonic, so AUC is
unchanged by construction (asserted within float tolerance).

Outputs:
    results/calibration_summary.json
    results/calibration_reliability.png

Usage:
    python -m src.compare.calibrate
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

from src.utils.common import (
    ensure_dirs,
    load_config,
    plus_class_index,
    validate_prediction_frame,
)

BRANCHES = {
    "A": ("branch_a_val_preds.csv", "branch_a_test_preds.csv", "p_plus_branch_a"),
    "B": ("branch_b_val_preds.csv", "branch_b_test_preds.csv", "p_plus_branch_b"),
    "C": ("branch_c_val_preds.csv", "branch_c_test_preds.csv", "p_plus_branch_c"),
}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _nll(y: np.ndarray, q: np.ndarray) -> float:
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def _ece(y: np.ndarray, q: np.ndarray, n_bins: int = 15) -> float:
    bins = np.minimum((q * n_bins).astype(int), n_bins - 1)
    ece, total = 0.0, len(y)
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / total) * abs(y[m].mean() - q[m].mean())
    return float(ece)


def _brier(y: np.ndarray, q: np.ndarray) -> float:
    return float(((q - y) ** 2).mean())


def _fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Calibration labels and probabilities must have equal length.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Calibration probabilities must be finite and within [0, 1].")
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration validation labels must contain both classes.")
    z = _logit(p)

    def obj(log_t: float) -> float:
        return _nll(y, _sigmoid(z / np.exp(log_t)))

    r = minimize_scalar(obj, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    if not r.success:
        raise RuntimeError(f"Temperature optimization failed: {r.message}")
    return float(np.exp(r.x))


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write reliability plot (default: on)",
    )
    args = ap.parse_args()

    res = cfg["paths"]["results_dir"]
    plus_idx = plus_class_index(cfg)
    summary: dict[str, dict] = {}
    fig, axes = plt.subplots(1, len(BRANCHES), figsize=(5 * len(BRANCHES), 4.6), squeeze=False)

    for k, (tag, (vname, tname, col)) in enumerate(BRANCHES.items()):
        vp, tp = res / vname, res / tname
        if not (vp.exists() and tp.exists()):
            print(f"[skip] Branch {tag}: missing {vname if not vp.exists() else tname}")
            continue
        dv = pd.read_csv(vp)
        dt = pd.read_csv(tp)
        validate_prediction_frame(dv, "val", cfg)
        validate_prediction_frame(dt, "test", cfg)
        yv = (dv["label"].astype(int) == plus_idx).astype(int).values
        yt = (dt["label"].astype(int) == plus_idx).astype(int).values
        pv = dv[col].values.astype(float)
        pt = dt[col].values.astype(float)

        temp = _fit_temperature(yv, pv)
        pt_cal = _sigmoid(_logit(pt) / temp)

        auc_raw, auc_cal = roc_auc_score(yt, pt), roc_auc_score(yt, pt_cal)
        drift = abs(auc_raw - auc_cal)
        assert drift < 1e-6, f"AUC changed after temperature scaling ({drift})"

        summary[tag] = {
            "temperature": temp,
            "test_ece_raw": _ece(yt, pt),
            "test_ece_calibrated": _ece(yt, pt_cal),
            "test_brier_raw": _brier(yt, pt),
            "test_brier_calibrated": _brier(yt, pt_cal),
            "test_nll_raw": _nll(yt, pt),
            "test_nll_calibrated": _nll(yt, pt_cal),
            "test_auc": float(auc_raw),
        }
        s = summary[tag]
        print(
            f"[{tag}] T={temp:.3f} ECE {s['test_ece_raw']:.4f}->{s['test_ece_calibrated']:.4f} "
            f"Brier {s['test_brier_raw']:.4f}->{s['test_brier_calibrated']:.4f} "
            f"NLL {s['test_nll_raw']:.4f}->{s['test_nll_calibrated']:.4f} AUC={auc_raw:.4f}"
        )

        ax = axes[0][k]
        for scores, label, style in [(pt, "raw", "--"), (pt_cal, "calibrated", "-")]:
            frac_pos, mean_pred, edges = [], [], np.linspace(0, 1, 11)
            bins_idx = np.minimum((scores * 10).astype(int), 9)
            for b in range(10):
                m = bins_idx == b
                if m.sum():
                    mean_pred.append(scores[m].mean())
                    frac_pos.append(yt[m].mean())
            ax.plot(mean_pred, frac_pos, style, marker="o", ms=3, label=label)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(f"Branch {tag} (T={temp:.2f})")
        ax.set_xlabel("predicted P(Plus)")
        ax.set_ylabel("observed frequency")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if args.plot:
        plt.savefig(res / "calibration_reliability.png", dpi=150)
    plt.close()

    with open(res / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] -> {res / 'calibration_summary.json'}")


if __name__ == "__main__":
    main()
