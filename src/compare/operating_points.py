"""Clinically framed operating points + source-macro AUC for each branch.

For every branch with val+test prediction files, evaluates three operating
points - all tuned on VALIDATION only, applied unchanged to test:
  youden      : max J = sens + spec - 1
  sens>=0.90  : most specific threshold still reaching target sensitivity on val
  spec>=0.90  : most sensitive threshold still reaching target specificity on val

Reports test sensitivity/specificity/F1/accuracy/PPV/NPV per operating point,
plus a source-macro AUC table (pooled vs unweighted mean of per-source AUCs)
to expose domain dominance in the mixed test set.

Outputs:
    results/operating_points.csv
    results/source_macro_auc.csv

Usage:
    python -m src.compare.operating_points [--sens-target 0.90] [--spec-target 0.90]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils.common import (
    ensure_dirs,
    load_config,
    plus_class_index,
    tune_binary_threshold,
    validate_prediction_frame,
)

BRANCHES = {
    "A": ("branch_a_val_preds.csv", "branch_a_test_preds.csv", "p_plus_branch_a"),
    "B": ("branch_b_val_preds.csv", "branch_b_test_preds.csv", "p_plus_branch_b"),
    "C": ("branch_c_val_preds.csv", "branch_c_test_preds.csv", "p_plus_branch_c"),
}


def _counts(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, float]:
    pred = (s >= thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())

    def safe(num: int, den: int) -> float:
        return num / den if den else float("nan")

    return {
        "threshold": float(thr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "f1": safe(2 * tp, 2 * tp + fp + fn),
        "ppv": safe(tp, tp + fp),
        "npv": safe(tn, tn + fn),
        "accuracy": safe(tp + tn, tp + fp + fn + tn),
    }


def _candidate_thresholds(s: np.ndarray) -> np.ndarray:
    if s.ndim != 1 or not np.isfinite(s).all() or len(s) == 0:
        raise ValueError("Scores must be a non-empty finite 1D array.")
    u = np.unique(s)
    return np.concatenate(
        [
            [np.nextafter(u[-1], np.inf)],
            u[::-1],
            [np.nextafter(u[0], -np.inf)],
        ]
    )


def _pick_threshold(y: np.ndarray, s: np.ndarray, target: float, mode: str) -> float:
    if mode not in {"sens", "spec"}:
        raise ValueError("mode must be 'sens' or 'spec'.")
    if not 0 <= target <= 1:
        raise ValueError("target must be between 0 and 1.")
    cands = _candidate_thresholds(s)
    rows = [_counts(y, s, t) for t in cands]
    ok = [r for r, t in zip(rows, cands) if (r["sensitivity"] >= target if mode == "sens" else r["specificity"] >= target)]
    if not ok:
        return float("nan")
    key = "specificity" if mode == "sens" else "sensitivity"
    return max(ok, key=lambda r: (r[key], r["threshold"]))["threshold"]


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens-target", type=float, default=0.90)
    ap.add_argument("--spec-target", type=float, default=0.90)
    args = ap.parse_args()

    res = cfg["paths"]["results_dir"]
    plus_idx = plus_class_index(cfg)
    op_rows: list[dict] = []
    macro_rows: list[dict] = []

    for tag, (vname, tname, col) in BRANCHES.items():
        vp, tp = res / vname, res / tname
        if not (vp.exists() and tp.exists()):
            print(f"[skip] Branch {tag}: missing preds")
            continue
        dv, dt = pd.read_csv(vp), pd.read_csv(tp)
        validate_prediction_frame(dv, "val", cfg)
        validate_prediction_frame(dt, "test", cfg)
        yv = (dv["label"].astype(int) == plus_idx).astype(int).values
        yt = (dt["label"].astype(int) == plus_idx).astype(int).values
        sv, st = dv[col].values.astype(float), dt[col].values.astype(float)

        thr_youden = tune_binary_threshold(yv, sv, method="youden")
        thr_sens = _pick_threshold(yv, sv, args.sens_target, "sens")
        thr_spec = _pick_threshold(yv, sv, args.spec_target, "spec")

        for name, thr in [
            ("val_youden", thr_youden),
            (f"val_sens_ge_{args.sens_target:.2f}", thr_sens),
            (f"val_spec_ge_{args.spec_target:.2f}", thr_spec),
        ]:
            row = {"branch": tag, "operating_point": name, **_counts(yt, st, thr)}
            op_rows.append(row)
            print(
                f"[{tag}] {name}: thr={thr:.3f} sens={row['sensitivity']:.3f} "
                f"spec={row['specificity']:.3f} f1={row['f1']:.3f} "
                f"ppv={row['ppv']:.3f} npv={row['npv']:.3f}"
            )

        entry = {"branch": tag, "auc_pooled": float(roc_auc_score(yt, st))}
        src_aucs = []
        if "source" in dt.columns:
            for src, sub in dt.groupby("source"):
                ys = (sub["label"].astype(int) == 2).astype(int).values
                ss = sub[col].values.astype(float)
                auc_s = float(roc_auc_score(ys, ss)) if len(np.unique(ys)) > 1 else float("nan")
                entry[f"auc_{src}"] = auc_s
                entry[f"n_{src}"] = int(len(sub))
                if not np.isnan(auc_s):
                    src_aucs.append(auc_s)
        entry["auc_source_macro"] = float(np.mean(src_aucs)) if src_aucs else float("nan")
        entry["pooled_minus_macro"] = entry["auc_pooled"] - entry["auc_source_macro"]
        macro_rows.append(entry)
        print(
            f"[{tag}] pooled={entry['auc_pooled']:.4f} source_macro={entry['auc_source_macro']:.4f}"
        )

    if op_rows:
        pd.DataFrame(op_rows).to_csv(res / "operating_points.csv", index=False)
    if macro_rows:
        pd.DataFrame(macro_rows).to_csv(res / "source_macro_auc.csv", index=False)
    print(f"[done] -> {res / 'operating_points.csv'}, {res / 'source_macro_auc.csv'}")


if __name__ == "__main__":
    main()
