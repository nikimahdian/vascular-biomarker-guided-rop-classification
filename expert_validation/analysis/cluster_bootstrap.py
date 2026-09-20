"""Cluster bootstrap utilities.

Two jobs:
  1. provide the resampling machinery to the rest of the analysis suite (re-exported from _common)
  2. compute the two confidence intervals the thesis still needs and that do NOT depend on the
     expert annotations: the farabi LOSO AUC for Branch B, and the paired C - B difference.

Why grouping matters: images from one infant are not independent. Resampling images would shrink
the interval artificially. Every replicate here resamples GROUPS with replacement and takes all
images belonging to the sampled groups.

Usage
-----
  python cluster_bootstrap.py --loso results/loo_results_full.json --out out/loso_ci.csv
  python cluster_bootstrap.py --predictions results/loo --out out/loso_ci.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import cluster_bootstrap_ci, cluster_bootstrap_paired_ci, safe_auc


def _auc_stat(df: pd.DataFrame, col: str):
    return lambda d: safe_auc((d["label"] == 2).astype(int).to_numpy(), d[col].to_numpy())


def auc_ci(df: pd.DataFrame, col: str, n_boot: int = 5000, seed: int = 20260120) -> dict:
    return cluster_bootstrap_ci(df, _auc_stat(df, col), n_boot=n_boot, seed=seed)


def paired_delta_ci(df: pd.DataFrame, col_a: str, col_b: str, n_boot: int = 5000,
                    seed: int = 20260120) -> dict:
    """AUC(col_a) - AUC(col_b) on the same resampled groups."""

    def stat(d):
        return safe_auc((d["label"] == 2).astype(int).to_numpy(), d[col_a].to_numpy()) - \
               safe_auc((d["label"] == 2).astype(int).to_numpy(), d[col_b].to_numpy())

    return cluster_bootstrap_paired_ci(df, stat, n_boot=n_boot, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loso", default=None,
                    help="path to loo_results_full.json (point estimates only)")
    ap.add_argument("--predictions", default=None,
                    help="directory holding <holdout>_branch_{a,b,c}_test_preds.csv files, "
                         "which is what CIs actually need")
    ap.add_argument("--out", default="loso_ci.csv")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260120)
    args = ap.parse_args()

    if args.predictions:
        rows = []
        for f in sorted(Path(args.predictions).glob("*_branch_*_test_preds.csv")):
            parts = f.stem.split("_")
            holdout, branch = parts[0], parts[2].upper()
            d = pd.read_csv(f)
            if "label" not in d.columns or "group_id" not in d.columns:
                print(f"[skip] {f.name}: needs label and group_id")
                continue
            cols = {"A": "p_plus_branch_a", "B": "p_plus_branch_b", "C": "p_plus_branch_c"}
            col = cols.get(branch)
            if col is None or col not in d.columns:
                print(f"[skip] {f.name}: no {col}")
                continue
            r = auc_ci(d, col, n_boot=args.n_boot, seed=args.seed)
            rows.append(dict(holdout=holdout, branch=branch, metric="auc", **r))
            print(f"{holdout:12s} {branch}  AUC {r['point']:.4f} "
                  f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]  groups={r['n_groups']}")

        # paired C - B on the same hold-out
        for holdout in sorted({p.name.split("_")[0] for p in
                               Path(args.predictions).glob("*_branch_*_test_preds.csv")}):
            def load(br):
                f = Path(args.predictions) / f"{holdout}_branch_{br}_test_preds.csv"
                return pd.read_csv(f) if f.exists() else None
            a, c = load("b"), load("c")
            if a is None or c is None:
                continue
            m = a[["image_path", "label", "group_id", "p_plus_branch_b"]].merge(
                c[["image_path", "p_plus_branch_c"]], on="image_path", how="inner")
            r = paired_delta_ci(m, "p_plus_branch_c", "p_plus_branch_b",
                                n_boot=args.n_boot, seed=args.seed)
            rows.append(dict(holdout=holdout, branch="C-B", metric="delta_auc", **r))
            sig = "excludes 0" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "includes 0"
            print(f"{holdout:12s} C-B  {r['point']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  {sig}")

        out = pd.DataFrame(rows)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"\n[done] -> {args.out}")
        return

    if args.loso:
        d = json.load(open(args.loso))
        res = d.get("results", d)
        df = pd.DataFrame([{"holdout": r["holdout"], "branch": r["branch"],
                            "auc": r["plus_ovr"]["auc"]} for r in res])
        piv = df.pivot(index="holdout", columns="branch", values="auc")
        print(piv.round(4).to_string())
        print("\nPoint estimates only. Confidence intervals need the prediction files:")
        print("  python cluster_bootstrap.py --predictions results/loo --out loso_ci.csv")
        return

    ap.error("give --predictions (for CIs) or --loso (point estimates only)")


if __name__ == "__main__":
    main()
