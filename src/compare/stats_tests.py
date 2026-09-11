"""Paired statistical comparison of branch test predictions.

Paired DeLong test + paired bootstrap for delta-AUC between branches evaluated
on the SAME test rows. Pairwise: B-vs-C, A-vs-C, A-vs-B. Bonferroni-adjusted
p-values across the 3 comparisons are reported alongside raw ones.

Outputs:
    results/stats_tests.json

Usage:
    python -m src.compare.stats_tests [--n-boot 2000] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np
from scipy.stats import norm

from src.utils.common import ensure_dirs, load_config, validate_prediction_frame


def _auc_and_placements(y: np.ndarray, s: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """AUC plus DeLong placement vectors V10 (positives) and V01 (negatives)."""
    if len(y) != len(s) or len(y) == 0:
        raise ValueError("DeLong labels and scores must have equal non-zero length.")
    if not np.isfinite(s).all():
        raise ValueError("DeLong scores must be finite.")
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("DeLong requires both positive and negative labels.")
    n_pos, n_neg = len(pos), len(neg)
    eps = 1e-12
    v10 = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (
        pos[:, None] == neg[None, :]
    )
    v01 = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (
        pos[:, None] == neg[None, :]
    )
    auc = float(v10.sum() / max(n_pos * n_neg, eps))
    return auc, v10.mean(axis=1), v01.mean(axis=0)


def _delong_paired(y: np.ndarray, sa: np.ndarray, sb: np.ndarray) -> dict[str, float]:
    """DeLong 1988 covariance for paired AUC difference."""
    auc_a, v10_a, v01_a = _auc_and_placements(y, sa)
    auc_b, v10_b, v01_b = _auc_and_placements(y, sb)
    n_pos, n_neg = len(v10_a), len(v01_a)
    if n_pos < 2 or n_neg < 2:
        return {"delta_auc": auc_a - auc_b, "z": float("nan"), "p": float("nan")}
    var_a = float(np.var(v10_a, ddof=1) / n_pos + np.var(v01_a, ddof=1) / n_neg)
    var_b = float(np.var(v10_b, ddof=1) / n_pos + np.var(v01_b, ddof=1) / n_neg)
    cov_ab = float(
        np.cov(v10_a, v10_b, ddof=1)[0, 1] / n_pos
        + np.cov(v01_a, v01_b, ddof=1)[0, 1] / n_neg
    )
    var = max(var_a + var_b - 2.0 * cov_ab, 0.0)
    delta = auc_a - auc_b
    if var <= 0:
        if np.isclose(delta, 0.0):
            return {
                "delta_auc": float(delta),
                "variance_delta": 0.0,
                "z": 0.0,
                "p": 1.0,
            }
        return {
            "delta_auc": float(delta),
            "variance_delta": 0.0,
            "z": float("nan"),
            "p": float("nan"),
        }
    z = delta / np.sqrt(var)
    p = 2.0 * norm.sf(abs(z))
    return {
        "delta_auc": float(delta),
        "variance_delta": float(var),
        "z": float(z),
        "p": float(p),
    }


def _auc_boot(y: np.ndarray, s: np.ndarray, idx: np.ndarray) -> float:
    yt, st = y[idx], s[idx]
    pos, neg = st[yt == 1], st[yt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).astype(float) + 0.5 * (diff == 0)).mean())


def _bootstrap_paired(
    y: np.ndarray,
    sa: np.ndarray,
    sb: np.ndarray,
    n_boot: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    row_indices = np.arange(n)
    unique_groups = None if groups is None else np.unique(groups)
    for _ in range(n_boot):
        if unique_groups is None:
            idx = rng.integers(0, n, size=n)
        else:
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([row_indices[groups == group] for group in sampled])
        d = _auc_boot(y, sa, idx) - _auc_boot(y, sb, idx)
        if not np.isnan(d):
            deltas.append(d)
    deltas = np.asarray(deltas)
    if len(deltas) == 0:
        return {
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "sign_probability": float("nan"),
        }
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "sign_probability": float(
            min(max(p, 1.0 / (len(deltas) + 1)), 1.0)
        ),
    }


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res = cfg["paths"]["results_dir"]
    files = {
        "A": res / "branch_a_test_preds.csv",
        "B": res / "branch_b_test_preds.csv",
        "C": res / "branch_c_test_preds.csv",
    }
    loaded = {}
    for tag, path in files.items():
        if path.exists():
            loaded[tag] = __import__("pandas").read_csv(path)
            validate_prediction_frame(loaded[tag], "test", cfg)
    if len(loaded) < 2:
        raise SystemExit("Need at least two branch prediction files.")

    base_tag = next(iter(loaded))
    canonical = __import__("pandas").read_csv(
        cfg["paths"]["splits_dir"] / "test.csv"
    )[["image_path", "group_id"]]
    merged = loaded[base_tag][["image_path", "label"]].rename(
        columns={"label": "label_ref"}
    )
    merged = merged.merge(canonical, on="image_path", validate="one_to_one")
    score_cols = {}
    for tag, df in loaded.items():
        col = f"p_plus_branch_{tag.lower()}"
        merged = merged.merge(df[["image_path", col]], on="image_path", how="inner")
        score_cols[tag] = col
    for tag, df in loaded.items():
        chk = df.set_index("image_path")["label"]
        ref = merged.set_index("image_path")["label_ref"]
        assert (chk.loc[ref.index] == ref).all(), f"label mismatch across files for branch {tag}"

    y = (merged["label_ref"].astype(int) == 2).astype(int).values
    print(f"[stats] paired rows: {len(y)}  positives(Plus): {int(y.sum())}")

    out: dict[str, dict] = {}
    pairs = list(combinations(sorted(score_cols), 2))
    for i, (xa, xb) in enumerate(pairs):
        sa = merged[score_cols[xa]].values.astype(float)
        sb = merged[score_cols[xb]].values.astype(float)
        dl = _delong_paired(y, sa, sb)
        bs = _bootstrap_paired(
            y,
            sa,
            sb,
            args.n_boot,
            args.seed,
            groups=merged["group_id"].values,
        )
        key = f"{xa}_vs_{xb}"
        out[key] = {
            f"auc_{xa}": _auc_boot(y, sa, np.arange(len(y))),
            f"auc_{xb}": _auc_boot(y, sb, np.arange(len(y))),
            **{f"delong_{k}": v for k, v in dl.items()},
            **{f"boot_{k}": v for k, v in bs.items()},
        }
        out[key]["p_bonferroni"] = float(min(dl["p"] * len(pairs), 1.0))
        print(
            f"{key}: dAUC={dl['delta_auc']:+.4f} delong_p={dl['p']:.2e} "
            f"boot_sign={bs['sign_probability']:.2e} "
            f"CI=[{bs['ci_low']:+.4f},{bs['ci_high']:+.4f}] "
            f"bonf_p={out[key]['p_bonferroni']:.2e}"
        )

    payload = {
        "n_test_rows": int(len(y)),
        "n_plus": int(y.sum()),
        "method": (
            "image-level paired DeLong + paired group-cluster bootstrap; "
            "Farabi groups are exams, not verified patients"
        ),
        "bootstrap_unit": "group_id",
        "delong_unit": "image (correlation limitation documented)",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "comparisons": out,
    }
    with open(res / "stats_tests.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] -> {res / 'stats_tests.json'}")


if __name__ == "__main__":
    main()
