"""Shared IO, metrics and statistics for the expert-validation analysis suite.

Import conventions
------------------
Every script in this directory takes:
  --key      expert_validation/blinding_key.csv      (private; study_id -> image_path, label, ...)
  --root     expert_validation/                       (masks live under <root>/grader_X/...)
  --out      a results directory

Design rules this module enforces
---------------------------------
* Every metric is computed on masks at NATIVE resolution, never on a resized copy.
* Tiered vessel recall is defined from the EXPERT mask's local width, never from the automatic one.
* ICC is ICC(2,1): two-way random effects, single measurement, absolute agreement, with a 95 % CI.
* Bland-Altman is always reported alongside any ICC. Correlation is not agreement.
* Every confidence interval is a CLUSTER bootstrap over group_id, never an image-level bootstrap.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats
from skimage.morphology import skeletonize

PRIMARY_BIOMARKERS = ["vessel_density", "width_p90_dd", "tort_p90"]
TIER_NAMES = ("thin", "medium", "thick")


# --------------------------------------------------------------------------- IO

def load_key(path: str | Path) -> pd.DataFrame:
    k = pd.read_csv(path)
    need = {"study_id", "cohort", "image_path", "group_id"}
    missing = need - set(k.columns)
    if missing:
        raise ValueError(f"blinding key missing columns: {sorted(missing)}")
    return k


def read_mask(path: str | Path) -> np.ndarray:
    """Binary mask. Accepts 0/255 PNG or 0/1; anything else is an error, not a silent threshold."""
    from PIL import Image
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    u = np.unique(a)
    if not set(np.unique(u).tolist()) <= {0, 1, 255}:
        raise ValueError(f"{path}: mask values must be in {{0,1,255}}, got {u[:8]}")
    return (a > 0)


def image_size(path: str | Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (w, h)


# --------------------------------------------------------------------------- segmentation metrics

def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    s = pred.sum() + gt.sum()
    return float(2 * (pred & gt).sum() / s) if s else 1.0


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    u = (pred | gt).sum()
    return float((pred & gt).sum() / u) if u else 1.0


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    tp = int((pred & gt).sum())
    p = tp / int(pred.sum()) if pred.sum() else float("nan")
    r = tp / int(gt.sum()) if gt.sum() else float("nan")
    return float(p), float(r)


def cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Centerline Dice (Shit et al.). Topology-aware: an overlap metric can stay high while the
    thin network is lost, which is exactly the failure mode observed on the second camera."""
    sp, sg = skeletonize(pred), skeletonize(gt)
    if sp.sum() == 0 or sg.sum() == 0:
        return float("nan")
    t_prec = float((sp & gt).sum() / sp.sum())
    t_sens = float((sg & pred).sum() / sg.sum())
    return float(2 * t_prec * t_sens / (t_prec + t_sens)) if (t_prec + t_sens) else float("nan")


def local_width(mask: np.ndarray) -> np.ndarray:
    """Full width at every vessel pixel, in pixels (2 * EDT)."""
    return 2.0 * ndi.distance_transform_edt(mask)


def tiered_centerline_recall(pred: np.ndarray, gt: np.ndarray,
                             names: tuple[str, ...] = TIER_NAMES) -> dict[str, float]:
    """Fraction of the EXPERT centreline recovered by `pred`, split by the EXPERT local width.

    Tiers are RANK-based thirds of the expert's own skeleton width (lower third / middle third /
    upper third), not value-based percentiles. A value-based cut degenerates whenever the width
    distribution is bimodal -- a large vessel next to a capillary can push the upper tercile edge
    onto the maximum and leave the top tier empty -- which silently produces a NaN where a number
    is expected. Rank-based thirds are always non-empty and equal-sized.

    The tiers come from the expert mask, never from the automatic one, so the definition cannot
    drift with the quality of the mask being evaluated.
    """
    sg = skeletonize(gt)
    if sg.sum() == 0:
        return {f"recall_{n}": float("nan") for n in names} | {"tier_n": 0}
    w = local_width(gt)[sg]
    covered = pred[sg]
    order = np.argsort(w)
    out = {}
    for n, idx in zip(names, np.array_split(order, len(names))):
        out[f"recall_{n}"] = float(covered[idx].mean()) if len(idx) else float("nan")
    out["tier_width_min"] = float(w.min())
    out["tier_width_max"] = float(w.max())
    out["tier_n"] = int(sg.sum())
    return out


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p, r = precision_recall(pred, gt)
    d = {"dice": dice(pred, gt), "iou": iou(pred, gt), "precision": p, "recall": r,
         "expert_vessel_fraction": float(gt.mean()),
         "pred_vessel_fraction": float(pred.mean())}
    d |= tiered_centerline_recall(pred, gt)
    d["cldice"] = cldice(pred, gt)
    return d


# --------------------------------------------------------------------------- disc metrics

def disc_geometry(mask: np.ndarray) -> dict[str, float]:
    if mask.sum() == 0:
        return {"area_px": 0.0, "radius_px": float("nan"), "cx": float("nan"), "cy": float("nan")}
    ys, xs = np.nonzero(mask)
    return {"area_px": float(mask.sum()), "radius_px": float(np.sqrt(mask.sum() / np.pi)),
            "cx": float(xs.mean()), "cy": float(ys.mean())}


def disc_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    g, p = disc_geometry(gt), disc_geometry(pred)
    out = {"disc_dice": dice(pred, gt), "disc_iou": iou(pred, gt)}
    if g["area_px"] > 0 and p["area_px"] > 0:
        dd = 2 * g["radius_px"]
        out["centre_error_px"] = float(np.hypot(p["cx"] - g["cx"], p["cy"] - g["cy"]))
        out["centre_error_dd"] = out["centre_error_px"] / dd if dd else float("nan")
        out["diameter_ratio"] = float(p["radius_px"] / g["radius_px"]) if g["radius_px"] else float("nan")
    else:
        out |= {"centre_error_px": float("nan"), "centre_error_dd": float("nan"),
                "diameter_ratio": float("nan")}
    return out


# --------------------------------------------------------------------------- agreement statistics

def icc21(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """ICC(2,1) point estimate. Two-way random effects, single measurement, absolute agreement.

    Absolute agreement rather than consistency: a systematic offset is a real measurement error and
    a consistency ICC would hide it. Single rather than average measurement: the clinical use is a
    single measurement per image.

    The confidence interval is deliberately NOT analytic here. The familiar closed-form limits are
    easy to get subtly wrong, and a wrong version produces a negative upper limit when the residual
    mean square collapses (which happens for two nearly identical measurement columns). The CI is
    obtained from `cluster_bootstrap_ci` instead, which also respects patient/exam grouping.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    n = len(x)
    if n < 4:
        return {"icc": float("nan"), "n": n}
    d = np.stack([x, y], axis=1).astype(float)
    k = 2
    gm = d.mean()
    msr = k * ((d.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((d.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    sse = ((d - d.mean(axis=1, keepdims=True) - d.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return {"icc": float((msr - mse) / den) if den else float("nan"), "n": int(n),
            "msr": float(msr), "msc": float(msc), "mse": float(mse)}


def icc21_ci(df: pd.DataFrame, col_a: str, col_b: str, n_boot: int = 5000, seed: int = 20260120,
             group_col: str = "group_id") -> dict[str, float]:
    """ICC(2,1) with a CI from a bootstrap that resamples GROUPS.

    This is the interval that gets reported. The point estimate comes from the full sample; the
    limits are percentiles of the resampled distribution.
    """
    point = icc21(df[col_a].to_numpy(float), df[col_b].to_numpy(float))["icc"]
    r = cluster_bootstrap_ci(df, lambda d: icc21(d[col_a].to_numpy(float),
                                                 d[col_b].to_numpy(float))["icc"],
                             n_boot=n_boot, seed=seed, group_col=group_col)
    r["point"] = float(point)
    return r


def bland_altman(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Bias and 95 % limits of agreement for `x - y`. Reported next to every ICC."""
    m = np.isfinite(x) & np.isfinite(y)
    d = np.asarray(x)[m] - np.asarray(y)[m]
    n = len(d)
    if n < 2:
        return {"n": n, "bias": float("nan"), "sd": float("nan"),
                "loa_low": float("nan"), "loa_high": float("nan")}
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return {"n": int(n), "bias": bias, "sd": sd,
            "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
            "bias_ci_low": bias - stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n),
            "bias_ci_high": bias + stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)}


def cohen_kappa(a: np.ndarray, b: np.ndarray, weights: str | None = None) -> float:
    """Cohen's kappa; `weights='quadratic'` for an ordinal scale."""
    lab = sorted(set(np.asarray(a).tolist()) | set(np.asarray(b).tolist()))
    idx = {l: i for i, l in enumerate(lab)}
    k = len(lab)
    O = np.zeros((k, k), dtype=float)
    for u, v in zip(a, b):
        O[idx[u], idx[v]] += 1
    n = O.sum()
    if n == 0:
        return float("nan")
    O /= n
    r, c = O.sum(1), O.sum(0)
    E = np.outer(r, c)
    if weights is None:
        W = 1 - np.eye(k)
    else:
        i = np.arange(k)[:, None]
        j = np.arange(k)[None, :]
        W = (i - j) ** 2 if weights == "quadratic" else np.abs(i - j)
    num = (W * O).sum()
    den = (W * E).sum()
    return float(1 - num / den) if den else float("nan")


# --------------------------------------------------------------------------- bootstrap

def cluster_bootstrap_ci(df: pd.DataFrame, statistic, n_boot: int = 5000, seed: int = 20260120,
                         alpha: float = 0.05, group_col: str = "group_id") -> dict[str, float]:
    """Percentile CI from a bootstrap that resamples GROUPS, not images.

    Resampling images would treat repeat photographs of one infant as independent observations.
    Every replicate draws groups with replacement and takes all images belonging to them.
    """
    groups = df[group_col].astype(str).unique()
    if len(groups) < 3:
        return {"point": float(statistic(df)), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot": 0, "n_groups": len(groups)}
    rng = np.random.default_rng(seed)
    by = {g: sub for g, sub in df.groupby(group_col)}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        rep = pd.concat([by[g] for g in pick], ignore_index=True)
        try:
            v = float(statistic(rep))
        except Exception:  # noqa: BLE001
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    if len(vals) < 50:
        return {"point": float(statistic(df)), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot": int(len(vals)), "n_groups": len(groups)}
    return {"point": float(statistic(df)),
            "ci_low": float(np.percentile(vals, 100 * alpha / 2)),
            "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "n_boot": int(len(vals)), "n_groups": int(len(groups))}


def cluster_bootstrap_paired_ci(df: pd.DataFrame, statistic, n_boot: int = 5000, seed: int = 20260120,
                                alpha: float = 0.05, group_col: str = "group_id") -> dict[str, float]:
    """Paired version: resample groups once and evaluate `statistic` on both paired columns.

    Used for AUC(C) - AUC(B), where the two models must see the same resampled groups.
    """
    groups = df[group_col].astype(str).unique()
    if len(groups) < 3:
        return {"point": float(statistic(df)), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot": 0, "n_groups": len(groups)}
    rng = np.random.default_rng(seed)
    by = {g: sub for g, sub in df.groupby(group_col)}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        rep = pd.concat([by[g] for g in pick], ignore_index=True)
        try:
            v = float(statistic(rep))
        except Exception:  # noqa: BLE001
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    if len(vals) < 50:
        return {"point": float(statistic(df)), "ci_low": float(np.nan), "ci_high": float(np.nan),
                "n_boot": int(len(vals)), "n_groups": len(groups)}
    return {"point": float(statistic(df)),
            "ci_low": float(np.percentile(vals, 100 * alpha / 2)),
            "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "n_boot": int(len(vals)), "n_groups": int(len(groups))}


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 20260120,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Plain percentile CI on an already-clustered quantity (e.g. a per-image metric list)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2)))


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(score)
    y, score = np.asarray(y)[m], np.asarray(score)[m]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


warnings.filterwarnings("ignore", category=RuntimeWarning)
