"""H1: fit residual combiner on OOF logits only. Validation labels never fit alpha."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _require_columns(frame: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"H1 frame missing {missing}")


def fit_residual_alpha(
    oof: pd.DataFrame,
    *,
    alpha_ridge: float = 10.0,
    logit_rgb: str = "plus_logit",
    logit_vessel: str = "plus_logit_vessel",
    target: str = "plus_target",
) -> dict[str, Any]:
    """z_hybrid = z_rgb + a * z_vessel. Fit `a` on OOF rows only (Ridge, intercept 0)."""
    _require_columns(oof, (logit_rgb, logit_vessel, target))
    if len(oof) == 0:
        raise ValueError("H1 OOF frame is empty")
    z_rgb = oof[logit_rgb].to_numpy(dtype=np.float64)
    z_v = oof[logit_vessel].to_numpy(dtype=np.float64)
    y = oof[target].to_numpy(dtype=np.float64)
    # Residual design: predict (z_target_proxy) via ridge on vessel logit with RGB offset.
    # Use logistic-scale least squares on residual y_centered vs z_v after removing z_rgb rank.
    residual = y - 1.0 / (1.0 + np.exp(-np.clip(z_rgb, -60, 60)))
    model = Ridge(alpha=float(alpha_ridge), fit_intercept=False)
    model.fit(z_v.reshape(-1, 1), residual)
    a = float(model.coef_.ravel()[0])
    return {
        "alpha": a,
        "alpha_ridge": float(alpha_ridge),
        "n_oof": int(len(oof)),
        "formula": "z_hybrid = z_rgb + alpha * z_vessel",
        "fitted_on": "oof_only",
        "validation_labels_used_for_fit": False,
    }


def apply_residual(frame: pd.DataFrame, alpha: float, *, logit_rgb="plus_logit", logit_vessel="plus_logit_vessel") -> np.ndarray:
    z_rgb = frame[logit_rgb].to_numpy(dtype=np.float64)
    z_v = frame[logit_vessel].to_numpy(dtype=np.float64)
    return z_rgb + float(alpha) * z_v


def run_h1(
    oof_csv: Path,
    val_csv: Path,
    dest: Path,
    *,
    alpha_ridge: float = 10.0,
) -> dict[str, Any]:
    oof = pd.read_csv(oof_csv)
    val = pd.read_csv(val_csv)
    fit = fit_residual_alpha(oof, alpha_ridge=alpha_ridge)
    val = val.copy()
    val["h1_logit"] = apply_residual(val, fit["alpha"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    val.to_csv(dest / "h1_val_predictions.csv", index=False)
    payload = {
        **fit,
        "oof_csv": str(oof_csv),
        "val_csv": str(val_csv),
        "canonical_test_used": False,
        "note": "Conditional information test, not a production model.",
    }
    (dest / "h1_fit.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
