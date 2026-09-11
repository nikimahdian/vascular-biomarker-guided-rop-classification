"""E8 sampler, stable sigmoid, summary collector, overlay panel. No GPU."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.classify.next_architecture.experiments import (
    BLOCKED_EXPERIMENTS,
    is_source_balanced,
    normalize_experiment,
)
from src.classify.next_architecture.sampling import source_inverse_frequency_weights
from src.classify.next_architecture.summary import build_summary
from src.classify.next_architecture.trainer import plus_prob_from_logits
from src.classify.next_architecture.vessel_audit import _overlay_panel


def test_e8_is_source_balanced_e6_e7_blocked() -> None:
    assert is_source_balanced("E8")
    assert not is_source_balanced("E2")
    assert normalize_experiment("e8") == "E8"
    with pytest.raises(SystemExit, match="E6 blocked"):
        normalize_experiment("E6")
    with pytest.raises(SystemExit, match="E7 blocked"):
        normalize_experiment("E7")
    assert "E6" in BLOCKED_EXPERIMENTS


def test_inverse_frequency_balances_expected_draw() -> None:
    sources = ["plus"] * 90 + ["farabi"] * 10
    weights, stats = source_inverse_frequency_weights(sources)
    assert stats["n"] == 100
    assert stats["n_sources"] == 2
    assert stats["source_counts"] == {"farabi": 10, "plus": 90}
    assert stats["expected_draw_fraction"]["plus"] == 0.5
    plus_w = float(weights[0])
    farabi_w = float(weights[-1])
    assert plus_w == pytest.approx(100 / (2 * 90))
    assert farabi_w == pytest.approx(100 / (2 * 10))
    # Expected mass per source is equal.
    assert 90 * plus_w == pytest.approx(10 * farabi_w)


def test_plus_prob_from_logits_no_overflow() -> None:
    p = plus_prob_from_logits(np.array([800.0, -800.0, 0.0]))
    assert np.isfinite(p).all()
    assert p[0] > 0.999
    assert p[1] < 0.001
    assert p[2] == pytest.approx(0.5)


def test_overlay_panel_three_wide() -> None:
    rgb = np.zeros((32, 40, 3), dtype=np.uint8)
    rgb[:, :] = (10, 20, 30)
    prob = np.linspace(0, 1, 32 * 40, dtype=np.float32).reshape(32, 40)
    mask = np.zeros((32, 40), dtype=np.uint8)
    mask[8:24, 8:32] = 255
    panel = _overlay_panel(rgb, prob, mask, max_side=40)
    assert panel.ndim == 3
    assert panel.shape[1] == 40 * 3
    assert panel.shape[2] == 3


def test_summary_collects_official_and_loso(tmp_path: Path) -> None:
    def _write(exp: str, seed: int, fold: str, auc: float, hold: float | None = None) -> None:
        dest = tmp_path / exp / "res384" / f"seed{seed}" / f"fold{fold}"
        dest.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": exp,
            "resolution": 384,
            "seed": seed,
            "fold": fold,
            "val_raw": {"auc": auc},
            "worst_source_auc": 0.7,
            "smoke_test": False,
            "test": {"skipped": True},
            "holdout": {"skipped": True},
        }
        if hold is not None:
            payload["holdout"] = {
                "skipped": False,
                "n": 100,
                "metrics_raw_at_val_threshold": {"auc": hold},
            }
        (dest / "results.json").write_text(json.dumps(payload) + "\n")

    _write("E2", 42, "official", 0.94)
    _write("E2", 43, "official", 0.96)
    _write("E2", 42, "loso_farabi", 0.95, hold=0.77)
    summary = build_summary(tmp_path)
    assert summary["n_runs"] == 3
    assert summary["all_test_skipped"] is True
    assert summary["e2_official_val"]["n"] == 2
    assert summary["e2_official_val"]["mean"] == pytest.approx(0.95)
    assert summary["loso"][0]["holdout_auc"] == pytest.approx(0.77)
    assert summary["do_not_compare_val_to_branch_b_test_0_928"] is True
