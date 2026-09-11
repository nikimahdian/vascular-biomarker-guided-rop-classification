"""Unit tests for next-architecture ladder helpers. No GPU, no real training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.classify.next_architecture.ladder import (
    e4_beats_e3,
    missing_probability_maps,
    pick_e1_resolution,
    prob_map_path,
    run_status,
)


def _fake_run(root: Path, experiment: str, resolution: int, seed: int, auc: float | None, *, train_pt: bool = False) -> Path:
    run = root / experiment / f"res{resolution}" / f"seed{seed}" / "foldofficial_deadbeef"
    run.mkdir(parents=True, exist_ok=True)
    if auc is not None:
        (run / "results.json").write_text(
            json.dumps({"experiment": experiment, "resolution": resolution, "seed": seed, "val_raw": {"auc": auc}})
            + "\n"
        )
    if train_pt:
        (run / "train.pt").write_bytes(b"not-a-real-checkpoint")
    return run


def test_run_status_complete_over_resume(tmp_path: Path) -> None:
    done = _fake_run(tmp_path, "E0", 224, 42, 0.94)
    _fake_run(tmp_path, "E0", 224, 43, None, train_pt=True)
    path, status = run_status(tmp_path, "E0", 224, 42)
    assert status == "complete"
    assert path == done
    path43, status43 = run_status(tmp_path, "E0", 224, 43)
    assert status43 == "resume"
    assert path43 is not None
    path44, status44 = run_status(tmp_path, "E0", 224, 44)
    assert status44 == "missing"
    assert path44 is None


def test_pick_e1_resolution_prefers_higher_auc_then_lower_res(tmp_path: Path) -> None:
    _fake_run(tmp_path, "E1", 384, 42, 0.91)
    _fake_run(tmp_path, "E1", 456, 42, 0.93)
    pick = pick_e1_resolution(tmp_path, seed=42)
    assert pick["winner_resolution"] == 456
    _fake_run(tmp_path, "E1", 384, 43, 0.95)
    _fake_run(tmp_path, "E1", 456, 43, 0.95)
    pick_tie = pick_e1_resolution(tmp_path, seed=43)
    assert pick_tie["winner_resolution"] == 384


def test_e4_gate_strict_greater() -> None:
    assert e4_beats_e3(0.90, 0.91) is True
    assert e4_beats_e3(0.90, 0.90) is False
    assert e4_beats_e3(0.91, 0.90) is False


def test_missing_probability_maps(tmp_path: Path) -> None:
    index = tmp_path / "all.csv"
    prob_dir = tmp_path / "prob"
    prob_dir.mkdir()
    img_a = "/data/a.png"
    img_b = "/data/b.png"
    pd.DataFrame({"image_path": [img_a, img_b]}).to_csv(index, index=False)
    npy_a = prob_map_path(prob_dir, img_a)
    np.save(npy_a, np.zeros((4, 4), dtype=np.float16))
    missing = missing_probability_maps(index, prob_dir)
    assert missing == [img_b]
