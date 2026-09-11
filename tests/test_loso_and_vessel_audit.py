"""LOSO split leakage tests and vessel-audit recall on synthetic maps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.classify.next_architecture.loso import build_loso_splits
from src.classify.next_architecture.vessel_audit import _centerline_recall, summarize


def _tiny_cohort(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for i in range(12):
        src = ["plus", "farabi", "farfum_rop"][i % 3]
        split = ["train", "val", "test"][i // 4]
        rows.append(
            {
                "image_path": str(tmp_path / f"{src}_{i}.png"),
                "label": i % 3,
                "source": src,
                "group_id": f"{src}_g{i // 2}",
                "split": split,
            }
        )
    all_df = pd.DataFrame(rows)
    return (
        all_df,
        all_df[all_df.split == "train"].copy(),
        all_df[all_df.split == "val"].copy(),
        all_df[all_df.split == "test"].copy(),
    )


def test_loso_splits_exclude_holdout_source(tmp_path: Path) -> None:
    all_df, tr, va, te = _tiny_cohort(tmp_path)
    out = tmp_path / "loso_farabi"
    counts = build_loso_splits(
        all_df=all_df, train_df=tr, val_df=va, test_df=te, holdout="farabi", out_dir=out
    )
    train_in = pd.read_csv(out / "train.csv")
    val_in = pd.read_csv(out / "val.csv")
    hold = pd.read_csv(out / "holdout.csv")
    assert set(train_in["source"]) == {"plus", "farfum_rop"}
    assert set(val_in["source"]) == {"plus", "farfum_rop"}
    assert set(hold["source"]) == {"farabi"}
    assert counts["n_holdout"] == int((all_df.source == "farabi").sum())
    assert set(train_in["group_id"]).isdisjoint(set(hold["group_id"]))
    assert set(val_in["group_id"]).isdisjoint(set(hold["group_id"]))


def test_centerline_recall_perfect_on_filled_prob() -> None:
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[20:28, 8:40] = 255
    prob = np.ones((48, 48), dtype=np.float32)
    rec = _centerline_recall(prob, mask, 0.20)
    assert rec["n_skel"] > 0
    assert rec["recall"] == 1.0
    assert rec["recall_thin"] == 1.0


def test_centerline_recall_zero_when_prob_empty() -> None:
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[20:28, 8:40] = 255
    prob = np.zeros((48, 48), dtype=np.float32)
    rec = _centerline_recall(prob, mask, 0.20)
    assert rec["n_skel"] > 0
    assert rec["recall"] == 0.0


def test_summarize_empty_rate() -> None:
    frame = pd.DataFrame(
        {
            "source": ["plus", "plus", "farabi"],
            "empty": [False, True, False],
            "coverage": [0.1, 0.0, 0.2],
            "recall": [0.9, np.nan, 0.5],
            "recall_thin": [0.8, np.nan, 0.4],
            "recall_mid": [0.9, np.nan, 0.5],
            "recall_thick": [1.0, np.nan, 0.6],
            "border_mean": [0.01, 0.02, 0.03],
            "has_mask": [True, True, True],
        }
    )
    s = summarize(frame, 0.20)
    assert s["n"] == 3
    assert s["clinical_dice_iou_claim"] is False
    assert abs(s["empty_frequency"] - (1 / 3)) < 1e-9
