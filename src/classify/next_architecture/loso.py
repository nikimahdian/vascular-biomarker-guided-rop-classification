"""Leave-one-source-out splits and training. Holdout labels never choose threshold or early-stop."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.classify.next_architecture.folds import assert_no_group_overlap, assert_no_test_images
from src.classify.next_architecture.trainer import train_one_run


def build_loso_splits(
    *,
    all_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout: str,
    out_dir: Path,
) -> dict[str, int]:
    holdout = str(holdout)
    train_in = train_df[train_df["source"].astype(str) != holdout].copy()
    val_in = val_df[val_df["source"].astype(str) != holdout].copy()
    hold = all_df[all_df["source"].astype(str) == holdout].copy()
    if len(train_in) == 0 or len(val_in) == 0 or len(hold) == 0:
        raise RuntimeError(f"LOSO split empty for holdout={holdout}")
    if holdout in set(train_in["source"].astype(str)) or holdout in set(val_in["source"].astype(str)):
        raise RuntimeError(f"Holdout source leaked into train/val: {holdout}")
    assert_no_group_overlap(train_in, val_in)
    assert_no_test_images(pd.concat([train_in, val_in], ignore_index=True), test_df)
    overlap_groups = set(train_in["group_id"]) & set(hold["group_id"])
    if overlap_groups:
        raise RuntimeError(f"Holdout groups in train: {sorted(overlap_groups)[:5]}")
    overlap_groups = set(val_in["group_id"]) & set(hold["group_id"])
    if overlap_groups:
        raise RuntimeError(f"Holdout groups in val: {sorted(overlap_groups)[:5]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_in.to_csv(out_dir / "train.csv", index=False)
    val_in.to_csv(out_dir / "val.csv", index=False)
    hold.to_csv(out_dir / "holdout.csv", index=False)
    return {
        "n_train": int(len(train_in)),
        "n_val": int(len(val_in)),
        "n_holdout": int(len(hold)),
        "n_train_groups": int(train_in["group_id"].nunique()),
        "n_val_groups": int(val_in["group_id"].nunique()),
        "n_holdout_groups": int(hold["group_id"].nunique()),
    }


def run_loso_training(
    cfg: dict[str, Any],
    *,
    experiment: str,
    resolution: int,
    seed: int,
    smoke_test: bool = False,
    resume: bool = True,
) -> int:
    splits = cfg["paths"]["splits_dir"]
    all_df = pd.read_csv(splits / "all.csv")
    train_df = pd.read_csv(splits / "train.csv")
    val_df = pd.read_csv(splits / "val.csv")
    test_df = pd.read_csv(splits / "test.csv")
    sources = sorted(all_df["source"].astype(str).unique())
    base = Path(cfg["paths"]["results_dir"]) / "loso" / experiment / f"res{resolution}" / f"seed{seed}"
    print(
        "[loso] Holdout labels used only for reporting after training. "
        "Early-stop + Youden from in-domain val. Canonical test stays locked."
    )
    for holdout in sources:
        side = base / str(holdout)
        counts = build_loso_splits(
            all_df=all_df,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            holdout=holdout,
            out_dir=side,
        )
        print(f"[loso] holdout={holdout} {counts}")
        if smoke_test:
            print("[loso-smoke] splits written; skip training.")
            continue
        train_one_run(
            cfg,
            experiment=experiment,
            resolution=resolution,
            seed=seed,
            smoke_test=False,
            allow_test_evaluation=False,
            resume=resume,
            split_dir=side,
            fold_id=f"loso_{holdout}",
            holdout_eval_csv=side / "holdout.csv",
        )
    return 0
