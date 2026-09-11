"""Smoke: one train/backward/checkpoint cycle on synthetic images. Never a real result."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.classify.next_architecture.config import load_next_config
from src.classify.next_architecture.trainer import train_one_run


def _tiny_split(tmp_path: Path, n: int = 8) -> None:
    rows = []
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(n):
        path = img_dir / f"im{i}.png"
        Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8) + (i * 3)).save(path)
        split = "train" if i < n // 2 else "val"
        rows.append(
            {
                "image_path": str(path),
                "label": [0, 1, 2, 0, 2, 1, 0, 2][i % 8],
                "source": "plus" if i % 2 == 0 else "farabi",
                "group_id": f"g{i // 2}",
                "patient_id": f"p{i // 2}",
                "exam_id": f"e{i}",
                "identity_level": "test",
                "split": split,
            }
        )
    frame = pd.DataFrame(rows)
    splits = tmp_path / "splits"
    splits.mkdir()
    frame.to_csv(splits / "all.csv", index=False)
    frame[frame.split == "train"].to_csv(splits / "train.csv", index=False)
    frame[frame.split == "val"].to_csv(splits / "val.csv", index=False)
    # disjoint dummy test
    tpath = img_dir / "test0.png"
    Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8) + 9).save(tpath)
    pd.DataFrame(
        [
            {
                "image_path": str(tpath),
                "label": 0,
                "source": "farfum_rop",
                "group_id": "gtest",
                "patient_id": "ptest",
                "exam_id": "etest",
                "identity_level": "test",
                "split": "test",
            }
        ]
    ).to_csv(splits / "test.csv", index=False)


def test_smoke_e2_forward_backward_resume(tmp_path, monkeypatch) -> None:
    pytest = __import__("pytest")
    _tiny_split(tmp_path)
    # Point config paths at tmp split; skip canonical hash via allow_noncanonical
    cfg = load_next_config()
    cfg["paths"]["splits_dir"] = tmp_path / "splits"
    cfg["paths"]["results_dir"] = tmp_path / "results"
    cfg["backbone"] = "resnet18"
    cfg["resolution"] = 32
    cfg["lr"] = 1e-3
    cfg["smoke"] = {"n_train": 4, "n_val": 4, "epochs": 1, "batches": 2, "batch_size": 2}
    cfg["expected_split_sha256"] = "will-not-match"
    # Patch verify by allowing noncanonical
    from src.classify.next_architecture import trainer as T

    monkeypatch.setattr(T, "verify_canonical_split", lambda cfg, allow_noncanonical=False: "noncanonical-test")
    monkeypatch.setattr(T, "assert_output_namespace", lambda path: path.mkdir(parents=True, exist_ok=True))
    r1 = train_one_run(
        cfg,
        experiment="E2",
        resolution=32,
        seed=0,
        smoke_test=True,
        allow_test_evaluation=False,
        allow_noncanonical=True,
        resume=False,
    )
    assert r1["smoke_test"] is True
    assert r1["test"]["skipped"] is True
    out = tmp_path / "results"
    ckpts = list(out.rglob("train.pt"))
    assert ckpts
    r2 = train_one_run(
        cfg,
        experiment="E2",
        resolution=32,
        seed=0,
        smoke_test=True,
        allow_test_evaluation=False,
        allow_noncanonical=True,
        resume=True,
    )
    assert r2["smoke_test"] is True
    assert (tmp_path / "results").exists()
    # smoke must not be mistaken for an experiment
    payload = r1
    assert payload.get("smoke_test") is True
