"""Provenance blocks for next-architecture runs."""
from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from src.classify.next_architecture.config import dump_resolved_config, resolved_config_sha
from src.utils.common import ROOT, sha256_file


def _git_state() -> tuple[str | None, bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    commit_id = commit.stdout.strip() if commit.returncode == 0 else None
    if commit_id == "HEAD":
        commit_id = None
    dirty_flag = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    return commit_id, dirty_flag


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def build_run_provenance(
    cfg: dict[str, Any],
    *,
    experiment_id: str,
    resolution: int,
    seed: int,
    fold: str,
    split_sha256: str,
    fold_manifest_sha256: str | None,
    n_train: int,
    n_val: int,
    train_groups: int,
    val_groups: int,
    source_class_counts: dict,
    device: str,
    checkpoint_path: Path | None = None,
    seg_checkpoint: Path | None = None,
    vessel_manifest: Path | None = None,
    best_epoch: int | None = None,
    early_stopping_metric: str | None = None,
    val_threshold: float | None = None,
    calibration_method: str | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    commit, dirty = _git_state()
    provenance = {
        "experiment_id": experiment_id,
        "model_name": cfg.get("backbone"),
        "resolution": int(resolution),
        "seed": int(seed),
        "fold": fold,
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "canonical_split_sha256": split_sha256,
        "fold_manifest_sha256": fold_manifest_sha256,
        "config_path": str(cfg["_config_path"]),
        "config_file_sha256": cfg["_config_sha256"],
        "resolved_config_sha256": resolved_config_sha(cfg),
        "resolved_config": json_safe_config(cfg),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "train_groups": int(train_groups),
        "val_groups": int(val_groups),
        "source_class_counts": source_class_counts,
        "python_version": platform.python_version(),
        "torch_version": package_version("torch"),
        "timm_version": package_version("timm"),
        "device": device,
        "best_epoch": best_epoch,
        "early_stopping_metric": early_stopping_metric,
        "val_threshold": val_threshold,
        "calibration_method": calibration_method,
        "smoke_test": bool(smoke_test),
        "allow_test_evaluation": bool(cfg.get("allow_test_evaluation", False)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_note": (
            "Development decisions must use train/val only. Canonical test is not pristine."
        ),
    }
    if checkpoint_path and Path(checkpoint_path).exists():
        provenance["checkpoint_sha256"] = sha256_file(checkpoint_path)
        provenance["checkpoint_path"] = str(Path(checkpoint_path).resolve())
    if seg_checkpoint and Path(seg_checkpoint).exists():
        provenance["segmentation_checkpoint_sha256"] = sha256_file(seg_checkpoint)
    if vessel_manifest and Path(vessel_manifest).exists():
        provenance["vessel_manifest_sha256"] = sha256_file(vessel_manifest)
    return provenance


def json_safe_config(cfg: dict[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(dump_resolved_config(cfg))
