"""Shared trainer for E0–E5 and E8. Test evaluation is off by default."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.classify.next_architecture.config import (
    assert_output_namespace,
    resolved_config_sha,
    short_hash,
    verify_canonical_split,
)
from src.classify.next_architecture.dataset import NextArchitectureDataset, collate_next
from src.classify.next_architecture.experiments import (
    gate_bias_init,
    is_early_fusion_4ch,
    is_gated_fusion,
    is_source_balanced,
    is_vessel_only,
    normalize_experiment,
    uses_multiclass_ce,
    uses_ordinal_aux,
    uses_vessel,
)
from src.classify.next_architecture.folds import assert_no_group_overlap, assert_no_test_images, official_reduced_fold
from src.classify.next_architecture.losses import (
    CoralOrdinalLoss,
    binary_pos_weight_from_labels,
    class_weights_from_labels,
    pairwise_ranking_loss,
)
from src.classify.next_architecture.sampling import make_source_balanced_sampler
from src.classify.next_architecture.metrics import (
    apply_temperature,
    discrimination_bundle,
    fit_temperature,
    grouped_bootstrap_auc,
    high_sensitivity_threshold,
    per_source_bundle,
    plus_binary,
    youden_threshold,
)
from src.classify.next_architecture.models import NextArchitectureNet
from src.classify.next_architecture.provenance import build_run_provenance
from src.utils.common import get_device, set_seed


CANONICAL_FORBIDDEN = ("results/branch_a_", "results/branch_b_", "results/branch_c_", "weights/branch_b_")


def _print_device(requested: str, actual: str) -> None:
    print(f"[device] requested={requested} actual={actual}")
    if requested != "cpu" and actual == "cpu":
        print("[device] WARNING: major ops fell back to CPU. This is not silent; throughput will drop.")


def output_dir(cfg: dict, experiment: str, resolution: int, seed: int, fold: str, smoke: bool) -> Path:
    cfg_hash = short_hash(resolved_config_sha(cfg))
    root = cfg["paths"]["results_dir"]
    if smoke:
        path = root / "smoke" / experiment / f"res{resolution}" / f"seed{seed}" / f"fold{fold}_{cfg_hash}"
    else:
        path = root / experiment / f"res{resolution}" / f"seed{seed}" / f"fold{fold}_{cfg_hash}"
    assert_output_namespace(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@torch.no_grad()
def collect_predictions(model, loader, device, experiment: str) -> dict[str, np.ndarray | list]:
    model.eval()
    plus_logits, labels, paths, groups, sources, gates = [], [], [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        vessel = batch.get("vessel")
        vessel = vessel.to(device) if vessel is not None else None
        out = model(images, vessel=vessel)
        plus_logits.append(out["plus_logit"].detach().cpu().numpy())
        labels.append(batch["label"].numpy())
        paths.extend(batch["image_path"])
        groups.extend(batch["group_id"])
        sources.extend(batch["sources"])
        gates.append(out["gate"].detach().cpu().numpy())
    return {
        "plus_logit": np.concatenate(plus_logits),
        "label": np.concatenate(labels),
        "image_path": paths,
        "group_id": groups,
        "source": sources,
        "gate": np.concatenate(gates),
    }


def plus_prob_from_logits(logits: np.ndarray) -> np.ndarray:
    """Stable sigmoid. Do not use for E0/E1 (those need softmax[:, 2])."""
    x = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _plus_prob_from_logits(logits: np.ndarray, experiment: str) -> np.ndarray:
    if uses_multiclass_ce(experiment):
        # plus_logit already stored as multiclass[..., 2] logit; convert via sigmoid of that
        # is not equal to softmax. Recompute isn't possible without all 3 logits.
        # Trainer stores softmax P(Plus) separately for E0/E1.
        raise RuntimeError("E0/E1 must pass softmax probabilities, not sigmoid(plus_logit).")
    return plus_prob_from_logits(logits)


@torch.no_grad()
def collect_predictions_full(model, loader, device, experiment: str) -> dict[str, np.ndarray | list]:
    model.eval()
    plus_logits, multi, ordinal, labels = [], [], [], []
    paths, groups, sources, gates = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        vessel = batch.get("vessel")
        vessel = vessel.to(device) if vessel is not None else None
        out = model(images, vessel=vessel)
        plus_logits.append(out["plus_logit"].detach().cpu().numpy())
        multi.append(torch.softmax(out["multiclass_logits"], dim=1).detach().cpu().numpy())
        ordinal.append(out["ordinal_logits"].detach().cpu().numpy())
        labels.append(batch["label"].numpy())
        paths.extend(batch["image_path"])
        groups.extend(batch["group_id"])
        sources.extend(batch["sources"])
        gates.append(out["gate"].detach().cpu().numpy())
    plus_logit = np.concatenate(plus_logits)
    multi_np = np.concatenate(multi)
    if uses_multiclass_ce(experiment):
        plus_prob = multi_np[:, 2]
    else:
        plus_prob = plus_prob_from_logits(plus_logit)
    return {
        "plus_logit": plus_logit,
        "plus_prob": plus_prob,
        "multiclass_prob": multi_np,
        "ordinal_logit": np.concatenate(ordinal),
        "label": np.concatenate(labels),
        "image_path": paths,
        "group_id": groups,
        "source": sources,
        "gate": np.concatenate(gates),
    }


def _loss_for_batch(model, batch, device, experiment, cfg, ce_loss, bce_loss, coral_loss):
    images = batch["image"].to(device)
    vessel = batch.get("vessel")
    vessel = vessel.to(device) if vessel is not None else None
    labels = batch["label"].to(device)
    plus = batch["plus_target"].to(device)
    out = model(images, vessel=vessel)
    weights = cfg.get("loss", {})
    if uses_multiclass_ce(experiment):
        loss = ce_loss(out["multiclass_logits"], labels)
        return loss, out
    plus_loss = bce_loss(out["plus_logit"], plus)
    loss = float(weights.get("plus_bce", 1.0)) * plus_loss
    if uses_ordinal_aux(experiment):
        loss = loss + float(weights.get("ordinal", 0.4)) * coral_loss(out["ordinal_logits"], labels)
        rank_w = float(weights.get("ranking", 0.2))
        if rank_w > 0:
            rank = pairwise_ranking_loss(
                out["plus_logit"],
                labels,
                sources=batch["sources"],
                max_pairs=int(weights.get("ranking_max_pairs", 64)),
                prefer_within_source=bool(weights.get("ranking_prefer_within_source", True)),
            )
            loss = loss + rank_w * rank
    bio_w = float(weights.get("biomarker_prediction", 0.0))
    if bio_w > 0:
        raise RuntimeError(
            "Biomarker auxiliary loss is disabled: anatomy-aware feature provenance is absent."
        )
    return loss, out


def train_one_run(
    cfg: dict[str, Any],
    *,
    experiment: str,
    resolution: int | None = None,
    seed: int | None = None,
    smoke_test: bool = False,
    allow_test_evaluation: bool = False,
    allow_noncanonical: bool = False,
    resume: bool = True,
    split_dir: Path | None = None,
    fold_id: str = "official",
    holdout_eval_csv: Path | None = None,
) -> dict[str, Any]:
    experiment = normalize_experiment(experiment)
    if float(cfg.get("loss", {}).get("biomarker_prediction", 0) or 0) > 0:
        raise SystemExit("E6 biomarker supervision is disabled until valid provenance exists.")

    split_sha = verify_canonical_split(cfg, allow_noncanonical=allow_noncanonical)
    resolution = int(resolution or cfg.get("resolution", 224))
    seed = int(seed if seed is not None else cfg.get("seed", 42))
    set_seed(seed)
    requested = "cpu" if smoke_test else "mps"
    device = get_device(requested)
    _print_device(requested, device)
    if device == "cpu" and not smoke_test:
        print("[oom-guidance] If MPS OOM: lower batch_size, raise grad_accumulation, or drop resolution.")

    official_splits = cfg["paths"]["splits_dir"]
    splits = Path(split_dir) if split_dir is not None else official_splits
    train_df = pd.read_csv(splits / "train.csv")
    val_df = pd.read_csv(splits / "val.csv")
    test_df = pd.read_csv(official_splits / "test.csv")
    assert_no_group_overlap(train_df, val_df)
    assert_no_test_images(pd.concat([train_df, val_df], ignore_index=True), test_df)
    if set(train_df["group_id"]) & set(test_df["group_id"]):
        raise RuntimeError("Train/test group overlap.")
    if set(val_df["group_id"]) & set(test_df["group_id"]):
        raise RuntimeError("Val/test group overlap.")

    smoke = bool(smoke_test or cfg.get("smoke_test"))
    n_train_keep = cfg.get("smoke", {}).get("n_train", 8) if smoke else None
    n_val_keep = cfg.get("smoke", {}).get("n_val", 8) if smoke else None
    epochs = int(cfg.get("smoke", {}).get("epochs", 1) if smoke else cfg.get("epochs", 50))
    batch_size = int(cfg.get("smoke", {}).get("batch_size", 2) if smoke else cfg.get("batch_size", 8))
    max_batches = int(cfg.get("smoke", {}).get("batches", 2)) if smoke else None
    accum = 1 if smoke else int(cfg.get("grad_accumulation", 1))

    if uses_vessel(experiment) and not smoke:
        prob_dir = cfg["paths"]["vessel_prob_dir"]
        if not Path(prob_dir).exists() or not any(Path(prob_dir).glob("*.npy")):
            raise SystemExit(
                f"Soft vessel maps missing under {prob_dir}. "
                "Run: python -m src.segmentation.infer_masks --save-probability-maps"
            )

    fold_info = official_reduced_fold(train_df, val_df)
    if split_dir is not None:
        fold_info["protocol"] = (
            "loso_in_domain_train_val_holdout_source_excluded"
            if str(fold_id).startswith("loso_")
            else "grouped_fold_train_val_not_using_canonical_test"
        )
        fold_info["split_dir"] = str(Path(split_dir))
    out_dir = output_dir(cfg, experiment, resolution, seed, str(fold_id), smoke)
    ckpt_path = out_dir / "best.pt"
    train_ckpt = out_dir / "train.pt"
    results_path = out_dir / "results.json"
    if resume and results_path.exists() and not smoke:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        same = (
            str(payload.get("experiment", "")).upper() == experiment
            and int(payload.get("seed", -1)) == seed
            and int(payload.get("resolution", resolution)) == resolution
            and str(payload.get("fold", fold_id)) == str(fold_id)
        )
        if same:
            print(f"[skip] complete val_auc={payload.get('val_raw', {}).get('auc')} -> {out_dir}")
            return payload

    ds_kwargs = dict(
        img_size=resolution,
        aug_cfg=cfg.get("augmentation", {}),
        experiment=experiment,
        vessel_prob_dir=cfg["paths"].get("vessel_prob_dir"),
        topology_threshold=float(cfg.get("vessel", {}).get("topology_threshold", 0.20)),
        plus_idx=int(cfg.get("plus_class_index", 2)),
        synthetic_vessel=smoke and uses_vessel(experiment),
    )
    train_ds = NextArchitectureDataset(train_df, train=True, smoke_limit=n_train_keep, **ds_kwargs)
    val_ds = NextArchitectureDataset(val_df, train=False, smoke_limit=n_val_keep, **ds_kwargs)
    nw = 0 if smoke else int(os.environ.get("NEXT_ARCH_NUM_WORKERS", cfg.get("num_workers", 0)))
    sampler = None
    shuffle = True
    source_balance: dict[str, Any] = {"enabled": False, "experiment": experiment}
    if is_source_balanced(experiment) and "source" in train_ds.df.columns:
        sources = train_ds.df["source"].astype(str).tolist()
        sampler, stats = make_source_balanced_sampler(sources, seed=seed)
        shuffle = False
        source_balance = {"enabled": True, "experiment": experiment, **stats}
        print(f"[e8] source-balanced sampler {stats['source_counts']}")
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=nw,
        collate_fn=collate_next,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=nw, collate_fn=collate_next
    )

    in_ch = 4 if is_early_fusion_4ch(experiment) or is_vessel_only(experiment) else 3
    backbone = "resnet18" if is_vessel_only(experiment) else cfg.get("backbone", "efficientnet_b5")
    pretrained = (not is_vessel_only(experiment)) and bool(cfg.get("pretrained", True)) and not smoke
    model = NextArchitectureNet(
        experiment,
        backbone=backbone,
        pretrained=pretrained,
        dropout=float(cfg.get("dropout", 0.5)),
        rgb_dim=int(cfg.get("gate", {}).get("rgb_dim", 512)),
        vessel_dim=int(cfg.get("gate", {}).get("vessel_dim", 128)),
        vessel_encoder=str(cfg.get("gate", {}).get("vessel_encoder", "resnet18")),
        gate_bias_init=gate_bias_init(experiment, float(cfg.get("gate", {}).get("bias_init", -2.5))),
        modality_dropout=float(cfg.get("gate", {}).get("modality_dropout", 0.2)),
        in_channels=in_ch,
    ).to(device)

    train_labels = torch.tensor(train_ds.labels, dtype=torch.long)
    plus_targets = (train_labels == int(cfg.get("plus_class_index", 2))).float()
    ce_loss = nn.CrossEntropyLoss(weight=class_weights_from_labels(train_labels, 3).to(device))
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=binary_pos_weight_from_labels(plus_targets).to(device))
    coral_loss = CoralOrdinalLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("lr", 1e-5)), weight_decay=float(cfg.get("weight_decay", 1e-4))
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg.get("scheduler_factor", 0.2)),
        patience=int(cfg.get("scheduler_patience", 3)),
    )

    start_epoch = 1
    best_auc = -1.0
    best_epoch = None
    patience = int(cfg.get("early_stopping_patience", 8))
    bad = 0
    history: list[dict[str, Any]] = []
    if resume and train_ckpt.exists() and not smoke:
        state = torch.load(train_ckpt, map_location=device)
        same_run = (
            state.get("split_sha256") == split_sha
            and state.get("experiment") == experiment
            and int(state.get("resolution", resolution)) == resolution
            and int(state.get("seed", seed)) == seed
        )
        if same_run:
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            if state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state.get("epoch", 0)) + 1
            best_auc = float(state.get("best_auc", -1.0))
            best_epoch = state.get("best_epoch")
            bad = int(state.get("bad", 0))
            history = list(state.get("history") or [])
            if state.get("stopped_early"):
                start_epoch = epochs + 1
                print(
                    f"[resume] early-stop already hit best_epoch={best_epoch} "
                    f"best_val_auc={best_auc:.4f}; writing metrics if missing"
                )
            else:
                print(f"[resume] epoch {start_epoch} best_val_auc={best_auc:.4f} bad={bad}")
        else:
            print("[stale] ignoring train checkpoint from different split/experiment/resolution/seed")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_dl, start=1):
            loss, _ = _loss_for_batch(
                model, batch, device, experiment, cfg, ce_loss, bce_loss, coral_loss
            )
            (loss / accum).backward()
            if step % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.item()) * batch["image"].size(0)
            seen += batch["image"].size(0)
            if max_batches is not None and step >= max_batches:
                break
        if seen == 0:
            raise RuntimeError("Empty training loader.")
        val_pack = collect_predictions_full(model, val_dl, device, experiment)
        y_bin = plus_binary(val_pack["label"], cfg.get("plus_class_index", 2))
        val_auc = float("nan")
        try:
            from sklearn.metrics import roc_auc_score

            if len(np.unique(y_bin)) > 1:
                val_auc = float(roc_auc_score(y_bin, val_pack["plus_prob"]))
        except Exception:
            val_auc = float("nan")
        print(f"epoch {epoch}: loss={running / seen:.4f} val_auc={val_auc:.4f}")
        history.append({"epoch": epoch, "loss": running / seen, "val_auc": val_auc})
        scheduler.step(0.0 if np.isnan(val_auc) else val_auc)
        improved = not np.isnan(val_auc) and val_auc > best_auc
        if improved:
            best_auc = val_auc
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "experiment": experiment,
                    "resolution": resolution,
                    "seed": seed,
                    "split_sha256": split_sha,
                    "epoch": epoch,
                },
                ckpt_path,
            )
        else:
            bad += 1
        stopped_early = (not smoke) and bad >= patience
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_auc": best_auc,
                "best_epoch": best_epoch,
                "bad": bad,
                "history": history,
                "experiment": experiment,
                "resolution": resolution,
                "seed": seed,
                "split_sha256": split_sha,
                "stopped_early": stopped_early,
            },
            train_ckpt,
        )
        if stopped_early:
            print(f"[early-stop] patience={patience} best_epoch={best_epoch}")
            break

    if ckpt_path.exists():
        best = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best["state_dict"])

    val_pack = collect_predictions_full(model, val_dl, device, experiment)
    y_bin = plus_binary(val_pack["label"], cfg.get("plus_class_index", 2))
    thr = youden_threshold(y_bin, val_pack["plus_prob"]) if len(np.unique(y_bin)) > 1 else 0.5
    raw_metrics = discrimination_bundle(y_bin, val_pack["plus_prob"], thr)
    try:
        temperature = fit_temperature(y_bin, val_pack["plus_prob"])
        cal_prob = apply_temperature(val_pack["plus_prob"], temperature)
    except ValueError:
        temperature = 1.0
        cal_prob = val_pack["plus_prob"]
    cal_metrics = discrimination_bundle(y_bin, cal_prob, youden_threshold(y_bin, cal_prob) if len(np.unique(y_bin)) > 1 else thr)
    hs_thr = high_sensitivity_threshold(y_bin, val_pack["plus_prob"], 0.95)
    hs_metrics = discrimination_bundle(y_bin, val_pack["plus_prob"], hs_thr)
    per_src = per_source_bundle(np.array(val_pack["source"]), y_bin, val_pack["plus_prob"], thr)
    boot_n = int(cfg.get("bootstrap", {}).get("smoke_n" if smoke else "n", 2000))
    boot = grouped_bootstrap_auc(
        y_bin, val_pack["plus_prob"], np.array(val_pack["group_id"]), n_boot=boot_n, seed=seed
    )
    if is_gated_fusion(experiment):
        gate = np.asarray(val_pack["gate"], dtype=float)
        gate_stats = {
            "mean": float(gate.mean()),
            "std": float(gate.std()),
            "collapse_zero": bool((gate < 1e-3).mean() > 0.95),
            "collapse_one": bool((gate > 1 - 1e-3).mean() > 0.95),
            "by_source": {
                str(s): float(gate[np.array(val_pack["source"]) == s].mean())
                for s in sorted(set(val_pack["source"]))
                if (np.array(val_pack["source"]) == s).any()
            },
        }
    else:
        gate_stats = None

    pred_df = pd.DataFrame(
        {
            "image_path": val_pack["image_path"],
            "label": val_pack["label"],
            "plus_target": y_bin,
            "group_id": val_pack["group_id"],
            "source": val_pack["source"],
            "split": "val",
            "fold": fold_id,
            "seed": seed,
            "plus_logit": val_pack["plus_logit"],
            "plus_prob_raw": val_pack["plus_prob"],
            "plus_prob_calibrated": cal_prob,
            "gate": val_pack["gate"],
            "experiment": experiment,
            "resolution": resolution,
            "smoke_test": smoke,
        }
    )
    pred_path = out_dir / "val_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    test_block = {"skipped": True, "reason": "allow_test_evaluation is false; canonical test is not pristine"}
    if allow_test_evaluation:
        print(
            "WARNING: evaluating canonical test.csv. This split was already used by "
            "exploratory probes and is NOT a pristine confirmatory test."
        )
        test_ds = NextArchitectureDataset(test_df, train=False, smoke_limit=n_val_keep if smoke else None, **ds_kwargs)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw, collate_fn=collate_next)
        test_pack = collect_predictions_full(model, test_dl, device, experiment)
        ty = plus_binary(test_pack["label"], cfg.get("plus_class_index", 2))
        t_raw = discrimination_bundle(ty, test_pack["plus_prob"], thr)
        test_block = {
            "skipped": False,
            "warning": "canonical test is not pristine for confirmatory claims",
            "metrics_raw_at_val_threshold": t_raw,
            "per_source": per_source_bundle(np.array(test_pack["source"]), ty, test_pack["plus_prob"], thr),
        }
        pd.DataFrame(
            {
                "image_path": test_pack["image_path"],
                "label": test_pack["label"],
                "plus_target": ty,
                "group_id": test_pack["group_id"],
                "source": test_pack["source"],
                "plus_prob_raw": test_pack["plus_prob"],
                "smoke_test": smoke,
            }
        ).to_csv(out_dir / "test_predictions.csv", index=False)

    holdout_block: dict[str, Any] = {"skipped": True, "reason": "no holdout_eval_csv"}
    if holdout_eval_csv is not None:
        hold_path = Path(holdout_eval_csv)
        hold_df = pd.read_csv(hold_path)
        print(
            f"[loso-eval] scoring {hold_path} at val Youden threshold={thr:.4f}. "
            "Holdout labels are for reporting only; not used for early-stop or threshold."
        )
        hold_ds = NextArchitectureDataset(
            hold_df, train=False, smoke_limit=n_val_keep if smoke else None, **ds_kwargs
        )
        hold_dl = DataLoader(
            hold_ds, batch_size=batch_size, shuffle=False, num_workers=nw, collate_fn=collate_next
        )
        hold_pack = collect_predictions_full(model, hold_dl, device, experiment)
        hy = plus_binary(hold_pack["label"], cfg.get("plus_class_index", 2))
        h_raw = discrimination_bundle(hy, hold_pack["plus_prob"], thr)
        holdout_block = {
            "skipped": False,
            "path": str(hold_path),
            "n": int(len(hold_df)),
            "threshold_source": "in_domain_val_youden",
            "metrics_raw_at_val_threshold": h_raw,
            "per_source": per_source_bundle(np.array(hold_pack["source"]), hy, hold_pack["plus_prob"], thr),
        }
        pd.DataFrame(
            {
                "image_path": hold_pack["image_path"],
                "label": hold_pack["label"],
                "plus_target": hy,
                "group_id": hold_pack["group_id"],
                "source": hold_pack["source"],
                "plus_prob_raw": hold_pack["plus_prob"],
                "split": "holdout",
                "fold": fold_id,
            }
        ).to_csv(out_dir / "holdout_predictions.csv", index=False)

    counts = {"train_n": int(len(train_ds)), "val_n": int(len(val_ds))}
    provenance = build_run_provenance(
        cfg,
        experiment_id=experiment,
        resolution=resolution,
        seed=seed,
        fold=fold_id,
        split_sha256=split_sha,
        fold_manifest_sha256=None,
        n_train=len(train_ds),
        n_val=len(val_ds),
        train_groups=int(train_df["group_id"].nunique()) if not smoke else len(train_ds),
        val_groups=int(val_df["group_id"].nunique()) if not smoke else len(val_ds),
        source_class_counts=counts,
        device=device,
        checkpoint_path=ckpt_path if ckpt_path.exists() else None,
        best_epoch=best_epoch,
        early_stopping_metric="val_plus_auc",
        val_threshold=float(thr),
        calibration_method="temperature_scaling",
        smoke_test=smoke,
    )
    provenance["fold_protocol"] = fold_info["protocol"]
    results = {
        "smoke_test": smoke,
        "experiment": experiment,
        "resolution": resolution,
        "seed": seed,
        "fold": fold_id,
        "e0_reproduction_deviations": (
            [
                "Writes results/next_architecture/E0, never results/branch_b_* or weights/branch_b_*.",
                "Default batch_size/grad_accumulation may differ from Branch B (16) for MPS memory.",
                "E0 uses official train/val reduced protocol, not a fresh data split.",
            ]
            if experiment == "E0"
            else []
        ),
        "val_raw": raw_metrics,
        "val_calibrated": cal_metrics,
        "val_high_sensitivity_0_95": hs_metrics,
        "temperature": temperature,
        "per_source_val": per_src,
        "worst_source_auc": per_src.get("_worst_source_auc"),
        "bootstrap": boot,
        "gate": gate_stats,
        "history": history,
        "source_balance": source_balance,
        "test": test_block,
        "holdout": holdout_block,
        "provenance": provenance,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"[done] {experiment} val_auc={raw_metrics.get('auc')} -> {out_dir}")
    if smoke:
        print("[smoke] metrics above are NOT experimental results.")
    return results
