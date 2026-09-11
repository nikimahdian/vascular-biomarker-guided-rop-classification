"""Branch B (baseline): fine-tune a timm CNN on the fundus images for Plus.

This is the image-only path that matches the prior work / group paper, kept on the
SAME split as Branch A for a fair comparison. Writes:
    weights/branch_b_<backbone>.pth   best checkpoint (by val AUC)
    results/branch_b_results.json     test metrics
    results/branch_b_test_preds.csv   per-sample P(Plus) on the test set

Usage:
    python -m src.classify.branch_b_cnn
    python -m src.classify.branch_b_cnn --backbone tf_efficientnetv2_m --epochs 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classify.image_dataset import FundusCsvDataset
from src.utils.common import (
    build_provenance,
    ensure_dirs,
    evaluate_plus_ovr_bundle,
    get_device,
    load_config,
    num_classes,
    plus_class_index,
    plus_ovr_metrics,
    provenance_matches_current,
    set_seed,
    tune_plus_threshold,
)
from src.utils.branch_eval import per_source_plus_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BackboneClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int = 2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        return self.head(self.dropout(self.backbone(x)))


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


@torch.no_grad()
def predict_scores(model, loader, device, plus_idx: int):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        proba = torch.softmax(model(x.to(device)), dim=1)
        scores.append(proba[:, plus_idx].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


@torch.no_grad()
def predict_all(model, loader, device, plus_idx: int):
    """Return P(Plus), labels, and multiclass argmax predictions."""
    model.eval()
    scores, labels, preds = [], [], []
    for x, y in loader:
        proba = torch.softmax(model(x.to(device)), dim=1)
        scores.append(proba[:, plus_idx].cpu().numpy())
        preds.append(proba.argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(preds)


def class_weights(labels: np.ndarray, n_cls: int, device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_cls).astype(float)
    w = counts.sum() / (n_cls * np.clip(counts, 1, None))
    return torch.tensor(w, dtype=torch.float32, device=device)


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg["seed"])
    cb = cfg["classify_b"]
    device = get_device()

    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=cb["backbone"])
    ap.add_argument("--epochs", type=int, default=cb["epochs"])
    ap.add_argument("--batch_size", type=int, default=cb["batch_size"])
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="continue training from branch_b_*_train.pth if present (default: on)",
    )
    ap.add_argument(
        "--skip-if-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip when branch_b_results.json already exists (default: on)",
    )
    ap.add_argument("--force", action="store_true", help="re-evaluate test metrics even if training skipped")
    ap.add_argument("--eval-only", action="store_true", help="skip training; eval test from saved ckpt")
    args = ap.parse_args()

    res_dir = cfg["paths"]["results_dir"]
    results_path = res_dir / "branch_b_results.json"
    ev = cfg.get("eval", {})
    if args.skip_if_done and results_path.exists() and not args.force:
        existing = json.loads(results_path.read_text())
        if provenance_matches_current(existing, cfg):
            print(f"[skip] current Branch B results exist -> {results_path}")
            return
        print(f"[stale] ignoring Branch B results from different split/config: {results_path}")

    n_cls = num_classes(cfg)
    plus_idx = plus_class_index(cfg)

    splits = cfg["paths"]["splits_dir"]
    img_size, nw = cb["img_size"], cb["num_workers"]
    train_ds = FundusCsvDataset(str(splits / "train.csv"), img_size, train=True)
    val_ds = FundusCsvDataset(str(splits / "val.csv"), img_size, train=False)
    test_ds = FundusCsvDataset(str(splits / "test.csv"), img_size, train=False)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    model = BackboneClassifier(args.backbone, num_classes=n_cls).to(device)
    ckpt = cfg["paths"]["weights_dir"] / f"branch_b_{args.backbone.replace('.', '_')}.pth"
    train_ckpt = cfg["paths"]["weights_dir"] / f"branch_b_{args.backbone.replace('.', '_')}_train.pth"

    if not args.eval_only:
        if cb["use_focal_loss"]:
            criterion = FocalLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds.labels, n_cls, device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cb["lr"], weight_decay=cb["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=3)

        best_auc = -1.0
        start_epoch = 1
        if args.resume and train_ckpt.exists():
            state = torch.load(train_ckpt, map_location=device)
            if (
                state.get("split_sha256") == _sha256(splits / "all.csv")
                and state.get("config_sha256")
                == _sha256(cfg["_root"] / "configs" / "config.yaml")
            ):
                model.load_state_dict(state["state_dict"])
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                best_auc = float(state.get("best_auc", -1.0))
                start_epoch = int(state.get("epoch", 0)) + 1
                print(
                    f"[resume] epoch {start_epoch}/{args.epochs} "
                    f"best_auc={best_auc:.3f}"
                )
            else:
                print("[stale] ignoring training checkpoint from different split/config")

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running = 0.0
            for x, y in tqdm(train_dl, desc=f"train {epoch}/{args.epochs}"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                running += loss.item() * x.size(0)

            val_scores, val_labels = predict_scores(model, val_dl, device, plus_idx)
            vm = plus_ovr_metrics(val_labels, val_scores, plus_idx)
            print(f"epoch {epoch}: loss={running / len(train_ds):.4f} val_auc={vm['auc']:.3f} val_f1={vm['f1']:.3f}")
            scheduler.step(vm["auc"] if not np.isnan(vm["auc"]) else 0.0)
            if not np.isnan(vm["auc"]) and vm["auc"] > best_auc:
                best_auc = vm["auc"]
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "backbone": args.backbone,
                        "split_sha256": _sha256(
                            cfg["paths"]["splits_dir"] / "all.csv"
                        ),
                        "config_sha256": _sha256(
                            cfg["_root"] / "configs" / "config.yaml"
                        ),
                    },
                    ckpt,
                )
                print(f"  saved best -> {ckpt} (val_auc={best_auc:.3f})")
            torch.save(
                {
                    "epoch": epoch,
                    "best_auc": best_auc,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "backbone": args.backbone,
                    "split_sha256": _sha256(splits / "all.csv"),
                    "config_sha256": _sha256(
                        cfg["_root"] / "configs" / "config.yaml"
                    ),
                },
                train_ckpt,
            )
    else:
        print("[eval-only] skipping training; loading best checkpoint")

    if not ckpt.exists():
        raise SystemExit(f"No Branch B checkpoint saved -> {ckpt}")

    class_names = cfg["data"]["class_names"]
    best_state = torch.load(ckpt, map_location=device)
    if (
        best_state.get("split_sha256") != _sha256(splits / "all.csv")
        or best_state.get("config_sha256")
        != _sha256(cfg["_root"] / "configs" / "config.yaml")
    ):
        raise SystemExit(
            "Best Branch B checkpoint is stale or lacks current split/config provenance."
        )
    model.load_state_dict(best_state["state_dict"])

    val_scores, val_labels, _ = predict_all(model, val_dl, device, plus_idx)
    thr = tune_plus_threshold(val_labels, val_scores, plus_idx, method=ev.get("threshold_method", "youden"))
    val_m = plus_ovr_metrics(val_labels, val_scores, plus_idx, threshold=thr)
    print(f"[val] tuned threshold={thr:.3f} f1={val_m['f1']:.3f} auc={val_m['auc']:.3f}")

    test_scores, test_labels, test_pred = predict_all(model, test_dl, device, plus_idx)
    meta_columns = [
        column
        for column in [
            "image_path",
            "label",
            "source",
            "group_id",
            "patient_id",
            "exam_id",
            "identity_level",
        ]
        if column in test_ds.df.columns
    ]
    test_meta = test_ds.df[meta_columns]
    test_bundle = evaluate_plus_ovr_bundle(
        test_labels,
        test_scores,
        test_pred,
        plus_idx,
        thr,
        class_names,
        n_boot=ev.get("bootstrap_n", 1000),
        seed=cfg["seed"],
        groups=(
            test_meta["group_id"].values
            if "group_id" in test_meta.columns
            else None
        ),
    )
    per_src = per_source_plus_metrics(test_meta, test_scores, plus_idx, thr)
    tm = test_bundle["plus_ovr"]
    print("[test]", {k: round(v, 4) for k, v in tm.items() if k in {"auc", "sensitivity", "specificity", "f1", "threshold"}})

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "backbone": args.backbone,
                "checkpoint": str(ckpt.resolve()),
                "checkpoint_sha256": _sha256(ckpt),
                "split_sha256": _sha256(cfg["paths"]["splits_dir"] / "all.csv"),
                "config_sha256": _sha256(cfg["_root"] / "configs" / "config.yaml"),
                "provenance": build_provenance(cfg),
                "val_threshold": float(thr),
                "val": val_m,
                "metrics": tm,
                "test": test_bundle,
                "per_source_test": per_src,
            },
            f,
            indent=2,
        )
    out_cols = {
        "image_path": test_ds.df["image_path"].values,
        "label": test_labels,
        "p_plus_branch_b": test_scores,
    }
    for column in ["source", "group_id", "patient_id", "exam_id", "identity_level"]:
        if column in test_ds.df.columns:
            out_cols[column] = test_ds.df[column].values
    pd.DataFrame(out_cols).to_csv(res_dir / "branch_b_test_preds.csv", index=False)
    val_cols = {
        "image_path": val_ds.df["image_path"].values,
        "label": val_labels,
        "p_plus_branch_b": val_scores,
    }
    for column in ["source", "group_id", "patient_id", "exam_id", "identity_level"]:
        if column in val_ds.df.columns:
            val_cols[column] = val_ds.df[column].values
    pd.DataFrame(val_cols).to_csv(res_dir / "branch_b_val_preds.csv", index=False)
    print(f"[done] test AUC={tm['auc']:.3f} -> {res_dir}")


if __name__ == "__main__":
    main()
