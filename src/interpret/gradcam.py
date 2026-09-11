"""Grad-CAM overlays for the Branch B CNN (qualitative comparison with the paper).

Loads the best Branch B checkpoint and renders Grad-CAM heatmaps for a sample of
test images into results/gradcam/.

Usage:
    python -m src.interpret.gradcam
    python -m src.interpret.gradcam --n 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.classify.branch_b_cnn import BackboneClassifier
from src.classify.image_dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from src.utils.common import (
    ensure_dirs,
    get_device,
    load_config,
    plus_class_index,
    validate_prediction_frame,
)


def pick_target_layer(model: BackboneClassifier):
    """Last conv-ish module in the timm backbone works for CNN families."""
    modules = [m for m in model.backbone.modules() if isinstance(m, torch.nn.Conv2d)]
    if not modules:
        raise RuntimeError("No Conv2d layer found; Grad-CAM here targets CNN backbones only.")
    return modules[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    device = get_device()

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=cfg["interpret"]["gradcam_samples"])
    ap.add_argument("--ckpt", default=None, help="path to branch B checkpoint (auto-detect if omitted)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path is None:
        results_path = cfg["paths"]["results_dir"] / "branch_b_results.json"
        if not results_path.exists():
            raise SystemExit("Missing branch_b_results.json; cannot identify best checkpoint.")
        checkpoint = json.loads(results_path.read_text()).get("checkpoint")
        if not checkpoint:
            raise SystemExit("branch_b_results.json does not record its checkpoint.")
        ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Recorded Branch B checkpoint missing: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device)
    current_split_hash = _sha256(cfg["paths"]["splits_dir"] / "all.csv")
    current_config_hash = _sha256(cfg["_root"] / "configs" / "config.yaml")
    if (
        state.get("split_sha256") != current_split_hash
        or state.get("config_sha256") != current_config_hash
    ):
        raise SystemExit(
            "Branch B checkpoint is stale or lacks current split/config provenance. "
            "Retrain Branch B before Grad-CAM."
        )
    backbone = state.get("backbone", cfg["classify_b"]["backbone"])
    sd = state["state_dict"]
    n_cls = int(sd["head.weight"].shape[0]) if "head.weight" in sd else 2

    model = BackboneClassifier(backbone, num_classes=n_cls).to(device)
    model.load_state_dict(sd)
    model.eval()

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam = GradCAM(model=model, target_layers=[pick_target_layer(model)])

    img_size = cfg["classify_b"]["img_size"]
    tf = build_transforms(img_size, train=False)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    plus_idx = plus_class_index(cfg)
    pred_path = cfg["paths"]["results_dir"] / "branch_b_test_preds.csv"
    if not pred_path.exists():
        raise SystemExit("Missing current Branch B test predictions.")
    test_df = pd.read_csv(pred_path)
    validate_prediction_frame(test_df, "test", cfg)
    threshold = json.loads(
        (cfg["paths"]["results_dir"] / "branch_b_results.json").read_text()
    )["val_threshold"]
    test_df["pred_plus"] = test_df["p_plus_branch_b"] >= threshold
    test_df["true_plus"] = test_df["label"].astype(int) == plus_idx
    test_df["outcome"] = np.select(
        [
            test_df["true_plus"] & test_df["pred_plus"],
            ~test_df["true_plus"] & test_df["pred_plus"],
            test_df["true_plus"] & ~test_df["pred_plus"],
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )
    strata = ["outcome"] + (["source"] if "source" in test_df.columns else [])
    sample = (
        test_df.groupby(strata, group_keys=False)
        .apply(lambda rows: rows.sample(1, random_state=cfg["seed"]))
        .reset_index(drop=True)
    )
    if len(sample) < args.n:
        remaining = test_df[~test_df["image_path"].isin(sample["image_path"])]
        extra = remaining.sample(
            min(args.n - len(sample), len(remaining)), random_state=cfg["seed"]
        )
        sample = pd.concat([sample, extra], ignore_index=True)
    sample = sample.iloc[: args.n]
    out_dir = cfg["paths"]["results_dir"] / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("cam_*.png"):
        stale.unlink()

    manifest = []
    for i, row in enumerate(sample.itertuples()):
        img = Image.open(row.image_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(torch.softmax(model(x), dim=1).argmax(1).item())
        grayscale = cam(
            input_tensor=x,
            targets=[ClassifierOutputTarget(plus_idx)],
        )[0]

        rgb = x.squeeze().cpu().numpy().transpose(1, 2, 0)
        rgb = np.clip(rgb * std + mean, 0, 1).astype(np.float32)
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
        output_path = (
            out_dir
            / f"cam_{i:03d}_{row.outcome}_{row.source}_true{row.label}_pred{pred}.png"
        )
        Image.fromarray(overlay).save(output_path)
        manifest.append(
            {
                "image_path": row.image_path,
                "overlay_path": str(output_path),
                "source": row.source,
                "group_id": row.group_id,
                "label": int(row.label),
                "predicted_class": pred,
                "p_plus": float(row.p_plus_branch_b),
                "threshold": float(threshold),
                "outcome": row.outcome,
                "target_class": plus_idx,
                "checkpoint": str(ckpt_path),
                "split_sha256": current_split_hash,
                "config_sha256": current_config_hash,
            }
        )
    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)

    print(f"[done] {len(sample)} Grad-CAM overlays -> {out_dir}")


if __name__ == "__main__":
    main()
