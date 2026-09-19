#!/usr/bin/env python
"""T1.1 — the project's MAnet against HVDROPDB expert masks.

Part 1: vessel Dice / IoU / recall / precision for the 100 HVDROPDB-BV images, evaluated at
        three inference resolutions (256 = the project's setting, 512, 1024) to test whether the
        256x256 mask is the factor limiting thin-vessel measurement.
        Pre- and post-processing replicate src/segmentation/infer_masks.py exactly:
        PIL RGB -> Resize((S,S)) -> ToTensor (no ImageNet normalisation) -> sigmoid ->
        post_process(threshold=0.20, min_area=50, close_k=3) -> cv2.resize NEAREST to native.
        Published reference on this dataset: Dice 0.52 (RetCam) / 0.66 (Neo), authors' AG U-Net.

Part 2: how far the project's pseudo-optic-disc (image centre, radius = max(8, min(h,w)/8))
        is from the expert optic-disc masks in HVDROPDB-OD.

Read-only on the repo; writes to results/hvdro_validation/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/Users/moniaz/niki")
from src.segmentation.infer_masks import post_process  # noqa: E402
from src.segmentation.models import build_model  # noqa: E402
from src.utils.common import get_device, load_config  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
SEG = ROOT / "data/raw/hvdro/segmentation/HVDROPDB_RetCam_Neo_Segmentation"
OUT = ROOT / "results/hvdro_validation"
OUT.mkdir(parents=True, exist_ok=True)

cfg = load_config()
sc = cfg["segmentation"]
THR, MIN_AREA, CLOSE_K = float(sc["threshold"]), int(sc["min_area"]), int(sc["morph_close_kernel"])
SIZES = (256, 512, 1024)

device = get_device()
print(f"device={device}  arch={sc['arch']}  encoder={sc['encoder']}")
print(f"threshold={THR}  min_area={MIN_AREA}  close_k={CLOSE_K}")

ckpt = ROOT / sc["weight_path"]
print(f"checkpoint: {ckpt}  exists={ckpt.exists()}")
model = build_model(sc["arch"], sc["encoder"], None, int(sc["in_channels"]), 1).to(device).eval()
model.load_state_dict(torch.load(str(ckpt), map_location=device))
print("model loaded")

tf_cache = {s: transforms.Compose([transforms.Resize((s, s)), transforms.ToTensor()]) for s in SIZES}


def predict(img: Image.Image, size: int) -> np.ndarray:
    x = tf_cache[size](img).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(x)).squeeze().cpu().numpy()


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred > 0, gt > 0
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    s = p.sum() + g.sum()
    return {
        "dice": float(2 * tp / s) if s else 1.0,
        "iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "gt_frac": float(g.mean()),
        "pred_frac": float(p.mean()),
    }


# ---------------------------------------------------------------- Part 1: vessel Dice
rows = []
for cam in ("RetCam_Vessels", "Neo_Vessels"):
    img_dir = SEG / "HVDROPDB-BV" / f"{cam}_images"
    msk_dir = SEG / "HVDROPDB-BV" / f"{cam}_masks"
    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    assert stems == sorted(p.stem for p in msk_dir.glob("*.png")), f"{cam}: image/mask names differ"
    print(f"\n=== {cam}: {len(stems)} image/mask pairs ===")
    for i, stem in enumerate(stems):
        img = Image.open(img_dir / f"{stem}.png").convert("RGB")
        gt_full = np.array(Image.open(msk_dir / f"{stem}.png").convert("L"))
        orig_w, orig_h = img.size
        row = {"camera": cam, "stem": stem, "w": orig_w, "h": orig_h}
        for size in SIZES:
            try:
                prob = predict(img, size)
            except RuntimeError as e:  # e.g. out of memory at 1024
                print(f"  [skip] {cam}/{stem} S={size}: {type(e).__name__}", flush=True)
                for tag in (f"native_S{size}", f"atS_S{size}"):
                    for k in ("dice", "iou", "recall", "precision", "gt_frac", "pred_frac"):
                        row[f"{tag}_{k}"] = float("nan")
                continue
            m = post_process(prob, THR, MIN_AREA, CLOSE_K)
            m_native = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            for tag, pred_arr, gt_arr in (
                (f"native_S{size}", m_native, gt_full),                       # the deployed behaviour
                (f"atS_S{size}", m, cv2.resize(gt_full, (size, size), interpolation=cv2.INTER_NEAREST)),
            ):
                for k, v in metrics(pred_arr, gt_arr).items():
                    row[f"{tag}_{k}"] = v
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(stems)} done", flush=True)

vdf = pd.DataFrame(rows)
vdf.to_csv(OUT / "vessel_dice_per_image.csv", index=False)

summary = []
for cam, g in vdf.groupby("camera"):
    for size in SIZES:
        summary.append({
            "camera": cam, "inference_size": size, "n": len(g),
            "dice_native": g[f"native_S{size}_dice"].mean(),
            "dice_atS": g[f"atS_S{size}_dice"].mean(),
            "iou_native": g[f"native_S{size}_iou"].mean(),
            "recall_native": g[f"native_S{size}_recall"].mean(),
            "precision_native": g[f"native_S{size}_precision"].mean(),
            "gt_vessel_fraction": g[f"native_S{size}_gt_frac"].mean(),
            "pred_vessel_fraction": g[f"native_S{size}_pred_frac"].mean(),
        })
sdf = pd.DataFrame(summary)
sdf.to_csv(OUT / "vessel_dice_summary.csv", index=False)
print("\n=== vessel Dice vs HVDROPDB expert masks ===")
print(sdf.to_string(index=False))
print("\npublished reference (authors' AG U-Net): RetCam 0.52, Neo 0.66")

# ---------------------------------------------------------------- Part 2: pseudo-disc error
disc = []
for cam in ("Retcam_OpticDisc", "Neo_OpticDisc"):
    msk_dir = SEG / "HVDROPDB-OD" / f"{cam}_masks"
    files = sorted(msk_dir.glob("*.png"))
    print(f"\n=== {cam}: {len(files)} expert optic-disc masks ===")
    for f in files:
        m = np.array(Image.open(f).convert("L")) > 0
        if m.sum() == 0:
            continue
        h, w = m.shape
        ys, xs = np.nonzero(m)
        cx, cy = xs.mean(), ys.mean()
        # what the project uses instead: image centre and radius = max(8, min(h,w)/8)
        pcx, pcy = w / 2, h / 2
        pra = max(8.0, min(h, w) / 8.0)
        true_ra = float(np.sqrt(m.sum() / np.pi))
        dist = float(np.hypot(cx - pcx, cy - pcy))
        disc.append({
            "camera": cam, "stem": f.stem, "w": w, "h": h,
            "true_radius_px": true_ra, "pseudo_radius_px": pra,
            "centre_offset_px": dist,
            "centre_offset_frac_of_width": dist / w,
            "radius_ratio_pseudo_over_true": pra / true_ra,
            "centre_offset_in_true_radii": dist / true_ra,
        })
ddf = pd.DataFrame(disc)
ddf.to_csv(OUT / "disc_error_per_image.csv", index=False)
dsum = ddf.groupby("camera").agg(
    n=("stem", "count"),
    true_radius_px=("true_radius_px", "mean"),
    pseudo_radius_px=("pseudo_radius_px", "mean"),
    centre_offset_px=("centre_offset_px", "mean"),
    centre_offset_frac_of_width=("centre_offset_frac_of_width", "mean"),
    radius_ratio=("radius_ratio_pseudo_over_true", "mean"),
    offset_in_true_radii=("centre_offset_in_true_radii", "mean"),
).reset_index()
dsum.to_csv(OUT / "disc_error_summary.csv", index=False)
print("\n=== pseudo-disc vs expert optic disc ===")
print(dsum.to_string(index=False))

json.dump({
    "checkpoint": str(ckpt), "arch": sc["arch"], "encoder": sc["encoder"],
    "threshold": THR, "min_area": MIN_AREA, "close_k": CLOSE_K,
    "sizes": list(SIZES), "published_reference": {"RetCam": 0.52, "Neo": 0.66},
    "note": "dice_native reproduces the deployed pipeline (upsample to native); dice_atS isolates the model.",
}, open(OUT / "manifest.json", "w"), indent=2)
print(f"\n[done] -> {OUT}")
