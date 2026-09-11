"""Predict a binary vessel mask for every fundus image in the shared split.

Reads data/splits/all.csv, runs the segmenter, applies the prior-work
post-processing (threshold -> small-component removal -> morphological close),
and writes one PNG (0/255) per image into paths.masks_dir. A manifest CSV maps
image_path -> mask_path and is consumed by the biomarker step.

Usage:
    python -m src.segmentation.infer_masks
    python -m src.segmentation.infer_masks --weight weights/best_weight_MAnet_res34_resize_31 --arch MAnet --threshold 0.2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.segmentation.models import build_model
from src.utils.common import ensure_dirs, get_device, load_config


def post_process(prob: np.ndarray, threshold: float, min_area: int, close_k: int) -> np.ndarray:
    """prob: float mask in [0,1] -> uint8 {0,255} cleaned mask."""
    binary = (prob > threshold).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    if close_k and close_k > 0:
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8)
        )
    return cleaned


def unique_mask_name(image_path: str) -> str:
    stem = Path(image_path).stem
    h = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:8]  # avoid collisions across sources
    return f"{stem}_{h}.png"


def find_existing_mask(masks_dir: Path, image_path: str) -> Path | None:
    """Reuse a mask if path-hash name OR unique stem_* file already exists.

    Allows restoring masks produced on another machine (different absolute path →
    different hash) without re-running segmentation.
    """
    preferred = masks_dir / unique_mask_name(image_path)
    if preferred.exists():
        return preferred
    stem = Path(image_path).stem
    matches = sorted(masks_dir.glob(f"{stem}_*.png"))
    if len(matches) == 1:
        return matches[0]
    return None


def _upsert_record(records: list[dict], have: set[str], rec: dict) -> None:
    img = str(rec["image_path"])
    if img in have:
        records[:] = [row for row in records if str(row.get("image_path")) != img]
    records.append(rec)
    have.add(img)


def _prob_record_from_npy(
    img_path: str,
    npy_path: Path,
    weight_path: Path,
    img_size: tuple,
    threshold: float,
    version: str,
) -> dict:
    arr = np.load(npy_path)
    if arr.ndim >= 2:
        orig_h, orig_w = int(arr.shape[0]), int(arr.shape[1])
    else:
        orig_h, orig_w = 0, 0
    return {
        "image_path": img_path,
        "prob_path": str(npy_path),
        "orig_width": orig_w,
        "orig_height": orig_h,
        "output_shape": [orig_h, orig_w],
        "probability_map_version": version,
        "preprocessing_version": f"resize_{img_size}_sigmoid_then_resize_orig",
        "segmentation_checkpoint": str(weight_path),
        "threshold_not_applied": True,
        "binarization_threshold_for_masks_only": float(threshold),
    }


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    sc = cfg["segmentation"]
    device = get_device()

    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default=sc["weight_path"])
    ap.add_argument("--arch", default=sc["arch"])
    ap.add_argument("--encoder", default=sc["encoder"])
    ap.add_argument("--threshold", type=float, default=sc["threshold"])
    ap.add_argument("--index", default=str(cfg["paths"]["splits_dir"] / "all.csv"))
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip images whose mask PNG already exists (default: on)",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="rewrite mask_manifest.csv every N newly written masks",
    )
    ap.add_argument(
        "--save-probability-maps",
        action="store_true",
        default=False,
        help="also write versioned float16 vessel probabilities (does not replace binary masks)",
    )
    ap.add_argument(
        "--probability-dir",
        default=None,
        help="directory for *.npy float16 maps (default: data/vessel_prob_v1)",
    )
    ap.add_argument(
        "--probability-manifest",
        default=None,
        help="JSON manifest path (default: data/manifests/vessel_prob_v1.json)",
    )
    args = ap.parse_args()

    weight_path = Path(args.weight)
    if not weight_path.is_absolute():
        weight_path = cfg["_root"] / weight_path
    if not weight_path.exists():
        raise SystemExit(
            f"Segmentation weight not found: {weight_path}\n"
            "Put a pretrained checkpoint there, or train one with src.segmentation.train."
        )

    model = build_model(
        args.arch, args.encoder, encoder_weights=None, in_channels=sc["in_channels"], classes=1
    ).to(device)
    state = torch.load(str(weight_path), map_location=device)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    model.eval()

    tf = transforms.Compose(
        [transforms.Resize(tuple(sc["img_size"])), transforms.ToTensor()]
    )

    df = pd.read_csv(args.index)
    masks_dir: Path = cfg["paths"]["masks_dir"]
    out = masks_dir / "mask_manifest.csv"
    records: list[dict] = []
    prob_records: list[dict] = []
    have_prob: set[str] = set()
    prob_dir = None
    prob_manifest_path = None
    prob_extra: dict = {}
    if args.save_probability_maps:
        from src.classify.next_architecture.vessel_maps import (
            VESSEL_PROB_VERSION,
            save_prob_map_f16,
            write_probability_manifest,
        )
        from src.utils.common import sha256_file

        prob_dir = Path(args.probability_dir) if args.probability_dir else (cfg["_root"] / "data" / "vessel_prob_v1")
        prob_dir.mkdir(parents=True, exist_ok=True)
        prob_manifest_path = (
            Path(args.probability_manifest)
            if args.probability_manifest
            else (cfg["_root"] / "data" / "manifests" / "vessel_prob_v1.json")
        )
        prob_extra = {
            "segmentation_checkpoint": str(weight_path.resolve()),
            "segmentation_checkpoint_sha256": sha256_file(weight_path),
            "probability_dtype": "float16",
            "probability_map_version": VESSEL_PROB_VERSION,
            "note": "Values are sigmoid vessel probabilities BEFORE 0.20 binarization, resized to original HxW.",
        }
        if args.resume and prob_manifest_path.exists():
            old = json.loads(prob_manifest_path.read_text(encoding="utf-8"))
            prob_records = list(old.get("records") or [])
            have_prob = {str(r["image_path"]) for r in prob_records if "image_path" in r}
            print(f"[resume] {len(prob_records)} probability maps already in manifest")
    if args.resume and out.exists():
        records = pd.read_csv(out).to_dict("records")
        print(f"[resume] {len(records)} masks already in manifest")

    done_images = {r["image_path"] for r in records}
    new_since_ckpt = 0
    skipped = 0

    def _flush_ckpts() -> None:
        pd.DataFrame(records).to_csv(out, index=False)
        if args.save_probability_maps and prob_manifest_path is not None:
            from src.classify.next_architecture.vessel_maps import write_probability_manifest

            write_probability_manifest(prob_manifest_path, prob_records, prob_extra)

    def _ensure_prob_record() -> None:
        if not args.save_probability_maps or need_prob:
            return
        if img_path in have_prob:
            return
        if prob_path_expected is not None and prob_path_expected.exists():
            from src.classify.next_architecture.vessel_maps import VESSEL_PROB_VERSION

            _upsert_record(
                prob_records,
                have_prob,
                _prob_record_from_npy(
                    img_path,
                    prob_path_expected,
                    weight_path,
                    tuple(sc["img_size"]),
                    args.threshold,
                    VESSEL_PROB_VERSION,
                ),
            )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="masks"):
        img_path = row["image_path"]
        prob_path_expected = None
        if args.save_probability_maps:
            prob_path_expected = prob_dir / (Path(unique_mask_name(img_path)).stem + ".npy")
        need_prob = bool(prob_path_expected is not None and not prob_path_expected.exists())
        if args.resume and img_path in done_images and not need_prob:
            _ensure_prob_record()
            skipped += 1
            continue
        existing = find_existing_mask(masks_dir, img_path) if args.resume else None
        mask_path = existing if existing is not None else (masks_dir / unique_mask_name(img_path))
        if existing is not None and not need_prob:
            records.append(
                {
                    "image_path": img_path,
                    "mask_path": str(mask_path),
                    "label": int(row["label"]),
                    "split": row["split"],
                    "source": row.get("source", ""),
                }
            )
            done_images.add(img_path)
            _ensure_prob_record()
            skipped += 1
            new_since_ckpt += 1
            if new_since_ckpt >= args.checkpoint_every:
                _flush_ckpts()
                new_since_ckpt = 0
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:  # unreadable image -> skip but log
            print(f"[skip] {img_path}: {e}")
            continue
        orig_w, orig_h = img.size
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
        mask = post_process(prob, args.threshold, sc["min_area"], sc["morph_close_kernel"])
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        if args.save_probability_maps:
            from src.classify.next_architecture.vessel_maps import (
                VESSEL_PROB_VERSION,
                save_prob_map_f16,
            )

            prob_full = cv2.resize(prob.astype("float32"), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            prob_path = prob_dir / (Path(unique_mask_name(img_path)).stem + ".npy")
            save_prob_map_f16(prob_path, prob_full)
            from src.classify.next_architecture.vessel_maps import VESSEL_PROB_VERSION

            _upsert_record(
                prob_records,
                have_prob,
                {
                    "image_path": img_path,
                    "prob_path": str(prob_path),
                    "orig_width": orig_w,
                    "orig_height": orig_h,
                    "output_shape": [orig_h, orig_w],
                    "probability_map_version": VESSEL_PROB_VERSION,
                    "preprocessing_version": f"resize_{tuple(sc['img_size'])}_sigmoid_then_resize_orig",
                    "segmentation_checkpoint": str(weight_path),
                    "threshold_not_applied": True,
                    "binarization_threshold_for_masks_only": float(args.threshold),
                },
            )

        cv2.imwrite(str(mask_path), mask)
        records.append(
            {
                "image_path": img_path,
                "mask_path": str(mask_path),
                "label": int(row["label"]),
                "split": row["split"],
                "source": row.get("source", ""),
            }
        )
        done_images.add(img_path)
        new_since_ckpt += 1
        if new_since_ckpt >= args.checkpoint_every:
            _flush_ckpts()
            new_since_ckpt = 0

    manifest = pd.DataFrame(records)
    manifest.to_csv(out, index=False)
    print(f"[done] {len(manifest)} masks ({skipped} skipped) -> {masks_dir}")
    print(f"[manifest] {out}")
    if args.save_probability_maps:
        from src.classify.next_architecture.vessel_maps import write_probability_manifest

        write_probability_manifest(prob_manifest_path, prob_records, prob_extra)
        print(f"[prob] {len(prob_records)} maps -> {prob_dir}")
        print(f"[prob-manifest] {prob_manifest_path}")


if __name__ == "__main__":
    main()
