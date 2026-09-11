"""Branch A: turn each vessel mask into a row of numeric vascular biomarkers (PVBM).

Reads the mask manifest from the segmentation step, computes PVBM geometrical
biomarkers (+ fractal dimensions + simple density features) per image, and writes
a feature table CSV that Branch A classifiers train on.

PVBM geometrical VBMs (compute_geomVBMs):
    area, tortuosity_index, median_tortuosity, overall_length,
    median_branching_angle, n_startpoints, n_endpoints, n_intersections

ROI modes:
    whole : use the full segmentation, optic-disc center approximated at image center.
    disc  : use PVBM DiscSegmenter to locate the optic disc and build zones (falls
            back to 'whole' per image if disc detection fails).

Usage:
    python -m src.biomarker.extract_pvbm
    python -m src.biomarker.extract_pvbm --roi disc
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.morphology import skeletonize
from tqdm import tqdm

from src.utils.common import append_csv_rows, ensure_dirs, load_config

META_COLS = {"image_path", "mask_path", "label", "split", "source"}

GEOM_COLUMNS = [
    "area",
    "tortuosity_index",
    "median_tortuosity",
    "overall_length",
    "median_branching_angle",
    "n_startpoints",
    "n_endpoints",
    "n_intersections",
]


def load_binary_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


def density_features(mask: np.ndarray) -> dict[str, float]:
    """Cheap, always-available features that do not depend on PVBM."""
    h, w = mask.shape
    total = float(mask.sum())
    feats = {
        "vessel_density": total / (h * w),
        "vessel_pixels": total,
    }
    # Regional density in a 3x3 grid (captures "clustering in a zone").
    ys = np.array_split(np.arange(h), 3)
    xs = np.array_split(np.arange(w), 3)
    for i, yy in enumerate(ys):
        for j, xx in enumerate(xs):
            block = mask[yy[0] : yy[-1] + 1, xx[0] : xx[-1] + 1]
            feats[f"density_r{i}c{j}"] = float(block.mean())
    return feats


def geom_features(mask: np.ndarray, roi_mode: str) -> dict[str, float]:
    """PVBM geometrical + fractal biomarkers. Returns NaNs on failure."""
    from PVBM.GeometryAnalysis import GeometricalVBMs

    seg = mask.astype(float)  # {0,1}
    skeleton = skeletonize(seg > 0).astype(int)
    h, w = mask.shape
    xc, yc = w // 2, h // 2
    radius = max(8, min(h, w) // 8)

    geom = GeometricalVBMs()
    seg_roi, skel_roi = seg, skeleton

    if roi_mode == "disc":
        try:
            from PVBM.DiscSegmenter import DiscSegmenter

            segmenter = DiscSegmenter()
            # API returns center, radius, roi, zones_ABC given a fundus/segmentation.
            center, radius, roi, zones_ABC = segmenter.post_processing(seg)
            xc, yc = int(center[0]), int(center[1])
            seg_roi, skel_roi = geom.apply_roi(
                segmentation=seg, skeleton=skeleton, zones_ABC=zones_ABC, roi=roi
            )
        except Exception:  # noqa: BLE001 - any disc-detection failure -> whole image
            seg_roi, skel_roi = seg, skeleton

    out = {c: np.nan for c in GEOM_COLUMNS}
    try:
        # PVBM graph walk can exceed default recursion on dense vessel trees.
        import sys

        _prev = sys.getrecursionlimit()
        if _prev < 10000:
            sys.setrecursionlimit(10000)
        try:
            vbms, _ = geom.compute_geomVBMs(
                blood_vessel=seg_roi, skeleton=skel_roi, xc=xc, yc=yc, radius=radius
            )
        finally:
            sys.setrecursionlimit(_prev)
        for name, val in zip(GEOM_COLUMNS, vbms):
            out[name] = float(val)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"geom VBM failed: {e}")

    # Fractal dimensions (capacity / entropy / correlation) if available.
    try:
        from PVBM.FractalAnalysis import MultifractalVBMs

        fractal = MultifractalVBMs(n_rotations=25, optimize=True, min_proba=0.0001, maxproba=0.9999)
        d0, d1, d2, sl = fractal.compute_multifractals(seg_roi.astype(float))
        out["fractal_d0"] = float(d0)
        out["fractal_d1"] = float(d1)
        out["fractal_d2"] = float(d2)
        out["singularity_length"] = float(sl)
    except Exception:  # noqa: BLE001 - fractal module optional
        pass

    return out


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", choices=["whole", "disc"], default=cfg["biomarker"]["roi"])
    ap.add_argument(
        "--manifest", default=str(cfg["paths"]["masks_dir"] / "mask_manifest.csv")
    )
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="continue from biomarker_features.partial.csv (default: on)",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="flush completed rows to the partial CSV every N images",
    )
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"Mask manifest not found: {manifest}. Run src.segmentation.infer_masks first.")

    out_path = cfg["paths"]["features_dir"] / "biomarker_features.csv"
    partial_path = cfg["paths"]["features_dir"] / "biomarker_features.partial.csv"
    if out_path.exists():
        print(f"[skip] final features already exist -> {out_path}")
        return

    done_images: set[str] = set()
    if args.resume and partial_path.exists():
        done_images = set(pd.read_csv(partial_path, usecols=["image_path"])["image_path"])
        print(f"[resume] {len(done_images)} biomarker rows already checkpointed")

    df = pd.read_csv(manifest)
    pending = df[~df["image_path"].isin(done_images)]
    if pending.empty:
        if partial_path.exists():
            partial_path.rename(out_path)
            print(f"[done] finalized {len(done_images)} rows -> {out_path}")
        return

    buffer: list[dict] = []
    for _, r in tqdm(pending.iterrows(), total=len(pending), desc="biomarkers"):
        mask = load_binary_mask(r["mask_path"])
        feats: dict[str, float] = {
            "image_path": r["image_path"],
            "mask_path": r["mask_path"],
            "label": int(r["label"]),
            "split": r["split"],
            "source": r.get("source", ""),
        }
        feats.update(density_features(mask))
        feats.update(geom_features(mask, args.roi))
        buffer.append(feats)
        if len(buffer) >= args.checkpoint_every:
            append_csv_rows(partial_path, buffer)
            buffer.clear()

    if buffer:
        append_csv_rows(partial_path, buffer)

    if partial_path.exists():
        partial_path.rename(out_path)
    n_feat = len(
        [c for c in pd.read_csv(out_path, nrows=1).columns if c not in META_COLS]
    )
    print(f"[done] {len(pd.read_csv(out_path))} rows x {n_feat} features -> {out_path}")


if __name__ == "__main__":
    main()
