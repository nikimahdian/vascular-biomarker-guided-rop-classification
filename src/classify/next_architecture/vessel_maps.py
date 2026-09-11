"""Soft vessel probability maps and deterministic derived E5 channels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

VESSEL_PROB_VERSION = "vessel_prob_v1"
DERIVED_VERSION = "vessel_derived_v1"
TOPOLOGY_THRESHOLD_DEFAULT = 0.20


def _finite(arr: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return out


def resize_prob_map(prob: np.ndarray, height: int, width: int) -> np.ndarray:
    if prob.shape[0] == height and prob.shape[1] == width:
        return _finite(prob)
    resized = cv2.resize(prob.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(_finite(resized), 0.0, 1.0)


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen-like thinning via OpenCV iterative erode/open."""
    img = (binary > 0).astype(np.uint8) * 255
    if img.sum() == 0:
        return np.zeros_like(img, dtype=np.float32)
    skeleton = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    remaining = img.copy()
    max_iter = int(max(img.shape) * 2 + 8)
    for _ in range(max_iter):
        eroded = cv2.erode(remaining, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(remaining, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        if cv2.countNonZero(eroded) == 0 or np.array_equal(eroded, remaining):
            remaining = eroded
            break
        remaining = eroded
    return (skeleton > 0).astype(np.float32)


def normalized_distance_transform(binary: np.ndarray) -> np.ndarray:
    """Distance to background, divided by image diagonal (size-independent)."""
    mask = (binary > 0).astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    h, w = mask.shape
    diagonal = float(np.hypot(h, w) + 1e-8)
    return _finite(dist / diagonal)


def curvature_heatmap(skeleton: np.ndarray, *, blur: int = 5) -> np.ndarray:
    """Unsigned discrete curvature on skeleton pixels, then slightly dilated."""
    skel = (skeleton > 0.5).astype(np.uint8)
    heat = np.zeros(skel.shape, dtype=np.float32)
    if skel.sum() == 0:
        return heat
    ys, xs = np.where(skel > 0)
    # 8-neighborhood turning proxy: count neighbors (endpoints=1, junctions>=3)
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    neighbor = cv2.filter2D(skel.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # High curvature proxy: deviation of neighbor count from 2 (smooth centerline)
    heat = np.abs(neighbor - 2.0) * skel.astype(np.float32)
    heat = np.clip(heat / 6.0, 0.0, 1.0)
    if blur and blur > 1:
        k = blur if blur % 2 else blur + 1
        heat = cv2.GaussianBlur(heat, (k, k), 0)
    if ys.size:
        heat = heat / max(float(heat.max()), 1e-8)
    return _finite(heat)


def derived_channels(
    probability: np.ndarray,
    *,
    topology_threshold: float = TOPOLOGY_THRESHOLD_DEFAULT,
) -> np.ndarray:
    """Return (4, H, W) float32 in documented ranges.

    0: vessel probability [0, 1]
    1: skeleton of thresholded map [0, 1]  (threshold documented, not clinical truth)
    2: normalized distance / caliber proxy [0, ~0.5]
    3: curvature heatmap [0, 1]
    """
    prob = np.clip(_finite(np.asarray(probability, dtype=np.float32)), 0.0, 1.0)
    if prob.ndim != 2:
        raise ValueError(f"probability map must be 2D, got {prob.shape}")
    hard = (prob >= topology_threshold).astype(np.uint8)
    skel = skeletonize(hard)
    dist = normalized_distance_transform(hard)
    curv = curvature_heatmap(skel)
    stacked = np.stack([prob, skel, dist, curv], axis=0)
    if not np.isfinite(stacked).all():
        stacked = _finite(stacked)
    return stacked.astype(np.float32)


def save_prob_map_f16(path: Path, prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(_finite(prob), 0.0, 1.0).astype(np.float16)
    np.save(path, clipped)


def load_prob_map(path: Path) -> np.ndarray:
    arr = np.load(path)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def write_probability_manifest(path: Path, records: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": VESSEL_PROB_VERSION, "n": len(records), **extra, "records": records}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def vessel_quality_audit(prob_maps: list[np.ndarray], sources: list[str], labels: list[int]) -> dict[str, Any]:
    rows = []
    for prob, source, label in zip(prob_maps, sources, labels):
        p = np.clip(prob.astype(np.float32), 0.0, 1.0)
        coverage = float((p >= TOPOLOGY_THRESHOLD_DEFAULT).mean())
        p_clip = np.clip(p, 1e-6, 1 - 1e-6)
        entropy = float(-(p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip)).mean())
        border = float(np.concatenate([p[0], p[-1], p[:, 0], p[:, -1]]).mean())
        hard = (p >= TOPOLOGY_THRESHOLD_DEFAULT).astype(np.uint8)
        ncc, _, _, _ = cv2.connectedComponentsWithStats(hard, connectivity=8) if hard.any() else (1, None, None, None)
        rows.append(
            {
                "source": source,
                "label": int(label),
                "coverage": coverage,
                "entropy": entropy,
                "border_mean": border,
                "n_components": int(max(ncc - 1, 0)),
                "empty": bool(hard.sum() == 0),
            }
        )
    import pandas as pd

    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "n": len(frame),
        "empty_frequency": float(frame["empty"].mean()) if len(frame) else float("nan"),
        "coverage_mean": float(frame["coverage"].mean()) if len(frame) else float("nan"),
        "entropy_mean": float(frame["entropy"].mean()) if len(frame) else float("nan"),
        "by_source": frame.groupby("source")["coverage"].mean().to_dict() if len(frame) else {},
        "by_class": frame.groupby("label")["coverage"].mean().to_dict() if len(frame) else {},
        "topology_threshold": TOPOLOGY_THRESHOLD_DEFAULT,
        "clinical_dice_iou_claim": False,
    }
    return summary
