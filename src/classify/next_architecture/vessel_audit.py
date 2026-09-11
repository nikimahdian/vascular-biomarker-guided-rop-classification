"""Segmentation proxy audit after E4 fails. Not clinician GT. Not a reason to train E5."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.classify.next_architecture.vessel_maps import (
    TOPOLOGY_THRESHOLD_DEFAULT,
    load_prob_map,
    skeletonize,
)


def prob_map_path(prob_dir: Path, image_path: str) -> Path:
    stem = Path(image_path).stem
    digest = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:8]
    return Path(prob_dir) / f"{stem}_{digest}.npy"

PROXY_NOTE = (
    "Centerline/caliber stats compare soft maps to production binary PNGs from the same "
    "segmenter (threshold 0.20). This is not clinician-annotated Dice/IoU."
)


def _mask_path(masks_dir: Path, image_path: str) -> Path | None:
    stem = Path(image_path).stem
    digest = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:8]
    preferred = masks_dir / f"{stem}_{digest}.png"
    if preferred.exists():
        return preferred
    matches = sorted(masks_dir.glob(f"{stem}_*.png"))
    if len(matches) == 1:
        return matches[0]
    return None


def _resize_max_side(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return arr
    scale = max_side / float(m)
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    interp = cv2.INTER_NEAREST if arr.dtype == np.uint8 and arr.max() > 1 else cv2.INTER_AREA
    return cv2.resize(arr, (nw, nh), interpolation=interp)


def _centerline_recall(prob: np.ndarray, mask_u8: np.ndarray, thr: float) -> dict[str, float]:
    hard_mask = (mask_u8 > 127).astype(np.uint8)
    skel = skeletonize(hard_mask)
    skel_bin = (skel > 0.5).astype(np.uint8)
    n_skel = int(skel_bin.sum())
    if n_skel == 0:
        return {
            "n_skel": 0,
            "recall": float("nan"),
            "recall_thin": float("nan"),
            "recall_mid": float("nan"),
            "recall_thick": float("nan"),
        }
    hit = (prob >= thr) & (skel_bin > 0)
    recall = float(hit.sum() / n_skel)
    dist = cv2.distanceTransform(hard_mask, cv2.DIST_L2, 3)
    vals = dist[skel_bin > 0]
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    thin = (vals <= q1)
    mid = (vals > q1) & (vals <= q2)
    thick = vals > q2
    skel_hit = (prob >= thr)[skel_bin > 0]

    def _rec(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        return float(skel_hit[mask].mean())

    return {
        "n_skel": n_skel,
        "recall": recall,
        "recall_thin": _rec(thin),
        "recall_mid": _rec(mid),
        "recall_thick": _rec(thick),
    }


def audit_one(
    image_path: str,
    *,
    prob_dir: Path,
    masks_dir: Path,
    label: int,
    source: str,
    split: str,
    max_side: int,
    thr: float,
) -> dict[str, Any] | None:
    npy = prob_map_path(prob_dir, image_path)
    if not npy.exists():
        return None
    prob = load_prob_map(npy)
    if prob.ndim != 2:
        return None
    prob = _resize_max_side(prob, max_side)
    coverage = float((prob >= thr).mean())
    empty = bool((prob >= thr).sum() == 0)
    p_clip = np.clip(prob, 1e-6, 1 - 1e-6)
    entropy = float(-(p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip)).mean())
    border = float(np.concatenate([prob[0], prob[-1], prob[:, 0], prob[:, -1]]).mean())
    row: dict[str, Any] = {
        "image_path": image_path,
        "source": source,
        "label": int(label),
        "split": split,
        "coverage": coverage,
        "empty": empty,
        "entropy": entropy,
        "border_mean": border,
        "mean_prob": float(prob.mean()),
        "n_skel": 0,
        "recall": float("nan"),
        "recall_thin": float("nan"),
        "recall_mid": float("nan"),
        "recall_thick": float("nan"),
        "has_mask": False,
    }
    mask_path = _mask_path(masks_dir, image_path)
    if mask_path is not None and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = _resize_max_side(mask, max_side)
            if mask.shape != prob.shape:
                mask = cv2.resize(mask, (prob.shape[1], prob.shape[0]), interpolation=cv2.INTER_NEAREST)
            row["has_mask"] = True
            row.update(_centerline_recall(prob, mask, thr))
    return row


def summarize(frame: pd.DataFrame, thr: float) -> dict[str, Any]:
    def _mean(col: str) -> float:
        s = pd.to_numeric(frame[col], errors="coerce")
        return float(s.mean()) if s.notna().any() else float("nan")

    by_source = {}
    for src, part in frame.groupby("source"):
        by_source[str(src)] = {
            "n": int(len(part)),
            "empty_rate": float(part["empty"].mean()),
            "coverage_mean": float(part["coverage"].mean()),
            "centerline_recall": float(pd.to_numeric(part["recall"], errors="coerce").mean()),
            "recall_thin": float(pd.to_numeric(part["recall_thin"], errors="coerce").mean()),
            "recall_thick": float(pd.to_numeric(part["recall_thick"], errors="coerce").mean()),
            "border_mean": float(part["border_mean"].mean()),
        }
    return {
        "n": int(len(frame)),
        "n_with_mask": int(frame["has_mask"].sum()) if "has_mask" in frame else 0,
        "empty_frequency": float(frame["empty"].mean()) if len(frame) else float("nan"),
        "coverage_mean": _mean("coverage"),
        "centerline_recall_mean": _mean("recall"),
        "recall_thin_mean": _mean("recall_thin"),
        "recall_mid_mean": _mean("recall_mid"),
        "recall_thick_mean": _mean("recall_thick"),
        "by_source": by_source,
        "topology_threshold": thr,
        "clinical_dice_iou_claim": False,
        "proxy_note": PROXY_NOTE,
        "e5_recommendation": "do_not_train_E5_until_vessel_maps_help_on_val",
    }


def run_vessel_audit(
    cfg: dict[str, Any],
    *,
    max_side: int = 384,
    limit: int | None = None,
    out_json: Path | None = None,
) -> dict[str, Any]:
    splits = cfg["paths"]["splits_dir"]
    frame = pd.read_csv(splits / "all.csv")
    if limit:
        frame = frame.iloc[: int(limit)].copy()
    prob_dir = Path(cfg["paths"]["vessel_prob_dir"])
    masks_dir = Path(cfg["paths"]["masks_dir"])
    out_dir = Path(cfg["paths"]["results_dir"]) / "ladder"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "vessel_audit_rows.csv"
    done: set[str] = set()
    rows: list[dict[str, Any]] = []
    if rows_path.exists():
        old = pd.read_csv(rows_path)
        rows = old.to_dict("records")
        done = {str(r["image_path"]) for r in rows}
        print(f"[vessel-audit] resume {len(done)} rows")
    thr = float(cfg.get("vessel", {}).get("topology_threshold", TOPOLOGY_THRESHOLD_DEFAULT))
    for i, rec in enumerate(frame.itertuples(index=False), start=1):
        image_path = str(rec.image_path)
        if image_path in done:
            continue
        row = audit_one(
            image_path,
            prob_dir=prob_dir,
            masks_dir=masks_dir,
            label=int(rec.label),
            source=str(rec.source),
            split=str(getattr(rec, "split", "")),
            max_side=max_side,
            thr=thr,
        )
        if row is None:
            continue
        rows.append(row)
        done.add(image_path)
        if len(rows) % 200 == 0:
            pd.DataFrame(rows).to_csv(rows_path, index=False)
            print(f"[vessel-audit] {len(rows)}/{len(frame)}")
    table = pd.DataFrame(rows)
    table.to_csv(rows_path, index=False)
    summary = summarize(table, thr)
    summary["max_side"] = max_side
    summary["rows_csv"] = str(rows_path)
    dest = out_json or (out_dir / "vessel_audit.json")
    dest.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    print(f"[vessel-audit] {dest}")
    return summary


def _overlay_panel(rgb: np.ndarray, prob: np.ndarray, mask: np.ndarray | None, max_side: int) -> np.ndarray:
    rgb = _resize_max_side(rgb, max_side)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    elif rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    h, w = rgb.shape[:2]
    heat = _resize_max_side(np.clip(prob, 0.0, 1.0), max_side)
    if heat.shape[:2] != (h, w):
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_AREA)
    heat_u8 = np.uint8(heat * 255)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    blend = cv2.addWeighted(rgb, 0.55, color, 0.45, 0)
    if mask is None:
        mask_bgr = np.zeros_like(rgb)
    else:
        m = _resize_max_side(mask, max_side)
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return np.concatenate([rgb, blend, mask_bgr], axis=1)


def write_vessel_overlays(
    cfg: dict[str, Any],
    *,
    n_per_cell: int = 2,
    max_side: int = 384,
    seed: int = 42,
) -> dict[str, Any]:
    """Stratified RGB | heatmap | binary-mask panels. Proxy QC only."""
    splits = cfg["paths"]["splits_dir"]
    frame = pd.read_csv(splits / "all.csv")
    frame["plus"] = (frame["label"].astype(int) == int(cfg.get("plus_class_index", 2))).astype(int)
    prob_dir = Path(cfg["paths"]["vessel_prob_dir"])
    masks_dir = Path(cfg["paths"]["masks_dir"])
    out_dir = Path(cfg["paths"]["results_dir"]) / "ladder" / "vessel_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    written: list[dict[str, Any]] = []
    for (source, plus), part in frame.groupby(["source", "plus"], sort=True):
        take = min(int(n_per_cell), len(part))
        if take <= 0:
            continue
        idxs = rng.choice(part.index.to_numpy(), size=take, replace=False)
        for i, idx in enumerate(idxs):
            rec = part.loc[idx]
            image_path = str(rec["image_path"])
            npy = prob_map_path(prob_dir, image_path)
            if not npy.exists():
                continue
            bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            prob = load_prob_map(npy)
            mask_path = _mask_path(masks_dir, image_path)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
            panel = _overlay_panel(bgr, prob, mask, max_side)
            stem = Path(image_path).stem
            dest = out_dir / f"{source}_plus{int(plus)}_{i}_{stem}.png"
            cv2.imwrite(str(dest), panel)
            written.append(
                {
                    "source": str(source),
                    "plus": int(plus),
                    "image_path": image_path,
                    "overlay": str(dest),
                }
            )
    manifest = {
        "n": len(written),
        "n_per_cell": int(n_per_cell),
        "max_side": int(max_side),
        "seed": int(seed),
        "note": "RGB | soft-map heatmap | production binary mask. Not clinician GT.",
        "clinical_dice_iou_claim": False,
        "rows": written,
    }
    dest_json = out_dir / "manifest.json"
    dest_json.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"[overlays] n={len(written)} -> {out_dir}")
    return manifest
