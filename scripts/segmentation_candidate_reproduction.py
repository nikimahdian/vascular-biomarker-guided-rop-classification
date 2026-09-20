#!/usr/bin/env python
"""Task 4B (J/K/L): candidate-generation test.

Re-runs vessel segmentation from the SAME checkpoint on a deterministic validation subset
of the 8,260 unaffected rows, then sweeps the binarization threshold and post-processing
and asks whether any configuration reproduces the HISTORICAL vessel pixel count.

Control: the pipeline is first verified to reproduce the CURRENT mask file bitwise at the
recorded configuration (threshold 0.20, min_area 50, close 3). If the control passes, the
harness is faithful and a threshold that reproduces historical values is a real recovery.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
F = f"{ROOT}/data/features"
PRIV = f"{ROOT}/_private_audit"

WEIGHT = f"{ROOT}/weights/best_weight_DeepLabV3+_resize_27"
ARCH = "MAnet"
ENCODER = "resnet34"
IMG_SIZE = (256, 256)
CUR_THRESHOLD = 0.20
MIN_AREA = 50
CLOSE_K = 3

import hashlib

print("checkpoint sha256 =", hashlib.sha256(open(WEIGHT, "rb").read()).hexdigest())
print("size =", os.path.getsize(WEIGHT), " mtime =", time.ctime(os.path.getmtime(WEIGHT)))


def post_process(prob, threshold, min_area, close_k):
    """Verbatim copy of src/segmentation/infer_masks.py:32-44."""
    binary = (prob > threshold).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    if close_k and close_k > 1:
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8)
        )
    return cleaned


def build():
    from src.segmentation.models import build_model
    from src.utils.common import get_device

    dev = get_device()
    model = build_model(ARCH, ENCODER, encoder_weights=None, in_channels=3, classes=1).to(dev)
    state = torch.load(WEIGHT, map_location=dev)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    model.eval()
    return model, dev


def main() -> None:
    hist = pd.read_csv(f"{F}/biomarker_features.csv")
    corr = pd.read_csv(f"{F}/biomarker_features_historical_equivalent_corrected_v1.csv")
    can = pd.read_csv(f"{ROOT}/data/masks/mask_manifest_canonical_v1.csv")
    j = hist[["image_path", "mask_path", "area", "split", "source", "label"]].merge(
        corr[["image_path", "area", "mask_path"]].rename(
            columns={"area": "area_c", "mask_path": "mask_c"}), on="image_path")
    j["affected"] = j.mask_path != j.mask_c
    j["abs_delta"] = (j.area - j.area_c).abs()
    una = j[~j.affected].copy()

    print("=" * 100)
    print("VALIDATION SUBSET SELECTION")
    print("=" * 100)
    print(f"  unaffected rows available: {len(una)}")
    with Image.open(una.mask_c.iloc[0]) as im:
        w0, h0 = im.size
    print(f"  sample mask size: {w0}x{h0}")

    def geom(p):
        with Image.open(p) as im:
            w, h = im.size
        return f"min{min(h, w)}"

    una["geometry"] = una.mask_c.map(geom)
    large = una.nlargest(40, "abs_delta")
    small = una[una.abs_delta == 0]
    mid = una[(una.abs_delta > 0)].nsmallest(40, "abs_delta")
    if len(small) == 0:
        small = una.nsmallest(20, "abs_delta")
    rest = una.drop(index=set(large.index) | set(mid.index) | set(small.index))
    rest = rest.groupby(["source", "split", "geometry"], group_keys=False).head(1)
    sel = pd.concat([large, mid, small, rest]).drop_duplicates(subset="image_path")
    print(f"  selected: {len(sel)}")
    print(f"    by source  : {sel.source.value_counts().to_dict()}")
    print(f"    by split   : {sel.split.value_counts().to_dict()}")
    print(f"    by geometry: {sel.geometry.value_counts().to_dict()}")
    print(f"    abs_delta  : min={sel.abs_delta.min():.0f} median={sel.abs_delta.median():.0f} "
          f"max={sel.abs_delta.max():.0f}  zero-delta rows={int((sel.abs_delta == 0).sum())}")
    sel.to_csv(f"{PRIV}/task4b_validation_subset.csv", index=False)

    model, dev = build()
    print(f"\n  device = {dev}")
    tf = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])

    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]
    probs: dict[str, tuple[np.ndarray, int, int]] = {}
    t0 = time.time()
    for i, r in enumerate(sel.itertuples(), 1):
        img = Image.open(r.image_path).convert("RGB")
        ow, oh = img.size
        x = tf(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            p = torch.sigmoid(model(x)).squeeze().cpu().numpy()
        probs[r.image_path] = (p.astype(np.float32), ow, oh)
        if i % 25 == 0 or i == len(sel):
            print(f"    forward {i}/{len(sel)}  {time.time() - t0:.1f}s", flush=True)
    print(f"  forward pass done in {time.time() - t0:.1f}s "
          f"({(time.time() - t0) / len(sel):.3f}s/image)")

    print()
    print("=" * 100)
    print("CONTROL: DOES THIS HARNESS REPRODUCE THE CURRENT MASK AT THRESHOLD 0.20?")
    print("=" * 100)
    exact = 0
    for r in sel.itertuples():
        p, ow, oh = probs[r.image_path]
        m = post_process(p, CUR_THRESHOLD, MIN_AREA, CLOSE_K)
        m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_NEAREST)
        cur = np.array(Image.open(r.mask_c).convert("L"))
        if np.array_equal(m, cur):
            exact += 1
    print(f"  bitwise identical to the current mask file: {exact} / {len(sel)}")
    control_ok = exact == len(sel)
    print(f"  CONTROL_PASS = {control_ok}")
    if not control_ok:
        print("  (a non-zero failure here means the harness itself is not faithful; any")
        print("   threshold result below would be uninterpretable)")

    print()
    print("=" * 100)
    print("THRESHOLD SWEEP: DOES ANY THRESHOLD REPRODUCE THE HISTORICAL area?")
    print("=" * 100)
    print(f"  {'thr':>6s} {'area_exact':>11s} {'rate':>8s} {'median|d|':>11s} "
          f"{'max|d|':>9s} {'<=1px':>7s}")
    results = []
    for thr in thresholds:
        hits, ds = 0, []
        for r in sel.itertuples():
            p, ow, oh = probs[r.image_path]
            m = post_process(p, thr, MIN_AREA, CLOSE_K)
            m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_NEAREST)
            vp = int((m > 127).sum())
            d = vp - int(r.area)
            ds.append(d)
            if d == 0:
                hits += 1
        ds = np.array(ds)
        results.append({
            "threshold": thr, "area_exact_match": hits, "n": len(sel),
            "rate": hits / len(sel), "median_abs_delta": float(np.median(np.abs(ds))),
            "max_abs_delta": int(np.abs(ds).max()), "within_1px": int((np.abs(ds) <= 1).sum()),
        })
        print(f"  {thr:6.2f} {hits:11d} {hits / len(sel):8.4f} "
              f"{np.median(np.abs(ds)):11.1f} {int(np.abs(ds).max()):9d} "
              f"{int((np.abs(ds) <= 1).sum()):7d}")

    R = pd.DataFrame(results)
    R.to_csv(f"{PRIV}/task4b_threshold_sweep.csv", index=False)
    best = R.loc[R.area_exact_match.idxmax()]
    print()
    print(f"  BEST THRESHOLD = {best.threshold:.2f}  area_exact={int(best.area_exact_match)}"
          f"/{len(sel)}  median|d|={best.median_abs_delta:.1f}")
    print(f"  CURRENT RECORDED THRESHOLD = {CUR_THRESHOLD}")
    if int(best.area_exact_match) == len(sel):
        print("  *** THRESHOLD RECOVERY CANDIDATE FOUND ***")
    else:
        print("  *** no threshold reproduces the historical pixel counts ***")

    summary = {
        "CONTROL_REPRODUCES_CURRENT_MASK": f"{exact}/{len(sel)}",
        "CONTROL_PASS": bool(control_ok),
        "DEVICE": str(dev),
        "CHECKPOINT_SHA256": hashlib.sha256(open(WEIGHT, "rb").read()).hexdigest(),
        "VALIDATION_N": len(sel),
        "BEST_THRESHOLD": float(best.threshold),
        "BEST_AREA_EXACT_MATCH": int(best.area_exact_match),
        "BEST_MEDIAN_ABS_DELTA": float(best.median_abs_delta),
        "CURRENT_THRESHOLD_AREA_EXACT":
            int(R.loc[R.threshold == CUR_THRESHOLD, "area_exact_match"].iloc[0]),
        "HISTORICAL_MASK_GENERATION_RECOVERED":
            "PARTIAL" if int(best.area_exact_match) == len(sel) else "NO",
    }
    json.dump(summary, open(f"{PRIV}/task4b_candidate_test.json", "w"), indent=2, default=str)
    print()
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
