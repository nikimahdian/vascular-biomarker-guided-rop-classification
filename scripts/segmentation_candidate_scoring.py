#!/usr/bin/env python
"""Task 4B (J/K/L) definitive: score every recovered checkpoint against the HISTORICAL
feature table on rows where the historical and current masks GENUINELY DIFFER.

Earlier scoring was contaminated by zero-delta rows (which any faithful reconstruction of
the current masks matches trivially). This version samples exclusively from the 5,801
unaffected rows whose historical `area` differs from the current one, so a score of 1.0
can only be achieved by a checkpoint that actually reproduces the historical generation.
"""
from __future__ import annotations

import hashlib
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
CANDS = f"{ROOT}/_private/historical_seg_candidates"
PRIV = f"{ROOT}/_private_audit"
IMG_SIZE = (256, 256)
MIN_AREA, CLOSE_K = 50, 3
ENCODERS = ["resnet34", "resnet50", "resnet101", "efficientnet-b0", "se_resnext50_32x4d"]
ARCHS = ["MAnet", "DeepLabV3Plus", "Unet", "UnetPlusPlus", "DeepLabV3", "Linknet"]
THRESHOLDS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
              0.22, 0.25, 0.30, 0.35, 0.40]


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def post_process(prob, thr, min_area, close_k):
    binary = (prob > thr).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    if close_k and close_k > 1:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE,
                                   np.ones((close_k, close_k), np.uint8))
    return cleaned


def main() -> None:
    from src.segmentation.models import build_model
    from src.utils.common import get_device

    dev = get_device()
    hist = pd.read_csv(f"{F}/biomarker_features.csv")
    corr = pd.read_csv(f"{F}/biomarker_features_historical_equivalent_corrected_v1.csv")
    k = hist[["image_path", "mask_path", "area", "split", "source", "label"]].merge(
        corr[["image_path", "area", "mask_path"]].rename(
            columns={"area": "area_c", "mask_path": "mask_c"}), on="image_path")
    k["affected"] = k.mask_path != k.mask_c
    una = k[~k.affected].copy()
    diff = una[una.area != una.area_c].copy()
    diff["abs_delta"] = (diff.area - diff.area_c).abs()
    print(f"unaffected rows          : {len(una)}")
    print(f"  with area != current   : {len(diff)}   <-- evaluation universe")
    print(f"  with area == current   : {len(una) - len(diff)}")

    diff = diff.sort_values("abs_delta", ascending=False)
    head = diff.head(150)
    tail = diff.tail(150)
    sel = pd.concat([head, tail]).drop_duplicates(subset="image_path")
    sel.to_csv(f"{PRIV}/task4b_eval_subset_diffonly.csv", index=False)
    print(f"  evaluation subset      : {len(sel)}")
    print(f"    abs_delta min/median/max = {sel.abs_delta.min():.0f} / "
          f"{sel.abs_delta.median():.0f} / {sel.abs_delta.max():.0f}")
    print(f"    by source {sel.source.value_counts().to_dict()}")
    print(f"    by split  {sel.split.value_counts().to_dict()}")

    imgs = {r.image_path: Image.open(r.image_path).convert("RGB")
            for r in sel.itertuples()}
    tf = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
    target = dict(zip(sel.image_path, sel.area.astype(int)))

    files = sorted(f for f in os.listdir(CANDS) if not f.endswith(".part"))
    print(f"\ncandidate files: {len(files)}")
    rows, meta = [], {}
    for fn in files:
        path = os.path.join(CANDS, fn)
        t0 = time.time()
        digest = sha(path)
        raw = torch.load(path, map_location="cpu")
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        found = None
        for a in ARCHS:
            for e in ENCODERS:
                try:
                    m = build_model(a, e, encoder_weights=None, in_channels=3, classes=1)
                    m.load_state_dict(state, strict=True)
                    found = (a, e)
                    break
                except Exception:  # noqa: BLE001
                    continue
            if found:
                break
        entry = {"file": fn, "size": os.path.getsize(path), "sha256": digest,
                 "architecture": found[0] if found else "NO_STRICT_MATCH_ANY_ARCH_ENCODER",
                 "encoder": found[1] if found else "UNKNOWN",
                 "load_probe_seconds": round(time.time() - t0, 1)}
        print(f"\n  {fn}  size={entry['size']:,}  sha={digest[:16]}")
        print(f"    arch/encoder = {entry['architecture']} / {entry['encoder']}")
        if not found:
            meta[fn] = entry
            continue
        a, e = found
        model = build_model(a, e, encoder_weights=None, in_channels=3, classes=1).to(dev)
        model.load_state_dict(state)
        model.eval()
        probs = {}
        for p in sel.image_path:
            x = tf(imgs[p]).unsqueeze(0).to(dev)
            with torch.no_grad():
                probs[p] = torch.sigmoid(model(x)).squeeze().cpu().numpy().astype(np.float32)
        del model
        best = (None, -1, None)
        for thr in THRESHOLDS:
            hits, ds = 0, []
            for r in sel.itertuples():
                m = post_process(probs[r.image_path], thr, MIN_AREA, CLOSE_K)
                w, h = imgs[r.image_path].size
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                d = int((m > 127).sum()) - target[r.image_path]
                ds.append(d)
                if d == 0:
                    hits += 1
            ds = np.abs(np.array(ds))
            rows.append({"file": fn, "architecture": a, "encoder": e, "threshold": thr,
                         "area_exact": hits, "n": len(sel), "rate": hits / len(sel),
                         "median_abs_delta": float(np.median(ds)),
                         "max_abs_delta": int(ds.max())})
            print(f"      thr={thr:.2f}  exact={hits:4d}/{len(sel)}  "
                  f"median|d|={np.median(ds):9.1f}  max|d|={int(ds.max()):7d}")
            if hits > best[1]:
                best = (thr, hits, float(np.median(ds)))
        entry["best_threshold"] = best[0]
        entry["best_area_exact"] = int(best[1])
        entry["best_rate"] = best[1] / len(sel)
        entry["best_median_abs_delta"] = best[2]
        meta[fn] = entry
        print(f"    ==> BEST thr={best[0]} exact={best[1]}/{len(sel)}")

    R = pd.DataFrame(rows)
    R.to_csv(f"{PRIV}/task4b_diffonly_scores.csv", index=False)
    json.dump(meta, open(f"{PRIV}/task4b_diffonly_candidates.json", "w"),
              indent=2, default=str)
    print()
    print("=" * 100)
    print("RANKING ON THE GENUINELY-DIFFERING SUBSET")
    print("=" * 100)
    if len(R):
        b = R.loc[R.groupby("file").area_exact.idxmax()].sort_values(
            "area_exact", ascending=False)
        print(b.to_string(index=False))
        top = b.iloc[0]
        gate = "YES" if int(top.area_exact) == len(sel) else "NO"
        print()
        print(f"  TOP: {top.file} ({top.architecture}/{top.encoder}) thr={top.threshold}")
        print(f"  AREA_EXACT_MATCH_RATE = {top.area_exact}/{len(sel)} = {top.rate:.4f}")
        print(f"  SUBSET_GATE = {gate}")
    else:
        gate = "NO"
        print("  no scoreable candidate")
    json.dump({"SUBSET_GATE": gate, "EVAL_UNIVERSE": len(diff)},
              open(f"{PRIV}/task4b_gate2.json", "w"), indent=2)


if __name__ == "__main__":
    main()
