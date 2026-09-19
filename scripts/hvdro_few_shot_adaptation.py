#!/usr/bin/env python
"""T1.2b - a *fair* test of few-shot camera adaptation.

hvdro_adapt.py (v1) fine-tuned for a fixed 40 epochs at a fixed decision threshold of
0.20, which is the threshold tuned for the *original* model. That protocol is unfair to
adaptation in two ways: (i) no early stopping, so 40 images overfit; (ii) the sigmoid
calibration shifts after fine-tuning, so a frozen 0.20 threshold can destroy Dice even
when the mask improved.

This script repeats the experiment properly:
  * each fold carves an inner validation split out of the adaptation pool,
  * the best epoch is chosen on that inner split (never on the outer test),
  * the decision threshold is also chosen on the inner split from a grid,
  * the outer test set is used exactly once, for the final number.

Two questions:
  Q1 per-camera      : adapting to a single camera, does it beat the zero-shot model
                       on the same held-out images?
  Q2 pooled two-camera: can ONE adapted model serve both cameras at once?

External data only (HVDROPDB). Nothing here touches the project's data, splits, locked
test, or weights. Writes to results/hvdro_validation/adapt2/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/Users/moniaz/niki")
from src.segmentation.infer_masks import post_process  # noqa: E402
from src.segmentation.models import build_model  # noqa: E402
from src.utils.common import get_device, load_config  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
SEG = ROOT / "data/raw/hvdro/segmentation/HVDROPDB_RetCam_Neo_Segmentation"
OUT = ROOT / "results/hvdro_validation/adapt2"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 256
EPOCHS = 30
BATCH = 2
LR = 5.0e-5
N_FOLDS = 5
SEED = 0
DEPLOY_THR = 0.20
THR_GRID = np.round(np.arange(0.05, 0.95, 0.05), 2)

cfg = load_config()
sc = cfg["segmentation"]
MIN_AREA, CLOSE_K = int(sc["min_area"]), int(sc["morph_close_kernel"])
device = get_device()
torch.manual_seed(SEED)
np.random.seed(SEED)

ckpt = ROOT / sc["weight_path"]
print(f"device={device}  size={SIZE}  epochs={EPOCHS}  lr={LR}  folds={N_FOLDS}  deploy_thr={DEPLOY_THR}")

tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])


def fresh_model():
    m = build_model(sc["arch"], sc["encoder"], None, int(sc["in_channels"]), 1)
    m.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    return m.to(device)


def prob_map(model, img):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(tf(img).unsqueeze(0).to(device))).squeeze().cpu().numpy()


def dice_at(prob, gt_native, w, h, thr):
    m = cv2.resize(post_process(prob, thr, MIN_AREA, CLOSE_K), (w, h), interpolation=cv2.INTER_NEAREST)
    p, g = m > 0, gt_native > 0
    s = p.sum() + g.sum()
    return float(2 * (p & g).sum() / s) if s else 1.0


def load_camera(cam):
    img_dir = SEG / "HVDROPDB-BV" / f"{cam}_images"
    msk_dir = SEG / "HVDROPDB-BV" / f"{cam}_masks"
    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    out = []
    for s in stems:
        img = Image.open(img_dir / f"{s}.png").convert("RGB")
        gt = np.array(Image.open(msk_dir / f"{s}.png").convert("L"))
        out.append({"cam": cam, "stem": s, "img": img, "gt": gt, "w": img.size[0], "h": img.size[1]})
    return out


def stratified_folds(items, k, seed):
    """k folds that keep the camera mix constant inside every fold."""
    rng = np.random.RandomState(seed)
    by_cam: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        by_cam.setdefault(it["cam"], []).append(i)
    assign = np.empty(len(items), dtype=int)
    for _, idx in by_cam.items():
        idx = np.array(idx)[rng.permutation(len(idx))]
        for f in range(k):
            assign[idx[f::k]] = f
    return [(np.where(assign != f)[0], np.where(assign == f)[0]) for f in range(k)]


def best_threshold(model, items):
    """Pick the threshold maximising mean Dice on a set the model never trained on."""
    cache = [(prob_map(model, it["img"]), it) for it in items]
    scores = {t: float(np.mean([dice_at(p, it["gt"], it["w"], it["h"], t) for p, it in cache])) for t in THR_GRID}
    t = max(scores, key=scores.get)
    return float(t), scores


def train_once(adapt_pool, seed):
    """Inner split, early stopping on inner Dice, threshold chosen on inner split."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(adapt_pool))
    n_inner = max(4, int(round(0.2 * len(adapt_pool))))
    inner_idx, fit_idx = order[:n_inner], order[n_inner:]
    fit = [adapt_pool[i] for i in fit_idx]
    inner = [adapt_pool[i] for i in inner_idx]

    model = fresh_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()

    best = {"inner_dice": -1.0, "epoch": -1, "state": None}
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.RandomState(seed * 1000 + ep).permutation(len(fit))
        for b in range(0, len(perm), BATCH):
            xs, ys = [], []
            for j in perm[b : b + BATCH]:
                it = fit[j]
                x = tf(it["img"])
                if np.random.rand() < 0.5:
                    x = torch.flip(x, dims=[2])
                y = torch.from_numpy(
                    (cv2.resize(it["gt"], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
                ).unsqueeze(0)
                xs.append(x)
                ys.append(y)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(torch.stack(xs).to(device)), torch.stack(ys).to(device))
            loss.backward()
            opt.step()
        d_in = float(np.mean([dice_at(prob_map(model, it["img"]), it["gt"], it["w"], it["h"], DEPLOY_THR) for it in inner]))
        if d_in > best["inner_dice"]:
            best = {
                "inner_dice": d_in, "epoch": ep,
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            }
        print(f"      ep {ep + 1:02d}/{EPOCHS} inner Dice@0.20 = {d_in:.4f}"
              f"{'  *' if best['epoch'] == ep else ''}", flush=True)

    model.load_state_dict(best["state"])
    thr, _ = best_threshold(model, inner)
    return model, best["epoch"], best["inner_dice"], thr


# ---------------------------------------------------------------- baselines
cache: dict[str, list] = {}
base_model = fresh_model()
print("computing zero-shot probabilities for all 100 images ...", flush=True)
for cam in ("RetCam_Vessels", "Neo_Vessels"):
    items = load_camera(cam)
    for it in items:
        it["zs_prob"] = prob_map(base_model, it["img"])
    cache[cam] = items
    print(f"  {cam}: {len(items)}", flush=True)
del base_model

rows = []

# ---------------------------------------------------------------- Q1: per camera
for cam in ("RetCam_Vessels", "Neo_Vessels"):
    items = cache[cam]
    print(f"\n=== Q1 {cam} ===", flush=True)
    for fi, (tr, te) in enumerate(stratified_folds(items, N_FOLDS, SEED)):
        pool = [items[i] for i in tr]
        print(f"  fold {fi + 1}/{N_FOLDS}: fine-tuning on {len(pool)} images", flush=True)
        model, ep, d_in, thr = train_once(pool, SEED + fi)
        print(f"    chosen epoch {ep + 1} (inner Dice {d_in:.4f}), inner threshold {thr:.2f}", flush=True)
        for j in te:
            it = items[j]
            rows.append({
                "setting": "per_camera", "camera": cam, "fold": fi, "stem": it["stem"],
                "epoch": ep, "thr_adapted": thr,
                "zero_shot_dice": dice_at(it["zs_prob"], it["gt"], it["w"], it["h"], DEPLOY_THR),
                "adapted_dice": dice_at(prob_map(model, it["img"]), it["gt"], it["w"], it["h"], thr),
                "adapted_dice_at_deploy_thr": dice_at(prob_map(model, it["img"]), it["gt"], it["w"], it["h"], DEPLOY_THR),
            })
        del model

# ---------------------------------------------------------------- Q2: pooled two-camera
all_items = cache["RetCam_Vessels"] + cache["Neo_Vessels"]
print("\n=== Q2 pooled two-camera ===", flush=True)
for fi, (tr, te) in enumerate(stratified_folds(all_items, N_FOLDS, SEED)):
    pool = [all_items[i] for i in tr]
    n_rc = sum(1 for it in pool if it["cam"] == "RetCam_Vessels")
    print(f"  fold {fi + 1}/{N_FOLDS}: fine-tuning on {len(pool)} images ({n_rc} RetCam / {len(pool) - n_rc} Neo)", flush=True)
    model, ep, d_in, thr = train_once(pool, SEED + 100 + fi)
    print(f"    chosen epoch {ep + 1} (inner Dice {d_in:.4f}), inner threshold {thr:.2f}", flush=True)
    for j in te:
        it = all_items[j]
        rows.append({
            "setting": "pooled", "camera": it["cam"], "fold": fi, "stem": it["stem"],
            "epoch": ep, "thr_adapted": thr,
            "zero_shot_dice": dice_at(it["zs_prob"], it["gt"], it["w"], it["h"], DEPLOY_THR),
            "adapted_dice": dice_at(prob_map(model, it["img"]), it["gt"], it["w"], it["h"], thr),
            "adapted_dice_at_deploy_thr": dice_at(prob_map(model, it["img"]), it["gt"], it["w"], it["h"], DEPLOY_THR),
        })
    del model

df = pd.DataFrame(rows)
df.to_csv(OUT / "adapt2_per_image.csv", index=False)

summary = (
    df.groupby(["setting", "camera"])
    .agg(
        n=("stem", "count"),
        zero_shot=("zero_shot_dice", "mean"),
        adapted=("adapted_dice", "mean"),
        adapted_at_deploy_thr=("adapted_dice_at_deploy_thr", "mean"),
        mean_epoch=("epoch", lambda s: float(s.mean()) + 1),
        mean_thr=("thr_adapted", "mean"),
    )
    .reset_index()
)
summary["delta_vs_zeroshot"] = summary["adapted"] - summary["zero_shot"]
summary["delta_calibration_only"] = summary["adapted"] - summary["adapted_at_deploy_thr"]
summary.to_csv(OUT / "adapt2_summary.csv", index=False)

from scipy import stats  # noqa: E402

print("\n=== fair few-shot adaptation (early stop + threshold on inner split) ===")
print(summary.to_string(index=False))

pooled_stats = {}
for setting in ("per_camera", "pooled"):
    for cam in ("RetCam_Vessels", "Neo_Vessels"):
        g = df[(df.setting == setting) & (df.camera == cam)]
        if len(g) == 0:
            continue
        d = (g.adapted_dice - g.zero_shot_dice).to_numpy()
        try:
            w, p = stats.wilcoxon(d)
        except ValueError:
            w, p = float("nan"), float("nan")
        key = f"{setting}|{cam}"
        pooled_stats[key] = {
            "n": int(len(d)), "delta_mean": float(d.mean()), "delta_median": float(np.median(d)),
            "wilcoxon_W": float(w), "wilcoxon_p": float(p),
        }
        print(f"{key:32s} n={len(d):3d} delta mean={d.mean():+.4f} median={np.median(d):+.4f} p={p:.3g}")

json.dump(
    {
        "protocol": {
            "size": SIZE, "epochs": EPOCHS, "lr": LR, "batch": BATCH, "folds": N_FOLDS,
            "inner_frac": 0.2, "seed": SEED, "deploy_threshold": DEPLOY_THR,
            "threshold_grid": [float(t) for t in THR_GRID],
            "init": str(ckpt),
            "difference_from_v1": "v1 used fixed 40 epochs and the deploy threshold; v1 numbers are not comparable.",
        },
        "summary": summary.to_dict(orient="records"),
        "paired": pooled_stats,
    },
    open(OUT / "adapt2_manifest.json", "w"),
    indent=2,
)
print(f"\n[done] -> {OUT}")
