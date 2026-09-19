#!/usr/bin/env python
"""T1.2c - is the camera gap a representation problem or a decision-threshold problem?

Zero-shot, at the deployed threshold of 0.20, the segmenter reaches Dice 0.754 on RetCam
vessels but only 0.473 on Neo vessels, and the Neo failure has a clear signature:
precision 0.816 but recall only 0.338 (the model is far too conservative). Threshold 0.20
was tuned on the project's own data, not on Neo.

So: how much of the Neo gap survives if the threshold is re-selected for the camera?
This needs no training at all -- only the logistic map already produced by the network.

Choice of threshold is validated the same way as any other hyper-parameter: 5-fold CV, the
threshold chosen on 4/5 of the images and scored on the held-out 1/5. The in-sample curve
is printed separately and is explicitly descriptive only.

External data only (HVDROPDB). Read-only on the project. Writes to
results/hvdro_validation/threshold/.
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
OUT = ROOT / "results/hvdro_validation/threshold"
OUT.mkdir(parents=True, exist_ok=True)

SIZE, N_FOLDS, SEED, DEPLOY_THR = 256, 5, 0, 0.20
THR_GRID = np.round(np.arange(0.02, 0.96, 0.02), 2)

cfg = load_config()
sc = cfg["segmentation"]
MIN_AREA, CLOSE_K = int(sc["min_area"]), int(sc["morph_close_kernel"])
device = get_device()
torch.manual_seed(SEED)
np.random.seed(SEED)

ckpt = ROOT / sc["weight_path"]
print(f"device={device}  deploy_thr={DEPLOY_THR}  grid={THR_GRID[0]}..{THR_GRID[-1]}")

model = build_model(sc["arch"], sc["encoder"], None, int(sc["in_channels"]), 1).to(device).eval()
model.load_state_dict(torch.load(str(ckpt), map_location=device))
tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])


def dice_for(prob, gt, w, h, thr):
    m = cv2.resize(post_process(prob, thr, MIN_AREA, CLOSE_K), (w, h), interpolation=cv2.INTER_NEAREST)
    p, g = m > 0, gt > 0
    s = p.sum() + g.sum()
    tp, fp, fn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum()
    return (
        float(2 * tp / s) if s else 1.0,
        float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        float(p.mean()),
    )


per_image, curve_rows = [], []
for cam in ("RetCam_Vessels", "Neo_Vessels"):
    img_dir, msk_dir = SEG / "HVDROPDB-BV" / f"{cam}_images", SEG / "HVDROPDB-BV" / f"{cam}_masks"
    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    print(f"\n=== {cam}: {len(stems)} ===", flush=True)
    recs = []
    for s in stems:
        img = Image.open(img_dir / f"{s}.png").convert("RGB")
        gt = np.array(Image.open(msk_dir / f"{s}.png").convert("L"))
        with torch.no_grad():
            prob = torch.sigmoid(model(tf(img).unsqueeze(0).to(device))).squeeze().cpu().numpy()
        recs.append({"stem": s, "prob": prob, "gt": gt, "w": img.size[0], "h": img.size[1]})

    # threshold curve over all images, for description only (in-sample)
    for t in THR_GRID:
        ds = [dice_for(r["prob"], r["gt"], r["w"], r["h"], t) for r in recs]
        curve_rows.append({
            "camera": cam, "threshold": float(t),
            "dice": float(np.mean([d[0] for d in ds])),
            "recall": float(np.nanmean([d[1] for d in ds])),
            "precision": float(np.nanmean([d[2] for d in ds])),
            "pred_fraction": float(np.mean([d[3] for d in ds])),
        })

    # cross-validated threshold selection
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(recs))
    for f in range(N_FOLDS):
        te, tr = idx[f::N_FOLDS], np.setdiff1d(idx, idx[f::N_FOLDS])
        scores = {float(t): float(np.mean([dice_for(recs[i]["prob"], recs[i]["gt"], recs[i]["w"], recs[i]["h"], t)[0] for i in tr])) for t in THR_GRID}
        t_sel = max(scores, key=scores.get)
        for i in te:
            r = recs[i]
            d20, r20, p20, f20 = dice_for(r["prob"], r["gt"], r["w"], r["h"], DEPLOY_THR)
            dsel, rsel, psel, fsel = dice_for(r["prob"], r["gt"], r["w"], r["h"], t_sel)
            per_image.append({
                "camera": cam, "fold": f, "stem": r["stem"], "thr_selected": t_sel,
                "dice_at_deploy_thr": d20, "recall_at_deploy_thr": r20, "precision_at_deploy_thr": p20, "pred_frac_at_deploy_thr": f20,
                "dice_at_selected_thr": dsel, "recall_at_selected_thr": rsel, "precision_at_selected_thr": psel, "pred_frac_at_selected_thr": fsel,
                "gt_fraction": float((r["gt"] > 0).mean()),
            })
    print("  fold threshold selection done", flush=True)

df = pd.DataFrame(per_image)
df.to_csv(OUT / "threshold_per_image.csv", index=False)
pd.DataFrame(curve_rows).to_csv(OUT / "threshold_curve.csv", index=False)

summary = (
    df.groupby("camera")
    .agg(
        n=("stem", "count"),
        thr_selected_mean=("thr_selected", "mean"),
        dice_deploy_thr=("dice_at_deploy_thr", "mean"),
        dice_selected_thr=("dice_at_selected_thr", "mean"),
        recall_deploy=("recall_at_deploy_thr", "mean"),
        recall_selected=("recall_at_selected_thr", "mean"),
        precision_deploy=("precision_at_deploy_thr", "mean"),
        precision_selected=("precision_at_selected_thr", "mean"),
        pred_frac_deploy=("pred_frac_at_deploy_thr", "mean"),
        pred_frac_selected=("pred_frac_at_selected_thr", "mean"),
        gt_fraction=("gt_fraction", "mean"),
    )
    .reset_index()
)
summary["delta_dice"] = summary["dice_selected_thr"] - summary["dice_deploy_thr"]
summary.to_csv(OUT / "threshold_summary.csv", index=False)

print("\n=== cross-validated threshold re-selection (no training) ===")
print(summary.to_string(index=False))

# in-sample optimum, descriptive only
for cam in ("RetCam_Vessels", "Neo_Vessels"):
    c = pd.DataFrame(curve_rows)
    c = c[c.camera == cam]
    b = c.loc[c.dice.idxmax()]
    print(f"{cam}: in-sample best threshold {b.threshold:.2f} -> Dice {b.dice:.4f} "
          f"(recall {b.recall:.3f}, precision {b.precision:.3f})")

json.dump(
    {
        "protocol": {"size": SIZE, "folds": N_FOLDS, "seed": SEED, "deploy_threshold": DEPLOY_THR,
                     "grid": [float(t) for t in THR_GRID], "init": str(ckpt),
                     "note": "threshold chosen on 4/5 and scored on 1/5; no weights are changed."},
        "summary": summary.to_dict(orient="records"),
    },
    open(OUT / "threshold_manifest.json", "w"),
    indent=2,
)
print(f"\n[done] -> {OUT}")
