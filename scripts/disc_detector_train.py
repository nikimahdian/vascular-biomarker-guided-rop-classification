#!/usr/bin/env python
"""Blok 2 -- optic-disc detector on HVDROPDB-OD (100 expert masks), 5-fold.

Purpose: the project's pseudo-disc (image centre, radius max(8, min(h,w)/8)) is ~2.4-2.8x too
large and ~7 disc-radii off centre. A real disc radius is needed to (a) normalise vessel width
into disc diameters and (b) restrict width measurement to the peripapillary annulus.

Protocol: 5-fold, stratified by camera (10 RetCam + 10 Neo per validation fold). 384x384.
No patient identifiers exist in HVDROPDB, so folds are image-level -- stated as a limitation.
The 5 fold models are then ensembled and applied to the 100 HVDROPDB-BV images, which are a
*different* photograph set (BV and OD images are not paired), to give DD for the width analysis.

Writes checkpoints + csv to results/hvdro_validation/disc/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/Users/moniaz/niki")
from src.segmentation.models import DiceBCELoss, build_model  # noqa: E402
from src.utils.common import get_device  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
SEG = ROOT / "data/raw/hvdro/segmentation/HVDROPDB_RetCam_Neo_Segmentation"
OUT = ROOT / "results/hvdro_validation/disc"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 384
BATCH = 8
EPOCHS = 50
PATIENCE = 10
LR = 1e-3
NFOLD = 5
SEED = 20260101
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

device = get_device()
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"device={device} size={SIZE} batch={BATCH} epochs={EPOCHS} folds={NFOLD}")


def binarize(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        a = a[..., 0]
    return (a > 127).astype(np.uint8)


def load_pairs(camera: str):
    idir = SEG / "HVDROPDB-OD" / f"{camera}_images"
    mdir = SEG / "HVDROPDB-OD" / f"{camera}_masks"
    out = []
    for p in sorted(idir.glob("*.png")):
        mp = mdir / p.name
        if mp.exists():
            out.append((p, mp, camera, p.stem))
    return out


items = load_pairs("Retcam_OpticDisc") + load_pairs("Neo_OpticDisc")
print(f"total images: {len(items)}  "
      f"(RetCam={sum(1 for i in items if i[2] == 'Retcam_OpticDisc')}, "
      f"Neo={sum(1 for i in items if i[2] == 'Neo_OpticDisc')})")


class DiscDS(Dataset):
    def __init__(self, rows, train: bool):
        self.rows = rows
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ip, mp, cam, stem = self.rows[i]
        img = Image.open(ip).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        msk = Image.open(mp).convert("L").resize((SIZE, SIZE), Image.NEAREST)
        x = np.asarray(img, dtype=np.float32) / 255.0
        y = (np.asarray(msk, dtype=np.float32) > 127).astype(np.float32)
        if self.train:
            k = np.random.randint(0, 4)
            if k:
                x, y = np.rot90(x, k), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
        x = (x - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(y))[None])


@torch.no_grad()
def predict_native(model, ip) -> np.ndarray:
    img = Image.open(ip).convert("RGB")
    w0, h0 = img.size
    x = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(device)
    prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    return np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize((w0, h0), Image.BILINEAR),
                      dtype=np.float32) / 255.0


def disc_geom(mask: np.ndarray):
    if mask.sum() == 0:
        return None
    ys, xs = np.nonzero(mask)
    return dict(cx=float(xs.mean()), cy=float(ys.mean()),
                radius=float(np.sqrt(mask.sum() / np.pi)))


# ------------------------------------------------------------------ folds
rng = np.random.default_rng(SEED)
fold_of = {}
for cam in ("Retcam_OpticDisc", "Neo_OpticDisc"):
    idx = [i for i, it in enumerate(items) if it[2] == cam]
    perm = rng.permutation(len(idx))
    for k, pos in enumerate(perm):
        fold_of[idx[pos]] = k % NFOLD
folds = pd.DataFrame([{"idx": i, "camera": items[i][2], "stem": items[i][3], "fold": fold_of[i]}
                      for i in range(len(items))])
folds.to_csv(OUT / "folds.csv", index=False)
print(folds.groupby(["fold", "camera"]).size().unstack(fill_value=0).to_string())

per_image, ckpts = [], []
t0 = time.time()
for f in range(NFOLD):
    tr = [items[i] for i in range(len(items)) if fold_of[i] != f]
    va = [items[i] for i in range(len(items)) if fold_of[i] == f]
    dl_tr = DataLoader(DiscDS(tr, True), batch_size=BATCH, shuffle=True, num_workers=0, drop_last=False)
    dl_va = DataLoader(DiscDS(va, False), batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_model("Unet", "resnet34", "imagenet", 3, 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = DiceBCELoss()
    best, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()
        model.eval()
        dices = []
        with torch.no_grad():
            for xb, yb in dl_va:
                p = torch.sigmoid(model(xb.to(device)))
                pb = (p > 0.5).float().view(-1)
                yb = yb.to(device).view(-1)
                inter = (pb * yb).sum()
                dices.append(float((2 * inter + 1e-6) / (pb.sum() + yb.sum() + 1e-6)))
        vd = float(np.mean(dices))
        if vd > best + 1e-4:
            best, bad = vd, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  fold{f} ep{ep + 1:3d} loss={tot / max(len(dl_tr), 1):.4f} valDice@384={vd:.4f} "
                  f"best={best:.4f} bad={bad}", flush=True)
        if bad >= PATIENCE:
            print(f"  fold{f} early stop at ep{ep + 1}", flush=True)
            break
    model.load_state_dict(best_state)
    ck = OUT / f"disc_unet_fold{f}.pt"
    torch.save(best_state, ck)
    ckpts.append(ck)
    model.eval()

    for ip, mp, cam, stem in va:
        prob = predict_native(model, ip)
        pm = prob > 0.5
        gtm = binarize(np.array(Image.open(mp)))
        s = pm.sum() + gtm.sum()
        dice = float(2 * (pm & gtm).sum() / s) if s else 1.0
        inter = int((pm & gtm).sum())
        union = int((pm | gtm).sum())
        iou = float(inter / union) if union else 1.0
        gh, ph = disc_geom(gtm), disc_geom(pm)
        row = {"fold": f, "camera": cam, "stem": stem, "val_dice_at384": best, "dice_native": dice, "iou_native": iou}
        if gh and ph:
            row.update({
                "gt_radius_px": gh["radius"], "pred_radius_px": ph["radius"],
                "radius_ratio": ph["radius"] / gh["radius"] if gh["radius"] else np.nan,
                "centre_err_px": float(np.hypot(ph["cx"] - gh["cx"], ph["cy"] - gh["cy"])),
                "centre_err_dd": float(np.hypot(ph["cx"] - gh["cx"], ph["cy"] - gh["cy"])
                                       / (2 * gh["radius"])) if gh["radius"] else np.nan,
            })
        else:
            row.update({"gt_radius_px": np.nan, "pred_radius_px": np.nan, "radius_ratio": np.nan,
                        "centre_err_px": np.nan, "centre_err_dd": np.nan})
        per_image.append(row)
    print(f"[fold {f}] done  elapsed {time.time() - t0:.0f}s", flush=True)

pidf = pd.DataFrame(per_image)
pidf.to_csv(OUT / "disc_cv_per_image.csv", index=False)
summ = pidf.groupby("camera").agg(
    n=("stem", "count"), dice_native=("dice_native", "mean"), iou_native=("iou_native", "mean"),
    gt_radius_px=("gt_radius_px", "mean"), pred_radius_px=("pred_radius_px", "mean"),
    radius_ratio=("radius_ratio", "mean"), centre_err_px=("centre_err_px", "mean"),
    centre_err_dd=("centre_err_dd", "mean")).reset_index()
summ.to_csv(OUT / "disc_cv_summary.csv", index=False)
print("\n=== disc detector, 5-fold native-resolution ===")
print(summ.to_string(index=False))
print(f"\npublished reference (authors' U-Net): RetCam 0.93 / Neo 0.92")
print(f"project pseudo-disc: radius ratio 2.35 (RetCam) / 2.84 (Neo), centre offset 7.5 / 7.1 disc-radii")

# ------------------------------------------------- ensemble -> HVDROPDB-BV images
bv = []
for cam in ("RetCam_Vessels", "Neo_Vessels"):
    for p in sorted((SEG / "HVDROPDB-BV" / f"{cam}_images").glob("*.png")):
        bv.append((p, cam))
print(f"\napplying 5-fold ensemble to {len(bv)} HVDROPDB-BV images ...")
models = []
for f in range(NFOLD):
    m = build_model("Unet", "resnet34", None, 3, 1).to(device)
    m.load_state_dict(torch.load(str(OUT / f"disc_unet_fold{f}.pt"), map_location=device))
    models.append(m.eval())

rows = []
with torch.no_grad():
    for p, cam in bv:
        probs = np.mean([predict_native(m, p) for m in models], axis=0)
        pm = probs > 0.5
        img = Image.open(p)
        w0, h0 = img.size
        g = disc_geom(pm)
        r = {"camera": cam, "stem": p.stem, "w": w0, "h": h0,
             "disc_area_frac": float(pm.mean()),
             "pseudo_radius_px": max(8.0, min(h0, w0) / 8.0)}
        if g:
            r.update({"pred_cx": g["cx"], "pred_cy": g["cy"], "pred_radius_px": g["radius"],
                      "pred_dd_px": 2 * g["radius"],
                      "centre_offset_px": float(np.hypot(g["cx"] - w0 / 2, g["cy"] - h0 / 2))})
        else:
            r.update({"pred_cx": np.nan, "pred_cy": np.nan, "pred_radius_px": np.nan,
                      "pred_dd_px": np.nan, "centre_offset_px": np.nan})
        rows.append(r)
bdf = pd.DataFrame(rows)
bdf.to_csv(OUT / "bv_disc_predictions.csv", index=False)
print(bdf.groupby("camera")[["pred_radius_px", "pred_dd_px", "disc_area_frac", "centre_offset_px"]]
      .mean().round(2).to_string())

json.dump({
    "size": SIZE, "batch": BATCH, "epochs": EPOCHS, "patience": PATIENCE, "lr": LR,
    "nfolds": NFOLD, "seed": SEED, "arch": "Unet", "encoder": "resnet34", "encoder_weights": "imagenet",
    "camera_stratified": True, "patient_grouped": False,
    "limitation": "HVDROPDB carries no patient IDs; folds are image-level, so a same-eye pair could span folds.",
    "published_reference_dice": {"RetCam": 0.93, "Neo": 0.92},
    "note": "BV images are a different photograph set from OD images; the ensemble is applied to BV to supply DD.",
}, open(OUT / "manifest.json", "w"), indent=2)
print(f"\n[done] -> {OUT}  total {time.time() - t0:.0f}s")
