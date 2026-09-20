#!/usr/bin/env python
"""Section E: prove the expected mask is genuinely the mask of its RGB image.

Filename matching is not evidence. This re-runs the FROZEN segmentation checkpoint with the exact
historical preprocessing and compares the fresh prediction separately against
  A. the registered (wrong) mask
  B. the proposed expected (correct) mask
If B does not strongly beat A, the repair is not justified and this script fails loudly.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.morphology import skeletonize
from torchvision import transforms

sys.path.insert(0, "/Users/moniaz/niki")
from src.segmentation.infer_masks import post_process  # noqa: E402
from src.segmentation.models import build_model  # noqa: E402
from src.utils.common import get_device, load_config  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"
OUT.mkdir(parents=True, exist_ok=True)
PAT = re.compile(r"^(?P<stem>.+)_(?P<h>[0-9a-f]{8})\.png$")
SEED = 20260920

cfg = load_config()
sc = cfg["segmentation"]
SIZE = int(sc["img_size"][0])
device = get_device()
print(f"device={device} arch={sc['arch']} encoder={sc['encoder']} size={SIZE} "
      f"thr={sc['threshold']} min_area={sc['min_area']} close_k={sc['morph_close_kernel']}")
ck = ROOT / sc["weight_path"]
print(f"checkpoint: {ck}  sha256={hashlib.sha256(ck.read_bytes()).hexdigest()[:16]}  exists={ck.exists()}")

model = build_model(sc["arch"], sc["encoder"], None, int(sc["in_channels"]), 1).to(device).eval()
model.load_state_dict(torch.load(str(ck), map_location=device))
tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])

feat = pd.read_csv(ROOT / "data/features/biomarker_features.csv",
                   usecols=["image_path", "mask_path", "split", "source", "label"])
mdir = Path(os.path.dirname(feat.mask_path.iloc[0]))
feat["_stem"] = [os.path.splitext(os.path.basename(p))[0] for p in feat.image_path]
parsed = [PAT.match(os.path.basename(p)) for p in feat.mask_path]
feat["_mstem"] = [m.group("stem") if m else None for m in parsed]
aff = feat[feat._stem != feat._mstem].copy()
aff["_exp"] = [next((p.name for p in mdir.iterdir()
                     if (PAT.match(p.name) and PAT.match(p.name).group("stem") == s)), None)
               for s in aff._stem]
print(f"affected rows: {len(aff)}  with a unique expected mask: {int(aff._exp.notna().sum())}")

# ---- deterministic stratified sample -------------------------------------------------
rng = np.random.default_rng(SEED)
aff = aff.sort_values("image_path").reset_index(drop=True)
# force-include the different-label donors and any row whose expected mask is missing
forced = aff[aff._exp.isna()]
parts = [aff[aff._exp.notna()].groupby(["source", "split"], group_keys=False)
         .apply(lambda g: g.sample(min(len(g), 12), random_state=SEED))]
sample_aff = pd.concat(parts + [forced]).drop_duplicates("image_path").head(60)
ctrl = feat[feat._stem == feat._mstem].sort_values("image_path")
ctrl = (ctrl.groupby(["source", "split"], group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 12), random_state=SEED))).head(60)
print(f"verification sample: {len(sample_aff)} affected + {len(ctrl)} control")


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def cldice(a, b):
    sa, sb = skeletonize(a), skeletonize(b)
    if sa.sum() == 0 or sb.sum() == 0:
        return float("nan")
    p = (sa & b).sum() / sa.sum()
    r = (sb & a).sum() / sb.sum()
    return float(2 * p * r / (p + r)) if (p + r) else float("nan")


def fresh(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    w0, h0 = img.size
    with torch.no_grad():
        prob = torch.sigmoid(model(tf(img).unsqueeze(0).to(device))).squeeze().cpu().numpy()
    m = post_process(prob, float(sc["threshold"]), int(sc["min_area"]), int(sc["morph_close_kernel"]))
    import cv2
    return cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST) > 0


rows = []
for kind, df in (("affected", sample_aff), ("control", ctrl)):
    for _, r in df.iterrows():
        f = fresh(r.image_path)
        wrong = np.array(Image.open(r.mask_path).convert("L")) > 127
        rec = {"kind": kind, "image_path": r.image_path, "source": r.source, "split": r.split,
               "label": int(r.label),
               "dice_vs_registered": dice(f, wrong), "cldice_vs_registered": cldice(f, wrong),
               "disagree_vs_registered": float((f != wrong).mean())}
        exp_name = r._exp if kind == "affected" else None
        if exp_name:
            exp = np.array(Image.open(mdir / exp_name).convert("L")) > 127
            rec |= {"dice_vs_expected": dice(f, exp), "cldice_vs_expected": cldice(f, exp),
                    "disagree_vs_expected": float((f != exp).mean()),
                    "expected_sha_matches_fresh": hashlib.sha256(exp.tobytes()).hexdigest() ==
                                                 hashlib.sha256(f.tobytes()).hexdigest()}
        rows.append(rec)
    print(f"  {kind}: {len(df)} done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(OUT / "task1_fresh_inference_verification.csv", index=False)

print("\n" + "=" * 100)
print("E. FRESH INFERENCE vs the two candidate masks")
print("=" * 100)
A = R[R.kind == "affected"]
print(f"affected rows n={len(A)}")
print(f"  Dice vs REGISTERED (wrong) mask : mean={A.dice_vs_registered.mean():.4f} "
      f"median={A.dice_vs_registered.median():.4f}")
print(f"  Dice vs EXPECTED   (right) mask : mean={A.dice_vs_expected.mean():.4f} "
      f"median={A.dice_vs_expected.median():.4f}")
print(f"  clDice vs REGISTERED            : mean={A.cldice_vs_registered.mean():.4f}")
print(f"  clDice vs EXPECTED              : mean={A.cldice_vs_expected.mean():.4f}")
print(f"  disagreement vs REGISTERED      : mean={A.disagree_vs_registered.mean():.4f}")
print(f"  disagreement vs EXPECTED        : mean={A.disagree_vs_expected.mean():.4f}")
print(f"  rows where EXPECTED beats REGISTERED on Dice: "
      f"{int((A.dice_vs_expected > A.dice_vs_registered).sum())}/{len(A)}")
print(f"  exact bitwise identity with fresh inference (expected): "
      f"{int(A.expected_sha_matches_fresh.sum())}/{len(A)}")

C = R[R.kind == "control"]
print(f"\ncontrol rows n={len(C)}  Dice vs their own registered mask: "
      f"mean={C.dice_vs_registered.mean():.4f} median={C.dice_vs_registered.median():.4f}")

verdict = (A.dice_vs_expected.mean() > A.dice_vs_registered.mean() + 0.05 and
           (A.dice_vs_expected > A.dice_vs_registered).mean() > 0.8)
print(f"\nCANONICAL_MAPPING_VERIFIED_BY_FRESH_INFERENCE = {'YES' if verdict else 'NO'}")
json.dump({"n_affected": len(A), "n_control": len(C),
           "dice_expected_mean": float(A.dice_vs_expected.mean()),
           "dice_registered_mean": float(A.dice_vs_registered.mean()),
           "control_dice_mean": float(C.dice_vs_registered.mean()),
           "verified": bool(verdict)}, open(OUT / "task1_fresh_inference.json", "w"), indent=2)
sys.exit(0 if verdict else 1)
