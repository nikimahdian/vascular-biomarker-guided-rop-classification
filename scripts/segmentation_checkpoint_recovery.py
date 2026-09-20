#!/usr/bin/env python
"""Task 4B (E/P/Q): identify the surviving checkpoint and attempt to recover the other
candidate segmentation checkpoints referenced by configs/config.yaml.

Part 1  architecture probe against the surviving checkpoint
Part 2  connectivity + download test for the 10 pretrained_weight_ids
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
WEIGHT = f"{ROOT}/weights/best_weight_DeepLabV3+_resize_27"
OUT = f"{ROOT}/_private/historical_seg_candidates"
os.makedirs(OUT, exist_ok=True)


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


print("=" * 100)
print("PART 1. SURVIVING CHECKPOINT: IDENTITY AND ARCHITECTURE")
print("=" * 100)
print(f"  path   : {WEIGHT}")
print(f"  size   : {os.path.getsize(WEIGHT):,} bytes")
print(f"  mtime  : {time.ctime(os.path.getmtime(WEIGHT))}")
print(f"  sha256 : {sha(WEIGHT)}")

raw = torch.load(WEIGHT, map_location="cpu")
print(f"  object type : {type(raw)}")
state = raw
if isinstance(raw, dict) and "state_dict" in raw:
    print(f"  wrapper keys: {[k for k in raw.keys() if k != 'state_dict']}")
    for k in raw.keys():
        if k != "state_dict":
            v = raw[k]
            print(f"    {k} = {str(v)[:200]}")
    state = raw["state_dict"]
elif isinstance(raw, dict):
    print(f"  top-level keys: {list(raw.keys())[:12]}")
print(f"  state_dict entries: {len(state)}")
print(f"  first keys: {list(state.keys())[:6]}")

from src.segmentation.models import build_model  # noqa: E402

arch_results = {}
for arch in ["DeepLabV3Plus", "MAnet", "UnetPlusPlus", "Unet", "FPN", "PAN"]:
    try:
        m = build_model(arch, "resnet34", encoder_weights=None, in_channels=3, classes=1)
        m.load_state_dict(state, strict=True)
        arch_results[arch] = "STRICT_MATCH"
        print(f"  {arch:16s} STRICT_MATCH")
    except Exception as e:  # noqa: BLE001
        arch_results[arch] = f"NO: {type(e).__name__}"
        print(f"  {arch:16s} NO  ({str(e)[:70]})")

print()
print(f"  ==> ARCHITECTURE = "
      f"{[k for k, v in arch_results.items() if v == 'STRICT_MATCH']}")

print()
print("=" * 100)
print("PART 2. CANDIDATE CHECKPOINT RECOVERY FROM THE CONFIGURED GOOGLE DRIVE IDS")
print("=" * 100)
import yaml  # noqa: E402

cfg = yaml.safe_load(open(f"{ROOT}/configs/config.yaml", encoding="utf-8"))
ids = cfg["segmentation"].get("pretrained_weight_ids", [])
print(f"  configured pretrained_weight_ids: {len(ids)}")
for i, fid in enumerate(ids, 1):
    print(f"    {i:2d}. {fid}")

print()
try:
    import gdown
    print(f"  gdown version: {gdown.__version__}")
except Exception as e:  # noqa: BLE001
    print(f"  gdown unavailable: {e}")
    gdown = None

before = set(os.listdir(OUT))
downloaded = []
if gdown is not None:
    for i, fid in enumerate(ids, 1):
        t0 = time.time()
        try:
            gdown.download(id=fid, output=OUT + "/", quiet=True, fuzzy=False)
            ok = True
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [{i:2d}] {fid} download error: {type(e).__name__}: {str(e)[:90]}")
        after = set(os.listdir(OUT))
        new = after - before
        print(f"  [{i:2d}] {fid}  ok={ok}  new_files={sorted(new)}  "
              f"{time.time() - t0:.1f}s")
        before = after
        downloaded += sorted(new)

print()
print("  files now in the candidate dir:")
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    if os.path.isfile(p):
        print(f"    {f:60s} {os.path.getsize(p):14,d} B  sha256={sha(p)[:16]}")

summary = {
    "SURVIVING_CHECKPOINT": WEIGHT,
    "SURVIVING_CHECKPOINT_SHA256": sha(WEIGHT),
    "SURVIVING_CHECKPOINT_MTIME": time.ctime(os.path.getmtime(WEIGHT)),
    "ARCHITECTURE_MATCH": arch_results,
    "ARCHITECTURE": [k for k, v in arch_results.items() if v == "STRICT_MATCH"],
    "CONFIGURED_CANDIDATE_IDS": len(ids),
    "CANDIDATE_FILES_RECOVERED": sorted(os.listdir(OUT)),
    "RECOVERED_N": len([f for f in os.listdir(OUT)
                        if os.path.isfile(os.path.join(OUT, f))]),
}
json.dump(summary, open(f"{ROOT}/_private_audit/task4b_checkpoint_recovery.json", "w"),
          indent=2, default=str)
print()
for k, v in summary.items():
    print(f"  {k}: {v}")
