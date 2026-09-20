#!/usr/bin/env python
"""Task 5A (H): compare regenerated masks against the frozen SEG_CURRENT_V1 masks.

Three independent comparisons, as required: byte (sha256) equality, pixel equality and
vessel-pixel equality. Any disagreement is a nondeterminism finding and must stop the task.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
PRIV = f"{ROOT}/_private_audit"
SCRATCH = f"{PRIV}/repro_scratch/SEG_CURRENT_V1_REPRO"


def sha(p: str) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "subset"
    canon = pd.read_csv(f"{ROOT}/data/masks/mask_manifest_canonical_v1.csv")
    regen = pd.read_csv(f"{SCRATCH}/mask_manifest.csv")
    if mode == "full":
        subset = canon[["image_path"]].copy()
    else:
        subset = pd.read_csv(f"{PRIV}/seg_current_v1_repro_index.csv")
    print(f"mode                  : {mode}")
    print(f"subset rows           : {len(subset)}")
    print(f"frozen mapping rows   : {len(canon)}")
    print(f"regenerated rows      : {len(regen)}")
    if "generation_id" in regen.columns:
        print(f"regenerated for       : {regen.generation_id.unique().tolist()}")

    frozen = dict(zip(canon.image_path, canon.mask_path))
    made = dict(zip(regen.image_path, regen.mask_path))
    missing = [p for p in subset.image_path if p not in made]
    if missing:
        print(f"FATAL: {len(missing)} subset images were not regenerated")
        raise SystemExit(2)

    rows = []
    for p in subset.image_path:
        f = frozen[p]
        r = made[p]
        fa = np.array(Image.open(f).convert("L"))
        ra = np.array(Image.open(r).convert("L"))
        sha_eq = sha(f) == sha(r)
        pix_eq = bool(np.array_equal(fa, ra))
        vp_eq = int((fa > 127).sum()) == int((ra > 127).sum())
        rows.append({
            "stable_image_id": hashlib.sha256(str(p).encode()).hexdigest()[:16],
            "mask_sha_equal": sha_eq, "pixel_equal": pix_eq, "vessel_pixels_equal": vp_eq,
            "frozen_vessel_pixels": int((fa > 127).sum()),
            "regenerated_vessel_pixels": int((ra > 127).sum()),
            "shape_equal": fa.shape == ra.shape,
        })
    R = pd.DataFrame(rows)
    R.to_csv(f"{PRIV}/seg_current_v1_repro_rows.csv", index=False)
    n = len(R)
    bits = int(R.mask_sha_equal.sum())
    pix = int(R.pixel_equal.sum())
    vps = int(R.vessel_pixels_equal.sum())
    shapes = int(R.shape_equal.sum())

    print()
    print("=" * 100)
    print("H. REPRODUCIBILITY — REGENERATED vs FROZEN SEG_CURRENT_V1")
    print("=" * 100)
    print(f"  SUBSET_N                    : {n}")
    print(f"  BITWISE_IDENTICAL_N (sha256): {bits}")
    print(f"  PIXEL_IDENTICAL_N           : {pix}")
    print(f"  VESSEL_PIXEL_IDENTICAL_N    : {vps}")
    print(f"  SHAPE_IDENTICAL_N           : {shapes}")
    print(f"  disagreements               : {n - bits}")
    if n - bits:
        bad = R[~R.mask_sha_equal]
        print()
        print("  MISMATCHES (first 10):")
        print(bad.head(10).to_string(index=False))
        print()
        print("  deterministic output was expected; STOP and investigate nondeterminism")
    else:
        print()
        print("  *** OUTPUT IS DETERMINISTIC: every regenerated mask is byte-identical ***")

    result = {
        "MODE": mode,
        "SUBSET_N": n,
        "BITWISE_IDENTICAL_N": bits,
        "PIXEL_IDENTICAL_N": pix,
        "VESSEL_PIXEL_IDENTICAL_N": vps,
        "SHAPE_IDENTICAL_N": shapes,
        "DETERMINISTIC": bits == n,
        "GATE": "PASS" if bits == n else "FAIL",
    }
    name = ("seg_current_v1_full_repro_result.json" if mode == "full"
            else "seg_current_v1_repro_result.json")
    json.dump(result, open(f"{PRIV}/{name}", "w"), indent=2)
    print()
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
