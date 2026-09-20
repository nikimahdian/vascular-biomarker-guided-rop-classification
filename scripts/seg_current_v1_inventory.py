#!/usr/bin/env python
"""Task 5A (D/E/G): private byte-level inventories for the SEG_CURRENT_V1 generation.

For every canonical image: the RGB bytes actually processed, and the mask bytes actually
produced. Identity is content-based (sha256), never filename-based.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
PRIV = f"{ROOT}/_private_audit"
GENERATION = "SEG_CURRENT_V1"
os.makedirs(PRIV, exist_ok=True)


def file_sha(p: str) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def stable_id(text: str, n: int = 16) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:n]


def probe_image(p: str):
    try:
        with Image.open(p) as im:
            w, h = im.size
        return p, file_sha(p), int(w), int(h), ""
    except Exception as e:  # noqa: BLE001
        return p, "", -1, -1, f"{type(e).__name__}: {e}"


def probe_mask(p: str):
    try:
        with Image.open(p) as im:
            arr = np.array(im.convert("L"))
        w, h = int(arr.shape[1]), int(arr.shape[0])
        vals = np.unique(arr)
        vp = int((arr > 127).sum())
        return p, file_sha(p), w, h, vp, "|".join(str(v) for v in vals[:6]), ""
    except Exception as e:  # noqa: BLE001
        return p, "", -1, -1, -1, "", f"{type(e).__name__}: {e}"


def main() -> None:
    t0 = time.time()
    splits = pd.read_csv(f"{ROOT}/data/splits/all.csv")
    canon = pd.read_csv(f"{ROOT}/data/masks/mask_manifest_canonical_v1.csv")
    job = splits[["image_path", "label", "split", "source", "group_id",
                  "patient_id", "exam_id", "identity_level"]].merge(
        canon, on="image_path", how="left", validate="one_to_one")
    print(f"canonical rows                 : {len(splits)}")
    print(f"canonical mask mapping rows    : {len(canon)}")
    print(f"joined rows                    : {len(job)}")
    print(f"images with no canonical mask  : {int(job.mask_path.isna().sum())}")
    if job.mask_path.isna().any() or len(job) != 8870:
        raise SystemExit("FATAL: canonical pairing is not complete at 8870")

    print("\nhashing RGB inputs ...", flush=True)
    img_rows = {}
    with get_context("fork").Pool(20) as pool:
        for i, (p, sha, w, h, err) in enumerate(
                pool.imap_unordered(probe_image, list(job.image_path), chunksize=16), 1):
            img_rows[p] = (sha, w, h, err)
            if i % 2000 == 0:
                print(f"  {i}/{len(job)}  {time.time() - t0:.0f}s", flush=True)
    print(f"  inputs hashed in {time.time() - t0:.0f}s")

    print("\nhashing masks ...", flush=True)
    t1 = time.time()
    mask_rows = {}
    with get_context("fork").Pool(20) as pool:
        for i, (p, sha, w, h, vp, vals, err) in enumerate(
                pool.imap_unordered(probe_mask, list(job.mask_path), chunksize=16), 1):
            mask_rows[p] = (sha, w, h, vp, vals, err)
            if i % 2000 == 0:
                print(f"  {i}/{len(job)}  {time.time() - t1:.0f}s", flush=True)
    print(f"  masks hashed in {time.time() - t1:.0f}s")

    inputs, masks = [], []
    for r in job.itertuples():
        isha, iw, ih, ierr = img_rows[r.image_path]
        msha, mw, mh, vp, vals, merr = mask_rows[r.mask_path]
        sid = stable_id(r.image_path)
        inputs.append({
            "stable_image_id": sid, "image_sha256": isha, "width": iw, "height": ih,
            "source": r.source, "split": r.split, "group_hash": stable_id(r.group_id),
            "probe_error": ierr,
        })
        masks.append({
            "stable_image_id": sid, "image_sha256": isha, "mask_path": r.mask_path,
            "mask_sha256": msha, "mask_width": mw, "mask_height": mh,
            "vessel_pixel_count": vp, "mask_values": vals, "generation_id": GENERATION,
            "probe_error": merr,
        })

    I = pd.DataFrame(inputs)
    M = pd.DataFrame(masks)
    I.to_csv(f"{PRIV}/seg_current_v1_inputs.csv", index=False)
    M.to_csv(f"{PRIV}/seg_current_v1_masks.csv", index=False)

    print()
    print("=" * 100)
    print("D. INPUT INVENTORY")
    print("=" * 100)
    print(f"  rows                       : {len(I)}")
    print(f"  unique stable_image_id     : {I.stable_image_id.nunique()}")
    print(f"  unique image_sha256        : {I.image_sha256.nunique()}")
    print(f"  probe errors               : {int((I.probe_error != '').sum())}")
    print(f"  dimensions                 : "
          f"{I.groupby(['width', 'height']).size().to_dict()}")
    print(f"  by source                  : {I.source.value_counts().to_dict()}")
    print(f"  by split                   : {I.split.value_counts().to_dict()}")
    print(f"  duplicate image bytes (same sha, different image): "
          f"{int((I.image_sha256.duplicated()).sum())}")
    print(f"  -> {PRIV}/seg_current_v1_inputs.csv")

    print()
    print("=" * 100)
    print("E. MASK INVENTORY")
    print("=" * 100)
    print(f"  rows                       : {len(M)}")
    print(f"  unique stable_image_id     : {M.stable_image_id.nunique()}")
    print(f"  unique mask_sha256         : {M.mask_sha256.nunique()}")
    print(f"  unique mask_path           : {M.mask_path.nunique()}")
    print(f"  probe errors               : {int((M.probe_error != '').sum())}")
    print(f"  mask pixel values          : {M.mask_values.value_counts().to_dict()}")
    print(f"  mask dims match image dims : "
          f"{int(((M.mask_width == I.width.values) & (M.mask_height == I.height.values)).sum())}"
          f" / {len(M)}")
    print(f"  vessel_pixel_count min/median/max : "
          f"{M.vessel_pixel_count.min()} / {M.vessel_pixel_count.median():.0f} / "
          f"{M.vessel_pixel_count.max()}")
    print(f"  total vessel pixels        : {int(M.vessel_pixel_count.sum()):,}")
    print(f"  duplicate mask bytes       : {int(M.mask_sha256.duplicated().sum())}")
    print(f"  -> {PRIV}/seg_current_v1_masks.csv")

    print()
    print("=" * 100)
    print("G. CONTENT-BASED IDENTITY CHECK")
    print("=" * 100)
    ok = (len(M) == 8870 and M.stable_image_id.nunique() == 8870
          and M.image_sha256.nunique() == 8870 and M.mask_sha256.nunique() == 8870
          and int((M.probe_error != '').sum()) == 0
          and int((I.probe_error != '').sum()) == 0)
    print(f"  8870 rows, unique image identity, unique input bytes, unique mask bytes: {ok}")
    print(f"  identity tuple is (image_sha256, {GENERATION}, mask_sha256) - not the filename")

    summary = {
        "generation_id": GENERATION,
        "input_rows": len(I),
        "input_unique_image_sha256": int(I.image_sha256.nunique()),
        "input_dimensions": {f"{k[0]}x{k[1]}": int(v)
                             for k, v in I.groupby(["width", "height"]).size().items()},
        "input_source_counts": I.source.value_counts().to_dict(),
        "input_split_counts": I.split.value_counts().to_dict(),
        "mask_rows": len(M),
        "mask_unique_sha256": int(M.mask_sha256.nunique()),
        "mask_values": M.mask_values.value_counts().to_dict(),
        "vessel_pixel_total": int(M.vessel_pixel_count.sum()),
        "identity_ok": bool(ok),
    }
    json.dump(summary, open(f"{PRIV}/seg_current_v1_inventory.json", "w"),
              indent=2, default=str)
    print(f"\n  wrote {PRIV}/seg_current_v1_inventory.json")
    print(f"  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
