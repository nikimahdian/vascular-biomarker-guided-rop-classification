#!/usr/bin/env python
"""Task 4 (E): build the historical-equivalent corrected PVBM feature table.

Inputs
  data/splits/all.csv                          canonical cohort + split metadata (8870)
  data/masks/mask_manifest_canonical_v1.csv    verified canonical image -> mask mapping

Feature computation is imported verbatim from src.biomarker.extract_pvbm
(load_binary_mask, density_features, geom_features) and driven with roi="whole",
which is the historical setting (configs/config.yaml: biomarker.roi).

Nothing about the feature definitions is re-implemented here.

Output
  data/features/biomarker_features_historical_equivalent_corrected_v1.csv
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = "/Users/moniaz/niki"
OUT = f"{ROOT}/data/features/biomarker_features_historical_equivalent_corrected_v1.csv"
STASH = f"{ROOT}/_private_audit/task4_reconstruction_stash.csv"

# exact historical column order, taken from data/features/biomarker_features.csv
FEATURES = [
    "vessel_density", "vessel_pixels",
    "density_r0c0", "density_r0c1", "density_r0c2",
    "density_r1c0", "density_r1c1", "density_r1c2",
    "density_r2c0", "density_r2c1", "density_r2c2",
    "area", "tortuosity_index", "median_tortuosity", "overall_length",
    "median_branching_angle", "n_startpoints", "n_endpoints", "n_intersections",
    "fractal_d0", "fractal_d1", "fractal_d2", "singularity_length",
]
META_ORDER = ["image_path", "mask_path"] + FEATURES + [
    "label", "split", "source", "group_id", "patient_id", "exam_id", "identity_level"]
ROI = "whole"


def _init() -> None:
    warnings.filterwarnings("ignore")
    sys.setrecursionlimit(10000)


def _work(item: tuple[str, str]) -> dict:
    from src.biomarker.extract_pvbm import density_features, geom_features, load_binary_mask

    image_path, mask_path = item
    rec: dict = {"image_path": image_path, "mask_path": mask_path}
    try:
        mask = load_binary_mask(mask_path)
        rec.update(density_features(mask))
        rec.update(geom_features(mask, ROI))
    except Exception as e:  # noqa: BLE001
        rec["_error"] = f"{type(e).__name__}: {e}"
    return rec


def main() -> None:
    print("=" * 100)
    print("ENVIRONMENT (must match requirements-lock.txt)")
    print("=" * 100)
    import numpy, scipy, skimage, pandas as _pd
    from importlib.metadata import version

    for pkg in ("numpy", "scipy", "scikit-image", "pandas", "pvbm", "Pillow"):
        try:
            print(f"  {pkg:14s} == {version(pkg)}")
        except Exception:  # noqa: BLE001
            print(f"  {pkg:14s} == UNKNOWN")
    print(f"  numpy.__version__={numpy.__version__} scipy={scipy.__version__} "
          f"skimage={skimage.__version__} pandas={_pd.__version__}")

    print()
    print("=" * 100)
    print("INPUTS")
    print("=" * 100)
    splits = pd.read_csv(f"{ROOT}/data/splits/all.csv")
    canon = pd.read_csv(f"{ROOT}/data/masks/mask_manifest_canonical_v1.csv")
    print(f"  all.csv      rows={len(splits)}  sha-fingerprint known: 0d4c3b3a...")
    print(f"  canonical mask mapping rows={len(canon)}")
    job = splits[["image_path", "label", "split", "source", "group_id",
                  "patient_id", "exam_id", "identity_level"]].merge(
        canon, on="image_path", how="left", validate="one_to_one")
    if job.mask_path.isna().any():
        raise SystemExit(f"FATAL: {int(job.mask_path.isna().sum())} canonical images "
                         f"have no canonical mask")
    missing = [p for p in job.mask_path if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} canonical mask files missing on disk")
    print(f"  extraction job rows={len(job)}  all masks present on disk: "
          f"{len(job) - len(missing)}/{len(job)}")

    done: dict[str, dict] = {}
    if os.path.exists(STASH):
        st = pd.read_csv(STASH)
        done = {r["image_path"]: r for r in st.to_dict("records")}
        print(f"  [resume] {len(done)} rows already in stash")
    todo = job[~job.image_path.isin(done)]
    print(f"  to extract: {len(todo)}")

    if len(todo):
        t0 = time.time()
        items = list(zip(todo.image_path, todo.mask_path))
        results: list[dict] = []
        with Pool(processes=20, initializer=_init) as pool:
            for i, rec in enumerate(pool.imap_unordered(_work, items, chunksize=8), 1):
                results.append(rec)
                if i % 500 == 0 or i == len(items):
                    el = time.time() - t0
                    print(f"    {i}/{len(items)}  {el:7.1f}s  "
                          f"eta {el / i * (len(items) - i):7.1f}s", flush=True)
                    merged = {**done, **{r["image_path"]: r for r in results}}
                    pd.DataFrame(list(merged.values())).to_csv(STASH, index=False)
        done = {**done, **{r["image_path"]: r for r in results}}
        pd.DataFrame(list(done.values())).to_csv(STASH, index=False)
        print(f"  extraction complete in {time.time() - t0:.1f}s")

    res = pd.DataFrame(list(done.values()))
    errs = res[res.get("_error").notna()] if "_error" in res.columns else res.iloc[0:0]
    print(f"  rows with extraction error: {len(errs)}")
    if len(errs):
        print(errs[["image_path", "_error"]].head(10).to_string(index=False))

    out = job.merge(res, on=["image_path", "mask_path"], how="left", validate="one_to_one")
    if "mask_path_x" in out.columns:
        raise SystemExit("FATAL: mask_path collided during merge")
    out = out[META_ORDER]
    out.to_csv(OUT, index=False)
    print()
    print(f"  wrote {OUT}")
    print(f"  rows={len(out)}  cols={out.shape[1]}  features={len(FEATURES)}")
    print(f"  byte size={os.path.getsize(OUT):,}")

    import hashlib

    h = hashlib.sha256()
    with open(OUT, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    print(f"  BYTE SHA256 = {h.hexdigest()}")


if __name__ == "__main__":
    main()
