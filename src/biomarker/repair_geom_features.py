"""Re-extract geom/fractal features for rows where geom columns are all-NaN.

Uses raised recursion limit (see extract_pvbm.geom_features). Density cols kept.

Usage:
    python -m src.biomarker.repair_geom_features
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.biomarker.extract_pvbm import GEOM_COLUMNS, geom_features, load_binary_mask
from src.utils.common import ensure_dirs, load_config

FRACTAL_COLS = ["fractal_d0", "fractal_d1", "fractal_d2", "singularity_length"]


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max rows to repair (0=all)")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    args = ap.parse_args()

    path = cfg["paths"]["features_dir"] / "biomarker_features.csv"
    df = pd.read_csv(path)
    for c in GEOM_COLUMNS + FRACTAL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    fail_mask = df[GEOM_COLUMNS].isna().all(axis=1)
    idxs = df.index[fail_mask].tolist()
    if args.limit:
        idxs = idxs[: args.limit]
    print(f"[repair] {len(idxs)} / {fail_mask.sum()} geom-failed rows (of {len(df)})")

    backup = path.with_suffix(".csv.bak_pre_geom_repair")
    if not backup.exists():
        df.to_csv(backup, index=False)
        print(f"[backup] {backup}")

    done = 0
    for i in tqdm(idxs, desc="repair geom"):
        r = df.loc[i]
        try:
            mask = load_binary_mask(r["mask_path"])
            feats = geom_features(mask, cfg["biomarker"]["roi"])
            for c, v in feats.items():
                df.at[i, c] = v
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {r['image_path']}: {e}")
        done += 1
        if done % args.checkpoint_every == 0:
            df.to_csv(path, index=False)

    df.to_csv(path, index=False)
    still = int(df[GEOM_COLUMNS].isna().all(axis=1).sum())
    print(f"[done] geom_all_nan now {still}/{len(df)} -> {path}")


if __name__ == "__main__":
    main()
