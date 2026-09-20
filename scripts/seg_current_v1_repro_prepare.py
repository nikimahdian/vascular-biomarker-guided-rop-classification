#!/usr/bin/env python
"""Task 5A (H): build the deterministic stratified reproducibility subset.

Covers every source, every acquisition geometry and every split, with at least 300 images.
Selection is deterministic (sorted, fixed per-stratum quota) so the subset is reproducible.

The private inventory carries no raw paths by design, so image identity is re-derived from
data/splits/all.csv via the same stable-id hash.
"""
from __future__ import annotations

import hashlib
import json
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
PRIV = f"{ROOT}/_private_audit"
TARGET = 300
PER_STRATUM = 12

GEOM = {(640, 480): "min480_640x480", (1280, 960): "min960_1280x960",
        (1440, 1080): "min1080_1440x1080", (1600, 1200): "min1200_1600x1200",
        (1240, 1240): "min1240_1240x1240"}


def sid(path: str) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    splits = pd.read_csv(f"{ROOT}/data/splits/all.csv")
    splits["stable_image_id"] = splits.image_path.map(sid)
    inv = pd.read_csv(f"{PRIV}/seg_current_v1_inputs.csv")
    df = splits[["image_path", "stable_image_id", "label", "split", "source"]].merge(
        inv[["stable_image_id", "width", "height", "image_sha256"]],
        on="stable_image_id", how="left", validate="one_to_one")
    print(f"joined rows: {len(df)}  missing dims: {int(df.width.isna().sum())}")
    df["geometry"] = [GEOM.get((int(w), int(h)), f"{int(w)}x{int(h)}")
                      for w, h in zip(df.width, df.height)]
    df = df.sort_values("image_path").reset_index(drop=True)

    g = df.groupby(["source", "geometry", "split"]).size()
    print(f"strata present (source x geometry x split): {len(g)}")

    picked = [sub.head(PER_STRATUM) for _, sub in df.groupby(["source", "geometry", "split"])]
    sel = pd.concat(picked).drop_duplicates(subset="image_path")

    if len(sel) < TARGET:
        chosen = set(sel.image_path)
        rest = df[~df.image_path.isin(chosen)]
        for _, sub in rest.groupby(["source", "geometry", "split"]):
            if len(sel) >= TARGET:
                break
            take = sub.head(TARGET - len(sel))
            sel = pd.concat([sel, take])

    sel = sel.sort_values("image_path").reset_index(drop=True)
    sel[["image_path", "label", "split", "source"]].to_csv(
        f"{PRIV}/seg_current_v1_repro_index.csv", index=False)

    cover = sel.groupby(["source", "geometry", "split"]).ngroups
    print()
    print("=" * 100)
    print("H. REPRODUCIBILITY SUBSET")
    print("=" * 100)
    print(f"  subset n            : {len(sel)}")
    print(f"  sources             : {sel.source.value_counts().to_dict()}")
    print(f"  geometries          : {sel.geometry.value_counts().to_dict()}")
    print(f"  splits              : {sel.split.value_counts().to_dict()}")
    print(f"  strata covered      : {cover} / {len(g)}")
    print(f"  covers all sources  : {set(sel.source) == set(df.source)}")
    print(f"  covers all splits   : {set(sel.split) == set(df.split)}")
    print(f"  covers all geometry : {set(sel.geometry) == set(df.geometry)}")

    json.dump({"subset_n": int(len(sel)),
               "sources": sorted(sel.source.unique().tolist()),
               "geometries": sorted(sel.geometry.unique().tolist()),
               "splits": sorted(sel.split.unique().tolist()),
               "strata_covered": int(cover), "strata_total": int(len(g)),
               "covers_all_sources": bool(set(sel.source) == set(df.source)),
               "covers_all_splits": bool(set(sel.split) == set(df.split)),
               "covers_all_geometries": bool(set(sel.geometry) == set(df.geometry))},
              open(f"{PRIV}/seg_current_v1_repro_subset.json", "w"), indent=2)


if __name__ == "__main__":
    main()
