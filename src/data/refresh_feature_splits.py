"""Re-attach split/label/source from data/splits/all.csv onto feature tables.

Run after `prepare_split` when biomarker extraction is already done — avoids
re-running PVBM on thousands of images.

Usage:
    python -m src.data.refresh_feature_splits
"""
from __future__ import annotations

import pandas as pd

from src.utils.common import ensure_dirs, load_config


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    all_path = cfg["paths"]["splits_dir"] / "all.csv"
    feats_path = cfg["paths"]["features_dir"] / "biomarker_features.csv"
    if not all_path.exists():
        raise SystemExit(f"Missing {all_path}. Run src.data.prepare_split first.")
    if not feats_path.exists():
        raise SystemExit(f"Missing {feats_path}. Run src.biomarker.extract_pvbm first.")

    splits = pd.read_csv(all_path)
    keep = ["image_path", "label", "split", "source"]
    keep.extend(
        column
        for column in ["group_id", "patient_id", "exam_id", "identity_level"]
        if column in splits.columns
    )
    meta = splits[keep].drop_duplicates(subset="image_path")
    feats = pd.read_csv(feats_path)
    feat_cols = [c for c in feats.columns if c not in set(keep)]
    merged = feats[["image_path"] + feat_cols].merge(meta, on="image_path", how="inner")
    missing = len(feats) - len(merged)
    if missing:
        print(f"[warn] {missing} feature rows had no split match and were dropped")
    merged.to_csv(feats_path, index=False)
    print(f"[done] refreshed splits on {len(merged)} rows -> {feats_path}")


if __name__ == "__main__":
    main()
