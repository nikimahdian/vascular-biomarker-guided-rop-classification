"""Download the prior-work pretrained vessel-segmentation checkpoints into weights/.

The Google Drive ids come from the reference notebooks. Drive assigns the original
filenames (e.g. best_weight_DeepLabV3+_resize_27). After download, inspect weights/
and set segmentation.weight_path in config.yaml to the file you want to use.

Usage:
    python -m src.segmentation.download_weights
"""
from __future__ import annotations

import gdown

from src.utils.common import ensure_dirs, load_config


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    weights_dir = cfg["paths"]["weights_dir"]
    ids = cfg["segmentation"].get("pretrained_weight_ids", [])
    if not ids:
        raise SystemExit("No pretrained_weight_ids in config.")
    for fid in ids:
        print(f"[download] {fid}")
        try:
            gdown.download(id=fid, output=str(weights_dir) + "/", quiet=False)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed {fid}: {e}")
    print("[done] inspect", weights_dir, "and set segmentation.weight_path accordingly")


if __name__ == "__main__":
    main()
