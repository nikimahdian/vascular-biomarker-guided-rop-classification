"""Download and unzip the three datasets from Google Drive into paths.raw_dir.

Usage:
    python -m src.data.download_data
    python -m src.data.download_data --only farabi plus   # subset
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import gdown

from src.utils.common import ensure_dirs, load_config


def download_and_extract(file_id: str, name: str, raw_dir: Path) -> Path:
    zip_path = raw_dir / f"{name}.zip"
    if not zip_path.exists():
        print(f"[download] {name}  (id={file_id})")
        gdown.download(id=file_id, output=str(zip_path), quiet=False)
    else:
        print(f"[skip download] {zip_path} already exists")

    out_dir = raw_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(zip_path):
        print(f"[unzip] {zip_path.name} -> {out_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    else:
        print(f"[warn] {zip_path} is not a zip (maybe a single file). Left as-is.")
    return out_dir


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    raw_dir: Path = cfg["paths"]["raw_dir"]
    ids: dict[str, str] = cfg["data"]["drive_ids"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(ids), default=list(ids))
    args = ap.parse_args()

    for name in args.only:
        download_and_extract(ids[name], name, raw_dir)
    print("[done] datasets in", raw_dir)


if __name__ == "__main__":
    main()
