"""Verify a segmentation generation contract against the current on-disk state.

Usage:
    python -m src.segmentation.verify_generation --generation SEG_CURRENT_V1

Exits 0 when every pinned byte hash matches, 2 otherwise. Nothing is written.
"""
from __future__ import annotations

import argparse
import sys

from src.segmentation.generation import (
    GenerationError,
    contract_path,
    load_contract,
    output_store,
    verify_contract,
)
from src.utils.common import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", required=True)
    args = ap.parse_args()

    cfg = load_config()
    path = contract_path(cfg, args.generation)
    print(f"contract : {path}")
    try:
        contract = load_contract(cfg, args.generation)
    except GenerationError as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(2) from exc

    print(f"id       : {contract['generation_id']}")
    print(f"created  : {contract['created_at']}")
    print(f"arch     : {contract['architecture']}/{contract['encoder']}")
    print(f"threshold: {contract['threshold']}")
    print(f"store    : {contract['output_store']} (frozen={contract['store_frozen']})")
    print(f"purpose  : {contract.get('purpose', 'scientific generation')}")
    try:
        observed = verify_contract(cfg, contract)
    except GenerationError as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(2) from exc

    for key in ("checkpoint_sha256", "checkpoint_size_bytes",
                "inference_script_sha256", "config_sha256"):
        print(f"  {key:26s} {observed[key]}")
    store = output_store(cfg, contract)
    print(f"  store resolved             {store} exists={store.exists()}")
    print("[OK] generation contract verified against the current state")
    raise SystemExit(0)


if __name__ == "__main__":
    sys.exit(main())
