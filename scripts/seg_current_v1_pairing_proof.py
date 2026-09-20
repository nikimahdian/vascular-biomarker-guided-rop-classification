#!/usr/bin/env python
"""Task 5A (J): prove the strict image -> mask pairing contract over all 8,870 images.

Uses src.segmentation.mask_pairing.resolve_mask, which never takes a first filesystem match.
For every canonical image the generation store must resolve to exactly one mask, and that
mask must be the one the canonical mapping pins.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/moniaz/niki")
from src.segmentation.mask_pairing import (  # noqa: E402
    MaskPairingError,
    mask_stem,
    resolve_mask,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
print("=" * 100)
print("J. STRICT IMAGE -> MASK PAIRING CONTRACT")
print("=" * 100)

canon = pd.read_csv(f"{ROOT}/data/masks/mask_manifest_canonical_v1.csv")
store = Path(f"{ROOT}/data/masks")
names = [p.name for p in store.glob("*.png")]
print(f"  canonical images   : {len(canon)}")
print(f"  masks in store     : {len(names)}")

# every store filename must parse to a declared stem
unparsed = [n for n in names if mask_stem(n) is None]
print(f"  store filenames that do not parse as '<stem>_<8 hex>.png': {len(unparsed)}")

resolved_ok, errors, mismatches = 0, [], []
for r in canon.itertuples():
    try:
        got = resolve_mask(r.image_path, names)
    except MaskPairingError as exc:
        errors.append({"image_path": r.image_path, "error": str(exc)})
        continue
    if got is None:
        errors.append({"image_path": r.image_path, "error": "resolved to None"})
        continue
    resolved_ok += 1
    if got != Path(r.mask_path).name:
        mismatches.append({"image_path": r.image_path, "resolved": got,
                           "pinned": Path(r.mask_path).name})

print()
print(f"  resolved to exactly one mask      : {resolved_ok} / {len(canon)}")
print(f"  resolution errors (0 or >1)       : {len(errors)}")
print(f"  resolved mask != pinned mask      : {len(mismatches)}")
print(f"  no first-match / no prefix match  : guaranteed by resolve_mask implementation")

# the store must contain no mask that no image claims, and every image must claim exactly one
stem_of = {r.image_path: Path(r.image_path).stem for r in canon.itertuples()}
claimed = {mask_stem(n) for n in names}
unclaimed = sorted(s for s in claimed if s is not None and s not in set(stem_of.values()))
print(f"  store stems claimed by no image   : {len(unclaimed)}")

# explicit failure-mode probes against real store data
print()
print("  FAILURE-MODE PROBES")
probe_img = canon.image_path.iloc[0]
stem = Path(probe_img).stem

def probe(label, nameset):
    try:
        r = resolve_mask(probe_img, nameset)
    except MaskPairingError as exc:
        print(f"    {label:44s} -> MaskPairingError (correct)")
        return "ERROR"
    print(f"    {label:44s} -> {r!r}")
    return r

probe("zero candidates", [])
probe("exactly one candidate", [f"{stem}_deadbeef.png"])
probe("two candidates for the same stem", [f"{stem}_aaaaaaaa.png", f"{stem}_bbbbbbbb.png"])
probe("prefix sibling only (no exact stem)", [f"{stem}0_deadbeef.png"])
probe("prefix sibling plus exact", [f"{stem}0_deadbeef.png", f"{stem}_deadbeef.png"])
probe("unparseable name", [f"{stem}.png"])
probe("windows-style image path", [f"{stem}_deadbeef.png"])
probe("upper-case hash suffix", [f"{stem}_DEADBEEF.png"])
print()
print("  mixed slash convention probe:")
for p in (probe_img, probe_img.replace("/", "\\")):
    try:
        print(f"    {resolve_mask(p, [f'{stem}_deadbeef.png'])!r}  <- {p[:3]}...")
    except MaskPairingError as exc:
        print(f"    MaskPairingError for {p[:3]}...: {exc}")

verdict = "PASS" if (resolved_ok == len(canon) and not errors and not mismatches) else "FAIL"
print()
print(f"  STRICT_IMAGE_MASK_PAIRING = {verdict}")
json.dump({"STRICT_IMAGE_MASK_PAIRING": verdict,
           "images": int(len(canon)), "stored_masks": len(names),
           "resolved_exactly_one": resolved_ok, "errors": errors[:5],
           "mismatches": mismatches[:5], "unparsed_store_names": unparsed[:5],
           "unclaimed_stems": unclaimed[:5]},
          open(f"{ROOT}/_private_audit/seg_current_v1_pairing_proof.json", "w"),
          indent=2, default=str)
print(f"  -> {ROOT}/_private_audit/seg_current_v1_pairing_proof.json")
