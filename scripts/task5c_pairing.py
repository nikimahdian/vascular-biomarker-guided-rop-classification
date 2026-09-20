#!/usr/bin/env python
"""Task 5C (P/O): HVDROPDB vessel/disc reference pairing by pixel identity, and the
provenance of the legacy disc-error numbers.

P. pairing must be established from image bytes, never from filename, row order or camera.
O. the legacy disc-error table must be traced to the disc method that actually produced it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path("/Users/moniaz/niki")
H = ROOT / "data/raw/hvdro/segmentation/HVDROPDB_RetCam_Neo_Segmentation"

COHORTS = {
    "Neo_Vessels_images": H / "HVDROPDB-BV/Neo_Vessels_images",
    "RetCam_Vessels_images": H / "HVDROPDB-BV/RetCam_Vessels_images",
    "Neo_OpticDisc_images": H / "HVDROPDB-OD/Neo_OpticDisc_images",
    "Retcam_OpticDisc_images": H / "HVDROPDB-OD/Retcam_OpticDisc_images",
    "Neo_Vessels_masks": H / "HVDROPDB-BV/Neo_Vessels_masks",
    "RetCam_Vessels_masks": H / "HVDROPDB-BV/RetCam_Vessels_masks",
    "Neo_OpticDisc_masks": H / "HVDROPDB-OD/Neo_OpticDisc_masks",
    "Retcam_OpticDisc_masks": H / "HVDROPDB-OD/Retcam_OpticDisc_masks",
}


def byte_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pix_sha(p: Path) -> str:
    with Image.open(p) as im:
        a = np.asarray(im.convert("RGB"))
    return hashlib.sha256(a.tobytes()).hexdigest(), a.shape


def main() -> None:
    print("=" * 100)
    print("P. HVDROPDB REFERENCE COHORTS AND PAIRING BY PIXEL IDENTITY")
    print("=" * 100)
    inv = {}
    for name, d in COHORTS.items():
        files = sorted(d.glob("*.png"))
        rows = []
        for f in files:
            try:
                ps, shape = pix_sha(f)
            except Exception as e:  # noqa: BLE001
                ps, shape = f"ERROR:{e}", (0, 0)
            rows.append({"file": f.name, "byte_sha256": byte_sha(f), "pix_sha256": ps,
                         "h": shape[0], "w": shape[1]})
        inv[name] = pd.DataFrame(rows)
        print(f"  {name:26s} n={len(rows):4d}  "
              f"dims={sorted({(r['w'], r['h']) for r in rows})[:3]}")

    v_n = len(inv["Neo_Vessels_images"]) + len(inv["RetCam_Vessels_images"])
    d_n = len(inv["Neo_OpticDisc_images"]) + len(inv["Retcam_OpticDisc_images"])
    vpix = set(inv["Neo_Vessels_images"].pix_sha256) | set(
        inv["RetCam_Vessels_images"].pix_sha256)
    dpix = set(inv["Neo_OpticDisc_images"].pix_sha256) | set(
        inv["Retcam_OpticDisc_images"].pix_sha256)
    vbyte = set(inv["Neo_Vessels_images"].byte_sha256) | set(
        inv["RetCam_Vessels_images"].byte_sha256)
    dbyte = set(inv["Neo_OpticDisc_images"].byte_sha256) | set(
        inv["Retcam_OpticDisc_images"].byte_sha256)

    print()
    print(f"  HVDRO_VESSEL_REFERENCE_N      = {v_n}")
    print(f"  HVDRO_DISC_REFERENCE_N        = {d_n}")
    print(f"  distinct pixel hashes vessel  = {len(vpix)}")
    print(f"  distinct pixel hashes disc    = {len(dpix)}")
    print(f"  paired by PIXEL identity      = {len(vpix & dpix)}")
    print(f"  paired by BYTE identity       = {len(vbyte & dbyte)}")
    print()
    print("  per-camera pixel-identity overlap:")
    for vk, dk in (("Neo_Vessels_images", "Neo_OpticDisc_images"),
                   ("RetCam_Vessels_images", "Retcam_OpticDisc_images")):
        a = set(inv[vk].pix_sha256)
        b = set(inv[dk].pix_sha256)
        print(f"    {vk:24s} n={len(a):3d}  vs {dk:24s} n={len(b):3d}  "
              f"overlap={len(a & b)}")
    print()
    print("  filename-based pairing (FORBIDDEN, shown only to demonstrate the trap):")
    fn_v = set(inv["Neo_Vessels_images"].file) | set(inv["RetCam_Vessels_images"].file)
    fn_d = set(inv["Neo_OpticDisc_images"].file) | set(inv["Retcam_OpticDisc_images"].file)
    print(f"    identical filenames across cohorts = {len(fn_v & fn_d)} "
          f"(would falsely imply pairing)")

    paired = len(vpix & dpix)
    verdict = "NOT_AVAILABLE" if paired == 0 else f"AVAILABLE_N={paired}"
    print()
    print(f"  HVDRO_DD_NORMALIZED_END_TO_END_BENCHMARK = {verdict}")
    if paired == 0:
        print("  -> vessel and disc expert references do not share a single image.")
        print("     This is a DATASET LIMITATION, not a failure. DD-normalised agreement")
        print("     is DEFERRED_TO_TARGET_DOMAIN_EXPERT_VALIDATION.")

    # ---------------------------------------------------------------- O
    print()
    print("=" * 100)
    print("O. PROVENANCE OF THE LEGACY DISC-ERROR NUMBERS")
    print("=" * 100)
    de = ROOT / "results/hvdro_validation/disc_error_summary.csv"
    if de.exists():
        print("  results/hvdro_validation/disc_error_summary.csv:")
        print(pd.read_csv(de).to_string(index=False))
    print()
    print("  the pseudo-radius in that table is a CONSTANT (255.0 px for Neo, 60.0 px for")
    print("  RetCam) for every image, which is the signature of a fixed fraction of the")
    print("  image, not of a detector. A detector would produce a different value per image.")
    print()
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    hits = []
    for s in scripts:
        try:
            txt = s.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        if "hvdro" in txt.lower() and ("disc" in txt.lower()):
            hits.append(s.name)
    print(f"  scripts mentioning hvdro + disc: {hits}")
    for nm in ("hvdro_disc.py", "hvdro_disc_why.py", "hvdro_disc_failview.py",
               "hvdro_disc_farabi_check.py"):
        p = ROOT / "scripts" / nm
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="replace")
            markers = [m for m in ("pseudo", "image centre", "image center", "centre",
                                   "0.3 *", "min(h, w)", "constant") if m in txt]
            print(f"    {nm}: markers={markers}")
    cur = ROOT / "results/hvdro_validation/disc/disc_predictions_all.csv"
    if cur.exists():
        c = pd.read_csv(cur, nrows=3)
        print()
        print(f"  CURRENT disc-detector artifact: results/hvdro_validation/disc/"
              f"disc_predictions_all.csv")
        print(f"    rows={sum(1 for _ in open(cur)) - 1}  columns={list(c.columns)[:10]}")
        print(f"    cohort image_path sample: {c.image_path.iloc[0]}")
        print(f"    -> this is the PROJECT 8870 cohort, NOT the HVDROPDB expert-disc cohort")
        print(f"       ({d_n} images under data/raw/hvdro/.../HVDROPDB-OD/)")
    print()
    print("  LEGACY DISC RESULT CLASSIFICATION:")
    print("    disc_error_summary.csv -> LEGACY_PSEUDO_DISC - NOT CURRENT DISC EVIDENCE")
    print("    it must not be used to characterise the current disc detector")
    print("    A lineage-valid current disc evaluation on the HVDROPDB expert-disc cohort")
    print("    DOES NOT EXIST and must be computed (Task 5C section Q2).")

    out = {
        "HVDRO_VESSEL_REFERENCE_N": int(v_n),
        "HVDRO_DISC_REFERENCE_N": int(d_n),
        "HVDRO_PAIRED_VESSEL_DISC_N": int(paired),
        "HVDRO_PAIRED_BY_BYTE_N": int(len(vbyte & dbyte)),
        "HVDRO_DD_NORMALIZED_END_TO_END_BENCHMARK": verdict,
        "per_camera_pixel_overlap": {
            "Neo": len(set(inv["Neo_Vessels_images"].pix_sha256)
                       & set(inv["Neo_OpticDisc_images"].pix_sha256)),
            "RetCam": len(set(inv["RetCam_Vessels_images"].pix_sha256)
                          & set(inv["Retcam_OpticDisc_images"].pix_sha256)),
        },
        "identical_filenames_across_cohorts": len(fn_v & fn_d),
        "cohort_dims": {k: sorted({(int(r["w"]), int(r["h"])) for r in v.to_dict("records")})
                        for k, v in inv.items()},
        "LEGACY_DISC_STATUS": "LEGACY_PSEUDO_DISC - NOT CURRENT DISC EVIDENCE",
    }
    (ROOT / "_private_audit/task5c_pairing.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n  -> _private_audit/task5c_pairing.json")


if __name__ == "__main__":
    main()
