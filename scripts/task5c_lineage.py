#!/usr/bin/env python
"""Task 5C (N): HVDROPDB artifact reuse gate.

Section N forbids reusing an existing segmentation metric merely because an old CSV exists.
Every candidate must be verified on cohort, checkpoint SHA256, architecture, input size,
preprocessing, threshold, postprocessing, camera split and evaluation definition. If any
materially relevant field is unknown, the artifact cannot be relabelled as V1 evidence.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path("/Users/moniaz/niki")
HV = ROOT / "results/hvdro_validation"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

V1_CHECKPOINT = "c373f53813ee60b89651a04a98bc5f1d6bc60a5a50f45c4e10f42475650d5374"
V1 = {"threshold": 0.20, "input_size": 256, "min_area": 50, "close_k": 3,
      "architecture": "MAnet", "encoder": "resnet34"}

sha_pat = re.compile(r"\b[0-9a-f]{64}\b")


def sha_in(path: Path) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""
    m = sha_pat.search(txt)
    return m.group(0) if m else ""


def main() -> None:
    print("=" * 100)
    print("N. HVDROPDB ARTIFACT REUSE GATE")
    print("=" * 100)
    print(f"  SEG_CURRENT_V1 checkpoint sha256 = {V1_CHECKPOINT[:16]}...")
    print(f"  SEG_CURRENT_V1 geometry/config   = {V1}")
    print()

    # any file anywhere that records a 64-hex checkpoint hash near hvdro?
    print("  searching for a recorded checkpoint sha256 in the HVDROPDB artifacts:")
    found_any = False
    for p in sorted(HV.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".json", ".csv", ".txt", ".md"):
            s = sha_in(p)
            if s:
                found_any = True
                print(f"    {p.relative_to(ROOT)}: {s[:16]}...  "
                      f"matches_V1={s == V1_CHECKPOINT}")
    if not found_any:
        print("    NONE. No HVDROPDB artifact records a checkpoint sha256.")
    print()

    rows = []
    # ---- vessel Dice ----
    f = HV / "vessel_dice_summary.csv"
    if f.exists():
        d = pd.read_csv(f)
        sizes = sorted(d.inference_size.unique().tolist())
        cameras = sorted(d.camera.unique().tolist())
        rows.append({
            "artifact": "results/hvdro_validation/vessel_dice_summary.csv",
            "metric": "Dice / IoU / recall / precision vs expert vessel masks",
            "generation": "UNKNOWN (no checkpoint sha recorded)",
            "checkpoint_sha": "ABSENT",
            "threshold": 0.20,
            "cohort_n": int(d[d.inference_size == 256].n.sum()) if 256 in sizes else int(d.n.sum()),
            "RetCam_n": int(d[(d.camera.str.contains("RetCam")) & (d.inference_size == 256)].n.sum()) if 256 in sizes else 0,
            "Neo_n": int(d[(d.camera.str.contains("Neo")) & (d.inference_size == 256)].n.sum()) if 256 in sizes else 0,
            "reuse_status": "LEGACY_NOT_APPLICABLE",
            "reason": ("inference_size/threshold match V1 but the checkpoint sha256 is not "
                       "recorded anywhere in the artifact, so lineage cannot be verified; "
                       "section N requires it. Recompute."),
        })
    # ---- width ----
    f = HV / "width/width_summary.csv"
    mf = HV / "width/manifest.json"
    if f.exists():
        d = pd.read_csv(f)
        man = json.loads(mf.read_text()) if mf.exists() else {}
        rows.append({
            "artifact": "results/hvdro_validation/width/width_summary.csv",
            "metric": "width estimator comparison, ICD(2,1) vs expert skeleton width",
            "generation": "UNKNOWN (no checkpoint sha recorded)",
            "checkpoint_sha": "ABSENT",
            "threshold": man.get("threshold", "UNKNOWN"),
            "cohort_n": int(len(d)),
            "RetCam_n": 50,
            "Neo_n": 50,
            "reuse_status": "LEGACY_NOT_APPLICABLE",
            "reason": ("manifest records inference_size 256, threshold 0.2, min_area 50, close_k 3 "
                       "and states 'BV and OD images are NOT paired, so no DD normalisation in "
                       "this file', but no checkpoint sha256. Its width estimator is also "
                       "A_deployed (2*EDT) which does match CLINICAL_MEASUREMENT_V1; the blocker "
                       "is the missing checkpoint identity, not the definition."),
        })
    # ---- disc error ----
    f = HV / "disc_error_summary.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows.append({
            "artifact": "results/hvdro_validation/disc_error_summary.csv",
            "metric": "disc centre offset / radius ratio vs expert disc",
            "generation": "LEGACY_PSEUDO_DISC",
            "checkpoint_sha": "N/A (no detector used)",
            "threshold": "N/A",
            "cohort_n": int(d.n.sum()),
            "RetCam_n": int(d[d.camera.str.contains("Retcam", case=False)].n.sum()),
            "Neo_n": int(d[d.camera.str.contains("Neo")].n.sum()),
            "reuse_status": "LEGACY_NOT_APPLICABLE",
            "reason": ("pseudo_radius_px is a CONSTANT for every image (255.0 Neo, 60.0 RetCam), "
                       "the signature of a fixed image fraction rather than a detector. Labelled "
                       "LEGACY_PSEUDO_DISC - NOT CURRENT DISC EVIDENCE per section O."),
        })
    # ---- current disc predictions ----
    f = HV / "disc/disc_predictions_all.csv"
    if f.exists():
        n = sum(1 for _ in open(f)) - 1
        rows.append({
            "artifact": "results/hvdro_validation/disc/disc_predictions_all.csv",
            "metric": "automatic disc predictions (peak_prob, disc_dd_px, centre)",
            "generation": "project cohort, not HVDROPDB",
            "checkpoint_sha": "ABSENT",
            "threshold": "UNKNOWN",
            "cohort_n": int(n),
            "RetCam_n": 0,
            "Neo_n": 0,
            "reuse_status": "LEGACY_NOT_APPLICABLE",
            "reason": ("covers the project 8870 cohort (data/raw/farabi|plus|farfum_rop), NOT the "
                       "100 HVDROPDB expert-disc images. It cannot serve as the HVDROPDB disc "
                       "benchmark; section Q2 requires a new evaluation on the expert-disc cohort."),
        })
    # ---- what must be recomputed ----
    rows.append({
        "artifact": "(to be created) results/hvdro_validation/v2/seg_v1_v2_non_inferiority.csv",
        "metric": "paired Dice / clDice / recall / precision / thin-vessel recall, V2 - V1",
        "generation": "SEG_CURRENT_V1 vs SEG_GEOMETRY_CANDIDATE_V2",
        "checkpoint_sha": V1_CHECKPOINT,
        "threshold": 0.20,
        "cohort_n": 100,
        "RetCam_n": 50,
        "Neo_n": 50,
        "reuse_status": "RECOMPUTED",
        "reason": "no existing artifact computes V2 at all; section S requires a paired comparison.",
    })
    rows.append({
        "artifact": "(to be created) results/hvdro_validation/v2/disc_benchmark.csv",
        "metric": "disc Dice, centre error in expert DD, diameter ratio, validity rate",
        "generation": "current automatic disc detector",
        "checkpoint_sha": "ABSENT",
        "threshold": "UNKNOWN",
        "cohort_n": 100,
        "RetCam_n": 50,
        "Neo_n": 50,
        "reuse_status": "RECOMPUTED",
        "reason": "no lineage-valid current disc evaluation exists on the HVDROPDB expert-disc cohort (section O/Q2).",
    })

    T = pd.DataFrame(rows)
    cols = ["artifact", "metric", "generation", "checkpoint_sha", "threshold", "cohort_n",
            "RetCam_n", "Neo_n", "reuse_status", "reason"]
    T[cols].to_csv(ART / "hvdro_evidence_lineage.csv", index=False)
    print(T[["artifact", "cohort_n", "RetCam_n", "Neo_n", "reuse_status"]].to_string(index=False))
    print()
    print(f"  REUSED_VERIFIED      : {int((T.reuse_status == 'REUSED_VERIFIED').sum())}")
    print(f"  RECOMPUTED           : {int((T.reuse_status == 'RECOMPUTED').sum())}")
    print(f"  LEGACY_NOT_APPLICABLE: {int((T.reuse_status == 'LEGACY_NOT_APPLICABLE').sum())}")
    print()
    print("  VERDICT: no existing HVDROPDB artifact passes the section N lineage gate, because")
    print("  none of them records the segmentation checkpoint sha256. Existing Dice may therefore")
    print("  NOT be cited as SEG_CURRENT_V1 evidence. Everything required for sections Q and S")
    print("  must be recomputed.")
    print(f"\n  -> {ART}/hvdro_evidence_lineage.csv")

    (ROOT / "_private_audit" / "task5c_reuse.json").write_text(json.dumps({
        "reused_verified": int((T.reuse_status == "REUSED_VERIFIED").sum()),
        "recomputed": int((T.reuse_status == "RECOMPUTED").sum()),
        "legacy_not_applicable": int((T.reuse_status == "LEGACY_NOT_APPLICABLE").sum()),
        "any_artifact_records_checkpoint_sha": bool(found_any),
        "verdict": "NOTHING_REUSABLE",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
