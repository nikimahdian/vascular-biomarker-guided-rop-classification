#!/usr/bin/env python
"""Task 5B (V): write configs/feature_contract.csv with the required columns.

Existing historical rows (feature_version pvbm_v1 / clinical_v3) are preserved and gain the new
columns with explicit UNKNOWN/FALSE values rather than invented ones. CLINICAL_MEASUREMENT_V1
rows are generated from configs/clinical_measurement_v1.yaml, so the contract and the frozen
definitions cannot drift apart.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path("/Users/moniaz/niki")
CONTRACT = ROOT / "configs" / "feature_contract.csv"
MEAS = ROOT / "configs" / "clinical_measurement_v1.yaml"

COLUMNS = [
    "feature_name", "measurement_version", "feature_version", "mathematical_definition",
    "unit", "roi_definition", "depends_on_disc", "depends_on_fov", "depends_on_resolution",
    "depends_on_av", "scale_invariant", "padding_invariant", "rotation_invariant",
    "missingness_policy", "admission_status", "allowed_in_final_classifier",
    "requires_expert_validation", "known_limitations", "generation_script",
]

DEFINITIONS = {
    "tort_geodesic_median": "median over branches of (smoothed path length / endpoint chord)",
    "tort_geodesic_p90": "90th percentile of the same per-branch tortuosity",
    "tort_geodesic_top3_mean": "mean of the three largest per-branch tortuosity values",
    "tort_pixelcount_median_provenance": "median over branches of (branch pixel count / hull chord); historical definition retained for provenance",
    "vessel_density_wholeframe": "vessel pixels / all pixels in the working frame",
    "vessel_density_fov": "vessel pixels inside the FOV / FOV pixels",
    "vessel_density_fov_ring_0_2dd": "vessel pixels inside ring[0,2)DD AND FOV / pixels of ring AND FOV",
    "vessel_density_fov_ring_2_3dd": "vessel pixels inside ring[2,3)DD AND FOV / pixels of ring AND FOV",
    "vessel_density_fov_ring_3_6dd": "vessel pixels inside ring[3,6)DD AND FOV / pixels of ring AND FOV",
    "skel_density_fov": "skeleton pixels inside the FOV / FOV pixels",
    "vessel_pixels_work": "count of vessel mask pixels at the working resolution",
    "vessel_area_fraction_work": "vessel mask pixels / all pixels in the working frame",
    "skeleton_px_work": "count of skeleton pixels at the working resolution",
    "branch_px_work": "total pixels belonging to scored skeleton branches",
    "n_branches": "number of 8-connected branch components after junction removal",
    "n_startpoints": "skeleton pixels with exactly two 8-neighbours (line ends)",
    "n_endpoints": "same quantity as n_startpoints in this implementation",
    "n_intersections": "skeleton pixels with four or more 8-neighbours",
    "n_startpoints_fov": "n_startpoints computed on the skeleton restricted to the FOV",
    "n_endpoints_fov": "n_endpoints computed on the skeleton restricted to the FOV",
    "n_intersections_fov": "n_intersections computed on the skeleton restricted to the FOV",
    "width_p50_px": "median over skeleton pixels of 2*EDT, in working pixels",
    "width_p90_px": "90th percentile of 2*EDT",
    "width_mean_px": "mean of 2*EDT",
    "width_p50_dd": "median over skeleton pixels of 2*EDT / disc diameter",
    "width_p90_dd": "90th percentile of 2*EDT / disc diameter",
    "width_p95_dd": "95th percentile of 2*EDT / disc diameter",
    "width_mean_dd": "mean of 2*EDT / disc diameter",
    "width_ann_p50_dd": "median 2*EDT / DD for skeleton pixels inside the [0.5,2)DD annulus",
    "width_ann_p90_dd": "90th percentile of the same",
    "width_ann_mean_dd": "mean of the same",    "width_shape_p90_over_p50": "width_p90_px / width_p50_px",
    "width_quantisation_floor_dd": "ceil(native_long_side/256) / dd_px",
    "fractal_d0": "capacity dimension from PVBM MultifractalVBMs on the FOV-masked mask",
    "fractal_d1": "information dimension, same implementation",
    "fractal_d2": "correlation dimension, same implementation",
    "singularity_length": "PVBM multifractal singularity length",
    "frame_sector_ne_density": "vessel fraction in the image-frame NE sector of the posterior pole (r<6DD)",
    "frame_sector_nw_density": "vessel fraction in the image-frame NW sector",
    "frame_sector_sw_density": "vessel fraction in the image-frame SW sector",
    "frame_sector_se_density": "vessel fraction in the image-frame SE sector",
    "sector_density_range": "max minus min of the four frame-sector densities",
    "n_sectors_above_median": "count of frame sectors whose density exceeds their median",
    "a_frac": "length fraction of branches assigned to the artery group by a median split of background-corrected green",
    "a_width_p90_px": "90th percentile of branch median 2*EDT for the artery group",
    "v_width_p90_px": "90th percentile of branch median 2*EDT for the vein group",
    "av_width_ratio_p90": "a_width_p90_px / v_width_p90_px",
    "disc_valid": "1 when the disc detector passed confidence and diameter-fraction gates",
    "disc_peak_prob": "peak probability of the disc detector",
    "disc_method": "label naming the disc source",
    "dd_px_work": "disc diameter in working pixels",
    "dd_over_min_side": "disc diameter / min(image height, width)",
    "disc_cx_frac": "disc centre x / width",
    "disc_cy_frac": "disc centre y / height",
    "disc_centre_offset_frac": "distance from disc centre to image centre / width",
    "fov_valid": "1 when the FOV detection passed the coverage and fragmentation gates",
    "fov_coverage_fraction": "FOV pixels / all pixels",
    "fov_n_components": "number of connected bright components before largest-component selection",
    "fov_border_contact": "fraction of the image border lying inside the FOV",
    "fov_centroid_offset": "distance from FOV centroid to image centre / width",
    "fov_failure_reason": "label naming the FOV gate that failed, empty when none",
    "roi_coverage_ring_0_2dd": "fraction of ring[0,2)DD lying inside the FOV",
    "roi_coverage_ring_2_3dd": "fraction of ring[2,3)DD lying inside the FOV",
    "roi_coverage_ring_3_6dd": "fraction of ring[3,6)DD lying inside the FOV",
    "roi_coverage_annulus": "fraction of the [0.5,2)DD annulus lying inside the FOV",
    "roi_coverage_pole": "fraction of r<6DD lying inside the FOV",
    "n_skel_px_disc": "skeleton pixels inside the juxta-papillary annulus",
    "measurement_version": "constant string identifying the measurement layer",
    "work_resolution": "long side of the working grid in pixels",
}

ROI = {
    "fov": "retinal field of view from the photograph",
    "rings": "concentric annulus about the disc centre",
    "annulus": "[0.5,2) disc diameters",
    "pole": "r < 6 disc diameters",
    "frame_sectors": "image-frame angular sectors about the disc centre",
    "none": "whole working frame",
}


def main() -> None:
    meas = yaml.safe_load(MEAS.read_text(encoding="utf-8"))
    feats = meas["features"]

    new_rows = []
    for name, spec in feats.items():
        adm = spec.get("admission", "EXPLORATORY_ONLY")
        allowed = adm in ("PRIMARY_ALLOWED", "SECONDARY_ALLOWED")
        roi = "none"
        if name.startswith("vessel_density_fov_ring") or name.startswith("roi_coverage_ring"):
            roi = "rings"
        elif "ann" in name or name.startswith("roi_coverage_annulus"):
            roi = "annulus"
        elif name.startswith("frame_sector") or name.startswith("sector_") or \
                name.startswith("n_sectors") or name == "roi_coverage_pole":
            roi = "frame_sectors"
        elif name.startswith("vessel_density_fov") or name.startswith("skel_density_fov"):
            roi = "fov"
        limits = spec.get("reason") or spec.get("known_limitation") or ""
        if spec.get("known_floor"):
            limits = (limits + f" measured width/DD scale spread {spec['known_floor']}").strip()
        new_rows.append({
            "feature_name": name,
            "measurement_version": "CLINICAL_MEASUREMENT_V1",
            "feature_version": "clinical_measurement_v1",
            "mathematical_definition": DEFINITIONS.get(name, "see configs/clinical_measurement_v1.yaml"),
            "unit": spec.get("unit", "UNKNOWN"),
            "roi_definition": ROI[roi],
            "depends_on_disc": str(bool(spec.get("depends_on_disc", False))).upper(),
            "depends_on_fov": str(bool(spec.get("depends_on_fov", False))).upper(),
            "depends_on_resolution": str(bool(spec.get("depends_on_resolution", False))).upper(),
            "depends_on_av": str(bool(spec.get("depends_on_av", False))).upper(),
            "scale_invariant": str(bool(spec.get("scale_invariant", False))).upper(),
            "padding_invariant": str(bool(spec.get("padding_invariant", False))).upper(),
            "rotation_invariant": str(bool(spec.get("rotation_invariant", False))).upper(),
            "missingness_policy": "NaN when undefined; never imputed at measurement time",
            "admission_status": adm,
            "allowed_in_final_classifier": str(allowed).upper(),
            "requires_expert_validation": str(bool(spec.get("depends_on_av", False))).upper(),
            "known_limitations": limits or "none recorded",
            "generation_script": "src/biomarker/clinical_measurement_v1.py",
        })

    # preserve historical rows, adding the new columns as UNKNOWN rather than inventing values.
    # Rows already carrying the V1 measurement version are skipped: they are regenerated above.
    old_rows = []
    if CONTRACT.exists():
        with CONTRACT.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("feature_version") == "clinical_measurement_v1" or \
                        r.get("measurement_version") == "CLINICAL_MEASUREMENT_V1":
                    continue
                old_rows.append({
                    "feature_name": r.get("feature_name", ""),
                    "measurement_version": "HISTORICAL_UNVERSIONED",
                    "feature_version": r.get("feature_version", r.get("feature_version", "")),
                    "mathematical_definition": r.get("definition", r.get("mathematical_definition", "")),
                    "unit": r.get("unit", "UNKNOWN"),
                    "roi_definition": "UNKNOWN (historical)",
                    "depends_on_disc": "UNKNOWN",
                    "depends_on_fov": "UNKNOWN",
                    "depends_on_resolution": r.get("depends_on_resolution", "UNKNOWN").upper(),
                    "depends_on_av": r.get("depends_on_av_heuristic", "UNKNOWN").upper(),
                    "scale_invariant": "UNKNOWN",
                    "padding_invariant": "UNKNOWN",
                    "rotation_invariant": "UNKNOWN",
                    "missingness_policy": r.get("missingness_policy", "UNKNOWN"),
                    "admission_status": "HISTORICAL_NOT_ADMITTED",
                    "allowed_in_final_classifier": "FALSE",
                    "requires_expert_validation": "TRUE",
                    "known_limitations": r.get("known_limitations", r.get("clinical_status", "")),
                    "generation_script": r.get("generation_script", "UNKNOWN"),
                })

    # drop the old header-only duplicate of the same feature/version pairs
    seen = set()
    merged = []
    for r in new_rows + old_rows:
        key = (r["feature_name"], r["feature_version"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(r)

    with CONTRACT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(merged)

    adm_counts: dict[str, int] = {}
    for r in merged:
        adm_counts[r["admission_status"]] = adm_counts.get(r["admission_status"], 0) + 1
    print(f"wrote {CONTRACT}  rows={len(merged)}  columns={len(COLUMNS)}")
    print(f"  new CLINICAL_MEASUREMENT_V1 rows: {len(new_rows)}")
    print(f"  preserved historical rows       : {len(old_rows)}")
    print(f"  admission counts: {adm_counts}")
    allowed = [r["feature_name"] for r in merged
               if r["allowed_in_final_classifier"] == "TRUE"]
    print(f"  allowed_in_final_classifier rows: {len(allowed)}")
    json.dump({"rows": len(merged), "new_rows": len(new_rows),
               "historical_rows": len(old_rows), "admission_counts": adm_counts,
               "allowed": allowed},
              open(ROOT / "_private_audit" / "task5b_contract_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
