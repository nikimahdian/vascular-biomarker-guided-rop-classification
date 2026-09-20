#!/usr/bin/env python
"""Section L: emit configs/feature_contract.csv from the real feature table.

Attributes are set from the audit findings, not invented. Any feature in the table without a row
here may not enter a model.
"""
from __future__ import annotations

import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
DEST = f"{ROOT}/feature_contract.csv"

PVBM = "pvbm_v1"
CLIN = "clinical_v3"

# feature -> (depends_on_disc, depends_on_fov, depends_on_resolution, depends_on_av,
#             clinical_status, role, allowed_in_classifier, unit, definition)
T = {
    # ---- density block -------------------------------------------------------------
    "vessel_density": (0, 1, 1, 0, "exploratory", "secondary", 1, "fraction of pixels",
                       "Vessel mask pixels / (H*W) over the full rectangular frame. "
                       "Acquisition-canvas dependent: changed by -46% with 100px of black "
                       "padding and -77% with 300px. Not a retinal quantity. See "
                       "fov_density() in geometry_core for the corrected form."),
    "vessel_pixels": (0, 1, 1, 0, "qc_only", "exploratory", 0, "count",
                      "Raw vessel pixel count. Scales with resolution and field of view; "
                      "QC only, never a clinical biomarker."),
    "area": (0, 1, 1, 0, "exploratory", "secondary", 1, "pixels",
             "PVBM mask area = sum of the binary mask. Disc-independent (measured 0.0% change "
             "under every disc perturbation) but resolution dependent."),
    # ---- 3x3 frame grid ------------------------------------------------------------
    "density_r0c0": (0, 1, 1, 0, "exploratory", "exploratory", 1, "fraction",
                     "Vessel fraction in the top-left ninth of the RECTANGLE. Frame-relative, "
                     "not anatomy-relative: the same retina lands in a different cell when the "
                     "canvas or framing changes."),
    "density_r0c1": (0, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_r0c0"),
    "density_r0c2": (0, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_r0c0"),
    "density_r1c0": (0, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_r0c0"),
    "density_r1c1": (0, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_r0c0"),
    "density_r1c2": (0, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_r0c0"),
    "density_r2c0": (0, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_r0c0"),
    "density_r2c1": (0, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_r0c0"),
    "density_r2c2": (0, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_r0c0"),
    # ---- PVBM geometry -------------------------------------------------------------
    "tortuosity_index": (0, 0, 1, 0, "validated_correct", "primary", 1, "ratio (unitless)",
                         "PVBM: sum(arc)/sum(chord) over all branch segments, arc computed with "
                         "geometric edge weights (diagonal = sqrt(2)). Audit: median relative "
                         "change <= 0.3% under every optic-disc perturbation, so it is "
                         "effectively disc-robust. Correct direction (>= 1)."),
    "median_tortuosity": (0, 0, 1, 0, "validated_correct", "primary", 1, "ratio (unitless)",
                          "PVBM: median(arc/chord) per segment. Audit: <= 0.3% median change "
                          "under disc perturbation. This is the only tortuosity feature whose "
                          "definition is verified correct against analytic arcs."),
    "overall_length": (1, 1, 1, 0, "exploratory", "secondary", 1, "pixels",
                       "PVBM: sum of branch arc lengths. CONFIRMED SENSITIVE: median 11% change "
                       "when the disc radius is inflated, 7.8% when the measured disc replaces "
                       "the fabricated one. Resolution and field-of-view dependent."),
    "median_branching_angle": (1, 1, 1, 0, "exploratory", "exploratory", 1, "degrees",
                               "PVBM: median angle at junctions. Median 62% change under centre "
                               "perturbation in the worst case, 10.4% when the measured disc is "
                               "used. Geometry-definition dependent."),
    "n_startpoints": (1, 1, 1, 0, "confirmed_defect", "secondary", 0, "count",
                      "PVBM: number of graph roots. Roots are chosen as the nearest skeleton point "
                      "to the disc centre within 100 + radius pixels, where radius came from the "
                      "FABRICATED disc max(8, min(H,W)//8) -- a pure function of image size. "
                      "Audit: median 50% change when the measured disc replaces it, up to 400%. "
                      "This is an acquisition-size feature, not a vascular one. Excluded from "
                      "classifiers pending a disc-valid recomputation."),
    "n_endpoints": (1, 1, 1, 0, "confirmed_defect", "exploratory", 0, "count",
                    "PVBM: endpoint count. Same root-selection dependency as n_startpoints. "
                    "Median 12% change with the measured disc, up to 120%. Excluded pending "
                    "recomputation."),
    "n_intersections": (1, 1, 1, 0, "confirmed_defect", "exploratory", 0, "count",
                        "PVBM: junction count. Same root-selection dependency. Median 4.5% "
                        "change with the measured disc, up to 100%. Excluded pending "
                        "recomputation."),
    # ---- fractal -------------------------------------------------------------------
    "fractal_d0": (0, 0, 1, 0, "exploratory", "secondary", 1, "dimension",
                   "PVBM multifractal capacity dimension of the mask. Independent of the disc "
                   "parameters because it is computed on the mask alone, but resolution dependent."),
    "fractal_d1": (0, 0, 1, 0, "exploratory", "secondary", 1, "dimension",
                   "PVBM multifractal information dimension. Disc-independent, resolution "
                   "dependent."),
    "fractal_d2": (0, 0, 1, 0, "exploratory", "secondary", 1, "dimension",
                   "PVBM multifractal correlation dimension. Disc-independent, resolution "
                   "dependent."),
    "singularity_length": (0, 0, 1, 0, "exploratory", "exploratory", 1, "pixels",
                           "PVBM multifractal singularity length. Disc-independent, resolution "
                           "dependent."),
}

CLINICAL = {
    "width_p50_px": (0, 0, 1, 0, "qc_only", "qc", 0, "pixels",
                     "Median vessel width at skeleton pixels, in working pixels. QC only: the "
                     "pixel is not a clinical unit."),
    "width_p90_px": (0, 0, 1, 0, "qc_only", "qc", 0, "pixels", "see width_p50_px"),
    "width_mean_px": (0, 0, 1, 0, "qc_only", "qc", 0, "pixels", "see width_p50_px"),
    "width_p50_dd": (1, 0, 1, 0, "requires_expert_data", "primary", 1, "disc diameters",
                     "Median full vessel width at skeleton pixels divided by the MEASURED disc "
                     "diameter. Verified scale-invariant in a synthetic 1x/2x/4x test. Requires "
                     "expert agreement data before it is a validated clinical measurement."),
    "width_p90_dd": (1, 0, 1, 0, "requires_expert_data", "primary", 1, "disc diameters",
                     "90th percentile width / DD. The primary calibre candidate."),
    "width_p95_dd": (1, 0, 1, 0, "requires_expert_data", "secondary", 1, "disc diameters",
                     "95th percentile width / DD."),
    "width_mean_dd": (1, 0, 1, 0, "requires_expert_data", "secondary", 1, "disc diameters",
                      "Mean width / DD."),
    "width_shape_p90_over_p50": (0, 0, 0, 0, "exploratory", "exploratory", 1, "ratio",
                                 "width_p90_px / width_p50_px. Scale-free by construction."),
    "width_ann_p50_dd": (1, 0, 1, 0, "requires_expert_data", "secondary", 1, "disc diameters",
                         "Width / DD restricted to the 0.5-2.0 DD peripapillary zone."),
    "width_ann_p90_dd": (1, 0, 1, 0, "requires_expert_data", "secondary", 1, "disc diameters",
                         "see width_ann_p50_dd"),
    "width_ann_mean_dd": (1, 0, 1, 0, "requires_expert_data", "secondary", 1, "disc diameters",
                          "see width_ann_p50_dd"),
    "skel_density": (0, 1, 1, 0, "exploratory", "secondary", 1, "fraction",
                     "Skeleton pixels / total pixels. Canvas dependent, like vessel_density."),
    "n_skel_px": (0, 1, 1, 0, "qc_only", "qc", 0, "count", "Skeleton pixel count. QC only."),
    "n_skel_ann_px": (1, 0, 1, 0, "qc_only", "qc", 0, "count",
                      "Skeleton pixels inside the annulus. QC and coverage check."),
    "n_branches": (0, 0, 1, 0, "exploratory", "exploratory", 1, "count",
                   "Junction-free skeleton segments. Resolution dependent."),
    "tort_median": (0, 0, 1, 0, "confirmed_defect", "exploratory", 0, "ratio",
                    "clinical_v3 tortuosity: skeleton pixel count / convex-hull chord. CONFIRMED "
                    "BUG: orientation dependent (1.005 at 0 deg, 0.712 at 45 deg for the same "
                    "straight vessel). Superseded by branch_tortuosity() in geometry_core. Never "
                    "consumed by Branch A or C."),
    "tort_p90": (0, 0, 1, 0, "confirmed_defect", "exploratory", 0, "ratio",
                 "see tort_median; same defect."),
    "tort_top3_mean": (0, 0, 1, 0, "confirmed_defect", "exploratory", 0, "ratio",
                       "see tort_median; same defect."),
    "a_frac": (0, 0, 1, 1, "exploratory", "exploratory", 1, "fraction",
               "Length-weighted fraction of skeleton assigned to the arterial class by an "
               "unsupervised green-intensity split. Exploratory until expert A/V labels exist."),
    "a_density": (0, 1, 1, 1, "exploratory", "exploratory", 1, "fraction",
                  "Arterial-class vessel fraction. A/V heuristic and canvas dependent."),
    "v_density": (0, 1, 1, 1, "exploratory", "exploratory", 1, "fraction",
                  "Venous-class vessel fraction. A/V heuristic and canvas dependent."),
    "a_width_p90_px": (0, 0, 1, 1, "qc_only", "qc", 0, "pixels", "see width_p50_px"),
    "v_width_p90_px": (0, 0, 1, 1, "qc_only", "qc", 0, "pixels", "see width_p50_px"),
    "a_width_p90_dd": (1, 0, 1, 1, "exploratory", "exploratory", 1, "disc diameters",
                       "Arterial width / DD. Depends on the unvalidated A/V heuristic; must not "
                       "be a primary clinical biomarker until expert A/V labels exist."),
    "v_width_p90_dd": (1, 0, 1, 1, "exploratory", "exploratory", 1, "disc diameters",
                       "Venous width / DD. Same A/V caveat."),
    "av_width_ratio_p90": (0, 0, 0, 1, "exploratory", "exploratory", 1, "ratio",
                           "Arterial/venous width ratio. The disc cancels, so it is defined even "
                           "without a disc. Same A/V caveat."),
    "a_tort_median": (0, 0, 1, 1, "confirmed_defect", "exploratory", 0, "ratio",
                      "Always NaN in the historical table: the column is 100% null. Dead feature."),
    "v_tort_median": (0, 0, 1, 1, "confirmed_defect", "exploratory", 0, "ratio",
                      "Always NaN in the historical table: the column is 100% null. Dead feature."),
    "density_ring_0_2dd": (1, 1, 1, 0, "exploratory", "secondary", 1, "fraction",
                           "Vessel fraction in the 0-2 DD ring. Depends on the disc and on how "
                           "much of the ring is inside the frame; coverage must be reported."),
    "coverage_ring_0_2dd": (1, 1, 1, 0, "qc_only", "qc", 0, "fraction",
                            "Fraction of the frame inside the ring. A coverage/QC feature."),
    "density_ring_2_3dd": (1, 1, 1, 0, "exploratory", "exploratory", 1, "fraction",
                           "see density_ring_0_2dd"),
    "coverage_ring_2_3dd": (1, 1, 1, 0, "qc_only", "qc", 0, "fraction", "see coverage_ring_0_2dd"),
    "density_ring_3_6dd": (1, 1, 1, 0, "exploratory", "exploratory", 1, "fraction",
                           "see density_ring_0_2dd"),
    "coverage_ring_3_6dd": (1, 1, 1, 0, "qc_only", "qc", 0, "fraction", "see coverage_ring_0_2dd"),
    "density_q_ne": (1, 1, 1, 0, "exploratory", "exploratory", 1, "fraction",
                     "Quadrant density. Quadrants are geometric (top-right etc.), NOT "
                     "laterality-aware, so they are not the clinical quadrants of ICROP."),
    "density_q_nw": (1, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_q_ne"),
    "density_q_sw": (1, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_q_ne"),
    "density_q_se": (1, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_q_ne"),
    "n_quad_above_median": (1, 1, 1, 0, "exploratory", "exploratory", 1, "count",
                            "Number of geometric quadrants above their median."),
    "quad_density_max": (1, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_q_ne"),
    "quad_density_min": (1, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_q_ne"),
    "quad_density_range": (1, 1, 1, 0, "exploratory", "exploratory", 1, "fraction", "see density_q_ne"),
    "disc_valid": (1, 0, 0, 0, "qc_only", "qc", 0, "boolean",
                   "DETECTOR STATUS. Must never be a predictor. Reported so missingness can be "
                   "analysed; barred from every classifier matrix by META_COLS."),
    "disc_peak_prob": (1, 0, 0, 0, "qc_only", "qc", 0, "probability",
                       "DETECTOR CONFIDENCE. Must never be a predictor."),
    "disc_method": (1, 0, 0, 0, "qc_only", "qc", 0, "category",
                    "DETECTOR STATUS. Must never be a predictor."),
    "dd_px_work": (1, 0, 1, 0, "qc_only", "qc", 0, "pixels",
                   "Disc diameter in working pixels. QC only."),
    "dd_over_min_side": (1, 1, 1, 0, "exploratory", "secondary", 1, "ratio",
                         "Disc diameter / min(H,W). An acquisition-scale feature."),
    "disc_cx_frac": (1, 1, 1, 0, "exploratory", "exploratory", 1, "fraction",
                     "Disc centre x as a fraction of the width. A framing feature."),
    "disc_cy_frac": (1, 1, 1, 0, "exploratory", "exploratory", 1, "fraction",
                     "Disc centre y as a fraction of the height. A framing feature."),
    "disc_centre_offset_frac": (1, 1, 1, 0, "exploratory", "exploratory", 1, "fraction",
                                "Distance from the disc centre to the image centre. A framing "
                                "feature; correlates with source."),
}

rows = []
for table, mapping, script in (
        ("biomarker_features.csv", T, "src/biomarker/extract_pvbm.py"),
        ("biomarker_features_clinical_v3.csv", CLINICAL, "scripts/clinical_features_v3.py")):
    try:
        cols = list(pd.read_csv(f"{ROOT}/data/features/{table}", nrows=1).columns)
    except Exception as e:  # noqa: BLE001
        print(f"[skip] {table}: {e}")
        continue
    meta = {"image_path", "mask_path", "label", "split", "source", "group_id",
            "patient_id", "exam_id", "identity_level", "feature_version"}
    for c in cols:
        if c in meta:
            continue
        spec = mapping.get(c)
        if spec is None:
            spec = (0, 0, 0, 0, "unresolved", "exploratory", 0, "unknown",
                    "NO CONTRACT ENTRY. May not enter a model until described.")
        d, fov, res, av, status, role, allowed, unit, definition = spec
        rows.append({
            "feature_name": c,
            "feature_version": "pvbm_v1" if table.startswith("biomarker_features.csv") else CLIN,
            "definition": definition,
            "unit": unit,
            "depends_on_disc": d, "depends_on_fov": fov, "depends_on_resolution": res,
            "depends_on_av_heuristic": av,
            "clinical_status": status,
            "primary_secondary_exploratory": role,
            "missingness_policy": ("NaN when disc_valid = 0; never imputed from the image centre"
                                   if d else "not disc dependent; NaN only on extraction failure"),
            "allowed_in_classifier": allowed,
            "validated_by_expert": 0,
            "source_sensitive": 1 if (fov or res or d) else 0,
            "generation_script": script,
        })

c = pd.DataFrame(rows)
c.to_csv(DEST, index=False)
print(f"[done] {DEST}  rows={len(c)}")
print(c.clinical_status.value_counts().to_string())
print()
print(c.groupby(["feature_version", "allowed_in_classifier"]).size().to_string())
