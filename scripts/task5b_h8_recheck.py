#!/usr/bin/env python
"""Task 5B-H8 section B: INDEPENDENT recheck of the H7 gate from the saved raw rows.

Deliberately standalone. It does not import, call or reuse the H7 gate-summary code, and it changes
no threshold. It re-derives every H7 gate item from `_private_audit/task5b_h7_rows.csv` and
`_private_audit/task5b_h7_replay.csv`, and it separates images that are already invalid at the
native baseline from failures induced by the perturbation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/Users/moniaz/niki/_private_audit")
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
TOL = 1e-12


def main() -> None:
    L = pd.read_csv(OUT / "task5b_h7_rows.csv", low_memory=False)
    R = pd.read_csv(OUT / "task5b_h7_replay.csv", low_memory=False)
    nat = L[L.condition == "NATIVE"].set_index("image_path")
    p = L[L.condition != "NATIVE"].copy()
    for c in ("v5_valid", "v4_valid", "v5_coverage", "v4_coverage", "fov_px", "content_area"):
        p["base_" + c] = p.image_path.map(nat[c])
    p["v5_cov_delta"] = p.v5_coverage - p.base_v5_coverage
    print("=" * 100)
    print("INDEPENDENT H7 RECHECK (standalone; no H7 gate code reused)")
    print("=" * 100)
    print(f"  locked pairs {len(L)}  perturbed {len(p)}  native {len(nat)}  replay rows {len(R)}")

    # 1 masks
    mask_ok = bool(L.mask_identical.all()) and bool(R.mask_identical.all())
    print(f"  1  FOV mask V4 == V5, bitwise        : {mask_ok}  "
          f"({int(L.mask_identical.sum())}/{len(L)} + {int(R.mask_identical.sum())}/{len(R)})")
    # 2 features
    fmax = float(L.max_feature_delta.max())
    fmax_r = float(R.max_feature_delta.max())
    feat_ok = (fmax <= TOL) and (fmax_r <= TOL)
    print(f"  2  five features |V4-V5| <= {TOL:g}     : {feat_ok}  "
          f"locked max {fmax:.3e}  replay max {fmax_r:.3e}")
    # 3 H6 replay
    resolved = int((~R.v4_valid & R.v5_valid).sum())
    replay_ok = (len(R) == 16) and (resolved == 16) and int((~R.v5_valid).sum()) == 0
    print(f"  3  H6 coverage failures resolved     : {replay_ok}  "
          f"{resolved} of {len(R)} recovered, {int((~R.v5_valid).sum())} still invalid")
    # 4 baseline-valid -> perturbed-invalid
    v2i = int((p.base_v5_valid & ~p.v5_valid).sum())
    v4_2i = int((p.base_v4_valid & ~p.v4_valid).sum())
    print(f"  4  V5 valid baseline -> invalid      : {v2i}   (V4 for comparison: {v4_2i})")
    # 5 coverage failures attributable to the perturbation
    covfail = p[(p.v5_reason == "coverage_below_15pct") & p.base_v5_valid]
    covfail_any = p[p.v5_reason == "coverage_below_15pct"]
    print(f"  5  coverage_below_15pct rows         : {len(covfail_any)}"
          f"   of which baseline-valid (perturbation-attributable): {len(covfail)}")
    print(f"     V5 coverage |delta| median {p.v5_cov_delta.abs().median():.6f}  "
          f"p95 {p.v5_cov_delta.abs().quantile(.95):.6f}  max {p.v5_cov_delta.abs().max():.6f}")
    print(f"     V4 coverage |delta| median {float((p.v4_coverage - p.base_v4_coverage).abs().median()):.6f}"
          f"  max {float((p.v4_coverage - p.base_v4_coverage).abs().max()):.6f}")
    # 6/7 source and geometry on baseline-valid images
    ok = p[p.base_v5_valid]
    src = ok.groupby("source").v5_valid.mean()
    geo = ok.groupby("geom").v5_valid.mean()
    src_ok = bool((src >= 0.99).all())
    geo_ok = bool((geo >= 0.99).all())
    print(f"  6  source-specific failure            : {not src_ok}   {src.round(4).to_dict()}")
    print(f"  7  geometry-specific failure          : {not geo_ok}   {geo.round(4).to_dict()}")
    # 8 NaN transitions
    nan_new = 0
    for f in FEATS:
        nan_new += int((p[f"v4_{f}"].notna() & p[f"v5_{f}"].isna()).sum())
    print(f"  8  new NaN (V4 finite -> V5 NaN)     : {nan_new}")

    native_bad = nat[~nat.v5_valid].reset_index()
    print()
    print(f"  native-invalid images (baseline): {len(native_bad)} of {len(nat)}")
    for r in native_bad.itertuples():
        print(f"    {str(r.image_path).split('/')[-1]:46s} {r.source:8s} {r.geom:10s} "
              f"v5_cov={r.v5_coverage:.4f} fov_px={int(r.fov_px)}")
    for r in covfail_any.itertuples():
        print(f"    perturbed-invalid {str(r.image_path).split('/')[-1]:44s} {r.condition:20s} "
              f"base_valid={r.base_v5_valid} base_cov={r.base_v5_coverage:.4f} "
              f"cov={r.v5_coverage:.4f} delta={r.v5_cov_delta:+.6f}")

    checks = {"1_mask": mask_ok, "2_features": feat_ok, "3_h6_replay": replay_ok,
              "4_baseline_valid_to_invalid": v2i == 0, "5_perturbation_coverage_failures":
              len(covfail) == 0, "6_source": src_ok, "7_geometry": geo_ok,
              "8_no_new_nan": nan_new == 0}
    print()
    for k, v in checks.items():
        print(f"    {k:34s} : {'PASS' if v else 'FAIL'}")
    verdict = "PASS" if all(checks.values()) else "FAIL"
    print()
    print(f"  INDEPENDENT_H7_RECHECK = {verdict}")
    (OUT / "task5b_h8_recheck.json").write_text(json.dumps(
        {"verdict": verdict, "checks": checks, "v5_valid_to_invalid": v2i,
         "v4_valid_to_invalid": v4_2i, "coverage_fail_rows": len(covfail_any),
         "coverage_fail_perturbation_attributable": len(covfail),
         "native_invalid_n": int(len(native_bad)),
         "v5_coverage_delta_median": float(p.v5_cov_delta.abs().median()),
         "v5_coverage_delta_max": float(p.v5_cov_delta.abs().max()),
         "v4_coverage_delta_median": float((p.v4_coverage - p.base_v4_coverage).abs().median()),
         "max_feature_delta_locked": fmax, "max_feature_delta_replay": fmax_r,
         "h6_resolved": resolved, "new_nan": nan_new,
         "per_source_valid": src.to_dict(), "per_geom_valid": geo.to_dict()},
        indent=2, default=str), encoding="utf-8")
    if verdict != "PASS":
        raise SystemExit("INDEPENDENT_H7_RECHECK_FAILED")


if __name__ == "__main__":
    main()
