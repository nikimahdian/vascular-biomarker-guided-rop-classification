#!/usr/bin/env python
"""Task 5B-H7: direct unit invariance, behaviour-preservation gate, H6 replay, and the locked
evaluation of CLINICAL_MEASUREMENT_V5_CANDIDATE (content-relative FOV coverage validity).

V5 changes one line of V4: `coverage = fov_pixels / content_area` instead of `/ full_canvas_area`.
The change is validity-only, so this script first PROVES that the FOV mask is bit-identical and the
five primary features are equal within 1e-12, and stops if they are not.

Order: freeze -> manifest -> D unit invariance -> C behaviour preservation -> E H6 replay ->
F locked N=100 -> G endpoints -> H gate -> J report.

No classifier. No biomarker formula change. No FOV mask generation change. No 8,870 regeneration.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.task5b_n2 import load  # noqa: E402
from scripts.task5b_h_fov_stress import GEOM, apply_border, pad_specs  # noqa: E402
from scripts.task5b_h3_eval import novel_specs  # noqa: E402
from src.biomarker import clinical_measurement_v4_candidate as v4  # noqa: E402
from src.biomarker import clinical_measurement_v5_candidate as v5  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
CONDS = ["irregular_frame_w3", "tb_thick_w3", "novel_black_w5", "asymmetric_black_w3",
         "dark_gray_40_w2", "novel_gray90_w2"]
NATIVE = "NATIVE"
TOL = 1e-12
GATE = {"G6_valid_to_invalid_max": 1}


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2.0 * int(np.logical_and(a, b).sum()) / s) if s else float("nan")


def unit_tests():
    print("=" * 100)
    print("D. DIRECT UNIT INVARIANCE TEST")
    print("=" * 100)
    rng = np.random.default_rng(7)
    base = np.zeros((72, 88), np.float32)
    yy, xx = np.mgrid[0:72, 0:88]
    base[:] = 25 + 130 * (((xx - 44) ** 2 + (yy - 36) ** 2) < 900)
    base += rng.normal(0, 3, base.shape).astype(np.float32)
    base = np.clip(base, 0, 255)
    rows = []
    for pad, level in ((0, 0), (5, 60), (20, 60), (60, 60), (140, 60), (300, 60), (140, 150)):
        g = np.pad(base, pad, constant_values=level) if pad else base.copy()
        f4, q4 = v4.retinal_fov_v4(g)
        f5, q5 = v5.retinal_fov_v5(g)
        top, bottom, left, right, has = v4.content_bounds_v4(np.nan_to_num(g, nan=0.0))
        content = g[top:g.shape[0] - bottom, left:g.shape[1] - right]
        rows.append({"pad": pad, "level": level, "canvas": f"{g.shape[0]}x{g.shape[1]}",
                     "content": f"{content.shape[0]}x{content.shape[1]}",
                     "content_identical": bool(np.array_equal(content, base)),
                     "fov_px": int(f5.sum()), "fov_identical_v4": bool(np.array_equal(f4, f5)),
                     "v4_coverage": float(q4.get("fov_coverage_fraction", np.nan)),
                     "v5_coverage": float(q5.get("fov_coverage_fraction", np.nan)),
                     "v4_valid": bool(q4.get("fov_valid")),
                     "v5_valid": bool(q5.get("fov_valid"))})
    U = pd.DataFrame(rows)
    print(U.to_string(index=False))
    U.to_csv(OUT / "task5b_h7_unit_tests.csv", index=False)
    ok = (bool(U.content_identical.all()) and bool(U.fov_identical_v4.all())
          and bool(U.fov_px.nunique() == 1) and bool(U.v5_coverage.nunique() == 1)
          and bool(U.v5_valid.nunique() == 1) and bool(U.canvas.nunique() > 1))
    print()
    print(f"  content rectangle identical after mapping : {bool(U.content_identical.all())}")
    print(f"  FOV pixels identical across all pads     : {bool(U.fov_px.nunique() == 1)}")
    print(f"  V5 content-relative coverage identical   : {bool(U.v5_coverage.nunique() == 1)}"
          f"   value {U.v5_coverage.iloc[0]:.6f}")
    print(f"  V5 validity identical                    : {bool(U.v5_valid.nunique() == 1)}")
    print(f"  full canvas size changes                 : {bool(U.canvas.nunique() > 1)}"
          f"   {U.canvas.tolist()}")
    print(f"  V4 coverage by contrast                  : "
          f"{U.v4_coverage.round(6).tolist()}")
    print(f"  CONTENT_RELATIVE_COVERAGE_INVARIANCE : {'PASS' if ok else 'FAIL'}")
    return ok, U


def pair_probe(arg):
    """C/E: one (image, condition) measured by V4 and V5 with a bitwise mask comparison."""
    image_path, mask_path, cond = arg
    rgb, msk = load(image_path, mask_path)
    h0, w0 = msk.shape
    if cond == NATIVE:
        prgb, pmsk, pad = rgb, msk, (0, 0, 0, 0)
    else:
        spec = [s for s in (pad_specs(h0, w0) + novel_specs(h0, w0)) if s[1] == cond][0]
        prgb, pmsk, pad = apply_border(rgb, msk, spec)
    o4, f4, q4 = v4.measure_v4(prgb, pmsk, {})
    o5, f5, q5 = v5.measure_v5(prgb, pmsk, {})
    dmax = 0.0
    for f in FEATS:
        a, b = o4.get(f, np.nan), o5.get(f, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            dmax = max(dmax, abs(a - b))
        elif not (np.isnan(a) and np.isnan(b)):
            dmax = float("inf")
    return {"image_path": image_path, "condition": cond,
            "mask_identical": bool(np.array_equal(f4, f5)),
            "fov_px": int(f5.sum()),
            "max_feature_delta": float(dmax),
            "v4_coverage": float(q4.get("fov_coverage_fraction", np.nan)),
            "v5_coverage": float(q5.get("fov_coverage_fraction", np.nan)),
            "v4_valid": bool(q4.get("fov_valid")),
            "v5_valid": bool(q5.get("fov_valid")),
            "v4_reason": str(q4.get("fov_failure_reason", "")),
            "v5_reason": str(q5.get("fov_failure_reason", "")),
            "content_area": int(q5.get("fov_content_area", 0)),
            "canvas_area": int(prgb.shape[0] * prgb.shape[1]),
            **{f"v4_{f}": o4.get(f, np.nan) for f in FEATS},
            **{f"v5_{f}": o5.get(f, np.nan) for f in FEATS}}


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("A. FROZEN PRIOR VERSIONS")
    print("=" * 100)
    frozen = {k: sha(p) for k, p in {
        "v1_sha256": ROOT / "src/biomarker/clinical_measurement_v1.py",
        "v2_sha256": ROOT / "src/biomarker/clinical_measurement_v2_candidate.py",
        "v3_sha256": ROOT / "src/biomarker/clinical_measurement_v3_candidate.py",
        "v4_sha256": ROOT / "src/biomarker/clinical_measurement_v4_candidate.py",
        "v5_sha256": ROOT / "src/biomarker/clinical_measurement_v5_candidate.py",
        "task5b_h6_manifest_sha256": OUT / "task5b_h6_manifest.json",
        "task5b_h6_summary_sha256": OUT / "task5b_h6_summary.json",
        "task5b_h6_rows_sha256": OUT / "task5b_h6_rows.csv",
    }.items()}
    for k, val in frozen.items():
        print(f"  {k:30s} : {val}")
    manifest = {
        "candidate": v5.MEASUREMENT_VERSION_V5,
        "single_change": "coverage = fov_pixels / (content_h * content_w), where the content "
                         "rectangle is the one V4 already derives; V4 used the full canvas area",
        "unchanged": ["0.15 failure threshold", "0.985 upper threshold", "0.90 component ratio",
                      "FOV mask generation", "Otsu", "morphology", "component selection",
                      "fractal domain", "all five biomarker formulas", "SEG_CURRENT_V1"],
        "behaviour_preservation_requirement": {"FOV mask": "bitwise identical",
                                               "five features": f"abs delta <= {TOL}"},
        "sample_rule": "all 8870 minus every previous sample (328+450+450+150 = 1378), sorted by "
                       "image_path, first k per (source, geom, split) with the smallest k in "
                       "{2,3,4,5,6,8,10} reaching N >= 100, then head(100)",
        "conditions": [NATIVE] + CONDS,
        "gate": GATE,
        "stop_rule": "fail -> no V6, no regeneration, no training",
    }
    (OUT / "task5b_h7_manifest.json").write_text(json.dumps(
        {"frozen": frozen, "manifest": manifest}, indent=2), encoding="utf-8")
    print("  manifest written: _private_audit/task5b_h7_manifest.json")

    d_ok, _U = unit_tests()
    if not d_ok:
        print("\n  STOP: unit coverage invariance failed")
        raise SystemExit("UNIT_COVERAGE_INVARIANCE_FAILED")

    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                       usecols=["image_path", "mask_path", "source", "split"])
    meta = meta.sort_values("image_path").reset_index(drop=True)
    meta["geom"] = [GEOM.get(Image.open(p).size, "other") for p in meta.image_path]
    mp = dict(zip(meta.image_path, meta.mask_path))

    print()
    print("=" * 100)
    print("E. REPLAY OF THE 16 H6 valid->invalid ROWS (development, not new validation)")
    print("=" * 100)
    h6 = pd.read_csv(OUT / "task5b_h6_rows.csv", low_memory=False)
    b6 = pd.read_csv(OUT / "task5b_h6_baseline.csv")
    j6 = h6.join(b6.set_index("image_path")["v4_valid"].rename("bv"), on="image_path")
    vi = j6[(j6.condition != NATIVE) & j6.bv & ~j6.v4_valid]
    print(f"  H6 rows with V4 valid baseline -> V4 invalid perturbed : {len(vi)}")
    args = [(r.image_path, mp[r.image_path], r.condition) for r in vi.itertuples()]
    if args:
        with get_context("fork").Pool(8) as p:
            R = pd.DataFrame(p.map(pair_probe, args))
        R.to_csv(OUT / "task5b_h7_replay.csv", index=False)
        print(R[["condition", "v4_coverage", "v5_coverage", "v4_valid", "v5_valid",
                 "v4_reason", "v5_reason", "mask_identical", "max_feature_delta",
                 "canvas_area", "content_area"]].to_string(index=False))
        resolved = int((~R.v4_valid & R.v5_valid).sum())
        remaining = int((~R.v5_valid).sum())
        print(f"  resolved by V5 : {resolved} of {len(R)}    still invalid under V5 : {remaining}")
        if remaining:
            print("  remaining reasons:", R[~R.v5_valid].v5_reason.value_counts().to_dict())
        replay = {"n": int(len(R)), "resolved": resolved, "remaining": remaining,
                  "mask_identical_all": bool(R.mask_identical.all()),
                  "max_feature_delta": float(R.max_feature_delta.max()),
                  "v5_reasons": R[~R.v5_valid].v5_reason.value_counts().to_dict()}
    else:
        replay = {"n": 0, "resolved": 0, "remaining": 0}

    print()
    print("=" * 100)
    print("F. LOCKED SAMPLE")
    print("=" * 100)
    prev = set()
    for f in ("task5b_h2_baseline.csv", "task5b_h3_baseline.csv", "task5b_h4_baseline.csv",
              "task5b_h6_baseline.csv"):
        prev |= set(pd.read_csv(OUT / f, usecols=["image_path"]).image_path)
    pool = meta[~meta.image_path.isin(prev)]
    strat = ["source", "geom", "split"]
    sel, used_k = pool.groupby(strat, group_keys=False).head(2), 2
    for k in (3, 4, 5, 6, 8, 10):
        if len(sel) >= 100:
            break
        sel, used_k = pool.groupby(strat, group_keys=False).head(k), k
    sel = sel.sort_values("image_path").head(100).reset_index(drop=True)
    print(f"  previous samples excluded : {len(prev)}")
    print(f"  pool                      : {len(pool)}")
    print(f"  per-stratum cap used      : {used_k}")
    print(f"  LOCKED_SAMPLE_N           : {len(sel)}")
    print(f"  disjoint                  : {not (set(sel.image_path) & prev)}")
    print(f"  sources {sel.source.value_counts().to_dict()}")
    print(f"  geoms   {sel.geom.value_counts().to_dict()}")
    print(f"  splits  {sel.split.value_counts().to_dict()}")
    if len(sel) != 100 or (set(sel.image_path) & prev):
        raise SystemExit("LOCKED_SAMPLE_INVALID")

    print()
    print("C/G. LOCKED RUN — V4 vs V5, behaviour-preservation gate plus endpoints")
    print("=" * 100)
    args = [(r.image_path, mp[r.image_path], NATIVE) for r in sel.itertuples()]
    for cond in CONDS:
        args += [(r.image_path, mp[r.image_path], cond) for r in sel.itertuples()]
    rows = []
    with get_context("fork").Pool(16) as p:
        for i, r in enumerate(p.imap_unordered(pair_probe, args, chunksize=1), 1):
            rows.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(args)}  {time.time() - t0:.0f}s", flush=True)
    L = pd.DataFrame(rows)
    L = L.merge(sel[["image_path", "source", "split", "geom"]], on="image_path", how="left")
    L.to_csv(OUT / "task5b_h7_rows.csv", index=False)
    pert = L[L.condition != NATIVE]
    print(f"  pairs {len(L)}  perturbed {len(pert)}  native {int((L.condition == NATIVE).sum())}")

    mask_ok = bool(L.mask_identical.all())
    fmax = float(L.max_feature_delta.max())
    print()
    print(f"  FOV mask bitwise identical on ALL {len(L)} pairs       : {mask_ok}")
    print(f"  max |V4 - V5| over the five features (all pairs)       : {fmax:.3e}"
          f"   (tolerance {TOL:g})")
    print(f"  PRIMARY_FEATURES_V4_V5_EQUIVALENT                      : {fmax <= TOL}")
    nb = L[L.condition == NATIVE]
    print(f"  native pairs {len(nb)}  coverage V4 median "
          f"{nb.v4_coverage.median():.6f} -> V5 {nb.v5_coverage.median():.6f}")
    print(f"  native validity V4->V5: lost {int((nb.v4_valid & ~nb.v5_valid).sum())}  "
          f"gained {int((~nb.v4_valid & nb.v5_valid).sum())}")
    print("  native coverage change distribution: "
          f"median {float((nb.v5_coverage - nb.v4_coverage).median()):+.6f}  "
          f"p95 {float((nb.v5_coverage - nb.v4_coverage).quantile(.95)):+.6f}  "
          f"max {float((nb.v5_coverage - nb.v4_coverage).max()):+.6f}")

    print()
    print("  perturbed endpoints:")
    v4i = int((pert.v4_valid & ~pert.v5_valid).sum())
    v5i = int((~pert.v4_valid & pert.v5_valid).sum())
    print(f"    V4 valid->invalid {v4i}   V5 valid->invalid {v5i}   "
          f"V4-invalid-recovered-by-V5 {int((~pert.v4_valid & pert.v5_valid).sum())}")
    print(f"    V4 coverage_below_15pct {int((pert.v4_reason == 'coverage_below_15pct').sum())}"
          f"   V5 coverage_below_15pct {int((pert.v5_reason == 'coverage_below_15pct').sum())}")
    print(f"    V4 coverage_above_98pct {int((pert.v4_reason.str.startswith('coverage_above') ).sum())}"
          f"   V5 coverage_above_98pct "
          f"{int((pert.v5_reason.str.startswith('coverage_above')).sum())}")
    print(f"    V5 other reasons: "
          f"{pert[pert.v5_reason != ''].v5_reason.value_counts().to_dict()}")
    print("    by condition (v5_reason):")
    print(pert[pert.v5_reason != ""].groupby("condition").v5_reason.value_counts().to_string())
    print("    by source (V5 valid):")
    print(pert.groupby("source").v5_valid.mean().round(4).to_string())
    print("    by geometry (V5 valid):")
    print(pert.groupby("geom").v5_valid.mean().round(4).to_string())
    nan_new = 0
    for f in FEATS:
        nan_new += int((pert[f"v4_{f}"].notna() & pert[f"v5_{f}"].isna()).sum())
    print(f"    new NaN (V4 finite -> V5 NaN) over the five features: {nan_new}")

    print()
    print("=" * 100)
    print("H. GATE")
    print("=" * 100)
    checks = {
        "G1_mask_identical": mask_ok,
        "G2_features_within_tol": fmax <= TOL,
        "G3_unit_coverage_invariance": d_ok,
        "G4_h6_failures_eliminated": replay["remaining"] == 0,
        "G5_no_canvas_only_coverage_failure":
            int((pert.v5_reason == "coverage_below_15pct").sum()) == 0,
        "G6_valid_to_invalid": v5i <= GATE["G6_valid_to_invalid_max"],
        "G7_no_source_failure": bool((pert.groupby("source").v5_valid.mean() >= 0.99).all()),
        "G8_no_geometry_failure": bool((pert.groupby("geom").v5_valid.mean() >= 0.99).all()),
        "G9_no_new_nan": nan_new == 0,
    }
    for k, vv in checks.items():
        print(f"  {k:38s} : {'PASS' if vv else 'FAIL'}")
    npass = sum(checks.values())
    ok = all(checks.values())
    print()
    print(f"  checks passed {npass}/{len(checks)}  -> CLINICAL_MEASUREMENT_V5_STATUS = "
          f"{'SUPPORTED' if ok else 'NOT_SUPPORTED'}")
    out = {"locked_N": int(len(sel)), "pairs": int(len(L)),
           "FOV_MASK_V4_V5_IDENTICAL": mask_ok,
           "PRIMARY_FEATURES_V4_V5_EQUIVALENT": fmax <= TOL,
           "MAX_PRIMARY_FEATURE_DELTA": fmax,
           "CONTENT_RELATIVE_COVERAGE_INVARIANCE": "PASS" if d_ok else "FAIL",
           "H6_COVERAGE_FAILURES_REPLAYED": replay.get("n", 0),
           "H6_COVERAGE_FAILURES_RESOLVED": replay.get("resolved", 0),
           "H6_REMAINING": replay.get("remaining", 0),
           "LOCKED_VALID_TO_INVALID_N": v5i,
           "LOCKED_COVERAGE_BELOW_15PCT_N":
               int((pert.v5_reason == "coverage_below_15pct").sum()),
           "NEW_NAN_FAILURES": nan_new > 0,
           "gate_checks": checks, "checks_passed": npass, "checks_total": len(checks),
           "CLINICAL_MEASUREMENT_V5_STATUS": "SUPPORTED" if ok else "NOT_SUPPORTED",
           "FULL_8870_REGENERATION_ALLOWED": bool(ok),
           "replay": replay, "frozen": frozen, "manifest": manifest}
    (OUT / "task5b_h7_summary.json").write_text(json.dumps(out, indent=2, default=str),
                                                encoding="utf-8")
    print(f"\n  elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
