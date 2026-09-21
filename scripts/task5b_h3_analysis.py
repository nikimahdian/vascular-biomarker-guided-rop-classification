#!/usr/bin/env python
"""Task 5B-H3 analysis pass: every endpoint re-derived from the saved locked-run tables.

The locked run's measurement finished and both CSVs were written; only the final verdict block
crashed, on a G6 aggregation bug (`summ` subgroup entries are dicts, not scalars). This script
recomputes G, H, J, K, the canvas control I, and the gate N from the saved artefacts, and adds the
failure-mechanism diagnosis for the V2 validity regression.

No measurement of the 36 conditions is repeated: it reads `task5b_h3_rows.csv` and
`task5b_h3_baseline.csv`. The only recomputation is the zero-padding control (a pure function of the
frozen binary support) and a 40-row sample used to name the failure reason behind V2's invalid FOVs.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.task5b_h3_eval import FEATS, GATE, OUT, canvas_control  # noqa: E402
from scripts.task5b_n2 import load  # noqa: E402
from scripts.task5b_h_fov_stress import apply_border, apply_brightness, pad_specs  # noqa: E402
from scripts.task5b_h3_eval import novel_specs  # noqa: E402
from src.biomarker import clinical_measurement_v2_candidate as v2  # noqa: E402


def stats(g, col):
    r = g[col].replace([np.inf, -np.inf], np.nan).dropna()
    if not len(r):
        return {}
    return {"n": int(len(r)), "median_rel": float(r.median()), "p95_rel": float(r.quantile(.95)),
            "max_rel": float(r.max()), "min_rel": float(r.min()),
            "frac_gt_1pct": float((r.abs() > .01).mean()),
            "frac_gt_2pct": float((r.abs() > .02).mean()),
            "frac_gt_5pct": float((r.abs() > .05).mean()),
            "frac_gt_10pct": float((r.abs() > .10).mean())}


def reason_probe(arg):
    image_path, mask_path, cond = arg
    rgb, msk = load(image_path, mask_path)
    h0, w0 = msk.shape
    if cond.startswith("x"):
        prgb, pmsk, _ = apply_brightness(rgb, msk, float(cond[1:]))
    else:
        spec = [s for s in (pad_specs(h0, w0) + novel_specs(h0, w0)) if s[1] == cond]
        if not spec:
            return None
        prgb, pmsk, _ = apply_border(rgb, msk, spec[0])
    _o, _f, q = v2.measure_v2(prgb, pmsk, {})
    H, W = pmsk.shape
    return {"image_path": image_path, "condition": cond,
            "reason": str(q.get("fov_failure_reason", "")),
            "coverage": float(q.get("fov_coverage_fraction", np.nan)),
            "frame_px": int(H * W), "border_contact": float(q.get("fov_border_contact", np.nan))}


def main() -> None:
    B = pd.read_csv(OUT / "task5b_h3_baseline.csv")
    C = pd.read_csv(OUT / "task5b_h3_rows.csv")
    cb, cn = C[C.family == "brightness"], C[C.family == "border"]
    res, summ = {}, {}
    print("=" * 100)
    print(f"rows {len(C)}  border {len(cn)}  brightness {len(cb)}  images {len(B)}")
    print("=" * 100)

    print("G. FOV ROBUSTNESS — V1 vs V2 (locked sample)")
    print("=" * 100)
    for ver in ("v1", "v2"):
        d = cn[f"{ver}_dice"].dropna()
        res[f"{ver}_border_median_dice"] = float(d.median())
        res[f"{ver}_border_mean_dice"] = float(d.mean())
        res[f"{ver}_border_p01_dice"] = float(d.quantile(.01))
        res[f"{ver}_border_min_dice"] = float(d.min())
        res[f"{ver}_border_frac_lt_099"] = float((d < 0.99).mean())
        res[f"{ver}_border_frac_lt_090"] = float((d < 0.90).mean())
        res[f"{ver}_rel_area_median"] = float(cn[f"{ver}_rel_area"].median())
        res[f"{ver}_rel_area_p95"] = float(cn[f"{ver}_rel_area"].quantile(.95))
        res[f"{ver}_rel_area_max"] = float(cn[f"{ver}_rel_area"].max())
        res[f"{ver}_centroid_median"] = float(cn[f"{ver}_centroid"].median())
        res[f"{ver}_centroid_p95"] = float(cn[f"{ver}_centroid"].quantile(.95))
        print(f"  {ver.upper()}: median Dice {d.median():.6f}  mean {d.mean():.6f}  "
              f"p01 {d.quantile(.01):.6f}  min {d.min():.6f}  <0.99 {float((d < 0.99).mean()):.4f}  "
              f"<0.90 {float((d < 0.90).mean()):.4f}")
        print(f"       rel area median {cn[f'{ver}_rel_area'].median():+.6f} p95 "
              f"{cn[f'{ver}_rel_area'].quantile(.95):+.6f} max {cn[f'{ver}_rel_area'].max():+.6f}"
              f"   centroid median {cn[f'{ver}_centroid'].median():.6f}")
    vt = {}
    for ver in ("v1", "v2"):
        j = cn.join(B.set_index("image_path")[f"{ver}_valid"].rename("bv"), on="image_path")
        vt[ver] = {"valid_to_invalid": int((j.bv & ~j[f"{ver}_valid"]).sum()),
                   "invalid_to_valid": int((~j.bv & j[f"{ver}_valid"]).sum())}
        print(f"  {ver.upper()} validity: valid->invalid {vt[ver]['valid_to_invalid']}  "
              f"invalid->valid {vt[ver]['invalid_to_valid']}")
    res["validity_transitions"] = vt
    conds = cn.groupby("condition").agg(v1=("v1_dice", "median"), v2=("v2_dice", "median"),
                                        v2min=("v2_dice", "min")).sort_values("v2")
    res["by_condition_v2_min_median_dice"] = float(conds.v2.min())
    print(f"  worst conditions by V2 median Dice: "
          f"{conds.head(5).v2.round(6).to_dict()}")
    print(f"  by source:\n{cn.groupby('source')[['v1_dice', 'v2_dice']].median().round(6).to_string()}")
    print(f"  by geometry:\n{cn.groupby('geom')[['v1_dice', 'v2_dice']].median().round(6).to_string()}")
    res["by_source"] = cn.groupby("source")[["v1_dice", "v2_dice"]].median().to_dict()
    res["by_geom"] = cn.groupby("geom")[["v1_dice", "v2_dice"]].median().to_dict()

    print()
    print("H. FIVE PRIMARY FEATURES")
    print("=" * 100)
    for label, g in (("BORDER", cn), ("BRIGHTNESS", cb)):
        for ver in ("v1", "v2"):
            print(f"  --- {label} / {ver.upper()} ---")
            for f in FEATS:
                s = stats(g, f"{ver}_{f}_rel")
                summ[f"{label}|{ver}|{f}"] = s
                print(f"    {f:20s} median {s['median_rel']:+.5f}  p95 {s['p95_rel']:+.5f}  "
                      f"max {s['max_rel']:+.5f}  >1% {s['frac_gt_1pct']:.4f}  "
                      f">2% {s['frac_gt_2pct']:.4f}  >5% {s['frac_gt_5pct']:.4f}  "
                      f">10% {s['frac_gt_10pct']:.4f}")
    for key in ("source", "geom"):
        for f in FEATS:
            gg = cn.groupby(key)[f"v2_{f}_rel"].apply(lambda s: float((s.abs() > .05).mean()))
            summ[f"subgroup|{key}|{f}"] = {k: float(v) for k, v in gg.items()}

    print()
    print("J. NaN TRANSITIONS")
    print("=" * 100)
    for label, g in (("BORDER", cn), ("BRIGHTNESS", cb)):
        for ver in ("v1", "v2"):
            tot = 0
            for f in FEATS:
                b = g[f"{ver}_{f}_base"].notna()
                p = g[f"{ver}_{f}"].notna()
                fin2nan, nan2fin = int((b & ~p).sum()), int((~b & p).sum())
                tot += fin2nan
                summ[f"nan|{label}|{ver}|{f}"] = {"finite_to_nan": fin2nan, "nan_to_finite": nan2fin}
            print(f"  {label:10s} {ver.upper()}: finite->NaN {tot}")
    res["v2_border_new_nan"] = int(sum(v["finite_to_nan"] for k, v in summ.items()
                                       if k.startswith("nan|BORDER|v2|")))

    print()
    print("I. FRACTAL ZERO-PADDING CONTROL (same support, larger canvases)")
    print("=" * 100)
    meta_all = pd.read_csv("/Users/moniaz/niki/data/features/final_biomarkers_v1.csv",
                           usecols=["image_path", "mask_path"])
    Bc = B.merge(meta_all, on="image_path", how="left")
    ct = canvas_control(Bc.sort_values("image_path").head(6))
    ct.to_csv(OUT / "task5b_h3_canvas_control.csv", index=False)
    for f in ("fractal_d0", "fractal_d1", "fractal_d2"):
        r1 = ct[f"v1_{f}_rel"].replace([np.inf, -np.inf], np.nan).dropna()
        r2 = ct[f"v2_{f}_rel"].replace([np.inf, -np.inf], np.nan).dropna()
        res[f"canvas_v1_{f}_median_abs_rel"] = float(r1.abs().median())
        res[f"canvas_v2_{f}_median_abs_rel"] = float(r2.abs().median())
        res[f"canvas_v2_{f}_max_abs_rel"] = float(r2.abs().max())
        print(f"  {f:12s} V1 median|rel| {r1.abs().median():.5f} max {r1.abs().max():.5f}   "
              f"V2 median|rel| {r2.abs().median():.5f} max {r2.abs().max():.5f}")

    print()
    print("K. NATIVE-IMAGE DRIFT V1 -> V2")
    print("=" * 100)
    dd = B.drift_fov_dice.dropna()
    print(f"  native FOV Dice: median {dd.median():.6f} p05 {dd.quantile(.05):.6f} min {dd.min():.6f}")
    for f in FEATS:
        r = B[f"drift_{f}_rel"].replace([np.inf, -np.inf], np.nan).dropna()
        res[f"native_drift_{f}_median_rel"] = float(r.median())
        res[f"native_drift_{f}_p95_abs_rel"] = float(r.abs().quantile(.95))
        print(f"  {f:20s} median {r.median():+.5f}  p95|.| {r.abs().quantile(.95):.5f}  "
              f"frac>5% {float((r.abs() > .05).mean()):.4f}")
    lost = int((B.v1_valid & ~B.v2_valid).sum())
    res["native_validity_lost"] = lost
    res["native_validity_loss_frac"] = float(lost / max(len(B), 1))
    print(f"  native validity: lost {lost}  gained {int((~B.v1_valid & B.v2_valid).sum())}")

    print()
    print("MECHANISM — why V2 has more invalid FOVs than V1 under borders")
    print("=" * 100)
    j = cn.join(B.set_index("image_path")["v1_valid"].rename("bv"), on="image_path")
    reg = j[(j.bv) & (~j.v2_valid)]
    print(f"  rows where V1 baseline valid and V2 perturbed invalid : {len(reg)}")
    print("  by condition:")
    print(reg.condition.value_counts().head(12).to_string())
    samp = reg.sort_values(["condition", "image_path"]).groupby("condition").head(3).head(40)
    meta = meta_all
    mp = dict(zip(meta.image_path, meta.mask_path))
    argl = [(r.image_path, mp[r.image_path], r.condition) for r in samp.itertuples()]
    with get_context("fork").Pool(8) as pool:
        rr = [x for x in pool.map(reason_probe, argl) if x]
    RD = pd.DataFrame(rr)
    RD.to_csv(OUT / "task5b_h3_invalid_reasons.csv", index=False)
    print(f"  sampled {len(RD)} of them, recomputed V2 failure reason:")
    print(RD.reason.value_counts().to_string())
    print(f"  coverage of those rows: median {RD.coverage.median():.4f} "
          f"max {RD.coverage.max():.4f}")
    res["v2_invalid_regression_rows"] = int(len(reg))
    res["v2_invalid_regression_reasons"] = RD.reason.value_counts().to_dict()

    print()
    print("N. GATE")
    print("=" * 100)
    checks = {
        "G1_median_dice": res["v2_border_median_dice"] >= GATE["G1_border_median_dice_min"],
        "G1_p01_dice": res["v2_border_p01_dice"] >= GATE["G1_border_p01_dice_min"],
        "G1_every_condition": res["by_condition_v2_min_median_dice"]
        >= GATE["G1_every_condition_median_dice_min"],
        "G2_no_recurrent_invalid": vt["v2"]["valid_to_invalid"]
        <= GATE["G2_border_valid_to_invalid_max"],
        "G3_vessel_density": summ["BORDER|v2|vessel_density_fov"]["frac_gt_1pct"]
        <= GATE["G3_vessel_density_border_frac_gt_1pct_max"],
        "G3_skel_density": summ["BORDER|v2|skel_density_fov"]["frac_gt_1pct"]
        <= GATE["G3_skel_density_border_frac_gt_1pct_max"],
        "G4_canvas_d0": res["canvas_v2_fractal_d0_median_abs_rel"]
        <= GATE["G4_canvas_median_abs_rel_max"],
        "G4_canvas_d1": res["canvas_v2_fractal_d1_median_abs_rel"]
        <= GATE["G4_canvas_median_abs_rel_max"],
        "G4_canvas_d2": res["canvas_v2_fractal_d2_median_abs_rel"]
        <= GATE["G4_canvas_median_abs_rel_max"],
        "G5_no_new_nan": res["v2_border_new_nan"] <= GATE["G5_new_nan_frac_max"] * len(cn),
        "G6_subgroups": all(max(gg.values()) <= GATE["G6_subgroup_border_frac_gt_5pct_max"]
                            for k, gg in summ.items()
                            if k.startswith("subgroup|")
                            and k.split("|")[-1] in ("vessel_density_fov", "skel_density_fov")),
        "G7_native_validity": res["native_validity_loss_frac"]
        <= GATE["G7_native_validity_loss_frac_max"],
    }
    for k, v in checks.items():
        print(f"  {k:34s} : {'PASS' if v else 'FAIL'}")
    npass = sum(checks.values())
    gate_pass = all(checks.values())
    verdict = "PASS" if gate_pass else ("CONCERN" if npass >= len(checks) - 2 else "FAIL")
    print()
    print(f"  checks passed {npass}/{len(checks)}   -> FOV_BORDER_ROBUSTNESS_V2 = {verdict}")
    out = {"gate_checks": checks, "gate_pass": gate_pass, "checks_passed": npass,
           "checks_total": len(checks), "FOV_BORDER_ROBUSTNESS_V2": verdict,
           "results": res, "stats": summ,
           "locked_N": int(len(B)), "rows": int(len(C))}
    (OUT / "task5b_h3_summary.json").write_text(json.dumps(out, indent=2, default=str),
                                                encoding="utf-8")
    print("  summary written")


if __name__ == "__main__":
    main()
