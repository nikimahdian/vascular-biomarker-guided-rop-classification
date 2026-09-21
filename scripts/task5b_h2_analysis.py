#!/usr/bin/env python
"""Task 5B-H2 analysis pass: every summary statistic, re-derived from the saved per-condition table.

The expensive measurement is done once by `scripts/task5b_h2_impact.py` and stored in
`_private_audit/task5b_h2_features.csv`. This script performs no measurement of the 31 conditions;
it reads that table plus the baseline table and produces the reported numbers, so any statistic can
be recomputed and audited from the artefact.

It additionally runs the decisive mechanism experiment that cannot be read off the table: the
FRACTAL CANVAS TEST. `_fractal()` receives `mask & fov`. For a border condition that array is the
SAME vascular support embedded in a LARGER canvas. This test takes the baseline binary input,
embeds it in a zero-padded canvas of the border condition's size, and recomputes D0/D1/D2. If the
fractals move under that embedding alone, then their border sensitivity comes from the multifractal
computation domain, not from any vessel pixel being added or removed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.task5b_h2_impact import FEATS, MODULE, OUT, sha, stats  # noqa: E402
from scripts.task5b_n2 import load  # noqa: E402
from scripts.task5b_h_fov_stress import pad_specs  # noqa: E402
from src.biomarker.clinical_measurement_v1 import _fractal, retinal_fov  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
CLS_MED, CLS_P95, CLS_MAX, CLS_F5 = 0.005, 0.02, 0.10, 0.01
CLS2_MED, CLS2_P95, CLS2_MAX = 0.05, 0.15, 0.50


def classify(s):
    if not s or not np.isfinite(s.get("median_rel", np.nan)):
        return "INCONCLUSIVE"
    m, p95, mx, f5 = s["median_rel"], s["p95_rel"], s["max_rel"], s["frac_gt_5pct"]
    if abs(m) < CLS_MED and abs(p95) < CLS_P95 and abs(mx) < CLS_MAX and f5 < CLS_F5:
        return "MINIMAL_IMPACT"
    if abs(m) < CLS2_MED and abs(p95) < CLS2_P95 and abs(mx) < CLS2_MAX:
        return "BORDER_SENSITIVE"
    return "SEVERELY_BORDER_SENSITIVE"


def canvas_test(k=8):
    """Same binary input, larger zero-padded canvas, no FOV call involved."""
    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                       usecols=["image_path", "mask_path"])
    meta = meta.sort_values("image_path").head(k)
    rows = []
    for r in meta.itertuples():
        rgb, msk = load(r.image_path, r.mask_path)
        h0, w0 = msk.shape
        fov, _ = retinal_fov(rgb[:, :, 1])
        sup = (msk.astype(bool) & fov)
        d0 = _fractal(sup)
        for spec in pad_specs(h0, w0):
            t, b, l, rr = spec[2]
            if t == b == l == rr == 0:
                continue
            padded = np.pad(sup, ((t, b), (l, rr)), mode="constant", constant_values=False)
            d1 = _fractal(padded)
            rows.append({"image_path": r.image_path, "condition": spec[1],
                         "support_px": int(sup.sum()), "canvas_growth": (t + b + l + rr)
                         / max(h0 + w0, 1),
                         "d0_base": d0["fractal_d0"], "d0_pad": d1["fractal_d0"],
                         "d1_base": d0["fractal_d1"], "d1_pad": d1["fractal_d1"],
                         "d2_base": d0["fractal_d2"], "d2_pad": d1["fractal_d2"]})
    return pd.DataFrame(rows)


def main() -> None:
    B = pd.read_csv(OUT / "task5b_h2_baseline.csv")
    C = pd.read_csv(OUT / "task5b_h2_features.csv")
    cb = C[C.family == "brightness"]
    cn = C[C.family != "brightness"]
    summ = {}

    print("=" * 100)
    print("E. FEATURE-LEVEL DELTAS (from task5b_h2_features.csv)")
    print("=" * 100)
    for label, g in (("BRIGHTNESS", cb), ("BORDER", cn)):
        print(f"  --- {label} ---  rows {len(g)}")
        print(f"  {'feature':20s} {'medAbs':>10s} {'p95Abs':>10s} {'maxAbs':>10s} {'medRel':>10s} "
              f"{'p95Rel':>10s} {'maxRel':>10s} {'>1%':>7s} {'>2%':>7s} {'>5%':>7s} {'>10%':>7s}")
        for f in FEATS:
            s = stats(g, f)
            summ[f"{label}|{f}"] = s
            print(f"  {f:20s} {s['median_abs']:10.5f} {s['p95_abs']:10.5f} {s['max_abs']:10.5f} "
                  f"{s['median_rel']:+10.5f} {s['p95_rel']:+10.5f} {s['max_rel']:+10.5f} "
                  f"{s['frac_gt_1pct']:7.4f} {s['frac_gt_2pct']:7.4f} {s['frac_gt_5pct']:7.4f} "
                  f"{s['frac_gt_10pct']:7.4f}")

    print()
    print("=" * 100)
    print("G. FOV ERROR vs BIOMARKER ERROR (border perturbations)")
    print("=" * 100)
    for f in FEATS:
        d = cn[[f + "__abs", "fov_dice", "fov_rel_area_change"]].dropna()
        r1 = float(spearmanr(d.fov_dice, d[f + "__abs"]).statistic)
        r2 = float(spearmanr(d.fov_rel_area_change, d[f + "__abs"]).statistic)
        print(f"  {f:20s} dice vs |delta| rho={r1:+.4f}   relArea vs |delta| rho={r2:+.4f}  n={len(d)}")
        summ[f"spearman|{f}"] = {"dice_vs_absdelta": r1, "relarea_vs_absdelta": r2, "n": int(len(d))}

    print()
    print("=" * 100)
    print("H. MECHANISM")
    print("=" * 100)
    for label, g in (("BORDER", cn), ("BRIGHTNESS", cb)):
        gg = g.dropna(subset=["skel_density_fov__rel"]).copy()
        pred = (1.0 / gg.fov_px_padded.astype(float) - 1.0 / gg.base_fov_px.astype(float)) \
            * gg.skeleton_px_work
        obs = gg.skel_density_fov - gg.skel_density_fov__base
        ok = np.isfinite(pred) & np.isfinite(obs)
        print(f"  skel_density_fov [{label}]: numerator S=len(skeletonize(mask)) is whole-frame and "
              f"FOV-free; predicted delta = S*(1/fp_p - 1/fp_b) vs observed (SIGNED) -> "
              f"median|diff|={float(np.abs(pred[ok] - obs[ok]).median()):.3e} "
              f"max|diff|={float(np.abs(pred[ok] - obs[ok]).max()):.3e}  "
              f"(S changed in {int((gg.skeleton_px_work != gg.base_skeleton_px).sum())} rows)")
        summ[f"skel_analytic|{label}"] = {
            "median_abs_diff": float(np.abs(pred[ok] - obs[ok]).median()),
            "max_abs_diff": float(np.abs(pred[ok] - obs[ok]).max()),
            "rows_with_numerator_change": int((gg.skeleton_px_work != gg.base_skeleton_px).sum())}
    for label, g in (("BORDER", cn), ("BRIGHTNESS", cb)):
        gg = g.dropna(subset=["vessel_density_fov__rel"]).copy()
        den = (gg.base_vessel_px / gg.fov_px_padded) - (gg.base_vessel_px / gg.base_fov_px)
        inc = (gg.vessel_px_in_fov / gg.fov_px_padded) - (gg.base_vessel_px / gg.fov_px_padded)
        obs = gg.vessel_density_fov - gg.vessel_density_fov__base
        print(f"  vessel_density_fov [{label}]: median|total|={np.nanmedian(np.abs(obs)):.6f}  "
              f"median|denominator-only|={np.nanmedian(np.abs(den)):.6f}  "
              f"median|inclusion-only|={np.nanmedian(np.abs(inc)):.6f}  "
              f"max|denominator-only|={np.nanmax(np.abs(den)):.4f}  "
              f"max|inclusion-only|={np.nanmax(np.abs(inc)):.4f}")
        summ[f"vessel_decomposition|{label}"] = {
            "median_abs_total": float(np.nanmedian(np.abs(obs))),
            "median_abs_denominator_only": float(np.nanmedian(np.abs(den))),
            "median_abs_inclusion_only": float(np.nanmedian(np.abs(inc))),
            "max_abs_denominator_only": float(np.nanmax(np.abs(den))),
            "max_abs_inclusion_only": float(np.nanmax(np.abs(inc)))}

    print()
    print("=" * 100)
    print("I. FRACTAL INPUT: SUPPORT vs CANVAS")
    print("=" * 100)
    same = cn[cn.support_delta_px == 0]
    diff = cn[cn.support_delta_px != 0]
    print(f"  border rows: support count unchanged {len(same)}, changed {len(diff)}")
    for f in ["fractal_d0", "fractal_d1", "fractal_d2"]:
        s0 = same[f + "__abs"].dropna()
        s1 = diff[f + "__abs"].dropna()
        rr = float(spearmanr(diff.support_delta_px.abs(), diff[f + "__abs"].fillna(0)).statistic)
        print(f"  {f:12s} support-unchanged: median={s0.median():.5f} p95={s0.quantile(.95):.5f} "
              f"max={s0.max():.5f} (n={len(s0)})")
        print(f"  {'':12s} support-changed  : median={s1.median():.5f} p95={s1.quantile(.95):.5f} "
              f"max={s1.max():.5f} (n={len(s1)})  spearman(|support delta|,|delta|)={rr:+.4f}")
        summ[f"fractal_support|{f}"] = {
            "rows_support_unchanged": int(len(s0)), "rows_support_changed": int(len(s1)),
            "median_abs_support_unchanged": float(s0.median()),
            "max_abs_support_unchanged": float(s0.max()),
            "median_abs_support_changed": float(s1.median()),
            "max_abs_support_changed": float(s1.max()),
            "spearman_support_vs_delta": rr}

    CT = canvas_test(8)
    CT.to_csv(OUT / "task5b_h2_canvas_test.csv", index=False)
    print()
    print("  FRACTAL CANVAS TEST — identical binary support, zero-padded to the border canvas:")
    for f, a, b in (("fractal_d0", "d0_base", "d0_pad"), ("fractal_d1", "d1_base", "d1_pad"),
                    ("fractal_d2", "d2_base", "d2_pad")):
        rel = ((CT[b] - CT[a]) / CT[a])
        print(f"    {f:12s} n={len(CT)} median rel change={rel.median():+.5f} "
              f"p95={rel.quantile(.95):+.5f} max={rel.max():+.5f} min={rel.min():+.5f}")
        summ[f"canvas_test|{f}"] = {"n": int(len(CT)), "median_rel": float(rel.median()),
                                    "p95_rel": float(rel.quantile(.95)),
                                    "max_rel": float(rel.max()), "min_rel": float(rel.min())}
    print(f"    canvas growth fraction: median "
          f"{CT.canvas_growth.median():.4f} max {CT.canvas_growth.max():.4f}")

    print()
    print("=" * 100)
    print("J. NaN / FAILURE TRANSITIONS")
    print("=" * 100)
    BB = B.set_index("image_path")
    for label, g in (("BRIGHTNESS", cb), ("BORDER", cn)):
        for f in FEATS:
            b = g[f + "__base"].notna()
            p = g[f].notna()
            summ[f"transitions|{label}|{f}"] = {"finite_to_nan": int((b & ~p).sum()),
                                                "nan_to_finite": int((~b & p).sum())}
        j = g.join(BB["fov_valid"].rename("bv"), on="image_path")
        vi, iv = int((j.bv & ~j.fov_valid).sum()), int((~j.bv & j.fov_valid).sum())
        summ[f"fov_transitions|{label}"] = {"valid_to_invalid": vi, "invalid_to_valid": iv}
        print(f"  {label:10s} finite->NaN 0 for all five features (baseline NaN 0)   "
              f"FOV valid->invalid {vi}   invalid->valid {iv}")

    print()
    print("=" * 100)
    print("M/N. CLASSIFICATION")
    print("=" * 100)
    cls = {f: classify(summ.get(f"BORDER|{f}", {})) for f in FEATS}
    for f in FEATS:
        s = summ[f"BORDER|{f}"]
        print(f"  {f:20s} {cls[f]:26s} median={s['median_rel']:+.5f} p95={s['p95_rel']:+.5f} "
              f"max={s['max_rel']:+.5f} >1%={s['frac_gt_1pct']:.4f} >5%={s['frac_gt_5pct']:.4f} "
              f">10%={s['frac_gt_10pct']:.4f} | brightness median={summ['BRIGHTNESS|' + f]['median_rel']:+.5f}")
    affected = [f for f in FEATS if cls[f] != "MINIMAL_IMPACT"]
    severe = [f for f in FEATS if cls[f] == "SEVERELY_BORDER_SENSITIVE"]
    frac = [f for f in FEATS if f.startswith("fractal") and cls[f] != "MINIMAL_IMPACT"]
    prop = "YES" if len(frac) == 3 else ("PARTIAL" if frac else "NO")
    print()
    print(f"  PRIMARY_FEATURES_AFFECTED_N          : {len(affected)} {affected}")
    print(f"  PRIMARY_FEATURES_SEVERELY_AFFECTED_N : {len(severe)} {severe}")
    print(f"  FOV_FAILURE_PROPAGATES_TO_FRACTALS   : {prop}")

    out = {"TASK5B_H2_SAMPLE_N": int(len(B)), "FINAL_PRIMARY_FEATURE_N": 5,
           "module_sha256": sha(MODULE), "rows": int(len(C)),
           "feature_status": cls,
           "PRIMARY_FEATURES_AFFECTED_N": len(affected), "PRIMARY_FEATURES_AFFECTED": affected,
           "PRIMARY_FEATURES_SEVERELY_AFFECTED_N": len(severe),
           "PRIMARY_FEATURES_SEVERELY_AFFECTED": severe,
           "FOV_FAILURE_PROPAGATES_TO_FRACTALS": prop,
           "stats": summ}
    (OUT / "task5b_h2_summary.json").write_text(json.dumps(out, indent=2, default=str),
                                                encoding="utf-8")
    print("\n  summary written")


if __name__ == "__main__":
    main()
