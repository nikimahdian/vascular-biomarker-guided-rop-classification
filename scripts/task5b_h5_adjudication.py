#!/usr/bin/env python
"""Task 5B-H5: residual failure adjudication for the Task 5B-H4 locked evaluation.

Analysis only. No new FOV candidate, no V3 change, no re-run of the 450 x 42 battery, no table
regeneration, no classifier.

  A  semantic class of every H4 condition, decided from its TRANSFORMATION DEFINITION only
  B  objective pixel-mapping verification: can the original image be recovered exactly from the
     mapped content region, and how many original pixels are modified or lost
  C  V3 metrics recomputed separately for CONTENT_PRESERVING and CONTENT_ALTERING conditions,
     from `task5b_h4_rows.csv` only
  D  every CONTENT_PRESERVING case with Dice < 0.95, valid->invalid, or density |rel| > 5 %,
     with the mechanism probed from the transform plus the frozen V3 content-box detection
  E  the two extreme padding conditions adjudicated by pixel mapping, not by performance
  F  fractal canvas status read from the existing H4 control

Pixel checks load images and call `content_bounds`; they perform NO feature measurement.
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

from scripts.task5b_n2 import load  # noqa: E402
from scripts.task5b_h_fov_stress import (  # noqa: E402
    BRIGHTNESS, apply_border, apply_brightness, pad_specs,
)
from scripts.task5b_h3_eval import novel_specs  # noqa: E402
from scripts.task5b_h4_eval import challenge_specs  # noqa: E402
from src.biomarker import clinical_measurement_v3_candidate as v3  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]


def specs_for(h, w, bg):
    return pad_specs(h, w) + novel_specs(h, w) + challenge_specs(h, w, bg)


def semantic_table():
    """A: class from the transformation definition. Nothing here reads a Dice or a feature value."""
    rows = [{"condition": f"x{c:.2f}", "family": "brightness",
             "semantic_class": "CONTENT_PRESERVING",
             "reason": "global multiplicative photometric change, np.clip(rgb*c, 0, 255); no pixel is "
                       "moved, overwritten or cropped",
             "original_pixels_preserved_exactly": "yes_in_value_model"
             if False else "yes_except_clipping"}
            for c in BRIGHTNESS]
    for s in pad_specs(384, 512):
        fam, name, pad, val, alters = s
        rows.append({"condition": name, "family": fam,
                     "semantic_class": "CONTENT_ALTERING" if alters else "CONTENT_PRESERVING",
                     "reason": ("paints a constant band OVER existing pixels, canvas unchanged"
                                if alters else
                                "np.pad grows the canvas with a constant value; every original pixel "
                                "keeps its position and value"),
                     "original_pixels_preserved_exactly": "no" if alters else "yes"})
    for s in novel_specs(384, 512):
        rows.append({"condition": s[1], "family": s[0], "semantic_class": "CONTENT_PRESERVING",
                     "reason": "np.pad canvas growth, constant value; original pixels untouched",
                     "original_pixels_preserved_exactly": "yes"})
    for s in challenge_specs(384, 512, 100):
        rows.append({"condition": s[1], "family": s[0], "semantic_class": "CONTENT_PRESERVING",
                     "reason": "np.pad canvas growth, constant value placed relative to the image's "
                               "own background median; original pixels untouched",
                     "original_pixels_preserved_exactly": "yes"})
    return pd.DataFrame(rows)


def pixel_probe(arg):
    """B: objective recovery test for every border condition on one image."""
    image_path, mask_path = arg
    rgb, msk = load(image_path, mask_path)
    h0, w0 = msk.shape
    g = np.nan_to_num(np.asarray(rgb[:, :, 1], np.float32), nan=0.0)
    top, bottom, left, right, trimmed = v3.content_bounds(g)
    bg = float(np.median(g[top:h0 - bottom, left:w0 - right])) if trimmed else float(np.median(g))
    out = []
    for s in specs_for(h0, w0, bg):
        prgb, pmsk, pad = apply_border(rgb, msk, s)
        t, b, l, r = pad
        H, W = prgb.shape[:2]
        if s[4]:          # overwrite: canvas unchanged, original pixels painted over
            modified = int(np.any(np.abs(prgb - rgb) > 1e-9, axis=2).sum())
            lost = modified
            recovered = prgb
            exact = False
        else:
            recovered = prgb[t:t + h0, l:l + w0]
            modified = int(np.any(np.abs(recovered - rgb) > 1e-9, axis=2).sum())
            lost = int(h0 * w0) - int(recovered.shape[0] * recovered.shape[1])
            exact = bool(np.array_equal(recovered, rgb))
        out.append({"image_path": image_path, "condition": s[1], "family": s[0],
                    "alters_content": bool(s[4]), "pad": f"{t},{b},{l},{r}",
                    "canvas_grew": (H - h0, W - w0), "exact_recovery": exact,
                    "modified_original_px": modified, "lost_original_px": lost,
                    "content_box": f"{top},{bottom},{left},{right}",
                    "detected_trim": bool(trimmed)})
    return out


def mechanism_probe(arg):
    """D/E: for one (image, condition), compare detected trimming with the actual added padding."""
    image_path, mask_path, cond = arg
    rgb, msk = load(image_path, mask_path)
    h0, w0 = msk.shape
    g = np.nan_to_num(np.asarray(rgb[:, :, 1], np.float32), nan=0.0)
    top, bottom, left, right, trimmed = v3.content_bounds(g)
    bg = float(np.median(g[top:h0 - bottom, left:w0 - right])) if trimmed else float(np.median(g))
    spec = [s for s in specs_for(h0, w0, bg) if s[1] == cond]
    if spec:
        prgb, pmsk, pad = apply_border(rgb, msk, spec[0])
        transform = "border_padding" if not spec[0][4] else "content_overwrite"
    else:
        prgb, pmsk, pad = apply_brightness(rgb, msk, float(cond[1:]))
        transform = "global_photometric"
    t, b, l, r = pad
    H, W = prgb.shape[:2]
    gp = np.nan_to_num(np.asarray(prgb[:, :, 1], np.float32), nan=0.0)
    T, B, L, R, ptrimmed = v3.content_bounds(gp)
    over = {"top": max(0, T - t), "bottom": max(0, B - b), "left": max(0, L - l),
            "right": max(0, R - r)}
    # original-content pixels excluded from the content box beyond the added padding
    orig_px_trimmed = int(over["top"] * W + over["bottom"] * W + over["left"] * h0
                          + over["right"] * h0)
    base_trimmed = int(top * w0 + bottom * w0 + left * h0 + right * h0)
    m_base = max(64, int(0.001 * h0 * w0))
    m_pad = max(64, int(0.001 * H * W))
    try:
        thr_base = v3.v2_threshold(np.nan_to_num(np.asarray(rgb[:, :, 1], np.float32),
                                                 nan=0.0))[0]
        thr_pad = v3.v2_threshold(gp)[0]
    except Exception:  # noqa: BLE001
        thr_base = thr_pad = float("nan")
    return {"image_path": image_path, "condition": cond, "transform": transform,
            "pad": f"{t},{b},{l},{r}",
            "min_size_base": m_base, "min_size_padded": m_pad,
            "min_size_ratio": float(m_pad / m_base),
            "thr_base": float(thr_base), "thr_padded": float(thr_pad),
            "thr_shift": float(thr_pad - thr_base),
            "pad_px": int(t * W + b * W + l * h0 + r * h0),
            "detected_trim_tblr": f"{T},{B},{L},{R}", "detected_trimmed": bool(ptrimmed),
            "over_trim_tblr": f"{over['top']},{over['bottom']},{over['left']},{over['right']}",
            "original_px_trimmed_beyond_pad": orig_px_trimmed,
            "baseline_native_trim_px": base_trimmed,
            "post_pad_frame_px": int(H * W),
            "canvas_grew": (H - h0, W - w0)}


def stats(g, col):
    r = g[col].replace([np.inf, -np.inf], np.nan).dropna()
    if not len(r):
        return {}
    return {"n": int(len(r)), "median_rel": float(r.median()), "p95_rel": float(r.quantile(.95)),
            "max_rel": float(r.max()),
            "frac_gt_1pct": float((r.abs() > .01).mean()),
            "frac_gt_5pct": float((r.abs() > .05).mean()),
            "frac_gt_10pct": float((r.abs() > .10).mean())}


def main() -> None:
    print("=" * 100)
    print("A. SEMANTIC CLASS FROM THE TRANSFORMATION DEFINITION")
    print("=" * 100)
    S = semantic_table()
    print(f"  conditions total {len(S)}   CONTENT_PRESERVING "
          f"{int((S.semantic_class == 'CONTENT_PRESERVING').sum())}   CONTENT_ALTERING "
          f"{int((S.semantic_class == 'CONTENT_ALTERING').sum())}")
    print(S[S.semantic_class == "CONTENT_ALTERING"][
        ["condition", "semantic_class", "reason"]].to_string(index=False))
    S.to_csv(OUT / "task5b_h5_semantics.csv", index=False)

    C = pd.read_csv(OUT / "task5b_h4_rows.csv", low_memory=False)
    B = pd.read_csv(OUT / "task5b_h4_baseline.csv")
    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                       usecols=["image_path", "mask_path"])
    mp = dict(zip(meta.image_path, meta.mask_path))
    C["semantic_class"] = np.where(C.condition.isin(
        S[S.semantic_class == "CONTENT_ALTERING"].condition), "CONTENT_ALTERING",
        "CONTENT_PRESERVING")
    C["is_brightness"] = C.family == "brightness"

    print()
    print("=" * 100)
    print("B. PIXEL-MAPPING VERIFICATION (objective, from the transformation code)")
    print("=" * 100)
    probe_imgs = B.sort_values("image_path").head(12)
    args = [(r.image_path, mp[r.image_path]) for r in probe_imgs.itertuples()]
    with get_context("fork").Pool(8) as p:
        pr = [x for sub in p.map(pixel_probe, args) for x in sub]
    P = pd.DataFrame(pr)
    P.to_csv(OUT / "task5b_h5_pixel_map.csv", index=False)
    agg = P.groupby("condition").agg(exact=("exact_recovery", "all"),
                                     mod=("modified_original_px", "max"),
                                     lost=("lost_original_px", "max"),
                                     alters=("alters_content", "first"),
                                     grew=("canvas_grew", "first")).reset_index()
    bad = agg[(~agg.exact) & (~agg.alters)]
    print(f"  verified on {len(probe_imgs)} images x {agg.shape[0]} border conditions "
          f"= {len(P)} checks")
    print(f"  conditions with EXACT recovery of every original pixel : "
          f"{int((agg.exact | agg.alters.eq(False) & agg.exact).sum())} of {len(agg)}")
    print(f"  padding conditions failing exact recovery : {len(bad)}")
    print(f"  CONTENT_ALTERING conditions (expected to fail) : "
          f"{agg[agg.alters].condition.tolist()}")
    print(agg.to_string(index=False))
    S2 = S.merge(agg[["condition", "exact", "mod", "lost"]], on="condition", how="left")
    S2["original_pixels_preserved_exactly"] = np.where(
        S2.semantic_class == "CONTENT_ALTERING", "no",
        np.where(S2.exact.fillna(False), "yes", "NO_VERIFY_FAILED"))
    S2.to_csv(OUT / "task5b_h5_semantics.csv", index=False)

    print()
    print("=" * 100)
    print("C. V3 METRICS SPLIT BY SEMANTIC CLASS (from task5b_h4_rows.csv only)")
    print("=" * 100)
    res = {}
    for cls in ("CONTENT_PRESERVING", "CONTENT_ALTERING"):
        g = C[C.semantic_class == cls]
        d = g.v3_dice.dropna()
        base = B.set_index("image_path")["v3_valid"]
        j = g.join(base.rename("bv"), on="image_path")
        vi = int((j.bv & ~j.v3_valid).sum())
        iv = int((~j.bv & j.v3_valid).sum())
        res[f"{cls}_conditions_n"] = int(g.condition.nunique())
        res[f"{cls}_pairs_n"] = int(len(g))
        res[f"{cls}_median_dice"] = float(d.median())
        res[f"{cls}_p01_dice"] = float(d.quantile(.01))
        res[f"{cls}_min_dice"] = float(d.min())
        res[f"{cls}_valid_to_invalid"] = vi
        res[f"{cls}_invalid_to_valid"] = iv
        print(f"  --- {cls} ---  conditions {g.condition.nunique()}  pairs {len(g)}")
        print(f"    median Dice {d.median():.6f}  p01 {d.quantile(.01):.6f}  min {d.min():.6f}"
              f"  <0.99 {float((d < 0.99).mean()):.4f}  <0.90 {float((d < 0.90).mean()):.4f}")
        print(f"    valid->invalid {vi}   invalid->valid {iv}")
        for f in FEATS:
            s = stats(g, f"v3_{f}_rel")
            res[f"{cls}|{f}"] = s
            if s:
                print(f"    {f:20s} median {s['median_rel']:+.5f}  p95 {s['p95_rel']:+.5f}  "
                      f"max {s['max_rel']:+.5f}  >1% {s['frac_gt_1pct']:.4f}  "
                      f">5% {s['frac_gt_5pct']:.4f}  >10% {s['frac_gt_10pct']:.4f}")
        nan_tot = 0
        for f in FEATS:
            b = g[f"v3_{f}_base"].notna()
            p_ = g[f"v3_{f}"].notna()
            nan_tot += int((b & ~p_).sum())
        res[f"{cls}_finite_to_nan"] = nan_tot
        print(f"    finite->NaN over the five features: {nan_tot}")
        if cls == "CONTENT_ALTERING":
            print("    NOTE: robustness-to-occlusion / content-change test, NOT an invariance test.")

    print()
    print("=" * 100)
    print("D/E. RESIDUAL TRUE FAILURES UNDER CONTENT-PRESERVING CONDITIONS")
    print("=" * 100)
    CP = C[C.semantic_class == "CONTENT_PRESERVING"].copy()
    CP = CP.join(B.set_index("image_path")["v3_valid"].rename("bv"), on="image_path")
    dens = CP[["v3_vessel_density_fov_rel", "v3_skel_density_fov_rel"]].abs().max(axis=1)
    CP["density_abs_rel_max"] = dens
    CP["is_valid_to_invalid"] = CP.bv & ~CP.v3_valid
    sel = CP[(CP.v3_dice < 0.95) | CP.is_valid_to_invalid | (CP.density_abs_rel_max > 0.05)]
    print(f"  content-preserving pairs {len(CP)}")
    print(f"  Dice < 0.95 : {int((CP.v3_dice < 0.95).sum())}")
    print(f"  valid->invalid : {int(CP.is_valid_to_invalid.sum())}")
    print(f"  density |rel| > 5 % : {int((CP.density_abs_rel_max > 0.05).sum())}")
    print(f"  union (any of the three) : {len(sel)}  = {len(sel)/len(CP):.4f} of pairs")
    res["content_preserving_residual_rows"] = int(len(sel))
    print("  by condition:")
    print(sel.condition.value_counts().to_string())
    print("  by source:", sel.source.value_counts().to_dict())
    print("  by geometry:", sel.geom.value_counts().to_dict())
    print("  by rule triggered:")
    print(f"    Dice<0.95 only {int(((sel.v3_dice < 0.95) & ~sel.is_valid_to_invalid & (sel.density_abs_rel_max <= 0.05)).sum())}"
          f"   valid->invalid {int(sel.is_valid_to_invalid.sum())}"
          f"   density>5% only {int(((sel.v3_dice >= 0.95) & ~sel.is_valid_to_invalid & (sel.density_abs_rel_max > 0.05)).sum())}")

    samp = sel.sort_values("v3_dice").groupby("condition", group_keys=False).head(6).head(90)
    margs = [(r.image_path, mp[r.image_path], r.condition) for r in samp.itertuples()]
    with get_context("fork").Pool(8) as p:
        mr = p.map(mechanism_probe, margs)
    M = pd.DataFrame(mr)
    M.to_csv(OUT / "task5b_h5_mechanism.csv", index=False)
    print()
    print(f"  mechanism probe on {len(M)} of the {len(sel)} residual rows:")
    print(f"    rows where V3 trimmed ORIGINAL content beyond the added padding : "
          f"{int((M.original_px_trimmed_beyond_pad > 0).sum())} of {len(M)}")
    print(f"    over-trim area median {M[M.original_px_trimmed_beyond_pad > 0].original_px_trimmed_beyond_pad.median() if (M.original_px_trimmed_beyond_pad > 0).any() else 0}")
    print(f"    rows where the detected box equals the added padding exactly : "
          f"{int((M.original_px_trimmed_beyond_pad == 0).sum())}")
    res["residual_rows_probed"] = int(len(M))
    res["residual_rows_over_trimmed"] = int((M.original_px_trimmed_beyond_pad > 0).sum())
    print(f"    rows where min_size (0.001*h*w) changed with the canvas : "
          f"{int((M.min_size_ratio > 1.0).sum())} of {len(M)}   "
          f"median ratio {M.min_size_ratio.median():.2f}  max {M.min_size_ratio.max():.2f}")
    print(f"    threshold shift median {M.thr_shift.median():+.3f}  "
          f"max |shift| {M.thr_shift.abs().max():.3f}")
    res["residual_rows_min_size_changed"] = int((M.min_size_ratio > 1.0).sum())
    res["residual_min_size_ratio_median"] = float(M.min_size_ratio.median())
    res["residual_min_size_ratio_max"] = float(M.min_size_ratio.max())
    jo = M.merge(sel[["image_path", "condition", "v3_dice", "v3_reason", "density_abs_rel_max",
                      "source", "geom", "v3_trimmed", "v3_coverage"]],
                 on=["image_path", "condition"], how="left")
    print("  sample rows:")
    for r in jo.sort_values("v3_dice").head(12).itertuples():
        rs = str(getattr(r, "v3_reason"))[:26]
        print(f"    {str(r.condition):22s} {str(r.source):11s} {str(r.geom):10s} "
              f"dice={float(r.v3_dice):.4f} densmax={float(r.density_abs_rel_max):.4f} "
              f"reason={rs:26s} pad={str(r.pad):14s} trim={str(r.detected_trim_tblr):14s} "
              f"overTrim={int(r.original_px_trimmed_beyond_pad):7d} "
              f"minSizeRatio={float(r.min_size_ratio):.2f} thrShift={float(r.thr_shift):+.3f}")

    print()
    print("  E. the two extreme padding conditions")
    for cond in ("irregular_frame_w3", "tb_thick_w3"):
        sub = P[P.condition == cond]
        print(f"    {cond}: exact_recovery={bool(sub.exact_recovery.all())} "
              f"modified_original_px max={int(sub.modified_original_px.max())} "
              f"lost_original_px max={int(sub.lost_original_px.max())} "
              f"canvas_grew={sub.canvas_grew.iloc[0]}  -> CONTENT_PRESERVING")
        g = CP[CP.condition == cond]
        print(f"      V3: median Dice {g.v3_dice.median():.6f}  min {g.v3_dice.min():.6f}  "
              f"invalid {int((~g.v3_valid).sum())}  density>5% "
              f"{float((g.density_abs_rel_max > .05).mean()):.4f}")
        res[f"extreme|{cond}"] = {"exact_recovery": bool(sub.exact_recovery.all()),
                                  "median_dice": float(g.v3_dice.median()),
                                  "min_dice": float(g.v3_dice.min()),
                                  "invalid": int((~g.v3_valid).sum()),
                                  "density_gt5pct": float((g.density_abs_rel_max > .05).mean())}

    print()
    print("=" * 100)
    print("F. FRACTAL CANVAS STATUS (from the existing H4 control)")
    print("=" * 100)
    ct = pd.read_csv(OUT / "task5b_h4_canvas_control.csv")
    fr = {}
    for f in ("fractal_d0", "fractal_d1", "fractal_d2"):
        r3 = ct[f"v3_{f}_rel"].replace([np.inf, -np.inf], np.nan).dropna()
        r1 = ct[f"v1_{f}_rel"].replace([np.inf, -np.inf], np.nan).dropna()
        ok = float(r3.abs().median()) <= 0.005
        fr[f] = {"v1_median_abs_rel": float(r1.abs().median()),
                 "v3_median_abs_rel": float(r3.abs().median()),
                 "v3_max_abs_rel": float(r3.abs().max()), "resolved": bool(ok)}
        print(f"  {f:12s} V1 median|rel| {r1.abs().median():.5f} -> V3 "
              f"{r3.abs().median():.5f} (max {r3.abs().max():.5f})  RESOLVED={ok}")
    res["fractal_canvas"] = fr

    print()
    print("=" * 100)
    print("G. FINAL OUTPUT")
    print("=" * 100)
    cpm = res["CONTENT_PRESERVING|vessel_density_fov"]
    csm = res["CONTENT_PRESERVING|skel_density_fov"]
    out = {
        "TASK5B_H5_CONTENT_PRESERVING_CONDITIONS_N": res["CONTENT_PRESERVING_conditions_n"],
        "TASK5B_H5_CONTENT_ALTERING_CONDITIONS_N": res["CONTENT_ALTERING_conditions_n"],
        "TASK5B_H5_CONTENT_PRESERVING_PAIRS_N": res["CONTENT_PRESERVING_pairs_n"],
        "V3_CONTENT_PRESERVING_MEDIAN_DICE": res["CONTENT_PRESERVING_median_dice"],
        "V3_CONTENT_PRESERVING_P01_DICE": res["CONTENT_PRESERVING_p01_dice"],
        "V3_CONTENT_PRESERVING_MIN_DICE": res["CONTENT_PRESERVING_min_dice"],
        "V3_CONTENT_PRESERVING_VALID_TO_INVALID_N": res["CONTENT_PRESERVING_valid_to_invalid"],
        "VESSEL_DENSITY_CONTENT_PRESERVING_GT1PCT": cpm["frac_gt_1pct"],
        "VESSEL_DENSITY_CONTENT_PRESERVING_GT5PCT": cpm["frac_gt_5pct"],
        "SKEL_DENSITY_CONTENT_PRESERVING_GT1PCT": csm["frac_gt_1pct"],
        "SKEL_DENSITY_CONTENT_PRESERVING_GT5PCT": csm["frac_gt_5pct"],
        "TRUE_RESIDUAL_FOV_FAILURES_N": res["content_preserving_residual_rows"],
        "results": res,
    }
    for k in ("TASK5B_H5_CONTENT_PRESERVING_CONDITIONS_N",
              "TASK5B_H5_CONTENT_ALTERING_CONDITIONS_N", "TASK5B_H5_CONTENT_PRESERVING_PAIRS_N",
              "V3_CONTENT_PRESERVING_MEDIAN_DICE", "V3_CONTENT_PRESERVING_P01_DICE",
              "V3_CONTENT_PRESERVING_MIN_DICE", "V3_CONTENT_PRESERVING_VALID_TO_INVALID_N",
              "VESSEL_DENSITY_CONTENT_PRESERVING_GT1PCT", "VESSEL_DENSITY_CONTENT_PRESERVING_GT5PCT",
              "SKEL_DENSITY_CONTENT_PRESERVING_GT1PCT", "SKEL_DENSITY_CONTENT_PRESERVING_GT5PCT",
              "TRUE_RESIDUAL_FOV_FAILURES_N"):
        print(f"  {k:48s} : {out[k]}")
    (OUT / "task5b_h5_summary.json").write_text(json.dumps(out, indent=2, default=str),
                                                encoding="utf-8")
    print("  summary written: _private_audit/task5b_h5_summary.json")


if __name__ == "__main__":
    main()
