#!/usr/bin/env python
"""Task 5B-H6: unit invariance tests, development check, and locked evaluation of
CLINICAL_MEASUREMENT_V4_CANDIDATE (content-relative FOV constants).

Order: freeze -> manifest -> D1/D2 unit tests (STOP if they fail) -> E development check on the
known H5 residual cases -> F locked sample N=150 disjoint from every previous sample -> G targeted
condition set -> H/I/J endpoints -> L gate -> N report.

No classifier. No SEG_CURRENT_V1 change. No fractal change. No regeneration of the 8,870-row table.
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
from src.biomarker import clinical_measurement_v3_candidate as v3  # noqa: E402
from src.biomarker import clinical_measurement_v4_candidate as v4  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
# PREDECLARED targeted condition set (G). Names are existing condition names.
CONDS = ["irregular_frame_w3", "tb_thick_w3", "novel_black_w5", "asymmetric_black_w3",
         "dark_gray_40_w2", "novel_gray90_w2", "uniform_black_w2"]
NATIVE = "NATIVE"
GATE = {
    "G1_fallback_residual_max": 0,
    "G2_canvas_minsize_residual_max": 0,
    "G3_median_dice_min": 0.999,
    "G4_p01_dice_min": 0.95,
    "G5_valid_to_invalid_max": 2,
    "G6_vessel_density_frac_gt_5pct_max": 0.05,
    "G7_skel_density_frac_gt_5pct_max": 0.05,
    "G10_canvas_median_abs_rel_max": 0.005,
}


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2.0 * int(np.logical_and(a, b).sum()) / s) if s else float("nan")


def disp(a, b, h0, w0):
    if not (a.sum() and b.sum()):
        return np.nan
    ay, ax = ndi.center_of_mass(a)
    by, bx = ndi.center_of_mass(b)
    return float(np.hypot(ax - bx, ay - by) / max(h0, w0))


# ---------------------------------------------------------------- D1/D2 unit tests
def unit_tests():
    print("=" * 100)
    print("D. UNIT / INVARIANCE TESTS (before any image-level evaluation)")
    print("=" * 100)
    rng = np.random.default_rng(20240921)
    base = np.zeros((64, 80), np.float32)
    yy, xx = np.mgrid[0:64, 0:80]
    base[:] = 30 + 120 * (((xx - 40) ** 2 + (yy - 32) ** 2) < 700)
    base += rng.normal(0, 3, base.shape).astype(np.float32)
    base = np.clip(base, 0, 255)
    rows = []
    for pad in (0, 5, 20, 60, 140, 300):
        g = np.pad(base, pad, constant_values=90) if pad else base.copy()
        v3_thr, v3_trimmed = v3.v2_threshold(g)
        v4_thr, bounds = v4.v4_threshold(g)
        content, _ = v4.v4_content(g)
        top, bottom, left, right, has = bounds
        ch, cw = content.shape
        msz = max(64, int(0.001 * ch * cw))
        msz_v3 = max(64, int(0.001 * g.shape[0] * g.shape[1]))
        rows.append({"pad": pad, "canvas": f"{g.shape[0]}x{g.shape[1]}",
                     "content_identical": bool(np.array_equal(content, base)),
                     "v3_threshold": v3_thr, "v4_threshold": v4_thr,
                     "v3_trimmed": bool(v3_trimmed), "v4_has_content": bool(has),
                     "v3_min_size": msz_v3, "v4_min_size": msz})
    U = pd.DataFrame(rows)
    print(U.to_string(index=False))
    t_ok = bool(U.content_identical.all()) and bool(U.v4_threshold.nunique() == 1) \
        and bool(U.v4_min_size.nunique() == 1)
    c_ok = bool(U.canvas.nunique() > 1) and bool(U.content_identical.all())
    print()
    print(f"  D1 content pixels identical across all pads : {bool(U.content_identical.all())}")
    print(f"  D1 V4 Otsu threshold identical              : {bool(U.v4_threshold.nunique() == 1)}"
          f"   value {U.v4_threshold.iloc[0]:.6f}")
    print(f"  D1 V4 min_size identical                    : {bool(U.v4_min_size.nunique() == 1)}"
          f"   value {int(U.v4_min_size.iloc[0])}")
    print(f"  (V3 by contrast: trimmed flags {U.v3_trimmed.tolist()}, "
          f"min_size {U.v3_min_size.tolist()})")
    print(f"  D2 full-frame dimensions change while both quantities stay constant : {c_ok}")
    print(f"  UNIT_CONTENT_THRESHOLD_INVARIANCE : {'PASS' if t_ok else 'FAIL'}")
    print(f"  UNIT_CONTENT_MINSIZE_INVARIANCE   : {'PASS' if bool(U.v4_min_size.nunique() == 1) else 'FAIL'}")
    U.to_csv(OUT / "task5b_h6_unit_tests.csv", index=False)
    return t_ok, bool(U.v4_min_size.nunique() == 1), c_ok


# ---------------------------------------------------------------- E development check
def dev_probe(arg):
    image_path, mask_path = arg
    rgb, msk = load(image_path, mask_path)
    h0, w0 = msk.shape
    g = np.nan_to_num(np.asarray(rgb[:, :, 1], np.float32), nan=0.0)
    rows = []
    for cond in CONDS:
        spec = [s for s in (pad_specs(h0, w0) + novel_specs(h0, w0)) if s[1] == cond]
        if not spec:
            continue
        prgb, pmsk, pad = apply_border(rgb, msk, spec[0])
        gp = np.nan_to_num(np.asarray(prgb[:, :, 1], np.float32), nan=0.0)
        thr3, b3 = v3.v2_threshold(gp)
        b3 = v3.content_bounds(gp)
        thr4, b4 = v4.v4_threshold(gp)
        content, _ = v4.v4_content(gp)
        ch, cw = content.shape
        H, W = gp.shape
        rows.append({"image_path": image_path, "condition": cond,
                     "v3_threshold": float(thr3), "v4_threshold": float(thr4),
                     "v3_fell_back": not bool(b3[4]),
                     "v3_min_size": max(64, int(0.001 * H * W)),
                     "v4_min_size": max(64, int(0.001 * ch * cw)),
                     "canvas_min_size_changed": bool(max(64, int(0.001 * H * W))
                                                     != max(64, int(0.001 * ch * cw))),
                     "content_area_frac_v3_style": float(ch * cw / (H * W))})
    return rows


def stats(g, col):
    r = g[col].replace([np.inf, -np.inf], np.nan).dropna()
    if not len(r):
        return {}
    return {"n": int(len(r)), "median_rel": float(r.median()), "p95_rel": float(r.quantile(.95)),
            "max_rel": float(r.max()),
            "frac_gt_1pct": float((r.abs() > .01).mean()),
            "frac_gt_5pct": float((r.abs() > .05).mean())}


def worker(arg):
    image_path, mask_path, source, split, geom = arg[:5]
    try:
        rgb, msk = load(image_path, mask_path)
        h0, w0 = msk.shape
        o3b, f3b, q3b = v3.measure_v3(rgb, msk, {})
        o4b, f4b, q4b = v4.measure_v4(rgb, msk, {})
        rows = []
        base = {"image_path": image_path, "source": source, "split": split, "geom": geom,
                "condition": NATIVE,
                "v3_fov_px": int(f3b.sum()), "v4_fov_px": int(f4b.sum()),
                "v3_valid": bool(q3b.get("fov_valid")), "v4_valid": bool(q4b.get("fov_valid")),
                "v3_dice": 1.0, "v4_dice": 1.0,
                "v3_rel_area": 0.0, "v4_rel_area": 0.0,
                "v3_centroid": 0.0, "v4_centroid": 0.0,
                "drift_dice_v4_v3": dice(f4b, f3b),
                "v3_threshold": float(q3b.get("fov_threshold", np.nan))
                if "fov_threshold" in q3b else np.nan,
                "v4_threshold": float(q4b.get("fov_threshold", np.nan)),
                "v3_min_size": q3b.get("fov_min_size", np.nan),
                "v4_min_size": q4b.get("fov_min_size", np.nan),
                "v3_fell_back": False, "canvas_min_size_changed": False,
                "v4_in_padding_px": 0, "v4_central_retained": True, "v4_reason": ""}
        for tag, o in (("v3", o3b), ("v4", o4b)):
            for f in FEATS:
                base[f"{tag}_{f}"] = o.get(f, np.nan)
        rows.append(base)
        for cond in CONDS:
            spec = [s for s in (pad_specs(h0, w0) + novel_specs(h0, w0)) if s[1] == cond]
            if not spec:
                continue
            prgb, pmsk, pad = apply_border(rgb, msk, spec[0])
            t, b_, l, r_ = pad
            H, W = pmsk.shape
            gp = np.nan_to_num(np.asarray(prgb[:, :, 1], np.float32), nan=0.0)
            thr3_chk, _v3_trim = v3.v2_threshold(gp)
            b3 = v3.content_bounds(gp)
            content, _ = v4.v4_content(gp)
            ch, cw = content.shape
            o3, pf3, q3 = v3.measure_v3(prgb, pmsk, {})
            o4, pf4, q4 = v4.measure_v4(prgb, pmsk, {})
            c3, c4 = pf3[t:t + h0, l:l + w0], pf4[t:t + h0, l:l + w0]
            row = {"image_path": image_path, "source": source, "split": split, "geom": geom,
                   "condition": cond,
                   "v3_dice": dice(c3, f3b), "v4_dice": dice(c4, f4b),
                   "v3_rel_area": float((int(c3.sum()) - int(f3b.sum())) / max(int(f3b.sum()), 1)),
                   "v4_rel_area": float((int(c4.sum()) - int(f4b.sum())) / max(int(f4b.sum()), 1)),
                   "v3_centroid": disp(c3, f3b, h0, w0), "v4_centroid": disp(c4, f4b, h0, w0),
                   "v3_valid": bool(q3.get("fov_valid")), "v4_valid": bool(q4.get("fov_valid")),
                   "v3_threshold": float(thr3_chk),
                   "v4_threshold": float(q4.get("fov_threshold", np.nan)),
                   "v3_min_size": max(64, int(0.001 * H * W)),
                   "v4_min_size": q4.get("fov_min_size", np.nan),
                   "v3_fell_back": bool(not b3[4]),
                   "canvas_min_size_changed": bool(max(64, int(0.001 * H * W))
                                                    != max(64, int(0.001 * ch * cw))),
                   "v4_in_padding_px": int((pf4 & v4.external_region((H, W), b3)).sum())
                   if b3[4] else 0,
                   "v4_reason": str(q4.get("fov_failure_reason", ""))}
            for tag, o in (("v3", o3), ("v4", o4)):
                for f in FEATS:
                    a = o3b.get(f, np.nan) if tag == "v4" else o3b.get(f, np.nan)
                    bb = o4b.get(f, np.nan) if tag == "v4" else o3b.get(f, np.nan)
                    p = o.get(f, np.nan)
                    row[f"{tag}_{f}"] = p
                    row[f"{tag}_{f}_base"] = bb
                    row[f"{tag}_{f}_rel"] = ((p - bb) / abs(bb)
                                             if np.isfinite(bb) and np.isfinite(p) and bb else np.nan)
            rows.append(row)
        return {"baseline": base, "rows": rows}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"_err": f"{type(e).__name__}: {e}", "image_path": image_path,
                "_tb": traceback.format_exc()[-400:]}


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("A. FROZEN PRIOR VERSIONS")
    print("=" * 100)
    frozen = {k: sha(p) for k, p in {
        "clinical_measurement_v1_sha256": ROOT / "src/biomarker/clinical_measurement_v1.py",
        "clinical_measurement_v2_sha256":
            ROOT / "src/biomarker/clinical_measurement_v2_candidate.py",
        "clinical_measurement_v3_sha256":
            ROOT / "src/biomarker/clinical_measurement_v3_candidate.py",
        "clinical_measurement_v4_sha256":
            ROOT / "src/biomarker/clinical_measurement_v4_candidate.py",
        "task5b_h4_manifest_sha256": OUT / "task5b_h4_manifest.json",
        "task5b_h4_summary_sha256": OUT / "task5b_h4_summary.json",
        "task5b_h5_summary_sha256": OUT / "task5b_h5_summary.json",
        "task5b_h5_semantics_sha256": OUT / "task5b_h5_semantics.csv",
    }.items()}
    for k, val in frozen.items():
        print(f"  {k:38s} : {val}")

    manifest = {
        "candidate": v4.MEASUREMENT_VERSION_V4,
        "change_1": "content_bounds_v4 has NO fraction test; Otsu always uses the detected content "
                    "rectangle; the only new failure is `no_content_region` when the rectangle is "
                    "empty. No new tuned fraction threshold exists.",
        "change_2": "min_size = max(64, int(0.001 * content_h * content_w)); the 0.001 rule and the "
                    "64 floor are unchanged, only the area is the content domain.",
        "unchanged": ["CONST_TOL 2.0", "Otsu implementation",
                      "external band removed before morphology and labelling",
                      "remove_small_objects / remove_small_holes rule",
                      "binary_closing(disk(3))", "largest-component selection",
                      "validity rules 0.15 / 0.985 / 0.90",
                      "fractal window DOMAIN_MARGIN 0.10 / DOMAIN_ALIGN 8",
                      "D0/D1/D2 estimator", "density formulas", "SEG_CURRENT_V1"],
        "sample_rule": "all 8870 minus every previous sample (328 dev, 450 H3, 450 H4), sorted by "
                       "image_path, first k per (source, geom, split) with the smallest k in "
                       "{2,3,4,5,6,8,10,12,15,20,25} reaching N >= 150, then head(150)",
        "conditions": CONDS + [NATIVE],
        "gate": GATE,
        "stop_rule": "fail -> no tuning, no V5, no regeneration, no training",
    }
    (OUT / "task5b_h6_manifest.json").write_text(json.dumps(
        {"frozen": frozen, "manifest": manifest}, indent=2), encoding="utf-8")
    print("  manifest written: _private_audit/task5b_h6_manifest.json")

    t_ok, m_ok, c_ok = unit_tests()
    if not (t_ok and m_ok and c_ok):
        print("\n  STOP: unit invariants failed before image-level evaluation")
        raise SystemExit("UNIT_INVARIANT_FAILED")

    print()
    print("=" * 100)
    print("E. DEVELOPMENT CHECK on the known H5 residual conditions (not the decision sample)")
    print("=" * 100)
    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                       usecols=["image_path", "mask_path", "source", "split"])
    meta = meta.sort_values("image_path").reset_index(drop=True)
    meta["geom"] = [GEOM.get(Image.open(p).size, "other") for p in meta.image_path]
    dev = pd.read_csv(OUT / "task5b_h2_baseline.csv", usecols=["image_path"]).image_path.tolist()[:10]
    dm = meta.set_index("image_path")
    dargs = [(p, dm.loc[p, "mask_path"]) for p in dev if p in dm.index]
    with get_context("fork").Pool(8) as p:
        dr = [x for sub in p.map(dev_probe, dargs) for x in sub]
    D = pd.DataFrame(dr)
    D.to_csv(OUT / "task5b_h6_development.csv", index=False)
    print(f"  images {len(dargs)}   rows {len(D)}")
    for cond, g in D.groupby("condition"):
        print(f"  {cond:22s} thr V3 {g.v3_threshold.median():7.2f} -> V4 {g.v4_threshold.median():7.2f}"
              f"   min_size V3 {int(g.v3_min_size.median()):5d} -> V4 {int(g.v4_min_size.median()):5d}"
              f"   V3 fell back in {int(g.v3_fell_back.sum())}/{len(g)}   "
              f"canvas-dependent min_size in {int(g.canvas_min_size_changed.sum())}/{len(g)}")
    print("  H5 mechanism exercised before the fix : "
          f"fallback {int(D.v3_fell_back.sum())}, min_size {int(D.canvas_min_size_changed.sum())}")
    print("  V4 removes both by construction: no fraction test exists, and min_size uses the "
          "content area.")

    print()
    print("=" * 100)
    print("F. LOCKED SAMPLE")
    print("=" * 100)
    prev = set()
    for f in ("task5b_h2_baseline.csv", "task5b_h3_baseline.csv", "task5b_h4_baseline.csv"):
        prev |= set(pd.read_csv(OUT / f, usecols=["image_path"]).image_path)
    pool = meta[~meta.image_path.isin(prev)]
    strat = ["source", "geom", "split"]
    sel, used_k = pool.groupby(strat, group_keys=False).head(2), 2
    for k in (3, 4, 5, 6, 8, 10, 12, 15, 20, 25):
        if len(sel) >= 150:
            break
        sel, used_k = pool.groupby(strat, group_keys=False).head(k), k
    sel = sel.sort_values("image_path").head(150).reset_index(drop=True)
    print(f"  previous samples excluded       : {len(prev)}")
    print(f"  pool                            : {len(pool)}")
    print(f"  per-stratum cap used            : {used_k}")
    print(f"  LOCKED_SAMPLE_N                 : {len(sel)}")
    print(f"  disjoint                        : {not (set(sel.image_path) & prev)}")
    print(f"  sources {sel.source.value_counts().to_dict()}")
    print(f"  geoms   {sel.geom.value_counts().to_dict()}")
    print(f"  splits  {sel.split.value_counts().to_dict()}")
    if len(sel) != 150 or (set(sel.image_path) & prev):
        raise SystemExit("LOCKED_SAMPLE_INVALID")

    print()
    print("G/H. LOCKED RUN — V3 vs V4 on the targeted condition set")
    print("=" * 100)
    args = list(zip(sel.image_path, sel.mask_path, sel.source, sel.split, sel.geom))
    brows, rows, errs = [], [], 0
    with get_context("fork").Pool(16) as p:
        for i, r in enumerate(p.imap_unordered(worker, args, chunksize=1), 1):
            if r.get("_err"):
                errs += 1
                print(f"  ERR {r['image_path']}: {r['_err']} {r.get('_tb','')}", flush=True)
                continue
            brows.append(r["baseline"])
            rows.extend(r["rows"])
            if i % 25 == 0:
                pd.DataFrame(rows).to_csv(OUT / "task5b_h6_rows.csv", index=False)
                print(f"  {i}/{len(args)}  {time.time() - t0:.0f}s rows={len(rows)}", flush=True)
    B, C = pd.DataFrame(brows), pd.DataFrame(rows)
    B.to_csv(OUT / "task5b_h6_baseline.csv", index=False)
    C.to_csv(OUT / "task5b_h6_rows.csv", index=False)
    cnp = C[C.condition != NATIVE]
    print(f"  images {len(B)}  errors {errs}  rows {len(C)}  perturbed pairs {len(cnp)}")

    res = {}
    print()
    print("  overall V3 vs V4 (all content-preserving conditions):")
    for tag in ("v3", "v4"):
        d = cnp[f"{tag}_dice"].dropna()
        res[f"{tag}_median_dice"] = float(d.median())
        res[f"{tag}_p01_dice"] = float(d.quantile(.01))
        res[f"{tag}_min_dice"] = float(d.min())
        res[f"{tag}_rel_area_median"] = float(cnp[f"{tag}_rel_area"].median())
        res[f"{tag}_rel_area_p95"] = float(cnp[f"{tag}_rel_area"].quantile(.95))
        res[f"{tag}_centroid_median"] = float(cnp[f"{tag}_centroid"].median())
        print(f"    {tag.upper()}: median Dice {d.median():.6f}  p01 {d.quantile(.01):.6f}  "
              f"min {d.min():.6f}  relArea median {cnp[f'{tag}_rel_area'].median():+.6f} "
              f"p95 {cnp[f'{tag}_rel_area'].quantile(.95):+.6f}  "
              f"centroid {cnp[f'{tag}_centroid'].median():.6f}")
    for tag in ("v3", "v4"):
        j = cnp.join(B.set_index("image_path")[f"{tag}_valid"].rename("bv"), on="image_path")
        res[f"{tag}_valid_to_invalid"] = int((j.bv & ~j[f"{tag}_valid"]).sum())
        print(f"    {tag.upper()} valid->invalid {res[f'{tag}_valid_to_invalid']}")
    print("  by condition:")
    print(cnp.groupby("condition")[["v3_dice", "v4_dice"]].median().round(6).to_string())
    print("  by source:")
    print(cnp.groupby("source")[["v3_dice", "v4_dice"]].median().round(6).to_string())
    print("  by geometry:")
    print(cnp.groupby("geom")[["v3_dice", "v4_dice"]].median().round(6).to_string())
    for tag in ("v3", "v4"):
        for f in FEATS:
            s = stats(cnp, f"{tag}_{f}_rel")
            res[f"{tag}|{f}"] = s
            print(f"    {tag.upper()} {f:20s} median {s['median_rel']:+.5f}  p95 {s['p95_rel']:+.5f}"
                  f"  max {s['max_rel']:+.5f}  >1% {s['frac_gt_1pct']:.4f}  "
                  f">5% {s['frac_gt_5pct']:.4f}")
    for tag in ("v3", "v4"):
        tot = sum(int((cnp[f"{tag}_{f}_base"].notna() & cnp[f"{tag}_{f}"].isna()).sum())
                  for f in FEATS)
        res[f"{tag}_finite_to_nan"] = tot
        print(f"    {tag.upper()} finite->NaN over the five features: {tot}")

    print()
    print("I. TARGETED MECHANISM CHECK")
    print("=" * 100)
    dens = cnp[["v4_vessel_density_fov_rel", "v4_skel_density_fov_rel"]].abs().max(axis=1)
    cnp = cnp.assign(dens_abs=dens)
    jb = cnp.join(B.set_index("image_path")["v4_valid"].rename("bv"), on="image_path")
    resid = cnp[(cnp.v4_dice < 0.95) | (jb.bv & ~jb.v4_valid).values | (cnp.dens_abs > 0.05)]
    print(f"  V4 residual cases (Dice<0.95 | valid->invalid | density>5%) : {len(resid)}"
          f" of {len(cnp)}")
    res["v4_residual_rows"] = int(len(resid))
    fallback_res = int(resid.v3_fell_back.sum())
    minsize_res = int(resid.canvas_min_size_changed.sum())
    res["FULL_FRAME_OTSU_FALLBACK_RESIDUAL_N"] = fallback_res
    res["CANVAS_DEPENDENT_MINSIZE_RESIDUAL_N"] = minsize_res
    print(f"  of those, rows where V3 had fallen back to full-frame Otsu : {fallback_res}")
    print(f"  of those, rows where the canvas-dependent min_size differed : {minsize_res}")
    print(f"  all V3 rows where the fallback was exercised : {int(cnp.v3_fell_back.sum())}"
          f"   all V3 rows where canvas min_size differed : "
          f"{int(cnp.canvas_min_size_changed.sum())}")
    print(f"  V4 rows with any FOV pixel inside the detected padding : "
          f"{int((cnp.v4_in_padding_px > 0).sum())}")
    print(f"  V4 threshold identical to V3 threshold where V3 did NOT fall back : "
          f"{int((~cnp.v3_fell_back & (cnp.v4_threshold.sub(cnp.v3_threshold).abs() < 1e-9)).sum())}"
          f" of {int((~cnp.v3_fell_back).sum())}")
    if len(resid):
        print("  residual rows by condition:", resid.condition.value_counts().to_dict())
        print("  residual sample:")
        for r in resid.sort_values("v4_dice").head(10).itertuples():
            print(f"    {r.condition:22s} {r.source:11s} {r.geom:10s} dice={r.v4_dice:.4f} "
                  f"dens={r.dens_abs:.4f} reason={str(r.v4_reason)[:24]:24s} "
                  f"v3FellBack={r.v3_fell_back} minSizeChanged={r.canvas_min_size_changed}")

    print()
    print("J. FRACTAL REGRESSION CHECK")
    print("=" * 100)
    Bc = B.merge(meta[["image_path", "mask_path"]], on="image_path", how="left")
    ct = canvas_control(Bc.sort_values("image_path").head(6))
    ct.to_csv(OUT / "task5b_h6_canvas_control.csv", index=False)
    for f in ("fractal_d0", "fractal_d1", "fractal_d2"):
        r3 = ct[f"v3_{f}_rel"].replace([np.inf, -np.inf], np.nan).dropna()
        r4 = ct[f"v4_{f}_rel"].replace([np.inf, -np.inf], np.nan).dropna()
        ok = float(r4.abs().median()) <= 0.005
        res[f"canvas_v4_{f}_median_abs_rel"] = float(r4.abs().median())
        res[f"canvas_v4_{f}_max_abs_rel"] = float(r4.abs().max())
        print(f"  {f:12s} V3 median|rel| {r3.abs().median():.5f}  V4 "
              f"{r4.abs().median():.5f} (max {r4.abs().max():.5f})  RESOLVED={ok}")

    print()
    print("  NATIVE control (unperturbed, V3 vs V4):")
    nv = B[B.condition == NATIVE]
    print(f"    native FOV Dice V4 vs V3 median {nv.drift_dice_v4_v3.median():.6f}  "
          f"p05 {nv.drift_dice_v4_v3.quantile(.05):.6f}  min {nv.drift_dice_v4_v3.min():.6f}")
    print(f"    native validity V3->V4: lost {int((nv.v3_valid & ~nv.v4_valid).sum())}  "
          f"gained {int((~nv.v3_valid & nv.v4_valid).sum())}")
    res["native_dice_median"] = float(nv.drift_dice_v4_v3.median())
    res["native_validity_lost"] = int((nv.v3_valid & ~nv.v4_valid).sum())
    for f in FEATS:
        a = nv[f"v3_{f}"].replace([np.inf, -np.inf], np.nan)
        b = nv[f"v4_{f}"].replace([np.inf, -np.inf], np.nan)
        rel = ((b - a) / a.abs()).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"    {f:20s} native median rel {rel.median():+.5f}  frac>5% "
              f"{float((rel.abs() > .05).mean()):.4f}")

    print()
    print("=" * 100)
    print("L. GATE")
    print("=" * 100)
    checks = {
        "G1_no_fallback_residual": fallback_res <= GATE["G1_fallback_residual_max"],
        "G2_no_canvas_minsize_residual": minsize_res <= GATE["G2_canvas_minsize_residual_max"],
        "G3_median_dice": res["v4_median_dice"] >= GATE["G3_median_dice_min"],
        "G4_p01_dice": res["v4_p01_dice"] >= GATE["G4_p01_dice_min"],
        "G5_valid_to_invalid": res["v4_valid_to_invalid"] <= GATE["G5_valid_to_invalid_max"],
        "G6_vessel_density_gt5pct": res["v4|vessel_density_fov"]["frac_gt_5pct"]
        <= GATE["G6_vessel_density_frac_gt_5pct_max"],
        "G7_skel_density_gt5pct": res["v4|skel_density_fov"]["frac_gt_5pct"]
        <= GATE["G7_skel_density_frac_gt_5pct_max"],
        "G8_no_source_geom_failure": bool(
            (cnp.groupby("source").v4_dice.median() >= 0.99).all()
            and (cnp.groupby("geom").v4_dice.median() >= 0.99).all()),
        "G9_no_new_nan": res["v4_finite_to_nan"] <= 0.005 * len(cnp),
        "G10_canvas_resolved": all(res[f"canvas_v4_{f}_median_abs_rel"] <= 0.005
                                   for f in ("fractal_d0", "fractal_d1", "fractal_d2")),
        "G11_native_reviewed": res["native_validity_lost"] <= 0.02 * len(nv),
    }
    for k, v in checks.items():
        print(f"  {k:32s} : {'PASS' if v else 'FAIL'}")
    npass = sum(checks.values())
    verdict = "PASS" if all(checks.values()) else ("CONCERN" if npass >= len(checks) - 2 else "FAIL")
    print()
    print(f"  checks passed {npass}/{len(checks)}  -> V4_STATUS = "
          f"{'SUPPORTED' if all(checks.values()) else 'NOT_SUPPORTED'} ({verdict})")
    print()
    print("K. PHOTOMETRIC CLIPPING recorded as a separate known operating-range limitation:")
    print("   x1.50-x1.80 saturation is NOT part of the border-specific V4 decision.")
    (OUT / "task5b_h6_summary.json").write_text(json.dumps(
        {"gate_checks": checks, "checks_passed": npass, "checks_total": len(checks),
         "verdict": verdict, "results": res, "frozen": frozen, "manifest": manifest,
         "locked_N": int(len(B)), "rows": int(len(C)), "errors": errs,
         "V4_STATUS": "SUPPORTED" if all(checks.values()) else "NOT_SUPPORTED"},
        indent=2, default=str), encoding="utf-8")
    print(f"\n  elapsed {time.time() - t0:.0f}s")


def canvas_control(selrow):
    from src.biomarker import clinical_measurement_v1 as v1
    rows = []
    for r in selrow.itertuples():
        rgb, msk = load(r.image_path, r.mask_path)
        h0, w0 = msk.shape
        f1, _ = v1.retinal_fov(rgb[:, :, 1])
        sup = (msk.astype(bool) & f1)
        b3, b4 = v3.fractal_v3(sup, f1), v4.fractal_v4(sup, f1)
        for spec in pad_specs(h0, w0):
            t, b, l, rr = spec[2]
            if t == b == l == rr == 0:
                continue
            ps = np.pad(sup, ((t, b), (l, rr)), constant_values=False)
            pf = np.pad(f1, ((t, b), (l, rr)), constant_values=False)
            p3, p4 = v3.fractal_v3(ps, pf), v4.fractal_v4(ps, pf)
            row = {"image_path": r.image_path, "condition": spec[1]}
            for f in ("fractal_d0", "fractal_d1", "fractal_d2"):
                row[f"v3_{f}_rel"] = ((p3[f] - b3[f]) / b3[f]) if b3[f] else np.nan
                row[f"v4_{f}_rel"] = ((p4[f] - b4[f]) / b4[f]) if b4[f] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
