#!/usr/bin/env python
"""Task 5B-H3: locked evaluation of the predeclared CLINICAL_MEASUREMENT_V2_CANDIDATE.

Order of operations is fixed:

  1. freeze V1 (module sha, feature-table sha, the five FINAL_PRIMARY formulas);
  2. WRITE THE MANIFEST — candidate spec, every constant, the sample-selection rule, the
     perturbation list and the acceptance gate — to `_private_audit/task5b_h3_manifest.json`
     BEFORE any locked-evaluation number exists;
  3. reproduce the known V1 failure on a small slice of the already-inspected 328-image
     development sample and confirm the candidate executes (development only);
  4. build the LOCKED evaluation sample, N >= 400, disjoint from all 328 development images;
  5. run the locked evaluation: V1 and V2 on the same conditions, vessel mask fixed;
  6. apply the gate exactly as written in step 2.

The development sample is never used for the decision. No disease label is read. No classifier is
trained. SEG_CURRENT_V1 is untouched. The 8,870-row feature table is not regenerated.
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
from scipy.stats import spearmanr

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.task5b_n2 import load  # noqa: E402
from scripts.task5b_h_fov_stress import (  # noqa: E402
    BRIGHTNESS, GEOM, apply_border, apply_brightness, pad_specs,
)
from src.biomarker import clinical_measurement_v2_candidate as v2  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"
V1_MODULE = ROOT / "src/biomarker/clinical_measurement_v1.py"
V2_MODULE = ROOT / "src/biomarker/clinical_measurement_v2_candidate.py"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
DEV_N = 328
NOVEL_FRACS = {"w1": 0.02, "w2": 0.06, "w3": 0.12, "w4": 0.035, "w5": 0.18}

# ---------------------------------------------------------------- PREDECLARED ACCEPTANCE GATE
GATE = {
    "G1_border_median_dice_min": 0.999,
    "G1_border_p01_dice_min": 0.95,
    "G1_every_condition_median_dice_min": 0.99,
    "G2_border_valid_to_invalid_max": 5,
    "G3_vessel_density_border_frac_gt_1pct_max": 0.05,
    "G3_skel_density_border_frac_gt_1pct_max": 0.05,
    "G4_canvas_median_abs_rel_max": 0.005,
    "G5_new_nan_frac_max": 0.005,
    "G6_subgroup_border_frac_gt_5pct_max": 0.10,
    "G7_native_validity_loss_frac_max": 0.02,
}


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def novel_specs(h: int, w: int):
    """PREDECLARED novel border conditions, never used while building the candidate."""
    mn = min(h, w)
    k = {n: max(4, int(round(f * mn))) for n, f in NOVEL_FRACS.items()}
    return [
        ("novel_uniform_black", "novel_black_w4", (k["w4"],) * 4, 0, False),
        ("novel_uniform_black", "novel_black_w5", (k["w5"],) * 4, 0, False),
        ("novel_gray20", "novel_gray20_w2", (k["w2"],) * 4, 20, False),
        ("novel_gray90", "novel_gray90_w2", (k["w2"],) * 4, 90, False),
        ("novel_asym", "novel_asym_A", (k["w4"], k["w2"], k["w5"], k["w1"]), 0, False),
        ("novel_asym", "novel_asym_B", (k["w5"], 0, 0, k["w4"]), 0, False),
    ]


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2.0 * int(np.logical_and(a, b).sum()) / s) if s else float("nan")


def disp(a, b, h0, w0):
    if not (a.sum() and b.sum()):
        return np.nan
    ay, ax = ndi.center_of_mass(a)
    by, bx = ndi.center_of_mass(b)
    return float(np.hypot(ax - bx, ay - by) / max(h0, w0))


def feats(out):
    return {f: out.get(f, np.nan) for f in FEATS}


def worker(arg):
    image_path, mask_path, source, split, geom = arg[:5]
    only = arg[5] if len(arg) > 5 else None
    try:
        rgb, msk = load(image_path, mask_path)
        h0, w0 = msk.shape
        o1b, f1b, q1b = v2.measure_v1(rgb, msk, {})
        o2b, f2b, q2b = v2.measure_v2(rgb, msk, {})
        rows = []
        base = {"image_path": image_path, "source": source, "split": split, "geom": geom,
                "v1_fov_px": int(f1b.sum()), "v2_fov_px": int(f2b.sum()),
                "v1_valid": bool(q1b.get("fov_valid")),
                "v2_valid": bool(q2b.get("fov_valid")),
                "drift_fov_dice": dice(f2b, f1b),
                "drift_area_rel": float((int(f2b.sum()) - int(f1b.sum())) / max(int(f1b.sum()), 1)),
                "drift_centroid": disp(f2b, f1b, h0, w0)}
        for f in FEATS:
            a, b = o1b.get(f, np.nan), o2b.get(f, np.nan)
            base["v1_" + f] = a
            base["v2_" + f] = b
            base["drift_" + f + "_rel"] = ((b - a) / abs(a)
                                           if np.isfinite(a) and np.isfinite(b) and a else np.nan)
        rows.append(base)
        conds = [("brightness", f"x{c:.2f}", apply_brightness(rgb, msk, c)) for c in BRIGHTNESS]
        conds += [(s[0], s[1], apply_border(rgb, msk, s))
                  for s in (pad_specs(h0, w0) + novel_specs(h0, w0))]
        for fam, cond, (prgb, pmsk, pad) in conds:
            if only is not None and cond not in only:
                continue
            t, b_, l, r_ = pad
            o1, pf1, q1 = v2.measure_v1(prgb, pmsk, {})
            o2, pf2, q2 = v2.measure_v2(prgb, pmsk, {})
            c1, c2 = pf1[t:t + h0, l:l + w0], pf2[t:t + h0, l:l + w0]
            row = {"image_path": image_path, "source": source, "split": split, "geom": geom,
                   "family": "brightness" if fam == "brightness" else "border",
                   "condition": cond,
                   "v1_dice": dice(c1, f1b), "v2_dice": dice(c2, f2b),
                   "v1_rel_area": float((int(c1.sum()) - int(f1b.sum())) / max(int(f1b.sum()), 1)),
                   "v2_rel_area": float((int(c2.sum()) - int(f2b.sum())) / max(int(f2b.sum()), 1)),
                   "v1_centroid": disp(c1, f1b, h0, w0), "v2_centroid": disp(c2, f2b, h0, w0),
                   "v1_valid": bool(q1.get("fov_valid")),
                   "v2_valid": bool(q2.get("fov_valid")),
                   "v1_ncomp": int(q1.get("fov_n_components", -1)),
                   "v2_ncomp": int(q2.get("fov_n_components", -1))}
            for f in FEATS:
                a, b = o1b.get(f, np.nan), o2b.get(f, np.nan)
                pa, pb = o1.get(f, np.nan), o2.get(f, np.nan)
                row["v1_" + f] = pa
                row["v2_" + f] = pb
                row["v1_" + f + "_base"] = a
                row["v2_" + f + "_base"] = b
                row["v1_" + f + "_rel"] = ((pa - a) / abs(a)
                                           if np.isfinite(a) and np.isfinite(pa) and a else np.nan)
                row["v2_" + f + "_rel"] = ((pb - b) / abs(b)
                                           if np.isfinite(b) and np.isfinite(pb) and b else np.nan)
            rows.append(row)
        return {"baseline": base, "rows": rows}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"_err": f"{type(e).__name__}: {e}", "image_path": image_path,
                "_tb": traceback.format_exc()[-400:]}


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


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("A. V1 FROZEN")
    print("=" * 100)
    frozen = {
        "clinical_measurement_v1_sha256": sha(V1_MODULE),
        "clinical_measurement_v2_candidate_sha256": sha(V2_MODULE),
        "final_feature_table": "data/features/final_biomarkers_v1.csv",
        "final_feature_table_sha256":
            "f1c41e923ae29d4e228097536765f5e7399963062253ad925657a077cddc10c0",
        "clinical_measurement_table_sha256":
            "db123ac5f663f4925ac9fff52d204bede85963966794062d1855e315a4e38ffc",
        "five_final_primary_formulas": {
            "vessel_density_fov": "mask[fov].sum() / fov_px",
            "skel_density_fov": "len(nonzero(skeletonize(mask))) / fov_px   # numerator whole-frame",
            "fractal_d0": "MultifractalVBMs(n_rotations=25, optimize=True, min_proba=1e-4, "
                          "maxproba=0.9999).compute_multifractals(mask & fov)[0]",
            "fractal_d1": "... same call, component 1",
            "fractal_d2": "... same call, component 2",
        },
    }
    for k, v in frozen.items():
        if not isinstance(v, dict):
            print(f"  {k:38s} : {v}")
    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                       usecols=["image_path", "mask_path", "source", "split"])
    meta = meta.sort_values("image_path").reset_index(drop=True)
    meta["geom"] = [GEOM.get(Image.open(p).size, "other") for p in meta.image_path]

    print()
    print("=" * 100)
    print("B/D/N. MANIFEST WRITTEN BEFORE ANY LOCKED NULL")
    print("=" * 100)
    dev = set(pd.read_csv(OUT / "task5b_h2_baseline.csv", usecols=["image_path"]).image_path)
    cand = {
        "name": v2.MEASUREMENT_VERSION_V2,
        "B1_fov_thresholding": {
            "rule": "Otsu estimated on the frame with contiguous near-constant edge rows/columns "
                    "removed; threshold then applied to the FULL frame; the same detected constant "
                    "edges are also excluded from the RETURNED mask; all other steps V1's",
            "CONST_TOL": v2.CONST_TOL, "MIN_CONTENT_FRAC": v2.MIN_CONTENT_FRAC,
            "revision_note": "the mask-exclusion half of B1 was added after the DEVELOPMENT check "
                             "(10 images, 6 conditions) showed externally added constant padding "
                             "could still enter the FOV mask, and BEFORE the locked evaluation was "
                             "started; no new constant, no parameter tuned on outcomes"},
        "B2_fractal_domain": {
            "rule": "same MultifractalVBMs estimator run on a square window centred on the FOV "
                    "bounding-box centre, side = ceil(max(bbox)* (1+2*margin)) rounded to 8, "
                    "zero outside the frame",
            "DOMAIN_MARGIN": v2.DOMAIN_MARGIN, "DOMAIN_ALIGN": v2.DOMAIN_ALIGN},
        "unchanged": ["SEG_CURRENT_V1 masks", "vessel_density_fov formula",
                      "skel_density_fov formula (numerator stays WHOLE-FRAME)", "tortuosity",
                      "width", "disc handling", "labels", "splits",
                      "remove_small_objects/holes min_size", "binary_closing(disk(3))",
                      "largest-component selection", "failure criteria 0.15/0.985/0.90"],
        "sample_rule": "all 8870 canonical images minus the 328 development images, sorted by "
                       "image_path, first k per (source, geom, split) with the smallest k in "
                       "{12,16,20,25,30,40,60} that reaches N >= 400; no label, no feature and "
                       "no FOV outcome used",
        "target_N_min": 400,
        "conditions": {"brightness": [f"x{c:.2f}" for c in BRIGHTNESS],
                       "legacy_border": [s[1] for s in pad_specs(384, 512)],
                       "novel_border": [s[1] for s in novel_specs(384, 512)]},
        "acceptance_gate": GATE,
        "stop_rule": "gate fail -> no tuning, no V3, no regeneration, no training",
    }
    (OUT / "task5b_h3_manifest.json").write_text(json.dumps(
        {"frozen": frozen, "candidate": cand}, indent=2), encoding="utf-8")
    print(f"  manifest written: _private_audit/task5b_h3_manifest.json")
    print(f"  gate: {GATE}")

    print()
    print("D. DEVELOPMENT CHECK (already-inspected 328; NOT used for the decision)")
    dev_sel = meta[meta.image_path.isin(dev)].sort_values("image_path").head(10)
    DEV_CONDS = ["asymmetric_black_w3", "dark_gray_40_w2", "irregular_frame_w3",
                 "uniform_black_w3", "novel_black_w5", "novel_gray90_w2"]
    dargs = list(zip(dev_sel.image_path, dev_sel.mask_path, dev_sel.source, dev_sel.split,
                     dev_sel.geom, [DEV_CONDS] * len(dev_sel)))
    dv1, dv2, ddd = [], [], []
    for a in dargs:
        r = worker(a)
        if r.get("_err"):
            print(f"  DEV ERR {r['_err']}  {r.get('_tb','')}")
            continue
        b = r["baseline"]
        ddd.append({"image_path": b["image_path"], "drift_fov_dice": b["drift_fov_dice"]})
        for row in r["rows"]:
            if "condition" not in row:
                continue
            if row["condition"] in ("asymmetric_black_w3", "dark_gray_40_w2", "irregular_frame_w3",
                                    "uniform_black_w3", "novel_black_w5", "novel_gray90_w2"):
                dv1.append(row["v1_vessel_density_fov_rel"])
                dv2.append(row["v2_vessel_density_fov_rel"])
    DD = pd.DataFrame(ddd)
    print(f"  dev images measured            : {len(DD)}")
    print(f"  V1/V2 native FOV Dice median   : {DD.drift_fov_dice.median():.6f} "
          f"min {DD.drift_fov_dice.min():.6f}")
    print(f"  worst-case border rows  V1 |rel density| median "
          f"{np.nanmedian(np.abs(dv1)):.5f}  max {np.nanmax(np.abs(dv1)):.5f}")
    print(f"  worst-case border rows  V2 |rel density| median "
          f"{np.nanmedian(np.abs(dv2)):.5f}  max {np.nanmax(np.abs(dv2)):.5f}")
    print(f"  KNOWN_V1_FAILURE_REPRODUCED    : {bool(np.nanmax(np.abs(dv1)) > 0.1)}")
    print(f"  V2_EXECUTES_AND_REDUCES        : {bool(np.nanmax(np.abs(dv2)) < np.nanmax(np.abs(dv1)))}")

    print()
    print("=" * 100)
    print("D. LOCKED EVALUATION SAMPLE")
    print("=" * 100)
    pool = meta[~meta.image_path.isin(dev)]
    strat = ["source", "geom", "split"]
    sel = pool.groupby(strat, group_keys=False).head(12)
    used_k = 12
    for k in (16, 20, 25, 30, 40, 60):
        if len(sel) >= 400:
            break
        sel = pool.groupby(strat, group_keys=False).head(k)
        used_k = k
    sel = sel.sort_values("image_path").head(900).reset_index(drop=True)
    overlap = set(sel.image_path) & dev
    print(f"  pool (8870 - 328 dev)          : {len(pool)}")
    print(f"  per-stratum cap used           : {used_k}")
    print(f"  LOCKED_EVALUATION_SAMPLE_N     : {len(sel)}")
    print(f"  disjoint from the 328 dev      : {len(overlap) == 0}")
    print(f"  sources    : {sel.source.value_counts().to_dict()}")
    print(f"  geometries : {sel.geom.value_counts().to_dict()}")
    print(f"  splits     : {sel.split.value_counts().to_dict()}")
    if not (len(overlap) == 0 and len(sel) >= 400):
        raise SystemExit("LOCKED_SAMPLE_INVALID")

    print()
    print("=" * 100)
    print("E/F. LOCKED RUN — V1 AND V2, SAME CONDITIONS, FIXED MASK")
    print("=" * 100)
    args = list(zip(sel.image_path, sel.mask_path, sel.source, sel.split, sel.geom))
    brows, rows, errs = [], [], 0
    with get_context("fork").Pool(24) as pool_:
        for i, r in enumerate(pool_.imap_unordered(worker, args, chunksize=1), 1):
            if r.get("_err"):
                errs += 1
                print(f"  ERR {r['image_path']}: {r['_err']} {r.get('_tb','')}", flush=True)
                continue
            brows.append(r["baseline"])
            rows.extend(r["rows"])
            if i % 25 == 0:
                pd.DataFrame(rows).to_csv(OUT / "task5b_h3_rows.csv", index=False)
                pd.DataFrame(brows).to_csv(OUT / "task5b_h3_baseline.csv", index=False)
                print(f"  {i}/{len(args)}  {time.time() - t0:.0f}s  rows={len(rows)}", flush=True)
    B = pd.DataFrame(brows)
    C = pd.DataFrame(rows)
    B.to_csv(OUT / "task5b_h3_baseline.csv", index=False)
    C.to_csv(OUT / "task5b_h3_rows.csv", index=False)
    cb, cn = C[C.family == "brightness"], C[C.family == "border"]
    print(f"  images {len(B)}  errors {errs}  rows {len(C)}  border {len(cn)}  brightness {len(cb)}")

    print()
    print("=" * 100)
    print("G. FOV ROBUSTNESS — V1 vs V2")
    print("=" * 100)
    res = {}
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
        print(f"  {ver.upper()} border: median Dice {d.median():.6f}  mean {d.mean():.6f}  "
              f"p01 {d.quantile(.01):.6f}  min {d.min():.6f}  <0.99 {float((d < 0.99).mean()):.4f}  "
              f"<0.90 {float((d < 0.90).mean()):.4f}")
        print(f"       rel area median {cn[f'{ver}_rel_area'].median():+.6f}  p95 "
              f"{cn[f'{ver}_rel_area'].quantile(.95):+.6f}  max {cn[f'{ver}_rel_area'].max():+.6f}"
              f"   centroid median {cn[f'{ver}_centroid'].median():.6f}")
    vt = {}
    for ver in ("v1", "v2"):
        j = cn.join(B.set_index("image_path")[f"{ver}_valid"].rename("bv"), on="image_path")
        vt[ver] = {"valid_to_invalid": int((j.bv & ~j[f"{ver}_valid"]).sum()),
                   "invalid_to_valid": int((~j.bv & j[f"{ver}_valid"]).sum())}
        print(f"  {ver.upper()} FOV validity: valid->invalid {vt[ver]['valid_to_invalid']}  "
              f"invalid->valid {vt[ver]['invalid_to_valid']}")
    res["validity_transitions"] = vt
    print("  by condition (median Dice), worst first:")
    conds = cn.groupby("condition").agg(v1=("v1_dice", "median"), v2=("v2_dice", "median"),
                                        v2min=("v2_dice", "min")).sort_values("v2")
    print(conds.round(6).to_string())
    res["by_condition_v2_min_median_dice"] = float(conds.v2.min())
    print("  by source:")
    print(cn.groupby("source")[["v1_dice", "v2_dice"]].median().round(6).to_string())
    print("  by geometry:")
    print(cn.groupby("geom")[["v1_dice", "v2_dice"]].median().round(6).to_string())
    res["by_source"] = cn.groupby("source")[["v1_dice", "v2_dice"]].median().to_dict()
    res["by_geom"] = cn.groupby("geom")[["v1_dice", "v2_dice"]].median().to_dict()

    print()
    print("=" * 100)
    print("H. FIVE PRIMARY FEATURES — V2 (and V1) UNDER BORDER AND BRIGHTNESS")
    print("=" * 100)
    summ = {}
    for label, g in (("BORDER", cn), ("BRIGHTNESS", cb)):
        for ver in ("v1", "v2"):
            print(f"  --- {label} / {ver.upper()} ---")
            print(f"  {'feature':20s} {'medRel':>10s} {'p95Rel':>10s} {'maxRel':>10s} {'>1%':>7s} "
                  f"{'>2%':>7s} {'>5%':>7s} {'>10%':>7s}")
            for f in FEATS:
                s = stats(g, f"{ver}_{f}_rel")
                summ[f"{label}|{ver}|{f}"] = s
                if not s:
                    continue
                print(f"  {f:20s} {s['median_rel']:+10.5f} {s['p95_rel']:+10.5f} "
                      f"{s['max_rel']:+10.5f} {s['frac_gt_1pct']:7.4f} {s['frac_gt_2pct']:7.4f} "
                      f"{s['frac_gt_5pct']:7.4f} {s['frac_gt_10pct']:7.4f}")
    print("  by source / geometry (V2 border median relative delta):")
    for key in ("source", "geom"):
        print(f"    by {key}:")
        for f in FEATS:
            g = cn.groupby(key)[f"v2_{f}_rel"].median()
            print(f"      {f:20s} " + "  ".join(f"{k}={v:+.5f}" for k, v in g.items()))
    for key in ("source", "geom"):
        for f in FEATS:
            g = cn.groupby(key)[f"v2_{f}_rel"].apply(lambda s: float((s.abs() > .05).mean()))
            summ[f"subgroup|{key}|{f}"] = {k: float(v) for k, v in g.items()}

    print()
    print("J. NaN AND FAILURE TRANSITIONS")
    print("=" * 100)
    for label, g in (("BORDER", cn), ("BRIGHTNESS", cb)):
        for ver in ("v1", "v2"):
            tot = 0
            for f in FEATS:
                b = g[f"{ver}_{f}_base"].notna()
                p = g[f"{ver}_{f}"].notna()
                fin2nan = int((b & ~p).sum())
                nan2fin = int((~b & p).sum())
                tot += fin2nan
                summ[f"nan|{label}|{ver}|{f}"] = {"finite_to_nan": fin2nan,
                                                  "nan_to_finite": nan2fin,
                                                  "baseline_nan": int((~b).sum())}
            print(f"  {label:10s} {ver.upper()}: total finite->NaN over the five features = {tot}"
                  f"   (baseline NaN {int(g[f'{ver}_vessel_density_fov_base'].isna().sum())})")
    new_nan = sum(v["finite_to_nan"] for k, v in summ.items() if k.startswith("nan|BORDER|v2|"))
    res["v2_border_new_nan"] = int(new_nan)

    print()
    print("I. FRACTAL ZERO-PADDING CONTROL (identical binary support, larger canvas)")
    print("=" * 100)
    ct = canvas_control(sel.head(6))
    ct.to_csv(OUT / "task5b_h3_canvas_control.csv", index=False)
    for f in ("fractal_d0", "fractal_d1", "fractal_d2"):
        r1 = ct[f"v1_{f}_rel"].replace([np.inf, -np.inf], np.nan).dropna()
        r2 = ct[f"v2_{f}_rel"].replace([np.inf, -np.inf], np.nan).dropna()
        res[f"canvas_v1_{f}_median_abs_rel"] = float(r1.abs().median())
        res[f"canvas_v2_{f}_median_abs_rel"] = float(r2.abs().median())
        res[f"canvas_v2_{f}_max_abs_rel"] = float(r2.abs().max())
        print(f"  {f:12s} V1 median|rel|={r1.abs().median():.5f} max={r1.abs().max():.5f}   "
              f"V2 median|rel|={r2.abs().median():.5f} max={r2.abs().max():.5f}")

    print()
    print("K. NATIVE-IMAGE DRIFT V1 -> V2 (unperturbed)")
    print("=" * 100)
    dd = B.drift_fov_dice.dropna()
    print(f"  native FOV Dice V1 vs V2 : median {dd.median():.6f}  p05 {dd.quantile(.05):.6f}  "
          f"min {dd.min():.6f}")
    print(f"  native FOV area rel change: median {B.drift_area_rel.median():+.6f}  "
          f"p05 {B.drift_area_rel.quantile(.05):+.6f}  p95 {B.drift_area_rel.quantile(.95):+.6f}")
    for f in FEATS:
        c = f"drift_{f}_rel"
        r = B[c].replace([np.inf, -np.inf], np.nan).dropna()
        res[f"native_drift_{f}_median_rel"] = float(r.median())
        res[f"native_drift_{f}_p95_abs_rel"] = float(r.abs().quantile(.95))
        res[f"native_drift_{f}_max_abs_rel"] = float(r.abs().max())
        print(f"  {f:20s} median {r.median():+.5f}  p95|.| {r.abs().quantile(.95):.5f}  "
              f"max|.| {r.abs().max():.5f}  frac>5% {float((r.abs() > .05).mean()):.4f}")
    lost = int((B.v1_valid & ~B.v2_valid).sum())
    gained = int((~B.v1_valid & B.v2_valid).sum())
    res["native_validity_lost"] = lost
    res["native_validity_gained"] = gained
    res["native_validity_loss_frac"] = float(lost / max(len(B), 1))
    print(f"  native validity V1->V2   : lost {lost}  gained {gained}  "
          f"({float(lost / max(len(B), 1)):.4f} of {len(B)})")
    worst = B.reindex(B.drift_fov_dice.sort_values().index).head(5)
    print("  largest native disagreements:")
    for r in worst.itertuples():
        print(f"    {r.image_path.split('/')[-1][:44]:44s} dice={r.drift_fov_dice:.4f} "
              f"v1_px={r.v1_fov_px} v2_px={r.v2_fov_px} valid {r.v1_valid}->{r.v2_valid}")

    print()
    print("=" * 100)
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
        "G6_subgroups": all(v <= GATE["G6_subgroup_border_frac_gt_5pct_max"]
                            for k, v in summ.items()
                            if k.startswith("subgroup|")
                            and k.split("|")[-1] in ("vessel_density_fov", "skel_density_fov")),
        "G7_native_validity": res["native_validity_loss_frac"]
        <= GATE["G7_native_validity_loss_frac_max"],
    }
    for k, v in checks.items():
        print(f"  {k:34s} : {'PASS' if v else 'FAIL'}")
    gate_pass = all(checks.values())
    print()
    print(f"  FOV_BORDER_ROBUSTNESS_V2               : "
          f"{'PASS' if gate_pass else 'CONCERN' if sum(checks.values()) >= len(checks) - 2 else 'FAIL'}")
    print(f"  VESSEL_DENSITY_BORDER_PROBLEM_RESOLVED : "
          f"{'YES' if checks['G3_vessel_density'] else 'NO'}")
    print(f"  SKEL_DENSITY_BORDER_PROBLEM_RESOLVED   : "
          f"{'YES' if checks['G3_skel_density'] else 'NO'}")
    for i, f in enumerate(("d0", "d1", "d2")):
        print(f"  FRACTAL_ZERO_PADDING_{f.upper()}_RESOLVED     : "
              f"{'YES' if checks['G4_canvas_' + f] else 'NO'}")
    print(f"  NATIVE_V1_V2_DRIFT_REVIEWED            : YES")
    out = {"gate_checks": checks, "gate_pass": gate_pass, "results": res,
           "v1_border_median_dice": res["v1_border_median_dice"],
           "v2_border_median_dice": res["v2_border_median_dice"],
           "stats": summ, "errors": errs, "locked_N": int(len(B)), "rows": int(len(C))}
    (OUT / "task5b_h3_summary.json").write_text(json.dumps(out, indent=2, default=str),
                                                encoding="utf-8")
    print(f"\n  elapsed {time.time() - t0:.0f}s")


def canvas_control(selrow):
    """Same binary support + same FOV, translated into larger zero canvases. No FOV detection."""
    from src.biomarker import clinical_measurement_v1 as v1
    rows = []
    for r in selrow.itertuples():
        rgb, msk = load(r.image_path, r.mask_path)
        h0, w0 = msk.shape
        f1, _ = v1.retinal_fov(rgb[:, :, 1])
        sup = (msk.astype(bool) & f1)
        b1 = v1._fractal(sup)
        b2 = v2.fractal_v2(sup, f1)
        for spec in pad_specs(h0, w0):
            t, b, l, rr = spec[2]
            if t == b == l == rr == 0:
                continue
            ps = np.pad(sup, ((t, b), (l, rr)), mode="constant", constant_values=False)
            pf = np.pad(f1, ((t, b), (l, rr)), mode="constant", constant_values=False)
            p1 = v1._fractal(ps)
            p2 = v2.fractal_v2(ps, pf)
            row = {"image_path": r.image_path, "condition": spec[1]}
            for f in ("fractal_d0", "fractal_d1", "fractal_d2"):
                row[f"v1_{f}_rel"] = ((p1[f] - b1[f]) / b1[f]) if b1[f] else np.nan
                row[f"v2_{f}_rel"] = ((p2[f] - b2[f]) / b2[f]) if b2[f] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
