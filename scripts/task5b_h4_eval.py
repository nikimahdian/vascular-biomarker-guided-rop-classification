#!/usr/bin/env python
"""Task 5B-H4: locked evaluation of CLINICAL_MEASUREMENT_V3_CANDIDATE (padding-aware selection).

Predeclared before the run, all of it in `_private_audit/task5b_h4_manifest.json`:

  * V3 algorithm and order of operations — the ONE change is that the detected external constant
    band is removed from the binarisation BEFORE morphology and BEFORE labelling, so a
    threshold-positive padding ring cannot be a candidate component and cannot inflate `frag`;
  * constants: CONST_TOL 2.0, MIN_CONTENT_FRAC 0.50 (V2's, unchanged), DOMAIN_MARGIN 0.10,
    DOMAIN_ALIGN 8 (V2's, unchanged), min_size / disk(3) / failure criteria 0.15 / 0.985 / 0.90 (V1's);
  * fallback: untrimmed frame -> no external band -> V3 selects exactly as V2/V1;
  * failure behaviour: unchanged reasons and thresholds, no new constant anywhere;
  * family classification: `overwrite_*` conditions OVERWRITE retinal pixels and are CONTENT_ALTERING;
    they are reported separately and are not part of the "plausible border" Dice floor;
  * cost declaration: V3 is evaluated on all 42 conditions; V1 and V2 on a predeclared
    8-condition mechanism subset on the same new sample, with their full-set numbers taken from the
    Task-5B-H3 locked 450-image run (identical conditions and procedure).

Locked sample: N >= 400, disjoint from the 328 development images and the 450 H3 locked images.
No disease label. No classifier. No SEG_CURRENT_V1 change. The 8,870-row table is not regenerated.
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
from scripts.task5b_h_fov_stress import (  # noqa: E402
    GEOM, apply_border, apply_brightness, pad_specs, BRIGHTNESS,
)
from scripts.task5b_h3_eval import novel_specs  # noqa: E402
from src.biomarker import clinical_measurement_v2_candidate as v2  # noqa: E402
from src.biomarker import clinical_measurement_v3_candidate as v3  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
V1_MOD = ROOT / "src/biomarker/clinical_measurement_v1.py"
V2_MOD = ROOT / "src/biomarker/clinical_measurement_v2_candidate.py"
V3_MOD = ROOT / "src/biomarker/clinical_measurement_v3_candidate.py"
MECH_SUBSET = ["novel_gray90_w2", "dark_gray_40_w2"] + [f"ch_grey_{n}" for n in
               ("bg_m30_w2", "bg_m10_w2", "bg_p10_w2", "bg_p30_w2", "asym_w2", "bg_p30_w5")]
CONTENT_ALTERING = ("overwrite_lr_w2", "overwrite_all_w2")

GATE = {
    "G1_median_dice_min": 0.999,
    "G2_p01_dice_min": 0.95,
    "G3_plausible_condition_median_dice_min": 0.99,
    "G4_valid_to_invalid_max": 5,
    "G5_grey_component_failure_frac_max": 0.005,
    "G6_vessel_density_frac_gt_1pct_max": 0.05,
    "G7_skel_density_frac_gt_1pct_max": 0.05,
    "G8_fractal_zero_padding_abs_rel_max": 0.005,
    "G9_new_nan_frac_max": 0.005,
    "G10_subgroup_frac_gt_5pct_max": 0.10,
    "G11_native_valid_to_invalid_frac_max": 0.02,
}


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def challenge_specs(h, w, bg):
    """PREDECLARED grey-border challenge: levels placed relative to this image's own background."""
    k = max(4, int(round(0.06 * min(h, w))))
    k5 = max(4, int(round(0.18 * min(h, w))))
    c = lambda d: int(np.clip(round(bg + d), 0, 255))
    return [
        ("challenge_grey", "ch_grey_bg_m30_w2", (k, k, k, k), c(-30), False),
        ("challenge_grey", "ch_grey_bg_m10_w2", (k, k, k, k), c(-10), False),
        ("challenge_grey", "ch_grey_bg_p10_w2", (k, k, k, k), c(+10), False),
        ("challenge_grey", "ch_grey_bg_p30_w2", (k, k, k, k), c(+30), False),
        ("challenge_grey", "ch_grey_asym_w2", (k, 2 * k, 3 * k, k), c(+30), False),
        ("challenge_grey", "ch_grey_bg_p30_w5", (k5, k5, k5, k5), c(+30), False),
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


def diagnostics(fov, qc, ext, cbox, shape):
    """Section I: component-selection diagnostics, computed from the returned mask only."""
    H, W = shape
    if cbox is None:
        central = np.nan
        contacts = np.nan
    else:
        y0, y1, x0, x1 = cbox
        cy, cx = int((y0 + y1) // 2), int((x0 + x1) // 2)
        central = bool(fov[cy, cx]) if 0 <= cy < H and 0 <= cx < W else np.nan
        contacts = bool((fov & ndi.binary_dilation(ext, np.ones((3, 3), bool))).any()) \
            if ext.any() else False
    return {"in_padding_px": int((fov & ext).sum()) if ext.any() else 0,
            "contacts_padding": contacts, "central_retained": central,
            "ncomp_raw": int(qc.get("fov_n_components", -1)),
            "largest_frac": qc.get("fov_largest_frac", np.nan),
            "trimmed": bool(qc.get("fov_canvas_trimmed", False)),
            "valid": bool(qc.get("fov_valid", False)),
            "reason": str(qc.get("fov_failure_reason", "")),
            "coverage": float(qc.get("fov_coverage_fraction", np.nan))}


def worker(arg):
    image_path, mask_path, source, split, geom = arg[:5]
    only = arg[5] if len(arg) > 5 else None
    try:
        rgb, msk = load(image_path, mask_path)
        h0, w0 = msk.shape
        o1b, f1b, q1b = v2.measure_v1(rgb, msk, {})
        o2b, f2b, q2b = v2.measure_v2(rgb, msk, {})
        o3b, f3b, q3b = v3.measure_v3(rgb, msk, {})
        base = {"image_path": image_path, "source": source, "split": split, "geom": geom}
        for tag, o, f, q in (("v1", o1b, f1b, q1b), ("v2", o2b, f2b, q2b), ("v3", o3b, f3b, q3b)):
            base[f"{tag}_fov_px"] = int(f.sum())
            base[f"{tag}_valid"] = bool(q.get("fov_valid"))
            for fth in FEATS:
                base[f"{tag}_{fth}"] = o.get(fth, np.nan)
        base["drift_fov_dice_v3_v1"] = dice(f3b, f1b)
        base["drift_fov_dice_v2_v1"] = dice(f2b, f1b)
        for fth in FEATS:
            a, c = o1b.get(fth, np.nan), o3b.get(fth, np.nan)
            base[f"drift_{fth}_rel"] = ((c - a) / abs(a)
                                        if np.isfinite(a) and np.isfinite(c) and a else np.nan)
            base[f"drift_v2_{fth}_rel"] = ((o2b.get(fth, np.nan) - a) / abs(a)
                                           if np.isfinite(a) and a else np.nan)

        g = np.nan_to_num(np.asarray(rgb[:, :, 1], np.float32), nan=0.0)
        top, bottom, left, right, trimmed = v3.content_bounds(g)
        cbox = (top, h0 - bottom, left, w0 - right) if trimmed else None
        bg = float(np.median(g[top:h0 - bottom, left:w0 - right])) if trimmed else float(np.median(g))
        ext0 = v3.external_region((h0, w0), (top, bottom, left, right, trimmed))

        conds = [("brightness", f"x{c:.2f}", apply_brightness(rgb, msk, c)) for c in BRIGHTNESS]
        conds += [(s[0], s[1], apply_border(rgb, msk, s))
                  for s in (pad_specs(h0, w0) + novel_specs(h0, w0) + challenge_specs(h0, w0, bg))]
        rows, nbase = [], None
        for fam, cond, (prgb, pmsk, pad) in conds:
            t, b_, l, r_ = pad
            H, W = pmsk.shape
            extp = np.zeros((H, W), bool)
            if t or b_ or l or r_:
                if t:
                    extp[:t, :] = True
                if b_:
                    extp[H - b_:, :] = True
                if l:
                    extp[:, :l] = True
                if r_:
                    extp[:, W - r_:] = True
            o3, pf3, q3 = v3.measure_v3(prgb, pmsk, {})
            c3 = pf3[t:t + h0, l:l + w0]
            row = {"image_path": image_path, "source": source, "split": split, "geom": geom,
                   "family": fam, "condition": cond, "content_altering": cond in CONTENT_ALTERING,
                   "v3_dice": dice(c3, f3b),
                   "v3_rel_area": float((int(c3.sum()) - int(f3b.sum())) / max(int(f3b.sum()), 1)),
                   "v3_centroid": disp(c3, f3b, h0, w0), "v3_valid": bool(q3.get("fov_valid"))}
            d3 = diagnostics(pf3, q3, extp,
                            (t + cbox[0], t + cbox[1], l + cbox[2], l + cbox[3]) if cbox else None,
                            (H, W))
            row.update({f"v3_{k}": v for k, v in d3.items()})
            for fth in FEATS:
                a = o3b.get(fth, np.nan)
                p = o3.get(fth, np.nan)
                row[f"v3_{fth}"] = p
                row[f"v3_{fth}_base"] = a
                row[f"v3_{fth}_rel"] = ((p - a) / abs(a)
                                        if np.isfinite(a) and np.isfinite(p) and a else np.nan)
            if cond in MECH_SUBSET:
                o1, pf1, q1 = v2.measure_v1(prgb, pmsk, {})
                o2, pf2, q2 = v2.measure_v2(prgb, pmsk, {})
                c1, c2 = pf1[t:t + h0, l:l + w0], pf2[t:t + h0, l:l + w0]
                row.update({"v1_dice": dice(c1, f1b), "v2_dice": dice(c2, f2b),
                            "v1_valid": bool(q1.get("fov_valid")),
                            "v2_valid": bool(q2.get("fov_valid")),
                            "v1_rel_area": float((int(c1.sum()) - int(f1b.sum()))
                                                 / max(int(f1b.sum()), 1)),
                            "v2_rel_area": float((int(c2.sum()) - int(f2b.sum()))
                                                 / max(int(f2b.sum()), 1)),
                            "v1_centroid": disp(c1, f1b, h0, w0),
                            "v2_centroid": disp(c2, f2b, h0, w0)})
                for fth in FEATS:
                    a = o1b.get(fth, np.nan)
                    row[f"v1_{fth}_rel"] = ((o1.get(fth, np.nan) - a) / abs(a)
                                            if np.isfinite(a) and a else np.nan)
                    row[f"v2_{fth}_rel"] = ((o2.get(fth, np.nan) - a) / abs(a)
                                            if np.isfinite(a) and a else np.nan)
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
            "max_rel": float(r.max()),
            "frac_gt_1pct": float((r.abs() > .01).mean()),
            "frac_gt_2pct": float((r.abs() > .02).mean()),
            "frac_gt_5pct": float((r.abs() > .05).mean()),
            "frac_gt_10pct": float((r.abs() > .10).mean())}


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("A. FROZEN PRIOR VERSIONS")
    print("=" * 100)
    frozen = {"clinical_measurement_v1_sha256": sha(V1_MOD),
              "clinical_measurement_v2_candidate_sha256": sha(V2_MOD),
              "clinical_measurement_v3_candidate_sha256": sha(V3_MOD),
              "task5b_h3_manifest_sha256": sha(OUT / "task5b_h3_manifest.json"),
              "task5b_h3_summary_sha256": sha(OUT / "task5b_h3_summary.json"),
              "task5b_h3_rows_sha256": sha(OUT / "task5b_h3_rows.csv")}
    for k, val in frozen.items():
        print(f"  {k:44s} : {val}")

    print()
    print("B/C/D/M. MANIFEST WRITTEN BEFORE ANY LOCKED NULL")
    print("=" * 100)
    manifest = {
        "candidate": v3.MEASUREMENT_VERSION_V3,
        "inherited_unchanged": {
            "B1_threshold": {"rule": "Otsu on the frame with contiguous near-constant edge "
                                     "rows/columns removed, applied to the full frame",
                             "CONST_TOL": v3.CONST_TOL, "MIN_CONTENT_FRAC": v3.MIN_CONTENT_FRAC},
            "B2_fractal_domain": {"rule": "same MultifractalVBMs estimator on a square window around "
                                          "the FOV bbox centre, zero outside the frame",
                                  "DOMAIN_MARGIN": v3.DOMAIN_MARGIN, "DOMAIN_ALIGN": v3.DOMAIN_ALIGN},
        },
        "single_change_component_selection": {
            "order": ["1 content-box detection",
                      "2 threshold = Otsu(content box), fallback full frame if box < "
                      "MIN_CONTENT_FRAC",
                      "3 binimg = green > threshold on the FULL frame",
                      "4 record raw component count (V1 QC semantics)",
                      "5 if trimmed: binimg[external_region] = False   <-- the only change vs V2",
                      "6 remove_small_objects(max(64, 0.001*h*w))",
                      "7 remove_small_holes(max(64, 0.001*h*w))",
                      "8 binary_closing(disk(3))",
                      "9 label, keep largest -> fov",
                      "10 coverage = fov.mean() over the file frame; frag = largest/sum of the "
                      "labelled content components",
                      "11 failure criteria coverage < 0.15 | > 0.985 | frag < 0.90"],
            "fallback": "untrimmed frame -> no external region -> V3 selects as V2/V1",
            "failure_behaviour": "unchanged reasons, unchanged thresholds, no new constant"},
        "sample_rule": "all 8870 minus the 328 development and 450 H3 locked images, sorted by "
                       "image_path, first k per (source, geom, split) with the smallest k in "
                       "{12,16,20,25,30,40,60,80} reaching N >= 400",
        "conditions": {"brightness": 8, "legacy_border": 22, "novel_border": 6, "challenge_grey": 6,
                       "total": 42, "versions": {"v3": "all 42",
                                                 "v1_v2": f"mechanism subset {MECH_SUBSET}"}},
        "family_classification": {"PADDING_ONLY": "all padding conditions",
                                  "CONTENT_ALTERING": list(CONTENT_ALTERING),
                                  "note": "overwrite_* overwrite retinal pixels; reported "
                                          "separately, excluded from the plausible-border Dice floor"},
        "acceptance_gate": GATE,
        "stop_rule": "fail -> no tuning, no V4, no regeneration, no training",
    }
    (OUT / "task5b_h4_manifest.json").write_text(json.dumps(
        {"frozen": frozen, "manifest": manifest}, indent=2), encoding="utf-8")
    print("  manifest written: _private_audit/task5b_h4_manifest.json")
    print(f"  gate: {GATE}")

    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                       usecols=["image_path", "mask_path", "source", "split"])
    meta = meta.sort_values("image_path").reset_index(drop=True)
    meta["geom"] = [GEOM.get(Image.open(p).size, "other") for p in meta.image_path]
    dev = set(pd.read_csv(OUT / "task5b_h2_baseline.csv", usecols=["image_path"]).image_path)
    h3 = set(pd.read_csv(OUT / "task5b_h3_baseline.csv", usecols=["image_path"]).image_path)

    print()
    print("F. LOCKED SAMPLE")
    print("=" * 100)
    pool = meta[~meta.image_path.isin(dev | h3)]
    strat = ["source", "geom", "split"]
    sel, used_k = pool.groupby(strat, group_keys=False).head(12), 12
    for k in (16, 20, 25, 30, 40, 60, 80):
        if len(sel) >= 400:
            break
        sel, used_k = pool.groupby(strat, group_keys=False).head(k), k
    sel = sel.sort_values("image_path").head(900).reset_index(drop=True)
    print(f"  pool (8870 - 328 dev - 450 H3)  : {len(pool)}")
    print(f"  per-stratum cap used            : {used_k}")
    print(f"  LOCKED_SAMPLE_N                 : {len(sel)}")
    print(f"  disjoint from dev and H3        : {not (set(sel.image_path) & (dev | h3))}")
    print(f"  sources {sel.source.value_counts().to_dict()}")
    print(f"  geoms   {sel.geom.value_counts().to_dict()}")
    print(f"  splits  {sel.split.value_counts().to_dict()}")
    if len(sel) < 400 or (set(sel.image_path) & (dev | h3)):
        raise SystemExit("LOCKED_SAMPLE_INVALID")

    print()
    print("G/H. LOCKED RUN — V3 on 42 conditions, V1/V2 on the mechanism subset")
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
            if i % 20 == 0:
                pd.DataFrame(rows).to_csv(OUT / "task5b_h4_rows.csv", index=False)
                pd.DataFrame(brows).to_csv(OUT / "task5b_h4_baseline.csv", index=False)
                print(f"  {i}/{len(args)}  {time.time() - t0:.0f}s rows={len(rows)}", flush=True)
    B, C = pd.DataFrame(brows), pd.DataFrame(rows)
    B.to_csv(OUT / "task5b_h4_baseline.csv", index=False)
    C.to_csv(OUT / "task5b_h4_rows.csv", index=False)
    C["family"] = np.where(C.family == "brightness", "brightness", "border")
    cn, cb = C[C.family == "border"], C[C.family == "brightness"]
    print(f"  images {len(B)}  errors {errs}  rows {len(C)}  border {len(cn)}  brightness {len(cb)}")

    print()
    print("H. FOV ENDPOINTS — V3 on all conditions, V1/V2 on the mechanism subset")
    print("=" * 100)
    res, summ = {}, {}
    for tag in ("v3",):
        d = cn[f"{tag}_dice"].dropna()
        res[f"{tag}_median_dice"] = float(d.median())
        res[f"{tag}_p01_dice"] = float(d.quantile(.01))
        res[f"{tag}_min_dice"] = float(d.min())
        res[f"{tag}_frac_lt_099"] = float((d < 0.99).mean())
        res[f"{tag}_frac_lt_090"] = float((d < 0.90).mean())
        res[f"{tag}_rel_area_median"] = float(cn[f"{tag}_rel_area"].median())
        res[f"{tag}_rel_area_p95"] = float(cn[f"{tag}_rel_area"].quantile(.95))
        res[f"{tag}_centroid_median"] = float(cn[f"{tag}_centroid"].median())
        print(f"  V3 all-border : median Dice {d.median():.6f}  p01 {d.quantile(.01):.6f}  "
              f"min {d.min():.6f}  <0.99 {float((d < 0.99).mean()):.4f}  "
              f"<0.90 {float((d < 0.90).mean()):.4f}")
        print(f"                  rel area median {cn[f'{tag}_rel_area'].median():+.6f} "
              f"p95 {cn[f'{tag}_rel_area'].quantile(.95):+.6f}  "
              f"centroid median {cn[f'{tag}_centroid'].median():.6f}")
    sub = cn.dropna(subset=["v1_dice"])
    for tag in ("v1", "v2", "v3"):
        d = sub[f"{tag}_dice"].dropna()
        res[f"{tag}_subset_median_dice"] = float(d.median())
        res[f"{tag}_subset_p01_dice"] = float(d.quantile(.01))
        res[f"{tag}_subset_min_dice"] = float(d.min())
        print(f"  {tag.upper()} mech-subset (n={len(sub)}): median Dice {d.median():.6f}  "
              f"p01 {d.quantile(.01):.6f}  min {d.min():.6f}  <0.90 {float((d < 0.90).mean()):.4f}")
    for tag in ("v1", "v2", "v3"):
        j = sub.join(B.set_index("image_path")[f"{tag}_valid"].rename("bv"), on="image_path")
        vi, iv = int((j.bv & ~j[f"{tag}_valid"]).sum()), int((~j.bv & j[f"{tag}_valid"]).sum())
        res[f"{tag}_subset_valid_to_invalid"] = vi
        print(f"  {tag.upper()} mech-subset validity: valid->invalid {vi}  invalid->valid {iv}")
    j3 = cn.join(B.set_index("image_path")["v3_valid"].rename("bv"), on="image_path")
    res["v3_valid_to_invalid"] = int((j3.bv & ~j3.v3_valid).sum())
    res["v3_invalid_to_valid"] = int((~j3.bv & j3.v3_valid).sum())
    print(f"  V3 all-border validity: valid->invalid {res['v3_valid_to_invalid']}  "
          f"invalid->valid {res['v3_invalid_to_valid']}")
    plaus = cn[~cn.content_altering]
    conds = plaus.groupby("condition").v3_dice.median().sort_values()
    res["plausible_min_condition_median_dice"] = float(conds.min())
    print(f"  plausible-border conditions: {conds.size}; worst medians "
          f"{conds.head(4).round(6).to_dict()}")
    print(f"  content-altering (reported separately): "
          f"{cn[cn.content_altering].groupby('condition').v3_dice.median().round(6).to_dict()}")
    print(f"  by source:\n{cn.groupby('source').v3_dice.median().round(6).to_string()}")
    print(f"  by geometry:\n{cn.groupby('geom').v3_dice.median().round(6).to_string()}")
    grey = cn[cn.condition.str.contains("gray|grey", regex=True)]
    print(f"  GREY conditions (grey-40/90 equivalents + challenge): n={len(grey)}  "
          f"median Dice {grey.v3_dice.median():.6f}  min {grey.v3_dice.min():.6f}  "
          f"invalid {int((~grey.v3_valid).sum())}")
    res["grey_median_dice"] = float(grey.v3_dice.median())
    res["grey_invalid"] = int((~grey.v3_valid).sum())

    print()
    print("I. COMPONENT-SELECTION DIAGNOSTIC")
    print("=" * 100)
    print(f"  V3 rows with any FOV pixel inside the detected padding : "
          f"{int((cn.v3_in_padding_px > 0).sum())} of {len(cn)}")
    print(f"  V3 rows contacting padding : {int(cn.v3_contacts_padding.fillna(False).sum())}")
    print(f"  V3 rows losing the central retinal pixel : "
          f"{int((cn.v3_central_retained == False).sum())}")  # noqa: E712
    for tag in ("v1", "v2", "v3"):
        d = sub[f"{tag}_valid"]
        print(f"  mech-subset invalid rows {tag.upper()}: {int((~d).sum())} of {len(sub)}  "
              f"reasons on V3: "
              f"{sub[(sub.v3_valid == False)].v3_reason.value_counts().to_dict() if tag == 'v3' else ''}")  # noqa: E712
    v2fail = sub[sub.v2_valid == False]  # noqa: E712
    print(f"  V2 rows invalid on the mechanism subset : {len(v2fail)}")
    if len(v2fail):
        print(f"    their V3 validity: valid {int(v2fail.v3_valid.sum())} / invalid "
              f"{int((~v2fail.v3_valid).sum())}")
        print(f"    their V3 Dice median {v2fail.v3_dice.median():.6f} "
              f"min {v2fail.v3_dice.min():.6f}")
        res["v2_invalid_rows_on_subset"] = int(len(v2fail))
        res["v2_invalid_rows_v3_valid"] = int(v2fail.v3_valid.sum())
    res["v3_rows_with_padding_pixels_in_fov"] = int((cn.v3_in_padding_px > 0).sum())
    res["v3_rows_losing_central_pixel"] = int((cn.v3_central_retained == False).sum())  # noqa: E712

    print()
    print("J. FIVE PRIMARY FEATURES — V3")
    print("=" * 100)
    for label, g in (("BORDER", cn), ("BRIGHTNESS", cb)):
        for tag in ("v3",):
            print(f"  --- {label} / {tag.upper()} ---")
            for fth in FEATS:
                s = stats(g, f"{tag}_{fth}_rel")
                summ[f"{label}|{tag}|{fth}"] = s
                if s:
                    print(f"    {fth:20s} median {s['median_rel']:+.5f}  p95 {s['p95_rel']:+.5f}  "
                          f"max {s['max_rel']:+.5f}  >1% {s['frac_gt_1pct']:.4f}  "
                          f">2% {s['frac_gt_2pct']:.4f}  >5% {s['frac_gt_5pct']:.4f}  "
                          f">10% {s['frac_gt_10pct']:.4f}")
    for key in ("source", "geom"):
        for fth in FEATS:
            gg = cn.groupby(key)[f"v3_{fth}_rel"].apply(lambda s: float((s.abs() > .05).mean()))
            summ[f"subgroup|{key}|{fth}"] = {k: float(v) for k, v in gg.items()}
    for label, g in (("BORDER", cn), ("BRIGHTNESS", cb)):
        tot = 0
        for fth in FEATS:
            b = g[f"v3_{fth}_base"].notna()
            p = g[f"v3_{fth}"].notna()
            fin2nan = int((b & ~p).sum())
            tot += fin2nan
            summ[f"nan|{label}|v3|{fth}"] = {"finite_to_nan": fin2nan,
                                             "nan_to_finite": int((~b & p).sum())}
        print(f"  {label} V3 finite->NaN over the five features: {tot}")
    res["v3_border_new_nan"] = int(sum(v["finite_to_nan"] for k, v in summ.items()
                                       if k.startswith("nan|BORDER|v3|")))

    print()
    print("K. FRACTAL ZERO-PADDING CONTROL")
    print("=" * 100)
    Bc = B.merge(meta[["image_path", "mask_path"]], on="image_path", how="left")
    ct = canvas_control(Bc.sort_values("image_path").head(6))
    ct.to_csv(OUT / "task5b_h4_canvas_control.csv", index=False)
    for fth in ("fractal_d0", "fractal_d1", "fractal_d2"):
        r3 = ct[f"v3_{fth}_rel"].replace([np.inf, -np.inf], np.nan).dropna()
        r1 = ct[f"v1_{fth}_rel"].replace([np.inf, -np.inf], np.nan).dropna()
        res[f"canvas_v3_{fth}_median_abs_rel"] = float(r3.abs().median())
        res[f"canvas_v3_{fth}_max_abs_rel"] = float(r3.abs().max())
        print(f"  {fth:12s} V1 median|rel| {r1.abs().median():.5f}   "
              f"V3 median|rel| {r3.abs().median():.5f}  max {r3.abs().max():.5f}")

    print()
    print("L. NATIVE-IMAGE PRESERVATION")
    print("=" * 100)
    dd = B.drift_fov_dice_v3_v1.dropna()
    print(f"  N={len(B)}  native FOV Dice V3 vs V1: median {dd.median():.6f} "
          f"p05 {dd.quantile(.05):.6f} min {dd.min():.6f}")
    print(f"  native FOV Dice V2 vs V1: median {B.drift_fov_dice_v2_v1.median():.6f}")
    for fth in FEATS:
        r = B[f"drift_{fth}_rel"].replace([np.inf, -np.inf], np.nan).dropna()
        res[f"native_drift_{fth}_median_rel"] = float(r.median())
        print(f"  {fth:20s} V3 vs V1 median {r.median():+.5f}  p95|.| {r.abs().quantile(.95):.5f}"
              f"  frac>5% {float((r.abs() > .05).mean()):.4f}")
    lost = int((B.v1_valid & ~B.v3_valid).sum())
    res["native_valid_to_invalid"] = lost
    res["native_valid_to_invalid_frac"] = float(lost / max(len(B), 1))
    print(f"  native valid->invalid V1->V3: {lost} ({res['native_valid_to_invalid_frac']:.4f})"
          f"   invalid->valid {int((~B.v1_valid & B.v3_valid).sum())}")

    print()
    print("M. GATE")
    print("=" * 100)
    checks = {
        "G1_median_dice": res["v3_median_dice"] >= GATE["G1_median_dice_min"],
        "G2_p01_dice": res["v3_p01_dice"] >= GATE["G2_p01_dice_min"],
        "G3_plausible_condition": res["plausible_min_condition_median_dice"]
        >= GATE["G3_plausible_condition_median_dice_min"],
        "G4_valid_to_invalid": res["v3_valid_to_invalid"] <= GATE["G4_valid_to_invalid_max"],
        "G5_grey_component_failure": (res["grey_invalid"] / max(len(grey), 1))
        <= GATE["G5_grey_component_failure_frac_max"],
        "G6_vessel_density": summ["BORDER|v3|vessel_density_fov"]["frac_gt_1pct"]
        <= GATE["G6_vessel_density_frac_gt_1pct_max"],
        "G7_skel_density": summ["BORDER|v3|skel_density_fov"]["frac_gt_1pct"]
        <= GATE["G7_skel_density_frac_gt_1pct_max"],
        "G8_canvas_d0": res["canvas_v3_fractal_d0_median_abs_rel"]
        <= GATE["G8_fractal_zero_padding_abs_rel_max"],
        "G8_canvas_d1": res["canvas_v3_fractal_d1_median_abs_rel"]
        <= GATE["G8_fractal_zero_padding_abs_rel_max"],
        "G8_canvas_d2": res["canvas_v3_fractal_d2_median_abs_rel"]
        <= GATE["G8_fractal_zero_padding_abs_rel_max"],
        "G9_no_new_nan": res["v3_border_new_nan"] <= GATE["G9_new_nan_frac_max"] * len(cn),
        "G10_subgroups": all(max(gg.values()) <= GATE["G10_subgroup_frac_gt_5pct_max"]
                             for k, gg in summ.items()
                             if k.startswith("subgroup|")
                             and k.split("|")[-1] in ("vessel_density_fov", "skel_density_fov")),
        "G11_native_validity": res["native_valid_to_invalid_frac"]
        <= GATE["G11_native_valid_to_invalid_frac_max"],
    }
    for k, vv in checks.items():
        print(f"  {k:32s} : {'PASS' if vv else 'FAIL'}")
    npass = sum(checks.values())
    verdict = "PASS" if all(checks.values()) else ("CONCERN" if npass >= len(checks) - 2 else "FAIL")
    print()
    print(f"  checks passed {npass}/{len(checks)}  ->  FOV_BORDER_ROBUSTNESS_V3 = {verdict}")
    (OUT / "task5b_h4_summary.json").write_text(json.dumps(
        {"gate_checks": checks, "checks_passed": npass, "checks_total": len(checks),
         "FOV_BORDER_ROBUSTNESS_V3": verdict, "results": res, "stats": summ,
         "locked_N": int(len(B)), "rows": int(len(C)), "errors": errs,
         "manifest": manifest, "frozen": frozen}, indent=2, default=str), encoding="utf-8")
    print(f"  elapsed {time.time() - t0:.0f}s")


def canvas_control(selrow):
    """Identical binary support and FOV, translated into larger zero canvases; no FOV detection."""
    from src.biomarker import clinical_measurement_v1 as v1
    rows = []
    for r in selrow.itertuples():
        rgb, msk = load(r.image_path, r.mask_path)
        h0, w0 = msk.shape
        f1, _ = v1.retinal_fov(rgb[:, :, 1])
        sup = (msk.astype(bool) & f1)
        b1 = v1._fractal(sup)
        b3 = v3.fractal_v3(sup, f1)
        for spec in pad_specs(h0, w0):
            t, b, l, rr = spec[2]
            if t == b == l == rr == 0:
                continue
            ps = np.pad(sup, ((t, b), (l, rr)), constant_values=False)
            pf = np.pad(f1, ((t, b), (l, rr)), constant_values=False)
            p1, p3 = v1._fractal(ps), v3.fractal_v3(ps, pf)
            row = {"image_path": r.image_path, "condition": spec[1]}
            for fth in ("fractal_d0", "fractal_d1", "fractal_d2"):
                row[f"v1_{fth}_rel"] = ((p1[fth] - b1[fth]) / b1[fth]) if b1[fth] else np.nan
                row[f"v3_{fth}_rel"] = ((p3[fth] - b3[fth]) / b3[fth]) if b3[fth] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
