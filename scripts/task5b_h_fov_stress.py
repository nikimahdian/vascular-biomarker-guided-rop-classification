#!/usr/bin/env python
"""Task 5B-H closure: FOV robustness to brightness extremes and black-border styles.

Scope, fixed before any number was produced:

  * The FOV implementation is FROZEN. `src/biomarker/clinical_measurement_v1.py::retinal_fov`
    is imported and called; nothing in it is edited, wrapped or reimplemented. The threshold,
    the morphology, the component-selection rule and the three failure criteria are those of
    the frozen file, whose sha256 is recorded in the output.
  * `SEG_CURRENT_V1` is untouched: the vessel mask is loaded once per image and held FIXED
    across every perturbation, so any change in `vessel_density_fov` is FOV detection only.
  * Loading uses `scripts.task5b_n2.load`, the same production-faithful loader that
    reproduced the frozen table in Task 5B-N2, so the baseline is comparable to it.
  * No disease label is read, used for selection, or reported. No classifier is trained.
  * No FOV threshold is changed after seeing results. The class definitions of section K and
    the pass conditions of section O were written before the run.

Sections: B sample, C baseline, D brightness, E borders, F stability, G density consequence,
H/I mechanism, J QC montages, K/L/N verdict.
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
from skimage.filters import threshold_otsu

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.task5b_n2 import load  # identical production-faithful loading  # noqa: E402
from src.biomarker.clinical_measurement_v1 import retinal_fov  # frozen implementation  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
MODULE = ROOT / "src/biomarker/clinical_measurement_v1.py"
GEOM = {(640, 480): "640x480", (1280, 960): "1280x960", (1440, 1080): "1440x1080",
        (1600, 1200): "1600x1200", (1240, 1240): "1240x1240"}

# ------------------------------------------------------------------ PREDECLARED (section D/E/K)
BRIGHTNESS = [0.25, 0.50, 0.70, 0.85, 1.15, 1.30, 1.50, 1.80]
WIDTH_FRACS = [("w1", 0.02), ("w2", 0.06), ("w3", 0.12)]

# section K class definitions, predeclared:
#   ROBUST  : every perturbation keeps FOV Dice >= 0.99 and |relative density change| small,
#             with no condition showing a systematic failure pattern.
#   CONCERN : some conditions exceed that but stay bounded (Dice >= 0.90, density rel < 5%).
#   FAIL    : common plausible perturbations substantially alter the FOV (Dice < 0.90) and
#             therefore FINAL_PRIMARY density.
PREDECLARED_ROBUST_DICE = 0.99
PREDECLARED_CONCERN_DICE = 0.90
PREDECLARED_ROBUST_DENSITY_REL = 0.01
PREDECLARED_CONCERN_DENSITY_REL = 0.05

POOL_PER_STRATUM = 160
PER_BRIGHTNESS_BIN = 20
N_BINS = 3


def diag_components(g):
    """DIAGNOSTIC ONLY — post-morphology component count.

    `retinal_fov` records `fov_n_components` on the RAW binarisation, before small-object and
    small-hole removal, so it is very noisy and is not what selects the FOV. This helper re-applies
    the same skimage primitives only to expose the count that survives cleanup. It never decides
    the FOV: every FOV mask in this run comes from `retinal_fov()`.
    """
    from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects
    gg = np.nan_to_num(np.asarray(g, np.float32), nan=0.0)
    h, w = gg.shape
    try:
        t = float(threshold_otsu(gg))
    except Exception:  # noqa: BLE001
        t = float(np.percentile(gg, 50))
    b = gg > t
    m = max(64, int(0.001 * h * w))
    b = remove_small_objects(b, min_size=m)
    b = remove_small_holes(b, area_threshold=m)
    b = binary_closing(b, disk(3))
    return int(ndi.label(b)[1])


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


# ------------------------------------------------------------------ perturbations
def pad_specs(h: int, w: int):
    """Deterministic border conditions. Returns (family, name, (t,b,l,r), value, alters_content).

    Canvas growth preserves every original retinal pixel; the overwrite family paints black bars
    over the existing edge pixels and is reported separately because it alters content.
    """
    mn = min(h, w)
    out = []
    for wname, frac in WIDTH_FRACS:
        k = max(4, int(round(frac * mn)))
        out += [
            ("uniform_black", f"uniform_black_{wname}", (k, k, k, k), 0, False),
            ("asymmetric_black", f"asymmetric_black_{wname}", (2 * k, k, 3 * k, 2 * k), 0, False),
            ("lr_thick", f"lr_thick_{wname}", (k, k, 3 * k, 3 * k), 0, False),
            ("tb_thick", f"tb_thick_{wname}", (3 * k, 3 * k, k, k), 0, False),
            ("one_sided_left", f"one_sided_left_{wname}", (0, 0, 4 * k, 0), 0, False),
            ("irregular_frame", f"irregular_frame_{wname}", (k, 4 * k, 2 * k, 3 * k), 0, False),
        ]
    k2 = max(4, int(round(0.06 * mn)))
    out += [
        ("near_black", "near_black_5_w2", (k2, k2, k2, k2), 5, False),
        ("dark_gray", "dark_gray_40_w2", (k2, k2, k2, k2), 40, False),
        ("overwrite_black", "overwrite_lr_w2", (0, 0, k2, k2), 0, True),
        ("overwrite_black", "overwrite_all_w2", (k2, k2, k2, k2), 0, True),
    ]
    return out


def apply_border(rgb, msk, spec):
    t, b, l, r = spec[2]
    val = spec[3]
    if spec[4]:  # overwrite: paint over existing pixels, canvas unchanged
        rgb2 = rgb.copy()
        msk2 = msk.copy()
        if t:
            rgb2[:t, :, :] = val
        if b:
            rgb2[-b:, :, :] = val
        if l:
            rgb2[:, :l, :] = val
        if r:
            rgb2[:, -r:, :] = val
        return rgb2, msk2, (0, 0, 0, 0)
    pad = ((t, b), (l, r), (0, 0))
    rgb2 = np.pad(rgb, pad, mode="constant", constant_values=val)
    msk2 = np.pad(msk, ((t, b), (l, r)), mode="constant", constant_values=0)
    return rgb2, msk2, (t, b, l, r)


def apply_brightness(rgb, msk, c):
    return np.clip(rgb * c, 0.0, 255.0), msk, (0, 0, 0, 0)


# ------------------------------------------------------------------ measurement
def metrics(rgb, msk, pad, base_shape):
    """FOV + density as PRODUCTION would compute them on this exact image, plus frame-corrected
    comparisons against the baseline frame."""
    g = np.asarray(rgb[:, :, 1], np.float32)
    fov, qc = retinal_fov(g)                      # frozen implementation, unmodified
    h0, w0 = base_shape
    t, b, l, r = pad
    fp_pad = int(fov.sum())
    dens = float(msk[fov].sum() / fp_pad) if fp_pad else float("nan")
    try:
        thr = float(threshold_otsu(np.nan_to_num(g, nan=0.0)))
    except Exception:  # noqa: BLE001
        thr = float("nan")
    if fp_pad:
        cy, cx = ndi.center_of_mass(fov)
    else:
        cy = cx = float("nan")
    cropped = fov[t:t + h0, l:l + w0]
    fp_crop = int(cropped.sum())
    if fp_crop:
        ccy, ccx = ndi.center_of_mass(cropped)
    else:
        ccy = ccx = float("nan")
    hi = float((rgb >= 255.0).mean())
    lo = float((rgb <= 0.0).mean())
    return {
        "fov_px_padded": fp_pad,
        "fov_px_cropped": fp_crop,
        "coverage_as_production": float(fov.mean()),
        "n_components_raw": int(qc.get("fov_n_components", -1)),
        "n_components_clean": diag_components(g),
        "border_contact": float(qc.get("fov_border_contact", np.nan)),
        "centroid_y_pad": float(cy), "centroid_x_pad": float(cx),
        "centroid_y_crop": float(ccy), "centroid_x_crop": float(ccx),
        "fov_valid": bool(qc.get("fov_valid", False)),
        "fov_failure_reason": str(qc.get("fov_failure_reason", "")),
        "vessel_density_fov": dens,
        "otsu_threshold": thr,
        "sat_hi_frac": hi, "sat_lo_frac": lo,
        "fov_mask_cropped": cropped,
    }


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else float("nan")


def worker(arg):
    image_path, mask_path, source, split, geom = arg
    rows = []
    try:
        rgb, msk = load(image_path, mask_path)
        h0, w0 = msk.shape
        base = metrics(rgb, msk, (0, 0, 0, 0), (h0, w0))
        bfov = base.pop("fov_mask_cropped")
        bdens = base["vessel_density_fov"]
        brows = {"image_path": image_path, "source": source, "split": split, "geom": geom,
                 "work_h": h0, "work_w": w0, **base,
                 "native_mean_green": float(rgb[:, :, 1].mean())}
        for c in BRIGHTNESS:
            rgb2, msk2, pad = apply_brightness(rgb, msk, c)
            m = metrics(rgb2, msk2, pad, (h0, w0))
            f = m.pop("fov_mask_cropped")
            rows.append(_row("brightness", f"x{c:.2f}", image_path, source, split, geom,
                             (h0, w0), pad, m, f, bfov, base, bdens))
        for spec in pad_specs(h0, w0):
            rgb2, msk2, pad = apply_border(rgb, msk, spec)
            m = metrics(rgb2, msk2, pad, (h0, w0))
            f = m.pop("fov_mask_cropped")
            rows.append(_row(spec[0], spec[1], image_path, source, split, geom,
                             (h0, w0), pad, m, f, bfov, base, bdens))
        return {"baseline": brows, "rows": rows}
    except Exception as e:  # noqa: BLE001
        return {"_err": f"{type(e).__name__}: {e}", "image_path": image_path}


def _row(family, cond, image_path, source, split, geom, shape, pad, m, fov, bfov, base, bdens):
    h0, w0 = shape
    fp = m["fov_px_cropped"]
    bfp = int(bfov.sum())
    d = m["vessel_density_fov"]
    dep = (m["centroid_y_crop"] - base["centroid_y_crop"],
           m["centroid_x_crop"] - base["centroid_x_crop"])
    disp = float(np.hypot(dep[0], dep[1]) / max(h0, w0)) if np.isfinite(dep).all() else np.nan
    return {
        "family": family, "condition": cond, "image_path": image_path, "source": source,
        "split": split, "geom": geom,
        "fov_dice": dice(fov, bfov),
        "rel_area_change": float((fp - bfp) / bfp) if bfp else np.nan,
        "fov_px_cropped": fp,
        "coverage_as_production": m["coverage_as_production"],
        "centroid_disp_norm": disp,
        "n_components_raw": m["n_components_raw"],
        "component_change": int(m["n_components_raw"] != base["n_components_raw"]),
        "n_components_clean": m["n_components_clean"],
        "component_change_clean": int(m["n_components_clean"] != base["n_components_clean"]),
        "border_contact": m["border_contact"],
        "fov_valid": m["fov_valid"],
        "fov_failure_reason": m["fov_failure_reason"],
        "vessel_density_fov": d,
        "density_delta": float(d - bdens) if np.isfinite(d) and np.isfinite(bdens) else np.nan,
        "density_rel_delta": (float((d - bdens) / bdens)
                              if np.isfinite(d) and np.isfinite(bdens) and bdens else np.nan),
        "otsu_threshold": m["otsu_threshold"],
        "sat_hi_frac": m["sat_hi_frac"], "sat_lo_frac": m["sat_lo_frac"],
    }


# ------------------------------------------------------------------ main
def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("A. FROZEN FOV IMPLEMENTATION")
    print("=" * 100)
    print(f"  file            : src/biomarker/clinical_measurement_v1.py")
    print(f"  function        : retinal_fov(green) -> (bool mask, qc dict)")
    print(f"  file sha256     : {sha(MODULE)}")
    print(f"  rule            : thr = threshold_otsu(green); binimg = green > thr")
    print(f"                    remove_small_objects / remove_small_holes "
          f"(min_size = max(64, 0.001*h*w)); binary_closing(disk(3)); keep largest component")
    print(f"  failures        : coverage < 0.15 | coverage > 0.985 | largest/sum < 0.90")
    print(f"  vessel mask     : SEG_CURRENT_V1, loaded once per image and held FIXED")
    print(f"  feature         : vessel_density_fov = vessel_mask_px_in_fov / fov_px")

    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                       usecols=["image_path", "mask_path", "source", "split"])
    print()
    print("=" * 100)
    print("B. DETERMINISTIC PROJECT SAMPLE — stratified by source x geometry x brightness")
    print("=" * 100)
    meta = meta.sort_values("image_path").reset_index(drop=True)
    meta["geom"] = [GEOM.get(Image.open(p).size, "other") for p in meta.image_path]
    pool = meta.groupby(["source", "geom"], group_keys=False).head(POOL_PER_STRATUM).copy()
    print(f"  pool (first {POOL_PER_STRATUM} per source x geom by path order) : {len(pool)}")
    bright = []
    for p in pool.image_path:
        with Image.open(p) as im:
            small = im.convert("L").resize((48, 48), Image.BILINEAR)
        bright.append(float(np.asarray(small, np.float32).mean()))
    pool["brightness"] = bright
    sel = []
    for (src, gm), gp in pool.groupby(["source", "geom"]):
        gp = gp.sort_values("brightness")
        bins = np.array_split(gp, N_BINS)
        for bn in bins:
            sel.append(bn.head(PER_BRIGHTNESS_BIN))
    sel = pd.concat(sel).drop_duplicates("image_path").sort_values("image_path").reset_index(drop=True)
    print(f"  AV/FOV SAMPLE_N : {len(sel)}")
    print(f"  sources         : {sel.source.value_counts().to_dict()}")
    print(f"  geometries      : {sel.geom.value_counts().to_dict()}")
    print(f"  splits          : {sel.split.value_counts().to_dict()}")
    q = sel.brightness.quantile([0, .25, .5, .75, 1]).round(2).to_dict()
    print(f"  native brightness (grey mean, 48px) min/p25/median/p75/max : {q}")
    print(f"  selection used : source, geometry, native brightness, image_path order ONLY")
    print(f"  selection did NOT use : label, split outcome, FOV outcome, any biomarker")

    print()
    print("=" * 100)
    print("C/D/E. BASELINE, BRIGHTNESS AND BORDER CONDITIONS")
    print("=" * 100)
    args = list(zip(sel.image_path, sel.mask_path, sel.source, sel.split, sel.geom))
    base_rows, cond_rows, errs = [], [], 0
    with get_context("fork").Pool(12) as pool_:
        for i, r in enumerate(pool_.imap_unordered(worker, args, chunksize=2), 1):
            if r.get("_err"):
                errs += 1
                print(f"  ERR {r['image_path']}: {r['_err']}")
                continue
            base_rows.append(r["baseline"])
            cond_rows.extend(r["rows"])
            if i % 60 == 0:
                print(f"  {i}/{len(args)}  {time.time() - t0:.0f}s", flush=True)
    B = pd.DataFrame(base_rows)
    C = pd.DataFrame(cond_rows)
    B.to_csv(ROOT / "_private_audit/task5b_h_baseline.csv", index=False)
    C.to_csv(ROOT / "_private_audit/task5b_h_conditions.csv", index=False)
    n_bright = C[C.family == "brightness"].condition.nunique()
    n_border = C[C.family != "brightness"].condition.nunique()
    print(f"  images measured        : {len(B)}   errors: {errs}")
    print(f"  brightness conditions  : {n_bright}")
    print(f"  border conditions      : {n_border}")
    print(f"  condition rows         : {len(C)}")
    print(f"  baseline FOV valid     : {int(B.fov_valid.sum())} / {len(B)}")
    if (~B.fov_valid).any():
        print(f"  baseline failure reasons: {B[~B.fov_valid].fov_failure_reason.value_counts().to_dict()}")
    print(f"  baseline coverage      : median {B.coverage_as_production.median():.4f}  "
          f"min {B.coverage_as_production.min():.4f}  max {B.coverage_as_production.max():.4f}")
    print(f"  baseline components    : median {B.n_components_raw.median():.0f}  "
          f"max {B.n_components_raw.max():.0f}")
    print(f"  baseline border contact: median {B.border_contact.median():.4f}")

    frozen = pd.read_csv(ROOT / "data/features/clinical_measurement_v1.csv",
                         usecols=["image_path", "vessel_density_fov", "fov_valid",
                                  "fov_coverage_fraction"])
    MM = frozen.merge(B[["image_path", "vessel_density_fov", "fov_valid",
                         "coverage_as_production"]], on="image_path", suffixes=("_frozen", "_here"))
    ctrl_d = float(np.nanmax(np.abs(MM.vessel_density_fov_frozen - MM.vessel_density_fov_here)))
    ctrl_v = int((MM.fov_valid_frozen != MM.fov_valid_here).sum())
    ctrl_c = float(np.nanmax(np.abs(MM.fov_coverage_fraction - MM.coverage_as_production)))
    print()
    print(f"  BASELINE_DENSITY_CONTROL_MAX_DELTA  = {ctrl_d:.3e}")
    print(f"  BASELINE_VALIDITY_MISMATCHES        = {ctrl_v} / {len(MM)}")
    print(f"  BASELINE_COVERAGE_CONTROL_MAX_DELTA = {ctrl_c:.3e}")

    print()
    print("=" * 100)
    print("F. FOV STABILITY — overall, by condition")
    print("=" * 100)
    print(f"  {'condition':28s} {'median Dice':>11s} {'min Dice':>9s} {'med relArea':>12s} "
          f"{'med centDisp':>13s} {'compChg':>8s} {'invalid':>8s}")
    for cond, g in C.groupby("condition"):
        print(f"  {cond:28s} {g.fov_dice.median():11.4f} {g.fov_dice.min():9.4f} "
              f"{g.rel_area_change.median():12.5f} {g.centroid_disp_norm.median():13.5f} "
              f"{g.component_change.mean():8.4f} {float((~g.fov_valid).mean()):8.4f}")
    for fam, g in C.groupby("family"):
        print(f"  FAMILY {fam:20s} median Dice {g.fov_dice.median():.4f}  min {g.fov_dice.min():.4f}"
              f"  images <0.99 {float((g.fov_dice < 0.99).mean()):.4f}"
              f"  <0.90 {float((g.fov_dice < 0.90).mean()):.4f}")
    cb = C[C.family == "brightness"]
    cn = C[C.family != "brightness"]
    print()
    print(f"  BRIGHTNESS  median Dice {cb.fov_dice.median():.6f}  min {cb.fov_dice.min():.6f}")
    print(f"  BORDER      median Dice {cn.fov_dice.median():.6f}  min {cn.fov_dice.min():.6f}")

    # The feature is only defined where the baseline FOV itself is valid. Images whose baseline
    # detection already failed (or is pathologically small) are not a robustness question, so the
    # primary endpoint is also reported on the baseline-valid subset. Both populations are shown;
    # no threshold is changed.
    C = C.merge(B[["image_path", "fov_valid", "coverage_as_production", "n_components_raw"]]
                .rename(columns={"fov_valid": "base_valid",
                                 "coverage_as_production": "base_coverage",
                                 "n_components_raw": "base_components_raw"}),
                on="image_path", how="left")
    Cv = C[C.base_valid]
    cbv = Cv[Cv.family == "brightness"]
    cnv = Cv[Cv.family != "brightness"]
    print()
    print(f"  baseline-VALID subset : {Cv.image_path.nunique()} images, {len(Cv)} rows")
    print(f"    BRIGHTNESS median Dice {cbv.fov_dice.median():.6f}  min {cbv.fov_dice.min():.6f}")
    print(f"    BORDER     median Dice {cnv.fov_dice.median():.6f}  min {cnv.fov_dice.min():.6f}")
    print()
    print("  worst 10 (image | condition | Dice | baseline coverage/valid | density rel delta):")
    for r in C.sort_values("fov_dice").head(10).itertuples():
        print(f"    {r.image_path.split('/')[-1][:46]:46s} {r.condition:22s} "
              f"dice={r.fov_dice:.4f} base_cov={r.base_coverage:.4f} "
              f"base_valid={r.base_valid} dens_rel={r.density_rel_delta:+.4f}")

    print()
    print("  by source:")
    print(C.groupby(["family", "source"]).fov_dice.median().unstack().round(5).to_string())
    print("  by geometry:")
    print(C.groupby(["family", "geom"]).fov_dice.median().unstack().round(5).to_string())

    print()
    print("=" * 100)
    print("G. PRIMARY FEATURE CONSEQUENCE — vessel_density_fov, fixed vessel mask")
    print("=" * 100)

    def dens_block(g, label):
        r = g.density_rel_delta.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {label:22s} n={len(r):6d}  median={r.median():+.5f}  p95={r.quantile(.95):+.5f}  "
              f"max={r.max():+.5f}  min={r.min():+.5f}")
        for t in (0.01, 0.02, 0.05, 0.10):
            print(f"      frac |rel delta| > {t:4.0%} : {float((r.abs() > t).mean()):.4f}")

    dens_block(cb, "BRIGHTNESS (all)")
    dens_block(cn, "BORDER (all)")
    print()
    dens_block(cbv, "BRIGHTNESS (base-valid)")
    dens_block(cnv, "BORDER (base-valid)")
    print()
    for fam, g in C.groupby("family"):
        r = g.density_rel_delta.dropna()
        print(f"  {fam:20s} median={r.median():+.5f} p95={r.quantile(.95):+.5f} max={r.max():+.5f}")

    print()
    print("=" * 100)
    print("H. BORDER MECHANISM")
    print("=" * 100)
    for cond, g in cn.groupby("condition"):
        print(f"  {cond:28s} dThr={g.otsu_threshold.median() - B.otsu_threshold.median():+8.3f}"
              f"  dCov={g.coverage_as_production.median() - B.coverage_as_production.median():+8.4f}"
              f"  borderContact={g.border_contact.median():.4f}"
              f"  compChg_clean={g.component_change_clean.mean():.3f}"
              f"  Dice={g.fov_dice.median():.4f}")

    print()
    print("=" * 100)
    print("I. BRIGHTNESS MECHANISM")
    print("=" * 100)
    Bm = B.set_index("image_path")
    for c in BRIGHTNESS:
        g = cb[cb.condition == f"x{c:.2f}"].set_index("image_path")
        j = g.join(Bm[["otsu_threshold", "vessel_density_fov"]], rsuffix="_b")
        ratio = (j.otsu_threshold / (j.otsu_threshold_b * c)).replace([np.inf, -np.inf], np.nan)
        print(f"  x{c:.2f}  Dice_median={j.fov_dice.median():.6f}  Dice_min={j.fov_dice.min():.6f}"
              f"  thr/scale_expected={ratio.median():.4f}"
              f"  sat_hi={j.sat_hi_frac.median():.4f}  sat_lo={j.sat_lo_frac.median():.4f}"
              f"  relDens_median={j.density_rel_delta.median():+.5f}")

    print()
    print("=" * 100)
    print("J. VISUAL QC SELECTION (no labels used)")
    print("=" * 100)
    worst = {}
    for fam, g in C.groupby("family"):
        worst[fam] = g.sort_values("fov_dice").head(3)[["image_path", "condition", "fov_dice"]]
        print(f"  {fam:20s} " + "; ".join(
            f"{r.image_path.split('/')[-1]}|{r.condition}|dice={r.fov_dice:.3f}"
            for r in worst[fam].itertuples()))
    stable = C[(C.fov_dice >= 0.9999)].image_path.drop_duplicates().head(6).tolist()
    print(f"  stable cases: {[p.split('/')[-1] for p in stable]}")

    print()
    print("=" * 100)
    print("K/L. VERDICT")
    print("=" * 100)
    allg = Cv
    worst_dice = float(allg.fov_dice.min())
    worst_dens = float(allg.density_rel_delta.abs().max())
    worst_dice_all = float(C.fov_dice.min())
    worst_dens_all = float(C.density_rel_delta.abs().max())
    med_dice_b, med_dice_n = float(cbv.fov_dice.median()), float(cnv.fov_dice.median())
    if worst_dice >= PREDECLARED_ROBUST_DICE and worst_dens <= PREDECLARED_ROBUST_DENSITY_REL:
        verdict = "ROBUST"
    elif worst_dice >= PREDECLARED_CONCERN_DICE and worst_dens <= PREDECLARED_CONCERN_DENSITY_REL:
        verdict = "CONCERN"
    else:
        verdict = "FAIL"
    print(f"  worst FOV Dice over all conditions (baseline-valid) : {worst_dice:.6f}")
    print(f"  worst FOV Dice over all conditions (all images)     : {worst_dice_all:.6f}")
    print(f"  worst |relative density delta| (baseline-valid)     : {worst_dens:.6f}")
    print(f"  worst |relative density delta| (all images)         : {worst_dens_all:.6f}")
    print(f"  FOV_ROBUSTNESS                     : {verdict}")
    fpr = "RETAIN" if verdict in ("ROBUST", "CONCERN") else "REQUIRES_REVIEW"
    print(f"  VESSEL_DENSITY_FOV_FINAL_PRIMARY   : {fpr}")
    print(f"  FINAL_PRIMARY_REQUIRES_REVIEW      : {'NO' if fpr == 'RETAIN' else 'YES'}")

    out = {
        "FOV_STRESS_TEST_EXECUTED": "YES",
        "module_sha256": sha(MODULE),
        "FOV_SAMPLE_N": int(len(B)),
        "BRIGHTNESS_CONDITIONS_N": int(n_bright),
        "BORDER_CONDITIONS_N": int(n_border),
        "BASELINE_FOV_VALID_N": int(B.fov_valid.sum()),
        "BASELINE_DENSITY_CONTROL_MAX_DELTA": ctrl_d,
        "BASELINE_VALIDITY_MISMATCHES": ctrl_v,
        "BASELINE_COVERAGE_CONTROL_MAX_DELTA": ctrl_c,
        "BASELINE_VALID_SUBSET_N": int(Cv.image_path.nunique()),
        "BASELINE_VALID_SUBSET_ROWS": int(len(Cv)),
        "BRIGHTNESS_MEDIAN_FOV_DICE": med_dice_b,
        "BRIGHTNESS_MIN_FOV_DICE": float(cbv.fov_dice.min()),
        "BORDER_MEDIAN_FOV_DICE": med_dice_n,
        "BORDER_MIN_FOV_DICE": float(cnv.fov_dice.min()),
        "BRIGHTNESS_MEDIAN_DENSITY_REL_DELTA": float(cbv.density_rel_delta.median()),
        "BRIGHTNESS_P95_DENSITY_REL_DELTA": float(cbv.density_rel_delta.quantile(.95)),
        "BRIGHTNESS_MAX_DENSITY_REL_DELTA": float(cbv.density_rel_delta.max()),
        "BORDER_MEDIAN_DENSITY_REL_DELTA": float(cnv.density_rel_delta.median()),
        "BORDER_P95_DENSITY_REL_DELTA": float(cnv.density_rel_delta.quantile(.95)),
        "BORDER_MAX_DENSITY_REL_DELTA": float(cnv.density_rel_delta.max()),
        "WORST_FOV_DICE_ALL_IMAGES": worst_dice_all,
        "WORST_DENSITY_REL_DELTA_ALL_IMAGES": worst_dens_all,
        "WORST_FOV_DICE_BASELINE_VALID": worst_dice,
        "WORST_DENSITY_REL_DELTA_BASELINE_VALID": worst_dens,
        "FOV_FAILURES_BASELINE": int((~B.fov_valid).sum()),
        "FOV_FAILURES_PERTURBED": int((~C.fov_valid).sum()),
        "FOV_ROBUSTNESS": verdict,
        "VESSEL_DENSITY_FOV_FINAL_PRIMARY": fpr,
        "FINAL_PRIMARY_REQUIRES_REVIEW": "NO" if fpr == "RETAIN" else "YES",
        "PREDECLARED": {"robust_dice": PREDECLARED_ROBUST_DICE,
                        "concern_dice": PREDECLARED_CONCERN_DICE,
                        "robust_density_rel": PREDECLARED_ROBUST_DENSITY_REL,
                        "concern_density_rel": PREDECLARED_CONCERN_DENSITY_REL},
        "by_family": {f: {"median_dice": float(g.fov_dice.median()),
                          "min_dice": float(g.fov_dice.min()),
                          "median_density_rel": float(g.density_rel_delta.median()),
                          "max_density_rel": float(g.density_rel_delta.max()),
                          "invalid_frac": float((~g.fov_valid).mean())}
                      for f, g in C.groupby("family")},
        "by_condition": {c: {"median_dice": float(g.fov_dice.median()),
                             "min_dice": float(g.fov_dice.min()),
                             "median_density_rel": float(g.density_rel_delta.median()),
                             "max_density_rel": float(g.density_rel_delta.max())}
                         for c, g in C.groupby("condition")},
        "by_source": {f"{f}|{s}": float(g.fov_dice.median())
                      for (f, s), g in C.groupby(["family", "source"])},
        "by_geom": {f"{f}|{g_}": float(g.fov_dice.median())
                    for (f, g_), g in C.groupby(["family", "geom"])},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (ROOT / "_private_audit/task5b_h_summary.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n  elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
