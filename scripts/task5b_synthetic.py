#!/usr/bin/env python
"""Task 5B synthetic audit: D tortuosity, E width, G density, I scale, J topology,
K fractal, L regional, Q aspect-ratio consequence.

All figures are produced on synthetic vessels whose geometry is known analytically, so a
measured value can be compared with an expected value. No label, no classifier, no AUC.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, skeletonize

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.biomarker.clinical_measurement_v1 import (  # noqa: E402
    MIN_ROI_COVERAGE, _fractal, _topology, measure, retinal_fov,
)
def rnd(d, n=4):
    return {k: round(v, n) for k, v in d.items()}


from src.biomarker.geometry_core import (  # noqa: E402
    branch_tortuosity, branch_tortuosity_pixelcount, branch_tortuosity_smoothed, fov_density,
    retinal_fov as gc_fov, width_px,
)

ROOT = "/Users/moniaz/niki"
OUT = f"{ROOT}/_private_audit"


# ----------------------------------------------------------------- synthetic geometry
def disk_at(r: float) -> np.ndarray:
    return disk(int(r))


def rasterise(shape, points, width: int) -> np.ndarray:
    """Draw a vessel of the requested width by dilating a centreline."""
    m = np.zeros(shape, bool)
    for y, x in points:
        iy, ix = int(round(y)), int(round(x))
        if 0 <= iy < shape[0] and 0 <= ix < shape[1]:
            m[iy, ix] = True
    if not m.any():
        return m
    return ndi.binary_dilation(m, disk_at(width / 2.0))


def straight(shape, angle_deg: float, width: int, length: int | None = None) -> np.ndarray:
    h, w = shape
    L = length or int(0.7 * min(h, w))
    cy, cx = h / 2, w / 2
    t = np.linspace(-L / 2, L / 2, 4 * L)
    a = math.radians(angle_deg)
    return rasterise(shape, list(zip(cy + t * math.sin(a), cx + t * math.cos(a))), width)


def arc(shape, radius: float, span_deg: float, width: int, offset_deg: float = 0.0) -> np.ndarray:
    h, w = shape
    cy, cx = h / 2, w / 2
    t = np.radians(np.linspace(-span_deg / 2, span_deg / 2, int(6 * abs(span_deg)) + 8))
    t = t + math.radians(offset_deg)
    return rasterise(shape, list(zip(cy + radius * np.sin(t), cx + radius * np.cos(t))), width)


def sine(shape, amp: float, period: float, width: int) -> np.ndarray:
    h, w = shape
    L = int(0.7 * w)
    x = np.linspace(-L / 2, L / 2, 8 * L)
    y = amp * np.sin(2 * math.pi * x / period)
    return rasterise(shape, list(zip(h / 2 + y, w / 2 + x)), width)


def y_junction(shape, width: int) -> np.ndarray:
    h, w = shape
    cy, cx = h / 2, w / 2
    pts = []
    t = np.linspace(0, 0.3 * min(h, w), 200)
    pts += list(zip(cy + t, cx + 0 * t))
    for s in (-1, 1):
        pts += list(zip(cy + 0.3 * min(h, w) + t * 0.7, cx + s * t))
    return rasterise(shape, pts, width)


def vignette(shape, vessel: np.ndarray, pad: int, bg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.full((*shape, 3), bg, np.float32)
    rgb[vessel] = 0.35
    return rgb, vessel.copy()


# ----------------------------------------------------------------- analytic expectations
def tort_arc_analytic(span_deg: float) -> float:
    th = math.radians(abs(span_deg))
    return th / (2 * math.sin(th / 2)) if th else 1.0


def tort_sine_analytic(amp: float, period: float, length: float) -> float:
    x = np.linspace(-length / 2, length / 2, 200001)
    dy = amp * (2 * math.pi / period) * np.cos(2 * math.pi * x / period)
    arc_len = float(np.trapezoid(np.sqrt(1 + dy ** 2), x)) if hasattr(np, "trapezoid") \
        else float(np.trapz(np.sqrt(1 + dy ** 2), x))
    chord = float(np.hypot(length, amp * math.sin(2 * math.pi * (length / 2) / period)
                           - amp * math.sin(2 * math.pi * (-length / 2) / period)))
    return arc_len / chord


def tort_of(mask: np.ndarray) -> float:
    t = branch_tortuosity(mask)
    return float(np.median(t)) if len(t) else float("nan")


def tort_sm_of(mask: np.ndarray) -> float:
    t = branch_tortuosity_smoothed(mask)
    return float(np.median(t)) if len(t) else float("nan")


def tort_pc_of(mask: np.ndarray) -> float:
    t = branch_tortuosity_pixelcount(mask)
    return float(np.median(t)) if len(t) else float("nan")


def main() -> None:
    S = (420, 420)
    results: dict = {}
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    # ================================================================ D
    say("=" * 100)
    say("D. TORTUOSITY Ã¢â‚¬â€ FROZEN GEODESIC DEFINITION, SYNTHETIC CORRECTNESS")
    say("=" * 100)
    say("  definition: skeleton -> remove junctions -> per-branch 8-connected geodesic path")
    say("              length (orthogonal 1, diagonal sqrt(2)); chord = Euclidean distance")
    say("              between the two degree-1 endpoints; T = arc/chord")
    say()
    say(f"  {'case':28s} {'expected':>10s} {'geodesic':>10s} {'smoothed':>10s} "
        f"{'abs err':>10s} {'pixelcount':>11s} {'pc err':>10s}")
    rows = []
    cases = [("horizontal line", 1.0, straight(S, 0.0, 9)),
             ("vertical line", 1.0, straight(S, 90.0, 9)),
             ("15 deg", 1.0, straight(S, 15.0, 9)),
             ("30 deg", 1.0, straight(S, 30.0, 9)),
             ("45 deg", 1.0, straight(S, 45.0, 9)),
             ("60 deg", 1.0, straight(S, 60.0, 9)),
             ("75 deg", 1.0, straight(S, 75.0, 9))]
    for nm, exp, m in cases:
        g, p, s = tort_of(m), tort_pc_of(m), tort_sm_of(m)
        rows.append({"case": nm, "expected": exp, "geodesic": g, "smoothed": s,
                     "pixelcount": p, "abs_err": abs(s - exp), "pc_err": abs(p - exp),
                     "geo_err": abs(g - exp)})
        say(f"  {nm:28s} {exp:10.4f} {g:10.4f} {s:10.4f} {abs(s - exp):10.4f} "
            f"{p:11.4f} {abs(p - exp):10.4f}")
    for span in (30, 60, 90, 120, 180):
        m = arc(S, 90.0, span, 9)
        exp = tort_arc_analytic(span)
        g, p, s = tort_of(m), tort_pc_of(m), tort_sm_of(m)
        nm = f"circular arc {span} deg"
        rows.append({"case": nm, "expected": exp, "geodesic": g, "smoothed": s,
                     "pixelcount": p, "abs_err": abs(s - exp), "pc_err": abs(p - exp),
                     "geo_err": abs(g - exp)})
        say(f"  {nm:28s} {exp:10.4f} {g:10.4f} {s:10.4f} {abs(s - exp):10.4f} "
            f"{p:11.4f} {abs(p - exp):10.4f}")
    for amp, per in ((20, 160), (40, 160), (60, 200)):
        m = sine(S, amp, per, 9)
        exp = tort_sine_analytic(amp, per, 0.7 * S[1])
        g, p, s = tort_of(m), tort_pc_of(m), tort_sm_of(m)
        nm = f"sine amp{amp} per{per}"
        rows.append({"case": nm, "expected": exp, "geodesic": g, "smoothed": s,
                     "pixelcount": p, "abs_err": abs(s - exp), "pc_err": abs(p - exp),
                     "geo_err": abs(g - exp)})
        say(f"  {nm:28s} {exp:10.4f} {g:10.4f} {s:10.4f} {abs(s - exp):10.4f} "
            f"{p:11.4f} {abs(p - exp):10.4f}")
    yj = y_junction(S, 9)
    say(f"  {'Y junction':28s} {'n/a':>10s} {tort_of(yj):10.4f} {tort_sm_of(yj):10.4f} "
        f"{'':>10s} {tort_pc_of(yj):11.4f}")
    rows.append({"case": "Y junction", "expected": None, "geodesic": tort_of(yj),
                 "smoothed": tort_sm_of(yj), "pixelcount": tort_pc_of(yj),
                 "abs_err": None, "pc_err": None, "geo_err": None})
    short = rasterise(S, [(S[0] / 2 + i, S[1] / 2) for i in range(5)], 5)
    n_short = len(branch_tortuosity(short))
    say(f"  {'short fragment (5 px)':28s} {'excluded':>10s} "
        f"{'n=' + str(n_short):>10s}")
    rows.append({"case": "short fragment", "expected": None,
                 "geodesic": None, "smoothed": None, "pixelcount": None, "abs_err": None,
                 "pc_err": None, "geo_err": None, "n_branches_geodesic": n_short})

    straight_rows = [r for r in rows if r["case"] in
                     ("horizontal line", "vertical line", "15 deg", "30 deg", "45 deg",
                      "60 deg", "75 deg")]
    gvals = [r["geodesic"] for r in straight_rows]
    svals = [r["smoothed"] for r in straight_rows]
    pvals = [r["pixelcount"] for r in straight_rows]
    disp_g = max(gvals) - min(gvals)
    disp_s = max(svals) - min(svals)
    disp_p = max(pvals) - min(pvals)
    say()
    say("  STRAIGHT-LINE ORIENTATION DISPERSION (max-min over 7 orientations)")
    say(f"    smoothed geodesic (FROZEN V1 definition) : {disp_s:.4f}")
    say(f"    raw geodesic                             : {disp_g:.4f}")
    say(f"    pixel-count (historical, rejected)       : {disp_p:.4f}")
    say()
    max_arc_err = max(abs(r["abs_err"]) for r in rows
                      if r["case"].startswith("circular") or r["case"].startswith("sine"))
    max_geo_err = max(abs(r["geo_err"]) for r in rows
                      if r["case"].startswith("circular") or r["case"].startswith("sine"))
    say(f"  max |error| vs analytic, smoothed : {max_arc_err:.4f}")
    say(f"  max |error| vs analytic, raw geodesic : {max_geo_err:.4f}")
    say(f"  all values >= 1 (tolerance 1e-6)  : "
        f"{all(r['smoothed'] is None or r['smoothed'] >= 1 - 1e-6 for r in rows)}")
    results["D"] = {
        "rows": rows,
        "straight_orientation_dispersion_smoothed": disp_s,
        "straight_orientation_dispersion_geodesic": disp_g,
        "straight_orientation_dispersion_pixelcount": disp_p,
        "max_analytic_error_smoothed": max_arc_err,
        "max_analytic_error_geodesic": max_geo_err,
        "all_at_least_one": bool(all(r["smoothed"] is None or r["smoothed"] >= 1 - 1e-6
                                     for r in rows)),
        "VERDICT": "PASS" if disp_s < 0.01 and max_arc_err < 0.05 else "FAIL",
    }
    say(f"  TORTUOSITY_INVARIANCE = {results['D']['VERDICT']}")

    # ================================================================ E
    say()
    say("=" * 100)
    say("E. WIDTH Ã¢â‚¬â€ END-TO-END, WITH QUANTISATION FLOOR")
    say("=" * 100)
    say(f"  {'true w':>7s} {'angle':>6s} {'measured p50':>13s} {'rel err':>9s} "
        f"{'w/DD':>8s} {'expected w/DD':>14s}")
    wrows = []
    for tw in (5, 9, 15, 21, 31):
        for ang in (0, 30, 45, 60, 90):
            m = straight(S, float(ang), tw)
            wx = width_px(m)
            meas = float(np.median(wx)) if len(wx) else float("nan")
            dd = 40.0
            wrows.append({"true_width_px": tw, "angle_deg": ang, "measured_p50_px": meas,
                          "rel_err": (meas - tw) / tw,
                          "measured_over_dd": meas / dd, "expected_over_dd": tw / dd})
            if ang in (0, 45, 90):
                say(f"  {tw:7d} {ang:6d} {meas:13.4f} {(meas - tw) / tw:9.4f} "
                    f"{meas / dd:8.5f} {tw / dd:14.5f}")
    errs = np.abs([r["rel_err"] for r in wrows])
    say()
    say(f"  median |relative error| over 25 width x orientation cases : {np.median(errs):.4f}")
    say(f"  max    |relative error|                                    : {errs.max():.4f}")
    # scale invariance of width/DD when vessel and disc scale together
    say()
    say("  CANDIDATE WIDTH ESTIMATORS, orientation dispersion at fixed true width")
    say(f"    {'true w':>7s} " + " ".join(f"{n:>14s}" for n in
                                          ("2*EDT", "2*EDT-1", "area/skeleton")))
    est_rows = []
    for tw in (5, 9, 15, 21, 31):
        vals = {"2*EDT": [], "2*EDT-1": [], "area/skeleton": []}
        for ang in (0, 15, 30, 45, 60, 75, 90):
            m = straight(S, float(ang), tw)
            sk = skeletonize(m)
            vals["2*EDT"].append(float(np.median(width_px(m))))
            vals["2*EDT-1"].append(float(np.median(width_px(m))) - 1.0)
            vals["area/skeleton"].append(float(m.sum() / max(sk.sum(), 1)))
        row = {"true_width_px": tw}
        for k, v in vals.items():
            row[f"{k}_median"] = float(np.median(v))
            row[f"{k}_dispersion"] = float(max(v) - min(v))
            row[f"{k}_rel_err"] = float((np.median(v) - tw) / tw)
        est_rows.append(row)
        say(f"    {tw:7d} " + " ".join(
            f"{row[k + '_median']:6.3f}(d{row[k + '_dispersion']:.2f})"
            for k in vals))
    say()
    say(f"    {'estimator':>16s} {'median |rel err|':>18s} {'max dispersion px':>18s}")
    best = None
    for k in ("2*EDT", "2*EDT-1", "area/skeleton"):
        me = float(np.median([abs(r[f"{k}_rel_err"]) for r in est_rows]))
        md = float(max(r[f"{k}_dispersion"] for r in est_rows))
        say(f"    {k:>16s} {me:18.4f} {md:18.3f}")
        if best is None or md < best[1]:
            best = (k, md, me)
    say(f"    FROZEN WIDTH ESTIMATOR = {best[0]} "
        f"(orientation dispersion {best[1]:.3f} px, median |rel err| {best[2]:.4f})")
    results_est = {"rows": est_rows, "selected": best[0],
                   "dispersion": best[1], "median_abs_rel_err": best[2]}

    say()
    say("  width/DD invariance when vessel and disc scale together")
    say(f"    {'scale':>6s} {'true w px':>10s} {'true w/DD':>10s} {'raw w/DD':>10s} "
        f"{'corr w/DD':>10s} {'raw err':>9s} {'corr err':>9s}")
    swrows = []
    for sc, w_true, dd in ((1, 9, 40), (2, 18, 80), (3, 36, 160)):
        m = np.zeros((40 * sc + 60, 400), bool)
        top = m.shape[0] // 2 - w_true // 2
        m[top:top + w_true, 50:350] = True
        raw = float(np.median(width_px(m)))
        corr = raw - 1.0
        swrows.append({"scale": sc, "true_px": w_true, "dd_px": dd,
                       "true_over_dd": w_true / dd, "raw_over_dd": raw / dd,
                       "corrected_over_dd": corr / dd,
                       "raw_rel_err": (raw - w_true) / w_true,
                       "corrected_rel_err": (corr - w_true) / w_true})
        r = swrows[-1]
        say(f"    {sc:6d} {w_true:10d} {r['true_over_dd']:10.5f} {r['raw_over_dd']:10.5f} "
            f"{r['corrected_over_dd']:10.5f} {r['raw_rel_err']:9.4f} "
            f"{r['corrected_rel_err']:9.4f}")
    raw_spread = max(r["raw_over_dd"] for r in swrows) - min(r["raw_over_dd"] for r in swrows)
    corr_spread = (max(r["corrected_over_dd"] for r in swrows)
                   - min(r["corrected_over_dd"] for r in swrows))
    say(f"    raw  width/DD spread across scales : {raw_spread:.6f}")
    say(f"    corrected width/DD spread          : {corr_spread:.6f}")
    # quantisation caused by 256x256 inference
    say()
    say("  QUANTISATION FLOOR FROM 256x256 INFERENCE (SEG_CURRENT_V1):")
    qrows = []
    for native_long in (480, 960, 1080, 1200, 1240, 1280, 1600):
        factor = native_long / 256.0
        for dd_native in (60, 90, 120, 180):
            qrows.append({"native_long_side": native_long, "upsample_factor": factor,
                          "dd_native_px": dd_native,
                          "floor_dd": factor / dd_native})
            say(f"    native long side {native_long:5d}  upsample x{factor:5.2f}  "
                f"DD {dd_native:4d} px  ->  width floor "
                f"{factor / dd_native:.5f} DD")
    floor_max = max(r["floor_dd"] for r in qrows)
    say()
    say(f"  worst-case width quantisation floor : {floor_max:.5f} DD")
    say("  A single binary step at 256 becomes a step of (native/256) px after NEAREST")
    say("  upsampling. Sub-step width precision is NOT supported by this pipeline.")
    results["E"] = {
        "width_cases": wrows, "orientation_median_abs_rel_err": float(np.median(errs)),
        "orientation_max_abs_rel_err": float(errs.max()),
        "estimators": results_est,
        "scale_rows": swrows,
        "raw_width_over_dd_scale_spread": float(raw_spread),
        "corrected_width_over_dd_scale_spread": float(corr_spread),
        "quantisation_rows": qrows, "worst_quantisation_floor_dd": float(floor_max),
        "parity_quantisation_px": 0.5,
        "orientation_dispersion_px": float(best[1]),
        "scale_criterion": 0.03,
        "scale_criterion_justification": (
            "1 px of digital width quantisation on the smallest plausible disc diameter "
            "(60 px) is 1/60 = 1.7%; the criterion allows 3% across a 4x scale range"),
        "VERDICT": "PASS" if raw_spread < 0.03 else "FAIL",
    }
    say(f"  WIDTH_SCALE_INVARIANCE = {results['E']['VERDICT']}")

    # ================================================================ G
    say()
    say("=" * 100)
    say("G. WHOLE-FRAME vs FOV-NORMALISED DENSITY UNDER PADDING AND CROP")
    say("=" * 100)
    m0 = straight(S, 30.0, 9) | straight(S, 70.0, 9)
    fov0 = np.ones(S, bool)
    base_whole = fov_density(m0, None)
    base_fov = fov_density(m0, fov0)
    say(f"  base whole-frame density : {base_whole:.8f}")
    say(f"  base FOV density         : {base_fov:.8f}")
    say()
    say(f"  {'perturbation':30s} {'whole-frame':>14s} {'rel chg':>10s} "
        f"{'FOV-normalised':>15s} {'rel chg':>10s}")
    grows = []
    for pad in (50, 100, 200, 300):
        mp = np.zeros((S[0] + 2 * pad, S[1] + 2 * pad), bool)
        mp[pad:pad + S[0], pad:pad + S[1]] = m0
        fp = np.zeros_like(mp)
        fp[pad:pad + S[0], pad:pad + S[1]] = True
        wp, fpv = fov_density(mp, None), fov_density(mp, fp)
        rel_w, rel_f = (wp - base_whole) / base_whole, (fpv - base_fov) / base_fov
        grows.append({"perturbation": f"+{pad}px black border", "whole": wp,
                      "whole_rel": rel_w, "fov": fpv, "fov_rel": rel_f})
        say(f"  {'+%dpx black border' % pad:30s} {wp:14.8f} {rel_w:10.4f} {fpv:15.8f} "
            f"{rel_f:10.6f}")
    # rectangular padding only on x
    pad = 200
    mr = np.zeros((S[0], S[1] + 2 * pad), bool)
    mr[:, pad:pad + S[1]] = m0
    fr = np.zeros_like(mr)
    fr[:, pad:pad + S[1]] = True
    wr, frv = fov_density(mr, None), fov_density(mr, fr)
    grows.append({"perturbation": f"rectangular +{pad}px on x", "whole": wr,
                  "whole_rel": (wr - base_whole) / base_whole, "fov": frv,
                  "fov_rel": (frv - base_fov) / base_fov})
    say(f"  {'rectangular +200px on x':30s} {wr:14.8f} "
        f"{(wr - base_whole) / base_whole:10.4f} {frv:15.8f} "
        f"{(frv - base_fov) / base_fov:10.6f}")
    # centred crop
    cr = 60
    mc = m0[cr:-cr, cr:-cr]
    wc, fc = fov_density(mc, None), fov_density(mc, np.ones_like(mc))
    grows.append({"perturbation": f"centred crop -{cr}px", "whole": wc,
                  "whole_rel": (wc - base_whole) / base_whole, "fov": fc,
                  "fov_rel": (fc - base_fov) / base_fov})
    say(f"  {'centred crop -60px per side':30s} {wc:14.8f} "
        f"{(wc - base_whole) / base_whole:10.4f} {fc:15.8f} "
        f"{(fc - base_fov) / base_fov:10.6f}")
    pad_rows = [r for r in grows if "border" in r["perturbation"]
                or "rectangular" in r["perturbation"]]
    crop_rows = [r for r in grows if "crop" in r["perturbation"]]
    worst_w = max(abs(r["whole_rel"]) for r in pad_rows)
    worst_f = max(abs(r["fov_rel"]) for r in pad_rows)
    crop_f = max(abs(r["fov_rel"]) for r in crop_rows)
    say()
    say(f"  PADDING ONLY  worst whole-frame relative change : {worst_w:.4f}")
    say(f"  PADDING ONLY  worst FOV-normalised relative chg : {worst_f:.8f}")
    say(f"  CENTRED CROP  FOV-normalised relative change    : {crop_f:.4f}")
    say()
    say("  A centred crop removes vessel and changes the density legitimately; it is not an")
    say("  invariance failure but it is exactly why the ROI coverage policy in section M is")
    say("  required. The padding result is the invariance claim, and it is exact.")
    results["G"] = {"rows": grows, "worst_whole_rel_padding": float(worst_w),
                    "worst_fov_rel_padding": float(worst_f),
                    "crop_fov_rel": float(crop_f),
                    "VERDICT": "PASS" if worst_f < 1e-6 and worst_w > 0.1 else "FAIL"}
    say(f"  FOV_DENSITY_INVARIANCE = {results['G']['VERDICT']}")

    # ================================================================ H
    say()
    say("=" * 100)
    say("H. FOV EXTRACTION Ã¢â‚¬â€ SYNTHETIC CORRECTNESS")
    say("=" * 100)
    say("  circular retina on black background, plus a black-border variant")
    frows = []
    for radius_frac, style in ((0.45, "circle on black"), (0.45, "rectangle")):  # noqa: B007
        h = w = 400
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
        disc = (r < radius_frac * min(h, w)).astype(np.float32)
        if style == "rectangle":
            disc = np.zeros((h, w), np.float32)
            disc[40:h - 40, 60:w - 60] = 1.0
        rgb = np.zeros((h, w, 3), np.float32)
        rgb[:, :, 1] = disc * 0.6
        fov, qc = retinal_fov(rgb[:, :, 1])
        cov = float(fov.mean())
        truth = float(disc.mean())
        frows.append({"style": style, "true_coverage": truth, "fov_coverage": cov,
                      "abs_err": abs(cov - truth), "fov_valid": qc["fov_valid"],
                      "reason": qc["fov_failure_reason"]})
        say(f"  {style:22s} true={truth:.4f} measured={cov:.4f} "
            f"err={abs(cov - truth):.4f} valid={qc['fov_valid']}")
    fov_err = max(r["abs_err"] for r in frows)
    results["H"] = {"rows": frows, "max_coverage_error": float(fov_err),
                    "VERDICT": "PASS" if fov_err < 0.01 else "FAIL"}
    say(f"  max coverage error = {fov_err:.4f}   VERDICT = {results['H']['VERDICT']}")

    # ================================================================ I
    say()
    say("=" * 100)
    say("I. SCALE INVARIANCE OF RAW COUNTS AND LENGTHS (1x / 2x / 4x)")
    say("=" * 100)
    say(f"  {'feature':26s} {'1x':>12s} {'2x':>12s} {'4x':>12s} {'ratio 4x/1x':>12s}")
    srows = []
    base_tree = None
    for sc in (1, 2, 4):
        m = np.zeros((200 * sc, 200 * sc), bool)
        m |= straight((200 * sc, 200 * sc), 20.0, 3 * sc)
        m |= straight((200 * sc, 200 * sc), 70.0, 5 * sc)
        if sc == 1:
            base_tree = m
        sk = skeletonize(m)
        t = _topology(sk)
        srows.append({"scale": sc, "vessel_pixels": float(m.sum()),
                      "area_fraction": float(m.mean()),
                      "skeleton_px": float(sk.sum()),
                      "n_startpoints": t["n_startpoints"],
                      "n_endpoints": t["n_endpoints"],
                      "n_intersections": t["n_intersections"]})
    keys = ["vessel_pixels", "area_fraction", "skeleton_px", "n_startpoints",
            "n_endpoints", "n_intersections"]
    for k in keys:
        v = [r[k] for r in srows]
        ratio = v[2] / v[0] if v[0] else float("nan")
        say(f"  {k:26s} {v[0]:12.1f} {v[1]:12.1f} {v[2]:12.1f} {ratio:12.3f}")
    scale_dependent = [k for k in keys
                       if abs(srows[2][k] / max(srows[0][k], 1e-9) - 1) > 0.05
                       and k != "area_fraction"]
    dimensionless = [k for k in keys if k not in scale_dependent]
    say()
    say(f"  RAW_SCALE_DEPENDENT : {scale_dependent}")
    say(f"  dimensionless-stable: {dimensionless}")
    results["I"] = {"rows": srows, "raw_scale_dependent": scale_dependent,
                    "dimensionless_stable": dimensionless}

    # ================================================================ J
    say()
    say("=" * 100)
    say("J. TOPOLOGY Ã¢â‚¬â€ VASCULAR OR ROI/BOUNDARY ARTEFACT?")
    say("=" * 100)
    m = (straight((300, 300), 30.0, 9, length=600)
         | straight((300, 300), 80.0, 9, length=600))
    fov = np.ones((300, 300), bool)
    sk = skeletonize(m)
    base = _topology(sk)
    base_fov = _topology(sk, fov)
    say(f"  full field           : {base}")
    jrows = [{"perturbation": "full field", **base}]
    # clip at the border, as a truncated FOV does
    for cut in (60, 100):
        region = np.zeros_like(m)
        region[cut:-cut, cut:-cut] = True
        t = _topology(skeletonize(m) & region)
        jrows.append({"perturbation": f"FOV crop -{cut}px per side", **t})
        say(f"  FOV crop -{cut}px/side : {t}")
    # zero padding (adds no skeleton but changes nothing inside)
    t = _topology(sk, fov)
    jrows.append({"perturbation": "zero padding", **t})
    # scale
    for sc in (2, 4):
        ms = np.zeros((300 * sc, 300 * sc), bool)
        ms |= straight((300 * sc, 300 * sc), 30.0, 9 * sc, length=600 * sc)
        ms |= straight((300 * sc, 300 * sc), 80.0, 9 * sc, length=600 * sc)
        t = _topology(skeletonize(ms), np.ones_like(ms))
        jrows.append({"perturbation": f"rescale x{sc}", **t})
        say(f"  rescale x{sc}          : {t}")
    say()
    nsp = [r["n_startpoints"] for r in jrows]
    say(f"  n_startpoints across perturbations: {nsp}")
    say(f"  spread = {max(nsp) - min(nsp)}")
    say()
    say("  INTERPRETATION: a startpoint in this implementation is a skeleton pixel with")
    say("  exactly two 8-neighbours, i.e. a line END. Every vessel that runs off the image")
    say("  or FOV boundary terminates there and manufactures one. Cutting the field changes")
    say("  the count without changing a single vessel, so the raw count is ROI/boundary")
    say("  topology, not vascular topology. The FOV-restricted variant is the same")
    say("  quantity measured inside the FOV and is no more biological.")
    results["J"] = {"rows": jrows, "spread": float(max(nsp) - min(nsp)),
                    "is_boundary_topology": bool(max(nsp) - min(nsp) > 0)}
    say(f"  n_startpoints is boundary topology: {results['J']['is_boundary_topology']}")

    # ================================================================ K
    say()
    say("=" * 100)
    say("K. FRACTAL FEATURES Ã¢â‚¬â€ SENSITIVITY")
    say("=" * 100)
    fr_base = sine((512, 512), 40, 160, 9)
    base_f = _fractal(fr_base)
    say(f"  base (512): {rnd(base_f, 5)}")
    krows = []
    for sc in (1, 2):
        m = np.kron(fr_base, np.ones((sc, sc), bool))
        f = _fractal(m)
        krows.append({"perturbation": f"resolution x{sc}", **f})
        say(f"  resolution x{sc} ({m.shape[0]}): "
            f"{rnd(f, 5)}")
    padded = np.zeros((712, 712), bool)
    padded[100:-100, 100:-100] = fr_base
    f = _fractal(padded)
    krows.append({"perturbation": "+100px padding", **f})
    say(f"  +100px padding: {rnd(f, 5)}")
    cropped = fr_base[120:-120, 120:-120]
    f = _fractal(cropped)
    krows.append({"perturbation": "central crop", **f})
    say(f"  central crop  : {rnd(f, 5)}")
    rot = ndi.rotate(fr_base.astype(np.uint8), 30, reshape=False, order=0).astype(bool)
    f = _fractal(rot)
    krows.append({"perturbation": "rotation 30 deg", **f})
    say(f"  rotation 30deg: {rnd(f, 5)}")
    for name, mm in (("erode 1px", ndi.binary_erosion(fr_base, disk(1))),
                     ("dilate 1px", ndi.binary_dilation(fr_base, disk(1)))):
        f = _fractal(mm)
        krows.append({"perturbation": name, **f})
        say(f"  {name:14s}: {rnd(f, 5)}")
    kcols = ["fractal_d0", "fractal_d1", "fractal_d2", "singularity_length"]
    say()
    say(f"  {'perturbation':18s} " + " ".join(f"{c:>18s}" for c in kcols))
    for r in krows:
        say(f"  {r['perturbation']:18s} "
            + " ".join(f"{r[c]:18.5f}" for c in kcols))
    spread = {c: float(max(r[c] for r in krows) - min(r[c] for r in krows)) for c in kcols}
    rel = {c: float(spread[c] / abs(base_f[c])) if base_f[c] else float("nan")
           for c in kcols}
    say()
    say(f"  absolute spread : {rnd(spread, 5)}")
    say(f"  relative spread : {rnd(rel, 4)}")
    results["K"] = {"base": base_f, "rows": krows, "spread": spread, "relative_spread": rel}

    # ================================================================ L
    say()
    say("=" * 100)
    say("L. REGIONAL FEATURES Ã¢â‚¬â€ FRAME GEOMETRY, NOT ANATOMY")
    say("=" * 100)
    m = np.zeros((400, 400), bool)
    m[100:300, 100:300] = True
    h = w = 400
    yy, xx = np.mgrid[0:h, 0:w]
    ang = np.arctan2(yy - h / 2, xx - w / 2)
    qs = {"ne": (ang >= 0) & (ang < np.pi / 2), "nw": (ang >= np.pi / 2) & (ang <= np.pi),
          "sw": (ang >= -np.pi) & (ang < -np.pi / 2), "se": (ang >= -np.pi / 2) & (ang < 0)}
    base_q = {k: float(m[q].mean()) for k, q in qs.items()}
    say(f"  base sector densities : {rnd(base_q, 4)}")
    lrows = [{"perturbation": "base", **base_q}]
    shifted = np.zeros_like(m)
    shifted[60:260, 60:260] = True
    sq = {k: float(shifted[q].mean()) for k, q in qs.items()}
    lrows.append({"perturbation": "translate -40px", **sq})
    say(f"  translated            : {rnd(sq)}")
    pad = np.zeros((600, 600), bool)
    pad[100:500, 100:500] = m
    yy2, xx2 = np.mgrid[0:600, 0:600]
    ang2 = np.arctan2(yy2 - 300, xx2 - 300)
    qs2 = {"ne": (ang2 >= 0) & (ang2 < np.pi / 2), "nw": (ang2 >= np.pi / 2) & (ang2 <= np.pi),
           "sw": (ang2 >= -np.pi) & (ang2 < -np.pi / 2), "se": (ang2 >= -np.pi / 2) & (ang2 < 0)}
    pq = {k: float(pad[qs2[k]].mean()) for k in qs}
    lrows.append({"perturbation": "+100px padding", **pq})
    say(f"  +100px padding        : {rnd(pq)}")
    rot_m = ndi.rotate(m.astype(np.uint8), 45, reshape=False, order=0).astype(bool)
    rq = {k: float(rot_m[qs[k]].mean()) for k in qs}
    lrows.append({"perturbation": "rotate 45 deg (frame fixed)", **rq})
    say(f"  rotated 45deg         : {rnd(rq)}")
    lspread = max(max(r[k] for r in lrows) - min(r[k] for r in lrows) for k in qs)
    say()
    say(f"  max sector-density spread across perturbations : {lspread:.4f}")
    say("  The sectors are defined by angle about the disc centre in IMAGE coordinates.")
    say("  They are NOT ICROP anatomical quadrants, which are defined relative to the")
    say("  disc-macula axis. Any sector feature must be labelled frame geometry.")
    results["L"] = {"rows": lrows, "max_spread": float(lspread),
                    "anatomical": False,
                    "definition": "image-frame angular sectors about the disc centre"}

    # ================================================================ Q
    say()
    say("=" * 100)
    say("Q. ASPECT-RATIO CONSEQUENCE OF THE 256x256 SEGMENTATION PATH")
    say("=" * 100)
    say("  SEG_CURRENT_V1 stretches every geometry to a 256x256 square. A native 1280x960")
    say("  frame is squeezed horizontally by 1280/960 = 1.333 before inference and the mask")
    say("  is stretched back afterwards. This is simulated by anisotropic resampling of a")
    say("  synthetic vessel at the same ratios.")
    say()
    say(f"  {'native':>12s} {'x/y ratio':>10s} {'width p50':>11s} {'rel chg':>9s} "
        f"{'tort med':>10s} {'rel chg':>9s} {'skel px':>10s} {'rel chg':>9s}")
    q2 = []
    for nat in ((1280, 960), (1440, 1080), (1600, 1200), (1240, 1240), (640, 480)):
        w0, h0 = nat
        base = sine((512, 512), 40, 160, 9)
        # effective squeeze: sample the pattern on a grid scaled by (w0/h0) in x
        ratio = w0 / h0
        sq = ndi.zoom(base.astype(np.float32), (1.0, 1.0 / ratio), order=0) > 0.5
        sq = sq[:, :512] if sq.shape[1] >= 512 else np.pad(
            sq, ((0, 0), (0, 512 - sq.shape[1])))
        b_w = float(np.median(width_px(base)))
        s_w = float(np.median(width_px(sq)))
        b_t = tort_of(base)
        s_t = tort_of(sq)
        b_s = float(skeletonize(base).sum())
        s_s = float(skeletonize(sq).sum())
        q2.append({"native": f"{w0}x{h0}", "ratio": ratio, "width_p50_native": b_w,
                   "width_p50_effective": s_w, "width_rel_change": (s_w - b_w) / b_w,
                   "tort_native": b_t, "tort_effective": s_t,
                   "tort_rel_change": (s_t - b_t) / b_t if b_t else float("nan"),
                   "skel_native": b_s, "skel_effective": s_s,
                   "skel_rel_change": (s_s - b_s) / b_s})
        r = q2[-1]
        say(f"  {r['native']:>12s} {ratio:10.4f} {b_w:11.3f} {r['width_rel_change']:9.4f} "
            f"{s_t:10.4f} {r['tort_rel_change']:9.4f} {s_s:10.1f} "
            f"{r['skel_rel_change']:9.4f}")
    worst_w = max(abs(r["width_rel_change"]) for r in q2)
    worst_t = max(abs(r["tort_rel_change"]) for r in q2)
    say()
    say(f"  worst width change   : {worst_w:.4f}")
    say(f"  worst tort change    : {worst_t:.4f}")
    say("  Only the non-square geometries are affected; a 1240x1240 frame is untouched.")
    results["Q"] = {"rows": q2, "worst_width_rel_change": float(worst_w),
                    "worst_tort_rel_change": float(worst_t),
                    "risk": ("FEATURE_DEPENDENT" if max(worst_w, worst_t) > 0.05
                             else "LOW")}

    # ================================================================ write
    json.dump(results, open(f"{OUT}/task5b_synthetic_audit.json", "w"), indent=2,
              default=str)
    with open(f"{OUT}/task5b_synthetic_audit.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    say()
    say("=" * 100)
    say("SUMMARY OF SYNTHETIC GATES")
    say("=" * 100)
    for k in ("D", "E", "G", "H"):
        say(f"  {k}: {results[k].get('VERDICT')}")
    say(f"  Q risk: {results['Q']['risk']}")
    say(f"  -> {OUT}/task5b_synthetic_audit.json")


if __name__ == "__main__":
    main()
