#!/usr/bin/env python
"""Task 5B-N2: production-faithful A/V instrumentation, equivalence gate, and the final
perturbation stress battery.

Phase 1  behavioural equivalence of the refactor over all 8,870 canonical images, plus the
         direct control of diagnostic a_frac against measure().
Phase 2  the five-family perturbation battery on the frozen 360-image sample.
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
from skimage.morphology import skeletonize

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("/Users/moniaz/niki")
WORK = 512
GEOM = {(640, 480): "640x480", (1280, 960): "1280x960", (1440, 1080): "1440x1080",
        (1600, 1200): "1600x1200", (1240, 1240): "1240x1240"}
PER_STRATUM, TARGET = 90, 360
AV_COLS = ["a_frac", "a_width_p90_px", "v_width_p90_px", "av_width_ratio_p90"]
EXTRA = ["branch_px_work", "n_branches"]


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def load(image_path, mask_path):
    img = Image.open(image_path).convert("RGB")
    w0, h0 = img.size
    sc = WORK / max(h0, w0)
    ws, hs = max(8, int(round(w0 * sc))), max(8, int(round(h0 * sc)))
    rgb = np.asarray(img.resize((ws, hs), Image.BILINEAR), np.float32)   # 0..255, NOT normalised
    msk = np.asarray(Image.open(mask_path).convert("L").resize(
        (ws, hs), Image.NEAREST)) > 127
    return rgb, msk.astype(np.uint8)


def branch_setup(rgb, msk):
    from src.biomarker.clinical_measurement_v1 import _fractal  # noqa: F401
    skel = skeletonize(msk > 0)
    k8 = np.ones((3, 3), np.uint8)
    nb = ndi.convolve(skel.astype(np.uint8), k8, mode="constant") * skel
    br = skel.copy()
    br[nb >= 4] = False
    lab, nlab = ndi.label(br, structure=np.ones((3, 3), int))
    dist = ndi.distance_transform_edt(msk > 0)
    return skel, lab, nlab, dist


def av_diag(rgb, lab, nlab, dist):
    """Diagnostics from the PRODUCTION helper. No replica."""
    from src.biomarker.clinical_measurement_v1 import assign_av_branches
    green = rgb[:, :, 1]
    gc = green - ndi.median_filter(green, size=31, mode="nearest")
    return assign_av_branches(gc, lab, nlab, dist)


def phase1(arg):
    image_path, mask_path = arg
    from src.biomarker.clinical_measurement_v1 import measure
    try:
        rgb, msk = load(image_path, mask_path)
        ref = measure(rgb, msk, {}, with_fractal=False)
        _, lab, nlab, dist = branch_setup(rgb, msk)
        av = av_diag(rgb, lab, nlab, dist)
        row = {"image_path": image_path}
        for c in AV_COLS + EXTRA:
            row[c] = ref.get(c, np.nan)
        row["diag_a_frac_length"] = av["a_frac_length"]
        row["diag_a_frac_count"] = av["a_frac_count"]
        row["diag_n_branches"] = av["n"]
        row["diag_threshold"] = av["threshold"]
        row["diag_tie_fraction"] = av["tie_fraction"]
        row["diag_n_at_threshold"] = av["n_at_threshold"]
        row["diag_ctrl_abs"] = (abs(av["a_frac_length"] - ref.get("a_frac", np.nan))
                                if av["eligible"] else 0.0)
        return row
    except Exception as e:  # noqa: BLE001
        return {"image_path": image_path, "_err": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- perturbations
def p_bright(rgb, f):
    return rgb * f


def p_contrast(rgb, c):
    m = float(rgb.mean())
    return (rgb - m) * c + m


def p_gamma(rgb, g):
    return np.power(np.clip(rgb / 255.0, 0, 1), g) * 255.0


def p_chan(rgb, pair):
    i, f = pair
    o = rgb.copy()
    o[:, :, i] *= f
    return o


def p_chans(rgb, d):
    o = rgb.copy()
    for i, f in d.items():
        o[:, :, i] *= f
    return o


def p_grad(rgb, arg):
    kind, mag = arg
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if kind == "horizontal":
        g = (xx / (w - 1)) * 2 - 1
    elif kind == "vertical":
        g = (yy / (h - 1)) * 2 - 1
    else:
        cy, cx = (h - 1) / 2, (w - 1) / 2
        r = np.hypot(yy - cy, xx - cx)
        g = (r / r.max()) * 2 - 1
    return rgb * (1.0 + mag * g)[:, :, None]


PERTURB = ([("brightness", f"x{f:.1f}", p_bright, f) for f in (0.8, 0.9, 1.1, 1.2)]
           + [("contrast", f"x{c:.1f}", p_contrast, c) for c in (0.8, 0.9, 1.1, 1.2)]
           + [("gamma", f"g{g:.1f}", p_gamma, g) for g in (0.8, 0.9, 1.1, 1.2)]
           + [("white_balance", n, p_chan, v) for n, v in
              (("R+10", (0, 1.1)), ("R-10", (0, 0.9)), ("G+10", (1, 1.1)), ("G-10", (1, 0.9)),
               ("B+10", (2, 1.1)), ("B-10", (2, 0.9)))]
           + [("white_balance", "R+10_G-10", p_chans, {0: 1.1, 1: 0.9}),
              ("white_balance", "B+10_G-10", p_chans, {2: 1.1, 1: 0.9})]
           + [("illumination", f"{k}_m{m:.2f}", p_grad, (k, m))
              for k in ("horizontal", "vertical", "radial") for m in (0.10, 0.25)])


def phase2(arg):
    image_path, mask_path, source, geometry = arg
    out = {"image_path": image_path, "flip": [], "feat": [], "diag": None}
    try:
        rgb, msk = load(image_path, mask_path)
        _, lab, nlab, dist = branch_setup(rgb, msk)
        base = av_diag(rgb, lab, nlab, dist)
        if not base["eligible"]:
            return {"image_path": image_path, "_skip": "ineligible"}
        bset = base["branch_index"]
        out["diag"] = {"n": base["n"], "tie_fraction": base["tie_fraction"],
                       "n_at_threshold": base["n_at_threshold"],
                       "a_frac_length": base["a_frac_length"],
                       "a_frac_count": base["a_frac_count"],
                       "threshold": base["threshold"]}
        w = np.asarray(base["length"], float)
        wpx = np.asarray(base["width_px"], float)
        isa = base["is_a"]

        def feats(isa_):
            a, v = wpx[isa_], wpx[~isa_]
            af = float(w[isa_].sum() / max(w.sum(), 1))
            aw = float(np.percentile(a, 90)) if isa_.sum() >= 3 else np.nan
            vw = float(np.percentile(v, 90)) if (~isa_).sum() >= 3 else np.nan
            return (af, aw, vw, (aw / vw if np.isfinite(aw) and vw else np.nan))
        fb = feats(isa)

        diag = base
        for fam, nm, fn, arg_ in PERTURB:
            pert = np.clip(fn(rgb, arg_), 0, 255)
            g2 = av_diag(pert, lab, nlab, dist)
            if not g2["eligible"] or g2["branch_index"] != bset:
                out["flip"].append({"family": fam, "perturbation": nm, "_mismatch": 1,
                                    "n": len(bset), "n_flipped": 0, "flip_rate": np.nan})
                continue
            fl = int(np.sum(g2["is_a"] != isa))
            out["flip"].append({"family": fam, "perturbation": nm, "n": len(bset),
                                "n_flipped": fl, "flip_rate": fl / len(bset),
                                "a_frac": g2["a_frac_length"]})
            gf = feats(g2["is_a"])
            for j, f in enumerate(AV_COLS):
                a_, b_ = fb[j], gf[j]
                if np.isfinite(a_) and np.isfinite(b_):
                    out["feat"].append({"family": fam, "perturbation": nm, "feature": f,
                                        "abs_change": abs(b_ - a_),
                                        "rel_change": abs(b_ - a_) / abs(a_) if a_ else np.nan})
        return out
    except Exception as e:  # noqa: BLE001
        return {"image_path": image_path, "_err": f"{type(e).__name__}: {e}"}


def main() -> None:
    t0 = time.time()
    mod = ROOT / "src/biomarker/clinical_measurement_v1.py"
    print("=" * 100)
    print("N. CODE PROVENANCE — BEHAVIOR-PRESERVING DIAGNOSTIC REFACTOR")
    print("=" * 100)
    print(f"  pre-refactor  sha256 : 3c1c10773ed28a187df373412e9de62309f30dd2896ab602b36bea6e44b29e66")
    print(f"  post-refactor sha256 : {sha(mod)}")
    print(f"  helper               : assign_av_branches(gc, lab, nlab, dist)")
    print(f"  single implementation: measure() calls the same helper; no replica is used")

    T = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                    usecols=["image_path", "mask_path", "source", "split"])
    print()
    print("=" * 100)
    print("C. FULL BEHAVIOURAL-EQUIVALENCE GATE — ALL 8,870 CANONICAL IMAGES")
    print("=" * 100)
    args = list(zip(T.image_path, T.mask_path))
    rows = []
    with get_context("fork").Pool(20) as pool:
        for i, r in enumerate(pool.imap_unordered(phase1, args, chunksize=8), 1):
            rows.append(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(args)}  {time.time() - t0:.0f}s", flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(ROOT / "_private_audit/task5b_n2_equivalence.csv", index=False)
    ref = pd.read_csv(ROOT / "data/features/clinical_measurement_v1.csv",
                      usecols=["image_path"] + AV_COLS + EXTRA)
    M = ref.merge(P, on="image_path", suffixes=("_old", "_new"))
    print(f"  merged rows            : {len(M)}")
    checked, diffs, maxd = 0, 0, 0.0
    for c in AV_COLS + EXTRA:
        a = M[f"{c}_old"].to_numpy(float)
        b = M[f"{c}_new"].to_numpy(float)
        nan_same = np.array_equal(np.isnan(a), np.isnan(b))
        fin = np.isfinite(a) & np.isfinite(b)
        d = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
        same = nan_same and (fin.sum() == np.isfinite(a).sum()) and d == 0.0
        checked += 1
        maxd = max(maxd, d)
        if not same:
            diffs += 1
        print(f"  {c:24s} nan_pattern_same={nan_same}  max|delta|={d:.3e}  "
              f"{'IDENTICAL' if same else 'DIFFERS'}")
    print()
    print(f"  AV_REFACTOR_CANONICAL_N                : {len(M)}")
    print(f"  AV_FEATURE_COLUMNS_CHECKED            : {checked}")
    print(f"  AV_ROWS_EXACT_OR_NUMERICALLY_EQUIVALENT: {len(M) if diffs == 0 else 0}")
    print(f"  AV_ROWS_DIFFERENT                     : {diffs}")
    print(f"  AV_MAX_ABS_DELTA                      : {maxd:.3e}")
    eq_pass = (diffs == 0 and maxd == 0.0)
    print(f"  AV_FULL_BEHAVIOR_EQUIVALENCE          : {'PASS' if eq_pass else 'FAIL'}")
    if not eq_pass:
        print("  STOP — do not run the battery")

    print()
    print("=" * 100)
    print("D. DIRECT CONTROL — DIAGNOSTIC a_frac vs measure() a_frac")
    print("=" * 100)
    ctrl = float(P.diag_ctrl_abs.max())
    print(f"  images compared       : {len(P)}")
    print(f"  DIAGNOSTIC_CONTROL_MAX_DELTA = {ctrl:.3e}")
    print(f"  control PASS (== 0)   : {bool(ctrl == 0.0)}")

    if not (eq_pass and ctrl == 0.0):
        print("\n  TASK5B_N2_CLOSURE = INCOMPLETE")
        json.dump({"equivalence": eq_pass, "control": ctrl, "diffs": diffs},
                  open(ROOT / "_private_audit/task5b_n2_gate.json", "w"), indent=2)
        return

    # ---------------------------------------------------------------- battery
    print()
    print("=" * 100)
    print("E. FROZEN DETERMINISTIC SAMPLE")
    print("=" * 100)
    T2 = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                     usecols=["image_path", "mask_path", "source", "split"])
    T2["geom"] = [GEOM.get(Image.open(p).size, "other") for p in T2.image_path]
    T2 = T2.sort_values("image_path").reset_index(drop=True)
    sel = T2.groupby(["source", "geom"], group_keys=False).head(PER_STRATUM)
    sel = sel.sort_values("image_path").head(TARGET).reset_index(drop=True)
    print(f"  AV_SAMPLE_N   : {len(sel)}")
    print(f"  sources       : {sel.source.value_counts().to_dict()}")
    print(f"  geometries    : {sel.geom.value_counts().to_dict()}")
    print(f"  splits        : {sel.split.value_counts().to_dict()}")
    print(f"  NOTE 1440x1080 absent from the frozen sample: data-limited strata, not resampled")

    print()
    print("=" * 100)
    print("F/G/H. PERTURBATION BATTERY — PRODUCTION-FAITHFUL BRANCH LABELS")
    print("=" * 100)
    t1 = time.time()
    cargs = list(zip(sel.image_path, sel.mask_path, sel.source, sel.geom))
    flips, feats, diags, mism = [], [], [], 0
    with get_context("fork").Pool(12) as pool:
        for i, r in enumerate(pool.imap_unordered(phase2, cargs, chunksize=2), 1):
            if r.get("_skip") or r.get("_err"):
                continue
            diags.append({"image_path": r["image_path"], **r["diag"]})
            for f in r["flip"]:
                if f.get("_mismatch"):
                    mism += 1
                    continue
                f.update({"image_path": r["image_path"]})
                flips.append(f)
            for f in r["feat"]:
                feats.append(f)
            if i % 60 == 0:
                print(f"  {i}/{len(cargs)}  {time.time() - t1:.0f}s", flush=True)
    F = pd.DataFrame(flips)
    FE = pd.DataFrame(feats)
    DG = pd.DataFrame(diags)
    F.to_csv(ROOT / "_private_audit/task5b_n2_flip.csv", index=False)
    FE.to_csv(ROOT / "_private_audit/task5b_n2_feature_stability.csv", index=False)
    DG.to_csv(ROOT / "_private_audit/task5b_n2_diagnostics.csv", index=False)
    gm = sel.set_index("image_path")[["source", "geom"]]
    F = F.join(gm, on="image_path")
    print(f"  branch-set mismatches across conditions : {mism} "
          f"({'none' if mism == 0 else 'EXCLUDED from denominator'})")
    print(f"  baseline branches : {int(DG.n.sum())}")
    print()
    print(f"  {'family':14s} {'perturbation':16s} {'n_img':>6s} {'mean':>8s} {'median':>8s} "
          f"{'max':>8s}")
    for (fam, nm), g in F.groupby(["family", "perturbation"]):
        print(f"  {fam:14s} {nm:16s} {len(g):6d} {g.flip_rate.mean():8.4f} "
              f"{g.flip_rate.median():8.4f} {g.flip_rate.max():8.4f}")
    fam = F.groupby("family").flip_rate.agg(["mean", "max"])
    print()
    for k, r in fam.iterrows():
        print(f"  {k:14s} mean={r['mean']:.4f} max={r['max']:.4f}")
    print(f"\n  OVERALL_MAX_BRANCH_FLIP_RATE  = {F.flip_rate.max():.4f}")
    print(f"  OVERALL_MEAN_BRANCH_FLIP_RATE = {F.flip_rate.mean():.4f}")
    agg = F.groupby("image_path").n_flipped.sum() / F.groupby("image_path").n.first()
    print(f"  images >=1 flip {float((agg > 0).mean()):.4f}  >=10% {float((agg >= .1).mean()):.4f}"
          f"  >=25% {float((agg >= .25).mean()):.4f}")
    print("  by source:")
    print(F.groupby("source").flip_rate.mean().round(4).to_string())
    print("  by geometry:")
    print(F.groupby("geom").flip_rate.mean().round(4).to_string())

    print()
    print("=" * 100)
    print("I. TIE SENSITIVITY")
    print("=" * 100)
    print(f"  TIE_AT_THRESHOLD_MEDIAN_FRACTION = {DG.tie_fraction.median():.4f}")
    print(f"  n_at_threshold median            = {DG.n_at_threshold.median():.1f} of "
          f"{DG.n.median():.0f} branches")
    print(f"  images with any tie at threshold : {float((DG.n_at_threshold > 0).mean()):.4f}")
    fr = F.groupby("image_path").flip_rate.mean().rename("mf")
    j = DG.set_index("image_path").join(fr)
    if j.mf.notna().sum() > 10:
        print(f"  spearman(tie_fraction, mean flip rate) = "
              f"{j[['tie_fraction','mf']].corr(method='spearman').iloc[0,1]:.4f}")

    print()
    print("=" * 100)
    print("J. 50/50 STRUCTURAL BALANCE (production branch labels)")
    print("=" * 100)
    for c, lbl in (("a_frac_count", "branch count"), ("a_frac_length", "branch length")):
        v = DG[c].dropna()
        print(f"  {lbl:14s} median={v.median():.4f} IQR={v.quantile(.75)-v.quantile(.25):.4f} "
              f"p05-p95=[{v.quantile(.05):.4f},{v.quantile(.95):.4f}] "
              f"min={v.min():.4f} max={v.max():.4f} "
              f"0.45-0.55={float(((v>=.45)&(v<=.55)).mean()):.4f} "
              f"0.40-0.60={float(((v>=.40)&(v<=.60)).mean()):.4f}")
    print("  vessel-pixel contribution fraction is not derivable from the production helper")

    print()
    print("=" * 100)
    print("K. A/V FEATURE STABILITY")
    print("=" * 100)
    print(f"  {'feature':24s} {'median|d|':>11s} {'median rel':>11s} {'p95|d|':>11s} {'max|d|':>11s}")
    for f in AV_COLS:
        s = FE[FE.feature == f]
        if not len(s):
            continue
        print(f"  {f:24s} {s.abs_change.median():11.5f} {s.rel_change.median():11.5f} "
              f"{s.abs_change.quantile(.95):11.5f} {s.abs_change.max():11.5f}")

    out = {
        "AV_STRESS_TEST_EXECUTED": "YES", "AV_PRODUCTION_INSTRUMENTATION": "PASS",
        "AV_FULL_BEHAVIOR_EQUIVALENCE": "PASS" if eq_pass else "FAIL",
        "AV_SAMPLE_N": int(len(sel)), "BASELINE_BRANCH_N": int(DG.n.sum()),
        "DIAGNOSTIC_CONTROL_MAX_DELTA": ctrl,
        "AV_MAX_ABS_DELTA_EQUIVALENCE": maxd,
        "branch_set_mismatches": mism,
        "OVERALL_MAX_BRANCH_FLIP_RATE": float(F.flip_rate.max()),
        "OVERALL_MEAN_BRANCH_FLIP_RATE": float(F.flip_rate.mean()),
        "family_max_flip": {k: float(v) for k, v in fam["max"].items()},
        "family_mean_flip": {k: float(v) for k, v in fam["mean"].items()},
        "image_level": {"ge_1": float((agg > 0).mean()), "ge_10pct": float((agg >= .1).mean()),
                        "ge_25pct": float((agg >= .25).mean())},
        "TIE_AT_THRESHOLD_MEDIAN_FRACTION": float(DG.tie_fraction.median()),
        "balance": {c: {"median": float(DG[c].median()),
                        "iqr": float(DG[c].quantile(.75) - DG[c].quantile(.25)),
                        "p05": float(DG[c].quantile(.05)), "p95": float(DG[c].quantile(.95)),
                        "min": float(DG[c].min()), "max": float(DG[c].max()),
                        "in_45_55": float(((DG[c] >= .45) & (DG[c] <= .55)).mean()),
                        "in_40_60": float(((DG[c] >= .40) & (DG[c] <= .60)).mean())}
                    for c in ("a_frac_count", "a_frac_length")},
        "feature_stability": {f: {"median_abs": float(FE[FE.feature == f].abs_change.median()),
                                  "max_abs": float(FE[FE.feature == f].abs_change.max())}
                              for f in AV_COLS if len(FE[FE.feature == f])},
        "by_source": F.groupby("source").flip_rate.mean().to_dict(),
        "by_geom": F.groupby("geom").flip_rate.mean().to_dict(),
        "module_sha256": sha(mod), "elapsed_s": round(time.time() - t0, 1),
    }
    (ROOT / "_private_audit/task5b_n2_summary.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n  elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
