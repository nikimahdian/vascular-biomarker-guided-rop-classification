#!/usr/bin/env python
"""Task 5B-N closure: A/V perturbation stress test.

Evaluates the EXISTING A/V heuristic. Does not change it. Does not train anything.

The A/V assignment is replicated verbatim from src/biomarker/clinical_measurement_v1.py::measure
(the branch loop and the length-weighted median split) because the frozen module does not expose
per-branch labels. A control asserts that the replicated assignment reproduces measure()'s own
a_frac on the same image, so divergence would be detected rather than assumed absent.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.biomarker.clinical_measurement_v1 import measure  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
WORK = 512
GEOM = {(640, 480): "640x480", (1280, 960): "1280x960", (1440, 1080): "1440x1080",
        (1600, 1200): "1600x1200", (1240, 1240): "1240x1240"}
TARGET = 360
PER_STRATUM = 90
AV_FEATS = ["a_frac", "a_width_p90_px", "v_width_p90_px", "av_width_ratio_p90"]


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


# ------------------------------------------------------------------ perturbations
def brightness(rgb, f):
    return rgb * f


def contrast(rgb, c):
    m = float(rgb.mean())
    return (rgb - m) * c + m


def gamma(rgb, g):
    return np.power(np.clip(rgb, 0, 1), g)


def channel(rgb, pair):
    idx, f = pair
    out = rgb.copy()
    out[:, :, idx] *= f
    return out


def channels(rgb, d):
    out = rgb.copy()
    for i, f in d.items():
        out[:, :, i] *= f
    return out


def gradient(rgb, kind, mag):
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


PERTURB = ([("brightness", f"x{f:.1f}", brightness, f) for f in (0.8, 0.9, 1.1, 1.2)]
           + [("contrast", f"x{c:.1f}", contrast, c) for c in (0.8, 0.9, 1.1, 1.2)]
           + [("gamma", f"g{g:.1f}", gamma, g) for g in (0.8, 0.9, 1.1, 1.2)]
           + [("white_balance", n, channel, v) for n, v in
              (("R+10", (0, 1.1)), ("R-10", (0, 0.9)), ("G+10", (1, 1.1)), ("G-10", (1, 0.9)),
               ("B+10", (2, 1.1)), ("B-10", (2, 0.9)))]
           + [("white_balance", "R+10_G-10", channels, {0: 1.1, 1: 0.9}),
              ("white_balance", "B+10_G-10", channels, {2: 1.1, 1: 0.9})]
           + [(f"illum_{k}", f"m{m:.2f}", gradient, (k, m))
              for k in ("horizontal", "vertical", "radial") for m in (0.10, 0.25)])


def av_assign(rgb_u8: np.ndarray, lab: np.ndarray, nlab: int, lens: dict, idx: dict):
    """Verbatim replication of the frozen A/V assignment."""
    green = rgb_u8[:, :, 1].astype(np.float32) / 255.0
    gc = green - ndi.median_filter(green, size=31, mode="nearest")
    vals, wts, keep = [], [], []
    for i in range(1, nlab + 1):
        cy, cx = idx.get(i, (None, None))
        if cy is None or len(cy) < 3:
            continue
        vals.append(float(np.median(gc[cy, cx])))
        wts.append(len(cy))
        keep.append(i)
    if len(vals) < 6:
        return None
    wts = np.asarray(wts, float)
    vals = np.asarray(vals, float)
    order = np.argsort(vals)
    cum = np.cumsum(wts[order])
    cut = int(np.searchsorted(cum, cum[-1] / 2.0))
    cut = min(max(cut, 0), len(order) - 1)
    thr = vals[order][cut]
    is_a = vals >= thr
    return {keep[j]: bool(is_a[j]) for j in range(len(keep))}, wts, keep, is_a


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("B. FROZEN A/V IMPLEMENTATION UNDER TEST")
    print("=" * 100)
    mod = ROOT / "src/biomarker/clinical_measurement_v1.py"
    print(f"  file            : src/biomarker/clinical_measurement_v1.py")
    print(f"  sha256          : {sha(mod)}")
    print(f"  function        : measure()  branch loop + length-weighted median split")
    print(f"  channel used    : GREEN only (rgb[:,:,1])")
    print(f"  normalisation   : background-corrected by gc = g - median_filter(g, 31, nearest)")
    print(f"  branch metric   : median of gc over the branch skeleton pixels")
    print(f"  split rule      : argsort(vals) -> cumulative branch LENGTH -> cut at cum[-1]/2")
    print(f"                    threshold = vals[order][cut];  is_a = vals >= threshold")
    
    T = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv")
    T["geom"] = [GEOM.get(Image.open(p).size, "other") for p in T.image_path]
    T = T.sort_values("image_path").reset_index(drop=True)
    sel = T.groupby(["source", "geom"], group_keys=False).head(PER_STRATUM)
    sel = sel.sort_values("image_path").head(TARGET).reset_index(drop=True)
    print()
    print("=" * 100)
    print("C. DETERMINISTIC STRATIFIED SAMPLE")
    print("=" * 100)
    print(f"  AV_SAMPLE_N   : {len(sel)}   (target >= 300)")
    print(f"  sources       : {sel.source.value_counts().to_dict()}")
    print(f"  geometries    : {sel.geom.value_counts().to_dict()}")
    print(f"  splits        : {sel.split.value_counts().to_dict()}")

    rows, flip_rows, feat_rows = [], [], []
    for k, r in enumerate(sel.itertuples(), 1):
        img = Image.open(r.image_path).convert("RGB")
        w0, h0 = img.size
        sc = WORK / max(h0, w0)
        ws, hs = max(8, int(round(w0 * sc))), max(8, int(round(h0 * sc)))
        rgb0 = np.asarray(img.resize((ws, hs), Image.BILINEAR), np.float32)
        msk = np.asarray(Image.open(r.mask_path).convert("L").resize(
            (ws, hs), Image.NEAREST)) > 127
        skel = skeletonize(msk)
        k8 = np.ones((3, 3), np.uint8)
        nb = ndi.convolve(skel.astype(np.uint8), k8, mode="constant") * skel
        br = skel.copy()
        br[nb >= 4] = False
        lab, nlab = ndi.label(br, structure=np.ones((3, 3), int))
        idx = {i: np.nonzero(lab == i) for i in range(1, nlab + 1)}
        lens = {i: len(idx[i][0]) for i in idx}
        dist = ndi.distance_transform_edt(msk)

        base = av_assign((np.clip(rgb0, 0, 1) * 255).astype(np.uint8), lab, nlab, lens, idx)
        if base is None:
            continue
        blab, wts, keep, is_a = base
        # control against the frozen module
        ref = measure(rgb0, msk.astype(np.uint8), {}, with_fractal=False)
        base_a_frac = float(wts[is_a].sum() / wts.sum())
        ctrl = abs(base_a_frac - ref.get("a_frac", np.nan))
        b_wpx = np.array([float(np.median(2.0 * dist[idx[i]])) for i in keep])

        def feats(labmap, isa):
            awpx = b_wpx[isa]
            vwpx = b_wpx[~isa]
            return {"a_frac": float(np.sum(np.asarray(wts)[isa]) / np.sum(wts)),
                    "a_width_p90_px": float(np.percentile(awpx, 90)) if isa.sum() >= 3 else np.nan,
                    "v_width_p90_px": float(np.percentile(vwpx, 90)) if (~isa).sum() >= 3 else np.nan}
        fb = feats(blab, is_a)
        fb["av_width_ratio_p90"] = (fb["a_width_p90_px"] / fb["v_width_p90_px"]
                                    if np.isfinite(fb.get("a_width_p90_px", np.nan))
                                    and fb.get("v_width_p90_px", 0) else np.nan)

        px_per_branch = np.array([len(idx[i][0]) for i in keep], float)
        px_branch_mask = np.zeros(nlab + 1, float)
        for j, i in enumerate(keep):
            px_branch_mask[i] = px_per_branch[j]

        rows.append({"image": r.image_path, "source": r.source, "geometry": r.geom,
                     "split": r.split, "n_baseline_branches": len(keep),
                     "control_abs_diff_a_frac": ctrl,
                     "a_frac_branchcount": float(is_a.sum() / len(is_a)),
                     "a_frac_length": base_a_frac,
                     "a_frac_pixels": float(np.sum(px_per_branch[is_a]) / np.sum(px_per_branch))})

        for fam, nm, fn, arg in PERTURB:
            if fam.startswith("illum"):
                k2, m2 = arg
                p = gradient(np.clip(rgb0, 0, 1), k2, m2)
            else:
                p = fn(np.clip(rgb0, 0, 1), arg)
            p = np.clip(p, 0, 1)
            got = av_assign((p * 255).astype(np.uint8), lab, nlab, lens, idx)
            if got is None:
                continue
            glab, gwts, gkeep, gis_a = got
            if gkeep != keep:
                continue
            flips = sum(1 for j in range(len(keep)) if bool(gis_a[j]) != bool(is_a[j]))
            flip_rows.append({"image": r.image_path, "source": r.source,
                              "geometry": r.geom, "family": fam, "perturbation": nm,
                              "n_branches": len(keep), "n_flipped": flips,
                              "flip_rate": flips / len(keep)})
            gf = feats(glab, gis_a)
            gf["av_width_ratio_p90"] = (gf["a_width_p90_px"] / gf["v_width_p90_px"]
                                        if np.isfinite(gf.get("a_width_p90_px", np.nan))
                                        and gf.get("v_width_p90_px", 0) else np.nan)
            for f in AV_FEATS:
                a, b = fb.get(f, np.nan), gf.get(f, np.nan)
                if np.isfinite(a) and np.isfinite(b):
                    feat_rows.append({"image": r.image_path, "family": fam,
                                      "perturbation": nm, "feature": f,
                                      "abs_change": abs(b - a),
                                      "rel_change": abs(b - a) / abs(a) if a else np.nan})
        if k % 50 == 0:
            print(f"  {k}/{len(sel)}  {time.time() - t0:.0f}s", flush=True)

    B = pd.DataFrame(rows)
    F = pd.DataFrame(flip_rows)
    S = pd.DataFrame(feat_rows)
    B.to_csv(ROOT / "_private_audit/task5b_n_baseline.csv", index=False)
    F.to_csv(ROOT / "_private_audit/task5b_n_flip.csv", index=False)
    S.to_csv(ROOT / "_private_audit/task5b_n_feature_stability.csv", index=False)

    print()
    print("=" * 100)
    print("D. BASELINE AND CONTROL")
    print("=" * 100)
    print(f"  images evaluated     : {len(B)}")
    print(f"  baseline branch total: {int(B.n_baseline_branches.sum())}")
    print(f"  branch count / image : median {B.n_baseline_branches.median():.0f}")
    print(f"  CONTROL max |replicated a_frac - measure() a_frac| = "
          f"{B.control_abs_diff_a_frac.max():.3e}")
    print(f"  control PASS (<=1e-6): {bool(B.control_abs_diff_a_frac.max() <= 1e-6)}")

    print()
    print("=" * 100)
    print("F. BRANCH LABEL FLIP RATE")
    print("=" * 100)
    print(f"  {'family':18s} {'perturbation':14s} {'n_img':>6s} {'mean flip':>10s} "
          f"{'median flip':>12s} {'max flip':>9s}")
    g = F.groupby(["family", "perturbation"]).flip_rate
    summ = g.agg(n_img="count", mean="mean", median="median", max="max").reset_index()
    for _, r in summ.iterrows():
        print(f"  {r.family:18s} {r.perturbation:14s} {int(r.n_img):6d} {r['mean']:10.4f} "
              f"{r['median']:12.4f} {r['max']:9.4f}")
    fam = F.groupby("family").flip_rate.agg(["mean", "max"])
    print()
    for k, r in fam.iterrows():
        print(f"  {k:18s} mean={r['mean']:.4f}  max={r['max']:.4f}")
    print()
    print(f"  OVERALL_MAX_BRANCH_FLIP_RATE = {F.flip_rate.max():.4f}")
    print(f"  OVERALL_MEAN_BRANCH_FLIP_RATE = {F.flip_rate.mean():.4f}")
    print()
    agg = F.groupby("image").n_flipped.sum() / F.groupby("image").n_branches.first()
    print("  image-level:")
    print(f"    images with >=1 flipped branch  : {float((agg > 0).mean()):.4f}")
    print(f"    images with >=10% flipped       : {float((agg >= 0.10).mean()):.4f}")
    print(f"    images with >=25% flipped       : {float((agg >= 0.25).mean()):.4f}")
    print()
    print("  by source:")
    print(F.groupby("source").flip_rate.agg(["mean", "max"]).round(4).to_string())
    print("  by geometry:")
    print(F.groupby("geometry").flip_rate.agg(["mean", "max"]).round(4).to_string())

    print()
    print("=" * 100)
    print("G. A/V FEATURE STABILITY")
    print("=" * 100)
    print(f"  {'feature':24s} {'median|d|':>11s} {'median rel':>11s} {'p95|d|':>11s} "
          f"{'max|d|':>11s}")
    for f in AV_FEATS:
        s = S[S.feature == f]
        if not len(s):
            continue
        print(f"  {f:24s} {s.abs_change.median():11.5f} {s.rel_change.median():11.5f} "
              f"{s.abs_change.quantile(0.95):11.5f} {s.abs_change.max():11.5f}")

    print()
    print("=" * 100)
    print("H. THE ~50/50 MECHANISM")
    print("=" * 100)
    for c in ("a_frac_branchcount", "a_frac_length", "a_frac_pixels"):
        v = B[c].dropna()
        print(f"  {c:22s} median={v.median():.4f} IQR=[{v.quantile(.25):.4f},"
              f"{v.quantile(.75):.4f}] p05={v.quantile(.05):.4f} p95={v.quantile(.95):.4f}  "
              f"0.45-0.55={float(((v >= .45) & (v <= .55)).mean()):.4f}  "
              f"0.40-0.60={float(((v >= .40) & (v <= .60)).mean()):.4f}")
    print()
    print("  MECHANISM (from code): the split point is chosen so that the cumulative branch")
    print("  LENGTH on the artery side is half the total branch length. Therefore")
    print("  a_frac_length is ~0.5 BY CONSTRUCTION for every image, and a deviation from 0.5")
    print("  can only come from the discrete cut index. Branch COUNT and VESSEL PIXEL")
    print("  fractions are not constrained and are free to differ.")

    out = {
        "AV_STRESS_TEST_EXECUTED": "YES", "AV_SAMPLE_N": int(len(B)),
        "BASELINE_BRANCH_N": int(B.n_baseline_branches.sum()),
        "CONTROL_MAX_ABS_DIFF_A_FRAC": float(B.control_abs_diff_a_frac.max()),
        "OVERALL_MAX_BRANCH_FLIP_RATE": float(F.flip_rate.max()),
        "OVERALL_MEAN_BRANCH_FLIP_RATE": float(F.flip_rate.mean()),
        "family_max_flip": {k: float(v) for k, v in fam["max"].items()},
        "family_mean_flip": {k: float(v) for k, v in fam["mean"].items()},
        "image_level": {"ge_1_flip": float((agg > 0).mean()),
                        "ge_10pct": float((agg >= 0.10).mean()),
                        "ge_25pct": float((agg >= 0.25).mean())},
        "balance": {c: {"median": float(B[c].median()), "iqr": float(
            B[c].quantile(.75) - B[c].quantile(.25)),
            "p05": float(B[c].quantile(.05)), "p95": float(B[c].quantile(.95)),
            "in_45_55": float(((B[c] >= .45) & (B[c] <= .55)).mean()),
            "in_40_60": float(((B[c] >= .40) & (B[c] <= .60)).mean())}
            for c in ("a_frac_branchcount", "a_frac_length", "a_frac_pixels")},
        "by_source": F.groupby("source").flip_rate.mean().to_dict(),
        "bygeometry": F.groupby("geometry").flip_rate.mean().to_dict(),
        "module_sha256": sha(mod),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (ROOT / "_private_audit/task5b_n_summary.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n  elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
