#!/usr/bin/env python
"""Task 5B-H2: how far does the Task-5B-H FOV border fragility propagate into the five
FINAL_PRIMARY biomarkers?

Fixed before the run:

  * SAMPLE — the exact Task-5B-H sample is rebuilt by importing the same selection constants and
    the same loader, then asserted set-equal to `_private_audit/task5b_h_baseline.csv`. No resampling.
  * CONDITIONS — the same 8 brightness and 22 border conditions, from the same `BRIGHTNESS` list and
    the same `pad_specs()`. No new perturbation.
  * MEASUREMENT — the production `measure()` from `src/biomarker/clinical_measurement_v1.py` is
    called for baseline and for every perturbation, with `with_fractal=True`, so all five
    FINAL_PRIMARY features come from the production path. No parallel biomarker replica exists.
  * ISOLATION — the vessel mask is loaded once per image and carried through the SAME transform as
    the RGB, so it is identical in content across all 31 measurements. Only the RGB changes, so only
    the RGB-derived FOV can move.
  * FOV masks for the Dice/centroid statistics come from `retinal_fov()`, the frozen production FOV
    function, called directly; `measure()` calls the same function internally.

Nothing is redesigned: no threshold, no crop, no normalisation, no new detector, no feature
definition, no demotion, and the 8,870-row feature table is not regenerated.
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
    BRIGHTNESS, GEOM, N_BINS, PER_BRIGHTNESS_BIN, POOL_PER_STRATUM,
    apply_border, apply_brightness, pad_specs,
)
from scripts.task5b_h_mechanism import montage  # noqa: E402
from src.biomarker.clinical_measurement_v1 import measure, retinal_fov  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"
MODULE = ROOT / "src/biomarker/clinical_measurement_v1.py"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def sample() -> pd.DataFrame:
    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                       usecols=["image_path", "mask_path", "source", "split"])
    meta = meta.sort_values("image_path").reset_index(drop=True)
    meta["geom"] = [GEOM.get(Image.open(p).size, "other") for p in meta.image_path]
    pool = meta.groupby(["source", "geom"], group_keys=False).head(POOL_PER_STRATUM).copy()
    bright = []
    for p in pool.image_path:
        with Image.open(p) as im:
            bright.append(float(np.asarray(im.convert("L").resize((48, 48), Image.BILINEAR),
                                           np.float32).mean()))
    pool["brightness"] = bright
    sel = []
    for (src, gm), gp in pool.groupby(["source", "geom"]):
        for bn in np.array_split(gp.sort_values("brightness"), N_BINS):
            sel.append(bn.head(PER_BRIGHTNESS_BIN))
    return (pd.concat(sel).drop_duplicates("image_path")
            .sort_values("image_path").reset_index(drop=True))


def worker(arg):
    image_path, mask_path, source, split, geom = arg
    rows = []
    try:
        rgb, msk = load(image_path, mask_path)
        h0, w0 = msk.shape
        bfov, bqc = retinal_fov(rgb[:, :, 1])
        bm = measure(rgb, msk, {}, with_fractal=True)
        bfeat = {f: bm.get(f, np.nan) for f in FEATS}
        b_fov_px = int(bfov.sum())
        b_vpx = int(msk[bfov].sum())
        b_skel = float(bm.get("skeleton_px_work", np.nan))
        b_support = int((msk.astype(bool) & bfov).sum())
        b_row = {"image_path": image_path, "source": source, "split": split, "geom": geom,
                 "fov_px": b_fov_px, "fov_valid": bool(bqc.get("fov_valid")),
                 "fov_failure_reason": str(bqc.get("fov_failure_reason", "")),
                 "coverage": float(bfov.mean()), "vessel_px_in_fov": b_vpx,
                 "skeleton_px_work": b_skel, "fractal_support_px": b_support, **bfeat}

        conds = [("brightness", f"x{c:.2f}", apply_brightness(rgb, msk, c)) for c in BRIGHTNESS]
        conds += [(s[0], s[1], apply_border(rgb, msk, s)) for s in pad_specs(h0, w0)]
        for fam, cond, (prgb, pmsk, pad) in conds:
            pfov, pqc = retinal_fov(prgb[:, :, 1])
            pm = measure(prgb, pmsk, {}, with_fractal=True)
            t, b_, l, r_ = pad
            H, W = pfov.shape
            crop = pfov[t:t + h0, l:l + w0]
            fp = int(pfov.sum())
            p_support = int((pmsk.astype(bool) & pfov).sum())
            row = {"image_path": image_path, "source": source, "split": split, "geom": geom,
                   "family": fam, "condition": cond,
                   "fov_dice": _dice(crop, bfov),
                   "fov_rel_area_change": float((int(crop.sum()) - b_fov_px) / b_fov_px)
                   if b_fov_px else np.nan,
                   "fov_centroid_disp": _disp(crop, bfov, h0, w0),
                   "fov_px_padded": fp, "coverage": float(pfov.mean()),
                   "fov_valid": bool(pqc.get("fov_valid")),
                   "fov_failure_reason": str(pqc.get("fov_failure_reason", "")),
                   "vessel_px_in_fov": int(pmsk[pfov].sum()),
                   "skeleton_px_work": float(pm.get("skeleton_px_work", np.nan)),
                   "fractal_support_px": p_support,
                   "support_delta_px": p_support - b_support,
                   "base_fov_px": b_fov_px, "base_vessel_px": b_vpx,
                   "base_skeleton_px": b_skel, "base_support_px": b_support}
            for f in FEATS:
                v = pm.get(f, np.nan)
                bv = bfeat[f]
                row[f] = v
                row[f + "__base"] = bv
                row[f + "__abs"] = abs(v - bv) if np.isfinite(v) and np.isfinite(bv) else np.nan
                row[f + "__rel"] = ((v - bv) / abs(bv)
                                    if np.isfinite(v) and np.isfinite(bv) and bv else np.nan)
            rows.append(row)
        return {"baseline": b_row, "rows": rows}
    except Exception as e:  # noqa: BLE001
        return {"_err": f"{type(e).__name__}: {e}", "image_path": image_path}


def _dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else float("nan")


def _disp(a, b, h0, w0):
    if not (a.sum() and b.sum()):
        return np.nan
    ay, ax = ndi.center_of_mass(a)
    by, bx = ndi.center_of_mass(b)
    return float(np.hypot(ax - bx, ay - by) / max(h0, w0))


def stats(g, f):
    r = g[f + "__rel"].replace([np.inf, -np.inf], np.nan).dropna()
    a = g[f + "__abs"].dropna()
    if not len(r) or not len(a):
        return {}
    return {"n": int(len(r)),
            "median_abs": float(a.median()), "p95_abs": float(a.quantile(.95)),
            "max_abs": float(a.max()),
            "median_rel": float(r.median()), "p95_rel": float(r.quantile(.95)),
            "max_rel": float(r.max()), "min_rel": float(r.min()),
            "frac_gt_1pct": float((r.abs() > .01).mean()),
            "frac_gt_2pct": float((r.abs() > .02).mean()),
            "frac_gt_5pct": float((r.abs() > .05).mean()),
            "frac_gt_10pct": float((r.abs() > .10).mean())}


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("A/B. SAMPLE AND PRODUCTION PATH")
    print("=" * 100)
    sel = sample()
    prev = pd.read_csv(OUT / "task5b_h_baseline.csv", usecols=["image_path"])
    same = set(sel.image_path) == set(prev.image_path)
    print(f"  rebuilt sample N            : {len(sel)}")
    print(f"  Task-5B-H baseline table N  : {len(prev)}")
    print(f"  SAME SAMPLE (set-equal)     : {same}")
    print(f"  module sha256               : {sha(MODULE)}")
    print(f"  features recomputed         : {FEATS}")
    print(f"  measure() with_fractal=True, vessel mask fixed across all 31 measurements")
    print(f"  conditions                  : {len(BRIGHTNESS)} brightness + "
          f"{len(pad_specs(384, 512))} border")
    if not same:
        raise SystemExit("SAMPLE_MISMATCH")

    args = list(zip(sel.image_path, sel.mask_path, sel.source, sel.split, sel.geom))
    brows, rows, errs = [], [], 0
    with get_context("fork").Pool(16) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, args, chunksize=1), 1):
            if r.get("_err"):
                errs += 1
                print(f"  ERR {r['image_path']}: {r['_err']}", flush=True)
                continue
            brows.append(r["baseline"])
            rows.extend(r["rows"])
            if i % 40 == 0:
                print(f"  {i}/{len(args)}  {time.time() - t0:.0f}s", flush=True)
    B = pd.DataFrame(brows)
    C = pd.DataFrame(rows)
    B.to_csv(OUT / "task5b_h2_baseline.csv", index=False)
    C.to_csv(OUT / "task5b_h2_features.csv", index=False)
    cb = C[C.family == "brightness"]
    cn = C[C.family != "brightness"]
    print(f"  images: {len(B)}  errors: {errs}  rows: {len(C)}")

    print()
    print("=" * 100)
    print("D. FOV DEPENDENCY TRACED IN CODE")
    print("=" * 100)
    for line in [
        "vessel_density_fov | yes | numerator mask[fov].sum() is FOV-restricted vessel pixels; "
        "denominator fov_px is the FOV area",
        "skel_density_fov   | yes | numerator len(nonzero(skeletonize(mask))) is the WHOLE-FRAME "
        "skeleton, NOT FOV-restricted; denominator fov_px is the FOV area",
        "fractal_d0/d1/d2   | yes | _fractal(mask & fov): the FOV is the SUPPORT DOMAIN of the "
        "binary input handed to MultifractalVBMs",
    ]:
        print("  " + line)

    print()
    print("=" * 100)
    print("E. FEATURE-LEVEL DELTAS")
    print("=" * 100)
    summ = {}
    for label, g in (("BRIGHTNESS", cb), ("BORDER", cn)):
        print(f"  --- {label} ---")
        print(f"  {'feature':20s} {'medAbs':>10s} {'p95Abs':>10s} {'maxAbs':>10s} {'medRel':>10s} "
              f"{'p95Rel':>10s} {'maxRel':>10s} {'>1%':>7s} {'>2%':>7s} {'>5%':>7s} {'>10%':>7s}")
        for f in FEATS:
            s = stats(g, f)
            summ[f"{label}|{f}"] = s
            if not s:
                print(f"  {f:20s} NOT_AVAILABLE")
                continue
            print(f"  {f:20s} {s['median_abs']:10.5f} {s['p95_abs']:10.5f} {s['max_abs']:10.5f} "
                  f"{s['median_rel']:+10.5f} {s['p95_rel']:+10.5f} {s['max_rel']:+10.5f} "
                  f"{s['frac_gt_1pct']:7.4f} {s['frac_gt_2pct']:7.4f} {s['frac_gt_5pct']:7.4f} "
                  f"{s['frac_gt_10pct']:7.4f}")

    print()
    print("=" * 100)
    print("F. SOURCE AND GEOMETRY BREAKDOWN (border perturbations)")
    print("=" * 100)
    for key in ("source", "geom"):
        for f in FEATS:
            print(f"  {f} by {key}:")
            tab = []
            for k, g in cn.groupby(key):
                r = g[f + "__rel"].replace([np.inf, -np.inf], np.nan).dropna()
                tab.append((k, len(r), r.median(), r.quantile(.95), r.max(), r.min()))
            for k, n, med, p95, mx, mn in sorted(tab, key=lambda z: -abs(z[4])):
                print(f"    {k:12s} n={n:5d} median={med:+.5f} p95={p95:+.5f} "
                      f"max={mx:+.5f} min={mn:+.5f}")

    print()
    print("=" * 100)
    print("G. FOV ERROR vs BIOMARKER ERROR (Spearman, border perturbations)")
    print("=" * 100)
    for f in FEATS:
        d = cn[[f + "__abs", "fov_dice", "fov_rel_area_change"]].dropna()
        if len(d) < 10:
            print(f"  {f:20s} NOT_AVAILABLE")
            continue
        r1 = spearmanr(d.fov_dice, d[f + "__abs"]).statistic
        r2 = spearmanr(d.fov_rel_area_change, d[f + "__abs"]).statistic
        print(f"  {f:20s} dice vs |delta| rho={r1:+.4f}   "
              f"relArea vs |delta| rho={r2:+.4f}   n={len(d)}")
        summ[f"spearman|{f}"] = {"dice_vs_absdelta": float(r1),
                                 "relarea_vs_absdelta": float(r2), "n": int(len(d))}

    print()
    print("=" * 100)
    print("H/I. MECHANISM PER FEATURE")
    print("=" * 100)
    # skel_density_fov: numerator FOV-independent -> delta must equal skel_px*(1/fp_p - 1/fp_b)
    # NOTE: the authoritative version of this check lives in scripts/task5b_h2_analysis.py, which
    # re-derives every summary statistic from the saved CSV. (An earlier revision here used
    # Series.rpow(-1), which is (-1)**x and not 1/x; that is fixed.)
    for label, g in (("BORDER", cn), ("BRIGHTNESS", cb)):
        gg = g.dropna(subset=["skel_density_fov__rel"]).copy()
        pred = (1.0 / gg.fov_px_padded.astype(float) - 1.0 / gg.base_fov_px.astype(float)) \
            * gg.skeleton_px_work
        obs = gg.skel_density_fov - gg.skel_density_fov__base
        ok = np.isfinite(pred) & np.isfinite(obs)
        print(f"  skel_density_fov [{label}] analytic denominator-only prediction vs observed "
              f"(SIGNED) max|diff|={float(np.abs(pred[ok] - obs[ok]).max()):.3e} "
              f"median|diff|={float(np.abs(pred[ok] - obs[ok]).median()):.3e}")
    # vessel_density_fov decomposition
    for label, g in (("BORDER", cn), ("BRIGHTNESS", cb)):
        gg = g.dropna(subset=["vessel_density_fov__rel"]).copy()
        den = (gg.base_vessel_px / gg.fov_px_padded) - (gg.base_vessel_px / gg.base_fov_px)
        inc = (gg.vessel_px_in_fov / gg.fov_px_padded) - (gg.base_vessel_px / gg.fov_px_padded)
        obs = gg.vessel_density_fov - gg.vessel_density_fov__base
        tot = float(np.nanmedian(np.abs(obs)))
        print(f"  vessel_density_fov [{label}] median|total|={tot:.6f}  "
              f"median|denominator-only|={float(np.nanmedian(np.abs(den))):.6f}  "
              f"median|inclusion-only|={float(np.nanmedian(np.abs(inc))):.6f}  "
              f"max|inclusion-only|={float(np.nanmax(np.abs(inc))):.6f}  "
              f"max|denominator-only|={float(np.nanmax(np.abs(den))):.6f}")
        summ[f"decomposition|{label}"] = {
            "median_abs_total": tot,
            "median_abs_denominator_only": float(np.nanmedian(np.abs(den))),
            "median_abs_inclusion_only": float(np.nanmedian(np.abs(inc))),
            "max_abs_inclusion_only": float(np.nanmax(np.abs(inc))),
            "max_abs_denominator_only": float(np.nanmax(np.abs(den)))}
    # fractal: does the binary input change?
    print()
    print("  FRACTAL INPUT SUPPORT (mask & fov), border perturbations:")
    sup_same = cn[cn.support_delta_px == 0]
    sup_diff = cn[cn.support_delta_px != 0]
    print(f"    rows support unchanged : {len(sup_same)}   support changed : {len(sup_diff)}")
    for f in ["fractal_d0", "fractal_d1", "fractal_d2"]:
        s0 = sup_same[f + "__abs"].dropna()
        s1 = sup_diff[f + "__abs"].dropna()
        print(f"    {f:12s} support-unchanged max|delta|={float(s0.max()) if len(s0) else float('nan'):.3e}"
              f" (n={len(s0)})   support-changed median={float(s1.median()) if len(s1) else float('nan'):.5f}"
              f" p95={float(s1.quantile(.95)) if len(s1) else float('nan'):.5f}"
              f" max={float(s1.max()) if len(s1) else float('nan'):.5f} (n={len(s1)})")
    for f in ["fractal_d0", "fractal_d1", "fractal_d2"]:
        d = cn[[f + "__abs", "support_delta_px", "fractal_support_px"]].dropna()
        if len(d) > 10:
            rr = spearmanr(d.support_delta_px.abs(), d[f + "__abs"]).statistic
            relsup = (d.support_delta_px.abs() / d.fractal_support_px.replace(0, np.nan)).dropna()
            print(f"    {f:12s} spearman(|support delta|, |delta|)={rr:+.4f}   "
                  f"median relative support change={float(relsup.median()):.5f} "
                  f"max={float(relsup.max()):.5f}")
            summ[f"fractal_support|{f}"] = {
                "spearman_abs_support_vs_abs_delta": float(rr),
                "median_relative_support_change": float(relsup.median()),
                "max_relative_support_change": float(relsup.max()),
                "rows_support_unchanged": int(len(sup_same)),
                "rows_support_changed": int(len(sup_diff)),
                "max_abs_delta_when_support_unchanged": float(s0.max()) if len(s0) else None}

    print()
    print("=" * 100)
    print("J. NaN / FAILURE TRANSITIONS")
    print("=" * 100)
    for label, g in (("BRIGHTNESS", cb), ("BORDER", cn)):
        print(f"  --- {label} ---   rows {len(g)}")
        for f in FEATS:
            b = g[f + "__base"].notna()
            p = g[f].notna()
            fin2nan = int((b & ~p).sum())
            nan2fin = int((~b & p).sum())
            print(f"  {f:20s} finite->NaN {fin2nan:5d}   NaN->finite {nan2fin:5d}   "
                  f"baseline NaN {int((~b).sum()):5d}")
            summ[f"transitions|{label}|{f}"] = {"finite_to_nan": fin2nan, "nan_to_finite": nan2fin,
                                                "baseline_nan": int((~b).sum())}
    BB = B.set_index("image_path")
    for label, g in (("BRIGHTNESS", cb), ("BORDER", cn)):
        j = g.join(BB["fov_valid"].rename("bv"), on="image_path")
        print(f"  {label:10s} FOV valid->invalid {int((j.bv & ~j.fov_valid).sum()):5d}   "
              f"invalid->valid {int((~j.bv & j.fov_valid).sum()):5d}")

    print()
    print("=" * 100)
    print("K. EXTREME-CASE QC")
    print("=" * 100)
    qc_rows = []
    for f in FEATS:
        top = cn.reindex(cn[f + "__abs"].abs().sort_values(ascending=False).index).head(4)
        for r in top.itertuples():
            qc_rows.append((f, r.image_path, r.condition, getattr(r, f + "__abs"),
                            getattr(r, f + "__base"), getattr(r, f)))
    for f, image_path, cond, ad, bv, pv in qc_rows:
        print(f"  {f:20s} {image_path.split('/')[-1][:34]:34s} {cond:22s} "
              f"base={bv:.5f} pert={pv:.5f} |d|={ad:.5f}")
    qcd = pd.DataFrame(qc_rows, columns=["feature", "image_path", "condition", "abs_delta",
                                         "base_value", "pert_value"])
    qcd.to_csv(OUT / "task5b_h2_qc_cases.csv", index=False)
    _write_montages(qcd, sel)

    print()
    print("=" * 100)
    print("M/N. CLASSIFICATION")
    print("=" * 100)
    cls = {}
    for f in FEATS:
        s = summ.get(f"BORDER|{f}", {})
        m, p95, mx = s.get("median_rel", np.nan), s.get("p95_rel", np.nan), s.get("max_rel", np.nan)
        f5 = s.get("frac_gt_5pct", np.nan)
        if not np.isfinite(m):
            cls[f] = "INCONCLUSIVE"
        elif abs(m) < 0.005 and abs(p95) < 0.02 and abs(mx) < 0.10 and f5 < 0.01:
            cls[f] = "MINIMAL_IMPACT"
        elif abs(m) < 0.05 and abs(p95) < 0.15 and abs(mx) < 0.50:
            cls[f] = "BORDER_SENSITIVE"
        else:
            cls[f] = "SEVERELY_BORDER_SENSITIVE"
    for f in FEATS:
        s = summ.get(f"BORDER|{f}", {})
        print(f"  {f:20s} {cls[f]:26s} median={s.get('median_rel', np.nan):+.5f} "
              f"p95={s.get('p95_rel', np.nan):+.5f} max={s.get('max_rel', np.nan):+.5f} "
              f">5%={s.get('frac_gt_5pct', np.nan):.4f}")
    affected = [f for f in FEATS if cls[f] in ("BORDER_SENSITIVE", "SEVERELY_BORDER_SENSITIVE")]
    severe = [f for f in FEATS if cls[f] == "SEVERELY_BORDER_SENSITIVE"]
    frac_status = "PARTIAL" if any(f.startswith("fractal") for f in affected) else "NO"
    print()
    print(f"  PRIMARY_FEATURES_AFFECTED_N          : {len(affected)} {affected}")
    print(f"  PRIMARY_FEATURES_SEVERELY_AFFECTED_N : {len(severe)} {severe}")
    print(f"  FOV_FAILURE_PROPAGATES_TO_FRACTALS   : {frac_status}")

    out = {"TASK5B_H2_SAMPLE_N": int(len(B)), "FINAL_PRIMARY_FEATURE_N": 5,
           "module_sha256": sha(MODULE), "sample_set_equal_to_5BH": bool(same),
           "conditions": {"brightness": len(BRIGHTNESS), "border": len(pad_specs(384, 512))},
           "rows": int(len(C)), "errors": errs,
           "feature_status": cls,
           "PRIMARY_FEATURES_AFFECTED_N": len(affected),
           "PRIMARY_FEATURES_AFFECTED": affected,
           "PRIMARY_FEATURES_SEVERELY_AFFECTED_N": len(severe),
           "PRIMARY_FEATURES_SEVERELY_AFFECTED": severe,
           "FOV_FAILURE_PROPAGATES_TO_FRACTALS": frac_status,
           "stats": summ, "qc_cases": qcd.to_dict("records"),
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "task5b_h2_summary.json").write_text(json.dumps(out, indent=2, default=str),
                                                encoding="utf-8")
    print(f"\n  elapsed {time.time() - t0:.0f}s")


def _write_montages(qcd, sel):
    from PIL import ImageDraw
    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                       usecols=["image_path", "mask_path"])
    mpath = dict(zip(meta.image_path, meta.mask_path))
    done = set()
    n = 0
    for r in qcd.itertuples():
        key = (r.image_path, r.condition)
        if key in done:
            continue
        done.add(key)
        rgb, msk = load(r.image_path, mpath[r.image_path])
        h0, w0 = msk.shape
        bfov, _ = retinal_fov(rgb[:, :, 1])
        if r.condition.startswith("x"):
            prgb, pmsk, pad = apply_brightness(rgb, msk, float(r.condition[1:]))
        else:
            spec = [s for s in pad_specs(h0, w0) if s[1] == r.condition]
            if not spec:
                continue
            prgb, pmsk, pad = apply_border(rgb, msk, spec[0])
        pfov, _ = retinal_fov(prgb[:, :, 1])
        name = (f"task5b_h2_qc_{r.feature}_{r.condition}__"
                f"{r.image_path.split('/')[-1][:12]}.png".replace("/", "_"))
        p = OUT / name
        montage(rgb, bfov, prgb, pfov, p)
        im = Image.open(p)
        dr = ImageDraw.Draw(im)
        dr.text((4, 4), f"{r.feature} base={r.base_value:.5f} pert={r.pert_value:.5f} "
                        f"|d|={r.abs_delta:.5f} {r.condition}", fill=(255, 255, 0))
        m = Image.fromarray((msk.astype(bool)[:, :, None] * np.array([255, 255, 255],
                                                                    np.uint8)).astype(np.uint8))
        m = m.resize((im.height, im.height))
        canvas = Image.new("RGB", (im.width + m.width + 4, im.height), (0, 0, 0))
        canvas.paste(im, (0, 0))
        canvas.paste(m, (im.width + 4, 0))
        canvas.save(p)
        n += 1
    print(f"  montages written: {n}")


if __name__ == "__main__":
    main()
