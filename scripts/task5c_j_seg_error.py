#!/usr/bin/env python
"""Task 5C-J: segmentation error -> FINAL_PRIMARY biomarker error, and thin/mid/thick vessel recall.

External development benchmark only (HVDROPDB vessel references). No re-training, no SEG_CURRENT_V1
change, no FINAL_BIOMARKERS_V2 change, no threshold tuning, no classifier.

Pairing is by CONTENT, never by filename: every HVDROPDB image is matched to a canonical cohort image
by decoded-pixel hash, and the match is confirmed by requiring the stored SEG_CURRENT_V1 mask of that
cohort image to exist. Unpaired references are reported, not silently dropped.
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
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from scipy.stats import spearmanr

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.task5b_n2 import load, WORK  # noqa: E402
from src.biomarker import clinical_measurement_v5_candidate as v5  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"
BV = ROOT / "data/raw/hvdro/segmentation/HVDROPDB_RetCam_Neo_Segmentation/HVDROPDB-BV"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
SECONDARY = "tort_geodesic_median"
CAMERAS = {"RetCam": "RetCam_Vessels", "Neo": "Neo_Vessels"}


def px_hash(p: Path) -> str:
    with Image.open(p) as im:
        a = np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR), np.uint8)
    return hashlib.md5(a.tobytes()).hexdigest()


def bv_cases():
    rows = []
    for cam, stem in CAMERAS.items():
        idir, mdir = BV / f"{stem}_images", BV / f"{stem}_masks"
        if not idir.exists():
            continue
        for ip in sorted(idir.iterdir()):
            if ip.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                continue
            hits = list(mdir.glob(ip.stem + ".*"))
            rows.append({"camera": cam, "image": str(ip),
                         "expert_mask": str(hits[0]) if hits else ""})
    return pd.DataFrame(rows)


def hash_one(arg):
    p, = arg
    try:
        return p, px_hash(Path(p))
    except Exception:  # noqa: BLE001
        return p, ""


def cl_dice(pred: np.ndarray, exp: np.ndarray) -> float:
    sp, se = skeletonize(pred), skeletonize(exp)
    if sp.sum() == 0 or se.sum() == 0:
        return float("nan")
    t_pre = float((sp & exp).sum() / sp.sum())
    t_sen = float((se & pred).sum() / se.sum())
    return float(2 * t_pre * t_sen / (t_pre + t_sen)) if (t_pre + t_sen) else float("nan")


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2.0 * int((a & b).sum()) / s) if s else float("nan")


def measure_pair(arg):
    """One case: build pred/expert masks on the V5 work frame, run V5 twice on the same RGB."""
    case_id, rgb_path, pred_mask_path, expert_mask_path = arg
    try:
        rgb, pred = load(rgb_path, pred_mask_path)
        h, w = pred.shape
        with Image.open(expert_mask_path) as im:
            exp = np.asarray(im.convert("L").resize((w, h), Image.NEAREST)) > 127
        pred = pred.astype(bool)
        p, _f, _q = v5.measure_v5(rgb, pred.astype(np.uint8), {}, with_fractal=True)
        e, _f2, _q2 = v5.measure_v5(rgb, exp.astype(np.uint8), {}, with_fractal=True)
        tp = int((pred & exp).sum())
        row = {"case": case_id, "rgb": rgb_path,
               "dice": dice(pred, exp), "cldice": cl_dice(pred, exp),
               "precision": float(tp / max(int(pred.sum()), 1)),
               "recall": float(tp / max(int(exp.sum()), 1)),
               "pred_px": int(pred.sum()), "expert_px": int(exp.sum())}
        for f in FEATS + [SECONDARY]:
            pv, ev = p.get(f, np.nan), e.get(f, np.nan)
            row[f + "_pred"] = pv
            row[f + "_expert"] = ev
            row[f + "_signed"] = (pv - ev) if np.isfinite(pv) and np.isfinite(ev) else np.nan
            row[f + "_abs"] = abs(pv - ev) if np.isfinite(pv) and np.isfinite(ev) else np.nan
            row[f + "_rel"] = ((pv - ev) / abs(ev)
                               if np.isfinite(pv) and np.isfinite(ev) and ev else np.nan)
        # thin / mid / thick expert support (diameter 2*EDT on the expert mask)
        edt = ndi.distance_transform_edt(exp)
        yy, xx = np.nonzero(exp)
        row["_expert_diam"] = (2.0 * edt[yy, xx]).astype(np.float32)
        row["_expert_hit"] = pred[yy, xx]
        return row
    except Exception as ex:  # noqa: BLE001
        return {"case": case_id, "_err": f"{type(ex).__name__}: {ex}"}


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("A. BENCHMARK SET AND CONTENT PAIRING")
    print("=" * 100)
    B = bv_cases()
    B["has_mask"] = B.expert_mask != ""
    print(f"  HVDROPDB vessel references: {len(B)}")
    print(B.camera.value_counts().to_string())
    print(f"  with an expert mask: {int(B.has_mask.sum())}")

    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v2.csv",
                       usecols=["image_path", "mask_path", "source", "split"])
    meta = meta[meta.mask_path.notna()].reset_index(drop=True)
    print(f"  canonical cohort rows: {len(meta)}")
    args = [(p,) for p in list(meta.image_path) + list(B.image)]
    with get_context("fork").Pool(24) as pool:
        H = dict(pool.map(hash_one, args))
    inv = {}
    for p in meta.image_path:
        inv.setdefault(H.get(p, ""), []).append(p)
    nt = meta.set_index("image_path").mask_path.to_dict()
    B["cohort_match"] = [inv.get(H.get(r.image, ""), [None])[0] for r in B.itertuples()]
    B["pair_method"] = ["content_hash" if m else "" for m in B.cohort_match]
    paired = B[B.cohort_match.notna()].copy()
    print(f"  paired by content hash: {len(paired)}   "
          f"RetCam {int((paired.camera == 'RetCam').sum())}   "
          f"Neo {int((paired.camera == 'Neo').sum())}")
    for cam in CAMERAS:
        n_all = int((B.camera == cam).sum())
        n_pair = int((paired.camera == cam).sum())
        print(f"    {cam}: {n_pair} of {n_all} references matched a canonical image")
    B.to_csv(OUT / "task5c_j_pairing.csv", index=False)
    if not len(paired):
        raise SystemExit("NO_CONTENT_PAIRS")

    print()
    print("=" * 100)
    print("B/C/D. SEGMENTATION METRICS AND V5 BIOMARKER ERROR (same RGB, only the mask differs)")
    print("=" * 100)
    cargs = [(f"{r.camera}_{Path(r.image).stem[:22]}", r.cohort_match, nt[r.cohort_match],
              r.expert_mask) for r in paired.itertuples()]
    rows, errs = [], 0
    with get_context("fork").Pool(12) as pool:
        for r in pool.imap_unordered(measure_pair, cargs, chunksize=1):
            if "_err" in r:
                errs += 1
                print(f"  ERR {r['case']}: {r['_err']}")
                continue
            rows.append(r)
    R = pd.DataFrame(rows)
    R["camera"] = [c.split("_")[0] for c in R.case]
    R["camera"] = np.where(R.camera == "RetCam", "RetCam", "Neo")
    R.drop(columns=[c for c in ("_expert_diam", "_expert_hit") if c in R.columns]) \
        .to_csv(OUT / "task5c_j_rows.csv", index=False)
    print(f"  cases measured {len(R)}   errors {errs}")
    print(f"  {'group':10s} {'N':>4s} {'Dice':>8s} {'clDice':>8s} {'precision':>10s} {'recall':>8s}")
    for g, sub in list(R.groupby("camera")) + [("OVERALL", R)]:
        print(f"  {g:10s} {len(sub):4d} {sub.dice.median():8.4f} {sub.cldice.median():8.4f} "
              f"{sub.precision.median():10.4f} {sub.recall.median():8.4f}")
    print("  FINAL_PRIMARY error (predicted mask vs expert mask, same RGB):")
    print(f"  {'feature':20s} {'N':>4s} {'medAbs':>10s} {'p95Abs':>10s} {'bias':>10s} {'medRel':>10s}")
    summary = {}
    for f in FEATS + [SECONDARY]:
        a = R[f + "_abs"].dropna()
        s = R[f + "_signed"].dropna()
        rel = R[f + "_rel"].dropna()
        summary[f] = {"n": int(len(a)), "median_abs": float(a.median()),
                      "p95_abs": float(a.quantile(.95)), "bias": float(s.median()),
                      "median_rel": float(rel.median()) if len(rel) else float("nan"),
                      "max_abs": float(a.max())}
        print(f"  {f:20s} {len(a):4d} {a.median():10.6f} {a.quantile(.95):10.6f} "
              f"{s.median():+10.6f} "
              f"{(rel.median() if len(rel) else float('nan')):+10.5f}")
    print("  by camera (median absolute error):")
    for cam, sub in R.groupby("camera"):
        line = "  %-8s " % cam + "  ".join(
            f"{f}: {sub[f + '_abs'].median():.6f}" for f in FEATS)
        print(line)

    print()
    print("=" * 100)
    print("E. SEGMENTATION QUALITY vs |BIOMARKER ERROR| (Spearman)")
    print("=" * 100)
    qual = ["dice", "cldice", "precision", "recall"]
    sp = {}
    for f in FEATS + [SECONDARY]:
        for q in qual:
            d = R[[q, f + "_abs"]].dropna()
            if len(d) < 8:
                continue
            rho = float(spearmanr(d[q], d[f + "_abs"]).statistic)
            sp[(f, q)] = {"rho": rho, "n": int(len(d))}
            print(f"  {f:20s} vs {q:10s} rho {rho:+.4f}  n={len(d)}")
    strong = max(sp.items(), key=lambda kv: abs(kv[1]["rho"])) if sp else (None, None)

    print()
    print("=" * 100)
    print("F. THIN / MID / THICK EXPERT-VESSEL RECALL (tertiles of 2*EDT, per camera)")
    print("=" * 100)
    cuts, tallies = {}, []
    for cam, sub in R.groupby("camera"):
        diam = np.concatenate([d for d in sub["_expert_diam"] if d is not None])
        hit = np.concatenate([h for h in sub["_expert_hit"] if h is not None])
        q1, q2 = np.quantile(diam, [1 / 3, 2 / 3])
        cuts[cam] = {"thin_lt_px": float(q1), "mid_lt_px": float(q2),
                     "p33": float(q1), "p67": float(q2)}
        print(f"  {cam}: tertile cut-points on expert diameter 2*EDT = "
              f"{q1:.3f} px (p33), {q2:.3f} px (p67)")
        for name, m in (("thin", diam <= q1), ("mid", (diam > q1) & (diam <= q2)),
                        ("thick", diam > q2)):
            tallies.append({"camera": cam, "bin": name, "pixel_n": int(m.sum()),
                            "recall": float(hit[m].mean()) if m.any() else float("nan")})
    T = pd.DataFrame(tallies)
    print(T.to_string(index=False))
    print("  overall (pooled over cameras, cuts applied per camera):")
    for name in ("thin", "mid", "thick"):
        tt = T[T.bin == name]
        n = int(tt.pixel_n.sum())
        w = float((tt.recall * tt.pixel_n).sum() / max(n, 1))
        print(f"    {name:6s} pixels {n:9d}  pooled recall {w:.4f}")

    print()
    print("=" * 100)
    print("G. ERROR CONCENTRATION BY VESSEL WIDTH")
    print("=" * 100)
    for cam, sub in R.groupby("camera"):
        diam = np.concatenate([d for d in sub["_expert_diam"] if d is not None])
        miss = np.concatenate([~h for h in sub["_expert_hit"] if h is not None])
        q1, q2 = cuts[cam]["p33"], cuts[cam]["p67"]
        rowsg = []
        for name, m in (("thin", diam <= q1), ("mid", (diam > q1) & (diam <= q2)),
                        ("thick", diam > q2)):
            rowsg.append((name, int(m.sum()), float(miss[m].mean())))
        tot = sum(r[1] * r[2] for r in rowsg)
        print(f"  {cam}: missed expert pixels {int(tot)}")
        for name, n, mr in rowsg:
            share = (n * mr / tot) if tot else float("nan")
            print(f"    {name:6s} n={n:8d} miss_rate={mr:.4f} share_of_missed={share:.4f}")

    print()
    print("=" * 100)
    print("H. EXTREME CASE QC")
    print("=" * 100)
    picks = []
    for label, col in (("worst Dice", "dice"), ("worst clDice", "cldice")):
        r = R.sort_values(col).head(1)
        picks.append((label, r.iloc[0]))
    for f in ("vessel_density_fov", "skel_density_fov", "fractal_d0"):
        r = R.sort_values(f + "_abs", ascending=False).head(1)
        picks.append((f"largest {f} error", r.iloc[0]))
    for label, r in picks:
        print(f"  {label:28s} {r.case:28s} dice={r.dice:.4f} cldice={r.cldice:.4f} "
              f"recall={r.recall:.4f} dens_abs={r.vessel_density_fov_abs:.6f}")
    mont = 0
    for label, r in picks[:4]:
        try:
            with Image.open(r.rgb) as im:
                rgb = np.asarray(im.convert("RGB"), np.uint8)
            _, pred = load(r.rgb, nt[r.rgb])
            with Image.open([p for p in paired[paired.cohort_match == r.rgb].expert_mask][0]) as em:
                exp = np.asarray(em.convert("L").resize((pred.shape[1], pred.shape[0]),
                                                        Image.NEAREST)) > 127
            panel = np.concatenate([rgb, np.stack([pred * 255] * 3, -1),
                                    np.stack([exp * 255] * 3, -1)], axis=1)
            p = OUT / f"task5c_j_qc_{label.replace(' ', '_')}_{r.case}.png"
            Image.fromarray(panel).save(p)
            d = ImageDraw.Draw(Image.open(p))
            mont += 1
        except Exception:  # noqa: BLE001
            pass
    pd.DataFrame([{"case": r.case, "label": l} for l, r in picks]).to_csv(
        OUT / "task5c_j_qc_cases.csv", index=False)
    print(f"  montages written: {mont} (left RGB, middle SEG_CURRENT_V1, right expert)")

    print()
    print("=" * 100)
    print("J. FINAL OUTPUT")
    print("=" * 100)
    rc, neo = R[R.camera == "RetCam"], R[R.camera == "Neo"]
    out = {
        "TASK5C_J_BENCHMARK_N": int(len(B)),
        "PAIRED_BY_CONTENT_HASH_N": int(len(R)),
        "RETCAM_N": int(len(rc)), "NEO_N": int(len(neo)),
        "requested_50_50_achievable": bool(len(rc) == 50 and len(neo) == 50),
        "SEG_DICE_RETCAM": float(rc.dice.median()), "SEG_DICE_NEO": float(neo.dice.median()),
        "SEG_CLDICE_RETCAM": float(rc.cldice.median()), "SEG_CLDICE_NEO": float(neo.cldice.median()),
        "cutpoints": cuts,
        "recall_table": T.to_dict("records"),
        "summary": summary, "spearman": {f"{k[0]}|{k[1]}": v for k, v in sp.items()},
        "strongest": {"feature": str(strong[0][0]) if strong[0] else None,
                      "quality": str(strong[0][1]) if strong[0] else None,
                      "rho": strong[1]["rho"] if strong[1] else None},
        "errors": errs,
    }
    for f in FEATS:
        out[f.upper() + "_ERROR_MEDIAN"] = summary[f]["median_abs"]
    print(f"  TASK5C_J_BENCHMARK_N            : {out['TASK5C_J_BENCHMARK_N']}")
    print(f"  PAIRED_BY_CONTENT_HASH_N        : {out['PAIRED_BY_CONTENT_HASH_N']}"
          f"   (RetCam {out['RETCAM_N']} / Neo {out['NEO_N']})")
    print(f"  requested 50/50 achievable      : {out['requested_50_50_achievable']}")
    print(f"  SEG_DICE RetCam {out['SEG_DICE_RETCAM']:.4f}  Neo {out['SEG_DICE_NEO']:.4f}")
    print(f"  SEG_CLDICE RetCam {out['SEG_CLDICE_RETCAM']:.4f}  Neo {out['SEG_CLDICE_NEO']:.4f}")
    for f in FEATS:
        print(f"  {f:20s} median |error| {summary[f]['median_abs']:.6f}  bias "
              f"{summary[f]['bias']:+.6f}")
    print(f"  strongest association           : {out['strongest']}")
    (OUT / "task5c_j_summary.json").write_text(json.dumps(out, indent=2, default=str),
                                               encoding="utf-8")
    print(f"\n  elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
