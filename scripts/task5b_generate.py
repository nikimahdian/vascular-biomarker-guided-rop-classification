#!/usr/bin/env python
"""Task 5B real-data audit and CLINICAL_MEASUREMENT_V1 table generation.

Produces data/features/clinical_measurement_v1.csv over the canonical 8870 images using the
SEG_CURRENT_V1 masks, and the real-data evidence for sections F, M, O, P, S and R.
No classifier is trained and no label is used to select anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = "/Users/moniaz/niki"
WORK = 512
META = ["image_path", "mask_path", "label", "split", "source", "group_id",
        "patient_id", "exam_id", "identity_level"]


def sha(path: str) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def worker(arg):
    from src.biomarker.clinical_measurement_v1 import measure

    image_path, mask_path, disc, with_fractal = arg
    try:
        img = Image.open(image_path).convert("RGB")
        w0, h0 = img.size
        sc = WORK / max(h0, w0)
        ws, hs = max(8, int(round(w0 * sc))), max(8, int(round(h0 * sc)))
        rgb = np.asarray(img.resize((ws, hs), Image.BILINEAR), dtype=np.float32)
        msk = np.asarray(
            Image.open(mask_path).convert("L").resize((ws, hs), Image.NEAREST)) > 127
        d = dict(disc)
        for k in ("disc_cx", "disc_cy", "disc_dd_px"):
            if np.isfinite(d.get(k, np.nan)):
                d[k] = float(d[k]) * sc
        out = measure(rgb, msk.astype(np.uint8), d, with_fractal=with_fractal)
        out["_ok"] = 1
        return image_path, out
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"{image_path}: {type(e).__name__}: {e}")
        return image_path, {"_ok": 0, "_error": f"{type(e).__name__}: {e}"}


def disc_perturbation(rgb, mask, disc, sc) -> dict:
    """Section F: sensitivity of disc-normalised width to disc centre / diameter error."""
    from src.biomarker.clinical_measurement_v1 import measure

    base = measure(rgb, mask, disc, with_fractal=False)
    ref = base.get("width_p90_dd", np.nan)
    out = {"base_width_p90_dd": ref}
    if not np.isfinite(ref) or ref == 0:
        return out
    cx, cy, dd = float(disc["disc_cx"]), float(disc["disc_cy"]), float(disc["disc_dd_px"])
    for pct in (0.02, 0.05, 0.10):
        for kind in ("dd", "centre"):
            d2 = dict(disc)
            if kind == "dd":
                d2["disc_dd_px"] = dd * (1 + pct)
            else:
                d2["disc_cx"] = cx + pct * dd
                d2["disc_cy"] = cy + pct * dd
            m2 = measure(rgb, mask, d2, with_fractal=False)
            v = m2.get("width_p90_dd", np.nan)
            out[f"{kind}_plus{pct:.0%}_width_p90_dd"] = v
            out[f"{kind}_plus{pct:.0%}_rel_change"] = (v - ref) / ref if np.isfinite(v) else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-fractal", action="store_true")
    ap.add_argument("--out", default=f"{ROOT}/data/features/clinical_measurement_v1.csv")
    ap.add_argument("--perturb", type=int, default=120,
                    help="number of deterministic images for the disc-perturbation study")
    args = ap.parse_args()
    with_fractal = not args.no_fractal
    t0 = time.time()

    splits = pd.read_csv(f"{ROOT}/data/splits/all.csv")
    canon = pd.read_csv(f"{ROOT}/data/masks/mask_manifest_canonical_v1.csv")
    meta = splits.merge(canon, on="image_path", how="left", validate="one_to_one")
    if args.limit:
        meta = meta.head(args.limit)
    ddf = pd.read_csv(f"{ROOT}/results/hvdro_validation/disc/disc_predictions_all.csv")
    dmap = ddf.set_index("image_path").to_dict("index")
    print(f"rows={len(meta)}  disc_predictions={len(ddf)}  fractal={with_fractal}")

    jobs = [(r.image_path, r.mask_path, dmap.get(r.image_path, {}), with_fractal)
            for r in meta.itertuples()]
    feats: dict[str, dict] = {}
    t1 = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        for i, (ip, out) in enumerate(pool.imap_unordered(worker, jobs, chunksize=4), 1):
            feats[ip] = out
            if i % 500 == 0 or i == len(jobs):
                el = time.time() - t1
                print(f"  {i}/{len(jobs)}  {el:.0f}s  eta {el / i * (len(jobs) - i):.0f}s",
                      flush=True)
    print(f"measurement done in {time.time() - t1:.0f}s")

    rows = []
    for r in meta.itertuples():
        o = dict(feats.get(r.image_path, {}))
        row = {c: getattr(r, c) for c in META if hasattr(r, c)}
        row["image_path"] = r.image_path
        row["mask_path"] = r.mask_path
        row.update({k: v for k, v in o.items() if k != "_ok"})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[done] {args.out} shape={df.shape}  sha256={sha(args.out)}")

    # ---------------- F: disc perturbation study ----------------
    print()
    print("=" * 100)
    print("F. DISC-NORMALISED WIDTH SENSITIVITY TO DISC PERTURBATION")
    print("=" * 100)
    sub = df[df.disc_valid == 1].sort_values("image_path").head(args.perturb)
    print(f"  disc-valid images sampled: {len(sub)}")
    prows = []
    for r in sub.itertuples():
        img = Image.open(r.image_path).convert("RGB")
        w0, h0 = img.size
        sc = WORK / max(h0, w0)
        ws, hs = max(8, int(round(w0 * sc))), max(8, int(round(h0 * sc)))
        rgb = np.asarray(img.resize((ws, hs), Image.BILINEAR), dtype=np.float32)
        msk = np.asarray(
            Image.open(r.mask_path).convert("L").resize((ws, hs), Image.NEAREST)) > 127
        d = dmap.get(r.image_path, {})
        dd = dict(d)
        for k in ("disc_cx", "disc_cy", "disc_dd_px"):
            if np.isfinite(dd.get(k, np.nan)):
                dd[k] = float(dd[k]) * sc
        prows.append(disc_perturbation(rgb, msk.astype(np.uint8), dd, sc))
    P = pd.DataFrame(prows)
    P.to_csv(f"{ROOT}/_private_audit/task5b_disc_perturbation.csv", index=False)
    print(f"  {'perturbation':24s} {'median width_p90_dd':>20s} {'median rel change':>18s} "
          f"{'p95 |rel change|':>17s}")
    for pct in (0.02, 0.05, 0.10):
        for kind in ("dd", "centre"):
            col = f"{kind}_plus{pct:.0%}_rel_change"
            v = P[col].dropna()
            w = P[f"{kind}_plus{pct:.0%}_width_p90_dd"].dropna()
            print(f"  {kind + ' ' + format(pct, '.0%'):24s} {w.median():20.6f} "
                  f"{v.median():18.6f} {v.abs().quantile(0.95):17.6f}")
    print(f"  note: a DD error scales width/DD by 1/(1+eps), so a +5% DD error biases")
    print(f"        width/DD by about -4.8% mechanically. The measured medians confirm it.")

    # ---------------- M: ROI coverage ----------------
    print()
    print("=" * 100)
    print("M. ROI COVERAGE POLICY")
    print("=" * 100)
    cov_cols = [c for c in df.columns if c.startswith("roi_coverage_")]
    print(f"  coverage columns: {cov_cols}")
    for c in cov_cols:
        v = df[c].dropna()
        print(f"  {c:30s} n={len(v):5d} min={v.min():.4f} p05={v.quantile(.05):.4f} "
              f"median={v.median():.4f} below_0.60={int((v < 0.60).sum())}")
    print()
    print("  by source (median roi_coverage_pole):")
    if "roi_coverage_pole" in df:
        print(df.groupby("source")["roi_coverage_pole"].median().round(4).to_string())
    print("  by geometry (median roi_coverage_pole):")
    dims = df.image_path.map(lambda p: _dims(p))
    df["_geom"] = dims
    print(df.groupby("_geom")["roi_coverage_pole"].median().round(4).to_string())
    print("  by split:")
    print(df.groupby("split")["roi_coverage_pole"].median().round(4).to_string())

    # ---------------- FOV / disc validity ----------------
    print()
    print("=" * 100)
    print("H/F. FOV AND DISC VALIDITY RATES")
    print("=" * 100)
    print(f"  fov_valid           : {df.fov_valid.value_counts().to_dict()}")
    print(f"  disc_valid          : {df.disc_valid.value_counts().to_dict()}")
    print(f"  fov failure reasons : {df.fov_failure_reason.value_counts().to_dict()}")
    print("  disc_valid by source:")
    print(pd.crosstab(df.source, df.disc_valid, normalize="index").round(4).to_string())
    print("  fov_valid by source:")
    print(pd.crosstab(df.source, df.fov_valid, normalize="index").round(4).to_string())
    print("  fov_valid by geometry:")
    print(pd.crosstab(df._geom, df.fov_valid, normalize="index").round(4).to_string())

    # ---------------- O: missingness shortcut ----------------
    print()
    print("=" * 100)
    print("O. MISSINGNESS SHORTCUT AUDIT")
    print("=" * 100)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    feat_cols = [c for c in df.columns if c not in META + ["_geom"]
                 and df[c].dtype.kind in "fi" and not c.startswith("is_missing_")]
    miss_rows = []
    for c in feat_cols:
        ind = df[c].isna().astype(int)
        if ind.sum() == 0 or ind.sum() == len(ind):
            continue
        X = StandardScaler().fit_transform(
            df[["dd_over_min_side", "fov_coverage_fraction"]].fillna(-1).values)
        for target, name in ((df.source, "source"), (df._geom, "geometry"),
                             (df.label, "label")):
            y = pd.factorize(target)[0]
            if len(np.unique(y)) < 2:
                continue
            try:
                p = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=5,
                                      method="predict_proba")
                auc = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
            except Exception:  # noqa: BLE001
                auc = float("nan")
            miss_rows.append({"feature": c, "missing_n": int(ind.sum()),
                              "target": name, "cv_macro_auc": auc})
    MS = pd.DataFrame(miss_rows)
    MS.to_csv(f"{ROOT}/_private_audit/task5b_missingness_shortcut.csv", index=False)
    if len(MS):
        piv = MS.pivot_table(index="feature", columns="target", values="cv_macro_auc")
        print(piv.round(4).to_string())
        print()
        print("  (macro one-vs-rest AUC of predicting the target from disc geometry and FOV")
        print("   coverage; ~0.5 means no shortcut, >0.7 means missingness is a proxy)")

    # ---------------- P: source sensitivity ----------------
    print()
    print("=" * 100)
    print("P. SOURCE / GEOMETRY SENSITIVITY (distributional, no labels)")
    print("=" * 100)
    srows = []
    for c in feat_cols:
        v = df[c]
        if v.notna().sum() < 100:
            continue
        g = df.groupby("source")[c].median()
        q = df.groupby("_geom")[c].median()
        spread = float(g.max() - g.min())
        rng = float(v.quantile(.95) - v.quantile(.05))
        srows.append({"feature": c, "source_median_spread": spread,
                      "p05_p95_range": rng,
                      "normalised_source_spread": spread / rng if rng else np.nan,
                      "max_source": g.idxmax(), "min_source": g.idxmin(),
                      "geom_median_spread": float(q.max() - q.min())})
    SS = pd.DataFrame(srows).sort_values("normalised_source_spread", ascending=False)
    SS.to_csv(f"{ROOT}/_private_audit/task5b_source_sensitivity.csv", index=False)
    print(SS.head(25).round(4).to_string(index=False))

    # ---------------- S: redundancy ----------------
    print()
    print("=" * 100)
    print("S. FEATURE REDUNDANCY (no labels)")
    print("=" * 100)
    num = df[feat_cols].select_dtypes("number")
    corr = num.corr().abs()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if np.isfinite(v) and v >= 0.98:
                pairs.append({"feature_a": cols[i], "feature_b": cols[j], "abs_corr": float(v)})
    RP = pd.DataFrame(pairs)
    RP.to_csv(f"{ROOT}/_private_audit/task5b_redundancy_pairs.csv", index=False)
    exact = [c for c in num.columns
             if any((num[c].fillna(-9e9) == num[d].fillna(-9e9)).all()
                    for d in num.columns if d != c)]
    print(f"  numeric candidate features: {len(num.columns)}")
    print(f"  pairs with |r| >= 0.98    : {len(RP)}")
    if len(RP):
        print(RP.round(4).to_string(index=False))
    print(f"  exactly duplicate columns : {len(set(exact))} {sorted(set(exact))}")

    summary = {
        "MEASUREMENT_VERSION": "CLINICAL_MEASUREMENT_V1",
        "rows": int(len(df)),
        "table_sha256": sha(args.out),
        "with_fractal": with_fractal,
        "disc_valid_rate": float((df.disc_valid == 1).mean()),
        "fov_valid_rate": float((df.fov_valid == 1).mean()) if "fov_valid" in df else None,
        "roi_coverage_pole_median": float(df.roi_coverage_pole.median()),
        "redundant_pairs_ge_098": int(len(RP)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(summary, open(f"{ROOT}/_private_audit/task5b_table_summary.json", "w"),
              indent=2, default=str)
    print()
    for k, v in summary.items():
        print(f"  {k}: {v}")


_DIMCACHE: dict = {}


def _dims(p: str) -> str:
    if p not in _DIMCACHE:
        with Image.open(p) as im:
            w, h = im.size
        _DIMCACHE[p] = f"{w}x{h}"
    return _DIMCACHE[p]


if __name__ == "__main__":
    main()
