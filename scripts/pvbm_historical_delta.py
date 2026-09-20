#!/usr/bin/env python
"""Task 4 (H/I/J/K/L/P/Q): quantify how the mask-pairing bug moved each historical feature.

No clinical interpretation and no classifier is run here. The single question is:
how much did the wrong image->mask association change each historical feature value?
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
F = f"{ROOT}/data/features"
PRIV = f"{ROOT}/_private_audit"
ART = f"{ROOT}/artifacts"
os.makedirs(ART, exist_ok=True)

HIST = f"{F}/biomarker_features.csv"
CORR = f"{F}/biomarker_features_historical_equivalent_corrected_v1.csv"
CANON = f"{ROOT}/data/masks/mask_manifest_canonical_v1.csv"
CONTRACT = f"{ROOT}/configs/feature_contract.csv"

META = {"image_path", "mask_path", "label", "split", "source",
        "group_id", "patient_id", "exam_id", "identity_level"}

# features whose historical form is knowingly weak but must NOT be changed in Task 4 (L)
LIMITATION_FEATURES = {
    "n_startpoints", "n_endpoints", "overall_length", "area",
    "vessel_density", "vessel_pixels",
    "density_r0c0", "density_r0c1", "density_r0c2",
    "density_r1c0", "density_r1c1", "density_r1c2",
    "density_r2c0", "density_r2c1", "density_r2c2",
}

GEOMETRY_BY_MINSIDE = {480: "min480_640x480", 960: "min960_1280x960",
                       1080: "min1080_1440x1080", 1200: "min1200_1600x1200",
                       1240: "min1240_1240x1240"}


def _shape(p: str) -> tuple[str, int]:
    try:
        with Image.open(p) as im:
            w, h = im.size
        return p, min(h, w)
    except Exception:  # noqa: BLE001
        return p, -1


def hid(text: str, n: int = 16) -> str:
    return hashlib.sha256(str(text).encode()).hexdigest()[:n]


# ---------------------------------------------------------------- load
hist = pd.read_csv(HIST)
corr = pd.read_csv(CORR)
canon = pd.read_csv(CANON)
contract = pd.read_csv(CONTRACT)
FEATS = [c for c in hist.columns if c not in META]

print("=" * 100)
print("A. AFFECTED SET AND DONOR IDENTIFICATION")
print("=" * 100)
j = hist.merge(corr, on="image_path", suffixes=("_h", "_c"), how="inner")
j["affected"] = j["mask_path_h"] != j["mask_path_c"]

rev = dict(zip(canon.mask_path, canon.image_path))
lab = dict(zip(canon.image_path, hist.set_index("image_path")["label"]))
grp = dict(zip(canon.image_path, hist.set_index("image_path")["group_id"]))
j["donor"] = j["mask_path_h"].map(rev)
j["donor_found"] = j.donor.notna()
j["donor_label"] = j.donor.map(lab)
j["donor_group"] = j.donor.map(grp)
j["donor_label_differs"] = j["donor_label"].notna() & (j["donor_label"] != j["label_h"])
j["donor_same_group"] = j["donor_group"].notna() & (j["donor_group"] == j["group_id_h"])

aff = j[j.affected].copy()
una = j[~j.affected].copy()
print(f"  affected rows            : {len(aff)}")
print(f"  donor image identified   : {int(aff.donor_found.sum())} / {len(aff)}")
print(f"  donor in SAME group      : {int(aff.donor_same_group.sum())} / {len(aff)}")
print(f"  donor label DIFFERS      : {int(aff.donor_label_differs.sum())} / {len(aff)}")
print(f"  donor label same         : {int((aff.donor_found & ~aff.donor_label_differs).sum())}")
print()
for name, sub in (("same-label donor", aff[aff.donor_found & ~aff.donor_label_differs]),
                  ("DIFFERENT-label donor", aff[aff.donor_label_differs])):
    c = sub.label_h.value_counts().to_dict()
    print(f"  {name:24s} n={len(sub):4d}  recipient labels={c}")

# ---------------------------------------------------------------- geometry
print()
print("=" * 100)
print("B. ACQUISITION GEOMETRY (from mask dimensions)")
print("=" * 100)
paths = list(j.mask_path_c.unique())
shp = {}
for path in paths:
    try:
        with Image.open(path) as im:
            w, h = im.size
        shp[path] = min(h, w)
    except Exception:  # noqa: BLE001
        shp[path] = -1
j["min_side"] = j.mask_path_c.map(shp)
j["geometry"] = j.min_side.map(GEOMETRY_BY_MINSIDE).fillna("other_" + j.min_side.astype(str))
aff = j[j.affected].copy()
print(f"  geometries present (all 8870): {sorted(j.geometry.unique())}")
print(f"  per geometry, all rows : {j.geometry.value_counts().to_dict()}")
print(f"  per geometry, affected : {aff.geometry.value_counts().to_dict()}")

# ---------------------------------------------------------------- per-feature deltas
print()
print("=" * 100)
print("C. PER-FEATURE PERTURBATION ON THE 610 AFFECTED ROWS (H)")
print("=" * 100)
cstat = contract[contract.feature_version == "pvbm_v1"].set_index("feature_name")

rows, split_rows, geom_rows, donor_rows = [], [], [], []
rowlevel = []
for f in FEATS:
    a = aff[f"{f}_h"].astype(float).values
    b = aff[f"{f}_c"].astype(float).values
    na, nb = np.isnan(a), np.isnan(b)
    both = na & nb
    mismatch = na ^ nb
    d = b - a
    changed = (d != 0) & ~both | mismatch
    n_ch = int(changed.sum())
    ad = np.abs(d[changed & ~mismatch])
    rel = np.abs(d[changed & ~mismatch]) / np.maximum(np.abs(a[changed & ~mismatch]), 1e-12)
    rows.append({
        "feature_name": f,
        "n_total": int(len(j)),
        "n_affected_rows": int(len(aff)),
        "n_changed_rows": n_ch,
        "median_abs_delta": float(np.median(ad)) if len(ad) else 0.0,
        "p95_abs_delta": float(np.percentile(ad, 95)) if len(ad) else 0.0,
        "max_abs_delta": float(ad.max()) if len(ad) else 0.0,
        "median_relative_delta": float(np.median(rel)) if len(rel) else 0.0,
        "p95_relative_delta": float(np.percentile(rel, 95)) if len(rel) else 0.0,
        "n_nan_mismatch": int(mismatch.sum()),
        "n_delta_positive": int((d[changed & ~mismatch] > 0).sum()),
        "n_delta_negative": int((d[changed & ~mismatch] < 0).sum()),
        "historical_status": cstat.clinical_status.get(f, "unlisted"),
        "limitation_marking": ("HISTORICALLY_REPRODUCED_SCI_LIMITATION_KNOWN"
                              if f in LIMITATION_FEATURES else ""),
    })
    for key, bucket in (("split_h", split_rows), ("source_h", geom_rows),
                        ("label_h", donor_rows)):
        for val, sub in aff.groupby(key):
            aa = sub[f"{f}_h"].astype(float).values
            bb = sub[f"{f}_c"].astype(float).values
            n = np.isnan(aa) & np.isnan(bb)
            mm = np.isnan(aa) ^ np.isnan(bb)
            dd = np.abs(bb - aa)
            ok = ~n & ~mm
            ch = (dd != 0) & ok
            bucket.append({
                "dimension": key, "value": str(val), "feature_name": f,
                "n_rows": len(sub), "n_changed": int(ch.sum()) + int(mm.sum()),
                "median_abs_delta": float(np.median(dd[ch])) if ch.sum() else 0.0,
                "p95_abs_delta": float(np.percentile(dd[ch], 95)) if ch.sum() else 0.0,
            })
    for val, sub in aff.groupby("geometry"):
        aa = sub[f"{f}_h"].astype(float).values
        bb = sub[f"{f}_c"].astype(float).values
        n = np.isnan(aa) & np.isnan(bb)
        mm = np.isnan(aa) ^ np.isnan(bb)
        dd = np.abs(bb - aa)
        ok = ~n & ~mm
        ch = (dd != 0) & ok
        geom_rows.append({
            "dimension": "geometry", "value": str(val), "feature_name": f,
            "n_rows": len(sub), "n_changed": int(ch.sum()) + int(mm.sum()),
            "median_abs_delta": float(np.median(dd[ch])) if ch.sum() else 0.0,
            "p95_abs_delta": float(np.percentile(dd[ch], 95)) if ch.sum() else 0.0,
        })

P = pd.DataFrame(rows)
P.to_csv(f"{ART}/historical_feature_correction_summary.csv", index=False)
print(f"  {'feature':26s} {'changed':>8s} {'medabs':>12s} {'p95abs':>12s} {'maxabs':>12s} "
      f"{'medrel':>10s} {'+':>5s} {'-':>5s}  status")
for _, r in P.iterrows():
    print(f"  {r.feature_name:26s} {int(r.n_changed_rows):8d} {r.median_abs_delta:12.6g} "
          f"{r.p95_abs_delta:12.6g} {r.max_abs_delta:12.6g} {r.median_relative_delta:10.4g} "
          f"{int(r.n_delta_positive):5d} {int(r.n_delta_negative):5d}  {r.historical_status}")
print()
print(f"  -> {ART}/historical_feature_correction_summary.csv")

# ---------------------------------------------------------------- different-label subset (I)
print()
print("=" * 100)
print("D. DIFFERENT-LABEL DONOR SUBSET (I)")
print("=" * 100)
dl = aff[aff.donor_label_differs]
sl = aff[aff.donor_found & ~aff.donor_label_differs]
out = []
for f in FEATS:
    r = {"feature_name": f}
    for nm, sub in (("difflabel", dl), ("samelabel", sl)):
        a = sub[f"{f}_h"].astype(float).values
        b = sub[f"{f}_c"].astype(float).values
        n = np.isnan(a) & np.isnan(b)
        mm = np.isnan(a) ^ np.isnan(b)
        d = np.abs(b - a)
        ok = ~n & ~mm
        ch = (d != 0) & ok
        scale = float(np.nanmax(np.abs(a[~n]))) if (~n).any() else 1.0
        r[f"{nm}_n"] = len(sub)
        r[f"{nm}_n_changed"] = int(ch.sum()) + int(mm.sum())
        r[f"{nm}_median_abs"] = float(np.median(d[ch])) if ch.sum() else 0.0
        r[f"{nm}_normalised"] = (float(np.median(d[ch])) / scale) if (ch.sum() and scale) else 0.0
    out.append(r)
D = pd.DataFrame(out)
D.to_csv(f"{PRIV}/task4_difflabel_donor_subset.csv", index=False)
print(f"  different-label subset n={len(dl)}   same-label subset n={len(sl)}")
print(f"  {'feature':26s} {'dl_chg':>7s} {'dl_med':>12s} {'dl_norm':>10s} "
      f"{'sl_med':>12s} {'sl_norm':>10s}  ratio")
for _, r in D.iterrows():
    ratio = (r.difflabel_normalised / r.samelabel_normalised
             if r.samelabel_normalised else float("nan"))
    print(f"  {r.feature_name:26s} {int(r.difflabel_n_changed):7d} {r.difflabel_median_abs:12.6g} "
          f"{r.difflabel_normalised:10.5f} {r.samelabel_median_abs:12.6g} "
          f"{r.samelabel_normalised:10.5f}  {ratio:6.3f}")
med_ratio = float(np.nanmedian([r.difflabel_normalised / r.samelabel_normalised
                                for _, r in D.iterrows() if r.samelabel_normalised]))
print()
print(f"  MEDIAN NORMALISED PERTURBATION RATIO (difflabel / samelabel) = {med_ratio:.3f}")
print(f"  -> {'LARGER' if med_ratio > 1.05 else 'NOT LARGER'} for the different-label subset")

# ---------------------------------------------------------------- split contamination (J)
print()
print("=" * 100)
print("E. SPLIT-LEVEL CONTAMINATION (J)")
print("=" * 100)
jc = j.merge(canon.rename(columns={"mask_path": "cm"}), on="image_path")
contam = []
for s in ("train", "val", "test"):
    sub = j[j.split_h == s]
    n_aff = int(sub.affected.sum())
    contam.append({"split": s, "n_total": len(sub), "n_wrong_mask_historical": n_aff,
                   "percent_wrong": 100.0 * n_aff / len(sub)})
    print(f"  {s:6s} n_total={len(sub):5d}  n_wrong_mask={n_aff:4d}  "
          f"percent_wrong={100.0 * n_aff / len(sub):6.2f}%")
C = pd.DataFrame(contam)
print(f"  VERIFY 414/106/90 : "
      f"{C.set_index('split').n_wrong_mask_historical.to_dict()}")
C.to_csv(f"{ART}/historical_feature_split_contamination.csv", index=False)

sp = pd.concat([pd.DataFrame(split_rows), pd.DataFrame(geom_rows), pd.DataFrame(donor_rows)])
sp.to_csv(f"{PRIV}/task4_subgroup_perturbation.csv", index=False)
print()
print("  per-split median |delta|, aggregated over the 23 features (mean of medians):")
for s in ("train", "val", "test"):
    x = sp[(sp.dimension == "split_h") & (sp.value == s)]
    print(f"    {s:6s} mean-of-medians={x.median_abs_delta.mean():.6g}  "
          f"mean p95={x.p95_abs_delta.mean():.6g}")
print(f"  -> {PRIV}/task4_subgroup_perturbation.csv")

# ---------------------------------------------------------------- row-level (Q)
print()
print("=" * 100)
print("F. PRIVATE ROW-LEVEL DELTAS (Q)")
print("=" * 100)
recs = []
for _, r in aff.iterrows():
    base = {
        "stable_image_id": hid(r.image_path),
        "hashed_group_id": hid(r.group_id_h),
        "split": r.split_h, "source": r.source_h, "label": int(r.label_h),
        "geometry": r.geometry,
        "donor_same_group": bool(r.donor_same_group),
        "donor_label_differs": bool(r.donor_label_differs),
    }
    for f in FEATS:
        a, b = r[f"{f}_h"], r[f"{f}_c"]
        try:
            a, b = float(a), float(b)
        except Exception:  # noqa: BLE001
            a, b = np.nan, np.nan
        delta = b - a
        recs.append({**base, "feature_name": f, "historical_value": a,
                     "corrected_value": b, "delta": delta,
                     "abs_delta": abs(delta) if np.isfinite(delta) else np.nan,
                     "relative_delta": (abs(delta) / abs(a)) if (np.isfinite(delta) and a)
                     else np.nan})
RL = pd.DataFrame(recs)
RL.to_csv(f"{PRIV}/historical_feature_row_deltas.csv", index=False)
print(f"  rows={len(RL)} (610 images x {len(FEATS)} features)")
print(f"  -> {PRIV}/historical_feature_row_deltas.csv")

# ---------------------------------------------------------------- integrity (O)
print()
print("=" * 100)
print("G. TABLE INTEGRITY OF THE CORRECTED TABLE (O)")
print("=" * 100)
splits = pd.read_csv(f"{ROOT}/data/splits/all.csv")
ck = {
    "rows_8870": len(corr) == 8870,
    "unique_image_ids_8870": corr.image_path.nunique() == 8870,
    "zero_duplicate_rows": int(corr.duplicated(subset=["image_path"]).sum()) == 0,
    "matches_locked_split_rows": len(corr) == len(splits),
    "matches_locked_split_ids": set(corr.image_path) == set(splits.image_path),
    "feature_count_23": len(FEATS) == 23,
    "no_inf_values": not np.isinf(corr[FEATS].to_numpy(dtype=float)).any(),
    "column_order_matches_historical": list(corr.columns) == list(hist.columns),
    "no_metadata_in_predictors": not (set(FEATS) & META),
}
for k, v in ck.items():
    print(f"  {k:34s} : {v}")
print()
print("  source counts  corrected :", corr.source.value_counts().to_dict())
print("  source counts  historical:", hist.source.value_counts().to_dict())
print("  label counts   corrected :", corr.label.value_counts().sort_index().to_dict())
print("  label counts   historical:", hist.label.value_counts().sort_index().to_dict())
print("  split counts   corrected :", corr.split.value_counts().to_dict())
print("  split counts   historical:", hist.split.value_counts().to_dict())
print()
print("  NaN counts per feature (corrected / historical):")
for f in FEATS:
    a, b = int(corr[f].isna().sum()), int(hist[f].isna().sum())
    if a or b:
        print(f"    {f:26s} corrected={a:4d}  historical={b:4d}")

# ---------------------------------------------------------------- summary
summary = {
    "AFFECTED_ROWS": int(len(aff)),
    "UNAFFECTED_ROWS": int(len(una)),
    "DONOR_IDENTIFIED": int(aff.donor_found.sum()),
    "DONOR_SAME_GROUP": int(aff.donor_same_group.sum()),
    "DONOR_LABEL_DIFFERS": int(aff.donor_label_differs.sum()),
    "DIFFLABEL_NORMALISED_RATIO": med_ratio,
    "SPLIT_CONTAMINATION": {r["split"]: r["n_wrong_mask_historical"] for r in contam},
    "PERCENT_WRONG": {r["split"]: round(r["percent_wrong"], 2) for r in contam},
    "FEATURES_EXACTLY_UNCHANGED_ON_ALL_610": [
        r.feature_name for _, r in P.iterrows() if int(r.n_changed_rows) == 0],
    "FEATURES_CHANGED_ON_ALL_610": [
        r.feature_name for _, r in P.iterrows() if int(r.n_changed_rows) == len(aff)],
    "GEOMETRY_AFFECTED": aff.geometry.value_counts().to_dict(),
}
json.dump(summary, open(f"{PRIV}/task4_delta_summary.json", "w"), indent=2, default=str)
print()
print("=" * 100)
for k, v in summary.items():
    print(f"  {k}: {v}")
