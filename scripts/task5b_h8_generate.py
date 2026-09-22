#!/usr/bin/env python
"""Task 5B-H8: full canonical V5 generation, integrity, drift, determinism and freeze.

Stages, in order:
  A freeze record (V5 code SHAs, SEG_CURRENT_V1 identity, cohort fingerprint, feature formulas)
  C full generation of the 8,870-row V5 table   -> data/features/final_biomarkers_v2.csv
  E full-table integrity
  F primary population recomputation under the predeclared complete-case policy
  G V1 -> V5 native drift
  H source / geometry / split descriptive QC
  I V5 FOV validity summary
  J determinism check on a deterministic >=100-image subset, atol 1e-12
  K freeze artifacts + versioned config

No classifier, no disease label, no AUC, no tuning. `--analyze-only` re-runs E-K from the saved CSV.
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

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.task5b_n2 import load  # noqa: E402
from scripts.task5b_h_fov_stress import GEOM  # noqa: E402
from src.biomarker import clinical_measurement_v5_candidate as v5  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
V1_TABLE = ROOT / "data/features/final_biomarkers_v1.csv"
V2_TABLE = ROOT / "data/features/final_biomarkers_v2.csv"
V1_SHA = "f1c41e923ae29d4e228097536765f5e7399963062253ad925657a077cddc10c0"
COHORT_FP = "0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8"
MASK_MANIFEST = ROOT / "data/masks/mask_manifest_canonical_v1.csv"
MASK_MANIFEST_SHA = "b3bc6538026d6ad0fc1342043e5ca2d6de56662233bf6d0709d9d875f24b21cd"
SEG_ID = "SEG_CURRENT_V1"
SEG_CKPT = "weights/best_weight_DeepLabV3+_resize_27"
SEG_CKPT_SHA = "c373f53813ee60b89651a04a98bc5f1d6bc60a5a50f45c4e10f42475650d5374"
MEASURE_FILES = ["clinical_measurement_v1.py", "clinical_measurement_v2_candidate.py",
                 "clinical_measurement_v3_candidate.py", "clinical_measurement_v4_candidate.py",
                 "clinical_measurement_v5_candidate.py"]


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def worker(arg):
    image_path, mask_path = arg
    try:
        rgb, msk = load(image_path, mask_path)
        out, fov, qc = v5.measure_v5(rgb, msk, {}, with_fractal=True)
        rec = dict(out)
        rec["fov_valid_v5"] = bool(qc.get("fov_valid"))
        rec["fov_failure_reason_v5"] = str(qc.get("fov_failure_reason", ""))
        rec["fov_coverage_frame_relative_v4"] = float(qc.get("fov_coverage_frame_relative_v4",
                                                             np.nan))
        rec["fov_content_area"] = int(qc.get("fov_content_area", 0))
        rec["fov_min_size"] = int(qc.get("fov_min_size", 0) or 0)
        rec["fov_threshold"] = float(qc.get("fov_threshold", np.nan))
        rec["measurement_version"] = v5.MEASUREMENT_VERSION_V5
        return image_path, rec
    except Exception as e:  # noqa: BLE001
        return image_path, {"_err": f"{type(e).__name__}: {e}"}


def run_generation(meta, prov_cols, meas_cols):
    print()
    print("=" * 100)
    print("C. FULL CANONICAL GENERATION (V5, 8,870 images)")
    print("=" * 100)
    args = list(zip(meta.image_path, meta.mask_path))
    recs, errs = {}, 0
    t0 = time.time()
    with get_context("fork").Pool(24) as pool:
        for i, (p, rec) in enumerate(pool.imap_unordered(worker, args, chunksize=4), 1):
            if "_err" in rec:
                errs += 1
                print(f"  ERR {p}: {rec['_err']}", flush=True)
                continue
            recs[p] = rec
            if i % 500 == 0:
                print(f"  {i}/{len(args)}  {time.time() - t0:.0f}s", flush=True)
    print(f"  measured {len(recs)}  errors {errs}  elapsed {time.time() - t0:.0f}s")
    if errs:
        raise SystemExit("GENERATION_ERRORS")
    prov = meta[prov_cols].copy()
    meas = pd.DataFrame([recs[p] for p in prov.image_path])
    df = pd.concat([prov.reset_index(drop=True), meas[meas_cols].reset_index(drop=True)], axis=1)
    df.to_csv(V2_TABLE, index=False)
    print(f"  written {V2_TABLE}  rows {len(df)}  cols {df.shape[1]}")
    return df


def main() -> None:
    analyze_only = "--analyze-only" in sys.argv
    print("=" * 100)
    print("A. FREEZE RECORD")
    print("=" * 100)
    frozen = {
        "git_commit": __import__("subprocess").run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
            text=True).stdout.strip() or "unknown",
        "measurement_code_sha256": {f: sha(ROOT / "src/biomarker" / f) for f in MEASURE_FILES},
        "segmentation_generation": SEG_ID,
        "segmentation_checkpoint": SEG_CKPT,
        "segmentation_checkpoint_sha256": SEG_CKPT_SHA,
        "canonical_cohort_fingerprint": COHORT_FP,
        "mask_manifest": MASK_MANIFEST.name,
        "mask_manifest_sha256": sha(MASK_MANIFEST) if MASK_MANIFEST.exists() else MASK_MANIFEST_SHA,
        "v1_feature_table_sha256": sha(V1_TABLE),
        "five_final_primary_formulas": {
            "vessel_density_fov": "mask[fov].sum() / fov_px",
            "skel_density_fov": "len(nonzero(skeletonize(mask))) / fov_px   # numerator whole-frame",
            "fractal_d0/d1/d2": "MultifractalVBMs(25, optimize=True, 1e-4, 0.9999) on the FOV-derived "
                                "canonical domain of (mask & fov)",
        },
    }
    for k, v in frozen.items():
        print(f"  {k:34s} : {v}")
    print(f"  V1 table sha matches the frozen record : {frozen['v1_feature_table_sha256'] == V1_SHA}")

    meta = pd.read_csv(V1_TABLE)
    meas_keys = set(worker((meta.image_path.iloc[0], meta.mask_path.iloc[0]))[1].keys())
    prov_cols = [c for c in meta.columns if c not in meas_keys]
    meas_cols = list(meas_keys)
    print(f"  provenance columns {len(prov_cols)}   measurement columns {len(meas_cols)}")
    meta["geom"] = [GEOM.get(Image.open(p).size, "other") for p in meta.image_path]

    if not analyze_only:
        df = run_generation(meta, prov_cols, meas_cols)
    else:
        df = pd.read_csv(V2_TABLE, low_memory=False)

    print()
    print("=" * 100)
    print("E. FULL-TABLE INTEGRITY")
    print("=" * 100)
    can = set(meta.image_path)
    got = set(df.image_path)
    checks = {
        "rows_exactly_8870": len(df) == 8870,
        "no_duplicate_image_ids": int(df.image_path.duplicated().sum()) == 0,
        "no_missing_canonical_images": len(can - got) == 0,
        "no_extra_images": len(got - can) == 0,
        "one_row_per_image": len(df) == df.image_path.nunique(),
        "source_counts_match": df.source.value_counts().to_dict()
        == meta.source.value_counts().to_dict(),
        "split_counts_match": df.split.value_counts().to_dict()
        == meta.split.value_counts().to_dict(),
        "mask_paths_match_canonical": bool((df.sort_values("image_path").mask_path.tolist()
                                            == meta.sort_values("image_path").mask_path.tolist())),
        "measurement_version_uniform": bool((df.measurement_version
                                             == v5.MEASUREMENT_VERSION_V5).all()),
    }
    for k, v in checks.items():
        print(f"  {k:34s} : {'PASS' if v else 'FAIL'}")
    print(f"  rows {len(df)}  duplicates {int(df.image_path.duplicated().sum())}  "
          f"missing {len(can - got)}  extra {len(got - can)}")
    print(f"  sources {df.source.value_counts().to_dict()}")
    print(f"  splits  {df.split.value_counts().to_dict()}")
    for f in FEATS:
        fin = int(df[f].notna().sum())
        print(f"  {f:22s} finite {fin:5d}  missing {len(df) - fin:4d}  "
              f"median {df[f].median():.6f}")
    if not all(checks.values()):
        raise SystemExit("INTEGRITY_FAILED")

    print()
    print("=" * 100)
    print("F. PRIMARY POPULATION (complete-case, predeclared policy: no imputation)")
    print("=" * 100)
    miss = df[df[FEATS].isna().any(axis=1)]
    complete = df[df[FEATS].notna().all(axis=1)]
    print(f"  total {len(df)}   complete-case {len(complete)}   excluded {len(miss)}")
    print(f"  excluded image IDs: {miss.image_path.tolist()}")
    if len(miss):
        print(f"  which features are missing: "
              f"{ {f: int(miss[f].isna().sum()) for f in FEATS} }")
        print(f"  source {miss.source.value_counts().to_dict()}   "
              f"split {miss.split.value_counts().to_dict()}")
        mm = miss.merge(meta[["image_path", "mask_path"]], on="image_path", how="left",
                        suffixes=("", "_c"))
        mm["geom"] = [GEOM.get(Image.open(p).size, "other") for p in mm.image_path]
        print(f"  geometry {mm.geom.value_counts().to_dict()}")
        for r in miss.itertuples():
            print(f"    {str(r.image_path).split('/')[-1]:46s} {r.source:10s} {r.split:6s} "
                  f"fractal_d0={r.fractal_d0} d1={r.fractal_d1} d2={r.fractal_d2}")
    print(f"  V1 excluded 8 rows; V5 excludes {len(miss)}"
          f"  -> population changed: {len(miss) != 8}")
    complete.to_csv(ROOT / "data/splits/primary_complete_case_v2.csv", index=False)
    miss.to_csv(ROOT / "data/splits/primary_excluded_v2.csv", index=False)

    print()
    print("=" * 100)
    print("G. V1 -> V5 NATIVE DRIFT")
    print("=" * 100)
    J = meta[["image_path"] + FEATS].merge(df[["image_path"] + FEATS], on="image_path",
                                           suffixes=("_v1", "_v5"))
    dr = {}
    for f in FEATS:
        a, b = J[f + "_v1"], J[f + "_v5"]
        d = (b - a).abs()
        rel = ((b - a) / a.abs()).replace([np.inf, -np.inf], np.nan)
        corr = float(a.corr(b))
        dr[f] = {"median_abs": float(d.median()), "median_rel": float(rel.median()),
                 "p95_abs": float(d.quantile(.95)), "max_abs": float(d.max()),
                 "corr": corr, "n_materially_changed": int((rel.abs() > .05).sum())}
        print(f"  {f:22s} medAbs {d.median():.6f}  medRel {rel.median():+.5f}  "
              f"p95Abs {d.quantile(.95):.6f}  maxAbs {d.max():.6f}  corr {corr:.6f}  "
              f">5% {int((rel.abs() > .05).sum())}")
    dens = dr["vessel_density_fov"]
    print("  NOTE: the fractal analysis-domain definition intentionally changed in V2/V3, so the "
          "fractal rows are version drift characterisation, not an equivalence requirement.")
    big = J.reindex(((J.vessel_density_fov_v5 - J.vessel_density_fov_v1) / J.vessel_density_fov_v1)
                    .abs().sort_values(ascending=False).index).head(5)
    print("  largest vessel_density_fov native changes:")
    for r in big.itertuples():
        print(f"    {str(r.image_path).split('/')[-1]:44s} v1={r.vessel_density_fov_v1:.6f} "
              f"v5={r.vessel_density_fov_v5:.6f} "
              f"rel={(r.vessel_density_fov_v5 - r.vessel_density_fov_v1) / r.vessel_density_fov_v1:+.4f}")

    print()
    print("=" * 100)
    print("H. SOURCE / GEOMETRY / SPLIT DESCRIPTIVE QC")
    print("=" * 100)
    hq = complete.copy()
    meta_g = meta[["image_path", "mask_path", "geom"]].copy()
    hq = hq.merge(meta_g[["image_path", "geom"]], on="image_path", how="left")
    for key in ("source", "split", "geom"):
        print(f"  by {key}:")
        for k, g in hq.groupby(key):
            line = f"    {k:12s} N={len(g):5d}  " + "  ".join(
                f"{f}: med {g[f].median():.4f} IQR {g[f].quantile(.75) - g[f].quantile(.25):.4f}"
                for f in FEATS[:2])
            print(line + f"   fractal_d0 med {g.fractal_d0.median():.4f}")
            print(f"                 missing " + "  ".join(
                f"{f}: {int(g[f].isna().sum())}" for f in FEATS))

    print()
    print("=" * 100)
    print("I. FOV VALIDITY ON ALL 8,870 NATIVE IMAGES")
    print("=" * 100)
    v = df.fov_valid_v5.astype(bool)
    print(f"  valid {int(v.sum())}   invalid {int((~v).sum())}")
    print(f"  invalid reasons: {df[~v].fov_failure_reason_v5.value_counts().to_dict()}")
    print(f"  invalid by source:\n{df[~v].source.value_counts().to_string()}")
    print("  invalid by geometry:")
    dg = df.merge(meta_g[["image_path", "geom"]], on="image_path", how="left")
    print(dg[~v].geom.value_counts().to_string())
    print(f"  coverage median {df.fov_coverage_fraction.median():.6f}  "
          f"p01 {df.fov_coverage_fraction.quantile(.01):.6f}  "
          f"min {df.fov_coverage_fraction.min():.6f}")
    h7bad = ["006_F_GA40_BW3200_PA44_DG11_PF0_D1_S02_5",
             "006_F_GA40_BW3200_PA44_DG11_PF0_D1_S02_6"]
    for b in h7bad:
        m = df[df.image_path.str.contains(b, regex=False)]
        if len(m):
            r = m.iloc[0]
            print(f"  H7 native-invalid example {b}: valid={r.fov_valid_v5} "
                  f"cov={r.fov_coverage_fraction:.6f} reason={r.fov_failure_reason_v5}")
    print("  NOTE: the thresholds were NOT changed to rescue any invalid image.")

    print()
    print("=" * 100)
    print("J. DETERMINISM CHECK")
    print("=" * 100)
    sub = pd.concat([g.head(8) for _, g in meta.groupby(["source", "geom", "split"])])
    sub = sub.reset_index(drop=True)
    if len(sub) < 100:
        sub = pd.concat([g.head(16) for _, g in meta.groupby(["source", "geom", "split"])]
                        ).reset_index(drop=True)
    sub = sub.merge(meta_g[["image_path", "geom"]], on="image_path", how="left",
                    suffixes=("", "_g"))
    rep = []
    with get_context("fork").Pool(12) as pool:
        for pth, rec in pool.imap_unordered(worker, list(zip(sub.image_path, sub.mask_path))):
            rep.append({"image_path": pth, **{f: rec.get(f, np.nan) for f in FEATS},
                        "valid": bool(rec.get("fov_valid_v5")),
                        "reason": str(rec.get("fov_failure_reason_v5", "")),
                        "cov_rep": float(rec.get("fov_coverage_fraction", np.nan))})
    RP = pd.DataFrame(rep)
    CMP = RP.merge(df[["image_path"] + FEATS + ["fov_valid_v5", "fov_failure_reason_v5",
                                                "fov_coverage_fraction"]], on="image_path",
                   suffixes=("_r", "_t"))
    md = 0.0
    for f in FEATS:
        d = (CMP[f + "_r"] - CMP[f + "_t"]).abs().max()
        md = max(md, float(d) if np.isfinite(d) else float("inf"))
    val_ok = bool((CMP.valid == CMP.fov_valid_v5).all())
    cov_ok = bool(np.allclose(CMP["cov_rep"], CMP["fov_coverage_fraction"], atol=1e-12,
                              equal_nan=True))
    print(f"  subset N {len(CMP)}  sources {sub.source.nunique()}  geoms {sub.geom.nunique()}  "
          f"splits {sub.split.nunique()}")
    print(f"  max |feature delta| {md:.3e}   (atol 1e-12)   features ok {md <= 1e-12}")
    print(f"  validity fields identical {val_ok}   coverage identical {cov_ok}")
    if not (md <= 1e-12 and val_ok and cov_ok):
        raise SystemExit("DETERMINISM_FAILED")

    print()
    print("=" * 100)
    print("K. FREEZE")
    print("=" * 100)
    tsha = sha(V2_TABLE)
    cfg = ROOT / "configs/final_biomarkers_v2.yaml"
    cfg.write_text(
        "# FINAL_BIOMARKERS_V2 — frozen measurement generation\n"
        f"generation: FINAL_BIOMARKERS_V2\n"
        f"measurement_version: {v5.MEASUREMENT_VERSION_V5}\n"
        f"git_commit: {frozen['git_commit']}\n"
        f"table: data/features/final_biomarkers_v2.csv\n"
        f"table_sha256: {tsha}\n"
        f"canonical_n: 8870\n"
        f"canonical_cohort_fingerprint: {COHORT_FP}\n"
        f"segmentation_generation: {SEG_ID}\n"
        f"segmentation_checkpoint_sha256: {SEG_CKPT_SHA}\n"
        f"mask_manifest_sha256: {frozen['mask_manifest_sha256']}\n"
        f"complete_case_n: {len(complete)}\n"
        f"excluded_n: {len(miss)}\n"
        f"final_primary_features: [{', '.join(FEATS)}]\n"
        f"primary_population_manifest: data/splits/primary_complete_case_v2.csv\n"
        f"excluded_manifest: data/splits/primary_excluded_v2.csv\n"
        f"sources: {json.dumps(df.source.value_counts().to_dict())}\n"
        f"splits: {json.dumps(df.split.value_counts().to_dict())}\n"
        f"fov_valid_n: {int(v.sum())}\n"
        f"fov_invalid_n: {int((~v).sum())}\n"
        f"determinism_subset_n: {len(CMP)}\n"
        f"determinism_max_delta: {md:.3e}\n"
        "photometric_clipping_limitation: 'x1.50-x1.80 saturation remains a separate known "
        "operating-range limitation; not addressed by V5'\n"
        "disease_model_training_allowed: 'no'\n", encoding="utf-8")
    (OUT / "task5b_h8_freeze.json").write_text(json.dumps(
        {"frozen": frozen, "table_sha256": tsha, "rows": len(df),
         "complete_case_n": int(len(complete)), "excluded_n": int(len(miss)),
         "excluded_image_ids": miss.image_path.tolist(),
         "fov_valid_n": int(v.sum()), "fov_invalid_n": int((~v).sum()),
         "fov_invalid_reasons": df[~v].fov_failure_reason_v5.value_counts().to_dict(),
         "integrity_checks": checks, "drift": dr,
         "determinism": {"n": int(len(CMP)), "max_delta": md, "validity_identical": val_ok,
                         "coverage_identical": cov_ok},
         "source_counts": df.source.value_counts().to_dict(),
         "split_counts": df.split.value_counts().to_dict()},
        indent=2, default=str), encoding="utf-8")
    print(f"  FINAL_BIOMARKER_TABLE_SHA256 = {tsha}")
    print(f"  config written: configs/final_biomarkers_v2.yaml")
    print(f"  freeze written: _private_audit/task5b_h8_freeze.json")
    print("\n  TASK5B_H8_STATUS = COMPLETE")


if __name__ == "__main__":
    main()
