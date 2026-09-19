#!/usr/bin/env python
"""Phase 0 + Phase 1 — Freeze manifest and redundancy diagnostic.

Reads : data/features/biomarker_features.csv
        data/splits/all.csv
        results/hybrid_v2/embeddings_current.npz  (+ .json)
Writes: results/fusion_diagnostic_v1/  (new folder; nothing existing is modified)

Discipline:
  * Row order of the embedding cache is pinned by the order hash recorded in the manifest.
    The recipe (documented only in scripts/hybrid_v2_experiment.py and scripts/lowcost_fusion_suite.py)
    is: order by split (train, val, test) then by image_path.
  * Alignment is verified with the ORDER HASH plus VALIDATION-set agreement only.
    The locked test is not read for any number in this script.
  * No preprocessing statistic is fitted on val or test.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/moniaz/niki")
os.chdir(ROOT)
OUT = ROOT / "results" / "fusion_diagnostic_v1"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
PLUS_IDX = 2
META = {
    "image_path", "mask_path", "label", "split", "source",
    "group_id", "patient_id", "exam_id", "identity_level",
}
FAMILIES = {
    "density": ["vessel_density", "vessel_pixels"] + [f"density_r{i}c{j}" for i in range(3) for j in range(3)],
    "geometry": ["area", "tortuosity_index", "median_tortuosity", "overall_length",
                 "median_branching_angle", "n_startpoints", "n_endpoints", "n_intersections"],
    "fractal": ["fractal_d0", "fractal_d1", "fractal_d2", "singularity_length"],
}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- Phase 0: freeze
log("=== Phase 0: freeze ===")
inputs = {
    "data/features/biomarker_features.csv": ROOT / "data/features/biomarker_features.csv",
    "data/splits/all.csv": ROOT / "data/splits/all.csv",
    "results/hybrid_v2/embeddings_current.npz": ROOT / "results/hybrid_v2/embeddings_current.npz",
    "results/hybrid_v2/embeddings_current.json": ROOT / "results/hybrid_v2/embeddings_current.json",
    "results/branch_b_results.json": ROOT / "results/branch_b_results.json",
}
hashes = {name: sha_file(p) for name, p in inputs.items()}
for name, h in hashes.items():
    log(f"  {name}: {h}")

manifest = json.loads(inputs["results/hybrid_v2/embeddings_current.json"].read_text())
log(f"  manifest: {json.dumps(manifest)}")

# ---------------------------------------------------------------- load + align
raw = pd.read_csv(inputs["data/features/biomarker_features.csv"])
log(f"\nfeature table: {raw.shape}")

split_order = {"train": 0, "val": 1, "test": 2}
frame = raw.copy()
frame["_o"] = frame["split"].map(split_order)
frame = frame.sort_values(["_o", "image_path"]).drop(columns="_o").reset_index(drop=True)
order_sha = hashlib.sha256("\n".join(frame["image_path"]).encode()).hexdigest()

log(f"computed order hash: {order_sha}")
log(f"manifest order hash: {manifest.get('image_order_sha256')}")
if order_sha != manifest.get("image_order_sha256"):
    log("FATAL: order hash mismatch — alignment not proven. Aborting without writing results.")
    sys.exit(2)
log("ORDER HASH MATCHES -> embedding rows are aligned to `frame`.")

emb = np.load(inputs["results/hybrid_v2/embeddings_current.npz"])["embeddings"].astype(np.float64)
if len(emb) != len(frame):
    log(f"FATAL: row mismatch {len(emb)} vs {len(frame)}")
    sys.exit(2)

y3 = frame["label"].to_numpy(int)
y_plus = (y3 == PLUS_IDX).astype(int)
tr = (frame["split"] == "train").to_numpy()
va = (frame["split"] == "val").to_numpy()
te = (frame["split"] == "test").to_numpy()
groups = frame["group_id"].to_numpy()

log(f"splits: train={tr.sum()} val={va.sum()} test={te.sum()}")
log("NOTE: the locked test is loaded but NOT used for any number in this script.")

# ---------------------------------------------------------------- alignment sanity (VALIDATION only)
log("\n=== alignment sanity on VALIDATION (no test) ===")
scaler = StandardScaler().fit(emb[tr])
clf = LogisticRegression(C=0.001, max_iter=2000).fit(scaler.transform(emb[tr]), y_plus[tr])
p_val = clf.predict_proba(scaler.transform(emb[va]))[:, 1]
auc_val = roc_auc_score(y_plus[va], p_val)
log(f"embedding-only logistic C=0.001  VAL AUC = {auc_val:.6f}  (recorded 0.920788)")

bb_val = pd.read_csv(ROOT / "results/branch_b_val_preds.csv")
pcol = [c for c in bb_val.columns if c.startswith("p_plus")][0]
val_meta = frame.loc[va, ["image_path"]].assign(our=p_val)
mg = val_meta.merge(bb_val[["image_path", pcol]], on="image_path", how="inner")
r = float(np.corrcoef(mg["our"], mg[pcol])[0, 1]) if len(mg) > 100 else float("nan")
log(f"VAL correlation with recorded Branch B scores: r = {r:.4f}  (rows matched {len(mg)}/{int(va.sum())})")
alignment_ok = (order_sha == manifest.get("image_order_sha256")) and (auc_val > 0.85) and (r > 0.7)
log(f"ALIGNMENT VERDICT: {'OK' if alignment_ok else 'SUSPECT'}")
if not alignment_ok:
    log("FATAL: alignment sanity failed. Aborting without writing results.")
    sys.exit(3)

# ---------------------------------------------------------------- Phase 1: redundancy
log("\n=== Phase 1: redundancy (how much of each biomarker does the embedding already encode?) ===")
feat_cols = [c for c in frame.columns if c not in META]
log(f"biomarker columns ({len(feat_cols)}): {feat_cols}")
missing_by_feature = frame[feat_cols].isna().mean().to_dict()
medians = frame.loc[tr, feat_cols].median().fillna(0.0)
B = frame[feat_cols].fillna(medians).to_numpy(float)
Btr = B[tr]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
results = {}
for alpha in (1.0, 10.0, 100.0):
    oof = np.full_like(Btr, np.nan)
    t0 = time.time()
    for fold, (i_in, i_out) in enumerate(sgkf.split(Btr, y_plus[tr], groups[tr])):
        sc = StandardScaler().fit(emb[tr][i_in])
        X_in = sc.transform(emb[tr][i_in])
        X_out = sc.transform(emb[tr][i_out])
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_in, Btr[i_in])
        oof[i_out] = ridge.predict(X_out)
    r2 = {c: float(r2_score(Btr[:, k], oof[:, k])) for k, c in enumerate(feat_cols)}
    results[alpha] = r2
    log(f"  alpha={alpha}: done in {time.time() - t0:.1f}s  mean R2={np.mean(list(r2.values())):.4f}")

primary = results[10.0]
rows = []
for fam, cols in FAMILIES.items():
    for c in cols:
        rows.append({
            "feature": c,
            "family": fam,
            "r2_from_embedding_alpha10": primary.get(c, float("nan")),
            "r2_alpha1": results[1.0].get(c, float("nan")),
            "r2_alpha100": results[100.0].get(c, float("nan")),
            "missing_frac": float(missing_by_feature.get(c, float("nan"))),
        })
red = pd.DataFrame(rows).sort_values("r2_from_embedding_alpha10", ascending=False)
red.to_csv(OUT / "redundancy_by_feature.csv", index=False)
red[["feature", "family", "r2_from_embedding_alpha10", "missing_frac"]].to_csv(
    OUT / "redundancy_by_feature.csv", index=False)

fam_summary = (
    red.groupby("family")["r2_from_embedding_alpha10"]
    .agg(["mean", "median", "min", "max", "count"])
    .reset_index()
)
fam_summary.to_csv(OUT / "redundancy_by_family.csv", index=False)

log("\nper-feature R2 from the frozen embedding (alpha=10):")
for _, r_ in red.iterrows():
    log(f"  {r_['feature']:26s} {r_['family']:9s} R2={r_['r2_from_embedding_alpha10']:.4f}")
log("\nfamily summary:")
for _, r_ in fam_summary.iterrows():
    log(f"  {r_['family']:9s} mean={r_['mean']:.4f} median={r_['median']:.4f} "
        f"min={r_['min']:.4f} max={r_['max']:.4f}")

# ---------------------------------------------------------------- Phase 1b: does the embedding encode source?
log("\n=== Phase 1b: does the embedding itself encode acquisition source? ===")
src = frame["source"].to_numpy()
src_clf = LogisticRegression(max_iter=2000).fit(scaler.transform(emb[tr]), src[tr])
sp = src_clf.predict_proba(scaler.transform(emb[va]))
auc_src = roc_auc_score(src[va], sp, multi_class="ovr", average="macro", labels=src_clf.classes_)
acc_src = float((src_clf.predict(scaler.transform(emb[va])) == src[va]).mean())
major = float(pd.Series(src[va]).value_counts(normalize=True).max())
log(f"embedding -> source, VAL macro-OvR AUC = {auc_src:.4f}  accuracy = {acc_src:.4f}  (majority = {major:.4f})")
json.dump(
    {"val_macro_ovr_auc": float(auc_src), "val_accuracy": acc_src, "val_majority_accuracy": major,
     "n_features": int(emb.shape[1]), "model": "multinomial logistic on frozen embeddings"},
    open(OUT / "embedding_source_probe.json", "w"), indent=2,
)

# ---------------------------------------------------------------- write manifest + report
json.dump(
    {
        "phase": "0_freeze + 1_redundancy",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "input_sha256": hashes,
        "embedding_manifest": manifest,
        "order_recipe": "sort by split (train,val,test) then image_path",
        "order_sha_verified": True,
        "n_rows": int(len(frame)),
        "alignment_sanity": {"val_auc_embedding_only_C0.001": float(auc_val),
                             "val_corr_with_branch_b": r},
        "test_set_used": False,
        "note": "No number in this run was computed on the locked test set.",
    },
    open(OUT / "manifest.json", "w"), indent=2,
)

lines = [
    "# Phase 1 — Redundancy diagnostic",
    "",
    f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')} · seed {SEED} · rows {len(frame)}",
    "",
    "**Purpose.** Measure how much of each of the 23 biomarkers the frozen Branch-B embedding",
    "already represents. A high R² means the feature is redundant with the image representation",
    "and cannot plausibly add complementary information by concatenation.",
    "",
    "**Method.** Standardise embeddings on the training fold, fit a multi-output Ridge",
    "(alpha = 1 / 10 / 100) inside a 5-fold StratifiedGroupKFold on **train only**, and score",
    "out-of-fold predictions with R². Row order of the embedding cache verified against the",
    "manifest order hash; alignment additionally confirmed on validation (VAL AUC "
    f"{auc_val:.4f}, correlation with recorded Branch B scores r = {r:.3f}).",
    "",
    "**The locked test set was not used in this run.**",
    "",
    "## Family summary",
    "",
    "| Family | mean R² | median | min | max | n |",
    "|---|---|---|---|---|---|",
]
for _, r_ in fam_summary.iterrows():
    lines.append(f"| {r_['family']} | {r_['mean']:.4f} | {r_['median']:.4f} | {r_['min']:.4f} | {r_['max']:.4f} | {int(r_['count'])} |")
lines += ["", "## Per feature (alpha = 10)", "", "| Feature | Family | R² from embedding | missing |", "|---|---|---|---|"]
for _, r_ in red.iterrows():
    lines.append(f"| `{r_['feature']}` | {r_['family']} | {r_['r2_from_embedding_alpha10']:.4f} | {r_['missing_frac']:.4f} |")
lines += [
    "",
    "## Embedding → source probe",
    "",
    f"Macro one-vs-rest AUC = **{auc_src:.4f}**, accuracy {acc_src:.4f} vs majority {major:.4f}.",
    "",
    "## How to read this",
    "",
    "* **Low R² features** are the only candidates for a complementarity test — they carry",
    "  information the CNN does not already have.",
    "* **High R² features** explain why raw concatenation adds nothing: the information is",
    "  already in the representation.",
    "* A high embedding→source AUC means part of the representation encodes the camera/source,",
    "  which is the mechanism behind the LOSO drop.",
    "",
    "## Files",
    "",
    "* `manifest.json` — input hashes, verified order recipe, alignment evidence",
    "* `redundancy_by_feature.csv`, `redundancy_by_family.csv`, `embedding_source_probe.json`",
]
(OUT / "phase1_report.md").write_text("\n".join(lines) + "\n")

log(f"\n[done] wrote {len(list(OUT.iterdir()))} files to {OUT}")
