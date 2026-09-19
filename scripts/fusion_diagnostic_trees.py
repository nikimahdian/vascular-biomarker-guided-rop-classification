#!/usr/bin/env python
"""Phase 2c — do TREE read-outs change the complementarity verdict?

Phase 2b used linear read-outs (logistic). The obvious objection is: "you only tested linear
models". This run repeats the same clean protocol (fit on train, evaluate on validation, locked
test never loaded, same pre-declared gate) with gradient-boosted tree read-outs, and compares
each tree arm against the matching linear arm.

Arms
    linear:  a_emb_only        , c_emb_plus_lowred
    trees :  t_emb_only        , t_emb_plus_all23  , t_emb_plus_lowred  , t_lowred_only
Backends: lightgbm, xgboost, sklearn HistGradientBoosting (whichever import).

PRE-DECLARED GATE (unchanged): dAUC >= 0.005 AND paired 95% CI lower bound > 0
AND not driven by one source AND no test used.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/moniaz/niki")
os.chdir(ROOT)
OUT = ROOT / "results" / "fusion_diagnostic_v1" / "phase2c_trees"
OUT.mkdir(parents=True, exist_ok=True)

SEED, PLUS, N_BOOT, GATE_DELTA = 42, 2, 2000, 0.005
META = {"image_path", "mask_path", "label", "split", "source",
        "group_id", "patient_id", "exam_id", "identity_level"}
LOW_REDUNDANCY = [
    "median_tortuosity", "singularity_length", "median_branching_angle", "n_startpoints",
    "tortuosity_index", "density_r0c0", "density_r1c2", "density_r2c2", "density_r1c0",
    "density_r2c0", "density_r0c2", "density_r0c1",
]


def log(m):
    print(m, flush=True)


# ---- backends
BACKENDS = {}
try:
    from lightgbm import LGBMClassifier

    BACKENDS["lightgbm"] = lambda: LGBMClassifier(
        n_estimators=500, learning_rate=0.03, random_state=SEED, n_jobs=2, verbose=-1)
except Exception as e:  # noqa: BLE001
    log(f"[warn] lightgbm unavailable: {type(e).__name__}")
try:
    from xgboost import XGBClassifier

    BACKENDS["xgboost"] = lambda: XGBClassifier(
        n_estimators=500, learning_rate=0.03, eval_metric="logloss",
        random_state=SEED, n_jobs=2, tree_method="hist")
except Exception as e:  # noqa: BLE001
    log(f"[warn] xgboost unavailable: {type(e).__name__}")
from sklearn.ensemble import HistGradientBoostingClassifier

BACKENDS["sklearn_hgb"] = lambda: HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.06, random_state=SEED)

log(f"tree backends available: {list(BACKENDS)}")

# ---- data, in the pinned order
raw = pd.read_csv(ROOT / "data/features/biomarker_features.csv")
f = raw.copy()
f["_o"] = f["split"].map({"train": 0, "val": 1, "test": 2})
f = f.sort_values(["_o", "image_path"]).drop(columns="_o").reset_index(drop=True)
order_sha = hashlib.sha256("\n".join(f["image_path"]).encode()).hexdigest()
if order_sha != json.loads((ROOT / "results/hybrid_v2/embeddings_current.json").read_text())["image_order_sha256"]:
    raise SystemExit("order hash mismatch")
emb = np.load(ROOT / "results/hybrid_v2/embeddings_current.npz")["embeddings"].astype(np.float64)

is_tr = (f["split"] == "train").to_numpy()
is_va = (f["split"] == "val").to_numpy()
feat_cols = [c for c in f.columns if c not in META]
med = f.loc[is_tr, feat_cols].median().fillna(0.0)


def blk(cols):
    M = f[cols].fillna(med[cols]).to_numpy(float)
    return M[is_tr], M[is_va]


btr_all, bva_all = blk(feat_cols)
btr_low, bva_low = blk(LOW_REDUNDANCY)
ytr = (f.loc[is_tr, "label"].to_numpy(int) == PLUS).astype(int)
yva = (f.loc[is_va, "label"].to_numpy(int) == PLUS).astype(int)
sva = f.loc[is_va, "source"].to_numpy()
gva = f.loc[is_va, "group_id"].to_numpy()
Xtr, Xva = emb[is_tr], emb[is_va]
log(f"fit rows={is_tr.sum()} eval rows={is_va.sum()} val Plus={yva.sum()}  | locked test NOT loaded")


def boot_delta(pa, pb, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(gva)
    idx = {g: np.flatnonzero(gva == g) for g in uniq}
    d = []
    for _ in range(n):
        rows = np.concatenate([idx[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(yva[rows])) < 2:
            continue
        d.append(roc_auc_score(yva[rows], pa[rows]) - roc_auc_score(yva[rows], pb[rows]))
    d = np.asarray(d)
    return {"mean_delta": float(d.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(2 * min((d > 0).mean(), (d < 0).mean()))}


results, preds = [], {}

# linear reference arms
for name, A, B in (("lin_emb_only", Xtr, Xva), ("lin_emb_plus_lowred",
                                                np.hstack([Xtr, btr_low]), np.hstack([Xva, bva_low]))):
    sc = StandardScaler().fit(A)
    clf = LogisticRegression(C=0.001, max_iter=3000).fit(sc.transform(A), ytr)
    p = clf.predict_proba(sc.transform(B))[:, 1]
    preds[name] = p
    per_src = {s: float(roc_auc_score(yva[sva == s], p[sva == s])) for s in np.unique(sva)
               if len(np.unique(yva[sva == s])) > 1}
    results.append({"arm": name, "backend": "logistic",
                    "val_auc": float(roc_auc_score(yva, p)),
                    **{f"auc_{k}": v for k, v in per_src.items()}})
    log(f"  {name:24s} VAL AUC={results[-1]['val_auc']:.6f}")

# tree arms
for be, make in BACKENDS.items():
    for name, A, B in (("emb_only", Xtr, Xva),
                       ("emb_plus_all23", np.hstack([Xtr, btr_all]), np.hstack([Xva, bva_all])),
                       ("emb_plus_lowred", np.hstack([Xtr, btr_low]), np.hstack([Xva, bva_low])),
                       ("lowred_only", btr_low, bva_low)):
        key = f"{be}__{name}"
        t0 = time.time()
        clf = make()
        clf.fit(A, ytr)
        p = clf.predict_proba(B)[:, 1]
        preds[key] = p
        auc = float(roc_auc_score(yva, p))
        per_src = {s: float(roc_auc_score(yva[sva == s], p[sva == s])) for s in np.unique(sva)
                   if len(np.unique(yva[sva == s])) > 1}
        results.append({"arm": key, "backend": be, "val_auc": auc,
                        "seconds": round(time.time() - t0, 1),
                        **{f"auc_{k}": v for k, v in per_src.items()}})
        log(f"  {key:24s} VAL AUC={auc:.6f}  {({k: round(v,4) for k,v in per_src.items()})}  ({results[-1]['seconds']}s)")

# comparisons
comparisons = {}
for key in preds:
    if key.startswith("lin_"):
        continue
    comparisons[f"{key}_minus_lin_emb_only"] = boot_delta(preds[key], preds["lin_emb_only"])
lin_low = comparisons.get("lin_emb_plus_lowred_minus_lin_emb_only")
if lin_low is None:
    comparisons["lin_emb_plus_lowred_minus_lin_emb_only"] = boot_delta(
        preds["lin_emb_plus_lowred"], preds["lin_emb_only"])

log("\n=== paired group-cluster bootstrap vs the linear embedding-only arm ===")
for k, v in comparisons.items():
    log(f"  {k:44s} dAUC={v['mean_delta']:+.6f} CI95=[{v['ci95'][0]:+.6f},{v['ci95'][1]:+.6f}] p={v['p_two_sided']:.4f}")

# gate over every arm that adds features to the embedding
base = float(roc_auc_score(yva, preds["lin_emb_only"]))
cands = [r for r in results if r["arm"].endswith("emb_plus_all23") or r["arm"].endswith("emb_plus_lowred")]
best = max(cands, key=lambda r: r["val_auc"])
r = comparisons[f"{best['arm']}_minus_lin_emb_only"]
per_src_best = {s: best.get(f"auc_{s}") for s in np.unique(sva)}
per_src_base = {s: float(roc_auc_score(yva[sva == s], preds["lin_emb_only"][sva == s])) for s in np.unique(sva)}
deltas = [per_src_best[s] - per_src_base[s] for s in per_src_best
          if per_src_best[s] is not None and np.isfinite(per_src_best[s])]
if not deltas:
    raise SystemExit(f"no per-source deltas available for {best['arm']}")
worst_delta = float(min(deltas))
gate = {"protocol": "fit train -> evaluate val (backbone-unseen rows), locked test not loaded",
        "baseline": "lin_emb_only", "best_feature_arm": best["arm"],
        "auc_baseline": base, "auc_best": best["val_auc"],
        "delta_auc": r["mean_delta"], "ci95": r["ci95"], "p_two_sided": r["p_two_sided"],
        "worst_source_delta": worst_delta, "threshold_delta_auc": GATE_DELTA,
        "criteria": {"delta_ge_threshold": bool(r["mean_delta"] >= GATE_DELTA),
                     "ci_lower_gt_zero": bool(r["ci95"][0] > 0),
                     "no_source_driven_gain": bool(worst_delta > -0.005),
                     "no_test_used": True}}
gate["pass"] = all(gate["criteria"].values())
log("\n=== GATE (tree backends included) ===\n" + json.dumps(gate, indent=2))

json.dump({"results": results, "comparisons": comparisons, "gate": gate,
           "backends": list(BACKENDS), "test_set_used": False},
          open(OUT / "decision.json", "w"), indent=2)
pd.DataFrame(results).to_csv(OUT / "val_table.csv", index=False)
v = f.loc[is_va, ["image_path", "label", "source", "group_id"]].copy()
for k, p in preds.items():
    v[f"p_{k}"] = p
v.to_csv(OUT / "val_predictions.csv", index=False)
log(f"[done] -> {OUT}")
