#!/usr/bin/env python
"""Phase 2b — corrected read-out experiment.

WHY THIS EXISTS
---------------
Phase 2 evaluated arms with grouped CV *inside* train+val. That is contaminated here: the
frozen embeddings come from Branch B's backbone, which was **fine-tuned on the train rows**.
Its embedding space has seen those images (and their near-duplicates), so any CV over train
rows is not an honest estimate. Evidence from the slice diagnostic:

    train rows only : AUC 0.999424   (plus source 1.000000)
    val   rows only : AUC 0.924882   (plus source 0.918329)

AUC = 1.000 on held-out train groups is the signature of representation-level memorisation,
not of an easy split. The locked test and the val split were never seen by the backbone and are
therefore the only clean evaluation rows available for a frozen-embedding read-out.

CORRECTED PROTOCOL
------------------
Fit the read-out on TRAIN rows; evaluate on VAL rows (1,328 rows, 62 groups, never seen by the
backbone). Paired group-cluster bootstrap over val groups. Locked test never loaded.
Both arms carry the same handicap (read-out fitted on backbone-seen embeddings), so the
comparison between arms is fair even though the absolute level is pessimistic.

PRE-DECLARED GATE (unchanged, from _feedback/MASTER_PLAN.md):
    dAUC >= 0.005 AND paired 95% CI lower bound > 0 AND not driven by one source AND no test used.
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
OUT = ROOT / "results" / "fusion_diagnostic_v1" / "phase2b_readout"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
PLUS = 2
N_BOOT = 2000
C_GRID = (0.001, 0.01, 0.1)
PRIMARY_C = 0.001
GATE_DELTA = 0.005
META = {"image_path", "mask_path", "label", "split", "source",
        "group_id", "patient_id", "exam_id", "identity_level"}
LOW_REDUNDANCY = [
    "median_tortuosity", "singularity_length", "median_branching_angle", "n_startpoints",
    "tortuosity_index", "density_r0c0", "density_r1c2", "density_r2c2", "density_r1c0",
    "density_r2c0", "density_r0c2", "density_r0c1",
]


def log(m):
    print(m, flush=True)


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


def block(cols):
    """Return (train_matrix, val_matrix) with train-median imputation."""
    M = f[cols].fillna(med[cols]).to_numpy(float)
    return M[is_tr], M[is_va]


Xtr_emb, Xva_emb = emb[is_tr], emb[is_va]
gva = f.loc[is_va, "group_id"].to_numpy()
sva = f.loc[is_va, "source"].to_numpy()
ytr = (f.loc[is_tr, "label"].to_numpy(int) == PLUS).astype(int)
yva = (f.loc[is_va, "label"].to_numpy(int) == PLUS).astype(int)
log(f"fit rows (train) = {is_tr.sum()}   eval rows (val) = {is_va.sum()}   val Plus = {yva.sum()}")
log("locked test NOT loaded")
btr_all, bva_all = block(feat_cols)
btr_low, bva_low = block(LOW_REDUNDANCY)

ARMS = {
    "a_emb_only": ("binary", Xtr_emb, Xva_emb),
    "b_emb_plus_all23": ("binary", np.hstack([Xtr_emb, btr_all]), np.hstack([Xva_emb, bva_all])),
    "c_emb_plus_lowred": ("binary", np.hstack([Xtr_emb, btr_low]), np.hstack([Xva_emb, bva_low])),
    "d_emb_3class": ("multiclass", Xtr_emb, Xva_emb),
    "e_lowred_only": ("binary", btr_low, bva_low),
    "f_all23_only": ("binary", btr_all, bva_all),
}

ytr3 = f.loc[is_tr, "label"].to_numpy(int)


def fit_predict(kind, Xtr, Xva, C):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    if kind == "binary":
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr)
        return clf.predict_proba(Xva)[:, 1]
    clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr3)
    pos = list(clf.classes_).index(PLUS)
    return clf.predict_proba(Xva)[:, pos]


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
    return {"n_valid": int(d.size), "mean_delta": float(d.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(2 * min((d > 0).mean(), (d < 0).mean()))}


table, preds = [], {}
for C in C_GRID:
    for name, (kind, Xtr, Xva) in ARMS.items():
        t0 = time.time()
        p = fit_predict(kind, Xtr, Xva, C)
        preds[(name, C)] = p
        auc = float(roc_auc_score(yva, p))
        per_src = {s: float(roc_auc_score(yva[sva == s], p[sva == s])) for s in np.unique(sva)
                   if len(np.unique(yva[sva == s])) > 1}
        log(f"  {name:18s} C={C:<6} val AUC={auc:.6f}  {({k: round(v,4) for k,v in per_src.items()})}  ({time.time()-t0:.1f}s)")
        if C == PRIMARY_C:
            table.append({"arm": name, "val_auc": auc, **{f"auc_{k}": v for k, v in per_src.items()}})

prim = {n: preds[(n, PRIMARY_C)] for n in ARMS}
comps = {}
for n in ("b_emb_plus_all23", "c_emb_plus_lowred", "d_emb_3class", "e_lowred_only", "f_all23_only"):
    comps[f"{n}_minus_a"] = boot_delta(prim[n], prim["a_emb_only"])
    r = comps[f"{n}_minus_a"]
    log(f"  vs a: {n:18s} dAUC={r['mean_delta']:+.6f} CI95=[{r['ci95'][0]:+.6f},{r['ci95'][1]:+.6f}] p={r['p_two_sided']:.4f}")

base = float(roc_auc_score(yva, prim["a_emb_only"]))
feat_arms = ("b_emb_plus_all23", "c_emb_plus_lowred")
best = max(feat_arms, key=lambda n: float(roc_auc_score(yva, prim[n])))
r = comps[f"{best}_minus_a"]
per_src_best = {s: float(roc_auc_score(yva[sva == s], prim[best][sva == s])) for s in np.unique(sva)}
per_src_base = {s: float(roc_auc_score(yva[sva == s], prim["a_emb_only"][sva == s])) for s in np.unique(sva)}
worst = min(per_src_best[s] - per_src_base[s] for s in per_src_best)
gate = {"protocol": "fit on train, evaluate on val (backbone-unseen rows)",
        "primary_C": PRIMARY_C, "baseline_arm": "a_emb_only", "best_feature_arm": best,
        "auc_baseline": base, "auc_best": float(roc_auc_score(yva, prim[best])),
        "delta_auc": r["mean_delta"], "ci95": r["ci95"], "p_two_sided": r["p_two_sided"],
        "worst_source_delta": worst, "threshold_delta_auc": GATE_DELTA,
        "criteria": {"delta_ge_threshold": bool(r["mean_delta"] >= GATE_DELTA),
                     "ci_lower_gt_zero": bool(r["ci95"][0] > 0),
                     "no_source_driven_gain": bool(worst > -0.005), "no_test_used": True}}
gate["pass"] = all(gate["criteria"].values())
log("\n=== GATE (corrected protocol) ===\n" + json.dumps(gate, indent=2))

json.dump({"primary_table": table, "comparisons_vs_a": comps, "gate": gate,
           "low_redundancy_features": LOW_REDUNDANCY, "test_set_used": False,
           "contamination_note": ("Frozen embeddings come from a backbone fine-tuned on the train "
                                  "rows; therefore only val/test rows give clean estimates. "
                                  "Phase 2 (grouped CV inside train+val) reached train-row AUC "
                                  "0.9994 vs val-row AUC 0.9249 and was discarded.")},
          open(OUT / "decision.json", "w"), indent=2)
pd.DataFrame(table).to_csv(OUT / "val_table.csv", index=False)
v = f.loc[is_va, ["image_path", "label", "source", "group_id"]].copy()
for n in ARMS:
    v[f"p_{n}"] = prim[n]
v.to_csv(OUT / "val_predictions.csv", index=False)
log(f"[done] -> {OUT}")
