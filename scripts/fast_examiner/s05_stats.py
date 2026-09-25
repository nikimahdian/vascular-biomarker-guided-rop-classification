"""Step 5 - pool the 3-fold OOF predictions, metrics, patient-level bootstrap,
biomarker redundancy audit and the two feature-selection sensitivities.

Outputs (all under results/fast_examiner/):
  03_current_models_oof.csv
  04_biomarker_redundancy.csv
  05_feature_sensitivity.csv
  06_model_metrics.csv
  07_paired_bootstrap.csv
  10_fast_decision.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score)

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "results/fast_examiner"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
MODELS = ["B", "C", "E", "G"]
XGB_PARAMS = {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0}
WEIGHTS = {0: 0.46051, 1: 3.18103, 2: 1.94512}
SEED = 42
NREP_MAX = 10000
NREP_MIN = 5000
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def auc_macro(y, P):
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro"))


def auc_per_class(y, P):
    return roc_auc_score(y, P, multi_class="ovr", average=None).tolist()


def ece15(y, P, bins=15):
    pred = P.argmax(1)
    conf = P.max(1)
    acc = (pred == y).astype(float)
    e, n = 0.0, len(y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi) if i < bins - 1 else (conf >= lo) & (conf <= hi)
        if m.sum():
            e += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(y, P):
    return float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1)))


def all_metrics(y, P):
    pred = P.argmax(1)
    return {"multiclass_auc": auc_macro(y, P),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "brier": brier(y, P), "ece": ece15(y, P)}


# ---------------------------------------------------------------- pool OOF
PARTIAL = bool(os.environ.get("FAST_PARTIAL"))
folds_present = [k for k in range(3) if (OUT / f"fold{k}_test_preds.csv").exists()]
if not PARTIAL and len(folds_present) != 3:
    sys.exit(f"only folds {folds_present} present - run all three outer folds first")
preds = [pd.read_csv(OUT / f"fold{k}_test_preds.csv") for k in folds_present]
oof = pd.concat(preds, ignore_index=True)
man = pd.read_csv(OUT / "02_common_farFUM_3fold_manifest.csv")
oof = oof.merge(man[["image_id", "fold"]].rename(columns={"fold": "fold_manifest"}),
                on="image_id", how="left")
assert oof.fold_manifest.eq(oof.fold).all(), "fold mismatch between OOF and manifest"
assert set(oof.true_label.unique()) == {0, 1, 2}
if not PARTIAL:
    assert len(oof) == 1528, f"OOF rows {len(oof)} != 1528"
    assert oof.image_id.nunique() == 1528, "duplicate image_id in OOF"
oof = oof.drop(columns=["fold_manifest"])
cols = (["image_id", "patient_id", "fold", "true_label"]
        + [f"{m}_p{c}" for m in MODELS for c in range(3)])
oof = oof[cols]
oof.to_csv(OUT / "03_current_models_oof.csv", index=False)
log(f"OOF pooled {oof.shape} patients {oof.patient_id.nunique()} "
    f"classes {oof.true_label.value_counts().to_dict()}")

P = {m: oof[[f"{m}_p{c}" for c in range(3)]].to_numpy(float) for m in MODELS}
y = oof.true_label.to_numpy(int)
pat = oof.patient_id.to_numpy()

# ---------------------------------------------------------------- metrics
rows = []
for m in MODELS:
    r = {"model": m, "n": len(y), **all_metrics(y, P[m])}
    pc = auc_per_class(y, P[m])
    r.update({"auc_normal": pc[0], "auc_pre_plus": pc[1], "auc_plus": pc[2]})
    rows.append(r)
met = pd.DataFrame(rows)
met.to_csv(OUT / "06_model_metrics.csv", index=False)
print(met.to_string(index=False))

# per-fold AUC
fold_rows = []
for k in folds_present:
    sel = oof.fold.to_numpy() == k
    for m in MODELS:
        fold_rows.append({"fold": k, "model": m, "n": int(sel.sum()),
                          "auc": auc_macro(y[sel], P[m][sel]),
                          "balanced_accuracy": all_metrics(y[sel], P[m][sel])["balanced_accuracy"]})
pd.DataFrame(fold_rows).to_csv(OUT / "06b_per_fold_metrics.csv", index=False)

# ---------------------------------------------------------------- patient bootstrap
# patient-level, stratified by the patient's dominant class so that every
# replicate keeps all three classes (the thesis bootstrap stratified by class in
# the same spirit, but at image level).
prof = oof.groupby("patient_id").true_label.agg(lambda s: int(s.value_counts().idxmax()))
groups = {lab: prof[prof == lab].index.to_numpy() for lab in (0, 1, 2)}
idx_by_pat = {p: np.where(pat == p)[0] for p in prof.index}
rng = np.random.default_rng(SEED)


def draw():
    picks = []
    for lab in (0, 1, 2):
        g = groups[lab]
        picks.append(rng.choice(g, size=len(g), replace=True))
    pids = np.concatenate(picks)
    return np.concatenate([idx_by_pat[p] for p in pids])


COMPARISONS = [("C", "B"), ("E", "B"), ("G", "E")]
METRICS = ["multiclass_auc", "balanced_accuracy", "macro_f1", "brier", "ece"]
obs = {}
for a, b in COMPARISONS:
    ma, mb = all_metrics(y, P[a]), all_metrics(y, P[b])
    for k in METRICS:
        obs[(a, b, k)] = ma[k] - mb[k]

deltas = {key: [] for key in obs}
t_chunk = time.time()
nrep = 0
while nrep < NREP_MAX:
    chunk = 250 if nrep < 1000 else 500
    for _ in range(chunk):
        ii = draw()
        yy = y[ii]
        mets = {m: all_metrics(yy, P[m][ii]) for m in MODELS}
        for a, b in COMPARISONS:
            for k in METRICS:
                deltas[(a, b, k)].append(mets[a][k] - mets[b][k])
    nrep += chunk
    rate = (time.time() - t_chunk) / nrep
    if nrep == NREP_MIN or (nrep % 2000 == 0):
        log(f"bootstrap {nrep} reps, {rate * 1000:.2f} ms/rep, "
            f"projected total for 10000: {rate * 10000 / 60:.1f} min")
    if nrep >= NREP_MIN and rate * NREP_MAX > 25 * 60:
        log(f"stopping at {nrep} replicates (runtime guard, 25 min cap)")
        break

rows = []
for (a, b, k), d in deltas.items():
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = float(np.mean(np.abs(d - obs[(a, b, k)]) >= abs(obs[(a, b, k)])))
    rows.append({"comparison": f"{a}-{b}", "metric": k, "delta": obs[(a, b, k)],
                 "ci_lo": lo, "ci_hi": hi, "p_value": p, "n_replicates": len(d),
                 "crosses_zero": bool(lo <= 0 <= hi)})
boot = pd.DataFrame(rows)
boot.to_csv(OUT / "07_paired_bootstrap.csv", index=False)
log("bootstrap done")
print(boot[boot.metric == "multiclass_auc"].to_string(index=False))

# ---------------------------------------------------------------- redundancy
red_rows = []
for k in folds_present + ["all"]:
    d = man if k == "all" else man[man.fold != k]
    X = d[FEATS].to_numpy(float)
    pear = np.corrcoef(X, rowvar=False)
    spear = pd.DataFrame(X).corr(method="spearman").to_numpy()
    # VIF
    vif = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(others)), others])
        beta, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        resid = X[:, j] - A @ beta
        r2 = 1 - (resid ** 2).sum() / ((X[:, j] - X[:, j].mean()) ** 2).sum()
        vif.append(float(1 / max(1e-9, 1 - r2)))
    for i, f in enumerate(FEATS):
        red_rows.append({"scope": f"fold{k}_development" if k != "all" else "farfum_all",
                         "feature": f, "n": len(d), "variance": float(X[:, i].var(ddof=1)),
                         "vif": vif[i],
                         "max_abs_pearson_with_other": float(np.max(np.delete(np.abs(pear[i]), i))),
                         "max_abs_spearman_with_other": float(np.max(np.delete(np.abs(spear[i]), i))),
                         "pearson_with_fractal_d0": float(pear[i, 2]),
                         "pearson_with_fractal_d1": float(pear[i, 3]),
                         "pearson_with_fractal_d2": float(pear[i, 4])})
    if k == 0:
        np.save(OUT / "redundancy_pearson_fold0dev.npy", pear)
        np.save(OUT / "redundancy_spearman_fold0dev.npy", spear)
red = pd.DataFrame(red_rows)
red.to_csv(OUT / "04_biomarker_redundancy.csv", index=False)
print(red[red.scope == "farfum_all"].to_string(index=False))
pair_hi = []
for i in range(5):
    for j in range(i + 1, 5):
        pair_hi.append((FEATS[i], FEATS[j], float(pear[i, j])))
print("largest |r| pairs:", sorted(pair_hi, key=lambda t: -abs(t[2]))[:4])

# ---------------------------------------------------------------- feature sensitivity
from sklearn.decomposition import PCA  # noqa: E402
import xgboost as xgb  # noqa: E402

fs_rows = []
fs_preds = {"FS0": {m: np.zeros((len(oof), 3)) for m in MODELS},
            "FS1": {m: np.zeros((len(oof), 3)) for m in MODELS},
            "FS2": {m: np.zeros((len(oof), 3)) for m in MODELS}}
kept = {}
for k in folds_present:
    cache = OUT / f"05c_fs_preds_fold{k}.npz"
    if cache.exists():
        d = np.load(cache)
        kept_file = OUT / f"05c_fs_detail_fold{k}.json"
        if kept_file.exists():
            for kk, vv in json.loads(kept_file.read_text()).items():
                kept.setdefault(kk, {})[f"fold{k}"] = vv
        for variant in ("FS0", "FS1", "FS2"):
            for name in ("C", "G"):
                fs_preds[variant][name] = d[f"{variant}_{name}"]
        log(f"fold{k} feature sensitivity loaded from cache")
        continue
    z = np.load(OUT / f"fold{k}_embeddings.npz", allow_pickle=True)
    ids = pd.Index([Path(p).stem for p in z["image_id"]])
    perm = np.array([ids.get_loc(s) for s in man.image_id])          # manifest -> embedding order
    rgb, ves = z["rgb"][perm], z["vessel"][perm]
    bio = man[FEATS].to_numpy(float)
    y_man = man.label.to_numpy(int)
    tr = (man.fold.to_numpy() != k)
    te = ~tr
    # oof rows of this fold are, in order, the manifest rows of this fold
    oof_pos = np.where(oof.fold.to_numpy() == k)[0]
    assert len(oof_pos) == int(te.sum()), (len(oof_pos), int(te.sum()))
    fold_store = {}
    for variant in ("FS0", "FS1", "FS2"):
        if variant == "FS0":
            Ftr, Fte = bio[tr], bio[te]
        elif variant == "FS1":
            r = np.abs(np.corrcoef(bio[tr], rowvar=False))
            drop = set()
            for i in range(5):
                for j in range(i + 1, 5):
                    if r[i, j] >= 0.90 and i not in drop and j not in drop:
                        drop.add(j)
            keep = [i for i in range(5) if i not in drop]
            kept.setdefault("FS1", {})[f"fold{k}"] = [FEATS[i] for i in keep]
            Ftr, Fte = bio[tr][:, keep], bio[te][:, keep]
        else:
            pca = PCA(n_components=0.95, random_state=SEED).fit(bio[tr])
            Ftr, Fte = pca.transform(bio[tr]), pca.transform(bio[te])
            kept.setdefault("FS2", {})[f"fold{k}"] = int(pca.n_components_)
        for name, base in (("C", rgb), ("G", np.hstack([rgb, ves]))):
            Xtr = np.hstack([base[tr], Ftr])
            Xte = np.hstack([base[te], Fte])
            ytr = y_man[tr]
            w = np.array([WEIGHTS[int(v)] for v in ytr], dtype=np.float32)
            clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                                    random_state=SEED, n_jobs=8, tree_method="hist", **XGB_PARAMS)
            clf.fit(Xtr, ytr, sample_weight=w)
            fs_preds[variant][name][oof_pos] = clf.predict_proba(Xte)
            fold_store[f"{variant}_{name}"] = fs_preds[variant][name]
        log(f"fold{k} {variant} done (kept/PC detail: {kept.get(variant, {}).get(f'fold{k}')})")
    np.savez_compressed(cache, **fold_store)
    (OUT / f"05c_fs_detail_fold{k}.json").write_text(json.dumps(
        {v: kept.get(v, {}).get(f"fold{k}") for v in ("FS1", "FS2")}))
    log(f"fold{k} feature sensitivity done")

for variant in ("FS0", "FS1", "FS2"):
    # B and E carry no scalar biomarker, so they are identical across variants;
    # their reference is the pooled fold-safe OOF prediction from s04.
    auc = {"B": auc_macro(y, P["B"]), "E": auc_macro(y, P["E"]),
           "C": auc_macro(y, fs_preds[variant]["C"]),
           "G": auc_macro(y, fs_preds[variant]["G"])}
    row = {"variant": variant, "n_features_bio": (5 if variant == "FS0" else
                                                 (len(kept.get("FS1", {}).get("fold0", [])) if variant == "FS1"
                                                  else kept.get("FS2", {}).get("fold0", 0))),
           "auc_B": auc["B"], "auc_C": auc["C"], "auc_E": auc["E"], "auc_G": auc["G"],
           "C_minus_B": auc["C"] - auc["B"], "G_minus_E": auc["G"] - auc["E"],
           "detail": json.dumps({k2: v for k2, v in kept.get(variant, {}).items()})}
    fs_rows.append(row)
fs = pd.DataFrame(fs_rows)
fs.to_csv(OUT / "05_feature_sensitivity.csv", index=False)
print(fs.to_string(index=False))
np.savez_compressed(OUT / "05b_feature_sensitivity_preds.npz",
                    **{f"{v}_{m}": fs_preds[v][m] for v in fs_preds for m in ("C", "G")})

# ---------------------------------------------------------------- machine summary
decision = {
    "cohort": {"name": "FARFUM-RoP", "n_images": int(len(oof)),
               "n_patients": int(oof.patient_id.nunique()), "k": 3,
               "class_counts": {str(k2): int(v) for k2, v in oof.true_label.value_counts().items()},
               "per_fold_images": {str(k2): int(v) for k2, v in oof.fold.value_counts().items()}},
    "oof_auc": {m: auc_macro(y, P[m]) for m in MODELS},
    "oof_metrics": met.set_index("model").to_dict(orient="index"),
    "bootstrap": {"n_replicates": int(boot.n_replicates.min()), "seed": SEED,
                  "unit": "patient", "stratified_by": "patient dominant class"},
    "primary_deltas": {f"{a}-{b}": {k: float(obs[(a, b, k)]) for k in METRICS}
                       for a, b in COMPARISONS},
    "feature_sensitivity": fs.to_dict(orient="records"),
    "redundancy_all": red[red.scope == "farfum_all"].to_dict(orient="records"),
}
(OUT / "10_fast_decision.json").write_text(json.dumps(decision, indent=1, default=float))
log("s05 complete")
