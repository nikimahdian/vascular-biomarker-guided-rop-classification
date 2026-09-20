#!/usr/bin/env python
"""Direct source-probe comparison: clinical_v2 (constant DD) vs clinical_v2c (measured DD).

The project's own harness reported raw23 macro-OvR source AUC = 0.9914 for v2. This script
reproduces the quantity under my own control, so the A/B is transparent, and adds the comparison
that the harness cannot do: restricted to the rows where the disc was actually found.

That restriction matters because the missingness pattern is itself source-informative
(disc found on 48.9% of farabi, 53.3% of farfum_rop, 38.2% of plus). A gradient-boosted model can
read NaN as a signal, so a table that is 57.5% missing in a source-dependent pattern can look MORE
source-separable, not less. The within-subset comparison removes that artefact.

HistGradientBoostingClassifier is used because it handles NaN natively, like LightGBM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = "/Users/moniaz/niki"
META = {"image_path", "mask_path", "label", "split", "source", "group_id", "patient_id",
        "exam_id", "identity_level", "disc_peak_prob"}
SEED = 42


def load(name):
    return pd.read_csv(f"{ROOT}/data/features/{name}")


def numeric_cols(df):
    cols = []
    for c in df.columns:
        if c in META:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1:
            cols.append(c)
    return cols


def source_probe(df, cols, label):
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["source"].to_numpy()
    groups = df["group_id"].to_numpy()
    classes = sorted(set(y))
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(df), len(classes)))
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                           random_state=SEED, early_stopping=False)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        for j, c in enumerate(m.classes_):
            oof[te, classes.index(c)] = p[:, j]
    aucs = []
    for j, c in enumerate(classes):
        yb = (y == c).astype(int)
        if 0 < yb.sum() < len(yb):
            aucs.append(roc_auc_score(yb, oof[:, j]))
    macro = float(np.mean(aucs))
    print(f"  {label:34s} n={len(df):5d} feats={len(cols):3d}  "
          f"macro-OvR AUC = {macro:.4f}   per-class " +
          " ".join(f"{c}:{a:.3f}" for c, a in zip(classes, aucs)))
    return macro


v2 = load("biomarker_features_clinical_v2.csv")
v2c = load("biomarker_features_clinical_v2c.csv")
ddok = pd.read_csv(f"{ROOT}/results/hvdro_validation/disc/disc_predictions_all.csv")
ddok = ddok[(ddok.peak_prob > 0.9) & ddok.dd_over_min_side.between(0.03, 0.25)]
ok_paths = set(ddok.image_path)
v2["_ok"] = v2["image_path"].isin(ok_paths)
v2c["_ok"] = v2c["image_path"].isin(ok_paths)
print(f"disc-found rows: {int(v2['_ok'].sum())} / {len(v2)}")

c2 = numeric_cols(v2.drop(columns=["_ok"]))
cc = numeric_cols(v2c.drop(columns=["_ok"]))
print(f"v2 numeric features: {len(c2)}   v2c numeric features: {len(cc)}")
print()

print("=== A) all rows ===")
a1 = source_probe(v2, c2, "v2  constant DD")
a2 = source_probe(v2c, cc, "v2c measured DD")

print("\n=== B) same subset: only rows where the disc was found ===")
s2 = v2[v2["_ok"]].reset_index(drop=True)
sc = v2c[v2c["_ok"]].reset_index(drop=True)
b1 = source_probe(s2, numeric_cols(s2), "v2  constant DD (subset)")
b2 = source_probe(sc, numeric_cols(sc), "v2c measured DD (subset)")

print("\n=== C) same subset, width columns REMOVED from both (what is left?) ===")
no_w = [c for c in c2 if "width" not in c and c != "dd_over_min_side"]
source_probe(s2, no_w, "v2  no width cols (subset)")
no_wc = [c for c in cc if "width" not in c and c != "dd_over_min_side" and c != "disc_method_pvbm"]
source_probe(sc, no_wc, "v2c no width cols (subset)")

print("\n=== summary ===")
print(f"  all rows      : v2 {a1:.4f} -> v2c {a2:.4f}   delta {a2 - a1:+.4f}")
print(f"  disc-found set: v2 {b1:.4f} -> v2c {b2:.4f}   delta {b2 - b1:+.4f}")
print(f"\n  project harness reported v2 raw23 macro-OvR = 0.9914")
print(f"  decision threshold source_probe_severe_auc = 0.80")
