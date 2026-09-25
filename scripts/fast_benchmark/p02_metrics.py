"""Part 2 / Part E-F - one shared evaluation over the six harmonized comparators.

Loads:
  results/fast_benchmark/01_legacy_oof_fold{0,1,2}.csv   (previous-lab B5, 3-class)
  results/fast_benchmark/03_ropdeepx_oof_fold{0,1,2}.csv (ROPDeepX harmonized)
  results/fast_benchmark/02b_current_models_oof.csv      (current B / C / E / G, Prompt 1)

Verifies that image ids, patient ids, fold ids and labels are identical, then writes
04_common_metrics.csv, 04b_confusion_matrices.json and 04c_intersection.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)

WS = Path("/root/niki_rop_task6_isolated")
OUT = WS / "results/fast_benchmark"
METHODS = [("Previous Lab B5 (3-class)", "Legacy"), ("ROPDeepX 2026 (harmonized)", "ROPDeepX"),
           ("Current RGB embedding (B)", "B"), ("Current RGB + biomarkers (C)", "C"),
           ("Current RGB + vessel (E)", "E"), ("Current RGB + vessel + biomarkers (G)", "G")]
NAMES = ["Normal", "Pre_Plus", "Plus"]


def log(m):
    print(m, flush=True)


def load_legacy():
    parts = [pd.read_csv(OUT / f"01_legacy_oof_fold{k}.csv") for k in range(3)]
    return pd.concat(parts, ignore_index=True)


def load_ropdeepx():
    parts = [pd.read_csv(OUT / f"03_ropdeepx_oof_fold{k}.csv") for k in range(3)]
    return pd.concat(parts, ignore_index=True)


def ece15(y, P, bins=15):
    pred, conf = P.argmax(1), P.max(1)
    acc = (pred == y).astype(float)
    e, n = 0.0, len(y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf <= hi) if i == bins - 1 else (conf >= lo) & (conf < hi)
        if m.sum():
            e += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(e)


legacy, ropdeepx = load_legacy(), load_ropdeepx()
legacy["fold"] = legacy.fold.astype(int)
ropdeepx["fold"] = ropdeepx.fold.astype(int)
# required merged OOF tables
legacy[["image_id", "patient_id", "fold", "true_label"] + [f"Legacy_p{c}" for c in range(3)]] \
    .to_csv(OUT / "01_legacy_oof.csv", index=False)
ropdeepx[["image_id", "patient_id", "fold", "true_label"] + [f"ROPDeepX_p{c}" for c in range(3)]] \
    .to_csv(OUT / "03_ropdeepx_oof.csv", index=False)
cur = pd.read_csv(OUT / "02b_current_models_oof.csv")

# ---- identity gates -------------------------------------------------------
base = cur[["image_id", "patient_id", "fold", "true_label"]].copy()
gates = {"legacy_rows": len(legacy) == 1528, "ropdeepx_rows": len(ropdeepx) == 1528,
         "current_rows": len(cur) == 1528,
         "legacy_ids_match": set(legacy.image_id) == set(base.image_id),
         "ropdeepx_ids_match": set(ropdeepx.image_id) == set(base.image_id)}
gl = legacy.set_index("image_id").reindex(base.image_id)
gr = ropdeepx.set_index("image_id").reindex(base.image_id)
gates["legacy_fold_match"] = bool(gl.fold.eq(base.fold.values).all())
gates["legacy_patient_match"] = bool(gl.patient_id.eq(base.patient_id.values).all())
gates["legacy_label_match"] = bool(gl.true_label.eq(base.true_label.values).all())
gates["ropdeepx_fold_match"] = bool(gr.fold.eq(base.fold.values).all())
gates["ropdeepx_patient_match"] = bool(gr.patient_id.eq(base.patient_id.values).all())
gates["ropdeepx_label_match"] = bool(gr.true_label.eq(base.true_label.values).all())
for k, v in gates.items():
    log(f"GATE {k:26s} {'PASS' if v else 'FAIL'}")
if not all(gates.values()):
    sys.exit("IDENTITY_GATE_FAILED")
(OUT / "04c_intersection.json").write_text(json.dumps(
    {"n_common_images": 1528, "n_common_patients": int(base.patient_id.nunique()),
     "folds": base.fold.value_counts().sort_index().to_dict(),
     "classes": base.true_label.value_counts().sort_index().to_dict(),
     "identity_gates": gates, "image_level_exclusions": 0}, indent=1))

y = base.true_label.to_numpy(int)
P = {"Legacy": gl[[f"Legacy_p{c}" for c in range(3)]].to_numpy(float),
     "ROPDeepX": gr[[f"ROPDeepX_p{c}" for c in range(3)]].to_numpy(float)}
for m in ("B", "C", "E", "G"):
    P[m] = base.join(cur.set_index("image_id")[[f"{m}_p{c}" for c in range(3)]],
                     on="image_id")[[f"{m}_p{c}" for c in range(3)]].to_numpy(float)
    assert P[m].shape == (1528, 3)

rows, confs = [], {}
for label, key in METHODS:
    p = P[key]
    pred = p.argmax(1)
    per_class = roc_auc_score(y, p, multi_class="ovr", average=None).tolist()
    rows.append({"method": label, "key": key, "n": len(y),
                 "auc_macro_ovr": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
                 "auc_normal": per_class[0], "auc_pre_plus": per_class[1], "auc_plus": per_class[2],
                 "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                 "macro_f1": float(f1_score(y, pred, average="macro")),
                 "accuracy": float(accuracy_score(y, pred)),
                 "brier": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
                 "ece15": ece15(y, p)})
    confs[key] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
met = pd.DataFrame(rows)
met.to_csv(OUT / "04_common_metrics.csv", index=False)
(OUT / "04b_confusion_matrices.json").write_text(json.dumps(
    {"labels": NAMES, "order": "true rows x predicted columns", "matrices": confs}, indent=1))
pd.set_option("display.width", 200)
print(met[["method", "auc_macro_ovr", "balanced_accuracy", "macro_f1", "accuracy", "brier",
           "ece15"]].to_string(index=False))
np.savez_compressed(OUT / "04d_probabilities.npz", y=y, patient_id=base.patient_id.values,
                    fold=base.fold.values, **{f"P_{k}": v for k, v in P.items()})
log("wrote 04_common_metrics.csv, 04b_confusion_matrices.json, 04c_intersection.json, 04d_probabilities.npz")
