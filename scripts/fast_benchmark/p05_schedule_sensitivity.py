"""Transparency check: metrics of the first harmonized ROPDeepX run (OneCycleLR schedule).

The primary ROPDeepX row in 04_common_metrics.csv is the plateau-schedule run. This script scores
the preserved OneCycle variant on the identical held-out rows so the schedule interaction is
documented rather than hidden.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

OUT = Path("/root/niki_rop_task6_isolated/results/fast_benchmark")
cur = pd.read_csv(OUT / "02b_current_models_oof.csv")[["image_id", "true_label"]]
rows = []
for tag in ("onecycle", "plateau"):
    files = (sorted(OUT.glob("03_ropdeepx_onecycle_oof_fold*.csv")) if tag == "onecycle"
             else sorted(OUT.glob("03_ropdeepx_oof_fold*.csv")))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).set_index("image_id")
    d = d.reindex(cur.image_id)
    y = cur.true_label.to_numpy(int)
    P = d[[f"ROPDeepX_p{c}" for c in range(3)]].to_numpy(float)
    pred = P.argmax(1)
    rows.append({"variant": tag, "schedule": "OneCycleLR(20 ep)" if tag == "onecycle"
                 else "ReduceLROnPlateau(constant, <=30 ep)",
                 "n": len(y),
                 "auc_macro_ovr": float(roc_auc_score(y, P, multi_class="ovr", average="macro")),
                 "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                 "macro_f1": float(f1_score(y, pred, average="macro")),
                 "accuracy": float((pred == y).mean()),
                 "brier": float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1))),
                 "selected_epochs": [json.loads(
                     (OUT / f"03_ropdeepx{'_onecycle' if tag == 'onecycle' else ''}"
                      f"_selection_fold{k}.json").read_text())["selected"]["epoch"]
                     for k in range(3)] if tag == "onecycle" else None})
df = pd.DataFrame(rows)
df.to_csv(OUT / "04e_ropdeepx_schedule_sensitivity.csv", index=False)
print(df.to_string(index=False))
