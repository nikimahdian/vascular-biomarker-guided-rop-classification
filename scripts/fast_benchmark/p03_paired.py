"""Part 2 / Part G - three primary paired comparisons on identical patients.

  E - ROPDeepX      (current vessel-fusion method vs harmonized recent publication)
  E - Previous B5   (current vessel-fusion method vs previous-lab architecture, 3-class)
  E - B             (does the vessel representation add value inside this benchmark)

Patient-level paired bootstrap, 10,000 replicates, seed 42, stratified by the patient's
dominant class so every replicate keeps all three classes. Metrics: Delta macro OvR AUC,
Delta macro F1, Delta Brier. No unpaired testing.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

WS = Path("/root/niki_rop_task6_isolated")
OUT = WS / "results/fast_benchmark"
SEED = 42
NREP = int(os.environ.get("NREP", "10000"))
T0 = time.time()
Z = np.load(OUT / "04d_probabilities.npz", allow_pickle=True)
y = Z["y"]
pat = Z["patient_id"]
P = {k[2:]: Z[k] for k in Z.files if k.startswith("P_")}
COMPARISONS = [("E", "ROPDeepX"), ("E", "Legacy"), ("E", "B")]
METRICS = ["auc", "macro_f1", "brier"]


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def metric(kind, yy, p):
    if kind == "auc":
        return float(roc_auc_score(yy, p, multi_class="ovr", average="macro"))
    if kind == "macro_f1":
        return float(f1_score(yy, p.argmax(1), average="macro"))
    return float(np.mean(np.sum((p - np.eye(3)[yy]) ** 2, axis=1)))


# patient-level, class-stratified resampling (same construction as Prompt 1)
prof = pd.Series(y).groupby(pat).agg(lambda s: int(pd.Series(s).value_counts().idxmax()))
groups = {lab: prof[prof == lab].index.to_numpy() for lab in (0, 1, 2)}
idx_by_pat = {p: np.where(pat == p)[0] for p in prof.index}
rng = np.random.default_rng(SEED)


def draw():
    return np.concatenate([np.concatenate([idx_by_pat[p] for p in
                                           rng.choice(groups[lab], size=len(groups[lab]),
                                                      replace=True)])
                           for lab in (0, 1, 2)])


obs = {}
for a, b in COMPARISONS:
    for k in METRICS:
        obs[(a, b, k)] = metric(k, y, P[a]) - metric(k, y, P[b])
    log(f"{a}-{b} observed " + " ".join(f"d{k} {obs[(a, b, k)]:+.6f}" for k in METRICS))

deltas = {key: [] for key in obs}
for rep in range(1, NREP + 1):
    ii = draw()
    yy = y[ii]
    for a, b in COMPARISONS:
        ma = {k: metric(k, yy, P[a][ii]) for k in METRICS}
        mb = {k: metric(k, yy, P[b][ii]) for k in METRICS}
        for k in METRICS:
            deltas[(a, b, k)].append(ma[k] - mb[k])
    if rep % 2500 == 0:
        log(f"bootstrap {rep}/{NREP}")

rows = []
for (a, b, k), d in deltas.items():
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    rows.append({"comparison": f"{a}-{b}", "metric": k, "delta": obs[(a, b, k)],
                 "ci_lo": float(lo), "ci_hi": float(hi),
                 "p_value": float(np.mean(np.abs(d - obs[(a, b, k)]) >= abs(obs[(a, b, k)]))),
                 "n_replicates": len(d), "crosses_zero": bool(lo <= 0 <= hi)})
stat = pd.DataFrame(rows)
stat.to_csv(OUT / "05_paired_statistics.csv", index=False)
for _, r in stat.iterrows():
    log(f"{r.comparison:12s} {r.metric:9s} {r.delta:+.6f} [{r.ci_lo:+.6f}, {r.ci_hi:+.6f}] "
        f"p={r.p_value:.4f} {'crosses0' if r.crosses_zero else 'SUPPORTED'}")
json.dump({f"{r.comparison}_{r.metric}": {"delta": r.delta, "ci": [r.ci_lo, r.ci_hi],
                                         "p": r.p_value, "crosses_zero": bool(r.crosses_zero)}
           for _, r in stat.iterrows()},
          open(OUT / "05b_paired_summary.json", "w"), indent=1)
log("wrote 05_paired_statistics.csv, 05b_paired_summary.json")
