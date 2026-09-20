#!/usr/bin/env python
"""Self-test for the analysis suite: analytical checks on synthetic data plus a run on real LOSO
predictions. Confirms the statistics behave before any expert data exists."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import _common as C

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        fails.append(name)


print("=== metric identities ===")
rng = np.random.default_rng(0)
x = rng.normal(10, 2, 60)
a = np.zeros((64, 64), bool)
a[10:20, 10:60] = True
b = a.copy()
check("dice identical == 1", abs(C.dice(a, b) - 1) < 1e-12)
check("iou identical == 1", abs(C.iou(a, b) - 1) < 1e-12)
check("cldice identical == 1", abs(C.cldice(a, b) - 1) < 1e-12)
d = a.copy()
d[:, 40:] = False
check("dice drops when half removed", C.dice(d, a) < 1, f"dice={C.dice(d, a):.3f}")
check("cldice finite", np.isfinite(C.cldice(d, a)), f"cldice={C.cldice(d, a):.3f}")

# a thin vessel lost almost entirely: Dice should stay high, tiered recall should not
thick = np.zeros((120, 120), bool)
thick[50:60, 5:115] = True                 # 10 px wide
thin = np.zeros((120, 120), bool)
thin[20:22, 5:115] = True                  # 2 px wide
gt = thick | thin
pred = thick.copy()                        # thin vessel entirely missed
check("Dice stays high when only the thin vessel is lost", C.dice(pred, gt) > 0.8,
      f"dice={C.dice(pred, gt):.3f}")
tr = C.tiered_centerline_recall(pred, gt)
check("thin recall is low", np.nan_to_num(tr.get("recall_thin", np.nan)) < 0.5,
      f"thin={tr.get('recall_thin')}")
check("thick recall is high", np.nan_to_num(tr.get("recall_thick", np.nan)) > 0.9,
      f"thick={tr.get('recall_thick')}")

print("\n=== ICC(2,1) ===")
check("ICC(x, x) == 1", abs(C.icc21(x, x)["icc"] - 1) < 0.05, f"{C.icc21(x, x)['icc']:.4f}")
sysb = x + 5.0
r = C.icc21(x, sysb)
check("systematic +5 offset lowers absolute-agreement ICC", r["icc"] < 1.0, f"{r['icc']:.4f}")
check("a consistency ICC would have hidden it: Pearson is ~1",
      abs(np.corrcoef(x, sysb)[0, 1] - 1) < 1e-9)
perm = rng.permutation(x)
check("ICC of a shuffled column collapses", C.icc21(x, perm)["icc"] < 0.3,
      f"{C.icc21(x, perm)['icc']:.4f}")
small = x + rng.normal(0, 0.3, 60)
check("ICC stays high under small noise", C.icc21(x, small)["icc"] > 0.9,
      f"{C.icc21(x, small)['icc']:.4f}")

print("\n=== ICC bootstrap CI ===")
dfi = pd.DataFrame({"group_id": [f"g{i % 20}" for i in range(60)],
                    "a": x, "b": small})
rc = C.icc21_ci(dfi, "a", "b", n_boot=300)
check("bootstrap CI brackets the point", rc["ci_low"] <= rc["point"] <= rc["ci_high"],
      f"{rc['point']:.3f} [{rc['ci_low']:.3f},{rc['ci_high']:.3f}]")
check("CI width is finite and positive", np.isfinite(rc["ci_low"]) and rc["ci_high"] > rc["ci_low"])

print("\n=== Bland-Altman ===")
ba = C.bland_altman(sysb, x)
check("bias recovers +5", abs(ba["bias"] - 5) < 1e-6, f"bias={ba['bias']:.6f}")
check("LoA brackets bias", ba["loa_low"] < ba["bias"] < ba["loa_high"])
check("bias CI brackets bias", ba["bias_ci_low"] <= ba["bias"] <= ba["bias_ci_high"])
# correlation is not agreement: perfectly correlated but biased
xx = np.array([10, 20, 30, 40, 50.0])
yy = xx + 5
check("correlation ~1 while biased", np.corrcoef(xx, yy)[0, 1] > 0.999
      and abs(C.bland_altman(yy, xx)["bias"] - 5) < 1e-9)

print("\n=== kappa ===")
g = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
check("kappa perfect == 1", abs(C.cohen_kappa(g, g) - 1) < 1e-12)
check("kappa chance == 0", abs(C.cohen_kappa(g, np.array([0, 1, 2, 0, 1, 2, 1, 2, 0, 1]))) < 0.5)
check("quadratic weighted kappa finite", np.isfinite(C.cohen_kappa(g, g, weights="quadratic")))

print("\n=== cluster bootstrap respects groups ===")
rows = []
for gi in range(40):
    for k in range(5):
        rows.append(dict(group_id=f"g{gi}", v=10 + gi * 0.1 + rng.normal(0, 1)))
df = pd.DataFrame(rows)
r = C.cluster_bootstrap_ci(df, lambda d: d.v.mean(), n_boot=300)
check("n_groups == 40", r["n_groups"] == 40)
check("CI brackets the mean", r["ci_low"] <= r["point"] <= r["ci_high"],
      f"[{r['ci_low']:.3f},{r['ci_high']:.3f}]")

print("\n=== real LOSO predictions ===")
from pathlib import Path
loo = Path("/Users/moniaz/niki/results/loo")
found = sorted(loo.glob("*_branch_*_test_preds.csv"))
check("prediction files present", len(found) > 0, f"{len(found)} files")
if found:
    d = pd.read_csv(found[0])
    print("   columns:", list(d.columns))
    check("has label", "label" in d.columns)
    check("has group_id", "group_id" in d.columns)

print()
if fails:
    print(f"{len(fails)} FAILURE(S): {fails}")
    sys.exit(1)
print("all self-tests passed")
