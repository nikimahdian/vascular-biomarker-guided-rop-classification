"""Part 2 - benchmark figures: AUC bar chart and the paired delta-AUC forest for the slide."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("/root/niki_rop_task6_isolated/results/fast_benchmark")
plt.rcParams.update({"figure.dpi": 170, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True})

met = pd.read_csv(OUT / "04_common_metrics.csv")
stat = pd.read_csv(OUT / "05_paired_statistics.csv")
SHORT = {"Previous Lab B5 (3-class)": "Previous Lab\nB5 (3-class)",
         "ROPDeepX 2026 (harmonized)": "ROPDeepX 2026\n(harmonized)",
         "Current RGB embedding (B)": "Current B\n(RGB)",
         "Current RGB + biomarkers (C)": "Current C\n(RGB+bio)",
         "Current RGB + vessel (E)": "Current E\n(RGB+vessel)",
         "Current RGB + vessel + biomarkers (G)": "Current G\n(RGB+vessel+bio)"}

# ---------------- benchmark AUC bars
fig, ax = plt.subplots(figsize=(7.6, 4.0))
order = list(SHORT)
vals = [float(met[met.method == m].auc_macro_ovr.iloc[0]) for m in order]
colors = ["#8b1a1a", "#7a4fbf", "#4a6fa5", "#4a6fa5", "#1b7f3b", "#1b7f3b"]
bars = ax.bar(range(len(order)), vals, color=colors, width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v + 0.006, f"{v:.4f}", ha="center", fontsize=8,
            fontweight="bold" if order[i].startswith("Current E") else "normal")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([SHORT[m] for m in order], fontsize=7.5)
ax.set_ylabel("macro OvR ROC-AUC")
ax.set_ylim(0.45, max(vals) + 0.06)
ax.set_title("Fair FARFUM patient-level 3-fold benchmark — identical patients, folds and metrics")
ax.annotate("current primary\nvessel-fusion method", xy=(4, vals[4]), xytext=(4, vals[4] - 0.10),
            ha="center", fontsize=7, color="#1b7f3b",
            arrowprops=dict(arrowstyle="->", color="#1b7f3b", lw=1))
fig.tight_layout()
fig.savefig(OUT / "10_benchmark_auc_bar.png")
plt.close(fig)
print("10_benchmark_auc_bar.png", dict(zip([SHORT[m].replace(chr(10), ' ') for m in order],
                                           [round(v, 4) for v in vals])))

# ---------------- paired delta AUC forest
a = stat[stat.metric == "auc"].set_index("comparison")
pairs = [("E-ROPDeepX", "E − ROPDeepX\n(current vs harmonized publication)"),
         ("E-Legacy", "E − Previous Lab B5\n(current vs legacy architecture)"),
         ("E-B", "E − Current B\n(vessel representation value)")]
fig, ax = plt.subplots(figsize=(7.4, 3.0))
ys = np.arange(len(pairs))[::-1]
for yv, (key, lab) in zip(ys, pairs):
    r = a.loc[key]
    col = "#1b7f3b" if r.ci_lo > 0 else ("#8b1a1a" if r.ci_hi < 0 else "#555555")
    ax.errorbar(r.delta, yv, xerr=[[r.delta - r.ci_lo], [r.ci_hi - r.delta]], fmt="o",
                color=col, capsize=4, lw=1.7, ms=7)
    ax.text(r.ci_hi + 0.004, yv, f"Δ {r.delta:+.4f}  [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]  "
            f"p={r.p_value:.4f}", va="center", fontsize=7)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_yticks(ys)
ax.set_yticklabels([lab for _k, lab in pairs], fontsize=7.5)
ax.set_xlabel("Δ macro OvR ROC-AUC (paired, patient-level bootstrap, 10,000 replicates)")
ax.set_title("Paired ΔAUC on identical held-out patients", fontsize=9)
lo = min(a.loc[k].ci_lo for k, _ in pairs)
hi = max(a.loc[k].ci_hi for k, _ in pairs)
span = hi - lo
ax.set_xlim(lo - 0.35 * span, hi + 1.15 * span)
fig.tight_layout()
fig.savefig(OUT / "11_paired_delta_auc_forest.png")
plt.close(fig)
print("11_paired_delta_auc_forest.png")
for key, _lab in pairs:
    r = a.loc[key]
    print(f"  {key:12s} {r.delta:+.6f} [{r.ci_lo:+.6f}, {r.ci_hi:+.6f}] p={r.p_value:.4f}")
