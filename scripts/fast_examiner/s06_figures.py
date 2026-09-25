"""Step 6 - the three required figures for the fast examiner analysis."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "results/fast_examiner"
MODELS = ["B", "C", "E", "G"]
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
plt.rcParams.update({"figure.dpi": 160, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

met = pd.read_csv(OUT / "06_model_metrics.csv").set_index("model")
perf = pd.read_csv(OUT / "06b_per_fold_metrics.csv")

# ---------------- F1: AUC per fold + pooled
fig, ax = plt.subplots(figsize=(7.2, 4.0))
folds = sorted(perf.fold.unique())
w = 0.18
x = np.arange(len(MODELS))
for i, k in enumerate(folds):
    vals = [perf[(perf.fold == k) & (perf.model == m)].auc.iloc[0] for m in MODELS]
    ax.bar(x + (i - 1.5) * w, vals, w, label=f"fold {k} (test)")
    for xi, v in zip(x + (i - 1.5) * w, vals):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.5)
pooled = [met.loc[m, "multiclass_auc"] for m in MODELS]
ax.bar(x + 1.5 * w, pooled, w, label="pooled OOF", color="black", alpha=0.75)
for xi, v in zip(x + 1.5 * w, pooled):
    ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["B\nRGB emb", "C\nRGB+bio", "E\nRGB+vessel", "G\nRGB+vessel+bio"])
ax.set_ylabel("macro OvR ROC-AUC")
ax.set_title("F1 — 3-fold patient-level CV on FARFUM-RoP (k=3, seed 42)")
ax.set_ylim(0.45, 0.92)
ax.legend(fontsize=7, ncol=4, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "F1_3fold_model_auc.png")
plt.close(fig)
print("F1 written; pooled:", dict(zip(MODELS, [round(v, 4) for v in pooled])))

# ---------------- F2: delta AUC forest
b = pd.read_csv(OUT / "07_paired_bootstrap.csv")
b = b[b.metric == "multiclass_auc"]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
labels = {"C-B": "C − B  (add 5 scalar biomarkers)",
          "E-B": "E − B  (add spatial vessel map)",
          "G-E": "G − E  (add 5 biomarkers on top of vessel)"}
ys = np.arange(len(b))[::-1]
for yv, (_, r) in zip(ys, b.iterrows()):
    col = "#1b7f3b" if r.ci_lo > 0 else ("#8b1a1a" if r.ci_hi < 0 else "#555555")
    ax.errorbar(r.delta, yv, xerr=[[r.delta - r.ci_lo], [r.ci_hi - r.delta]], fmt="o",
                color=col, capsize=4, lw=1.6, ms=6)
    ax.text(r.ci_hi + 0.0008, yv, f"Δ {r.delta:+.4f}  [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]  p={r.p_value:.4f}",
            va="center", fontsize=7)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_yticks(ys)
ax.set_yticklabels([labels.get(c, c) for c in b.comparison], fontsize=8)
ax.set_xlabel("Δ macro OvR ROC-AUC (paired, patient-level bootstrap)")
ax.set_xlim(-0.006, 0.030)
ax.set_title(f"F2 — paired ΔAUC, patient-level bootstrap ({int(b.n_replicates.max())} replicates, seed 42)")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "F2_delta_auc_forest.png")
plt.close(fig)
print("F2 written")

# ---------------- F3: biomarker correlation heatmap
pear = np.load(OUT / "redundancy_pearson_fold0dev.npy")
spear = np.load(OUT / "redundancy_spearman_fold0dev.npy")
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
for ax, M, name in ((axes[0], pear, "Pearson"), (axes[1], spear, "Spearman")):
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(5)); ax.set_xticklabels(FEATS, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(5)); ax.set_yticklabels(FEATS, fontsize=7)
    ax.grid(False)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(M[i, j]) > 0.6 else "black")
    ax.set_title(f"{name} correlation — fold-0 development data (n=1039)", fontsize=8)
fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
fig.suptitle("F3 — redundancy audit of the five FINAL_PRIMARY scalar biomarkers", fontsize=9)
fig.savefig(OUT / "F3_biomarker_correlation.png", bbox_inches="tight")
plt.close(fig)
print("F3 written")
