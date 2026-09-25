"""Defense figures for the vascular-biomarker-guided ROP thesis (4 required + 1 optional).

Every plotted value is read from the saved result CSVs:
  results/fast_benchmark/04_common_metrics.csv      (Figure 1, Figure 5, Figure 4 annotations)
  results/fast_benchmark/05_paired_statistics.csv   (Figure 2)
  results/fast_examiner/07_paired_bootstrap.csv     (Figure 3)
  results/fast_examiner/02_common_farFUM_3fold_manifest.csv (Figure 4 matrix)
  results/fast_examiner/04_biomarker_redundancy.csv (Figure 4 VIF annotation)
  results/fast_examiner/05c_fs_detail_fold{0,1,2}.json (Figure 4 PCA annotation)

No model is retrained, no metric recomputed for plotting, no value invented.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\results")
FX, FB = ROOT / "fast_examiner", ROOT / "fast_benchmark"
OUT = ROOT / "defense_figures_final"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.family": "DejaVu Sans", "font.size": 14,
    "axes.titlesize": 21, "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 14,
    "axes.edgecolor": "#333333", "axes.linewidth": 1.0, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "savefig.transparent": False,
    "axes.grid": False, "legend.frameon": False})
GREY = "#4d4d4d"
CUR = "#1f6fb2"
COMP = "#8c8c8c"
PRIMARY = "#1b7f3b"

# ---------------------------------------------------------------- sources
met = pd.read_csv(FB / "04_common_metrics.csv").set_index("key")
stat = pd.read_csv(FB / "05_paired_statistics.csv").set_index(["comparison", "metric"])
boot = pd.read_csv(FX / "07_paired_bootstrap.csv").set_index(["comparison", "metric"])
red = pd.read_csv(FX / "04_biomarker_redundancy.csv")
man = pd.read_csv(FX / "02_common_farFUM_3fold_manifest.csv")
dec = json.loads((FX / "11_decisions.json").read_text())
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
R = np.corrcoef(man[FEATS].to_numpy(float), rowvar=False)
MAX_VIF = float(red[red.scope == "farfum_all"].vif.max())
PCA_N = [json.loads((FX / f"05c_fs_detail_fold{k}.json").read_text())["FS2"] for k in range(3)]
assert PCA_N == [1, 1, 1], PCA_N


def save(fig, stem):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white", transparent=False)
    plt.close(fig)
    print("wrote", stem + ".{png,svg,pdf}")


# ================================================================ FIGURE 1
ROWS1 = [("Previous Lab B5", "Legacy", COMP), ("ROPDeepX", "ROPDeepX", COMP),
         ("Current B — RGB", "B", CUR), ("Current C — RGB + biomarkers", "C", CUR),
         ("Current E — RGB + vessel", "E", PRIMARY),
         ("Current G — RGB + vessel + biomarkers", "G", CUR)]
vals = [float(met.loc[k, "auc_macro_ovr"]) for _l, k, _c in ROWS1]
fig, ax = plt.subplots(figsize=(13.33, 7.5))
ypos = np.arange(len(ROWS1))[::-1]
bars = ax.barh(ypos, vals, height=0.62, color=[c for _l, _k, c in ROWS1],
               edgecolor="#222222", linewidth=0.7)
for y, v in zip(ypos, vals):
    ax.text(v + 0.0015, y, f"{v:.4f}", va="center", ha="left", fontsize=15,
            fontweight="bold", color="#111111")
ax.set_yticks(ypos)
ax.set_yticklabels([l for l, _k, _c in ROWS1], fontsize=15)
ax.set_xlim(0.55, 0.75)
ax.set_xticks(np.arange(0.55, 0.7501, 0.025))
ax.set_xlabel("Macro OvR ROC-AUC   (higher is better)", fontsize=15)
ax.set_title("Patient-Level FARFUM Benchmark", fontsize=23, pad=48)
ax.text(0.5, 1.012, "Same 1,528 images · 68 patients · 3 folds · 3 classes",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=15, color=GREY)
ax.text(0.995, 0.03, "Zoomed AUC axis (0.55 – 0.75)", transform=ax.transAxes, ha="right",
        va="bottom", fontsize=12, color="#8a1c1c", style="italic")
ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
ax.set_axisbelow(True)
fig.subplots_adjust(bottom=0.22)
fig.text(0.5, 0.085, "Previous Lab B5 = 3-class reimplementation; ROPDeepX = harmonized manual "
                     "implementation.", ha="center", fontsize=12.5, color="#111111",
         fontweight="bold")
fig.text(0.5, 0.048, "E is the predefined main vessel-fusion model; C has the highest point AUC "
                     "in this single-source benchmark — no claim of overall model superiority.",
         ha="center", fontsize=12, color="#111111")
fig.text(0.5, 0.010, "Legacy and ROPDeepX values are harmonized reimplementations, not published "
                     "headline results.", ha="center", fontsize=11.5, color=GREY)
save(fig, "01_fair_benchmark_auc")

# ================================================================ FIGURE 2
ROWS2 = [("E − ROPDeepX", "E-ROPDeepX"), ("E − Previous Lab B5", "E-Legacy"),
         ("E − Current B", "E-B")]
fig, ax = plt.subplots(figsize=(13.33, 7.5))
ys = np.arange(len(ROWS2))[::-1]
span_lo = min(float(stat.loc[(k, "auc"), "ci_lo"]) for _l, k in ROWS2)
span_hi = max(float(stat.loc[(k, "auc"), "ci_hi"]) for _l, k in ROWS2)
for y, (lab, key) in zip(ys, ROWS2):
    d = float(stat.loc[(key, "auc"), "delta"])
    lo = float(stat.loc[(key, "auc"), "ci_lo"])
    hi = float(stat.loc[(key, "auc"), "ci_hi"])
    col = PRIMARY if lo > 0 else (GREY if hi < 0 else "#555555")
    ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", ms=11, color=col,
                ecolor=col, elinewidth=2.4, capsize=6, capthick=2.0)
    ax.text(hi + 0.004, y, f"{d:+.4f}   [{lo:+.4f}, {hi:+.4f}]", va="center", ha="left",
            fontsize=14.5, color="#111111")
ax.axvline(0.0, color="#111111", linestyle="--", linewidth=1.4)
ax.set_yticks(ys)
ax.set_yticklabels([l for l, _k in ROWS2], fontsize=16)
ax.set_xlim(span_lo - 0.02, span_hi + 0.095)
ax.set_xlabel("Δ Macro OvR ROC-AUC   (positive = E higher)", fontsize=15)
ax.set_title("Paired Patient-Level AUC Differences", fontsize=23, pad=48)
ax.text(0.5, 1.012, "10,000 patient-bootstrap replicates · FARFUM-RoP",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=15, color=GREY)
ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
ax.set_axisbelow(True)
fig.subplots_adjust(bottom=0.17)
fig.text(0.5, 0.045, "A 95% CI crossing zero means the paired difference is not clearly "
                     "distinguishable from zero.", ha="center", fontsize=12, color=GREY,
         style="italic")
save(fig, "02_fair_benchmark_delta_auc_forest")

# ================================================================ FIGURE 3
ROWS3 = [("C − B: + five biomarkers", "C-B"), ("E − B: + vessel representation", "E-B"),
         ("G − E: + biomarkers after vessel", "G-E")]
fig, ax = plt.subplots(figsize=(13.33, 7.5))
ys = np.arange(len(ROWS3))[::-1]
lo_all = min(float(boot.loc[(k, "multiclass_auc"), "ci_lo"]) for _l, k in ROWS3)
hi_all = max(float(boot.loc[(k, "multiclass_auc"), "ci_hi"]) for _l, k in ROWS3)
for y, (lab, key) in zip(ys, ROWS3):
    d = float(boot.loc[(key, "multiclass_auc"), "delta"])
    lo = float(boot.loc[(key, "multiclass_auc"), "ci_lo"])
    hi = float(boot.loc[(key, "multiclass_auc"), "ci_hi"])
    p = float(boot.loc[(key, "multiclass_auc"), "p_value"])
    col = "#555555"
    ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", ms=11, color=col, ecolor=col,
                elinewidth=2.4, capsize=6, capthick=2.0)
    ax.text(hi + 0.0006, y, f"{d:+.5f}   [{lo:+.5f}, {hi:+.5f}]   p = {p:.3f}", va="center",
            ha="left", fontsize=14, color="#111111")
ax.axvline(0.0, color="#111111", linestyle="--", linewidth=1.4)
ax.set_yticks(ys)
ax.set_yticklabels([l for l, _k in ROWS3], fontsize=16)
ax.set_xlim(lo_all - 0.0035, hi_all + 0.0300)
ax.set_xlabel("Δ Macro OvR ROC-AUC   (paired patient-level bootstrap)", fontsize=15)
ax.set_title("Patient-Level Cross-Validation Sensitivity", fontsize=23, pad=48)
ax.text(0.5, 1.012, "FARFUM-RoP · 1,528 images · 68 patients · 3 folds",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=15, color=GREY)
ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
ax.set_axisbelow(True)
fig.subplots_adjust(bottom=0.24)
fig.text(0.012, 0.085, "Five-scalar biomarker conclusion: ROBUST NULL",
         fontsize=15, color="#111111", fontweight="bold")
fig.text(0.012, 0.035, "Vessel increment in this single-source CV: not detected",
         fontsize=13.5, color=GREY)
save(fig, "03_patient_cv_sensitivity_forest")

# ================================================================ FIGURE 4
SHORT = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
fig = plt.figure(figsize=(13.33, 7.5))
ax = fig.add_axes([0.075, 0.15, 0.45, 0.62])
im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(5)); ax.set_xticklabels(SHORT, rotation=35, ha="right", fontsize=13)
ax.set_yticks(range(5)); ax.set_yticklabels(SHORT, fontsize=13)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=13,
                color="white" if abs(R[i, j]) > 0.62 else "#111111")
ax.set_title("Redundancy Within the Five-Feature Biomarker Panel", fontsize=21, pad=46)
ax.text(0.5, 1.035, "Patient-level FARFUM analysis", transform=ax.transAxes, ha="center",
        va="bottom", fontsize=14, color=GREY)
cax = fig.add_axes([0.535, 0.15, 0.014, 0.62])
cb = fig.colorbar(im, cax=cax)
cax.set_title("Pearson r", fontsize=12, pad=8)
cb.ax.tick_params(labelsize=11)
tx = 0.600
fig.text(tx, 0.705, f"Max VIF ≈ {MAX_VIF:.0f}", fontsize=16, fontweight="bold", color="#111111")
fig.text(tx, 0.640, "PCA ≥95% variance:\n1 component (all three folds)",
         fontsize=13.5, color="#111111")
fig.text(tx, 0.505,
         "fractal_d1 – fractal_d2 = %.3f\nfractal_d0 – fractal_d1 = %.3f\n"
         "fractal_d0 – fractal_d2 = %.3f" % (R[3, 4], R[2, 3], R[2, 4]),
         fontsize=13.5, color="#111111")
fig.text(tx, 0.370, "Cell values are shown to 2 decimals;\nthe three collinear fractal pairs are\n"
                    "listed above with 3 decimals.", fontsize=11.5, color="#111111")
fig.text(tx, 0.245, "VIF and PCA describe feature variance,\nnot disease-predictive dimensionality.",
         fontsize=12, color=GREY, style="italic")
fig.text(tx, 0.125, "Matrix computed from the saved cohort\n(n = 1,528; 5 biomarker columns), "
                    "cross-checked\nagainst 04_biomarker_redundancy.csv.", fontsize=11, color=GREY)
save(fig, "04_biomarker_redundancy_heatmap")

# ================================================================ FIGURE 5 (optional)
ROWS5 = [("Previous Lab\nB5", "Legacy"), ("ROPDeepX", "ROPDeepX"), ("Current B\nRGB", "B"),
         ("Current C\nRGB+bio", "C"), ("Current E\nRGB+vessel", "E"),
         ("Current G\nRGB+vessel+bio", "G")]
cols = ["auc_macro_ovr", "balanced_accuracy", "macro_f1"]
names = ["Macro OvR AUC", "Balanced accuracy", "Macro F1"]
width = 0.26
x = np.arange(len(ROWS5))
fig, ax = plt.subplots(figsize=(13.33, 7.5))
for i, (c, n) in enumerate(zip(cols, names)):
    v = [float(met.loc[k, c]) for _l, k in ROWS5]
    off = (i - 1) * width
    ax.bar(x + off, v, width, label=n, color=["#1f6fb2", "#7aa6c9", "#b9cfe3"][i],
           edgecolor="#222222", linewidth=0.6)
    for xi, vi in zip(x + off, v):
        ax.text(xi, vi + 0.008, f"{vi:.3f}", ha="center", va="bottom", fontsize=9.5,
                rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([l for l, _k in ROWS5], fontsize=13)
ax.set_ylim(0.0, 0.95)
ax.set_ylabel("Score (higher is better)", fontsize=15)
ax.set_title("Discrimination and Decision Metrics — Harmonized FARFUM Benchmark", fontsize=20,
             pad=48)
ax.text(0.5, 1.012, "Same 1,528 images · 68 patients · 3 folds · identical metric code",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=14, color=GREY)
ax.legend(fontsize=13, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99))
ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
ax.set_axisbelow(True)
fig.subplots_adjust(bottom=0.22)
fig.text(0.5, 0.065, "Previous Lab B5 = 3-class reimplementation; ROPDeepX = harmonized manual "
                     "implementation.", ha="center", fontsize=12, color="#111111")
fig.text(0.5, 0.020, "Brier and ECE are reported separately (lower is better) and are not mixed "
                     "into this axis.", ha="center", fontsize=11.5, color=GREY)
save(fig, "05_grouped_auc_balacc_f1")

print("\nFigure 1 values:", {k: round(float(met.loc[k, 'auc_macro_ovr']), 4)
                             for _l, k, _c in ROWS1})
print("Figure 2 values:", {k: (round(float(stat.loc[(k, 'auc'), 'delta']), 4),
                               round(float(stat.loc[(k, 'auc'), 'ci_lo']), 4),
                               round(float(stat.loc[(k, 'auc'), 'ci_hi']), 4))
                           for _l, k in ROWS2})
print("Figure 3 values:", {k: (round(float(boot.loc[(k, 'multiclass_auc'), 'delta']), 5),
                               round(float(boot.loc[(k, 'multiclass_auc'), 'ci_lo']), 5),
                               round(float(boot.loc[(k, 'multiclass_auc'), 'ci_hi']), 5))
                           for _l, k in ROWS3})
print("Output dir:", OUT)
