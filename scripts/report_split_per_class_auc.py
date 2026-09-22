import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
A = Path("/root/niki_rop_task6_isolated/artifacts")
OUT = Path("/root/niki_rop_task6_isolated/report_assets_final/figures/main")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
t6 = pd.read_csv(A/"task6/metrics_summary.csv").set_index("model")
t8 = pd.read_csv(A/"task8_spatial_vessel_fusion/metrics_summary.csv").set_index("model")
MODS = ["B_EMBEDDING_ONLY", "C_PRIMARY", "D_VESSEL_MAP", "E_RGB_VESSEL_FEATURE_FUSION", "G_RGB_VESSEL_SCALAR_FUSION"]
SH = {"B_EMBEDDING_ONLY": "B-emb", "C_PRIMARY": "C", "D_VESSEL_MAP": "D",
      "E_RGB_VESSEL_FEATURE_FUSION": "E", "G_RGB_VESSEL_SCALAR_FUSION": "G"}
COLS = [("auc_Normal", "R3_normal", "Normal"), ("auc_Pre_Plus", "R3_preplus", "Pre-Plus"), ("auc_Plus", "R3_plus", "Plus")]
rows = []
for col, name, lab in COLS:
    v = []
    for m in MODS:
        v.append(float((t6 if m in t6.index else t8).loc[m][col]))
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(MODS))
    ax.bar(x, v, color=["#c8c8c8", "#a6a6a6", "#8a8a8a", "#5a5a5a", "#1b1b1b"],
           edgecolor="#333333", linewidth=0.8, width=0.6)
    ax.set_ylim(0, 1.08)
    for xx, vv in zip(x, v):
        ax.text(xx, vv + 0.015, f"{vv:.3f}", ha="center", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels([SH[m] for m in MODS])
    ax.set_ylabel("one-vs-rest AUC")
    ax.set_title(f"Canonical test set, N = 1331 - {lab} class", fontsize=12)
    fig.tight_layout()
    for e in ("png", "pdf", "svg"):
        fig.savefig(OUT/f"{name}.{e}", dpi=300 if e == "png" else None)
    plt.close(fig)
    for m, vv in zip(MODS, v):
        rows.append({"model": m, "class": lab, "auc": vv})
pd.DataFrame(rows).to_csv("/root/niki_rop_task6_isolated/report_assets_final/source_data/R3_per_class_auc.csv", index=False)
print("R3_SPLIT_DONE")
