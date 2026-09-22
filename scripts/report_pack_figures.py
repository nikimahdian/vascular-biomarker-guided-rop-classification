"""Final report asset pack - PART 1: figures + source data. Frozen artifacts only, no training.

Every plotted number is read from a frozen artifact under
/root/niki_rop_task6_isolated/artifacts. Nothing is retrained, no prediction is regenerated, and no
number is entered by hand. Each figure is written as PNG (300 dpi), PDF and SVG, and every figure
gets a matching CSV in source_data/.
"""
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

WS = Path("/root/niki_rop_task6_isolated")
A = WS / "artifacts"
OUT = WS / "report_assets_final"
MAIN, METH, APP = OUT / "figures/main", OUT / "figures/methods", OUT / "figures/appendix"
DIA, SRC = OUT / "diagrams", OUT / "source_data"
for d in (MAIN, METH, APP, DIA, SRC, OUT / "tables/main", OUT / "tables/appendix",
          OUT / "captions", OUT / "latex/tables", OUT / "metadata"):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": False, "font.size": 8, "axes.titlesize": 9,
                     "axes.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})

LABEL = {"A_PRIMARY": "Scalar biomarkers", "B_RGB": "RGB CNN",
         "B_EMBEDDING_ONLY": "RGB embedding", "C_PRIMARY": "RGB + biomarkers",
         "D_VESSEL_MAP": "Vessel map CNN", "E_RGB_VESSEL_FEATURE_FUSION": "RGB + vessel",
         "F_RGB_VESSEL_LATE_FUSION": "RGB + vessel late fusion",
         "G_RGB_VESSEL_SCALAR_FUSION": "RGB + vessel + biomarkers",
         "H_JOINT_RGB_VESSEL": "Joint RGB-vessel",
         "I_JOINT_RGB_VESSEL_BIOMARKER": "Joint RGB-vessel + biomarkers",
         "J0_FROZEN_FUSION_CONTROL": "Frozen fusion control",
         "J1_BIOMARKER_FILM": "Biomarker-conditioned FiLM",
         "K0_DOMAIN_NEUTRAL_CONTROL": "Domain-neutral control",
         "K1_CC_DANN_MMD": "CC-DANN + MMD",
         "L_ROPDEEPX_STYLE_RGB": "Dual-RGB attention (L)",
         "M0_ROPDEEPX_EMBEDDING_ONLY": "Dual-RGB embedding (M0)",
         "M1_ROPDEEPX_PLUS_VESSEL": "Dual-RGB + vessel (M1)",
         "B_LOSO": "RGB embedding (LOSO)", "E_LOSO": "RGB + vessel (LOSO)",
         "G_LOSO": "RGB + vessel + biomarkers (LOSO)",
         "TASK11_E_LOSO": "RGB + vessel (Task 11)"}
ORDER = ["A_PRIMARY", "B_RGB", "B_EMBEDDING_ONLY", "C_PRIMARY", "D_VESSEL_MAP",
         "E_RGB_VESSEL_FEATURE_FUSION", "F_RGB_VESSEL_LATE_FUSION",
         "G_RGB_VESSEL_SCALAR_FUSION"]
SHORT = {"A_PRIMARY": "A", "B_RGB": "B", "B_EMBEDDING_ONLY": "B-emb", "C_PRIMARY": "C",
         "D_VESSEL_MAP": "D", "E_RGB_VESSEL_FEATURE_FUSION": "E",
         "F_RGB_VESSEL_LATE_FUSION": "F", "G_RGB_VESSEL_SCALAR_FUSION": "G",
         "H_JOINT_RGB_VESSEL": "H", "I_JOINT_RGB_VESSEL_BIOMARKER": "I",
         "J0_FROZEN_FUSION_CONTROL": "J0", "J1_BIOMARKER_FILM": "J1",
         "L_ROPDEEPX_STYLE_RGB": "L", "M0_ROPDEEPX_EMBEDDING_ONLY": "M0",
         "M1_ROPDEEPX_PLUS_VESSEL": "M1"}
GREY = ["#1b1b1b", "#4d4d4d", "#7a7a7a", "#a6a6a6", "#c8c8c8"]
FAIL = []
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def save(fig, folder, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(folder / f"{name}.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)


def sd(df, name):
    df.to_csv(SRC / f"{name}.csv", index=False)


def fig_guard(name, fn):
    try:
        fn()
        log(f"  ok  {name}")
    except Exception as e:  # noqa: BLE001
        FAIL.append({"asset": name, "error": f"{type(e).__name__}: {e}",
                     "trace": traceback.format_exc()[-1200:]})
        log(f"  FAIL {name}: {type(e).__name__}: {e}")
        plt.close("all")


# ---------------------------------------------------------------- load frozen artifacts
t6m = pd.read_csv(A / "task6/metrics_summary.csv").set_index("model")
t8m = pd.read_csv(A / "task8_spatial_vessel_fusion/metrics_summary.csv").set_index("model")
t13m = pd.read_csv(A / "task13_ropdeepx_style/metrics_summary.csv").set_index("model")
t9m = pd.read_csv(A / "task9_joint_multimodal_fusion/metrics_summary.csv").set_index("model")
t10m = pd.read_csv(A / "task10_biomarker_film/metrics_summary.csv").set_index("model")
t7p = pd.read_csv(A / "task7_paired_statistics/paired_primary_metrics.csv")
t8bp = pd.read_csv(A / "task8_spatial_vessel_fusion/statistical_closure_10k/paired_metrics_10k.csv")
t9p = pd.read_csv(A / "task9_joint_multimodal_fusion/paired_all_10k.csv")
t10p = pd.read_csv(A / "task10_biomarker_film/paired_all_10k.csv")
t11p = pd.read_csv(A / "task11_source_heldout/paired_bootstrap_10k.csv")
t11m = pd.read_csv(A / "task11_source_heldout/metrics_per_source.csv")
t11s = pd.read_csv(A / "task11_source_heldout/domain_shift_diagnostics.csv")
t11b = pd.read_csv(A / "task11_source_heldout/biomarker_shift.csv")
t11c = pd.read_csv(A / "task11_source_heldout/population_audit.csv")
t12p = pd.read_csv(A / "task12_class_conditional_domain_generalization/paired_bootstrap_10k.csv")
t12m = pd.read_csv(A / "task12_class_conditional_domain_generalization/heldout_metrics.csv")
t12s = pd.read_csv(A / "task12_class_conditional_domain_generalization/representation_shift.csv")
t12d = pd.read_csv(A / "task12_class_conditional_domain_generalization/domain_predictability.csv")
t13p = pd.read_csv(A / "task13_ropdeepx_style/paired_all_10k.csv")
t13att = pd.read_csv(A / "task13_ropdeepx_style/attention_by_class_and_source.csv")
t13attst = pd.read_csv(A / "task13_ropdeepx_style/attention_stats_val_test.csv")
Lsel = json.loads((A / "task13_ropdeepx_style/L_selection.json").read_text())
Lhist = pd.DataFrame(json.loads((A / "task13_ropdeepx_style/L_history.json").read_text()))
Hhist = pd.DataFrame(json.loads((A / "task9_joint_multimodal_fusion/H_JOINT_RGB_VESSEL_history.json").read_text()))
Ihist = pd.DataFrame(json.loads((A / "task9_joint_multimodal_fusion/I_JOINT_RGB_VESSEL_BIOMARKER_history.json").read_text()))
J0hist = pd.DataFrame(json.loads((A / "task10_biomarker_film/J0_FROZEN_FUSION_CONTROL_history.json").read_text()))
J1hist = pd.DataFrame(json.loads((A / "task10_biomarker_film/J1_BIOMARKER_FILM_history.json").read_text()))
FILM = pd.read_csv(A / "task10_biomarker_film/film_modulation_diagnostics.csv")
SENS = pd.read_csv(A / "task10_biomarker_film/biomarker_sensitivity.csv")
manifest = pd.read_csv(WS / "primary_complete_case_v2_server2.csv", low_memory=False)

MET = ["multiclass_auc", "balanced_accuracy", "macro_f1", "brier", "ece"]
PRETTY = {"multiclass_auc": "Multiclass AUC", "balanced_accuracy": "Balanced accuracy",
          "macro_f1": "Macro F1", "brier": "Brier", "ece": "ECE",
          "restricted_binary_auc": "Restricted binary AUC (Normal vs Plus)"}
DIRN = {"multiclass_auc": "up", "balanced_accuracy": "up", "macro_f1": "up",
        "brier": "down", "ece": "down"}


def prow(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v)
    return df[m].iloc[0]


def canon_metrics(model):
    for tbl in (t6m, t8m, t9m, t10m, t13m):
        if model in tbl.index:
            r = tbl.loc[model]
            return {k: float(r[k]) for k in MET} | {"auc_Normal": float(r["auc_Normal"]),
                                                    "auc_Pre_Plus": float(r["auc_Pre_Plus"]),
                                                    "auc_Plus": float(r["auc_Plus"])}
    raise KeyError(f"{model} not present in any frozen metrics table")


# ================================================================ PART A - METHODS
def m1():
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.axis("off")

    def box(x, y, w, h, t, fc="#f2f2f2", fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                    fc=fc, ec="#333333", lw=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.4)

    def arrow(x1, y1, x2, y2, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, lw=0.9,
                                     color="#333333", linestyle=ls, mutation_scale=11))

    box(0.36, 0.90, 0.28, 0.075, "RGB fundus image\n384 x 384")
    box(0.06, 0.71, 0.30, 0.075, "RGB CNN\nEfficientNet-B5 -> 2048-d")
    box(0.63, 0.71, 0.30, 0.075, "Vessel segmentation\n(frozen)")
    box(0.63, 0.56, 0.30, 0.07, "Vessel map\nbinary, 384 x 384")
    box(0.63, 0.40, 0.30, 0.075, "Spatial vessel CNN\nEfficientNet-B4 -> 1792-d")
    box(0.06, 0.40, 0.30, 0.075, "FOV-aware vascular\nmeasurement")
    box(0.06, 0.25, 0.30, 0.07, "5 scalar biomarkers", fc="#e8e8e8")
    box(0.30, 0.09, 0.40, 0.075, "Fusion models\n(concatenation / joint / FiLM)", fc="#dcdcdc")
    box(0.72, 0.09, 0.26, 0.075, "Normal / Pre-Plus / Plus", fc="#dcdcdc")
    box(0.02, 0.90, 0.28, 0.075, "Canonical split\n6203 / 1328 / 1331", fc="#f7f7f7", fs=7)
    box(0.02, 0.005, 0.44, 0.065, "Paired class-stratified bootstrap\n(seed 42, 10,000 replicates)",
        fc="#f7f7f7", fs=7)
    box(0.52, 0.005, 0.46, 0.065, "Source-held-out evaluation\n(plus / FARFUM-RoP / Farabi)",
        fc="#f7f7f7", fs=7)

    arrow(0.50, 0.90, 0.21, 0.785)
    arrow(0.50, 0.90, 0.78, 0.785)
    arrow(0.78, 0.71, 0.78, 0.63)
    arrow(0.78, 0.56, 0.78, 0.475)
    arrow(0.21, 0.71, 0.21, 0.475)
    arrow(0.21, 0.40, 0.21, 0.32)
    arrow(0.21, 0.25, 0.38, 0.165)
    arrow(0.78, 0.40, 0.62, 0.165)
    arrow(0.70, 0.128, 0.72, 0.128)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("M1  Complete study pipeline", fontsize=10, pad=6)
    save(fig, METH, "M1_complete_pipeline")
    sd(pd.DataFrame({"stage": ["RGB fundus", "RGB CNN", "segmentation", "vessel map",
                               "vessel CNN", "vascular measurement", "5 biomarkers", "fusion",
                               "output", "canonical split", "paired bootstrap",
                               "source-held-out"],
                     "detail": ["384x384", "EfficientNet-B5 2048-d", "frozen SEG_CURRENT_V2",
                                "binary 384x384", "EfficientNet-B4 1792-d", "FOV-aware",
                                "vessel_density_fov, skel_density_fov, fractal_d0/d1/d2",
                                "concatenation / joint / FiLM", "Normal / Pre-Plus / Plus",
                                "6203/1328/1331", "seed 42, 10000 replicates",
                                "plus / FARFUM-RoP / Farabi"]},
                    columns=["stage", "detail"]), "M1_complete_pipeline_nodes")


def m2():
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.axis("off")

    def box(x, y, w, h, t, fc="#f2f2f2", fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                    fc=fc, ec="#333333", lw=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.4)
    box(0.02, 0.42, 0.20, 0.30, "RGB fundus\nimage")
    box(0.28, 0.42, 0.20, 0.30, "Vessel\nsegmentation")
    box(0.54, 0.34, 0.20, 0.46, "FOV-aware\nvascular\nmeasurement")
    box(0.80, 0.42, 0.18, 0.30, "5 final scalar\nbiomarkers", fc="#dcdcdc")
    for x in (0.22, 0.48, 0.74):
        ax.add_patch(FancyArrowPatch((x, 0.57), (x + 0.06, 0.57), arrowstyle="-|>",
                                     lw=1.0, color="#333333", mutation_scale=11))
    ax.text(0.5, 0.09, "vessel_density_fov   -   skel_density_fov   -   fractal_d0   -   "
                       "fractal_d1   -   fractal_d2", ha="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("M2  Five scalar biomarker pipeline (FINAL_BIOMARKERS_V2)", fontsize=10, pad=4)
    save(fig, METH, "M2_biomarker_pipeline")
    sd(pd.DataFrame({"step": ["RGB fundus image", "vessel segmentation",
                              "FOV-aware vascular measurement", "five scalar biomarkers"],
                     "detail": ["canonical 384x384 input", "frozen SEG_CURRENT_V2_RESOLVER_SAFE",
                                "coverage denominator = content area (V5)",
                                "vessel_density_fov, skel_density_fov, fractal_d0, fractal_d1, "
                                "fractal_d2"]}), "M2_biomarker_pipeline_steps")


def m3():
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.axis("off")
    rows = [
        ("A  Scalar biomarkers", "5 biomarkers  ->  XGBoost  ->  3 classes", "#f2f2f2"),
        ("B  RGB CNN", "RGB  ->  EfficientNet-B5  ->  classifier  ->  3 classes", "#f2f2f2"),
        ("B-emb  RGB embedding", "RGB  ->  EfficientNet-B5  ->  2048-d  ->  XGBoost", "#e8e8e8"),
        ("C  RGB + biomarkers", "2048-d RGB + 5 biomarkers (2053-d)  ->  XGBoost", "#e8e8e8"),
        ("D  Vessel map CNN", "Vessel map  ->  EfficientNet-B4  ->  classifier  ->  3 classes",
         "#f2f2f2"),
        ("E  RGB + vessel", "2048-d RGB + 1792-d vessel (3840-d)  ->  XGBoost", "#dcdcdc"),
        ("G  RGB + vessel + biomarkers",
         "2048-d RGB + 1792-d vessel + 5 biomarkers (3845-d)  ->  XGBoost", "#dcdcdc"),
    ]
    y = 0.90
    for title, body, fc in rows:
        ax.add_patch(FancyBboxPatch((0.02, y - 0.095), 0.96, 0.095,
                                    boxstyle="round,pad=0.008", fc=fc, ec="#333333", lw=0.9))
        ax.text(0.04, y - 0.030, title, fontsize=8.5, va="center", fontweight="bold")
        ax.text(0.04, y - 0.070, body, fontsize=8, va="center")
        y -= 0.125
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("M3  Primary model architectures and feature dimensionalities", fontsize=10, pad=4)
    save(fig, METH, "M3_model_architectures")
    sd(pd.DataFrame([{"model": t, "architecture": b} for t, b, _ in rows]),
       "M3_model_architectures")


def m4():
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.axis("off")
    tasks = [("Task 6", "Which representation predicts best on the canonical split?"),
             ("Task 7", "Are the differences statistically supported?"),
             ("Task 8", "Does spatial vessel morphology add information?"),
             ("Task 8B", "Does that survive a 10,000-replicate closure?"),
             ("Task 9", "Can learned joint fusion beat concatenation?"),
             ("Task 10", "Can biomarkers help by conditioning the vessel branch?"),
             ("Task 11", "How well does this transfer to an unseen source?"),
             ("Task 12", "Can class-conditional domain invariance fix the transfer?"),
             ("Task 13", "Does vessel complementarity survive a different RGB backbone?")]
    y = 0.94
    for i, (t, q) in enumerate(tasks):
        fc = "#dcdcdc" if i % 2 == 0 else "#f2f2f2"
        ax.add_patch(FancyBboxPatch((0.10, y - 0.082), 0.86, 0.082,
                                    boxstyle="round,pad=0.008", fc=fc, ec="#333333", lw=0.9))
        ax.text(0.14, y - 0.041, f"{t}:  {q}", fontsize=8, va="center")
        if i < len(tasks) - 1:
            ax.add_patch(FancyArrowPatch((0.53, y - 0.082), (0.53, y - 0.108),
                                         arrowstyle="-|>", lw=0.9, color="#333333",
                                         mutation_scale=10))
        y -= 0.108
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("M4  Experimental question map", fontsize=10, pad=4)
    save(fig, METH, "M4_experimental_question_map")
    sd(pd.DataFrame([{"task": t, "scientific_question": q} for t, q in tasks]),
       "M4_experimental_question_map")


# ================================================================ PART B - COHORT
def d1():
    df = t11c.copy()
    df = df[df.source.isin(["plus", "farfum_rop", "farabi"])]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(df))
    w = 0.26
    for j, (col, lab, fc) in enumerate([("normal", "Normal", "#1b1b1b"),
                                        ("pre_plus", "Pre-Plus", "#7a7a7a"),
                                        ("plus_cls", "Plus", "#c8c8c8")]):
        b = ax.bar(x + (j - 1) * w, df[col].values, w, label=lab, color=fc, edgecolor="#333333",
                   linewidth=0.6)
        ax.bar_label(b, fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "-") for s in df.source])
    ax.set_ylabel("Images")
    ax.set_ylim(0, df[["normal", "pre_plus", "plus_cls"]].values.max() * 1.20)
    ax.legend(frameon=False)
    ax.set_title("D1  Class distribution by acquisition source (N = 8,862)", fontsize=9)
    ax.annotate("Pre-Plus = 0\n(absent from this source)", xy=(0, 0), xytext=(0.02, 0.62),
                textcoords="axes fraction", fontsize=7,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))
    save(fig, MAIN, "D1_source_class_distribution")
    sd(df[["source", "n", "groups", "normal", "pre_plus", "plus_cls"]],
       "D1_source_class_distribution")


def d2():
    rows = []
    for s in ("train", "val", "test"):
        sub = manifest[manifest.split == s]
        rows.append({"split": s, "n": len(sub), "Normal": int((sub.label == 0).sum()),
                     "Pre-Plus": int((sub.label == 1).sum()),
                     "Plus": int((sub.label == 2).sum())})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x = np.arange(len(df))
    w = 0.26
    for j, (col, fc) in enumerate([("Normal", "#1b1b1b"), ("Pre-Plus", "#7a7a7a"),
                                   ("Plus", "#c8c8c8")]):
        b = ax.bar(x + (j - 1) * w, df[col].values, w, label=col, color=fc, edgecolor="#333333",
                   linewidth=0.6)
        ax.bar_label(b, fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(df.split, df.n)])
    ax.set_ylabel("Images")
    ax.set_ylim(0, df[["Normal", "Pre-Plus", "Plus"]].values.max() * 1.20)
    ax.legend(frameon=False)
    ax.set_title("D2  Class distribution across canonical splits", fontsize=9)
    save(fig, MAIN, "D2_split_class_distribution")
    sd(df, "D2_split_class_distribution")


def d3():
    df = t11c[t11c.source.isin(["plus", "farfum_rop", "farabi"])].copy()
    tot = df.n.sum()
    df["pct"] = 100.0 * df.n / tot
    df = df.sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    b = ax.barh([s.replace("_", "-") for s in df.source], df.n.values, color="#7a7a7a",
                edgecolor="#333333", linewidth=0.6)
    for i, (n, p) in enumerate(zip(df.n.values, df.pct.values)):
        ax.text(n + tot * 0.012, i, f"{n}  ({p:.1f}%)", va="center", fontsize=7)
    ax.set_xlabel("Images")
    ax.set_xlim(0, df.n.max() * 1.28)
    ax.set_title(f"D3  Source contribution to the canonical population (N = {tot})", fontsize=9)
    save(fig, MAIN, "D3_source_population")
    sd(df[["source", "n", "pct"]], "D3_source_population")


# ================================================================ PART C - PERFORMANCE
def r1_fig():
    models = ["A_PRIMARY", "B_RGB", "B_EMBEDDING_ONLY", "C_PRIMARY", "D_VESSEL_MAP",
              "E_RGB_VESSEL_FEATURE_FUSION", "G_RGB_VESSEL_SCALAR_FUSION"]
    vals = [canon_metrics(m)["multiclass_auc"] for m in models]
    df = pd.DataFrame({"model": models, "label": [LABEL[m] for m in models], "auc": vals})
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    y = np.arange(len(df))[::-1]
    ax.hlines(y, 0.60, df.auc.values, color="#c8c8c8", lw=1.0)
    ax.plot(df.auc.values, y, "o", ms=7, color="#1b1b1b")
    for yy, v in zip(y, df.auc.values):
        ax.text(v + 0.004, yy, f"{v:.6f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(df.label)
    ax.set_xlim(0.60, max(df.auc) + 0.018)
    ax.set_xlabel("Multiclass macro OVR ROC-AUC (canonical TEST, N = 1,331)")
    ax.set_title("R1  Canonical test discrimination by model", fontsize=9)
    save(fig, MAIN, "R1_model_auc_comparison")
    sd(df, "R1_model_auc_comparison")


def r2_fig():
    models = ["A_PRIMARY", "B_RGB", "B_EMBEDDING_ONLY", "C_PRIMARY", "D_VESSEL_MAP",
              "E_RGB_VESSEL_FEATURE_FUSION", "G_RGB_VESSEL_SCALAR_FUSION"]
    M = pd.DataFrame([{**canon_metrics(m), "model": m} for m in models])
    fig, axes = plt.subplots(1, 5, figsize=(13.2, 3.1))
    for ax, k in zip(axes, MET):
        v = M[k].values
        y = np.arange(len(models))[::-1]
        ax.hlines(y, min(v) * 0.98 if k in ("brier", "ece") else 0, v, color="#c8c8c8", lw=1.0)
        ax.plot(v, y, "o", ms=6, color="#1b1b1b")
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[m] for m in models], fontsize=7)
        ax.set_title(f"{PRETTY[k]} {'(higher better)' if DIRN[k]=='up' else '(lower better)'}",
                     fontsize=8)
        if k in ("brier", "ece"):
            ax.set_xlim(min(v) * 0.95, max(v) * 1.03)
        else:
            ax.set_xlim(0, max(v) * 1.18)
        for yy, vv in zip(y, v):
            ax.text(vv, yy + 0.28, f"{vv:.3f}", fontsize=6, ha="center")
    fig.suptitle("R2  Multi-metric comparison on canonical TEST (N = 1,331)", fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, MAIN, "R2_model_metric_panels")
    sd(M, "R2_model_metric_panels")


def r3_fig():
    models = ["B_EMBEDDING_ONLY", "C_PRIMARY", "D_VESSEL_MAP", "E_RGB_VESSEL_FEATURE_FUSION",
              "G_RGB_VESSEL_SCALAR_FUSION"]
    cols = ["auc_Normal", "auc_Pre_Plus", "auc_Plus"]
    M = pd.DataFrame([{**canon_metrics(m), "model": m} for m in models])
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, c, nm in zip(axes, cols, ["Normal", "Pre-Plus", "Plus"]):
        v = M[c].values
        x = np.arange(len(models))
        ax.bar(x, v, color=["#c8c8c8", "#a6a6a6", "#7a7a7a", "#4d4d4d", "#1b1b1b"],
               edgecolor="#333333", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in models], fontsize=7)
        ax.set_ylim(0, 1.06)
        ax.set_ylabel("AUC" if nm == "Normal" else "")
        ax.set_title(f"{nm} (one-vs-rest)", fontsize=9)
        for xx, vv in zip(x, v):
            ax.text(xx, vv + 0.015, f"{vv:.3f}", ha="center", fontsize=6.5)
    fig.suptitle("R3  Per-class AUC on canonical TEST", fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, MAIN, "R3_per_class_auc")
    sd(M[["model"] + cols], "R3_per_class_auc")


# ================================================================ PART D - CENTRAL RESULT
def forest(ax, rows, xlabel, title, annot=True):
    y = np.arange(len(rows))[::-1]
    for yy, r in zip(y, rows):
        c = "#1b1b1b" if r["lo"] > 0 or r["hi"] < 0 else "#a6a6a6"
        ax.plot([r["lo"], r["hi"]], [yy, yy], color=c, lw=1.4)
        ax.plot([r["delta"]], [yy], "o", ms=6, color=c)
        if annot:
            ax.text(r["hi"] + 0.0012, yy, f"{r['delta']:+.6f}  [{r['lo']:+.6f}, {r['hi']:+.6f}]",
                    va="center", fontsize=6.5)
    ax.axvline(0, color="#333333", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9)


def c1():
    cb = prow(t7p, metric="multiclass_auc")
    eb = prow(t8bp, comparison="E_minus_B", metric="multiclass_auc")
    ge = prow(t8bp, comparison="G_minus_E", metric="multiclass_auc")
    rows = [{"label": "C - RGB embedding\n(+ 5 scalar biomarkers)",
             "delta": float(cb.delta), "lo": float(cb.ci95_lo), "hi": float(cb.ci95_hi),
             "p": float(cb.p_two_sided_null_centered), "reps": 10000,
             "source": "task7 paired_primary_metrics.csv"},
            {"label": "E - RGB embedding\n(+ spatial vessel representation)",
             "delta": float(eb.delta), "lo": float(eb.ci95_lo), "hi": float(eb.ci95_hi),
             "p": float(eb.p_two_sided_null_centered), "reps": 10000,
             "source": "task8b paired_metrics_10k.csv"},
            {"label": "G - E\n(+ 5 scalar biomarkers after vessel)",
             "delta": float(ge.delta), "lo": float(ge.ci95_lo), "hi": float(ge.ci95_hi),
             "p": float(ge.p_two_sided_null_centered), "reps": 10000,
             "source": "task8b paired_metrics_10k.csv"}]
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    forest(ax, rows, "Delta multiclass AUC (paired class-stratified bootstrap, 10,000 replicates)",
           "C1  Incremental information on the frozen RGB representation (canonical TEST)")
    fig.tight_layout()
    save(fig, MAIN, "C1_incremental_auc_forest")
    sd(pd.DataFrame(rows), "C1_incremental_auc_forest")


def c3():
    eb = prow(t8bp, comparison="E_minus_B", metric="multiclass_auc")
    m1m0 = prow(t13p, comparison="M1_minus_M0", metric="multiclass_auc")
    rows = [{"label": "EfficientNet-B5 RGB representation\n(E - B embedding)",
             "delta": float(eb.delta), "lo": float(eb.ci95_lo), "hi": float(eb.ci95_hi),
             "p": float(eb.p_two_sided_null_centered), "reps": 10000,
             "source": "task8b paired_metrics_10k.csv"},
            {"label": "Dual-backbone attention RGB representation\n(M1 - M0)",
             "delta": float(m1m0.delta), "lo": float(m1m0.ci95_lo), "hi": float(m1m0.ci95_hi),
             "p": float(m1m0.p_two_sided_null_centered), "reps": 10000,
             "source": "task13 paired_all_10k.csv"}]
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    forest(ax, rows, "Delta multiclass AUC from adding the spatial vessel representation",
           "C3  Vessel complementarity across distinct RGB feature extractors (canonical TEST)")
    fig.tight_layout()
    save(fig, MAIN, "C3_vessel_complementarity_across_rgb_architectures")
    sd(pd.DataFrame(rows), "C3_vessel_complementarity_across_rgb_architectures")


def c2():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.axis("off")

    def box(x, y, w, h, t, fc, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.014",
                                    fc=fc, ec="#333333", lw=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.4)
    box(0.05, 0.62, 0.26, 0.26, "RGB embedding\nAUC 0.924903", "#e8e8e8")
    box(0.38, 0.62, 0.24, 0.26, "C  RGB + biomarkers\nAUC 0.925988", "#f2f2f2")
    box(0.38, 0.16, 0.24, 0.26, "E  RGB + vessel\nAUC 0.932841", "#dcdcdc")
    box(0.70, 0.16, 0.26, 0.26, "G  RGB + vessel\n+ biomarkers\nAUC 0.934880", "#dcdcdc")
    ax.add_patch(FancyArrowPatch((0.31, 0.75), (0.38, 0.75), arrowstyle="-|>", lw=1.0,
                                 color="#333333", mutation_scale=11))
    ax.add_patch(FancyArrowPatch((0.31, 0.68), (0.38, 0.29), arrowstyle="-|>", lw=1.0,
                                 color="#333333", mutation_scale=11))
    ax.add_patch(FancyArrowPatch((0.62, 0.29), (0.70, 0.29), arrowstyle="-|>", lw=1.0,
                                 color="#333333", mutation_scale=11))
    ax.text(0.345, 0.80, "+ 5 biomarkers\nD = +0.001085\n[-0.001691, +0.003868]\np = 0.431\n"
                         "no supported increment", fontsize=6.6, ha="center", va="bottom")
    ax.text(0.335, 0.50, "+ spatial vessel\nD = +0.007938\n[+0.002630, +0.013131]\np = 0.0031\n"
                         "supported increment", fontsize=6.6, ha="center", va="center")
    ax.text(0.66, 0.36, "+ 5 biomarkers\nD = +0.002039\n[-0.000274, +0.004397]\np = 0.090\n"
                         "no supported increment", fontsize=6.6, ha="center", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.0)
    ax.set_title("C2  Complementarity summary: what each representation adds to the RGB embedding",
                 fontsize=9, pad=4)
    save(fig, MAIN, "C2_complementarity_summary")
    sd(pd.DataFrame([
        {"step": "RGB embedding", "adds": "-", "auc": 0.924903, "delta_auc": np.nan,
         "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan},
        {"step": "C  RGB + biomarkers", "adds": "5 scalar biomarkers", "auc": 0.925988,
         "delta_auc": 0.001085, "ci_lo": -0.001691, "ci_hi": 0.003868, "p": 0.4306},
        {"step": "E  RGB + vessel", "adds": "spatial vessel representation", "auc": 0.932841,
         "delta_auc": 0.007938, "ci_lo": 0.002630, "ci_hi": 0.013131, "p": 0.0031},
        {"step": "G  RGB + vessel + biomarkers", "adds": "5 scalar biomarkers after vessel",
         "auc": 0.934880, "delta_auc": 0.002039, "ci_lo": -0.000274, "ci_hi": 0.004397,
         "p": 0.0900}]), "C2_complementarity_summary")


# ================================================================ PART E - CONFUSION / CALIB
def p1():
    spec = [("A_PRIMARY", A / "task6/confusion_A_PRIMARY.csv"),
            ("B_EMBEDDING_ONLY", A / "task6/confusion_B_EMBEDDING_ONLY.csv"),
            ("E_RGB_VESSEL_FEATURE_FUSION",
             A / "task8_spatial_vessel_fusion/confusion_E_RGB_VESSEL_FEATURE_FUSION.csv"),
            ("G_RGB_VESSEL_SCALAR_FUSION",
             A / "task8_spatial_vessel_fusion/confusion_G_RGB_VESSEL_SCALAR_FUSION.csv")]
    mats, rows = {}, []
    for nm, p in spec:
        c = pd.read_csv(p, header=None).values.astype(float)
        mats[nm] = c
        for i in range(3):
            for j in range(3):
                rows.append({"model": nm, "true_class": ["Normal", "Pre-Plus", "Plus"][i],
                             "predicted_class": ["Normal", "Pre-Plus", "Plus"][j],
                             "count": int(c[i, j]),
                             "row_fraction": float(c[i, j] / c[i].sum()) if c[i].sum() else np.nan})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    names = ["Normal", "Pre-Plus", "Plus"]
    for ax, (nm, _p) in zip(axes.ravel(), spec):
        c = mats[nm]
        n = c / np.clip(c.sum(1, keepdims=True), 1e-9, None)
        ax.imshow(n, cmap="Greys", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{n[i,j]*100:.1f}%\n({int(c[i,j])})", ha="center", va="center",
                        fontsize=6.5, color="white" if n[i, j] > 0.55 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=7, rotation=20)
        ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("predicted", fontsize=7); ax.set_ylabel("true", fontsize=7)
        ax.set_title(f"{LABEL[nm]}", fontsize=8.5)
    fig.suptitle("P1  Row-normalised confusion matrices, canonical TEST (N = 1,331)", fontsize=10,
                 y=0.99)
    fig.tight_layout()
    save(fig, MAIN, "P1_confusion_matrices_main")
    for nm, _p in spec:
        c = mats[nm]
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        n = c / np.clip(c.sum(1, keepdims=True), 1e-9, None)
        ax.imshow(n, cmap="Greys", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{n[i,j]*100:.1f}%\n({int(c[i,j])})", ha="center", va="center",
                        fontsize=7, color="white" if n[i, j] > 0.55 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=7, rotation=20)
        ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=7)
        ax.set_title(LABEL[nm], fontsize=8)
        fig.tight_layout()
        save(fig, MAIN, f"P1_confusion_{SHORT[nm].replace('-','_')}")
    sd(pd.DataFrame(rows), "P1_confusion_matrices_main")


def p2():
    srcs = [("B_EMBEDDING_ONLY", A / "task6/test_predictions_B_EMBEDDING_ONLY.csv"),
            ("E_RGB_VESSEL_FEATURE_FUSION",
             A / "task8_spatial_vessel_fusion/test_predictions_E_RGB_VESSEL_FEATURE_FUSION.csv"),
            ("G_RGB_VESSEL_SCALAR_FUSION",
             A / "task8_spatial_vessel_fusion/test_predictions_G_RGB_VESSEL_SCALAR_FUSION.csv")]
    pc = ["prob_Normal", "prob_Pre_Plus", "prob_Plus"]
    rows = []
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    for nm, p in srcs:
        d = pd.read_csv(p)
        P = d[pc].values
        y = d.true_label.values
        conf, corr = P.max(1), (P.argmax(1) == y)
        ece = 0.0
        for lo in np.linspace(0, 1, 16)[:-1]:
            m = (conf >= lo) & (conf < lo + 1 / 15)
            if m.sum():
                ece += m.mean() * abs(corr[m].mean() - conf[m].mean())
                rows.append({"model": nm, "bin_lo": round(lo, 4), "bin_hi": round(lo + 1 / 15, 4),
                             "n": int(m.sum()), "mean_confidence": float(conf[m].mean()),
                             "accuracy": float(corr[m].mean()),
                             "gap": float(corr[m].mean() - conf[m].mean())})
        xs = [r["mean_confidence"] for r in rows if r["model"] == nm]
        ys = [r["accuracy"] for r in rows if r["model"] == nm]
        ax.plot(xs, ys, "o-", ms=4, lw=1.0, label=f"{LABEL[nm]}  (ECE {ece:.4f})")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("P2  Reliability, 15 equal-width bins, canonical TEST", fontsize=9)
    save(fig, MAIN, "P2_reliability_main")
    sd(pd.DataFrame(rows), "P2_reliability_main")


# ================================================================ PART F - BIOMARKER STORY
def b1():
    cb = prow(t7p, metric="multiclass_auc")
    ge = prow(t8bp, comparison="G_minus_E", metric="multiclass_auc")
    ih = prow(t9p, comparison="I_minus_H", metric="multiclass_auc")
    j1j0 = prow(t10p, comparison="J1_minus_J0", metric="multiclass_auc")
    fe = prow(t11p, comparison="G_minus_E_farfum_rop", metric="multiclass_auc")
    fa = prow(t11p, comparison="G_minus_E_farabi", metric="multiclass_auc")
    fp = prow(t11p, comparison="G_minus_E_plus", metric="restricted_binary_auc")
    rows = [{"label": "Canonical:  C - B embedding", "delta": float(cb.delta),
             "lo": float(cb.ci95_lo), "hi": float(cb.ci95_hi),
             "p": float(cb.p_two_sided_null_centered), "reps": 10000,
             "analysis": "canonical", "source": "task7"},
            {"label": "Canonical:  G - E", "delta": float(ge.delta), "lo": float(ge.ci95_lo),
             "hi": float(ge.ci95_hi), "p": float(ge.p_two_sided_null_centered), "reps": 10000,
             "analysis": "canonical", "source": "task8b"},
            {"label": "Joint model:  I - H", "delta": float(ih.delta), "lo": float(ih.ci95_lo),
             "hi": float(ih.ci95_hi), "p": float(ih.p_two_sided_null_centered), "reps": 10000,
             "analysis": "joint model", "source": "task9"},
            {"label": "FiLM:  J1 - J0", "delta": float(j1j0.delta), "lo": float(j1j0.ci95_lo),
             "hi": float(j1j0.ci95_hi), "p": float(j1j0.p_two_sided_null_centered), "reps": 10000,
             "analysis": "FiLM", "source": "task10"},
            {"label": "Held-out FARFUM-RoP:  G - E", "delta": float(fe.delta),
             "lo": float(fe.ci95_lo), "hi": float(fe.ci95_hi),
             "p": float(fe.p_two_sided_null_centered), "reps": 10000,
             "analysis": "held-out FARFUM-RoP", "source": "task11"},
            {"label": "Held-out Farabi:  G - E", "delta": float(fa.delta),
             "lo": float(fa.ci95_lo), "hi": float(fa.ci95_hi),
             "p": float(fa.p_two_sided_null_centered), "reps": 10000,
             "analysis": "held-out Farabi", "source": "task11"},
            {"label": "Held-out Plus (restricted):  G - E", "delta": float(fp.delta),
             "lo": float(fp.ci95_lo), "hi": float(fp.ci95_hi),
             "p": float(fp.p_two_sided_null_centered), "reps": 10000,
             "analysis": "held-out Plus (restricted binary)", "source": "task11"}]
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    forest(ax, rows,
           "Delta AUC (paired class-stratified bootstrap, 10,000 replicates, seed 42)",
           "B1  Six independent tests of the five scalar biomarkers")
    fig.tight_layout()
    save(fig, MAIN, "B1_biomarker_incremental_auc_forest")
    sd(pd.DataFrame(rows), "B1_biomarker_incremental_auc_forest")


def b2():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    ax = axes[0]
    f = FILM[FILM.split == "test"].iloc[0]
    for i, (nm, mean, p05, p50, p95, ref) in enumerate([
            ("gamma", float(f.gamma_mean), float(f.gamma_p05), float(f.gamma_p50),
             float(f.gamma_p95), 1.0),
            ("beta", float(f.beta_mean), float(f.beta_p05), float(f.beta_p50),
             float(f.beta_p95), 0.0)]):
        ax.hlines(i, p05, p95, color="#1b1b1b", lw=2.2)
        ax.plot([p50], [i], "o", ms=7, color="#1b1b1b")
        ax.plot([mean], [i], "D", ms=5, color="#7a7a7a")
        ax.text(p95, i + 0.16, f"mean {mean:.6f} | p50 {p50:.6f} | p05 {p05:.6f} | p95 {p95:.6f}",
                fontsize=6.4, ha="right")
        ax.axvline(ref, color="#333333", ls="--", lw=0.8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["gamma\n(identity = 1)", "beta\n(identity = 0)"],
                                              fontsize=7.5)
    ax.set_xlabel("FiLM modulation value (TEST)")
    ax.set_title("B2a  Learned FiLM modulation is near identity", fontsize=8.5)
    ax = axes[1]
    s = SENS.copy()
    x = np.arange(len(s))
    ax.bar(x - 0.19, s.mean_abs_prob_change.values, 0.38, label="mean", color="#1b1b1b",
           edgecolor="#333333", linewidth=0.6)
    ax.bar(x + 0.19, s.p95_abs_prob_change.values, 0.38, label="p95", color="#c8c8c8",
           edgecolor="#333333", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in s.biomarker], fontsize=6.4)
    ax.set_ylabel("absolute probability change")
    ax.legend(frameon=False)
    ax.set_title("B2b  Single-biomarker ablation sensitivity of frozen J1", fontsize=8.5)
    fig.suptitle("B2  FiLM diagnostics: biomarker conditioning had no measurable effect", fontsize=9.5,
                 y=1.03)
    fig.tight_layout()
    save(fig, APP, "B2_film_diagnostics")
    sd(FILM.assign(value_mean__=FILM.gamma_mean), "B2_film_modulation")
    sd(SENS, "B2_biomarker_sensitivity")


# ================================================================ PART G - OVERFIT / NEGATIVE
def n1():
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2))
    for h, nm, sel in ((Hhist, "H  Joint RGB-vessel", 4), (Ihist, "I  Joint RGB-vessel + biomarkers", 6)):
        axes[0].plot(h.epoch, h.train_loss, "o-", ms=3, lw=1.0, label=f"{nm} train")
        axes[0].plot(h.epoch, h.val_loss, "s--", ms=3, lw=1.0, label=f"{nm} val")
        axes[1].plot(h.epoch, h.val_auc, "o-", ms=3, lw=1.0, label=nm)
        axes[1].axvline(sel, color="#a6a6a6", ls=":", lw=0.9)
    axes[0].axvline(3.5, color="#333333", ls=":", lw=0.9)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("N1a  Loss trajectories (encoder unfrozen at 3.5)", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=5.6)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation multiclass AUC")
    axes[1].set_title("N1b  Validation AUC; dotted = selected epoch", fontsize=8.5)
    axes[1].legend(frameon=False, fontsize=6)
    m = pd.DataFrame([{**canon_metrics(x), "model": x} for x in
                      ("H_JOINT_RGB_VESSEL", "I_JOINT_RGB_VESSEL_BIOMARKER")])
    x = np.arange(2)
    axes[2].bar(x - 0.2, m.multiclass_auc.values, 0.38, label="AUC", color="#1b1b1b",
                edgecolor="#333333", linewidth=0.6)
    axes[2].bar(x + 0.2, m.macro_f1.values, 0.38, label="Macro F1", color="#c8c8c8",
                edgecolor="#333333", linewidth=0.6)
    axes[2].set_xticks(x); axes[2].set_xticklabels(["H", "I"])
    axes[2].set_ylim(0, 1.05); axes[2].legend(frameon=False)
    axes[2].set_title("N1c  Combined canonical TEST summary", fontsize=8.5)
    fig.suptitle("N1  Task 9 learning curves: overfitting after encoder unfreezing", fontsize=9.5,
                 y=1.03)
    fig.tight_layout()
    save(fig, APP, "N1_task9_learning_curves")
    sd(pd.concat([Hhist.assign(model="H_JOINT_RGB_VESSEL"),
                  Ihist.assign(model="I_JOINT_RGB_VESSEL_BIOMARKER")]),
       "N1_task9_learning_curves")


def n2():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for h, nm, sel in ((J0hist, "J0  Frozen fusion control", 1),
                       (J1hist, "J1  Biomarker-conditioned FiLM", 1)):
        axes[0].plot(h.epoch, h.train_loss, "o-", ms=3, lw=1.0, label=f"{nm} train")
        axes[0].plot(h.epoch, h.val_loss, "s--", ms=3, lw=1.0, label=f"{nm} val")
        axes[1].plot(h.epoch, h.val_auc, "o-", ms=3, lw=1.0, label=nm)
        axes[1].axvline(sel, color="#a6a6a6", ls=":", lw=0.9)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("N2a  Loss trajectories", fontsize=8.5); axes[0].legend(frameon=False, fontsize=5.8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation multiclass AUC")
    axes[1].set_title("N2b  Validation AUC; dotted = selected epoch", fontsize=8.5)
    axes[1].legend(frameon=False, fontsize=6)
    fig.suptitle("N2  Task 10 learning curves (frozen embeddings, head-only training)", fontsize=9.5,
                 y=1.03)
    fig.tight_layout()
    save(fig, APP, "N2_task10_learning_curves")
    sd(pd.concat([J0hist.assign(model="J0_FROZEN_FUSION_CONTROL"),
                  J1hist.assign(model="J1_BIOMARKER_FILM")]), "N2_task10_learning_curves")


def n3():
    h = Lhist
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    axes[0].plot(h.epoch, h.train_loss, "o-", ms=3, lw=1.0, label="train loss")
    axes[0].plot(h.epoch, h.val_loss, "s--", ms=3, lw=1.0, label="val loss")
    axes[0].axvline(3.5, color="#333333", ls=":", lw=0.9)
    axes[0].axvline(Lsel["selected_epoch"], color="#a6a6a6", ls="--", lw=0.9)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("N3a  Loss (backbones unfrozen at 3.5; dashed = selected epoch 2)", fontsize=8.5)
    axes[0].legend(frameon=False)
    axes[1].plot(h.epoch, h.val_auc, "o-", ms=3, lw=1.0, label="val AUC")
    axes[1].plot(h.epoch, h.val_macro_f1, "s-", ms=3, lw=1.0, label="val macro F1")
    axes[1].axvline(Lsel["selected_epoch"], color="#a6a6a6", ls="--", lw=0.9)
    axes[1].axvline(7, color="#333333", ls=":", lw=0.9)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation metric")
    axes[1].set_title("N3b  Validation metrics (dotted = min val-loss epoch 7)", fontsize=8.5)
    axes[1].legend(frameon=False)
    fig.suptitle("N3  Task 13 learning curves (ROPDeepX-style dual-RGB attention)",
                 fontsize=9.5, y=1.03)
    fig.tight_layout()
    save(fig, APP, "N3_task13_learning_curves")
    sd(h, "N3_task13_learning_curves")
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    ax.plot(h.epoch, h.attn_resnet_mean, "o-", ms=3, lw=1.0, label="mean ResNet attention")
    ax.plot(h.epoch, h.attn_effnet_mean, "s-", ms=3, lw=1.0, label="mean EfficientNet attention")
    ax.axhline(0.95, color="#333333", ls="--", lw=0.8)
    ax.axhline(0.05, color="#333333", ls="--", lw=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("mean attention weight")
    ax.set_title("N3c  Attention weights across epochs (no collapse)", fontsize=9)
    ax.legend(frameon=False)
    save(fig, APP, "N3c_task13_attention_epochs")
    sd(h[["epoch", "attn_resnet_mean", "attn_effnet_mean"]], "N3c_task13_attention_epochs")


# ================================================================ PART H - SOURCE HOLDOUT
def s1():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2),
                             gridspec_kw={"width_ratios": [2.0, 1.0]})
    hs = ["farfum_rop", "farabi"]
    mods = ["B_LOSO", "E_LOSO", "G_LOSO"]
    x = np.arange(len(hs))
    w = 0.25
    rows = []
    for j, (m, fc) in enumerate(zip(mods, ["#c8c8c8", "#7a7a7a", "#1b1b1b"])):
        v = [float(prow(t11m, held_out_source=h, model=m).multiclass_auc) for h in hs]
        axes[0].bar(x + (j - 1) * w, v, w, label=LABEL[m], color=fc, edgecolor="#333333",
                    linewidth=0.6)
        for xx, vv in zip(x + (j - 1) * w, v):
            axes[0].text(xx, vv + 0.012, f"{vv:.4f}", ha="center", fontsize=6.4)
        for h, vv in zip(hs, v):
            rows.append({"panel": "3-class AUC", "held_out_source": h, "model": m, "value": vv})
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([h.replace("_", "-") for h in hs])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("3-class macro OVR AUC")
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[0].set_title("S1a  Held-out source, 3-class AUC", fontsize=8.5)
    v = [float(prow(t11m, held_out_source="plus", model=m).restricted_binary_auc) for m in mods]
    axes[1].bar(np.arange(3), v, color=["#c8c8c8", "#7a7a7a", "#1b1b1b"], edgecolor="#333333",
                linewidth=0.6)
    for xx, vv in zip(np.arange(3), v):
        axes[1].text(xx, vv + 0.004, f"{vv:.4f}", ha="center", fontsize=6.4)
    axes[1].set_xticks(np.arange(3)); axes[1].set_xticklabels(["B", "E", "G"])
    axes[1].set_ylim(0.95, 1.0)
    axes[1].set_title("S1b  Held-out Plus,\nrestricted Normal-vs-Plus AUC", fontsize=8.5)
    for m, vv in zip(mods, v):
        rows.append({"panel": "restricted binary AUC", "held_out_source": "plus", "model": m,
                     "value": vv})
    fig.suptitle("S1  Source-held-out performance (no held-out data used in fitting)", fontsize=9.5,
                 y=1.04)
    fig.tight_layout()
    save(fig, MAIN, "S1_source_heldout_performance")
    sd(pd.DataFrame(rows), "S1_source_heldout_performance")


def s2():
    fe = prow(t11p, comparison="E_minus_B_farfum_rop", metric="multiclass_auc")
    fa = prow(t11p, comparison="E_minus_B_farabi", metric="multiclass_auc")
    fp = prow(t11p, comparison="E_minus_B_plus", metric="restricted_binary_auc")
    rows = [{"label": "Held-out FARFUM-RoP\n(3-class AUC)", "delta": float(fe.delta),
             "lo": float(fe.ci95_lo), "hi": float(fe.ci95_hi),
             "p": float(fe.p_two_sided_null_centered), "reps": 10000},
            {"label": "Held-out Farabi\n(3-class AUC)", "delta": float(fa.delta),
             "lo": float(fa.ci95_lo), "hi": float(fa.ci95_hi),
             "p": float(fa.p_two_sided_null_centered), "reps": 10000},
            {"label": "Held-out Plus\n(restricted Normal-vs-Plus AUC)", "delta": float(fp.delta),
             "lo": float(fp.ci95_lo), "hi": float(fp.ci95_hi),
             "p": float(fp.p_two_sided_null_centered), "reps": 10000}]
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    forest(ax, rows, "Delta AUC from adding the spatial vessel representation (E - B)",
           "S2  Vessel complementarity under source-held-out evaluation")
    fig.tight_layout()
    save(fig, MAIN, "S2_vessel_complementarity_heldout")
    sd(pd.DataFrame(rows), "S2_vessel_complementarity_heldout")


def s3():
    fe = prow(t11p, comparison="G_minus_E_farfum_rop", metric="multiclass_auc")
    fa = prow(t11p, comparison="G_minus_E_farabi", metric="multiclass_auc")
    fp = prow(t11p, comparison="G_minus_E_plus", metric="restricted_binary_auc")
    rows = [{"label": "Held-out FARFUM-RoP\n(3-class AUC)", "delta": float(fe.delta),
             "lo": float(fe.ci95_lo), "hi": float(fe.ci95_hi),
             "p": float(fe.p_two_sided_null_centered), "reps": 10000},
            {"label": "Held-out Farabi\n(3-class AUC)", "delta": float(fa.delta),
             "lo": float(fa.ci95_lo), "hi": float(fa.ci95_hi),
             "p": float(fa.p_two_sided_null_centered), "reps": 10000},
            {"label": "Held-out Plus\n(restricted Normal-vs-Plus AUC)", "delta": float(fp.delta),
             "lo": float(fp.ci95_lo), "hi": float(fp.ci95_hi),
             "p": float(fp.p_two_sided_null_centered), "reps": 10000}]
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    forest(ax, rows,
           "Delta AUC from adding the five scalar biomarkers after the vessel representation (G - E)",
           "S3  Biomarker increment under source-held-out evaluation")
    fig.tight_layout()
    save(fig, APP, "S3_biomarker_heldout_forest")
    sd(pd.DataFrame(rows), "S3_biomarker_heldout_forest")


# ================================================================ PART I - DOMAIN SHIFT
def ds1():
    s = t11s.copy()
    srcs = ["plus", "farfum_rop", "farabi"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    rows = []
    for ax, blk, nm in ((axes[0], "rgb_embedding", "RGB embedding (2048-d)"),
                        (axes[1], "vessel_embedding", "Vessel embedding (1792-d)")):
        x = np.arange(len(srcs))
        w = 0.26
        for j, (c, lab, fc) in enumerate([("median_abs_smd", "median |SMD|", "#c8c8c8"),
                                          ("p90_abs_smd", "p90 |SMD|", "#7a7a7a"),
                                          ("p95_abs_smd", "p95 |SMD|", "#1b1b1b")]):
            v = [float(prow(s, held_out_source=q, block=blk)[c]) for q in srcs]
            ax.bar(x + (j - 1) * w, v, w, label=lab, color=fc, edgecolor="#333333", linewidth=0.6)
            for xx, vv in zip(x + (j - 1) * w, v):
                ax.text(xx, vv + 0.03, f"{vv:.3f}", ha="center", fontsize=6.2)
            for q, vv in zip(srcs, v):
                rows.append({"block": blk, "held_out_source": q, "statistic": c, "value": vv})
        ax.set_xticks(x)
        ax.set_xticklabels([q.replace("_", "-") for q in srcs])
        ax.set_ylabel("standardized mean difference")
        ax.set_title(nm, fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("DS1  Representation domain shift, held-out source vs training sources", fontsize=9.5,
                 y=1.04)
    fig.tight_layout()
    save(fig, MAIN, "DS1_representation_domain_shift")
    sd(pd.DataFrame(rows), "DS1_representation_domain_shift")


def ds2():
    s = t11s.copy()
    srcs = ["plus", "farfum_rop", "farabi"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(len(srcs))
    w = 0.36
    for j, (blk, lab, fc) in enumerate([("rgb_embedding", "RGB embedding", "#c8c8c8"),
                                        ("vessel_embedding", "Vessel embedding", "#1b1b1b")]):
        v = [float(prow(s, held_out_source=q, block=blk).centroid_distance_standardized)
             for q in srcs]
        ax.bar(x + (j - 0.5) * w, v, w, label=lab, color=fc, edgecolor="#333333", linewidth=0.6)
        for xx, vv in zip(x + (j - 0.5) * w, v):
            ax.text(xx, vv + 1.1, f"{vv:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace("_", "-") for q in srcs])
    ax.set_ylabel("standardized centroid distance")
    ax.legend(frameon=False)
    ax.set_title("DS2  Standardized centroid distance between held-out and training sources",
                 fontsize=9)
    save(fig, MAIN, "DS2_centroid_shift")
    sd(pd.DataFrame([{"held_out_source": q, "block": b,
                      "centroid_distance_standardized":
                          float(prow(s, held_out_source=q, block=b).centroid_distance_standardized)}
                     for q in srcs for b in ("rgb_embedding", "vessel_embedding")]),
       "DS2_centroid_shift")


def ds3():
    b = t11b[t11b.source.str.startswith("HELDOUT::")].copy()
    b["held_out"] = b.source.str.replace("HELDOUT::", "", regex=False)
    piv = b.pivot_table(index="biomarker", columns="held_out", values="smd_vs_train")
    piv = piv[["plus", "farfum_rop", "farabi"]]
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    vmax = float(np.nanmax(np.abs(piv.values)))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:+.3f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(piv.values[i, j]) > 0.6 * vmax else "black")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([c.replace("_", "-") for c in piv.columns], fontsize=7.5)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.036)
    cb.set_label("held-out vs training SMD", fontsize=7)
    ax.set_title("DS3  Scalar biomarker source shift", fontsize=9)
    save(fig, MAIN, "DS3_biomarker_smd_heatmap")
    sd(piv.reset_index().melt(id_vars="biomarker", var_name="held_out_source",
                              value_name="smd_vs_train"), "DS3_biomarker_smd_heatmap")


# ================================================================ PART J - TASK 12
def da1():
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2),
                             gridspec_kw={"width_ratios": [2.0, 1.0]})
    hs = ["farfum_rop", "farabi"]
    kinds = ["K0", "K1"]
    rows = []
    x = np.arange(len(hs))
    w = 0.36
    for j, (k, fc) in enumerate(zip(kinds, ["#c8c8c8", "#1b1b1b"])):
        v = [float(prow(t12m, held_out_source=h, model=k).multiclass_auc) for h in hs]
        axes[0].bar(x + (j - 0.5) * w, v, w, label=LABEL["K0_DOMAIN_NEUTRAL_CONTROL" if k == "K0"
                                                         else "K1_CC_DANN_MMD"],
                    color=fc, edgecolor="#333333", linewidth=0.6)
        for xx, vv in zip(x + (j - 0.5) * w, v):
            axes[0].text(xx, vv + 0.012, f"{vv:.4f}", ha="center", fontsize=6.4)
        for h, vv in zip(hs, v):
            rows.append({"panel": "3-class AUC", "held_out_source": h, "model": k, "value": vv})
    axes[0].set_xticks(x); axes[0].set_xticklabels([h.replace("_", "-") for h in hs])
    axes[0].set_ylim(0, 1.0); axes[0].set_ylabel("3-class macro OVR AUC")
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[0].set_title("DA1a  Held-out source, K0 vs K1", fontsize=8.5)
    v = [float(prow(t12m, held_out_source="plus", model=k).restricted_binary_auc)
         for k in kinds]
    axes[1].bar(np.arange(2), v, color=["#c8c8c8", "#1b1b1b"], edgecolor="#333333", linewidth=0.6)
    for xx, vv in zip(np.arange(2), v):
        axes[1].text(xx, vv + 0.002, f"{vv:.4f}", ha="center", fontsize=6.4)
    axes[1].set_xticks(np.arange(2)); axes[1].set_xticklabels(["K0", "K1"])
    axes[1].set_ylim(0.98, 0.995)
    axes[1].set_title("DA1b  Held-out Plus,\nrestricted Normal-vs-Plus AUC", fontsize=8.5)
    for k, vv in zip(kinds, v):
        rows.append({"panel": "restricted binary AUC", "held_out_source": "plus", "model": k,
                     "value": vv})
    fig.suptitle("DA1  Task 12 domain-alignment performance (target source never used in training)",
                 fontsize=9.5, y=1.04)
    fig.tight_layout()
    save(fig, MAIN, "DA1_domain_alignment_performance")
    sd(pd.DataFrame(rows), "DA1_domain_alignment_performance")


def da2():
    s = t12s.copy()
    folds = ["plus", "farfum_rop", "farabi"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    rows = []
    x = np.arange(len(folds))
    w = 0.36
    for ax, col, nm in ((axes[0], "centroid_distance_standardized", "standardized centroid distance"),
                        (axes[1], "median_abs_smd", "median |SMD|")):
        for j, (k, fc) in enumerate(zip(["K0", "K1"], ["#c8c8c8", "#1b1b1b"])):
            v = [float(prow(s, held_out=q, kind=k)[col]) for q in folds]
            ax.bar(x + (j - 0.5) * w, v, w, label=k, color=fc, edgecolor="#333333", linewidth=0.6)
            for xx, vv in zip(x + (j - 0.5) * w, v):
                ax.text(xx, vv + max(v) * 0.02, f"{vv:.3f}", ha="center", fontsize=6.2)
            for q, vv in zip(folds, v):
                rows.append({"statistic": col, "held_out_source": q, "model": k, "value": vv})
        ax.set_xticks(x); ax.set_xticklabels([q.replace("_", "-") for q in folds])
        ax.set_ylabel(nm); ax.legend(frameon=False, fontsize=6.5)
        ax.set_title(nm, fontsize=8.5)
    fig.suptitle("DA2  Representation shift before (K0) and after (K1) class-conditional alignment",
                 fontsize=9.5, y=1.04)
    fig.tight_layout()
    save(fig, MAIN, "DA2_alignment_shift_diagnostics")
    sd(pd.DataFrame(rows), "DA2_alignment_shift_diagnostics")


def da3():
    d = t12d.copy()
    folds = ["plus", "farfum_rop", "farabi"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(folds))
    w = 0.36
    rows = []
    for j, (k, fc) in enumerate(zip(["K0", "K1"], ["#c8c8c8", "#1b1b1b"])):
        v = [float(prow(d, held_out=q, kind=k).domain_predictability_acc_cv5) for q in folds]
        ax.bar(x + (j - 0.5) * w, v, w, label=k, color=fc, edgecolor="#333333", linewidth=0.6)
        for xx, vv in zip(x + (j - 0.5) * w, v):
            ax.text(xx, vv + 0.012, f"{vv:.4f}", ha="center", fontsize=6.5)
        for q, vv in zip(folds, v):
            rows.append({"held_out_source": q, "model": k, "domain_predictability_acc_cv5": vv})
    ax.axhline(0.5, color="#333333", ls="--", lw=0.9)
    ax.text(len(folds) - 0.5, 0.515, "chance = 0.50", fontsize=6.5, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([q.replace("_", "-") for q in folds])
    ax.set_ylim(0, 1.12); ax.set_ylabel("domain-prediction accuracy (5-fold CV)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("DA3  Source identity remained highly predictable after alignment", fontsize=9)
    save(fig, MAIN, "DA3_domain_predictability")
    sd(pd.DataFrame(rows), "DA3_domain_predictability")


# ================================================================ TASK 13 EXTRAS
def r4():
    models = ["L_ROPDEEPX_STYLE_RGB", "M0_ROPDEEPX_EMBEDDING_ONLY",
              "M1_ROPDEEPX_PLUS_VESSEL", "B_EMBEDDING_ONLY",
              "E_RGB_VESSEL_FEATURE_FUSION", "G_RGB_VESSEL_SCALAR_FUSION"]
    M = pd.DataFrame([{**canon_metrics(m), "model": m} for m in models])
    fig, axes = plt.subplots(1, 5, figsize=(13.2, 3.1))
    fcs = ["#1b1b1b", "#4d4d4d", "#7a7a7a", "#c8c8c8", "#a6a6a6", "#8a8a8a"]
    for ax, k in zip(axes, MET):
        v = M[k].values
        x = np.arange(len(models))
        ax.bar(x, v, color=fcs, edgecolor="#333333", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in models], fontsize=7)
        ax.set_title(f"{PRETTY[k]} {'(higher better)' if DIRN[k]=='up' else '(lower better)'}",
                     fontsize=8)
        top = max(v)
        ax.set_ylim(0 if k not in ("brier", "ece") else min(v) * 0.9, top * 1.18)
        for xx, vv in zip(x, v):
            ax.text(xx, vv + top * 0.03, f"{vv:.3f}", ha="center", fontsize=6)
    fig.suptitle("R4  Task 13 ROPDeepX-style results against the frozen reference models "
                 "(canonical TEST, N = 1,331)", fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, MAIN, "R4_ropdeepx_style_results")
    sd(M, "R4_ropdeepx_style_results")


def r5():
    rows = []
    for tag, lab in (("M0_minus_B", "M0 - B embedding\n(dual-RGB embedding vs original RGB embedding)"),
                     ("M1_minus_M0", "M1 - M0\n(+ spatial vessel representation)"),
                     ("M1_minus_E", "M1 - E\n(dual-RGB + vessel vs original + vessel)"),
                     ("M1_minus_G", "M1 - G\n(dual-RGB + vessel vs RGB + vessel + biomarkers)")):
        r = prow(t13p, comparison=tag, metric="multiclass_auc")
        rows.append({"label": lab, "delta": float(r.delta), "lo": float(r.ci95_lo),
                     "hi": float(r.ci95_hi), "p": float(r.p_two_sided_null_centered),
                     "reps": 10000})
    fig, ax = plt.subplots(figsize=(7.8, 2.7))
    forest(ax, rows, "Delta multiclass AUC (paired, 10,000 replicates, seed 42)",
           "R5  Task 13 paired AUC comparisons")
    fig.tight_layout()
    save(fig, MAIN, "R5_task13_paired_auc")
    sd(pd.DataFrame(rows), "R5_task13_paired_auc")


def a1():
    d = t13att.copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1))
    cl = d[d.grouping == "true_class"]
    sr = d[d.grouping == "source"]
    for ax, sub, nm in ((axes[0], cl, "by true class"), (axes[1], sr, "by acquisition source")):
        x = np.arange(len(sub))
        w = 0.36
        ax.bar(x - 0.5 * w, sub.attn_resnet_mean.values, w, label="ResNet50", color="#1b1b1b",
               edgecolor="#333333", linewidth=0.6)
        ax.bar(x + 0.5 * w, sub.attn_effnet_mean.values, w, label="EfficientNet-B4", color="#c8c8c8",
               edgecolor="#333333", linewidth=0.6)
        for xx, vv in zip(x - 0.5 * w, sub.attn_resnet_mean.values):
            ax.text(xx, vv + 0.015, f"{vv:.4f}", ha="center", fontsize=6.2)
        for xx, vv in zip(x + 0.5 * w, sub.attn_effnet_mean.values):
            ax.text(xx, vv + 0.015, f"{vv:.4f}", ha="center", fontsize=6.2)
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("_", "-") for v in sub.value], fontsize=7)
        ax.set_ylim(0, 1.12); ax.set_ylabel("mean attention weight")
        ax.set_title(f"A1{'a' if nm.startswith('by true') else 'b'}  Mean attention {nm}", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("A1  Task 13 attention weights; descriptive only, not a clinical explanation",
                 fontsize=9.5, y=1.04)
    fig.tight_layout()
    save(fig, APP, "A1_attention_by_class_and_source")
    sd(d, "A1_attention_by_class_and_source")


def a2():
    st = t13attst[["split", "attn_resnet_mean", "attn_resnet_std", "attn_resnet_p05",
                   "attn_resnet_p50", "attn_resnet_p95"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0))
    for ax, split, lab in ((axes[0], "val", "validation (n=1,328)"),
                           (axes[1], "test", "test (n=1,331)")):
        r = st[st.split == split].iloc[0]
        ax.hlines(0, r.attn_resnet_p05, r.attn_resnet_p95, color="#1b1b1b", lw=2.6)
        ax.plot([r.attn_resnet_p50], [0], "o", ms=8, color="#1b1b1b", label="median")
        ax.plot([r.attn_resnet_mean], [0], "D", ms=6, color="#7a7a7a", label="mean")
        ax.plot([r.attn_resnet_p05, r.attn_resnet_p95], [0, 0], "|", ms=12, color="#1b1b1b")
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xlabel("ResNet50 attention weight")
        ax.set_title(f"A2{'a' if split=='val' else 'b'}  {lab}: mean {r.attn_resnet_mean:.4f}, "
                     f"std {r.attn_resnet_std:.4f}", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        ax.axvline(0.95, color="#333333", ls="--", lw=0.8)
        ax.text(0.945, 0.45, "collapse threshold 0.95", fontsize=6, ha="right")
    fig.suptitle("A2  Attention distribution: no collapse (p05-p95 range, median, mean)",
                 fontsize=9.5, y=1.04)
    fig.tight_layout()
    save(fig, APP, "A2_attention_distribution")
    sd(st, "A2_attention_distribution")


# ================================================================ FINAL SUMMARY
def final1():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    eb = prow(t8bp, comparison="E_minus_B", metric="multiclass_auc")
    ge = prow(t8bp, comparison="G_minus_E", metric="multiclass_auc")
    cb = prow(t7p, metric="multiclass_auc")
    rows = [{"label": "C - B embedding\n+ 5 scalar biomarkers", "delta": float(cb.delta),
             "lo": float(cb.ci95_lo), "hi": float(cb.ci95_hi),
             "p": float(cb.p_two_sided_null_centered), "reps": 10000},
            {"label": "E - B embedding\n+ spatial vessel", "delta": float(eb.delta),
             "lo": float(eb.ci95_lo), "hi": float(eb.ci95_hi),
             "p": float(eb.p_two_sided_null_centered), "reps": 10000},
            {"label": "G - E\n+ 5 biomarkers after vessel", "delta": float(ge.delta),
             "lo": float(ge.ci95_lo), "hi": float(ge.ci95_hi),
             "p": float(ge.p_two_sided_null_centered), "reps": 10000}]
    forest(axes[0], rows, "Delta multiclass AUC", "A  Canonical complementarity", annot=False)
    fe = prow(t11p, comparison="E_minus_B_farfum_rop", metric="multiclass_auc")
    fa = prow(t11p, comparison="E_minus_B_farabi", metric="multiclass_auc")
    m1m0 = prow(t13p, comparison="M1_minus_M0", metric="multiclass_auc")
    rows2 = [{"label": "E - B\nheld-out FARFUM-RoP", "delta": float(fe.delta),
              "lo": float(fe.ci95_lo), "hi": float(fe.ci95_hi),
              "p": float(fe.p_two_sided_null_centered), "reps": 10000},
             {"label": "E - B\nheld-out Farabi", "delta": float(fa.delta),
              "lo": float(fa.ci95_lo), "hi": float(fa.ci95_hi),
              "p": float(fa.p_two_sided_null_centered), "reps": 10000},
             {"label": "M1 - M0\ndual-RGB architecture", "delta": float(m1m0.delta),
              "lo": float(m1m0.ci95_lo), "hi": float(m1m0.ci95_hi),
              "p": float(m1m0.p_two_sided_null_centered), "reps": 10000}]
    forest(axes[1], rows2, "Delta multiclass AUC", "B  Vessel benefit replicates", annot=False)
    srcs = ["plus", "farfum_rop", "farabi"]
    x = np.arange(len(srcs))
    w = 0.36
    for j, (blk, lab, fc) in enumerate([("rgb_embedding", "RGB embedding", "#c8c8c8"),
                                        ("vessel_embedding", "Vessel embedding", "#1b1b1b")]):
        v = [float(prow(t11s, held_out_source=q, block=blk).median_abs_smd) for q in srcs]
        axes[2].bar(x + (j - 0.5) * w, v, w, label=lab, color=fc, edgecolor="#333333", linewidth=0.6)
        for xx, vv in zip(x + (j - 0.5) * w, v):
            axes[2].text(xx, vv + 0.03, f"{vv:.2f}", ha="center", fontsize=6.2)
    axes[2].set_xticks(x); axes[2].set_xticklabels([q.replace("_", "-") for q in srcs])
    axes[2].set_ylabel("median |SMD|")
    axes[2].legend(frameon=False, fontsize=6.5)
    axes[2].set_title("C  Representation domain shift", fontsize=9)
    fig.suptitle("FINAL1  Spatial vessel representations provide reproducible complementary "
                 "predictive information; five scalar vascular summaries show little incremental "
                 "value beyond learned image representations", fontsize=9.5, y=1.06)
    fig.tight_layout()
    save(fig, MAIN, "FINAL1_thesis_summary")
    sd(pd.DataFrame(rows + [{"label": r["label"], "delta": r["delta"], "lo": r["lo"],
                             "hi": r["hi"], "p": r["p"], "reps": r["reps"]} for r in rows2]),
       "FINAL1_thesis_summary_panels_AB")
    sd(pd.DataFrame([{"held_out_source": q, "block": b, "median_abs_smd":
                      float(prow(t11s, held_out_source=q, block=b).median_abs_smd)}
                     for q in srcs for b in ("rgb_embedding", "vessel_embedding")]),
       "FINAL1_thesis_summary_panel_C")


def main():
    log("PART A methods figures")
    for nm, fn in (("M1", m1), ("M2", m2), ("M3", m3), ("M4", m4)):
        fig_guard(nm, fn)
    log("PART B cohort figures")
    for nm, fn in (("D1", d1), ("D2", d2), ("D3", d3)):
        fig_guard(nm, fn)
    log("PART C performance figures")
    for nm, fn in (("R1", r1_fig), ("R2", r2_fig), ("R3", r3_fig), ("R4", r4), ("R5", r5)):
        fig_guard(nm, fn)
    log("PART D central result")
    for nm, fn in (("C1", c1), ("C2", c2), ("C3", c3)):
        fig_guard(nm, fn)
    log("PART E confusion / calibration")
    for nm, fn in (("P1", p1), ("P2", p2)):
        fig_guard(nm, fn)
    log("PART F biomarker story")
    for nm, fn in (("B1", b1), ("B2", b2)):
        fig_guard(nm, fn)
    log("PART G overfitting / negative")
    for nm, fn in (("N1", n1), ("N2", n2), ("N3", n3)):
        fig_guard(nm, fn)
    log("PART H source held-out")
    for nm, fn in (("S1", s1), ("S2", s2), ("S3", s3)):
        fig_guard(nm, fn)
    log("PART I domain shift")
    for nm, fn in (("DS1", ds1), ("DS2", ds2), ("DS3", ds3)):
        fig_guard(nm, fn)
    log("PART J task 12")
    for nm, fn in (("DA1", da1), ("DA2", da2), ("DA3", da3)):
        fig_guard(nm, fn)
    log("Task 13 extras + final summary")
    for nm, fn in (("A1", a1), ("A2", a2), ("FINAL1", final1)):
        fig_guard(nm, fn)
    (OUT / "metadata/figure_generation_failures.json").write_text(json.dumps(FAIL, indent=2))
    log(f"figures done, failures {len(FAIL)}")
    print(json.dumps([f["asset"] + ": " + f["error"] for f in FAIL], indent=2))


if __name__ == "__main__":
    main()
