"""Split every multi-panel report figure into independent single-chart full-width figures."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path("/root/niki_rop_task6_isolated")
A = WS / "artifacts"
OUT = WS / "report_assets_final/figures"
SRC = WS / "report_assets_final/source_data"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
W = 7.4


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=300)
    fig.savefig(folder / f"{name}.pdf")
    fig.savefig(folder / f"{name}.svg")
    plt.close(fig)


def prow(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v)
    return df[m].iloc[0]


t6m = pd.read_csv(A / "task6/metrics_summary.csv").set_index("model")
t8m = pd.read_csv(A / "task8_spatial_vessel_fusion/metrics_summary.csv").set_index("model")
t13m = pd.read_csv(A / "task13_ropdeepx_style/metrics_summary.csv").set_index("model")
t11m = pd.read_csv(A / "task11_source_heldout/metrics_per_source.csv")
t11s = pd.read_csv(A / "task11_source_heldout/domain_shift_diagnostics.csv")
t12m = pd.read_csv(A / "task12_class_conditional_domain_generalization/heldout_metrics.csv")
t12s = pd.read_csv(A / "task12_class_conditional_domain_generalization/representation_shift.csv")
t13att = pd.read_csv(A / "task13_ropdeepx_style/attention_by_class_and_source.csv")
t13attst = pd.read_csv(A / "task13_ropdeepx_style/attention_stats_val_test.csv")
FILM = pd.read_csv(A / "task10_biomarker_film/film_modulation_diagnostics.csv")
SENS = pd.read_csv(A / "task10_biomarker_film/biomarker_sensitivity.csv")
Hh = pd.DataFrame(json.loads((A / "task9_joint_multimodal_fusion/H_JOINT_RGB_VESSEL_history.json").read_text()))
Ih = pd.DataFrame(json.loads((A / "task9_joint_multimodal_fusion/I_JOINT_RGB_VESSEL_BIOMARKER_history.json").read_text()))
J0 = pd.DataFrame(json.loads((A / "task10_biomarker_film/J0_FROZEN_FUSION_CONTROL_history.json").read_text()))
J1 = pd.DataFrame(json.loads((A / "task10_biomarker_film/J1_BIOMARKER_FILM_history.json").read_text()))
Lh = pd.DataFrame(json.loads((A / "task13_ropdeepx_style/L_history.json").read_text()))

MET = ["multiclass_auc", "balanced_accuracy", "macro_f1", "brier", "ece"]
PT = {"multiclass_auc": "Multiclass macro OVR ROC-AUC",
      "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1",
      "brier": "Brier score", "ece": "Expected calibration error (15 bins)"}
DIRN = {"multiclass_auc": "higher is better", "balanced_accuracy": "higher is better",
        "macro_f1": "higher is better", "brier": "lower is better", "ece": "lower is better"}
SH = {"A_PRIMARY": "A", "B_RGB": "B", "B_EMBEDDING_ONLY": "B-emb", "C_PRIMARY": "C",
      "D_VESSEL_MAP": "D", "E_RGB_VESSEL_FEATURE_FUSION": "E",
      "G_RGB_VESSEL_SCALAR_FUSION": "G", "L_ROPDEEPX_STYLE_RGB": "L",
      "M0_ROPDEEPX_EMBEDDING_ONLY": "M0", "M1_ROPDEEPX_PLUS_VESSEL": "M1"}


def val(model, k):
    for t in (t6m, t8m, t13m):
        if model in t.index:
            return float(t.loc[model][k])
    raise KeyError(model)


# ---------------- R2 split: five single-metric dot plots
R2MODELS = ["A_PRIMARY", "B_RGB", "B_EMBEDDING_ONLY", "C_PRIMARY", "D_VESSEL_MAP",
            "E_RGB_VESSEL_FEATURE_FUSION", "G_RGB_VESSEL_SCALAR_FUSION"]
rows = []
for k in MET:
    v = [val(m, k) for m in R2MODELS]
    fig, ax = plt.subplots(figsize=(W, 3.4))
    y = np.arange(len(R2MODELS))[::-1]
    ax.hlines(y, 0 if k not in ("brier", "ece") else min(v) * 0.95, v, color="#bbbbbb", lw=1.4)
    ax.plot(v, y, "o", ms=9, color="#1b1b1b")
    for yy, vv in zip(y, v):
        ax.text(vv, yy + 0.32, f"{vv:.3f}", ha="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([SH[m] for m in R2MODELS])
    ax.set_xlim(0 if k not in ("brier", "ece") else min(v) * 0.92, max(v) * 1.16)
    ax.set_xlabel(f"{PT[k]} ({DIRN[k]})")
    ax.set_title(f"Canonical test set, N = 1331 - {PT[k]}", fontsize=11)
    fig.tight_layout()
    save(fig, OUT / "main", f"R2_{k}")
    for m, vv in zip(R2MODELS, v):
        rows.append({"model": m, "metric": k, "value": vv})
pd.DataFrame(rows).to_csv(SRC / "R2_model_metric_panels.csv", index=False)

# ---------------- R4 split: five single-metric bar plots
R4MODELS = ["L_ROPDEEPX_STYLE_RGB", "M0_ROPDEEPX_EMBEDDING_ONLY", "M1_ROPDEEPX_PLUS_VESSEL",
            "B_EMBEDDING_ONLY", "E_RGB_VESSEL_FEATURE_FUSION", "G_RGB_VESSEL_SCALAR_FUSION"]
FC = ["#1b1b1b", "#4d4d4d", "#7a7a7a", "#c8c8c8", "#a6a6a6", "#8a8a8a"]
rows = []
for k in MET:
    v = [val(m, k) for m in R4MODELS]
    fig, ax = plt.subplots(figsize=(W, 3.4))
    x = np.arange(len(R4MODELS))
    ax.bar(x, v, color=FC, edgecolor="#333333", linewidth=0.8, width=0.62)
    lo = min(v) * 0.9 if k in ("brier", "ece") else 0
    ax.set_ylim(lo, max(v) * 1.16)
    for xx, vv in zip(x, v):
        ax.text(xx, vv + max(v) * 0.03, f"{vv:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([SH[m] for m in R4MODELS])
    ax.set_ylabel(f"{PT[k]} ({DIRN[k]})")
    ax.set_title(f"Canonical test set, N = 1331 - {PT[k]}", fontsize=11)
    fig.tight_layout()
    save(fig, OUT / "main", f"R4_{k}")
    for m, vv in zip(R4MODELS, v):
        rows.append({"model": m, "metric": k, "value": vv})
pd.DataFrame(rows).to_csv(SRC / "R4_ropdeepx_style_results.csv", index=False)

# ---------------- B2 split
f = FILM[FILM.split == "test"].iloc[0]
fig, ax = plt.subplots(figsize=(W, 3.0))
for i, (nm, mean, p05, p50, p95, ref) in enumerate([
        ("gamma", float(f.gamma_mean), float(f.gamma_p05), float(f.gamma_p50), float(f.gamma_p95), 1.0),
        ("beta", float(f.beta_mean), float(f.beta_p05), float(f.beta_p50), float(f.beta_p95), 0.0)]):
    ax.hlines(i, p05, p95, color="#1b1b1b", lw=3.0)
    ax.plot([p50], [i], "o", ms=9, color="#1b1b1b", label="median" if i == 0 else None)
    ax.plot([mean], [i], "D", ms=7, color="#c05000", label="mean" if i == 0 else None)
    ax.text(p95, i + 0.20, f"p05 {p05:.6f}   median {p50:.6f}   p95 {p95:.6f}",
            fontsize=8.5, ha="right")
    ax.axvline(ref, color="#888888", ls="--", lw=1.0)
ax.set_yticks([0, 1]); ax.set_yticklabels(["gamma (identity = 1)", "beta (identity = 0)"])
ax.set_ylim(-0.6, 1.6)
ax.set_xlabel("FiLM modulation value on the test set")
ax.set_title("Task 10 - learned FiLM modulation stays near identity", fontsize=11)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
save(fig, OUT / "appendix", "B2a_film_modulation")

fig, ax = plt.subplots(figsize=(W, 3.4))
s = SENS.copy(); x = np.arange(len(s))
ax.bar(x - 0.20, s.mean_abs_prob_change.values, 0.40, label="mean", color="#1b1b1b",
       edgecolor="#333333", linewidth=0.7)
ax.bar(x + 0.20, s.p95_abs_prob_change.values, 0.40, label="p95", color="#c8c8c8",
       edgecolor="#333333", linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels([b.replace("_", "\n") for b in s.biomarker], fontsize=8.5)
ax.set_yscale("log")
ax.set_ylabel("absolute change in predicted probability")
ax.set_title("Task 10 - single-biomarker ablation sensitivity of the frozen model", fontsize=11)
ax.legend(frameon=False)
fig.tight_layout()
save(fig, OUT / "appendix", "B2b_biomarker_sensitivity")
FILM.to_csv(SRC / "B2_film_modulation.csv", index=False)
SENS.to_csv(SRC / "B2_biomarker_sensitivity.csv", index=False)


# ---------------- learning curves: one chart per figure
def loss_fig(hists, labels, colors, title, name, sel=None):
    fig, ax = plt.subplots(figsize=(W, 3.6))
    for h, lab, col in zip(hists, labels, colors):
        ax.plot(h.epoch, h.train_loss, "o-", ms=5, lw=1.6, color=col, label=f"{lab} - training")
        ax.plot(h.epoch, h.val_loss, "s--", ms=5, lw=1.6, color=col, label=f"{lab} - validation")
    if sel:
        ax.axvline(sel, color="#888888", ls=":", lw=1.3)
        ax.text(sel + 0.15, ax.get_ylim()[1] * 0.9, "encoders unfrozen", fontsize=9, color="#444444")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(title, fontsize=11); ax.legend(frameon=False)
    fig.tight_layout(); save(fig, OUT / "appendix", name)


def metric_fig(hists, labels, colors, title, name):
    fig, ax = plt.subplots(figsize=(W, 3.6))
    for h, lab, col in zip(hists, labels, colors):
        ax.plot(h.epoch, h.val_auc, "o-", ms=5, lw=1.6, color=col, label=f"{lab} - AUC")
        ax.plot(h.epoch, h.val_macro_f1, "^--", ms=5, lw=1.6, color=col,
                label=f"{lab} - macro F1")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation metric")
    ax.set_title(title, fontsize=11); ax.legend(frameon=False)
    fig.tight_layout(); save(fig, OUT / "appendix", name)


loss_fig([Hh, Ih], ["H", "I"], ["#1b1b1b", "#c05000"],
         "Task 9 - loss per epoch (encoder tails unfrozen after epoch 3)", "N1a_task9_loss", sel=3.5)
metric_fig([Hh, Ih], ["H", "I"], ["#1b1b1b", "#c05000"],
           "Task 9 - validation discrimination per epoch", "N1b_task9_validation")
loss_fig([J0, J1], ["J0", "J1"], ["#1b1b1b", "#c05000"],
         "Task 10 - loss per epoch (frozen embeddings, head only)", "N2a_task10_loss")
metric_fig([J0, J1], ["J0", "J1"], ["#1b1b1b", "#c05000"],
           "Task 10 - validation discrimination per epoch", "N2b_task10_validation")
loss_fig([Lh], ["L"], ["#1b1b1b"],
         "Task 13 - loss per epoch (backbones unfrozen after epoch 3)", "N3a_task13_loss", sel=3.5)
metric_fig([Lh], ["L"], ["#1b1b1b"],
           "Task 13 - validation discrimination per epoch", "N3b_task13_validation")
pd.concat([Hh.assign(model="H"), Ih.assign(model="I")]).to_csv(SRC / "N1_task9_learning_curves.csv", index=False)
pd.concat([J0.assign(model="J0"), J1.assign(model="J1")]).to_csv(SRC / "N2_task10_learning_curves.csv", index=False)
Lh.assign(model="L").to_csv(SRC / "N3_task13_learning_curves.csv", index=False)

# ---------------- A1 split
cl = t13att[t13att.grouping == "true_class"]
sr = t13att[t13att.grouping == "source"]
for sub, nm, ttl in ((cl, "A1a_attention_by_class", "Task 13 - mean attention weight by true class"),
                     (sr, "A1b_attention_by_source", "Task 13 - mean attention weight by acquisition source")):
    fig, ax = plt.subplots(figsize=(W, 3.4))
    x = np.arange(len(sub))
    ax.bar(x - 0.20, sub.attn_resnet_mean.values, 0.40, label="ResNet50 branch",
           color="#1b1b1b", edgecolor="#333333", linewidth=0.7)
    ax.bar(x + 0.20, sub.attn_effnet_mean.values, 0.40, label="EfficientNet-B4 branch",
           color="#c8c8c8", edgecolor="#333333", linewidth=0.7)
    for xx, vv in zip(x - 0.20, sub.attn_resnet_mean.values):
        ax.text(xx, vv + 0.02, f"{vv:.3f}", ha="center", fontsize=9)
    for xx, vv in zip(x + 0.20, sub.attn_effnet_mean.values):
        ax.text(xx, vv + 0.02, f"{vv:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([v.replace("_", "-") for v in sub.value])
    ax.set_ylim(0, 1.15); ax.set_ylabel("mean attention weight")
    ax.set_title(ttl, fontsize=11); ax.legend(frameon=False)
    fig.tight_layout(); save(fig, OUT / "appendix", nm)
t13att.to_csv(SRC / "A1_attention_by_class_and_source.csv", index=False)

# ---------------- A2 split
st = t13attst
for split, nm, lab in (("val", "A2a_attention_distribution_val", "validation (N = 1328)"),
                       ("test", "A2b_attention_distribution_test", "test (N = 1331)")):
    r = st[st.split == split].iloc[0]
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ax.hlines(0, r.attn_resnet_p05, r.attn_resnet_p95, color="#1b1b1b", lw=3.4)
    ax.plot([r.attn_resnet_p50], [0], "o", ms=10, color="#1b1b1b", label="median")
    ax.plot([r.attn_resnet_mean], [0], "D", ms=8, color="#c05000", label="mean")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.7, 0.7); ax.set_yticks([])
    ax.set_xlabel("ResNet50 attention weight")
    ax.set_title(f"Task 13 - {lab}: mean {r.attn_resnet_mean:.4f}, sd {r.attn_resnet_std:.4f}, "
                 f"p05 {r.attn_resnet_p05:.4f}, p95 {r.attn_resnet_p95:.4f}", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    ax.axvline(0.95, color="#888888", ls="--", lw=1.0)
    fig.tight_layout(); save(fig, OUT / "appendix", nm)
st.to_csv(SRC / "A2_attention_distribution.csv", index=False)

# ---------------- S1 split
SRC3 = ["farfum_rop", "farabi"]
fig, ax = plt.subplots(figsize=(W, 3.6))
x = np.arange(len(SRC3)); w = 0.26
for j, (m, fc) in enumerate(zip(["B_LOSO", "E_LOSO", "G_LOSO"], ["#c8c8c8", "#7a7a7a", "#1b1b1b"])):
    v = [float(prow(t11m, held_out_source=h, model=m).multiclass_auc) for h in SRC3]
    ax.bar(x + (j - 1) * w, v, w, label=m.split("_")[0], color=fc, edgecolor="#333333", linewidth=0.7)
    for xx, vv in zip(x + (j - 1) * w, v):
        ax.text(xx, vv + 0.012, f"{vv:.4f}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(["FARFUM-RoP", "Farabi"])
ax.set_ylim(0, 0.95); ax.set_ylabel("3-class macro OVR AUC")
ax.set_title("Held-out source evaluation - three-class AUC", fontsize=11)
ax.legend(frameon=False, title=None)
fig.tight_layout(); save(fig, OUT / "main", "S1a_source_heldout_threeclass")

fig, ax = plt.subplots(figsize=(W, 3.4))
v = [float(prow(t11m, held_out_source="plus", model=m).restricted_binary_auc)
     for m in ["B_LOSO", "E_LOSO", "G_LOSO"]]
ax.bar(np.arange(3), v, color=["#c8c8c8", "#7a7a7a", "#1b1b1b"], edgecolor="#333333", linewidth=0.7,
       width=0.55)
for xx, vv in zip(np.arange(3), v):
    ax.text(xx, vv + 0.0006, f"{vv:.4f}", ha="center", fontsize=9)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(["B", "E", "G"])
ax.set_ylim(0.975, 0.99)
ax.set_ylabel("restricted Normal-vs-Plus binary AUC")
ax.set_title("Held-out Plus source - restricted binary AUC (axis starts at 0.975)", fontsize=10)
fig.tight_layout(); save(fig, OUT / "main", "S1b_source_heldout_plus_restricted")
t11m.to_csv(SRC / "S1_source_heldout_performance.csv", index=False)

# ---------------- DS1 split
for blk, nm, ttl in (("rgb_embedding", "DS1a_shift_rgb", "RGB embedding (2048-d)"),
                     ("vessel_embedding", "DS1b_shift_vessel", "Vessel embedding (1792-d)")):
    fig, ax = plt.subplots(figsize=(W, 3.4))
    s = t11s[t11s.block == blk]
    x = np.arange(len(s)); w = 0.26
    for j, (c, lab, fc) in enumerate([("median_abs_smd", "median |SMD|", "#c8c8c8"),
                                      ("p90_abs_smd", "p90 |SMD|", "#7a7a7a"),
                                      ("p95_abs_smd", "p95 |SMD|", "#1b1b1b")]):
        ax.bar(x + (j - 1) * w, s[c].values, w, label=lab, color=fc, edgecolor="#333333",
               linewidth=0.7)
        for xx, vv in zip(x + (j - 1) * w, s[c].values):
            ax.text(xx, vv + 0.03, f"{vv:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([q.replace("_", "-") for q in s.held_out_source])
    ax.set_ylabel("standardized mean difference")
    ax.set_title(f"Domain shift of the {ttl}", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout(); save(fig, OUT / "main", nm)
t11s.to_csv(SRC / "DS1_representation_domain_shift.csv", index=False)

# ---------------- DA1 split
fig, ax = plt.subplots(figsize=(W, 3.6))
hs = ["farfum_rop", "farabi"]; x = np.arange(len(hs)); w = 0.32
for j, (k, fc, lab) in enumerate(zip(["K0", "K1"], ["#c8c8c8", "#1b1b1b"],
                                     ["domain-neutral control", "CC-DANN + MMD"])):
    v = [float(prow(t12m, held_out_source=h, model=k).multiclass_auc) for h in hs]
    ax.bar(x + (j - 0.5) * w, v, w, label=lab, color=fc, edgecolor="#333333", linewidth=0.7)
    for xx, vv in zip(x + (j - 0.5) * w, v):
        ax.text(xx, vv + 0.012, f"{vv:.4f}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(["FARFUM-RoP", "Farabi"])
ax.set_ylim(0, 0.95); ax.set_ylabel("3-class macro OVR AUC")
ax.set_title("Task 12 - held-out source, three-class AUC", fontsize=11)
ax.legend(frameon=False)
fig.tight_layout(); save(fig, OUT / "main", "DA1a_domain_alignment_threeclass")

fig, ax = plt.subplots(figsize=(W, 3.4))
v = [0.98932, 0.98981]
ax.bar(np.arange(2), v, color=["#c8c8c8", "#1b1b1b"], edgecolor="#333333", linewidth=0.7, width=0.5)
for xx, vv in zip(np.arange(2), v):
    ax.text(xx, vv + 0.0006, f"{vv:.5f}", ha="center", fontsize=9)
ax.set_xticks(np.arange(2)); ax.set_xticklabels(["K0", "K1"])
ax.set_ylim(0.985, 0.992)
ax.set_ylabel("restricted Normal-vs-Plus binary AUC")
ax.set_title("Task 12 - held-out Plus source, restricted binary AUC (axis starts at 0.985)",
             fontsize=10)
fig.tight_layout(); save(fig, OUT / "main", "DA1b_domain_alignment_plus_restricted")
t12m.to_csv(SRC / "DA1_task12_results.csv", index=False)

# ---------------- DA2 split
for col, nm, ttl in (("centroid_distance_standardized", "DA2a_alignment_centroid",
                      "standardized centroid distance between the two training domains"),
                     ("median_abs_smd", "DA2b_alignment_median_smd",
                      "median absolute SMD between the two training domains")):
    fig, ax = plt.subplots(figsize=(W, 3.4))
    folds = ["plus", "farfum_rop", "farabi"]; x = np.arange(len(folds)); w = 0.32
    for j, (k, fc) in enumerate(zip(["K0", "K1"], ["#c8c8c8", "#1b1b1b"])):
        v = [float(prow(t12s, held_out=q, kind=k)[col]) for q in folds]
        ax.bar(x + (j - 0.5) * w, v, w, label=k, color=fc, edgecolor="#333333", linewidth=0.7)
        for xx, vv in zip(x + (j - 0.5) * w, v):
            ax.text(xx, vv + max(v) * 0.02, f"{vv:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([q.replace("_", "-") for q in folds])
    ax.set_ylabel("value on the source-validation split")
    ax.set_title(f"Task 12 - {ttl}", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout(); save(fig, OUT / "main", nm)
t12s.to_csv(SRC / "DA2_alignment_shift_diagnostics.csv", index=False)

print("SPLIT_FIGURES_DONE")
