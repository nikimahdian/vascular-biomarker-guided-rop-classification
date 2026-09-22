"""Regenerate the learning-curve figures as full-width stacked single-column panels."""
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
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
SRC = WS / "report_assets_final/source_data"


def save(fig, folder, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(folder / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def hist(p):
    return pd.DataFrame(json.loads(Path(p).read_text()))


def stack2(hists, labels, colors, title, folder, name, sel=None):
    """Two full-width stacked panels: loss, then validation metrics."""
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 8.0))
    for h, lab, col in zip(hists, labels, colors):
        ax[0].plot(h.epoch, h.train_loss, "o-", ms=4, lw=1.4, color=col, label=f"{lab} - train")
        ax[0].plot(h.epoch, h.val_loss, "s--", ms=4, lw=1.4, color=col, label=f"{lab} - validation")
        ax[1].plot(h.epoch, h.val_auc, "o-", ms=4, lw=1.4, color=col, label=f"{lab} - AUC")
        ax[1].plot(h.epoch, h.val_macro_f1, "^--", ms=4, lw=1.4, color=col,
                   label=f"{lab} - macro F1")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[0].set_title("(a) training and validation loss", loc="left")
    ax[0].legend(frameon=False, ncol=2)
    if sel:
        ax[0].axvline(sel, color="#888888", ls=":", lw=1.2)
        ax[0].annotate("encoders unfrozen", xy=(sel, ax[0].get_ylim()[1] * 0.92),
                       xytext=(sel + 0.2, ax[0].get_ylim()[1] * 0.92), fontsize=8, color="#444444")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation metric")
    ax[1].set_title("(b) validation discrimination", loc="left")
    ax[1].legend(frameon=False, ncol=2)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout()
    save(fig, folder, name)


H = hist(A / "task9_joint_multimodal_fusion/H_JOINT_RGB_VESSEL_history.json")
I = hist(A / "task9_joint_multimodal_fusion/I_JOINT_RGB_VESSEL_BIOMARKER_history.json")
J0 = hist(A / "task10_biomarker_film/J0_FROZEN_FUSION_CONTROL_history.json")
J1 = hist(A / "task10_biomarker_film/J1_BIOMARKER_FILM_history.json")
L = hist(A / "task13_ropdeepx_style/L_history.json")
Lsel = json.loads((A / "task13_ropdeepx_style/L_selection.json").read_text())

stack2([H, I], ["H joint RGB-vessel", "I joint + biomarkers"], ["#1b1b1b", "#c05000"],
       "Task 9 learning curves (encoder tails unfrozen after epoch 3)",
       OUT / "appendix", "N1_task9_learning_curves", sel=3.5)
stack2([J0, J1], ["J0 frozen fusion control", "J1 biomarker FiLM"], ["#1b1b1b", "#c05000"],
       "Task 10 learning curves (frozen embeddings, head-only training)",
       OUT / "appendix", "N2_task10_learning_curves")
stack2([L], ["L dual-RGB attention"], ["#1b1b1b"],
       "Task 13 learning curves (backbones unfrozen after epoch 3)",
       OUT / "appendix", "N3_task13_learning_curves", sel=3.5)

# attention over epochs, single full-width panel
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(L.epoch, L.attn_resnet_mean, "o-", ms=5, lw=1.6, color="#1b1b1b", label="ResNet50 branch")
ax.plot(L.epoch, L.attn_effnet_mean, "s--", ms=5, lw=1.6, color="#c05000",
        label="EfficientNet-B4 branch")
ax.axhline(0.95, color="#888888", ls=":", lw=1.0)
ax.axhline(0.05, color="#888888", ls=":", lw=1.0)
ax.set_xlabel("epoch"); ax.set_ylabel("mean attention weight")
ax.set_ylim(0, 1)
ax.set_title("Task 13 mean attention weight per epoch (dotted lines mark the collapse thresholds)",
             fontsize=11)
ax.legend(frameon=False)
fig.tight_layout()
save(fig, OUT / "appendix", "N3c_task13_attention_epochs")

for nm, df in (("N1_task9_learning_curves", pd.concat([H.assign(model="H"), I.assign(model="I")])),
               ("N2_task10_learning_curves", pd.concat([J0.assign(model="J0"), J1.assign(model="J1")])),
               ("N3_task13_learning_curves", L.assign(model="L")),
               ("N3c_task13_attention_epochs", L[["epoch", "attn_resnet_mean", "attn_effnet_mean"]])):
    df.to_csv(SRC / f"{nm}.csv", index=False)
print("FIGURES_REGENERATED")
