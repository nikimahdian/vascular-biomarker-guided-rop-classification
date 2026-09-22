"""Final report asset pack - PART 2: tables, LaTeX, captions, claim matrix, index, metadata."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path("/root/niki_rop_task6_isolated")
A = WS / "artifacts"
OUT = WS / "report_assets_final"
TM, TA = OUT / "tables/main", OUT / "tables/appendix"
LT = OUT / "latex/tables"
CAP, SRC, META = OUT / "captions", OUT / "source_data", OUT / "metadata"
GEN = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def esc(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("$", r"\$").replace("~", r"\textasciitilde{}")
            .replace("^", r"\textasciicircum{}"))


def latex_table(df, colfmt=None, caption="", label="", bold_cols=(), note=""):
    cols = list(df.columns)
    colfmt = colfmt or ("l" + "r" * (len(cols) - 1))
    out = [r"\begin{table}[htbp]", r"\centering", r"\small",
           rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{colfmt}}}", r"\toprule",
           " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            s = v if isinstance(v, str) else ("" if pd.isna(v) else
                                              (f"{v:.4f}" if isinstance(v, (float, np.floating))
                                               else str(v)))
            if c in bold_cols and s:
                s = rf"\textbf{{{esc(s)}}}"
            cells.append(esc(s) if not s.startswith(r"\textbf") else s)
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule"]
    if note:
        out.append(rf"\multicolumn{{{len(cols)}}}{{l}}{{\footnotesize {esc(note)}}} \\")
    out += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def md_table(df, caption=""):
    o = [f"**{caption}**", "", "| " + " | ".join(map(str, df.columns)) + " |",
         "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        o.append("| " + " | ".join(
            ("" if (not isinstance(v, str) and pd.isna(v)) else
             (f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)))
            for v in r.values) + " |")
    return "\n".join(o) + "\n"


def emit(df, folder, name, caption, label, colfmt=None, bold_cols=(), note="", markdown=True):
    df.to_csv(folder / f"{name}.csv", index=False)
    (LT / f"{name}.tex").write_text(
        latex_table(df, colfmt, caption, label, bold_cols, note), encoding="utf-8")
    if markdown:
        (folder / f"{name}.md").write_text(md_table(df, caption), encoding="utf-8")
    log(f"  table {name} ({len(df)} rows)")


# ---------------------------------------------------------------- load
t6m = pd.read_csv(A / "task6/metrics_summary.csv").set_index("model")
t8m = pd.read_csv(A / "task8_spatial_vessel_fusion/metrics_summary.csv").set_index("model")
t9m = pd.read_csv(A / "task9_joint_multimodal_fusion/metrics_summary.csv").set_index("model")
t10m = pd.read_csv(A / "task10_biomarker_film/metrics_summary.csv").set_index("model")
t13m = pd.read_csv(A / "task13_ropdeepx_style/metrics_summary.csv").set_index("model")
t7p = pd.read_csv(A / "task7_paired_statistics/paired_primary_metrics.csv")
t8bp = pd.read_csv(A / "task8_spatial_vessel_fusion/statistical_closure_10k/paired_metrics_10k.csv")
t9p = pd.read_csv(A / "task9_joint_multimodal_fusion/paired_all_10k.csv")
t10p = pd.read_csv(A / "task10_biomarker_film/paired_all_10k.csv")
t11p = pd.read_csv(A / "task11_source_heldout/paired_bootstrap_10k.csv")
t11m = pd.read_csv(A / "task11_source_heldout/metrics_per_source.csv")
t11s = pd.read_csv(A / "task11_source_heldout/domain_shift_diagnostics.csv")
t11c = pd.read_csv(A / "task11_source_heldout/population_audit.csv")
t12p = pd.read_csv(A / "task12_class_conditional_domain_generalization/paired_bootstrap_10k.csv")
t12m = pd.read_csv(A / "task12_class_conditional_domain_generalization/heldout_metrics.csv")
t12s = pd.read_csv(A / "task12_class_conditional_domain_generalization/representation_shift.csv")
t12d = pd.read_csv(A / "task12_class_conditional_domain_generalization/domain_predictability.csv")
t13p = pd.read_csv(A / "task13_ropdeepx_style/paired_all_10k.csv")
manifest = pd.read_csv(WS / "primary_complete_case_v2_server2.csv", low_memory=False)

LABEL = {"A_PRIMARY": "Scalar biomarkers", "B_RGB": "RGB CNN",
         "B_EMBEDDING_ONLY": "RGB embedding", "C_PRIMARY": "RGB + biomarkers",
         "D_VESSEL_MAP": "Vessel map CNN", "E_RGB_VESSEL_FEATURE_FUSION": "RGB + vessel",
         "F_RGB_VESSEL_LATE_FUSION": "RGB + vessel late fusion",
         "G_RGB_VESSEL_SCALAR_FUSION": "RGB + vessel + biomarkers",
         "H_JOINT_RGB_VESSEL": "Joint RGB-vessel",
         "I_JOINT_RGB_VESSEL_BIOMARKER": "Joint RGB-vessel + biomarkers",
         "J0_FROZEN_FUSION_CONTROL": "Frozen fusion control",
         "J1_BIOMARKER_FILM": "Biomarker-conditioned FiLM",
         "L_ROPDEEPX_STYLE_RGB": "Dual-RGB attention (L)",
         "M0_ROPDEEPX_EMBEDDING_ONLY": "Dual-RGB embedding (M0)",
         "M1_ROPDEEPX_PLUS_VESSEL": "Dual-RGB + vessel (M1)"}
ARCH = {"A_PRIMARY": "5 biomarkers -> XGBoost",
        "B_RGB": "RGB -> EfficientNet-B5 (2048) -> linear classifier",
        "B_EMBEDDING_ONLY": "RGB -> EfficientNet-B5 (2048) -> XGBoost",
        "C_PRIMARY": "2048 RGB + 5 biomarkers (2053) -> XGBoost",
        "D_VESSEL_MAP": "Vessel map -> EfficientNet-B4 (1792) -> linear classifier",
        "E_RGB_VESSEL_FEATURE_FUSION": "2048 RGB + 1792 vessel (3840) -> XGBoost",
        "F_RGB_VESSEL_LATE_FUSION": "alpha-weighted late fusion of B_EMBEDDING_ONLY and D (alpha = 1.0)",
        "G_RGB_VESSEL_SCALAR_FUSION": "2048 RGB + 1792 vessel + 5 biomarkers (3845) -> XGBoost",
        "H_JOINT_RGB_VESSEL": "frozen B5 + frozen B4 -> projected 512 + 512 -> interaction -> MLP",
        "I_JOINT_RGB_VESSEL_BIOMARKER": "as H + 32-d biomarker branch on the 128-d trunk",
        "J0_FROZEN_FUSION_CONTROL": "frozen 2048 + 1792 -> 512 + 512 -> interaction -> MLP",
        "J1_BIOMARKER_FILM": "as J0 with bounded residual FiLM on the vessel branch",
        "L_ROPDEEPX_STYLE_RGB": "RGB -> ResNet50 + EfficientNet-B4 -> soft attention -> 1024 -> linear",
        "M0_ROPDEEPX_EMBEDDING_ONLY": "1024-d attended RGB embedding -> XGBoost",
        "M1_ROPDEEPX_PLUS_VESSEL": "1024-d attended RGB + 1792 vessel (2816) -> XGBoost"}
MET = ["multiclass_auc", "balanced_accuracy", "macro_f1", "brier", "ece"]


def cm(model):
    for t in (t6m, t8m, t9m, t10m, t13m):
        if model in t.index:
            return t.loc[model]
    raise KeyError(model)


def prow(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v)
    return df[m].iloc[0]


def pr(df, tag, metric, col="delta"):
    r = prow(df, comparison=tag, metric=metric)
    return float(r[col])


def ci(df, tag, metric):
    r = prow(df, comparison=tag, metric=metric)
    return float(r.ci95_lo), float(r.ci95_hi), float(r.p_two_sided_null_centered)


def fmtp(p):
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


# ================================================================ TABLES
def tables():
    log("cohort tables")
    c = t11c[t11c.source.isin(["plus", "farfum_rop", "farabi"])].copy()
    tot = int(c.n.sum())
    c["pct"] = (100.0 * c.n / tot).round(1)
    d1 = pd.DataFrame([{"Source": r.source.replace("_", "-"), "Images": int(r.n),
                        "Groups": int(r.groups), "Normal": int(r.normal),
                        "Pre-Plus": int(r.pre_plus), "Plus": int(r.plus_cls),
                        "% total": f"{r.pct:.1f}"} for _, r in c.iterrows()]
                     + [{"Source": "Total", "Images": tot, "Groups": int(c.groups.sum()),
                         "Normal": int(c.normal.sum()), "Pre-Plus": int(c.pre_plus.sum()),
                         "Plus": int(c.plus_cls.sum()), "% total": "100.0"}])
    emit(d1, TM, "D1_cohort_summary",
         "Cohort summary by acquisition source (final complete-case population).",
         "tab:cohort", colfmt="lrrrrrr")

    rows = []
    for s in ("train", "val", "test"):
        sub = manifest[manifest.split == s]
        rows.append({"Split": s.capitalize(), "N": len(sub), "Normal": int((sub.label == 0).sum()),
                     "Pre-Plus": int((sub.label == 1).sum()), "Plus": int((sub.label == 2).sum()),
                     "Groups": int(sub.group_id.nunique())})
    rows.append({"Split": "Total", "N": len(manifest),
                 "Normal": int((manifest.label == 0).sum()),
                 "Pre-Plus": int((manifest.label == 1).sum()),
                 "Plus": int((manifest.label == 2).sum()),
                 "Groups": int(manifest.group_id.nunique())})
    d2 = pd.DataFrame(rows)
    emit(d2, TM, "D2_canonical_split_summary",
         "Canonical group-disjoint split of the complete-case population.", "tab:split",
         colfmt="lrrrrr")

    log("performance tables")
    order = ["A_PRIMARY", "B_RGB", "B_EMBEDDING_ONLY", "C_PRIMARY", "D_VESSEL_MAP",
             "E_RGB_VESSEL_FEATURE_FUSION", "F_RGB_VESSEL_LATE_FUSION",
             "G_RGB_VESSEL_SCALAR_FUSION"]
    r1 = pd.DataFrame([{"Model": LABEL[m], "Input / architecture": ARCH[m],
                        "AUC": float(cm(m).multiclass_auc),
                        "Balanced accuracy": float(cm(m).balanced_accuracy),
                        "Macro F1": float(cm(m).macro_f1), "Brier": float(cm(m).brier),
                        "ECE": float(cm(m).ece)} for m in order])
    best = {k: r1[k].max() if k in ("AUC", "Balanced accuracy", "Macro F1") else r1[k].min()
            for k in ("AUC", "Balanced accuracy", "Macro F1", "Brier", "ECE")}
    emit(r1, TM, "R1_master_model_results",
         "Master test-set performance table. AUC, balanced accuracy and macro F1 are "
         "higher-is-better; Brier and ECE are lower-is-better. Bold marks the numerically best "
         "value in each metric column; models are not ranked by any composite score.",
         "tab:master", colfmt="p{4.2cm}p{5.4cm}rrrrr",
         note=f"Bold = best per column. AUC {best['AUC']:.6f}; balanced accuracy "
              f"{best['Balanced accuracy']:.6f}; macro F1 {best['Macro F1']:.6f}; Brier "
              f"{best['Brier']:.6f}; ECE {best['ECE']:.6f}. F_RGB_VESSEL_LATE_FUSION collapsed to "
              f"B_EMBEDDING_ONLY because its alpha saturated at 1.0.")

    r2o = ["A_PRIMARY", "B_RGB", "B_EMBEDDING_ONLY", "C_PRIMARY", "D_VESSEL_MAP",
           "E_RGB_VESSEL_FEATURE_FUSION", "G_RGB_VESSEL_SCALAR_FUSION"]
    r2 = pd.DataFrame([{"Model": LABEL[m], "Normal AUC": float(cm(m).auc_Normal),
                        "Pre-Plus AUC": float(cm(m).auc_Pre_Plus),
                        "Plus AUC": float(cm(m).auc_Plus)} for m in r2o])
    emit(r2, TM, "R2_per_class_auc", "Per-class one-vs-rest AUC on the canonical test set.",
         "tab:perclass", colfmt="lrrr", markdown=False)

    log("paired statistics tables")
    crows = []
    for tag, lab, src in (("C_minus_B", "C vs B (biomarkers on RGB embedding)", t7p),
                          ("E_minus_B", "E vs B (vessel on RGB embedding)", t8bp),
                          ("G_minus_E", "G vs E (biomarkers after vessel)", t8bp)):
        for met, nm in (("multiclass_auc", "AUC"), ("balanced_accuracy", "Balanced accuracy"),
                        ("macro_f1", "Macro F1"), ("brier", "Brier"), ("ece", "ECE")):
            if tag == "C_minus_B":
                r = prow(t7p, metric=met)
                d, lo, hi, p = float(r.delta), float(r.ci95_lo), float(r.ci95_hi), \
                    float(r.p_two_sided_null_centered)
            else:
                r = prow(t8bp, comparison=tag, metric=met)
                d, lo, hi, p = float(r.delta), float(r.ci95_lo), float(r.ci95_hi), \
                    float(r.p_two_sided_null_centered)
            crows.append({"Comparison": lab, "Metric": nm, "Delta": d,
                          "95% CI": f"[{lo:+.6f}, {hi:+.6f}]", "p": fmtp(p),
                          "CI excludes 0": "yes" if (lo > 0 or hi < 0) else "no"})
    c1 = pd.DataFrame(crows)
    emit(c1, TM, "C1_primary_paired_statistics",
         "Primary paired statistics on the canonical test set (paired class-stratified bootstrap, "
         "seed 42, 10,000 replicates, N = 1,331). C vs B is from Task 7; E vs B and G vs E are the "
         "10,000-replicate Task 8B closure.", "tab:primarypaired",
         colfmt="p{5.0cm}lrrrl")

    ebr = prow(t8bp, comparison="E_minus_B", metric="multiclass_auc")
    m1r = prow(t13p, comparison="M1_minus_M0", metric="multiclass_auc")
    c2 = pd.DataFrame([
        {"RGB representation": "Original single-backbone (EfficientNet-B5)",
         "RGB-only AUC": float(cm("B_EMBEDDING_ONLY").multiclass_auc),
         "RGB + vessel AUC": float(cm("E_RGB_VESSEL_FEATURE_FUSION").multiclass_auc),
         "Delta AUC": float(ebr.delta),
         "95% CI": f"[{float(ebr.ci95_lo):+.6f}, {float(ebr.ci95_hi):+.6f}]",
         "p": fmtp(float(ebr.p_two_sided_null_centered)),
         "Delta macro F1": pr(t8bp, "E_minus_B", "macro_f1"),
         "Delta Brier": pr(t8bp, "E_minus_B", "brier")},
        {"RGB representation": "Dual-backbone attention (ResNet50 + EfficientNet-B4)",
         "RGB-only AUC": float(cm("M0_ROPDEEPX_EMBEDDING_ONLY").multiclass_auc),
         "RGB + vessel AUC": float(cm("M1_ROPDEEPX_PLUS_VESSEL").multiclass_auc),
         "Delta AUC": float(m1r.delta),
         "95% CI": f"[{float(m1r.ci95_lo):+.6f}, {float(m1r.ci95_hi):+.6f}]",
         "p": fmtp(float(m1r.p_two_sided_null_centered)),
         "Delta macro F1": pr(t13p, "M1_minus_M0", "macro_f1"),
         "Delta Brier": pr(t13p, "M1_minus_M0", "brier")}])
    emit(c2, TM, "C2_architecture_robustness",
         "Architecture robustness of the spatial-vessel contribution. Each row adds the same frozen "
         "1792-d vessel embedding to a different RGB representation, with its own matched RGB-only "
         "control; the two rows are independent experiments, not a single model comparison.",
         "tab:archrobust", colfmt="p{5.6cm}rrrrrrr")

    log("biomarker tables")
    brows = [("Canonical split: C vs B embedding", "RGB embedding",
              "RGB + biomarkers", t7p, "multiclass_auc", "C_minus_B"),
             ("Canonical split: G vs E", "RGB + vessel", "RGB + vessel + biomarkers", t8bp,
              "multiclass_auc", "G_minus_E"),
             ("Joint neural fusion: I vs H", "Joint RGB-vessel",
              "Joint RGB-vessel + biomarkers", t9p, "multiclass_auc", "I_minus_H"),
             ("FiLM conditioning: J1 vs J0", "Frozen fusion control",
              "Biomarker-conditioned FiLM", t10p, "multiclass_auc", "J1_minus_J0"),
             ("Held-out FARFUM-RoP: G vs E", "RGB + vessel (LOSO)",
              "RGB + vessel + biomarkers (LOSO)", t11p, "multiclass_auc", "G_minus_E_farfum_rop"),
             ("Held-out Farabi: G vs E", "RGB + vessel (LOSO)",
              "RGB + vessel + biomarkers (LOSO)", t11p, "multiclass_auc", "G_minus_E_farabi"),
             ("Held-out Plus (restricted): G vs E", "RGB + vessel (LOSO)",
              "RGB + vessel + biomarkers (LOSO)", t11p, "restricted_binary_auc",
              "G_minus_E_plus")]
    brow = []
    for exp, base, biom, tbl, met, tag in brows:
        if tag == "C_minus_B":
            r = prow(t7p, metric=met)
        else:
            r = prow(tbl, comparison=tag, metric=met)
        d, lo, hi, p = float(r.delta), float(r.ci95_lo), float(r.ci95_hi), \
            float(r.p_two_sided_null_centered)
        if lo > 0:
            interp = "supported positive increment"
        elif hi < 0:
            interp = "statistically supported decrease"
        else:
            interp = "no statistically supported increment"
        brow.append({"Experiment": exp, "Baseline": base, "Biomarker model": biom,
                     "Delta AUC": d, "95% CI": f"[{lo:+.6f}, {hi:+.6f}]", "p": fmtp(p),
                     "Interpretation": interp})
    b1 = pd.DataFrame(brow)
    emit(b1, TM, "B1_biomarker_evidence_summary",
         "Evidence summary for the five scalar biomarkers across every experiment in which they "
         "were tested. The last row uses the restricted Normal-vs-Plus binary AUC because the Plus "
         "source contains no Pre-Plus.", "tab:biomarkers",
         colfmt="p{4.4cm}p{3.0cm}p{3.4cm}rrrp{3.2cm}")

    log("negative experiment table")
    nrows = []
    for m, note in (("H_JOINT_RGB_VESSEL", "learned cross-modal interaction; selected stage-2 epoch 4"),
                    ("I_JOINT_RGB_VESSEL_BIOMARKER", "as H plus biomarker branch; selected epoch 6"),
                    ("J0_FROZEN_FUSION_CONTROL", "frozen-embedding MLP control; selected epoch 1"),
                    ("J1_BIOMARKER_FILM", "bounded residual FiLM; selected epoch 1")):
        r = cm(m)
        nrows.append({"Model": LABEL[m], "AUC": float(r.multiclass_auc),
                      "Balanced accuracy": float(r.balanced_accuracy),
                      "Macro F1": float(r.macro_f1), "Brier": float(r.brier),
                      "ECE": float(r.ece), "Interpretation": note})
    n1 = pd.DataFrame(nrows)
    emit(n1, TA, "N1_negative_architecture_experiments",
         "Secondary exploratory neural experiments that did not improve on the frozen "
         "concatenation models. Values are canonical test-set metrics; none of these models is a "
         "primary result.", "tab:negative", colfmt="p{4.6cm}rrrrrp{5.4cm}")

    log("source-held-out tables")
    srows = []
    for h, lab in (("plus", "Plus (restricted Normal-vs-Plus)"), ("farfum_rop", "FARFUM-RoP"),
                   ("farabi", "Farabi")):
        for m, ms in (("B_LOSO", "B"), ("E_LOSO", "E"), ("G_LOSO", "G")):
            r = prow(t11m, held_out_source=h, model=m)
            srows.append({"Held-out source": lab, "Model": ms, "N": int(r.n),
                          "3-class AUC": float(r.multiclass_auc),
                          "Restricted binary AUC": float(r.restricted_binary_auc),
                          "Balanced accuracy": float(r.balanced_accuracy),
                          "Macro F1": float(r.macro_f1),
                          "Brier": float(r.brier_multiclass_3col), "ECE": float(r.ece)})
    s1 = pd.DataFrame(srows)
    emit(s1, TM, "S1_source_heldout_performance",
         "Source-held-out (LOSO) performance. The held-out source contributed nothing to fitting, "
         "preprocessing or model choice. Three-class AUC is undefined for the Plus source because "
         "it contains no Pre-Plus; a restricted Normal-vs-Plus binary AUC is reported separately "
         "and is not a substitute for the three-class metric.", "tab:loso",
         colfmt="p{4.4cm}lrrrrrrr")

    ds = []
    for h in ("plus", "farfum_rop", "farabi"):
        for blk, nm in (("rgb_embedding", "RGB embedding"), ("vessel_embedding", "Vessel embedding")):
            r = prow(t11s, held_out_source=h, block=blk)
            ds.append({"Held-out source": h.replace("_", "-"), "Representation": nm,
                       "median |SMD|": float(r.median_abs_smd),
                       "p90 |SMD|": float(r.p90_abs_smd), "p95 |SMD|": float(r.p95_abs_smd),
                       "standardized centroid distance":
                           float(r.centroid_distance_standardized)})
    ds1 = pd.DataFrame(ds)
    emit(ds1, TM, "DS1_domain_shift_summary",
         "Descriptive domain-shift diagnostics for the frozen representations. No target labels "
         "were used in these statistics.", "tab:domainshift", colfmt="lrrrrr")

    log("task 12 table")
    darows = []
    for h, lab in (("farfum_rop", "FARFUM-RoP"), ("farabi", "Farabi"),
                   ("plus", "Plus (restricted)")):
        for met, nm in (("multiclass_auc", "AUC"), ("balanced_accuracy", "Balanced accuracy"),
                        ("macro_f1", "Macro F1"), ("brier_multiclass_3col", "Brier"),
                        ("ece", "ECE")):
            if h == "plus" and met == "multiclass_auc":
                k0 = k1 = np.nan
                tag = "K1_minus_K0_plus"
                r = prow(t12p, comparison=tag, metric="restricted_binary_auc")
            else:
                met_b = met
                if h == "plus" and met == "brier_multiclass_3col":
                    met_b = "brier_restricted_2class"
                if h == "plus":
                    k0 = float(prow(t12m, held_out_source=h, model="K0")[
                        "brier_restricted_2class" if met == "brier_multiclass_3col" else met])
                    k1 = float(prow(t12m, held_out_source=h, model="K1")[
                        "brier_restricted_2class" if met == "brier_multiclass_3col" else met])
                else:
                    k0 = float(prow(t12m, held_out_source=h, model="K0")[met])
                    k1 = float(prow(t12m, held_out_source=h, model="K1")[met])
                r = prow(t12p, comparison=f"K1_minus_K0_{h}", metric=met_b)
            lab_met = "Restricted binary AUC (Normal vs Plus)" if h == "plus" and \
                met == "multiclass_auc" else (
                    "Restricted 2-class Brier" if h == "plus" and
                    met == "brier_multiclass_3col" else nm)
            darows.append({"Held-out source": lab, "Metric": lab_met, "K0": k0, "K1": k1,
                           "K1-K0": float(r.delta),
                           "95% CI": f"[{float(r.ci95_lo):+.6f}, {float(r.ci95_hi):+.6f}]",
                           "p": fmtp(float(r.p_two_sided_null_centered))})
    da1 = pd.DataFrame(darows)
    emit(da1, TA, "DA1_task12_results",
         "Task 12 class-conditional domain-invariance results. K0 is the matched domain-neutral "
         "control and K1 adds the GRL domain discriminator and class-conditional MMD. The Plus row "
         "reports the restricted Normal-vs-Plus binary AUC for the AUC line.", "tab:task12",
         colfmt="p{3.6cm}p{5.0cm}rrrrr")

    log("final experiment summary")
    fin = pd.DataFrame([
        {"Task": "6", "Scientific question": "Which representation predicts ROP-Plus best on the "
         "canonical split?", "Models compared": "A, B (RGB CNN), B embedding, C",
         "Dataset protocol": "canonical split 6203/1328/1331, single test pass",
         "Main result": "C AUC 0.925988 highest; A AUC 0.659582 lowest",
         "Statistical result": "descriptive; forms the reference for Task 7",
         "Conclusion": "RGB-derived representations dominate scalar biomarkers",
         "Status": "primary, complete"},
        {"Task": "7", "Scientific question": "Are the model differences statistically supported?",
         "Models compared": "C vs B embedding",
         "Dataset protocol": "paired class-stratified bootstrap, 10,000 replicates, seed 42",
         "Main result": "delta AUC +0.001085",
         "Statistical result": "95% CI [-0.001691, +0.003868], p = 0.431",
         "Conclusion": "five scalar biomarkers add no supported increment on the RGB embedding",
         "Status": "primary, complete"},
        {"Task": "8", "Scientific question": "Does spatial vessel morphology add information?",
         "Models compared": "D, E, F, G vs B",
         "Dataset protocol": "canonical split, single test pass",
         "Main result": "E AUC 0.932841; G AUC 0.934880",
         "Statistical result": "E - B delta AUC +0.007938 (Task 8B closure)",
         "Conclusion": "supported spatial-vessel complementarity",
         "Status": "complete"},
        {"Task": "8B", "Scientific question": "Does the vessel finding survive a 10,000-replicate "
         "statistical closure?", "Models compared": "E vs B embedding; G vs E",
         "Dataset protocol": "paired class-stratified bootstrap, 10,000 replicates, seed 42",
         "Main result": "E - B AUC +0.007938; G - E AUC +0.002039",
         "Statistical result": "E - B CI [+0.002630, +0.013131] p = 0.0031; Brier -0.027693 "
         "p = 0.0011; G - E CI [-0.000274, +0.004397] p = 0.090",
         "Conclusion": "vessel increment supported; biomarker increment not", "Status": "complete"},
        {"Task": "9", "Scientific question": "Can learned joint fusion beat concatenation?",
         "Models compared": "H, I vs E, G",
         "Dataset protocol": "canonical split, staged CNN fine-tuning, single test pass",
         "Main result": "H AUC 0.925825; I AUC 0.916247", "Statistical result": "H - E AUC "
         "-0.007016 (p = 0.131); I - H AUC -0.009578 (p = 0.0049, harmful)",
         "Conclusion": "learned interaction underperformed concatenation; biomarkers harmful "
         "in-network", "Status": "secondary exploratory, complete"},
        {"Task": "10", "Scientific question": "Can biomarkers help by conditioning the vessel "
         "branch?", "Models compared": "J1 (FiLM) vs J0 (matched control)",
         "Dataset protocol": "canonical split, frozen embeddings, head-only training",
         "Main result": "J0 AUC 0.929754; J1 AUC 0.927631",
         "Statistical result": "J1 - J0 AUC -0.002123, p = 0.068; gamma within +/-0.3% of 1; "
         "single-biomarker ablation changes probabilities by ~2e-5 with 0 argmax flips",
         "Conclusion": "FiLM behaved near identity; no measurable conditioning effect",
         "Status": "secondary exploratory, complete"},
        {"Task": "11", "Scientific question": "How well does this transfer to an unseen source?",
         "Models compared": "B_LOSO, E_LOSO, G_LOSO",
         "Dataset protocol": "three source-held-out folds; held-out source never used in fitting",
         "Main result": "FARFUM E 0.809779 vs B 0.787537; Farabi E 0.782898 vs B 0.769944",
         "Statistical result": "E - B FARFUM +0.022242 p < 0.0001; Farabi +0.012954 p = 0.0018; "
         "G - E null in every fold",
         "Conclusion": "vessel benefit transfers; biomarker benefit does not; absolute performance "
         "drops by 0.12-0.15 AUC", "Status": "diagnostic, complete"},
        {"Task": "12", "Scientific question": "Can class-conditional domain invariance improve "
         "unseen-source performance?", "Models compared": "K1 (CC-DANN + MMD) vs K0 (matched control)",
         "Dataset protocol": "three source-held-out folds, target source fully unseen",
         "Main result": "K0 0.83293 vs K1 0.83013 (FARFUM); K0 0.79192 vs K1 0.79103 (Farabi)",
         "Statistical result": "K1 - K0 AUC -0.002796 (p = 0.143) and -0.000883 (p = 0.259); "
         "domain predictability unchanged at 0.88-0.98; representation shift reduced",
         "Conclusion": "outcome B: alignment reduced shift diagnostics but did not improve "
         "held-out disease performance", "Status": "secondary exploratory, complete"},
        {"Task": "13", "Scientific question": "1) Does a ROPDeepX-style dual-backbone RGB "
         "representation improve RGB performance? 2) Does the vessel representation remain "
         "complementary to it?",
         "Models compared": "L, M0, M1 vs B embedding, E, G",
         "Dataset protocol": "canonical split, staged CNN training, single test pass",
         "Main result": "1) No: M0 AUC 0.905978 vs B 0.924903. 2) Yes: M1 AUC 0.926913 vs M0 "
         "0.905978",
         "Statistical result": "M0 - B -0.018925 (p = 0.0063); M1 - M0 +0.020936 "
         "(p < 0.0001); M1 vs E and M1 vs G null",
         "Conclusion": "vessel complementarity is architecture-robust; the tested dual-RGB "
         "representation was not better than the original embedding",
         "Status": "secondary exploratory, complete"}])
    emit(fin, TM, "FINAL1_experiment_summary",
         "Complete experiment summary for the canonical-split and source-held-out studies. "
         "Historical Phase-5 results are excluded by construction.", "tab:experimentsummary",
         colfmt="lp{3.4cm}p{3.0cm}p{3.2cm}p{3.2cm}p{4.0cm}p{3.4cm}p{2.2cm}")

    log("tables done")


# ================================================================ CAPTIONS
EN = {
 "M1": ("Complete study pipeline",
  "Schematic of the analysis pipeline. A canonical 384x384 RGB fundus image is used twice: once by "
  "an EfficientNet-B5 RGB encoder producing a 2048-d representation, and once by the frozen "
  "segmentation model that yields a binary vessel map, which a spatial vessel encoder "
  "(EfficientNet-B4) converts into a 1792-d representation. In parallel, FOV-aware vascular "
  "measurement produces five scalar biomarkers. Fusion models combine these representations and "
  "output Normal / Pre-Plus / Plus. The canonical group-disjoint split (6,203 / 1,328 / 1,331), the "
  "paired class-stratified bootstrap (seed 42, 10,000 replicates) and the source-held-out evaluation "
  "protocols are indicated. No performance values appear in this figure."),
 "M2": ("Five scalar biomarker pipeline",
  "Schematic of the scalar biomarker pathway. A canonical RGB fundus image is segmented into a "
  "binary vessel map, which is passed through FOV-aware vascular measurement (coverage denominator "
  "= content area) to produce exactly five scalar biomarkers: vessel_density_fov, skel_density_fov, "
  "fractal_d0, fractal_d1 and fractal_d2. This is a methods schematic, not a statistical result."),
 "M3": ("Primary model architectures",
  "Architectures and feature dimensionalities of the primary models. A: five biomarkers into "
  "XGBoost. B: RGB into EfficientNet-B5 with a linear classifier. B embedding: RGB into "
  "EfficientNet-B5 producing a 2048-d vector into XGBoost. C: 2048-d RGB plus five biomarkers "
  "(2,053-d) into XGBoost. D: vessel map into EfficientNet-B4 with a linear classifier. E: 2048-d "
  "RGB plus 1,792-d vessel (3,840-d) into XGBoost. G: 2048-d RGB plus 1,792-d vessel plus five "
  "biomarkers (3,845-d) into XGBoost. All downstream XGBoost models share one frozen "
  "hyperparameter configuration."),
 "M4": ("Experimental question map",
  "Sequence of the nine experiments and the single scientific question each one addresses. Task 6 "
  "is the primary canonical comparison, Task 7 provides paired inference, Task 8 introduces the "
  "spatial vessel representation, Task 8B closes it statistically at 10,000 replicates, Tasks 9 and "
  "10 test learned fusion and biomarker conditioning, Task 11 quantifies source-held-out "
  "generalisation, Task 12 tests class-conditional domain invariance, and Task 13 tests whether the "
  "vessel result survives a different RGB backbone. No metrics are shown."),
 "D1": ("Class distribution by acquisition source",
  "Class distribution of the final complete-case population by acquisition source (N = 8,862; 414 "
  "groups). The Plus source contains 5,925 images of which 5,304 are Normal and 621 are Plus, and "
  "**zero Pre-Plus**; this is annotated in the figure. FARFUM-RoP contains 1,528 images (780 / 478 / "
  "270) and Farabi 1,409 images (369 / 452 / 588). Counts were read from the frozen population "
  "audit artifact."),
 "D2": ("Class distribution across canonical splits",
  "Class counts within the canonical group-disjoint training, validation and test splits of the "
  "complete-case population (train 6,203; validation 1,328; test 1,331; total 8,862). Group "
  "disjointness was verified: no group appears in more than one split."),
 "D3": ("Source contribution to the canonical population",
  "Contribution of each acquisition source to the canonical complete-case population of 8,862 "
  "images. The Plus source supplies 5,925 images (66.9%), FARFUM-RoP 1,528 (17.2%) and Farabi 1,409 "
  "(15.9%). A bar chart is used rather than a pie chart so that the counts remain directly "
  "comparable."),
 "R1": ("Canonical test discrimination by model",
  "Multiclass macro one-vs-rest ROC-AUC on the canonical test set (N = 1,331) for the primary "
  "models. Values are frozen Task 6 and Task 8 results. The x-axis starts at 0.60 to keep the "
  "differences legible; the absolute scale is labelled explicitly and the axis is not truncated "
  "below the lowest plotted value. F_RGB_VESSEL_LATE_FUSION is omitted because its mixture "
  "coefficient saturated at alpha = 1.0, making its probability table bit-identical to "
  "B_EMBEDDING_ONLY."),
 "R2": ("Multi-metric comparison on canonical test",
  "Five separate panels for multiclass AUC, balanced accuracy, macro F1, Brier score and expected "
  "calibration error (ECE, 15 equal-width bins) on the canonical test set (N = 1,331). AUC, "
  "balanced accuracy and macro F1 are higher-is-better; Brier and ECE are lower-is-better. Metrics "
  "are deliberately not combined into any composite score."),
 "R3": ("Per-class AUC",
  "One-vs-rest AUC for each class separately on the canonical test set (N = 1,331; Normal 982, "
  "Pre-Plus 140, Plus 209). Pre-Plus is the smallest class and the hardest for every model. Values "
  "are frozen Task 6 and Task 8 results."),
 "C1": ("Incremental information on the frozen RGB representation",
  "Paired class-stratified bootstrap deltas in multiclass AUC on the canonical test set (N = 1,331, "
  "seed 42, 10,000 replicates). Three increments are shown against the 2,048-d RGB embedding: "
  "adding the five scalar biomarkers (C - B), adding the spatial vessel representation (E - B), and "
  "adding the five scalar biomarkers after the vessel representation (G - E). The dashed line marks "
  "zero. Black intervals exclude zero; grey intervals cross it. C - B and G - E both cross zero, "
  "while E - B excludes zero positively."),
 "C2": ("Complementarity summary",
  "Single-figure summary of what each representation adds to the 2,048-d RGB embedding. Adding the "
  "five scalar biomarkers (C) yields delta AUC +0.001085 with 95% CI [-0.001691, +0.003868] "
  "(p = 0.431): no supported increment. Adding the spatial vessel representation (E) yields delta "
  "AUC +0.007938 with 95% CI [+0.002630, +0.013131] (p = 0.0031): a supported increment. Adding the "
  "five scalar biomarkers after the vessel representation (G) yields delta AUC +0.002039 with 95% "
  "CI [-0.000274, +0.004397] (p = 0.090): no supported increment. C and G are secondary "
  "exploratory models."),
 "C3": ("Vessel complementarity across distinct RGB feature extractors",
  "Paired 95% confidence intervals for the change in multiclass AUC when the same frozen 1,792-d "
  "vessel embedding is added to two different RGB representations on the canonical test set "
  "(N = 1,331, 10,000 paired class-stratified replicates, seed 42). With the original "
  "single-backbone EfficientNet-B5 representation the increment is +0.007938 "
  "[+0.002630, +0.013131], p = 0.0031; with the dual-backbone attention representation (Task 13) it "
  "is +0.020936 [+0.011426, +0.030828], p < 0.0001. The figure shows that the vessel contribution "
  "is not specific to one RGB feature extractor; it does **not** imply that the dual-backbone model "
  "itself is superior, and in fact its RGB-only control was worse than the original embedding."),
 "P1": ("Confusion matrices",
  "Row-normalised confusion matrices on the canonical test set (N = 1,331) with raw counts in "
  "parentheses, using identical class ordering for all panels: Normal, Pre-Plus, Plus. Rows are "
  "true classes and columns predicted classes, so each row sums to 100%. Pre-Plus is the smallest "
  "and most frequently missed class in every model. Individual panels are also provided as separate "
  "files."),
 "P2": ("Reliability",
  "Reliability diagrams on the canonical test set using the same 15 equal-width confidence bins as "
  "the reported ECE. The dashed diagonal is perfect calibration. ECE is shown in the legend for "
  "each model. All three models are under-confident at low confidence and over-confident at high "
  "confidence, and none lies on the diagonal."),
 "B1": ("Six independent tests of the five scalar biomarkers",
  "Forest plot of the change in AUC attributable to the five scalar biomarkers, shown separately "
  "for each experiment in which they were tested, with paired 95% confidence intervals from the "
  "frozen 10,000-replicate bootstraps (seed 42). Rows are labelled by analysis type: canonical "
  "split, joint neural model, FiLM conditioning, and source-held-out folds. The last row uses the "
  "restricted Normal-vs-Plus binary AUC because the Plus source has no Pre-Plus, and is therefore "
  "not directly comparable to the three-class rows. No compatible effect is statistically "
  "supported; the only interval excluding zero is negative (joint model)."),
  "R4": ("Task 13 ROPDeepX-style results",
  "Canonical test-set metrics (N = 1,331) for the Task 13 models against the frozen reference "
  "models, in five separate panels for multiclass AUC, balanced accuracy, macro F1, Brier and ECE. "
  "L is a trained neural head on the dual-RGB attention representation, M0 is XGBoost on the frozen "
  "1,024-d attended RGB embedding, and M1 adds the frozen 1,792-d vessel embedding to M0. L's own "
  "linear classifier is clearly weaker than XGBoost on L's own representation. L is not directly "
  "comparable to the embedding-based models because it is a trained end-to-end classifier rather "
  "than a downstream learner on frozen features."),
 "R5": ("Task 13 paired AUC comparisons",
  "Paired class-stratified bootstrap deltas in multiclass AUC on the canonical test set (N = 1,331, "
  "10,000 replicates, seed 42) for the four Task 13 comparisons. M0 - B is significantly negative, "
  "showing the dual-backbone attention RGB representation was worse than the original "
  "EfficientNet-B5 embedding; M1 - M0 is significantly positive, showing the vessel representation "
  "remains complementary; M1 - E and M1 - G are null, so the dual-RGB pipeline is statistically "
  "indistinguishable from the original RGB + vessel models."),
 "B2": ("FiLM diagnostics",
  "Panel a: learned FiLM modulation on the test set. The bounded residual parameterisation "
  "gamma = 1 + 0.10*tanh(delta_gamma) and beta = 0.10*tanh(beta_raw) produced gamma with mean "
  "0.999580, median 0.999624 and 5th-95th percentile range [0.996997, 1.001924], and beta with "
  "mean -0.000006 and percentile range [-0.002695, +0.002616]. Dashed lines mark identity "
  "(gamma = 1, beta = 0). Panel b: mean and 95th-percentile absolute change in the frozen J1 test "
  "probabilities when each biomarker is individually replaced by its training mean. All five "
  "biomarkers produce changes of order 2 x 10^-5 and no argmax flips, so the conditioner behaved "
  "effectively as an identity map."),
 "N1": ("Task 9 learning curves",
  "Training and validation loss and validation AUC per epoch for the two Task 9 joint models, with "
  "the encoder-unfreezing boundary at epoch 3.5 and the selected epoch marked. After unfreezing, "
  "training loss falls monotonically while validation loss rises steeply and validation AUC decays, "
  "the signature of severe overfitting; both models selected an epoch within two epochs of "
  "unfreezing. Panel c shows the canonical test AUC and macro F1 for reference. These are secondary "
  "exploratory experiments, not primary models."),
 "N2": ("Task 10 learning curves",
  "Training and validation loss and validation AUC per epoch for the Task 10 frozen-embedding "
  "models, with the selected epoch marked. Training loss falls monotonically while validation loss "
  "rises from its first-epoch minimum, so both models selected epoch 1. Even with both CNNs frozen "
  "into precomputed features, a 2.5 M-parameter head overfits a 6,203-sample training set. "
  "Secondary exploratory experiment."),
 "N3": ("Task 13 learning curves",
  "Panel a: training and validation loss per epoch for the ROPDeepX-style dual-RGB model, with the "
  "backbone-unfreezing boundary at epoch 3.5 and the selected epoch 2 marked. Panel b: validation "
  "AUC and macro F1, with the minimum-validation-loss epoch 7 marked. Training loss falls "
  "monotonically from 0.7638 to 0.5574 while validation AUC peaks at epoch 2 (0.86624) and declines "
  "to 0.84014 by epoch 7. Panel c: mean attention weight per epoch with collapse thresholds at 0.05 "
  "and 0.95; no collapse occurred. Secondary exploratory experiment."),
 "S1": ("Source-held-out performance",
  "Panel a: three-class macro one-vs-rest AUC on each held-out source for the LOSO models, where "
  "the held-out source contributed nothing to fitting or preprocessing. Panel b uses a different "
  "axis and is explicitly labelled **restricted Normal-vs-Plus** binary AUC, because the Plus source "
  "contains no Pre-Plus and a three-class AUC is undefined there. The two panels must not be "
  "compared on one axis. Panel b's axis covers 0.95 to 1.00 to resolve the three models; the "
  "absolute range is labelled."),
 "S2": ("Vessel complementarity under source-held-out evaluation",
  "Paired 95% confidence intervals for the change in AUC when the spatial vessel representation is "
  "added (E - B), computed separately within each source-held-out fold with 10,000 paired "
  "class-stratified replicates (seed 42). The first two rows are three-class macro AUC; the third "
  "row is the restricted Normal-vs-Plus binary AUC for the Plus fold and is labelled as such. The "
  "dashed line marks zero. Both comparable three-class folds exclude zero positively."),
 "S3": ("Biomarker increment under source-held-out evaluation",
  "Paired 95% confidence intervals for the change in AUC when the five scalar biomarkers are added "
  "after the spatial vessel representation (G - E), computed within each source-held-out fold with "
  "10,000 paired class-stratified replicates (seed 42). As in S2, the Plus row is the restricted "
  "Normal-vs-Plus binary AUC. Every interval crosses zero, so no biomarker increment is supported "
  "under source shift."),
 "DS1": ("Representation domain shift",
  "Descriptive shift between each held-out source and the pooled training sources for the frozen "
  "RGB (2,048-d) and vessel (1,792-d) embeddings, measured as the per-feature standardized mean "
  "difference SMD = (mean_target - mean_train) / sd_train. Median, 90th-percentile and "
  "95th-percentile absolute SMD are shown. No target labels were used. The vessel embedding is "
  "shifted more than the RGB embedding in every fold, and Plus is the most shifted source in both "
  "blocks."),
 "DS2": ("Standardized centroid distance",
  "Standardized centroid distance between each held-out source and the pooled training sources in "
  "the frozen RGB and vessel embedding spaces, computed as the Euclidean norm of the per-feature "
  "standardized mean differences. Plus is the most distant source in both representations; Farabi "
  "is more distant than FARFUM-RoP. Descriptive diagnostic only."),
 "DS3": ("Scalar biomarker source shift",
  "Heatmap of the standardized mean difference of each of the five scalar biomarkers between the "
  "held-out source and the pooled training sources. The colour scale is diverging and centred at "
  "zero; exact values are annotated. All five biomarkers are strongly source-dependent in every "
  "fold, with absolute SMD between 0.30 and 1.11, i.e. up to a full training standard deviation of "
  "shift. Plus lies highest and Farabi lowest on all five. This information was not used to retune "
  "any model."),
 "DA1": ("Task 12 domain-alignment performance",
  "Panel a: three-class macro AUC on each held-out source for the matched Task 12 pair, K0 "
  "(domain-neutral control) and K1 (class-conditional DANN + MMD). Panel b uses a different axis "
  "and reports the **restricted Normal-vs-Plus** binary AUC for the Plus fold, where a three-class "
  "AUC is undefined. The held-out source was never used in training, validation, alignment, early "
  "stopping or model selection. K1 did not improve on K0 in either comparable fold."),
 "DA2": ("Representation shift before and after alignment",
  "Standardized centroid distance and median absolute standardized mean difference between the two "
  "training domains, measured on the source-validation split of the learned 256-d representation "
  "for K0 and K1. Alignment reduced both statistics in all three folds, most strongly for the "
  "Farabi fold. No held-out source information entered this diagnostic."),
 "DA3": ("Domain predictability after alignment",
  "Five-fold cross-validated accuracy of a logistic regression trained on the source-validation "
  "representation to predict which of the two training domains a sample came from, for K0 and K1. "
  "The dashed line marks chance (0.50). Despite the reduced representation shift shown in DA2, "
  "domain identity remained highly predictable at 0.88-0.98 in every fold, so the alignment did not "
  "remove source information from the representation."),
 "A1": ("Attention weights by class and source",
  "Panel a: mean soft-attention weight assigned to the ResNet50 branch by true class on the "
  "canonical test set. Panel b: the same weight grouped by acquisition source. **Attention weights "
  "are descriptive diagnostics only and must not be interpreted as purely disease-specific "
  "explanations, because strong source dependence was observed**: the mean ResNet weight is 0.71 on "
  "Plus, 0.19 on FARFUM-RoP and 0.04 on Farabi, which is close to a source fingerprint. These "
  "weights were never used for model selection."),
 "A2": ("Attention distribution",
  "Distribution of the ResNet50 soft-attention weight at the selected Task 13 checkpoint, summarised "
  "by the 5th-95th percentile range with median and mean markers. Validation: mean 0.5335, "
  "sd 0.3991, p05 0.0105, median 0.5997, p95 0.9930. Test: mean 0.5111, sd 0.3893, p05 0.0087, "
  "median 0.5141, p95 0.9925. **ATTENTION_COLLAPSE = NO**: no branch mean approaches the 0.95 "
  "threshold, although the large standard deviation and extreme percentiles show that the attention "
  "is effectively near-binary per image, committing to one backbone for most images. Descriptive "
  "diagnostic only."),
 "FINAL1": ("Thesis summary",
  "Three-panel summary. Panel A: paired increments on the canonical test set. Adding five scalar "
  "biomarkers to the RGB embedding gives delta AUC +0.001085 [-0.001691, +0.003868], p = 0.431; "
  "adding the spatial vessel representation gives +0.007938 [+0.002630, +0.013131], p = 0.0031; "
  "adding the biomarkers after the vessel representation gives +0.002039 [-0.000274, +0.004397], "
  "p = 0.090. Panel B: the vessel increment replicates under source-held-out evaluation (FARFUM-RoP "
  "+0.022242, p < 0.0001; Farabi +0.012954, p = 0.0018) and under a different RGB architecture "
  "(dual-RGB M1 - M0 +0.020936, p < 0.0001). Panel C: median absolute standardized mean difference "
  "between held-out and training sources for the RGB and vessel embeddings. Headline: spatial "
  "vessel representations provide reproducible complementary predictive information across distinct "
  "RGB representations and under source-held-out evaluation, whereas five scalar vascular summaries "
  "show little incremental predictive value beyond learned image representations. This is not a "
  "claim of universal generalisation."),
 "SEG1": ("External segmentation performance - NOT AVAILABLE",
  "The external HVDROPDB segmentation benchmark table (Dice, clDice, precision, recall by RetCam and "
  "Neo) is **NOT AVAILABLE** in the frozen artifact set. A search of both servers found no "
  "authoritative per-camera Dice/clDice table; only `hvdro_evidence_lineage.csv` and "
  "`seg_current_v1_summary.json` exist, which are provenance records and do not contain these "
  "metrics. No values are reported rather than approximated."),
 "SEG2": ("Example segmentations - NOT AVAILABLE",
  "Example segmentation panels are **NOT AVAILABLE**. Producing them would require expert reference "
  "masks for the specific selected cases, and no such expert masks exist in the frozen artifact "
  "set. Per the task instruction, this figure was skipped rather than fabricated."),
 "S1T": ("Source-held-out performance table", ""),
}

FA = {
 "M1": ("نمودار کامل خط لوله مطالعه",
  "طرح‌واره‌ی خط لوله‌ی تحلیل. یک تصویر فوندوس RGB استاندارد با اندازه‌ی ۳۸۴×۳۸۴ دو بار استفاده "
  "می‌شود: یک بار توسط رمزگذار RGB از نوع EfficientNet-B5 که نمایشی ۲۰۴۸ بعدی تولید می‌کند، و یک بار "
  "توسط مدل قطعه‌بندی تثبیت‌شده که نقشه‌ی دودویی عروق را می‌سازد و رمزگذار عروقی فضایی "
  "(EfficientNet-B4) آن را به نمایشی ۱۷۹۲ بعدی تبدیل می‌کند. به‌طور موازی، اندازه‌گیری عروقی "
  "آگاه‌به‌میدان دید پنج نشانگر اسکالر تولید می‌کند. مدل‌های همجوشی این نمایش‌ها را ترکیب کرده و "
  "خروجی Normal / Pre-Plus / Plus می‌دهند. تقسیم استاندارد گروه‌مجزا (۶۲۰۳ / ۱۳۲۸ / ۱۳۳۱)، "
  "بوت‌استرپ زوجی طبقه‌بندی‌شده (seed ۴۲، ۱۰٬۰۰۰ تکرار) و پروتکل‌های ارزیابی منبع‌کنارگذاشته در "
  "شکل مشخص شده‌اند. هیچ عدد عملکردی در این شکل نمایش داده نشده است."),
 "M2": ("خط لوله‌ی پنج نشانگر اسکالر",
  "طرح‌واره‌ی مسیر نشانگرهای اسکالر. تصویر RGB استاندارد به نقشه‌ی دودویی عروق قطعه‌بندی می‌شود و "
  "پس از اندازه‌گیری عروقی آگاه‌به‌میدان دید (مخرج پوشش = مساحت محتوا) دقیقاً پنج نشانگر اسکالر "
  "تولید می‌شود: vessel_density_fov، skel_density_fov، fractal_d0، fractal_d1 و fractal_d2. این "
  "شکل یک طرح‌واره‌ی روش‌شناسی است، نه یک نتیجه‌ی آماری."),
 "M3": ("معماری مدل‌های اصلی",
  "معماری و ابعاد ویژگی مدل‌های اصلی. A: پنج نشانگر به XGBoost. B: تصویر RGB به EfficientNet-B5 با "
  "طبقه‌بند خطی. B embedding: تصویر RGB به EfficientNet-B5 و بردار ۲۰۴۸ بعدی به XGBoost. C: ۲۰۴۸ "
  "بعد RGB به‌همراه پنج نشانگر (۲۰۵۳ بعد) به XGBoost. D: نقشه‌ی عروق به EfficientNet-B4 با طبقه‌بند "
  "خطی. E: ۲۰۴۸ بعد RGB به‌همراه ۱۷۹۲ بعد عروق (۳۸۴۰ بعد) به XGBoost. G: ۲۰۴۸ بعد RGB به‌همراه "
  "۱۷۹۲ بعد عروق و پنج نشانگر (۳۸۴۵ بعد) به XGBoost. همه‌ی مدل‌های XGBoost از یک پیکربندی "
  "ابرقوسقه‌ی تثبیت‌شده استفاده می‌کنند."),
 "M4": ("نقشه‌ی پرسش‌های آزمایشی",
  "توالی نُه آزمایش و پرسش علمی واحدی که هر یک پاسخ می‌دهد. Task 6 مقایسه‌ی اصلی استاندارد است، "
  "Task 7 استنباط زوجی را فراهم می‌کند، Task 8 نمایش عروقی فضایی را معرفی می‌کند، Task 8B آن را با "
  "۱۰٬۰۰۰ تکرار آماری می‌بندد، Task 9 و 10 همجوشی یادگیرنده و شرطی‌سازی نشانگری را می‌آزمایند، "
  "Task 11 تعمیم‌پذیری منبع‌کنارگذاشته را می‌سنجد، Task 12 ناوردایی دامنه‌ی شرطی‌به‌کلاس را می‌آزماید "
  "و Task 13 بررسی می‌کند که آیا نتیجه‌ی عروقی با ستون فقرات RGB متفاوت هم پایدار می‌ماند. هیچ "
  "معیاری نمایش داده نشده است."),
 "D1": ("توزیع کلاس بر حسب منبع تصویربرداری",
  "توزیع کلاس در جمعیت نهایی موارد کامل بر حسب منبع تصویربرداری (N = ۸۸۶۲؛ ۴۱۴ گروه). منبع Plus "
  "شامل ۵۹۲۵ تصویر است که ۵۳۰۴ مورد Normal و ۶۲۱ مورد Plus است و **هیچ مورد Pre-Plus ندارد**؛ این "
  "نکته در شکل حاشیه‌نویسی شده است. FARFUM-RoP شامل ۱۵۲۸ تصویر (۷۸۰ / ۴۷۸ / ۲۷۰) و Farabi شامل "
  "۱۴۰۹ تصویر (۳۶۹ / ۴۵۲ / ۵۸۸) است. شمارش‌ها از فایل تثبیت‌شده‌ی ممیزی جمعیت خوانده شده‌اند."),
 "D2": ("توزیع کلاس در تقسیم‌های استاندارد",
  "شمارش کلاس‌ها در تقسیم‌های گروه‌مجزای آموزش، اعتبارسنجی و آزمون در جمعیت موارد کامل (آموزش ۶۲۰۳؛ "
  "اعتبارسنجی ۱۳۲۸؛ آزمون ۱۳۳۱؛ مجموع ۸۸۶۲). گروه‌مجزا بودن تأیید شد: هیچ گروهی در بیش از یک تقسیم "
  "ظاهر نمی‌شود."),
 "D3": ("سهم منابع در جمعیت استاندارد",
  "سهم هر منبع تصویربرداری در جمعیت استاندارد موارد کامل با ۸۸۶۲ تصویر. منبع Plus ۵۹۲۵ تصویر "
  "(۶۶٫۹٪)، FARFUM-RoP ۱۵۲۸ تصویر (۱۷٫۲٪) و Farabi ۱۴۰۹ تصویر (۱۵٫۹٪) تأمین می‌کند. به‌جای نمودار "
  "دایره‌ای از نمودار میله‌ای استفاده شده تا شمارش‌ها مستقیماً قابل مقایسه بمانند."),
 "R1": ("تمایزدهی مدل‌ها روی مجموعه‌ی آزمون استاندارد",
  "سطح زیر منحنی راک ماکرو یک‌دربرابر‌بقیه روی مجموعه‌ی آزمون استاندارد (N = ۱۳۳۱) برای مدل‌های "
  "اصلی. مقادیر، نتایج تثبیت‌شده‌ی Task 6 و Task 8 هستند. محور افقی از ۰٫۶۰ آغاز می‌شود تا تفاوت‌ها "
  "خوانا بمانند؛ مقیاس مطلق به‌صراحت برچسب‌گذاری شده و محور پایین‌تر از کمترین مقدار رسم‌شده بریده "
  "نشده است. مدل F_RGB_VESSEL_LATE_FUSION حذف شده است زیرا ضریب اختلاط آن روی alpha = ۱٫۰ اشباع شد و "
  "جدول احتمال آن بیت‌به‌بیت با B_EMBEDDING_ONLY یکسان است."),
 "R2": ("مقایسه‌ی چندمعیاره روی مجموعه‌ی آزمون استاندارد",
  "پنج پنل جداگانه برای AUC چندکلاسه، دقت متعادل، F1 ماکرو، امتیاز Brier و خطای کالیبراسیون انتظاری "
  "(ECE، ۱۵ سبد با عرض برابر) روی مجموعه‌ی آزمون استاندارد (N = ۱۳۳۱). AUC، دقت متعادل و F1 ماکرو "
  "بزرگ‌تر‌بهتر هستند؛ Brier و ECE کوچک‌تر‌بهتر. معیارها عمداً در هیچ امتیاز ترکیبی ادغام نشده‌اند."),
 "R3": ("AUC هر کلاس",
  "AUC یک‌دربرابر‌بقیه برای هر کلاس به‌صورت جداگانه روی مجموعه‌ی آزمون استاندارد (N = ۱۳۳۱؛ Normal "
  "۹۸۲، Pre-Plus ۱۴۰، Plus ۲۰۹). Pre-Plus کوچک‌ترین کلاس و سخت‌ترین کلاس برای همه‌ی مدل‌هاست. مقادیر "
  "نتایج تثبیت‌شده‌ی Task 6 و Task 8 هستند."),
 "C1": ("اطلاعات افزایشی روی نمایش تثبیت‌شده‌ی RGB",
  "تفاضل‌های بوت‌استرپ زوجی طبقه‌بندی‌شده در AUC چندکلاسه روی مجموعه‌ی آزمون استاندارد (N = ۱۳۳۱، "
  "seed ۴۲، ۱۰٬۰۰۰ تکرار). سه افزایش نسبت به نمایش ۲۰۴۸ بعدی RGB نشان داده شده است: افزودن پنج "
  "نشانگر اسکالر (C - B)، افزودن نمایش عروقی فضایی (E - B) و افزودن پنج نشانگر اسکالر پس از نمایش "
  "عروقی (G - E). خط‌چین، صفر را نشان می‌دهد. بازه‌های مشکی صفر را شامل نمی‌شوند و بازه‌های خاکستری "
  "شامل می‌شوند. بازه‌های C - B و G - E هر دو صفر را در بر می‌گیرند، در حالی که E - B صفر را به‌طور "
  "مثبت کنار می‌گذارد."),
 "C2": ("خلاصه‌ی مکمل‌بودن",
  "خلاصه‌ی تک‌شکلی آنچه هر نمایش به نمایش ۲۰۴۸ بعدی RGB می‌افزاید. افزودن پنج نشانگر اسکالر (C) "
  "تفاضل AUC برابر +۰٫۰۰۱۰۸۵ با بازه‌ی اطمینان ۹۵٪ [−۰٫۰۰۱۶۹۱، +۰٫۰۰۳۸۶۸] (p = ۰٫۴۳۱) می‌دهد: بدون "
  "افزایش پشتیبانی‌شده. افزودن نمایش عروقی فضایی (E) تفاضل AUC برابر +۰٫۰۰۷۹۳۸ با بازه‌ی "
  "[+۰٫۰۰۲۶۳۰، +۰٫۰۱۳۱۳۱] (p = ۰٫۰۰۳۱) می‌دهد: افزایش پشتیبانی‌شده. افزودن پنج نشانگر اسکالر پس از "
  "نمایش عروقی (G) تفاضل AUC برابر +۰٫۰۰۲۰۳۹ با بازه‌ی [−۰٫۰۰۰۲۷۴، +۰٫۰۰۴۳۹۷] (p = ۰٫۰۹۰) می‌دهد: "
  "بدون افزایش پشتیبانی‌شده. مدل‌های C و G اکتشافی ثانویه هستند."),
 "C3": ("مکمل‌بودن عروق در دو استخراج‌کننده‌ی متفاوت ویژگی RGB",
  "بازه‌های اطمینان زوجی ۹۵٪ برای تغییر AUC چندکلاسه هنگامی که همان نمایش تثبیت‌شده‌ی ۱۷۹۲ بعدی عروق "
  "به دو نمایش RGB متفاوت افزوده می‌شود، روی مجموعه‌ی آزمون استاندارد (N = ۱۳۳۱، ۱۰٬۰۰۰ تکرار زوجی "
  "طبقه‌بندی‌شده، seed ۴۲). با نمایش اصلی تک‌ستون‌فقرات EfficientNet-B5 افزایش برابر +۰٫۰۰۷۹۳۸ "
  "[+۰٫۰۰۲۶۳۰، +۰٫۰۱۳۱۳۱] و p = ۰٫۰۰۳۱ است؛ با نمایش دو‌ستون‌فقرات توجهی (Task 13) برابر +۰٫۰۲۰۹۳۶ "
  "[+۰٫۰۱۱۴۲۶، +۰٫۰۳۰۸۲۸] و p < ۰٫۰۰۰۱ است. این شکل نشان می‌دهد سهم عروقی به یک استخراج‌کننده‌ی "
  "خاص RGB محدود نیست؛ اما **به این معنا نیست** که مدل دو‌ستون‌فقرات خودش برتر است — در واقع کنترل "
  "فقط‌RGB آن از نمایش اصلی ضعیف‌تر بود."),
 "P1": ("ماتریس‌های درهم‌ریختگی",
  "ماتریس‌های درهم‌ریختگی نرمال‌شده‌ی سطری روی مجموعه‌ی آزمون استاندارد (N = ۱۳۳۱) با شمارش خام در "
  "پرانتز و ترتیب کلاس‌های یکسان در همه‌ی پنل‌ها: Normal، Pre-Plus، Plus. سطرها کلاس واقعی و ستون‌ها "
  "کلاس پیش‌بینی‌شده هستند، بنابراین هر سطر به ۱۰۰٪ جمع می‌شود. Pre-Plus کوچک‌ترین و پرخطاترین کلاس "
  "در همه‌ی مدل‌هاست. پنل‌های مستقل نیز به‌صورت فایل‌های جداگانه ارائه شده‌اند."),
 "P2": ("نمودارهای اطمینان‌پذیری",
  "نمودارهای اطمینان‌پذیری روی مجموعه‌ی آزمون استاندارد با همان ۱۵ سبد با عرض برابر که در ECE "
  "گزارش‌شده استفاده شده است. خط‌چین مورب، کالیبراسیون کامل است. مقدار ECE برای هر مدل در راهنما "
  "آمده است. هر سه مدل در اطمینان پایین کم‌اطمینان و در اطمینان بالا پراطمینان هستند و هیچ‌کدام روی "
  "خط مورب قرار نمی‌گیرند."),
 "B1": ("شش آزمون مستقل پنج نشانگر اسکالر",
  "نمودار جنگلی تغییر AUC منتسب به پنج نشانگر اسکالر، به‌صورت جداگانه برای هر آزمایشی که در آن "
  "آزموده شده‌اند، همراه با بازه‌های اطمینان زوجی ۹۵٪ از بوت‌استرپ‌های تثبیت‌شده‌ی ۱۰٬۰۰۰ تکراره "
  "(seed ۴۲). سطرها بر حسب نوع تحلیل برچسب‌گذاری شده‌اند: تقسیم استاندارد، مدل عصبی مشترک، "
  "شرطی‌سازی FiLM و فولدهای منبع‌کنارگذاشته. سطر آخر از AUC دودویی محدود Normal در برابر Plus "
  "استفاده می‌کند زیرا منبع Plus هیچ مورد Pre-Plus ندارد و بنابراین مستقیماً با سطرهای سه‌کلاسه "
  "قابل مقایسه نیست. هیچ اثر سازگاری از نظر آماری پشتیبانی نمی‌شود؛ تنها بازه‌ای که صفر را کنار "
  "می‌گذارد منفی است (مدل مشترک)."),
 "B2": ("تشخیص‌های FiLM",
  "پنل الف: تعدیل یادگرفته‌شده‌ی FiLM روی مجموعه‌ی آزمون. پارامترسازی باقی‌مانده‌ی کراندار "
  "gamma = ۱ + ۰٫۱۰·tanh(delta_gamma) و beta = ۰٫۱۰·tanh(beta_raw) مقادیر gamma با میانگین "
  "۰٫۹۹۹۵۸۰، میانه ۰٫۹۹۹۶۲۴ و دامنه‌ی صدک ۵ تا ۹۵ برابر [۰٫۹۹۶۹۹۷، ۱٫۰۰۱۹۲۴] و beta با میانگین "
  "−۰٫۰۰۰۰۰۶ و دامنه‌ی [−۰٫۰۰۲۶۹۵، +۰٫۰۰۲۶۱۶] تولید کرد. خطوط چین، همانی (gamma = ۱، beta = ۰) را "
  "نشان می‌دهند. پنل ب: میانگین و صدک ۹۵ تغییر مطلق در احتمالات تثبیت‌شده‌ی J1 روی آزمون، هنگامی که "
  "هر نشانگر به‌تنهایی با میانگین آموزشی خود جایگزین می‌شود. هر پنج نشانگر تغییراتی در حدود "
  "۲×۱۰⁻⁵ و صفر تغییر آرگ‌مکس ایجاد می‌کنند، بنابراین شرطی‌ساز عملاً مانند نگاشت همانی عمل کرده است."),
 "N1": ("منحنی‌های یادگیری Task 9",
  "خطای آموزش و اعتبارسنجی و AUC اعتبارسنجی به‌ازای هر دوره برای دو مدل مشترک Task 9، با مرز "
  "آزادسازی رمزگذار در دوره‌ی ۳٫۵ و دوره‌ی انتخاب‌شده مشخص‌شده. پس از آزادسازی، خطای آموزش یکنوا "
  "کاهش می‌یابد در حالی که خطای اعتبارسنجی به‌شدت بالا می‌رود و AUC اعتبارسنجی افت می‌کند؛ نشانه‌ی "
  "بیش‌برازش شدید. هر دو مدل دوره‌ای را در فاصله‌ی دو دوره پس از آزادسازی انتخاب کردند. پنل پ برای "
  "مرجع، AUC و F1 ماکرو روی مجموعه‌ی آزمون استاندارد را نشان می‌دهد. این‌ها آزمایش‌های اکتشافی "
  "ثانویه‌اند، نه مدل‌های اصلی."),
 "N2": ("منحنی‌های یادگیری Task 10",
  "خطای آموزش و اعتبارسنجی و AUC اعتبارسنجی به‌ازای هر دوره برای مدل‌های نمایش تثبیت‌شده‌ی Task 10، "
  "با دوره‌ی انتخاب‌شده مشخص‌شده. خطای آموزش یکنوا کاهش می‌یابد در حالی که خطای اعتبارسنجی از کمینه‌ی "
  "دوره‌ی اول بالا می‌رود، بنابراین هر دو مدل دوره‌ی ۱ را انتخاب کردند. حتی با تثبیت هر دو شبکه‌ی "
  "عصبی کانولوشنی در قالب ویژگی‌های پیش‌محاسبه‌شده، سری ۲٫۵ میلیون پارامتری روی ۶۲۰۳ نمونه‌ی آموزش "
  "بیش‌برازش می‌کند. آزمایش اکتشافی ثانویه."),
 "N3": ("منحنی‌های یادگیری Task 13",
  "پنل الف: خطای آموزش و اعتبارسنجی به‌ازای هر دوره برای مدل دو‌RGB سبک ROPDeepX، با مرز آزادسازی "
  "ستون فقرات در دوره‌ی ۳٫۵ و دوره‌ی انتخاب‌شده‌ی ۲ مشخص‌شده. پنل ب: AUC و F1 ماکرو اعتبارسنجی با "
  "دوره‌ی کمینه‌ی خطای اعتبارسنجی (۷) مشخص‌شده. خطای آموزش یکنوا از ۰٫۷۶۳۸ به ۰٫۵۵۷۴ کاهش می‌یابد "
  "در حالی که AUC اعتبارسنجی در دوره‌ی ۲ (۰٫۸۶۶۲۴) به اوج می‌رسد و تا دوره‌ی ۷ به ۰٫۸۴۰۱۴ افت "
  "می‌کند. پنل پ: میانگین وزن توجه در هر دوره با آستانه‌های فروپاشی ۰٫۰۵ و ۰٫۹۵؛ هیچ فروپاشی رخ "
  "نداد. آزمایش اکتشافی ثانویه."),
 "R4": ("نتایج سبک ROPDeepX در Task 13",
  "معیارهای مجموعه‌ی آزمون استاندارد (N = ۱۳۳۱) برای مدل‌های Task 13 در برابر مدل‌های مرجع "
  "تثبیت‌شده، در پنج پنل جداگانه برای AUC چندکلاسه، دقت متعادل، F1 ماکرو، Brier و ECE. مدل L یک سر "
  "عصبی آموزش‌دیده روی نمایش توجهی دو‌RGB است، M0 یک XGBoost روی نمایش تثبیت‌شده‌ی ۱۰۲۴ بعدی RGB "
  "توجهی است و M1 نمایش تثبیت‌شده‌ی ۱۷۹۲ بعدی عروق را به M0 می‌افزاید. طبقه‌بند خطی خود L به‌وضوح از "
  "XGBoost روی همان نمایش L ضعیف‌تر است. L مستقیماً با مدل‌های مبتنی بر نمایش قابل مقایسه نیست "
  "زیرا یک طبقه‌بند سرتاسری آموزش‌دیده است، نه یک یادگیرنده‌ی پایین‌دستی روی ویژگی‌های تثبیت‌شده."),
 "R5": ("مقایسه‌های زوجی AUC در Task 13",
  "تفاضل‌های بوت‌استرپ زوجی طبقه‌بندی‌شده در AUC چندکلاسه روی مجموعه‌ی آزمون استاندارد (N = ۱۳۳۱، "
  "۱۰٬۰۰۰ تکرار، seed ۴۲) برای چهار مقایسه‌ی Task 13. مقدار M0 - B به‌طور معنادار منفی است و نشان "
  "می‌دهد نمایش RGB دو‌ستون‌فقرات توجهی از نمایش اصلی EfficientNet-B5 بدتر بود؛ M1 - M0 به‌طور "
  "معنادار مثبت است و نشان می‌دهد نمایش عروقی همچنان مکمل باقی می‌ماند؛ M1 - E و M1 - G بی‌اثر "
  "هستند، بنابراین خط لوله‌ی دو‌RGB از نظر آماری از مدل‌های اصلی RGB + عروق قابل تفکیک نیست."),
 "S1": ("عملکرد با منبع کنارگذاشته",
  "پنل الف: AUC ماکرو سه‌کلاسه روی هر منبع کنارگذاشته برای مدل‌های LOSO، که در آن منبع کنارگذاشته "
  "هیچ سهمی در برازش یا پیش‌پردازش نداشته است. پنل ب از محور متفاوتی استفاده می‌کند و به‌صراحت به‌عنوان "
  "AUC دودویی **محدود Normal در برابر Plus** برچسب‌گذاری شده است، زیرا منبع Plus هیچ مورد Pre-Plus "
  "ندارد و AUC سه‌کلاسه در آن تعریف نشده است. دو پنل نباید روی یک محور مقایسه شوند. محور پنل ب بازه‌ی "
  "۰٫۹۵ تا ۱٫۰۰ را پوشش می‌دهد تا سه مدل قابل تفکیک باشند؛ دامنه‌ی مطلق برچسب‌گذاری شده است."),
 "S2": ("مکمل‌بودن عروق در ارزیابی با منبع کنارگذاشته",
  "بازه‌های اطمینان زوجی ۹۵٪ برای تغییر AUC هنگام افزودن نمایش عروقی فضایی (E - B)، که به‌صورت "
  "جداگانه در هر فولد منبع‌کنارگذاشته با ۱۰٬۰۰۰ تکرار زوجی طبقه‌بندی‌شده (seed ۴۲) محاسبه شده است. "
  "دو سطر اول AUC ماکرو سه‌کلاسه هستند؛ سطر سوم AUC دودویی محدود Normal در برابر Plus برای فولد Plus "
  "است و به‌همین‌صورت برچسب‌گذاری شده. خط‌چین، صفر را نشان می‌دهد. هر دو فولد سه‌کلاسه‌ی قابل مقایسه، "
  "صفر را به‌طور مثبت کنار می‌گذارند."),
 "S3": ("افزایش نشانگری در ارزیابی با منبع کنارگذاشته",
  "بازه‌های اطمینان زوجی ۹۵٪ برای تغییر AUC هنگام افزودن پنج نشانگر اسکالر پس از نمایش عروقی فضایی "
  "(G - E)، که در هر فولد منبع‌کنارگذاشته با ۱۰٬۰۰۰ تکرار زوجی طبقه‌بندی‌شده (seed ۴۲) محاسبه شده "
  "است. همانند S2، سطر Plus همان AUC دودویی محدود Normal در برابر Plus است. همه‌ی بازه‌ها صفر را در "
  "بر می‌گیرند، بنابراین هیچ افزایش نشانگری تحت جابه‌جایی منبع پشتیبانی نمی‌شود."),
 "DS1": ("جابه‌جایی دامنه‌ی نمایش‌ها",
  "جابه‌جایی توصیفی میان هر منبع کنارگذاشته و منابع آموزشی تجمیع‌شده برای نمایش‌های تثبیت‌شده‌ی RGB "
  "(۲۰۴۸ بعد) و عروق (۱۷۹۲ بعد)، بر حسب تفاوت میانگین استانداردشده‌ی هر ویژگی "
  "SMD = (میانگین هدف − میانگین آموزش) / انحراف معیار آموزش. میانه و صدک‌های ۹۰ و ۹۵ قدر مطلق SMD "
  "نمایش داده شده‌اند. هیچ برچسب هدفی استفاده نشده است. نمایش عروقی در هر فولد بیشتر از نمایش RGB "
  "جابه‌جا شده و Plus در هر دو بلوک جابه‌جاشده‌ترین منبع است."),
 "DS2": ("فاصله‌ی مرکزوار استانداردشده",
  "فاصله‌ی مرکزوار استانداردشده میان هر منبع کنارگذاشته و منابع آموزشی تجمیع‌شده در فضای نمایش‌های "
  "تثبیت‌شده‌ی RGB و عروق، محاسبه‌شده به‌صورت نُرم اقلیدسی تفاوت‌های میانگین استانداردشده‌ی هر ویژگی. "
  "Plus در هر دو نمایش دورترین منبع است؛ Farabi از FARFUM-RoP دورتر است. صرفاً یک تشخیص توصیفی."),
 "DS3": ("جابه‌جایی منبع در نشانگرهای اسکالر",
  "نقشه‌ی حرارتی تفاوت میانگین استانداردشده‌ی هر یک از پنج نشانگر اسکالر میان منبع کنارگذاشته و منابع "
  "آموزشی تجمیع‌شده. مقیاس رنگی واگرا و مرکز آن صفر است؛ مقادیر دقیق حاشیه‌نویسی شده‌اند. هر پنج "
  "نشانگر در هر فولد به‌شدت وابسته به منبع هستند، با قدر مطلق SMD بین ۰٫۳۰ و ۱٫۱۱، یعنی تا یک انحراف "
  "معیار کامل جابه‌جایی در داده‌ی آموزش. Plus در همه‌ی پنج نشانگر بالاترین و Farabi پایین‌ترین است. "
  "این اطلاعات برای بازتنظیم هیچ مدلی استفاده نشد."),
 "DA1": ("عملکرد هم‌ترازی دامنه در Task 12",
  "پنل الف: AUC ماکرو سه‌کلاسه روی هر منبع کنارگذاشته برای جفت هم‌تای Task 12، یعنی K0 (کنترل "
  "بی‌طرف نسبت به دامنه) و K1 (DANN شرطی‌به‌کلاس + MMD). پنل ب از محور متفاوتی استفاده می‌کند و AUC "
  "دودویی **محدود Normal در برابر Plus** را برای فولد Plus گزارش می‌کند، جایی که AUC سه‌کلاسه تعریف "
  "نشده است. منبع کنارگذاشته هرگز در آموزش، اعتبارسنجی، هم‌ترازی، توقف زودهنگام یا انتخاب مدل "
  "استفاده نشد. K1 در هیچ‌یک از دو فولد قابل مقایسه نسبت به K0 بهبود نیافت."),
 "DA2": ("جابه‌جایی نمایش پیش و پس از هم‌ترازی",
  "فاصله‌ی مرکزوار استانداردشده و میانه‌ی قدر مطلق تفاوت میانگین استانداردشده میان دو دامنه‌ی آموزشی، "
  "اندازه‌گیری‌شده روی تقسیم اعتبارسنجی منبع در نمایش یادگرفته‌شده‌ی ۲۵۶ بعدی برای K0 و K1. هم‌ترازی "
  "هر دو آماره را در هر سه فولد کاهش داد، با بیشترین کاهش در فولد Farabi. هیچ اطلاعاتی از منبع "
  "کنارگذاشته وارد این تشخیص نشد."),
 "DA3": ("قابلیت پیش‌بینی دامنه پس از هم‌ترازی",
  "دقت اعتبارسنجی متقابل پنج‌تایی یک رگرسیون لجستیک که روی نمایش تقسیم اعتبارسنجی منبع آموزش "
  "دیده و پیش‌بینی می‌کند هر نمونه از کدام یک از دو دامنه‌ی آموزشی آمده است، برای K0 و K1. خط‌چین، "
  "سطح شانس (۰٫۵۰) را نشان می‌دهد. با وجود کاهش جابه‌جایی نمایش که در DA2 نشان داده شد، هویت دامنه "
  "در هر فولد با دقت ۰٫۸۸ تا ۰٫۹۸ به‌شدت قابل پیش‌بینی ماند، بنابراین هم‌ترازی اطلاعات منبع را از "
  "نمایش حذف نکرد."),
 "A1": ("وزن‌های توجه بر حسب کلاس و منبع",
  "پنل الف: میانگین وزن توجه نرم اختصاص‌یافته به شاخه‌ی ResNet50 بر حسب کلاس واقعی روی مجموعه‌ی آزمون "
  "استاندارد. پنل ب: همان وزن، گروه‌بندی‌شده بر حسب منبع تصویربرداری. **وزن‌های توجه فقط تشخیص‌های "
  "توصیفی هستند و نباید به‌عنوان توضیح صرفاً بیماری‌محور تفسیر شوند، زیرا وابستگی شدید به منبع مشاهده "
  "شد**: میانگین وزن ResNet روی Plus برابر ۰٫۷۱، روی FARFUM-RoP برابر ۰٫۱۹ و روی Farabi برابر ۰٫۰۴ "
  "است که به یک اثر انگشت منبع نزدیک است. این وزن‌ها هرگز برای انتخاب مدل استفاده نشدند."),
 "A2": ("توزیع توجه",
  "توزیع وزن توجه نرم ResNet50 در نقطه‌ی کنترل انتخاب‌شده‌ی Task 13، خلاصه‌شده با دامنه‌ی صدک ۵ تا ۹۵ "
  "و نشانگرهای میانه و میانگین. اعتبارسنجی: میانگین ۰٫۵۳۳۵، انحراف معیار ۰٫۳۹۹۱، صدک ۵ برابر "
  "۰٫۰۱۰۵، میانه ۰٫۵۹۹۷، صدک ۹۵ برابر ۰٫۹۹۳۰. آزمون: میانگین ۰٫۵۱۱۱، انحراف معیار ۰٫۳۸۹۳، صدک ۵ "
  "برابر ۰٫۰۰۸۷، میانه ۰٫۵۱۴۱، صدک ۹۵ برابر ۰٫۹۹۲۵. **ATTENTION_COLLAPSE = NO**: میانگین هیچ شاخه‌ای "
  "به آستانه‌ی ۰٫۹۵ نزدیک نمی‌شود، هرچند انحراف معیار بزرگ و صدک‌های حدی نشان می‌دهند توجه عملاً برای "
  "هر تصویر نزدیک به دودویی است و برای بیشتر تصاویر به یک ستون فقرات متعهد می‌شود. صرفاً تشخیص "
  "توصیفی."),
 "FINAL1": ("خلاصه‌ی پایان‌نامه",
  "خلاصه‌ی سه‌پنلی. پنل الف: افزایش‌های زوجی روی مجموعه‌ی آزمون استاندارد. افزودن پنج نشانگر اسکالر "
  "به نمایش RGB تفاضل AUC برابر +۰٫۰۰۱۰۸۵ [−۰٫۰۰۱۶۹۱، +۰٫۰۰۳۸۶۸] با p = ۰٫۴۳۱ می‌دهد؛ افزودن نمایش "
  "عروقی فضایی برابر +۰٫۰۰۷۹۳۸ [+۰٫۰۰۲۶۳۰، +۰٫۰۱۳۱۳۱] با p = ۰٫۰۰۳۱؛ و افزودن نشانگرها پس از نمایش "
  "عروقی برابر +۰٫۰۰۲۰۳۹ [−۰٫۰۰۰۲۷۴، +۰٫۰۰۴۳۹۷] با p = ۰٫۰۹۰. پنل ب: افزایش عروقی در ارزیابی با "
  "منبع کنارگذاشته (FARFUM-RoP برابر +۰٫۰۲۲۲۴۲ با p < ۰٫۰۰۰۱؛ Farabi برابر +۰٫۰۱۲۹۵۴ با p = ۰٫۰۰۱۸) "
  "و با معماری RGB متفاوت (M1 - M0 دو‌RGB برابر +۰٫۰۲۰۹۳۶ با p < ۰٫۰۰۰۱) تکرار می‌شود. پنل پ: میانه‌ی "
  "قدر مطلق تفاوت میانگین استانداردشده میان منبع کنارگذاشته و منابع آموزش برای نمایش‌های RGB و عروق. "
  "پیام اصلی: نمایش‌های عروقی فضایی اطلاعات پیش‌بینی‌کننده‌ی مکمل و تکرارپذیری فراهم می‌کنند — هم در "
  "معماری‌های متفاوت RGB و هم در ارزیابی با منبع کنارگذاشته — در حالی که پنج خلاصه‌ی عروقی اسکالر "
  "ارزش پیش‌بینی‌کننده‌ی افزایشی اندکی فراتر از نمایش‌های تصویری یادگرفته‌شده نشان می‌دهند. این یک "
  "ادعای تعمیم‌پذیری جهانی نیست."),
 "SEG1": ("عملکرد قطعه‌بندی خارجی — در دسترس نیست",
  "جدول معیارهای قطعه‌بندی خارجی HVDROPDB (Dice، clDice، دقت، بازخوانی به تفکیک RetCam و Neo) در "
  "مجموعه‌ی آثار تثبیت‌شده **در دسترس نیست**. جست‌وجو در هر دو سرور هیچ جدول معتبر Dice/clDice "
  "به‌تفکیک دوربین پیدا نکرد؛ تنها `hvdro_evidence_lineage.csv` و `seg_current_v1_summary.json` "
  "وجود دارند که سوابق منشأ هستند و این معیارها را در بر نمی‌گیرند. هیچ مقداری به‌جای تقریب گزارش "
  "نشده است."),
 "SEG2": ("نمونه‌های قطعه‌بندی — در دسترس نیست",
  "پنل‌های نمونه‌ی قطعه‌بندی **در دسترس نیستند**. تولید آن‌ها به ماسک‌های مرجع متخصص برای موارد "
  "انتخاب‌شده نیاز دارد و چنین ماسک‌های متخصصی در مجموعه‌ی آثار تثبیت‌شده وجود ندارد. طبق دستور کار، "
  "این شکل به‌جای ساختگی‌سازی حذف شد."),
}

TABLE_FA = {
 "D1_cohort_summary": ("جدول خلاصه‌ی جمعیت",
  "خلاصه‌ی جمعیت نهایی موارد کامل به تفکیک منبع تصویربرداری: تعداد تصاویر، تعداد گروه‌ها و توزیع "
  "کلاس‌ها. منبع Plus هیچ مورد Pre-Plus ندارد. درصدها نسبت به کل ۸۸۶۲ تصویر محاسبه شده‌اند."),
 "D2_canonical_split_summary": ("جدول خلاصه‌ی تقسیم استاندارد",
  "شمارش کلاس‌ها و گروه‌ها در تقسیم‌های گروه‌مجزای آموزش، اعتبارسنجی و آزمون. هیچ گروهی در بیش از یک "
  "تقسیم ظاهر نمی‌شود."),
 "R1_master_model_results": ("جدول جامع عملکرد مدل‌ها",
  "عملکرد مدل‌های اصلی روی مجموعه‌ی آزمون استاندارد. AUC، دقت متعادل و F1 ماکرو بزرگ‌تر‌بهتر و Brier "
  "و ECE کوچک‌تر‌بهتر هستند. عدد پررنگ بهترین مقدار هر ستون است؛ مدل‌ها با هیچ امتیاز ترکیبی رتبه‌بندی "
  "نشده‌اند."),
 "R2_per_class_auc": ("جدول AUC هر کلاس",
  "AUC یک‌دربرابر‌بقیه برای هر کلاس روی مجموعه‌ی آزمون استاندارد (N = ۱۳۳۱؛ Normal ۹۸۲، Pre-Plus ۱۴۰، "
  "Plus ۲۰۹)."),
 "C1_primary_paired_statistics": ("جدول آمار زوجی اصلی",
  "آمار زوجی اصلی روی مجموعه‌ی آزمون استاندارد: بوت‌استرپ زوجی طبقه‌بندی‌شده با seed ۴۲ و ۱۰٬۰۰۰ "
  "تکرار، N = ۱۳۳۱. مقایسه‌ی C در برابر B از Task 7 و مقایسه‌های E در برابر B و G در برابر E از "
  "بستن آماری Task 8B هستند."),
 "C2_architecture_robustness": ("جدول استواری معماری",
  "استواری سهم عروقی نسبت به معماری RGB. هر سطر همان نمایش تثبیت‌شده‌ی ۱۷۹۲ بعدی عروق را به یک نمایش "
  "RGB متفاوت با کنترل هم‌تای خودش اضافه می‌کند. دو سطر دو آزمایش مستقل‌اند، نه یک مقایسه‌ی مدل واحد."),
 "B1_biomarker_evidence_summary": ("جدول شواهد نشانگرها",
  "خلاصه‌ی شواهد برای پنج نشانگر اسکالر در همه‌ی آزمایش‌هایی که آزموده شدند. سطر آخر از AUC دودویی "
  "محدود Normal در برابر Plus استفاده می‌کند زیرا منبع Plus هیچ مورد Pre-Plus ندارد. زبان تفسیر "
  "کنترل‌شده است و به‌معنای بی‌فایده بودن نشانگرها نیست."),
 "N1_negative_architecture_experiments": ("جدول آزمایش‌های معماری منفی",
  "آزمایش‌های عصبی اکتشافی ثانویه که نسبت به مدل‌های همجوشی الحاقی تثبیت‌شده بهبود نیافتند. مقادیر "
  "معیارهای مجموعه‌ی آزمون استاندارد هستند و هیچ‌یک از این مدل‌ها نتیجه‌ی اصلی نیست."),
 "S1_source_heldout_performance": ("جدول عملکرد با منبع کنارگذاشته",
  "عملکرد LOSO. منبع کنارگذاشته هیچ سهمی در برازش، پیش‌پردازش یا انتخاب مدل نداشته است. AUC سه‌کلاسه "
  "برای منبع Plus تعریف‌نشده است زیرا هیچ مورد Pre-Plus ندارد؛ AUC دودویی محدود Normal در برابر Plus "
  "به‌صورت جداگانه و نه به‌عنوان جایگزین گزارش شده است."),
 "DS1_domain_shift_summary": ("جدول خلاصه‌ی جابه‌جایی دامنه",
  "تشخیص‌های توصیفی جابه‌جایی دامنه برای نمایش‌های تثبیت‌شده. هیچ برچسب هدفی در این آماره‌ها استفاده "
  "نشده است."),
 "DA1_task12_results": ("جدول نتایج Task 12",
  "نتایج ناوردایی دامنه‌ی شرطی‌به‌کلاس. K0 کنترل هم‌تای بی‌طرف نسبت به دامنه است و K1 متمایزکننده‌ی "
  "دامنه با GRL و MMD شرطی‌به‌کلاس را می‌افزاید. سطر Plus برای خط AUC، AUC دودویی محدود Normal در "
  "برابر Plus را گزارش می‌کند."),
 "FINAL1_experiment_summary": ("جدول خلاصه‌ی کامل آزمایش‌ها",
  "خلاصه‌ی کامل آزمایش‌ها برای مطالعات تقسیم استاندارد و منبع‌کنارگذاشته. نتایج تاریخی فاز ۵ طبق ساختار "
  "حذف شده‌اند."),
}

# ---------------------------------------------------------------- write captions
def captions():
    def block(d, title):
        o = [f"# {title}", "", f"Generated {GEN} from frozen artifacts only.", ""]
        order = ["M1", "M2", "M3", "M4", "D1", "D2", "D3", "R1", "R2", "R3", "R4", "R5",
                 "C1", "C2", "C3", "P1", "P2", "B1", "B2", "N1", "N2", "N3",
                 "S1", "S2", "S3", "DS1", "DS2", "DS3", "DA1", "DA2", "DA3",
                 "A1", "A2", "FINAL1", "SEG1", "SEG2"]
        for k in order:
            if k not in d:
                continue
            t, body = d[k]
            o += [f"## Figure {k} — {t}", "", body, ""]
        return "\n".join(o)
    en = block(EN, "Figure and table captions (English)")
    en += "\n## Table captions\n\n"
    for k, (t, body) in EN_TABLES.items():
        en += f"### Table {k} — {t}\n\n{body}\n\n"
    (CAP / "captions_en.md").write_text(en, encoding="utf-8")
    fa = block(FA, "شرح شکل‌ها و جدول‌ها (فارسی)")
    fa += "\n## شرح جدول‌ها\n\n"
    for k, (t, body) in TABLE_FA.items():
        fa += f"### جدول {k} — {t}\n\n{body}\n\n"
    (CAP / "captions_fa.md").write_text(fa, encoding="utf-8")
    log("captions written")


EN_TABLES = {
 "D1_cohort_summary": ("Cohort summary",
  "Complete-case cohort by acquisition source: images, groups, class counts and percentage of the "
  "8,862-image total. The Plus source contains no Pre-Plus, which is why the three-class metric is "
  "undefined for it in every held-out analysis."),
 "D2_canonical_split_summary": ("Canonical split summary",
  "Class and group counts within the canonical group-disjoint training, validation and test splits. "
  "The split is group-disjoint by construction; no group appears in more than one split."),
 "R1_master_model_results": ("Master model results",
  "Canonical test-set performance (N = 1,331) for the primary models. AUC, balanced accuracy and "
  "macro F1 are higher-is-better; Brier and ECE are lower-is-better. Bold marks the numerically best "
  "value in each metric column; no composite score or ranking is used. F_RGB_VESSEL_LATE_FUSION is "
  "listed for completeness and is identical to B_EMBEDDING_ONLY because its mixture coefficient "
  "saturated at alpha = 1.0."),
 "R2_per_class_auc": ("Per-class AUC",
  "One-vs-rest AUC per class on the canonical test set (N = 1,331: Normal 982, Pre-Plus 140, Plus "
  "209). Frozen Task 6 and Task 8 results."),
 "C1_primary_paired_statistics": ("Primary paired statistics",
  "Paired class-stratified bootstrap statistics on the canonical test set (seed 42, 10,000 "
  "replicates, N = 1,331). C vs B is the Task 7 comparison; E vs B and G vs E are the Task 8B "
  "10,000-replicate closure."),
 "C2_architecture_robustness": ("Architecture robustness of the vessel contribution",
  "The same frozen 1,792-d vessel embedding added to two different RGB representations, each with "
  "its own matched RGB-only control and its own paired bootstrap. The two rows are independent "
  "experiments, not one model comparison. This is a main table for the thesis."),
 "B1_biomarker_evidence_summary": ("Biomarker evidence summary",
  "Every experiment in which the five scalar biomarkers were tested, with the frozen paired delta, "
  "95% CI and p value. Interpretation uses controlled language: supported positive increment, no "
  "statistically supported increment, or statistically supported decrease. The final row uses the "
  "restricted Normal-vs-Plus binary AUC. This table does not support the statement that biomarkers "
  "are useless."),
 "N1_negative_architecture_experiments": ("Negative architecture experiments",
  "Secondary exploratory neural models that did not improve on the frozen concatenation models "
  "(canonical test set, N = 1,331). These are not primary results and are reported for transparency."),
 "S1_source_heldout_performance": ("Source-held-out performance",
  "LOSO performance. The held-out source contributed nothing to fitting, preprocessing or model "
  "choice. Three-class AUC is undefined for Plus (no Pre-Plus); the restricted Normal-vs-Plus binary "
  "AUC is reported separately and is not a substitute for the three-class metric."),
 "DS1_domain_shift_summary": ("Domain shift summary",
  "Descriptive shift statistics for the frozen RGB and vessel embeddings, computed without target "
  "labels."),
 "DA1_task12_results": ("Task 12 results",
  "Class-conditional domain-invariance results. K0 is the matched domain-neutral control and K1 adds "
  "the GRL domain discriminator and class-conditional MMD with coefficients fixed at 0.10. The Plus "
  "row reports the restricted Normal-vs-Plus binary AUC for the AUC line."),
 "FINAL1_experiment_summary": ("Complete experiment summary",
  "One row per experiment. Historical Phase-5 results are excluded by construction. No claim is made "
  "beyond what the frozen statistics support."),
}

if __name__ == "__main__":
    tables()
    captions()
    log("PART 2 core done")
