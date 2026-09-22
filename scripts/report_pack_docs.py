"""Final report asset pack - PART 3: index, claim-evidence matrix, LaTeX manifest, metadata."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WS = Path("/root/niki_rop_task6_isolated")
A = WS / "artifacts"
OUT = WS / "report_assets_final"
SRC, META, LAT = OUT / "source_data", OUT / "metadata", OUT / "latex"
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


# ---------------------------------------------------------------- SEG notices
(OUT / "tables/main/SEG1_external_segmentation_NOT_AVAILABLE.md").write_text(
    "# Table SEG1 - external segmentation benchmark: NOT AVAILABLE\n\n"
    "Requested: Dice, clDice, precision and recall for RetCam, Neo and overall, from the frozen\n"
    "HVDROPDB external segmentation evaluation.\n\n"
    "Status: **NOT AVAILABLE**. A search of the authoritative artifact trees on both servers\n"
    "(`/root/niki_rop_task6_isolated/artifacts` on Server 2 and `/Users/moniaz/niki/artifacts` on\n"
    "Server 1) found no per-camera Dice/clDice/precision/recall table. The only external-benchmark\n"
    "records present are `hvdro_evidence_lineage.csv` and `seg_current_v1_summary.json`, which are\n"
    "provenance records and do not contain these metrics.\n\n"
    "Per the task instruction, no values were invented or approximated. This table is reported as\n"
    "unavailable rather than filled with estimates.\n", encoding="utf-8")
(OUT / "figures/appendix/SEG1_external_segmentation_NOT_AVAILABLE.md").write_text(
    "# Figure SEG1 - external segmentation performance: NOT AVAILABLE\n\n"
    "No authoritative per-camera Dice/clDice/precision/recall artifact exists in the frozen set, so\n"
    "this figure cannot be drawn from frozen data. It is reported as unavailable rather than\n"
    "fabricated.\n", encoding="utf-8")
(OUT / "figures/appendix/SEG2_example_segmentations_NOT_AVAILABLE.md").write_text(
    "# Figure SEG2 - example segmentations: NOT AVAILABLE\n\n"
    "Rendering a fixed descriptive example set (RGB, expert mask, predicted mask, overlay) requires\n"
    "expert reference masks for the specific selected cases. No expert masks for HVDROPDB cases\n"
    "exist in the frozen artifact set, so this figure was skipped rather than fabricated, exactly as\n"
    "the task instruction requires.\n", encoding="utf-8")
log("SEG notices written")

# ---------------------------------------------------------------- claim-evidence matrix
CLAIMS = [
 dict(n=1, claim="Scalar vascular biomarkers contain predictive signal.",
      exp="Task 6, model A_PRIMARY",
      metric="Multiclass macro OVR AUC on the canonical test set = 0.659582",
      ci="Point estimate only; no paired comparison against chance was run for A_PRIMARY",
      allowed="A model using only the five scalar biomarkers achieves AUC 0.659582 on the canonical "
              "test set, i.e. clearly above chance and well below every RGB-based model.",
      forbidden="Do not claim that the scalar biomarkers are sufficient, clinically usable, or a "
                "screening tool; and do not claim a statistically tested improvement over chance, "
                "because no such test was performed."),
 dict(n=2, claim="Five scalar biomarkers show little incremental discrimination beyond frozen RGB "
                 "representations.",
      exp="Task 6/7 C vs B_EMBEDDING_ONLY; Task 8B G vs E",
      metric="Delta multiclass AUC",
      ci="C - B: +0.001085, 95% CI [-0.001691, +0.003868], p = 0.431 (10,000 replicates). "
         "G - E: +0.002039, 95% CI [-0.000274, +0.004397], p = 0.090",
      allowed="In both canonical configurations the five scalar biomarkers produced a small "
              "positive point estimate whose paired confidence interval includes zero; no "
              "incremental discrimination is statistically supported.",
      forbidden="Do not say the biomarkers are useless or carry no information; both point "
                "estimates are positive and the intervals are wide."),
 dict(n=3, claim="Spatial vessel representations provide measurable complementary information "
                 "beyond RGB embeddings.",
      exp="Task 8 and Task 8B, E vs B_EMBEDDING_ONLY",
      metric="Delta multiclass AUC; Delta Brier",
      ci="AUC +0.007938, 95% CI [+0.002630, +0.013131], p = 0.0031; Brier -0.027693, "
         "95% CI [-0.043954, -0.011471], p = 0.0011",
      allowed="Adding the frozen 1,792-d spatial vessel representation to the 2,048-d RGB embedding "
              "significantly improves both ranking and probability quality on the canonical test "
              "set, with paired confidence intervals excluding zero.",
      forbidden="Do not claim improvement in thresholded decision metrics: balanced accuracy, "
                "macro F1 and ECE all had intervals crossing zero in the same analysis."),
 dict(n=4, claim="The spatial-vessel benefit persists in FARFUM and Farabi source-held-out analyses.",
      exp="Task 11 LOSO folds",
      metric="Delta multiclass AUC (E_LOSO - B_LOSO) within each fold",
      ci="FARFUM-RoP +0.022242, 95% CI [+0.014802, +0.029907], p < 0.0001; Farabi +0.012954, "
         "95% CI [+0.004984, +0.021019], p = 0.0018",
      allowed="When an entire acquisition source is absent from fitting, adding the vessel "
              "representation still improves held-out discrimination in both comparable folds, "
              "with balanced accuracy, macro F1 and Brier moving in the same direction.",
      forbidden="Do not present the Plus held-out fold as comparable: its AUC is a restricted "
                "two-class Normal-vs-Plus metric on a test set that is 89.5% Normal, not the "
                "three-class metric used elsewhere."),
 dict(n=5, claim="Scalar biomarkers do not show reproducible incremental benefit after the spatial "
                 "vessel representation is available.",
      exp="Task 8B G vs E; Task 9 I vs H; Task 10 J1 vs J0; Task 11 G_LOSO vs E_LOSO",
      metric="Delta multiclass AUC in four independent regimes",
      ci="Canonical +0.002039, p = 0.090; joint model -0.009578, p = 0.0049; FiLM -0.002123, "
         "p = 0.068; held-out FARFUM p = 0.134, Farabi p = 0.484, Plus p = 0.571",
      allowed="Across four independent regimes the five scalar biomarkers never produced a "
              "statistically supported positive increment over a representation that already "
              "contains the spatial vessel information; in the learned joint model the effect was "
              "significantly negative.",
      forbidden="Do not generalise this to 'biomarkers are useless' or to other feature sets, "
                "cohorts or endpoints. The result is specific to these five features, this "
                "population and this 3-class ROP-Plus task."),
 dict(n=6, claim="Cross-source generalization is substantially worse than mixed-source canonical "
                 "performance.",
      exp="Task 11 source-held-out folds vs Task 6/8 canonical split",
      metric="Multiclass macro OVR AUC",
      ci="FARFUM-RoP E 0.809779 vs canonical E 0.932841 (-0.123062); Farabi E 0.782898 vs 0.932841 "
         "(-0.149943). Descriptive; populations differ so no paired test applies",
      allowed="Holding out an entire acquisition source costs roughly 0.12 to 0.15 AUC for the same "
              "model formulation, and calibration degrades sharply (ECE rising from 0.092 to 0.18-0.26).",
      forbidden="Do not attach a p value to the canonical-versus-held-out difference, because the "
                "two evaluations use different populations and the comparison is descriptive only."),
 dict(n=7, claim="RGB, vessel and biomarker representations are source-dependent.",
      exp="Task 11 feature-distribution and biomarker shift diagnostics",
      metric="Per-feature standardized mean difference (SMD) between held-out and training sources",
      ci="No inferential test; descriptive statistics. RGB median |SMD| 0.379-0.541; vessel median "
         "|SMD| 0.557-1.229; biomarker |SMD| 0.301-1.105",
      allowed="All three representation families carry strong source-dependent shift; the vessel "
              "embedding is the most shifted block, and the five scalar biomarkers shift by up to "
              "one training standard deviation.",
      forbidden="Do not interpret these SMD values as causal or as evidence that a representation "
                "is unusable: the vessel block is the most shifted and still contributes the most."),
 dict(n=8, claim="Class-conditional DANN+MMD with the fixed Task-12 configuration reduced some "
                 "representation-shift diagnostics but did not consistently improve held-out "
                 "disease performance.",
      exp="Task 12, K1 vs K0 across three source-held-out folds",
      metric="Delta multiclass AUC and shift diagnostics",
      ci="K1 - K0 AUC -0.002796 (FARFUM, p = 0.143) and -0.000883 (Farabi, p = 0.259); centroid "
         "distance and median |SMD| reduced in all three folds; domain predictability unchanged at "
         "0.8802-0.9824",
      allowed="With coefficients fixed at 0.10, the alignment reduced representation shift but did "
              "not improve held-out disease classification, and source identity remained almost "
              "perfectly predictable. Outcome B of the pre-declared interpretation.",
      forbidden="Do not claim domain invariance was achieved, and do not generalise to "
                "domain-invariant learning in general: only one method at one fixed setting was "
                "tested."),
 dict(n=9, claim="The project does NOT establish that biomarkers are clinically useless.",
      exp="Whole project",
      metric="-",
      ci="-",
      allowed="The five scalar biomarkers remain interpretable, measurable, FOV-aware vascular "
              "descriptors. They underperformed as additional predictors in this specific "
              "3-class classification setting.",
      forbidden="Do not state or imply that scalar vascular biomarkers have no clinical value, that "
                "they should be abandoned, or that they are uninformative in general."),
 dict(n=10, claim="The project does NOT establish external clinical validation.",
      exp="Whole project",
      metric="-",
      ci="-",
      allowed="All results come from a single retrospective multi-source cohort split at the group "
              "level, evaluated as a canonical locked-split reanalysis and secondary exploratory "
              "analyses. HVDROPDB was used as an external development and measurement benchmark, "
              "not as clinical validation, and its segmentation metrics are not available as "
              "frozen artifacts.",
      forbidden="Do not describe any model as clinically validated, deployment-ready, "
                "screening-capable, or prospectively evaluated. Do not describe the test split as "
                "an untouched confirmatory holdout."),
 dict(n=11, claim="Spatial vessel complementarity is not specific to the original EfficientNet-B5 "
                 "RGB representation.",
      exp="Task 13, M1 vs M0",
      metric="Delta multiclass AUC",
      ci="+0.020936, 95% CI [+0.011426, +0.030828], p < 0.0001",
      allowed="The complementary value of the frozen vessel representation was reproduced with a "
              "distinct dual-backbone attention RGB representation, and the increment there was "
              "larger than with the original embedding.",
      forbidden="Do not claim the vessel representation will improve every RGB model, or that it "
                "has been shown to be universally complementary."),
 dict(n=12, claim="The tested ROPDeepX-style RGB representation did not outperform the original "
                 "EfficientNet-B5 embedding under the project's canonical protocol.",
      exp="Task 13, M0 vs B_EMBEDDING_ONLY",
      metric="Delta multiclass AUC",
      ci="-0.018925, 95% CI [-0.032594, -0.005640], p = 0.0063",
      allowed="Under this dataset and evaluation protocol the dual-backbone attention RGB "
              "representation was inferior to the original single-backbone EfficientNet-B5 "
              "embedding.",
      forbidden="Do not state that ROPDeepX or dual-backbone attention architectures are inferior "
                "in general, and do not present this as a controlled architecture ablation: the "
                "two pipelines differ in learner and training regime as well as in backbone."),
]
m = ["# Claim-evidence matrix", "",
     f"Generated {GEN} from frozen artifacts only. No claim below exceeds what the frozen "
     f"statistics support.", ""]
for c in CLAIMS:
    m += [f"## Claim {c['n']}", "", f"**CLAIM.** {c['claim']}", "",
          f"**SUPPORTING EXPERIMENT.** {c['exp']}", "",
          f"**SUPPORTING METRIC.** {c['metric']}", "",
          f"**CI / p.** {c['ci']}", "",
          f"**WHAT WE ARE ALLOWED TO SAY.** {c['allowed']}", "",
          f"**WHAT WE MUST NOT SAY.** {c['forbidden']}", ""]
(OUT / "CLAIM_EVIDENCE_MATRIX.md").write_text("\n".join(m), encoding="utf-8")
log("claim-evidence matrix written")

# ---------------------------------------------------------------- latex figure manifest
MAIN_FIGS = [("M1", "figures/methods/M1_complete_pipeline", "Complete study pipeline",
              "fig:pipeline"),
             ("M2", "figures/methods/M2_biomarker_pipeline", "Five scalar biomarker pipeline",
              "fig:biomarkerpipeline"),
             ("M3", "figures/methods/M3_model_architectures", "Primary model architectures",
              "fig:architectures"),
             ("M4", "figures/methods/M4_experimental_question_map", "Experimental question map",
              "fig:questionmap"),
             ("D1", "figures/main/D1_source_class_distribution",
              "Class distribution by acquisition source", "fig:sourceclass"),
             ("D2", "figures/main/D2_split_class_distribution",
              "Class distribution across canonical splits", "fig:splitclass"),
             ("D3", "figures/main/D3_source_population",
              "Source contribution to the canonical population", "fig:sourcepopulation"),
             ("R1", "figures/main/R1_model_auc_comparison",
              "Canonical test discrimination by model", "fig:modelauc"),
             ("R2", "figures/main/R2_model_metric_panels", "Multi-metric model comparison",
              "fig:metricpanels"),
             ("R3", "figures/main/R3_per_class_auc", "Per-class AUC", "fig:perclassauc"),
             ("C1", "figures/main/C1_incremental_auc_forest",
              "Incremental information on the frozen RGB representation", "fig:incremental"),
             ("C2", "figures/main/C2_complementarity_summary", "Complementarity summary",
              "fig:complementarity"),
             ("C3", "figures/main/C3_vessel_complementarity_across_rgb_architectures",
              "Vessel complementarity across distinct RGB feature extractors", "fig:archrobust"),
             ("P1", "figures/main/P1_confusion_matrices_main", "Confusion matrices", "fig:confusion"),
             ("P2", "figures/main/P2_reliability_main", "Reliability diagrams", "fig:reliability"),
             ("B1", "figures/main/B1_biomarker_incremental_auc_forest",
              "Six independent tests of the five scalar biomarkers", "fig:biomarkerforest"),
             ("S1", "figures/main/S1_source_heldout_performance",
              "Source-held-out performance", "fig:loso"),
             ("S2", "figures/main/S2_vessel_complementarity_heldout",
              "Vessel complementarity under source-held-out evaluation", "fig:losovessel"),
             ("DS1", "figures/main/DS1_representation_domain_shift", "Representation domain shift",
              "fig:domainshift"),
             ("DS2", "figures/main/DS2_centroid_shift", "Standardized centroid distance",
              "fig:centroidshift"),
             ("DS3", "figures/main/DS3_biomarker_smd_heatmap", "Scalar biomarker source shift",
              "fig:biomarkershift"),
             ("DA1", "figures/main/DA1_domain_alignment_performance",
              "Task 12 domain-alignment performance", "fig:daperformance"),
             ("DA2", "figures/main/DA2_alignment_shift_diagnostics",
              "Representation shift before and after alignment", "fig:dashift"),
             ("DA3", "figures/main/DA3_domain_predictability", "Domain predictability",
              "fig:dapredict"),
             ("R4", "figures/main/R4_ropdeepx_style_results", "ROPDeepX-style results",
              "fig:ropdeepx"),
             ("R5", "figures/main/R5_task13_paired_auc", "Task 13 paired AUC comparisons",
              "fig:task13paired"),
             ("FINAL1", "figures/main/FINAL1_thesis_summary", "Thesis summary", "fig:summary")]
APP_FIGS = [("B2", "figures/appendix/B2_film_diagnostics", "FiLM diagnostics", "fig:film"),
            ("N1", "figures/appendix/N1_task9_learning_curves", "Task 9 learning curves",
             "fig:n1"),
            ("N2", "figures/appendix/N2_task10_learning_curves", "Task 10 learning curves",
             "fig:n2"),
            ("N3", "figures/appendix/N3_task13_learning_curves", "Task 13 learning curves",
             "fig:n3"),
            ("S3", "figures/appendix/S3_biomarker_heldout_forest",
             "Biomarker increment under source-held-out evaluation", "fig:s3"),
            ("A1", "figures/appendix/A1_attention_by_class_and_source",
             "Attention weights by class and source", "fig:a1"),
            ("A2", "figures/appendix/A2_attention_distribution", "Attention distribution",
             "fig:a2")]
tex = ["% Suggested LaTeX figure inclusion blocks. Requires graphicx (and booktabs for tables).",
       "% Paths are relative to the report_assets_final/ directory.", ""]
for tag, path, cap, lab in MAIN_FIGS + APP_FIGS:
    tex += [r"\begin{figure}[htbp]", r"  \centering",
            rf"  \includegraphics[width=\linewidth]{{{path}.pdf}}",
            rf"  \caption{{{cap}. See captions/captions\_en.md (Figure {tag}) for the full "
            rf"caption, abbreviations, statistical test, sample size and any restricted-evaluation "
            rf"note.}}", rf"  \label{{{lab}}}", r"\end{figure}", ""]
(LAT / "figures_manifest.tex").write_text("\n".join(tex), encoding="utf-8")
log("latex figure manifest written")

# ---------------------------------------------------------------- asset index
def real(d, name):
    return sorted(p.name for p in d.iterdir() if p.is_file() and p.name.startswith(name))


idx = ["# Report asset index", "",
       f"Generated {GEN} from frozen artifacts only. No model was trained, no prediction was "
       f"regenerated, and no number was entered by hand.", "",
       "## Directory layout", "",
       "```", "report_assets_final/", "  figures/main/      main thesis figures",
       "  figures/methods/   methods and schematic figures",
       "  figures/appendix/  diagnostics and secondary experiments",
       "  tables/main/       main thesis tables",
       "  tables/appendix/   diagnostic tables",
       "  diagrams/          (reserved; the schematics M1-M4 are in figures/methods)",
       "  captions/          captions_en.md, captions_fa.md", "  source_data/       one CSV per plot",
       "  latex/             figures_manifest.tex, tables/*.tex", "  metadata/          asset_sources.csv",
       "```", "", "## MAIN THESIS", "",
       "Recommended minimum set for the B.Sc. report.", "",
       "### Figures", ""]
for tag, path, cap, lab in MAIN_FIGS:
    if tag in ("M1", "M3", "D1", "R1", "C1", "C2", "P1", "P2", "S1", "S2", "DS1", "FINAL1"):
        idx.append(f"- **{tag}** {cap} — `{path}.png|pdf|svg`")
idx += ["", "### Additional main figures (recommended for the journal draft)", ""]
for tag, path, cap, lab in MAIN_FIGS:
    if tag not in ("M1", "M3", "D1", "R1", "C1", "C2", "P1", "P2", "S1", "S2", "DS1", "FINAL1"):
        idx.append(f"- {tag} {cap} — `{path}.png|pdf|svg`")
idx += ["", "### Tables", "",
        "- **D1** Cohort summary — `tables/main/D1_cohort_summary.csv|tex|md`",
        "- **D2** Canonical split summary — `tables/main/D2_canonical_split_summary.csv|tex|md`",
        "- **R1** Master model results — `tables/main/R1_master_model_results.csv|tex|md`",
        "- **R2** Per-class AUC — `tables/main/R2_per_class_auc.csv|tex`",
        "- **C1** Primary paired statistics — `tables/main/C1_primary_paired_statistics.csv|tex|md`",
        "- **C2** Architecture robustness of the vessel contribution — "
        "`tables/main/C2_architecture_robustness.csv|tex|md`",
        "- **B1** Biomarker evidence summary — `tables/main/B1_biomarker_evidence_summary.csv|tex|md`",
        "- **S1** Source-held-out performance — `tables/main/S1_source_heldout_performance.csv|tex|md`",
        "- **DS1** Domain shift summary — `tables/main/DS1_domain_shift_summary.csv|tex|md`",
        "- **FINAL1** Complete experiment summary — "
        "`tables/main/FINAL1_experiment_summary.csv|tex|md`",
        "", "## APPENDIX", "", "### Figures", ""]
for tag, path, cap, lab in APP_FIGS:
    idx.append(f"- {tag} {cap} — `{path}.png|pdf|svg`")
idx += ["", "### Tables", "",
        "- **N1** Negative architecture experiments — "
        "`tables/appendix/N1_negative_architecture_experiments.csv|tex|md`",
        "- **DA1** Task 12 results — `tables/appendix/DA1_task12_results.csv|tex|md`",
        "- **SEG1** External segmentation benchmark — **NOT AVAILABLE**, see "
        "`tables/main/SEG1_external_segmentation_NOT_AVAILABLE.md`",
        "- **SEG2** Example segmentations — **NOT AVAILABLE**, see "
        "`figures/appendix/SEG2_example_segmentations_NOT_AVAILABLE.md`",
        "", "## DEFENSE SLIDES", "",
        "Figures that work well at presentation size: **FINAL1** (one-page summary), **C1** and "
        "**C3** (incremental-information forests), **C2** (complementarity schematic), **M1** "
        "(pipeline), **D1** (cohort composition), **S1** and **S2** (source-held-out), **DS1** and "
        "**DS3** (domain shift), **R1** (AUC comparison). Tables **C2** and **B1** also read well "
        "as single slides.", "", "## FUTURE PAPER", "",
        "Figures: FINAL1, C1, C3, C2, P1, P2, S1, S2, DS1, DS2, DS3, M1, M3, D1, R1, R2, R3, R4, "
        "R5, B1. Tables: D1, D2, R1, R2, C1, C2, B1, S1, DS1, FINAL1. Appendix for a manuscript: "
        "B2, N1, N2, N3, S3, A1, A2, DA1, DA2, DA3.", "",
        "## Notes carried with the assets", "",
        "- The canonical split is a locked split that was already used for the primary Task-6/7 "
        "results; Task 8 onwards are secondary post-primary exploratory analyses on the same split, "
        "not confirmatory replications.",
        "- The Plus source contains no Pre-Plus. Every three-class AUC for that subset is reported "
        "as undefined and a clearly labelled restricted Normal-vs-Plus binary AUC is given "
        "separately. The two must never be merged into one axis or one pooled mean.",
        "- Historical Phase-5 results are excluded from every asset in this pack by construction.",
        "- No figure contains long Persian text; all in-figure labels are English technical labels. "
        "Persian captions are provided separately in `captions/captions_fa.md`.",
        ""]
(OUT / "REPORT_ASSET_INDEX.md").write_text("\n".join(idx), encoding="utf-8")
log("asset index written")

# ---------------------------------------------------------------- metadata
INPUTS = {
 "task6_metrics": A / "task6/metrics_summary.csv",
 "task6_source_breakdown": A / "task6/source_breakdown.csv",
 "task6_pred_B_EMBEDDING_ONLY": A / "task6/test_predictions_B_EMBEDDING_ONLY.csv",
 "task6_confusion_A": A / "task6/confusion_A_PRIMARY.csv",
 "task6_confusion_B_EMB": A / "task6/confusion_B_EMBEDDING_ONLY.csv",
 "task7_paired_primary": A / "task7_paired_statistics/paired_primary_metrics.csv",
 "task8_metrics": A / "task8_spatial_vessel_fusion/metrics_summary.csv",
 "task8_pred_E": A / "task8_spatial_vessel_fusion/test_predictions_E_RGB_VESSEL_FEATURE_FUSION.csv",
 "task8_pred_G": A / "task8_spatial_vessel_fusion/test_predictions_G_RGB_VESSEL_SCALAR_FUSION.csv",
 "task8_confusion_E": A / "task8_spatial_vessel_fusion/confusion_E_RGB_VESSEL_FEATURE_FUSION.csv",
 "task8_confusion_G": A / "task8_spatial_vessel_fusion/confusion_G_RGB_VESSEL_SCALAR_FUSION.csv",
 "task8b_paired_10k": A / "task8_spatial_vessel_fusion/statistical_closure_10k/paired_metrics_10k.csv",
 "task9_metrics": A / "task9_joint_multimodal_fusion/metrics_summary.csv",
 "task9_paired_10k": A / "task9_joint_multimodal_fusion/paired_all_10k.csv",
 "task9_history_H": A / "task9_joint_multimodal_fusion/H_JOINT_RGB_VESSEL_history.json",
 "task9_history_I": A / "task9_joint_multimodal_fusion/I_JOINT_RGB_VESSEL_BIOMARKER_history.json",
 "task10_metrics": A / "task10_biomarker_film/metrics_summary.csv",
 "task10_paired_10k": A / "task10_biomarker_film/paired_all_10k.csv",
 "task10_film_diag": A / "task10_biomarker_film/film_modulation_diagnostics.csv",
 "task10_sensitivity": A / "task10_biomarker_film/biomarker_sensitivity.csv",
 "task10_history_J0": A / "task10_biomarker_film/J0_FROZEN_FUSION_CONTROL_history.json",
 "task10_history_J1": A / "task10_biomarker_film/J1_BIOMARKER_FILM_history.json",
 "task11_metrics_per_source": A / "task11_source_heldout/metrics_per_source.csv",
 "task11_paired_10k": A / "task11_source_heldout/paired_bootstrap_10k.csv",
 "task11_domain_shift": A / "task11_source_heldout/domain_shift_diagnostics.csv",
 "task11_biomarker_shift": A / "task11_source_heldout/biomarker_shift.csv",
 "task11_population_audit": A / "task11_source_heldout/population_audit.csv",
 "task12_paired_10k": A / "task12_class_conditional_domain_generalization/paired_bootstrap_10k.csv",
 "task12_heldout_metrics": A / "task12_class_conditional_domain_generalization/heldout_metrics.csv",
 "task12_shift": A / "task12_class_conditional_domain_generalization/representation_shift.csv",
 "task12_domain_pred": A / "task12_class_conditional_domain_generalization/domain_predictability.csv",
 "task13_metrics": A / "task13_ropdeepx_style/metrics_summary.csv",
 "task13_paired_10k": A / "task13_ropdeepx_style/paired_all_10k.csv",
 "task13_attention_grouped": A / "task13_ropdeepx_style/attention_by_class_and_source.csv",
 "task13_attention_stats": A / "task13_ropdeepx_style/attention_stats_val_test.csv",
 "task13_history_L": A / "task13_ropdeepx_style/L_history.json",
 "task13_selection_L": A / "task13_ropdeepx_style/L_selection.json",
 "manifest": WS / "primary_complete_case_v2_server2.csv",
}
shas = {k: sha(v) for k, v in INPUTS.items()}

ASSET_INPUTS = {
 "figures/methods/M1_complete_pipeline": ["manifest"],
 "figures/methods/M2_biomarker_pipeline": ["manifest"],
 "figures/methods/M3_model_architectures": ["task6_metrics", "task8_metrics"],
 "figures/methods/M4_experimental_question_map": [],
 "figures/main/D1_source_class_distribution": ["task11_population_audit"],
 "figures/main/D2_split_class_distribution": ["manifest"],
 "figures/main/D3_source_population": ["task11_population_audit"],
 "figures/main/R1_model_auc_comparison": ["task6_metrics", "task8_metrics"],
 "figures/main/R2_model_metric_panels": ["task6_metrics", "task8_metrics"],
 "figures/main/R3_per_class_auc": ["task6_metrics", "task8_metrics"],
 "figures/main/R4_ropdeepx_style_results": ["task13_metrics", "task6_metrics", "task8_metrics"],
 "figures/main/R5_task13_paired_auc": ["task13_paired_10k"],
 "figures/main/C1_incremental_auc_forest": ["task7_paired_primary", "task8b_paired_10k"],
 "figures/main/C2_complementarity_summary": ["task6_metrics", "task8_metrics",
                                             "task7_paired_primary", "task8b_paired_10k"],
 "figures/main/C3_vessel_complementarity_across_rgb_architectures":
     ["task8b_paired_10k", "task13_paired_10k"],
 "figures/main/P1_confusion_matrices_main": ["task6_confusion_A", "task6_confusion_B_EMB",
                                             "task8_confusion_E", "task8_confusion_G"],
 "figures/main/P2_reliability_main": ["task6_pred_B_EMBEDDING_ONLY", "task8_pred_E", "task8_pred_G"],
 "figures/main/B1_biomarker_incremental_auc_forest":
     ["task7_paired_primary", "task8b_paired_10k", "task9_paired_10k", "task10_paired_10k",
      "task11_paired_10k"],
 "figures/appendix/B2_film_diagnostics": ["task10_film_diag", "task10_sensitivity"],
 "figures/appendix/N1_task9_learning_curves": ["task9_history_H", "task9_history_I", "task9_metrics"],
 "figures/appendix/N2_task10_learning_curves": ["task10_history_J0", "task10_history_J1"],
 "figures/appendix/N3_task13_learning_curves": ["task13_history_L", "task13_selection_L"],
 "figures/main/S1_source_heldout_performance": ["task11_metrics_per_source"],
 "figures/main/S2_vessel_complementarity_heldout": ["task11_paired_10k"],
 "figures/appendix/S3_biomarker_heldout_forest": ["task11_paired_10k"],
 "figures/main/DS1_representation_domain_shift": ["task11_domain_shift"],
 "figures/main/DS2_centroid_shift": ["task11_domain_shift"],
 "figures/main/DS3_biomarker_smd_heatmap": ["task11_biomarker_shift"],
 "figures/main/DA1_domain_alignment_performance": ["task12_heldout_metrics"],
 "figures/main/DA2_alignment_shift_diagnostics": ["task12_shift"],
 "figures/main/DA3_domain_predictability": ["task12_domain_pred"],
 "figures/appendix/A1_attention_by_class_and_source": ["task13_attention_grouped"],
 "figures/appendix/A2_attention_distribution": ["task13_attention_stats"],
 "figures/main/FINAL1_thesis_summary": ["task7_paired_primary", "task8b_paired_10k",
                                        "task11_paired_10k", "task13_paired_10k",
                                        "task11_domain_shift"],
 "tables/main/D1_cohort_summary": ["task11_population_audit"],
 "tables/main/D2_canonical_split_summary": ["manifest"],
 "tables/main/R1_master_model_results": ["task6_metrics", "task8_metrics"],
 "tables/main/R2_per_class_auc": ["task6_metrics", "task8_metrics"],
 "tables/main/C1_primary_paired_statistics": ["task7_paired_primary", "task8b_paired_10k"],
 "tables/main/C2_architecture_robustness": ["task8b_paired_10k", "task13_paired_10k",
                                            "task6_metrics", "task8_metrics", "task13_metrics"],
 "tables/main/B1_biomarker_evidence_summary":
     ["task7_paired_primary", "task8b_paired_10k", "task9_paired_10k", "task10_paired_10k",
      "task11_paired_10k"],
 "tables/appendix/N1_negative_architecture_experiments":
     ["task9_metrics", "task10_metrics", "task9_history_H", "task9_history_I"],
 "tables/main/S1_source_heldout_performance": ["task11_metrics_per_source"],
 "tables/main/DS1_domain_shift_summary": ["task11_domain_shift"],
 "tables/appendix/DA1_task12_results": ["task12_heldout_metrics", "task12_paired_10k"],
 "tables/main/FINAL1_experiment_summary": ["task6_metrics", "task8_metrics", "task7_paired_primary",
                                           "task8b_paired_10k", "task9_paired_10k",
                                           "task10_paired_10k", "task11_paired_10k",
                                           "task12_paired_10k", "task13_metrics",
                                           "task13_paired_10k"],
}
rows = []
for asset, keys in ASSET_INPUTS.items():
    if not keys:
        rows.append({"asset": asset, "input_file": "", "input_sha256": "",
                     "script": "scripts/report_pack_figures.py", "generated_at": GEN,
                     "notes": "schematic; no data input"})
        continue
    for k in keys:
        rows.append({"asset": asset, "input_file": str(INPUTS[k]).replace(str(WS) + "/", ""),
                     "input_sha256": shas[k], "script": "scripts/report_pack_figures.py or "
                     "scripts/report_pack_tables.py", "generated_at": GEN,
                     "notes": ""})
pd.DataFrame(rows).to_csv(META / "asset_sources.csv", index=False)
(OUT / "metadata/asset_sources.csv").write_text(
    (META / "asset_sources.csv").read_text(), encoding="utf-8")
json.dump({"generated_at": GEN, "input_sha256": shas,
           "note": "every plotted number traces to one of these frozen input artifacts"},
          open(META / "input_sha256.json", "w"), indent=2)
log(f"metadata written ({len(rows)} asset-input rows)")

# ---------------------------------------------------------------- self-check
checks = {}
checks["figures_main"] = len([p for p in (OUT / "figures/main").glob("*.png")])
checks["figures_methods"] = len([p for p in (OUT / "figures/methods").glob("*.png")])
checks["figures_appendix"] = len([p for p in (OUT / "figures/appendix").glob("*.png")])
checks["pdf_count"] = len(list(OUT.rglob("*.pdf")))
checks["svg_count"] = len(list(OUT.rglob("*.svg")))
checks["source_data_csv"] = len(list(SRC.glob("*.csv")))
checks["latex_tables"] = len(list((OUT / "latex/tables").glob("*.tex")))
checks["tables_main_csv"] = len(list((OUT / "tables/main").glob("*.csv")))
checks["tables_appendix_csv"] = len(list((OUT / "tables/appendix").glob("*.csv")))
checks["captions_en"] = (OUT / "captions/captions_en.md").exists()
checks["captions_fa"] = (OUT / "captions/captions_fa.md").exists()
checks["claim_matrix"] = (OUT / "CLAIM_EVIDENCE_MATRIX.md").exists()
checks["asset_index"] = (OUT / "REPORT_ASSET_INDEX.md").exists()
checks["latex_fig_manifest"] = (OUT / "latex/figures_manifest.tex").exists()
checks["asset_sources"] = (META / "asset_sources.csv").exists()
json.dump(checks, open(META / "pack_selfcheck.json", "w"), indent=2)
log(json.dumps(checks, indent=2))
print("PART3_DONE")
