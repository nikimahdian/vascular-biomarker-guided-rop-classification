# Report asset index

Generated 2026-09-22T17:11:21Z from frozen artifacts only. No model was trained, no prediction was regenerated, and no number was entered by hand.

## Directory layout

```
report_assets_final/
  figures/main/      main thesis figures
  figures/methods/   methods and schematic figures
  figures/appendix/  diagnostics and secondary experiments
  tables/main/       main thesis tables
  tables/appendix/   diagnostic tables
  diagrams/          (reserved; the schematics M1-M4 are in figures/methods)
  captions/          captions_en.md, captions_fa.md
  source_data/       one CSV per plot
  latex/             figures_manifest.tex, tables/*.tex
  metadata/          asset_sources.csv
```

## MAIN THESIS

Recommended minimum set for the B.Sc. report.

### Figures

- **M1** Complete study pipeline — `figures/methods/M1_complete_pipeline.png|pdf|svg`
- **M3** Primary model architectures — `figures/methods/M3_model_architectures.png|pdf|svg`
- **D1** Class distribution by acquisition source — `figures/main/D1_source_class_distribution.png|pdf|svg`
- **R1** Canonical test discrimination by model — `figures/main/R1_model_auc_comparison.png|pdf|svg`
- **C1** Incremental information on the frozen RGB representation — `figures/main/C1_incremental_auc_forest.png|pdf|svg`
- **C2** Complementarity summary — `figures/main/C2_complementarity_summary.png|pdf|svg`
- **P1** Confusion matrices — `figures/main/P1_confusion_matrices_main.png|pdf|svg`
- **P2** Reliability diagrams — `figures/main/P2_reliability_main.png|pdf|svg`
- **S1** Source-held-out performance — `figures/main/S1_source_heldout_performance.png|pdf|svg`
- **S2** Vessel complementarity under source-held-out evaluation — `figures/main/S2_vessel_complementarity_heldout.png|pdf|svg`
- **DS1** Representation domain shift — `figures/main/DS1_representation_domain_shift.png|pdf|svg`
- **FINAL1** Thesis summary — `figures/main/FINAL1_thesis_summary.png|pdf|svg`

### Additional main figures (recommended for the journal draft)

- M2 Five scalar biomarker pipeline — `figures/methods/M2_biomarker_pipeline.png|pdf|svg`
- M4 Experimental question map — `figures/methods/M4_experimental_question_map.png|pdf|svg`
- D2 Class distribution across canonical splits — `figures/main/D2_split_class_distribution.png|pdf|svg`
- D3 Source contribution to the canonical population — `figures/main/D3_source_population.png|pdf|svg`
- R2 Multi-metric model comparison — `figures/main/R2_model_metric_panels.png|pdf|svg`
- R3 Per-class AUC — `figures/main/R3_per_class_auc.png|pdf|svg`
- C3 Vessel complementarity across distinct RGB feature extractors — `figures/main/C3_vessel_complementarity_across_rgb_architectures.png|pdf|svg`
- B1 Six independent tests of the five scalar biomarkers — `figures/main/B1_biomarker_incremental_auc_forest.png|pdf|svg`
- DS2 Standardized centroid distance — `figures/main/DS2_centroid_shift.png|pdf|svg`
- DS3 Scalar biomarker source shift — `figures/main/DS3_biomarker_smd_heatmap.png|pdf|svg`
- DA1 Task 12 domain-alignment performance — `figures/main/DA1_domain_alignment_performance.png|pdf|svg`
- DA2 Representation shift before and after alignment — `figures/main/DA2_alignment_shift_diagnostics.png|pdf|svg`
- DA3 Domain predictability — `figures/main/DA3_domain_predictability.png|pdf|svg`
- R4 ROPDeepX-style results — `figures/main/R4_ropdeepx_style_results.png|pdf|svg`
- R5 Task 13 paired AUC comparisons — `figures/main/R5_task13_paired_auc.png|pdf|svg`

### Tables

- **D1** Cohort summary — `tables/main/D1_cohort_summary.csv|tex|md`
- **D2** Canonical split summary — `tables/main/D2_canonical_split_summary.csv|tex|md`
- **R1** Master model results — `tables/main/R1_master_model_results.csv|tex|md`
- **R2** Per-class AUC — `tables/main/R2_per_class_auc.csv|tex`
- **C1** Primary paired statistics — `tables/main/C1_primary_paired_statistics.csv|tex|md`
- **C2** Architecture robustness of the vessel contribution — `tables/main/C2_architecture_robustness.csv|tex|md`
- **B1** Biomarker evidence summary — `tables/main/B1_biomarker_evidence_summary.csv|tex|md`
- **S1** Source-held-out performance — `tables/main/S1_source_heldout_performance.csv|tex|md`
- **DS1** Domain shift summary — `tables/main/DS1_domain_shift_summary.csv|tex|md`
- **FINAL1** Complete experiment summary — `tables/main/FINAL1_experiment_summary.csv|tex|md`

## APPENDIX

### Figures

- B2 FiLM diagnostics — `figures/appendix/B2_film_diagnostics.png|pdf|svg`
- N1 Task 9 learning curves — `figures/appendix/N1_task9_learning_curves.png|pdf|svg`
- N2 Task 10 learning curves — `figures/appendix/N2_task10_learning_curves.png|pdf|svg`
- N3 Task 13 learning curves — `figures/appendix/N3_task13_learning_curves.png|pdf|svg`
- S3 Biomarker increment under source-held-out evaluation — `figures/appendix/S3_biomarker_heldout_forest.png|pdf|svg`
- A1 Attention weights by class and source — `figures/appendix/A1_attention_by_class_and_source.png|pdf|svg`
- A2 Attention distribution — `figures/appendix/A2_attention_distribution.png|pdf|svg`

### Tables

- **N1** Negative architecture experiments — `tables/appendix/N1_negative_architecture_experiments.csv|tex|md`
- **DA1** Task 12 results — `tables/appendix/DA1_task12_results.csv|tex|md`
- **SEG1** External segmentation benchmark — **NOT AVAILABLE**, see `tables/main/SEG1_external_segmentation_NOT_AVAILABLE.md`
- **SEG2** Example segmentations — **NOT AVAILABLE**, see `figures/appendix/SEG2_example_segmentations_NOT_AVAILABLE.md`

## DEFENSE SLIDES

Figures that work well at presentation size: **FINAL1** (one-page summary), **C1** and **C3** (incremental-information forests), **C2** (complementarity schematic), **M1** (pipeline), **D1** (cohort composition), **S1** and **S2** (source-held-out), **DS1** and **DS3** (domain shift), **R1** (AUC comparison). Tables **C2** and **B1** also read well as single slides.

## FUTURE PAPER

Figures: FINAL1, C1, C3, C2, P1, P2, S1, S2, DS1, DS2, DS3, M1, M3, D1, R1, R2, R3, R4, R5, B1. Tables: D1, D2, R1, R2, C1, C2, B1, S1, DS1, FINAL1. Appendix for a manuscript: B2, N1, N2, N3, S3, A1, A2, DA1, DA2, DA3.

## Notes carried with the assets

- The canonical split is a locked split that was already used for the primary Task-6/7 results; Task 8 onwards are secondary post-primary exploratory analyses on the same split, not confirmatory replications.
- The Plus source contains no Pre-Plus. Every three-class AUC for that subset is reported as undefined and a clearly labelled restricted Normal-vs-Plus binary AUC is given separately. The two must never be merged into one axis or one pooled mean.
- Historical Phase-5 results are excluded from every asset in this pack by construction.
- No figure contains long Persian text; all in-figure labels are English technical labels. Persian captions are provided separately in `captions/captions_fa.md`.
