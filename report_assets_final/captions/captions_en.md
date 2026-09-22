# Figure and table captions (English)

Generated 2026-09-22T17:10:38Z from frozen artifacts only.

## Figure M1 — Complete study pipeline

Schematic of the analysis pipeline. A canonical 384x384 RGB fundus image is used twice: once by an EfficientNet-B5 RGB encoder producing a 2048-d representation, and once by the frozen segmentation model that yields a binary vessel map, which a spatial vessel encoder (EfficientNet-B4) converts into a 1792-d representation. In parallel, FOV-aware vascular measurement produces five scalar biomarkers. Fusion models combine these representations and output Normal / Pre-Plus / Plus. The canonical group-disjoint split (6,203 / 1,328 / 1,331), the paired class-stratified bootstrap (seed 42, 10,000 replicates) and the source-held-out evaluation protocols are indicated. No performance values appear in this figure.

## Figure M2 — Five scalar biomarker pipeline

Schematic of the scalar biomarker pathway. A canonical RGB fundus image is segmented into a binary vessel map, which is passed through FOV-aware vascular measurement (coverage denominator = content area) to produce exactly five scalar biomarkers: vessel_density_fov, skel_density_fov, fractal_d0, fractal_d1 and fractal_d2. This is a methods schematic, not a statistical result.

## Figure M3 — Primary model architectures

Architectures and feature dimensionalities of the primary models. A: five biomarkers into XGBoost. B: RGB into EfficientNet-B5 with a linear classifier. B embedding: RGB into EfficientNet-B5 producing a 2048-d vector into XGBoost. C: 2048-d RGB plus five biomarkers (2,053-d) into XGBoost. D: vessel map into EfficientNet-B4 with a linear classifier. E: 2048-d RGB plus 1,792-d vessel (3,840-d) into XGBoost. G: 2048-d RGB plus 1,792-d vessel plus five biomarkers (3,845-d) into XGBoost. All downstream XGBoost models share one frozen hyperparameter configuration.

## Figure M4 — Experimental question map

Sequence of the nine experiments and the single scientific question each one addresses. Task 6 is the primary canonical comparison, Task 7 provides paired inference, Task 8 introduces the spatial vessel representation, Task 8B closes it statistically at 10,000 replicates, Tasks 9 and 10 test learned fusion and biomarker conditioning, Task 11 quantifies source-held-out generalisation, Task 12 tests class-conditional domain invariance, and Task 13 tests whether the vessel result survives a different RGB backbone. No metrics are shown.

## Figure D1 — Class distribution by acquisition source

Class distribution of the final complete-case population by acquisition source (N = 8,862; 414 groups). The Plus source contains 5,925 images of which 5,304 are Normal and 621 are Plus, and **zero Pre-Plus**; this is annotated in the figure. FARFUM-RoP contains 1,528 images (780 / 478 / 270) and Farabi 1,409 images (369 / 452 / 588). Counts were read from the frozen population audit artifact.

## Figure D2 — Class distribution across canonical splits

Class counts within the canonical group-disjoint training, validation and test splits of the complete-case population (train 6,203; validation 1,328; test 1,331; total 8,862). Group disjointness was verified: no group appears in more than one split.

## Figure D3 — Source contribution to the canonical population

Contribution of each acquisition source to the canonical complete-case population of 8,862 images. The Plus source supplies 5,925 images (66.9%), FARFUM-RoP 1,528 (17.2%) and Farabi 1,409 (15.9%). A bar chart is used rather than a pie chart so that the counts remain directly comparable.

## Figure R1 — Canonical test discrimination by model

Multiclass macro one-vs-rest ROC-AUC on the canonical test set (N = 1,331) for the primary models. Values are frozen Task 6 and Task 8 results. The x-axis starts at 0.60 to keep the differences legible; the absolute scale is labelled explicitly and the axis is not truncated below the lowest plotted value. F_RGB_VESSEL_LATE_FUSION is omitted because its mixture coefficient saturated at alpha = 1.0, making its probability table bit-identical to B_EMBEDDING_ONLY.

## Figure R2 — Multi-metric comparison on canonical test

Five separate panels for multiclass AUC, balanced accuracy, macro F1, Brier score and expected calibration error (ECE, 15 equal-width bins) on the canonical test set (N = 1,331). AUC, balanced accuracy and macro F1 are higher-is-better; Brier and ECE are lower-is-better. Metrics are deliberately not combined into any composite score.

## Figure R3 — Per-class AUC

One-vs-rest AUC for each class separately on the canonical test set (N = 1,331; Normal 982, Pre-Plus 140, Plus 209). Pre-Plus is the smallest class and the hardest for every model. Values are frozen Task 6 and Task 8 results.

## Figure R4 — Task 13 ROPDeepX-style results

Canonical test-set metrics (N = 1,331) for the Task 13 models against the frozen reference models, in five separate panels for multiclass AUC, balanced accuracy, macro F1, Brier and ECE. L is a trained neural head on the dual-RGB attention representation, M0 is XGBoost on the frozen 1,024-d attended RGB embedding, and M1 adds the frozen 1,792-d vessel embedding to M0. L's own linear classifier is clearly weaker than XGBoost on L's own representation. L is not directly comparable to the embedding-based models because it is a trained end-to-end classifier rather than a downstream learner on frozen features.

## Figure R5 — Task 13 paired AUC comparisons

Paired class-stratified bootstrap deltas in multiclass AUC on the canonical test set (N = 1,331, 10,000 replicates, seed 42) for the four Task 13 comparisons. M0 - B is significantly negative, showing the dual-backbone attention RGB representation was worse than the original EfficientNet-B5 embedding; M1 - M0 is significantly positive, showing the vessel representation remains complementary; M1 - E and M1 - G are null, so the dual-RGB pipeline is statistically indistinguishable from the original RGB + vessel models.

## Figure C1 — Incremental information on the frozen RGB representation

Paired class-stratified bootstrap deltas in multiclass AUC on the canonical test set (N = 1,331, seed 42, 10,000 replicates). Three increments are shown against the 2,048-d RGB embedding: adding the five scalar biomarkers (C - B), adding the spatial vessel representation (E - B), and adding the five scalar biomarkers after the vessel representation (G - E). The dashed line marks zero. Black intervals exclude zero; grey intervals cross it. C - B and G - E both cross zero, while E - B excludes zero positively.

## Figure C2 — Complementarity summary

Single-figure summary of what each representation adds to the 2,048-d RGB embedding. Adding the five scalar biomarkers (C) yields delta AUC +0.001085 with 95% CI [-0.001691, +0.003868] (p = 0.431): no supported increment. Adding the spatial vessel representation (E) yields delta AUC +0.007938 with 95% CI [+0.002630, +0.013131] (p = 0.0031): a supported increment. Adding the five scalar biomarkers after the vessel representation (G) yields delta AUC +0.002039 with 95% CI [-0.000274, +0.004397] (p = 0.090): no supported increment. C and G are secondary exploratory models.

## Figure C3 — Vessel complementarity across distinct RGB feature extractors

Paired 95% confidence intervals for the change in multiclass AUC when the same frozen 1,792-d vessel embedding is added to two different RGB representations on the canonical test set (N = 1,331, 10,000 paired class-stratified replicates, seed 42). With the original single-backbone EfficientNet-B5 representation the increment is +0.007938 [+0.002630, +0.013131], p = 0.0031; with the dual-backbone attention representation (Task 13) it is +0.020936 [+0.011426, +0.030828], p < 0.0001. The figure shows that the vessel contribution is not specific to one RGB feature extractor; it does **not** imply that the dual-backbone model itself is superior, and in fact its RGB-only control was worse than the original embedding.

## Figure P1 — Confusion matrices

Row-normalised confusion matrices on the canonical test set (N = 1,331) with raw counts in parentheses, using identical class ordering for all panels: Normal, Pre-Plus, Plus. Rows are true classes and columns predicted classes, so each row sums to 100%. Pre-Plus is the smallest and most frequently missed class in every model. Individual panels are also provided as separate files.

## Figure P2 — Reliability

Reliability diagrams on the canonical test set using the same 15 equal-width confidence bins as the reported ECE. The dashed diagonal is perfect calibration. ECE is shown in the legend for each model. All three models are under-confident at low confidence and over-confident at high confidence, and none lies on the diagonal.

## Figure B1 — Six independent tests of the five scalar biomarkers

Forest plot of the change in AUC attributable to the five scalar biomarkers, shown separately for each experiment in which they were tested, with paired 95% confidence intervals from the frozen 10,000-replicate bootstraps (seed 42). Rows are labelled by analysis type: canonical split, joint neural model, FiLM conditioning, and source-held-out folds. The last row uses the restricted Normal-vs-Plus binary AUC because the Plus source has no Pre-Plus, and is therefore not directly comparable to the three-class rows. No compatible effect is statistically supported; the only interval excluding zero is negative (joint model).

## Figure B2 — FiLM diagnostics

Panel a: learned FiLM modulation on the test set. The bounded residual parameterisation gamma = 1 + 0.10*tanh(delta_gamma) and beta = 0.10*tanh(beta_raw) produced gamma with mean 0.999580, median 0.999624 and 5th-95th percentile range [0.996997, 1.001924], and beta with mean -0.000006 and percentile range [-0.002695, +0.002616]. Dashed lines mark identity (gamma = 1, beta = 0). Panel b: mean and 95th-percentile absolute change in the frozen J1 test probabilities when each biomarker is individually replaced by its training mean. All five biomarkers produce changes of order 2 x 10^-5 and no argmax flips, so the conditioner behaved effectively as an identity map.

## Figure N1 — Task 9 learning curves

Training and validation loss and validation AUC per epoch for the two Task 9 joint models, with the encoder-unfreezing boundary at epoch 3.5 and the selected epoch marked. After unfreezing, training loss falls monotonically while validation loss rises steeply and validation AUC decays, the signature of severe overfitting; both models selected an epoch within two epochs of unfreezing. Panel c shows the canonical test AUC and macro F1 for reference. These are secondary exploratory experiments, not primary models.

## Figure N2 — Task 10 learning curves

Training and validation loss and validation AUC per epoch for the Task 10 frozen-embedding models, with the selected epoch marked. Training loss falls monotonically while validation loss rises from its first-epoch minimum, so both models selected epoch 1. Even with both CNNs frozen into precomputed features, a 2.5 M-parameter head overfits a 6,203-sample training set. Secondary exploratory experiment.

## Figure N3 — Task 13 learning curves

Panel a: training and validation loss per epoch for the ROPDeepX-style dual-RGB model, with the backbone-unfreezing boundary at epoch 3.5 and the selected epoch 2 marked. Panel b: validation AUC and macro F1, with the minimum-validation-loss epoch 7 marked. Training loss falls monotonically from 0.7638 to 0.5574 while validation AUC peaks at epoch 2 (0.86624) and declines to 0.84014 by epoch 7. Panel c: mean attention weight per epoch with collapse thresholds at 0.05 and 0.95; no collapse occurred. Secondary exploratory experiment.

## Figure S1 — Source-held-out performance

Panel a: three-class macro one-vs-rest AUC on each held-out source for the LOSO models, where the held-out source contributed nothing to fitting or preprocessing. Panel b uses a different axis and is explicitly labelled **restricted Normal-vs-Plus** binary AUC, because the Plus source contains no Pre-Plus and a three-class AUC is undefined there. The two panels must not be compared on one axis. Panel b's axis covers 0.95 to 1.00 to resolve the three models; the absolute range is labelled.

## Figure S2 — Vessel complementarity under source-held-out evaluation

Paired 95% confidence intervals for the change in AUC when the spatial vessel representation is added (E - B), computed separately within each source-held-out fold with 10,000 paired class-stratified replicates (seed 42). The first two rows are three-class macro AUC; the third row is the restricted Normal-vs-Plus binary AUC for the Plus fold and is labelled as such. The dashed line marks zero. Both comparable three-class folds exclude zero positively.

## Figure S3 — Biomarker increment under source-held-out evaluation

Paired 95% confidence intervals for the change in AUC when the five scalar biomarkers are added after the spatial vessel representation (G - E), computed within each source-held-out fold with 10,000 paired class-stratified replicates (seed 42). As in S2, the Plus row is the restricted Normal-vs-Plus binary AUC. Every interval crosses zero, so no biomarker increment is supported under source shift.

## Figure DS1 — Representation domain shift

Descriptive shift between each held-out source and the pooled training sources for the frozen RGB (2,048-d) and vessel (1,792-d) embeddings, measured as the per-feature standardized mean difference SMD = (mean_target - mean_train) / sd_train. Median, 90th-percentile and 95th-percentile absolute SMD are shown. No target labels were used. The vessel embedding is shifted more than the RGB embedding in every fold, and Plus is the most shifted source in both blocks.

## Figure DS2 — Standardized centroid distance

Standardized centroid distance between each held-out source and the pooled training sources in the frozen RGB and vessel embedding spaces, computed as the Euclidean norm of the per-feature standardized mean differences. Plus is the most distant source in both representations; Farabi is more distant than FARFUM-RoP. Descriptive diagnostic only.

## Figure DS3 — Scalar biomarker source shift

Heatmap of the standardized mean difference of each of the five scalar biomarkers between the held-out source and the pooled training sources. The colour scale is diverging and centred at zero; exact values are annotated. All five biomarkers are strongly source-dependent in every fold, with absolute SMD between 0.30 and 1.11, i.e. up to a full training standard deviation of shift. Plus lies highest and Farabi lowest on all five. This information was not used to retune any model.

## Figure DA1 — Task 12 domain-alignment performance

Panel a: three-class macro AUC on each held-out source for the matched Task 12 pair, K0 (domain-neutral control) and K1 (class-conditional DANN + MMD). Panel b uses a different axis and reports the **restricted Normal-vs-Plus** binary AUC for the Plus fold, where a three-class AUC is undefined. The held-out source was never used in training, validation, alignment, early stopping or model selection. K1 did not improve on K0 in either comparable fold.

## Figure DA2 — Representation shift before and after alignment

Standardized centroid distance and median absolute standardized mean difference between the two training domains, measured on the source-validation split of the learned 256-d representation for K0 and K1. Alignment reduced both statistics in all three folds, most strongly for the Farabi fold. No held-out source information entered this diagnostic.

## Figure DA3 — Domain predictability after alignment

Five-fold cross-validated accuracy of a logistic regression trained on the source-validation representation to predict which of the two training domains a sample came from, for K0 and K1. The dashed line marks chance (0.50). Despite the reduced representation shift shown in DA2, domain identity remained highly predictable at 0.88-0.98 in every fold, so the alignment did not remove source information from the representation.

## Figure A1 — Attention weights by class and source

Panel a: mean soft-attention weight assigned to the ResNet50 branch by true class on the canonical test set. Panel b: the same weight grouped by acquisition source. **Attention weights are descriptive diagnostics only and must not be interpreted as purely disease-specific explanations, because strong source dependence was observed**: the mean ResNet weight is 0.71 on Plus, 0.19 on FARFUM-RoP and 0.04 on Farabi, which is close to a source fingerprint. These weights were never used for model selection.

## Figure A2 — Attention distribution

Distribution of the ResNet50 soft-attention weight at the selected Task 13 checkpoint, summarised by the 5th-95th percentile range with median and mean markers. Validation: mean 0.5335, sd 0.3991, p05 0.0105, median 0.5997, p95 0.9930. Test: mean 0.5111, sd 0.3893, p05 0.0087, median 0.5141, p95 0.9925. **ATTENTION_COLLAPSE = NO**: no branch mean approaches the 0.95 threshold, although the large standard deviation and extreme percentiles show that the attention is effectively near-binary per image, committing to one backbone for most images. Descriptive diagnostic only.

## Figure FINAL1 — Thesis summary

Three-panel summary. Panel A: paired increments on the canonical test set. Adding five scalar biomarkers to the RGB embedding gives delta AUC +0.001085 [-0.001691, +0.003868], p = 0.431; adding the spatial vessel representation gives +0.007938 [+0.002630, +0.013131], p = 0.0031; adding the biomarkers after the vessel representation gives +0.002039 [-0.000274, +0.004397], p = 0.090. Panel B: the vessel increment replicates under source-held-out evaluation (FARFUM-RoP +0.022242, p < 0.0001; Farabi +0.012954, p = 0.0018) and under a different RGB architecture (dual-RGB M1 - M0 +0.020936, p < 0.0001). Panel C: median absolute standardized mean difference between held-out and training sources for the RGB and vessel embeddings. Headline: spatial vessel representations provide reproducible complementary predictive information across distinct RGB representations and under source-held-out evaluation, whereas five scalar vascular summaries show little incremental predictive value beyond learned image representations. This is not a claim of universal generalisation.

## Figure SEG1 — External segmentation performance - NOT AVAILABLE

The external HVDROPDB segmentation benchmark table (Dice, clDice, precision, recall by RetCam and Neo) is **NOT AVAILABLE** in the frozen artifact set. A search of both servers found no authoritative per-camera Dice/clDice table; only `hvdro_evidence_lineage.csv` and `seg_current_v1_summary.json` exist, which are provenance records and do not contain these metrics. No values are reported rather than approximated.

## Figure SEG2 — Example segmentations - NOT AVAILABLE

Example segmentation panels are **NOT AVAILABLE**. Producing them would require expert reference masks for the specific selected cases, and no such expert masks exist in the frozen artifact set. Per the task instruction, this figure was skipped rather than fabricated.

## Table captions

### Table D1_cohort_summary — Cohort summary

Complete-case cohort by acquisition source: images, groups, class counts and percentage of the 8,862-image total. The Plus source contains no Pre-Plus, which is why the three-class metric is undefined for it in every held-out analysis.

### Table D2_canonical_split_summary — Canonical split summary

Class and group counts within the canonical group-disjoint training, validation and test splits. The split is group-disjoint by construction; no group appears in more than one split.

### Table R1_master_model_results — Master model results

Canonical test-set performance (N = 1,331) for the primary models. AUC, balanced accuracy and macro F1 are higher-is-better; Brier and ECE are lower-is-better. Bold marks the numerically best value in each metric column; no composite score or ranking is used. F_RGB_VESSEL_LATE_FUSION is listed for completeness and is identical to B_EMBEDDING_ONLY because its mixture coefficient saturated at alpha = 1.0.

### Table R2_per_class_auc — Per-class AUC

One-vs-rest AUC per class on the canonical test set (N = 1,331: Normal 982, Pre-Plus 140, Plus 209). Frozen Task 6 and Task 8 results.

### Table C1_primary_paired_statistics — Primary paired statistics

Paired class-stratified bootstrap statistics on the canonical test set (seed 42, 10,000 replicates, N = 1,331). C vs B is the Task 7 comparison; E vs B and G vs E are the Task 8B 10,000-replicate closure.

### Table C2_architecture_robustness — Architecture robustness of the vessel contribution

The same frozen 1,792-d vessel embedding added to two different RGB representations, each with its own matched RGB-only control and its own paired bootstrap. The two rows are independent experiments, not one model comparison. This is a main table for the thesis.

### Table B1_biomarker_evidence_summary — Biomarker evidence summary

Every experiment in which the five scalar biomarkers were tested, with the frozen paired delta, 95% CI and p value. Interpretation uses controlled language: supported positive increment, no statistically supported increment, or statistically supported decrease. The final row uses the restricted Normal-vs-Plus binary AUC. This table does not support the statement that biomarkers are useless.

### Table N1_negative_architecture_experiments — Negative architecture experiments

Secondary exploratory neural models that did not improve on the frozen concatenation models (canonical test set, N = 1,331). These are not primary results and are reported for transparency.

### Table S1_source_heldout_performance — Source-held-out performance

LOSO performance. The held-out source contributed nothing to fitting, preprocessing or model choice. Three-class AUC is undefined for Plus (no Pre-Plus); the restricted Normal-vs-Plus binary AUC is reported separately and is not a substitute for the three-class metric.

### Table DS1_domain_shift_summary — Domain shift summary

Descriptive shift statistics for the frozen RGB and vessel embeddings, computed without target labels.

### Table DA1_task12_results — Task 12 results

Class-conditional domain-invariance results. K0 is the matched domain-neutral control and K1 adds the GRL domain discriminator and class-conditional MMD with coefficients fixed at 0.10. The Plus row reports the restricted Normal-vs-Plus binary AUC for the AUC line.

### Table FINAL1_experiment_summary — Complete experiment summary

One row per experiment. Historical Phase-5 results are excluded by construction. No claim is made beyond what the frozen statistics support.

