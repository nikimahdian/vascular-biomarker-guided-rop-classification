# Claim-evidence matrix

Generated 2026-09-22T17:11:21Z from frozen artifacts only. No claim below exceeds what the frozen statistics support.

## Claim 1

**CLAIM.** Scalar vascular biomarkers contain predictive signal.

**SUPPORTING EXPERIMENT.** Task 6, model A_PRIMARY

**SUPPORTING METRIC.** Multiclass macro OVR AUC on the canonical test set = 0.659582

**CI / p.** Point estimate only; no paired comparison against chance was run for A_PRIMARY

**WHAT WE ARE ALLOWED TO SAY.** A model using only the five scalar biomarkers achieves AUC 0.659582 on the canonical test set, i.e. clearly above chance and well below every RGB-based model.

**WHAT WE MUST NOT SAY.** Do not claim that the scalar biomarkers are sufficient, clinically usable, or a screening tool; and do not claim a statistically tested improvement over chance, because no such test was performed.

## Claim 2

**CLAIM.** Five scalar biomarkers show little incremental discrimination beyond frozen RGB representations.

**SUPPORTING EXPERIMENT.** Task 6/7 C vs B_EMBEDDING_ONLY; Task 8B G vs E

**SUPPORTING METRIC.** Delta multiclass AUC

**CI / p.** C - B: +0.001085, 95% CI [-0.001691, +0.003868], p = 0.431 (10,000 replicates). G - E: +0.002039, 95% CI [-0.000274, +0.004397], p = 0.090

**WHAT WE ARE ALLOWED TO SAY.** In both canonical configurations the five scalar biomarkers produced a small positive point estimate whose paired confidence interval includes zero; no incremental discrimination is statistically supported.

**WHAT WE MUST NOT SAY.** Do not say the biomarkers are useless or carry no information; both point estimates are positive and the intervals are wide.

## Claim 3

**CLAIM.** Spatial vessel representations provide measurable complementary information beyond RGB embeddings.

**SUPPORTING EXPERIMENT.** Task 8 and Task 8B, E vs B_EMBEDDING_ONLY

**SUPPORTING METRIC.** Delta multiclass AUC; Delta Brier

**CI / p.** AUC +0.007938, 95% CI [+0.002630, +0.013131], p = 0.0031; Brier -0.027693, 95% CI [-0.043954, -0.011471], p = 0.0011

**WHAT WE ARE ALLOWED TO SAY.** Adding the frozen 1,792-d spatial vessel representation to the 2,048-d RGB embedding significantly improves both ranking and probability quality on the canonical test set, with paired confidence intervals excluding zero.

**WHAT WE MUST NOT SAY.** Do not claim improvement in thresholded decision metrics: balanced accuracy, macro F1 and ECE all had intervals crossing zero in the same analysis.

## Claim 4

**CLAIM.** The spatial-vessel benefit persists in FARFUM and Farabi source-held-out analyses.

**SUPPORTING EXPERIMENT.** Task 11 LOSO folds

**SUPPORTING METRIC.** Delta multiclass AUC (E_LOSO - B_LOSO) within each fold

**CI / p.** FARFUM-RoP +0.022242, 95% CI [+0.014802, +0.029907], p < 0.0001; Farabi +0.012954, 95% CI [+0.004984, +0.021019], p = 0.0018

**WHAT WE ARE ALLOWED TO SAY.** When an entire acquisition source is absent from fitting, adding the vessel representation still improves held-out discrimination in both comparable folds, with balanced accuracy, macro F1 and Brier moving in the same direction.

**WHAT WE MUST NOT SAY.** Do not present the Plus held-out fold as comparable: its AUC is a restricted two-class Normal-vs-Plus metric on a test set that is 89.5% Normal, not the three-class metric used elsewhere.

## Claim 5

**CLAIM.** Scalar biomarkers do not show reproducible incremental benefit after the spatial vessel representation is available.

**SUPPORTING EXPERIMENT.** Task 8B G vs E; Task 9 I vs H; Task 10 J1 vs J0; Task 11 G_LOSO vs E_LOSO

**SUPPORTING METRIC.** Delta multiclass AUC in four independent regimes

**CI / p.** Canonical +0.002039, p = 0.090; joint model -0.009578, p = 0.0049; FiLM -0.002123, p = 0.068; held-out FARFUM p = 0.134, Farabi p = 0.484, Plus p = 0.571

**WHAT WE ARE ALLOWED TO SAY.** Across four independent regimes the five scalar biomarkers never produced a statistically supported positive increment over a representation that already contains the spatial vessel information; in the learned joint model the effect was significantly negative.

**WHAT WE MUST NOT SAY.** Do not generalise this to 'biomarkers are useless' or to other feature sets, cohorts or endpoints. The result is specific to these five features, this population and this 3-class ROP-Plus task.

## Claim 6

**CLAIM.** Cross-source generalization is substantially worse than mixed-source canonical performance.

**SUPPORTING EXPERIMENT.** Task 11 source-held-out folds vs Task 6/8 canonical split

**SUPPORTING METRIC.** Multiclass macro OVR AUC

**CI / p.** FARFUM-RoP E 0.809779 vs canonical E 0.932841 (-0.123062); Farabi E 0.782898 vs 0.932841 (-0.149943). Descriptive; populations differ so no paired test applies

**WHAT WE ARE ALLOWED TO SAY.** Holding out an entire acquisition source costs roughly 0.12 to 0.15 AUC for the same model formulation, and calibration degrades sharply (ECE rising from 0.092 to 0.18-0.26).

**WHAT WE MUST NOT SAY.** Do not attach a p value to the canonical-versus-held-out difference, because the two evaluations use different populations and the comparison is descriptive only.

## Claim 7

**CLAIM.** RGB, vessel and biomarker representations are source-dependent.

**SUPPORTING EXPERIMENT.** Task 11 feature-distribution and biomarker shift diagnostics

**SUPPORTING METRIC.** Per-feature standardized mean difference (SMD) between held-out and training sources

**CI / p.** No inferential test; descriptive statistics. RGB median |SMD| 0.379-0.541; vessel median |SMD| 0.557-1.229; biomarker |SMD| 0.301-1.105

**WHAT WE ARE ALLOWED TO SAY.** All three representation families carry strong source-dependent shift; the vessel embedding is the most shifted block, and the five scalar biomarkers shift by up to one training standard deviation.

**WHAT WE MUST NOT SAY.** Do not interpret these SMD values as causal or as evidence that a representation is unusable: the vessel block is the most shifted and still contributes the most.

## Claim 8

**CLAIM.** Class-conditional DANN+MMD with the fixed Task-12 configuration reduced some representation-shift diagnostics but did not consistently improve held-out disease performance.

**SUPPORTING EXPERIMENT.** Task 12, K1 vs K0 across three source-held-out folds

**SUPPORTING METRIC.** Delta multiclass AUC and shift diagnostics

**CI / p.** K1 - K0 AUC -0.002796 (FARFUM, p = 0.143) and -0.000883 (Farabi, p = 0.259); centroid distance and median |SMD| reduced in all three folds; domain predictability unchanged at 0.8802-0.9824

**WHAT WE ARE ALLOWED TO SAY.** With coefficients fixed at 0.10, the alignment reduced representation shift but did not improve held-out disease classification, and source identity remained almost perfectly predictable. Outcome B of the pre-declared interpretation.

**WHAT WE MUST NOT SAY.** Do not claim domain invariance was achieved, and do not generalise to domain-invariant learning in general: only one method at one fixed setting was tested.

## Claim 9

**CLAIM.** The project does NOT establish that biomarkers are clinically useless.

**SUPPORTING EXPERIMENT.** Whole project

**SUPPORTING METRIC.** -

**CI / p.** -

**WHAT WE ARE ALLOWED TO SAY.** The five scalar biomarkers remain interpretable, measurable, FOV-aware vascular descriptors. They underperformed as additional predictors in this specific 3-class classification setting.

**WHAT WE MUST NOT SAY.** Do not state or imply that scalar vascular biomarkers have no clinical value, that they should be abandoned, or that they are uninformative in general.

## Claim 10

**CLAIM.** The project does NOT establish external clinical validation.

**SUPPORTING EXPERIMENT.** Whole project

**SUPPORTING METRIC.** -

**CI / p.** -

**WHAT WE ARE ALLOWED TO SAY.** All results come from a single retrospective multi-source cohort split at the group level, evaluated as a canonical locked-split reanalysis and secondary exploratory analyses. HVDROPDB was used as an external development and measurement benchmark, not as clinical validation, and its segmentation metrics are not available as frozen artifacts.

**WHAT WE MUST NOT SAY.** Do not describe any model as clinically validated, deployment-ready, screening-capable, or prospectively evaluated. Do not describe the test split as an untouched confirmatory holdout.

## Claim 11

**CLAIM.** Spatial vessel complementarity is not specific to the original EfficientNet-B5 RGB representation.

**SUPPORTING EXPERIMENT.** Task 13, M1 vs M0

**SUPPORTING METRIC.** Delta multiclass AUC

**CI / p.** +0.020936, 95% CI [+0.011426, +0.030828], p < 0.0001

**WHAT WE ARE ALLOWED TO SAY.** The complementary value of the frozen vessel representation was reproduced with a distinct dual-backbone attention RGB representation, and the increment there was larger than with the original embedding.

**WHAT WE MUST NOT SAY.** Do not claim the vessel representation will improve every RGB model, or that it has been shown to be universally complementary.

## Claim 12

**CLAIM.** The tested ROPDeepX-style RGB representation did not outperform the original EfficientNet-B5 embedding under the project's canonical protocol.

**SUPPORTING EXPERIMENT.** Task 13, M0 vs B_EMBEDDING_ONLY

**SUPPORTING METRIC.** Delta multiclass AUC

**CI / p.** -0.018925, 95% CI [-0.032594, -0.005640], p = 0.0063

**WHAT WE ARE ALLOWED TO SAY.** Under this dataset and evaluation protocol the dual-backbone attention RGB representation was inferior to the original single-backbone EfficientNet-B5 embedding.

**WHAT WE MUST NOT SAY.** Do not state that ROPDeepX or dual-backbone attention architectures are inferior in general, and do not present this as a controlled architecture ablation: the two pipelines differ in learner and training regime as well as in backbone.
