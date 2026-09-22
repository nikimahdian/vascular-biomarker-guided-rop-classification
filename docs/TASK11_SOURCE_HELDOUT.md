# Task 11 — Source-Held-Out (LOSO) Robustness Baseline

Status: `TASK11_STATUS = COMPLETE`
Classification: `SECONDARY_SOURCE_GENERALIZATION_ANALYSIS`
Diagnostic only — no domain adaptation method was started, no CNN was executed, no embedding was
regenerated, and no held-out source ever influenced fitting, preprocessing or model choice.

**Headline: under source shift, absolute performance collapses (3-class AUC ≈ 0.77–0.81 vs 0.93 in
the canonical mixed split), but the spatial vessel representation's contribution survives the shift
while the scalar biomarker panel fails again — for the fourth time.**

## 1. Frozen inputs and contract

| input | shape | status |
|---|---|---|
| Task-6 B_RGB embedding `embeddings_b5_8862.parquet` | 8862 × 2048 | SHA unchanged ✓ |
| Task-8 D vessel embedding `vessel_embeddings_b4_8862.parquet` | 8862 × 1792 | SHA unchanged ✓ |
| `FINAL_BIOMARKERS_V2` five scalars | 8862 × 5 | finite ✓ |

Frozen downstream XGBoost configuration, taken verbatim from
`artifacts/task6/downstream_xgb_selected.json` and used **identically for every model in every
fold**: `n_estimators 500, max_depth 3, learning_rate 0.1, subsample 0.8, colsample_bytree 0.8,
min_child_weight 1, reg_lambda 1.0`, seed 42, weighted CE-equivalent sample weights
0.46051 / 3.18103 / 1.94512 from Task 6. No per-source tuning of any parameter, feature set,
preprocessing step, class weight or capacity. No scaler is fitted anywhere in this task except
train-side SMD statistics used for descriptive diagnostics only.

Formulations: `B_LOSO` = 2048-d RGB · `E_LOSO` = RGB + 1792-d vessel (3840) · `G_LOSO` = RGB +
vessel + 5 biomarkers (3845).

## 2. Population audit

| source | n | groups | Normal | Pre_Plus | Plus |
|---|---|---|---|---|---|
| plus | 5925 | 186 | 5304 | **0** | 621 |
| farfum_rop | 1528 | 68 | 780 | 478 | 270 |
| farabi | 1409 | 160 | 369 | 452 | 588 |

Total 8,862. `plus` contains no Pre_Plus at all and is 89.5 % Normal.

## 3. Folds

| fold | held out | training sources | n train | n held-out test | classes in test |
|---|---|---|---|---|---|
| A | plus | farfum_rop + farabi | 2937 | 5925 | {Normal, Plus} |
| B | farfum_rop | plus + farabi | 7334 | 1528 | {Normal, Pre_Plus, Plus} |
| C | farabi | plus + farfum_rop | 7453 | 1409 | {Normal, Pre_Plus, Plus} |

Every fold's training set contains all three classes (verified and asserted before fitting), so the
3-class XGBoost objective is valid in all nine fits. No image or group from the held-out source
entered training; the script aborts if it detects leakage.

## 4. Metrics per held-out source

**Held out plus** (n = 5925; Pre_Plus absent, so 3-class AUC is undefined and is reported as NaN,
never substituted):

| model | restricted binary AUC (Normal vs Plus) | balanced acc | macro F1 | Brier (3-col) | ECE |
|---|---|---|---|---|---|
| B_LOSO | 0.983824 | 0.888654 | 0.867921 | 0.130810 | 0.032410 |
| E_LOSO | 0.984579 | 0.901969 | 0.888724 | 0.104545 | 0.016582 |
| G_LOSO | 0.984132 | 0.902853 | 0.888583 | 0.096911 | 0.017145 |

The 3-col Brier for this fold is reported with an explicit caveat: its one-hot target has a
Pre_Plus column that is structurally always zero, so the 3-column Brier is inflated relative to a
genuine two-class Brier and is not comparable to the other folds or to the canonical values. A
restricted two-class Brier on the renormalised Normal/Plus columns is also computed and stored in
`metrics_per_source.csv` and `predictions_plus.csv`; its paired deltas are in §5. It is a
separately labelled restricted metric and is not the project's standard multiclass Brier.

**Held out farfum_rop** (n = 1528, all three classes):

| model | 3-class macro AUC | balanced acc | macro F1 | Brier | ECE |
|---|---|---|---|---|---|
| B_LOSO | 0.787537 | 0.632840 | 0.633878 | 0.547509 | 0.214301 |
| E_LOSO | 0.809779 | 0.663516 | 0.658966 | 0.510685 | 0.183187 |
| G_LOSO | 0.806198 | 0.657584 | 0.651424 | 0.520413 | 0.199070 |

**Held out farabi** (n = 1409, all three classes):

| model | 3-class macro AUC | balanced acc | macro F1 | Brier | ECE |
|---|---|---|---|---|---|
| B_LOSO | 0.769944 | 0.559828 | 0.532086 | 0.648428 | 0.258291 |
| E_LOSO | 0.782898 | 0.586040 | 0.572047 | 0.618713 | 0.239106 |
| G_LOSO | 0.784477 | 0.574624 | 0.563077 | 0.613727 | 0.247180 |

Per-class AUCs and confusion matrices are in `metrics_per_source.csv` and
`confusion_{model}_{source}.csv` / `confusion_matrices.png`.

## 5. Paired class-stratified bootstrap — 10,000 replicates, seed 42

For the two three-class folds the standard metric set is used. For the plus fold the metric set is
restricted to what its two-class label distribution supports, and every restricted number is
labelled as such.

**E_LOSO − B_LOSO**

| fold | metric | delta | 95% CI | p |
|---|---|---|---|---|
| farfum_rop | 3-class AUC | **+0.022242** | [+0.014802, +0.029907] | <0.0001 |
| farfum_rop | balanced acc | **+0.030676** | [+0.014126, +0.047013] | 0.0005 |
| farfum_rop | macro F1 | **+0.025087** | [+0.007679, +0.042199] | 0.0051 |
| farfum_rop | Brier | **−0.036824** | [−0.054076, −0.019765] | <0.0001 |
| farfum_rop | ECE | **−0.031115** | [−0.047811, −0.012180] | 0.0008 |
| farabi | 3-class AUC | **+0.012954** | [+0.004984, +0.021019] | 0.0018 |
| farabi | balanced acc | **+0.026213** | [+0.006431, +0.045944] | 0.0082 |
| farabi | macro F1 | **+0.039962** | [+0.018671, +0.060945] | 0.0001 |
| farabi | Brier | **−0.029716** | [−0.048599, −0.010733] | 0.0016 |
| farabi | ECE | −0.019182 | [−0.036153, +0.002724] | 0.0582 |
| plus (restricted) | binary AUC | +0.000754 | [−0.001369, +0.002783] | 0.4847 |
| plus (restricted) | balanced acc | **+0.013315** | [+0.003799, +0.023064] | 0.0069 |
| plus (restricted) | macro F1 | **+0.020803** | [+0.011670, +0.030019] | <0.0001 |
| plus (restricted) | Brier (2-class) | **−0.014038** | [−0.018415, −0.009730] | <0.0001 |
| plus (restricted) | ECE | **−0.015835** | [−0.021216, −0.010332] | <0.0001 |

**G_LOSO − E_LOSO**

| fold | metric | delta | 95% CI | p |
|---|---|---|---|---|
| farfum_rop | 3-class AUC | −0.003581 | [−0.008233, +0.001189] | 0.1339 |
| farfum_rop | balanced acc | −0.005932 | [−0.019128, +0.007894] | 0.3864 |
| farfum_rop | macro F1 | −0.007542 | [−0.021019, +0.006535] | 0.2794 |
| farfum_rop | Brier | +0.009728 | [−0.001394, +0.020693] | 0.0805 |
| farfum_rop | ECE | **+0.015883** | [+0.001684, +0.029108] | 0.0246 |
| farabi | 3-class AUC | +0.001579 | [−0.002785, +0.005956] | 0.4839 |
| farabi | balanced acc | −0.011416 | [−0.026732, +0.004111] | 0.1429 |
| farabi | macro F1 | −0.008970 | [−0.025118, +0.007346] | 0.2702 |
| farabi | Brier | −0.004985 | [−0.015616, +0.005441] | 0.3553 |
| farabi | ECE | +0.008077 | [−0.009038, +0.019660] | 0.2959 |
| plus (restricted) | binary AUC | −0.000446 | [−0.001982, +0.001100] | 0.5705 |
| plus (restricted) | balanced acc | +0.000884 | [−0.004941, +0.006426] | 0.7613 |
| plus (restricted) | macro F1 | −0.000140 | [−0.006113, +0.005829] | 0.9599 |
| plus (restricted) | Brier (2-class) | +0.000476 | [−0.001889, +0.002758] | 0.6896 |
| plus (restricted) | ECE | +0.000563 | [−0.004408, +0.004260] | 0.7990 |

## 6. Domain-shift summary

| held out | B AUC | E AUC | G AUC | Δ E−B | Δ G−E |
|---|---|---|---|---|---|
| plus (restricted binary) | 0.983824 | 0.984579 | 0.984132 | +0.000754 | −0.000446 |
| farfum_rop (3-class) | 0.787537 | 0.809779 | 0.806198 | +0.022242 | −0.003581 |
| farabi (3-class) | 0.769944 | 0.782898 | 0.784477 | +0.012954 | +0.001579 |

Descriptive degradation against the canonical mixed-source values (B = 0.924903, E = 0.932841,
G = 0.934880). Populations differ, so these are **not** paired comparisons and carry no
significance:

| held out | B degradation | E degradation | G degradation |
|---|---|---|---|
| farfum_rop | −0.137366 | −0.123062 | −0.128682 |
| farabi | −0.154959 | −0.149943 | −0.150403 |
| plus | +0.058921 | +0.051738 | +0.049252 |

The plus row is **not** a like-for-like comparison and must not be read as improvement under shift:
its number is a restricted two-class Normal-vs-Plus AUC on a test set that is 89.5 % Normal, whereas
the canonical reference is a three-class macro OVR AUC. The two quantities measure different tasks.
Only the farfum_rop and farabi rows are comparable, and both show a loss of 0.12–0.155 AUC.

## 7. Feature-distribution shift diagnostics (no labels used)

Per-feature standardized mean difference `SMD_j = (mean_target,j − mean_train,j) / std_train,j`, with
`std_train` from the pooled training sources of that fold.

| block | held out | standardized centroid distance | median abs SMD | p90 abs SMD | p95 abs SMD |
|---|---|---|---|---|---|
| RGB embedding (2048-d) | plus | 56.511 | 0.5406 | 1.8314 | 2.5360 |
| RGB embedding | farfum_rop | 26.038 | 0.3793 | 0.8431 | 1.1086 |
| RGB embedding | farabi | 35.253 | 0.4966 | 1.1258 | 1.4847 |
| vessel embedding (1792-d) | plus | 63.404 | 1.2291 | 2.3345 | 2.6296 |
| vessel embedding | farfum_rop | 22.953 | 0.5565 | 0.7307 | 0.7579 |
| vessel embedding | farabi | 28.324 | 0.6972 | 0.8829 | 0.9280 |

`plus` is the most shifted source in both blocks and the vessel embedding is shifted substantially
more than the RGB embedding in every fold (plus: median |SMD| 1.23 vs 0.54). Farfum_rop is the least
shifted block-wise despite being the smallest source. `domain_shift.png` plots these.

## 8. Biomarker shift

Per-source descriptive statistics (all five biomarkers):

| source | biomarker | mean | std | median | IQR |
|---|---|---|---|---|---|
| plus | vessel_density_fov | 0.10724 | 0.03380 | 0.10611 | 0.04452 |
| plus | skel_density_fov | 0.02208 | 0.01060 | 0.02024 | 0.01167 |
| plus | fractal_d0 | 1.33468 | 0.08194 | 1.34691 | 0.10053 |
| plus | fractal_d1 | 1.30850 | 0.08273 | 1.32152 | 0.09671 |
| plus | fractal_d2 | 1.29585 | 0.08261 | 1.30886 | 0.09501 |
| farfum_rop | vessel_density_fov | 0.08551 | 0.03294 | 0.08141 | 0.03855 |
| farfum_rop | skel_density_fov | 0.01581 | 0.00953 | 0.01379 | 0.00803 |
| farfum_rop | fractal_d0 | 1.28170 | 0.07765 | 1.28718 | 0.09934 |
| farfum_rop | fractal_d1 | 1.26287 | 0.07814 | 1.27065 | 0.10040 |
| farfum_rop | fractal_d2 | 1.25444 | 0.07798 | 1.26218 | 0.09825 |
| farabi | vessel_density_fov | 0.06725 | 0.02519 | 0.06613 | 0.03319 |
| farabi | skel_density_fov | 0.01057 | 0.00437 | 0.01014 | 0.00527 |
| farabi | fractal_d0 | 1.24269 | 0.08329 | 1.24583 | 0.11457 |
| farabi | fractal_d1 | 1.22650 | 0.08536 | 1.23107 | 0.11447 |
| farabi | fractal_d2 | 1.21889 | 0.08549 | 1.22558 | 0.11415 |

Train-vs-held-out standardized mean differences:

| held out | vessel_density | skel_density | fractal_d0 | fractal_d1 | fractal_d2 |
|---|---|---|---|---|---|
| plus | +0.988 | +1.105 | +0.867 | +0.754 | +0.700 |
| farfum_rop | −0.391 | −0.379 | −0.393 | −0.335 | −0.301 |
| farabi | −1.023 | −0.956 | −0.968 | −0.866 | −0.821 |

Every one of the five scalar biomarkers is strongly source-dependent in every fold — absolute SMD
between 0.30 and 1.11, i.e. up to a full training standard deviation of shift. `plus` sits highest
and `farabi` lowest on all five, consistently. These biomarkers are therefore largely encoding
acquisition characteristics rather than transferable pathology, which is a direct mechanistic
explanation for why they have never added incremental value. This information was not used to
retune G.

## 9. What Task 11 establishes

1. **Absolute cross-source performance is poor.** Holding out farfum_rop or farabi costs
   0.12–0.155 three-class AUC relative to the mixed-source canonical result (0.79 and 0.77 against
   0.925/0.933/0.935). ECE rises from 0.09 to 0.18–0.26 and Brier from 0.25 to 0.51–0.65. The
   frozen representations do not generalize cleanly across acquisition sources, and their
   probabilities are not trustworthy out of source.
2. **The vessel branch's contribution survives the shift — and this is the most reproducible result
   in the whole series.** `E − B` is positive in all three folds and significant in both comparable
   ones (farfum_rop: +0.0222 AUC, p < 0.0001; farabi: +0.0130 AUC, p = 0.0018), with balanced
   accuracy, macro F1 and Brier all moving the same way. In the plus fold the restricted binary AUC
   gain is not significant (p = 0.485) but balanced accuracy, macro F1, 2-class Brier and ECE all
   improve significantly. Adding the spatial vessel representation helps under source shift, in the
   same direction as in the canonical split (Task 8B: +0.0079 AUC, p = 0.0031).
3. **The five scalar biomarkers fail again, now under domain shift.** `G − E` is null or slightly
   negative in every fold (AUC p = 0.134, 0.484, 0.571) and the one significant effect is *worse*
   calibration on farfum_rop (ECE +0.0159, p = 0.0246). That is the fourth independent failure:
   concatenation in the canonical split (Task 8B, p = 0.090), learned joint fusion (Task 9,
   p = 0.0049, harmful), FiLM conditioning (Task 10, no effect), and now source-held-out
   concatenation.
4. **The shift diagnostics suggest why.** The vessel embedding is the most shifted block yet still
   helps, and the biomarkers are strongly source-dependent (|SMD| up to 1.1) yet add nothing. Being
   sensitive to the acquisition source is not by itself what prevents a representation from being
   useful; the biomarkers simply carry no information that the frozen RGB and vessel embeddings do
   not already contain.
5. **This is the baseline Task 12 should be measured against.** Any domain adaptation method must be
   compared against these numbers using the same folds, the same frozen features and the same
   XGBoost configuration, with no target labels.

## 10. Limitations, stated plainly

1. `plus` cannot yield a three-class metric. Its column is reported as restricted Normal-vs-Plus and
   is not evidence of better transfer.
2. The plus evaluation set is 89.5 % Normal and 0 % Pre_Plus, so its balanced accuracy and macro F1
   are computed over two classes only.
3. Fold training-set sizes differ enormously (2937 vs 7334 vs 7453), so cross-fold differences mix
   domain shift with training-set size. Only within-fold B/E/G comparisons are matched.
4. Degradation values against the canonical split compare different populations and are descriptive
   only; no significance is claimed.
5. Group-level structure was respected by construction (whole sources are held out), but no
   patient-level LOSO was performed — that is a different and harder question.
6. No operational, screening, or clinical claim follows from any of these numbers.

## 11. Artifacts

`artifacts/task11_source_heldout/`: `fold_definitions.json` · `population_audit.csv` ·
`metrics_per_source.csv` · `domain_shift_summary.csv` · `paired_bootstrap_10k.csv` and the six
per-comparison files · `domain_shift_diagnostics.csv` · `biomarker_shift.csv` ·
`predictions_{source}.csv` (per-image probabilities for all three models) · `probs_{model}_{source}.npy` ·
nine `{model}_{heldout}.json` XGBoost models · confusion matrices per model and fold ·
`calibration_by_heldout_source.png` · `confusion_matrices.png` · `domain_shift.png` ·
`task11_summary.json` · `artifact_sha256.json`.
