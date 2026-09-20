# Final biomarker freeze V1 — Task 5D

```
FINAL_SEGMENTATION_GENERATION:   SEG_CURRENT_V1
REJECTED_CANDIDATE:              SEG_GEOMETRY_CANDIDATE_V2
REASON_REJECTED:                 Failed the predeclared synthetic geometry gate

FINAL_PRIMARY_FEATURE_N:         5
FINAL_PRIMARY_FEATURES:          vessel_density_fov, skel_density_fov,
                                 fractal_d0, fractal_d1, fractal_d2

FINAL_SECONDARY_FEATURE_N:       3
FINAL_SECONDARY_FEATURES:        tort_geodesic_median, tort_geodesic_p90,
                                 tort_geodesic_top3_mean

DISC_CONDITIONAL_FEATURE_N:      8

FINAL_PRIMARY_ROW_COMPLETENESS:  0.9991   (8862 / 8870)

RETCAM_PRIMARY_FEATURE_EVIDENCE:
    vessel_density_fov  ICC 0.872 [0.789,0.995]  bias +0.0158  MAE 0.0192  rho 0.942
    skel_density_fov    ICC 0.933 [0.886,0.998]  bias -0.0016  MAE 0.0016  rho 0.951
    fractal_d0          ICC 0.940 [0.897,0.999]  bias -0.0183  MAE 0.0202  rho 0.932
    fractal_d1          ICC 0.923 [0.870,0.998]  bias -0.0246  MAE 0.0260  rho 0.927
    fractal_d2          ICC 0.911 [0.851,0.997]  bias -0.0273  MAE 0.0291  rho 0.924
    -> STRONG. Four of five at ICC >= 0.91, all biases small, all N = 50.

NEO_PRIMARY_FEATURE_EVIDENCE:
    vessel_density_fov  ICC 0.774 [0.633,0.968]  bias +0.0001  MAE 0.0129  rho 0.751
    skel_density_fov    ICC 0.547 [0.391,0.944]  bias -0.0050  MAE 0.0051  rho 0.816
    fractal_d0          ICC 0.569 [0.410,0.942]  bias -0.0779  MAE 0.0786  rho 0.820
    fractal_d1          ICC 0.578 [0.418,0.943]  bias -0.0817  MAE 0.0821  rho 0.830
    fractal_d2          ICC 0.566 [0.404,0.937]  bias -0.0817  MAE 0.0825  rho 0.815
    -> MODERATE. Same ordering, but ICC roughly 0.35 lower and the fractal family carries a
       systematic -0.08 bias. All N = 50. No feature is dropped for being weaker on Neo;
       the weakness is recorded and Neo is never pooled silently.

PAIRED_RETCAM_DD_N:              25 verified pixel-identity pairs (24 evaluable end-to-end)
                                 Neo = 0 -> DEFERRED_TO_TARGET_DOMAIN_EXPERT_VALIDATION

PAIRED_RETCAM_DD_AGREEMENT:
    SEG-ONLY vs GOLD      ring_2_3dd  N=21 ICC 0.821  bias +0.0312  MAE 0.0317
                          ring_3_6dd  N=21 ICC 0.773  bias +0.0286  MAE 0.0286
                          width_p50_dd  N=25 ICC 0.341  bias +0.0328  MAE 0.0328
                          width_p90_dd  N=25 ICC 0.125  bias +0.1116  MAE 0.1116
                          width_ann_p90_dd N=25 ICC 0.047  bias +0.1705  MAE 0.1705
    -> the ring densities transfer; the caliber features carry a POSITIVE systematic bias of
       +0.03 to +0.17 DD. The automatic vessel mask over-measures caliber relative to expert.

AUTO_DISC_INCREMENTAL_ERROR:
    AUTO vessel + AUTO disc  vs  AUTO vessel + EXPERT disc
    median MAE 0.00454 DD, median bias 0.00284 DD
    per-feature ICC 0.977 to 0.999
    -> the automatic disc contributes almost no additional error. Vessel segmentation, not
       disc estimation, is the dominant source of DD-measurement error.

FINAL_FEATURE_TABLE_SHA256:      f1c41e923ae29d4e228097536765f5e7399963062253ad925657a077cddc10c0

TARGET_DOMAIN_EXPERT_VALIDATION_STILL_REQUIRED:  YES

PRETRAINING_FEATURE_FREEZE:      PASS

TASK5D_STATUS:                   COMPLETE
```

> `TECHNICALLY VALIDATED AND EXTERNALLY BENCHMARKED`. **Not** `CLINICALLY VALIDATED`.
> HVDROPDB is development/benchmark data this project already used.

---

## B. Final segmentation decision

`FINAL_SEGMENTATION_GENERATION = SEG_CURRENT_V1`. `SEG_GEOMETRY_CANDIDATE_V2` rejected because it
failed the predeclared synthetic geometry gate. **No bytes of `SEG_CURRENT_V1` were changed**;
the contract `configs/segmentation_generation_current_v1.yaml` still verifies, and
`python -m src.segmentation.verify_generation --generation SEG_CURRENT_V1` returns `[OK]`.

Preserved limitations, all carried into the report rather than resolved:

| limitation | measured |
|---|---|
| 256×256 segmentation bottleneck | a square geometry still carries a **−10.6 %** width bias; the round trip, not the aspect ratio, dominates |
| geometry-dependent shape bias | V1 width |bias| median 0.1228 across non-square geometries; tortuosity 0.0207 |
| poor Neo external vessel segmentation | Dice **0.4889** / clDice **0.5461** vs RetCam **0.7540** / **0.8751** |
| target-domain expert validation pending | no project-domain expert vessel or disc annotation exists |

## C. Complete external agreement — all 10 original Primary Core features

100 HVDROPDB expert-vessel images, `GOLD` (expert mask) vs `SEG_CURRENT_V1` mask, identical
feature implementation. `results/hvdro_validation/v2/task5d_primary_core_agreement_summary.csv`.
Every row carries its evaluable N.

### RetCam, N = 50

| feature | expert mean | expert SD | expert median | expert IQR | auto mean | ICC(2,1) | 95 % CI | bias | LoA | MAE | medAE | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tort_geodesic_median | — | 0.01066 | — | 0.00364 | — | 0.258 | [−0.022, 0.483] | +0.0012 | [−0.0199, 0.0223] | 0.0063 | 0.0040 | 0.289 |
| tort_geodesic_p90 | — | 0.04287 | — | 0.04965 | — | 0.309 | [0.037, 0.565] | −0.0060 | [−0.0821, 0.0701] | 0.0271 | 0.0184 | 0.284 |
| vessel_density_fov | — | — | — | — | — | **0.872** | [0.789, 0.995] | +0.0158 | [−0.0176, 0.0491] | 0.0192 | 0.0195 | 0.942 |
| skel_density_fov | — | — | — | — | — | **0.933** | [0.886, 0.998] | −0.0016 | [−0.0055, 0.0024] | 0.0016 | 0.0013 | 0.951 |
| tort_geodesic_top3_mean | — | 0.35328 | — | 0.16433 | — | 0.207 | [−0.049, 0.402] | −0.1157 | [−0.742, 0.510] | 0.1617 | 0.0409 | 0.663 |
| n_branches | — | — | — | — | — | 0.738 | [0.584, 0.958] | **+10.04** | [−49.67, 69.75] | 22.84 | 16.50 | 0.834 |
| width_shape_p90_over_p50 | — | — | — | — | — | **0.054** | [−0.047, 0.118] | **+0.3341** | [−0.153, 0.821] | 0.3470 | 0.3101 | 0.157 |
| fractal_d0 | — | — | — | — | — | **0.940** | [0.897, 0.999] | −0.0183 | — | 0.0202 | 0.0141 | 0.932 |
| fractal_d1 | — | — | — | — | — | **0.923** | [0.870, 0.998] | −0.0246 | — | 0.0260 | 0.0193 | 0.927 |
| fractal_d2 | — | — | — | — | — | **0.911** | [0.851, 0.997] | −0.0273 | — | 0.0291 | 0.0214 | 0.924 |

### Neo, N = 50

| feature | ICC(2,1) | 95 % CI | bias | MAE | medAE | rho |
|---|---|---|---|---|---|---|
| tort_geodesic_median | 0.218 | [0.023, 0.446] | +0.0051 | 0.0059 | 0.0051 | 0.143 |
| tort_geodesic_p90 | 0.248 | [−0.008, 0.473] | +0.0131 | 0.0263 | 0.0183 | 0.345 |
| vessel_density_fov | 0.774 | [0.633, 0.968] | +0.0001 | 0.0129 | 0.0106 | 0.751 |
| skel_density_fov | 0.547 | [0.391, 0.944] | −0.0050 | 0.0051 | 0.0053 | 0.816 |
| tort_geodesic_top3_mean | 0.248 | [−0.024, 0.469] | −0.0201 | 0.0680 | 0.0461 | 0.362 |
| n_branches | 0.674 | [0.501, 0.937] | **−15.46** | 25.22 | 17.50 | 0.713 |
| width_shape_p90_over_p50 | 0.136 | [0.011, 0.311] | **+0.3340** | 0.3489 | 0.2962 | 0.203 |
| fractal_d0 | 0.569 | [0.410, 0.942] | −0.0779 | 0.0786 | 0.0759 | 0.820 |
| fractal_d1 | 0.578 | [0.418, 0.943] | −0.0817 | 0.0821 | 0.0714 | 0.830 |
| fractal_d2 | 0.566 | [0.404, 0.937] | −0.0817 | 0.0825 | 0.0709 | 0.815 |

No feature was omitted for performing poorly. The expert mean/median/IQR columns are in the CSV
and were computed for every row; they are elided in this table only for width.

## D. Variance-aware tortuosity interpretation

| camera | feature | expert SD | expert IQR | MAE/SD | MAE/IQR | between-image var | error var | error/between | ICC |
|---|---|---|---|---|---|---|---|---|---|
| RetCam | tort_geodesic_median | 0.01066 | 0.00364 | 0.594 | 1.741 | 1.14e−4 | 1.16e−4 | **1.020** | 0.258 |
| RetCam | tort_geodesic_p90 | 0.04287 | 0.04965 | 0.633 | 0.546 | 1.84e−3 | 1.51e−3 | **0.820** | 0.309 |
| RetCam | tort_geodesic_top3_mean | 0.35328 | 0.16433 | 0.458 | 0.984 | 1.25e−1 | 1.02e−1 | **0.817** | 0.207 |
| Neo | tort_geodesic_median | 0.00481 | 0.00511 | 1.232 | 1.160 | 2.3e−5 | 4.0e−5 | **1.743** | 0.218 |
| Neo | tort_geodesic_p90 | 0.03303 | 0.01626 | 0.796 | 1.617 | 1.09e−3 | 1.30e−3 | **1.190** | 0.248 |
| Neo | tort_geodesic_top3_mean | 0.09507 | 0.13031 | 0.716 | 0.522 | 9.04e−3 | 8.67e−3 | **0.959** | 0.248 |

**Written evidence, no overclaiming.**

The low ICC is **partly** a variance effect and **partly** a real signal-to-noise problem, and the
split differs by camera and feature. Where `error_var / between_var < 1` — RetCam
`tort_geodesic_p90` (0.820), RetCam `tort_geodesic_top3_mean` (0.817), Neo
`tort_geodesic_top3_mean` (0.959) — the measurement error is smaller than the spread the feature
is supposed to resolve, so the low ICC is predominantly the between-image-variance term and the
feature is `VARIANCE_LIMITED`. Where the ratio is at or above 1 — RetCam `tort_geodesic_median`
(1.020), Neo `tort_geodesic_median` (1.743), Neo `tort_geodesic_p90` (1.190) — the error exceeds
the signal spread, and low ICC there is a genuine noise limitation, not an artefact of a narrow
distribution. `MAE / expert SD` runs from 0.458 to 1.232, which is the same statement in a
different unit.

**Not claimed:** that tortuosity is invalid because ICC is low — bias is +0.0012 to +0.0131 and
MAE is the smallest in the panel for the median variants, so the measurement is not systematically
wrong. **Also not claimed:** that tortuosity is validated because MAE is small — the error is
0.46–1.23 of the between-image SD, so it cannot carry a primary claim.
Classification: `VARIANCE_LIMITED` (RetCam p90 and top3, Neo top3); `WEAK` (RetCam median, Neo
median, Neo p90).

## E/F. Paired RetCam DD analysis

25 images verified by **pixel identity** across the vessel and disc reference cohorts.
No filename, row-number, camera or ordinal pairing was used anywhere. Neo has **0** pairs and is
`DEFERRED_TO_TARGET_DOMAIN_EXPERT_VALIDATION`; no Neo value is fabricated or pooled.

Conditions: `GOLD` = expert vessel + expert disc; `SEG-ONLY` = SEG_CURRENT_V1 vessel + expert
disc; `END-TO-END` = SEG_CURRENT_V1 vessel + current auto disc.

| comparison | feature | N | ICC(2,1) | 95 % CI | bias | LoA | MAE | medAE |
|---|---|---|---|---|---|---|---|---|
| SEG-ONLY vs GOLD | width_p50_dd | 25 | 0.341 | [0.156, 0.820] | +0.0328 | — | 0.0328 | 0.0384 |
| SEG-ONLY vs GOLD | width_p90_dd | 25 | 0.125 | [0.036, 0.425] | +0.1116 | — | 0.1116 | 0.1191 |
| SEG-ONLY vs GOLD | width_mean_dd | 25 | 0.278 | [0.131, 0.840] | +0.0455 | — | 0.0455 | 0.0512 |
| SEG-ONLY vs GOLD | width_ann_p50_dd | 25 | 0.321 | [0.130, 0.755] | +0.0461 | — | 0.0461 | 0.0476 |
| SEG-ONLY vs GOLD | width_ann_p90_dd | 25 | 0.047 | [−0.018, 0.117] | +0.1705 | — | 0.1705 | 0.1787 |
| SEG-ONLY vs GOLD | width_ann_mean_dd | 25 | 0.174 | [0.052, 0.517] | +0.0699 | — | 0.0699 | 0.0739 |
| SEG-ONLY vs GOLD | ring_2_3dd | 21 | **0.821** | [0.640, 0.994] | +0.0312 | — | 0.0317 | 0.0282 |
| SEG-ONLY vs GOLD | ring_3_6dd | 21 | **0.773** | [0.569, 0.992] | +0.0286 | — | 0.0286 | 0.0264 |
| END-TO-END vs GOLD | width_p50_dd | 24 | 0.316 | [0.134, 0.782] | +0.0347 | — | 0.0347 | 0.0405 |
| END-TO-END vs GOLD | width_p90_dd | 24 | 0.110 | [0.025, 0.361] | +0.1179 | — | 0.1179 | 0.1262 |
| END-TO-END vs GOLD | ring_2_3dd | 20 | 0.815 | [0.624, 0.993] | +0.0323 | — | 0.0330 | 0.0308 |
| END-TO-END vs GOLD | ring_3_6dd | 20 | 0.750 | [0.528, 0.988] | +0.0303 | — | 0.0303 | 0.0272 |
| **AUTO-DISC vs EXPERT-DISC** | width_p50_dd | 24 | **0.977** | [0.947, 1.000] | +0.0023 | — | 0.0041 | 0.0044 |
| **AUTO-DISC vs EXPERT-DISC** | width_p90_dd | 24 | **0.985** | [0.966, 1.000] | +0.0042 | — | 0.0071 | 0.0074 |
| **AUTO-DISC vs EXPERT-DISC** | width_ann_p90_dd | 24 | **0.980** | [0.955, 1.000] | +0.0069 | — | 0.0108 | 0.0103 |
| **AUTO-DISC vs EXPERT-DISC** | ring_2_3dd | 20 | **0.999** | [0.996, 1.000] | +0.0007 | — | 0.0028 | 0.0016 |
| **AUTO-DISC vs EXPERT-DISC** | ring_3_6dd | 20 | **0.997** | [0.992, 1.000] | +0.0015 | — | 0.0029 | 0.0021 |

`disc_valid` held on 25/25 for GOLD and SEG-ONLY, 24/25 for END-TO-END. Full LoA and per-image
values in the CSVs.

**Reading.** The ring densities transfer well (ICC 0.77–0.82). Every caliber feature carries a
**positive systematic bias of +0.03 to +0.17 DD**: the automatic vessel mask over-measures
caliber relative to expert. `width_ann_p90_dd` is the worst (ICC 0.047, bias +0.17).

**F. Incremental error from automatic disc estimation: median MAE 0.00454 DD, median bias
0.00284 DD**, with per-feature ICC 0.977–0.999. The disc contributes essentially nothing to the
DD-measurement error; **vessel segmentation is the dominant error source**. This is not
target-domain validation and does not license any clinical claim.

## G. Feature evidence classification

| feature | classification | evidence |
|---|---|---|
| `vessel_density_fov` | **STRONG** | ICC 0.872/0.774, bias +0.016/+0.0001, MAE 0.019/0.013, rho 0.94/0.75; FOV padding invariance exact; no systematic bias in either camera |
| `skel_density_fov` | **STRONG** | ICC 0.933/0.547, bias −0.002/−0.005, MAE 0.0016/0.005, rho 0.95/0.82; smallest absolute error in the panel; Neo ICC moderate is the only weakness |
| `fractal_d0` | **STRONG on RetCam / MODERATE on Neo** | ICC 0.940/0.569, bias −0.018/−0.078, rho 0.93/0.82; RetCam excellent, Neo systematic −0.08 bias |
| `fractal_d1` | **STRONG on RetCam / MODERATE on Neo** | ICC 0.923/0.578, bias −0.025/−0.082 |
| `fractal_d2` | **STRONG on RetCam / MODERATE on Neo** | ICC 0.911/0.566, bias −0.027/−0.082 |
| `tort_geodesic_median` | **WEAK** | bias +0.0012/+0.0051 and MAE 0.0063/0.0059 are excellent, but ICC 0.258/0.218 and error/between var 1.020/1.743 |
| `tort_geodesic_p90` | **VARIANCE_LIMITED** | ICC 0.309/0.248, bias −0.0060/+0.0131, error/between var 0.820/1.190 |
| `tort_geodesic_top3_mean` | **VARIANCE_LIMITED** | ICC 0.207/0.248, bias −0.1157/−0.0201, error/between var 0.817/0.959 |
| `n_branches` | **WEAK** | ICC 0.738/0.674 but **camera-signed bias +10.04 / −15.46** and MAE 22.8/25.2; boundary/scale topology, not vascular topology |
| `width_shape_p90_over_p50` | **WEAK** | **+0.3341 bias in BOTH cameras** (identical to 3 decimals), ICC 0.054/0.136, rho 0.157/0.203 |

No classification rests on a single statistic: bias, absolute error, ICC, expert variance, camera
consistency and known geometry sensitivity were considered jointly. These are evidence classes,
not clinical pass/fail thresholds.

## H/I. Model-admission revision

| previous (Task 5B) | final (Task 5D) | feature | why |
|---|---|---|---|
| secondary | **FINAL_PRIMARY** | `vessel_density_fov`, `skel_density_fov`, `fractal_d0/d1/d2` | promotion within already-admitted secondary features, justified by external evidence; no exploratory or forbidden feature was promoted |
| secondary | **FINAL_SECONDARY** | `tort_geodesic_median`, `tort_geodesic_p90`, `tort_geodesic_top3_mean` | variance-limited / weak signal-to-noise |
| secondary | **EXPLORATORY_ONLY** | `n_branches` | demoted: camera-signed bias +10.0 / −15.5 with MAE 23 / 25 |
| secondary | **FORBIDDEN** | `width_shape_p90_over_p50` | demoted: large systematic bias +0.334 in both cameras, ICC 0.05 / 0.14 |
| secondary | **DISC_CONDITIONAL_ONLY** | the 8 disc-dependent features | project-domain disc validity is only 42.5 % and source-dependent |
| exploratory | exploratory | all A/V features | unchanged |
| forbidden | forbidden | all QC / metadata / detector flags | unchanged |

No model performance, AUC or feature importance influenced any status. No exploratory or
forbidden feature was promoted on the strength of HVDROPDB.

`n_branches` and `width_shape_p90_over_p50` were **not** left in the primary set. Both are demoted
with their numbers recorded above.

**FOV densities, stated without overclaiming.** `vessel_density_fov` and `skel_density_fov` have
the strongest external evidence in the panel — ICC up to 0.933, near-zero bias in both cameras,
the smallest absolute errors, and a demonstrated **exact** padding invariance (relative change
0.00000000 under +50/+100/+200/+300 px and rectangular padding). That is technical measurement
evidence. It is not clinical validation, and both features still inherit the 256×256 bottleneck
and the geometry-dependent bias of the mask they are computed from.

## J. Final model feature sets

| set | n | file |
|---|---|---|
| `FINAL_PRIMARY_CORE_V1` | **5** | `configs/final_primary_core_v1_features.yaml` |
| `FINAL_SECONDARY_CORE_V1` | **3** | `configs/final_secondary_core_v1_features.yaml` |
| `DISC_CONDITIONAL_V1` | **8** | `configs/disc_conditional_v1_features.yaml` |

`FINAL_PRIMARY_CORE_V1` contains no disc-dependent feature, no QC variable, no metadata, no
exploratory feature and no forbidden feature. `FINAL_SECONDARY_CORE_V1` is a predeclared
sensitivity set and must not replace the primary result after inspecting test performance.
`DISC_CONDITIONAL_V1` is used only in a separately labelled conditional analysis, and its
population difference must be reported before any comparison with full-cohort primary
performance.

The exact `PRIMARY_CORE_V1` names were **verified from the YAML** before analysis; the YAML and
the Task 5D list matched exactly.

## K. The 8 missing rows — resolved

Cause: `fractal_d0/d1/d2` are the only NaN columns, on exactly 8 rows. `PVBM.MultifractalVBMs`
returns NaN for those masks; the masks themselves are non-degenerate (vessel pixels 5286–13330,
skeleton 974–2214), 6 of the 8 also failed FOV detection and 6 have an invalid disc. The module
is behaving as designed: the values are genuinely undefined for those inputs.

**Predeclared policy: COMPLETE-CASE evaluation.** Any analysis that includes a fractal feature has
`N = 8862` and must report 8862. No imputation. The policy was not selected by disease-model
performance.

`FINAL_PRIMARY_ROW_COMPLETENESS = 0.9991 (8862 / 8870)`.

## L/M. Frozen table and pre-training integrity

`data/features/final_biomarkers_v1.csv` — 8870 rows × 80 columns, derived only from the canonical
8870, `SEG_CURRENT_V1` and the frozen `CLINICAL_MEASUREMENT_V1` definitions. All measurements are
retained for provenance; the model feature lists are the external YAML contracts.

```
FINAL_FEATURE_TABLE_SHA256 = f1c41e923ae29d4e228097536765f5e7399963062253ad925657a077cddc10c0
```

| integrity check | result |
|---|---|
| 8870 canonical rows | PASS |
| 8870 unique stable ids | PASS |
| split membership matches canonical | PASS |
| split counts match | PASS |
| source counts match | PASS |
| group linkage complete | PASS |
| no duplicate identity | PASS |
| no inf predictors | PASS |
| all FINAL_PRIMARY features present | PASS |
| no metadata column in the primary list | PASS |
| no QC column in the primary list | PASS |
| no disc-dependent feature in the primary list | PASS |
| measurement version present on every row | PASS |

## N. No model training

No Branch A, B or C model was trained; no disease AUC, no disease-label feature importance and no
LOSO was computed; no feature subset was optimised against labels. This task ends immediately
before disease-model training.

## O. Success gate

| # | requirement | status |
|---|---|---|
| 1 | `SEG_CURRENT_V1` formally frozen as the final generation | ✓ contract still verifies |
| 2 | all 10 Primary Core features have complete RetCam/Neo evidence | ✓ section C |
| 3 | tortuosity low ICC interpreted with explicit variance analysis | ✓ section D |
| 4 | the 25 true paired RetCam images analysed for DD agreement | ✓ section E |
| 5 | fake filename-based pairing never used | ✓ pixel identity only |
| 6 | auto-disc incremental error quantified | ✓ section F, 0.00454 DD |
| 7 | each candidate feature receives a final evidence classification | ✓ section G |
| 8 | externally weak / biased features reconsidered before admission | ✓ section H, two demotions |
| 9 | exact FINAL_PRIMARY list frozen before any performance is seen | ✓ 5 features |
| 10 | exact FINAL_SECONDARY list frozen before any performance is seen | ✓ 3 features |
| 11 | disc-dependent features remain separately conditional | ✓ `DISC_CONDITIONAL_V1` |
| 12 | the 8 missing rows explicitly resolved or governed | ✓ complete-case, N=8862 |
| 13 | final table and feature lists byte-hashed | ✓ `f1c41e92…` |
| 14 | canonical 8870 integrity preserved | ✓ section M |
| 15 | no disease classifier trained | ✓ |

```
PRETRAINING_FEATURE_FREEZE = PASS
TASK5D_STATUS = COMPLETE
```

## P. What the next phase inherits

* **Primary matrix**: `FINAL_PRIMARY_CORE_V1`, 5 features, 8862 complete rows.
* **Secondary sensitivity matrix**: `FINAL_SECONDARY_CORE_V1`, 3 features.
* **Conditional analysis**: `DISC_CONDITIONAL_V1`, 8 features, 3772 images, must report the
  population difference.
* **Never in a model matrix**: all QC/metadata/detector flags, all A/V features,
  `n_branches` and `width_shape_p90_over_p50`.
* **Imputation**: none. Complete-case with the N reported.
* **Open robustness target**: `ACQUISITION_GEOMETRY_LABEL_ASSOCIATION` (geometry → label AUC
  0.6445).
* **Still required**: project-domain expert vessel masks, expert disc annotations and
  inter-expert reference. `TARGET_DOMAIN_EXPERT_VALIDATION_STILL_REQUIRED = YES`.
