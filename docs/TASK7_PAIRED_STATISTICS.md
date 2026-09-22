# TASK 7 — Paired statistical analysis of the frozen Task-6 test predictions

Analysis only. No model was trained, tuned, refitted, recalibrated or altered; no probability,
threshold or prediction was changed. Inputs are the frozen Task-6 per-image test prediction tables
and their recorded SHA256 values.

Test status: `CANONICAL_LOCKED_SPLIT_REANALYSIS`. The canonical test split has historical exposure,
so it is **not** an untouched confirmatory test. `HVDROPDB` remains an external development /
measurement benchmark, not confirmatory target-domain validation.

## 1. Frozen input verification — PASS (11 of 11)

`test_predictions_B_EMBEDDING_ONLY.csv` and `test_predictions_C_PRIMARY.csv`:

| check | result |
|---|---|
| N = 1331 in each file | PASS |
| identical image_id set | PASS |
| merge by image_id gives 1331 rows, no duplicates | PASS |
| identical `true_label` for every paired image | PASS |
| probabilities finite | PASS |
| probabilities sum to 1 (atol 1e-6) | PASS |
| both file SHA256 match `artifact_sha256.json` | PASS |

Test class support: Normal 982, Pre_Plus 140, Plus 209. Nothing was repaired or substituted.

## 2–3. Primary paired comparison (seed 42, 10,000 stratified paired bootstrap replicates)

Resampling is paired: both models are always evaluated on the same resampled indices, stratified by
true class so every replicate preserves class support. Delta is defined as `C_PRIMARY − B_EMBEDDING_ONLY`.

| Metric | B_EMBEDDING_ONLY | C_PRIMARY | Delta C−B | 95 % paired CI | p (null-centred, two-sided) |
|---|---|---|---|---|---|
| multiclass AUC (frozen thesis definition) | 0.924903 | 0.925988 | **+0.001085** | **[−0.001691, +0.003868]** | 0.431 |
| macro OVR AUC | 0.924903 | 0.925988 | +0.001085 | [−0.001691, +0.003868] | 0.431 |
| balanced accuracy | 0.723425 | 0.720417 | −0.003008 | [−0.019805, +0.012895] | 0.717 |
| macro F1 | 0.708656 | 0.709347 | +0.000691 | [−0.015127, +0.015549] | 0.926 |
| multiclass Brier (lower better) | 0.276283 | 0.269028 | **−0.007255** | **[−0.015464, +0.000865]** | 0.082 |
| top-label ECE, 15 equal-width bins (lower better) | 0.102817 | 0.098564 | −0.004253 | [−0.014093, +0.005995] | 0.403 |

Bootstrap mean and median deltas are recorded in `paired_primary_metrics.csv`; the full 10,000 × 9
distribution is in `bootstrap_distributions.parquet`.

**p-value method.** Two-sided null-centred bootstrap: the observed delta is subtracted from each
replicate delta and p = P(|centred delta| ≥ |observed delta|). This is a standard, documented
bootstrap test and is reported for every metric; the CI remains the primary inference.

## 5. Per-class AUC (paired bootstrap; DeLong not reported)

No validated paired DeLong implementation exists in the environment or repository, and an unvalidated
one was deliberately not hand-written, so per-class inference uses the paired bootstrap only.

| class | B | C | Delta C−B | 95 % paired CI | p |
|---|---|---|---|---|---|
| Normal | 0.916762 | 0.918735 | +0.001972 | [−0.001421, +0.005451] | 0.256 |
| Pre_Plus | 0.931282 | 0.935366 | +0.004084 | [+0.000150, +0.008084] | 0.044 |
| Plus | 0.926665 | 0.923863 | −0.002802 | [−0.006333, +0.000584] | 0.118 |

`Pre_Plus` is the only per-class CI excluding zero. It is a **supportive** endpoint: with three
per-class comparisons and five other metrics, no endpoint was selected on the basis of significance,
and this single marginal result is not treated as a claim.

## 6. Classification disagreement

| quantity | value |
|---|---|
| images where argmax class changed | 46 of 1331 (3.46 %) |
| B wrong → C correct | 25 |
| B correct → C wrong | 20 |
| both wrong, different class | 1 |
| net corrected classifications | **+5** |
| McNemar exact p (B vs C correctness) | **0.5515** |

Transition matrix (rows B predicted, columns C predicted; labels 0 Normal, 1 Pre_Plus, 2 Plus):

| B \ C | 0 | 1 | 2 |
|---|---|---|---|
| **0** | 945 | 10 | 3 |
| **1** | 11 | 161 | 4 |
| **2** | 14 | 4 | 179 |

## 7. Probability-level change

| quantity | median | IQR | p95 | max |
|---|---|---|---|---|
| absolute change in true-class probability | 0.000777 | 0.031817 | 0.166220 | 0.499407 |
| L1 distance between B and C probability vectors | 0.001980 | 0.072781 | 0.349067 | 0.999157 |
| maximum absolute class-probability change | 0.000990 | 0.036391 | 0.174533 | 0.499578 |

The medians are near zero while the tails are substantial, so the biomarkers **do** move probabilities
for a minority of images, but not in a way that improves discrimination. This distinguishes
"no probability movement" from "movement without measurable discrimination gain".

## 8. Source-specific paired analysis (secondary)

| source | n | classes | metric | B | C | Delta | 95 % CI |
|---|---|---|---|---|---|---|---|
| farfum_rop | 230 | 0,1,2 | multiclass AUC | 0.865237 | 0.867276 | +0.002040 | [−0.007172, +0.011988] |
| farfum_rop | 230 | 0,1,2 | balanced accuracy | 0.720389 | 0.746743 | +0.026353 | [+0.002849, +0.051656] |
| farfum_rop | 230 | 0,1,2 | macro F1 | 0.720659 | 0.747561 | +0.026902 | [+0.003989, +0.052336] |
| farabi | 212 | 0,1,2 | multiclass AUC | 0.830400 | 0.834086 | +0.003686 | [−0.009398, +0.016166] |
| farabi | 212 | 0,1,2 | balanced accuracy | 0.668936 | 0.649324 | −0.019612 | [−0.056536, +0.018471] |
| farabi | 212 | 0,1,2 | macro F1 | 0.670590 | 0.651716 | −0.018875 | [−0.054134, +0.018442] |
| plus | 889 | 0,2 | balanced accuracy | 0.670433 | 0.642491 | −0.027942 | [−0.057751, −0.003227] |
| plus | 889 | 0,2 | macro F1 | 0.665669 | 0.647496 | −0.018173 | [−0.045136, +0.004875] |
| plus | 889 | 0,2 | AUC Normal | 0.877684 | 0.870870 | −0.006814 | [−0.013581, −0.000311] |
| plus | 889 | 0,2 | AUC Plus | 0.891170 | 0.886232 | −0.004938 | [−0.010643, +0.000516] |

Multiclass AUC is **not reported for `plus`** because Pre_Plus is absent there (2 classes); only
mathematically defined metrics are shown. Source Ns are small, 2,000 replicates each, and uncertainty
is correspondingly large; no model was tuned or selected by source. The farfum_rop and plus rows move
in opposite directions, which is consistent with noise at these sample sizes.

## 9. Calibration comparison

Brier delta −0.00726 [−0.01546, +0.00087], ECE delta −0.00425 [−0.01409, +0.00600]: both point toward
better-calibrated probabilities for C_PRIMARY, neither CI excludes zero. Neither model was
recalibrated and no temperature scaling was applied. Figures: `calibration_comparison.png`
(reliability curves, same 15-bin definition as Task 6) and `delta_forest_plot.png`.

## 10–11. Interpretation (no post-hoc endpoint selection)

The frozen thesis convention makes **multiclass AUC the main discrimination endpoint**. Its delta is
**+0.0011 with a 95 % CI of [−0.0017, +0.0039]** — a narrow interval tightly centred on zero. This is
interpretation case **A**: *the evidence supports only a very small incremental effect under this
evaluation*, and the interval excludes gains of practical interest.

Two secondary patterns are recorded without overclaiming:

* **case D, partially:** discrimination is unchanged but calibration moves consistently in one
  direction (Brier p = 0.082, ECE p = 0.403). A weak, non-significant tendency toward better-calibrated
  probabilities is a *different* potential form of incremental value, not an improvement in
  discrimination;
* **classification is essentially unaffected:** only 3.5 % of argmax predictions change, corrections
  (25) and corruptions (20) nearly cancel, net +5, McNemar p = 0.55.

No clinical significance is claimed. The correct summary sentence is: *adding the five frozen
interpretable vascular biomarkers to the frozen CNN representation did not measurably improve
discrimination on this locked-split re-analysis; the largest movement is a small, non-significant
improvement in probability calibration.*

## Outputs

`artifacts/task7_paired_statistics/`: `paired_primary_metrics.csv`, `paired_per_class_auc.csv`,
`paired_source_analysis.csv`, `prediction_disagreement.csv`, `probability_change_summary.csv`,
`bootstrap_distributions.parquet`, `paired_statistics_summary.json`, `calibration_comparison.png`,
`delta_forest_plot.png`, `artifact_sha256.json` (all SHA256 frozen).
