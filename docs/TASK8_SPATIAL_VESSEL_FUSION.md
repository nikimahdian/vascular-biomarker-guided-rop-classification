# Task 8 — Two-Server Spatial Vessel Fusion (D / E / F / G)

Status: `TASK8_STATUS = COMPLETE`
Classification: `SECONDARY_POST_PRIMARY_EXPLORATORY_CANONICAL_SPLIT_REANALYSIS`
TEST policy: opened exactly once, after `TASK8_ALL_SELECTION_FROZEN = YES`.

This is **not** a confirmatory result. The canonical split was already used by the primary
Task 6 / Task 7 analysis. Task 8 is a secondary post-primary exploratory reanalysis on the same
locked canonical split. No historical A/C or LOSO number is treated as a headline result here;
those remain provenance evidence only.

## 1. Frozen contract (nothing selected on TEST)

| Element | Frozen value |
|---|---|
| Backbone | `efficientnet_b4`, timm, ImageNet pretrained, `num_classes=0`, `global_pool=avg` |
| Embedding dim | 1792 |
| Head | `Dropout(0.3)` + `Linear(1792, 3)` |
| Optimizer | AdamW, lr 1e-4, weight_decay 1e-4 |
| Batch / epochs / patience | 16 / max 30 / 6 |
| Seed | 42 |
| Loss | weighted CE, Task-6 **train-only** class weights 0.46051 / 3.18103 / 1.94512 |
| Mask input | `SEG_CURRENT_V2_RESOLVER_SAFE` binary PNG, NEAREST resize 384x384, replicated to 3 channels |
| Augmentation | geometric only (RandomResizedCrop scale 0.95–1.0 NEAREST, HFlip, Rotation 10 NEAREST) |
| Mask resolver | content-keyed registry only; no stem / prefix / glob fallback |
| E / G learner | frozen B_EMBEDDING_ONLY XGBoost config (`downstream_xgb_selected.json`): 500 trees, depth 3, lr 0.1, subsample 0.8, colsample 0.8, min_child_weight 1, lambda 1.0 |
| F alpha | selected on **validation only** |

Population 8,862 (complete-case). Train 6,203 / Val 1,328 / Test 1,331.
Mask preflight on Server 2: `masks missing = 0` (BASE2 prefix mapping).

## 2. D_VESSEL_MAP training (Server 2, RTX 4090)

~52–57 s/epoch. Early stop at epoch 9. Selection: val multiclass AUC > macro F1 > −val loss > earlier epoch.

| epoch | seconds | train loss | val loss | val AUC | val macro F1 |
|---|---|---|---|---|---|
| 1 | 56.5 | 0.8228 | 0.7692 | 0.77629 | 0.4864 |
| 2 | 54.6 | 0.6185 | 0.7691 | 0.78250 | 0.5085 |
| **3** | 52.4 | 0.5355 | 0.7360 | **0.80334** | **0.5352** |
| 4 | 50.9 | 0.4684 | 0.8113 | 0.78386 | 0.4970 |
| 5 | 53.9 | 0.4016 | 0.8644 | 0.78104 | 0.4897 |
| 6 | 52.3 | 0.3343 | 0.9776 | 0.76438 | 0.4834 |
| 7 | 51.9 | 0.2967 | 1.0485 | 0.76851 | 0.5023 |
| 8 | 52.4 | 0.2170 | 1.2680 | 0.78544 | 0.5041 |
| 9 | 53.3 | 0.1675 | 1.4383 | 0.77522 | 0.4892 |

Selected: epoch 3. Checkpoint `d_selected.pth` sha256
`3e1719ebfa93d106be89c27a08f7974e0ec0f6cd4d6cc0b8f2bd140643ee5021`.
Overfitting is unambiguous from epoch 4 onward (train loss 0.47 → 0.17 while val loss 0.81 → 1.44).

Vessel embeddings: 8,862 x 1,792, NaN 0, sha256
`9c7aa91f3a45ae95404ecc123a64376fb8155804d38320e9e043d4f960cc9d91`.

## 3. Fusion variants

- **E** = RGB EfficientNet-B5 embedding (2,048) + vessel embedding (1,792) = 3,840-d → XGBoost
- **F** = late fusion `alpha * P_B_EMBEDDING_ONLY + (1-alpha) * P_D`, alpha on VAL
- **G** = E + the 5 FINAL_PRIMARY scalar biomarkers = 3,845-d → XGBoost

F alpha selection returned **alpha = 1.0** (val AUC 0.90352, val macro F1 0.63180). The grid was
0.0 … 1.0 in 0.1 steps. Because alpha saturated at the boundary, the vessel-late-fusion component
contributes nothing at the selected operating point and **F collapses to B_EMBEDDING_ONLY**. This is
a genuine negative result and is reported as such, not as a fusion gain.

## 4. TEST results (N = 1,331, single use)

| model | AUC | balanced acc | macro F1 | Brier | ECE |
|---|---|---|---|---|---|
| D_VESSEL_MAP | 0.837215 | 0.620032 | 0.588736 | 0.381230 | — |
| E_RGB_VESSEL_FEATURE_FUSION | 0.932841 | 0.735049 | 0.724571 | 0.248590 | — |
| F_RGB_VESSEL_LATE_FUSION | 0.924903 | 0.723425 | 0.708656 | 0.276283 | — |
| G_RGB_VESSEL_SCALAR_FUSION | 0.934880 | 0.738895 | 0.729163 | 0.246515 | — |

Reference frozen Task-6 TEST values on the same 1,331: A_PRIMARY 0.659582, B_RGB 0.913334,
B_EMBEDDING_ONLY 0.924903, C_PRIMARY 0.925988.

- Vessel-only D is far weaker than any RGB branch (0.8372). Vessel morphology alone is not a
  competitive ROP-Plus classifier in this cohort.
- E > B_EMBEDDING_ONLY by +0.00794 AUC (paired, see below) and G is the highest AUC observed in
  the entire Task 6–8 series (0.93488), but the margin over C_PRIMARY (0.92599) is small and both
  E and G are secondary exploratory comparisons, not primary claims.
- Per-class TEST AUC (Normal / Pre_Plus / Plus) is in `metrics_summary.csv`.

## 5. Paired statistics — E vs B_EMBEDDING_ONLY (2,000 class-stratified bootstrap replicates)

> **Provenance, not superseded.** These 2,000-replicate numbers are the numbers produced during the
> original Task-8 run and are preserved exactly as written. Task 8B (§10) re-ran the same comparison
> at 10,000 replicates under the identical protocol; §10 is the reporting reference, §5 stays as
> provenance. Neither file was overwritten.

| metric | delta E − B | 95% CI | p |
|---|---|---|---|
| multiclass AUC | **+0.007938** | [+0.002852, +0.013193] | **0.0010** |
| balanced accuracy | +0.011624 | [−0.011154, +0.034441] | 0.3155 |
| macro F1 | +0.015915 | [−0.005758, +0.036942] | 0.1415 |
| Brier | **−0.027693** | [−0.043995, −0.011463] | **0.0015** |
| ECE | −0.010675 | [−0.025979, +0.003870] | 0.1670 |

Reading: adding the spatial vessel embedding to the RGB embedding improves ranking (AUC) and
probability quality (Brier) with CIs excluding zero. It does **not** produce a detectable
improvement in thresholded decision metrics (balanced accuracy, macro F1) — the CIs straddle zero.
So the honest statement is narrower than "vessel fusion improves ROP-Plus classification": it
improves discrimination and calibration, not operating-point performance.

## 6. Source breakdown (TEST)

| source | n | D AUC | E AUC | F AUC | G AUC | D F1 | E F1 | F F1 | G F1 |
|---|---|---|---|---|---|---|---|---|---|
| plus | 889 | n/a | n/a | n/a | n/a | 0.4389 | 0.4882 | 0.6657 | 0.7286 |
| farfum_rop | 230 | 0.8204 | 0.8733 | 0.8652 | 0.8703 | 0.6231 | 0.7351 | 0.7207 | 0.7300 |
| farabi | 212 | 0.6915 | 0.8214 | 0.8304 | 0.8329 | 0.5258 | 0.6129 | 0.6706 | 0.6394 |

`plus` AUC is undefined (NaN) because only two of the three classes are present in that source
subgroup — the same two-present-class guard already documented in Task 7. Subgroup n are small;
these are descriptive only.

## 7. What was frozen before TEST was opened

`task8_selection_frozen.json` (`TASK8_ALL_SELECTION_FROZEN = YES`), written before any TEST read:

- D selected checkpoint sha256 `3e1719eb…`
- vessel embeddings sha256 `9c7aa91f…`
- E config = G config = frozen B_EMBEDDING_ONLY downstream XGBoost params
- F alpha = 1.0 (VAL only)

`d_selection.json` records `test_touched = false`. No epoch, alpha, or hyperparameter was revisited
after TEST was read.

## 8. Limitations stated plainly

1. Secondary exploratory reanalysis on an already-used canonical split — not confirmatory.
2. F's alpha saturated at the grid boundary, so late fusion is untested rather than positive.
3. Decision-metric gains (balanced accuracy, macro F1) are not statistically supported.
4. D is a mask-only model; its 0.8372 is evidence that spatial vessel morphology carries signal but
   is not a standalone solution.
5. The `plus` source subgroup cannot yield 3-class AUC at all.
6. Single seed (42) for D; no seed replication was performed.
7. No operational, screening, or clinical claim follows from any of these numbers.

## 9. Artifact integrity

`artifacts/task8_spatial_vessel_fusion/artifact_sha256.json` on Server 2 lists all 31 artifacts.
Key hashes: `d_selected.pth` `3e1719eb…`, `vessel_embeddings_b4_8862.parquet` `9c7aa91f…`,
`e_rgb_vessel_feature_fusion.json` `8f524832…`, `g_rgb_vessel_scalar_fusion.json` `3a118f7f…`,
`task8_summary.json` `a30e99a7…`.

---

## 10. Task 8B — statistical closure at 10,000 replicates

**No model was trained, refit, tuned or recalibrated. No prediction table was replaced or modified.**
Both the 2,000-replicate numbers in §5 and every Task-8 artifact are preserved unchanged; the Task-8B
output lives in the new subdirectory `artifacts/task8_spatial_vessel_fusion/statistical_closure_10k/`.

Protocol is copied from `scripts/task7_paired_statistics.py` verbatim: paired, class-stratified,
seed 42, per-replicate RNG `default_rng(SEED * 1000003 + r)`, 40-replicate chunks over a 24-worker
fork pool, percentile 2.5 / 97.5 CI, null-centered two-sided p = mean(|d − obs| >= |obs|), and the
same `macro_auc` / `ece15` / `brier` / `balanced_accuracy_score` / `f1_score` implementations.
N = 1,331; classes 982 / 140 / 209.

Input verification before any statistic was computed: all seven frozen prediction CSVs hashed and
matched against the Task-6 and Task-8 manifests — A `e560d8e0`, B `21f0e2f7`, C `63ef3a85`,
D `a534344d`, E `ac9279ac`, F `21f0e2f7`, G `4e067e25`. All UNCHANGED.

Note on F: its SHA equals B's exactly. With `alpha = 1.0` the late-fusion table is bit-identical to
`B_EMBEDDING_ONLY`, which independently confirms the §3 finding that F collapsed to B.

### 10.1 Primary comparison — E vs B_EMBEDDING_ONLY (10,000 replicates)

| metric | delta E − B | 95% CI | p | crosses 0 |
|---|---|---|---|---|
| multiclass AUC | **+0.007938** | [+0.002630, +0.013131] | **0.0031** | no |
| balanced accuracy | +0.011624 | [−0.011994, +0.034846] | 0.3278 | yes |
| macro F1 | +0.015915 | [−0.006211, +0.037655] | 0.1530 | yes |
| Brier | **−0.027693** | [−0.043954, −0.011471] | **0.0011** | no |
| ECE | −0.010675 | [−0.026072, +0.003869] | 0.1642 | yes |

The 10,000-replicate run reproduces the 2,000-replicate conclusion and tightens the intervals by
roughly 30 %. AUC and Brier remain the only metrics whose paired CI excludes zero. Thresholded
decision metrics and ECE remain unsupported.

### 10.2 G vs E — do the five scalar biomarkers add anything on top of the spatial vessel map?

| metric | delta G − E | 95% CI | p | crosses 0 |
|---|---|---|---|---|
| multiclass AUC | +0.002039 | [−0.000274, +0.004397] | 0.0900 | **yes (marginally)** |
| balanced accuracy | +0.003846 | [−0.012053, +0.020061] | 0.6383 | yes |
| macro F1 | +0.004592 | [−0.010536, +0.019581] | 0.5440 | yes |
| Brier | −0.002075 | [−0.009236, +0.004947] | 0.5707 | yes |
| ECE | −0.000063 | [−0.009202, +0.011057] | 0.9906 | yes |

Answer to the pre-declared question: **no measurable incremental information.** Every CI crosses
zero; the AUC interval misses by 0.000274 and its p of 0.090 does not survive any multiplicity
consideration. Once the spatial vessel representation is already in the feature vector, the five
FINAL_PRIMARY scalar biomarkers add nothing detectable.

### 10.3 Point estimates (no CI — descriptive only)

| comparison | AUC | bal acc | macro F1 | Brier | ECE |
|---|---|---|---|---|---|
| D_VESSEL_MAP vs A_PRIMARY | +0.177633 | +0.121536 | +0.154100 | −0.186248 | +0.029829 |
| E vs B_EMBEDDING_ONLY | +0.007938 | +0.011624 | +0.015915 | −0.027693 | −0.010675 |
| E vs C_PRIMARY | +0.006853 | +0.014632 | +0.015224 | −0.020438 | −0.006422 |
| G vs E | +0.002039 | +0.003846 | +0.004592 | −0.002075 | −0.000063 |
| G vs B_EMBEDDING_ONLY | +0.009977 | +0.015470 | +0.020507 | −0.029768 | −0.010738 |

Two of these point estimates are worth stating carefully. D beats A_PRIMARY by a wide margin on
every metric except ECE, which is the first direct evidence in this series that spatial vessel
morphology carries more ROP-Plus signal than the tabular biomarker panel alone. And E's point
advantage over C_PRIMARY (+0.006853 AUC) is of similar size to E's advantage over B_EMBEDDING_ONLY,
but unlike the E − B comparison it has no paired CI here, so it is descriptive only.

### 10.4 Status

Classification unchanged: `SECONDARY_POST_PRIMARY_EXPLORATORY_CANONICAL_SPLIT_REANALYSIS`.
No further experiment was started after this analysis.
