# Task 13 — ROPDeepX-Style Dual-RGB Soft Attention + Spatial Vessel Complementarity

Status: `TASK13_STATUS = COMPLETE`
Classification: `SECONDARY_POST_PRIMARY_EXPLORATORY_ARCHITECTURE_EXPERIMENT`
`TASK13_TEST_TOUCHED = NO` at freeze; TEST opened exactly once afterwards. No previous artifact was
modified. No scalar biomarker appears anywhere in this task.

**Headline — two answers that point in opposite directions.**
Question A (does dual-backbone RGB beat single-backbone RGB?): **no, it is significantly worse**
(M0 − B_EMBEDDING_ONLY AUC −0.018925, p = 0.0063).
Question B (does the frozen vessel representation stay complementary after RGB is strengthened?):
**yes, and the gain is larger than in any previous task** (M1 − M0 AUC +0.020936, p < 0.0001).
So the vessel finding is robust to swapping the RGB backbone, while the ROPDeepX-style dual-RGB
representation itself did not improve on the project's original EfficientNet-B5 embedding.

## 1. Pre-flight

| check | result |
|---|---|
| population 8,862 | PASS |
| split counts 6203 / 1328 / 1331 | PASS |
| no group overlap | PASS |
| labels {0,1,2} | PASS |
| vessel embedding SHA unchanged | PASS |
| B_EMBEDDING_ONLY / E / G prediction SHAs unchanged | PASS |
| RGB files exist | PASS |

Canonical split fingerprint: `0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8`.
That value is the SHA-256 of `data/splits/all.csv` (8,870 rows, 8,871 lines including the header) —
it is a file hash, not a derived hash. That file exists only on Server 1, so Server 1 recomputed it
during this task and it matched byte for byte; Server 2 re-verified the population, split counts,
label set and group disjointness against the 8,862-row complete-case manifest.

Class weights, recomputed from canonical TRAIN only as §7 requires:
**Normal 0.46050, Pre_Plus 3.18103, Plus 1.94512** — identical to the frozen Task-6 weights, which
is a useful independent confirmation that the Task-6 weighting was inverse-frequency on this same
train population. Label smoothing 0.05.

Batch size was **not** fixed by the contract. 16 was chosen before any training, matching the
Task-6/9 convention, and is recorded in `loss_config.json`.

## 2. Model L_ROPDEEPX_STYLE_RGB

Both branches receive the same 384×384 canonical RGB image with the established project
preprocessing and ImageNet normalization.

| element | specification |
|---|---|
| Branch 1 | ResNet50, ImageNet pretrained, global average pool → 2048 |
| Branch 2 | EfficientNet-B4, ImageNet pretrained, global average pool → 1792 |
| Projections | `Linear(2048,1024)→BatchNorm1d→ReLU` → `z_r`; `Linear(1792,1024)→BatchNorm1d→ReLU` → `z_e` |
| Fusion block | `Linear(2048,1024)→BatchNorm1d→ReLU→Dropout(0.30)` → `f_joint` |
| Attention | `Linear(1024,2)→Softmax(dim=1)` → `a_r + a_e = 1` |
| Attended representation | `f_att = a_r·z_r + a_e·z_e` ∈ R^1024 |
| Classifier | `Linear(1024,3)` |

Backbone parameters ≈ 87 M, head parameters ≈ 6.3 M.

Augmentation (TRAIN only): RandomResizedCrop 384 scale 0.95–1.0, horizontal flip, rotation ±10°,
ColorJitter(0.10, 0.10, 0.10, 0). VAL/TEST: deterministic resize to 384×384.

Staged training exactly as specified: stage 1 freezes both backbones for 3 epochs with head lr 3e-4,
weight decay 1e-4; stage 2 unfreezes ResNet50 `layer4` and the final EfficientNet-B4 block group plus
`conv_head`/`bn2`, with differential lr 5e-6 (backbone) / 5e-5 (head) under OneCycleLR, weight decay
5e-4, max 20 epochs, patience 5. AdamW throughout, seed 42, weighted CE with label smoothing 0.05.

## 3. L training history

| stage | epoch | seconds | train loss | val loss | val AUC | val macro F1 | mean a_resnet | mean a_effnet |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 32.4 | 0.7638 | 0.9127 | 0.83988 | 0.5417 | 0.4822 | 0.5178 |
| 1 | **2** | 29.0 | 0.6607 | 0.8718 | **0.86624** | 0.5752 | 0.5335 | 0.4665 |
| 1 | 3 | 28.2 | 0.6257 | 0.9026 | 0.85772 | 0.5966 | 0.5016 | 0.4984 |
| 2 | 4 | 36.9 | 0.5771 | 0.8742 | 0.85449 | 0.5978 | 0.5525 | 0.4475 |
| 2 | 5 | 35.0 | 0.5769 | 0.8711 | 0.85904 | 0.6136 | 0.5908 | 0.4092 |
| 2 | 6 | 34.1 | 0.5645 | 0.8762 | 0.85132 | 0.6041 | 0.6004 | 0.3996 |
| 2 | 7 | 34.6 | 0.5574 | 0.8683 | 0.84014 | 0.6040 | 0.6344 | 0.3656 |

≈30–37 s per epoch, 7 epochs run then early stop.

## 4. Model selection (VAL only, frozen before TEST)

```
L_SELECTED_EPOCH:        2   (stage 1)
L_SELECTED_VAL_AUC:      0.8662357
L_SELECTED_VAL_MACRO_F1: 0.5752488
L_CHECKPOINT_SHA256:     42dc5239739a4f00ffaaa22b2dec8916fcfdca4032a29d28edad1386c431551d
ROPDEEPX_EMBEDDING_SHA256: f73572c6dcf6df81ba2b40bb78a43bfcbcca72c349376cd85ba97c64934a15fd
M0_FROZEN:               YES   (499dabe5b691371c…)
M1_FROZEN:               YES   (c6e3e0832be60c38…)
TASK13_ALL_SELECTION_FROZEN: YES
TASK13_TEST_TOUCHED:     NO
```

Selection rule: highest val macro OVR AUC → highest macro F1 → lowest val loss → earlier epoch.
The selected checkpoint is stage-1 epoch 2.

## 5. Attention diagnostics

At the selected checkpoint (validation):

| split | n | resnet mean | resnet std | p05 | p50 | p95 | effnet mean | effnet std | p05 | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | 1328 | 0.5335 | 0.3991 | 0.0105 | 0.5997 | 0.9930 | 0.4665 | 0.3991 | 0.0070 | 0.4003 | 0.9895 |
| test | 1331 | 0.5111 | 0.3893 | 0.0087 | 0.5141 | 0.9925 | 0.4889 | 0.3893 | 0.0075 | 0.4859 | 0.9913 |

**Attention collapse: NO.** The means sit at 0.53/0.47 on validation and 0.51/0.49 on test, nowhere
near a 0.95 threshold. But `std ≈ 0.39` with `p05 ≈ 0.01` and `p95 ≈ 0.99` shows the attention is
effectively near-binary **per image**: for most images the model commits almost entirely to one
backbone, and the balanced mean is an average over images that split both ways. The mean alone hides
that; the quantiles are what matter here.

Attention by true class and by source (TEST, descriptive, post-freeze):

| grouping | value | n | mean a_resnet | mean a_effnet |
|---|---|---|---|---|
| true class | Normal | 982 | 0.6439 | 0.3561 |
| true class | Pre_Plus | 140 | 0.1072 | 0.8928 |
| true class | Plus | 209 | 0.1575 | 0.8425 |
| source | plus | 889 | 0.7079 | 0.2921 |
| source | farfum_rop | 230 | 0.1879 | 0.8121 |
| source | farabi | 212 | 0.0363 | 0.9637 |

Two readings, and the second is a caution:

1. The attention is **disease-relevant**: it favours ResNet50 for Normal and EfficientNet-B4 for both
   disease classes, which is a sensible division of labour rather than a degenerate one.
2. It is also **strongly source-dependent**: mean a_resnet is 0.71 on `plus`, 0.19 on farfum_rop and
   0.04 on farabi. Even though source identity was never a training target, the attention weight is
   close to a source fingerprint. This is consistent with Tasks 11 and 12, where source shift was
   large and the representation stayed almost perfectly source-identifiable, and it means the
   attention distribution should not be interpreted as a purely clinical quantity.

Attention drift over training is worth noting: mean a_resnet rose monotonically 0.4822 → 0.6344
across epochs 1–7 while validation AUC fell from 0.86624 (epoch 2) to 0.84014 (epoch 7). The
attention was still moving in a direction that made validation worse.

## 6. Attended embedding extraction

`f_att` extracted for all 8,862 samples with columns `sample_id, group_id, source, split, label` plus
1,024 features. Shape 8,862 × 1,024; NaN 0; Inf 0; all finite. SHA-256
`f73572c6dcf6df81ba2b40bb78a43bfcbcca72c349376cd85ba97c64934a15fd`.

## 7. M0 and M1

Both use the exact frozen Task-6/8 downstream XGBoost configuration, unchanged and untuned:
`n_estimators 500, max_depth 3, learning_rate 0.1, subsample 0.8, colsample_bytree 0.8,
min_child_weight 1, reg_lambda 1.0`, seed 42, Task-6 sample weights, fitted on the 6,203 canonical
training rows.

* **M0_ROPDEEPX_EMBEDDING_ONLY** — 1,024-d `f_att` only.
* **M1_ROPDEEPX_PLUS_VESSEL** — 1,024-d `f_att` + 1,792-d frozen Task-8 vessel embedding = **2,816**.

The only difference is the spatial vessel representation.

## 8. TEST results (N = 1,331, single use)

| model | AUC | balanced acc | macro F1 | Brier | ECE |
|---|---|---|---|---|---|
| `L_ROPDEEPX_STYLE_RGB` | 0.889458 | 0.677733 | 0.642466 | 0.323118 | 0.065 |
| `M0_ROPDEEPX_EMBEDDING_ONLY` | 0.905978 | 0.693861 | 0.671755 | 0.313569 | 0.116 |
| `M1_ROPDEEPX_PLUS_VESSEL` | 0.926913 | 0.724167 | 0.712159 | 0.264388 | 0.100 |
| `B_EMBEDDING_ONLY` (frozen) | 0.924903 | 0.723425 | 0.708656 | 0.276283 | 0.103 |
| `E_RGB_VESSEL_FEATURE_FUSION` (frozen) | **0.932841** | **0.735049** | **0.724571** | **0.248590** | **0.092** |
| `G_RGB_VESSEL_SCALAR_FUSION` (frozen) | **0.934880** | **0.738895** | **0.729163** | 0.246515 | 0.092 |

Two incidental observations. L's own linear classifier (AUC 0.889458) is clearly worse than XGBoost
on L's own 1,024-d representation (M0, 0.905978) — the neural classification head is the weaker part,
not the representation. And L has the best ECE of any model in the series (0.065) while having the
worst Brier (0.323), a combination produced by a much flatter probability distribution; calibration
in the ECE sense does not imply good probability quality here. Full-precision values are in
`metrics_summary.csv`.

## 9. Paired statistics — 10,000 class-stratified bootstrap replicates, seed 42

**Primary: M1 − M0**

| metric | delta | 95% CI | p |
|---|---|---|---|
| multiclass AUC | **+0.020936** | [+0.011426, +0.030828] | <0.0001 |
| balanced accuracy | +0.030306 | [−0.003035, +0.063993] | 0.0784 |
| macro F1 | **+0.040404** | [+0.010742, +0.070270] | 0.0080 |
| Brier | **−0.049181** | [−0.073529, −0.025358] | <0.0001 |
| ECE | −0.015802 | [−0.034543, +0.001425] | 0.0858 |

**Secondary: M0 − B_EMBEDDING_ONLY**

| metric | delta | 95% CI | p |
|---|---|---|---|
| multiclass AUC | **−0.018925** | [−0.032594, −0.005640] | 0.0063 |
| balanced accuracy | −0.029564 | [−0.068491, +0.008869] | 0.1369 |
| macro F1 | **−0.036901** | [−0.071458, −0.002904] | 0.0355 |
| Brier | **+0.037286** | [+0.005752, +0.069267] | 0.0218 |
| ECE | +0.013462 | [−0.007924, +0.036422] | 0.2304 |

**Secondary: M1 − E**

| metric | delta | 95% CI | p |
|---|---|---|---|
| multiclass AUC | −0.005928 | [−0.015565, +0.003477] | 0.2210 |
| balanced accuracy | −0.010882 | [−0.049273, +0.026804] | 0.5775 |
| macro F1 | −0.012412 | [−0.046174, +0.020216] | 0.4602 |
| Brier | +0.015798 | [−0.009155, +0.041431] | 0.2195 |
| ECE | +0.008335 | [−0.009551, +0.027293] | 0.3776 |

**Secondary: M1 − G**

| metric | delta | 95% CI | p |
|---|---|---|---|
| multiclass AUC | −0.007967 | [−0.017443, +0.001306] | 0.0949 |
| balanced accuracy | −0.014727 | [−0.052495, +0.022740] | 0.4428 |
| macro F1 | −0.017003 | [−0.049910, +0.015754] | 0.3123 |
| Brier | +0.017873 | [−0.007002, +0.043628] | 0.1601 |
| ECE | +0.008398 | [−0.010268, +0.026418] | 0.3581 |

## 10. Source breakdown (TEST)

| model | source | n | 3-class AUC | balanced acc | macro F1 |
|---|---|---|---|---|---|
| M0 | plus | 889 | n/a | 0.629372 | 0.631570 |
| M1 | plus | 889 | n/a | 0.741131 | 0.743706 |
| L | plus | 889 | n/a | 0.680935 | 0.664755 |
| B | plus | 889 | n/a | 0.670433 | 0.665669 |
| E | plus | 889 | n/a | 0.720136 | 0.732306 |
| M0 | farfum_rop | 230 | 0.851556 | 0.707395 | 0.701446 |
| M1 | farfum_rop | 230 | 0.868804 | 0.736710 | 0.723169 |
| L | farfum_rop | 230 | 0.805899 | 0.640088 | 0.614911 |
| B | farfum_rop | 230 | 0.865237 | 0.720389 | 0.720659 |
| M0 | farabi | 212 | 0.733750 | 0.539962 | 0.536559 |
| M1 | farabi | 212 | 0.756392 | 0.566793 | 0.568069 |
| L | farabi | 212 | 0.713728 | 0.493790 | 0.474652 |
| B | farabi | 212 | 0.830400 | 0.668936 | 0.670590 |

`plus` has no Pre_Plus, so its 3-class AUC is undefined and is not substituted; balanced accuracy and
macro F1 there are over the classes actually present. No source-specific tuning was done.

The same pattern repeats in every subgroup: M1 > M0 by a wide margin (plus balanced accuracy
0.741 vs 0.629; farfum_rop AUC 0.869 vs 0.852; farabi AUC 0.756 vs 0.734), and the ROPDeepX-style
RGB representation trails the original B5 embedding most sharply on farabi (0.734 vs 0.830).

## 11. Overfitting diagnostics

* **Selected epoch**: 2. **Minimum val-loss epoch**: 7 (0.8683). **Maximum val-AUC epoch**: 2 (0.86624).
* Train loss fell monotonically 0.7638 → 0.5574 over 7 epochs (a 27 % reduction) while val AUC peaked
  at epoch 2 and declined to 0.84014 by epoch 7. Validation macro F1 was still rising
  (0.5752 → 0.6040), which is why the AUC-first selection rule picked epoch 2 and not a later epoch.
* So yes: **train loss kept improving while validation AUC degraded.** The overfit is far milder than
  in Tasks 9 and 10 — plateau rather than blow-up, val loss staying in a narrow 0.868–0.913 band —
  which is what the staged freezing plus OneCycleLR was meant to achieve.
* **Attention collapse check: NO.** Neither branch's mean attention approaches 0.95 at any epoch
  (range 0.48–0.63 for ResNet). Nothing was retrained on the basis of this observation.

## 12. Interpretation against §20

§20 lists four possible outcomes. What happened is a **hybrid of A and D**:

* **Question B — vessel complementarity: yes, and strongly.** M1 − M0 AUC +0.020936 with a CI well
  clear of zero, macro F1 +0.040404 and Brier −0.049181 also significant. This is the largest
  M-vs-no-vessel gain measured anywhere in the series (Task 8B: +0.0079 for E − B). Adding spatial
  vascular morphology helps even after the RGB side is replaced by a two-backbone attention model.
  The vessel representation is not simply compensating for a weak RGB encoder.
* **Question A — a stronger RGB representation: no.** M0 − B_EMBEDDING_ONLY AUC is −0.018925 with
  p = 0.0063: the ROPDeepX-style dual-RGB attended representation is *significantly worse* than the
  project's existing single-backbone EfficientNet-B5 embedding on this dataset. Macro F1 and Brier
  agree. This is the D-flavoured part: swapping in an architecture from the literature did not help
  here.
* **M1 vs E and M1 vs G are null.** Point estimates slightly favour E and G (−0.0059 and −0.0080
  AUC) but every CI crosses zero (p = 0.221 and 0.095). So the original B5 + vessel model and the
  dual-RGB + vessel model are statistically indistinguishable on this test set, and the earlier
  models remain the best point estimates.

Two honest caveats on Question A. The dual-RGB model is not a like-for-like architectural swap: it
has two backbones, ≈6.3 M head parameters with BatchNorm, label smoothing, OneCycleLR and staged
fine-tuning, whereas B_EMBEDDING_ONLY is XGBoost on a frozen single-backbone embedding. The finding
is that *this* end-to-end trained dual-RGB pipeline produces a less useful representation for
downstream classification, not that ResNet50 or attention fusion are inherently inferior. And the
selection landed on a stage-1 checkpoint, i.e. before any backbone fine-tuning occurred, so the
fine-tuning stages never paid off on validation.

## 13. Limitations, stated plainly

1. Batch size was not specified by the contract; 16 was chosen before training and is documented.
2. The dual-RGB pipeline and B_EMBEDDING_ONLY differ in learner and training regime, not only in
   backbone, so the M0 − B comparison is not a controlled architecture ablation.
3. The selected checkpoint is a frozen-backbone stage-1 model; stage-2 fine-tuning was never selected.
4. Attention weights are strongly source-dependent, so attention-based interpretation carries an
   acquisition confound and should not be read clinically.
5. This is a secondary post-primary exploratory architecture experiment on an already-used canonical
   split; it is not confirmatory, and no architecture variant was created after TEST was read.
6. No operational, screening, or clinical claim follows from any of these numbers.

## 14. Artifacts

`artifacts/task13_ropdeepx_style/`: `L_selected.pth` and per-epoch checkpoints · `L_history.json` ·
`L_selection.json` · `loss_config.json` · `attention_stats_val.csv` · `attention_stats_val_test.csv` ·
`attention_by_class_and_source.csv` · `ropdeepx_attended_embeddings_8862.parquet` ·
`M0_ROPDEEPX_EMBEDDING_ONLY.json` · `M1_ROPDEEPX_PLUS_VESSEL.json` · `task13_selection_frozen.json` ·
`test_predictions_{L,M0,M1}.csv` · `metrics_summary.csv` · `paired_all_10k.csv` and the four
per-comparison files · bootstrap distributions · `source_breakdown.csv` · confusion matrices ·
`learning_curves_and_attention.png` · `calibration_and_forest.png` · `task13_summary.json` ·
`artifact_sha256.json`.
