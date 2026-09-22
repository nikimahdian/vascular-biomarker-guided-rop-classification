# Task 9 — Joint RGB–Vessel Learned Fusion with Supportive Biomarker Branch

Status: `TASK9_STATUS = COMPLETE`
Classification: `SECONDARY_POST_PRIMARY_EXPLORATORY_CANONICAL_SPLIT_REANALYSIS`
TEST policy: opened exactly once, after `TASK9_ALL_SELECTION_FROZEN = YES`.

**Headline: Task 9 is a negative result.** The learned joint multimodal fusion did not beat the
frozen XGBoost feature concatenation, and adding the five scalar biomarkers inside the learned
network made discrimination significantly worse. Details and numbers below.

## 1. Predeclared models

| model | definition |
|---|---|
| `H_JOINT_RGB_VESSEL` | frozen B_RGB EfficientNet-B5 (2048) + frozen D EfficientNet-B4 (1792), each projected to 512, interaction `z = [r, v, r*v, |r−v|]` (2048), MLP trunk 2048→512→128, `Linear(128,3)` |
| `I_JOINT_RGB_VESSEL_BIOMARKER` | identical trunk; `h(128) ++ biomarker32` → `Linear(160,3)`; biomarkers standardised on TRAIN only |

Both were trained; neither existence nor execution depended on the other's result. No further
architecture variant was created, and no variant was created after TEST.

## 2. Pre-flight (all PASS, before any training)

`population_8862` · `split_counts` (6203/1328/1331) · `no_group_overlap` · `labels_0_1_2` ·
`biomarkers_finite` · `rgb_exist` · `masks_exist` · `B_ckpt_sha` = `2b0b434c…` ·
`D_ckpt_sha` = `3e1719eb…`.

Encoders were initialised from the frozen selected checkpoints, not randomly: both checkpoint
SHA-256 values were verified before training started.

Biomarker scaler fitted on TRAIN only (n = 6203):
mean `[0.0981, 0.0191, 1.3134, 1.2900, 1.2790]`, std `[0.0356, 0.0102, 0.0867, 0.0855, 0.0844]`
(ddof = 0). SHA-256 `7e8d888feced3f34…`.

## 3. Training contract

| element | value |
|---|---|
| Loss | weighted CE, Task-6 TRAIN-only weights 0.46051 / 3.18103 / 1.94512 |
| Seed / batch | 42 / 16 (never changed for performance) |
| Precision | CUDA bfloat16 autocast — **used: yes** (`torch.cuda.is_bf16_supported()` True) |
| Stage 1 | both encoders frozen, lr 3e-4, weight decay 1e-4, 3 epochs |
| Stage 2 | last two EfficientNet block groups + `conv_head`/`bn2` of both encoders unfrozen, differential lr 1e-5 (encoder) / 1e-4 (new layers), weight decay 1e-4, max 25 epochs, patience 6 |
| Augmentation | **one** geometric draw per sample applied identically to RGB and mask: RandomResizedCrop scale 0.95–1.0 with the same `get_params` crop box, HFlip, rotation ±10°; RGB BILINEAR, mask NEAREST; photometric ColorJitter(0.10,0.10,0.10,0.0) on RGB only |
| VAL / TEST | no augmentation |
| Selection | VAL only: AUC > macro F1 > −loss > earlier epoch |

Both models used seed 42 in the same order, so the augmentation stream and the fusion-layer
initialisation are identical for H and I; the biomarker branch is the only difference.

## 4. Model selection (frozen before TEST)

```
H_SELECTED_EPOCH:      4   (stage 2)
H_SELECTED_VAL_AUC:    0.8842844
H_SELECTED_VAL_MACRO_F1: 0.6044541
H_CHECKPOINT_SHA256:   7c0b6d2cc63f7f2f73fa1c1d4974a9eb7b9cb313b16c9f4c442ac8e1e59297c6

I_SELECTED_EPOCH:      6   (stage 2)
I_SELECTED_VAL_AUC:    0.8884681
I_SELECTED_VAL_MACRO_F1: 0.6047706
I_CHECKPOINT_SHA256:   001a2e91b95503eaf2b327d228ed417d2baa35e9a4e02dbfe95ee0cab59b9446

TASK9_ALL_SELECTION_FROZEN: YES
TEST_TOUCHED: NO
```

Epochs run: H 10 (3 stage-1 + 7 stage-2, early stop), I 12 (3 + 9, early stop).

Note already at this point: H's best validation AUC (0.88428) is **below** the frozen B_RGB
validation AUC (0.8915684). The joint network never matched the single-modality RGB model on
validation.

## 5. TEST results (N = 1331, single use)

| model | AUC | balanced acc | macro F1 | Brier | ECE |
|---|---|---|---|---|---|
| `H_JOINT_RGB_VESSEL` | 0.925825 | 0.657449 | 0.684038 | 0.275935 | 0.115912 |
| `I_JOINT_RGB_VESSEL_BIOMARKER` | 0.916247 | 0.673028 | 0.697780 | 0.278647 | 0.115207 |
| `E_RGB_VESSEL_FEATURE_FUSION` (frozen) | **0.932841** | 0.735049 | 0.724571 | 0.248590 | 0.092142 |
| `G_RGB_VESSEL_SCALAR_FUSION` (frozen) | **0.934880** | 0.738895 | 0.729163 | 0.246515 | 0.092080 |

Both Task-9 models are worse than both Task-8 XGBoost models on every metric except balanced
accuracy / macro F1, where I sits between H and E, and both Task-9 models have substantially worse
calibration (ECE ≈ 0.115 vs 0.092).

## 6. Primary comparison — H vs E (10,000 paired class-stratified bootstrap, seed 42)

| metric | delta H − E | 95% CI | p |
|---|---|---|---|
| multiclass AUC | −0.007016 | [−0.016296, +0.001999] | 0.1311 |
| balanced accuracy | **−0.077600** | [−0.112761, −0.042519] | <0.0001 |
| macro F1 | **−0.040533** | [−0.074377, −0.007469] | 0.0169 |
| Brier | **+0.027346** | [+0.002996, +0.051529] | 0.0281 |
| ECE | **+0.023770** | [+0.006476, +0.041514] | 0.0083 |

The primary Task-9 hypothesis is **not supported**. AUC is worse as a point estimate with the CI
crossing zero; balanced accuracy, macro F1, Brier and ECE are all significantly worse. Learned
cross-modal interaction did not improve on simple feature concatenation feeding XGBoost.

## 7. Biomarker question — I vs H (same 10,000-replicate protocol)

| metric | delta I − H | 95% CI | p |
|---|---|---|---|
| multiclass AUC | **−0.009578** | [−0.016273, −0.002854] | 0.0049 |
| balanced accuracy | +0.015580 | [−0.013382, +0.045035] | 0.2933 |
| macro F1 | +0.013743 | [−0.015333, +0.043408] | 0.3519 |
| Brier | +0.002711 | [−0.017307, +0.024149] | 0.7953 |
| ECE | −0.000705 | [−0.015489, +0.014349] | 0.9209 |

Direct answer to the pre-declared question: inside a learned joint multimodal fusion network the
five scalar biomarkers do **not** provide incremental value. They significantly **reduce**
discrimination (AUC −0.0096, p = 0.0049) while leaving the decision-threshold metrics and
calibration unchanged. This is the opposite direction from the Task-8 XGBoost setting, where the
same five biomarkers gave a small unsupported positive point estimate (+0.002039, p = 0.0900).

## 8. Comparison against the Task-8 best point estimate — H/I vs G

| metric | delta H − G | 95% CI | p | delta I − G | 95% CI | p |
|---|---|---|---|---|---|---|
| multiclass AUC | −0.009055 | [−0.018393, +0.000094] | 0.0549 | **−0.018633** | [−0.027955, −0.009289] | 0.0002 |
| balanced accuracy | **−0.081446** | [−0.115821, −0.046884] | <0.0001 | **−0.065866** | [−0.100097, −0.031469] | 0.0001 |
| macro F1 | **−0.045125** | [−0.078698, −0.012035] | 0.0083 | −0.031382 | [−0.063098, +0.000485] | 0.0541 |
| Brier | **+0.029420** | [+0.005381, +0.054016] | 0.0170 | **+0.032132** | [+0.006752, +0.057741] | 0.0143 |
| ECE | **+0.023832** | [+0.006297, +0.040441] | 0.0063 | **+0.023127** | [+0.004599, +0.040365] | 0.0115 |

G remains the best point estimate in the whole Task-6→9 series. I is significantly worse than G on
AUC, balanced accuracy, Brier and ECE. These are exploratory comparisons and are not converted into
confirmatory claims.

## 9. Source breakdown (TEST, frozen predictions)

| source | n | H AUC | H bal acc | H F1 | I AUC | I bal acc | I F1 |
|---|---|---|---|---|---|---|---|
| plus | 889 | n/a | 0.678301 | 0.699532 | n/a | 0.493210 | 0.473341 |
| farfum_rop | 230 | 0.832686 | 0.687739 | 0.650206 | 0.873055 | 0.756532 | 0.749556 |
| farabi | 212 | 0.825987 | 0.624200 | 0.627182 | 0.829857 | 0.642103 | 0.643631 |

`plus` has no Pre_Plus, so 3-class multiclass AUC is undefined there and was not substituted — the
NaN is reported rather than an invalid two-class AUC. No source-specific tuning was performed.

The subgroup pattern explains the overall aggregate: I is much better than H on both minority
sources (farfum_rop balanced accuracy 0.7565 vs 0.6877; farabi 0.6421 vs 0.6242) but collapses to
near chance on the majority `plus` source (balanced accuracy 0.4932). Because `plus` is 67 % of TEST,
that collapse dominates I's aggregate AUC and is why I's AUC is worse while its balanced accuracy
and macro F1 are marginally higher than H's.

For reference on the same subgroups: E `plus` balanced accuracy 0.7201, farfum_rop 0.7406,
farabi 0.6095; G `plus` 0.7195, farfum_rop 0.7342, farabi 0.6375.

## 10. Overfitting diagnostics (per epoch, required by §25)

**H** (early stop after stage-2 epoch 10):

| stage | epoch | train loss | val loss | val AUC | val macro F1 |
|---|---|---|---|---|---|
| 1 | 1 | 0.3926 | **0.9316** | 0.86422 | 0.5847 |
| 1 | 2 | 0.3312 | 1.2948 | 0.86657 | 0.5761 |
| 1 | 3 | 0.2920 | 1.2094 | 0.84606 | 0.5750 |
| 2 | **4** | 0.2658 | 1.3088 | **0.88428** | **0.6045** |
| 2 | 5 | 0.2312 | 2.0687 | 0.85622 | 0.4925 |
| 2 | 6 | 0.1869 | 1.4511 | 0.88048 | 0.5519 |
| 2 | 7 | 0.1554 | 1.5204 | 0.87269 | 0.5350 |
| 2 | 8 | 0.1223 | 1.4610 | 0.87964 | 0.6018 |
| 2 | 9 | 0.1074 | 1.8241 | 0.87206 | 0.5409 |
| 2 | 10 | 0.0814 | 2.8079 | 0.84080 | 0.4953 |

**I** (early stop after stage-2 epoch 12):

| stage | epoch | train loss | val loss | val AUC | val macro F1 |
|---|---|---|---|---|---|
| 1 | 1 | 0.3817 | **1.0162** | 0.86939 | 0.5940 |
| 1 | 2 | 0.3383 | 1.4477 | 0.85216 | 0.5625 |
| 1 | 3 | 0.2961 | 1.2361 | 0.87459 | 0.5578 |
| 2 | 4 | 0.2571 | 1.0855 | 0.87740 | 0.5933 |
| 2 | 5 | 0.2098 | 1.1035 | 0.88436 | 0.5932 |
| 2 | **6** | 0.1786 | 1.1767 | **0.88847** | **0.6048** |
| 2 | 7 | 0.1381 | 1.4367 | 0.88672 | 0.5679 |
| 2 | 8 | 0.1265 | 1.8791 | 0.87146 | 0.5454 |
| 2 | 9 | 0.0915 | 2.0958 | 0.86129 | 0.5103 |
| 2 | 10 | 0.0914 | 1.7132 | 0.85214 | 0.5533 |
| 2 | 11 | 0.0840 | 1.5166 | 0.87381 | 0.5431 |
| 2 | 12 | 0.0583 | 2.3971 | 0.85389 | 0.5293 |

Explicit answers:

* **Selected epoch**: H stage-2 epoch 4; I stage-2 epoch 6.
* **Epoch of minimum validation loss**: H epoch 1 (0.9316); I epoch 1 (1.0162). Both are stage-1
  epochs, i.e. the minimum validation loss occurs *before* any encoder is unfrozen.
* **Did validation degrade while train loss kept improving?** Yes, unambiguously, for both models.
  Train loss falls monotonically the whole way (H 0.3926 → 0.0814; I 0.3817 → 0.0583) while
  validation loss rises from its epoch-1 minimum to 2.81 (H) and 2.40 (I). Validation AUC peaks
  within two epochs of unfreezing and then decays.
* **Generalization gap** at the final epoch: H train 0.0814 vs val 2.8079 (ratio ≈ 34×); I train
  0.0583 vs val 2.3971 (ratio ≈ 41×).

This is a stronger version of the pattern already seen in Task-8 model D: with 6,203 training
images, both 384×384 encoders and 5.4 M fusion parameters overfit very quickly once the encoder
tails become trainable, even at an encoder lr of 1e-5. Learning curves:
`learning_curves.png`.

Stage-1-only checkpoints were available to the selection rule and none of them won on AUC — the
best stage-1 validation AUC was 0.86657 (H) and 0.87459 (I), so selection preferred a stage-2
checkpoint even though the stage-1 epochs had the lower validation loss. This is the selection rule
working as frozen (AUC first), not a post-hoc choice.

## 11. Optional 3-seed ensemble — NOT executed

`H_JOINT_3SEED_ENSEMBLE` was **not** trained. Each H/I run took ≈ 10–12 minutes, and the section-17
precondition was that the ensemble only be attempted if runtime remained very low after H and I
finished; it did not. The declaration was written into `task9_selection_frozen.json` **before** any
TEST access, and no H ensemble TEST prediction exists. No member was selected.

## 12. Integrity

**Task 6 / 7 / 8 artifacts were not modified.** Verified by recomputing every hash in each
directory's own `artifact_sha256.json`:

| directory | manifest entries | mismatched or missing |
|---|---|---|
| `artifacts/task6` | 37 | none |
| `artifacts/task7_paired_statistics` | 8 | none |
| `artifacts/task8_spatial_vessel_fusion` | 32 | 2 — see below |

The two Task-8 mismatches are `task8_progress.log` and `train.log`, and the cause is fully
explained, not silent: in the original Task-8 run the SHA manifest was computed *before* the last
two writes of the same process — the `TASK8_STATUS = COMPLETE` line appended to
`task8_progress.log`, and the final stdout metric echo written to `train.log` by the `nohup`
redirect. The current tails are exactly those lines (progress log ends
`[726.5s] TASK8_STATUS = COMPLETE`; train.log ends with the metric JSON array). Current hashes:
`task8_progress.log` `8893fe4ea85ab407…`, `train.log` `67fd401ae8f19b64…`. All 30 non-log Task-8
artifacts — checkpoints, embeddings, predictions, metrics, paired statistics and the whole
`statistical_closure_10k/` directory — match their manifest hashes byte for byte. No scientific
artifact was altered; the affected files are append-only execution logs of the run that produced
them. Lesson recorded in `docs/SERVER_RUNTIME_LOG.md`: write the SHA manifest as the last action
after every log write, or exclude logs from the manifest.

Task-9 artifact hashes: `H_JOINT_RGB_VESSEL_selected.pth`
`7c0b6d2cc63f7f2f73fa1c1d4974a9eb7b9cb313b16c9f4c442ac8e1e59297c6`,
`I_JOINT_RGB_VESSEL_BIOMARKER_selected.pth`
`001a2e91b95503eaf2b327d228ed417d2baa35e9a4e02dbfe95ee0cab59b9446`,
`test_predictions_H_JOINT_RGB_VESSEL.csv` `35cc64b2ca807b44…`,
`test_predictions_I_JOINT_RGB_VESSEL_BIOMARKER.csv` `2d725e3b70e4249e…`,
`biomarker_scaler.json` `7e8d888feced3f34…`, `metrics_summary.csv` `eb736f4948ff87ac…`,
`task9_selection_frozen.json` `050c1a8d4a3cea23…`.

## 13. Interpretation, stated plainly

1. The primary Task-9 hypothesis fails. Learned cross-modal interaction (multiplicative +
   absolute-difference fusion of projected branch embeddings) is worse than concatenating the raw
   RGB and vessel embeddings and fitting XGBoost on 3,840 features.
2. Adding the five scalar biomarkers inside the network significantly **hurts** discrimination
   (I − H AUC p = 0.0049). Combined with Task-8B's G − E result, the scalar biomarker panel has now
   failed to show incremental value twice, in two different fusion regimes, and in the learned regime
   it is actively harmful.
3. Both Task-9 models overfit fast and hard. The selected checkpoints land two epochs after
   unfreezing, and validation loss is lowest before unfreezing at all. With 6,203 training images
   this architecture has far more capacity than the cohort supports.
4. Calibration is clearly worse than the XGBoost models (ECE ≈ 0.115 vs ≈ 0.092), which matters if
   any future work wants probabilities rather than rankings.
5. The subgroup pattern is a caution: I is better than H on both minority sources yet collapses on
   `plus` (balanced accuracy 0.4932, i.e. chance). Aggregate TEST numbers here are dominated by a
   67 % majority source, so the aggregate AUC ranking and the per-source ranking disagree.
6. Nothing here changes the primary Task-6/7 conclusions, and G (frozen, Task 8) remains the best
   point estimate in the series — as an exploratory result, not a confirmatory one.
7. No operational, screening, or clinical claim follows from any of these numbers.

## 14. Artifacts

`artifacts/task9_joint_multimodal_fusion/`:
selected checkpoints (H, I) · per-epoch checkpoints · training histories · selection metadata ·
`biomarker_scaler.json` · `task9_selection_frozen.json` · frozen TEST predictions for H and I ·
`metrics_summary.csv` · `paired_H_minus_E.csv` · `paired_I_minus_H.csv` · `paired_H_minus_G.csv` ·
`paired_I_minus_G.csv` · `paired_all_10k.csv` · bootstrap distributions · `source_breakdown.csv` ·
confusion matrices · `learning_curves.png` · `calibration_and_forest.png` · `artifact_sha256.json`.
