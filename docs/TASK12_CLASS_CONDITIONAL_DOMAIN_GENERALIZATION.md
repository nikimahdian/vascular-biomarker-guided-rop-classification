# Task 12 — Class-Conditional Domain-Invariant RGB–Vessel Representation

Status: `TASK12_STATUS = COMPLETE`
Classification: `SECONDARY_SOURCE_GENERALIZATION_DOMAIN_INVARIANCE_EXPERIMENT`
`HELDOUT_SOURCE_USED_DURING_TRAINING = NO` — not for training, validation, normalization, alignment,
early stopping, thresholding or hyperparameter choice, and not even as unlabeled features.
No CNN was executed, no embedding regenerated, and the five scalar biomarkers were not used.

**Headline: outcome B, with signs of C. The invariance terms did reduce representation shift, but
domain predictability stayed at 0.88–0.98 and held-out disease classification did not improve.**
Per §23 this is a negative result and is not reinterpreted as success.

## 1. Folds and target isolation

| fold | held out | training domains (2) | optimization train | source-validation | held-out test |
|---|---|---|---|---|---|
| A | plus | farfum_rop + farabi | 2530 | 407 | 5925 |
| B | farfum_rop | plus + farabi | 6224 | 1110 | 1528 |
| C | farabi | plus + farfum_rop | 6335 | 1118 | 1409 |

Group-level 85/15 split, seed 42, stratified by source × class. No group appears in both halves.
Every feasible source/class combination is present in both the optimization train and the
source-validation halves; the one case that cannot be (Pre_Plus inside `plus`, which has none) is
simply absent from that source and is never fabricated. Full per-cell group and row counts are in
`fold_definitions.json` and `source_splits_and_class_weights.json`.

Class weights were computed from the **optimization train only** by inverse frequency,
`w_c = n / (3 n_c)`, and frozen before training:

| fold | Normal | Pre_Plus | Plus |
|---|---|---|---|
| plus | 0.85185 | 1.06080 | 1.13199 |
| farfum_rop | 0.43277 | 5.24473 | 2.00549 |
| farabi | 0.41013 | 5.44186 | 2.64573 |

These differ sharply across folds — the farfum_rop and farabi folds carry a Pre_Plus weight an order
of magnitude larger than the plus fold — which is exactly why §14 of the contract forbids reusing the
global Task-6 weights here.

## 2. Models

Frozen input: 3840-d `[RGB 2048 | vessel 1792]`, no biomarkers, no CNN.

Shared architecture, deliberately small to avoid the Task-9/10 overfitting pattern:
`LayerNorm(2048)→Linear(2048,256)→GELU→Dropout(0.40)` for RGB,
`LayerNorm(1792)→Linear(1792,256)→GELU→Dropout(0.40)` for vessel, concatenated to 512,
then `Linear(512,256)→GELU→Dropout(0.50)` giving `z ∈ R^256`, then `Linear(256,3)`.

* **K0_DOMAIN_NEUTRAL_CONTROL** — weighted CE only.
* **K1_CC_DANN_MMD** — identical, plus (i) a GRL domain discriminator `256→128→2` with cross-entropy
  and `lambda_dann(p) = 2/(1+exp(-10p)) - 1`, `p` running 0→1 across the whole training budget, the
  adversarial contribution fixed at `0.10 * lambda_dann(p)`; and (ii) **class-conditional** multi-kernel
  RBF MMD on `z` with coefficient `0.10` fixed. Bandwidths are `[0.5, 1.0, 2.0, 4.0] ×` the batch
  median pairwise squared distance, i.e. `k_m(x,y) = exp(-||x-y||² / (m · base))`, averaged over the
  four kernels. MMD is computed **per disease class**, only between the two training domains, and a
  class absent from either domain is skipped — for folds B and C that is Pre_Plus inside `plus`.
  Nothing is synthesised and no cross-class alignment is ever forced.

Everything else is identical between K0 and K1: input, architecture, split, optimizer
(AdamW lr 1e-4, weight decay 1e-3), 64+64 balanced domain batches of 128, weighted CE with the
frozen per-fold weights, seed 42, 40 epochs maximum, patience 6. Selection uses source-validation
only (AUC > macro F1 > −classification loss > earlier epoch); domain-classifier accuracy is never
used for selection.

Measured domain loss during K1 training shows the discriminator degrading as `alpha` grows — e.g.
plus fold `dom` 0.6832 → 0.6152, and the farabi fold `dom` 0.5775 → 0.6065 while `mmd` fell
0.12264 → 0.08115 and `alpha` rose 0.0061 → 0.0786. So the adversarial term was active; the question
is whether it helped the disease task.

## 3. Loss curves

Recorded separately per epoch for every fold: classification loss, domain loss, MMD loss and the
current `alpha`; plus source-validation loss, AUC and macro F1.
See `loss_and_selection_curves.png` and `{K0,K1}_{fold}_history.json`.

## 4. Model selection (frozen before any held-out access)

| fold | K0 selected epoch | K0 val AUC | K1 selected epoch | K1 val AUC |
|---|---|---|---|---|
| plus | 4 | 0.89806 | 4 | 0.89645 |
| farfum_rop | 3 | 0.96466 | 4 | 0.96432 |
| farabi | 3 | 0.96548 | 3 | 0.96490 |

K1's selected source-validation AUC is marginally *lower* than K0's in all three folds. All six
checkpoint SHA-256 values were written to `task12_selection_frozen.json` together with
`TASK12_ALL_FOLDS_SELECTION_FROZEN = YES` and `HELDOUT_TARGETS_TOUCHED = NO`.

## 5. Domain-leakage and representation-shift diagnostics (source-validation only)

Domain predictability: logistic regression on the 256-d `z`, 5-fold CV accuracy on source-validation.

| fold | K0 | K1 | chance |
|---|---|---|---|
| plus | 0.8900 | 0.8802 | 0.5 |
| farfum_rop | 0.9650 | **0.9659** | 0.5 |
| farabi | 0.9894 | 0.9824 | 0.5 |

**Domain predictability was not meaningfully reduced.** Two folds move down by 0.010 and 0.007, and
the farfum_rop fold moves slightly *up*. All six values remain between 0.88 and 0.99 against a 0.5
chance level. The representation is still almost perfectly source-identifiable.

Representation shift in `z` on source-validation:

| fold | K0 centroid dist (std) | K1 centroid dist (std) | K0 median \|SMD\| | K1 median \|SMD\| | K0 p90 | K1 p90 | K0 p95 | K1 p95 |
|---|---|---|---|---|---|---|---|---|
| plus | 12.003 | **11.134** | 0.6261 | **0.5770** | 1.1725 | 1.0649 | 1.2467 | 1.1496 |
| farfum_rop | 82.271 | **71.044** | 2.6314 | **2.2003** | 8.0875 | 6.8800 | 10.1842 | 8.4201 |
| farabi | 81.804 | **63.861** | 1.9353 | **1.5503** | 7.3480 | 6.3867 | 12.9265 | 8.0449 |

**Representation shift was reduced** in all three folds on every summary statistic — centroid
distance by 7–22 %, median |SMD| by 8–20 %. So K1 did move the marginal distributions closer. The
magnitudes are still large (median |SMD| 0.58–2.20), and none of it removed source identifiability.

## 6. Held-out evaluation (single pass, after the freeze)

**Held out farfum_rop** (n = 1528):

| model | 3-class AUC | balanced acc | macro F1 | Brier | ECE |
|---|---|---|---|---|---|
| K0 | 0.83293 | 0.62545 | 0.57160 | 0.64594 | 0.26048 |
| K1 | 0.83013 | 0.62818 | 0.55934 | 0.70019 | 0.29956 |
| Task-11 E_LOSO | 0.80978 | 0.66352 | 0.65897 | 0.51069 | 0.18319 |

**Held out farabi** (n = 1409):

| model | 3-class AUC | balanced acc | macro F1 | Brier | ECE |
|---|---|---|---|---|---|
| K0 | 0.79192 | 0.53726 | 0.50112 | 0.59932 | 0.21763 |
| K1 | 0.79103 | 0.53457 | 0.50348 | 0.59448 | 0.21711 |
| Task-11 E_LOSO | 0.78290 | 0.58604 | 0.57205 | 0.61871 | 0.23911 |

**Held out plus** (n = 5925; Pre_Plus absent → three-class AUC = NaN, reported as NaN and never
substituted):

| model | RESTRICTED_NORMAL_VS_PLUS AUC | balanced acc (2 classes present) | macro F1 (2 classes present) | Brier (2-class) | ECE |
|---|---|---|---|---|---|
| K0 | 0.98932 | 0.94173 | 0.87418 | — | 0.23208 |
| K1 | 0.98981 | 0.94192 | 0.87091 | — | 0.23111 |
| Task-11 E_LOSO | 0.98458 | 0.90197 | 0.88872 | — | 0.01658 |

Per-class AUCs and confusion matrices are in `heldout_metrics.csv` and
`confusion_{K0,K1}_{fold}.csv`.

## 7. Paired statistics — 10,000 class-stratified bootstrap replicates, seed 42

**Primary: K1 − K0**

| fold | metric | delta | 95% CI | p |
|---|---|---|---|---|
| farfum_rop | 3-class AUC | −0.002796 | [−0.006546, +0.000985] | 0.1428 |
| farfum_rop | balanced acc | +0.002727 | [−0.007247, +0.012701] | 0.5865 |
| farfum_rop | macro F1 | **−0.012262** | [−0.023740, −0.001240] | 0.0317 |
| farfum_rop | Brier | **+0.054253** | [+0.048319, +0.060152] | <0.0001 |
| farfum_rop | ECE | **+0.039079** | [+0.030698, +0.051016] | <0.0001 |
| farabi | 3-class AUC | −0.000883 | [−0.002439, +0.000685] | 0.2588 |
| farabi | balanced acc | −0.002692 | [−0.010735, +0.005277] | 0.5089 |
| farabi | macro F1 | +0.002356 | [−0.007202, +0.012421] | 0.6351 |
| farabi | Brier | **−0.004839** | [−0.007563, −0.002140] | 0.0009 |
| farabi | ECE | −0.000528 | [−0.008015, +0.007136] | 0.8908 |
| plus (restricted) | RESTRICTED_NORMAL_VS_PLUS AUC | **+0.000485** | [+0.000146, +0.000900] | 0.0123 |
| plus (restricted) | balanced acc | +0.000189 | [−0.000943, +0.001320] | 0.8088 |
| plus (restricted) | macro F1 | **−0.003273** | [−0.005930, −0.000756] | 0.0137 |
| plus (restricted) | Brier (2-class) | **+0.005121** | [+0.004792, +0.005451] | <0.0001 |
| plus (restricted) | ECE | −0.000968 | [−0.002998, +0.001084] | 0.3546 |

**Secondary: K1 − Task-11 E_LOSO**

| fold | metric | delta | 95% CI | p |
|---|---|---|---|---|
| farfum_rop | 3-class AUC | **+0.020350** | [+0.010190, +0.030581] | 0.0001 |
| farfum_rop | balanced acc | **−0.035341** | [−0.055650, −0.014500] | 0.0011 |
| farfum_rop | macro F1 | **−0.099625** | [−0.122703, −0.076197] | <0.0001 |
| farfum_rop | Brier | **+0.189505** | [+0.163685, +0.214475] | <0.0001 |
| farfum_rop | ECE | **+0.116374** | [+0.092094, +0.140781] | <0.0001 |
| farabi | 3-class AUC | +0.008135 | [−0.002668, +0.018738] | 0.1373 |
| farabi | balanced acc | **−0.051470** | [−0.076861, −0.026469] | 0.0001 |
| farabi | macro F1 | **−0.068567** | [−0.097025, −0.040867] | <0.0001 |
| farabi | Brier | −0.024231 | [−0.050713, +0.002539] | 0.0738 |
| farabi | ECE | −0.022000 | [−0.048999, −0.000029] | 0.0827 |
| plus (restricted) | RESTRICTED_NORMAL_VS_PLUS AUC | **+0.005229** | [+0.003422, +0.007117] | <0.0001 |
| plus (restricted) | balanced acc | **+0.039947** | [+0.029000, +0.051915] | <0.0001 |
| plus (restricted) | macro F1 | **−0.017817** | [−0.028107, −0.007135] | 0.0009 |
| plus (restricted) | Brier (2-class) | **+0.047405** | [+0.043041, +0.051816] | <0.0001 |
| plus (restricted) | ECE | **+0.214537** | [+0.206093, +0.218362] | <0.0001 |

The K1 − E comparisons are **not** like-for-like and must be read with that caveat: Task-11 `E_LOSO`
is XGBoost fitted on 3840 features using **100 %** of the two training sources, whereas K1 is a
neural head fitted on the **85 %** group-level optimization split of the same sources. Different
learner, different training rows, different calibration behaviour. The consistent pattern is that
the neural head ranks better (positive AUC) but thresholds and calibrates worse (negative macro F1,
positive Brier/ECE on farfum).

## 8. Interpretation against §23 and §24

§23 states the clinically relevant success criterion is that **K1 improves held-out disease
classification over K0**, with the strongest evidence being a positive held-out AUC delta whose CI
excludes zero without major degradation in macro F1, Brier or ECE. It explicitly says domain
invariance without disease improvement is not sufficient.

That criterion is **not met**:

* K1 − K0 held-out 3-class AUC is negative in both comparable folds (−0.002796 farfum_rop,
  −0.000883 farabi) with CIs crossing zero — no improvement, not even a positive point estimate.
* Where the K1 − K0 comparisons reach significance, they are mostly **harmful**: macro F1 −0.0123 and
  Brier +0.0543 and ECE +0.0391 on farfum_rop; macro F1 −0.0033 and 2-class Brier +0.0051 on plus.
  The single favourable significant result is Brier −0.0048 on farabi.
* The two invariance diagnostics disagree with each other: representation shift clearly fell, but
  domain predictability stayed at 0.88–0.98. Reducing marginal distance between the training
  distributions did not make the representation less source-identifiable.

This is **outcome B** — measurable source shift exists, and this class-conditional DANN+MMD
alignment does not improve unseen-source generalization — with elements of **outcome C** on
farfum_rop, where the alignment degraded macro F1 and calibration. It is not reinterpreted as
success, and no claim is made that domain invariance was achieved.

A plausible reading, stated as a hypothesis rather than a finding: the source differences that
matter here are not shifts in the *marginal* feature distribution within a class (which is what MMD
attacks) but differences in how the label relates to the features across sources, which marginal
alignment cannot fix and can disturb. That hypothesis is untested by this experiment.

## 9. Overfitting and training behaviour

Both K0 and K1 early-stopped around epoch 9–10 in every fold, with the selected checkpoint at
epoch 3–4. Source-validation AUC peaked early and then drifted down while the classification loss
kept falling (e.g. farabi K1: `cls` 0.5221 → 0.0516 while val loss 0.3650 → 0.7425). The smaller
architecture did not remove the overfitting, but it moved the onset later in absolute terms than
Tasks 9/10 and the source-validation AUCs (0.896–0.965) are far above the held-out values
(0.791–0.833) for the 3-class folds — a large source-validation-to-held-out gap, which is itself a
warning that source-validation model selection on two sources does not predict transfer to a third.

The plus fold is the outlier in the other direction: source-validation class counts are reasonably
balanced ({Normal 159, Pre_Plus 135, Plus 113}) while the held-out `plus` source is 89.5 % Normal and
contains no Pre_Plus at all. Its restricted AUC of ~0.989 therefore reflects an easier two-class task
rather than better transfer, and the negative macro F1 delta reflects the model paying for minority
recall it can no longer use.

## 10. Limitations, stated plainly

1. Only one invariance method and one hyperparameter setting were tried. `lambda_dann` and
   `lambda_mmd` were fixed at 0.10 by the contract and were never tuned, so "this method at these
   coefficients does not help" is the correct scope of the negative result, not "domain-invariant
   learning cannot help here".
2. The held-out source is a whole acquisition source, but not a whole patient population; no
   patient-level LOSO was performed.
3. Fold training sizes differ (2530 vs 6224 vs 6335), so cross-fold comparisons mix domain shift
   with training-set size.
4. The K1 − E_LOSO comparison differs in learner and training rows, as described in §7.
5. `plus` cannot yield a three-class metric; every number from that fold is labelled
   `RESTRICTED_NORMAL_VS_PLUS`.
6. This is a secondary post-primary exploratory analysis on a canonical cohort that was already used
   for the primary results; it is not confirmatory.
7. No operational, screening, or clinical claim follows from any of these numbers.

## 11. Provenance note on the evaluation pass

The first Task-12 run trained all six models, selected all six from source-validation only, wrote
`task12_selection_frozen.json`, and then crashed at held-out evaluation time on a pure
column-naming bug — `pr[f"{kind}_{c}"]` used a stale loop variable `c` left over from an earlier
loop instead of a class name, raising `KeyError: 2`.

`HELDOUT_TARGETS_TOUCHED` was still `NO` in the freeze manifest at that moment, and the failure
occurred after the freeze and after two lines of held-out metrics had been printed. Rather than
retrain, the held-out pass was re-run by a separate script (`scripts/task12_eval_frozen.py`) that
does **not** train: it re-verifies all six checkpoint SHA-256 values against the freeze manifest and
then performs only evaluation, statistics and figures. All six hashes matched. No model was
retrained, no parameter was reselected, and the frozen selection is exactly the one recorded before
any held-out access.

## 12. Artifacts

`artifacts/task12_class_conditional_domain_generalization/`: `fold_definitions.json` ·
`source_splits_and_class_weights.json` · K0/K1 selected checkpoints and per-epoch checkpoints ·
`{K0,K1}_{fold}_history.json` and `..._selection.json` · `task12_selection_frozen.json` ·
`domain_predictability.csv` · `representation_shift.csv` · `heldout_metrics.csv` ·
`heldout_predictions_{fold}.csv` · `heldout_probs_{K0,K1}_{fold}.npy` · `paired_bootstrap_10k.csv`
and the six per-comparison files · confusion matrices · `loss_and_selection_curves.png` ·
`calibration_heldout.png` · `domain_invariance_diagnostics.png` · `task12_summary.json` ·
`artifact_sha256.json`.
