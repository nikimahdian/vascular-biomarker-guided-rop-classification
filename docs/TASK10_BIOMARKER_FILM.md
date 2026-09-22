# Task 10 — Frozen Biomarker-Conditioned FiLM Fusion

Status: `TASK10_STATUS = COMPLETE`
Classification: `SECONDARY_POST_PRIMARY_EXPLORATORY_CANONICAL_SPLIT_REANALYSIS`
TEST policy: opened exactly once, after `TASK10_ALL_SELECTION_FROZEN = YES`.
No CNN was instantiated, loaded, or executed anywhere in this task.

**Headline: the matched FiLM comparison is null.** Biomarker conditioning left the vascular
representation essentially untouched — the learned modulation stayed within ±0.3 % of identity and
ablating any single biomarker moved the frozen probabilities by ≈2 × 10⁻⁵ with zero argmax flips.
J1 is not better than J0 on AUC, and the small Brier/ECE improvement cannot be attributed to the
biomarkers (see §7).

## 1. Frozen inputs (all hashes verified before training)

| input | shape | status |
|---|---|---|
| Task-6 B_RGB embedding (`embeddings_b5_8862.parquet`) | 8862 × 2048 | SHA unchanged ✓ |
| Task-8 D vessel embedding (`vessel_embeddings_b4_8862.parquet`) | 8862 × 1792 | SHA unchanged ✓ |
| E test predictions | 1331 × 3 | SHA unchanged ✓ |
| G test predictions | 1331 × 3 | SHA unchanged ✓ |

Population 8,862 = train 6,203 / val 1,328 / test 1,331. All five biomarkers finite.
Biomarker scaler fitted on TRAIN only: mean `[0.0981, 0.0191, 1.3134, 1.2900, 1.2790]`,
std `[0.0356, 0.0102, 0.0867, 0.0855, 0.0844]` (ddof = 0).

## 2. Matched architectures

| element | J0_FROZEN_FUSION_CONTROL | J1_BIOMARKER_FILM |
|---|---|---|
| RGB projection | `LayerNorm(2048) → Linear(2048,512) → GELU → Dropout(0.30)` | identical |
| Vessel projection | `LayerNorm(1792) → Linear(1792,512) → GELU → Dropout(0.30)` | identical |
| Conditioning | none, `v_conditioned = v` | `5→Linear(5,32)→GELU→Linear(32,64)→GELU`, then `delta_gamma = Linear(64,512)`, `beta_raw = Linear(64,512)` |
| Modulation | none | `gamma = 1 + 0.10·tanh(delta_gamma)`, `beta = 0.10·tanh(beta_raw)`, `v_film = gamma·v + beta` |
| Interaction | `z = [r, v, r*v, |r−v|]` (2048) | `z = [r, v_film, r*v_film, |r−v_film|]` (2048) |
| Fusion MLP | `LayerNorm(2048)→Linear(2048,256)→GELU→Dropout(0.40)→Linear(256,64)→GELU→Dropout(0.20)→Linear(64,3)` | identical |
| Parameters | 2,520,067 | 2,588,931 |

`delta_gamma` / `beta_raw` were initialised with `normal(0, 0.01)` weights and zero bias, so the
modulation starts essentially at identity. The FiLM design was not changed at any point.

## 3. Training contract

AdamW lr 3e-4, weight decay 1e-3, batch 128, max 50 epochs, patience 8, seed 42, weighted CE with
the frozen Task-6 weights 0.46051 / 3.18103 / 1.94512. The optimizer contains **only** the two
projections, the fusion MLP and (for J1) the FiLM conditioner — no CNN tensor exists in the graph
(`any CNN tensor: False`, verified at build time). Selection on VAL only: AUC > macro F1 > −loss >
earlier epoch. J0 was trained first and fully selected before J1 started; TEST was not inspected
between them.

## 4. Selection (frozen before TEST)

```
J0_SELECTED_EPOCH:       1
J0_VAL_AUC:              0.91591
J0_VAL_MACRO_F1:         0.6712

J1_SELECTED_EPOCH:       1
J1_VAL_AUC:              0.91670
J1_VAL_MACRO_F1:         0.6644

TASK10_ALL_SELECTION_FROZEN: YES
TEST_TOUCHED: NO
```

J0 checkpoint SHA-256 `3e1ebd55cf0c6b1e5fae4d6e873c2ad04ecf033992650c5274f8faec85013519`.
J1 checkpoint SHA-256 `330be575e07faa797e7a6d441d317fe033a32f5d101c743ec35572d34fd51b6e`.
`task10_selection_frozen.json` SHA-256 `c5039755e613bd0831a86989d3f3f3a522b3cf2291a7a7b3fc389a8305925dd4`.

Both models early-stopped at epoch 9 with epoch 1 selected: validation AUC peaked at the first epoch
in both cases.

## 5. TEST results (N = 1,331, single use)

| model | AUC | balanced acc | macro F1 | Brier | ECE |
|---|---|---|---|---|---|
| `J0_FROZEN_FUSION_CONTROL` | 0.929754 | 0.795392 | 0.685178 | 0.376967 | 0.167648 |
| `J1_BIOMARKER_FILM` | 0.927631 | 0.789728 | 0.688514 | 0.361425 | 0.153610 |
| `E_RGB_VESSEL_FEATURE_FUSION` (frozen) | **0.932841** | 0.735049 | 0.724571 | 0.248590 | 0.092142 |
| `G_RGB_VESSEL_SCALAR_FUSION` (frozen) | **0.934880** | 0.738895 | 0.729163 | 0.246515 | 0.092080 |

An unusual and important pattern: J0/J1 achieve **much higher balanced accuracy** (0.795 / 0.790)
than E/G (0.735 / 0.739) while having **lower AUC and macro F1** and **much worse calibration**
(Brier 0.38 / 0.36 vs 0.25, ECE 0.17 / 0.15 vs 0.09). This is the signature of the weighted
cross-entropy objective driving aggressive minority-class prediction: recall on Pre_Plus/Plus rises,
which balanced accuracy rewards, while ranking quality, macro F1 and probability calibration all
suffer. Balanced accuracy here is therefore not evidence of a better model, and the two metric
families disagree in direction. Any claim from these numbers must say which family it is using.

## 6. Paired statistics — 10,000 paired class-stratified bootstrap, seed 42 (Task-8B method)

**Primary: J1 vs J0**

| metric | delta J1 − J0 | 95% CI | p |
|---|---|---|---|
| multiclass AUC | −0.002123 | [−0.004427, +0.000143] | 0.0682 |
| balanced accuracy | −0.005665 | [−0.019504, +0.006812] | 0.4010 |
| macro F1 | +0.003336 | [−0.007633, +0.013963] | 0.5496 |
| Brier | **−0.015542** | [−0.022625, −0.008573] | 0.0002 |
| ECE | **−0.014038** | [−0.022744, −0.004676] | 0.0021 |

The primary comparison does **not** support biomarker conditioning. AUC is worse as a point estimate
with the CI marginally crossing zero; balanced accuracy and macro F1 are flat. Brier and ECE do
improve significantly — but §7 shows that improvement cannot be attributed to the biomarkers.

**Secondary: J1 vs E**

| metric | delta J1 − E | 95% CI | p |
|---|---|---|---|
| multiclass AUC | −0.005211 | [−0.011815, +0.001308] | 0.1192 |
| balanced accuracy | **+0.054679** | [+0.024163, +0.085784] | 0.0007 |
| macro F1 | **−0.036057** | [−0.061663, −0.009333] | 0.0071 |
| Brier | **+0.112836** | [+0.087253, +0.138440] | <0.0001 |
| ECE | **+0.061468** | [+0.041464, +0.080720] | <0.0001 |

**Secondary: J1 vs G**

| metric | delta J1 − G | 95% CI | p |
|---|---|---|---|
| multiclass AUC | **−0.007250** | [−0.014300, −0.000383] | 0.0424 |
| balanced accuracy | **+0.050833** | [+0.021542, +0.081603] | 0.0010 |
| macro F1 | **−0.040648** | [−0.065593, −0.014436] | 0.0015 |
| Brier | **+0.114910** | [+0.089799, +0.139921] | <0.0001 |
| ECE | **+0.061531** | [+0.041341, +0.079758] | <0.0001 |

J1 is significantly worse than G on AUC, macro F1, Brier and ECE, and better only on balanced
accuracy. G remains the best point estimate in the Task-6→10 series. These are exploratory
comparisons and are not converted into confirmatory claims.

## 7. FiLM diagnostics — did the network use the biomarkers?

| split | n | gamma mean | gamma std | gamma p05 | gamma p50 | gamma p95 | mean abs(gamma−1) |
|---|---|---|---|---|---|---|---|
| val | 1328 | 0.999575 | 0.001702 | 0.997063 | 0.999620 | 1.001834 | 0.001168 |
| test | 1331 | 0.999580 | 0.001685 | 0.996997 | 0.999624 | 1.001924 | 0.001190 |

| split | beta mean | beta std | beta p05 | beta p50 | beta p95 | mean abs(beta) |
|---|---|---|---|---|---|---|
| val | −0.0000177 | 0.001835 | −0.002629 | −0.0000076 | 0.002515 | 0.001210 |
| test | −0.0000060 | 0.001819 | −0.002695 | +0.0000011 | 0.002616 | 0.001238 |

`FILM_GAMMA_MEAN_ABS_DEVIATION_FROM_1 = 0.001190` (test)
`FILM_BETA_MEAN_ABS = 0.001238` (test)

The modulation is numerically near-identity: gamma stays inside [0.99700, 1.00192], i.e. within
±0.3 % of 1, and |beta| averages 1.2 × 10⁻³ on a 512-dimensional projection whose own scale is far
larger. The learned conditioner is effectively a no-op.

## 8. Biomarker sensitivity (post-freeze, descriptive, no retraining)

Each biomarker was replaced, one at a time, by its TRAIN mean (standardised value 0), all other
inputs unchanged, and the change in the frozen J1 TEST probabilities was measured.

| biomarker | mean abs Δp | median abs Δp | p95 abs Δp | max abs Δp | argmax flips |
|---|---|---|---|---|---|
| vessel_density_fov | 1.892e−05 | 2.55e−07 | 6.93e−05 | 1.86e−03 | 0 |
| skel_density_fov | 1.786e−05 | 2.38e−07 | 5.91e−05 | 2.13e−03 | 0 |
| fractal_d0 | 1.755e−05 | 3.12e−07 | 9.37e−05 | 7.48e−04 | 0 |
| fractal_d1 | 1.760e−05 | 2.38e−07 | 1.00e−04 | 6.81e−04 | 0 |
| fractal_d2 | 2.149e−05 | 2.38e−07 | 1.15e−04 | 1.11e−03 | 0 |

None of the five biomarkers drives the modulation, not even weakly: no single-biomarker ablation
flips a single TEST prediction, and the median change is at float32 noise level. No biomarker was
ranked or selected for a new model on the basis of this table.

## 9. Two readings, and why this design cannot separate them

Because FiLM stayed at identity, the honest reading is ambiguous, and both possibilities must be
stated:

1. **Biomarkers carry no information beyond the frozen representations.** This is consistent with
   Task-8B (G − E: +0.002039, p = 0.090, CI crossing zero) and Task 9 (I − H: −0.009578, p = 0.0049,
   i.e. actively harmful). Under this reading FiLM correctly learned to ignore them, and the bounded
   residual design did its job: it let the network leave the vascular representation alone.
2. **The modulation capacity was too tightly bounded to express whatever signal exists.** With
   `gamma = 1 + 0.10·tanh(·)` the maximum reachable gain is ±10 %, and the near-identity
   initialisation plus only 68,864 FiLM parameters and 6,203 training samples give the conditioner
   little pressure to move. A conditioner that starts at identity and is regularised with weight
   decay 1e-3 can stay at identity for the whole run.

This experiment cannot distinguish (1) from (2), and no claim should be made either way. What it
does establish is narrower and still useful: **with this bounded residual FiLM parameterisation,
biomarker conditioning of the vessel representation produced no measurable effect on any prediction,
and no AUC improvement over its matched control.**

The Brier/ECE improvement of J1 over J0 (§6) must therefore not be read as a biomarker effect, since
ablating every biomarker individually moves nothing. A second reason to distrust it: although J0 and
J1 share architecture, data, seed, optimizer and batch order, their fusion MLPs do **not** share a
bit-identical initialisation. In `Fusion.__init__` the FiLM modules are constructed before
`self.fuse`, so J1 consumes additional RNG draws and the fusion head starts from different weights.
Given that FiLM is empirically near-identity, an ordinary initialisation difference between two
otherwise identical 2.5 M-parameter heads is a sufficient explanation for a Brier gap of 0.0155.
This is a limitation of the matched design as executed and is recorded rather than papered over;
pinning the fusion-head initialisation identically (for example by re-seeding immediately before
building the fusion MLP) is the fix for any future replication. See §13.

## 10. Source breakdown (TEST, frozen predictions)

| source | n | J0 AUC | J0 bal acc | J0 F1 | J1 AUC | J1 bal acc | J1 F1 |
|---|---|---|---|---|---|---|---|
| plus | 889 | n/a | 0.816784 | 0.486464 | n/a | 0.820488 | 0.487862 |
| farfum_rop | 230 | 0.858916 | 0.642693 | 0.555712 | 0.860141 | 0.661924 | 0.594801 |
| farabi | 212 | 0.799322 | 0.528916 | 0.454313 | 0.799362 | 0.516799 | 0.445238 |

`plus` has no Pre_Plus, so 3-class multiclass AUC is undefined there and was not substituted. No
source-specific tuning was performed.

J0 and J1 are within 0.004 AUC of each other on every source — consistent with FiLM being a no-op.
Both are well below E and G on all three sources except on `plus` balanced accuracy, where the
weighted-CE minority-recall behaviour already described in §5 puts J0/J1 at 0.82 against 0.72 for
E/G while their `plus` macro F1 remains far worse (0.49 vs 0.73). Task 9's finding that biomarker
behaviour differs by source could not be reproduced as a FiLM effect because FiLM has no effect.

## 11. Overfitting diagnostics (§11)

| model | epoch | train loss | val loss | val AUC | val macro F1 |
|---|---|---|---|---|---|
| J0 | 1 | 0.4041 | **0.6866** | **0.91591** | 0.6712 |
| J0 | 2 | 0.2715 | 0.9169 | 0.90916 | 0.6660 |
| J0 | 3 | 0.2071 | 1.0452 | 0.90875 | 0.6285 |
| J0 | 4 | 0.1892 | 1.0577 | 0.91307 | 0.6786 |
| J0 | 5 | 0.1261 | 1.3266 | 0.90709 | 0.6623 |
| J0 | 6 | 0.1071 | 1.5765 | 0.90389 | 0.6298 |
| J0 | 7 | 0.0925 | 1.6595 | 0.89721 | 0.6225 |
| J0 | 8 | 0.0525 | 1.9065 | 0.90491 | 0.6584 |
| J0 | 9 | 0.0460 | 1.8711 | 0.91048 | 0.6691 |
| J1 | 1 | 0.3988 | **0.7145** | **0.91670** | 0.6644 |
| J1 | 2 | 0.2728 | 0.8900 | 0.90611 | 0.6493 |
| J1 | 3 | 0.2120 | 0.9464 | 0.90988 | 0.6498 |
| J1 | 4 | 0.1745 | 1.2670 | 0.90865 | 0.6536 |
| J1 | 5 | 0.1400 | 1.4423 | 0.91030 | 0.6440 |
| J1 | 6 | 0.1099 | 2.0002 | 0.90625 | 0.6165 |
| J1 | 7 | 0.0788 | 1.5874 | 0.90351 | 0.6318 |
| J1 | 8 | 0.0646 | 1.9487 | 0.90309 | 0.6382 |
| J1 | 9 | 0.0482 | 1.8764 | 0.91125 | 0.6713 |

* **Selected epoch**: 1 for both models.
* **Epoch of minimum validation loss**: 1 for both (J0 0.6866, J1 0.7145).
* **Did validation degrade while train loss kept improving?** Yes. Train loss falls monotonically to
  0.046 (J0) and 0.048 (J1) while validation loss climbs to 1.87 / 1.88 — a ~2.7× rise from the
  epoch-1 minimum. Validation AUC oscillates in the 0.897–0.916 band and never beats epoch 1.
* **Generalization gap** at the final epoch: J0 train 0.0460 vs val 1.8711 (≈ 41×); J1 train 0.0482
  vs val 1.8764 (≈ 39×).

This is the third consecutive task with the same failure mode. Even with both CNNs frozen into
precomputed features and only 2.5 M head parameters, a 6,203-sample training set overfits from the
second epoch. Freezing the encoders removed the Task-9 problem of catastrophic encoder drift but did
not remove the head-level overfitting: the head still has far more capacity than 6,203 samples
support. Learning curves: `learning_curves.png`.

## 12. Integrity

* Frozen RGB embedding, vessel embedding, E predictions and G predictions all re-hashed and matched
  against the Task-6 and Task-8 manifests before any model was built — all PASS.
* No CNN parameter entered the optimizer; no encoder was instantiated or executed.
* Task-6/7/8/9 artifacts were not modified by this task.
* E and G test predictions were reused read-only; they were not regenerated.
* The SHA manifest excludes `task10_progress.log` and `train.log` by construction (marked
  `EXCLUDED_APPEND_ONLY_LOG`) so that the Task-8/9 manifest-ordering problem is not repeated.
* Task-10 hashes: `J0_FROZEN_FUSION_CONTROL_selected.pth` `3e1ebd55cf0c6b1e…`,
  `J1_BIOMARKER_FILM_selected.pth` `330be575e07faa79…`,
  `test_predictions_J0_FROZEN_FUSION_CONTROL.csv` `09d070719719f06a…`,
  `test_predictions_J1_BIOMARKER_FILM.csv` `a1f0df33efa69a5c…`,
  `biomarker_scaler.json` `7e8d888feced3f34…`.

## 13. Interpretation, stated plainly

1. The primary Task-10 hypothesis — that biomarkers help when they condition the vascular
   representation rather than being appended — is **not supported**. J1 is not better than J0 on AUC.
2. FiLM learned essentially nothing: gamma within ±0.3 % of 1, |beta| ≈ 1.2 × 10⁻³, and per-biomarker
   ablation changes TEST probabilities by ≈2 × 10⁻⁵ with zero argmax flips. The network ignored the
   biomarkers.
3. Because of that, the J1 − J0 Brier/ECE improvement is not a biomarker effect and coincides with a
   non-identical fusion-head initialisation between the two models. It should not be reported as a
   finding.
4. The scalar biomarker panel has now failed to show incremental value in three regimes: simple
   concatenation (Task 8B, G − E, p = 0.090), inside a learned joint network (Task 9, I − H,
   p = 0.0049, harmful), and as conditioning of a frozen representation (Task 10, near-identity).
   On the present evidence the five scalar biomarkers add nothing to the frozen RGB + vessel
   representations.
5. Whether that is a property of the biomarkers or of the overly conservative modulation bound
   cannot be decided here. A future replication with a wider bound or an unbounded FiLM and a fixed
   initialization would separate the two; that is a new experiment and was not started.
6. The metric-family disagreement (§5) is a caution for the whole series: weighted CE on this cohort
   buys balanced accuracy at the cost of AUC, macro F1 and calibration.
7. No operational, screening, or clinical claim follows from any of these numbers.

## 14. Artifacts

`artifacts/task10_biomarker_film/`: J0 and J1 selected checkpoints · per-epoch checkpoints ·
training histories · selection metadata · `biomarker_scaler.json` · `film_config.json` ·
`task10_selection_frozen.json` · frozen TEST predictions for J0 and J1 · `metrics_summary.csv` ·
`paired_J1_minus_J0.csv` · `paired_J1_minus_E.csv` · `paired_J1_minus_G.csv` · `paired_all_10k.csv` ·
bootstrap distributions · `film_modulation_diagnostics.csv` · `biomarker_sensitivity.csv` ·
`source_breakdown.csv` · confusion matrices · `learning_curves.png` ·
`calibration_and_forest.png` · `artifact_sha256.json`.
