# Fast examiner analysis — patient-level 3-fold sensitivity on FARFUM-RoP

Status: `FAST_EXAMINER_KFOLD = COMPLETE`
Classification: `POST_HOC_ROBUSTNESS_SENSITIVITY_SINGLE_SOURCE`
Scope: **patient-level 3-fold cross-validation on FARFUM-RoP only** (1,528 images / 68 patients).
Not cross-validation of the multisource cohort. No canonical split, artifact, prediction or number
was read, modified or replaced.

## Question answered

1. Could the weak scalar-biomarker result be an artefact of one fixed split?
2. Why should the biomarkers matter if they do not help?
3. Are the five scalar biomarkers redundant with each other?

## Decisions

```
K_FOLD_BIOMARKER_CONCLUSION = ROBUST_NULL
K_FOLD_VESSEL_CONCLUSION = NULL
BIOMARKER_REDUNDANCY = HIGH
FEATURE_SELECTION_CHANGED_CONCLUSION = NO
```

## Design

- one common de-duplicated cohort rebuilt from `data/splits/primary_complete_case_v2.csv`
  (`source == farfum_rop`): 1,528 images, 68 patients, classes 780 / 478 / 270, all five biomarkers
  complete, all frozen masks present;
- one deterministic patient-level 3-fold manifest (`StratifiedGroupKFold`, K = 3, seed 42, zero
  patient overlap), sha256 `cfa4ae82a98ff26da572340c4829abb663ec72e87b59d38de9000626d982fa4f`;
- per outer fold: RGB EfficientNet-B5 and vessel EfficientNet-B4 **retrained from ImageNet weights on
  outer-training patients only** (frozen Task-6 / Task-8 recipes, no hyper-parameter search,
  checkpoint selected on internal validation patients only), embeddings from that fold's
  checkpoints, then B / C / E / G fitted on outer-training rows only with the frozen shared XGBoost
  configuration;
- every image has exactly one held-out prediction; differences tested with a **patient-level**
  stratified bootstrap (10,000 replicates, seed 42), percentile 2.5 / 97.5, null-centred two-sided p.

## Results (pooled out-of-fold, N = 1,528)

| model | AUC | balanced acc | macro F1 | Brier | ECE |
|---|---|---|---|---|---|
| B (RGB embedding) | 0.7126 | 0.5685 | 0.5881 | 0.7347 | 0.3416 |
| C (RGB + 5 biomarkers) | 0.7129 | 0.5750 | 0.5952 | 0.7312 | 0.3369 |
| E (RGB + vessel embedding) | 0.7077 | 0.5691 | 0.5878 | 0.7393 | 0.3386 |
| G (RGB + vessel + 5 biomarkers) | 0.7094 | 0.5694 | 0.5875 | 0.7422 | 0.3416 |

| comparison | ΔAUC | 95% CI | p | crosses 0 |
|---|---|---|---|---|
| C − B | +0.000252 | [−0.003536, +0.003907] | 0.8891 | yes |
| E − B | −0.004878 | [−0.011221, +0.001549] | 0.1317 | yes |
| G − E | +0.001667 | [−0.001663, +0.004975] | 0.3205 | yes |

Per-fold AUC (fold 0 / 1 / 2): B 0.7161 / 0.8241 / 0.6706; C 0.7113 / 0.8277 / 0.6708;
E 0.7126 / 0.8153 / 0.6698; G 0.7095 / 0.8160 / 0.6758.

### Redundancy of the five FINAL_PRIMARY biomarkers

| pair | Pearson r |
|---|---|
| fractal_d1 – fractal_d2 | 0.995 |
| fractal_d0 – fractal_d1 | 0.975 |
| fractal_d0 – fractal_d2 | 0.956 |
| vessel_density_fov – skel_density_fov | 0.876 |

VIF: fractal_d1 366, fractal_d2 210, fractal_d0 42, vessel_density_fov 9.1, skel_density_fov 5.7.
A PCA retaining 95 % of the variance needs **one component in every fold**.
Correlation pruning at |r| ≥ 0.90 keeps 3 of 5 markers (`vessel_density_fov`, `skel_density_fov`,
`fractal_d0`) in every fold.

### Feature-selection sensitivity (fitted on training data only)

| variant | C − B | G − E |
|---|---|---|
| FS0 all five | +0.000252 | +0.001667 |
| FS1 |r| ≥ 0.90 pruning (3 features) | −0.000757 | +0.000746 |
| FS2 PCA-95 % (1 component) | +0.002396 | −0.001311 |

Both variants stay inside the paired 95 % CI of the primary estimate
(C − B [−0.0035, +0.0039], G − E [−0.0017, +0.0050]).

## Interpretation

- The five scalar biomarkers add nothing measurable, in either direction, under patient-level
  cross-validation (C − B and G − E centred on zero, both CIs crossing zero) — the same conclusion
  as the locked canonical split, so the earlier result is not a split artefact.
- The redundancy audit explains part of the mechanism: the three fractal dimensions are
  near-collinear (r ≥ 0.956, PCA-95 % = 1 component), so the panel is effectively one fractal axis
  plus two density measures, not five independent measurements.
- The spatial vessel embedding keeps its canonical in-distribution increment (+0.007938 on the
  locked multi-source test) and its cross-source held-out increment (+0.022242 on the FARFUM LOSO
  fold), but **that increment is not detectable in this smaller single-source in-domain regime**
  (E − B −0.004878, CI crossing zero). With 771 training images per fold the encoders are much
  weaker (B 0.7126 vs 0.9249 canonical), which is a limit of this sensitivity design, not a
  refutation of the canonical comparison; the dominant constraint remains generalisation.

## Artifacts

Produced on `moniaz@100.115.180.59` (Apple M2 Ultra, MPS) under
`~/niki/results/fast_examiner/` (cohort, manifest + hash, OOF predictions, metrics, bootstrap,
redundancy, feature sensitivity, figures, per-fold selection histories, writeups) with the scripts in
`~/niki/_fast_examiner/` — versioned copies of the scripts are in `scripts/fast_examiner/`.
`results/` is git-ignored in this repository by design; a local copy of every output (except the
21 MB per-fold embedding bundles) lives outside the repository, and this document carries the
reporting reference.

Reproduce with `s02 → s04 (fold 0,1,2) → s05 → s06 → s07`. Compute used: ≈ 38 min per outer fold
(RGB + vessel training, embedding extraction, XGBoost) plus ≈ 17 min of pooled analysis.
