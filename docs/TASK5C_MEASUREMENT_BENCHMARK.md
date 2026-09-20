# Task 5C — measurement benchmark and geometry candidate

```
PRIMARY_CORE_FEATURE_N:                                10
PRIMARY_CORE_ROW_COMPLETENESS:                         0.9991   (8862 / 8870)
PRIMARY_CORE_MISSINGNESS_GATE:                         PASS

DISC_AUGMENTED_FEATURE_N:                              8

ACQUISITION_GEOMETRY_LABEL_ASSOCIATION:                PRESENT

HVDROPDB_ROLE:  EXTERNAL_DEVELOPMENT_MEASUREMENT_BENCHMARK

HVDRO_VESSEL_REFERENCE_N:                              100
HVDRO_DISC_REFERENCE_N:                                100
HVDRO_PAIRED_VESSEL_DISC_N:                            25

HVDRO_PAIRED_RETCAM_N:                                 25
HVDRO_PAIRED_NEO_N:                                    0

HVDRO_DD_NORMALIZED_END_TO_END_BENCHMARK:              PARTIALLY_AVAILABLE
                                                       (available on 25 RetCam images only;
                                                        Neo DEFERRED_TO_TARGET_DOMAIN_EXPERT_VALIDATION)

V1_RETCAM_DICE:                                        0.7540
V1_RETCAM_CLDICE:                                      0.8751
V1_NEO_DICE:                                           0.4889
V1_NEO_CLDICE:                                         0.5461

V1_CURRENT_DISC_RETCAM_DICE:                           0.9473
V1_CURRENT_DISC_NEO_DICE:                              0.9479
V1_CURRENT_DISC_VALIDITY_RETCAM:                       1.0000
V1_CURRENT_DISC_VALIDITY_NEO:                          1.0000

V1_PRIMARY_MEASUREMENT_AGREEMENT:                      SUMMARY
    vessel_density_fov      RetCam ICC 0.872, MAE 0.0192 | Neo ICC 0.774, MAE 0.0129
    skel_density_fov        RetCam ICC 0.933, MAE 0.0016 | Neo ICC 0.547, MAE 0.0051
    tort_geodesic_median    RetCam ICC 0.258, MAE 0.0063 | Neo ICC 0.218, MAE 0.0059
    tort_geodesic_p90       RetCam ICC 0.309, MAE 0.0271 | Neo ICC 0.248, MAE 0.0263
    width_shape_p90_over_p50 RetCam bias +0.334, MAE 0.347 | Neo bias +0.334, MAE 0.349
    n_branches              RetCam bias +10.0, MAE 22.8  | Neo bias -15.5, MAE 25.2

V2_SYNTHETIC_GEOMETRY:                                 FAIL

V2_HVDROPDB_EVALUATION_RUN:                            NO

V2_RETCAM_SEGMENTATION_NONINFERIORITY:                 NOT_RUN
V2_NEO_SEGMENTATION_NONINFERIORITY:                    NOT_RUN
V2_PRIMARY_MEASUREMENT_NONINFERIORITY:                 NOT_RUN

SEG_GEOMETRY_CANDIDATE_V2_STATUS:                      NOT_SUPPORTED

FINAL_SEGMENTATION_GENERATION_DECISION:                DEFERRED

TARGET_DOMAIN_EXPERT_VALIDATION_STILL_REQUIRED:        YES

TASK5C_STATUS:                                         COMPLETE
                                                       (Route B — V2 failed the
                                                        predeclared synthetic gate)
```

> **`TECHNICALLY VALIDATED AND EXTERNALLY BENCHMARKED`. NOT `CLINICALLY VALIDATED`.**
> HVDROPDB is development/benchmark data that this project has already used. Final
> target-domain validation still requires project-domain expert vessel masks, expert disc
> annotations and inter-expert reference.

---

## A. PRIMARY_CORE_V1 — verified from file

`configs/primary_core_v1_features.yaml`, **10 features**, all computable without a successful
optic-disc detection:

```
tort_geodesic_median      tort_geodesic_p90        vessel_density_fov
skel_density_fov          tort_geodesic_top3_mean  n_branches
width_shape_p90_over_p50  fractal_d0               fractal_d1
fractal_d2
```

Missingness: **8 rows of 8870** (0.0902 %). Seven features have zero missing; `fractal_d0/d1/d2`
have 8 each.

| grouping | share of rows with any core feature NaN |
|---|---|
| source | farabi 0.00071 · farfum_rop 0.00065 · plus 0.00101 |
| geometry | 1240×1240 0.0 · 1280×960 0.00072 · 1440×1080 0.00103 · 1600×1200 0.00064 · 640×480 0.00199 |
| label | normal 0.00093 · pre_plus 0.00107 · plus 0.00068 |

Corrected association tests, with the **missingness indicator** as the predictor:

| target | χ² | p |
|---|---|---|
| source | 0.24 | 0.886 |
| geometry | 5.68 | 0.224 |
| label | 0.12 | 0.942 |

`PRIMARY_CORE_MISSINGNESS_GATE = PASS`. Imputation is not required and must not be used.

`configs/disc_augmented_v1_features.yaml` — **8 features**, valid only when `disc_valid = 1`
(3772 of 8870). Reserved for disc-valid conditional, sensitivity and later target-domain
analysis. It is not a primary pooled matrix.

## B. Three separate concepts, never merged

| concept | status | evidence |
|---|---|---|
| `MISSINGNESS_SHORTCUT` | **NOT_DETECTED** for PRIMARY_CORE | 8/8870 missing; χ² p ≥ 0.224 |
| `ACQUISITION_GEOMETRY_LABEL_ASSOCIATION` | **PRESENT** | geometry → label macro AUC **0.6445** |
| `FEATURE_GEOMETRY_SENSITIVITY` | measured per feature | V1 vs V2 synthetic table; Task 5B sections E/G/Q |

The Task-5B `MISSINGNESS_SHORTCUT_STATUS = FAIL` headline based on 0.6445 is **withdrawn**. The
0.6445 value is retained only under `ACQUISITION_GEOMETRY_LABEL_ASSOCIATION`. It is not evidence
that geometry causes disease, and not evidence that a classifier necessarily uses geometry. It
was **not** removed, balanced or regressed out in Task 5C; it becomes a robustness target for
later disease-model evaluation.

## C. HVDROPDB role and reference pairing

`HVDROPDB = EXTERNAL DEVELOPMENT / MEASUREMENT BENCHMARK`, not untouched confirmatory external
validation.

Pairing was established from **image bytes only**. Both cohorts name their 50 images
`1.png … 50.png`, so filename matching would have claimed **50 false pairs**; that trap is
recorded and rejected in `scripts/task5c_pairing.py`.

| camera | vessel refs | disc refs | pixel-hash overlap |
|---|---|---|---|
| RetCam | 50 | 50 | **25** |
| Neo | 50 | 50 | **0** |
| total | 100 | 100 | **25** |

`HVDRO_DD_NORMALIZED_END_TO_END_BENCHMARK = PARTIALLY_AVAILABLE — AVAILABLE ON 25 RETCAM IMAGES
ONLY`. Neo is `DEFERRED_TO_TARGET_DOMAIN_EXPERT_VALIDATION`. No synthetic pairing was created.

**Documented gap:** the DD-normalised agreement itself was **not computed in this phase**. The
25-image paired RetCam subset is identified and verified; the computation is deferred. This is
stated rather than implied.

## D. Legacy evidence lineage

`artifacts/hvdro_evidence_lineage.csv` — `REUSED_VERIFIED 0`, `RECOMPUTED 2`,
`LEGACY_NOT_APPLICABLE 4`.

**No existing HVDROPDB artifact records the segmentation checkpoint sha256**, so none passes the
section N lineage gate and none may be cited as `SEG_CURRENT_V1` evidence. Everything was
recomputed.

Two classes of legacy material:

* `vessel_dice_summary.csv`, `width/width_summary.csv` — inference size, threshold and
  postprocessing match V1, but the checkpoint identity is absent, so lineage fails. Recomputed
  Dice reproduced the legacy RetCam value exactly (0.7540), which confirms the configuration but
  does not retroactively license the artifact.
* `disc_error_summary.csv` — permanently classified
  **`LEGACY_PSEUDO_DISC_NOT_CURRENT_EVIDENCE`**. Its `pseudo_radius_px` is a constant for every
  image (255.0 px for Neo, 60.0 px for RetCam), the signature of a fixed image fraction rather
  than a detector. The legacy figures of ≈7.1 and ≈7.5 true radii centre error and 2.35–2.84×
  radius ratio describe the pseudo-disc and **must not be quoted as performance of the current
  detector**. The artifacts are preserved, not overwritten.

## E. Current `SEG_CURRENT_V1` segmentation, recomputed

100 expert-vessel images, `results/hvdro_validation/v2/v1_segmentation_summary.csv`.

| camera | N | Dice | clDice | precision | recall |
|---|---|---|---|---|---|
| RetCam | 50 | **0.7540** | **0.8751** | 0.7085 | 0.8138 |
| Neo | 50 | **0.4889** | **0.5461** | 0.5060 | 0.4889 |
| pooled (secondary) | 100 | 0.6215 | 0.7106 | 0.6073 | 0.6513 |

Segmentation quality is **strongly camera-dependent**. Neo is roughly 26 Dice points below
RetCam.

## F. Current disc detector, recomputed

100 expert-disc images, `results/hvdro_validation/v2/disc_benchmark_summary.csv`.

| camera | N | validity | Dice | centre error (expert DD) | diameter ratio |
|---|---|---|---|---|---|
| RetCam | 50 | **1.0000** | **0.9473** | **0.0242** | 0.9922 |
| Neo | 50 | **1.0000** | **0.9479** | **0.0246** | 1.0213 |

**The current disc detector is accurate**, with a centre error of about 0.025 expert disc
diameters and a diameter ratio within 2 %. This directly overturns the legacy pseudo-disc
picture, which reported 7+ true radii of centre error. Consequence: disc-normalised features are
not limited by disc-detector accuracy on HVDROPDB; on the project cohort the limit is the 42.5 %
validity rate, which is a different problem (domain transfer, not geometry).

## G. `CLINICAL_MEASUREMENT_V1` disc-independent agreement, GOLD vs V1

`results/hvdro_validation/v2/v1_biomarker_agreement_summary.csv`. Every row carries its
evaluable N. ICC is ICC(2,1), two-way random, absolute agreement, single measurement.

| camera | feature | N | ICC(2,1) | 95 % CI | bias | MAE | median AE |
|---|---|---|---|---|---|---|---|
| RetCam | `vessel_density_fov` | 50 | 0.872 | [0.789, 0.995] | +0.0158 | 0.0192 | 0.0195 |
| RetCam | `skel_density_fov` | 50 | **0.933** | [0.886, 0.998] | −0.0016 | 0.0016 | 0.0013 |
| RetCam | `tort_geodesic_median` | 50 | 0.258 | [−0.022, 0.483] | +0.0012 | 0.0063 | 0.0040 |
| RetCam | `tort_geodesic_p90` | 50 | 0.309 | [0.037, 0.565] | −0.0060 | 0.0271 | 0.0184 |
| RetCam | `tort_geodesic_top3_mean` | 50 | 0.207 | [−0.049, 0.402] | −0.116 | 0.162 | 0.041 |
| RetCam | `width_shape_p90_over_p50` | 50 | 0.055 | [−0.047, 0.118] | **+0.334** | 0.347 | 0.310 |
| RetCam | `n_branches` | 50 | 0.738 | [0.584, 0.958] | +10.04 | 22.84 | 16.50 |
| Neo | `vessel_density_fov` | 50 | 0.774 | [0.633, 0.968] | +0.0001 | 0.0129 | 0.0106 |
| Neo | `skel_density_fov` | 50 | 0.547 | [0.391, 0.944] | −0.0050 | 0.0051 | 0.0053 |
| Neo | `tort_geodesic_median` | 50 | 0.218 | [0.023, 0.446] | +0.0051 | 0.0059 | 0.0051 |
| Neo | `tort_geodesic_p90` | 50 | 0.248 | [−0.008, 0.473] | +0.0131 | 0.0263 | 0.0183 |
| Neo | `tort_geodesic_top3_mean` | 50 | 0.248 | [−0.024, 0.469] | −0.0201 | 0.0680 | 0.0461 |
| Neo | `width_shape_p90_over_p50` | 50 | 0.136 | [0.011, 0.311] | **+0.334** | 0.349 | 0.296 |
| Neo | `n_branches` | 50 | 0.674 | [0.501, 0.937] | −15.46 | 25.22 | 17.50 |

**Reading.** The density family transfers well: `skel_density_fov` ICC 0.933 on RetCam with MAE
0.0016, and `vessel_density_fov` ICC 0.87/0.77 with MAE 0.019/0.013. Tortuosity shows
**near-zero bias and small absolute error but low ICC** (0.22–0.31); that is the
ICC-versus-MAE divergence the protocol warns about, and here it is caused by low between-image
variance in tortuosity, not by disagreement in level. `width_shape_p90_over_p50` carries a
systematic **+0.334 bias in both cameras** and `n_branches` a large, camera-signed error
(+10 RetCam, −15 Neo) consistent with its boundary-topology character. No arbitrary clinical
pass/fail cutoff is asserted. RetCam and Neo are never pooled for the primary reading.

## H. V2 geometry candidate — frozen, then failed

`SEG_GEOMETRY_CANDIDATE_V2`, contract `configs/seg_geometry_candidate_v2.yaml`, implementation
`src/segmentation/letterbox.py` (sha256 `2467c32e…b62324`). Exactly one policy:
**aspect-preserving letterbox** — `s = min(256/H, 256/W)`, isotropic resize to
`round(H·s)×round(W·s)`, zero pad to 256×256 with floor excess on top/left and the remaining
pixel on bottom/right, inverse by cropping the bands then `INTER_NEAREST` to native. Threshold
timing unchanged. No reflection, edge padding, alternative fill, alternative target size, crop
variant, patching, tiling or test-time augmentation was attempted. `infer_masks.py` was **not**
modified, so the `SEG_CURRENT_V1` contract still verifies.

The contract was written and hashed **before** the gate was run, and **no parameter was altered
afterwards** (section Y5).

### Synthetic gate result — `FAIL`

| geometry | \|width bias\| V1 | \|width bias\| V2 | \|tort bias\| V1 | \|tort bias\| V2 |
|---|---|---|---|---|
| 640×480 | 0.1161 | 0.1161 | 0.0022 | 0.0054 |
| 1280×960 | 0.1125 | 0.1586 | 0.0208 | 0.0496 |
| 1440×1080 | 0.1295 | 0.1295 | 0.0587 | 0.0776 |
| 1600×1200 | 0.1565 | 0.1645 | 0.0207 | 0.0628 |
| **median** | **0.1228** | **0.1440** | **0.0207** | **0.0562** |
| criterion ≤ 5 % | fail | fail | fail | fail |
| reduction vs V1 | — | **−17.3 %** | — | **−171.0 %** |

Density passes (V2 median |change| 0.0005 ≤ 1 %), but width and tortuosity both fail both
predeclared conditions. `V2_SYNTHETIC_GEOMETRY = FAIL`.

**Mechanism.** At the square geometry 1240×1240 the two paths are the same spatial map and still
carry a **−10.6 % width bias**. The 256×256 round trip therefore dominates, and an absolute ≤ 5 %
criterion is unreachable by *any* geometry policy at this input size — V1 fails it too. Isolated
against V1, letterboxing does not reduce the anisotropy term and **worsens** tortuosity by
+1.8 % to +4.1 %: V1 stretched the short axis **up** to 256 (640×480: ×1.875) while V2 downsamples
it isotropically (×2.500). The letterbox is geometrically correct and loses vascular detail
exactly where the anisotropy used to add resolution.

### Early stop applied

```
V2_HVDROPDB_EVALUATION_RUN = NO
V2_RETCAM_SEGMENTATION_NONINFERIORITY = NOT_RUN
V2_NEO_SEGMENTATION_NONINFERIORITY = NOT_RUN
V2_PRIMARY_MEASUREMENT_NONINFERIORITY = NOT_RUN
SEG_GEOMETRY_CANDIDATE_V2_STATUS = NOT_SUPPORTED
```

No HVDROPDB masks were generated under V2, and no alternative geometry candidate was tried
inside Task 5C. A later V3 would be a new development experiment with a new generation id and a
new protocol.

## I. Automatic adoption: none

`SEG_GEOMETRY_CANDIDATE_V2` was not renamed, not promoted, `SEG_CURRENT_V1` was not replaced,
the project feature table was not regenerated, and no disease classifier was trained.

`FINAL_SEGMENTATION_GENERATION_DECISION = DEFERRED`. A separate decision step must freeze either
`SEG_CURRENT_V1` or a new immutable generation. Only after that may the final feature table be
regenerated and frozen for disease-model training.

## J. Evidence tables

| # | table | path |
|---|---|---|
| 1 | PRIMARY_CORE_V1 feature list | `configs/primary_core_v1_features.yaml` |
| 2 | DISC_AUGMENTED_V1 feature list | `configs/disc_augmented_v1_features.yaml` |
| 3 | HVDROPDB reference-pairing census | `_private_audit/task5c_pairing.json` |
| 4 | legacy/current evidence lineage | `artifacts/hvdro_evidence_lineage.csv` |
| 5 | V1 segmentation metrics by camera | `results/hvdro_validation/v2/v1_segmentation_summary.csv` (+ per-image) |
| 6 | current disc-detector metrics by camera | `results/hvdro_validation/v2/disc_benchmark_summary.csv` (+ per-image) |
| 7 | V1 biomarker agreement by feature and camera | `results/hvdro_validation/v2/v1_biomarker_agreement_summary.csv` (+ per-image) |
| 8 | V1 vs V2 synthetic geometry | `artifacts/v1_v2_synthetic_geometry.csv` |
| 9 | V1 vs V2 paired segmentation differences | NOT PRODUCED — V2 early stop |
| 10 | V1 vs V2 measurement agreement | NOT PRODUCED — V2 early stop |
| — | missingness / permitted feature table | `_private_audit/task5c_allowed_features.csv`, `task5c_missingness.json` |
| — | V2 contract and gate result | `configs/seg_geometry_candidate_v2.yaml`, `_private_audit/task5c_v2_synthetic_gate.json` |

All statistical tables carry N, camera, feature, generation, reference type, estimate,
confidence interval where defined, and the evaluable count.

## K. Success gate — Route B

| # | requirement | status |
|---|---|---|
| 1–14 | PRIMARY_CORE verified; missingness corrected; 0.6445 withdrawn and re-labelled; geometry→label recorded separately; HVDROPDB role labelled; content-identity pairing; false filename pairing documented and rejected; legacy evidence classified; V1 segmentation evaluated on RetCam and Neo; current disc detector evaluated; CLINICAL_MEASUREMENT_V1 disc-independent biomarkers recomputed; RetCam and Neo reported separately; DD-normalised restricted to paired images; Neo DD-normalised not fabricated | ✓ |
| 15 | exactly one V2 policy implemented and frozen before evaluation | ✓ |
| 16 | V2 synthetic gate executed correctly | ✓ |
| 17 | `V2_SYNTHETIC_GEOMETRY = FAIL` | ✓ |
| 18 | `SEG_GEOMETRY_CANDIDATE_V2_STATUS = NOT_SUPPORTED` | ✓ |
| 19 | V2 HVDROPDB generation recorded as `NO` due to predeclared early stopping | ✓ |
| 20 | no alternative geometry candidate tried inside Task 5C | ✓ |
| 21 | `FINAL_SEGMENTATION_GENERATION_DECISION = DEFERRED` | ✓ |
| 22 | no disease classifier trained | ✓ |
| 23 | no historical result overwritten | ✓ |

```
TASK5C_STATUS = COMPLETE
```

## L. Limitations that survive Task 5C

* **HVDROPDB is development data**, already used by this project for checkpoint and threshold
  development. It is a benchmark, not confirmatory external validation.
* **Neo is weak.** Dice 0.489 and clDice 0.546. Any Neo measurement claim rests on that.
* **DD-normalised end-to-end benchmarking exists for 25 RetCam images only**, and was not
  computed in this phase. Neo has no paired reference at all.
* **The 256×256 model input, not the aspect ratio, is the dominant geometry loss** (−10.6 % width
  at a square geometry). The aspect-preserving candidate does not fix it and makes tortuosity
  worse. Fixing it would require a different input resolution, which is a new generation and a
  new protocol, not a geometry patch.
* **`ACQUISITION_GEOMETRY_LABEL_ASSOCIATION = PRESENT`** at AUC 0.6445 remains an unresolved
  robustness target.
* **Only 42.5 % of the project cohort has a valid disc**, with source-dependent survival.
* **No clinical validation.** Target-domain expert vessel masks, expert disc annotations,
  biomarker agreement and inter-expert reference are still required.
