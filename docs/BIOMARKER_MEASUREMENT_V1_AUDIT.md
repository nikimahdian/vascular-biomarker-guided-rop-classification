# Biomarker measurement V1 audit — Task 5B

```
MEASUREMENT_VERSION:              CLINICAL_MEASUREMENT_V1

TOTAL_FEATURES_MEASURED:          78 columns = 47 predictor features + 22 QC + 9 metadata
                                  (69 contract rows: 47 features + 22 QC)

PRIMARY_ALLOWED_N:                 4
SECONDARY_ALLOWED_N:              14
EXPLORATORY_ONLY_N:               27
FORBIDDEN_N:                      24
                                  (sums to 69; the 65 historical contract rows are
                                   kept separately as HISTORICAL_NOT_ADMITTED)

TORTUOSITY_INVARIANCE:            PASS

WIDTH_SCALE_INVARIANCE:           PASS

FOV_DENSITY_INVARIANCE:           PASS

ROI_POLICY:                       PASS

AV_STATUS:                        EXPLORATORY

MISSINGNESS_SHORTCUT_STATUS:      FAIL

ASPECT_RATIO_MEASUREMENT_RISK:    FEATURE_DEPENDENT

CLASSIFIER_FEATURE_N:             18

FEATURE_TABLE_SHA256:             db123ac5f663f4925ac9fff52d204bede85963966794062d1855e315a4e38ffc

TASK5B_STATUS:                    COMPLETE
```

> **`MEASUREMENT IMPLEMENTATION TECHNICALLY VALIDATED`. This is NOT `CLINICALLY VALIDATED`.**
> No expert mask, no expert artery/vein label and no clinical endpoint was used anywhere in this
> task. Clinical validation is a later task.

---

## A. Hard rules respected

No branch was trained, no LOSO was run, no threshold was tuned, no feature was selected using a
label or an AUC, `SEG_CURRENT_V1` was not modified, the canonical 8870 cohort and Pilot 30 /
Final 90 were not touched, and no historical feature table was overwritten.

Every admission status in this document comes from a **measurement** property — scale, padding,
rotation, ROI coverage, definition validity — never from predictive performance.

---

## B. Measurement version and frozen definitions

`CLINICAL_MEASUREMENT_V1`, contract `configs/clinical_measurement_v1.yaml`, implementation
`src/biomarker/clinical_measurement_v1.py`, geometry primitives
`src/biomarker/geometry_core.py`.

| field | value |
|---|---|
| segmentation generation | `SEG_CURRENT_V1` |
| checkpoint sha256 | `c373f53813ee60b89651a04a98bc5f1d6bc60a5a50f45c4e10f42475650d5374` |
| mask mapping | `data/masks/mask_manifest_canonical_v1.csv`, sha256 `b3bc6538026d6ad0fc1342043e5ca2d6de56662233bf6d0709d9d875f24b21cd` |
| working resolution | long side 512 px, one fixed grid for every image |
| population | canonical 8870, fingerprint `0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8` |
| output | `data/features/clinical_measurement_v1.csv`, 8870 × 78 |

The contract records, for every feature: mathematical definition, unit, ROI definition, and the
five dependency flags. No scientifically relevant default is implicit.

---

## C. Feature admission states

Exactly one status per feature, in `configs/feature_contract.csv` (134 rows, 19 columns) and in
the measurement contract.

| status | n | meaning |
|---|---|---|
| `PRIMARY_ALLOWED` | 4 | technically validated, clinically interpretable, acceptable invariance |
| `SECONDARY_ALLOWED` | 14 | technically valid, more assumption-dependent |
| `EXPLORATORY_ONLY` | 27 | reportable, cannot support primary claims |
| `FORBIDDEN_FROM_FINAL_CLASSIFIER` | 24 | acquisition shortcut, metadata proxy, invalid definition, detector-status shortcut, dead feature, unresolved defect |

`PRIMARY_ALLOWED`: `tort_geodesic_median`, `tort_geodesic_p90`, `vessel_density_fov`,
`skel_density_fov`.

---

## D. Tortuosity — frozen definition, synthetic correctness

**Frozen definition.** Skeleton → remove junctions → per branch: order the branch pixels along the
path, smooth their coordinates with a 5-pixel moving average, arc = polyline length of the smoothed
path, chord = Euclidean distance between its two ends, `T = arc/chord`. Branches shorter than 8 px
are excluded, never scored.

**Why the smoothing.** The raw 8-connected geodesic (orthogonal 1, diagonal √2) is exact on the
pixel graph, but a rasterised straight line is a staircase whose length exceeds its chord. The
smoothing removes the staircase while preserving curvature.

| case | expected | raw geodesic | **smoothed** | historical pixel-count |
|---|---|---|---|---|
| horizontal line | 1.0000 | 1.0000 | **1.0000** | 1.0034 |
| vertical line | 1.0000 | 1.0000 | **1.0000** | 1.0034 |
| 15° | 1.0000 | 1.0731 | **1.0039** | 0.9694 |
| 30° | 1.0000 | 1.0727 | **1.0012** | 0.8681 |
| 45° | 1.0000 | 1.0000 | **1.0000** | 0.7105 |
| 60° | 1.0000 | 1.0727 | **1.0014** | 0.8681 |
| 75° | 1.0000 | 1.0732 | **1.0038** | 0.9692 |
| arc 30° | 1.0115 | 1.0552 | **1.0141** | 1.0222 |
| arc 60° | 1.0472 | 1.1105 | **1.0476** | 1.0111 |
| arc 90° | 1.1107 | 1.1663 | **1.1030** | 1.0078 |
| arc 120° | 1.2092 | 1.2718 | **1.2044** | 1.0644 |
| arc 180° | 1.5708 | 1.6583 | **1.5670** | 1.4191 |
| sine amp20 per160 | 1.1275 | 1.1951 | **1.1282** | 1.0011 |
| sine amp40 per160 | 1.4209 | 1.4839 | **1.4142** | 1.1695 |
| sine amp60 per200 | 1.5111 | 1.5999 | **1.5145** | 1.2760 |
| Y junction | n/a | 1.0610 | **1.0043** | 0.8399 |
| short fragment (5 px) | excluded | n=0 | **n=0** | n=0 |

| metric | smoothed | raw geodesic | historical |
|---|---|---|---|
| straight-line orientation dispersion (max−min over 7 orientations) | **0.0039** | 0.0732 | 0.2929 |
| max abs error vs analytic arc/sine | **0.0077** | 0.0887 | 0.2489 |
| all values ≥ 1 | yes | yes | **no** |

**`TORTUOSITY_INVARIANCE = PASS`.** The frozen definition is **75× better** than the historical
one on orientation dispersion and reproduces analytic arcs to better than 0.8 %.

**Rejected:** the historical clinical definition (branch pixel count / hull chord) scores a
straight vessel 1.003 at 0° and **0.710 at 45°** — it is orientation-dependent and is retained
only as `tort_pixelcount_median_provenance`, marked `EXPLORATORY_ONLY`.

---

## E. Width — end-to-end, with the quantisation floor

**Estimator selection.** Three candidate estimators were measured on synthetic strips, 5 widths ×
7 orientations:

| estimator | median \|rel err\| | max orientation dispersion |
|---|---|---|
| **`2 × EDT`** | **0.0124** | 1.858 px |
| `2 × EDT − 1` | 0.0686 | 1.858 px |
| `area / skeleton length` | 0.0574 | 12.431 px |

`2 × EDT` sampled on the skeleton is frozen. Sampling every vessel pixel instead would be
edge-weighted and biased low.

**Scale invariance, vessel and disc scaled together:**

| scale | true w px | true w/DD | raw w/DD | rel err |
|---|---|---|---|---|
| 1× | 9 | 0.22500 | 0.25000 | 0.1111 |
| 2× | 18 | 0.22500 | 0.22500 | 0.0000 |
| 3× | 36 | 0.22500 | 0.22500 | 0.0000 |

raw width/DD spread across scales = **0.025**, against a pre-specified criterion of 0.03 whose
geometric justification is that 1 px of digital width quantisation on the smallest plausible disc
diameter (60 px) is already 1/(60) = 1.7 %.

**Quantisation floor from 256×256 inference.** `SEG_CURRENT_V1` resamples to 256×256, so one
binary step becomes `ceil(native_long_side/256)` pixels after NEAREST upsampling:

| native long side | upsample factor | width floor at DD 60 px | at DD 180 px |
|---|---|---|---|
| 480 | 1.88 | 0.031 DD | 0.010 DD |
| 960 | 3.75 | 0.063 DD | 0.021 DD |
| 1280 | 5.00 | 0.083 DD | 0.028 DD |
| 1600 | 6.25 | **0.104 DD** | 0.035 DD |

**`WIDTH_SCALE_INVARIANCE = PASS`** with a stated floor. **No sub-pixel precision is claimed:**
the pipeline supports width/DD to about 2.5 % across a 4× scale range and no better than about
1 px across orientations. That is why `width_*_dd` is `SECONDARY_ALLOWED`, not `PRIMARY_ALLOWED`.

---

## F. Disc-normalised width validity

`disc_valid` is a hard gate. There is **no image-centre and no constant-radius fallback** in the
code, and the contract asserts `fallback: NONE`.

| perturbation | median width_p90_dd | median rel change | p95 \|rel change\| |
|---|---|---|---|
| DD +2 % | 0.250615 | **−0.0196** | 0.0196 |
| DD +5 % | 0.243455 | **−0.0476** | 0.0476 |
| DD +10 % | 0.232389 | **−0.0909** | 0.0909 |
| centre +2 % | 0.255628 | 0.0000 | 0.0000 |
| centre +5 % | 0.255628 | 0.0000 | 0.0000 |
| centre +10 % | 0.255628 | 0.0000 | 0.0000 |

Measured on 120 deterministic disc-valid images. A DD error scales width/DD by exactly `1/(1+ε)`
mechanically, and the medians confirm it to three digits: **a 5 % disc-diameter error biases
disc-normalised caliber by −4.8 %.** Centre error does not move a whole-skeleton percentile,
which is expected and is not a validation of the centre.

**Disc validity is scarce and source-dependent:**

| source | disc_valid = 0 | disc_valid = 1 |
|---|---|---|
| farabi | 0.5113 | 0.4887 |
| farfum_rop | 0.4670 | 0.5330 |
| plus | 0.6176 | 0.3824 |
| **overall** | **0.5747** | **0.4253** |

Only **3772 of 8870 images (42.5 %)** have a valid disc. Every disc-dependent feature is NaN for
the other 5098. `disc_valid` itself is a pure acquisition proxy and is
`FORBIDDEN_FROM_FINAL_CLASSIFIER`.

---

## G. Whole-frame vs FOV-normalised density

| perturbation | whole-frame | rel change | FOV-normalised | rel change |
|---|---|---|---|---|
| +50 px black border | 0.01861317 | −0.3476 | 0.02853175 | **0.00000000** |
| +100 px | 0.01309313 | −0.5411 | 0.02853175 | **0.00000000** |
| +200 px | 0.00748513 | −0.7377 | 0.02853175 | **0.00000000** |
| +300 px | 0.00483756 | −0.8304 | 0.02853175 | **0.00000000** |
| rectangular +200 px on x | 0.01461382 | −0.4878 | 0.02853175 | **0.00000000** |
| centred crop −60 px/side | 0.05592222 | +0.9600 | 0.05592222 | +0.9600 |

**`FOV_DENSITY_INVARIANCE = PASS`.** The FOV-normalised density is **exactly** padding-invariant
(relative change 0.00000000 in every padding case, including rectangular), while whole-frame
density moves by up to −83 %.

The centred-crop row is not an invariance failure: cropping removes vessel, so the density of the
remaining region legitimately changes. It is precisely why the ROI coverage policy in section M
exists.

**New names, no silent replacement.** The historical `vessel_density` is retained under the
explicit name `vessel_density_wholeframe` and marked `EXPLORATORY_ONLY`. The corrected feature is
a new column, `vessel_density_fov`.

---

## H. FOV mask correctness

| check | result |
|---|---|
| `fov_valid` | **8825 / 8870 (99.49 %)** |
| failures | 45 — `fragmented_bright_region` 27, `coverage_below_15pct` 18 |
| synth. circular retina, coverage error | 0.0000 |
| synth. rectangular FOV, coverage error | 0.0000 |
| failure by geometry | worst 1240×1240 at 1.14 % |
| failure by source | worst farfum_rop at 0.72 % |

Deterministic QC quantities are recorded for **every** image — `fov_coverage_fraction`,
`fov_n_components`, `fov_border_contact`, `fov_centroid_offset`, `fov_failure_reason` — so a failed
FOV is visible rather than silent. All five are QC-only and `FORBIDDEN_FROM_FINAL_CLASSIFIER`.

---

## I. Area, length and absolute counts — scale dependence

Synthetic vascular tree at 1× / 2× / 4×:

| feature | 1× | 2× | 4× | ratio 4×/1× | verdict |
|---|---|---|---|---|---|
| `vessel_pixels` | 1129 | 4965 | 18851 | **16.70** | RAW_SCALE_DEPENDENT |
| `skeleton_px` | 261 | 525 | 1042 | **3.99** | RAW_SCALE_DEPENDENT |
| `n_intersections` | 7 | 7 | 8 | 1.14 | RAW_SCALE_DEPENDENT (marginal) |
| `area_fraction` | — | — | — | 1.04 | scale-stable but padding-sensitive |
| `n_startpoints` | 4 | 4 | 4 | 1.00 | scale-stable, boundary artefact (J) |
| `n_endpoints` | 4 | 4 | 4 | 1.00 | duplicate of the above |

`area`, `overall_length` and every absolute vessel-pixel or skeleton-pixel count are
`RAW_SCALE_DEPENDENT` and are `EXPLORATORY_ONLY`. The dimensionless normalisations
(`vessel_density_fov`, `skel_density_fov`) are the admitted forms.

---

## J. Topology — vascular or ROI/boundary topology?

A startpoint in this implementation is a skeleton pixel with exactly two 8-neighbours, i.e. a line
**end**. Every vessel that runs off the image or FOV boundary terminates there and manufactures
one.

| observation | value |
|---|---|
| `n_endpoints` vs `n_startpoints` | **identical column** (they are the same expression) |
| `n_endpoints_fov` vs `n_startpoints_fov` | **identical column** |
| `n_intersections` under 1× → 2× → 4× | 7 → 19 → 18 |
| `n_intersections` vs `n_branches` | r = 0.9937 |

**Conclusion: neither count is a biological vascular-topology biomarker.** They are ROI/boundary
topology and raster topology. All six variants are `EXPLORATORY_ONLY`.

---

## K. Fractal features

Same vascular pattern under perturbation, relative spread against the base value:

| feature | base | relative spread | worst perturbation |
|---|---|---|---|
| `fractal_d0` | 1.07415 | **0.1177** | central crop (+6.8 %) |
| `fractal_d1` | 1.02524 | **0.0819** | central crop (+3.7 %) |
| `fractal_d2` | 1.01155 | **0.0789** | central crop (+3.3 %) |
| `singularity_length` | 0.95168 | **1.2429** | rotation 30° (**+110 %**) |

`fractal_d0/d1/d2` are `SECONDARY_ALLOWED` with their sensitivity recorded as a known limitation.
`singularity_length` is `FORBIDDEN_FROM_FINAL_CLASSIFIER`: a 110 % change under a pure rotation is
an unresolved measurement defect. **Reproducibility is not validity** — the values reproduce
bit-for-bit (Task 5A) and are still too unstable to admit.

---

## L. Regional / quadrant features

Definition recorded in the contract: **four image-frame angular sectors about the disc centre**,
restricted to the posterior pole (r < 6 DD). Coordinate origin = disc centre. Units = fraction of
vessel pixels inside the sector.

| perturbation | ne | nw | sw | se |
|---|---|---|---|---|
| base | 0.2488 | 0.2512 | 0.2500 | 0.2500 |
| translate −40 px | **0.0890** | 0.2104 | **0.4900** | 0.2100 |
| +100 px padding | 0.1104 | 0.1118 | 0.1111 | 0.1111 |
| rotate 45° | 0.2480 | 0.2525 | 0.2503 | 0.2503 |

Max spread 0.379. These are **image-frame geometry, not anatomy**. They are **not** ICROP
quadrants: ICROP quadrants are defined relative to the disc-macula axis and no such axis is
detected here. The contract states `anatomical: false` and the columns are named `frame_sector_*`.
`EXPLORATORY_ONLY`, translation- and rotation-invariant = false.

---

## M. ROI coverage policy

**Pre-specified minimum coverage = 0.60**, justified geometrically: a region with less than 60 % of
its area inside the detected FOV is more than 40 % truncated, so its density is a biased estimate
whose bias direction depends on where the truncation falls. The threshold was not chosen from any
label performance.

Coverage is only defined for the 3772 disc-valid images.

| region | n | min | p05 | median | below 0.60 |
|---|---|---|---|---|---|
| `roi_coverage_ring_0_2dd` | 3772 | 0.0000 | 0.6185 | 0.9981 | 161 |
| `roi_coverage_ring_2_3dd` | 3772 | 0.0000 | 0.4867 | 0.8346 | 439 |
| `roi_coverage_ring_3_6dd` | 3772 | 0.0000 | 0.2635 | 0.7045 | **1207 (32 %)** |
| `roi_coverage_annulus` | 3772 | 0.0000 | 0.6029 | 0.9980 | 187 |
| `roi_coverage_pole` | 3772 | 0.0038 | 0.4064 | 0.7524 | 912 |

Median `roi_coverage_pole` by source — farabi 0.8685, farfum_rop 0.7470, plus 0.6961.
By geometry — **1240×1240 0.5754** (below the threshold), 1600×1200 0.7488, 640×480 0.8187,
1280×960 0.8690, 1440×1080 0.9301. By split — train 0.7703, val 0.7206, test 0.6914.

**`ROI_POLICY = PASS`:** below-threshold regions yield NaN, never a silently biased number.
**But coverage is geometry-dependent**, so every ring feature carries geometry-dependent
missingness. That is recorded, not hidden, and it is the main input to section O.

---

## N. Artery / vein heuristic

### A/V PERTURBATION STRESS TEST — Task 5B-N closure

```
AV_STRESS_TEST_EXECUTED:              YES (attempted; own control FAILED, flip rates invalid)

AV_SAMPLE_N:                          360
BASELINE_BRANCH_N:                    27596

BRIGHTNESS_MAX_FLIP_RATE:             NOT_REPORTED  (control failed)
CONTRAST_MAX_FLIP_RATE:               NOT_REPORTED  (control failed)
GAMMA_MAX_FLIP_RATE:                  NOT_REPORTED  (control failed)
WHITE_BALANCE_MAX_FLIP_RATE:          NOT_REPORTED  (control failed)
ILLUMINATION_GRADIENT_MAX_FLIP_RATE:  NOT_REPORTED  (control failed)

OVERALL_MAX_BRANCH_FLIP_RATE:         NOT_REPORTED  (control failed)

AV_BALANCE_BRANCH_MEDIAN:             NOT_AVAILABLE (the frozen module emits no branch-count fraction)
AV_BALANCE_LENGTH_MEDIAN:             0.5402        (frozen table, n = 8865)
AV_BALANCE_PIXEL_MEDIAN:              NOT_AVAILABLE (the frozen module emits no vessel-pixel fraction)

AV_STATUS:                            EXPLORATORY_ONLY

EXPERT_AV_VALIDATION_AVAILABLE:       NO

DOCUMENTATION_OVERCLAIM_CORRECTED:    YES

TASK5B_N_CLOSURE:                     INCOMPLETE
```

> **CORRECTION (Task 5B-N closure).** This section previously opened with the words
> *"Stress-tested and not validated"*. That was **unsupported**: the perturbation stress tests
> requested in Task 5B section N were never executed. The wording is corrected here before any
> new experiment is run, and the section is replaced by the measured result once the closure task
> completes. The unsupported claim is recorded rather than silently removed.

**A/V heuristic remains unvalidated; perturbation stress testing ATTEMPTED AND INVALID.**

A concrete attempt was made to run the requested perturbation battery (360-image stratified
sample, 26 perturbations, 27,596 baseline branches). **The attempt failed its own control and its
numbers must not be used.**

The harness replicates the A/V assignment from `src/biomarker/clinical_measurement_v1.py::measure`
because the frozen module does not expose per-branch labels. A control compared the replicated
artery fraction against `measure()`'s own `a_frac` on the same image:

```
CONTROL max |replicated a_frac - measure() a_frac| = 0.4998
```

The replication returns `a_frac = 1.0` on all 360 images while the frozen implementation returns
approximately 0.5, so the replica is **not faithful** and every derived quantity, including the
branch flip rates, is meaningless. **The closure gate was NOT passed.**

Raw attempt, explicitly **not** usable as evidence:

| field | value |
|---|---|
| `AV_SAMPLE_N` | 360 (plus 152, farabi 118, farfum_rop 90) |
| geometries | 1600×1200 118, 1280×960 90, 640×480 90, 1240×1240 62; **1440×1080 absent** (data-limited strata) |
| `BASELINE_BRANCH_N` | 27,596 |
| `CONTROL_MAX_ABS_DIFF_A_FRAC` | **0.4998 → FAIL** |
| reported flip rates | suppressed, invalid |

The one code-level observation that does **not** depend on the broken replica: `measure()` computes

```
is_a = vals >= thr        # thr = vals[order][cut], the length-weighted median
```

and because `gc = green - median_filter(green, 31)` leaves a large mass of near-identical values
on short branches, this comparison is tie-dominated, so the split is decided by tie ordering
rather than by intensity. **Whether that also degrades the frozen implementation is not
established** and needs a correct harness.

The method is a per-**branch** length-weighted median split of corrected green intensity. By
construction it splits the branch population into two groups of equal total length, so
**`a_frac` is forced to ≈ 0.5 by construction** — it is mechanically balanced in branch length,
not in branch count, branch width or vessel pixels. That statement is a property of the code and
needs no experiment.

**Valid section-H evidence, computed from the frozen table and independent of the broken
replica** (`data/features/clinical_measurement_v1.csv`, `a_frac`, n = 8865):

| statistic | value |
|---|---|
| median | **0.5402** |
| IQR | 0.0646 |
| p05 – p95 | **0.5025 – 0.6797** |
| minimum | **0.5001** |
| images in 0.45 – 0.55 | 0.5768 |
| images in 0.40 – 0.60 | 0.8205 |

**The minimum never falls below 0.5001.** The artery fraction is bounded below by one half by
construction, and 57.7 % of images sit inside 0.45–0.55. That is the mechanical 50/50 signature,
measured on the frozen implementation itself. Independently, `a_width_p90_px` has median 8.4827
against `v_width_p90_px` median 8.4853, giving a median `av_width_ratio_p90` of **exactly
1.0000** — a second, independent sign that the two groups are constructed to be balanced rather
than discovered.

Since the frozen module emits no branch-count or vessel-pixel artery fraction, the corresponding
balance statistics cannot be computed without changing the module, which this task forbids.
`AV_BALANCE_BRANCH_MEDIAN` and `AV_BALANCE_PIXEL_MEDIAN` are therefore `NOT_AVAILABLE`, not zero.

**A/V remains `EXPLORATORY_ONLY`.** No expert A/V reference labels exist, no acceptance threshold
was invented, and nothing here validates biological correctness of the labels.

### Task 5B-N2 — production-faithful instrumentation: gate PASSED under a predeclared tolerance, battery executed

The first closure attempt was invalidated by a **scaling bug in the replica**, not by any property
of the heuristic: the replica passed the 0–255 float image through `np.clip(rgb, 0, 1)`, which
destroyed the picture, drove `gc` to zero everywhere, made every branch value tie, and produced
`a_frac = 1.0` instead of ≈0.5. The earlier speculation in this document that the production rule
is "tie-dominated" was a consequence of that bug and is **withdrawn**.

A production-faithful second attempt was then built: the inline A/V block was moved, unedited,
into a module-level helper `assign_av_branches(gc, lab, nlab, dist)` that returns the branch
table, and `measure()` consumes the same helper, so exactly one implementation exists and no
replica supplies any number.

```
pre-refactor  sha256          : 3c1c10773ed28a187df373412e9de62309f30dd2896ab602b36bea6e44b29e66
attempt-2 refactor sha256     : 171fe7ab1bd4edc9942d60b80043d45ee39ca79ec301d970182bdad3154b490b
attempt-3 refactor sha256     : 6aea0d6856d75e205f88201cfc20311a4d13acfdac839aa5ac8bfac3a1f13542
```

The attempt-3 refactor is a fresh re-application of the same move, **not a byte-copy of the
attempt-2 file**; the frozen baseline is `3c1c1077…` in both cases, and every gate number below is
a comparison against that frozen baseline, never against attempt 2.

**The instrumentation itself is faithful.** On all 8,870 canonical images the artery fraction
reconstructed directly from the helper's own `is_a` labels equals `measure()`'s `a_frac` exactly:

```
DIAGNOSTIC_CONTROL_MAX_DELTA = 0.000e+00
```

**Attempt 2's gate failed.** Its record is kept here as history, because the tolerance used in
attempt 3 was predeclared only after this failure was seen and left untouched:

| column | NaN pattern | max abs delta | verdict |
|---|---|---|---|
| `branch_px_work` | same | 0.000e+00 | IDENTICAL |
| `n_branches` | same | 0.000e+00 | IDENTICAL |
| `a_frac` | same | 1.110e-16 | differs |
| `a_width_p90_px` | same | 1.776e-15 | differs |
| `v_width_p90_px` | same | 3.553e-15 | differs |
| `av_width_ratio_p90` | same | 2.220e-16 | differs |

```
AV_REFACTOR_CANONICAL_N                 : 8870
AV_FEATURE_COLUMNS_CHECKED              : 6
AV_ROWS_EXACT_OR_NUMERICALLY_EQUIVALENT : 0
AV_ROWS_DIFFERENT                       : 4
AV_MAX_ABS_DELTA                        : 3.553e-15
AV_FULL_BEHAVIOR_EQUIVALENCE            : FAIL  -> STOP
```

The deltas are 1–16 ULP of float64, and Task 5B-N2 section C does permit "machine-level numerical
equivalence with explicit maximum difference". **That allowance was deliberately not invoked in
attempt 2**, because the predeclared criterion for that run was exact equality and the stopping
rule was keyed on `AV_ROWS_DIFFERENT != 0`. Relaxing the criterion after seeing the number would be
post-hoc threshold movement, which the protocol forbids. Attempt 2 therefore stopped, the
perturbation battery was **not** executed, and the refactor was **not** committed. The tolerance was
instead **predeclared for the next run, before that run was started** — see attempt 3 below.

Consequences, applied literally:

* production code is unchanged; the module on the analysis host was restored to the frozen sha
  `3c1c10773ed28a187df373412e9de62309f30dd2896ab602b36bea6e44b29e66` and verified;
* `clinical_measurement_v1.py` is **not** in the refactored state anywhere;
* no flip rate, no feature-stability figure and no tie statistic from either attempt is reported;
* `TASK5B_N2_CLOSURE = INCOMPLETE`.

All four consequences held at the end of attempt 2 and are recorded as such. Attempt 3 superseded
the first three: production code now carries the refactor, flip rates and stability figures are
reported, and the closure is complete.

Attempt 2's closure block, exactly as printed by that run:

```
AV_STRESS_TEST_EXECUTED:              YES (two attempts; neither produced usable flip rates)
AV_PRODUCTION_INSTRUMENTATION:        PASS  (helper control delta = 0 on 8,870 images)
AV_FULL_BEHAVIOR_EQUIVALENCE:         FAIL  (predeclared exact criterion; deltas 1e-16..3.6e-15)
AV_SAMPLE_N:                          360   (drawn, battery not executed)
BASELINE_BRANCH_N:                    NOT_MEASURED
DIAGNOSTIC_CONTROL_MAX_DELTA:         0.000e+00
BRIGHTNESS_MAX_FLIP_RATE:             NOT_REPORTED
CONTRAST_MAX_FLIP_RATE:               NOT_REPORTED
GAMMA_MAX_FLIP_RATE:                  NOT_REPORTED
WHITE_BALANCE_MAX_FLIP_RATE:          NOT_REPORTED
ILLUMINATION_GRADIENT_MAX_FLIP_RATE:  NOT_REPORTED
OVERALL_MAX_BRANCH_FLIP_RATE:         NOT_REPORTED
TIE_AT_THRESHOLD_MEDIAN_FRACTION:     NOT_REPORTED
AV_BALANCE_BRANCH_MEDIAN:             NOT_AVAILABLE
AV_BALANCE_LENGTH_MEDIAN:             0.5402   (frozen table, n = 8865, valid)
AV_BALANCE_PIXEL_MEDIAN:              NOT_AVAILABLE
AV_STATUS:                            EXPLORATORY_ONLY
EXPERT_AV_VALIDATION_AVAILABLE:       NO
DOCUMENTATION_OVERCLAIM_CORRECTED:    YES
FAILED_REPLICA_RESULTS_USED:          NO
TASK5B_N2_CLOSURE:                    INCOMPLETE
```

#### Attempt 3 — tolerance predeclared before the run, gate PASSED, battery executed

The criterion was locked in `scripts/task5b_n2.py` **before starting the third execution** and was
not edited afterwards:

```
PREDECLARED_ATOL = 1e-12
EXACT_FIELDS     = ["branch_px_work", "n_branches"]     # delta must be exactly 0
FLOAT_FIELDS     = ["a_frac", "a_width_p90_px", "v_width_p90_px", "av_width_ratio_p90"]
NaN pattern must be identical for every field
```

All 8,870 canonical images, refactored module vs frozen table, in-memory:

| column | NaN pattern | max abs delta | tolerance | n over tolerance | verdict |
|---|---|---|---|---|---|
| `a_frac` | same | 1.110e-16 | 1e-12 | 0 | OK |
| `a_width_p90_px` | same | 1.776e-15 | 1e-12 | 0 | OK |
| `v_width_p90_px` | same | 3.553e-15 | 1e-12 | 0 | OK |
| `av_width_ratio_p90` | same | 2.220e-16 | 1e-12 | 0 | OK |
| `branch_px_work` | same | **0.000e+00** | 0 (exact) | 0 | OK |
| `n_branches` | same | **0.000e+00** | 0 (exact) | 0 | OK |

```
AV_REFACTOR_CANONICAL_N                 : 8870
AV_FEATURE_COLUMNS_CHECKED              : 6
AV_ROWS_EXACT_OR_NUMERICALLY_EQUIVALENT : 8870
AV_ROWS_DIFFERENT                       : 0
AV_MAX_ABS_DELTA_FLOAT_FIELDS           : 3.553e-15   (atol 1e-12)   PASS
AV_MAX_ABS_DELTA_EXACT_FIELDS           : 0.000e+00   (atol 0e+00)   PASS
DIAGNOSTIC_CONTROL_MAX_DELTA            : 0.000e+00   control PASS (== 0)
AV_FULL_BEHAVIOR_EQUIVALENCE            : PASS  -> battery executed
```

**Resolution limit of the saved artefacts — stated so it is not over-read.** Comparing the *saved
CSV files* as written decimal strings gives **0 differences in all six columns over all 8,870
rows**, because the frozen artefact carries at most 16 significant decimal digits and a 1-ULP
difference (1.11e-16 near 0.5, 3.55e-15 near 30) falls below that resolution. Textual identity is
therefore *consistent with* bit-identity but does not prove it, and the CSV-level comparison can
only ever return 0. The authoritative record is the run's own in-memory comparison above. The
frozen table used was verified at
sha256 `db123ac5f663f4925ac9fff52d204bede85963966794062d1855e315a4e38ffc`.
These artefact-level numbers are regenerated by `scripts/task5b_n2_gate_recheck.py`, which imports
the constants from the file that ran and refuses to write a record if the run's own summary
disagrees.

#### Battery — 360 frozen images × 26 perturbations, production branch labels

Sample: 360 images (`plus` 152, `farabi` 118, `farfum_rop` 90; splits train 249 / val 60 / test 51;
1600×1200 118, 1280×960 90, 640×480 90, 1240×1240 62 — 1440×1080 absent as a data-limited stratum,
not resampled). Baseline branches 27,596. **Branch-set mismatches across all conditions: 0**, so
every condition is comparable to its own baseline.

| family | mean flip | max flip |
|---|---|---|
| illumination gradient | 0.0492 | 0.3846 |
| gamma | 0.0342 | 0.3333 |
| brightness | 0.0282 | **0.7727** |
| contrast | 0.0181 | 0.3333 |
| white balance | 0.0093 | 0.3750 |

```
OVERALL_MEAN_BRANCH_FLIP_RATE = 0.0266
OVERALL_MAX_BRANCH_FLIP_RATE  = 0.7727   (brightness x1.2 on a single image)
images with >=1 flipped branch  : 0.9944
images with >=10% flipped       : 0.9639
images with >=25% flipped       : 0.8306
by source : plus 0.0187  farabi 0.0311  farfum_rop 0.0340
by geometry: 640x480 0.0184  1240x1240 0.0192  1280x960 0.0317  1600x1200 0.0329
```

Two structural results matter more than the averages:

* **The rule is blind to the red and blue channels.** Scaling R alone or B alone (`R+10`, `R-10`,
  `B+10`, `B-10`) gives a flip rate of **exactly 0.0000** on every image. The A/V split is a
  function of the green channel only, so any perturbation confined to R or B cannot change a label.
  Green scaling does move labels (`G+10` mean 0.0309, max 0.3750).
* **Illumination gradient is the strongest single family on average**, and brightness ×1.2 is the
  strongest single condition in the tail. Label assignment is driven by a per-image
  background-corrected green level, so a spatially varying or globally rescaled exposure shifts the
  length-weighted median directly.

Tie behaviour, now measured on production labels rather than inferred from a broken replica:

```
TIE_AT_THRESHOLD_MEDIAN_FRACTION      = 0.0758   (median 5 of 61 branches sit exactly at threshold)
images with at least one tie          = 1.0000
spearman(tie_fraction, mean flip rate) = 0.4320
```

Ties are real — every image has at least one branch exactly at the split value — but they are
**7.6% of branches, not the whole rule**, and their rank correlation with instability is moderate
(ρ = 0.43). The earlier "tie-dominated" hypothesis remains **withdrawn**: it was produced by the
attempt-1 scaling bug, and the measured evidence does not support it.

50/50 structural balance under production labels (`n = 360`):

| quantity | median | IQR | p05 | p95 | min | max | in 0.45–0.55 |
|---|---|---|---|---|---|---|---|
| `a_frac` by branch count | 0.6069 | 0.0891 | 0.5035 | 0.7647 | 0.3864 | 0.8889 | 0.1639 |
| `a_frac` by branch length | 0.5411 | 0.0635 | 0.5035 | 0.7011 | 0.5002 | 0.9777 | 0.6139 |

Feature stability under perturbation:

| feature | median \|Δ\| | median rel | p95 \|Δ\| | max \|Δ\| |
|---|---|---|---|---|
| `a_frac` | 0.00185 | 0.00349 | 0.11916 | 0.47377 |
| `a_width_p90_px` | 0.00000 | 0.00000 | 0.71833 | 3.69283 |
| `v_width_p90_px` | 0.00000 | 0.00000 | 0.74085 | 5.71484 |
| `av_width_ratio_p90` | 0.00000 | 0.00000 | 0.15171 | 0.70088 |

Median changes are zero or near-zero because the p90 is an order statistic over a small branch set
and usually does not move; when it does move it jumps by whole rank positions, which is why the
tail reaches several pixels. A feature whose value is usually identical and occasionally jumps is
not a stable measurement, and this is consistent with the external-benchmark demotion of
`width_shape_p90_over_p50`.

**What is and is not established.** The instrumentation is now proven faithful: one implementation,
used by both the production path and the diagnostic path, with branch labels and threshold
bit-identical to the frozen table within 1e-12 and the direct `a_frac` control exactly 0 on 8,870
images. The perturbation battery has been executed and reported. Neither fact validates the labels
biologically: there are still **no expert artery/vein annotations**, a median of 7.6% of branches
sit exactly on the split value with no tie-break rule, 99.4% of images change at least one label
under some mild intensity perturbation, and 83.1% change at least a quarter of their labels. The
honest status is therefore unchanged and remains `EXPLORATORY_ONLY` — now for measured reasons
rather than for lack of a test.

The attempt-1 replica results stay invalidated by the `np.clip(rgb, 0, 1)` scaling bug, and its
tie-dominated hypothesis stays withdrawn.

```
AV_STRESS_TEST_EXECUTED:              YES (production-faithful; 360 x 26 battery executed)
AV_PRODUCTION_INSTRUMENTATION:        PASS
AV_FULL_BEHAVIOR_EQUIVALENCE:         PASS (predeclared atol 1e-12; max 3.553e-15)
AV_SAMPLE_N:                          360
BASELINE_BRANCH_N:                    27596
DIAGNOSTIC_CONTROL_MAX_DELTA:         0.000e+00
BRANCH_SET_MISMATCHES:                0
OVERALL_MAX_BRANCH_FLIP_RATE:         0.7727
OVERALL_MEAN_BRANCH_FLIP_RATE:        0.0266
TIE_AT_THRESHOLD_MEDIAN_FRACTION:     0.0758
AV_BALANCE_BRANCH_MEDIAN:             0.6069
AV_BALANCE_LENGTH_MEDIAN:             0.5411
AV_STATUS:                            EXPLORATORY_ONLY
EXPERT_AV_VALIDATION_AVAILABLE:       NO
DOCUMENTATION_OVERCLAIM_CORRECTED:    YES
FAILED_REPLICA_RESULTS_USED:          NO
TASK5B_N2_CLOSURE:                    COMPLETE
```

Original section text retained below for provenance.

| feature | admission |
|---|---|
| `a_frac` | `EXPLORATORY_ONLY` — mechanically ≈ 0.5 by construction |
| `a_width_p90_px`, `v_width_p90_px` | `EXPLORATORY_ONLY` |
| `av_width_ratio_p90` | `EXPLORATORY_ONLY` |

`AV_STATUS = EXPLORATORY`. The branch labels have not been compared against expert artery/vein
annotations, and no brightness, contrast, gamma, white-balance or channel-scaling stress test can
substitute for that. **Physiological plausibility is not accepted as validation.**

---

## O. Missingness shortcut audit

> **CORRECTION (Task 5C).** The AUCs reported in this section were computed with acquisition
> geometry — `dd_over_min_side` and `fov_coverage_fraction` — as the **predictor**, not with the
> missingness indicator. They are therefore `geometry → target` AUCs and are mislabelled here as
> a missingness shortcut. Task 5C recomputed the statistic correctly: with the missingness
> indicator as the predictor, `PRIMARY_CORE_V1` missingness shows **no material association** with
> source (χ²=0.24, p=0.886), geometry (χ²=5.68, p=0.224) or label (χ²=0.12, p=0.942), on 8
> missing rows of 8870. The `MISSINGNESS_SHORTCUT_STATUS = FAIL` headline in the header block of
> this document is **withdrawn**; the geometry → label AUC of 0.6445 remains a real and separate
> acquisition-shortcut finding, reported in section P.

`is_missing(feature)` was derived for every feature and used as the target of a 5-fold logistic
regression on `(dd_over_min_side, fov_coverage_fraction)`. Macro one-vs-rest AUC:

| target | CV macro AUC |
|---|---|
| geometry | **0.794** |
| source | **0.7165** |
| label | **0.6445** |

Every disc-dependent feature shares one missingness pattern (they are all NaN together when
`disc_valid = 0`), which is why the table shows one block of identical values across ~35 features.

**`MISSINGNESS_SHORTCUT_STATUS = FAIL`.** Missingness is predictable from acquisition geometry at
AUC 0.79 and from the disease label at 0.64. Consequently:

* every `is_missing(<feature>)` indicator is `FORBIDDEN_FROM_FINAL_CLASSIFIER`;
* `disc_valid`, `disc_peak_prob`, `disc_method`, `fov_valid`, `fov_failure_reason` and all five
  ROI coverage gates are metadata/QC only and `FORBIDDEN_FROM_FINAL_CLASSIFIER`;
* the classifier matrix must never contain a missingness indicator — enforced by test.

---

## P. Source and geometry sensitivity (distributional, no labels)

Ranked by `source_median_spread / (p95−p05)`:

| feature | normalised source spread | max source | min source |
|---|---|---|---|
| `disc_valid` | **1.0000** | farfum_rop | farabi |
| `fov_n_components` | 0.6136 | farfum_rop | farabi |
| `roi_coverage_ring_2_3dd` | 0.4381 | farabi | farfum_rop |
| `fov_border_contact` | 0.3858 | farabi | plus |
| `skeleton_px_work` | 0.3768 | plus | farabi |
| `disc_peak_prob` | 0.3743 | farfum_rop | plus |
| `vessel_density_fov` | 0.3383 | plus | farabi |
| `width_p50_dd` | 0.2919 | farabi | plus |
| `tort_geodesic_median` | — | — | — |

The interpretation is recorded in two parts, as required:

* **Known acquisition dependence.** `disc_valid` (1.00), `fov_n_components` (0.61),
  `fov_border_contact` (0.39), `disc_peak_prob` (0.37) and every `roi_coverage_*` are acquisition
  and detector properties. They are forbidden or QC.
* **Plausible biology or genuine clinical difference, not eliminated.** Absolute counts and raw
  densities differ between cohorts, but the cohorts are different populations imaged on different
  cameras and the disease prevalence differs. `vessel_density_fov` and `tort_geodesic_median` are
  **not** removed on this evidence; a source difference is a reason to report uncertainty, not to
  delete a true measurement.

Full table: `_private_audit/task5b_source_sensitivity.csv`.

---

## Q. Aspect-ratio consequence of the 256×256 segmentation path

`SEG_CURRENT_V1` does **not** preserve aspect ratio. A 1280×960 frame is squeezed horizontally by
1280/960 = 1.333 before inference and stretched back afterwards. Simulated by anisotropic
resampling of a synthetic vessel:

| native | x/y ratio | width change | tortuosity change | skeleton-pixel change |
|---|---|---|---|---|
| 1280×960 | 1.3333 | **−15.0 %** | **+19.6 %** | −6.1 % |
| 1440×1080 | 1.3333 | −15.0 % | +19.6 % | −6.1 % |
| 1600×1200 | 1.3333 | −15.0 % | +19.6 % | −6.1 % |
| 640×480 | 1.3333 | −15.0 % | +19.6 % | −6.1 % |
| 1240×1240 | 1.0000 | 0.0 % | 0.0 % | 0.0 % |

**`ASPECT_RATIO_MEASUREMENT_RISK = FEATURE_DEPENDENT`.** Shape-sensitive features (tortuosity,
width, branch geometry) carry a geometry-dependent bias of up to ~20 %, while the 1240×1240 frames
— 2446 of 8870, the largest single geometry group — are unaffected. Density is a ratio and is much
less exposed. `SEG_CURRENT_V1` was **not** changed; this is a limitation analysis only.

---

## R. Dead and defective features, one by one

| feature_name | defect | evidence | historical use | current proposed status |
|---|---|---|---|---|
| `n_endpoints` | **exact duplicate** of `n_startpoints` — identical expression in the implementation | r = 1.0000; 6 exactly duplicate columns in section S | `extract_pvbm` GEOM_COLUMNS | `EXPLORATORY_ONLY` |
| `n_endpoints_fov` | **exact duplicate** of `n_startpoints_fov` | r = 1.0000 | new in V1 | `EXPLORATORY_ONLY` |
| `vessel_area_fraction_work` | **exact duplicate** of `vessel_density_wholeframe` | r = 1.0000 | new in V1 | `EXPLORATORY_ONLY` |
| `singularity_length` | unstable: relative spread 1.243; +110 % under a pure rotation | section K | PVBM `MultifractalVBMs` output | `FORBIDDEN_FROM_FINAL_CLASSIFIER` |
| `width_quantisation_floor_dd` | deterministic function of image size and disc size; a QC quantity, not a measurement | contract definition | new in V1 | `FORBIDDEN_FROM_FINAL_CLASSIFIER` |
| `n_startpoints` | ROI/boundary topology; a line end manufactured by the FOV or frame edge | section J | `extract_pvbm` GEOM_COLUMNS | `EXPLORATORY_ONLY` |
| `n_intersections` | raster topology, not vascular topology; 7 → 19 → 18 under 1×/2×/4× | sections I and J | `extract_pvbm` GEOM_COLUMNS | `EXPLORATORY_ONLY` |
| `a_frac` | mechanically ≈ 0.5 by construction (equal-length split) | section N | `clinical_features_v3` | `EXPLORATORY_ONLY` |
| `tort_pixelcount_median_provenance` | orientation-dependent definition, rejected | section D | historical clinical tortuosity | `EXPLORATORY_ONLY` |
| `fractal_d1` vs `fractal_d2` | near-redundant pair | r = 0.9950 | PVBM fractal | both `SECONDARY_ALLOWED`, pair flagged |
| `roi_coverage_ring_0_2dd` vs `roi_coverage_annulus` | near-redundant pair | r = 0.9989 | new in V1 | both `FORBIDDEN` (QC) |
| `skeleton_px_work` vs `branch_px_work` | near-redundant pair | r = 0.9990 | new in V1 | both `EXPLORATORY_ONLY` |

No aggregate statement is made: each row above is a separate finding with its own evidence.

---

## S. Feature redundancy

Computed without labels, on 65 numeric candidate columns.

**Exactly duplicate columns (6):** `n_endpoints`, `n_endpoints_fov`, `n_startpoints`,
`n_startpoints_fov`, `vessel_area_fraction_work`, `vessel_density_wholeframe`.

**Pairs with |r| ≥ 0.98 (7):**

| a | b | \|r\| |
|---|---|---|
| `vessel_density_wholeframe` | `vessel_area_fraction_work` | 1.0000 |
| `n_startpoints` | `n_endpoints` | 1.0000 |
| `n_startpoints_fov` | `n_endpoints_fov` | 1.0000 |
| `skeleton_px_work` | `branch_px_work` | 0.9990 |
| `roi_coverage_ring_0_2dd` | `roi_coverage_annulus` | 0.9989 |
| `fractal_d1` | `fractal_d2` | 0.9950 |
| `n_intersections` | `n_branches` | 0.9937 |

None of these duplicates is in the classifier matrix, so the matrix carries no meaningless
duplicate measurement. Full list: `_private_audit/task5b_redundancy_pairs.csv`.

---

## T. New feature table

`data/features/clinical_measurement_v1.csv`

| property | value |
|---|---|
| rows | **8870** (canonical, one per image) |
| columns | 78 |
| segmentation generation | `SEG_CURRENT_V1` only |
| measurement version field | `measurement_version = CLINICAL_MEASUREMENT_V1` on every row |
| source / split / group linkage | `source`, `split`, `group_id`, `patient_id`, `exam_id`, `identity_level` |
| byte-level SHA256 | **`db123ac5f663f4925ac9fff52d204bede85963966794062d1855e315a4e38ffc`** |
| historical tables overwritten | **none** |

This is **not** the final classifier table: it contains every measured feature plus admission
metadata, including the QC and forbidden columns.

---

## U. Classifier matrix is separate

`configs/clinical_measurement_v1_classifier_features.yaml` — **18 features**, frozen.

`PRIMARY_ALLOWED` (4): `tort_geodesic_median`, `tort_geodesic_p90`, `vessel_density_fov`,
`skel_density_fov`.

Approved `SECONDARY_ALLOWED` (14): `tort_geodesic_top3_mean`, `n_branches`, `width_p50_dd`,
`width_p90_dd`, `width_mean_dd`, `width_ann_p50_dd`, `width_ann_p90_dd`, `width_ann_mean_dd`,
`width_shape_p90_over_p50`, `vessel_density_fov_ring_2_3dd`, `vessel_density_fov_ring_3_6dd`,
`fractal_d0`, `fractal_d1`, `fractal_d2`.

Excluded by construction: metadata, identifiers, `source`, `split`, geometry, raw image
dimensions, detector confidence, detector-validity flags, all QC variables, all ROI coverage gates,
all exploratory A/V variables and every forbidden scale-dependent feature. Enforced by test.

Missingness policy: training-split medians only, fitted per fold; missingness indicators never in
the matrix.

---

## V. Feature contract

`configs/feature_contract.csv` — **134 rows × 19 columns**, including every required column:
`feature_name, measurement_version, mathematical_definition, unit, roi_definition,
depends_on_disc, depends_on_fov, depends_on_resolution, depends_on_av, scale_invariant,
padding_invariant, rotation_invariant, missingness_policy, admission_status,
allowed_in_final_classifier, requires_expert_validation, known_limitations` (plus
`feature_version` and `generation_script` for provenance).

* 69 rows for `CLINICAL_MEASUREMENT_V1` — one per generated column, generated from the frozen
  measurement contract so the two cannot drift apart.
* 65 preserved historical rows (`pvbm_v1`, `clinical_v3`), marked
  `HISTORICAL_NOT_ADMITTED` with `UNKNOWN` dependency flags rather than invented values.

---

## W. Test suite

`tests/test_measurement_v1.py` — **42 passed, 3 skipped** (skips are the table/contract tests when
the git-ignored data directory is absent). Coverage:

tortuosity rotation invariance · straight tortuosity ≈ 1 at 7 orientations · straight-line
dispersion < 0.01 · the historical definition still fails (so the difference stays visible) ·
curved > straight · analytic circular arcs · T ≥ 1 · short fragments excluded · raw width accuracy ·
width/DD scale invariance · width orientation dispersion bounded · whole-frame density is **not**
padding invariant · FOV-normalised density **is** exactly padding invariant · rectangular padding
too · FOV recovers a synthetic retina · FOV reports a failure reason · disc invalid ⇒ every
disc-dependent feature NaN · low detector confidence invalidates the disc · implausible disc
diameter invalidates the disc · ROI below coverage ⇒ NaN · coverage reported for every region ·
ROI coordinates scale with the grid · topology scales with resolution · every emitted feature is
declared in the contract · every emitted feature has an admission status · A/V is exploratory only ·
no fallback disc rule · classifier matrix contains only allowed features · classifier matrix
excludes metadata and QC · no missingness indicator in the matrix · no infinities · canonical row
count · measurement version present · contract completeness and required columns.

Regression: the existing suite (`tests/` minus the new file) continues to pass.

---

## X. What this does and does not mean

**Established:** `MEASUREMENT IMPLEMENTATION TECHNICALLY VALIDATED`. Every measured feature has a
frozen mathematical definition, an admission status, measured scale/FOV/padding/rotation
dependencies, a quantified floor where one exists, and an explicit guard where it is unreliable.

**Not established:** any clinical validity. No expert mask, no expert artery/vein annotation and no
clinical endpoint was used. A feature being admissible is a statement about the measurement, not
about the disease.

**Two findings that constrain everything downstream:**

1. **Only 42.5 % of the cohort has a valid disc**, and disc validity is source-dependent
   (38 % for plus, 53 % for farfum_rop, 49 % for farabi). Any disc-dependent model is fitted on a
   source-biased subsample, and the missingness itself predicts the label at AUC 0.64.
2. **The 256×256 inference path biases shape-sensitive features by up to ~20 %** on the four
   non-square geometries, which together are 6424 of 8870 images.

Neither was introduced by this task and neither may be changed here. Both must be addressed
explicitly — by a new segmentation generation and a pre-specified missingness policy — before any
corrected classifier is fitted.

---

## Y. Success gate

| # | requirement | status |
|---|---|---|
| 1 | every feature has an explicit mathematical definition | ✓ `configs/clinical_measurement_v1.yaml` |
| 2 | every feature has an admission status | ✓ 69 V1 rows, 4 statuses, test-enforced |
| 3 | scale/FOV/padding/rotation dependencies measured | ✓ sections D, E, G, H, I, K, L, Q |
| 4 | tortuosity passes synthetic correctness | ✓ dispersion 0.0039, analytic error 0.0077 |
| 5 | width evaluated end-to-end | ✓ estimator selected, floor quantified |
| 6 | whole-frame density not silently treated as invariant | ✓ renamed, marked exploratory, alternatives added |
| 7 | A/V appropriately restricted without expert validation | ✓ `EXPLORATORY_ONLY` |
| 8 | ROI coverage explicitly handled | ✓ 0.60 gate, NaN below, reported per feature |
| 9 | missingness/QC shortcuts cannot enter the classifier | ✓ 24 forbidden, test-enforced, shortcut measured at AUC 0.79 |
| 10 | final allowed classifier-feature list frozen | ✓ 18 features |
| 11 | new table byte-hashed | ✓ `db123ac5…8ffc` |
| 12 | no model trained | ✓ |

```
TASK5B_STATUS = COMPLETE
```
