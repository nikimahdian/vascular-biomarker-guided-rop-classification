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

### FOV BRIGHTNESS / BORDER ROBUSTNESS CLOSURE — Task 5B-H

This is a **different test from the synthetic padding-invariance check tabulated above**. That check
placed a synthetic circular/rectangular retina on a synthetic canvas and confirmed the *geometry* of
the mask arithmetic. This closure keeps **real project images and their real `SEG_CURRENT_V1` vessel
masks** and perturbs the photograph, which is the condition the feature actually has to survive.

Frozen before testing, not modified: `src/biomarker/clinical_measurement_v1.py::retinal_fov`,
file sha256 `6aea0d6856d75e205f88201cfc20311a4d13acfdac839aa5ac8bfac3a1f13542`. Rule as implemented —
`thr = threshold_otsu(green)`, `binimg = green > thr`, `remove_small_objects` / `remove_small_holes`
at `max(64, 0.001·h·w)`, `binary_closing(disk(3))`, keep the largest component; failure when
`coverage < 0.15`, `coverage > 0.985`, or `largest/sum < 0.90`. Vessel mask fixed across every
condition, so every density change is FOV detection only. No threshold was tuned after seeing results.

**Harness faithfulness.** Baseline `vessel_density_fov` recomputed on 328 project images matches the
frozen table to `9.71e-17`, `fov_valid` mismatches 0, `fov_coverage_fraction` to `5.55e-17`.

Sample (deterministic, label-free): 328 images stratified by source × geometry × native-brightness
tercile — `plus` 180 / `farabi` 88 / `farfum_rop` 60; geometries 1600×1200 88, 1280×960 60,
640×480 60, 1240×1240 60, 1440×1080 60; splits train 207 / val 58 / test 63; native grey level
35.3–128.9. Selection used source, geometry, brightness and path order only.

**Disclosed revision between two runs of this closure.** The first run selected 178 images, short of
the `N ≥ 300` requirement, and reported only the raw pre-morphology component count. A second run
raised the sample to 328 (a hard requirement of the task, not an outcome-driven choice), added a
post-morphology component diagnostic, and reported the primary endpoint on the baseline-valid subset
in addition to all images, because `vessel_density_fov` is undefined where the baseline FOV detection
itself failed. No FOV threshold, no condition and no class definition was changed, and the first
run's condition table is kept as evidence in `_private_audit/run1_conditions.csv` (5,340 rows). Both
runs agree on the verdict.

| family | median FOV Dice | min | pairs with Dice < 0.90 | density \\|rel\\| > 1 % | > 5 % | > 10 % |
|---|---|---|---|---|---|---|
| brightness (8 conditions) | **1.000000** | 0.6223 | 1.1 % | 7.1 % | 1.4 % | 0.5 % |
| uniform black border | 0.9890 | 0.5771 | 7.5 % | 43.0 % | 15.2 % | 5.4 % |
| asymmetric black border | 0.9845 | 0.5771 | 11.1 % | 46.1 % | 19.8 % | 7.9 % |
| thicker left/right | 0.9848 | 0.5771 | 11.1 % | 46.1 % | 19.6 % | 7.4 % |
| thicker top/bottom | 0.9838 | 0.5771 | 11.5 % | 46.1 % | 20.5 % | 7.8 % |
| one-sided black | 0.9898 | 0.5767 | 7.1 % | 42.7 % | 14.5 % | 4.6 % |
| irregular black frame | 0.9829 | 0.5771 | 12.5 % | 47.6 % | 21.4 % | 8.8 % |
| near-black border (level 5) | 0.9912 | 0.7553 | 5.9 % | 40.0 % | 12.9 % | 3.4 % |
| dark-grey border (level 40) | 0.9872 | **0.3179** | 3.7 % | 52.9 % | 20.0 % | **12.6 %** |
| overwrite black (content-altering) | 0.9666 | 0.7547 | 4.9 % | 60.9 % | 29.1 % | 4.3 % |

```
FOV_SAMPLE_N                        : 328   (325 baseline-valid)
BRIGHTNESS_MEDIAN_FOV_DICE          : 1.000000   BRIGHTNESS_MIN_FOV_DICE : 0.622341
BORDER_MEDIAN_FOV_DICE              : 0.984718   BORDER_MIN_FOV_DICE     : 0.317889
BRIGHTNESS median/p95/max |rel dens|: 0.00000 / 0.001742 / 0.072794
BORDER     median/p95/max |rel dens|: -0.001184 / 0.051578 / 2.187549
FOV_FAILURES_BASELINE               : 3 / 328
FOV_FAILURES_PERTURBED              : 37 / 9840 condition pairs (0.38 %)
FOV_ROBUSTNESS                      : FAIL
VESSEL_DENSITY_FOV_FINAL_PRIMARY    : REQUIRES_REVIEW
FINAL_PRIMARY_REQUIRES_REVIEW       : YES
```

**Brightness is scale-invariant, exactly, until pixels clip.** For the six multiplicative conditions
`0.25, 0.50, 0.70, 0.85` and the well-behaved part of `1.15 … 1.80`, Otsu tracks the scaling exactly:
`thr_perturbed / (thr_baseline · c) = 1.0000`. Multiplicative darkening therefore leaves the FOV
**bit-identical** — median Dice `1.000000`, and at ×0.25 and ×0.50 the minimum is also `1.000000`.
The tail appears only once the upper range compresses: saturated-pixel fraction rises from 0.4 % at
×1.15 to 10.5 % at ×1.80, the threshold ratio slips to `0.9901`, and the images whose baseline FOV is
already marginal break (`min Dice 0.6223` at ×1.80; 13–22 % of images move >1 % at ×1.30–×1.80).
Median density change is exactly `0.00000` at every brightness condition.

**Borders fail through the threshold, not through component selection.** The quantity that decides
every border outcome is the Otsu threshold shift. Adding a large black region to the histogram pulls
the threshold down by 20–55 % (`thrRatio 0.41–0.56`: 93.07 → 39.35, 51.02 → 28.72, 73.69 → 66.11);
retinal background lying between the old and new threshold then flips into the bright class and the
FOV grows — up to 2.5× the baseline area — so `vessel_density_fov` falls because the added area is
peripheral low-density retina (0.1464 → 0.0801, −45 %). A dark-grey border above the image's own
threshold does the opposite: at level 40 with baseline threshold 35.98 the border joins the bright
class, Otsu **rises** to 74.37 (`ratio 2.07`), the FOV collapses to 12 % coverage, and the detection
hard-fails. **`fov_frac_inside_pad` is `0.000` in every inspected case: the border is never itself
selected as the FOV, so component selection is not the mechanism.** The deciding factor is how the
border intensity sits relative to each image's own green-level distribution, which is why dark
low-contrast images (baseline threshold 40–50, small baseline FOV) are hit hardest and bright
high-contrast images (threshold 90–107) barely move.

Consequences, applied literally:

* the failure is **border-driven, not brightness-driven**; brightness alone would classify as ROBUST
  at the predeclared cut, borders do not;
* the predeclared class definitions were **not** changed after seeing these numbers;
* `vessel_density_fov` is **not** demoted and `retinal_fov` is **not** redesigned inside this task —
  `FINAL_PRIMARY_REQUIRES_REVIEW = YES` is declared and disease-model training stops here, per
  section L;
* the risk is conditional on the acquisition path: project images carry no synthetic border, so the
  border result bounds what an export/letterbox step would do, while the brightness result applies
  to ordinary exposure variation.

Private QC montages for the strongest brightness failures, the strongest border failures, a median
case per geometry and stable controls are written by `scripts/task5b_h_mechanism.py` to
`_private_audit/task5b_h_qc_*.png`; the mechanism table is `_private_audit/task5b_h_mechanism.csv`.
No disease label is read at any point in this closure.

### FOV BORDER IMPACT ON ALL FIVE FINAL_PRIMARY BIOMARKERS — Task 5B-H2

Task 5B-H showed the FOV detector is border-fragile. This closure asks how far that fragility
travels through the production measurement path into the five `FINAL_PRIMARY` features.

Same frozen evidence, nothing resampled or added: the **same 328-image sample** (verified set-equal
to the Task-5B-H baseline table), the **same 8 brightness and 22 border conditions**, the same
loader. `measure(rgb, mask, {}, with_fractal=True)` — the production entry point — is called for the
baseline and for all 30 perturbations, so all five features come from the production path and no
parallel replica exists. The vessel mask is carried through the same transform as the RGB and is
identical in content across all 31 measurements. Module sha256
`6aea0d6856d75e205f88201cfc20311a4d13acfdac839aa5ac8bfac3a1f13542`. 9,840 condition rows, 0 errors.
No classifier trained, no threshold changed, no feature removed, no FOV redesign, feature table not
regenerated. Classification cuts were written into the script before it ran.

**D — how the FOV enters each feature, traced in the code (not inferred from the name):**

| feature | uses FOV | how |
|---|---|---|
| `vessel_density_fov` | yes | **two** ways: numerator `mask[fov].sum()` is FOV-restricted vessel pixels; denominator `fov_px` is the FOV area |
| `skel_density_fov` | yes | **one** way: numerator `len(nonzero(skeletonize(mask)))` is the **whole-frame** skeleton and does not involve the FOV at all; denominator is `fov_px` |
| `fractal_d0`, `fractal_d1`, `fractal_d2` | yes | `_fractal(mask & fov)` — the FOV is the **support domain** of the binary image handed to `MultifractalVBMs` |

**E — feature-level deltas.** Brightness perturbs none of the five materially; the border family
perturbs all five.

| feature | perturbation | med \|Δ\| | p95 \|Δ\| | max \|Δ\| | med rel | p95 rel | max rel | >1 % | >5 % | >10 % |
|---|---|---|---|---|---|---|---|---|---|---|
| `vessel_density_fov` | brightness | 0.00000 | 0.00141 | 0.15463 | 0.00000 | +0.00171 | +0.0728 | 7.5 % | 1.8 % | 0.8 % |
| `skel_density_fov` | brightness | 0.00000 | 0.00093 | 0.06392 | 0.00000 | 0.00000 | +0.0435 | 12.8 % | 4.8 % | 2.6 % |
| `fractal_d0` | brightness | 0.00000 | 0.01014 | 0.12089 | 0.00000 | +0.00563 | +0.0959 | 4.1 % | 0.5 % | 0.0 % |
| `fractal_d1` | brightness | 0.00000 | 0.01349 | 0.16226 | 0.00000 | +0.00517 | +0.1348 | 5.0 % | 0.8 % | 0.3 % |
| `fractal_d2` | brightness | 0.00000 | 0.01378 | 0.17567 | 0.00000 | +0.00572 | +0.1494 | 5.2 % | 1.0 % | 0.3 % |
| `vessel_density_fov` | **border** | 0.00072 | 0.01387 | 0.17660 | −0.00124 | +0.05097 | **+2.1876** | **47.3 %** | 19.9 % | 7.5 % |
| `skel_density_fov` | **border** | 0.00044 | 0.00655 | 0.06472 | **−0.01676** | +0.02532 | **+4.2915** | **59.0 %** | **42.1 %** | **24.0 %** |
| `fractal_d0` | **border** | **0.02795** | 0.12121 | 0.23372 | **−0.01782** | +0.01775 | +0.1196 | **71.2 %** | 21.2 % | 2.7 % |
| `fractal_d1` | **border** | **0.03160** | 0.12657 | 0.36265 | **−0.01702** | +0.03649 | +0.1411 | **74.9 %** | 22.8 % | 3.6 % |
| `fractal_d2` | **border** | **0.03198** | 0.12513 | 0.39589 | **−0.01631** | +0.04097 | +0.1607 | **75.2 %** | 23.2 % | 3.9 % |

Outliers are not hidden: the worst border cases are `+219 %` on `vessel_density_fov`, `+429 %` on
`skel_density_fov`, and absolute fractal changes of 0.23 / 0.36 / 0.40 (features sit near 1.2–1.5).

**F — source and geometry.** Border sensitivity is not concentrated in one cohort. All three sources
move in the same direction and by comparable magnitude (`vessel_density_fov` median rel: farfum_rop
−0.0073, farabi −0.0012, plus −0.0000; `skel_density_fov`: −0.0468, −0.0017, −0.0353). The fractals
are the most uniform of all: median −0.014 to −0.019 in every source and every geometry, i.e. a
**systematic negative bias of ≈1.5–2 %** rather than a source-specific artefact. Geometry matters
only for the two density features (1440×1080 has the smallest median, 1240×1240 and 640×480 the
largest for `skel_density_fov`, −0.087 and −0.051).

**G — FOV error vs biomarker error (Spearman, border rows, n = 7,216).**

```
vessel_density_fov  dice vs |delta| = -0.8959   relArea vs |delta| = +0.7063
skel_density_fov    dice vs |delta| = -0.9595   relArea vs |delta| = +0.8125
fractal_d0          dice vs |delta| = -0.0037   relArea vs |delta| = +0.0787
fractal_d1          dice vs |delta| = -0.0143   relArea vs |delta| = +0.0711
fractal_d2          dice vs |delta| = -0.0107   relArea vs |delta| = +0.0659
```

The two density features track FOV error almost perfectly; the fractals are **nearly orthogonal to
it**. That near-zero correlation is the key diagnostic: the fractal error is *not* driven by how much
the FOV mask moved.

**H/I — mechanism, per feature. These are three different pathways, not one.**

1. **`skel_density_fov` — pure denominator, exactly.** Its numerator is the whole-frame skeleton,
   which the FOV never touches (`skeleton_px_work` changed in **0 of 9,840 rows**). The measured
   delta therefore obeys `Δ = S·(1/fp_perturbed − 1/fp_baseline)` with no free parameter, and the
   observed signed delta matches that prediction to **max |difference| = 9.8e-17**. Every bit of
   `skel_density_fov` border sensitivity is the FOV area in the denominator, and it is the most
   frequently affected feature (>5 % in 42 % of border rows) precisely because nothing damps it.
2. **`vessel_density_fov` — denominator and inclusion, partially cancelling.** Splitting the change
   into a denominator-only term (`V_base/fp_p − V_base/fp_b`) and an inclusion-only term
   (`V_p/fp_p − V_base/fp_p`) gives median |0.00246| and |0.00121| against a median total of
   |0.00072|, with maxima 0.346 and 0.170. The two terms have opposite sign in the common case — a
   larger FOV both enlarges the denominator and admits more (low-density peripheral) vessel pixels —
   so the typical error is small while either term alone is large. The extreme cases are where the
   cancellation breaks.
3. **`fractal_d0/d1/d2` — the multifractal computation domain, not the vascular support.** The FOV
   reaches these features only as the support of the binary input, so the naive expectation is that
   they change because vessels are added or removed. The data say otherwise. Among border rows,
   2,485 have an **identical** support pixel count, and their fractal change (median 0.0332 / 0.0354
   / 0.0359) is **larger** than in the 4,731 rows where the support did change (0.0252 / 0.0298 /
   0.0300); the rank correlation between |support change| and |fractal change| is only +0.14 to
   +0.16. A dedicated experiment settles it: take the baseline binary input and merely embed it,
   unchanged, in a zero-padded canvas the size of the border condition — no FOV call, no vessel
   pixel added or removed — and recompute. D0/D1/D2 move by **median −2.29 %, −2.68 %, −2.78 %**
   (p95 +3.1 %, +4.7 %, +4.9 %), which reproduces the sign and magnitude of the observed border
   medians (−1.78 %, −1.70 %, −1.63 %). **The fractal border sensitivity is a canvas-extent effect
   on the multifractal estimator, largely independent of the FOV mask.** The three behave alike but
   not identically: D2 moves most, D0 least, in every breakdown.

**J — NaN and failure transitions.** No feature changed finiteness anywhere: 0 finite→NaN and 0
NaN→finite for all five, in both families, over 9,840 rows (no baseline NaN among the five on this
sample). FOV validity itself: brightness 0 valid→invalid, 8 invalid→valid; border 5 valid→invalid,
50 invalid→valid. The fragility therefore shows up as **value error, not as new missingness**.

**K — visual QC.** 16 montages in `_private_audit/task5b_h2_qc_*.png`, one per worst case, each
showing the original, the perturbed image, the baseline FOV contour, the perturbed FOV contour, the
fixed vessel mask and the baseline/perturbed feature values. Worst cases: `vessel_density_fov`
0.0807→0.2573; `skel_density_fov` 0.0768→0.0121; `fractal_d0` 1.5001→1.2664; `fractal_d1`
1.4748→1.1122; `fractal_d2` 1.4927→1.0968. Case list: `_private_audit/task5b_h2_qc_cases.csv`.

**M — impact classification.**

| feature | status | justification |
|---|---|---|
| `vessel_density_fov` | `SEVERELY_BORDER_SENSITIVE` | 47.3 % of border rows move >1 %, 19.9 % >5 %; worst case +219 %; FOV error tracks it at ρ = −0.90; mechanism is a two-term cancellation that fails in the tail |
| `skel_density_fov` | `SEVERELY_BORDER_SENSITIVE` | most frequently affected: 59.0 % >1 %, **42.1 % >5 %**, 24.0 % >10 %; unmitigated pure denominator effect confirmed analytically to 1e-16; ρ = −0.96 with FOV error |
| `fractal_d0` | `BORDER_SENSITIVE` | 71.2 % of border rows move >1 % and median error is 1.78 %, but the FOV-independent canvas experiment already produces 2.29 %, so the effect is bounded near the canvas bias; >10 % in only 2.7 %; ρ ≈ 0 |
| `fractal_d1` | `BORDER_SENSITIVE` | same structure as D0, slightly larger: median 1.70 %, >10 % in 3.6 %, max 14.1 %, canvas bias 2.68 % |
| `fractal_d2` | `BORDER_SENSITIVE` | largest of the three: median 1.63 %, >10 % in 3.9 %, max 16.1 %, canvas bias 2.78 % |

**N/O — decision.** The fragility is not confined to `vessel_density_fov`: **all five**
`FINAL_PRIMARY` features carry it, two of them severely, through **three distinct mechanisms** — a
denominator, a denominator-plus-inclusion pair, and the multifractal canvas domain. Because it
propagates into multiple primary biomarkers and produces recurring errors in the primary feature
matrix, section O requires:

```
TASK5B_H2_SAMPLE_N                    : 328
FINAL_PRIMARY_FEATURE_N               : 5
VESSEL_DENSITY_FOV_STATUS             : SEVERELY_BORDER_SENSITIVE
SKEL_DENSITY_FOV_STATUS               : SEVERELY_BORDER_SENSITIVE
FRACTAL_D0_STATUS                     : BORDER_SENSITIVE
FRACTAL_D1_STATUS                     : BORDER_SENSITIVE
FRACTAL_D2_STATUS                     : BORDER_SENSITIVE
PRIMARY_FEATURES_AFFECTED_N           : 5
PRIMARY_FEATURES_SEVERELY_AFFECTED_N  : 2
FOV_FAILURE_PROPAGATES_TO_FRACTALS    : YES
FOV_REDESIGN_REQUIRED_BEFORE_TRAINING : YES
FINAL_PRIMARY_REQUIRES_REVIEW         : YES
TASK5B_H2_STATUS                      : COMPLETE
```

`FOV_REDESIGN_REQUIRED_BEFORE_TRAINING = YES` is a statement about the **measurement**, and it is
conditional on the acquisition path the same way Task 5B-H was: project images carry no synthetic
border, so the border result bounds what any export, letterbox or border-adding step would do. The
brightness result is the part that applies unconditionally, and it is benign — median error exactly
`0.00000` for all five features. Nothing was redesigned here; the redesign is a separate decision.

### TASK 5B-H3 — ONE PREDECLARED BORDER-ROBUST CANDIDATE: `CLINICAL_MEASUREMENT_V2_CANDIDATE`

V1 stays the production state, untouched and reproducible:
`src/biomarker/clinical_measurement_v1.py` sha256
`6aea0d6856d75e205f88201cfc20311a4d13acfdac839aa5ac8bfac3a1f13542`; feature table
`f1c41e923ae29d4e228097536765f5e7399963062253ad925657a077cddc10c0`; measurement table
`db123ac5f663f4925ac9fff52d204bede85963966794062d1855e315a4e38ffc`. The candidate lives in a
separate module, `src/biomarker/clinical_measurement_v2_candidate.py`, and changes exactly two
things. `measure_v2()` substitutes them inside V1 and calls **V1's own `measure()`**, so everything
between the FOV and the fractal features is V1's code, not a replica.

**B1 — border-independent FOV.** Otsu is estimated on the frame with contiguous near-constant edge
rows/columns removed (`CONST_TOL = 2.0` grey levels, `MIN_CONTENT_FRAC = 0.50`), then applied to the
full frame; the same detected constant edges are also excluded from the **returned mask**. All other
steps are V1's, byte for byte: `remove_small_objects`/`remove_small_holes` at `max(64, 0.001·h·w)`,
`binary_closing(disk(3))`, largest-component selection, failure criteria `coverage < 0.15` /
`> 0.985` / `largest/sum < 0.90`. The mask-exclusion half was added after the development check
(10 images, 6 conditions) showed externally added constant padding could still enter the mask, and
before the locked run started — recorded in the manifest, no new constant, nothing tuned on outcomes.

**B2 — border-independent fractal domain.** The same `MultifractalVBMs` estimator, run on a square
window centred on the FOV bounding-box centre, `side = ceil(max(bbox)·(1+2·0.10))` rounded up to 8,
zero outside the frame. Adding constant padding translates the bbox and its centre but not the
window's content, so the domain array is identical.

**Development vs locked evaluation.** The 328-image Task-5B-H/H2 sample was used only to reproduce
the known failure, debug and confirm execution. The decision rests on a **new locked sample of 450
images**, verified disjoint from all 328, stratified over source × geometry × split (plus 270 /
farabi 90 / farfum_rop 90; all five geometries; train 150 / val 150 / test 150). Conditions: the same
8 brightness + 22 border conditions, plus a **6-condition novel border set** frozen before the run
(widths 0.035 and 0.18, grey levels 20 and 90, two asymmetric combinations). Acceptance gate G1–G7
was written into `_private_audit/task5b_h3_manifest.json` before the run and not changed afterwards.
16,650 condition rows, 0 errors. No disease label, no classifier, no `SEG_CURRENT_V1` change, no
feature removed, no regeneration of the 8,870-row table.

**G — FOV robustness, V1 vs V2 (12,600 border rows).**

| endpoint | V1 | V2 |
|---|---|---|
| median FOV Dice | 0.991526 | **0.999894** |
| mean FOV Dice | 0.965228 | **0.988830** |
| 1st-percentile Dice | 0.736531 | 0.782499 |
| minimum Dice | 0.000000 | 0.459247 |
| fraction Dice < 0.99 | 0.4913 | **0.1200** |
| fraction Dice < 0.90 | 0.0994 | **0.0381** |
| relative area, median | +0.002979 | **−0.000090** |
| relative area, p95 | +0.346622 | **+0.068011** |
| centroid displacement, median | 0.002276 | **0.000017** |
| FOV valid → invalid | 30 | **444** |

V2 improves the typical case by an order of magnitude and is uniform across sources and geometries
(V2 median Dice 0.99983–0.99996 everywhere; V1 ranges 0.943 for 1240×1240 to 0.99979 for 1440×1080).
It also **regresses validity**: 444 rows where a valid V1 baseline becomes an invalid V2 FOV, against
30 for V1.

**H — the five FINAL_PRIMARY features, border family.**

| feature | version | median rel | p95 rel | max rel | >1 % | >5 % | >10 % |
|---|---|---|---|---|---|---|---|
| `vessel_density_fov` | V1 | −0.00104 | +0.03685 | +0.43042 | 44.4 % | 19.6 % | 9.3 % |
| `vessel_density_fov` | **V2** | **+0.00008** | **+0.01696** | +0.91074 | **12.2 %** | **5.8 %** | **2.4 %** |
| `skel_density_fov` | V1 | −0.00476 | +0.01819 | +1.91954 | 53.7 % | 38.8 % | 23.8 % |
| `skel_density_fov` | **V2** | **+0.00009** | +0.05224 | +2.35496 | **12.5 %** | **10.5 %** | **7.2 %** |
| `fractal_d0` | V1 | −0.01722 | +0.01761 | +0.11616 | 71.2 % | 18.2 % | 1.7 % |
| `fractal_d0` | **V2** | **0.00000** | **+0.00904** | +0.17054 | **10.1 %** | **1.8 %** | 0.13 % |
| `fractal_d1` | V1 | −0.01710 | +0.03009 | +0.13864 | 74.6 % | 18.1 % | 1.7 % |
| `fractal_d1` | **V2** | **0.00000** | **+0.01148** | +0.17330 | **11.4 %** | **2.0 %** | 0.17 % |
| `fractal_d2` | V1 | −0.01661 | +0.03388 | +0.15812 | 74.8 % | 18.5 % | 1.9 % |
| `fractal_d2` | **V2** | **0.00000** | **+0.01103** | +0.19241 | **11.8 %** | **1.8 %** | 0.18 % |

Every feature improves in the typical case; the density features improve 3.6× and 4.3× in the >1 %
fraction, and the three fractals stop moving at the median **exactly** (`0.00000`). The tails do not
disappear: the worst density errors get slightly larger, because the rows where V2's FOV collapses
produce a much smaller denominator. Under brightness both versions are unchanged and benign (median
`0.00000` for all five).

**I — fractal zero-padding control (mandatory).** The identical binary support and the identical FOV,
translated into larger zero canvases, with no FOV detection involved:

| feature | V1 median \|rel\| | V1 max | V2 median \|rel\| | V2 max |
|---|---|---|---|---|
| `fractal_d0` | 0.02599 | 0.11383 | **0.00000** | 0.03757 |
| `fractal_d1` | 0.02722 | 0.12934 | **0.00000** | 0.04258 |
| `fractal_d2` | 0.02880 | 0.13922 | **0.00000** | 0.04350 |

The canvas mechanism is resolved exactly: with an unchanged support the V2 fractals are bit-identical
under padding, where V1 moved by ~2.6–2.9 % at the median. Residual V2 maxima come only from cases
where the FOV itself changed, which the control does not exercise.

**J — missingness.** Border family: V1 generates 3 finite→NaN transitions, V2 generates **0**. No
baseline NaN among the five features on this sample.

**K — native-image drift.** On unperturbed images V2's FOV equals V1's (median Dice `1.000000`, p05
0.985650, min 0.812880) and validity is unchanged in both directions (0 lost, 0 gained, of 450). The
two density features are unchanged at the median (`0.00000`; >5 % in 1.1 % and 3.6 % of images). The
fractals move by construction, `−0.0346 / −0.0351 / −0.0348` at the median, because the analysis
domain changed; 29–30 % of native images move more than 5 %. **Without expert FOV annotation nothing
here shows V2 is more anatomically correct than V1** — the native fractal shift is a definition
change of the same size as the border effect it was meant to remove, and both readings are reported,
not adjudicated.

**Mechanism of the V2 validity regression — identified, not speculated.** Of the 444 rows, **375 are
`novel_gray90_w2`** and 45 are `dark_gray_40_w2`; the remaining 24 are spread thinly. Re-measuring a
28-row sample gives `coverage_below_15pct` 18 and `fragmented_bright_region` 10, with a median
coverage of 0.1430. The pathway is: a border whose grey level sits **above** the retinal background
(90, and 40 on dark images) survives `green > thr` as a bright ring; `binary_closing(disk(3))`
connects that ring to the optic disc, and the **largest connected component becomes the border ring
rather than the retina**; B1's mask exclusion then deletes exactly that ring, leaving little or
nothing, and the coverage and fragmentation criteria fire. V2's mask exclusion is therefore correct
for its stated purpose and simultaneously **incompatible with a component-selection rule that can
choose the padding**. The residual Dice tails have the same root: `overwrite_all_w2` (0.9342) and
`overwrite_lr_w2` (0.9847) alter the retinal content itself, and `irregular_frame_w3` (0.9870) and
`tb_thick_w3` (0.9875) sit just below the per-condition bar for the same ring-selection reason.

**N/O — gate outcome: 6 of 12 checks pass.**

```
G1_median_dice                     : PASS   (0.999894 >= 0.999)
G1_p01_dice                        : FAIL   (0.782499 <  0.95)
G1_every_condition                 : FAIL   (worst condition median 0.9342 < 0.99)
G2_no_recurrent_invalid            : FAIL   (444 > 5)
G3_vessel_density                  : FAIL   (12.2 % > 5 %)
G3_skel_density                    : FAIL   (12.5 % > 5 %)
G4_canvas_d0 / d1 / d2             : PASS   (median |rel| 0.00000 <= 0.005)
G5_no_new_nan                      : PASS   (0)
G6_subgroups                       : FAIL
G7_native_validity                 : PASS   (0 of 450)
FOV_BORDER_ROBUSTNESS_V2           : FAIL
```

Per the predeclared stop rule nothing was tuned, no V3 was written, the feature table was **not**
regenerated and no classifier was trained. The candidate is a **partial** success and is recorded as
such: the fractal canvas mechanism is solved exactly, the typical FOV and feature error drops by
3.6–10×, and the brightness behaviour is untouched — but the gate fails on the per-condition Dice
floor, on validity, and on the density error fractions, all through one identified interaction between
border exclusion and largest-component selection.

```
TASK5B_H3_CANDIDATE                      : CLINICAL_MEASUREMENT_V2_CANDIDATE
V1_FROZEN                                : YES
V2_PREDECLARED_BEFORE_LOCKED_EVAL        : YES
DEVELOPMENT_SAMPLE_N                     : 328
LOCKED_EVALUATION_SAMPLE_N               : 450
LOCKED_SAMPLE_DISJOINT_FROM_DEVELOPMENT  : YES
FOV_BORDER_ROBUSTNESS_V1                 : FAIL
FOV_BORDER_ROBUSTNESS_V2                 : FAIL
VESSEL_DENSITY_BORDER_PROBLEM_RESOLVED   : NO
SKEL_DENSITY_BORDER_PROBLEM_RESOLVED     : NO
FRACTAL_ZERO_PADDING_D0_RESOLVED         : YES
FRACTAL_ZERO_PADDING_D1_RESOLVED         : YES
FRACTAL_ZERO_PADDING_D2_RESOLVED         : YES
NEW_NAN_FAILURES                         : NO
SOURCE_OR_GEOMETRY_SPECIFIC_FAILURE      : NO   (V2 is uniform; the regression is condition-specific:
                                                grey-90 and grey-40 borders, 420 of 444 rows)
NATIVE_V1_V2_DRIFT_REVIEWED              : YES
CLINICAL_MEASUREMENT_V2_CANDIDATE_STATUS : NOT_SUPPORTED
FULL_8870_REGENERATION_ALLOWED           : NO
DISEASE_MODEL_TRAINING_ALLOWED           : NO
TASK5B_H3_STATUS                         : INCOMPLETE
```

The failure is returned for a separate decision, as the task requires: the next candidate has to fix
component selection against a selectable padding, which is a different change from the one tested
here, and it must be predeclared and re-evaluated on a new locked sample rather than tuned on this
one.

### TASK 5B-H4 — `CLINICAL_MEASUREMENT_V3_CANDIDATE`: padding-aware component selection

V1 and V2 are untouched and reproducible (shas in the H3 block above). V3 inherits B1's threshold and
B2's fractal domain with **no constant changed** (`CONST_TOL 2.0`, `MIN_CONTENT_FRAC 0.50`,
`DOMAIN_MARGIN 0.10`, `DOMAIN_ALIGN 8`), and changes exactly one thing — the order of operations:

```
V2:  threshold -> binarise -> small objects -> small holes -> closing -> label -> largest -> THEN
     zero the detected external band out of the mask
V3:  threshold -> binarise -> raw component count -> ZERO THE DETECTED EXTERNAL BAND -> small
     objects -> small holes -> closing -> label -> largest
```

The external band is the frame minus the content box, i.e. exactly the four detected constant edge
bands. Removing it *before* morphology and labelling means a threshold-positive padding ring can
never be a candidate component and can never inflate the `frag` denominator. Fallback: an untrimmed
frame has no external band and V3 selects exactly as V2/V1. Failure reasons and thresholds are V1's,
unchanged; no new constant exists anywhere.

Locked sample: **450 images, disjoint from the 328 development and the 450 H3 images**, stratified
over source × geometry × split. Conditions: the same 8 brightness + 22 border + 6 H3 novel, plus a
**6-condition challenge set** whose grey levels are placed at −30/−10/+10/+30 relative to each
image's own background median, one asymmetric and one at 0.18·min width. V3 is evaluated on all 42
conditions; V1 and V2 on a predeclared 8-condition mechanism subset on the same new sample, with
their full-set numbers taken from the H3 locked run. `overwrite_*` was predeclared CONTENT_ALTERING
(it overwrites retinal pixels) and is reported separately. Gate G1–G11 frozen in
`task5b_h4_manifest.json` before the run. 18,900 rows, 0 errors.

**The known mechanism is reproduced and then resolved.**

| mechanism subset (n = 3,600; grey-40/90, challenge greys) | V1 | V2 | **V3** |
|---|---|---|---|
| median FOV Dice | 0.989966 | 0.999618 | **0.999940** |
| 1st-percentile Dice | 0.000000 | 0.000000 | **0.943752** |
| minimum Dice | 0.000000 | 0.000000 | **0.756814** |
| fraction Dice < 0.90 | 8.5 % | 3.31 % | **0.67 %** |
| FOV valid → invalid | **3600** | **3600** | **2** |

V1 and V2 invalidate **every** grey-border row in the subset. V3 invalidates two. Of V2's 2,731
component-destroyed rows, V3 returns a valid FOV for **2,729** — with median Dice 0.999937 against
the baseline. Across all 42 conditions the diagnostic is unambiguous:

```
V3 rows with any FOV pixel inside the detected padding : 0 of 15,300
V3 grey rows (n = 4,050): median Dice 0.999940, invalid 2   (0.05 %)
V3 all-border: median Dice 0.999931, p01 0.777813, min 0.162949, invalid 20
V3 by source   0.999858 - 0.999958      V3 by geometry 0.999765 - 1.000000
```

**Five FINAL_PRIMARY features under border (n = 15,300).**

| feature | median rel | p95 rel | max rel | >1 % | >2 % | >5 % | >10 % |
|---|---|---|---|---|---|---|---|
| `vessel_density_fov` | +0.00007 | +0.00508 | +4.1445 | 10.8 % | 8.9 % | 5.3 % | 2.6 % |
| `skel_density_fov` | +0.00008 | +0.00607 | +10.274 | 12.3 % | 11.4 % | 9.3 % | 6.5 % |
| `fractal_d0` | **0.00000** | +0.00813 | +0.4849 | 7.2 % | 3.6 % | 0.93 % | 0.12 % |
| `fractal_d1` | **0.00000** | +0.00837 | +0.4482 | 8.3 % | 4.7 % | 1.08 % | 0.25 % |
| `fractal_d2` | **0.00000** | +0.00929 | +0.4590 | 9.0 % | 4.8 % | 1.10 % | 0.27 % |

Zero-padding control (identical support, larger canvases, no FOV detection):

```
fractal_d0  V1 median|rel| 0.03133  ->  V3 0.00000  (max 0.00897)
fractal_d1  V1 median|rel| 0.02998  ->  V3 0.00000  (max 0.02004)
fractal_d2  V1 median|rel| 0.03180  ->  V3 0.00000  (max 0.02590)
```

Native preservation (450 unperturbed images): FOV Dice V3 vs V1 median `1.000000`, p05 0.984481;
validity unchanged in both directions (0 lost, 0 gained); both density features unchanged at the
median (`0.00000`); fractals move `−0.0349 / −0.0356 / −0.0367` at the median because the analysis
domain changed — a definition change of the same size as the artefact it removes. Without expert FOV
annotation this is reported, not adjudicated.

**Gate: 7 of 13 pass → `FOV_BORDER_ROBUSTNESS_V3 = FAIL`.**

```
G1_median_dice             PASS  0.999931
G2_p01_dice                FAIL  0.777813 < 0.95
G3_plausible_condition     FAIL  irregular_frame_w3 0.9855, tb_thick_w3 0.9862 < 0.99
G4_valid_to_invalid        FAIL  20 > 5
G5_grey_component_failure  PASS  2 of 4050 = 0.05 %
G6_vessel_density          FAIL  10.8 % > 5 %
G7_skel_density            FAIL  12.3 % > 5 %
G8_canvas D0/D1/D2         PASS  0.00000 median
G9_no_new_nan              PASS  12 finite->NaN of 15300 (bound 76.5)
G10_subgroups              FAIL
G11_native_validity        PASS  0 of 450
```

**Residual mechanism, decomposed — two specific families, nothing else.**

| subset | n | median Dice | p01 Dice | min | invalid | >1 % vessel | >1 % skel |
|---|---|---|---|---|---|---|---|
| all border | 15,300 | 0.999931 | 0.7778 | 0.1629 | 20 | 10.8 % | 12.3 % |
| plausible padding only | 14,400 | 0.999944 | 0.8353 | 0.5286 | 6 | 6.9 % | 7.9 % |
| excluding the two w3 asymmetric pads | 13,500 | **0.999952** | **0.9027** | 0.5286 | **5** | **3.98 %** | **4.77 %** |
| `irregular_frame_w3` + `tb_thick_w3` | 900 | 0.985918 | 0.6455 | 0.5286 | 1 | 50.4 % | 55.3 % |
| content-altering `overwrite_*` | 900 | 0.953727 | 0.3858 | 0.1629 | **14** | 73.1 % | 81.3 % |

* **14 of the 20 invalid rows and 727 of the 1,742 rows below Dice 0.99 are the content-altering
  `overwrite_*` conditions**, which paint black bars over retinal pixels. The retina genuinely
  changes there, so a changed FOV is the correct answer, not a defect. They were predeclared out of
  the plausible-border floor and are the single largest contributor to the failing density fractions.
* **The remaining density failure comes from two conditions**, `irregular_frame_w3` and
  `tb_thick_w3`, whose largest band is `4·w3 = 0.48·min` — a bottom band of nearly half the image
  height. On images whose retina reaches the frame edge the content box then clips retinal content,
  and the FOV changes by ~1.4 % at the median with a tail to Dice 0.53.
* **With those two families set aside the density endpoints are 3.98 % and 4.77 %**, i.e. inside the
  predeclared 5 % bar, and p01 Dice is 0.9027. That is reported for a separate decision, **not** used
  to change the gate: the gate was frozen before the run and V3 is recorded as failing it.

```
TASK5B_H4_CANDIDATE                        : CLINICAL_MEASUREMENT_V3_CANDIDATE
LOCKED_SAMPLE_N                            : 450
LOCKED_SAMPLE_DISJOINT                     : YES
KNOWN_BORDER_COMPONENT_FAILURE_REPRODUCED  : YES   (V1 and V2 invalidate 3600/3600 mechanism rows)
V3_PREDECLARED                             : YES
GREY_BORDER_COMPONENT_FAILURE_RESOLVED     : YES   (2 invalid of 4050; 2729 of V2's 2731 recovered)
FOV_BORDER_ROBUSTNESS_V3                   : FAIL  (7 of 13)
VESSEL_DENSITY_BORDER_PROBLEM_RESOLVED     : NO    (10.8 % > 5 %; 3.98 % if the two extreme pads
                                                    and the content-altering family are excluded)
SKEL_DENSITY_BORDER_PROBLEM_RESOLVED       : NO    (12.3 % > 5 %; 4.77 % on the same exclusion)
FRACTAL_D0_ZERO_PADDING_RESOLVED           : YES
FRACTAL_D1_ZERO_PADDING_RESOLVED           : YES
FRACTAL_D2_ZERO_PADDING_RESOLVED           : YES
NEW_NAN_FAILURES                           : NO    (12 isolated rows, bound was 76.5)
NATIVE_DRIFT_REVIEWED                      : YES
CLINICAL_MEASUREMENT_V3_CANDIDATE_STATUS   : NOT_SUPPORTED
FULL_8870_REGENERATION_ALLOWED             : NO
DISEASE_MODEL_TRAINING_ALLOWED             : NO
TASK5B_H4_STATUS                           : COMPLETE
```

Per the stop rule nothing was tuned, no V4 was written, the 8,870-row table was not regenerated and
no classifier was trained. The exact residual mechanism is returned: the two extreme asymmetric
padding widths, and the content-altering overwrite family that changes the retina itself.

### TASK 5B-H5 — adjudication of the H4 residual (analysis only)

No new candidate, no V3 change, no re-run of the 450 × 42 battery, no table regeneration, no
classifier. This task only re-partitions the **existing** H4 rows and probes the residual cases.

**A/B — semantic class from the transformation definition, then verified by pixel mapping.** 42
conditions: **40 CONTENT_PRESERVING**, **2 CONTENT_ALTERING**. Verified on 12 images × 34 border
conditions = 408 checks by recovering the mapped content region and comparing arrays:

* all **32 padding conditions recover every original pixel exactly** — `modified_original_px = 0`,
  `lost_original_px = 0`;
* only `overwrite_all_w2` and `overwrite_lr_w2` modify original pixels (39,100 and 17,664 pixels of
  a 512-frame) — CONTENT_ALTERING, and their numbers are an occlusion test, **not** an invariance test;
* the 8 brightness conditions are CONTENT_PRESERVING by definition (global photometric change).

**E — the two extreme pads are content-preserving.** `irregular_frame_w3` and `tb_thick_w3` recover
every original pixel exactly (0 modified, 0 lost, canvas grows by 230×230 and 276×92). Their failures
are therefore **legitimate FOV invariance failures and are not dismissed**. They are not edge cases
either: `density >5 %` in 47.3 % and 46.7 % of their rows.

**C — V3 metrics split by class.**

| | CONTENT_PRESERVING | CONTENT_ALTERING (occlusion test) |
|---|---|---|
| conditions / pairs | 40 / 18,000 | 2 / 900 |
| median FOV Dice | **0.999957** | 0.953727 |
| p01 / min FOV Dice | 0.833822 / 0.528559 | 0.385834 / 0.162949 |
| Dice < 0.99 / < 0.90 | 8.51 % / 2.24 % | 80.8 % / 26.1 % |
| FOV valid → invalid | **6** | 14 |
| `vessel_density_fov` >1 % / >5 % / >10 % | 7.84 % / 3.03 % / 1.48 % | 73.1 % / 43.8 % / 20.6 % |
| `skel_density_fov` >1 % / >5 % / >10 % | 10.29 % / 5.82 % / 4.13 % | 81.3 % / 74.4 % / 46.3 % |
| `fractal_d0/d1/d2` median rel | **0.00000** each | −0.0025 / −0.0031 / −0.0030 |
| finite → NaN | **0** | 12 |

**D — residual true failures.** Union of (Dice < 0.95) ∪ (valid→invalid) ∪ (density |rel| > 5 %) over
content-preserving pairs: **1,055 of 18,000 = 5.86 %**. Trigger decomposition: **0 rows fail on Dice
alone** — every Dice < 0.95 row also has density > 5 %; 6 rows are validity failures; 293 fail on
density only. By condition: `irregular_frame_w3` 213 and `tb_thick_w3` 210 dominate, then brightness
×1.80 128, `asymmetric_black_w3` 119, `lr_thick_w3` 86, ×1.50 79, ×1.30 40, ×1.15 22, ×0.25 17, and
~13 each for the six grey conditions. By source `plus` 815 / farfum_rop 205 / farabi 35; by geometry
1240×1240 470, 640×480 326, 1600×1200 205, 1280×960 35, 1440×1080 19.

**The residual mechanism is a frame-area-dependent constant, not an unfixable border effect.**
Mechanism probe on 90 residual rows:

```
rows where V3 trimmed ORIGINAL content beyond the added padding : 13 of 90
rows where the detected content box equals the added padding exactly : 77 of 90
rows where min_size = max(64, 0.001*h*w) CHANGED with the canvas : 60 of 90
     median ratio 1.22, max 2.55        Otsu shift median -28.6 grey levels, max |shift| 59.5
```

Two frame-area dependencies survive in V3:

1. **`MIN_CONTENT_FRAC = 0.50` falls back to the full-frame estimate.** When the padding is large
   enough that the retained content is less than half the padded frame, `content_bounds` returns
   `trimmed = False`, the border exclusion is silently skipped, and Otsu runs on the padded frame
   again — reintroducing the exact border sensitivity V2/V3 were built to remove. That is what the
   −28.6 to −59.5 grey-level threshold shifts are: `lr_thick_w3`, `tb_thick_w3`, `irregular_frame_w3`
   and `asymmetric_black_w3` on 512-frames pad 244 px on one axis, leaving 47 % content.
2. **`min_size = max(64, int(0.001*h*w))` uses the file frame area.** Padding enlarges `h*w`, so the
   small-object and small-hole thresholds grow with the canvas (median 1.22×, max 2.55×), removing
   structures that survive at baseline.

The brightness tail is a separate, honest third mechanism: global photometric change is
content-preserving but clipping is not information-preserving — `sat_hi` reaches 10.5 % at ×1.80 and
the threshold moves `+4.0` there (`min_size` unchanged at 1.00). That is an exposure-range statement,
not a border defect.

**F — fractal status confirmed from the existing H4 control** (no re-run): D0/D1/D2 all
`V1 3.13 % / 3.00 % / 3.18 % → V3 0.00000` median |rel|, max 0.90 % / 2.00 % / 2.59 % — resolved.

```
CONTENT_PRESERVING_CONDITIONS_N          : 40
CONTENT_ALTERING_CONDITIONS_N            : 2
CONTENT_PRESERVING_PAIRS_N               : 18000
V3_CONTENT_PRESERVING_MEDIAN_DICE        : 0.999957
V3_CONTENT_PRESERVING_P01_DICE           : 0.833822
V3_CONTENT_PRESERVING_MIN_DICE           : 0.528559
V3_CONTENT_PRESERVING_VALID_TO_INVALID_N : 6
VESSEL_DENSITY_CONTENT_PRESERVING_GT1PCT : 0.0784
VESSEL_DENSITY_CONTENT_PRESERVING_GT5PCT : 0.0303
SKEL_DENSITY_CONTENT_PRESERVING_GT1PCT   : 0.1029
SKEL_DENSITY_CONTENT_PRESERVING_GT5PCT   : 0.0582
TRUE_RESIDUAL_FOV_FAILURES_N             : 1055   (5.86 % of content-preserving pairs)
TRUE_RESIDUAL_FAILURE_MECHANISM          : frame-area-dependent constants — MIN_CONTENT_FRAC 0.50
                                           silently falls back to full-frame Otsu when the pad
                                           leaves <50 % content, and min_size = 0.001*h*w grows
                                           with the padded canvas; plus photometric clipping at
                                           x1.50-x1.80
FRACTAL_D0_CANVAS_PROBLEM_RESOLVED       : YES
FRACTAL_D1_CANVAS_PROBLEM_RESOLVED       : YES
FRACTAL_D2_CANVAS_PROBLEM_RESOLVED       : YES
V3_MEASUREMENT_PATH_STATUS               : REQUIRES_TARGETED_FIX
FULL_8870_REGENERATION_ALLOWED           : NO
TASK5B_H5_STATUS                         : COMPLETE
```

`TASK5B_H4 = FAIL` stands as recorded. H5 answers only the different question — and the answer is
that the residual is **not** rare and **is** systematic: the failure appears exactly when padding is
large enough to trip one of two frame-area-dependent constants, and it disappears when they are held
to the content (3.98 % / 4.77 % density error and p01 0.9027 on the subset that avoids them). One
targeted fix is therefore identified and predeclarable: **make both constants content-relative
(forbid the MIN_CONTENT_FRAC fallback from re-admitting the border, and scale `min_size` by the
content area, not the file canvas)**. That fix is NOT implemented here.

### TASK 5B-H6 — `CLINICAL_MEASUREMENT_V4_CANDIDATE`: content-relative FOV constants

V1, V2 and V3 are untouched and reproducible (shas in the blocks above). V4 is V3 with **exactly two
changes**, both named by H5:

* **Change 1 — no canvas-fraction fallback.** `content_bounds_v4` has no fraction test. Otsu is
  computed on the detected content rectangle whenever one exists; the only new failure is
  `no_content_region`, for a frame whose content rectangle is empty. No new tuned threshold.
* **Change 2 — content-relative morphology size.** `min_size = max(64, int(0.001·content_h·content_w))`.
  The `0.001` rule and the `64` floor are unchanged; only the area is the content domain.

Everything else is V3: `CONST_TOL 2.0`, the Otsu implementation, the external band removed before
morphology and labelling, `remove_small_objects`/`holes`, `binary_closing(disk(3))`, largest-component
selection, the validity rules, the fractal window, the D0/D1/D2 estimator, the density formulas and
`SEG_CURRENT_V1`.

**D1/D2 unit invariance — PASS.** The same retinal content embedded in pads of 0, 5, 20, 60, 140 and
300 px, canvas growing 64×80 → 664×680:

| | pad 0 | 5 | 20 | 60 | 140 | 300 |
|---|---|---|---|---|---|---|
| content pixels identical to base | yes | yes | yes | yes | yes | yes |
| **V4** Otsu threshold | 40.3731 | 40.3731 | 40.3731 | 40.3731 | 40.3731 | 40.3731 |
| **V4** `min_size` | 64 | 64 | 64 | 64 | 64 | 64 |
| V3 `trimmed` flag | True | True | **False** | **False** | **False** | **False** |
| V3 `min_size` | 64 | 64 | 64 | 64 | **123** | **451** |

`UNIT_CONTENT_THRESHOLD_INVARIANCE = PASS`, `UNIT_CONTENT_MINSIZE_INVARIANCE = PASS`, and D2 shows the
full-frame dimensions change while both quantities stay constant. V3 by contrast loses its border
exclusion at pad ≥ 20 and its `min_size` grows 64 → 123 → 451.

**E — development check on the known H5 cases** (10 images, 70 rows). V3 fell back to full-frame Otsu
in **20 of 70** rows — 10/10 on each of `irregular_frame_w3` and `tb_thick_w3` — with threshold
40.17/40.47 against V4's 48.97; and V3's `min_size` was canvas-scaled in **70 of 70** (239–455)
against V4's 196. Both H5 mechanisms were exercised, and V4 removes both.

**F/H — locked evaluation.** New sample of **150 images**, disjoint from the 328 + 450 + 450 previous
samples, stratified over source × geometry × split; targeted condition set of 7 perturbations plus a
native control; gate frozen in `task5b_h6_manifest.json` before the run. 1,050 perturbed pairs.

| endpoint | V3 | **V4** |
|---|---|---|
| median FOV Dice | 0.999857 | **0.999951** |
| p01 FOV Dice | 0.675896 | **0.983288** |
| minimum FOV Dice | 0.447492 | 0.524130 |
| Dice < 0.99 / < 0.95 | 230 / 168 | **20 / 6** |
| relative FOV area, median / p95 | −0.000064 / +0.286955 | −0.000090 / **+0.000000** |
| centroid displacement, median | 0.000027 | **0.000010** |
| `irregular_frame_w3` median Dice | 0.969981 | **0.999952** |
| `tb_thick_w3` median Dice | 0.971376 | **0.999952** |
| `vessel_density_fov` >1 % / >5 % | 21.1 % / 10.7 % | **1.52 % / 0.38 %** |
| `skel_density_fov` >1 % / >5 % | 23.7 % / 19.5 % | **3.05 % / 0.76 %** |
| `fractal_d0/d1/d2` >5 % | 1.4 / 2.2 / 2.5 % | **0.0 / 0.57 / 0.57 %** |
| NaN transitions | 0 | 0 |
| **FOV valid → invalid** | **4** | **16** |

By source and geometry V4 is uniform (medians 0.99987–1.000000). The zero-padding control stays
resolved (`V4 0.00000` median |rel| for D0/D1/D2, max 0.19 / 0.13 / 0.15 %). On the **native**
control V4 is *identical* to V3 on all 150 images — FOV Dice `1.000000` (p05 and min also 1.000000),
validity 0 lost / 0 gained, and all five features `median rel +0.00000` with `frac>5 % = 0.0000`.

**I — targeted mechanism check.** Both H5 mechanisms are gone:

```
V3 full-frame fallback exercised            : 349 of 1050 rows
V4 fallback path                            : none exists; content threshold used in 1050/1050
V4 threshold == V3 threshold where V3 did not fall back : 701 of 701
V3-style canvas-scaled min_size             : 1050 of 1050 rows
V4 min_size range                           : 191..262  (baseline range 191..262; the canvas
                                              formula would give 239..455)
V4 rows with any FOV pixel inside the padding : 0
FULL_FRAME_OTSU_FALLBACK_RESIDUAL_N          : 0
CANVAS_DEPENDENT_MINSIZE_RESIDUAL_N          : 0
```

**The one residual is a third frame-area dependence, and it is in the validity rule, not the FOV.**
All 16 V4 valid→invalid rows carry reason `coverage_below_15pct`, on the largest pads
(`irregular_frame_w3` 7, `tb_thick_w3` 4, `asymmetric_black_w3` 3, `novel_black_w5` 2), and their FOV
Dice is **0.9993–0.9999** — the detected field of view is correct. `fov_coverage_fraction` is
`fov.mean()` over the **padded file frame**, so a large pad dilutes coverage past its own 0.15 rule
even though nothing is wrong with the measurement. V4's better threshold is what exposes it: V3's
fallback produced a *larger* FOV, which kept those rows above 0.15 for the wrong reason.

**K — photometric clipping** (`x1.50`–`x1.80` saturation) is recorded as a separate known
operating-range limitation and is not part of this border-specific decision.

```
TASK5B_H6_CANDIDATE                        : CLINICAL_MEASUREMENT_V4_CANDIDATE
LOCKED_SAMPLE_N                            : 150
V4_PREDECLARED                             : YES
UNIT_CONTENT_THRESHOLD_INVARIANCE          : PASS
UNIT_CONTENT_MINSIZE_INVARIANCE            : PASS
FULL_FRAME_OTSU_FALLBACK_RESIDUAL_N        : 0
CANVAS_DEPENDENT_MINSIZE_RESIDUAL_N         : 0
FOV_CONTENT_PRESERVING_MEDIAN_DICE         : 0.999951
FOV_CONTENT_PRESERVING_P01_DICE            : 0.983288
FOV_CONTENT_PRESERVING_MIN_DICE            : 0.524130
VALID_TO_INVALID_N                         : 16      (gate required <= 2)
VESSEL_DENSITY_GT5PCT                      : 0.0038
SKEL_DENSITY_GT5PCT                        : 0.0076
FRACTAL_D0_ZERO_PADDING_RESOLVED           : YES
FRACTAL_D1_ZERO_PADDING_RESOLVED           : YES
FRACTAL_D2_ZERO_PADDING_RESOLVED           : YES
SOURCE_OR_GEOMETRY_FAILURE                 : NO
V4_STATUS                                  : NOT_SUPPORTED   (10 of 12 gate checks; G5 fails)
FULL_8870_REGENERATION_ALLOWED             : NO
DISEASE_MODEL_TRAINING_ALLOWED             : NO
TASK5B_H6_STATUS                           : COMPLETE
```

Per the stop rule nothing was tuned, no V5 was written, the 8,870-row table was not regenerated and no
classifier was trained. `TASK5B_H4 = FAIL` and `TASK5B_H5 = REQUIRES_TARGETED_FIX` stand as recorded.
The remaining mechanism is now isolated to a single line of the validity criterion — the
`coverage < 0.15` rule divides by the padded canvas — and it is the same *class* of defect as the two
this task fixed, which is why it is reported for one more predeclared, content-relative change rather
than patched here.

### TASK 5B-H7 — `CLINICAL_MEASUREMENT_V5_CANDIDATE`: content-relative FOV coverage validity

V1–V4 untouched and reproducible. V5 is V4 with **exactly one line changed**:

```
V4:  coverage = fov_pixels / (full_canvas_h * full_canvas_w)
V5:  coverage = fov_pixels / (content_h * content_w)      # the rectangle V4 already derives
```

Thresholds `0.15` / `0.985` / `0.90`, the FOV mask generation, Otsu, morphology, component
selection, the fractal domain, the D0/D1/D2 estimator and all five biomarker formulas are unchanged.
No new constant is introduced.

**D — direct unit invariance.** The same retinal content padded by 0/5/20/60/140/300 px, canvas
growing `72×88 → 672×688`:

| pad | content | FOV px | V4 coverage | **V5 coverage** | V4 valid | **V5 valid** |
|---|---|---|---|---|---|---|
| 0 | 72×88 | 2809 | 0.443340 | **0.443340** | True | True |
| 5 | 72×88 | 2809 | 0.349552 | **0.443340** | True | True |
| 20 | 72×88 | 2809 | 0.195940 | **0.443340** | True | True |
| 60 | 72×88 | 2809 | 0.070338 | **0.443340** | **False** | True |
| 140 | 72×88 | 2809 | 0.021685 | **0.443340** | **False** | True |
| 300 | 72×88 | 2809 | 0.006076 | **0.443340** | **False** | True |

Content rectangle identical after mapping, FOV pixels identical, V5 coverage and validity identical
while the canvas changes. `CONTENT_RELATIVE_COVERAGE_INVARIANCE = PASS`.

**E — replay of the 16 H6 `valid→invalid` rows** (development cases, not new validation): V4 coverage
0.0893–0.1494 (all below 0.15) → V5 coverage 0.2069–0.3551; **V4 invalid → V5 valid in 16 of 16**, FOV
masks bit-identical, max feature delta `0.0`.

**F/G — locked evaluation.** New sample of **100 images**, disjoint from all previous samples
(1,378 excluded), stratified over source × geometry × split; conditions: native plus
`irregular_frame_w3`, `tb_thick_w3`, `novel_black_w5`, `asymmetric_black_w3`, `dark_gray_40_w2`,
`novel_gray90_w2`. 700 pairs (600 perturbed + 100 native).

```
FOV mask bitwise identical, V4 vs V5        : 700 of 700
max |V4 - V5| over the five features        : 0.000e+00      (tolerance 1e-12)
new NaN (V4 finite -> V5 NaN)               : 0
```

| endpoint | V4 | **V5** |
|---|---|---|
| perturbed rows failing `coverage_below_15pct` | 22 | **12** |
| perturbed rows failing on baseline-valid images | **10** | **0** |
| coverage change caused by the perturbation, median | −0.262020 | **−0.000056** |
| coverage change, p95 / max | −0.080907 / −0.020920 | **+0.000000 / +0.022035** |
| `coverage_above_98pct` | 0 | 0 |

**Disclosed reporting correction.** The verdict block printed by the first scripted gate said 5/9.
Two arithmetic defects in that reporting code were found and corrected from the same saved rows, and
both readings are recorded rather than only the favourable one:

1. the transition counters compared *V4-perturbed vs V5-perturbed* instead of *baseline vs
   perturbed*, and the printed labels were swapped. Corrected: **V5 valid→invalid = 0**, V4
   valid→invalid = 10 (V5 recovers exactly the 10 rows V4 broke);
2. gate items 5/7/8 used an absolute validity fraction, which counts images that are already invalid
   in the **native, unperturbed** state as border failures. There are exactly **two** such images in
   the 100 (`…S02_5` and `…S02_6`, both `plus`, 640×480), with native V5 coverage `0.1325` and
   `0.1171` and `fov_px` 25,390 and 22,757 against a cohort median of 139,741 — a genuinely small
   field of view, six times smaller than typical. Their coverage is **identical** before and after
   every perturbation, so padding does not cause their failures. Excluding them, every source and
   every geometry has V5 validity `1.0000`.

With the predeclared wording measured correctly, all nine gate items pass: masks identical (1),
features within tolerance (2), unit invariance (3), H6 failures eliminated 16/16 (4), zero
canvas-attributable `coverage_below_15pct` (5), valid→invalid 0 ≤ 1 (6), no source-specific
systematic failure (7), no geometry-specific systematic failure (8), no new NaN (9), no post-hoc
tuning (10) — nothing was retuned, only the reporting arithmetic was fixed.

```
TASK5B_H7_CANDIDATE                  : CLINICAL_MEASUREMENT_V5_CANDIDATE
LOCKED_SAMPLE_N                      : 100
V5_PREDECLARED                       : YES
FOV_MASK_V4_V5_IDENTICAL             : YES  (700 of 700)
PRIMARY_FEATURES_V4_V5_EQUIVALENT    : YES
MAX_PRIMARY_FEATURE_DELTA            : 0.000e+00
CONTENT_RELATIVE_COVERAGE_INVARIANCE : PASS
H6_COVERAGE_FAILURES_REPLAYED        : 16
H6_COVERAGE_FAILURES_RESOLVED        : 16
LOCKED_VALID_TO_INVALID_N            : 0
LOCKED_COVERAGE_BELOW_15PCT_N        : 12   (all 12 from 2 images already invalid natively;
                                            0 attributable to the perturbation)
NEW_NAN_FAILURES                     : NO
SOURCE_OR_GEOMETRY_FAILURE           : NO   (1.0000 per source and per geometry on baseline-valid
                                            images; the 2 native low-coverage images are reported,
                                            not hidden)
CLINICAL_MEASUREMENT_V5_STATUS       : SUPPORTED
FULL_8870_REGENERATION_ALLOWED       : YES
DISEASE_MODEL_TRAINING_ALLOWED       : NO
TASK5B_H7_STATUS                     : COMPLETE
```

The 8,870-row table was **not** regenerated in this task. `TASK5B_H4 = FAIL` and
`TASK5B_H5 = REQUIRES_TARGETED_FIX` stand as recorded. The photometric clipping range
(`x1.50`–`x1.80`) remains a separate, unfixed operating-range limitation and is not part of this
border decision. The next task must independently validate and freeze this measurement generation
before Task 6.

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

3. **`vessel_density_fov`, a `FINAL_PRIMARY` feature, is border-sensitive (Task 5B-H).** It is
   exactly invariant to multiplicative brightness change while no pixel clips, but a synthetic black
   border moves the Otsu threshold by 20–55 %, grows the detected FOV up to 2.5×, and changes the
   density by more than 1 % in ~43–61 % of (image, border) pairs and by more than 10 % in up to
   12.6 %. `FINAL_PRIMARY_REQUIRES_REVIEW = YES`; the feature was **not** demoted and the FOV rule
   was **not** redesigned. This constrains any model whose inputs pass through an export, letterbox
   or border-adding step, and must be resolved before disease-model training.
4. **The FOV fragility is not confined to that one feature (Task 5B-H2).** All five `FINAL_PRIMARY`
   features move under the same 22 border conditions, through three separate mechanisms:
   `skel_density_fov` is a pure FOV-area denominator (analytic match to 1e-16, >5 % error in 42.1 %
   of rows), `vessel_density_fov` combines a denominator and an inclusion term that partially cancel
   (+219 % worst case), and `fractal_d0/d1/d2` shift by ~1.6–2.8 % median because the multifractal
   estimator responds to the **canvas extent** — a zero-padded embed of the *identical* binary
   support reproduces the effect without any FOV involvement. Brightness perturbs none of the five
   (median relative error exactly 0.00000, no new NaN anywhere). `FOV_REDESIGN_REQUIRED_BEFORE_
   TRAINING = YES`.

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

Task 5B-H closure, added afterwards and separately gated:

| # | requirement | status |
|---|---|---|
| 1 | existing FOV implementation frozen before testing | ✓ `retinal_fov`, sha `6aea0d68…` |
| 2 | deterministic project sample documented | ✓ 328 images, `N ≥ 300` |
| 3 | brightness extremes tested | ✓ 8 conditions, ×0.25–×1.80 |
| 4 | multiple black-border styles tested | ✓ 22 conditions, 10 styles |
| 5 | FOV mask stability quantified | ✓ Dice, area, centroid, components, validity |
| 6 | `vessel_density_fov` consequence quantified | ✓ median/p95/max and 1/2/5/10 % fractions |
| 7 | source and geometry subgroup results reported | ✓ |
| 8 | strongest failure cases inspected | ✓ mechanism table + QC montages |
| 9 | no threshold tuned post hoc | ✓ predeclared classes unchanged |
| 10 | consequence for FINAL_PRIMARY stated | ✓ `REQUIRES_REVIEW = YES` |
| 11 | no disease classifier trained | ✓ |

```
FOV_STRESS_TEST_EXECUTED          : YES
FOV_SAMPLE_N                      : 328   (325 baseline-valid)
BRIGHTNESS_CONDITIONS_N           : 8
BORDER_CONDITIONS_N               : 22
BASELINE_FOV_VALID_N              : 325
BRIGHTNESS_MEDIAN_FOV_DICE        : 1.000000
BRIGHTNESS_MIN_FOV_DICE           : 0.622341
BORDER_MEDIAN_FOV_DICE            : 0.984718
BORDER_MIN_FOV_DICE               : 0.317889
BRIGHTNESS_MEDIAN_DENSITY_REL_DELTA : 0.00000
BRIGHTNESS_P95_DENSITY_REL_DELTA    : 0.001742
BRIGHTNESS_MAX_DENSITY_REL_DELTA    : 0.072794
BORDER_MEDIAN_DENSITY_REL_DELTA     : -0.001184
BORDER_P95_DENSITY_REL_DELTA        : 0.051578
BORDER_MAX_DENSITY_REL_DELTA        : 2.187549
FOV_FAILURES_BASELINE             : 3
FOV_FAILURES_PERTURBED            : 37
FOV_ROBUSTNESS                    : FAIL
VESSEL_DENSITY_FOV_FINAL_PRIMARY  : REQUIRES_REVIEW
FINAL_PRIMARY_REQUIRES_REVIEW     : YES
TASK5B_H_CLOSURE                  : PASS
```

`TASK5B_H_CLOSURE = PASS` means the closure task was executed against all eleven of its own
requirements. It does **not** mean the FOV passed: `FOV_ROBUSTNESS = FAIL` is the finding.

Task 5B-H2 then measured how far that finding propagates. Every one of the five `FINAL_PRIMARY`
features is affected under border perturbations through three different mechanisms, and none of the
five is affected by brightness perturbations (median relative error exactly `0.00000`):

| feature | status | border median rel | border p95 rel | border max rel | >5 % of rows |
|---|---|---|---|---|---|
| `vessel_density_fov` | `SEVERELY_BORDER_SENSITIVE` | −0.00124 | +0.05097 | +2.1876 | 19.9 % |
| `skel_density_fov` | `SEVERELY_BORDER_SENSITIVE` | −0.01676 | +0.02532 | +4.2915 | 42.1 % |
| `fractal_d0` | `BORDER_SENSITIVE` | −0.01782 | +0.01775 | +0.1196 | 21.2 % |
| `fractal_d1` | `BORDER_SENSITIVE` | −0.01702 | +0.03649 | +0.1411 | 22.8 % |
| `fractal_d2` | `BORDER_SENSITIVE` | −0.01631 | +0.04097 | +0.1607 | 23.2 % |

```
TASK5B_H2_SAMPLE_N                    : 328
FINAL_PRIMARY_FEATURE_N               : 5
PRIMARY_FEATURES_AFFECTED_N           : 5
PRIMARY_FEATURES_SEVERELY_AFFECTED_N  : 2
FOV_FAILURE_PROPAGATES_TO_FRACTALS    : YES
FOV_REDESIGN_REQUIRED_BEFORE_TRAINING : YES
FINAL_PRIMARY_REQUIRES_REVIEW         : YES
TASK5B_H2_STATUS                      : COMPLETE
```

No FOV algorithm was redesigned, no feature was demoted or removed, and no classifier was trained.
The redesign is now a required, separately-decided step before disease-model training.

Task 5B-H3 then built and tested **one** predeclared candidate against that requirement, on a new
locked sample of 450 images disjoint from the 328 development images, with the acceptance gate
written into the manifest before the run. Result: **6 of 12 gate checks pass**, so the candidate is
not supported and nothing was regenerated.

| endpoint | V1 | V2 candidate |
|---|---|---|
| border median FOV Dice | 0.991526 | **0.999894** |
| border fraction Dice < 0.99 | 49.1 % | **12.0 %** |
| border FOV valid → invalid | 30 | **444** |
| `vessel_density_fov` border >1 % | 44.4 % | **12.2 %** |
| `skel_density_fov` border >1 % | 53.7 % | **12.5 %** |
| fractal border median rel | −1.7 % | **0.00000** |
| zero-padding control, median \|rel\| | 2.60–2.88 % | **0.00000** |
| new finite→NaN, border | 3 | **0** |

```
TASK5B_H3_CANDIDATE                      : CLINICAL_MEASUREMENT_V2_CANDIDATE
FOV_BORDER_ROBUSTNESS_V2                 : FAIL   (6 / 12 gate checks)
FRACTAL_ZERO_PADDING_D0/D1/D2_RESOLVED   : YES / YES / YES
VESSEL_DENSITY_BORDER_PROBLEM_RESOLVED   : NO
SKEL_DENSITY_BORDER_PROBLEM_RESOLVED     : NO
CLINICAL_MEASUREMENT_V2_CANDIDATE_STATUS : NOT_SUPPORTED
FULL_8870_REGENERATION_ALLOWED           : NO
DISEASE_MODEL_TRAINING_ALLOWED           : NO
FINAL_PRIMARY_REQUIRES_REVIEW            : YES
TASK5B_H3_STATUS                         : INCOMPLETE
```

Measured partial success, with the residual failure traced to one interaction: excluding the
detected constant border from the FOV mask is incompatible with a largest-component rule that can
select the border ring itself, which is what invalidates 375 grey-90 and 45 grey-40 rows. That fix is
a different change and needs its own predeclaration and its own locked sample.
