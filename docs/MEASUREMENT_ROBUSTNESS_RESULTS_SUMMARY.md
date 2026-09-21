# Measurement robustness — consolidated results for reporting

Every number below was produced by a committed script and is reproducible from the frozen artefacts.
Artifacts live in `_private_audit/` on the analysis host (`moniaz@100.115.180.59:/Users/moniaz/niki`),
which is **not** in the repository; the scripts that generate them are.

Frozen throughout every task, never modified:

| item | value |
|---|---|
| `src/biomarker/clinical_measurement_v1.py` sha256 | `6aea0d6856d75e205f88201cfc20311a4d13acfdac839aa5ac8bfac3a1f13542` |
| `data/features/clinical_measurement_v1.csv` sha256 | `db123ac5f663f4925ac9fff52d204bede85963966794062d1855e315a4e38ffc` |
| `data/features/final_biomarkers_v1.csv` sha256 | `f1c41e923ae29d4e228097536765f5e7399963062253ad925657a077cddc10c0` |
| canonical split | 8870 images / 414 groups / 6211-1328-1331 |
| `SEG_CURRENT_V1` | `weights/best_weight_DeepLabV3+_resize_27`, sha `c373f53813ee60b89651a04a98bc5f1d6bc60a5a50f45c4e10f42475650d5374` |
| FINAL_PRIMARY | `vessel_density_fov`, `skel_density_fov`, `fractal_d0`, `fractal_d1`, `fractal_d2` |

---

## 1. Task 5B-H — FOV under brightness extremes and black-border styles

Script: `scripts/task5b_h_fov_stress.py` (+ `scripts/task5b_h_mechanism.py`).
Sample 328 images, deterministic, stratified by source × geometry × brightness; 8 brightness + 22
border conditions; vessel mask fixed.

| endpoint | value |
|---|---|
| harness faithfulness: baseline `vessel_density_fov` vs frozen table | `9.71e-17` |
| baseline `fov_valid` mismatches | 0 |
| brightness: median / min FOV Dice | `1.000000` / `0.622341` |
| border: median / min FOV Dice | `0.984718` / `0.317889` |
| brightness density \|rel\|: median / p95 / max | `0.00000` / `0.001742` / `0.072794` |
| border density \|rel\|: median / p95 / max | `−0.001184` / `0.051578` / `2.187549` |
| border pairs with density >1 % / >5 % / >10 % | 49.4 % / 21.6 % / 9.3 % |
| FOV failures baseline / perturbed | 3 / 328 → 37 / 9840 |
| verdict | `FOV_ROBUSTNESS = FAIL`, `FINAL_PRIMARY_REQUIRES_REVIEW = YES` |

Mechanisms established:
* **brightness is exactly scale-invariant** — Otsu tracks the scaling (`thr/(thr_b·c) = 1.0000`) while
  no pixel clips, so dark scaling leaves the FOV bit-identical (min Dice `1.000000` at ×0.25 and
  ×0.50); the tail appears only with clipping (saturated fraction 0.4 % → 10.5 % from ×1.15 to ×1.80);
* **borders act through the threshold**, which drops 20–55 % (93.07 → 39.35; 51.02 → 28.72) so retinal
  background flips into the bright class and the FOV grows up to 2.5×; a grey border *above* the
  image's own threshold does the opposite (threshold ×2.07, FOV collapses to 12 % coverage);
* `fov_frac_inside_pad = 0.000` in every inspected case — the border is never itself selected.

## 2. Task 5B-H2 — impact on all five FINAL_PRIMARY biomarkers

Scripts: `scripts/task5b_h2_impact.py`, `scripts/task5b_h2_analysis.py`.
Same 328 images, same conditions, production `measure()` with `with_fractal=True`, vessel mask fixed,
9,840 rows.

| feature | med rel | p95 rel | max rel | >1 % | >5 % | >10 % |
|---|---|---|---|---|---|---|
| `vessel_density_fov` | −0.00124 | +0.05097 | +2.1876 | 47.3 % | 19.9 % | 7.5 % |
| `skel_density_fov` | −0.01676 | +0.02532 | +4.2915 | 59.0 % | 42.1 % | 24.0 % |
| `fractal_d0` | −0.01782 | +0.01775 | +0.1196 | 71.2 % | 21.2 % | 2.7 % |
| `fractal_d1` | −0.01702 | +0.03649 | +0.1411 | 74.9 % | 22.8 % | 3.6 % |
| `fractal_d2` | −0.01631 | +0.04097 | +0.1607 | 75.2 % | 23.2 % | 3.9 % |

Brightness median relative delta: `0.00000` for all five.

Three separate mechanisms, each measured rather than asserted:
* `skel_density_fov` = pure denominator. Its numerator is the **whole-frame** skeleton, which changed
  in **0 of 9,840 rows**; `Δ = S·(1/fp_p − 1/fp_b)` matches the observed signed delta to `9.8e-17`.
* `vessel_density_fov` = denominator **and** inclusion, partially cancelling (median |0.00246| and
  |0.00121| against a median total of |0.00072|).
* `fractal_d0/d1/d2` = **canvas extent, not vascular support**. Rows with an *unchanged* support count
  moved more (0.0332 / 0.0354 / 0.0359) than rows where support changed (0.0252 / 0.0298 / 0.0300);
  rank correlation only +0.14…+0.16; and zero-padding the identical binary support to the border
  canvas reproduces median `−2.29 % / −2.68 % / −2.78 %` with no FOV call at all.

Spearman with FOV error: density features `−0.8959` and `−0.9595`; fractals `−0.0037 … −0.0143`
(essentially orthogonal). No NaN transitions anywhere (0 finite→NaN, 0 NaN→finite).

## 3. Task 5B-H3 — one predeclared candidate (`CLINICAL_MEASUREMENT_V2_CANDIDATE`)

Scripts: `scripts/task5b_h3_eval.py`, `_analysis.py`, `_qc.py`;
candidate `src/biomarker/clinical_measurement_v2_candidate.py`.
Development 328 (already inspected) used only to reproduce the failure; the decision rests on a
**new locked sample of 450 images**, disjoint, stratified, with the same 8 + 22 conditions plus 6 new
novel border conditions; gate G1–G7 written to `task5b_h3_manifest.json` before the run. 16,650 rows.

| endpoint | V1 | V2 |
|---|---|---|
| border median FOV Dice | 0.991526 | **0.999894** |
| border mean FOV Dice | 0.965228 | 0.988830 |
| p01 / min Dice | 0.736531 / 0.000000 | 0.782499 / 0.459247 |
| fraction Dice < 0.99 | 49.1 % | **12.0 %** |
| fraction Dice < 0.90 | 9.9 % | **3.8 %** |
| relative area median / p95 | +0.002979 / +0.346622 | −0.000090 / +0.068011 |
| centroid displacement median | 0.002276 | **0.000017** |
| FOV valid → invalid | 30 | **444** (regression) |
| `vessel_density_fov` >1 % / >5 % | 44.4 % / 19.6 % | **12.2 % / 5.8 %** |
| `skel_density_fov` >1 % / >5 % | 53.7 % / 38.8 % | **12.5 % / 10.5 %** |
| fractal median rel | −1.7 % | **0.00000** |
| zero-padding control median \|rel\| (D0/D1/D2) | 2.60 / 2.72 / 2.88 % | **0.00000 / 0.00000 / 0.00000** |
| new finite→NaN (border) | 3 | **0** |
| native FOV Dice V2 vs V1 | — | median 1.000000, validity lost 0 of 450 |
| native fractal drift V2 vs V1 | — | −3.46 % / −3.51 % / −3.48 % median (definition change) |

Gate: **6 of 12 checks pass → `FOV_BORDER_ROBUSTNESS_V2 = FAIL`**. Files
`_private_audit/task5b_h3_summary.json`, `task5b_h3_rows.csv`, `task5b_h3_canvas_control.csv`,
`task5b_h3_invalid_reasons.csv`, 15 QC montages `task5b_h3_qc_*.png`.

Failure mechanism, identified: a border whose grey level sits above the retinal background (90, and
40 on dark images) survives `green > thr` as a bright ring, `binary_closing(disk(3))` joins it to the
disc, the **largest connected component becomes the border ring**, and V2's mask exclusion then
deletes exactly that component. 375 of the 444 rows are grey-90, 45 are grey-40; a 28-row re-measure
gives `coverage_below_15pct` 18 and `fragmented_bright_region` 10, median coverage 0.1430.

## 4. Task 5B-H4 — V3 candidate, padding-aware component selection

Script: `scripts/task5b_h4_eval.py`; candidate `src/biomarker/clinical_measurement_v3_candidate.py`.
**RUNNING.** Locked sample 450 images, disjoint from the 328 development and the 450 H3 images,
stratified over source × geometry × split; V3 evaluated on all 42 conditions, V1 and V2 on a
predeclared 8-condition mechanism subset; gate G1–G11 frozen in `task5b_h4_manifest.json`.

Single change, predeclared: the detected external constant band is removed from the binarisation
**before** morphology and **before** labelling, so a threshold-positive padding ring can never be a
candidate component and can never inflate `frag`. Everything else is V1/V2 unchanged, including
`CONST_TOL 2.0`, `MIN_CONTENT_FRAC 0.50`, `DOMAIN_MARGIN 0.10`, `DOMAIN_ALIGN 8`, `min_size`,
`disk(3)` and the failure criteria `0.15 / 0.985 / 0.90`.

Single-image benchmark proving the mechanism is addressed (one `155…` plus image, 384×512):

| condition | V1 | V2 | V3 |
|---|---|---|---|
| grey-90 border, validity | valid | **invalid** (`fragmented_bright_region`) | **valid** |
| grey-90 border, `vessel_density_fov` | 0.05974 | 0.07643 | 0.07648 |
| grey-90 border, Dice vs V1 | 1.0000 | 0.8774 | 0.8771 |

V3 returns the same value V2 computes, without destroying the selected component.

Outputs when finished: `_private_audit/task5b_h4_summary.json`, `task5b_h4_rows.csv`,
`task5b_h4_baseline.csv`, `task5b_h4_canvas_control.csv`, `task5b_h4_manifest.json`.

---

## Standing conclusions

* No number in this document is a clinical claim. No expert artery/vein annotation and no expert FOV
  annotation exists; nothing here shows any version is anatomically more correct than another.
* Historical Branch A/C and LOSO outputs remain `HISTORICAL_CONTAMINATED_NONREPRODUCIBLE` and
  `NONCANONICAL_AND_REQUIRES_RERUN`; they are provenance evidence, not headline results.
* `A/V = EXPLORATORY_ONLY`; the earlier "tie-dominated" hypothesis stays withdrawn (it was a
  `np.clip(rgb, 0, 1)` scaling bug in a replica).
* Disease-model training has not started and stays blocked until the FOV question closes.
* The 8,870-row feature table has **not** been regenerated by any of these tasks.
