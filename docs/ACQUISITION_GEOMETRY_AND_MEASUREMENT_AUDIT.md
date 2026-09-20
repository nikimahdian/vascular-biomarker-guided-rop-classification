# Acquisition geometry, the measurement layer, and the biomarker branch

This note records a measurement audit of the tabular biomarker branch. Everything below is
reproducible from the scripts listed at the end; each number is computed on the frozen split
`0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8` (8870 images, 414 groups,
6211 / 1328 / 1331) unless stated otherwise.

**Summary.** The tabular branch does not generalise to an unseen acquisition source (Plus-vs-rest
AUC 0.514 on `farabi`, i.e. chance), and adding it to the image branch helps in only one of three
held-out sources. The causes are measurable: the clinical feature table that carried the calibre
features was never consumed by the classifier and contained five concrete defects; the width
signal it did carry was largely a proxy for image size rather than for vessel calibre; and the
dataset is a mosaic of five acquisition geometries whose Plus prevalence ranges from 0 % to 42 %,
which the grouped split did not balance.

---

## 1. The clinical feature table existed but the classifier never read it

`src/biomarker/extract_clinical_v2.py` builds `biomarker_features_clinical_v2.csv` (8870 x 43),
which carries the features the clinical definition of Plus actually depends on: vessel width in
disc diameters (`width_p50_dd`, `width_p90_dd`, `width_p95_dd`, `width_mean_dd`), artery/vein
specific width (`a_width_p90_dd`, `v_width_p90_dd`, `av_width_ratio_p90`), posterior-pole ring
densities, quadrant densities and tortuosity.

`src/classify/branch_a_tabular.py` reads `biomarker_features.csv` (32 columns), which contains
none of those. The clinical table was only ever used by one diagnostic configuration
(`configs/biomarker_diagnostics_clinical_v2.yaml`).

### Five defects in that table

| # | Defect | Evidence |
|---|---|---|
| 1 | The disc diameter is a constant, not a disc | `dd_over_min_side == 0.1000` for all 8870 rows (std 0). `PVBM.DiscSegmenter` raises, the bare `except Exception: pass` swallows it, and the fallback `max(8, min(h,w)//10)` is used for every image. |
| 2 | Width is sampled on every vessel pixel | `vals = dist[mask > 0]`. That distribution is edge-weighted (triangular, zero at the vessel border) and biased low. It should be sampled on the skeleton. |
| 3 | No anatomical region | Medians are taken over the whole frame, so they track how much periphery is in view, not calibre. `corr(skeleton_px, median_width) = -0.422` on the Neo subset. |
| 4 | Artery/vein split per pixel | `thr = median(g[vessel]); artery = vessel & (g >= thr)`. This forces a 50/50 pixel split and shreds single vessels across both classes. Result: `a_width_p90_dd` 0.1497 > `v_width_p90_dd` 0.1356, i.e. arteries reported wider than veins, which is the wrong direction. |
| 5 | Tortuosity chord is not a path | `chord = hypot(xs[-1]-xs[0], ys[-1]-ys[0])` with `xs, ys` from `np.where`, which is not path order. Additionally `a_tort_median` and `v_tort_median` are 100 % null. |

`scripts/clinical_features_v3.py` rebuilds the layer: real disc geometry, width on the skeleton,
a 0.5-2.0 DD peripapillary annulus, per-branch artery/vein assignment using background-corrected
green intensity, and a convex-hull chord for tortuosity. The artery/vein direction is corrected
by that change (`av_width_ratio_p90` median 1.110 -> 0.934, veins wider, as expected).

---

## 2. The width signal was largely an image-size proxy

Normalising width by the *measured* disc instead of by `min(h,w)/10` reduces the cross-source
median spread of `width_p90_dd` from 0.064 to 0.015. It also removes much of the apparent
disease signal:

| column | subset | n | AUC (Plus vs rest) | Pearson r vs image min-side |
|---|---|---|---|---|
| `width_p90_dd` (constant DD) | all rows | 8870 | **0.6501** | -0.692 |
| `width_p90_dd` (constant DD) | disc found only | 3772 | **0.6545** | -0.680 |
| `width_p90_dd_true` (measured DD) | disc found only | 3772 | **0.5874** | -0.533 |

The comparison in the last two rows is on identical rows, so the drop of 0.067 is caused by the
re-normalisation and not by reduced coverage. A column that is anti-correlated with image size at
r = -0.69, on a dataset where Plus prevalence depends strongly on image size, will look like a
disease predictor without measuring calibre.

Removing the width columns entirely barely changes source separability (macro one-vs-rest AUC
0.9986 -> 0.9980 on the disc-found subset), so the source confound is not carried by width: it is
multivariate, with no single feature above univariate AUC 0.78.

---

## 3. The label is confounded with acquisition geometry

The dataset is not one acquisition. It is five:

| min-side | resolution | source | images | Plus |
|---|---|---|---|---|
| 480 | 640x480 | plus | 2516 | 18.4 % |
| 960 | 1280x960 | farabi | 1382 (+28) | 42.4 % |
| 1080 | 1440x1080 | plus | 969 | **0.0 %** |
| 1200 | 1600x1200 | farfum_rop (+28 farabi) | 1557 | 17.5 % |
| 1240 | 1240x1240 | plus | 2446 | 6.4 % |

And the grouped split did not balance that:

| min-side | split | images | Plus | groups | groups containing Plus |
|---|---|---|---|---|---|
| 480 | train | 1998 | **464** | 58 | **1** |
| 480 | val | 308 | 0 | 9 | 0 |
| 480 | test | 210 | 0 | 12 | 0 |
| 960 | train / val / test | 959 / 211 / 212 | 409 / 88 / 89 | 104 / 24 / 24 | 44 / 15 / 10 |
| 1080 | train / val / test | 563 / 122 / 284 | 0 / 0 / 0 | 26 / 10 / 10 | 0 / 0 / 0 |
| 1200 | train / val / test | 1098 / 229 / 230 | 191 / 41 / 41 | 56 / 10 / 10 | 12 / 2 / 2 |
| 1240 | train | 1593 | **0** | 46 | **0** |
| 1240 | val | 458 | 78 | 9 | 1 |
| 1240 | test | 395 | 79 | 6 | 1 |

Two consequences. All 464 Plus images at 640x480 come from a single group and lie entirely in
train, so val and test cannot evaluate that geometry at all. Conversely train contains no Plus
image at 1240x1240, while 157 Plus images at that geometry sit in val and test, so 79 of the 209
test Plus images come from a geometry the model never saw labelled Plus. Excluding the score, the
row and column totals reconcile exactly with the locked split (6211 / 1328 / 1331).

This is not an artefact of the post-hoc disc work: `scripts/acquisition_geometry_audit.py`
recomputes it from the original feature table and mask manifest.

### What it does *not* mean

It does not mean the headline test AUC is inflated. Within geometry, branch B scores 0.875
(960), 0.964 (1200) and 0.951 (1240) against a pooled 0.928, so the pooled figure is not higher
than the within-group figures. The image branch does generalise to a geometry it never saw
labelled Plus. What the imbalance does mean is that the headline number mixes "detect Plus" with
"transfer the Plus concept across acquisition geometries", and the two cannot be separated on
this split.

---

## 4. Complete leave-one-source-out grid

`artifacts/loo_summary_full.csv` — 9 of 9 rows. Six rows are from the earlier run; the three
`farabi` rows were produced by `src.compare.leave_one_source_out --holdouts farabi --reuse-b`,
which evaluates the existing farabi Branch B checkpoint rather than retraining it.

| hold-out source | n | A (biomarkers) | B (image) | C (fusion) | C − B |
|---|---|---|---|---|---|
| farabi | 1410 | **0.5140** | 0.7371 | 0.7351 | -0.0019 |
| farfum_rop | 1533 | 0.6812 | 0.8634 | 0.8721 | +0.0088 |
| plus | 6004 | 0.7747 | 0.8865 | 0.8429 | -0.0436 |

| branch | locked test | LOSO mean | drop |
|---|---|---|---|
| A | 0.7999 | 0.6566 | -0.1433 |
| B | 0.9280 | 0.8290 | -0.0990 |
| C | 0.9123 | 0.8167 | -0.0956 |

Fusion beats the image branch in one of three held-out sources, and loses on the locked test
(0.9123 vs 0.9280, DeLong p = 0.00806).

---

## 5. The measurement layer, validated externally

`scripts/hvdro_seg_validation.py` and `scripts/disc_detector_train.py` evaluate against the
HVDROPDB expert masks (100 vessel masks, 100 optic-disc masks) at the deployed inference setting
(256x256, threshold 0.20, min_area 50, closing 3).

| task | camera | ours | published reference |
|---|---|---|---|
| vessel Dice | RetCam | **0.754** (IoU 0.610, recall 0.814, precision 0.709) | 0.52 |
| vessel Dice | Neo | 0.473 (IoU 0.312, recall 0.338, precision 0.816) | 0.66 |
| optic disc Dice | RetCam | **0.9066** | 0.93 |
| optic disc Dice | Neo | **0.9203** | 0.92 |

The disc detector replaces the pseudo-disc. On HVDROPDB its radius ratio is 0.991 (RetCam) and
1.024 (Neo) with centre error 0.060 and 0.029 disc diameters, against 2.35 / 2.84 and 7.5 / 7.1
disc radii for the image-centre heuristic.

Applied zero-shot to this project's own images the disc detector fires confidently on only 42.5 %
of them, and the failure is per-image rather than per-patient (of 228 exam identifiers, 6 fail
everywhere, 12 succeed everywhere, 185 are mixed). Inspecting overlays confirms the disc is
present in some failures, so this is a domain-transfer limit and not an absence of anatomy. No
optic-disc ground truth exists for this project's data, so the detector cannot be fine-tuned
here; the measured-disc features are therefore reported as missing rather than imputed for the
images without a confident detection.

### The disc-validity rule, and the fallback that was removed

`scripts/clinical_features_v3.py` defines a single locked rule:

```
disc_valid = peak_prob > 0.9  AND  0.03 <= disc_dd_px / min(h, w) <= 0.25
```

When `disc_valid` is false, **every** disc-relative feature (25 columns: the diameter and centre
fractions, the ring and quadrant densities and coverages, the annulus-restricted widths, and all
widths expressed in disc diameters) is returned as `NaN`. There is no image-centre fallback.
An earlier revision of that script fell back to `dd = min(h,w)/10` and the image centre and then
computed the disc-relative features anyway, which re-introduced precisely the defect the script
exists to remove; `docs` and code disagreed about it. That path is gone, and
`scripts/verify_disc_rule.py` asserts the invariant on any generated table.

Features that do not need a disc are still computed for those images, so the 57.5 % without a
valid disc are not dropped from everything: vessel density, skeleton density, skeleton pixel
count, branch count, width in **pixels** (`width_p50_px`, `width_p90_px`, `width_mean_px`),
`width_shape_p90_over_p50`, the tortuosity statistics, `a_frac`, `a_width_p90_px`,
`v_width_p90_px` and `av_width_ratio_p90`. The artery/vein width ratio is a ratio of two
identically normalised widths, so the disc cancels and the feature is defined even when the disc
is not.

The 0.9 confidence floor is a convention, not a validated cut. Recalibrating it is one of the
explicit goals of the expert disc annotation in `expert_validation/PROTOCOL.md`.

The vessel mask is produced at 256x256 and upsampled, so any width derived from it quantises at
roughly (1/256) / (DD fraction ~0.095) ≈ 0.041 disc diameters, independently of the working
resolution. Against a label effect of 0.19 -> 0.21 disc diameters that is about 20 % of the
effect size.

---

## Reproducing

```bash
# 1. disc detector (5-fold on HVDROPDB-OD) and inference over all project images
python scripts/disc_detector_train.py
python scripts/disc_inference_all_images.py

# 2. width estimators against the expert vessel masks (pass 1 and the corrected pass)
python scripts/width_estimators_benchmark.py
python scripts/width_estimators_benchmark_v2.py

# 3. rebuild the clinical feature table with a measured disc
python scripts/clinical_features_v3.py

# 4. the audit itself
python scripts/acquisition_geometry_audit.py      # geometry x label x split table
python scripts/within_geometry_auc.py             # AUC within each acquisition geometry
python scripts/isolate_renormalisation_effect.py  # renormalisation vs coverage
python scripts/source_probe_ab.py                 # source separability, constant vs measured DD

# 5. complete the LOSO grid
python -m src.compare.leave_one_source_out --holdouts farabi --reuse-b
python scripts/merge_loso_grid.py
```

Diagnostics for the corrected table:

```bash
python -m src.biomarker_diagnostics --config configs/biomarker_diagnostics_clinical_v2c.yaml \
    --run-name diag_v5_clinical_v2c
```

## Limitations

- No expert vessel or disc masks exist for this project's own data, so mask validity and
  measurement ICC for it remain unknown. The external numbers above are from HVDROPDB.
- The disc detector transfers to only 42.5 % of this project's images, and the missingness
  pattern is itself source-dependent (48.9 % farabi, 53.3 % farfum_rop, 38.2 % plus).
- Five acquisition geometries with strongly unequal Plus prevalence, unbalanced across the
  locked split, mean the headline test number mixes detection with cross-geometry transfer.
- The locked test has already been observed by earlier exploratory runs and is not a fresh
  confirmation set; nothing here selects a claim on it.
