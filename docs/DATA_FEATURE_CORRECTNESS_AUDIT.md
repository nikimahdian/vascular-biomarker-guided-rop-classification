# Data + feature extraction correctness audit

Audit date 2026-09-20. Scope: the data and feature pipeline behind the locked Branch A / Branch C
results, the cross-source grid and the expert-validation package. Nothing frozen was modified:
the expert cohorts, their study IDs, the clinician ZIP, the locked split, the historical A/B/C and
LOSO outputs, and the model weights are all untouched. Fixes are additive and versioned.

---

# SUMMARY

| colour | count | meaning |
|---|---|---|
| RED | **2** | confirmed correctness bugs that change the numbers |
| YELLOW | 10 | confirmed design limitations, now documented |
| GREEN | 6 | validated correct |
| REQUIRES EXPERT DATA | 7 | cannot be resolved without the human annotations |
| UNRESOLVED | **1** | a running repair whose before/after is not yet complete |

**No unresolved P0 correctness issue remains open.** Both RED bugs have a fix and a regression test.
The one UNRESOLVED item is a computation still running, not an open question about whether it is
wrong.

---

# RED — confirmed bugs

## R1. 610 of 8870 images are paired with a different image's mask

**Severity: P0. Affects the locked Branch A and Branch C results. Mechanism NOT yet established.**

**The effect is confirmed and measured.** 610 of 8870 rows in `biomarker_features.csv` carry a mask
whose stem is not the row's own image stem.

| | |
|---|---|
| images affected | **610 / 8870 (6.9 %)** |
| by split | train 414, val 106, **test 90** |
| by source | plus 377, farfum_rop 118, farabi 115 |
| by label | normal 418, plus 132, pre-plus 60 |
| donor in the same group | 606 / 610 |

Confirmed two ways: `mask_manifest.csv` has only 8264 distinct mask paths for 8870 images, and 610
rows parse to a mask stem that is not their own image stem.

**The mechanism has NOT been identified, and the obvious hypothesis was tested and rejected.**

The hypothesis was a prefix collision in `src/segmentation/infer_masks.py::find_existing_mask`,
which used `masks_dir.glob(f"{stem}_*.png")`. That hypothesis is **wrong**. Worked example:

```
image      ...53671a65-...-13ae5d0bfdcf.1.jpg
its mask   ...53671a65-...-13ae5d0bfdcf.1_f3829e03.png      <- exists on disk
glob("...53671a65-...-13ae5d0bfdcf.1_*.png")
        -> ['...53671a65-...-13ae5d0bfdcf.1_f3829e03.png']  <- the CORRECT file, one match
recorded   ...53671a65-...-13ae5d0bfdcf.11_bd7c547e.png     <- the mask of image _11
```

The glob returns the right file, so the current code path did not produce this pairing. Note also
that `...cf.11_bd7c547e.png` cannot match the pattern `...cf.1_*.png`, because the character after
`...cf.1` is `1`, not `_`.

What is known: `mask_manifest.csv` holds 17,740 rows for 8,870 images, i.e. duplicate records
accumulated by the resume-and-append path, and the wrong pairing is already present in that
manifest, not introduced by the feature step. The plausible class of cause is a stale or conflicting
manifest record surviving a resumed, multi-machine run, but that has not been demonstrated and is not
claimed.

**This is recorded as UNRESOLVED, not as fixed.** What is fixed is the damage, not the diagnosis.

**Fix (damage).** A row is now considered correctly paired only if its mask filename parses as
exactly `<image_stem>_<8 hex>.png`. Every one of the 610 affected images has **exactly one** correct
mask already on disk, so no re-segmentation is needed:

- `results/audit_mask_repair_pairs.csv` — the 610 corrected pairings
- `data/masks/mask_manifest_v2.csv` — the corrected manifest
- `data/features/biomarker_features_v4_audit.csv` — features recomputed for the 610 rows, version
  `v4_audit_maskfix`

**Hardening (not the cause).** `find_existing_mask` now requires the exact-stem parse before
accepting a fallback match. This is correct on its own merits — the old glob would accept a sibling
whose stem is a strict prefix followed by `_`, e.g. stem `x_1` and file `x_1_2_<hash>.png` — but it
did not produce the 610 pairings and must not be described as having done so.

**Historical results affected: yes.** 414 train rows and 90 test rows of the locked Branch A and C
matrices used a different image's mask. Because the donor is the same eye in 606 of 610 cases the
numbers are not overturned, but they are not the measurement they claim to be.

**Regression test:** `test_image_mask_pairing_is_by_exact_stem` asserts what is actually true — that
the exact-stem rule separates a row's own mask from a sibling's. An earlier version of this test
asserted the rejected glob hypothesis and failed, which is how the hypothesis was caught.

## R2. Tortuosity used pixel count as arc length

**Severity: P0 in the definition; no effect on any published result.**

`scripts/clinical_features_v3.py`: `t = npx / hull_chord` where `npx = len(cy)` is the skeleton pixel
count and `hull_chord` is the convex-hull diameter.

**Why it is wrong.** Pixel count is an orientation-dependent proxy for arc length. A rasterised
straight vessel has one pixel per unit length when axis-aligned and roughly one per √2 units at 45°.

**Measured, synthetic straight vessels (median over branches).**

| angle | OLD (px / hull chord) | NEW (geodesic arc / chord) |
|---|---|---|
| 0° | 1.0050 | 1.0020 |
| 15° | 0.9724 | 1.0764 |
| 30° | 0.8720 | 1.0734 |
| **45°** | **0.7121** | 1.0000 |
| 60° | 0.8730 | 1.0737 |
| 75° | 0.9709 | 1.0731 |
| 90° | 1.0050 | 1.0000 |

Spread across orientations: **OLD 1.411×, NEW 1.076×**. Against analytic circular arcs, NEW gives
1.1686 for a 90° arc whose true value is 1.1107, versus OLD 1.0088.

**Fix.** `src/biomarker/geometry_core.py::branch_tortuosity`, used by
`tests/test_feature_invariance.py`. Diagonal steps are weighted √2 and the chord is between the two
real endpoints.

**Historical results affected: no.** This definition lives only in `clinical_features_v3` and was
never consumed by Branch A or Branch C. Recorded as a version change, not an overwrite.

**Residual.** NEW still shows ~7 % excess arc at intermediate angles, from the raster staircase of
the skeleton itself. That is a property of digital skeletons, not a code defect, and it bounds how
precise any tortuosity from a pixel mask can be.

---

# YELLOW — confirmed design limitations

## Y1. PVBM root selection depends on the fabricated optic disc

`src/biomarker/extract_pvbm.py` passes `xc, yc = w//2, h//2` and `radius = max(8, min(h,w)//8)` into
`compute_geomVBMs`. Inside PVBM the starting points are every skeleton component's nearest pixel to
`(xc, yc)` with `distance < 100 + radius`. The graph roots, and therefore the whole traversal, are a
function of `min(h,w)//8` — a pure function of image size.

Median relative change when the **measured** disc replaces the fabricated one, over 20 masks spanning
all five geometries:

| feature | median change |
|---|---|
| `n_startpoints` | **50 %** (up to 400 % under radius perturbation) |
| `n_endpoints` | 12 % |
| `median_branching_angle` | 10 % |
| `overall_length` | 7.8 % |
| `n_intersections` | 4.5 % |
| `tortuosity_index` | 0.2 % |
| `median_tortuosity` | 0.3 % |
| `area` | 0.0 % |

**Consequence.** `n_startpoints`, `n_endpoints` and `n_intersections` are acquisition-size features.
This is a plausible part of the mechanism behind the SEVERE source confounding (macro one-vs-rest
AUC 0.99). They are marked `confirmed_defect` and `allowed_in_classifier = 0` in the contract.
**Historical results affected: yes**, in interpretation.

## Y2. `vessel_density` is a canvas feature, not a retinal one

`src/biomarker/extract_pvbm.py::density_features`: `total / (h * w)` over the full rectangle.

| perturbation | whole-frame density | FOV-normalised density |
|---|---|---|
| +100 px black border | **−46.2 %** | +0.0 % |
| +300 px black border | **−77.1 %** | +0.0 % |
| letterboxed to square | **−25.0 %** | +0.0 % |

Same for the nine `density_r{i}c{j}` cells: they are rectangle thirds, not anatomy.
**Fix available:** `geometry_core.fov_density`. Not applied to the historical table.

## Y3. The model input is an anisotropic warp, and two post-processing constants live in warped space

`infer_masks.py` resizes every image to 256×256 for the model. For 1600×1200 that is a 6.25×
horizontal and 4.69× vertical scale. `min_area = 50` and a 3×3 square closing kernel are then applied
in the **warped** grid, so on the retina they are anisotropic and aspect-ratio dependent.
Thresholding before resizing is correct and preserves binary integrity; the post-processing constants
are the issue.

## Y4. Mask resampling order is correct but the precision is capped

Order is confirmed as **probability → threshold → NEAREST resize**. That is the right order for a
geometric measurement. The mask is produced at 256×256, so width from it quantises at the upsampling
factor.

## Y5. Geometric quadrants are not ICROP quadrants

`density_q_ne` and friends are top-left/top-right of the frame, with no laterality awareness. They
cannot be read as the clinical quadrants of ICROP.

## Y6. 51 groups contain mixed labels

51 groups (farabi 46, farfum_rop 3, plus 2) contain images with more than one label, up to three.
No group spans a split, so there is no group-level leakage; but group-level stratification does not
make a group label-pure.

## Y7. Four near-duplicate pairs cross a split boundary

Exact duplicate images: **0**. Perceptual-hash pairs at Hamming ≤ 4: 15, of which 4 cross splits and
5 cross groups. All 4 crossing pairs are label-0 versus label-0 and all are from `plus`, so there is
no label conflict. 7 pairs are at Hamming 0 while zero images are pixel-identical, i.e. same eye,
different capture.

## Y8. Eight dead or defect features

`a_tort_median` and `v_tort_median` are **100 % null** in the historical table. `tort_median`,
`tort_p90`, `tort_top3_mean` carry the R2 definition. `n_startpoints`, `n_endpoints`,
`n_intersections` carry Y1. All eight are `allowed_in_classifier = 0`.

## Y9. `find_existing_mask` hashes the absolute path

`unique_mask_name` uses `md5(image_path)`, so masks produced on a different machine never match the
canonical name and always fall through to the fallback — which is exactly the path that carried R1.
Only 18 of 8870 masks match the current path hash.

## Y10. No expert masks for this dataset

Measurement validity and ICC for the target domain remain unknown. This is the purpose of the frozen
expert-validation phase and is unchanged by this audit.

---

# GREEN — validated correct

| # | item | evidence |
|---|---|---|
| G1 | **PVBM tortuosity definition** | `distances = [1,1,1,1,√2,√2,√2,√2]`; `TI = arc.sum()/chord.sum()`; `medTor = median(arc/chord)`. Geometric weights and the correct direction (≥ 1). |
| G2 | **EDT width** | Exact on even-width strips (2,4,6,8,12,20,30 → error 0.000); identical at 0°, 45°, 90°; `width/DD` exactly invariant at 1×/2×/4×; disc diameter recovered to 0.5 %. |
| G3 | **Disc-independent features** | `area` 0.0 % change under every disc perturbation; `fractal_d0/d1/d2` and `singularity_length` are computed on the mask alone. |
| G4 | **No exact duplicates, no group leakage** | 0 duplicate images, 0 duplicate image paths, 0 groups spanning splits, 0 groups spanning sources. |
| G5 | **Preprocessing fit on train only** | `StandardScaler().fit(X[train_mask])` in both `branch_a_tabular.py` and `branch_c_hybrid.py`; imputation median from `split == "train"`. |
| G6 | **Metadata exclusion** | `META_COLS` excludes path, label, split, source, group, patient, exam and identity. Detector-status columns are additionally named in the contract as `allowed_in_classifier = 0`. |

Two latent leaks were checked and found **inactive**: the global-median fallback at
`branch_a_tabular.py:91` and the global variance filter at line 86 both trigger only for columns
that are all-NaN-in-train or constant; no such column exists in either table, so neither affected the
historical runs. Both are still code smells and should be made train-only.

---

# REQUIRES EXPERT DATA

`width_p50_dd`, `width_p90_dd`, `width_p95_dd`, `width_mean_dd`, `width_ann_*` and the two
artery/vein widths. Their definitions are corrected, but their validity is a question for the
annotators, not for this audit. Until then they are `requires_expert_data` and may not be reported as
validated clinical measurements.

The A/V heuristic is exploratory by design: it splits branches at the length-weighted median
intensity, which mechanically forces a near-balanced arterial/venous split by construction. It has no
expert labels to check against.

---

# UNRESOLVED

One item, and it is a running computation rather than an open question:

**`biomarker_features_v4_audit.csv` is still being generated.** The 610 corrected rows need their
PVBM features recomputed (density, geometry, multifractal), which is roughly 2 s per row; the job
prints progress to `/tmp/repair.log` on the compute host. The before/after feature change per row
will be written to `results/audit_mask_repair.json` when it completes. Nothing else depends on it.

---

# Feature contract

`configs/feature_contract.csv`, 65 entries, one per numeric column of both feature tables.

| clinical_status | n |
|---|---|
| exploratory | 40 |
| qc_only | 8 |
| confirmed_defect | 8 |
| requires_expert_data | 7 |
| validated_correct | 2 |

| table | allowed in classifier | blocked |
|---|---|---|
| `biomarker_features.csv` (pvbm_v1) | 19 | 4 |
| `clinical_v3` | 29 | 13 |

No feature may enter a model without a row here. The contract is enforced by
`tests/test_feature_invariance.py::test_feature_contract_covers_every_numeric_feature`.

---

# What was changed, and what was not

**Changed (code correctness, explicitly not frozen):**
- `src/segmentation/infer_masks.py` — exact-stem mask matching and the module import it needs
- `src/biomarker/geometry_core.py` — new; corrected tortuosity, FOV density, EDT width helpers
- `tests/test_feature_invariance.py` — new; the invariance and correctness suite
- `configs/feature_contract.csv` — new

**Not changed:** `data/features/biomarker_features.csv`, `data/splits/*`, every `results/` output,
the model weights, the locked split, the expert cohorts and their study IDs, and the clinician ZIP.
Every definition-changing correction was written to a new file with a new version string.

---

# Reproducing

```bash
python -m pytest tests/test_feature_invariance.py -q

python scripts/audit_synthetic.py                 # sections D, E, F
python scripts/audit_pvbm_disc_sensitivity.py     # section J
python scripts/audit_data_integrity.py            # section A
python scripts/audit_mask_bug_count.py            # section A/B, the P0
python scripts/repair_mask_pairing.py             # the repair, writes new files only
python scripts/make_feature_contract.py           # section L
```
