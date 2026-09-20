# Historical-equivalent PVBM reconstruction — Task 4

```
CANONICAL_ROW_N:                        8870
CANONICAL_UNIQUE_IMAGE_N:               8870

HISTORICAL_FEATURE_COUNT:                 23
CORRECTED_FEATURE_COUNT:                  23

HISTORICAL_FEATURE_REPRODUCIBILITY:     FAIL

UNAFFECTED_ROWS_COMPARED:               8260
UNAFFECTED_ROWS_EXACT_OR_EQUIVALENT:       0
UNAFFECTED_ROWS_NONREPRODUCIBLE:        8260

AFFECTED_ROWS:                           610
EXPECTED_610_CONFIRMED:                  YES

HISTORICAL_A_C_TORTUOSITY_IMPLEMENTATION:       PVBM
LOCAL_TORTUOSITY_BUG_AFFECTED_HISTORICAL_A_C:   NO

CORRECTED_TABLE_SHA256:  90ce160727347f1d73f571ba4969638550ae534cb13cf891bf814fce92f7a463

TASK4_STATUS:                        INCOMPLETE
```

```
BRANCH_A_HISTORICAL_FEATURE_TABLE:        data/features/biomarker_features.csv
BRANCH_C_HISTORICAL_FEATURE_TABLE:        data/features/biomarker_features.csv

BRANCH_A_FEATURE_SCHEMA_IDENTICAL_TO_C:   YES

BRANCH_A_HISTORICAL_ROW_N:                8870
BRANCH_C_HISTORICAL_ROW_N:                8870
```

> **Read section F before using any number in section H or I.** The reconstruction is valid as a
> table, but the equivalence gate that would license attributing the 610-row delta to the mask
> bug alone **failed**, and the reason is a destroyed input rather than a code defect. The delta
> figures below are therefore *confounded estimates*, not a clean attribution.

---

## A. Repository evidence

| field | value |
|---|---|
| HEAD at time of task | `eabc341` — *Task 3: resolve the 23 LOSO-only images and quantify exact-duplicate leakage* |
| lineage | `docs/PVBM_HISTORICAL_LINEAGE.md` |
| historical table | `data/features/biomarker_features.csv`, 8870 × 32 |
| historical table byte sha256 | `d3fb94dbe2fc6b9b13fc02163e9a79df70df3b5c380701878bd3f148e2d92aa7` |
| corrected table | `data/features/biomarker_features_historical_equivalent_corrected_v1.csv` |
| corrected table byte sha256 | `90ce160727347f1d73f571ba4969638550ae534cb13cf891bf814fce92f7a463` |
| corrected table size | 5,778,666 bytes, 8870 × 32 |

Scripts added by this task:

* `scripts/pvbm_historical_reconstruct.py` — builds the corrected table (E, O);
* `scripts/pvbm_historical_reproducibility.py` — the 8,260-row equivalence gate (F, G);
* `scripts/pvbm_historical_delta.py` — per-feature and per-subgroup deltas (H, I, J, P, Q);
* `scripts/mask_artifact_provenance.py` — the diagnostic that explains the gate failure (G).

---

## B. What was built, and what was deliberately not changed (A, E)

The corrected table was produced by importing the historical functions **verbatim**:

```python
from src.biomarker.extract_pvbm import density_features, geom_features, load_binary_mask
```

and driving them with `roi = "whole"`, over the canonical cohort
(`data/splits/all.csv`, fingerprint `0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8`)
paired with the verified canonical mapping
(`data/masks/mask_manifest_canonical_v1.csv`, sha256 `b3bc6538026d6ad0fc1342043e5ca2d6de56662233bf6d0709d9d875f24b21cd`).

Nothing about the feature mathematics is re-implemented. `scripts/pvbm_historical_reconstruct.py`
contains no PVBM formula.

Exactly two things changed relative to the historical run, as required:

1. **cohort membership** — the canonical 8,870 rather than the historical 8,947;
2. **image→mask pairing** — `mask_manifest_canonical_v1.csv` rather than the historical
   `mask_path` values.

Everything else is unchanged, including all of section L of the task brief:
`n_startpoints`, `n_endpoints`, `overall_length`, `area`, whole-frame density, the 3×3 regional
density grid, and the fabricated optic-disc proxy `xc = w // 2, yc = h // 2,
radius = max(8, min(h, w) // 8)`. No FOV normalisation was introduced. `geometry_core.py` was not
substituted for PVBM tortuosity. Fractal parameters are unchanged. Feature names, order and the
metadata column set are identical.

Environment: `numpy 2.0.2`, `scipy 1.13.1`, `scikit-image 0.24.0`, `pandas 2.3.3`,
`pvbm 3.0.1.0`, `Pillow 11.3.0` — matching `requirements-lock.txt` exactly. Extraction used 20
worker processes over 8,870 images in 1,714 s with **zero extraction errors**.

`data/features/biomarker_features.csv` was not overwritten, and no historical file was modified.

---

## C. Table integrity (O)

| check | result |
|---|---|
| rows == 8870 | pass |
| unique canonical image IDs == 8870 | pass |
| duplicate rows by image identity | 0 |
| image ID set equals the locked split | pass |
| feature count == 23 | pass |
| column order identical to the historical table | pass |
| `inf` values | 0 |
| predictor/metadata column overlap | none |
| one feature row per canonical mask mapping | pass |

Population is identical to the historical table on every axis:

| axis | corrected | historical |
|---|---|---|
| source | plus 5931, farfum_rop 1529, farabi 1410 | identical |
| label | normal 6459, pre_plus 931, plus 1480 | identical |
| split | train 6211, val 1328, test 1331 | identical |

NaN pattern, per feature (only three features have any NaN):

| feature | corrected | historical |
|---|---|---|
| `tortuosity_index` | 6 | 6 |
| `median_tortuosity` | 6 | 6 |
| `median_branching_angle` | 60 | 61 |

The corrected table is internally sound. **Its soundness is not what failed.**

---

## D. The affected set, independently recomputed (J)

Comparing the historical `mask_path` against the canonical mapping, per row:

| quantity | value |
|---|---|
| rows joined | 8870 |
| **affected** (mask path differs) | **610** |
| unaffected (mask path identical) | 8260 |

```
VERIFY 414/106/90 : {'train': 414, 'val': 106, 'test': 90}
```

| split | n_total | n_wrong_mask_historical | percent_wrong |
|---|---|---|---|
| train | 6211 | 414 | 6.67 % |
| val | 1328 | 106 | 7.98 % |
| test | 1331 | 90 | 6.76 % |

Affected by source: plus 377, farfum_rop 118, farabi 115.
Affected by label: normal 418, plus 132, pre_plus 60.
Affected by geometry: min1240 169, min480 134, min1200 120, min960 113, min1080 74.

All four Task 1 figures reproduce exactly. `EXPECTED_610_CONFIRMED = YES`.

A mechanism check: **606 of the 610** affected historical `mask_path` values are themselves the
canonical mask of a *different* canonical image, and in **606 of 610** cases that other image
belongs to the **same group**. The historical table therefore paired each affected image with a
sibling frame from the same patient/exam.

---

## E. Equivalence gate on the 8,260 unaffected rows (F) — FAILED

For an unaffected row the historical `mask_path` and the canonical `mask_path` are the *same
string*, so the same file was read. If the pipeline were reproducible from disk, every feature
should have come back bit-for-bit. It did not.

Tolerance: `rel <= 1e-9` with an absolute floor of `1e-12` — a strict test, not a loose one.

| feature | n | exact | max abs diff | median abs diff | max rel diff | nan mismatch | verdict |
|---|---|---|---|---|---|---|---|
| vessel_density | 8260 | 2459 | 1.38e-03 | 1.63e-05 | 5.97e-02 | 0 | NONREPRODUCIBLE |
| vessel_pixels | 8260 | 2459 | 2117 | 13 | 5.97e-02 | 0 | NONREPRODUCIBLE |
| density_r0c0 | 8260 | 7032 | 8.38e-03 | 0 | 8.04e+08 | 0 | NONREPRODUCIBLE |
| density_r0c1 | 8260 | 5916 | 8.92e-03 | 0 | 3.23e-01 | 0 | NONREPRODUCIBLE |
| density_r0c2 | 8260 | 7075 | 7.22e-03 | 0 | 6.71e+08 | 0 | NONREPRODUCIBLE |
| density_r1c0 | 8260 | 6540 | 1.17e-02 | 0 | 1.67e+00 | 0 | NONREPRODUCIBLE |
| density_r1c1 | 8260 | 5811 | 1.07e-02 | 0 | 1.82e-01 | 0 | NONREPRODUCIBLE |
| density_r1c2 | 8260 | 6715 | 7.74e-03 | 0 | 6.50e-01 | 0 | NONREPRODUCIBLE |
| density_r2c0 | 8260 | 7204 | 1.24e-02 | 0 | 7.31e+09 | 0 | NONREPRODUCIBLE |
| density_r2c1 | 8260 | 6120 | 9.26e-03 | 0 | 1.00e+00 | 0 | NONREPRODUCIBLE |
| density_r2c2 | 8260 | 7274 | 7.75e-03 | 0 | 1.00e+00 | 0 | NONREPRODUCIBLE |
| area | 8260 | 2459 | 2117 | 13 | 5.97e-02 | 0 | NONREPRODUCIBLE |
| tortuosity_index | 8254 | 3215 | 6.33e-02 | 8.96e-05 | 5.41e-02 | 0 | NONREPRODUCIBLE |
| median_tortuosity | 8254 | 6199 | 2.59e-02 | 0 | 2.23e-02 | 0 | NONREPRODUCIBLE |
| overall_length | 8260 | 3308 | 729.458 | 0.5858 | 2.13e-01 | 0 | NONREPRODUCIBLE |
| median_branching_angle | 8201 | 7686 | 57.169 | 0 | 8.82e-01 | 1 | NONREPRODUCIBLE |
| n_startpoints | 8260 | 8207 | 1 | 0 | 1.00e+00 | 0 | NONREPRODUCIBLE |
| n_endpoints | 8260 | 7616 | 6 | 0 | 1.00e+00 | 0 | NONREPRODUCIBLE |
| n_intersections | 8260 | 7222 | 6 | 0 | 3.33e-01 | 0 | NONREPRODUCIBLE |
| fractal_d0 | 8260 | 3979 | 5.05e-02 | 3.04e-05 | 4.26e-02 | 0 | NONREPRODUCIBLE |
| fractal_d1 | 8260 | 2083 | 7.56e-02 | 4.29e-05 | 6.41e-02 | 0 | NONREPRODUCIBLE |
| fractal_d2 | 8260 | 2082 | 8.45e-02 | 4.82e-05 | 7.18e-02 | 0 | NONREPRODUCIBLE |
| singularity_length | 8260 | 2080 | 1.717 | 1.25e-04 | 2.99e+00 | 0 | NONREPRODUCIBLE |

```
*** HISTORICAL_FEATURE_REPRODUCIBILITY = FAIL ***
BITWISE_IDENTICAL:       0
NUMERICALLY_EQUIVALENT:  0
NONREPRODUCIBLE:        23
```

`UNAFFECTED_ROWS_EXACT_OR_EQUIVALENT = 0`: no single row reproduces all 23 features. Per-feature
exactness ranges from 8207/8260 (`n_startpoints`) down to 2080/8260 (`singularity_length`).

Per section G of the task brief, interpretation stops here until the cause is established.

---

## F. Why the gate failed (G) — the historical mask artifacts no longer exist

### F1. What the failure is *not*

* **Not nondeterminism.** Section H of the lineage document: three identical runs gave bitwise
  identical values for all 23 features, with identical NaN placement, and the installed `PVBM`
  package contains no unseeded randomness.
* **Not a package-version mismatch.** The venv matches `requirements-lock.txt` on all six pins.
* **Not a row misalignment.** `area` and `vessel_pixels` are integer pixel counts; they agree with
  each other on all 8870 rows in both tables. Under a row shift, `n_startpoints` — a graph
  property that varies wildly between images — would be wrong almost everywhere. It differs on
  only **53 of 8260** rows.
* **Not a wrong-mask substitution for the unaffected rows.** Those rows read the same path in both
  tables.

### F2. What it is

`vessel_pixels` and `area` are pure integer functions of the mask. Both fail on **5801 of 8260**
unaffected rows, and the two mismatch sets are *identical* (`outside vessel_pixels set = 0`). The
only way the same path can yield a different pixel count is if the file at that path is not the
file that was read historically.

Two independent measurements confirm it.

**(a) The historical values are absent from every mask on disk.** `vessel_pixels` was computed for
**all 8960** mask PNGs under `data/masks`, yielding 8621 distinct values. Of the 5801
non-reproducing rows:

| quantity | value |
|---|---|
| historical value achievable by *some* current mask | 529 / 5801 (9.1 %) |
| historical value present in **no** current mask | **5272 / 5801 (90.9 %)** |

The 529 matches are chance collisions of a pixel count, not file identity. **90.9 % of the
historical pixel counts cannot be produced by any mask that still exists.**

**(b) The differences look like a re-run of segmentation, not a swap.** Distribution of
`historical − current` over those 5801 rows:

| statistic | value |
|---|---|
| median | −4 pixels |
| mean | −8 pixels |
| min / max | −1908 / +2117 |
| median relative | −0.0123 % |
| max relative | 6.34 % |
| within ±100 pixels | 5225 / 5801 (90.1 %) |
| within ±500 pixels | 5703 / 5801 (98.3 %) |
| positive / negative / zero | 2644 / 3157 / **0** |

Small, bidirectional, zero exact matches. A swapped image→mask association would move the values
by a *frame*'s worth of vessel difference and would move `n_startpoints` too. This is the
signature of the same segmentation pipeline run again and landing slightly differently.

**(c) The files are newer than the table.**

```
biomarker_features.csv mtime : Thu Aug 27 01:58:25 2026
sample of 400 unaffected mask files:
  oldest = Sun Aug 30 07:21:06 2026
  newest = Sun Aug 30 07:21:20 2026
  newer than the feature table : 400 / 400
```

Every sampled mask is about three days **newer** than the feature table that supposedly consumed
it.

**(d) The filenames are deterministic per image path, so a re-run overwrites in place.**
`src/segmentation/infer_masks.py:48-50`:

```python
stem = Path(image_path).stem
h = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:8]  # avoid collisions across sources
return f"{stem}_{h}.png"
```

The 8-hex suffix is an MD5 of the **image path**, not of the mask content. Re-running inference for
an image therefore targets the *same filename*. A consistency check sharpens this: the suffix on
disk is **not** the MD5 of the current absolute path under any of five path variants or three
digest algorithms, which indicates the surviving filenames were minted when the project lived at a
different absolute path and later adopted by `find_existing_mask` (`infer_masks.py:54-69`), which
deliberately reuses any existing `stem_<8 hex>.png` rather than insisting on a freshly computed
name.

There is exactly **one generation of masks on disk**: 8960 PNGs for 8870 images, all 8870
canonical references present, all 8264 distinct historical references present, and only 86 files
referenced by no manifest. Nothing was left behind to compare against.

### F3. Conclusion of the diagnosis

> The feature-extraction code, the package versions and the feature definitions are all
> reproducible — Task 4 proved this with three bitwise-identical runs. **The mask PNG files that
> produced `data/features/biomarker_features.csv` were overwritten in place after that table was
> written, and no copy survives.** The historical table's inputs are destroyed, so the historical
> feature values cannot be reproduced from any artifact in the project.

`HISTORICAL_FEATURE_REPRODUCIBILITY = FAIL` and `TASK4_STATUS = INCOMPLETE` follow directly. Per
section U of the brief, classifier reruns do not proceed.

### F4. Consequence for the 610-row delta

The delta between the historical and corrected tables on the 610 affected rows mixes **two**
effects and cannot separate them:

1. the **mask-pairing correction** — the intended subject of this task;
2. the **mask-content regeneration** measured in F2(b), which perturbs *every* image including the
   8260 unaffected ones (median −4 pixels, 90.1 % within ±100 pixels).

Because (2) has a median of ~4 pixels and the affected-row median is ~24 pixels for
`vessel_pixels`, effect (1) is plausibly the larger term but is **not isolated**. Section H below
is reported as a confounded estimate for that reason. This is precisely the situation section G
of the brief requires to be documented rather than interpreted.

---

## G. Per-feature perturbation on the 610 affected rows (H)

From `artifacts/historical_feature_correction_summary.csv`. All 23 features report
`n_total = 8870`, `n_affected_rows = 610`. Deltas are `corrected − historical`.

| feature | n_changed | median abs | p95 abs | max abs | median rel | sign + / − | historical status |
|---|---|---|---|---|---|---|---|
| vessel_density | 415 | 2.85e-05 | 1.14e-04 | 9.48e-04 | 3.95e-04 | 224 / 191 | exploratory |
| vessel_pixels | 415 | 24 | 120.6 | 1165 | 3.95e-04 | 224 / 191 | qc_only |
| density_r0c0 | 82 | 1.46e-04 | 4.38e-04 | 4.26e-03 | 2.59e-03 | 51 / 31 | exploratory |
| density_r0c1 | 174 | 1.46e-04 | 5.51e-04 | 2.25e-03 | 1.37e-03 | 85 / 89 | exploratory |
| density_r0c2 | 69 | 1.46e-04 | 2.93e-04 | 6.16e-04 | 2.62e-03 | 26 / 43 | exploratory |
| density_r1c0 | 132 | 1.46e-04 | 5.52e-04 | 6.87e-03 | 2.05e-03 | 80 / 52 | exploratory |
| density_r1c1 | 171 | 1.47e-04 | 7.80e-04 | 5.87e-03 | 1.38e-03 | 88 / 83 | exploratory |
| density_r1c2 | 130 | 1.47e-04 | 6.19e-04 | 3.55e-03 | 2.13e-03 | 72 / 58 | exploratory |
| density_r2c0 | 72 | 1.46e-04 | 5.52e-04 | 1.65e-03 | 2.90e-03 | 33 / 39 | exploratory |
| density_r2c1 | 137 | 1.46e-04 | 6.29e-04 | 8.64e-03 | 1.59e-03 | 74 / 63 | exploratory |
| density_r2c2 | 51 | 1.47e-04 | 4.40e-04 | 5.28e-04 | 3.96e-03 | 27 / 24 | exploratory |
| area | 415 | 24 | 120.6 | 1165 | 3.95e-04 | 224 / 191 | exploratory |
| tortuosity_index | 370 | 2.40e-04 | 1.40e-03 | 4.08e-03 | 2.12e-04 | 178 / 192 | validated_correct |
| median_tortuosity | 140 | 1.30e-03 | 5.85e-03 | 6.98e-03 | 1.15e-03 | 70 / 70 | validated_correct |
| overall_length | 367 | 2 | 18.02 | 534.10 | 4.40e-04 | 199 / 168 | exploratory |
| median_branching_angle | 38 | 1.28 | 21.24 | 34.19 | 1.54e-02 | 17 / 21 | exploratory |
| n_startpoints | **3** | 1 | 1 | 1 | 0.5 | 1 / 2 | confirmed_defect |
| n_endpoints | 42 | 1 | 2 | 2 | 4.12e-02 | 22 / 20 | confirmed_defect |
| n_intersections | 63 | 1 | 2 | 4 | 2.94e-02 | 35 / 28 | confirmed_defect |
| fractal_d0 | 309 | 9.89e-05 | 1.04e-03 | 2.20e-02 | 7.23e-05 | 174 / 135 | exploratory |
| fractal_d1 | 436 | 7.12e-05 | 6.67e-04 | 5.93e-02 | 5.37e-05 | 229 / 207 | exploratory |
| fractal_d2 | 436 | 8.08e-05 | 6.59e-04 | 6.21e-02 | 6.02e-05 | 228 / 208 | exploratory |
| singularity_length | 436 | 2.30e-04 | 1.54e-02 | 1.33 | 1.52e-04 | 217 / 219 | exploratory |

No feature is changed on all 610 rows, and no feature is unchanged on all 610 rows.

**The headline observation.** `n_startpoints` changes on only **3 of 610** affected rows, and
`vessel_pixels` on **415 of 610** with a median of **24 pixels out of roughly 50,000 (≈0.05 %)**.
The reason is in section D: in 606 of 610 cases the historical pairing pointed at a **sibling
frame of the same patient/exam**. Consecutive fundus frames of one eye produce near-identical
vessel masks, so substituting one for another perturbs the biomarkers only slightly. **The 610-row
mask-pairing defect is a real defect and a real contamination of the historical Branch A and
Branch C inputs, but its measured effect on feature values is small, not catastrophic.** That
statement is subject to the F4 confound and to the absence of any clinical interpretation, which
section R defers.

## H. The different-label donor subset (I)

19 of the 610 affected rows had a donor (the image whose mask was used) with a **different label**,
while remaining in the same group. Recipient labels in that subset: plus 9, pre_plus 8, normal 2.
The remaining 587 affected rows had a same-label donor.

Median normalised perturbation (`median |Δ| / max|historical|`), different-label versus
same-label:

```
MEDIAN NORMALISED PERTURBATION RATIO (difflabel / samelabel) = 1.555
```

The 19 different-label cases are perturbed by roughly **1.6×** the same-label cases. Per feature
the ratio ranges from 0.44 (`median_branching_angle`) to 4.89 (`overall_length`), with
`vessel_pixels` at 2.06 and `mean_tortuosity` at 4.51.

This **does not establish label leakage** — the donor's label was never used as an input, and the
two labels concerned are both present in the group. It quantifies the most concerning subset of the
defect: 19 rows where an image labelled `plus` or `pre_plus` received the vessel geometry of a
sibling frame labelled differently. Full per-pair comparison:
`_private_audit/task4_difflabel_donor_subset.csv`.

## I. Split-level contamination (J)

Verified independently in section D: train 414 (6.67 %), val 106 (7.98 %), test 90 (6.76 %).
`artifacts/historical_feature_split_contamination.csv` carries the same numbers.

Per-split perturbation, averaged over the 23 features (mean of the per-feature medians):

| split | mean-of-medians | mean p95 |
|---|---|---|
| train | 2.371 | 11.85 |
| val | 2.449 | 19.23 |
| test | 2.305 | 9.26 |

The three splits are perturbed to a similar degree; the test split is not privileged. Full
per-feature, per-subgroup table (also broken down by source, label and geometry):
`_private_audit/task4_subgroup_perturbation.csv`.

**No classifier was run** (R).

## J. Feature-contract cross-check (K) and limitation markings (L)

`configs/feature_contract.csv` was joined on `(feature_name, feature_version = pvbm_v1)` and its
`clinical_status` recorded in the `historical_status` column of
`artifacts/historical_feature_correction_summary.csv`. Statuses present among the 23:
`exploratory` 17, `qc_only` 1 (`vessel_pixels`), `validated_correct` 2 (`tortuosity_index`,
`median_tortuosity`), `confirmed_defect` 3 (`n_startpoints`, `n_endpoints`, `n_intersections`).

**This metadata altered no value.** The corrected table reproduces every historical feature
faithfully, including the three marked `confirmed_defect`.

Per the brief's section L, the acquisition-sensitive features are marked in the artifact with
`limitation_marking = HISTORICALLY_REPRODUCED_SCI_LIMITATION_KNOWN`:
`n_startpoints`, `n_endpoints`, `overall_length`, `area`, `vessel_density`, `vessel_pixels` and all
nine `density_r{0,1,2}c{0,1,2}` features. **None of them was corrected in Task 4.** No FOV
normalisation, no disc replacement, no startpoint removal, no tortuosity substitution.

## K. Artifacts produced (P, Q)

| artifact | visibility | contents |
|---|---|---|
| `artifacts/historical_feature_correction_summary.csv` | public, committed | 23 rows, one per feature, with the columns specified in section P plus `n_nan_mismatch`, sign counts, `historical_status` and `limitation_marking` |
| `artifacts/historical_feature_split_contamination.csv` | public, committed | 3 rows: split, n_total, n_wrong_mask_historical, percent_wrong |
| `_private_audit/historical_feature_row_deltas.csv` | private, on the analysis host | 14,030 rows = 610 images × 23 features; historical value, corrected value, delta, absolute and relative delta; `stable_image_id` and `hashed_group_id` are SHA-256 truncated to 16 hex characters — **no patient UUIDs** |
| `_private_audit/task4_subgroup_perturbation.csv` | private | perturbation by split, source, label and geometry |
| `_private_audit/task4_difflabel_donor_subset.csv` | private | per-feature comparison of the 19 different-label rows |
| `_private_audit/task4_unaffected_reproducibility.csv` | private | the full section E table |
| `_private_audit/task4_reproducibility.json`, `task4_delta_summary.json` | private | machine-readable summaries |
| `_private_audit/pixel_sha256_cache.csv` (Task 3) | private | unrelated, listed for completeness |

The public correction summary contains no patient identifiers and no image paths.

## L. Status labels (S)

Applied from this task onward:

* `HISTORICAL_CONTAMINATED` — the original Branch A feature table and results, and the original
  Branch C biomarker input and results. They consumed the wrong mask for 610 of 8870 rows and a
  non-canonical 8947-image population.
* `HISTORICAL_EQUIVALENT_CORRECTED` — `data/features/biomarker_features_historical_equivalent_corrected_v1.csv`:
  canonical 8870, canonical masks, historical feature definitions unchanged. **This is the only
  label the new table carries.**
* *Corrected measurement layer* — reserved for later clinical-v4-type feature definitions.
  **Not used here.** No prediction, AUC or performance statement in this document carries any
  label, because no model was run.

The three labels are never mixed.

## M. What Task 4 does and does not license (U)

**Licensed.**

* The exact historical feature extraction lineage is proven:
  `extract_pvbm` (`roi=whole`, PVBM 3.0.1.0) → `refresh_feature_splits`, with a feature-computation
  diff of exactly zero across the file's entire history (`docs/PVBM_HISTORICAL_LINEAGE.md`).
* Historical Branch A and Branch C consumed the *same* table, `data/features/biomarker_features.csv`,
  with the *same* 23-feature schema (`BRANCH_A_FEATURE_SCHEMA_IDENTICAL_TO_C = YES`).
* Historical Branch A/C tortuosity is PVBM tortuosity; the local `geometry_core.py` raster defect
  did not reach them.
* A corrected table exists, is internally valid, matches the locked canonical split exactly on
  rows, IDs, sources, labels, splits, feature count and column order, and has a recorded byte-level
  SHA-256. The 610-row and 414/106/90 affected sets are independently reproduced.
* The perturbation of each feature on the 610 rows is quantified, with the most concerning
  different-label subset isolated and shown to be ~1.6× worse.

**Not licensed.**

* **`TASK4_STATUS = INCOMPLETE`.** Success gate item 4 — "8,260 unaffected rows reproduce historical
  values or every deviation is explained" — is only half met: the deviation is explained, but the
  explanation is that the inputs were destroyed, not that the reconstruction is faithful. Task 4
  therefore does **not** establish that the corrected table differs from the historical table
  *only* by cohort membership and mask pairing. That proposition remains unproven.
* **No classifier was trained and none should be trained from this table until the question below is
  settled** (R): the corrected table is a valid, reproducible measurement of the canonical cohort
  under the *current* masks, but it is not a controlled counterfactual against the historical
  results, because the current masks are not the historical masks.
* **The 610-row delta is confounded** (F4) by the mask regeneration that affects all 8870 images.
  The per-feature numbers in section G are upper-bounded estimates of the pairing effect, not
  isolated measurements of it.
* **No clinical interpretation is offered.** Section H of the brief defers it explicitly.

### The decision this task hands back

There are now two defensible readings, and they lead to different Task 5 designs:

1. **Treat the regenerated masks as the measurement layer.** The historical results are
   `HISTORICAL_CONTAMINATED` for two independent reasons (non-canonical cohort, wrong pairing) and
   also measured under a superseded segmentation. Rebuild Branch A/C and the LOSO on the canonical
   8870 with the **current** masks — the table already built here — and report that as the
   corrected experiment, stating plainly that the historical numbers are not recoverable and never
   were comparable.
2. **Recover the historical masks first.** Locate an archived copy of the pre-2026-08-30 mask set
   (a backup, a drive snapshot, a zipped artifact) and re-run the gate. If the 8260 rows then
   reproduce, the delta in section G becomes a clean attribution measurement and the historical
   Branch A/C results acquire a defensible correction rather than a replacement.

Reading 2 is strictly more informative and should be attempted before committing to reading 1,
because it is cheap to test and it is the only path that preserves the attribution experiment the
brief was designed around. If no archived mask set exists, reading 1 is the honest fallback and
`HISTORICAL_FEATURE_REPRODUCIBILITY` must be recorded permanently as `FAIL` for this project.

**Until that is decided, no classifier is trained (R).**
