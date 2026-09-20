# Dataset census and 8,870 vs 8,947 reconciliation — Task 2

```
CANONICAL_POPULATION_N:             8870
CANONICAL_UNIQUE_IMAGE_N:           8870

HISTORICAL_LOSO_POPULATION_N:       8947
HISTORICAL_LOSO_UNIQUE_IMAGE_N:     8947

LOSO_ONLY_UNIQUE_IMAGES:              77
CANONICAL_ONLY_UNIQUE_IMAGES:          0

PLUS_DIFFERENCE:                      73
FARFUM_DIFFERENCE:                     4
FARABI_DIFFERENCE:                     0

HISTORICAL_LOSO_STATUS:  NONCANONICAL_AND_REQUIRES_RERUN

DIVERGENCE_POINT:
  data/masks/mask_manifest_legacy_image_level_20260826.csv  (8960 rows, dated 2026-08-26)
  minus the 13 farabi rows that the LOSO artifacts do not contain
  versus
  data/splits/all.csv  (8870 rows, dated 2026-08-27), built AFTER the LOSO ran

FARABI_FOLD_EXTRA_TRAIN_IMAGES:       77
FARFUM_FOLD_EXTRA_TRAIN_IMAGES:       73
PLUS_FOLD_EXTRA_TRAIN_IMAGES:          4

RERUN_REQUIRED:  YES
```

---

## A. Repository evidence

| field | value |
|---|---|
| HEAD | `22df490` — *Task 1: mask pairing root-cause forensics* |
| branch / status | `main`, working tree clean |
| recent log | `22df490`, `0101a6b`, `bf9d9aa`, `77c1f6c`, `9d1c72a`, `6750f62`, `dd18a84`, `2f0214a`, `e3ea17e`, `8a3c455`, `6927575`, `204b1b2`, `243f23a`, `0b96808`, `6fec680` |

Artifacts used, with hashes measured on the compute host in this task:

| artifact | rows | sha256 |
|---|---|---|
| `data/features/biomarker_features.csv` | 8870 | `d3fb94dbe2fc6b9b13fc02163e9a79df70df3b5c380701878bd3f148e2d92aa7` |
| `data/splits/all.csv` | 8870 | `0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8` |
| `data/masks/mask_manifest_legacy_image_level_20260826.csv` | **8960** | `aae3ec84da2d32ab725716159f93e1b73343bb78b1d73c72d3daab78f2b9ebb9` |
| `results/loo_results_full.json` | — | `2e2cb468c0c256a62f922400d333735a44ab3fea7b82b35f9185969c9ff9b88f` |
| `results/loo_results_partial.json` | — | `d5512f520da297301246c2ddf47feb43effbbb318da135cad3bcfbf6b07cd0a1` |
| `data/splits/excluded_ambiguous_exact_duplicates.csv` | 54 | `3ca1473db51fd3b4…` |
| `data/splits/missing_masks.csv` | 9 | `b2a036066b362476…` |
| `data/splits/rerun_masks.csv` | 9 | `47616cc4cff058c0…` |

`data/splits/all.csv` sha256 equals the locked split fingerprint exactly, so the canonical cohort is
the frozen one.

---

## B. The canonical 8,870, derived not assumed

```
biomarker_features.csv            8870 rows, 8870 unique image_path
data/splits/all.csv               8870 rows, train 6211 / val 1328 / test 1331, 414 groups
source counts                     plus 5931 / farfum_rop 1529 / farabi 1410
                                  -> total 8870
```

Reproduced from `data/splits/all.csv`, `biomarker_features.csv` and `split_counts.csv`, which all
agree. Timestamp `2026-08-27 01:57`.

## C. The historical LOSO 8,947, derived not assumed

`results/loo_results_full.json` records `plus_ovr` per hold-out. The `n_test` field of branch A is the
size of the held-out source:

| hold-out | `n_test` |
|---|---|
| `farabi` | 1410 |
| `farfum_rop` | 1533 |
| `plus` | 6004 |
| | **8947** |

The farabi row was produced by the Task-1-era run; `loo_results_partial.json` records the same
`farabi = 1410`. The legacy manifest's own per-source counts are plus 6004 / farfum_rop 1533 /
**farabi 1423**, i.e. 8960 rows.

## D. The third artifact that closes the gap

`data/masks/mask_manifest_legacy_image_level_20260826.csv`, **dated 2026-08-26** — the same day the
LOSO checkpoints and `loo_partial.json` were written:

```
rows                8960
unique image_path   8960
sources             plus 6004 / farfum_rop 1533 / farabi 1423
labels              0: 6530, 2: 1493, 1: 937
mask_path on disk   18 of 8960 do not exist
split               train 6272 / val 1344 / test 1344
```

The LOSO population is this 8960-row population restricted to `farabi = 1410`, i.e. the farabi part
of the LOSO equals the canonical farabi set while the plus and farfum parts are the larger legacy
sets. That reconstruction is deterministic and reproduces **all three** source counts exactly:

```
8947 rows   plus 6004 / farfum_rop 1533 / farabi 1410   -> matches the LOSO artifacts
```

## E. Set difference on stable image identity

| quantity | value |
|---|---|
| canonical rows / unique images | 8870 / 8870 |
| LOSO rows / unique images | 8947 / 8947 |
| intersection | **8870** |
| **LOSO-only** | **77** |
| **canonical-only** | **0** |

`CANONICAL_ONLY = 0` is the important structural fact: **the canonical cohort is a strict subset of
the historical LOSO population.** No canonical image is missing from LOSO; LOSO simply contains 77
extra ones.

The difference **is** exactly 77 unique images, and it decomposes as:

| source | LOSO | canonical | difference |
|---|---|---|---|
| plus | 6004 | 5931 | **+73** |
| farfum_rop | 1533 | 1529 | **+4** |
| farabi | 1410 | 1410 | 0 |

## F. Classification of the 77

Mutually exclusive, computed against `data/splits/excluded_ambiguous_exact_duplicates.csv`:

| reason | count |
|---|---|
| `exact_duplicate_excluded_from_canonical` | **54** |
| `UNKNOWN` | **23** |

All 54 documented exclusions are `plus`, all are absent from canonical, and all carry the recorded
reason `identity_linked_by_cross_patient_exact_duplicate`. The remaining **23** — 19 `plus` and 4
`farfum_rop` — are **not** in the excluded table, **not** in `missing_masks.csv` and **not** in
`rerun_masks.csv`. They are left as **UNKNOWN** rather than forced into a category.

Group relationship: **0 of 77** share a group with any canonical image; their groups are absent from
the canonical cohort entirely. Labels: normal 67, plus 9, pre-plus 1.

`missing_masks.csv` holds 9 farfum_rop rows and `rerun_masks.csv` 9 rows across all three sources;
neither table accounts for the 23.

Private table: `_private_audit/loso_vs_canonical_forensics.csv`, 77 rows.

## G. How the historical LOSO script built its dataset

`src/compare/leave_one_source_out.py`:

| question | answer |
|---|---|
| read the canonical split manifest? | only for grouping metadata; the row set comes from the feature table |
| input table | `data/features/biomarker_features.csv` |
| scan raw source directories? | no |
| read an older metadata table? | no |
| merge tables? | grouping columns joined from `data/splits/all.csv` |
| apply canonical exclusions? | **no** — it has no exclusion logic at all |
| its own deduplication? | **no** |
| patient/group IDs from the canonical split? | yes, joined |
| input population versioned or hashed? | **no** — it reads whatever the feature table currently is |

It defines the hold-out as `feats_all[feats_all["source"] == holdout]` and the training pool as
everything else. So the population is whatever the feature table contained **at the moment it ran**.
On 2026-08-26 that was the 8947-row pre-canonical table; the canonical 8870-row table was written on
2026-08-27.

### Lineage

```
raw source inventories (plus / FARFUM-RoP / Farabi)
        |
        v
mask_manifest_legacy_image_level_20260826.csv     8960 rows   <- LOSO's ancestor, 2026-08-26
        |
        |  feature extraction; 13 farabi rows absent from the LOSO artifacts
        v
historical feature table                          8947 rows   <- what LOSO read
        |
        +--> leave_one_source_out.py --> folds --> loo_results_*.json   (8947 population)
        |
        v
exact-duplicate and ambiguity exclusions (54 plus rows, plus 23 further removals)
        |
        v
data/splits/all.csv + biomarker_features.csv      8870 rows   <- canonical, 2026-08-27
```

## H. Fold-level impact — the population that matters is the TRAINING one

| hold-out | canonical holdout | LOSO holdout | LOSO-only in **holdout** | LOSO-only in **TRAIN** | canonical train | LOSO train |
|---|---|---|---|---|---|---|
| farabi | 1410 | 1410 | 0 | **77** | 7460 | 7537 |
| farfum_rop | 1529 | 1533 | 4 | **73** | 7341 | 7414 |
| plus | 5931 | 6004 | 73 | **4** | 2939 | 2943 |

**The farabi hold-out count is identical at 1410, yet its training pool contains all 77 LOSO-only
images.** An unchanged hold-out count does not mean an unaffected fold.

## I. Artifacts currently claiming to be LOSO results

| artifact | rows / rows used | population | n per hold-out | status |
|---|---|---|---|---|
| `results/loo_results_full.json` | 9 rows (3 hold-outs × A/B/C) | 8947 | 1410 / 1533 / 6004 | **noncanonical** |
| `results/loo_results_partial.json` | 3 rows (farabi only) | 8947 | 1410 | **noncanonical** |
| `results/loo_summary_full.csv` | 9 rows | 8947 | as above | **noncanonical** |
| `artifacts/loo_auc_with_ci.csv`, `artifacts/loo_fusion_delta_with_ci.csv` | derived | 8947 | as above | **noncanonical** |
| `results/loo/loo_b_*_efficientnet_b5.pth`, `*.legacy_backup` | 3 checkpoints | 8947 | — | **trained on a noncanonical pool** |
| `results/loo/<holdout>_branch_{a,b,c}_test_preds.csv` | farabi only from the original run | 8947 | — | **noncanonical** |
| `results/next_architecture/` E2/E8 development rows | development | not reconciled here | — | **UNKNOWN** |

No AUC is reconciled in this task, by instruction. Only lineage.

**If historical models were trained using LOSO-only images, dropping those rows from the stored
hold-out predictions does not reconstruct a canonical experiment.** The farabi fold's checkpoint was
trained on 77 images that the canonical cohort excludes, and the farfum fold's on 73. A canonical
LOSO requires retraining, not row filtering.

## J. Regression — nothing altered

`mask_manifest_canonical_v1.csv`, the canonical mask mapping, the historical feature tables, the
historical A/B/C result files, the expert Pilot 30, the Final 90, the clinician ZIP, the split
assignment and the model weights are all unchanged. No resampling, no retraining, no feature
regeneration. The only new public files are this report and the census summary; the only new private
files are under `_private_audit/`.

---

## Consequence

The historical LOSO grid was computed on **8,947** images while the canonical cohort contains
**8,870**, and the canonical cohort is a strict subset of the LOSO one. Every LOSO fold trained on at
least 4 and at most 77 images that the canonical cohort excludes, including the **farabi** fold whose
hold-out count happens to be identical. No LOSO number can be presented as a canonical result.

The historical LOSO results are **historical only** until re-run on the canonical cohort. That re-run
is not performed here, by instruction.
