# Cohort exclusion and duplicate forensics — Task 3

```
LEGACY_IMAGE_MANIFEST_N:                8960
QUALITY_EXCLUSIONS_TOTAL_N:               36
QUALITY_EXCLUSIONS_BEFORE_LOSO_N:         13
QUALITY_EXCLUSIONS_BEFORE_CANONICAL_N:    23
EXACT_DUPLICATE_GROUP_REMOVAL_N:          54
EXACT_DUPLICATE_IDENTITIES:                2
EXACT_DUPLICATE_TRIGGERING_HASHES:        19
EXACT_DUPLICATE_TRIGGERING_ROWS:          38
EXACT_DUPLICATE_DRAGGED_SIBLING_ROWS:     16

HISTORICAL_LOSO_POPULATION_N:           8947
CANONICAL_POPULATION_N:                 8870

LOSO_ONLY_UNIQUE_IMAGES:                  77
  LOSO_ONLY_EXACT_DUPLICATE_N:            54
  LOSO_ONLY_QUALITY_EXCLUSION_N:          23
  LOSO_ONLY_UNKNOWN_N:                     0
CANONICAL_ONLY_UNIQUE_IMAGES:              0

ARITHMETIC_CLOSURE_LOSO:        8960 - 13      = 8947
ARITHMETIC_CLOSURE_CANONICAL:   8960 - 36 - 54 = 8870

UNKNOWN23_RESOLVED:                      YES
CANONICAL_COHORT_REQUIRES_REVIEW:         NO
CANONICAL_RESIDUAL_EXACT_DUPLICATES:       0
CANONICAL_PER_FOLD_EXACT_DUPLICATE_LEAKAGE: 0

HISTORICAL_LOSO_STATUS:  NONCANONICAL_AND_REQUIRES_RERUN
RERUN_REQUIRED:          YES
```

---

## A. Repository evidence

| field | value |
|---|---|
| HEAD at time of task | `01c7ae5` — *Task 2: canonical dataset census and 8870 vs 8947 reconciliation* |
| branch / status | `main`, working tree clean |
| governing source file | `src/data/prepare_split.py` |
| its sha256 | `5c1af0fd68f21e4afb40fa56b098d387dc5a24a3acc44a91914247ea5f097f3c` |

Input artifacts, all hashed by file bytes:

| artifact | rows | sha256 |
|---|---|---|
| `data/masks/mask_manifest_legacy_image_level_20260826.csv` | 8960 | `aae3ec84da2d32ab725716159f93e1b73343bb78b1d73c72d3daab78f2b9ebb9` |
| `data/metadata/quality_exclusions.csv` | 36 | `500c88d0de4f0a02ed26f7124e0111efb6376bc97e9ee6df0a3be9c3de1949f6` |
| `data/splits/excluded_ambiguous_exact_duplicates.csv` | 54 | `3ca1473db51fd3b497b95ddedcd5de91a8a40281b8f9052d64e9006247f02439` |
| `data/splits/all.csv` | 8870 | `0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8` |
| `data/metadata/identity_map.csv` | 8960 | `5dc3680ddd4c573dd52dcd2c64ac095119ed2e72d2ed8390fd0eeadb833db83d` |

`data/splits/all.csv` reproduces the locked canonical fingerprint
`0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8`, so the locked cohort is
unmodified by this audit.

Reproduction scripts added by this task:

* `scripts/cohort_exclusion_forensics.py` — arithmetic closure, pixel hashing, duplicate
  topology, per-fold leakage, quality-exclusion independence.
* `scripts/cohort_exclusion_corroboration.py` — cross-checks every conclusion against
  `identity_map.csv` and calibrates the perceptual-hash threshold.
* `scripts/cross_registered_identity_metadata.py` — metadata consistency of the two
  identities that account for the 54-row removal.

---

## B. The two exclusion mechanisms, in code

`src/data/prepare_split.py` contains exactly two filters that can shrink the scanned image
set. Both are applied in a fixed order.

### B1. Documented quality exclusions — `prepare_split.py:371-389`

```python
371  quality_exclusions_path = (
372      cfg["paths"]["raw_dir"].parent / "metadata" / "quality_exclusions.csv"
373  )
374  if quality_exclusions_path.exists():
375      quality_exclusions = pd.read_csv(quality_exclusions_path)
376      excluded_paths = set(quality_exclusions["image_path"])
377      quality_excluded = df[df["image_path"].isin(excluded_paths)].copy()
378      missing_exclusions = excluded_paths - set(df["image_path"])
379      if missing_exclusions:
380          raise ValueError(
381              f"Quality exclusions not present in scanned data: {missing_exclusions}"
382          )
383      df = df[~df["image_path"].isin(excluded_paths)].copy()
```

`quality_exclusions.csv` is an explicit, human-authored list. Line 378-382 raises rather than
silently continuing when a listed path is absent from the scan, so the exclusion list cannot
drift out of sync with the data.

**Audit gap (confirmed, now closed by this document).** `quality_excluded` is built at line
377 but never written to disk. Its size reaches the summary JSON only as a bare count
(`prepare_split.py:469`, `"quality_rows": len(quality_excluded)`), and line 385 prints the
count alone. Before this task, therefore, the *identities* of the quality-excluded images
existed only in the source CSV and were not materialised in any split artifact. That is the
single reason the 23 images were unreconciled.

### B2. Cross-patient exact-duplicate group removal — `prepare_split.py:179-192`

```python
179  def exclude_ambiguous_exact_duplicates(
180      df: pd.DataFrame,
181  ) -> tuple[pd.DataFrame, pd.DataFrame]:
182      """Exclude identities linked by cross-patient duplicate pixels."""
183      out = df.copy()
184      out["sha256"] = out["image_path"].map(_sha256)
185      identity_counts = out.groupby("sha256")["group_id"].nunique()
186      ambiguous_hashes = set(identity_counts[identity_counts > 1].index)
187      ambiguous_ids = set(out.loc[out["sha256"].isin(ambiguous_hashes), "group_id"])
188      excluded = out[out["group_id"].isin(ambiguous_ids)].copy()
189      excluded["exclusion_reason"] = "identity_linked_by_cross_patient_exact_duplicate"
190      excluded["triggering_duplicate"] = excluded["sha256"].isin(ambiguous_hashes)
191      retained = out[~out["group_id"].isin(ambiguous_ids)].drop(columns="sha256").copy()
192      return retained, excluded
```

with `_sha256` at `prepare_split.py:171-176` — a plain SHA-256 over the raw file bytes in
1 MiB blocks.

Two properties matter:

1. **The unit of removal is the identity, not the file.** Line 187 collects every `group_id`
   that touches an ambiguous hash; line 188 removes *all* rows of those identities. A row that
   is not itself duplicated is still removed if a sibling in its identity is. This is why 54
   rows are removed when only 38 rows carry a cross-identity twin.
2. **Ambiguity is defined by a pixel hash shared across two different `group_id` values, not
   merely by a repeated pixel hash.** A same-patient repeat capture is retained by design.

Order of operations: quality exclusions (B1) run first at line 383, duplicate removal (B2)
runs second at line 391. Consequently the duplicate detector never sees the quality-excluded
images — a fact that matters in section G.

---

## C. Arithmetic closure of 8960 → 8947 → 8870

```
legacy image manifest                                        8960
  - quality exclusions applied before the LOSO run             13   (all farabi)
  = historical LOSO population                               8947   (8960 - 13)

legacy image manifest                                        8960
  - quality exclusions, total                                  36
  - cross-patient exact-duplicate identity removal              54
  = canonical population                                     8870   (8960 - 36 - 54)
```

Both identities hold exactly, with no residual term. The 36 quality exclusions partition into
the 13 applied before the LOSO run and the 23 applied before the canonical build:

| subset | n | removed before | effect |
|---|---|---|---|
| pre-LOSO quality exclusions | 13 | the LOSO run | 8960 → 8947 |
| post-LOSO quality exclusions | 23 | the canonical build | part of 8947 → 8870 |
| exact-duplicate identity removal | 54 | the canonical build | part of 8947 → 8870 |

`23 + 54 = 77`, which is exactly the LOSO-only image count. The divergence therefore has no
unexplained mass.

---

## D. The 23 quality exclusions — provenance

All 23 are present in `data/metadata/quality_exclusions.csv` (column `image_path`), which is
the file consumed at `prepare_split.py:375`:

| field | value for the 23 |
|---|---|
| `reason` | `phase4_mask_review_exclude` (23 / 23) |
| `phase` | `4` (23 / 23) |
| `source` | `plus` 19, `farfum_rop` 4 |
| manifest label | normal 13, plus 9, pre-plus 1 |

They were excluded in phase 4 for a mask-review reason, i.e. the segmentation mask for these
images failed review. The exclusion is deliberate, documented, dated and attributable.

Independent confirmation: `data/metadata/identity_map.csv` is a phase-2 artifact. For these
23 images it records `included_phase2 = True` and a null `exclusion_reason`, i.e. they were
admissible at phase 2 and were removed later, at phase 4. Its `exact_duplicate_count` is 1 for
all 23, so the phase-2 duplicate analysis did not consider them duplicates either.

For the 13 pre-LOSO exclusions the same file gives `phase = 4`, source `farabi` (13 / 13), and
reasons `phase4_mask_review_exclude` (12) and
`ungradable_fundus_no_visible_vessels_and_empty_segmentation` (1).

**Conclusion.** The 23 are a documented phase-4 mask-review exclusion applied after the
historical LOSO run. They are not evidence of an accidental or unexplained cohort change.

---

## E. The 54 exact-duplicate removals — topology and root cause

### E1. Topology

| quantity | value |
|---|---|
| rows removed | 54 |
| distinct `group_id` values involved | 2 |
| rows carrying a cross-identity twin (`triggering_duplicate = True`) | 38 |
| sibling rows removed only because their identity was implicated | 16 |
| distinct ambiguous pixel hashes | 19 |
| images per ambiguous hash | exactly 2, for all 19 |
| hashes spanning more than one identity | 19 / 19 |
| hashes spanning more than one **source** | 0 / 19 |
| hashes spanning more than one **label** | 0 / 19 |

All 54 rows are `source = plus`, `label = normal`. All 19 ambiguous hashes join the **same**
pair of identities; there is exactly one distinct identity-pair behind the entire removal.
The identity with 21 rows contributes 19 triggering rows and 2 siblings; the identity with 33
rows contributes 19 triggering rows and 14 siblings.

### E2. Root cause

The two identities are consecutive registrations in the public PLUS cohort. Comparing the
filename metadata of the 19 pixel-identical pairs:

| token | agreement |
|---|---|
| gestational age | 19 / 19 |
| birth weight | 19 / 19 |
| disease-grade token | **0 / 19** — systematically different between the two registrations |
| manifest label | 19 / 19 (`normal`) |

Both registrations carry four exams. Identical pixels, identical gestational age, identical
birth weight and consecutive identifiers, differing only in a per-registration disease-grade
token, is the signature of **one infant registered twice under two identifiers**, not of two
patients who happen to share images.

Two consequences:

1. `exclude_ambiguous_exact_duplicates` (`prepare_split.py:179-192`) acted correctly. Removing
   both identities is the conservative and correct response to a double registration; keeping
   either would place the same pixels in a single cohort under a false identity boundary.
2. The disagreement in the disease-grade token is a **data-integrity observation about the
   source cohort** and should be recorded as such. It did not change any label here, because
   both registrations map to `normal`.

The exact identifiers, the token values and the per-pair comparison are held in the private
artifact `_private_audit/cross_registered_metadata_consistency.csv` and are deliberately not
reproduced in this public document.

---

## F. Per-fold exact-duplicate leakage in the canonical split: measured zero

`data/splits/all.csv` (8870 rows) was joined to independently computed SHA-256 values:

| quantity | value |
|---|---|
| canonical images with a computed hash | 8870 / 8870 |
| distinct SHA-256 values | 8870 |
| hashes occurring more than once | **0** |
| hashes spanning more than one identity | **0** |

Every one of the 8870 canonical images is byte-unique. There is no exact-duplicate pair of any
kind left in the cohort, so there is no exact-duplicate path between `train`, `val` and `test`.
The three fold-pair counts are therefore all zero:

| fold pair | shared hashes | shared hashes across identities |
|---|---|---|
| train ↔ val | 0 | 0 |
| train ↔ test | 0 | 0 |
| val ↔ test | 0 | 0 |

This is a stronger result than the design requires. `split_by_patient` keeps `group_id`
indivisible, so even a surviving same-patient repeat could not cross a fold; here there are no
surviving repeats at all.

---

## G. Did the quality exclusions hide duplicates?

No. Of the 36 quality exclusions, **0** share a pixel hash with any retained canonical image.
The two mechanisms are disjoint by construction and disjoint in fact:

* quality exclusions remove images for clinical/technical unusability (a documented list,
  `prepare_split.py:371-389`);
* duplicate removal removes identities linked by cross-patient identical pixels
  (`prepare_split.py:179-192`).

Because the quality exclusions are applied first (line 383), the duplicate detector at line 391
operates on 8924 rows and cannot observe them at all. Neither mechanism is doing the other's
job, and none of the 36 is a disguised duplicate.

---

## H. Independent corroboration

Every Task 3 conclusion was re-derived against artifacts that this audit did not produce.

**H1. Hash correctness.** SHA-256 was recomputed from raw bytes for all 8960 images and compared
with the `sha256` column of `data/metadata/identity_map.csv`:

```
agreement: 8960 / 8960   (100.0000%)
```

**H2. The 54.** `identity_map.csv` independently agrees on all 54:
`sha256` identical 54 / 54, `included_phase2 = False` 54 / 54, and
`exclusion_reason = identity_linked_by_cross_patient_exact_duplicate` 54 / 54. Its
`exact_duplicate_count` is 2 for exactly 38 of the rows and 1 for the other 16, reproducing the
38 / 16 triggering-versus-sibling split of section E1 from an independent source.

**H3. The 23.** `identity_map.csv` records `included_phase2 = True` for all 23, with a null
`exclusion_reason`, confirming they were removed after phase 2 by the phase-4 mask review
recorded in `quality_exclusions.csv`.

```
CORROBORATION_STATUS: PASS
```

---

## I. Near-duplicate analysis: a retracted claim and its calibration

This section records a claim made during the audit **and withdrawn**, because the withdrawal is
methodologically load-bearing for any later near-duplicate work.

Using a 64-bit difference hash on the 23, an initial pass reported that 2 of the 23 had an
"exact twin" inside the canonical cohort (`hamming = 0`) and that 20 of 23 had a neighbour at
`hamming <= 6`. The `sha256` test of section G contradicts any byte-identical twin for those
rows, so the two results cannot both be right.

The dHash threshold was therefore calibrated on the cohort itself, sampling 400 canonical
images and measuring each one's nearest-neighbour dHash distance:

| statistic | value |
|---|---|
| median nearest-neighbour distance | 2 |
| 5th percentile / 25th percentile | 0 / 1 |
| fraction with a neighbour at `hamming <= 6` | **0.9575** |
| fraction with a neighbour at `hamming == 2` or less | 0.542 |
| fraction with an **exact** dHash match (`hamming == 0`) | **0.1550** |

**Verdict: the threshold is saturated and the claim is withdrawn.** In a cohort of same-session
fundus frames, dHash nearest-neighbour distance is not discriminative: 95.8 % of all ordinary
canonical images have a neighbour within the threshold that was used to flag the 23, and 15.5 %
have an exact dHash match without being byte-identical. The observed 2 / 23 and 20 / 23 are
therefore *at or below* the cohort baseline and carry no information about the 23. Byte-identical
twinning is settled by `sha256` alone, and by `sha256` the answer is zero.

The private table `_private_audit/near_duplicate_unknown23.csv` has been rewritten with the
baseline columns, a `verdict` column and an explicit interpretation string so that the
withdrawn reading cannot be cited later.

**Secondary finding worth carrying forward.** The same calibration is a genuine property of the
cohort: no threshold on a 64-bit perceptual hash separates "duplicate" from "routine" in this
data, because consecutive frames of one session are near-identical. Any future near-duplicate
screen must use an adjudicated, feature-based criterion — as the existing
`data/metadata/perceptual_duplicate_adjudication.csv` does with SSIM and ORB ratios — and must
be calibrated against this baseline before any image is excluded on it. It also quantifies why
image-level splitting is unsafe here and grouped splitting is required: the effective number of
independent observations is far below 8870. None of the 23 appears in the existing 89-pair
perceptual audit, so that audit raises no additional concern about them.

---

## J. Private artifacts

Written to `_private_audit/` on the analysis host; not for publication.

| file | contents |
|---|---|
| `task3_summary.json` | all Task 3 scalars |
| `task3_corroboration.json` | corroboration verdict and calibration values |
| `unknown23_forensics.csv` | all 77 LOSO-only rows with reason, hash, membership flags |
| `exact_duplicate_loso_topology.csv` | one row per ambiguous hash: images, identities, sources, labels |
| `canonical_fold_exact_duplicate_leakage.csv` | the three fold-pair counts |
| `near_duplicate_unknown23.csv` | dHash rows with calibration baseline and verdict |
| `cross_registered_metadata_consistency.csv` | per-pair token comparison and the two identifiers |
| `pixel_sha256_cache.csv` | SHA-256 and dHash for all 8960 images |

---

## K. What this licenses, and what it does not

**Licensed.**

* The canonical cohort of 8870 is fully accounted for. There is no unexplained exclusion, no
  LOSO-only image of unknown provenance, and no canonical-only image.
* The canonical cohort contains no exact duplicate, within or across identities, and therefore
  no exact-duplicate leakage between folds.
* `CANONICAL_COHORT_REQUIRES_REVIEW = NO`. The cohort does not need to be reopened on account of
  the 77 images.

**Not licensed.**

* **The historical LOSO results are still `NONCANONICAL_AND_REQUIRES_RERUN`.** This task
  explains *why* the LOSO population differed from the canonical one; it does not make the two
  equal. The LOSO ran on 8947 images, of which 77 are excluded from the canonical cohort, and
  those 77 sat in the *training* pool of every fold that did not hold their own source
  (Task 2: `FARABI_FOLD_EXTRA_TRAIN_IMAGES = 77`, `FARFUM_FOLD_EXTRA_TRAIN_IMAGES = 73`,
  `PLUS_FOLD_EXTRA_TRAIN_IMAGES = 4`). Because those images were used for training, the
  contamination cannot be removed by filtering stored predictions. **A rerun is required.**
* **The 610-image mask-pairing defect is untouched by this document.** Historical Branch A and
  Branch C remain contaminated by it (Task 1). This task reconciles cohort *membership*; it does
  not reconcile any AUC.
* No statement is made here about near-duplicate redundancy between identities beyond the two
  that account for the 54. Section I shows that the available perceptual hash cannot settle that
  question, and the adjudicated 89-pair review is the only current evidence on it.
