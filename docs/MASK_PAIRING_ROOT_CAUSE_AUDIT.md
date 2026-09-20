# Mask pairing root-cause forensics — Task 1

```
ROOT_CAUSE:                                        UNRESOLVED
CANONICAL_MAPPING_VERIFIED_BY_FRESH_INFERENCE:     YES
HISTORICAL_BRANCH_A_AFFECTED:                      YES
HISTORICAL_BRANCH_C_AFFECTED:                      YES
CROSS_SPLIT_DONOR_CASES:                           0
DIFFERENT_LABEL_DONOR_CASES:                       19
```

**`MAPPING_CORRECTED_AND_VERIFIED — HISTORICAL CREATION MECHANISM UNRESOLVED`**

The intended image→mask mapping has been determined and independently **proved by fresh inference**.
How the wrong mapping was created has **not** been determined and is not claimed. The word "fixed"
is used below only for the mapping, never for the mechanism.

---

## A. Repository evidence

| field | value |
|---|---|
| HEAD | `0101a6be5f86a037b32b9d6ad4a0d7f3906fbdc0` |
| branch | `main` |
| `git status --short` | empty (clean) |
| audit commit | `0101a6b Data and feature extraction correctness audit` |

Blob hashes from `0101a6b`:

| path | blob |
|---|---|
| `src/biomarker/geometry_core.py` | `e309571467b61768f2224a7ee4f8247a8eeef26e` |
| `src/segmentation/infer_masks.py` | `3fcda37eab447633a32f0d94c8f6ad51707b801d` |
| `tests/test_feature_invariance.py` | `25a5545ea41f464a84aa55b62225a8bd8c225159` |
| `configs/feature_contract.csv` | `6309c87ef57cee60469ea681cb425e6398a49745` |
| `scripts/repair_mask_pairing.py` | `34603a012f902f121d18e81be51f009d5f231fa5` |

Artifact hashes measured on the compute host in this task:

| artifact | sha256 |
|---|---|
| `data/masks/mask_manifest.csv` | `8badf26b9452a6724c565c82d0aca0e2b4977dfbdfc797e13c946eff80ef159f` |
| `data/features/biomarker_features.csv` | `d3fb94dbe2fc6b9b13fc02163e9a79df70df3b5c380701878bd3f148e2d92aa7` |
| `data/masks/mask_manifest_canonical_v1.csv` | `b3bc6538026d6ad0fc1342043e5ca2d6de56662233bf6d0709d9d875f24b21cd` |
| `data/splits/all.csv` | `0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8` |
| segmentation checkpoint `weights/best_weight_DeepLabV3+_resize_27` | `c373f53813ee60b8…` |
| `expert_validation/blinding_key.csv` (file bytes) | `85da530cef1cd09d83eda399d9597bcce6109335a2c909994268783a9ff0031f` |

---

## B. Why 17,740 = 2 × 8,870

`mask_manifest.csv`, `sha256 8badf26b…`, 3,580,990 bytes, columns
`image_path, mask_path, label, split, source`.

```
rows                      17740
unique image_path          8870
unique mask_path           8874
rows / images              2.000000 exactly
images with k rows         k=2 for all 8870; no image has 1 and none has >2
```

What distinguishes the two rows:

| observation | count |
|---|---|
| images whose two rows carry the **same** `mask_path` | 8260 |
| images whose two rows carry **different** `mask_path` | **610** |
| rows differing in `label` | 0 of 8870 |
| rows differing in `split` | 0 of 8870 |
| rows differing in `source` | 0 of 8870 |

So the manifest is a concatenation of two passes. `label`, `split` and `source` agree in every pair,
so the two passes saw the same split table. They differ **only** in which mask file was attached, and
they differ for exactly 610 images.

For every one of the 610, both files exist on disk and they are different files. Worked examples:

| image | row 0 (taken) | row 1 (correct) |
|---|---|---|
| `…cf.1.jpg` | `…cf.11_bd7c547e.png` sha `8066c3da4e113a19…` | `…cf.1_f3829e03.png` sha `98f180c1672199e7…` |
| `…cf.2.jpg` | `…cf.20_3e3a039b.png` sha `6da6130af77aa0d5…` | `…cf.2_29685f11.png` sha `12a6bdfd483ce187…` |
| `…2f.3.jpg` | `…2f.30_d99062e4.png` sha `651626588dc790fa…` | `…2f.3_27988681.png` sha `4ae2be4cd744c6e3…` |

The pattern is consistent: the row-0 mask belongs to the sibling whose decimal index begins with the
row's own index.

---

## C. Every writer of `mask_manifest.csv`

Searched the **complete history** (`git log --all`) for `mask_manifest`, `mask_path`, `to_csv`,
`concat`, `append`, `drop_duplicates`, `merge`, `glob`, `rglob`, `stem`, `startswith`.

| script | commits | role |
|---|---|---|
| `src/segmentation/infer_masks.py` | `e2b9a48`, `0101a6b` | **the only writer** (`to_csv` at lines 218 and 332) |
| `src/biomarker/extract_pvbm.py` | `e2b9a48` | reader (line 179) |
| `src/biomarker/quality_gates.py` | `e2b9a48` | reader (lines 115, 407) |
| `scripts/verify_phase5_ab_reality.py` | `e2b9a48` | reader (line 100) |

No other historical blob writes the manifest. The mask-matching function has had exactly two
versions in the entire history:

```
e2b9a48   matches = sorted(masks_dir.glob(f"{stem}_*.png"))
0101a6b   matches = sorted(p for p in masks_dir.glob(f"{stem}_*.png") if pat.match(p.name))
```

The prefix-collision hypothesis is **tested and rejected**, on the actual data:

```
image      …cf.1.jpg
its mask   …cf.1_f3829e03.png                      exists on disk
glob("…cf.1_*.png") -> ['…cf.1_f3829e03.png']      one match, and it is the CORRECT file
recorded   …cf.11_bd7c547e.png                     the mask of image _11
```

`…cf.11_bd7c547e.png` cannot match the pattern `…cf.1_*.png`: the character after `…cf.1` is `1`,
not `_`. The single writer, in both of its versions, returns the correct file here.

### `ROOT_CAUSE_UNRESOLVED`

The manifest was not produced by any code state present in this repository. The remaining
hypotheses, listed separately and **not** treated as fact:

1. The manifest was written on the compute host by a working-copy version of `infer_masks.py` that
   was never committed, in which the mask lookup differed.
2. A post-processing step outside this repository rewrote `mask_path` for the 610 rows, for example a
   join or a rename keyed on a truncated stem.
3. Two runs were merged by a script that does not exist in the repository, and one run resolved
   masks before the sibling files existed.

Nothing in the repository distinguishes these. Option 1 is the most economical, because
`infer_masks.py` was only introduced to git at the initial public release, so a working copy on the
compute host is the expected state of affairs.

---

## D. Forensic table for all 610 mismatches

Private, gitignored: `_private_audit/mask_pair_mismatch_forensics.csv`, 610 rows, no patient UUIDs
(identifiers are replaced by stable ordinals).

| field | value |
|---|---|
| expected-mask candidates per row | exactly 1 for all 610 |
| `same_split` | 606 true, 4 unknown |
| `same_source` | 606 true, 4 unknown |
| `same_geometry` | 606 true, 4 unknown |
| `same_label` | **587 true, 19 false**, 4 unknown |

| gate | count |
|---|---|
| `CROSS_SPLIT_DONOR_CASES` | **0** |
| `DIFFERENT_LABEL_DONOR_CASES` | **19** |
| cross-group **and** cross-split | 0 |
| cross-group **and** cross-source | 0 |
| cross-group **and** different-label | 0 |
| same-group **but** different-label | **19** |

**No recipient in train was given a test mask, or vice versa.** The grouped split held, because 606
of 610 donors are in the same group and no group spans a split.

The 4 "unknown" rows are not cross-group cases: for those the borrowed mask belongs to an image that
is **not in the dataset at all**, so no donor row exists to compare. They are repaired the same way —
the canonical rule is unaffected — but their donor cannot be characterised.

The 19 different-label cases matter for interpretation: the same eye contributed a mask belonging to
a sibling capture that carries a different grade.

---

## E. Proof that the "expected" mask is the right mask

Filename matching is not evidence, so the frozen pipeline was re-run.

```
device=mps arch=MAnet encoder=resnet34 size=256 thr=0.2 min_area=50 close_k=3
checkpoint: weights/best_weight_DeepLabV3+_resize_27  sha256=c373f53813ee60b8…  exists=True
verification sample: 60 affected + 60 control, stratified over source × split,
                     forced to include every row whose expected mask was missing
```

| comparison | Dice mean | clDice mean | pixel disagreement |
|---|---|---|---|
| fresh inference vs the **registered (wrong)** mask | **0.0860** | 0.0854 | 0.1023 |
| fresh inference vs the **expected (correct)** mask | **1.0000** | 1.0000 | 0.0000 |
| control images vs their own registered mask | **1.0000** | — | — |

- expected beats registered on Dice in **60 / 60** rows
- **60 / 60** rows are **bit-for-bit identical** to fresh inference
- the control set reproduces at Dice 1.0000, so **inference is deterministic** and no tolerance or
  equivalence criterion is needed

`CANONICAL_MAPPING_VERIFIED_BY_FRESH_INFERENCE = YES`

The registered masks score Dice 0.086 against a fresh prediction of the same image, i.e. they are a
different image's mask. The expected masks reproduce exactly.

Artifacts: `_private_audit/task1_fresh_inference_verification.csv`,
`_private_audit/task1_fresh_inference.json`.

---

## F. Downstream consumption

| table | sha256 | rows | generation | mask mapping source | affected rows | consumed by |
|---|---|---|---|---|---|---|
| `data/features/biomarker_features.csv` | `d3fb94dbe2fc6b9b…` | 8870 | `src/biomarker/extract_pvbm.py`, `--manifest data/masks/mask_manifest.csv` | **manifest row 0 for all 8870** | **610 (6.88 %)** | Branch A, Branch C |
| `data/features/biomarker_features_clinical_v3.csv` | (see report) | 8870 | `scripts/clinical_features_v3.py`, reads `mask_path` from the feature table | inherited from row 0 | 610 | diagnostics only |

Measured, not inferred: the feature table's `mask_path` equals **manifest row 0 for 8870 of 8870**
rows and manifest row 1 for **0**. There is no ambiguity about which pass was consumed.

Split of the 610: train **414**, val **106**, test **90**.
Source: plus 377, farfum_rop 118, farabi 115. Label: normal 418, plus 132, pre-plus 60.

---

## G. Branch A / Branch C impact gate

| branch | verdict | evidence |
|---|---|---|
| **Branch A** | **YES** | `src/classify/branch_a_tabular.py` reads `biomarker_features.csv`; 610 of its 8870 rows were generated from a different image's mask |
| **Branch C** | **YES** | `src/classify/branch_c_hybrid.py` concatenates the same biomarker columns onto the frozen embedding, so the same 610 rows enter the fusion matrix |
| **Branch B** | **NO** | RGB-only; it never reads a mask, so this bug cannot reach it |

No model was retrained or re-run in this task. Only the lineage is reported.

---

## H. Repair logic

Adopted as a tested module, `src/segmentation/mask_pairing.py`:
`resolve_mask(image_path, mask_names)` requires the exact stem parse
`<image_stem>_<8 hex>.png`, collapses duplicate basenames across directories, and **raises
`MaskPairingError` on zero or multiple candidates**. It never returns the first candidate and never
matches on a prefix.

`tests/test_mask_pairing.py`, 9 tests, all passing:

| requirement | test |
|---|---|
| exact stem extraction, rejects short and non-hex names | `test_exact_stem_is_extracted` |
| `cf.1` must never select `cf.11` | `test_prefix_sibling_is_not_selected` |
| the strict-prefix form the old glob would have accepted | `test_strict_prefix_with_underscore_is_not_selected` |
| two candidates fail loudly | `test_two_candidates_fail_loudly` |
| missing mask fails loudly | `test_missing_mask_fails_loudly` |
| same basename in two directories is one candidate | `test_duplicate_filenames_in_different_directories_collapse_to_one` |
| mixed path separators | `test_mixed_path_separators_are_handled` |
| independent of filesystem ordering | `test_result_is_independent_of_input_order` |
| uppercase hex is rejected | `test_hash_suffix_case_is_significant` |

Combined with the invariance suite: **31 tests, 31 passing**.

A test written first asserted the rejected glob hypothesis and failed; that is how the hypothesis was
disproved. It was corrected rather than deleted, and the reason is recorded in its docstring.

---

## I. Canonical manifest

```
data/masks/mask_manifest_canonical_v1.csv
  rows              8870
  unique image_path 8870
  unique mask_path  8870
  sha256            b3bc6538026d6ad0fc1342043e5ca2d6de56662233bf6d0709d9d875f24b21cd
  original sha256  8badf26b9452a6724c565c82d0aca0e2b4977dfbdfc797e13c946eff80ef159f
  mapping changes vs manifest row 0: 610
```

The original `mask_manifest.csv` is **preserved unchanged**. The canonical manifest is a new file.
No row was guessed: the builder refuses to emit a row unless exactly one candidate exists.

---

## J. Regression check — nothing frozen changed

| item | value | verdict |
|---|---|---|
| expert pilot IDs | 30, `ROP_0002 … ROP_0120` | unchanged |
| expert final IDs | 90, `ROP_0001 … ROP_0119` | unchanged |
| `blinding_key.csv` file sha256 | `85da530cef1cd09d83eda399d9597bcce6109335a2c909994268783a9ff0031f` | **matches the frozen value exactly** |
| locked split | 6211 / 1328 / 1331, 414 groups | unchanged |
| `data/splits/all.csv` sha256 | `0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8` | **the canonical locked hash** |
| Branch B checkpoint `branch_b_efficientnet_b5.pth` | `ef5667d305fdf79a…`, mtime unchanged | unchanged |
| `biomarker_features.csv` | `d3fb94db…` | unchanged |
| `mask_manifest.csv` | `8badf26b…` | unchanged |
| `results/branch_a_results.json` | `c42dd8b16c523d8a…` | unchanged |
| `results/branch_c_results.json` | `1ee2605cd6922503…` | unchanged |
| `results/loo_results_full.json` | `2e2cb468c0c256a6…` | unchanged |
| optic-disc fallback rule | the `disc_valid` rule in `scripts/clinical_features_v3.py` | unchanged |
| clinician ZIP | `0338c81ce129e05d100f711127154b6583c03deabaa10cba3bd3b3c45ebf0091` (on the desktop) | unchanged |

**New files created by this task:** `data/masks/mask_manifest_canonical_v1.csv`,
`src/segmentation/mask_pairing.py`, `tests/test_mask_pairing.py`,
`scripts/forensics_mask_pairing.py`, and the gitignored `_private_audit/` directory.

**A note on the frozen hash's form.** `MANIFEST_SHA256.txt` gives the key's hash as a *pandas
re-serialisation* of its contents, which is not reproducible across pandas versions (pandas 2.3.3 on
the compute host produces a different string). The **file** hash is stable and matches the frozen
value. Future checks should hash the file bytes.

---

## Consequences, stated plainly

Historical Branch A and Branch C results were produced from a feature table in which **610 of 8870
rows (6.88 %) — 414 train, 106 val, 90 test — described a different image's vasculature.** The donors
are the same eye and the same geometry in 606 of 610 cases and the same label in 587 of 610, and no
donor crosses a split boundary, so the historical numbers are unlikely to be reversed by this. They
are nonetheless not the measurement they claim to be, and any new claim must rest on the canonical
mapping.

Nothing was overwritten. The canonical mapping is verified; the creation mechanism is not.
