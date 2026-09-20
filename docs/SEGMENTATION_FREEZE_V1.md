# Segmentation freeze v1 — Task 5A completion report

```
SEGMENTATION_GENERATION_ID:            SEG_CURRENT_V1

CANONICAL_IMAGE_N:                     8870
CANONICAL_MASK_N:                      8870

CHECKPOINT_SHA256:                     c373f53813ee60b89651a04a98bc5f1d6bc60a5a50f45c4e10f42475650d5374
CONFIG_SHA256:                         77a6c9230fd8b6f7230e1085d3415f46ece862d3e160ace983a1be83b97f0a60
INFERENCE_CODE_SHA256:                 b76f820025353841908111674a9ca3460776f7add496d6c2a27ff4bbf09f1a1b

SUBSET_REPRODUCTION_N:                 300
SUBSET_BITWISE_IDENTICAL_N:            300

FULL_REPRODUCTION_RUN:                 YES
FULL_REPRODUCTION_N:                   8870
FULL_BITWISE_IDENTICAL_N:              8870

STRICT_IMAGE_MASK_PAIRING:             PASS

SILENT_OVERWRITE_PROTECTION:           PASS

CURRENT_BASELINE_LABEL:                CURRENT_MASK_HISTORICAL_DEFINITION_BASELINE

TASK5A_STATUS:                         COMPLETE
```

---

## A. Scope

Reproducibility infrastructure only. The segmentation model was not modified or retrained, the
threshold, resize, preprocessing and postprocessing were not changed, the canonical 8870 cohort
and the expert Pilot 30 / Final 90 samples were not touched, no A/B/C branch was trained, no
LOSO was run and no biomarker definition was redesigned.

The single code change in the inference path is **output handling**: a run is now bound to a
generation contract and refuses to overwrite. `post_process`, `unique_mask_name`,
`find_existing_mask`, the model builder, the transforms and the threshold logic are untouched.

---

## B. Generation identity

```
generation_id : SEG_CURRENT_V1
contract      : configs/segmentation_generation_current_v1.yaml
contract sha  : c0be250323acc36ff488193ad014eb18725a523b2c8ae9d1593bd190807f6966
```

`SEG_CURRENT_V1` corresponds to exactly one combination of checkpoint bytes, inference-code
bytes, config bytes, preprocessing, threshold and postprocessing, all recorded explicitly in the
contract. Nothing scientifically relevant is left implicit: the contract fixes the input loader,
input size, resize policy, aspect-ratio policy, normalisation, activation, logit handling,
threshold, the binarization operator, connected-component policy, minimum component area,
morphological operations and their order, output values, output dtype, output resize policy and
the mask read-back threshold.

Verify at any time:

```
python -m src.segmentation.verify_generation --generation SEG_CURRENT_V1
```

It exits 0 only when all four byte hashes still match. A mismatch aborts the run rather than
producing masks of unknown provenance.

### B1. Covered by the contract

| required field | value |
|---|---|
| architecture / encoder | MAnet / resnet34 |
| checkpoint path | `weights/best_weight_DeepLabV3+_resize_27` |
| checkpoint sha256 / size | `c373f538…5374` / 127,358,298 bytes |
| input size | `[256, 256]` |
| resize policy | `torchvision.transforms.Resize((256,256))`, bilinear, before `ToTensor` |
| aspect-ratio policy | not preserved — all geometries stretched to square |
| normalisation | none beyond `ToTensor` scaling to [0,1] |
| sigmoid handling | raw logits through `sigmoid` exactly once |
| threshold | 0.20 |
| connected-component policy | `cv2.connectedComponentsWithStats`, 8-connectivity |
| minimum component area | 50 |
| morphological operations | `MORPH_CLOSE`, 3×3, after component removal |
| binary output values / dtype | `{0, 255}` / uint8 grayscale PNG |
| output resize policy | `cv2.resize(..., INTER_NEAREST)` to source size |
| inference script sha256 | `b76f8200…1a1b` |
| config sha256 | `77a6c923…0a60` |
| created at | 2026-09-20 |
| canonical population fingerprint | `0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8` |

---

## C. Input inventory (private)

`_private_audit/seg_current_v1_inputs.csv` — one row per canonical image:
`stable_image_id, image_sha256, width, height, source, split, group_hash`.

| check | result |
|---|---|
| rows | 8870 |
| unique `stable_image_id` | 8870 |
| unique `image_sha256` | 8870 |
| probe errors | 0 |
| geometry distribution | 640×480 2516 · 1240×1240 2446 · 1280×960 1382 · 1440×1080 969 · 1600×1200 1557 |
| source counts | plus 5931 · farfum_rop 1529 · farabi 1410 |
| split counts | train 6211 · val 1328 · test 1331 |

No patient UUIDs appear in any public artifact; group identity is a truncated sha256.

---

## D. Mask inventory (private)

`_private_audit/seg_current_v1_masks.csv` — one row per canonical image:
`stable_image_id, image_sha256, mask_path, mask_sha256, mask_width, mask_height, vessel_pixel_count, generation_id`.

| check | result |
|---|---|
| rows | 8870 |
| unique `stable_image_id` | 8870 |
| unique `mask_sha256` | 8870 |
| unique `mask_path` | 8870 |
| duplicate mask bytes | 0 |
| probe errors | 0 |
| mask pixel values | `{0, 255}` on all 8870 |
| mask dimensions equal source dimensions | 8870 / 8870 |
| vessel pixel count min / median / max | 4037 / 77833 / 261016 |
| total vessel pixels | 694,116,606 |

Requirements met: exactly 8870 canonical images, exactly 8870 mask mappings, no duplicate image
identity, no ambiguous mask association.

The frozen store directory contains 8960 PNGs: 8870 canonical masks plus 90 belonging to images
outside the canonical cohort. Those 90 are untouched and excluded from the inventory.

---

## E. No more silent overwrite

Three behaviours, all exercised on the analysis host:

| behaviour | result |
|---|---|
| run against a **frozen** generation | refused: *"SEG_CURRENT_V1 is FROZEN (output_store=data/masks). Its masks are the reference bytes for the canonical cohort and must not be recomputed in place."* |
| run with `--no-resume` onto a populated per-generation store | refused: *"refusing to overwrite existing mask … Default behaviour is ERROR, not overwrite."* |
| run with `--allow-overwrite` on a per-generation store | refused: *"--allow-overwrite does not apply to a per-generation store: write the new generation into its own directory instead of overwriting an existing one."* |

Each generation writes only into its own directory. A change of checkpoint, code, config,
preprocessing, threshold or postprocessing requires a **new generation id** and a **new
contract**; the verifier refuses to let an old contract describe a new state.

`schema` for the refusal is `GenerationError` / `MaskOverwriteError` in
`src/segmentation/generation.py`. 21 hermetic tests cover contract loading, field completeness,
id mismatch, byte-hash verification for checkpoint / code / config, store policy, the overwrite
guard and content-based identity.

---

## F. Content-based output identity

Filenames are `md5(image_path)[:8]`, i.e. a function of the *path*. They remain convenient but
carry no provenance. Scientific identity is

```
(image_sha256, generation_id, mask_sha256)
```

and the manifest now records `generation_id`, `image_sha256`, `mask_sha256` and `mask_bytes`
for every row it writes. 8870 distinct mask hashes for 8870 canonical images means no two
canonical images share mask bytes.

---

## G. Reproducibility

Method: the frozen recipe was re-run under a separate verification contract
(`SEG_CURRENT_V1_REPRO`, `purpose: reproducibility_check`, `not_a_scientific_generation: true`,
store `_private_audit/repro_scratch/SEG_CURRENT_V1_REPRO`) so that not one byte was written into
the frozen store. The verifier re-checks the same four hashes, so the check can only pass while
the state really is the `SEG_CURRENT_V1` state.

| gate | N | sha256 identical | pixel identical | vessel-pixel identical |
|---|---|---|---|---|
| stratified subset | 300 | **300** | 300 | 300 |
| full population | 8870 | **8870** | 8870 | 8870 |

Subset coverage: every source (farabi 156, plus 108, farfum_rop 36), every acquisition geometry
(all five), every split (test 152, train 88, val 60), **16 / 16 strata**.

Deterministic output was expected, so `BITWISE_IDENTICAL_N == SUBSET_N` was required and holds.
The full population was regenerated as well and matched byte for byte, so
`FULL_REPRODUCTION_RUN = YES` and no nondeterminism investigation was needed.

---

## H. Strict image → mask pairing

`src/segmentation/mask_pairing.py::resolve_mask` decides by an exact stem parse, never by a
first filesystem match and never by a prefix match. Over the canonical population:

| check | result |
|---|---|
| images | 8870 |
| stored masks | 8960 |
| store filenames that fail to parse as `<stem>_<8 hex>.png` | 0 |
| resolved to exactly one mask | **8870 / 8870** |
| resolution errors (zero or multiple candidates) | 0 |
| resolved mask differs from the pinned canonical mask | 0 |

Failure-mode probes, all correct:

| probe | behaviour |
|---|---|
| zero candidates | `MaskPairingError` |
| exactly one candidate | resolved |
| two candidates for one stem | `MaskPairingError` |
| prefix sibling only, no exact stem | `MaskPairingError` |
| prefix sibling alongside an exact match | exact match selected, sibling ignored |
| unparseable name (`<stem>.png`) | `MaskPairingError` |
| Windows-style image path | resolved identically to the POSIX path |
| upper-case hash suffix | `MaskPairingError` (the contract is `[0-9a-f]{8}`) |

`STRICT_IMAGE_MASK_PAIRING = PASS`. The 9 tests in `tests/test_mask_pairing.py` pass, and the
full suite runs **124 passed**.

---

## I. Public generation summary

`artifacts/seg_current_v1_summary.json` carries the generation id, population, checkpoint /
config / code hashes, threshold, input size, mask count, aggregate dimensions, the
reproducibility results, the pairing verdict and the overwrite-protection policy. It contains no
patient identifiers and no absolute local paths.

---

## J. Historical boundary

See `docs/SEGMENTATION_GENERATION_PROVENANCE.md`. In one line:

* **Historical Aug-27 generation — unrecovered.** It produced
  `data/features/biomarker_features.csv` and the contaminated, non-reproducible historical
  Branch A and Branch C results. It cannot be reconstructed by any available state.
* **`SEG_CURRENT_V1` — frozen and fully specified.** It is the starting point for a new corrected
  measurement pipeline. **It is not historical-equivalent and must not be described as such.**

---

## K. Baseline label

`data/features/biomarker_features_historical_equivalent_corrected_v1.csv`, sha256
`90ce160727347f1d73f571ba4969638550ae534cb13cf891bf814fce92f7a463`, is labelled

```
CURRENT_MASK_HISTORICAL_DEFINITION_BASELINE
```

and **not** `HISTORICAL_EQUIVALENT_CORRECTED`. Its bytes and its filename are unchanged, as
required. Machine-readable sidecar: `artifacts/task4_table_classification.json`.

---

## L. Nothing historical was overwritten

The historical biomarker table, the historical Branch A and Branch C results, the historical
LOSO artifacts, the historical masks and every forensic artifact from Tasks 1 to 4B remain in
place, including the failed Task-4 reconstruction. This task wrote only to new paths: two new
contracts, one new module, one new CLI, one new test file, two new documents, one new public
artifact, and private inventories under `_private_audit/` (now gitignored).

---

## M. Success gate

| # | requirement | status |
|---|---|---|
| 1 | current generation has a unique immutable id | `SEG_CURRENT_V1` ✓ |
| 2 | checkpoint / config / code byte-hashed | ✓ and re-verified at every run |
| 3 | all 8870 inputs and masks have private byte hashes | ✓ `seg_current_v1_inputs.csv`, `seg_current_v1_masks.csv` |
| 4 | mask association is unambiguous | ✓ 8870/8870 strict, 0 errors |
| 5 | silent overwrite prevented by default | ✓ three refusal behaviours exercised |
| 6 | deterministic reproduction passes | ✓ 300/300 subset and 8870/8870 full, bitwise |
| 7 | historical and current generations explicitly separated | ✓ provenance document |
| 8 | no model or biomarker definition changed | ✓ inference path changed only in output handling |

```
TASK5A_STATUS = COMPLETE
```
