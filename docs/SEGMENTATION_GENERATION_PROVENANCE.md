# Segmentation generation provenance

This document separates two vessel-mask generations that must never be confused. One is
lost; the other is frozen from now on.

```
                         RGB bytes → masks → PVBM features → Branch A / Branch C
                         ─────────────────────────────────────────────────────
HISTORICAL (2026-08-27)  │  unknown state        UNRECOVERED   → contaminated A/C
CURRENT    (2026-08-30)  │  SEG_CURRENT_V1       FROZEN        → baseline, NOT historical
```

---

## 1. Historical generation — 2026-08-27 — UNRECOVERED

| | |
|---|---|
| what it produced | `data/features/biomarker_features.csv`, 8870 × 32, written `2026-08-27 01:58:25` |
| consumed by | historical Branch A and Branch C, and the historical LOSO artifacts |
| checkpoint / threshold / code | **UNKNOWN** — no contemporaneous record survives |
| status | `HISTORICAL_SEGMENTATION_STATE_IRRECOVERABLE` |
| classification | `HISTORICAL_CONTAMINATED_NONREPRODUCIBLE` |

**What is known.** The mask files at the historical paths had different content on 2026-08-27
than they do today, and the change happened after `2026-08-27 02:04`. The current masks are
exactly what the surviving checkpoint produces at threshold 0.20, and no threshold on that
checkpoint and no other recoverable checkpoint reproduces the historical values.

**What was done to try to recover it.** All ten checkpoints named in `configs/config.yaml` were
downloaded from Google Drive, hashed and scored against the historical `area` on 300 rows drawn
exclusively from the 5,801 unaffected rows whose historical value genuinely differs. Every one
scored **0 / 300**. The repository history, the `_remote/` project mirror, the whole local
filesystem under `$HOME`, `~/.Trash`, `~/.cache`, the project backup, Time Machine state, APFS
snapshots, shell history and every log were searched. No pre-2026-08-30 mask set exists.

Full record: **`docs/HISTORICAL_SEGMENTATION_RECOVERY.md`**.

**What this means.** The historical Branch A and Branch C results cannot be corrected; they can
only be replaced. They are not a baseline and must not be cited as a current measurement.

---

## 2. Current generation — `SEG_CURRENT_V1` — FROZEN

| | |
|---|---|
| generation id | `SEG_CURRENT_V1` |
| contract | `configs/segmentation_generation_current_v1.yaml` |
| architecture / encoder | MAnet / resnet34 |
| checkpoint | `weights/best_weight_DeepLabV3+_resize_27`, sha256 `c373f53813ee60b89651a04a98bc5f1d6bc60a5a50f45c4e10f42475650d5374`, 127,358,298 bytes, mtime 2026-08-22 |
| inference code | `src/segmentation/infer_masks.py`, sha256 `b76f820025353841908111674a9ca3460776f7add496d6c2a27ff4bbf09f1a1b` |
| config | `configs/config.yaml`, sha256 `77a6c9230fd8b6f7230e1085d3415f46ece862d3e160ace983a1be83b97f0a60` |
| preprocessing | `PIL.convert("RGB")` → `Resize((256,256))` bilinear → `ToTensor` ([0,1]); **no mean/std normalisation**; aspect ratio deliberately not preserved |
| decision rule | `sigmoid` once → strict `prob > 0.20` → 8-connected components with area `< 50` dropped → `MORPH_CLOSE` 3×3 |
| output | uint8 PNG `{0,255}`, `cv2.resize(..., INTER_NEAREST)` back to source size |
| store | `data/masks` — **frozen** |
| population | canonical 8870, fingerprint `0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8` |
| public summary | `artifacts/seg_current_v1_summary.json` |

### 2.1 Identity is content, not filename

Mask filenames are `md5(image_path)[:8]`, so a re-run targets the same path. That is how the
historical generation was destroyed. Provenance therefore rests on the tuple

```
(image_sha256, generation_id, mask_sha256)
```

and never on a filename.

* inputs: `_private_audit/seg_current_v1_inputs.csv` — 8870 rows, sha256 / width / height /
  source / split / hashed group id. 8870 unique image hashes, 0 probe errors.
* masks: `_private_audit/seg_current_v1_masks.csv` — 8870 rows, mask sha256 / dimensions /
  vessel pixel count / generation id. **8870 unique mask sha256 values.** Every mask is
  `{0,255}` and its dimensions equal the source image's on all 8870 rows.

The frozen store holds 8960 PNGs: the 8870 canonical masks plus 90 files belonging to images
outside the canonical cohort. Those 90 are left untouched and are not part of this generation's
inventory.

### 2.2 Reproducibility

The recipe was re-run under a separate, clearly labelled verification contract
(`configs/segmentation_generation_current_v1_repro.yaml`, purpose `reproducibility_check`,
`not_a_scientific_generation: true`) into a scratch directory. **The frozen store was never
written to.**

| gate | N | bitwise identical (sha256) | pixel identical | vessel-pixel identical |
|---|---|---|---|---|
| stratified subset | 300 | **300** | 300 | 300 |
| full population | 8870 | **8870** | 8870 | 8870 |

Subset coverage: all 3 sources, all 5 acquisition geometries, all 3 splits, 16 / 16 strata.
Output is deterministic, so the required `BITWISE_IDENTICAL_N == SUBSET_N` holds, and the full
population was regenerated as well and matched byte for byte.

### 2.3 Pairing is unambiguous

`src/segmentation/mask_pairing.py::resolve_mask` decides image→mask by an exact stem parse and
never by a first filesystem match. Over all 8870 canonical images:

| check | result |
|---|---|
| resolved to exactly one mask | 8870 / 8870 |
| resolution errors (0 or more than 1 candidate) | 0 |
| resolved mask differs from the pinned canonical mask | 0 |
| store filenames that do not parse as `<stem>_<8 hex>.png` | 0 |

Failure modes are probed explicitly and all behave correctly: zero candidates, two candidates
for one stem, a prefix sibling with no exact match, a prefix sibling alongside an exact match,
an unparseable name, a Windows-style image path and an upper-case hash suffix. Prefix siblings
are never selected; the strict resolver raises rather than guessing.

### 2.4 Silent overwrite is now impossible by default

* Each generation writes only into its own store. A frozen store refuses **every** write, with
  or without the destructive flag.
* A mask that already exists is an error unless `--allow-overwrite` is passed. Default is
  ERROR, not overwrite.
* `--allow-overwrite` is itself rejected for a per-generation store: the correct answer to
  "I need to produce masks again" is a new generation id, not a clobber.
* The contract is re-verified at startup. If the checkpoint, the inference code or the config
  changed, the run aborts instead of producing masks of unknown provenance.

All three behaviours were exercised on the analysis host and all three produced the intended
refusal.

### 2.5 How to add a new generation

1. Change whatever you need to change (checkpoint, threshold, code, config).
2. Create `configs/segmentation_generation_<new_id>.yaml` with a fresh `generation_id`, the new
   hashes, and `output_store: data/masks/<NEW_ID>` with `store_frozen: false`.
3. Run `python -m src.segmentation.verify_generation --generation <NEW_ID>`.
4. Run `python -m src.segmentation.infer_masks --generation <NEW_ID>`.
5. Freeze it only once its own reproducibility gate has passed.

**Never edit an existing contract to make a mismatch go away.** A mismatch means the state is
not the state the contract describes, which is precisely the condition that produced the
historical loss.

---

## 3. `SEG_CURRENT_V1` is NOT historical-equivalent

Both the PVBM feature definitions used downstream and the canonical cohort are unchanged, so it
is tempting to treat the current masks as "the historical experiment with corrected data". That
inference is wrong and is explicitly forbidden here:

| | historical 2026-08-27 | `SEG_CURRENT_V1` |
|---|---|---|
| cohort | 8947 (non-canonical) | 8870 canonical |
| image → mask pairing | wrong for 610 images | strict, verified, 8870/8870 |
| segmentation state | unknown, unrecoverable | pinned by four byte hashes |
| reproducible | no | yes, 8870/8870 |

Three or four things differ at once, and one of them cannot be measured. Any feature difference
between the two tables therefore mixes cohort membership, pairing and an unmeasurable
segmentation drift, and the drift alone moves every image.

**Status language to use:**

| label | applies to |
|---|---|
| `HISTORICAL_CONTAMINATED_NONREPRODUCIBLE` | `data/features/biomarker_features.csv`; historical Branch A and Branch C |
| `CURRENT_MASK_HISTORICAL_DEFINITION_BASELINE` | `data/features/biomarker_features_historical_equivalent_corrected_v1.csv`, sha256 `90ce160727347f1d73f571ba4969638550ae534cb13cf891bf814fce92f7a463` |
| `CURRENT_MASK_GENERATION` | `SEG_CURRENT_V1` and its mask store |
| *corrected measurement layer* | reserved for later clinical-v4-type definitions; not yet used |

The baseline table keeps its bytes and its filename; only its label changed. Machine-readable
sidecar: `artifacts/task4_table_classification.json`.

---

## 4. Preservation

Nothing historical was overwritten, moved or deleted by this work. The historical feature
table, the historical A/C/LOSO artifacts, the historical masks and every Task-1 to Task-4B
forensic artifact remain in place, including the failed Task-4 reconstruction. New outputs went
to new paths only.
