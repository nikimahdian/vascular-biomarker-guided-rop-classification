# Historical segmentation mask generation — recovery attempt (Task 4B)

```
HISTORICAL_MASK_ARCHIVE_FOUND:                        NO

HISTORICAL_CHECKPOINT_CANDIDATES_FOUND:               11
    (10 recovered from the configured Google Drive ids; 1 in-project,
     byte-identical to one of the 10 -> 10 distinct checkpoints)

HISTORICAL_MASK_GENERATION_RECOVERED:                 NO

RECOVERY_METHOD:                                      NONE

CURRENT_VS_HISTORICAL_DIFFERENCE_CAUSE:               UNRESOLVED

SUBSET_REPRODUCTION_N:                                300
SUBSET_AREA_EXACT_MATCH_N:                            0

FULL_UNAFFECTED_REPRODUCTION_N:                       not run (subset gate failed)
FULL_UNAFFECTED_EXACT_OR_EQUIVALENT_N:                not run (subset gate failed)

PAIRING_BUG_EFFECT_ISOLATABLE:                        NO

TASK4_CAN_BE_REOPENED:                                NO

TASK4B_STATUS:                                        COMPLETE
```

```
HISTORICAL_SEGMENTATION_STATE_IRRECOVERABLE:          YES
```

---

## A. Hard stop respected

`TASK4B` performed forensic recovery only. No classifier was trained, no LOSO was rerun, no
historical AUC was reinterpreted, no clinical-v4 feature was regenerated, the canonical cohort
and the expert samples were not touched, and **no current mask, checkpoint or historical file
was overwritten, moved or deleted**. Every write by this task went to a new private location
(`_private/historical_seg_candidates/`, `_private_audit/`) or to new documentation.

Reproduction scripts added by this task:

| script | purpose |
|---|---|
| `scripts/segmentation_run_timeline.py` | freeze inventory, checkpoint candidate table, reclassification (B, C, E, S) |
| `scripts/segmentation_checkpoint_recovery.py` | architecture probe and Drive candidate recovery (E) |
| `scripts/segmentation_candidate_reproduction.py` | first-pass reproduction test and threshold sweep (J, K) |
| `scripts/segmentation_candidate_scoring.py` | definitive diff-only scoring on the 5,801 differing rows (K, L) |
| `scripts/search_for_historical_masks.sh` | filesystem/archive/backup sweep (F, H) |
| `scripts/phase4_mask_fingerprint_test.py` | dated 2026-08-27 content-change proof (G) |


---

## B. Frozen state before searching

`_private_audit/task4b_freeze_inventory.csv` records `path, size, mtime, sha256` for every
object the task touches. Nothing in it was modified.

| object | size | mtime | sha256 (first 16) |
|---|---|---|---|
| `src/segmentation/infer_masks.py` | 13,577 | 2026-09-20 16:55 | — |
| `src/segmentation/models.py` | — | — | — |
| `configs/config.yaml` | 3,968 | 2026-08-26 23:30 | — |
| `requirements-lock.txt` | 1,427 | 2026-08-26 23:30 | — |
| `weights/best_weight_DeepLabV3+_resize_27` | 127,358,298 | **2026-08-22 01:52** | `c373f53813ee60b8` |
| `data/masks/mask_manifest.csv` | 3,580,990 | **2026-08-30 07:26** | — |
| `data/masks/mask_manifest_v2.csv` | 3,580,380 | 2026-09-20 16:53 | — |
| `data/masks/mask_manifest_canonical_v1.csv` | 1,666,874 | 2026-09-20 17:16 | — |
| `data/masks/mask_manifest_legacy_image_level_20260826.csv` | 1,808,267 | **2026-08-22 03:38** | — |
| `data/features/biomarker_features.csv` | 5,672,895 | **2026-08-27 01:58** | `d3fb94dbe2fc6b9b` |
| `..._historical_equivalent_corrected_v1.csv` | 5,778,666 | 2026-09-20 18:24 | `90ce160727347f1d` |
| `data/manifests/vessel_prob_v1.json` | 6,018,512 | **2026-08-30 07:26** | — |
| `data/masks/*.png` | 8,960 files | see section C | — |

---

## C. Segmentation generation timeline

Reconstructed from mtimes, dated result directories, logs and `environment.freeze.txt`.
**Unknown fields are marked `UNKNOWN`; nothing is inferred.**

| date/time | event | evidence | detail |
|---|---|---|---|
| 2024-05-04 | checkpoint `best_weight_DeepLabV3+_28` produced | Drive file mtime | prior work, not this project |
| 2024-05-06 | `best_weight_MAnet_res34_resize_31`, `best_weight_DeepLabV3+_resize_27`, `best_weight_Unet++_maskresize_29` produced | Drive file mtime | prior work |
| 2024-08-26 | `checkpoint.pth` produced | Drive file mtime | prior work |
| 2026-01-04 | `best_weight_Default_39`, `best_weight_Unet_new_resize_30` produced | Drive file mtime | prior work |
| **2026-08-22 01:52** | `weights/best_weight_DeepLabV3+_resize_27` present, sha `c373f538…` | file mtime + sha | **unchanged ever since** |
| 2026-08-22 03:38 | `mask_manifest_legacy_image_level_20260826.csv` written | file mtime | 8960 rows, image-level |
| 2026-08-25 → 08-29 | `_remote/` project mirror snapshot | file mtimes | contains the pre-fix `infer_masks.py` |
| 2026-08-26 22:19 | `environment.freeze.txt` written | file mtime | **byte-identical to today's `requirements-lock.txt`** |
| 2026-08-27 01:58 | **historical PVBM feature table written** | `biomarker_features.csv` mtime | 8870 × 32 |
| 2026-08-27 02:04 | phase-4 quality gates, mask review, empty-mask list | `results/phase4_*` mtimes | 124 review rows: 120 accept, **1 rerun**, 3 exclude |
| 2026-08-30 07:21 | **mask PNGs rewritten** | `data/masks` dir mtime | 8,960 PNGs |
| 2026-08-30 07:22 | vessel probability maps written | `data/vessel_prob_v1/*.npy` mtimes | 8,870 float16 maps |
| 2026-08-30 07:26 | `mask_manifest.csv` and `vessel_prob_v1.json` written | file mtimes | manifest = 17,740 rows = 2 per image |
| 2026-09-11 17:15 | **repository created** (first commit `e2b9a48`) | `git log` | after all of the above |
| 2026-09-20 | Task 1 mask-pairing repair | `data/masks` dir mtime | glob→regex fix |

**Two facts dominate everything that follows.**

1. **The repository postdates the events.** Its first commit is `2026-09-11 17:15`, i.e. two
   weeks *after* the historical feature table (08-27) and the mask regeneration (08-30). Git
   history therefore contains **no** contemporaneous record of the segmentation run that made
   the historical masks.
2. **A mask regeneration event exists and its date is pinned to 2026-08-30 07:21–07:26**, four
   days after the historical table.

### C1. What the generation-run record does and does not contain

| field | value | source |
|---|---|---|
| script | `src/segmentation/infer_masks.py` | only mask writer in the project |
| git commit of the 08-30 run | **UNKNOWN** — no commit predates 09-11 | `git log` |
| checkpoint path | `weights/best_weight_DeepLabV3+_resize_27` | `configs/config.yaml:41` |
| checkpoint sha256 | `c373f53813ee60b89651a04a98bc5f1d6bc60a5a50f45c4e10f42475650d5374` | measured |
| threshold | 0.20 | `configs/config.yaml:57` |
| input size | `[256, 256]` | `configs/config.yaml:39` |
| normalisation | none — `Resize` then `ToTensor` (scale to [0,1]) | `infer_masks.py:169-171` |
| resize method | `transforms.Resize` (bilinear) for input; `cv2.INTER_NEAREST` for mask output; `cv2.INTER_LINEAR` for the saved probability map | `infer_masks.py:170, 286, 293` |
| postprocessing | threshold → 8-connected component removal (`min_area=50`) → `MORPH_CLOSE` 3×3 | `infer_masks.py:32-44`, `config.yaml:58-59` |
| output directory | `data/masks` | `config.yaml:9` |
| masks generated | 8,870 PNGs (17,740 manifest rows = two runs) | measured |
| **was an existing mask overwritten?** | **YES** — mask filenames are `md5(image_path)[:8]`, so a re-run targets the same filename | `infer_masks.py:48-50` |
| machine | `moniaz` local Mac (files present on it) | ssh session |

---

## D. `infer_masks.py` version audit

Every version available anywhere was diffed.

| source | sha256 (first 16) | size | mtime | note |
|---|---|---|---|---|
| `_remote/src/segmentation/infer_masks.py` | `688EB10001EFD015` | 12,736 | 2026-08-29 20:33 | snapshot closest to the event |
| `_repo_clean/src/segmentation/infer_masks.py` | `0BC56E54CA607FFA` | 13,577 | 2026-09-20 16:55 | after the Task-1 fix |
| `e2b9a48` (first commit) | — | — | 2026-09-11 | pre-fix |
| `0b96808` | — | — | 2026-09-20 | post-fix |

**The diff is exactly one hunk**, and it changes only *mask reuse*, not mask production:

```diff
-    matches = sorted(masks_dir.glob(f"{stem}_*.png"))
+    import re
+    pat = re.compile(rf"^{re.escape(stem)}_[0-9a-f]{{8}}\.png$")
+    matches = sorted(p for p in masks_dir.glob(f"{stem}_*.png") if pat.match(p.name))
```

Tracked behaviour, one by one:

| behaviour | changed between the historical run and Aug 30? | evidence |
|---|---|---|
| checkpoint path | NO | `config.yaml` content identical between `_remote` and current |
| model architecture | NO | `models.py` byte-identical between `_remote` and current |
| image normalization | NO | `infer_masks.py:169-171` unchanged |
| resize | NO | unchanged |
| aspect-ratio handling | NO | `Resize((256,256))` unchanged |
| sigmoid / logits | NO | `torch.sigmoid` at line 284 unchanged |
| threshold | NO | `config.yaml` unchanged; threshold is a CLI default from config |
| connected-component removal | NO | `post_process` unchanged |
| morphological closing | NO | unchanged |
| min-area filtering | NO | unchanged |
| output resize | NO | `cv2.resize(... INTER_NEAREST)` unchanged |
| output binarization | NO | `(a > 127)` in the PVBM loader, unchanged |
| datatype | NO | `cv2.imwrite` PNG, unchanged |
| **filename generation** | NO | `md5(image_path)[:8]` present in both versions |
| **mask reuse rule** | **YES** | the single glob→regex hunk above |

**Conclusion: no mask-producing behaviour changed.** The only code change affects which
existing mask file is *reused*, and it was made by Task 1 on 2026-09-20, i.e. after the event.

---

## E. Checkpoint forensics

`_private_audit/historical_segmentation_checkpoint_candidates.csv`, 11 rows, 10 distinct
sha256 values.

### E1. The surviving segmentation checkpoint

```
weights/best_weight_DeepLabV3+_resize_27
  size   : 127,358,298 bytes
  mtime  : 2026-08-22 01:52:52      <-- BEFORE the historical feature table (08-27)
  sha256 : c373f53813ee60b89651a04a98bc5f1d6bc60a5a50f45c4e10f42475650d5374
  object : OrderedDict, 366 entries, first key encoder.conv1.weight
  strict load: MAnet = MATCH;  DeepLabV3Plus / Unet / UnetPlusPlus / FPN / PAN = NO
```

**It is byte-identical to Google Drive file `best_weight_MAnet_res34_resize_31`**
(sha `c373f53813ee60b8…`, same 127,358,298 bytes). This *proves* the config comment — "the
filename says DeepLabV3+ but the architecture is MAnet" — and it also proves that the file,
despite its misleading name, is the epoch-31 MAnet checkpoint the stale docstring at
`infer_masks.py:10` refers to.

### E2. Ten candidate checkpoints recovered

`configs/config.yaml:45-55` lists ten `pretrained_weight_ids`. `download_weights.py` fetches
them. All ten downloaded successfully into `_private/historical_seg_candidates/`:

| file | size | sha256 (first 16) | arch / encoder |
|---|---|---|---|
| `best_weight_MAnet_res34_resize_31` | 127,358,298 | `c373f53813ee60b8` | MAnet / resnet34 |
| `best_weight_DeepLabV3+_resize_27` | 89,937,913 | `0b278e84733321f7` | DeepLabV3Plus / resnet34 |
| `best_weight_DeepLabV3+_28` | 183,363,914 | `779257cfd4b14e7a` | DeepLabV3Plus / resnet101 |
| `best_weight_DeepLabV3+_resize_39` | 63,710,225 | `c4032e08473dc428` | no strict match |
| `best_weight_MAnet_13` | 666,474,330 | `93ff95935f6d77a5` | MAnet / resnet101 |
| `best_weight_Unet++_maskresize_29` | 64,032,858 | `c589dd87b81b4998` | no strict match |
| `best_weight_Unet++_maskresize_resnet101_25` | 272,637,978 | `1d8f8288ddceb6b9` | UnetPlusPlus / resnet101 |
| `best_weight_Unet_new_resize_30` | 57,428,727 | `1783d0e6a4ba2a14` | no strict match |
| `checkpoint.pth` | 124,266,637 | `bc1cde01ea9c1164` | no strict match |
| `deeplabv3_best (1).pth` | 87,972,906 | `e54219573f83e7ac` | no strict match |

`best_weight_DeepLabV3+_resize_27` from Drive is **a genuinely different file** from the
in-project one (89,937,913 vs 127,358,298 bytes) and really is a DeepLabV3Plus model. This
resolves the naming confusion: the two checkpoints share a name but not their bytes.

### E3. Searched and found nothing

* in-project `weights/` — one segmentation checkpoint, plus the branch-B archive
  `weights/archive_pre_phase5_20260827/` which contains **only** two branch-B `.pth` files;
* `/Users/moniaz/niki_backup_phase1` — a project backup, but **only five `.py` files, 48 KB**; no
  `data/`, no `weights/`;
* `/Users/moniaz/.Trash` — unrelated archives only (`vosk-model-fa-0.42.zip`, a July output zip);
* `~/.cache/uv/archive-v0`, `~/.cache/huggingface` — package and model caches, unrelated;
* Time Machine — `tmutil listbackups` → *"No machine directory found for host"*;
* local APFS snapshots — only the sealed, read-only system snapshot on `disk3s1s1`; no data
  snapshot to mount;
* no `*.zip`/`*.tar`/`*.7z` anywhere under `$HOME` (excluding caches) contains a mask set.

---

## F. Search for historical mask copies

| location searched | result |
|---|---|
| `data/masks/` | 8,960 PNGs total — **one generation only** |
| 8,870 historical `mask_path` values | 8,264 distinct files, **all present** |
| 87,870 canonical `mask_path` values | 8,870 distinct files, all present |
| PNGs referenced by neither manifest | 86 |
| `results/biomarker_diagnostics/expert_audit_pack_v1/auto_masks/` | 90 PNGs, named `AUD-*.png`, mtime 2026-08-30 07:21 — a review copy, not a full set |
| `.../expert_masks/` | empty |
| `data/vessel_prob_v1/` | 8,870 `.npy` probability maps, mtime 2026-08-30 07:22 |
| `data/raw/farfum_rop/archives/`, `results/legacy_invalid/` | no masks |
| per-image mask-file count | **1 file per image stem** — no second generation survives anywhere |

**`HISTORICAL_MASK_ARCHIVE_FOUND = NO`.**

### F1. The probability maps are not a usable substitute

`data/vessel_prob_v1/*.npy` holds the float16 sigmoid output **after** `cv2.INTER_LINEAR`
upsampling to the original image size (`infer_masks.py:293-295`), whereas the mask is
thresholded at 256×256 and then upsampled with `INTER_NEAREST` (`infer_masks.py:285-286`).
Thresholding the saved map therefore cannot reproduce the mask algorithm, and the float16
quantisation adds a second, uncontrolled perturbation. They are useful for *later*
measurement-layer work but they cannot reconstruct the historical binarisation.

---

## G. mtime was not treated as identity

Every claim in this document is anchored on content, not timestamps:

* mask generation identity is tested by **`area` reproduction** (historical `area` equals the
  vessel pixel count, and that is a pure integer function of the mask), not by dates;
* the surviving checkpoint is identified by **sha256 equality** with the Drive file, not by its
  name or its 08-22 mtime;
* the "current masks differ from historical" claim rests on 5,801 differing pixel counts and on
  90.9 % of historical counts being absent from every mask on disk — measured in Task 4 — with
  mtimes used only as corroboration.

### G1. A dated fingerprint that pins the change to after 2026-08-27 02:04

`results/phase4_mask_review_manifest.csv` (mtime **2026-08-27 02:04:06**, six minutes after the
feature table at 01:58:25) records 32 `(image_path, mask_path, vessel_density)` triples.
Recomputing `vessel_density` from the file at each recorded `mask_path` today:

```
files present       : 32 / 32
EXACT density match : 14 / 32
differing           : 18 / 32
```

18 of 32 mask files changed content after 2026-08-27 02:04 while keeping the same path.
`_private_audit/phase4_fingerprint_test.csv` holds all 32 comparisons. This is a **dated** proof
of content mutation and it does not depend on the 08-27 feature table at all.

---

## H. Cloud and version-history evidence

| source | status |
|---|---|
| OneDrive version history | **NOT CHECKED — requires a GUI/account action by the user.** See below |
| Windows File History / Previous Versions | **NOT CHECKED — requires GUI.** See below |
| macOS local APFS snapshots | checked; only the sealed system snapshot, no data snapshot |
| Time Machine | checked; no backup directory configured |
| external drives | none mounted |
| old machine copies | none found under `$HOME` |

**Explicitly not checked, and what the user must do** — I will not claim these were inspected:

1. **OneDrive version history for the project folder.** In Windows Explorer, right-click
   `C:\Users\nikim\OneDrive\Desktop\ROP_FINAL`, choose *Version history*, and look for any restore
   point between 2026-08-26 and 2026-08-31. The relevant question is whether any version contains
   a `data\masks\` directory with PNGs whose mtime predates 2026-08-30. Record the version
   timestamp and, if present, the file count.
2. **Windows *Previous Versions* / File History** on `C:\Users\nikim\OneDrive\Desktop\ROP_FINAL`
   and on any drive that ever held the project.
3. **Whether the analysis Mac ever had Time Machine enabled**, and whether an external disk with
   an old backup exists.
4. **Any Google Drive copy of the masks.** The project already uses Drive for checkpoints; a
   Drive folder holding `data/masks` from before 2026-08-30 would be the single fastest recovery.

Any of these could overturn `HISTORICAL_SEGMENTATION_STATE_IRRECOVERABLE`. Until one is checked
and comes back empty, the classification rests on the local filesystem, the repository, the
project mirror and the recovery of every configured checkpoint — not on an exhaustive search of
all possible backups.

---

## I. Segmentation run metadata from logs and artifacts

| source | what it contains |
|---|---|
| `results/logs/phase5_branch_b.log` (1.4 MB, 2026-08-27 05:48) | Branch B training only; **no mask-inference output** |
| `results/logs/phase5_branch_c.log` (150 KB, 2026-08-27 18:53) | Branch C only |
| `results/phase4_quality_gates.json`, `phase4_empty_masks.csv` | phase-4 gates; empty-mask list is **header only** |
| `results/phase4_mask_review_queue.csv` | 124 rows; decisions **120 accept / 1 rerun / 3 exclude** |
| `results/current_provisional_20260826/environment.freeze.txt` | byte-identical to today's lock file |
| `data/manifests/vessel_prob_v1.json` | per-image records naming `segmentation_checkpoint`, `preprocessing_version`, `probability_map_version` |
| `grep -rl infer_masks logs results` | no file in `logs/` or `results/` logs a mask-inference run |
| shell history on the Mac | no `infer_masks` invocation recorded |

**The August mask-inference run left no log, summary or console record.** Its checkpoint,
threshold and preprocessing are known only because `config.yaml` never changed and the code
never changed.

The phase-4 review explains at most **one** rerun, so it cannot account for 5,801 differing
pixel counts; it is recorded here because it is the only mask-touching activity between the
feature table and the regeneration.

---

## J, K, L. Candidate generation test

### J1. Harness and control

A re-implementation of the exact inference recipe was driven on a deterministic validation
subset. `post_process` is a verbatim copy of `infer_masks.py:32-44`; the model is built by the
project's own `build_model` and loaded with `strict=True`; input transform is
`Compose([Resize((256,256)), ToTensor()])`; output is `cv2.resize(..., INTER_NEAREST)`.

**Control:** with the in-project checkpoint at threshold 0.20, min_area 50, close 3, the
reconstructed mask is **bitwise identical to the current mask file for 2,555 / 2,555** sampled
images. `CONTROL_PASS = True`. The harness is faithful; a failure to reproduce historical
values is therefore a property of the checkpoint, not of the harness.

### J2. The evaluation subset — and a corrected metric

The first pass scored 300 rows drawn from all 8,260 unaffected rows and produced 294/300 for the
in-project checkpoint, which was **misleading**: 2,459 of those rows have `area` equal to the
current value, and any faithful reconstruction of the current masks matches them trivially.

The definitive subset is therefore drawn **exclusively from the 5,801 unaffected rows whose
historical `area` differs from the current one**:

| property | value |
|---|---|
| evaluation universe | 5,801 |
| subset size | 300 (150 largest deltas + 150 smallest non-zero deltas) |
| `abs_delta` min / median / max | 1 / 154 / 2,117 |
| by source | plus 276, farabi 24 |
| by split | train 224, test 40, val 36 |

A score of 1.0 on this subset can only be achieved by a checkpoint that genuinely reproduces the
historical generation.

### K. Results

Threshold sweep per candidate: 15 values from 0.02 to 0.40 (plus 0.05–0.40 for the late arrivals).
`AREA_EXACT_MATCH_RATE` is the count of rows whose vessel pixel count equals the historical
`area` exactly.

| checkpoint | arch / encoder | best threshold | **AREA_EXACT_MATCH** | median abs delta |
|---|---|---|---|---|
| `best_weight_MAnet_res34_resize_31` *(= in-project)* | MAnet / resnet34 | 0.20 | **0 / 300** | **153.5** |
| `best_weight_Unet++_maskresize_resnet101_25` | UnetPlusPlus / resnet101 | 0.05 | 0 / 300 | 4,044.5 |
| `best_weight_Unet++_maskresize_resnet101_25` | UnetPlusPlus / resnet101 | 0.10 | 0 / 300 | 4,609.0 |
| `best_weight_DeepLabV3+_resize_27` *(Drive)* | DeepLabV3Plus / resnet34 | 0.02 | 0 / 300 | 77,245.0 |
| `best_weight_DeepLabV3+_28` | DeepLabV3Plus / resnet101 | 0.02 | 0 / 300 | 111,650.0 |
| `best_weight_MAnet_13` | MAnet / resnet101 | 0.02 | 0 / 300 | 141,501.5 |
| `best_weight_DeepLabV3+_resize_39` | no strict match | — | not scoreable | — |
| `best_weight_Unet++_maskresize_29` | no strict match | — | not scoreable | — |
| `best_weight_Unet_new_resize_30` | no strict match | — | not scoreable | — |
| `checkpoint.pth` | no strict match | — | not scoreable | — |
| `deeplabv3_best (1).pth` | no strict match | — | not scoreable | — |

```
SUBSET_GATE_FINAL = NO
BEST AREA_EXACT_MATCH = 0 / 300
```

**No recovered checkpoint reproduces the historical vessel pixel counts on any row.** The best
configuration anywhere is the in-project MAnet at threshold 0.20 — which is the *current* mask
generation — and even it misses all 300 rows with a median error of 153.5 pixels. Every
alternative checkpoint is worse by one to three orders of magnitude.

The threshold sweep is decisive on its own: for the in-project checkpoint the median absolute
delta traces a clean minimum at 0.20 (0.18 → 1,285; 0.20 → 153.5; 0.22 → 1,008) while never
reaching zero. **0.20 is unambiguously the threshold that produced the current masks, and no
threshold on this checkpoint produces the historical ones.**

`_private_audit/task4b_diffonly_scores.csv`, `task4b_diffonly_scores_pass2.csv`,
`task4b_threshold_sweep.csv`, `task4b_candidate_test.json`.

---

## M, N. Archive materialisation and the full 8,260-row gate — not reached

Section M requires an old mask archive; none exists. Section N is gated on a subset pass; the
subset gate failed, so the full 8,260-row reproduction gate was **not run**, and is reported as
`not run (subset gate failed)` rather than as a pass or a failure.

---

## O. The 610 pairing bug is NOT isolatable

```
PAIRING_BUG_EFFECT_ISOLATABLE = NO
```

Because the historical segmentation generation cannot be reconstructed, the historical side of
the comparison cannot be rebuilt. Any difference measured between the historical feature table
and the Task-4 table mixes the pairing correction with the unrecovered mask-generation drift,
and the drift alone moves every image (median −4 pixels on the 8,260 "unaffected" rows,
90.9 % of historical counts unmatched by any mask on disk).

Per the brief, the earlier Task-4 statement that the pairing bug had a *small* feature-level
impact is **withdrawn as a claim about the pairing bug**. Task 4's measured deltas remain valid
as a description of the difference between two tables; they are not an attribution.

---

## P. Was the checkpoint overwritten, like the masks?

Explicit question: did a checkpoint path stay constant while its bytes changed?

**No evidence of that for the segmentation checkpoint.**

| question | answer | evidence |
|---|---|---|
| Is there more than one segmentation checkpoint in the project? | No — exactly one | `weights/` listing |
| Did its bytes change? | No — its sha256 equals the Drive original | `c373f538…` both |
| Its mtime vs the historical table | **08-22 < 08-27** — older than the table | file mtimes |
| Was it renamed at some point? | Yes, in effect: the Drive name is `best_weight_MAnet_res34_resize_31` | sha256 equality |
| Is a prior byte-version recoverable? | Not needed — the bytes never changed | sha256 equality with Drive |

So the *mask* files were overwritten in place, because their names are a deterministic function
of the image path; the *checkpoint* was not, because its name is not. Both facts are consistent
with the observed state, and together they mean the model state cannot be the source of the
discrepancy — unless the historical masks came from a checkpoint that no longer exists anywhere,
which the ten-candidate recovery was designed to test and which it failed to find.

---

## Q. Root cause of the current-versus-historical mask difference

```
CURRENT_VS_HISTORICAL_DIFFERENCE_CAUSE = UNRESOLVED
```

Categories excluded, each with evidence:

| category | verdict | evidence |
|---|---|---|
| `THRESHOLD_CHANGED` | **EXCLUDED** | `config.yaml` content unchanged since `_remote`; a 15-point sweep per checkpoint never reaches zero, and 0.20 is the unique best threshold for the current masks |
| `CHECKPOINT_CHANGED` | **EXCLUDED for the in-project checkpoint** | sha256 `c373f538…` is byte-identical to Drive `best_weight_MAnet_res34_resize_31`; mtime 08-22 predates the table |
| `MODEL_CODE_CHANGED` | **EXCLUDED** | `models.py` byte-identical between `_remote` and current; `infer_masks.py` diff is one reuse-glob hunk |
| `POSTPROCESSING_CHANGED` | **EXCLUDED** | `post_process` (threshold → CC removal → MORPH_CLOSE) unchanged; the control reproduces current masks bitwise |
| `PREPROCESSING_CHANGED` | **EXCLUDED from the code record** | `Resize(256,256)` + `ToTensor` unchanged; no normalisation anywhere in either version |
| `INPUT_IMAGE_CHANGED` | **EXCLUDED** | same RGB paths, unchanged; and Task 4's determinism test showed PVBM reproduces bitwise on identical mask input |
| `MULTIPLE_FACTORS` | **not established** | no factor was identified, so naming several would be an invention |

**What the evidence does establish:** the masks at the historical paths had *different content*
on 2026-08-27 than they do now, the change happened after 2026-08-27 02:04, and it is **not**
reproducible by any of the ten recoverable candidate checkpoints at any threshold under the
project's own inference code. The most likely remaining explanation — stated as a hypothesis,
not a finding — is that the 08-30 regeneration used a model state (for example a
project-trained segmentation checkpoint, since `config.yaml` carries a full `seg_train` section)
that was later deleted and is not among the ten Drive artifacts. **`UNRESOLVED` is the honest
classification**, and the search record above is what makes it an endpoint rather than a gap.

---

## R. Interpretation rule, applied

Three generations are kept strictly apart, as required:

* **`HISTORICAL MASK GENERATION`** — the state that produced the masks behind
  `biomarker_features.csv` on 2026-08-27. **Not recovered; no artifact reproduces it.**
* **`CURRENT MASK GENERATION`** — checkpoint `c373f538…`, MAnet/resnet34, threshold 0.20,
  masks of 2026-08-30. Reproduced bitwise by the control.
* **`CORRECTED / VALIDATED MEASUREMENT GENERATION`** — reserved for later clinical-v4-type
  definitions. Not used.

The current generation being reproducible, deterministic and internally consistent does **not**
make it historical-equivalent. Task 4B makes no claim that either generation is scientifically
better.

---

## S. Endpoint B — irrecoverable

```
HISTORICAL_SEGMENTATION_STATE_IRRECOVERABLE
```

### S1. Exactly what was searched

Repository history and every file version in it; the `_remote/` project mirror (files dated
2026-08-25 … 08-29); the whole local filesystem under `$HOME` for archives, backups,
legacy/old/pre directories, mask-named directories and any directory holding more than 100 PNGs;
`data/`, `results/`, `logs/`, `weights/`, `snapshots/`; `~/.Trash`; `~/.cache`; the project
backup `/Users/moniaz/niki_backup_phase1`; Time Machine and APFS snapshot state; all four mask
manifests; the phase-4 review artifacts; the vessel probability-map store and its manifest;
shell history; and **all ten** checkpoints named in `configs/config.yaml`, downloaded and hashed.

### S2. Permanently reclassified

* **historical Branch A** — non-reproducible contaminated historical result.
* **historical Branch C** — non-reproducible contaminated historical result.

### S3. The Task-4 table is reclassified

```
PREVIOUS CLASSIFICATION : historical-equivalent corrected feature table
NEW CLASSIFICATION      : CURRENT_MASK_HISTORICAL_DEFINITION_BASELINE
```

Justification: its PVBM feature definitions are historical and unchanged, but its segmentation
measurement generation is the **current** one. It cannot be called *historical-equivalent*.
Machine-readable sidecar: `artifacts/task4_table_classification.json`
(sha256 `90ce160727347f1d73f571ba4969638550ae534cb13cf891bf814fce92f7a463`, unchanged).

**The failed Task-4 reconstruction is preserved, not erased.** The file keeps its name, its
contents and its hash; only its label changes. `docs/HISTORICAL_PVBM_RECONSTRUCTION.md` carries
the superseding note.

### S4. What the reclassified table is still good for

It is a deterministic, internally valid measurement of the canonical 8,870 cohort under a known
and reproducible segmentation generation, with historical PVBM definitions. That makes it a
sound *baseline* for later work. It is not, and must never be presented as, a reconstruction of
the historical experiment.

---

## T. Task 4 was not reopened

`TASK4_CAN_BE_REOPENED = NO`. Section T requires a recovered historical generation; none was
found. Branch A/C historical-equivalent reconstruction therefore remains blocked, and no
classifier was trained.

---

## V. Completion gate

**ENDPOINT B — IRRECOVERABLE** is reached: a documented search of historical masks, checkpoints,
configs, code versions, logs and accessible local backup/version-history locations found no
state capable of reproducing the historical table.

The one honest qualification is section H: **OneDrive version history, Windows Previous
Versions and any Google Drive copy of the masks were not inspected because they need a
user-side GUI or account action.** Those three remain the only realistic routes back to
Endpoint A, and the exact steps to check them are written out in section H. If any of them
yields a pre-2026-08-30 mask set, Task 4B should be reopened with that archive as the candidate
and the subset gate re-run before anything else happens.

`TASK4B_STATUS = COMPLETE`
