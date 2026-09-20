# Audit of the pre-existing `_expert_pilot30.zip`

**Classification: UNSAFE. Do not send.**

The archive was **not** modified, extracted in place, or deleted. It was read only, and the
findings below are what a recipient would have received.

- file: `_expert_pilot30.zip`, 18,985,848 bytes
- sha256: `e3265ff27a70c9e72f66d74445ac9305b36ef9cc35aa197f92e9edc01ce10b0b`
- 69 zip entries, 65 files, all under a single `expert_pilot30/` folder

## Contents

| path | files | note |
|---|---|---|
| `images/` | 30 | `AUD-###.jpg`, JPEG, 640×480 to 1600×1200 |
| `auto_masks/` | 30 | `AUD-###_auto.png`, the automatic vessel masks |
| root | 5 | `START-HERE.txt`, `INSTRUCTIONS.md`, `PILOT30_README.md`, `PROTOCOL.md`, `pilot30_manifest.csv` |

## Findings, in order of severity

### 1. `pilot30_manifest.csv` unblinds every image completely

Columns: `audit_id, source, label, split, image_path_original, auto_mask_path_original,
pack_image, pack_auto_mask, expert_mask_path, annotator, notes, pilot`.

| column | value counts | what it reveals |
|---|---|---|
| `source` | three sources, 10 images each | the acquisition source of every image |
| `label` | Plus 15, Normal 10, Pre-Plus 5 | the diagnosis of every image |
| `split` | train 23, val 7 | the split assignment of every image |
| `image_path_original` | 30 absolute paths | e.g. `/Users/<user>/data/raw/<source>/<class folder>/<patient-uuid>.<n>.jpg` — source, **class folder name**, and **patient UUID** |

The original path alone carries the source, the diagnosis in the folder name, and a patient
identifier. Every blunting measure elsewhere in the archive is undone by this one file.

### 2. `PILOT30_README.md` prints the composition table

```
source      label
<source A>  0  2 | 1  3 | 2  5
<source B>  0  3 | 1  2 | 2  5
<source C>  0  5 |        2  5
```

The grader is told the source and label breakdown before annotating. The source names and the
per-cell counts are reproduced here from the archive; the names are elided in this report because
this repository is public.

### 3. `PROTOCOL.md` and `INSTRUCTIONS.md` expose internal results and protocol

- "`diag_v3_clinical` failed complementarity and showed **worse** source confounding (macro ≈ 0.99)"
- "Fusion stays **blocked**"
- acceptance targets: "Dice >= 0.80", "feature ICC >= 0.75"
- a mention of the locked test and of the internal script `scripts/expert_mask_audit_eval.py`
- "source-specific failure (<source A> vs <source C> vs <source B>)"

### 4. 30 automatic vessel masks are shipped

`auto_masks/AUD-###_auto.png`. The task design explicitly requires independent annotation from the
RGB image; supplying the automatic mask invites anchoring, and `START-HERE.txt` goes further and
asks the grader to **correct** it ("لازم نیست از صفر بکشید"). Output produced that way must be
described as *expert-corrected algorithmic masks*, which is a materially weaker claim than
independent expert annotation — and is a different design from the frozen protocol.

### 5. EXIF survives on 10 of the 30 images

All ten 1600×1200 images carry an EXIF IFD with:

| tag | value |
|---|---|
| 800 | `CMSI RetCam Media` (device/vendor) |
| 305 | `RetCamV6_20141027_…` (software version) |
| 270 | an XML `MediaFileMetadata` document |
| 769 | `2.199978000219998` (exposure) |

Device make, software version and an embedded XML block are an acquisition fingerprint. The other
20 images carry JFIF/density fields.

### 6. The identifiers and the sample do not match the frozen cohort

Identifiers are `AUD-002 … AUD-087`; the frozen manifest assigns `ROP_0001 … ROP_0120`. The
overlap between the two sets is empty. `PILOT30_README.md` states the selection was drawn from a
90-image pack with seed 42 as "5 Plus + 5 non-Plus per source", which is neither the frozen pilot
composition nor the frozen sampling rule. **The archive is therefore scientifically stale as well
as unsafe** — sending it would run a different study from the one the protocol describes.

### 7. Images are JPEG

The RGB content has already been through lossy compression, so pixel-level fidelity to the source
cannot be guaranteed, and the metadata above travelled with it.

## Verdict

`UNSAFE`, on all of: identity, label and source disclosure; patient UUIDs; device metadata;
automatic masks; internal results; and a sample that does not match the frozen manifest.

It must not be sent to a clinician, and it must not be edited and reused. The replacement build is
`expert_pilot30_v1.0.zip`, produced by `expert_validation/build_pilot_package.py` and gated by
`scripts/validate_expert_pilot_package.py`.
