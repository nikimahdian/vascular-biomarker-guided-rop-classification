# Annotation Manual — version 1.0 (pilot)

**Read this once before you start. Keep it open while you work. It is deliberately short.**

You are annotating **120 retinopathy-of-prematurity fundus photographs** for a study that is
checking whether an automatic measurement pipeline measures the retinal vessels the way a
clinician does. You are not training a model and you are not being scored. Where you are unsure,
**recording the uncertainty is more useful than guessing.**

Work in two sessions maximum per batch, with a break. Do not annotate when tired; a fatigued
mask is worse than a missing one.

---

## What you receive

- `ROP_0001.png` … a folder of de-identified fundus photographs.
- Empty folders `vessel_masks/` and `disc_masks/` for your output.
- `grading_sheet.csv` — one row per image, to be filled in.
- `av_labels.csv` — only for the images listed in it (see Section 7).

You will **not** receive: the diagnosis, the camera or dataset, the train/validation/test
assignment, any automatic mask, any model prediction, or any heat map. This is intentional and is
part of the study design. If you are ever shown one of these before you have saved your
annotation, stop and tell the study coordinator.

---

## Rules that apply to every image

1. Work at the image's **native resolution**. You may zoom and pan freely.
2. You may adjust brightness and contrast **to see better**. You may not crop, rotate, resize or
   otherwise change the image geometry.
3. Vessel masks and disc masks are always the **same width × height as the photograph**.
4. Mask format: PNG, values `0` = background, `255` = structure. No JPEG, no screenshots, no
   resized copies.
5. Save your work for one image completely before opening the next.
6. If something is unclear, write it in the `notes` column. Those notes are analysed.

---

## Per image, in this exact order

### Step 1 — Look at the image alone

Open the photograph. Do not open anything else.

### Step 2 — Image quality → `image_quality`

| value | meaning |
|---|---|
| `gradable` | vessels can be judged and traced |
| `borderline` | traceable but degraded; still annotate |
| `ungradable` | cannot be judged reliably |

If `borderline` or `ungradable`, set `quality_reason` to one or more of
`blur`, `low_contrast`, `illumination`, `artifact`, `incomplete_field`, `other` (separate with `|`).

An `ungradable` image gets **no masks**. Still fill in the rest of the row with
`cannot_determine` where asked.

### Step 3 — Vessel mask → `vessel_masks/ROP_XXXX.png`

Trace the retinal vasculature as a filled mask. You are delineating the **vessel lumen**, not the
centreline and not the reflection.

**Include**
- retinal arteries
- retinal veins
- visible branches, down to the finest vessel you can follow with confidence

**Exclude**
- the choroidal background
- haemorrhage, exudate, scar, laser marks
- the optic-disc border itself
- camera artifact, eyelashes, dust, the black border, reflections

**Minimum visible width rule.** Annotate a vessel only if you can follow its course in the original
image with confidence at reasonable zoom. If you cannot, **leave it out** and note it.

> A missing thin vessel is a small, honest error. A traced vessel that does not exist is a large
> error, because the topology measurements that matter most in this study are exactly the ones a
> hallucinated trace corrupts.

Where two vessels touch or cross, keep them in the mask as one blob; do not attempt to separate
them by drawing a gap.

### Step 4 — Optic disc → `disc_masks/ROP_XXXX.png`

Outline the **full optic-disc boundary** as a filled mask, and enter the centre and the radius in
the sheet.

**If the disc is not visible, or is cut off by the field edge, or you cannot define its boundary:**

- set `gradable_disc = FALSE`
- leave the disc mask empty (do not create a blank file, just skip it)
- leave `disc_x`, `disc_y`, `disc_radius` empty

**Do not estimate a disc you cannot see.** The disc is used as the ruler for every size
measurement in this study, so an invented ruler is worse than a missing one.

### Step 5 — Independent clinical grading

Record these **before** you see anything automatic.

`rop_vascular_grade` ∈ `normal`, `pre_plus`, `plus`, `cannot_determine`

Base this on the standard ICROP criteria you use in practice — venous dilation and arterial
tortuosity in the posterior pole, assessed across quadrants.

`grade_confidence` 1–5, where 1 = a guess and 5 = certain.

### Step 6 — Two severity scores

Both 0–4, judged independently of the category you just chose.

| score | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `arterial_tortuosity_score` | none | mild | moderate | marked | severe |
| `venous_dilation_score` | none | mild | moderate | marked | severe |

Score the posterior-pole vessels, not the periphery.

### Step 7 — Artery / vein labels (subset only)

For the images listed in `av_labels.csv`, set `av_annotation_available = TRUE` and label each
**major vessel branch** leaving the disc:

- `A` = artery
- `V` = vein
- `U` = uncertain

Label at whole-branch level, not per pixel. One label per branch, applied along the branch.
Return the branch outlines as an ID map or a list of branch masks plus `av_labels.csv`.

If you are not confident about a branch, `U` is the correct answer, not a guess between `A` and `V`.

---

## Worked examples

See `examples/` in this package. The images are from a **different public dataset**, not from
this study.

| file | what it shows |
|---|---|
| `example_vessel_correct.png` | the level of detail expected: main arcades, second-order branches, and thin vessels that are genuinely traceable |
| `example_vessel_common_error.png` | the three errors to avoid: tracing a reflection, filling the disc, and following a vessel into a haemorrhage |
| `example_disc_correct.png` | a complete disc boundary, centre marked |
| `example_disc_not_gradable.png` | a disc that is cut off by the field edge — the correct action is `gradable_disc = FALSE` |
| `example_uncertain.png` | a peripheral vessel that fades below visibility — the correct action is to stop the trace, note it, and not guess |

---

## Before you submit

- Every gradable image has a vessel mask, and its width × height matches the photograph.
- Every gradable-disc image has a disc mask and centre/radius values.
- Every row of `grading_sheet.csv` has `image_quality`, `gradable_vessels`, `gradable_disc`,
  `rop_vascular_grade`, `grade_confidence`, `arterial_tortuosity_score`, `venous_dilation_score`.
- No `ungradable` image has a mask.
- No file names other than `ROP_XXXX` appear anywhere in your output.
- Masks are PNG with values 0 and 255 only.

---

## Questions to raise rather than decide alone

Write these in `notes` and tell the coordinator. They are expected; the pilot exists to find them.

- Where exactly does the vessel end at the disc margin?
- Should a vessel that fades gradually be traced to where it disappears, or to where you can no
  longer be confident?
- How should a vessel be handled where it passes under a haemorrhage?
- Borderline images: is the current definition of `borderline` usable in practice?
- Is the 0–4 severity scale usable, or are there too few / too many levels?
