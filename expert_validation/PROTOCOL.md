# Human Expert Validation Protocol

**Scope.** This protocol validates the *measurement layer* of the vascular-biomarker pipeline on
the target domain: the vessel segmentation, the optic-disc geometry used for scale normalisation,
and every vascular measurement derived from them. It is not a classifier benchmark, it does not
re-select a model, and it makes no selection of any kind on the locked evaluation split.

**Why it is needed.** The project's own diagnostic reports `mask_validity: UNKNOWN` and
`measurement_reliability: UNKNOWN`, and names `MASK_VALIDITY_UNKNOWN` as its primary failure mode.
No expert vessel or disc annotation exists for this project's own images. External validation on
HVDROPDB (vessel Dice 0.754 RetCam / 0.473 Neo; disc Dice 0.9066 / 0.9203) shows the pipeline is
capable on one camera family and degrades on another, but HVDROPDB contains no Plus labels and is
not this domain. A classifier AUC says nothing about whether the underlying measurements are
correct.

---

## 1. Scientific questions

Four questions, analysed separately, sharing one image set and one annotation session.

| ID | Question | Primary endpoint |
|----|----------|------------------|
| **Q-A** | Does the automatic vessel mask reproduce clinically visible vasculature? | Dice against the expert mask; clDice; tiered recall |
| **Q-B** | Is the automatic optic-disc geometry usable as a scale reference? | Disc Dice; centre error in expert disc diameters; diameter ratio |
| **Q-C** | Do automatic vascular *measurements* agree with expert-derived measurements? | ICC(2,1) with 95 % CI and Bland–Altman bias/LoA per primary biomarker |
| **Q-D** | Do the extracted features track clinically assessed severity? | Spearman correlation of each primary biomarker with the expert tortuosity and dilation ordinal scores |

**Q-C is the question the thesis actually needs.** A good Dice with a poor measurement ICC is a
real and reportable outcome, and the mask resolution ceiling (below) predicts it is possible.

---

## 2. What this study explicitly is not

- Not a model or threshold selection procedure. No weights, thresholds, architectures or fusion
  schemes are chosen on the basis of these results.
- Not a use of the locked test as a fresh confirmation set. The locked split was already observed
  by earlier exploratory runs; nothing here is selected on it.
- Not a clinical validation of the artery/vein biomarkers. Section 7 states what is required for
  that and what is reported instead if it is not obtained.
- Not a claim that the pipeline is clinically deployable.

---

## 3. Image selection

### 3.1 Pool

Drawn from **val ∪ test only** (2659 images), because those are the images the segmentation and
disc models were never trained on. A measurement-validation study must not be scored on images the
measurement model memorised.

### 3.2 Grouping rule

**At most one image per `group_id`.** Repeated images of one infant are not independent
observations. For `farabi`, `group_id` is the exam and not the patient, because patient identity is
unavailable there; that is a stated limitation, not a solved problem.

### 3.3 The binding constraint — state this before promising a design

Under one-image-per-group the val ∪ test pool offers at most:

| class | available groups |
|---|---|
| Plus | **31** |
| Pre-Plus | **27** |
| Normal | 81 |

and the Plus groups are distributed as: geometry 960 → 25, geometry 1200 → 4, geometry 1240 → 2,
**geometry 480 → 0**. Two consequences that must not be hidden:

1. **Geometry 480 cannot contribute any Plus image** to an honest expert set.
2. Roughly three quarters of the Plus annotations will come from geometry 960 (`farabi`).

This is a direct consequence of the dataset pathology documented in
`docs/ACQUISITION_GEOMETRY_AND_MEASUREMENT_AUDIT.md`, not a design choice.

Groups can carry more than one label (51 groups in the full dataset do), so allocation proceeds
scarcest-class-first: Pre-Plus, then Plus, then Normal.

### 3.4 Allocation actually realised

`expert_validation/sampling_summary.csv`, produced by `scripts/build_expert_sample.py`,
seed 20260120.

**Pilot cohort — 30 images**

| geometry | Normal | Pre-Plus | Plus | n |
|---|---|---|---|---|
| 480 | 3 | 0 | 0 | 3 |
| 960 | 2 | 4 | 6 | 12 |
| 1080 | 3 | 0 | 0 | 3 |
| 1200 | 2 | 2 | 1 | 5 |
| 1240 | 7 | 0 | 0 | 7 |
| **total** | 17 | 6 | 7 | **30** |

**Final cohort — 90 images**

| geometry | Normal | Pre-Plus | Plus | n |
|---|---|---|---|---|
| 480 | 16 | 0 | 0 | 16 |
| 960 | 6 | 12 | 17 | 35 |
| 1080 | 17 | 0 | 0 | 17 |
| 1200 | 7 | 4 | 3 | 14 |
| 1240 | 6 | 0 | 2 | 8 |
| **total** | 52 | 16 | 22 | **90** |

By source: `farabi` 35, `plus` 41, `farfum_rop` 14. Plus-vs-rest in the final cohort is 22 : 68.

**Pre-specified challenge subset.** 35 of the 90 (39 %) have automatic disc confidence below 0.5.
These are reported as a separate subgroup, never as the headline, because they are exactly the
images where the measurement layer's validity is in question.

Manifest hash: see `expert_validation/MANIFEST_SHA256.txt`. The hash is recorded before any
annotation begins and must not change.

### 3.5 Cohorts must stay disjoint

Pilot images may not appear in the final cohort and vice versa; the selection enforces this through
the group rule and the two cohorts are drawn from disjoint groups.

---

## 4. Experts

**Minimum:** a clinician who screens or treats ROP (pediatric ophthalmologist or retina specialist
with ROP grading experience), fluent in ICROP3 terminology. A trained retinal-image grader may
perform vessel tracing, but the vascular grade must come from a clinician.

**Design: two independent graders plus adjudication for disagreements.**

- Grader A annotates all 120 images.
- Grader B annotates the 30 pilot images and at least 30 of the final 90, chosen to span all five
  geometries and all three classes.
- Disagreements exceeding the pre-specified tolerances (Section 9) go to a third clinician or to a
  recorded consensus session.

Rationale: agreement between ROP experts on Plus disease is modest. Published multi-expert studies
report full agreement on a three-way Normal / Pre-Plus / Plus decision for only a small minority of
images. Treating one annotator as ground truth would make the study's ceiling uninterpretable and
would invite the obvious reviewer objection. Grader B's subset also produces the human ceiling that
Section 9 needs in place of arbitrary thresholds.

---

## 5. Annotation tasks

Each grader receives only de-identified RGB images under blinded study IDs. Nothing else.

### 5.1 Image quality

`gradable` / `borderline` / `ungradable`, with one or more reason codes: blur, low contrast, poor
illumination, artifact, incomplete field, other (free text).

`ungradable` images are excluded from all endpoints and reported only as a count. `borderline`
images stay in, and a sensitivity analysis excluding them is reported.

### 5.2 Vessel mask

Binary mask at the image's native resolution, `0` = background, `255` = retinal vessel.

Include retinal arteries, retinal veins and visible branches. Exclude choroidal structures,
haemorrhage, the optic-disc border, camera artifact, eyelashes and the black border.

**Minimum-visible-width rule.** A vessel is annotated only if the grader can trace its course in
the original image with confidence at reasonable zoom. If uncertain, the grader marks the region as
uncertain rather than drawing a trace they cannot see. Hallucinated centreline is a worse error
than an omission, because topology metrics punish it.

### 5.3 Optic disc

If the disc is visible: full boundary mask, plus centre and diameter. If not visible:
`disc_gradable = False`, and no mask is drawn. Do not invent a disc.

### 5.4 Independent vascular grading

`Normal` / `Pre-Plus` / `Plus` / `Cannot determine`, with a confidence score 1–5. This is recorded
**before** any automatic output is revealed.

### 5.5 Clinical severity scores

Two independent ordinal scores, both 0–4:

- **Arterial tortuosity severity** — 0 none, 1 mild, 2 moderate, 3 marked, 4 severe.
- **Venous dilation severity** — same scale.

These are what make Q-D possible: they let the study ask whether the automatic tortuosity and
caliber features track what a clinician calls tortuosity and dilation.

---

## 6. Two-stage display rule

**First pass:** the grader sees only the RGB image.
**After the annotation is saved:** the automatic mask may be shown for quality assurance, and any
change must be logged as an adjudication event with a reason.

If instead the grader is asked to *correct* the automatic mask, the output must be described in
the thesis as **expert-corrected algorithmic masks**, not as independent expert annotations. It is
a cheaper and weaker design, and the wording must not blur the difference.

---

## 7. Artery/vein validation

The current artery/vein assignment is an unsupervised background-corrected green-intensity split.
It is a heuristic, and correcting the reported width ratio from 1.110 to 0.934 restores the
expected biological direction but does **not** demonstrate correct labelling.

Choose one, and state it in the thesis explicitly:

| Option | Requirement on the grader | What becomes claimable |
|---|---|---|
| A. Full pixel-level A/V masks | expensive; only feasible on a subset (e.g. 30 images) | artery-specific and vein-specific biomarkers may be reported as validated measurements |
| B. Major-vessel branch labels (`A` / `V` / `U` uncertain) on a subset | moderate | A/V biomarkers may be reported as validated **on that subset**, with the heuristic retained elsewhere and labelled as such |
| C. No A/V annotation | none | A/V features are **exploratory only** and must not appear among primary clinical results |

**Recommendation: Option B on the 30 pilot images plus 30 of the final cohort.** It is the cheapest
route to being able to say anything defensible about `a_width_p90`, `v_width_p90` or
`av_width_ratio_p90`. Under Option C those features are removed from every headline claim.

---

## 8. Analysis plan

### 8.1 Q-A — segmentation agreement

Against the expert vessel mask: Dice, IoU, precision, recall, sensitivity, specificity, **clDice**,
and centreline recall stratified into thin / medium / thick by expert-mask local width.

Topology-aware metrics are required because overlap metrics are dominated by the large vessels: a
mask can lose the entire thin tortuous network and still score a respectable Dice. The observed
failure mode on the second camera is a recall collapse (0.338) with high precision, i.e. missing
thin vessels, which Dice alone would understate.

### 8.2 Q-B — disc agreement

Disc Dice, centre error expressed in expert disc diameters, and predicted/expert diameter ratio.
The clinically relevant question is not Dice but whether the scale reference is accurate enough:
a 5 % radius error propagates directly into every disc-normalised measurement.

### 8.3 Q-C — measurement agreement, two separate comparisons

Both comparisons are run, because they answer different questions and conflating them hides the
source of error.

| | Vessel mask | Disc geometry | Question answered |
|---|---|---|---|
| **Measurement 1** | automatic | **expert** | segmentation-induced measurement error |
| **Measurement 2** | automatic | automatic | whole-pipeline measurement error |

Biomarkers measured from the expert mask and from the automatic mask use the **same** extraction
code, so the comparison is of masks, not of implementations.

**Metrics per primary biomarker:**

- **ICC(2,1)** — two-way random effects, absolute agreement, single measurement. Absolute agreement
  rather than consistency, because a systematic offset is a real measurement error and consistency
  ICC would hide it. Single rather than average measurement, because the clinical use is a single
  measurement per image. Reported with a 95 % CI.
- **Bland–Altman** — bias, 95 % limits of agreement, and the plot. Correlation alone is not
  agreement: widths 10/20/30 against 15/25/35 correlate almost perfectly and are biased by +5.

**Pre-specified primary biomarkers** (everything else is secondary, to avoid testing thirty
features and reporting the survivors):

1. vessel density
2. vessel calibre — `width_p90` in disc diameters (and in pixels when the disc is invalid)
3. tortuosity — `tort_p90`

Secondary: fractal dimension, branching, quadrant and ring densities, artery/vein features.

### 8.4 Q-D — clinical interpretation

Spearman correlation between each primary biomarker and the expert arterial-tortuosity and
venous-dilation ordinal scores, with the sign hypothesis stated in advance (higher tortuosity score
↔ higher automatic tortuosity; higher dilation score ↔ higher automatic calibre). Also
Plus-vs-non-Plus AUC per biomarker as a descriptive quantity, not as a claim.

### 8.5 Expert-vs-expert agreement — the human ceiling

On the doubly annotated subset: Dice and clDice between graders for masks; weighted Cohen's kappa
for the three-way grade; Cohen's kappa for Plus-vs-non-Plus; ICC and Bland–Altman for the ordinal
severity scores.

This is what replaces arbitrary acceptance thresholds. The interpretable statement is *"the
automatic measurement differs from the expert by about as much as two experts differ from each
other"*, not *"Dice exceeds 0.80"*.

### 8.6 Uncertainty

**Cluster bootstrap by `group_id`, 5000 replicates**, resampling groups with replacement and taking
all images of each sampled group. Image-level bootstrap is not used. For `farabi` the cluster is the
exam; the limitation that several exams may belong to one infant is stated.

Multiplicity: the three primary biomarkers across the primary endpoint are Holm-corrected. The rest
is descriptive and labelled as such.

---

## 9. Acceptance — reference-based, not invented

The existing config carries `mask_dice_min: 0.80`, `mask_cldice_min: 0.80`, `biomarker_icc_min:
0.75` and itself labels them `thresholds_are: provisional_sensitivity_analysis`. They are
engineering conventions, not medical standards, and this protocol does not promote them.

**What is pre-specified instead:**

1. Every endpoint is reported as a point estimate with a 95 % CI and a Bland–Altman bias.
2. Each is compared against the corresponding **expert-vs-expert** value from Section 8.5.
3. The interpretation rule is fixed in advance:

| Outcome | Interpretation |
|---|---|
| automatic error ≈ expert-vs-expert error | measurement layer supports the biomarker analysis |
| automatic error clearly larger, bias small and correctable | usable after calibration; the calibration must be reported |
| automatic error clearly larger, bias large or non-correctable | the affected biomarker is reported as a measurement limitation and removed from clinical claims |

**Stopping rule.** If automatic-vs-expert Dice is below the expert-vs-expert Dice by more than the
pre-specified margin on the pilot, or if more than 40 % of pilot images are ungradable, stop and
re-scope to a pilot-only report rather than proceeding to the final cohort.

---

## 10. Workflow

**Before.** Scrub EXIF and any burned-in identifiers; verify no patient name, no visible ID, no
laterality text that identifies a patient; rename to `ROP_0001`…`ROP_0120`; split the key
(`blinding_key.csv`) from the delivered pack (`pilot_manifest_blinded.csv`). Ship the two files
separately. Use the institution's approved transfer channel if the images fall under a data-use
agreement.

**During.** Fixed session length (suggest ≤ 90 minutes with a break); a short calibration set of
5–10 images, disjoint from both cohorts, annotated jointly first; a written SOP handed over before
the first annotated image; uncertainty recorded rather than resolved.

**After.** Schema validation; a second reviewer checks a 10 % random sample of masks for format and
obvious protocol violations; disagreements resolved by adjudication with a logged reason; the frozen
analysis script is run once and its output hash recorded.

**Protocol freeze.** Version 1.0 is dated and frozen after the pilot. Anything changed afterwards
is recorded in a changelog, and the pilot images affected by the change are reported separately
rather than folded into the final numbers.

---

## 11. File structure

```
expert_validation/
├── PROTOCOL.md                 this document
├── ANNOTATION_MANUAL.md        the frozen master, written for the full 120-image study
├── ANNOTATION_MANUAL_PILOT.md  operational derivative for the 30-image pilot; the only body
│                               change is the workload count. scripts/make_pilot_manual.py
│                               asserts that diff is exactly one line.
├── manifest_cohorts_internal.csv  study_id, cohort for all 120   <- internal, not delivered
├── pilot_manifest_blinded.csv     the 30 pilot study_ids         <- delivered to the grader
├── final_manifest_blinded.csv     the 90 final study_ids         <- internal until phase 3
├── blinding_key.csv            study_id -> image_path, label, source, split, group_id, flags
│                               PRIVATE. Never delivered, never committed (gitignored).
├── MANIFEST_SHA256.txt         hash of the frozen selection
├── sampling_summary.csv        realised allocation
├── images/                     ROP_0001.png ...
├── grader_A/{vessel_masks,disc_masks,grading.csv}
├── grader_B/{vessel_masks,disc_masks,grading.csv}
├── adjudicated/{vessel_masks,disc_masks,consensus.csv}
└── analysis/                   outputs, one directory per frozen run
```

### `grading.csv`

| column | type | values |
|---|---|---|
| `study_id` | string | `ROP_0001` … |
| `grader_id` | string | `A`, `B` |
| `image_quality` | enum | `gradable`, `borderline`, `ungradable` |
| `quality_reason` | string | `blur`, `low_contrast`, `illumination`, `artifact`, `incomplete_field`, `other` (pipe-separated) |
| `gradable_vessels` | bool | |
| `gradable_disc` | bool | |
| `rop_vascular_grade` | enum | `normal`, `pre_plus`, `plus`, `cannot_determine` |
| `grade_confidence` | int | 1–5 |
| `arterial_tortuosity_score` | int | 0–4 |
| `venous_dilation_score` | int | 0–4 |
| `av_annotation_available` | bool | |
| `notes` | string | free text |
| `annotated_at` | ISO 8601 | |

### Disc geometry CSV

`study_id, grader_id, disc_x, disc_y, disc_radius, disc_area_px, gradable_disc`

### A/V branch CSV (Option B)

`study_id, grader_id, branch_id, label` where `label ∈ {A, V, U}`, plus a branch mask or a
centreline ID map.

---

## 12. Expected tables

**Table 1 — cohort description.** source × geometry × class counts, and image-quality counts.
**Table 2 — Q-A.** Dice, clDice, precision, recall, thin/medium/thick recall, overall and by source
and geometry, with cluster-bootstrap CIs.
**Table 3 — Q-B.** Disc Dice, centre error / expert DD, diameter ratio.
**Table 4 — Q-C.** Per primary biomarker, Measurement 1 and Measurement 2: ICC(2,1), 95 % CI, bias,
95 % LoA, n.
**Table 5 — human ceiling.** Expert A vs expert B: Dice, clDice, weighted κ, ICC on ordinal scores.
**Table 6 — Q-D.** Spearman ρ between each primary biomarker and the two clinical severity scores.
**Table 7 — limitations, per biomarker.** Which claims are supported and which are not.

**Figures.** Bland–Altman plots for the three primary biomarkers; a Dice-vs-ICC scatter showing
that segmentation quality does not imply measurement agreement; a per-geometry forest plot of Dice
and ICC; representative overlays (RGB | expert mask | automatic mask) for agreement and
disagreement cases, including the disc-failure subgroup.

---

## 13. Known limitations, to be stated up front

- **Sample size.** 90 images supports a usable overall estimate and coarse subgroup estimates. It
  cannot support tight per-geometry × per-class estimates: the Plus stratum at geometry 1200 has
  three images and at 1240 has two.
- **Geometry 480 contributes no Plus image.** Under the one-image-per-group rule, and because all
  Plus images at that geometry come from a single group.
- **Plus annotations are concentrated in geometry 960.** A Dice or ICC dominated by one geometry is
  not a domain-general result.
- **Single-centre annotation.** Two graders from, at most, two centres.
- **Exam-level grouping for `farabi`.** No patient identity.
- **External HVDROPDB numbers do not substitute for target-domain numbers.** They show capability,
  not validity here.
- **Mask-resolution ceiling.** The vessel mask is produced at 256×256 and upsampled, so width
  derived from it quantises at roughly 0.041 disc diameters, against a label effect of about
  0.19 → 0.21. If the width ICC is poor while the Dice is good, this is the first explanation to
  test, and the honest response is to report it as a limitation rather than to train a new
  high-resolution model inside this thesis.
- **The locked split is a locked evaluation split, not a fresh unbiased test set.**

---

## 14. Pre-registration block

> **Question.** Do automatic vascular measurements derived from a deep vessel segmentation and a
> deep optic-disc detector agree with expert measurements on target-domain ROP fundus images, well
> enough to support interpretable biomarker claims?
>
> **Primary endpoints.** (i) vessel Dice and clDice against expert masks; (ii) disc centre error in
> expert disc diameters; (iii) ICC(2,1) with 95 % CI for vessel density, calibre (width p90 in disc
> diameters) and tortuosity (tort p90).
>
> **Secondary endpoints.** Bland–Altman bias and LoA per biomarker; tiered vessel recall; disc
> diameter ratio; expert-vs-expert agreement; Spearman correlation of primary biomarkers with
> expert tortuosity and dilation scores; A/V labelling agreement on the subset.
>
> **Analysis.** Cluster bootstrap by `group_id`, 5000 replicates. Holm correction across the three
> primary biomarkers. No threshold, weight, feature set or model is selected on these data.
>
> **Stopping rule.** If pilot Dice is below expert-vs-expert Dice by more than the pre-specified
> margin, or more than 40 % of pilot images are ungradable, the study is reported as a pilot only.

---

## 15. Timeline and risk register

| Phase | Content | Effort | Depends on |
|---|---|---|---|
| 0 | De-identification, pack build, manual, calibration set | 2–3 days | this protocol |
| 1 | Pilot: 30 images × grader A (grader B on a subset) | 1–2 weeks elapsed | Phase 0 |
| 2 | Analysis of pilot, manual revision, protocol freeze v1.0 | 3–5 days | Phase 1 |
| 3 | Final: 90 images, blinded | 2–3 weeks elapsed | Phase 2 |
| 4 | Agreement analysis | 3–5 days | Phase 3 |
| 5 | Clinical interpretation analysis and write-up | 1 week | Phase 4 |

| Risk | P | Impact | Mitigation |
|---|---|---|---|
| Grader availability slips | high | high | start with the 30-image pilot; the pilot alone yields a reportable result |
| Masks returned in the wrong format or resolution | medium | medium | ship a template and one worked example; validate on receipt before the grader continues |
| Two graders disagree more than expected | medium | medium | that is itself a result; adjudicate and report the ceiling |
| Artery/vein labels not obtainable | medium | medium | fall back to Option C and demote A/V features to exploratory |
| Width ICC poor because of the 256-pixel mask ceiling | medium | high | pre-specified as a possible outcome; report as a limitation |
| Grading contaminated by seeing automatic output | low | high | blinded first pass, two-stage display rule, audit trail |

---

## 16. Reproducing the selection

```bash
python scripts/expert_sampling_headroom.py    # what the one-per-group rule permits
python scripts/build_expert_sample.py         # the realised 30 + 90 selection
python scripts/verify_disc_rule.py <features.csv>   # disc-dependent features are NaN without a disc
```
