# Expert validation analysis suite

Run order is not optional. Each step consumes the previous one, and the first step is a gate.

```bash
cd expert_validation/analysis

# 0. verify the toolkit behaves, before any expert data exists
python selftest_analysis.py

# 1. GATE — refuse to continue if the returned annotations are malformed
python validate_annotations.py --key ../blinding_key.csv --root ..
#    exits non-zero and prints every problem; do not continue until it passes

# 2. Q-A  vessel segmentation validity
python segmentation_agreement.py --key ../blinding_key.csv --root .. --out out/seg

# 3. Q-B  optic disc validity
python disc_agreement.py --key ../blinding_key.csv --root .. \
    --disc-pred /path/to/disc_predictions_all.csv --out out/disc

# 4. re-measure with the pipeline's own code, three variants
python extract_expert_biomarkers.py --key ../blinding_key.csv --root .. \
    --disc-pred /path/to/disc_predictions_all.csv --repo-root ../.. --out out/biomarkers

# 5. Q-C  measurement agreement (the question the thesis actually needs)
python biomarker_agreement.py --measurements out/biomarkers/biomarker_measurements.csv \
    --out out/agreement

# 6. human ceiling
python intergrader_agreement.py --key ../blinding_key.csv --root .. --out out/intergrader

# 7. Q-D  feature versus clinician
python clinical_agreement.py --key ../blinding_key.csv --root .. \
    --measurements out/biomarkers/biomarker_measurements.csv --out out/clinical

# 8. assemble the thesis tables
python make_final_report.py --analysis out --out out/report \
    --key ../blinding_key.csv --grading ../grader_A/grading.csv
```

Independent of the expert data, and runnable now:

```bash
# confidence intervals for the cross-source results
python cluster_bootstrap.py --predictions /path/to/results/loo --out out/loso_ci.csv
```

## What each script refuses to do

- `validate_annotations.py` will not let the rest of the suite run on masks whose dimensions differ
  from the photograph, whose values are not 0/255, or whose filenames leak the source or label.
- `segmentation_agreement.py` defines its thin/medium/thick tiers from the **expert** mask, so the
  definition cannot drift with the quality of the mask being evaluated.
- `disc_agreement.py` keeps the automatic-confidence subgroups separate and labels the
  confidence-band table as descriptive only. A new operational floor may be developed on the pilot
  and frozen; it may not be chosen on the final cohort.
- `extract_expert_biomarkers.py` imports `measure()` from `scripts/clinical_features_v3.py` rather
  than reimplementing it, so Measurement 1 and Measurement 2 compare **masks**, not two different
  measurement implementations. The disc-validity rule comes with it: no disc, no disc-relative
  feature.
- `biomarker_agreement.py` reports ICC(2,1) with a group-clustered bootstrap CI **and**
  Bland-Altman. Correlation is not agreement.
- `cluster_bootstrap.py` resamples `group_id`, never images.

## Two demonstrations worth quoting in the thesis

Both are asserted by `selftest_analysis.py`, on synthetic data, so they can be reproduced at any
time:

```
a thin vessel lost entirely:   Dice = 0.909   thin recall = 0.000   thick recall = 1.000
a systematic +5 offset:        ICC  = 0.208   Pearson    = 0.9999
```

The first is why overlap metrics and topology metrics are both reported. The second is why an ICC
with Bland-Altman is reported and a correlation is not.

## Expected inputs

```
expert_validation/
├── blinding_key.csv                        private; never shipped to a grader
├── grader_A/{vessel_masks,disc_masks,grading.csv}
├── grader_B/{vessel_masks,disc_masks,grading.csv}
└── adjudicated/{vessel_masks,disc_masks,consensus.csv}
```

Mask files are `<study_id>.png`, at the photograph's native resolution, values 0 and 255.
