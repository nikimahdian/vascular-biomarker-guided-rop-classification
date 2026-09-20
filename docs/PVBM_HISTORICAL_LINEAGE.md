# PVBM historical feature lineage — Task 4

```
HISTORICAL_EXTRACTION_SCRIPT:        src/biomarker/extract_pvbm.py
HISTORICAL_EXTRACTION_COMMITS:       e2b9a48 (initial), 0b96808 (grouping metadata only)
FEATURE_CODE_CHANGED_BETWEEN_THEM:   NO
HISTORICAL_PVBM_VERSION:             3.0.1.0
HISTORICAL_ROI_MODE:                 whole
HISTORICAL_TORTUOSITY_IMPLEMENTATION: PVBM
HISTORICAL_FEATURE_COUNT:            23
HISTORICAL_FEATURE_TABLE:            data/features/biomarker_features.csv
LINEAGE_COMPLETE:                    YES
```

---

## A. Repository evidence

| field | value |
|---|---|
| HEAD at time of task | `eabc341` — *Task 3: resolve the 23 LOSO-only images and quantify exact-duplicate leakage* |
| `src/biomarker/extract_pvbm.py` sha256 | `5c1af0fd68f21e4afb40fa56b098d387dc5a24a3acc44a91914247ea5f097f3c` |
| `configs/config.yaml` sha256 | unchanged since `e2b9a48` (single version in history) |
| `requirements-lock.txt` | `pvbm==3.0.1.0`, `numpy==2.0.2`, `scipy==1.13.1`, `scikit-image==0.24.0`, `pandas==2.3.3`, `pillow==11.3.0` |

Version history, newest first:

```
eabc341  Task 3: resolve the 23 LOSO-only images and quantify exact-duplicate leakage
01c7ae5  Task 2: canonical dataset census and 8870 vs 8947 reconciliation
22df490  Task 1: mask pairing root-cause forensics
0101a6b  Data and feature extraction correctness audit
0b96808  Add diagnostics, grouping-metadata fix, and reproducibility docs
e2b9a48  Initial public release: vascular biomarker-guided ROP classification
```

---

## B. The extraction script

`src/biomarker/extract_pvbm.py` is the only script in the repository that writes a PVBM
feature table. It has exactly two versions in git history, `e2b9a48` and `0b96808`.

```
$ git log --oneline --follow -- src/biomarker/extract_pvbm.py
0b96808 Add diagnostics, grouping-metadata fix, and reproducibility docs
e2b9a48 Initial public release: vascular biomarker-guided ROP classification
```

**The diff between the two versions touches no feature computation.** It adds
`GROUP_COLS`, the `attach_grouping()` function (`extract_pvbm.py:49-73`), four new entries in
`META_COLS` (`extract_pvbm.py:34-44`), and a single call `attach_grouping(out_path, cfg)` before
the final count at `extract_pvbm.py:240`. `GEOM_COLUMNS`, `load_binary_mask`,
`density_features` and `geom_features` are byte-identical between the two versions.

**Consequence: the historical feature definitions have been stable for the entire life of the
repository.** There is no "older feature definition" to recover and none to accidentally use.

### B1. Mask loading — `extract_pvbm.py:87-89`

```python
def load_binary_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)
```

Grayscale, threshold `> 127`, `uint8`. No resize, no cropping, no interpolation. The mask is
consumed at its native resolution.

### B2. Density features — `extract_pvbm.py:92-107`

```python
h, w = mask.shape
total = float(mask.sum())
feats = {"vessel_density": total / (h * w), "vessel_pixels": total}
ys = np.array_split(np.arange(h), 3)
xs = np.array_split(np.arange(w), 3)
for i, yy in enumerate(ys):
    for j, xx in enumerate(xs):
        block = mask[yy[0] : yy[-1] + 1, xx[0] : xx[-1] + 1]
        feats[f"density_r{i}c{j}"] = float(block.mean())
```

Whole-frame density **without FOV normalisation**, and a fixed 3×3 grid **without FOV
normalisation**. Both are pure functions of the mask.

### B3. Geometrical VBMs — `extract_pvbm.py:110-154`

```python
seg = mask.astype(float)                     # {0,1}
skeleton = skeletonize(seg > 0).astype(int)
h, w = mask.shape
xc, yc = w // 2, h // 2
radius = max(8, min(h, w) // 8)
...
vbms, _ = geom.compute_geomVBMs(
    blood_vessel=seg_roi, skeleton=skel_roi, xc=xc, yc=yc, radius=radius
)
for name, val in zip(GEOM_COLUMNS, vbms):
    out[name] = float(val)
```

* the skeleton is produced by `skimage.morphology.skeletonize` on the raw mask;
* the optic-disc centre is the **image centre** `(w // 2, h // 2)` and the radius is
  `max(8, min(h, w) // 8)` — a fabricated disc, not a detected one;
* `roi_mode` is `whole`, so `apply_roi` and `DiscSegmenter` are never reached
  (`extract_pvbm.py:123-135` is dead code under the historical configuration);
* the recursion limit is raised to 10000 for the graph walk (`extract_pvbm.py:140-150`).

`GEOM_COLUMNS` (`extract_pvbm.py:75-84`) is
`area, tortuosity_index, median_tortuosity, overall_length, median_branching_angle,
n_startpoints, n_endpoints, n_intersections`, zipped positionally onto the PVBM return value.

**Return-order check.** `PVBM/GeometryAnalysis.py:494` returns

```python
return [area, TI, medTor, ovlen, medianba, startp, endp, interp], (endpoints, ...)
```

which matches `GEOM_COLUMNS` element for element. **No column is mislabelled.**

### B4. Fractal features — `extract_pvbm.py:156-167`

```python
from PVBM.FractalAnalysis import MultifractalVBMs
fractal = MultifractalVBMs(n_rotations=25, optimize=True, min_proba=0.0001, maxproba=0.9999)
d0, d1, d2, sl = fractal.compute_multifractals(seg_roi.astype(float))
out["fractal_d0"], out["fractal_d1"], out["fractal_d2"] = float(d0), float(d1), float(d2)
out["singularity_length"] = float(sl)
```

Wrapped in `try/except` (`extract_pvbm.py:166-167`): if the fractal module fails the keys are
**absent** from the returned dict, so those columns become NaN. This is the only
optional-feature path in the pipeline.

### B5. ROI setting — `configs/config.yaml:67-69`

```yaml
biomarker:
  # ROI for PVBM. 'disc' uses PVBM optic-disc segmentation to build zones; 'whole' uses full image.
  roi: whole
```

`configs/config.yaml` has a single version in git history (`e2b9a48`), and `roi: whole` is
present in that version:

```
$ git log --oneline --follow -- configs/config.yaml
e2b9a48 Initial public release: vascular biomarker-guided ROP classification
$ git log -p --follow -- configs/config.yaml | grep roi
+  roi: whole
```

`extract_pvbm.py:177` takes `--roi` with default `cfg["biomarker"]["roi"]`, so `whole` is the
historical setting unless the CLI overrode it. The four dead-code branches for `disc` cannot
have run without leaving a trace, and none of the historical artifacts records a `disc` run.

### B6. Missing-value policy

`geom_features` initialises `out = {c: np.nan for c in GEOM_COLUMNS}` (`extract_pvbm.py:137`)
and only overwrites on success. NaN is therefore the only missing-value sentinel; there is no
imputation at extraction time. Imputation happens later and per-branch:
`branch_a_tabular.py:133-134` and `branch_c_hybrid.py:175-176` both fill with **training-split
medians**.

### B7. Output assembly and the table's column order

`extract_pvbm.py:218-240` accumulates one dict per manifest row, flushes every 50 rows through
`append_csv_rows` (`src/utils/common.py:338-346`, a plain `DataFrame.to_csv`), renames
`biomarker_features.partial.csv` to `biomarker_features.csv`, and then calls
`attach_grouping`.

**The historical table's column order is not the order that `extract_pvbm` alone would
produce.** `src/data/refresh_feature_splits.py:27-36`:

```python
keep = ["image_path", "label", "split", "source"]
keep.extend(c for c in ["group_id", "patient_id", "exam_id", "identity_level"] if c in splits.columns)
meta = splits[keep].drop_duplicates(subset="image_path")
feats = pd.read_csv(feats_path)
feat_cols = [c for c in feats.columns if c not in set(keep)]
merged = feats[["image_path"] + feat_cols].merge(meta, on="image_path", how="inner")
```

This moves `label/split/source` from position 2-4 to just after the feature block, producing

```
image_path, mask_path,
vessel_density, vessel_pixels, density_r0c0..density_r2c2,
area, tortuosity_index, median_tortuosity, overall_length,
median_branching_angle, n_startpoints, n_endpoints, n_intersections,
fractal_d0, fractal_d1, fractal_d2, singularity_length,
label, split, source, group_id, patient_id, exam_id, identity_level
```

which is exactly the order observed in `data/features/biomarker_features.csv`. **The lineage is
therefore `extract_pvbm` → `refresh_feature_splits`.** `refresh_feature_splits` reorders and
re-attaches metadata; it computes nothing.

---

## C. Frozen historical feature schema

```
historical_feature_count : 23
historical_feature_order : as listed in B7, positions 2..24 of the table
metadata_columns         : image_path, mask_path, label, split, source,
                           group_id, patient_id, exam_id, identity_level   (9)
total columns            : 32
```

| # | feature | type | historical scientific status |
|---|---|---|---|
| 1 | `vessel_density` | project-derived whole-frame density | exploratory |
| 2 | `vessel_pixels` | project-derived whole-frame density | qc_only |
| 3-11 | `density_r0c0` … `density_r2c2` | 3×3 regional density (9) | exploratory |
| 12 | `area` | PVBM library output | exploratory |
| 13 | `tortuosity_index` | PVBM library output | validated_correct |
| 14 | `median_tortuosity` | PVBM library output | validated_correct |
| 15 | `overall_length` | PVBM library output | exploratory |
| 16 | `median_branching_angle` | PVBM library output | exploratory |
| 17 | `n_startpoints` | PVBM library output | confirmed_defect |
| 18 | `n_endpoints` | PVBM library output | confirmed_defect |
| 19 | `n_intersections` | PVBM library output | confirmed_defect |
| 20-22 | `fractal_d0`, `fractal_d1`, `fractal_d2` | fractal feature | exploratory |
| 23 | `singularity_length` | fractal feature | exploratory |

Statuses are read from `configs/feature_contract.csv` rows with `feature_version = pvbm_v1`
(23 rows). **This classification is descriptive only.** No feature is removed, renamed,
reordered or redefined because of its status — including the three marked `confirmed_defect`.

---

## D. Which table historical Branch A and Branch C consumed

Traced from executable code, not from prose.

| branch | path | evidence |
|---|---|---|
| Branch A | `data/features/biomarker_features.csv` | `src/classify/branch_a_tabular.py:115` |
| Branch C | `data/features/biomarker_features.csv` | `src/classify/branch_c_hybrid.py:159` |

```
BRANCH_A_HISTORICAL_FEATURE_TABLE:        data/features/biomarker_features.csv
BRANCH_C_HISTORICAL_FEATURE_TABLE:        data/features/biomarker_features.csv

BRANCH_A_FEATURE_SCHEMA_IDENTICAL_TO_C:   YES

BRANCH_A_HISTORICAL_ROW_N:                8870
BRANCH_C_HISTORICAL_ROW_N:                8870
```

Both branches import one shared schema definition:
`branch_c_hybrid.py:23` is `from src.classify.branch_a_tabular import META_COLS`, so the single
`META_COLS` literal at `branch_a_tabular.py:30-40` governs both. Predictor selection is the same
expression in both files:

* `branch_a_tabular.py:123-132`
* `branch_c_hybrid.py:166-174`

```python
candidate_cols = [c for c in df.columns if c not in META_COLS]
numeric = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
train_numeric = numeric.loc[split == "train"]
feat_cols = [c for c in numeric.columns
             if not train_numeric[c].isna().all()
             and float(train_numeric[c].std(skipna=True) or 0.0) > 0.0]
```

**Both branches consumed the same PVBM rows.** They differ only in that Branch C concatenates
`extract_embeddings(...)` onto the biomarker matrix (`branch_c_hybrid.py:228`).

### D1. A third table exists and must not be confused with these

`data/features/biomarker_features_normalized.csv` is written by
`src/biomarker/quality_gates.py:179-194` with "resolution-safe" columns. It is read by exactly
two places:

* `src/classify/grouped_cv.py:33-34` — the grouped-CV **stability report**, not the headline
  Branch A result;
* `scripts/hybrid_meta_fusion_experiment.py:118` — a later experiment.

**Neither historical Branch A nor historical Branch C reads it.** For the 23 historical features
the historical row count on this table is 0.

---

## E. Tortuosity lineage

```
HISTORICAL_A_C_TORTUOSITY_IMPLEMENTATION:       PVBM
LOCAL_TORTUOSITY_BUG_AFFECTED_HISTORICAL_A_C:   NO
```

`GEOM_COLUMNS[1]` and `GEOM_COLUMNS[2]` are `tortuosity_index` and `median_tortuosity`, filled at
`extract_pvbm.py:151-152` from the tuple returned by
`PVBM.GeometryAnalysis.GeometricalVBMs.compute_geomVBMs`. Inside PVBM the two values are computed
at `PVBM/GeometryAnalysis.py:449` (`TI = arc_list.sum() / chord_list.sum()`) using
`PVBM/helpers/tortuosity.py::compute_tortuosity` over the 8-neighbour distance set
`[1,1,1,1,√2,√2,√2,√2]`.

The locally re-implemented tortuosity lives in `src/biomarker/geometry_core.py`. A repository-wide
search for its consumers returns only:

```
tests/test_feature_invariance.py:14   from src.biomarker.geometry_core import (branch_tortuosity, ...)
scripts/make_feature_contract.py:28   (documentation string)
scripts/make_feature_contract.py:130  (documentation string)
```

**No production extraction path imports `geometry_core`.** `extract_pvbm.py` does not import it.
`branch_a_tabular.py` and `branch_c_hybrid.py` do not import it. The historical Branch A and
Branch C biomarker inputs therefore contain **PVBM tortuosity**, and the raster-orientation defect
in the local re-implementation cannot have reached them.

**Task 4 therefore uses `compute_geomVBMs` and does not substitute `geometry_core.py`.**

---

## F. Fractal lineage

Unchanged and preserved: `MultifractalVBMs(n_rotations=25, optimize=True, min_proba=0.0001,
maxproba=0.9999)` then `compute_multifractals(seg_roi.astype(float))`
(`extract_pvbm.py:158-165`). A scan of every `.py` file under the installed
`PVBM/` package for `np.random`, `random.`, `RandomState`, `default_rng`, `seed(`, `shuffle`
and `choice(` returns **no matches**, so `n_rotations=25` is a deterministic rotation schedule,
not a random sample. Task 4 reproduces the implementation and establishes reproducibility only;
it makes no claim that the fractal features are validated.

---

## G. Environment actually used for the reconstruction

Installed in `/Users/moniaz/niki/.venv` and verified before extraction:

```
numpy         == 2.0.2
scipy         == 1.13.1
scikit-image  == 0.24.0
pandas        == 2.3.3
pvbm          == 3.0.1.0
Pillow        == 11.3.0
```

Every pin matches `requirements-lock.txt` exactly, and `pvbm==3.0.1.0` matches the locked
`pvbm==3.0.1.0`. The reconstruction therefore ran on the same package versions the historical
table was produced with, as far as the repository records them.

## H. Determinism of the historical pipeline

The identical extraction was run three times over the same 12 masks:

| check | result |
|---|---|
| features bitwise identical across runs 1 and 2 | yes, 23 / 23 |
| features bitwise identical across runs 1 and 3 | yes, 23 / 23 |
| NaN positions identical | yes |
| unseeded randomness anywhere in `PVBM/` | none found |

`tortuosity_index` returned NaN on 1 image, `median_tortuosity` on 1, `median_branching_angle`
on 2, and the NaNs landed on the same rows every time. The two `RuntimeWarning`s emitted from
`PVBM/GeometryAnalysis.py:449` and `numpy/_core/fromnumeric.py:3596` are the source of those
NaNs and are themselves deterministic.

**The historical pipeline is reproducible given its inputs.** Any failure to reproduce the
historical table is therefore attributable to the inputs, not to nondeterminism — which is what
Task 4 subsequently found.
