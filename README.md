# Vascular Biomarker–Guided ROP Classification

Can **vascular biomarkers** extracted from infant fundus photos help detect **ROP Plus disease** — and does fusing them with a deep CNN beat either alone?

This project runs a fair three-way comparison on one **leakage-free** split: no patient or exam appears in more than one of train / val / test.

**Author:** Niki Mahdian · K. N. Toosi University of Technology (KNTU)

---

## Why this exists

Plus disease is judged largely from **dilation and tortuosity** of posterior-pole vessels. That clinical cue suggests two complementary signals:

1. **Explicit biomarkers** — segment vessels, measure density / tortuosity / branching / fractal features (PVBM).
2. **Learned appearance** — let a CNN read the raw fundus.

The open question is whether putting those signals together (early fusion) is better than the CNN alone. On a clean split, the honest answer here is **no** — but biomarkers still matter as an interpretable baseline, and the pipeline stays reproducible.

---

## Dataset & split

| | |
|--|--:|
| Images after exclusions | **8,870** |
| Groups (patient or exam) | **414** |
| Sources | Plus dataset · FARFUM-RoP · Farabi |
| Labels | `Normal` · `Pre_Plus` · `Plus` |
| Train / val / test | 6,211 / 1,328 / 1,331 |

Grouping: Plus & FARFUM by **patient**; Farabi by **exam** (true patient ID unavailable). Exact-hash and quality exclusions happen **before** the split. Group overlap across splits is zero.

Split fingerprint (`data/splits/all.csv` SHA-256):

```
0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8
```

---

## Results (locked test)

Primary metric: **Plus vs rest (Plus-OvR) AUC**. Model selection and decision threshold use **validation only**; the test set is scored once.

| Branch | Approach | Test AUC | Sens. | Spec. | F1 |
|--------|----------|---------:|------:|------:|----:|
| **B** | EfficientNet-B5 on RGB fundus | **0.928** | 0.861 | 0.803 | 0.590 |
| **C** | Early fusion `[23 biomarkers ‖ 2048-d CNN emb]` → LightGBM | 0.912 | 0.837 | 0.785 | 0.560 |
| **A** | 23 PVBM biomarkers → XGBoost | 0.800 | 0.646 | 0.762 | 0.442 |

**Takeaway**

- **Branch B is the canonical winner** on this split.
- Production hybrid **C does not beat B**. An ablation in the same run shows **embedding-only (0.916) > fusion (0.912)** — concatenating raw biomarkers onto the embedding did not help.
- Branch A trails but remains the transparent, feature-level story.
- Older hybrid numbers around **~0.98** came from an **image-level leaky** split. They are invalid — do not cite them.

<p align="center">
  <img src="docs/figures/roc_phase5.png" width="480" alt="ROC Plus vs rest on locked test">
</p>

<p align="center">
  <img src="docs/figures/comparison_phase5.png" width="560" alt="Phase 5 metrics comparison">
</p>

Summary CSVs: [`artifacts/`](artifacts/).

### What the model looks at

Vessel masks drive Branch A (and the biomarker half of C). Grad-CAM on Branch B usually lights up **posterior-pole vessels** — the same region clinicians weight for Plus.

<p align="center">
  <img src="docs/figures/vessel_overlay_plus.png" width="720" alt="Fundus, vessel probability, binary mask">
  <br>
  <em>Fundus → vessel probability → binary mask</em>
</p>

<p align="center">
  <img src="docs/figures/gradcam_plus.png" width="220" alt="Grad-CAM Plus case">
  &nbsp;
  <img src="docs/figures/gradcam_preplus.png" width="220" alt="Grad-CAM Pre-Plus case">
  <br>
  <em>Grad-CAM examples (Plus · Pre-Plus)</em>
</p>

### Source shift (honest stress test)

A leave-one-source-out holdout of **Farabi** is the hard domain. The best development-ladder model (E2) falls to **AUC 0.774**. In-domain test scores are not the whole story.

<p align="center">
  <img src="docs/figures/loso_farabi_roc_e2_e8.png" width="420" alt="LOSO Farabi ROC">
</p>

---

## How it works

All three branches share the same grouped CSVs and eval protocol.

```
fundus images
    │
    ├─► Branch B: EfficientNet-B5 (fine-tuned)     → P(Plus)
    │
    ├─► MAnet vessel mask → PVBM (23 features)
    │         └─► Branch A: tabular models         → P(Plus)
    │
    └─► freeze B backbone → 2048-d embedding
              concat with biomarkers → 2071-d
              └─► Branch C: LightGBM / XGBoost     → P(Plus)
```

| Piece | Detail |
|-------|--------|
| Segmentation | MAnet + ResNet34, threshold `0.20` (checkpoint filename may say DeepLabV3+ — architecture is MAnet) |
| Biomarkers | Density, geometry, fractal (23 raw features; NaNs imputed with **train** median) |
| Branch B | EfficientNet-B5, class-weighted CE, best checkpoint by val AUC |
| Branch C | Early feature fusion — not late score fusion, not joint end-to-end training |
| Eval | Plus-OvR from 3-class probabilities; Youden threshold on val; group-level bootstrap where reported |

Config hub: `configs/config.yaml` (`seed: 42`).

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

Edit paths in `configs/config.yaml`, then run in order:

```bash
python -m src.data.prepare_split
python -m src.segmentation.infer_masks
python -m src.biomarker.extract_pvbm
# extract_pvbm now writes the grouping metadata itself (group_id, patient_id, exam_id,
# identity_level). For a feature table produced before that change, re-attach it here — without
# these columns the Branch A/C evaluation silently falls back to IMAGE-level bootstrap
# uncertainty instead of group-level:
python -m src.data.refresh_feature_splits
python -m src.classify.branch_a_tabular
python -m src.classify.branch_b_cnn
python -m src.classify.branch_c_hybrid
python -m src.compare.compare
```

Optional next-architecture ladder (development / LOSO; **test stays locked**):

```bash
bash scripts/run_next_architecture_ladder.sh
```

```bash
pytest -q
```

Embedding cache identity, row order, and its one hard rule (never cross-validate over train rows):
`docs/EMBEDDING_CACHE_CONTRACT.md`.

Raw images, masks, features, full `results/`, and weights are **not** in Git — regenerate on your machine or compute host (see `.gitignore`).

---

## Repo map

| Path | What |
|------|------|
| `src/` | Split, segmentation, biomarkers, branches A/B/C, next-arch ladder |
| `scripts/` | Verify helpers and experiment runners |
| `configs/` | Central YAML |
| `data/splits/` | Official train / val / test CSVs |
| `docs/figures/` | Plots used in this README |
| `artifacts/` | Locked metric tables |
| `tests/` | Regression suites |
