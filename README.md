# Niki ROP — Vascular Biomarker–Guided Plus Disease Classification

> **Giving this project to someone new?** Start here → **[`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md)**  
> Why current hybrid fails and what to build next → **[`NEXT_ARCHITECTURE.md`](NEXT_ARCHITECTURE.md)**  
> What is actually coded vs run → **[`NEXT_ARCHITECTURE_IMPLEMENTATION.md`](NEXT_ARCHITECTURE_IMPLEMENTATION.md)**  
> Measured next-arch numbers → **[`NEXT_ARCHITECTURE_RESULTS.md`](NEXT_ARCHITECTURE_RESULTS.md)**  
> **Full post-prompt report (all work + all results) → [`NEXT_ARCHITECTURE_FULL_REPORT.md`](NEXT_ARCHITECTURE_FULL_REPORT.md)**

**Thesis:** Niki Mahdian, M.Sc., K. N. Toosi University of Technology (KNTU)
**Remote compute:** `moniaz-studio:/Users/moniaz/niki` (macOS Apple Silicon / MPS)  
**Local mirror / docs:** this workspace  
**Document status:** canonical project snapshot as of **2026-08-31**  
**Canonical split hash (`all.csv` SHA-256):**  
`0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8`

Older audit narrative: `README_FINAL_AUDIT_FIX_PHASES.md`.  
Priority if numbers conflict: artifacts in `results/` + `split_report.json` → `PROJECT_HANDOFF.md` → this README.

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Pipeline architecture](#2-pipeline-architecture)
3. [Dataset](#3-dataset)
4. [Identity, grouping, exclusions](#4-identity-grouping-exclusions)
5. [Vessel segmentation](#5-vessel-segmentation)
6. [Biomarker features (23 + release-safe 17)](#6-biomarker-features-23--release-safe-17)
7. [Branch A / B / C (current architecture)](#7-branch-a--b--c-current-architecture)
8. [Eval protocol](#8-eval-protocol)
9. [Work completed by phase](#9-work-completed-by-phase)
10. [Canonical Phase 5 results](#10-canonical-phase-5-results)
11. [Hybrid deep-dive (exploratory, post Phase 5)](#11-hybrid-deep-dive-exploratory-post-phase-5)
12. [Known limitations](#12-known-limitations)
13. [Code & artifact map](#13-code--artifact-map)
14. [How to run](#14-how-to-run)
15. [What remains](#15-what-remains)

---

## 1. What this project is

Hybrid ML pipeline for detecting **Plus disease** in retinopathy of prematurity (ROP) from infant fundus photographs.

Three branches share **one grouped train/val/test split**:

| Branch | Input | Model | Role |
|---|---|---|---|
| **A** | 23 PVBM vascular biomarkers (from vessel masks) | Tabular: XGBoost / LightGBM / RF / LogReg / MLP | Interpretable biomarker baseline |
| **B** | Raw RGB fundus | EfficientNet-B5 (`timm`), fine-tuned | Deep image baseline |
| **C** | Early fusion: `[23 biomarkers ‖ 2048 CNN emb]` | LightGBM / XGBoost on concatenated vector | Hybrid (thesis fusion design) |

**Labels (3-class):** `0 = Normal` · `1 = Pre_Plus` · `2 = Plus`  
**Primary metric:** binary **Plus vs rest** (Plus-OvR) AUC, sensitivity, specificity, F1  
Threshold tuned on **validation only** (Youden J), then applied once on test.

**Research question:** Can interpretable vascular biomarkers compete with a CNN, and does early fusion beat both?

**Canonical Phase 5 answer on the clean split:**  
**B wins** (test AUC **0.928**). Production C early-fusion (**0.912**) does **not** beat B. Biomarkers alone (**0.800**) trail both.

---

## 2. Pipeline architecture

```
Fundus images (Plus + FARFUM-RoP + Farabi)
        │
        ▼
prepare_split.py
  identity parse → quality/dupe exclusions → grouped split
  → data/splits/{all,train,val,test}.csv   (8,870 images)
        │
        ▼
infer_masks.py
  MAnet + ResNet34 (pretrained; file name says DeepLabV3+ but arch is MAnet)
  thr=0.20, min_area=50
  → data/masks/*.png + mask_manifest.csv
        │
        ├──────────────────────────────┐
        ▼                              ▼
extract_pvbm.py                 branch_b_cnn.py
PVBM density + geom + fractal   EfficientNet-B5 fine-tune (50 ep)
→ biomarker_features.csv (23)   → weights/branch_b_efficientnet_b5.pth
        │                              │
        ▼                              │
branch_a_tabular.py                    │
5 models, select by val F1 (Plus-OvR)  │
        │                              ▼
        │                     freeze backbone → emb[2048]
        │                              │
        └──────────────┬───────────────┘
                       ▼
            branch_c_hybrid.py
            X = concat(bio[23], emb[2048]) → 2071-d
            StandardScaler(train) → LightGBM/XGBoost multiclass
            ablations: fusion / biomarkers_only / embedding_only
                       ▼
   compare · rigorous_eval · stats · Grad-CAM · SHAP · LOSO (Phase 6)
```

Everything is driven by `configs/config.yaml` (`seed: 42`).

---

## 3. Dataset

### Sources

| Source | Images (final) | Notes |
|---|---:|---|
| **plus** (`Plus_dataset_main`) | 5,931 | Dominant source; folder keywords for labels |
| **farfum_rop** | 1,529 | Public; real `patient_id` from spreadsheet |
| **farabi** | 1,410 | Private RetCam; exam-level UUID grouping |
| **Total** | **8,870** | After Phase 2/4 exclusions (was 8,960 indexed) |

### Class balance (all 8,870)

| Class | n | share |
|---|---:|---:|
| Normal (0) | 6,459 | 72.8% |
| Pre_Plus (1) | 931 | 10.5% |
| Plus (2) | 1,480 | 16.7% |

### Official grouped split

| Split | Images | Groups | Normal | Pre_Plus | Plus | Farabi | FARFUM | Plus src |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 6,211 | 290 | 4,496 | 651 | 1,064 | 987 | 1,070 | 4,154 |
| val | 1,328 | 62 | 981 | 140 | 207 | 211 | 229 | 888 |
| test | 1,331 | 62 | 982 | 140 | 209 | 212 | 230 | 889 |
| **all** | **8,870** | **414** | 6,459 | 931 | 1,480 | 1,410 | 1,529 | 5,931 |

**Integrity (from `data/splits/split_report.json`):**

- status: `OFFICIAL_PHASE2_GROUPED_SPLIT`
- unique paths = 8,870
- **group overlap train∩val / train∩test / val∩test = 0**
- image-set overlap across splits = 0

---

## 4. Identity, grouping, exclusions

### Identity rules

| Source | Group key | Level |
|---|---|---|
| Plus | leading filename integer patient ID | patient |
| FARFUM-RoP | spreadsheet patient ID | patient |
| Farabi | RetCam Exam UUID (before final image suffix) | **exam** (true patient ID unavailable) |

### Exclusions applied before split

| Reason | Count |
|---|---:|
| Duplicate-identity / exact-hash conflict rows | 54 |
| Quality exclusions (`data/metadata/quality_exclusions.csv`) | 36 |
| Triggering exact-hash groups | 19 |
| Example dropped Plus groups | `plus::040`, `plus::041` |

Phase 4 mask review: **124** stratified/flagged cases; decisions `accept` / `exclude` / `rerun`; reviewer tag `agent-auto`. Quality exclusions include ungradable fundus and review excludes.

---

## 5. Vessel segmentation

| Item | Value |
|---|---|
| Architecture | **MAnet + ResNet34** (checkpoint filename historically says DeepLabV3+ — do not rename/change arch) |
| Weights | `weights/best_weight_DeepLabV3+_resize_27` |
| Input size | 256×256 |
| Threshold | 0.20 |
| Min component area | 50 |
| ROI for biomarkers | whole image |

**Clinical scope:** automated gates pass; **no** target-domain expert Dice/IoU. See `phase4_clinical_scope.md` / remote `results/phase4_clinical_scope.md`.

---

## 6. Biomarker features (23 + release-safe 17)

Extracted by PVBM from vessel masks → `data/features/biomarker_features.csv`.

### Raw 23 features (used by production Branch A and Branch C)

**Density (11)**

1. `vessel_density`  
2. `vessel_pixels`  
3–11. `density_r0c0` … `density_r2c2` (3×3 regional grid)

**Geometry (8)**

12. `area`  
13. `tortuosity_index`  
14. `median_tortuosity`  
15. `overall_length`  
16. `median_branching_angle`  
17. `n_startpoints`  
18. `n_endpoints`  
19. `n_intersections`

**Fractal (4)**

20. `fractal_d0`  
21. `fractal_d1`  
22. `fractal_d2`  
23. `singularity_length`

### Geometry repair history

- Pre-repair: ~85% of rows had **all** geometry NaN (PVBM recursion limit).  
- Patch: raise recursion limit to 10,000 in `geom_features()`.  
- After repair: all-geometry-missing **0%**; residual any-geometry-NaN ~0.84% (train-median imputed).

### Phase 4 derived tables

| File | Role |
|---|---|
| `biomarker_features.csv` | Raw 23 (production A/C) |
| `biomarker_features_normalized.csv` | Resolution-normalized helpers + features |
| `biomarker_features_phase4.csv` | **Release-safe ~17** — drops raw pixel area/length/counts that encode source |

Release-safe drop list (resolution / count confounds):  
`vessel_pixels`, `area`, `overall_length`, `n_startpoints`, `n_endpoints`, `n_intersections`  
(+ `singularity_length` treated as resolution-dependent in hybrid probes).

Source-confounding gate (Phase 4): raw macro-OvR AUC for predicting source ~0.995 → release-safe ~0.887 (reduction ≥0.05 required; **pass**, residual warning remains).

---

## 7. Branch A / B / C (current architecture)

### Branch A — biomarkers only

1. Load `biomarker_features.csv` + split labels.  
2. Drop `META_COLS`; keep numeric cols with non-zero train std → **23 feats**.  
3. Impute NaN with **train median**; `StandardScaler` fit on **train only**.  
4. Train candidates: lightgbm, xgboost, random_forest, logreg, mlp.  
5. Select by **val F1 (Plus-OvR)**; threshold by **Youden on val**.  
6. Report test once + group-bootstrap CI + per-source metrics.

**Canonical winner:** `xgboost` · test Plus AUC **0.800**

### Branch B — CNN

1. EfficientNet-B5, ImageNet init, class-weighted CE, AdamW, ReduceLROnPlateau.  
2. 50 epochs; best checkpoint by **val AUC**.  
3. Softmax 3-class → `P(Plus)`; Youden threshold on val.  
4. Same test protocol.

**Canonical winner:** `efficientnet_b5` · test Plus AUC **0.928**  
Checkpoint: `weights/branch_b_efficientnet_b5.pth`

### Branch C — early fusion hybrid (production)

1. Freeze Branch B backbone → **2048-d** embedding per image.  
2. Concatenate with 23 biomarkers → **2071-d** vector.  
3. Same scaler + tabular models as A (xgboost / lightgbm).  
4. Multiclass 0/1/2 → `P(Plus)` + val Youden threshold.  
5. Ablations in same run: `fusion` / `biomarkers_only` / `embedding_only`.

**Canonical winner:** `lightgbm` fusion · test Plus AUC **0.912**

This is **early feature fusion**, not late score fusion and not joint end-to-end training.

```
fundus ──► frozen EfficientNet-B5 ──► emb[2048]
mask/PVBM ──► bio[23] ──────────────────┤
                                        ▼
                              concat → 2071-d
                                        ▼
                         StandardScaler (train)
                                        ▼
                         LightGBM / XGBoost (3-class)
                                        ▼
                              P(Plus) + threshold
```

---

## 8. Eval protocol

| Setting | Value |
|---|---|
| Shared split | Grouped; zero group/image leakage across splits |
| Primary task | Plus vs rest (from 3-class probs) |
| Model select (A/C) | `val_f1_plus` |
| Threshold | Youden on validation only |
| Bootstrap | Group-level (`group_id`), n≈1000 in branch reports |
| Seed | 42 |
| Config hash | `77a6c9230fd8b6f7230e1085d3415f46ece862d3e160ace983a1be83b97f0a60` |

Do **not** compare current numbers to pre-Phase-5 / leaky-split runs (legacy C AUC ~0.98 was inflated).

---

## 9. Work completed by phase

| Phase | Status | What shipped |
|---|---|---|
| **0** | Done | Checkpoint / env hashes, backup discipline |
| **1** | Done | Identity schema (Plus patient / FARFUM patient / Farabi exam) |
| **2** | Done | Official grouped split; dupe + quality exclusions → **8,870** |
| **3** | Done | Code fixes: train-only impute/scale, threshold, DeLong, grouped CV hooks, Grad-CAM/SHAP Plus targeting, LOSO resume; regression tests (12 passed) |
| **4** | Done (scoped) | Quality gates `AUTOMATED_GATES_PASS`; 124 mask reviews; release-safe feature table; clinical Dice/IoU + generation-time provenance **scoped as unavailable** |
| **5** | Done | Fresh A/B/C on `split_sha256=0d4c3b3a…`; reality check script 48/48 before C |
| **5+ hybrid probes** | Exploratory | Late-fusion probe, hybrid_v2 binary head, meta-fusion, multiclass linear head — **not** replacing canonical C unless thesis redefines deliverable |
| **6** | Partial | Next-architecture **E2 LOSO** on the official 8,870 split is done (Farabi holdout AUC **0.774**). Phase-6 A/B/C LOSO on the same split is a separate older track and is **not** the next-arch table. |
| **7–8** | Pending | Final stats package, thesis tables, delivery bundle |

Phase 4 authorization note: Phase 5 rerun was authorized under limitations in `phase4_clinical_scope.md`.

---

## 10. Canonical Phase 5 results

All on the **same** 8,870-image split (`0d4c3b3a…`). Metric = test **Plus-OvR AUC** unless noted.

### Head-to-head

| Branch | Best model | Test AUC | Sens | Spec | F1 | Thr (val) |
|---|---|---:|---:|---:|---:|---:|
| **A** | xgboost | **0.800** | 0.646 | 0.762 | 0.442 | 0.123 |
| **B** | efficientnet_b5 | **0.928** | 0.861 | 0.803 | 0.590 | 0.058 |
| **C fusion** | lightgbm | **0.912** | 0.837 | 0.785 | 0.560 | ~2e-6 |

### Branch C ablations (same protocol)

| Ablation | Best model | Test AUC |
|---|---|---:|
| fusion (bio‖emb) | lightgbm | 0.912 |
| embedding_only | lightgbm | **0.916** |
| biomarkers_only | xgboost | 0.808 |

**Takeaway:** on this split, adding raw biomarkers to embeddings **hurts** vs embedding-only tree; neither beats full Branch B head.

### Per-source test AUC (Plus-OvR)

| Branch | Farabi | FARFUM | Plus |
|---|---:|---:|---:|
| A | 0.802 | 0.932 | 0.679 |
| B | 0.875 | 0.964 | 0.913 |
| C | 0.860 | 0.977 | 0.908 |

Plus-source remains hardest for biomarkers; CNN carries Plus-domain signal.

### Group bootstrap CI (examples)

| Branch | AUC point | 95% CI (group bootstrap) |
|---|---:|---|
| A | 0.800 | [0.668, 0.950] |
| C fusion | 0.912 | [0.851, 0.983] |

---

## 11. Hybrid deep-dive (exploratory, post Phase 5)

These are **extra experiments** after canonical C. Useful for thesis discussion; **do not** silently replace Phase 5 C without an explicit protocol change. Test set was already used for several probes — further tuning on the same test is discouraged.

### Why production C loses to B

1. Trees on **2048 dense embeddings + 23 sparse biomarkers** underuse the embedding geometry that the B linear head was trained for.  
2. Production C still reads **raw** 23-feature CSV, not Phase-4 release-safe profile.  
3. Ablation: embedding_only (0.916) > fusion (0.912) → biomarkers add noise in this setup.  
4. B’s softmax head is a better Plus score than a fresh tree on frozen emb alone under multiclass LightGBM.

### Late-fusion probe (`results/branch_c_late_fusion_probe.json`)

Weighted blend of A and B probabilities (weights fit on val):

- Best ~ **3% A + 97% B** → test AUC **0.9288** (Δ vs B ≈ **+0.0008**) — not meaningful.  
- Logistic stack of scores does not beat B.

### Hybrid v2 — binary regularized logistic on embeddings (`results/hybrid_v2/`)

- Locked by grouped train-OOF → val; test once.  
- Winner: **`emb_logit_c0.001`** (no biomarkers).  
- Test AUC **0.940** · sens 0.880 · spec 0.819 · F1 0.617  
- Paired **group** bootstrap vs B: ΔAUC **+0.012**, CI95% **[+0.0016, +0.0244]**, p≈**0.023**  
- **Not true hybrid** — binary embedding head only. Nested-encoder limitation documented in artifact JSON.

### Meta-fusion (`results/hybrid_meta_fusion/`)

- Candidates: B score ± safe biomarkers, grouped OOF on val.  
- Locked: **`B_only_logit`** — biomarkers gave no gain (ΔAUC vs B = 0).

### Multiclass linear head (`results/hybrid_v2_multiclass/`)

- Locked: embedding-only `C=0.0001`.  
- Test Plus AUC **0.927** — does **not** beat B (0.928).

### Thesis stance (recommended)

- **Canonical multiclass system:** Branch **B**.  
- **Canonical hybrid design (C):** early concat; report honestly that it **underperforms** B on the clean split.  
- **Exploratory screening head:** C-v2 binary emb logit (0.940) — label as exploratory / not nested encoder.

---

## 12. Known limitations

1. **Farabi grouping is exam-level**, not proven patient-level.  
2. **No clinical Dice/IoU** for vessel masks on this cohort.  
3. **Provenance:** segmentation/feature hashes largely retrospective (`generation_time_binding=UNAVAILABLE_RETROSPECTIVE_HASHES_ONLY`).  
4. **Source confounding** reduced but residual warning on release-safe biomarkers.  
5. **Production C** still uses raw 23 features (not phase4 release-safe).  
6. **Legacy inflated AUCs** (leaky image-level split) are not comparable.  
7. **Next-arch LOSO (E2)** is done; Farabi holdout **0.774** is the honest worst source. That is not a Phase-5 replacement and not a “beats B 0.928” claim.  
8. Hybrid v2 / meta / late probes **touched the test set** for scientific exploration — freeze further test peeking.

---

## 13. Code & artifact map

### Source (`src/`)

| Path | Role |
|---|---|
| `data/prepare_split.py` | Identity + grouped split |
| `segmentation/infer_masks.py` | MAnet mask inference |
| `biomarker/extract_pvbm.py` | 23-feature PVBM extract |
| `biomarker/repair_geom_features.py` | Geometry NaN repair |
| `biomarker/quality_gates.py` | Phase 4 gates + release-safe CSV |
| `classify/branch_a_tabular.py` | Branch A |
| `classify/branch_b_cnn.py` | Branch B |
| `classify/branch_c_hybrid.py` | Branch C early fusion |
| `utils/branch_eval.py` | Shared fit → val thr → test harness |
| `utils/common.py` | Metrics, threshold, bootstrap, config |
| `compare/*` | Compare, LOSO, stats, calibration, operating points |
| `interpret/gradcam.py` | Grad-CAM |
| `classify/next_architecture/` | Post-prompt ladder E0–E5, E8, LOSO, vessel audit (writes only `results/next_architecture/`) |

### Key data / results

| Path | Content |
|---|---|
| `data/splits/*.csv` | Official split |
| `data/splits/split_report.json` | Counts, hashes, overlap checks |
| `data/features/biomarker_features*.csv` | Raw / normalized / phase4 |
| `results/branch_{a,b,c}_results.json` | Canonical metrics |
| `results/branch_{a,b,c}_*_preds.csv` | Predictions |
| `results/hybrid_v2/` | Exploratory binary emb head |
| `results/hybrid_meta_fusion/` | Score±bio meta probe |
| `results/hybrid_v2_multiclass/` | Multiclass linear head probe |
| `results/branch_c_late_fusion_probe.json` | A+B late blend |
| `results/phase4_quality_gates.json` | Gate status + provenance |
| `weights/branch_b_efficientnet_b5.pth` | B checkpoint |
| `results/next_architecture/` | Post-prompt E0–E4, E2 LOSO, vessel audit, SUMMARY (studio) |
| `results/next_architecture/SUMMARY.json` | Machine summary of all next-arch `results.json` |
| `data/vessel_prob_v1/` | Soft vessel maps (float16 `.npy`, 8870) |

### Scripts (selected)

| Script | Role |
|---|---|
| `scripts/verify_phase5_ab_reality.py` | A/B reality checks before C |
| `scripts/late_fusion_probe.py` | A+B weight/stack probe |
| `scripts/hybrid_v2_experiment.py` | Binary emb / safe-bio logistic |
| `scripts/hybrid_meta_fusion_experiment.py` | Meta on B score |
| `scripts/hybrid_v2_multiclass_experiment.py` | Multiclass linear head |
| `scripts/run_next_architecture_ladder.sh` | Sequential E0–E5 / followup / E8; test locked |

---

## 14. How to run

On `moniaz-studio`:

```bash
cd /Users/moniaz/niki
source .venv/bin/activate

# Full orchestrated pipeline (see script for stage flags)
./run_pipeline.sh

# Or module-by-module (typical order)
python -m src.data.prepare_split
python -m src.segmentation.infer_masks
python -m src.biomarker.extract_pvbm
python -m src.classify.branch_a_tabular
python -m src.classify.branch_b_cnn
python -m src.classify.branch_c_hybrid
python -m src.compare.compare
```

Config: `configs/config.yaml`.  
Tests: `pytest tests/` (Phase 3 regression suite).

---

## 15. What remains

1. **Thesis writing** from `results/next_architecture/report_bundle/` + Phase 5 locked numbers.  
2. Nested grouped CV only if a paper claim must beat B **test 0.928**.  
3. Do **not** unlock canonical test. Do **not** `--force-e5`. Do **not** train E5X. Do **not** extra E8 seeds.

---

## Quick verdict

| Question | Answer |
|---|---|
| Best canonical model? | **Branch B** (EfficientNet-B5), test AUC **0.928** |
| Does production hybrid C beat B? | **No** (0.912) |
| Do biomarkers help fusion here? | **No** — embedding_only ablation higher; meta locked B-only |
| Clean patient/exam split? | **Yes** for current cohort (0 group overlap); Farabi = exam-level |
| Thesis-final? | **Compute ladder done** — write from `report_bundle/`; canonical B **0.928** unchanged |

---

*Maintainer note: keep this README aligned with `results/branch_*_results.json` and `data/splits/split_report.json`. Update numbers only from those artifacts.*
