# Fair FARFUM patient-level 3-fold benchmark (Part 2)

Classification: `POST_HOC_HARMONIZED_BENCHMARK_SINGLE_SOURCE`
Scope: one common protocol on **FARFUM-RoP only** — 1,528 images / 68 patients, patient-level 3-fold
manifest from the fast-examiner analysis (sha256 `cfa4ae82…`), zero patient overlap. Every method is
scored by one script on the identical held-out rows (N = 1,528, 0 exclusions). No canonical artifact
was modified; current B/C/E/G were reused, not retrained.

## Main table — macro OvR ROC-AUC on identical patients

| Method | Architecture | AUC | Bal. acc | Macro F1 | Brier | ECE |
|---|---|---|---|---|---|---|
| Previous Lab B5 — 3-class reimplementation | EfficientNet-B5 @224, no augmentation | 0.6404 | 0.4807 | 0.4712 | **0.6556** | **0.1904** |
| ROPDeepX 2026 — harmonized implementation | ResNet50 + EfficientNet-B4, 1024-d soft-attention fusion | 0.6246 | 0.4539 | 0.4607 | 0.7854 | 0.2979 |
| Current B | EfficientNet-B5 @384 embedding → XGBoost | 0.7126 | 0.5685 | 0.5881 | 0.7347 | 0.3416 |
| Current C | B + 5 scalar biomarkers | 0.7129 | 0.5750 | 0.5952 | 0.7312 | 0.3369 |
| **Current E (primary vessel fusion)** | B + spatial vessel embedding | 0.7077 | 0.5691 | 0.5878 | 0.7393 | 0.3386 |
| Current G | E + 5 scalar biomarkers | 0.7094 | 0.5694 | 0.5875 | 0.7422 | 0.3416 |

## Paired comparisons (patient-level bootstrap, 10,000 replicates, seed 42)

| Comparison | ΔAUC | 95% CI | Δ macro F1 | 95% CI | Δ Brier | 95% CI |
|---|---|---|---|---|---|---|
| E − ROPDeepX | **+0.0832** | [+0.0240, +0.1449] | **+0.1271** | [+0.0604, +0.1982] | −0.0461 | [−0.1794, +0.0825] |
| E − Previous Lab B5 | **+0.0673** | [+0.0031, +0.1331] | **+0.1166** | [+0.0416, +0.1991] | +0.0837 | [−0.0504, +0.2108] |
| E − Current B | −0.0049 | [−0.0112, +0.0015] | −0.0002 | [−0.0089, +0.0084] | +0.0046 | [−0.0110, +0.0202] |

Allowed statements: E achieved higher AUC and macro F1 than *the harmonized reimplementation* of
ROPDeepX and than *the 3-class reimplementation* of the previous-lab architecture; the vessel-versus-
RGB increment is not distinguishable in this single-source regime. Not a state-of-the-art claim; the
historical binary ~0.98 and the published ROPDeepX headline are context only.

## Previous-lab audit (from `plus.zip` code)

Binary Plus vs No Plus; image-level random 70/15/15 (`train_test_split`, seed 42, file as the unit);
timm `efficientnet_b5`, 224×224, ImageNet normalization, **no augmentation**; `Dropout(0.5)` +
`Linear`; AdamW lr 1e-5, wd 1e-4, `ReduceLROnPlateau`; ≤100 epochs, early stop patience 10;
checkpoint on best validation macro F1. `LEGACY_GROUP_OVERLAP = YES (by construction)`, exact count
UNKNOWN (file lists absent). "No Plus" = **Normal only** (inferred: the Plus-source lineage contains no
Pre-Plus images).

## ROPDeepX protocol

Publisher PDF not retrievable from this environment (identity-provider redirect) → architecture
verified from indexed full text (ResNet-50 + EfficientNet-B4 → 1024-d projections → concatenated →
soft-attention fusion), training recipe unverified. No official code/weights found in 30 minutes →
`ROPDEEPX_IMPLEMENTATION = MANUAL_REIMPLEMENTATION_FROM_PAPER` using the repository's earlier
paper-derived design. Declared deviation: the first harmonized run used `OneCycleLR` and early-stopped
inside the LR warm-up (pooled AUC 0.5595, selected epochs 2/6/1); the reported run uses a
plateau-decayed constant schedule (pooled AUC 0.6246). Both variants are preserved.

## Artifacts

Produced on `root@185.213.165.199` (RTX 4090) under `~/niki_rop_task6_isolated/results/fast_benchmark/`:
`00_protocol.md`, `01_legacy_oof.csv`, `02_ropdeepx_protocol.md`, `03_ropdeepx_oof.csv`,
`04_common_metrics.csv` (+ `04b` confusion matrices, `04c` intersection, `04d` probabilities,
`04e` schedule sensitivity), `05_paired_statistics.csv`, `06_legacy_audit.md`,
`07_literature_context.csv`, `08_final_comparison.md`, `09_defense_slide.md`, plus
`10_benchmark_auc_bar.png` and `11_paired_delta_auc_forest.png`. Scripts: `p00_port.py`,
`p01_train.py`, `p02_metrics.py`, `p03_paired.py`, `p04_figures.py`, `p05_schedule_sensitivity.py`
(versioned here under `scripts/fast_benchmark/`). `results/` is git-ignored by design.
