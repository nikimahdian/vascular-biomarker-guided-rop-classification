# Vascular Biomarker–Guided ROP Classification

Can **vascular biomarkers** from infant fundus photos help detect **ROP Plus disease** — and does fusing them with a CNN beat either alone?

This repo answers that on one **leakage-free**, patient/exam-grouped split (8,870 images · 414 groups).

**Author:** Niki Mahdian · K. N. Toosi University of Technology (KNTU)

---

## Results (locked test)

Primary metric: **Plus vs rest AUC**. Thresholds chosen on validation only.

| Branch | Approach | Test AUC |
|--------|----------|---------:|
| **B** | EfficientNet-B5 on RGB fundus | **0.928** |
| **C** | Early fusion `[biomarkers ‖ CNN embedding]` | 0.912 |
| **A** | PVBM biomarkers → XGBoost | 0.800 |

**Takeaway:** the CNN wins. Early fusion does **not** beat Branch B on this split. Biomarkers alone are weaker but still useful as an interpretable baseline. (A legacy ~0.98 hybrid number came from an image-level leaky split — do not cite it.)

<p align="center">
  <img src="docs/figures/roc_phase5.png" width="480" alt="ROC Plus vs rest on locked test">
</p>

<p align="center">
  <img src="docs/figures/comparison_phase5.png" width="560" alt="Phase 5 metrics comparison">
</p>

Split fingerprint (`data/splits/all.csv` SHA-256):

```
0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8
```

Extra tables: [`artifacts/`](artifacts/).

### What the model looks at

Vessel masks feed biomarkers. Grad-CAM on Branch B tends to land on posterior-pole vessels — the same region clinicians use for Plus.

<p align="center">
  <img src="docs/figures/vessel_overlay_plus.png" width="720" alt="Fundus, vessel probability, binary mask">
  <br>
  <em>Fundus → vessel probability → mask</em>
</p>

<p align="center">
  <img src="docs/figures/gradcam_plus.png" width="220" alt="Grad-CAM Plus case">
  &nbsp;
  <img src="docs/figures/gradcam_preplus.png" width="220" alt="Grad-CAM Pre-Plus case">
  <br>
  <em>Grad-CAM examples (Plus · Pre-Plus)</em>
</p>

### Source shift (honest stress test)

Holding out Farabi (LOSO), the best development ladder model (E2) drops to **AUC 0.774**. Domain shift is real.

<p align="center">
  <img src="docs/figures/loso_farabi_roc_e2_e8.png" width="420" alt="LOSO Farabi ROC">
</p>

---

## How it works

Three branches share the same grouped train / val / test CSVs:

```
fundus images
    ├─► Branch B: EfficientNet-B5  → P(Plus)
    ├─► MAnet vessel mask → PVBM (23 feats) → Branch A
    └─► [biomarkers ‖ frozen CNN emb] → Branch C (LightGBM)
```

Labels: `Normal` · `Pre_Plus` · `Plus`. Sources: Plus dataset, FARFUM-RoP, Farabi.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

Point paths in `configs/config.yaml`, then:

```bash
python -m src.data.prepare_split
python -m src.segmentation.infer_masks
python -m src.biomarker.extract_pvbm
python -m src.classify.branch_a_tabular
python -m src.classify.branch_b_cnn
python -m src.classify.branch_c_hybrid
python -m src.compare.compare
```

Next-architecture ladder (test stays locked):

```bash
bash scripts/run_next_architecture_ladder.sh
```

```bash
pytest -q
```

Raw images, masks, features, and weights are **not** in Git (see `.gitignore`).

---

## Repo map

| Path | What |
|------|------|
| `src/` | Split, segmentation, biomarkers, A/B/C, next-arch ladder |
| `scripts/` | Verify + experiment runners |
| `configs/` | Central YAML |
| `data/splits/` | Official CSVs |
| `docs/figures/` | README plots |
| `artifacts/` | Locked metric tables |
| `tests/` | Regression suites |

---

## Cite

If this code or split helps your work, please cite the thesis / this repository, plus:

- PVBM — [arXiv:2208.00392](https://arxiv.org/abs/2208.00392)
- FARFUM-RoP — Sci Data 2024
