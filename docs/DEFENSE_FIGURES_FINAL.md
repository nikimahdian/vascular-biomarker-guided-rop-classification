# Defense figures (final set)

Publication/slide-quality figures generated **only** from the saved result CSVs of the fast-examiner
and fast-benchmark analyses. No model retrained, no metric recomputed for plotting, no value invented,
no CI altered, no result omitted.

Output directory (local): `results/defense_figures_final/`

| file | content |
|---|---|
| `01_fair_benchmark_auc.{png,svg,pdf}` | horizontal bar chart, macro OvR AUC of the six harmonized methods, zoomed 0.55–0.75 axis explicitly labelled |
| `02_fair_benchmark_delta_auc_forest.{png,svg,pdf}` | paired ΔAUC forest: E−ROPDeepX, E−Legacy B5, E−B (10,000 patient-bootstrap replicates) |
| `03_patient_cv_sensitivity_forest.{png,svg,pdf}` | examiner-driven sensitivity: C−B, E−B, G−E under patient-level 3-fold CV |
| `04_biomarker_redundancy_heatmap.{png,svg,pdf}` | Pearson correlation of the five scalar biomarkers, max VIF ≈ 366, PCA ≥95 % variance = 1 component |
| `05_grouped_auc_balacc_f1.{png,svg,pdf}` | optional grouped AUC / balanced accuracy / macro F1 (Brier and ECE deliberately excluded) |
| `make_defense_figures.py` | generator (matplotlib, 13.33 × 7.5 in, 300 dpi, PNG + SVG + PDF, white background) |
| `FIGURE_NOTES.md` | per figure: source CSV, columns, every plotted value, what it does/does not show, 20-second defense explanation, likely examiner questions and safe answers |
| `verify_values.py`, `final_qc.py` | value verification (145 numeric + 22 integrity checks) and the final QC report |

Key numbers carried by the figures (identical to the saved CSVs):

| method | AUC | Bal. acc | Macro F1 | Brier | ECE |
|---|---|---|---|---|---|
| Previous Lab B5 — 3-class reimplementation | 0.6404 | 0.4807 | 0.4712 | 0.6556 | 0.1904 |
| ROPDeepX — harmonized manual implementation | 0.6246 | 0.4539 | 0.4607 | 0.7854 | 0.2979 |
| Current B | 0.7126 | 0.5685 | 0.5881 | 0.7347 | 0.3416 |
| Current C | 0.7129 | 0.5750 | 0.5952 | 0.7312 | 0.3369 |
| Current E (predefined vessel-fusion model) | 0.7077 | 0.5691 | 0.5878 | 0.7393 | 0.3386 |
| Current G | 0.7094 | 0.5694 | 0.5875 | 0.7422 | 0.3416 |

Paired ΔAUC: E−ROPDeepX +0.0832 [+0.0240, +0.1449]; E−Legacy +0.0673 [+0.0031, +0.1331];
E−B −0.0049 [−0.0112, +0.0015]. Sensitivity: C−B +0.00025 [−0.00354, +0.00391];
E−B −0.00488 [−0.01122, +0.00155]; G−E +0.00167 [−0.00166, +0.00497].

`final_qc.py` reports: `FIGURES_CREATED = 5 required (+1 optional) × PNG/SVG/PDF`,
`VALUE_CHECK = PASS`, `N_COMMON = 1528`, `PATIENTS = 68`, `ZERO_PATIENT_OVERLAP = PASS`,
`UNSUPPORTED_CLAIMS_FOUND = 0`.

Two rounding-level discrepancies in the supplied ground-truth text block were reported rather than
silently resolved (per-fold fold-0 G AUC 0.709476 vs supplied 0.710 — not plotted anywhere; G−E
p = 0.3205 vs supplied 0.321 — the figure prints the saved value). Details in `FIGURE_NOTES.md`.

`results/` is git-ignored in this repository by design; the generator and the notes are versioned here
under `scripts/defense_figures/` and this document carries the reporting reference.
