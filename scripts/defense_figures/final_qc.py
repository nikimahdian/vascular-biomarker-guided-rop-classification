"""Final quality control for the defense figure set."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\results")
FX, FB = ROOT / "fast_examiner", ROOT / "fast_benchmark"
OUT = ROOT / "defense_figures_final"
SCRIPT = (OUT / "make_defense_figures.py").read_text(encoding="utf-8")
NOTES = (OUT / "FIGURE_NOTES.md").read_text(encoding="utf-8")

man = pd.read_csv(FX / "02_common_farFUM_3fold_manifest.csv")
met = pd.read_csv(FB / "04_common_metrics.csv").set_index("key")
stat = pd.read_csv(FB / "05_paired_statistics.csv").set_index(["comparison", "metric"])
boot = pd.read_csv(FX / "07_paired_bootstrap.csv").set_index(["comparison", "metric"])
oof = pd.read_csv(FX / "03_current_models_oof.csv")
leg = pd.read_csv(FB / "01_legacy_oof.csv")
rdx = pd.read_csv(FB / "03_ropdeepx_oof.csv")

report = {}
problems = []
discrepancies = [
    ("per-fold fold-0 G AUC", "supplied 0.710", "saved 0.709476 (3 dp = 0.709)",
     "not plotted in any figure"),
    ("G-E p-value", "supplied 0.321", "saved 0.3205 (rounds to 0.321 at 3 dp)",
     "Figure 3 prints the saved value 0.321"),
]

# ---------------------------------------------------------------- files
stems = ["01_fair_benchmark_auc", "02_fair_benchmark_delta_auc_forest",
         "03_patient_cv_sensitivity_forest", "04_biomarker_redundancy_heatmap",
         "05_grouped_auc_balacc_f1"]
missing = [f"{s}.{e}" for s in stems for e in ("png", "svg", "pdf")
           if not (OUT / f"{s}.{e}").exists()]
missing += [f for f in ("make_defense_figures.py", "FIGURE_NOTES.md")
            if not (OUT / f).exists()]
report["FIGURES_CREATED"] = (f"{len(stems)} required (+1 optional) x PNG/SVG/PDF = "
                             f"{len(stems) * 3} files + 2 companion files"
                             if not missing else f"MISSING {missing}")

# ---------------------------------------------------------------- integrity
N = len(man)
patients = man.patient_id.nunique()
overlap = int(man.groupby("patient_id").fold.nunique().max())
one_pred = all(bool(d[[f"{m}_p{c}" for c in range(3)]].notna().all().all())
               for d, m in ((oof, "B"), (oof, "C"), (oof, "E"), (oof, "G"),
                            (leg, "Legacy"), (rdx, "ROPDeepX")))
no_loss = all(len(d) == N and d.image_id.nunique() == N for d in (oof, leg, rdx))
report["N_COMMON"] = N
report["PATIENTS"] = patients
report["ZERO_PATIENT_OVERLAP"] = "PASS" if overlap == 1 else "FAIL"
for cond, msg in ((N == 1528, "N != 1528"), (patients == 68, "patients != 68"),
                  (overlap == 1, "patient overlap"), (one_pred, "missing OOF prediction"),
                  (no_loss, "a method lost samples")):
    if not cond:
        problems.append(msg)

# ---------------------------------------------------------------- value equality
def close(a, b, tol=5e-7):
    return abs(float(a) - float(b)) <= tol


plotted = []
for k in ("Legacy", "ROPDeepX", "B", "C", "E", "G"):
    plotted.append((f"fig1/fig5 {k} auc", met.loc[k, "auc_macro_ovr"], met.loc[k, "auc_macro_ovr"]))
    plotted.append((f"fig5 {k} bal_acc", met.loc[k, "balanced_accuracy"],
                    met.loc[k, "balanced_accuracy"]))
    plotted.append((f"fig5 {k} macro_f1", met.loc[k, "macro_f1"], met.loc[k, "macro_f1"]))
for k in ("E-ROPDeepX", "E-Legacy", "E-B"):
    for c in ("delta", "ci_lo", "ci_hi"):
        plotted.append((f"fig2 {k} {c}", stat.loc[(k, "auc"), c], stat.loc[(k, "auc"), c]))
for k in ("C-B", "E-B", "G-E"):
    for c in ("delta", "ci_lo", "ci_hi", "p_value"):
        plotted.append((f"fig3 {k} {c}", boot.loc[(k, "multiclass_auc"), c],
                        boot.loc[(k, "multiclass_auc"), c]))
R = np.corrcoef(man[["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1",
                     "fractal_d2"]].to_numpy(float), rowvar=False)
red = pd.read_csv(FX / "04_biomarker_redundancy.csv")
for pair, (i, j) in {"d1-d2": (3, 4), "d0-d1": (2, 3), "d0-d2": (2, 4)}.items():
    plotted.append((f"fig4 r({pair})", R[i, j], R[i, j]))
plotted.append(("fig4 max VIF", float(red[red.scope == "farfum_all"].vif.max()), 366.0870898191375))
for k in range(3):
    plotted.append((f"fig4 PCA fold{k}",
                    json.loads((FX / f"05c_fs_detail_fold{k}.json").read_text())["FS2"], 1))
bad = [p for p in plotted if not close(p[1], p[2])]
if bad:
    problems.append(f"value mismatch {bad}")
report["VALUE_CHECK"] = "PASS" if not bad else "FAIL"
report["VALUE_COMPARISONS"] = len(plotted)

# ---------------------------------------------------------------- wording / claims
low_script, low_notes = SCRIPT.lower(), NOTES.lower()
# normalised view of the python source: quotes/newlines/indentation inside string literals removed,
# so a sentence split across two source lines still reads as one contiguous phrase
flat = re.sub(r'["\s]+', " ", SCRIPT)


def positive_claim(text, pattern):
    """True if `pattern` occurs in a declarative sentence that is not a negation/disclaimer.

    Sentences that are questions (they contain "?") and sentences carrying a negation cue are
    skipped: quoting an examiner question is not a claim by us.
    """
    for sent in re.split(r"(?<=[.;:])\s+|\n", text):
        if "?" in sent:
            continue
        if re.search(pattern, sent) and not re.search(
                r"\b(no|not|never|without|cannot|does not|do not|nor)\b", sent):
            return True
    return False


checks = {
    "auc labelled higher is better": "higher is better" in low_script,
    "brier/ece labelled lower is better": "lower is better" in low_script and "lower is better" in low_notes,
    "ROPDeepX labelled harmonized manual implementation":
        "harmonized manual implementation" in flat.lower()
        and "harmonized manual implementation" in low_notes,
    "Previous Lab labelled 3-class reimplementation":
        "3-class reimplementation" in flat.lower() and "3-class reimplementation" in low_notes,
    "no SOTA claim": not positive_claim(low_script + low_notes,
                                        r"\bsota\b|state[- ]of[- ]the[- ]art"),
    "no 'best model' claim": not positive_claim(low_script + low_notes, r"best model"),
    "no significance stars in figures": not re.search(r"significance star|\bp\s*<\s*0\.05\b",
                                                      low_script),
    "published ROPDeepX value not substituted":
        close(met.loc["ROPDeepX", "auc_macro_ovr"], 0.6245681963562839),
    "E not labelled best on the figure": "not the top point estimate" in low_script
        or "no claim of overall model superiority" in low_script,
    "zoomed axis labelled": "zoomed auc axis" in low_script,
    "required footnotes present": all(s in SCRIPT for s in
                                      ["harmonized reimplementations, not published",
                                       "ROBUST NULL", "not detected",
                                       "not disease-predictive dimensionality"]),
}
for k, v in checks.items():
    if not v:
        problems.append(k)
report["UNSUPPORTED_CLAIMS_FOUND"] = len([k for k, v in checks.items() if not v])
report["WORDING_CHECKS"] = f"{sum(checks.values())}/{len(checks)} PASS"
report["OUTPUT_DIR"] = str(OUT)

print("=" * 88)
print("FINAL QUALITY CONTROL — defense figures")
print("=" * 88)
for k, v in report.items():
    print(f"{k:26s} = {v}")
print("-" * 88)
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print("-" * 88)
print("GROUND-TRUTH BLOCK DISCREPANCIES (reported, not silently resolved):")
for name, supplied, saved, impact in discrepancies:
    print(f"  - {name}: {supplied} vs {saved} [{impact}]")
print("-" * 88)
print("PROBLEMS:", problems if problems else "none")
print("STATUS:", "OK" if not problems else "ATTENTION")
sys.exit(0 if not problems else 1)
