"""Verification of every value that the defense figures will plot.

Source of truth: the saved result CSVs under results/fast_examiner and results/fast_benchmark.
Ground truth list: the value block supplied with the figure request.
Prints PASS/FAIL per item and exits non-zero on any mismatch.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\results")
FX, FB = ROOT / "fast_examiner", ROOT / "fast_benchmark"
TOL = 5e-5
TOL3 = 5e-4
rows = []


def chk(name, got, want, tol=TOL):
    ok = abs(float(got) - float(want)) <= tol
    rows.append((name, f"{float(got):.6f}", f"{float(want):.6f}", "PASS" if ok else "FAIL"))
    return ok


# ---------------------------------------------------------------- integrity
man = pd.read_csv(FX / "02_common_farFUM_3fold_manifest.csv")
oof = pd.read_csv(FX / "03_current_models_oof.csv")
leg = pd.read_csv(FB / "01_legacy_oof.csv")
rdx = pd.read_csv(FB / "03_ropdeepx_oof.csv")
cur = pd.read_csv(FB / "02b_current_models_oof.csv")
metrics_fx = pd.read_csv(FX / "06_model_metrics.csv").set_index("model")
metrics_fb = pd.read_csv(FB / "04_common_metrics.csv").set_index("key")
boot_fx = pd.read_csv(FX / "07_paired_bootstrap.csv").set_index(["comparison", "metric"])
boot_fb = pd.read_csv(FB / "05_paired_statistics.csv").set_index(["comparison", "metric"])
red = pd.read_csv(FX / "04_biomarker_redundancy.csv")
fs = pd.read_csv(FX / "05_feature_sensitivity.csv")
dec = json.loads((FX / "11_decisions.json").read_text())

checks = []
checks.append(("N = 1528 manifest", len(man) == 1528))
checks.append(("patients = 68", man.patient_id.nunique() == 68))
checks.append(("zero patient overlap", int(man.groupby("patient_id").fold.nunique().max()) == 1))
checks.append(("3 classes", set(man.label.unique()) == {0, 1, 2}))
for nm, d in (("fast_examiner OOF", oof), ("legacy OOF", leg), ("ropdeepx OOF", rdx),
              ("current OOF (benchmark copy)", cur)):
    checks.append((f"{nm}: 1528 rows", len(d) == 1528))
    checks.append((f"{nm}: unique image ids", d.image_id.nunique() == 1528))
    checks.append((f"{nm}: ids == manifest",
                   set(d.image_id) == set(man.image_id)))
for m in ("B", "C", "E", "G"):
    checks.append((f"OOF {m}: one prediction per image",
                   bool(oof[[f"{m}_p{c}" for c in range(3)]].notna().all().all())))
for m in ("Legacy", "ROPDeepX"):
    d = leg if m == "Legacy" else rdx
    checks.append((f"OOF {m}: one prediction per image",
                   bool(d[[f"{m}_p{c}" for c in range(3)]].notna().all().all())))
checks.append(("legacy fold/patient/label identical",
               leg.set_index("image_id").reindex(man.image_id).fold.eq(man.fold.values).all()
               and leg.set_index("image_id").reindex(man.image_id).patient_id.eq(
                   man.patient_id.values).all()
               and leg.set_index("image_id").reindex(man.image_id).true_label.eq(
                   man.label.values).all()))
checks.append(("ropdeepx fold/patient/label identical",
               rdx.set_index("image_id").reindex(man.image_id).fold.eq(man.fold.values).all()
               and rdx.set_index("image_id").reindex(man.image_id).patient_id.eq(
                   man.patient_id.values).all()
               and rdx.set_index("image_id").reindex(man.image_id).true_label.eq(
                   man.label.values).all()))

# ---------------------------------------------------------------- recompute (consistency)
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score)  # noqa: E402

y = man.set_index("image_id").loc[oof.image_id].label.to_numpy(int)
probs = {"B": oof[[f"B_p{c}" for c in range(3)]].to_numpy(float),
         "C": oof[[f"C_p{c}" for c in range(3)]].to_numpy(float),
         "E": oof[[f"E_p{c}" for c in range(3)]].to_numpy(float),
         "G": oof[[f"G_p{c}" for c in range(3)]].to_numpy(float),
         "Legacy": leg.set_index("image_id").reindex(oof.image_id)[
             [f"Legacy_p{c}" for c in range(3)]].to_numpy(float),
         "ROPDeepX": rdx.set_index("image_id").reindex(oof.image_id)[
             [f"ROPDeepX_p{c}" for c in range(3)]].to_numpy(float)}
recomputed = {}
for k, P in probs.items():
    recomputed[k] = {"auc": float(roc_auc_score(y, P, multi_class="ovr", average="macro")),
                     "bal_acc": float(balanced_accuracy_score(y, P.argmax(1))),
                     "macro_f1": float(f1_score(y, P.argmax(1), average="macro")),
                     "brier": float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1)))}

# ---------------------------------------------------------------- ground truth: metrics
GT_FX = {"B": 0.7126, "C": 0.7129, "E": 0.7077, "G": 0.7094}
GT_FB = {"Legacy": (0.6404, 0.4807, 0.4712, 0.6556, 0.1904),
         "ROPDeepX": (0.6246, 0.4539, 0.4607, 0.7854, 0.2979),
         "B": (0.7126, 0.5685, 0.5881, 0.7347, 0.3416),
         "C": (0.7129, 0.5750, 0.5952, 0.7312, 0.3369),
         "E": (0.7077, 0.5691, 0.5878, 0.7393, 0.3386),
         "G": (0.7094, 0.5694, 0.5875, 0.7422, 0.3416)}

for k, v in GT_FX.items():
    chk(f"fast_examiner 06 CSV AUC {k}", metrics_fx.loc[k, "multiclass_auc"], v)
    chk(f"recomputed AUC {k}", recomputed[k]["auc"], v)
for k, (auc, ba, f1, br, ece) in GT_FB.items():
    row = metrics_fb.loc[k]
    chk(f"benchmark CSV AUC {k}", row.auc_macro_ovr, auc)
    chk(f"benchmark CSV bal.acc {k}", row.balanced_accuracy, ba)
    chk(f"benchmark CSV macroF1 {k}", row.macro_f1, f1)
    chk(f"benchmark CSV Brier {k}", row.brier, br)
    chk(f"benchmark CSV ECE {k}", row.ece15, ece)
    chk(f"recomputed AUC {k} (benchmark)", recomputed[k]["auc"], auc)

# ---------------------------------------------------------------- ground truth: paired
GT_FX_BOOT = {("C-B", "multiclass_auc"): (0.00025, -0.00354, 0.00391, 0.889),
              ("E-B", "multiclass_auc"): (-0.00488, -0.01122, 0.00155, 0.132),
              ("G-E", "multiclass_auc"): (0.00167, -0.00166, 0.00497, 0.321)}
for (cmp_, met), (d, lo, hi, p) in GT_FX_BOOT.items():
    r = boot_fx.loc[(cmp_, met)]
    chk(f"fx bootstrap {cmp_} delta", r.delta, d)
    chk(f"fx bootstrap {cmp_} ci_lo", r.ci_lo, lo)
    chk(f"fx bootstrap {cmp_} ci_hi", r.ci_hi, hi)
    chk(f"fx bootstrap {cmp_} p", r.p_value, p, tol=5e-4)
chk("fx bootstrap replicates", boot_fx.loc[("C-B", "multiclass_auc"), "n_replicates"], 10000, tol=0)

GT_FB_BOOT = {("E-ROPDeepX", "auc"): (0.0832, 0.0240, 0.1449),
              ("E-ROPDeepX", "macro_f1"): (0.1271, 0.0604, 0.1982),
              ("E-ROPDeepX", "brier"): (-0.0461, -0.1794, 0.0825),
              ("E-Legacy", "auc"): (0.0673, 0.0031, 0.1331),
              ("E-Legacy", "macro_f1"): (0.1166, 0.0416, 0.1991),
              ("E-Legacy", "brier"): (0.0837, -0.0504, 0.2108),
              ("E-B", "auc"): (-0.0049, -0.0112, 0.0015),
              ("E-B", "macro_f1"): (-0.0002, -0.0089, 0.0084),
              ("E-B", "brier"): (0.0046, -0.0110, 0.0202)}
for (cmp_, met), (d, lo, hi) in GT_FB_BOOT.items():
    r = boot_fb.loc[(cmp_, met)]
    chk(f"fb stats {cmp_} {met} delta", r.delta, d)
    chk(f"fb stats {cmp_} {met} ci_lo", r.ci_lo, lo)
    chk(f"fb stats {cmp_} {met} ci_hi", r.ci_hi, hi)

# ---------------------------------------------------------------- per fold
GT_FOLD = {0: (0.716, 0.711, 0.713, 0.710), 1: (0.824, 0.828, 0.815, 0.816),
           2: (0.671, 0.671, 0.670, 0.676)}
pf = pd.read_csv(FX / "06b_per_fold_metrics.csv")
for k, vals in GT_FOLD.items():
    for m, v in zip(("B", "C", "E", "G"), vals):
        chk(f"per-fold {k} {m} AUC", pf[(pf.fold == k) & (pf.model == m)].auc.iloc[0], v, TOL3)

# ---------------------------------------------------------------- redundancy
red_all = red[red.scope == "farfum_all"].set_index("feature")
feats = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
X = man[feats].to_numpy(float)
R = np.corrcoef(X, rowvar=False)
chk("r(fractal_d1, fractal_d2)", R[3, 4], 0.995, 5e-4)
chk("r(fractal_d0, fractal_d1)", R[2, 3], 0.975, 5e-4)
chk("r(fractal_d0, fractal_d2)", R[2, 4], 0.956, 5e-4)
chk("CSV max VIF", red_all.vif.max(), 366, tol=0.5)
chk("CSV r(fractal_d1,fractal_d2)",
    red_all.loc["fractal_d1", "pearson_with_fractal_d2"], 0.995, 5e-4)
np.save(FX / "verify_pearson_all.npy", R)
pca_components = [json.loads((FX / f"05c_fs_detail_fold{k}.json").read_text())["FS2"]
                  for k in range(3)]
checks.append(("PCA >=95% variance = 1 component in all folds",
               pca_components == [1, 1, 1]))
checks.append(("FS conclusion NO", dec["FEATURE_SELECTION_CHANGED_CONCLUSION"] == "NO"))
checks.append(("biomarker conclusion ROBUST_NULL",
               dec["K_FOLD_BIOMARKER_CONCLUSION"] == "ROBUST_NULL"))
checks.append(("vessel conclusion NULL", dec["K_FOLD_VESSEL_CONCLUSION"] == "NULL"))
checks.append(("FS0 C-B matches bootstrap delta",
               abs(float(fs[fs.variant == "FS0"].C_minus_B.iloc[0])
                   - boot_fx.loc[("C-B", "multiclass_auc"), "delta"]) < 1e-9))

# ---------------------------------------------------------------- report
print("=" * 92)
print(f"{'CHECK':52s} {'GOT':>14s} {'WANT':>14s}  RESULT")
print("=" * 92)
bad = 0
for name, got, want, res in rows:
    if res == "FAIL":
        bad += 1
    print(f"{name:52s} {got:>14s} {want:>14s}  {res}")
for name, ok in checks:
    if not ok:
        bad += 1
    print(f"{name:52s} {'':>14s} {'':>14s}  {'PASS' if ok else 'FAIL'}")
print("=" * 92)
print(f"VALUE_CHECK = {'PASS' if bad == 0 else 'FAIL'}   mismatches={bad}   "
      f"numeric_checks={len(rows)}   integrity_checks={len(checks)}")
print("recomputed (independent sklearn recomputation from saved OOF):")
for k, v in recomputed.items():
    print(f"  {k:9s} AUC {v['auc']:.6f}  bal.acc {v['bal_acc']:.6f}  "
          f"macroF1 {v['macro_f1']:.6f}  Brier {v['brier']:.6f}")
print("pearson matrix (n=1528, first 3 decimals):")
for i, f in enumerate(feats):
    print("   ", f, [round(float(x), 3) for x in R[i]])
sys.exit(0 if bad == 0 else 1)
