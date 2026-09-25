"""Step 2 - common FARFUM cohort + deterministic patient-level 3-fold manifest.

Writes:
  results/fast_examiner/01_common_farFUM_cohort.csv
  results/fast_examiner/02_common_farFUM_3fold_manifest.csv
  results/fast_examiner/02_common_farFUM_3fold_manifest.sha256
Exits non-zero if any patient appears in more than one fold.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "results/fast_examiner"
MAN = ROOT / "data/splits/primary_complete_case_v2.csv"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
SEED = 42
K = 3

m = pd.read_csv(MAN, low_memory=False)
f = m[m.source == "farfum_rop"].copy().reset_index(drop=True)
f = f.sort_values("image_path").reset_index(drop=True)   # deterministic order

# ---- integrity gates -------------------------------------------------------
gates = {}
gates["n_1528"] = len(f) == 1528
gates["labels_012"] = set(f.label.unique()) == {0, 1, 2}
gates["patient_id_complete"] = bool(f.patient_id.notna().all())
gates["dup_image_path"] = int(f.image_path.duplicated().sum()) == 0
gates["dup_image_id"] = int(f.image_path.duplicated().sum()) == 0
gates["biomarkers_complete"] = bool(f[FEATS].notna().all().all())
gates["masks_exist"] = bool(f.mask_path.map(lambda p: Path(p).exists()).all())
gates["images_exist"] = bool(f.image_path.map(lambda p: Path(p).exists()).all())
gates["groups_eq_patients"] = f.group_id.nunique() == f.patient_id.nunique()
for k, v in gates.items():
    print(f"GATE {k:22s} {'PASS' if v else 'FAIL'}")
if not all(gates.values()):
    sys.exit("COHORT_GATE_FAILED")

# redundant identity columns kept for downstream joins
for c in ("group_id", "exam_id"):
    if c not in f.columns:
        f[c] = f.patient_id

# ---- deterministic class-stratified patient-level fold assignment ---------
# StratifiedGroupKFold: folds are stratified on the image label while no patient
# (group) can appear in two folds. Deterministic (shuffle with fixed seed over a
# manifest that is itself sorted by image_path).
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

class_imgs = {int(k2): int(v) for k2, v in f.label.value_counts().items()}
print("class image totals:", class_imgs)
prof = (f.groupby("patient_id").label.value_counts().unstack(fill_value=0)
          .reindex(columns=[0, 1, 2], fill_value=0))
prof["n_images"] = prof.sum(axis=1)
prof = prof.reset_index()
prof["n_labels"] = (prof[[0, 1, 2]] > 0).sum(axis=1)
print("patients by label profile:",
      prof.groupby(prof[[0, 1, 2]].gt(0).apply(tuple, axis=1)).size().to_dict())

sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)
assign = {}
for k, (_, te_idx) in enumerate(sgkf.split(f, y=f.label, groups=f.patient_id)):
    for pid in f.patient_id.iloc[te_idx].unique():
        assert pid not in assign, "patient assigned twice"
        assign[pid] = k
assert len(assign) == f.patient_id.nunique(), "not every patient assigned"
prof["fold"] = prof.patient_id.map(assign).astype(int)
pat = prof.copy()

counts = np.zeros((K, 3))
totals = np.zeros(K)
for k in range(K):
    s = f[f.patient_id.map(assign) == k]
    for lab in range(3):
        counts[k, lab] = int((s.label == lab).sum())
    totals[k] = len(s)

f["fold"] = f.patient_id.map(assign).astype(int)
n_pat_fold = pat.fold.value_counts().sort_index().to_dict()

# ---- STOP if patient overlap -------------------------------------------------
ov = f.groupby("patient_id").fold.nunique()
print("patients in >1 fold:", int((ov > 1).sum()))
if int((ov > 1).sum()) != 0:
    sys.exit("PATIENT_OVERLAP_DETECTED")

print("\nper-fold table")
rows = []
for k in range(K):
    s = f[f.fold == k]
    row = {"fold": k, "patients": int(s.patient_id.nunique()), "images": len(s),
           "Normal": int((s.label == 0).sum()), "Pre_Plus": int((s.label == 1).sum()),
           "Plus": int((s.label == 2).sum())}
    rows.append(row)
    print(row)
tab = pd.DataFrame(rows)
print("\nper-fold patient x label-profile presence")
print(pd.crosstab(pat.fold, pat[[0, 1, 2]].gt(0).apply(lambda r: "+".join(str(i) for i in [0, 1, 2] if r[i]), axis=1)))
print("fold image totals:", [int(v) for v in totals])
print("fold class matrices:\n", counts.astype(int))

# ---- export ------------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
cols = ["image_path", "mask_path", "label", "patient_id", "group_id", "source",
        "fold"] + FEATS
coh = f[cols].rename(columns={"image_path": "path"})
coh.insert(0, "image_id", coh.path.map(lambda p: Path(p).stem))
coh["image_sha256_source"] = ""
coh.to_csv(OUT / "01_common_farFUM_cohort.csv", index=False)
man = f[["image_path", "mask_path", "label", "patient_id", "group_id", "fold"] + FEATS].copy()
man.insert(0, "image_id", man.image_path.map(lambda p: Path(p).stem))
man.to_csv(OUT / "02_common_farFUM_3fold_manifest.csv", index=False)
h = hashlib.sha256((OUT / "02_common_farFUM_3fold_manifest.csv").read_bytes()).hexdigest()
(OUT / "02_common_farFUM_3fold_manifest.sha256").write_text(h + "\n")
meta = {"seed": SEED, "k": K, "n_images": len(f), "n_patients": int(f.patient_id.nunique()),
        "manifest_sha256": h, "fold_table": rows,
        "mixed_label_patients": int((f.groupby("patient_id").label.nunique() > 1).sum()),
        "class_totals": {str(k2): int(v) for k2, v in class_imgs.items()}}
(OUT / "02_common_farFUM_3fold_manifest_meta.json").write_text(json.dumps(meta, indent=2))
print("\nMANIFEST_SHA256", h)
print("wrote", OUT / "01_common_farFUM_cohort.csv")
print("wrote", OUT / "02_common_farFUM_3fold_manifest.csv")
