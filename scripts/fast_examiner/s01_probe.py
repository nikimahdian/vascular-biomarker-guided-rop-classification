"""Step 1 probe - inventory the FARFUM-RoP subset in the canonical manifest."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/moniaz/niki")
M = ROOT / "data/splits/primary_complete_case_v2.csv"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]

m = pd.read_csv(M, low_memory=False)
print("MANIFEST", M, m.shape)
print("sources:", m.source.value_counts().to_dict())
print("split x source:")
print(pd.crosstab(m.split, m.source))
print("label counts overall:", m.label.value_counts().to_dict())
for src in sorted(m.source.dropna().unique()):
    s = m[m.source == src]
    print(f"--- source={src} n={len(s)} "
          f"labels={s.label.value_counts().to_dict()} "
          f"split={s.split.value_counts().to_dict()} "
          f"groups={s.group_id.nunique()} "
          f"patient_id_nonnull={int(s.patient_id.notna().sum())} "
          f"identity_level={s.identity_level.value_counts().to_dict()}")
f = m[m.source == "farfum_rop"].copy()
if len(f) == 0:
    for cand in m.source.dropna().unique():
        if "farfum" in cand.lower() or "farf" in cand.lower():
            f = m[m.source == cand].copy()
            print("FALLBACK source name:", cand)
print("FARFUM n:", len(f))
print("FARFUM labels:", f.label.value_counts().to_dict())
print("FARFUM split:", f.split.value_counts().to_dict())
print("FARFUM group_id nunique:", f.group_id.nunique())
print("FARFUM patient_id nonnull:", int(f.patient_id.notna().sum()),
      "unique:", f.patient_id.nunique())
print("FARFUM dup image_path:", int(f.image_path.duplicated().sum()))
print("FARFUM missing biomarkers:", int(f[FEATS].isna().any(axis=1).sum()))
print("FARFUM missing mask_path:", int(f.mask_path.isna().sum()))
miss_img = [p for p in f.image_path if not Path(p).exists()]
miss_msk = [p for p in f.mask_path if not Path(p).exists()]
print("FARFUM missing image files:", len(miss_img))
print("FARFUM missing mask files:", len(miss_msk))
if miss_msk:
    print("  e.g.", miss_msk[:3])
print("FARFUM class x patient table (first 15):")
g = f.groupby("patient_id").agg(n=("label", "size"),
                                labs=("label", lambda s: sorted(set(s))))
print(g.head(15))
print("patients with mixed labels:", int((g.labs.map(len) > 1).sum()))
print("images per patient: min %d max %d mean %.2f" % (g.n.min(), g.n.max(), g.n.mean()))
print("label x identity_level:")
print(pd.crosstab(f.label, f.identity_level))
