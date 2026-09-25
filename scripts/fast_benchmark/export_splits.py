"""Export the exact Prompt-1 internal development split (train/val) per outer fold.

Must run on the host that produced Prompt 1 (server 1) so that the
StratifiedGroupKFold implementation version is identical.
Writes results/fast_examiner/devsplit_fold{k}.csv with image_id, role.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "results/fast_examiner"
SEED = 42
m = pd.read_csv(OUT / "02_common_farFUM_3fold_manifest.csv")
print("sklearn split export; manifest", m.shape)
for fold in (0, 1, 2):
    dv = m[m.fold != fold].reset_index(drop=True)
    tr = va = None
    for seed in (SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4):
        sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        t_i, v_i = next(iter(sgkf.split(dv, y=dv.label, groups=dv.patient_id)))
        c_tr, c_va = dv.iloc[t_i], dv.iloc[v_i]
        if set(c_va.label.unique()) == {0, 1, 2} and set(c_tr.label.unique()) == {0, 1, 2}:
            tr, va = c_tr, c_va
            print(f"fold {fold}: seed {seed} train {len(tr)} val {len(va)}")
            break
    assert tr is not None
    rows = pd.concat([tr.assign(role="train"), va.assign(role="val")])[
        ["image_id", "patient_id", "label", "fold", "role"]]
    rows.to_csv(OUT / f"devsplit_fold{fold}.csv", index=False)
    assert set(tr.patient_id) & set(va.patient_id) == set()
    assert len(rows) == len(dv)
print("wrote devsplit_fold{0,1,2}.csv to", OUT)
sys.exit(0)
