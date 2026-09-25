"""Part 2 / P0 - port the Prompt-1 FARFUM cohort to server 2 and verify every path.

Reads the Prompt-1 outputs (already uploaded next to this script), remaps the
server-1 absolute image/mask paths to the server-2 mirror, verifies existence,
and re-checks that fold / patient / label columns are byte-identical to Prompt 1.
"""
import json
import sys
from pathlib import Path

import pandas as pd

WS = Path("/root/niki_rop_task6_isolated")
B = WS / "_bench"
OUT = WS / "results/fast_benchmark"
OUT.mkdir(parents=True, exist_ok=True)
MIRROR = WS / "server1_data/Users/moniaz/niki"
S1_PREFIX = "/Users/moniaz/niki/"

man = pd.read_csv(B / "02_common_farFUM_3fold_manifest.csv")
oof = pd.read_csv(B / "03_current_models_oof.csv")
coh = pd.read_csv(B / "01_common_farFUM_cohort.csv")
print("manifest", man.shape, "oof", oof.shape, "cohort", coh.shape)


def remap(p: str) -> str:
    return str(MIRROR / p[len(S1_PREFIX):]) if p.startswith(S1_PREFIX) else p


man["image_path_s2"] = man.image_path.map(remap)
man["mask_path_s2"] = man.mask_path.map(remap)

gates = {
    "n_1528": len(man) == 1528,
    "labels_012": set(man.label.unique()) == {0, 1, 2},
    "folds_012": set(man.fold.unique()) == {0, 1, 2},
    "patients_68": man.patient_id.nunique() == 68,
    "no_patient_in_two_folds": int(man.groupby("patient_id").fold.nunique().max()) == 1,
}
for name, col in (("images_exist_s2", "image_path_s2"), ("masks_exist_s2", "mask_path_s2")):
    miss = [p for p in man[col] if not Path(p).exists()]
    gates[name] = len(miss) == 0
    if miss:
        print(f"  MISSING {name}: {len(miss)} e.g. {miss[:2]}")
# OOF integrity against the manifest
m = man.set_index("image_id")
j = oof.join(m[["fold", "patient_id", "label"]], on="image_id", rsuffix="_man")
gates["oof_rows"] = len(oof) == 1528
gates["oof_ids_unique"] = oof.image_id.nunique() == 1528
gates["oof_fold_match"] = bool((j.fold == j.fold_man).all()) if "fold_man" in j else False
gates["oof_patient_match"] = bool((j.patient_id == j.patient_id_man).all())
gates["oof_label_match"] = bool((j.true_label == j.label).all())
gates["oof_models_present"] = all(f"{x}_p{c}" in oof.columns for x in ("B", "C", "E", "G")
                                  for c in range(3))
for k, v in gates.items():
    print(f"GATE {k:26s} {'PASS' if v else 'FAIL'}")
if not all(gates.values()):
    sys.exit("PORT_GATE_FAILED")

man.to_csv(OUT / "02_manifest_server2.csv", index=False)
oof.to_csv(OUT / "02b_current_models_oof.csv", index=False)
(OUT / "02c_port_gates.json").write_text(json.dumps(gates, indent=1))
print("folds:", man.fold.value_counts().sort_index().to_dict())
print("classes per fold:")
print(pd.crosstab(man.fold, man.label))
print("wrote", OUT / "02_manifest_server2.csv", OUT / "02b_current_models_oof.csv")
