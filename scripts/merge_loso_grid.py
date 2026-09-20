"""Merge the legacy 6 LOSO rows with the 3 new farabi rows into the canonical 9/9 grid."""
import json

import pandas as pd

B = r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_work"
legacy = json.load(open(f"{B}\\loo_partial.json"))
new = json.load(open(f"{B}\\loo_new.json"))["results"]

rows = []
for r in legacy + new:
    rows.append(dict(holdout=r["holdout"], branch=r["branch"],
                     auc=r["plus_ovr"]["auc"], sensitivity=r["plus_ovr"]["sensitivity"],
                     specificity=r["plus_ovr"]["specificity"], f1=r["plus_ovr"]["f1"],
                     n_test=r.get("n_test"), best_model=r.get("best_model")))
d = pd.DataFrame(rows).drop_duplicates(subset=["holdout", "branch"], keep="last")
d = d.sort_values(["holdout", "branch"]).reset_index(drop=True)
print("rows:", len(d))
print(d.to_string(index=False))

piv = d.pivot(index="holdout", columns="branch", values="auc")
piv["C_minus_B"] = piv["C"] - piv["B"]
piv = piv[["A", "B", "C", "C_minus_B"]]
print("\n=== canonical LOSO AUC grid ===")
print(piv.round(4).to_string())

in_dist = {"A": 0.799926, "B": 0.928008, "C": 0.912315}
print("\n=== in-distribution (locked test) vs LOSO mean ===")
for br in ("A", "B", "C"):
    m = piv[br].mean()
    print(f"  branch {br}: locked test {in_dist[br]:.4f}   LOSO mean {m:.4f}   drop {m - in_dist[br]:+.4f}")

print("\n=== how often does fusion beat the image alone? ===")
w = (piv["C"] > piv["B"]).sum()
print(f"  C > B in {w}/3 held-out sources")
print(f"  on the locked test C < B (0.9123 vs 0.9280, DeLong p=0.00806)")

d.to_csv(f"{B}\\loo_summary_full.csv", index=False)
json.dump({"results": rows, "protocol": {
    "holdouts": ["farabi", "farfum_rop", "plus"], "branches": ["A", "B", "C"],
    "complete_cartesian_product": len(d) == 9, "observed_rows": len(d), "expected_rows": 9,
    "note": ("6 rows from the legacy run (loo_partial.json) plus 3 farabi rows produced by "
             "src.compare.leave_one_source_out --holdouts farabi --reuse-b, which evaluated the "
             "existing farabi Branch B checkpoint instead of retraining it."),
}}, open(f"{B}\\loo_results_full.json", "w"), indent=2)
print(f"\n[done] -> {B}\\loo_summary_full.csv, loo_results_full.json")
