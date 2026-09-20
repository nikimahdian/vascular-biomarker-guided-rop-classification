"""Is the loss of disease signal from v2 -> v2c caused by re-normalising, or merely by the
smaller coverage (42.5% of rows have a measured disc)?

The v2b table carries BOTH normalisations side by side for every row plus an `dd_ok` flag, so the
two effects can be separated.

  auc_v2_all      : v2 width (fake DD), all rows
  auc_v2_okset    : v2 width (fake DD), restricted to the rows where a disc was found  <- key control
  auc_v2c_okset   : v2 width / measured DD, same rows

If auc_v2_okset is still near 0.70 and auc_v2c_okset is near 0.55, the signal really was a
normalisation artefact (i.e. a camera-zoom proxy), not a subset/coverage effect.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

B = r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_work"
d = pd.read_csv(f"{B}\\v2b.csv")
print("rows", len(d), " dd_ok", int(d.dd_ok.sum()))

FEATS = ["width_p50_dd", "width_p90_dd", "width_p95_dd", "width_mean_dd",
         "a_width_p90_dd", "v_width_p90_dd"]


def auc(y, x):
    m = np.isfinite(x)
    if m.sum() < 30 or len(np.unique(y[m])) < 2:
        return np.nan, int(m.sum())
    return roc_auc_score(y[m], x[m]), int(m.sum())


rows = []
ok = d[d.dd_ok == 1]
for f in FEATS:
    a_all, n_all = auc((d.label == 2).astype(int).to_numpy(), d[f].to_numpy())
    a_ok, n_ok = auc((ok.label == 2).astype(int).to_numpy(), ok[f].to_numpy())
    true_col = f + "_true"
    a_true, n_true = auc((ok.label == 2).astype(int).to_numpy(), ok[true_col].to_numpy())
    rows.append(dict(feature=f, auc_v2_all=a_all, n_all=n_all, frac_all=n_all / len(d),
                     auc_v2_okset=a_ok, n_ok=n_ok,
                     auc_v2c_okset=a_true, n_true=n_true,
                     drop_from_renorm=a_true - a_ok, drop_from_subset=a_ok - a_all))
out = pd.DataFrame(rows)
print("\n=== Plus(=2) vs rest, AUC ===")
print(out.round(4).to_string(index=False))

print("\n=== Plus(=2) vs No-Plus(=0), excluding Pre-Plus(=1) ===")
sub2 = d[d.label.isin([0, 2])]
sub2ok = sub2[sub2.dd_ok == 1]
rows2 = []
for f in FEATS:
    a_all, _ = auc((sub2.label == 2).astype(int).to_numpy(), sub2[f].to_numpy())
    a_ok, _ = auc((sub2ok.label == 2).astype(int).to_numpy(), sub2ok[f].to_numpy())
    a_true, _ = auc((sub2ok.label == 2).astype(int).to_numpy(), sub2ok[f + "_true"].to_numpy())
    rows2.append(dict(feature=f, auc_v2_all=a_all, auc_v2_okset=a_ok, auc_v2c_okset=a_true,
                      drop_from_renorm=a_true - a_ok))
print(pd.DataFrame(rows2).round(4).to_string(index=False))

print("\n=== is the v2 width column just an image-size proxy? ===")
d["min_side"] = np.minimum(
    d.image_path.map(pd.read_csv(f"{B}\\disc_predictions_all.csv").set_index("image_path")["w"]),
    d.image_path.map(pd.read_csv(f"{B}\\disc_predictions_all.csv").set_index("image_path")["h"]))
print("  corr(width_p90_dd, min_side)      =", round(d["width_p90_dd"].corr(d["min_side"]), 4))
print("  corr(width_p90_dd_true, min_side) =",
      round(d.loc[d.dd_ok == 1, "width_p90_dd_true"].corr(d.loc[d.dd_ok == 1, "min_side"]), 4))
print("\n  median min_side by label (all rows):")
print(d.groupby("label")["min_side"].describe()[["count", "mean", "50%"]].round(1).to_string())
print("\n  label x image-size crosstab (share of each size that is Plus):")
d["size"] = d.min_side.astype(int)
ct = pd.crosstab(d["size"], d["label"], normalize="index").round(3)
ct["n"] = d.groupby("size").size()
print(ct.to_string())
print("\n  -> if Plus is concentrated in particular image sizes, any width normalised by image")
print("     size will look like a Plus predictor without measuring caliber.")
