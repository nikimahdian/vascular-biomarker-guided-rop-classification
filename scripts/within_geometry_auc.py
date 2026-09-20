"""Is the WHOLE project riding the acquisition-geometry confound, or only the biomarker branch?

Plus prevalence ranges from 0% (image min-side 1080, n=969) to 42% (min-side 960, n=1382).
If any model scores well overall mainly because it can infer the image geometry, its AUC should
collapse when computed WITHIN a single geometry group, where that cue is constant.

Reported for branch A (biomarkers), branch B (CNN) and branch C (fusion), on the val and test
prediction files that already existed. This re-analyses stored outputs; it does not select or
report a new headline number for the locked test.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

B = r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_work"
disc = pd.read_csv(f"{B}\\disc_predictions_all.csv")[["image_path", "w", "h"]]
disc["min_side"] = np.minimum(disc.w, disc.h)
feats = pd.read_csv(f"{B}\\v2b.csv")[["image_path", "label", "source", "split", "group_id"]]


def load(name):
    d = pd.read_csv(f"{B}\\preds\\{name}")
    return d


for nm in ("branch_a_val_preds.csv", "branch_b_val_preds.csv", "branch_c_val_preds.csv",
           "branch_a_test_preds.csv", "branch_b_test_preds.csv", "branch_c_test_preds.csv"):
    d = load(nm)
    print("=" * 100)
    print(nm, d.shape, list(d.columns)[:12])
    break

print()
for split in ("val", "test"):
    print("#" * 100)
    print(f"# {split.upper()}")
    print("#" * 100)
    tables = {}
    for br in ("a", "b", "c"):
        d = load(f"branch_{br}_{split}_preds.csv")
        key = next((c for c in d.columns if c.lower() in ("image_path", "path", "image", "id")), None)
        prob = [c for c in d.columns if c.startswith("p_") and "plus" in c.lower()]
        lab = [c for c in d.columns if c.lower() in ("label", "y", "target", "y_true")]
        if key is None or not prob or not lab:
            print(f"  branch_{br}: cannot parse columns -> {list(d.columns)}")
            continue
        t = d[[key, prob[0], lab[0]]].rename(columns={key: "image_path", prob[0]: "p", lab[0]: "label"})
        tables[br] = t
    if not tables:
        continue
    m = None
    for br, t in tables.items():
        t = t.rename(columns={"p": f"p_{br}"})
        m = t if m is None else m.merge(t[["image_path", f"p_{br}"]], on="image_path", how="outer")
    m = m.merge(disc, on="image_path", how="left").merge(
        feats[["image_path", "source"]], on="image_path", how="left")
    m["y"] = (m.label == 2).astype(int)
    print(f"\n  merged n={len(m)}  Plus={int(m.y.sum())}  missing min_side={int(m.min_side.isna().sum())}")

    print(f"\n  --- overall AUC ({split}) ---")
    for br in tables:
        s = m.dropna(subset=[f"p_{br}", "y"])
        print(f"    branch {br.upper()}: AUC={roc_auc_score(s.y, s[f'p_{br}']):.4f}  n={len(s)}")

    print(f"\n  --- AUC WITHIN each image-geometry group ---")
    hdr = f"    {'min_side':>8s} {'n':>6s} {'Plus':>6s} " + " ".join(f"{'B' if b == 'b' else b.upper():>8s}" for b in tables)
    print(hdr)
    for ms, g in m.groupby("min_side"):
        line = f"    {int(ms):8d} {len(g):6d} {int(g.y.sum()):6d} "
        for br in tables:
            s = g.dropna(subset=[f"p_{br}"])
            if len(s) > 20 and 0 < s.y.sum() < len(s):
                line += f"{roc_auc_score(s.y, s[f'p_{br}']):8.4f} "
            else:
                line += f"{'n/a':>8s} "
        print(line)

    print(f"\n  --- AUC within each SOURCE ---")
    for src, g in m.groupby("source"):
        line = f"    {src:12s} {len(g):6d} {int(g.y.sum()):6d} "
        for br in tables:
            s = g.dropna(subset=[f"p_{br}"])
            if len(s) > 20 and 0 < s.y.sum() < len(s):
                line += f"{roc_auc_score(s.y, s[f'p_{br}']):8.4f} "
            else:
                line += f"{'n/a':>8s} "
        print(line)

    print(f"\n  --- can the label be predicted from image geometry alone? ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    gg = m.dropna(subset=["min_side"]).copy()
    if "group_id" not in gg.columns:
        gg = gg.merge(feats[["image_path", "group_id"]], on="image_path", how="left")
    print(f"    n_groups={gg['group_id'].nunique()}")
    X = gg[["w", "h", "min_side"]].to_numpy(float)
    y = gg["y"].to_numpy()
    grp = gg["group_id"].fillna("x").to_numpy()
    oof = np.zeros(len(gg))
    for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    print(f"    logistic on (w, h, min_side) only -> AUC {roc_auc_score(y, oof):.4f}")
    X2 = np.column_stack([X, X[:, 0] * X[:, 1]])
    oof2 = np.zeros(len(gg))
    for tr, te in GroupKFold(n_splits=5).split(X2, y, grp):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X2[tr], y[tr])
        oof2[te] = lr.predict_proba(X2[te])[:, 1]
    print(f"    + pixel count (w*h)                -> AUC {roc_auc_score(y, oof2):.4f}")
