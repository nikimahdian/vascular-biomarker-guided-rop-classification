#!/usr/bin/env python3
"""Late-fusion probe: stack Branch A/B probs. Fit on val only, evaluate test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

ROOT = Path("/Users/moniaz/niki")


def plus_bin(y) -> np.ndarray:
    return (np.asarray(y) == 2).astype(int)


def metrics(y, scores, thr: float | None = None) -> dict:
    y_bin = plus_bin(y)
    scores = np.asarray(scores, dtype=float)
    auc = float(roc_auc_score(y_bin, scores))
    if thr is None:
        best_thr, best_j = 0.5, -1.0
        for t in np.unique(np.round(scores, 4)):
            pred = (scores >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_bin, pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn + 1e-12)
            spec = tn / (tn + fp + 1e-12)
            j = sens + spec - 1
            if j > best_j:
                best_j, best_thr = j, float(t)
        thr = best_thr
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn + 1e-12))
    spec = float(tn / (tn + fp + 1e-12))
    return {
        "auc": auc,
        "thr": float(thr),
        "sens": sens,
        "spec": spec,
        "f1": float(f1_score(y_bin, pred, zero_division=0)),
    }


def score_col(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c.startswith("p_plus")]
    if not cols:
        raise SystemExit(f"no p_plus col in {list(df.columns)}")
    return cols[0]


def main() -> None:
    av = pd.read_csv(ROOT / "results/branch_a_val_preds.csv")
    at = pd.read_csv(ROOT / "results/branch_a_test_preds.csv")
    bv = pd.read_csv(ROOT / "results/branch_b_val_preds.csv")
    bt = pd.read_csv(ROOT / "results/branch_b_test_preds.csv")

    a_col = score_col(av)
    b_col = score_col(bv)
    assert score_col(at) == a_col and score_col(bt) == b_col

    val = av[["image_path", "label", a_col]].merge(
        bv[["image_path", b_col]], on="image_path", how="inner"
    )
    test = at[["image_path", "label", a_col]].merge(
        bt[["image_path", b_col]], on="image_path", how="inner"
    )
    assert len(val) == len(av) == len(bv), (len(val), len(av), len(bv))
    assert len(test) == len(at) == len(bt), (len(test), len(at), len(bt))

    Xv = val[[a_col, b_col]].to_numpy()
    Xt = test[[a_col, b_col]].to_numpy()
    yv = plus_bin(val.label)
    yt = plus_bin(test.label)

    print(f"n_val={len(val)} n_test={len(test)} a_col={a_col} b_col={b_col}")

    print("\n=== BASELINES test (thr from val Youden) ===")
    baselines = {}
    for name, sval, stest in [
        ("A", val[a_col], test[a_col]),
        ("B", val[b_col], test[b_col]),
        ("mean", 0.5 * val[a_col] + 0.5 * val[b_col], 0.5 * test[a_col] + 0.5 * test[b_col]),
    ]:
        mval = metrics(val.label, sval)
        mt = metrics(test.label, stest, thr=mval["thr"])
        baselines[name] = mt
        print(
            f"{name}: AUC={mt['auc']:.4f} thr={mt['thr']:.4f} "
            f"sens={mt['sens']:.4f} spec={mt['spec']:.4f} f1={mt['f1']:.4f}"
        )

    print("\n=== WEIGHT GRID w*A+(1-w)*B  (select by val AUC) ===")
    best = None
    ranked = []
    for w in np.linspace(0.0, 1.0, 101):
        sv = w * val[a_col].to_numpy() + (1.0 - w) * val[b_col].to_numpy()
        st = w * test[a_col].to_numpy() + (1.0 - w) * test[b_col].to_numpy()
        mval = metrics(val.label, sv)
        mt = metrics(test.label, st, thr=mval["thr"])
        ranked.append((float(w), mval["auc"], mt["auc"], mt))
        if best is None or mval["auc"] > best[1]:
            best = (float(w), mval["auc"], mt["auc"], mt, mval["thr"])

    w_star, val_auc, test_auc, mt_w, thr_w = best
    print(
        f"best_wA={w_star:.2f} wB={1-w_star:.2f} valAUC={val_auc:.4f} "
        f"testAUC={test_auc:.4f} thr={thr_w:.4f} sens={mt_w['sens']:.4f} "
        f"spec={mt_w['spec']:.4f} f1={mt_w['f1']:.4f}"
    )
    print("top5 by valAUC:")
    for w, va, ta, mt in sorted(ranked, key=lambda x: -x[1])[:5]:
        print(
            f"  wA={w:.2f} val={va:.4f} test={ta:.4f} "
            f"sens={mt['sens']:.4f} spec={mt['spec']:.4f} f1={mt['f1']:.4f}"
        )

    print("\n=== LOGISTIC STACK (fit val only) ===")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    pt = clf.predict_proba(Xt)[:, 1]
    mval = metrics(val.label, pv)
    mt_s = metrics(test.label, pt, thr=mval["thr"])
    print(
        f"coefA={clf.coef_[0][0]:.4f} coefB={clf.coef_[0][1]:.4f} "
        f"intercept={clf.intercept_[0]:.4f}"
    )
    print(
        f"stack valAUC={mval['auc']:.4f} testAUC={mt_s['auc']:.4f} "
        f"thr={mt_s['thr']:.4f} sens={mt_s['sens']:.4f} "
        f"spec={mt_s['spec']:.4f} f1={mt_s['f1']:.4f}"
    )

    c_path = ROOT / "results/branch_c_results.json"
    c_auc = None
    if c_path.exists():
        c = json.loads(c_path.read_text())
        bm = c["best_model"]
        c_auc = float(c["metrics"][bm]["test"]["plus_ovr"]["auc"])

    b_auc = baselines["B"]["auc"]
    print("\n=== VERDICT ===")
    print(f"B={b_auc:.4f}")
    if c_auc is not None:
        print(f"C_fusion={c_auc:.4f}")
    print(f"weight_fuse={test_auc:.4f}  vsB={test_auc-b_auc:+.4f}")
    print(f"stack={mt_s['auc']:.4f}  vsB={mt_s['auc']-b_auc:+.4f}")
    if c_auc is not None:
        print(f"weight vsC={test_auc-c_auc:+.4f}  stack vsC={mt_s['auc']-c_auc:+.4f}")

    beat_b_w = test_auc > b_auc + 1e-6
    beat_b_s = mt_s["auc"] > b_auc + 1e-6
    print(f"weight beats B? {beat_b_w}")
    print(f"stack beats B? {beat_b_s}")

    out = {
        "method": "late_fusion_probe_val_fit_test_eval",
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "baselines_test": baselines,
        "c_fusion_test_auc": c_auc,
        "weight_grid_best": {
            "w_A": w_star,
            "w_B": 1.0 - w_star,
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
            "thr_from_val": float(thr_w),
            "test_sens": mt_w["sens"],
            "test_spec": mt_w["spec"],
            "test_f1": mt_w["f1"],
            "beats_B": beat_b_w,
        },
        "logistic_stack": {
            "coef_A": float(clf.coef_[0][0]),
            "coef_B": float(clf.coef_[0][1]),
            "intercept": float(clf.intercept_[0]),
            "val_auc": float(mval["auc"]),
            "test_auc": float(mt_s["auc"]),
            "thr_from_val": float(mval["thr"]),
            "test_sens": mt_s["sens"],
            "test_spec": mt_s["spec"],
            "test_f1": mt_s["f1"],
            "beats_B": beat_b_s,
        },
    }
    out_path = ROOT / "results/branch_c_late_fusion_probe.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
