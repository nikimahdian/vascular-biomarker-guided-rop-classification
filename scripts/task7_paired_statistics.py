"""Task 7 — paired statistical analysis of the frozen Task-6 test predictions.

Analysis only. Reads the frozen per-image prediction tables, recomputes every metric independently,
runs a stratified paired bootstrap (seed 42, 10,000 replicates), per-class AUCs, disagreement
analysis, probability-level change, source-specific paired analysis and calibration comparison.
No model is trained, tuned, recalibrated or otherwise altered.
"""
import hashlib
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score)

WS = Path("/root/niki_rop_task6_isolated")
ART6 = WS / "artifacts/task6"
OUT = WS / "artifacts/task7_paired_statistics"
NAMES = ["Normal", "Pre_Plus", "Plus"]
PCOLS = ["prob_Normal", "prob_Pre_Plus", "prob_Plus"]
SEED = 42
NB = 10000
NCHUNK = 40


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def macro_auc(y, P):
    """Frozen Task-6 multiclass definition: 3-class macro one-vs-rest ROC-AUC.

    Undefined when fewer than three classes are present (the `plus` test subgroup has no Pre_Plus),
    in which case NaN is returned rather than an invalid two-class substitute.
    """
    present = sorted(int(v) for v in np.unique(y))
    if len(present) < 3:
        return float("nan")
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=present))


def per_class_auc(y, P, k):
    yk = (y == k).astype(int)
    return float(roc_auc_score(yk, P[:, k])) if 0 < yk.sum() < len(yk) else float("nan")


def ece15(P, y):
    pred = P.argmax(1)
    conf, corr = P.max(1), (pred == y)
    e = 0.0
    for lo in np.linspace(0, 1, 16)[:-1]:
        m = (conf >= lo) & (conf < lo + 1 / 15)
        if m.sum():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def brier(P, y):
    return float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1)))


def metrics(y, P):
    pred = P.argmax(1)
    return {"multiclass_auc": macro_auc(y, P), "macro_ovr_auc": macro_auc(y, P),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "brier": brier(P, y), "ece": ece15(P, y),
            "auc_Normal": per_class_auc(y, P, 0), "auc_Pre_Plus": per_class_auc(y, P, 1),
            "auc_Plus": per_class_auc(y, P, 2)}


KEYS = ["multiclass_auc", "macro_ovr_auc", "balanced_accuracy", "macro_f1", "brier", "ece",
        "auc_Normal", "auc_Pre_Plus", "auc_Plus"]


def strat_idx(y, rng):
    idx = np.concatenate([rng.choice(np.nonzero(y == k)[0], size=int((y == k).sum()),
                                     replace=True) for k in np.unique(y)])
    return idx


def boot_chunk(arg):
    reps, y, PB, PC = arg
    rows = []
    for r in reps:
        rng = np.random.default_rng(SEED * 1000003 + r)
        i = strat_idx(y, rng)
        mb, mc = metrics(y[i], PB[i]), metrics(y[i], PC[i])
        rows.append([mc[k] - mb[k] for k in KEYS])
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== 1. VERIFY FROZEN INPUTS")
    fb = ART6 / "test_predictions_B_EMBEDDING_ONLY.csv"
    fc = ART6 / "test_predictions_C_PRIMARY.csv"
    tsha = json.loads((ART6 / "artifact_sha256.json").read_text())
    B = pd.read_csv(fb)
    C = pd.read_csv(fc)
    checks = {"n_B_1331": len(B) == 1331, "n_C_1331": len(C) == 1331,
              "same_image_set": set(B.image_id) == set(C.image_id),
              "no_dup_B": not B.image_id.duplicated().any(),
              "no_dup_C": not C.image_id.duplicated().any(),
              "sha_B_matches": sha(fb) == tsha.get(fb.name),
              "sha_C_matches": sha(fc) == tsha.get(fc.name)}
    P = B[["image_id", "group_id", "source", "true_label"]].merge(
        C[["image_id", "true_label"] + PCOLS], on="image_id", suffixes=("", "_c"))
    checks["merged_n_1331"] = len(P) == 1331
    checks["labels_identical"] = bool((P.true_label == P.true_label_c).all())
    checks["probs_finite"] = bool(np.isfinite(P[PCOLS]).all().all())
    sums = P[PCOLS].sum(1).values
    checks["probs_sum_to_1"] = bool(np.allclose(sums, 1.0, atol=1e-6))
    for k, v in checks.items():
        print(f"  {k:24s} : {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit("INPUT_INTEGRITY_FAILED")

    P = P.rename(columns={"true_label_c": "true_label_c"})
    Bp = B.set_index("image_id").loc[P.image_id, PCOLS].values
    Cp = C.set_index("image_id").loc[P.image_id, PCOLS].values
    y = P.true_label.values
    PB, PC = Bp, Cp
    print(f"  classes {np.bincount(y).tolist()}  test_n {len(y)}")

    print("=== 2. OBSERVED METRICS")
    mb, mc = metrics(y, PB), metrics(y, PC)
    obs = {k: mc[k] - mb[k] for k in KEYS}
    for k in KEYS:
        print(f"  {k:20s} B {mb[k]:.6f}  C {mc[k]:.6f}  delta {obs[k]:+.6f}")

    print(f"=== 3. PAIRED STRATIFIED BOOTSTRAP  seed={SEED}  B={NB}")
    chunks = [list(range(i, min(i + NCHUNK, NB))) for i in range(0, NB, NCHUNK)]
    with get_context("fork").Pool(24) as pool:
        res = pool.map(boot_chunk, [(c, y, PB, PC) for c in chunks])
    D = np.array([row for chunk in res for row in chunk])
    BD = pd.DataFrame(D, columns=KEYS)
    BD.to_parquet(OUT / "bootstrap_distributions.parquet", index=False)
    print(f"  replicates {len(BD)}")

    rows = []
    for k in KEYS:
        d = BD[k].values
        lo, hi = np.percentile(d, [2.5, 97.5])
        centered = d - obs[k]
        p = float(np.mean(np.abs(centered) >= abs(obs[k])))
        rows.append({"metric": k, "B_value": mb[k], "C_value": mc[k], "delta": obs[k],
                     "boot_mean_delta": float(d.mean()), "boot_median_delta": float(np.median(d)),
                     "ci95_lo": float(lo), "ci95_hi": float(hi),
                     "p_two_sided_null_centered": p,
                     "ci_crosses_zero": bool(lo <= 0 <= hi)})
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "paired_primary_metrics.csv", index=False)
    print()
    print(M[["metric", "B_value", "C_value", "delta", "ci95_lo", "ci95_hi",
             "p_two_sided_null_centered"]].to_string(index=False))

    print("=== 5. PER-CLASS AUC")
    pc = M[M.metric.str.startswith("auc_")].copy()
    pc.to_csv(OUT / "paired_per_class_auc.csv", index=False)
    print(pc[["metric", "B_value", "C_value", "delta", "ci95_lo", "ci95_hi"]].to_string(index=False))

    print("=== 6. DISAGREEMENT")
    pb, pcc = PB.argmax(1), PC.argmax(1)
    changed = pb != pcc
    cb, cc = pb == y, pcc == y
    disc = {"n": int(len(y)), "changed_n": int(changed.sum()),
            "changed_pct": float(changed.mean() * 100),
            "B_wrong_C_correct": int((~cb & cc).sum()),
            "B_correct_C_wrong": int((cb & ~cc).sum()),
            "both_wrong_different_class": int((~cb & ~cc & changed).sum()),
            "net_corrected": int((~cb & cc).sum() - (cb & ~cc).sum())}
    disc["mcnemar_exact_p"] = float(binomtest(disc["B_wrong_C_correct"],
                                              disc["B_wrong_C_correct"] + disc["B_correct_C_wrong"],
                                              0.5).pvalue)
    trans = pd.crosstab(pd.Series(pb, name="B_pred"), pd.Series(pcc, name="C_pred"))
    trans.to_csv(OUT / "prediction_disagreement.csv")
    print(json.dumps(disc, indent=2))
    print(trans.to_string())

    print("=== 7. PROBABILITY-LEVEL CHANGE")
    ptrue = PB[np.arange(len(y)), y]
    ptrue_c = PC[np.arange(len(y)), y]
    ch = pd.DataFrame({"abs_change_true_class_prob": np.abs(ptrue_c - ptrue),
                       "l1_distance": np.abs(PC - PB).sum(1),
                       "max_abs_class_change": np.abs(PC - PB).max(1)})
    summ = []
    for c in ch.columns:
        v = ch[c].values
        summ.append({"quantity": c, "median": float(np.median(v)),
                     "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
                     "p95": float(np.percentile(v, 95)), "max": float(v.max())})
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "probability_change_summary.csv", index=False)
    print(S.to_string(index=False))

    print("=== 8. SOURCE-SPECIFIC PAIRED ANALYSIS")
    srows = []
    for src in ("farfum_rop", "farabi", "plus"):
        sel = (P.source == src).values
        if sel.sum() < 10:
            continue
        ys, PBs, PCs = y[sel], PB[sel], PC[sel]
        present = sorted(int(v) for v in np.unique(ys))
        keys = ["multiclass_auc", "balanced_accuracy", "macro_f1"] if len(present) == 3 else \
               ["balanced_accuracy", "macro_f1", "auc_Normal", "auc_Plus"]
        rng = np.random.default_rng(SEED)
        Ds = []
        for r in range(2000):
            i = strat_idx(ys, np.random.default_rng(SEED * 7919 + r))
            a, b_ = metrics(ys[i], PBs[i]), metrics(ys[i], PCs[i])
            Ds.append([b_[k] - a[k] for k in keys])
        Ds = np.array(Ds)
        full_b, full_c = metrics(ys, PBs), metrics(ys, PCs)
        for j, k in enumerate(keys):
            lo, hi = np.percentile(Ds[:, j], [2.5, 97.5])
            srows.append({"source": src, "n": int(sel.sum()), "classes_present": present,
                          "metric": k, "B_value": full_b[k], "C_value": full_c[k],
                          "delta": full_c[k] - full_b[k], "ci95_lo": float(lo),
                          "ci95_hi": float(hi), "boot_replicates": 2000,
                          "note": "secondary source analysis; small N; multiclass AUC undefined "
                                  "for plus (no Pre_Plus)" if len(present) < 3 else "secondary"})
    SS = pd.DataFrame(srows)
    SS.to_csv(OUT / "paired_source_analysis.csv", index=False)
    print(SS[["source", "n", "metric", "B_value", "C_value", "delta", "ci95_lo",
              "ci95_hi"]].to_string(index=False))

    print("=== 9. CALIBRATION FIGURE")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        for nm, P_ in (("B_EMBEDDING_ONLY", PB), ("C_PRIMARY", PC)):
            conf, corr = P_.max(1), (P_.argmax(1) == y)
            xs, ys_ = [], []
            for lo in np.linspace(0, 1, 16)[:-1]:
                m = (conf >= lo) & (conf < lo + 1 / 15)
                if m.sum():
                    xs.append(conf[m].mean()); ys_.append(corr[m].mean())
            ax[0].plot(xs, ys_, marker="o", ms=3, label=nm)
        ax[0].plot([0, 1], [0, 1], "k--", lw=0.7)
        ax[0].set_xlabel("confidence"); ax[0].set_ylabel("accuracy")
        ax[0].set_title("reliability, 15 equal-width bins", fontsize=8); ax[0].legend(fontsize=7)
        mm = M[M.metric.isin(["multiclass_auc", "balanced_accuracy", "macro_f1", "brier", "ece"])]
        ypos = np.arange(len(mm))
        ax[1].errorbar(mm.delta, ypos,
                       xerr=[mm.delta - mm.ci95_lo, mm.ci95_hi - mm.delta], fmt="o", ms=4)
        ax[1].axvline(0, color="k", lw=0.8, ls="--")
        ax[1].set_yticks(ypos); ax[1].set_yticklabels(mm.metric, fontsize=7)
        ax[1].set_title("delta C - B, 95% paired bootstrap CI", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "calibration_comparison.png", dpi=170)
        fig.savefig(OUT / "delta_forest_plot.png", dpi=170)
        plt.close(fig)
        print("  figures written")
    except Exception as e:  # noqa: BLE001
        print("  figures skipped:", e)

    shas = {p.name: sha(p) for p in sorted(OUT.glob("*")) if p.is_file()}
    (OUT / "artifact_sha256.json").write_text(json.dumps(shas, indent=2))
    summary = {"paired_n": int(len(y)), "bootstrap_replicates": NB, "seed": SEED,
               "input_checks": checks, "observed_B": mb, "observed_C": mc, "deltas": obs,
               "paired_metrics": M.to_dict("records"), "disagreement": disc,
               "probability_change": summ, "source_analysis": srows,
               "artifact_sha256": shas,
               "test_status": "CANONICAL_LOCKED_SPLIT_REANALYSIS",
               "model_selection_used_test": False}
    (OUT / "paired_statistics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print()
    print("TASK7_STATUS = COMPLETE")
    return summary


if __name__ == "__main__":
    main()
