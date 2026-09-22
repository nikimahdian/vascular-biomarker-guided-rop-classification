"""Task 8B — statistical closure only. No training, no refit, no tuning.

Re-runs the primary paired comparison E_RGB_VESSEL_FEATURE_FUSION vs B_EMBEDDING_ONLY with the
EXACT Task-7 bootstrap protocol (paired, class-stratified, seed 42, 10,000 replicates, N=1331, same
frozen prediction CSVs, same metric implementations) and adds a second comparison
G_RGB_VESSEL_SCALAR_FUSION vs E_RGB_VESSEL_FEATURE_FUSION under the identical protocol.

Read-only with respect to every model and every prediction table: input SHA-256 values are checked
against the frozen Task-6 / Task-8 manifests before any statistic is computed, and no file in the
parent artifact directory is written or replaced. The original 2,000-replicate Task-8 provenance
stays exactly as it is.
"""
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

WS = Path("/root/niki_rop_task6_isolated")
ART6 = WS / "artifacts/task6"
ART8 = WS / "artifacts/task8_spatial_vessel_fusion"
OUT = ART8 / "statistical_closure_10k"
NAMES = ["Normal", "Pre_Plus", "Plus"]
PCOLS = ["prob_Normal", "prob_Pre_Plus", "prob_Plus"]
SEED = 42
NB = 10000
NCHUNK = 40
KEYS = ["multiclass_auc", "balanced_accuracy", "macro_f1", "brier", "ece"]
ALLKEYS = KEYS + ["auc_Normal", "auc_Pre_Plus", "auc_Plus"]


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


# ---- metric implementations, copied verbatim from scripts/task7_paired_statistics.py
def macro_auc(y, P):
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


def strat_idx(y, rng):
    return np.concatenate([rng.choice(np.nonzero(y == k)[0], size=int((y == k).sum()),
                                      replace=True) for k in np.unique(y)])


def boot_chunk(arg):
    reps, y, PA, PB = arg
    rows = []
    for r in reps:
        rng = np.random.default_rng(SEED * 1000003 + r)
        i = strat_idx(y, rng)
        ma, mb = metrics(y[i], PA[i]), metrics(y[i], PB[i])
        rows.append([mb[k] - ma[k] for k in ALLKEYS])
    return rows


def run_bootstrap(tag, y, PA, PB, obs):
    chunks = [list(range(i, min(i + NCHUNK, NB))) for i in range(0, NB, NCHUNK)]
    with get_context("fork").Pool(24) as pool:
        res = pool.map(boot_chunk, [(c, y, PA, PB) for c in chunks])
    D = np.array([row for chunk in res for row in chunk])
    pd.DataFrame(D, columns=ALLKEYS).to_parquet(OUT / f"bootstrap_distributions_{tag}.parquet",
                                                index=False)
    rows = []
    for k in ALLKEYS:
        d = D[:, ALLKEYS.index(k)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(np.mean(np.abs(d - obs[k]) >= abs(obs[k])))
        rows.append({"comparison": tag, "metric": k, "delta": obs[k],
                     "boot_mean_delta": float(d.mean()),
                     "boot_median_delta": float(np.median(d)),
                     "ci95_lo": float(lo), "ci95_hi": float(hi),
                     "p_two_sided_null_centered": p,
                     "ci_crosses_zero": bool(lo <= 0 <= hi),
                     "replicates": NB, "seed": SEED})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== 0. FROZEN INPUT VERIFICATION (nothing is written back)")
    t6 = json.loads((ART6 / "artifact_sha256.json").read_text())
    t8 = json.loads((ART8 / "artifact_sha256.json").read_text())
    files = {
        "A_PRIMARY": (ART6 / "test_predictions_A_PRIMARY.csv", t6),
        "B_EMBEDDING_ONLY": (ART6 / "test_predictions_B_EMBEDDING_ONLY.csv", t6),
        "C_PRIMARY": (ART6 / "test_predictions_C_PRIMARY.csv", t6),
        "D_VESSEL_MAP": (ART8 / "test_predictions_D_VESSEL_MAP.csv", t8),
        "E_RGB_VESSEL_FEATURE_FUSION":
            (ART8 / "test_predictions_E_RGB_VESSEL_FEATURE_FUSION.csv", t8),
        "F_RGB_VESSEL_LATE_FUSION":
            (ART8 / "test_predictions_F_RGB_VESSEL_LATE_FUSION.csv", t8),
        "G_RGB_VESSEL_SCALAR_FUSION":
            (ART8 / "test_predictions_G_RGB_VESSEL_SCALAR_FUSION.csv", t8),
    }
    sha_now, checks = {}, {}
    for nm, (p, man) in files.items():
        s = sha(p)
        sha_now[nm] = s
        checks[f"sha_unchanged_{nm}"] = (s == man.get(p.name))
        print(f"  {nm:32s} sha {s[:16]}  {'UNCHANGED' if checks[f'sha_unchanged_{nm}'] else 'MISMATCH'}")
    if not all(checks.values()):
        raise SystemExit("FROZEN_PREDICTION_MISMATCH_ABORT")

    meta = pd.read_csv(files["B_EMBEDDING_ONLY"][0])
    y = meta.true_label.values
    n = len(y)
    print(f"  paired N = {n}   classes {np.bincount(y).tolist()}")
    P = {nm: pd.read_csv(p).set_index("image_id").loc[meta.image_id, PCOLS].values
         for nm, (p, _m) in files.items()}
    checks["all_probabilities_finite"] = bool(all(np.isfinite(v).all() for v in P.values()))
    checks["all_images_aligned"] = all(len(v) == n for v in P.values())
    if not all(checks.values()):
        raise SystemExit("ALIGNMENT_FAILED")

    print("=== 1. POINT ESTIMATES")
    M = {nm: metrics(y, v) for nm, v in P.items()}
    pairs = [("D_VESSEL_MAP", "A_PRIMARY"), ("E_RGB_VESSEL_FEATURE_FUSION", "B_EMBEDDING_ONLY"),
             ("E_RGB_VESSEL_FEATURE_FUSION", "C_PRIMARY"),
             ("G_RGB_VESSEL_SCALAR_FUSION", "E_RGB_VESSEL_FEATURE_FUSION"),
             ("G_RGB_VESSEL_SCALAR_FUSION", "B_EMBEDDING_ONLY")]
    prows = []
    for lhs, rhs in pairs:
        for k in ALLKEYS:
            a, b = M[lhs][k], M[rhs][k]
            prows.append({"comparison": f"{lhs} vs {rhs}", "lhs": lhs, "rhs": rhs, "metric": k,
                          "lhs_value": a, "rhs_value": b, "delta_lhs_minus_rhs": a - b})
    PT = pd.DataFrame(prows)
    PT.to_csv(OUT / "point_estimates.csv", index=False)
    for lhs, rhs in pairs:
        s = PT[(PT.lhs == lhs) & (PT.rhs == rhs) & (PT.metric.isin(KEYS))]
        print(f"  {lhs} vs {rhs}: " + "  ".join(
            f"{r.metric} {r.delta_lhs_minus_rhs:+.6f}" for _, r in s.iterrows()))

    print(f"=== 2. PAIRED STRATIFIED BOOTSTRAP  seed={SEED}  B={NB}  N={n}")
    Be, Ge = P["B_EMBEDDING_ONLY"], P["E_RGB_VESSEL_FEATURE_FUSION"]
    Gg = P["G_RGB_VESSEL_SCALAR_FUSION"]
    obs_eb = {k: M["E_RGB_VESSEL_FEATURE_FUSION"][k] - M["B_EMBEDDING_ONLY"][k] for k in ALLKEYS}
    obs_ge = {k: M["G_RGB_VESSEL_SCALAR_FUSION"][k] - M["E_RGB_VESSEL_FEATURE_FUSION"][k]
              for k in ALLKEYS}
    R1 = run_bootstrap("E_minus_B", y, Be, Ge, obs_eb)
    print("  E - B done")
    R2 = run_bootstrap("G_minus_E", y, Ge, Gg, obs_ge)
    print("  G - E done")
    R = pd.concat([R1, R2], ignore_index=True)
    R.to_csv(OUT / "paired_metrics_10k.csv", index=False)
    for tag in ("E_minus_B", "G_minus_E"):
        print(f"  --- {tag} ---")
        print(R[(R.comparison == tag) & (R.metric.isin(KEYS))][
            ["metric", "delta", "ci95_lo", "ci95_hi", "p_two_sided_null_centered", "ci_crosses_zero"]
        ].to_string(index=False))

    summary = {
        "TASK8B_STATUS": "COMPLETE",
        "NO_MODEL_RETRAINING": True,
        "TEST_PREDICTIONS_MODIFIED": False,
        "paired_n": int(n),
        "bootstrap_replicates": NB,
        "seed": SEED,
        "protocol": "identical to Task 7: paired, class-stratified, seed 42, chunked fork pool, "
                    "percentile 2.5/97.5 CI, null-centered two-sided p",
        "input_sha256": sha_now,
        "input_checks": checks,
        "observed_metrics": M,
        "point_estimates": PT.to_dict("records"),
        "paired_metrics": R.to_dict("records"),
        "classification": "SECONDARY_POST_PRIMARY_EXPLORATORY_CANONICAL_SPLIT_REANALYSIS",
        "model_selection_used_test": False,
        "provenance_note": "the original 2000-replicate Task-8 paired_E_vs_B.csv is preserved "
                           "unchanged in the parent directory; this file is additional, not a "
                           "replacement",
    }
    (OUT / "statistical_closure_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    shas = {p.name: sha(p) for p in sorted(OUT.glob("*")) if p.is_file()}
    (OUT / "artifact_sha256.json").write_text(json.dumps(shas, indent=2))
    print("TASK8B_STATUS = COMPLETE")


if __name__ == "__main__":
    main()
