"""Task 11 — source-held-out (LOSO) robustness baseline. Diagnostic only, no domain adaptation.

Three folds, each holding out one acquisition source entirely:

  hold out plus        train on farfum_rop + farabi
  hold out farfum_rop  train on plus + farabi
  hold out farabi      train on plus + farfum_rop

Frozen feature formulations only (no CNN is executed and no embedding is regenerated):
  B_LOSO = 2048-d RGB embedding
  E_LOSO = RGB + 1792-d vessel embedding           (3840)
  G_LOSO = RGB + vessel + five FINAL_PRIMARY scalars (3845)

Every fold uses the SAME frozen Task-6/8 downstream XGBoost configuration for all three models; no
per-source tuning of any kind. No preprocessing statistic is fitted on the held-out source. The
held-out source is never inspected before its metrics are computed.
"""
import hashlib
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

WS = Path("/root/niki_rop_task6_isolated")
ART6 = WS / "artifacts/task6"
ART8 = WS / "artifacts/task8_spatial_vessel_fusion"
OUT = WS / "artifacts/task11_source_heldout"
MANIFEST = WS / "primary_complete_case_v2_server2.csv"
RGB_EMB = ART6 / "embeddings_b5_8862.parquet"
VES_EMB = ART8 / "vessel_embeddings_b4_8862.parquet"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
WEIGHTS = {0: 0.46051, 1: 3.18103, 2: 1.94512}
NAMES = ["Normal", "Pre_Plus", "Plus"]
PCOLS = ["prob_Normal", "prob_Pre_Plus", "prob_Plus"]
SEED = 42
NB = 10000
NCHUNK = 40
FOLDS = [("plus", ["farfum_rop", "farabi"]),
         ("farfum_rop", ["plus", "farabi"]),
         ("farabi", ["plus", "farfum_rop"])]
CANON = {"B_LOSO": 0.924903, "E_LOSO": 0.932841, "G_LOSO": 0.934880}
T0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "task11_progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


# ---------------------------------------------------------------- metrics
def multiclass_auc(y, P):
    present = sorted(int(v) for v in np.unique(y))
    if len(present) < 3:
        return float("nan")
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=present))


def per_class_auc(y, P, k):
    yk = (y == k).astype(int)
    return float(roc_auc_score(yk, P[:, k])) if 0 < yk.sum() < len(yk) else float("nan")


def ece15(P, y):
    pred, conf, corr = P.argmax(1), P.max(1), (P.argmax(1) == y)
    e = 0.0
    for lo in np.linspace(0, 1, 16)[:-1]:
        m = (conf >= lo) & (conf < lo + 1 / 15)
        if m.sum():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def bal_acc_present(y, pred):
    present = sorted(int(v) for v in np.unique(y))
    return float(np.mean([(pred[y == k] == k).mean() for k in present]))


def metrics(y, P):
    """Standard 3-class metrics plus explicitly labelled restricted variants.

    multiclass_auc is NaN whenever fewer than three classes are present - it is never replaced by a
    two-class substitute. balanced_accuracy / macro_f1 are computed over the classes actually present
    in y, which coincides with the usual definition when all three are present.
    """
    present = sorted(int(v) for v in np.unique(y))
    pred = P.argmax(1)
    r = {"n": int(len(y)), "classes_present": present,
         "multiclass_auc": multiclass_auc(y, P), "macro_ovr_auc": multiclass_auc(y, P),
         "balanced_accuracy": bal_acc_present(y, pred),
         "macro_f1": float(f1_score(y, pred, labels=present, average="macro")),
         "brier_multiclass_3col": float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1))),
         "ece": ece15(P, y),
         "auc_Normal": per_class_auc(y, P, 0), "auc_Pre_Plus": per_class_auc(y, P, 1),
         "auc_Plus": per_class_auc(y, P, 2)}
    if len(present) == 2:
        a, b = present
        yb = (y == b).astype(int)
        r["restricted_binary_auc"] = float(roc_auc_score(yb, P[:, b]))
        r["restricted_pair"] = f"{NAMES[a]}_vs_{NAMES[b]}"
        P2 = P[:, present] / np.clip(P[:, present].sum(1, keepdims=True), 1e-12, None)
        oh2 = np.eye(2)[[present.index(v) for v in y]]
        r["brier_restricted_2class"] = float(np.mean(np.sum((P2 - oh2) ** 2, axis=1)))
    else:
        r["restricted_binary_auc"] = float("nan")
        r["restricted_pair"] = ""
        r["brier_restricted_2class"] = float("nan")
    return r


KEYS3 = ["multiclass_auc", "balanced_accuracy", "macro_f1", "brier_multiclass_3col", "ece"]
KEYS2 = ["restricted_binary_auc", "balanced_accuracy", "macro_f1", "brier_restricted_2class", "ece"]


def strat_idx(y, rng):
    return np.concatenate([rng.choice(np.nonzero(y == k)[0], size=int((y == k).sum()),
                                      replace=True) for k in np.unique(y)])


def boot_chunk(arg):
    reps, y, PA, PB, keys = arg
    rows = []
    for r in reps:
        rng = np.random.default_rng(SEED * 1000003 + r)
        i = strat_idx(y, rng)
        ma, mb = metrics(y[i], PA[i]), metrics(y[i], PB[i])
        rows.append([ma[k] - mb[k] for k in keys])
    return rows


def run_bootstrap(tag, y, PA, PB, keys):
    obs = {k: metrics(y, PA)[k] - metrics(y, PB)[k] for k in keys}
    chunks = [list(range(i, min(i + NCHUNK, NB))) for i in range(0, NB, NCHUNK)]
    with get_context("fork").Pool(24) as pool:
        res = pool.map(boot_chunk, [(c, y, PA, PB, keys) for c in chunks])
    D = np.array([row for ch in res for row in ch])
    pd.DataFrame(D, columns=keys).to_parquet(OUT / f"bootstrap_{tag}.parquet", index=False)
    return pd.DataFrame([{"comparison": tag, "metric": k, "delta": obs[k],
                          "ci95_lo": float(np.percentile(D[:, j], 2.5)),
                          "ci95_hi": float(np.percentile(D[:, j], 97.5)),
                          "p_two_sided_null_centered":
                              float(np.mean(np.abs(D[:, j] - obs[k]) >= abs(obs[k]))),
                          "ci_crosses_zero": bool(np.percentile(D[:, j], 2.5) <= 0
                                                  <= np.percentile(D[:, j], 97.5)),
                          "replicates": NB, "seed": SEED}
                         for j, k in enumerate(keys)])


# ---------------------------------------------------------------- shift diagnostics
def shift_block(tag, Xtr, Xte):
    mu_tr, sd_tr, mu_te = Xtr.mean(0), Xtr.std(0, ddof=0), Xte.mean(0)
    sd_safe = np.where(sd_tr > 0, sd_tr, 1.0)
    smd = (mu_te - mu_tr) / sd_safe
    a = np.abs(smd)
    return {"block": tag, "dim": int(Xtr.shape[1]),
            "train_n": int(Xtr.shape[0]), "heldout_n": int(Xte.shape[0]),
            "centroid_distance_standardized": float(np.sqrt(np.sum(smd ** 2))),
            "centroid_distance_raw_l2": float(np.linalg.norm(mu_te - mu_tr)),
            "median_abs_smd": float(np.median(a)), "mean_abs_smd": float(a.mean()),
            "p90_abs_smd": float(np.percentile(a, 90)), "p95_abs_smd": float(np.percentile(a, 95)),
            "max_abs_smd": float(a.max()),
            "frac_features_abs_smd_gt_0p5": float((a > 0.5).mean()),
            "frac_features_abs_smd_gt_1p0": float((a > 1.0).mean())}


def main():
    np.random.seed(SEED)
    log("=== 0. PRE-FLIGHT")
    t6 = json.loads((ART6 / "artifact_sha256.json").read_text())
    t8 = json.loads((ART8 / "artifact_sha256.json").read_text())
    params = json.loads((ART6 / "downstream_xgb_selected.json").read_text())["params"]
    log(f"    frozen downstream XGBoost config {params}")
    checks = {"rgb_emb_sha_unchanged": sha(RGB_EMB) == t6.get(RGB_EMB.name),
              "vessel_emb_sha_unchanged": sha(VES_EMB) == t8.get(VES_EMB.name)}
    m = pd.read_csv(MANIFEST, low_memory=False)
    RE = pd.read_parquet(RGB_EMB); VE = pd.read_parquet(VES_EMB)
    ec = [c for c in RE.columns if c.startswith("embedding_")]
    vc = [c for c in VE.columns if c.startswith("vessel_emb_")]
    ids = m.image_path.values
    XB = RE.set_index("image_id").loc[ids, ec].values.astype(np.float32)
    XV = VE.set_index("image_id").loc[ids, vc].values.astype(np.float32)
    Bm = m[FEATS].values.astype(np.float32)
    checks["population_8862"] = len(m) == 8862
    checks["rgb_shape"] = XB.shape == (8862, 2048)
    checks["vessel_shape"] = XV.shape == (8862, 1792)
    checks["biomarkers_finite"] = bool(np.isfinite(Bm).all())
    checks["three_sources"] = set(m.source.unique()) == {"plus", "farfum_rop", "farabi"}
    for k, v in checks.items():
        log(f"    {k:26s} : {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit("PREFLIGHT_FAILED")
    y_all = m.label.values
    src = m.source.values

    cov = m.groupby("source").agg(n=("label", "size"), groups=("group_id", "nunique"),
                                  normal=("label", lambda s: int((s == 0).sum())),
                                  pre_plus=("label", lambda s: int((s == 1).sum())),
                                  plus_cls=("label", lambda s: int((s == 2).sum())))
    cov.to_csv(OUT / "population_audit.csv")
    log("    population audit"); log(cov.to_string())

    fold_defs = [{"held_out": h, "train_sources": t,
                  "n_train": int(np.isin(src, t).sum()), "n_heldout": int((src == h).sum()),
                  "heldout_classes": sorted(int(v) for v in np.unique(y_all[src == h]))}
                 for h, t in FOLDS]
    (OUT / "fold_definitions.json").write_text(json.dumps(fold_defs, indent=2))
    for f in fold_defs:
        log(f"    fold holdout={f['held_out']:11s} train={f['train_sources']} "
            f"n_train={f['n_train']} n_test={f['n_heldout']} classes={f['heldout_classes']}")

    blocks = {"B_LOSO": XB, "E_LOSO": np.hstack([XB, XV]),
              "G_LOSO": np.hstack([XB, XV, Bm])}
    for k, v in blocks.items():
        log(f"    {k} dim {v.shape[1]}")

    all_rows, all_pred, shift_rows, boot_frames = [], {}, [], []
    for held, trainy in FOLDS:
        trm, tem = np.isin(src, trainy), src == held
        log(f"=== FOLD hold out {held}  (train n {int(trm.sum())}, test n {int(tem.sum())})")
        if set(src[trm]) & {held}:
            raise SystemExit("TARGET_SOURCE_LEAKED_INTO_TRAINING")
        ytr, yte = y_all[trm], y_all[tem]
        ctr = sorted(int(v) for v in np.unique(ytr))
        log(f"    train class counts {np.bincount(ytr).tolist()}  test class counts "
            f"{np.bincount(yte).tolist()}")
        if ctr != [0, 1, 2]:
            raise SystemExit(f"TRAIN_FOLD_MISSING_CLASSES {ctr}")
        wtr = pd.Series(ytr).map(WEIGHTS).values
        for nm, X in blocks.items():
            import xgboost as xgb
            clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3,
                                    eval_metric="mlogloss", random_state=SEED, n_jobs=16,
                                    tree_method="hist", **params)
            clf.fit(X[trm], ytr, sample_weight=wtr)
            P = clf.predict_proba(X[tem])
            clf.save_model(str(OUT / f"{nm}_{held}.json"))
            all_pred[(held, nm)] = P
            r = {**metrics(yte, P), "held_out_source": held, "model": nm,
                 "train_sources": "+".join(trainy), "n_train": int(trm.sum())}
            all_rows.append(r)
            log(f"    {nm:7s} auc3 {r['multiclass_auc']:.5f} bin {r['restricted_binary_auc']:.5f} "
                f"bal {r['balanced_accuracy']:.5f} f1 {r['macro_f1']:.5f} "
                f"brier {r['brier_multiclass_3col']:.5f} ece {r['ece']:.5f}")
            np.save(OUT / f"probs_{nm}_{held}.npy", P)
            pd.DataFrame(confusion_matrix(yte, P.argmax(1),
                                          labels=sorted(int(v) for v in np.unique(yte)))).to_csv(
                OUT / f"confusion_{nm}_{held}.csv", index=False)
        # shift diagnostics (no labels used)
        for tag, X in (("rgb_embedding", XB), ("vessel_embedding", XV)):
            shift_rows.append({"held_out_source": held, **shift_block(tag, X[trm], X[tem])})
        te_df = pd.DataFrame({"held_out_source": held, "image_id": ids[tem], "true_label": yte})
        for nm in blocks:
            P = all_pred[(held, nm)]
            te_df[f"{nm}_prob_Normal"] = P[:, 0]
            te_df[f"{nm}_prob_Pre_Plus"] = P[:, 1]
            te_df[f"{nm}_prob_Plus"] = P[:, 2]
            te_df[f"{nm}_pred"] = P.argmax(1)
        te_df.to_csv(OUT / f"predictions_{held}.csv", index=False)

        keys = KEYS3 if len(np.unique(yte)) == 3 else KEYS2
        log(f"    paired bootstrap 10k  metrics {keys}")
        boot_frames.append(run_bootstrap(f"E_minus_B_{held}", yte, all_pred[(held, "E_LOSO")],
                                         all_pred[(held, "B_LOSO")], keys))
        boot_frames.append(run_bootstrap(f"G_minus_E_{held}", yte, all_pred[(held, "G_LOSO")],
                                         all_pred[(held, "E_LOSO")], keys))

    R = pd.DataFrame(all_rows)
    R.to_csv(OUT / "metrics_per_source.csv", index=False)
    BS = pd.concat(boot_frames, ignore_index=True)
    BS.to_csv(OUT / "paired_bootstrap_10k.csv", index=False)
    for tag in BS.comparison.unique():
        BS[BS.comparison == tag].to_csv(OUT / f"paired_{tag}.csv", index=False)
    SH = pd.DataFrame(shift_rows)
    SH.to_csv(OUT / "domain_shift_diagnostics.csv", index=False)

    log("=== biomarker shift")
    bm_rows = []
    for s in ("plus", "farfum_rop", "farabi"):
        sel = src == s
        for f in FEATS:
            v = m.loc[sel, f].values
            bm_rows.append({"source": s, "biomarker": f, "n": int(sel.sum()),
                            "mean": float(v.mean()), "std": float(v.std(ddof=0)),
                            "median": float(np.median(v)),
                            "iqr": float(np.percentile(v, 75) - np.percentile(v, 25))})
    for held, trainy in FOLDS:
        trm, tem = np.isin(src, trainy), src == held
        for f in FEATS:
            a, b = m.loc[trm, f].values, m.loc[tem, f].values
            bm_rows.append({"source": f"HELDOUT::{held}", "biomarker": f, "n": int(tem.sum()),
                            "mean": float(b.mean()), "std": float(b.std(ddof=0)),
                            "median": float(np.median(b)),
                            "iqr": float(np.percentile(b, 75) - np.percentile(b, 25)),
                            "smd_vs_train": float((b.mean() - a.mean()) / (a.std(ddof=0) or 1.0))})
    BM = pd.DataFrame(bm_rows)
    BM.to_csv(OUT / "biomarker_shift.csv", index=False)

    log("=== domain-shift summary table")
    summ = []
    for held, _ in FOLDS:
        row = {"held_out_source": held}
        for nm in blocks:
            r = R[(R.held_out_source == held) & (R.model == nm)].iloc[0]
            row[f"{nm}_auc3"] = r.multiclass_auc
            row[f"{nm}_restricted_binary_auc"] = r.restricted_binary_auc
            row[f"{nm}_balanced_accuracy"] = r.balanced_accuracy
            row[f"{nm}_macro_f1"] = r.macro_f1
            valid = r.multiclass_auc if np.isfinite(r.multiclass_auc) else r.restricted_binary_auc
            row[f"{nm}_auc_used"] = valid
            row[f"{nm}_degradation_vs_canonical"] = valid - CANON[nm]
        row["delta_E_minus_B_auc"] = row["E_LOSO_auc_used"] - row["B_LOSO_auc_used"]
        row["delta_G_minus_E_auc"] = row["G_LOSO_auc_used"] - row["E_LOSO_auc_used"]
        summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "domain_shift_summary.csv", index=False)
    log(S[["held_out_source", "B_LOSO_auc_used", "E_LOSO_auc_used", "G_LOSO_auc_used",
           "delta_E_minus_B_auc", "delta_G_minus_E_auc"]].to_string(index=False))
    log(S[["held_out_source", "B_LOSO_degradation_vs_canonical",
           "E_LOSO_degradation_vs_canonical", "G_LOSO_degradation_vs_canonical"]].to_string(
        index=False))
    log("    RGB shift"); log(SH[SH.block == "rgb_embedding"][
        ["held_out_source", "centroid_distance_standardized", "median_abs_smd", "p90_abs_smd",
         "p95_abs_smd"]].to_string(index=False))
    log("    vessel shift"); log(SH[SH.block == "vessel_embedding"][
        ["held_out_source", "centroid_distance_standardized", "median_abs_smd", "p90_abs_smd",
         "p95_abs_smd"]].to_string(index=False))

    log("=== figures")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        for j, (held, _) in enumerate(FOLDS):
            yte = y_all[src == held]
            for nm in blocks:
                P = all_pred[(held, nm)]
                conf, corr, xs, ys_ = P.max(1), (P.argmax(1) == yte), [], []
                for lo in np.linspace(0, 1, 16)[:-1]:
                    mm = (conf >= lo) & (conf < lo + 1 / 15)
                    if mm.sum():
                        xs.append(conf[mm].mean()); ys_.append(corr[mm].mean())
                ax[j].plot(xs, ys_, marker="o", ms=3, label=nm)
            ax[j].plot([0, 1], [0, 1], "k--", lw=0.7)
            ax[j].set_title(f"hold out {held}", fontsize=8); ax[j].legend(fontsize=6)
            ax[j].set_xlabel("confidence", fontsize=7); ax[j].set_ylabel("accuracy", fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / "calibration_by_heldout_source.png", dpi=170)
        plt.close(fig)
        fig, ax = plt.subplots(len(FOLDS), 3, figsize=(8, 7))
        for i, (held, _) in enumerate(FOLDS):
            yte = y_all[src == held]
            present = sorted(int(v) for v in np.unique(yte))
            for j, nm in enumerate(blocks):
                cm = confusion_matrix(yte, all_pred[(held, nm)].argmax(1), labels=present)
                ax[i][j].imshow(cm, cmap="Blues")
                for a in range(len(present)):
                    for b in range(len(present)):
                        ax[i][j].text(b, a, cm[a, b], ha="center", va="center", fontsize=7)
                ax[i][j].set_xticks(range(len(present)))
                ax[i][j].set_xticklabels([NAMES[k] for k in present], fontsize=6)
                ax[i][j].set_yticks(range(len(present)))
                ax[i][j].set_yticklabels([NAMES[k] for k in present], fontsize=6)
                ax[i][j].set_title(f"{held} / {nm}", fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=170); plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for blk, a in (("rgb_embedding", ax[0]), ("vessel_embedding", ax[1])):
            s = SH[SH.block == blk]
            a.bar(range(len(s)), s.p95_abs_smd, color="tab:blue", alpha=0.7, label="p95 |SMD|")
            a.plot(range(len(s)), s.median_abs_smd, "ko-", ms=4, label="median |SMD|")
            a.set_xticks(range(len(s))); a.set_xticklabels(s.held_out_source, fontsize=7)
            a.set_title(f"{blk} shift vs training sources", fontsize=8); a.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(OUT / "domain_shift.png", dpi=170); plt.close(fig)
        log("    figures written")
    except Exception as ex:  # noqa: BLE001
        log(f"    figures skipped: {ex}")

    summary = {"TASK11_STATUS": "COMPLETE",
               "classification": "SECONDARY_SOURCE_GENERALIZATION_ANALYSIS",
               "frozen_xgboost_config": params, "folds": fold_defs,
               "population_audit": cov.reset_index().to_dict("records"),
               "metrics_per_source": R.to_dict("records"),
               "domain_shift_summary": S.to_dict("records"),
               "paired_bootstrap": BS.to_dict("records"),
               "domain_shift_diagnostics": SH.to_dict("records"),
               "biomarker_shift": bm_rows,
               "canonical_mixed_source_reference": CANON,
               "cnn_retrained": False, "target_source_used_for_selection": False,
               "note": "multiclass_auc is NaN for the plus fold because Pre_Plus is absent there and "
                       "was never substituted; restricted_binary_auc is a separately labelled "
                       "Normal-vs-Plus metric and is NOT the project's standard 3-class AUC"}
    (OUT / "task11_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    shas = {p.name: sha(p) for p in sorted(OUT.glob("*"))
            if p.is_file() and p.name not in ("task11_progress.log", "train.log")}
    shas["task11_progress.log"] = "EXCLUDED_APPEND_ONLY_LOG"
    (OUT / "artifact_sha256.json").write_text(json.dumps(shas, indent=2))
    log("TASK11_STATUS = COMPLETE")


if __name__ == "__main__":
    main()
