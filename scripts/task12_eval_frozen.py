"""Task 12 — frozen held-out evaluation pass.

The first Task-12 run crashed at held-out evaluation time on a pure column-naming bug
(`pr[f"{kind}_{c}"]` used a stale loop variable `c` instead of a class name). No model, no
hyperparameter and no selection outcome depends on that line: all six K0/K1 checkpoints had already
been selected from source-validation only and their SHA-256 values were already written to
`task12_selection_frozen.json`, and `HELDOUT_TARGETS_TOUCHED` was still `NO` at the moment of the
freeze.

So this script does NOT retrain. It verifies the six frozen checkpoint hashes against the freeze
manifest and then performs only the held-out evaluation, the paired statistics and the figures.
"""
import hashlib
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

WS = Path("/root/niki_rop_task6_isolated")
ART6 = WS / "artifacts/task6"
ART8 = WS / "artifacts/task8_spatial_vessel_fusion"
ART11 = WS / "artifacts/task11_source_heldout"
OUT = WS / "artifacts/task12_class_conditional_domain_generalization"
MANIFEST = WS / "primary_complete_case_v2_server2.csv"
RGB_EMB = ART6 / "embeddings_b5_8862.parquet"
VES_EMB = ART8 / "vessel_embeddings_b4_8862.parquet"
NAMES = ["Normal", "Pre_Plus", "Plus"]
SEED = 42
NB = 10000
NCHUNK = 40
FOLDS = [("plus", ["farfum_rop", "farabi"]),
         ("farfum_rop", ["plus", "farabi"]),
         ("farabi", ["plus", "farfum_rop"])]
KEYS3 = ["multiclass_auc", "balanced_accuracy", "macro_f1", "brier_multiclass_3col", "ece"]
KEYS2 = ["restricted_binary_auc", "balanced_accuracy", "macro_f1", "brier_restricted_2class", "ece"]
T0 = time.time()


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] (eval) {msg}"
    print(line, flush=True)
    with open(OUT / "task12_progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


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


def metrics(y, P):
    present = sorted(int(v) for v in np.unique(y))
    pred = P.argmax(1)
    r = {"n": int(len(y)), "classes_present": present,
         "multiclass_auc": multiclass_auc(y, P), "macro_ovr_auc": multiclass_auc(y, P),
         "balanced_accuracy": float(np.mean([(pred[y == k] == k).mean() for k in present])),
         "macro_f1": float(f1_score(y, pred, labels=present, average="macro")),
         "brier_multiclass_3col": float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1))),
         "ece": ece15(P, y), "auc_Normal": per_class_auc(y, P, 0),
         "auc_Pre_Plus": per_class_auc(y, P, 1), "auc_Plus": per_class_auc(y, P, 2)}
    if len(present) == 2:
        a, b = present
        r["restricted_binary_auc"] = float(roc_auc_score((y == b).astype(int), P[:, b]))
        r["restricted_pair"] = f"{NAMES[a]}_vs_{NAMES[b]}"
        P2 = P[:, present] / np.clip(P[:, present].sum(1, keepdims=True), 1e-12, None)
        oh2 = np.eye(2)[[present.index(v) for v in y]]
        r["brier_restricted_2class"] = float(np.mean(np.sum((P2 - oh2) ** 2, axis=1)))
    else:
        r["restricted_binary_auc"] = float("nan"); r["restricted_pair"] = ""
        r["brier_restricted_2class"] = float("nan")
    return r


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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 256), nn.GELU(),
                                 nn.Dropout(0.40))
        self.ves = nn.Sequential(nn.LayerNorm(1792), nn.Linear(1792, 256), nn.GELU(),
                                 nn.Dropout(0.40))
        self.shared = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.50))
        self.cls = nn.Linear(256, 3)
        self.dom = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.30),
                                 nn.Linear(128, 2))

    def forward(self, x):
        z = self.shared(torch.cat([self.rgb(x[:, :2048]), self.ves(x[:, 2048:])], 1))
        return z, self.cls(z)


def infer(net, X, dev, bs=1024):
    net.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            _z, logits = net(torch.from_numpy(X[i:i + bs]).to(dev))
            P.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(P)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {dev}  frozen-checkpoint evaluation pass, no retraining")
    fr = json.loads((OUT / "task12_selection_frozen.json").read_text())
    log("=== verify the six frozen checkpoints against the freeze manifest")
    ok = True
    for held, _ in FOLDS:
        for kind in ("K0", "K1"):
            p = OUT / f"{kind}_{held}_selected.pth"
            want = fr["folds"][held][f"{kind}_checkpoint_sha256"]
            got = sha(p)
            good = got == want
            ok &= good
            log(f"    {kind} {held:11s} {'MATCH' if good else 'MISMATCH'} {got[:16]}")
    if not ok:
        raise SystemExit("FROZEN_CHECKPOINT_MISMATCH_ABORT")
    log(f"    TASK12_ALL_FOLDS_SELECTION_FROZEN = "
        f"{fr['TASK12_ALL_FOLDS_SELECTION_FROZEN']}  HELDOUT_TARGETS_TOUCHED(before) = "
        f"{fr['HELDOUT_TARGETS_TOUCHED']}")

    m = pd.read_csv(MANIFEST, low_memory=False)
    RE = pd.read_parquet(RGB_EMB); VE = pd.read_parquet(VES_EMB)
    ec = [c for c in RE.columns if c.startswith("embedding_")]
    vc = [c for c in VE.columns if c.startswith("vessel_emb_")]
    ids = m.image_path.values
    X = np.hstack([RE.set_index("image_id").loc[ids, ec].values,
                   VE.set_index("image_id").loc[ids, vc].values]).astype(np.float32)
    y_all = m.label.values.astype(np.int64)
    src = m.source.values

    rows, preds, boot = [], {}, []
    log("=== held-out evaluation (single pass, all three folds)")
    for held, _ in FOLDS:
        tem = src == held
        yte = y_all[tem]
        for kind in ("K0", "K1"):
            net = Net().to(dev)
            net.load_state_dict(torch.load(OUT / f"{kind}_{held}_selected.pth",
                                           map_location="cpu")["model"])
            P = infer(net, X[tem], dev)
            preds[(kind, held)] = P
            r = {**metrics(yte, P), "held_out_source": held, "model": kind, "n": int(tem.sum())}
            rows.append(r)
            np.save(OUT / f"heldout_probs_{kind}_{held}.npy", P)
            lab = sorted(int(v) for v in np.unique(yte))
            pd.DataFrame(confusion_matrix(yte, P.argmax(1), labels=lab)).to_csv(
                OUT / f"confusion_{kind}_{held}.csv", index=False)
            log(f"    {kind} {held:11s} auc3 {r['multiclass_auc']:.5f} "
                f"binAUC {r['restricted_binary_auc']:.5f} bal {r['balanced_accuracy']:.5f} "
                f"f1 {r['macro_f1']:.5f} brier {r['brier_multiclass_3col']:.5f} ece {r['ece']:.5f} "
                f"aucN {r['auc_Normal']:.5f} aucPre {r['auc_Pre_Plus']:.5f} aucPlus {r['auc_Plus']:.5f}")
        e = pd.read_csv(ART11 / f"predictions_{held}.csv")
        PE = e[["E_LOSO_prob_Normal", "E_LOSO_prob_Pre_Plus", "E_LOSO_prob_Plus"]].values
        preds[("E11", held)] = PE
        rows.append({**metrics(yte, PE), "held_out_source": held, "model": "TASK11_E_LOSO",
                     "n": int(tem.sum())})
        log(f"    T11 {held:11s} auc3 {rows[-1]['multiclass_auc']:.5f} "
            f"binAUC {rows[-1]['restricted_binary_auc']:.5f}")
        pr = pd.DataFrame({"image_id": ids[tem], "source": src[tem], "true_label": yte})
        for kind in ("K0", "K1"):
            P = preds[(kind, held)]
            pr[f"{kind}_prob_Normal"] = P[:, 0]
            pr[f"{kind}_prob_Pre_Plus"] = P[:, 1]
            pr[f"{kind}_prob_Plus"] = P[:, 2]
            pr[f"{kind}_pred"] = P.argmax(1)
        pr.to_csv(OUT / f"heldout_predictions_{held}.csv", index=False)
        keys = KEYS3 if len(np.unique(yte)) == 3 else KEYS2
        boot.append(run_bootstrap(f"K1_minus_K0_{held}", yte, preds[("K1", held)],
                                  preds[("K0", held)], keys))
        boot.append(run_bootstrap(f"K1_minus_Task11E_{held}", yte, preds[("K1", held)], PE, keys))
        log(f"    {held} paired bootstraps done ({keys})")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "heldout_metrics.csv", index=False)
    BS = pd.concat(boot, ignore_index=True)
    BS.to_csv(OUT / "paired_bootstrap_10k.csv", index=False)
    for tag in BS.comparison.unique():
        BS[BS.comparison == tag].to_csv(OUT / f"paired_{tag}.csv", index=False)
    log("=== paired results")
    for _, r in BS[BS.metric.isin(KEYS3 + KEYS2)].iterrows():
        log(f"    {r.comparison:26s} {r.metric:24s} d={r.delta:+.6f} "
            f"[{r.ci95_lo:+.6f},{r.ci95_hi:+.6f}] p={r.p_two_sided_null_centered:.4f}")

    log("=== figures")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(13, 6.4))
        for j, (held, _) in enumerate(FOLDS):
            for kind, col in (("K0", "tab:blue"), ("K1", "tab:red")):
                h = pd.DataFrame(json.loads((OUT / f"{kind}_{held}_history.json").read_text()))
                ax[0][j].plot(h.epoch, h["class"], "-", color=col, label=f"{kind} class")
                if kind == "K1":
                    ax[0][j].plot(h.epoch, h.domain, "--", color=col, label="K1 domain")
                    ax[0][j].plot(h.epoch, h.mmd, ":", color=col, label="K1 mmd")
                    ax[0][j].plot(h.epoch, h.alpha, "-.", color="gray", label="alpha")
                    ax[0][j].set_yscale("symlog", linthresh=1e-3)
                ax[1][j].plot(h.epoch, h.val_auc, "-o", ms=2.5, color=col, label=f"{kind} val AUC")
            ax[0][j].set_title(f"losses, hold out {held}", fontsize=8); ax[0][j].legend(fontsize=5)
            ax[1][j].set_title(f"source-val AUC, hold out {held}", fontsize=8)
            ax[1][j].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(OUT / "loss_and_selection_curves.png", dpi=170); plt.close(fig)
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        for j, (held, _) in enumerate(FOLDS):
            yte = y_all[src == held]
            for nm, P in (("K0", preds[("K0", held)]), ("K1", preds[("K1", held)]),
                          ("Task11 E", preds[("E11", held)])):
                conf, corr, xs, ys_ = P.max(1), (P.argmax(1) == yte), [], []
                for lo in np.linspace(0, 1, 16)[:-1]:
                    mm = (conf >= lo) & (conf < lo + 1 / 15)
                    if mm.sum():
                        xs.append(conf[mm].mean()); ys_.append(corr[mm].mean())
                ax[j].plot(xs, ys_, marker="o", ms=3, label=nm)
            ax[j].plot([0, 1], [0, 1], "k--", lw=0.7)
            ax[j].set_title(f"hold out {held}", fontsize=8); ax[j].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(OUT / "calibration_heldout.png", dpi=170); plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        DP = pd.read_csv(OUT / "domain_predictability.csv")
        for i, x in enumerate(("K0", "K1")):
            s = DP[DP.kind == x]
            ax[0].bar(np.arange(len(s)) + i * 0.38, s.domain_predictability_acc_cv5, 0.38, label=x)
        ax[0].set_xticks(np.arange(len(FOLDS)) + 0.19)
        ax[0].set_xticklabels([h for h, _ in FOLDS], fontsize=7)
        ax[0].axhline(0.5, color="k", ls="--", lw=0.7); ax[0].legend(fontsize=6)
        ax[0].set_title("source-val domain predictability (5-fold CV)", fontsize=8)
        SD = pd.read_csv(OUT / "representation_shift.csv")
        for i, x in enumerate(("K0", "K1")):
            s = SD[SD.kind == x]
            ax[1].bar(np.arange(len(s)) + i * 0.38, s.median_abs_smd, 0.38, label=x)
        ax[1].set_xticks(np.arange(len(FOLDS)) + 0.19)
        ax[1].set_xticklabels([h for h, _ in FOLDS], fontsize=7); ax[1].legend(fontsize=6)
        ax[1].set_title("source-val representation median |SMD|", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "domain_invariance_diagnostics.png", dpi=170)
        plt.close(fig)
        log("    figures written")
    except Exception as ex:  # noqa: BLE001
        log(f"    figures skipped: {ex}")

    prev = json.loads((OUT / "task12_summary.json").read_text()) if \
        (OUT / "task12_summary.json").exists() else {}
    summary = {"TASK12_STATUS": "COMPLETE",
               "classification": "SECONDARY_SOURCE_GENERALIZATION_DOMAIN_INVARIANCE_EXPERIMENT",
               "fold_definitions": json.loads((OUT / "fold_definitions.json").read_text()),
               "selection_frozen": fr, "heldout_metrics": R.to_dict("records"),
               "paired_bootstrap": BS.to_dict("records"),
               "domain_predictability": pd.read_csv(OUT / "domain_predictability.csv").to_dict("records"),
               "representation_shift": pd.read_csv(OUT / "representation_shift.csv").to_dict("records"),
               "heldout_source_used_during_training": False, "cnn_retrained": False,
               "biomarkers_used": False,
               "note": "the first Task-12 run crashed at held-out evaluation on a stale loop variable "
                       "used as a column name; the six checkpoints were already selected from "
                       "source-validation only and frozen, so this pass verifies their hashes and "
                       "performs evaluation only - no retraining, no reselection"}
    (OUT / "task12_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    shas = {p.name: sha(p) for p in sorted(OUT.glob("*"))
            if p.is_file() and p.name not in ("task12_progress.log", "train.log")}
    shas["task12_progress.log"] = "EXCLUDED_APPEND_ONLY_LOG"
    (OUT / "artifact_sha256.json").write_text(json.dumps(shas, indent=2))
    log("TASK12_STATUS = COMPLETE")


if __name__ == "__main__":
    main()
