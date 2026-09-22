"""Task 10 — frozen biomarker-conditioned FiLM fusion (Server 2, authoritative).

Both CNNs are already frozen into embedding tables. This task never instantiates, loads or runs a
CNN encoder: it trains two small matched heads on the frozen Task-6 RGB embedding (8862 x 2048) and
the frozen Task-8 vessel embedding (8862 x 1792).

J0_FROZEN_FUSION_CONTROL : r = proj(rgb), v = proj(vessel), z = [r, v, r*v, |r-v|] -> fusion MLP
J1_BIOMARKER_FILM        : identical, except v is replaced by bounded residual FiLM modulation
                           gamma = 1 + 0.10*tanh(delta_gamma), beta = 0.10*tanh(beta_raw),
                           v_film = gamma*v + beta, with delta_gamma/beta_raw produced from the five
                           train-standardised biomarkers through 5->32->64.

The only scientific difference between the two models is biomarker-conditioned FiLM. Both are
selected and frozen on VAL before TEST is opened once.
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

WS = Path("/root/niki_rop_task6_isolated")
ART6 = WS / "artifacts/task6"
ART8 = WS / "artifacts/task8_spatial_vessel_fusion"
OUT = WS / "artifacts/task10_biomarker_film"
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
BATCH = 128
KEYS = ["multiclass_auc", "balanced_accuracy", "macro_f1", "brier", "ece"]
ALLKEYS = KEYS + ["auc_Normal", "auc_Pre_Plus", "auc_Plus"]
T0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "task10_progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def macro_auc(y, P):
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
    pred = P.argmax(1)
    return {"multiclass_auc": macro_auc(y, P), "macro_ovr_auc": macro_auc(y, P),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "brier": float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1))), "ece": ece15(P, y),
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
        rows.append([ma[k] - mb[k] for k in ALLKEYS])
    return rows


def run_bootstrap(tag, y, PA, PB, obs):
    chunks = [list(range(i, min(i + NCHUNK, NB))) for i in range(0, NB, NCHUNK)]
    with get_context("fork").Pool(24) as pool:
        res = pool.map(boot_chunk, [(c, y, PA, PB) for c in chunks])
    D = np.array([row for ch in res for row in ch])
    pd.DataFrame(D, columns=ALLKEYS).to_parquet(OUT / f"bootstrap_{tag}.parquet", index=False)
    return pd.DataFrame([{"comparison": tag, "metric": k, "delta": obs[k],
                          "ci95_lo": float(np.percentile(D[:, j], 2.5)),
                          "ci95_hi": float(np.percentile(D[:, j], 97.5)),
                          "p_two_sided_null_centered":
                              float(np.mean(np.abs(D[:, j] - obs[k]) >= abs(obs[k]))),
                          "ci_crosses_zero": bool(np.percentile(D[:, j], 2.5) <= 0
                                                  <= np.percentile(D[:, j], 97.5)),
                          "replicates": NB, "seed": SEED}
                         for j, k in enumerate(ALLKEYS)])


# ---------------------------------------------------------------- models
class Fusion(nn.Module):
    """Matched pair. use_film=False -> J0, use_film=True -> J1. Only FiLM differs."""

    def __init__(self, use_film):
        super().__init__()
        self.use_film = use_film
        self.rgb_proj = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 512), nn.GELU(),
                                      nn.Dropout(0.30))
        self.ves_proj = nn.Sequential(nn.LayerNorm(1792), nn.Linear(1792, 512), nn.GELU(),
                                      nn.Dropout(0.30))
        if use_film:
            self.cond = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 64), nn.GELU())
            self.to_gamma = nn.Linear(64, 512)
            self.to_beta = nn.Linear(64, 512)
            for lin in (self.to_gamma, self.to_beta):  # small weights, zero bias -> near identity
                nn.init.normal_(lin.weight, std=0.01)
                nn.init.zeros_(lin.bias)
        self.fuse = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 256), nn.GELU(),
                                  nn.Dropout(0.40), nn.Linear(256, 64), nn.GELU(),
                                  nn.Dropout(0.20), nn.Linear(64, 3))

    def modulate(self, v, b):
        c = self.cond(b)
        gamma = 1.0 + 0.10 * torch.tanh(self.to_gamma(c))
        beta = 0.10 * torch.tanh(self.to_beta(c))
        return gamma * v + beta, gamma, beta

    def forward(self, xr, xv, xb, want_mod=False):
        r = self.rgb_proj(xr)
        v = self.ves_proj(xv)
        gamma = beta = None
        if self.use_film:
            v, gamma, beta = self.modulate(v, xb)
        z = torch.cat([r, v, r * v, torch.abs(r - v)], 1)
        out = self.fuse(z)
        if want_mod:
            return out, gamma, beta
        return out


def predict(model, X, dev, bs=512):
    model.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(X[0]), bs):
            sl = slice(i, i + bs)
            o = model(X[0][sl].to(dev), X[1][sl].to(dev), X[2][sl].to(dev))
            P.append(torch.softmax(o.float(), 1).cpu().numpy())
    return np.concatenate(P)


def train_model(name, use_film, D, dev):
    log(f"=== {name}  use_film={use_film}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = Fusion(use_film).to(dev)
    fused = [(n, p.shape) for n, p in model.named_parameters()]
    n_par = sum(p.numel() for p in model.parameters())
    log(f"    parameters {n_par}  tensors {len(fused)}  any CNN tensor: "
        f"{any(n.startswith(('rgb.', 'ves.')) for n, _ in model.named_parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss(weight=D["w"])
    Xtr = (D["xtr"], D["vtr"], D["btr"]); ytr = D["ytr"]
    Xva = (D["xva"], D["vva"], D["bva"]); yva = D["yva"]
    n = len(ytr)
    hist = []
    best, best_idx = None, -1
    for ep in range(50):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n, device=dev)
        tl = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr[0][idx], Xtr[1][idx], Xtr[2][idx]), ytr[idx])
            loss.backward(); opt.step()
            tl += float(loss.item()) * len(idx)
        tl /= n
        model.eval(); vl = 0.0
        with torch.no_grad():
            for i in range(0, len(yva), 512):
                sl = slice(i, i + 512)
                o = model(Xva[0][sl], Xva[1][sl], Xva[2][sl])
                vl += float(lossf(o, yva[sl]).item()) * len(yva[sl])
        vl /= len(yva)
        P = predict(model, Xva, dev)
        yv = yva.cpu().numpy()
        auc, f1 = macro_auc(yv, P), float(f1_score(yv, P.argmax(1), average="macro"))
        ck = OUT / f"{name}_epoch{ep+1}.pth"
        torch.save({"model": model.state_dict(), "epoch": ep + 1, "use_film": use_film}, ck)
        hist.append({"epoch": ep + 1, "train_loss": tl, "val_loss": vl, "val_auc": auc,
                     "val_macro_f1": f1, "seconds": round(time.time() - t_ep, 2), "ckpt": str(ck)})
        (OUT / f"{name}_history.json").write_text(json.dumps(hist, indent=2))
        log(f"    {name} epoch {ep+1:2d} {time.time()-t_ep:5.1f}s train {tl:.4f} val {vl:.4f} "
            f"auc {auc:.5f} f1 {f1:.4f}")
        key = (auc, f1, -vl, -(ep + 1))
        bk = (best["val_auc"], best["val_macro_f1"], -best["val_loss"], -best["epoch"]) if best else None
        if bk is None or key > bk:
            best, best_idx = hist[-1], ep
        if ep - best_idx >= 8:
            log(f"    {name} early stop at epoch {ep+1} (best {best['epoch']})")
            break
    import shutil
    shutil.copy(best["ckpt"], OUT / f"{name}_selected.pth")
    sel = {"model": name, "use_film": use_film, "selected_epoch": best["epoch"],
           "val_auc": best["val_auc"], "val_macro_f1": best["val_macro_f1"],
           "val_loss": best["val_loss"], "checkpoint_sha256": sha(OUT / f"{name}_selected.pth"),
           "epochs_run": len(hist), "parameters": n_par, "history": hist, "test_touched": False}
    (OUT / f"{name}_selection.json").write_text(json.dumps(sel, indent=2))
    log(f"    {name} SELECTED epoch {best['epoch']} val_auc {best['val_auc']:.5f} "
        f"sha {sel['checkpoint_sha256'][:16]}")
    return sel


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {dev}   NOTE: no CNN is loaded or executed anywhere in this task")

    log("=== 0. PRE-FLIGHT (frozen embedding integrity)")
    t6 = json.loads((ART6 / "artifact_sha256.json").read_text())
    t8 = json.loads((ART8 / "artifact_sha256.json").read_text())
    checks = {"rgb_emb_sha_unchanged": sha(RGB_EMB) == t6.get(RGB_EMB.name),
              "vessel_emb_sha_unchanged": sha(VES_EMB) == t8.get(VES_EMB.name),
              "E_pred_sha_unchanged": sha(ART8 / "test_predictions_E_RGB_VESSEL_FEATURE_FUSION.csv")
              == t8.get("test_predictions_E_RGB_VESSEL_FEATURE_FUSION.csv"),
              "G_pred_sha_unchanged": sha(ART8 / "test_predictions_G_RGB_VESSEL_SCALAR_FUSION.csv")
              == t8.get("test_predictions_G_RGB_VESSEL_SCALAR_FUSION.csv")}
    m = pd.read_csv(MANIFEST, low_memory=False)
    checks["population_8862"] = len(m) == 8862
    checks["split_counts"] = (m.split.value_counts().get("train") == 6203
                              and m.split.value_counts().get("val") == 1328
                              and m.split.value_counts().get("test") == 1331)
    checks["biomarkers_finite"] = bool(m[FEATS].notna().all().all())
    RE = pd.read_parquet(RGB_EMB)
    VE = pd.read_parquet(VES_EMB)
    ec = [c for c in RE.columns if c.startswith("embedding_")]
    vc = [c for c in VE.columns if c.startswith("vessel_emb_")]
    checks["rgb_emb_shape"] = (RE.shape[0], len(ec)) == (8862, 2048)
    checks["vessel_emb_shape"] = (VE.shape[0], len(vc)) == (8862, 1792)
    for k, v in checks.items():
        log(f"    {k:26s} : {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit("PREFLIGHT_FAILED")

    ids = m.image_path.values
    XB = RE.set_index("image_id").loc[ids, ec].values.astype(np.float32)
    XV = VE.set_index("image_id").loc[ids, vc].values.astype(np.float32)
    log(f"    RGB {XB.shape}  vessel {XV.shape}  finite "
        f"{bool(np.isfinite(XB).all() and np.isfinite(XV).all())}")

    tr = (m.split == "train").values
    va = (m.split == "val").values
    te = (m.split == "test").values
    bmu, bsd = m.loc[tr, FEATS].mean().values, m.loc[tr, FEATS].std(ddof=0).values
    Bz = ((m[FEATS].values - bmu) / bsd).astype(np.float32)
    json.dump({"features": FEATS, "mean": bmu.tolist(), "std": bsd.tolist(),
               "fitted_on": "TRAIN only", "n_train": int(tr.sum())},
              open(OUT / "biomarker_scaler.json", "w"), indent=2)
    log(f"    scaler train-only mean {np.round(bmu,4).tolist()} std {np.round(bsd,4).tolist()}")

    tX, tV, tB = torch.from_numpy(XB).to(dev), torch.from_numpy(XV).to(dev), \
        torch.from_numpy(Bz).to(dev)
    y_all = torch.from_numpy(m.label.values.astype(np.int64)).to(dev)
    D = {"xtr": tX[tr], "vtr": tV[tr], "btr": tB[tr], "ytr": y_all[tr],
         "xva": tX[va], "vva": tV[va], "bva": tB[va], "yva": y_all[va],
         "w": torch.tensor([WEIGHTS[0], WEIGHTS[1], WEIGHTS[2]], dtype=torch.float32, device=dev)}

    selJ0 = train_model("J0_FROZEN_FUSION_CONTROL", False, D, dev)
    selJ1 = train_model("J1_BIOMARKER_FILM", True, D, dev)

    frozen = {"TASK10_ALL_SELECTION_FROZEN": True,
              "J0_checkpoint_sha256": selJ0["checkpoint_sha256"],
              "J1_checkpoint_sha256": selJ1["checkpoint_sha256"],
              "J0_selected": {k: selJ0[k] for k in ("selected_epoch", "val_auc", "val_macro_f1",
                                                    "val_loss")},
              "J1_selected": {k: selJ1[k] for k in ("selected_epoch", "val_auc", "val_macro_f1",
                                                    "val_loss")},
              "biomarker_scaler_sha256": sha(OUT / "biomarker_scaler.json"),
              "film_architecture": {"hidden": 64, "dim": 512, "gamma": "1 + 0.10*tanh(delta_gamma)",
                                    "beta": "0.10*tanh(beta_raw)", "form": "gamma*v + beta",
                                    "init": "normal(0,0.01) weights, zero bias -> near identity"},
              "no_cnn_parameters_in_optimizer": True,
              "cnn_encoder_invoked": False}
    (OUT / "task10_selection_frozen.json").write_text(json.dumps(frozen, indent=2))
    (OUT / "film_config.json").write_text(json.dumps(frozen["film_architecture"], indent=2))
    log("TASK10_ALL_SELECTION_FROZEN = YES -> TEST opens once")

    models = {}
    for nm, use_film in (("J0_FROZEN_FUSION_CONTROL", False), ("J1_BIOMARKER_FILM", True)):
        mod = Fusion(use_film).to(dev)
        mod.load_state_dict(torch.load(OUT / f"{nm}_selected.pth", map_location="cpu")["model"])
        models[nm] = mod

    y = m.loc[te, "label"].values
    Xte = (tX[te], tV[te], tB[te])
    preds = {nm: predict(mod, Xte, dev) for nm, mod in models.items()}
    e = pd.read_csv(ART8 / "test_predictions_E_RGB_VESSEL_FEATURE_FUSION.csv").set_index("image_id")
    g = pd.read_csv(ART8 / "test_predictions_G_RGB_VESSEL_SCALAR_FUSION.csv").set_index("image_id")
    tids = m.loc[te, "image_path"].values
    PE, PG = e.loc[tids, PCOLS].values, g.loc[tids, PCOLS].values

    M = {nm: metrics(y, P) for nm, P in preds.items()}
    M["E_RGB_VESSEL_FEATURE_FUSION"] = metrics(y, PE)
    M["G_RGB_VESSEL_SCALAR_FUSION"] = metrics(y, PG)
    rows = []
    for nm, P in list(preds.items()) + [("E_RGB_VESSEL_FEATURE_FUSION", PE),
                                        ("G_RGB_VESSEL_SCALAR_FUSION", PG)]:
        rows.append({**M[nm], "model": nm, "n": int(len(y))})
        pd.DataFrame(confusion_matrix(y, P.argmax(1), labels=[0, 1, 2])).to_csv(
            OUT / f"confusion_{nm}.csv", index=False)
        if nm in preds:
            pr = pd.DataFrame({"image_id": tids, "group_id": m.loc[te, "group_id"].values,
                               "source": m.loc[te, "source"].values, "true_label": y})
            pr["prob_Normal"], pr["prob_Pre_Plus"], pr["prob_Plus"] = P[:, 0], P[:, 1], P[:, 2]
            pr["predicted_label"] = P.argmax(1)
            pr.to_csv(OUT / f"test_predictions_{nm}.csv", index=False)
    R = pd.DataFrame(rows)[["model", "n"] + ALLKEYS]
    R.to_csv(OUT / "metrics_summary.csv", index=False)
    log(R[["model"] + KEYS].to_string(index=False))

    log("=== FiLM modulation diagnostics")
    diag = []
    for split, sel in (("val", va), ("test", te)):
        Xs = (tX[sel], tV[sel], tB[sel])
        mod = models["J1_BIOMARKER_FILM"]
        mod.eval()
        with torch.no_grad():
            _, gm, bt = mod(Xs[0], Xs[1], Xs[2], want_mod=True)
        gm, bt = gm.cpu().numpy(), bt.cpu().numpy()
        diag.append({"split": split, "n": int(sel.sum()),
                     "gamma_mean": float(gm.mean()), "gamma_std": float(gm.std()),
                     "gamma_p05": float(np.percentile(gm, 5)), "gamma_p50": float(np.percentile(gm, 50)),
                     "gamma_p95": float(np.percentile(gm, 95)),
                     "beta_mean": float(bt.mean()), "beta_std": float(bt.std()),
                     "beta_p05": float(np.percentile(bt, 5)), "beta_p50": float(np.percentile(bt, 50)),
                     "beta_p95": float(np.percentile(bt, 95)),
                     "mean_abs_gamma_minus_1": float(np.abs(gm - 1.0).mean()),
                     "mean_abs_beta": float(np.abs(bt).mean())})
        log(f"    {split}: gamma mean {gm.mean():.6f} std {gm.std():.6f} p05 {np.percentile(gm,5):.6f} "
            f"p95 {np.percentile(gm,95):.6f} | beta mean {bt.mean():.6f} std {bt.std():.6f} | "
            f"mean|gamma-1| {np.abs(gm-1).mean():.6f} mean|beta| {np.abs(bt).mean():.6f}")
    DG = pd.DataFrame(diag)
    DG.to_csv(OUT / "film_modulation_diagnostics.csv", index=False)

    log("=== biomarker sensitivity (post-freeze, descriptive, no retraining)")
    mod = models["J1_BIOMARKER_FILM"]
    base = preds["J1_BIOMARKER_FILM"]
    srows = []
    for k, f in enumerate(FEATS):
        B2 = tB[te].clone(); B2[:, k] = 0.0  # train mean in standardised units
        P2 = predict(mod, (tX[te], tV[te], B2), dev)
        ch = np.abs(P2 - base)
        srows.append({"biomarker": f, "replaced_with": "TRAIN mean (standardised 0)",
                      "mean_abs_prob_change": float(ch.mean()),
                      "median_abs_prob_change": float(np.median(ch)),
                      "p95_abs_prob_change": float(np.percentile(ch, 95)),
                      "max_abs_prob_change": float(ch.max()),
                      "mean_abs_change_argmax_flips": int((P2.argmax(1) != base.argmax(1)).sum())})
        log(f"    {f:20s} mean {ch.mean():.6f} median {np.median(ch):.6f} p95 "
            f"{np.percentile(ch,95):.6f} argmax flips {int((P2.argmax(1)!=base.argmax(1)).sum())}")
    pd.DataFrame(srows).to_csv(OUT / "biomarker_sensitivity.csv", index=False)

    log("=== source breakdown")
    srcs = m.loc[te, "source"].values
    srows2 = []
    for src in ("plus", "farfum_rop", "farabi"):
        sl = srcs == src
        if sl.sum() < 10:
            continue
        for nm, P in list(preds.items()) + [("E_RGB_VESSEL_FEATURE_FUSION", PE),
                                            ("G_RGB_VESSEL_SCALAR_FUSION", PG)]:
            srows2.append({**metrics(y[sl], P[sl]), "model": nm, "source": src, "n": int(sl.sum())})
    pd.DataFrame(srows2).to_csv(OUT / "source_breakdown.csv", index=False)

    log("=== paired bootstrap 10k")
    comps = [("J1_minus_J0", preds["J1_BIOMARKER_FILM"], preds["J0_FROZEN_FUSION_CONTROL"]),
             ("J1_minus_E", preds["J1_BIOMARKER_FILM"], PE),
             ("J1_minus_G", preds["J1_BIOMARKER_FILM"], PG)]
    Pb = []
    for tag, PA, PB_ in comps:
        o = {k: metrics(y, PA)[k] - metrics(y, PB_)[k] for k in ALLKEYS}
        Pb.append(run_bootstrap(tag, y, PA, PB_, o))
        log(f"    {tag} done")
    PBdf = pd.concat(Pb, ignore_index=True)
    PBdf.to_csv(OUT / "paired_all_10k.csv", index=False)
    for tag in ("J1_minus_J0", "J1_minus_E", "J1_minus_G"):
        PBdf[PBdf.comparison == tag].to_csv(OUT / f"paired_{tag}.csv", index=False)

    log("=== figures")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
        for j, nm in enumerate(("J0_FROZEN_FUSION_CONTROL", "J1_BIOMARKER_FILM")):
            h = pd.DataFrame(json.loads((OUT / f"{nm}_history.json").read_text()))
            ax[0][j].plot(h.epoch, h.train_loss, "o-", ms=3, label="train")
            ax[0][j].plot(h.epoch, h.val_loss, "s-", ms=3, label="val")
            ax[0][j].set_title(f"{nm[:24]} loss", fontsize=8); ax[0][j].legend(fontsize=7)
            ax[1][j].plot(h.epoch, h.val_auc, "o-", ms=3, label="val AUC")
            ax[1][j].plot(h.epoch, h.val_macro_f1, "s-", ms=3, label="val macro F1")
            ax[1][j].set_title(f"{nm[:24]} val AUC / F1", fontsize=8); ax[1][j].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / "learning_curves.png", dpi=170); plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for nm, P in list(preds.items()) + [("E", PE), ("G", PG)]:
            conf, corr, xs, ys_ = P.max(1), (P.argmax(1) == y), [], []
            for lo in np.linspace(0, 1, 16)[:-1]:
                mm = (conf >= lo) & (conf < lo + 1 / 15)
                if mm.sum():
                    xs.append(conf[mm].mean()); ys_.append(corr[mm].mean())
            ax[0].plot(xs, ys_, marker="o", ms=3, label=nm[:12])
        ax[0].plot([0, 1], [0, 1], "k--", lw=0.7); ax[0].legend(fontsize=6)
        ax[0].set_title("reliability, 15 equal-width bins", fontsize=8)
        pp = PBdf[PBdf.metric.isin(KEYS)]
        labs = (pp.comparison + " " + pp.metric).tolist(); ypos = np.arange(len(labs))
        ax[1].errorbar(pp.delta, ypos, xerr=[pp.delta - pp.ci95_lo, pp.ci95_hi - pp.delta],
                       fmt="o", ms=3)
        ax[1].axvline(0, color="k", lw=0.8, ls="--")
        ax[1].set_yticks(ypos); ax[1].set_yticklabels(labs, fontsize=5)
        ax[1].set_title("paired delta, 95% CI", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "calibration_and_forest.png", dpi=170); plt.close(fig)
        log("    figures written")
    except Exception as ex:  # noqa: BLE001
        log(f"    figures skipped: {ex}")

    summary = {"TASK10_STATUS": "COMPLETE", "classification":
               "SECONDARY_POST_PRIMARY_EXPLORATORY_CANONICAL_SPLIT_REANALYSIS",
               "J0_selection": {k: v for k, v in selJ0.items() if k != "history"},
               "J1_selection": {k: v for k, v in selJ1.items() if k != "history"},
               "frozen": frozen, "input_checks": checks,
               "metrics": R.to_dict("records"), "film_diagnostics": diag,
               "biomarker_sensitivity": srows, "source_breakdown": srows2,
               "paired_all_10k": PBdf.to_dict("records"), "test_n": int(te.sum()),
               "test_used_for_selection": False, "cnn_retraining": False}
    (OUT / "task10_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    shas = {p.name: sha(p) for p in sorted(OUT.glob("*")) if p.is_file() and p.name
            not in ("task10_progress.log", "train.log")}
    shas["task10_progress.log"] = "EXCLUDED_APPEND_ONLY_LOG"
    shas["train.log"] = "EXCLUDED_APPEND_ONLY_LOG"
    (OUT / "artifact_sha256.json").write_text(json.dumps(shas, indent=2))
    log("TASK10_STATUS = COMPLETE")


if __name__ == "__main__":
    main()
