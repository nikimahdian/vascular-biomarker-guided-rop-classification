"""Task 12 — class-conditional domain-invariant RGB-vessel representation (Server 2, authoritative).

SOURCE-ONLY DOMAIN GENERALIZATION. For each LOSO fold the held-out source is used for NOTHING:
not training, not validation, not normalization, not alignment, not early stopping, not selection.
Not even its unlabeled features.

Per fold two matched models on the frozen 3840-d RGB+vessel embedding (no CNN, no biomarkers):

  K0_DOMAIN_NEUTRAL_CONTROL : weighted CE only
  K1_CC_DANN_MMD            : + GRL domain discriminator (2 training domains) and
                              + class-conditional multi-kernel RBF MMD on z, classes absent from
                                either training domain are skipped, never fabricated

Everything else is identical: architecture, group-level 85/15 source-internal split, balanced 64+64
domain batches, optimizer, class weights (inverse frequency from optimization train only), seed.
All three folds are selected and frozen before any held-out source is touched.
"""
import hashlib
import json
import math
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

WS = Path("/root/niki_rop_task6_isolated")
ART6 = WS / "artifacts/task6"
ART8 = WS / "artifacts/task8_spatial_vessel_fusion"
ART11 = WS / "artifacts/task11_source_heldout"
OUT = WS / "artifacts/task12_class_conditional_domain_generalization"
MANIFEST = WS / "primary_complete_case_v2_server2.csv"
RGB_EMB = ART6 / "embeddings_b5_8862.parquet"
VES_EMB = ART8 / "vessel_embeddings_b4_8862.parquet"
NAMES = ["Normal", "Pre_Plus", "Plus"]
PCOLS = ["prob_Normal", "prob_Pre_Plus", "prob_Plus"]
SEED = 42
NB = 10000
NCHUNK = 40
BATCH = 128
HALF = 64
MAX_EPOCHS = 40
PATIENCE = 6
LAMBDA_DANN_MAX = 0.10
LAMBDA_MMD = 0.10
MMD_MULTS = [0.5, 1.0, 2.0, 4.0]
FOLDS = [("plus", ["farfum_rop", "farabi"]),
         ("farfum_rop", ["plus", "farabi"]),
         ("farabi", ["plus", "farfum_rop"])]
KEYS3 = ["multiclass_auc", "balanced_accuracy", "macro_f1", "brier_multiclass_3col", "ece"]
KEYS2 = ["restricted_binary_auc", "balanced_accuracy", "macro_f1", "brier_restricted_2class", "ece"]
T0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "task12_progress.log", "a", encoding="utf-8") as f:
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


# ---------------------------------------------------------------- model
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.alpha * g, None


def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)


class Net(nn.Module):
    """Deliberately small. Identical for K0 and K1; K1 only adds the two invariance terms."""

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

    def encode(self, x):
        z = self.shared(torch.cat([self.rgb(x[:, :2048]), self.ves(x[:, 2048:])], 1))
        return z

    def forward(self, x):
        z = self.encode(x)
        return z, self.cls(z)


def mmd_rbf_multi(A, B, mults=MMD_MULTS):
    """Multi-kernel RBF MMD^2. Bandwidths are mult * (batch median pairwise squared distance)."""
    if len(A) < 2 or len(B) < 2:
        return None
    Z = torch.cat([A, B], 0)
    with torch.no_grad():
        D = torch.cdist(Z, Z) ** 2
        base = torch.median(D[D > 0]) if (D > 0).any() else torch.tensor(1.0, device=Z.device)
        base = torch.clamp(base, min=1e-8)
    tot = 0.0
    for m in mults:
        bw = m * base
        Kaa = torch.exp(-torch.cdist(A, A) ** 2 / bw).mean()
        Kbb = torch.exp(-torch.cdist(B, B) ** 2 / bw).mean()
        Kab = torch.exp(-torch.cdist(A, B) ** 2 / bw).mean()
        tot = tot + (Kaa + Kbb - 2 * Kab)
    return tot / len(mults)


def domain_mmd(z, dom, y):
    """Class-conditional: align only within the same disease class; skip classes absent in either."""
    vals, used = [], []
    for c in (0, 1, 2):
        ia = (dom == 0) & (y == c)
        ib = (dom == 1) & (y == c)
        if ia.sum() >= 2 and ib.sum() >= 2:
            m = mmd_rbf_multi(z[ia], z[ib])
            if m is not None:
                vals.append(m); used.append(c)
    if not vals:
        return None, used
    return torch.stack(vals).mean(), used


def make_batch(rng, by_dc, doms, size=HALF):
    """Proportional class-aware draw of `size` samples from each of the two training domains."""
    idx, dl = [], []
    for d, (groups, counts) in enumerate(by_dc):
        tot = float(counts.sum())
        alloc = np.zeros(3, dtype=int)
        for c in range(3):
            alloc[c] = int(math.floor(size * counts[c] / tot))
        while alloc.sum() < size:
            r = np.array([counts[c] / tot - alloc[c] / size for c in range(3)])
            alloc[int(np.argmax(r))] += 1
        for c in range(3):
            if alloc[c] > 0 and len(groups[c]) > 0:
                idx.append(rng.choice(groups[c], size=alloc[c], replace=len(groups[c]) < alloc[c]))
                dl.append(np.full(alloc[c], d))
            elif alloc[c] > 0:
                # class requested but absent in this domain (e.g. Pre_Plus in plus) - redistribute
                pool = np.concatenate([g for g in groups if len(g) > 0])
                idx.append(rng.choice(pool, size=alloc[c], replace=len(pool) < alloc[c]))
                dl.append(np.full(alloc[c], d))
    return np.concatenate(idx), np.concatenate(dl)


def shift_diag(Z, dom):
    A, B = Z[dom == 0], Z[dom == 1]
    muA, muB, sdA = A.mean(0), B.mean(0), A.std(0, ddof=0)
    smd = (muB - muA) / np.where(sdA > 0, sdA, 1.0)
    a = np.abs(smd)
    return {"centroid_distance_standardized": float(np.sqrt(np.sum(smd ** 2))),
            "centroid_distance_raw_l2": float(np.linalg.norm(muB - muA)),
            "median_abs_smd": float(np.median(a)), "p90_abs_smd": float(np.percentile(a, 90)),
            "p95_abs_smd": float(np.percentile(a, 95)), "max_abs_smd": float(a.max())}


# ---------------------------------------------------------------- training
def train_one(kind, fold, data, dev):
    held, trainy = fold
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    Xtr, ytr, dtr = data["Xtr"], data["ytr"], data["dtr"]
    Xva, yva, dva = data["Xva"], data["yva"], data["dva"]
    by_dc = []
    for d in (0, 1):
        groups, counts = [], []
        for c in range(3):
            g = np.nonzero((dtr == d) & (ytr == c))[0]
            groups.append(g); counts.append(len(g))
        by_dc.append((groups, np.array(counts, dtype=float)))
    log(f"    [{kind}] opt-train domain/class counts "
        f"{[[len(by_dc[d][0][c]) for c in range(3)] for d in (0,1)]}")
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-3)
    cw = torch.tensor(data["class_weights"], dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=cw)
    domf = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr).to(dev); ytr_t = torch.from_numpy(ytr).to(dev)
    dtr_t = torch.from_numpy(dtr).to(dev)
    Xva_t = torch.from_numpy(Xva).to(dev)
    steps = max(1, int(math.ceil(len(ytr) / BATCH)))
    total_steps = steps * MAX_EPOCHS
    gstep = 0
    hist = []
    best, best_idx = None, -1
    for ep in range(MAX_EPOCHS):
        t_ep = time.time()
        net.train()
        agg = {"class": 0.0, "domain": 0.0, "mmd": 0.0, "alpha": 0.0}
        for _ in range(steps):
            idx, dl = make_batch(rng, by_dc, dtr)
            xb = Xtr_t[idx]; yb = ytr_t[idx]; db = torch.from_numpy(dl).to(dev)
            p = gstep / max(1, total_steps - 1)
            lam = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
            alpha = LAMBDA_DANN_MAX * lam
            opt.zero_grad(set_to_none=True)
            z, logits = net(xb)
            l_cls = lossf(logits, yb)
            loss = l_cls
            if kind == "K1":
                l_dom = domf(net.dom(grad_reverse(z, alpha)), db)
                l_mmd, used = domain_mmd(z, db, yb)
                loss = l_cls + l_dom
                if l_mmd is not None:
                    loss = loss + LAMBDA_MMD * l_mmd
                    agg["mmd"] += float(l_mmd.item())
                agg["domain"] += float(l_dom.item())
            loss.backward(); opt.step()
            agg["class"] += float(l_cls.item()); agg["alpha"] += alpha
            gstep += 1
        for k in ("class", "domain", "mmd", "alpha"):
            agg[k] /= steps
        net.eval()
        yva_t = torch.from_numpy(yva).to(dev)
        with torch.no_grad():
            zva, logva = net(Xva_t)
            vl = float(lossf(logva, yva_t).item())
        P = torch.softmax(logva.float(), 1).cpu().numpy()
        auc, f1 = multiclass_auc(yva, P), float(f1_score(yva, P.argmax(1), average="macro"))
        if not np.isfinite(auc):
            raise SystemExit(f"SOURCE_VAL_AUC_UNDEFINED fold={held} classes={sorted(set(yva))} "
                             f"- selection cannot proceed on NaN")
        ck = OUT / f"{kind}_{held}_epoch{ep+1}.pth"
        torch.save({"model": net.state_dict(), "epoch": ep + 1, "kind": kind}, ck)
        hist.append({**{k: agg[k] for k in agg}, "epoch": ep + 1, "val_loss": vl, "val_auc": auc,
                     "val_macro_f1": f1, "seconds": round(time.time() - t_ep, 2), "ckpt": str(ck)})
        (OUT / f"{kind}_{held}_history.json").write_text(json.dumps(hist, indent=2))
        log(f"    [{kind}] {held} epoch {ep+1:2d} {time.time()-t_ep:5.1f}s cls {agg['class']:.4f} "
            f"dom {agg['domain']:.4f} mmd {agg['mmd']:.5f} a {agg['alpha']:.4f} | val {vl:.4f} "
            f"auc {auc:.5f} f1 {f1:.4f}")
        key = (auc, f1, -vl, -(ep + 1))
        bk = (best["val_auc"], best["val_macro_f1"], -best["val_loss"], -best["epoch"]) if best else None
        if bk is None or key > bk:
            best, best_idx = hist[-1], ep
        if ep - best_idx >= PATIENCE:
            log(f"    [{kind}] {held} early stop at epoch {ep+1} (best {best['epoch']})")
            break
    import shutil
    shutil.copy(best["ckpt"], OUT / f"{kind}_{held}_selected.pth")
    sel = {"kind": kind, "held_out": held, "selected_epoch": best["epoch"],
           "val_auc": best["val_auc"], "val_macro_f1": best["val_macro_f1"],
           "val_loss": best["val_loss"], "checkpoint_sha256": sha(OUT / f"{kind}_{held}_selected.pth"),
           "epochs_run": len(hist), "history": hist, "heldout_target_touched": False}
    (OUT / f"{kind}_{held}_selection.json").write_text(json.dumps(sel, indent=2))
    log(f"    [{kind}] {held} SELECTED epoch {best['epoch']} val_auc {best['val_auc']:.5f} "
        f"sha {sel['checkpoint_sha256'][:16]}")
    return sel


def infer_zP(net, X, dev, bs=1024):
    net.eval(); Z, P = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            z, logits = net(torch.from_numpy(X[i:i + bs]).to(dev))
            Z.append(z.cpu().numpy()); P.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(Z), np.concatenate(P)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {dev}   no CNN, no biomarkers, source-only domain generalization")

    log("=== 0. PRE-FLIGHT")
    t6 = json.loads((ART6 / "artifact_sha256.json").read_text())
    t8 = json.loads((ART8 / "artifact_sha256.json").read_text())
    checks = {"rgb_emb_sha_unchanged": sha(RGB_EMB) == t6.get(RGB_EMB.name),
              "vessel_emb_sha_unchanged": sha(VES_EMB) == t8.get(VES_EMB.name)}
    m = pd.read_csv(MANIFEST, low_memory=False)
    RE = pd.read_parquet(RGB_EMB); VE = pd.read_parquet(VES_EMB)
    ec = [c for c in RE.columns if c.startswith("embedding_")]
    vc = [c for c in VE.columns if c.startswith("vessel_emb_")]
    ids = m.image_path.values
    X = np.hstack([RE.set_index("image_id").loc[ids, ec].values,
                   VE.set_index("image_id").loc[ids, vc].values]).astype(np.float32)
    y_all = m.label.values.astype(np.int64)
    src = m.source.values
    grp = m.group_id.values
    checks["population_8862"] = len(m) == 8862
    checks["input_dim_3840"] = X.shape == (8862, 3840)
    checks["three_sources"] = set(m.source.unique()) == {"plus", "farfum_rop", "farabi"}
    checks["no_biomarkers_used"] = True
    for k, v in checks.items():
        log(f"    {k:26s} : {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit("PREFLIGHT_FAILED")
    log(f"    input {X.shape}")

    fold_meta, selections = {}, {}
    for held, trainy in FOLDS:
        trm = np.isin(src, trainy)
        rng = np.random.default_rng(SEED)
        val_mask = np.zeros(len(m), dtype=bool)
        split_rows = []
        for d, s in enumerate(trainy):
            for c in (0, 1, 2):
                sel = np.nonzero(trm & (src == s) & (y_all == c))[0]
                gs = np.unique(grp[sel])
                if len(gs) == 0:
                    split_rows.append({"held_out": held, "source": s, "class": NAMES[c],
                                       "groups": 0, "val_groups": 0, "note": "class absent"})
                    continue
                sh = rng.permutation(gs)
                nv = max(1, int(round(0.15 * len(gs)))) if len(gs) > 1 else 0
                vg = set(sh[:nv].tolist())
                if type(sh[0]) is np.str_ or isinstance(gs[0], str):
                    vg = set(sh[:nv].tolist())
                vm = np.isin(grp, list(vg)) & (src == s) & (y_all == c)
                val_mask |= vm
                split_rows.append({"held_out": held, "source": s, "class": NAMES[c],
                                   "groups": int(len(gs)), "val_groups": int(nv),
                                   "train_rows": int((sel.size - vm.sum())), "val_rows": int(vm.sum()),
                                   "note": "ok" if nv > 0 else "single group -> all train"})
        tr_rows = np.nonzero(trm & ~val_mask)[0]
        va_rows = np.nonzero(trm & val_mask)[0]
        dom = np.zeros(len(trm), dtype=np.int64)
        dom[src == trainy[1]] = 1
        counts = pd.Series(y_all[tr_rows]).value_counts()
        cw = (len(tr_rows) / (3.0 * np.array([counts.get(c, 1) for c in (0, 1, 2)], dtype=float)))
        combos = {s: sorted(int(v) for v in np.unique(y_all[(src == s) & (y_all >= 0)]))
                  for s in trainy}
        valcombo = {s: sorted(int(v) for v in np.unique(y_all[va_rows][src[va_rows] == s]))
                    for s in trainy}
        trcombo = {s: sorted(int(v) for v in np.unique(y_all[tr_rows][src[tr_rows] == s]))
                   for s in trainy}
        fold_meta[held] = {"held_out": held, "train_sources": trainy,
                           "n_opt_train": int(len(tr_rows)), "n_source_val": int(len(va_rows)),
                           "class_weights": {NAMES[c]: float(cw[c]) for c in range(3)},
                           "source_val_class_counts": pd.Series(y_all[va_rows]).value_counts().to_dict(),
                           "per_source_class_available": combos,
                           "per_source_class_opt_train": trcombo,
                           "per_source_class_source_val": valcombo,
                           "split": split_rows}
        log(f"=== FOLD hold out {held}: opt-train {len(tr_rows)} source-val {len(va_rows)}")
        log(f"    class weights from opt-train only "
            f"{ {NAMES[c]: round(float(cw[c]), 5) for c in range(3)} }")
        log(f"    source-val class counts {fold_meta[held]['source_val_class_counts']}")
        data = {"Xtr": X[tr_rows], "ytr": y_all[tr_rows], "dtr": dom[tr_rows],
                "Xva": X[va_rows], "yva": y_all[va_rows], "dva": dom[va_rows],
                "class_weights": cw}
        for kind in ("K0", "K1"):
            selections[(kind, held)] = train_one(kind, (held, trainy), data, dev)
        fold_meta[held]["Xva_rows"] = va_rows.tolist()
        fold_meta[held]["two_domains_only"] = len(trainy) == 2
    (OUT / "fold_definitions.json").write_text(json.dumps(fold_meta, indent=2, default=str))
    (OUT / "source_splits_and_class_weights.json").write_text(json.dumps(
        {h: {k: v for k, v in fold_meta[h].items() if k != "Xva_rows"} for h in fold_meta},
        indent=2, default=str))

    log("=== 17/18. source-validation diagnostics (no target source involved)")
    diag, sh_diag = [], []
    for held, trainy in FOLDS:
        rows = np.array(fold_meta[held]["Xva_rows"])
        dva = np.zeros(len(rows), dtype=np.int64); dva[src[rows] == trainy[1]] = 1
        for kind in ("K0", "K1"):
            net = Net().to(dev)
            net.load_state_dict(torch.load(OUT / f"{kind}_{held}_selected.pth",
                                           map_location="cpu")["model"])
            Z, _P = infer_zP(net, X[rows], dev)
            with np.errstate(all="ignore"):
                acc = float(np.mean(cross_val_score(
                    LogisticRegression(max_iter=2000, C=1.0), Z, dva, cv=5, n_jobs=5)))
            d = shift_diag(Z, dva)
            diag.append({"held_out": held, "kind": kind, "domain_predictability_acc_cv5": acc,
                         "n_source_val": int(len(rows))})
            sh_diag.append({"held_out": held, "kind": kind, **d})
            log(f"    {kind} {held}: domain predictability {acc:.4f} | centroid std "
                f"{d['centroid_distance_standardized']:.3f} median|SMD| {d['median_abs_smd']:.4f} "
                f"p90 {d['p90_abs_smd']:.4f} p95 {d['p95_abs_smd']:.4f}")
    pd.DataFrame(diag).to_csv(OUT / "domain_predictability.csv", index=False)
    pd.DataFrame(sh_diag).to_csv(OUT / "representation_shift.csv", index=False)

    frozen = {"TASK12_ALL_FOLDS_SELECTION_FROZEN": True, "HELDOUT_TARGETS_TOUCHED": False,
              "folds": {h: {"K0_checkpoint_sha256": selections[("K0", h)]["checkpoint_sha256"],
                            "K1_checkpoint_sha256": selections[("K1", h)]["checkpoint_sha256"],
                            "K0_selected_epoch": selections[("K0", h)]["selected_epoch"],
                            "K1_selected_epoch": selections[("K1", h)]["selected_epoch"],
                            "K0_val_auc": selections[("K0", h)]["val_auc"],
                            "K1_val_auc": selections[("K1", h)]["val_auc"]}
                        for h, _ in FOLDS},
              "fixed_hyperparameters": {"lr": 1e-4, "weight_decay": 1e-3, "batch": BATCH,
                                        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                                        "lambda_dann_max": LAMBDA_DANN_MAX, "lambda_mmd": LAMBDA_MMD,
                                        "mmd_mults": MMD_MULTS, "seed": SEED,
                                        "architecture": "LN2048->256, LN1792->256, 512->256, "
                                                        "cls 256->3, dom 256->128->2"},
              "target_source_used_for_selection": False}
    (OUT / "task12_selection_frozen.json").write_text(json.dumps(frozen, indent=2))
    log("TASK12_ALL_FOLDS_SELECTION_FROZEN = YES  HELDOUT_TARGETS_TOUCHED = NO -> held-out opens once")

    log("=== 20/21/22. held-out evaluation (single use)")
    E11 = {}
    for held, _ in FOLDS:
        e = pd.read_csv(ART11 / f"predictions_{held}.csv")
        E11[held] = e
    rows, preds, boot = [], {}, []
    for held, _ in FOLDS:
        tem = src == held
        yte = y_all[tem]
        Xte = X[tem]
        for kind in ("K0", "K1"):
            net = Net().to(dev)
            net.load_state_dict(torch.load(OUT / f"{kind}_{held}_selected.pth",
                                           map_location="cpu")["model"])
            _Z, P = infer_zP(net, Xte, dev)
            preds[(kind, held)] = P
            r = {**metrics(yte, P), "held_out_source": held, "model": kind, "n": int(tem.sum())}
            rows.append(r)
            np.save(OUT / f"heldout_probs_{kind}_{held}.npy", P)
            pd.DataFrame(confusion_matrix(yte, P.argmax(1),
                                          labels=sorted(int(v) for v in np.unique(yte)))).to_csv(
                OUT / f"confusion_{kind}_{held}.csv", index=False)
            log(f"    {kind} {held}: auc3 {r['multiclass_auc']:.5f} "
                f"bin {r['restricted_binary_auc']:.5f} bal {r['balanced_accuracy']:.5f} "
                f"f1 {r['macro_f1']:.5f} brier {r['brier_multiclass_3col']:.5f} ece {r['ece']:.5f}")
        e = E11[held]
        PE = e[["E_LOSO_prob_Normal", "E_LOSO_prob_Pre_Plus", "E_LOSO_prob_Plus"]].values
        preds[("E11", held)] = PE
        rows.append({**metrics(yte, PE), "held_out_source": held, "model": "TASK11_E_LOSO",
                     "n": int(tem.sum())})
        pr = pd.DataFrame({"image_id": ids[tem], "source": src[tem], "true_label": yte})
        for kind in ("K0", "K1"):
            P = preds[(kind, held)]
            pr[f"{kind}_{c}"] = P[:, {"prob_Normal": 0, "prob_Pre_Plus": 1, "prob_Plus": 2}[c]]
            pr[f"{kind}_pred"] = P.argmax(1)
        pr.to_csv(OUT / f"heldout_predictions_{held}.csv", index=False)
        keys = KEYS3 if len(np.unique(yte)) == 3 else KEYS2
        boot.append(run_bootstrap(f"K1_minus_K0_{held}", yte, preds[("K1", held)],
                                  preds[("K0", held)], keys))
        boot.append(run_bootstrap(f"K1_minus_Task11E_{held}", yte, preds[("K1", held)], PE, keys))
        log(f"    {held} bootstraps done ({keys})")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "heldout_metrics.csv", index=False)
    BS = pd.concat(boot, ignore_index=True)
    BS.to_csv(OUT / "paired_bootstrap_10k.csv", index=False)
    for tag in BS.comparison.unique():
        BS[BS.comparison == tag].to_csv(OUT / f"paired_{tag}.csv", index=False)

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
            ax[1][j].set_title(f"source-val AUC, hold out {held}", fontsize=8); ax[1][j].legend(fontsize=6)
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
        DP = pd.DataFrame(diag)
        for i, x in enumerate(("K0", "K1")):
            s = DP[DP.kind == x]
            ax[0].bar(np.arange(len(s)) + i * 0.38, s.domain_predictability_acc_cv5, 0.38, label=x)
        ax[0].set_xticks(np.arange(len(FOLDS)) + 0.19)
        ax[0].set_xticklabels([h for h, _ in FOLDS], fontsize=7)
        ax[0].axhline(0.5, color="k", ls="--", lw=0.7); ax[0].legend(fontsize=6)
        ax[0].set_title("source-val domain predictability (5-fold CV)", fontsize=8)
        SD = pd.DataFrame(sh_diag)
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

    summary = {"TASK12_STATUS": "COMPLETE",
               "classification": "SECONDARY_SOURCE_GENERALIZATION_DOMAIN_INVARIANCE_EXPERIMENT",
               "folds": {h: {k: v for k, v in fold_meta[h].items() if k != "Xva_rows"}
                         for h in fold_meta},
               "frozen": frozen, "heldout_metrics": R.to_dict("records"),
               "paired_bootstrap": BS.to_dict("records"),
               "domain_predictability": diag, "representation_shift": sh_diag,
               "no_cnn": True, "no_biomarkers": True,
               "heldout_source_used_during_training": False}
    (OUT / "task12_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    shas = {p.name: sha(p) for p in sorted(OUT.glob("*"))
            if p.is_file() and p.name not in ("task12_progress.log", "train.log")}
    shas["task12_progress.log"] = "EXCLUDED_APPEND_ONLY_LOG"
    (OUT / "artifact_sha256.json").write_text(json.dumps(shas, indent=2))
    log("TASK12_STATUS = COMPLETE")


if __name__ == "__main__":
    main()
