"""Task 13 — ROPDeepX-style dual-RGB soft attention + spatial vessel complementarity test (Server 2).

Model L_ROPDEEPX_STYLE_RGB: ResNet50 (2048) and EfficientNet-B4 (1792) both fed the same 384x384
canonical RGB image, each projected to 1024 by Linear+BatchNorm1d+ReLU, concatenated (2048) into a
fusion block Linear(2048,1024)+BN+ReLU+Dropout(0.30) whose output produces a 2-way softmax attention,
and the attended representation f_att = a_r*z_r + a_e*z_e (1024) feeds Linear(1024,3).

Staged conservative training (stage 1 freezes both backbones for 3 epochs at 3e-4; stage 2 unfreezes
ResNet50 layer4 and the final EfficientNet-B4 block group + conv_head/bn2 with differential lr
5e-6 / 5e-5 under OneCycleLR, weight decay 5e-4, max 20 epochs, patience 5). Weighted CE with label
smoothing 0.05 and canonical-train-only class weights.

Then two matched XGBoost models on frozen features, using the exact frozen Task-6/8 configuration:
  M0_ROPDEEPX_EMBEDDING_ONLY  = 1024-d f_att
  M1_ROPDEEPX_PLUS_VESSEL     = 1024-d f_att + 1792-d frozen Task-8 vessel embedding (2816)
The only difference between M0 and M1 is the spatial vessel representation. No biomarkers anywhere.
"""
import hashlib
import json
import math
import random
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

WS = Path("/root/niki_rop_task6_isolated")
ART6 = WS / "artifacts/task6"
ART8 = WS / "artifacts/task8_spatial_vessel_fusion"
OUT = WS / "artifacts/task13_ropdeepx_style"
MANIFEST = WS / "primary_complete_case_v2_server2.csv"
VES_EMB = ART8 / "vessel_embeddings_b4_8862.parquet"
CANON_FP = "0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8"
WEIGHTS_T6 = {0: 0.46051, 1: 3.18103, 2: 1.94512}
NAMES = ["Normal", "Pre_Plus", "Plus"]
PCOLS = ["prob_Normal", "prob_Pre_Plus", "prob_Plus"]
SEED = 42
NB = 10000
NCHUNK = 40
BATCH = 16
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
KEYS = ["multiclass_auc", "balanced_accuracy", "macro_f1", "brier", "ece"]
ALLKEYS = KEYS + ["auc_Normal", "auc_Pre_Plus", "auc_Plus"]
T0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "task13_progress.log", "a", encoding="utf-8") as f:
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
    return {"n": int(len(y)), "classes_present": present,
            "multiclass_auc": multiclass_auc(y, P), "macro_ovr_auc": multiclass_auc(y, P),
            "balanced_accuracy": float(np.mean([(pred[y == k] == k).mean() for k in present])),
            "macro_f1": float(f1_score(y, pred, labels=present, average="macro")),
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


def run_bootstrap(tag, y, PA, PB):
    obs = {k: metrics(y, PA)[k] - metrics(y, PB)[k] for k in ALLKEYS}
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


class RGBDS(Dataset):
    def __init__(self, paths, labels, train):
        self.p, self.y, self.train = list(paths), list(labels), train
        if train:
            self.tf = T.Compose([T.RandomResizedCrop(384, scale=(0.95, 1.0), ratio=(1.0, 1.0)),
                                 T.RandomHorizontalFlip(0.5), T.RandomRotation(10),
                                 T.ColorJitter(0.10, 0.10, 0.10, 0.0),
                                 T.ToTensor(), T.Normalize(MEAN, STD)])
        else:
            self.tf = T.Compose([T.Resize((384, 384), IM.BILINEAR), T.ToTensor(),
                                 T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), self.y[i]


class LModel(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.r50 = timm.create_model("resnet50", pretrained=True, num_classes=0, global_pool="avg")
        self.b4 = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0,
                                    global_pool="avg")
        self.pr = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.pe = nn.Sequential(nn.Linear(1792, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                  nn.Dropout(0.30))
        self.attn = nn.Linear(1024, 2)
        self.cls = nn.Linear(1024, 3)

    def forward(self, x, want_att=False):
        z_r = self.pr(self.r50(x))
        z_e = self.pe(self.b4(x))
        f_joint = self.fuse(torch.cat([z_r, z_e], 1))
        a = torch.softmax(self.attn(f_joint), 1)
        f_att = a[:, 0:1] * z_r + a[:, 1:2] * z_e
        out = self.cls(f_att)
        if want_att:
            return out, a, f_att
        return out

    def embed(self, x):
        return self.forward(x, want_att=True)[2]


def set_backbones(net, on):
    for p in net.r50.parameters():
        p.requires_grad = False
    for p in net.b4.parameters():
        p.requires_grad = False
    if on:
        for p in net.r50.layer4.parameters():
            p.requires_grad = True
        for blk in list(net.b4.blocks)[-1:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in list(net.b4.conv_head.parameters()) + list(net.b4.bn2.parameters()):
            p.requires_grad = True


def head_params(net):
    return (list(net.pr.parameters()) + list(net.pe.parameters()) + list(net.fuse.parameters())
            + list(net.attn.parameters()) + list(net.cls.parameters()))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {dev}  batch {BATCH} (batch size was not fixed by the contract; 16 is the "
        f"Task-6/9 convention and was chosen before any training)")

    log("=== 0. PRE-FLIGHT")
    t6 = json.loads((ART6 / "artifact_sha256.json").read_text())
    t8 = json.loads((ART8 / "artifact_sha256.json").read_text())
    params = json.loads((ART6 / "downstream_xgb_selected.json").read_text())["params"]
    log(f"    frozen downstream XGBoost config {params}")
    m = pd.read_csv(MANIFEST, low_memory=False)
    ids = m.image_path.values
    checks = {"population_8862": len(m) == 8862,
              "split_counts": (m.split.value_counts().get("train") == 6203
                               and m.split.value_counts().get("val") == 1328
                               and m.split.value_counts().get("test") == 1331),
              "no_group_overlap": bool((m.groupby("group_id").split.nunique() == 1).all()),
              "labels_0_1_2": set(m.label.unique()) == {0, 1, 2},
              "vessel_emb_sha_unchanged": sha(VES_EMB) == t8.get(VES_EMB.name),
              "B_EMBEDDING_ONLY_pred_unchanged":
                  sha(ART6 / "test_predictions_B_EMBEDDING_ONLY.csv")
                  == t6.get("test_predictions_B_EMBEDDING_ONLY.csv"),
              "E_pred_unchanged": sha(ART8 / "test_predictions_E_RGB_VESSEL_FEATURE_FUSION.csv")
              == t8.get("test_predictions_E_RGB_VESSEL_FEATURE_FUSION.csv"),
              "G_pred_unchanged": sha(ART8 / "test_predictions_G_RGB_VESSEL_SCALAR_FUSION.csv")
              == t8.get("test_predictions_G_RGB_VESSEL_SCALAR_FUSION.csv"),
              "rgb_exist": bool(m.image_path.map(lambda p: Path(p).exists()).all())}
    for k, v in checks.items():
        log(f"    {k:32s} : {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit("PREFLIGHT_FAILED")
    log(f"    canonical split fingerprint {CANON_FP[:16]}... is the SHA-256 of "
        f"data/splits/all.csv (8870 rows). That file exists on Server 1 only; Server 1 verified it "
        f"byte-for-byte during this task and the population/split/group checks above are re-verified "
        f"here against the 8862-row complete-case manifest.")

    tr = m[m.split == "train"]; va = m[m.split == "val"]; te = m[m.split == "test"]
    counts = pd.Series(tr.label.values).value_counts()
    cw = np.array([len(tr) / (3.0 * counts.get(c, 1)) for c in (0, 1, 2)])
    log(f"    canonical-TRAIN-only class weights {dict(zip(NAMES, np.round(cw, 5).tolist()))}")
    log(f"    (Task-6 frozen weights were {WEIGHTS_T6} - recomputed here as the contract requires)")
    json.dump({"class_names": NAMES, "weights": cw.tolist(), "computed_on": "canonical TRAIN only",
               "n_train": int(len(tr)), "train_class_counts": counts.sort_index().tolist(),
               "label_smoothing": 0.05, "batch_size": BATCH},
              open(OUT / "loss_config.json", "w"), indent=2)

    dl_tr = DataLoader(RGBDS(tr.image_path, tr.label, True), batch_size=BATCH, shuffle=True,
                       num_workers=12, pin_memory=True, persistent_workers=True, drop_last=True)
    dl_va = DataLoader(RGBDS(va.image_path, va.label, False), batch_size=32, shuffle=False,
                       num_workers=8, pin_memory=True)
    yv = va.label.values

    log("=== L_ROPDEEPX_STYLE_RGB")
    net = LModel().to(dev)
    n_bb = sum(p.numel() for p in list(net.r50.parameters()) + list(net.b4.parameters()))
    log(f"    backbone params {n_bb}  head params {sum(p.numel() for p in head_params(net))}")
    w = torch.tensor(cw, dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    hist = []
    set_backbones(net, False)
    opt = torch.optim.AdamW(head_params(net), lr=3e-4, weight_decay=1e-4)
    sched = None
    log("    stage1: both backbones frozen, head lr 3e-4, wd 1e-4, 3 epochs")
    steps = len(dl_tr)
    best, best_idx = None, -1
    for ep in range(23):
        stage = 1 if ep < 3 else 2
        if ep == 3:
            set_backbones(net, True)
            bb = [p for p in list(net.r50.parameters()) + list(net.b4.parameters())
                  if p.requires_grad]
            opt = torch.optim.AdamW([{"params": bb, "lr": 5e-6},
                                     {"params": head_params(net), "lr": 5e-5}], weight_decay=5e-4)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=[5e-6, 5e-5], total_steps=20 * steps, pct_start=0.10)
            log("    stage2: ResNet50 layer4 + EfficientNet-B4 last block group/conv_head/bn2 "
                "unfrozen, lr 5e-6 backbone / 5e-5 head, wd 5e-4, OneCycleLR, max 20 epochs, "
                "patience 5")
        t_ep = time.time()
        net.train(); tl = 0.0
        for x, y in dl_tr:
            x = x.to(dev, non_blocking=True); y = torch.as_tensor(y).to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(x), y)
            loss.backward(); opt.step()
            if sched is not None:
                sched.step()
            tl += float(loss.item()) * len(y)
        tl /= (steps * BATCH)
        net.eval(); vl, P, A = 0.0, [], []
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(dev)
                o, a, _f = net(x, want_att=True)
                vl += float(lossf(o, torch.as_tensor(y).to(dev)).item()) * len(y)
                P.append(torch.softmax(o.float(), 1).cpu().numpy()); A.append(a.cpu().numpy())
        vl /= len(va); P = np.concatenate(P); A = np.concatenate(A)
        auc, f1 = multiclass_auc(yv, P), float(f1_score(yv, P.argmax(1), average="macro"))
        ck = OUT / f"L_stage{stage}_epoch{ep+1}.pth"
        torch.save({"model": net.state_dict(), "stage": stage, "epoch": ep + 1}, ck)
        hist.append({"epoch": ep + 1, "stage": stage, "train_loss": tl, "val_loss": vl,
                     "val_auc": auc, "val_macro_f1": f1,
                     "attn_resnet_mean": float(A[:, 0].mean()),
                     "attn_effnet_mean": float(A[:, 1].mean()),
                     "seconds": round(time.time() - t_ep, 1), "ckpt": str(ck)})
        (OUT / "L_history.json").write_text(json.dumps(hist, indent=2))
        np.save(OUT / f"L_val_probs_epoch{ep+1}.npy", P)
        log(f"    L s{stage} epoch {ep+1:2d} {time.time()-t_ep:6.1f}s train {tl:.4f} val {vl:.4f} "
            f"auc {auc:.5f} f1 {f1:.4f} | attn r {A[:,0].mean():.4f} e {A[:,1].mean():.4f}")
        key = (auc, f1, -vl, -(ep + 1))
        bk = (best["val_auc"], best["val_macro_f1"], -best["val_loss"], -best["epoch"]) \
            if best else None
        if bk is None or key > bk:
            best, best_idx = hist[-1], ep
        if stage == 2 and ep - best_idx >= 5:
            log(f"    L early stop at epoch {ep+1} (best {best['epoch']})")
            break
    import shutil
    shutil.copy(best["ckpt"], OUT / "L_selected.pth")
    sel = {"model": "L_ROPDEEPX_STYLE_RGB", "selected_epoch": best["epoch"],
           "selected_stage": best["stage"], "val_auc": best["val_auc"],
           "val_macro_f1": best["val_macro_f1"], "val_loss": best["val_loss"],
           "checkpoint_sha256": sha(OUT / "L_selected.pth"), "epochs_run": len(hist),
           "history": hist, "test_touched": False}
    (OUT / "L_selection.json").write_text(json.dumps(sel, indent=2))
    log(f"    L SELECTED epoch {best['epoch']} stage {best['stage']} "
        f"val_auc {best['val_auc']:.5f} sha {sel['checkpoint_sha256'][:16]}")

    log("=== 10. attention diagnostics at the selected checkpoint")
    ck = torch.load(OUT / "L_selected.pth", map_location="cpu")
    net.load_state_dict(ck["model"]); net.eval()
    with torch.no_grad():
        Aval = []
        for x, y in dl_va:
            Aval.append(net(x.to(dev), want_att=True)[1].cpu().numpy())
    Aval = np.concatenate(Aval)
    att = {"split": "val", "n": int(len(Aval))}
    for j, nm in enumerate(("resnet", "effnet")):
        v = Aval[:, j]
        att.update({f"attn_{nm}_mean": float(v.mean()), f"attn_{nm}_std": float(v.std()),
                    f"attn_{nm}_p05": float(np.percentile(v, 5)),
                    f"attn_{nm}_p50": float(np.percentile(v, 50)),
                    f"attn_{nm}_p95": float(np.percentile(v, 95))})
    pd.DataFrame([att]).to_csv(OUT / "attention_stats_val.csv", index=False)
    log(f"    val attention resnet mean {att['attn_resnet_mean']:.4f} std {att['attn_resnet_std']:.4f}"
        f" p05 {att['attn_resnet_p05']:.4f} p50 {att['attn_resnet_p50']:.4f} "
        f"p95 {att['attn_resnet_p95']:.4f}")
    log(f"    val attention effnet mean {att['attn_effnet_mean']:.4f} std {att['attn_effnet_std']:.4f}"
        f" p05 {att['attn_effnet_p05']:.4f} p50 {att['attn_effnet_p50']:.4f} "
        f"p95 {att['attn_effnet_p95']:.4f}")
    collapse = bool(max(att["attn_resnet_mean"], att["attn_effnet_mean"]) > 0.95)
    log(f"    attention collapse (mean > 0.95 for one branch): {collapse}")

    log("=== 12. extract 1024-d f_att for all 8862")
    dl_all = DataLoader(RGBDS(m.image_path, m.label, False), batch_size=32, shuffle=False,
                        num_workers=12, pin_memory=True)
    E = np.zeros((len(m), 1024), np.float32)
    i0 = 0
    with torch.no_grad():
        for x, _y in dl_all:
            E[i0:i0 + len(x)] = net.embed(x.to(dev)).cpu().numpy(); i0 += len(x)
    cols = [f"ropdeepx_att_{i:04d}" for i in range(1024)]
    EF = pd.DataFrame(E, columns=cols)
    EF.insert(0, "sample_id", ids)
    EF.insert(1, "group_id", m.group_id.values)
    EF.insert(2, "source", m.source.values)
    EF.insert(3, "split", m.split.values)
    EF.insert(4, "label", m.label.values)
    EF.to_parquet(OUT / "ropdeepx_attended_embeddings_8862.parquet", index=False)
    ok = EF[cols].shape == (8862, 1024) and bool(np.isfinite(EF[cols].values).all())
    log(f"    embeddings {EF[cols].shape} nan {int(EF[cols].isna().sum().sum())} "
        f"inf {int(np.isinf(EF[cols].values).sum())} finite {ok}")
    if not ok:
        raise SystemExit("EMBEDDING_INVALID")
    emb_sha = sha(OUT / "ropdeepx_attended_embeddings_8862.parquet")
    log(f"    embedding sha {emb_sha[:16]}")

    log("=== 13. M0 / M1 XGBoost on frozen features")
    import xgboost as xgb
    sp = m.split.values
    wtr = pd.Series(m.label.values[sp == "train"]).map(WEIGHTS_T6).values
    VE = pd.read_parquet(VES_EMB)
    vc = [c for c in VE.columns if c.startswith("vessel_emb_")]
    Z = np.hstack([E, VE.set_index("image_id").loc[ids, vc].values.astype(np.float32)])
    log(f"    vessel block {len(vc)}-d, combined M1 dim {Z.shape[1]}")
    feats = {"M0_ROPDEEPX_EMBEDDING_ONLY": E, "M1_ROPDEEPX_PLUS_VESSEL": Z}
    y = m.label.values
    ytr = y[sp == "train"]
    preds_te, preds_va, rows = {}, {}, []
    for nm, X in feats.items():
        clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                                random_state=SEED, n_jobs=16, tree_method="hist", **params)
        clf.fit(X[sp == "train"], ytr, sample_weight=wtr)
        clf.save_model(str(OUT / f"{nm}.json"))
        preds_te[nm] = clf.predict_proba(X[sp == "test"])
        preds_va[nm] = clf.predict_proba(X[sp == "val"])
        log(f"    {nm} dim {X.shape[1]} fit on {int((sp=='train').sum())} rows  sha "
            f"{sha(OUT / f'{nm}.json')[:16]}")

    frozen = {"TASK13_ALL_SELECTION_FROZEN": True, "TASK13_TEST_TOUCHED": False,
              "L_SELECTED_CHECKPOINT_FROZEN": True,
              "L_checkpoint_sha256": sel["checkpoint_sha256"],
              "L_selected_epoch": sel["selected_epoch"], "L_selected_val_auc": sel["val_auc"],
              "ropdeepx_embedding_sha256": emb_sha,
              "M0_frozen": True, "M1_frozen": True,
              "xgboost_config_unchanged": params, "batch_size": BATCH, "seed": SEED,
              "no_biomarkers": True}
    (OUT / "task13_selection_frozen.json").write_text(json.dumps(frozen, indent=2))
    log("TASK13_ALL_SELECTION_FROZEN = YES  TASK13_TEST_TOUCHED = NO -> TEST opens once")

    log("=== 16/17. TEST evaluation (single use)")
    tids = te.image_path.values
    yte = te.label.values
    dl_te = DataLoader(RGBDS(te.image_path, te.label, False), batch_size=32, shuffle=False,
                       num_workers=8, pin_memory=True)
    Pt, At = [], []
    with torch.no_grad():
        for x, _y in dl_te:
            o, a, _f = net(x.to(dev), want_att=True)
            Pt.append(torch.softmax(o.float(), 1).cpu().numpy()); At.append(a.cpu().numpy())
    PL = np.concatenate(Pt); AT = np.concatenate(At)
    preds_te["L_ROPDEEPX_STYLE_RGB"] = PL
    A = pd.read_csv(ART6 / "test_predictions_B_EMBEDDING_ONLY.csv").set_index("image_id")
    Bp = A.loc[tids, PCOLS].values
    e = pd.read_csv(ART8 / "test_predictions_E_RGB_VESSEL_FEATURE_FUSION.csv").set_index("image_id")
    g = pd.read_csv(ART8 / "test_predictions_G_RGB_VESSEL_SCALAR_FUSION.csv").set_index("image_id")
    PE, PG = e.loc[tids, PCOLS].values, g.loc[tids, PCOLS].values
    preds_te["B_EMBEDDING_ONLY"] = Bp
    preds_te["E_RGB_VESSEL_FEATURE_FUSION"] = PE
    preds_te["G_RGB_VESSEL_SCALAR_FUSION"] = PG
    for nm, P in preds_te.items():
        rows.append({**metrics(yte, P), "model": nm})
        pd.DataFrame(confusion_matrix(yte, P.argmax(1), labels=[0, 1, 2])).to_csv(
            OUT / f"confusion_{nm}.csv", index=False)
        if nm in feats or nm == "L_ROPDEEPX_STYLE_RGB":
            pr = pd.DataFrame({"image_id": tids, "group_id": te.group_id.values,
                               "source": te.source.values, "true_label": yte})
            pr["prob_Normal"], pr["prob_Pre_Plus"], pr["prob_Plus"] = P[:, 0], P[:, 1], P[:, 2]
            pr["predicted_label"] = P.argmax(1)
            pr.to_csv(OUT / f"test_predictions_{nm}.csv", index=False)
    R = pd.DataFrame(rows)[["model", "n"] + ALLKEYS]
    R.to_csv(OUT / "metrics_summary.csv", index=False)
    log(R[["model"] + KEYS].to_string(index=False))

    log("=== 10b. attention by true class and source (descriptive, post-freeze)")
    ar = []
    for k in (0, 1, 2):
        s = yte == k
        ar.append({"grouping": "true_class", "value": NAMES[k], "n": int(s.sum()),
                   "attn_resnet_mean": float(AT[s, 0].mean()), "attn_effnet_mean": float(AT[s, 1].mean())})
    for s_ in ("plus", "farfum_rop", "farabi"):
        s = te.source.values == s_
        ar.append({"grouping": "source", "value": s_, "n": int(s.sum()),
                   "attn_resnet_mean": float(AT[s, 0].mean()), "attn_effnet_mean": float(AT[s, 1].mean())})
    pd.DataFrame(ar).to_csv(OUT / "attention_by_class_and_source.csv", index=False)
    av = {"split": "test", "n": int(len(AT))}
    for j, nm in enumerate(("resnet", "effnet")):
        v = AT[:, j]
        av.update({f"attn_{nm}_mean": float(v.mean()), f"attn_{nm}_std": float(v.std()),
                   f"attn_{nm}_p05": float(np.percentile(v, 5)),
                   f"attn_{nm}_p50": float(np.percentile(v, 50)),
                   f"attn_{nm}_p95": float(np.percentile(v, 95))})
    pd.DataFrame([att, av]).to_csv(OUT / "attention_stats_val_test.csv", index=False)
    log(f"    test attention resnet mean {av['attn_resnet_mean']:.4f} "
        f"effnet mean {av['attn_effnet_mean']:.4f}")

    log("=== 22. source breakdown")
    srcs = te.source.values
    sb = []
    for s_ in ("plus", "farfum_rop", "farabi"):
        s = srcs == s_
        if s.sum() < 10:
            continue
        for nm, P in preds_te.items():
            sb.append({**metrics(yte[s], P[s]), "model": nm, "source": s_})
    pd.DataFrame(sb).to_csv(OUT / "source_breakdown.csv", index=False)

    log("=== 18/19. paired bootstrap 10k")
    comps = [("M1_minus_M0", preds_te["M1_ROPDEEPX_PLUS_VESSEL"],
              preds_te["M0_ROPDEEPX_EMBEDDING_ONLY"]),
             ("M0_minus_B", preds_te["M0_ROPDEEPX_EMBEDDING_ONLY"], Bp),
             ("M1_minus_E", preds_te["M1_ROPDEEPX_PLUS_VESSEL"], PE),
             ("M1_minus_G", preds_te["M1_ROPDEEPX_PLUS_VESSEL"], PG)]
    Pb = []
    for tag, PA, PB_ in comps:
        Pb.append(run_bootstrap(tag, yte, PA, PB_)); log(f"    {tag} done")
    BS = pd.concat(Pb, ignore_index=True)
    BS.to_csv(OUT / "paired_all_10k.csv", index=False)
    for tag in ("M1_minus_M0", "M0_minus_B", "M1_minus_E", "M1_minus_G"):
        BS[BS.comparison == tag].to_csv(OUT / f"paired_{tag}.csv", index=False)

    log("=== figures")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = pd.DataFrame(hist)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].plot(h.epoch, h.train_loss, "o-", ms=3, label="train")
        ax[0].plot(h.epoch, h.val_loss, "s-", ms=3, label="val")
        ax[0].axvline(3.5, color="gray", ls=":", lw=0.8)
        ax[0].axvline(sel["selected_epoch"], color="r", ls="--", lw=0.8)
        ax[0].legend(fontsize=6); ax[0].set_title("L loss (stage2 starts at 3.5)", fontsize=8)
        ax[1].plot(h.epoch, h.val_auc, "o-", ms=3, label="val AUC")
        ax[1].plot(h.epoch, h.val_macro_f1, "s-", ms=3, label="val macro F1")
        ax[1].legend(fontsize=6); ax[1].set_title("L validation", fontsize=8)
        ax[2].plot(h.epoch, h.attn_resnet_mean, "o-", ms=3, label="a_resnet")
        ax[2].plot(h.epoch, h.attn_effnet_mean, "s-", ms=3, label="a_effnet")
        ax[2].axhline(0.95, color="r", ls="--", lw=0.7); ax[2].axhline(0.05, color="r", ls="--", lw=0.7)
        ax[2].legend(fontsize=6); ax[2].set_title("attention weights", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "learning_curves_and_attention.png", dpi=170)
        plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for nm, P in preds_te.items():
            conf, corr, xs, ys_ = P.max(1), (P.argmax(1) == yte), [], []
            for lo in np.linspace(0, 1, 16)[:-1]:
                mm = (conf >= lo) & (conf < lo + 1 / 15)
                if mm.sum():
                    xs.append(conf[mm].mean()); ys_.append(corr[mm].mean())
            ax[0].plot(xs, ys_, marker="o", ms=3, label=nm[:16])
        ax[0].plot([0, 1], [0, 1], "k--", lw=0.7); ax[0].legend(fontsize=5)
        ax[0].set_title("reliability, 15 equal-width bins", fontsize=8)
        pp = BS[BS.metric.isin(KEYS)]
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

    minvl = min(hist, key=lambda r: r["val_loss"])
    maxauc = max(hist, key=lambda r: r["val_auc"])
    summary = {"TASK13_STATUS": "COMPLETE",
               "classification": "SECONDARY_POST_PRIMARY_EXPLORATORY_ARCHITECTURE_EXPERIMENT",
               "L_selection": {k: v for k, v in sel.items() if k != "history"},
               "frozen": frozen, "input_checks": checks,
               "metrics": R.to_dict("records"), "source_breakdown": sb,
               "paired_all_10k": BS.to_dict("records"),
               "attention_val": att, "attention_test": av, "attention_grouped": ar,
               "attention_collapse": collapse,
               "overfitting": {"selected_epoch": best["epoch"],
                               "min_val_loss_epoch": minvl["epoch"],
                               "min_val_loss": minvl["val_loss"],
                               "max_val_auc_epoch": maxauc["epoch"],
                               "max_val_auc": maxauc["val_auc"],
                               "epochs_run": len(hist)},
               "test_n": int(len(yte)), "test_used_for_selection": False}
    (OUT / "task13_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    shas = {p.name: sha(p) for p in sorted(OUT.glob("*"))
            if p.is_file() and p.name not in ("task13_progress.log", "train.log")}
    shas["task13_progress.log"] = "EXCLUDED_APPEND_ONLY_LOG"
    (OUT / "artifact_sha256.json").write_text(json.dumps(shas, indent=2))
    log("TASK13_STATUS = COMPLETE")


if __name__ == "__main__":
    main()
