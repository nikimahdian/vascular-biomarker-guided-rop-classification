"""Task 9 — joint RGB-vessel learned fusion with supportive biomarker branch (Server 2, authoritative).

H_JOINT_RGB_VESSEL              : frozen B_RGB EfficientNet-B5 (2048) + frozen D EfficientNet-B4 (1792)
                                  -> LayerNorm+Linear(512)+GELU+Dropout(0.20) per branch
                                  -> z = [r, v, r*v, |r-v|]  (2048)
                                  -> LayerNorm(2048)->Linear(2048,512)->GELU->Dropout(0.30)
                                     ->Linear(512,128)->GELU->Dropout(0.20)->Linear(128,3)
I_JOINT_RGB_VESSEL_BIOMARKER    : identical trunk; h(128) ++ biomarker32 (train-only standardised)
                                  -> Linear(160,3)

Staged joint finetuning: stage 1 freezes both encoders for 3 epochs at lr 3e-4, stage 2 unfreezes the
final two EfficientNet block groups + conv_head/bn2 of each encoder with differential lr (1e-5 encoder,
1e-4 new layers), max 25 epochs, patience 6. Weighted CE with the Task-6 train-only weights, seed 42,
batch 16, CUDA bfloat16 autocast. Selection on VAL only, frozen rule (AUC, macro F1, -loss, earlier).

Both H and I are predeclared; neither is conditional on the other. TEST is opened once, after
TASK9_ALL_SELECTION_FROZEN = YES. Task-6/7/8 artifacts are read-only throughout.
"""
import hashlib
import json
import random
import shutil
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
import torchvision.transforms.functional as TF
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

WS = Path("/root/niki_rop_task6_isolated")
ART6 = WS / "artifacts/task6"
ART8 = WS / "artifacts/task8_spatial_vessel_fusion"
OUT = WS / "artifacts/task9_joint_multimodal_fusion"
MANIFEST = WS / "primary_complete_case_v2_server2.csv"
S1 = "/Users/moniaz/niki"
BASE2 = "/root/niki_rop_task6_isolated/server1_data/Users/moniaz/niki"
B_CKPT = ART6 / "b_rgb_selected.pth"
D_CKPT = ART8 / "d_selected.pth"
B_SHA = "2b0b434cdd60233bc1cc0fa1549beeaafe0174cb6b9a5f546f5cbc6310f019d3"
D_SHA = "3e1719ebfa93d106be89c27a08f7974e0ec0f6cd4d6cc0b8f2bd140643ee5021"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
WEIGHTS = {0: 0.46051, 1: 3.18103, 2: 1.94512}
NAMES = ["Normal", "Pre_Plus", "Plus"]
PCOLS = ["prob_Normal", "prob_Pre_Plus", "prob_Plus"]
SEED = 42
NB = 10000
NCHUNK = 40
BATCH = 16
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
KEYS = ["multiclass_auc", "balanced_accuracy", "macro_f1", "brier", "ece"]
T0 = time.time()
BF16_USED = []
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "task9_progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


# ---------------------------------------------------------------- metrics (Task-6/7/8 definitions)
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


ALLKEYS = KEYS + ["auc_Normal", "auc_Pre_Plus", "auc_Plus"]


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
    return pd.DataFrame([{ "comparison": tag, "metric": k, "delta": obs[k],
                           "ci95_lo": float(np.percentile(D[:, j], 2.5)),
                           "ci95_hi": float(np.percentile(D[:, j], 97.5)),
                           "p_two_sided_null_centered":
                               float(np.mean(np.abs(D[:, j] - obs[k]) >= abs(obs[k]))),
                           "ci_crosses_zero": bool(np.percentile(D[:, j], 2.5) <= 0
                                                   <= np.percentile(D[:, j], 97.5)),
                           "replicates": NB, "seed": SEED}
                         for j, k in enumerate(ALLKEYS)])


# ---------------------------------------------------------------- data
class JointDS(Dataset):
    def __init__(self, rgb, msk, bio, y, train):
        self.rgb, self.msk, self.bio, self.y, self.train = list(rgb), list(msk), bio, list(y), train
        self.jit = T.ColorJitter(0.10, 0.10, 0.10, 0.0)
        self.norm = T.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        r = Image.open(self.rgb[i]).convert("RGB")
        k = Image.open(self.msk[i]).convert("L")
        if self.train:
            a, b, h, w = T.RandomResizedCrop.get_params(r, scale=(0.95, 1.0), ratio=(1.0, 1.0))
            r = TF.resized_crop(r, a, b, h, w, [384, 384], IM.BILINEAR)
            k = TF.resized_crop(k, a, b, h, w, [384, 384], IM.NEAREST)
            if random.random() < 0.5:
                r, k = TF.hflip(r), TF.hflip(k)
            ang = random.uniform(-10.0, 10.0)
            r = TF.rotate(r, ang, interpolation=IM.BILINEAR, fill=0)
            k = TF.rotate(k, ang, interpolation=IM.NEAREST, fill=0)
            r = self.jit(r)
        else:
            r = TF.resize(r, [384, 384], IM.BILINEAR)
            k = TF.resize(k, [384, 384], IM.NEAREST)
        xr = self.norm(TF.to_tensor(r))
        xk = (TF.to_tensor(k) > 0.5).float().repeat(3, 1, 1)
        return xr, xk, torch.from_numpy(self.bio[i]), self.y[i]


# ---------------------------------------------------------------- model
class Joint(nn.Module):
    def __init__(self, use_bio):
        super().__init__()
        import timm
        self.use_bio = use_bio
        self.rgb = timm.create_model("efficientnet_b5", pretrained=False, num_classes=0,
                                     global_pool="avg")
        self.ves = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0,
                                     global_pool="avg")
        b = torch.load(B_CKPT, map_location="cpu")
        self.rgb.load_state_dict(b["backbone"])
        d = torch.load(D_CKPT, map_location="cpu")
        self.ves.load_state_dict(d["backbone"])
        self.rgb_proj = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 512), nn.GELU(),
                                      nn.Dropout(0.20))
        self.ves_proj = nn.Sequential(nn.LayerNorm(1792), nn.Linear(1792, 512), nn.GELU(),
                                      nn.Dropout(0.20))
        self.trunk = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 512), nn.GELU(),
                                   nn.Dropout(0.30), nn.Linear(512, 128), nn.GELU(),
                                   nn.Dropout(0.20))
        if use_bio:
            self.bio = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.LayerNorm(32))
            self.cls = nn.Linear(160, 3)
        else:
            self.bio = None
            self.cls = nn.Linear(128, 3)

    def encode(self, xr, xk):
        r = self.rgb_proj(self.rgb(xr))
        v = self.ves_proj(self.ves(xk))
        return self.trunk(torch.cat([r, v, r * v, torch.abs(r - v)], 1))

    def forward(self, xr, xk, xb):
        h = self.encode(xr, xk)
        if self.use_bio:
            h = torch.cat([h, self.bio(xb)], 1)
        return self.cls(h)


def new_params(model):
    return [p for n, p in model.named_parameters()
            if not (n.startswith("rgb.") or n.startswith("ves."))]


def encoder_tail_params(model):
    out = []
    for tag in ("rgb", "ves"):
        enc = getattr(model, tag)
        for blk in list(enc.blocks)[-2:]:
            out += list(blk.parameters())
        out += list(enc.conv_head.parameters()) + list(enc.bn2.parameters())
    return out


def set_encoder_grad(model, on, tail_only):
    for tag in ("rgb", "ves"):
        enc = getattr(model, tag)
        for p in enc.parameters():
            p.requires_grad = False
        if on:
            for blk in list(enc.blocks)[-2:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in list(enc.conv_head.parameters()) + list(enc.bn2.parameters()):
                p.requires_grad = True


def train_model(name, use_bio, m, bio_all, dev):
    log(f"=== {name}  use_bio={use_bio}")
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    tr = m[m.split == "train"].reset_index(drop=True)
    va = m[m.split == "val"].reset_index(drop=True)
    itr = m.index[m.split == "train"].values
    iva = m.index[m.split == "val"].values
    dl_tr = DataLoader(JointDS(tr.image_path, tr.mask_local, bio_all[itr], tr.label, True),
                       batch_size=BATCH, shuffle=True, num_workers=12, pin_memory=True,
                       persistent_workers=True, drop_last=False)
    dl_va = DataLoader(JointDS(va.image_path, va.mask_local, bio_all[iva], va.label, False),
                       batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    model = Joint(use_bio).to(dev)
    w = torch.tensor([WEIGHTS[0], WEIGHTS[1], WEIGHTS[2]], dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    yv = va.label.values
    hist_p = OUT / f"{name}_history.json"
    hist = json.loads(hist_p.read_text()) if hist_p.exists() else []
    best, best_idx = None, -1
    for j, h in enumerate(hist):  # resume support: rebuild the running best from the saved history
        if best is None or (h["val_auc"], h["val_macro_f1"], -h["val_loss"], -h["epoch"]) > \
                (best["val_auc"], best["val_macro_f1"], -best["val_loss"], -best["epoch"]):
            best, best_idx = h, j
    stage = 1
    set_encoder_grad(model, False, False)
    opt = torch.optim.AdamW([{"params": new_params(model), "lr": 3e-4}], weight_decay=1e-4)
    log(f"    stage1: encoders frozen, lr 3e-4, 3 epochs   (resume from epoch {len(hist)})")

    for idx in range(len(hist), 28):
        if idx >= 3 and stage == 1:
            stage = 2
            set_encoder_grad(model, True, True)
            opt = torch.optim.AdamW([{"params": encoder_tail_params(model), "lr": 1e-5},
                                     {"params": new_params(model), "lr": 1e-4}], weight_decay=1e-4)
            log(f"    stage2: last two block groups + conv_head/bn2 unfrozen, lr 1e-5/1e-4, "
                f"max 25 epochs, patience 6")
        t_ep = time.time()
        model.train(); tl = 0.0
        for xr, xk, xb, y in dl_tr:
            xr = xr.to(dev, non_blocking=True); xk = xk.to(dev, non_blocking=True)
            xb = xb.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(xr, xk, xb)
                BF16_USED.append(1)
            else:
                out = model(xr, xk, xb)
            loss = lossf(out.float(), y)
            loss.backward(); opt.step()
            tl += float(loss.item()) * len(y)
        tl /= len(tr)
        model.eval(); vl, P = 0.0, []
        with torch.no_grad():
            for xr, xk, xb, y in dl_va:
                xr = xr.to(dev); xk = xk.to(dev); xb = xb.to(dev)
                out = model(xr, xk, xb)
                vl += float(lossf(out.float(), y.to(dev)).item()) * len(y)
                P.append(torch.softmax(out.float(), 1).cpu().numpy())
        vl /= len(va); P = np.concatenate(P)
        auc, f1 = macro_auc(yv, P), float(f1_score(yv, P.argmax(1), average="macro"))
        ck = OUT / f"{name}_stage{stage}_epoch{idx+1}.pth"
        torch.save({"model": model.state_dict(), "stage": stage, "epoch": idx + 1,
                    "use_bio": use_bio}, ck)
        hist.append({"epoch": idx + 1, "stage": stage, "train_loss": tl, "val_loss": vl,
                     "val_auc": auc, "val_macro_f1": f1, "seconds": round(time.time() - t_ep, 1),
                     "ckpt": str(ck)})
        hist_p.write_text(json.dumps(hist, indent=2))
        np.save(OUT / f"{name}_val_probs_epoch{idx+1}.npy", P)
        log(f"    {name} s{stage} epoch {idx+1:2d} {time.time()-t_ep:6.1f}s train {tl:.4f} "
            f"val {vl:.4f} auc {auc:.5f} f1 {f1:.4f}")
        key = (auc, f1, -vl, -(idx + 1))
        bk = (best["val_auc"], best["val_macro_f1"], -best["val_loss"], -(best["epoch"])) \
            if best else None
        if bk is None or key > bk:
            best, best_idx = hist[-1], idx
        if stage == 2 and idx - best_idx >= 6:
            log(f"    {name} early stop at epoch {idx+1} (best epoch {best['epoch']} s{best['stage']})")
            break
    shutil.copy(best["ckpt"], OUT / f"{name}_selected.pth")
    sel = {"model": name, "selected_epoch": best["epoch"], "selected_stage": best["stage"],
           "val_auc": best["val_auc"], "val_macro_f1": best["val_macro_f1"],
           "val_loss": best["val_loss"], "checkpoint_sha256": sha(OUT / f"{name}_selected.pth"),
           "epochs_run": len(hist), "use_bio": use_bio, "history": hist, "test_touched": False}
    (OUT / f"{name}_selection.json").write_text(json.dumps(sel, indent=2))
    log(f"    {name} SELECTED epoch {best['epoch']} stage {best['stage']} "
        f"val_auc {best['val_auc']:.5f} sha {sel['checkpoint_sha256'][:16]}")
    return sel


def infer(model, dl, dev):
    model.eval(); P = []
    with torch.no_grad():
        for xr, xk, xb, _y in dl:
            out = model(xr.to(dev), xk.to(dev), xb.to(dev))
            P.append(torch.softmax(out.float(), 1).cpu().numpy())
    return np.concatenate(P)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {dev}  bf16 available {torch.cuda.is_bf16_supported() if dev.type=='cuda' else False}")

    log("=== 0. PRE-FLIGHT")
    m = pd.read_csv(MANIFEST, low_memory=False)
    m["mask_local"] = m.mask_path.astype(str).str.replace(S1, BASE2, regex=False)
    checks = {"population_8862": len(m) == 8862,
              "split_counts": (m.split.value_counts().get("train") == 6203
                               and m.split.value_counts().get("val") == 1328
                               and m.split.value_counts().get("test") == 1331),
              "no_group_overlap": bool((m.groupby("group_id").split.nunique() == 1).all()),
              "labels_0_1_2": set(m.label.unique()) == {0, 1, 2},
              "biomarkers_finite": bool(m[FEATS].notna().all().all()),
              "rgb_exist": bool(m.image_path.map(lambda p: Path(p).exists()).all()),
              "masks_exist": bool(m.mask_local.map(lambda p: Path(p).exists()).all()),
              "B_ckpt_sha": sha(B_CKPT) == B_SHA, "D_ckpt_sha": sha(D_CKPT) == D_SHA}
    for k, v in checks.items():
        log(f"    {k:22s} : {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit("PREFLIGHT_FAILED")

    btr = m[m.split == "train"]
    bmu, bsd = btr[FEATS].mean().values, btr[FEATS].std(ddof=0).values
    bio_all = ((m[FEATS].values - bmu) / bsd).astype(np.float32)
    json.dump({"features": FEATS, "mean": bmu.tolist(), "std": bsd.tolist(),
               "fitted_on": "TRAIN only", "n_train": int(len(btr))},
              open(OUT / "biomarker_scaler.json", "w"), indent=2)
    log(f"    scaler train-only  mean {np.round(bmu,4).tolist()}  std {np.round(bsd,4).tolist()}")

    selH = train_model("H_JOINT_RGB_VESSEL", False, m, bio_all, dev)
    selI = train_model("I_JOINT_RGB_VESSEL_BIOMARKER", True, m, bio_all, dev)

    frozen = {"TASK9_ALL_SELECTION_FROZEN": True,
              "H_checkpoint_sha256": selH["checkpoint_sha256"],
              "I_checkpoint_sha256": selI["checkpoint_sha256"],
              "H_selected": {k: selH[k] for k in ("selected_epoch", "selected_stage", "val_auc",
                                                  "val_macro_f1", "val_loss")},
              "I_selected": {k: selI[k] for k in ("selected_epoch", "selected_stage", "val_auc",
                                                  "val_macro_f1", "val_loss")},
              "biomarker_scaler_sha256": sha(OUT / "biomarker_scaler.json"),
              "architecture_frozen": True, "ensemble_executed": False,
              "ensemble_declaration": "section 17 not executed: runtime was not low enough to add "
                                      "a 3-seed ensemble after H and I",
              "bf16_autocast_used": bool(BF16_USED)}
    (OUT / "task9_selection_frozen.json").write_text(json.dumps(frozen, indent=2))
    log("TASK9_ALL_SELECTION_FROZEN = YES -> TEST opens once")

    te = m[m.split == "test"].reset_index(drop=True)
    ite = m.index[m.split == "test"].values
    dl_te = DataLoader(JointDS(te.image_path, te.mask_local, bio_all[ite], te.label, False),
                       batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    y = te.label.values
    preds = {}
    for nm, use_bio, sel in (("H_JOINT_RGB_VESSEL", False, selH),
                             ("I_JOINT_RGB_VESSEL_BIOMARKER", True, selI)):
        mod = Joint(use_bio).to(dev)
        mod.load_state_dict(torch.load(OUT / f"{nm}_selected.pth", map_location="cpu")["model"])
        preds[nm] = infer(mod, dl_te, dev)
        log(f"    {nm} TEST probabilities {preds[nm].shape}")

    e = pd.read_csv(ART8 / "test_predictions_E_RGB_VESSEL_FEATURE_FUSION.csv").set_index("image_id")
    g = pd.read_csv(ART8 / "test_predictions_G_RGB_VESSEL_SCALAR_FUSION.csv").set_index("image_id")
    ids = te.image_path.values
    PE, PG = e.loc[ids, PCOLS].values, g.loc[ids, PCOLS].values
    M = {nm: metrics(y, P) for nm, P in preds.items()}
    M["E_RGB_VESSEL_FEATURE_FUSION"] = metrics(y, PE)
    M["G_RGB_VESSEL_SCALAR_FUSION"] = metrics(y, PG)
    rows = []
    for nm, P in list(preds.items()) + [("E_RGB_VESSEL_FEATURE_FUSION", PE),
                                        ("G_RGB_VESSEL_SCALAR_FUSION", PG)]:
        pr = pd.DataFrame({"image_id": ids, "group_id": te.group_id.values,
                           "source": te.source.values, "true_label": y})
        pr["prob_Normal"], pr["prob_Pre_Plus"], pr["prob_Plus"] = P[:, 0], P[:, 1], P[:, 2]
        pr["predicted_label"] = P.argmax(1)
        if nm.startswith(("H_", "I_")):
            pr.to_csv(OUT / f"test_predictions_{nm}.csv", index=False)
        rows.append({**{k: v for k, v in M[nm].items()}, "model": nm, "n": int(len(y))})
        pd.DataFrame(confusion_matrix(y, P.argmax(1), labels=[0, 1, 2])).to_csv(
            OUT / f"confusion_{nm}.csv", index=False)
    R = pd.DataFrame(rows)[["model", "n"] + ALLKEYS]
    R.to_csv(OUT / "metrics_summary.csv", index=False)
    log(R[["model"] + KEYS].to_string(index=False))

    log("=== source breakdown")
    srows = []
    for src in ("plus", "farfum_rop", "farabi"):
        s = (te.source == src).values
        if s.sum() < 10:
            continue
        for nm, P in list(preds.items()) + [("E_RGB_VESSEL_FEATURE_FUSION", PE),
                                            ("G_RGB_VESSEL_SCALAR_FUSION", PG)]:
            srows.append({**metrics(y[s], P[s]), "model": nm, "source": src, "n": int(s.sum())})
    pd.DataFrame(srows).to_csv(OUT / "source_breakdown.csv", index=False)

    log("=== paired bootstrap 10k")
    obs = {}
    comps = [("H_minus_E", preds["H_JOINT_RGB_VESSEL"], PE),
             ("I_minus_H", preds["I_JOINT_RGB_VESSEL_BIOMARKER"],
              preds["H_JOINT_RGB_VESSEL"]),
             ("H_minus_G", preds["H_JOINT_RGB_VESSEL"], PG),
             ("I_minus_G", preds["I_JOINT_RGB_VESSEL_BIOMARKER"], PG)]
    Pb = []
    for tag, PA, PB in comps:
        o = {k: metrics(y, PA)[k] - metrics(y, PB)[k] for k in ALLKEYS}
        obs[tag] = o
        Pb.append(run_bootstrap(tag, y, PA, PB, o))
        log(f"    {tag} done")
    PBdf = pd.concat(Pb, ignore_index=True)
    PBdf.to_csv(OUT / "paired_all_10k.csv", index=False)
    for tag in obs:
        PBdf[PBdf.comparison == tag].to_csv(OUT / f"paired_{tag}.csv", index=False)

    log("=== figures")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
        for j, (nm, sel) in enumerate((("H", selH), ("I", selI))):
            h = pd.DataFrame(sel["history"])
            ax[0][j].plot(h.epoch, h.train_loss, "o-", ms=3, label="train")
            ax[0][j].plot(h.epoch, h.val_loss, "s-", ms=3, label="val")
            ax[0][j].axvline(3.5, color="gray", ls=":", lw=0.8)
            ax[0][j].axvline(sel["selected_epoch"], color="r", ls="--", lw=0.8)
            ax[0][j].set_title(f"{nm} loss (stage2 starts at 3.5)", fontsize=8)
            ax[0][j].legend(fontsize=7)
            ax[1][j].plot(h.epoch, h.val_auc, "o-", ms=3, label="val AUC")
            ax[1][j].plot(h.epoch, h.val_macro_f1, "s-", ms=3, label="val macro F1")
            ax[1][j].set_title(f"{nm} val AUC / macro F1", fontsize=8)
            ax[1][j].legend(fontsize=7)
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
        pp = PBdf[(PBdf.comparison.isin(["H_minus_E", "I_minus_H", "H_minus_G", "I_minus_G"]))
                  & (PBdf.metric.isin(KEYS))]
        labs = (pp.comparison + " " + pp.metric).tolist()
        ypos = np.arange(len(labs))
        ax[1].errorbar(pp.delta, ypos, xerr=[pp.delta - pp.ci95_lo, pp.ci95_hi - pp.delta],
                       fmt="o", ms=3)
        ax[1].axvline(0, color="k", lw=0.8, ls="--")
        ax[1].set_yticks(ypos); ax[1].set_yticklabels(labs, fontsize=5)
        ax[1].set_title("paired delta, 95% CI", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "calibration_and_forest.png", dpi=170); plt.close(fig)
        log("    figures written")
    except Exception as ex:  # noqa: BLE001
        log(f"    figures skipped: {ex}")

    summary = {"TASK9_STATUS": "COMPLETE", "classification":
               "SECONDARY_POST_PRIMARY_EXPLORATORY_CANONICAL_SPLIT_REANALYSIS",
               "H_selection": {k: v for k, v in selH.items() if k != "history"},
               "I_selection": {k: v for k, v in selI.items() if k != "history"},
               "frozen": frozen, "metrics": R.to_dict("records"),
               "source_breakdown": srows, "paired_all_10k": PBdf.to_dict("records"),
               "test_n": int(len(y)), "test_used_for_selection": False,
               "bf16_used": bool(BF16_USED)}
    (OUT / "task9_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    shas = {p.name: sha(p) for p in sorted(OUT.glob("*")) if p.is_file()}
    (OUT / "artifact_sha256.json").write_text(json.dumps(shas, indent=2))
    log("TASK9_STATUS = COMPLETE")


if __name__ == "__main__":
    main()
