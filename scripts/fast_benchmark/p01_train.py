"""Part 2 - harmonized FARFUM 3-fold benchmark trainers.

  python p01_train.py legacy   <fold>   # previous-lab EfficientNet-B5 @224, 3-class head
  python p01_train.py ropdeepx <fold>   # ROPDeepX ResNet50 + EfficientNet-B4 soft-attention fusion
  python p01_train.py collect           # merge the three folds into the two OOF tables

Both use the Prompt-1 patient-level folds and the Prompt-1 internal validation split,
so held-out patients are identical across every comparator.
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

WS = Path("/root/niki_rop_task6_isolated")
OUT = WS / "results/fast_benchmark"
CK = WS / "_bench/ckpt"
OUT.mkdir(parents=True, exist_ok=True)
CK.mkdir(parents=True, exist_ok=True)
SEED = 42
NAMES = ["Normal", "Pre_Plus", "Plus"]
WEIGHTS = {0: 0.46051, 1: 3.18103, 2: 1.94512}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BATCH = 16
SMOKE = bool(os.environ.get("BENCH_SMOKE"))
T0 = time.time()


def log(msg):
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "p01_train.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def auc_macro(y, P):
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro"))


def macro_f1(y, P):
    return float(f1_score(y, P.argmax(1), average="macro"))


class DS(Dataset):
    def __init__(self, paths, labels, tf):
        self.p, self.y, self.tf = list(paths), np.asarray(labels), tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), int(self.y[i])


# ------------------------------------------------------------------ splits
EXPECTED_DEV = {0: (771, 268), 1: (793, 246), 2: (736, 242)}


def fold_split(m, k):
    """Exactly the Prompt-1 split: the exported devsplit_fold{k}.csv from server 1.

    The internal train/val split is materialised by the host that produced Prompt 1 so
    that the StratifiedGroupKFold version difference between hosts cannot move a single
    image between training and internal validation.
    """
    te = m[m.fold == k].reset_index(drop=True)
    ds = pd.read_csv(OUT / f"devsplit_fold{k}.csv")[["image_id", "role"]]
    mm = m.merge(ds, on="image_id", how="left")
    tr = mm[mm.role == "train"].drop(columns=["role"]).reset_index(drop=True)
    va = mm[mm.role == "val"].drop(columns=["role"]).reset_index(drop=True)
    assert len(tr) + len(va) == len(m) - len(te), "dev split does not cover the development folds"
    assert (len(tr), len(va)) == EXPECTED_DEV[k], f"dev split mismatch fold {k}: {len(tr)}/{len(va)}"
    assert set(tr.patient_id) & set(va.patient_id) == set()
    assert set(tr.patient_id) & set(te.patient_id) == set()
    return tr, va, te


# ------------------------------------------------------------------ models
class LegacyB5(nn.Module):
    """Previous laboratory architecture: timm backbone -> Dropout(0.5) -> Linear."""

    def __init__(self, n_classes=3, backbone="efficientnet_b5", drop=0.5):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0,
                                          global_pool="avg")
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(self.backbone.num_features, n_classes)

    def forward(self, x):
        return self.classifier(self.dropout(self.backbone(x)))


class ROPDeepX(nn.Module):
    """ResNet50 + EfficientNet-B4 -> 1024-d projections -> soft-attention fusion -> 3 classes.

    Architecture per the ROPDeepX paper description (EfficientNet-B4 and ResNet-50 features
    projected to 1024 dimensions, concatenated, refined by a soft attention-based fusion) and
    the repository's earlier paper-derived implementation (task13_ropdeepx_style.py).
    """

    def __init__(self, n_classes=3, proj=1024, drop=0.30):
        super().__init__()
        self.r50 = timm.create_model("resnet50", pretrained=True, num_classes=0, global_pool="avg")
        self.b4 = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0,
                                    global_pool="avg")
        self.proj_r = nn.Sequential(nn.Linear(self.r50.num_features, proj), nn.BatchNorm1d(proj),
                                    nn.ReLU(inplace=True))
        self.proj_e = nn.Sequential(nn.Linear(self.b4.num_features, proj), nn.BatchNorm1d(proj),
                                    nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Linear(2 * proj, proj), nn.BatchNorm1d(proj),
                                    nn.ReLU(inplace=True), nn.Dropout(drop))
        self.attn = nn.Linear(proj, 2)
        self.head = nn.Linear(proj, n_classes)

    def forward(self, x):
        zr = self.proj_r(self.r50(x))
        ze = self.proj_e(self.b4(x))
        f = self.fusion(torch.cat([zr, ze], dim=1))
        a = torch.softmax(self.attn(f), dim=1)
        f_att = a[:, 0:1] * zr + a[:, 1:2] * ze
        return self.head(f_att)


# ------------------------------------------------------------------ training
def run_legacy(tr, va, te, device, fold):
    tf_tr = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_ev = tf_tr
    model = LegacyB5().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.2, patience=3)
    crit = nn.CrossEntropyLoss()                      # as in the laboratory notebook
    dl_tr = DataLoader(DS(tr.image_path_s2, tr.label, tf_tr), batch_size=BATCH, shuffle=True,
                       num_workers=8, pin_memory=True)
    dl_va = DataLoader(DS(va.image_path_s2, va.label, tf_ev), batch_size=32, shuffle=False,
                       num_workers=8, pin_memory=True)
    hist, best, best_state, no_improve, best_loss = [], None, None, 0, None
    for ep in range(1, (2 if SMOKE else 100) + 1):     # notebook allowed 100 epochs
        model.train(); tl, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tl += float(loss.item()) * len(y); n += len(y)
        tl /= max(n, 1)
        model.eval(); vl, P = 0.0, []
        with torch.no_grad():
            for x, y in dl_va:
                out = model(x.to(device))
                vl += float(crit(out, y.to(device)).item()) * len(y)
                P.append(torch.softmax(out, 1).cpu().numpy())
        vl /= len(va); P = np.concatenate(P)
        auc, f1 = auc_macro(va.label.values, P), macro_f1(va.label.values, P)
        sched.step(vl)
        hist.append({"epoch": ep, "train_loss": tl, "val_loss": vl, "val_auc": auc, "val_f1": f1,
                     "lr": opt.param_groups[0]["lr"]})
        log(f"legacy fold{fold} ep{ep:03d} train {tl:.4f} val {vl:.4f} auc {auc:.5f} f1 {f1:.4f}")
        if best is None or (f1, auc, -vl, -ep) > (best["val_f1"], best["val_auc"], -best["val_loss"],
                                                  -best["epoch"]):
            best = hist[-1]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_loss is None or vl < best_loss - 1e-6:
            best_loss, no_improve = vl, 0
        else:
            no_improve += 1
        if no_improve >= (99 if SMOKE else 10):         # notebook: patience 10 on val loss
            log(f"legacy fold{fold} early stop at ep{ep} (best f1 ep{best['epoch']})")
            break
    torch.save(best_state, CK / f"legacy_fold{fold}.pth")
    (CK / f"legacy_fold{fold}_selection.json").write_text(json.dumps(
        {"arch": "LEGACY_B5_3CLASS", "fold": fold, "selected": best, "history": hist}, indent=1))
    log(f"legacy fold{fold} SELECTED ep{best['epoch']} val_f1 {best['val_f1']:.5f} "
        f"val_auc {best['val_auc']:.5f}")
    return predict(model, best_state, te, tf_ev, device)


def run_ropdeepx(tr, va, te, device, fold):
    tf_tr = T.Compose([T.RandomResizedCrop(384, scale=(0.95, 1.0), ratio=(1.0, 1.0)),
                       T.RandomHorizontalFlip(0.5), T.RandomRotation(10),
                       T.ColorJitter(0.10, 0.10, 0.10, 0.0),
                       T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_ev = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize(MEAN, STD)])
    model = ROPDeepX().to(device)
    w = torch.tensor([WEIGHTS[0], WEIGHTS[1], WEIGHTS[2]], dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    dl_tr = DataLoader(DS(tr.image_path_s2, tr.label, tf_tr), batch_size=BATCH, shuffle=True,
                       num_workers=8, pin_memory=True)
    dl_va = DataLoader(DS(va.image_path_s2, va.label, tf_ev), batch_size=32, shuffle=False,
                       num_workers=8, pin_memory=True)
    hist, best, best_state = [], None, None
    # stage 1 - frozen backbones, 3 epochs
    for p in list(model.r50.parameters()) + list(model.b4.parameters()):
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4,
                            weight_decay=5e-4)
    for ep in range(1, (2 if SMOKE else 4)):
        model.train(); tl, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tl += float(loss.item()) * len(y); n += len(y)
        log(f"ropdeepx fold{fold} stage1 ep{ep} train {tl / max(n, 1):.4f}")
    # stage 2 - unfreeze resnet layer4 + last efficientnet block group, OneCycleLR
    for p in model.r50.layer4.parameters():
        p.requires_grad = True
    for name, p in model.b4.named_parameters():
        if name.startswith(("blocks.6", "conv_head", "bn2")):
            p.requires_grad = True
    groups = [{"params": [p for n_, p in model.r50.named_parameters() if p.requires_grad],
               "lr": 5e-6},
              {"params": [p for n_, p in model.b4.named_parameters() if p.requires_grad],
               "lr": 5e-5},
              {"params": [p for n_, p in model.named_parameters()
                          if p.requires_grad and not n_.startswith(("r50.", "b4."))], "lr": 3e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=5e-4)
    # Schedule choice: the first harmonized run used OneCycleLR(20 epochs); with early stopping that
    # schedule stopped inside its own LR warm-up (selected epoch 2), which handicaps the comparator
    # for a scheduling reason rather than for an architectural one. The primary run therefore uses a
    # plateau-decayed constant schedule -- declared as a deviation in 02_ropdeepx_protocol.md.
    if os.environ.get("ROPDEEPX_SCHED", "plateau") == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in groups],
            total_steps=(2 if SMOKE else 20) * max(1, len(dl_tr)), pct_start=0.3)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.3, patience=2)
    MAX_EP = 2 if SMOKE else 30
    no_improve = 0
    for ep in range(1, MAX_EP + 1):                     # plateau schedule: <=30 epochs, patience 8
        model.train(); tl, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tl += float(loss.item()) * len(y); n += len(y)
        tl /= max(n, 1)
        model.eval(); vl, P = 0.0, []
        with torch.no_grad():
            for x, y in dl_va:
                out = model(x.to(device))
                vl += float(nn.functional.cross_entropy(out, y.to(device)).item()) * len(y)
                P.append(torch.softmax(out, 1).cpu().numpy())
        vl /= len(va); P = np.concatenate(P)
        auc, f1 = auc_macro(va.label.values, P), macro_f1(va.label.values, P)
        hist.append({"epoch": ep, "train_loss": tl, "val_loss": vl, "val_auc": auc, "val_f1": f1})
        log(f"ropdeepx fold{fold} ep{ep:02d} train {tl:.4f} val {vl:.4f} auc {auc:.5f} f1 {f1:.4f}")
        if best is None or (auc, f1, -vl, -ep) > (best["val_auc"], best["val_f1"], -best["val_loss"],
                                                 -best["epoch"]):
            best = hist[-1]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if os.environ.get("ROPDEEPX_SCHED", "plateau") == "plateau":
            sched.step(vl)
        if no_improve >= (99 if SMOKE else (5 if os.environ.get("ROPDEEPX_SCHED", "plateau") == "onecycle" else 8)):
            log(f"ropdeepx fold{fold} early stop at ep{ep} (best ep{best['epoch']})")
            break
    torch.save(best_state, CK / f"ropdeepx_fold{fold}.pth")
    (CK / f"ropdeepx_fold{fold}_selection.json").write_text(json.dumps(
        {"arch": "ROPDEEPX_HARMONIZED", "fold": fold, "selected": best, "history": hist}, indent=1))
    log(f"ropdeepx fold{fold} SELECTED ep{best['epoch']} val_auc {best['val_auc']:.5f}")
    return predict(model, best_state, te, tf_ev, device)


def predict(model, state, te, tf_ev, device):
    model.load_state_dict(state)
    model.to(device).eval()
    dl = DataLoader(DS(te.image_path_s2, te.label, tf_ev), batch_size=32, shuffle=False,
                    num_workers=8, pin_memory=True)
    P = []
    with torch.no_grad():
        for x, _y in dl:
            P.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    return np.concatenate(P)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    arch, fold = sys.argv[1], int(sys.argv[2])
    device = torch.device("cuda")
    m = pd.read_csv(OUT / "02_manifest_server2.csv")
    tr, va, te = fold_split(m, fold)
    if SMOKE:
        tr = pd.concat([g.head(12) for _, g in tr.groupby("label")]).reset_index(drop=True)
        va = pd.concat([g.head(6) for _, g in va.groupby("label")]).reset_index(drop=True)
        te = pd.concat([g.head(6) for _, g in te.groupby("label")]).reset_index(drop=True)
    log(f"{arch} fold{fold}: train {len(tr)} ({tr.patient_id.nunique()} pat) "
        f"val {len(va)} ({va.patient_id.nunique()} pat) test {len(te)} ({te.patient_id.nunique()} pat)")
    if arch == "legacy":
        P = run_legacy(tr, va, te, device, fold)
    elif arch == "ropdeepx":
        P = run_ropdeepx(tr, va, te, device, fold)
    else:
        raise SystemExit("arch must be legacy|ropdeepx")
    out = pd.DataFrame({"image_id": te.image_id, "patient_id": te.patient_id, "fold": fold,
                        "true_label": te.label})
    for c in range(3):
        out[f"{'Legacy' if arch == 'legacy' else 'ROPDeepX'}_p{c}"] = P[:, c]
    name = "01_legacy_oof_fold%d.csv" % fold if arch == "legacy" else "03_ropdeepx_oof_fold%d.csv" % fold
    out.to_csv(OUT / name, index=False)
    log(f"{arch} fold{fold} test AUC {auc_macro(te.label.values, P):.5f} -> {name}")
    print(f"{arch.upper()}_FOLD{fold}_DONE")


if __name__ == "__main__":
    main()
