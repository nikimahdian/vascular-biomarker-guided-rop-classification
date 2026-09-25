"""Step 4 - one outer fold of the fast patient-level 3-fold sensitivity analysis.

Usage:  python s04_fold_run.py <fold 0|1|2>

Per fold:
  * outer test   = the fold's patients
  * development  = the other two folds' patients
  * development is split once, patient-level, into outer-train / internal-val
  * EfficientNet-B5 RGB encoder  trained on outer-train only (frozen Task-6 recipe)
  * EfficientNet-B4 vessel encoder trained on outer-train only (frozen Task-8 recipe)
  * embeddings extracted for all 1528 FARFUM images with the fold's checkpoints
  * B / C / E / G XGBoost fitted on outer-train only with the frozen shared config
  * outer-test probabilities saved
No hyper-parameter search, no test-driven selection.
"""
import json
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
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode as IM

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "results/fast_examiner"
CK = ROOT / "_fast_examiner/ckpt"
RUN = ROOT / "_fast_examiner"
OUT.mkdir(parents=True, exist_ok=True)
CK.mkdir(parents=True, exist_ok=True)

SEED = 42
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
WEIGHTS = {0: 0.46051, 1: 3.18103, 2: 1.94512}          # frozen Task-6 train-only weights
XGB_PARAMS = {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
B5_EPOCHS, B5_PATIENCE, B5_LR, B5_DROP = 40, 8, 1e-5, 0.2
B4_EPOCHS, B4_PATIENCE, B4_LR, B4_DROP = 30, 6, 1e-4, 0.3
SMOKE = bool(os.environ.get("FAST_SMOKE"))
if SMOKE:
    B5_EPOCHS = B4_EPOCHS = 1
    B5_PATIENCE = B4_PATIENCE = 1
T0 = time.time()


def log(msg):
    line = f"[{time.time() - T0:7.1f}s][fold] {msg}"
    print(line, flush=True)
    with open(RUN / "s04_fold_run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def multiclass_auc(y, P):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro"))


def macro_f1(y, P):
    from sklearn.metrics import f1_score
    return float(f1_score(y, P.argmax(1), average="macro"))


class RGBDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.p, self.y, self.tf = list(paths), np.asarray(labels), tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), int(self.y[i])


class MaskDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.p, self.y, self.tf = list(paths), np.asarray(labels), tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        m = Image.open(self.p[i]).convert("L")
        m = self.tf(m)
        if m.shape[0] == 1:
            m = m.repeat(3, 1, 1)
        return m, int(self.y[i])


class Wrap(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        return self.head(self.backbone(x))


def train_encoder(kind, tr, va, dev, tag):
    """kind: 'rgb' | 'vessel'. Returns (backbone_state, head_state, history, selected)."""
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    if kind == "rgb":
        arch, tf_tr, tf_ev, max_ep, pat, lr, drop = (
            "efficientnet_b5",
            transforms.Compose([transforms.RandomResizedCrop(384, scale=(0.95, 1.0), ratio=(1.0, 1.0)),
                                transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10),
                                transforms.ColorJitter(0.10, 0.10, 0.10, 0.0),
                                transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
            transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(),
                                transforms.Normalize(MEAN, STD)]),
            B5_EPOCHS, B5_PATIENCE, B5_LR, B5_DROP)
        DS = RGBDS
        paths_tr, paths_va = tr.image_path, va.image_path
    else:
        arch, tf_tr, tf_ev, max_ep, pat, lr, drop = (
            "efficientnet_b4",
            transforms.Compose([transforms.RandomResizedCrop(384, scale=(0.95, 1.0), ratio=(1.0, 1.0),
                                                             interpolation=IM.NEAREST),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomRotation(10, interpolation=IM.NEAREST),
                                transforms.ToTensor()]),
            transforms.Compose([transforms.Resize((384, 384), interpolation=IM.NEAREST),
                                transforms.ToTensor()]),
            B4_EPOCHS, B4_PATIENCE, B4_LR, B4_DROP)
        DS = MaskDS
        paths_tr, paths_va = tr.mask_path, va.mask_path

    dl_tr = DataLoader(DS(paths_tr, tr.label, tf_tr), batch_size=16, shuffle=True, num_workers=0)
    dl_va = DataLoader(DS(paths_va, va.label, tf_ev), batch_size=32, shuffle=False, num_workers=0)
    backbone = timm.create_model(arch, pretrained=True, num_classes=0, global_pool="avg").to(dev)
    head = nn.Sequential(nn.Dropout(drop), nn.Linear(backbone.num_features, 3)).to(dev)
    net = Wrap(backbone, head).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    w = torch.tensor([WEIGHTS[0], WEIGHTS[1], WEIGHTS[2]], dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    hist, best, best_key, best_state = [], None, None, None
    for ep in range(1, max_ep + 1):
        net.train(); tl, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), torch.as_tensor(y, device=dev)
            opt.zero_grad()
            loss = lossf(net(x), y)
            loss.backward(); opt.step()
            tl += float(loss.item()) * len(y); n += len(y)
        tl /= max(n, 1)
        net.eval(); vl, P = 0.0, []
        with torch.no_grad():
            for x, y in dl_va:
                out = net(x.to(dev))
                vl += float(lossf(out, torch.as_tensor(y, device=dev)).item()) * len(y)
                P.append(torch.softmax(out, 1).cpu().numpy())
        vl /= len(va)
        P = np.concatenate(P)
        auc, f1 = multiclass_auc(va.label.values, P), macro_f1(va.label.values, P)
        rec = {"epoch": ep, "train_loss": tl, "val_loss": vl, "val_auc": auc, "val_macro_f1": f1}
        hist.append(rec)
        log(f"{tag} ep{ep:02d} train {tl:.4f} val {vl:.4f} auc {auc:.5f} f1 {f1:.4f}")
        key = (auc, f1, -vl, -ep)
        if best_key is None or key > best_key:
            best_key, best = key, rec
            best_state = ({k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        if ep - best["epoch"] >= pat:
            log(f"{tag} early stop at ep{ep} (best ep{best['epoch']})")
            break
    torch.save({"backbone": best_state[0], "head": best_state[1]}, CK / f"fold{fold}_{kind}_selected.pth")
    (RUN / f"fold{fold}_{kind}_history.json").write_text(json.dumps(hist, indent=1))
    log(f"{tag} SELECTED ep{best['epoch']} val_auc {best['val_auc']:.5f}")
    return best_state, hist, best


def extract(kind, state, df, dev):
    backbone, head = state
    arch = "efficientnet_b5" if kind == "rgb" else "efficientnet_b4"
    if kind == "rgb":
        tf = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(),
                                 transforms.Normalize(MEAN, STD)])
        DS = RGBDS
        paths = df.image_path
    else:
        tf = transforms.Compose([transforms.Resize((384, 384), interpolation=IM.NEAREST),
                                 transforms.ToTensor()])
        DS = MaskDS
        paths = df.mask_path
    m = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="avg")
    m.load_state_dict(backbone); m.to(dev).eval()
    dl = DataLoader(DS(paths, np.zeros(len(df)), tf), batch_size=32, shuffle=False, num_workers=0)
    E = np.zeros((len(df), m.num_features), np.float32)
    with torch.no_grad():
        for i, (x, _y) in enumerate(dl):
            E[i * 32:i * 32 + len(x)] = m(x.to(dev)).cpu().numpy()
    del m
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    log(f"{kind} embeddings {E.shape} nan {int(np.isnan(E).sum())}")
    return E


def main(fold):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = pd.read_csv(OUT / "02_common_farFUM_3fold_manifest.csv").reset_index(drop=True)
    te = m[m.fold == fold].reset_index(drop=True)
    dv = m[m.fold != fold].reset_index(drop=True)
    # patient-level development split (deterministic, class-stratified);
    # first seed whose internal validation fold holds all three classes wins.
    tr = va = None
    for seed in (SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4):
        sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        t_i, v_i = next(iter(sgkf.split(dv, y=dv.label, groups=dv.patient_id)))
        c_tr, c_va = dv.iloc[t_i], dv.iloc[v_i]
        if set(c_va.label.unique()) == {0, 1, 2} and set(c_tr.label.unique()) == {0, 1, 2}:
            tr, va = c_tr.reset_index(drop=True), c_va.reset_index(drop=True)
            if seed != SEED:
                log(f"dev split used seed {seed} (seed {SEED} gave an incomplete class set)")
            break
    assert tr is not None, "no development split with all three classes"
    assert set(tr.patient_id) & set(va.patient_id) == set()
    assert set(dv.patient_id) & set(te.patient_id) == set()
    if SMOKE:
        tr = pd.concat([g.head(8) for _, g in tr.groupby("label")]).reset_index(drop=True)
        va = pd.concat([g.head(8) for _, g in va.groupby("label")]).reset_index(drop=True)
        te = pd.concat([g.head(8) for _, g in te.groupby("label")]).reset_index(drop=True)
    log(f"fold {fold} dev {len(dv)}: train {len(tr)} ({tr.patient_id.nunique()} pat) "
        f"val {len(va)} ({va.patient_id.nunique()} pat) test {len(te)} ({te.patient_id.nunique()} pat)")
    log(f"class counts train {tr.label.value_counts().to_dict()} "
        f"val {va.label.value_counts().to_dict()} test {te.label.value_counts().to_dict()}")

    rgb_state, rgb_hist, rgb_sel = train_encoder("rgb", tr, va, device, f"fold{fold} RGB")
    ves_state, ves_hist, ves_sel = train_encoder("vessel", tr, va, device, f"fold{fold} VESSEL")
    Ergb = extract("rgb", rgb_state, m, device)
    Eves = extract("vessel", ves_state, m, device)
    np.savez_compressed(OUT / f"fold{fold}_embeddings.npz", rgb=Ergb, vessel=Eves,
                        image_id=m.image_path.values, fold=m.fold.values)

    import xgboost as xgb
    y = m.label.values
    w = np.array([WEIGHTS[int(v)] for v in y], dtype=np.float32)
    Bx = Ergb
    Cx = np.hstack([Ergb, m[FEATS].to_numpy(np.float32)])
    Ex = np.hstack([Ergb, Eves])
    Gx = np.hstack([Ergb, Eves, m[FEATS].to_numpy(np.float32)])
    mats = {"B": Bx, "C": Cx, "E": Ex, "G": Gx}
    tr_mask = m.fold != fold
    te_mask = m.fold == fold
    preds, fits = {}, {}
    for name, X in mats.items():
        clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                                random_state=SEED, n_jobs=8, tree_method="hist", **XGB_PARAMS)
        clf.fit(X[tr_mask.values], y[tr_mask.values], sample_weight=w[tr_mask.values])
        P = clf.predict_proba(X[te_mask.values])
        preds[name] = P
        fits[name] = {"n_features": int(X.shape[1]),
                      "train_auc": multiclass_auc(y[tr_mask.values],
                                                  clf.predict_proba(X[tr_mask.values]))}
        log(f"fold{fold} {name}: dim {X.shape[1]} train_auc {fits[name]['train_auc']:.5f} "
            f"test_auc {multiclass_auc(y[te_mask.values], P):.5f}")
    te_df = m[te_mask].reset_index(drop=True)
    out = pd.DataFrame({"image_id": te_df.image_path.map(lambda p: Path(p).stem),
                        "image_path": te_df.image_path, "patient_id": te_df.patient_id,
                        "fold": fold, "true_label": te_df.label})
    for name in ("B", "C", "E", "G"):
        for c in range(3):
            out[f"{name}_p{c}"] = preds[name][:, c]
    out.to_csv(OUT / f"fold{fold}_test_preds.csv", index=False)
    (OUT / f"fold{fold}_selection.json").write_text(json.dumps(
        {"fold": fold, "rgb_selected": rgb_sel, "vessel_selected": ves_sel,
         "rgb_history": rgb_hist, "vessel_history": ves_hist, "xgboost": fits,
         "n_train": len(tr), "n_val": len(va), "n_test": int(te_mask.sum()),
         "test_auc": {k: multiclass_auc(te_df.label.values, v) for k, v in preds.items()}},
        indent=1))
    np.save(OUT / f"fold{fold}_test_probs.npy",
            np.stack([preds[k] for k in ("B", "C", "E", "G")]))
    log(f"fold {fold} DONE")
    print(f"FOLD{fold}_DONE")


if __name__ == "__main__":
    fold = int(sys.argv[1])
    main(fold)
