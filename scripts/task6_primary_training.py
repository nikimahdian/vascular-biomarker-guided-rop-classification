#!/usr/bin/env python
"""Task 6 — final primary training pipeline, exactly per configs/final_primary_model_contract_v1.yaml.

Stages, all resumable, each writing artifacts + a runtime log line:
  A pre-flight integrity gate (STOP on any mismatch)
  C A_PRIMARY            XGBoost, 5 biomarkers, train+val selection
  D/E B_RGB              EfficientNet-B5, per-epoch checkpoint, frozen selection rule
  F embeddings           8862 x 2048 from the selected checkpoint
  G/H B_EMBEDDING_ONLY   XGBoost, 2048-d, shared hyperparameters
  I C_PRIMARY            XGBoost, 2053-d, identical config
  J/K/L/M test evaluation, per-image predictions, source breakdown, calibration, figures

No contract change, no test use in selection, no post-hoc tuning.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/moniaz/niki")
ART = ROOT / "artifacts/task6"
OUT = ROOT / "_private_audit"
LOG = ROOT / "docs/TASK6_RUNTIME_LOG.md"
MANIFEST = ROOT / "data/splits/primary_complete_case_v2.csv"
FEATS5 = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
V2_SHA = "b4661dffd2e082f93d183cdb26ae0462dce8af32b135c1968db2deacc493365e"
SEED = 42
WEIGHTS = {0: 0.46051, 1: 3.18103, 2: 1.94512}
NAMES = ["Normal", "Pre_Plus", "Plus"]
A_GRID = [{"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": s,
           "colsample_bytree": c, "min_child_weight": 1, "reg_lambda": 1.0}
          for n in (200, 500, 800) for d in (2, 3, 4) for lr in (0.03, 0.05, 0.10)
          for s in (0.8, 1.0) for c in (0.8, 1.0)]
X_GRID = [{"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": 0.8,
           "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0}
          for n in (200, 500) for d in (3, 5) for lr in (0.05, 0.10)]
T0 = time.time()


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "task6_progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def runtime(stage, secs, note=""):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"| {stage} | {secs / 60:.1f} min | {note} |\n")


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def multiclass_auc(y, P):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro"))


def macro_f1(y, P):
    from sklearn.metrics import f1_score
    return float(f1_score(y, P.argmax(1), average="macro"))


# ---------------------------------------------------------------- A
def preflight():
    log("A. PRE-FLIGHT INTEGRITY GATE")
    m = pd.read_csv(MANIFEST, low_memory=False)
    checks = {}
    checks["population_n_8862"] = len(m) == 8862
    c = m.split.value_counts().to_dict()
    checks["split_counts"] = (c.get("train") == 6203 and c.get("val") == 1328
                              and c.get("test") == 1331)
    checks["no_image_overlap"] = (m.groupby("image_path").split.nunique().max() == 1)
    g = m.groupby("group_id").split.nunique()
    checks["no_group_overlap"] = bool((g == 1).all())
    checks["labels_0_1_2"] = set(m.label.unique()) == {0, 1, 2}
    checks["features_finite"] = bool(m[FEATS5].notna().all().all())
    checks["rgb_paths_exist"] = bool(m.image_path.map(lambda p: Path(p).exists()).all())
    checks["v2_table_present"] = (ROOT / "data/features/final_biomarkers_v2.csv").exists()
    for k, v in checks.items():
        log(f"   {k:26s} : {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit("PREFLIGHT_FAILED")
    log(f"   groups {m.group_id.nunique()}  sources {m.source.value_counts().to_dict()}")
    return m


# ---------------------------------------------------------------- C
def run_xgb(Xtr, ytr, wtr, Xva, yva, grid, tag):
    import xgboost as xgb
    best, best_auc, rows = None, -1.0, []
    for i, p in enumerate(grid):
        clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                                random_state=SEED, n_jobs=8, tree_method="hist", **p)
        clf.fit(Xtr, ytr, sample_weight=wtr)
        auc = multiclass_auc(yva, clf.predict_proba(Xva))
        rows.append({**p, "val_auc": auc})
        if auc > best_auc:
            best, best_auc = p, auc
        if (i + 1) % 12 == 0:
            log(f"   {tag} grid {i+1}/{len(grid)} best_val_auc {best_auc:.5f}")
    pd.DataFrame(rows).sort_values("val_auc", ascending=False).to_csv(
        ART / f"{tag}_grid.csv", index=False)
    log(f"   {tag} SELECTED {best}  val_auc {best_auc:.5f}")
    return best, best_auc


def xgb_fit_predict(params, Xtr, ytr, wtr, Xva, Xte):
    import xgboost as xgb
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                            random_state=SEED, n_jobs=8, tree_method="hist", **params)
    clf.fit(Xtr, ytr, sample_weight=wtr)
    return clf, clf.predict_proba(Xva), clf.predict_proba(Xte)


# ---------------------------------------------------------------- D
def train_b(m):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import timm
    from PIL import Image
    from torchvision import transforms

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"D. B_RGB training on {dev}")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tr_tf = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.95, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.10, 0.10, 0.10, 0.0),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    ev_tf = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

    class DS(Dataset):
        def __init__(self, paths, labels, tf):
            self.p, self.y, self.tf = list(paths), list(labels), tf

        def __len__(self):
            return len(self.p)

        def __getitem__(self, i):
            im = Image.open(self.p[i]).convert("RGB")
            return self.tf(im), self.y[i]

    tr = m[m.split == "train"]; va = m[m.split == "val"]
    dl_tr = DataLoader(DS(tr.image_path, tr.label, tr_tf), batch_size=16, shuffle=True,
                       num_workers=8, drop_last=False)
    dl_va = DataLoader(DS(va.image_path, va.label, ev_tf), batch_size=32, shuffle=False,
                       num_workers=8)
    model = timm.create_model("efficientnet_b5", pretrained=True, num_classes=0,
                              global_pool="avg").to(dev)
    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.num_features, 3)).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-5,
                            weight_decay=1e-4)
    w = torch.tensor([WEIGHTS[0], WEIGHTS[1], WEIGHTS[2]], dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    hist_p = ART / "b_rgb_history.json"
    hist = json.loads(hist_p.read_text()) if hist_p.exists() else []
    start = len(hist)
    best_sel, best_key = None, None
    for ep in range(start, 40):
        model.train(); tl = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev), torch.as_tensor(y).to(dev)
            opt.zero_grad()
            out = head(model(x))
            loss = lossf(out, y)
            loss.backward(); opt.step()
            tl += float(loss.item()) * len(y)
        tl /= len(tr)
        model.eval(); vl, P = 0.0, []
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(dev)
                out = head(model(x))
                vl += float(lossf(out, torch.as_tensor(y).to(dev)).item()) * len(y)
                P.append(torch.softmax(out, 1).cpu().numpy())
        vl /= len(va)
        P = np.concatenate(P)
        auc, f1 = multiclass_auc(va.label.values, P), macro_f1(va.label.values, P)
        ck = ART / f"b_rgb_epoch{ep+1}.pth"
        torch.save({"backbone": model.state_dict(), "head": head.state_dict()}, ck)
        hist.append({"epoch": ep + 1, "train_loss": tl, "val_loss": vl, "val_auc": auc,
                     "val_macro_f1": f1, "ckpt": str(ck)})
        hist_p.write_text(json.dumps(hist))
        log(f"   epoch {ep+1:2d} train {tl:.4f} val {vl:.4f} auc {auc:.5f} f1 {f1:.4f}")
        key = (auc, f1, -vl, -(ep + 1))
        if best_key is None or key > best_key:
            best_key, best_sel = key, hist[-1]
        if ep + 1 - best_sel["epoch"] >= 8:
            log(f"   early stop at epoch {ep+1} (best {best_sel['epoch']})")
            break
    import shutil
    shutil.copy(best_sel["ckpt"], ART / "b_rgb_selected.pth")
    (ART / "b_rgb_selection.json").write_text(json.dumps(
        {"selected_epoch": best_sel["epoch"], "val_auc": best_sel["val_auc"],
         "val_macro_f1": best_sel["val_macro_f1"], "val_loss": best_sel["val_loss"],
         "sha256": sha(ART / "b_rgb_selected.pth"), "epochs_run": len(hist),
         "history": hist}, indent=2))
    log(f"   SELECTED epoch {best_sel['epoch']} val_auc {best_sel['val_auc']:.5f} "
        f"sha {sha(ART / 'b_rgb_selected.pth')[:16]}")
    bck = torch.load(best_sel["ckpt"], map_location="cpu")
    model.load_state_dict(bck["backbone"]); head.load_state_dict(bck["head"]); model.eval()

    def infer(df):
        dl = DataLoader(DS(df.image_path, df.label, ev_tf), batch_size=32, shuffle=False,
                        num_workers=8)
        P = []
        with torch.no_grad():
            for x, _y in dl:
                P.append(torch.softmax(head(model(x.to(dev))), 1).cpu().numpy())
        return np.concatenate(P)
    te_df = m[m.split == "test"]
    Pte = infer(te_df); Pva = infer(va)
    np.save(ART / "b_rgb_test_probs.npy", Pte)
    np.save(ART / "b_rgb_val_probs.npy", Pva)
    log(f"   B_RGB test probabilities saved {Pte.shape}")
    return dev, Pva, Pte


# ---------------------------------------------------------------- F
def extract_embeddings(m, dev):
    import torch
    import timm
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from torchvision import transforms
    out_p = ART / "embeddings_b5_8862.parquet"
    if out_p.exists():
        log("F. embeddings already present, reusing")
        return pd.read_parquet(out_p)
    ck = torch.load(ART / "b_rgb_selected.pth", map_location="cpu")
    model = timm.create_model("efficientnet_b5", pretrained=False, num_classes=0,
                              global_pool="avg")
    model.load_state_dict(ck["backbone"]); model.to(dev).eval()
    tf = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])])

    class DS(Dataset):
        def __init__(self, paths):
            self.p = list(paths)

        def __len__(self):
            return len(self.p)

        def __getitem__(self, i):
            return tf(Image.open(self.p[i]).convert("RGB")), i
    dl = DataLoader(DS(m.image_path), batch_size=32, shuffle=False, num_workers=8)
    E = np.zeros((len(m), 2048), np.float32)
    with torch.no_grad():
        for x, idx in dl:
            E[idx.numpy()] = model(x.to(dev)).cpu().numpy()
    cols = [f"embedding_{i:04d}" for i in range(2048)]
    df = pd.DataFrame(E, columns=cols)
    df.insert(0, "image_id", m.image_path.values)
    df.insert(1, "group_id", m.group_id.values)
    df.insert(2, "source", m.source.values)
    df.insert(3, "split", m.split.values)
    df.insert(4, "label", m.label.values)
    df.to_parquet(out_p, index=False)
    log(f"F. embeddings {df.shape} nan {int(df[cols].isna().sum().sum())} "
        f"sha {sha(out_p)[:16]}")
    return df


# ---------------------------------------------------------------- J-M
def evaluate(name, y, P, meta, res):
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                                 roc_auc_score)
    pred = P.argmax(1)
    row = {"model": name, "n": len(y),
           "multiclass_auc": multiclass_auc(y, P),
           "macro_ovr_auc": multiclass_auc(y, P),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro")),
           "brier": float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1)))}
    conf, corr = P.max(1), (pred == y)
    ece = 0.0
    for lo in np.linspace(0, 1, 16)[:-1]:
        m_ = (conf >= lo) & (conf < lo + 1 / 15)
        if m_.sum():
            ece += m_.mean() * abs(corr[m_].mean() - conf[m_].mean())
    row["ece"] = float(ece)
    for k in range(3):
        yk = (y == k).astype(int)
        row[f"auc_{NAMES[k]}"] = float(roc_auc_score(yk, P[:, k])) if 0 < yk.sum() < len(yk) \
            else float("nan")
    res.append(row)
    pd.DataFrame(confusion_matrix(y, pred)).to_csv(ART / f"confusion_{name}.csv", index=False)
    pr = meta[["image_id", "group_id", "source"]].copy()
    pr["true_label"] = y
    pr["prob_Normal"], pr["prob_Pre_Plus"], pr["prob_Plus"] = P[:, 0], P[:, 1], P[:, 2]
    pr["predicted_label"] = pred
    pr.to_csv(ART / f"test_predictions_{name}.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y, pred)
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks(range(3)); ax.set_xticklabels(NAMES, fontsize=7)
        ax.set_yticks(range(3)); ax.set_yticklabels(NAMES, fontsize=7)
        ax.set_title(f"{name}  macroAUC {row['multiclass_auc']:.3f}", fontsize=8)
        fig.tight_layout(); fig.savefig(ART / f"confusion_{name}.png", dpi=160)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"   figure skipped {name}: {e}")
    return row


def main():
    ART.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("# Task 6 runtime log\n\n| stage | wall clock | note |\n|---|---|---|\n",
                       encoding="utf-8")
    t = time.time(); m = preflight(); runtime("A pre-flight", time.time() - t, "8/8 PASS")

    # ---- A_PRIMARY
    t = time.time(); log("C. A_PRIMARY")
    tr, va, te = m[m.split == "train"], m[m.split == "val"], m[m.split == "test"]
    wtr = tr.label.map(WEIGHTS).values
    pa, vauc = run_xgb(tr[FEATS5].values, tr.label.values, wtr, va[FEATS5].values,
                       va.label.values, A_GRID, "a_primary")
    clf_a, Pva_a, Pte_a = xgb_fit_predict(pa, tr[FEATS5].values, tr.label.values, wtr,
                                          va[FEATS5].values, te[FEATS5].values)
    clf_a.save_model(str(ART / "a_primary.json"))
    (ART / "a_primary_selected.json").write_text(json.dumps(
        {"params": pa, "seed": SEED, "val_auc": vauc, "features": FEATS5,
         "class_weights": WEIGHTS, "model_sha256": sha(ART / "a_primary.json")}, indent=2))
    runtime("C A_PRIMARY", time.time() - t, f"selected {pa}")

    # ---- B_RGB
    t = time.time(); dev, Pva_rgb, Pte_rgb = train_b(m)
    sel = json.loads((ART / "b_rgb_selection.json").read_text())
    runtime("D/E B_RGB", time.time() - t, f"{sel['epochs_run']} epochs, selected {sel['selected_epoch']}")

    # ---- embeddings
    t = time.time(); E = extract_embeddings(m, dev)
    ecols = [c for c in E.columns if c.startswith("embedding_")]
    runtime("F embeddings", time.time() - t, f"{E.shape[0]}x{len(ecols)} sha {sha(ART/'embeddings_b5_8862.parquet')[:16]}")

    # ---- B_EMBEDDING_ONLY
    t = time.time(); log("G/H. B_EMBEDDING_ONLY")
    Etr = E[E.split == "train"][ecols].values
    Eva = E[E.split == "val"][ecols].values
    Ete = E[E.split == "test"][ecols].values
    pb, vb = run_xgb(Etr, tr.label.values, wtr, Eva, va.label.values, X_GRID, "b_embedding_only")
    (ART / "downstream_xgb_selected.json").write_text(json.dumps(
        {"params": pb, "selected_on": "B_EMBEDDING_ONLY train+val", "seed": SEED, "val_auc": vb},
        indent=2))
    clf_b, Pva_b, Pte_b = xgb_fit_predict(pb, Etr, tr.label.values, wtr, Eva, Ete)
    clf_b.save_model(str(ART / "b_embedding_only.json"))
    runtime("G/H B_EMBEDDING_ONLY", time.time() - t, f"selected {pb}")

    # ---- C_PRIMARY
    t = time.time(); log("I. C_PRIMARY")
    Xtr = np.hstack([Etr, tr[FEATS5].values]); Xva = np.hstack([Eva, va[FEATS5].values])
    Xte = np.hstack([Ete, te[FEATS5].values])
    assert Xtr.shape[1] == 2053, Xtr.shape
    diff = np.abs(Xtr[:, :2048] - Etr).max() + np.abs(Xtr[:, 2048:] - tr[FEATS5].values).max()
    log(f"   input dim {Xtr.shape[1]}  C-vs-B input difference confined to the 5 columns: {diff == 0}")
    clf_c, Pva_c, Pte_c = xgb_fit_predict(pb, Xtr, tr.label.values, wtr, Xva, Xte)
    clf_c.save_model(str(ART / "c_primary.json"))
    runtime("I C_PRIMARY", time.time() - t, "shared hyperparameters, no C-specific tuning")

    # ---- J/K/L/M
    t = time.time(); log("J/K/L/M. TEST EVALUATION (frozen, single pass)")
    meta_te = pd.DataFrame({"image_id": te.image_path.values, "group_id": te.group_id.values,
                            "source": te.source.values})
    y = te.label.values
    res = []
    evaluate("A_PRIMARY", y, Pte_a, meta_te, res)
    evaluate("B_RGB", y, Pte_rgb, meta_te, res)
    for nm, P in (("A_PRIMARY", Pte_a), ("B_RGB", Pte_rgb),
                  ("B_EMBEDDING_ONLY", Pte_b), ("C_PRIMARY", Pte_c)):
        log(f"   {nm:18s} macro_auc {multiclass_auc(y, P):.5f}")
    res = [r for r in res if r]
    R = pd.DataFrame(res)
    R.to_csv(ART / "metrics_summary.csv", index=False)
    rows = []
    for src in ("plus", "farfum_rop", "farabi"):
        s = meta_te.source == src
        if s.sum() < 10:
            continue
        for nm, P in (("A_PRIMARY", Pte_a), ("B_RGB", Pte_rgb),
                      ("B_EMBEDDING_ONLY", Pte_b), ("C_PRIMARY", Pte_c)):
            from sklearn.metrics import balanced_accuracy_score, f1_score
            rows.append({"model": nm, "source": src, "n": int(s.sum()),
                         "macro_auc": multiclass_auc(y[s], P[s]),
                         "balanced_accuracy": float(balanced_accuracy_score(y[s], P[s].argmax(1))),
                         "macro_f1": float(f1_score(y[s], P[s].argmax(1), average="macro"))})
    pd.DataFrame(rows).to_csv(ART / "source_breakdown.csv", index=False)
    runtime("J-M test evaluation", time.time() - t, "3 models with frozen probabilities")
    shas = {p.name: sha(p) for p in sorted(ART.glob("*")) if p.is_file()}
    (ART / "artifact_sha256.json").write_text(json.dumps(shas, indent=2))
    (OUT / "task6_summary.json").write_text(json.dumps(
        {"metrics": R.to_dict("records"), "source_breakdown": rows,
         "b_selection": sel, "a_params": pa, "downstream_params": pb,
         "embedding_rows": int(E.shape[0]), "artifact_sha256": shas,
         "elapsed_s": round(time.time() - T0, 1)}, indent=2, default=str))
    log("TASK6_STATUS = COMPLETE")


if __name__ == "__main__":
    main()
