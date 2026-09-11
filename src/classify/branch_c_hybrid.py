"""Branch C (main deliverable): HYBRID model = PVBM biomarkers + CNN embedding -> XGBoost.

Train on train; tune threshold on val; report test with bootstrap CI.
Includes ablation: biomarkers-only, embedding-only, full fusion.

Usage:
    python -m src.classify.branch_c_hybrid
    python -m src.classify.branch_c_hybrid --force
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.classify.branch_a_tabular import META_COLS
from src.classify.branch_b_cnn import BackboneClassifier
from src.classify.image_dataset import build_transforms
from src.utils.branch_eval import scores_from_classifier, train_select_eval_tabular
from src.utils.common import (
    build_provenance,
    ensure_dirs,
    get_device,
    load_config,
    num_classes,
    plus_class_index,
    provenance_matches_current,
    set_seed,
    sha256_file,
)


class _PathDataset(Dataset):
    def __init__(self, image_paths, img_size: int):
        self.paths = list(image_paths)
        self.tf = build_transforms(img_size, train=False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), idx


@torch.no_grad()
def extract_embeddings(model: BackboneClassifier, image_paths, img_size, device, batch_size=16):
    model.eval()
    ds = _PathDataset(image_paths, img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats = [None] * len(ds)
    for x, idxs in tqdm(dl, desc="cnn embeddings"):
        emb = model.backbone(x.to(device)).cpu().numpy()
        for j, i in zip(range(len(idxs)), idxs.numpy()):
            feats[int(i)] = emb[j]
    return np.vstack(feats)


def build_models(seed: int):
    models: dict = {}
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
        )
    except ImportError:
        print("[warn] xgboost not installed")
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=600, learning_rate=0.03, class_weight="balanced", random_state=seed
        )
    except ImportError:
        pass
    if not models:
        from sklearn.ensemble import RandomForestClassifier

        models["random_forest"] = RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=seed
        )
    return models


def _run_ablation(
    name: str,
    X,
    y,
    split,
    test_meta,
    plus_idx,
    class_names,
    ev,
    seed,
    models,
):
    from sklearn.preprocessing import StandardScaler

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    scaler = StandardScaler().fit(X[train_mask])
    summary, best_name, best_scores, best_model = train_select_eval_tabular(
        models,
        scaler.transform(X[train_mask]),
        y[train_mask],
        scaler.transform(X[val_mask]),
        y[val_mask],
        scaler.transform(X[test_mask]),
        y[test_mask],
        test_meta,
        plus_idx,
        class_names,
        threshold_method=ev.get("threshold_method", "youden"),
        model_select=ev.get("model_select", "val_f1_plus"),
        n_boot=ev.get("bootstrap_n", 1000),
        seed=seed,
    )
    print(f"[ablation:{name}] best={best_name}")
    return summary, best_name, best_scores, best_model


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg["seed"])
    device = get_device()
    ev = cfg.get("eval", {})

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="Branch B checkpoint (auto-detect if omitted)")
    ap.add_argument("--emb_prefix", default="cnn_emb_", help="column prefix for embedding dims")
    ap.add_argument("--force", action="store_true", help="recompute even if results exist")
    args = ap.parse_args()

    res_dir = cfg["paths"]["results_dir"]
    out_json = res_dir / "branch_c_results.json"
    if out_json.exists() and not args.force:
        existing = json.loads(out_json.read_text())
        if provenance_matches_current(existing, cfg):
            print(f"[skip] current Branch C results exist -> {out_json}")
            return
        print(f"[stale] ignoring Branch C results from different split/config: {out_json}")

    feats_path = cfg["paths"]["features_dir"] / "biomarker_features.csv"
    if not feats_path.exists():
        raise SystemExit(f"{feats_path} not found. Run src.biomarker.extract_pvbm first.")
    df = pd.read_csv(feats_path).reset_index(drop=True)
    y = df["label"].astype(int).values
    split = df["split"].values
    train_mask = split == "train"
    candidate_cols = [column for column in df.columns if column not in META_COLS]
    numeric = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    train_numeric = numeric.loc[train_mask]
    bio_cols = [
        column
        for column in numeric.columns
        if not train_numeric[column].isna().all()
        and float(train_numeric[column].std(skipna=True) or 0.0) > 0.0
    ]
    bio_medians = train_numeric[bio_cols].median().fillna(0.0)
    X_bio = numeric[bio_cols].fillna(bio_medians)

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path is None:
        branch_b_results = res_dir / "branch_b_results.json"
        if branch_b_results.exists():
            checkpoint = json.loads(branch_b_results.read_text()).get("checkpoint")
            ckpt_path = Path(checkpoint) if checkpoint else None
        if ckpt_path is None:
            backbone_name = cfg["classify_b"]["backbone"].replace(".", "_")
            ckpt_path = cfg["paths"]["weights_dir"] / f"branch_b_{backbone_name}.pth"
        if not ckpt_path.exists():
            raise SystemExit(
                f"Recorded/default Branch B checkpoint missing: {ckpt_path}. "
                "Train src.classify.branch_b_cnn first or pass --ckpt."
            )
    state = torch.load(str(ckpt_path), map_location=device)
    if (
        state.get("split_sha256")
        != sha256_file(cfg["paths"]["splits_dir"] / "all.csv")
        or state.get("config_sha256")
        != sha256_file(cfg["_root"] / "configs" / "config.yaml")
    ):
        raise SystemExit(
            "Branch B checkpoint is stale or lacks current split/config provenance."
        )
    backbone = state.get("backbone", cfg["classify_b"]["backbone"])
    sd = state["state_dict"]
    n_cls = int(sd["head.weight"].shape[0]) if "head.weight" in sd else num_classes(cfg)
    plus_idx = plus_class_index(cfg)
    class_names = cfg["data"]["class_names"]
    model = BackboneClassifier(backbone, num_classes=n_cls).to(device)
    model.load_state_dict(sd)
    print(f"[hybrid] backbone={backbone}  ckpt={ckpt_path.name}")

    emb = extract_embeddings(model, df["image_path"].tolist(), cfg["classify_b"]["img_size"], device)
    test_mask = split == "test"
    meta_cols = [
        column
        for column in [
            "image_path",
            "label",
            "source",
            "group_id",
            "patient_id",
            "exam_id",
            "identity_level",
        ]
        if column in df.columns
    ]
    test_meta = df.loc[test_mask, meta_cols]

    X_full = np.hstack([X_bio.values, emb])
    X_bio_np = X_bio.values
    X_emb_np = emb
    ablation_specs = [
        ("fusion", X_full, bio_cols + [f"{args.emb_prefix}{i}" for i in range(emb.shape[1])]),
        ("biomarkers_only", X_bio_np, bio_cols),
        ("embedding_only", X_emb_np, [f"{args.emb_prefix}{i}" for i in range(emb.shape[1])]),
    ]

    ablation_out: dict = {}
    fusion_summary, best_name, best_scores, best_model = None, None, None, None
    feat_names: list[str] = []

    for ab_name, X_ab, fnames in ablation_specs:
        summary, bname, bscores, bmodel = _run_ablation(
            ab_name,
            X_ab,
            y,
            split,
            test_meta,
            plus_idx,
            class_names,
            ev,
            cfg["seed"],
            build_models(cfg["seed"]),
        )
        ablation_out[ab_name] = summary
        if ab_name == "fusion":
            fusion_summary, best_name, best_scores, best_model = summary, bname, bscores, bmodel
            feat_names = fnames

    payload = {
        "best_model": best_name,
        "backbone": backbone,
        "n_biomarker_features": len(bio_cols),
        "n_cnn_features": emb.shape[1],
        "model_select": ev.get("model_select", "val_f1_plus"),
        "threshold_method": ev.get("threshold_method", "youden"),
        "checkpoint": str(ckpt_path.resolve()),
        "preprocessing": {
            "feature_selection": "training_split_only",
            "imputation": "training_median",
            "scaling": "StandardScaler fit on training only",
        },
        "provenance": build_provenance(cfg),
        "metrics": fusion_summary["metrics"] if fusion_summary else {},
        "ablation": ablation_out,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if best_scores is not None:
        pred_df = test_meta.copy()
        pred_df["p_plus_branch_c"] = best_scores
        pred_df.to_csv(res_dir / "branch_c_test_preds.csv", index=False)

    if best_model is not None:
        from sklearn.preprocessing import StandardScaler

        train_mask = split == "train"
        val_mask_c = split == "val"
        scaler = StandardScaler().fit(X_full[train_mask])
        val_scores_best = scores_from_classifier(best_model, scaler.transform(X_full[val_mask_c]), plus_idx)[0]
        val_df_c = df.loc[val_mask_c, meta_cols].copy()
        val_df_c["p_plus_branch_c"] = val_scores_best
        val_df_c.to_csv(res_dir / "branch_c_val_preds.csv", index=False)
        if feat_names:
            import joblib

            joblib.dump(
                {
                    "model": best_model,
                    "feature_columns": feat_names,
                    "biomarker_medians": bio_medians.to_dict(),
                    "scaler": scaler,
                    "plus_idx": plus_idx,
                    "branch_b_checkpoint": str(ckpt_path.resolve()),
                },
                res_dir / "branch_c_model.joblib",
            )
            _shap_plot(
                best_model,
                best_name,
                scaler.transform(X_full[train_mask]),
                feat_names,
                plus_idx,
                res_dir,
            )

    if fusion_summary and best_name:
        bt = fusion_summary["metrics"][best_name]["test"]["plus_ovr"]
        print(f"[done] fusion best={best_name} test_AUC={bt['auc']:.3f} sens={bt['sensitivity']:.3f} -> {res_dir}")


def _shap_plot(model, best_name, X_tr, feat_names, plus_idx: int, res_dir: Path) -> None:
    if model is None:
        return
    output_path = res_dir / "branch_c_shap.png"
    output_path.unlink(missing_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_tr)
        classes = np.asarray(model.classes_)
        positions = np.flatnonzero(classes == plus_idx)
        if len(positions) != 1:
            raise ValueError(f"Plus class {plus_idx} missing from {classes.tolist()}")
        class_position = int(positions[0])
        if isinstance(sv, list):
            sv = sv[class_position]
        elif getattr(sv, "ndim", 0) == 3:
            sv = sv[:, :, class_position]
        if np.shape(sv) != (len(X_tr), len(feat_names)):
            raise ValueError(
                f"Unexpected Plus SHAP shape {np.shape(sv)}; "
                f"expected {(len(X_tr), len(feat_names))}"
            )
        biomarker_indices = [
            index
            for index, name in enumerate(feat_names)
            if not name.startswith("cnn_emb_")
        ]
        shap.summary_plot(
            sv[:, biomarker_indices],
            features=X_tr[:, biomarker_indices],
            feature_names=[feat_names[index] for index in biomarker_indices],
            show=False,
            max_display=20,
        )
        plt.title(f"Hybrid ({best_name}) feature importance")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[shap] saved Plus-class explanation -> {output_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] SHAP plot skipped: {e}")


if __name__ == "__main__":
    main()
