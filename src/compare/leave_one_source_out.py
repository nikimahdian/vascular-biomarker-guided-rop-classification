"""Leave-one-source-out (LOSO) external validation.

For each held-out source:
  - train/val = other sources only (patient-aware within FARFUM when present)
  - test = ALL images from held-out source
  - Branch A: retrain tabular (fast)
  - Branch B: retrain CNN (slow) unless --skip-b or --reuse-embeddings
  - Branch C: hybrid ablation-style fusion on A features + B embeddings

Usage:
    python -m src.compare.leave_one_source_out
    python -m src.compare.leave_one_source_out --holdouts farfum_rop --epochs 30
    python -m src.compare.leave_one_source_out --skip-b   # A only (quick probe)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.classify.branch_a_tabular import META_COLS, build_models
from src.utils.branch_eval import scores_from_classifier, train_select_eval_tabular
from src.utils.common import (
    ensure_dirs,
    evaluate_plus_ovr_bundle,
    get_device,
    load_config,
    num_classes,
    plus_class_index,
    plus_ovr_metrics,
    set_seed,
    tune_plus_threshold,
)


def _atomic_write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_predictions(
    path: Path,
    metadata: pd.DataFrame,
    labels,
    scores,
    score_column: str,
) -> None:
    frame = metadata.reset_index(drop=True).copy()
    frame["label"] = np.asarray(labels).astype(int)
    frame[score_column] = np.asarray(scores, dtype=float)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _prepare_train_only(feats: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    split = feats["split"].values
    train_mask = split == "train"
    columns = [column for column in feats.columns if column not in META_COLS]
    numeric = feats[columns].apply(pd.to_numeric, errors="coerce")
    train = numeric.loc[train_mask]
    keep = [
        column
        for column in columns
        if not train[column].isna().all()
        and float(train[column].std(skipna=True) or 0.0) > 0.0
    ]
    medians = train[keep].median().fillna(0.0)
    return numeric[keep].fillna(medians).values, keep


def _split_in_domain(df: pd.DataFrame, val_frac: float, seed: int) -> pd.DataFrame:
    """Assign grouped train/val across every in-domain source."""
    out = df.copy()
    out["split"] = "train"
    if "group_id" not in out.columns or out["group_id"].isna().any():
        raise ValueError("LOSO requires non-null group_id for every image.")
    n_splits = max(2, round(1.0 / val_frac))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    _, val_idx = next(
        splitter.split(out, out["label"].astype(int), groups=out["group_id"])
    )
    out.iloc[val_idx, out.columns.get_loc("split")] = "val"
    train_groups = set(out.loc[out["split"] == "train", "group_id"])
    val_groups = set(out.loc[out["split"] == "val", "group_id"])
    assert not train_groups & val_groups
    return out


def run_branch_a(feats: pd.DataFrame, cfg: dict, holdout: str) -> dict:
    ev = cfg.get("eval", {})
    X, _ = _prepare_train_only(feats)
    y = feats["label"].astype(int).values
    split = feats["split"].values
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    scaler = StandardScaler().fit(X[train_mask])
    models = build_models(cfg["seed"], cfg["classify_a"]["models"])
    test_meta = feats.loc[
        test_mask,
        [
            c
            for c in [
                "image_path",
                "label",
                "source",
                "group_id",
                "patient_id",
                "exam_id",
            ]
            if c in feats.columns
        ],
    ]
    summary, best, scores, best_model = train_select_eval_tabular(
        models,
        scaler.transform(X[train_mask]),
        y[train_mask],
        scaler.transform(X[val_mask]),
        y[val_mask],
        scaler.transform(X[test_mask]),
        y[test_mask],
        test_meta,
        plus_class_index(cfg),
        cfg["data"]["class_names"],
        threshold_method=ev.get("threshold_method", "youden"),
        model_select=ev.get("model_select", "val_f1_plus"),
        n_boot=ev.get("bootstrap_n", 500),
        seed=cfg["seed"],
    )
    po = summary["metrics"][best]["test"]["plus_ovr"] if best else {}
    val_scores = (
        scores_from_classifier(
            best_model, scaler.transform(X[val_mask]), plus_class_index(cfg)
        )[0]
        if best_model is not None
        else np.array([])
    )
    result = {
        "holdout": holdout,
        "branch": "A",
        "best_model": best,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_groups_test": int(feats.loc[test_mask, "group_id"].nunique()),
        "plus_ovr": po,
        "bootstrap_ci": summary["metrics"][best]["test"].get("bootstrap_ci") if best else None,
        "test_scores": scores.tolist() if scores is not None else [],
        "test_labels": y[test_mask].tolist(),
        "test_meta": test_meta.to_dict("records"),
        "val_scores": val_scores.tolist(),
        "val_labels": y[val_mask].tolist(),
        "val_meta": feats.loc[val_mask, test_meta.columns].to_dict("records"),
    }
    return result


def run_branch_b(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    cfg: dict,
    holdout: str,
    epochs: int,
    out_dir: Path,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from src.classify.branch_b_cnn import (
        BackboneClassifier,
        FocalLoss,
        class_weights,
        predict_all,
        predict_scores,
    )
    from src.classify.image_dataset import FundusCsvDataset
    from src.utils.branch_eval import per_source_plus_metrics

    device = get_device()
    cb = cfg["classify_b"]
    n_cls = num_classes(cfg)
    plus_idx = plus_class_index(cfg)
    ev = cfg.get("eval", {})
    backbone = cb["backbone"]
    img_size, bs, nw = cb["img_size"], cb["batch_size"], min(2, cb.get("num_workers", 0))

    train_ds = FundusCsvDataset(str(train_csv), img_size, train=True)
    val_ds = FundusCsvDataset(str(val_csv), img_size, train=False)
    test_ds = FundusCsvDataset(str(test_csv), img_size, train=False)
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=nw, pin_memory=False)
    val_dl = DataLoader(val_ds, bs, shuffle=False, num_workers=nw, pin_memory=False)
    test_dl = DataLoader(test_ds, bs, shuffle=False, num_workers=nw, pin_memory=False)

    model = BackboneClassifier(backbone, num_classes=n_cls).to(device)
    if cb["use_focal_loss"]:
        criterion = FocalLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_ds.labels, n_cls, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cb["lr"], weight_decay=cb["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=3)

    ckpt = out_dir / f"loo_b_{holdout}_{backbone.replace('.', '_')}.pth"
    best_auc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        val_scores, val_labels = predict_scores(model, val_dl, device, plus_idx)
        vm = plus_ovr_metrics(val_labels, val_scores, plus_idx)
        print(f"[LOO-B {holdout}] epoch {epoch}/{epochs} val_auc={vm['auc']:.3f}")
        scheduler.step(vm["auc"] if not np.isnan(vm["auc"]) else 0.0)
        if not np.isnan(vm["auc"]) and vm["auc"] > best_auc:
            best_auc = vm["auc"]
            torch.save({"state_dict": model.state_dict(), "backbone": backbone}, ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    val_scores, val_labels, _ = predict_all(model, val_dl, device, plus_idx)
    thr = tune_plus_threshold(val_labels, val_scores, plus_idx, method=ev.get("threshold_method", "youden"))
    test_scores, test_labels, test_pred = predict_all(model, test_dl, device, plus_idx)
    test_meta = test_ds.df
    bundle = evaluate_plus_ovr_bundle(
        test_labels,
        test_scores,
        test_pred,
        plus_idx,
        thr,
        cfg["data"]["class_names"],
        n_boot=ev.get("bootstrap_n", 500),
        seed=cfg["seed"],
        groups=(
            test_meta["group_id"].values
            if "group_id" in test_meta.columns
            else None
        ),
    )
    return {
        "holdout": holdout,
        "branch": "B",
        "backbone": backbone,
        "epochs": epochs,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "n_groups_test": int(test_meta["group_id"].nunique()),
        "best_val_auc": best_auc,
        "val_threshold": float(thr),
        "plus_ovr": bundle["plus_ovr"],
        "bootstrap_ci": bundle["bootstrap_ci"],
        "ckpt": str(ckpt),
        "test_paths": test_ds.df["image_path"].tolist(),
        "test_scores": test_scores.tolist(),
        "test_labels": test_labels.tolist(),
        "test_meta": test_meta.to_dict("records"),
        "val_paths": val_ds.df["image_path"].tolist(),
        "val_scores": val_scores.tolist(),
        "val_labels": val_labels.tolist(),
        "val_meta": val_ds.df.to_dict("records"),
    }


def run_branch_c(
    feats: pd.DataFrame,
    emb: np.ndarray,
    cfg: dict,
    holdout: str,
) -> dict:
    from src.classify.branch_c_hybrid import build_models as build_c_models

    ev = cfg.get("eval", {})
    X_bio, _ = _prepare_train_only(feats)
    X = np.hstack([X_bio, emb])
    y = feats["label"].astype(int).values
    split = feats["split"].values
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    scaler = StandardScaler().fit(X[train_mask])
    test_meta = feats.loc[
        test_mask,
        [
            c
            for c in [
                "image_path",
                "label",
                "source",
                "group_id",
                "patient_id",
                "exam_id",
            ]
            if c in feats.columns
        ],
    ]
    summary, best, test_scores, best_model = train_select_eval_tabular(
        build_c_models(cfg["seed"]),
        scaler.transform(X[train_mask]),
        y[train_mask],
        scaler.transform(X[val_mask]),
        y[val_mask],
        scaler.transform(X[test_mask]),
        y[test_mask],
        test_meta,
        plus_class_index(cfg),
        cfg["data"]["class_names"],
        threshold_method=ev.get("threshold_method", "youden"),
        model_select=ev.get("model_select", "val_f1_plus"),
        n_boot=ev.get("bootstrap_n", 500),
        seed=cfg["seed"],
    )
    po = summary["metrics"][best]["test"]["plus_ovr"] if best else {}
    val_scores = (
        scores_from_classifier(
            best_model, scaler.transform(X[val_mask]), plus_class_index(cfg)
        )[0]
        if best_model is not None
        else np.array([])
    )
    return {
        "holdout": holdout,
        "branch": "C",
        "best_model": best,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_groups_test": int(feats.loc[test_mask, "group_id"].nunique()),
        "plus_ovr": po,
        "bootstrap_ci": summary["metrics"][best]["test"].get("bootstrap_ci") if best else None,
        "test_scores": test_scores.tolist() if test_scores is not None else [],
        "test_labels": y[test_mask].tolist(),
        "test_meta": test_meta.to_dict("records"),
        "val_scores": val_scores.tolist(),
        "val_labels": y[val_mask].tolist(),
        "val_meta": feats.loc[val_mask, test_meta.columns].to_dict("records"),
    }


def extract_emb_for_paths(ckpt_path: Path, image_paths: list[str], cfg: dict) -> np.ndarray:
    import torch

    from src.classify.branch_b_cnn import BackboneClassifier
    from src.classify.branch_c_hybrid import extract_embeddings

    device = get_device()
    state = torch.load(str(ckpt_path), map_location=device)
    backbone = state.get("backbone", cfg["classify_b"]["backbone"])
    sd = state["state_dict"]
    n_cls = int(sd["head.weight"].shape[0]) if "head.weight" in sd else num_classes(cfg)
    model = BackboneClassifier(backbone, num_classes=n_cls).to(device)
    model.load_state_dict(sd)
    return extract_embeddings(model, image_paths, cfg["classify_b"]["img_size"], device)


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg["seed"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--holdouts", nargs="*", default=None, help="sources to hold out (default: all)")
    ap.add_argument("--epochs", type=int, default=30, help="Branch B epochs per holdout (default 30)")
    ap.add_argument("--skip-b", action="store_true", help="only run Branch A LOSO")
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    feats_path = cfg["paths"]["features_dir"] / "biomarker_features.csv"
    if not feats_path.exists():
        raise SystemExit(f"Missing {feats_path}")
    feats_all = pd.read_csv(feats_path)
    sources = sorted(feats_all["source"].dropna().unique())
    holdouts = args.holdouts or sources
    for h in holdouts:
        if h not in sources:
            raise SystemExit(f"Unknown holdout {h}; have {sources}")

    res_dir = cfg["paths"]["results_dir"]
    loo_dir = res_dir / "loo"
    loo_dir.mkdir(parents=True, exist_ok=True)
    splits_tmp = cfg["paths"]["splits_dir"] / "loo_tmp"
    splits_tmp.mkdir(parents=True, exist_ok=True)

    partial_path = loo_dir / "loo_partial.json"
    split_sha256 = _sha256(cfg["paths"]["splits_dir"] / "all.csv")
    config_sha256 = _sha256(cfg["_root"] / "configs" / "config.yaml")
    existing = json.loads(partial_path.read_text()) if partial_path.exists() else []
    results_by_key = {
        (row["holdout"], row["branch"], split_sha256, config_sha256): row
        for row in existing
        if "holdout" in row and "branch" in row
        and row.get("split_sha256") == split_sha256
        and row.get("config_sha256") == config_sha256
    }

    def save_result(row: dict) -> None:
        row["split_sha256"] = split_sha256
        row["config_sha256"] = config_sha256
        results_by_key[
            (row["holdout"], row["branch"], split_sha256, config_sha256)
        ] = row
        ordered = [
            results_by_key[key]
            for key in sorted(results_by_key, key=lambda item: (item[0], item[1]))
        ]
        _atomic_write_json(partial_path, ordered)

    private_prediction_keys = {
        "test_paths",
        "test_scores",
        "test_labels",
        "test_meta",
        "val_paths",
        "val_scores",
        "val_labels",
        "val_meta",
    }

    for holdout in holdouts:
        print(f"\n======== LOSO holdout={holdout} ========")
        test_df = feats_all[feats_all["source"] == holdout].copy()
        train_pool = feats_all[feats_all["source"] != holdout].copy()
        if test_df.empty or train_pool.empty:
            print(f"[skip] empty for {holdout}")
            continue
        if (test_df["label"] == plus_class_index(cfg)).sum() < 5:
            print(f"[warn] few Plus in holdout {holdout}; metrics may be unstable")

        in_dom = _split_in_domain(train_pool, args.val_frac, cfg["seed"])
        test_df = test_df.copy()
        test_df["split"] = "test"
        tagged = pd.concat([in_dom, test_df], ignore_index=True)

        # Branch A
        a_res = run_branch_a(tagged, cfg, holdout)
        for split_name in ["val", "test"]:
            _write_predictions(
                loo_dir / f"{holdout}_branch_a_{split_name}_preds.csv",
                pd.DataFrame(a_res[f"{split_name}_meta"]),
                a_res[f"{split_name}_labels"],
                a_res[f"{split_name}_scores"],
                "p_plus_branch_a",
            )
        save_result(
            {k: v for k, v in a_res.items() if k not in private_prediction_keys}
        )
        print(f"[A] holdout={holdout} AUC={a_res['plus_ovr'].get('auc')} sens={a_res['plus_ovr'].get('sensitivity')}")

        if args.skip_b:
            continue

        # Write CSVs for B
        for name in ("train", "val", "test"):
            part = tagged[tagged["split"] == name][
                [
                    c
                    for c in [
                        "image_path",
                        "label",
                        "source",
                        "group_id",
                        "patient_id",
                        "exam_id",
                        "identity_level",
                    ]
                    if c in tagged.columns
                ]
            ].copy()
            part.to_csv(splits_tmp / f"{holdout}_{name}.csv", index=False)

        b_res = run_branch_b(
            splits_tmp / f"{holdout}_train.csv",
            splits_tmp / f"{holdout}_val.csv",
            splits_tmp / f"{holdout}_test.csv",
            cfg,
            holdout,
            args.epochs,
            loo_dir,
        )
        for split_name in ["val", "test"]:
            _write_predictions(
                loo_dir / f"{holdout}_branch_b_{split_name}_preds.csv",
                pd.DataFrame(b_res[f"{split_name}_meta"]),
                b_res[f"{split_name}_labels"],
                b_res[f"{split_name}_scores"],
                "p_plus_branch_b",
            )
        # strip large arrays from saved summary later
        save_result(
            {
                k: v
                for k, v in b_res.items()
                if k not in private_prediction_keys
            }
        )
        print(f"[B] holdout={holdout} AUC={b_res['plus_ovr'].get('auc')} sens={b_res['plus_ovr'].get('sensitivity')}")

        emb = extract_emb_for_paths(Path(b_res["ckpt"]), tagged["image_path"].tolist(), cfg)
        c_res = run_branch_c(tagged, emb, cfg, holdout)
        for split_name in ["val", "test"]:
            _write_predictions(
                loo_dir / f"{holdout}_branch_c_{split_name}_preds.csv",
                pd.DataFrame(c_res[f"{split_name}_meta"]),
                c_res[f"{split_name}_labels"],
                c_res[f"{split_name}_scores"],
                "p_plus_branch_c",
            )
        save_result(
            {k: v for k, v in c_res.items() if k not in private_prediction_keys}
        )
        print(f"[C] holdout={holdout} AUC={c_res['plus_ovr'].get('auc')} sens={c_res['plus_ovr'].get('sensitivity')}")

    # Flat summary table
    results = [
        results_by_key[key]
        for key in sorted(results_by_key, key=lambda item: (item[0], item[1]))
    ]
    flat = []
    for r in results:
        po = r.get("plus_ovr") or {}
        flat.append({
            "holdout": r["holdout"],
            "branch": r["branch"],
            "auc": po.get("auc"),
            "sensitivity": po.get("sensitivity"),
            "specificity": po.get("specificity"),
            "f1": po.get("f1"),
            "threshold": po.get("threshold"),
            "n_test": r.get("n_test"),
            "n_groups_test": r.get("n_groups_test"),
            "best_model": r.get("best_model") or r.get("backbone"),
        })
    summary = pd.DataFrame(flat)
    expected = {(source, branch) for source in sources for branch in ["A", "B", "C"]}
    observed = {(row["holdout"], row["branch"]) for row in results}
    complete = observed == expected
    summary_name = "loo_summary.csv" if complete else "loo_summary_partial.csv"
    results_name = "loo_results.json" if complete else "loo_results_partial.json"
    summary_tmp = res_dir / f"{summary_name}.tmp"
    summary.to_csv(summary_tmp, index=False)
    summary_tmp.replace(res_dir / summary_name)
    _atomic_write_json(res_dir / results_name, {"results": results, "protocol": {
            "holdouts": holdouts,
            "epochs_b": None if args.skip_b else args.epochs,
            "val_frac": args.val_frac,
            "skip_b": args.skip_b,
            "complete_cartesian_product": complete,
            "expected_rows": 9,
            "observed_rows": len(observed),
        }})
    print("\n=== LOSO summary ===")
    print(summary.to_string(index=False))
    print(f"[done] complete={complete} -> {res_dir / summary_name}")


if __name__ == "__main__":
    main()
