"""CLI: python -m src.classify.next_architecture <audit|train|cv|compare|loso|vessel-audit|summarize|mil-check>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.classify.next_architecture.config import (
    CANONICAL_SPLIT_SHA,
    load_next_config,
    verify_canonical_split,
)
from src.classify.next_architecture.folds import (
    assert_no_group_overlap,
    assert_no_test_images,
    load_or_create_folds,
)
from src.classify.next_architecture.metrics import paired_grouped_delta_auc, plus_binary
from src.classify.next_architecture.mil import assert_mil_blocked, mil_readiness, refuse_group_id_mil
from src.classify.next_architecture.trainer import train_one_run
from src.utils.common import ROOT


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=str(ROOT / "configs" / "next_architecture.yaml"))
    p.add_argument("--experiment", default=None)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument(
        "--allow-test-evaluation",
        action="store_true",
        help="Unlock canonical test.csv. Prints a warning; test is not pristine.",
    )
    p.add_argument("--allow-noncanonical", action="store_true")
    p.add_argument("--backbone", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accumulation", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")


def cmd_audit(args: argparse.Namespace) -> int:
    cfg = load_next_config(args.config)
    split_sha = verify_canonical_split(cfg, allow_noncanonical=args.allow_noncanonical)
    splits = cfg["paths"]["splits_dir"]
    all_df = pd.read_csv(splits / "all.csv")
    train_df = pd.read_csv(splits / "train.csv")
    val_df = pd.read_csv(splits / "val.csv")
    test_df = pd.read_csv(splits / "test.csv")
    assert_no_group_overlap(train_df, val_df)
    assert_no_test_images(pd.concat([train_df, val_df], ignore_index=True), test_df)
    mil = mil_readiness(all_df)
    masks = cfg["paths"]["masks_dir"]
    n_masks = len(list(Path(masks).glob("*.png"))) if Path(masks).exists() else 0
    prob_dir = Path(cfg["paths"]["vessel_prob_dir"])
    n_prob = len(list(prob_dir.glob("*.npy"))) if prob_dir.exists() else 0
    report = {
        "split_sha256": split_sha,
        "expected_split_sha256": cfg.get("expected_split_sha256", CANONICAL_SPLIT_SHA),
        "split_ok": split_sha == cfg.get("expected_split_sha256", CANONICAL_SPLIT_SHA),
        "n_all": int(len(all_df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "group_overlap_train_val": 0,
        "farabi_grouping": cfg.get("farabi_grouping"),
        "allow_test_evaluation_default": bool(cfg.get("allow_test_evaluation", False)),
        "n_binary_masks_png": n_masks,
        "n_soft_prob_npy": n_prob,
        "soft_prob_ready": n_prob > 0,
        "mil": mil,
        "canonical_artifacts_untouched_policy": True,
        "branch_b_input": {
            "backbone": "efficientnet_b5",
            "img_size": 224,
            "resize": "torchvision.Resize((224,224)) — distorts aspect ratio",
            "loss": "3-class CrossEntropy with train-only inverse-frequency weights",
            "threshold": "Youden on validation P(Plus) from softmax[:, 2]",
            "test_loader": "Branch B always evaluates test.csv after training; next_architecture does not by default",
        },
        "soft_vessel_current": (
            "infer_masks computes sigmoid probabilities then discards them after threshold 0.20; "
            "only binary PNG masks are stored unless --save-probability-maps is used."
        ),
        "grouped_cv_neural": "Previous grouped_cv.py is Branch A tabular only. Neural folds live in this package.",
        "loso_neural": "Previous LOSO retrains A/B/C. Next-architecture LOSO is src.classify.next_architecture loso.",
    }
    print(json.dumps(report, indent=2))
    if not report["split_ok"]:
        return 2
    if not mil["ready"]:
        print("[mil] blocked (expected):", mil["message"])
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_next_config(args.config)
    if getattr(args, "backbone", None):
        cfg["backbone"] = args.backbone
    if getattr(args, "batch_size", None):
        cfg["batch_size"] = int(args.batch_size)
    if getattr(args, "grad_accumulation", None):
        cfg["grad_accumulation"] = int(args.grad_accumulation)
    experiment = (args.experiment or cfg.get("experiment_id") or "E0").upper()
    train_one_run(
        cfg,
        experiment=experiment,
        resolution=args.resolution,
        seed=args.seed,
        smoke_test=args.smoke_test,
        allow_test_evaluation=args.allow_test_evaluation or bool(cfg.get("allow_test_evaluation")),
        allow_noncanonical=args.allow_noncanonical,
        resume=not getattr(args, "no_resume", False),
    )
    return 0


def cmd_cv(args: argparse.Namespace) -> int:
    cfg = load_next_config(args.config)
    experiment = (args.experiment or cfg.get("experiment_id") or "E3").upper()
    splits = cfg["paths"]["splits_dir"]
    train_df = pd.read_csv(splits / "train.csv")
    val_df = pd.read_csv(splits / "val.csv")
    test_df = pd.read_csv(splits / "test.csv")
    dev = pd.concat([train_df, val_df], ignore_index=True)
    assert_no_test_images(dev, test_df)
    n_folds = int(cfg.get("grouped_cv", {}).get("folds", 5))
    seed = int(args.seed or cfg.get("seed", 42))
    if cfg.get("grouped_cv", {}).get("use_official_split_as_reduced_mode", True) and not args.full_nested:
        print(
            "[protocol] reduced development mode: official train/val. "
            "This is NOT nested grouped CV."
        )
        train_one_run(
            cfg,
            experiment=experiment,
            resolution=args.resolution,
            seed=seed,
            smoke_test=args.smoke_test,
            allow_test_evaluation=False,
            resume=not getattr(args, "no_resume", False),
        )
        return 0
    manifest_path = Path(cfg["paths"]["fold_manifest"])
    payload = load_or_create_folds(
        manifest_path, dev, n_folds, seed, regenerate=args.regenerate_folds
    )
    print(f"[folds] {manifest_path} sha={payload.get('manifest_sha256')}")
    for fold in payload["folds"]:
        tr = dev.iloc[fold["train_index"]].copy()
        va = dev.iloc[fold["val_index"]].copy()
        side = cfg["paths"]["results_dir"] / "folds" / f"seed{seed}" / f"fold{fold['fold']}"
        side.mkdir(parents=True, exist_ok=True)
        tr.to_csv(side / "train.csv", index=False)
        va.to_csv(side / "val.csv", index=False)
        print(f"[fold {fold['fold']}] n_train={len(tr)} n_val={len(va)}")
        train_one_run(
            cfg,
            experiment=experiment,
            resolution=args.resolution,
            seed=seed,
            smoke_test=args.smoke_test,
            allow_test_evaluation=False,
            resume=not getattr(args, "no_resume", False),
            split_dir=side,
            fold_id=str(fold["fold"]),
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    if not args.development_only:
        raise SystemExit("compare requires --development-only (canonical test must not decide architecture).")
    ref = Path(args.reference)
    cand = Path(args.candidate)
    a = pd.read_csv(ref)
    b = pd.read_csv(cand)
    merged = a.merge(b, on="image_path", suffixes=("_ref", "_cand"))
    if len(merged) == 0:
        raise SystemExit("No overlapping image_path rows.")
    y = plus_binary(merged["label_ref"].values)
    groups = merged["group_id_ref"].values
    delta = paired_grouped_delta_auc(
        y,
        merged["plus_prob_raw_ref"].values,
        merged["plus_prob_raw_cand"].values,
        groups,
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )
    payload = {
        "reference": str(ref),
        "candidate": str(cand),
        "n": int(len(merged)),
        "development_only": True,
        "canonical_test_used": False,
        "delta_auc_candidate_minus_reference": delta,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    return 0


def cmd_loso(args: argparse.Namespace) -> int:
    cfg = load_next_config(args.config)
    verify_canonical_split(cfg, allow_noncanonical=args.allow_noncanonical)
    print(
        "LOSO for next-architecture: hold out one entire source; train/val on remaining sources only. "
        "Held-out-source labels must not be used for early stopping or threshold selection."
    )
    splits = cfg["paths"]["splits_dir"]
    all_df = pd.read_csv(splits / "all.csv")
    sources = sorted(all_df["source"].unique())
    plan = []
    for holdout in sources:
        rest = all_df[all_df["source"] != holdout]
        hold = all_df[all_df["source"] == holdout]
        plan.append(
            {
                "holdout": holdout,
                "n_rest": int(len(rest)),
                "n_holdout": int(len(hold)),
                "note": "Do not tune on holdout labels. Threshold from in-domain val only.",
            }
        )
    print(json.dumps({"experiment": args.experiment, "plan": plan}, indent=2))
    if args.smoke_test and not args.run:
        print("[loso-smoke] plan only; no training.")
        return 0
    if not args.run:
        print("[loso] Pass --run to train sequentially (long GPU job). Canonical test stays locked.")
        return 0
    from src.classify.next_architecture.loso import run_loso_training

    experiment = (args.experiment or "E2").upper()
    resolution = int(args.resolution or cfg.get("resolution", 384))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    return run_loso_training(
        cfg,
        experiment=experiment,
        resolution=resolution,
        seed=seed,
        smoke_test=bool(args.smoke_test),
        resume=not getattr(args, "no_resume", False),
    )


def cmd_vessel_audit(args: argparse.Namespace) -> int:
    cfg = load_next_config(args.config)
    verify_canonical_split(cfg, allow_noncanonical=args.allow_noncanonical)
    from src.classify.next_architecture.vessel_audit import run_vessel_audit, write_vessel_overlays

    if getattr(args, "overlays_only", False):
        write_vessel_overlays(
            cfg,
            n_per_cell=int(args.overlays or 2),
            max_side=int(args.max_side),
        )
        return 0
    run_vessel_audit(cfg, max_side=int(args.max_side), limit=args.limit)
    if int(getattr(args, "overlays", 0) or 0) > 0:
        write_vessel_overlays(cfg, n_per_cell=int(args.overlays), max_side=int(args.max_side))
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    cfg = load_next_config(args.config)
    from src.classify.next_architecture.summary import write_summary

    dest = Path(args.out) if args.out else Path(cfg["paths"]["results_dir"]) / "SUMMARY.json"
    payload = write_summary(Path(cfg["paths"]["results_dir"]), dest)
    print(json.dumps({k: payload[k] for k in payload if k != "runs"}, indent=2, default=str))
    print(f"[summary] n_runs={payload['n_runs']} all_test_skipped={payload['all_test_skipped']} -> {dest}")
    return 0


def cmd_sampler_audit(args: argparse.Namespace) -> int:
    cfg = load_next_config(args.config)
    verify_canonical_split(cfg, allow_noncanonical=args.allow_noncanonical)
    from src.classify.next_architecture.sampling import audit_image_level_source_sampler, write_sampler_audit

    splits = cfg["paths"]["splits_dir"]
    train_df = pd.read_csv(splits / "train.csv")
    payload = audit_image_level_source_sampler(
        train_df, plus_idx=int(cfg.get("plus_class_index", 2))
    )
    dest = Path(args.out) if args.out else Path(cfg["paths"]["results_dir"]) / "E8" / "sampler_audit.json"
    write_sampler_audit(payload, dest)
    print(json.dumps({k: payload[k] for k in payload if k != "top_oversampled_groups"}, indent=2, default=str))
    print(f"[sampler-audit] -> {dest}")
    return 0


def cmd_e4_audit(args: argparse.Namespace) -> int:
    cfg = load_next_config(args.config)
    from src.classify.next_architecture.e4_definition import write_e4_definition_audit

    dest = Path(args.out) if args.out else Path(cfg["paths"]["results_dir"]) / "hybrid_followup" / "e4_definition_audit.json"
    payload = write_e4_definition_audit(dest)
    print(json.dumps(payload, indent=2, default=str))
    print(f"[e4-audit] -> {dest}")
    return 0


def cmd_h1(args: argparse.Namespace) -> int:
    from src.classify.next_architecture.h1 import run_h1

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    payload = run_h1(Path(args.oof), Path(args.val), dest, alpha_ridge=float(args.ridge))
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_pack_report(args: argparse.Namespace) -> int:
    cfg = load_next_config(args.config)
    verify_canonical_split(cfg, allow_noncanonical=args.allow_noncanonical)
    from src.classify.next_architecture.report_bundle import pack_report

    dest = Path(args.out) if args.out else Path(cfg["paths"]["results_dir"]) / "report_bundle"
    payload = pack_report(Path(cfg["paths"]["results_dir"]), dest)
    print(json.dumps(payload.get("e8_decision"), indent=2))
    print(f"[pack-report] -> {dest}")
    return 0


def cmd_e5x_gate(args: argparse.Namespace) -> int:
    cfg = load_next_config(args.config)
    from src.classify.next_architecture.e5x_gate import write_e5x_gate

    evidence_path = Path(args.evidence) if args.evidence else None
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path and evidence_path.exists() else {}
    dest = Path(args.out) if args.out else Path(cfg["paths"]["results_dir"]) / "hybrid_followup" / "e5x_gate.json"
    payload = write_e5x_gate(dest, evidence)
    print(json.dumps(payload, indent=2, default=str))
    if payload["train_e5x"]:
        print("[e5x-gate] PASSED. Historical E5 still skipped. Train E5X separately. Do not --force-e5.")
    else:
        print(f"[e5x-gate] FAILED blocker={payload.get('blocker')}. Do not train E5X. Do not --force-e5.")
    return 0


def cmd_mil(args: argparse.Namespace) -> int:
    cfg = load_next_config(args.config)
    frame = pd.read_csv(cfg["paths"]["splits_dir"] / "all.csv")
    print(json.dumps(mil_readiness(frame), indent=2))
    if args.force_group_id:
        refuse_group_id_mil()
    try:
        assert_mil_blocked(frame)
    except Exception as exc:
        print(str(exc))
        return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.classify.next_architecture")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_audit = sub.add_parser("audit")
    _add_common(p_audit)

    p_train = sub.add_parser("train")
    _add_common(p_train)

    p_cv = sub.add_parser("cv")
    _add_common(p_cv)
    p_cv.add_argument("--full-nested", action="store_true")
    p_cv.add_argument("--regenerate-folds", action="store_true")

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--reference", required=True, help="val_predictions.csv of reference run")
    p_cmp.add_argument("--candidate", required=True, help="val_predictions.csv of candidate run")
    p_cmp.add_argument("--development-only", action="store_true", required=True)
    p_cmp.add_argument("--n-boot", type=int, default=2000)
    p_cmp.add_argument("--seed", type=int, default=42)
    p_cmp.add_argument("--out", default=None)

    p_loso = sub.add_parser("loso")
    _add_common(p_loso)
    p_loso.add_argument("--run", action="store_true")

    p_va = sub.add_parser("vessel-audit")
    _add_common(p_va)
    p_va.add_argument("--max-side", type=int, default=384)
    p_va.add_argument("--limit", type=int, default=None)
    p_va.add_argument("--overlays", type=int, default=0, help="Write N overlays per source×plus cell")
    p_va.add_argument("--overlays-only", action="store_true")

    p_sum = sub.add_parser("summarize")
    _add_common(p_sum)
    p_sum.add_argument("--out", default=None)

    p_mil = sub.add_parser("mil-check")
    _add_common(p_mil)
    p_mil.add_argument("--force-group-id", action="store_true")

    p_sa = sub.add_parser("sampler-audit")
    _add_common(p_sa)
    p_sa.add_argument("--out", default=None)

    p_e4a = sub.add_parser("e4-audit")
    _add_common(p_e4a)
    p_e4a.add_argument("--out", default=None)

    p_h1 = sub.add_parser("h1")
    p_h1.add_argument("--oof", required=True)
    p_h1.add_argument("--val", required=True)
    p_h1.add_argument("--out", required=True)
    p_h1.add_argument("--ridge", type=float, default=10.0)

    p_g = sub.add_parser("e5x-gate")
    _add_common(p_g)
    p_g.add_argument("--evidence", default=None)
    p_g.add_argument("--out", default=None)

    p_pack = sub.add_parser("pack-report")
    _add_common(p_pack)
    p_pack.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "audit":
        return cmd_audit(args)
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "cv":
        return cmd_cv(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "loso":
        return cmd_loso(args)
    if args.cmd == "vessel-audit":
        return cmd_vessel_audit(args)
    if args.cmd == "summarize":
        return cmd_summarize(args)
    if args.cmd == "mil-check":
        return cmd_mil(args)
    if args.cmd == "sampler-audit":
        return cmd_sampler_audit(args)
    if args.cmd == "e4-audit":
        return cmd_e4_audit(args)
    if args.cmd == "h1":
        return cmd_h1(args)
    if args.cmd == "e5x-gate":
        return cmd_e5x_gate(args)
    if args.cmd == "pack-report":
        return cmd_pack_report(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
