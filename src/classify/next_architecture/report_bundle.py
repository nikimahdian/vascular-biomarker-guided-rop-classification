"""Pack development artifacts for thesis writing. Never copies canonical test predictions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.classify.next_architecture.e5x_gate import build_e5x_evidence, write_e5x_gate
from src.classify.next_architecture.metrics import paired_grouped_delta_auc, plus_binary

MATERIAL_AUC = 0.01
MEANINGFUL_FARABI = 0.01


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hold_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    hold = payload.get("holdout") or {}
    if hold.get("skipped", True):
        return {}
    return dict(hold.get("metrics_raw_at_val_threshold") or {})


def _find_run(root: Path, experiment: str, resolution: int, seed: int, fold_prefix: str) -> Path | None:
    parent = root / experiment / f"res{resolution}" / f"seed{seed}"
    if not parent.exists():
        return None
    hits = sorted(parent.glob(f"{fold_prefix}*"))
    with_json = [p for p in hits if (p / "results.json").exists()]
    return with_json[-1] if with_json else None


def _paired_from_preds(ref_csv: Path, cand_csv: Path, n_boot: int, seed: int) -> dict[str, Any]:
    a = pd.read_csv(ref_csv)
    b = pd.read_csv(cand_csv)
    merged = a.merge(b, on="image_path", suffixes=("_ref", "_cand"))
    if len(merged) == 0:
        raise RuntimeError(f"No overlap {ref_csv} vs {cand_csv}")
    y = plus_binary(merged["label_ref"].values)
    groups = merged["group_id_ref"].values
    delta = paired_grouped_delta_auc(
        y,
        merged["plus_prob_raw_ref"].values,
        merged["plus_prob_raw_cand"].values,
        groups,
        n_boot=n_boot,
        seed=seed,
    )
    return {"n": int(len(merged)), "n_groups": int(pd.Series(groups).nunique()), **delta}


def e8_decision(
    *,
    official_delta: float,
    farabi_delta: float,
    farfum_delta: float,
    plus_delta: float,
) -> dict[str, Any]:
    farabi_up = farabi_delta >= MEANINGFUL_FARABI
    pooled_ok = official_delta >= -MATERIAL_AUC
    farfum_ok = farfum_delta >= -MATERIAL_AUC
    plus_ok = plus_delta >= -MATERIAL_AUC
    if farabi_up and pooled_ok and farfum_ok and plus_ok:
        verdict = "supported"
    elif farabi_up and (not plus_ok or not farfum_ok or not pooled_ok):
        verdict = "mixed"
    else:
        verdict = "not_supported"
    reasons = []
    if farabi_up:
        reasons.append(f"Farabi LOSO ΔAUC={farabi_delta:+.4f} (>= {MEANINGFUL_FARABI})")
    else:
        reasons.append(f"Farabi LOSO ΔAUC={farabi_delta:+.4f} not meaningful")
    if not plus_ok:
        reasons.append(f"Plus LOSO ΔAUC={plus_delta:+.4f} material drop (margin {MATERIAL_AUC})")
    if not farfum_ok:
        reasons.append(f"FARFUM LOSO ΔAUC={farfum_delta:+.4f} material drop")
    if not pooled_ok:
        reasons.append(f"official val ΔAUC={official_delta:+.4f} material drop")
    return {
        "verdict": verdict,
        "run_more_e8_seeds": verdict == "supported",
        "material_auc_margin": MATERIAL_AUC,
        "reasons": reasons,
        "deltas": {
            "official_val": official_delta,
            "loso_farabi": farabi_delta,
            "loso_farfum_rop": farfum_delta,
            "loso_plus": plus_delta,
        },
    }


def _roc_png(path: Path, curves: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for name, y, s in curves:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y, s):.3f}")
    ax.plot([0, 1], [0, 1], color="0.7", ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Official val ROC (development only)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _bar_png(path: Path, rows: list[tuple[str, float, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r[0] for r in rows]
    e2 = [r[1] for r in rows]
    e8 = [r[2] for r in rows]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - w / 2, e2, w, label="E2")
    ax.bar(x + w / 2, e8, w, label="E8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Plus-OvR AUC")
    ax.set_title("E2 vs E8 (development / LOSO holdout)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _archive_all_runs(results_root: Path, dest: Path) -> int:
    """Copy every development results.json + prediction CSV. Never copies .pt or test preds."""
    from src.classify.next_architecture.summary import collect_run_rows

    json_runs = dest / "json" / "runs"
    json_runs.mkdir(parents=True, exist_ok=True)
    preds = dest / "predictions"
    preds.mkdir(parents=True, exist_ok=True)
    rows = collect_run_rows(results_root)
    pd.DataFrame(rows).to_csv(dest / "tables" / "all_runs.csv", index=False)
    n = 0
    for row in rows:
        src = Path(row["path"])
        tag = f"{row['experiment']}_res{row['resolution']}_seed{row['seed']}_{row['fold']}"
        if (src / "results.json").exists():
            shutil.copy2(src / "results.json", json_runs / f"{tag}_results.json")
            n += 1
        for name in ("val_predictions.csv", "holdout_predictions.csv"):
            p = src / name
            if p.exists():
                shutil.copy2(p, preds / f"{tag}_{name}")
    extras = [
        results_root / "SUMMARY.json",
        results_root / "E8" / "sampler_audit.json",
        results_root / "hybrid_followup" / "e4_definition_audit.json",
        results_root / "hybrid_followup" / "e5x_gate.json",
        results_root / "hybrid_followup" / "e8_decision.json",
        results_root / "ladder" / "vessel_audit.json",
        results_root / "ladder" / "vessel_audit_rows.csv",
        results_root / "ladder" / "STATUS.json",
    ]
    extra_dir = dest / "json" / "audits"
    extra_dir.mkdir(parents=True, exist_ok=True)
    for path in extras:
        if path.exists():
            shutil.copy2(path, extra_dir / path.name)
    split = results_root.parent.parent / "data" / "splits" / "all.csv"
    if split.exists():
        data_dir = dest / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(split, data_dir / "all.csv")
        frame = pd.read_csv(split)
        counts = (
            frame.groupby(["split", "source", "label"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        counts.to_csv(dest / "tables" / "split_counts.csv", index=False)
    va_rows = results_root / "ladder" / "vessel_audit_rows.csv"
    if va_rows.exists():
        shutil.copy2(va_rows, dest / "tables" / "vessel_audit_rows.csv")
    return n


def _loso_roc_png(path: Path, e2_csv: Path, e8_csv: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    a = pd.read_csv(e2_csv)
    b = pd.read_csv(e8_csv)
    merged = a.merge(b, on="image_path", suffixes=("_e2", "_e8"))
    y = plus_binary(merged["label_e2"].values)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for name, col in (("E2", "plus_prob_raw_e2"), ("E8", "plus_prob_raw_e8")):
        fpr, tpr, _ = roc_curve(y, merged[col].values)
        ax.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y, merged[col].values):.3f}")
    ax.plot([0, 1], [0, 1], color="0.7", ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _official_rows(results_root: Path, experiment: str, seeds: tuple[int, ...] = (42, 43, 44)) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run = _find_run(results_root, experiment, 384, seed, "foldofficial_")
        if run is None:
            continue
        p = _load(run / "results.json")
        v = p.get("val_raw") or {}
        src = p.get("per_source_val") or {}
        rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "val_auc": v.get("auc"),
                "auprc": v.get("auprc"),
                "brier": v.get("brier"),
                "worst_source_auc": p.get("worst_source_auc"),
                "farabi_auc": (src.get("farabi") or {}).get("auc"),
                "farfum_auc": (src.get("farfum_rop") or {}).get("auc"),
                "plus_auc": (src.get("plus") or {}).get("auc"),
                "best_epoch": (p.get("provenance") or {}).get("best_epoch"),
            }
        )
    return rows


def _loso_rows(results_root: Path, experiment: str, seed: int = 42) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src in ("farabi", "farfum_rop", "plus"):
        run = _find_run(results_root, experiment, 384, seed, f"foldloso_{src}_")
        if run is None:
            continue
        p = _load(run / "results.json")
        hold = p.get("holdout") or {}
        m = hold.get("metrics_raw_at_val_threshold") or {}
        rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "holdout": src,
                "in_domain_val_auc": (p.get("val_raw") or {}).get("auc"),
                "holdout_auc": m.get("auc"),
                "holdout_auprc": m.get("auprc"),
                "holdout_brier": m.get("brier"),
                "n": hold.get("n"),
            }
        )
    return rows


def _e4b_vs_e2_table(results_root: Path) -> pd.DataFrame:
    rows = []
    for seed in (42, 43, 44):
        e2 = _find_run(results_root, "E2", 384, seed, "foldofficial_")
        e4b = _find_run(results_root, "E4B", 384, seed, "foldofficial_")
        if e2 is None or e4b is None:
            continue
        j2, j4 = _load(e2 / "results.json"), _load(e4b / "results.json")
        a2 = float((j2.get("val_raw") or {}).get("auc"))
        a4 = float((j4.get("val_raw") or {}).get("auc"))
        rows.append({"seed": seed, "e2_val_auc": a2, "e4b_val_auc": a4, "delta_e4b_minus_e2": a4 - a2})
    return pd.DataFrame(rows)


def pack_report(results_root: Path, dest: Path, *, n_boot: int = 2000, seed: int = 42) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    tables = dest / "tables"
    figs = dest / "figures"
    preds = dest / "predictions"
    json_dir = dest / "json"
    overlays_dest = dest / "vessel_overlays"
    for p in (tables, figs, preds, json_dir):
        p.mkdir(parents=True, exist_ok=True)

    e2_off = _find_run(results_root, "E2", 384, 42, "foldofficial_")
    e8_off = _find_run(results_root, "E8", 384, 42, "foldofficial_")
    if e2_off is None or e8_off is None:
        raise SystemExit("Need E2 and E8 official results.json")
    e2 = _load(e2_off / "results.json")
    e8 = _load(e8_off / "results.json")
    official_paired = _paired_from_preds(
        e2_off / "val_predictions.csv", e8_off / "val_predictions.csv", n_boot, seed
    )

    loso_rows = []
    loso_paired = {}
    for src in ("farabi", "farfum_rop", "plus"):
        a = _find_run(results_root, "E2", 384, 42, f"foldloso_{src}_")
        b = _find_run(results_root, "E8", 384, 42, f"foldloso_{src}_")
        if a is None or b is None:
            continue
        ja, jb = _load(a / "results.json"), _load(b / "results.json")
        ha, hb = _hold_metrics(ja), _hold_metrics(jb)
        row = {
            "holdout": src,
            "e2_in_domain_val_auc": (ja.get("val_raw") or {}).get("auc"),
            "e8_in_domain_val_auc": (jb.get("val_raw") or {}).get("auc"),
            "e2_holdout_auc": ha.get("auc"),
            "e8_holdout_auc": hb.get("auc"),
            "e2_holdout_auprc": ha.get("auprc"),
            "e8_holdout_auprc": hb.get("auprc"),
            "e2_holdout_brier": ha.get("brier"),
            "e8_holdout_brier": hb.get("brier"),
            "e2_n": ha.get("n") or (ja.get("holdout") or {}).get("n"),
            "e8_n": hb.get("n") or (jb.get("holdout") or {}).get("n"),
        }
        if ha.get("auc") is not None and hb.get("auc") is not None:
            row["delta_holdout_auc"] = float(hb["auc"]) - float(ha["auc"])
        hold_a = a / "holdout_predictions.csv"
        hold_b = b / "holdout_predictions.csv"
        if hold_a.exists() and hold_b.exists():
            loso_paired[src] = _paired_from_preds(hold_a, hold_b, n_boot, seed)
            shutil.copy2(hold_a, preds / f"E2_loso_{src}_holdout_predictions.csv")
            shutil.copy2(hold_b, preds / f"E8_loso_{src}_holdout_predictions.csv")
        loso_rows.append(row)

    def _f(x) -> float:
        return float(x) if x is not None else float("nan")

    loso_map = {r["holdout"]: r for r in loso_rows}
    decision = e8_decision(
        official_delta=_f((e8.get("val_raw") or {}).get("auc")) - _f((e2.get("val_raw") or {}).get("auc")),
        farabi_delta=_f(loso_map.get("farabi", {}).get("delta_holdout_auc")),
        farfum_delta=_f(loso_map.get("farfum_rop", {}).get("delta_holdout_auc")),
        plus_delta=_f(loso_map.get("plus", {}).get("delta_holdout_auc")),
    )

    e2_val = e2.get("val_raw") or {}
    e8_val = e8.get("val_raw") or {}
    e2_src = e2.get("per_source_val") or {}
    e8_src = e8.get("per_source_val") or {}

    official_table = pd.DataFrame(
        [
            {
                "model": "E2",
                "auc": e2_val.get("auc"),
                "auprc": e2_val.get("auprc"),
                "brier": e2_val.get("brier"),
                "ece": e2_val.get("ece"),
                "sens": e2_val.get("sensitivity"),
                "spec": e2_val.get("specificity"),
                "f1": e2_val.get("f1"),
                "threshold": e2_val.get("threshold"),
                "worst_source_auc": e2.get("worst_source_auc"),
                "farabi_auc": (e2_src.get("farabi") or {}).get("auc"),
                "farabi_spec": (e2_src.get("farabi") or {}).get("specificity"),
                "farfum_auc": (e2_src.get("farfum_rop") or {}).get("auc"),
                "plus_auc": (e2_src.get("plus") or {}).get("auc"),
            },
            {
                "model": "E8",
                "auc": e8_val.get("auc"),
                "auprc": e8_val.get("auprc"),
                "brier": e8_val.get("brier"),
                "ece": e8_val.get("ece"),
                "sens": e8_val.get("sensitivity"),
                "spec": e8_val.get("specificity"),
                "f1": e8_val.get("f1"),
                "threshold": e8_val.get("threshold"),
                "worst_source_auc": e8.get("worst_source_auc"),
                "farabi_auc": (e8_src.get("farabi") or {}).get("auc"),
                "farabi_spec": (e8_src.get("farabi") or {}).get("specificity"),
                "farfum_auc": (e8_src.get("farfum_rop") or {}).get("auc"),
                "plus_auc": (e8_src.get("plus") or {}).get("auc"),
            },
        ]
    )
    official_table.to_csv(tables / "e2_e8_official_val.csv", index=False)
    pd.DataFrame(loso_rows).to_csv(tables / "e2_e8_loso.csv", index=False)
    _e4b_vs_e2_table(results_root).to_csv(tables / "e4b_vs_e2_official.csv", index=False)
    pd.DataFrame(_official_rows(results_root, "E4B")).to_csv(tables / "e4b_official.csv", index=False)
    pd.DataFrame(_official_rows(results_root, "V1")).to_csv(tables / "v1_official.csv", index=False)
    pd.DataFrame(_loso_rows(results_root, "V1", 42)).to_csv(tables / "v1_loso_seed42.csv", index=False)

    locked = pd.DataFrame(
        [
            {"claim": "B_test_plus_ovr", "auc": 0.928, "role": "canonical_phase5"},
            {"claim": "C_fusion_test_plus_ovr", "auc": 0.912, "role": "canonical_phase5"},
            {"claim": "C_embedding_only_test", "auc": 0.916, "role": "canonical_phase5"},
            {"claim": "legacy_leaky_hybrid", "auc": 0.98, "role": "invalid_do_not_cite"},
        ]
    )
    locked.to_csv(tables / "phase5_locked.csv", index=False)

    payload = {
        "protocol": "official_train_val_not_nested_cv + LOSO",
        "canonical_test_used": False,
        "do_not_compare_val_to_b_test_0_928": True,
        "e2_official": str(e2_off),
        "e8_official": str(e8_off),
        "official_paired_e8_minus_e2": official_paired,
        "loso_paired_e8_minus_e2": loso_paired,
        "e8_decision": decision,
        "e2_val_raw": e2_val,
        "e8_val_raw": e8_val,
        "e2_high_sens": e2.get("val_high_sensitivity_0_95"),
        "e8_high_sens": e8.get("val_high_sensitivity_0_95"),
        "e2_per_source_val": e2_src,
        "e8_per_source_val": e8_src,
        "loso": loso_rows,
    }
    (json_dir / "e8_vs_e2.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (results_root / "hybrid_followup").mkdir(parents=True, exist_ok=True)
    shutil.copy2(json_dir / "e8_vs_e2.json", results_root / "hybrid_followup" / "e8_vs_e2.json")
    (results_root / "hybrid_followup" / "e8_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )

    shutil.copy2(e2_off / "val_predictions.csv", preds / "E2_official_val_predictions.csv")
    shutil.copy2(e8_off / "val_predictions.csv", preds / "E8_official_val_predictions.csv")
    shutil.copy2(e2_off / "results.json", json_dir / "E2_official_results.json")
    shutil.copy2(e8_off / "results.json", json_dir / "E8_official_results.json")
    for src in ("farabi", "farfum_rop", "plus"):
        for exp in ("E2", "E8"):
            run = _find_run(results_root, exp, 384, 42, f"foldloso_{src}_")
            if run and (run / "results.json").exists():
                shutil.copy2(run / "results.json", json_dir / f"{exp}_loso_{src}_results.json")

    sampler = results_root / "E8" / "sampler_audit.json"
    if sampler.exists():
        shutil.copy2(sampler, json_dir / "e8_sampler_audit.json")
    ov = results_root / "ladder" / "vessel_overlays"
    if ov.exists():
        if overlays_dest.exists():
            shutil.rmtree(overlays_dest)
        shutil.copytree(ov, overlays_dest)

    e2p = pd.read_csv(e2_off / "val_predictions.csv")
    e8p = pd.read_csv(e8_off / "val_predictions.csv")
    y2 = plus_binary(e2p["label"].values)
    y8 = plus_binary(e8p["label"].values)
    _roc_png(
        figs / "official_val_roc_e2_e8.png",
        [("E2", y2, e2p["plus_prob_raw"].values), ("E8", y8, e8p["plus_prob_raw"].values)],
    )
    bar_rows = [("official val", _f(e2_val.get("auc")), _f(e8_val.get("auc")))]
    for src, lab in (("farabi", "LOSO Farabi"), ("farfum_rop", "LOSO FARFUM"), ("plus", "LOSO Plus")):
        r = loso_map.get(src) or {}
        if r.get("e2_holdout_auc") is not None:
            bar_rows.append((lab, _f(r["e2_holdout_auc"]), _f(r["e8_holdout_auc"])))
    _bar_png(figs / "e2_e8_auc_bars.png", bar_rows)
    for src, lab in (("farabi", "LOSO Farabi"), ("farfum_rop", "LOSO FARFUM"), ("plus", "LOSO Plus")):
        a = preds / f"E2_loso_{src}_holdout_predictions.csv"
        b = preds / f"E8_loso_{src}_holdout_predictions.csv"
        if a.exists() and b.exists():
            _loso_roc_png(figs / f"loso_{src}_roc_e2_e8.png", a, b, f"{lab} ROC (development)")

    n_archived = _archive_all_runs(results_root, dest)

    e5x_evidence = build_e5x_evidence(results_root)
    (json_dir / "e5x_evidence.json").write_text(json.dumps(e5x_evidence, indent=2) + "\n", encoding="utf-8")
    e5x_gate = write_e5x_gate(results_root / "hybrid_followup" / "e5x_gate.json", e5x_evidence)
    shutil.copy2(json_dir / "e5x_evidence.json", json_dir / "audits" / "e5x_evidence.json")
    shutil.copy2(results_root / "hybrid_followup" / "e5x_gate.json", json_dir / "audits" / "e5x_gate.json")

    index = dest / "INDEX.md"
    index.write_text(
        "\n".join(
            [
                "# Next-architecture report bundle",
                "",
                f"E8 decision: **{decision['verdict']}**",
                f"More E8 seeds: **{decision['run_more_e8_seeds']}**",
                f"E5X gate: **{'pass' if e5x_gate.get('train_e5x') else 'fail'}** (blocker={e5x_gate.get('blocker')})",
                "",
                "- Canonical test was not used.",
                "- Do not compare these val/LOSO numbers to Branch B test 0.928.",
                "- Phase 5 locked table: `tables/phase5_locked.csv`",
                "- E2 vs E8 official: `tables/e2_e8_official_val.csv`",
                "- LOSO E2/E8: `tables/e2_e8_loso.csv`",
                "- E4B vs E2: `tables/e4b_vs_e2_official.csv`",
                "- V1 official / LOSO: `tables/v1_official.csv`, `tables/v1_loso_seed42.csv`",
                "- Paired bootstrap: `json/e8_vs_e2.json`",
                "- E5X evidence: `json/e5x_evidence.json`",
                "- ROC: `figures/official_val_roc_e2_e8.png`",
                "- Bars: `figures/e2_e8_auc_bars.png`",
                "- Predictions: `predictions/`",
                "- Vessel overlays: `vessel_overlays/`",
                f"- All run JSONs archived: {n_archived}",
                "- Master run table: `tables/all_runs.csv`",
                "- Split counts: `tables/split_counts.csv`",
                "- Audits: `json/audits/`",
                "",
                "Reasons:",
                *[f"- {r}" for r in decision["reasons"]],
                "",
            ]
        ),
        encoding="utf-8",
    )
    payload["e5x_gate"] = e5x_gate
    payload["e5x_evidence"] = e5x_evidence
    return payload
