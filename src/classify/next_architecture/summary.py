"""Collect next-architecture results.json into one development summary. Test stays locked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


LOCKED_PHASE5 = {
    "B_test_plus_ovr_auc": 0.928,
    "C_fusion_test_plus_ovr_auc": 0.912,
    "concat_hurt": "embedding_only 0.916 → fusion 0.912",
    "legacy_0_98": "leaky image-level split; invalid",
    "mil": "blocked; no eye_id/visit_id",
    "canonical_test": "not pristine",
}


def _auc(payload: dict[str, Any]) -> float | None:
    try:
        return float((payload.get("val_raw") or {}).get("auc"))
    except (TypeError, ValueError):
        return None


def _hold_auc(payload: dict[str, Any]) -> float | None:
    hold = payload.get("holdout") or {}
    if hold.get("skipped", True):
        return None
    try:
        return float((hold.get("metrics_raw_at_val_threshold") or {}).get("auc"))
    except (TypeError, ValueError):
        return None


def collect_run_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("E*/res*/seed*/fold*/results.json")):
        if "smoke" in path.as_posix():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("smoke_test"):
            continue
        hold = payload.get("holdout") or {}
        rows.append(
            {
                "experiment": str(payload.get("experiment", "")).upper(),
                "resolution": payload.get("resolution"),
                "seed": payload.get("seed"),
                "fold": payload.get("fold"),
                "val_auc": _auc(payload),
                "worst_source_auc": payload.get("worst_source_auc"),
                "holdout_auc": _hold_auc(payload),
                "holdout_n": hold.get("n") if not hold.get("skipped", True) else None,
                "best_epoch": (payload.get("provenance") or {}).get("best_epoch"),
                "source_balance": payload.get("source_balance"),
                "path": str(path.parent),
                "test_skipped": bool((payload.get("test") or {}).get("skipped", True)),
            }
        )
    return rows


def _mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def build_summary(results_root: Path) -> dict[str, Any]:
    rows = collect_run_rows(results_root)
    e0 = [r for r in rows if r["experiment"] == "E0" and r["fold"] == "official"]
    e2 = [r for r in rows if r["experiment"] == "E2" and r["fold"] == "official"]
    loso = [r for r in rows if str(r["fold"]).startswith("loso_")]
    e8 = [r for r in rows if r["experiment"] == "E8"]
    e8_decision = None
    decision_path = results_root / "hybrid_followup" / "e8_decision.json"
    if decision_path.exists():
        e8_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    return {
        "protocol": "official_train_val_not_nested_cv unless fold starts with loso_",
        "canonical_test_used": False,
        "do_not_compare_val_to_branch_b_test_0_928": True,
        "locked_phase5": LOCKED_PHASE5,
        "e0_val": _mean_std([r["val_auc"] for r in e0 if r["val_auc"] is not None]),
        "e2_official_val": _mean_std([r["val_auc"] for r in e2 if r["val_auc"] is not None]),
        "loso": [
            {
                "experiment": r["experiment"],
                "fold": r["fold"],
                "in_domain_val_auc": r["val_auc"],
                "holdout_auc": r["holdout_auc"],
                "holdout_n": r["holdout_n"],
            }
            for r in loso
        ],
        "e8": e8,
        "e8_decision": e8_decision,
        "n_runs": len(rows),
        "all_test_skipped": all(r["test_skipped"] for r in rows),
        "runs": rows,
    }


def write_summary(results_root: Path, dest: Path | None = None) -> dict[str, Any]:
    payload = build_summary(results_root)
    out = dest or (results_root / "SUMMARY.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return payload
