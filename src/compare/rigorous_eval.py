"""Rigorous evaluation report: CIs, 3-class metrics, ablation, per-source breakdown.

Usage:
    python -m src.compare.rigorous_eval
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.common import ensure_dirs, load_config, plus_class_index


def _plus_from_branch(payload: dict | None, branch: str) -> dict | None:
    if payload is None:
        return None
    if branch in {"a", "c"}:
        best = payload.get("best_model")
        metrics = payload.get("metrics", {})
        if best and best in metrics:
            node = metrics[best]
            if "test" in node and "plus_ovr" in node["test"]:
                return node["test"]["plus_ovr"]
            return node.get("val") or node
    if branch == "b":
        if "test" in payload and "plus_ovr" in payload["test"]:
            return payload["test"]["plus_ovr"]
        return payload.get("metrics")
    return None


def _ci_row(payload: dict | None, branch: str) -> dict | None:
    if payload is None:
        return None
    if branch in {"a", "c"}:
        best = payload.get("best_model")
        node = payload.get("metrics", {}).get(best, {})
        return node.get("test", {}).get("bootstrap_ci")
    if branch == "b":
        return payload.get("test", {}).get("bootstrap_ci")
    return None


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    res = cfg["paths"]["results_dir"]
    plus_idx = plus_class_index(cfg)

    a = json.loads((res / "branch_a_results.json").read_text(encoding="utf-8")) if (res / "branch_a_results.json").exists() else None
    b = json.loads((res / "branch_b_results.json").read_text(encoding="utf-8")) if (res / "branch_b_results.json").exists() else None
    c = json.loads((res / "branch_c_results.json").read_text(encoding="utf-8")) if (res / "branch_c_results.json").exists() else None

    rows = []
    for name, payload, tag in [
        ("Branch A", a, "a"),
        ("Branch B", b, "b"),
        ("Branch C fusion", c, "c"),
    ]:
        m = _plus_from_branch(payload, tag)
        ci = _ci_row(payload, tag)
        if not m:
            continue
        row = {
            "branch": name,
            "auc": m.get("auc"),
            "sensitivity": m.get("sensitivity"),
            "specificity": m.get("specificity"),
            "f1": m.get("f1"),
            "threshold": m.get("threshold") or payload.get("best_val_threshold") or payload.get("val_threshold"),
        }
        if ci:
            for k in ("auc", "sensitivity", "specificity", "f1"):
                if k in ci:
                    row[f"{k}_ci"] = f"[{ci[k]['ci_low']:.3f}, {ci[k]['ci_high']:.3f}]"
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(res / "rigorous_summary.csv", index=False)
    print(summary.to_string(index=False))

    ablation_rows = []
    if c and "ablation" in c:
        for ab_name, ab in c["ablation"].items():
            best = ab.get("best_model")
            node = ab.get("metrics", {}).get(best, {})
            po = node.get("test", {}).get("plus_ovr", {})
            ablation_rows.append(
                {
                    "variant": ab_name,
                    "best_model": best,
                    "test_auc": po.get("auc"),
                    "test_sens": po.get("sensitivity"),
                    "test_spec": po.get("specificity"),
                    "test_f1": po.get("f1"),
                    "val_threshold": ab.get("best_val_threshold"),
                }
            )
    if ablation_rows:
        ab_df = pd.DataFrame(ablation_rows)
        ab_df.to_csv(res / "hybrid_ablation.csv", index=False)
        print("\n[ablation]")
        print(ab_df.to_string(index=False))

    report = {
        "eval_protocol": cfg.get("eval", {}),
        "plus_class_index": plus_idx,
        "summary": rows,
        "ablation": ablation_rows,
        "per_source": {
            "branch_a": a.get("metrics", {}).get(a.get("best_model", ""), {}).get("per_source_test") if a else None,
            "branch_b": b.get("per_source_test") if b else None,
            "branch_c": c.get("metrics", {}).get(c.get("best_model", ""), {}).get("per_source_test") if c else None,
        },
    }
    with open(res / "rigorous_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] -> {res / 'rigorous_eval.json'}")


if __name__ == "__main__":
    main()
