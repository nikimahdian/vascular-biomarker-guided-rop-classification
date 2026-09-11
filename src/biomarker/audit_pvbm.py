"""Audit PVBM biomarker table quality (NaNs, geom failures, by source/class).

Does not re-extract. Writes:
    results/pvbm_audit.json
    results/pvbm_audit_summary.csv

Usage:
    python -m src.biomarker.audit_pvbm
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.biomarker.extract_pvbm import GEOM_COLUMNS
from src.utils.common import ensure_dirs, load_config

FRACTAL_COLS = ["fractal_d0", "fractal_d1", "fractal_d2", "singularity_length"]
DENSITY_PREFIX = ("vessel_", "density_")


def _feat_cols(df: pd.DataFrame) -> list[str]:
    meta = {
        "image_path",
        "mask_path",
        "label",
        "split",
        "source",
        "group_id",
        "patient_id",
        "exam_id",
        "identity_level",
    }
    return [c for c in df.columns if c not in meta]


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    path = cfg["paths"]["features_dir"] / "biomarker_features.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    df = pd.read_csv(path)
    feats = _feat_cols(df)
    geom = [c for c in GEOM_COLUMNS if c in df.columns]
    fractal = [c for c in FRACTAL_COLS if c in df.columns]
    density = [c for c in feats if c.startswith(DENSITY_PREFIX) or c in {"vessel_density", "vessel_pixels"}]

    def nan_rate(cols: list[str], sub: pd.DataFrame) -> float:
        if not cols:
            return float("nan")
        return float(sub[cols].isna().any(axis=1).mean())

    def all_nan_geom(sub: pd.DataFrame) -> float:
        if not geom:
            return float("nan")
        return float(sub[geom].isna().all(axis=1).mean())

    rows = []
    overall = {
        "n_rows": int(len(df)),
        "n_features": len(feats),
        "any_nan_row_frac": float(df[feats].isna().any(axis=1).mean()),
        "geom_any_nan_frac": nan_rate(geom, df),
        "geom_all_nan_frac": all_nan_geom(df),
        "fractal_any_nan_frac": nan_rate(fractal, df),
        "density_any_nan_frac": nan_rate(density, df),
        "median_nan_frac_per_col": {
            c: float(df[c].isna().mean()) for c in feats if df[c].isna().any()
        },
    }
    rows.append({"slice": "ALL", **{k: overall[k] for k in (
        "n_rows", "geom_any_nan_frac", "geom_all_nan_frac", "fractal_any_nan_frac", "any_nan_row_frac"
    )}})

    by_source = {}
    for src, sub in df.groupby("source"):
        s = {
            "n": int(len(sub)),
            "geom_any_nan_frac": nan_rate(geom, sub),
            "geom_all_nan_frac": all_nan_geom(sub),
            "fractal_any_nan_frac": nan_rate(fractal, sub),
            "any_nan_row_frac": float(sub[feats].isna().any(axis=1).mean()),
            "mean_vessel_density": float(sub["vessel_density"].mean()) if "vessel_density" in sub else None,
        }
        by_source[str(src)] = s
        rows.append({
            "slice": f"source={src}",
            "n_rows": s["n"],
            "geom_any_nan_frac": s["geom_any_nan_frac"],
            "geom_all_nan_frac": s["geom_all_nan_frac"],
            "fractal_any_nan_frac": s["fractal_any_nan_frac"],
            "any_nan_row_frac": s["any_nan_row_frac"],
        })

    class_names = cfg["data"]["class_names"]
    by_class = {}
    for lab, sub in df.groupby("label"):
        name = class_names[int(lab)] if int(lab) < len(class_names) else str(lab)
        s = {
            "n": int(len(sub)),
            "geom_any_nan_frac": nan_rate(geom, sub),
            "geom_all_nan_frac": all_nan_geom(sub),
        }
        by_class[name] = s
        rows.append({
            "slice": f"class={name}",
            "n_rows": s["n"],
            "geom_any_nan_frac": s["geom_any_nan_frac"],
            "geom_all_nan_frac": s["geom_all_nan_frac"],
            "fractal_any_nan_frac": nan_rate(fractal, sub),
            "any_nan_row_frac": float(sub[feats].isna().any(axis=1).mean()),
        })

    # Cross: source x class geom failure
    cross = []
    for src, sub_s in df.groupby("source"):
        for lab, sub in sub_s.groupby("label"):
            name = class_names[int(lab)] if int(lab) < len(class_names) else str(lab)
            cross.append({
                "source": str(src),
                "class": name,
                "n": int(len(sub)),
                "geom_all_nan_frac": all_nan_geom(sub),
                "geom_any_nan_frac": nan_rate(geom, sub),
            })

    # Correlation: does geom failure associate with Plus?
    geom_fail = df[geom].isna().all(axis=1) if geom else pd.Series(False, index=df.index)
    plus_idx = class_names.index("Plus") if "Plus" in class_names else 2
    fail_plus = float(((df["label"] == plus_idx) & geom_fail).sum()) / max(1, (df["label"] == plus_idx).sum())
    fail_nonplus = float(((df["label"] != plus_idx) & geom_fail).sum()) / max(1, (df["label"] != plus_idx).sum())

    # Recommendation
    fail_rate = overall["geom_all_nan_frac"]
    if fail_rate < 0.05:
        recommendation = "KEEP_AS_IS"
        reason = f"geom total failure {fail_rate:.1%} < 5%; median impute already handles sparse NaNs."
    elif fail_rate < 0.25:
        recommendation = "NOTE_IN_THESIS"
        reason = (
            f"geom total failure {fail_rate:.1%}; usable with impute. "
            "Optional: raise recursion limit / re-extract failed rows only."
        )
    else:
        recommendation = "REEXTRACT_OR_DROP_GEOM"
        reason = f"geom total failure {fail_rate:.1%} too high; Branch A may be density-driven."

    # Farabi-specific check (weak Branch A there)
    farabi_fail = by_source.get("farabi", {}).get("geom_all_nan_frac", float("nan"))
    if not np.isnan(farabi_fail) and farabi_fail > 0.2:
        recommendation = "INVESTIGATE_FARABI"
        reason += f" Farabi geom_all_nan={farabi_fail:.1%} likely hurts Branch A on that source."

    report = {
        "overall": overall,
        "by_source": by_source,
        "by_class": by_class,
        "cross_source_class": cross,
        "geom_fail_rate_among_plus": fail_plus,
        "geom_fail_rate_among_nonplus": fail_nonplus,
        "recommendation": recommendation,
        "reason": reason,
    }

    res = cfg["paths"]["results_dir"]
    with open(res / "pvbm_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(rows).to_csv(res / "pvbm_audit_summary.csv", index=False)
    pd.DataFrame(cross).to_csv(res / "pvbm_audit_cross.csv", index=False)

    print(json.dumps({
        "recommendation": recommendation,
        "geom_all_nan_frac": overall["geom_all_nan_frac"],
        "by_source_geom_all_nan": {k: v["geom_all_nan_frac"] for k, v in by_source.items()},
        "reason": reason,
    }, indent=2))
    print(f"[done] -> {res / 'pvbm_audit.json'}")


if __name__ == "__main__":
    main()
