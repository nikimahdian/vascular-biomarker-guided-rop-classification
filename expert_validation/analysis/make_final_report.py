"""Assemble the thesis tables from the frozen analysis outputs.

Runs at whatever stage the analysis has reached: any missing input is reported as not yet
available rather than crashing, so this can be run after the pilot and again after the final
cohort.

Produces 7 tables (see expert_validation/PROTOCOL.md section 12) as markdown plus CSV.

Usage
-----
  python make_final_report.py --analysis out --out out/report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TABLES: list[tuple[str, str, list[str]]] = [
    ("table1_cohort.md", "Table 1 — cohort and image quality",
     ["study_id", "cohort", "source", "min_side", "label", "image_quality"]),
    ("table2_vessel_validity.md", "Table 2 — Q-A vessel segmentation validity",
     ["subset", "metric", "n", "point", "ci_low", "ci_high"]),
    ("table3_disc_validity.md", "Table 3 — Q-B optic-disc validity",
     ["subset", "metric", "n", "point", "ci_low", "ci_high"]),
    ("table4_measurement_validity.md", "Table 4 — Q-C measurement agreement",
     ["comparison", "biomarker", "role", "n", "icc", "icc_ci_low", "icc_ci_high",
      "bias", "loa_low", "loa_high"]),
    ("table5_human_ceiling.md", "Table 5 — expert-vs-expert ceiling",
     ["endpoint", "metric", "value", "ci_low", "ci_high", "n"]),
    ("table6_clinical_association.md", "Table 6 — Q-D feature versus clinician",
     ["biomarker", "clinical_score", "n", "spearman_rho", "p_value", "auc",
      "expected_direction", "direction_matches"]),
    ("table7_limitations.md", "Table 7 — what is and is not supported",
     ["biomarker", "role", "icc", "icc_ci_low", "icc_ci_high", "bias", "claim"]),
]


def _md(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"## {title}\n\n_not yet available_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(4)
    head = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = "\n".join("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
                     for row in d.itertuples(index=False))
    return f"## {title}\n\n{head}\n{sep}\n{body}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True, help="directory holding the analysis outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--grading", default=None, help="path to a grading.csv for Table 1")
    args = ap.parse_args()

    a = Path(args.analysis)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def rd(*cands):
        for c in cands:
            p = a / c
            if p.exists():
                return pd.read_csv(p)
        return pd.DataFrame()

    seg = rd("seg/segmentation_summary.csv", "segmentation_summary.csv")
    disc = rd("disc/disc_summary.csv", "disc_summary.csv")
    agree = rd("agreement/biomarker_agreement.csv", "biomarker_agreement.csv")
    inter = rd("intergrader/intergrader_summary.csv", "intergrader_summary.csv")
    clinical = rd("clinical/clinical_association.csv", "clinical_association.csv")

    # Table 1
    t1 = pd.DataFrame()
    if args.key and args.grading:
        k = pd.read_csv(args.key)
        g = pd.read_csv(args.grading)
        j = k.merge(g, on="study_id", how="inner")
        if not j.empty:
            j["gradable"] = (~j["image_quality"].astype(str).str.lower()
                             .eq("ungradable")).astype(int)
            t1 = (j.groupby(["cohort", "source", "min_side", "label"], dropna=False)
                  .agg(n=("study_id", "size"),
                       gradable=("gradable", "sum"),
                       vessel_masks=("study_id", "size"))
                  .reset_index())
    elif args.key:
        k = pd.read_csv(args.key)
        t1 = (k.groupby(["cohort", "source", "min_side", "label"], dropna=False)
              .size().reset_index(name="n"))

    # Table 7: translate the measurement agreement into a claim per biomarker
    t7 = pd.DataFrame()
    if not agree.empty:
        prim = agree[agree.role == "primary"].copy()
        b = prim[prim.comparison == "m2_vs_ref"] if "m2_vs_ref" in set(prim.comparison) else prim

        def claim(r):
            if not np.isfinite(r.get("icc", np.nan)):
                return "not estimable"
            lo = r.get("icc_ci_low", np.nan)
            if np.isfinite(lo) and lo > 0.75:
                return "supported — absolute agreement acceptable"
            if np.isfinite(lo) and lo > 0.5:
                return "conditional — usable with the reported calibration, stated as such"
            return "not supported — report as a measurement limitation"

        b["claim"] = b.apply(claim, axis=1)
        t7 = b[["biomarker", "role", "icc", "icc_ci_low", "icc_ci_high", "bias", "claim"]]

    frames = {
        "table1_cohort.md": t1.rename(columns={"min_side": "geometry_min_side"}),
        "table2_vessel_validity.md": seg,
        "table3_disc_validity.md": disc,
        "table4_measurement_validity.md": agree[["comparison", "biomarker", "role", "n", "icc",
                                                 "icc_ci_low", "icc_ci_high", "bias", "loa_low",
                                                 "loa_high"]] if not agree.empty else pd.DataFrame(),
        "table5_human_ceiling.md": inter,
        "table6_clinical_association.md": clinical,
        "table7_limitations.md": t7,
    }

    report = ["# Expert validation — frozen analysis report\n",
              f"Inputs read from `{a}`. Missing tables are reported as not yet available.\n"]
    for fname, title, _cols in TABLES:
        df = frames.get(fname, pd.DataFrame())
        report.append(_md(df, title))
        if not df.empty:
            df.to_csv(out / fname.replace(".md", ".csv"), index=False)

    (out / "FINAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[done] -> {out / 'FINAL_REPORT.md'}")
    for fname, title, _ in TABLES:
        df = frames.get(fname)
        n = 0 if df is None or len(df) == 0 else len(df)
        print(f"  {'OK ' if n else '-- '} {title}  ({n} rows)")


if __name__ == "__main__":
    main()
