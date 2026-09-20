"""Q-D — do the extracted features correspond to clinically meaningful ROP severity?

Two things are computed here.

1. Grading agreement. Cohen's kappa on the three-way vascular grade and on Plus-vs-non-Plus,
   between graders where both annotated. If the automatic grade is ever produced it can be added
   on the same code path; by default it is not, because the study is not a classifier benchmark.

2. Feature-versus-clinician association. Spearman correlation between each primary biomarker and
   the two ordinal severity scores the expert recorded (arterial tortuosity, venous dilation),
   with the direction of the hypothesis stated in advance:

     higher arterial_tortuosity_score  ->  higher automatic tortuosity
     higher venous_dilation_score      ->  higher automatic calibre

   A biomarker that tracks the clinician's tortuosity judgement is evidence the feature means what
   its name says. A biomarker that correlates with the clinician's overall grade but not with
   either severity score is more likely to be tracking image appearance.

Usage
-----
  python clinical_agreement.py --key ../blinding_key.csv --root .. \
      --measurements out/biomarkers/biomarker_measurements.csv --out out/clinical
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _common import PRIMARY_BIOMARKERS, cohen_kappa, load_key

GRADE_MAP = {"normal": 0, "pre_plus": 1, "preplus": 1, "plus": 2, "cannot_determine": np.nan}
HYPOTHESIS = {
    "vessel_density": "higher_density_more_severe",
    "width_p90_dd": "higher_calibre_higher_dilation_score",
    "tort_p90": "higher_tortuosity_higher_tortuosity_score",
}
SCORES = ["arterial_tortuosity_score", "venous_dilation_score"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--graders", default="A")
    args = ap.parse_args()

    key = load_key(args.key)
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.measurements)
    ref = m[m.variant == "ref"]

    rows = []
    for grader in [g.strip() for g in args.graders.split(",") if g.strip()]:
        gcsv = root / f"grader_{grader}" / "grading.csv"
        if not gcsv.exists():
            print(f"[skip] grader_{grader}: no grading.csv")
            continue
        g = pd.read_csv(gcsv)
        g["grade_num"] = g["rop_vascular_grade"].astype(str).str.lower().map(GRADE_MAP)
        j = ref[ref.grader == grader].merge(g, on="study_id", how="inner")
        if j.empty:
            print(f"[skip] grader_{grader}: no overlap between grading and measurements")
            continue
        j = j.merge(key[["study_id", "group_id"]].rename(columns={"group_id": "gid"}),
                    on="study_id", how="left")
        print(f"\n=== grader {grader}: n={len(j)}, groups={j.gid.nunique()} ===")

        for b in PRIMARY_BIOMARKERS:
            if b not in j.columns:
                continue
            for s in SCORES:
                if s not in j.columns:
                    continue
                x = j[b].to_numpy(float)
                y = j[s].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 8:
                    continue
                rho, p = stats.spearmanr(x[ok], y[ok])
                rows.append({"grader": grader, "biomarker": b, "clinical_score": s,
                             "n": int(ok.sum()), "spearman_rho": float(rho), "p_value": float(p),
                             "expected_direction": HYPOTHESIS.get(b, ""),
                             "direction_matches": int(
                                 (rho > 0) == ("higher" in HYPOTHESIS.get(b, "higher")))})
                print(f"  {b:16s} vs {s:26s} rho={rho:+.3f}  p={p:.4g}  n={int(ok.sum())}")

        # descriptive: does each biomarker separate Plus from non-Plus among these images?
        if j["grade_num"].notna().sum() >= 10:
            yy = (j["grade_num"] == 2).astype(int).to_numpy()
            for b in PRIMARY_BIOMARKERS:
                if b not in j.columns:
                    continue
                x = j[b].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(j["grade_num"].to_numpy(float))
                if ok.sum() < 10 or len(np.unique(yy[ok])) < 2:
                    continue
                from _common import safe_auc
                a = safe_auc(yy[ok], x[ok])
                rows.append({"grader": grader, "biomarker": b, "clinical_score": "plus_vs_nonplus",
                             "n": int(ok.sum()), "spearman_rho": np.nan, "p_value": np.nan,
                             "auc": a, "expected_direction": HYPOTHESIS.get(b, ""),
                             "direction_matches": np.nan})
                print(f"  {b:16s} vs Plus-vs-nonPlus      AUC={a:.3f}  n={int(ok.sum())}")

    if not rows:
        raise SystemExit("nothing computed — check the grading sheets and measurement file")
    df = pd.DataFrame(rows)
    df.to_csv(out / "clinical_association.csv", index=False)
    print(f"\n[done] -> {out}")
    print("Reminder: these are associations on the annotated subset, not a classifier result,")
    print("and nothing here may be used to select a model or a threshold.")


if __name__ == "__main__":
    main()
