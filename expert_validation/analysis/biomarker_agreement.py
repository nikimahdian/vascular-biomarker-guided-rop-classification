"""Q-C — do automatic vascular MEASUREMENTS agree with expert-derived measurements?

The question the thesis actually needs, and the one most often skipped. Good Dice with a poor
measurement ICC is a real, reportable outcome; the mask resolution ceiling makes it a live
possibility here.

Primary endpoints, pre-specified (everything else is secondary and labelled as such):

  1. vessel density
  2. calibre      -- width_p90 in disc diameters
  3. tortuosity   -- tort_p90

For each, against both comparison variants:
  m1 vs ref   segmentation-induced measurement error
  m2 vs ref   whole-pipeline measurement error

Reported per endpoint: ICC(2,1) with a 95 % CI resampled by group, Bland-Altman bias and limits of
agreement, and a paired cluster-bootstrap CI on the ICC itself. Correlation alone is not agreement:
widths 10/20/30 against 15/25/35 correlate almost perfectly and are biased by +5.

Usage
-----
  python biomarker_agreement.py --measurements out/biomarkers/biomarker_measurements.csv --out out/agreement
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import PRIMARY_BIOMARKERS, bland_altman, icc21, icc21_ci

SECONDARY = ["width_p50_dd", "width_p95_dd", "width_mean_dd", "width_p90_px",
             "skel_density", "n_skel_px", "n_branches", "tort_median", "tort_top3_mean",
             "av_width_ratio_p90", "a_width_p90_dd", "v_width_p90_dd", "a_frac"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--graders", default="A")
    args = ap.parse_args()

    m = pd.read_csv(args.measurements)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"rows={len(m)}  variants={sorted(m.variant.unique())}")

    ref = m[m.variant == "ref"].set_index(["study_id", "grader"])
    rows = []

    for grader in [g.strip() for g in args.graders.split(",") if g.strip()]:
        for variant in ("m1", "m2"):
            cmp_df = m[(m.variant == variant) & (m.grader == grader)].set_index(["study_id", "grader"])
            common = ref.index.intersection(cmp_df.index)
            if len(common) < 5:
                print(f"[skip] grader {grader} / {variant}: only {len(common)} paired images")
                continue
            a = ref.loc[common].reset_index()
            b = cmp_df.loc[common].reset_index()
            j = a.merge(b, on=["study_id", "grader"], suffixes=("_ref", "_cmp"))
            print(f"\n=== grader {grader}  {variant} vs ref   n={len(j)}  "
                  f"groups={j.group_id_ref.nunique()} ===")

            for col in PRIMARY_BIOMARKERS + SECONDARY:
                cr, cc = f"{col}_ref", f"{col}_cmp"
                if cr not in j.columns or cc not in j.columns:
                    continue
                x, y = j[cr].to_numpy(float), j[cc].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 5:
                    continue
                ic = icc21(x, y)
                ba = bland_altman(y, x)                     # y = automatic, x = expert
                sub = j[ok].rename(columns={"group_id_ref": "group_id"})
                r = icc21_ci(sub, cr, cc, n_boot=args.n_boot)
                rows.append({
                    "grader": grader, "comparison": f"{variant}_vs_ref",
                    "biomarker": col,
                    "role": "primary" if col in PRIMARY_BIOMARKERS else "secondary",
                    "n": int(ok.sum()),
                    "n_groups": int(sub["group_id"].nunique()),
                    "icc": ic["icc"],
                    "icc_ci_low": r["ci_low"], "icc_ci_high": r["ci_high"],
                    "icc_n_boot": r["n_boot"],
                    "bias": ba["bias"], "bias_ci_low": ba["bias_ci_low"],
                    "bias_ci_high": ba["bias_ci_high"],
                    "loa_low": ba["loa_low"], "loa_high": ba["loa_high"],
                    "expert_mean": float(np.nanmean(x[ok])),
                    "auto_mean": float(np.nanmean(y[ok])),
                    "expert_sd": float(np.nanstd(x[ok], ddof=1)),
                    "auto_sd": float(np.nanstd(y[ok], ddof=1)),
                })
                print(f"  {col:22s} n={ok.sum():4d}  ICC={ic['icc']:.3f} "
                      f"[{r['ci_low']:.3f},{r['ci_high']:.3f}]  bias={ba['bias']:+.4f} "
                      f"LoA=[{ba['loa_low']:+.4f},{ba['loa_high']:+.4f}]"
                      f"{'' if col in PRIMARY_BIOMARKERS else '   (secondary)'}")

    if not rows:
        raise SystemExit("no paired measurements — check the variant and grader arguments")
    df = pd.DataFrame(rows)
    df.to_csv(out / "biomarker_agreement.csv", index=False)

    prim = df[df.role == "primary"]
    if len(prim):
        print("\n=== PRIMARY endpoints ===")
        print(prim[["comparison", "biomarker", "n", "icc", "icc_ci_low", "icc_ci_high",
                    "bias", "loa_low", "loa_high"]].round(4).to_string(index=False))
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
