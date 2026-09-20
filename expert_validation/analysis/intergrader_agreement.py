"""The human ceiling — expert A against expert B on the doubly annotated subset.

This is what replaces an arbitrary acceptance threshold. The interpretable statement is
"the automatic measurement differs from the expert by about as much as two experts differ from
each other", not "Dice exceeds 0.80". It also quantifies the irreducible disagreement in the
target itself, which is the number a reviewer will ask for.

Reported: vessel Dice and clDice, disc Dice and centre error, ICC on the two severity scores, and
Cohen's kappa (linear and quadratic weighted) on the three-way vascular grade.

Usage
-----
  python intergrader_agreement.py --key ../blinding_key.csv --root .. --out out/intergrader
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (cohen_kappa, dice, disc_geometry, disc_metrics, icc21_ci, load_key, read_mask,
                     segmentation_metrics)

GRADE_MAP = {"normal": 0, "pre_plus": 1, "preplus": 1, "plus": 2, "cannot_determine": np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--a", default="A")
    ap.add_argument("--b", default="B")
    args = ap.parse_args()

    key = load_key(args.key)
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ga = pd.read_csv(root / f"grader_{args.a}" / "grading.csv")
    gb = pd.read_csv(root / f"grader_{args.b}" / "grading.csv")
    both = sorted(set(ga.study_id) & set(gb.study_id))
    if not both:
        raise SystemExit(f"graders {args.a} and {args.b} share no study_id")
    print(f"doubly annotated images: {len(both)}")
    k = key.set_index("study_id")

    rows = []
    for sid in both:
        rec = {"study_id": sid, "cohort": k.loc[sid, "cohort"], "source": k.loc[sid, "source"],
               "min_side": int(k.loc[sid, "min_side"]), "group_id": k.loc[sid, "group_id"]}
        va = root / f"grader_{args.a}" / "vessel_masks" / f"{sid}.png"
        vb = root / f"grader_{args.b}" / "vessel_masks" / f"{sid}.png"
        if va.exists() and vb.exists():
            ma, mb = read_mask(va), read_mask(vb)
            if ma.shape == mb.shape:
                rec["vessel_dice"] = dice(ma, mb)
                rec["vessel_iou"] = segmentation_metrics(ma, mb)["iou"]
                rec["vessel_cldice"] = segmentation_metrics(ma, mb)["cldice"]
        da = root / f"grader_{args.a}" / "disc_masks" / f"{sid}.png"
        db = root / f"grader_{args.b}" / "disc_masks" / f"{sid}.png"
        if da.exists() and db.exists():
            ma, mb = read_mask(da), read_mask(db)
            if ma.shape == mb.shape and ma.sum() and mb.sum():
                rec |= {f"disc_{kk}": vv for kk, vv in disc_metrics(ma, mb).items()}
                rec["expert_dd_px"] = 2 * disc_geometry(mb)["radius_px"]
        rows.append(rec)

    per = pd.DataFrame(rows)
    per.to_csv(out / "intergrader_per_image.csv", index=False)

    print("\n=== mask agreement between graders ===")
    for c in ("vessel_dice", "vessel_cldice", "disc_dice", "centre_error_dd", "diameter_ratio"):
        if c in per.columns and per[c].notna().sum():
            v = per[c].dropna()
            print(f"  {c:18s} n={len(v):3d}  mean={v.mean():.4f}  median={v.median():.4f}  "
                  f"range=[{v.min():.4f},{v.max():.4f}]")

    summary = []
    a = ga.set_index("study_id").loc[both]
    b = gb.set_index("study_id").loc[both]
    paired = per[["study_id", "group_id"]].merge(
        a.reset_index()[["study_id"] + [c for c in ("arterial_tortuosity_score",
                                                    "venous_dilation_score", "grade_confidence")
                                        if c in a.columns]],
        on="study_id", how="left").merge(
        b.reset_index()[["study_id"] + [c for c in ("arterial_tortuosity_score",
                                                    "venous_dilation_score", "grade_confidence")
                                        if c in b.columns]],
        on="study_id", how="left", suffixes=("_a", "_b"))
    paired = paired.rename(columns={"group_id": "group_id"})
    for col in ("arterial_tortuosity_score", "venous_dilation_score", "grade_confidence"):
        ca, cb = f"{col}_a", f"{col}_b"
        if ca not in paired.columns or cb not in paired.columns:
            continue
        sub = paired[["study_id", "group_id", ca, cb]].dropna()
        if len(sub) < 5:
            continue
        r = icc21_ci(sub, ca, cb)
        summary.append({"endpoint": col, "metric": "icc21", "value": r["point"],
                        "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": int(len(sub))})
        print(f"  {col:26s} ICC(2,1)={r['point']:.3f} "
              f"[{r['ci_low']:.3f},{r['ci_high']:.3f}]  n={len(sub)}")

    if "rop_vascular_grade" in a.columns and "rop_vascular_grade" in b.columns:
        ca = a["rop_vascular_grade"].astype(str).str.lower().map(GRADE_MAP).to_numpy(float)
        cb = b["rop_vascular_grade"].astype(str).str.lower().map(GRADE_MAP).to_numpy(float)
        ok = np.isfinite(ca) & np.isfinite(cb)
        if ok.sum() >= 5:
            ka = cohen_kappa(ca[ok], cb[ok])
            kw = cohen_kappa(ca[ok], cb[ok], weights="quadratic")
            agree = float((ca[ok] == cb[ok]).mean())
            plus = cohen_kappa((ca[ok] == 2).astype(int), (cb[ok] == 2).astype(int))
            print(f"\n  three-class  kappa={ka:.3f}   quadratic-weighted kappa={kw:.3f}   "
                  f"exact agreement={agree:.3f}  n={int(ok.sum())}")
            print(f"  Plus vs non-Plus  kappa={plus:.3f}")
            summary += [
                {"endpoint": "vascular_grade_3class", "metric": "cohen_kappa", "value": ka,
                 "n": int(ok.sum())},
                {"endpoint": "vascular_grade_3class", "metric": "quadratic_weighted_kappa",
                 "value": kw, "n": int(ok.sum())},
                {"endpoint": "vascular_grade_3class", "metric": "exact_agreement",
                 "value": agree, "n": int(ok.sum())},
                {"endpoint": "plus_vs_nonplus", "metric": "cohen_kappa", "value": plus,
                 "n": int(ok.sum())},
            ]
    s = pd.DataFrame(summary)
    s.to_csv(out / "intergrader_summary.csv", index=False)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
