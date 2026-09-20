"""Q-B — is the automatic optic-disc geometry usable as a scale reference?

The downstream pipeline consumes only the disc CENTRE and RADIUS, so the automatic disc is
reconstructed here as a filled circle of the predicted radius at the predicted centre. Where the
saved disc inference is available the stored geometry is used directly.

Metrics: disc Dice, centre error in expert disc diameters, and the predicted/expert diameter ratio.
Dice is reported for completeness, but the quantity that matters is the diameter ratio, because a
radius error propagates one-for-one into every disc-normalised measurement.

Usage
-----
  python disc_agreement.py --key ../blinding_key.csv --root .. --disc-pred results/.../disc_predictions_all.csv --out out/disc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import cluster_bootstrap_ci, disc_geometry, disc_metrics, load_key, read_mask

METRICS = ["disc_dice", "disc_iou", "centre_error_px", "centre_error_dd", "diameter_ratio"]


def circle(shape: tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--disc-pred", required=True,
                    help="disc_predictions_all.csv with disc_cx, disc_cy, disc_dd_px, peak_prob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--graders", default="A,B")
    ap.add_argument("--peak-min", type=float, default=0.9,
                    help="the locked disc-validity confidence floor; rows below it are reported "
                         "as a separate subgroup, not pooled, because that floor is a convention "
                         "this study is meant to recalibrate")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    key = load_key(args.key)
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dp = pd.read_csv(args.disc_pred).set_index("image_path")
    rows = []

    for grader in [g.strip() for g in args.graders.split(",") if g.strip()]:
        ddir = root / f"grader_{grader}" / "disc_masks"
        if not ddir.exists():
            print(f"[skip] grader_{grader}: no disc_masks/")
            continue
        for _, r in key.iterrows():
            p = ddir / f"{r.study_id}.png"
            if not p.exists():
                continue
            gt = read_mask(p)
            if r.image_path not in dp.index:
                print(f"[warn] {r.study_id}: no automatic disc prediction")
                continue
            d = dp.loc[r.image_path]
            cx, cy, dd = d.get("disc_cx"), d.get("disc_cy"), d.get("disc_dd_px")
            if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(dd) and dd > 2):
                rows.append({"study_id": r.study_id, "grader": grader, "cohort": r.cohort,
                             "source": r.source, "min_side": r.min_side, "group_id": r.group_id,
                             "peak_prob": float(d.get("peak_prob", np.nan)),
                             "auto_disc_available": 0, **{m: np.nan for m in METRICS}})
                continue
            ai = circle(gt.shape, float(cx), float(cy), float(dd) / 2.0)
            rows.append({"study_id": r.study_id, "grader": grader, "cohort": r.cohort,
                         "source": r.source, "min_side": r.min_side, "group_id": r.group_id,
                         "peak_prob": float(d.get("peak_prob", np.nan)),
                         "auto_disc_available": 1,
                         "expert_dd_px": 2 * disc_geometry(gt)["radius_px"],
                         **disc_metrics(ai, gt)})

    if not rows:
        raise SystemExit("no expert disc masks found — run validate_annotations.py first")
    per = pd.DataFrame(rows)
    per["above_floor"] = (per.peak_prob > args.peak_min).astype(int)
    per.to_csv(out / "disc_per_image.csv", index=False)
    print(f"per-image rows: {len(per)}  auto disc missing: {int((per.auto_disc_available == 0).sum())}")

    summary = []
    def agg(df, name, **extra):
        for m in METRICS:
            if m not in df.columns or df[m].notna().sum() < 5:
                continue
            r = cluster_bootstrap_ci(df.dropna(subset=[m]), lambda d, m=m: d[m].mean(),
                                     n_boot=args.n_boot)
            summary.append({"subset": name, "metric": m, "n": int(df[m].notna().sum()), **r, **extra})

    for grader, g in per.groupby("grader"):
        av = g[g.auto_disc_available == 1]
        agg(av, f"grader_{grader}_overall", grader=grader)
        for src, s in av.groupby("source"):
            agg(s, f"grader_{grader}_{src}", grader=grader, source=src)
        agg(av[av.above_floor == 1], f"grader_{grader}_above_confidence_floor",
            grader=grader, above_floor=1)
        agg(av[av.above_floor == 0], f"grader_{grader}_below_confidence_floor",
            grader=grader, above_floor=0)

    s = pd.DataFrame(summary)
    s.to_csv(out / "disc_summary.csv", index=False)
    print("\n=== disc agreement ===")
    print(s[s.metric.isin(["disc_dice", "centre_error_dd", "diameter_ratio"])]
          [["subset", "metric", "n", "point", "ci_low", "ci_high"]].round(4).to_string(index=False))

    # descriptive calibration of the 0.9 floor: where does expert agreement actually break down?
    cal = per[per.auto_disc_available == 1].copy()
    if len(cal) > 10:
        cal["bin"] = pd.cut(cal.peak_prob, [0, 0.3, 0.5, 0.7, 0.9, 1.01],
                            labels=["0-.3", ".3-.5", ".5-.7", ".7-.9", ".9-1"])
        t = cal.groupby("bin", observed=True).agg(n=("study_id", "size"),
                                                  disc_dice=("disc_dice", "mean"),
                                                  centre_error_dd=("centre_error_dd", "mean"),
                                                  diameter_ratio=("diameter_ratio", "mean")).round(4)
        t.to_csv(out / "disc_confidence_calibration.csv")
        print("\n=== descriptive only: agreement by automatic confidence band ===")
        print(t.to_string())
        print("  This is descriptive. Do NOT pick a new floor on the final cohort; anything")
        print("  operational must be developed on the pilot and frozen before the final run.")
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
