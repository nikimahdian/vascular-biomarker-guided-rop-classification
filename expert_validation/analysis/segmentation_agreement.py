"""Q-A — does the automatic vessel mask reproduce clinically visible vasculature?

Per image, against the expert mask: Dice, IoU, precision, recall, clDice, and centreline recall
split into thin / medium / thick by the EXPERT mask's own local width.

Topology-aware and tiered metrics are not decoration. A mask can lose the entire thin network and
still score a respectable Dice, because Dice is dominated by the large vessels; the observed
cross-camera failure is a recall collapse with high precision, which is precisely that pattern.

Usage
-----
  python segmentation_agreement.py --key ../blinding_key.csv --root .. --out out/seg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import cluster_bootstrap_ci, load_key, read_mask, segmentation_metrics

METRICS = ["dice", "iou", "precision", "recall", "cldice",
           "recall_thin", "recall_medium", "recall_thick",
           "expert_vessel_fraction", "pred_vessel_fraction"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--graders", default="A,B")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    key = load_key(args.key)
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for grader in [g.strip() for g in args.graders.split(",") if g.strip()]:
        vdir = root / f"grader_{grader}" / "vessel_masks"
        if not vdir.exists():
            print(f"[skip] grader_{grader}: no vessel_masks/")
            continue
        for _, r in key.iterrows():
            p = vdir / f"{r.study_id}.png"
            if not p.exists():
                continue
            gt = read_mask(p)
            ai = read_mask(r.mask_path)
            if gt.shape != ai.shape:
                print(f"[warn] {r.study_id}: expert {gt.shape} vs auto {ai.shape} — skipped")
                continue
            rows.append({"study_id": r.study_id, "grader": grader, "cohort": r.cohort,
                         "source": r.source, "min_side": r.min_side, "label": r.label,
                         "group_id": r.group_id, "flag_challenge": r.flag_challenge,
                         **segmentation_metrics(ai, gt)})

    if not rows:
        raise SystemExit("no expert vessel masks found — run validate_annotations.py first")
    per = pd.DataFrame(rows)
    per.to_csv(out / "segmentation_per_image.csv", index=False)
    print(f"per-image rows: {len(per)}")

    summary = []
    def agg(df, name, **extra):
        for m in METRICS:
            if m not in df.columns or df[m].notna().sum() < 5:
                continue
            r = cluster_bootstrap_ci(df.dropna(subset=[m]), lambda d, m=m: d[m].mean(),
                                     n_boot=args.n_boot)
            summary.append({"subset": name, "metric": m, "n": int(df[m].notna().sum()), **r, **extra})

    for grader, g in per.groupby("grader"):
        agg(g, f"grader_{grader}_overall", grader=grader)
        for src, s in g.groupby("source"):
            agg(s, f"grader_{grader}_{src}", grader=grader, source=src)
        for geo, s in g.groupby("min_side"):
            agg(s, f"grader_{grader}_geo{int(geo)}", grader=grader, min_side=int(geo))
        ch = g[g.flag_challenge == 1]
        if len(ch) >= 5:
            agg(ch, f"grader_{grader}_disc_low_confidence", grader=grader, challenge=1)

    s = pd.DataFrame(summary)
    s.to_csv(out / "segmentation_summary.csv", index=False)
    print("\n=== overall, per grader ===")
    print(s[s.subset.str.contains("overall")][["subset", "metric", "n", "point", "ci_low", "ci_high"]]
          .round(4).to_string(index=False))
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
