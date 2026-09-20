#!/usr/bin/env python
"""Verify the fixed disc rule: disc-dependent features NaN when disc_valid == 0, disc-independent
features still present."""
import sys

import pandas as pd

d = pd.read_csv(sys.argv[1])
print("columns:", len(d.columns))
print("disc_method:", d.disc_method.value_counts().to_dict())
inv = d[d.disc_valid == 0]
val = d[d.disc_valid == 1]
print(f"disc_valid=0: {len(inv)}   disc_valid=1: {len(val)}")

DEP = [c for c in d.columns
       if c.endswith("_dd") or "ring" in c or "quad" in c
       or c in ("dd_px_work", "dd_over_min_side", "disc_cx_frac", "disc_cy_frac",
                "disc_centre_offset_frac", "n_skel_ann_px")]
print(f"\n{len(DEP)} disc-dependent columns")
print(f"  non-null cells among disc_valid==0 rows: {int(inv[DEP].notna().sum().sum())} "
      f"(must be 0)")
print(f"  non-null cells among disc_valid==1 rows: {int(val[DEP].notna().sum().sum())}")

INDEP = ["vessel_density", "skel_density", "n_skel_px", "n_branches",
         "width_p50_px", "width_p90_px", "width_mean_px", "width_shape_p90_over_p50",
         "tort_median", "tort_p90", "tort_top3_mean", "a_frac",
         "a_width_p90_px", "v_width_p90_px", "av_width_ratio_p90"]
INDEP = [c for c in INDEP if c in d.columns]
print(f"\n{len(INDEP)} disc-independent columns, non-null count among disc_valid==0 rows:")
print(inv[INDEP].notna().sum().to_string())
print("\nsample disc_valid==0 row (disc-independent values):")
print(inv[INDEP].head(3).to_string(index=False))
print("\nall disc-dependent columns:")
print(DEP)
