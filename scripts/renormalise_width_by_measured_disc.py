#!/usr/bin/env python
"""Re-normalise the EXISTING clinical_v2 width columns by the real disc, and add scale-free ratios.

Why this is possible without re-running feature extraction: v2 stored width_normalised =
2*EDT/dd_v2 with dd_v2 = max(8, min(h,w)//10) exactly (verified: dd_over_min_side == 0.1 for all
8870 rows). So the physical width in pixels is recoverable as width_px = width_dd * dd_v2, and the
correct normalisation is width_px / DD_px with DD from the disc detector.

Adds:
  width_*_dd_true   the v2 width columns re-normalised by the measured disc diameter
                    (NaN unless the detector fired confidently, peak_prob > 0.9)
  disc_peak_prob, dd_ok, dd_px     measurability flags so missingness is explicit, not silent
  w90_sqrtarea, a90_sqrtarea, v90_sqrtarea, w50_sqrtarea
                    width / sqrt(vessel area), which is scale-free by construction (both the
                    numerator and sqrt(area) scale linearly with magnification) and therefore
                    needs no disc at all -- available on every row

Writes data/features/biomarker_features_clinical_v2b.csv. Overwrites nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/moniaz/niki")

ROOT = Path("/Users/moniaz/niki")
V2 = ROOT / "data/features/biomarker_features_clinical_v2.csv"
DISC = ROOT / "results/hvdro_validation/disc/disc_predictions_all.csv"
OUT = ROOT / "data/features/biomarker_features_clinical_v2b.csv"
PEAK_OK = 0.9
DD_LO, DD_HI = 0.03, 0.25

v2 = pd.read_csv(V2)
disc = pd.read_csv(DISC)
print("v2:", v2.shape, " disc:", disc.shape)

d = disc.set_index("image_path")
miss = v2[~v2["image_path"].isin(d.index)]
print("v2 rows with no disc prediction:", len(miss))

dd = v2["image_path"].map(d["disc_dd_px"])
pk = v2["image_path"].map(d["peak_prob"])
ok = v2["image_path"].map(d["ok"]).fillna(False).astype(bool)
iw = v2["image_path"].map(d["w"])
ih = v2["image_path"].map(d["h"])

# v2's own DD, reconstructed from the image size (integer division, as in extract_clinical_v2)
dd_v2 = v2["image_path"].map(d["w"]).combine(
    v2["image_path"].map(d["h"]), lambda a, b: float(max(8, int(min(a, b)) // 10)))
print("\nreconstructed dd_v2 vs the stored dd_over_min_side:")
chk = (dd_v2 / v2["image_path"].map(d["w"]).combine(
    v2["image_path"].map(d["h"]), lambda a, b: float(min(a, b))))
print(f"  max |ratio - 0.1| = {np.abs(chk - 0.1).max():.2e}  (should be ~0 except tiny images)")

good = ok & (pk > PEAK_OK) & dd.between(DD_LO * np.minimum(iw, ih), DD_HI * np.minimum(iw, ih))
print(f"\nusable DD: {int(good.sum())} / {len(v2)} = {good.mean() * 100:.1f}%")
print(pd.crosstab(v2["source"], good, normalize="index").round(3).to_string())

out = v2.copy()
out["disc_peak_prob"] = pk
out["dd_ok"] = good.astype(int)
out["dd_px"] = np.where(good, dd, np.nan)
out["dd_over_min_side_true"] = out["dd_px"] / np.minimum(iw, ih)

for col in ("width_p50_dd", "width_p90_dd", "width_p95_dd", "width_mean_dd",
            "a_width_p90_dd", "v_width_p90_dd"):
    if col not in v2.columns:
        continue
    width_px = v2[col] * dd_v2                      # recover the physical width
    new = width_px / out["dd_px"]                   # re-normalise by the real disc
    out[col.replace("_dd", "_dd_true")] = np.where(good, new, np.nan)

# scale-free: width / sqrt(vessel area in px). Both scale linearly with magnification.
area_px = v2["vessel_density"] * iw * ih
s = np.sqrt(area_px)
for src, dst in (("width_p50_dd", "w50_sqrtarea"), ("width_p90_dd", "w90_sqrtarea"),
                 ("a_width_p90_dd", "a90_sqrtarea"), ("v_width_p90_dd", "v90_sqrtarea")):
    if src in v2.columns:
        out[dst] = v2[src] * dd_v2 / s

out.to_csv(OUT, index=False)
print(f"\n[done] {OUT} shape={out.shape}")

cols = ["width_p90_dd", "width_p90_dd_true", "w90_sqrtarea", "dd_over_min_side_true",
        "a_width_p90_dd", "v_width_p90_dd", "av_width_ratio_p90"]
print("\n=== by source (median) ===")
print(out.groupby("source")[cols].median().round(4).to_string())
print("\n=== spread across sources, normalised (max-min)/mean -> lower is less confounding ===")
med = out.groupby("source")[cols].median()
print(((med.max() - med.min()) / med.abs().mean()).round(3).to_string())
print("\n=== by label (median) ===")
print(out.groupby("label")[cols].median().round(4).to_string())
