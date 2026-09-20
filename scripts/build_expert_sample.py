"""Build the actual expert-validation sample: 30 pilot + 90 final, stratified, one image per group.

Pool = val + test only, because those are the images the segmentation and disc models were never
trained on, and this is a measurement-validation study. Grouping unit is group_id (for farabi that
is the exam, not the patient -- patient identity is unavailable there; stated as a limitation).

Allocation is driven by the head-room that actually exists (see scripts/expert_headroom.py):
under one-image-per-group the val+test pool offers at most 31 Plus, 27 Pre-Plus and 81 Normal
groups, and geometry 480 contributes no Plus group at all.

Writes expert_validation/{manifest_blinded.csv,blinding_key.csv,sampling_summary.csv}.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_work")
OUT = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_repo_clone\expert_validation")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260120

d = pd.read_csv(B / "v2b.csv")[["image_path", "mask_path", "label", "source", "split",
                                "group_id", "vessel_density"]]
disc = pd.read_csv(B / "disc_predictions_all.csv")[["image_path", "w", "h", "peak_prob",
                                                    "dd_over_min_side"]]
d = d.merge(disc, on="image_path", how="left")
d["min_side"] = np.minimum(d.w, d.h).astype(int)
d = d[d.split.isin(["val", "test"])].copy()
print("honest pool (val+test):", len(d))

# ---------------------------------------------------------------- allocation
# Scarcest class first: Pre-Plus has the fewest available groups (27), then Plus (31), then
# Normal (81). Groups can carry more than one label (51 groups in the full dataset do), so
# allocating the abundant class first would silently consume the scarce one.
PLAN = {"final": {1: 16, 2: 24, 0: 50}, "pilot": {1: 6, 2: 7, 0: 17}}
GEO = {
    "final": {1: {960: 12, 1200: 4},
              2: {960: 19, 1200: 3, 1240: 2},
              0: {480: 16, 960: 6, 1080: 17, 1200: 7, 1240: 6}},
    "pilot": {1: {960: 4, 1200: 2},
              2: {960: 6, 1200: 1},
              0: {480: 3, 960: 2, 1080: 3, 1200: 2, 1240: 7}},
}

rng = np.random.default_rng(SEED)
picked: set[str] = set()
used_groups: set[str] = set()
rows = []

for cohort in ("pilot", "final"):
    for lab in (1, 2, 0):
        for geo, n_want in GEO[cohort][lab].items():
            cand = d[(d.label == lab) & (d.min_side == geo)
                     & (~d.group_id.isin(used_groups)) & (~d.image_path.isin(picked))]
            if cand.empty:
                print(f"  [warn] {cohort} label={lab} geo={geo}: no candidates")
                continue
            # one image per group: shuffle groups, take the first n
            grp = cand.groupby("group_id").head(1).sample(frac=1.0, random_state=SEED)
            take = grp.head(n_want)
            if len(take) < n_want:
                print(f"  [warn] {cohort} label={lab} geo={geo}: wanted {n_want}, got {len(take)}")
            for _, r in take.iterrows():
                picked.add(r.image_path)
                used_groups.add(r.group_id)
                rows.append(dict(image_path=r.image_path, mask_path=r.mask_path,
                                 cohort=cohort, label=int(lab), source=r.source,
                                 split=r.split, group_id=r.group_id, min_side=int(geo),
                                 w=int(r.w), h=int(r.h), peak_prob=float(r.peak_prob),
                                 vessel_density=float(r.vessel_density),
                                 dd_over_min_side=float(r.dd_over_min_side)))

sel = pd.DataFrame(rows)
print("\nselected:", len(sel), " unique images:", sel.image_path.nunique(),
      " unique groups:", sel.group_id.nunique())
print(pd.crosstab([sel.cohort, sel.label], sel.min_side).to_string())

# ---------------------------------------------------------------- challenge flags
# Geometry is a stratification axis, not a challenge flag. The one objective, protocol-relevant
# challenge condition is a disc detection the automatic pipeline cannot be trusted on, because
# that is exactly where the measurement layer's own validity is in question.
dens_lo = d.vessel_density.quantile(0.10)
dens_hi = d.vessel_density.quantile(0.90)
sel["flag_disc_low_confidence"] = (sel.peak_prob < 0.5).astype(int)
sel["flag_geometry_1240"] = (sel.min_side == 1240).astype(int)
sel["flag_geometry_480"] = (sel.min_side == 480).astype(int)
sel["flag_density_extreme"] = ((sel.vessel_density <= dens_lo) |
                               (sel.vessel_density >= dens_hi)).astype(int)
sel["flag_challenge"] = sel["flag_disc_low_confidence"]
print("\nflags inside the final cohort:")
fin = sel[sel.cohort == "final"]
print(fin[["flag_disc_low_confidence", "flag_geometry_1240", "flag_geometry_480",
           "flag_density_extreme"]].sum().to_string())
print(f"  primary challenge subset (disc not confidently detected): "
      f"{int(fin.flag_challenge.sum())} / {len(fin)}")

# ---------------------------------------------------------------- blinding
shuf = sel.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
shuf["study_id"] = [f"ROP_{i + 1:04d}" for i in range(len(shuf))]

manifest = shuf[["study_id", "cohort"]].copy()          # what the grader receives
key = shuf[["study_id", "cohort", "image_path", "mask_path", "label", "source", "split",
            "group_id", "min_side", "w", "h", "peak_prob", "vessel_density",
            "dd_over_min_side", "flag_challenge", "flag_disc_low_confidence",
            "flag_geometry_1240", "flag_geometry_480", "flag_density_extreme"]].copy()
manifest.to_csv(OUT / "manifest_blinded.csv", index=False)
key.to_csv(OUT / "blinding_key.csv", index=False)

summary = (sel.groupby(["cohort", "label", "min_side"])
           .agg(n=("image_path", "size"), groups=("group_id", "nunique"),
                sources=("source", lambda s: "|".join(sorted(set(s)))))
           .reset_index())
summary.to_csv(OUT / "sampling_summary.csv", index=False)
print("\n=== sampling summary ===")
print(summary.to_string(index=False))

sha = hashlib.sha256(key.to_csv(index=False).encode()).hexdigest()
(OUT / "MANIFEST_SHA256.txt").write_text(
    f"blinding_key.csv content sha256 = {sha}\n"
    f"seed = {SEED}\npool = val+test (segmentation and disc models never trained on these)\n"
    f"rule = at most one image per group_id\n"
    f"n = {len(sel)}   pilot = {(sel.cohort == 'pilot').sum()}   final = {(sel.cohort == 'final').sum()}\n")
print(f"\nmanifest sha256 = {sha}")
print(f"[done] -> {OUT}")
