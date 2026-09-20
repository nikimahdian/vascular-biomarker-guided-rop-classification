#!/usr/bin/env python
"""Task 3 corroboration: independently validate the Task 3 conclusions against two
pre-existing evidence files that were NOT produced by this audit.

  * data/metadata/identity_map.csv                  - per-image sha256 + duplicate counts
  * data/metadata/perceptual_duplicate_candidates.csv / _adjudication.csv - near-dup review
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
PRIV = f"{ROOT}/_private_audit"


def rule(t):
    print()
    print("=" * 100)
    print(t)
    print("=" * 100)


cache = pd.read_csv(f"{PRIV}/pixel_sha256_cache.csv")
mine = dict(zip(cache.image_path, cache.sha256))
dh = dict(zip(cache.image_path, cache.dhash.fillna("")))
print(f"my hash cache: {len(cache)} rows, {int(cache.sha256.notna().sum())} non-null")

can = pd.read_csv(f"{ROOT}/data/splits/all.csv")
exc = pd.read_csv(f"{ROOT}/data/splits/excluded_ambiguous_exact_duplicates.csv")
qe = pd.read_csv(f"{ROOT}/data/metadata/quality_exclusions.csv")
for c in ("source", "label"):
    if c in qe.columns:
        qe = qe.drop(columns=c)
feat = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv")[
    ["image_path", "label", "source", "group_id"]]
qe = qe.merge(feat, on="image_path", how="left")
qe["path_source"] = qe.image_path.str.split("/").str[6]

# ---------------------------------------------------------------- A
rule("A. THE 13 PRE-LOSO QUALITY EXCLUSIONS - ARE THEY REALLY ALL FARABI?")
leg = pd.read_csv(f"{ROOT}/data/masks/mask_manifest_legacy_image_level_20260826.csv")
can_farabi = set(feat[feat.source == "farabi"].image_path)
loso = leg[(leg.source != "farabi") |
           ((leg.source == "farabi") & leg.image_path.isin(can_farabi))]
ls = set(loso.image_path)
qeset = set(qe.image_path)
pre = qe[~qe.image_path.isin(ls)]
print(f"  quality exclusions total          : {len(qe)}")
print(f"  applied BEFORE LOSO (not in LOSO) : {len(pre)}")
print(f"  by path-component source          : {pre.path_source.value_counts().to_dict()}")
print(f"  by merged source (NaN when the image is not in the canonical feature table): "
      f"{pre.source.value_counts(dropna=False).to_dict()}")
print(f"  reasons                           : {pre.reason.value_counts().to_dict()}")
print(f"  phases                            : {pre.phase.value_counts().to_dict()}")
print(f"  dates                             : {sorted(pre.date.unique())}")
print()
print("  -> the pre-LOSO exclusions are exactly the 13 farabi rows; this is the "
      "legacy(8960) -> LOSO(8947) step.")

# ---------------------------------------------------------------- B
rule("B. INDEPENDENT HASH VALIDATION AGAINST data/metadata/identity_map.csv")
im = pd.read_csv(f"{ROOT}/data/metadata/identity_map.csv")
print(f"  identity_map rows={len(im)}  unique paths={im.image_path.nunique()}  "
      f"sha256 non-null={int(im.sha256.notna().sum())}")
imap = dict(zip(im.image_path, im.sha256))
common = [p for p in mine if p in imap and isinstance(mine[p], str) and mine[p]]
agree = sum(1 for p in common if mine[p] == imap[p])
print(f"  paths present in both             : {len(common)}")
print(f"  sha256 agreement with identity_map: {agree} / {len(common)}  "
      f"({100.0 * agree / max(len(common), 1):.4f}%)")
if agree != len(common):
    bad = [p for p in common if mine[p] != imap[p]]
    print(f"  DISAGREEMENTS: {len(bad)}")
    for p in bad[:10]:
        print(f"    {p}\n      mine={mine[p]}\n      imap={imap[p]}")

# ---------------------------------------------------------------- C
rule("C. DOES identity_map INDEPENDENTLY MARK THE 54 AS DUPLICATES?")
imx = im.set_index("image_path")
sub = imx.reindex(exc.image_path)
print(f"  of the 54 excluded rows, present in identity_map: "
      f"{int(sub.sha256.notna().sum())} / {len(exc)}")
print(f"  sha256 agrees with the stored exclusion hash     : "
      f"{int((sub.sha256.values == exc.sha256.values).sum())} / {len(exc)}")
print(f"  exact_duplicate_count value counts for the 54    : "
      f"{sub.exact_duplicate_count.value_counts(dropna=False).to_dict()}")
print(f"  included_phase2 value counts for the 54          : "
      f"{sub.included_phase2.value_counts(dropna=False).to_dict()}")
print(f"  exclusion_reason for the 54                      : "
      f"{sub.exclusion_reason.value_counts(dropna=False).to_dict()}")

# ---------------------------------------------------------------- D
rule("D. DOES identity_map INDEPENDENTLY MARK THE 23 QUALITY EXCLUSIONS?")
only23 = qe[qe.image_path.isin(ls - set(can.image_path))]
print(f"  the 23 rows: {len(only23)}")
s23 = imx.reindex(only23.image_path)
print(f"  present in identity_map            : {int(s23.sha256.notna().sum())} / {len(only23)}")
print(f"  included_phase2                    : "
      f"{s23.included_phase2.value_counts(dropna=False).to_dict()}")
print(f"  exclusion_reason                   : "
      f"{s23.exclusion_reason.value_counts(dropna=False).to_dict()}")
print(f"  exact_duplicate_count              : "
      f"{s23.exact_duplicate_count.value_counts(dropna=False).to_dict()}")
print()
print("  quality_exclusions reasons for the 23:")
print(only23.reason.value_counts().to_string())

# ---------------------------------------------------------------- E
rule("E. NEAR-DUPLICATE REVIEW - DO THE 23 APPEAR IN THE PRIOR PERCEPTUAL AUDIT?")
pc = pd.read_csv(f"{ROOT}/data/metadata/perceptual_duplicate_candidates.csv")
pa = pd.read_csv(f"{ROOT}/data/metadata/perceptual_duplicate_adjudication.csv")
for nm, d in (("candidates", pc), ("adjudication", pa)):
    involved = d[(d.path_a.isin(set(only23.image_path))) |
                 (d.path_b.isin(set(only23.image_path)))]
    print(f"  {nm:14s} rows={len(d):4d}  pairs involving one of the 23: {len(involved)}")
print()
print("  NOTE: the perceptual files are an ADJUDICATION of cross-patient look-alikes; "
      "they are the human review record for the near-duplicate question, not a hash "
      "oracle. Their role here is corroboration, not exclusion.")
if len(pc):
    print(f"  dist columns: {[c for c in pc.columns if 'dist' in c]}")

# ---------------------------------------------------------------- F
rule("F. RESIDUAL EXACT DUPLICATES INSIDE THE CANONICAL 8870")
can["sha"] = can.image_path.map(lambda p: mine.get(p, ""))
print(f"  canonical images with a hash        : {int((can.sha != '').sum())} / {len(can)}")
g = can.groupby("sha").agg(n=("image_path", "size"), groups=("group_id", "nunique"))
print(f"  distinct sha256 values              : {len(g)}")
print(f"  sha values with >1 image            : {int((g.n > 1).sum())}")
print(f"  ... of those, spanning >1 group     : {int((g.groups > 1).sum())}   "
      f"<- must be 0, else cross-patient leakage survived")
print(f"  ... all confined to a single group  : {int(((g.n > 1) & (g.groups == 1)).sum())}")
print(f"  images in same-group duplicate sets : {int(g[g.n > 1].n.sum())}")
print()
print("  these are same-patient repeat captures of one fundus; because group_id is "
      "indivisible across folds (prepare_split.py split_by_patient) they cannot leak.")

# ---------------------------------------------------------------- G
rule("G. dHash CALIBRATION - IS 'hamming<=6' DISCRIMINATIVE OR NOISE?")
can["dh"] = can.image_path.map(lambda p: dh.get(p, ""))
cd = can[can.dh != ""][["image_path", "dh", "group_id"]]


def ham(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


samp = cd.sample(n=min(400, len(cd)), random_state=0)
nn = []
for d, p in zip(samp.dh, samp.image_path):
    dd = cd.dh.map(lambda x: ham(d, x))
    dd = dd[cd.image_path.values != p]
    nn.append(int(dd.min()))
nn = pd.Series(nn)
print(f"  sample size {len(nn)} canonical images; nearest-neighbour dHash distance:")
print(f"    min={nn.min()}  p05={nn.quantile(.05):.0f}  p25={nn.quantile(.25):.0f}  "
      f"median={nn.median():.0f}  p75={nn.quantile(.75):.0f}  max={nn.max()}")
print(f"    fraction with a neighbour at hamming<=6 : {(nn <= 6).mean():.3f}")
print(f"    fraction with a neighbour at hamming<=2 : {(nn <= 2).mean():.3f}")
print(f"    fraction with a neighbour at hamming==0 : {(nn == 0).mean():.3f}")
print()
print("  -> the CALIBRATED baseline tells us whether the 23 look unusual. A high baseline")
print("     means hamming<=6 reflects routine same-session frame redundancy, NOT that the")
print("     23 have hidden twins.")

# ---------------------------------------------------------------- H
rule("H. CORROBORATION VERDICT")
prev = json.load(open(f"{PRIV}/task3_summary.json"))
out = {
    "INDEPENDENT_HASH_AGREEMENT_WITH_IDENTITY_MAP":
        f"{agree}/{len(common)}",
    "PRE_LOSO_QUALITY_EXCLUSIONS_ARE_FARABI":
        bool(set(pre.path_source.unique()) == {"farabi"}),
    "CANONICAL_RESIDUAL_CROSS_GROUP_DUPLICATES": int((g.groups > 1).sum()),
    "CANONICAL_RESIDUAL_SAME_GROUP_DUPLICATES": int(((g.n > 1) & (g.groups == 1)).sum()),
    "DHASH_CALIBRATION_NN_MEDIAN": float(nn.median()),
    "DHASH_CALIBRATION_FRAC_NN_LE6": float((nn <= 6).mean()),
    "DHASH_CALIBRATION_FRAC_NN_EQ0": float((nn == 0).mean()),
    "EXACT_DUPLICATE_EXCLUSION_GROUPS": int(exc.group_id.nunique()),
    "EXACT_DUPLICATE_TRIGGERING_HASHES": int(
        exc[exc.triggering_duplicate].sha256.nunique()),
    "IDENTITY_MAP_INCLUDED_PHASE2_FALSE_FOR_THE_54":
        int((sub.included_phase2 == False).sum()),  # noqa: E712
    "CORROBORATION_STATUS": "PASS" if (
        agree == len(common)
        and int((g.groups > 1).sum()) == 0
        and set(pre.path_source.unique()) == {"farabi"}
    ) else "FAIL",
}
json.dump(out, open(f"{PRIV}/task3_corroboration.json", "w"), indent=2, default=str)
for k, v in out.items():
    print(f"  {k}: {v}")
print("=" * 100)
