#!/usr/bin/env python
"""Sections D, F, G, I: forensic table, downstream lineage, Branch A/C gate, canonical manifest."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
PRIV = "/Users/moniaz/niki/_private_audit"
os.makedirs(PRIV, exist_ok=True)
PAT = re.compile(r"^(?P<stem>.+)_(?P<h>[0-9a-f]{8})\.png$")


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


man = pd.read_csv(f"{ROOT}/data/masks/mask_manifest.csv")
feat = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv")
mdir = os.path.dirname(man.mask_path.iloc[0])

# stable ids so no patient UUID reaches a public artifact
uids = sorted({os.path.splitext(os.path.basename(p))[0].split(".")[0] for p in man.image_path})
uid2 = {u: f"U{i:05d}" for i, u in enumerate(uids)}
gids = sorted(man.group_id.unique()) if "group_id" in man.columns else []
gid2 = {g: f"G{i:05d}" for i, g in enumerate(gids)}


def stable(p):
    return uid2.get(os.path.splitext(os.path.basename(p))[0].split(".")[0], "UNK")


def geom(p):
    try:
        from PIL import Image
        with Image.open(p) as im:
            return f"{im.size[0]}x{im.size[1]}"
    except Exception:  # noqa: BLE001
        return "?"


print("=" * 100)
print("F/G. WHICH TABLE CONSUMED WHICH MANIFEST ROW")
print("=" * 100)
first = man.groupby("image_path", sort=False).nth(0).reset_index(drop=True)
second = man.groupby("image_path", sort=False).nth(1).reset_index(drop=True)
fm = dict(zip(feat.image_path, feat.mask_path))
m0 = dict(zip(first.image_path, first.mask_path))
m1 = dict(zip(second.image_path, second.mask_path))
n0 = n1 = nn = 0
for ip, m in fm.items():
    if m0.get(ip) == m:
        n0 += 1
    elif m1.get(ip) == m:
        n1 += 1
    else:
        nn += 1
print(f"  feature table mask == manifest row 0 : {n0}")
print(f"  feature table mask == manifest row 1 : {n1}")
print(f"  neither                              : {nn}")

# which of row0/row1 is the CORRECT one by exact-stem parse?
def correct(p, stem):
    m = PAT.match(os.path.basename(p))
    return bool(m) and m.group("stem") == stem

row0_correct = row1_correct = 0
for ip in first.image_path:
    stem = os.path.splitext(os.path.basename(ip))[0]
    if correct(m0[ip], stem):
        row0_correct += 1
    if correct(m1[ip], stem):
        row1_correct += 1
print(f"\n  manifest row 0 is the correct mask for : {row0_correct} images")
print(f"  manifest row 1 is the correct mask for : {row1_correct} images")
print(f"  -> the feature table took row 0, so it used {len(fm) - row0_correct} wrong masks"
      if n0 > n1 else "  -> the feature table took row 1")

# affected rows actually present in the feature table
feat["_stem"] = [os.path.splitext(os.path.basename(p))[0] for p in feat.image_path]
feat["_mstem"] = [PAT.match(os.path.basename(p)).group("stem") if PAT.match(os.path.basename(p)) else None
                  for p in feat.mask_path]
aff = feat[feat._stem != feat._mstem]
print(f"\n  affected rows IN biomarker_features.csv : {len(aff)}")
print(f"    by split : {aff.split.value_counts().to_dict()}")
print(f"    by source: {aff.source.value_counts().to_dict()}")
print(f"    by label : {aff.label.value_counts().to_dict()}")

print()
print("=" * 100)
print("G. BRANCH A / BRANCH C IMPACT GATE")
print("=" * 100)
# which table does branch A read?
import subprocess
g = subprocess.run(["git", "-C", f"{ROOT}", "grep", "-n", "features_dir",
                    "--", "src/classify/branch_a_tabular.py", "src/classify/branch_c_hybrid.py"],
                   capture_output=True, text=True).stdout
print("  code path evidence (grep of the two branch scripts):")
for line in g.splitlines():
    if "biomarker_features" in line:
        print(f"    {line}")
bio_sha = sha(f"{ROOT}/data/features/biomarker_features.csv")
print(f"\n  biomarker_features.csv sha256 = {bio_sha}")
print(f"  rows                          = {len(feat)}")
print(f"  rows with a wrong mask        = {len(aff)}  ({100 * len(aff) / len(feat):.2f} %)")
print("  HISTORICAL_BRANCH_A_AFFECTED = YES" if len(aff) else "  = NO")
print("  HISTORICAL_BRANCH_C_AFFECTED = YES" if len(aff) else "  = NO")
print("  (Branch C consumes the same biomarker columns by concatenation, per")
print("   src/classify/branch_c_hybrid.py; Branch B is RGB-only and cannot be affected.)")

print()
print("=" * 100)
print("D. FORENSIC TABLE (private)")
print("=" * 100)
rows = []
for _, r in aff.iterrows():
    stem = r._stem
    donor_stem = r._mstem
    # the donor image is the row whose own stem equals the borrowed mask stem
    dm = man[[os.path.splitext(os.path.basename(p))[0] == donor_stem for p in man.image_path]]
    donor = dm.iloc[0] if len(dm) else None
    exp = [p for p in os.listdir(mdir) if correct(os.path.join(mdir, p), stem)]
    rec = {
        "recipient_stable_id": stable(r.image_path),
        "recipient_split": r.split, "recipient_source": r.source, "recipient_label": int(r.label),
        "recipient_geometry": geom(r.image_path),
        "registered_mask_basename": os.path.basename(r.mask_path),
        "registered_mask_sha256": sha(r.mask_path) if os.path.exists(r.mask_path) else None,
        "expected_mask_basename": exp[0] if len(exp) == 1 else None,
        "expected_mask_sha256": sha(os.path.join(mdir, exp[0])) if len(exp) == 1 else None,
        "n_expected_candidates": len(exp),
    }
    if donor is not None:
        rec |= {"donor_stable_id": stable(donor.image_path), "donor_split": donor.split,
                "donor_source": donor.source, "donor_label": int(donor.label),
                "donor_geometry": geom(donor.image_path),
                "same_split": int(donor.split == r.split),
                "same_source": int(donor.source == r.source),
                "same_label": int(int(donor.label) == int(r.label)),
                "same_geometry": int(rec["recipient_geometry"] == geom(donor.image_path))}
    else:
        rec |= {"donor_stable_id": None, "donor_split": None, "donor_source": None,
                "donor_label": None, "donor_geometry": None,
                "same_split": None, "same_source": None, "same_label": None, "same_geometry": None}
    rows.append(rec)

# group comparison needs the key, which is private
try:
    key = pd.read_csv(f"{ROOT}/_private_audit/blinding_key.csv")
except Exception:  # noqa: BLE001
    key = None
D = pd.DataFrame(rows)
D.to_csv(f"{PRIV}/mask_pair_mismatch_forensics.csv", index=False)
print(f"  written (gitignored): {PRIV}/mask_pair_mismatch_forensics.csv  rows={len(D)}")
for c in ("same_split", "same_source", "same_label", "same_geometry"):
    if c in D:
        print(f"    {c}: {D[c].value_counts(dropna=False).to_dict()}")
print(f"    expected-mask candidates: {D.n_expected_candidates.value_counts().to_dict()}")
xor = D[(D.same_split == 0)]
print(f"\n  CROSS_SPLIT_DONOR_CASES        = {len(xor)}")
print(f"  DIFFERENT_LABEL_DONOR_CASES    = {int((D.same_label == 0).sum())}")
print(f"  cross-split rows by split pair : "
      f"{xor.groupby(['recipient_split', 'donor_split']).size().to_dict() if len(xor) else '{}'}")

print()
print("=" * 100)
print("I. CANONICAL MANIFEST")
print("=" * 100)
canon = []
for ip in man.image_path.drop_duplicates():
    stem = os.path.splitext(os.path.basename(ip))[0]
    exp = [p for p in os.listdir(mdir) if correct(os.path.join(mdir, p), stem)]
    if len(exp) != 1:
        print(f"  [FATAL] {stem}: {len(exp)} candidates -> refusing to guess")
        continue
    canon.append({"image_path": ip, "mask_path": os.path.join(mdir, exp[0])})
C = pd.DataFrame(canon)
cp = f"{ROOT}/data/masks/mask_manifest_canonical_v1.csv"
C.to_csv(cp, index=False)
print(f"  rows                : {len(C)}")
print(f"  unique image_path   : {C.image_path.nunique()}")
print(f"  unique mask_path    : {C.mask_path.nunique()}")
print(f"  original sha256     : {sha(f'{ROOT}/data/masks/mask_manifest.csv')}")
print(f"  canonical sha256    : {sha(cp)}")
chg = sum(1 for ip in C.image_path if m0.get(ip) != dict(zip(C.image_path, C.mask_path))[ip])
print(f"  mapping changes vs manifest row 0 : {chg}")
json.dump({"original_manifest_sha256": sha(f"{ROOT}/data/masks/mask_manifest.csv"),
           "canonical_manifest_sha256": sha(cp), "rows": len(C),
           "mapping_changes_vs_row0": chg,
           "biomarker_features_sha256": bio_sha,
           "affected_rows_in_feature_table": len(aff),
           "affected_by_split": aff.split.value_counts().to_dict(),
           "affected_by_source": aff.source.value_counts().to_dict(),
           "cross_split_donor_cases": len(xor),
           "different_label_donor_cases": int((D.same_label == 0).sum())},
          open(f"{PRIV}/task1_summary.json", "w"), indent=2, default=str)
print(f"\n[done]")
