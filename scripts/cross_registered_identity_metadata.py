#!/usr/bin/env python
"""Task 3 addendum: do the two cross-registered identities carry CONSISTENT metadata?

If the same pixels are registered under two identities, any disagreement in the filename
metadata (GA / BW / disease grade) is a label-integrity finding, not merely a duplicate.
"""
from __future__ import annotations

import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
PRIV = f"{ROOT}/_private_audit"

exc = pd.read_csv(f"{ROOT}/data/splits/excluded_ambiguous_exact_duplicates.csv")
trig = exc[exc.triggering_duplicate].copy()

TOK = ["ga", "bw", "pa", "dg", "pf", "d", "s"]


def parse(p: str) -> dict:
    name = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    out = {}
    for part in name.split("_"):
        for t in TOK:
            if part.startswith(t.upper()) and part[len(t):].isdigit():
                out[t] = part[len(t):]
    return out


print("=" * 100)
print("METADATA CONSISTENCY ACROSS THE 19 CROSS-REGISTERED EXACT-DUPLICATE PAIRS")
print("=" * 100)
pairs = []
for sha, g in trig.groupby("sha256"):
    if len(g) != 2:
        continue
    a, b = g.iloc[0], g.iloc[1]
    ma, mb = parse(a.image_path), parse(b.image_path)
    row = {"sha256": sha[:16], "group_a": a.group_id, "group_b": b.group_id,
           "label_a": int(a.label), "label_b": int(b.label)}
    for t in ("ga", "bw", "dg"):
        row[f"{t}_a"] = ma.get(t, "")
        row[f"{t}_b"] = mb.get(t, "")
        row[f"{t}_agree"] = ma.get(t) == mb.get(t)
    pairs.append(row)
P = pd.DataFrame(pairs)
print(f"  pairs analysed: {len(P)}")
for t in ("ga", "bw", "dg"):
    n = int(P[f"{t}_agree"].sum())
    print(f"  {t.upper():3s} agreement: {n} / {len(P)}"
          + ("   <- DISAGREEMENT" if n != len(P) else ""))
print()
print(f"  manifest label agreement: {int((P.label_a == P.label_b).sum())} / {len(P)}")
print()
print(P[["sha256", "group_a", "group_b", "ga_a", "ga_b", "bw_a", "bw_b",
         "dg_a", "dg_b", "label_a", "label_b"]].to_string(index=False))
P.to_csv(f"{PRIV}/cross_registered_metadata_consistency.csv", index=False)
print()
print(f"  -> private table: {PRIV}/cross_registered_metadata_consistency.csv")

print()
print("=" * 100)
print("AGGREGATE VIEW OF THE TWO IDENTITIES")
print("=" * 100)
allm = exc.copy()
allm["ga"] = allm.image_path.map(lambda p: parse(p).get("ga", ""))
allm["bw"] = allm.image_path.map(lambda p: parse(p).get("bw", ""))
allm["dg"] = allm.image_path.map(lambda p: parse(p).get("dg", ""))
for g, sub in allm.groupby("group_id"):
    print(f"  {g}: rows={len(sub)}  GA={sorted(set(sub.ga))}  BW={sorted(set(sub.bw))}  "
          f"DG={sorted(set(sub.dg))}  labels={sorted(set(sub.label))}  "
          f"exams={sub.exam_id.nunique() if 'exam_id' in sub else 'n/a'}")
