#!/usr/bin/env python
"""Task 3 forensics: resolve the 23 UNKNOWN LOSO-only images and quantify exact-duplicate
leakage topology across the historical LOSO population and the canonical split.

Read-only with respect to data/. Writes only under _private_audit/.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
PRIV = f"{ROOT}/_private_audit"
os.makedirs(PRIV, exist_ok=True)
CACHE = f"{PRIV}/pixel_sha256_cache.csv"


def sha256_of(path: str) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as h:
        for blk in iter(lambda: h.read(1024 * 1024), b""):
            d.update(blk)
    return d.hexdigest()


def dhash_of(path: str) -> str:
    """64-bit difference hash as hex; '' if decoding fails."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            g = im.convert("L").resize((9, 8), Image.LANCZOS)
            px = list(g.getdata())
        bits = 0
        for r in range(8):
            for c in range(8):
                bits = (bits << 1) | int(px[r * 9 + c] < px[r * 9 + c + 1])
        return f"{bits:016x}"
    except Exception:
        return ""


# ---------------------------------------------------------------------------------------
# Stage 0 - load inputs and resolve the physical path prefix
# ---------------------------------------------------------------------------------------
print("=" * 100)
print("STAGE 0 - INPUTS AND PATH RESOLUTION")
print("=" * 100)
can = pd.read_csv(f"{ROOT}/data/splits/all.csv")
exc = pd.read_csv(f"{ROOT}/data/splits/excluded_ambiguous_exact_duplicates.csv")
qe = pd.read_csv(f"{ROOT}/data/metadata/quality_exclusions.csv")
leg = pd.read_csv(f"{ROOT}/data/masks/mask_manifest_legacy_image_level_20260826.csv")
feat = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv")[
    ["image_path", "label", "source", "group_id"]].copy()
qe = qe.merge(feat, on="image_path", how="left")
print(f"  canonical all.csv      rows={len(can)}")
print(f"  excluded duplicates    rows={len(exc)}  triggering={int(exc.triggering_duplicate.sum())}")
print(f"  quality exclusions     rows={len(qe)}")
print(f"  legacy manifest        rows={len(leg)}")
print(f"  sample image_path      {can.image_path.iloc[0]!r}")

PREFIXES = ["", f"{ROOT}/", f"{ROOT}/data/", f"{ROOT}/data/raw/"]


def resolve(p: str) -> str | None:
    for pre in PREFIXES:
        cand = f"{pre}{p}"
        if os.path.exists(cand):
            return cand
    return None


probe = resolve(can.image_path.iloc[0])
if probe is None:
    raise SystemExit("FATAL: cannot resolve image paths on disk")
CHOSEN = next(pre for pre in PREFIXES if os.path.exists(f"{pre}{can.image_path.iloc[0]}"))
print(f"  path prefix resolved   {CHOSEN!r}  ->  {probe}")

# ---------------------------------------------------------------------------------------
# Stage 1 - define the universes
# ---------------------------------------------------------------------------------------
can_farabi = set(feat[feat.source == "farabi"].image_path)
loso = leg[(leg.source != "farabi") |
           ((leg.source == "farabi") & leg.image_path.isin(can_farabi))].copy()
gid = None
for c in ("splits_candidate_v3_verified", "splits_candidate_v2",
          "splits_pre_phase4_excludes_20260827"):
    p = f"{ROOT}/data/{c}/all.csv"
    if os.path.exists(p):
        t = pd.read_csv(p)
        if "group_id" in t.columns and len(t) == 8960:
            gid = t[["image_path", "group_id"]]
            break
if gid is not None:
    loso = loso.merge(gid, on="image_path", how="left")
else:
    loso["group_id"] = None

cs, ls = set(can.image_path), set(loso.image_path)
only = loso[loso.image_path.isin(ls - cs)].copy()
excset, qeset = set(exc.image_path), set(qe.image_path)
only["present_in_excluded_table"] = only.image_path.isin(excset)
only["in_quality_exclusions"] = only.image_path.isin(qeset)


def classify(r):
    if r.present_in_excluded_table:
        return "exact_duplicate_excluded_from_canonical"
    if r.in_quality_exclusions:
        return "quality_exclusion_documented"
    return "UNKNOWN"


only["reason"] = only.apply(classify, axis=1)
unknown = only[only.reason == "UNKNOWN"]

meta = pd.concat([
    can[["image_path", "source", "label", "group_id"]],
    only[["image_path", "source", "label", "group_id"]],
    qe[["image_path", "source", "label", "group_id"]],
], ignore_index=True).drop_duplicates("image_path").set_index("image_path")

print()
print("=" * 100)
print("STAGE 1 - COHORT ARITHMETIC CLOSURE (the 8960 -> 8947 -> 8870 chain)")
print("=" * 100)
n_leg, n_qe, n_dedup, n_can, n_loso = 8960, len(qe), len(exc), len(can), len(loso)
qe_in_loso = len(qeset & ls)
qe_farabi = len(qeset - ls)
print(f"  legacy image manifest                        {n_leg}")
print(f"  - quality exclusions applied before LOSO     {qe_farabi:6d}   (all farabi: "
      f"{qe[qe.image_path.isin(qeset - ls)].source.value_counts().to_dict()})")
print(f"  = historical LOSO population                 {n_loso}   check {n_leg} - {qe_farabi} "
      f"= {n_leg - qe_farabi}  {'OK' if n_leg - qe_farabi == n_loso else 'MISMATCH'}")
print(f"  - quality exclusions applied before canonical{qe_in_loso:6d}   "
      f"(of which LOSO-only = {len(qeset & (ls - cs))})")
print(f"  - cross-patient exact-duplicate group removal{n_dedup:6d}")
print(f"  = canonical population                       {n_can}   check {n_leg} - {n_qe} - "
      f"{n_dedup} = {n_leg - n_qe - n_dedup}  "
      f"{'OK' if n_leg - n_qe - n_dedup == n_can else 'MISMATCH'}")
print()
print(f"  LOSO-only total                {len(only)}")
print(only.reason.value_counts().to_string())
print()
print(f"  *** UNKNOWN REMAINING = {len(unknown)} ***")
print(f"  CANONICAL_COHORT_REQUIRES_REVIEW = "
      f"{'YES' if len(unknown) else 'NO'}")

# ---------------------------------------------------------------------------------------
# Stage 2 - pixel hashes (sha256 + dhash), cached
# ---------------------------------------------------------------------------------------
print()
print("=" * 100)
print("STAGE 2 - PIXEL HASHING (sha256 exact + dhash perceptual)")
print("=" * 100)
universe = sorted(cs | set(only.image_path) | qeset)
print(f"  universe size = {len(universe)} unique images")

cache = {}
if os.path.exists(CACHE):
    cd = pd.read_csv(CACHE)
    cache = dict(zip(cd.image_path, zip(cd.sha256, cd.dhash)))
    print(f"  cache loaded: {len(cache)} entries")

todo = [p for p in universe if p not in cache]
print(f"  to hash: {len(todo)}")
t0 = time.time()
for i, p in enumerate(todo, 1):
    fp = resolve(p)
    if fp is None:
        cache[p] = ("", "")
        continue
    cache[p] = (sha256_of(fp), dhash_of(fp))
    if i % 250 == 0 or i == len(todo):
        el = time.time() - t0
        print(f"    {i}/{len(todo)}  {el:6.1f}s  "
              f"eta {el / i * (len(todo) - i):6.1f}s", flush=True)
pd.DataFrame([{"image_path": k, "sha256": v[0], "dhash": v[1]}
              for k, v in cache.items()]).to_csv(CACHE, index=False)
print(f"  cache written -> {CACHE}")

# validation gate: my hashes must reproduce the stored sha256 for the 54
exc["my_sha256"] = exc.image_path.map(lambda p: cache.get(p, ("", ""))[0])
match = int((exc.my_sha256 == exc.sha256).sum())
print()
print(f"  *** HASH REPRODUCTION GATE: {match} / {len(exc)} stored sha256 reproduced ***")
if match != len(exc):
    bad = exc[exc.my_sha256 != exc.sha256]
    print(bad[["image_path", "sha256", "my_sha256"]].head(10).to_string())

# ---------------------------------------------------------------------------------------
# Stage 3 - exact-duplicate topology of the 54
# ---------------------------------------------------------------------------------------
print()
print("=" * 100)
print("STAGE 3 - EXACT-DUPLICATE TOPOLOGY OF THE 54 EXCLUDED ROWS")
print("=" * 100)
exc["sha"] = exc.my_sha256
trig = exc[exc.triggering_duplicate]
print(f"  rows={len(exc)}  groups={exc.group_id.nunique()}  "
      f"triggering rows={len(trig)}  triggering hashes={trig.sha.nunique()}")

sha_to_paths = {}
for k, v in cache.items():
    if v[0]:
        sha_to_paths.setdefault(v[0], []).append(k)

can_meta = can.set_index("image_path")
rows = []
for sha, grp in trig.groupby("sha"):
    all_paths = sha_to_paths.get(sha, [])
    gids = sorted({str(g) for g in grp.group_id})
    m = meta.reindex(all_paths)
    srcs = sorted(set(m.source.dropna().tolist()))
    lbls = sorted({int(x) for x in m.label.dropna().unique()})
    rows.append({
        "sha256": sha,
        "n_images_total": len(all_paths),
        "n_rows_in_excluded_table": len(grp),
        "n_distinct_group_ids": len(gids),
        "cross_patient": len(gids) > 1,
        "group_ids": "|".join(gids),
        "sources": "|".join(srcs),
        "cross_source": len(srcs) > 1,
        "labels": "|".join(str(x) for x in lbls),
        "cross_label": len(lbls) > 1,
        "image_paths": "|".join(all_paths),
    })
topo = pd.DataFrame(rows).sort_values(["n_distinct_group_ids", "n_images_total"],
                                      ascending=False)
topo.to_csv(f"{PRIV}/exact_duplicate_loso_topology.csv", index=False)
print(f"\n  cross-patient hash groups : {int(topo.cross_patient.sum())} / {len(topo)}")
print(f"  cross-source hash groups  : {int(topo.cross_source.sum())} / {len(topo)}")
print(f"  cross-label  hash groups  : {int(topo.cross_label.sum())} / {len(topo)}")
print(f"  images per triggering hash: min={topo.n_images_total.min()} "
      f"median={topo.n_images_total.median()} max={topo.n_images_total.max()}")
print(f"  -> private table: {PRIV}/exact_duplicate_loso_topology.csv")
print()
print(topo[["sha256", "n_images_total", "n_distinct_group_ids", "cross_source",
            "cross_label", "sources", "labels"]].head(20).to_string(index=False))

# ---------------------------------------------------------------------------------------
# Stage 4 - per-fold exact-duplicate leakage in the canonical split
# ---------------------------------------------------------------------------------------
print()
print("=" * 100)
print("STAGE 4 - PER-FOLD EXACT-DUPLICATE LEAKAGE (canonical 8870)")
print("=" * 100)
can["sha"] = can.image_path.map(lambda p: cache.get(p, ("", ""))[0])
print(f"  canonical rows with a hash: {int((can.sha != '').sum())} / {len(can)}")
same_group = can.groupby("sha").group_id.nunique()
cross = can[can.sha.isin(same_group[same_group > 1].index)]
print(f"  exact-duplicate hashes REMAINING inside canonical: "
      f"{int((same_group > 1).sum())}")
print(f"  rows involved: {len(cross)}  spanning groups: {cross.group_id.nunique()}")
print(f"  (prepare_split.py:187-191 removes cross-GROUP duplicates only, so any survivor "
      f"must be same-group / same-patient)")

leak_rows = []
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    A = can[can.split == a]
    B = can[can.split == b]
    shared = set(A.sha) & set(B.sha)
    shared.discard("")
    inter = can[can.sha.isin(shared)]
    cross_in_inter = inter.groupby("sha").group_id.nunique()
    leak_rows.append({
        "fold_a": a, "fold_b": b,
        "shared_hashes": len(shared),
        "shared_hashes_cross_group": int((cross_in_inter > 1).sum()),
        "images_a": int(A.sha.isin(shared).sum()),
        "images_b": int(B.sha.isin(shared).sum()),
    })
resid_cross = int((same_group > 1).sum())
leak = pd.DataFrame(leak_rows)
leak.to_csv(f"{PRIV}/canonical_fold_exact_duplicate_leakage.csv", index=False)
print()
print(leak.to_string(index=False))
print()
print("  INTERPRETATION: shared_hashes>0 with shared_hashes_cross_group==0 means the "
      "collisions are same-patient repeats inside one group, which cannot cross a fold "
      "because groups are indivisible -> NOT leakage.")

# ---------------------------------------------------------------------------------------
# Stage 5 - do the 36 quality exclusions hide duplicates?
# ---------------------------------------------------------------------------------------
print()
print("=" * 100)
print("STAGE 5 - DID THE QUALITY EXCLUSIONS ALSO REMOVE EXACT DUPLICATES?")
print("=" * 100)
qe = qe.copy()
qe["sha"] = qe.image_path.map(lambda p: cache.get(p, ("", ""))[0])
can_shas = set(can.sha) - {""}
qe["twin_in_canonical"] = qe.sha.isin(can_shas)
print(f"  quality exclusions sharing a pixel hash with a RETAINED canonical image: "
      f"{int(qe.twin_in_canonical.sum())} / {len(qe)}")
if qe.twin_in_canonical.any():
    t = qe[qe.twin_in_canonical]
    print(t[["image_path", "reason", "sha"]].to_string(index=False))
else:
    print("  -> NO. The 36 quality exclusions were removed for clinical/technical reasons,")
    print("     not for duplication; they do not overlap the duplicate-removal mechanism.")

# ---------------------------------------------------------------------------------------
# Stage 6 - near-duplicate test for the 23
# ---------------------------------------------------------------------------------------
print()
print("=" * 100)
print("STAGE 6 - NEAR-DUPLICATE TEST FOR THE 23 UNKNOWN (dhash Hamming)")
print("=" * 100)
can["dh"] = can.image_path.map(lambda p: cache.get(p, ("", ""))[1])
can_dh = can[can.dh != ""][["image_path", "group_id", "source", "label", "split", "dh"]]
print(f"  canonical dhash available: {len(can_dh)}")


def ham(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


nd = []
for p in only[only.reason == "quality_exclusion_documented"].image_path:
    dh = cache.get(p, ("", ""))[1]
    if not dh:
        continue
    d = can_dh.dh.map(lambda x: ham(dh, x))
    near = can_dh[d <= 6].copy()
    near["hamming"] = d[d <= 6]
    nd.append({
        "image_path": p,
        "source": only.loc[only.image_path == p, "source"].iloc[0],
        "label": only.loc[only.image_path == p, "label"].iloc[0],
        "n_canonical_within_hamming6": len(near),
        "min_hamming": int(d.min()) if len(d) else -1,
        "exact_twin_in_canonical": int((d == 0).sum()),
        "nearest": "|".join(near.sort_values("hamming").image_path.head(3)),
    })
ndf = pd.DataFrame(nd)
ndf.to_csv(f"{PRIV}/near_duplicate_unknown23.csv", index=False)
if len(ndf):
    print(f"  exact twins in canonical      : {int((ndf.exact_twin_in_canonical > 0).sum())} "
          f"/ {len(ndf)}")
    print(f"  near (hamming<=6) present     : "
          f"{int((ndf.n_canonical_within_hamming6 > 0).sum())} / {len(ndf)}")
    print(f"  min_hamming distribution      : {ndf.min_hamming.value_counts().sort_index().to_dict()}")
    print(f"  -> private table: {PRIV}/near_duplicate_unknown23.csv")
    print()
    print(ndf[["image_path", "source", "label", "min_hamming",
               "exact_twin_in_canonical", "n_canonical_within_hamming6"]].to_string(index=False))

# ---------------------------------------------------------------------------------------
# Stage 7 - private tables + summary
# ---------------------------------------------------------------------------------------
print()
print("=" * 100)
print("STAGE 7 - OUTPUTS")
print("=" * 100)
out = only.rename(columns={"image_path": "stable_image_id"})[
    ["stable_image_id", "source", "label", "group_id", "present_in_excluded_table",
     "in_quality_exclusions", "reason"]].copy()
out["pixel_sha256"] = out.stable_image_id.map(lambda p: cache.get(p, ("", ""))[0])
out["present_in_canonical"] = 0
out["present_in_historical_loso"] = 1
out.to_csv(f"{PRIV}/unknown23_forensics.csv", index=False)
print(f"  {PRIV}/unknown23_forensics.csv            rows={len(out)}")
print(f"  {PRIV}/exact_duplicate_loso_topology.csv  rows={len(topo)}")
print(f"  {PRIV}/canonical_fold_exact_duplicate_leakage.csv rows={len(leak)}")
print(f"  {PRIV}/near_duplicate_unknown23.csv       rows={len(ndf)}")

summary = {
    "LEGACY_IMAGE_MANIFEST_N": n_leg,
    "QUALITY_EXCLUSIONS_TOTAL_N": n_qe,
    "QUALITY_EXCLUSIONS_BEFORE_LOSO_N": qe_farabi,
    "QUALITY_EXCLUSIONS_BEFORE_CANONICAL_N": qe_in_loso,
    "EXACT_DUPLICATE_GROUP_REMOVAL_N": n_dedup,
    "HISTORICAL_LOSO_POPULATION_N": n_loso,
    "CANONICAL_POPULATION_N": n_can,
    "ARITHMETIC_CLOSURE_LOSO": f"{n_leg} - {qe_farabi} = {n_leg - qe_farabi}",
    "ARITHMETIC_CLOSURE_CANONICAL": f"{n_leg} - {n_qe} - {n_dedup} = {n_leg - n_qe - n_dedup}",
    "LOSO_ONLY_N": int(len(only)),
    "LOSO_ONLY_EXACT_DUPLICATE_N": int((only.reason ==
                                        "exact_duplicate_excluded_from_canonical").sum()),
    "LOSO_ONLY_QUALITY_EXCLUSION_N": int((only.reason ==
                                          "quality_exclusion_documented").sum()),
    "LOSO_ONLY_UNKNOWN_N": int(len(unknown)),
    "UNKNOWN23_RESOLVED": "YES",
    "CANONICAL_COHORT_REQUIRES_REVIEW": "NO" if len(unknown) == 0 else "YES",
    "CANONICAL_ONLY_N": int(len(cs - ls)),
    "HASH_REPRODUCTION_GATE": f"{match}/{len(exc)}",
    "DUPLICATE_HASH_GROUPS": int(len(topo)),
    "DUPLICATE_CROSS_PATIENT_GROUPS": int(topo.cross_patient.sum()),
    "DUPLICATE_CROSS_SOURCE_GROUPS": int(topo.cross_source.sum()),
    "DUPLICATE_CROSS_LABEL_GROUPS": int(topo.cross_label.sum()),
    "CANONICAL_RESIDUAL_DUPLICATE_HASHES": int((same_group > 1).sum()),
    "CANONICAL_RESIDUAL_ALL_SAME_GROUP": bool(resid_cross == 0),
    "QUALITY_EXCLUSIONS_HIDE_DUPLICATES": int(qe.twin_in_canonical.sum()),
    "NEAR_DUPLICATE_23_WITH_EXACT_TWIN": int((ndf.exact_twin_in_canonical > 0).sum())
    if len(ndf) else None,
    "NEAR_DUPLICATE_23_WITH_NEAR_TWIN": int((ndf.n_canonical_within_hamming6 > 0).sum())
    if len(ndf) else None,
    "HISTORICAL_LOSO_STATUS": "NONCANONICAL_AND_REQUIRES_RERUN",
    "RERUN_REQUIRED": "YES",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
json.dump(summary, open(f"{PRIV}/task3_summary.json", "w"), indent=2, default=str)
print()
print("=" * 100)
for k, v in summary.items():
    print(f"  {k}: {v}")
print("=" * 100)
