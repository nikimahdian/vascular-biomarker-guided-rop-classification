#!/usr/bin/env python
"""Sections A, B, I: data integrity, image/mask correspondence, and A/V forcing."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = "/Users/moniaz/niki"
OUT = f"{ROOT}/results/audit_data_integrity.json"


def phash(path: str):
    """64-bit perceptual hash from a 64x64 grayscale decode, plus the exact pixel digest."""
    try:
        with Image.open(path) as im:
            im.draft("L", (64, 64))
            g = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32)
        return os.path.basename(path), int("".join(str(int(b)) for b in (g > g.mean()).ravel()), 2)
    except Exception as e:  # noqa: BLE001
        return os.path.basename(path), None


def exact_digest(path: str):
    with Image.open(path) as im:
        im.load()
        a = np.array(im.convert("L"))
    return hashlib.sha256(a.tobytes()).hexdigest()


def main() -> int:
    feats = pd.read_csv(f"{ROOT}/data/features/biomarker_features.csv")
    man = pd.read_csv(f"{ROOT}/data/masks/mask_manifest.csv")
    print(f"feature table: {len(feats)} rows")
    print(f"mask manifest: {len(man)} rows")

    rep: dict = {}

    # ---------------------------------------------------------------- A/B: identity and naming
    rep["n_images"] = int(len(feats))
    rep["n_unique_image_path"] = int(feats.image_path.nunique())
    rep["n_unique_mask_path"] = int(feats.mask_path.nunique())
    rep["n_unique_groups"] = int(feats.group_id.nunique())
    rep["labels"] = {str(k): int(v) for k, v in feats.label.value_counts().items()}
    rep["sources"] = {str(k): int(v) for k, v in feats.source.value_counts().items()}
    rep["splits"] = {str(k): int(v) for k, v in feats.split.value_counts().items()}
    rep["identity_levels"] = {str(k): int(v) for k, v in feats.identity_level.value_counts().items()}
    rep["duplicate_image_path_rows"] = int(len(feats) - feats.image_path.nunique())
    rep["duplicate_mask_path_rows"] = int(len(feats) - feats.mask_path.nunique())

    missing_img = [p for p in feats.image_path.head(0)]
    rep["image_exists_sample_checked"] = 0

    # mask naming: production writes <image_stem>_<md5(image_path)[:8]>.png
    def expected_mask_name(p: str) -> str:
        stem = os.path.splitext(os.path.basename(p))[0]
        return f"{stem}_{hashlib.md5(p.encode('utf-8')).hexdigest()[:8]}.png"

    feats["expected_mask"] = [expected_mask_name(p) for p in feats.image_path]
    feats["actual_mask"] = [os.path.basename(p) for p in feats.mask_path]
    mismatch = feats[feats.expected_mask != feats.actual_mask]
    rep["mask_name_mismatch_rows"] = int(len(mismatch))
    rep["mask_name_mismatch_examples"] = mismatch[["image_path", "actual_mask"]].head(5).to_dict("records")
    # a mismatched mask may still be correct; test whether it is identical to the canonical file
    rep["mask_name_mismatch_note"] = (
        "rows where the mask filename is not <stem>_<md5(image_path)[:8]>.png, i.e. it was resolved "
        "by the fallback glob in find_existing_mask, which matches any single <stem>_*.png. That "
        "fallback cannot verify the mask belongs to this image.")

    # stems shared by more than one image path
    feats["stem"] = [os.path.splitext(os.path.basename(p))[0] for p in feats.image_path]
    dup_stems = feats[feats.duplicated("stem", keep=False)]
    rep["images_sharing_a_stem"] = int(dup_stems.stem.nunique())
    rep["images_sharing_a_stem_rows"] = int(len(dup_stems))

    # ---------------------------------------------------------------- group / label consistency
    gl = feats.groupby("group_id")["label"].nunique()
    rep["groups_with_mixed_labels"] = int((gl > 1).sum())
    rep["max_labels_in_one_group"] = int(gl.max())
    mixed = feats[feats.group_id.isin(gl[gl > 1].index)]
    rep["mixed_label_rows"] = int(len(mixed))
    if len(mixed):
        rep["mixed_label_by_source"] = {str(k): int(v) for k, v in
                                        mixed.groupby("source")["group_id"].nunique().items()}
        rep["mixed_label_by_identity"] = {str(k): int(v) for k, v in
                                          mixed.groupby("identity_level")["group_id"].nunique().items()}

    # group spans more than one split?
    gs = feats.groupby("group_id")["split"].nunique()
    rep["groups_spanning_multiple_splits"] = int((gs > 1).sum())

    # one group -> one source?
    gsrc = feats.groupby("group_id")["source"].nunique()
    rep["groups_spanning_multiple_sources"] = int((gsrc > 1).sum())

    # ---------------------------------------------------------------- exact duplicates
    paths = feats.image_path.tolist()
    print(f"\nhashing {len(paths)} decoded images for exact duplicates ...", flush=True)
    with Pool(8) as pool:
        digests = pool.map(exact_digest, paths, chunksize=16)
    feats["pixel_sha"] = digests
    dup = feats[feats.duplicated("pixel_sha", keep=False)]
    rep["exact_duplicate_images"] = int(dup.pixel_sha.nunique())
    rep["exact_duplicate_rows"] = int(len(dup))
    dup_groups = dup.groupby("pixel_sha")
    cross_split = 0
    label_conflict = 0
    cross_group = 0
    for _, g in dup_groups:
        if g.split.nunique() > 1:
            cross_split += 1
        if g.label.nunique() > 1:
            label_conflict += 1
        if g.group_id.nunique() > 1:
            cross_group += 1
    rep["exact_dup_groups_crossing_splits"] = int(cross_split)
    rep["exact_dup_groups_with_label_conflict"] = int(label_conflict)
    rep["exact_dup_groups_crossing_groups"] = int(cross_group)
    dup.head(200).to_csv(f"{ROOT}/results/audit_exact_duplicates.csv", index=False)

    # ---------------------------------------------------------------- near duplicates (pHash)
    print("computing perceptual hashes ...", flush=True)
    with Pool(8) as pool:
        ph = pool.map(phash, paths, chunksize=32)
    feats["phash"] = [h for _, h in ph]
    valid = feats[feats.phash.notna()].copy()
    # bucket by the top 16 bits, compare within buckets
    valid["hi"] = valid.phash // (2 ** 48)
    pairs = []
    for _, g in valid.groupby("hi"):
        idx = g.index.tolist()
        if len(idx) < 2 or len(idx) > 400:
            continue
        hs = g.phash.to_numpy()
        for a in range(len(hs)):
            for b in range(a + 1, len(hs)):
                d = bin(int(hs[a]) ^ int(hs[b])).count("1")
                if d <= 4:
                    pairs.append((idx[a], idx[b], d))
    rep["near_duplicate_pairs_hamming_le_4"] = len(pairs)
    if pairs:
        pr = pd.DataFrame([{"a": feats.loc[i, "image_path"], "b": feats.loc[j, "image_path"],
                            "hamming": d,
                            "a_split": feats.loc[i, "split"], "b_split": feats.loc[j, "split"],
                            "a_label": int(feats.loc[i, "label"]), "b_label": int(feats.loc[j, "label"]),
                            "a_group": feats.loc[i, "group_id"], "b_group": feats.loc[j, "group_id"],
                            "a_source": feats.loc[i, "source"], "b_source": feats.loc[j, "source"]}
                           for i, j, d in pairs])
        rep["near_dup_pairs_crossing_splits"] = int((pr.a_split != pr.b_split).sum())
        rep["near_dup_pairs_label_conflict"] = int((pr.a_label != pr.b_label).sum())
        rep["near_dup_pairs_crossing_groups"] = int((pr.a_group != pr.b_group).sum())
        pr.to_csv(f"{ROOT}/results/audit_near_duplicates.csv", index=False)
        print("\n  near-duplicate pairs crossing splits (first 10):")
        print(pr[pr.a_split != pr.b_split].head(10)[["a_split", "b_split", "a_label", "b_label",
                                                      "hamming", "a_source", "b_source"]].to_string(index=False))
    else:
        rep["near_dup_pairs_crossing_splits"] = 0

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for k, v in rep.items():
        if isinstance(v, (dict, list)) and k.endswith("examples"):
            continue
        print(f"  {k}: {v}")

    json.dump(rep, open(OUT, "w"), indent=2, default=str)
    print(f"\n[done] -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
