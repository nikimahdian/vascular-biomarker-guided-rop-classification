"""Regression test: the biomarker feature table must carry grouping metadata.

Motivation: src.biomarker.extract_pvbm originally wrote only image_path/mask_path/label/
split/source, so group_id had to be re-attached by src.data.refresh_feature_splits. If that
step was skipped, Branch A/C evaluation silently used image-level bootstrap uncertainty
instead of group-level. This test makes that failure mode loud.

Also asserts the property the whole evaluation depends on: no group_id appears in more than
one split.

Skips (rather than fails) when the feature table or the split file is absent, so the test suite
still runs on a checkout without the git-ignored data directory.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.utils.common import ROOT

FEATURES = ROOT / "data/features/biomarker_features.csv"
SPLITS = ROOT / "data/splits/all.csv"
REQUIRED = ["group_id", "patient_id", "exam_id", "identity_level"]


def _load(path):
    if not path.exists():
        pytest.skip(f"{path} not present (git-ignored data directory)")
    return pd.read_csv(path)


def test_feature_table_has_grouping_metadata():
    feats = _load(FEATURES)
    missing = [c for c in REQUIRED if c not in feats.columns]
    assert not missing, (
        f"biomarker_features.csv is missing {missing}; without group_id the branch evaluation "
        "falls back to image-level bootstrap uncertainty. Run src.data.refresh_feature_splits "
        "or re-run src.biomarker.extract_pvbm with the grouping patch applied."
    )


def test_grouping_metadata_is_complete():
    feats = _load(FEATURES)
    if "group_id" not in feats.columns:
        pytest.skip("no group_id column")
    assert feats["group_id"].notna().all(), "feature table has null group_id values"


def test_feature_rows_match_split_rows():
    feats = _load(FEATURES)
    splits = _load(SPLITS)
    assert len(feats) == len(splits), (
        f"feature rows ({len(feats)}) != split rows ({len(splits)}); "
        "the feature table and the canonical split are out of sync"
    )
    assert set(feats["image_path"]) == set(splits["image_path"])


def test_no_group_crosses_splits():
    splits = _load(SPLITS)
    if "group_id" not in splits.columns:
        pytest.skip("no group_id column in split file")
    per_group = splits.groupby("group_id")["split"].nunique()
    offenders = per_group[per_group > 1]
    assert offenders.empty, (
        f"{len(offenders)} group_id values appear in more than one split, e.g. "
        f"{offenders.index[:5].tolist()}"
    )


def test_labels_are_the_documented_three_class_encoding():
    splits = _load(SPLITS)
    assert set(splits["label"].unique()) <= {0, 1, 2}
