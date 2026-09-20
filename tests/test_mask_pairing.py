"""Section H tests: the canonical image -> mask matching rule.

Requirement: one RGB image resolves to exactly one mask, by an unambiguous stable identity, with no
prefix collision, no dependence on filesystem ordering, deterministic across platforms, and a loud
failure on zero or multiple candidates. It must never silently select the first candidate.
"""
from __future__ import annotations

import pytest

from src.segmentation.mask_pairing import MaskPairingError, mask_stem, resolve_mask

IMAGE = "data/raw/plus/aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1.jpg"


def test_exact_stem_is_extracted():
    assert mask_stem("aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1_bd7c547e.png") == \
           "aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1"
    assert mask_stem("not-a-mask.png") is None
    assert mask_stem("x_short.png") is None
    assert mask_stem("x_GGGGGGGG.png") is None


def test_prefix_sibling_is_not_selected():
    """cf.1 must never resolve to cf.11, cf.12 ... even when it is the only file on disk."""
    sibling = "aaaaaaaa-bbbb-4ccc-8ddd-000000000000.11_bd7c547e.png"
    own = "aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1_f3829e03.png"
    assert resolve_mask(IMAGE, [own, sibling]) == own
    with pytest.raises(MaskPairingError):
        resolve_mask(IMAGE, [sibling])          # only the sibling exists -> loud failure
    with pytest.raises(MaskPairingError):
        resolve_mask(IMAGE.replace(".1.jpg", ".12.jpg"), [own])


def test_strict_prefix_with_underscore_is_not_selected():
    """The pattern the historical glob WOULD have accepted: stem + '_' + more."""
    stem = "aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1"
    prefixed = f"{stem}_2_bd7c547e.png"
    own = f"{stem}_f3829e03.png"
    assert resolve_mask(IMAGE, [prefixed, own]) == own
    with pytest.raises(MaskPairingError):
        resolve_mask(IMAGE, [prefixed])


def test_two_candidates_fail_loudly():
    a = "aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1_aaaaaaaa.png"
    b = "aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1_bbbbbbbb.png"
    with pytest.raises(MaskPairingError):
        resolve_mask(IMAGE, [a, b])
    assert resolve_mask(IMAGE, [a, b], strict=False) is None


def test_missing_mask_fails_loudly():
    with pytest.raises(MaskPairingError):
        resolve_mask(IMAGE, [])
    with pytest.raises(MaskPairingError):
        resolve_mask(IMAGE, ["unrelated_12345678.png"])


def test_duplicate_filenames_in_different_directories_collapse_to_one():
    """The same mask basename found in two folders is one mask, not two candidates."""
    a = "masks_a/aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1_f3829e03.png"
    b = "masks_b/aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1_f3829e03.png"
    assert resolve_mask(IMAGE, [a, b]) == "aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1_f3829e03.png"


def test_mixed_path_separators_are_handled():
    win = "data\\raw\\plus\\aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1.jpg"
    assert resolve_mask(win, ["aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1_f3829e03.png"]) == \
           "aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1_f3829e03.png"


def test_result_is_independent_of_input_order():
    """A resolvable set must give the same answer whatever order the filesystem returns."""
    names = ["aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1_f3829e03.png",
             "aaaaaaaa-bbbb-4ccc-8ddd-000000000000.11_bd7c547e.png",
             "aaaaaaaa-bbbb-4ccc-8ddd-000000000000.19_78bc1e6e.png",
             "unrelated_12345678.png"]
    assert resolve_mask(IMAGE, names) == resolve_mask(IMAGE, list(reversed(names))) == \
           "aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1_f3829e03.png"


def test_hash_suffix_case_is_significant():
    """Uppercase hex is not the output format, so an uppercase name must not be accepted."""
    upper = "aaaaaaaa-bbbb-4ccc-8ddd-000000000000.1_BD7C547E.png"
    assert mask_stem(upper) is None
    with pytest.raises(MaskPairingError):
        resolve_mask(IMAGE, [upper])
