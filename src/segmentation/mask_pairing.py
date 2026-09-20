"""Section H: the canonical image->mask matching rule, and its required tests.

The rule is one RGB image to exactly one mask, decided by an exact stem parse, never by the first
filesystem match. Adopted from scripts/repair_mask_pairing.py so it can be tested directly.
"""
from __future__ import annotations

import re
from pathlib import Path

MASK_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<h>[0-9a-f]{8})\.png$")


class MaskPairingError(RuntimeError):
    """Raised when an image does not resolve to exactly one mask. Never resolved by guessing."""


def mask_stem(mask_name: str) -> str | None:
    """The image stem a mask filename declares, or None if the name does not parse."""
    m = MASK_NAME_RE.match(Path(mask_name).name)
    return m.group("stem") if m else None


def resolve_mask(image_path: str, mask_names, *, strict: bool = True) -> str | None:
    """Return the single mask filename whose parsed stem equals this image's stem.

    `mask_names` is any iterable of filenames. Raises MaskPairingError when the result is not
    unique, unless strict=False, in which case it returns None. It never selects the first
    candidate, never uses a prefix match, and never depends on filesystem ordering.
    """
    stem = Path(str(image_path).replace("\\", "/")).stem
    hits = sorted({Path(n).name for n in mask_names if mask_stem(n) == stem})
    if len(hits) == 1:
        return hits[0]
    if strict:
        raise MaskPairingError(
            f"{stem}: expected exactly one mask, found {len(hits)} -> {hits[:4]}")
    return None
