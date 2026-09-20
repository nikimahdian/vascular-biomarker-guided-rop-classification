"""Aspect-preserving letterbox geometry for SEG_GEOMETRY_CANDIDATE_V2.

This module is the ONLY scientific difference between SEG_CURRENT_V1 and
SEG_GEOMETRY_CANDIDATE_V2. It changes the spatial mapping between native RGB geometry and the
256x256 model input and nothing else.

SEG_CURRENT_V1: torchvision Resize((256, 256)) -- anisotropic, every geometry stretched to a
                square. A 1280x960 frame is squeezed horizontally by 1280/960 = 1.333.

SEG_GEOMETRY_CANDIDATE_V2: one isotropic scale s = min(256/H, 256/W); resize to
                round(H*s) x round(W*s); pad to exactly 256x256 with zeros, floor excess on
                top/left and the remaining pixel on bottom/right when odd.

Padding value is black / zero in the same tensor space used by V1: V1 applies ToTensor, which
scales uint8 [0, 255] to float [0, 1], so a zero-filled uint8 border is a 0.0 border in the
tensor. No reflection, no edge padding, no alternative fill value, no second target size.

The inverse transform removes the letterbox bands and maps the valid prediction region back to
the native grid with the same interpolation family V1 uses for masks (cv2.INTER_NEAREST), so the
probability/binarization ordering is unchanged: threshold first, then resize.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

TARGET = 256
PAD_VALUE = 0


def letterbox_params(h: int, w: int, target: int = TARGET) -> dict:
    """Deterministic letterbox geometry. No randomness, no dependence on content."""
    s = min(target / h, target / w)
    new_h = max(1, int(round(h * s)))
    new_w = max(1, int(round(w * s)))
    excess_h = target - new_h
    excess_w = target - new_w
    top = excess_h // 2          # floor excess on top
    left = excess_w // 2         # floor excess on left
    bottom = excess_h - top      # remaining pixel on bottom when odd
    right = excess_w - left      # remaining pixel on right when odd
    return {"scale": s, "new_h": new_h, "new_w": new_w,
            "top": top, "left": left, "bottom": bottom, "right": right}


def to_model_input(rgb: np.ndarray | Image.Image, target: int = TARGET):
    """Native RGB -> 256x256 letterboxed RGB. Returns (array, params)."""
    img = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
    w, h = img.size
    p = letterbox_params(h, w, target)
    resized = img.resize((p["new_w"], p["new_h"]), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), (PAD_VALUE, PAD_VALUE, PAD_VALUE))
    canvas.paste(resized, (p["left"], p["top"]))
    return np.asarray(canvas), p


def crop_valid(arr: np.ndarray, p: dict, target: int = TARGET) -> np.ndarray:
    """Remove the letterbox bands from a 256x256 prediction."""
    return arr[p["top"]:p["top"] + p["new_h"], p["left"]:p["left"] + p["new_w"]]


def mask_to_native(mask256: np.ndarray, p: dict, native_h: int, native_w: int) -> np.ndarray:
    """Inverse spatial transform for a binary mask, same ordering as V1.

    V1 does: post_process at 256 -> cv2.resize(mask, (orig_w, orig_h), INTER_NEAREST).
    V2 does: post_process at 256 -> crop the letterbox bands -> cv2.resize(..., INTER_NEAREST).
    Threshold timing is identical; only the crop is inserted.
    """
    import cv2

    valid = crop_valid(mask256, p)
    return cv2.resize(valid, (native_w, native_h), interpolation=cv2.INTER_NEAREST)


def prob_to_native(prob256: np.ndarray, p: dict, native_h: int, native_w: int) -> np.ndarray:
    """Inverse spatial transform for a probability map, matching V1's INTER_LINEAR ordering."""
    import cv2

    valid = crop_valid(prob256, p)
    return cv2.resize(valid.astype("float32"), (native_w, native_h),
                      interpolation=cv2.INTER_LINEAR)


def simulate_v1_mask_roundtrip(mask_native: np.ndarray) -> np.ndarray:
    """V1 spatial path applied to a native mask: anisotropic resize to 256, then back."""
    import cv2

    h, w = mask_native.shape
    small = cv2.resize(mask_native.astype(np.uint8), (TARGET, TARGET),
                       interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def simulate_v2_mask_roundtrip(mask_native: np.ndarray) -> np.ndarray:
    """V2 spatial path applied to a native mask: isotropic resize, letterbox, crop, back."""
    import cv2

    h, w = mask_native.shape
    p = letterbox_params(h, w)
    small = cv2.resize(mask_native.astype(np.uint8), (p["new_w"], p["new_h"]),
                       interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((TARGET, TARGET), np.uint8)
    canvas[p["top"]:p["top"] + p["new_h"], p["left"]:p["left"] + p["new_w"]] = small
    return mask_to_native(canvas, p, h, w)
