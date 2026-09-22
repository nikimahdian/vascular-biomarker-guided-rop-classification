"""CLINICAL_MEASUREMENT_V4_CANDIDATE — content-relative FOV constants.

V1, V2 and V3 are NOT modified. V4 is V3 with exactly TWO changes, both identified by Task 5B-H5:

  Change 1 — remove the canvas-fraction fallback.
      V3's `content_bounds` returns `trimmed=False` when the retained content would be less than
      MIN_CONTENT_FRAC (0.50) of the file canvas, and `v2_threshold` then silently falls back to
      Otsu on the FULL padded frame, re-admitting the border. V4 uses the detected content
      rectangle whenever one exists — no fraction test, no new tuned threshold. An explicit
      failure is kept only for the case where no content rectangle can be identified at all
      (a frame that is constant in every row and column, i.e. empty content).

  Change 2 — content-relative morphology size.
      V3: `min_size = max(64, int(0.001 * full_h * full_w))` — the file canvas area, so external
      padding raises it (H5 measured median 1.22x, max 2.55x).
      V4: `min_size = max(64, int(0.001 * content_h * content_w))` — the same 0.001 rule and the
      same 64 floor, applied to the detected content domain. Neither constant is changed.

EVERYTHING ELSE IS V3, byte for byte: the edge-constancy tolerance CONST_TOL = 2.0, the Otsu
implementation, the removal of the detected external band BEFORE morphology and labelling,
remove_small_objects / remove_small_holes with the same rule, binary_closing(disk(3)), the
largest-component selection, the validity rules (coverage < 0.15 | > 0.985 | largest/sum < 0.90),
the fractal analysis window (DOMAIN_MARGIN 0.10, DOMAIN_ALIGN 8) and the D0/D1/D2 estimator.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects

import src.biomarker.clinical_measurement_v1 as v1
from src.biomarker.clinical_measurement_v2_candidate import CONST_TOL
from src.biomarker.clinical_measurement_v3_candidate import (
    DOMAIN_ALIGN, DOMAIN_MARGIN, analysis_domain, external_region,
)

MEASUREMENT_VERSION_V4 = "CLINICAL_MEASUREMENT_V4_CANDIDATE"

_ORIG = (v1.retinal_fov, v1._fractal)
_ORIG_FRACTAL = v1._fractal
_STATE: dict = {"fov": None, "qc": None, "external": None, "content_box": None, "trimmed": False,
                "content_area": None, "min_size": None, "threshold": None, "fell_back": False}


def content_bounds_v4(g: np.ndarray):
    """Change 1: contiguous near-constant edge rows/columns removed, with NO fraction test.

    Returns (top, bottom, left, right, has_content). `has_content` is False only when the content
    rectangle is empty, which is the single case that is now an explicit failure.
    """
    h, w = g.shape
    const_r = (g.max(axis=1) - g.min(axis=1)) <= CONST_TOL
    const_c = (g.max(axis=0) - g.min(axis=0)) <= CONST_TOL
    top = 0
    while top < h and const_r[top]:
        top += 1
    bottom = 0
    while bottom < h - top and const_r[h - 1 - bottom]:
        bottom += 1
    left = 0
    while left < w and const_c[left]:
        left += 1
    right = 0
    while right < w - left and const_c[w - 1 - right]:
        right += 1
    has_content = (h - top - bottom) > 0 and (w - left - right) > 0
    return top, bottom, left, right, has_content


def v4_content(g: np.ndarray):
    top, bottom, left, right, has = content_bounds_v4(g)
    if not has:
        return None, (top, bottom, left, right, has)
    return g[top:g.shape[0] - bottom, left:g.shape[1] - right], (top, bottom, left, right, has)


def v4_threshold(g: np.ndarray):
    content, bounds = v4_content(g)
    if content is None:
        return float("nan"), bounds
    try:
        return float(threshold_otsu(content)), bounds
    except Exception:  # noqa: BLE001
        return float(np.percentile(content, 50)), bounds


def retinal_fov_v4(green: np.ndarray):
    g = np.asarray(green, dtype=np.float32)
    h, w = g.shape
    qc: dict = {}
    finite = g[np.isfinite(g)]
    if finite.size == 0 or float(finite.max()) <= 0:
        qc.update(fov_valid=False, fov_failure_reason="empty_image")
        _STATE.update(fov=np.zeros((h, w), bool), qc=qc, external=np.zeros((h, w), bool),
                      content_box=None, trimmed=False, content_area=0, min_size=None,
                      threshold=None, fell_back=False)
        return _STATE["fov"], qc
    g = np.nan_to_num(g, nan=0.0)

    content, bounds = v4_content(g)                      # Change 1
    top, bottom, left, right, has_content = bounds
    if content is None:
        qc.update(fov_valid=False, fov_failure_reason="no_content_region",
                  fov_n_components=0, fov_coverage_fraction=0.0)
        _STATE.update(fov=np.zeros((h, w), bool), qc=qc, external=np.zeros((h, w), bool),
                      content_box=None, trimmed=False, content_area=0, min_size=None,
                      threshold=None, fell_back=False)
        return _STATE["fov"], qc
    thr = v4_threshold(g)[0]                             # Otsu on the content rectangle, always
    ch, cw = content.shape
    min_size = max(64, int(0.001 * ch * cw))             # Change 2
    ext = external_region((h, w), (top, bottom, left, right, has_content))
    _STATE.update(external=ext, content_box=(top, h - bottom, left, w - right),
                  trimmed=bool(ext.any()), content_area=int(ch * cw), min_size=int(min_size),
                  threshold=float(thr), fell_back=False)

    binimg = g > thr
    n_components = int(ndi.label(binimg)[1])
    if ext.any():
        binimg = binimg.copy()
        binimg[ext] = False
    binimg = remove_small_objects(binimg, min_size=min_size)
    binimg = remove_small_holes(binimg, area_threshold=min_size)
    binimg = binary_closing(binimg, disk(3))
    lab, n = ndi.label(binimg)
    if n == 0:
        qc.update(fov_valid=False, fov_failure_reason="no_bright_region",
                  fov_n_components=n_components, fov_coverage_fraction=0.0)
        _STATE.update(fov=np.zeros((h, w), bool), qc=qc)
        return _STATE["fov"], qc
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    fov = lab == (int(np.argmax(sizes)) + 1)
    coverage = float(fov.mean())
    frag = float(sizes.max() / max(sizes.sum(), 1))
    border = np.concatenate([fov[0, :], fov[-1, :], fov[:, 0], fov[:, -1]])
    if fov.any():
        cy, cx = ndi.center_of_mass(fov)
    else:
        cy, cx = h / 2, w / 2
    reason = ""
    if coverage < 0.15:
        reason = "coverage_below_15pct"
    elif coverage > 0.985:
        reason = "coverage_above_98pct_no_border_detected"
    elif frag < 0.90:
        reason = "fragmented_bright_region"
    qc.update(fov_valid=(reason == ""), fov_failure_reason=reason,
              fov_coverage_fraction=coverage, fov_n_components=n_components,
              fov_border_contact=float(border.mean()),
              fov_centroid_offset=float(np.hypot(cx - w / 2, cy - h / 2) / max(w, 1)),
              fov_selected_components=n, fov_largest_frac=frag,
              fov_canvas_trimmed=bool(ext.any()), fov_content_area=int(ch * cw),
              fov_min_size=int(min_size), fov_threshold=float(thr))
    _STATE.update(fov=fov, qc=qc)
    return fov, qc


def fractal_v4(binary: np.ndarray, fov) -> dict:
    """Byte-identical to V3's fractal path (Change 3 would be a change; there is none)."""
    return _ORIG_FRACTAL(analysis_domain(binary, fov))


def _fractal_hook(binary: np.ndarray) -> dict:
    return fractal_v4(binary, _STATE.get("fov"))


def measure_v4(rgb: np.ndarray, mask: np.ndarray, disc: dict, *, scale: float = 1.0,
               with_fractal: bool = True):
    """V1's `measure()` with the V4 substitutions; returns (features, fov, qc)."""
    _STATE.update(fov=None, qc=None)
    v1.retinal_fov = retinal_fov_v4
    v1._fractal = _fractal_hook
    try:
        out = v1.measure(rgb, mask, disc, scale=scale, with_fractal=with_fractal)
    finally:
        v1.retinal_fov, v1._fractal = _ORIG
    return out, _STATE.get("fov"), _STATE.get("qc") or {}


def measure_v3(rgb, mask, disc, *, scale=1.0, with_fractal=True):
    from src.biomarker import clinical_measurement_v3_candidate as v3
    return v3.measure_v3(rgb, mask, disc, scale=scale, with_fractal=with_fractal)
