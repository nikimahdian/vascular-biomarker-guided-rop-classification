"""CLINICAL_MEASUREMENT_V5_CANDIDATE — content-relative FOV coverage validity.

V1..V4 are NOT modified. V5 is V4 with exactly ONE change, named by Task 5B-H6:

  coverage denominator.
      V4: `coverage = fov_pixels / (full_canvas_h * full_canvas_w)`
      V5: `coverage = fov_pixels / (content_h * content_w)`

  `content_h * content_w` is the SAME deterministic rectangle V4 already derives in its
  border-trimming path (`content_bounds_v4`), so no new constant, no new detector and no new
  parameter is introduced.

UNCHANGED, byte for byte: the 0.15 failure threshold, the 0.985 upper threshold, the 0.90 component
ratio threshold, the FOV mask generation (threshold, external-band removal before morphology and
labelling, remove_small_objects/holes with the content-relative min_size, binary_closing(disk(3)),
largest-component selection), the fractal analysis window, the D0/D1/D2 estimator and all five
biomarker formulas.

The change is validity-only. Because `fov` is always a subset of the content rectangle (the external
band is zeroed before labelling), the V5 coverage is bounded by 1 and the FOV mask is bit-identical
to V4's on every input. Only `fov_coverage_fraction`, `fov_valid` and `fov_failure_reason` can differ.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects

import src.biomarker.clinical_measurement_v1 as v1
from src.biomarker.clinical_measurement_v2_candidate import CONST_TOL
from src.biomarker.clinical_measurement_v3_candidate import analysis_domain, external_region
from src.biomarker.clinical_measurement_v4_candidate import content_bounds_v4, v4_content, v4_threshold

MEASUREMENT_VERSION_V5 = "CLINICAL_MEASUREMENT_V5_CANDIDATE"

_ORIG = (v1.retinal_fov, v1._fractal)
_ORIG_FRACTAL = v1._fractal
_STATE: dict = {"fov": None, "qc": None, "content_area": None, "coverage": None}


def retinal_fov_v5(green: np.ndarray):
    """V4's `retinal_fov` with the coverage denominator changed to the content area."""
    g = np.asarray(green, dtype=np.float32)
    h, w = g.shape
    qc: dict = {}
    finite = g[np.isfinite(g)]
    if finite.size == 0 or float(finite.max()) <= 0:
        qc.update(fov_valid=False, fov_failure_reason="empty_image")
        _STATE.update(fov=np.zeros((h, w), bool), qc=qc, content_area=0, coverage=0.0)
        return _STATE["fov"], qc
    g = np.nan_to_num(g, nan=0.0)

    content, bounds = v4_content(g)
    top, bottom, left, right, has_content = bounds
    if content is None:
        qc.update(fov_valid=False, fov_failure_reason="no_content_region",
                  fov_n_components=0, fov_coverage_fraction=0.0)
        _STATE.update(fov=np.zeros((h, w), bool), qc=qc, content_area=0, coverage=0.0)
        return _STATE["fov"], qc
    thr = v4_threshold(g)[0]
    ch, cw = content.shape
    content_area = int(ch * cw)
    min_size = max(64, int(0.001 * content_area))
    ext = external_region((h, w), (top, bottom, left, right, has_content))

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
                  fov_n_components=n_components, fov_coverage_fraction=0.0,
                  fov_coverage_content_relative=0.0)
        _STATE.update(fov=np.zeros((h, w), bool), qc=qc, content_area=content_area, coverage=0.0)
        return _STATE["fov"], qc
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    fov = lab == (int(np.argmax(sizes)) + 1)
    coverage = float(fov.sum() / max(content_area, 1))       # <-- the ONLY change vs V4
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
              fov_coverage_fraction=coverage,
              fov_coverage_frame_relative_v4=float(fov.mean()),
              fov_content_area=content_area, fov_min_size=int(min_size), fov_threshold=float(thr),
              fov_n_components=n_components, fov_border_contact=float(border.mean()),
              fov_centroid_offset=float(np.hypot(cx - w / 2, cy - h / 2) / max(w, 1)),
              fov_selected_components=n, fov_largest_frac=frag,
              fov_canvas_trimmed=bool(ext.any()))
    _STATE.update(fov=fov, qc=qc, content_area=content_area, coverage=coverage)
    return fov, qc


def fractal_v5(binary: np.ndarray, fov) -> dict:
    """Byte-identical to V4's fractal path."""
    return _ORIG_FRACTAL(analysis_domain(binary, fov))


def _fractal_hook(binary: np.ndarray) -> dict:
    return fractal_v5(binary, _STATE.get("fov"))


def measure_v5(rgb: np.ndarray, mask: np.ndarray, disc: dict, *, scale: float = 1.0,
               with_fractal: bool = True):
    """V1's `measure()` with the V5 substitutions; returns (features, fov, qc)."""
    _STATE.update(fov=None, qc=None)
    v1.retinal_fov = retinal_fov_v5
    v1._fractal = _fractal_hook
    try:
        out = v1.measure(rgb, mask, disc, scale=scale, with_fractal=with_fractal)
    finally:
        v1.retinal_fov, v1._fractal = _ORIG
    return out, _STATE.get("fov"), _STATE.get("qc") or {}


def measure_v4(rgb, mask, disc, *, scale=1.0, with_fractal=True):
    from src.biomarker import clinical_measurement_v4_candidate as v4c
    return v4c.measure_v4(rgb, mask, disc, scale=scale, with_fractal=with_fractal)
