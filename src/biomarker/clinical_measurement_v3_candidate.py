"""CLINICAL_MEASUREMENT_V3_CANDIDATE — padding-aware FOV component selection.

V1 and V2 ARE NOT MODIFIED. V3 imports V1, retains V2's two mechanisms, and changes exactly ONE
thing: the ORDER of the padding exclusion relative to component selection.

--------------------------------------------------------------------------------------------
WHAT V3 INHERITS UNCHANGED
--------------------------------------------------------------------------------------------
B1 threshold  : Otsu estimated on the frame with contiguous near-constant edge rows/columns removed
                (CONST_TOL = 2.0 grey levels, MIN_CONTENT_FRAC = 0.50), then applied to the full
                frame. Identical constants to V2.
B2 fractal    : the analysis domain is unchanged from V2 — the same `MultifractalVBMs` estimator run
                on a square window centred on the FOV bounding-box centre, side =
                ceil(max(bbox) * (1 + 2*0.10)) rounded up to 8, zero outside the frame. Constants
                DOMAIN_MARGIN = 0.10 and DOMAIN_ALIGN = 8 are NOT touched, so the demonstrated
                zero-padding invariance is retained by construction.

--------------------------------------------------------------------------------------------
THE ONE CHANGE — exact algorithm, order of operations
--------------------------------------------------------------------------------------------
V2 ran:  threshold -> binarise -> remove_small_objects -> remove_small_holes -> closing ->
         label -> keep largest -> THEN zero the detected external padding out of the mask.
That order lets a threshold-positive external padding ring be selected as the largest component and
then be deleted, producing `coverage_below_15pct` / `fragmented_bright_region`.

V3 runs: 1. content-box detection (contiguous near-constant edge rows/columns; CONST_TOL 2.0).
         2. threshold = Otsu of the content box; if the box would fall below MIN_CONTENT_FRAC of the
            frame, keep the full frame and mark the case untrimmed.
         3. binimg = green > threshold, on the FULL frame (unchanged binarisation rule).
         4. raw component count of binimg is recorded for QC (same semantics as V1).
         5. IF trimmed: binimg[external_region] = False, where external_region is the frame minus the
            content box, i.e. exactly the four detected edge bands.
         6. remove_small_objects(min_size = max(64, 0.001*h*w))
         7. remove_small_holes(area_threshold = max(64, 0.001*h*w))
         8. binary_closing(disk(3))
         9. label; keep the largest component -> fov.
        10. coverage = fov.mean() over the FILE frame; frag = largest/sum of the labelled content
            components; border_contact and centroid as V1.
        11. failure criteria unchanged: coverage < 0.15 | coverage > 0.985 | frag < 0.90.

Step 5 is the whole candidate. Because the external padding is removed BEFORE morphology and before
labelling, it cannot be a candidate component, cannot survive closing into a ring, and cannot inflate
the `frag` denominator. When no padding is detected (untrimmed) step 5 is a no-op and V3 selects the
component exactly as V1 does with the content-derived threshold — which is also what V2 does.

FALLBACK: untrimmed frame -> no external region -> V3 == V2 (and == V1 given the same threshold).
FAILURE: unchanged reasons, unchanged thresholds. No new constant exists anywhere in V3.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects

import src.biomarker.clinical_measurement_v1 as v1
from src.biomarker.clinical_measurement_v2_candidate import (
    CONST_TOL, DOMAIN_ALIGN, DOMAIN_MARGIN, MIN_CONTENT_FRAC,
    analysis_domain, content_bounds, v2_threshold,
)

MEASUREMENT_VERSION_V3 = "CLINICAL_MEASUREMENT_V3_CANDIDATE"

_ORIG = (v1.retinal_fov, v1._fractal)
_ORIG_FRACTAL = v1._fractal
_STATE: dict = {"fov": None, "qc": None, "external": None, "content_box": None, "trimmed": False}


def external_region(shape, bounds) -> np.ndarray:
    """The frame minus the content box: the four detected constant edge bands."""
    top, bottom, left, right, trimmed = bounds
    ext = np.zeros(shape, bool)
    if not trimmed:
        return ext
    h, w = shape
    if top:
        ext[:top, :] = True
    if bottom:
        ext[h - bottom:, :] = True
    if left:
        ext[:, :left] = True
    if right:
        ext[:, w - right:] = True
    return ext


def retinal_fov_v3(green: np.ndarray):
    """V1's `retinal_fov` with B1's threshold AND the padding exclusion moved before selection."""
    g = np.asarray(green, dtype=np.float32)
    h, w = g.shape
    qc: dict = {}
    finite = g[np.isfinite(g)]
    if finite.size == 0 or float(finite.max()) <= 0:
        qc.update(fov_valid=False, fov_failure_reason="empty_image")
        _STATE.update(fov=np.zeros((h, w), bool), qc=qc, external=np.zeros((h, w), bool),
                      content_box=None, trimmed=False)
        return _STATE["fov"], qc
    g = np.nan_to_num(g, nan=0.0)

    bounds = content_bounds(g)                       # step 1
    thr, _ = v2_threshold(g)                         # step 2 (same constants as V2)
    top, bottom, left, right, trimmed = bounds
    _STATE["external"] = external_region((h, w), bounds)
    _STATE["content_box"] = (top, h - bottom, left, w - right)
    _STATE["trimmed"] = bool(trimmed)

    binimg = g > thr                                 # step 3
    n_components = int(ndi.label(binimg)[1])         # step 4
    if trimmed:                                      # step 5  <-- THE ONLY CHANGE vs V2
        binimg = binimg.copy()
        binimg[_STATE["external"]] = False
    m = max(64, int(0.001 * h * w))
    binimg = remove_small_objects(binimg, min_size=m)          # 6
    binimg = remove_small_holes(binimg, area_threshold=m)      # 7
    binimg = binary_closing(binimg, disk(3))                   # 8
    lab, n = ndi.label(binimg)                                 # 9
    if n == 0:
        qc.update(fov_valid=False, fov_failure_reason="no_bright_region",
                  fov_n_components=n_components, fov_coverage_fraction=0.0)
        _STATE.update(fov=np.zeros((h, w), bool), qc=qc)
        return _STATE["fov"], qc
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    fov = lab == keep
    coverage = float(fov.mean())                     # 10
    frag = float(sizes.max() / max(sizes.sum(), 1))
    border = np.concatenate([fov[0, :], fov[-1, :], fov[:, 0], fov[:, -1]])
    border_contact = float(border.mean())
    if fov.any():
        cy, cx = ndi.center_of_mass(fov)
    else:
        cy, cx = h / 2, w / 2
    offset = float(np.hypot(cx - w / 2, cy - h / 2) / max(w, 1))
    reason = ""                                      # 11
    if coverage < 0.15:
        reason = "coverage_below_15pct"
    elif coverage > 0.985:
        reason = "coverage_above_98pct_no_border_detected"
    elif frag < 0.90:
        reason = "fragmented_bright_region"
    qc.update(fov_valid=(reason == ""), fov_failure_reason=reason,
              fov_coverage_fraction=coverage, fov_n_components=n_components,
              fov_border_contact=border_contact, fov_centroid_offset=offset,
              fov_selected_components=n, fov_largest_frac=frag,
              fov_canvas_trimmed=bool(trimmed))
    _STATE.update(fov=fov, qc=qc)
    return fov, qc


def fractal_v3(binary: np.ndarray, fov) -> dict:
    """Byte-identical to V2's fractal path: same estimator, same domain, same constants."""
    return _ORIG_FRACTAL(analysis_domain(binary, fov))


def _fractal_hook(binary: np.ndarray) -> dict:
    return fractal_v3(binary, _STATE.get("fov"))


def measure_v3(rgb: np.ndarray, mask: np.ndarray, disc: dict, *, scale: float = 1.0,
               with_fractal: bool = True):
    """V1's `measure()` with the V3 substitutions; returns (features, fov, qc)."""
    _STATE.update(fov=None, qc=None)
    v1.retinal_fov = retinal_fov_v3
    v1._fractal = _fractal_hook
    try:
        out = v1.measure(rgb, mask, disc, scale=scale, with_fractal=with_fractal)
    finally:
        v1.retinal_fov, v1._fractal = _ORIG
    return out, _STATE.get("fov"), _STATE.get("qc") or {}


def measure_v2(rgb: np.ndarray, mask: np.ndarray, disc: dict, *, scale: float = 1.0,
               with_fractal: bool = True):
    """Re-exported so a single evaluation harness can call all three versions identically."""
    from src.biomarker import clinical_measurement_v2_candidate as v2
    return v2.measure_v2(rgb, mask, disc, scale=scale, with_fractal=with_fractal)
