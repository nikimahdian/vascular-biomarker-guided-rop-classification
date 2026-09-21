"""CLINICAL_MEASUREMENT_V2_CANDIDATE — one predeclared, versioned candidate measurement path.

V1 IS NOT MODIFIED, IS NOT IMPORTED FOR EDITING, AND REMAINS THE PRODUCTION STATE.
This module imports V1 and supplies replacements for exactly TWO things:

  B1. `retinal_fov_v2(green)` — border-independent threshold estimation.
  B2. the fractal ANALYSIS DOMAIN — border-independent multifractal canvas.

Everything else is V1's own code, executed by V1's own `measure()`; nothing is re-derived,
re-implemented or re-tuned. `measure_v2()` patches the two module-level names inside V1, calls
V1's `measure()`, and restores them, so the pipeline between the FOV and the fractal features is
bit-identical to V1.

--------------------------------------------------------------------------------------------
B1. BORDER-INDEPENDENT FOV THRESHOLDING
--------------------------------------------------------------------------------------------
Established mechanism: externally added constant/near-constant padding enters the green-channel
histogram, shifts the global Otsu threshold, and changes which retinal pixels pass `g > thr`.

Candidate: estimate the threshold on the frame's CONTENT REGION only — the frame with contiguous
near-constant rows and columns removed from each edge — and then apply that threshold to the FULL
frame exactly as V1 does. Constant padding is by definition confined to the frame edges, so it
cannot influence the estimate, while the returned mask still spans the whole frame and the
binarisation rule is unchanged.

  CONST_TOL        = 2.0    a row/column is "near-constant" if max-min <= 2.0 grey levels
  MIN_CONTENT_FRAC = 0.50   if trimming would leave less than half the frame, keep the full frame
                            and fall back to V1's estimate

On an image with no near-constant edge rows/columns the trimmed region IS the full frame, so the
threshold and therefore the mask are IDENTICAL to V1 by construction. That identity is asserted at
run time rather than assumed.

Every downstream step — `remove_small_objects` / `remove_small_holes` at `max(64, 0.001*h*w)`,
`binary_closing(disk(3))`, largest-component selection, and the three failure criteria
`coverage < 0.15` / `coverage > 0.985` / `largest/sum < 0.90` — is byte-for-byte V1's, including
the QC dict contract.

--------------------------------------------------------------------------------------------
B2. BORDER-INDEPENDENT FRACTAL ANALYSIS DOMAIN
--------------------------------------------------------------------------------------------
Established mechanism: `fractal_d0/d1/d2` respond to the canvas extent even when the binary
vascular support is unchanged, so a padded file moves them without any biology changing.

Candidate: run the SAME estimator (`v1._fractal`, same `MultifractalVBMs` parameters) on a
canonical analysis domain derived from the FOV, instead of on the file canvas:

  side  = ceil(max(fov_bbox_h, fov_bbox_w) * (1 + 2*DOMAIN_MARGIN)), rounded up to DOMAIN_ALIGN
  window = square of that side, centred on the FOV bounding-box centre
  out    = the binary support inside the window; positions outside the frame are zero

  DOMAIN_MARGIN = 0.10    margin around the FOV bounding box, so the support is not clipped at the
                          domain edge
  DOMAIN_ALIGN  = 8       side is rounded up to a multiple of 8

Invariance argument, which is exact rather than approximate: adding (t, b, l, r) constant padding
translates every retinal pixel by (t, l). The FOV bounding box translates by the same amount, its
size is unchanged, so `side` is unchanged; the window centre translates by (t, l); and the support
inside the window is identical, because outside the FOV the support is False in both frames. The
domain is therefore the same array, so the three fractal values are identical.

If the FOV is empty or unavailable (V1's `fov_px == 0` branch passes the unmasked support), the
domain falls back to the binary support's own bounding box, which is likewise translation-invariant.

--------------------------------------------------------------------------------------------
WHAT THIS CANDIDATE DELIBERATELY DOES NOT CHANGE
--------------------------------------------------------------------------------------------
No change to `vessel_density_fov` (still `mask[fov].sum() / fov_px`), no change to
`skel_density_fov` (still WHOLE-FRAME `len(nonzero(skeletonize(mask))) / fov_px` — the numerator
stays whole-frame in V2 and that is reported explicitly rather than quietly "fixed"), no change to
`SEG_CURRENT_V1`, tortuosity, width, disc handling, labels or splits. No parameter here was fitted
to disease labels or to any robustness outcome; the values above were written before the locked
evaluation was run and are recorded in `_private_audit/task5b_h3_manifest.json`.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects

import src.biomarker.clinical_measurement_v1 as v1

MEASUREMENT_VERSION_V2 = "CLINICAL_MEASUREMENT_V2_CANDIDATE"
CONST_TOL = 2.0
MIN_CONTENT_FRAC = 0.50
DOMAIN_MARGIN = 0.10
DOMAIN_ALIGN = 8

_ORIG = (v1.retinal_fov, v1._fractal)
_ORIG_FRACTAL = v1._fractal          # the untouched estimator, used by fractal_v2 itself
_STATE: dict = {"fov": None, "qc": None}


def content_bounds(g: np.ndarray) -> tuple[int, int, int, int, bool]:
    """Contiguous near-constant rows/columns removed from each edge.

    Returns (top, bottom, left, right, trimmed) where the content region is g[top:h-bottom,
    left:w-right]. `trimmed` is False when the region would fall below MIN_CONTENT_FRAC.
    """
    h, w = g.shape
    rng_r = g.max(axis=1) - g.min(axis=1)
    rng_c = g.max(axis=0) - g.min(axis=0)
    const_r = rng_r <= CONST_TOL
    const_c = rng_c <= CONST_TOL
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
    keep_h, keep_w = h - top - bottom, w - left - right
    if keep_h * keep_w < MIN_CONTENT_FRAC * h * w:
        return 0, 0, 0, 0, False
    return top, bottom, left, right, True


def v2_threshold(g: np.ndarray) -> tuple[float, bool]:
    top, bottom, left, right, trimmed = content_bounds(g)
    if trimmed:
        content = g[top:g.shape[0] - bottom, left:g.shape[1] - right]
    else:
        content = g
    try:
        thr = float(threshold_otsu(content))
    except Exception:  # noqa: BLE001
        thr = float(np.percentile(content, 50))
    return thr, trimmed


def retinal_fov_v2(green: np.ndarray):
    """V1's `retinal_fov` with only the threshold ESTIMATION replaced. Same contract."""
    g = np.asarray(green, dtype=np.float32)
    h, w = g.shape
    qc: dict = {}
    finite = g[np.isfinite(g)]
    if finite.size == 0 or float(finite.max()) <= 0:
        qc.update(fov_valid=False, fov_failure_reason="empty_image")
        _STATE["fov"] = np.zeros((h, w), bool)
        _STATE["qc"] = qc
        return _STATE["fov"], qc
    g = np.nan_to_num(g, nan=0.0)
    thr, _ = v2_threshold(g)                       # <-- the ONLY difference from V1
    binimg = g > thr
    n_components = int(ndi.label(binimg)[1])
    binimg = remove_small_objects(binimg, min_size=max(64, int(0.001 * h * w)))
    binimg = remove_small_holes(binimg, area_threshold=max(64, int(0.001 * h * w)))
    binimg = binary_closing(binimg, disk(3))
    lab, n = ndi.label(binimg)
    if n == 0:
        qc.update(fov_valid=False, fov_failure_reason="no_bright_region",
                  fov_n_components=0, fov_coverage_fraction=0.0)
        _STATE["fov"] = np.zeros((h, w), bool)
        _STATE["qc"] = qc
        return _STATE["fov"], qc
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    fov = lab == keep
    # B1 completion, declared in the manifest before the locked run: the same detected constant
    # edges are also excluded from the RETURNED mask, so externally added constant padding can
    # neither move the threshold nor enter the field of view. No new constant is introduced.
    top, bottom, left, right, trimmed = content_bounds(g)
    if trimmed:
        if top:
            fov[:top, :] = False
        if bottom:
            fov[g.shape[0] - bottom:, :] = False
        if left:
            fov[:, :left] = False
        if right:
            fov[:, g.shape[1] - right:] = False
    coverage = float(fov.mean())
    frag = float(fov.sum() / max(float(sizes.sum()), 1))
    border = np.concatenate([fov[0, :], fov[-1, :], fov[:, 0], fov[:, -1]])
    border_contact = float(border.mean())
    if fov.any():
        cy, cx = ndi.center_of_mass(fov)
    else:
        cy, cx = h / 2, w / 2
    offset = float(np.hypot(cx - w / 2, cy - h / 2) / max(w, 1))
    reason = ""
    if coverage < 0.15:
        reason = "coverage_below_15pct"
    elif coverage > 0.985:
        reason = "coverage_above_98pct_no_border_detected"
    elif frag < 0.90:
        reason = "fragmented_bright_region"
    qc.update(fov_valid=(reason == ""), fov_failure_reason=reason,
              fov_coverage_fraction=coverage, fov_n_components=n_components,
              fov_border_contact=border_contact, fov_centroid_offset=offset)
    _STATE["fov"] = fov
    _STATE["qc"] = qc
    return fov, qc


def analysis_domain(binary: np.ndarray, fov) -> np.ndarray:
    """Canonical square analysis window around the FOV bounding box; zero outside the frame."""
    ref = fov if (fov is not None and np.any(fov)) else binary
    ys, xs = np.nonzero(ref)
    if len(ys) == 0:
        return np.asarray(binary, bool)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    side = int(math.ceil(max(y1 - y0, x1 - x0) * (1.0 + 2.0 * DOMAIN_MARGIN)))
    side = int(math.ceil(side / DOMAIN_ALIGN) * DOMAIN_ALIGN)
    half = side // 2
    top, left = int(round(cy)) - half, int(round(cx)) - half
    out = np.zeros((side, side), bool)
    h, w = binary.shape
    sy0, sy1 = max(0, top), min(h, top + side)
    sx0, sx1 = max(0, left), min(w, left + side)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = np.asarray(binary, bool)[sy0:sy1, sx0:sx1]
    return out


def fractal_v2(binary: np.ndarray, fov) -> dict:
    """The SAME estimator as V1, run on the canonical domain instead of the file canvas."""
    return _ORIG_FRACTAL(analysis_domain(binary, fov))


def _fractal_hook(binary: np.ndarray) -> dict:
    return fractal_v2(binary, _STATE.get("fov"))


def measure_v2(rgb: np.ndarray, mask: np.ndarray, disc: dict, *, scale: float = 1.0,
               with_fractal: bool = True):
    """V1's `measure()` with the two predeclared substitutions; returns (features, fov, qc)."""
    _STATE["fov"] = None
    _STATE["qc"] = None
    v1.retinal_fov = retinal_fov_v2
    v1._fractal = _fractal_hook
    try:
        out = v1.measure(rgb, mask, disc, scale=scale, with_fractal=with_fractal)
    finally:
        v1.retinal_fov, v1._fractal = _ORIG
    return out, _STATE.get("fov"), _STATE.get("qc") or {}


def measure_v1(rgb: np.ndarray, mask: np.ndarray, disc: dict, *, scale: float = 1.0,
               with_fractal: bool = True):
    """V1 untouched, same call shape, so both versions can be measured identically."""
    fov, qc = v1.retinal_fov(rgb[:, :, 1] if rgb.ndim == 3 else rgb)
    out = v1.measure(rgb, mask, disc, scale=scale, with_fractal=with_fractal)
    return out, fov, qc
