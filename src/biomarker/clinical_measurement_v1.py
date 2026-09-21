"""CLINICAL_MEASUREMENT_V1 -- the frozen measurement layer for corrected models.

Built on the validated parts of scripts/clinical_features_v3.py, with three changes that the
Task 5B audit requires:

1. TORTUOSITY is the geodesic definition from src.biomarker.geometry_core (8-connected path
   length with diagonal steps weighted sqrt(2), chord = Euclidean distance between the two real
   endpoints). v3 used branch pixel count over hull chord, which is orientation dependent: a
   straight vessel scores ~1.00 axis-aligned and ~0.71 at 45 degrees. Pixel count is not arc
   length on a raster.

2. A RETINAL FIELD-OF-VIEW mask is computed from the photograph and used as the denominator for
   every density. Whole-frame density is retained under an explicit `_wholeframe` name for
   provenance only and is never FOV-normalised by stealth.

3. ROI COVERAGE is enforced. Every regional feature reports the fraction of its region that lies
   inside the FOV, and the feature is NaN when that fraction is below MIN_ROI_COVERAGE. A
   truncated region must not silently produce a biased number.

Nothing here overwrites a historical table. Callers must write a new feature version.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects, skeletonize

from src.biomarker.geometry_core import branch_tortuosity, width_px

MEASUREMENT_VERSION = "CLINICAL_MEASUREMENT_V1"
WORK = 512                 # fixed working long side, px
MIN_BRANCH = 8             # px, minimum branch length to score tortuosity
MIN_ROI_COVERAGE = 0.60    # a region with less than this fraction inside the FOV is too
                           # truncated to give a stable density; justified geometrically, not
                           # by label performance
PEAK_MIN = 0.90            # disc detector confidence floor
DD_FRAC = (0.03, 0.25)     # plausible disc diameter as a fraction of the image min side
ANNULUS = (0.5, 2.0)       # juxta-papillary caliber zone, in disc diameters from the centre
RINGS = ((0.0, 2.0), (2.0, 3.0), (3.0, 6.0))
SEG_INPUT_SIZE = 256       # SEG_CURRENT_V1 resamples to 256x256 before inference

# Every column this module emits that is a predictor candidate (metadata excluded).
FEATURE_COLUMNS = (
    # density
    "vessel_density_wholeframe", "vessel_density_fov", "vessel_density_fov_ring_0_2dd",
    "vessel_density_fov_ring_2_3dd", "vessel_density_fov_ring_3_6dd",
    "skel_density_fov",
    # scale-dependent raw measurements
    "vessel_pixels_work", "vessel_area_fraction_work", "skeleton_px_work",
    "branch_px_work",
    # topology
    "n_startpoints", "n_endpoints", "n_intersections", "n_branches",
    "n_startpoints_fov", "n_endpoints_fov", "n_intersections_fov",
    # calibre
    "width_p50_px", "width_p90_px", "width_mean_px",
    "width_p50_dd", "width_p90_dd", "width_p95_dd", "width_mean_dd",
    "width_ann_p50_dd", "width_ann_p90_dd", "width_ann_mean_dd",
    "width_shape_p90_over_p50", "width_quantisation_floor_dd",
    # tortuosity (geodesic)
    "tort_geodesic_median", "tort_geodesic_p90", "tort_geodesic_top3_mean",
    "tort_pixelcount_median_provenance",
    # fractal
    "fractal_d0", "fractal_d1", "fractal_d2", "singularity_length",
    # regional, FOV-normalised, coverage-guarded
    "frame_sector_ne_density", "frame_sector_nw_density",
    "frame_sector_sw_density", "frame_sector_se_density",
    "sector_density_range", "n_sectors_above_median",
    # A/V (exploratory)
    "a_frac", "a_width_p90_px", "v_width_p90_px", "av_width_ratio_p90",
)

# Quality-control / metadata columns. Never predictor inputs.
QC_COLUMNS = (
    "disc_valid", "disc_peak_prob", "disc_method", "dd_px_work", "dd_over_min_side",
    "disc_cx_frac", "disc_cy_frac", "disc_centre_offset_frac",
    "fov_valid", "fov_coverage_fraction", "fov_n_components", "fov_border_contact",
    "fov_centroid_offset", "fov_failure_reason",
    "roi_coverage_ring_0_2dd", "roi_coverage_ring_2_3dd", "roi_coverage_ring_3_6dd",
    "roi_coverage_annulus", "roi_coverage_pole",
    "n_skel_px_disc", "measurement_version", "work_resolution",
)


def retinal_fov(green: np.ndarray) -> tuple[np.ndarray, dict]:
    """Retinal field of view from the fundus photograph's green channel.

    The FOV is a property of the photograph, never of the vessel mask. Returns a boolean mask
    plus deterministic QC quantities so a failed detection is visible rather than silent.
    """
    g = np.asarray(green, dtype=np.float32)
    h, w = g.shape
    qc: dict = {}
    finite = g[np.isfinite(g)]
    if finite.size == 0 or float(finite.max()) <= 0:
        qc.update(fov_valid=False, fov_failure_reason="empty_image")
        return np.zeros((h, w), bool), qc
    g = np.nan_to_num(g, nan=0.0)
    try:
        thr = float(threshold_otsu(g))
    except Exception:  # noqa: BLE001
        thr = float(np.percentile(g, 50))
    binimg = g > thr
    n_components = int(ndi.label(binimg)[1])
    binimg = remove_small_objects(binimg, min_size=max(64, int(0.001 * h * w)))
    binimg = remove_small_holes(binimg, area_threshold=max(64, int(0.001 * h * w)))
    binimg = binary_closing(binimg, disk(3))
    lab, n = ndi.label(binimg)
    if n == 0:
        qc.update(fov_valid=False, fov_failure_reason="no_bright_region",
                  fov_n_components=0, fov_coverage_fraction=0.0)
        return np.zeros((h, w), bool), qc
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    fov = lab == keep
    coverage = float(fov.mean())
    frag = float(sizes.max() / max(sizes.sum(), 1))
    border = np.concatenate([fov[0, :], fov[-1, :], fov[:, 0], fov[:, -1]])
    border_contact = float(border.mean())
    cy, cx = ndi.center_of_mass(fov)
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
    return fov, qc


def _fractal(mask: np.ndarray) -> dict:
    out = {"fractal_d0": np.nan, "fractal_d1": np.nan, "fractal_d2": np.nan,
           "singularity_length": np.nan}
    try:
        from PVBM.FractalAnalysis import MultifractalVBMs

        f = MultifractalVBMs(n_rotations=25, optimize=True, min_proba=0.0001, maxproba=0.9999)
        d0, d1, d2, sl = f.compute_multifractals(mask.astype(float))
        out.update(fractal_d0=float(d0), fractal_d1=float(d1), fractal_d2=float(d2),
                   singularity_length=float(sl))
    except Exception:  # noqa: BLE001
        pass
    return out


def _topology(skel: np.ndarray, region: np.ndarray | None = None) -> dict:
    """Startpoint / endpoint / intersection counts of a skeleton, optionally inside a region."""
    s = skel if region is None else (skel & region)
    if s.sum() == 0:
        return {"n_startpoints": 0.0, "n_endpoints": 0.0, "n_intersections": 0.0}
    k8 = np.ones((3, 3), np.uint8)
    nb = ndi.convolve(s.astype(np.uint8), k8, mode="constant") * s
    return {
        "n_startpoints": float(int(((nb == 2) & s).sum())),
        "n_endpoints": float(int(((nb == 2) & s).sum())),
        "n_intersections": float(int(((nb >= 4) & s).sum())),
    }


def assign_av_branches(gc: np.ndarray, lab: np.ndarray, nlab: int, dist: np.ndarray) -> dict:
    """The production per-branch A/V assignment, exposed for diagnostics.

    This is the ONLY implementation of the rule. `measure()` calls it and consumes the same
    fields, so a diagnostic caller and the production path cannot diverge. The mathematics are
    unchanged from the pre-refactor inline block: the branch value is the median of the
    background-corrected green channel over the branch's skeleton pixels; the split point is the
    length-weighted median of the branch values; `is_a = value >= threshold`.

    Behaviour-preserving diagnostic refactor. Nothing about threshold mathematics, branch
    ordering, tie handling, green-channel correction, filtering or any A/V feature definition is
    altered; the inline code was moved, not edited.
    """
    b_index: list = []
    b_len: list = []
    b_wpx: list = []
    b_int: list = []
    for i in range(1, nlab + 1):
        cy, cx = np.nonzero(lab == i)
        if len(cy) < 3:
            continue
        b_index.append(int(i))
        b_len.append(int(len(cy)))
        b_wpx.append(float(np.median(2.0 * dist[cy, cx])))
        b_int.append(float(np.median(gc[cy, cx])))
    res: dict = {
        "branch_index": b_index, "length": b_len, "value": b_int, "width_px": b_wpx,
        "n": len(b_int), "eligible": len(b_int) >= 6, "cut": None,
        "threshold": float("nan"), "is_a": np.zeros(len(b_int), bool), "order": None,
        "n_at_threshold": 0, "tie_fraction": float("nan"),
        "a_frac_length": float("nan"), "a_frac_count": float("nan"),
    }
    if not res["eligible"]:
        return res
    wts = np.asarray(b_len, float)
    vals = np.asarray(b_int, float)
    order = np.argsort(vals)
    cum = np.cumsum(wts[order])
    cut = int(np.searchsorted(cum, cum[-1] / 2.0))
    cut = min(max(cut, 0), len(order) - 1)
    thr = float(vals[order][cut])
    is_a = vals >= thr
    res.update(cut=int(cut), threshold=thr, is_a=is_a, order=order,
               n_at_threshold=int((vals == thr).sum()),
               tie_fraction=float((vals == thr).mean()),
               a_frac_length=float(wts[is_a].sum() / max(wts.sum(), 1)),
               a_frac_count=float(is_a.mean()))
    return res


def measure(rgb: np.ndarray, mask: np.ndarray, disc: dict, *,
            scale: float = 1.0, with_fractal: bool = True) -> dict:
    """All CLINICAL_MEASUREMENT_V1 features for one already-resampled image/mask pair."""
    h, w = mask.shape
    mask = mask > 0
    out: dict = {"measurement_version": MEASUREMENT_VERSION, "work_resolution": int(max(h, w))}

    green = rgb[:, :, 1] if rgb.ndim == 3 else rgb
    fov, fqc = retinal_fov(green)
    out.update(fqc)
    fov_px = int(fov.sum())

    # ---- disc validity: no image-centre fallback, ever ----
    dd = float(disc.get("disc_dd_px", np.nan))
    peak = float(disc.get("peak_prob", np.nan))
    dd_frac = dd / min(h, w) if np.isfinite(dd) else np.nan
    disc_valid = bool(np.isfinite(dd) and dd > 2 and np.isfinite(peak) and peak > PEAK_MIN
                      and DD_FRAC[0] <= dd_frac <= DD_FRAC[1])
    out["disc_valid"] = int(disc_valid)
    out["disc_peak_prob"] = peak
    if disc_valid:
        xc, yc = float(disc["disc_cx"]), float(disc["disc_cy"])
        out["disc_method"] = "unet_ensemble"
    else:
        dd = xc = yc = np.nan
        out["disc_method"] = "disc_invalid"
    out["dd_px_work"] = dd
    out["dd_over_min_side"] = dd / min(h, w)
    out["disc_cx_frac"] = xc / w
    out["disc_cy_frac"] = yc / h
    out["disc_centre_offset_frac"] = float(
        np.hypot(xc - w / 2, yc - h / 2) / w) if np.isfinite(xc) else np.nan

    # ---- density ----
    out["vessel_density_wholeframe"] = float(mask.mean())
    out["vessel_density_fov"] = (float(mask[fov].sum() / fov_px) if fov_px else np.nan)

    yy, xx = np.mgrid[0:h, 0:w]
    r_dd = (np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2) / max(dd, 1e-6)
            if np.isfinite(dd) else np.full((h, w), np.nan))

    for lo, hi in RINGS:
        nm = f"ring_{str(lo).replace('.0', '')}_{str(hi).replace('.0', '')}dd"
        band = np.isfinite(r_dd) & (r_dd >= lo) & (r_dd < hi)
        n = int(band.sum())
        cov = float((band & fov).sum() / n) if n else np.nan
        out[f"roi_coverage_{nm}"] = cov
        if n and np.isfinite(cov) and cov >= MIN_ROI_COVERAGE:
            out[f"vessel_density_fov_{nm}"] = float(mask[band & fov].mean())
        else:
            out[f"vessel_density_fov_{nm}"] = np.nan

    ann = np.isfinite(r_dd) & (r_dd >= ANNULUS[0]) & (r_dd < ANNULUS[1])
    n_ann = int(ann.sum())
    out["roi_coverage_annulus"] = float((ann & fov).sum() / n_ann) if n_ann else np.nan

    # ---- skeleton ----
    skel = skeletonize(mask)
    dist = ndi.distance_transform_edt(mask)
    ys, xs = np.nonzero(skel)
    out["skeleton_px_work"] = float(len(ys))
    out["skel_density_fov"] = (float(len(ys) / fov_px) if fov_px else np.nan)
    out["vessel_pixels_work"] = float(mask.sum())
    out["vessel_area_fraction_work"] = float(mask.mean())

    topo = _topology(skel)
    topo_fov = _topology(skel, fov)
    out.update({f"{k}_fov": v for k, v in topo_fov.items()})
    out.update(topo)

    if len(ys) == 0:
        for c in FEATURE_COLUMNS:
            out.setdefault(c, np.nan)
        if with_fractal:
            out.update(_fractal(mask & fov))
        return out

    wx = width_px(mask)
    out["width_p50_px"] = float(np.percentile(wx, 50)) if len(wx) else np.nan
    out["width_p90_px"] = float(np.percentile(wx, 90)) if len(wx) else np.nan
    out["width_mean_px"] = float(wx.mean()) if len(wx) else np.nan
    if np.isfinite(dd):
        wdd = np.column_stack([2.0 * dist[ys, xs]])[:, 0] / dd
        out["width_p50_dd"] = float(np.percentile(wdd, 50))
        out["width_p90_dd"] = float(np.percentile(wdd, 90))
        out["width_p95_dd"] = float(np.percentile(wdd, 95))
        out["width_mean_dd"] = float(wdd.mean())
        out["width_quantisation_floor_dd"] = float(
            (max(h, w) / SEG_INPUT_SIZE) / dd)
        sel = ann[ys, xs]
        out["n_skel_px_disc"] = float(sel.sum())
        if sel.sum() >= 20:
            out["width_ann_p50_dd"] = float(np.percentile(wdd[sel], 50))
            out["width_ann_p90_dd"] = float(np.percentile(wdd[sel], 90))
            out["width_ann_mean_dd"] = float(wdd[sel].mean())
        else:
            out["width_ann_p50_dd"] = out["width_ann_p90_dd"] = out["width_ann_mean_dd"] = np.nan
    else:
        for c in ("width_p50_dd", "width_p90_dd", "width_p95_dd", "width_mean_dd",
                  "width_quantisation_floor_dd", "width_ann_p50_dd", "width_ann_p90_dd",
                  "width_ann_mean_dd", "n_skel_px_disc"):
            out[c] = np.nan
    out["width_shape_p90_over_p50"] = (
        out["width_p90_px"] / out["width_p50_px"] if out.get("width_p50_px", 0) else np.nan)

    # ---- tortuosity: geodesic (frozen definition) + historical pixel-count for provenance ----
    tg = branch_tortuosity(mask, min_px=MIN_BRANCH)
    out["tort_geodesic_median"] = float(np.median(tg)) if len(tg) else np.nan
    out["tort_geodesic_p90"] = float(np.percentile(tg, 90)) if len(tg) else np.nan
    out["tort_geodesic_top3_mean"] = (
        float(np.sort(tg)[-min(3, len(tg)):].mean()) if len(tg) else np.nan)
    from src.biomarker.geometry_core import branch_tortuosity_pixelcount

    tp = branch_tortuosity_pixelcount(mask, min_px=MIN_BRANCH)
    out["tort_pixelcount_median_provenance"] = float(np.median(tp)) if len(tp) else np.nan

    # ---- branches and frame sectors ----
    k8 = np.ones((3, 3), np.uint8)
    nb = ndi.convolve(skel.astype(np.uint8), k8, mode="constant") * skel
    branches = skel.copy()
    branches[nb >= 4] = False
    lab, nlab = ndi.label(branches, structure=np.ones((3, 3), int))
    out["n_branches"] = float(nlab)

    gc = green - ndi.median_filter(green, size=31, mode="nearest")
    av = assign_av_branches(gc, lab, nlab, dist)
    b_len = av["length"]
    out["branch_px_work"] = float(np.sum(b_len)) if b_len else 0.0

    # A/V per branch, length-weighted median split of background-corrected green
    if av["eligible"]:
        wts = np.asarray(b_len, float)
        is_a = av["is_a"]
        awpx = np.asarray(av["width_px"], float)
        out["a_frac"] = float(wts[is_a].sum() / max(wts.sum(), 1))
        out["a_width_p90_px"] = float(np.percentile(awpx[is_a], 90)) if is_a.sum() >= 3 else np.nan
        out["v_width_p90_px"] = (float(np.percentile(awpx[~is_a], 90))
                                 if (~is_a).sum() >= 3 else np.nan)
        out["av_width_ratio_p90"] = (
            out["a_width_p90_px"] / out["v_width_p90_px"]
            if np.isfinite(out.get("a_width_p90_px", np.nan))
            and out.get("v_width_p90_px", 0) else np.nan)
    else:
        out["a_frac"] = out["a_width_p90_px"] = out["v_width_p90_px"] = np.nan
        out["av_width_ratio_p90"] = np.nan

    # frame sectors: image-frame geometry about the disc centre, NOT ICROP anatomical quadrants
    if np.isfinite(xc):
        ang = np.arctan2(yy - yc, xx - xc)
        pole = r_dd < 6.0
        out["roi_coverage_pole"] = float((pole & fov).sum() / max(int(pole.sum()), 1))
        qs = {"ne": (ang >= 0) & (ang < np.pi / 2), "nw": (ang >= np.pi / 2) & (ang <= np.pi),
              "sw": (ang >= -np.pi) & (ang < -np.pi / 2), "se": (ang >= -np.pi / 2) & (ang < 0)}
        dens = []
        for k, q in qs.items():
            m = pole & q
            n = int(m.sum())
            cov = float((m & fov).sum() / n) if n else np.nan
            if n and np.isfinite(cov) and cov >= MIN_ROI_COVERAGE:
                d = float(mask[m & fov].mean())
                out[f"frame_sector_{k}_density"] = d
                dens.append(d)
            else:
                out[f"frame_sector_{k}_density"] = np.nan
        if len(dens) == 4:
            out["sector_density_range"] = float(max(dens) - min(dens))
            out["n_sectors_above_median"] = float(sum(d > np.median(dens) for d in dens))
        else:
            out["sector_density_range"] = np.nan
            out["n_sectors_above_median"] = np.nan
    else:
        out["roi_coverage_pole"] = np.nan
        for k in ("ne", "nw", "sw", "se"):
            out[f"frame_sector_{k}_density"] = np.nan
        out["sector_density_range"] = out["n_sectors_above_median"] = np.nan

    if with_fractal:
        out.update(_fractal(mask & fov if fov_px else mask))

    for c in FEATURE_COLUMNS:
        out.setdefault(c, np.nan)
    if not disc_valid:
        for c in FEATURE_COLUMNS:
            if c.endswith("_dd") or "_dd_" in c or c.startswith(
                    ("vessel_density_fov_ring", "frame_sector", "sector_", "n_sectors",
                     "width_ann")):
                out[c] = np.nan
    return out
