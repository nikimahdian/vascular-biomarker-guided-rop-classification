"""Geometry primitives for vascular measurement, with the correctness fixes from the audit.

This module exists so that the corrected definitions are importable and testable, rather than
living inside an experiment script. Nothing here overwrites a historical feature table; callers
that adopt it must write a new feature version.

The three corrections relative to the historical code:

1. TORTUOSITY. The historical clinical definition used (number of skeleton pixels) / (hull chord).
   Pixel count is an orientation-dependent proxy for arc length: a rasterised straight vessel at 45
   degrees has ~sqrt(2) fewer pixels per unit length than an axis-aligned one, so the same straight
   vessel scored 1.005 at 0 degrees and 0.712 at 45 degrees. Here the arc is the geodesic path
   length over the 8-connected skeleton graph with diagonal steps weighted sqrt(2), and the chord
   is the Euclidean distance between the two real endpoints of the branch. Note that PVBM's own
   `median_tortuosity` and `tortuosity_index` already do this correctly; only the local clinical
   reimplementation was wrong.

2. DENSITY. The historical `vessel_density` is `mask.mean()` over the full rectangular frame, so it
   changes by -46 % when 100 px of black border is added and by -77 % when 300 px is added. A
   retinal field-of-view mask fixes this: the added pixels then sit outside both the numerator and
   the denominator.

3. WIDTH. The EDT half-width is correct and orientation-invariant on a clean binary mask, but its
   precision is set by the raster. A width taken from a mask that was produced at 256x256 and
   upsampled is quantised at the upsampling factor, which must be reported rather than hidden.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

R2 = float(np.sqrt(2))
MIN_BRANCH_PX = 8


def retinal_fov(mask: np.ndarray, *, min_frac: float = 0.02, largest: bool = True) -> np.ndarray:
    """Crude retinal field-of-view: the largest bright region of the photograph.

    The fundus image is not passed here; the caller supplies a brightness mask, because the FOV is
    a property of the photograph, not of the vessel mask. Kept simple and testable.
    """
    if mask.sum() == 0:
        return mask.astype(bool)
    lab, n = ndi.label(mask > 0)
    if n == 0:
        return mask.astype(bool)
    if not largest:
        return mask.astype(bool)
    sizes = ndi.sum(mask > 0, lab, index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    out = lab == keep
    if out.mean() < min_frac:
        return mask.astype(bool)
    return out


def fov_density(vessel_mask: np.ndarray, fov: np.ndarray | None = None) -> float:
    """Vessel pixels divided by retinal field-of-view pixels.

    With `fov=None` this reduces to the historical whole-frame density and is then, by
    construction, sensitive to black padding.
    """
    v = vessel_mask > 0
    if fov is None:
        return float(v.mean())
    denom = int((fov > 0).sum())
    return float(v.sum() / denom) if denom else float("nan")


def skeleton_branches(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Skeleton with junctions removed, so each connected component is one branch."""
    skel = skeletonize(mask > 0)
    k8 = np.ones((3, 3), np.uint8)
    nb = ndi.convolve(skel.astype(np.uint8), k8, mode="constant") * skel
    branches = skel.copy()
    branches[nb >= 4] = False
    return branches, skeletonize(mask > 0), nb  # type: ignore[return-value]


def branch_tortuosity(mask: np.ndarray, min_px: int = MIN_BRANCH_PX) -> np.ndarray:
    """arc / chord per skeleton branch, with a geometrically weighted geodesic arc.

    Returns one value per branch (>= min_px pixels). Straight branches give ~1.0 at any
    orientation; curved branches give more.
    """
    skel = skeletonize(mask > 0)
    k8 = np.ones((3, 3), np.uint8)
    nb = ndi.convolve(skel.astype(np.uint8), k8, mode="constant") * skel
    branches = skel.copy()
    branches[nb >= 4] = False
    lab, n = ndi.label(branches, structure=np.ones((3, 3), int))
    out: list[float] = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < min_px:
            continue
        idx = {(int(y), int(x)): k for k, (y, x) in enumerate(zip(ys, xs))}
        rows, cols, vals = [], [], []
        deg = np.zeros(len(ys), int)
        for k, (y, x) in enumerate(zip(ys, xs)):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    j = idx.get((int(y) + dy, int(x) + dx))
                    if j is not None:
                        rows.append(k)
                        cols.append(j)
                        vals.append(R2 if (dy and dx) else 1.0)
                        deg[k] += 1
        if not rows:
            continue
        g = coo_matrix((vals, (rows, cols)), shape=(len(ys), len(ys))).tocsr()
        ends = np.flatnonzero(deg == 1)
        if len(ends) < 2:
            ends = np.flatnonzero(deg == deg.min())
        if len(ends) < 2:
            continue
        d = dijkstra(g, indices=ends[0], directed=False)
        far = ends[int(np.argmax(d[ends]))]
        arc = float(d[far])
        chord = float(np.hypot(ys[far] - ys[ends[0]], xs[far] - xs[ends[0]]))
        if np.isfinite(arc) and chord > 1e-6:
            t = arc / chord
            if np.isfinite(t) and t < 20:
                out.append(t)
    return np.asarray(out, dtype=float)


def _order_branch(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Order one branch's pixels along the path, walking the 8-connected graph."""
    idx = {(int(y), int(x)): k for k, (y, x) in enumerate(zip(ys, xs))}
    adj: dict[int, list[int]] = {k: [] for k in idx.values()}
    for (y, x), k in idx.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = idx.get((y + dy, x + dx))
                if j is not None:
                    adj[k].append(j)
    ends = [k for k, v in adj.items() if len(v) == 1]
    start = ends[0] if ends else 0
    seen = {start}
    order = [start]
    stack = [start]
    while stack:
        cur = stack.pop()
        nxt = [j for j in adj[cur] if j not in seen]
        if nxt:
            seen.add(nxt[0])
            order.append(nxt[0])
            stack.append(nxt[0])
    return np.asarray(order, int)


def branch_tortuosity_smoothed(mask: np.ndarray, min_px: int = MIN_BRANCH_PX,
                               window: int = 5) -> np.ndarray:
    """arc/chord per branch with the raster staircase suppressed by path smoothing.

    The geodesic definition (branch_tortuosity) is exact on the pixel graph, but a rasterised
    straight line is a staircase whose 8-connected length exceeds its chord by up to 7 % at
    15/30/60/75 degrees, so straight vessels do not all score 1.0. Ordering the branch pixels
    along the path and averaging their coordinates with a small moving window removes that
    staircase; the arc is then the polyline length of the smoothed path and the chord is the
    distance between its two ends. Curvature is preserved because the window is short.
    """
    skel = skeletonize(mask > 0)
    k8 = np.ones((3, 3), np.uint8)
    nb = ndi.convolve(skel.astype(np.uint8), k8, mode="constant") * skel
    branches = skel.copy()
    branches[nb >= 4] = False
    lab, n = ndi.label(branches, structure=np.ones((3, 3), int))
    out: list[float] = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < min_px:
            continue
        order = _order_branch(ys, xs)
        if len(order) < min_px:
            continue
        py = ys[order].astype(float)
        px = xs[order].astype(float)
        w = max(3, int(window) | 1)
        if len(py) >= w:
            k = np.ones(w, float) / w
            py = np.convolve(np.pad(py, w // 2, mode="edge"), k, mode="valid")
            px = np.convolve(np.pad(px, w // 2, mode="edge"), k, mode="valid")
        arc = float(np.sum(np.hypot(np.diff(py), np.diff(px))))
        chord = float(np.hypot(py[-1] - py[0], px[-1] - px[0]))
        if np.isfinite(arc) and chord > 1e-6:
            t = arc / chord
            if np.isfinite(t) and t < 20:
                out.append(t)
    return np.asarray(out, dtype=float)


def branch_tortuosity_pixelcount(mask: np.ndarray, min_px: int = MIN_BRANCH_PX) -> np.ndarray:
    """The historical definition, kept only so tests can demonstrate what was wrong."""
    from scipy.spatial import ConvexHull
    skel = skeletonize(mask > 0)
    k8 = np.ones((3, 3), np.uint8)
    nb = ndi.convolve(skel.astype(np.uint8), k8, mode="constant") * skel
    branches = skel.copy()
    branches[nb >= 4] = False
    lab, n = ndi.label(branches, structure=np.ones((3, 3), int))
    out: list[float] = []
    for i in range(1, n + 1):
        cy, cx = np.nonzero(lab == i)
        if len(cy) < min_px:
            continue
        coords = np.column_stack([cx, cy]).astype(float)
        try:
            h = coords[np.unique(ConvexHull(coords).vertices)]
        except Exception:  # noqa: BLE001
            h = coords
        d = np.hypot(h[:, 0][:, None] - h[:, 0][None, :], h[:, 1][:, None] - h[:, 1][None, :])
        chord = float(d.max()) if len(h) > 1 else float("nan")
        if np.isfinite(chord) and chord > 1e-6:
            out.append(len(cy) / chord)
    return np.asarray(out, dtype=float)


def width_px(mask: np.ndarray) -> np.ndarray:
    """Full vessel width (2 * EDT) at every skeleton pixel, in pixels."""
    skel = skeletonize(mask > 0)
    if skel.sum() == 0:
        return np.asarray([], dtype=float)
    return 2.0 * ndi.distance_transform_edt(mask > 0)[skel]


def width_quantisation(dd_px: float, upsampling_factor: float) -> float:
    """Width quantisation in disc diameters for a mask upsampled by `upsampling_factor`.

    A binary mask produced at 1/N of the native grid has edges that step by the upsampling factor,
    so the EDT half-width steps with it.
    """
    return float(upsampling_factor / dd_px) if dd_px else float("nan")
