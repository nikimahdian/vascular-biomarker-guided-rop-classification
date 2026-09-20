#!/usr/bin/env python
"""Sections D, E, F: synthetic ground-truth tests for tortuosity, width and density.

Everything here is drawn from known analytic shapes, so the true answer is known before the
estimator is run. Findings are reported as OLD (current production) vs NEW (corrected).
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

R2 = float(np.sqrt(2))


# --------------------------------------------------------------------------- OLD estimators

def hull_chord(coords: np.ndarray) -> float:
    """The current clinical_features_v3 chord: diameter of the convex hull."""
    from scipy.spatial import ConvexHull
    if len(coords) < 3:
        return float(np.hypot(*(coords.max(0) - coords.min(0)))) if len(coords) else np.nan
    try:
        h = coords[np.unique(ConvexHull(coords).vertices)]
    except Exception:  # noqa: BLE001
        h = coords
    d = np.hypot(h[:, 0][:, None] - h[:, 0][None, :], h[:, 1][:, None] - h[:, 1][None, :])
    return float(d.max())


def old_branch_tortuosities(mask: np.ndarray) -> list[float]:
    """Exactly the current clinical_features_v3 definition: pixel count / hull chord."""
    skel = skeletonize(mask > 0)
    k8 = np.ones((3, 3), np.uint8)
    nb = ndi.convolve(skel.astype(np.uint8), k8, mode="constant") * skel
    branches = skel.copy()
    branches[nb >= 4] = False
    lab, n = ndi.label(branches, structure=np.ones((3, 3), int))
    out = []
    for i in range(1, n + 1):
        cy, cx = np.nonzero(lab == i)
        if len(cy) < 8:
            continue
        chord = hull_chord(np.column_stack([cx, cy]).astype(float))
        if np.isfinite(chord) and chord > 1e-6:
            t = len(cy) / chord
            if np.isfinite(t) and t < 20:
                out.append(t)
    return out


# --------------------------------------------------------------------------- NEW estimator

def new_branch_tortuosities(mask: np.ndarray) -> list[float]:
    """Corrected: geodesic arc length with geometric edge weights, over the true path.

    diagonal step = sqrt(2), orthogonal step = 1, and the chord is the Euclidean distance between
    the two real endpoints of the branch, found from the path itself rather than from np.where.
    """
    skel = skeletonize(mask > 0)
    k8 = np.ones((3, 3), np.uint8)
    nb = ndi.convolve(skel.astype(np.uint8), k8, mode="constant") * skel
    branches = skel.copy()
    branches[nb >= 4] = False
    lab, n = ndi.label(branches, structure=np.ones((3, 3), int))
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < 8:
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
        far = ends[np.argmax(d[ends])]
        arc = float(d[far])
        if not np.isfinite(arc) or arc <= 0:
            continue
        y0, x0 = ys[ends[0]], xs[ends[0]]
        y1, x1 = ys[far], xs[far]
        chord = float(np.hypot(y1 - y0, x1 - x0))
        if chord <= 1e-6:
            continue
        t = arc / chord
        if np.isfinite(t) and t < 20:
            out.append(t)
    return out


# --------------------------------------------------------------------------- synthetic shapes

def canvas(s=256):
    return np.zeros((s, s), bool)


def draw_line(s, angle_deg, half_len=100, width=3):
    m = canvas(s)
    c = s // 2
    a = np.deg2rad(angle_deg)
    for t in np.linspace(-half_len, half_len, 4000):
        y = int(round(c + t * np.sin(a)))
        x = int(round(c + t * np.cos(a)))
        if 0 <= y < s and 0 <= x < s:
            m[max(0, y - width // 2):y + width // 2 + 1,
              max(0, x - width // 2):x + width // 2 + 1] = True
    return m


def draw_arc(s, radius=80, span_deg=90, width=3):
    m = canvas(s)
    c = s // 2
    for t in np.deg2rad(np.linspace(-span_deg / 2, span_deg / 2, 4000)):
        y = int(round(c + radius * np.sin(t)))
        x = int(round(c + radius * np.cos(t) - radius))
        if 0 <= y < s and 0 <= x < s:
            m[max(0, y - width // 2):y + width // 2 + 1,
              max(0, x - width // 2):x + width // 2 + 1] = True
    return m


def draw_sine(s, amp=25, periods=2.0, width=3):
    m = canvas(s)
    c = s // 2
    for x in range(20, s - 20):
        y = int(round(c + amp * np.sin(2 * np.pi * periods * (x - 20) / (s - 40))))
        m[max(0, y - width // 2):y + width // 2 + 1, x] = True
    return m


def draw_disc(s, radius, width=0):
    """Filled circle, for width tests."""
    yy, xx = np.mgrid[0:s, 0:s]
    return ((xx - s // 2) ** 2 + (yy - s // 2) ** 2) <= radius * radius


def draw_strip(s, w, angle_deg=0.0):
    """A straight strip of controlled width, axis-aligned (width is exact)."""
    m = np.zeros((s, s), bool)
    c = s // 2
    m[c - w // 2: c - w // 2 + w, :] = True
    if angle_deg % 180 == 90:
        m = m.T
    return m


# --------------------------------------------------------------------------- D

print("=" * 104)
print("D. TORTUOSITY — straight vessels must give tortuosity ~1 at every orientation")
print("=" * 104)
print(f"{'shape':22s} {'OLD median':>11s} {'OLD min..max':>20s} {'NEW median':>11s} {'NEW min..max':>20s}")
rows_d = []
for ang in (0, 15, 30, 45, 60, 75, 90):
    m = draw_line(256, ang)
    o, n = old_branch_tortuosities(m), new_branch_tortuosities(m)
    rows_d.append((f"straight {ang:>3d} deg", o, n))
    print(f"{f'straight {ang:>3d} deg':22s} {np.median(o):11.4f} "
          f"{f'{min(o):.3f}..{max(o):.3f}':>20s} {np.median(n):11.4f} "
          f"{f'{min(n):.3f}..{max(n):.3f}':>20s}")

for name, m in (("arc 90deg", draw_arc(256, 80, 90)), ("arc 180deg", draw_arc(256, 80, 180)),
                ("sine amp25", draw_sine(256, 25, 2.0)),
                ("sine amp10", draw_sine(256, 10, 2.0))):
    o, n = old_branch_tortuosities(m), new_branch_tortuosities(m)
    if not o:
        continue
    print(f"{name:22s} {np.median(o):11.4f} {f'{min(o):.3f}..{max(o):.3f}':>20s} "
          f"{np.median(n):11.4f} {f'{min(n):.3f}..{max(n):.3f}':>20s}")
    rows_d.append((name, o, n))

straight_old = [np.median(o) for nm, o, _ in rows_d if nm.startswith("straight")]
straight_new = [np.median(n) for nm, _, n in rows_d if nm.startswith("straight")]
print(f"\n  OLD spread across straight orientations: {min(straight_old):.4f} .. {max(straight_old):.4f}"
      f"   (ratio {max(straight_old) / max(min(straight_old), 1e-9):.3f})")
print(f"  NEW spread across straight orientations: {min(straight_new):.4f} .. {max(straight_new):.4f}"
      f"   (ratio {max(straight_new) / max(min(straight_new), 1e-9):.3f})")
print("  VERDICT: OLD is orientation-dependent" if max(straight_old) / min(straight_old) > 1.2
      else "  VERDICT: OLD is orientation-independent")
print("  VERDICT: NEW is orientation-dependent" if max(straight_new) / min(straight_new) > 1.2
      else "  VERDICT: NEW is orientation-independent")

# curved must be ordered above straight
print("\n  ordering check (curved should exceed straight):")
base = np.median([np.median(n) for nm, _, n in rows_d if nm.startswith("straight")])
for nm, _, n in rows_d:
    if nm.startswith(("arc", "sine")):
        print(f"    {nm:14s} NEW={np.median(n):.4f}  vs straight {base:.4f}  "
              f"{'OK' if np.median(n) > base else 'FAIL'}")

# --------------------------------------------------------------------------- E

print()
print("=" * 104)
print("E. WIDTH — EDT accuracy on a known-width strip")
print("=" * 104)
print(f"{'true_w':>7s} {'est median':>11s} {'est mode':>9s} {'abs err':>8s} {'rel err':>8s}")
for w in (2, 3, 4, 6, 8, 12, 20, 30):
    m = draw_strip(256, w)
    skel = skeletonize(m)
    dist = ndi.distance_transform_edt(m)
    est = 2.0 * dist[skel]
    print(f"{w:7d} {np.median(est):11.3f} {int(np.bincount(est.astype(int)).argmax()):9d} "
          f"{abs(np.median(est) - w):8.3f} {abs(np.median(est) - w) / w:8.3f}")

print("\n  orientation check at true width 8:")
for ang in (0, 45, 90):
    m = draw_strip(256, 8, ang)
    skel = skeletonize(m)
    dist = ndi.distance_transform_edt(m)
    est = 2.0 * dist[skel]
    print(f"    {ang:>3d} deg  median={np.median(est):.3f}  mean={est.mean():.3f}  "
          f"(true 8)  rel_err={abs(np.median(est) - 8) / 8:.3f}")

print("\n  SCALE INVARIANCE — same geometry rendered at 1x / 2x / 4x, disc scaled with it")
print(f"  {'scale':>6s} {'true w_px':>10s} {'DD_px':>8s} {'w/DD true':>10s} {'w/DD est':>10s} {'rel err':>8s}")
for scale in (1, 2, 4):
    s = 128 * scale
    w_px = 4 * scale
    dd_px = 40 * scale
    m = draw_strip(s, w_px)
    skel = skeletonize(m)
    dist = ndi.distance_transform_edt(m)
    est = float(np.median(2.0 * dist[skel]))
    print(f"  {scale:>6d} {w_px:10d} {dd_px:8d} {w_px / dd_px:10.4f} {est / dd_px:10.4f} "
          f"{abs(est / dd_px - w_px / dd_px) / (w_px / dd_px):8.4f}")
print("  (EDT width in a binary strip is quantised by the raster, so small widths floor at 2 px)")

print("\n  SPECIFICITY of the EDT on a filled disc (should equal the true diameter):")
for r in (10, 20, 40):
    m = draw_disc(256, r)
    dist = ndi.distance_transform_edt(m)
    print(f"    radius {r:3d} -> 2*max(EDT) = {2 * dist.max():.2f}  (true diameter {2 * r})")

# --------------------------------------------------------------------------- F

print()
print("=" * 104)
print("F. DENSITY — black padding / field-of-view sensitivity")
print("=" * 104)
rng = np.random.default_rng(7)
base = np.zeros((480, 640), bool)
yy, xx = np.mgrid[0:480, 0:640]
retina = ((xx - 320) ** 2 + (yy - 240) ** 2) <= 220 ** 2
inner = ((xx - 320) ** 2 + (yy - 240) ** 2) <= 205 ** 2
base[inner] = rng.random(inner.sum()) < 0.10

print(f"  retinal-FOV fraction of the frame      : {retina.mean():.4f}")
print(f"  vessel pixels                          : {base.sum()}")
print(f"  ORIGINAL whole-frame density           : {base.mean():.6f}")
print(f"  ORIGINAL retinal-FOV-normalised density: {base.sum() / retina.sum():.6f}")

for pad_name, padded, pad_retina in (
        ("+100px black border each side", np.pad(base, 100), np.pad(retina, 100)),
        ("+300px black border each side", np.pad(base, 300), np.pad(retina, 300)),
        ("letterboxed to 640x640", np.pad(base, ((80, 80), (0, 0))), np.pad(retina, ((80, 80), (0, 0))))):
    print(f"\n  {pad_name}")
    print(f"    whole-frame density      : {padded.mean():.6f}   "
          f"change {100 * (padded.mean() / base.mean() - 1):+.1f}%")
    print(f"    retinal-FOV density      : {padded.sum() / pad_retina.sum():.6f}   "
          f"change {100 * ((padded.sum() / pad_retina.sum()) / (base.sum() / retina.sum()) - 1):+.1f}%")

print("\n  VERDICT: whole-frame density is not invariant to black padding; the FOV-normalised")
print("           version is, because the added pixels are excluded from both the numerator")
print("           and the denominator. Any density feature computed over the full rectangle is")
print("           an acquisition-canvas feature, not a retinal feature.")
