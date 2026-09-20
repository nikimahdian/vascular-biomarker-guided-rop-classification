#!/usr/bin/env python3
"""Clinical biomarker v3 -- the measurement layer, rebuilt.

Defects in extract_clinical_v2.py that this file fixes:
  1. DD was a CONSTANT (min(h,w)//10 for all 8870 rows, because PVBM's DiscSegmenter always
     raised and the bare `except` fell through). Now DD comes from the HVDROPDB-trained
     U-Net ensemble. This is the mechanism behind the project's own SEVERE_SOURCE_CONFOUNDING
     verdict -- "width in DD" was really "width in units of image size", i.e. a camera fingerprint.
     There is NO fallback: if the disc does not pass DISC_VALID (peak probability and a
     plausible diameter as a fraction of the image), every disc-relative feature is NaN and the
     image is excluded from disc-relative analysis. Replacing a failed disc detection with the
     image centre re-introduces exactly the defect this file exists to remove.
     Features that do not need a disc (vessel density, skeleton density, branch count, width in
     pixels, tortuosity ratios, and the artery/vein width ratio) are still computed, so the
     images without a valid disc are not simply dropped from everything.
  2. Width was sampled on EVERY vessel pixel (dist[mask>0]); that distribution is edge-weighted
     (triangular, zero at the vessel border) and biased low. Now width is sampled on the
     vessel SKELETON only.
  3. No anatomical restriction. Now also reported inside a peripapillary annulus of 0.5-2.0 DD.
  4. A/V was a per-PIXEL median split of the green channel, which forces a 50/50 split and
     shreds single vessels into artery and vein pixels. Now A/V is assigned per skeleton
     BRANCH, using background-corrected green intensity.
  5. Tortuosity used xs[-1]-xs[0] from np.where, which is not a path order. Now the chord is the
     diameter of the branch's convex hull and the arc is its pixel count.

Everything is computed at a fixed working resolution (long side 512) and divided by DD in the
same pixels, so the features are invariant to the acquisition resolution.

Writes data/features/biomarker_features_clinical_v3.csv  (new table only; overwrites nothing).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage.morphology import skeletonize

sys.path.insert(0, "/Users/moniaz/niki")
from src.utils.common import append_csv_rows, ensure_dirs, load_config  # noqa: E402

WORK = 512          # working long side, px
MIN_BRANCH = 8      # px, minimum branch length to score tortuosity
BAND = (0.5, 2.0)   # juxta-papillary caliber zone, in disc diameters from the disc CENTRE
                    # (= from the disc margin out to 1.5 DD). ICROP3 assesses Plus over a wider
                    # posterior region than this, so the wider ring_2_3dd / ring_3_6dd densities
                    # are reported alongside and should be used for the Plus-related analysis.
PEAK_MIN = 0.9      # disc-detector confidence floor. LOCKED at this value for the frozen
                    # analysis; the expert disc annotation is what should recalibrate it.
DD_FRAC = (0.03, 0.25)  # plausible disc diameter as a fraction of the image min side

# Features that are only defined relative to a real optic disc. If DISC_VALID is false these are
# returned as NaN. There is deliberately NO image-centre fallback: a fabricated disc diameter is
# exactly what turned "width in disc diameters" into a proxy for image size.
DISC_DEPENDENT = (
    "dd_px_work", "dd_over_min_side", "disc_cx_frac", "disc_cy_frac", "disc_centre_offset_frac",
    "density_ring_0_2dd", "coverage_ring_0_2dd",
    "density_ring_2_3dd", "coverage_ring_2_3dd",
    "density_ring_3_6dd", "coverage_ring_3_6dd",
    "density_q_ne", "density_q_nw", "density_q_sw", "density_q_se",
    "n_quad_above_median", "quad_density_max", "quad_density_min", "quad_density_range",
    "width_p50_dd", "width_p90_dd", "width_p95_dd", "width_mean_dd",
    "width_ann_p50_dd", "width_ann_p90_dd", "width_ann_mean_dd", "n_skel_ann_px",
    "a_width_p90_dd", "v_width_p90_dd",
)
META = ["image_path", "mask_path", "label", "split", "source", "group_id",
        "patient_id", "exam_id", "identity_level"]


def load_pair(image_path: str, mask_path: str):
    img = Image.open(image_path).convert("RGB")
    w0, h0 = img.size
    sc = WORK / max(h0, w0)
    ws, hs = max(8, int(round(w0 * sc))), max(8, int(round(h0 * sc)))
    rgb = np.asarray(img.resize((ws, hs), Image.BILINEAR), dtype=np.float32)
    msk = np.asarray(Image.open(mask_path).convert("L").resize((ws, hs), Image.NEAREST)) > 127
    return rgb, msk.astype(np.uint8), sc


def hull_chord(coords: np.ndarray) -> float:
    if len(coords) < 3:
        return float(np.hypot(*(coords.max(0) - coords.min(0)))) if len(coords) else np.nan
    try:
        h = coords[np.unique(ConvexHull(coords).vertices)]
    except Exception:  # noqa: BLE001  degenerate/collinear hull
        h = coords
    if len(h) < 2:
        return np.nan
    d = np.hypot(h[:, 0][:, None] - h[:, 0][None, :], h[:, 1][:, None] - h[:, 1][None, :])
    return float(d.max())


def measure(rgb: np.ndarray, mask: np.ndarray, disc: dict) -> dict:
    """All features for one image, computed at the working resolution.

    If the disc geometry does not pass DISC_VALID, every disc-relative feature is returned as
    NaN. There is no image-centre fallback.
    """
    h, w = mask.shape
    out: dict[str, float | str] = {}
    dd = float(disc.get("disc_dd_px", np.nan))
    peak = float(disc.get("peak_prob", np.nan))
    dd_frac = dd / min(h, w) if np.isfinite(dd) else np.nan
    disc_valid = bool(np.isfinite(dd) and dd > 2 and np.isfinite(peak)
                      and peak > PEAK_MIN and DD_FRAC[0] <= dd_frac <= DD_FRAC[1])
    out["disc_valid"] = int(disc_valid)
    out["disc_peak_prob"] = peak
    if disc_valid:
        xc, yc = float(disc["disc_cx"]), float(disc["disc_cy"])
        out["disc_method"] = "unet_ensemble"
    else:
        dd = xc = yc = np.nan          # nothing disc-relative is computable
        out["disc_method"] = "disc_invalid"
    out["dd_px_work"] = dd
    out["dd_over_min_side"] = dd / min(h, w)
    out["disc_cx_frac"] = xc / w
    out["disc_cy_frac"] = yc / h
    out["disc_centre_offset_frac"] = float(np.hypot(xc - w / 2, yc - h / 2) / w)

    yy, xx = np.mgrid[0:h, 0:w]
    r_dd = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2) / max(dd, 1e-6)
    ann = (r_dd >= BAND[0]) & (r_dd < BAND[1])

    for lo, hi, nm in ((0.0, 2.0, "ring_0_2dd"), (2.0, 3.0, "ring_2_3dd"), (3.0, 6.0, "ring_3_6dd")):
        band = (r_dd >= lo) & (r_dd < hi)
        out[f"density_{nm}"] = float(mask[band].mean()) if band.sum() else np.nan
        out[f"coverage_{nm}"] = float(band.mean())
    ang = np.arctan2(yy - yc, xx - xc)
    in_pole = r_dd < 6.0
    qs = {"ne": (ang >= 0) & (ang < np.pi / 2), "nw": (ang >= np.pi / 2) & (ang <= np.pi),
          "sw": (ang >= -np.pi) & (ang < -np.pi / 2), "se": (ang >= -np.pi / 2) & (ang < 0)}
    dens = []
    for k, q in qs.items():
        m = in_pole & q
        d = float(mask[m].mean()) if m.sum() else np.nan
        out[f"density_q_{k}"] = d
        if np.isfinite(d):
            dens.append(d)
    if dens:
        out["n_quad_above_median"] = float(sum(d > np.median(dens) for d in dens))
        out["quad_density_max"] = float(np.max(dens))
        out["quad_density_min"] = float(np.min(dens))
        out["quad_density_range"] = out["quad_density_max"] - out["quad_density_min"]
    out["vessel_density"] = float(mask.mean())

    if mask.sum() == 0:
        for k in ("skel_density", "width_p50_px", "width_p90_px", "width_mean_px",
                  "width_p50_dd", "width_p90_dd", "width_p95_dd", "width_mean_dd",
                  "width_ann_p50_dd", "width_ann_p90_dd", "width_ann_mean_dd", "n_skel_px",
                  "n_skel_ann_px", "width_shape_p90_over_p50", "tort_median", "tort_p90",
                  "tort_top3_mean", "n_branches", "a_width_p90_px", "v_width_p90_px",
                  "a_width_p90_dd", "v_width_p90_dd",
                  "av_width_ratio_p90", "a_frac", "a_tort_median", "v_tort_median"):
            out[k] = np.nan
        if not disc_valid:
            for k in DISC_DEPENDENT:
                out[k] = np.nan
        return out

    skel = skeletonize(mask > 0)
    dist = ndi.distance_transform_edt(mask > 0)
    ys, xs = np.nonzero(skel)
    out["n_skel_px"] = float(len(ys))
    out["skel_density"] = float(skel.mean())
    width_px = 2.0 * dist[ys, xs]                       # full width in working pixels
    out["width_p50_px"] = float(np.percentile(width_px, 50))
    out["width_p90_px"] = float(np.percentile(width_px, 90))
    out["width_mean_px"] = float(width_px.mean())
    width = (width_px / max(dd, 1e-6)) if np.isfinite(dd) else np.full(len(width_px), np.nan)
    out["width_p50_dd"] = float(np.percentile(width, 50)) if np.isfinite(dd) else np.nan
    out["width_p90_dd"] = float(np.percentile(width, 90)) if np.isfinite(dd) else np.nan
    out["width_p95_dd"] = float(np.percentile(width, 95)) if np.isfinite(dd) else np.nan
    out["width_mean_dd"] = float(width.mean()) if np.isfinite(dd) else np.nan
    out["width_shape_p90_over_p50"] = (out["width_p90_px"] / out["width_p50_px"]
                                       if out["width_p50_px"] > 0 else np.nan)
    sel = ann[ys, xs] if np.isfinite(dd) else np.zeros(len(ys), dtype=bool)
    out["n_skel_ann_px"] = float(sel.sum())
    if sel.sum() >= 20:
        wa = width[sel]
        out["width_ann_p50_dd"] = float(np.percentile(wa, 50))
        out["width_ann_p90_dd"] = float(np.percentile(wa, 90))
        out["width_ann_mean_dd"] = float(wa.mean())
    else:
        out["width_ann_p50_dd"] = out["width_ann_p90_dd"] = out["width_ann_mean_dd"] = np.nan

    # ---- branches: skeleton minus junctions
    k8 = np.ones((3, 3), np.uint8)
    nb = ndi.convolve(skel.astype(np.uint8), k8, mode="constant") * skel
    branches = skel.copy()
    branches[nb >= 4] = False
    lab, nlab = ndi.label(branches, structure=np.ones((3, 3), int))

    g = rgb[:, :, 1]
    gbg = ndi.median_filter(g, size=31, mode="nearest")
    gc = g - gbg

    torts, b_int, b_len, b_w, b_wpx = [], [], [], [], []
    for i in range(1, nlab + 1):
        cy, cx = np.nonzero(lab == i)
        npx = len(cy)
        if npx < 3:
            continue
        med_g = float(np.median(gc[cy, cx]))
        b_int.append(med_g)
        b_len.append(npx)
        wpx = float(np.median(2.0 * dist[cy, cx]))
        b_wpx.append(wpx)
        b_w.append(wpx / dd if np.isfinite(dd) else np.nan)
        if npx >= MIN_BRANCH:
            chord = hull_chord(np.column_stack([cx, cy]).astype(float))
            if np.isfinite(chord) and chord > 1e-6:
                t = npx / chord
                if np.isfinite(t) and t < 20:
                    torts.append(t)
    out["n_branches"] = float(nlab)
    if torts:
        tt = np.asarray(torts)
        out["tort_median"] = float(np.median(tt))
        out["tort_p90"] = float(np.percentile(tt, 90))
        out["tort_top3_mean"] = float(np.sort(tt)[-min(3, len(tt)):].mean())
    else:
        out["tort_median"] = out["tort_p90"] = out["tort_top3_mean"] = np.nan

    # ---- A/V per BRANCH (not per pixel), length-weighted split of background-corrected green
    if len(b_int) >= 6:
        wts = np.asarray(b_len, dtype=float)
        vals = np.asarray(b_int, dtype=float)
        order = np.argsort(vals)
        cum = np.cumsum(wts[order])
        cut = int(np.searchsorted(cum, cum[-1] / 2.0))
        cut = min(max(cut, 0), len(order) - 1)
        thr = vals[order][cut]
        is_a = vals >= thr
        aw = np.asarray(b_w)
        awpx = np.asarray(b_wpx)
        alen, vlen = wts[is_a].sum(), wts[~is_a].sum()
        out["a_frac"] = float(alen / max(alen + vlen, 1))
        out["a_width_p90_px"] = float(np.percentile(awpx[is_a], 90)) if is_a.sum() >= 3 else np.nan
        out["v_width_p90_px"] = float(np.percentile(awpx[~is_a], 90)) if (~is_a).sum() >= 3 else np.nan
        out["a_width_p90_dd"] = float(np.percentile(aw[is_a], 90)) if is_a.sum() >= 3 else np.nan
        out["v_width_p90_dd"] = float(np.percentile(aw[~is_a], 90)) if (~is_a).sum() >= 3 else np.nan
        # the A/V width ratio is a ratio of two disc-normalised widths, so the disc cancels and
        # this feature is defined even when the disc is not.
        if np.isfinite(out["a_width_p90_px"]) and out["v_width_p90_px"] > 0:
            out["av_width_ratio_p90"] = out["a_width_p90_px"] / out["v_width_p90_px"]
        else:
            out["av_width_ratio_p90"] = np.nan
    else:
        out["a_frac"] = out["a_width_p90_px"] = out["v_width_p90_px"] = np.nan
        out["a_width_p90_dd"] = out["v_width_p90_dd"] = out["av_width_ratio_p90"] = np.nan
    out["a_tort_median"] = np.nan   # per-A/V tortuosity needs branch labels kept; deferred
    out["v_tort_median"] = np.nan
    # enforce the disc rule: nothing disc-relative survives without a validated disc
    if not disc_valid:
        for k in DISC_DEPENDENT:
            out[k] = np.nan
    return out


def worker(arg):
    image_path, mask_path, disc = arg
    try:
        rgb, msk, sc = load_pair(image_path, mask_path)
        d = dict(disc)
        for k in ("disc_cx", "disc_cy", "disc_dd_px"):
            if np.isfinite(d.get(k, np.nan)):
                d[k] = float(d[k]) * sc
        return measure(rgb, msk, d)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"{image_path}: {type(e).__name__}: {e}")
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=None)
    ap.add_argument("--disc", default="/Users/moniaz/niki/results/hvdro_validation/disc/disc_predictions_all.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    meta_path = Path(args.meta or (cfg["paths"]["features_dir"] / "biomarker_features.csv"))
    out_path = Path(args.out or (cfg["paths"]["features_dir"] / "biomarker_features_clinical_v3.csv"))
    partial = out_path.with_suffix(".partial.csv")

    meta = pd.read_csv(meta_path)
    if args.limit:
        meta = meta.head(args.limit)
    ddf = pd.read_csv(args.disc).set_index("image_path")
    print(f"[clinical_v3] rows={len(meta)} disc_predictions={len(ddf)} work={WORK}")

    done = set()
    if partial.exists():
        done = set(pd.read_csv(partial, usecols=["image_path"])["image_path"])
        print("[resume]", len(done))

    jobs = []
    for _, r in meta.iterrows():
        if r["image_path"] in done:
            continue
        d = ddf.loc[r["image_path"]].to_dict() if r["image_path"] in ddf.index else {}
        jobs.append((r["image_path"], r["mask_path"], d))
    print("[pending]", len(jobs))

    buf = []
    if jobs:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for i, feats in enumerate(pool.imap(worker, jobs, chunksize=8)):
                jp, jm, _ = jobs[i]
                row = {"image_path": jp, "mask_path": jm}
                src = meta.loc[meta["image_path"] == jp]
                if len(src):
                    for c in META:
                        if c in src.columns:
                            row[c] = src.iloc[0][c]
                row.update(feats)
                buf.append(row)
                if len(buf) >= 250:
                    append_csv_rows(partial, buf)
                    buf.clear()
                    print(f"  {i + 1}/{len(jobs)}", flush=True)
        if buf:
            append_csv_rows(partial, buf)

    df = pd.read_csv(partial)
    df.to_csv(out_path, index=False)
    partial.unlink(missing_ok=True)
    print(f"[done] {out_path} shape={df.shape}")
    print(df[["disc_method"]].value_counts().to_string() if "disc_method" in df else "")
    cols = [c for c in ("dd_over_min_side", "width_p90_dd", "width_ann_p90_dd",
                        "a_width_p90_dd", "v_width_p90_dd", "av_width_ratio_p90") if c in df]
    print(df[cols].describe().round(4).to_string())
    if "source" in df:
        print("\nby source:")
        print(df.groupby("source")[cols].median().round(4).to_string())


if __name__ == "__main__":
    main()
