#!/usr/bin/env python
"""Task 5B-H sections H/I/J: mechanism of the FOV failures, and private QC montages.

Reads the condition table produced by `scripts/task5b_h_fov_stress.py`, picks the strongest
failures per family plus a median case per geometry, and re-measures those exact cases in-process
to expose the mechanism:

  * `otsu_threshold` before/after, and its ratio to the scale-invariant expectation for brightness;
  * whether the detected FOV is the BORDER ITSELF: the fraction of the padded-frame FOV that lies
    inside the added border region (`fov_frac_inside_pad`);
  * raw vs post-morphology component counts;
  * production coverage, validity and `vessel_density_fov` for the same vessel mask.

It also writes the QC montages: baseline image with baseline FOV contour in green, perturbed image
with perturbed FOV contour in red. No disease label is read or shown.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.task5b_n2 import load  # noqa: E402
from scripts.task5b_h_fov_stress import (  # noqa: E402
    BRIGHTNESS, apply_border, apply_brightness, diag_components, pad_specs,
)
from src.biomarker.clinical_measurement_v1 import retinal_fov  # noqa: E402

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"


def contour(m):
    return m & ~ndi.binary_erosion(m, np.ones((3, 3), bool))


def overlay(rgb, masks_colors):
    img = np.clip(rgb, 0, 255).astype(np.uint8).copy()
    for m, col in masks_colors:
        c = contour(m)
        img[c] = col
    return img


def to_png(arr, h):
    im = Image.fromarray(arr)
    w = max(1, int(round(im.width * h / im.height)))
    return im.resize((w, h), Image.BILINEAR)


def montage(base_rgb, base_fov, pert_rgb, pert_fov, path, title_px=0):
    h = base_rgb.shape[0]
    a = to_png(overlay(base_rgb, [(base_fov, (0, 255, 0))]), h)
    b = to_png(overlay(pert_rgb, [(pert_fov, (255, 0, 0))]), h)
    canvas = Image.new("RGB", (a.width + b.width + 4, h), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + 4, 0))
    canvas.save(path)


def main() -> None:
    C = pd.read_csv(OUT / "task5b_h_conditions.csv")
    B = pd.read_csv(OUT / "task5b_h_baseline.csv")
    Bm = B.set_index("image_path")
    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v1.csv",
                       usecols=["image_path", "mask_path"])
    mpath = dict(zip(meta.image_path, meta.mask_path))
    print("=" * 100)
    print("worst cases and their baseline state")
    print("=" * 100)
    for r in C.sort_values("fov_dice").head(12).itertuples():
        b = Bm.loc[r.image_path]
        print(f"  {r.image_path.split('/')[-1][:44]:44s} {r.condition:22s} dice={r.fov_dice:.4f} "
              f"base_cov={b.coverage_as_production:.4f} base_valid={b.fov_valid} "
              f"base_fov_px={b.fov_px_padded:6d} base_dens={b.vessel_density_fov:.4f} "
              f"dens_rel={r.density_rel_delta:+.4f}")

    cases = []
    for fam in ["brightness", "uniform_black", "dark_gray", "overwrite_black", "irregular_frame"]:
        g = C[C.family == fam].sort_values("fov_dice")
        for r in g.head(2).itertuples():
            cases.append((r.image_path, r.condition, r.family))
    for gm, g in C[C.family == "uniform_black"].groupby("geom"):
        m = g.iloc[(g.fov_dice - g.fov_dice.median()).abs().argsort()[:1]]
        for r in m.itertuples():
            cases.append((r.image_path, r.condition, f"median_{gm}"))
    stable = C[C.fov_dice > 0.99999].iloc[0]
    cases.append((stable.image_path, stable.condition, "stable"))
    Vv = C.merge(B[["image_path", "fov_valid"]].rename(columns={"fov_valid": "bv"}),
                 on="image_path", how="left")
    for r in Vv[Vv.bv].sort_values("fov_dice").head(4).itertuples():
        cases.append((r.image_path, r.condition, "worst_valid"))
    seen, uniq = set(), []
    for c in cases:
        if (c[0], c[1]) not in seen:
            seen.add((c[0], c[1]))
            uniq.append(c)
    cases = uniq

    print()
    print("=" * 100)
    print("H/I. MECHANISM TABLE (re-measured in-process, frozen retinal_fov)")
    print("=" * 100)
    hdr = (f"  {'case':30s} {'condition':22s} {'thr_b':>7s} {'thr_p':>7s} {'thrRatio':>8s} "
           f"{'fovPx_pad':>9s} {'crop':>7s} {'inPad':>6s} {'covP':>6s} {'nRaw':>5s} {'nCln':>5s} "
           f"{'bc':>5s} {'valid':>5s} {'dens_b':>7s} {'dens_p':>7s} {'reason':>28s}")
    print(hdr)
    rows = []
    for image_path, cond, tag in cases:
        rgb, msk = load(image_path, mpath[image_path])
        h0, w0 = msk.shape
        bfov, bqc = retinal_fov(rgb[:, :, 1])
        bthr = float(threshold_otsu(np.nan_to_num(rgb[:, :, 1], nan=0.0)))
        bdens = float(msk[bfov].sum() / max(int(bfov.sum()), 1))

        if cond.startswith("x"):
            c = float(cond[1:])
            prgb, pmsk, pad = apply_brightness(rgb, msk, c)
            scale = c
        else:
            spec = [s for s in pad_specs(h0, w0) if s[1] == cond][0]
            prgb, pmsk, pad = apply_border(rgb, msk, spec)
            scale = 1.0
        pfov, pqc = retinal_fov(prgb[:, :, 1])
        pthr = float(threshold_otsu(np.nan_to_num(prgb[:, :, 1], nan=0.0)))
        fp_pad = int(pfov.sum())
        t, b_, l, r_ = pad
        H, W = pfov.shape
        padreg = np.zeros_like(pfov)
        if t:
            padreg[:t, :] = True
        if b_:
            padreg[H - b_:, :] = True
        if l:
            padreg[:, :l] = True
        if r_:
            padreg[:, W - r_:] = True
        inpad = float(pfov[padreg].sum() / fp_pad) if fp_pad and padreg.any() else 0.0
        pdens = float(pmsk[pfov].sum() / fp_pad) if fp_pad else float("nan")
        print(f"  {image_path.split('/')[-1][:30]:30s} {cond:22s} {bthr:7.2f} {pthr:7.2f} "
              f"{pthr / (bthr * scale) if bthr else float('nan'):8.4f} {fp_pad:9d} "
              f"{int(pfov[t:t + h0, l:l + w0].sum()):7d} {inpad:6.3f} "
              f"{float(pfov.mean()):6.3f} {int(pqc.get('fov_n_components', -1)):5d} "
              f"{diag_components(prgb[:, :, 1]):5d} {float(pqc.get('fov_border_contact', 0)):5.3f} "
              f"{str(bool(pqc.get('fov_valid'))):>5s} {bdens:7.4f} {pdens:7.4f} "
              f"{str(pqc.get('fov_failure_reason', '')):>28s}")
        rows.append({"tag": tag, "image_path": image_path, "condition": cond,
                     "thr_base": bthr, "thr_pert": pthr, "thr_ratio": pthr / (bthr * scale) if bthr else np.nan,
                     "fov_px_padded": fp_pad, "fov_px_cropped": int(pfov[t:t + h0, l:l + w0].sum()),
                     "fov_frac_inside_pad": inpad, "coverage_prod": float(pfov.mean()),
                     "n_components_raw": int(pqc.get("fov_n_components", -1)),
                     "n_components_clean": diag_components(prgb[:, :, 1]),
                     "border_contact": float(pqc.get("fov_border_contact", 0)),
                     "fov_valid": bool(pqc.get("fov_valid")),
                     "failure_reason": str(pqc.get("fov_failure_reason", "")),
                     "density_base": bdens, "density_pert": pdens,
                     "base_coverage": float(bfov.mean()), "base_valid": bool(bqc.get("fov_valid"))})
        montage(rgb, bfov, prgb, pfov,
                OUT / f"task5b_h_qc_{tag[:18]}_{cond[:16]}__{image_path.split('/')[-1][:12]}.png".replace("/", "_"))
    pd.DataFrame(rows).to_csv(OUT / "task5b_h_mechanism.csv", index=False)
    print()
    print(f"  montages written to {OUT}/task5b_h_qc_*.png")
    print(f"  mechanism rows : {len(rows)}")


def msk_path_of(Bm, image_path):
    return Bm.loc[image_path, "mask_path"]


if __name__ == "__main__":
    main()
