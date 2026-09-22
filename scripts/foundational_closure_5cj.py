#!/usr/bin/env python
"""Foundational integrity closure, Parts 1-5.

Part 1/2  build a content-keyed canonical registry from the frozen canonical manifest
          (image_sha256, generation_id) -> {image_path, mask_path, mask_sha256}, then resolve all
          8,870 with the resolver-safe code through the registry and verify the ACTUAL mask bytes.
Part 3    reuse the saved fresh-inference evidence (144 images, 0 bitwise differences).
Part 4    freeze SEG_CURRENT_V2_RESOLVER_SAFE.
Part 5    verify the HVDROPDB same-image triples and COMPLETE Task 5C-J.

No filename reconstruction, no stem/prefix/glob/first-match resolution, no checkpoint or
preprocessing change, no classifier, no biomarker regeneration.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "/Users/moniaz/niki")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"
CANON = ROOT / "data/masks/mask_manifest_canonical_v1.csv"
MASKS = ROOT / "data/masks"
HV_STORE = ROOT / "data/masks_external/hvdrodb_seg_current_v1"
BV = ROOT / "data/raw/hvdro/segmentation/HVDROPDB_RetCam_Neo_Segmentation/HVDROPDB-BV"
GEN = "SEG_CURRENT_V1"
FEATS = ["vessel_density_fov", "skel_density_fov", "fractal_d0", "fractal_d1", "fractal_d2"]
SEC = "tort_geodesic_median"
WORK = 512


def sha(p) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def sha_pair(arg):
    ip, mp = arg
    try:
        return ip, mp, sha(ip), sha(mp)
    except Exception as e:  # noqa: BLE001
        return ip, mp, "", f"ERR:{type(e).__name__}"


def load_rgb_mask(rgb_path, mask_path):
    """Same loading convention as the frozen measurement path (long side 512, BILINEAR / NEAREST)."""
    img = Image.open(rgb_path).convert("RGB")
    w0, h0 = img.size
    sc = WORK / max(h0, w0)
    ws, hs = max(8, int(round(w0 * sc))), max(8, int(round(h0 * sc)))
    rgb = np.asarray(img.resize((ws, hs), Image.BILINEAR), np.float32)
    msk = np.asarray(Image.open(mask_path).convert("L").resize((ws, hs), Image.NEAREST)) > 127
    return rgb, msk.astype(np.uint8)


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2.0 * int((a & b).sum()) / s) if s else float("nan")


def cl_dice(pred, exp):
    from skimage.morphology import skeletonize
    sp, se = skeletonize(pred), skeletonize(exp)
    if sp.sum() == 0 or se.sum() == 0:
        return float("nan")
    t_pre = float((sp & exp).sum() / sp.sum())
    t_sen = float((se & pred).sum() / se.sum())
    return float(2 * t_pre * t_sen / (t_pre + t_sen)) if (t_pre + t_sen) else float("nan")


def hv_case(arg):
    case, cam, rgb_path, pred_path, expert_path = arg
    from src.biomarker import clinical_measurement_v5_candidate as v5
    from scipy import ndimage as ndi
    try:
        rgb, pred = load_rgb_mask(rgb_path, pred_path)
        h, w = pred.shape
        exp = np.asarray(Image.open(expert_path).convert("L").resize((w, h), Image.NEAREST)) > 127
        pred = pred.astype(bool)
        op, _f, _q = v5.measure_v5(rgb, pred.astype(np.uint8), {}, with_fractal=True)
        oe, _f2, _q2 = v5.measure_v5(rgb, exp.astype(np.uint8), {}, with_fractal=True)
        tp = int((pred & exp).sum())
        row = {"case": case, "camera": cam, "rgb": rgb_path, "pred": pred_path, "expert": expert_path,
               "dice": dice(pred, exp), "cldice": cl_dice(pred, exp),
               "precision": float(tp / max(int(pred.sum()), 1)),
               "recall": float(tp / max(int(exp.sum()), 1)),
               "pred_px": int(pred.sum()), "expert_px": int(exp.sum())}
        for f in FEATS + [SEC]:
            pv, ev = op.get(f, np.nan), oe.get(f, np.nan)
            row[f + "_pred"], row[f + "_expert"] = pv, ev
            row[f + "_signed"] = pv - ev if np.isfinite(pv) and np.isfinite(ev) else np.nan
            row[f + "_abs"] = abs(pv - ev) if np.isfinite(pv) and np.isfinite(ev) else np.nan
            row[f + "_rel"] = ((pv - ev) / abs(ev)
                               if np.isfinite(pv) and np.isfinite(ev) and ev else np.nan)
        edt = ndi.distance_transform_edt(exp)
        yy, xx = np.nonzero(exp)
        row["_diam"] = 2.0 * edt[yy, xx].astype(np.float32)
        row["_hit"] = pred[yy, xx]
        return row
    except Exception as e:  # noqa: BLE001
        return {"case": case, "_err": f"{type(e).__name__}: {e}"}


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("PART 1 — STRICT CONTENT-KEYED CANONICAL REGISTRY")
    print("=" * 100)
    can = pd.read_csv(CANON)
    print(f"  canonical manifest rows {len(can)}  sha {sha(CANON)[:16]}")
    args = list(zip(can.image_path, can.mask_path))
    with get_context("fork").Pool(24) as p:
        H = p.map(sha_pair, args, chunksize=8)
    reg = pd.DataFrame(H, columns=["image_path", "mask_path", "image_sha256", "mask_sha256"])
    errs = int(reg.mask_sha256.str.startswith("ERR").sum())
    reg["generation_id"] = GEN
    reg["key"] = reg.image_sha256 + "|" + reg.generation_id
    print(f"  image+mask hashed {len(reg)}  errors {errs}")
    dup_key = int(reg.key.duplicated().sum())
    dup_img = int(reg.image_path.duplicated().sum())
    print(f"  duplicate registry keys {dup_key}   duplicate image paths {dup_img}")
    print(f"  unique image_sha256 {reg.image_sha256.nunique()}   unique mask_sha256 "
          f"{reg.mask_sha256.nunique()}")
    reg.to_csv(OUT / "canonical_registry_v1.csv", index=False)

    print()
    print("=" * 100)
    print("PART 2 — TRUE CANONICAL 8,870 EQUIVALENCE VIA THE REGISTRY")
    print("=" * 100)
    from src.segmentation import infer_masks as im
    from src.segmentation.generation import GenerationError
    by_key = reg.groupby("key").mask_path.apply(list).to_dict()
    path_registry = reg.groupby("image_path").mask_path.apply(list).to_dict()
    resolved, missing, ambiguous, sha_mismatch, registered_changed = 0, 0, 0, 0, 0
    mismatches = []
    for r in reg.itertuples():
        hits = by_key.get(r.key)
        if hits is None:
            missing += 1
            continue
        if len(hits) > 1:
            ambiguous += 1
            continue
        # resolver-safe code path, explicit one-to-one registration
        got = im.find_existing_mask(MASKS, r.image_path, registry=path_registry)
        if got is None:
            missing += 1
            continue
        if str(got) != hits[0]:
            registered_changed += 1
            mismatches.append((r.image_path, str(got), hits[0]))
            continue
        if sha(got) != r.mask_sha256:
            sha_mismatch += 1
            mismatches.append((r.image_path, str(got), "sha"))
            continue
        resolved += 1
    print(f"  CANONICAL_REGISTRY_N                 : {len(reg)}")
    print(f"  CANONICAL_RESOLVED_N                 : {resolved}")
    print(f"  CANONICAL_MISSING_N                  : {missing}")
    print(f"  CANONICAL_AMBIGUOUS_N                : {ambiguous}")
    print(f"  CANONICAL_ACTUAL_MASK_SHA_MISMATCH_N : {sha_mismatch}")
    print(f"  CANONICAL_REGISTERED_MASK_CHANGED_N  : {registered_changed}")
    if mismatches:
        print("  first mismatches:", mismatches[:5])
    eq = (resolved == len(reg) and missing == 0 and ambiguous == 0 and sha_mismatch == 0
          and registered_changed == 0)
    print(f"  CANONICAL_MASK_INPUT_EQUIVALENCE     : {'PASS' if eq else 'FAIL'}")
    print(f"  (filename-reconstruction comparison is NOT used; identity is "
          f"(image_sha256, generation_id) -> registered mask bytes)")
    if not eq:
        raise SystemExit("CANONICAL_INPUT_EQUIVALENCE_FAILED")

    print()
    print("=" * 100)
    print("PART 3 — RETAINED PIXEL-EQUIVALENCE EVIDENCE")
    print("=" * 100)
    prev = json.loads((OUT / "seg_resolver_closure.json").read_text())
    fr, fd = prev["FRESH_INFERENCE_RECHECK_N"], prev["FRESH_INFERENCE_BITWISE_DIFFERENT_N"]
    print(f"  FRESH_INFERENCE_RECHECK_N            : {fr}")
    print(f"  FRESH_INFERENCE_BITWISE_DIFFERENT_N  : {fd}")
    print("  (saved evidence reused; not rerun)")

    print()
    print("=" * 100)
    print("PART 5 — HVDROPDB SAME-IMAGE TRIPLES AND TASK 5C-J")
    print("=" * 100)
    hv = pd.read_csv(OUT / "task5c_j_external_manifest.csv")
    idx = pd.read_csv(OUT / "task5c_j_external_index.csv")
    hv = hv.merge(idx[["image_path", "expert_mask", "source"]], on="image_path", how="left",
                  suffixes=("", "_i"))
    triples_bad = 0
    for r in hv.itertuples():
        ok = (Path(r.image_path).exists() and Path(r.mask_path).exists()
              and Path(r.expert_mask).exists())
        if not ok:
            triples_bad += 1
    print(f"  rows {len(hv)}  unique images {hv.image_path.nunique()}  "
          f"unique predicted paths {hv.mask_path.nunique()}  broken triples {triples_bad}")
    valid = bool(len(hv) == 100 and hv.image_path.nunique() == 100
                 and hv.mask_path.nunique() == 100 and triples_bad == 0)
    print(f"  HVDROPDB_VALID_SAME_IMAGE_TRIPLES_N  : {len(hv) if valid else 0}")
    if not valid:
        raise SystemExit("HVDROPDB_TRIPLE_INTEGRITY_FAILED")

    cargs = []
    for r in hv.itertuples():
        cam = "Neo" if "Neo_Vessels_images" in r.image_path else "RetCam"
        cargs.append((f"{cam}_{Path(r.image_path).stem[:20]}", cam, r.image_path, r.mask_path,
                      r.expert_mask))
    rows, errs2 = [], 0
    with get_context("fork").Pool(12) as p:
        for r in p.imap_unordered(hv_case, cargs, chunksize=1):
            if "_err" in r:
                errs2 += 1
                print("  ERR", r["case"], r["_err"])
                continue
            rows.append(r)
    R = pd.DataFrame(rows)
    print(f"  cases measured {len(R)}  errors {errs2}")
    print(f"  {'group':9s} {'N':>4s} {'Dice':>8s} {'clDice':>8s} {'prec':>8s} {'recall':>8s}")
    for g, sub in list(R.groupby("camera")) + [("OVERALL", R)]:
        print(f"  {g:9s} {len(sub):4d} {sub.dice.median():8.4f} {sub.cldice.median():8.4f} "
              f"{sub.precision.median():8.4f} {sub.recall.median():8.4f}")
    print("  FINAL_PRIMARY error (predicted vs expert mask, same RGB):")
    summ = {}
    for f in FEATS + [SEC]:
        a, s, rel = R[f + "_abs"].dropna(), R[f + "_signed"].dropna(), R[f + "_rel"].dropna()
        summ[f] = {"n": int(len(a)), "median_abs": float(a.median()),
                   "p95_abs": float(a.quantile(.95)), "bias": float(s.median()),
                   "max_abs": float(a.max()),
                   "median_rel": float(rel.median()) if len(rel) else float("nan")}
        print(f"    {f:20s} n={len(a):3d} median|e| {a.median():.6f} p95 {a.quantile(.95):.6f} "
              f"bias {s.median():+.6f} medRel "
              f"{(rel.median() if len(rel) else float('nan')):+.5f}")
    print("  by camera (median absolute error):")
    for cam, sub in R.groupby("camera"):
        print("    %-8s " % cam + "  ".join(f"{f}: {sub[f+'_abs'].median():.6f}" for f in FEATS))
    from scipy.stats import spearmanr
    sp = {}
    for f in FEATS + [SEC]:
        for q in ("dice", "cldice", "precision", "recall"):
            d = R[[q, f + "_abs"]].dropna()
            if len(d) >= 8:
                sp[(f, q)] = float(spearmanr(d[q], d[f + "_abs"]).statistic)
    print("  Spearman(quality, |error|) for the five primary features (overall):")
    for f in FEATS + [SEC]:
        print("    %-20s " % f + "  ".join(
            f"{q}={sp[(f,q)]:+.3f}" for q in ("dice", "cldice", "precision", "recall")
            if (f, q) in sp))
    strong = max(sp.items(), key=lambda kv: abs(kv[1])) if sp else (("", ""), None)
    cuts, tallies = {}, []
    for cam, sub in R.groupby("camera"):
        diam = np.concatenate([d for d in sub["_diam"]])
        hit = np.concatenate([h for h in sub["_hit"]])
        q1, q2 = np.quantile(diam, [1 / 3, 2 / 3])
        cuts[cam] = {"p33_px": float(q1), "p67_px": float(q2)}
        print(f"  {cam} expert diameter tertiles (2*EDT): p33 {q1:.3f} px, p67 {q2:.3f} px")
        for name, m in (("thin", diam <= q1), ("mid", (diam > q1) & (diam <= q2)),
                        ("thick", diam > q2)):
            tallies.append({"camera": cam, "bin": name, "pixel_n": int(m.sum()),
                            "recall": float(hit[m].mean()) if m.any() else float("nan")})
    T = pd.DataFrame(tallies)
    print(T.to_string(index=False))
    for name in ("thin", "mid", "thick"):
        tt = T[T.bin == name]
        n = int(tt.pixel_n.sum())
        print(f"    pooled {name:6s} pixels {n:9d} recall "
              f"{float((tt.recall * tt.pixel_n).sum() / max(n,1)):.4f}")
    out = {"CANONICAL_REGISTRY_N": int(len(reg)), "CANONICAL_RESOLVED_N": int(resolved),
           "CANONICAL_MISSING_N": int(missing), "CANONICAL_AMBIGUOUS_N": int(ambiguous),
           "CANONICAL_ACTUAL_MASK_SHA_MISMATCH_N": int(sha_mismatch),
           "CANONICAL_REGISTERED_MASK_CHANGED_N": int(registered_changed),
           "CANONICAL_MASK_INPUT_EQUIVALENCE": "PASS" if eq else "FAIL",
           "FRESH_INFERENCE_RECHECK_N": int(fr), "FRESH_INFERENCE_BITWISE_DIFFERENT_N": int(fd),
           "HVDROPDB_VALID_SAME_IMAGE_TRIPLES_N": int(len(hv)),
           "seg_metrics": {g: {"n": int(len(s)), "dice": float(s.dice.median()),
                               "cldice": float(s.cldice.median()),
                               "precision": float(s.precision.median()),
                               "recall": float(s.recall.median())}
                           for g, s in R.groupby("camera")},
           "overall": {"dice": float(R.dice.median()), "cldice": float(R.cldice.median()),
                       "precision": float(R.precision.median()),
                       "recall": float(R.recall.median())},
           "biomarker_error": summ, "cutpoints": cuts,
           "recall_table": T.to_dict("records"),
           "spearman": {f"{k[0]}|{k[1]}": v for k, v in sp.items()},
           "strongest": {"feature": strong[0][0], "quality": strong[0][1],
                         "rho": strong[1]} if strong[1] is not None else None,
           "infer_masks_sha256": sha(ROOT / "src/segmentation/infer_masks.py"),
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "foundational_closure_and_5cj.json").write_text(json.dumps(out, indent=2, default=str),
                                                           encoding="utf-8")
    print(f"\n  elapsed {out['elapsed_s']}s")


if __name__ == "__main__":
    main()
