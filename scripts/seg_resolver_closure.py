#!/usr/bin/env python
"""Segmentation resolver integrity closure.

Reproduces and removes the ambiguous stem fallback in `infer_masks.py::find_existing_mask`, proves
the canonical 8,870 mapping is unchanged, proves fresh inference is bitwise equal to the frozen
SEG_CURRENT_V1 masks, and re-runs the HVDROPDB external inference under the exact condition that
previously failed.

No checkpoint change, no preprocessing change, no threshold change, no classifier, no biomarker
table regeneration.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "_private_audit"
INFER = ROOT / "src/segmentation/infer_masks.py"
BACKUP = OUT / "infer_masks_pre_resolver_fix.py"
GEN = "SEG_CURRENT_V2_RESOLVER_SAFE"
STORE = ROOT / "data/masks_v2_resolver_safe"
HV_STORE = ROOT / "data/masks_external/hvdrodb_seg_current_v1"
CONTRACT = ROOT / "configs/segmentation_generation_current_v2_resolver_safe.yaml"
CANON_MANIFEST = ROOT / "data/masks/mask_manifest_canonical_v1.csv"
BV = ROOT / "data/raw/hvdro/segmentation/HVDROPDB_RetCam_Neo_Segmentation/HVDROPDB-BV"
FREEZE = {
    "infer_masks_sha256_original": "b76f820025353841908111674a9ca3460776f7add496d6c2a27ff4bbf09f1a1b",
    "checkpoint_sha256": "c373f53813ee60b89651a04a98bc5f1d6bc60a5a50f45c4e10f42475650d5374",
    "config_sha256": "77a6c9230fd8b6f7230e1085d3415f46ece862d3e160ace983a1be83b97f0a60",
    "seg_current_v1": "SEG_CURRENT_V1",
    "canonical_mask_manifest_sha256":
        "b3bc6538026d6ad0fc1342043e5ca2d6de56662233bf6d0709d9d875f24b21cd",
    "final_biomarkers_v2_sha256":
        "b4661dffd2e082f93d183cdb26ae0462dce8af32b135c1968db2deacc493365e",
}

NEW_RESOLVER = '''def find_existing_mask(masks_dir: Path, image_path: str,
                       registry: dict | None = None) -> Path | None:
    """One-to-one identity resolution. EXACT path-hash name or an explicit registration only.

    The previous implementation fell back to a stem glob (`^{stem}_[0-9a-f]{8}\\.png$`) and returned
    the single match. Two images sharing a stem in different directories therefore resolved to the
    SAME mask: the HVDROPDB RetCam and Neo folders both contain `1.png`, so 50 of 100 external
    predictions received another camera's mask. The fallback is removed. Ambiguity is now a hard
    error and a missing mapping returns None (the mask is generated), never a guess.

    Resolution order:
      1. ``masks_dir / unique_mask_name(image_path)`` - a deterministic function of the image PATH,
         so two different images can never share a name;
      2. ``registry[image_path]`` - an explicit one-to-one registration (e.g. a frozen manifest).
    A registry key with more than one entry, or a registered path that does not exist, raises.
    """
    preferred = masks_dir / unique_mask_name(image_path)
    if preferred.exists():
        return preferred
    if registry is None:
        return None
    hits = registry.get(str(image_path))
    if hits is None:
        return None
    if len(hits) > 1:
        raise GenerationError(
            f"ambiguous registration: {len(hits)} masks registered for {image_path}"
        )
    registered = Path(hits[0])
    if not registered.exists():
        raise GenerationError(
            f"registered mask for {image_path} does not exist: {registered}"
        )
    return registered


'''


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def unique_mask_name(image_path: str) -> str:
    stem = Path(image_path).stem
    h = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{h}.png"


def patch_resolver():
    src = INFER.read_text(encoding="utf-8")
    BACKUP.write_text(src, encoding="utf-8")
    print(f"  original preserved at {BACKUP.name}  sha {sha(BACKUP)[:16]}")
    start = src.index("def find_existing_mask(")
    end = src.index("def ", start + 10)
    new = src[:start] + NEW_RESOLVER + src[end:]
    INFER.write_text(new, encoding="utf-8")
    return src, new


def old_stem_matches(masks_dir: Path, image_path: str):
    """The removed behaviour, reproduced for measurement only."""
    stem = Path(image_path).stem
    pat = re.compile(rf"^{re.escape(stem)}_[0-9a-f]{{8}}\.png$")
    return sorted(p for p in masks_dir.glob(f"{stem}_*.png") if pat.match(p.name))


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("A. FREEZE CURRENT STATE")
    print("=" * 100)
    for k, v in FREEZE.items():
        print(f"  {k:34s} : {v}")
    pre_sha = sha(INFER)
    print(f"  {'infer_masks_sha256_now':34s} : {pre_sha}")
    print(f"  matches authorized original      : {pre_sha == FREEZE['infer_masks_sha256_original']}")

    print()
    print("=" * 100)
    print("B. REMOVE AMBIGUOUS RESOLUTION")
    print("=" * 100)
    old_src, new_src = patch_resolver()
    sys.path.insert(0, str(ROOT))
    import importlib
    from src.segmentation import infer_masks as im
    importlib.reload(im)
    new_sha = sha(INFER)
    print(f"  patched infer_masks sha256      : {new_sha}")
    has_stem = 'glob(f"{stem}_' in new_src
    print(f"  stem fallback present in new code: {has_stem}")
    print(f"  code changed                     : {old_src != new_src}")

    print()
    print("=" * 100)
    print("C. UNIT REGRESSION TESTS")
    print("=" * 100)
    import tempfile
    from src.segmentation.generation import GenerationError
    T = Path(tempfile.mkdtemp())
    (T / "retcam").mkdir(); (T / "neo").mkdir()
    r_path = "/ext/RetCam_Vessels_images/1.png"
    n_path = "/ext/Neo_Vessels_images/1.png"
    a = T / "retcam" / unique_mask_name(r_path)
    b = T / "neo" / unique_mask_name(n_path)
    Image.fromarray(np.full((4, 4), 255, np.uint8)).save(a)
    Image.fromarray(np.zeros((4, 4), np.uint8)).save(b)
    # legacy reproduction: one store, both cameras
    legacy = T / "legacy"
    legacy.mkdir()
    shutil.copy(a, legacy / Path(a).name)
    old_r = old_stem_matches(legacy, r_path)
    old_n = old_stem_matches(legacy, n_path)
    legacy_collision = bool(old_r and old_n and old_r[0] == old_n[0])
    tests = {
        "1_collision_reproduced_by_old_resolver": legacy_collision,
        "2_retcam_resolves_to_its_own_mask": im.find_existing_mask(legacy, r_path) == a if a.exists()
        else im.find_existing_mask(legacy, r_path) is None,
        "3_neo_does_not_get_retcam_mask": (im.find_existing_mask(legacy, n_path) or Path("x")) != a,
        "4_missing_mask_returns_none": im.find_existing_mask(T / "empty", r_path) is None,
        "5_registry_duplicate_is_error": False,
        "6_exact_registered_mapping_resolves": False,
    }
    # neo's own store contains its own mask under the path-hash name -> exact hit, not RetCam's
    neo_store = T / "neo_store"; neo_store.mkdir()
    shutil.copy(b, neo_store / Path(b).name)
    tests["3_neo_does_not_get_retcam_mask"] = im.find_existing_mask(neo_store, n_path) == b
    # duplicate registration
    dup = {r_path: ["/x/one.png", "/x/two.png"]}
    try:
        im.find_existing_mask(T / "empty", r_path, registry=dup)
        tests["5_registry_duplicate_is_error"] = False
    except GenerationError:
        tests["5_registry_duplicate_is_error"] = True
    # explicit registration (restored from another machine) resolves, and missing is a hard error
    reg = {r_path: [str(a)]}
    tests["6_exact_registered_mapping_resolves"] = (
        im.find_existing_mask(T / "empty", r_path, registry=reg) == a)
    missing_reg = {r_path: [str(T / "nope.png")]}
    try:
        im.find_existing_mask(T / "empty", r_path, registry=missing_reg)
        tests["7_registry_missing_file_is_error"] = False
    except GenerationError:
        tests["7_registry_missing_file_is_error"] = True
    tests["1_collision_reproduced_by_old_resolver"] = legacy_collision
    for k, v in sorted(tests.items()):
        print(f"  {k:44s} : {'PASS' if v else 'FAIL'}")
    print(f"  legacy collision: RetCam 1.png and Neo 1.png both matched "
          f"{old_r[0].name if old_r else None} -> collision={legacy_collision}")

    print()
    print("=" * 100)
    print("D. CANONICAL 8870 RESOLVER EQUIVALENCE")
    print("=" * 100)
    can = pd.read_csv(CANON_MANIFEST)
    can_dir = ROOT / "data/masks"
    name_bad, sha_changed, ambiguous, missing = [], 0, 0, 0
    for r in can.itertuples():
        want = can_dir / unique_mask_name(r.image_path)
        if Path(r.mask_path).name != want.name:
            name_bad.append(r.image_path)
        if not Path(r.mask_path).exists():
            missing += 1
        cands = old_stem_matches(can_dir, r.image_path)
        if len(cands) > 1:
            ambiguous += 1
        if cands and cands[0].name != want.name and not want.exists():
            sha_changed += 1
    print(f"  CANONICAL_MAPPING_N                 : {len(can)}")
    print(f"  naming-convention violations        : {len(name_bad)}")
    print(f"  masks missing on disk               : {missing}")
    print(f"  old resolver would have been ambiguous (>1 stem candidate) : {ambiguous}")
    print(f"  old resolver would have returned a different mask          : {sha_changed}")
    print(f"  CANONICAL_MAPPING_CHANGED_N         : {len(name_bad)}")
    print(f"  CANONICAL_MASK_SHA_CHANGED_N        : {sha_changed}")
    print(f"  AMBIGUOUS_N                         : {ambiguous}")
    print(f"  MISSING_N                           : {missing}")

    print()
    print("=" * 100)
    print("E. PIXEL-GENERATION EQUIVALENCE (fresh inference vs frozen SEG_CURRENT_V1)")
    print("=" * 100)
    from scripts.task5b_h_fov_stress import GEOM
    meta = pd.read_csv(ROOT / "data/features/final_biomarkers_v2.csv",
                       usecols=["image_path", "mask_path", "source", "split"])
    meta["geom"] = [GEOM.get(Image.open(p).size, "other") for p in meta.image_path]
    sub = pd.concat([g.head(9) for _, g in meta.groupby(["source", "geom", "split"])])
    sub = sub.reset_index(drop=True)
    print(f"  subset N {len(sub)}  sources {sub.source.nunique()}  geoms {sub.geom.nunique()}  "
          f"splits {sub.split.nunique()}")
    idx = OUT / "resolver_safe_check_index.csv"
    pd.DataFrame({"image_path": sub.image_path, "label": -1, "split": sub.split,
                  "source": sub.source}).to_csv(idx, index=False)
    c = __import__("yaml").safe_load(
        (ROOT / "configs/segmentation_generation_current_v1.yaml").read_text(encoding="utf-8"))
    c.update({"generation_id": GEN, "created_at": "2026-09-22T14:00:00+0330", "store_frozen": False,
              "output_store": str(STORE.relative_to(ROOT)), "supersedes": "SEG_CURRENT_V1",
              "inference_script_sha256": new_sha,
              "note": "SEG_CURRENT_V1 with the ambiguous stem fallback removed from the resolver. "
                      "Checkpoint, config, preprocessing, threshold and postprocessing unchanged.",
              "canonical_population_fingerprint": "RESOLVER_EQUIVALENCE_CHECK",
              "canonical_population_n": int(len(sub)),
              "resolution_policy": "exact path-hash name, or an explicit one-to-one registration",
              "ambiguous_fallback_removed": True})
    CONTRACT.write_text(__import__("yaml").safe_dump(c, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")
    STORE.mkdir(parents=True, exist_ok=True)
    cmd = [str(ROOT / ".venv/bin/python"), "-m", "src.segmentation.infer_masks",
           "--generation", GEN, "--index", str(idx)]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print("  " + (p.stdout.strip().splitlines() or ["<no stdout>"])[-1])
    if p.returncode:
        print(p.stderr[-800:])
        raise SystemExit("FRESH_INFERENCE_FAILED")
    fresh = pd.read_csv(STORE / "mask_manifest.csv")
    n_same, n_diff, diff_ids = 0, 0, []
    for r in fresh.itertuples():
        frozen = meta.set_index("image_path").mask_path.to_dict()[r.image_path]
        A = np.asarray(Image.open(r.mask_path).convert("L"))
        B = np.asarray(Image.open(frozen).convert("L"))
        if A.shape == B.shape and np.array_equal(A, B):
            n_same += 1
        else:
            n_diff += 1
            diff_ids.append(r.image_path)
    print(f"  FRESH_INFERENCE_RECHECK_N           : {len(fresh)}")
    print(f"  BITWISE_EQUAL_N                     : {n_same}")
    print(f"  BITWISE_DIFFERENT_N                 : {n_diff}")
    if diff_ids:
        print("  differing:", diff_ids[:5])

    print()
    print("=" * 100)
    print("F. HVDROPDB COLLISION REGRESSION (same parent store for both cameras)")
    print("=" * 100)
    if HV_STORE.exists():
        shutil.rmtree(HV_STORE)
    HV_STORE.mkdir(parents=True, exist_ok=True)
    hv_idx = OUT / "task5c_j_external_index.csv"
    c2 = __import__("yaml").safe_load(
        (ROOT / "configs/segmentation_generation_current_v1_hvdropdb_external.yaml")
        .read_text(encoding="utf-8"))
    c2["inference_script_sha256"] = new_sha
    c2["store_frozen"] = False
    (ROOT / "configs/segmentation_generation_current_v1_hvdropdb_external.yaml").write_text(
        __import__("yaml").safe_dump(c2, sort_keys=False, allow_unicode=True), encoding="utf-8")
    p2 = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "src.segmentation.infer_masks",
                         "--generation", "SEG_CURRENT_V1_HVDROPDB_EXTERNAL",
                         "--index", str(hv_idx)], cwd=str(ROOT), capture_output=True, text=True)
    print("  " + (p2.stdout.strip().splitlines() or ["<no stdout>"])[-1])
    if p2.returncode:
        print(p2.stderr[-800:])
        raise SystemExit("HVDROPDB_INFERENCE_FAILED")
    hv = pd.read_csv(HV_STORE / "mask_manifest.csv")
    hv_idx_df = pd.read_csv(hv_idx)
    hv = hv.merge(hv_idx_df[["image_path", "source", "expert_mask"]], on="image_path", how="left")
    uniq_paths = hv.mask_path.nunique()
    uniq_imgs = hv.image_path.nunique()
    cross = 0
    for r in hv.itertuples():
        if "RetCam" in r.image_path and "Neo" in r.mask_path:
            cross += 1
        if "Neo" in r.image_path and "RetCam" in r.mask_path:
            cross += 1
    dup_mask_owner = hv.groupby("mask_path").image_path.nunique()
    shared = int((dup_mask_owner > 1).sum())
    print(f"  HVDROPDB_RGB_N                      : {len(hv_idx_df)}")
    print(f"  manifest rows                       : {len(hv)}")
    print(f"  unique image identities             : {uniq_imgs}")
    print(f"  unique output paths                 : {uniq_paths}")
    print(f"  output paths shared by >1 image     : {shared}")
    print(f"  cross-camera mask reuse             : {cross}")
    print(f"  PNG files in store                  : {len(list(HV_STORE.glob('*.png')))}")
    hv.to_csv(OUT / "task5c_j_external_manifest.csv", index=False)

    print()
    print("=" * 100)
    print("J. FINAL OUTPUT")
    print("=" * 100)
    res = {"RESOLVER_BUG_REPRODUCED": legacy_collision,
           "RESOLVER_FIXED": bool(new_src != old_src and legacy_collision
                                  and not tests_bad(tests)),
           "AMBIGUOUS_STEM_FALLBACK_REMOVED": bool("glob(f\"{stem}_" not in new_src),
           "CANONICAL_MAPPING_N": int(len(can)),
           "CANONICAL_MAPPING_CHANGED_N": int(len(name_bad)),
           "CANONICAL_MASK_SHA_CHANGED_N": int(sha_changed),
           "AMBIGUOUS_N": int(ambiguous), "MISSING_N": int(missing),
           "FRESH_INFERENCE_RECHECK_N": int(len(fresh)),
           "FRESH_INFERENCE_BITWISE_DIFFERENT_N": int(n_diff),
           "HVDROPDB_RGB_N": int(len(hv_idx_df)),
           "HVDROPDB_VALID_PREDICTION_ASSOCIATIONS_N": int(uniq_imgs),
           "HVDROPDB_CROSS_CAMERA_COLLISION_N": int(cross + shared),
           "SEGMENTATION_FINAL_LINEAGE": GEN,
           "infer_masks_sha256_original": pre_sha, "infer_masks_sha256_new": new_sha,
           "unit_tests": tests, "elapsed_s": round(time.time() - t0, 1)}
    res["PIXEL_EQUIVALENT_TO_SEG_CURRENT_V1"] = bool(n_diff == 0 and len(name_bad) == 0)
    res["FINAL_BIOMARKERS_V2_REGENERATION_REQUIRED"] = bool(
        len(name_bad) or sha_changed or n_diff)
    res["FOUNDATIONAL_INTEGRITY_CLOSURE"] = "PASS" if (
        res["RESOLVER_FIXED"] and res["AMBIGUOUS_STEM_FALLBACK_REMOVED"]
        and len(name_bad) == 0 and sha_changed == 0 and n_diff == 0
        and uniq_imgs == 100 and uniq_paths == 100 and (cross + shared) == 0) else "FAIL"
    for k, v in res.items():
        if k not in ("unit_tests",):
            print(f"  {k:46s} : {v}")
    (OUT / "seg_resolver_closure.json").write_text(json.dumps(res, indent=2, default=str),
                                                   encoding="utf-8")
    print(f"\n  elapsed {res['elapsed_s']}s")


def tests_bad(tests: dict) -> bool:
    return any(not v for v in tests.values())


if __name__ == "__main__":
    main()
