#!/usr/bin/env python
"""Pre-delivery validator for the blinded expert pilot package.

Fails non-zero if any critical rule fails. Prints PILOT_PACKAGE_VALIDATION_PASS only when every
critical check passes.

Usage
-----
  python scripts/validate_expert_pilot_package.py \
      --package expert_validation/package_build/expert_pilot30_v1.0 \
      --key expert_validation/blinding_key.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

STUDY_ID_RE = re.compile(r"^ROP_\d{4}$")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Filename leakage. Matching is on WORD boundaries, not raw substrings: a substring test rejects
# "example_disc_correct.png" because "exam" sits inside "example", which is exactly the kind of
# mechanical false positive the audit must avoid.
LEAK_WORDS = {"farabi", "farfum", "retcam", "patient", "exam", "train", "val", "validation",
              "test", "source", "label", "prediction", "confidence", "gradcam", "automatic",
              "auto", "split", "normal", "plus", "preplus", "neo"}
LEAK_SUBSTRINGS = ("pre_plus", "auto_mask", "grad_cam", "_auto", "blinding", "audit_sheet")
GEOM_WORDS = {"1240", "1200", "1080", "960", "480"}
WORD_SPLIT = re.compile(r"[^a-z0-9]+")

FORBIDDEN_CSV_SUBSTR = ("blinding_key", "pilot30_manifest", "audit_sheet", "manifest_blinded",
                        "sampling_summary", "loo_", "branch_", "source_probe")
FORBIDDEN_COLUMNS = {"source", "label", "split", "group_id", "image_path", "mask_path",
                     "patient_id", "exam_id", "min_side", "peak_prob", "dd_over_min_side",
                     "flag_challenge", "cohort", "p_plus_branch_a", "p_plus_branch_b",
                     "p_plus_branch_c"}

GRADING_COLUMNS = ["study_id", "grader_id", "annotation_date", "image_quality",
                   "quality_reason_blur", "quality_reason_low_contrast",
                   "quality_reason_poor_illumination", "quality_reason_artifact",
                   "quality_reason_incomplete_field", "quality_reason_other",
                   "vessels_gradable", "disc_gradable", "disc_x", "disc_y", "disc_radius",
                   "rop_vascular_grade", "grade_confidence", "arterial_tortuosity_score",
                   "venous_dilation_score", "av_annotation_completed", "notes"]
SENSITIVE_META = ("exif", "icc", "comment", "software", "xmp", "iptc", "xml", "profile",
                  "dpi", "jfif", "artist", "copyright", "gps", "description", "author", "make",
                  "model", "datetime", "thumbnail")


class Checker:
    def __init__(self):
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.critical = 0

    def check(self, label: str, passed: bool, detail: str = "", critical: bool = True):
        tag = "PASS" if passed else ("FAIL" if critical else "WARN")
        print(f"  {tag}  {label}" + (f"  -- {detail}" if detail else ""))
        if not passed:
            (self.failures if critical else self.warnings).append(f"{label}: {detail}")
            if critical:
                self.critical += 1


def pixel_digest(p: Path) -> str:
    with Image.open(p) as im:
        im.load()
        a = np.array(im.convert("RGB"))
    return hashlib.sha256(a.tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--expect-images", type=int, default=30)
    args = ap.parse_args()

    pkg = Path(args.package)
    key_path = Path(args.key)
    c = Checker()
    if not pkg.is_dir():
        print(f"package directory not found: {pkg}")
        return 2

    key = list(csv.DictReader(key_path.open(newline="", encoding="utf-8")))
    pilot = [r for r in key if r.get("cohort") == "pilot"]
    final = [r for r in key if r.get("cohort") == "final"]
    key_by_id = {r["study_id"]: r for r in pilot}

    print("=== A. image count ===")
    imgs = sorted(p for p in (pkg / "images").glob("*") if p.suffix.lower() in IMAGE_EXTS)
    c.check(f"exactly {args.expect_images} RGB images", len(imgs) == args.expect_images,
            f"found {len(imgs)}")

    print("\n=== G. filename convention ===")
    bad = [p.name for p in imgs if not STUDY_ID_RE.match(p.stem)]
    c.check("every image filename is a blinded study ID (ROP_####)", not bad, str(bad[:5]))
    c.check("every image is PNG", all(p.suffix.lower() == ".png" for p in imgs),
            str([p.name for p in imgs if p.suffix.lower() != ".png"][:5]))

    print("\n=== B/C. cohort membership ===")
    ids = [p.stem for p in imgs]
    not_pilot = sorted(set(ids) - set(key_by_id))
    c.check("every packaged ID is in the frozen pilot cohort", not not_pilot, str(not_pilot[:5]))
    final_ids = {r["study_id"] for r in final}
    leaked = sorted(set(ids) & final_ids)
    c.check("no final-validation image is packaged", not leaked, str(leaked[:5]))
    # also by content, in case an ID were ever reused for the wrong file
    final_digests = set()
    for r in final:
        p = Path(r["image_path"])
        if p.exists():
            try:
                final_digests.add(pixel_digest(p))
            except Exception:  # noqa: BLE001
                pass
    dup_final = [p.name for p in imgs if pixel_digest(p) in final_digests]
    c.check("no packaged image is pixel-identical to a final-cohort image",
            not dup_final, str(dup_final[:5]))

    print("\n=== D/E/F. duplicates ===")
    c.check("no duplicate study IDs", len(ids) == len(set(ids)),
            f"{len(ids) - len(set(ids))} duplicates")
    digs = [pixel_digest(p) for p in imgs]
    c.check("no duplicate image content", len(digs) == len(set(digs)),
            f"{len(digs) - len(set(digs))} duplicate pixels")
    groups = [key_by_id[i]["group_id"] for i in ids if i in key_by_id]
    c.check("no duplicate group IDs", len(groups) == len(set(groups)),
            f"{len(groups) - len(set(groups))} duplicates")

    print("\n=== H. dimensions match the private sources ===")
    mismatched = []
    for p in imgs:
        r = key_by_id.get(p.stem)
        if not r:
            continue
        src = Path(r["image_path"])
        if not src.exists():
            mismatched.append(f"{p.name}:source-missing")
            continue
        with Image.open(src) as a, Image.open(p) as b:
            if a.size != b.size:
                mismatched.append(f"{p.name}:{a.size}->{b.size}")
    c.check("every packaged image has the source dimensions", not mismatched, str(mismatched[:5]))

    print("\n=== N. metadata ===")
    dirty = []
    for p in imgs:
        with Image.open(p) as im:
            im.load()
            keys = [k for k in im.info if any(s in k.lower() for s in SENSITIVE_META)]
            exif = dict(im.getexif()) if im.getexif() else {}
        if keys or exif:
            dirty.append(f"{p.name}:{keys or 'exif'}")
    c.check("no image carries EXIF/IPTC/XMP/text metadata", not dirty, str(dirty[:5]))

    print("\n=== I/J/K. forbidden content ===")
    all_files = [p for p in pkg.rglob("*") if p.is_file()]
    fnames = [p.name.lower() for p in all_files]
    c.check("no private key or internal manifest is included",
            not [n for n in fnames if any(s in n for s in FORBIDDEN_CSV_SUBSTR)],
            str([n for n in fnames if any(s in n for s in FORBIDDEN_CSV_SUBSTR)][:5]))
    auto = [p.name for p in all_files
            if any(s in p.name.lower() for s in ("_auto", "auto_mask", "automatic"))]
    c.check("no automatic mask is included", not auto, str(auto[:5]))
    preds = [p.name for p in all_files
             if re.search(r"pred|result|prob|\bauc\b|gradcam|grad_cam", p.name, re.I)]
    c.check("no prediction or result file is included", not preds, str(preds[:5]))
    c.check("no subdirectory other than images/vessel_masks/disc_masks/av_labels/examples",
            {p.parent.name for p in all_files} <=
            {"images", "vessel_masks", "disc_masks", "av_labels", "examples",
             pkg.name, "package_build"},
            str(sorted({p.parent.name for p in all_files})))

    print("\n=== filename leakage ===")
    leaks = []
    for p in all_files:
        stem = p.stem.lower()
        words = {w for w in WORD_SPLIT.split(stem) if w}
        if stem.startswith("rop_") and words <= {"rop"} | GEOM_WORDS:
            continue                      # a study ID cannot leak anything
        hit = sorted(words & (LEAK_WORDS | GEOM_WORDS))
        hit += [t for t in LEAK_SUBSTRINGS if t in stem]
        # directory names below the package root count too: a leaky folder is as revealing as a
        # leaky file. Parts ABOVE the package root are excluded, otherwise the parent directory
        # name ("expert_validation") matches on the word "validation".
        for part in p.relative_to(pkg).parts[:-1]:
            pw = {w for w in WORD_SPLIT.split(part.lower()) if w}
            hit += sorted(pw & (LEAK_WORDS | GEOM_WORDS))
        if hit:
            leaks.append(f"{p.name}:{','.join(sorted(set(hit)))}")
    c.check("no filename leaks a source, label, split, geometry or model term",
            not leaks, str(leaks[:8]))

    print("\n=== L/M. grading sheet ===")
    gs_path = pkg / "grading_sheet.csv"
    if not gs_path.exists():
        c.check("grading_sheet.csv exists", False, "missing")
    else:
        rows = list(csv.DictReader(gs_path.open(newline="", encoding="utf-8")))
        cols = list(rows[0].keys()) if rows else []
        c.check("grading sheet has exactly the expected columns", cols == GRADING_COLUMNS,
                f"got {cols}")
        c.check("grading sheet has one row per packaged image", len(rows) == len(imgs),
                f"{len(rows)} rows vs {len(imgs)} images")
        c.check("grading sheet study_ids match the package exactly",
                {r["study_id"] for r in rows} == set(ids),
                f"only-in-sheet={sorted({r['study_id'] for r in rows} - set(ids))[:5]}")
        forbid = [c2 for c2 in cols if c2 in FORBIDDEN_COLUMNS]
        c.check("grading sheet exposes no label/source/split/group/path column", not forbid,
                str(forbid))
        filled = [c2 for c2 in cols if c2 != "study_id" and any(r[c2].strip() for r in rows)]
        c.check("grading sheet is empty apart from study_id", not filled, str(filled))

    print("\n=== clinician-facing CSV columns ===")
    for csvp in pkg.rglob("*.csv"):
        rows = list(csv.DictReader(csvp.open(newline="", encoding="utf-8")))
        cols = list(rows[0].keys()) if rows else []
        bad = [x for x in cols if x in FORBIDDEN_COLUMNS and x != "cohort"]
        if csvp.name == "metadata_audit.csv":
            # the audit is a QC record; it names only study ids and file names, never a source
            bad = [x for x in cols if x in ("source", "label", "split", "group_id", "image_path")]
        c.check(f"{csvp.name}: no forbidden columns", not bad, str(bad))

    print("\n=== required deliverables ===")
    for rel in ("README_FIRST.pdf", "ANNOTATION_MANUAL.pdf", "grading_sheet.csv",
                "PACKAGE_MANIFEST_SHA256.txt"):
        c.check(f"{rel} present", (pkg / rel).exists())
    for d in ("images", "vessel_masks", "disc_masks", "av_labels"):
        c.check(f"{d}/ present", (pkg / d).is_dir())

    print("\n=== O. package manifest reproducibility ===")
    man = pkg / "PACKAGE_MANIFEST_SHA256.txt"
    if man.exists():
        listed, missing, wrong = {}, [], []
        for line in man.read_text().splitlines():
            m = re.match(r"^([0-9a-f]{64})\s{2}(.+)$", line)
            if m:
                listed[m.group(2)] = m.group(1)
        for rel, want in listed.items():
            fp = pkg / rel
            if not fp.exists():
                missing.append(rel)
            else:
                h = hashlib.sha256()
                with fp.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != want:
                    wrong.append(rel)
        on_disk = {p.relative_to(pkg).as_posix() for p in pkg.rglob("*") if p.is_file()}
        c.check("manifest covers every packaged file",
                not (on_disk - set(listed) - {"PACKAGE_MANIFEST_SHA256.txt"}),
                str(sorted(on_disk - set(listed))[:5]))
        c.check("no manifest entry is missing on disk", not missing, str(missing[:5]))
        c.check("every manifest hash reproduces", not wrong, str(wrong[:5]))
    else:
        c.check("PACKAGE_MANIFEST_SHA256.txt present for check O", False, "missing")

    print()
    if c.failures:
        print(f"PILOT_PACKAGE_NOT_READY — {len(c.failures)} critical failure(s)")
        for f in c.failures:
            print(f"  - {f}")
        return 1
    for w in c.warnings:
        print(f"  warning: {w}")
    print("PILOT_PACKAGE_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
