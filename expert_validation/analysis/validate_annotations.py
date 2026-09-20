"""Gate: validate the returned expert annotations before any analysis runs.

If this exits non-zero, do not run the rest of the suite and do not report numbers. The failures
it catches are the ones that silently corrupt an agreement study: resized masks, JPEG round-trips,
gradable/un gradable inconsistencies, and metadata that leaked the source or the label to the
grader.

Usage
-----
  python validate_annotations.py --key ../blinding_key.csv --root .. [--graders A,B]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import image_size, load_key, read_mask

LEAK_TOKENS = ("farabi", "farfum", "plus", "normal", "pre_plus", "preplus", "480", "960",
               "1080", "1200", "1240", "train", "val", "test", "patient")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--graders", default="A,B")
    ap.add_argument("--adjudicated", default="adjudicated")
    args = ap.parse_args()

    key = load_key(args.key)
    root = Path(args.root)
    errors: list[str] = []
    warns: list[str] = []

    for grader in [g.strip() for g in args.graders.split(",") if g.strip()]:
        gdir = root / f"grader_{grader}"
        if not gdir.exists():
            warns.append(f"grader_{grader}: directory absent, skipped")
            continue
        _check_grader(key, gdir, grader, errors, warns)

    adj = root / args.adjudicated
    if adj.exists():
        _check_grader(key, adj, "adjudicated", errors, warns, is_adjudicated=True)

    print(f"\n{len(errors)} error(s), {len(warns)} warning(s)")
    for w in warns:
        print(f"  [warn] {w}")
    for e in errors:
        print(f"  [ERROR] {e}")
    if errors:
        print("\nVALIDATION FAILED — do not analyse or report until these are fixed.")
        sys.exit(1)
    print("\nVALIDATION PASSED")


def _check_grader(key: pd.DataFrame, gdir: Path, grader: str,
                  errors: list[str], warns: list[str], is_adjudicated: bool = False) -> None:
    print(f"\n=== grader_{grader} ===")
    vm = gdir / "vessel_masks"
    dm = gdir / "disc_masks"
    gcsv = gdir / "grading.csv"
    for p in (vm, dm):
        if not p.exists():
            errors.append(f"grader_{grader}: missing {p.name}/")
    if not gcsv.exists():
        errors.append(f"grader_{grader}: missing grading.csv")
        return

    g = pd.read_csv(gcsv)
    required = ["study_id", "grader_id", "image_quality", "gradable_vessels", "gradable_disc",
                "rop_vascular_grade", "grade_confidence", "arterial_tortuosity_score",
                "venous_dilation_score"]
    miss = [c for c in required if c not in g.columns]
    if miss:
        errors.append(f"grader_{grader}: grading.csv missing columns {miss}")
        return

    for leak in ("source", "label", "split", "group_id", "image_path"):
        if leak in g.columns:
            errors.append(f"grader_{grader}: grading.csv contains '{leak}' — this leaks to the grader")

    if g["study_id"].duplicated().any():
        n = int(g["study_id"].duplicated().sum())
        errors.append(f"grader_{grader}: {n} duplicate study_id rows")
    if set(g["grader_id"].astype(str).unique()) - {grader, "adjudicated"}:
        errors.append(f"grader_{grader}: grading.csv contains a foreign grader_id")

    k = key.set_index("study_id")
    unknown = set(g["study_id"]) - set(k.index)
    if unknown:
        errors.append(f"grader_{grader}: {len(unknown)} study_id not in the blinding key")

    cohorts = {sid: k.loc[sid, "cohort"] for sid in g["study_id"] if sid in k.index}
    if not is_adjudicated:
        n_pilot = sum(1 for c in cohorts.values() if c == "pilot")
        n_final = sum(1 for c in cohorts.values() if c == "final")
        print(f"  rows={len(g)}  pilot={n_pilot}  final={n_final}")
        if n_pilot and n_final:
            warns.append(f"grader_{grader}: both cohorts in one delivery — allowed, but confirm "
                         f"the pilot was annotated before the protocol freeze")

    n_missing_mask = n_dim = n_val = 0
    for _, r in g.iterrows():
        sid = r["study_id"]
        if sid not in k.index:
            continue
        ip = Path(k.loc[sid, "image_path"])
        if not ip.exists():
            errors.append(f"grader_{grader}: study image missing on disk: {ip}")
            continue
        w0, h0 = image_size(ip)
        ung = str(r["image_quality"]).lower() == "ungradable"
        vpath = vm / f"{sid}.png"
        if ung and vpath.exists():
            errors.append(f"grader_{grader}/{sid}: ungradable image has a vessel mask")
        if not ung and not vpath.exists():
            n_missing_mask += 1
        if vpath.exists():
            try:
                m = read_mask(vpath)
            except ValueError as e:
                n_val += 1
                errors.append(f"grader_{grader}/{sid}: {e}")
                m = None
            if m is not None and m.shape != (h0, w0):
                n_dim += 1
                errors.append(f"grader_{grader}/{sid}: vessel mask {m.shape[::-1]} != image {(w0, h0)}")
        gd = bool(r["gradable_disc"]) and str(r["gradable_disc"]).lower() not in ("false", "0", "nan")
        dpath = dm / f"{sid}.png"
        if gd and not dpath.exists():
            errors.append(f"grader_{grader}/{sid}: gradable_disc true but no disc mask")
        if not gd and dpath.exists():
            errors.append(f"grader_{grader}/{sid}: gradable_disc false but a disc mask exists")
        if gd and str(r["image_quality"]).lower() == "ungradable":
            errors.append(f"grader_{grader}/{sid}: ungradable but gradable_disc true")

    if n_missing_mask:
        errors.append(f"grader_{grader}: {n_missing_mask} gradable image(s) without a vessel mask")
    print(f"  dimension mismatches={n_dim}  invalid mask values={n_val}")

    for folder in (vm, dm):
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.suffix.lower() != ".png":
                errors.append(f"grader_{grader}: non-PNG mask {f.name}")
            txt = f.stem.lower()
            for tok in LEAK_TOKENS:
                if tok in txt and not txt.startswith("rop_"):
                    errors.append(f"grader_{grader}: filename leaks '{tok}': {f.name}")


if __name__ == "__main__":
    main()
