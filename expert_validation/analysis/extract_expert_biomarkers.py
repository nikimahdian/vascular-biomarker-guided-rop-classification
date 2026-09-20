"""Re-extract the vascular features from the EXPERT masks, using the pipeline's own code.

Three variants per image, and they answer different questions:

  ref  expert vessel  + expert disc   the reference measurement
  m1   automatic vessel + EXPERT disc Measurement 1 -- isolates segmentation error, because the
                                       disc is held fixed on both sides of the comparison
  m2   automatic vessel + automatic disc  Measurement 2 -- what a user actually gets

`measure()` is imported directly from scripts/clinical_features_v3.py rather than reimplemented, so
the comparison is of masks and not of two different measurement implementations. The same disc
validity rule therefore applies: if the disc does not pass, disc-relative features come back NaN.

Usage
-----
  python extract_expert_biomarkers.py --key ../blinding_key.csv --root .. \
      --disc-pred /path/disc_predictions_all.csv --out out/biomarkers
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import disc_geometry, load_key, read_mask

VARIANTS = ("ref", "m1", "m2")


def load_pipeline(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    path = repo_root / "scripts" / "clinical_features_v3.py"
    if not path.exists():
        raise SystemExit(f"cannot find {path}; run from inside the repository")
    spec = importlib.util.spec_from_file_location("clinical_features_v3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--disc-pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--graders", default="A")
    args = ap.parse_args()

    cf3 = load_pipeline(Path(args.repo_root).resolve())
    key = load_key(args.key)
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dp = pd.read_csv(args.disc_pred).set_index("image_path")
    print(f"pipeline loaded; DISC_DEPENDENT columns: {len(cf3.DISC_DEPENDENT)}")

    rows = []
    for grader in [g.strip() for g in args.graders.split(",") if g.strip()]:
        vdir = root / f"grader_{grader}" / "vessel_masks"
        ddir = root / f"grader_{grader}" / "disc_masks"
        if not vdir.exists():
            print(f"[skip] grader_{grader}")
            continue
        for _, r in key.iterrows():
            vp = vdir / f"{r.study_id}.png"
            if not vp.exists():
                continue
            auto_mask = read_mask(r.mask_path)

            # expert disc geometry, at native resolution, then scaled into working space by load_pair
            expert_disc = None
            dpath = ddir / f"{r.study_id}.png"
            if dpath.exists():
                g = disc_geometry(read_mask(dpath))
                if np.isfinite(g["radius_px"]) and g["radius_px"] > 0:
                    expert_disc = {"disc_dd_px": 2 * g["radius_px"], "disc_cx": g["cx"],
                                   "disc_cy": g["cy"], "peak_prob": 1.0}

            auto_disc = None
            if r.image_path in dp.index:
                d = dp.loc[r.image_path]
                if np.isfinite(d.get("disc_dd_px", np.nan)) and d["disc_dd_px"] > 2:
                    auto_disc = {"disc_dd_px": float(d["disc_dd_px"]),
                                 "disc_cx": float(d["disc_cx"]), "disc_cy": float(d["disc_cy"]),
                                 "peak_prob": float(d.get("peak_prob", np.nan))}

            rgb, expert_mask, sc = cf3.load_pair(r.image_path, str(vp))

            def run(mask: np.ndarray, disc: dict | None, variant: str):
                if disc is None:
                    return None
                d2 = {k: (v * sc if k.startswith("disc_") else v) for k, v in disc.items()}
                try:
                    f = cf3.measure(rgb, mask, d2)
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] {r.study_id} {variant}: {type(e).__name__}: {e}")
                    return None
                return {"study_id": r.study_id, "grader": grader, "variant": variant,
                        "cohort": r.cohort, "source": r.source, "min_side": r.min_side,
                        "label": r.label, "group_id": r.group_id,
                        "flag_challenge": r.flag_challenge, **f}

            for variant, mask, disc in (("ref", expert_mask, expert_disc),
                                        ("m1", auto_mask, expert_disc),
                                        ("m2", auto_mask, auto_disc)):
                rec = run(mask, disc, variant)
                if rec is not None:
                    rows.append(rec)

    if not rows:
        raise SystemExit("nothing extracted — check --root and the expert masks")
    df = pd.DataFrame(rows)
    df.to_csv(out / "biomarker_measurements.csv", index=False)
    print(f"\nrows={len(df)}  variants={df.variant.value_counts().to_dict()}")
    print(f"disc_valid rate by variant: "
          f"{df.groupby('variant').disc_valid.mean().round(3).to_dict()}")
    cols = [c for c in ("vessel_density", "width_p90_dd", "width_p90_px", "tort_p90",
                        "av_width_ratio_p90") if c in df.columns]
    print("\nnon-null counts per variant:")
    print(df.groupby("variant")[cols].apply(lambda g: g.notna().sum()).to_string())
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
