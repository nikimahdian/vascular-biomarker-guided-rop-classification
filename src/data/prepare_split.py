"""Build ONE shared train/val/test index over labelled fundus images.

Both branches (biomarker table and image CNN) read the same split so the
comparison at the end is fair. Output: data/splits/{train,val,test}.csv with
columns [image_path, label, split, source, group_id, patient_id, exam_id,
identity_level].

Labels are inferred from folder names (see config keywords). Default is
3-class: Normal (0), Pre_Plus (1), Plus (2).

Usage:
    python -m src.data.prepare_split
    python -m src.data.prepare_split --roots data/raw/farabi data/raw/plus
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from src.utils.common import ensure_dirs, load_config, set_seed


def _norm(text: str) -> str:
    return text.lower().replace("-", " ")


def _matches(text: str, keywords: list[str]) -> bool:
    for kw in keywords:
        if _norm(kw) in text:
            return True
    return False


def infer_label(
    path: Path,
    pre_plus_kw: list[str],
    plus_kw: list[str],
    negative_kw: list[str],
) -> int | None:
    """Return 0 (Normal), 1 (Pre_Plus), 2 (Plus), or None if unlabelled."""
    parts = [_norm(p) for p in path.parts]
    joined = " / ".join(parts)
    parent = _norm(path.parent.name)

    # Order matters: "no plus" before "plus"; "pre plus" before "plus".
    if _matches(joined, negative_kw) or _matches(parent, negative_kw):
        return 0
    if _matches(joined, pre_plus_kw) or _matches(parent, pre_plus_kw):
        return 1
    if _matches(joined, plus_kw) or _matches(parent, plus_kw):
        return 2
    return None


def infer_identity(root: Path, image: Path) -> tuple[str, str | None, str, str]:
    """Return group ID, patient ID, exam ID, and evidence level."""
    stem = image.stem
    if root.name == "plus":
        # Official dataset convention: first filename token is unique patient ID.
        tokens = stem.split("_")
        patient_id = f"plus::{tokens[0]}"
        exam_id = f"{patient_id}::S{tokens[8][1:]}::PA{tokens[4][2:]}"
        return (
            patient_id,
            patient_id,
            exam_id,
            "verified_patient_official_dataset",
        )
    if root.name == "farabi":
        # Native RetCam metadata proves filename UUID is Exam ID and differs from
        # RetCam Patient ID. Patient linkage is unavailable for most exams.
        exam_id = f"farabi::{stem.rsplit('.', 1)[0]}"
        return exam_id, None, exam_id, "verified_exam_patient_unknown"
    raise ValueError(f"No identity rule for source root: {root}")


def scan(
    roots,
    exts,
    pre_plus_kw: list[str],
    plus_kw: list[str],
    negative_kw: list[str],
) -> pd.DataFrame:
    rows = []
    exts = {e.lower() for e in exts}
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"[warn] root not found: {root}")
            continue
        for f in root.rglob("*"):
            if f.suffix.lower() not in exts or not f.is_file():
                continue
            # Search only below source root. Absolute paths can contain words such
            # as "plus" and silently label every image positive.
            label = infer_label(f.relative_to(root), pre_plus_kw, plus_kw, negative_kw)
            if label is None:
                continue
            group_id, patient_id, exam_id, identity_level = infer_identity(root, f)
            rows.append({
                "image_path": str(f.resolve()),
                "label": int(label),
                "source": root.name,
                "group_id": group_id,
                "patient_id": patient_id,
                "exam_id": exam_id,
                "identity_level": identity_level,
            })
    df = pd.DataFrame(rows).drop_duplicates(subset="image_path").reset_index(drop=True)
    return df


def scan_farfum(farfum_root: Path, labels_xlsx: Path) -> pd.DataFrame:
    """Load FARFUM-RoP via Dataset_Labels.xlsx.

    Official Label column: 1=Normal, 2=Pre-Plus, 3=Plus → mapped to 0/1/2.
    """
    if not farfum_root.exists() or not labels_xlsx.exists():
        print(f"[warn] FARFUM missing: root={farfum_root.exists()} labels={labels_xlsx.exists()}")
        return pd.DataFrame(columns=["image_path", "label", "source"])

    meta = pd.read_excel(labels_xlsx, skiprows=1)
    # Columns after skiprows=1: patient id, image name, ..., consensus label (last col)
    id_col, name_col, label_col = meta.columns[0], meta.columns[1], meta.columns[-1]
    # Index images under patients/ by stem (filename without extension).
    by_stem: dict[str, Path] = {}
    for f in (farfum_root / "patients").rglob("*"):
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            by_stem[f.stem] = f

    rows = []
    miss = 0
    for _, r in meta.iterrows():
        stem = str(r[name_col]).strip()
        raw_lab = r[label_col]
        if pd.isna(raw_lab) or stem.lower() in {"image name", "nan"}:
            continue
        lab = int(float(raw_lab))  # 1/2/3
        if lab not in (1, 2, 3):
            continue
        path = by_stem.get(stem)
        if path is None:
            miss += 1
            continue
        rows.append(
            {
                "image_path": str(path.resolve()),
                "label": lab - 1,  # → 0/1/2
                "source": "farfum_rop",
                "group_id": f"farfum_{r[id_col]}",
                "patient_id": f"farfum_{r[id_col]}",
                "exam_id": None,
                "identity_level": "verified_patient_official_metadata",
            }
        )
    if miss:
        print(f"[farfum] {miss} labelled rows had no matching image file")
    df = pd.DataFrame(rows).drop_duplicates(subset="image_path").reset_index(drop=True)
    print(f"[farfum] loaded {len(df)} images from labels")
    return df


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclude_ambiguous_exact_duplicates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude identities linked by cross-patient duplicate pixels."""
    out = df.copy()
    out["sha256"] = out["image_path"].map(_sha256)
    identity_counts = out.groupby("sha256")["group_id"].nunique()
    ambiguous_hashes = set(identity_counts[identity_counts > 1].index)
    ambiguous_ids = set(out.loc[out["sha256"].isin(ambiguous_hashes), "group_id"])
    excluded = out[out["group_id"].isin(ambiguous_ids)].copy()
    excluded["exclusion_reason"] = "identity_linked_by_cross_patient_exact_duplicate"
    excluded["triggering_duplicate"] = excluded["sha256"].isin(ambiguous_hashes)
    retained = out[~out["group_id"].isin(ambiguous_ids)].drop(columns="sha256").copy()
    return retained, excluded


def split_by_patient(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
    group_col: str = "group_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Balance image/source/class counts while keeping groups indivisible."""
    del seed  # HiGHS is deterministic for this fixed optimization model.
    split_names = ["train", "val", "test"]
    fractions = np.array([1.0 - val_size - test_size, val_size, test_size])
    group_ids = sorted(df[group_col].unique())
    group_rows = {group_id: df[df[group_col] == group_id] for group_id in group_ids}
    metrics: list[tuple[str, np.ndarray, float]] = []

    def add_metric(name, counter, weight) -> None:
        values = np.array([float(counter(group_rows[group_id])) for group_id in group_ids])
        metrics.append((name, values, weight))

    add_metric("images_total", len, 4.0)
    for source in sorted(df["source"].unique()):
        add_metric(
            f"images_source_{source}",
            lambda rows, source=source: (rows["source"] == source).sum(),
            2.0,
        )
    for label in sorted(df["label"].unique()):
        add_metric(
            f"images_label_{label}",
            lambda rows, label=label: (rows["label"] == label).sum(),
            2.0,
        )
    for source in sorted(df["source"].unique()):
        for label in sorted(df["label"].unique()):
            if ((df["source"] == source) & (df["label"] == label)).any():
                add_metric(
                    f"images_{source}_label_{label}",
                    lambda rows, source=source, label=label: (
                        (rows["source"] == source) & (rows["label"] == label)
                    ).sum(),
                    1.0,
                )
    add_metric("groups_total", lambda rows: 1, 0.35)
    for source in sorted(df["source"].unique()):
        add_metric(
            f"groups_source_{source}",
            lambda rows, source=source: (rows["source"] == source).any(),
            0.35,
        )

    n_groups, n_splits, n_metrics = len(group_ids), 3, len(metrics)
    n_assign = n_groups * n_splits
    n_deviation = n_splits * n_metrics
    n_variables = n_assign + 2 * n_deviation
    objective = np.zeros(n_variables)
    for split_index in range(n_splits):
        for metric_index, (_, values, weight) in enumerate(metrics):
            target = max(fractions[split_index] * values.sum(), 1.0)
            coefficient = weight / target
            objective[n_assign + split_index * n_metrics + metric_index] = coefficient
            objective[
                n_assign + n_deviation + split_index * n_metrics + metric_index
            ] = coefficient

    integrality = np.zeros(n_variables)
    integrality[:n_assign] = 1
    lower = np.zeros(n_variables)
    upper = np.full(n_variables, np.inf)
    upper[:n_assign] = 1
    matrix = lil_matrix((n_groups + n_splits * n_metrics, n_variables))
    constraint_lower = np.zeros(matrix.shape[0])
    constraint_upper = np.zeros(matrix.shape[0])

    for group_index in range(n_groups):
        for split_index in range(n_splits):
            matrix[group_index, group_index * n_splits + split_index] = 1
        constraint_lower[group_index] = constraint_upper[group_index] = 1

    row = n_groups
    for split_index in range(n_splits):
        for metric_index, (_, values, _) in enumerate(metrics):
            for group_index in range(n_groups):
                matrix[row, group_index * n_splits + split_index] = values[group_index]
            matrix[row, n_assign + split_index * n_metrics + metric_index] = -1
            matrix[
                row,
                n_assign + n_deviation + split_index * n_metrics + metric_index,
            ] = 1
            target = fractions[split_index] * values.sum()
            constraint_lower[row] = constraint_upper[row] = target
            row += 1

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            matrix.tocsr(), constraint_lower, constraint_upper
        ),
        options={"time_limit": 180, "mip_rel_gap": 0.02},
    )
    if result.x is None:
        raise RuntimeError(f"Grouped split optimization failed: {result.message}")
    print(f"[split] optimizer: {result.message}; objective={result.fun:.6f}")
    assignments = {
        group_ids[group_index]: split_names[
            int(
                np.argmax(
                    result.x[
                        group_index * n_splits : (group_index + 1) * n_splits
                    ]
                )
            )
        ]
        for group_index in range(n_groups)
    }
    train_ids = {key for key, value in assignments.items() if value == "train"}
    val_ids = {key for key, value in assignments.items() if value == "val"}
    test_ids = {key for key, value in assignments.items() if value == "test"}
    return (
        df[df[group_col].isin(train_ids)].copy(),
        df[df[group_col].isin(val_ids)].copy(),
        df[df[group_col].isin(test_ids)].copy(),
    )


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg["seed"])
    d = cfg["data"]
    class_names: list[str] = d["class_names"]

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="*",
        default=[str(cfg["paths"]["raw_dir"] / "farabi"), str(cfg["paths"]["raw_dir"] / "plus")],
        help="Folders to scan recursively for labelled fundus images.",
    )
    ap.add_argument(
        "--include-farfum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="merge FARFUM-RoP (xlsx labels) into the shared split (default: on)",
    )
    ap.add_argument(
        "--patient-split",
        action=argparse.BooleanOptionalAction,
        default=cfg.get("eval", {}).get("patient_level_split", True),
        help="split at patient_id level (default from config eval.patient_level_split)",
    )
    args = ap.parse_args()

    df = scan(
        args.roots,
        d["image_exts"],
        d["pre_plus_keywords"],
        d["plus_keywords"],
        d["negative_keywords"],
    )
    if args.include_farfum:
        farfum_root = Path(d.get("farfum_dir", cfg["paths"]["raw_dir"] / "farfum_rop"))
        if not farfum_root.is_absolute():
            farfum_root = cfg["_root"] / farfum_root
        labels_xlsx = Path(d.get("farfum_labels", farfum_root / "metadata" / "Dataset_Labels.xlsx"))
        if not labels_xlsx.is_absolute():
            labels_xlsx = cfg["_root"] / labels_xlsx
        far = scan_farfum(farfum_root, labels_xlsx)
        if not far.empty:
            df = pd.concat([df, far], ignore_index=True).drop_duplicates(subset="image_path")
    if df.empty:
        raise SystemExit(
            "No labelled images found. Check --roots and the label keywords in config."
        )

    quality_exclusions_path = (
        cfg["paths"]["raw_dir"].parent / "metadata" / "quality_exclusions.csv"
    )
    if quality_exclusions_path.exists():
        quality_exclusions = pd.read_csv(quality_exclusions_path)
        excluded_paths = set(quality_exclusions["image_path"])
        quality_excluded = df[df["image_path"].isin(excluded_paths)].copy()
        missing_exclusions = excluded_paths - set(df["image_path"])
        if missing_exclusions:
            raise ValueError(
                f"Quality exclusions not present in scanned data: {missing_exclusions}"
            )
        df = df[~df["image_path"].isin(excluded_paths)].copy()
        print(
            f"[quality] excluded {len(quality_excluded)} documented unusable images "
            f"from {quality_exclusions_path}"
        )
    else:
        quality_excluded = pd.DataFrame(columns=df.columns)

    df, excluded = exclude_ambiguous_exact_duplicates(df)
    excluded_path = cfg["paths"]["splits_dir"] / "excluded_ambiguous_exact_duplicates.csv"
    excluded.to_csv(excluded_path, index=False)
    print(
        f"[dedupe] excluded {len(excluded)} rows from "
        f"{excluded['group_id'].nunique()} identities linked by "
        f"{excluded.loc[excluded['triggering_duplicate'], 'sha256'].nunique()} "
        f"cross-patient exact-image groups -> {excluded_path}"
    )

    print("[scan] total images:", len(df))
    counts = df["label"].value_counts().sort_index()
    print(counts.rename({i: class_names[i] for i in range(len(class_names))}))
    if "group_id" in df.columns:
        print(f"[scan] unique leakage-prevention groups: {df['group_id'].nunique()}")

    test_size = d["test_size"]
    val_size = d["val_size"]
    if args.patient_split and "group_id" in df.columns:
        print("[split] grouped MILP split")
        train_df, val_df, test_df = split_by_patient(df, test_size, val_size, cfg["seed"])
    else:
        raise SystemExit(
            "Image-level splitting is disabled because it leaks patient/case identity. "
            "Use --patient-split."
        )
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part = part.copy()
        part["split"] = name
        out = cfg["paths"]["splits_dir"] / f"{name}.csv"
        part.to_csv(out, index=False)
        lbl = part["label"].value_counts().sort_index().to_dict()
        print(f"[write] {out}  n={len(part)}  labels={lbl}")

    alldf = pd.concat(
        [train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")]
    )
    alldf.to_csv(cfg["paths"]["splits_dir"] / "all.csv", index=False)
    split_sets = {
        name: set(part["group_id"])
        for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]
    }
    assert not split_sets["train"] & split_sets["val"]
    assert not split_sets["train"] & split_sets["test"]
    assert not split_sets["val"] & split_sets["test"]
    assert alldf["image_path"].nunique() == len(alldf)
    counts = []
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts.append(
            {
                "split": name,
                "images": len(part),
                "groups": part["group_id"].nunique(),
                "normal": int((part["label"] == 0).sum()),
                "pre_plus": int((part["label"] == 1).sum()),
                "plus": int((part["label"] == 2).sum()),
                **{
                    f"source_{source}": int((part["source"] == source).sum())
                    for source in sorted(alldf["source"].unique())
                },
            }
        )
    pd.DataFrame(counts).to_csv(
        cfg["paths"]["splits_dir"] / "split_counts.csv", index=False
    )
    report = {
        "status": "OFFICIAL_PHASE2_GROUPED_SPLIT",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "identity_parser": {
            "plus": "official unique patient ID: leading filename integer",
            "farfum_rop": "official spreadsheet patient ID",
            "farabi": (
                "Native RetCam Exam ID: UUID before final image-number suffix; "
                "patient ID unavailable"
            ),
        },
        "exclusions": {
            "duplicate_identity_rows": len(excluded),
            "quality_rows": len(quality_excluded),
            "group_ids": sorted(excluded["group_id"].unique()),
            "triggering_exact_hash_groups": int(
                excluded.loc[excluded["triggering_duplicate"], "sha256"].nunique()
            ),
        },
        "checks": {
            "rows": len(alldf),
            "unique_paths": alldf["image_path"].nunique(),
            "groups": alldf["group_id"].nunique(),
            "group_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
        },
        "counts": counts,
        "hashes": {
            name: _sha256(str(cfg["paths"]["splits_dir"] / name))
            for name in ["all.csv", "train.csv", "val.csv", "test.csv"]
        },
    }
    (cfg["paths"]["splits_dir"] / "split_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("[done]")


if __name__ == "__main__":
    main()
