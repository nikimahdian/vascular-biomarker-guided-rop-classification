"""Build ONE shared train/val/test index over labelled fundus images.

Both branches (biomarker table and image CNN) read the same split so the
comparison at the end is fair. Output: data/splits/{train,val,test}.csv with
columns [image_path, label, split, source].

Labels are inferred from folder names (see config keywords). Default is
3-class: Normal (0), Pre_Plus (1), Plus (2).

Usage:
    python -m src.data.prepare_split
    python -m src.data.prepare_split --roots data/raw/farabi data/raw/plus
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

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
            label = infer_label(f, pre_plus_kw, plus_kw, negative_kw)
            if label is None:
                continue
            rows.append({
                "image_path": str(f.resolve()),
                "label": int(label),
                "source": root.name,
                # Image-level groups for Farabi/Plus (no patient metadata).
                "patient_id": f"{root.name}__img__{f.resolve()}",
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
                "patient_id": f"farfum_{r[id_col]}",
            }
        )
    if miss:
        print(f"[farfum] {miss} labelled rows had no matching image file")
    df = pd.DataFrame(rows).drop_duplicates(subset="image_path").reset_index(drop=True)
    print(f"[farfum] loaded {len(df)} images from labels")
    return df


def split_by_patient(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
    group_col: str = "patient_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test at patient (group) level to avoid leakage."""
    rest_frac = val_size + test_size
    groups = (
        df.groupby(group_col, as_index=False)
        .agg(label=("label", lambda s: int(s.mode().iloc[0])), n=("label", "size"))
        .reset_index(drop=True)
    )
    train_g, rest_g = train_test_split(
        groups, test_size=rest_frac, stratify=groups["label"], random_state=seed
    )
    val_g, test_g = train_test_split(
        rest_g,
        test_size=test_size / rest_frac,
        stratify=rest_g["label"],
        random_state=seed,
    )
    train_ids = set(train_g[group_col])
    val_ids = set(val_g[group_col])
    test_ids = set(test_g[group_col])
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

    print("[scan] total images:", len(df))
    counts = df["label"].value_counts().sort_index()
    print(counts.rename({i: class_names[i] for i in range(len(class_names))}))
    if "patient_id" in df.columns:
        print(f"[scan] unique patients/groups: {df['patient_id'].nunique()}")

    test_size = d["test_size"]
    val_size = d["val_size"]
    if args.patient_split and "patient_id" in df.columns:
        print("[split] patient-level stratified split")
        train_df, val_df, test_df = split_by_patient(df, test_size, val_size, cfg["seed"])
    else:
        rest_frac = val_size + test_size
        train_df, rest_df = train_test_split(
            df, test_size=rest_frac, stratify=df["label"], random_state=cfg["seed"]
        )
        val_df, test_df = train_test_split(
            rest_df,
            test_size=test_size / rest_frac,
            stratify=rest_df["label"],
            random_state=cfg["seed"],
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
    print("[done]")


if __name__ == "__main__":
    main()
