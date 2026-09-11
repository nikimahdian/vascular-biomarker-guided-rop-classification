#!/usr/bin/env python3
"""Strict Phase 5 A/B reality check before Branch C."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

root = Path("/Users/moniaz/niki")
fails: list[tuple[str, str]] = []
passes: list[tuple[str, str]] = []


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ok(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        passes.append((name, detail))
    else:
        fails.append((name, detail or "failed"))


def pick_score(frame: pd.DataFrame) -> pd.Series:
    for col in [
        "p_plus_branch_a",
        "p_plus_branch_b",
        "p_plus_branch_c",
        "prob_plus",
        "plus_prob",
        "y_score",
        "score",
        "prob_2",
        "p_plus",
        "plus_score",
    ]:
        if col in frame.columns:
            return frame[col].astype(float)
    # any p_plus* column
    for col in frame.columns:
        if col.startswith("p_plus"):
            return frame[col].astype(float)
    # probability columns
    prob_cols = [c for c in frame.columns if c.startswith("prob_")]
    if "prob_2" in frame.columns:
        return frame["prob_2"].astype(float)
    if len(prob_cols) >= 3:
        return frame[sorted(prob_cols)[-1]].astype(float)
    raise SystemExit(f"No score column in {list(frame.columns)}")


def pick_label(frame: pd.DataFrame) -> pd.Series:
    for col in ["label", "y_true", "true_label"]:
        if col in frame.columns:
            return frame[col].astype(int)
    raise SystemExit(f"No label column in {list(frame.columns)}")


def plus_binary(labels: pd.Series) -> np.ndarray:
    return (labels.astype(int) == 2).astype(int).values


def recompute_auc(preds: pd.DataFrame) -> float:
    y = plus_binary(pick_label(preds))
    s = pick_score(preds).values
    return float(roc_auc_score(y, s))


def main() -> None:
    splits = {
        name: pd.read_csv(root / "data" / "splits" / f"{name}.csv")
        for name in ["all", "train", "val", "test"]
    }
    all_sha = sha(root / "data" / "splits" / "all.csv")
    ok(
        "split rows sum",
        len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        == len(splits["all"]),
        f"{len(splits['train'])}+{len(splits['val'])}+{len(splits['test'])}={len(splits['all'])}",
    )
    ok("all.csv sha recorded", True, all_sha)

    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        g_overlap = set(splits[left].group_id) & set(splits[right].group_id)
        i_overlap = set(splits[left].image_path) & set(splits[right].image_path)
        ok(f"group overlap {left}/{right}=0", len(g_overlap) == 0, str(len(g_overlap)))
        ok(f"image overlap {left}/{right}=0", len(i_overlap) == 0, str(len(i_overlap)))

    feats = pd.read_csv(root / "data" / "features" / "biomarker_features.csv")
    masks = pd.read_csv(root / "data" / "masks" / "mask_manifest.csv")
    ok(
        "features==split set",
        set(feats.image_path) == set(splits["all"].image_path),
        f"f={len(feats)} s={len(splits['all'])}",
    )
    ok(
        "masks==split set",
        set(masks.image_path) == set(splits["all"].image_path),
        f"m={len(masks)} s={len(splits['all'])}",
    )
    ok("no dup features", not feats.image_path.duplicated().any(), "")
    ok("no dup masks", not masks.image_path.duplicated().any(), "")
    ok("no NaN vessel_density", not feats["vessel_density"].isna().any(), str(feats["vessel_density"].isna().sum()))

    qg = json.loads((root / "results" / "phase4_quality_gates.json").read_text())
    ok("P4 gates PASS", qg["status"] == "AUTOMATED_GATES_PASS", qg["status"])
    ok(
        "P4 review unresolved=0",
        qg["mask_review_queue"]["unresolved"] == 0,
        str(qg["mask_review_queue"]),
    )
    ok("P4 rows==split", qg["rows"] == len(splits["all"]), f"{qg['rows']} vs {len(splits['all'])}")

    a = json.loads((root / "results" / "branch_a_results.json").read_text())
    b = json.loads((root / "results" / "branch_b_results.json").read_text())
    a_split = a["provenance"]["split_sha256"]
    b_split = b["split_sha256"]
    ok("A split matches all.csv", a_split == all_sha, a_split)
    ok("B split matches all.csv", b_split == all_sha, b_split)
    ok("A/B same split", a_split == b_split, "")

    ckpt = Path(b["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    ok("B checkpoint exists", ckpt.exists(), str(ckpt))
    if ckpt.exists() and b.get("checkpoint_sha256"):
        ok("B ckpt sha matches json", sha(ckpt) == b["checkpoint_sha256"], b["checkpoint_sha256"])

    # C must be legacy / not matching
    c_path = root / "results" / "branch_c_results.json"
    if c_path.exists():
        c = json.loads(c_path.read_text())
        c_split = (c.get("provenance") or {}).get("split_sha256") or c.get("split_sha256")
        ok("C is NOT current-split (must rerun)", c_split != all_sha, f"c_split={c_split}")

    def check_preds(tag: str, rel: str, split_df: pd.DataFrame) -> pd.DataFrame | None:
        path = root / rel
        if not path.exists():
            fails.append((f"{tag} preds exist", f"missing {rel}"))
            return None
        frame = pd.read_csv(path)
        ok(f"{tag} n rows", len(frame) == len(split_df), f"{len(frame)} vs {len(split_df)}")
        ok(
            f"{tag} image set",
            set(frame.image_path) == set(split_df.image_path),
            f"sym={len(set(frame.image_path) ^ set(split_df.image_path))}",
        )
        ok(f"{tag} no dup", not frame.image_path.duplicated().any(), "")
        merged = frame.merge(
            split_df[["image_path", "label"]],
            on="image_path",
            suffixes=("_pred", "_split"),
        )
        pred_label = "label_pred" if "label_pred" in merged.columns else "label"
        if pred_label in merged.columns and "label_split" in merged.columns:
            ok(
                f"{tag} labels==split",
                (merged[pred_label].astype(int) == merged["label_split"].astype(int)).all(),
                "",
            )
        # score finite
        score = pick_score(frame)
        ok(f"{tag} score finite", np.isfinite(score).all(), str((~np.isfinite(score)).sum()))
        return frame

    a_test = check_preds("A_test", "results/branch_a_test_preds.csv", splits["test"])
    a_val = check_preds("A_val", "results/branch_a_val_preds.csv", splits["val"])
    b_test = check_preds("B_test", "results/branch_b_test_preds.csv", splits["test"])
    b_val = check_preds("B_val", "results/branch_b_val_preds.csv", splits["val"])

    print("A_test cols:", list(a_test.columns) if a_test is not None else None)
    print("B_test cols:", list(b_test.columns) if b_test is not None else None)

    # independent AUC recompute vs JSON
    if a_test is not None:
        a_auc = recompute_auc(a_test)
        a_json = a["metrics"]["xgboost"]["test"]["plus_ovr"]["auc"]
        ok("A AUC recompute==json", abs(a_auc - a_json) < 1e-9, f"{a_auc} vs {a_json}")
        ok("A best_model xgboost", a.get("best_model") == "xgboost", str(a.get("best_model")))
    if b_test is not None:
        b_auc = recompute_auc(b_test)
        b_json = b["test"]["plus_ovr"]["auc"]
        ok("B AUC recompute==json", abs(b_auc - b_json) < 1e-9, f"{b_auc} vs {b_json}")
        ok("B backbone efficientnet_b5", b.get("backbone") == "efficientnet_b5", str(b.get("backbone")))

    # A/B cover same test images
    if a_test is not None and b_test is not None:
        ok(
            "A/B same test images",
            set(a_test.image_path) == set(b_test.image_path),
            "",
        )

    # pytest
    proc = subprocess.run(
        [
            str(root / ".venv" / "bin" / "python"),
            "-m",
            "pytest",
            "tests/test_prepare_split.py",
            "tests/test_phase3_regressions.py",
            "-q",
            "--tb=no",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    ok("pytest regressions", proc.returncode == 0, proc.stdout.strip().split("\n")[-1])

    print("\nPASS:")
    for name, detail in passes:
        print(f"  PASS | {name} | {detail}")
    print("\nFAIL:")
    for name, detail in fails:
        print(f"  FAIL | {name} | {detail}")
    print(f"\nSUMMARY pass={len(passes)} fail={len(fails)}")
    if fails:
        raise SystemExit(1)
    print("REALITY_CHECK_OK")


if __name__ == "__main__":
    main()
