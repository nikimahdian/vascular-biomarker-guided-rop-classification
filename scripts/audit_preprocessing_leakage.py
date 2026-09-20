#!/usr/bin/env python
"""Section K: is the global-median fallback and the global variance filter actually active?"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = "/Users/moniaz/niki"
META = {"image_path", "mask_path", "label", "split", "source", "group_id",
        "patient_id", "exam_id", "identity_level"}

for name in ("biomarker_features.csv", "biomarker_features_clinical_v3.csv"):
    try:
        df = pd.read_csv(f"{ROOT}/data/features/{name}")
    except Exception as e:  # noqa: BLE001
        print(f"{name}: {e}")
        continue
    cols = [c for c in df.columns if c not in META]
    X = df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    tr = df.split.values == "train"
    all_nan_train = [c for c in X.columns if X.loc[tr, c].isna().all()]
    print(f"\n=== {name}  rows={len(df)}  numeric cols={len(cols)} -> {len(X.columns)} after dropna-all ===")
    print(f"  columns all-NaN in train                : {len(all_nan_train)} {all_nan_train[:6]}")
    leaky = [c for c in all_nan_train if X[c].notna().any()]
    print(f"  ... of which have values outside train   : {len(leaky)} {leaky[:6]}")
    print(f"  -> global-median fallback is "
          f"{'ACTIVE (a real leak)' if leaky else 'INACTIVE (latent only)'}")

    # variance filter leakage
    std_all = X.std(numeric_only=True).fillna(0)
    std_tr = X.loc[tr].std(numeric_only=True).fillna(0)
    dropped_all = set(std_all[std_all <= 0].index)
    dropped_tr = set(std_tr[std_tr <= 0].index)
    only_test = dropped_tr - dropped_all
    print(f"  columns constant globally               : {len(dropped_all)}")
    print(f"  columns constant in train only          : {len(only_test)} {sorted(only_test)[:6]}")
    print(f"  -> global variance filter is "
          f"{'ACTIVE (drops a column using val/test variance)' if dropped_tr != dropped_all else 'INACTIVE'}")

    # source separability of the missingness pattern alone (section H)
    miss = X.isna().astype(int)
    if miss.values.sum() > 0:
        print(f"  missingness pattern non-empty: {int(miss.values.sum())} missing cells "
              f"over {int((miss.sum(axis=1) > 0).sum())} rows")
    else:
        print("  missingness pattern: no missing cells at all")
