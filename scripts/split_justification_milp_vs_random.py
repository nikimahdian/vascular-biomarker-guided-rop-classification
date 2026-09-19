#!/usr/bin/env python
"""MILP assignment vs random group-level assignment.

Answers the reviewer's only falsifiable demand: is the MILP split actually better than a plain
group-level random split, and by how much?

The cost function is re-implemented EXACTLY as in src/data/prepare_split.py::split_by_patient
(metrics, weights, targets and the max(target,1) guard). The recorded MILP assignment is read
from data/splits/all.csv; nothing is re-optimised.

Baselines (both keep groups indivisible, so neither can leak):
  uniform    : each of the 414 groups assigned uniformly at random to train/val/test
  size_aware : groups shuffled once, then walked in order accumulating images; boundaries at
               70% and 85% of the total image count (the GroupShuffleSplit analogue)

Read-only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/moniaz/niki")
os.chdir(ROOT)
OUT = ROOT / "results" / "split_justification"
OUT.mkdir(parents=True, exist_ok=True)

SEED, N_RANDOM = 42, 2000
VAL_SIZE = TEST_SIZE = 0.15
FRACTIONS = np.array([1.0 - VAL_SIZE - TEST_SIZE, VAL_SIZE, TEST_SIZE])
SPLIT_NAMES = ["train", "val", "test"]

df = pd.read_csv("data/splits/all.csv")
groups = sorted(df["group_id"].unique())
n_groups = len(groups)
rows_by_group = {g: df[df["group_id"] == g] for g in groups}
sources_sorted = sorted(df["source"].unique())

# ---- per-group metric values, exactly as prepare_split builds them
metrics: list[tuple[str, np.ndarray, float]] = []


def add_metric(name, counter, weight):
    metrics.append((name, np.array([float(counter(rows_by_group[g])) for g in groups]), weight))


add_metric("images_total", len, 4.0)
for source in sources_sorted:
    add_metric(f"images_source_{source}", lambda r, s=source: (r["source"] == s).sum(), 2.0)
for label in sorted(df["label"].unique()):
    add_metric(f"images_label_{label}", lambda r, l=label: (r["label"] == l).sum(), 2.0)
for source in sources_sorted:
    for label in sorted(df["label"].unique()):
        add_metric(
            f"images_{source}_label_{label}",
            lambda r, s=source, l=label: ((r["source"] == s) & (r["label"] == l)).sum(),
            1.0,
        )
add_metric("groups_total", lambda r: 1, 0.35)
for source in sources_sorted:
    add_metric(f"groups_source_{source}", lambda r, s=source: (r["source"] == s).any(), 0.35)

# ---- fast per-group vectors for the descriptive statistics
g_images = np.array([len(rows_by_group[g]) for g in groups], float)
g_plus = np.array([(rows_by_group[g]["label"] == 2).sum() for g in groups], float)
g_src = np.array([sources_sorted.index(rows_by_group[g]["source"].iloc[0]) for g in groups])
total_images = g_images.sum()

print(f"groups={n_groups}  metrics={len(metrics)}  images={int(total_images)}  fractions={FRACTIONS.tolist()}")


def cost(assign: np.ndarray) -> float:
    total = 0.0
    for s in range(3):
        mask = assign == s
        for _, values, weight in metrics:
            ssum = values.sum()
            coef = weight / max(FRACTIONS[s] * ssum, 1.0)
            total += coef * abs(values[mask].sum() - FRACTIONS[s] * ssum)
    return total


def stats(assign: np.ndarray) -> dict:
    out = {}
    for s, name in enumerate(SPLIT_NAMES):
        m = assign == s
        out[f"{name}_images"] = int(g_images[m].sum())
        out[f"{name}_plus"] = int(g_plus[m].sum())
        out[f"{name}_sources"] = int(np.unique(g_src[m]).size) if m.any() else 0
    out["degenerate"] = bool(out["val_plus"] == 0 or out["test_plus"] == 0
                            or min(out["train_sources"], out["val_sources"], out["test_sources"]) < 3)
    return out


# ---- recorded MILP assignment
group_split = df.groupby("group_id")["split"].first()
milp = np.array([SPLIT_NAMES.index(group_split[g]) for g in groups])
milp_cost, milp_stats = cost(milp), stats(milp)
print(f"\nMILP objective = {milp_cost:.6f}")
print(f"MILP stats     = {milp_stats}")

rng = np.random.default_rng(SEED)
uni, sia = [], []
b1, b2 = FRACTIONS[0] * total_images, (FRACTIONS[0] + FRACTIONS[1]) * total_images

for _ in range(N_RANDOM):
    a = rng.integers(0, 3, size=n_groups)
    uni.append((cost(a), stats(a)))

    perm = rng.permutation(n_groups)
    cum = np.cumsum(g_images[perm])
    pos = np.where(cum <= b1, 0, np.where(cum <= b2, 1, 2))
    b = np.empty(n_groups, dtype=int)
    b[perm] = pos
    sia.append((cost(b), stats(b)))


def summarise(name, pairs):
    costs = np.array([c for c, _ in pairs])
    extras = [e for _, e in pairs]
    return {
        "name": name,
        "n": int(len(costs)),
        "cost_mean": float(costs.mean()),
        "cost_median": float(np.median(costs)),
        "cost_min": float(costs.min()),
        "cost_p5": float(np.percentile(costs, 5)),
        "cost_p95": float(np.percentile(costs, 95)),
        "cost_max": float(costs.max()),
        "frac_random_at_least_as_good": float((costs <= milp_cost).mean()),
        "degenerate_rate": float(np.mean([e["degenerate"] for e in extras])),
        "val_plus_median": float(np.median([e["val_plus"] for e in extras])),
        "val_plus_min": int(min(e["val_plus"] for e in extras)),
        "test_plus_median": float(np.median([e["test_plus"] for e in extras])),
        "test_plus_min": int(min(e["test_plus"] for e in extras)),
        "val_images_median": float(np.median([e["val_images"] for e in extras])),
        "test_images_median": float(np.median([e["test_images"] for e in extras])),
    }


u, s = summarise("uniform_group_random", uni), summarise("size_aware_group_random", sia)

print("\n=== objective value: MILP vs random group-level splits ===")
print(f"{'MILP (recorded)':<26}: {milp_cost:.6f}")
for d in (u, s):
    print(f"{d['name']:<26}: mean={d['cost_mean']:.6f}  median={d['cost_median']:.6f}  "
          f"min={d['cost_min']:.6f}  p5={d['cost_p5']:.6f}  p95={d['cost_p95']:.6f}")
    print(f"{'':<26}  P(random <= MILP) = {d['frac_random_at_least_as_good']:.4f}   "
          f"degenerate = {d['degenerate_rate']:.4f}")

print("\n=== images per split (target: 6211 / 1328 / 1331) ===")
print(f"{'MILP':<26}: {milp_stats['train_images']} / {milp_stats['val_images']} / {milp_stats['test_images']}")
for d in (u, s):
    print(f"{d['name']:<26}: - / {d['val_images_median']:.0f} / {d['test_images_median']:.0f}  (medians)")

print("\n=== Plus images in val / test (MILP: 207 / 209) ===")
print(f"{'MILP':<26}: {milp_stats['val_plus']} / {milp_stats['test_plus']}")
for d in (u, s):
    print(f"{d['name']:<26}: median {d['val_plus_median']:.0f} / {d['test_plus_median']:.0f}   "
          f"worst-case {d['val_plus_min']} / {d['test_plus_min']}")

json.dump({
    "milp": {"objective": milp_cost, "stats": milp_stats},
    "baselines": [u, s],
    "n_random": N_RANDOM, "seed": SEED, "fractions": FRACTIONS.tolist(),
    "cost_function": "identical to src/data/prepare_split.py::split_by_patient",
    "note": ("every group-level assignment satisfies the leakage constraint, so the comparison "
             "is about balance and rare-class coverage only"),
}, open(OUT / "milp_vs_random.json", "w"), indent=2)
pd.DataFrame([u, s]).to_csv(OUT / "milp_vs_random.csv", index=False)
print(f"\n[done] -> {OUT}")
