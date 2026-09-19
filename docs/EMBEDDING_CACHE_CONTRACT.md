# Embedding cache contract — `results/hybrid_v2/embeddings_current.npz`

Status: **documented retrospectively.** The cache was written without a row index or a written
contract; this file records what it actually is, so it can be used safely and rebuilt correctly.

## What the file contains

```python
np.load("results/hybrid_v2/embeddings_current.npz")["embeddings"]
# shape (8870, 2048), dtype float32
```

The archive has **exactly one key, `embeddings`**. It stores **no image paths, no group IDs and no
split labels.** The sidecar `embeddings_current.json` stores only aggregate identity:

```json
{
  "split_sha256": "0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8",
  "checkpoint_sha256": "ef5667d305fdf79a7c4cb1c8c981e797dfe09f46ce66f8d320235275cae28299",
  "n_rows": 8870,
  "image_order_sha256": "6b2e5699d3e2de738fd624e0c087bf2e99bd18a96426423530ba6034ea226550",
  "shape": [8870, 2048],
  "backbone": "efficientnet_b5"
}
```

## Row order — the recipe (this is the part that was missing)

Row *i* of `embeddings` corresponds to row *i* of a frame built from
`data/features/biomarker_features.csv` sorted by:

1. `split`, in the fixed order **train → val → test**
2. then `image_path`, ascending

```python
frame = pd.read_csv("data/features/biomarker_features.csv")
frame["_o"] = frame["split"].map({"train": 0, "val": 1, "test": 2})
frame = frame.sort_values(["_o", "image_path"]).reset_index(drop=True)
assert hashlib.sha256("\n".join(frame["image_path"]).encode()).hexdigest() \
       == "6b2e5699d3e2de738fd624e0c087bf2e99bd18a96426423530ba6034ea226550"
```

This recipe was recovered from `scripts/hybrid_v2_experiment.py` (lines ~237–244) and
`scripts/lowcost_fusion_suite.py` (lines ~813–826). Sorting by `all.csv` order instead gives a
**wrong** alignment: an embedding-only logistic read-out then scores AUC ≈ 0.72 on validation
instead of the correct 0.92.

**Verification that the alignment is correct** (all three must hold):

| Check | Expected | Observed |
|---|---|---|
| order hash | `6b2e5699…` | matches |
| embedding-only logistic (C = 0.001), **validation** AUC | ≈ 0.9208 | 0.920878 |
| Pearson r vs `results/branch_b_val_preds.csv` on validation | high | 0.9631 |

## Hard rule: do not cross-validate over train rows with this cache

The embeddings come from Branch B's backbone, which was **fine-tuned on the train rows**. The
representation has therefore already seen those images and their near-duplicates, so any
cross-validation restricted to train rows is contaminated. Measured with the identical model and
features:

| Evaluation rows | AUC |
|---|---|
| held-out **train** rows | 0.999424 (Plus source 1.000000) |
| held-out **val** rows | 0.924882 (Plus source 0.918329) |

AUC ≈ 1.000 on held-out train groups is memorisation, not an easy split.

**Consequences:**

* Read-out comparisons must be fitted on train and evaluated on val (or a genuinely held-out
  source). That is what `results/fusion_diagnostic_v1/phase2b_readout/` does.
* Any earlier analysis that selected or scored candidates on "train grouped OOF" scores computed
  from this cache is over-optimistic and must be re-derived. `lowcost_fusion_suite.py` flagged
  part of this ("Early heads use frozen B embeddings; not nested encoder CV"); this document
  quantifies it.

## If the cache is rebuilt

Store the paths with the matrix, and keep the existing fields:

```python
np.savez_compressed(
    path,
    embeddings=embeddings.astype(np.float32),
    image_path=frame["image_path"].to_numpy(dtype=object),   # <-- currently missing
)
```

Keep `n_rows`, `split_sha256`, `checkpoint_sha256`, `shape`, `backbone`, and keep
`image_order_sha256` as a cross-check. With `image_path` stored, alignment is self-describing and
the order recipe stops being tribal knowledge.
