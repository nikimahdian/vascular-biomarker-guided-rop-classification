# Analysis-host runtime log

Every number is the wall-clock time the work actually occupied on the analysis host
(`moniaz@100.115.180.59`, Apple M2 Ultra, 24 CPU cores), taken from the recorded run logs in
`_private_audit/`. Times for runs that were discarded or superseded are listed separately so they
are not confused with the frozen results.

## Committed results

| task | what ran on the host | elapsed | source of the number |
|---|---|---|---|
| 5B-N2 (run 3) | 8,870-image A/V equivalence gate + the 360 × 26 perturbation battery, instrumented A/V, 2 versions per cell | **8,092 s = 2 h 15 m** | `task5b_n2_run3.log` (`elapsed 8092s`); started 10:09, log closed 12:24 |
| 5B-H (run 2) | 328 images × 30 conditions, FOV detection and the two density features only — no full `measure()` | **82 s = 1 m 22 s** | `task5b_h_run.log` |
| 5B-H2 (run 2) | 328 images × 31 conditions × full `measure()` with fractal, V1 instrumentation equivalence | **3,981 s = 1 h 6 m** | `task5b_h2_run.log` |
| 5B-H3 (locked run) | 450 images × 37 conditions × V1 + V2, full `measure()`; all data written, then the process crashed in the verdict block | **≈ 11,900 s = 3 h 18 m** | measured window 17:04 → 20:22 from `task5b_h3_run.log` and its mtime; no `elapsed` line because of the crash |
| 5B-H3 (re-analysis) | endpoints and gate re-derived from the saved CSVs + zero-padding control + 28-row failure-reason probe | **≈ 5 m** | not logged |
| 5B-H4 | 450 images: V3 on all 42 conditions, V1 and V2 on the 8-condition mechanism subset | **9,721 s = 2 h 42 m** | `task5b_h4_run.log` |
| 5B-H5 | adjudication only: pixel-mapping verification (408 checks) + 90-row mechanism probe, no measurement battery | **≈ 4 m** | not logged |
| 5B-H6 (locked run) | unit invariance + 150 images × 7 conditions × V3 + V4 | **1,213 s = 20 m** | `task5b_h6_run.log` |
| 5B-H7 (locked run) | unit invariance + H6 replay + 100 images × 7 conditions × V4 + V5 | **596 s = 10 m** | `task5b_h7_run.log` |
| 5B-H8 (§B) | independent H7 recheck from the saved rows | **≈ 5 s** | `task5b_h8_recheck.py` |
| 5B-H8 (§C–K) | full canonical V5 generation, 8,870 images, plus integrity, complete-case, drift, QC, validity and determinism | **≈ 2,885 s = 48 m** (in flight: 5,000 / 8,870 at 1,627 s) | `task5b_h8_run.log` |

**Total committed compute: ≈ 35,600 s ≈ 9 h 54 m** of host time, plus the in-flight V5 generation.

## Discarded or superseded runs

| task | what ran | elapsed | why it was discarded |
|---|---|---|---|
| 5B-N2 (runs 1–2) | a replica with the `np.clip(rgb, 0, 1)` scaling bug, then the exact-equality gate attempt | not logged separately | run 1 invalidated by the scaling bug; run 2 stopped at the predeclared exact-equality gate |
| 5B-H (run 1) | 178-image FOV battery | ≈ 5 m | sample below the required N ≥ 300 |
| 5B-H2 (run 1) | 178-image full `measure()` grid | **34 s** | sample below N ≥ 300; archived as `_private_audit/run1_*` |
| 5B-H3 (run 1) | sample ladder only, stopped at N = 300 < 400 | ≈ 16 m | no measurement was run |
| 5B-H6 (run 1) | unit tests + development check, stopped at N = 120 < 150 | ≈ 6 m | sample ladder too short |
| 5B-H7 (run 1) | unit tests + H6 replay, killed by SIGPIPE when the log was piped into `head` | ≈ 5 m | output truncation, not a code failure |
| 5B-H8 (first launch) | none | 0 | the `&&` chain stopped before launch on a wrong mask-manifest path |

## Why the costs differ so much between tasks

`retinal_fov()` is cheap (≈ 16 ms per call), while `measure()` computes all 69 columns — skeleton,
Euclidean distance transform, width, geodesic tortuosity, A/V branch table, rings, sectors, topology
and the multifractal estimator — at roughly 2 s of single-thread CPU per call, more on padded
canvases. A task's cost is therefore set almost entirely by how many full `measure()` calls it makes:

* 5B-H ran 9,840 **FOV-only** evaluations and finished in 82 s;
* 5B-H3 ran ≈ 33,300 **full** evaluations and needed 3 h 18 m;
* 5B-H8 runs 8,870 full evaluations plus generation bookkeeping and is on track for ≈ 48 m.

Everything ran on CPU. The GPU (60-core Metal) is only usable by the segmentation model, whose
output — `SEG_CURRENT_V1` — is frozen and reused from disk in all of these tasks, so no GPU work was
needed. Nothing here trains a model.

---

## Task 8 — two-server spatial vessel fusion (D / E / F / G)

Server 2 (Ubuntu 24.04, RTX 4090 24,564 MiB, driver 580.65.06, torch 2.14.0+cu130), isolated
workspace `/root/niki_rop_task6_isolated`. Script `task8_run.py` sha256
`da78251a09333ff1ae09f61314973b3d2efc0721803f8f9931422b7b4a79b22c`, launched with `nohup` under
the project venv. Whole chain (D training -> selection -> 8,862 vessel embeddings -> E -> G ->
F alpha on VAL -> freeze -> single TEST read -> paired E-vs-B bootstrap) finished in 726 s wall.

Throughput: EfficientNet-B4 mask branch at 52-57 s/epoch on 6,203 training masks (batch 16), i.e.
the same order as the frozen B_RGB reference of 65.7 s/epoch on this GPU. `pfm-vllm.service` was
inactive and the GPU was free (0 % util, 1 MiB) for the whole run; no OOM this time.

The mask preflight is the important two-server fact: the Server-2 runtime manifest carries rewritten
`image_path` values, and `mask_path` maps as `BASE2 + original_absolute_path` with
`BASE2 = /root/niki_rop_task6_isolated/server1_data`. With that mapping, **0 of 8,862 masks were
missing on Server 2**, so the D branch consumed exactly the same frozen
`SEG_CURRENT_V2_RESOLVER_SAFE` masks as the canonical measurement pipeline.

TEST was read once, only after `task8_selection_frozen.json` recorded `TASK8_ALL_SELECTION_FROZEN =
YES` together with the D checkpoint hash, the vessel-embedding hash, the E/G learner config and the
F alpha. `d_selection.json` carries `test_touched: false`.

Two operational notes worth keeping:

* The first launch died instantly because the output directory did not exist, so the shell could not
  open `task8_spatial_vessel_fusion/train.log` for the `nohup` redirect. `mkdir -p` first, then
  launch - the `nohup` redirect target must exist before the process starts.
* `pgrep -fc "s2_runner[.]py"` returned 0 before launch, confirming no stale Task-6/7 runner was
  holding GPU memory. The bracket pattern still matters: an unbracketed pattern matches the
  invoking shell itself.

Server 1 was used only as the artifact sink for this task; no training ran on the Mac.
