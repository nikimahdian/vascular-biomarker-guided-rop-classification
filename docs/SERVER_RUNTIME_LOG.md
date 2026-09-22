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

---

## Task 9 — joint RGB-vessel learned fusion (H / I)

Server 2, RTX 4090, isolated workspace `/root/niki_rop_task6_isolated`, script
`task9_joint_fusion.py`, `nohup` under the project venv. Whole chain (pre-flight -> H training ->
selection -> I training -> selection -> freeze -> single TEST read -> 4 x 10,000-replicate paired
bootstrap -> figures -> SHA manifest) finished in 1,418 s wall, about 24 minutes.

Throughput: 40.3 s per stage-1 epoch for H, rising to 60-70 s per stage-2 epoch once the encoder
tails became trainable (gradients through the last two EfficientNet blocks of both a B5 and a B4).
GPU memory peaked around 4.1 GiB of 24,564 MiB at batch 16, so the frozen batch size was never a
constraint and was left alone. CUDA bfloat16 autocast was active throughout
(`torch.cuda.is_bf16_supported()` is True on this driver) and no loss scaling was needed.

Both H and I ran to early stop: H 10 epochs, I 12 epochs. Validation loss was lowest at epoch 1 for
both, i.e. before any encoder was unfrozen, and validation AUC peaked two epochs after unfreezing
and then decayed while training loss kept falling monotonically. This is the same overfitting
pattern Task-8 model D showed, reproduced harder with two encoders.

Two operational notes:

* The interrupted-network failure mode from Task 8B recurred once: an `ssh` call timed out at
  connect and the upload never happened. Re-running with an explicit `ConnectTimeout=30` and
  checking `$LASTEXITCODE` before the next chained step caught it immediately; the retry succeeded
  and the script was compiled with `py_compile` on the server before launch.
* The `JointDS.__getitem__` path draws the geometric augmentation exactly once per sample and
  applies the identical crop box, flip flag and rotation angle to the RGB image and the mask, with
  BILINEAR for RGB and NEAREST for the mask. That is the only way to keep a mask registered to its
  image; a second independent draw would silently misalign vascular morphology from appearance.

**Manifest timing lesson (found while auditing Task 8 during Task 9).** In the original Task-8 run
the SHA manifest was computed before the process wrote its final two lines: the
`TASK8_STATUS = COMPLETE` entry in `task8_progress.log` and the metric echo that the `nohup`
redirect sent to `train.log`. Re-hashing those two files therefore reports a mismatch even though
nothing was altered — they are append-only execution logs of the very run that hashed them. Rule
going forward: write the artifact SHA manifest as the absolute last action, after every log write,
or leave logs out of the manifest.

---

## Task 10 — frozen biomarker-conditioned FiLM fusion (J0 / J1)

Server 2, RTX 4090, script `task10_biomarker_film.py`. This task runs no CNN at all: both encoders
exist only as frozen embedding tables (8862 x 2048 RGB, 8862 x 1792 vessel), so the whole chain -
pre-flight, J0 training, selection, J1 training, selection, freeze, single TEST read, FiLM
diagnostics, per-biomarker ablation, source breakdown, three 10,000-replicate paired bootstraps and
figures - finished in **69 seconds**.

Each epoch is 0.4-0.6 s: 6,203 samples at batch 128 is 49 optimizer steps over a 2.5 M-parameter
MLP, and the validation pass is a single 1,328-row forward. There is no data-loader bottleneck and
no GPU-capacity question; memory use is negligible. The 10,000-replicate bootstraps dominate the
wall clock at ~18 s each.

Overfitting is the same story as Task 9 but at the head level: train loss falls monotonically to
0.046 while validation loss climbs from its epoch-1 minimum of 0.687 to 1.87. Freezing the encoders
into precomputed features removed catastrophic encoder drift, but a 2.5 M-parameter head still has
far more capacity than 6,203 samples support, and both models selected epoch 1.

Two implementation notes worth keeping:

* The matched-pair design has a subtle defect that was found by reading the code rather than the
  results. `Fusion.__init__` builds the FiLM modules *before* `self.fuse`, so for J1 the extra
  `nn.Linear` constructions consume RNG draws and the fusion MLP starts from different weights than
  J0's. Architecture, data, seed, optimizer and batch order are identical, so the pairing holds for
  everything except fusion-head initialisation. Because FiLM turned out empirically near-identity,
  that difference is a sufficient explanation for the small J1-vs-J0 Brier gap, and the gap must not
  be read as a biomarker effect. Fix for a future replication: re-seed immediately before building
  the fusion MLP so both arms start bit-identically.
* The Task-8/9 manifest-ordering problem was designed out here: `task10_progress.log` and `train.log`
  are marked `EXCLUDED_APPEND_ONLY_LOG` in `artifact_sha256.json` instead of being hashed before the
  final log write. Server-1/Server-2 equality is verified by direct cross-server hashing rather than
  by trusting the manifest.

---

## Task 11 — source-held-out (LOSO) baseline

Server 2, CPU only, script `task11_source_heldout.py`, nine XGBoost fits (3 folds x B/E/G) over the
frozen 2048-d RGB and 1792-d vessel embeddings. Whole chain - pre-flight, three folds, six
10,000-replicate paired bootstraps, shift diagnostics, biomarker shift, figures - finished in
**542 seconds**.

Cost profile: a B_LOSO fit (2048 features) is ~26 s, an E_LOSO or G_LOSO fit (3840/3845 features)
is ~50-65 s with `n_jobs=16` and `tree_method=hist`. The largest fold trains on 7,453 rows. The
paired bootstraps dominate again at ~33 s each because `multiclass_auc` is recomputed on every
replicate.

Two things the folds forced into the open:

* The `plus` fold is not a three-class problem. That source has 5,925 images of which 5,304 are
  Normal and 621 are Plus, and it contains **zero** Pre_Plus. The script returns NaN for the
  three-class macro AUC there rather than substituting a two-class value, computes balanced accuracy
  and macro F1 over the classes actually present, and reports a separately labelled
  Normal-vs-Plus restricted AUC. Its 3-column Brier is inflated by a structurally empty Pre_Plus
  column, so a renormalised two-class Brier is reported alongside it.
* Every fold's training set does contain all three classes even though `plus` does not, because
  farabi and farfum_rop each carry Pre_Plus. The script asserts this before fitting so that a fold
  can never silently fall back to a two-class XGBoost objective.

The `plus` held-out row in the degradation table is the one number in this task that is easy to
misread: restricted two-class AUC on an 89.5 % Normal test set is numerically *higher* than the
canonical 0.93, which says nothing about transfer quality. It measures a different task and is
labelled as such everywhere it appears.

---

## Task 12 — class-conditional domain-invariant RGB-vessel representation (K0 / K1)

Server 2, RTX 4090, scripts `task12_cc_domain.py` (training) and `task12_eval_frozen.py` (held-out
evaluation pass). Frozen 3840-d input, no CNN, no biomarkers. Six models trained (3 folds x K0/K1)
in **62 seconds** - a 128-sample batch through the small MLP is ~0.02 s, so an epoch is 0.1 s for K0
and 0.9-2.0 s for K1, the extra time being the discriminator and the four-kernel MMD over up to
three classes. Evaluation, six 10,000-replicate bootstraps and figures added another 105 s.

The architecture was deliberately shrunk after Tasks 9 and 10 (256-d projections, 256-d shared
representation, a 3-way linear classifier) and that reduced the parameter count by roughly two
orders of magnitude against Task 9. Both models still selected an early checkpoint (epoch 3-4 of
about 10) and the source-validation-to-held-out AUC gap stayed large, so capacity was not the whole
story behind the earlier overfitting.

**A crash and how it was handled.** The first run trained and selected all six models, wrote
`task12_selection_frozen.json` with `HELDOUT_TARGETS_TOUCHED = NO`, and then died during held-out
evaluation with `KeyError: 2`. The cause was a stale loop variable: `pr[f"{kind}_{c}"]` reused `c`
from an earlier `for c in (0, 1, 2)` loop instead of a class name. The correct response was *not* to
retrain. A separate script `task12_eval_frozen.py` re-verifies all six checkpoint SHA-256 values
against the freeze manifest and then performs only evaluation, statistics and figures. All six
matched; nothing was retrained and no parameter was reselected. Worth remembering: a crash after the
freeze does not licence a fresh training run, and keeping checkpoints plus a hash manifest is what
makes a verify-only recovery possible.

---

## Task 13 — ROPDeepX-style dual-RGB soft attention (model L, M0, M1)

Server 2, RTX 4090. Frozen 3840-d input was not used for L - this task trains CNNs again, the first
time since Task 9. Whole chain (pre-flight, 7 epochs of L, selection, attention diagnostics,
8,862-sample attended-embedding extraction, two XGBoost fits, single TEST read, four
10,000-replicate paired bootstraps, figures) finished in **575 seconds**.

Cost profile: 28-32 s per stage-1 epoch when both backbones are frozen, rising to 35-37 s in stage 2
once ResNet50 `layer4` and the last EfficientNet-B4 block group become trainable. Peak GPU memory
observed was about 3.5 GiB of 24,564 MiB at batch 16 with two 384x384 backbones in the graph. The
staged freeze plus OneCycleLR kept the run to 7 epochs and a mild plateau instead of the blow-up seen
in Tasks 9 and 10: validation loss stayed inside 0.868-0.913 while train loss fell 0.7638 -> 0.5574.

Two results worth carrying forward:

* The vessel result is the most robust finding in the whole series. M1 - M0 AUC was +0.020936
  [+0.011426, +0.030828], p < 0.0001 - the largest vessel contribution measured anywhere, larger
  than Task 8B's +0.0079 for E - B. Spatial vascular morphology keeps adding information even after
  the RGB side is replaced by a two-backbone attention model.
* The ROPDeepX-style RGB representation itself did not help: M0 - B_EMBEDDING_ONLY AUC was
  -0.018925, p = 0.0063, i.e. significantly worse than the project's original single-backbone
  EfficientNet-B5 embedding on this cohort. Selection landed on a stage-1 checkpoint, so the
  fine-tuning stages never paid off on validation either.

Attention did not collapse (mean a_resnet 0.5335 validation, 0.5111 test, threshold 0.95) but it is
effectively near-binary per image - std ~0.39 with p05 ~0.01 and p95 ~0.99 - and it is strongly
source-dependent (mean a_resnet 0.71 on plus, 0.19 on farfum_rop, 0.04 on farabi). Reading the
attention as a clinical quantity would import an acquisition confound; it is reported as a
diagnostic only and was never used for selection.

The canonical split fingerprint is the SHA-256 of `data/splits/all.csv` (8,870 rows), which lives on
Server 1 only, so Server 1 recomputed it during this task and it matched. Server 2 re-verified the
population, split counts, label set and group disjointness against the 8,862-row complete-case
manifest. Class weights recomputed from canonical TRAIN alone came out at 0.46050 / 3.18103 /
1.94512 - identical to the frozen Task-6 weights, an independent confirmation that those were
inverse-frequency on this same train population.
