# V8.3 M2 Temporal Visual Reasoning — Training and Closed-Loop Report

- Date: 2026-10-18
- Branch: `experiment/v83-spatial-foundation`
- Source commit at training time: `9d635b2` (source fixes committed before training)
- Scope: **Transformer-M M2 only**. No Small model, no full five-MEC retraining, no Depth, no MEC BEV.
- M0/M1 reference run: `runs/v83_spatial_training/run_20261018T023830Z/`
- M2 run: `runs/v83_spatial_training/m2_run_20261018/`
- 60-episode evaluation: `runs/v83_spatial_training/m2_eval_60ep_20261018/`

This is a single-seed, 60-episode development-validation experiment. The 60 episodes
have already been used for V8.3 threshold work and are **not** an untouched final test.
Results are exploratory; correlated timesteps are not independent trials.

## 1. Pre-training verification

- `git status` clean before training; branch up to date with `origin/experiment/v83-spatial-foundation`.
- Unit tests: **25/25 OK** (`v83_spatial_training`, visual grounding, MEC geometry, threshold pipeline, depth evidence, TensorBoard logging). `compileall` OK.
- Single-batch M2 forward/backward: loss finite, gradients finite, spatial maps `(8, 4, 24, 24)`, base initialization exactly equal to M0 (`max diff 0.0`).
- 2-episode closed-loop smoke: OK. Tracker is re-created per episode; the first frame has no temporal prior; the previous action passed to the tracker is the **actual executed** action.
- Existing cache reused; **no Train/Val data regeneration**.

### Cache alignment audit

`python v83_spatial_training/verify_m2_alignment.py` independently rebuilt the
validity mask from archived labels and re-derived the turn warps from the cached
previous coherent maps.

| Split | Samples | Cache valid | Expected valid | Mask mismatches | Invalid temporal != coherent | Invalid innovation != 0 | Turn warp mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 26,693 | 7,100 | 7,100 | 0 | 0 | 0 | 0 |
| Val | 5,070 | 1,501 | 1,501 | 0 | 0 | 0 | 0 |
| Val_hard | 5,448 | 1,546 | 1,546 | 0 | 0 | 0 | 0 |

- Validity is 1 only for natural samples with `step_id > 0` whose previous archived
  label (same episode, `step_id-1`) is LEFT or RIGHT.
- First frames (300 / 60 / 60), FORWARD-previous samples (17,493 / 3,509 / 3,482)
  and hard independent samples (1,800 / 0 / 360) are invalid, matching the cache manifest.
- Valid Temporal24/Innovation24 match the cached arithmetic to float16 storage
  precision (max abs error 3.9e-3, mean ~1e-7).
- Token-level spot checks reproduce the cached baseline/coherent maps from RGB tokens
  and confirm the previous coherent map comes from the `step_id-1` frame.
- Training uses the archived previous **expert** action (teacher forcing). Closed-loop
  evaluation uses the model's **actual executed** previous action. No future action and
  no Goal truth/position/mask enters Temporal24 or Innovation24.

Artifact: `runs/v83_spatial_training/m2_alignment_20261018.json`.

## 2. M2 architecture and training protocol

M2 = M1 + `temporal24` and `innovation24` with an explicit temporal-validity mask.
The fusion head receives 5 channels: baseline24, coherent24, masked temporal24,
masked innovation24, and the validity mask. Network 5D and the Arrival Head are
unchanged across M0/M1/M2.

| Model | Head input | Parameters | Extra vs M0 | Extra vs M1 |
|---|---|---:|---:|---:|
| M0 | original V8.2 | 2,722,164 | — | — |
| M1 | baseline + coherent | 2,724,839 | +2,675 (+0.0983%) | — |
| M2 | baseline + coherent + temporal + innovation + mask | 2,725,271 | +3,107 (+0.1141%) | +432 (+0.0159%) |

Protocol: seed `20264017`; batch 48; AdamW lr 6e-5, weight decay 1e-4;
ReduceLROnPlateau (patience 3, factor 0.5); max 80 epochs; early-stopping patience 10;
same shard order, loader seeds, loss, class weights and 70/30 checkpoint score as M0/M1.

- Best epoch: **33** (checkpoint score 71.20%)
- Early stop: epoch 43
- Training time: **909.3 s = 15.2 min** (one run, RTX 5090)
- The M2 run wrote only its own `best.pt`; archived M0/M1 and V8.2 weights were not modified.

Provenance and hashes: `runs/v83_spatial_training/m2_run_20261018/provenance.json`;
checkpoint sha256 `c412ed50742da2e50ca76edf25b2eabdf741e4f5a435f47aaad48cb340977206`.

## 3. Offline validation at the best checkpoint

`Movement Macro-F1 (3-way)` is recomputed over Forward/Left/Right only. The
`4-way macro-F1` includes Stop and is the checkpoint-selection movement term.
`Directional BA` is the mean recall over the three movement actions.

| Model | Movement Macro-F1 (3-way) | Macro-F1 (4-way) | Directional BA | Natural Arrival-F1 | Hard Arrival-F1 | Checkpoint score |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 64.44% | 62.85% | 65.29% | 59.26% | 73.89% | 63.97% |
| M1 | 65.11% | 63.16% | 67.57% | 58.54% | 75.12% | 64.26% |
| M2 | **82.13%** | **74.26%** | **82.22%** | **52.75%** | 75.36% | **71.20%** |

M2 substantially improves offline movement classification and directional balance,
but its **natural Arrival-F1 decreases** by 5.8–6.5 points versus M0/M1. This is the
first warning sign for closed-loop stop behavior.

## 4. Closed-loop 60-episode development validation (Stop threshold = 0.5)

All three models were evaluated in one run on the same 60 episode seeds, the same
network layout seeds, the same fixed Stop threshold, and a per-episode tracker reset.
M0/M1 were read from the archived reference run; M2 from its own run. The M0/M1
SR/OSR/SPL/nDTW/premature-stop/timeout values reproduce the archived evaluation
**exactly**, validating the pipeline.

| Metric | M0 | M1 | M2 |
|---|---:|---:|---:|
| SR | 16.67% (10/60) | **21.67% (13/60)** | 15.00% (9/60) |
| OSR | 16.67% (10/60) | **25.00% (15/60)** | 16.67% (10/60) |
| SPL | 0.1667 | **0.2167** | 0.1500 |
| nDTW | 0.2180 | **0.2288** | 0.1688 |
| Premature-Stop | 55.00% | 51.67% | **76.67%** |
| Timeout | 28.33% | 26.67% | **8.33%** |
| Mean steps | 160.2 | 153.5 | 91.5 |
| Mean min distance (m) | 149.6 | 151.0 | 146.3 |
| Model ms/step | 0.958 | 1.022 | 1.048 |
| Spatial map ms/step | 0.005 | 0.200 | 0.454 |
| Total ms/step | 0.963 | 1.222 | 1.501 |

Latency uses `torch.cuda.synchronize()` around both timed segments, excludes the first
episode as warm-up, and reports per-step means over 59 episodes (p95 total:
M0 1.004, M1 1.281, M2 1.527 ms). These absolute values are **not comparable** to the
earlier unsynchronized ~1.2 ms numbers. M2 is not faster: the temporal warp adds
~0.25 ms/step over M1.

### SR / OSR by distance band

| Model | Short (18) SR / OSR | Medium (30) SR / OSR | Long (12) SR / OSR |
|---|---|---:|---:|
| M0 | 16.67% / 16.67% | 23.33% / 23.33% | 0% / 0% |
| M1 | 22.22% / 27.78% | 30.00% / 33.33% | 0% / 0% |
| M2 | **33.33% / 33.33%** | **10.00% / 13.33%** | 0% / 0% |

M2 gains on short episodes but collapses on medium episodes; long episodes remain
unsolved for all models.

## 5. Paired per-episode comparisons

Each row compares the two models on the same 60 episode indices. For Premature-Stop
and Timeout, **fewer** is better; `Better` therefore counts fewer occurrences. Deltas
are `other - base` for the raw rate (so a negative premature-stop delta is favorable).
95% CIs are paired percentile bootstrap intervals over episodes (10,000 resamples);
p-values are exact two-sided sign tests on discordant pairs.

### M2 vs M1

| Metric | M2 better | M2 worse | Ties | Delta (M2−M1) | 95% CI | Sign-test p |
|---|---:|---:|---:|---:|---:|---:|
| SR | 3 | 7 | 50 | −6.67 pp | [−16.67, +3.33] pp | 0.3438 |
| OSR | 3 | 8 | 49 | −8.33 pp | [−18.33, +1.67] pp | 0.2266 |
| Premature-Stop | 3 | 18 | 39 | +25.00 pp | [+11.67, +38.33] pp | **0.0015** |
| Timeout | 13 | 2 | 45 | −18.33 pp | [−30.00, −6.67] pp | **0.0074** |
| SPL | 3 | 7 | 50 | −0.0667 | [−0.1667, +0.0333] | 0.3438 |
| nDTW | 19 | 32 | 9 | −0.0600 | [−0.1324, +0.0082] | 0.0919 |

### M2 vs M0

| Metric | M2 better | M2 worse | Ties | Delta (M2−M0) | 95% CI | Sign-test p |
|---|---:|---:|---:|---:|---:|---:|
| SR | 5 | 6 | 49 | −1.67 pp | [−13.33, +8.33] pp | 1.0000 |
| OSR | 6 | 6 | 48 | 0.00 pp | [−11.67, +11.67] pp | 1.0000 |
| Premature-Stop | 3 | 16 | 41 | +21.67 pp | [+8.33, +35.00] pp | **0.0044** |
| Timeout | 14 | 2 | 44 | −20.00 pp | [−31.67, −8.33] pp | **0.0042** |
| SPL | 5 | 6 | 49 | −0.0167 | [−0.1333, +0.0833] | 1.0000 |
| nDTW | 24 | 25 | 11 | −0.0492 | [−0.1348, +0.0366] | 1.0000 |

The SR/OSR differences versus M1 are not statistically resolved on n=60, but their
point estimates are negative. The **increase in premature stopping is the clear,
consistent paired effect** (M2 worse in 18 episodes vs M1; +25 pp; p=0.0015).

## 6. Why M2 fails closed loop

The per-episode traces identify a stop-head failure rather than a movement failure:

- M2 gains successes on short episodes 2, 5, 36 and medium episode 48 by reaching the
  goal in fewer steps (e.g., episode 2: 400-step timeout under M1 → success in 68 steps under M2).
- M2 loses medium episodes that both M0 and M1 solve: 3, 18, 29, 37, 40, 49, 50.
  In these it stops after 14–112 steps at minimum distances of 40–220 m, i.e. a
  **far premature stop**, not a failure to find the goal.
- M2's 46 premature stops have median minimum distance 129.6 m; 45 of 46 are >40 m
  from the goal. Its mean episode length is 91.5 steps versus 153.5 for M1.
- Offline, M2's natural Arrival-F1 is 52.75% versus 58.54% (M1) / 59.26% (M0), and its
  far-40 false-stop rate is 1.19% versus 0.879% / 0.835%. The 70/30 checkpoint score
  rewards the large movement Macro-F1 gain and selects an epoch whose arrival head is
  less calibrated for closed-loop states.
- Better movement classification changes the state distribution visited by the policy.
  The Arrival Head, trained on expert trajectories, is then evaluated off-policy and
  fires false stops far from the goal. Temporal correlation information improves the
  movement branch but does not fix this stop/calibration bottleneck.

## 7. Decision

**M2 does not pass the pre-registered decision gate.** Relative to M1 it does not
improve paired reachability: SR 21.67% → 15.00% (−6.67 pp), OSR 25.00% → 16.67%
(−8.33 pp), SPL 0.2167 → 0.1500, and Premature-Stop 51.67% → 76.67% (+25 pp,
p=0.0015). Relative to M0, SR is −1.67 pp and OSR is unchanged while premature stops
increase by 21.67 pp (p=0.0044). Offline movement Macro-F1 improves dramatically
(64.44% → 82.13%, 3-way), but this does not translate into closed-loop success.

Therefore:

1. **Do not expand to Small or all five MEC models** on the basis of M2.
2. Keep M2 artifacts for analysis, but treat M1 as the current best spatial variant.
3. Focus the next experiment on **stop calibration / checkpoint selection and
   off-policy arrival supervision**, not additional movement capacity. The failure
   mode is a far false stop after a better movement policy reaches new states.
4. Long-distance and invisible-goal navigation remain unsolved for M0/M1/M2;
   off-policy recovery data is still needed.
5. Do not tune the Stop threshold on these 60 development episodes. A future stop
   gate/calibration should be selected on an independent validation split and then
   confirmed on a held-out final set.

## 8. Artifacts and reproduction

All large artifacts are under Git-ignored `runs/`:

```
runs/v83_spatial_training/m2_run_20261018/
  provenance.json
  summary.json
  Transformer-M/M2/{best.pt, history.csv, metrics.json}

runs/v83_spatial_training/m2_eval_60ep_20261018/
  per_episode.csv
  summary.json
  comparison.json
  comparison.md

runs/v83_spatial_training/m2_alignment_20261018.json
runs/v83_spatial_training/m2_train_20261018.log
runs/v83_spatial_training/m2_eval_60ep_20261018.log
```

Reproduction:

```bash
cd /home/ubuntu5/Workspace/MEC-V8.2
source /home/ubuntu5/miniconda3/etc/profile.d/conda.sh
conda activate DRL

CACHE=runs/v83_spatial_training/cache_20261018T023637Z
BASE=runs/v83_spatial_training/run_20261018T023830Z
M2RUN=runs/v83_spatial_training/m2_run_20261018

python v83_spatial_training/verify_m2_alignment.py --cache-dir "$CACHE"
python v83_spatial_training/train_spatial_mec.py \
  --models M2 --cache-dir "$CACHE" --output-dir "$M2RUN" \
  --seed 20264017 --max-epochs 80 --patience 10
python v83_spatial_training/eval_closedloop_spatial.py \
  --model-dir "$M2RUN" --reference-model-dir "$BASE" \
  --modes M0 M1 M2 --max-episodes 60 --threshold 0.5 \
  --output-dir runs/v83_spatial_training/m2_eval_60ep_20261018
python v83_spatial_training/summarize_m2_results.py \
  --base-dir "$BASE" --m2-dir "$M2RUN" \
  --eval-dir runs/v83_spatial_training/m2_eval_60ep_20261018
```

No weights, caches or logs are committed.
