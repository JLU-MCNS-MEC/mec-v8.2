# V8.3 Stage-2 Preparation Report

- Date: 2026-10-18
- Branch: `experiment/v83-spatial-foundation`
- HEAD: `77921ef` (M2 report); Stage-2 preparation files are not committed yet
- Status: **preparation complete; FULL TRAINING NOT STARTED**
- Final Test is sealed and was not used for tuning, caching, smoke or model selection.

## 1. Data design and actual generation

Equal bands:

| Split | Episodes | Short/Medium/Long | Trajectory seeds | Hard states | Samples | Disk |
|---|---:|---:|---|---:|---:|---:|
| Train | 600 | 200 / 200 / 200 | 1,000,000+ | 3,600 | 55,565 | 24.7 GB |
| Dev Val | 180 | 60 / 60 / 60 | 2,000,000+ | 1,080 | 17,234 | 7.7 GB |
| Final Test | 300 | 100 / 100 / 100 | 3,000,000+ | 0 | 26,355 | 11.7 GB |

- Total new data: **44.14 GB**; free disk after generation: **38 GB**.
- Generation time: Train 385.0 s, Dev Val 116.2 s, Final Test 179.8 s; **~11.3 min total**.
- Start-distance ranges:
  - Train: Short 151.8–249.8 m (mean 199.8), Medium 251.2–398.7 m (311.3), Long 400.3–549.4 m (465.3).
  - Dev Val: Short mean 198.7, Medium 322.0, Long 464.5 m.
  - Final Test: Short mean 198.5, Medium 321.5, Long 466.2 m.
- Arrival labels:
  - Train 1,800 positive / 53,765 negative; arrival mask included 14,441.
  - Dev Val 540 positive / 16,694 negative; included 4,381.
  - Final Test 300 positive / 26,055 negative; included 5,450 (not used for tuning).

### Seed and network-layout isolation

New layout seeds follow `820000 + seed_start*17 + episode_id`:
≈17.82M (Train), 34.82M (Dev Val), 51.82M (Final Test).

Audit compared new splits, old Train/Val/Val_hard/Test, and the V8.3 60-episode dev spec:

- Trajectory seed set sizes: 600 / 180 / 300 / 300 / 60 / 60 / 60 / 60.
- Network-layout set sizes: same.
- **Pairwise intersections: none** (0 trajectory and 0 layout overlaps).
- Final Test contains 300 episodes and is sealed:
  - `runs/v83_spatial_stage2/datasets/final_test_300/SEALED.json`
  - SHA256 per shard + stats; files read-only (`-r--r--r--`).

## 2. Existing data distribution and M1 Long failures

Old splits:

| Split | Short | Medium | Long | Total |
|---|---:|---:|---:|---:|
| old Train | 90 | 150 | 60 | 300 |
| old Val / Val_hard / Test | 18 | 30 | 12 | 60 each |
| V8.3 dev | — | — | — | 60 (mixed) |

M1 Long failures on the V8.3 60-episode dev set: **12/12 failed, all OSR=0**.

| Episode | Seed | Start (m) | Outcome | Steps | Min dist (m) |
|---|---:|---:|---|---:|---:|
| 9 | 900012 | 406.3 | premature_stop | 9 | 366.38 |
| 11 | 900016 | 441.5 | premature_stop | 47 | 217.26 |
| 12 | 900018 | 440.9 | premature_stop | 22 | 368.38 |
| 20 | 900028 | 500.7 | timeout | 400 | 261.53 |
| 22 | 900033 | 495.7 | timeout | 400 | 480.93 |
| 24 | 900036 | 422.9 | timeout | 400 | 138.73 |
| 27 | 900039 | 494.3 | premature_stop | 49 | 303.34 |
| 28 | 900042 | 540.0 | premature_stop | 23 | 442.26 |
| 33 | 900048 | 420.7 | premature_stop | 17 | 345.37 |
| 42 | 900057 | 462.1 | premature_stop | 80 | 67.19 |
| 52 | 900068 | 521.4 | premature_stop | 59 | 292.80 |
| 57 | 900073 | 448.1 | premature_stop | 39 | 304.19 |

- **9/12 are far premature stops** (67–442 m minimum distance); **3/12 are movement timeouts** (139–481 m minimum). No Long episode reached the 20 m radius.
- Old Train had only 60/300 Long episodes (20%); new Train has 200/600 (33.3%), and Dev Val has 60 Long (vs 12).
- The balanced Long data directly targets the underrepresented band, but far-stop calibration and long-distance navigation remain the two distinct failure modes.

## 3. Arrival far-negative retention (3%)

Old Train audit:

- Natural near (≤60 m): 4,769, all included.
- Natural far (>60 m): 20,124, only **616 included (3.06%)**.
- Hard samples: 1,800, all included.
- Far bins: 60–100 m 3.01%, 100–200 m 2.97%, 200+ m 3.18%.

New Train 600 audit:

- Natural near: 9,511, all included.
- Natural far: 42,454, only **1,330 included (3.13%)**.
- Hard: 3,600; positives: 1,800.

Proposed controlled far-negative ablation (same tokens/cache/seed/budget; only the arrival mask/sampler changes; no modification of the old V8.2 protocol):

| Variant | Far-negative retention | Included far samples |
|---|---:|---:|
| A0 (old behavior) | 3% | ~1,330 |
| A1 | 10% | ~4,245 |
| A2 | 25% | ~10,614 |
| A3 distance-stratified | 10% each far bin | 808 / 1,634 / 1,804 |

Evaluate Natural/Hard Arrival-F1, far-40/far-100 false-stop rates, and closed-loop Dev Val SR/OSR/Premature-Stop/Timeout. Because M1's false stops occur on **policy-visited** states, also propose an off-policy negative axis (far negatives sampled from M1 rollouts) rather than relying only on expert-state negatives.

## 4. M1 spatial fusion head direction

Current `SpatialFusionHead` uses `AdaptiveAvgPool2d(1)`: global average removes left/right location.

- Symmetric left/right blob test:
  - current pooled-feature max difference: **7.45e-9** (direction-invariant);
  - proposed `AdaptiveAvgPool2d((2,2))` feature difference: **0.0181** (direction-sensitive).
- Parameter cost: current head 2,675; proposed head 2,819; **+144 params**.
- M1 total: 2,724,839 → **2,724,983** for the minimal M1-Dir variant.
- Recommendation: keep the original M1 baseline for the first Stage-2 run; add M1-Dir as a separate later ablation (2×2 pooling preserves coarse horizontal/vertical direction).

## 5. Stage-2 smoke verification

- Stage-2 spatial cache: 398 MB; Train / Val / Val_hard aligned:
  - first-frame invalid 600/180/180;
  - forward-previous invalid 37,125 / 11,251 / 11,251;
  - turn-valid 14,240 / 4,723 / 4,723;
  - missing previous 0; duplicate keys 0.
- Transformer-M M1 single-batch on new Train:
  - loss 2.7157, finite, gradients finite;
  - params 2,724,839; base init identical to M0 (`max diff=0`);
  - maps `(8,4,24,24)`.
- Small M1 single-batch on new Train:
  - loss 1.8082, finite, gradients finite;
  - params 303,335; base init `max diff=0`.
- 2-episode closed-loop smoke on new Dev Val spec (existing M1 checkpoint, no training):
  - pipeline completed; tracker reset per episode; actual previous action used;
  - synchronized timing: map ~0.207 ms/step, model ~1.04 ms/step in the timed episode.

## 6. Planned experiments (paused before full training)

1. Backbones: **Transformer-M** and **Small**; first compare **M0 vs M1**.
2. Same backbone: same seed, shard/loader order, loss, checkpoint score, LR and batch for M0/M1.
   - Transformer-M: lr 6e-5, batch 48, original seed 20264017.
   - Small: lr 1e-4, batch 64, original seed 20261017.
3. Max 80 epochs, early-stopping patience 10 on the existing 70/30 checkpoint score.
4. Stage 2a M0/M1 on Train 600 / Dev Val 180.
5. Stage 2b controlled Arrival far-negative A0/A1/A2/A3 if Stage 2a confirms far-stop calibration as a bottleneck.
6. Stage 2c M1-Dir (2×2 pooling) as a separate direction-preserving head ablation.
7. Final Test 300 remains sealed; no threshold tuning on Dev Val; final report must use a future held-out final split.

### Estimated training cost

- New Train+Dev Val sample count 72,799 vs old 37,211; prior M0/M1 epoch ≈21 s → new ≈41 s/epoch.
- Best epoch + patience ~60 epochs → **~40–45 min per Transformer-M M0/M1**, **~80–90 min for M0+M1**; if all 80 epochs run, ~55 min/model.
- Small M1 single-batch is much cheaper; preliminary estimate **~25–40 min per model**, **~50–80 min for M0+M1**, to be confirmed before full training.
- Disk: data 44.1 GB + cache 0.4 GB; 38 GB free; training checkpoints/history are small.

## 7. Artifacts and safety

- Data: `runs/v83_spatial_stage2/datasets/` (Train 600, Dev Val 180, sealed Final Test 300).
- Audit: `runs/v83_spatial_stage2/prep_audit_20261018T043420Z/` (`stage2_audit.json`, `m1_long_failures.csv`, `dev_val_180_spec.json`).
- Cache: `runs/v83_spatial_stage2/cache_20261018T043434Z/`.
- Smoke: `runs/v83_spatial_stage2/smoke_m1_20261018T043506Z/`, `runs/v83_spatial_stage2/smoke_small_m1_20261018T043538Z/`, `runs/v83_spatial_stage2/closedloop_smoke_20261018T043506Z/`.
- Old checkpoints/datasets/results unchanged; no M2/Depth/MEC BEV; no full training started.
- Stage-2 preparation source/test/docs are committed with the Stage-2a change; the optional `--train-dir/--val-dir/--val-hard-dir/--val-specs` CLI additions do not change old defaults.
