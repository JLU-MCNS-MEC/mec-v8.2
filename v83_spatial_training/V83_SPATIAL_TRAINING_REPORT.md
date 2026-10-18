# V8.3 Spatial MEC Training Report — Transformer-M M0/M1

- Date: 2026-10-18
- Branch: `experiment/v83-spatial-foundation`
- Code HEAD at run time: `8559a6e`; new V8.3 training code committed with this report
- Model: Transformer-M, single seed `20264017`
- No original V8.2 training entry was modified. No five-MEC expansion. No Depth, no MEC BEV.

## 1. Pre-training validation

- All related CPU tests: **36/36 OK** (18 existing + 6 depth probe + 7 spatial diagnostic + 5 spatial training).
- `compileall`: OK.
- Single-batch forward/backward for M0/M1: finite loss and finite gradients.
- 2-episode closed-loop smoke: OK (vision state reset per episode; actual previous action used).
- 1-epoch full preflight: M0 epoch-1 loss `1.32390`, val macro-F1 `0.20767`, matching the archived V8.2 history (`1.32390`, `0.20767`), confirming deterministic reproduction.
- M0/M1 base initialization is identical (max diff 0.0); M1 fusion head is zero-initialized so its epoch-1 behavior matches M0 exactly.

## 2. Spatial feature cache

Cache keys: archived `(episode_id, step_id)`; previous action: archived label at `step_id-1` of the same episode. No future action and no Goal truth/position/mask is used.

Full cache alignment:

| Split | Samples | Natural | Hard | First-frame invalid | Forward-prev invalid | Turn-prev valid | Missing prev | Duplicate keys |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 26,693 | 24,893 | 1,800 | 300 | 17,493 | 7,100 | 0 | 0 |
| Val | 5,070 | 5,070 | 0 | 60 | 3,509 | 1,501 | 0 | 0 |
| Val_hard | 5,448 | 5,088 | 360 | 60 | 3,482 | 1,546 | 0 | 0 |

Temporal valid = 1 only for natural samples with a LEFT/RIGHT previous action. First frames, hard independent samples, missing previous states, and FORWARD previous actions have valid=0 and fall back to coherent24 with zero innovation.

## 3. Models and training protocol

| Model | Input to Movement Head | Parameters | Extra vs M0 |
|---|---|---:|---:|
| M0 | original V8.2 inputs | 2,722,164 | — |
| M1 | M0 + learnable fusion of baseline24 + coherent24 | 2,724,839 | +2,675 (0.098%) |
| M2 | M1 + temporal24 + innovation24 with validity mask | implementable | not trained |

- Same seed `20264017`, same shard/loader shuffle order, batch size 48, AdamW 6e-5, weight decay 1e-4, ReduceLROnPlateau (patience 3), same movement/arrival loss and 70/30 checkpoint score.
- Max 80 epochs, early stopping patience 10 on the V8.2 checkpoint score; best checkpoint saved per model.
- M2 was **not trained**. The code/cache support it, but the M1 signal is not robust enough to auto-expand (see decision).

## 4. Offline results (best epochs)

| Metric | M0 | M1 | Δ (M1−M0) |
|---|---:|---:|---:|
| Best epoch | **48** | **51** | — |
| Checkpoint score | 0.63971 | 0.64259 | +0.00288 |
| Val accuracy | 0.72071 | 0.71341 | -0.00730 |
| Val balanced4 | 0.68967 | 0.70676 | +0.01709 |
| Val directional balanced | 0.65290 | 0.67567 | +0.02277 |
| Movement Macro-F1 | 0.62854 | 0.63158 | +0.00304 |
| Natural Arrival-F1 | 0.59259 | 0.58537 | -0.00722 |
| Hard Arrival-F1 | 0.73892 | 0.75122 | +0.01230 |
| Hard negative false-stop | 3.33% | 3.33% | 0.00% |
| Far-40 false-stop | 0.835% | 0.879% | +0.044% |
| Near-20 miss-stop | 20.0% | 20.0% | 0.00% |

Offline gains are small; M1 improves movement macro-F1/directional balance and hard-arrival F1, but slightly reduces natural-arrival F1. This alone would not justify expansion.

Training time (single run, Transformer-M):

- M0: **1,214.7 s = 20.2 min** (best epoch 48; early stop around epoch 58)
- M1: **1,292.7 s = 21.5 min** (best epoch 51; early stop around epoch 61)
- Total: **2,507.4 s = 41.8 min**

## 5. Closed-loop development-validation results

60 V8.3 development-validation episodes; Stop threshold fixed at 0.5 for both models; visual state reset per episode; M1 uses actual executed previous actions only where relevant (M1 is static; M2 would use temporal).

| Metric | M0 | M1 | Δ |
|---|---:|---:|---:|
| SR | 16.67% (10/60) | **21.67% (13/60)** | **+5.00 pp** |
| OSR | 16.67% (10/60) | **25.00% (15/60)** | **+8.33 pp** |
| Premature-Stop | 55.00% | 51.67% | -3.33 pp |
| Timeout | 28.33% | 26.67% | -1.67 pp |
| SPL | 0.1667 | **0.2167** | +0.0500 |
| nDTW | 0.2180 | 0.2288 | +0.0108 |
| Model ms/step | 1.313 | 1.043 | -0.270 |
| Spatial-map ms/step | 0 | 0.197 | +0.197 |
| Total inference ms/step | 1.313 | 1.240 | -0.073 |

Paired by episode:

- Success: M1 better **3**, M0 better **0**, ties 57; delta +5.00 pp; exact sign-test p = 0.25.
- OSR: M1 better **5**, M0 better **0**, ties 55; delta +8.33 pp; exact sign-test p = 0.0625.
- Premature-Stop: M1 better 3, M0 better 5, ties 52; delta -3.33 pp.
- Timeout: M1 better 4, M0 better 5, ties 51; delta -1.67 pp.
- SPL delta +0.0500.

M1 gained successes on episodes 5, 37, 56; gained OSR on 5, 34, 37, 48, 56; no episode lost success or OSR. By band:

| Band | M0 SR/OSR | M1 SR/OSR |
|---|---|---:|
| Short (18) | 16.7% / 16.7% | 22.2% / 27.8% |
| Medium (30) | 23.3% / 23.3% | 30.0% / 33.3% |
| Long (12) | 0% / 0% | 0% / 0% |

The gain is concentrated in short/medium episodes; long episodes remain unsolved.

## 6. Decision: is M1 a substantial improvement?

**M1 is directionally promising but not yet statistically robust.** It gains 3 SR episodes and 5 OSR episodes with zero paired losses, and improves SPL/nDTW. However:

- The gains are 3–5 episodes on a 60-episode development set that was already used for threshold tuning;
- Exact sign-test p = 0.25 for SR and 0.0625 for OSR;
- Offline checkpoint score improves only +0.0029, and natural-arrival F1 slightly decreases;
- M2 and Small are not trained in this run; the explicit authorization was the first Transformer-M M0/M1 group.

Therefore, **M2 was not started automatically**, and no five-MEC expansion occurred. A confirmation/expanded validation is recommended before treating the +5 pp SR / +8.3 pp OSR as a reproducible spatial-reasoning gain. If this signal is accepted, the next authorized step is M2 (temporal24 + innovation with validity mask), followed by Small only after M2.

## 7. Artifacts

Training run (Git-ignored):

`runs/v83_spatial_training/run_20261018T023830Z/`

- `Transformer-M/M0/best.pt`, `history.csv`, `metrics.json`
- `Transformer-M/M1/best.pt`, `history.csv`, `metrics.json`
- `summary.json`

Closed-loop:

`runs/v83_spatial_training/run_20261018T023830Z/closedloop_20261018T032039385722Z/`

- `per_episode.csv`
- `summary.json`

Cache:

`runs/v83_spatial_training/cache_20261018T023637Z/`
- `train.npz`, `val.npz`, `val_hard.npz`, `cache_manifest.json`

Training log: `runs/v83_spatial_training/train_20261018T023830Z.log`
Closed-loop log: `runs/v83_spatial_training/closedloop_20261018T032038Z.log`

No weights/caches are committed. The original V8.2 entry, checkpoints, datasets, and historical results are unchanged.

## 8. Next steps

1. Decide whether the paired +5 pp SR / +8.3 pp OSR (zero losses, p 0.25/0.0625) is sufficient to authorize M2.
2. If yes: train M2 with the prepared temporal cache (same seed/data/budget), then run the same 60-episode paired closed-loop evaluation.
3. If M2 also improves: run Small M0/M1/M2 under the same protocol.
4. If not: improve coherent long-band/near-goal localization and consider a distance/hysteresis Stop gate before expanding model capacity.
5. Keep the 60 episodes as development validation; select a final test split before claiming final SR/OSR.
