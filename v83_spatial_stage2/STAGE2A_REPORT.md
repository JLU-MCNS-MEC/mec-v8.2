# V8.3 Stage-2a Transformer-M M0/M1 Report

- Date: 2026-10-18
- Branch: `experiment/v83-spatial-foundation`
- Scope: **Transformer-M M0/M1 only**. No Small, no M2, no Depth, no MEC BEV.
- Train: Stage-2 Train 600 (200/200/200); Dev Val: Stage-2 Dev Val 180 (60/60/60).
- Final Test 300: sealed; not read/evaluated/tuned.
- Training seed `20264017`, lr `6e-5`, batch `48`, max 80 epochs, early-stopping patience 10.
- Stop threshold for closed-loop evaluation: `0.5`; Network 5D and Arrival mask unchanged.

## 1. Entry/data/cache verification

- New Stage-2 paths passed explicitly to the existing V8.3 entry point:
  - `--train-dir runs/v83_spatial_stage2/datasets/train_600`
  - `--val-dir/--val-hard-dir runs/v83_spatial_stage2/datasets/dev_val_180`
  - `--cache-dir runs/v83_spatial_stage2/cache_20261018T043434Z`
- Stats:
  - Train: 600 episodes, bands 200/200/200, 55,565 samples.
  - Dev Val: 180 episodes, bands 60/60/60, 17,234 samples.
- Cache-key alignment:
  - Train: 55,565/55,565 keys, 0 missing/extra.
  - Dev Val: 17,234/17,234 keys, 0 missing/extra.
- Final Test not accessed in Stage-2a.
- No NaN, no CUDA OOM, no path/cache/model exception; strict-finite enabled.

## 2. Training results

| Model | Parameters | Best epoch | Train time | Movement Macro-F1 (3-way) | Directional BA | Macro-F1 (4-way) | Natural Arrival-F1 | Hard Arrival-F1 | Checkpoint score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 2,722,164 | 71 | 71.9 min | 74.75% | 74.84% | 70.84% | 59.11% | 73.37% | 69.46% |
| M1 | 2,724,839 (+2,675) | 62 | 66.1 min | 75.03% | 75.13% | 69.94% | 54.69% | 70.69% | 67.77% |

- M0 ran all 80 epochs; M1 early-stopped at epoch 72 (best 62).
- M0 is better offline on checkpoint score, movement Macro-F1, natural arrival F1 and hard arrival F1 in this Stage-2 dataset.
- Both models use identical seed/data order/optimizer/loss/budget; base initialization is identical, M1 has only the +2,675-parameter fusion head.

## 3. Closed-loop Dev Val 180 (Stop threshold 0.5)

| Band | Model | SR | OSR | SPL | nDTW | Premature-Stop | Timeout | Mean steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | M0 | 28.89% | 31.11% | 0.2877 | 0.3570 | 56.67% | 14.44% | 132.9 |
| Overall | M1 | 27.78% | 30.56% | 0.2778 | 0.3648 | 55.00% | 17.22% | 138.1 |
| Short | M0 | 41.67% | 46.67% | 0.4167 | 0.5235 | 48.33% | 10.00% | 118.2 |
| Short | M1 | 40.00% | 41.67% | 0.4000 | 0.5412 | 45.00% | 15.00% | 122.5 |
| Medium | M0 | 30.00% | 31.67% | 0.2979 | 0.3445 | 56.67% | 13.33% | 139.5 |
| Medium | M1 | 23.33% | 28.33% | 0.2333 | 0.3285 | 53.33% | 23.33% | 155.3 |
| Long | M0 | 15.00% | 15.00% | 0.1486 | 0.2030 | 65.00% | 20.00% | 141.1 |
| Long | M1 | **20.00%** | **21.67%** | **0.2000** | **0.2247** | 66.67% | **13.33%** | 136.5 |

- M1 improves Long SR +5.00 pp, OSR +6.67 pp, SPL +0.0514, nDTW +0.0217, and reduces Long Timeout 20.0%→13.3%.
- M1 is slightly worse overall: SR −1.11 pp, OSR −0.56 pp, SPL −0.0099; nDTW +0.0078.
- Short is nearly flat/slightly worse; Medium is the main regression: SR 30.0%→23.3%, OSR 31.7%→28.3%, Timeout 13.3%→23.3%.

### Paired per-episode statistics

Overall (M1 vs M0):

| Metric | M1 better | M1 worse | Ties | Delta | 95% CI | Sign-test p |
|---|---:|---:|---:|---:|---:|---:|
| SR | 17 | 19 | 144 | −1.11 pp | [−7.78, +5.56] | 0.868 |
| OSR | 18 | 19 | 143 | −0.56 pp | [−7.22, +6.11] | 1.000 |
| Premature-Stop | 23 | 20 | 137 | −1.67 pp | [−8.89, +5.56] | — |
| Timeout | 15 | 20 | 145 | +2.78 pp | [−3.89, +9.44] | — |
| SPL | 17 | 19 | 144 | −0.0099 | [−0.0761, +0.0556] | 0.868 |
| nDTW | 74 | 85 | 21 | +0.0078 | [−0.0424, +0.0593] | 0.428 |

Long band (60 episodes):

| Metric | M1 better | M1 worse | Ties | Delta | 95% CI | Sign-test p |
|---|---:|---:|---:|---:|---:|---:|
| SR | 7 | 4 | 49 | +5.00 pp | [−5.00, +16.67] | 0.549 |
| OSR | 8 | 4 | 48 | +6.67 pp | [−5.00, +18.33] | 0.388 |
| Premature-Stop | 9 | 10 | 41 | +1.67 pp | [−13.33, +15.00] | — |
| Timeout | 7 | 3 | 50 | −6.67 pp | [−16.67, +3.33] | — |
| SPL | 7 | 4 | 49 | +0.0514 | [−0.05, +0.1667] | 0.549 |
| nDTW | 28 | 27 | 5 | +0.0217 | — | 1.000 |

### Long failure analysis

| Model | Success | Premature Stop | Far premature (>20 m) | Timeout | OSR |
|---|---:|---:|---:|---:|---:|
| M0 | 9 | 39 | 39 | 12 | 9 |
| M1 | 12 | 40 | 39 | 8 | 13 |

- M0 Long failures are 39 far premature stops (minimum distance 20.4–483.4 m) plus 12 movement timeouts (42.1–445.3 m); no OSR-only episode.
- M1 converts 4 timeouts into 3 successes plus one near-miss (minimum distance 1.30 m but Stop occurred outside the success logic; OSR=1, success=0). It still has 39 far premature stops (minimum distance up to 483.4 m) and 8 timeouts (84.2–410.4 m).
- Therefore the Long gain is mostly fewer timeouts / slightly better reachability; **the dominant Long error remains far premature Stop**, not near-goal missed Stop.
- The Long improvement is not statistically resolved on 60 episodes (SR p=0.549, OSR p=0.388). The Medium regression is larger in absolute terms than the Long gain; overall M1 is neutral-to-worse.

### Inference latency

- M0: total 0.964 ms/step (model 0.959, map 0.005); p95 0.992 ms.
- M1: total 1.231 ms/step (model 1.028, map 0.202); p95 1.309 ms.
- Timing uses `torch.cuda.synchronize()` around timed segments and excludes the first episode as warm-up; M1 is slower, not faster.

## 3.1 Conclusion

- Stage-2 balanced data improves both models substantially versus the old 60-episode development set; M0 overall SR 28.9%, Long SR 15%.
- **M1 does not provide a robust improvement over M0**: offline score/Macro-F1/Arrival-F1 are lower; closed-loop overall SR/OSR/SPL are slightly lower, and the Long gain is offset by a Medium regression.
- The Long failure mode is still dominated by far premature Stop (39/60 for M1), not simply timeout; arrival calibration/off-policy negatives remain the main target.
- Per the Stage-2a scope: **do not start Small, Far-Negative ablations, or M1-Dir** in this round.
- Suggested next work: stop calibration/checkpoint selection and off-policy arrival negatives; revisit M1 fusion only after M0 stop behavior improves.

## 4. Artifacts

- Run: `runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z/`
  - `Transformer-M/M0/{best.pt,history.csv,metrics.json}`
  - `Transformer-M/M1/{best.pt,history.csv,metrics.json}`
  - `summary.json`
- Training log: `runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z.log`
- Data: `runs/v83_spatial_stage2/datasets/`
- Cache: `runs/v83_spatial_stage2/cache_20261018T043434Z/`
- Final Test: sealed under `runs/v83_spatial_stage2/datasets/final_test_300/` (not used).
