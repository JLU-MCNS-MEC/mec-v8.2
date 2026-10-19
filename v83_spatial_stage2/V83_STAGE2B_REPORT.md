# V8.3 Stage-2b Never-Stop and Arrival-only A0/A1 Report

> **Correction (2026-10-19, Stage-2c):** the A0/A1 closed-loop collapse was
> caused by `BCEWithLogitsLoss(pos_weight = negatives/positives)` = 7.02/8.64,
> not by head-only training or the cached-feature path. With plain BCE the same
> 3%/10% masks recover B0-level SR (B1 29.44%, B2 28.89% vs B0 28.89%); the 3%→10%
> far-negative change itself is inert. See `V83_STAGE2C_REPORT.md`.

- Date: 2026-10-18
- Branch: `experiment/v83-spatial-foundation`
- Scope: Stage-2b only; Stage-2a checkpoints preserved; no Small/M1-Dir/Depth/MEC BEV.
- Final Test 300 sealed; never read/evaluated/tuned.
- Stop threshold for closed-loop: 0.5.

## 1. Never-Stop counterfactual (M0/M1; movement weights unchanged)

Dev Val 180 episodes. Never-Stop forces the movement argmax, so success is impossible
and Timeout is 100% by construction; OSR/min-distance measure reachability.

| Model | Band | Episodes | OSR | Min distance mean | Min distance median | Timeout |
|---|---|---:|---:|---:|---:|---:|
| M0 | Short | 60 | 60.0% | 39.1 m | 3.1 m | 100% |
| M0 | Medium | 60 | 46.7% | 89.0 m | 56.4 m | 100% |
| M0 | Long | 60 | 45.0% | 118.4 m | 108.3 m | 100% |
| M1 | Short | 60 | 56.7% | 37.9 m | 5.4 m | 100% |
| M1 | Medium | 60 | 50.0% | 75.4 m | 23.6 m | 100% |
| M1 | Long | 60 | 41.7% | 120.2 m | 89.7 m | 100% |

Comparison with normal-policy OSR from Stage-2a:

- M0: Short 46.7%→60.0%, Medium 31.7%→46.7%, Long **15.0%→45.0%**.
- M1: Short 41.7%→56.7%, Medium 28.3%→50.0%, Long **21.7%→41.7%**.
- The movement policy can reach the goal radius much more often than the normal
  Stop policy; the largest gap is Long (M0 +30 pp, M1 +20 pp).

## 2. Far premature-Stop diagnosis

Stop events from the normal policy (offline Goal visibility only; never a model input):

| Model | Band | Stops | Premature >20 m | Far >40 m | Far >100 m | Distance median | Arrival p mean | Goal visible at stop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | Short | 54 | 29 | 24 | 19 | 21.0 m | 0.775 | 66.7% |
| M0 | Medium | 52 | 34 | 29 | 27 | 115.5 m | 0.773 | 67.3% |
| M0 | Long | 48 | 39 | 34 | 32 | 236.8 m | 0.724 | 77.1% |
| M1 | Short | 51 | 27 | 21 | 16 | 20.5 m | 0.760 | 76.5% |
| M1 | Medium | 46 | 32 | 27 | 25 | 119.9 m | 0.736 | 71.7% |
| M1 | Long | 52 | 40 | 37 | 35 | 213.3 m | 0.722 | 67.3% |

Key observations:

- Long false stops are far (median 213–237 m) and are not caused by an invisible goal:
  the Goal is visible at 67–77% of them, with mean arrival probability ~0.72–0.78.
- M1 does not reduce the far stops: Long far>40 m 37 vs M0 34, far>100 m 35 vs 32.
- The normal-vs-never OSR gap and the high arrival probability at far visible targets
  point to Arrival calibration / Stop decision as the main Stage-2b lever.

## 3. Arrival-only A0/A1

- A0: Stage-2 dataset mask, 3% far-negative retention.
- A1: A0 plus additional far negatives up to 10%; positives, near negatives and hard
  samples are unchanged; A0 ⊂ A1.
- Frozen M0 arrival features are cached; only the Arrival Head `net` is trained.
  Movement/backbone weights are not optimized. The full checkpoint movement hash is
  verified unchanged.
- Same seed `20264017`, same data order, max 80 epochs, patience 10, batch 256, lr 6e-5.
- BCEWithLogitsLoss uses `pos_weight = negatives / positives` per variant so total
  positive/negative loss weight remains comparable.

| Variant | Included | Positives | Negatives | Pos weight | Far included / total | Best epoch | Head train time |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 14,441 | 1,800 | 12,641 | 7.02 | 1,330 / 42,454 (3.13%) | 23 | ~1.0 s |
| A1 | 17,356 | 1,800 | 15,556 | 8.64 | 4,245 / 42,454 (10.0%) | 28 | ~1.4 s |

Head-only validation metrics at best epoch (Dev Val 180):

| Variant | Natural Arrival-F1 | Hard Arrival-F1 | Natural Brier | Hard Brier | Natural ECE | Hard ECE | Far-40 false stop | Far-100 false stop | Near-20 miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 19.27% | 78.88% | 0.0737 | 0.1217 | 0.1778 | 0.0934 | 8.72% | 10.39% | 4.44% |
| A1 | **28.01%** | 78.60% | **0.0527** | **0.1196** | **0.1405** | **0.0834** | **4.86%** | **5.64%** | 5.00% |

A1 improves Natural Arrival-F1 and calibration and roughly halves far false-stop rates;
Hard Arrival-F1 is essentially unchanged. Movement weights untouched.

## 4. A0/A1 closed-loop Dev Val 180 (Stop=0.5)

| Band | Variant | SR | OSR | SPL | nDTW | Premature-Stop | Timeout | Mean steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | A0 | 13.89% | 13.89% | 0.1389 | 0.2569 | 81.67% | 4.44% | 63.0 |
| Overall | A1 | **16.67%** | **16.67%** | **0.1660** | **0.2902** | **76.11%** | 7.22% | 82.4 |
| Short | A0 | 30.00% | 30.00% | 0.3000 | 0.4213 | 65.00% | 5.00% | 54.9 |
| Short | A1 | **31.67%** | **31.67%** | **0.3167** | **0.4683** | **63.33%** | 5.00% | 68.3 |
| Medium | A0 | 8.33% | 8.33% | 0.0833 | 0.2349 | 91.67% | 0.00% | 62.2 |
| Medium | A1 | **13.33%** | **13.33%** | **0.1315** | **0.2605** | **80.00%** | 6.67% | 94.8 |
| Long | A0 | 3.33% | 3.33% | 0.0333 | 0.1146 | 88.33% | 8.33% | 71.8 |
| Long | A1 | **5.00%** | **5.00%** | **0.0500** | **0.1419** | **85.00%** | 10.00% | 84.2 |

Paired A1 vs A0:

| Metric | A1 Better | A1 Worse | Ties | Delta | 95% CI | Sign-test p |
|---|---:|---:|---:|---:|---:|---:|
| SR | 7 | 2 | 171 | +2.78 pp | [−0.56, +6.11] | 0.180 |
| OSR | 7 | 2 | 171 | +2.78 pp | [−0.56, +6.11] | 0.180 |
| Premature-Stop | 12 | 2 | 166 | −5.56 pp | [−9.44, −1.67] | — |
| Timeout | 0 | 5 | 175 | +2.78 pp | [+0.56, +5.56] | — |
| SPL | 7 | 2 | 171 | +0.0272 | [−0.0056, +0.0605] | 0.180 |
| nDTW | 52 | 9 | 119 | +0.0333 | [+0.0142, +0.0561] | 1.8e−8 |

- A1 improves over A0 on SR/OSR/SPL/nDTW and reduces premature stops, but increases Timeout.
- Short/Medium SR/OSR do not regress (Short +1.7 pp, Medium +5.0 pp); Medium Timeout rises 0→6.67%.
- Long improves only +1.67 pp SR/OSR; far premature Stop remains dominant.

### Long stop events (offline stop-distance/visibility diagnostics)

| Variant | Long stops | Premature >20 m | Far >40 m | Far >100 m | Distance median | Arrival p mean | Goal visible at stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 55 | 53 | 48 | 46 | 398.4 m | 0.670 | 90.9% |
| A1 | 54 | 51 | 45 | 43 | 341.7 m | 0.684 | 85.2% |

A1 reduces Long far stops (>40 m 48→45, >100 m 46→43) and shortens the median stop
distance by ~57 m, but **51/60 Long episodes still fail by premature Stop**, mostly
far from the Goal. The far-stop Arrival probability remains ~0.68 and the Goal is
visible at 85% of A1 Long stops: this is still arrival calibration, not invisibility.

### Key interpretation

- A1 (10% far negatives) is a **controlled improvement over A0**, but both A0/A1 are
  far below the original Stage-2a M0 (SR 28.9%, Long SR 15.0%). Head-only retraining
  with cached frozen features did not preserve the original Arrival calibration.
- The far-negative increase helps calibration/far false-stop and modestly improves
  closed-loop SR/OSR, but it is not sufficient: 51/60 Long episodes still premature-stop,
  and the Timeout rate increases.
- Therefore, do not adopt A1 as the deployment Arrival head and do not start
  M1-Dir/Small. The next work should preserve the original Arrival training context
  (not just a cached 104-D head) and add off-policy far negatives / stop calibration.

## 5. Artifacts

- Stage-2a run/checkpoints preserved: `runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z/`
- Never-Stop diagnostics: `runs/v83_spatial_stage2/stage2b/never_20261018T143815Z/`
- Arrival ablation: `runs/v83_spatial_stage2/stage2b/arrival_20261018T150031Z/`
  - A0/A1 checkpoints, head-only metrics, cached features, closed-loop results
- Stage-2 data/cache not modified; Final Test sealed.
