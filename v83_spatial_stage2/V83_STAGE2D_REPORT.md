# V8.3 Stage-2d Arrival Calibration and Stop-Decision Report

- Date: 2026-10-19
- Branch: `experiment/v83-spatial-foundation`
- Scope: Transformer-M M0 Arrival/Stop only. No Small, M1-Dir, M2, Depth or
  full-MEC training.
- Data: existing Train-600 B0 off-policy rollout cache (600 episodes,
  74,001 states), split 480/120 by episode; Dev Val 180 for development
  closed-loop evaluation.
- Final Test 300: sealed; not read, evaluated or tuned.
- Frozen: Stage-2a M0 checkpoint (sha256
  `1c496fa8bd880592089ad85408ad7244c4c3e95817c6c4fdd23a7f5ecd5bd9e4`),
  backbone, context encoder, movement classifier and original 104-D Arrival
  Head. No old checkpoint was modified.

## 1. Executive summary

1. **B0 replay is exact in outcomes.** The frozen M0 rerun reproduces
   Stage-2a/2c: SR 28.89%, OSR 31.11%, premature 56.67%, 154 stops, 102
   premature, 87 far>40 m, median stop distance 102.3 m. Outcome mismatches:
   0/180. Numeric trajectory diagnostics differ only at float level (35
   `min_distance_m` differences ≤5e-5 and one step-count difference on an
   episode with identical success/OSR/premature/timeout).
2. **R2 is the first method that materially reduces the high-confidence far
   tail.** On Dev Val natural samples, R2 pushes 73 of the 98 B0 high-p
   far>40 m states below 0.5 (74.5%) while retaining 127 of 146
   high-confidence near positives (87.0%); the offline far-40 FPR falls
   0.672%→0.171% (3.9×), far-100 0.743%→0.176%, and the high-p far count
   falls 98→25. R3 (random far negatives, identical architecture/budget)
   removes none (98/98) and preserves 100% of near positives. R1 removes
   19.4% at 94.5% near preservation. The 104-D feature probe explains why:
   held-out AUC 0.783 from the features vs 0.502 from the B0 logit alone.
3. **R2's closed loop improves SR/OSR and reduces far risk, at a moderate
   near cost.** SR 28.89→32.22 (+3.33 pp, 7W/1L, p=0.070), OSR 31.11→36.67
   (+5.56 pp, 10W/0L, p=0.002), premature 56.67→48.89 (-7.78 pp), SPL
   +0.033. Far-40 stops 87→76 (-12.6% cumulative; per-step 0.375%→0.273%,
   -27%), far>100 78→69, median stop distance 102.3→66.5 m. Reach-
   radius-without-success rises 4→8; one B0 success is lost. The gain is
   concentrated in Long (+8.3 pp, 5W/0L); Medium gets no far benefit and
   pays a near cost.
4. **R4 threshold 0.7 is the cleanest decision-layer rule.** SR 34.44%
   (+5.56 pp, 11W/1L, p=0.0063), OSR 38.33% (+7.22 pp, 13W/0L, p=0.00024),
   SPL +0.0555, premature 48.89%. Near cost is modest (1 B0 success lost;
   reach-no-success 4→7; absolute near stops 52→62). It does **not** solve
   the far tail: far-40 stops 87→82, far>100 78→75. Nine of its 11 wins are
   20–28 m boundary Stops and only two are far (67.6 m, 231.6 m). It is a
   boundary/stop-timing improvement, not a material far-risk reduction.
5. **R4 persistence 2-of-2 @0.5 is the strongest far-risk decision rule.**
   SR 36.11% (+7.22 pp, 16W/3L, p=0.0044), OSR 41.11% (+10.0 pp, 18W/0L,
   p=7.6e-06), far-40 stops 87→78 (-10.3%; per-step 0.285%, -24%), far>100
   78→70, boundary 15→3. But the confirmation delay has a clear near cost:
   3 B0 successes lost, reach-no-success 4→9, near-20 Stop rate
   48.6%→30.1% (absolute near stops 52→65). 2-of-2 @0.6 reduces far-40 to 75
   (-13.8%) and far>100 to 68, but costs 5 B0 successes and 11 near misses.
6. **Stronger confirmation is too aggressive.** 3-of-3 @0.5 reduces far-40
   to 73 (-16.1%) and raises OSR to 43.89% (+12.78 pp), but loses 9 B0
   successes, drives Timeout to 25.0%, and reach-no-success to 18/180; SR is
   only +5.0 pp (p=0.122). It is not keepable.
7. **The far tail is high-confidence, so thresholds cannot remove it.** The
   far>100 m Stops surviving threshold 0.7 have median p=0.83 (p25 0.77),
   versus B0 median 0.74. Global thresholds remove low-confidence
   boundary/near Stops (mostly p 0.5–0.7), not the confident far tail. The
   feature-based R2 residual is the only method in this stage that attacks
   that tail directly.
8. **Acceptance:** R2 passes the 3 pp SR/OSR gate (borderline SR p=0.070,
   OSR p=0.002) and materially reduces the offline far tail and closed-loop
   far risk; its near cost is moderate. Threshold 0.7 passes SR/OSR but not
   the far-risk gate; it is the safe decision-layer alternative. 2-of-2 @0.5
   passes SR/OSR and far risk but has a material near cost. R1, threshold
   0.6 and R3 are below the gate / controls. Results are single-seed
   development evidence. The natural next experiment, not run here, is R2 +
   threshold 0.7 (plus a confirmation seed).

## 2. Protocol and frozen assets

- The Stage-2c Train-600 rollout cache was reused as-is; no states were
  re-collected. Rollout sha256
  `47d1548a62a5532490e6c51b7e0d30cd1ab930a9f3eaba7e7c0c5af054060219`.
- Dev Val features sha256
  `1d7f1c786d5cc47b19588c101382c2e381bc68f189e60e73cee33a3d4f88659a`;
  val specs sha256
  `ef9c8b6805cfb75bffebaa553a41c568b0f359e83548ce6aab3bb5898b07e5a6`.
- Episode split: seed `20261019`, 480 calibration-train / 120
  calibration-validation, no episode overlap.
- Base calibration rows (R1/R2/R3): all Train-480 states with
  `true_distance <= 40 m` (3,065 rows; 374 positives).
- R2 hard far negatives: 355 Train-480 states with `true_distance > 40 m`
  and B0 `p >= 0.3`.
- R3 random control: 355 Train-480 far negatives with B0 `p < 0.3`,
  disjoint from the hard set. R2/R3 therefore have identical row counts
  (3,420), architecture (3,393 parameters), initialization and training
  budget.
- R1 Platt scaling (2 parameters) used the R2 row set. R2/R3 add a
  104→32→GELU→1 residual with a zero-initialized last layer. All variants
  exactly reproduce the B0 output at initialization (`max |Δ| = 0.0`).
- Loss: plain BCE, no `pos_weight`, plus an anchor
  `KL(B0 || calibrated)` on `distance <= 40 m` rows and a residual
  `mean(r²)` penalty (R2/R3). Selection used the binary far-40 false-Stop
  count on the 120 internal-validation episodes plus a near-20 recall
  shortfall penalty; F1/Brier/ECE were reported but not used for selection.
- R4 re-ran the full Dev Val 180 closed loop for every rule; the
  Stop-confirmation state was created fresh per episode. Hazard/risk tables
  use the actual per-step Stop decision, including confirmation.
- The B0 replay check gates on outcome fields only (success/OSR/premature/
  timeout); step-count and 1e-5-level distance differences are reported
  separately because CUDA kernels are not bit-reproducible.

## 3. R4 Stop-decision rules

All rules re-ran the full Dev Val 180 closed loop. The B0 (threshold 0.5)
re-run is an exact outcome replay of Stage-2a/2c: SR 28.89%, OSR 31.11%,
premature 56.67%, 154 stops, 102 premature, 87 far>40 m, median stop
distance 102.3 m; 0/180 outcome mismatches.

| Rule | SR | OSR | SPL | nDTW | Premature | Timeout | Steps | Far-40 stop/step | Near-20 stop rate | Far-40 stops/ep | Boundary stops/ep | Reach<20 no success | SR vs B0 (W/L, p) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| threshold 0.5 (B0) | 28.89% | 31.11% | 0.2877 | 0.3570 | 56.67% | 14.44% | 132.9 | 0.375% | 48.6% | 0.483 | 0.083 | 2.22% | — |
| threshold 0.6 | 31.11% | 33.33% | 0.3099 | 0.3607 | 53.89% | 15.00% | 137.8 | 0.354% | 48.3% | 0.472 | 0.067 | 2.22% | +2.22 pp (4W/0L, p=0.125) |
| threshold 0.7 | 34.44% | 38.33% | 0.3432 | 0.3573 | 48.89% | 16.67% | 147.9 | 0.319% | 41.6% | 0.456 | 0.033 | 3.89% | +5.56 pp (11W/1L, p=0.006) |
| 2-of-2 @0.5 | 36.11% | 41.11% | 0.3598 | 0.3576 | 45.00% | 18.89% | 157.2 | 0.285% | 30.1% | 0.433 | 0.017 | 5.00% | +7.22 pp (16W/3L, p=0.004) |
| 3-of-3 @0.5 | 33.89% | 43.89% | 0.3349 | 0.3237 | 41.11% | 25.00% | 191.0 | 0.220% | 17.2% | 0.406 | 0.006 | 10.00% | +5.00 pp (18W/9L, p=0.122) |
| 2-of-2 @0.6 | 35.00% | 41.11% | 0.3487 | 0.3483 | 43.33% | 21.67% | 172.6 | 0.249% | 26.5% | 0.417 | 0.017 | 6.11% | +6.11 pp (16W/5L, p=0.027) |

Cumulative false-Stop risk decomposition (every premature Stop is far-40 or
boundary 20–40 m; the sum matches the observed premature rate):

| Rule | Far-40 stops/ep | Boundary stops/ep | Sum | Observed premature |
|---|---:|---:|---:|---:|
| threshold 0.5 | 0.483 | 0.083 | 0.567 | 56.67% |
| threshold 0.6 | 0.472 | 0.067 | 0.539 | 53.89% |
| threshold 0.7 | 0.456 | 0.033 | 0.489 | 48.89% |
| 2-of-2 @0.5 | 0.433 | 0.017 | 0.450 | 45.00% |
| 3-of-3 @0.5 | 0.406 | 0.006 | 0.411 | 41.11% |
| 2-of-2 @0.6 | 0.417 | 0.017 | 0.433 | 43.33% |

Stop decisions by distance bin (actual rule decision; rate (stops/steps)):

| Rule | 0–20 m | 20–40 m | 40–60 m | 60–100 m | 100–200 m | 200 m+ |
|---|---|---|---|---|---|---|
| threshold 0.5 | 48.60% (52/107) | 2.46% (15/610) | 0.19% (2/1068) | 0.29% (7/2447) | 0.32% (24/7577) | 0.45% (54/12112) |
| threshold 0.6 | 48.28% (56/116) | 1.72% (12/698) | 0.19% (2/1072) | 0.28% (7/2488) | 0.31% (24/7790) | 0.41% (52/12646) |
| threshold 0.7 | 41.61% (62/149) | 0.81% (6/738) | 0.18% (2/1117) | 0.17% (5/2955) | 0.30% (24/7989) | 0.37% (51/13680) |
| 2-of-2 @0.5 | 30.09% (65/216) | 0.39% (3/760) | 0.21% (3/1455) | 0.17% (5/2992) | 0.27% (22/8278) | 0.33% (48/14602) |
| 3-of-3 @0.5 | 17.23% (61/354) | 0.12% (1/828) | 0.06% (1/1543) | 0.22% (7/3181) | 0.20% (18/9217) | 0.24% (47/19254) |
| 2-of-2 @0.6 | 26.47% (63/238) | 0.39% (3/768) | 0.14% (2/1465) | 0.16% (5/3033) | 0.25% (22/8632) | 0.27% (46/16931) |

Per band:

| Rule | Band | SR | OSR | SPL | Premature | Timeout | Far-40 stop/step | Near-20 stop rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| threshold 0.5 | short | 41.67% | 46.67% | 0.4167 | 48.33% | 10.00% | 0.356% | 44.6% |
| threshold 0.5 | medium | 30.00% | 31.67% | 0.2979 | 56.67% | 13.33% | 0.357% | 43.9% |
| threshold 0.5 | long | 15.00% | 15.00% | 0.1486 | 65.00% | 20.00% | 0.407% | 90.0% |
| threshold 0.7 | short | 48.33% | 56.67% | 0.4833 | 36.67% | 15.00% | 0.261% | 36.7% |
| threshold 0.7 | medium | 33.33% | 36.67% | 0.3312 | 51.67% | 15.00% | 0.323% | 37.0% |
| threshold 0.7 | long | 21.67% | 21.67% | 0.2152 | 58.33% | 20.00% | 0.367% | 81.2% |
| 2-of-2 @0.5 | short | 48.33% | 58.33% | 0.4833 | 38.33% | 13.33% | 0.297% | 26.6% |
| 2-of-2 @0.5 | medium | 33.33% | 38.33% | 0.3310 | 48.33% | 18.33% | 0.294% | 26.3% |
| 2-of-2 @0.5 | long | 26.67% | 26.67% | 0.2650 | 48.33% | 25.00% | 0.268% | 51.6% |
| 3-of-3 @0.5 | short | 46.67% | 60.00% | 0.4667 | 33.33% | 20.00% | 0.210% | 17.8% |
| 3-of-3 @0.5 | medium | 28.33% | 41.67% | 0.2833 | 45.00% | 26.67% | 0.222% | 13.2% |
| 3-of-3 @0.5 | long | 26.67% | 30.00% | 0.2546 | 45.00% | 28.33% | 0.226% | 23.5% |
| 2-of-2 @0.6 | short | 48.33% | 58.33% | 0.4833 | 35.00% | 16.67% | 0.239% | 25.9% |
| 2-of-2 @0.6 | medium | 31.67% | 38.33% | 0.3144 | 46.67% | 21.67% | 0.252% | 21.8% |
| 2-of-2 @0.6 | long | 25.00% | 26.67% | 0.2484 | 48.33% | 26.67% | 0.256% | 38.5% |

Paired per-episode comparison vs B0 (bootstrap 95% CI over episodes; exact
sign test):

| Rule | SR Δ (W/L) | OSR Δ (W/L) | Premature Δ (W/L) | Timeout Δ (W/L) | SPL Δ | SR sign p |
|---|---|---|---|---|---|---|
| threshold 0.6 | +2.22 pp (4W/0L) | +2.22 pp (4W/0L) | -2.78 pp (5W/0L) | +0.56 pp (0W/1L) | +0.0222 | 0.125 |
| threshold 0.7 | +5.56 pp (11W/1L) | +7.22 pp (13W/0L) | -7.78 pp (15W/1L) | +2.22 pp (0W/4L) | +0.0555 | 0.006 |
| 2-of-2 @0.5 | +7.22 pp (16W/3L) | +10.00 pp (18W/0L) | -11.67 pp (23W/2L) | +4.44 pp (0W/8L) | +0.0721 | 0.004 |
| 3-of-3 @0.5 | +5.00 pp (18W/9L) | +12.78 pp (23W/0L) | -15.56 pp (33W/5L) | +10.56 pp (0W/19L) | +0.0472 | 0.122 |
| 2-of-2 @0.6 | +6.11 pp (16W/5L) | +10.00 pp (18W/0L) | -13.33 pp (28W/4L) | +7.22 pp (0W/13L) | +0.0610 | 0.027 |

Per-episode outcome transitions vs B0:

| Rule | premature→success | success→premature | premature→timeout | success→timeout |
|---|---:|---:|---:|---:|
| threshold 0.6 | 4 | 0 | 1 | 0 |
| threshold 0.7 | 11 | 1 | 4 | 0 |
| 2-of-2 @0.5 | 16 | 2 | 7 | 1 |
| 3-of-3 @0.5 | 18 | 5 | 15 | 4 |
| 2-of-2 @0.6 | 16 | 4 | 12 | 1 |

Static threshold exposure on the B0 trajectory (diagnostic only; the rule
table above is the actual re-run):

| Threshold | Far-40 decisions/ep | Far-40 decision rate | Near-20 Stop rate |
|---|---:|---:|---:|
| 0.5 | 0.483 | 0.375% | 48.6% |
| 0.6 | 0.383 | 0.297% | 44.9% |
| 0.7 | 0.272 | 0.211% | 40.2% |
| 0.8 | 0.156 | 0.121% | 33.6% |

### R4 interpretation

- **Threshold 0.6** is a weaker version of the same boundary/stop-timing
  effect: +2.22 pp SR (4W/0L) with near Stop preserved, but far-40
  cumulative risk only -2.3%. Its four wins are three 20.7–23.0 m boundary
  Stops and one 231.6 m far Stop. It is below the 3 pp gate.
- **Threshold 0.7** is the cleanest closed-loop rule: +5.56 SR/+7.22 OSR,
  SPL better, near cost modest. Nine of its 11 wins are boundary Stops and
  only two are far. The far tail barely moves (87→82 far-40; 78→75
  far>100); the surviving far>100 Stops have median p=0.83. It is a
  **safe decision-layer candidate**, not a far-tail solution.
- **2-of-2 @0.5** is the strongest far-risk decision rule: -10.3% far-40
  cumulative risk, -24% per-step, far>100 78→70, with +7.22 SR/+10 OSR.
  Four of its 16 wins are far Stops, versus two for 0.7. But it loses 3 B0
  successes and raises reach-no-success 4→9; the confirmation delay can
  make the UAV fly through the 20 m radius without stopping.
- **2-of-2 @0.6** pushes far risk further (-13.8%, far>100 68) but costs
  5 B0 successes and 11 near misses; it is more aggressive than 0.5.
- **3-of-3 @0.5** is not keepable: far risk -16.1% and OSR +12.78 pp, but 9
  B0 successes lost, 18/180 reach-no-success, Timeout 25%, and SR only
  +5.0 pp (p=0.122).
- The confirmation delay's near cost is concentrated in Short/Medium; Long
  gains the most from far-Stop suppression (2-of-2 Long +11.67 pp, 7W/0L,
  no near misses).

## 4. R1/R2/R3 off-policy calibration

Data: 480/120 episode split; 3,065 base rows (374 positives) + 355 hard
(R2) or 355 random (R3) far negatives; R1 uses the R2 row set. R2/R3 have
identical architecture (3,393 parameters), initialization, optimizer budget
and grid; the only difference is the far-negative selection. All variants
reproduce B0 exactly at initialization (`max |Δ| = 0.0`).

Feature separability diagnostic (CPU linear probe, same 480/120 split):
hard far negatives vs near positives on the held-out 120 episodes reach
73.9% accuracy / AUC 0.783 from the 104-D features; the B0 logit alone is at
chance (AUC 0.502). The signal for R2 is present, though weaker than the
offline natural comparison in Stage-2c. (`feature_probe_*.json`.)

The effect is stable across the grid: the top-5 R2 configs all give
far-40 stops/ep 0.167–0.192 with near recall 0.511–0.523; no R2 config
collapses near recall. The top R3 configs all give far-40 stops/ep 0.317
and near recall 0.5568 (no effect). R1's best mild config gives 0.267; a
more aggressive R1 config reaches 0.242 but at near recall 0.511.

Selected configurations (internal 120-episode validation):

| Variant | LR | Anchor λ | Residual λ | Best epoch | Params | Train rows | Init max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 0.1 | 10 | 0 | 20 | 2 (scale 0.854, bias −0.262) | 3,420 | 0.0 |
| R2 | 1e-3 | 1 | 1e-3 | 5 | 3,393 | 3,420 | 0.0 |
| R3 | 1e-3 | 10 | 1e-3 | 1 | 3,393 | 3,420 | 0.0 |

R2/R3 grid top-3 (internal validation):

| Variant | LR | Anchor λ | Residual λ | Best epoch | Score | Near-20 recall | Far-40 stop/step | Far-40 stops/ep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R2 | 1e-3 | 1 | 1e-3 | 5 | 0.2119 | 51.1% | 0.135% | 0.167 |
| R2 | 1e-3 | 1 | 1e-2 | 6 | 0.2176 | 51.1% | 0.141% | 0.175 |
| R2 | 1e-2 | 10 | 1e-3 | 28 | 0.2178 | 52.3% | 0.148% | 0.183 |
| R3 | 1e-3 | 10 | 1e-3 | 1 | 0.3697 | 55.7% | 0.256% | 0.317 |
| R3 | 1e-3 | 10 | 1e-2 | 1 | 0.3697 | 55.7% | 0.256% | 0.317 |
| R3 | 1e-2 | 10 | 1e-2 | 2 | 0.3712 | 55.7% | 0.256% | 0.317 |

Internal calibration-validation (B0-rollout states, 120 episodes):

| Variant | Far-40 stop/step | Far-40 stops/ep | Far-40 p-sum/ep | Near-20 stop rate | Boundary stop rate | Natural F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.263% | 0.325 | 2.553 | 55.7% | 2.71% | 0.519 | 0.00562 |
| R1 | 0.215% | 0.267 | 2.963 | 54.5% | 2.08% | 0.539 | 0.00547 |
| R2 | **0.135%** | **0.167** | **2.263** | 51.1% | 2.08% | 0.552 | **0.00450** |
| R3 | 0.256% | 0.317 | 2.650 | 55.7% | 2.92% | 0.519 | 0.00575 |

Dev Val 180 offline Arrival metrics:

| Variant | Natural F1 | Hard F1 | Natural Brier | Natural ECE | Far-40 FPR | Far-100 FPR | Near-20 recall | High-p far n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.593 | 0.735 | 0.00970 | 0.0257 | 0.672% | 0.743% | 81.11% | 98 |
| R1 | 0.605 | 0.725 | 0.00881 | 0.0275 | 0.542% | 0.613% | 76.67% | 79 |
| R2 | **0.665** | 0.699 | **0.00693** | **0.0206** | **0.171%** | **0.176%** | 71.11% | **25** |
| R3 | 0.582 | **0.739** | 0.01026 | 0.0273 | 0.713% | 0.780% | 81.11% | 104 |

Dev Val natural FPR by distance bin:

| Variant | <20 recall | 20–40 | 40–60 | 60–100 | 100–200 | 200+ |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 81.11% | 4.89% | 0.37% | 0.53% | 0.43% | 1.04% |
| R1 | 76.67% | 4.24% | 0.30% | 0.36% | 0.35% | 0.86% |
| R2 | 71.11% | 3.74% | **0.07%** | **0.20%** | **0.10%** | **0.25%** |
| R3 | 81.11% | 5.17% | 0.45% | 0.57% | 0.46% | 1.07% |

High-confidence subset (natural Dev Val, B0 p ≥ 0.5):

| Variant | High-p far>40 still ≥0.5 | Mean p on those | High-p far pushed <0.5 | Near positives (B0 p≥0.5) preserved |
|---|---:|---:|---:|---:|
| B0 | 98 / 98 | 0.690 | 0.0% | 146 / 146 |
| R1 | 79 / 98 | 0.610 | 19.4% | 94.5% |
| R2 | **25 / 98** | **0.418** | **74.5%** | 87.0% |
| R3 | 98 / 98 | 0.712 | 0.0% | 100% |

Logit-shift diagnostic (Dev Val natural, mean Δlogit vs B0): R2 shifts
high-p far states by −1.20 (median −1.18) and high-confidence near positives
by −0.93 (median −0.99), i.e., it is selective but not perfectly so. R1
shifts high-p far by −0.39 but near positives by −0.54 (anti-selective),
which is why its near recall falls while its far reduction stays modest. R3
shifts both slightly positive (+0.11 / +0.03). The near cost of R2 is
structural: any feature-based correction that lowers far p also lowers some
near p.

Hard sample-type diagnostics (Dev Val): R2 hard-positive recall
68.9%→61.7%, boundary-negative FPR 6.7%→5.0%, distractor-negative FPR
11.9%→9.7%; R1/R3 are much closer to B0. R2 improves both hard-negative
types but loses hard-positive recall.

Closed loop (Dev Val 180, threshold 0.5):

| Variant | SR | OSR | SPL | nDTW | Premature | Timeout | Steps | Min dist m | Far-40 stop/step | Near-20 stop rate | Far-40 stops/ep | Far-40 p-sum/ep | Reach<20 no success | SR vs B0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B0 | 28.89% | 31.11% | 0.2877 | 0.3570 | 56.67% | 14.44% | 132.9 | 134.1 | 0.375% | 48.6% | 0.483 | 2.614 | 2.22% | — |
| R1 | 30.00% | 32.22% | 0.2988 | 0.3606 | 55.56% | 14.44% | 135.5 | 131.6 | 0.365% | 47.4% | 0.478 | 3.169 | 2.22% | +1.11 pp (2W/0L, p=0.5) |
| R2 | 32.22% | 36.67% | 0.3210 | 0.3639 | 48.89% | 18.89% | 159.6 | 116.6 | **0.273%** | 37.4% | **0.422** | 3.286 | 4.44% | **+3.33 pp (7W/1L, p=0.070)** |
| R3 | 27.78% | 30.00% | 0.2766 | 0.3563 | 57.78% | 14.44% | 132.7 | 134.5 | 0.375% | 48.1% | 0.483 | 2.739 | 2.22% | -1.11 pp (0W/2L, p=0.5) |

Cumulative false-Stop risk decomposition:

| Variant | Far-40 stops/ep | Boundary stops/ep | Sum | Observed premature |
|---|---:|---:|---:|---:|
| B0 | 0.483 | 0.083 | 0.567 | 56.67% |
| R1 | 0.478 | 0.078 | 0.556 | 55.56% |
| R2 | 0.422 | 0.067 | 0.489 | 48.89% |
| R3 | 0.483 | 0.094 | 0.578 | 57.78% |

Stop decisions by distance bin:

| Variant | 0–20 m | 20–40 m | 40–60 m | 60–100 m | 100–200 m | 200 m+ |
|---|---|---|---|---|---|---|
| B0 | 48.60% (52/107) | 2.46% (15/610) | 0.19% (2/1068) | 0.29% (7/2447) | 0.32% (24/7577) | 0.45% (54/12112) |
| R1 | 47.37% (54/114) | 2.01% (14/695) | 0.19% (2/1072) | 0.28% (7/2488) | 0.30% (23/7700) | 0.44% (54/12317) |
| R2 | 37.42% (58/155) | 1.59% (12/753) | 0.18% (2/1132) | 0.19% (5/2701) | 0.28% (25/8782) | 0.29% (44/15197) |
| R3 | 48.08% (50/104) | 2.79% (17/609) | 0.19% (2/1068) | 0.25% (6/2445) | 0.33% (25/7549) | 0.45% (54/12112) |

Per band:

| Variant | Band | SR | OSR | SPL | Premature | Timeout | Far-40 stop/step | Near-20 stop rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | short | 41.67% | 46.67% | 0.4167 | 48.33% | 10.00% | 0.356% | 44.6% |
| B0 | medium | 30.00% | 31.67% | 0.2979 | 56.67% | 13.33% | 0.357% | 43.9% |
| B0 | long | 15.00% | 15.00% | 0.1486 | 65.00% | 20.00% | 0.407% | 90.0% |
| R1 | short | 43.33% | 48.33% | 0.4333 | 46.67% | 10.00% | 0.348% | 44.8% |
| R1 | medium | 30.00% | 31.67% | 0.2979 | 56.67% | 13.33% | 0.356% | 41.9% |
| R1 | long | 16.67% | 16.67% | 0.1652 | 63.33% | 20.00% | 0.387% | 76.9% |
| R2 | short | 43.33% | 50.00% | 0.4333 | 40.00% | 16.67% | 0.241% | 37.7% |
| R2 | medium | 30.00% | 36.67% | 0.2979 | 55.00% | 15.00% | 0.316% | 26.1% |
| R2 | long | **23.33%** | **23.33%** | **0.2318** | 51.67% | 25.00% | 0.259% | 82.4% |
| R3 | short | 40.00% | 45.00% | 0.4000 | 50.00% | 10.00% | 0.358% | 44.4% |
| R3 | medium | 28.33% | 30.00% | 0.2812 | 58.33% | 13.33% | 0.357% | 42.5% |
| R3 | long | 15.00% | 15.00% | 0.1486 | 65.00% | 20.00% | 0.407% | 90.0% |

Paired per-episode comparison vs B0:

| Variant | SR Δ (W/L) | OSR Δ (W/L) | Premature Δ (W/L) | Timeout Δ (W/L) | SPL Δ | nDTW Δ | SR sign p |
|---|---|---|---|---|---|---|---|
| R1 | +1.11 pp (2W/0L) | +1.11 pp (2W/0L) | -1.11 pp (2W/0L) | +0.00 pp | +0.0111 | +0.0037 | 0.500 |
| R2 | +3.33 pp (7W/1L) | +5.56 pp (10W/0L) | -7.78 pp (15W/1L) | +4.44 pp (0W/8L) | +0.0333 | +0.0069 | 0.070 |
| R3 | -1.11 pp (0W/2L) | -1.11 pp (0W/2L) | +1.11 pp (0W/2L) | +0.00 pp | -0.0111 | -0.0006 | 0.500 |

Per-episode outcome transitions vs B0:

| Variant | premature→success | success→premature | premature→timeout |
|---|---:|---:|---:|
| R1 | 2 | 0 | 0 |
| R2 | 7 | 1 | 8 |
| R3 | 0 | 2 | 0 |

On-policy base→calibrated shift (each variant's own trajectory):

| Variant | Far-40 steps | Far base p | Far cal p | High-p base far n | High-p base p | High-p cal p | High-p below 0.5 | Near steps | Near stop rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 23,204 | 0.0203 | 0.0203 | 87 | 0.733 | 0.733 | 0.0% | 107 | 48.6% |
| R1 | 23,577 | 0.0211 | 0.0242 | 109 | 0.722 | 0.646 | 21.1% | 114 | 47.4% |
| R2 | 27,812 | 0.0258 | 0.0213 | 188 | 0.727 | 0.495 | 60.1% | 155 | 37.4% |
| R3 | 23,174 | 0.0202 | 0.0213 | 85 | 0.734 | 0.745 | 0.0% | 104 | 48.1% |

### R2/R3 interpretation

- **R2 vs R3 is a clean hard-mining ablation.** Same 3,393-parameter
  residual, same 3,420 rows, same initialization/budget; only the 355 far
  negatives differ. R2 reduces internal far-40 stops/ep 0.325→0.167 (-49%)
  and offline far-40 FPR 0.672%→0.171% (-75%); R3 does nothing (0.317,
  0.713%). On the high-confidence subset, R2 pushes 73/98 below 0.5, R3
  pushes 0/98. The 104-D linear probe (held-out AUC 0.783) shows the signal
  exists; the random far negatives do not expose the model to it.
- **R2's closed-loop gain is real but concentrated in Long.** SR +3.33 pp
  (7W/1L), OSR +5.56 pp (10W/0L), premature -7.78 pp, SPL +0.033. Long
  +8.33 pp (5W/0L), far-40/ep 0.567→0.433; Short +1.67 pp (1W/0L), far
  0.400→0.333; Medium 0 (1W/1L), far 0.483→0.500 (no far benefit) and
  reach-no-success 1→4. The R2 correction helps the far-dominated Long band
  but pays a near cost in Short/Medium.
- **R2's cumulative far reduction is smaller than its offline correction.**
  On R2's own trajectory, 113/188 (60.1%) high-p base far states are pushed
  below 0.5, but the trajectory is longer (27.8k vs 23.2k far steps) and
  visits more high-p far states (188 vs 87), so far-40 stops only fall
  87→76 (-12.6%). The median stop distance falls 102.3→66.5 m, which is the
  mechanism behind the Long gain. The policy still visits and stops at the
  confident far tail; the correction is partial, not a 4× reduction.
- **R2's near cost is structural and moderate.** Offline near-20 recall
  81.1%→71.1% and hard-positive recall 68.9%→61.7%; closed-loop reach-
  no-success 4→8 and one B0 success lost. Absolute near stops rise 52→58
  (near steps 107→155), so the lower near rate is longer near exposure, not
  a collapse of near stopping. The 3 pp SR gain outweighs the one success
  loss (7W/1L), but the near cost is real.
- **R1 is a calibration-only control, not a candidate.** It shifts high-p
  far by −0.39 and near positives by −0.54 (anti-selective), so its offline
  far reduction is modest and its near recall falls; closed-loop SR +1.11
  pp (2W/0L), far-40/ep -1%. R1 is below the gate.
- **R3 matches B0 within noise** (SR -1.11 pp, 0W/2L; far-40/ep unchanged,
  0.483; high-p far unchanged). It confirms that the R2 gain comes from
  hard-negative mining, not from the residual architecture or budget.

## 5. Acceptance gate

Development gate: (1) SR/OSR ≈ +3 pp over B0 with no SPL regression;
(2) material per-step/cumulative far-40 risk reduction, not only offline F1;
(3) no near-20 Stop collapse and no timeout increase that cancels the SR
gain. Single-seed results are development evidence, not a deployable claim.

| Candidate | SR Δ | OSR Δ | SPL Δ | Far-40 stops/ep Δ | Near-20 Stop cost | Verdict |
|---|---:|---:|---:|---:|---|---|
| R4 threshold 0.6 | +2.22 pp (4W/0L, p=0.125) | +2.22 pp | +0.022 | −2.3% | none (48.3% vs 48.6%; reach 4→4) | Below SR/far gate; not a candidate |
| R4 threshold 0.7 | +5.56 pp (11W/1L, p=0.006) | +7.22 pp (13W/0L) | +0.056 | −5.7% | modest (1 success lost; near 41.6%; reach 4→7) | **Safe decision-layer candidate**; not a far-tail solution |
| R4 2-of-2 @0.5 | +7.22 pp (16W/3L, p=0.004) | +10.0 pp (18W/0L) | +0.072 | −10.3% | material (3 successes lost; near 30.1%; reach 4→9) | **Aggressive far-risk candidate**; near cost material |
| R4 2-of-2 @0.6 | +6.11 pp (16W/5L, p=0.027) | +10.0 pp (18W/0L) | +0.061 | −13.8% | material (5 successes lost; reach 4→11) | Strong far-risk candidate, higher near cost |
| R4 3-of-3 @0.5 | +5.00 pp (18W/9L, p=0.122) | +12.78 pp (23W/0L) | +0.047 | −16.1% | severe (9 successes lost; reach 4→18; Timeout +10.6 pp) | Not keepable |
| R1 Platt | +1.11 pp (2W/0L, p=0.5) | +1.11 pp | +0.011 | −1.1% | none (reach 4→4) | Below gate; calibration-only control |
| R2 hard residual | +3.33 pp (7W/1L, p=0.070) | +5.56 pp (10W/0L, p=0.002) | +0.033 | −12.6% (offline −75%) | moderate (1 success lost; reach 4→8; near 37.4%) | **Learned far-tail candidate**; passes SR/OSR and far-risk, near cost moderate |
| R3 random residual | −1.11 pp (0W/2L, p=0.5) | −1.11 pp | −0.011 | ~0 (0.483) | none (reach 4→4) | No-op control; confirms mining |

**Which methods actually reduce cumulative false-Stop risk?**
- **R2** is the only method that materially reduces the **high-confidence
  far tail** offline (74.5% of high-p far pushed below 0.5; far-40 FPR
  3.9× lower) and also reduces closed-loop far-40 cumulative stops by
  12.6% (per-step -27%, far>100 78→69). R3 does nothing; R1 is weak.
- **2-of-2 @0.5 / @0.6** reduce far-40 cumulative risk by 10.3% / 13.8%
  via decision-layer confirmation, with larger SR/OSR gains than R2 but a
  larger near Stop cost. **3-of-3** reduces far risk most (-16.1%) but is
  not keepable because of the near damage.
- **Threshold 0.7** improves SR/OSR most cleanly but reduces far-40 risk
  only 5.7%; it removes low-confidence boundary Stops, not the confident
  far tail.

**Which methods improve closed-loop SR/OSR without obvious near damage?**
- **R2** (+3.33/+5.56) and **threshold 0.7** (+5.56/+7.22) are the two
  candidates whose near cost is moderate (reach 4→8 and 4→7, one B0
  success lost each). R2 is the only one that also attacks the far tail;
  threshold 0.7 has the stronger SR/OSR and SPL gain.
- **2-of-2 @0.5/@0.6** improve SR/OSR more but lose 3–5 B0 successes and
  raise reach-no-success to 9–11; they are candidates only if that near
  cost is acceptable.
- **3-of-3, threshold 0.6, R1 and R3 are not keepable** under the stated
  gate.

**Single-seed caveat.** All numbers are one seed, Dev Val 180. The R2
closed-loop SR gain is borderline (p=0.070); threshold 0.7 and 2-of-2 @0.5
are stronger (p=0.006/0.004). A confirmation seed is required before any
deployment claim. The natural next experiment, not run here, is **R2 +
threshold 0.7** (feature-based far-tail correction plus decision-layer
boundary/stop-timing), with a near-recall safeguard (stronger anchor or
band-aware confirmation) and a DAgger round if the calibrated trajectory
changes materially.

## 6. Discipline / what was not run

- No Final Test read; Final Test 300 remains sealed.
- No Stage-2a/2b/2c checkpoint modified. The frozen M0 sha256 is unchanged.
- Movement and original Arrival-Head hashes verified unchanged in every
  calibration run.
- No Small, M1-Dir, M2, Depth or full-MEC training started.
- No `pos_weight`; plain BCE only.
- Calibration modules accept only `(features, base_logit)`; true distance
  and visibility are labels/diagnostics only.
- R2/R3 share architecture, initialization, row count and budget; the only
  difference is the far-negative selection.
- No second seed, no DAgger round, no R2+threshold combination was run.
- Unit tests: 25 pass (`python -m unittest v83_spatial_stage2.test_stage2d`).

## 7. Artifacts and reproduction

- R4 run: `runs/v83_spatial_stage2/stage2d/r4_20261019T002626Z/`
  - `metrics.json` / `metrics_recomputed.json`, `per_episode.csv`,
    `<rule>_per_episode.csv`, `steps/`, `stops/`, `risk.csv`,
    `risk_recomputed.csv`, `replay_check.json`, `provenance.json`.
- Calibration run: `runs/v83_spatial_stage2/stage2d/calibration_20261019T005253Z/`
  - `split.json`, `mining_stats.json`, `calibration_val_b0.json`,
    `calibration_val_metrics.json`, `dev_val_offline_metrics.json`,
    `dev_val_offline_probabilities.npz`, `closedloop_metrics.json`,
    `closedloop_metrics_recomputed.json`, `closedloop_per_episode.csv`,
    `risk.csv`, `risk_recomputed.csv`, `paired.json`, `replay_check.json`,
    `provenance.json`;
  - `R1/`, `R2/`, `R3/`: `calibration.pt`, `best.pt`, `metrics.json`,
    `history.csv`, `grid_summary.csv`, `grid_history.csv`.
- Feature-separability probe:
  `runs/v83_spatial_stage2/stage2d/feature_probe_20261019T004443Z.json`.
- Generated comparison tables:
  `runs/v83_spatial_stage2/stage2d/stage2d_summary.md`.
- Code:
  - `v83_spatial_stage2/stage2d_common.py`
  - `v83_spatial_stage2/stage2d_r4_decision_rules.py`
  - `v83_spatial_stage2/stage2d_assemble_r4.py`
  - `v83_spatial_stage2/stage2d_calibration.py`
  - `v83_spatial_stage2/stage2d_recompute_hazards.py`
  - `v83_spatial_stage2/stage2d_summarize.py`
  - `v83_spatial_stage2/stage2d_feature_probe.py`
  - `v83_spatial_stage2/test_stage2d.py`
  - `v83_spatial_stage2/STAGE2D_PROTOCOL.md`

```bash
# R4 full closed-loop re-run
python v83_spatial_stage2/stage2d_r4_decision_rules.py \
  --output-dir runs/v83_spatial_stage2/stage2d/r4_<stamp> --max-episodes 180

# Rebuild R4 metrics from saved CSVs if the runner exits before aggregation
python v83_spatial_stage2/stage2d_assemble_r4.py \
  --run-dir runs/v83_spatial_stage2/stage2d/r4_<stamp>

# Persistence-rule hazard from the actual Stop flags
python v83_spatial_stage2/stage2d_recompute_hazards.py \
  --run-dir runs/v83_spatial_stage2/stage2d/r4_<stamp> --kind r4

# R1/R2/R3 calibration (480/120 split, Dev Val closed loop)
python v83_spatial_stage2/stage2d_calibration.py \
  --output-dir runs/v83_spatial_stage2/stage2d/calibration_<stamp> \
  --max-episodes 180

# Feature separability probe
python v83_spatial_stage2/stage2d_feature_probe.py \
  --output runs/v83_spatial_stage2/stage2d/feature_probe_<stamp>.json

# Comparison tables
python v83_spatial_stage2/stage2d_summarize.py \
  --r4-dir runs/v83_spatial_stage2/stage2d/r4_<stamp> \
  --calibration-dir runs/v83_spatial_stage2/stage2d/calibration_<stamp> \
  --output runs/v83_spatial_stage2/stage2d/stage2d_summary.md

# Tests
python -m unittest v83_spatial_stage2.test_stage2d -v
```
