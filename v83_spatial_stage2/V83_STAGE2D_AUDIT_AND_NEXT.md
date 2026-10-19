# V8.3 Stage-2d Audit, Stop-Error Diagnosis and Next-Step Decision

- Date: 2026-10-19
- Branch: `experiment/v83-spatial-foundation`
- Scope: audit of Stage-2a/2b/2c/2d results and a single next-step recommendation.
  No model training was started in this round.
- Task type: **image-goal UAV navigation** on CityNav/CityRefer (the goal is an
  image reference, not a language instruction). It is not a
  language-instruction VLN benchmark.
- Data: Train-600 is balanced 200/200/200 (Short/Medium/Long); Dev Val 180 is
  balanced 60/60/60; Final Test 300 (100/100/100) is **sealed and not read**.
- Evaluation protocol: Overall metrics are direct means over all 180 Dev Val
  episodes. No 30/50/20 training-mix weighting or re-weighting is used. The
  legacy `TRAIN.trajectory_mix` 30/50/20 is a training-data construction
  setting; it is not an evaluation weight.

## 1. Audit result

All Stage-2d runs are complete and internally consistent.

- **R0** = original Stage-2a M0 head (no training).
- **R1** = Platt scaling (2 parameters).
- **R2** = 104-D hard-negative residual (3,393 parameters).
- **R3** = same residual with random far negatives (3,393 parameters).
- **R4** = thresholds 0.5/0.6/0.7 and 2-of-2/3-of-3 consecutive Stop
  confirmation (2-of-2 @0.6 supplementary).
- Every R4 rule re-ran the full Dev Val 180 closed loop; the calibration
  R1/R2/R3 variants each re-ran the full Dev Val 180 closed loop.
- The R4 runner exited at the final replay gate because of numeric
  trajectory differences; all `steps/<rule>.csv`, `stops/<rule>.csv` and
  `<rule>_per_episode.csv` files were complete. `stage2d_assemble_r4.py`
  rebuilt the aggregate metrics from those files. The rebuilt metrics match
  an independent per-episode recomputation.
- B0 replay outcome mismatches: **0/180** for both the R4 and calibration
  runs. Numeric differences are only CUDA non-determinism (one step count on
  an episode with identical outcome; 1e-5-level distances). This is not a
  policy or metric discrepancy.
- Every method has 180 episodes, 60 per band; overall = direct mean over
  180; band success counts sum to the overall success count;
  `OSR − SR = reach-radius-without-success`; nDTW is finite.
- The 30/50/20 weighted band SR differs from the direct overall SR for every
  method; the check confirms the 30/50/20 weights were not used.
- Calibration movement and original Arrival-Head hashes are identical to
  M0 for R1/R2/R3. No old checkpoint was modified.
- The full audit is in `runs/v83_spatial_stage2/stage2d/stage2d_audit.json`;
  every check passes. Tests: 25 pass.

## 2. Table 1 — Stage-2d closed-loop metrics (Dev Val 180, N=180)

| ID | Method | SR | Success | OSR | SPL | nDTW | Premature | Timeout | Mean steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | M0 (threshold 0.5) | 28.89% | 52/180 | 31.11% | 0.2877 | 0.3570 | 56.67% | 14.44% | 132.9 |
| R1 | Platt scaling | 30.00% | 54/180 | 32.22% | 0.2988 | 0.3606 | 55.56% | 14.44% | 135.5 |
| R2 | Hard-negative residual | 32.22% | 58/180 | 36.67% | 0.3210 | **_0.3639_** | 48.89% | 18.89% | 159.6 |
| R3 | Random-negative residual | 27.78% | 50/180 | 30.00% | 0.2766 | 0.3563 | 57.78% | 14.44% | 132.7 |
| R4 | threshold 0.6 | 31.11% | 56/180 | 33.33% | 0.3099 | 0.3607 | 53.89% | 15.00% | 137.8 |
| R4 | threshold 0.7 | 34.44% | 62/180 | 38.33% | 0.3432 | 0.3573 | 48.89% | 16.67% | 147.9 |
| R4 | 2-of-2 @0.5 | **_36.11%_** | **_65/180_** | **_41.11%_** | **_0.3598_** | 0.3576 | 45.00% | 18.89% | 157.2 |
| R4 | 2-of-2 @0.6 | 35.00% | 63/180 | 41.11% | 0.3487 | 0.3483 | 43.33% | 21.67% | 172.6 |
| R4 | 3-of-3 @0.5 | 33.89% | 61/180 | 43.89% | 0.3349 | 0.3237 | 41.11% | 25.00% | 191.0 |

Reading the table:
- The largest **SR/OSR/SPL** gains are 2-of-2 @0.5: +7.22 pp SR
  (65 vs 52 successes), +10.00 pp OSR, +0.0721 SPL, premature −11.67 pp.
- **R2** has the best **nDTW** (0.3639) and the best far-tail offline
  calibration, but its SR gain is smaller (+3.33 pp, 58 vs 52) and it runs
  longer (159.6 steps) with more timeouts (18.89%).
- **Threshold 0.7** is the best **low-risk** decision rule: +5.56 pp SR
  (62 vs 52), +7.22 pp OSR, +0.0555 SPL, with a modest near cost (7 vs 4
  reach-no-success, 1 success lost).
- **R3** is a no-op control (50 vs 52 successes, SR −1.11 pp) and confirms
  that the residual architecture alone does not create the R2 gain.
- **3-of-3** has the highest OSR (43.89%) and lowest premature-Stop rate
  (41.11%) but the worst nDTW (0.3237) and Timeout (25.00%); it is not a
  keepable policy.
- **nDTW is not broadly improved.** R1 (0.3606), threshold 0.6 (0.3607) and
  R2 (0.3639) are above B0 by ≥0.003; 2-of-2 @0.5 is +0.0006 overall (flat),
  threshold 0.7 is +0.0003, and 3-of-3 (−0.0333) / 2-of-2 @0.6 (−0.0087) are
  below B0. The success gains are real; the trajectory-quality gain is not
  broad.

## 3. Table 2 — Best method (2-of-2 @0.5) vs original M0, by band

The best closed-loop success/SPL method is 2-of-2 @0.5. Pairing is by
episode; CIs are 95% bootstrap over episodes; W/L counts are paired
improvements/regressions.

| Band | M0 success | M0 SR | 2-of-2 success | 2-of-2 SR | SR Δ [95% CI] (W/L) | OSR Δ [95% CI] (W/L) | Premature Δ [95% CI] (W/L) | SPL Δ [95% CI] | nDTW Δ [95% CI] |
|---|---:|---:|---:|---:|---|---|---|---|---|
| Overall | 52/180 | 28.89% | 65/180 | 36.11% | **+7.22 pp [2.78, 12.22] (16W/3L)** | **+10.00 pp [6.11, 14.44] (18W/0L)** | **−11.67 pp [−17.22, −6.67] (23W/2L)** | **+0.0721 [0.0276, 0.1220]** | +0.0006 [−0.0209, 0.0222] |
| Short | 25/60 | 41.67% | 29/60 | 48.33% | +6.67 pp [0.00, 15.00] (5W/1L) | +11.67 pp [5.00, 20.00] (7W/0L) | −10.00 pp [−20.00, −1.67] (7W/1L) | +0.0667 [0.0000, 0.1500] | −0.0167 [−0.0574, 0.0122] |
| Medium | 18/60 | 30.00% | 20/60 | 33.33% | +3.33 pp [−5.00, 11.67] (4W/2L) | +6.67 pp [1.67, 13.33] (4W/0L) | −8.33 pp [−16.67, 0.00] (6W/1L) | +0.0332 [−0.0501, 0.1165] | −0.0137 [−0.0515, 0.0169] |
| Long | 9/60 | 15.00% | 16/60 | 26.67% | **+11.67 pp [5.00, 20.00] (7W/0L)** | **+11.67 pp [5.00, 20.00] (7W/0L)** | **−16.67 pp [−26.67, −8.33] (10W/0L)** | **+0.1165 [0.0496, 0.1999]** | **+0.0321 [0.0003, 0.0759]** |

- The success/OSR/SPL gains are significant overall and in Short/Long;
  Medium SR is not significant (CI includes 0) and has 2 losses.
- **nDTW is only significantly improved in Long** (+0.0321); Short and
  Medium are slightly negative (CIs include 0); overall is flat. This is the
  main trajectory-quality caveat.
- Near cost: 3 B0 successes lost, reach-no-success 4→9, Timeout +4.44 pp;
  the near-20 Stop rate falls 48.6→30.1% (absolute near stops 52→65 because
  near exposure rises). The confirmation delay can make the UAV pass
  through the 20 m radius without stopping.

**Safe alternative: threshold 0.7** (overall): 62/180 vs 52/180 successes,
SR +5.56 pp [2.22, 9.44] (11W/1L), OSR +7.22 pp [3.89, 11.11] (13W/0L),
premature −7.78 pp [−12.22, −3.89] (15W/1L), SPL +0.0555, nDTW +0.0003,
reach-no-success 7 vs 4, 1 B0 success lost, Timeout +2.22 pp, and the near
Stop ability is mostly preserved. It is the safer candidate if the near
Stop cost matters.

## 4. Table 3 — Hypotheses verified from Stage-2a to Stage-2d

| # | Hypothesis / question | Result | Conclusion | Worth continuing? |
|---|---|---|---|---|
| H1 | Stage-2b A0/A1 collapse was caused by `pos_weight` | Stage-2c plain-BCE B1/B2 recover B0-level SR (29.44/28.89 vs 28.89); `pos_weight` A0/A1 collapse (13.89/16.67). Stage-2d uses plain BCE; tests forbid `pos_weight=` | **Solved/excluded** | No; keep plain BCE |
| H2 | Cached 104-D features/logits are inaccurate | Stage-2c full-M0 vs cached-head max abs diff 3.8e-6; Stage-2d initialization logit diff 0.0 | **Excluded** | No |
| H3 | More random far negatives (3%→10%) improve Stop | Stage-2c B2 added 2,915 far negatives, only 22 with B0 p≥0.5 (mean p 0.031); far-40 FPR 0.67→0.50% offline, on-policy 0.365→0.347%, SR unchanged 28.89% | **Excluded**; added negatives are too easy | No |
| H4 | Far false Stops are low-confidence noise; a higher threshold is a free fix | B0 far>100 m Stops median p=0.74; at threshold 0.7 surviving far>100 Stops median p=0.83. Threshold 0.7: +5.56 SR/+7.22 OSR but far-40 risk only −5.7%; threshold 0.6: −2.3% | **Excluded as a free fix**; threshold is a boundary/stop-timing rule, not a far-tail fix | Partially; threshold 0.7 is a safe rule |
| H5 | Goal invisibility causes far false Stops | Goal visible at 65.7–77.1% of stops; mean Arrival p 0.72–0.78 | **Excluded** as the main cause | No |
| H6 | Continuous decision risk accumulation dominates the episode-level failure | Stage-2c: 132.3 far decisions × 0.365% + 3.4 boundary × 2.46% = 0.483 + 0.084 = 0.567, matching 56.7% premature. Long far decisions/ep 140.9 vs Short 117.3/Medium 138.8; Long far/ep 0.567 vs 0.400/0.483 | **Confirmed**; a small per-decision far FPR repeated over ~130 decisions is the mechanism; Long has more exposure | Yes; target per-decision far risk |
| H7 | Off-policy hard negatives reduce the policy's own high-confidence false Stops | R2/R3 same architecture/budget/initialization. R2 pushes 73/98 (74.5%) Dev Val high-p far states below 0.5; far-40 FPR 0.672→0.171% (3.9×); R3 0/98. On R2's own trajectory, 113/188 (60.1%) high-p far states are pushed below 0.5; far-40 stops 87→76, far>100 78→69; R3 is a no-op | **Confirmed offline and partially closed-loop**; hard mining is causal, but longer exposure and near cost limit the cumulative gain to −12.6% | Yes; R2 + threshold 0.7 / DAgger |
| H8 | Threshold/confirmation improves SR/OSR without near damage | threshold 0.7: +5.56/+7.22 pp, 1 success lost, reach 4→7, Timeout +2.22 pp; 2-of-2 @0.5: +7.22/+10.0 pp, 3 successes lost, reach 4→9, Timeout +4.44 pp; 3-of-3: 9 successes lost, reach 18, Timeout 25%, nDTW 0.3237 | **Partially confirmed**: threshold 0.7 is safe; confirmation has a material near cost; 3-of-3 is not keepable | Yes; threshold 0.7 first |
| H9 | Normal OSR−SR gap and Never-Stop OSR measure the same thing | Normal M0 OSR−SR: 5.0/1.7/0.0 pp by Short/Medium/Long (overall 2.22 pp; 4/180 reached radius but did not succeed). Never-Stop OSR: 60.0/46.7/45.0% (overall 50.6%); Never-Stop − Normal OSR = 13.3/15.0/30.0 pp | **Excluded / clarified**: normal OSR−SR = reached but did not stop/succeed; Never-Stop OSR = movement-only reachability with success impossible by construction. The large gap is Stop-policy cost, not the OSR−SR gap | No (conceptual clarification) |
| H10 | Stop fixes improve trajectory quality (nDTW) | 2-of-2 @0.5 nDTW +0.0006 overall (CI includes 0); Long +0.0321 (CI >0); Short −0.0167, Medium −0.0137 (CIs include 0). Threshold 0.7 +0.0003; R2 +0.0069; 3-of-3 −0.0334 | **Partially**: success/OSR/SPL improve; nDTW is flat overall and only Long improves; Short/Medium are flat/slightly down | Yes, but nDTW needs a separate movement/goal-current study |
| H11 | M1 static visual fusion inherits the Stop improvement | Not tested in Stage-2d. M1 Stage-2a baseline: SR 27.78/OSR 30.56/SPL 0.2778/nDTW 0.3648; Long SR 20.0, nDTW 0.2247 | **Open** | Yes: M1 + threshold 0.7 eval-only (recommended next) |

## 5. Stop-error diagnosis

### 5.1 Why does Long have more false Stops?

Long episodes start at 400–550 m (vs Short 150–250 m, Medium 250–400 m), so
they make more far-distance decisions. On B0:

- Far>40 decisions per episode: Short 117.3, Medium 138.8, Long 140.9
  (Stage-2c); far>40 Stop rate: Short 0.356%, Medium 0.357%, Long 0.407%.
- Far>40 false Stops per episode: Short 0.400, Medium 0.483, Long 0.567.
- Premature-Stop rate: Short 48.33%, Medium 56.67%, Long 65.00%.
- Long Stop median distance: 236.8 m; Goal visible at 77.1% of Long stops;
  mean Arrival p ≈ 0.72–0.78.

Long is harder because it has **more exposure** (more far decisions per
episode) and a slightly higher per-decision far Stop rate. Its false Stops
are high-confidence and often with the Goal visible. This is a cumulative
Arrival/Stop decision problem, not a visibility problem.

### 5.2 What is the main source of the error?

The primary **mechanism** is continuous decision risk accumulation: a
per-decision far FPR of ~0.37% over ~130 far decisions yields ~0.48
expected far false Stops per episode, plus ~0.08 boundary false Stops,
matching the observed ~0.57 premature-Stop rate.

The **source** of each risky decision is the high Arrival probability on a
small high-confidence far subset (the head's mapping), not a lack of visual
signal:

- The 104-D features separate on-policy hard far states from near positives
  on held-out episodes with AUC 0.783; the B0 logit alone is at chance
  (AUC 0.502).
- The high-p far states differ in `best_y` (Cohen d −1.32 vs near
  positives), `weighted_y` (−1.09), entropy (+0.66) and `mean_match`
  (+0.52).
- Goal invisibility is not the cause: the Goal is visible at 65.7–77.1% of
  Stops.

Training coverage is the third factor: the original 3% random far-negative
retention did not expose the head to the high-p far tail. The 3%→10%
increase added mostly easy negatives; the off-policy hard negatives (R2)
are the effective coverage fix. The remaining unsolved part is that the
closed-loop cumulative reduction is partial (far-40 stops 87→76, −12.6%),
because R2's longer trajectory visits more high-p far states and the
correction is not a 4× per-decision reduction.

### 5.3 Is the `pos_weight` issue fully excluded?

Yes. Stage-2c showed that plain BCE recovers B0-level SR while the
Stage-2b `pos_weight` variants collapse. Stage-2d uses plain
`binary_cross_entropy_with_logits` with no `pos_weight`; a unit test asserts
that the training source contains no `pos_weight=`. The issue is excluded
and the protocol is locked to plain BCE.

### 5.4 Why did 3%→10% random far negatives not improve closed-loop SR?

Because the added negatives were mostly already easy. B2 added 2,915 far
negatives, but only 22 had B0 p≥0.5 (mean p 0.031). The extra data moved
offline far-40 FPR from 0.67% to 0.50%, but the on-policy far Stop rate only
from 0.365% to 0.347% (5% relative), which is far below the 4–5× reduction
needed. The high-probability far tail was unchanged; the added negatives
did not teach the head the hard states.

### 5.5 Did Stage-2d off-policy hard negatives actually reduce the policy's own high-confidence false Stops?

Yes, but only partially. R2 vs R3 is a controlled ablation (same 3,393
parameters, 3,420 rows, initialization and budget; only the 355 far
negatives differ):

- Dev Val natural high-p far>40: B0 98; R2 25 (73 pushed below 0.5, 74.5%);
  R3 98 (0%). Far-40 FPR 0.672%→0.171% (R2) vs 0.713% (R3).
- On R2's own closed-loop trajectory, 113/188 (60.1%) high-p base far
  states are pushed below 0.5; high-p far mean p 0.726→0.495. R3 pushes
  0/85 below 0.5.
- Closed-loop: R2 far-40 stops 87→76 (−12.6%), far>100 78→69, median stop
  distance 102.3→66.5 m, SR 28.89→32.22, OSR 31.11→36.67. R3 is a no-op
  (50/180 successes, far-40 0.483).

So the hard-negative residual genuinely reduces high-confidence false Stop
probability on the policy's own states, but the trajectory is longer and
visits more high-p far states (188 vs 87), so the cumulative far-stop
reduction is 12.6%, not 74.5%. The near cost (reach-no-success 4→8, one
success lost) is the other limiter.

### 5.6 Do higher thresholds / confirmation cause near-goal missed Stops, more timeouts and lower nDTW?

| Policy | Near-20 Stop rate | Near stops/steps | Reach-no-success | Timeout | Success losses | nDTW |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 48.6% | 52/107 | 4/180 | 14.44% | — | 0.3570 |
| threshold 0.7 | 41.6% | 62/149 | 7/180 | 16.67% | 1 | 0.3573 |
| 2-of-2 @0.5 | 30.1% | 65/216 | 9/180 | 18.89% | 3 | 0.3576 |
| 2-of-2 @0.6 | 26.5% | 63/238 | 11/180 | 21.67% | 5 | 0.3483 |
| 3-of-3 @0.5 | 17.2% | 61/354 | 18/180 | 25.00% | 9 | 0.3237 |

Yes: higher thresholds/confirmation cause near-goal missed Stops and more
timeouts. The near-20 Stop **rate** falls, but the absolute near stops rise
because the policy stays in the near region longer; the real cost is
reach-no-success (4→7/9/11/18) and lost successes. nDTW is flat for
threshold 0.7 and 2-of-2 @0.5, but drops for 2-of-2 @0.6 and 3-of-3. The
safe rule is threshold 0.7; 2-of-2 @0.5 is the aggressive rule; 3-of-3 is
not keepable.

### 5.7 Normal OSR−SR gap vs Never-Stop counterfactual

These are different quantities and must not be conflated:

| Band | Normal OSR | Normal SR | OSR−SR (reached but no success) | Never-Stop OSR (movement-only reach) | Never-Stop − Normal OSR (Stop-policy cost) |
|---|---:|---:|---:|---:|---:|
| Short | 46.7% | 41.7% | 5.0 pp | 60.0% | 13.3 pp |
| Medium | 31.7% | 30.0% | 1.7 pp | 46.7% | 15.0 pp |
| Long | 15.0% | 15.0% | 0.0 pp | 45.0% | 30.0 pp |
| Overall | 31.1% | 28.9% | 2.2 pp (4/180) | 50.6% | 19.5 pp |

- Normal OSR−SR is the small "reached the radius but did not stop/succeed"
  gap: 4/180 overall, 3 Short, 1 Medium, 0 Long.
- Never-Stop OSR is the movement-only reachability with Stop disabled;
  success is impossible by construction (Timeout 100%). It is not a success
  rate.
- The large gap is Never-Stop OSR − Normal OSR: premature Stops prevent the
  policy from reaching the radius. This is the Stop-policy cost, not the
  OSR−SR gap.

## 6. Model arrangement

- Performance baseline: Transformer-M M0. The original M0 weights are
  preserved (sha256
  `1c496fa8bd880592089ad85408ad7244c4c3e95817c6c4fdd23a7f5ecd5bd9e4`).
- M1 static visual fusion is preserved:
  `runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z/Transformer-M/M1/best.pt`
  (sha256 `6cac7883419ebeae172b6f0e9dc1ddf6b87e4c0dc7f8994bb3e185e044287b00`).
  Stage-2a M1 baseline: SR 27.78%, OSR 30.56%, SPL 0.2778, nDTW 0.3648,
  premature 55.0%, Timeout 17.22%; Short/Medium/Long SR
  40.0/23.3/20.0%.
- M2, Depth and MEC BEV remain paused.
- No Small, Transformer-S, Medium or Large training is started.
- No M0/M1 weights or historical results are overwritten.

## 7. Final judgment

**A — Stage-2d has significantly improved the Stop decision and overall
closed-loop navigation ability, and M1 validation should be the next step.**

Conditions:
- The strongest **safe** result is threshold 0.7: +5.56 pp SR (62 vs 52
  successes), +7.22 pp OSR, +0.0555 SPL, with a modest near cost (7 vs 4
  reach-no-success, 1 success lost, Timeout +2.22 pp). 2-of-2 @0.5 gives a
  larger +7.22 pp SR/+10.0 pp OSR/+0.0721 SPL but a material near cost
  (3 successes lost, 9 reach-no-success). The gains are significant
  (p=0.006/0.004), not small.
- The far-tail offline correction is real and causal: R2 pushes 74.5% of
  high-p far states below 0.5; R3 pushes 0%; closed-loop far-40 stops
  87→76. But the cumulative far reduction is 12.6%, not the 4–5× target,
  and the trajectory is longer. The far tail is improved, not solved.
- **nDTW is not broadly improved.** Overall nDTW is flat for the decision
  rules (0.3570→0.3573–0.3576); R2 is +0.0069; 2-of-2 @0.5 improves Long
  nDTW (+0.032) but Short/Medium are slightly down. SPL improves more
  consistently. If the priority is trajectory quality rather than success,
  a movement/Goal-Current study is still needed.
- Single seed and Dev Val 180 only; Final Test 300 is sealed. A confirmation
  seed is required before deployment.

This is **not B** because the SR/OSR/SPL gains are large and statistically
significant, and the near cost is manageable for threshold 0.7. It is not
**C** because Stop fixes clearly improve success; the far tail is partially
reducible and the hard-mining control is causal.

## 8. Recommended next experiment (one)

**M1 + threshold 0.7, eval-only (no training).**

Why this one:
- M1 is the existing next candidate and has higher nDTW (0.3648 vs 0.3570)
  and higher Long SR (20.0% vs 15.0%) than M0, despite a lower overall SR.
- The safe Stop improvement is threshold 0.7; applying it to M1 is the
  cheapest way to test whether the Stop gain transfers to the static visual
  fusion architecture. No training is required, and no M1 weights are
  overwritten.

Code changes:
- Add `v83_spatial_stage2/stage2d_m1_threshold_eval.py` (or add
  `--mode {M0,M1}`/`--checkpoint` to the R4 evaluator).
- Reuse `evaluate_mode("M1", "Transformer-M", M1_CHECKPOINT, ...)` from
  `v83_spatial_training/eval_closedloop_spatial.py`, which already builds
  the M1 static grounding maps and resets the tracker per episode.
- Run the same Dev Val 180 specs at threshold 0.5 (reference) and 0.7.
- Save per-episode CSVs, paired comparison vs M1@0.5 (bootstrap CI + sign
  test), per-band metrics, success counts, and a short report.
- Do not modify M1 weights or the Stage-2a artifacts.

Estimated cost:
- No training. M1 adds static grounding maps per step; estimate ~6–8 min
  per 180-episode threshold on the RTX 5090, so ~15–20 min for 0.5 + 0.7,
  plus ~15 min analysis/report. Total ~30–40 min.

Success threshold (development gate):
- M1@0.7 vs M1@0.5: SR and OSR improvement ≥ 3 pp each; SPL no regression
  (Δ ≥ 0); nDTW no regression (Δ ≥ −0.002, and the 95% CI lower bound not
  below −0.01); premature-Stop reduction ≥ 3 pp; Timeout increase ≤ 2.5 pp;
  reach-no-success increase ≤ 3 episodes and ≤1 M1@0.5 success lost;
  paired SR wins > losses (sign test p < 0.10 or bootstrap CI lower bound
  > 0).
- Also compare M1@0.7 to M0@0.7 (34.44% SR / 38.33% OSR / 0.3432 SPL):
  M1 should be at least comparable in SR/OSR and better in nDTW (0.3648
  baseline) to justify the M1 path.
- If M1@0.7 does not reproduce the M0 gain, do **not** start M1 training;
  first analyze M1's Stop features and its near/far error split.

## 9. Data/evaluation protocol compliance

- Dev Val 180 is preserved unchanged (60/60/60); no re-generation or
  overwrite.
- Final Test 300 (100/100/100) is sealed: not read, not evaluated, not used
  for model selection.
- Overall SR/OSR/SPL/nDTW are computed directly over all 180 executed
  episodes; no 30/50/20 weighting. The audit checks both the direct mean and
  the 30/50/20 weighted value and confirms they differ.
- Overall and Short/Medium/Long results are reported with success counts.
- Dev Val is called development validation, not Final Test.
- The task is called image-goal UAV navigation (CityNav/CityRefer), not
  language-instruction VLN.
- Train-600 is balanced 200/200/200; Dev Val is balanced 60/60/60. The
  training data can use balanced difficulty; paper reporting must state the
  construction and the evaluation weighting explicitly.

## 10. Artifacts

- Audit JSON:
  `runs/v83_spatial_stage2/stage2d/stage2d_audit.json` (all checks pass).
- Stage-2d run:
  `runs/v83_spatial_stage2/stage2d/r4_20261019T002626Z/`;
  `runs/v83_spatial_stage2/stage2d/calibration_20261019T005253Z/`.
- Feature probe:
  `runs/v83_spatial_stage2/stage2d/feature_probe_20261019T004443Z.json`.
- Generated tables:
  `runs/v83_spatial_stage2/stage2d/stage2d_summary.md`.
- Main report: `v83_spatial_stage2/V83_STAGE2D_REPORT.md`.
- Audit code: `v83_spatial_stage2/stage2d_audit.py`.
- Tests: `python -m unittest v83_spatial_stage2.test_stage2d -v` (25 pass).
