# V8.3 Stage-2f — Goal Reacquisition / Movement Recovery Diagnosis

- Date: 2026-10-19
- Branch: `experiment/v83-spatial-foundation`
- Scope: eval-only diagnosis of Goal Reacquisition / Movement Recovery on the
  saved M0@0.7 and M1@0.7 trajectories. **No training, no checkpoint
  modification, no M1-Dir/M2/Depth/MEC-BEV training.**
- Data: Dev Val 180 (Short/Medium/Long 60/60/60); 55,414 replayed states
  (180 M0 + 180 M1 trajectories). Final Test 300 is sealed and not read.
- M0 checkpoint:
  `runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z/Transformer-M/M0/best.pt`
  (sha256 `1c496fa8bd880592089ad85408ad7244c4c3e95817c6c4fdd23a7f5ecd5bd9e4`).
- M1 checkpoint:
  `runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z/Transformer-M/M1/best.pt`
  (sha256 `6cac7883419ebeae172b6f0e9dc1ddf6b87e4c0dc7f8994bb3e185e044287b00`).

## Key conclusions (TL;DR)

1. **Goal invisibility is the dominant movement failure mode.** On M0 move
   states, movement accuracy against the heading oracle is 44.9% when the goal
   is visible and **25.1% when it is invisible** (move error 44.9% vs 74.9%;
   mean |heading error| 12.3° vs 117.4°; wrong-run>=2 15.7% vs 56.1%). The gap
   is consistent in Short, Medium and Long and in the M1 trajectory.
2. **The observable Current–Goal features separate visibility only
   moderately.** On M0 move states the best features are `sim_max` (AUC 0.685),
   `baseline_top1` (0.641), `heatmap_max_top1` (0.638) and
   `baseline_entropy` (0.637); Short is the most separable (`sim_max` 0.764),
   Long is ~0.69–0.70, Medium ~0.57–0.67. No feature reaches the ~0.8 AUC
   needed for a high-precision trigger.
3. **Yaw-aligned temporal evidence does not identify visibility.** After
   LEFT/RIGHT in-place turns, `temporal_consistency` AUC is 0.529 overall,
   0.546 in Medium and **0.446 in Long**; `innovation_mean_abs` is 0.488. The
   history trigger does not improve on the current-frame trigger. FORWARD
   translation was never warped, as required.
4. **The best held-out trigger is still not precise enough.** An even/odd
   episode sweep selected `heatmap_max_top1 < 0.02 OR heatmap_max_entropy >
   0.98`, `K=1`. On held-out odd episodes: trigger rate 24.7%, invisibility
   precision 50.9%, recall 39.5%; unreliable-localization precision 67.3%,
   recall 31.6%. The default operating point is much worse (trigger rate
   76.2%, precision 36.0% ≈ the 34.8% base rate).
5. **The recovery overlay is harmful in closed loop.** On 12 episodes per band
   (36 total), M0@0.7 baseline SR is 36.1%. A 12-step-budget recovery gives
   SR 19.4% (−16.67 pp, 1W/7L, CI [−30.56, −2.78]) and Timeout 25.0% vs
   13.9%. Without a budget it is worse: SR 16.7% (−19.44 pp, 0W/7L, CI
   [−33.33, −8.33]), Timeout 52.8%. Recovery actions are 59.2% wrong against
   the heading oracle, re-visibility within 10 steps is 11.0%, and **0% of
   triggers enter the 20 m radius within 40 steps**.
6. **Premature Stop is a related but separate lever.** 88/150 M0 stops are
   premature (58.7%); premature stops are low-confidence more often than
   correct stops (heat top1 median 0.066 vs 0.248, entropy 0.952 vs 0.792) and
   only 43.2% are goal-visible vs 100% of correct stops. However, a naive
   low-confidence stop veto at the default threshold would also veto 40.3% of
   correct stops; confidence alone is not a safe veto signal.
7. **Decision: do not adopt this rule-based recovery policy and do not start
   any training.** The offline evidence is insufficient for a reliable
   no-leakage recovery trigger, and the closed-loop smoke is clearly harmful.
   The next step should be a better observable target-presence/localization
   representation (or temporal memory with validated forward-motion handling),
   not another hand-written scan rule. No M1-Dir/M2/Depth/BEV training was
   started.

## 0. Observables and no-leakage contract

All controller-side features are computed from run-time inputs:

- Current RGB tokens and the three Goal-reference tokens.
- The 3×8×8 Goal-Current heatmap.
- The parameter-free baseline/coherent spatial maps.
- The previous coherent map and previous executed action, used only for the
  yaw-aligned in-place-turn warp (LEFT/RIGHT). FORWARD translation is never
  warped with a fixed pixel shift.
- The frozen M0 movement/arrival outputs.

Feature groups:

| Group | Features |
|---|---|
| Goal-Current similarity | `sim_max`, `sim_top5_mean`, `sim_std`, per-scale max |
| Heatmap confidence | per-scale and scale-max entropy/top1/top5 mass/centroid/peak |
| Spatial maps | baseline/coherent entropy/top1/top5 mass/centroid/peak/raw std, left/right mass |
| Temporal evidence | `temporal_valid`, `temporal_consistency`, innovation mean/max, temporal entropy/top1/centroid |
| Movement confidence | M0 arrival probability, move top1, move margin, move action |

True labels (`goal_visible`, true goal projection, true distance, true
heading error, oracle action) are stored in separate columns and are used
**only** for offline evaluation. The recovery policy accesses features through
`_get`, which raises for every forbidden true-label key (`FORBIDDEN_KEYS`).
`test_stage2f.py` checks that observable keys are disjoint from the forbidden
set and that the policy runs on observable-only dictionaries.

The state/action replay is the same exact replay as Stage-2e: 360/360
trajectory passes, 55,414 states, zero distance/action/arrival mismatch. The
feature collector re-verifies the owner model's logged move action and arrival
probability at every state.

## 1. Goal visible vs invisible

M0 move states (`m0_arrival_probability < 0.7`); true goal visibility is an
offline label only.

| Band | Goal | States | Share | Move err | Missed | Wrong | Spurious | Wrong-run>=2 | \|Heading\| | Heat top1 | Heat ent | Coh top1 | Temporal cons |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | visible | 17,255 | 65.2% | 44.9% | 17.0% | 7.8% | 20.1% | 15.7% | 12.3° | 0.170 | 0.872 | 0.132 | 0.708 |
| Overall | invisible | 9,223 | 34.8% | **74.9%** | 48.9% | 26.0% | 0.0% | **56.1%** | **117.4°** | 0.127 | 0.901 | 0.100 | 0.688 |
| Short | visible | 5,363 | 63.8% | 38.3% | 8.3% | 1.4% | 28.6% | 7.2% | 8.8° | 0.203 | 0.838 | 0.122 | 0.731 |
| Short | invisible | 3,048 | 36.2% | **87.1%** | 67.7% | 19.4% | 0.0% | **75.0%** | **146.0°** | 0.128 | 0.895 | 0.091 | 0.686 |
| Medium | visible | 5,276 | 57.4% | 44.7% | 16.7% | 10.1% | 17.9% | 15.0% | 12.8° | 0.152 | 0.894 | 0.145 | 0.705 |
| Medium | invisible | 3,909 | 42.6% | **72.2%** | 41.6% | 30.6% | 0.0% | **52.6%** | **115.3°** | 0.162 | 0.878 | 0.112 | 0.682 |
| Long | visible | 6,616 | 74.5% | 50.5% | 24.4% | 11.3% | 14.9% | 23.2% | 14.6° | 0.158 | 0.881 | 0.130 | 0.685 |
| Long | invisible | 2,266 | 25.5% | **63.4%** | 36.3% | 27.1% | 0.0% | **36.6%** | **82.6°** | 0.066 | 0.947 | 0.090 | 0.700 |

The M1 trajectory shows the same pattern (move error 44.6% visible vs 73.6%
invisible; |heading| 11.0° vs 112.9°; wrong-run>=2 13.0% vs 52.4%; heat top1
0.204 vs 0.065). It is a property of the environment/observability, not of the
M0 checkpoint.

Where the invisible states occur (M0, all states):

| Band | States | Invisible | Invisible rate | At Stop | At move | Movable share |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 26,628 | 9,273 | 34.8% | 50 | 9,223 | **99.5%** |
| Short | 8,462 | 3,066 | 36.2% | 18 | 3,048 | 99.4% |
| Medium | 9,236 | 3,928 | 42.5% | 19 | 3,909 | 99.5% |
| Long | 8,930 | 2,279 | 25.5% | 13 | 2,266 | 99.4% |

So the recovery opportunity is not blocked by terminal Stop states: almost all
invisible states are on executed move steps. The limitation is the trigger and
the recovery action, not the absence of actionable states.

Consecutive wrong-turn diagnostics (M0 move states, oracle for evaluation
only):

| Band | Move states | Wrong-run>=2 | Wrong-run>=3 | Run starts | Start low-conf | Start triggered | Pre-step triggered | \|Heading\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 26,478 | 29.8% | 26.6% | 6,726 | 4,651 (69.2%) | 4,281 (63.7%) | 4,289 (63.8%) | 48.9° |
| Short | 8,411 | 31.8% | 29.0% | 2,025 | 1,406 | 1,344 | 1,346 | 58.5° |
| Medium | 9,185 | 31.0% | 27.4% | 2,314 | 1,467 | 1,320 | 1,322 | 56.5° |
| Long | 8,882 | 26.6% | 23.3% | 2,388 | 1,779 | 1,617 | 1,621 | 32.0° |

The trigger fires before most wrong-turn runs, but it also fires on most move
states (see §2), so this is not evidence of selectivity.

## 2. Can observable features identify invisible / unreliable localization?

Feature AUC for goal **visibility** on M0 move states (higher = better; entropy
and innovation are negated so every AUC > 0.5 is informative). Episode-level
bootstrap 95% CI (200 resamples over episodes).

| Band | Feature | AUC(visible) | 95% CI | N |
|---|---|---:|---|---:|
| Overall | `sim_max` | **0.685** | [0.608, 0.755] | 26,478 |
| Overall | `baseline_top1` | 0.641 | [0.547, 0.738] | 26,478 |
| Overall | `heatmap_max_top1` | 0.638 | [0.551, 0.725] | 26,478 |
| Overall | `baseline_entropy` | 0.637 | [0.535, 0.745] | 26,478 |
| Overall | `coherent_top1` | 0.625 | [0.542, 0.715] | 26,478 |
| Overall | `temporal_consistency` | 0.529 | [0.430, 0.641] | 14,778 |
| Short | `sim_max` | **0.764** | [0.650, 0.885] | 8,411 |
| Short | `heatmap_max_top1` | 0.678 | [0.496, 0.850] | 8,411 |
| Short | `temporal_valid` | 0.675 | [0.569, 0.766] | 8,411 |
| Short | `heatmap_max_entropy` | 0.650 | [0.470, 0.832] | 8,411 |
| Medium | `sim_max` | **0.672** | [0.555, 0.793] | 9,185 |
| Medium | `baseline_top1` | 0.629 | [0.513, 0.722] | 9,185 |
| Medium | `heatmap_max_top1` | 0.572 | [0.446, 0.702] | 9,185 |
| Medium | `temporal_consistency` | 0.546 | [0.422, 0.673] | 5,285 |
| Long | `baseline_entropy` | **0.698** | [0.474, 0.864] | 8,882 |
| Long | `baseline_top1` | 0.697 | [0.437, 0.906] | 8,882 |
| Long | `heatmap_max_top1` | 0.694 | [0.530, 0.814] | 8,882 |
| Long | `coherent_entropy` | 0.693 | [0.504, 0.843] | 8,882 |
| Long | `temporal_consistency` | 0.446 | [0.299, 0.635] | 4,889 |

The best single feature is ~0.64–0.69 AUC overall and up to 0.764 in Short;
Long is the easiest band by confidence features, Medium the hardest. The
arrival probability is not a visibility signal
(AUC 0.442 overall; higher arrival p is associated with invisible, because the
base head can fire prematurely). Temporal consistency is near chance overall
and below chance in Long.

Trigger precision/recall for **invisibility** at the default thresholds
(`top1 < 0.15 OR entropy > 0.85`, K=2):

| Band | Trigger | Trigger rate | Invisible base rate | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Overall | current-step low-conf | 76.2% | 34.8% | **36.0%** | 78.7% | 49.4% |
| Overall | K=2 history | 72.2% | 34.8% | **36.4%** | 75.4% | 49.0% |
| Overall | temporal-only | 5.9% | 34.8% | 37.9% | 6.4% | 10.9% |
| Short | current-step low-conf | 72.8% | 36.2% | 37.9% | 76.1% | 50.6% |
| Short | temporal-only | 4.5% | 36.2% | 29.5% | 3.6% | 6.5% |
| Medium | current-step low-conf | 73.9% | 42.6% | 42.4% | 73.6% | 53.8% |
| Medium | temporal-only | 7.1% | 42.6% | **58.2%** | 9.7% | 16.6% |
| Long | current-step low-conf | 81.8% | 25.5% | 28.3% | 90.8% | 43.2% |

The default trigger fires on 76–82% of move states and its precision is
approximately the base rate; it is not a usable "lost" detector. The temporal
trigger is more precise in Medium but catches only 9.7% of invisible states.

Even/odd episode threshold sweep (thresholds selected on even episodes,
reported on held-out odd episodes):

| Selected rule | Split | Trigger rate | Invisible base | Invisible precision | Invisible recall | Unreliable precision | Unreliable recall |
|---|---|---:|---:|---:|---:|---:|---:|
| top1<0.02 OR ent>0.98, K=1 | even | 23.1% | 33.8% | 53.9% | 36.8% | 69.9% | 30.0% |
| top1<0.02 OR ent>0.98, K=1 | **odd (held out)** | **24.7%** | 31.8% | **50.9%** | **39.5%** | **67.3%** | **31.6%** |

The best held-out trigger is moderately better than chance but not a
high-precision signal: about half of its triggers are on goal-visible states,
and it catches only ~40% of invisible states.

## 3. Previous-frame and history evidence

- **Last known target side.** When the goal is visible and the observation is
  high-confidence (`heatmap_max_top1 >= 0.20`, entropy <= 0.85), the
  baseline-map centroid side has AUC 0.953 and balanced accuracy 0.873 for the
  true target side. At a stricter high-confidence threshold (`>=0.30`,
  entropy <= 0.75), AUC is 0.970 and balanced accuracy 0.906. High confidence
  occurs on 22.6% (0.20/0.85) or 13.5% (0.30/0.75) of M0 move states.
  So the last known side is reliable **conditional on high confidence**.
- **But the trigger that would use it is not reliable.** The low-confidence
  trigger fires on 72–82% of move states and is not selective; the
  side-history and scan-only policies are nearly identical in the smoke,
  confirming that the side history does not rescue the policy.
- **Yaw-aligned previous-frame evidence is insufficient.** The warp is only
  valid after LEFT/RIGHT; temporal consistency is near chance for visibility
  (0.529 overall, 0.446 Long) and innovation is 0.488. It cannot substitute
  for target presence, especially after FORWARD motion where no validated warp
  exists.
- **Base-rate caution.** The goal is often on one side (the visible-state map
  centroid has a strong prior); raw side accuracy can look high while balanced
  accuracy is lower. The high-confidence conditional AUC above is the
  meaningful number, and it still does not solve the trigger problem.

**Answer for item 3:** previous-frame target position, the previous action and
yaw-aligned evidence are enough to *store a last known side*, but not enough
to build a reliable lightweight search/relocalization policy. The missing
piece is a trustworthy target-presence signal on move states, not a better
turn rule.

## 4. Recovery policy candidate and offline counterfactual

The candidate is an eval-time overlay on the frozen M0@0.7 controller. It does
not change the M0 Stop threshold: when the base arrival probability is
>= 0.7 the episode still ends with STOP. The overlay only chooses among
L/F/R when the base policy would move.

Observable rule:

1. **Confidence.** `low = heatmap_max_top1 < tau_top1 OR
   heatmap_max_entropy > tau_entropy`; `high = heatmap_max_top1 >= h_top1 AND
   heatmap_max_entropy <= h_entropy`.
2. **Last known target.** While `high`, store the baseline-map centroid side
   (`< -0.05` left, `> 0.05` right) and the centroid value. This uses only the
   observable Current-Goal spatial map.
3. **Trigger.** Enter recovery after `K` consecutive `low` move states, or
   when a yaw-aligned in-place-turn warp is inconsistent with the current
   coherent map (`temporal_valid` and `temporal_consistency < tau_cons`).
4. **Recovery action.** If a last high-confidence side exists, turn toward it;
   otherwise run a deterministic scan (12 steps LEFT, 24 RIGHT, 24 LEFT, then
   RIGHT). **No FORWARD while recovering.**
5. **Re-acquire.** One `high` step exits recovery and returns to the base
   action.

Variants compared: `side_history` (rule above) and `scan_only` (same trigger,
ignores the side history and always scans). A `max_recovery_steps` budget is an
optional cap; both 0 (unlimited) and 12 were tested.

Offline counterfactual on the 55,414 saved states (M0 move states, privileges
oracle only for scoring):

| Policy | Triggers | Changed actions | Beneficial | Harmful | Net | Turn rate |
|---|---:|---:|---:|---:|---:|---:|
| side_history | 19,136 | 14,457 | 4,728 | 5,691 | **−963** | 100% |
| scan_only | 19,136 | 14,409 | 4,743 | 5,559 | **−816** | 100% |

The static counterfactual already leans harmful overall, with a small positive
net only in Medium (+567 for side_history) and negative net in Short (−882)
and Long (−647).

## 5. Closed-loop smoke

36 episodes (12 per band, evenly spaced over each band), same seeds and network
layouts as the saved Dev-Val episodes. The baseline is the frozen M0@0.7
policy; it reproduces the saved per-episode outcomes exactly (0 mismatches).
The recovery runs use the held-out operating point
`top1<0.02 OR entropy>0.98, K=1`, high-confidence side threshold
`top1>=0.20, entropy<=0.85`, and the temporal trigger disabled
(`consistency=-1`) because the offline AUC showed it was uninformative.

| Policy | Budget | SR | OSR | SPL | nDTW | Premature | Timeout | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline M0@0.7 | — | **36.1%** | 38.9% | 0.361 | 0.409 | 50.0% | 13.9% | 129.0 |
| side_history | 12 | 19.4% | 19.4% | 0.184 | 0.168 | 55.6% | 25.0% | 176.4 |
| scan_only | 12 | 19.4% | 19.4% | 0.184 | 0.168 | 58.3% | 22.2% | 165.4 |
| side_history | 0 (unlimited) | 16.7% | 16.7% | 0.171 | 0.171 | 30.6% | 52.8% | 243.2 |
| scan_only | 0 (unlimited) | 16.7% | 16.7% | 0.171 | 0.171 | 33.3% | 50.0% | 242.9 |

Paired vs baseline (per episode):

| Policy | Budget | Band | SR Δ (W/L, 95% CI) | Timeout Δ | Premature Δ |
|---|---:|---|---|---:|---:|
| side_history | 12 | Overall | **−16.67 pp (1W/7L) [−30.56, −2.78]** | +11.11 pp | +5.56 pp |
| side_history | 12 | Short | −16.67 pp (0W/2L) [−41.67, 0.00] | +25.00 pp | −8.33 pp |
| side_history | 12 | Medium | −16.67 pp (1W/3L) [−50.00, +16.67] | 0.00 pp | +16.67 pp |
| side_history | 12 | Long | −16.67 pp (0W/2L) [−41.67, 0.00] | +8.33 pp | +8.33 pp |
| side_history | 0 | Overall | **−19.44 pp (0W/7L) [−33.33, −8.33]** | +38.89 pp | −19.44 pp |
| scan_only | 12 | Overall | −16.67 pp (1W/7L) [−30.56, −2.78] | +8.33 pp | +8.33 pp |

Trigger and recovery diagnostics (side_history):

| Budget | Triggers | Episodes with trigger | Trigger invisible rate | Re-visible <=10 steps | Enter <20 m <=40 steps | Recovery actions | Recovery error rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 1,073 | 23/36 | 82.1% | 11.0% | **0.0%** | 1,311 | 59.2% |
| 0 | 28 | 23/36 | 17.9% | 40.0%* | **0.0%** | 7,217 | 53.6% |

\* Only 5 unlimited-budget triggers were at true-invisible states; the
unlimited policy then stays in recovery for hundreds of steps.

By band with the 12-step budget (side_history): Short SR 50.0%→33.3%,
Medium 33.3%→16.7%, Long 25.0%→8.3%; re-visible within 10 steps is 18.4% /
18.9% / 3.6%; entry into 20 m within 40 steps is 0% in every band. Recovery
turns are wrong against the heading oracle 50.4% / 60.7% / 63.6% of the time.

Interpretation:

- The trigger fires very often once the policy is active (1,073 triggers over
  23 episodes with the 12-step budget), replacing forward motion with turns.
- The recovery action does not re-acquire the target: only 11% become visible
  within 10 steps and **0% enter the 20 m radius within 40 steps**.
- The unlimited budget is worse: the policy never leaves recovery in many
  episodes and times out (52.8% Timeout vs 13.9% baseline).
- The side history does not help; `side_history` and `scan_only` have nearly
  identical episode outcomes.

## 6. Medium vs Long

| Band | Invisible share of move states | Best visibility AUC | Trigger invisible precision (default) | Smoke SR baseline → budget-12 | Re-visible <=10 | Enter <20 m <=40 |
|---|---:|---:|---:|---|---:|---:|
| Short | 36.2% | **0.764** (`sim_max`) | 37.9% | 50.0% → 33.3% | 18.4% | 0% |
| Medium | **42.6%** | 0.672 (`sim_max`) | **42.4%** | 33.3% → 16.7% | 18.9% | 0% |
| Long | 25.5% | **0.698** (`baseline_entropy`) | 28.3% | 25.0% → 8.3% | 3.6% | 0% |

- **Medium** has the highest invisible share (42.6%) and the largest absolute
  number of invisible move states; its observable AUC is intermediate
  (0.57–0.67), and the default trigger precision is the best of the three
  bands (42.4%) but still close to the 42.6% base rate. The recovery policy
  cuts Medium SR roughly in half (33.3→16.7) because it turns instead of
  making forward progress.
- **Long** has the best single-feature AUC (~0.70) but the lowest invisible
  share (25.5%) and the weakest recovery outcome: Long SR 25.0→8.3, and only
  3.6% of triggers become re-visible within 10 steps. Long failures are mostly
  far from the goal; a local in-place scan cannot recover them.
- **Short** is similar to Medium in invisible share but has the strongest
  baseline SR; the recovery policy still loses 16.7 pp.

## 7. Decision and next steps

**Decision: the offline evidence is insufficient for a reliable no-leakage
rule-based Goal-Reacquisition policy, and the closed-loop smoke is harmful.
Do not adopt this policy; do not start a full training run.**

Why:

1. On Medium/Long move states no observable feature exceeds ~0.70 AUC for
   visibility (Short reaches 0.764 with `sim_max`); the best held-out trigger
   has ~51% invisibility precision and ~40% recall.
2. The policy's recovery action (turn-only scan) does not re-acquire the
   target: 11% re-visible within 10 steps and 0% entry into 20 m within 40
   steps; 59% of recovery turns are wrong against the heading oracle.
3. Both budgeted and unlimited variants reduce SR by ~17–19 pp and increase
   Timeout; the effect is consistent across Short/Medium/Long.
4. The yaw-aligned temporal evidence is near chance and cannot replace target
   presence, especially after FORWARD motion where no validated warp exists.
5. The last-known side is reliable only when confidence is high, but the
   low-confidence trigger is not selective enough to use it.

Recommended next steps (all future work; none started here):

- **Target-presence / localization representation.** A calibrated head trained
  on observable features to predict whether the target is currently
  localizable, with episode-level splits, is the missing signal. The current
  hand-crafted features are not sufficient; a learned representation may be.
- **Temporal memory with validated forward-motion handling.** M2-style
  temporal belief is the principled way to maintain the last known target
  through FORWARD motion. It requires training and is out of scope for this
  round.
- **Stop-veto diagnostic only after a better presence signal.** Premature
  stops are strongly associated with low visual confidence
  (heat top1 0.066 vs 0.248 for correct stops; 58.7% of stops premature), but
  a naive confidence veto would also veto 40.3% of correct stops. Do not
  implement a low-confidence stop veto without a target-presence signal.
- **Do not continue with Stop-threshold sweeps.** The Stage-2d/2e work already
  showed the binding issue is not the threshold alone; this stage confirms the
  movement/recovery side also lacks a reliable observable trigger.

## 8. Artifacts and compliance

- No training; no checkpoint modification. M0/M1 sha256 values are recorded in
  the provenance files.
- Final Test 300 was not read; all data are Dev Val 180.
- No M1-Dir, M2, Depth, MEC-BEV or other MEC training was started.
- The only closed-loop runs are the 36-episode baseline and recovery smoke;
  all other analyses are exact-state replays plus forward passes.

Artifacts:

- Feature collection:
  `runs/v83_spatial_stage2/stage2f/collect_20261019T024000Z/`
  (`features.csv` 55,414 states, `replay_check.json`, `provenance.json`).
- Offline analysis:
  `runs/v83_spatial_stage2/stage2f/analysis_20261019T025000Z/`
  (`visible_vs_invisible.csv`, `feature_auc.csv`, `trigger_metrics.csv`,
  `wrong_runs.csv`, `stop_states.csv`, `visibility_actions.csv`,
  `policy_counterfactual.csv`, `tables.md`, `summary.json`).
- Trigger sweep:
  `runs/v83_spatial_stage2/stage2f/sweep_20261019T025500Z/`
  (`sweep.csv`, `summary.json`).
- Closed-loop smoke (fixed JSON output):
  `runs/v83_spatial_stage2/stage2f/smoke_budget12_20261019T031000Z/` and
  `runs/v83_spatial_stage2/stage2f/smoke_nobudget_20261019T031500Z/`
  (`*_per_episode.csv`, `steps/`, `events/`, `metrics.json`,
  `provenance.json`, `baseline_check.json` is inside `metrics.json`).
- Code:
  `v83_spatial_stage2/stage2f_features.py`,
  `stage2f_collect.py`, `stage2f_analyze.py`, `stage2f_policy.py`,
  `stage2f_recovery_eval.py`, `stage2f_trigger_sweep.py`,
  `test_stage2f.py`.
- Tests: 6 CPU-only tests pass (`python -m unittest -v
  v83_spatial_stage2.test_stage2f`).
