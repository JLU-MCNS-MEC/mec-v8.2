# V8.3 Full-Trajectory Depth Coverage Audit

- Date: 2026-10-18
- Branch: `experiment/v83-spatial-foundation`
- HEAD: `fc79dba` (depth fixes + tests)
- Model: frozen Transformer-M Best-50 + frozen shared RGB encoder
- Validation: 60 held-out V8.3 episodes, seeds 900000–900077
- Policies audited: Stop threshold 0.6 (best SR) and 0.9 (highest OSR among deployable thresholds)
- Raw results: `runs/v83_depth_prevalidation/20261018T015027817453Z/`
  - `trajectory_steps.csv` (30,715 step records)
  - `milestones.csv` (121 milestone records)
  - `episode_summary.csv` (120 episode-policy records)
  - `audit_summary.log`
- Audit script: `v83_depth_prevalidation/trajectory_depth_audit.py`
- Important: true Goal-region masks are **offline diagnostics only**; the model input remains RGB-only and never receives Goal coordinates, true distance, or oracle masks.

## 1. Method

For every step of both policies, the audit records:

1. **True Goal-region Depth** (simulator-only, offline):
   - Goal projection and silhouette mask from `env.goal_projection()` and the Goal landmark shape.
   - `target_depth_available`: any valid Depth pixel inside the Goal silhouette.
   - `target_valid_fraction`: fraction of Goal-silhouette pixels with valid Depth.
   - `target_goal_match_fraction`: fraction of Goal-silhouette pixels whose Depth is within ±1 m of the Goal slant range. This distinguishes real target returns from occluders/other objects.
2. **Visual-match-region Depth** (legal MEC-side signal):
   - Top-12 Current–Goal cosine-match patches; top-1 patch.
   - `visual_depth_available`, `visual_valid_fraction`, `visual_matched_validity`.
   - `visual_goal_iou`: IoU between the top-k patch mask and the true Goal silhouette.
   - `top1_overlap_goal`, `top1_contains_goal_center`.
   - `visual_goal_match_fraction`: fraction of visual-match pixels whose Depth is within ±1 m of Goal slant.
   - Mismatch flags:
     - `visual_depth_from_non_target`: visual region has Depth but zero IoU with Goal.
     - `target_missed_by_visual`: Goal region has Depth but visual region has zero IoU.
     - `visual_depth_not_goal`: visual region has Depth but <50% matches Goal slant.
     - `target_occluded`: Goal region has Depth but <50% matches Goal slant.
3. Horizontal distance milestones: first step entering 60 m, 40 m, 20 m.

## 2. Episode coverage

| Policy | Success | Reached 60 m | Reached 40 m | Reached 20 m | Mean steps |
|---|---:|---:|---:|---:|---:|
| 0.6 | 13/60 | 24/60 | 16/60 | 14/60 | 173.2 |
| 0.9 | 6/60 | 28/60 | 20/60 | 19/60 | 338.7 |

The 0.9 policy reaches 19 episodes (the same movement ceiling found by `never` in the earlier 60-episode evaluation), but succeeds in only 6 because most near-goal episodes time out without stopping.

## 3. Milestone Depth coverage

### 3.1 True Goal region (offline truth)

| Policy | Range | Episodes | Goal visible | Target Depth available | Target valid fraction* | Target vs Goal slant* |
|---|---:|---:|---:|---:|---:|---:|
| 0.6 | 60 m | 24 | 19 | 19 (79.2%) | 1.000 | 97.8% |
| 0.6 | 40 m | 16 | 16 | 16 (100%) | 1.000 | 100.0% |
| 0.6 | 20 m | 14 | 14 | 14 (100%) | 1.000 | 100.0% |
| 0.9 | 60 m | 28 | 23 | 23 (82.1%) | 1.000 | 98.1% |
| 0.9 | 40 m | 20 | 20 | 20 (100%) | 1.000 | 100.0% |
| 0.9 | 20 m | 19 | 19 | 19 (100%) | 1.000 | 99.9% |

\* Conditional on Goal visibility; `target_valid_fraction` and `target_goal_match_fraction` are 1.0 for every available target region at 40/20 m.

Interpretation:
- **The true target Depth is essentially perfect when the Goal is visible and within range.** At 40 m and 20 m it is available in 100% of episodes, with 100% valid pixels and correct slant range.
- The only 60 m shortfall is Goal not visible (5/24 for 0.6, 5/28 for 0.9); when visible, target Depth is still ~98% correct.
- Therefore the Depth sensor/geometry is **not** the limiting factor near the goal.

### 3.2 Visual-match region (legal MEC feature region)

| Policy | Range | Visual Depth available | Visual valid fraction | Matched validity | IoU with Goal | Top-1 contains Goal center | Visual Depth not Goal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | 60 m | 95.8% | 0.453 | 0.478 | 0.316 | 0.0% | 54.2% |
| 0.6 | 40 m | 93.8% | 0.557 | 0.573 | 0.318 | 6.2% | 43.8% |
| 0.6 | 20 m | 100% | 0.573 | 0.579 | 0.341 | 0.0% | 50.0% |
| 0.9 | 60 m | 96.4% | 0.458 | 0.481 | 0.314 | 0.0% | 57.1% |
| 0.9 | 40 m | 95.0% | 0.530 | 0.548 | 0.306 | 5.0% | 50.0% |
| 0.9 | 20 m | 100% | 0.551 | 0.556 | 0.318 | 0.0% | 52.6% |

`Visual Depth not Goal` = visual region has valid Depth, but <50% of its pixels are within ±1 m of the Goal slant range.

Key observations:
- Visual-match Depth is almost always **available** (94–100%), but its **content is often not the Goal**: at 20 m, 50.0% (0.6) / 52.6% (0.9) of episodes have a visual region dominated by non-Goal Depth.
- The top-1 match patch contains the Goal center in **0%** of 20 m cases; top-k IoU with the Goal silhouette is only ~0.32–0.34.
- `visual_depth_from_non_target` and `target_missed_by_visual` are rare (0–10% at milestones); the dominant failure is **partial/mixed matching**, not a complete wrong-object match.
- `target_occluded` is 0% at 40/20 m: when the Goal is visible, its Depth is not occluded. The mismatch comes from the visual selection, not the sensor.

### 3.3 Distance-bin view (all steps)

| Policy | Distance | Steps | Target Depth available | Target valid* | Visual Depth available | Visual not Goal |
|---|---:|---:|---:|---:|---:|---:|
| 0.6 | 0–20 m | 22 | 72.7% | 1.000 | 81.8% | 31.8% |
| 0.6 | 20–40 m | 150 | 97.3% | 1.000 | 98.0% | 44.7% |
| 0.6 | 40–60 m | 1,024 | 93.7% | 1.000 | 99.5% | 52.6% |
| 0.6 | >60 m | 9,195 | 38.0% | 0.736 | 79.7% | 43.7% |
| 0.9 | 0–20 m | 122 | 43.4% | 1.000 | 80.3% | 25.4% |
| 0.9 | 20–40 m | 442 | 88.2% | 1.000 | 95.2% | 33.0% |
| 0.9 | 40–60 m | 1,105 | 89.4% | 1.000 | 98.4% | 50.1% |
| 0.9 | >60 m | 18,655 | 24.9% | 0.698 | 65.7% | 30.8% |

\* Conditional on target region having any valid Depth.

- At 0–20 m, the 0.9 policy's target availability drops to 43.4% over all near-goal steps because it often overshoots/turns and the Goal leaves the camera view; at the **first entry** into 20 m, availability is still 100%. The 0.6 policy stops soon after entry and retains 72.7%.
- Beyond 60 m, target Depth availability falls (38.0%/24.9%), consistent with the 250 m depth range and Goal visibility.

## 4. Error-matching examples at 20 m

Representative episodes (target Depth is correct in all cases; visual-match Depth is not):

| Episode | Seed | Band | Visual valid fraction | Visual Goal match | IoU | Top-1 contains Goal center | Visual Depth not Goal |
|---|---:|---|---:|---:|---:|---:|---:|
| 1 | 900001 | short | 0.450 | 0.215 | 0.174 | false | true |
| 6 | 900008 | short | 0.084 | 0.084 | 0.037 | false | true |
| 16 | 900024 | medium | 0.202 | 0.179 | 0.054 | false | true |
| 26 | 900038 | short | 0.156 | 0.131 | 0.063 | false | true |
| 29 | 900043 | medium | 0.375 | 0.209 | 0.143 | false | true |
| 42 | 900057 | long | 0.250 | 0.250 | 0.198 | false | true |
| 50 | 900066 | medium | 1.000 | 1.000 | 0.467 | false | false |

Best cases (visual match largely follows the Goal): ep2 (0.773), ep3 (0.858), ep9 (0.972), ep31 (0.833), ep49 (0.806), ep50 (1.000).

The mismatch is best described as **visual match region partially overlapping the Goal but including surrounding/other objects**; the depth mean and proximity therefore mix Goal and non-Goal returns. This is the main risk for the RGB+Depth probe.

## 5. Interpretation

1. **True target Depth is available and correct near the goal.** The sensor is not the blocker; if the MEC knew where the target were, Depth would provide a clean near-range signal.
2. **The legal visual-matching region is the blocker.** At 20 m, Depth is available in 100% of episodes, but only ~47–50% of them have a visual region whose majority Depth matches the Goal. The top-1 patch never contains the Goal center at 20 m.
3. **Why this matters for the ablation:** the proposed `goal_conditioned_depth_evidence` is built from the visual-match region, not the true target region. A probe using these features may show limited gain even though the sensor signal is good. That is a valid and important negative-control result.
4. **Near-goal availability is not constant.** The 0.9 policy overshoots/turns and loses the Goal from view within 0–20 m; at first entry it is 100%. A distance-aware stop gate at first entry is the natural target.
5. **Movement remains the dominant limitation from the earlier evaluation:** 41/60 episodes never reach 20 m even with forced movement. Arrival-only Depth cannot fix that.

## 6. Artifacts

- Raw audit: `runs/v83_depth_prevalidation/20261018T015027817453Z/`
- Audit script: `v83_depth_prevalidation/trajectory_depth_audit.py`
- Stdout log: `runs/v83_depth_prevalidation/trajectory_audit_20261018T015026Z.log`
- All raw results are Git-ignored.
