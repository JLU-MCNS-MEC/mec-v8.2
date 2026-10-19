# V8.3 Stage-2f — Goal Reacquisition / Movement Recovery Report

- Date: 2026-10-19 (resume verification + consolidated report)
- Branch: `experiment/v83-spatial-foundation`
- Scope: **eval-only** diagnosis of Goal Reacquisition / Movement Recovery on
  the saved M0@0.7 and M1@0.7 trajectories. No training, no checkpoint
  modification, no M1-Dir/M2/Depth/MEC-BEV training.
- Data: Dev Val 180 (Short/Medium/Long 60/60/60), 55,414 replayed states
  (26,628 M0 + 28,786 M1; 26,478 M0 move states). Final Test 300 is sealed and
  was not read.
- M0 checkpoint:
  `runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z/Transformer-M/M0/best.pt`,
  sha256 `1c496fa8bd880592089ad85408ad7244c4c3e95817c6c4fdd23a7f5ecd5bd9e4`.
- M1 checkpoint:
  `runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z/Transformer-M/M1/best.pt`,
  sha256 `6cac7883419ebeae172b6f0e9dc1ddf6b87e4c0dc7f8994bb3e185e044287b00`.
- Detailed predecessor report (unchanged):
  `V83_STAGE2F_RECOVERY_DIAGNOSIS.md`.
- This report adds (a) the resume/verification record, (b) the required
  consolidated metric tables, and (c) supplementary offline analyses computed
  from the existing feature table: trigger distance, transient vs sustained
  invisibility, turn-induced loss, and the observable proximity proxy.

## 结论摘要 (Chinese TL;DR)

1. 上次运行在 10:45–11:01 已完成全部阶段并已提交/推送（`1daa433`）。唯一失败点是
   第一次 smoke 在**所有 rollout 结束后写 JSON** 时因 `PosixPath` 不可序列化而崩溃；
   修复后 budget-12 与 unlimited 两臂均完成。本次未重跑任何昂贵阶段。
2. Goal 不可见是 Movement 失败的主因：M0 move states 的 move error 为
   **visible 44.9% vs invisible 74.9%**；连续错误转向（wrong-run>=2）为
   **15.7% vs 56.1%**；平均 |heading error| 为 **12.3° vs 117.4°**。
3. 可观测 Current–Goal 特征对 visibility 只有中等区分度：最好单特征
   `sim_max` AUC **0.685**（Short 0.764 / Medium 0.672 / Long
   `baseline_entropy` 0.698）；even/odd 留出 trigger 的 invisibility precision
   仅 **50.9%**、recall **39.5%**。
4. 36-episode closed-loop smoke 中，12-step recovery 把 SR 从 **36.1% 降到
   19.4%**（1W/7L），Timeout 从 **13.9% 升到 25.0%**；unlimited 变体 SR
   **16.7%**（0W/7L），Timeout **52.8%**。**0/1073** 个 trigger 在 40 步内进入
   20 m。
5. 新补充分析给出机制解释：1073 个 closed-loop trigger **全部 >50 m**、99.9%
   >100 m、中位距离 **421.6 m**；它们检测到的主要是"目标太远/太小"，不是
   "在目标附近暂时丢失"。真正的近距不可见（<100 m，741 个状态，占 invisible
   的 8.0%）move error 高达 **85.8%**，oracle 全部是转向，而 M0 有 **60.5%**
   在 forward；这里才是 recovery 的潜在机会，但当前没有可靠的可观测近距门控
   （`last_high_top1` 对 <100 m 的 AUC 仅 **0.588**）。
6. **结论：不值得继续研究当前这套手写 scan-recovery 规则**。值得继续的是
   "可观测 target-presence / localization 表征 + 近距门控"；在此之前不应再跑
   完整训练或把 recovery 接入主策略。
7. **下一步唯一最值得开展的实验**：冻结主干、按 episode 划分的 calibrated
   target-presence probe（输入为可观测特征 + 合法的尺度/近距代理，标签为
   "当前可定位"，重点评估近距不可见状态），先离线验证 held-out precision/recall；
   只有近距 precision 达标才允许做 36-episode 的 near-gated、保留前进的 smoke。
   本轮不启动该实验。

## 1. Resume status: where the previous run stopped

The previous session's artifacts were inspected rather than trusted by
existence alone. Stage-by-stage evidence:

| Stage | Artifact / log | Evidence of completion | Status |
|---|---|---|---|
| Feature collection | `runs/.../stage2f/collect_20261019T024000Z/` | `features.csv` 55,415 lines (55,414 states + header), `provenance.json` states 55,414, `replay_check.json` 360/360 trajectory passes, zero distance/action/arrival mismatch, log ends with `Results:` | complete |
| Offline analysis | `runs/.../stage2f/analysis_20261019T025000Z/` | `summary.json` has all sections (55,414 states), `tables.md` complete, log ends with `Results:` | complete |
| Trigger sweep | `runs/.../stage2f/sweep_20261019T025500Z/` | `sweep.csv` and `summary.json` contain even/odd entries and the selected rule; no separate log | complete |
| First smoke | `runs/.../stage2f/smoke_20261019T030000Z/` | all 36 episodes × 3 policies rolled out, but no `metrics.json`; log ends in `TypeError: Object of type PosixPath is not JSON serializable` | **incomplete (final JSON write only)** |
| Code fix | `v83_spatial_stage2/stage2f_recovery_eval.py` mtime 10:52 | re-run succeeds | complete |
| Budget-12 smoke | `runs/.../stage2f/smoke_budget12_20261019T031000Z/` | `metrics.json`, per-episode CSVs, `provenance.json`, `baseline_check` 0 mismatches, log ends with `Results:` | complete |
| Unlimited smoke | `runs/.../stage2f/smoke_nobudget_20261019T031500Z/` | same artifacts, log ends with `Results:` | complete |
| Report | `V83_STAGE2F_RECOVERY_DIAGNOSIS.md` | 426 lines, committed in `1daa433` | complete |
| Commit/push | `1daa433` | local HEAD == `origin/experiment/v83-spatial-foundation` | complete |

No Stage-2f, Python, or analysis process was still running at resume time. The
failed first smoke directory was preserved (not deleted or overwritten).

### What this resume did

- Did **not** re-run collection, analysis, sweep, or the closed-loop smoke.
- Independently recomputed the key closed-loop metrics from
  `*_per_episode.csv` and `steps/*.csv`: SR/OSR/SPL/nDTW, premature stop,
  timeout, paired W/L, trigger counts, re-visible rates, enter-20 m rate, and
  recovery error. All matched the saved `metrics.json` and the report.
- Independently recomputed the offline visible/invisible tables and rank AUC
  from `features.csv`; all matched.
- Ran the CPU-only tests: 6 existing Stage-2f tests + 4 new supplementary
  helper tests, all pass.
- Added the supplementary analyses in `stage2f_supplement.py`. Their output is
  in `runs/.../stage2f/supplement_20261019T033000Z/` (`supplement.json`,
  `supplement.md`). These use true labels only as offline evaluation targets.

## 2. Data, configuration, and no-leakage contract

**Data.** Dev Val 180, 55,414 states. Stage-2e's exact-state replay was reused;
no new rollouts were generated for the offline diagnosis. Final Test 300 was
not read. The 36-episode closed-loop smoke uses 12 **evenly spaced** episodes
per band (`--selection even` means `np.linspace` over the band's 60 episodes,
not parity). The threshold sweep itself selects on even-*parity* episodes and
reports odd-parity episodes as held out; the smoke subset therefore contains
both parities and is a smoke test, not an independent held-out closed-loop
evaluation.

**Frozen controllers.** M0@0.7 and M1@0.7 are the Stage-2a checkpoints, used
unchanged. The recovery smoke overlays a rule on frozen M0 only.

**No-leakage contract.** All controller-side features come from run-time
inputs: current RGB tokens, three Goal-reference tokens, the 3×8×8
Goal-Current heatmap, the parameter-free baseline/coherent spatial maps, the
previous coherent map and previous executed action (only for yaw-aligned
LEFT/RIGHT warps), and the frozen M0 movement/arrival outputs. FORWARD
translation is **never** warped with a fixed pixel shift. True goal position,
true visibility, true distance, true heading error, and the oracle action are
stored in separate offline-label columns. `stage2f_policy.py` raises on every
`FORBIDDEN_KEYS` access; `test_stage2f.py` checks the observable/forbidden sets
are disjoint and that the policy runs on observable-only dictionaries.

**Smoke configuration.** 36 episodes (12 per band, evenly spaced), baseline
M0@0.7 + `side_history` + `scan_only`; trigger `heatmap_max_top1 < 0.02 OR
heatmap_max_entropy > 0.98`, K=1; high-confidence side storage
`top1 >= 0.20 AND entropy <= 0.85`; temporal trigger disabled
(`consistency=-1`) because the offline AUC showed it was uninformative;
`max_recovery_steps` = 12 and 0 (unlimited).

## 3. Offline diagnosis: visible vs invisible Movement

M0 move states (`m0_arrival_probability < 0.7`). True visibility is an offline
label only.

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
0.204 vs 0.065), so this is an environment/observability property, not an M0
checkpoint artifact.

Where invisible states occur (M0, all states):

| Band | States | Invisible | Invisible rate | At Stop | At move | Movable share |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 26,628 | 9,273 | 34.8% | 50 | 9,223 | **99.5%** |
| Short | 8,462 | 3,066 | 36.2% | 18 | 3,048 | 99.4% |
| Medium | 9,236 | 3,928 | 42.5% | 19 | 3,909 | 99.5% |
| Long | 8,930 | 2,279 | 25.5% | 13 | 2,266 | 99.4% |

Consecutive wrong-turn diagnostics (M0 move states):

| Band | Move states | Wrong-run>=2 | Wrong-run>=3 | Run starts | Start low-conf | Start triggered | Pre-step triggered | \|Heading\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 26,478 | 29.8% | 26.6% | 6,726 | 4,651 (69.2%) | 4,281 (63.7%) | 4,289 (63.8%) | 48.9° |
| Short | 8,411 | 31.8% | 29.0% | 2,025 | 1,406 | 1,344 | 1,346 | 58.5° |
| Medium | 9,185 | 31.0% | 27.4% | 2,314 | 1,467 | 1,320 | 1,322 | 56.5° |
| Long | 8,882 | 26.6% | 23.3% | 2,388 | 1,779 | 1,617 | 1,621 | 32.0° |

The trigger precedes most wrong-turn runs, but it also fires on most move
states, so this is not selectivity.

**Premature Stop is a related but separate lever.** 88/150 M0 stops are
premature (58.7%). Premature stops are low-confidence more often than correct
stops (heat top1 median 0.066 vs 0.248, entropy 0.952 vs 0.792) and only 43.2%
are goal-visible vs 100% of correct stops. A naive low-confidence stop veto
would also veto 40.3% of correct stops, so confidence alone is not a safe veto.

## 4. Observable features and trigger quality

Feature AUC for goal visibility on M0 move states (higher = better; entropy
and innovation are sign-corrected). Episode-bootstrap 95% CI, 200 resamples.

| Band | Feature | AUC(visible) | 95% CI | N |
|---|---|---:|---|---:|
| Overall | `sim_max` | **0.685** | [0.608, 0.755] | 26,478 |
| Overall | `baseline_top1` | 0.641 | [0.547, 0.738] | 26,478 |
| Overall | `heatmap_max_top1` | 0.638 | [0.551, 0.725] | 26,478 |
| Overall | `baseline_entropy` | 0.637 | [0.535, 0.745] | 26,478 |
| Overall | `coherent_top1` | 0.625 | [0.542, 0.715] | 26,478 |
| Overall | `temporal_consistency` | 0.529 | [0.430, 0.641] | 14,778 |
| Short | `sim_max` | **0.764** | [0.650, 0.885] | 8,411 |
| Medium | `sim_max` | **0.672** | [0.555, 0.793] | 9,185 |
| Long | `baseline_entropy` | **0.698** | [0.474, 0.864] | 8,882 |
| Long | `baseline_top1` | 0.697 | [0.437, 0.906] | 8,882 |
| Long | `temporal_consistency` | 0.446 | [0.299, 0.635] | 4,889 |

No observable feature reaches the ~0.8 AUC needed for a high-precision
trigger. Arrival probability is not a visibility signal (AUC 0.442 overall;
higher arrival p is associated with invisible because the base head can fire
prematurely). Temporal consistency is near chance overall and below chance in
Long.

Default trigger (`top1 < 0.15 OR entropy > 0.85`, K=2) precision/recall for
invisibility:

| Band | Trigger | Trigger rate | Invisible base rate | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Overall | current-step low-conf | 76.2% | 34.8% | **36.0%** | 78.7% | 49.4% |
| Overall | K=2 history | 72.2% | 34.8% | **36.4%** | 75.4% | 49.0% |
| Overall | temporal-only | 5.9% | 34.8% | 37.9% | 6.4% | 10.9% |
| Short | current-step low-conf | 72.8% | 36.2% | 37.9% | 76.1% | 50.6% |
| Medium | current-step low-conf | 73.9% | 42.6% | 42.4% | 73.6% | 53.8% |
| Long | current-step low-conf | 81.8% | 25.5% | 28.3% | 90.8% | 43.2% |

The default trigger fires on 76–82% of move states and its precision is
approximately the base rate; it is not a usable "lost" detector.

Even/odd episode threshold sweep (thresholds selected on even-parity episodes,
reported on held-out odd-parity episodes):

| Selected rule | Split | Trigger rate | Invisible base | Invisible precision | Invisible recall | Unreliable precision | Unreliable recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `top1<0.02 OR ent>0.98`, K=1 | even | 23.1% | 33.8% | 53.9% | 36.8% | 69.9% | 30.0% |
| `top1<0.02 OR ent>0.98`, K=1 | **odd (held out)** | **24.7%** | 31.8% | **50.9%** | **39.5%** | **67.3%** | **31.6%** |

The selected operating point is the one used by the closed-loop smoke. It is
moderately better than chance offline but not high-precision: about half of
its triggers are on goal-visible states and it catches only ~40% of invisible
states.

**Distance-stratified AUC (new).** The overall AUC is partly a distance/size
effect: visible rate falls from 87.5% at <50 m to 30.6% at >400 m. Within
distance bands, the best feature changes:

| Distance | States | Invisible | Visible rate | `sim_max` AUC | `heatmap_max_top1` AUC | `baseline_top1` AUC |
|---|---:|---:|---:|---:|---:|---:|
| 0–50 m | 1,324 | 165 | 87.5% | 0.609 | **0.801** | 0.605 |
| 50–100 m | 3,560 | 576 | 83.8% | 0.577 | 0.656 | 0.480 |
| 100–200 m | 7,965 | 2,242 | 71.9% | 0.594 | 0.613 | 0.572 |
| 200–400 m | 10,357 | 3,968 | 61.7% | **0.698** | 0.563 | 0.651 |
| 400+ m | 3,272 | 2,272 | 30.6% | 0.691 | 0.678 | **0.776** |

So visibility confidence is best at close range (`heatmap_max_top1` 0.801 at
<50 m), but there are only 165 invisible states there and the current trigger
threshold almost never fires on them (only 33.7% of near-target invisible
states meet the selected extreme low-confidence rule). The 100–400 m band,
where most invisible states are, has AUC 0.56–0.70.

## 5. History, previous belief, and previous action

- **Last known target side works only when confidence is high.** With
  `heatmap_max_top1 >= 0.20` and entropy <= 0.85, the baseline-map centroid
  side has AUC 0.953 and balanced accuracy 0.873 for the true target side;
  at `>= 0.30`/`<= 0.75`, AUC 0.970 and balanced accuracy 0.906. High
  confidence occurs on 22.6% (0.20/0.85) or 13.5% (0.30/0.75) of M0 move
  states. The low-confidence trigger that would use it is not selective.
- **Yaw-aligned previous-frame evidence is insufficient.** It is valid only
  after LEFT/RIGHT. `temporal_consistency` AUC for visibility is 0.529 overall
  and 0.446 in Long; `innovation_mean_abs` is 0.488. FORWARD translation was
  never warped.
- **Turn-induced temporary loss is real but self-resolving (new).** States
  where the previous state was visible, the previous action was a turn, and
  the current state is invisible: 557 states (2.1% of move states; 6.0% of
  invisible states), median true distance 332.5 m, only 1.1% within 100 m.
  The target is re-visible within 1 step in **87.1%**, within 2 steps 91.0%,
  within 5 steps 94.3%. The executed action equals the oracle in 87.4% of
  these states. This is a transient field-of-view effect, not a failure mode
  that needs a recovery overlay.
- **Transient vs sustained invisibility (new).** Only 6.5% of invisible move
  states are re-visible within 2 steps (602/9,223); these transient states have
  a 9.0% move error. The other 93.5% are sustained, with **79.5%** move error
  (k=1: 94.1% sustained, 79.7% error). Recovery-by-scanning targets the wrong
  regime: the sustained states are mostly far from the goal.
- **Observable proximity gate is weak (new).** `last_high_top1` has AUC 0.588
  for true distance <100 m; `heatmap_max_top1` has 0.576. A recency-gated
  trigger (`low` (top1<0.10 or ent>0.90) AND last high-confidence within N
  steps AND last high top1 >= 0.20) fires on 2.5–6.5% of move states but its
  precision for *near-target invisible* states is only 4.9–5.7% and recall
  4.3–12.1%. It cannot identify the near-target regime where recovery could
  help.

**Answer for item 3:** previous-frame target position, the previous action and
yaw-aligned evidence are enough to *store a last known side* and to recognize a
transient turn-induced loss, but not enough to build a reliable lightweight
search/relocalization policy. The missing piece is a trustworthy
near-target/presence signal, not a better turn rule.

## 6. Recovery policy and closed-loop smoke

The candidate is an eval-time overlay on frozen M0@0.7. It does not change the
M0 Stop threshold: when the base arrival probability is >= 0.7 the episode
still ends with STOP. The overlay only chooses among L/F/R when the base policy
would move.

Observable rule:

1. **Confidence.** `low = heatmap_max_top1 < tau_top1 OR
   heatmap_max_entropy > tau_entropy`; `high = heatmap_max_top1 >= h_top1 AND
   heatmap_max_entropy <= h_entropy`.
2. **Last known target.** While `high`, store the baseline-map centroid side
   (`< -0.05` left, `> 0.05` right) and the centroid value.
3. **Trigger.** Enter recovery after `K` consecutive `low` move states, or
   when a yaw-aligned in-place-turn warp is inconsistent with the current
   coherent map (`temporal_valid` and consistency < tau).
4. **Recovery action.** If a last high-confidence side exists, turn toward it;
   otherwise run a deterministic scan (12 LEFT, 24 RIGHT, 24 LEFT, then
   RIGHT). **No FORWARD while recovering.**
5. **Re-acquire.** One `high` step exits recovery and returns to the base
   action.

Variants: `side_history` (rule above) and `scan_only` (same trigger, always
scans). Budgets: 12 steps and unlimited.

### 6.1 Core closed-loop metrics

36 episodes (12 per band, evenly spaced), frozen M0@0.7 baseline. The baseline
reproduces the saved per-episode outcomes exactly (0 mismatches). The full
180-episode M0@0.7 context is SR 62/180 = 34.4% (Stage-2e); the 36-episode
smoke baseline is 36.1%.

| Policy | Budget | SR | OSR | SPL | nDTW | Premature | Timeout | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline M0@0.7 | — | **36.1%** | 38.9% | 0.361 | 0.409 | 50.0% | 13.9% | 129.0 |
| side_history | 12 | 19.4% | 19.4% | 0.184 | 0.168 | 55.6% | 25.0% | 176.4 |
| scan_only | 12 | 19.4% | 19.4% | 0.184 | 0.168 | 58.3% | 22.2% | 165.4 |
| side_history | 0 (unlimited) | 16.7% | 16.7% | 0.171 | 0.171 | 30.6% | 52.8% | 243.2 |
| scan_only | 0 (unlimited) | 16.7% | 16.7% | 0.171 | 0.171 | 33.3% | 50.0% | 242.9 |

By band (budget 12; baseline → side_history / scan_only):

| Band | Baseline SR | side_history SR | scan_only SR | Baseline SPL | side_history SPL | Baseline Timeout | side_history Timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 50.0% | 33.3% | 33.3% | 0.500 | 0.333 | 8.3% | 33.3% |
| Medium | 33.3% | 16.7% | 16.7% | 0.333 | 0.136 | 16.7% | 16.7% |
| Long | 25.0% | 8.3% | 8.3% | 0.250 | 0.083 | 16.7% | 25.0% |

Paired per-episode vs baseline:

| Policy | Budget | Band | SR Δ (W/L, 95% CI) | Timeout Δ | Premature Δ |
|---|---:|---|---|---:|---:|
| side_history | 12 | Overall | **−16.67 pp (1W/7L) [−30.56, −2.78]** | +11.11 pp | +5.56 pp |
| side_history | 12 | Short | −16.67 pp (0W/2L) [−41.67, 0.00] | +25.00 pp | −8.33 pp |
| side_history | 12 | Medium | −16.67 pp (1W/3L) [−50.00, +16.67] | 0.00 pp | +16.67 pp |
| side_history | 12 | Long | −16.67 pp (0W/2L) [−41.67, 0.00] | +8.33 pp | +8.33 pp |
| side_history | 0 | Overall | **−19.44 pp (0W/7L) [−33.33, −8.33]** | +38.89 pp | −19.44 pp |
| scan_only | 12 | Overall | −16.67 pp (1W/7L) [−30.56, −2.78] | +8.33 pp | +8.33 pp |

The overall paired loss is statistically clear; the band-level CIs are wide
because there are only 12 episodes per band.

### 6.2 Recovery trigger rate and re-acquisition

| Policy | Budget | Triggers | Episodes with trigger | Episode trigger rate | Decision steps | Trigger-step rate |
|---|---:|---:|---:|---:|---:|---:|
| side_history | 12 | 1,073 | 23/36 | **63.9%** | 6,349 | **16.9%** |
| scan_only | 12 | 1,035 | 23/36 | **63.9%** | 5,954 | 17.4% |
| side_history | 0 | 28 | 23/36 | 63.9% | 8,755 | 0.3%* |
| scan_only | 0 | 29 | 23/36 | 63.9% | 8,746 | 0.3%* |

\* The unlimited policy counts only the initial trigger; it then stays in
recovery for hundreds of steps, so the trigger-step rate is not comparable.

Trigger / re-acquisition diagnostics (side_history):

| Budget | Triggers at true-invisible states | Re-visible ≤5 steps | Re-visible ≤10 steps | Enter <20 m ≤40 steps | Recovery actions | Wrong vs oracle | Error rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 881/1,073 (82.1%) | **6.1%** (54/881) | **11.0%** (97/881) | **0.0%** (0/1,073) | 1,311 | 776 | **59.2%** |
| 0 | 5/28 (17.9%) | 40.0% (2/5) | 40.0% (2/5) | **0.0%** (0/28) | 7,217 | 3,865 | 53.6% |

The unlimited-budget 40% is based on only 5 true-invisible triggers and is not
reliable; the unlimited policy mostly stays in recovery after the first
trigger.

By band with the 12-step budget (side_history):

| Band | Triggers | Episodes | Trigger invisible rate | Re-visible ≤10 | Enter <20 m ≤40 | Recovery error rate |
|---|---:|---:|---:|---:|---:|---:|
| Short | 301 | 6/12 | 81.1% | 18.4% | 0.0% | 50.4% |
| Medium | 236 | 9/12 | 80.5% | 18.9% | 0.0% | 60.7% |
| Long | 536 | 8/12 | 83.4% | **3.6%** | 0.0% | 63.6% |

### 6.3 Where the triggers fire (new)

The closed-loop trigger distance explains the failure. All 1,073 budget-12
triggers are >50 m from the true goal; 99.9% are >100 m; 85.5% are >200 m;
57.1% are >400 m; the median is **421.6 m**.

| Band | Median | p10 | p90 | >100 m | >200 m |
|---|---:|---:|---:|---:|---:|
| Short | 198.1 m | 154.0 m | 263.2 m | 99.7% | 48.2% |
| Medium | 353.1 m | 276.0 m | 501.5 m | 100.0% | 100.0% |
| Long | 481.1 m | 401.7 m | 560.7 m | 100.0% | 100.0% |

The trigger is not detecting "temporarily lost near the target". It is detecting
"the target is far away / not visible at long range". At those distances, an
in-place scan cannot re-acquire the target, and it abandons forward progress:
the 12-step policy spends 1,311 recovery actions, 59.2% of which are wrong
against the heading oracle, and none of the 1,073 triggers reaches 20 m within
40 steps.

### 6.4 The near-target regime (new)

Invisible M0 move states by true distance:

| Distance | States | Share of invisible | Move err | Executed = oracle | Oracle L/R | Executed F/L/R | Median m | \|Heading\| |
|---|---:|---:|---:|---:|---|---|---:|---:|
| 0–50 m | 165 | 1.8% | **89.1%** | 10.9% | L 130 / R 35 | F 130 / L 17 / R 18 | 32.3 | 126.1° |
| 50–100 m | 576 | 6.2% | **84.9%** | 15.1% | L 354 / R 222 | F 318 / L 132 / R 126 | 73.2 | 116.4° |
| 100–200 m | 2,242 | 24.3% | 79.2% | 20.8% | L 1,102 / R 1,140 | F 955 / L 638 / R 649 | 146.9 | 97.9° |
| 200–400 m | 3,968 | 43.0% | 66.3% | 33.7% | L 1,431 / R 2,537 | F 1,793 / L 1,020 / R 1,155 | 303.9 | 111.3° |
| 400+ m | 2,272 | 24.6% | 82.4% | 17.6% | L 1,429 / R 843 | F 1,314 / L 483 / R 475 | 574.0 | 151.5° |

The near-target (<100 m) subset is 741 states (8.0% of invisible states). Its
move error is **85.8%**; the oracle action is **always a turn** (L 484 /
R 257), but M0 executes **FORWARD 448/741 = 60.5%** and matches the oracle only
14.2% of the time. This is the genuine recovery opportunity: when the target is
near but out of view, the base policy often advances instead of turning toward
it. The current trigger misses this regime because the strict closed-loop
low-confidence threshold (`top1<0.02 OR entropy>0.98`) fires there only 33.7%
of the time (84.1% meet the looser `top1<0.10 OR entropy>0.90` rule used by the
recency diagnostic), and it fires far more often at long range.

## 7. Failure analysis

1. **The trigger is a poor "lost" detector.** Observable confidence separates
   visibility at only ~0.64–0.69 AUC overall; the best held-out rule has ~51%
   invisibility precision and ~40% recall. The default rule is at the base
   rate.
2. **The trigger fires in the wrong regime.** In closed loop, all triggers are
   >50 m away (median 421.6 m). It detects far/small targets, not temporary
   loss near the goal.
3. **The recovery action is wrong for the regime.** In-place scanning at
   200–500 m cannot re-acquire the target and removes forward progress. Re-
   visible within 10 steps is 11.0% overall and 3.6% in Long; entry into 20 m
   within 40 steps is 0%.
4. **History/temporal evidence cannot rescue it.** The yaw-aligned temporal
   signal is near chance for visibility (0.529 overall, 0.446 Long). Turn-
   induced transient loss is rare and self-resolves within 1 step (87.1%).
   The observable proximity proxy is weak (AUC 0.588 for <100 m), and a
   recency-gated trigger has only ~5% precision for near-target invisible
   states.
5. **The closed-loop cost is consistent.** Both budgeted and unlimited variants
   reduce SR by ~17–19 pp, increase Timeout, and worsen SPL/nDTW; the side
   history and scan-only variants are nearly identical (side history does not
   help).
6. **Premature Stop is separate.** 58.7% of stops are premature, but a naive
   low-confidence stop veto would also veto 40.3% of correct stops; a better
   target-presence signal is needed before a stop veto is safe.

## 8. Decision: is Recovery worth continuing?

**The current hand-written scan-recovery policy is not worth continuing.** It
is harmful in closed loop and its trigger is not measuring the right thing.

**The underlying problem is worth continuing**, but not with another
hand-written turn rule. The evidence points to two distinct regimes:

- **Far target (92% of invisible states, >100 m):** this is a navigation/prior
  problem. The target is not "temporarily lost"; it is far away. Turning in
  place is the wrong action. The base policy needs a map/coarse-navigation
  prior or a learned far-range localization signal, not a recovery scan.
- **Near target (<100 m, 741 states, 8% of invisible):** this is the genuine
  re-acquisition problem. The oracle is always a turn, but M0 moves forward
  60.5% of the time. A recovery policy could help here **if** it can gate on
  near-target status without leakage. The current observable features cannot:
  `last_high_top1` AUC 0.588; recency-gated near-invisible precision ~5%.

No training was started, no checkpoint was modified, and no policy was
adopted. The negative result is conservative: the closed-loop smoke used an
operating point selected on even-parity episodes (with odd-parity offline
holdout), and still lost 16.7–19.4 pp SR.

## 9. Limitations

- The closed-loop smoke is 36 episodes (12 per band), not the full 180. Band-
  level paired CIs are wide; only the overall paired effect is clear.
- The smoke uses 12 evenly spaced episodes per band, which include both
  parities; the threshold sweep selected on even parity and reports odd parity
  offline. The smoke is not an independent held-out closed-loop evaluation.
  The negative result is still conservative because the trigger was tuned on a
  superset of the smoke's state distribution.
- All offline analyses use true visibility/distance/heading as evaluation
  labels only. No policy input uses them; `FORBIDDEN_KEYS` and the CPU tests
  enforce this.
- One checkpoint per model (M0/M1, seed from Stage-2a); no seed replication.
  The M1 trajectory corroborates the visibility pattern but is not an
  independent training seed.
- The supplementary distance-stratified AUC and near-target analysis are
  post-hoc diagnostics, not pre-registered hypotheses. They are consistent
  with the closed-loop failure but should be confirmed on more episodes before
  driving a new architecture.

## 10. Next single experiment (not run this round)

**A calibrated, episode-split target-presence / localization probe, with a
legal scale/proximity signal, evaluated first offline.**

Why this one: the binding missing signal is not another turn rule; it is a
trustworthy "is the target currently localizable, and am I near it?" signal.
The hand-crafted confidence features and the observable history/proximity
proxy are insufficient (AUC 0.588 for <100 m; ~5% near-target trigger
precision). A small probe on frozen representations can test whether the
missing signal is learnable before any closed-loop work.

Concrete design:

- **Data:** the existing 55,414 Stage-2f states; labels are offline only.
  Primary label: goal localizable now (true visibility). Secondary diagnostic
  label: invisible but within 100 m (the recoverable near-target regime).
- **Inputs:** the existing observable features plus a legal scale/proximity
  proxy (e.g., heatmap peak scale, expected projected size, odometry/step
  count since the last high-confidence observation). No true goal coordinates,
  true visibility, true distance, or heading error.
- **Split:** grouped by episode (e.g., 5-fold episode-level CV); no state-level
  leakage. Report calibration and precision/recall at fixed operating points.
- **Gate:** require held-out near-target invisible precision >= 0.80 and recall
  >= 0.50 before any closed-loop use. If the probe cannot reach this, stop the
  rule-based recovery line and move to a map/coarse-navigation or temporal-
  memory approach instead.
- **Only if the gate passes:** a 36-episode closed-loop smoke with recovery
  restricted to the near-target regime and forward progress preserved; no
  full training.

This is eval/probe-only, does not train a new MEC, does not modify the
checkpoint, and keeps Final Test 300 sealed.

## 11. Artifacts, commands, and tests

Feature collection (already complete, not re-run):

```
runs/v83_spatial_stage2/stage2f/collect_20261019T024000Z/
  features.csv (55,414 states), replay_check.json, provenance.json
```

Offline analysis / sweep / smoke:

```
runs/v83_spatial_stage2/stage2f/analysis_20261019T025000Z/
runs/v83_spatial_stage2/stage2f/sweep_20261019T025500Z/
runs/v83_spatial_stage2/stage2f/smoke_budget12_20261019T031000Z/
runs/v83_spatial_stage2/stage2f/smoke_nobudget_20261019T031500Z/
```

Supplementary analyses added by this resume:

```
runs/v83_spatial_stage2/stage2f/supplement_20261019T033000Z/
  supplement.json, supplement.md
```

Code:

```
v83_spatial_stage2/stage2f_features.py
v83_spatial_stage2/stage2f_collect.py
v83_spatial_stage2/stage2f_analyze.py
v83_spatial_stage2/stage2f_policy.py
v83_spatial_stage2/stage2f_recovery_eval.py
v83_spatial_stage2/stage2f_trigger_sweep.py
v83_spatial_stage2/stage2f_supplement.py   (new)
v83_spatial_stage2/test_stage2f.py
v83_spatial_stage2/test_stage2f_supplement.py  (new)
```

Tests:

```
python -m unittest -v v83_spatial_stage2.test_stage2f
python -m unittest -v v83_spatial_stage2.test_stage2f_supplement
```

Result: 6 + 4 tests pass (CPU only). No training, no checkpoint modification,
Final Test 300 sealed and unread.

## Appendix A. Supplementary metric definitions

- `move_state`: `owner_arrival_probability < 0.7`.
- `move_error`: `owner_move_action != oracle_move_action` on move states.
- `high`: `heatmap_max_top1 >= 0.20 AND heatmap_max_entropy <= 0.85`.
- `low`: `heatmap_max_top1 < 0.10 OR heatmap_max_entropy > 0.90` for the
  recency-trigger diagnostic; the closed-loop operating point uses the stricter
  `top1 < 0.02 OR entropy > 0.98`.
- `steps_since_high`: steps since the last high-confidence state in the same
  episode; a high state resets it to 0.
- transient within k: an invisible state with `goal_visible = 1` at any of the
  next k states in the same episode; sustained otherwise.
- turn-induced loss: `prev_visible = 1`, `prev_action in {LEFT, RIGHT}`,
  current `goal_visible = 0`.
- near-target invisible: `goal_visible = 0 AND distance_m < 100` (true distance
  is an offline label only).
- trigger distance: true distance at the trigger state, used to explain the
  closed-loop failure; it is never a policy input.
