# V8.3 Stage-2c Off-policy Arrival Calibration Protocol (prepared, not started)

- Date: 2026-10-19
- Status: **preparation only**. No residual calibration has been trained or run.
- Scope: Transformer-M M0, Arrival/Stop only. No Small, M1-Dir, M2, Depth or MEC BEV.
- Data: Train 600 rollouts for fitting; Dev Val 180 for development evaluation.
  Final Test 300 is sealed and must not be read.
- Stop threshold: 0.5 unless a decision-rule control explicitly says otherwise.

## 1. Why this stage exists

Stage-2c B0/B1/B2 showed that removing Stage-2b's `pos_weight` restores the
original M0 closed-loop behaviour, but changing the far-negative retention from
3% to 10% with plain BCE does not improve closed-loop SR/OSR. The diagnostics
in `V83_STAGE2C_REPORT.md` identify two reasons:

1. The extra far negatives are mostly already easy: B2 adds 2,915 samples, but
   only 22 of them have B0 probability >= 0.5 (mean probability 0.031).
2. Closed-loop failure is a per-decision hazard problem: an episode makes many
   far-distance Arrival decisions, so the episode premature-Stop rate is driven
   by `1 - prod(1 - per-step FPR)`, not by a single per-sample FPR.

The next stage therefore targets the **high-probability far states actually
visited by the policy**, not more random/expert far negatives.

## 2. Rollout collection (Train 600 only)

Script: `v83_spatial_stage2/stage2c_collect_rollouts.py`.

- Policy / features: original Stage-2a M0 checkpoint `B0` (frozen).
- Episode reconstruction: `Train 600/dataset_stats.json` seeds, band distance
  ranges and network-layout seeds. This is deterministic and reproduces the
  dataset start distances exactly (verified).
- Do not use Final Test or Dev Val for fitting.
- Per-step record:
  - 104-D frozen Arrival features from `base.arrival_head.extract_features`;
  - original B0 arrival logit/probability;
  - executed action, Stop flag, episode outcome;
  - true distance to goal and offline Goal visibility **as labels/diagnostics
    only**.
- The model input remains only current RGB tokens, Goal tokens, explicit
  heatmap and network state. True distance, position, visibility and simulator
  state are never inputs.

### Label and mining definition

- `arrival_label = 1` iff `true_distance < 20 m` (environment Stop radius).
- Off-policy far negative: `true_distance > 40 m` and `arrival_label = 0`.
- Hard-negative candidate: far negative with B0 `p >= 0.3` (primary) and
  `p >= 0.5` (secondary). These are the states that can terminate an episode
  prematurely.
- If R2 changes the trajectory materially, repeat collection with the
  calibrated policy for one or two DAgger-style rounds; do not reuse Dev Val
  rollouts for fitting.
- Mining controls:
  - `N_hard`: all high-p far negatives.
  - `N_random`: an equal-size random sample of far negatives, to isolate the
    effect of mining from simply adding more negatives.
  - `A0`: the original Stage-2b/2c 3% configuration, included as the anchor set.

### Smoke validation

The collector was smoke-tested on 2 episodes per band (6 total): 677 steps,
618 far>40 negatives, 1 with B0 p ≥ 0.5, 6 stops (1 at >40 m). The full
Train-600 run then completed: 600 episodes, 74,001 steps, 70,368 far>40
negatives, 181 with B0 p ≥ 0.5, 520 stops (181 at >40 m, 160 at >100 m),
SR 45.5%, premature-Stop 41.2%, Timeout 13.3%. Train rollouts are easier than
Dev Val (the base model was trained on these episodes), but they still contain
181 high-p far negatives, comparable to the 98 high-p far negatives in the
entire Dev Val 180 set. Use this pool for the first calibration and repeat
collection with the calibrated policy if the trajectory changes materially.

### Split

- Split Train 600 by **episode** into 480 calibration-train / 120
  calibration-validation (fixed seed; no episode overlap).
- Early stopping and variant selection use calibration-validation.
- Dev Val 180 is used only for development evaluation and paired comparison
  against B0. Final Test is untouched.

## 3. Calibration variants (Arrival only)

The original Arrival Head `net` is preserved. Backbone, context encoder and
movement weights are frozen; movement hashes must stay byte-identical.

| ID | Model | Parameters trained | Purpose |
|---|---|---:|---|
| R0 | original B0 head | 0 | reference |
| R1 | Platt scaling on base logit: `logit' = a*logit + b` | 2 | calibration-only control |
| R2 | `logit' = logit_base + r(features)`, `r` = 104→32→1 MLP, last layer zero-init | ~3.4k | lightweight residual calibration |
| R3 | R2 with random far negatives only (no `N_hard`) | ~3.4k | isolates hard-negative mining |
| R4 | B0 head + fixed threshold 0.6/0.7 or k-of-n consecutive Stop confirmation | 0 | decision-layer control, only if per-step FPR remains high |

The offline threshold curve already shows the trade-off: B0 far-40 FPR falls
from 0.67% at 0.5 to 0.32% at 0.7, while near-20 recall falls from 81.1% to
70.0%. A closed-loop threshold/persistence sweep is cheap and should be run
before concluding that a new head is required.

Constraints:

- R1 initializes at `a=1, b=0`; R2/R3 initialize at exactly B0 output
  (`r=0`).
- R1/R2/R3 loss = plain BCE on mixed rows + `lambda_residual * ||r||^2`
  (R2/R3) + `lambda_anchor * D(base || calibrated)` on anchor A0 positives and
  near negatives. The anchor term is the "original model prediction constraint";
  it prevents the residual from drifting on states the base head already
  handles.
- Do not use `pos_weight`. The Stage-2b regression demonstrated that changing
  the positive weight shifts probability calibration and destroys closed-loop
  SR.
- Same seed, optimizer, learning-rate search and early-stopping patience across
  R1/R2/R3. Report the exact budget in the next run's provenance.

## 4. Evaluation and acceptance

Offline (Dev Val 180, threshold 0.5):

- Natural / Hard Arrival-F1, precision, recall, Brier, ECE.
- Far-40 and far-100 false-Stop rate; near-20 missed-Stop rate.
- Distance-stratified per-sample FPR, especially `>40 m` and `>100 m`.
- High-p far-negative count and probability distribution.

Closed-loop (Dev Val 180, threshold 0.5):

- Overall and Short / Medium / Long SR, OSR, SPL, nDTW, premature-Stop,
  Timeout, mean steps, mean minimum distance.
- Paired per-episode comparison against B0 (bootstrap 95% CI and exact sign
  test).
- Per-decision hazard: far-step exposure, per-step Stop rate, predicted
  cumulative false-Stop probability vs observed premature-Stop.
- Stop-event diagnostics: distance, arrival probability, offline Goal
  visibility.

Acceptance gate (development, single seed is not evidence):

- SR/OSR improvement of about 3 pp over B0 and no SPL regression; and
- a material reduction in per-step far FPR that predicts a lower cumulative
  false-Stop probability, not only a better offline F1; and
- no near-20 recall collapse and no increase in Timeout that cancels the SR
  gain.

If R2/R3 do not reduce per-step far FPR, stop and analyze the residual
feature/label separability before adding more negatives or training a larger
model. Given the Stage-2c decomposition (132 far decisions/episode at 0.365%),
a 20–25% relative reduction is not expected to move SR; the target is roughly a
4× reduction (to ~0.08%) or a decision rule that removes most far false
decisions.

## 5. Leakage and discipline checks

- Unit test that calibration features are exactly the 104-D Arrival features
  and that true distance / visibility are not passed to `predict_action`.
- Unit test that movement/backbone hashes are unchanged.
- Never read `final_test_300`; never modify Stage-2a/B0/B1/B2 checkpoints.
- Do not start Small, M1-Dir, M2 or any full MEC training in this stage.

## 6. Commands prepared for the next stage

```bash
# Full Train 600 B0 rollout collection (prepared; run next stage)
python v83_spatial_stage2/stage2c_collect_rollouts.py \
  --dataset-root runs/v83_spatial_stage2/datasets/train_600 \
  --output-dir runs/v83_spatial_stage2/stage2c/offpolicy_train_rollouts_<stamp>

# Per-decision hazard decomposition from the Stage-2c diagnostics
python v83_spatial_stage2/stage2c_risk_analysis.py \
  --diagnostics-dir runs/v83_spatial_stage2/stage2c/diag_<stamp> \
  --output-dir runs/v83_spatial_stage2/stage2c/risk_<stamp>
```

The residual-calibration trainer is intentionally **not** implemented in this
stage; it is the first task of the next stage, using the collected rollout
features and the protocol above.
