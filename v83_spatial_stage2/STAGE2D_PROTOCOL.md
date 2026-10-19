# V8.3 Stage-2d Off-policy Arrival Calibration and Stop-Decision Rules

- Date: 2026-10-19
- Branch: `experiment/v83-spatial-foundation`
- Status: protocol fixed before running Stage-2d; results are reported in
  `V83_STAGE2D_REPORT.md`.
- Scope: Transformer-M M0 Arrival/Stop only. No Small, M1-Dir, M2, Depth or
  full-MEC training. Final Test 300 stays sealed.

## 1. Frozen assets

- Base checkpoint: Stage-2a M0
  `runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z/Transformer-M/M0/best.pt`
  (sha256 `1c496fa8bd880592089ad85408ad7244c4c3e95817c6c4fdd23a7f5ecd5bd9e4`).
- Backbone, context encoder, movement classifier and the original 104-D
  Arrival Head are frozen for every Stage-2d variant.
- R1/R2/R3 do **not** retrain the original Arrival Head. They add a separate
  calibration module on top of the frozen base logit. The original head weights
  are never in an optimizer.
- Stage-2d writes only to new run directories under
  `runs/v83_spatial_stage2/stage2d/`. Existing Stage-2a/2b/2c artifacts are
  read-only.

## 2. Data and split

- Reuse the existing Train-600 B0 off-policy rollout cache:
  `runs/v83_spatial_stage2/stage2c/offpolicy_train_rollouts_20261019T013500Z/rollout_steps.npz`
  (600 episodes, 74,001 states). No re-collection is performed.
- Every state carries the observable 104-D Arrival feature and the original
  B0 logit/probability. True distance and Goal visibility are labels /
  offline diagnostics only and are never model inputs.
- Episode split: fixed seed `20261019`, permute the 600 episode ids, first
  480 = calibration-train, last 120 = calibration-validation. No episode
  appears in both splits.
- Calibration-validation is used for LR search, early stopping and variant /
  epoch selection. Dev Val 180 is used only after the configuration is fixed,
  for development closed-loop evaluation and paired comparison against B0.
- Final Test 300 is never read.

## 3. Labels and far-negative mining

- `arrival_label = 1` iff `true_distance < 20 m` (environment Stop radius).
- Far negative: `true_distance > 40 m`, label 0.
- Base training rows (both R2 and R3, identical):
  every Train-480 rollout state with `true_distance <= 40 m` (near positives,
  boundary negatives, and 20–40 m negatives). This preserves the on-policy
  near-distance state distribution.
- Hard far negatives (R2 only): all Train-480 far negatives with B0
  `p >= 0.3` (the Stage-2c primary mining threshold).
- Random far-negative control (R3 only): an equal-size random sample
  (seed `20261019`) of Train-480 far negatives with B0 `p < 0.3`, disjoint
  from the hard set.
- R2 and R3 therefore have exactly the same number of rows, the same
  optimizer budget, the same architecture and the same initialization. Their
  only difference is the far-negative selection.
- R1 uses the R2 row set so that it is a calibration-only control on the same
  data, but has no feature input and only two parameters.

## 4. Variants

| ID | Model | Trainable params | Input | Purpose |
|---|---|---:|---|---|
| R0 | original B0 head | 0 | — | reference |
| R1 | Platt scaling `logit' = a·logit_b + b`, init `a=1,b=0` | 2 | base logit | calibration-only control |
| R2 | `logit' = logit_b + r(features)`, `r` = 104→32→GELU→1, last layer zero-init | 3,393 | 104-D features + base logit | hard-mined residual calibration |
| R3 | same as R2, random far negatives instead of hard | 3,393 | 104-D features + base logit | mining control |
| R4 | original B0 head + fixed decision rule | 0 | base probability | decision-layer control |

R2/R3 exactly reproduce the B0 output at initialization because the last
residual layer is zero-initialized. R1 reproduces B0 at `(a,b)=(1,0)`.

### Loss (plain BCE, no `pos_weight`)

```
L = BCE(calibrated_logit, arrival_label)
    + lambda_anchor * KL(Bernoulli(sigmoid(base_logit)) || Bernoulli(sigmoid(calibrated_logit)))
      on training rows with true_distance <= 40 m
    + lambda_residual * mean((calibrated_logit - base_logit)^2)      # R2/R3 only
```

The anchor term is the original-model prediction constraint that protects
near-distance Stop ability. The base probability is detached. Stage-2b
demonstrated that `pos_weight` destroys closed-loop SR, so no positive
weighting is used anywhere.

### Training budget

- R2/R3 share exactly: AdamW, batch 256, max 200 epochs, patience 20, seed
  `20264017`, weight decay `1e-4`, LR grid `{1e-3, 3e-3, 1e-2}`, anchor grid
  `{1, 10}`, residual grid `{1e-3, 1e-2}`.
- R1: AdamW, same batch/epochs/patience/seed, LR grid `{1e-2, 1e-1}`,
  anchor grid `{1, 10}`. Two parameters need a different LR grid; the budget
  (epochs, patience, batch size) is identical.
- Feature standardization statistics are computed once from all Train-480
  rollout states and shared by R2 and R3, so initialization is identical.
- The checkpoint for each variant is the epoch with the lowest
  calibration-validation selection score.

### Selection metric (no offline F1)

Per calibration-validation rollout episode, the primary risk is the number of
binary false Stop decisions at `true_distance > 40 m`:

```
far40_false_stops_per_episode = mean_episodes sum_{d>40m} 1[p_calibrated >= 0.5]
```

Selection score = `far40_false_stops_per_episode`
`+ 0.02 × mean_episodes sum_{d>40m} p_calibrated` (a small smooth tie-breaker)
`+ 20 × max(0, B0_near20_recall − 5 pp − near20_recall)`.
F1/Brier/ECE are reported but are not used to choose the final model.

## 5. R4 decision rules

All rules use the frozen B0 head and re-run the full closed loop. Decision
state is created fresh for every episode.

| Rule | Threshold | Confirmation |
|---|---:|---|
| `threshold_0.5` | 0.50 | 1 step (B0 reference, re-run) |
| `threshold_0.6` | 0.60 | 1 step |
| `threshold_0.7` | 0.70 | 1 step |
| `persist_2of2_t0.5` | 0.50 | 2 consecutive Stop decisions |
| `persist_3of3_t0.5` | 0.50 | 3 consecutive Stop decisions |
| `persist_2of2_t0.6` | 0.60 | 2 consecutive Stop decisions (supplementary) |

When confirmation is not satisfied the executed action is the frozen
movement head's argmax; when it is satisfied the action is STOP. `confirm_k=n`
means "all of the last n decisions are above threshold"; the n-step delay is
the price of confirmation.

## 6. Evaluation

### First (per-decision risk)

- Far-40 per-step false Stop rate (`d > 40 m`), far-100, and the 20–40 m
  boundary rate.
- Near-20 per-step Stop rate / offline recall.
- Far-step exposure per episode and expected/observed cumulative false-Stop
  probability (`mean_episodes sum_{d>40m} 1[stop]` vs
  `mean_episodes sum_{d>40m} p`).
- Stop-event distance distribution.

### Then (closed loop, Dev Val 180, threshold fixed per rule)

- Overall and Short/Medium/Long SR, OSR, SPL, nDTW, premature-Stop, Timeout,
  mean steps, mean minimum distance.
- Paired per-episode comparison against B0: bootstrap 95% CI and exact sign
  test; wins/losses/ties.
- "Reached radius but not success" count.

### Acceptance gate

Keep a variant only if it

1. improves closed-loop SR/OSR by about 3 pp over B0 with no SPL regression;
2. materially reduces the per-step far FPR / predicted cumulative false-Stop
   probability (not only offline F1); and
3. does not collapse near-20 m recall or increase Timeout enough to cancel
   the SR gain.

Single-seed results remain development evidence, not a deployable claim.

## 7. Leakage / discipline

- No Final Test data is loaded by any Stage-2d script.
- No Stage-2a/2b/2c checkpoint is modified.
- Movement/backbone hashes are checked before/after every calibration run.
- The calibration modules accept only `(features, base_logit)`; tests assert
  that true distance / visibility are not part of the input interface.
- `pos_weight` is forbidden; a unit test checks the training source.
- Every closed-loop rule resets its decision state per episode; a unit test
  covers this.
- Only code, tests and reports are committed. Weights, rollout caches,
  per-episode CSVs and provenance stay under the gitignored `runs/` directory.

## 8. Commands (run in this order)

```bash
# R4 decision rules, full closed-loop re-run
python v83_spatial_stage2/stage2d_r4_decision_rules.py \
  --output-dir runs/v83_spatial_stage2/stage2d/r4_<stamp>

# R1/R2/R3 calibration: train on 480, select on 120, then Dev Val closed loop
python v83_spatial_stage2/stage2d_calibration.py \
  --output-dir runs/v83_spatial_stage2/stage2d/calibration_<stamp>

# Tests
python -m unittest v83_spatial_stage2.test_stage2d -v
```
