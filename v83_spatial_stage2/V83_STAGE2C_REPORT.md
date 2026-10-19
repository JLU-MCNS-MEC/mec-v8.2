# V8.3 Stage-2c Arrival Calibration Report

- Date: 2026-10-19
- Branch: `experiment/v83-spatial-foundation`
- Scope: Transformer-M M0 Arrival/Stop only. No Small, M1-Dir, M2, Depth or MEC BEV.
- Data: Stage-2 Train 600 (200/200/200), Dev Val 180 (60/60/60).
- Final Test 300: sealed; not read, evaluated or tuned.
- Closed-loop Stop threshold: `0.5`.
- No old checkpoint was modified. B1/B2 movement and backbone weights are
  byte-identical to the Stage-2a M0 checkpoint.

## 1. Executive summary

1. **The Stage-2b A0/A1 regression was caused by `pos_weight`, not by
   head-only training or the cached feature path.** The original V8.2/V8.3
   Arrival loss is plain `binary_cross_entropy_with_logits` with no positive
   weighting (`train_gbs_v3.py:298`). Stage-2b changed it to
   `BCEWithLogitsLoss(pos_weight = negatives / positives)` = 7.02 (A0) and
   8.64 (A1). With plain BCE, B1/B2 recover M0-level closed-loop SR
   (29.44% / 28.89% vs B0 28.89%); with `pos_weight`, A0/A1 collapse to
   13.89% / 16.67%.
2. **The cached 104-D feature path is exact.** On all 17,234 Dev Val samples,
   cached features + head reproduce the full M0 forward logits with maximum
   absolute difference 3.8e-6 (mean 6.1e-7), well inside float tolerance.
3. **B1/B2 do not improve closed-loop behaviour over B0.** B1 gains one
   episode (SR +0.56 pp), B2 is net zero (Short −1, Long +1); all paired 95%
   CIs include zero and SR/OSR sign-test p = 1.0. B2 does not fail to recover
   M0: it matches B0 on SR/OSR/SPL to reported precision (13/180 episodes
   differ in outcome, wins and losses trading places).
4. **Far-negative retention from 3% to 10% is inert because the added samples
   are mostly already easy.** B2 adds 2,915 far negatives (>60 m); only 22 of
   them have B0 probability ≥ 0.5 (mean probability 0.031). The extra data
   changes offline Brier/F1 slightly but not the high-probability tail that
   causes false Stops.
5. **Closed-loop failure is a per-decision hazard problem.** The on-policy
   far-40 Stop rate is only ~0.35% per decision, but an episode makes ~132
   far-distance decisions. The exact decomposition is
   `132.3 × 0.365% + 3.4 × 2.46% = 0.483 + 0.084 = 0.567`, matching the observed
   56.7% premature-Stop rate. B2's 25% relative offline far-FPR reduction
   becomes only a 5% on-policy reduction (0.365% → 0.347%), far too small to
   change SR. This explains the null result and sets the bar for the next stage.
6. **The next lever is off-policy far-negative mining plus a constrained
   residual calibration, not a larger far-negative ratio.** The protocol is
   prepared in `STAGE2C_OFFPOLICY_PROTOCOL.md`; no calibration training was
   started in this stage.

## 2. Arrival Loss audit

### Original training

`train_gbs_v3.py:298`:

```python
arrival_loss = F.binary_cross_entropy_with_logits(
    arrival_logit[arrival_mask],
    arrival_labels[arrival_mask],
)
```

No `pos_weight`, reduction `mean`, weight 1.0. Stage-2a M0 used seed
`20264017`, batch 48, max 80 epochs, patience 10.

### Stage-2b A0/A1

`stage2b_arrival_ablation.py:223`:

```python
pos_weight = torch.tensor([neg / max(pos, 1)], ...)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

With 1,800 positives and 12,641/15,556 negatives, `pos_weight` is 7.02/8.64.
This is a deliberate re-weighting of the positive class, not a neutral
re-balancing: it shifts the logit at which `sigmoid(logit) = 0.5` and inflates
probabilities on both positives and hard negatives. The resulting closed-loop
collapse is a calibration effect, not a feature or checkpoint-selection effect.

### Controlled contrast

Same 3% / 10% masks, same seed, same head-only budget; the only change is
`pos_weight` → none:

| Mask | Variant | pos_weight | SR | OSR | Premature-Stop | Far-40 FPR | Natural F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 3% | A0 (Stage-2b) | 7.02 | 13.89% | 13.89% | 81.67% | 8.72% | 19.27% |
| 3% | **B1 (Stage-2c)** | none | **29.44%** | **31.67%** | **56.67%** | **0.64%** | **60.37%** |
| 10% | A1 (Stage-2b) | 8.64 | 16.67% | 16.67% | 76.11% | 4.86% | 28.01% |
| 10% | **B2 (Stage-2c)** | none | **28.89%** | **31.11%** | **56.11%** | **0.50%** | **62.23%** |

The plain-BCE variants recover B0 (28.89% SR) to within one episode. The
`pos_weight` variants lose 12–16 pp SR relative to their plain-BCE counterparts
and inflate far false Stops by roughly 10×. This is the main Stage-2c loss
finding.

Closed-loop SR by band isolates the same effect:

| Band | A0 (3%, pos_weight 7.02) | B1 (3%, plain BCE) | A1 (10%, pos_weight 8.64) | B2 (10%, plain BCE) |
|---|---:|---:|---:|---:|
| Short | 30.00% | **41.67%** | 31.67% | **40.00%** |
| Medium | 8.33% | **30.00%** | 13.33% | **30.00%** |
| Long | 3.33% | **16.67%** | 5.00% | **16.67%** |

The pos_weight collapse is largest on Medium/Long, where the policy has more
far-distance decisions and the inflated probability crosses the Stop threshold
more often.

## 3. Cached feature / logit verification

`stage2c_arrival_calibration.py::verify_cached_logits` recomputes the full M0
forward on the Dev Val 180 shards and compares it with
`head(cached 104-D features)` in the same order:

| n | max abs diff | mean abs diff | allclose 1e-6 | allclose 1e-5 |
|---:|---:|---:|---|---|
| 17,234 | 3.81e-6 | 6.06e-7 | true | true |

The cached-feature path and sample ordering are correct. B0/B1/B2 differences
are not caused by feature caching.

## 4. B0/B1/B2 protocol

| Item | B0 | B1 | B2 |
|---|---|---|---|
| Initialization | Stage-2a M0 head | same M0 head | same M0 head |
| Backbone / context encoder | frozen M0 | frozen M0 | frozen M0 |
| Movement | frozen M0 | frozen M0 | frozen M0 |
| Train samples | none | 14,441 | 17,356 |
| Positives / negatives | — | 1,800 / 12,641 | 1,800 / 15,556 |
| Far >60 m included / total | — | 1,330 / 42,454 (3.13%) | 4,245 / 42,454 (10.0%) |
| Loss | original | plain BCE, no pos_weight | plain BCE, no pos_weight |
| Optimizer | — | AdamW, lr 6e-5, wd 1e-4 | same |
| Batch / max epochs / patience | — | 256 / 80 / 10 | 256 / 80 / 10 |
| Seed | — | 20264017 | 20264017 |
| Selection | — | 0.5·(Natural F1 + Hard F1) | same |
| Best epoch | — | 23 (stopped 33) | 20 (stopped 30) |
| Movement hash | reference | unchanged | unchanged |

B2 has 20.2% more optimizer steps per epoch than B1 (more included samples),
so its null result is not caused by a smaller budget.

## 5. Offline Dev Val 180 Arrival metrics (threshold 0.5)

| Variant | Natural F1 | Natural P / R | Hard F1 | Hard P / R | Natural Brier | Hard Brier | Natural ECE | Hard ECE | Far-40 false Stop | Far-100 false Stop | Near-20 miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 59.35% | 0.468 / 0.811 | 73.48% | 0.787 / 0.689 | 0.0097 | 0.1243 | 0.0257 | 0.0823 | 0.67% | 0.74% | 18.89% |
| B1 | 60.37% | 0.479 / 0.817 | 73.48% | 0.787 / 0.689 | 0.0116 | 0.1174 | 0.0433 | 0.0720 | 0.64% | 0.72% | 18.33% |
| B2 | **62.23%** | 0.507 / 0.806 | **73.54%** | 0.796 / 0.683 | **0.0091** | 0.1197 | 0.0312 | 0.0790 | **0.50%** | **0.57%** | 19.44% |

Per band (Natural F1 / Hard F1 / Natural Brier):

| Band | B0 | B1 | B2 |
|---|---:|---:|---:|
| Short | 72.70 / 74.90 / 0.0089 | 72.70 / 75.20 / 0.0100 | 71.60 / 74.70 / 0.0086 |
| Medium | 56.30 / 74.30 / 0.0107 | 56.60 / 74.20 / 0.0121 | **60.50** / 74.70 / **0.0099** |
| Long | 52.70 / 71.20 / 0.0093 | 54.90 / 70.90 / 0.0118 | **56.50** / 71.20 / **0.0088** |

Distance-stratified natural-negative FPR at threshold 0.5 (all bands):

| Distance | n | B0 | B1 | B2 |
|---|---:|---:|---:|---:|
| 20–40 m | 1,392 | 4.89% | 4.81% | 4.89% |
| 40–60 m | 1,337 | 0.37% | 0.37% | 0.22% |
| 60–100 m | 2,474 | 0.53% | 0.40% | 0.36% |
| 100–200 m | 5,172 | 0.43% | 0.39% | 0.29% |
| 200 m+ | 5,599 | 1.04% | 1.04% | 0.82% |

B2 is the best offline model on Natural F1, Brier and far FPR, but the
improvements are concentrated in already-low-probability samples. The
20–40 m annulus remains the hardest region (≈5% FPR), and the far tail is still
around 0.1–1% per sample.

### Training-distribution check

All natural samples in Train 600 and Dev Val 180 have `goal_visible = 1` by
construction, so Goal invisibility is not what separates easy from hard far
negatives. The far-negative pool itself is easy under B0:

| Train natural far >60 m | Count | B0 p ≥ 0.5 | Rate | Mean p |
|---|---:|---:|---:|---:|
| A0 included (3%) | 1,330 | 6 | 0.45% | 0.026 |
| B2 extra (3%→10%) | 2,915 | 22 | 0.75% | 0.031 |
| All far negatives | 42,454 | 318 | 0.75% | 0.030 |

The 3%→10% change adds only 22 high-probability far negatives to a 17k-sample
epoch. It cannot reshape the high-probability tail.

## 6. Closed-loop Dev Val 180 (threshold 0.5)

Overall:

| Variant | SR | OSR | SPL | nDTW | Premature-Stop | Timeout | Mean steps | Mean min distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 28.89% | 31.11% | 0.2877 | 0.3570 | 56.67% | 14.44% | 132.9 | 134.1 m |
| B1 | **29.44%** | **31.67%** | **0.2933** | **0.3615** | 56.67% | **13.89%** | 132.7 | 132.9 m |
| B2 | 28.89% | 31.11% | 0.2877 | 0.3609 | **56.11%** | 15.00% | 136.8 | 131.3 m |

Per band:

| Band | Variant | SR | OSR | SPL | nDTW | Premature-Stop | Timeout | Mean steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Short | B0 | 41.67% | 46.67% | 0.4167 | 0.5235 | 48.33% | 10.00% | 118.2 |
| Short | B1 | 41.67% | 46.67% | 0.4167 | 0.5234 | 50.00% | 8.33% | 115.2 |
| Short | B2 | 40.00% | 45.00% | 0.4000 | 0.5221 | 50.00% | 10.00% | 118.5 |
| Medium | B0 | 30.00% | 31.67% | 0.2979 | 0.3445 | 56.67% | 13.33% | 139.5 |
| Medium | B1 | 30.00% | 31.67% | 0.2979 | 0.3445 | 56.67% | 13.33% | 139.5 |
| Medium | B2 | 30.00% | 31.67% | 0.2979 | 0.3446 | 55.00% | 15.00% | 146.1 |
| Long | B0 | 15.00% | 15.00% | 0.1486 | 0.2030 | 65.00% | 20.00% | 141.1 |
| Long | B1 | **16.67%** | **16.67%** | **0.1652** | **0.2166** | 63.33% | 20.00% | 143.5 |
| Long | B2 | **16.67%** | **16.67%** | **0.1652** | **0.2160** | 63.33% | 20.00% | 145.8 |

Paired per-episode deltas versus B0 (variant − B0; 95% bootstrap CI over
episodes; SR/OSR exact sign-test p):

| Band | Comparison | SR Δ | OSR Δ | Premature Δ | Timeout Δ | SPL Δ | nDTW Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| Overall | B1 | +0.56 pp [0.00, +1.67] | +0.56 pp [0.00, +1.67] | 0.00 pp [−1.67, +1.67] | −0.56 pp [−1.67, 0.00] | +0.0056 [0.0000, +0.0167] | +0.0045 [−0.0000, +0.0136] |
| Overall | B2 | 0.00 pp [−1.67, +1.67] | 0.00 pp [−1.67, +1.67] | −0.56 pp [−2.78, +1.11] | +0.56 pp [0.00, +1.67] | 0.0000 [−0.0167, +0.0167] | +0.0039 [−0.0013, +0.0136] |
| Short | B1 | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | +1.67 pp [0.00, +5.00] | −1.67 pp [−5.00, 0.00] | 0.0000 | −0.00002 |
| Short | B2 | −1.67 pp [−5.00, 0.00] | −1.67 pp [−5.00, 0.00] | +1.67 pp [0.00, +5.00] | 0.00 pp | −0.0167 [−0.05, 0.00] | −0.0013 [−0.0039, 0.0000] |
| Medium | B1 | 0.00 pp | 0.00 pp | 0.00 pp | 0.00 pp | 0.0000 | 0.0000 |
| Medium | B2 | 0.00 pp | 0.00 pp | −1.67 pp [−5.00, 0.00] | +1.67 pp [0.00, +5.00] | 0.0000 | +0.0001 |
| Long | B1 | +1.67 pp [0.00, +5.00] | +1.67 pp [0.00, +5.00] | −1.67 pp [−5.00, 0.00] | 0.00 pp | +0.0167 [0.00, +0.05] | +0.0136 [−0.0000, +0.0407] |
| Long | B2 | +1.67 pp [0.00, +5.00] | +1.67 pp [0.00, +5.00] | −1.67 pp [−5.00, 0.00] | 0.00 pp | +0.0167 [0.00, +0.05] | +0.0130 [−0.0018, +0.0407] |

- SR/OSR sign-test p = 1.0 for every band and both variants; B1 differs from
  B0 on 6/180 episodes and B2 on 13/180, with wins and losses trading places.
- B1 vs B2 is also null: SR −0.56 pp [−1.67, 0.00], premature-Stop −0.56 pp
  [−2.22, +1.11], Timeout +1.11 pp [0.00, +2.78].
- **Neither B1 nor B2 is a candidate improvement.** They are successful
  protocol recoveries, not new policies.
- Closed-loop "reached the radius but did not succeed" is only 4/180 (2.22%)
  and identical for B0/B1/B2 (Short 3, Medium 1, Long 0). The dominant failure
  is premature Stop, not a missed near-distance Stop.
- Compared with the never-Stop counterfactual (B0 OSR Short/Medium/Long
  60.0%/46.7%/45.0%; overall 50.6%), the current stop policy still leaves
  17–30 pp of reachable-success headroom, largest on Long. Fixing Stop decisions
  is therefore worth up to ~21.7 pp overall before changing movement.

## 7. Stop diagnostics and per-decision hazard

### 7.1 Stop events

Stop events on Dev Val 180 (Goal visibility is an offline simulator diagnostic,
never a model input):

| Variant | Band | Stops | Premature >20 m | Far >40 m | Far >100 m | Stop distance median | Arrival p at Stop | Goal visible at Stop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | all | 154 | 102 | 87 | 78 | 102.3 m | 0.761 | 70.1% |
| B0 | short | 54 | 29 | 24 | 19 | 21.0 m | 0.775 | 66.7% |
| B0 | medium | 52 | 34 | 29 | 27 | 115.5 m | 0.773 | 67.3% |
| B0 | long | 48 | 39 | 34 | 32 | 236.8 m | 0.732 | 77.1% |
| B1 | all | 155 | 102 | 87 | 78 | 101.2 m | 0.709 | 69.7% |
| B1 | short | 55 | 30 | 25 | 20 | 21.3 m | 0.720 | 65.5% |
| B1 | medium | 52 | 34 | 29 | 27 | 115.5 m | 0.724 | 67.3% |
| B1 | long | 48 | 38 | 33 | 31 | 229.1 m | 0.680 | 77.1% |
| B2 | all | 153 | 101 | 85 | 76 | 91.8 m | 0.716 | 68.6% |
| B2 | short | 54 | 30 | 24 | 19 | 21.0 m | 0.729 | 64.8% |
| B2 | medium | 51 | 33 | 28 | 26 | 101.2 m | 0.723 | 66.7% |
| B2 | long | 48 | 38 | 33 | 31 | 229.1 m | 0.693 | 75.0% |

The stop-event distributions are essentially identical. B2 shortens the Long
median stop distance by ~8 m and reduces Long far>40 stops by one, but the
number of Long premature stops is unchanged (38 vs 39). Most stops are far
away, the head is confident at them (p ≈ 0.68–0.78), and the Goal is projected
visible at 65–77% of stops: the failure is not invisibility.

### 7.2 Offline vs on-policy calibration by distance

Per-decision false-Stop rate by distance (offline: natural-negative FPR;
on-policy: fraction of rollout decisions with Stop in that distance bin):

| Distance | B0 offline | B0 rollout | B1 rollout | B2 offline | B2 rollout |
|---|---:|---:|---:|---:|---:|
| 20–40 m | 4.89% | 2.46% | 2.16% | 4.89% | 2.31% |
| 40–60 m | 0.37% | 0.19% | 0.19% | 0.22% | 0.19% |
| 60–100 m | 0.53% | 0.29% | 0.29% | 0.36% | 0.28% |
| 100–200 m | 0.43% | 0.32% | 0.32% | 0.29% | 0.31% |
| 200 m+ | 1.04% | 0.45% | 0.45% | 0.82% | 0.41% |

Two observations:

1. On-policy per-decision rates are **not systematically worse** than offline:
   for B0/B1 they are lower at every far bin, and for B2 they are similar (one
   bin is 0.02 pp higher). The policy does not visit a materially harder state
   distribution per decision. (The earlier hypothesis that rollout far states
   are systematically out of distribution is not supported at this level.)
2. B2's offline improvement does not reach the on-policy tail. B2 cuts offline
   200 m+ FPR from 1.04% to 0.82% (21% relative) and far-40 from 0.67% to 0.50%
   (25% relative), but the on-policy 200 m+ Stop rate only moves from 0.45% to
   0.41% (9% relative). The states that actually cause B0/B2 stops are not the
   ones B2 improved.

### 7.3 Hazard decomposition

Because an episode terminates at the first Stop, the episode premature-Stop
rate equals the expected number of Stop decisions per episode. Summing
per-decision rates over the decisions actually made gives an exact
decomposition:

| Variant | Far>40 decisions / episode | Far>40 Stop rate / decision | 20–40 decisions / episode | 20–40 Stop rate / decision | Premature-Stop | Decomposition |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 132.3 | 0.365% | 3.4 | 2.46% | 56.67% | 0.483 + 0.084 = 0.567 |
| B1 | 132.1 | 0.366% | 3.9 | 2.16% | 56.67% | 0.483 + 0.084 = 0.567 |
| B2 | 136.2 | 0.347% | 3.9 | 2.31% | 56.11% | 0.472 + 0.089 = 0.561 |

This is the central Stage-2c diagnostic result:

- A per-decision far Stop rate of **0.35% is not small** when the policy makes
  ~132 far decisions per episode. It yields ~0.48 expected far false Stops per
  episode; the 20–40 m annulus adds ~0.09. The sum is the observed ~0.57
  premature-Stop probability.
- To bring premature Stop to ~20%, the far per-decision rate would have to fall
  from ~0.35% to ~0.08% (4–5×), even before the boundary term. B2's 25%
  relative offline FPR reduction is far short of that; its 5% relative
  on-policy reduction explains the null SR change.
- This is why more random/expert far negatives cannot work, and why the next
  stage must target the actual high-probability rollout decisions (off-policy
  mining) and should judge candidates by per-decision far FPR, not offline F1.
  A threshold/persistence decision rule is a complementary cheap control: the
  same arithmetic says a rule that removes most far false decisions is worth
  more than a small probability recalibration.

The Never-Stop counterfactual (Stage-2b) puts the movement-only reachable
ceiling at 50.6% overall OSR; B0 SR is 28.9%, so Stop errors cost ~21.7 pp even
without changing movement. The decomposition above says that headroom is
distributed over many rare far decisions, not one systematic near-goal error.

### 7.4 Feature signal and threshold trade-off

The 104-D Arrival features already encode distance. On Dev Val natural
negatives, the correlation with log-distance is −0.75 for `best_y`, −0.60 for
`weighted_y` and −0.45 for `topk_mean`; a linear probe predicts log-distance
with held-out-episode R² = 0.61. The 98 high-p far negatives are also not
feature-identical to near positives: compared with the 180 near positives,
`best_y` is −0.49 vs +0.54 (Cohen d −2.14), entropy 5.99 vs 5.12 (d 1.13) and
`mean_match` 0.68 vs 0.57 (d 0.81). The base head gives them mean probability
0.69 anyway. The signal for a residual is present; the base mapping is the
problem.

The offline threshold trade-off (B0) is:

| Threshold | Natural F1 | Hard F1 | Far-40 FPR | Near-20 recall |
|---:|---:|---:|---:|---:|
| 0.5 | 59.35% | 73.48% | 0.67% | 81.11% |
| 0.6 | 61.93% | 72.09% | 0.47% | 75.00% |
| 0.7 | 65.12% | 69.65% | 0.32% | 70.00% |
| 0.8 | 65.86% | 63.32% | 0.14% | 60.56% |

Raising the threshold cuts far FPR but also near recall, so it is not a free
fix; a closed-loop threshold/persistence sweep is a cheap decision-layer
control for the next stage (R4 in the off-policy protocol). The B0/B1/B2
comparison above deliberately stays at threshold 0.5.

## 8. Why B2 is inert (and what would move the needle)

Four independent pieces of evidence:

1. **The added far negatives are already classified correctly.** Only 22 of the
   2,915 extra samples have B0 p ≥ 0.5; their mean p is 0.031. Increasing the
   ratio from 3% to 10% adds almost no gradient in the high-probability tail.
2. **The offline gain does not reach the on-policy tail.** B2 reduces offline
   far-40 FPR by 25% relative (0.67%→0.50%), but the on-policy far-40 Stop rate
   only falls 5% relative (0.365%→0.347%). The states that cause rollout stops
   are not the expert far states that B2 improved.
3. **The head remains in the B0 basin.** B1/B2 first-layer and hidden weights
   have cosine similarity 0.998–0.999 with B0; relative parameter change is only
   4–8%. Offline metrics move by small amounts, and the decision boundary at
   0.5 barely moves.
4. **Closed-loop failure is dominated by repeated decisions, not one bad
   sample.** A 0.35% per-decision far Stop rate over ~132 far decisions per
   episode already yields ~0.48 expected false stops per episode. The correct
   target is the high-p far tail *on the policy's own state distribution*, not
   more random expert far negatives.

This is why the next stage collects off-policy far negatives from Train 600
rollouts, labels them with true distance only, and fits a lightweight residual
on top of the preserved B0 head with an explicit base-prediction constraint.
The full Train-600 collection is already available (181 high-p far negatives,
462 near-20 positives); a 3.4k-parameter residual can use it, but the small
hard-negative count means strong regularization, an episode-level split and the
base-prediction anchor are mandatory. If the calibrated policy changes the
trajectory, collect one or two DAgger rounds. The protocol (data split,
variants, metrics, acceptance gate, leakage checks) is in
`STAGE2C_OFFPOLICY_PROTOCOL.md`.

## 9. Discipline / what was not run

- No Final Test read.
- No old checkpoint modified; B1/B2 movement hashes verified unchanged.
- No Small, M1-Dir, M2, Depth or full MEC training started.
- No Stage-2b pos_weight protocol reused.
- No further increase of the far-negative ratio after B2.

## 10. Artifacts and reproduction

- Stage-2c B0/B1/B2 run:
  `runs/v83_spatial_stage2/stage2c/20261018T164111Z/`
  - `feature_logit_verification.json`, `mask_stats.json`, `train_summary.json`
  - `B0/metrics.json`, `B1/`, `B2/` (head, full checkpoint, history, metrics)
  - `B1_per_episode.csv`, `B2_per_episode.csv`, `stage2c_analysis.json`
- Diagnostics:
  `runs/v83_spatial_stage2/stage2c/diag_20261019T010500Z/`
  - `offline_arrival_metrics.json`, `offline_distance_bins.csv`,
    `offline_feature_stats.csv`
  - `rollout_steps.csv`, `rollout_stop_events.csv`,
    `rollout_distance_bins.csv`, `rollout_stop_summary.csv`,
    `rollout_episode_summary.csv`, `rollout_per_episode.csv`
  - `replay_check.json` (0/180 outcome mismatches for B0/B1/B2),
    `paired_vs_B0.csv`, `stage2c_diagnostics.json`
- Hazard decomposition:
  `runs/v83_spatial_stage2/stage2c/risk_20261019T012000Z/`
- Off-policy preparation:
  - Smoke: `runs/v83_spatial_stage2/stage2c/offpolicy_smoke_20261019T013000Z/`
    (6 episodes, 677 steps, 618 far negatives, 1 with B0 p ≥ 0.5).
  - Full Train 600:
    `runs/v83_spatial_stage2/stage2c/offpolicy_train_rollouts_20261019T013500Z/`
    (600 episodes, 74,001 steps, 70,368 far>40 negatives, 181 with B0
    p ≥ 0.5, 520 stops with 181 at >40 m and 160 at >100 m; SR 45.5%,
    premature-Stop 41.2%, Timeout 13.3%). Train rollouts are easier than
    Dev Val because the base model was trained on them, but the pool is large
    enough for a first residual fit and matches the Dev Val high-p tail size
    (181 vs 98).
- Code:
  - `v83_spatial_stage2/stage2c_arrival_calibration.py`
  - `v83_spatial_stage2/stage2c_diagnostics.py`
  - `v83_spatial_stage2/stage2c_finalize_diagnostics.py`
  - `v83_spatial_stage2/stage2c_risk_analysis.py`
  - `v83_spatial_stage2/stage2c_collect_rollouts.py`
  - `v83_spatial_stage2/test_stage2c.py`
- Commands:

```bash
# B0/B1/B2 controls (already run; do not overwrite)
python v83_spatial_stage2/stage2c_arrival_calibration.py \
  --output-dir runs/v83_spatial_stage2/stage2c/<new-stamp>

# Offline + rollout calibration/stop diagnostics
python v83_spatial_stage2/stage2c_diagnostics.py \
  --output-dir runs/v83_spatial_stage2/stage2c/diag_<new-stamp>

# Recompute post-processing from saved rollout CSVs if the replay check was interrupted
python v83_spatial_stage2/stage2c_finalize_diagnostics.py \
  --diagnostics-dir runs/v83_spatial_stage2/stage2c/diag_<new-stamp>

# Per-decision hazard decomposition
python v83_spatial_stage2/stage2c_risk_analysis.py \
  --diagnostics-dir runs/v83_spatial_stage2/stage2c/diag_<new-stamp> \
  --output-dir runs/v83_spatial_stage2/stage2c/risk_<new-stamp>

# Off-policy Train-600 rollout collection (prepared; no training)
python v83_spatial_stage2/stage2c_collect_rollouts.py \
  --dataset-root runs/v83_spatial_stage2/datasets/train_600 \
  --output-dir runs/v83_spatial_stage2/stage2c/offpolicy_train_rollouts_<new-stamp>

# Tests
python -m unittest v83_spatial_stage2.test_stage2c -v
```
