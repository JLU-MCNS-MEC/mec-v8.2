# V8.3 Stage-2e — M1 Spatial Fusion and Movement Diagnosis

- Date: 2026-10-19
- Branch: `experiment/v83-spatial-foundation`
- Scope: eval-only diagnosis of the existing Transformer-M M0@0.7 and M1@0.7
  checkpoints. **No training, no architecture change, no checkpoint
  modification, no other MEC trained.**
- Data: Dev Val 180 (Short/Medium/Long 60/60/60). Final Test 300 is sealed
  and was not read.
- M0 checkpoint:
  `runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z/Transformer-M/M0/best.pt`
  (sha256 `1c496fa8bd880592089ad85408ad7244c4c3e95817c6c4fdd23a7f5ecd5bd9e4`).
- M1 checkpoint:
  `runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z/Transformer-M/M1/best.pt`
  (sha256 `6cac7883419ebeae172b6f0e9dc1ddf6b87e4c0dc7f8994bb3e185e044287b00`).
- M0@0.7 and M1@0.7 logs are the existing Stage-2d runs; M0 was not re-run.
- The only closed-loop re-run is the **fusion-residual-off control** (§6),
  which re-evaluates the same 180 episodes with the M1 fusion output forced to
  zero. Everything else is exact-state replay plus extra forward passes.

## 0. Evidence base and replay verification

| Item | Result |
|---|---|
| Replayed episodes | 180 M0 trajectories + 180 M1 trajectories = 360 |
| Replayed states | 55,414 |
| Distance mismatch | 0 passes (max error 0.0 m) |
| Logged move-action mismatch | 0 passes |
| Logged arrival-probability mismatch | 0 passes (max error 0.0) |
| Off-policy models evaluated on identical states | M0, M1 base (no residual), M1 fused, mirrored-token M0/M1 base, mirrored-map fusion |
| Fusion-off closed loop | 180 episodes at threshold 0.7 |

Because every saved state is reconstructed exactly and the owner model
reproduces its logged `move_action` and `arrival_probability`, the off-policy
comparisons below are on identical inputs, not on stitched trajectories.

The 3-way "oracle" movement label is the privileged heading rule
(`|heading error| <= 7.5° -> FORWARD`, otherwise turn toward the goal). It is
a diagnostic only and is never used as controller input.

## 1. Compact band summary

| Band | Method | SR | OSR | SPL | nDTW | Premature | Timeout | Reach-no-success | Min dist m | Steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | M0@0.7 | 34.44% | 38.33% | 0.343 | 0.357 | 48.89% | 16.67% | 3.89% | 126.4 | 147.9 |
| Overall | M1@0.7 | 31.11% | 36.67% | 0.311 | 0.368 | 47.22% | 21.67% | 5.56% | 119.2 | 159.9 |
| Short | M0@0.7 | 48.33% | 56.67% | 0.483 | 0.518 | 36.67% | 15.00% | 8.33% | 51.8 | 141.0 |
| Short | M1@0.7 | 43.33% | 48.33% | 0.433 | 0.531 | 38.33% | 18.33% | 5.00% | 51.2 | 140.8 |
| Medium | M0@0.7 | 33.33% | 36.67% | 0.331 | 0.337 | 51.67% | 15.00% | 3.33% | 126.3 | 153.9 |
| Medium | M1@0.7 | 23.33% | 33.33% | 0.233 | 0.331 | 48.33% | 28.33% | 10.00% | 113.8 | 181.6 |
| Long | M0@0.7 | 21.67% | 21.67% | 0.215 | 0.217 | 58.33% | 20.00% | 0.00% | 201.3 | 148.8 |
| Long | M1@0.7 | 26.67% | 28.33% | 0.267 | 0.243 | 55.00% | 18.33% | 1.67% | 192.5 | 157.4 |

Paired M1@0.7 vs M0@0.7 (from the Stage-2d report, bootstrap 95% CI):

| Band | SR Δ (W/L, CI) | Timeout Δ (CI) | SPL Δ (CI) | nDTW Δ (CI) |
|---|---|---|---|---|
| Overall | −3.33 pp [−10.00, +3.33] (16W/22L) | +5.00 pp [−2.22, +12.22] | −0.0321 [−0.0978, +0.0340] | +0.0111 [−0.0389, +0.0626] |
| Short | −5.00 pp [−18.33, +8.33] (6W/9L) | +3.33 pp [−8.33, +15.00] | −0.0500 [−0.1833, +0.0833] | +0.0134 [−0.0895, +0.1182] |
| Medium | −10.00 pp [−21.67, 0.00] (3W/9L) | +13.33 pp [+1.67, +26.67] | −0.0979 [−0.2125, +0.0042] | −0.0058 [−0.0953, +0.0827] |
| Long | +5.00 pp [−5.00, +16.67] (7W/4L) | −1.67 pp [−13.33, +10.00] | +0.0515 [−0.0485, +0.1667] | +0.0256 [−0.0472, +0.0992] |

Only the Medium Timeout increase is statistically resolved. The Long SR gain
is not (sign-test p≈0.55) and is analysed in §7.

## 2. Exact failure partition and loss attribution

Every episode falls in exactly one bucket. `P` = premature Stop, `T` =
timeout, `far` = never entered the 20 m radius, `reached` = entered the radius
at least once; `S` = success.

| Band | Method | S | P-far | P-reached | T-far | T-reached |
|---|---|---:|---:|---:|---:|---:|
| Overall | M0 | 62 | 83 | 5 | 28 | 2 |
| Overall | M1 | 56 | 80 | 5 | 34 | 5 |
| Short | M0 | 29 | 18 | 4 | 8 | 1 |
| Short | M1 | 26 | 21 | 2 | 10 | 1 |
| Medium | M0 | 20 | 30 | 1 | 8 | 1 |
| Medium | M1 | 14 | 27 | 2 | 13 | 4 |
| Long | M0 | 13 | 35 | 0 | 12 | 0 |
| Long | M1 | 16 | 32 | 1 | 11 | 0 |

Attribution used below:

- **Stop-attributable failure** = P-far + P-reached + T-reached.
  (Stopped before/outside the target, or reached but never crossed the Stop
  threshold.)
- **Movement/search-attributable failure** = T-far.
  (Never entered the 20 m radius within 400 steps.)
- `ΔSR = −Δ(Stop failures) − Δ(Movement/search failures)` exactly.

| Band | ΔSR (episodes) | Δ Stop failures | Δ Movement/search failures | Reading |
|---|---:|---:|---:|---|
| Overall | −6 (−3.33 pp) | 0 | +6 | M1's Stop-error count is unchanged in aggregate (−3 P-far, +3 T-reached); **all net SR loss is 6 extra far timeouts** |
| Short | −3 | +1 | +2 | +3 P-far, −2 P-reached, +2 T-far |
| Medium | −6 | +1 | +5 | **5 extra far-timeout failures net (4 of the 9 lost successes)**; Stop is −3 P-far but +3 T-reached +1 P-reached |
| Long | +3 | −2 | −1 | mostly fewer far premature stops (−3 P-far), offset by +1 P-reached |

This is the main correction to the M1@0.7 headline: at 0.7, M1 is not simply
"stopping worse" or "moving worse". It trades far premature Stops for
timeouts almost one-for-one overall, and its net regression is concentrated in
**never reaching the target** (far timeouts), especially in Medium.

## 3. Medium band deep dive

M1 Medium 14/60 vs M0 Medium 20/60: **9 M0 successes lost, 3 M0 failures
gained, net −6**.

| Lost-success outcome under M1 | Count | Share |
|---|---:|---:|
| Never reached — timeout (T-far) | 4 | 44% |
| Never reached — premature Stop (P-far) | 2 | 22% |
| Reached but missed Stop — timeout (T-reached) | 2 | 22% |
| Reached but missed Stop — premature Stop (P-reached) | 1 | 11% |
| Other | 0 | 0% |

So the requested decomposition is:

- **Never reached: 6/9 (67%)**, of which 4 are movement/search timeouts and 2
  are premature Stops before arrival.
- **Reached but missed Stop: 3/9 (33%)**, of which 2 are timeouts after
  entering the radius and 1 is a premature Stop after entering the radius.
- **Other: 0.**

The Medium near-goal summary is 85 near steps, 54.1% goal visible, mean
|heading error| 62.5°, oracle needs turn on 84.7% of near steps, M1 movement
accuracy 31.0%, and 59.2% missed turns on executed move steps.

The 3 gained episodes are 2 M0 far premature Stops and 1 M0 far timeout, so
they do not change the attribution.

Case evidence for the "reached but missed Stop" group:

| Episode | Reached min dist | Steps | Near-goal heading error | Goal visible near goal | M1 arrival p near goal | Outcome |
|---|---:|---:|---|---|---:|---|
| 3 | 11.5 m | 400 | 46.6° to 141.7° | 0/10 near steps | 0.006–0.29 | T-reached |
| 177 | 5.4 m | 400 | −8.2° to −158.9° | 5/12 near steps | 0.027–0.538 | T-reached |

In both cases the agent enters the radius but turns away from the target. M0's
arrival head on the same states is also low (0.01–0.14), so this is not simply
"the M1 threshold is too high"; the state is visually unfavourable because the
goal is out of view. This is an orientation/search failure coupled to the
Stop head, not a global threshold problem.

Near-goal evidence on M1's own trajectory inside 20 m:

| Band | Near steps | Moves | Goal visible | Mean \|heading error\| | Oracle needs turn (all near steps) | M1 move acc | Missed turn (moves) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 61 | 35 | 73.8% | 40.6° | 73.8% | 37.1% | 60.0% |
| Medium | 85 | 71 | **54.1%** | **62.5°** | **84.7%** | **31.0%** | **59.2%** |
| Long | 29 | 13 | 75.9% | 34.1° | 82.8% | 38.5% | 61.5% |

On identical near-goal states from the M1 Medium trajectory, M0's arrival
head is only slightly more confident than M1's (mean p 0.309 vs 0.271; Stop
rate at 0.7 22.4% vs 16.5%). The M1 arrival head is also worse offline
(natural Arrival-F1 54.69% vs M0 59.11%). So the Medium missed-Stop loss has
two contributors: a weaker M1 arrival head and, more importantly, arriving
near the goal without facing it.

## 4. Movement errors

Movement accuracy against the privileged heading oracle, on executed move
steps (Stop steps excluded). Full tables are in `movement_stats.csv` and
`movement_distance_stats.csv`.

M1's own trajectory, by true distance:

| Distance | Steps | M0 acc | M1 base acc | M1 fused acc | M1 missed turn | M1 spurious turn |
|---|---:|---:|---:|---:|---:|---:|
| 0–20 m | 119 | 37.8% | 33.6% | 33.6% | 59.7% | 0.0% |
| 20–40 m | 1,255 | 71.9% | 65.0% | 65.3% | 6.0% | 28.0% |
| 40–100 m | 6,177 | 62.4% | 53.3% | 53.1% | 8.7% | 29.4% |
| 100–200 m | 9,017 | 55.8% | 52.0% | 52.0% | 15.0% | 25.0% |
| 200 m+ | 12,077 | 34.7% | 36.9% | 37.0% | 40.6% | 6.3% |

Key points:

- Movement accuracy is high at 20–40 m (65–80%) and collapses **inside 20 m**
  (34–46%) and far away (33–37%).
- The near-goal collapse is dominated by **missed turns**: the oracle says
  turn on 80.6% of near-goal steps, but M1 keeps going forward 59.7% of
  near-goal move steps. This is exactly the orientation failure behind the
  Medium reached-timeouts.
- M1 is not uniformly a better or worse movement controller than M0. On M1's
  own states it is worse in Short and Medium (e.g. Medium 42.9% vs 48.0%) and
  better only in Long (46.1% vs 42.3%). On M0's states, Medium M1 is 38.2% vs
  43.6% (worse), Long both 46.2% (equal). The Long movement advantage is
  state-distribution dependent, not a robust policy improvement.
- Timeouts and minimum distance agree with this: M1 Medium has 28.3% timeouts
  vs 15.0%, 181.6 steps vs 153.9, while its mean minimum distance is actually
  lower (113.8 m vs 126.3 m). M1 gets closer on average but fails to convert.

## 5. SpatialFusionHead check

Architecture (unchanged): `Conv2d(2→16,3×3) → GELU → Conv2d(16→16,3×3) →
GELU → AdaptiveAvgPool2d(1) → Flatten → Linear(16→3)`, 2,675 parameters.
The residual is added to the base 3-way movement logits.

### 5.1 Mirror tests on real states

Horizontal mirror = token x-axis / spatial-map x-axis flip. A direction-aware,
mirror-equivariant head should satisfy `r_left(mirror) ≈ r_right(original)`
and `L−R(mirror) ≈ −(L−R)(original)`.

| Trajectory | Steps | Residual invariance mean / max | L−R mirror flip rate | Fusion equivariance relative error | Base equivariance relative error | M0 equivariance relative error |
|---|---:|---|---:|---:|---:|---:|
| M0 states | 16,075 | 1.12e−3 / 9.02e−3 | 0.02% | **1.999** | 0.731 | 0.827 |
| M1 states | 18,055 | 1.14e−3 / 9.32e−3 | 0.03% | **1.999** | 0.670 | 0.750 |

Scale: 0 = perfect mirror equivariance; 2 = perfectly mirror-invariant (the
head cannot tell left from right). The fusion residual is at the invariant
end; the base is partially equivariant.

### 5.2 Directional logit response

States where the goal is visible and left/right of image centre, M1 arrival
p < 0.7 (`L−R` score; AUC 1.0 = perfect left/right discrimination):

| Trajectory | Score | Steps | Left mean | Right mean | AUC(left) | Balanced acc |
|---|---|---:|---:|---:|---:|---:|
| M1 states | Fusion residual | 18,055 | +0.073 | +0.066 | **0.560** | **0.500** |
| M1 states | M1 base | 18,055 | +5.661 | −5.635 | **0.901** | 0.849 |
| M1 states | M1 fused | 18,055 | +5.734 | −5.569 | 0.902 | 0.849 |
| M1 states | M0 | 18,055 | +5.684 | −5.595 | 0.903 | 0.839 |
| M1 states | Coherent map centroid | 18,055 | +0.038 | −0.023 | 0.684 | 0.632 |
| M0 states | Fusion residual | 16,075 | +0.061 | +0.061 | **0.517** | **0.499** |
| M0 states | M1 base | 16,075 | +5.217 | −4.192 | 0.868 | 0.812 |

The fusion residual carries essentially no left/right signal (AUC ≈ 0.5),
while the M1 base already carries strong directional signal when the goal is
visible (AUC 0.87–0.90). The fusion head does not add a directional path; it
adds a near-constant global bias.

### 5.3 Where the direction information is lost

Synthetic left/right Gaussian blobs through the trained fusion trunk:

| Probe | Left/right max abs difference |
|---|---:|
| Conv feature map (16×24×24) | 1.3305 |
| Pooled 1×1 (16) | 1.305e−4 |
| Pooled 2×2 (64) | 0.1340 |
| Trained fusion residual | 5.865e−5 |

The conv trunk represents the left/right difference, but `AdaptiveAvgPool2d(1)`
collapses it to floating-point noise before the linear layer. A frozen-feature
linear probe on real states confirms the same thing:

| Features | Dim | Train AUC | Test AUC | Test balanced acc |
|---|---:|---:|---:|---:|
| Pooled 1×1 (current) | 16 | 0.648 | **0.548** | 0.526 |
| Pooled 2×2 | 64 | 0.920 | **0.832** | 0.762 |

**Conclusion for item 3: yes, `AdaptiveAvgPool2d(1)` makes left and right
spatially indistinguishable at the head output.** This is a proven
architectural limitation, not a hypothesis.

## 6. Fusion residual magnitude and effect

Residual = `M1 fused move_logits − M1 base move_logits`, evaluated on M1's own
trajectory at states where M1 would not Stop (`p_arrival < 0.7`).

| Distance | Steps | Residual L2 | Relative L2 | Action changed | Beneficial/1k | Harmful/1k | Net/1k |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–20 m | 119 | 0.096 | 1.72% | **0.00%** | 0.00 | 0.00 | 0.00 |
| 20–40 m | 1,255 | 0.196 | 3.09% | 0.24% | 2.39 | 0.00 | +2.39 |
| 40–100 m | 6,177 | 0.143 | 2.15% | 0.45% | 0.97 | 3.08 | −2.10 |
| 100–200 m | 9,017 | 0.088 | 1.31% | 0.40% | 1.77 | 2.00 | −0.22 |
| 200 m+ | 12,077 | 0.089 | 1.41% | 0.33% | 1.66 | 0.50 | +1.16 |
| All | 28,645 | 0.105 | ≈1.5% | **0.37% (107 steps)** | 1.57 | 1.50 | **+0.07** |

Beneficial = residual makes the argmax agree with the heading oracle while the
base argmax did not; harmful = the reverse. By band (M1 trajectory):

| Band | Changed steps | Beneficial | Harmful | Net per 1k |
|---|---:|---:|---:|---:|
| Short | 35 | 6 | **27** | **−2.50** |
| Medium | 51 | 28 | 8 | +1.84 |
| Long | 21 | 11 | 8 | +0.32 |

### 6.1 Which actions change? (base → fused transition matrix)

On M1's own trajectory, p_arrival < 0.7. `F/L/R` are the 3-way Movement
argmax before/after the fusion residual.

| Distance | Steps | Changed | Rate | F→L | F→R | L→F | L→R | R→F | R→L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–20 m | 119 | 0 | 0.00% | 0 | 0 | 0 | 0 | 0 | 0 |
| 20–40 m | 1,255 | 3 | 0.24% | 2 | 1 | 0 | 0 | 0 | 0 |
| 40–100 m | 6,177 | 28 | 0.45% | 28 | 0 | 0 | 0 | 0 | 0 |
| 100–200 m | 9,017 | 36 | 0.40% | 27 | 9 | 0 | 0 | 0 | 0 |
| 200 m+ | 12,077 | 40 | 0.33% | 25 | 14 | 0 | 0 | 1 | 0 |
| **All** | **28,645** | **107** | **0.37%** | **82** | **24** | **0** | **0** | **1** | **0** |

| Band | Steps | Changed | Rate | F→L | F→R | R→F | Beneficial | Harmful | Neutral | Heading improved | Heading worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 8,400 | 35 | 0.42% | 31 | 3 | 1 | 6 | **27** | 2 | 18 | 17 |
| Medium | 10,850 | 51 | 0.47% | 35 | 16 | 0 | 28 | 8 | 15 | 35 | 16 |
| Long | 9,395 | 21 | 0.22% | 16 | 5 | 0 | 11 | 8 | 2 | 17 | 4 |

Interpretation:

- **106 of 107 changed decisions are FORWARD→turn**; there is exactly one
  R→F and **zero** L→F, L→R, R→L. The fusion branch is therefore a global
  "turn more" bias, not a left/right correction. This is exactly what a
  translation-invariant 1×1-pooled head can express.
- The 82 F→L vs 24 F→R split is a learned global asymmetry, not a response to
  the target side: the mirror test shows the head gives the same output when
  the map is flipped.
- **Short:** 27 of 35 changes are harmful; they are mostly spurious F→L turns
  (31 F→L, 6 beneficial vs 27 harmful). This agrees with the high spurious-turn
  rate on Short (29.1%).
- **Medium:** 28 beneficial vs 8 harmful and 35 vs 16 on the one-step heading
  metric, so the local changes are directionally sensible, but they are only
  0.47% of steps; zeroing the residual leaves Medium SR at 14/60.
- **Long:** 11 beneficial vs 8 harmful, 17 vs 4 on heading, but only 0.22% of
  steps; zeroing the residual leaves Long SR at 16/60 with zero success
  changes.

One-step heading is `|heading error|` after the action relative to the base
action: LEFT = −15°, RIGHT = +15°, FORWARD uses the 5 m step and the current
goal geometry. Immediate distance is not a useful harm measure for these
changes: a turn keeps distance while the FORWARD baseline moves 5 m closer, so
92/107 are mechanically "farther than the forward action"; the heading and
oracle columns are the meaningful ones.

The residual is tiny relative to the base logit scale, changes only 0.37% of
movement decisions, is **completely inert inside 20 m** (0/119 changes), and
its net effect is indistinguishable from zero overall. On Short it is locally
net harmful.

### Fusion-off closed loop (eval only)

The M1 checkpoint was loaded unchanged and `model.fusion` was replaced by an
exact-zero module; base weights and arrival head are still M1's. This isolates
the inference-time residual of the current checkpoint. **It is not an
independently trained M0**: M1's base weights differ from M0's.

| Band | M1 SR | Fusion-off SR | SR Δ (W/L) | SPL Δ | nDTW Δ | Timeout Δ |
|---|---|---:|---|---:|---:|---:|
| Overall | 31.11% | 32.22% | +1.11 pp (4W/2L) | +0.0111 | +0.0067 | −1.67 pp |
| Short | 43.33% | 46.67% | +3.33 pp (2W/0L) | +0.0333 | +0.0144 | −3.33 pp |
| Medium | 23.33% | 23.33% | 0.00 pp (2W/2L) | 0.0000 | +0.0016 | 0.00 pp |
| Long | 26.67% | 26.67% | **0.00 pp (0W/0L)** | 0.0000 | +0.0041 | −1.67 pp |

Long outcome transitions with the residual off: 16 success→success,
32 P-far→P-far, 1 P-reached→P-reached, 1 T-far→P-far, 10 T-far→T-far.
**Zero Long successes change.** Short gains 2 successes (both T-far→success).

The overall gain from removing the residual is +1.11 pp with bootstrap CI
[−1.67, +3.89] (not resolved); the Short gain is +3.33 pp (2W/0L, CI
[0.00, +8.33]) and the Long effect is exactly 0.

**Conclusion for item 6:** the current fusion residual is neutral overall and
slightly harmful on Short; removing it does not reproduce M0, but it does
show that the residual is not responsible for any M1 advantage. Fusion-off is
a diagnostic, not a deployable candidate: at 32.22% overall it is still below
M0@0.7 (34.44%), and it keeps M1's weaker arrival head.

## 7. Is the Long advantage due to directional correction or weight variation?

Facts:

1. M1 Long SR 16/60 vs M0 13/60: +5.00 pp, paired 7W/4L, 95% CI
   [−5.00, +16.67], sign-test p≈0.55. **Not statistically resolved.**
2. Six of the seven gains are M0 **far premature Stops** that M1 did not make;
   one is an M0 far timeout. Three of the four losses are M1 far premature
   Stops. The exact Long partition is −3 P-far, +1 P-reached, −1 T-far:
   **Stop behaviour accounts for +2 of the +3 SR, movement/search for +1.**
3. The fusion residual on Long states changes 21/9,395 movement decisions
   (0.22%), 11 beneficial vs 8 harmful, and has an L−R direction AUC ≈ 0.5.
4. The fusion-off Long closed loop is **exactly 16/60**, with zero success
   changes. The Long gain survives complete removal of the fusion residual.
5. M1 base weights differ substantially from M0 (relative L2: movement head
   ~13%, arrival head ~8%, goal-current attention ~54%). Offline movement
   Macro-F1 is essentially equal (M1 75.03% vs M0 74.75%), while natural
   Arrival-F1 is worse (54.69% vs 59.11%). There is no evidence of a better
   learned movement representation.
6. Off-policy movement accuracy on Long: on M1's states M1 is 46.1% vs M0
   42.3%, but on M0's states both are 46.2%. The apparent Long movement gain
   is state-distribution dependent.

**Verdict for item 7:** the Long advantage is **not** attributable to spatial
fusion or effective direction correction. It is a Stop-timing reshuffle
produced by the jointly trained M1 base/arrival weights (and is within noise
on 60 episodes). The fusion head contributes essentially nothing. Treat the
Long +5 pp as weight/training variation, not a mechanism win.

## 8. Movement, Stop and trajectory quality: what actually costs SR

Δ = M1 − M0; positive failure Δ is worse. `nDTW(succ/fail)` is conditioned on
the episode outcome.

| Band | ΔSR | Δ Stop failures | Δ Movement/search failures | Δ nDTW | Δ nDTW(succ) | Δ nDTW(fail) | Δ SPL | Δ steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | −6 | 0 | **+6** | +0.011 | −0.009 | +0.059 | −0.032 | +12.0 |
| Short | −3 | +1 | +2 | +0.013 | −0.050 | +0.135 | −0.050 | −0.2 |
| Medium | −6 | +1 | **+5** | −0.006 | +0.051 | +0.081 | −0.098 | +27.7 |
| Long | +3 | −2 | −1 | +0.026 | +0.049 | −0.025 | +0.052 | +8.6 |

Trajectory-quality reading:

- nDTW is **not** the source of the loss. Success-conditioned nDTW is similar
  or better for M1 (Medium +0.051, Long +0.049); failure-conditioned nDTW is
  also higher for M1 (0.146 vs 0.087 overall), meaning M1's failed paths are
  closer to the reference, not more random.
- The overall nDTW point estimate favours M1 (+0.011) even though SR is lower;
  this is a composition effect (which episodes succeed), not better control.
- SPL is dominated by the success indicator here: successful episodes have
  SPL ≈ 1.0 in both models, failures have SPL = 0. The Medium SPL drop
  (−0.098) therefore echoes the SR drop rather than an independent path-length
  penalty.
- The real Medium cost is **27.7 more steps per episode** and +5 far timeouts;
  M1 searches/wanders longer and still fails to convert.

## 9. Should a minimal M1-Dir (2×2 pooling) controlled training be run?

**Verdict: not now.** The current head is provably direction-blind, and
replacing `AdaptiveAvgPool2d(1)` with `AdaptiveAvgPool2d((2,2))` is the
architecturally correct fix (+144 parameters). But the evidence does not make
a 2×2 pooling **controlled training** the highest-value next experiment.

Reasons:

1. **The residual is nearly inert and not directional.** It changes 0.37% of
   movement decisions (107/28,645), has 45 beneficial vs 43 harmful changes,
   and 106 of the 107 changes are FORWARD→turn (zero L→R/L→F, one R→F). It is
   a global "turn more" bias, not a left/right correction; its removal leaves
   Long SR at exactly 16/60 and raises overall SR 56→58. A better pooling layer
   has to overcome a very small learned residual, not a dominant directional
   path.
2. **The base already has direction.** When the goal is visible, the M1 base
   L−R logit AUC is 0.87–0.90, and M0's is 0.86–0.90. The missing piece is not
   "no directional signal"; it is that the agent often faces away from the
   goal near the end (M1 Medium: goal visible only 54.1% inside 20 m, mean
   |heading error| 62.5°, 59.7% missed turns).
3. **A static 2×2 map cannot fix the dominant failure.** The Medium loss is
   5 extra far timeouts plus near-goal orientation with the target out of
   view. A 2×2 pooling of the current frame still has no target to pool when
   the target is outside the FOV. Temporal memory/search (M2-direction work)
   targets this; M1-Dir does not.
4. **The Long gain is not fusion-driven and is not significant.** There is no
   band-specific signal that a directional residual would plausibly amplify.
5. **Single-seed evidence is insufficient per the experiment discipline.** The
   observed band effects are 3–10 pp on 60 episodes; paired CIs are ±7–17 pp.
   A 3 pp target cannot be established with one seed.

Cost if the team nevertheless wants the controlled training:

| Item | Estimate |
|---|---:|
| Architecture delta | 16→64 linear input: +144 parameters (2,675 → 2,819) |
| Training | 66–72 min for one seed (M1 took 66.1 min / 72 epochs; M0 71.9 min / 80 epochs) |
| Closed-loop eval 180 episodes at one threshold | ~4.5–6 min |
| Replay/off-policy diagnostics | ~10–15 min |
| **One seed total** | **~1.5 h** |
| 3-seed controlled run | ~4–4.5 h plus analysis |
| Cheaper alternative: freeze M1 base, train only a 2×2 head / linear turn decoder on cached replay states | ~10–20 min |

Recommendation: if the mechanism question must be closed, run the **head-only
offline probe first**. If a 2×2 head trained on cached states shows a
meaningful net beneficial action-change rate (>1–2% of near-goal movement
steps) against the heading oracle, then one controlled M1-Dir seed is
justified. If it does not, do not spend the 1.5–4.5 h: the binding problem is
arrival/orientation/search, not left/right pooling.

## 10. Compliance and artifacts

- No training was run. No checkpoint was modified (both sha256 values are in
  the provenance of the replay and fusion-off runs).
- Final Test 300 was not read; all data are Dev Val 180.
- No other MEC (Small, M2, Depth, BEV) was trained or evaluated.
- The only closed-loop re-run is the fusion-off control in §6; all other
  comparisons are exact state replays with additional forward passes.
- The fusion-off control is an **inference-time residual ablation inside the
  M1 checkpoint**, not an independent M0. M1 base/arrival weights are still
  jointly trained and differ from M0.

Artifacts:

- Replay: `runs/v83_spatial_stage2/stage2e/replay_20261019T015500Z/`
  (`replay_steps.csv`, `replay_check.json`, `fusion_pool_features.npz`,
  `provenance.json`).
- Analysis: `runs/v83_spatial_stage2/stage2e/analysis_20261019T023000Z/`
  (`analysis.json`, `tables.md`, `band_summary.csv`,
  `movement_stats.csv`, `movement_distance_stats.csv`, `near_goal_stats.csv`,
  `map_available_stats.csv`, `arrival_stats.csv`, `fusion_effects.csv`,
  `direction_stats.csv`, `mirror_stats.csv`).
- Fusion-off closed loop:
  `runs/v83_spatial_stage2/stage2e/no_fusion_20261019T020700Z/`
  (`M1-no-fusion@0.7_per_episode.csv`, `steps/`, `stops/`, `metrics.json`,
  `provenance.json`).
- Action-transition addendum:
  `runs/v83_spatial_stage2/stage2e/action_change_20261019T033000Z/`
  (`action_change_by_distance.csv`, `action_change_by_band.csv`,
  `action_transition_oracle.csv`, `summary.json`).
- Code:
  `v83_spatial_stage2/stage2e_movement_replay.py`,
  `v83_spatial_stage2/stage2e_analyze.py`,
  `v83_spatial_stage2/stage2e_action_change_analysis.py`,
  `v83_spatial_stage2/stage2e_m1_no_fusion_eval.py`,
  `v83_spatial_stage2/test_stage2e.py`.
- Tests: 11 CPU-only tests pass (`python -m unittest -v
  v83_spatial_stage2.test_stage2e`).
