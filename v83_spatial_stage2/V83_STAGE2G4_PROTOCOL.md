# V8.3 Stage-2g.4 — Long-Horizon Target-Loss Event Probe Protocol

Status: frozen before probe training (2026-10-19).

Branch: `experiment/v83-spatial-foundation`
Base commit: `9787a0f`

## 1. Scope

- Reuse `runs/v83_spatial_stage2/stage2g3/real_train_20261019T132500Z/` and
  `runs/v83_spatial_stage2/stage2g3/real_val_20261019T132000Z/`.
- Do **not** recollect Stage-2g.3 data, modify old checkpoints, or overwrite
  old runs.
- Do **not** train the full MEC, do **not** connect a Recovery controller,
  do **not** read Final Test 300.
- Only lightweight event probes are trained. All probe weights are saved.

## 2. Event label

For K ∈ {4, 8, 16}:

```text
recent_near_loss_K(t) =
    distance_m(t) < 100
    AND target_visible(t) == 0
    AND exists j in [1, K] with target_localizable(t-j) == 1
```

`distance_m`, `target_visible` and `target_localizable` are used only to
construct the offline label. They are never model inputs.

Negative subsets for evaluation:

- `sustained_near_inv`: near-target-invisible but not an event (last
  localizable > K steps ago, or never);
- `near_visible`: distance < 100 m and target visible;
- `far_visible`: distance ≥ 100 m and target visible;
- `far_invisible`: distance ≥ 100 m and target invisible.

## 3. Event coverage (existing data; no new collection)

Common probe/evaluation set: `step >= 16` and `action != STOP`.

| Split | K | States | Positives | Positive rate | Positive episodes | Bands (short/med/long) |
|---|---:|---:|---:|---:|---:|---:|
| Train | 4 | 71,576 | 170 | 0.238% | 41 | 78 / 50 / 42 |
| Train | 8 | 71,576 | 320 | 0.447% | 42 | 146 / 96 / 78 |
| Train | 16 | 71,576 | 609 | 0.851% | 42 | 275 / 184 / 150 |
| Dev Val | 4 | 23,665 | 82 | 0.347% | 19 | 59 / 19 / 4 |
| Dev Val | 8 | 23,665 | 163 | 0.689% | 21 | 112 / 43 / 8 |
| Dev Val | 16 | 23,665 | 328 | 1.386% | 22 | 220 / 92 / 16 |

The minimum coverage criterion for not collecting new data was: at least
10 Dev-Val positive episodes and 50 positives for every K. All K pass
(19/21/22 episodes; 82/163/328 positives), so **no new M0@0.7 episodes are
collected**.

Most event positives never become localizable again: 152/171 (K=4),
302/321 (K=8), 591/610 (K=16) on Train; 75/82, 156/163, 321/328 on Dev Val.
Only 7 Dev-Val events re-localize within 10 steps. This is a property of the
existing trajectories, not a model result.

## 4. Variants and inputs

| Variant | Input |
|---|---|
| P0 | current 2,858 observable features |
| P1-2 | current + compact history t-1..t-2 |
| P1-8 | current + compact history t-1..t-8 |
| P1-16 | current + compact history t-1..t-16 |
| P1-16+mem | P1-16 + label-free observable last-seen memory |

Each compact history frame = 74 compact observable features + 4 action
one-hot + dt = 79 dimensions. History order is always t-1 first.

The last-seen memory is a deterministic causal summary of six observable
confidence columns (`sim_max_all_top5`, `heat_0/1/2_top1`, `base_max`,
`coh_max`), normalized with Train-only mean/std and thresholded at the 70th
percentile of the Train composite distribution. It adds: confidence, age
since last high-confidence frame, has-high-confidence flag, last confidence,
count/max/mean/argmax-age in the last 16 frames, last action one-hot, and the
last high-confidence compact feature. It never reads `target_visible` or
`target_localizable`. A pre-training audit showed that this hand-crafted
confidence alone is weak for event-vs-sustained discrimination; the memory
variant is therefore both a requested feature and a negative-control ablation.

## 5. Training and threshold rules

- 5-fold **episode-level** OOF on Train; external model on all Train.
- Dev Val only for evaluation; no threshold or hyperparameter selection on it.
- MLP: 128 hidden, dropout 0.2, 30 epochs, Adam (1e-3, weight decay 1e-4),
  batch 8,192, seed 0 (+fold for OOF, +1000 for external).
- Standardization mean/std from the training fold; clip [-20, 20].
- `pos_weight = n_neg / n_pos`.
- Threshold selection on Train OOF only, with negative-subset constraints:
  `near_visible <= 2%`, `far_visible <= 2%`, `far_invisible <= 2%`,
  `sustained_near_inv <= 5%`.
  - primary: constrained maximum F1 (`f1_joint`);
  - also report unconstrained F1, and precision-0.5 / recall-0.5 operating
    points if they exist on Train OOF.
- All CIs resample **episodes**, not steps.

## 6. Gate (frozen)

The primary long-horizon claim is evaluated at **K=16**. Gate pass requires
all of the following for the primary condition (`P1-16` or `P1-16+mem`,
whichever is better on Train OOF AUPRC):

1. A constrained Train-OOF threshold with **Precision ≥ 0.50** exists
   (otherwise no usable operating point is possible).
2. On independent Dev Val at the primary Train-OOF threshold:
   **Precision ≥ 0.50 and Recall ≥ 0.50**.
3. Absolute false-trigger budget: total FP ≤ TP (equivalently Precision ≥ 0.5)
   and **mean false triggers per Dev-Val episode ≤ 1.0**.
4. Event-vs-sustained-near-inv AUROC ≥ **0.70**, and event-vs-far-invisible
   AUROC ≥ **0.75** on Dev Val.
5. Longer history helps: ΔAUROC of P1-16 (or P1-16+mem) over P0 ≥ **+0.03**
   and over P1-2 ≥ **+0.02**, with paired episode-bootstrap 95% CI lower
   bound > 0 for at least one of AUROC/AUPRC.
6. Leave-one-extreme-episode-out: excluding the largest Dev-Val event-positive
   episode, checks 2–3 still hold and AUROC drops by < 0.05.

K=4/K=8 results are reported for comparison, but they cannot substitute for
a K=16 pass: the claim under test is that long horizon solves the problem.

## 7. Decision rule

- **If any gate check fails:** stop the Recovery probe line. Do not add more
  MLP layers or blindly add history frames. The next direction should be
  explicit target spatial belief / coarse navigation localization.
- **If all checks pass:** report only a controlled MEC integration ablation
  plan. Do not train the full MEC and do not connect the controller in this
  round.

Final Test 300 remains sealed. Old data, checkpoints and reports are not
modified.
