# V8.3 Stage-2 Preparation

Status: Stage-2a Transformer-M M0/M1 training and 180-episode closed-loop
evaluation complete; **M1 was not a robust improvement**. Stage-2b Never-Stop
and Arrival-only A0/A1 are complete: A1 improves A0 but remains far below the
original M0. Stage-2c B0/B1/B2 is complete: **the A0/A1 collapse was caused by
`pos_weight`; plain BCE restores M0-level SR (B1 29.44%, B2 28.89% vs B0
28.89%)**, but increasing far negatives from 3% to 10% does not improve
closed-loop behaviour. Diagnostics show the binding issue is per-decision
hazard: ~132 far decisions/episode at a ~0.35% on-policy far Stop rate gives
the observed 56.7% premature-Stop rate; B2's offline FPR gain shrinks from 25%
to 5% on-policy. Stage-2d completed the off-policy calibration controls and
the Stop-rule sweep; the M1@0.7 threshold migration was evaluated but is not a
robust improvement. Stage-2e (eval-only) shows that the M1
`AdaptiveAvgPool2d(1)` fusion head is exactly left/right invariant, its learned
residual changes only 0.37% of movement decisions, and the M1 Long gain
survives removing the residual entirely (16/60 in both cases). The Long +5 pp
is therefore stop-timing/training-weight variation, not spatial-fusion
direction correction. Stage-2f (eval-only) shows that observable confidence
features separate goal visibility only moderately (AUC ~0.64–0.76) and that a
rule-based Goal-Reacquisition overlay is harmful in a 36-episode closed-loop
smoke (SR 36.1%→19.4%, Timeout 13.9%→25.0%); the offline evidence is
insufficient for a reliable no-leakage recovery trigger. Stage-2g (probe-only)
shows that frozen token features predict localizability (AUROC 0.86) but
cannot gate near-target recovery: 90% of low-localizability flags are
far-visible states and near-invisible precision among flags is only 0.27%.
Train 600 contains only 47 near-target invisible states (versus 747 in the M0
closed-loop distribution), so the next step is a targeted near-target token
supplement, not controller integration. Stage-2g.1 (data supplement + probe)
completes that supplement: 2,000 matched near-target pairs and hard-state short
histories were added, and P1-2/P1-3 near-target-invisible history probes pass
the offline near-target gate (near-invisible precision 0.914/0.935, far-visible
false-trigger 0.07%/0.06%). The localizability probe still fails, and the
visible probe has a high far-invisible false-trigger; a small closed-loop smoke
with the P1-2 near-target-invisible trigger is the next step, not controller
integration. Stage-2g.2 was **executed after approval**: the P1-2 probe was
re-run with saved weights and reproduced the old Dev-Val predictions exactly
(max |Δp| = 0.0; Train-OOF threshold 0.13694876571430528). The 36-episode R1
smoke showed **no SR/SPL/nDTW improvement** (0W/0L/36T; nDTW +3.5e-7); only
2 triggers occurred, both far-invisible (166 m / 335 m), with 4/4 recovery
actions wrong, 0 re-visible and 0 enter-20 m. The recovery direction is
**STOPPED**; no expansion to Dev Val 180 and no controller integration. See
`V83_STAGE2G2_RECOVERY_SMOKE_REPORT.md` §7. No M1-Dir or other training was
started.
See `STAGE2A_REPORT.md`, `V83_STAGE2B_REPORT.md`,
`V83_STAGE2C_REPORT.md`, `V83_STAGE2D_M1_THRESHOLD_REPORT.md`,
`V83_STAGE2E_MOVEMENT_DIAGNOSIS.md`,
`V83_STAGE2F_RECOVERY_DIAGNOSIS.md`,
`V83_STAGE2F_MOVEMENT_RECOVERY_REPORT.md`,
`V83_STAGE2G_TARGET_PRESENCE_REPORT.md`,
`V83_STAGE2G1_NEAR_TARGET_REPORT.md`,
`V83_STAGE2G2_RECOVERY_SMOKE_REPORT.md` and
`STAGE2C_OFFPOLICY_PROTOCOL.md`.

## Data design

| Split | Episodes | Short / Medium / Long | Seeds | Notes |
|---|---:|---:|---|---|
| Train | 600 | 200 / 200 / 200 | 1,000,000+ | natural + hard arrival states |
| Dev Val | 180 | 60 / 60 / 60 | 2,000,000+ | natural + hard states; development only |
| Final Test | 300 | 100 / 100 / 100 | 3,000,000+ | natural only; sealed with SHA256/read-only |

Network layout seeds use `NETWORK.layout_seed_base + seed_start*17 + episode_id`,
so the new ranges (≈17.82M / 34.82M / 51.82M) are disjoint from all old splits
(old layouts ≈1.16M / 1.50M / 1.67M / 2.01M; V8.3 dev ≈16.12M).

The existing V8.2 data/training protocol is not modified. Equal band weights are
applied only inside the Stage-2 generation process.

## Commands

```bash
# smoke generation (6 episodes per split, separate seeds)
python v83_spatial_stage2/generate_stage2_datasets.py --smoke

# full generation (guards against overwriting)
python v83_spatial_stage2/generate_stage2_datasets.py \
  --output-root runs/v83_spatial_stage2/datasets

# audit: split counts/seeds, far-negative retention, M1 Long failures, head direction
python v83_spatial_stage2/audit_stage2.py \
  --data-root runs/v83_spatial_stage2/datasets \
  --output-dir runs/v83_spatial_stage2/prep_audit_<timestamp>

# Stage-2 spatial cache (Train + Dev Val only; Final Test not cached)
python v83_spatial_stage2/build_stage2_cache.py \
  --data-root runs/v83_spatial_stage2/datasets \
  --output-dir runs/v83_spatial_stage2/cache_<timestamp>

# single-batch M1 smoke on new Train
python v83_spatial_training/train_spatial_mec.py --models M1 --smoke \
  --cache-dir <stage2-cache> --train-dir <stage2-train> \
  --val-dir <stage2-dev-val> --val-hard-dir <stage2-dev-val> \
  --output-dir runs/v83_spatial_stage2/smoke_<timestamp>

# 2-episode closed-loop M1 smoke on new Dev Val (existing M1 checkpoint; no training)
python v83_spatial_training/eval_closedloop_spatial.py \
  --model-dir runs/v83_spatial_training/run_20261018T023830Z \
  --modes M1 --max-episodes 2 \
  --val-specs <prep-audit>/dev_val_180_spec.json \
  --output-dir runs/v83_spatial_stage2/closedloop_smoke_<timestamp>
```

## Stage-2c Arrival calibration controls

```bash
# B0/B1/B2: original head, 3% and 10% far negatives with plain BCE
python v83_spatial_stage2/stage2c_arrival_calibration.py \
  --output-dir runs/v83_spatial_stage2/stage2c/<new-stamp>

# Offline Dev Val + on-policy rollout calibration/stop diagnostics
python v83_spatial_stage2/stage2c_diagnostics.py \
  --output-dir runs/v83_spatial_stage2/stage2c/diag_<new-stamp>

# Recompute post-processing from saved rollout CSVs if needed
python v83_spatial_stage2/stage2c_finalize_diagnostics.py \
  --diagnostics-dir runs/v83_spatial_stage2/stage2c/diag_<new-stamp>

# Per-decision premature-Stop hazard decomposition
python v83_spatial_stage2/stage2c_risk_analysis.py \
  --diagnostics-dir runs/v83_spatial_stage2/stage2c/diag_<new-stamp> \
  --output-dir runs/v83_spatial_stage2/stage2c/risk_<new-stamp>

# Prepare off-policy negatives from Train 600 B0 rollouts (next stage)
python v83_spatial_stage2/stage2c_collect_rollouts.py \
  --dataset-root runs/v83_spatial_stage2/datasets/train_600 \
  --output-dir runs/v83_spatial_stage2/stage2c/offpolicy_train_rollouts_<stamp>
```

## Stage-2e Movement / fusion diagnostics (eval only)

```bash
# Exact-state replay of the saved M0@0.7 and M1@0.7 action sequences.
# Writes replay_steps.csv, replay_check.json and fusion_pool_features.npz.
python v83_spatial_stage2/stage2e_movement_replay.py \
  --output-dir runs/v83_spatial_stage2/stage2e/replay_<stamp>

# Band / failure / movement / fusion-residual / mirror / pooling analysis.
python v83_spatial_stage2/stage2e_analyze.py \
  --output-dir runs/v83_spatial_stage2/stage2e/analysis_<stamp> \
  --no-fusion-episodes <no-fusion>/M1-no-fusion@0.7_per_episode.csv

# Action-transition addendum: which Forward/Left/Right decisions the fusion
# residual changes, and the one-step heading/oracle benefit of each change.
python v83_spatial_stage2/stage2e_action_change_analysis.py \
  --output-dir runs/v83_spatial_stage2/stage2e/action_change_<stamp>

# Eval-only control: same M1 checkpoint, fusion residual forced to zero.
python v83_spatial_stage2/stage2e_m1_no_fusion_eval.py \
  --output-dir runs/v83_spatial_stage2/stage2e/no_fusion_<stamp>

# CPU-only tests (mirror invariance, 2x2 pooling sensitivity, label logic).
python -m unittest -v v83_spatial_stage2.test_stage2e
```

## Stage-2f Goal Reacquisition / Movement Recovery (eval only)

```bash
# Collect observable features on the saved M0@0.7 / M1@0.7 states.
python v83_spatial_stage2/stage2f_collect.py \
  --output-dir runs/v83_spatial_stage2/stage2f/collect_<stamp> --max-episodes 180

# Visible/invisible movement, confidence, entropy, wrong-run and trigger analysis.
python v83_spatial_stage2/stage2f_analyze.py \
  --features runs/v83_spatial_stage2/stage2f/collect_<stamp>/features.csv \
  --output-dir runs/v83_spatial_stage2/stage2f/analysis_<stamp>

# Even/odd episode threshold sweep for the observable recovery trigger.
python v83_spatial_stage2/stage2f_trigger_sweep.py \
  --features runs/v83_spatial_stage2/stage2f/collect_<stamp>/features.csv \
  --output-dir runs/v83_spatial_stage2/stage2f/sweep_<stamp>

# Closed-loop smoke: baseline M0@0.7, side-history recovery, scan-only recovery.
python v83_spatial_stage2/stage2f_recovery_eval.py \
  --output-dir runs/v83_spatial_stage2/stage2f/smoke_<stamp> \
  --episodes-per-band 12 --selection even \
  --policies baseline side_history scan_only \
  --low-top1 0.02 --low-entropy 0.98 --trigger-k 1 \
  --high-top1 0.20 --high-entropy 0.85 --consistency -1.0 \
  --max-recovery-steps 12

# CPU-only feature/policy tests.
python -m unittest -v v83_spatial_stage2.test_stage2f

# Supplementary analyses: trigger distance, transient vs sustained loss,
# turn-induced loss and the observable proximity proxy.
python v83_spatial_stage2/stage2f_supplement.py \
  --features runs/v83_spatial_stage2/stage2f/collect_<stamp>/features.csv \
  --smoke-dir runs/v83_spatial_stage2/stage2f/smoke_budget12_<stamp> \
  --output-dir runs/v83_spatial_stage2/stage2f/supplement_<stamp>

# CPU-only supplementary tests.
python -m unittest -v v83_spatial_stage2.test_stage2f_supplement
```

Consolidated metrics and resume-verification report:
`V83_STAGE2F_MOVEMENT_RECOVERY_REPORT.md`. The detailed predecessor is
`V83_STAGE2F_RECOVERY_DIAGNOSIS.md`.

## Stage-2g Target-Presence / Localization Probe (eval only)

```bash
# Label-only replay: projection geometry, occlusion-aware visible pixels and
# target_visible / target_localizable labels. Does not regenerate tokens.
python v83_spatial_stage2/stage2g_labels.py --split train \
  --output-dir runs/v83_spatial_stage2/stage2g/labels_train_<stamp>
python v83_spatial_stage2/stage2g_labels.py --split val \
  --output-dir runs/v83_spatial_stage2/stage2g/labels_val_<stamp>

# Observable feature matrices (current/goal tokens, heatmap, spatial maps).
python v83_spatial_stage2/stage2g_features.py --split train \
  --labels-dir runs/v83_spatial_stage2/stage2g/labels_train_<stamp> \
  --output-dir runs/v83_spatial_stage2/stage2g/features_<stamp>
python v83_spatial_stage2/stage2g_features.py --split val \
  --labels-dir runs/v83_spatial_stage2/stage2g/labels_val_<stamp> \
  --output-dir runs/v83_spatial_stage2/stage2g/features_<stamp>

# Linear/MLP probes, episode-level 5-fold OOF + Train->Dev-Val validation,
# and the Stage-2f M0 scalar closed-loop distribution check.
python v83_spatial_stage2/stage2g_probe.py --mode both --epochs 30 \
  --bootstrap 200 --output-dir runs/v83_spatial_stage2/stage2g/probe_<stamp>

# CPU-only label/feature/metric tests.
python -m unittest -v v83_spatial_stage2.test_stage2g
```

Report: `V83_STAGE2G_TARGET_PRESENCE_REPORT.md`. Key finding: the token probe
is a useful general localizability signal (localizable AUROC 0.856 CV / 0.857
Dev Val) but not a near-target recovery trigger; at the 50%-negative-recall
operating point 90% of flags are far-visible states and near-target invisible
precision among flags is 0.27%. Train 600 has 47 near-target invisible states
versus 747 in the M0 closed-loop distribution; a targeted token supplement is
required before any controller work.

## Stage-2g.1 Near-Target Data Supplement (eval/probe only)

```bash
# 2,000 matched near-target pairs (1,600 Train + 400 Dev Val)
python v83_spatial_stage2/stage2g1_supplement.py --split train --pairs 1600 \
  --output-dir runs/v83_spatial_stage2/stage2g1/supplement_train_<stamp>
python v83_spatial_stage2/stage2g1_supplement.py --split val --pairs 400 \
  --output-dir runs/v83_spatial_stage2/stage2g1/supplement_val_<stamp>

# Short history for the isolated hard states (compact features only)
python v83_spatial_stage2/stage2g1_hard_history.py --split train \
  --output-dir runs/v83_spatial_stage2/stage2g1/hard_history_train_<stamp>
python v83_spatial_stage2/stage2g1_hard_history.py --split val \
  --output-dir runs/v83_spatial_stage2/stage2g1/hard_history_val_<stamp>

# Combined near/far dataset, then P0 vs P1-1/2/3 probes + analysis
python v83_spatial_stage2/stage2g1_dataset.py --split train \
  --far-features .../features_train.npz \
  --supplement .../supplement_train.npz \
  --hard-history .../hard_history_train.npz \
  --data-dir runs/v83_spatial_stage2/datasets/train_600 \
  --output .../combined_train.npz
python v83_spatial_stage2/stage2g1_probe.py \
  --train .../combined_train.npz --val .../combined_val.npz \
  --output-dir runs/v83_spatial_stage2/stage2g1/probe_<stamp> \
  --epochs 30 --bootstrap 200
python v83_spatial_stage2/stage2g1_analyze.py \
  --probe-dir .../probe_<stamp> --combined-val .../combined_val.npz \
  --output .../probe_<stamp>/analysis.json

# CPU-only tests
python -m unittest -v v83_spatial_stage2.test_stage2g1
```

Report: `V83_STAGE2G1_NEAR_TARGET_REPORT.md`. Key finding: two/three frames of
visual history turn the near-target-invisible probe from a failing current-only
detector (precision 0.204) into a passing one (P1-2 precision 0.914, far-visible
false-trigger 0.07%; P1-3 0.935/0.06%). The localizability probe still fails,
and the visibility probe's far-invisible false-trigger is 43.6-45.7%. The
follow-up 36-episode closed-loop smoke is complete: exact probe reproduction,
but no SR/SPL/nDTW improvement and 2/2 far-invisible triggers; the recovery
direction is stopped (`V83_STAGE2G2_RECOVERY_SMOKE_REPORT.md` §7).

## Planned Stage-2 experiments (paused for approval)

- Backbones: Transformer-M and Small only.
- Within each backbone, M0/M1 use identical seed/data order/LR/batch/80-epoch/patience-10; across backbones use the original reasonable settings:
  - Transformer-M: lr 6e-5, batch 48.
  - Small: lr 1e-4, batch 64.
- First compare M0/M1 on Train 600 and Dev Val 180 (natural/hard + closed-loop).
- Then, if warranted, controlled Arrival far-negative ablations A0=3% (old), A1=10%, A2=25%, A3=distance-stratified 10%, using the same tokens and training budget; no old protocol changes.
- Then a minimal M1-Dir head variant (2×2 pooled spatial summary) if the baseline confirms direction loss matters.
- Final Test 300 is sealed and never used for selection/tuning.

See `STAGE2_PREP_REPORT.md` for diagnostics, data statistics, and time/disk estimates.

## Stage-2l Fixed-City navigation RL (2026-10-21)

Stage-2l replaces the per-episode city regeneration with a fixed-city /
task-seed split and trains standard DQN and PPO on three deployment
configurations (A/B/C). The five Stage-2j T5 MECs stay frozen; the state is
Pure MEC v2 (384-D + 20-D mask) and the reward is JointR-aligned with no
shaping.

- Protocol: `STAGE2L_FIXED_CITY_PROTOCOL.md`
- Setup report (environment fixedness, task distribution, smoke): 
  `STAGE2L_FIXED_CITY_SETUP.md`
- Progress report: `STAGE2L_FIXED_CITY_REPORT.md`
- Build: `python v83_spatial_stage2/stage2l_build.py`
- Tests: `python v83_spatial_stage2/test_stage2l_fixed_city.py`
- Smoke: `python v83_spatial_stage2/stage2l_smoke.py`
- Supervised training: `python v83_spatial_stage2/stage2l_supervisor.py`
- Status: `python v83_spatial_stage2/stage2l_status.py`
- Report/figures: `python v83_spatial_stage2/stage2l_report.py`
- Dev evaluation: `python v83_spatial_stage2/stage2l_eval.py --config A --split dev`

All run artifacts live under `runs/v83_spatial_stage2/stage2l_fixed_city/`
(git-ignored). Final Test stays sealed until the fixed-environment baseline is
established and the protocol is frozen.
