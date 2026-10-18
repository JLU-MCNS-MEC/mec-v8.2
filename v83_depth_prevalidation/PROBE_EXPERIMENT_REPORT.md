# V8.3 Lightweight RGB / RGB+Depth / Shuffled-Depth Probe — Experiment Report

- Date: 2026-10-18
- Branch: `experiment/v83-spatial-foundation`
- Code HEAD at run time: `fc79dba` (Depth fixes); probe pipeline committed together with this report
- Frozen: shared RGB encoder + Transformer-M Best-50
- Seed: `20261018` (single seed; no multiple-seed claims)
- Command:

```bash
python v83_depth_prevalidation/depth_probe_experiment.py --max-epochs 50 --patience 8
```

- Results directory: `runs/v83_depth_probe/20261018T020614605676Z/`
- Full stdout: `runs/v83_depth_probe/full_20261018T020613Z.log`
- Total wall time: **41.5 s** (data build + feature caching + P0/P1/P2 training), well under the ~10 min budget.

## 1. Data alignment and sample counts

Archived natural samples are replayed with `ToyUAVEnv`; RGB features use the archived tokens, Depth is rendered at the verified replay state. Hard-arrival samples are regenerated deterministically from the same Train/Val_hard episode seeds with `token_data._collect_hard_state`; RGB and Depth for those samples come from the same pose, so they cannot be misaligned.

| Split | Episodes | Natural samples | Generated hard | Total | Positives | Alignment |
|---|---:|---:|---:|---:|---:|---|
| Train | 300 | 5,385 | 1,800 (600/600/600) | 7,185 | 900 | distance error 0; arrival mismatch 0; goal_visible mismatch 0; episode_id mismatch 0; no step gaps |
| Natural Val | 60 | 1,064 | 0 | 1,064 | 60 | same: all zero mismatches |
| Hard Val | 60 | 1,086 | 360 (120/120/120) | 1,446 | 180 | same: all zero mismatches |

- Episode/step/arrival-label/replay checks: **0 mismatches** across all 420 natural episodes (300 + 60 + 60) and 7,535 natural samples.
- Hard-state generation: 1,800/1,800 Train and 360/360 Hard Val requested samples generated.
- The archived hard samples themselves were not used because their exact poses are not stored in the token shards; the regenerated hard set is deterministic and identical for P0/P1/P2. This is documented as a controlled-probe limitation.

## 2. Probe inputs and training

| Probe | Input | Dimension |
|---|---|---:|
| P0 | frozen `arrival_features` (104) + RGB peak/margin (2) | **106** |
| P1 | P0 + 6 visual-match Depth features | **112** |
| P2 | P1 with the 6 Depth columns shuffled across samples (fixed permutation) | **112** |

- Same samples, labels, seed, MLP body (106/112→64→32→1, GELU, dropout 0.1), AdamW, batch 256, max 50 epochs, early stopping on **Natural Val AUPRC** (patience 8).
- `pos_weight = 6,285 / 900 ≈ 6.98`; no class resampling.
- P2 shuffles Depth within each split with the fixed seed; RGB features and labels remain sample-aligned.
- No Goal position, horizontal distance, Oracle mask, `sample_type`, seed, or `goal_visible` is an input; Depth is the legal visual-match evidence only.
- True Goal-region Depth from the trajectory audit was **not** used in any probe feature.

## 3. Results

Natural Val (1,064 samples, 60 positives, 5.64% positive):

| Probe | Best epoch | AUROC | AUPRC | Brier | ECE | Recall | Precision | Threshold | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 RGB-only | 50* | 0.98287 | 0.82659 | **0.06796** | **0.10821** | 0.8333 | 0.6944 | 0.86 | 0.7576 |
| P1 RGB+Depth | 19 | **0.98597** | **0.83699** | 0.07026 | 0.12414 | 0.7833 | 0.7460 | 0.93 | **0.7642** |
| P2 Shuffled Depth | 42 | 0.98298 | 0.81621 | 0.07006 | 0.11143 | 0.8833 | 0.6235 | 0.80 | 0.7310 |

Hard Val (1,446 samples, 180 positives, 12.45% positive):

| Probe | Best epoch | AUROC | AUPRC | Brier | ECE | Recall | Precision | Threshold | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 RGB-only | 50* | 0.97043 | 0.83964 | **0.06605** | **0.08465** | 0.7333 | 0.7811 | 0.86 | 0.7564 |
| P1 RGB+Depth | 19 | **0.97179** | **0.84068** | 0.07632 | 0.11320 | 0.6389 | 0.8042 | 0.93 | 0.7121 |
| P2 Shuffled Depth | 42 | 0.96980 | 0.84024 | 0.06932 | 0.09008 | 0.7722 | 0.7596 | 0.80 | **0.7658** |

\* P0 reached the 50-epoch cap without early stopping; its comparison is therefore conservative.

### 3.1 Deltas (P1 vs P0 / P2)

| Split | ΔAUPRC P1−P0 | ΔAUPRC P1−P2 | ΔAUROC P1−P0 | ΔBrier P1−P0 | ΔECE P1−P0 | ΔBrier P1−P2 | ΔECE P1−P2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Natural Val | +0.0104 | +0.0208 | +0.0031 | +0.0023 (worse) | +0.0159 (worse) | +0.0002 (worse) | +0.0127 (worse) |
| Hard Val | +0.0010 | +0.0004 | +0.0014 | +0.0103 (worse) | +0.0286 (worse) | +0.0070 (worse) | +0.0231 (worse) |

### 3.2 Common-threshold Recall/Precision

Because the per-probe max-F1 thresholds differ (P0 0.86, P1 0.93, P2 0.80), selected-threshold precision/recall are not directly comparable. Common-threshold results:

| Split | Threshold | P0 P/R | P1 P/R | P2 P/R |
|---|---:|---:|---:|---:|
| Natural Val | 0.5 | 0.343 / 0.983 | 0.358 / 0.983 | 0.326 / 0.967 |
| Natural Val | 0.8 | 0.593 / 0.850 | 0.607 / 0.900 | 0.624 / 0.883 |
| Natural Val | 0.9 | 0.689 / 0.700 | 0.694 / 0.833 | 0.694 / 0.717 |
| Hard Val | 0.5 | 0.574 / 0.900 | 0.557 / 0.950 | 0.549 / 0.906 |
| Hard Val | 0.8 | 0.774 / 0.800 | 0.713 / 0.800 | 0.760 / 0.772 |
| Hard Val | 0.9 | 0.835 / 0.644 | 0.799 / 0.683 | 0.877 / 0.633 |

At common thresholds P1 does not consistently dominate P0 or P2; its advantage is mostly a small Natural Val AUPRC/AUROC shift accompanied by worse calibration.

## 4. Error analysis

1. **P1 is not robustly better than P0.** Natural Val AUPRC +0.0104 and AUROC +0.0031 are small; Hard Val AUPRC +0.0010 is negligible. Brier and ECE are **worse** for P1 on both val sets, and selected-threshold Hard-Val F1 is lower (0.712 vs 0.756 for P0).
2. **P2 (shuffled Depth) essentially matches P1 on Hard Val** (AUPRC 0.8402 vs 0.8407; AUROC 0.9698 vs 0.9718) and has better Brier/ECE. This control indicates the extra 6 Depth dimensions do not provide a reliable aligned signal; the small Natural-Val P1 edge may come from extra capacity/early stopping rather than Depth content.
3. **Convergence asymmetry.** P0 hit the 50-epoch cap (best epoch 50, still improving); P1 early-stopped at 27 with best epoch 19; P2 hit the cap with best epoch 42. P0 may still have headroom, which would further reduce P1's small edge. The single fixed seed and single run preclude a variance claim.
4. **High offline AUROC/AUPRC vs low closed-loop SR.** All probes reach AUROC ≈0.97–0.99 and AUPRC ≈0.82–0.84, but the earlier closed-loop evaluation had best SR 21.67% and movement ceiling 31.67%. Offline arrival classification is not the closed-loop bottleneck; stop timing and movement/grounding dominate.
5. **Calibration matters.** P1's threshold shifts to 0.93 (vs 0.86 for P0, 0.80 for P2); its higher precision is bought by lower recall. This is not a clear improvement in the operating regime relevant to Stop-gate development.
6. **Data limitation.** Natural samples are exact archived-token + replayed-Depth aligned; hard samples are regenerated deterministically. The Hard Val metrics therefore include regenerated hard states (same for all probes) rather than the exact archived hard tokens. The natural alignment audit is exact (zero mismatches).

## 5. Decision: Stop-gate development validation

The agreed condition was to run Stop-gate development validation only if the probe shows a clear gain. **That condition is not met:**

- P1's Natural Val AUPRC gain is +0.010 over P0 and +0.021 over P2, but
- Hard Val AUPRC is essentially tied (+0.001 / +0.0004),
- P1 Brier/ECE are worse on both val sets,
- P2 (shuffled Depth) is essentially tied with P1 on Hard Val and better calibrated,
- P0 hit the 50-epoch cap and may have additional headroom.

Therefore **no Stop-gate development validation was run**, and no full RGB vs RGB+D training is recommended on this evidence. If a Stop-gate is still desired, it should be treated as a separate exploratory experiment, not as a confirmation of a probe gain.

## 6. Next steps

1. **Improve target localization before adding Depth.** The trajectory audit showed that at 20 m the true Goal Depth is 100% available and correct, but the visual-match region has non-Goal Depth in ~50% of episodes and top-1 patch never contains the Goal center. The Depth features are limited by visual matching, not the sensor.
2. **Run an oracle target-region Depth upper-bound probe** (offline diagnostic only, never an input to a deployable MEC) to determine whether perfect target localization would make Depth useful. If the oracle upper bound is also marginal, do not invest in RGB-D training.
3. **For arrival/stop timing**, consider a distance/visibility-aware or hysteresis Stop gate rather than adding Depth to the current matching features.
4. **For movement failures**, prioritize the BEV/geometry direction (`mec_geometry_v83`), since 41/60 episodes never reach 20 m even with forced movement; arrival-only Depth cannot fix that.
5. Do not freeze a Stop threshold from this probe; the earlier threshold sweep already showed 0.6 vs 0.5 is a one-episode difference.
6. If a larger probe is desired later, use multiple seeds and a pre-registered stopping rule; the current run is a single-seed preliminary result.

## 7. Artifacts

All under `runs/v83_depth_probe/20261018T020614605676Z/` (Git-ignored):

- `cache/train.npz`, `cache/val_natural.npz`, `cache/val_hard.npz`
- `P0/`, `P1/`, `P2/` — best weights (`.pt`), `curves.csv`, `curve.png`, `metrics.json`, `scaler.json`
- `comparison.csv`
- `common_threshold_metrics.log`
- `summary.json`
- `alignment_checks.json`
- Full stdout: `runs/v83_depth_probe/full_20261018T020613Z.log`

No probe weights or caches are committed to Git. No existing checkpoints, datasets, or historical results were modified.
