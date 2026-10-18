# V8.3 Lightweight RGB-only vs RGB+Depth Arrival Probe — Design

- Status: **design executed on 2026-10-18.** See `PROBE_EXPERIMENT_REPORT.md` for results; P1 did not show a robust gain and Stop-gate development validation was not run.
- Branch: `experiment/v83-spatial-foundation`
- Frozen components: shared RGB encoder + Transformer-M Best-50. No five-MEC retraining, no checkpoint modification.
- Purpose: test whether the legal, visual-match-based Depth evidence adds incremental value to the existing RGB arrival signal under identical samples, seed, and training budget.

## 1. Feature dimensions

Frozen `Transformer-M` arrival representation (`decision["arrival_features"]`) is **104-D**:

- 8 RGB matching features: max match, top-10 mean, mean match, attention entropy, weighted x/y, best x/y;
- 64-D heatmap context feature;
- 32-D network context feature.

Shared RGB confidence features (identical in both probes):

- `visual_similarity_peak` (normalized to [0,1]);
- `visual_similarity_margin` (normalized to [0,1]).

Probe inputs:

| Probe | Feature vector | Dimension |
|---|---|---:|
| RGB-only | frozen arrival_features (104) + peak + margin (2) | **106** |
| RGB+Depth | 104 + 2 + 6 depth-only features | **112** |

Depth-only features (from the visual-match region only):

1. `matched_depth_mean_normalized`
2. `matched_depth_std_normalized`
3. `matched_depth_min_normalized`
4. `matched_depth_valid_fraction`
5. `matched_depth_inverse_proximity`
6. `scene_valid_depth_fraction`

The two shared RGB confidence features are computed identically for both probes. The 8-feature Depth evidence vector contains peak/margin; those two are not added twice.

Optional lean variant for interpretability (same samples/seed/budget):

- RGB-only: 8 matching features + peak + margin = **10-D**;
- RGB+Depth: 10 + 6 depth-only = **16-D**.

Primary comparison is 106 vs 112; the lean variant isolates the matching/depth effect.

## 2. Labels and class balance

Label: `arrival_label = 1` when simulator `distance_to_goal < stop_radius` (20 m), else 0. This is the existing dataset label; distance is used only to create the label, never as an input feature.

Exact counts from `dataset_stats.json` and shard masks:

| Split | Total samples | Positives | Negatives | Arrival-mask included | Included pos | Included neg | Included positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 26,693 | 900 | 25,793 | 7,185 | 900 | 6,285 | 12.53% |
| Val | 5,070 | 60 | 5,010 | 1,064 | 60 | 1,004 | 5.64% |
| Val_hard | 5,448 | 180 | 5,268 | 1,446 | 180 | 1,266 | 12.45% |
| Val+Val_hard | 10,518 | 240 | 10,278 | 2,510 | 240 | 2,270 | 9.56% |

- Train included: **900 positives / 6,285 negatives**.
- Use `pos_weight = 6285/900 ≈ 6.98` (or balanced batch sampling) for both probes. Report AUPRC and calibration, not only accuracy.
- Hard-arrival samples account for 600 of the 900 Train positives; they are essential for a meaningful positive rate.

## 3. Train/Validation/Test isolation

| Split | Episodes | Seeds | Use |
|---|---:|---|---|
| Train | 300 | 20000–20358 | Probe fitting only |
| Val | 60 | 40000–40069 | Early stopping / hyperparameter-free model selection |
| Val_hard | 60 | 70001–70068 | Additional validation; disjoint from Val |
| Test | 60 | 900000–900077 | Final probe comparison; never used for fitting or model selection |

- All splits are disjoint by **episode seed**; no timestep-level random split is used.
- The V8.3 60-episode test set is the same held-out set used in the earlier threshold evaluation; it remains untouched for probe training/selection.
- Normalization statistics (mean/std) are fit on Train only and applied to Val/Test.

## 4. Target-information leakage risk

Inputs allowed for the probe:

- Current RGB tokens, Goal reference tokens, spatial heatmap, network state (existing V8.2 inputs);
- Frozen Transformer-M `arrival_features` (104-D);
- RGB visual confidence features (peak/margin);
- Depth features from the **visual-match region** only.

Inputs explicitly excluded:

- `distance_to_goal_m`, `goal_visible`, Goal coordinates, UAV position, oracle masks, `episode_seeds`, `network_layout_seeds`, `sample_type`, `success`, `arrival_label`, `arrival_train_mask` (mask used for sample selection, not as an input).

The true Goal-region Depth coverage measured in the trajectory audit is **offline diagnostic only** and is **not** a probe feature. The probe uses the same legal `goal_conditioned_depth_evidence()` output that a deployed MEC could compute from RGB tokens + sensor Depth.

Hard-arrival state generation uses simulator Goal position to construct near-Goal poses (as the existing dataset generator does). This affects the state distribution, not the input feature vector; labels remain sensor-derived. This is documented as a dataset-construction dependency.

Additional safeguards:

- Fit scaler on Train only;
- Freeze encoder and Transformer-M; no fine-tuning;
- Use identical sample indices, labels, seed, architecture, optimizer, epochs, and batch size for both probes;
- Report probe metrics on Test without any threshold tuning on Test.

## 5. Data generation plan

### 5.1 Natural Train samples

- Replay each Train episode with its seed, band distance range, and oracle actions.
- Replay was verified exact: distance error 0.0 m, `goal_visible` mismatches 0 on shard_0000.
- Use the existing shard `current_tokens`, `goal_tokens`, `spatial_heatmaps`, `network_states`, `arrival_labels`, and `arrival_train_mask` for the included natural samples (5,385 samples).
- Render Depth at the corresponding states for the Depth probe.
- `render_depth()` benchmark: **0.46 ms/call**; 5,385 renders ≈ 2.5 s single-process.

### 5.2 Hard-arrival samples

- Generate 600 hard positives + 600 boundary negatives + 600 distractor negatives using the existing `_sample_pose_around_target` logic with a fixed RNG seed.
- Compute RGB tokens with the frozen encoder, heatmap, network state, arrival features, and Depth features.
- These are newly generated, deterministic samples; they follow the same distribution as the archived hard samples but are not byte-identical to them. Both probes use exactly the same generated samples, so the comparison remains controlled.
- Alternative: skip hard samples and use natural-only (300 positives / 5,085 negatives); this weakens positive balance and is not recommended.

### 5.3 Validation/Test

- Val and Val_hard: natural included samples + generated hard samples, same feature pipeline.
- V8.3 Test: replay the 60 held-out episodes.
  - Primary test: oracle natural trajectories (distribution matches Train natural states).
  - Secondary deployment-style test: 0.6-policy trajectories from the completed depth audit (13 successes, 14 episodes reach 20 m); 0.9-policy trajectories can be added to stress near-miss behavior.
- No Test episode is used for early stopping or feature scaling.

## 6. Probe architecture and training budget

Primary probe (both input variants):

- 2-layer MLP: input → 64 → 32 → 1, GELU, dropout 0.1;
- BCEWithLogitsLoss with `pos_weight ≈ 6.98`;
- AdamW, lr 1e-3, weight decay 1e-4;
- batch 256, max 50 epochs, early stopping patience 8 on Val AUPRC;
- fixed seed `20261018`; deterministic dataloaders;
- same architecture, optimizer, epochs, batch size, and early-stopping rule for both probes; only input dimension differs (106 vs 112).

Secondary probe (optional, same budget):

- Logistic regression (L2) on standardized features; same split/seed; useful to check whether the MLP result is driven by nonlinear interactions.

Metrics:

- AUROC, AUPRC, balanced accuracy, Brier score / ECE on Val and Test;
- Downstream stop-gate on the V8.3 Test trajectories:
  - choose the probe threshold on Val (e.g., maximize balanced accuracy or F1),
  - report SR, OSR, Premature-Stop, Timeout, SPL;
  - compare RGB-only vs RGB+Depth on the same episodes and threshold-selection rule.
- Report confidence intervals / paired episode-level differences; do not select on Test.

Estimated compute:

| Step | Volume | Estimate |
|---|---:|---:|
| Natural Train feature caching | 5,385 samples | ~2–3 min |
| Hard Train generation + feature caching | 1,800 samples | ~1–2 min |
| Val/Val_hard feature caching | 2,510 samples | ~1 min |
| V8.3 Test feature caching | ~1,000–3,000 step samples | ~1–2 min |
| Probe training (106/112-D MLP, 50 epochs) | 7,185 samples | <1 min |
| Total | | **~5–10 min conservative** |

No five-MEC retraining; no checkpoint modification. The probe cache is expected to be ~3.2 MB at float32.

## 7. Expected communication cost

The probe is an offline experiment; its 106/112-D vector is not the deployment payload. For a future deployed depth-aware arrival head:

- Additional Depth features: **6 scalars/step** (mean, std, min, matched-validity, proximity, scene-validity) = **12 bytes/step at float16**, or **24 bytes/step at float32**.
- Full 8-feature Depth evidence (including the shared peak/margin) = 16 bytes/step float16, 32 bytes/step float32.
- Raw Depth map 336×336 float16 = **225,792 bytes/step (220.5 KiB)**.
- Legacy depth-token payload: 576 tokens × 3 dims × 16 bits = **3,456 bytes/step**.

Ratios:

- 8 scalar features vs raw float16 Depth: ~**14,112×** smaller.
- 8 scalar features vs legacy depth tokens: ~**216×** smaller.
- Existing RGB token payload at float16: 576×384×2 = 442,368 bytes/step; 16 scalar bytes ≈ **0.0036%** overhead (0.0072% if RGB tokens are 8-bit).
- True Goal-region Depth is never transmitted; only the legal visual-match Depth features are.

## 8. Risks and expected interpretation

1. **Visual matching is the main risk.** The trajectory audit shows that at 20 m the true Goal Depth is 100% available and correct, but the visual-match region has non-Goal Depth in ~50% of episodes; top-1 patch never contains the Goal center at 20 m. The Depth probe may therefore show little gain even though the sensor signal is good.
2. **A negative probe result is informative:** it would say the current visual matching/arrival representation cannot exploit Depth, and the next step should be better target localization/matching, not full RGB-D training.
3. **Label imbalance** requires pos_weight/AUPRC; a balanced accuracy-only report would be misleading.
4. **Hard-sample generation** is deterministic but not byte-identical to the archived hard samples; both probes use the same generated samples. If exact archived hard samples are required, a separate replay of the dataset generator RNG is needed.
5. **Test positives are limited** (60 oracle episodes, 19 near-goal model-policy episodes); report confidence intervals and paired comparisons.

## 9. Decision rule after the probe

Proceed to a formal RGB vs RGB+Depth training ablation only if:

- RGB+Depth improves AUPRC and calibration on Val and Test consistently; and
- the downstream stop-gate improves SR/OSR without an unacceptable Timeout increase; and
- the gain is robust to seed variance (at least 3 seeds).

Otherwise:

- Improve visual matching / target localization first;
- Consider an **oracle target-region Depth upper-bound probe** (diagnostic only, not deployable) to see whether perfect localization would make Depth useful;
- For the 68.33% movement failures, prioritize BEV/geometry (`mec_geometry_v83`) rather than arrival-only Depth.

## 10. Approval gate

**No probe training has been started.** Please confirm:

1. the Train/Val/Test split above;
2. use of generated hard-arrival samples (600/600/600) rather than skipping them;
3. the ~5–10 min feature-caching + <1 min training budget;
4. the 106 vs 112 primary probe and 10 vs 16 lean variant.

After confirmation, the probe data can be cached and trained in a new independent output directory without touching the five MEC models or archived results.
