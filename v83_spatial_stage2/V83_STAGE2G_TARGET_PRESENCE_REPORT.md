# V8.3 Stage-2g — Target-Presence / Localization Probe Report

- Date: 2026-10-19
- Branch: `experiment/v83-spatial-foundation`
- Scope: **eval/probe-only** target-presence and localizability diagnosis. No
  MEC training, no checkpoint modification, no M1-Dir/M2/Depth/MEC-BEV
  training, no controller integration.
- Data: Train 600 (55,565 samples), Dev Val 180 (17,234 samples), and the
  existing Stage-2f Dev-Val M0 closed-loop features (26,628 states).
  **Final Test 300 was not read or evaluated.**
- Code: `stage2g_labels.py`, `stage2g_features.py`, `stage2g_probe.py`,
  `test_stage2g.py`. Reports are in `runs/v83_spatial_stage2/stage2g/`.

## 结论摘要 (Chinese TL;DR)

1. **数据审计通过，标签可复现。** 现有 Train 600 / Dev Val 180 的 Token
   shards 与 spatial cache 按 `(episode_id, step_id)` 精确对齐；shard 缺少
   goal 投影几何与 occlusion-aware 可见像素，本轮用**纯标签重放**生成（不重建
   token、不重跑 Stage-2f），逐状态与 shard 的 `goal_visible` / distance /
   sample_type **0 mismatch**，最大距离误差 1.37e-4 m。Train 标签生成 62 s，
   Dev Val 20 s。
2. **标签定义（预注册）：** `target_visible = occlusion-aware visible
   pixels >= 1`；`target_localizable = target_visible AND
   goal_half_size_px >= 10 AND visible_pixels >= 16`。10 px 阈值来自 Dev Val
   Stage-2f 的标定：half-size 8–10 px 时 coherent peak error <=42 px 占
   58.2%，10–15 px 时占 82.8%。Occlusion 很少（Train 110 / Val 110 个状态）。
3. **关键数据缺口：** Train 600 的 oracle 轨迹 98.2% 时间目标可见，真正的
   **近目标不可见（<100 m 且不可见）只有 47 个状态**（Dev Val 33 个）；
   而 Stage-2f M0 闭环分布有 **747** 个。`target_visible` 正类率 98.2%/97.8%，
   AUPRC 被 base rate 主导；`target_localizable` 正类率 72.4%/71.8%，相对可用。
4. **Token probe 整体能预测 localizability：** MLP-balanced 在 5-fold
   episode CV 上 `target_localizable` AUROC **0.856**、AUPRC **0.935**、
   AP-neg **0.715**、Brier **0.141**、ECE **0.090**；在独立 Dev Val 180 上
   AUROC **0.857**、AUPRC **0.929**、AP-neg **0.732**、Brier **0.139**、
   ECE **0.088**。`target_visible` 的 AUROC 为 **0.901 / 0.841**（CV / Dev
   Val）。Token 特征明显强于 Stage-2f 手工标量特征（后者 localizable AUROC
   仅 **0.689**）。
5. **但它不能作为近目标 recovery 触发器。** 在 localizable 的
   negative-recall=0.5 工作点（阈值 0.403），CV 共 flag 10,549 个状态，其中
   **89.9% 是远距离可见状态**；近目标不可见只占 **0.27%**（28/10,549），
   近目标不可见的 flag 率 59.6%，但相对 base rate 只有约 3.2x enrichment。
   独立 Dev Val 更差：90.6% 的 flag 是 far-visible，近目标不可见 precision
   仍为 **0.27%**。
6. **近目标丢失 vs 远距离不可见无法区分。** 近目标不可见 vs 远距离不可见的
   separation AUC（越高越能分开；0.5=随机）为：visible head **0.323 / 0.384**
   （CV / Dev Val），localizable head **0.443 / 0.309**。方向相反：模型给
   近目标不可见更高的“可见/可定位”分数。Stage-2f M0 闭环标量 probe 同样失败：
   near-invisible vs far-visible separation **0.568 / 0.529**，near-invisible
   vs far-invisible **0.371 / 0.405**。
7. **结论：** 该 probe 可作为“当前目标是否可定位”的**通用诊断信号**，
   **不能**作为“近目标定位失败、需要恢复搜索”的闭环触发器。不要接入控制器，
   不要据此启动 MEC 训练。现有数据不支持这一 regime 的可学习性；问题不是标签
   无法生成，而是近目标不可见样本太少，且与远距离不可定位状态混在一起。
8. **下一步唯一实验：** 定向补充近目标不可见 Token 数据并重训 probe（2,000 个
   near-invisible + 2,000 个 matched near-visible，冻结 encoder，只保存
   tokens/heatmaps/labels；约 1.8 GB、2–4 min；重训约 2 min）。验收门槛：
   held-out episode 上近目标不可见 precision among flags >= 0.5，且
   far-visible flag rate <= 5%；否则停止这条手写 recovery 路线。

## 1. Data audit

### 1.1 What exists and what aligns

| Artifact | Size | Content | Alignment |
|---|---:|---|---|
| `datasets/train_600` | 24.7 GB, 557 shards | 55,565 samples, 600 episodes (200 Short / 200 Medium / 200 Long); 51,965 natural + 3,600 hard | keys `(episode_id, step_id)` |
| `datasets/dev_val_180` | 7.7 GB, 173 shards | 17,234 samples, 180 episodes (60/60/60); 16,154 natural + 1,080 hard | same |
| `cache_20261018T043434Z/train.npz` | 257 MB | 55,565 × 24×24 baseline/coherent/temporal/innovation maps + valid | exact `(episode_id, step_id)` match |
| `cache_20261018T043434Z/val.npz` | 80 MB | 17,234 × 24×24 maps | exact match |
| Stage-2f `features.csv` | 93 MB | 55,414 M0/M1 closed-loop replay states (26,628 M0) | keys `(episode_index, step)`; not the same states as the shards after early steps |

Shard fields: `tokens` (N×576×384), `goal_tokens` (N×3×384),
`spatial_heatmaps` (N×3×8×8), `network_states` (N×5), `labels` (action),
`arrival_labels`, `move_valid`, `arrival_train_mask`, `distance_to_goal_m`,
`sample_type`, `goal_visible`, `episode_ids`, `step_ids`, `episode_seeds`,
`network_layout_seeds`, `trajectory_band_ids`.

**Missing fields for the requested labels:** target projection geometry
(`goal_px`, `goal_py`, `goal_half_size_px`, `goal_slant_m`), occlusion-aware
visible pixel count, and true heading error. `goal_visible` in the shards is
the geometric projection flag (`_project(...) is not None`), not a pixel-level
visibility label.

The Stage-2f `features.csv` does contain projection geometry and heading for
Dev Val 180 M0/M1 rollout states, but it does not contain raw tokens, and its
states are the closed-loop M0/M1 trajectories — not the reference/oracle
trajectory states stored in the shards. The exact `(episode, step)`
intersection with `dev_val_180` is 12,925 pairs, but 2,565 have different
`goal_visible` and the distance can differ by hundreds of metres after the
trajectories diverge. **It is therefore not used for token/label alignment.**

### 1.2 Label-only replay (minimal supplement, no token regeneration)

`stage2g_labels.py` replays the deterministic oracle trajectory for each
accepted episode, then reproduces the six hard states per episode with the same
`arrival_rng` stream as `token_data.py` (including the natural-step
`arrival_train_mask` draws). It computes:

- `goal_px`, `goal_py`, `goal_half_size_px`, `goal_slant_m` from
  `env.goal_projection()`;
- an occlusion-aware visible pixel count: draw the target's base shape, then
  draw every strictly nearer landmark's base shape on top, following the
  simulator's far-to-near render order; count target pixels not covered;
- the two labels below.

Verification against the existing shards:

| Split | Replayed rows | Missing / extra keys | `goal_visible` mismatch | distance mismatch >1e-3 | sample_type mismatch | max distance error |
|---|---:|---:|---:|---:|---:|---:|
| Train 600 | 55,565 | 0 / 0 | 0 | 0 | 0 | 1.37e-4 m |
| Dev Val 180 | 17,234 | 0 / 0 | 0 | 0 | 0 | 1.37e-4 m |

Label definition (pre-registered, not tuned on probe results):

```
target_visible     = (occlusion-aware visible_pixels >= 1)
target_localizable = target_visible AND goal_half_size_px >= 10.0
                     AND visible_pixels >= 16
```

The 10 px half-size threshold is calibrated from the existing Dev-Val Stage-2f
diagnostics on visible M0 move states:

| Half-size bin (px) | n | Median coherent peak err (px) | Share <=42 px | Share <=21 px |
|---|---:|---:|---:|---:|
| 2–3 | 21 | 66.2 | 28.6% | 28.6% |
| 3–4 | 190 | 65.4 | 21.6% | 20.0% |
| 4–5 | 486 | 78.2 | 28.2% | 19.3% |
| 5–6 | 1,049 | 116.2 | 19.8% | 17.4% |
| 6–8 | 2,214 | 82.1 | 33.3% | 30.2% |
| 8–10 | 1,457 | 12.9 | 58.2% | 54.4% |
| **10–15** | **4,011** | **9.9** | **82.8%** | **81.7%** |
| 15–20 | 4,002 | 5.9 | 81.8% | 80.4% |
| 20–25 | 1,003 | 12.0 | 84.2% | 77.4% |
| 25–40 | 2,822 | 22.2 | 87.6% | 46.0% |

The knee is at 10 px: below 8–10 px only 58.2% of visible states localize within
42 px, while 10–15 px reaches 82.8%. The 16-pixel minimum prevents a one-pixel
sliver from being labeled localizable. Occlusion is rare: 110 Train and 110
Dev-Val states are geometrically visible but have zero visible pixels (0.2% and
0.64%). Haze, brightness, blur and sensor noise are **not** modelled; the label
is a simulator-geometry occlusion-aware pixel count, not a sensor-pixel
segmentation.

### 1.3 Label distribution and the critical data gap

| Split | Samples | Episodes | Natural / Hard | `target_visible` | `target_localizable` | Near-invisible (<100 m, invisible) | Far-visible (>=100 m, visible) | Far-not-localizable (>=100 m, localizable=0) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train 600 (oracle) | 55,565 | 600 | 51,965 / 3,600 | **98.2%** | **72.4%** | **47** | 34,626 | 15,257 |
| Dev Val 180 (oracle) | 17,234 | 180 | 16,154 / 1,080 | **97.8%** | **71.8%** | **33** | 10,783 | 4,824 |
| Stage-2f M0 (closed loop) | 26,628 | 180 | natural rollout | **65.2%** | **44.8%** | **747** | 13,143 | 13,961 |

The core problem is visible immediately:

- The reference/oracle trajectories keep the target visible 98% of the time.
  The hard distractor negatives are mostly **far** invisible (median distance
  646.8 m), not near-target loss.
- In Train 600 only **47 / 55,565 (0.085%)** states are near-target invisible;
  Dev Val has 33. A model trained on this split has almost no examples of the
  failure regime that Stage-2f showed is the recovery opportunity.
- The closed-loop M0 distribution has 747 near-target invisible states, but the
  token shards do not contain those states. The only available M0-state data
  are the Stage-2f scalar features (no raw tokens).

## 2. Probe design

### 2.1 Inputs

**Token probe** (Train 600 -> Dev Val 180), 2,858 observable dimensions:

| Group | Dimensions | Content |
|---|---:|---|
| Current RGB token summary | 1,152 | mean / std / max over 576 tokens × 384 |
| Goal token summary | 768 | mean / std of 3 × 384 goal tokens |
| Current–Goal similarity | 594 | per-scale max/top5/mean/std (12), per-patch max-over-scales (576), top5/mean/std/argmax x,y (6) |
| Goal–Current heatmap | 207 | per-channel entropy/top1/top5 mass/centroid (15) + flattened 3×8×8 (192) |
| Baseline/coherent spatial maps | 138 | 8×8 adaptive-pooled maps (128) + max/mean/std/centroid (10) |

These are all observable: current tokens, goal reference tokens, the
Goal–Current heatmap and parameter-free spatial maps. No true goal position,
distance, visibility, heading, oracle action, sample type, episode id or band
enters the feature matrix. `FORBIDDEN_FEATURE_TERMS` is checked when the
features are written; `test_stage2g.py` checks the names.

**Scalar probe** (Stage-2f M0 closed-loop distribution), 77 observable
dimensions selected by an explicit whitelist from `features.csv`: similarity
statistics, heatmap confidence/entropy/centroid statistics, baseline/coherent
map statistics, and temporal/innovation features. True labels
(`goal_visible`, `goal_px/py`, `goal_half_size_px`, `distance_m`,
`heading_error_deg`, `oracle_move_action`, `coherent/baseline_peak_error_px`)
and model outputs (`m0_arrival_probability`, `m0_move_*`, `logged_*`,
`owner_*`) are excluded. The scalar localizable label is
`goal_visible AND goal_half_size_px >= 10`; `visible_pixels` is unavailable
for these states, and the shard audit showed occlusion changes only ~0.2–0.6%
of labels.

### 2.2 Models and splits

- Linear logistic probe and a one-hidden-layer MLP (128 units, ReLU, dropout
  0.2), 30 epochs, Adam (lr 1e-3, weight decay 1e-4), batch 8,192.
- Class-weighted BCE for the "balanced" variants; the class prior is undone by
  the exact prior-correction formula `odds_corrected = odds / pos_weight`
  before Brier/ECE/confusion metrics. The plain linear model has no weighting
  and no prior correction.
- **5-fold episode-level OOF** on Train 600 (600 episodes; adjacent frames of
  an episode always share a fold) and Dev Val 180 (180 episodes, scalar mode).
- **External validation:** train on all Train 600 episodes, evaluate on all
  Dev Val 180 oracle episodes.
- The Stage-2f scalar probe uses the same 5-fold episode split on the 180
  Dev-Val episodes (M0 states only).

### 2.3 Compute and artifacts

| Item | Result |
|---|---|
| Token feature build | 69 s Train (55,565×2,858), 15 s Dev Val (17,234×2,858) |
| Feature files | 393 MB Train, 122 MB Dev Val |
| Label files | 6.7 MB Train, 2.1 MB Dev Val |
| Probe run (both modes, 30 epochs, 200 bootstrap) | 80 s on RTX 5090; prediction/results artifacts 7.0 MB |
| Peak GPU memory | well below 32 GB (feature tensor + one fold/model at a time) |
| Disk headroom | 37 GB free on `/` at run time |

## 3. Results

Primary model for the tables: **MLP-balanced** (best overall ranking and
calibration); the linear baselines are kept in §3.1. All metrics are on the
offline labels; the token mode is on oracle states, the scalar mode on the M0
closed-loop states.

### 3.1 Overall model comparison

#### Token probe, 5-fold episode OOF (Train 600)

| Head | Model | AUROC | AUPRC | AUPRC-neg | Brier | ECE | P@0.5 | R@0.5 | Spec@0.5 | BAcc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| visible | linear-plain | 0.887 | 0.996 | 0.397 | 0.206 | 0.436 | 0.996 | 0.847 | 0.794 | 0.820 |
| visible | linear-balanced | 0.904 | 0.997 | 0.395 | 0.014 | 0.012 | 0.987 | 0.996 | 0.286 | 0.641 |
| visible | **MLP-balanced** | **0.901** | **0.997** | **0.413** | **0.014** | **0.009** | 0.988 | 0.995 | 0.337 | 0.666 |
| localizable | linear-plain | 0.842 | 0.922 | 0.709 | 0.167 | 0.171 | 0.897 | 0.753 | 0.773 | 0.763 |
| localizable | linear-balanced | 0.841 | 0.926 | 0.694 | 0.137 | 0.029 | 0.847 | 0.894 | 0.576 | 0.735 |
| localizable | **MLP-balanced** | **0.856** | **0.935** | **0.715** | **0.141** | **0.090** | 0.845 | 0.913 | 0.559 | 0.736 |

#### Token probe, external Dev Val 180 (train on all Train 600)

| Head | Model | AUROC | AUPRC | AUPRC-neg | Brier | ECE | P@0.5 | R@0.5 | Spec@0.5 | BAcc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| visible | linear-plain | 0.798 | 0.991 | 0.367 | 0.202 | 0.428 | 0.991 | 0.868 | 0.635 | 0.751 |
| visible | linear-balanced | 0.821 | 0.993 | 0.375 | 0.017 | 0.007 | 0.984 | 0.997 | 0.254 | 0.626 |
| visible | **MLP-balanced** | **0.841** | **0.995** | **0.367** | **0.017** | **0.011** | 0.984 | 0.997 | 0.275 | 0.636 |
| localizable | linear-plain | 0.842 | 0.921 | 0.706 | 0.169 | 0.170 | 0.894 | 0.752 | 0.774 | 0.763 |
| localizable | linear-balanced | 0.846 | 0.928 | 0.699 | 0.137 | 0.023 | 0.837 | 0.904 | 0.551 | 0.728 |
| localizable | **MLP-balanced** | **0.857** | **0.929** | **0.732** | **0.139** | **0.088** | 0.843 | 0.917 | 0.566 | 0.742 |

#### Scalar probe, 5-fold episode OOF (Stage-2f M0, 26,628 states)

| Head | Model | AUROC | AUPRC | AUPRC-neg | Brier | ECE | P@0.5 | R@0.5 | Spec@0.5 | BAcc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| visible | linear-plain | 0.594 | 0.676 | 0.553 | 0.232 | 0.162 | 0.711 | 0.685 | 0.480 | 0.582 |
| visible | linear-balanced | 0.575 | 0.666 | 0.537 | 0.215 | 0.099 | 0.699 | 0.953 | 0.233 | 0.593 |
| visible | **MLP-balanced** | **0.661** | **0.738** | **0.614** | **0.204** | **0.091** | 0.732 | 0.900 | 0.383 | 0.641 |
| localizable | linear-plain | 0.637 | 0.517 | 0.728 | 0.239 | 0.072 | 0.563 | 0.558 | 0.649 | 0.603 |
| localizable | linear-balanced | 0.636 | 0.517 | 0.725 | 0.240 | 0.099 | 0.554 | 0.449 | 0.707 | 0.578 |
| localizable | **MLP-balanced** | **0.689** | **0.603** | **0.757** | **0.229** | **0.086** | 0.602 | 0.595 | 0.681 | 0.638 |

Reading the table:

- The token probe is much stronger than the hand-crafted scalar probe:
  localizable AUROC **0.856 / 0.857** vs **0.689**; visible AUROC **0.901 /
  0.841** vs **0.661**.
- The visible label is 98% positive, so AUPRC (~0.997) is mostly the base rate.
  The useful rare-negative metric is **AUPRC-neg** (0.413 token CV, 0.367 token
  external) and specificity/balanced accuracy, not AUPRC.
- Plain BCE on the 98/2 label is badly calibrated (ECE 0.436) and conservative
  at 0.5. Class weighting + prior correction gives ECE 0.009 and high positive
  recall, at the cost of low specificity at the default 0.5 threshold. This is
  why the operating point must be chosen for the negative class, not 0.5.
- The scalar localizable head has the best negative-class AP (0.757) because
  localizable is more balanced there (44.8% positive), but its overall AUROC
  is only 0.689 and its CIs are wide.

### 3.2 Overall / Short / Medium / Long

MLP-balanced, 5-fold episode OOF or external validation as marked. P/R/Spec/
BAcc are at threshold 0.5.

| Mode / split | Head | Group | n | AUROC | AUPRC | Brier | ECE | P@0.5 | R@0.5 | Spec@0.5 | BAcc@0.5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Token CV | visible | Overall | 55,565 | 0.901 | 0.997 | 0.014 | 0.009 | 0.988 | 0.995 | 0.337 | 0.666 |
| Token CV | visible | Short | 11,896 | 0.929 | 0.997 | 0.018 | 0.015 | 0.982 | 0.996 | 0.362 | 0.679 |
| Token CV | visible | Medium | 17,997 | 0.891 | 0.997 | 0.014 | 0.010 | 0.987 | 0.996 | 0.307 | 0.652 |
| Token CV | visible | Long | 25,672 | 0.898 | 0.998 | 0.011 | 0.007 | 0.992 | 0.995 | 0.344 | 0.669 |
| Token CV | localizable | Overall | 55,565 | 0.856 | 0.935 | 0.141 | 0.090 | 0.845 | 0.913 | 0.559 | 0.736 |
| Token CV | localizable | Short | 11,896 | 0.868 | 0.985 | 0.070 | 0.063 | 0.962 | 0.936 | 0.553 | 0.744 |
| Token CV | localizable | Medium | 17,997 | 0.838 | 0.947 | 0.134 | 0.087 | 0.872 | 0.903 | 0.513 | 0.708 |
| Token CV | localizable | Long | 25,672 | 0.853 | 0.887 | 0.178 | 0.135 | 0.753 | 0.904 | 0.576 | 0.740 |
| Token Dev Val | visible | Overall | 17,234 | 0.841 | 0.995 | 0.017 | 0.011 | 0.984 | 0.997 | 0.275 | 0.636 |
| Token Dev Val | visible | Short | 3,534 | 0.837 | 0.991 | 0.032 | 0.028 | 0.968 | 0.996 | 0.241 | 0.619 |
| Token Dev Val | visible | Medium | 5,653 | 0.860 | 0.996 | 0.013 | 0.008 | 0.987 | 0.997 | 0.301 | 0.649 |
| Token Dev Val | visible | Long | 8,047 | 0.872 | 0.997 | 0.013 | 0.008 | 0.988 | 0.997 | 0.292 | 0.645 |
| Token Dev Val | localizable | Overall | 17,234 | 0.857 | 0.929 | 0.139 | 0.088 | 0.843 | 0.917 | 0.566 | 0.742 |
| Token Dev Val | localizable | Short | 3,534 | 0.826 | 0.977 | 0.092 | 0.084 | 0.940 | 0.927 | 0.442 | 0.685 |
| Token Dev Val | localizable | Medium | 5,653 | 0.854 | 0.934 | 0.118 | 0.073 | 0.873 | 0.937 | 0.533 | 0.735 |
| Token Dev Val | localizable | Long | 8,047 | 0.847 | 0.888 | 0.175 | 0.118 | 0.764 | 0.892 | 0.592 | 0.742 |
| Scalar CV | visible | Overall | 26,628 | 0.661 | 0.738 | 0.204 | 0.091 | 0.732 | 0.900 | 0.383 | 0.641 |
| Scalar CV | visible | Short | 8,462 | 0.668 | 0.710 | 0.199 | 0.122 | 0.740 | 0.888 | 0.450 | 0.669 |
| Scalar CV | visible | Medium | 9,236 | 0.622 | 0.652 | 0.249 | 0.144 | 0.639 | 0.933 | 0.288 | 0.611 |
| Scalar CV | visible | Long | 8,930 | 0.752 | 0.894 | 0.162 | 0.074 | 0.825 | 0.882 | 0.455 | 0.668 |
| Scalar CV | localizable | Overall | 26,628 | 0.689 | 0.603 | 0.229 | 0.086 | 0.602 | 0.595 | 0.681 | 0.638 |
| Scalar CV | localizable | Short | 8,462 | 0.731 | 0.747 | 0.231 | 0.199 | 0.777 | 0.582 | 0.732 | 0.657 |
| Scalar CV | localizable | Medium | 9,236 | 0.674 | 0.586 | 0.236 | 0.111 | 0.563 | 0.642 | 0.623 | 0.633 |
| Scalar CV | localizable | Long | 8,930 | 0.666 | 0.497 | 0.220 | 0.144 | 0.451 | 0.552 | 0.704 | 0.628 |

### 3.3 Episode-level 95% confidence intervals (MLP-balanced)

Bootstrap resamples **episodes**, not steps; 200 resamples. The CI is over the
pooled states of the resampled episodes.

| Mode / split | Head | AUROC [95% CI] | AUPRC [95% CI] | Brier [95% CI] |
|---|---|---|---|---|
| Token CV | visible | 0.901 [0.884, 0.919] | 0.997 [0.9967, 0.9980] | 0.014 [0.0122, 0.0152] |
| Token CV | localizable | 0.856 [0.840, 0.869] | 0.935 [0.923, 0.943] | 0.141 [0.133, 0.150] |
| Token Dev Val | visible | 0.841 [0.783, 0.913] | 0.995 [0.991, 0.998] | 0.017 [0.0119, 0.0227] |
| Token Dev Val | localizable | 0.857 [0.823, 0.882] | 0.929 [0.904, 0.948] | 0.139 [0.1225, 0.1566] |
| Scalar CV | visible | 0.661 [0.569, 0.737] | 0.738 [0.630, 0.824] | 0.204 [0.172, 0.245] |
| Scalar CV | localizable | 0.689 [0.607, 0.763] | 0.603 [0.477, 0.717] | 0.229 [0.195, 0.267] |

The token head is stable across episodes; the scalar head's CIs are wide and
include weak-AUC values. The external Dev-Val token CIs are wider than the CV
CIs because there are only 180 episodes, but the point estimates are
consistent.

### 3.4 Key subsets (MLP-balanced)

Definitions: near-invisible = true distance <100 m and target invisible;
far-visible = distance >=100 m and target visible; far-not-localizable =
distance >=100 m and target not localizable; far-invisible = distance >=100 m
and target invisible; near-localizable = distance <100 m and target
localizable. `p>=0.5` is the fraction of the subset predicted positive by the
corresponding head.

#### Token CV

| Head | Subset | n | Positive rate | Mean p | p>=0.5 | AUROC |
|---|---|---:|---:|---:|---:|---:|
| visible | near-invisible | 47 | 0.000 | 0.874 | 0.915 | — |
| visible | far-visible | 34,626 | 1.000 | 0.989 | 0.994 | — |
| visible | far-not-localizable | 15,257 | 0.938 | 0.959 | 0.967 | 0.852 |
| visible | far-invisible | 949 | 0.000 | 0.636 | 0.650 | — |
| visible | near-localizable | 19,933 | 1.000 | 0.995 | 0.998 | — |
| localizable | near-invisible | 47 | 0.000 | 0.370 | 0.319 | — |
| localizable | far-visible | 34,626 | 0.587 | 0.667 | 0.687 | 0.788 |
| localizable | far-not-localizable | 15,257 | 0.000 | 0.448 | 0.442 | — |
| localizable | far-invisible | 949 | 0.000 | 0.337 | 0.320 | — |
| localizable | near-localizable | 19,933 | 1.000 | 0.951 | 0.972 | — |

#### Token Dev Val 180 (external)

| Head | Subset | n | Positive rate | Mean p | p>=0.5 | AUROC |
|---|---|---:|---:|---:|---:|---:|
| visible | near-invisible | 33 | 0.000 | 0.839 | 0.909 | — |
| visible | far-visible | 10,783 | 1.000 | 0.990 | 0.996 | — |
| visible | far-not-localizable | 4,824 | 0.928 | 0.964 | 0.973 | 0.785 |
| visible | far-invisible | 345 | 0.000 | 0.702 | 0.707 | — |
| visible | near-localizable | 6,070 | 1.000 | 0.994 | 0.998 | — |
| localizable | near-invisible | 33 | 0.000 | 0.654 | 0.697 | — |
| localizable | far-visible | 10,783 | 0.585 | 0.667 | 0.683 | 0.791 |
| localizable | far-not-localizable | 4,824 | 0.000 | 0.447 | 0.432 | — |
| localizable | far-invisible | 345 | 0.000 | 0.375 | 0.354 | — |
| localizable | near-localizable | 6,070 | 1.000 | 0.960 | 0.980 | — |

#### Scalar CV (Stage-2f M0 closed loop)

| Head | Subset | n | Positive rate | Mean p | p>=0.5 | AUROC |
|---|---|---:|---:|---:|---:|---:|
| visible | near-invisible | 747 | 0.000 | 0.686 | 0.780 | — |
| visible | far-visible | 13,143 | 1.000 | 0.752 | 0.913 | — |
| visible | far-not-localizable | 13,961 | 0.389 | 0.620 | 0.718 | 0.673 |
| visible | far-invisible | 8,526 | 0.000 | 0.545 | 0.603 | — |
| visible | near-localizable | 4,212 | 1.000 | 0.693 | 0.858 | — |
| localizable | near-invisible | 747 | 0.000 | 0.427 | 0.380 | — |
| localizable | far-visible | 13,143 | 0.586 | 0.448 | 0.435 | 0.654 |
| localizable | far-not-localizable | 13,961 | 0.000 | 0.359 | 0.315 | — |
| localizable | far-invisible | 8,526 | 0.000 | 0.350 | 0.312 | — |
| localizable | near-localizable | 4,212 | 1.000 | 0.589 | 0.741 | — |
### 3.5 Regime separation

`far-visible vs near-invisible` asks: can the score separate the
"far target visible but hard to match, keep navigating" regime (label 1) from
the "near target invisible, needs recovery" regime (label 0)? `far-invisible
vs near-invisible` asks whether the probe can tell a far invisible target from
a near invisible target; 0.5 means indistinguishable, <0.5 means the near
state gets the higher score.

| Mode / split | Head | Far-visible vs near-invisible | Far-invisible vs near-invisible |
|---|---|---:|---:|
| Token CV | visible | 0.816 | **0.323** |
| Token Dev Val | visible | 0.738 | **0.384** |
| Token CV | localizable | 0.722 | **0.443** |
| Token Dev Val | localizable | 0.533 | **0.309** |
| Scalar CV | visible | 0.568 | **0.371** |
| Scalar CV | localizable | 0.529 | **0.405** |

The far-visible vs near-invisible rank separation is moderate for the token
head, but the absolute near-invisible probability is high (0.874 visible /
0.370 localizable CV; 0.839 / 0.654 Dev Val). More importantly, the
near-invisible vs far-invisible separation is **below chance in every mode**:
the probe gives the near-target lost state a higher "visible/localizable"
score than the far invisible state. It has no usable proximity signal.

### 3.6 Flag composition at the operating point (the decisive test)

A recovery trigger would flag a state when the localizable probability is low
(`p <= threshold`). The threshold is chosen for negative-class recall 0.5 on
the OOF predictions (the same values used in §3.1). The table asks what the
flag actually contains. `Near-inv flag rate` = flagged near-invisible states /
all near-invisible states. `Near-inv precision among flags` = flagged
near-invisible states / all flagged states (not precision against the entire
non-localizable class).

| Mode / split | Head | Threshold | Total flags | Near-invisible flags | Near-inv precision among flags | Near-inv flag rate | Far-visible flags | Share of flags far-visible | Far-invisible flags |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Token CV | localizable | 0.403 | 10,549 | 28 | **0.27%** | 59.6% | 9,481 | **89.9%** | 608 |
| Token Dev Val | localizable | 0.403 | 3,323 | 9 | **0.27%** | 27.3% | 3,011 | **90.6%** | 211 |
| Token CV | visible | 0.844 | 1,106 | 10 | **0.90%** | 21.3% | 480 | 43.4% | 489 |
| Token Dev Val | visible | 0.844 | 343 | 9 | **2.62%** | 27.3% | 139 | 40.5% | 148 |
| Scalar CV | localizable | 0.345 | 10,119 | 317 | **3.14%** | 42.4% | 4,648 | 45.9% | 4,495 |
| Scalar CV | visible | 0.640 | 8,933 | 254 | **2.84%** | 34.0% | 3,013 | 33.7% | 4,406 |

Base rates for comparison:

- Token CV near-invisible: 47 / 55,565 = **0.085%**; at the localizable
  operating point flags contain 0.27% near-invisible (≈3.2× enrichment, but
  absolute precision is useless).
- Token Dev Val near-invisible: 33 / 17,234 = **0.191%**; flags contain 0.27%
  (≈1.4× enrichment).
- Scalar CV near-invisible: 747 / 26,628 = **2.80%**; flags contain 3.14%
  (≈1.1× enrichment).

The trigger is dominated by far states. On both token and scalar modes, a
recovery gate built from this flag would fire mostly at long range — exactly
the Stage-2f failure mode that reduced SR from 36.1% to 19.4%.

### 3.7 Confusion matrices at different thresholds (MLP-balanced, OOF)

Token mode, Train 600 OOF:

| Head | Threshold | TP | FP | FN | TN | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| visible | 0.1 | 54,502 | 810 | 67 | 186 | 0.985 | 0.999 | 0.187 |
| visible | 0.2 | 54,457 | 754 | 112 | 242 | 0.986 | 0.998 | 0.243 |
| visible | 0.3 | 54,414 | 715 | 155 | 281 | 0.987 | 0.997 | 0.282 |
| visible | 0.5 | 54,318 | 660 | 251 | 336 | 0.988 | 0.995 | 0.337 |
| visible | 0.7 | 54,195 | 583 | 374 | 413 | 0.989 | 0.993 | 0.415 |
| localizable | 0.1 | 39,389 | 10,805 | 862 | 4,509 | 0.785 | 0.979 | 0.294 |
| localizable | 0.2 | 38,758 | 9,410 | 1,493 | 5,904 | 0.805 | 0.963 | 0.386 |
| localizable | 0.3 | 38,158 | 8,367 | 2,093 | 6,947 | 0.820 | 0.948 | 0.454 |
| localizable | 0.5 | 36,729 | 6,761 | 3,522 | 8,553 | 0.845 | 0.913 | 0.559 |
| localizable | 0.7 | 34,687 | 5,198 | 5,564 | 10,116 | 0.870 | 0.862 | 0.661 |

Scalar mode, Stage-2f M0 OOF:

| Head | Threshold | TP | FP | FN | TN | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| visible | 0.1 | 17,331 | 7,872 | 24 | 1,401 | 0.688 | 0.999 | 0.151 |
| visible | 0.2 | 17,243 | 7,462 | 112 | 1,811 | 0.698 | 0.994 | 0.195 |
| visible | 0.3 | 16,952 | 7,030 | 403 | 2,243 | 0.707 | 0.977 | 0.242 |
| visible | 0.5 | 15,611 | 5,725 | 1,744 | 3,548 | 0.732 | 0.900 | 0.383 |
| visible | 0.7 | 11,498 | 3,951 | 5,857 | 5,322 | 0.744 | 0.663 | 0.574 |
| localizable | 0.1 | 11,509 | 11,610 | 411 | 3,098 | 0.498 | 0.966 | 0.211 |
| localizable | 0.2 | 10,804 | 9,705 | 1,116 | 5,003 | 0.527 | 0.906 | 0.340 |
| localizable | 0.3 | 9,784 | 8,006 | 2,136 | 6,702 | 0.550 | 0.821 | 0.456 |
| localizable | 0.5 | 7,095 | 4,686 | 4,825 | 10,022 | 0.602 | 0.595 | 0.681 |
| localizable | 0.7 | 2,973 | 1,930 | 8,947 | 12,778 | 0.606 | 0.249 | 0.869 |

The tables show the expected trade-off: on the rare visible-negative class,
lowering the threshold from 0.7 to 0.1 increases negative recall
(specificity 0.415 → 0.187) but never reaches a useful near-invisible
precision. On localizable, 0.5 is a reasonable positive-class operating point
(recall 0.913, specificity 0.559), but it is not the near-target trigger.

## 4. Error analysis: why the probe cannot gate near-target recovery

1. **The training distribution has almost no near-target invisible states.**
   47 Train / 33 Dev Val vs 747 in the M0 closed-loop distribution. The
   visible head sees a 98% positive class and learns "visible" as the default;
   the near-invisible mean p is 0.874 (CV) / 0.839 (Dev Val). It flags only
   8.5% / 9.1% of near-invisible states at 0.5.
2. **The localizable head learns size/distance, not proximity-to-a-lost-target.**
   It gives high p to near-localizable states (0.951 CV / 0.960 external), but
   near-invisible p is 0.370 CV / 0.654 external. The external value is close
   to far-visible p (0.667), so the apparent CV separation (0.722) does not
   generalize (0.533).
3. **The flags are dominated by far states.** At the localizable operating
   point, 89.9% / 90.6% of flags are far-visible. Near-invisible precision
   among flags is 0.27% (CV and external). A trigger built on this signal
   would fire at long range, not near the target.
4. **Near vs far invisible is below chance.** The probe gives near-invisible
   states a higher visibility/localizability score than far-invisible states
   (separation AUC 0.323/0.384 visible; 0.443/0.309 localizable). The missing
   variable is proximity, not visual confidence.
5. **The closed-loop scalar probe confirms the hand-crafted observables are
   weaker.** On the actual M0 distribution, near-invisible p is 0.686
   (visible) / 0.427 (localizable), far-visible p is 0.752 / 0.448, and the
   far-visible vs near-invisible separation is only 0.568 / 0.529. The scalar
   probe cannot compensate for the token data gap.
6. **Overall AUC is not the right decision metric.** The token localizable
   AUROC 0.856 is real and useful for a general "is the target localizable"
   score, but it is driven by the easy far-visible vs far-not-localizable
   contrast. It does not answer the control question: "is this a near-target
   temporary loss that warrants search, or a far target that warrants
   continued navigation?" The flag-composition table is the decision-relevant
   metric, and it fails.

**Answer to the requested key judgment:** The model **cannot** reliably
distinguish "far target hard to match but should continue navigating" from
"near target localization failure requiring recovery search" with the
available data. The visible head can rank far-visible above near-invisible,
but its absolute near-invisible probability is high; the localizable head's
operating point flags mostly far-visible states; and near-invisible vs
far-invisible is at or below chance. It is a general localizability signal,
not a recovery trigger.

## 5. Data gap and minimal supplement

**The labels are not the blocker.** They are computable and were verified
against every shard state. The blocker is that the reference/oracle dataset
does not contain enough near-target invisible states: 47 Train / 33 Dev Val
versus 747 in the M0 closed-loop distribution. The hard distractor negatives
are far (median 646.8 m), not near.

**Minimal supplement (not run this round):**

1. Sample near-target poses at 5–100 m from the goal, with yaw offsets that
   put the target outside the camera FOV (or behind an occluder), and accept
   only states with `visible_pixels == 0`; 2,000 states.
2. Sample 2,000 matched controls at the same positions/yaws but facing the
   target (`target_visible=1`, `target_localizable` by the same rule).
3. Encode current RGB with the frozen shared encoder; compute the
   Goal–Current heatmap; save `tokens`, `goal_tokens`, `spatial_heatmaps`,
   labels and IDs only. Do **not** regenerate the full Train 600 shards or
   modify the existing ones.
4. Estimated cost: 4,000 states × 576×384 float16 ≈ **1.8 GB** for current
   tokens, plus heatmaps/labels; **2–4 min** on the RTX 5090 (the Stage-2f
   collector processed ~11 ms/state including forward passes), plus a few
   minutes for environment sampling. Disk headroom is 37 GB. Retraining the
   token probe takes ~2 min.
5. Acceptance gate before any controller work: on held-out episodes,
   near-invisible precision among flags **>= 0.50** and far-visible flag rate
   **<= 5%** at a 50% negative-recall operating point. If the supplemented
   model cannot reach this, stop the near-target recovery line.

If this supplement is not approved, use the probe only as a general
localizability diagnostic (not a recovery gate), and move to map/coarse
navigation or M2 temporal memory instead.

## 6. Decision

- **Do not connect this probe to the controller.** It does not provide a
  no-leakage near-target trigger with usable precision.
- **Do not start MEC training** based on these results. No checkpoint was
  modified; no M1-Dir/M2/Depth/MEC-BEV training was started.
- **Keep the token localizable probe as a diagnostic.** It is a strong general
  localizability signal (AUROC 0.856 CV / 0.857 Dev Val, Brier 0.141/0.139,
  ECE 0.090/0.088) and a useful baseline for future work.
- **The next experiment is the targeted near-target supplement** described in
  §5. It is the only way to test whether the near-target failure regime is
  learnable; the current data cannot answer it.
- Final Test 300 remains sealed and was not read.

## 7. Limitations

- The token probe is trained and evaluated on **oracle/reference** states
  (98% visible), not the M0 closed-loop distribution. The scalar probe covers
  the M0 distribution but uses 77 hand-crafted features, not raw tokens.
- Near-target invisible is only 47 Train / 33 Dev Val in token mode and 747 in
  scalar mode. All near-target estimates have very few positives; the
  token Dev-Val near-invisible separation (0.533) is based on 33 states and is
  not stable.
- The labels are simulator-geometry, occlusion-aware pixel counts; haze,
  brightness, blur and sensor noise are not modelled. Occlusion changes only
  0.2–0.6% of shard visibility labels.
- The 10 px localizability threshold is a design choice calibrated on Dev-Val
  Stage-2f; different thresholds would change the positive rate and the
  operating points.
- 5-fold episode-level OOF with one seed; the scalar CIs are wide (180
  episodes). No multi-seed replication.
- The probe is offline; no closed-loop policy was evaluated this round.
  The Stage-2f negative closed-loop result remains the reference for any
  recovery overlay.

## 8. Artifacts, commands, tests

Labels (generated this round; shard-verified):

```
runs/v83_spatial_stage2/stage2g/labels_train_20261019T115000Z/labels_train.csv
runs/v83_spatial_stage2/stage2g/labels_train_20261019T115000Z/labels_train_summary.json
runs/v83_spatial_stage2/stage2g/labels_val_20261019T115200Z/labels_val.csv
runs/v83_spatial_stage2/stage2g/labels_val_20261019T115200Z/labels_val_summary.json
```

Observable features:

```
runs/v83_spatial_stage2/stage2g/features_20261019T120000Z/features_train.npz  (393 MB)
runs/v83_spatial_stage2/stage2g/features_20261019T120000Z/features_val.npz    (122 MB)
```

Probe results (all metrics, OOF and validation predictions):

```
runs/v83_spatial_stage2/stage2g/probe_20261019T123000Z/
  probe_results.json, tables.md,
  token_oof_*.npz, token_val_*.npz, scalar_oof_*.npz
```

Code:

```
v83_spatial_stage2/stage2g_labels.py
v83_spatial_stage2/stage2g_features.py
v83_spatial_stage2/stage2g_probe.py
v83_spatial_stage2/test_stage2g.py
```

Commands:

```bash
# Labels (label-only replay; no token regeneration)
python v83_spatial_stage2/stage2g_labels.py --split train \
  --output-dir runs/v83_spatial_stage2/stage2g/labels_train_<stamp>
python v83_spatial_stage2/stage2g_labels.py --split val \
  --output-dir runs/v83_spatial_stage2/stage2g/labels_val_<stamp>

# Observable features
python v83_spatial_stage2/stage2g_features.py --split train \
  --labels-dir runs/.../labels_train_<stamp> --output-dir runs/.../features_<stamp>
python v83_spatial_stage2/stage2g_features.py --split val \
  --labels-dir runs/.../labels_val_<stamp> --output-dir runs/.../features_<stamp>

# Probes (token + Stage-2f scalar)
python v83_spatial_stage2/stage2g_probe.py --mode both --epochs 30 \
  --bootstrap 200 --output-dir runs/.../probe_<stamp>

# CPU-only tests
python -m unittest -v v83_spatial_stage2.test_stage2g
```

Tests: **11 Stage-2g tests pass** (plus 6 Stage-2f + 4 Stage-2f supplement =
21 total). The tests cover the label definition, AUC/AP/Brier/ECE, confusion,
negative operating points, episode-fold isolation, scalar whitelist and
feature-name leakage.

## Appendix A. Metric definitions

- `AUROC`: rank AUC over states; episodes bootstrap for CIs.
- `AUPRC`: average precision for the positive class; `AUPRC-neg`: average
  precision for the negative class (`1-y`, `1-p`), which is the relevant rare
  class for `target_visible`.
- `Brier`: mean squared error of the probability.
- `ECE`: 10 equal-width bins; weighted |confidence − accuracy|.
- `P@0.5` / `R@0.5` / `Spec@0.5`: positive-class precision/recall and
  negative-class recall at threshold 0.5.
- `BAcc@0.5`: 0.5 × (positive recall + negative recall).
- Negative operating point: flag `p <= t`; choose the lowest threshold that
  reaches a target negative recall (best precision at that recall), or the
  highest threshold that reaches a target negative precision (best recall at
  that precision).
- `near-invisible`: true distance <100 m and target invisible.
- `far-visible`: true distance >=100 m and target visible.
- `far-not-localizable`: true distance >=100 m and target not localizable.
- `far-invisible`: true distance >=100 m and target invisible.
- Episode folds: 5-fold grouped by `episode_id`; every state of an episode is
  in exactly one fold.
