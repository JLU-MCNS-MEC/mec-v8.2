# Stage-2s Smoke Test（训练前）

- 时间：2026-10-25T14:25:43.573327+00:00
- 部署哈希：`1fce485ce6256a288fb0aecd525afbf2830467b5ee9f2918952af54148a081cd`
- 模型：Small×5 + Transformer-M×5
- 真实闭环：`full_closed_loop_heterogeneous`，未使用 Stage-2r lane 缓存。

## 检查结果

- `two_models_loaded`: 通过
- `small_inference_expected`: 通过
- `transformer_m_inference_expected`: 通过
- `inference_ratio_gt_100`: 通过
- `e2e_includes_inference`: 通过
- `dual_model_actions_differ`: 通过
- `model_choice_changes_pose`: 通过
- `model_choice_changes_next_state`: 通过
- `state_dim_194`: 通过
- `mask_dim_10`: 通过
- `termination_within_400`: 通过
- `scheme_d_identity`: 通过
- `runner_scheme_d_identity`: 通过
- `model_usage_accounted`: 通过
- `e2e_components_ok`: 通过
- `closed_loop_no_lane_cache`: 通过
- `ok`: 通过

## 推理时延（0.2 TFLOPS 有效吞吐）

| 模型 | 解析 FLOPs | 推理时延 (ms) | 通信时延 (ms) | E2E (ms) |
|---|---:|---:|---:|---:|
| Small | 0.0142 G | 0.071130 | 11.147 | 33.934 |
| Transformer-M | 2.8146 G | 14.072978 | 11.147 | 47.935 |

## 同观测双模型动作

- Dev 任务索引：0
- 前缀步数：0
- Small 动作：1，Transformer-M 动作：0
- 选择不同模型后轨迹位姿改变：Small=True，Transformer-M=True
- 下一 194-D 状态不同：True

## 完整闭环 Episode

- Episode：`stage2n-dev-00000`，步数 400，成功 False
- 平均 E2E：83.889 ms
- 方案 D 奖励：0.161110，闭式 1-mean/100：0.161110，误差 5.55e-17
- Small/Transformer-M 步数：335/65

## 10–20 Episode 训练 Smoke

| 算法 | Episodes | 模式 | Small 选择 | 最大奖励误差 | 状态 |
|---|---:|---|---:|---:|---|
| DQN | 14 | full_closed_loop_heterogeneous | 59.22% | 9.99e-16 | completed |
| PPO | 14 | full_closed_loop_heterogeneous | 62.40% | 2.22e-16 | completed |

## 结论

全部检查通过，可以启动 6 个正式 run。
