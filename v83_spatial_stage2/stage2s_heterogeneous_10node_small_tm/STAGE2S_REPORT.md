# Stage-2s 最终报告：5×Small + 5×Transformer-M 异构 10 节点 MEC

- 阶段：`stage2s_heterogeneous_10node_small_tm`
- 分支：`experiment/v83-spatial-foundation`
- 报告生成时间：2026-10-25T21:30:46.111052+00:00
- 部署哈希：`1fce485ce6256a288fb0aecd525afbf2830467b5ee9f2918952af54148a081cd`
- 奖励：方案 D（仅 E2E）；种子：101/202/303；每个 run 4000 Episodes
- 数据：Stage-2n 固定城市 Train-4000 / Dev-60；**Test-400 全程未读取**
- 训练模式：`full_closed_loop_heterogeneous`，**未使用 Stage-2r lane 缓存**

## 1. 执行摘要

- Dev-60 平均 E2E 较低的学习算法：**PPO**（68.93 ms；另一算法 83.13 ms）。
- DQN 最佳 seed=202：Dev E2E 82.03 ms，Dev SR 36.67%，Small 选择 49.21%。
- PPO 最佳 seed=303：Dev E2E 68.52 ms，Dev SR 35.00%，Small 选择 51.78%。
- 达到预注册严格收敛判据的 run：**无** （共 6 个 run）。
- Smoke：两种模型均加载；Small 推理 0.0711 ms，Transformer-M 14.0730 ms；动作/位姿/下一状态随模型选择改变；方案 D 奖励误差 <1e-15。

## 2. 代码、协议与可复现性

- 训练代码 commit：`7f61e9acaf3a9643f8d4ae016a251e5f234d2f6a`；续训兼容修复 commit：`a41a170`（仅修复 CUDA RNG 恢复，不改变训练动力学）。
- 部署哈希：`1fce485ce6256a288fb0aecd525afbf2830467b5ee9f2918952af54148a081cd`。
- MEC 检查点：Small `7349ec9d…44e1f6`，Transformer-M `57c88b7d…73c6d`（完整 SHA256 见协议/部署 JSON）。
- 冻结协议：`STAGE2S_PROTOCOL.md`；部署明细：`deployment_heterogeneous.json`。
- 测试：`test_stage2s.py`（CPU，17/17）与 `test_stage2s_gpu.py`（GPU 真实闭环，7/7）全部通过。
- Smoke 报告：`smoke_report.json` / `SMOKE_TEST.md`。
- 未修改 Stage-2n/2o/2p/2r 的任何代码、任务、检查点或结果；新增文件仅位于本 Stage-2s 目录。

节点分配：

| 节点 | 模型 | x (m) | y (m) |
|---|---|---:|---:|
| M01 | Transformer-M | 61.5 | 159.9 |
| M02 | Transformer-M | 309.3 | 1229.7 |
| M03 | Small | 411.5 | 259.6 |
| M04 | Transformer-M | 1117.3 | 1254.9 |
| M05 | Transformer-M | 857.0 | 463.1 |
| M06 | Small | 836.3 | 949.7 |
| M07 | Small | 1305.0 | 161.1 |
| M08 | Transformer-M | 893.7 | 99.4 |
| M09 | Small | 123.6 | 589.0 |
| M10 | Small | 1321.6 | 689.1 |

## 3. 训练执行与累计统计

| 算法 | seed | Episodes | 成功 | 失败 | 成功率 | 成功 S/M/L | 失败 S/M/L | Small 选择 | MA100 E2E (ms) | 最佳 Dev E2E (ms) | 耗时 (h) | 收敛 |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|
| DQN | 101 | 4000 | 1574 | 2426 | 39.35% | 609/529/436 | 725/804/897 | 50.09% | 90.88 | 82.94 | 11.93 | not_converged |
| DQN | 202 | 4000 | 1564 | 2436 | 39.10% | 601/547/416 | 733/786/917 | 51.33% | 91.03 | 82.03 | 11.50 | not_converged |
| DQN | 303 | 4000 | 1567 | 2433 | 39.17% | 602/517/448 | 732/816/885 | 50.52% | 90.41 | 84.43 | 11.64 | not_converged |
| PPO | 101 | 4000 | 1531 | 2469 | 38.27% | 584/521/426 | 750/812/907 | 52.53% | 77.89 | 69.47 | 10.92 | not_converged |
| PPO | 202 | 4000 | 1603 | 2397 | 40.08% | 611/537/455 | 723/796/878 | 56.63% | 77.91 | 68.78 | 11.32 | not_converged |
| PPO | 303 | 4000 | 1487 | 2513 | 37.18% | 566/502/419 | 768/831/914 | 57.18% | 77.20 | 68.52 | 11.12 | not_converged |

### 3.1 算法聚合（3 seeds 均值 ± 总体标准差）

| 算法 | 成功率 | Small 选择 | MA100 E2E (ms) | 最佳 Dev E2E (ms) | 收敛 run |
|---|---:|---:|---:|---:|---:|
| DQN | 0.3921 ± 0.0010 | 0.5065 ± 0.0051 | 90.78 ± 0.26 | 83.13 ± 0.99 | 0/3 |
| PPO | 0.3851 ± 0.0120 | 0.5545 ± 0.0207 | 77.67 ± 0.33 | 68.93 ± 0.40 | 0/3 |

### 3.2 中断与恢复

训练启动器曾在首次运行约 Ep60 时被人工停止（仅停止启动器，训练子进程随之退出）。4 个 run 从 Ep50 的 `last_full.pt` 恢复；`training_history.json` 被截断到 Ep50 后重放，历史中无重复 Episode。另 2 个 run 从 0 开始。续训时发现共享 Stage-2k agent 的 CUDA RNG 恢复问题，已在 Stage-2s 本地包装修复（commit `a41a170`），未修改历史共享代码。

### 3.3 收敛性

预注册判据：最近三次 Dev E2E 波动 ≤ 最佳值 1%，且末次 ≤ 最佳值 101%。

- 满足判据：**无**。
- 未满足的 run 仍按最低 Dev E2E 选择 best checkpoint，并标记 `not_converged`；这不等于 E2E 没有下降，见 MA100 与最佳 Dev E2E。

## 4. Dev-60 最终验证

| 策略 | 类型 | SR | OSR | nDTW | SPL | E2E (ms) | P95 E2E (ms) | RewardD | 违约率 | Small 选择 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DQN-s101 | learned | 40.00% | 66.67% | 0.4087 | 0.3993 | 82.94 | 160.33 | 0.1706 | 27.24% | 48.08% |
| DQN-s202 | learned | 36.67% | 68.33% | 0.3640 | 0.3653 | 82.03 | 158.21 | 0.1797 | 26.29% | 49.21% |
| DQN-s303 | learned | 31.67% | 60.00% | 0.3612 | 0.3160 | 84.43 | 160.33 | 0.1557 | 30.16% | 57.15% |
| PPO-s101 | learned | 31.67% | 56.67% | 0.3598 | 0.3160 | 69.47 | 145.94 | 0.3053 | 11.40% | 54.12% |
| PPO-s202 | learned | 30.00% | 55.00% | 0.3478 | 0.2993 | 68.78 | 144.88 | 0.3122 | 12.25% | 62.20% |
| PPO-s303 | learned | 35.00% | 61.67% | 0.3707 | 0.3493 | 68.52 | 136.50 | 0.3148 | 11.74% | 51.78% |
| Greedy-Min-E2E | baseline | 35.00% | 61.67% | 0.3670 | 0.3493 | 66.26 | 129.04 | 0.3374 | 8.78% | 53.05% |
| Max-Uplink-Rate | baseline | 33.33% | 61.67% | 0.3640 | 0.3326 | 66.88 | 131.11 | 0.3312 | 9.05% | 47.53% |
| Nearest | baseline | 33.33% | 61.67% | 0.3640 | 0.3326 | 66.88 | 131.11 | 0.3312 | 9.05% | 47.53% |
| Random | baseline | 43.33% | 73.33% | 0.4582 | 0.4326 | 101.61 | 165.93 | -0.0161 | 49.75% | 49.50% |

### 4.1 学习算法的 3-seed 均值 ± 标准差

| 算法 | SR | OSR | nDTW | SPL | E2E (ms) | P95 E2E (ms) | RewardD | 违约率 | Small 选择 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DQN | 0.3611 ± 0.0342 | 0.6500 ± 0.0360 | 0.3780 ± 0.0218 | 0.3602 ± 0.0342 | 83.13 ± 0.99 | 159.62 ± 1.00 | 0.1687 ± 0.0099 | 0.2790 ± 0.0165 | 0.5148 ± 0.0404 |
| PPO | 0.3222 ± 0.0208 | 0.5778 ± 0.0283 | 0.3594 ± 0.0094 | 0.3215 ± 0.0208 | 68.93 ± 0.40 | 142.44 ± 4.22 | 0.3107 ± 0.0040 | 0.1180 ± 0.0035 | 0.5603 ± 0.0446 |

### 4.2 相对 baseline 的 E2E / 奖励 / 导航差

| 算法 | baseline | 学习均值 E2E (ms) | baseline E2E (ms) | ΔE2E (ms) | ΔE2E (%) | ΔRewardD | ΔSR | ΔnDTW |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DQN | Random | 83.13 | 101.61 | -18.48 | -18.19% | +0.1848 | -0.0722 | -0.0803 |
| DQN | Nearest | 83.13 | 66.88 | +16.26 | +24.31% | -0.1626 | +0.0278 | +0.0140 |
| DQN | Max-Uplink-Rate | 83.13 | 66.88 | +16.26 | +24.31% | -0.1626 | +0.0278 | +0.0140 |
| DQN | Greedy-Min-E2E | 83.13 | 66.26 | +16.88 | +25.47% | -0.1688 | +0.0111 | +0.0109 |
| PPO | Random | 68.93 | 101.61 | -32.69 | -32.17% | +0.3269 | -0.1111 | -0.0988 |
| PPO | Nearest | 68.93 | 66.88 | +2.05 | +3.07% | -0.0205 | -0.0111 | -0.0045 |
| PPO | Max-Uplink-Rate | 68.93 | 66.88 | +2.05 | +3.07% | -0.0205 | -0.0111 | -0.0045 |
| PPO | Greedy-Min-E2E | 68.93 | 66.26 | +2.67 | +4.03% | -0.0267 | -0.0278 | -0.0076 |

负 ΔE2E 表示学习策略时延更低；ΔRewardD 为正表示方案 D 奖励更高。baseline 与学习策略使用同一 10 节点部署和同一真实模型推理时延；Random 使用固定 RNG 种子 20261025，Nearest / Max-Uplink-Rate / Greedy-Min-E2E 为确定性规则；Greedy-Min-E2E 只使用当前状态可获得的 `total_ms`，不使用未来轨迹、目标位置或 Oracle 动作。

## 5. 异构性分析

### 5.1 时延分解与模型选择

| 策略 | 类型 | Small 步数比例 | 推理 Small (ms) | 推理 TM (ms) | 通信 Small (ms) | 通信 TM (ms) | E2E Small (ms) | E2E TM (ms) | 总体 E2E (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | baseline | 49.50% | 0.071 | 14.073 | 73.89 | 75.85 | 96.67 | 112.64 | 101.61 |
| nearest | baseline | 47.53% | 0.071 | 14.073 | 40.66 | 38.96 | 63.44 | 75.74 | 66.88 |
| max_uplink_rate | baseline | 47.53% | 0.071 | 14.073 | 40.66 | 38.96 | 63.44 | 75.74 | 66.88 |
| greedy_min_e2e | baseline | 53.05% | 0.071 | 14.073 | 42.37 | 37.54 | 65.16 | 74.33 | 66.26 |
| dqn_s101 | learned | 48.08% | 0.071 | 14.073 | 58.21 | 61.75 | 80.99 | 98.54 | 82.94 |
| dqn_s202 | learned | 49.21% | 0.071 | 14.073 | 63.96 | 57.49 | 86.75 | 94.27 | 82.03 |
| dqn_s303 | learned | 57.15% | 0.071 | 14.073 | 59.89 | 65.19 | 82.67 | 101.98 | 84.43 |
| ppo_s101 | learned | 54.12% | 0.071 | 14.073 | 46.61 | 38.70 | 69.40 | 75.48 | 69.47 |
| ppo_s202 | learned | 62.20% | 0.071 | 14.073 | 48.36 | 36.20 | 71.14 | 72.99 | 68.78 |
| ppo_s303 | learned | 51.78% | 0.071 | 14.073 | 42.83 | 40.72 | 65.61 | 77.51 | 68.52 |

### 5.2 策略是否利用模型异构性

- DQN：Dev-60 每 Episode 的 Small 步数比例均值 50.76%，标准差 0.341，范围 0.00%–100.00%；148/180 个 Episode 偏离 50% 超过 10 个百分点。
- PPO：Dev-60 每 Episode 的 Small 步数比例均值 53.05%，标准差 0.397，范围 0.00%–100.00%；165/180 个 Episode 偏离 50% 超过 10 个百分点。

`policy_diversity.csv` 给出任意两策略在同一 Dev 任务上的动作、节点、成功不一致计数。Stage-2s 中这些不一致是预期现象：模型选择会改变导航动作，不同于 Stage-2r 中“导航与关联无关”的同质设置。

训练前后选择行为（前 200 vs 末 500 Episodes）：

| 算法 | seed | 前 200 Small | 末 500 Small | ΔSmall | 前 200 E2E | 末 500 E2E | ΔE2E | 前 200 成功率 | 末 500 成功率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DQN | 101 | 49.74% | 46.38% | -3.36% | 98.62 | 90.03 | -8.59 | 43.00% | 35.40% |
| DQN | 202 | 47.17% | 52.14% | 4.97% | 97.46 | 91.06 | -6.40 | 50.00% | 34.00% |
| DQN | 303 | 48.82% | 50.48% | 1.66% | 98.34 | 89.49 | -8.86 | 43.50% | 35.20% |
| PPO | 101 | 50.49% | 46.63% | -3.86% | 98.47 | 78.01 | -20.46 | 44.50% | 36.40% |
| PPO | 202 | 51.11% | 57.67% | 6.56% | 96.90 | 77.00 | -19.90 | 46.50% | 37.20% |
| PPO | 303 | 47.02% | 56.86% | 9.84% | 98.44 | 75.87 | -22.57 | 47.00% | 32.60% |

配对检验（同一 Dev 任务，学习策略 − Greedy-Min-E2E；E2E 为负更好）：

| 对比 | n | ΔE2E 均值 (ms) | 95% CI | 学习更低/更高/持平 | ΔnDTW 均值 | ΔSR 均值 |
|---|---:|---:|---|---:|---:|---:|
| DQN vs Greedy-Min-E2E | 60 | +16.876 | [+12.848, +20.904] | 2/56/2 | 0.0109 | 0.0111 |
| PPO vs Greedy-Min-E2E | 60 | +2.669 | [+1.716, +3.622] | 7/46/7 | -0.0076 | -0.0278 |
| DQN vs PPO（负=DQN 更低） | 60 | +14.207 | [+10.351, +18.063] | 6/51/3 | 0.0185 | 0.0389 |

## 6. 导航质量是否出现代价

E2E 奖励只优化时延；SR/OSR/nDTW/SPL 不进入训练。如果策略偏好低推理成本的 Small 节点，导航质量可能下降。以下按“Small 步数占比 >50% / <50% / =50%”分组：

| 算法 | 分组 | Episodes | 平均 Small 比例 | 成功率 | nDTW | E2E (ms) | 步数 | 相关系数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DQN | Small | 30 | 74.42% | 23.33% | 0.3201 | 78.83 | 165.7 | - |
| DQN | Transformer-M | 30 | 21.87% | 50.00% | 0.4079 | 85.23 | 151.7 | - |
| DQN | corr_small_ratio_vs_success | 60 | 48.14% | 36.67% | 0.3640 | 82.03 | 158.7 | -0.298 |
| DQN | corr_small_ratio_vs_ndtw | 60 | 48.14% | 36.67% | 0.3640 | 82.03 | 158.7 | -0.182 |
| DQN | corr_small_ratio_vs_e2e_ms | 60 | 48.14% | 36.67% | 0.3640 | 82.03 | 158.7 | -0.192 |
| DQN | corr_small_ratio_vs_steps | 60 | 48.14% | 36.67% | 0.3640 | 82.03 | 158.7 | 0.040 |
| PPO | Small | 28 | 85.92% | 25.00% | 0.3160 | 63.78 | 186.0 | - |
| PPO | Transformer-M | 32 | 15.53% | 43.75% | 0.4186 | 72.68 | 161.2 | - |
| PPO | corr_small_ratio_vs_success | 60 | 48.38% | 35.00% | 0.3707 | 68.52 | 172.8 | -0.307 |
| PPO | corr_small_ratio_vs_ndtw | 60 | 48.38% | 35.00% | 0.3707 | 68.52 | 172.8 | -0.185 |
| PPO | corr_small_ratio_vs_e2e_ms | 60 | 48.38% | 35.00% | 0.3707 | 68.52 | 172.8 | -0.165 |
| PPO | corr_small_ratio_vs_steps | 60 | 48.38% | 35.00% | 0.3707 | 68.52 | 172.8 | 0.100 |

相关行为 Pearson 相关（Small 步数比例 vs 指标）：`corr_small_ratio_vs_success`、`_vs_ndtw`、`_vs_e2e_ms`、`_vs_steps`。相关不等于因果；模型选择与任务难度/位置同时变化，需结合配对 baseline 阅读。

### 6.1 奖励—成功解耦诊断（方案 D 无成功项）

方案 D 只依赖每步 E2E 与终止修正，没有成功奖励或 Stop 奖励。因此低 E2E 可能来自“短而失败”的 Episode。下表把成功/失败分开统计，用于检查最优 Dev E2E 检查点是否以导航质量为代价：

| 策略 | 算法 | 类型 | 结果 | Episodes | 平均 E2E (ms) | 平均步数 | 平均 nDTW | 平均 RewardD |
|---|---|---|---:|---:|---:|---:|---:|---:|
| random | heuristic | baseline | 成功 | 26 | 101.37 | 76.5 | 0.8088 | -0.0137 |
| random | heuristic | baseline | 失败 | 34 | 101.80 | 210.8 | 0.1901 | -0.0180 |
| nearest | heuristic | baseline | 成功 | 20 | 64.36 | 66.5 | 0.8541 | 0.3564 |
| nearest | heuristic | baseline | 失败 | 40 | 68.13 | 219.1 | 0.1189 | 0.3187 |
| max_uplink_rate | heuristic | baseline | 成功 | 20 | 64.36 | 66.5 | 0.8541 | 0.3564 |
| max_uplink_rate | heuristic | baseline | 失败 | 40 | 68.13 | 219.1 | 0.1189 | 0.3187 |
| greedy_min_e2e | heuristic | baseline | 成功 | 21 | 64.07 | 65.9 | 0.8395 | 0.3593 |
| greedy_min_e2e | heuristic | baseline | 失败 | 39 | 67.43 | 227.7 | 0.1126 | 0.3257 |
| dqn_s101 | dqn | learned | 成功 | 24 | 87.17 | 67.2 | 0.8337 | 0.1283 |
| dqn_s101 | dqn | learned | 失败 | 36 | 80.11 | 221.2 | 0.1254 | 0.1989 |
| dqn_s202 | dqn | learned | 成功 | 22 | 84.68 | 67.5 | 0.8297 | 0.1532 |
| dqn_s202 | dqn | learned | 失败 | 38 | 80.50 | 211.5 | 0.0943 | 0.1950 |
| dqn_s303 | dqn | learned | 成功 | 19 | 80.35 | 63.2 | 0.8853 | 0.1965 |
| dqn_s303 | dqn | learned | 失败 | 41 | 86.32 | 214.9 | 0.1183 | 0.1368 |
| ppo_s101 | ppo | learned | 成功 | 19 | 67.96 | 64.2 | 0.8621 | 0.3204 |
| ppo_s101 | ppo | learned | 失败 | 41 | 70.17 | 215.0 | 0.1270 | 0.2983 |
| ppo_s202 | ppo | learned | 成功 | 18 | 67.34 | 66.3 | 0.8608 | 0.3266 |
| ppo_s202 | ppo | learned | 失败 | 42 | 69.40 | 217.1 | 0.1279 | 0.3060 |
| ppo_s303 | ppo | learned | 成功 | 21 | 65.14 | 65.3 | 0.8556 | 0.3486 |
| ppo_s303 | ppo | learned | 失败 | 39 | 70.35 | 230.6 | 0.1096 | 0.2965 |

### 6.2 强制模型诊断（仅诊断，不进入预注册主比较）

把学习策略的动作 mask 限制为单一模型类型；若该模型在覆盖外，回退到不受限贪心动作并计数。该诊断用于看同一策略在“只能选 Small”与“只能选 Transformer-M”下的导航/时延差异。

| 算法 | 最佳 seed | 强制模型 | SR | OSR | nDTW | SPL | E2E (ms) | RewardD | 覆盖外回退步数 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| DQN | 202 | Small | 20.00% | 38.33% | 0.2730 | 0.2000 | 88.27 | 0.1173 | 1764 |
| DQN | 202 | Transformer-M | 58.33% | 81.67% | 0.5525 | 0.5826 | 102.61 | -0.0261 | 752 |
| PPO | 303 | Small | 20.00% | 38.33% | 0.2730 | 0.2000 | 85.39 | 0.1461 | 1764 |
| PPO | 303 | Transformer-M | 58.33% | 81.67% | 0.5525 | 0.5826 | 99.48 | 0.0052 | 752 |

## 7. 真实闭环轨迹

| 算法 | seed | Episode | 角色 | 成功 | 步数 | 路径 (m) | E2E (ms) | RewardD | Small 比例 | 选择状态 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| DQN | 202 | stage2n-dev-00013 | success_short | 是 | 42 | 195.0 | 101.85 | -0.0185 | 45.24% | exact |
| DQN | 202 | stage2n-dev-00008 | failure_short | 否 | 232 | 628.7 | 65.15 | 0.3485 | 1.29% | exact |
| DQN | 202 | stage2n-dev-00039 | success_medium | 是 | 58 | 275.0 | 77.52 | 0.2248 | 65.52% | exact |
| DQN | 202 | stage2n-dev-00020 | failure_medium | 否 | 162 | 565.0 | 96.46 | 0.0354 | 56.79% | exact |
| DQN | 202 | stage2n-dev-00054 | success_long | 是 | 88 | 425.0 | 64.61 | 0.3539 | 60.23% | exact |
| DQN | 202 | stage2n-dev-00049 | failure_long | 否 | 152 | 430.0 | 67.05 | 0.3295 | 55.92% | exact |
| PPO | 303 | stage2n-dev-00009 | success_short | 是 | 41 | 175.0 | 57.50 | 0.4250 | 0.00% | exact |
| PPO | 303 | stage2n-dev-00000 | failure_short | 否 | 400 | 982.6 | 59.91 | 0.4009 | 81.50% | exact |
| PPO | 303 | stage2n-dev-00039 | success_medium | 是 | 58 | 275.0 | 47.08 | 0.5292 | 100.00% | exact |
| PPO | 303 | stage2n-dev-00024 | failure_medium | 否 | 133 | 510.0 | 57.75 | 0.4225 | 79.70% | exact |
| PPO | 303 | stage2n-dev-00054 | success_long | 是 | 88 | 425.0 | 62.66 | 0.3734 | 69.32% | exact |
| PPO | 303 | stage2n-dev-00047 | failure_long | 否 | 177 | 580.0 | 58.87 | 0.4113 | 40.11% | exact |

轨迹审计：通过；选中轨迹 12 条。

轨迹图：`fig_traj_dqn.png`、`fig_traj_ppo.png`、`fig_traj_overview.png`、`fig_traj_model_usage.png`、`fig_traj_selection_vs_outcome.png`、`fig_traj_latency_example.png`。

## 8. 结果图

- `fig01_reward_ma100.png`：DQN/PPO Reward-MA100。
- `fig02_e2e_ma100.png`：DQN/PPO E2E-MA100。
- `fig03_dev_e2e_curves.png`：Dev E2E 随训练 Episode。
- `fig04_validation_metrics.png`：SR/OSR/nDTW/SPL/E2E/Reward 对比。
- `fig05_seed_variability.png` / `fig05b_training_stability.png`：三种子变异性与稳定性。
- `fig06_heterogeneity.png`：Small/TM 选择比例、推理/通信时延、E2E 分解。
- `fig07_baseline_e2e.png`：学习策略 vs baseline E2E。
- `fig08_success_failure.png`：成功/失败 band 分布。

所有图均为 300 dpi PNG，原始数据在 `figure_data/*.csv`。

## 9. 对用户问题的直接回答

1. **DQN/PPO 是否学会利用模型异构性？** 证据是选择比例随 Episode 变化且不固定在 50%：DQN 的 Small 选择 50.76%（标准差 0.341）；PPO 的 Small 选择 53.05%（标准差 0.397）。 训练前后：DQN Small 比例 48.58%→49.67%（Δ1.09%），E2E 98.14→90.19 ms；PPO Small 比例 49.54%→53.72%（Δ4.18%），E2E 97.94→76.96 ms。 配对检验：DQN 相对 Greedy 的 ΔE2E +16.88 ms（95% CI [+12.85, +20.90]，显著更高（更差））；PPO 相对 Greedy 的 ΔE2E +2.67 ms（95% CI [+1.72, +3.62]，显著更高（更差））。但“利用”不等于“优于 baseline”：如果学习策略的 E2E 不优于 Greedy-Min-E2E，则只能说明它学会了按当前时延选择，而不是学会了利用未来导航收益；若连 Random/Nearest 也不优于，则不能声称学会了有效的时延选择。
2. **哪种算法时延更低？** Dev-60 平均 E2E：PPO 68.93 ms，低于另一算法。差值是否稳定需同时看 3 seed 标准差与配对 CI。
3. **Small/Transformer-M 的选择比例？** 学习策略 Dev-60 每 Episode 的 Small 步数比例范围 48.08%–62.20%，Episode 平均 53.75%；Transformer-M Episode 平均 46.25%。 按步数加权（Dev-60 全部步）：DQN 51.48%，PPO 56.03%。 Greedy-Min-E2E 的 Small 比例为 53.05%，可作为“只按当前时延选择”的参照。
4. **E2E 改善是否超过 baseline？** DQN 仅在 1/4 个 baseline 上 E2E 更低；逐项 ΔE2E 为 Random -18.19%, Nearest +24.31%, Max-Uplink-Rate +24.31%, Greedy-Min-E2E +25.47%；PPO 仅在 1/4 个 baseline 上 E2E 更低；逐项 ΔE2E 为 Random -32.17%, Nearest +3.07%, Max-Uplink-Rate +3.07%, Greedy-Min-E2E +4.03%。负值表示学习策略更低；是否“超过”按逐 baseline 的符号判断，不把任意一个 baseline 当成唯一对手。
5. **导航质量是否出现代价？** DQN：Small 主导组 SR 23.33%、nDTW 0.3201、E2E 78.83 ms；Transformer-M 主导组 SR 50.00%、nDTW 0.4079、E2E 85.23 ms；PPO：Small 主导组 SR 25.00%、nDTW 0.3160、E2E 63.78 ms；Transformer-M 主导组 SR 43.75%、nDTW 0.4186、E2E 72.68 ms；DQN：成功 Episode 平均 E2E 84.07 ms/66 步，失败 82.31 ms/216 步；PPO：成功 Episode 平均 E2E 66.81 ms/65 步，失败 69.97 ms/221 步。判定：存在明显的导航质量代价（Small 主导组 SR 低于 Transformer-M 主导组）。 这与第 6 节的相关性方向一致；方案 D 没有成功/Stop 奖励，因此低 E2E 与较低 SR 可以同时出现，必须如实报告。
6. **哪些 run 满足预先规定的收敛判据？** 无。

## 10. 关键发现（自动汇总）

- 算法：PPO E2E 68.93 ms vs DQN 83.13 ms（Δ-14.21 ms，-17.09%）；SR 32.22% vs 36.11%（-3.89 pp）；nDTW 0.3594 vs 0.3780。
- DQN vs Greedy-Min-E2E：ΔE2E +16.88 ms （+25.47%），ΔSR +1.11 pp，ΔnDTW +0.0109；未超过 Greedy-Min-E2E。
- PPO vs Greedy-Min-E2E：ΔE2E +2.67 ms （+4.03%），ΔSR -2.78 pp，ΔnDTW -0.0076；未超过 Greedy-Min-E2E。
- 最低 E2E 学习 run：ppo_s303，E2E 68.52 ms，SR 35.00%，RewardD 0.3148。 Greedy-Min-E2E 为 66.26 ms / SR 35.00% / RewardD 0.3374。
- DQN 时延分解：通信 53.45 ms，推理 6.97 ms，Small 选择 51.48%。
- PPO 时延分解：通信 39.57 ms，推理 6.65 ms，Small 选择 56.03%。
- 解释：PPO 相对 DQN 的 E2E 优势主要来自更低的通信时延（选择更近/信道更好的节点），而不是单纯因为 Small 推理便宜；推理时延本身只有约 6.6–7.0 ms，通信约 39.6–53.5 ms。
- 导航代价（DQN）：Small 主导组 SR 23.33% / nDTW 0.3201 / E2E 78.83 ms；Transformer-M 主导组 SR 50.00% / nDTW 0.4079 / E2E 85.23 ms。
- 导航代价（PPO）：Small 主导组 SR 25.00% / nDTW 0.3160 / E2E 63.78 ms；Transformer-M 主导组 SR 43.75% / nDTW 0.4186 / E2E 72.68 ms。
- 强制模型（DQN，仅诊断）：强制 Small SR 20.00% / nDTW 0.2730 / E2E 88.27 ms（回退 1764 步）；强制 Transformer-M SR 58.33% / nDTW 0.5525 / E2E 102.61 ms（回退 752 步）。
- 强制模型（PPO，仅诊断）：强制 Small SR 20.00% / nDTW 0.2730 / E2E 85.39 ms（回退 1764 步）；强制 Transformer-M SR 58.33% / nDTW 0.5525 / E2E 99.48 ms（回退 752 步）。
- 奖励—成功解耦（DQN）：成功 E2E 84.07 ms / 66 步；失败 82.31 ms / 216 步；失败−成功 -1.76 ms。
- 奖励—成功解耦（PPO）：成功 E2E 66.81 ms / 65 步；失败 69.97 ms / 221 步；失败−成功 +3.16 ms。
- 收敛：0/6 达到严格判据；最接近的是 ppo_s202，最后三次波动 1.22%，末次差距 3.37%。

## 11. 限制

1. 单城市、单任务生成规则、两个冻结导航模型；结论不外推到新城市或重新训练导航模型。
2. 只有 3 个随机种子；不足以做强统计推断，只能检查种子间方向一致性。
3. 导航由冻结 MEC 模型产生，但模型选择会改变轨迹、结果和后续状态；不同算法/种子的 SR/OSR/nDTW/SPL 不同是设计事实，不是实现缺陷。
4. 0.2 TFLOP/s 是显式有效吞吐设定；E2E 可能超过 100 ms，方案 D 奖励可能为负。
5. Test-400 全程未读取；最终成绩仅基于 Dev-60。
6. 强制 Small/Transformer-M 诊断只限制动作 mask；若该模型在覆盖外则回退并计数，不是严格的同轨迹反事实。

## 12. 复现

```bash
PY=/home/ubuntu5/miniconda3/envs/DRL/bin/python
$PY -m unittest -v v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.test_stage2s
$PY -m unittest -v v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.test_stage2s_gpu
$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/smoke.py --episodes 14 --dev-episodes 10
$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/launch.py --wait --max-parallel 6 --min-free-mb 5000
bash v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/run_postprocess.sh
```

原始 checkpoint、TensorBoard Event 和逐 Episode 大日志在 `runs/v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/`（Git 忽略）。
