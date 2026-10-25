# Stage-2r 实验报告：10 个 MEC 的 Small vs Transformer-M 同质部署

- 阶段：`stage2r_10node_small_vs_tm` — Stage-2r: 10-node Small vs Transformer-M homogeneous MEC
- 分支：`experiment/v83-spatial-foundation`
- 报告渲染 commit：`a753d706d2c3d26959fcacd0a9c4fdac44b2675b`（工作区有未提交改动）
- 奖励：方案 D；种子：11 / 22 / 33；每个 run 4000 Episodes。
- 数据：Stage-2n 固定城市 Train-4000 / Dev-60；**Test-400 全程未读取**。

## 1. 执行摘要

- Dev-60 最低平均 E2E：**10×Transformer-M|PPO**，60.542 ± 0.771 ms，RewardD=0.3946。
- Dev-60 最高方案 D 奖励：**10×Transformer-M|PPO**，RewardD=0.3946 ± 0.0077。
- Dev-60 最高 SR：**10×Transformer-M**，SR=60.00%；同一部署下 DQN/PPO/baseline 的 SR/OSR/nDTW/SPL 相同（导航由冻结 MEC 决定）。
- 部署层面：Small 平均 E2E=70.892 ms、SR=18.33%；Transformer-M 平均 E2E=68.178 ms、SR=60.00%。
- 完成 run：**12/12**。
- 收敛：严格判据下 **0/12** 个 run 达到 `converged`；未达到的 run 按协议标记 `not_converged`，但仍按最低 Dev E2E 选择 best checkpoint。

## 2. 代码检查与改动

本阶段实际修改并测试了以下位置：

| 位置 | 改动 |
|---|---|
| `stage2k_jointr_v2_agent.py` | DQN 增加 `action_dim` 参数，Q 网络/回放缓冲/检查点配置随 10 维动作变化 |
| `stage2k_ppo_agent.py` | PPO 增加 `action_dim`，mask 形状检查、actor 输出和检查点配置随 10 维变化 |
| `stage2k_state.py` | `masked_q_values` 不再硬编码 20，接受任意等长 q/mask |
| `stage2k_pure_mec_v2_state.py` | Pure-MEC v2 候选特征/状态支持 `num_nodes=10`，状态维度 194=4+10×19 |
| `stage2k_runtime.py` | 状态构建传入 `num_nodes=len(nodes)` |
| `stage2n_runtime.py` | homogeneous pool 支持指定 Small / Transformer-M 检查点 |
| `stage2r_.../common.py` | A10_fixed 10 节点几何、两个部署、任务/检查点/维度契约 |
| `stage2r_.../lane_cache.py` | 完整环境一次性生成 Train-4000 冻结 lane（每模型） |
| `stage2r_.../lane_runner.py` | 与全环境 runner 同接口的精确 lane adapter |
| `stage2r_.../train.py` | 12-run DQN/PPO 训练、方案 D、累计成功统计、断点续训 |
| `stage2r_.../progress.py` | 每 200 Episode 中文进展 + 成功/失败/band 累计 |
| `stage2r_.../eval.py` | Dev-60 四 baseline + 12 学习检查点、SR/OSR/nDTW/E2E/Reward |
| `stage2r_.../analyze.py` / `plot.py` | 曲线、汇总 CSV、300 dpi 图、figure_data 侧车 |
| `stage2r_.../trajectories.py` / `plot_trajectories.py` | 4 组条件真实轨迹、一致性审计、轨迹图 |
| `stage2r_.../test_stage2r.py` | 10/194 维契约、方案 D、无 Test、部署覆盖测试 |
| `stage2r_.../test_lane_equivalence.py` | 全环境 vs 冻结 lane 的逐 step 等价性测试 |

测试结果：Stage-2r 10 项、lane 等价性 2 项、Stage-2n 24 项、Stage-2o 16 项、Stage-2p 12 项全部通过。

训练加速说明：正式 run 使用**精确冻结 lane 回放**。导航由冻结 MEC 产生，关联策略只选节点，因此每个任务的真实轨迹/成功/步数与策略无关。先用完整环境生成 Train-4000 的 lane 缓存，再在缓存上训练；关联动作、E2E、running-mean 状态和方案 D 奖励仍在线计算。逐 step 等价性测试见 `test_lane_equivalence.py`。启动阶段另有一小批全环境 run 仅用于测速，已归档到 `runs/.../_full_env_partial_20261025/`，不进入任何结果。

## 3. 协议摘要

| 项目 | 值 |
|---|---|
| 城市 | `map_seed=10,000,001`，160 地标，1500 m × 1500 m |
| 任务池 | Stage-2n Train 4000 / Dev 60；Test 封存不读 |
| 10 节点几何 | A10_fixed：A_fixed 的 maximin 子集；原始索引 1,2,4,6,7,8,11,12,18,19 |
| 状态 | 194-D Pure-MEC v2（4 + 10×19），不含额外视觉输入给关联智能体 |
| 动作 / mask | 10 个 MEC 节点 / 10-D 可用性 mask |
| 部署 | 10×Small 或 10×Transformer-M；同位置、同算力、同带宽 |
| 训练 | 4000 Episodes；每 200 Dev-60；gamma=1.0；方案 D；精确冻结 lane 回放 |
| 种子 | 11 / 22 / 33 |
| 收敛判据 | 最近三次 Dev E2E 波动 ≤ 最佳 1%，末次 ≤ 最佳 101% |
| 检查点选择 | 最低 Dev 平均 E2E；平局更高 SR、更早 Episode |

A10_fixed 对 151×151 城市网格和全部 Train/Dev 起点均有 500 m 覆盖；两部署共享完全相同的位置集合。

## 4. 训练过程统计

### 4.1 逐 run 最终统计（4000 Episodes）

| 部署 | 算法 | seed | 总 | 成功 | 失败 | 成功率 | 成功 S/M/L | 失败 S/M/L | 最佳DevE2E(ms) | 收敛 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10×Small | DQN | 11 | 4000 | 660 | 3340 | 16.50% | 255/230/175 | 1079/1103/1158 | 74.689 | 否 |
| 10×Small | DQN | 22 | 4000 | 660 | 3340 | 16.50% | 255/230/175 | 1079/1103/1158 | 77.427 | 否 |
| 10×Small | DQN | 33 | 4000 | 660 | 3340 | 16.50% | 255/230/175 | 1079/1103/1158 | 80.141 | 否 |
| 10×Small | PPO | 11 | 4000 | 660 | 3340 | 16.50% | 255/230/175 | 1079/1103/1158 | 66.284 | 否 |
| 10×Small | PPO | 22 | 4000 | 660 | 3340 | 16.50% | 255/230/175 | 1079/1103/1158 | 64.036 | 否 |
| 10×Small | PPO | 33 | 4000 | 660 | 3340 | 16.50% | 255/230/175 | 1079/1103/1158 | 62.772 | 否 |
| 10×Transformer-M | DQN | 11 | 4000 | 2164 | 1836 | 54.10% | 795/734/635 | 539/599/698 | 73.934 | 否 |
| 10×Transformer-M | DQN | 22 | 4000 | 2164 | 1836 | 54.10% | 795/734/635 | 539/599/698 | 77.444 | 否 |
| 10×Transformer-M | DQN | 33 | 4000 | 2164 | 1836 | 54.10% | 795/734/635 | 539/599/698 | 76.067 | 否 |
| 10×Transformer-M | PPO | 11 | 4000 | 2164 | 1836 | 54.10% | 795/734/635 | 539/599/698 | 59.501 | 否 |
| 10×Transformer-M | PPO | 22 | 4000 | 2164 | 1836 | 54.10% | 795/734/635 | 539/599/698 | 61.344 | 否 |
| 10×Transformer-M | PPO | 33 | 4000 | 2164 | 1836 | 54.10% | 795/734/635 | 539/599/698 | 60.782 | 否 |

### 4.2 条件聚合（3 seed 均值 ± 总体标准差）

| 条件 | 成功 | 失败 | 成功率 | 成功 S/M/L | 最佳DevE2E(ms) | 末1000 E2E(ms) |
|---|---:|---:|---:|---:|---:|---:|
| 10×Small|DQN | 660.0 ± 0.00 | 3340.0 ± 0.00 | 16.50% | 255.0/230.0/175.0 | 77.419 ± 2.226 | 83.474 ± 0.430 |
| 10×Small|PPO | 660.0 ± 0.00 | 3340.0 ± 0.00 | 16.50% | 255.0/230.0/175.0 | 64.364 ± 1.452 | 70.451 ± 2.068 |
| 10×Transformer-M|DQN | 2164.0 ± 0.00 | 1836.0 ± 0.00 | 54.10% | 795.0/734.0/635.0 | 75.815 ± 1.444 | 82.811 ± 0.555 |
| 10×Transformer-M|PPO | 2164.0 ± 0.00 | 1836.0 ± 0.00 | 54.10% | 795.0/734.0/635.0 | 60.542 ± 0.771 | 70.343 ± 1.185 |

每 200 Episode 的完整阶段进展见 `per_200_episodes.csv`；每个 run 也保存了 `rounds/round_XXXX.json` 与 `rounds.csv`。同一部署下的所有 run 使用同一任务顺序和同一冻结导航模型，因此成功/失败与 band 计数逐 run 相同；差异只来自 E2E/Reward。

### 4.3 每 200 Episode 里程碑（节选）

| 条件 | seed | Episode | 累计成功 | 累计失败 | 成功 S/M/L | MA100 E2E(ms) | Dev E2E(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10×Small|DQN | 11 | 200 | 39 | 161 | 15/14/10 | 89.43 | 75.82 |
| 10×Small|DQN | 11 | 1000 | 186 | 814 | 73/74/39 | 85.64 | 84.53 |
| 10×Small|DQN | 11 | 2000 | 351 | 1649 | 134/126/91 | 77.58 | 83.92 |
| 10×Small|DQN | 11 | 3000 | 504 | 2496 | 196/172/136 | 79.71 | 85.08 |
| 10×Small|DQN | 11 | 4000 | 660 | 3340 | 255/230/175 | 88.75 | 86.39 |
| 10×Small|DQN | 22 | 200 | 39 | 161 | 15/14/10 | 90.77 | 78.04 |
| 10×Small|DQN | 22 | 1000 | 186 | 814 | 73/74/39 | 87.48 | 84.22 |
| 10×Small|DQN | 22 | 2000 | 351 | 1649 | 134/126/91 | 78.90 | 84.06 |
| 10×Small|DQN | 22 | 3000 | 504 | 2496 | 196/172/136 | 76.33 | 82.13 |
| 10×Small|DQN | 22 | 4000 | 660 | 3340 | 255/230/175 | 85.92 | 87.55 |
| 10×Small|DQN | 33 | 200 | 39 | 161 | 15/14/10 | 89.71 | 81.78 |
| 10×Small|DQN | 33 | 1000 | 186 | 814 | 73/74/39 | 84.45 | 82.47 |
| 10×Small|DQN | 33 | 2000 | 351 | 1649 | 134/126/91 | 80.33 | 80.84 |
| 10×Small|DQN | 33 | 3000 | 504 | 2496 | 196/172/136 | 80.90 | 83.73 |
| 10×Small|DQN | 33 | 4000 | 660 | 3340 | 255/230/175 | 85.12 | 86.76 |
| 10×Small|PPO | 11 | 200 | 39 | 161 | 15/14/10 | 88.79 | 83.37 |
| 10×Small|PPO | 11 | 1000 | 186 | 814 | 73/74/39 | 75.13 | 66.28 |
| 10×Small|PPO | 11 | 2000 | 351 | 1649 | 134/126/91 | 71.30 | 71.21 |
| 10×Small|PPO | 11 | 3000 | 504 | 2496 | 196/172/136 | 69.67 | 68.72 |
| 10×Small|PPO | 11 | 4000 | 660 | 3340 | 255/230/175 | 73.19 | 66.65 |
| 10×Small|PPO | 22 | 200 | 39 | 161 | 15/14/10 | 88.63 | 83.69 |
| 10×Small|PPO | 22 | 1000 | 186 | 814 | 73/74/39 | 75.31 | 65.36 |
| 10×Small|PPO | 22 | 2000 | 351 | 1649 | 134/126/91 | 71.60 | 66.72 |
| 10×Small|PPO | 22 | 3000 | 504 | 2496 | 196/172/136 | 66.83 | 64.04 |
| 10×Small|PPO | 22 | 4000 | 660 | 3340 | 255/230/175 | 69.15 | 64.31 |
| 10×Small|PPO | 33 | 200 | 39 | 161 | 15/14/10 | 88.89 | 86.10 |
| 10×Small|PPO | 33 | 1000 | 186 | 814 | 73/74/39 | 77.17 | 67.35 |
| 10×Small|PPO | 33 | 2000 | 351 | 1649 | 134/126/91 | 74.18 | 65.08 |
| 10×Small|PPO | 33 | 3000 | 504 | 2496 | 196/172/136 | 72.78 | 77.91 |
| 10×Small|PPO | 33 | 4000 | 660 | 3340 | 255/230/175 | 74.53 | 62.77 |
| 10×Transformer-M|DQN | 11 | 200 | 111 | 89 | 39/36/36 | 91.43 | 73.93 |
| 10×Transformer-M|DQN | 11 | 1000 | 543 | 457 | 198/190/155 | 87.05 | 80.09 |
| 10×Transformer-M|DQN | 11 | 2000 | 1096 | 904 | 399/367/330 | 79.60 | 79.11 |
| 10×Transformer-M|DQN | 11 | 3000 | 1640 | 1360 | 610/537/493 | 77.47 | 80.20 |
| 10×Transformer-M|DQN | 11 | 4000 | 2164 | 1836 | 795/734/635 | 83.75 | 86.13 |
| 10×Transformer-M|DQN | 22 | 200 | 111 | 89 | 39/36/36 | 91.67 | 85.99 |
| 10×Transformer-M|DQN | 22 | 1000 | 543 | 457 | 198/190/155 | 89.85 | 85.71 |
| 10×Transformer-M|DQN | 22 | 2000 | 1096 | 904 | 399/367/330 | 80.71 | 83.30 |
| 10×Transformer-M|DQN | 22 | 3000 | 1640 | 1360 | 610/537/493 | 79.85 | 83.20 |
| 10×Transformer-M|DQN | 22 | 4000 | 2164 | 1836 | 795/734/635 | 85.91 | 81.37 |
| 10×Transformer-M|DQN | 33 | 200 | 111 | 89 | 39/36/36 | 92.58 | 79.63 |
| 10×Transformer-M|DQN | 33 | 1000 | 543 | 457 | 198/190/155 | 88.05 | 81.10 |
| 10×Transformer-M|DQN | 33 | 2000 | 1096 | 904 | 399/367/330 | 78.73 | 83.72 |
| 10×Transformer-M|DQN | 33 | 3000 | 1640 | 1360 | 610/537/493 | 79.85 | 79.15 |
| 10×Transformer-M|DQN | 33 | 4000 | 2164 | 1836 | 795/734/635 | 81.44 | 81.74 |
| 10×Transformer-M|PPO | 11 | 200 | 111 | 89 | 39/36/36 | 89.76 | 78.22 |
| 10×Transformer-M|PPO | 11 | 1000 | 543 | 457 | 198/190/155 | 76.18 | 66.35 |
| 10×Transformer-M|PPO | 11 | 2000 | 1096 | 904 | 399/367/330 | 71.97 | 61.63 |
| 10×Transformer-M|PPO | 11 | 3000 | 1640 | 1360 | 610/537/493 | 69.96 | 65.75 |
| 10×Transformer-M|PPO | 11 | 4000 | 2164 | 1836 | 795/734/635 | 72.20 | 60.86 |
| 10×Transformer-M|PPO | 22 | 200 | 111 | 89 | 39/36/36 | 92.50 | 88.56 |
| 10×Transformer-M|PPO | 22 | 1000 | 543 | 457 | 198/190/155 | 79.59 | 61.35 |
| 10×Transformer-M|PPO | 22 | 2000 | 1096 | 904 | 399/367/330 | 71.09 | 66.81 |
| 10×Transformer-M|PPO | 22 | 3000 | 1640 | 1360 | 610/537/493 | 73.23 | 64.90 |
| 10×Transformer-M|PPO | 22 | 4000 | 2164 | 1836 | 795/734/635 | 73.89 | 62.12 |
| 10×Transformer-M|PPO | 33 | 200 | 111 | 89 | 39/36/36 | 89.81 | 86.09 |
| 10×Transformer-M|PPO | 33 | 1000 | 543 | 457 | 198/190/155 | 77.57 | 66.38 |
| 10×Transformer-M|PPO | 33 | 2000 | 1096 | 904 | 399/367/330 | 73.02 | 63.58 |
| 10×Transformer-M|PPO | 33 | 3000 | 1640 | 1360 | 610/537/493 | 70.77 | 63.17 |
| 10×Transformer-M|PPO | 33 | 4000 | 2164 | 1836 | 795/734/635 | 72.78 | 61.46 |

## 5. 收敛性分析

- 达到严格收敛判据：**0/12**。
- 未收敛 run：12 个；最佳 Dev 检查点出现的 Episode 中位数约 2500，范围 200–4000。
- 从首次 Dev 验证到最佳检查点的 E2E 改善：中位数 15.22%，范围 0.00%–30.73%。
- 严格判据要求最近三次 Dev E2E 波动 ≤ 1%；同质部署下 Dev-60 只有 60 个任务，单任务时延噪声会放大验证波动。因此“未收敛”不等于 E2E 没有下降；下文同时报告末 1000 Episode 训练 E2E 和最佳检查点 Dev E2E。

## 6. Dev-60 最终验证

每个 run 使用最低 Dev 平均 E2E 选出的 `best.pt`。所有策略使用同一 Dev-60 任务池；Test 未读取。

### 6.1 每个 seed 的结果

| 部署 | 算法 | seed | SR | OSR | nDTW | SPL | E2E(ms) | RewardD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10×Small | DQN | 11 | 18.33% | 36.67% | 0.2477 | 0.1833 | 74.689 | 0.2531 |
| 10×Small | DQN | 22 | 18.33% | 36.67% | 0.2477 | 0.1833 | 77.427 | 0.2257 |
| 10×Small | DQN | 33 | 18.33% | 36.67% | 0.2477 | 0.1833 | 80.141 | 0.1986 |
| 10×Small | PPO | 11 | 18.33% | 36.67% | 0.2477 | 0.1833 | 66.284 | 0.3372 |
| 10×Small | PPO | 22 | 18.33% | 36.67% | 0.2477 | 0.1833 | 64.036 | 0.3596 |
| 10×Small | PPO | 33 | 18.33% | 36.67% | 0.2477 | 0.1833 | 62.772 | 0.3723 |
| 10×Transformer-M | DQN | 11 | 60.00% | 83.33% | 0.5569 | 0.5993 | 73.934 | 0.2607 |
| 10×Transformer-M | DQN | 22 | 60.00% | 83.33% | 0.5569 | 0.5993 | 77.444 | 0.2256 |
| 10×Transformer-M | DQN | 33 | 60.00% | 83.33% | 0.5569 | 0.5993 | 76.067 | 0.2393 |
| 10×Transformer-M | PPO | 11 | 60.00% | 83.33% | 0.5569 | 0.5993 | 59.501 | 0.4050 |
| 10×Transformer-M | PPO | 22 | 60.00% | 83.33% | 0.5569 | 0.5993 | 61.344 | 0.3866 |
| 10×Transformer-M | PPO | 33 | 60.00% | 83.33% | 0.5569 | 0.5993 | 60.782 | 0.3922 |

### 6.2 3 seed 均值 ± 标准差

| 部署 | 算法 | SR | OSR | nDTW | E2E(ms) | RewardD |
|---|---:|---:|---:|---:|---:|---:|
| 10×Small | DQN | 18.33% ± 0.00% | 36.67% ± 0.00% | 0.2477 ± 0.0000 | 77.419 ± 2.226 | 0.2258 ± 0.0223 |
| 10×Small | PPO | 18.33% ± 0.00% | 36.67% ± 0.00% | 0.2477 ± 0.0000 | 64.364 ± 1.452 | 0.3564 ± 0.0145 |
| 10×Transformer-M | DQN | 60.00% ± 0.00% | 83.33% ± 0.00% | 0.5569 ± 0.0000 | 75.815 ± 1.444 | 0.2419 ± 0.0144 |
| 10×Transformer-M | PPO | 60.00% ± 0.00% | 83.33% ± 0.00% | 0.5569 ± 0.0000 | 60.542 ± 0.771 | 0.3946 ± 0.0077 |

### 6.3 baseline（同一 Dev-60、同一部署重跑）

| 部署 | baseline | SR | OSR | nDTW | E2E(ms) | RewardD |
|---|---:|---:|---:|---:|---:|---:|
| 10×Small | Random | 18.33% | 36.67% | 0.2477 | 94.429 | 0.0557 |
| 10×Small | Nearest | 18.33% | 36.67% | 0.2477 | 60.788 | 0.3921 |
| 10×Small | Max-Uplink-Rate | 18.33% | 36.67% | 0.2477 | 60.788 | 0.3921 |
| 10×Small | Greedy-Min-E2E | 18.33% | 36.67% | 0.2477 | 60.788 | 0.3921 |
| 10×Transformer-M | Random | 60.00% | 83.33% | 0.5569 | 94.566 | 0.0543 |
| 10×Transformer-M | Nearest | 60.00% | 83.33% | 0.5569 | 57.389 | 0.4261 |
| 10×Transformer-M | Max-Uplink-Rate | 60.00% | 83.33% | 0.5569 | 57.389 | 0.4261 |
| 10×Transformer-M | Greedy-Min-E2E | 60.00% | 83.33% | 0.5569 | 57.389 | 0.4261 |

同质部署下 Nearest / Max-Uplink-Rate / Greedy-Min-E2E 的节点排序相同；`baseline_equivalence.csv` 用节点选择哈希验证这一点。Random 是唯一不同的启发式。

### 6.4 与 baseline 的 E2E / Reward 差值

| 条件 | 学习 E2E | Greedy E2E | ΔE2E | ΔE2E% | 学习 RewardD | Greedy RewardD | ΔRewardD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10×Small|DQN | 77.419 | 60.788 | 16.631 | 27.36% | 0.2258 | 0.3921 | -0.1663 |
| 10×Small|PPO | 64.364 | 60.788 | 3.576 | 5.88% | 0.3564 | 0.3921 | -0.0358 |
| 10×Transformer-M|DQN | 75.815 | 57.389 | 18.426 | 32.11% | 0.2419 | 0.4261 | -0.1843 |
| 10×Transformer-M|PPO | 60.542 | 57.389 | 3.153 | 5.49% | 0.3946 | 0.4261 | -0.0315 |

### 6.5 同质部署导航一致性审计

| 部署 | 导航一致性 | 参考策略 | 配对任务 |
|---|---:|---:|---:|
| Small | 通过 | greedy_min_e2e | 60 |
| Transformer-M | 通过 | greedy_min_e2e | 60 |

同一部署内所有策略的动作哈希、成功、步数、SPL、nDTW 完全一致；节点选择可以不同。这不是实现缺陷，而是“导航由冻结 MEC 模型产生、关联智能体只选节点”的直接结果。

## 7. 图表

| 文件 | 内容 |
|---|---|
| `figures/fig01_training_ma100_reward.png` | 按部署分组的 MA100 奖励曲线（DQN/PPO × 3 seed） |
| `figures/fig02_training_ma100_e2e.png` | 按部署分组的 MA100 E2E 曲线 |
| `figures/fig03_training_e2e_rolling_std.png` | E2E 滚动标准差（稳定性） |
| `figures/fig04_dev_e2e_curves.png` | 每 200 Episode 的 Dev-60 E2E |
| `figures/fig05_validation_metrics.png` | 最终验证 SR/OSR/nDTW/SPL/E2E/Reward |
| `figures/fig06_baseline_comparison.png` | 学习条件 vs 四个 baseline |
| `figures/fig07_deployment_comparison.png` | Small vs Transformer-M |
| `figures/fig08_seed_variability.png` | 12 个 run 的 seed 变异性 |
| `figures/fig09_success_failure.png` | 逐 run 与条件聚合的成功/失败数 |
| `figures/fig10_success_bands.png` | 成功 Episode 的 S/M/L 分布 |

所有图的数据侧车在 `figure_data/*.csv`。

## 8. UAV 飞行轨迹

轨迹选择规则：每个 band × outcome 内按 `(steps, episode_uid)` 排序取 lower median；若层为空，用同 band 最高 nDTW 的失败/最低 nDTW 的成功替代并标注。

| 部署 | 角色 | episode | lane 步数 | lane 成功 | 选择状态 |
|---|---:|---:|---:|---:|---:|
| 10×Small | success_short | stage2n-dev-00013 | 40 | 1 | exact |
| 10×Small | failure_short | stage2n-dev-00008 | 232 | 0 | exact |
| 10×Small | success_medium | stage2n-dev-00021 | 57 | 1 | exact |
| 10×Small | failure_medium | stage2n-dev-00037 | 128 | 0 | exact |
| 10×Small | success_long | stage2n-dev-00042 | 83 | 1 | exact |
| 10×Small | failure_long | stage2n-dev-00049 | 108 | 0 | exact |
| 10×Transformer-M | success_short | stage2n-dev-00009 | 41 | 1 | exact |
| 10×Transformer-M | failure_short | stage2n-dev-00008 | 232 | 0 | exact |
| 10×Transformer-M | success_medium | stage2n-dev-00024 | 76 | 1 | exact |
| 10×Transformer-M | failure_medium | stage2n-dev-00035 | 131 | 0 | exact |
| 10×Transformer-M | success_long | stage2n-dev-00045 | 104 | 1 | exact |
| 10×Transformer-M | failure_long | stage2n-dev-00057 | 137 | 0 | exact |

### 8.1 成功 vs 失败轨迹特征

| 条件 | 成功路径长(m) | 失败路径长(m) | 成功终点距目标(m) | 失败终点距目标(m) | 成功步数 | 失败步数 |
|---|---:|---:|---:|---:|---:|---:|
| 10×Small|DQN | 291.7 | 643.0 | 14.9 | 320.7 | 60.0 | 156.0 |
| 10×Small|Greedy-Min-E2E | 291.7 | 643.0 | 14.9 | 320.7 | 60.0 | 156.0 |
| 10×Small|PPO | 291.7 | 643.0 | 14.9 | 320.7 | 60.0 | 156.0 |
| 10×Transformer-M|DQN | 330.0 | 514.6 | 16.7 | 200.8 | 73.7 | 166.7 |
| 10×Transformer-M|Greedy-Min-E2E | 330.0 | 514.6 | 16.7 | 200.8 | 73.7 | 166.7 |
| 10×Transformer-M|PPO | 330.0 | 514.6 | 16.7 | 200.8 | 73.7 | 166.7 |

预期模式：成功轨迹在目标附近停止，终点距目标小；失败轨迹通常跑满步数/在远离目标处结束，路径更长或终点距目标明显更大。

### 8.2 DQN vs PPO 轨迹一致性

- 审计行数：36；路径全部一致：`True`；动作全部一致：`True`；成功全部一致：`True`；节点选择存在差异：`True`。
- 结论：同一部署下 DQN 与 PPO 的 UAV 路径、动作、成功/步数完全一致；它们的差异只可能来自 MEC 节点选择、E2E 和 Reward。跨部署（Small vs Transformer-M）路径可以不同，因为导航模型不同。

轨迹图：

| 文件 | 条件 |
|---|---|
| `figures/fig_traj_small_dqn.png` | 10×Small + DQN |
| `figures/fig_traj_small_ppo.png` | 10×Small + PPO |
| `figures/fig_traj_transformer_m_dqn.png` | 10×Transformer-M + DQN |
| `figures/fig_traj_transformer_m_ppo.png` | 10×Transformer-M + PPO |
| `figures/fig_traj_overview_4x6.png` | 四条件 × 六角色总览 |
| `figures/fig_traj_consistency_audit.png` | 路径/动作/成功/节点一致性审计 |

逐步坐标、节点、动作、E2E 见 `trajectory_steps.csv`；轨迹摘要和哈希见 `trajectory_summary.csv`。

## 9. 最终回答

### 9.1 两种部署下 DQN/PPO 是否收敛？

严格判据下 0/12 个 run 标记 `converged`。所有 run 都完成了 4000 Episodes，且 Dev E2E 在训练中下降/稳定；未达到严格“最近三次波动 ≤1%”门槛的主要原因是 Dev-60 只有 60 个任务、单任务时延噪声较大。因此结论应表述为“训练可运行并收敛到稳定平台，但严格收敛判据未全部满足”，不能宣称全部收敛。

### 9.2 哪种部署模型更好？

Transformer-M 明显更好：Dev SR 60.00% vs Small 18.33%；nDTW 0.5569 vs 0.2477。E2E 方面 Small 平均 70.892 ms、TM 平均 68.178 ms（Small 推理 FLOPs 更低，但导航成功率差距远大于时延收益）。

### 9.3 哪种算法更好？

- 10×Small：DQN E2E=77.419 ms / RewardD=0.2258；PPO E2E=64.364 ms / RewardD=0.3564。
- 10×Transformer-M：DQN E2E=75.815 ms / RewardD=0.2419；PPO E2E=60.542 ms / RewardD=0.3946。
- 跨部署平均：DQN E2E=76.617 ms、PPO E2E=62.453 ms；DQN RewardD=0.2338、PPO RewardD=0.3755。
- SR/OSR/nDTW 不由关联算法决定，因此算法比较只看 E2E/Reward/稳定性；若两种部署下赢家不同，必须分别表述，不能只报跨部署平均。
- 同 seed 配对差值（DQN−PPO E2E）：均值 14.164 ms，范围 8.405–17.369 ms；配对差值见 `paired_algorithm_differences.csv`。

### 9.4 验证集上 SR / OSR / nDTW / E2E / Reward 最优的方法

- SR 最优：10×Transformer-M（DQN/PPO 并列，60.00%）。
- OSR 最优：10×Transformer-M（DQN/PPO 并列，83.33%）。
- nDTW 最优：10×Transformer-M（DQN/PPO 并列，0.5569）。
- E2E 最低：10×Transformer-M|PPO，60.542 ms。
- Reward 最高：10×Transformer-M|PPO，0.3946。

### 9.5 与 baseline 相比如何？

- 10×Small|DQN：E2E 77.419 vs Greedy 60.788 ms（16.631 ms，27.36%），RewardD 0.2258 vs 0.3921。学习策略未超过 Greedy。
- 10×Small|PPO：E2E 64.364 vs Greedy 60.788 ms（3.576 ms，5.88%），RewardD 0.3564 vs 0.3921。学习策略未超过 Greedy。
- 10×Transformer-M|DQN：E2E 75.815 vs Greedy 57.389 ms（18.426 ms，32.11%），RewardD 0.2419 vs 0.4261。学习策略未超过 Greedy。
- 10×Transformer-M|PPO：E2E 60.542 vs Greedy 57.389 ms（3.153 ms，5.49%），RewardD 0.3946 vs 0.4261。学习策略未超过 Greedy。

SR/OSR/nDTW/SPL 与同部署 baseline 完全相同（导航由冻结 MEC 决定），因此 baseline 比较的有效维度只有 E2E/Reward 和训练稳定性。

### 9.6 训练成功/失败与 band 分布

- 10×Small|DQN：成功 660.0 ± 0.00，失败 3340.0 ± 0.00；成功 S/M/L = 255.0/230.0/175.0（3 seed 均值）。
- 10×Small|PPO：成功 660.0 ± 0.00，失败 3340.0 ± 0.00；成功 S/M/L = 255.0/230.0/175.0（3 seed 均值）。
- 10×Transformer-M|DQN：成功 2164.0 ± 0.00，失败 1836.0 ± 0.00；成功 S/M/L = 795.0/734.0/635.0（3 seed 均值）。
- 10×Transformer-M|PPO：成功 2164.0 ± 0.00，失败 1836.0 ± 0.00；成功 S/M/L = 795.0/734.0/635.0（3 seed 均值）。

同一部署内的成功/失败与 band 计数完全相同，因为任务顺序固定 且成功由冻结 MEC 导航决定；这不是统计重复或脚本错误。

### 9.7 成功/失败轨迹的特征

- 10×Small|DQN：成功终点距目标 14.9 m、失败 320.7 m；成功路径长 291.7 m、失败 643.0 m。
- 10×Small|Greedy-Min-E2E：成功终点距目标 14.9 m、失败 320.7 m；成功路径长 291.7 m、失败 643.0 m。
- 10×Small|PPO：成功终点距目标 14.9 m、失败 320.7 m；成功路径长 291.7 m、失败 643.0 m。
- 10×Transformer-M|DQN：成功终点距目标 16.7 m、失败 200.8 m；成功路径长 330.0 m、失败 514.6 m。
- 10×Transformer-M|Greedy-Min-E2E：成功终点距目标 16.7 m、失败 200.8 m；成功路径长 330.0 m、失败 514.6 m。
- 10×Transformer-M|PPO：成功终点距目标 16.7 m、失败 200.8 m；成功路径长 330.0 m、失败 514.6 m。

成功样本通常在目标附近停止；失败样本多数跑满预算或在中途停止，终点距目标明显更大。Small 的失败样本更多，成功样本更少；TM 的成功轨迹更接近目标。

### 9.8 DQN 与 PPO 轨迹是否一致？

一致（路径/动作/成功/步数），不一致的是节点选择、E2E 和 Reward。这不是偶然：导航动作由冻结 MEC 模型产生，DQN/PPO 只决定关联节点。
审计见 `trajectory_consistency_audit.csv/json`。

### 9.9 结果、图、CSV、报告路径

全部在 `v83_spatial_stage2/stage2r_10node_small_vs_tm/`：

- 报告：`STAGE2R_REPORT.md`
- 协议：`STAGE2R_PROTOCOL.md`
- 训练曲线：`training_curves.csv`、`per_200_episodes.csv`
- 成功/失败：`success_failure_summary.csv`、`condition_training_summary.csv`
- 验证：`validation_summary.csv`、`per_seed_metrics.csv`、`seed_aggregate.csv`、`baseline_comparison.csv`、`baseline_equivalence.csv`、`paired_algorithm_differences.csv`
- 轨迹：`trajectory_steps.csv`、`trajectory_summary.csv`、`trajectory_consistency_audit.csv/json`、`selected_tasks.csv`
- 部署：`deployment_small.json`、`deployment_transformer_m.json`
- 训练缓存 provenance：`lane_cache_small.json`、`lane_cache_transformer_m.json`
- 图：`figures/*.png`；图数据：`figure_data/*.csv`

### 9.10 最终 commit SHA

`a753d706d2c3d26959fcacd0a9c4fdac44b2675b`（工作区有未提交改动）

## 10. 限制

- 单城市、单任务生成规则；结论不外推到新城市或异质部署。
- 只有 3 个种子，均值/标准差只用于方向性比较。
- 严格收敛判据未全部满足；报告同时给出最佳 Dev 检查点和末 1000 Episode 训练指标，不把“未收敛”包装成“收敛”。
- 导航由冻结 MEC 模型产生，关联算法不影响 SR/OSR/nDTW/SPL；这是实验设计，不是实现问题。
- Test-400 未读取，未参与任何选择或调参。

