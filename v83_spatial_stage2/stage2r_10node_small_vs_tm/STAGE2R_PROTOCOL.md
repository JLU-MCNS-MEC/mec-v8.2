# Stage-2r 实验协议：10 个 MEC 的 Small vs Transformer-M 同质部署

日期：2026-10-25  
分支：`experiment/v83-spatial-foundation`  
仓库：`JLU-MCNS-MEC/mec-v8.2`  
实验根目录：`runs/v83_spatial_stage2/stage2r_10node_small_vs_tm/`（`.gitignore` 排除）  
可追踪产物：`v83_spatial_stage2/stage2r_10node_small_vs_tm/`  
父协议：Stage-2n / Stage-2o / Stage-2p（只读，不覆盖）

本协议在正式训练启动前冻结。Test 集全程不读取；Stage-2n/2o/2p 的历史结果不覆盖。

---

## 1. 目标与实验条件

在固定城市、固定任务生成规则、固定训练/验证划分、固定 Pure-MEC v2
关联状态（不含额外视觉输入）下，把 MEC 数量从 20 个改为 **10 个**，比较：

- **部署模型**：10 个 MEC 全部 Small；10 个 MEC 全部 Transformer-M；
- **关联算法**：DQN、PPO；
- **奖励函数**：方案 D（Stage-2o/2p 已冻结）；
- **随机种子**：11、22、33（不再使用 0/1/2）。

共 4 个条件 × 3 个种子 = **12 个训练 run**，每个 4000 Episodes。

| 条件 | 部署 | 算法 | 种子 |
|---|---|---|---|
| S-DQN | 10×Small | DQN | 11/22/33 |
| S-PPO | 10×Small | PPO | 11/22/33 |
| T-DQN | 10×Transformer-M | DQN | 11/22/33 |
| T-PPO | 10×Transformer-M | PPO | 11/22/33 |

---

## 2. 固定条件

| 项目 | 固定值 |
|---|---|
| 城市 | `map_seed = 10,000,001`，160 个地标，1500 m × 1500 m |
| 任务池 | Stage-2n Train 4000 / Dev 60；Test 400 封存不读 |
| 任务顺序 | Stage-2n 固定顺序（种子 20261022） |
| 状态 | 194-D Pure-MEC v2：4 + 10 × 19 |
| 动作空间 | 10 个 MEC 节点；独立 10-D 可用性 Mask |
| 时延阈值 | `tau = 100 ms` |
| 最大步数 | `H = 400` |
| 折扣因子 | `gamma = 1.0` |
| 奖励 | 方案 D |
| 验证 | Dev-60，每 200 Episode |
| 训练预算 | 4000 Episodes，不早停 |

### 2.1 10 节点几何（A10_fixed）

从 Stage-2l `A_fixed` 的 20 个位置中确定性地选 10 个，规则仅使用几何：

1. 先选离城市中心 `(750, 750)` 最近的 A_fixed 节点；
2. 反复选择“到已选节点集合的最小距离最大”的节点，平局取较小原始索引；
3. 最后按原始索引升序排列，编号 `M01..M10`。

选中的 A_fixed 原始索引：`[1, 2, 4, 6, 7, 8, 11, 12, 18, 19]`（0-based），
即 `G2, G3, G5, G7, G8, G9, G12, G13, G19, G20`。
该子集对 151×151 城市网格的 500 m 覆盖率为 100%，Train-4000 + Dev-60
全部起点均在至少一个节点的 500 m 内；节点最小间距约 363.9 m。

### 2.2 两个同质部署

两个部署共享完全相同的 10 个位置、计算能力、带宽和覆盖半径，只改变
MEC 模型类型、检查点和 Dev 能力先验：

| 项目 | 10×Small | 10×Transformer-M |
|---|---|---|
| 检查点 | `stage2j/Small/best.pt` | `stage2j/Transformer-M/best.pt` |
| SHA256 | `7349ec9d…44e1f6` | `57c88b7d…73c6d` |
| Dev-180 SR | 16.11% | 57.22% |
| Dev-180 nDTW | 0.2314 | 0.5575 |
| 计算能力 | 2.0 TFLOP/s | 2.0 TFLOP/s |
| 带宽 | 30 MHz | 30 MHz |
| 覆盖半径 | 500 m | 500 m |

所有节点模型类型相同；E2E 差异只来自节点几何、模型推理 FLOPs 和
能力先验。

---

## 3. 方案 D 奖励

记 Episode 实际步数 `T`，第 `t` 步 E2E 为 `e2e_t`，`J = 1 - mean_E2E/100`：

```text
u_t = 1 - e2e_t / 100
r_t = u_t / 400
r_T <- r_T + (1 - T/400) * J
R_D = 1 - mean_E2E / 100
```

逐步奖励只依赖当前 E2E；终止修正只加到最后一个转移。实现直接复用
Stage-2o 的 `EpisodeRewardController`，训练时逐步校验
`|R_D - (1 - mean_E2E/100)| <= 1e-8`。

---

## 4. 训练与验证

### 4.0 精确冻结 lane 回放训练

导航动作由冻结 MEC 模型产生，关联策略只选择服务节点，因此同一任务的
真实轨迹、成功、步数、SPL、nDTW 与关联策略无关。Stage-2n 已用
`test_stage2n.py` 的 lane 等价性测试证明这一点。Stage-2r 在正式训练前
先用**完整环境**为每个模型生成 Train-4000 的冻结 lane 缓存：

- 每个 step 保存真实 pose、10×19 候选特征、10-D mask、各节点 E2E；
- 训练时用缓存重建 194-D 状态（running mean 由策略自己的选择更新），
  不重新跑视觉编码器/MEC 前向；
- 关联动作仍在线选择，E2E/Reward/状态完全由策略决定；
- 新增 `test_lane_equivalence.py`，在真实 Dev 任务上逐 step 比较
  全环境 runner 与 lane adapter 的 194-D 状态、E2E、成功、SPL、nDTW、
  步数和方案 D 奖励，要求一致（状态允许 float32 舍入误差 ≤1e-7）。

该加速只去掉重复的视觉/MEC 前向，不改变任务、环境动力学、动作、奖励
或评估协议。正式 run 必须使用 `--lane-cache`；全环境 smoke/对照仍保留。

### 4.1 超参数

- DQN/PPO 网络、优化器、epsilon/GAE/clip 等超参数与 Stage-2n/2o 完全一致：
  - DQN：MLP 194→256→256→10，replay 200k，batch 256，lr 3e-4，gamma 1.0；
  - PPO：actor-critic 194→256→256→{10,1}，rollout 2048，minibatch 256，
    10 epochs，clip 0.2，GAE λ=0.95，lr 3e-4，entropy 0.01，gamma 1.0。
- 每 200 Episode 在固定 Dev-60 上验证；第 4000 Episode 再验证一次。
- 训练日志每 10 Episode 落盘；`last_full.pt` 每 100 Episode 及每次验证落盘。
- 崩溃自动续训：从 `last_full.pt` 恢复模型/优化器/RNG，并截断历史中比
  检查点更新的行，避免跳过 Episode。

### 收敛判据（沿用 Stage-2o/2p，训练前冻结）

```text
最近三次 Dev 验证的 max(E2E) - min(E2E) <= 0.01 * 最佳 E2E
且 最后一次 E2E <= 1.01 * 最佳 E2E
```

4000 Episodes 仍不满足则标记 `not_converged`。

### 最佳检查点选择

```text
最低 Dev 平均 E2E；E2E 完全相同时依次比较更高 SR、更早 Episode。
```

---

## 5. 训练过程统计（每个 run）

- 总 Episode 数、成功数、失败数；
- 成功 Episode 中 short / medium / long 数量；
- 失败 Episode 中 short / medium / long 数量；
- 每 200 Episode 输出：当前 Episode、MA100 奖励、MA100 E2E、累计
  成功/失败、成功 band 分布、Dev-60 指标、DQN/PPO 诊断；
- 最终 4000 Episode 后在 `training_result.json` 和
  `success_failure_summary.csv` 中给出最终统计。

**注意（设计事实）**：训练任务顺序固定，且每个任务的成功/失败由冻结 MEC
导航模型决定，因此同一部署下的所有 DQN/PPO/seed run 会看到完全相同的
任务结果序列，累计成功/失败和 band 计数相同。差异只来自关联节点选择
造成的 E2E/Reward。这不是统计脚本错误，报告会明确说明。

---

## 6. 最终验证

每个 run 使用 Dev 选出的 `best.pt`，在固定 Dev-60 上评估：

- **SR、OSR、nDTW、E2E 时延、方案 D Reward**；
- 每个 seed 结果、3 seed 均值/标准差、与 baseline 的对比。

Baseline（同一验证任务池、同一部署重跑）：

- Random；
- Nearest；
- Max-Uplink-Rate；
- Greedy-Min-E2E。

同质部署下 Nearest / Max-Uplink-Rate / Greedy-Min-E2E 的排序相同；
Random 不同。所有 baseline 不使用目标位置、未来轨迹或 Oracle 动作。

---

## 7. 图表与轨迹

- 训练曲线：MA100 Reward、MA100 E2E、E2E 滚动标准差、Dev E2E；
- 最终验证：SR/OSR/nDTW/SPL/E2E/Reward 柱状图、baseline 对比、
  部署对比、seed 变异性、成功/失败 band 分布；
- 每个图都有 `figure_data/*.csv` 数据侧车；
- 4 组 UAV 轨迹图（Small-DQN、Small-PPO、TM-DQN、TM-PPO），
  每组 2×3：成功/失败 × short/medium/long；
- 轨迹选择规则：每个 band × outcome 内按 `(steps, episode_uid)` 排序取
  下中位 `(n-1)//2`；若某层为空，用同 band 最高 nDTW 的失败/最低 nDTW
  的成功替代，并在标题、CSV 和报告中标注；
- 轨迹一致性审计：同一部署下比较 DQN/PPO/Greedy-Min-E2E 的动作哈希、
  路径哈希、成功、步数和节点选择哈希。**导航由冻结 MEC 产生，预期路径、
  动作、成功完全一致，只有节点选择/E2E 可能不同**；如果不一致，按实现
  问题报告，不回避。

---

## 8. 产物与隔离

- 原始 checkpoint、TensorBoard Event、逐 Episode 大日志：
  `runs/v83_spatial_stage2/stage2r_10node_small_vs_tm/`（`.gitignore` 排除）；
- 代码、协议、测试、汇总 CSV、PNG、审计 JSON、中文报告：
  `v83_spatial_stage2/stage2r_10node_small_vs_tm/`；
- 不覆盖 Stage-2n/2o/2p/2q 的任何任务、检查点、日志或结果；
- 不停止、不覆盖服务器上已有的其他实验；
- Test-400 全程不读取，不参与训练、调参、收敛判断或检查点选择。

---

## 9. 预先声明的限制

1. 单城市、单任务生成规则、两个同质部署；结论不外推到异质算力/带宽或新城市。
2. 只有 3 个随机种子；3 seeds 仍不足以做强统计推断，但可以检查种子间方向一致性。
3. **导航动作由冻结 MEC 模型产生，DQN/PPO 只决定关联节点**。因此同一
   部署下不同关联策略的 SR/OSR/nDTW/SPL 必然相同；算法差异只能体现在
   E2E、Reward 和训练稳定性上。这是实验设计事实，不是实现缺陷。
4. Small 部署的导航能力显著低于 Transformer-M，因此 10×Small 的 SR
   会明显偏低；本阶段比较的是“同质部署 + 关联策略”的系统表现，而不是
   重新训练导航模型。
5. Test-400 不读取。
