# Stage-2s 实验协议：5×Small + 5×Transformer-M 异构 10 节点 MEC

日期：2026-10-25  
分支：`experiment/v83-spatial-foundation`  
仓库：`JLU-MCNS-MEC/mec-v8.2`  
实验根目录：`runs/v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/`（`.gitignore` 排除）  
可追踪产物：`v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/`  
父协议：Stage-2n / Stage-2o / Stage-2p / Stage-2r（只读，不修改、不覆盖）

本协议在正式训练启动前冻结。Test-400 全程不读取；Stage-2n/2o/2p/2r 的历史代码和结果不修改。

---

## 1. 目标与实验条件

在固定城市、固定任务生成规则、固定 Train-4000 / Dev-60 划分、固定 Pure-MEC v2
关联状态（不含额外视觉输入）下，部署 **10 个 MEC 节点**：

- 5 个节点加载冻结的 **Small** 检查点；
- 5 个节点加载冻结的 **Transformer-M** 检查点；
- 节点位置沿用 Stage-2r 的 **A10_fixed** 十个位置；
- 采用确定性、空间均衡的 5+5 模型分配；
- 关联算法：DQN、PPO；
- 奖励：方案 D；
- 随机种子：101、202、303。

共 2 算法 × 3 种子 = **6 个正式 run**，每个 **4000 Episodes**。

| run | 算法 | 种子 |
|---|---|---|
| DQN-s101 | DQN | 101 |
| DQN-s202 | DQN | 202 |
| DQN-s303 | DQN | 303 |
| PPO-s101 | PPO | 101 |
| PPO-s202 | PPO | 202 |
| PPO-s303 | PPO | 303 |

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
| 验证 | Dev-60，每 200 Episode，真实闭环 |
| 训练预算 | 4000 Episodes，不早停 |

### 2.1 A10_fixed 十个节点位置

与 Stage-2r 完全一致。从 Stage-2l `A_fixed` 的 20 个位置中按几何规则选出：

1. 先选离城市中心 `(750, 750)` 最近的 A_fixed 节点；
2. 反复选择“到已选节点集合的最小距离最大”的节点，平局取较小原始索引；
3. 按原始索引升序排列，编号 `M01..M10`。

选中原始索引：`[1, 2, 4, 6, 7, 8, 11, 12, 18, 19]`（0-based），
即 `G2, G3, G5, G7, G8, G9, G12, G13, G19, G20`。

### 2.2 确定性 5+5 空间均衡分配

规则（只使用几何与任务起点，不使用任何导航结果、成功率或策略行为）：

1. 枚举 A10 的全部 `C(10,5)=252` 个五节点子集及其补集；
2. 在确定性 `76×76` 均匀城市网格（0–1500 m，20 m 间距）上计算 500 m 覆盖率；
3. 在 Train-4000 + Dev-60 全部起点上计算 500 m 覆盖率与平均最近距离（仅用于空间均衡审计）；
4. 依次最大化：最小网格覆盖率、最小起点覆盖率；最小化：平均最近距离差、较差一侧的平均最近距离；最后按节点索引字典序；
5. 选中子集中包含“A10 选点种子节点（离城市中心最近节点）”的一侧分配给
   **Small**，补集分配给 **Transformer-M**；
6. 节点编号保持 A10 顺序 `M01..M10`。

分配结果：

| 节点 | A_fixed 原始索引 | 坐标 (m) | 模型 |
|---|---:|---|---|
| M01 | 1 | (61.5, 159.9) | Transformer-M |
| M02 | 2 | (309.3, 1229.7) | Transformer-M |
| M03 | 4 | (411.5, 259.6) | Small |
| M04 | 6 | (1117.3, 1254.9) | Transformer-M |
| M05 | 7 | (857.0, 463.1) | Transformer-M |
| M06 | 8 | (836.3, 949.7) | Small |
| M07 | 11 | (1305.0, 161.1) | Small |
| M08 | 12 | (893.7, 99.4) | Transformer-M |
| M09 | 18 | (123.6, 589.0) | Small |
| M10 | 19 | (1321.6, 689.1) | Small |

空间均衡审计：

- 网格覆盖率：Small = 0.8395，Transformer-M = 0.9153，差 0.0758；
- Train+Dev 起点覆盖率：Small = 0.9268，Transformer-M = 0.9581，差 0.0313；
- 平均最近起点距离：Small = 288.70 m，Transformer-M = 297.93 m，差 9.23 m；
- 全部 10 节点对 Train/Dev 起点覆盖率 100%。

**部署哈希**：

```text
stage2s-A10-5small-5transformer_m
sha256 = 1fce485ce6256a288fb0aecd525afbf2830467b5ee9f2918952af54148a081cd
```

（哈希覆盖节点位置、模型分配、名义/有效算力、带宽、覆盖半径、功率、源拓扑哈希与分配规则。）

### 2.3 节点参数与两个算力概念

所有 10 个节点保持 Stage-2r 的系统参数：

| 参数 | 值 |
|---|---:|
| 名义计算能力 | **2.0 TFLOP/s** |
| 推理有效计算吞吐率 | **0.2 TFLOP/s** |
| 带宽 | 30 MHz |
| 覆盖半径 | 500 m |
| 发射功率 | 2.0 W |

**两个算力概念严格分离**：

- `compute_capacity_flops_per_s = 2.0e12` 保留在拓扑、状态特征和日志中；
- `effective_inference_compute_flops_per_s = 0.2e12` **只用于推理时延**：

```text
inference_ms = 1000 * model_flops / 0.2e12
E2E_ms = extraction_ms + communication_ms + inference_ms
```

- 模型 FLOPs 由 `stage2j_model_flops(model)` 按架构解析计算，不手写 Small 或
  Transformer-M 的固定时延；
- 计算结果：Small 推理约 **0.0711 ms**，Transformer-M 约 **14.0730 ms**；
- 新的推理时延进入 E2E、194-D 状态特征（`inference_ms_norm`、`e2e_ms_norm`）
  和方案 D 奖励（`r_t=(1-e2e_t/100)/400`）。

### 2.4 冻结检查点

| 模型 | 检查点 | SHA256 |
|---|---|---|
| Small | `runs/v83_spatial_stage2/stage2j/Small/best.pt` | `7349ec9d…44e1f6` |
| Transformer-M | `runs/v83_spatial_stage2/stage2j/Transformer-M/best.pt` | `57c88b7d…73c6d` |

两个检查点均只读加载，`eval()`、`requires_grad=False`，不参与训练。

---

## 3. 真实异构闭环（本阶段核心要求）

**不使用 Stage-2r 的冻结 lane 缓存，也不使用任何固定轨迹缓存。**

每个训练 step：

1. 关联策略选择节点 `i`；
2. 读取该节点 `model_type`；
3. 由对应冻结模型（Small 或 Transformer-M）对当前观测产生导航动作；
4. 环境执行动作，更新 UAV 位姿、目标距离、可用节点和后续状态；
5. 用该节点实际的通信时延与模型推理时延计算 E2E；
6. 方案 D 奖励只由 E2E 计算；SR、OSR、nDTW、SPL 只用于单独评估。

因此，模型选择会真实影响：导航动作 → 轨迹 → 下一状态 / Mask → 后续关联决策。
同一任务在不同策略下的轨迹允许不同；不做任何“导航结果与关联无关”的假设。

---

## 4. 方案 D 奖励

记 Episode 实际步数 `T`，第 `t` 步 E2E 为 `e2e_t`，`J = 1 - mean_E2E/100`：

```text
u_t = 1 - e2e_t / 100
r_t = u_t / 400
r_T <- r_T + (1 - T/400) * J
R_D = 1 - mean_E2E / 100
```

- 逐步奖励只依赖当前 E2E；
- 终止修正只加到最后一个转移；
- 不添加成功奖励、Stop 奖励或导航质量奖励；
- 训练时逐步校验 `|R_D - (1 - mean_E2E/100)| <= 1e-8`，并校验由逐步
  E2E 重构的方案 D 奖励与闭式一致（容差 `1e-8`）。

---

## 5. 训练与验证

### 5.1 超参数（与 Stage-2n/2o/2p/2r 完全一致）

- DQN：MLP 194→256→256→10，replay 200k，batch 256，lr 3e-4，gamma 1.0，
  epsilon 线性衰减，target update 与梯度裁剪沿用 Stage-2k；
- PPO：actor-critic 194→256→256→{10,1}，rollout 2048，minibatch 256，
  10 epochs，clip 0.2，GAE λ=0.95，lr 3e-4，entropy 0.01，gamma 1.0。

### 5.2 验证与检查点

- 每 200 Episode 在固定 Dev-60 上做**真实闭环**验证；
- 第 4000 Episode 再验证一次；
- 训练日志每 10 Episode 落盘；`last_full.pt` 每 100 Episode 及每次验证落盘；
- 崩溃自动续训：从 `last_full.pt` 恢复模型/优化器/RNG，并截断历史中比
  检查点更新的行，避免跳过 Episode；
- 最佳检查点：**最低 Dev 平均 E2E**；E2E 完全相同时依次比较更高 SR、更早 Episode。

### 5.3 收敛判据（训练前冻结，沿用 Stage-2o/2p/2r）

```text
最近三次 Dev 验证的 max(E2E) - min(E2E) <= 0.01 * 最佳 E2E
且 最后一次 E2E <= 1.01 * 最佳 E2E
```

4000 Episodes 仍不满足则标记 `not_converged`。

---

## 6. 最终验证与 baseline

每个 run 使用 Dev 选出的 `best.pt`，在同一 Dev-60 任务池上评估：

- SR、OSR、nDTW、SPL；
- 平均 E2E、P95 E2E、方案 D Reward、时延违约率；
- 每个 seed 结果、3 seed 均值/标准差。

Baseline（同一 10 节点部署、同一真实模型推理时延）：

- **Random**：固定随机种子；
- **Nearest**：选择距离最近的可用节点；
- **Max-Uplink-Rate**：选择当前上行速率最大的可用节点；
- **Greedy-Min-E2E**：只依据**当前状态**下各可用节点的 `total_ms` 选择最小值。

Greedy-Min-E2E 不使用未来轨迹、目标位置、Oracle 动作或任何特权信息。

---

## 7. 统计、图表与轨迹

每个 run 保存：

- 总 Episodes、成功/失败 Episodes；
- 成功任务 Short/Medium/Long 数量；
- 失败任务 Short/Medium/Long 数量；
- Reward-MA100、E2E-MA100、E2E 滚动标准差；
- 每 200 Episode 的 Dev-60 指标（含 SR/OSR/nDTW/SPL/E2E/P95/RewardD/
  违约率/Small 与 Transformer-M 选择比例/推理与通信时延分解）。

图（300 dpi PNG + `figure_data/*.csv` 原始数据）：

1. DQN/PPO Reward-MA100 曲线；
2. DQN/PPO E2E-MA100 曲线；
3. Dev E2E 随训练 Episode 变化；
4. SR、OSR、nDTW、SPL、E2E、Reward 验证对比（含 baseline）；
5. 三个种子的变异性与稳定性；
6. Small/Transformer-M 选择比例、推理时延、通信时延、E2E 分解。

轨迹：

- DQN/PPO 各输出真实闭环轨迹图；
- 尽可能覆盖成功/失败的 Short/Medium/Long；
- 每条轨迹标注起点、终点、目标、成功状态、轨迹长度、Episode ID；
- 逐步记录 UAV 坐标、导航动作、MEC 编号与模型类型、通信/推理/E2E、
  累计奖励、目标距离；
- 统计 DQN/PPO 的 Small/Transformer-M 选择比例，以及模型选择对轨迹和任务结果的影响；
- 额外做一个“强制 Small / 强制 Transformer-M”的诊断对照（仅诊断，不进入预注册主比较）。

---

## 8. Smoke Test（正式训练前）

`smoke.py` 已完成并保存于 `smoke_report.json` / `SMOKE_TEST.md`：

- 两种模型均正确加载，检查点哈希匹配；
- 同观测双模型动作不同（Small=1，Transformer-M=0）；
- 选择不同模型后 UAV 位姿和下一 194-D 状态均改变；
- Small 推理 0.0711 ms，Transformer-M 14.0730 ms，E2E 包含推理项；
- 状态 194-D、动作 Mask 10-D、终止在 400 步内；
- 方案 D 完整 Episode 奖励误差 <1e-15；
- 训练模式 `full_closed_loop_heterogeneous`，无 lane 缓存。

14-Episode × DQN/PPO smoke 均完成，奖励恒等式误差 <1e-15。

---

## 9. 产物与隔离

- 原始 checkpoint、TensorBoard Event、逐 Episode 大日志：
  `runs/v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/`（`.gitignore` 排除）；
- 代码、协议、测试、汇总 CSV、PNG、轨迹数据、中文报告：
  `v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/`；
- 不覆盖 Stage-2n/2o/2p/2q/2r 的任何任务、检查点、日志或结果；
- 不停止、不覆盖服务器上已有的其他实验；
- Test-400 全程不读取，不参与训练、调参、收敛判断或检查点选择。

---

## 10. 预先声明的限制

1. 单城市、单任务生成规则、两个冻结导航模型；结论不外推到新城市或重新训练导航模型。
2. 只有 3 个随机种子；3 seeds 不足以做强统计推断，但可以检查种子间方向一致性。
3. 导航由冻结 MEC 模型产生，但**模型选择会改变导航轨迹**；因此不同算法/种子的
   SR/OSR/nDTW/SPL 允许不同，必须如实报告。
4. 有效推理吞吐 0.2 TFLOP/s 是显式设定，使得 Transformer-M 推理成本显著高于
   Small；E2E 可能超过 100 ms，方案 D 奖励可能为负。这是实验设计事实。
5. Test-400 不读取。
6. 代码版本：每个 run 的 `protocol.json` 记录训练启动时的 git commit 与 dirty 状态；
   tracked 协议与部署哈希在训练前已提交。

---

## 11. 代码版本记录

- 训练代码 commit：见 `runs/v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/<run>/protocol.json`
  的 `launch_environment.git.commit`；
- 协议冻结与部署哈希 commit：见本目录 Git 历史中 `Stage-2s: freeze protocol and deployment`
  提交；
- 部署哈希：`1fce485ce6256a288fb0aecd525afbf2830467b5ee9f2918952af54148a081cd`；
- 检查点哈希：Small `7349ec9d…44e1f6`，Transformer-M `57c88b7d…73c6d`。
