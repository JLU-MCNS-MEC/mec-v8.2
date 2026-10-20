# Stage-2k — 基于五种冻结 T5 MEC 的 DQN 关联实验

日期：2026-10-20  
分支：`experiment/v83-spatial-foundation`  
仓库：`JLU-MCNS-MEC/mec-v8.2`

---

## 0. 结论摘要

- DQN 实际训练 **2000 / 4000 Episode**；最佳 Validation 在 **Ep800**
  （JointR 0.2235，固定 60 Episode），最后一次 Ep2000 为 0.1439。
- 按预注册早停规则在 Ep2000 停止，`training_result.json` 记录
  `converged=true`（协议标记）。但最近 6 次 Validation 无实质改善，
  最佳点在 800、之后下降，属于**平台化/退化**，不是有竞争力的稳定收敛。
- **训练时 Validation 的 split 用错了**：`stage2k_dqn.py` 把
  `load_split_specs("train")` 传给 `validation_specs`，所以 10 次
  “Validation”实际是 **train Episode + dev topology**。用该 train 子集
  可 0 差异复现 Ep800 的 0.2235；真实 Dev60 仅 0.1386，Dev180 为 0.1133。
  早停和 `best.pt` 选择都基于 in-sample 指标，不能作为 Dev 收敛依据。
- **完整 Dev Val 180**：DQN **JointR 0.1133**、SR 47.22%、nDTW 0.4821、
  E2E 147.53 ms。低于全部五种 Baseline：Heuristic 0.2617、Min-Latency
  0.2477、Max-Quality 0.2256、Random 0.2010、Oracle 0.4236。
- 状态审计：478-D 中的时延特征已正确进入，Mask、索引、归一化、上一步
  选中标志均通过检查，**没有状态实现错误**。
- 奖励审计：累计时延惩罚只有 JointR 时延惩罚的 **T/200**（均值 0.663、
  中位数 0.334）；训练回报平均高估 JointR 0.1235。前 2000 Episode 只有
  short/medium，**没有 long**（long 从第 2664 条才开始）。
- 行为审计：DQN 每步平均比最快 MEC 慢 **81.16 ms**，质量只比最高质量候选
  低 **0.098**；对 Min-Latency 有 114/180 Episode “更慢且 nDTW 未改善”，
  其中 **0 个** JointR 改善；38/180 Episode 的动作序列与 Min-Latency 完全相同，
  但 E2E 平均高 **75.73 ms**。
- 本轮**没有进入 Association Final Test 180**；提出“band 交错 + 时延
  divisor 150”的修正实验，但**未执行、未启动新的训练**。

---

## 1. 恢复来源与数据隔离

本轮是从中断现场继续，不是重新开始整套实验。

- 上次训练循环实际完成 **2000 / 4000 Episode**，写入了 `best.pt`、`last.pt`、
  `training_history.json`、`progress.json` 和 TensorBoard；随后在收尾绘图
  `plot_training_curves` 处崩溃。
- 根因：旧代码读取 `row["episode_return"]`，训练记录字段是 `row["return"]`，
  触发 `KeyError`。已修复并加入回归测试。
- 恢复脚本 `stage2k_finalize.py` 只补写 `training_result.json`、曲线和
  `progress.json`，**没有重训、没有改写 checkpoint、没有覆盖日志**。
- 本轮新增的状态/奖励/评估脚本只读取 Dev/Train 记录，**未读取 Association
  Final Test 180，也未读取 Stage-2j Final Test 300**。

数据隔离：

| 用途 | Episode seed | Topology seed | Episode 数 | 本轮是否读取 |
|---|---|---|---:|---|
| DQN Train | 5,000,000+ | 8,000,000+ | 4,000 | 只读已有训练记录 |
| Dev Validation | 6,000,000+ | 8,100,000+ | 180 | 是（评估） |
| Association Final Test | 7,000,000+ | 8,200,000+ | 180 | **否** |

---

## 2. 五种 MEC 冻结与质量先验

五种 Stage-2j `best.pt` 全部存在，`strict=True` 可加载；固定输入两次前向
最大绝对差为 0。`Transformer-M/best.pt` 是软链接，不是缺失。

| MEC | 参数量 | SHA256 前 16 位 | Dev Val mean nDTW | SR |
|---|---:|---|---:|---:|
| Small | 611,716 | `7349ec9d0c809a09` | 0.2314 | 16.11% |
| Transformer-S | 1,332,868 | `ceac7506b3b3e18` | 0.5469 | 53.33% |
| Medium | 1,991,044 | `1fcce0f33dfff368` | 0.2428 | 18.33% |
| Transformer-M | 3,033,732 | `57c88b7d71e1a7f1` | 0.5575 | 57.22% |
| Large | 5,248,932 | `310c9d2648dec7fe` | 0.5197 | 48.89% |

质量先验来自 Stage-2j Dev Val 180 闭环 mean nDTW；`quality_prior.json` 中
5 个 metrics SHA256 与 5 个 checkpoint SHA256 已逐一复核一致。

---

## 3. Baseline Dev Val 180

五种 Baseline 已完成，5 策略 × 180 = 900 条逐 Episode 记录。本轮**没有重跑
Baseline**。完整表见 `v83_spatial_stage2/STAGE2K_BASELINE_RESULTS.md`。

| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | 违约率 | 平均步数 | JointR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 55.56% | 0.5582 | 142.87 | 246.01 | 65.26% | 138.8 | 0.2010 |
| Min-Latency | 36.11% | 0.4192 | 68.60 | 94.18 | 14.75% | 158.0 | 0.2477 |
| Max-Quality | 52.78% | 0.5417 | 126.47 | 171.11 | 55.12% | 143.9 | 0.2256 |
| Heuristic / Weighted Greedy | 43.33% | 0.4810 | 87.72 | 134.96 | 23.94% | 154.2 | **0.2617** |
| Oracle（特权上界） | 66.67% | 0.6496 | 90.44 | 160.39 | 31.35% | 113.1 | 0.4236 |

Dev Val 180 的最强非特权 Baseline 是 **Heuristic（JointR 0.2617）**。

---

## 4. DQN 训练实况

### 4.1 配置

| 项 | 值 |
|---|---|
| 状态 / 动作 | 478-D / 20 个 MEC |
| 网络 | 标准 MLP `478→256→256→20` |
| γ / 学习率 / Batch | 0.99 / 1e-4 Adam / 128 |
| Replay capacity | 50,000 |
| ε | 1.0 → 0.05，200,000 步 |
| 实际训练 Episode | **2000 / 4000** |
| 总环境步数 / 更新次数 | 265,956 / 263,956 |
| 最终 ε | 0.05 |
| Seed | 20261031 |

### 4.2 训练时 Validation 曲线（实际为 train Episode）

> **关键问题：这 10 个点不是 Dev Val。** 训练脚本先执行
> `spec_payload = load_split_specs("train")`，随后
> `validation_specs(spec_payload)` 从 **train** 中取 6 Short / 10 Medium /
> 4 Long × 3 场景 = 60 个 Episode，topology 则使用 dev。复现见 §7.2：
> 用这 60 个 train Episode 可以**完全复现**记录（Δ=0）；真实 Dev60 只有
> 0.1386，Dev180 为 0.1133。因此下图和“最佳 Ep800”是 **in-sample** 指标，
> 不是 Dev 收敛证据。

| Episode | JointR | SR | nDTW | E2E (ms) | P95 (ms) | 平均步数 |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.1803 | 55.00% | 0.5519 | 148.64 | 243.21 | 151.6 |
| 400 | 0.1710 | 55.00% | 0.5521 | 152.48 | 242.69 | 150.2 |
| 600 | 0.1661 | 50.00% | 0.5491 | 153.20 | 241.66 | 157.6 |
| **800** | **0.2235** | **60.00%** | **0.5951** | **148.64** | **238.70** | **129.8** |
| 1000 | 0.2199 | 56.67% | 0.5778 | 143.19 | 236.16 | 134.6 |
| 1200 | 0.1890 | 51.67% | 0.5545 | 146.19 | 237.65 | 143.2 |
| 1400 | 0.1606 | 51.67% | 0.5279 | 146.94 | 241.21 | 165.3 |
| 1600 | 0.1866 | 51.67% | 0.5500 | 145.35 | 236.31 | 154.5 |
| 1800 | 0.1691 | 48.33% | 0.5223 | 141.29 | 238.40 | 152.2 |
| 2000 | 0.1439 | 48.33% | 0.5073 | 145.34 | 237.18 | 158.1 |

曲线：

- `runs/v83_spatial_stage2/stage2k/plots/dqn_episode_return.png`
- `runs/v83_spatial_stage2/stage2k/plots/dqn_return_moving_average_100.png`
- `runs/v83_spatial_stage2/stage2k/plots/dqn_validation_joint_reward.png`
- `runs/v83_spatial_stage2/stage2k/plots/dqn_validation_metrics.png`

TensorBoard 与 `training_history.json` 逐点核对一致：2000 条训练标量、
10 条 Validation 标量，最大绝对差 < 1.5e-5（浮点打印精度），没有补零或插值。

### 4.3 早停与 Checkpoint

`should_early_stop(2000, 6, joints)` 返回：

```
(True, "no JointR improvement over 5 validations and last-3 trend is flat/declining")
```

最近 3 次 JointR 为 `0.1866 → 0.1691 → 0.1439`，斜率 `-0.02135`。按预注册
规则，训练在 Ep2000 停止；`training_result.json` 记录 `converged=true`
（协议标记），最佳 Validation 在 **Ep800**。

Checkpoint 内容：

| 内容 | `best.pt`（Ep800） | `last.pt`（Ep2000） |
|---|---|---|
| 在线网络 / Target / Adam | 有 | 有 |
| 当前 Episode | 800 | 2000 |
| Replay Buffer | **无** | **无** |
| ε 状态 / RNG 状态 | **未单独保存** | **未单独保存** |

因此这不是严格断点续训；原训练脚本的 `--resume` 也会重新初始化 Replay Buffer。

---

## 5. 478-D 状态审计

实际调用链：`EpisodeRunner._build_step_data → candidate_features →
global_features → build_state`。在三种 Dev 场景各取 1 个 Episode，打印 20 个
候选 MEC 的归一化前后特征，并做单步动作检查。完整数据：

- `runs/v83_spatial_stage2/stage2k/state_audit/state_audit.json`
- `runs/v83_spatial_stage2/stage2k/state_audit/{balanced,quality_rich,tradeoff_stress}_step000.csv`

### 5.1 检查结果

| 检查项 | 结果 |
|---|---|
| `state_dim` | 478（128 视觉 + 10 全局 + 20×17 候选） |
| 字段顺序 | 与 `state_layout()` 完全一致 |
| Mask 与 500 m 覆盖半径 | 一致 |
| 动作索引 0..19 ↔ 拓扑节点 | 一致（单步 `transition_node_index == action`） |
| 不可用节点网络字段 | 上行/通信/推理/E2E 为 0；几何/带宽/算力/质量/类型保留 |
| 质量先验 | 与 `quality_prior.json` 一致 |
| 模型 one-hot | 与 `model_index` 一致 |
| E2E 特征 | `min(E2E_ms/100, 3.0)`；有限值；随节点真实变化 |
| 上一步选中标志 | 下一状态中仅选中节点置 1 |
| 全局 previous-E2E | 与上一步实际 E2E/100 一致 |

可用节点间归一化 E2E 的标准差：Balanced 0.498、Quality-rich 0.743、
Tradeoff-stress 1.031。Tradeoff-stress 有 2 个节点 E2E > 300 ms，被截断到
`3.0`；其余节点仍可区分。

**结论：时延特征已经正确进入 478-D 状态，没有发现状态实现错误。**

---

## 6. 奖励量级审计

实际奖励：

```
r_t = r_navigation_t − 0.25 × min(E2E_ms / 100, 3.0) / 200
```

其中 `r_navigation_t` 是 potential 差分，未折扣累计严格等于最终 nDTW
（用 900 条 Baseline 记录复核，`sum(nav_reward) − nDTW` 最大绝对误差为 0）。

### 6.1 已有 2000 条训练记录的统计

| 指标 | 数值 |
|---|---:|
| 导航奖励累计均值 / std | 0.5284 / 0.4064 |
| 时延惩罚累计均值 / std | −0.2416 / 0.2468 |
| 两者绝对值比 | 0.457 |
| 单步导航奖励 / 单步时延惩罚 | 0.00397 / −0.00182 |
| JointR 时延惩罚均值 | 0.3651 |
| 奖励累计时延惩罚 ÷ JointR 时延惩罚 | 均值 0.663，中位数 0.334 |
| 训练回报 − 同轨迹 JointR | +0.1235 |
| 训练回报与 Validation JointR 相关 | +0.270 |
| 训练回报与成功率相关 | +0.739 |
| 训练回报与平均 E2E 相关 | −0.074 |

原因：奖励把每步时延惩罚除以 200，而 JointR 使用 `mean(E2E)/100`。对 T 步
Episode，奖励累计时延惩罚恰好是 JointR 时延惩罚的 **T/200**。训练 Episode
步数中位数只有 67，因此典型 Episode 只拿到约 **1/3** 的时延惩罚；均值 133 步
也只有约 2/3。训练回报因此系统性高估 JointR。

### 6.2 训练 band 覆盖

| Band | 训练记录数 | 平均步数 | 中位步数 | 导航奖励 | 时延惩罚 | 累计惩罚 ÷ JointR |
|---|---:|---:|---:|---:|---:|---:|
| Short | 1,332 | 125.0 | 56.0 | 0.5323 | −0.2255 | 0.624 |
| Medium | 668 | 148.8 | 85.0 | 0.5208 | −0.2738 | 0.743 |
| Long | **0** | — | — | — | — | — |

**训练顺序按 band 分块**：前 2000 Episode 只有 short/medium；long 从训练顺序
第 2664 条才开始。早停发生在 2000，恰好发生在看到 long 之前。Validation 与
Dev 180 都包含 long，训练/验证分布不一致。

**结论：**
1. 时延特征在状态中正确存在，但奖励的累计时延惩罚相对 JointR 被
   `T/200` 系统性削弱，短 Episode 尤其明显。
2. 训练集在前 2000 Episode 没有 long，训练/验证分布错配。
3. 训练回报与 Validation JointR 的相关只有 0.270，训练回报上升不能可靠
   预测验证 JointR 改善。

## 7. DQN 与 Baseline 的完整 Dev Val 180 对比

评估对象：`runs/v83_spatial_stage2/stage2k/dqn/best.pt`（Ep800）。
评估数据：固定 Dev Val 180，greedy rollout；Baseline 复用已有 900 条记录。
场景、topology、Episode seeds、时延模型与 JointR 完全相同。
**未读取 Association Final Test。**

### 7.1 总体指标

| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | 违约率 | 平均步数 | JointR |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DQN best @800** | 47.22% | 0.4821 | 147.53 | 238.68 | 67.18% | 156.4 | **0.1133** |
| Random | 55.56% | 0.5582 | 142.87 | 246.01 | 65.26% | 138.8 | 0.2010 |
| Min-Latency | 36.11% | 0.4192 | **68.60** | **94.18** | **14.75%** | 158.0 | 0.2477 |
| Max-Quality | 52.78% | 0.5417 | 126.47 | 171.11 | 55.12% | 143.9 | 0.2256 |
| Heuristic | 43.33% | 0.4810 | 87.72 | 134.96 | 23.94% | 154.2 | **0.2617** |
| Oracle（特权上界） | **66.67%** | **0.6496** | 90.44 | 160.39 | 31.35% | 113.1 | 0.4236 |

DQN 的 nDTW 0.4821 与 Heuristic 0.4810 几乎相同，但 E2E 高 **59.82 ms**，
JointR 低 **0.1484**。DQN 低于全部五种 Baseline，包括 Random。

### 7.2 训练时 Validation split 复现（关键）

训练脚本 `stage2k_dqn.py` 执行：

```python
spec_payload = load_split_specs("train")
...
validation_specs_list = validation_specs(spec_payload)   # train episodes
...
{scenario: load_topology(scenario, "dev")}               # dev topology
```

因此 10 次“Validation”使用的是 **train Episode + dev topology**，不是
Dev Val Episode。用 `best.pt` 重放该 train 子集可以**完全复现** Ep800 记录：

| 评估集 | SR | nDTW | E2E (ms) | JointR | 平均步数 |
|---|---:|---:|---:|---:|---:|
| 训练记录 Ep800 “Validation” | 60.00% | 0.5951 | 148.64 | 0.2235 | 129.8 |
| 用 train 60 复现 | 60.00% | 0.5951 | 148.64 | 0.2235 | 129.8 |
| 真实 Dev60 子集 | 46.67% | 0.5225 | 153.56 | 0.1386 | 158.8 |
| **真实 Dev180（正式对比）** | **47.22%** | **0.4821** | **147.53** | **0.1133** | **156.4** |

复现的 SR / nDTW / E2E / 步数差异均为 **0**，确认训练时 validation 就是
train 子集。这意味着：

- `best.pt` 是按 **in-sample** JointR 选的；
- 早停规则也是在 train 子集上触发的；
- 报告里的 0.2235 / 0.1439 **不能当作 Dev Val 指标**；
- 真实 Dev Val 180 是 **0.1133**。

### 7.3 每步时延/质量选择行为

| 指标 | 值 |
|---|---:|
| 选择最快可用 MEC 的步比例 | **19.06%** |
| 选择最高质量可用 MEC 的步比例 | 31.77% |
| 选中 E2E − 当前最低 E2E，均值 | **+81.16 ms** |
| 选中 E2E − 当前最低 E2E，中位数 | +67.25 ms |
| 选中 E2E − 当前最低 E2E，P95 | +220.11 ms |
| 选中质量 − 当前最高质量，均值 | **−0.0985** |
| 选中质量 − 当前最高质量，中位数 | −0.0272 |

DQN 既不是时延最优（只有 19% 的步选最快），也不是质量最优（质量平均低
0.0985，只有 31.8% 的步选最高质量）；它大量选择“较慢但质量并非最高”的
MEC。

### 7.4 逐 Episode 配对比较（DQN − Baseline）

| Baseline | 配对 n | ΔJointR [95% CI] | ΔnDTW | ΔSR | ΔE2E (ms) | 胜/负/平 |
|---|---:|---|---:|---:|---:|---:|
| Random | 180 | **−0.0877** [−0.1382, −0.0393] | −0.0761 | −0.0833 | +4.66 | 79/101/0 |
| Min-Latency | 180 | **−0.1344** [−0.1907, −0.0770] | +0.0630 | +0.1111 | +78.93 | 36/144/0 |
| Max-Quality | 180 | **−0.1123** [−0.1651, −0.0605] | −0.0596 | −0.0556 | +21.07 | 64/116/0 |
| Heuristic | 180 | **−0.1484** [−0.2026, −0.0921] | +0.0012 | +0.0389 | +59.82 | 34/146/0 |

四个 ΔJointR 的 95% CI 全部在 0 以下。对 Heuristic：nDTW 只高 0.0012
（实质为 0），SR 高 3.89 pp，但 E2E 高 59.82 ms，净 JointR **−0.1484**。

### 7.5 更慢但质量更高 / 更慢且无改善（逐 Episode）

| 对照 | 更慢 Episode | 更慢且 nDTW 更高 | 其中 JointR 更高 | 更慢但 nDTW 未改善 | 其中 JointR 更高 |
|---|---:|---:|---:|---:|---:|
| Random | 100 | 32 | 16 | 68 | **0** |
| Min-Latency | 179 | 65 | 35 | 114 | **0** |
| Max-Quality | 119 | 34 | 16 | 85 | **0** |
| Heuristic | 167 | 52 | 24 | 115 | **0** |

对 Min-Latency：179/180 Episode 更慢；其中 114 个 nDTW 没有改善，这 114 个
**没有一个** JointR 更高。对 Heuristic：115 个“更慢且 nDTW 未改善”Episode
也**没有一个** JointR 更高。时延代价在大多数 Episode 是纯损失。

### 7.6 分部署条件

| 场景 | DQN JointR | Heuristic JointR | ΔJointR | DQN E2E (ms) | DQN SR |
|---|---:|---:|---:|---:|---:|
| Balanced / Original | 0.1104 | 0.2626 | **−0.1522** | 144.33 | 48.33% |
| Quality-rich | 0.2340 | 0.4238 | **−0.1898** | 109.37 | 53.33% |
| Tradeoff-stress | **−0.0046** | 0.0986 | **−0.1032** | **188.90** | 40.00% |

Tradeoff-stress 的 JointR 为负值（−0.0046），E2E 188.90 ms，是 DQN 最差的
场景。该场景正是“高质量节点慢、低质量节点快”的冲突设置，说明 DQN 没有学会
权衡。

### 7.7 MEC 类型选择频率（DQN Dev180）

| MEC | 步数 | 比例 |
|---|---:|---:|
| Transformer-S | 7,071 | 25.12% |
| Large | 6,583 | 23.39% |
| Transformer-M | 6,421 | 22.81% |
| Medium | 4,696 | 16.69% |
| Small | 3,373 | 11.98% |

前三名高质量模型合计约 **71.3%** 的步数。DQN 明显偏向高质量模型，但没有
为这种偏好付出足够的时延代价。

### 7.8 动作序列一致性

将 DQN 的逐 Step 导航动作与 Baseline 比对（按 Step 索引；轨迹不同会分叉，
因此这是描述性统计，不是因果估计）：

| 对照 | 动作序列完全相同的 Episode | 相同序列下 ΔnDTW | 相同序列下 ΔE2E | 平均前缀一致率 | 中位前缀一致率 |
|---|---:|---:|---:|---:|---:|
| Min-Latency | **38 / 180** | 0.0000 | **+75.73 ms** | 0.733 | 0.814 |
| Max-Quality | 42 / 180 | 0.0000 | +26.86 ms | 0.752 | 0.880 |
| Heuristic | 37 / 180 | 0.0000 | +65.20 ms | 0.714 | 0.822 |

在约 21%–23% 的 Episode 中，DQN 输出的导航动作序列与某个 Baseline 完全相同，
nDTW 差异为 0，但 E2E 更高。也就是说，**换 MEC 没有改变导航动作，只增加了
时延**。这直接解释了“更慢但 nDTW 没有改善”的大量 Episode。

### 7.9 小结

- DQN 在真实 Dev Val 180 上 **JointR 0.1133**，低于全部 Baseline；
- 它确实学到了一些导航质量（nDTW 0.4821 ≈ Heuristic 0.4810，SR 47.22% >
  43.33%），但代价是 E2E 高 59.82 ms，净 JointR 低 0.1484；
- 训练时“Validation”误用 train 子集，导致 0.2235 的 in-sample 指标掩盖了
  真实 Dev 性能；真实 Dev60 只有 0.1386；
- 动作一致性审计显示，大量 Episode 的 MEC 选择没有改变导航动作，却增加了
  时延；
- **不能把当前 DQN 送入 Association Final Test。**

---

## 8. 收敛判断与下一步

### 8.1 判定：Case C + 验证协议错误

- 预注册早停规则在 Ep2000 触发，`training_result.json` 的
  `converged=true` 只是**协议标记**。
- 但 10 次“Validation”使用的是 **train Episode**，不是 Dev Val；真实
  Dev Val 180 的 JointR 只有 **0.1133**。
- 最近 6 次“Validation”没有实质改善，最佳在 Ep800、之后下降；训练曲线
  属于**平台化/退化**。
- 因此：**不能声称 DQN 已在 Dev 上稳定收敛，也不能进入 Association
  Final Test。** 当前情况下应归为 Case C（平台化）并先修正协议。

### 8.2 根因（按优先级）

1. **Validation split 错配（代码 bug）**：训练脚本把
   `load_split_specs("train")` 传给 `validation_specs`，用 train Episode
   做验证，却用 dev topology。用该 train 子集可 0 差异复现 Ep800 的
   0.2235；真实 Dev60 只有 0.1386，Dev180 为 0.1133。最佳 checkpoint 和
   早停都基于 in-sample 指标。
2. **奖励时延项与 JointR 不一致**：每步惩罚除以 200，累计惩罚 =
   JointR 时延惩罚 × T/200；均值 0.663、中位数 0.334，短 Episode 严重
   低估时延。
3. **Band 训练覆盖错配**：训练顺序按 band 分块，前 2000 Episode 只有
   short/medium，long 从第 2664 条才开始；早停发生在看到 long 之前。
4. **行为层面没有学会权衡**：每步平均比最快 MEC 慢 81.16 ms，质量只比
   最高质量候选低 0.0985；对 Min-Latency 有 114/180 Episode “更慢且 nDTW
   未改善”，其中 **0 个** JointR 改善；38/180 Episode 动作序列与
   Min-Latency 完全相同但 E2E 高 75.73 ms。

### 8.3 已修复的代码问题（本轮只修代码，未重训）

- **Validation split 已修复**：`stage2k_dqn.py` 现在通过
  `dev_validation_specs()` 使用 `load_split_specs("dev")` +
  `validation_specs`，而不是 train specs；新增
  `test_13_dqn_validation_specs_are_dev_and_disjoint`，断言验证 Episode
  的 `split == "dev"`、数量为 60、band 为 6/10/4，且与 train UID 不相交。
- 旧代码仍在 git 历史 `ebec155` 中；现有 checkpoint、日志、TensorBoard
  和 `training_result.json` 没有改动。
- 本轮**没有用修复后的代码重新训练**；现有 2000 Episode 结果仍然是在旧
  validation split 下产生的，不能与新代码的结果混淆。

### 8.4 修正实验提案（未执行）

如果继续 Stage-2k，建议按以下顺序修正；**不要直接复用当前训练结果**：

1. **Band 交错训练顺序**：用固定 seed 对每个场景的 short/medium/long 做
   轮转或确定性洗牌，保证前 2000 Episode 覆盖三个 band；不改变 4000
   Episode 池、Episode seed、拓扑或奖励定义。
2. **时延系数**：把 `REWARD_DELAY_DIVISOR` 从 200 改为 **150**。理由：
   10 次 Validation 的平均步数为 149.7，150 使奖励累计时延惩罚与 JointR
   在验证长度上基本一致（当前 200 少算约 25%）。若希望按 band 精确对齐，
   可用 band 平均步数（short≈125、medium≈150、long≈175）作为
   band-specific divisor。该修改不引入成功终止奖励、失败惩罚、目标距离
   奖励或 Oracle 奖励。
3. **Warm Start**：从现有 `best.pt` 初始化，新建独立实验目录，**清空
   Replay Buffer**，重新记录训练配置；明确标记为 Warm Start，不是严格
   断点续训。累计训练不超过 4000 Episode。
4. **用完整 Dev Val 180 做最终比较**：60 Episode 的验证子集只能用于训练
   期监控；进入正式测试前必须在完整 Dev Val 180 上比较，且不能只看
   60 子集的 in-sample 指标。
5. **进入 Final Test 的条件**：完整 Dev Val 180 的 JointR 稳定超过
   Heuristic（0.2617），且最近几次验证没有明显退化；然后冻结 checkpoint、
   状态设计、奖励函数、质量先验和评估代码，只评估一次。

### 8.5 本轮明确没有做的事

- 没有继续训练 DQN，没有启动新的 4000 Episode 训练；
- 没有重训任何 MEC，没有重跑任何 Baseline；
- 没有读取 Association Final Test 180 或 Stage-2j Final Test 300；
- 没有修改现有 `best.pt` / `last.pt` / TensorBoard / 训练历史。

---

## 9. 产物与提交状态

### 9.1 训练与评估状态

| 项 | 值 |
|---|---|
| 实际训练 Episode | **2000 / 4000** |
| 最佳 Checkpoint | `runs/v83_spatial_stage2/stage2k/dqn/best.pt`（Ep800） |
| 最后 Checkpoint | `runs/v83_spatial_stage2/stage2k/dqn/last.pt`（Ep2000） |
| 训练时 Validation 最佳 | 0.2235（train 子集，**不是 Dev**） |
| 真实 Dev60 子集 | 0.1386 |
| **真实 Dev180** | **0.1133** |
| 协议 `converged` | `true`（早停规则触发） |
| 是否可进入 Final Test | **否** |
| Final Test 是否读取 | **否** |

### 9.2 本轮产物

- 训练曲线：`runs/v83_spatial_stage2/stage2k/plots/`（4 张）
- 状态审计：`runs/v83_spatial_stage2/stage2k/state_audit/`
- 奖励审计：`runs/v83_spatial_stage2/stage2k/reward_audit/`
- Validation split 复现：`runs/v83_spatial_stage2/stage2k/dqn/validation_split_audit/`
- DQN Dev180 评估：`runs/v83_spatial_stage2/stage2k/dqn/dev180_eval/`
- 动作一致性：`runs/v83_spatial_stage2/stage2k/dqn/dev180_eval/action_agreement.json`
- Baseline 完整结果：`runs/v83_spatial_stage2/stage2k/baselines/dev/`

### 9.3 提交与工作区

- 本地提交：本轮 Stage-2k 审计/评估提交（见 `git log -1`）。
- 远程：`origin/experiment/v83-spatial-foundation`；**未 push**。
- 工作区：提交后仅 `runs/`（已忽略）不进入版本库。
