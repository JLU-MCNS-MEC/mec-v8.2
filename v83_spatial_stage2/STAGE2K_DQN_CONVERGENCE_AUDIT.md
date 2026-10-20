# Stage-2k DQN 收敛与后期退化审计

日期：2026-10-20  
分支：`experiment/v83-spatial-foundation`  
仓库：`JLU-MCNS-MEC/mec-v8.2`  
性质：**只读诊断**。没有启动训练，没有修改 DQN 状态、奖励函数、网络结构或 Checkpoint，
没有读取 Association Final Test 180。

---

## 0. 结论摘要

- DQN 实际训练 **2000 / 4000 Episode**；`best.pt` 对应 **Episode 800**，
  `last.pt` 对应 **Episode 2000**，两个文件都存在且 metadata 可核验。
- 训练时的 10 次“Validation”实际是 **60 个 train Episode + dev topology**，
  不是 Dev Val；因此 `0.2235 → 0.1439` 是 in-sample 曲线，不是 Dev 泛化曲线。
- 完整 Dev Val 180：
  - `best.pt`（Ep800）：JointR **0.1133**、SR 47.22%、nDTW 0.4821、
    E2E 147.53 ms。
  - `last.pt`（Ep2000）：JointR **0.1570**、SR 52.78%、nDTW 0.5197、
    E2E 145.08 ms。
  - **反转：Ep2000 在真实 Dev180 上优于 Ep800**（ΔJointR +0.0438），
    尽管训练子集 Validation 曲线是下降的。`best.pt` 是按 train 子集选的，
    不是 Dev 最优。
- 固定 60 Episode 的 in-sample Validation 从 Ep800 到 Ep2000：
  JointR **−0.0796**、SR **−11.67 pp**、nDTW **−0.0878**；
  但 E2E **−3.30 ms**、P95 **−1.51 ms**（略好）。这个下降**没有转化为
  Dev 退化**；它主要是 train 子集上的 in-sample 现象。
- 真实 Dev180 的场景差异：Ep2000 在 Quality-rich（+0.0533）和
  Tradeoff-stress（+0.0946）明显更好，在 Balanced 更差（−0.0166）。
- 两个 Checkpoint 在 Dev180 上仍都低于全部 Baseline（Heuristic 0.2617、
  Min-Latency 0.2477、Max-Quality 0.2256、Random 0.2010）。
- 训练 100-Episode 移动平均从峰值 0.4054（Ep343）降到 0.2211（Ep2000）；
  但训练顺序按 band 分块（700–900 全是 short，1801–2000 全是 medium），
  这个下降**不能直接当作策略退化**，需要看固定 Validation 子集和 Dev180。
- Q 值审计（同一组 20 个固定 Dev 状态）：Ep800 平均 Q **19.73**，
  Ep2000 平均 Q **38.82**（约 2 倍）；Q 最高值 27.74 → 59.67；
  无 NaN/Inf；贪心动作一致率仅 **10%**。这是明显的值尺度漂移，
  但不能仅凭此证明发散原因。
- 没有发现 NaN/Inf 或数值崩溃；也没有发现需要修改算法的确定性错误。
  退化原因中，Q 值漂移、epsilon 退火后的分布变化、train 子集验证选择
  都属于**有数据支持的观察/假设**，不是已证明的因果结论。

---

## 1. 数据来源与曲线

### 1.1 数据来源

- 训练/验证记录：`runs/v83_spatial_stage2/stage2k/dqn/training_history.json`
- TensorBoard：`runs/tensorboard/association_stage2k/events.out.tfevents...`
  （原始 Event 文件未修改，未提交）
- 训练元数据：`runs/v83_spatial_stage2/stage2k/dqn/training_result.json`
- 进度：`runs/v83_spatial_stage2/stage2k/progress.json`
- Checkpoint：`runs/v83_spatial_stage2/stage2k/dqn/{best.pt,last.pt}`

TensorBoard 与 `training_history.json` 逐点核对：2000 条训练标量、10 条
Validation 标量，step 完全一致，最大绝对差 < 1.5e-5（浮点打印精度）。
没有补零、插值或推算缺失点。

**数据来源标注：**

- 训练曲线：`Train`，2000 个 Episode 的逐 Episode 记录。
- Validation 曲线：`Training-time validation`，60 个 **train Episode** +
  dev topology；**不是 Dev Val**。
- 所有 Validation 图都明确标注了这一点；最佳 Checkpoint 的实际 Episode
  是 **Ep800**。

### 1.2 五张图

| 图 | 路径 | 内容 |
|---|---|---|
| 原始奖励 | `runs/v83_spatial_stage2/stage2k/plots/dqn_reward_raw.png` | 逐 Episode 原始 `return`，保留全部波动 |
| 奖励 + MA100 | `.../dqn_reward_ma100.png` | 原始 return + 100-Episode 移动平均，标注峰值 Ep343 |
| Validation JointR | `.../dqn_validation_joint_reward.png` | 10 个 Validation 点，标注 Ep800/Ep2000 |
| 四项 Validation 指标 | `.../dqn_validation_four_metrics.png` | SR、nDTW、平均 E2E、JointR |
| 奖励分量 | `.../dqn_reward_components.png` | 导航奖励、时延惩罚（Episode 和与每步） |

归档副本：`v83_spatial_stage2/figures/stage2k_convergence/`。

### 1.3 曲线摘要

| 指标 | 数值 |
|---|---:|
| 训练 return 均值 / std | 0.2869 / 0.6112 |
| 训练 return 最小 / 最大 | −1.1356 / 0.9477 |
| 100-MA 峰值 | **0.4054 @ Ep343** |
| 100-MA @ Ep800 | 0.3506 |
| 100-MA @ Ep2000 | **0.2211** |
| Validation 最佳 | **0.2235 @ Ep800** |
| Validation 最后 | **0.1439 @ Ep2000** |

`training_history.json` 只记录了两个奖励分量：`navigation_reward` 和
`latency_penalty`。**没有记录成功终止奖励、失败惩罚或其他分量**，因此
`dqn_reward_components.png` 只画这两项，没有推算第三项。

---

## 2. Checkpoint 核验

### 2.1 文件与 metadata

| 项 | `best.pt` | `last.pt` |
|---|---|---|
| 存在 / 大小 | 是 / 3,107,081 B | 是 / 3,106,697 B |
| SHA256 | `325c45986b7183427c5c04f22f5d344d07a58e6481c3f55c7891ebb0d78728c0` | `703e31155ac94e487a77619d71a46f39e5957eeb518e370c6b5773a3ee56c375` |
| `episode` | **800** | **2000** |
| `best_joint` | 0.2235284622859482 | 0.2235284622859482 |
| `no_improve` | 未记录 | 6 |
| `total_steps` / `update_steps` | 98,883 / 96,883 | 265,956 / 263,956 |
| 在线网络 / Target / Adam | 有 / 有 / 有 | 有 / 有 / 有 |
| Replay Buffer | 无 | 无 |
| ε / RNG 状态 | 未单独保存 | 未单独保存 |
| `validation` 记录 | **有**（Ep800，train 子集） | **无** |
| 配置 | `mlp_478_256_256_20`, 478-D, 20 actions, γ=0.99, lr=1e-4, batch=128, replay=50,000, seed=20261031 | 同左 |
| `final_test_read` | false | false |

`best.pt` 的 `validation` 记录与 `training_history.json` 的 Ep800 记录一致；
但该记录是 **train 子集**，不是 Dev。`last.pt` 没有内嵌 validation 记录，
其对应 Validation 是 history 中的 Ep2000 记录。

### 2.2 完整 Dev Val 180 对照

`best.pt` 的 Dev180 评估已存在，且 summary 中的 checkpoint SHA256 与
`best.pt` 完全一致，直接复用；`last.pt` 在相同 Dev180、相同场景、相同
topology/episode seeds、相同时延模型上补跑一次评估（只评估，不训练）。

| 指标 | Ep800 `best.pt` | Ep2000 `last.pt` | Δ（last − best） |
|---|---:|---:|---:|
| SR | 47.22% | **52.78%** | **+5.56 pp** |
| nDTW | 0.4821 | **0.5197** | **+0.0376** |
| 平均 E2E (ms) | 147.53 | **145.08** | **−2.46** |
| JointR | 0.1133 | **0.1570** | **+0.0438** |
| OSR | **84.44%** | 76.67% | −7.78 pp |
| SPL | 0.4722 | **0.5276** | +0.0554 |
| P95 E2E (ms) | 238.68 | **237.10** | −1.58 |
| 时延违约率 | 67.18% | **66.21%** | −0.97 pp |
| 平均步数 | 156.4 | 146.9 | −9.45 |

Ep2000 在 Dev180 的 SR、nDTW、E2E、JointR、SPL、P95、违约率上都优于
Ep800；只有 OSR 更低（76.67% vs 84.44%）。**因此 `best.pt` 的 “best”
只对训练子集成立，对 Dev180 不成立。** 两者仍都低于所有 Baseline。

OSR / SPL / P95 / 违约率均来自逐 Episode 记录；没有缺失值。

---

## 3. 三个部署场景覆盖

Topology metadata（Dev split）：

| 场景 | Topology seed | 节点数 | `split` |
|---|---:|---:|---|
| Balanced / Original | **8,100,000** | 20 | dev |
| Quality-rich | **8,101,000** | 20 | dev |
| Tradeoff-stress | **8,102,000** | 20 | dev |

覆盖表（以逐 Episode 的 `episode_uid`、`seed`、`scenario` 字段为准）：

| 策略 | Balanced | Quality-rich | Tradeoff-stress | 总 Episode 数 |
|---|---:|---:|---:|---:|
| Random | 60 | 60 | 60 | 180 |
| Min-Latency | 60 | 60 | 60 | 180 |
| Max-Quality | 60 | 60 | 60 | 180 |
| Heuristic | 60 | 60 | 60 | 180 |
| Oracle | 60 | 60 | 60 | 180 |
| DQN@800 | 60 | 60 | 60 | 180 |
| DQN@2000 | 60 | 60 | 60 | 180 |

检查项：

- 每个策略的 Episode UID 集合必须与 Dev specs 的 180 个 UID 完全一致；
- 每条记录的 `scenario` 与 `seed` 必须与 specs 匹配；
- 每个策略三个场景各 60 条，总数 180；
- Baseline 与 DQN 使用同一 Episode 清单、同一场景 topology。

检查结果：

- 7 个策略/Checkpoint 全部为 **60 / 60 / 60，总计 180**；
- 每个策略的 Episode UID 集合都与 Dev specs 的 180 个 UID 完全一致；
- `scenario` 与 `seed` 不匹配数均为 **0**；
- Baseline 与 DQN@800 / DQN@2000 使用同一 Episode 清单；
- topology 文件均为 `split=dev`、20 节点、`final_test_read=false`；
- **没有缺失的策略或场景**，不需要补跑。

审计文件：`runs/v83_spatial_stage2/stage2k/dqn/scenario_coverage_audit/scenario_coverage.json`。

---

## 4. 后期退化分析

### 4.1 Return 与 Validation JointR 是否同步

| 指标 | Ep800 附近 | Ep2000 附近 | 变化 |
|---|---:|---:|---:|
| 训练 100-MA return | 0.3506 | 0.2211 | −0.1295 |
| in-sample Validation JointR | 0.2235 | 0.1439 | **−0.0796** |
| trailing-200 return 与 Validation JointR 相关 | — | — | **+0.270** |

两者在 **in-sample 记录上**方向一致（都下降），但相关性只有 **+0.27**；
训练 return 受 band 组成影响很大。更关键的是，**真实 Dev180 上 Ep2000
反而优于 Ep800**，所以 in-sample 的同步下降不能外推为 Dev 退化。
**不能只凭训练 return 曲线判断策略好坏。**

### 4.2 导航奖励与时延惩罚

固定 60 Episode 的 in-sample Validation：

| 指标 | Ep800 | Ep2000 | Δ |
|---|---:|---:|---:|
| JointR | 0.2235 | 0.1439 | **−0.0796** |
| SR | 60.00% | 48.33% | **−11.67 pp** |
| nDTW | 0.5951 | 0.5073 | **−0.0878** |
| E2E (ms) | 148.64 | 145.34 | −3.30 |
| P95 (ms) | 238.70 | 237.18 | −1.51 |
| 平均步数 | 129.8 | 158.1 | +28.3 |

**退化主要来自 nDTW/SR（导航质量），不是时延恶化。** E2E 和 P95 反而略好。

训练窗口（注意 band 组成不同）：

| 窗口 | 训练 Episode | return | 导航奖励 | 时延惩罚 | E2E (ms) | 步数 | Loss | ε | 成功率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 700–900 | 全部 short | 0.2726 | 0.5085 | −0.2359 | 144.72 | 129.1 | 0.2039 | 0.5268 | 44.78% |
| 1801–2000 | 全部 medium | 0.2268 | 0.5032 | −0.2764 | 146.72 | 148.4 | 0.7481 | 0.05 | 51.50% |

这两个窗口不是同一分布：700–900 全是 short，1801–2000 全是 medium。
因此训练窗口的 return/步数/时延差异**不能直接归因于策略退化**；固定
Validation 子集（同一批 60 Episode）才是更干净的比较。

### 4.3 MEC 类型使用率变化

固定 60 Episode 的 in-sample Validation：

| MEC | Ep800 | Ep2000 | Δ |
|---|---:|---:|---:|
| Transformer-M | 23.60% | 25.75% | +2.15 pp |
| Transformer-S | 23.48% | 25.56% | +2.08 pp |
| Large | **26.08%** | **19.15%** | **−6.93 pp** |
| Medium | 14.65% | 15.69% | +1.04 pp |
| Small | 12.19% | 13.85% | +1.66 pp |

Policy 明显减少了 Large 的使用，转向 Transformer-M / Transformer-S /
Medium / Small。这个变化与 in-sample nDTW 下降同时发生，但仅凭使用率
不能证明因果。

Dev180 上的使用率变化较小，但方向一致：

| MEC | Ep800 | Ep2000 | Δ |
|---|---:|---:|---:|
| Transformer-S | 25.12% | 26.11% | +0.99 pp |
| Large | 23.39% | 20.50% | **−2.89 pp** |
| Transformer-M | 22.81% | 22.35% | −0.46 pp |
| Medium | 16.69% | 19.67% | **+2.98 pp** |
| Small | 11.98% | 11.38% | −0.60 pp |

Dev180 上 Large 减少、Medium 增加；这与 Tradeoff-stress / Quality-rich 的
提升和 Balanced 的下降同时发生，说明策略变化是场景相关的，而不是单纯的
“模型质量下降”。

### 4.4 E2E / P95 / 时延违约率

- in-sample Validation：E2E 148.64 → 145.34 ms（−3.30），
  P95 238.70 → 237.18 ms（−1.51），**没有恶化**。
- 训练窗口 700–900 → 1801–2000：E2E 144.72 → 146.72 ms（+2.00），
  但窗口 band 不同，不能直接比较。
- Dev180 `best.pt`：E2E 147.53 ms、P95 238.68 ms、违约率 67.18%；
- Dev180 `last.pt`：E2E **145.08 ms**、P95 **237.10 ms**、违约率
  **66.21%**，三项都略好于 Ep800。

结论：没有证据表明 800→2000 的变化由时延恶化驱动。相反，固定 Validation
子集的 E2E/P95 略有改善，Dev180 的 E2E/P95/违约率也略好；变化主要来自
nDTW/SR，而且在不同场景方向不同。

### 4.5 ε、Loss、Q 值

| 指标 | Ep800 附近 | Ep2000 附近 |
|---|---:|---:|
| ε | 0.5268 | 0.0500 |
| Loss 均值 | 0.2039 | 0.7481 |
| Loss NaN/Inf | 无 | 无 |
| Q 均值（20 个固定 Dev 状态） | **19.73** | **38.82** |
| Q 最大值 | 27.74 | 59.67 |
| Q top-2 gap 均值 | 0.5461 | 1.8467 |
| Q 均值相关（两组状态） | — | 0.514 |
| 贪心动作一致率 | — | **10%** |

Q 值审计方法：在 `balanced/dev-00000` 上用确定性 Min-Latency 走 20 步，
收集同一组 478-D 状态，分别加载 `best.pt` 和 `last.pt` 的在线网络前向。
没有训练、没有改权重。

观察：

- 没有 NaN/Inf，也没有数值崩溃；
- Q 均值从 19.73 升到 38.82（约 2 倍），Q 最大值 27.74 → 59.67；
- 单步奖励量级约 0.002，Q 值远高于奖励量级；这与标准 DQN 的 max 目标
  过高估计**一致**，但本轮没有 double-DQN / Q-clip 对照，不能证明唯一原因；
- 贪心动作在 20 个固定状态上只有 10% 一致，说明 800→2000 的策略变化很大。

### 4.6 退化是否集中在某个部署场景

| 场景 | Ep800 JointR | Ep2000 JointR | ΔJointR | Ep800 E2E | Ep2000 E2E | Ep800 SR | Ep2000 SR | Ep800 nDTW | Ep2000 nDTW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced | **0.1104** | 0.0938 | **−0.0166** | 144.33 | **138.60** | **48.33%** | 43.33% | **0.4713** | 0.4403 |
| Quality-rich | 0.2340 | **0.2873** | **+0.0533** | 109.37 | 109.49 | 53.33% | **58.33%** | 0.5074 | **0.5610** |
| Tradeoff-stress | −0.0046 | **0.0900** | **+0.0946** | 188.90 | **187.14** | 40.00% | **56.67%** | 0.4677 | **0.5579** |
| **总体** | 0.1133 | **0.1570** | **+0.0438** | 147.53 | **145.08** | 47.22% | **52.78%** | 0.4821 | **0.5197** |

结论：**不是全局退化，而是场景特定变化。**

- **Tradeoff-stress 大幅改善**：JointR −0.0046 → 0.0900，SR 40.00% →
  56.67%，nDTW 0.4677 → 0.5579，E2E 还略好（188.90 → 187.14 ms）。
- **Quality-rich 改善**：JointR 0.2340 → 0.2873，SR +5 pp，nDTW +0.0536，
  E2E 基本不变。
- **Balanced 变差**：JointR 0.1104 → 0.0938，SR −5 pp，nDTW −0.0310；
  E2E 虽然从 144.33 降到 138.60 ms，但质量下降更多。
- 总体 Dev180 由 Balanced 的下降与另外两个场景的提升共同决定，
  **净效果是提升**。

这也说明：固定 train 子集 Validation 的下降不能外推到 Dev；它的场景/
band 组成与 Dev180 不同，且是 in-sample。

### 4.7 事实与假设

**有数据支持的事实：**

1. 固定 60 Episode in-sample Validation 的 JointR、SR、nDTW 从 Ep800 到
   Ep2000 明显下降；
2. 同期的 E2E/P95 没有恶化，反而略好；
3. **真实 Dev180 上 Ep2000 优于 Ep800**：JointR +0.0438、SR +5.56 pp、
   nDTW +0.0376、E2E −2.46 ms；
4. Dev180 的场景变化是分裂的：Balanced −0.0166，Quality-rich +0.0533，
   Tradeoff-stress +0.0946；
5. MEC 使用率从 Large 转向其他模型；
6. Loss 上升、ε 降到 0.05、Q 值约 2 倍、贪心动作一致率仅 10%；
7. 没有 NaN/Inf 或数值崩溃；
8. 两个 Checkpoint 仍都低于所有 Baseline。

**尚未证明的假设（不修改算法）：**

1. **Q 值过高估计/漂移**：标准 DQN 的 `max` 目标 + 无 double-DQN/Q-clip，
   可能导致 Q 尺度上升；但 Ep2000 在 Dev 上反而更好，因此 Q 尺度上升
   本身**不是**退化的充分证据，需要对照实验判断是否过度估计。
2. **ε 退火后的分布变化**：训练顺序按 band 分块，后期主要是 medium；
   策略在 medium 上更新更多，可能改变行为。
3. **in-sample Validation 与 Dev 方向相反的原因**：固定 train 子集的
   场景/band 组成与 Dev180 不同，且是 in-sample；策略可能向
   Tradeoff-stress/Quality-rich 有利的方向移动，但这在 train 子集上
   表现为下降。该解释有场景数据支持，但仍是假设。
4. **导航质量与时延权衡没有学好**：之前的审计已显示 DQN 大量选择较慢
   MEC 但 nDTW 没有改善；这解释了整体 Dev180 仍低于 Baseline，
   但不是 800→2000 变化的原因。

以上假设都需要新的对照实验验证；本轮不修改状态、奖励或网络结构。

---

## 5. 主要发现

1. `best.pt` 确实是 Ep800，`last.pt` 确实是 Ep2000；两者的 hash、配置、
   step 数、Target/Optimizer 都存在，未发现 checkpoint 缺失或损坏。
2. 训练时 Validation 是 **train Episode**，不是 Dev；0.2235 / 0.1439
   是 in-sample 曲线，不能当作 Dev 泛化指标。
3. **真实 Dev180 上，Ep2000 反而优于 Ep800**：JointR 0.1570 vs 0.1133，
   SR 52.78% vs 47.22%，nDTW 0.5197 vs 0.4821，E2E 145.08 vs 147.53 ms。
   `best.pt` 的 “best” 只对 train 子集成立；按 Dev 选，应该选 Ep2000。
4. 800→2000 的 in-sample Validation 下降主要是 **nDTW/SR 下降**，
   E2E/P95 没有恶化；这个下降**没有转化为 Dev 退化**。训练 return 的下降
   还受 band 组成影响（700–900 全是 short，1801–2000 全是 medium），
   不能单独作为退化证据。
5. Dev180 的变化是**场景特定**的：Balanced 变差（−0.0166），
   Quality-rich（+0.0533）和 Tradeoff-stress（+0.0946）变好；
   总体净效果是提升。
6. MEC 使用率从 Large 明显转向其他模型；Q 值约翻倍、贪心动作一致率
   仅 10%，说明策略和价值尺度都发生了较大变化。
7. 没有 NaN/Inf、没有 checkpoint 损坏、没有需要立即修复的数值错误；
   但 Q 值尺度异常和 train 子集验证选择仍是需要优先调查的风险点。
8. 两个 Checkpoint 在 Dev180 上仍都低于全部 Baseline
   （Heuristic 0.2617、Min-Latency 0.2477、Max-Quality 0.2256、
   Random 0.2010）；本轮没有继续训练、没有进入 Final Test、
   没有修改奖励/状态/网络。

---

## 6. 产物与提交

### 6.1 曲线

- `runs/v83_spatial_stage2/stage2k/plots/dqn_reward_raw.png`
- `runs/v83_spatial_stage2/stage2k/plots/dqn_reward_ma100.png`
- `runs/v83_spatial_stage2/stage2k/plots/dqn_validation_joint_reward.png`
- `runs/v83_spatial_stage2/stage2k/plots/dqn_validation_four_metrics.png`
- `runs/v83_spatial_stage2/stage2k/plots/dqn_reward_components.png`

归档副本：`v83_spatial_stage2/figures/stage2k_convergence/`。

### 6.2 审计数据

- 曲线摘要：`runs/v83_spatial_stage2/stage2k/plots/convergence_curve_summary.json`
- Q 值审计：`runs/v83_spatial_stage2/stage2k/dqn/q_value_audit/q_value_audit.json`
- 场景覆盖：`runs/v83_spatial_stage2/stage2k/dqn/scenario_coverage_audit/scenario_coverage.json`
- Dev180 best：`runs/v83_spatial_stage2/stage2k/dqn/dev180_eval/summary.json`
- Dev180 last：`runs/v83_spatial_stage2/stage2k/dqn/dev180_eval_last/summary.json`
- 退化汇总：`runs/v83_spatial_stage2/stage2k/dqn/convergence_audit/convergence_audit.json`

### 6.3 提交

- 报告、曲线图与只读审计脚本提交到当前分支；
- 不提交权重、Replay Buffer、TensorBoard 原始 Event 文件。
