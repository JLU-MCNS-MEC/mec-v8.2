# Stage-2k 状态空间、奖励与实验设计

日期：2026-10-20  
分支：`experiment/v83-spatial-foundation`

本轮只训练 DQN 关联智能体。五种 Stage-2j T5 MEC（Small、
Transformer-S、Medium、Transformer-M、Large）全部冻结，不训练、不微调、
不重新初始化。Stage-2j 的 T5 输入契约保持不变：没有独立的 3×8×8
Heatmap 分支，也没有直接输入 5D Network State。

---

## 1. 数据隔离与固定清单

| 用途 | Episode seed 范围 | Topology seed 范围 | Episode 数 |
|---|---|---|---:|
| DQN Train | 5,000,000+（每个场景独立 100k 块） | 8,000,000+ | 4,000 |
| Dev Validation | 6,000,000+（每个场景独立 100k 块） | 8,100,000+ | 180 |
| Association Final Test | 7,000,000+（每个场景独立 100k 块） | 8,200,000+ | 180 |

- Train / Dev / Test 的 episode seed 与 topology seed 均不相交，并且与
  Stage-2j Train 600（1,000,000+）、Dev Val 180（2,000,000+）、Final Test
  300（3,000,000+）的 seed 集合不相交。
- `stage2k_build_specs.py` 生成并保存完整清单；
  `stage2k/specs/test_episodes.json` 在生成时即标记 `sealed: true`，
  但直到 Stage F 才被评估代码读取。
- 每个部署条件：Dev/Test 各 60 个 Episode，18 Short / 30 Medium / 12 Long。
- DQN 固定 Validation 使用 Dev 中每个场景前 20 个 Episode，共 60 个
  （6 Short / 10 Medium / 4 Long）。

### 1.1 三种部署场景

三种场景使用同一套 1500×1500 m、20 节点、500 m 覆盖半径、120 m 最小间距
的拓扑生成器，节点位置由独立 topology seed 固定。

- **Balanced / Original**：五种模型各 4 个节点；带宽
  5/10/20/30 MHz 与算力 0.5/1/2 TFLOPS 按原始生成器均衡分配。
- **Quality-rich**：按 Dev Val mean nDTW 排序，前 3 名模型各 5 个节点，
  后 2 名模型分配 3 + 2 个节点；前 3 名节点使用 2 TFLOPS 和 30 MHz，
  其余节点使用 1 TFLOPS 和 10 MHz。质量与低时延一致。
- **Tradeoff-stress**：同样的模型数量分布，但前 3 名模型使用
  **0.5 TFLOPS + 5 MHz**，后 2 名模型使用 **2 TFLOPS + 30 MHz**，
  使“高质量”与“低时延”冲突，迫使关联策略权衡。

三种场景的拓扑在 Dev 与 Test 使用不同的 topology seed。

实际 Dev Val 质量先验（Stage-2j Dev Val 180 closed-loop mean nDTW）：

| MEC | mean nDTW | SR |
|---|---:|---:|
| Transformer-M | 0.5575 | 57.22% |
| Transformer-S | 0.5469 | 53.33% |
| Large | 0.5197 | 48.89% |
| Medium | 0.2428 | 18.33% |
| Small | 0.2314 | 16.11% |

因此 Quality-rich 场景的前 3 名节点为 Transformer-M / Transformer-S /
Large；Tradeoff-stress 将这 3 类模型放在 0.5 TFLOPS + 5 MHz 节点上。

---

## 2. 478-D DQN 状态

```
128-D visual context
+ 10-D global context
+ 20 x 17-D candidate MEC features
= 478
```

### 2.1 128-D 视觉上下文

从当前视觉、目标视觉、历史视觉、GAB 与地图检索的可观测摘要构建 3114-D
向量，再用**只在 Train 600 natural 样本上拟合**的 PCA 压缩到 128-D：

```
current mean(384) + current max(384)
goal mean(384) + goal max(384)
history mean over valid slots(768)
history most-recent slot(768)
GAB(20) + map_feat(22)
= 3114
```

- PCA 拟合样本：Train 600 中随机 20,000 个 natural 状态（seed 20261031）。
- 标准化 mean/std 与 128 个主成分一起冻结；128 维解释方差为 **94.92%**。
- 不使用 Dev、Test 或 Stage-2j Final Test 拟合投影或标准化参数。
- 不把完整 576×384 Token、完整 16 步历史 Token 或完整 3×8×8 Heatmap
  拼进 DQN 状态。

### 2.2 10-D 全局上下文

1. `step / 400`
2. `history_mask.sum() / 16`
3. Current–Goal 最大 cosine 相似度
4. GAB confidence（`gab[16]`）
5. GAB entropy（`gab[17]`）
6. 上一步 E2E 时延 `/ 100 ms`，截断到 `[0, 3]`
7–10. 上一步导航动作 4D one-hot（Forward/Left/Right/Stop）

### 2.3 20 × 17-D 候选 MEC 特征

每个 MEC 节点：

1. 可用性（距离 ≤ 500 m 为 1）
2. UAV–MEC 相对 x 位移 `/ 500 m`
3. UAV–MEC 相对 y 位移 `/ 500 m`
4. 距离 `/ 500 m`
5. 带宽 `/ 30 MHz`
6. 算力 `/ 2 TFLOPS`
7. 上行速率 `/ 200 Mbps`
8. 通信时延 `/ 100 ms`
9. MEC 推理时延 `/ 100 ms`
10. E2E 时延 `/ 100 ms`，截断到 `[0, 3]`
11. Dev Val 质量先验（mean nDTW，天然在 `[0,1]`）
12. 上一步是否选择该节点（0/1）
13–17. 模型类型 one-hot：Small、Transformer-S、Medium、Transformer-M、Large

不可用节点的可用性、上行速率、通信/推理/E2E 字段为 0，但保留位置、
带宽、算力、质量与类型；动作 mask 禁止选择它们。若没有正常覆盖节点，
按环境既有规则启用最近节点，并在状态中把其可用性置 1。

不使用目标真实坐标、真实距离、Oracle Action、Oracle 参考轨迹或成功标签
作为 DQN 状态。

---

## 3. E2E 时延模型

每步上行负载按 **Stage-2j 八项模型输入的实际字节数**计算：

| 输入 | 字节 |
|---|---:|
| current_tokens 576×384 float16 | 442,368 |
| goal_tokens 3×384 float16 | 2,304 |
| history_vis 16×768 float16 | 24,576 |
| history_pose 16×10 float32 | 640 |
| history_action 16×4 float16 | 128 |
| history_mask 16 uint8 | 16 |
| gab 20 float32 | 80 |
| map_feat 22 float32 | 88 |
| **合计** | **470,200** |

假设每步完整上传这八项，不使用 MEC 侧历史缓存；这是保守口径。

```
E2E_ms = extraction_ms + transmission_ms + inference_ms
extraction_ms = visual_encoder_FLOPs / UAV_compute
transmission_ms = 470,200 × 8 / uplink_rate_bps × 1000
inference_ms = Stage-2j_model_FLOPs / node_compute
```

- UAV 算力沿用 `NETWORK.uav_compute_flops_per_s = 235.8 GFLOPS`；
  共享视觉编码器约 5.356 GFLOPs → **22.71 ms** 抽取时延。
- 节点带宽使用该节点实际 5/10/20/30 MHz；链路使用现有
  `LatencyModel.a2g_channel`（2 GHz、0.1 W、LOS/NLOS 模型）。
- 模型 FLOPs（batch 1，解析式）：

| MEC | 解析 FLOPs | 0.5 TFLOPS | 1 TFLOPS | 2 TFLOPS | RTX 5090 实测 |
|---|---:|---:|---:|---:|---:|
| Small | 0.014 G | 0.03 ms | 0.01 ms | 0.007 ms | 0.552 ms |
| Medium | 0.361 G | 0.72 ms | 0.36 ms | 0.18 ms | 0.702 ms |
| Transformer-S | 0.878 G | 1.76 ms | 0.88 ms | 0.44 ms | 0.850 ms |
| Transformer-M | 2.815 G | 5.63 ms | 2.81 ms | 1.41 ms | 0.957 ms |
| Large | 4.891 G | 9.78 ms | 4.89 ms | 2.45 ms | 1.085 ms |

实测列来自本轮同机 batch-1 审计，仅作为复杂度排序的 sanity check；
节点时延使用解析 FLOPs / 节点算力，而不是把 RTX 5090 实测值直接搬到
0.5/1/2 TFLOPS 节点上。

---

## 4. 奖励函数

沿用现有 `NavigationRewardTracker` 的逐步导航质量奖励 `r_navigation`，
加上当前选中 MEC 的预测 E2E 时延成本：

```
r_t = r_navigation - 0.25 * min(E2E_ms / 100, 3.0) / 200
```

- 不新增 `success_bonus`。
- 不新增失败终止惩罚。
- 不使用真实目标距离新奖励。
- 不使用 Oracle Action 正确性奖励。
- 不使用未来状态或测试标签作为奖励输入。

`NavigationRewardTracker` 使用参考轨迹计算逐步 nDTW 质量增量；这是现有的
**训练期参考轨迹奖励**，属于特权信号。本轮不增加其他 Oracle 或目标真值
信号，并在报告中明确标注这一点。奖励只进入 DQN 更新，不进入 478-D 状态。

---

## 5. DQN 网络与训练

### 5.1 网络

标准 MLP DQN（没有 Attention、GNN、Transformer、RNN）：

```
Linear(478, 256) → ReLU
Linear(256, 256) → ReLU
Linear(256, 20)
```

保留 Target Network、Replay Buffer、Adam 优化器与动作 Mask。

仓库中没有可直接沿用的 Stage-2k DQN 实现（仅有旧的 V8.2 关联评估脚本），
因此本轮新增标准 MLP DQN；没有引入任何额外网络模块。

### 5.2 超参数

| 项 | 值 |
|---|---|
| Seed | 20261031 |
| γ | 0.99 |
| Learning rate | 1e-4 (Adam) |
| Batch size | 128 |
| Replay capacity | 50,000 |
| Warm-up | 2,000 transitions |
| Target update | every 1,000 gradient steps |
| ε | 1.0 → 0.05 over 200,000 steps |
| Gradient clip | 10 |
| Loss | Smooth L1 |
| Min / max episodes | 800 / 4,000 |
| Validation | every 200 episodes, fixed 60 |
| Early-stop improvement | 0.002 JointR |
| Patience | 5 validations；再观察最多 3 个周期 |

### 5.3 提前停止

- 800 Episode 之前不因普通波动停止，除非 NaN、崩溃或实现错误。
- 800 之后，连续 5 次 Validation 未超过历史最佳 JointR 至少 0.002，
  且最近 3 次趋势平坦/下降 → 停止。
- 若仍有实质提升，继续训练。
- 若奖励明显震荡，不因单次低值停止；再观察最多 3 个 Validation 周期
  （即连续未提升达到 8 次）→ 停止。
- 4,000 Episode 为硬上限；未收敛则明确报告，不把训练 Return 误判为收敛。

只按 Validation JointR 保存最佳 DQN Checkpoint。

---

## 6. Baseline 定义

- **Random**：在可用节点中均匀随机；每个 Episode 固定随机种子。
- **Min-Latency**：选择预测 E2E 最低的可用节点。
- **Max-Quality**：选择 Dev Val 质量先验最高的节点；同质量选时延更低者。
- **Heuristic / Weighted Greedy**：
  `0.5 × quality + 0.5 × (1 - clip(E2E_ms/100, 0, 1))`；同分选时延更低者。
- **Oracle**：仅作为特权信息上界。用 `env.oracle_action()` 找到输出与
  oracle 一致的节点；若没有精确一致，则选择对 oracle 动作概率最高者，
  再以时延打破平局。Oracle 不参与 DQN 训练或状态构造。

不加入 Nearest、Max-Rate、QoS-Constrained 或其他额外 Baseline。

---

## 7. 评价

```
JointR = mean(nDTW) - 0.25 × mean(E2E_ms) / 100
```

所有 Baseline、DQN、Dev Validation 与 Association Final Test 使用同一
JointR、同一时延模型、同一 Episode 清单与同一拓扑。辅助指标：
SR、P95 E2E、时延违约率（E2E > 100 ms）、平均步数、模型使用分布。

Association Final Test 在 DQN 训练完成并冻结配置后只评估一次，不做权重更新、
阈值调整或重新训练。

---

## 8. 2026-10-20 恢复审计：状态与奖励

本节是在**不重新训练**的前提下，对实际运行代码和已有 2000 Episode 记录的审计。
详细数据：

- 状态审计：`runs/v83_spatial_stage2/stage2k/state_audit/state_audit.json`
- 奖励审计：`runs/v83_spatial_stage2/stage2k/reward_audit/reward_audit.json`
- 训练曲线：`runs/v83_spatial_stage2/stage2k/plots/`

### 8.1 478-D 状态审计

实际调用链为 `EpisodeRunner._build_step_data → candidate_features →
global_features → build_state`。在三种 Dev 场景各取 1 个 Episode，打印 20 个
候选 MEC 的归一化前后特征，并做单步动作检查。结果：

| 检查项 | 结果 |
|---|---|
| `state_dim` | 478（128 视觉 + 10 全局 + 20×17 候选） |
| 候选字段顺序 | 与 `state_layout()` 完全一致 |
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

### 8.2 奖励量级审计

实际奖励：

```
r_t = r_navigation_t − 0.25 × min(E2E_ms / 100, 3.0) / 200
```

其中 `r_navigation_t` 是 potential 差分，未折扣累计严格等于最终 nDTW
（用 900 条 Baseline 记录复核，`sum(nav_reward) − nDTW` 最大绝对误差为 0）。

已有 2000 条训练记录统计：

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

原因：奖励把每步时延惩罚除以 200，而 JointR 使用 mean(E2E)/100。对 T 步
Episode，奖励累计时延惩罚恰好是 JointR 时延惩罚的 **T/200**。训练 Episode
步数中位数只有 67，因此典型 Episode 只拿到约 **1/3** 的时延惩罚；均值 133 步
也只有约 2/3。训练回报因此系统性高估 JointR。

**额外发现：训练顺序按 band 分块。** 前 2000 Episode 只有
`short`（1332）和 `medium`（668），**没有 `long`**；long 从训练顺序第 2664 条
才开始。早停发生在 2000，恰好发生在看到 long 之前。Validation 与 Dev 180
都包含 long，训练/验证分布不一致。

### 8.3 Validation split 错配（关键发现）

`stage2k_dqn.py` 先执行 `spec_payload = load_split_specs("train")`，随后
`validation_specs(spec_payload)` 从 **train** 中取 60 个 Episode，topology
却用 dev。用该 train 子集可以 0 差异复现 Ep800 的 0.2235；真实 Dev60
只有 0.1386，Dev180 为 0.1133。最佳 checkpoint 和早停均基于 **in-sample**
指标，不能作为 Dev 收敛依据。复现数据：
`runs/v83_spatial_stage2/stage2k/dqn/validation_split_audit/`。

### 8.4 已修复的代码问题（本轮只修代码，未重训）

1. **Validation split 已修复**：`stage2k_dqn.py` 现在通过
   `dev_validation_specs()` 使用 `load_split_specs("dev")` +
   `validation_specs`；新增 `test_13_dqn_validation_specs_are_dev_and_disjoint`
   断言验证 Episode 的 `split == "dev"`、数量为 60、band 为 6/10/4，且与
   train UID 不相交。旧代码保留在 git 历史 `ebec155` 中，现有
   checkpoint、日志和 TensorBoard 未改动。本轮**没有用修复后的代码重新训练**。

### 8.5 修正实验提案（未执行）

在下一次训练前，建议继续修正以下问题：

1. **Band 交错训练顺序**：用固定 seed 对每个场景的 short/medium/long 做
   轮转或确定性洗牌，保证前 2000 Episode 覆盖三个 band；不改变 4000
   Episode 池、Episode seed、拓扑或奖励定义。
2. **时延系数**：把 `REWARD_DELAY_DIVISOR` 从 200 改为 **150**。
   理由：10 次 Validation 的平均步数为 149.7，150 使奖励累计时延惩罚与
   JointR 在验证长度上基本一致（当前 200 少算约 25%）；若要按 band 精确
   对齐，可用 band 平均步数（short≈125、medium≈150、long≈175）作为
   band-specific divisor。该修改不引入成功终止奖励、失败惩罚、目标距离
   奖励或 Oracle 奖励。
3. 修正实验必须从现有权重 Warm Start，新建独立目录，**清空 Replay
   Buffer**，重新记录配置；不能把含旧奖励值的 Replay Buffer 当作严格续训。
4. 修正实验的最终比较必须使用完整 Dev Val 180；在完整 Dev180 JointR
   稳定超过 Heuristic（0.2617）之前，不进入 Association Final Test 180。

以上 band/divisor/warm-start 修改本轮**没有执行**，也没有启动新的
4000 Episode 训练。
