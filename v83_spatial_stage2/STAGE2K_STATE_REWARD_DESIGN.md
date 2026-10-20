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
