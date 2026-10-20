# Stage-2k v2 — JointR 严格一致的奖励与训练设计

日期：2026-10-20  
分支：`experiment/v83-spatial-foundation`  
新运行：`runs/v83_spatial_stage2/stage2k_jointr_v2/`  
新 TensorBoard：`runs/tensorboard/association_stage2k_jointr_v2/`

本轮只重新训练 DQN；五种 T5 MEC 保持冻结，不重跑 Baseline，不读取
Association Final Test 180 或 Stage-2j Final Test 300。

---

## 1. 目标指标与统计口径

评价指标保持不变：

```
JointR = mean(nDTW) - 0.25 * mean(E2E_ms) / 100
```

`mean(E2E_ms)` 的口径与现有评价代码一致：**先对每个 Episode 求该 Episode
的平均 E2E，再对所有 Episode 求平均**；不是把所有 Step 混在一起求平均。
`mean(nDTW)` 同理，是逐 Episode nDTW 的平均。

因此对 Episode i：

```
JointR_i = nDTW_i - 0.25 * mean_E2E_i / 100
mean_i(JointR_i) = mean(nDTW) - 0.25 * mean_i(mean_E2E_i) / 100
```

新训练的每个 Episode 必须满足 `Episode Return = JointR_i`；对所有 Episode
求平均后，`mean(Episode Return) = JointR`。

---

## 2. 旧奖励的问题

旧奖励：

```
r_t = r_nav_t - 0.25 * min(E2E_t / 100, 3.0) / 200
```

`r_nav_t` 的定义是正确的：potential 差分，未折扣累计等于最终 nDTW。
但时延项被多除了 200。对 T 步 Episode：

```
sum_t r_delay_t = -0.25 * mean_E2E / 100 * (T / 200)
```

即累计时延惩罚只有 JointR 时延惩罚的 **T/200**。实测训练 Episode 步数中位数
67，典型 Episode 只拿到约 1/3 的时延惩罚；均值 133 步也只有约 2/3。
另外旧训练的 Validation 误用了 Train Episode。

---

## 3. 新逐步奖励

### 3.1 导航奖励（不变）

```
r_nav_t = Phi_t - Phi_{t-1},   Phi_0 = 0
sum_t r_nav_t = Phi_T = final_nDTW
```

不增加成功终止奖励、失败终止惩罚、目标距离奖励或 Oracle 奖励。

### 3.2 运行平均时延

第 t 步实际预测 E2E 为 `d_t` 毫秒，运行平均：

```
mean_0 = 0
mean_t = mean_{t-1} + (d_t - mean_{t-1}) / t
```

只使用已经执行的动作和实际产生的 E2E，不读取未来状态或特权信息。

### 3.3 时延奖励

```
r_delay_t = -0.25 / 100 * (mean_t - mean_{t-1})
          = -0.0025 * (d_t - mean_{t-1}) / t
```

不使用 `min(E2E/100, 3)` 截断，不放大/缩小系数。

### 3.4 总奖励与严格一致性

```
r_t = r_nav_t + r_delay_t
```

时延项望远镜求和：

```
sum_t r_delay_t = -0.0025 * (mean_T - mean_0)
                = -0.0025 * mean_E2E
```

因此：

```
sum_t r_t = final_nDTW - 0.0025 * mean_E2E
          = JointR_i
```

对所有 Episode 平均后，`mean(Episode Return) = JointR`。

---

## 4. 状态修改

478-D 状态总维度不变：

```
128-D visual + 10-D global + 20 x 17-D candidate = 478
```

10-D 全局上下文中的第 6 个特征（0-based 索引 5）由“上一步 E2E/100”替换为：

```
running_mean_E2E_ms / 100
```

- 不做 `min(..., 3)` 截断；
- 初始状态该字段为 0；
- 每步执行后先更新运行平均，再构建下一状态；
- 候选 MEC 的 17 维特征、Mask、动作空间和网络结构全部不变。

旧模式（`jointr_aligned=False`）仍保留上一步 E2E 字段，旧脚本/旧 Checkpoint
的行为不变。

---

## 5. DQN 折扣因子与目标

新训练使用：

```
gamma = 1.0
target = reward + (1 - done) * max_a Q_target(next_state, a)
```

终止状态不 Bootstrap。网络、Target Network、Replay Buffer、Adam、动作 Mask
和 MLP `478 → 256 → 256 → 20` 均保持不变；不引入 Double DQN、Attention、
GNN、Transformer 或其他新模块。

因为 `gamma=1.0` 可能提高价值估计方差，训练中每个 Episode 记录
Q mean/max、Loss、梯度范数；出现 NaN/Inf 或 Q 绝对值超过 `1e6` 时立即停止
并保存 Checkpoint，不自动修改算法。

---

## 6. 训练数据顺序

保留原 4000 个 Train Episode 的完整清单、Episode Seed、Topology Seed、
部署场景和 band 标签；只改变采样顺序。

新顺序：

- 3 个场景 × 3 个 band = 9 个组；
- 每组用固定 seed 做确定性 shuffle；
- 按 9 组轮转（round-robin）取样，直到 4000 个 Episode 全部取完。

验证结果：

| 组 | Episode 数 |
|---|---:|
| Balanced / Short | 444 |
| Balanced / Medium | 444 |
| Balanced / Long | 445 |
| Quality-rich / Short | 444 |
| Quality-rich / Medium | 444 |
| Quality-rich / Long | 445 |
| Tradeoff-stress / Short | 444 |
| Tradeoff-stress / Medium | 444 |
| Tradeoff-stress / Long | 446 |
| **合计** | **4,000** |

每个完整 200 Episode 窗口都覆盖 **9 / 9** 个组；没有新增、删除、重复或改变
任何 Episode。

---

## 7. 真实 Dev Validation

训练期间每 200 Episode 使用固定的 **60 个 Dev Episode**：

- Balanced / Quality-rich / Tradeoff-stress 各 20 个；
- 每个场景 6 Short / 10 Medium / 4 Long；
- `split == "dev"`；
- 与 Train Episode UID 交集为 0；
- Dev topology seeds：8,100,000 / 8,101,000 / 8,102,000；
- Train topology seeds 与 Dev 不同。

只根据这 60 个 Dev Episode 的 JointR 选择 `best.pt` 和触发早停。训练结束后，
最佳 Checkpoint 在完整 Dev Val 180 上评估一次。

---

## 8. 严格一致性测试

实现：`v83_spatial_stage2/test_stage2k_jointr_v2.py`。

覆盖：

1. 单步 Episode；
2. 多步 Episode；
3. 恒定 E2E；
4. 变化 E2E；
5. Short / Medium / Long（由 900 条 Baseline 真实轨迹覆盖）；
6. 最大 400 步；
7. 已保存的 900 条 Baseline Episode 记录；
8. Episode reset 后运行平均时延清零；
9. Replay Buffer / RNG 断点往返；
10. 真实 Dev Episode 的 closed-loop 一致性。

实际结果（23 项测试全部通过）：

| 检查 | 结果 |
|---|---:|
| 900 条 Baseline 的 `abs(Return − JointR)` 最大误差 | **3.33e-16** |
| 900 条 Baseline 的平均误差 | **4.76e-17** |
| `sum(nav) − nDTW` 最大误差 | **1.51e-16** |
| `sum(delay) + 0.25 × mean_E2E / 100` 最大误差 | **3.33e-16** |
| 冒烟 4 Episode 的 `abs(Return − JointR)` 最大误差 | **1.11e-16** |
| 2 个 Dev Episode 的闭环误差 | **2.22e-16 / 1.11e-16** |
| Validation 隔离 | 通过（60 Dev，交集 0） |
| 训练顺序覆盖 | 每个 200 窗口 9/9 组 |

训练循环中每个 Episode 还会断言：

```
abs(episode_return - episode_jointR) < 1e-6
```

不成立立即停止，并保存当前 Checkpoint。

---

## 9. Checkpoint 与断点续训

新运行目录：

```
runs/v83_spatial_stage2/stage2k_jointr_v2/
  checkpoints/
    last_full.pt   # 在线网络、Target、Optimizer、Replay Buffer、RNG、epsilon、episode、配置
    last.pt        # 最新模型 + 元数据（无 Replay Buffer）
    best.pt        # 最佳 Dev Validation 模型 + validation 记录
  training_order.json
  validation_isolation.json
  training_history.json
  training_result.json
  progress.json
```

- `last_full.pt` 每 50 Episode 和每次 Validation 保存一次；中断后最多回退
  50 Episode。
- `best.pt` 只在 Dev Validation JointR 超过历史最佳至少 0.002 时保存。
- 恢复时校验：奖励模式、gamma、epsilon、历史行数、Replay Buffer 和 RNG 状态。
- 大文件只保留在 `runs/`，不提交到 Git。

---

## 10. 收敛状态定义

训练结束时区分：

| 状态 | 含义 |
|---|---|
| `improving` | 最近 Validation JointR 仍在实质提升 |
| `plateaued` | 长时间没有改善，但尚未触发停止 |
| `early_stopped_no_improvement` | 因早停规则结束，未形成稳定平台 |
| `converged` | 最近至少 5 次 Validation 形成稳定平台（最近 3 次极差 ≤ 0.01，且最后值不低于最佳 −0.005），且数值稳定 |
| `not_converged` | 达到 4000 Episode 上限但没有稳定有效策略 |

`early_stop_triggered=true` **不会**自动令 `converged=true`；两者分别记录。

最终是否进入 Association Final Test 180，还要看完整 Dev Val 180 是否稳定超过
Heuristic 的 JointR 0.2617。本轮不读取 Final Test。
