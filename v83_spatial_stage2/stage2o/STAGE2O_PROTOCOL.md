# Stage-2o 实验协议：时延奖励设计对 DQN/PPO 收敛的影响

日期：2026-10-23  
分支：`experiment/v83-spatial-foundation`  
仓库：`JLU-MCNS-MEC/mec-v8.2`  
实验根目录：`runs/v83_spatial_stage2/stage2o/`（被 `.gitignore` 排除）  
可追踪产物：`v83_spatial_stage2/stage2o/`（代码、协议、CSV、图、报告）  
父协议：`v83_spatial_stage2/STAGE2N_PROTOCOL.md`（只读，不修改）

本协议在 Stage-2o 正式训练启动前冻结。Stage-2l、2m、2n 的任务、检查点、
日志与结果一律不覆盖；被封存的 Test-400 不读取。

---

## 1. 固定实验条件

本轮只改变 **奖励函数**，其余全部沿用 Stage-2n：

| 项目 | 固定值 |
|---|---|
| 城市 | `map_seed = 10,000,001`，160 个地标，1500 m × 1500 m |
| 拓扑 | Stage-2l `A_fixed` 的 20 个节点位置（拓扑种子 9,000,000） |
| MEC | 20 个完全同质节点，同一 Transformer-M（Stage-2j） |
| 权重 SHA256 | `57c88b7d71e1a7f17862aaa8429afa8a041cba5e747f744f3a7febf47f273c6d` |
| 部署 SHA256 | `4c0950cf6e438cff41454eee6daefb831fa984f425ec1a41cf3974257c6d1148` |
| 任务池 | Stage-2n Train 4000 / Dev 60（Test 400 封存） |
| 任务顺序 | Stage-2n 固定顺序（种子 20261022），六组实验共用 |
| 状态 | 384-D Pure-MEC v2 + 独立 20-D 可用性 Mask |
| 动作空间 | 20 个 MEC 节点 |
| 时延阈值 | `tau = 100 ms` |
| 最大决策步数 | `H = 400` |
| 折扣因子 | `gamma = 1.0` |
| 算法超参数 | 与 Stage-2n/Stage-2l 标准 DQN、PPO 完全一致 |

没有任何任务成功奖励、Stop 成功奖励、导航进展奖励、nDTW 奖励或其他
导航质量奖励。SR/SPL/nDTW 只作为导航质量合理性检查，不参与训练目标。

---

## 2. 三个奖励方案

记 Episode 实际步数 `T`，第 `t` 步的端到端时延为 `e2e_t`，运行均值为

```text
mean_t = mean_{t-1} + (e2e_t - mean_{t-1}) / t,   mean_0 = 0
```

Episode 平均时延与平均效用为

```text
mean_E2E = (1/T) * sum_t e2e_t
J        = 1 - mean_E2E / tau
```

### 方案 B：只移除原奖励的 0.25 系数

```text
r_t = -(mean_t - mean_{t-1}) / tau
R_B = sum_t r_t = -mean_E2E / tau
```

这是 Stage-2n 奖励 `-0.25/100 * (mean_t - mean_{t-1})` 去掉 0.25 后的
结果，逐步与 Episode 累计回报都精确等于原奖励的 4 倍。

### 方案 D：固定尺度逐步反馈 + 时延归一化终止修正

```text
u_t = 1 - e2e_t / tau
r_t = u_t / H                        （每个转移）
r_T <- r_T + (1 - T/H) * J           （仅 Episode 最后一个转移）
```

因此

```text
R_D = (1/H) * sum_t u_t + (1 - T/H) * J
    = (T/H) * (1 - mean_E2E/tau) + (1 - T/H) * J
    = J = 1 - mean_E2E / tau
```

终止修正只依赖实际时延与 Episode 长度，不依赖成功标签、目标位置或
任何导航质量量。

### 方案一：直接优化 Episode 平均回报

目标函数明确为

```text
J_episode = mean_t (1 - e2e_t / tau) = 1 - mean_E2E / tau
```

轨迹级实现：

```text
r_t = 0        (t < T)
r_T = J_episode
R_one = J_episode
```

DQN 与 PPO 的训练目标、价值目标、经验回放与优势估计都按该轨迹级回报
处理，而不是把 `1 - e2e_t/tau` 当作逐步奖励后继续使用 Stage-2n 的累计
回报目标。

### 关键恒等式与比较口径

```text
R_D = R_one = 1 + R_B
```

- D 与方案一的 Episode 累计回报完全相同，但逐步分配不同；
- B 与它们的 Episode 累计回报只差固定常数 `+1`；
- **不能只比较奖励曲线绝对高度**。所有结论以实际 E2E 时延、MA100 趋势、
  滚动标准差、收敛判据和 Dev-60 表现为准。奖励图同时标注公式和常数平移。

---

## 3. 实现要点与数学审计

### 3.1 终止转移的赋值

- 方案 B：每一步的奖励在 `apply()` 后立即可算，最后一步无需修正。
- 方案 D：非终止步奖励只依赖当前 `e2e_t`；Episode 结束时用实际 `T` 和
  `mean_E2E` 计算终止修正，**只加到最后一个转移**。
- 方案一：非终止步奖励恒为 0；Episode 结束时把 `J_episode` **只赋给最后
  一个转移**（`done=True`）。

因此不存在把未来信息泄漏到早期决策的问题：改变未来某一步的时延不会改变
任何更早的转移奖励。该性质由 `test_stage2o.py` 的
`NoFutureLeakageTests` 逐方案验证。

### 3.2 DQN 目标

- 终止转移：`done=1`，TD 目标为 `r_T`（方案一即 `J_episode`），不 bootstrap；
- 非终止转移：`gamma=1.0`，TD 目标为 `r_t + max_a' Q_target(s_{t+1}, a')`；
- 方案一的链式目标因此从终止的 `J_episode` 沿轨迹反向传播，固定点为
  轨迹上所有状态的动作价值等于 `J_episode`；
- 经验回放按完整轨迹存储，终止状态 `next_state=None` 在缓冲区中写零向量
  且 `done=1`，目标计算不会使用它。

单元测试用常量 Q 网络验证：终止转移的 target 精确等于 `r_T`，非终止转移
的 target 精确等于 `gamma * Q_target(s_{t+1})`。

### 3.3 PPO 优势估计

- 沿用 Stage-2n 的 PPO：`gamma=1.0`、`GAE lambda=0.95`、clip 0.2 等
  超参数不变；
- `RolloutBuffer.compute_gae` 在 `done` 处重置优势递推，多个 Episode 可以
  共用一个 rollout；
- 方案一下，`lambda=1.0` 时所有状态的值目标精确等于 `J_episode`；
  `lambda=0.95` 时终止状态的值目标仍精确为 `J_episode`，早期状态使用标准
  GAE 的 lambda 加权（这是 PPO 的方差—偏差折中，不改变 Episode 回报目标）。

单元测试验证了单 Episode、双 Episode 的 GAE 终止重置和 `J_episode` 目标。

### 3.4 每 Episode 一致性检查

训练时每个 Episode 都检查

```text
abs(R_scheme - expected_return(scheme, mean_E2E)) <= 1e-8
```

失败立即中止。Dev-60 的冻结 lane 评估与 Stage-2n 使用同一套代码，关联策略
不改变 MEC 导航动作，因此 SR/SPL/nDTW 必须与 Stage-2n 一致，只允许 E2E
时延不同。

---

## 4. 算法与超参数（不修改）

### DQN

```text
MLP 384 → 256 → 256 → 20
gamma = 1.0
replay = 50,000
target update = 1,000 update steps
batch = 128
epsilon 1.0 → 0.05 / 200,000 environment steps
Adam lr = 1e-4
grad clip = 10
训练从 total_steps >= 2,000 开始
```

### PPO

```text
actor-critic 384 → 256 → 256 → {20, 1}
gamma = 1.0
GAE lambda = 0.95
clip epsilon = 0.2
rollout = 2,048 steps
minibatch = 256
update epochs = 10
value coef = 0.5
entropy coef = 0.01
Adam lr = 3e-4
max grad norm = 0.5
target KL = 0.05
```

不引入 Double DQN、优先回放、奖励归一化、奖励裁剪或其他算法改动。

---

## 5. 训练与验证协议

- 6 个实验：`{B, D, one} × {DQN, PPO}`，统一 `seed=0`，每个 4000 Episodes；
- 同一 Train 4000 任务顺序、同一固定城市、同一部署、同一 Dev-60；
- 每 200 Episodes 在固定 Dev-60 上评估一次，第 4000 Episode 再评估一次；
- 每 200 Episodes 输出中文进展并保存训练日志、Dev 验证结果、
  `rounds/round_XXXX.json` 与 `rounds.csv`；
- 不早停；只有 NaN/Inf、Q 值发散、PPO KL 爆炸或程序错误会中止；
- Test-400 全程不读取。

### 收敛判据（沿用 Stage-2n，训练前冻结）

只用 Dev 平均 E2E 时延：

```text
最近三次 Dev 验证的 max(E2E) - min(E2E) <= 0.01 * 最佳 E2E
且 最后一次 E2E <= 1.01 * 最佳 E2E
```

达到 4000 Episodes 仍不满足则标记 `not_converged`。

### 最佳检查点选择

```text
最低 Dev 平均 E2E；E2E 完全相同时依次比较更高 SR、更早 Episode。
```

---

## 6. 每 200 Episodes 的汇报内容

每个实验在 Episode 200、400、…、4000 输出：

- 当前 Episode、训练耗时、GPU 利用率与显存；
- 最近 100 Episode 的平均奖励 MA100、平均 E2E 时延 MA100 及其标准差；
- 固定 Dev-60 的平均 E2E、P95、SR、SPL、nDTW；
- DQN：Q mean、Q max、Target、TD Loss；
- PPO：Value Loss、Explained Variance、Entropy、KL；
- 奖励震荡、Q 值漂移、价值估计异常、Dev 性能退化的判定。

---

## 7. 产物与隔离

- 原始 checkpoint、TensorBoard Event、逐 Episode 大日志：
  `runs/v83_spatial_stage2/stage2o/`（被 `.gitignore` 排除）；
- 代码：`v83_spatial_stage2/stage2o/*.py`；
- 测试：`v83_spatial_stage2/stage2o/test_stage2o.py`；
- 协议：本文件与 `evidence/protocol_freeze.json`；
- 逐 200 Episode 汇总 CSV、训练曲线 CSV/PNG、分析脚本、中文报告：
  `v83_spatial_stage2/stage2o/evidence/` 与 `STAGE2O_REPORT.md`；
- 不修改 Stage-2l、2m、2n 的任务、检查点、日志或结果；
- 不停止、不覆盖服务器上已有的 Stage-2l B/DQN、B/PPO（SIGSTOP 暂停态）
  进程。

---

## 8. 预先声明的限制

1. 单城市、单部署、单 MEC 检查点、单随机种子（seed 0）；本轮结果只用于
   初步筛选，不构成统计显著性证据。
2. 同质部署下导航动作与关联策略无关，SR/SPL/nDTW 不能用于区分奖励方案；
   方案差异只能来自 E2E 时延、收敛速度与稳定性。
3. 单种子下 DQN/PPO 的高方差可能掩盖方案差异；任何“下一轮扩到三个种子”
   的建议必须基于本轮实际分离度与稳定性证据。
4. Test-400 在方案冻结后的最终报告中才允许读取；本阶段不读取。
