# Stage-2n 实验协议：固定城市、同质 MEC、纯时延关联

日期：2026-10-22  
分支：`experiment/v83-spatial-foundation`  
仓库：`JLU-MCNS-MEC/mec-v8.2`  
实验根目录：`runs/v83_spatial_stage2/stage2n_latency_only/`  
冻结文件：`runs/v83_spatial_stage2/stage2n_latency_only/protocol_freeze.json`  
冻结协议 SHA256：`e5b8faa4ddb8ade68bbcc5101c401ded845e9404e3f050409f6ce59d2184c1df`

本协议在 Stage-2n 正式训练启动前冻结。Stage-2l/2m 的任务、检查点、日志与
结果一律不覆盖；Stage-2j 的 Transformer-M 检查点只读加载，不重新训练。

---

## 1. 目标

在**一张固定城市地图**、**一套完全同质的 20 节点 MEC 部署**上，只训练
DQN/PPO 关联智能体，并把奖励从 JointR 改为**仅 E2E 时延**：

1. 固定城市与任务生成规则沿用 Stage-2l，但 Stage-2n 使用独立任务清单与
   种子段（Train 4000 / Dev 60 / 封存 Test 400）；
2. 20 个 MEC 节点位置不同，其余全部相同：同一 Transformer-M 权重、同一
   算力、同一带宽、同一覆盖半径、同一通信参数；
3. MEC 仍使用正常 RGB、目标图像与历史视觉输入完成 VLN 推理；
   **只有 DQN/PPO 关联智能体不接收任何视觉信息**；
4. 删除导航进展 / nDTW 奖励，只保留 E2E 时延奖励；
5. 先确认奖励恒等式、Mask、状态维度与数值正常，再跑 4000 Episodes ×
   3 seeds × 2 算法；
6. 模型与超参数冻结后，在同一个封存 Test-400 上公平比较 Random、
   Nearest、Max-Uplink-Rate、Greedy-Min-E2E、DQN、PPO。

---

## 2. 固定城市与任务

### 2.1 城市

```text
map_seed = 10,000,001
```

该城市与 Stage-2l 第一阶段相同（160 个地标，1500 m × 1500 m），城市指纹：

```text
392500aa0df1d87b341f2fbd3df2fe4f2dc0405624a1ecc91cd33eca616bb182
```

所有 Stage-2n 任务共享这一张地图；任务之间不会重建、新增或修改地标。
复用该地图的理由是：Stage-2j Transformer-M 的闭环记录是在这一视觉分布上
测得的，固定地图可避免把“换地图”与“换奖励/换部署”混在一起。

### 2.2 任务生成与验收

复用 `stage2l_fixed_city.FixedCityEnv` 与 Stage-2l 的 oracle 验收规则：

- 目标从城市已有 160 个地标中选择；
- UAV 起点、初始航向、目标按 `task_seed` 采样；
- 用 oracle expert 模拟到 Stop，只有成功且专家步数落在 Band 范围内才接受；
- Short：150–250 m，30–80 步；
- Medium：250–400 m，50–115 步；
- Long：400–550 m，80–160 步；
- Stop 成功半径 20 m；未 Stop 最多 400 步。

### 2.3 数量与划分

| Split | 总数 | Short | Medium | Long | task_seed base | 封存 |
|---|---:|---:|---:|---:|---:|---|
| Train | 4000 | 1334 | 1333 | 1333 | 21,000,000 | 否 |
| Dev | 60 | 20 | 20 | 20 | 22,000,000 | 否 |
| Test | 400 | 134 | 133 | 133 | 23,000,000 | 是 |

- Task ID 前缀：`stage2n-{split}-{index:05d}`；
- Train/Dev/Test 的 task seed 与 task ID 两两不相交，并与 Stage-2l 的
  11M/12M/13M 池做交叉审计（`cross_stage_isolation.json`，全部通过）；
- Train 任务池使用固定种子 `20261022` 全局随机打乱；DQN 与 PPO 的 3 个
  seed 共用同一任务顺序；
- 每条任务保存起点、`start_pose`、目标、Band、专家步数、城市指纹、
  单任务 SHA256；任务池保存 `tasks_sha256` 与文件 SHA256。

### 2.4 Test 隔离

- Test-400 在构建时写入 `sealed: true`；
- 训练器只加载 Train/Dev；`load_task_pool("test")` 默认拒绝；
- 只有最终评估器带 `--allow-final-test` 才能读取，并写出
  `evaluation/test/test_read.json`；
- Test 不参与训练、超参数选择、收敛判断或最佳检查点选择。

---

## 3. 完全同质的 MEC 部署

### 3.1 节点位置

复用 Stage-2l `A_fixed` 的 20 个位置（拓扑种子 9,000,000），但**去除 B/C
多拓扑和场景混合**。位置文件 SHA256：

```text
29a7d91d77d65b07787189ec95dcf893a5723ea64207927f8675276bb85a7e56
```

### 3.2 完全相同的节点参数

| 项目 | 值 |
|---|---|
| 模型 | Transformer-M（Stage-2j，3,033,732 参数） |
| 权重 SHA256 | `57c88b7d71e1a7f17862aaa8429afa8a041cba5e747f744f3a7febf47f273c6d` |
| 算力 | 2.0 TFLOPS / 节点 |
| 带宽 | 30 MHz / 节点 |
| 覆盖半径 | 500 m |
| 发射功率 | 0.1 W（全局通信参数） |
| 部署 SHA256 | `4c0950cf6e438cff41454eee6daefb831fa984f425ec1a41cf3974257c6d1148` |

每个节点只有 `(x, y)` 不同；模型类型、模型索引、质量先验、算力、带宽、
覆盖半径全部相同。构建时做 9 项同质性审计（`homogeneity_checks`），全部
通过。

### 3.3 MEC 选择依据

Transformer-M 是 Stage-2j 五个 MEC 中 **Dev Val 180 SR 最高**的检查点：

| 记录 | Episodes | SR | OSR | SPL | nDTW |
|---|---:|---:|---:|---:|---:|
| Dev Val 180（选择依据） | 180 | 57.22% | 85.56% | 0.5722 | 0.5575 |
| Final Test 300（已发布记录，仅作溯源） | 300 | 55.33% | 82.00% | 0.5525 | 0.5660 |

Final Test 300 记录只用于证明检查点身份，不用于选择。Stage-2n 不重新训练
任何 MEC。

### 3.4 MEC 视觉输入与关联智能体输入

- **MEC 侧**：仍接收 current RGB tokens `[1,576,384]`、goal tokens
  `[1,3,384]`、history visual/pose/action/mask、GAB `[1,20]`、map feature
  `[1,22]`，正常输出 Forward/Left/Right/Stop 与 arrival probability；
- **关联智能体侧**：只接收 384-D Pure-MEC v2 状态 + 独立 20-D 可用性 Mask；
  不接收 RGB、目标图像、视觉 PCA、历史视觉、目标坐标、Oracle 动作或成功
  标签。

### 3.5 同质性一致性检查

因为节点身份不是 MEC 输入，且 20 个节点共享同一权重，所以：

- 相同观测输入不同节点必须产生相同导航动作；
- 任意关联策略在同一条任务上的动作序列、SR、SPL、nDTW 必须逐任务一致；
- 只有 E2E 时延允许不同。

该检查在训练前用 6 条 Dev 任务 × 4 种策略实际跑通
（`v83_spatial_stage2/stage2n_evidence/consistency_probe.json`）：动作不一致
0、SR/SPL/nDTW 差异 0；Random E2E 94.10 ms，Nearest/Max-Uplink/Greedy 均
43.98 ms。同质部署下这三种启发式排序相同，这是预期结果，不是实现错误。

---

## 4. 状态、动作 Mask 与奖励

### 4.1 状态

沿用 Stage-2k v2 Pure-MEC v2：

```text
2-D UAV position / 1500
+ 1-D episode step t / 400
+ 1-D running_mean_E2E_ms / 100
+ 20 × 19-D candidate MEC features
= 384-D MLP input
```

独立 20-D 可用性 Mask 只用于动作屏蔽，不拼进 MLP 输入。每个候选的 19 维
包括相对位置/距离、算力、带宽、上行速率、通信时延、推理时延、E2E 时延、
SNR、独立 Dev 闭环 SR/OSR/SPL/nDTW、模型类型 one-hot。

明确排除：视觉/PCA、Goal token、目标视觉相似度、历史视觉、热图、GAB、
地图检索、导航动作、目标位置、目标距离、Oracle action、成功标签。

### 4.2 奖励

```text
r_t = -0.25 / 100 * (mean_E2E_t - mean_E2E_{t-1})
```

其中 `mean_E2E_t` 是已执行步骤的运行平均：

```text
mean_t = mean_{t-1} + (E2E_t - mean_{t-1}) / t
```

因此整条 Episode 的未折扣回报严格满足：

```text
LatencyReturn = Σ_t r_t = -0.25 * mean_E2E_ms / 100
```

- 导航进展 / nDTW 奖励项已删除；
- 无成功终止奖励、失败惩罚、Oracle 奖励或任何新的 reward shaping；
- DQN 与 PPO 均使用 `gamma = 1.0`；
- 每个 Episode 训练/验证时都检查
  `|return - (-0.25 × mean_E2E / 100)| <= 1e-8`，失败即中止。

### 4.3 指标角色

- **主要优化目标**：平均 E2E 时延（ms）与 LatencyReturn。两者存在确定
  线性关系，不是相互独立的性能证据。
- **导航质量合理性检查**：SR、OSR、SPL、nDTW。在同质部署中它们不随关联
  策略变化，只用于确认 MEC 输入/动作没有被改动。
- **鲁棒性指标**：P95 E2E、100 ms 时延违约率。

---

## 5. 算法与超参数

算法与 Stage-2k v2 / Stage-2l 标准实现一致，不引入 Double DQN、优先回放、
奖励归一化或其他算法改动。

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
GAE λ = 0.95
clip ε = 0.2
rollout = 2,048 steps
minibatch = 256
update epochs = 10
value coef = 0.5
entropy coef = 0.01
Adam lr = 3e-4
max grad norm = 0.5
target KL = 0.05
```

DQN/PPO 使用完全相同的训练任务顺序、固定城市、固定部署、奖励定义、
验证任务与评估代码。

---

## 6. 训练协议

- 每种算法 × 3 个 seed（0/1/2），每个 run 4000 Episodes；
- 每 200 Episodes 在固定 Dev-60 上评估一次；第 4000 Episode 再评估一次；
- 记录：训练 LatencyReturn、MA100、平均 E2E、MA100 E2E、SR、nDTW；
  Dev LatencyReturn、平均 E2E、P95、违约率、SR、OSR、SPL、nDTW；
  DQN Q mean/Q max/target/TD loss；PPO entropy/KL/value loss/
  explained variance；
- 保存 `best.pt`、`last.pt`、`last_full.pt`；
- **不早停**；只有 NaN/Inf、Q 值发散、PPO KL 爆炸或程序错误会中止；
- 先跑 15-Episode smoke（含一次 Dev-60 验证），确认奖励恒等式、Mask、
  状态维度、Q/PPO 数值后再启动 6 个正式 run。

### 收敛判据

只用 Dev 平均 E2E 时延（越低越好）：

```text
最近三次 Dev 验证的 max(E2E) - min(E2E) <= 0.01 * 最佳 E2E
且 最后一次 E2E <= 1.01 * 最佳 E2E
```

- 达到 4000 Episodes 仍不满足 → 明确标记 `not_converged`；
- 不沿用 JointR 的收敛判据；
- 训练预算内不因“看起来收敛”而提前停止。

### 最佳检查点选择

```text
最低 Dev 平均 E2E 时延；
E2E 完全相同时依次比较更高 SR、更早 Episode。
```

Test 不参与该选择。

---

## 7. Test-400 基线比较

### 7.1 冻结 MEC lane 评估

由于 20 个节点共享同一 Transformer-M 且 MEC 输入不含节点身份，同一条任务
的导航动作序列与轨迹与关联策略无关。因此 Dev 验证与 Test 评估先用完整
闭环把每条任务的 MEC 运行一次，缓存每一步的位姿、20×19 候选特征、Mask
与 20 个节点的时延特征（“frozen MEC lane”），再在该 lane 上回放不同关联
策略。

回放时，384-D 状态由缓存位姿、步数和**该策略自己的 running mean E2E**
重新构造，因此状态与训练时完全一致；MEC 动作、成功/失败、SR/SPL/nDTW
来自 lane。该过程是精确重放，不是近似。`test_stage2n.py` 在真实任务上
逐字段对比 lane 与完整闭环（4 种启发式 + DQN），要求动作序列、SR、SPL、
nDTW、E2E、P95、LatencyReturn 完全一致。

训练本身仍使用完整闭环环境；lane 只用于 Dev/Test 评估。

### 7.2 策略

模型和超参数冻结后，在同一封存 Test-400、同一固定部署上比较：

| 策略 | 规则 | 信息条件 |
|---|---|---|
| Random | 在可用 MEC 中均匀随机 | 仅当前可用 Mask |
| Nearest MEC | 选择当前距离最近的可用节点 | 当前 UAV 位置 |
| Max-Uplink-Rate | 选择当前上行速率最高的可用节点 | 当前可观测上行速率 |
| Greedy-Min-E2E | 选择当前解析 E2E 估计最低的可用节点 | 当前 UAV 位置、可用 Mask、解析时延模型 |
| DQN | 最优 Dev 检查点，贪心 | 384-D 状态 + 20-D Mask |
| PPO | 最优 Dev 检查点，贪心 | 384-D 状态 + 20-D Mask |

- 所有策略使用相同任务、相同固定拓扑、相同 MEC 检查点；
- 不允许访问目标位置、未来轨迹、Oracle 动作或其他 Oracle 信息；
- Greedy-Min-E2E 使用与 384-D 状态候选 E2E 特征相同的解析时延模型，
  该信息条件在报告中明确标注为“可观测时延特征”，不是特权信息；
- 报告平均 E2E、LatencyReturn、SR、OSR、SPL、nDTW、P95 E2E、时延违约率、
  Short/Medium/Long 分层与多 seed 结果。

---

## 8. 产物与隔离

- 原始 checkpoint、TensorBoard Event、逐 Episode 日志：
  `runs/v83_spatial_stage2/stage2n_latency_only/`（被 `.gitignore` 排除）；
- 可追踪结果：`v83_spatial_stage2/stage2n_evidence/`（CSV、PNG、报告）；
- 代码：`v83_spatial_stage2/stage2n_*.py`、`test_stage2n.py`；
- 不修改 Stage-2l/2m 的任务、检查点、日志或结果；
- 不停止、不覆盖服务器上已有的 Stage-2l B/DQN、B/PPO（SIGSTOP 暂停态）
  进程。

---

## 9. 预先声明的限制

1. 单城市、单部署、单 MEC 检查点；结论不外推到异质算力/带宽或新城市。
2. 同质部署下 Nearest / Max-Uplink-Rate / Greedy-Min-E2E 的排序相同；
   这是受控设计的直接结果，不应解释为三种启发式“恰好表现一样”。
3. 关联策略不改变导航动作，因此 SR/SPL/nDTW 不能用于区分关联算法；
   算法差异只能来自 E2E 时延。
4. Test-400 只读一次；任何后续调参都会破坏本协议的公平性。
