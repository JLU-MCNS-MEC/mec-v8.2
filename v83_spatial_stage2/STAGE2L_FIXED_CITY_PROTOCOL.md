# Stage-2l 固定城市闭环导航协议（预注册）

日期：2026-10-21  
分支：`experiment/v83-spatial-foundation`  
仓库：`JLU-MCNS-MEC/mec-v8.2`  
新实验根目录：`runs/v83_spatial_stage2/stage2l_fixed_city/`

本协议在 Stage-2l 正式训练启动前冻结。旧 Stage-2k / Stage-2j 运行的全部权重、
日志、TensorBoard 与报告保留，不停止、不覆盖、不重启。五种 Stage-2j T5 MEC
Checkpoint 全部冻结（`eval()` + `requires_grad=False`），不重新训练 MEC。
Stage-2j Final Test 300 与 Association Final Test 180 均不读取。

---

## 1. 目标

本轮只改变**导航环境与训练任务生成方式**，算法、状态与奖励保持不变：

1. 把旧 `ToyUAVEnv.reset(seed)` 拆成 `map_seed`（固定城市）与 `task_seed`
   （随机导航任务）；
2. 从同一固定城市中随机生成 Short / Medium / Long 各 1/3 的 VLN 任务，
   完整 Train 任务池全局随机打乱；
3. 在 A/B/C 三种 MEC 部署配置上并行训练标准 DQN 与标准 PPO；
4. 先确认配置 A 能否学习并收敛，再比较拓扑复杂度的影响；
5. 收敛后先在固定环境内与五种 Baseline 比较，再考虑新拓扑/新城市泛化。

本协议不引入成功终止奖励、失败惩罚、Oracle 奖励或任何新的 reward shaping。

---

## 2. 固定城市

### 2.1 `map_seed`

第一阶段统一使用：

```text
map_seed = 10,000,001
```

`map_seed` 固定以下内容：

- 城市地标的位置、数量（160）、物理尺寸、基础颜色、形状、内部 marker；
- 背景基准亮度、道路网格间距/宽度/透明度、街区明暗扰动；
- 地标之间的离散视觉签名互不重复。

城市由 `stage2l_fixed_city.build_city(map_seed)` 构建并缓存；任何任务 reset
都不会重建城市，也不会新增/修改地标。

### 2.2 `task_seed`

`task_seed` 只决定：

- UAV 起点 `(x, y)`；
- 初始航向与 ±30° 航向误差；
- 从固定城市已有 160 个地标中选择目标 `goal_index`；
- 起终点距离落在对应 Band 范围内。

目标必须来自固定城市已有地标；任务不会为目标临时创建建筑，也不会因为
选择不同目标而修改其他地标。RGB 观测仍由 UAV 位置/航向渲染，不是固定图像。

### 2.3 目标参考与 T5 输入

- Goal reference 仍由目标地标自身的 shape/color/marker 渲染为多尺度图像；
- 复用现有 frozen encoder 与 T5 输入接口：
  - current tokens `[1, 576, 384]`
  - goal tokens `[1, 3, 384]`
  - history visual/pose/action/mask
  - GAB `[1, 20]`、anonymous map `[1, 22]`
- 固定城市地图的 160 个匿名节点 embeddings 只按 `map_seed` 构建一次并复用；
  目标节点不携带 `is_goal` 字段。

### 2.4 固定性验收

启动前必须通过 `test_stage2l_fixed_city.py` 与 `stage2l_smoke.py`：

- 不同 `task_seed` 下城市指纹不变；
- 目标必为既有地标；
- 任务间地标集合/属性不变；
- 同任务同帧可复现，不同位姿帧不同；
- T5 输入有限且形状正确；
- 同一位姿、不同目标的 Pure-MEC 状态完全相同（无目标泄漏）；
- Stop 成功必须满足 20 m 半径；未 Stop 最多 400 步。

---

## 3. 任务池

### 3.1 数量与比例

| Split | 总数 | Short | Medium | Long |
|---|---:|---:|---:|---:|
| Train | 3500 | 1167 | 1167 | 1166 |
| Dev | 60 | 20 | 20 | 20 |
| Test（封存） | 60 | 20 | 20 | 20 |

- Short / Medium / Long 的距离范围与专家步数范围沿用
  `config.TRAIN.trajectory_mix`：
  - Short：150–250 m，30–80 步
  - Medium：250–400 m，50–115 步
  - Long：400–550 m，80–160 步
- 每个候选任务都用 oracle expert 模拟到 Stop，只有成功且步数落在 Band
  范围内才接受。
- Train 任务池生成后使用固定种子 `20261021` **全局随机打乱**；不存在
  “先 Short、再 Medium、最后 Long”的分段训练。

### 3.2 Seed / ID 隔离

```text
Train task_seed base = 11,000,000
Dev   task_seed base = 12,000,000
Test  task_seed base = 13,000,000
Task ID prefix        = stage2l-{split}-{index:05d}
```

- Train/Dev/Test 的 task seed 集合与 task ID 集合两两不相交；
- 第一阶段允许共享同一张固定城市地图（`map_seed` 相同）；
- Test 任务池在构建时写入 `sealed: true`，训练脚本不读取。

### 3.3 闭环定义

每个 Episode：

1. RL 策略选择一个可用 MEC；
2. 该 MEC（冻结）输出 Forward / Left / Right / Stop；
3. UAV 执行动作，环境返回 RGB、位姿与奖励；
4. `Stop` 立即结束 Episode；成功必须满足距目标 < 20 m；
5. 未 Stop 最多执行 400 步。

---

## 4. 三种部署配置

| 配置 | 场景 | Train 拓扑 | Dev 拓扑 |
|---|---|---|---|
| A | Balanced | 1 张固定拓扑 `A_fixed` | 同一张 `A_fixed` |
| B | Balanced | 10 张独立 Train 拓扑 `B_train_00..09`，按任务固定分配 | 独立固定 `B_dev` |
| C | Balanced / Quality-rich / Tradeoff-stress | 每种场景 1 张固定拓扑 `C_balanced` / `C_quality_rich` / `C_tradeoff_stress` | 同一组固定拓扑 |

- 拓扑种子使用独立的 9,000,000+ 段，与 Stage-2k 的 8,000,000+、Stage-2j 的
  17.8M / 34.8M / 48.8M 均不重叠。
- B 的 10 张训练拓扑只从独立 Train 拓扑池抽样；Dev 拓扑不在训练池中。
- C 在生成配置清单时把同一批导航任务近似 1:1:1 分配给三种场景；
  任务本身（起点/目标/band）与 A/B 完全相同。
- 所有配置共享同一固定城市、同一 Train 任务顺序、同一 60 条 Dev 任务、
  同一奖励、同一五种冻结 MEC 与同一验证代码。

拓扑角色与种子：

```text
A_fixed            = 9,000,000  balanced
B_dev              = 9,000,050  balanced
B_train_00..09     = 9,000,100..9,000,109  balanced
B_test             = 9,000,060  balanced（封存，不用于训练）
C_balanced         = 9,000,000  balanced
C_quality_rich     = 9,000,200  quality_rich
C_tradeoff_stress  = 9,000,300  tradeoff_stress
```

---

## 5. 状态、奖励与算法

### 5.1 状态

沿用 Stage-2k v2 **Pure MEC** 状态：

```text
2-D UAV position / 1500
+ 1-D episode step t / 400
+ 1-D running_mean_E2E_ms / 100
+ 20 x 19-D candidate MEC features
= 384-D MLP input

+ 20-D availability mask（独立，仅用于动作屏蔽）
```

每个候选的 19 维：相对位置/距离、算力、带宽、上行速率、通信时延、
推理时延、E2E 时延、SNR、独立 Dev 闭环 SR/OSR/SPL/nDTW、模型类型 one-hot。

明确排除：视觉/PCA、Goal token、目标视觉相似度、历史视觉、热图、GAB、
地图检索、导航动作、目标位置、目标距离、Oracle action、成功标签。

### 5.2 奖励

```text
r_t = r_nav_t + (-0.25 / 100) * (mean_E2E_t - mean_E2E_{t-1})
```

其中 `mean_E2E_t` 是已执行步骤的运行平均。整条 Episode 的未折扣回报
精确等于 `final_nDTW - 0.25 * mean_E2E / 100 = JointR`。

不增加成功终止奖励、失败惩罚或 Oracle 奖励。

### 5.3 算法

- DQN：标准 MLP `384 → 256 → 256 → 20`，gamma=1.0，
  replay 50,000，target update 1,000，batch 128，
  epsilon 1.0→0.05 / 500,000 环境步，Adam lr=1e-4，grad clip 10。
- PPO：标准 actor-critic `384 → 256 → 256 → {20, 1}`，gamma=1.0，
  GAE λ=0.95，clip 0.2，rollout 2,048，minibatch 256，10 epochs，
  value coef 0.5，entropy coef 0.01，Adam lr=3e-4。
- 两者使用完全相同的任务清单、任务顺序、奖励、MEC、验证任务与状态。
- 不做 Double DQN、优先回放、奖励归一化或任何算法改动。

---

## 6. 训练协议

- 每种配置 × 每种算法最多 **3500 Episodes**；
- 每 **200 Episodes** 在 60 条固定独立 Dev 任务上验证一次；
- 验证指标：JointR、SR、OSR、SPL、nDTW、E2E Delay、P95、平均步数；
- TensorBoard 记录：真实奖励、100-Episode 移动平均、Validation JointR /
  SR / nDTW / E2E、DQN Q mean/Q max、PPO policy entropy / value / KL /
  clip fraction / explained variance；
- 保存 best validation checkpoint 与 last checkpoint；
- **不早停**：只有 NaN/Inf、Q 发散、KL 异常或程序错误会中止；
- 训练结束后的 `classify_status` 只根据最后几次固定 Dev 验证判断是否形成
  稳定平台；不因为触发早停而宣称收敛。

启动顺序：

1. 立即启动 A 的 DQN 与 PPO；
2. 现有 Stage-2k 训练进程退出后，启动 B 的 DQN 与 PPO；
3. 新训练进程总数达到上限后，C 排队；有空位时启动；
4. 监督脚本从不停止、重启或修改现有进程。

---

## 7. 评测协议

### 7.1 固定环境内

收敛后先在**同一固定城市、同一配置拓扑、独立 Dev 任务**上比较：

- DQN（best checkpoint，贪心）
- PPO（best checkpoint，贪心）
- Random
- Min-Latency
- Max-Quality
- Heuristic / Weighted Greedy
- Oracle（特权上界，只作参考，不参与训练）

指标：SR、SPL、nDTW、平均 E2E Delay、JointR。

### 7.2 Final Test

只有在固定环境内 baseline 成立、方案冻结后才允许运行：

```bash
python v83_spatial_stage2/stage2l_eval.py --config A --split test \
    --allow-final-test
```

Test 结果不得用于调参。新拓扑/新城市泛化必须另建任务池与拓扑池后单独运行。

---

## 8. 产物

```text
runs/v83_spatial_stage2/stage2l_fixed_city/
  protocol_freeze.json
  specs/{train_tasks,dev_tasks,test_tasks,training_order,isolation,distribution}.json
  topologies/*.json
  configs/{A,B,C}.json
  smoke/smoke_report.{json,md}
  A/{dqn,ppo}/...
  B/{dqn,ppo}/...
  C/{dqn,ppo}/...
  evaluation/{A,B,C}/{dev,test}/...
  figures/*.png
  logs/*.log
  SETUP_REPORT.md
  REPORT.md
  PROGRESS.md
```

权重、TensorBoard 与逐 Episode 日志留在服务器；源码、测试与中文 Markdown
报告提交到当前分支。

---

## 9. 冻结声明

- 本协议在两臂（A/B/C × DQN/PPO）训练启动前冻结；
- 训练期间不修改奖励函数、状态定义、任务分布或验证任务；
- 不根据 Dev 结果临时改变训练预算或选择 seed；
- 旧 Stage-2k / Stage-2j 结果不删除、不覆盖。
