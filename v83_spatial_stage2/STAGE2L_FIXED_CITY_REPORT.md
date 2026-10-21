# Stage-2l 固定城市实验报告（进行中）

日期：2026-10-21  
分支：`experiment/v83-spatial-foundation`  
运行目录：`runs/v83_spatial_stage2/stage2l_fixed_city/`  
协议：`STAGE2L_FIXED_CITY_PROTOCOL.md`  
启动前报告：`STAGE2L_FIXED_CITY_SETUP.md`

> 状态：A 的 DQN 与 PPO 已启动；B 在现有 Stage-2k 训练进程退出后启动；
> C 排队等待资源。本文档在训练/验证推进后更新，收敛结论与 Baseline
> 对照表以最终版本为准。

---

## 1. 本轮改动

### 1.1 环境初始化拆分

旧 `ToyUAVEnv.reset(seed)` 每次重新生成城市与目标。Stage-2l 改为：

- `map_seed = 10,000,001` 固定城市：160 个地标的位置、数量、尺寸、
  颜色、形状、内部 marker，以及道路网格、背景亮度、街区明暗；
- `task_seed` 只决定 UAV 起点、朝向、航向误差与从固定城市已有地标中
  选择的目标；
- 同 `map_seed` 下所有 Episode 共享同一张地图；RGB 仍随 UAV 位姿变化；
- 目标选择不新建建筑、不修改其他地标。

实现：`v83_spatial_stage2/stage2l_fixed_city.py`（`FixedCityEnv`）。

### 1.2 任务生成

- Train 3500（Short/Medium/Long = 1167/1167/1166）；
- Dev 60（20/20/20）；
- Test 60（20/20/20，`sealed: true`，训练不读取）；
- 每个候选用 oracle 模拟到 Stop，成功且专家步数落在 Band 范围才接受；
- 完整 Train 任务池用固定种子全局打乱，不分段训练；
- Train/Dev/Test task seed 与 task ID 两两隔离，第一阶段共享同一张地图。

### 1.3 三种部署配置

| 配置 | 说明 |
|---|---|
| A | Balanced，单张固定拓扑（训练与验证共用） |
| B | Balanced，10 张独立 Train 拓扑按任务固定分配；独立固定 Dev 拓扑 |
| C | Balanced / Quality-rich / Tradeoff-stress，每种一张固定拓扑 |

所有配置共享同一任务清单、任务顺序、奖励、五种冻结 MEC 与验证任务；
只有部署拓扑不同。

### 1.4 算法与状态

- 标准 DQN：`384 → 256 → 256 → 20`，gamma=1.0，epsilon 1.0→0.05/500k 步；
- 标准 PPO：`384 → 256 → 256 → {20,1}`，gamma=1.0，GAE 0.95，clip 0.2；
- 状态：Pure MEC v2，384-D 输入 + 20-D 可用性 mask；
- 奖励：与 JointR 一致的 `r_nav + Δrunning_mean_E2E`，无额外 shaping；
- 五种 Stage-2j T5 MEC 冻结。

---

## 2. 环境固定性与 T5 兼容（启动前）

详见 `stage2l_fixed_city/smoke/smoke_report.md`。关键检查全部通过：

| 检查 | 结果 |
|---|---|
| 城市在任务间不变 | 通过 |
| 目标来自既有地标 | 通过 |
| 任务不新增/修改地标 | 通过 |
| RGB 随位姿变化 | 通过 |
| 同帧可复现 | 通过 |
| T5 输入有限、形状正确 | 通过 |
| 状态维度 = 384 | 通过 |
| 状态无目标字段 | 通过 |
| 同一位姿/不同目标的 Pure-MEC 状态完全相同 | 通过 |

五种冻结 MEC 在 6 条固定 Dev 任务上的轻量闭环（仅用于排除完全崩溃）：

| MEC | SR | nDTW | E2E (ms) | Dev prior SR |
|---|---:|---:|---:|---:|
| Small | 16.67% | 0.2253 | 176.8 | 16.11% |
| Transformer-S | 33.33% | 0.3345 | 206.7 | 53.33% |
| Medium | 16.67% | 0.2277 | 143.3 | 18.33% |
| Transformer-M | 33.33% | 0.3468 | 176.9 | 57.22% |
| Large | 50.00% | 0.4342 | 141.5 | 48.89% |

样本很少，不作为训练效果结论；Transformer-S/M 偏低但未出现性能崩溃。

---

## 3. 任务分布

| Split | 总数 | Short / Medium / Long | 专家步数范围 |
|---|---:|---|---|
| Train | 3500 | 1167 / 1167 / 1166 | 30–80 / 50–115 / 80–160 |
| Dev | 60 | 20 / 20 / 20 | 30–80 / 50–115 / 80–160 |
| Test | 60 | 20 / 20 / 20 | 30–80 / 50–115 / 80–160 |

Seed / ID 隔离与封存检查全部通过；完整分布见
`stage2l_fixed_city/specs/distribution.json` 与 `isolation.json`。

---

## 4. 训练状态与验证

每次固定 Dev 验证（每 200 Episodes）写入
`stage2l_fixed_city/PROGRESS.md`；实时状态可用：

```bash
python v83_spatial_stage2/stage2l_status.py
```

当前启动顺序：

1. A/DQN、A/PPO 已启动（2026-10-21 11:40 本地时间，PID 439796 / 439797）；
2. 旧 Stage-2k DQN 已完成 3500 Episodes 并退出；B/DQN、B/PPO
   已于约 12:07 启动（PID 446206 / 446207）；
3. 新训练进程达到 4 个并发后，C 排队。

监督脚本不会停止、重启或修改现有 Stage-2k 进程。A/B 的验证每 200
Episodes 执行一次，结果写入 `PROGRESS.md`。

### 固定 Dev 验证记录

| 配置 | Episode | 算法 | JointR | SR | nDTW | E2E (ms) | SPL | OSR |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A | 200 | DQN | 0.0056 | 33.33% | 0.3754 | 147.9 | 0.3333 | 50.00% |
| A | 200 | PPO | 0.0258 | 35.00% | 0.4166 | 156.3 | 0.3500 | 55.00% |
| A | 400 | DQN | -0.0168 | 40.00% | 0.3427 | 143.8 | 0.3918 | 61.67% |
| A | 400 | PPO | 0.0800 | 35.00% | 0.4022 | 128.9 | 0.3378 | 51.67% |
| A | 600 | DQN | 0.0111 | 38.33% | 0.3835 | 149.0 | 0.3833 | 65.00% |
| A | 600 | PPO | **0.1410** | 38.33% | 0.4678 | 130.7 | 0.3833 | 58.33% |
| B | 200 | DQN | 0.0402 | 43.33% | 0.4341 | 157.6 | 0.4316 | 58.33% |
| B | 200 | PPO | 0.0089 | 43.33% | 0.4493 | 176.1 | 0.4333 | 65.00% |
| B | 400 | PPO | 0.0911 | 48.33% | 0.4782 | 154.8 | 0.4833 | 65.00% |

说明：

- 这些都是训练早期结果（A/DQN Ep400 时 ε≈0.89），只用于检查新环境
  是否可训练；**不构成收敛结论，也不构成优于/劣于 Baseline 的结论**。
- A/PPO 的 JointR 从 Ep200 的 0.0258 → Ep400 的 0.0800 → Ep600 的
  0.1410；Ep400 的改善主要来自 E2E 降低，Ep600 的改善主要来自
  nDTW 提高（0.4022→0.4678）。A/DQN 在 Ep400 回落到 -0.0168，Ep600
  恢复到 0.0111（新高但仍远低于 PPO），说明单次验证波动仍大，
  不能据此宣称平台。
- B/PPO 的 JointR 从 Ep200 的 0.0089 升到 Ep400 的 0.0911，SR 从
  43.33% 升到 48.33%；B 使用独立 `B_dev` 拓扑，仍不能与 A 直接横比。
- Ep200 量级与旧 Stage-2k 同轮次接近：
  - 旧 Stage-2k PPO Ep200：JointR 0.0344 / SR 38.33% / nDTW 0.427 / E2E 156.9 ms；
  - 旧 Stage-2k Pure-MEC DQN Ep200：JointR 0.0750 / SR 35.00% / nDTW 0.450 / E2E 150.1 ms。
- 因此固定城市切换没有造成闭环性能崩溃；但 Dev 任务分布已改为
  20/20/20 单一 Balanced 场景，不能直接用旧数值做严格横比。
- B 使用独立 `B_dev` 拓扑，其 Ep200 数值不能与 A 的 `A_fixed` 直接横比；
  它用于观察多拓扑训练是否稳定。
- 完整 Baseline 对照必须在固定环境、同一 60 条 Dev 任务上单独运行，
  且只在训练收敛后解释。

---

## 5. 待完成

- [ ] 每 200 Episodes 记录 DQN/PPO 的 Validation JointR、SR、nDTW、E2E；
- [ ] 训练结束后判断是否形成稳定平台（不以早停作为收敛证据）；
- [ ] 在固定城市、固定配置拓扑、独立 Dev 任务上评估 DQN、PPO 与五种 Baseline；
- [ ] 生成收敛曲线与 Baseline 对照表；
- [ ] 仅在固定环境内 baseline 成立后，另行规划新拓扑/新城市泛化；
- [ ] Final Test 保持封存，直到方案冻结。

---

## 6. 复现命令

```bash
# 1) 构建固定城市、任务池、顺序与拓扑
python v83_spatial_stage2/stage2l_build.py --force

# 2) 测试 + 轻量闭环 smoke
python v83_spatial_stage2/test_stage2l_fixed_city.py
python v83_spatial_stage2/stage2l_smoke.py

# 3) 启动前报告
python v83_spatial_stage2/stage2l_setup_report.py

# 4) 训练（监督脚本会自动排队 A→B→C）
nohup setsid python -u v83_spatial_stage2/stage2l_supervisor.py \
    --max-concurrent 4 --poll-seconds 30 \
    > runs/v83_spatial_stage2/stage2l_fixed_city/logs/supervisor_stdout.log 2>&1 &

# 5) 状态与报告
python v83_spatial_stage2/stage2l_status.py
python v83_spatial_stage2/stage2l_report.py

# 6) 收敛后的固定环境评测（Dev）
python v83_spatial_stage2/stage2l_eval.py --config A --split dev
```

---

## 7. 限制

- 本报告当前只覆盖环境/任务生成改造与启动前验证；训练与基线结果待补。
- 固定城市内 baseline 成立前，不做新拓扑/新城市泛化结论。
- 所有旧 Stage-2k / Stage-2j 结果保留，不删除、不覆盖。
