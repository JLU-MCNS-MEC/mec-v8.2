# Stage-2l 固定城市实验报告（进行中）

日期：2026-10-21  
分支：`experiment/v83-spatial-foundation`  
运行目录：`runs/v83_spatial_stage2/stage2l_fixed_city/`  
协议：`STAGE2L_FIXED_CITY_PROTOCOL.md`  
启动前报告：`STAGE2L_FIXED_CITY_SETUP.md`

> 状态：A/DQN、A/PPO 继续训练到 3500 Episodes；B/DQN、B/PPO 已于
> 2026-10-21 13:16 按用户决定暂停（SIGSTOP，保留 checkpoint）；
> C/DQN、C/PPO 已于 13:19 手动启动。原监督脚本已停止，改由
> `stage2l_progress_logger.py` 继续写 `PROGRESS.md`。本文档在训练/验证
> 推进后更新，收敛结论与 Baseline 对照表以最终版本为准。

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

1. A/DQN、A/PPO 已启动（2026-10-21 11:40 本地时间，PID 439796 / 439797），
   继续训练到 3500 Episodes；
2. 旧 Stage-2k DQN 已完成 3500 Episodes 并退出；B/DQN、B/PPO
   于约 12:07 启动（PID 446206 / 446207）；
3. **2026-10-21 13:16：按用户决定，B/DQN（Ep1030）与 B/PPO（Ep1120）
   以 SIGSTOP 暂停**；checkpoint、TensorBoard、日志全部保留；
4. **2026-10-21 13:19：手动启动 C/DQN、C/PPO**（PID 469303 / 469304），
   同样跑到 3500 Episodes；
5. 原监督脚本已停止，避免 A 完成后重复启动 C；改由
   `stage2l_progress_logger.py` 持续把 A/C 验证写入 `PROGRESS.md`。

B 暂停的 PID、checkpoint 和恢复/终止命令见
`runs/v83_spatial_stage2/stage2l_fixed_city/PAUSE_NOTES.md`；C 的启动记录见
`runs/v83_spatial_stage2/stage2l_fixed_city/C/LAUNCH.json`。

### 固定 Dev 验证记录

| 配置 | Episode | 算法 | JointR | SR | nDTW | E2E (ms) | SPL | OSR |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A | 200 | DQN | 0.0056 | 33.33% | 0.3754 | 147.9 | 0.3333 | 50.00% |
| A | 200 | PPO | 0.0258 | 35.00% | 0.4166 | 156.3 | 0.3500 | 55.00% |
| A | 400 | DQN | -0.0168 | 40.00% | 0.3427 | 143.8 | 0.3918 | 61.67% |
| A | 400 | PPO | 0.0800 | 35.00% | 0.4022 | 128.9 | 0.3378 | 51.67% |
| A | 600 | DQN | 0.0111 | 38.33% | 0.3835 | 149.0 | 0.3833 | 65.00% |
| A | 600 | PPO | 0.1410 | 38.33% | 0.4678 | 130.7 | 0.3833 | 58.33% |
| A | 800 | DQN | -0.0242 | 31.67% | 0.3504 | 149.9 | 0.3167 | 51.67% |
| A | 800 | PPO | **0.1454** | 40.00% | 0.4899 | 137.8 | 0.4000 | 60.00% |
| A | 1000 | DQN | -0.0078 | 31.67% | 0.3470 | 141.9 | 0.3167 | 50.00% |
| A | 1000 | PPO | 0.1378 | 36.67% | 0.4543 | 126.6 | 0.3667 | 51.67% |
| A | 1200 | DQN | -0.0619 | 31.67% | 0.3166 | 151.4 | 0.3059 | 56.67% |
| A | 1200 | PPO | 0.1450 | 43.33% | 0.4446 | 119.9 | 0.4222 | 65.00% |
| A | 1400 | DQN | -0.0254 | 30.00% | 0.3229 | 139.3 | 0.3000 | 51.67% |
| A | 1400 | PPO | 0.0848 | 36.67% | 0.4005 | 126.3 | 0.3666 | 53.33% |
| A | 1600 | DQN | -0.0584 | 30.00% | 0.3194 | 151.1 | 0.3000 | 46.67% |
| A | 1600 | PPO | 0.1507 | 38.33% | 0.4451 | 117.8 | 0.3833 | 60.00% |
| A | 1800 | DQN | 0.0025 | 36.67% | 0.3582 | 142.3 | 0.3667 | 58.33% |
| A | 1800 | PPO | **0.1801** | 45.00% | 0.4859 | 122.3 | 0.4500 | 68.33% |
| A | 2000 | PPO | 0.1584 | 43.33% | 0.4804 | 128.8 | 0.4333 | 68.33% |
| B | 200 | DQN | 0.0402 | 43.33% | 0.4341 | 157.6 | 0.4316 | 58.33% |
| B | 200 | PPO | 0.0089 | 43.33% | 0.4493 | 176.1 | 0.4333 | 65.00% |
| B | 400 | DQN | 0.0390 | 40.00% | 0.4100 | 148.4 | 0.3994 | 55.00% |
| B | 400 | PPO | **0.0911** | 48.33% | 0.4782 | 154.8 | 0.4833 | 65.00% |
| B | 600 | DQN | -0.0649 | 36.67% | 0.3231 | 155.2 | 0.3661 | 53.33% |
| B | 600 | PPO | -0.0286 | 41.67% | 0.4040 | 173.0 | 0.4161 | 63.33% |
| B | 800 | DQN | -0.0063 | 36.67% | 0.3918 | 159.2 | 0.3667 | 55.00% |
| B | 800 | PPO | 0.0534 | 41.67% | 0.4204 | 146.8 | 0.4154 | 58.33% |
| B | 1000 | DQN | -0.0598 | 28.33% | 0.3416 | 160.6 | 0.2833 | 48.33% |
| B | 1000 | PPO | -0.0354 | 41.67% | 0.3831 | 167.4 | 0.4153 | 56.67% |
| C | 200 | DQN | 0.0510 | 30.00% | 0.4083 | 142.9 | 0.3000 | 48.33% |
| C | 200 | PPO | -0.0369 | 33.33% | 0.3556 | 157.0 | 0.3333 | 48.33% |
| C | 400 | DQN | 0.0222 | 33.33% | 0.3993 | 150.8 | 0.3333 | 55.00% |
| C | 400 | PPO | -0.0327 | 26.67% | 0.3378 | 148.2 | 0.2667 | 41.67% |
| C | 600 | DQN | **0.0732** | 38.33% | 0.4162 | 137.2 | 0.3833 | 60.00% |
| C | 600 | PPO | 0.0630 | 41.67% | 0.4603 | 158.9 | 0.4167 | 58.33% |

说明：

- 这些是截至 A/Ep1200、B/Ep1000 的训练中途结果（A/DQN 的 ε 仍在
  衰减），只用于检查新环境是否可训练；**不构成收敛结论，也不构成
  优于/劣于 Baseline 的结论**。
- A/PPO 的 JointR 从 Ep200 的 0.0258 → Ep400 的 0.0800 → Ep600 的
  0.1410 → Ep800 的 0.1454 → Ep1000 的 0.1378 → Ep1200 的 0.1450
  → Ep1400 的 0.0848 → Ep1600 的 0.1507 → Ep1800 的 0.1801
  → Ep2000 的 0.1584。Ep800–1200 曾连续 3 次落在 0.01 带内，但
  Ep1400 回落后继续创新高，Ep2000 又回落；最后 3 次（0.1507 /
  0.1801 / 0.1584）跨度 0.0294，**不是稳定平台**；正式收敛判断仍
  只在 3500 Episodes 结束后进行。
  A/DQN 在 Ep400 回落到 -0.0168，Ep600 恢复到 0.0111（best），
  Ep800 又降到 -0.0242，Ep1000 仍为 -0.0078，Ep1200 进一步降到
  -0.0619，Ep1400 为 -0.0254，Ep1600 又降到 -0.0584，Ep1800 恢复到
  0.0025（SR 36.67%，nDTW 0.3582）；DQN 在 A 配置上明显不如 PPO
  稳定，best 仍只有 0.0111，不能据此宣称平台。
- B/PPO 在 Ep400/600/800/1000 分别为 0.0911 / -0.0286 / 0.0534 /
  -0.0354，围绕 0 大幅振荡；Ep1000 的 E2E 又升到 167.4 ms（Large
  39.4% + Medium 28.5%），best 仍是 Ep400 的 0.0911。B/DQN 从 Ep200
  的 0.0402 经 Ep400 的 0.0390、Ep600 的 -0.0649、Ep800 的 -0.0063
  到 Ep1000 的 -0.0598（SR 28.33%，nDTW 0.3416），best 仍是 Ep200
  的 0.0402。B 两臂都明显比 A 不稳定，没有形成平台；B 使用独立
  `B_dev` 拓扑，仍不能与 A 直接横比。
- C 两臂：C/DQN 0.0510（Ep200）→ 0.0222（Ep400）→ 0.0732（Ep600，
  新 best，SR 38.33%）；C/PPO -0.0369（Ep200）→ -0.0327（Ep400）
  → 0.0630（Ep600）。C 两臂在 Ep600 都转正，但仍只是训练中途，
  且使用三种部署场景各一张固定拓扑，不能与 A/B 直接横比。
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

## 5. 配置 B 暂停记录（2026-10-21 13:16）

| 运行 | PID | 暂停时 Episode | 最后验证 | Best JointR |
|---|---|---:|---|---:|
| B/DQN | 446206 | 1030 | Ep1000 | 0.0402 (Ep200) |
| B/PPO | 446207 | 1120 | Ep1000 | 0.0911 (Ep400) |

- 暂停方式为 `SIGSTOP`，不是删除；进程状态为 `T`，不再占用 CPU，
  但仍保留 CUDA 上下文/显存。
- Checkpoint、TensorBoard、`training_history.json`、`progress.json`
  全部保留，未删除、未覆盖。
- A/DQN、A/PPO 不受影响。
- 恢复：`kill -CONT 446206 446207`。
- 若确认不再恢复并需要释放显存：`kill -TERM 446206 446207`，之后可用
  `last_full.pt` 续训。
- 详细记录：`runs/v83_spatial_stage2/stage2l_fixed_city/PAUSE_NOTES.md`。

### C 启动记录（2026-10-21 13:19）

- 原监督脚本停止后，C/DQN、C/PPO 手动启动：
  - PID 469303 / 469304
  - 参数：`--config C --max-episodes 3500 --validation-every 200 --threads 4`
  - 日志：`logs/C_dqn.log`、`logs/C_ppo.log`
  - 启动记录：`C/LAUNCH.json`
- C 使用三种部署场景（Balanced / Quality-rich / Tradeoff-stress），
  每种一张固定拓扑；任务清单、任务顺序与奖励与 A/B 相同。
- A 完成后不会再触发 C 的重复启动；验证由
  `stage2l_progress_logger.py` 写入 `PROGRESS.md`。

---

## 6. 待完成

- [ ] 每 200 Episodes 记录 DQN/PPO 的 Validation JointR、SR、nDTW、E2E；
- [ ] 训练结束后判断是否形成稳定平台（不以早停作为收敛证据）；
- [ ] 在固定城市、固定配置拓扑、独立 Dev 任务上评估 DQN、PPO 与五种 Baseline；
- [ ] 生成收敛曲线与 Baseline 对照表；
- [ ] 仅在固定环境内 baseline 成立后，另行规划新拓扑/新城市泛化；
- [ ] Final Test 保持封存，直到方案冻结。

---

## 7. 复现命令

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

## 8. 限制

- 本报告当前只覆盖环境/任务生成改造与启动前验证；训练与基线结果待补。
- 固定城市内 baseline 成立前，不做新拓扑/新城市泛化结论。
- 所有旧 Stage-2k / Stage-2j 结果保留，不删除、不覆盖。
