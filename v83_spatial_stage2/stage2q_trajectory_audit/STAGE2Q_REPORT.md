# Stage-2q：DQN/PPO UAV 飞行轨迹可视化与跨方法一致性审计

- 日期：2026-10-25
- 仓库：`JLU-MCNS-MEC/mec-v8.2`
- 分支：`experiment/v83-spatial-foundation`
- 实验根目录（可追踪）：`v83_spatial_stage2/stage2q_trajectory_audit/`
- 原始运行日志（被 `.gitignore` 排除）：`runs/v83_spatial_stage2/stage2q_collect.log`
- 任务池：Stage-2n **Dev-60**，`map_seed = 10,000,001`
- Test-400：**未读取**（`final_test_read = false`、`test_pool_read = false`）
- 训练：**未重新训练**；只加载各实验的 `best.pt` 做推理与真实环境闭环
- 历史结果：**未覆盖** Stage-2n / Stage-2o / Stage-2p 的任何检查点、日志或结果

---

## 0. 直接回答

### 0.1 GitHub 和服务器上原来有没有保存真实轨迹？

**没有。** GitHub 的 Stage-2n/2o/2p 证据目录中只有逐 Episode 指标、动作哈希、
训练曲线和奖励曲线；`navigation_invariance.csv` 只有
`action_sha256 / success / steps / SPL / nDTW` 等审计列，没有坐标。
服务器 `runs/v83_spatial_stage2/{stage2n_latency_only,stage2o,stage2p}/`
也没有保存任何 lane/pose/trajectory 缓存（没有 `.npz/.npy/.pkl` 的 pose 或
trajectory 文件）。`LaneStep.pose` 确实存在于 `stage2n_lanes.py`，但 lane 只在
评测进程内存中构建，评测器只把动作哈希和指标写入 CSV，没有把逐步坐标落盘。

因此本次按协议重新执行了 Dev 验证：重建 Dev-60 frozen MEC lane，并对 6 条
代表任务做完整真实闭环，逐步记录坐标、动作、E2E 和关联节点。没有从 Test
补样本，也没有用任务文字或示意图虚构轨迹。

### 0.2 六类成功/失败样本分别用了哪些 Episode？

| 类别 | 结果 | Episode | 实际步数 | 起点–目标距离 | 选择依据 |
|---|---|---|---:|---:|---|
| Short | 成功 | `stage2n-dev-00009` | 41 | 190.9 m | Short 成功层中位步数 |
| Short | 失败 | `stage2n-dev-00008` | 232 | 171.0 m | Short 失败层中位步数 |
| Medium | 成功 | `stage2n-dev-00024` | 76 | 384.3 m | Medium 成功层中位步数 |
| Medium | 失败 | `stage2n-dev-00035` | 131 | 337.7 m | Medium 失败层中位步数 |
| Long | 成功 | `stage2n-dev-00045` | 104 | 446.3 m | Long 成功层中位步数 |
| Long | 失败 | `stage2n-dev-00057` | 137 | 428.9 m | Long 失败层中位步数 |

选择规则在采集前固定：在每个 band × 成功/失败层内，按 `(steps, episode_index)`
升序排序，取下中位 `(n-1)//2`。六个层在 Dev-60 中都有样本，未使用 Test。
注意：Short/Medium/Long 是协议按**起点–目标距离**定义的 band，不是事后按
实际步数重新分箱；失败轨迹的实际步数可以超出该 band 的 oracle 步数范围
（例如 Short 失败样本跑了 232 步）。

### 0.3 同一任务在 DQN/PPO 和不同奖励方案下的 UAV 轨迹是否一致？

**完全一致。** 在同一任务上，18 个 learned 关联方法（Stage-2n 6 个、
Stage-2o 6 个、Stage-2p 6 个）以及 Greedy-Min-E2E、Random 两个对照方法：

- 导航动作序列逐步相同；
- UAV 决策坐标 `(x, y, heading)` 逐步相同（float32 逐位相同）；
- 轨迹步数相同；
- 成功/失败结果相同。

三个证据层级全部为 0 不一致：

| 证据层级 | 比较数 | 动作不一致 | 坐标哈希不一致 | 步数不一致 | 成功状态不一致 | 最大坐标偏差 |
|---|---:|---:|---:|---:|---:|---:|
| Dev-60 lane 回放（60 任务 × 20 方法） | 1200 | 0 | 0 | 0 | 0 | **0.000000 m** |
| 6 任务真实闭环（6 × 20） | 120 | 0 | 0 | 0 | 0 | **0.000000 m** |
| 闭环 vs frozen lane（6 × 20） | 120 | 0 | 0 | 0 | 0 | **0.000000 m** |

差异确实存在，但只在**关联层**：不同方法选择了不同的 MEC 节点，因而每步
E2E、平均 E2E、P95 和时延违约率不同。6 条代表任务上，20 个方法产生了
**14–16 种不同的节点选择序列**（每任务），同一任务的 E2E 跨度达
**37.4–68.0 ms**。这证明“轨迹相同”不是因为方法没有区别，而是因为节点选择
不进入导航模型。

### 0.4 如果一致，是否有动作哈希和坐标偏差数据作为证据？

有，而且有三层：

1. **动作哈希**：`trajectory_summary.csv` / `trajectory_invariance.csv` 中的
   `action_sha256` 与 Stage-2n 历史格式完全一致
   （`sha256(",".join(str(action)))`）。6 条代表任务每条都有 1 个唯一动作
   哈希（20 个方法共用），1200 次 lane 比较和 120 次闭环比较的
   `actions_equal` 全部为 true。
2. **坐标哈希**：`decision_coord_sha256` 是 float32 `[x, y, heading]` 数组的
   SHA256，`path_coord_sha256` 是 float32 后动作轨迹 `[x, y]` 的 SHA256。
   三个层级的 `coord_hash_equal` 全部为 true。
3. **逐步最大坐标偏差**：`max_coord_deviation_m` 在三个层级全部为
   **0.000000 m**（不是“看起来接近”，是 float32 坐标逐位相同）。
   `max_heading_deviation_rad` 也全部为 0.0。

此外，历史复现审计证明本次重建没有换模型/换任务池：

- Stage-2n 逐 Episode：8 个策略 × 60 任务 = 480 次比较，动作哈希/SR/步数/
  SPL/nDTW/mean E2E/P95/违约率/LatencyReturn **最大绝对误差 0.0**。
- Stage-2n 汇总：8 个策略的全部 10 个汇总字段**最大绝对误差 0.0**。
- Stage-2o/2p 的 12 个 `best.pt` 验证指标：E2E/E2E std/P95/SR/OSR/SPL/
  nDTW/违约率/平均步数**最大绝对误差 0.0**。

### 0.5 哪些方法或任务确实存在不同轨迹？差异来自哪里？

**没有任何“同一任务、不同方法”的导航轨迹不同。** 6 条代表任务各自有不同
起点、目标和 band，因此**不同任务之间**轨迹当然不同；这不是不一致，也不应
要求不同任务产生相同轨迹。本审计只比较同一任务跨方法。

方法之间的真实差异全部在关联层：

- MEC 节点选择序列不同：Dev-60 lane 中累计 **107,111** 个 step 的节点选择
  与参考 `2n-DQN-s0` 不同；6 任务真实闭环中累计 **9,600** 个 step 不同；
  闭环 vs frozen lane 中累计 **7,521** 个 step 不同。
- 节点选择不一致比例：learned 方法在 6 任务上为 **56.1%–84.8%**；
  Greedy-Min-E2E 为 63.6%，Random 为 84.0%。
- 平均 E2E 差异（6 任务，相对 `2n-DQN-s0`）：learned 方法从
  **−16.17 ms**（2o-B-PPO-s0）到 **+15.33 ms**（2o-one-DQN-s0）；
  Greedy 为 **−21.04 ms**，Random 为 **+25.14 ms**。
- 逐任务 E2E 跨度：37.4–68.0 ms；同一任务上 20 个方法的平均 E2E 最小/最大
  差异显著。

原因：20 个 MEC 节点使用**同一冻结 Transformer-M**，算力、带宽、覆盖半径
相同，只有位置不同；MEC 输入（RGB tokens、goal tokens、history、GAB、
map feature）**不包含节点编号、关联动作或 E2E 时延**。因此关联策略只能
改变“这一步用哪个节点的时延记账”，不能改变 MEC 的导航输出，也不能改变
环境动力学。6 任务 × 20 方法的真实闭环独立地验证了这一点；不是把同质
MEC 的预期直接当作结果。

### 0.6 JSON/CSV/PNG 路径与 Git

| 产物 | 路径 |
|---|---|
| 中文报告 | `v83_spatial_stage2/stage2q_trajectory_audit/STAGE2Q_REPORT.md` |
| 逐步轨迹 | `v83_spatial_stage2/stage2q_trajectory_audit/trajectory_steps.csv` |
| 轨迹汇总 | `v83_spatial_stage2/stage2q_trajectory_audit/trajectory_summary.csv` |
| 跨方法审计 | `v83_spatial_stage2/stage2q_trajectory_audit/trajectory_invariance.csv` |
| 检查点清单 | `v83_spatial_stage2/stage2q_trajectory_audit/checkpoint_inventory.csv` |
| 审计汇总 | `v83_spatial_stage2/stage2q_trajectory_audit/audit_summary.json` |
| 运行 provenance | `v83_spatial_stage2/stage2q_trajectory_audit/run_manifest.json` |
| 2×3 轨迹示例 | `v83_spatial_stage2/stage2q_trajectory_audit/trajectory_examples_2x3.png` |
| Stage-2n 方法对照 | `v83_spatial_stage2/stage2q_trajectory_audit/stage2n_method_overlay.png` |
| Stage-2o 奖励对照 | `v83_spatial_stage2/stage2q_trajectory_audit/stage2o_reward_overlay.png` |
| Stage-2p 奖励对照 | `v83_spatial_stage2/stage2q_trajectory_audit/stage2p_ppo_reward_overlay.png` |
| 差异审计图 | `v83_spatial_stage2/stage2q_trajectory_audit/trajectory_difference_audit.png` |
| 采集/绘图/测试代码 | `collect_trajectories.py` / `plot_trajectories.py` / `test_trajectory_audit.py` |

Git 提交：见文末 **§10**。主提交 SHA 为 `23b7b19c8f30395c585120b00e742eb606a16bee`，提交链接为
`https://github.com/JLU-MCNS-MEC/mec-v8.2/commit/23b7b19c8f30395c585120b00e742eb606a16bee`。

---

## 1. 原始轨迹数据检查

### 1.1 GitHub 侧

检查了以下文件/目录（均为 `experiment/v83-spatial-foundation` 分支已有内容）：

- `v83_spatial_stage2/STAGE2N_RESULTS.md`
- `v83_spatial_stage2/stage2n_evidence/navigation_invariance.csv`
- `v83_spatial_stage2/stage2n_evidence/figures/`
- `v83_spatial_stage2/stage2n_lanes.py`
- `v83_spatial_stage2/stage2n_eval.py`
- `v83_spatial_stage2/stage2o/evidence/`
- `v83_spatial_stage2/stage2p/evidence/`

结论：

- `navigation_invariance.csv` 的列是
  `policy,task_set_equal,action_mismatches,success_mismatches,step_mismatches,max_spl_abs_diff,max_ndtw_abs_diff,identical`，
  没有逐步坐标。
- `stage2n_eval.py` 写出的 `per_episode.csv` 只有
  `action_sha256, episode_uid, band, success, steps, mean_e2e_ms, p95_e2e_ms, latency_return` 等列；
  `_strip_row()` 明确去掉了 `actions`、`step_e2e_ms`、`node_choices`，也没有 pose。
- `stage2n_lanes.py` 的 `LaneStep` dataclass 有 `pose` 字段；`Lane` 有
  `steps/success/osr/spl/ndtw/action_sha256`。但 `build_lanes()` 返回的
  `Lane` 只用于内存评测，`stage2n_eval.py` 没有把 lane 序列化到磁盘。
- Stage-2o/2p 的 evidence 是训练曲线、每 200 Episode 验证、汇总表、诊断图；
  没有轨迹/坐标文件。

### 1.2 服务器侧

在以下目录搜索了 `*.npz`、`*.npy`、`*.pkl`、`*lane*`、`*traj*`、`*pose*`：

- `runs/v83_spatial_stage2/stage2n_latency_only/`
- `runs/v83_spatial_stage2/stage2o/`
- `runs/v83_spatial_stage2/stage2p/`

没有 Stage-2n/2o/2p 的 lane/pose/trajectory 缓存。仓库中确实存在
`.npz` 文件，但属于 Stage-2b/2c/2g 的特征/探针缓存，与本次轨迹无关。

### 1.3 坐标来源

本次真实坐标来自：

1. **frozen MEC lane 重建**：`build_lane()` 在真实环境中逐步执行 MEC，
   `LaneStep.pose` 记录每步决策前的 `env.pose = [x, y, yaw]`（float32）。
2. **6 任务完整闭环**：`LatencyOnlyEpisodeRunner` 每步由所选关联策略独立
   选择节点，`runner.step_data["pose"]` 记录决策前坐标，`runner.finalize()`
   返回真实后动作轨迹 `trajectory` 和 `step_records`。
3. **起点/目标/地标几何**：起点、目标来自冻结任务 spec；目标地标来自
   `build_city(MAP_SEED)` 的真实 `Landmark`（位置、`physical_size_m`、shape），
   不是虚构建筑。

`z` 的记录：环境是固定高度 80 m 的 2.5D 仿真，`env.pose` 只有
`[x, y, yaw]`，没有可变 z。`trajectory_steps.csv` 的 `z_m` 因此恒为
`ENV.uav_altitude_m = 80.0` m，并在报告中明确标注为固定高度，而不是每步
传感器读数。heading 是 `yaw`（弧度）。

---

## 2. 检查点、任务池、部署与复现审计

### 2.1 固定条件

| 项目 | 值 |
|---|---|
| 城市 | `map_seed = 10,000,001`，160 个地标，1500 m × 1500 m |
| 城市指纹 | `392500aa0df1d87b341f2fbd3df2fe4f2dc0405624a1ecc91cd33eca616bb182` |
| 拓扑 | Stage-2l `A_fixed` 的 20 个节点位置（拓扑种子 9,000,000） |
| MEC | 20 个完全同质节点，同一 Transformer-M（Stage-2j） |
| MEC 权重 SHA256 | `57c88b7d71e1a7f17862aaa8429afa8a041cba5e747f744f3a7febf47f273c6d` |
| Dev-60 任务文件 SHA256 | `1b0518b59f86c58c896d63a240629bedb78065fb059194d0fe123d658dc0653a` |
| Dev-60 tasks_sha256 | `aefde2aadaf443d32c70faebd74624431d165f6ef1b27d0fe3f1480572ade41d` |
| 部署文件 SHA256 | `11de01122d853d4e32a62965d92f4fadb8c9fc988d781907ad36823d3d19748c` |
| 部署规范内容 SHA256 | `4c0950cf6e438cff41454eee6daefb831fa984f425ec1a41cf3974257c6d1148` |
| 状态 | 384-D Pure-MEC v2 + 独立 20-D 可用性 Mask |
| 动作空间 | 20 个 MEC 节点 |
| 奖励 | 训练奖励随实验不同；评测 LatencyReturn 统一为 `-0.25 × mean(E2E ms) / 100` |
| Test-400 | 未读取 |

### 2.2 检查点选择

主结果一律使用各实验按 **Dev 平均 E2E 最低**选出的 `best.pt`
（并列时更高 SR、更早 Episode）。18 个 learned run 的 `last.pt` 全部存在，
但**没有**混入主比较；
这样避免把“最低 Dev E2E”和“最终 Episode”两种选择标准混在一起。若后续
需要最终检查点轨迹，可单独用 `last.pt` 重跑，但本次报告不声称覆盖它。

### 2.3 18 个 learned 检查点

`SHA256` 在下表中只显示前 16 位；完整 64 位哈希见
`checkpoint_inventory.csv` 的 `checkpoint_sha256` 列。

**Stage-2n（原始纯时延奖励）**

| 方法 | seed | 最佳 Dev E2E (ms) | @Ep | best.pt SHA256（前 16 位） | last.pt |
|---|---:|---:|---:|---|---|
| `2n-dqn-s0` | 0 | 75.7583 | 200 | `0d50093cfea030e8` | 存在 |
| `2n-dqn-s1` | 1 | 80.1130 | 200 | `c746538edf544879` | 存在 |
| `2n-dqn-s2` | 2 | 78.5271 | 200 | `63dcedac45f27d02` | 存在 |
| `2n-ppo-s0` | 0 | 62.6302 | 4000 | `de824f55e0467d11` | 存在 |
| `2n-ppo-s1` | 1 | 58.6988 | 4000 | `3a7d53ede99dcfe6` | 存在 |
| `2n-ppo-s2` | 2 | 58.3934 | 2600 | `19fa0dafc2dc72d6` | 存在 |

**Stage-2o（奖励 B / D / 方案一，seed 0）**

| 方法 | 奖励 | 最佳 Dev E2E (ms) | @Ep | best.pt SHA256（前 16 位） | last.pt |
|---|---|---:|---:|---|---|
| `2o-B-dqn-s0` | B | 73.0096 | 600 | `978cbb27e528a07b` | 存在 |
| `2o-B-ppo-s0` | B | 58.5116 | 3400 | `d010aa40b32fadc3` | 存在 |
| `2o-D-dqn-s0` | D | 83.3409 | 200 | `643ad55b911c7cd1` | 存在 |
| `2o-D-ppo-s0` | D | 61.0064 | 4000 | `1b5c6aaec0407a92` | 存在 |
| `2o-one-dqn-s0` | one | 87.2592 | 3400 | `2558215840f71d39` | 存在 |
| `2o-one-ppo-s0` | one | 65.9930 | 2800 | `f0c6174f9baef003` | 存在 |

**Stage-2p（PPO baseline / D，seed 0/1/2）**

| 方法 | 奖励 | 最佳 Dev E2E (ms) | @Ep | best.pt SHA256（前 16 位） | last.pt |
|---|---|---:|---:|---|---|
| `2p-base-ppo-s0` | baseline | 62.6302 | 4000 | `27db8252711a26c4` | 存在 |
| `2p-base-ppo-s1` | baseline | 58.6988 | 4000 | `1ed4f9f318b78c40` | 存在 |
| `2p-base-ppo-s2` | baseline | 58.3934 | 2600 | `bfb076d2c165e10f` | 存在 |
| `2p-D-ppo-s0` | D | 61.0064 | 4000 | `12a8e298eb6a69cd` | 存在 |
| `2p-D-ppo-s1` | D | 59.8767 | 3600 | `befadbd8be931bd8` | 存在 |
| `2p-D-ppo-s2` | D | 60.3846 | 3200 | `94b8dd6a8d147adc` | 存在 |

**对照方法**：`control-greedy-min-e2e`（确定性贪心最小时延）和
`control-random`（RNG 种子 `20261022 + episode_index`，与 Stage-2n 历史
random 基线一致）。对照没有检查点。

### 2.4 一个检查点溯源发现：文件 SHA 不同但模型权重相同

对 18 个 `state_dict` 逐张量做 SHA256 分组，得到 4 组完全相同的模型权重：

| model_state_sha256（前 16 位） | 方法 |
|---|---|
| `707cd38e4908bd50` | `2n-ppo-s0`, `2p-base-ppo-s0` |
| `7096506db8c30835` | `2n-ppo-s1`, `2p-base-ppo-s1` |
| `46cf4b1ae4aaee0f` | `2n-ppo-s2`, `2p-base-ppo-s2` |
| `49fc677e81949fd2` | `2o-D-ppo-s0`, `2p-D-ppo-s0` |

也就是说，Stage-2p 的 baseline PPO 三个 seed 和 Stage-2p 的 D/PPO seed0
在**模型张量层面**与 Stage-2n/2o 对应 run 完全相同；文件 SHA256 不同来自
优化器状态/元数据等非模型字段。

影响：

- 这几对方法的“轨迹一致”是**数学上必然**的，不能当作独立证据；
- 但审计中仍有大量**权重不同**的方法对：DQN vs PPO、B/D/one 三种奖励的
  DQN、B/D/one 三种奖励的 PPO、Stage-2p D seed1/2 vs baseline seed1/2 等。
  这些方法仍然逐任务轨迹完全一致，因此结论不依赖重复权重组。
- 报告在解释“一致性”时明确区分这两类证据。

### 2.5 部署哈希口径差异（不是实验条件变化）

Stage-2n 的 `protocol_freeze.json` 中 `deployment` 对象同时有：

- `sha256 = 11de0112…`：`deployment.json` 的**文件** SHA256；
- `deployment_sha256 = 4c0950…`：部署内容的**规范** SHA256。

Stage-2o/2p 的 `freeze_protocol.py` 把父协议的 `deployment["sha256"]`
复制到字段名 `deployment_sha256`，因此它们的 `protocol_freeze.json` 中
该字段显示 `11de0112…`；但 Stage-2o/2p 报告正文和父协议都写
`4c0950…`。本次核对确认：

- 两者指向**同一个** `deployment.json`；
- 文件实际 SHA256 = `11de0112…`；
- 文件内部规范 `deployment_sha256` = `4c0950…`；
- 20 个节点位置、同质性检查与 Stage-2n 完全一致。

这是**字段命名/记录口径**问题，不是部署或数据发生变化。已在
`run_manifest.json` 的 `deployment_note` 中记录。

### 2.6 历史复现

| 审计 | 策略数 | 比较数（任务或字段） | 最大绝对误差 | 通过 |
|---|---:|---:|---:|---|
| Stage-2n 逐 Episode（60 任务 × 8 策略） | 8 | 480（8×60） | 动作哈希/成功/步数/SR/SPL/nDTW/E2E/P95/违约率/LatencyReturn 全部 **0.0** | 是 |
| Stage-2n 汇总（10 个汇总字段） | 8 | 80（8×10） | **0.0** | 是 |
| Stage-2o/2p best.pt 验证指标 | 12 | 120（12×10） | **0.0** | 是 |

Stage-2n 的 8 个策略是 6 个 learned + Greedy + Random；Stage-2o/2p 的
12 个策略是 6 个 Stage-2o + 6 个 Stage-2p。所有 `best_dev_e2e_ms` 与
本次 Dev-60 lane 回放的平均 E2E 逐位一致（见 `checkpoint_inventory.csv`）。

---

## 3. 六条代表任务

选择规则：每个 band（Short/Medium/Long）× 成功/失败层内，按
`(steps, episode_index)` 升序，取 `(n-1)//2` 下中位。Dev-60 的层计数：

| band | 成功 | 失败 | 合计 |
|---|---:|---:|---:|
| Short | 12 | 8 | 20 |
| Medium | 13 | 7 | 20 |
| Long | 11 | 9 | 20 |
| 合计 | 36 | 24 | 60 |

六条样本明细：

| band | 结果 | Episode | ep_idx | task_seed | 步数 | 起点–目标 (m) | goal_idx | 目标地标尺寸/形状 | 起点 (x,y) | 最终 (x,y) | 层内候选 | 中位 rank |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|
| Short | 成功 | `stage2n-dev-00009` | 9 | 22000011 | 41 | 190.9 | 9 | 36.2 m / circle | (917.8, 483.4) | (936.0, 657.1) | 12 | 5 |
| Short | 失败 | `stage2n-dev-00008` | 8 | 22000010 | 232 | 171.0 | 21 | 37.3 m / cross | (182.6, 889.4) | (292.1, 1500.0) | 8 | 3 |
| Medium | 成功 | `stage2n-dev-00024` | 24 | 22000028 | 76 | 384.3 | 118 | 41.8 m / bar | (1174.4, 293.6) | (1210.2, 661.9) | 13 | 6 |
| Medium | 失败 | `stage2n-dev-00035` | 35 | 22000039 | 131 | 337.7 | 83 | 28.4 m / square | (1148.8, 1210.8) | (1386.9, 918.4) | 7 | 3 |
| Long | 成功 | `stage2n-dev-00045` | 45 | 22000054 | 104 | 446.3 | 65 | 25.9 m / triangle | (646.8, 633.9) | (561.7, 198.2) | 11 | 5 |
| Long | 失败 | `stage2n-dev-00057` | 57 | 22000071 | 137 | 428.9 | 8 | 40.3 m / triangle | (885.6, 114.7) | (1418.5, 119.4) | 9 | 4 |

说明：

- Short 失败样本最终 `y = 1500.0` 是世界边界；这是真实环境中的失败行为，
  不是绘图裁剪或虚构。
- 目标地标尺寸/形状来自 `build_city(map_seed=10,000,001)` 的真实
  `Landmark`；目标坐标与地标位置一致（采集时逐条断言）。
- 成功/失败是环境 `env.step()` 的 `success`（Stop 时距离目标 < 20 m），
  不是事后按 SPL 或图示判断。

---

## 4. 轨迹提取方法

### 4.1 三层证据

1. **Dev-60 lane 回放**：对全部 60 条 Dev 任务，用 `build_lanes()` 在真实
   环境中把 MEC 跑一遍，缓存每步 pose、20×19 候选特征、Mask、20 个节点的
   时延特征和导航动作；然后在该 lane 上回放 18 个 learned 策略和 2 个
   对照策略。回放只重新计算关联节点与 E2E，导航动作来自冻结 lane。
   这一层覆盖全部 60 任务，但“导航动作相同”在构造上是共享的，不能单独
   作为独立证据。
2. **6 任务真实闭环**：对六条代表任务，20 个方法各自从 reset 开始，每步
   独立选择 MEC 节点，重新执行 MEC 前向和环境 `step()`，记录决策前
   `pose`、动作、E2E、节点、距离目标、成功/done，并用 `finalize()` 取得
   真实后动作轨迹。这一层不共享导航动作，是独立验证。
3. **闭环 vs frozen lane**：把第 2 层的每一步坐标/动作与第 1 层的冻结 lane
   对比，验证 lane 构建本身是精确的（即 lane 回放在这些任务上等于完整
   闭环）。

三层都通过，才把 60 任务 lane 的结论推广为“Dev-60 上导航轨迹与关联策略
无关”。

### 4.2 坐标与哈希定义

- 世界坐标：固定城市 1500 m × 1500 m，原点在 (0,0)，x/y 单位米。
- `pose = [x, y, yaw]`，来自 `env.pose`；`z_m = 80.0` 是固定高度。
- 动作：`0=Forward, 1=Left, 2=Right, 3=Stop`；Forward 5 m，Left/Right
  15°，Stop 保持位置。
- 动作哈希：`sha256(",".join(str(action)))`，与 Stage-2n 历史完全一致。
- 决策坐标哈希：float32 `(T,3)` 数组 `[x,y,heading]` 的 SHA256。
- 路径坐标哈希：float32 `(T+1,2)` 后动作轨迹 `[x,y]` 的 SHA256。
- 节点哈希：`sha256(",".join(str(node_index)))`。
- 最大坐标偏差：同一任务、同一 step 的 `[x,y]` 欧氏距离的最大值；
  同时记录 heading 偏差（wrap 到 `[-π, π]`）。

### 4.3 路径重建校验

为了从 `LaneStep.pose + action` 得到真实后动作路径，采集器实现了
`advance_pose()`，逐条复现 `ToyUAVEnv.step()` 的 Forward/Left/Right/Stop、
x/y 裁剪和 yaw wrap。校验：

- 重建路径的下一步坐标与 lane 下一步决策坐标最大偏差约
  1.2e-4 m（float32 pose 舍入量级），远低于 1e-3 m 阈值；
- 重建 SPL 与 lane `spl` 的差为 0.0；
- 6 任务真实闭环与 frozen lane 的决策坐标最大偏差为 0.0 m。

---

## 5. 跨方法一致性结果

### 5.1 三层审计总表

| 证据层级 | 比较数 | 动作不一致 | 坐标哈希不一致 | 步数不一致 | 成功状态不一致 | 最大坐标偏差 | 最大 heading 偏差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dev-60 lane（60 × 20） | 1200 | 0 | 0 | 0 | 0 | 0.000000 m | 0.0 rad |
| 6 任务真实闭环（6 × 20） | 120 | 0 | 0 | 0 | 0 | 0.000000 m | 0.0 rad |
| 闭环 vs frozen lane（6 × 20） | 120 | 0 | 0 | 0 | 0 | 0.000000 m | 0.0 rad |

### 5.2 六条代表任务的唯一动作哈希

20 个方法在每条任务上共用同一个动作哈希；同时 `decision_coord_sha256`
和 `path_coord_sha256` 也各只有一个唯一值。完整哈希：

```text
stage2n-dev-00009  133624fbd5728feceb3b0b7e9e69026f09daca4583e7e85b0ab2ae70c9671d8c
stage2n-dev-00008  150564bad9f174c6bde59d05f965d9011ca29a8e7d191493ae5867b106546617
stage2n-dev-00024  4724fcdfa81846acf38a338d0613296cc8f38553854eeeae2a543ec9bea92d7c
stage2n-dev-00035  af248306ba0e45743d616deabb19af838a856d5d966b90146366f597710a5212
stage2n-dev-00045  184e767390e1e312df0eb0c17407fb693932c10529ca831a26f32e92d4e6087d
stage2n-dev-00057  0a6ac9feb11f4071d01bdf5a238e824c88f34fbf0fc0910c98ae39b7e94a9b99
```

`trajectory_invariance.csv` 中每行都带 `action_sha256`、
`reference_action_sha256`、`coord_sha256`、`reference_coord_sha256`、
`actions_equal`、`coord_hash_equal`、`max_coord_deviation_m`、
`node_choice_mismatch_steps`、`e2e_delta_ms`，可以逐任务复核。

### 5.3 关联层确实不同

Dev-60 lane 回放（60 任务平均）：

| 方法 | Dev-60 平均 E2E (ms) | 相对 2n-DQN-s0 Δ (ms) | 节点选择不一致步数占比 |
|---|---:|---:|---:|
| `2n-dqn-s0` | 75.76 | 0.00 | 0.0% |
| `2n-dqn-s1` | 80.11 | +4.35 | 79.5% |
| `2n-dqn-s2` | 78.53 | +2.77 | 73.3% |
| `2n-ppo-s0` | 62.63 | −13.13 | 71.9% |
| `2n-ppo-s1` | 58.70 | −17.06 | 78.6% |
| `2n-ppo-s2` | 58.39 | −17.36 | 74.1% |
| `2o-B-dqn-s0` | 73.01 | −2.75 | 69.2% |
| `2o-B-ppo-s0` | 58.51 | −17.25 | 63.7% |
| `2o-D-dqn-s0` | 83.34 | +7.58 | 64.2% |
| `2o-D-ppo-s0` | 61.01 | −14.75 | 68.6% |
| `2o-one-dqn-s0` | 87.26 | +11.50 | 81.9% |
| `2o-one-ppo-s0` | 65.99 | −9.77 | 70.8% |
| `2p-base-ppo-s0` | 62.63 | −13.13 | 71.9% |
| `2p-base-ppo-s1` | 58.70 | −17.06 | 78.6% |
| `2p-base-ppo-s2` | 58.39 | −17.36 | 74.1% |
| `2p-D-ppo-s0` | 61.01 | −14.75 | 68.6% |
| `2p-D-ppo-s1` | 59.88 | −15.88 | 67.7% |
| `2p-D-ppo-s2` | 60.38 | −15.37 | 76.3% |
| `control-greedy-min-e2e` | 49.18 | −26.58 | 62.6% |
| `control-random` | 95.43 | +19.68 | 80.0% |

6 条代表任务的真实闭环（每任务 20 个方法）：

| 任务 | band/结果 | 不同节点序列 / 20 | 平均 E2E 最小 (ms) | 平均 E2E 最大 (ms) | 跨度 (ms) |
|---|---|---:|---:|---:|---:|
| `stage2n-dev-00009` | Short 成功 | 14 | 43.00 | 110.73 | 67.73 |
| `stage2n-dev-00008` | Short 失败 | 16 | 48.36 | 85.78 | 37.43 |
| `stage2n-dev-00024` | Medium 成功 | 15 | 38.06 | 91.58 | 53.52 |
| `stage2n-dev-00035` | Medium 失败 | 16 | 42.01 | 95.30 | 53.29 |
| `stage2n-dev-00045` | Long 成功 | 16 | 59.72 | 127.68 | 67.96 |
| `stage2n-dev-00057` | Long 失败 | 15 | 36.64 | 83.85 | 47.20 |

结论：20 个方法在同一任务上确实选择了不同节点（每任务 14–16 种序列），
E2E 差异可达 68 ms；但导航动作/坐标/步数/成功状态完全不变。

### 5.4 哪些任务/方法不同？

- **同一任务跨方法**：没有不同轨迹；所有 20 个方法逐 step 坐标完全重合。
- **不同任务之间**：轨迹不同，因为起点、目标、band 不同。例如六条代表
  任务的起点/终点分别为 Short 成功 (917.8,483.4)→(936.0,657.1)、
  Medium 失败 (1148.8,1210.8)→(1386.9,918.4)、Long 失败
  (885.6,114.7)→(1418.5,119.4) 等；这属于任务差异，不是方法不一致。
- **权重相同的方法对**：§2.4 的 4 组（2n/2p baseline PPO、2o/2p D PPO s0）
  轨迹相同是必然；权重不同的 DQN/PPO、B/D/one、2p D s1/s2 也相同，才是
  本次审计的独立证据。
- **没有需要定位的“某 Episode 某 step 动作不一致”**：三层审计的
  `actions_equal` 全部为 true，`max_coord_deviation_m` 全部为 0.0；不存在
  可定位的分歧点。若有分歧，`trajectory_invariance.csv` 的逐任务行会给出
  具体 Episode、步数、节点哈希和 E2E 差。

---

## 6. 图表说明

### 6.1 `trajectory_examples_2x3.png`

2 行（成功/失败）× 3 列（Short/Medium/Long）。每个面板：

- 背景是固定城市真实地标足迹和道路网格（来自 `city_geometry.csv` /
  `city_meta.json`），没有虚构建筑或道路；
- 真实路径只画一条（参考方法 `2n DQN s0`），因为 20 个方法逐位重合；
- 绿色=成功路径，红色=失败路径；
- 三角形=起点，星形=目标地标，方块=成功终点，X=失败终点；
- 虚线圆=20 m 停止半径；
- 标题标注 band、成功/失败、Episode ID、步数、起点–目标距离；
- 图级说明标注参考方法和全部通过一致性审计的方法组；没有人为偏移轨迹。

### 6.2 `stage2n_method_overlay.png`

Stage-2n 的 6 个 DQN/PPO seed 在同一任务（`stage2n-dev-00009`）上的对照。
左侧只画一条真实路径（6 个方法重合），右侧画每步 E2E 时延（6 条曲线），
图例给出每个 seed 的平均 E2E。路径颜色不用于区分方法；方法差异只由右侧
E2E 曲线承载。

### 6.3 `stage2o_reward_overlay.png`

Stage-2o 的 B/D/方案一 × DQN/PPO（6 个方法）在同一任务
（`stage2n-dev-00024`）上的对照。左路径、右 E2E，布局同 6.2。
该任务上 6 个方法的 E2E 差异最明显（跨度 49.71 ms），因此选作对照。

### 6.4 `stage2p_ppo_reward_overlay.png`

Stage-2p 的 baseline/D × PPO seed 0/1/2（6 个方法）在同一任务
（`stage2n-dev-00009`）上的对照。左路径、右 E2E；该任务上 6 个方法的
E2E 跨度为 31.91 ms。

### 6.5 `trajectory_difference_audit.png`

四个面板：

- 左上：三层审计总表（动作/坐标哈希/步数/成功/max Δ）；
- 右上：20 个方法相对 `2n-DQN-s0` 的逐步最大坐标偏差，全部为 0.000000 m
  （x 轴范围 ±1e-9 m，明确展示零偏差，而不是把它藏起来）；
- 左下：20 个方法的节点选择不一致比例（相对参考），显示真实关联差异；
- 右下：20 个方法的平均 E2E 差（相对参考），显示真实时延差异。

---

## 7. 差异来源分析

### 7.1 为什么导航轨迹必然相同

- 20 个 MEC 节点加载**同一个** Stage-2j Transformer-M 检查点，权重 SHA256
  为 `57c88b7d…`；算力 2.0 TFLOPS、带宽 30 MHz、覆盖半径 500 m 全部相同，
  只有 `(x,y)` 位置不同。
- MEC 导航输入是当前 RGB tokens、goal tokens、history visual/pose/action/mask、
  GAB、map feature；**没有节点编号、节点位置、关联动作或 E2E 时延**。
- 关联策略的动作只决定“这一步用哪个节点记账”，不会改变 MEC 的
  `move_logits/arrival_probability`，也不会改变 `env.step()` 的动力学。
- 因此同一任务的动作序列、坐标、步数、成功/失败与关联策略无关；只有
  每步 E2E、平均 E2E、P95、时延违约率和节点选择序列随策略变化。

### 7.2 实证而不是假设

- 6 任务 × 20 方法真实闭环：每步重新选择节点、重新执行 MEC 与环境，
  动作/坐标/步数/成功全部一致；节点序列每任务 14–16 种。
- 闭环与 frozen lane 逐 step 坐标偏差 0.0 m，说明 lane 回放不是
  “另一条近似路径”。
- Dev-60 lane 的 1200 次比较全部通过；这是覆盖全部 Dev-60 的广覆盖证据。

### 7.3 适用范围

该结论只适用于当前**同质部署 + 节点身份不进入 MEC 输入**的设定。
如果以后引入异质算力/带宽/模型、把节点身份拼进 MEC 输入、或让导航模型
接收关联动作/时延反馈，则关联策略可能改变导航动作；不能用本报告的结论
外推。报告和 Stage-2n/2o/2p 协议都已声明这一限制。

---

## 8. 限制与声明

1. **单城市、单部署、单 MEC 检查点**：`map_seed=10,000,001`、A_fixed 20
   节点、Stage-2j Transformer-M；不外推到新城市或异质 MEC。
2. **只用 Dev-60**：未读取 Test-400；6 条代表任务用于完整闭环，全部 60
   条用于 lane 回放。6 条样本是按 band × 成功/失败的中位规则选的，不是
   随机抽样；选择规则和候选数已记录。
3. **只用 best.pt**：`last.pt` 全部存在但未混入主比较；若要用最终检查点，
   需要单独跑并单独标注。
4. **没有重新训练**：只 `load_model` + `eval()`，没有 optimizer step、
   没有更新检查点、没有覆盖 Stage-2n/2o/2p 产物。
5. **float32 坐标**：本次所有最大偏差为 0.0 m；不同 GPU/驱动重跑可能
   出现 ≤1e-6 m 的浮点差异，审计阈值设为 1e-9 m。若未来重跑超过该阈值，
   应按 `trajectory_invariance.csv` 的 Episode/step 定位，而不是归因于绘图。
6. **地图背景**：只使用环境真实地标足迹和道路网格，没有卫星影像或虚构
   建筑；`city_geometry.csv` 可复核。
7. **重复权重**：§2.4 的 4 组方法不是独立证据；结论依赖权重不同的方法对。
8. **landmark 位置/形状**：目标地标来自固定城市几何；目标点就是该地标的
   位置，成功半径 20 m。图中星形/虚线圆是目标与停止半径，不是额外传感器。

---

## 9. 复现

```bash
cd /home/ubuntu5/Workspace/MEC-V8.2

# 1) 采集（单卡；本次 364.4 s）
OMP_NUM_THREADS=2 MPLBACKEND=Agg \
  /home/ubuntu5/miniconda3/envs/DRL/bin/python -u \
  v83_spatial_stage2/stage2q_trajectory_audit/collect_trajectories.py \
  --device cuda

# 2) 绘图（CPU；只读本目录 CSV）
/home/ubuntu5/miniconda3/envs/DRL/bin/python \
  v83_spatial_stage2/stage2q_trajectory_audit/plot_trajectories.py

# 3) 测试（CPU；20 项）
/home/ubuntu5/miniconda3/envs/DRL/bin/python -m unittest \
  v83_spatial_stage2.stage2q_trajectory_audit.test_trajectory_audit -v
```

本次运行的日志：`runs/v83_spatial_stage2/stage2q_collect.log`（被 `.gitignore`
排除）。`run_manifest.json` 记录 Python/torch/GPU、任务池/部署/MEC 哈希、
18 个检查点路径与 SHA256、历史复现结果、输出 CSV/JSON 的 SHA256。

---

## 10. 产物与 Git 提交

### 10.1 文件清单

| 文件 | 说明 |
|---|---|
| `STAGE2Q_REPORT.md` | 本报告 |
| `trajectory_steps.csv` | 6 任务 × 20 方法的逐步真实坐标/动作/E2E/节点 |
| `trajectory_summary.csv` | 每条轨迹的起终点、目标地标、成功/步数/SPL/nDTW、检查点、动作/坐标哈希 |
| `trajectory_invariance.csv` | 三层级的动作/坐标哈希、步数/成功差异、最大偏差、节点/E2E 差异 |
| `dev60_task_summary.csv` | 全部 60 条 Dev 任务的 lane 级上下文 |
| `selected_tasks.csv` | 6 条代表任务及选择规则/候选数/中位 rank |
| `checkpoint_inventory.csv` | 18 个 best.pt 的路径、SHA256、Dev 指标、模型权重哈希、last.pt |
| `city_geometry.csv` / `city_meta.json` | 160 个真实地标、道路网格、停止半径、固定高度 |
| `audit_summary.json` | 一致性、节点多样性、重复权重、历史复现的汇总数字 |
| `run_manifest.json` | 运行 provenance 和输出文件哈希 |
| `collect_trajectories.py` | 采集器 |
| `plot_trajectories.py` | 绘图脚本 |
| `test_trajectory_audit.py` | 20 项坐标/分层/一致性测试 |
| `trajectory_examples_2x3.png` | 2×3 成功/失败 × Short/Medium/Long |
| `stage2n_method_overlay.png` | Stage-2n 6 方法同任务对照 |
| `stage2o_reward_overlay.png` | Stage-2o B/D/one 6 方法同任务对照 |
| `stage2p_ppo_reward_overlay.png` | Stage-2p baseline/D 6 方法同任务对照 |
| `trajectory_difference_audit.png` | 三层审计与零偏差/节点/E2E 差异图 |

原始检查点、TensorBoard、训练日志、lane 构建日志仍在 `.gitignore` 排除的
`runs/` 下，没有上传到 Git；没有覆盖 Stage-2n/2o/2p 历史产物。

### 10.2 Git

- 分支：`experiment/v83-spatial-foundation`
- Stage-2q 主提交（代码 + CSV + PNG + 报告初版）：`23b7b19c8f30395c585120b00e742eb606a16bee`
- 主提交链接：`https://github.com/JLU-MCNS-MEC/mec-v8.2/commit/23b7b19c8f30395c585120b00e742eb606a16bee`
- 目录链接：`https://github.com/JLU-MCNS-MEC/mec-v8.2/tree/experiment/v83-spatial-foundation/v83_spatial_stage2/stage2q_trajectory_audit`
- 报告回填 SHA 的后续提交只更新本文件，不改变任何 CSV/PNG/代码。
