# Stage-2m 协议：DQN Q 值高估诊断 + 固定城市 A 拓扑视觉增强

日期：2026-10-21  
分支：`experiment/v83-spatial-foundation`  
代码根：`v83_spatial_stage2/`  
运行根：`runs/v83_spatial_stage2/stage2m_q_audit/`、
`runs/v83_spatial_stage2/stage2m_visual_fixed_city/`

本协议在 Stage-2m 正式训练启动前冻结。已有 Stage-2l A/B/C 的权重、日志、
TensorBoard、Checkpoint 与报告全部保留，不停止、不覆盖、不重启（B 的两个
进程仍为 SIGSTOP 状态）。Stage-2l Final Test 60 条任务继续封存，本阶段不读取。

---

## 1. 实验一：DQN Q 值高估诊断

### 1.1 问题

1. 无效动作 Mask、Bellman Target、终止状态与奖励计算是否正确？
2. 固定 Dev 任务上，预测 Q 与 Monte-Carlo 实际折扣回报的偏差、MAE、
   Q_mean / Q_max 是多少？
3. 相同数据、状态、任务顺序与随机种子下，Double DQN 是否降低高估并改善收敛？

### 1.2 审计方法

- 固定 Dev 60 条任务（20 Short / 20 Medium / 20 Long），配置 A、固定城市
  `map_seed=10,000,001`、固定训练/验证拓扑。
- 对每个 Checkpoint 用其贪心策略跑完 60 条 Dev，逐状态记录
  `Q(s,·)`、可用 Mask、实际动作与每步奖励；按 `G_t = Σ_{k>=t} r_k`
  （γ=1.0）计算 Monte-Carlo 回报。
- 主判据：`bias_t = Q(s_t,a_t) − G_t`；统计 mean / median / MAE / RMSE /
  高估比例 / Pearson 相关 / Q_mean / Q_max。
- 补充判据：同一组确定性 Min-Latency 行为状态集上评估所有 Checkpoint，
  用于 DQN 与 Double DQN 的跨算法相对比较（该表的 G_t 是行为策略回报，
  因此包含策略改进项，只作相对比较）。
- 终止/奖励/Mask 审计：终止转换必须 `next_state=None`、`next_mask=None`、
  target 不 bootstrap；每步 Mask 至少一个可用动作；所有 Episode 的
  累计奖励与 JointR 误差必须 ≤1e-6。

### 1.3 实现问题与修复

审计确认的修复（仅影响 target 构造）：

- 旧实现：`next_q.masked_fill(~next_mask, 0.0).max()`。当所有可用动作的
  Q 都为负时，不可用的“0 动作”会赢得 max，产生向上偏置。
- 修复：不可用动作填 `-1e9`，仅当 Mask 全空时回退到 0；Double DQN 由
  online 网络在可用动作中选 argmax、target 网络评估该动作。
- `DQNAgentV2.config()` 原先硬编码 `seed=SEED`，改为记录实际 seed。

保留修复前后对照：

- `pre`：Stage-2l A/DQN `best.pt`（Ep2400）与 `last.pt`（Ep3500）的 Q 审计；
- `dqn_legacy`：2000 Episodes、seed 0、`--legacy-target` 的训练对照；
- `dqn` / `dqn_double`：2000 Episodes、seed 0/1/2、修复后 target 的对照。

### 1.4 Double DQN 训练对照

- 数据、状态（384-D Pure MEC v2）、任务顺序、奖励、ε 预算、网络结构、
  优化器、验证代码完全相同；唯一差异是 Bellman Target 的 max 规则。
- 3 个 seed（0/1/2），2000 Episodes，每 200 Episodes 固定 Dev60 验证。
- DQN 对照直接复用实验二的 `dqn_pure` 三个 seed，避免重复训练。

---

## 2. 实验二：固定城市 A 拓扑视觉增强

### 2.1 固定条件

- 环境：Stage-2l `FixedCityEnv`，`map_seed=10,000,001`；
- 配置 A：`A_fixed`（seed 9,000,000，balanced，20 节点）同时用于训练与验证；
- 任务池：Train 3500 全局乱序，本实验按同一顺序取前 2000 条；Dev 60 固定；
- 奖励：`r_nav + (−0.25/100)·Δrunning_mean_E2E`，无成功奖励/失败惩罚；
- 五种 Stage-2j T5 MEC 全部冻结（`eval()` + `requires_grad=False`）；
- 评估：每 200 Episodes 在 60 条 Dev 任务上贪心评估 SR/OSR/SPL/nDTW/
  JointR/E2E；Final Test 不读取。

### 2.2 四组对照

| 算法 | 状态 | MLP 输入 | Seed |
|---|---|---|---:|
| DQN | Pure MEC | 384-D | 0,1,2 |
| DQN | +128-D 视觉 | 512-D | 0,1,2 |
| PPO | Pure MEC | 384-D | 0,1,2 |
| PPO | +128-D 视觉 | 512-D | 0,1,2 |

- 视觉特征：冻结的 Stage-2k `VisualProjector`（`visual_pca.npz`，
  SHA256 `e3cf2e84935d8a82108f7aa8fa54395c39186b90383da58df3c9fa72cdd153bd`），
  由 Train-600 自然状态拟合，`final_test_read=False`；
- 投影前 GAB/map 辅助块置为训练均值，128-D 只含 current/goal/history 视觉；
- 不训练视觉骨干，不新增 GAB/热图/新模块，不加入目标位置/距离/Oracle/成功标签；
- 512-D = `[visual128, pure_mec_384]`，可用性 Mask 仍为独立 20-D。

### 2.3 预算与随机性

- 每种算法/状态 2000 Episodes；每 200 Episodes 验证一次；不早停。
- 四个 arm 使用相同的训练任务顺序（由 Stage-2l manifest 固定）与相同的
  0/1/2 agent seed；seed 只影响网络初始化、探索与 replay 采样。
- DQN：γ=1.0，replay 50k，target update 1000，batch 128，ε 1.0→0.05/500k，
  Adam lr=1e-4，grad clip 10。
- PPO：γ=1.0，GAE λ=0.95，clip 0.2，rollout 2048，minibatch 256，
  10 epochs，lr 3e-4（沿用 Stage-2k PPO 常量）。

### 2.4 记录与判定

- 训练：MA100 return、JointR、SR、nDTW、E2E；DQN 额外记录 Q_mean/Q_max/
  target_mean/TD loss；PPO 额外记录 Entropy/KL/Value loss/explained variance。
- 验证：Dev60 的 JointR、SR、OSR、SPL、nDTW、E2E、P95、MEC 使用率。
- 视觉增益：同算法、同 seed 的配对差（visual − pure），报告 3-seed 均值、
  逐 seed 差与 bootstrap 95% CI；同时比较收敛速度（达到 best 的 episode）
  与稳定性（验证曲线后半段方差）。
- 收敛判定沿用 `classify_status`：最近 3 次验证平台 ≤0.01 且最后 ≥best−0.005
  才算 `converged`；否则明确报告 `plateaued_not_converged`。
- 不以单 seed 或单次验证下结论。

---

## 3. 运行与产物

- 启动器：`python v83_spatial_stage2/stage2m_launch.py --group wave1`
  （wave2 在 wave1 释放 GPU 后启动）；
- 每个 run 独立目录，含 `checkpoints/`（best/last/last_full）、
  `training_history.json`、`validation.csv`、`protocol.json`、`commands.json`、
  `progress.json`、`tensorboard/`；
- Q 审计输出 `q_audit.json`、`per_step_<name>.csv`、`REPORT.md`；
- 总报告：`STAGE2M_RESULTS.md`，不覆盖 Stage-2l 任何文件。

## 4. 资源

- 单卡 RTX 5090 32 GB；已有两个 Stage-2l B 进程处于 SIGSTOP，保留约 6 GB；
- 每个训练进程约 2.5–3 GB 显存、2 个 torch 线程；
- 启动器在每次拉起前检查 `nvidia-smi` 空闲显存，低于阈值则等待，避免 OOM。
