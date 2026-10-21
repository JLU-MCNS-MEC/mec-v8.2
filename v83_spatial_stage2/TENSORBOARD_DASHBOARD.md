# Stage-2k / Stage-2l TensorBoard 汇总看板

日期：2026-10-21  
项目目录：`/home/ubuntu5/Workspace/MEC-V8.2`  
Conda 环境：`DRL`  
分支：`experiment/v83-spatial-foundation`

本看板汇总六组已完成实验的原始记录，统一使用真实训练 Episode 作为横轴；
**没有停止或修改任何训练任务、原始 Event、Checkpoint 或 `training_history.json`**。
B/DQN、B/PPO 仍处于暂停状态（分别约 1,030 / 1,120 Episode），不在本看板的六组
已完成实验中。

## 1. 六组实验

| TensorBoard Run | 算法 | 状态 | 环境 | MEC 拓扑 | 验证 |
|---|---|---|---|---|---|
| `01_Stage2k_V2_PureMEC_DQN_384D` | DQN | Pure MEC v2，384-D + 20-D mask | Stage-2k V2 ToyUAVEnv（每 Episode 生成城市） | Balanced / Quality-rich / Tradeoff-stress，每场景一张固定 Train 拓扑 | 60 Dev（6/10/4） |
| `02_Stage2k_V2_PureMEC_Visual_DQN_512D` | DQN | 128-D frozen VisualProjector + 384-D Pure MEC = 512-D + 20-D mask | Stage-2k V2 ToyUAVEnv（每 Episode 生成城市） | 同上，每场景一张固定 Train 拓扑 | 60 Dev（6/10/4） |
| `03_Stage2l_A_DQN` | DQN | Pure MEC v2，384-D + 20-D mask | Stage-2l 固定城市 `map_seed=10,000,001` | Balanced 单张固定拓扑 `A_fixed` | 60 Dev（20/20/20） |
| `04_Stage2l_A_PPO` | PPO | Pure MEC v2，384-D + 20-D mask | Stage-2l 固定城市 `map_seed=10,000,001` | Balanced 单张固定拓扑 `A_fixed` | 60 Dev（20/20/20） |
| `05_Stage2l_C_DQN` | DQN | Pure MEC v2，384-D + 20-D mask | Stage-2l 固定城市 `map_seed=10,000,001` | Balanced / Quality-rich / Tradeoff-stress，各一张固定拓扑 | 60 Dev（20/20/20） |
| `06_Stage2l_C_PPO` | PPO | Pure MEC v2，384-D + 20-D mask | Stage-2l 固定城市 `map_seed=10,000,001` | 同上，三种部署各一张固定拓扑 | 60 Dev（20/20/20） |

## 2. 服务状态（已实际验证）

- 监听：`127.0.0.1:6006`
- tmux session：`tb_stage2l_dashboard`
- 进程 PID：`588396`（本次启动；以 `tensorboard_service.json` 为准）
- 采样参数：`--samples_per_plugin scalars=5000`
- 日志：`runs/tensorboard/rl_stage2k_stage2l_dashboard/tensorboard.log`
- 汇总事件：`runs/tensorboard/rl_stage2k_stage2l_dashboard/events/`
- 加载报告：`runs/tensorboard/rl_stage2k_stage2l_dashboard/LOAD_REPORT.json`
- 数据级验证：`runs/tensorboard/rl_stage2k_stage2l_dashboard/SERVICE_VERIFICATION.json`
- 服务状态文件：`runs/tensorboard/rl_stage2k_stage2l_dashboard/tensorboard_service.json`

服务器端实测：

```bash
curl -I http://127.0.0.1:6006/
# HTTP/1.1 200 OK

curl -s http://127.0.0.1:6006/data/runs
# ["01_Stage2k_V2_PureMEC_DQN_384D", "02_Stage2k_V2_PureMEC_Visual_DQN_512D",
#  "03_Stage2l_A_DQN", "04_Stage2l_A_PPO", "05_Stage2l_C_DQN", "06_Stage2l_C_PPO"]

python v83_spatial_stage2/verify_tensorboard_dashboard.py \
  --url http://127.0.0.1:6006
# SERVICE_VERIFICATION: PASS
# 每个 Run 的 train/jointr 实测 3500 点、validation/jointr 18 点
```

**采样参数必须用 5000，不能用 0。** 实测发现 TensorBoard 2.21 会把
`--samples_per_plugin scalars=0` 解释为“保留 0 个样本”：HTTP 200、Run 列表和
Tag 列表都正常，但所有标量曲线为空。改用 5000 后六个 Run 的 3500 个真实训练点
全部返回，没有降采样。

## 3. VS Code Remote SSH 端口转发

1. 用 VS Code Remote-SSH 连接到服务器。
2. 打开 **Ports** 面板（`Terminal: Focus on Ports`，或底部面板的 Ports 标签）。
3. 点击 **Forward a Port**，输入 `6006`。
4. 在本机浏览器打开：`http://localhost:6006/`。

TensorBoard 只监听服务器 `127.0.0.1`，所以必须经过 VS Code 的端口转发；
在服务器以外不能直接访问。若本机 6006 已被占用，VS Code 可能分配另一个本地端口，
以 Ports 面板显示的本地端口为准。服务器端 6006 已测试 HTTP 200；本机浏览器地址
依赖 VS Code 转发是否成功。

## 4. 曲线标签与数据来源

### 所有 Agent

- `train/jointr`：逐 Episode JointR
- `train/jointr_ma100`：从 Episode 100 起的 100-Episode 移动平均
- `train/navigation_reward`、`train/latency_penalty`（逐 Episode）
- `train/navigation_reward_ma100`、`train/latency_penalty_ma100`
- `train/sr`、`train/ndtw`、`train/e2e_ms`（逐 Episode）
- `train/sr_ma100`、`train/ndtw_ma100`、`train/e2e_ms_ma100`

### DQN

- `dqn/td_loss`（SmoothL1 TD Loss）
- `dqn/q_mean`、`dqn/q_max`
- `dqn/grad_norm`
- `dqn/epsilon`

### PPO

- `ppo/policy_entropy`：逐 Episode，来自 `training_history.json`
- `ppo/policy_loss`、`ppo/value_loss`、`ppo/approx_kl`、`ppo/clip_fraction`、
  `ppo/explained_variance`、`ppo/update_entropy`：从原始 TensorBoard 的
  `train/update_*` 事件重放，步数是真实更新 Episode（A/PPO 222 点，C/PPO 218 点）。
  如果原始事件缺失，脚本会回退到每 200 Episode 的验证记录并加
  `_validation_cadence` 后缀。

### Validation（每 200 Episodes）

- `validation/jointr`、`validation/sr`、`validation/osr`、`validation/spl`、
  `validation/ndtw`、`validation/e2e_ms`
- 固定 18 个点（200、400、…、3400、3500）；不是逐 Episode 曲线，未插值。

## 5. 指标完整性

六组实验的预期指标均已加载；`LOAD_REPORT.json` 中 `missing_metrics` 全部为空。
需要特别说明的差异：

- DQN 没有 PPO 的 policy/value/KL 曲线；PPO 没有 DQN 的 Q/ε/TD Loss 曲线，
  这是算法差异，不是缺失。
- PPO 的 policy/value/KL 是 **update 粒度**（222/218 点），不是逐 Episode。
- 验证曲线只有 18 个点，不能与逐 Episode 训练 rewards 混为一条曲线。
- Stage-2k V2 的环境是每 Episode 重新生成城市；Stage-2l 是固定城市。两者在同一
  TensorBoard 中仅用于结构对比，不能把绝对值直接当作同一环境下的改进。

### 已发现的 Q 值异常（真实记录，未修图）

`02_Stage2k_V2_PureMEC_Visual_DQN_512D` 存在明显的 Q 值高估/发散：

- `dqn/q_mean`：前 500 Episode 中位数约 3.6，最后 500 Episode 中位数约 944.7，
  末尾单点 1,726.5，全程最大 2,134.2。
- `dqn/q_max` 全程最大 2,446.3。
- `dqn/td_loss` 中位数从约 0.008 升到约 15.6，最大 22.0；`dqn/grad_norm`
  最大 192.4（记录的是裁剪前范数，实际裁剪阈值 10）。
- 同期训练 SR 仍约 0.49–0.55，末尾验证 SR 为 0.50，说明行为指标没有同步
  崩溃，但 Q 值已经不能作为可靠的价值尺度。

其余三个 DQN 的 Q 值也随训练上升，但终点 `q_mean` 分别为 27.1、9.7、11.1；
视觉臂是数量级异常。DQN 使用 `gamma=1.0`，Q 是未折扣的未来奖励和，增长本身可预期；
但视觉臂的幅度、TD Loss 与梯度同时抬升表明存在数值高估/发散。根因尚未定位，
优先排查视觉特征尺度、目标网络稳定性与 replay 分布；这是训练侧问题，不是
TensorBoard 缺数据或绘图错误。

## 6. 重新生成与启动

```bash
cd /home/ubuntu5/Workspace/MEC-V8.2
/home/ubuntu5/miniconda3/envs/DRL/bin/python \
  v83_spatial_stage2/stage2l_tensorboard_dashboard.py

bash v83_spatial_stage2/start_tensorboard_dashboard.sh
```

启动脚本会优先使用 6006；被占用时依次尝试 6007–6012，并写入
`tensorboard_service.json`。它还会调用
`verify_tensorboard_dashboard.py` 检查**实际标量点**（不是只看 HTTP 200）；
如果发现旧服务返回空曲线，会自动重启。TensorBoard 使用
`--samples_per_plugin scalars=5000`，在 tmux 中运行，关闭 SSH 终端后仍保留。

重新生成只替换 `runs/tensorboard/rl_stage2k_stage2l_dashboard/events/`，
不会改写原始 Event、Checkpoint 或 `training_history.json`。`runs/` 不进入
Git，Event 文件保留在服务器。

## 7. 配置文件

- `v83_spatial_stage2/tensorboard_dashboard_config.json`：六组实验、标签、采样节奏
- `v83_spatial_stage2/stage2l_tensorboard_dashboard.py`：汇总/重放脚本
- `v83_spatial_stage2/start_tensorboard_dashboard.sh`：持久化启动脚本
- `v83_spatial_stage2/verify_tensorboard_dashboard.py`：数据级服务检查（HTTP 外的标量点校验）
