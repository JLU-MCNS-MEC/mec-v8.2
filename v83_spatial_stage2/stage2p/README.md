# Stage-2p：PPO 原始奖励 vs 方案 D

本目录用**同一个 Stage-2p PPO 训练循环**比较两种奖励：

- **Baseline（原始奖励）**：
  `r_t = -0.25 * (mean_t - mean_{t-1}) / 100`，`R = -0.25 * mean_E2E / 100`
- **方案 D**：
  `u_t = 1 - e2e_t/tau`，`r_t = u_t/H`，末步加 `(1 - T/H) * J`，
  `J = 1 - mean_E2E/tau`，`R = J`

两组的累计回报满足 `4 * R_baseline + 1 = R_D`，奖励图同时给出原始尺度与
`4*R_baseline+1` 对齐尺度。主要判据是实际 E2E 时延、Dev-60 与稳定性。

固定 Stage-2n 条件：单城市、A 拓扑、20 个同质 Transformer-M MEC、
Train-4000 / Dev-60、同一任务顺序、384-D Pure-MEC v2 状态、20-D Mask、
同一 PPO 网络/超参数、`gamma=1.0`、4000 Episodes、Dev 每 200 Episodes。
种子 0/1/2，共 6 个 run。

## 复用审计

`audit_reuse.py` 比较 Stage-2n 原始奖励 PPO 与 Stage-2o D/PPO 的任务池、
部署、MEC 权重、state/action 契约和 PPO 超参数。它们一致，但训练脚本
不同；因此 Stage-2p 用统一代码重跑 6 个 run，旧 seed0 结果只作为复现
对照。审计结果见 `evidence/reuse_audit.json`。

## 文件

| 文件 | 作用 |
|---|---|
| `rewards.py` | baseline/D 奖励、恒等式、控制器、对齐关系 |
| `train.py` | 统一 PPO 训练入口（6 个 run） |
| `progress.py` | 每 200 Episode 中文进展、诊断与 CSV/JSON |
| `launch.py` | 六路并行启动器（自动续训） |
| `plot.py` | 300 dpi 图（原始/对齐奖励、E2E、Dev、滚动σ、PPO 诊断、配对差值） |
| `analyze.py` | 逐 seed/汇总/配对差值、CSV、图表 |
| `test_stage2p.py` | 12 项单元测试（奖励恒等式、对齐、无泄漏、PPO GAE） |
| `audit_reuse.py` | 跨 stage 复用审计 |
| `freeze_protocol.py` | 单元测试与六组 smoke 通过后冻结协议 |
| `STAGE2P_PROTOCOL.md` | 训练前冻结的协议 |
| `STAGE2P_REPORT.md` | 最终中文报告 |
| `evidence/` | CSV、PNG、冻结协议、审计与汇总 |

## 运行

```bash
PY=/home/ubuntu5/miniconda3/envs/DRL/bin/python

$PY v83_spatial_stage2/stage2p/test_stage2p.py
$PY v83_spatial_stage2/stage2p/audit_reuse.py
$PY v83_spatial_stage2/stage2p/launch.py --smoke --max-episodes 30 \
    --validation-every 30 --wait
$PY v83_spatial_stage2/stage2p/freeze_protocol.py
$PY v83_spatial_stage2/stage2p/launch.py --wait --poll-seconds 120 \
    --max-parallel 6 --min-free-mb 10000 --max-restarts 2
$PY v83_spatial_stage2/stage2p/analyze.py
```

原始 checkpoint、TensorBoard Event 和逐 Episode 大日志在
`runs/v83_spatial_stage2/stage2p/`（`.gitignore` 排除）。Test-400 不读取。
