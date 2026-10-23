# Stage-2o：时延奖励设计对 DQN/PPO 收敛的影响

本目录是 Stage-2o 的独立实现、协议、测试、分析与报告。固定 Stage-2n 的
城市、A 拓扑、20 个同质 Transformer-M MEC、Train 4000 / Dev 60 任务池、
任务顺序、384-D Pure-MEC v2 状态、动作 Mask、DQN/PPO 超参数与 `gamma=1.0`，
只改变奖励函数。

## 三个方案

| 方案 | 逐步奖励 | Episode 累计回报 |
|---|---|---|
| B | `r_t = -(mean_t - mean_{t-1}) / tau` | `-mean_E2E / tau` |
| D | `u_t = 1 - e2e_t/tau`；`r_t = u_t/H`；末步加 `(1-T/H)*J` | `1 - mean_E2E / tau` |
| one | 非终止步 0；末步 `J = 1 - mean_E2E/tau` | `1 - mean_E2E / tau` |

`tau = 100 ms`，`H = 400`。`R_D = R_one = 1 + R_B`，所以奖励曲线绝对高度
不可直接比较；结论以实际 E2E 时延、MA100 趋势、滚动标准差、Dev-60 与收敛
判据为准。

## 文件

| 文件 | 作用 |
|---|---|
| `rewards.py` | 三个方案的纯函数实现、恒等式检查、在线控制器 |
| `train.py` | DQN/PPO 训练入口（6 个实验，seed 0，4000 Episodes） |
| `progress.py` | 每 200 Episode 的中文进展、诊断与 CSV/JSON |
| `launch.py` | 六路并行启动器（不停止已有进程，支持崩溃自动续训） |
| `status.py` | 只读实时状态表 |
| `plot.py` | 300 dpi 图（中文标签、蓝/橙/青三色 + 线型二次编码） |
| `analyze.py` | 汇总 CSV、MA100/AUC/稳定性指标、图表与数据表 |
| `freeze_protocol.py` | 单元测试与 6 个 smoke 通过后冻结协议 |
| `test_stage2o.py` | 16 项单元测试（奖励恒等式、无泄漏、DQN 目标、PPO GAE） |
| `STAGE2O_PROTOCOL.md` | 训练前冻结的实验协议 |
| `STAGE2O_REPORT.md` | 最终中文报告 |
| `evidence/` | 汇总 CSV、PNG、`protocol_freeze.json`、`summary.json` |

## 运行

```bash
PY=/home/ubuntu5/miniconda3/envs/DRL/bin/python

# 单元测试
$PY v83_spatial_stage2/stage2o/test_stage2o.py

# 30-Episode smoke（六路）
$PY v83_spatial_stage2/stage2o/launch.py --smoke --max-episodes 30 \
    --validation-every 30 --wait

# 冻结协议（要求单元测试与六个 smoke 都通过）
$PY v83_spatial_stage2/stage2o/freeze_protocol.py

# 正式训练（六路，4000 Episodes，Dev-60 every 200）
$PY v83_spatial_stage2/stage2o/launch.py --wait

# 实时状态
$PY v83_spatial_stage2/stage2o/status.py

# 汇总与绘图
$PY v83_spatial_stage2/stage2o/analyze.py
```

原始 checkpoint、TensorBoard Event 和逐 Episode 大日志写入
`runs/v83_spatial_stage2/stage2o/`（被 `.gitignore` 排除），不进入仓库。
Test-400 在本阶段不读取。
