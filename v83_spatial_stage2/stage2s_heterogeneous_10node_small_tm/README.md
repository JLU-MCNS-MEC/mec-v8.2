# Stage-2s：5×Small + 5×Transformer-M 异构 10 节点 MEC

本目录是 Stage-2s 的独立实现、冻结协议、测试、汇总结果、图和中文报告。
实验在 Stage-2r 的 A10_fixed 十个位置部署 5 个 Small 与 5 个
Transformer-M MEC，推理时延使用显式 0.2 TFLOP/s 有效吞吐，比较 DQN/PPO
× 三个新种子（101/202/303），共 6 个 4000-Episode 正式 run。奖励统一为方案 D。

**本阶段不使用 Stage-2r 的冻结 lane 缓存**：每个 step 由所选节点的模型真实
产生导航动作，模型选择会改变 UAV 轨迹、状态和后续关联决策。

## 关键常量

- 部署哈希：`1fce485ce6256a288fb0aecd525afbf2830467b5ee9f2918952af54148a081cd`
- 节点：M01–M10；Small = M03/M06/M07/M09/M10，Transformer-M = M01/M02/M04/M05/M08
- 名义算力 2.0 TFLOP/s；推理有效吞吐 0.2 TFLOP/s（两者严格分离）
- 状态 194-D（4 + 10×19），动作 10-D，独立 10-D Mask
- 方案 D；`tau=100 ms`；`H=400`；`gamma=1.0`
- Test-400 全程封存不读

## 文件

| 文件 | 内容 |
|---|---|
| `common.py` | A10 几何、确定性 5+5 空间均衡分配、部署哈希、任务/检查点/维度契约 |
| `runtime.py` | 双模型后端、0.2 TFLOP/s 异构时延、真实闭环 runner、Dev 评估器 |
| `train.py` | DQN/PPO 正式训练入口（6 run，4000 Episodes，真实闭环，断点续训） |
| `progress.py` | 每 200 Episode 中文进展 + 累计成功/失败/band 统计 + 异构选择/时延分解 |
| `launch.py` | 6 路启动器（默认 3 路并行、GPU 显存保护、崩溃续训） |
| `status.py` | 只读查看 6 个 run 的实时状态 |
| `smoke.py` | 训练前闭环不变量检查 + 10–20 Episode × DQN/PPO smoke |
| `eval.py` | Dev-60 最终评估（4 baseline + 6 学习检查点，真实闭环） |
| `analyze.py` | 汇总 CSV/JSON、训练曲线与验证统计 |
| `plot.py` | 训练/验证/种子变异性/异构性图（300 dpi，CSV 侧车） |
| `trajectories.py` | DQN/PPO 真实闭环轨迹采集与逐步数据 |
| `plot_trajectories.py` | 轨迹图、模型选择着色、强制模型诊断图 |
| `render_report.py` | 从汇总 CSV/JSON 生成最终中文报告（数字不手写） |
| `test_stage2s.py` | CPU 单元测试（部署、算力、时延、状态、奖励、隔离） |
| `test_stage2s_gpu.py` | GPU 闭环测试（双模型加载、调度、动作影响、奖励恒等式） |
| `STAGE2S_PROTOCOL.md` | 训练前冻结的协议 |
| `STAGE2S_REPORT.md` | 最终中文报告 |
| `deployment_heterogeneous.json` | 10 节点位置、模型分配、部署哈希 |
| `smoke_report.json` / `SMOKE_TEST.md` | Smoke Test 结果 |
| `figures/` | 所有 PNG |
| `figure_data/` | 每张图的 CSV 数据 |
| `*.csv` / `summary.json` | 训练、验证、baseline、种子聚合、轨迹汇总 |

## 运行

```bash
PY=/home/ubuntu5/miniconda3/envs/DRL/bin/python

# CPU + GPU 测试
$PY -m unittest -v \
    v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.test_stage2s
$PY -m unittest -v \
    v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.test_stage2s_gpu

# 训练前 smoke（14 Episodes × DQN/PPO + 闭环不变量检查）
$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/smoke.py \
    --episodes 14 --dev-episodes 10

# 6 个正式 run（默认 3 路并行，自动重试/续训）
$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/launch.py \
    --wait --max-parallel 3 --min-free-mb 5000 --poll-seconds 120

# 查看进度
$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/status.py

# 汇总 + 图（训练结束后）
$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/analyze.py
$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/plot.py

# Dev-60 最终评估（训练结束后）
$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/eval.py

# 轨迹与轨迹图（评估后）
$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/trajectories.py
$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/plot_trajectories.py

# 最终中文报告（在 eval / analyze / trajectories 之后）
$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/render_report.py

# 或：等待 6 个 run 完成后自动执行 eval → analyze → plot → trajectories → 报告
bash v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/run_postprocess.sh
```

原始 checkpoint、TensorBoard Event 和逐 Episode 大日志在
`runs/v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/`（`.gitignore` 排除）。
Test-400 不读取。

## 关键事实

- 同一观测下 Small 与 Transformer-M 的导航动作可以不同；模型选择改变轨迹和结果。
- Small 推理约 0.071 ms，Transformer-M 约 14.07 ms（0.2 TFLOP/s 有效吞吐）。
- E2E 主要由通信时延与 Transformer-M 推理时延共同决定；Small 节点可能因通信
  距离近而被偏好，导航质量是否因此下降必须按真实结果报告。
- 所有 baseline 使用同一 10 节点部署和同一真实模型推理时延；Greedy-Min-E2E
  只使用当前状态可获得的 `total_ms`。
