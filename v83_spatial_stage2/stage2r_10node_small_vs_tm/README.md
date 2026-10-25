# Stage-2r：10 个 MEC 的 Small vs Transformer-M 同质部署

本目录是 Stage-2r 的独立实现、冻结协议、测试、汇总结果、图和中文报告。
实验把 MEC 数量从 20 改为 10，比较两种同质部署（全 Small / 全
Transformer-M）× 两种关联算法（DQN/PPO）× 三个新种子（11/22/33），
共 12 个 4000-Episode run。奖励统一为方案 D。

## 文件

| 文件 | 内容 |
|---|---|
| `common.py` | 10 节点 A10 几何、两个同质部署、任务/检查点/维度契约 |
| `runtime.py` | 模型专用 homogeneous runtime、方案 D 返回值 |
| `train.py` | DQN/PPO 训练入口（12 个 run，正式 run 使用冻结 lane 缓存） |
| `lane_cache.py` | 用完整环境一次性生成 Train-4000 冻结 lane（每模型一份） |
| `lane_runner.py` | 与全环境 runner 接口一致的精确 lane adapter |
| `test_lane_equivalence.py` | 全环境 vs lane 的逐 step 等价性测试（GPU） |
| `progress.py` | 每 200 Episode 中文进展 + 累计成功/失败/band 统计 |
| `launch.py` | 12 路启动器（6 路并行两批、崩溃续训） |
| `eval.py` | Dev-60 最终评估（4 baseline + 12 学习检查点） |
| `analyze.py` | 汇总 CSV/JSON、训练曲线与验证图 |
| `plot.py` | 训练/验证/成功失败图（300 dpi，CSV 侧车） |
| `trajectories.py` | 4 组条件真实 UAV 轨迹采集与一致性审计 |
| `plot_trajectories.py` | 2×3 条件轨迹图、4×6 总览、审计图 |
| `test_stage2r.py` | CPU 单元测试（10→10 动作、194 状态、方案 D、无 Test） |
| `STAGE2R_PROTOCOL.md` | 训练前冻结的协议 |
| `STAGE2R_REPORT.md` | 最终中文报告 |
| `figures/` | 所有 PNG |
| `figure_data/` | 每张图的 CSV 数据 |
| `training_curves.csv` | 每 run 每 Episode 曲线 + 累计统计 |
| `per_200_episodes.csv` | 每 200 Episode 进展表 |
| `success_failure_summary.csv` | 每 run 最终成功/失败与 band 分布 |
| `validation_summary.csv` | Dev-60 全部策略（学习 + baseline） |
| `per_seed_metrics.csv` | 12 个学习 run 的 Dev-60 指标 |
| `seed_aggregate.csv` | 4 个条件 × 3 seed 均值/标准差 |
| `trajectory_steps.csv` | 逐步真实坐标、动作、节点、E2E |
| `trajectory_summary.csv` | 每条轨迹摘要、哈希、起终点/目标 |
| `trajectory_consistency_audit.csv/json` | DQN/PPO/control 一致性审计 |
| `deployment_small.json` / `deployment_transformer_m.json` | 两个 10 节点同质部署（位置、检查点、哈希） |
| `lane_cache_small.json` / `lane_cache_transformer_m.json` | 冻结 lane 缓存 provenance（任务/部署/源码哈希） |

## 运行

```bash
PY=/home/ubuntu5/miniconda3/envs/DRL/bin/python

$PY -m unittest -v \
    v83_spatial_stage2.stage2r_10node_small_vs_tm.test_stage2r
$PY -m unittest -v \
    v83_spatial_stage2.stage2r_10node_small_vs_tm.test_lane_equivalence

# 一次性构建 Train-4000 冻结 lane（每个模型；正式训练前）
$PY v83_spatial_stage2/stage2r_10node_small_vs_tm/lane_cache.py \
    --model Small --workers 6
$PY v83_spatial_stage2/stage2r_10node_small_vs_tm/lane_cache.py \
    --model Transformer-M --workers 6

# 4 条件 smoke（20 Episode，全环境）
$PY v83_spatial_stage2/stage2r_10node_small_vs_tm/launch.py \
    --smoke --max-episodes 20 --validation-every 20 --wait

# 正式 12 run（6 路并行，约两批）
$PY v83_spatial_stage2/stage2r_10node_small_vs_tm/launch.py \
    --wait --poll-seconds 120 --max-parallel 6 --min-free-mb 12000

# 汇总（训练结束后）
$PY v83_spatial_stage2/stage2r_10node_small_vs_tm/analyze.py

# Dev-60 最终评估（训练结束后）
$PY v83_spatial_stage2/stage2r_10node_small_vs_tm/eval.py

# 最终图（评估 + 轨迹后）
$PY v83_spatial_stage2/stage2r_10node_small_vs_tm/analyze.py

# 轨迹采集与绘图（评估后）
$PY v83_spatial_stage2/stage2r_10node_small_vs_tm/trajectories.py
$PY v83_spatial_stage2/stage2r_10node_small_vs_tm/plot_trajectories.py

# 或：等待 12 个 run 完成后自动执行 eval → analyze → trajectories →
# plot_trajectories → report
bash v83_spatial_stage2/stage2r_10node_small_vs_tm/run_postprocess.sh
```

原始 checkpoint、TensorBoard Event 和逐 Episode 大日志在
`runs/v83_spatial_stage2/stage2r_10node_small_vs_tm/`（`.gitignore` 排除）。
早期全环境批次（仅用于速度评估，未用于最终结果）归档在
`runs/v83_spatial_stage2/stage2r_10node_small_vs_tm/_full_env_partial_20261025/`。
Test-400 不读取。

## 关键事实

- 导航动作由冻结 MEC 模型产生，DQN/PPO 只选择服务节点；同一部署下
  不同关联策略的 UAV 轨迹、SR、OSR、nDTW 相同，只有 E2E/Reward 不同。
- 10×Small 的导航 SR 明显低于 10×Transformer-M；这是模型能力差异，
  不是关联策略差异。
- A10 几何是 A_fixed 的确定性 maximin 10 节点子集，对全部 Train/Dev
  起点有 500 m 覆盖。
