# Stage-2o 证据目录

本目录由 `analyze.py` 生成，所有数据来自 `runs/v83_spatial_stage2/stage2o/`
下六个正式 run（seed 0，4000 Episodes）。Test-400 未读取。

| 文件 | 内容 |
|---|---|
| `protocol_freeze.json` | 训练前冻结的协议、源码/任务/部署/MEC 哈希、单元测试与六个 smoke 结果 |
| `training_curves.csv` | 六个 run 的逐 Episode 训练曲线（含 MA100 奖励/E2E、奖励分解、DQN/PPO 诊断） |
| `validation_curves.csv` | 每 200 Episode 的 Dev-60 验证结果 |
| `per_200_episodes.csv` | 六个 run × 20 个汇报点的中文进展数据（含 GPU 状态与异常标记） |
| `summary.json` | 每个 run 的最佳/最终 Dev、MA100 E2E、AUC、稳定性、诊断与排名 |
| `SUMMARY_TABLES.md` | 上述汇总的中文表格 |
| `fig01_ma100_reward.png` | 六组 MA100 奖励曲线（原始尺度） |
| `fig02_ma100_reward_aligned.png` | 方案 B +1 对齐后的奖励曲线 |
| `fig03_ma100_e2e.png` | 六组 MA100 E2E 时延曲线 |
| `fig04_dev_e2e.png` | 六组 Dev-60 E2E 验证曲线 |
| `fig05_rolling_std.png` | 奖励与 E2E 的滚动标准差 |
| `fig06_dqn_diagnostics.png` | DQN Q mean/Q max/Target/TD Loss |
| `fig07_ppo_diagnostics.png` | PPO Value Loss/Explained Variance/Entropy/KL |
| `fig08_dev_summary.png` | 六组最佳/最终 Dev E2E 柱状图 |
| `fig09_terminal_diagnostics.png` | D 终止修正与方案一终止奖励的尺度/占比 |
| `figure_data/*.csv` | 每张图对应的绘图数据 |
| `REPORT_TABLES.md` | 报告用的 7 张数据表 |
| `ANSWERS.md` | 五个问题的数据化回答草稿 |
| `BASELINE_COMPARE.md` | 与 Dev-60 Greedy 49.18 ms、Stage-2n 原奖励的参照 |
| `NOISE_REFERENCE.md` | Stage-2n 六次 run 的 Dev-60 验证噪声 |
| `provenance_check.json` | 训练相关源码与冻结哈希的一致性校验 |

复现：

```bash
PY=/home/ubuntu5/miniconda3/envs/DRL/bin/python
$PY v83_spatial_stage2/stage2o/analyze.py
```

原始 checkpoint、TensorBoard Event、逐 Episode 大日志在
`runs/v83_spatial_stage2/stage2o/`（`.gitignore` 排除），不进入版本库。
