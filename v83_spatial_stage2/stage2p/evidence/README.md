# Stage-2p 证据目录

本目录由 `analyze.py`、`render_report.py`、`answer_questions.py` 和
`reproduction_check.py` 生成。六个 run 位于
`runs/v83_spatial_stage2/stage2p/{baseline,D}/ppo_s{0,1,2}/`
（`.gitignore` 排除）。

| 文件 | 内容 |
|---|---|
| `reuse_audit.json` | Stage-2n/2o 复用审计：任务池/部署/MEC/PPO 配置一致，但 trainer 路径不同，因此统一重跑 |
| `protocol_freeze.json` | 训练前冻结的协议、源码/任务/部署/MEC 哈希、单元测试与六组 smoke |
| `reproduction_check.json` | 新 baseline/D seed0 与旧 Stage-2n/2o seed0 的逐 Episode 复现校验 |
| `training_curves.csv` | 六个 run 的逐 Episode 训练曲线（含 MA100、对齐奖励、PPO 诊断） |
| `validation_curves.csv` | 每 200 Episode 的 Dev-60 验证结果 |
| `per_200_episodes.csv` | 六个 run × 20 个汇报点（含 GPU、违约率、PPO 诊断与异常） |
| `per_seed_metrics.csv` | 逐 seed 的最佳/最终 Dev、MA100 AUC、末500 σ、退化、Greedy 差距、收敛 |
| `seed_aggregate.csv` | 两种奖励的三种子均值/标准差 |
| `paired_differences.csv` | 同一 seed 的 D − Baseline 配对差值 |
| `summary.json` | 上述指标的严格 JSON 汇总 |
| `SUMMARY_TABLES.md` | 中文逐 seed/汇总/配对表 |
| `ANSWERS.md` | 五个问题的数据化回答草稿 |
| `fig01_ma100_reward.png` | MA100 奖励（原始尺度 + 4*R_baseline+1 对齐） |
| `fig02_ma100_e2e.png` | MA100 E2E 时延（三种子） |
| `fig03_dev_e2e.png` | Dev-60 E2E 验证曲线（含 Greedy 49.18 ms） |
| `fig04_rolling_std.png` | 奖励与 E2E 滚动标准差 |
| `fig05_ppo_diagnostics.png` | PPO Value Loss/EV/Entropy/KL |
| `fig06_seed_summary.png` | 逐 seed 最佳/最终 Dev 柱状图 |
| `fig07_paired_differences.png` | 逐 seed 配对差值 |
| `figure_data/*.csv` | 每张图的绘图数据 |

复现：

```bash
PY=/home/ubuntu5/miniconda3/envs/DRL/bin/python
$PY v83_spatial_stage2/stage2p/analyze.py
$PY v83_spatial_stage2/stage2p/render_report.py
$PY v83_spatial_stage2/stage2p/answer_questions.py
$PY v83_spatial_stage2/stage2p/reproduction_check.py
$PY v83_spatial_stage2/stage2p/verify_provenance.py
```

Test-400 未读取。
