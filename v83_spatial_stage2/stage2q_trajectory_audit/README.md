# Stage-2q：UAV 轨迹可视化与跨方法一致性审计

本目录保存 Stage-2q 的代码、真实轨迹 CSV、审计数据、PNG 图表和中文报告。
任务范围：固定城市、同质 20 节点 MEC、Stage-2n Dev-60 任务池；**不读取
Test-400、不重新训练 DQN/PPO、不覆盖 Stage-2n/2o/2p 历史产物**。

## 文件

| 文件 | 内容 |
|---|---|
| `collect_trajectories.py` | 重建 Dev-60 frozen lane、回放 18 个 learned 检查点 + 2 个对照、运行 6 任务真实闭环，写入全部 CSV/JSON |
| `plot_trajectories.py` | 只读本目录 CSV，生成 5 张 PNG（无需 GPU/检查点） |
| `test_trajectory_audit.py` | 坐标、任务分层、动作/坐标哈希、闭环一致性测试（无需 GPU） |
| `STAGE2Q_REPORT.md` | 中文分析报告与最终答案 |
| `trajectory_steps.csv` | 6 条代表任务 × 20 方法的逐步真实坐标、动作、E2E、关联节点 |
| `trajectory_summary.csv` | 每条轨迹的成功状态、band、步数、起终点、检查点信息、动作/坐标哈希 |
| `trajectory_invariance.csv` | 三个证据层级的跨方法动作/坐标哈希、步数/成功差异、最大坐标偏差、节点/E2E 差异 |
| `dev60_task_summary.csv` | 全部 60 条 Dev 任务的 lane 级上下文 |
| `selected_tasks.csv` | 6 条代表任务的确定性选择记录（band × 成功/失败，中位步数） |
| `checkpoint_inventory.csv` | 18 个 best.pt 的路径、SHA256、Dev 选择指标、模型权重哈希与重复分组 |
| `city_geometry.csv` / `city_meta.json` | 固定城市真实地标几何、道路网格参数、停止半径 |
| `audit_summary.json` | 汇总审计数字（一致性、节点多样性、历史复现、重复权重） |
| `run_manifest.json` | 运行 provenance、任务池/部署/MEC 哈希、输出文件 SHA256 |
| `trajectory_examples_2x3.png` | 2×3 Short/Medium/Long × 成功/失败真实轨迹 |
| `stage2n_method_overlay.png` | Stage-2n 6 个 DQN/PPO seed 的同任务轨迹与 E2E |
| `stage2o_reward_overlay.png` | Stage-2o B/D/方案一 × DQN/PPO 的同任务轨迹与 E2E |
| `stage2p_ppo_reward_overlay.png` | Stage-2p baseline/D × 3 seeds 的同任务轨迹与 E2E |
| `trajectory_difference_audit.png` | 三层级审计表、零坐标偏差、节点选择/E2E 差异 |

## 复现

```bash
# 1) 采集（单卡，约 10–15 分钟；只读 Dev-60）
cd /home/ubuntu5/Workspace/MEC-V8.2
OMP_NUM_THREADS=2 MPLBACKEND=Agg \
  /home/ubuntu5/miniconda3/envs/DRL/bin/python -u \
  v83_spatial_stage2/stage2q_trajectory_audit/collect_trajectories.py \
  --device cuda

# 2) 绘图（CPU）
/home/ubuntu5/miniconda3/envs/DRL/bin/python \
  v83_spatial_stage2/stage2q_trajectory_audit/plot_trajectories.py

# 3) 测试（CPU）
/home/ubuntu5/miniconda3/envs/DRL/bin/python -m unittest \
  v83_spatial_stage2.stage2q_trajectory_audit.test_trajectory_audit -v
```

原始检查点、lane 构建日志、训练日志仍在 `.gitignore` 排除的 `runs/` 下；
本目录只保存可追踪的 CSV、PNG、JSON 和报告。

## 关键约定

- 主结果一律使用各实验按 **Dev 平均 E2E 最低**选出的 `best.pt`；`last.pt`
  存在但未混入主比较，避免把不同选择标准混在一起。
- 坐标是固定城市世界坐标（米）；`z` 是环境固定高度 80 m；heading 是
  `env.yaw`（弧度）。
- 动作哈希与 Stage-2n 历史格式一致：`sha256(",".join(str(action)))`。
- 坐标哈希是 float32 `[x, y, heading]` 数组的 SHA256；路径哈希是
  float32 后动作轨迹 `[x, y]` 的 SHA256。
- 同质 MEC 下关联策略只改变节点选择和 E2E 时延，不改变导航动作；报告
  用“60 任务 lane 回放 + 6 任务真实闭环 + 闭环 vs lane”三层证据验证，
  没有把预期行为直接当作已验证结果。
