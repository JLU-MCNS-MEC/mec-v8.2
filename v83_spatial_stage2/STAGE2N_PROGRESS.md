# Stage-2n 训练进度（每 200 Episodes）

启动：2026-10-22 20:45 CST  
协议：`STAGE2N_PROTOCOL.md`  
运行目录：`runs/v83_spatial_stage2/stage2n_latency_only/`

六个 run：DQN/PPO × seeds 0/1/2，每个 4000 Episodes，每 200 Episodes
在固定 Dev-60 上验证。原始验证表由
`python v83_spatial_stage2/stage2n_round_report.py --episode <N> --write`
写入 `runs/.../reports/round_<N>.md`。

## 启动状态

| Run | PID | 状态 |
|---|---:|---|
| DQN s0 | 976316 | 构建 Dev-60 lanes → 训练 |
| DQN s1 | 976319 | 构建 Dev-60 lanes → 训练 |
| DQN s2 | 976384 | 构建 Dev-60 lanes → 训练 |
| PPO s0 | 976387 | 构建 Dev-60 lanes → 训练 |
| PPO s1 | 976452 | 构建 Dev-60 lanes → 训练 |
| PPO s2 | 976455 | 构建 Dev-60 lanes → 训练 |

GPU：RTX 5090 32 GB；启动后占用约 23.5 GB，旧 Stage-2l 暂停进程约 6 GB
未停止。

## 轮次汇总

（每完成一轮 200 Episodes 验证后追加。）

| Episode | DQN E2E mean | PPO E2E mean | DQN SR | PPO SR | 备注 |
|---:|---:|---:|---:|---:|---|
| 200 | 78.13 | 99.48 | 60.00% | 60.00% | DQN s2 在 lane 阶段 SIGSEGV，单独重启后到达 Ep200；六 run SR/SPL/nDTW 完全一致 |
| 400 | 88.60 | 88.09 | 60.00% | 60.00% | DQN 非单调回退、最佳仍在 Ep200；PPO s0/s2 改善；一致性 PASS |
| 600 | 82.74 | 87.03 | 60.00% | 60.00% | DQN 回升；PPO s1 恢复至 90.40 ms；Q 值缓慢上升但有限；一致性 PASS |
| 800 | 88.85 | 83.19 | 60.00% | 60.00% | PPO 反超；DQN 退化、Q 值继续上升但未发散；一致性 PASS |
| 1000 | 88.19 | 82.84 | 60.00% | 60.00% | PPO 最佳 74.75 ms @Ep1200；DQN Q max 2.4，仍有限；一致性 PASS |
| 1200 | 89.49 | 77.95 | 60.00% | 60.00% | PPO 最佳 71.00 ms @Ep1400；DQN Q max 3.2、Dev 退化；一致性 PASS |
| 1400 | 89.77 | 72.44 | 60.00% | 60.00% | PPO 均值 72.44 ms；DQN Q max 4.3、仍退化；一致性 PASS |
| 1600 | 90.62 | 73.88 | 60.00% | 60.00% | PPO 最佳 66.23 ms @Ep1800；DQN Q max 5.0；一致性 PASS |
| 1800 | 87.58 | 70.40 | 60.00% | 60.00% | PPO 最佳 65.89 ms @Ep2000；DQN Q max 6.2；一致性 PASS |
| 2000 | 88.07 | 68.72 | 60.00% | 60.00% | PPO 均值 68.72 ms；DQN Q max 8.5、仍无改善；一致性 PASS |
| 2200 | 88.14 | 70.70 | 60.00% | 60.00% | PPO 最佳 66.18 ms @Ep2400；DQN Q max 11.8；一致性 PASS |
| 2400 | 88.07 | 67.34 | 60.00% | 60.00% | PPO s2 最佳 58.39 ms @Ep2600；DQN Q max 12.4；一致性 PASS |
| 2600 | 87.33 | 63.23 | 60.00% | 60.00% | PPO 均值 63.23 ms；s2 P95 127 ms、违约率 5.57%；一致性 PASS |
| 2800 | 88.87 | 64.71 | 60.00% | 60.00% | PPO s2 entropy 0.96；DQN Q max 14.9；一致性 PASS |
| 3000 | 88.80 | 60.83 | 60.00% | 60.00% | PPO 均值 60.83 ms、最佳 58.39 ms；DQN Q max 15.5；一致性 PASS |
| 3200 | 89.80 | 62.72 | 60.00% | 60.00% | PPO s1 最佳 60.23 ms @Ep3600；DQN Q max 16.7；一致性 PASS |
| 3400 | 87.00 | 63.04 | 60.00% | 60.00% | PPO 进入 60–63 ms 平台；DQN Q max 17.4；一致性 PASS |
| 3600 | 86.96 | 61.30 | 60.00% | 60.00% | PPO 已完成 4000 Episodes，s0/s1/s2 最佳 62.63/58.70/58.39 ms，全部 not_converged；DQN Q max 24.7 |
| 3800 | 89.06 | 63.18 | 60.00% | 60.00% | PPO 全部完成；DQN 剩 110–200 Episodes；DQN Q max 27.1；一致性 PASS |
| 4000 | 87.24 | 61.16 | 60.00% | 60.00% | 六 run 全部完成；全部 not_converged；DQN 最佳在 Ep200，PPO 最佳 62.63/58.70/58.39 ms |

## 最终评估（2026-10-23 00:57 CST）

- **Dev-60**：Greedy-Min-E2E / Nearest / Max-Uplink-Rate 均为 49.18 ms；
  Random 95.43 ms；DQN 最佳 75.76 ms；PPO 最佳 58.39 ms。
- **Test-400（封存，一次性读取）**：Greedy/Nearest/Max-Uplink 均为
  49.11 ms；Random 96.76 ms；DQN 3-seed 均值 77.04 ms（最佳 76.13）；
  PPO 3-seed 均值 60.00 ms（最佳 58.55）。
- 六次训练全部 `not_converged`。DQN 最佳 Dev 检查点均在 Ep200，之后
  Q 正偏移扩大且 Dev E2E 退化；PPO 最佳在 Ep2600–4000。
- 所有策略的动作序列、SR、OSR、SPL、nDTW 在 Test-400 上逐任务完全
  一致（54.00% / 76.25% / 0.5396 / 0.5772），同质一致性 PASS。
- **结论**：在该同质部署 + 纯时延奖励设定下，简单 Greedy-Min-E2E
  明显优于所训练的 DQN/PPO；RL 关联未带来时延收益。

完整表格、图与讨论见 `STAGE2N_RESULTS.md` 与
`stage2n_evidence/STAGE2N_RESULTS.md`。
