# Stage-2k Baseline 结果 — Dev Validation 180

日期：2026-10-20  
分支：`experiment/v83-spatial-foundation`  
数据：固定 Dev Validation 180（三个部署条件各 60；每个条件 18 Short / 30 Medium / 12 Long）

---

## 0. 结论

五种 Baseline 已全部完成，共 5 策略 × 180 Episode = 900 条逐 Episode 记录。
本轮恢复**没有重跑任何 Baseline**；以下结果直接复用
`runs/v83_spatial_stage2/stage2k/baselines/dev/` 下的原始产物，并从
`per_episode.csv` 重新计算 SR / nDTW / E2E 与 `summary.json` 核对，完全一致。

Dev Val 180 的最强非特权 Baseline 是 **Heuristic / Weighted Greedy
（JointR 0.2617）**；Min-Latency 的 JointR 为 0.2477，Max-Quality 为 0.2256，
Random 为 0.2010，Oracle 特权上界为 0.4236。

---

## 1. 总体结果

| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | 时延违约率 | 平均步数 | JointR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 55.56% | 0.5582 | 142.87 | 246.01 | 65.26% | 138.8 | 0.2010 |
| Min-Latency | 36.11% | 0.4192 | **68.60** | **94.18** | **14.75%** | 158.0 | 0.2477 |
| Max-Quality | 52.78% | 0.5417 | 126.47 | 171.11 | 55.12% | 143.9 | 0.2256 |
| Heuristic / Weighted Greedy | 43.33% | 0.4810 | 87.72 | 134.96 | 23.94% | 154.2 | **0.2617** |
| Oracle（特权上界，不参与训练） | **66.67%** | **0.6496** | 90.44 | 160.39 | 31.35% | 113.1 | 0.4236 |

`JointR = mean(nDTW) − 0.25 × mean(E2E_ms) / 100`。

---

## 2. 分部署条件

### Balanced / Original

| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | JointR |
|---|---:|---:|---:|---:|---:|
| Random | 51.67% | 0.4852 | 134.35 | 248.94 | 0.1493 |
| Min-Latency | 40.00% | 0.3968 | 66.96 | 91.42 | 0.2294 |
| Max-Quality | 48.33% | 0.5051 | 110.97 | 149.67 | 0.2277 |
| Heuristic | 45.00% | 0.4439 | 72.54 | 106.95 | 0.2626 |
| Oracle | 68.33% | 0.6109 | 89.92 | 160.61 | 0.3861 |

### Quality-rich

| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | JointR |
|---|---:|---:|---:|---:|---:|
| Random | 51.67% | 0.5639 | 111.09 | 190.20 | 0.2862 |
| Min-Latency | 41.67% | 0.5194 | 55.84 | 74.37 | 0.3798 |
| Max-Quality | 55.00% | 0.5964 | 78.66 | 113.56 | 0.3997 |
| Heuristic | 46.67% | 0.5720 | 59.31 | 84.61 | 0.4238 |
| Oracle | 66.67% | 0.6893 | 70.22 | 123.45 | 0.5137 |

### Tradeoff-stress

| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | JointR |
|---|---:|---:|---:|---:|---:|
| Random | 63.33% | 0.6255 | 183.18 | 298.89 | 0.1675 |
| Min-Latency | 26.67% | 0.3413 | 83.01 | 116.75 | 0.1338 |
| Max-Quality | 55.00% | 0.5237 | 189.78 | 250.09 | 0.0493 |
| Heuristic | 38.33% | 0.4269 | 131.30 | 213.32 | 0.0986 |
| Oracle | 65.00% | 0.6488 | 111.17 | 197.11 | 0.3709 |

Tradeoff-stress 是质量与时延冲突最强的场景：Max-Quality 的 nDTW 尚可，但
E2E 接近 190 ms，JointR 只有 0.0493；Min-Latency 反过来只有 0.3413 nDTW。
任何关联策略都必须在该场景真正权衡，而不能只追质量或只追时延。

---

## 3. 定义与数据隔离

- **Random**：在可用节点中均匀随机；每个 Episode 固定种子
  `episode_seed + 1,000,003`，可复现。
- **Min-Latency**：选择预测 E2E 最低的可用节点。
- **Max-Quality**：选择 Dev Val 质量先验最高的节点；同质量选时延更低者。
- **Heuristic / Weighted Greedy**：
  `0.5 × quality + 0.5 × (1 − clip(E2E_ms/100, 0, 1))`；同分选时延更低者。
- **Oracle**：使用 `env.oracle_action()` 的特权信息；只作为上界，
  不参与 DQN 训练、状态构造或奖励。

所有策略共享同一 Episode 清单、同一三种部署拓扑、同一 E2E 时延模型与同一
JointR。质量先验来自 Stage-2j Dev Val 180 closed-loop mean nDTW；
**未使用 Stage-2j Final Test 300**。

---

## 4. 产物位置

- 总表：`runs/v83_spatial_stage2/stage2k/baselines/dev/summary.json`
- 总表 CSV：`runs/v83_spatial_stage2/stage2k/baselines/dev/summary.csv`
- 逐 Episode：`runs/v83_spatial_stage2/stage2k/baselines/dev/per_episode.csv`
- 逐 Step：`runs/v83_spatial_stage2/stage2k/baselines/dev/steps.csv`
- 质量先验：`runs/v83_spatial_stage2/stage2k/quality_prior.json`
- 原始 Baseline 报告：`runs/v83_spatial_stage2/stage2k/baselines/STAGE2K_BASELINE_RESULTS.md`

本轮未重新训练 MEC，未重新生成质量先验，未重跑 Baseline。

---

## 5. DQN 对比

当前 DQN 与上述五种 Baseline 的完整 Dev Val 180 对比、状态/奖励审计和
验证集问题见 `v83_spatial_stage2/STAGE2K_ASSOCIATION_RL_REPORT.md`。
本文件只记录 Baseline 本身，不重复 DQN 指标。
