# Stage-2k 恢复审计与 DQN 训练结论

日期：2026-10-20  
分支：`experiment/v83-spatial-foundation`  
仓库：`JLU-MCNS-MEC/mec-v8.2`  
恢复方式：新会话现场审计；**未重训 MEC、未重跑 Baseline、未读取 Final Test 300**。

---

## 0. 结论摘要

1. 五种 Stage-2j MEC checkpoint、五种质量先验、五种 Baseline 的 Dev Val 180
   均已完整保存并通过校验，可直接复用。
2. DQN **真实训练到第 2000 / 4000 Episode**（265,956 步，263,956 次梯度更新），
   TensorBoard 中有完整的 2000 条训练记录和 10 次 Validation 记录。
3. 原进程在写完 `last.pt`、`training_history.json`、`progress.json` 后，
   在收尾绘图处崩溃：`plot_training_curves` 读取 `row["episode_return"]`，
   而训练记录字段是 `row["return"]`，触发 `KeyError`。因此缺少
   `training_result.json` 和曲线图，`progress.json` 停在 `training`。
4. 该 KeyError 已在本次恢复中修复并加入回归测试；恢复脚本
   `stage2k_finalize.py` 只补写收尾产物，不训练、不改权重、不覆盖日志。
5. 按预注册早停规则，训练应在第 2000 Episode 停止：
   连续 6 次 Validation 未超过历史最佳 0.002，且最近 3 次 JointR 趋势下降
   （0.1866 → 0.1691 → 0.1439）。最佳 Validation 出现在第 **800** Episode。
6. **最佳 DQN 的 Dev Val JointR = 0.2235，低于 Heuristic（0.2617）、
   Min-Latency（0.2477）和 Max-Quality（0.2256），仅高于 Random（0.2010）。**
   因此本次**没有进入 Association Final Test 180**；正式测试是一次性评估，
   不应在候选明显不具竞争力时消耗。

---

## 1. 现场审计

| 项目 | 审计结果 |
|---|---|
| Git 分支 | `experiment/v83-spatial-foundation` |
| 远程 | `https://github.com/JLU-MCNS-MEC/mec-v8.2.git` |
| Stage-2k 源码 | 审计时 14 个文件均为未跟踪新文件，尚无提交版本可 diff |
| `stage2k_report.py` | 已保存，13,812 B，修改时间 2026-10-20 12:43 |
| 运行进程 | 无 Stage-2k / DQN / Baseline 进程；GPU 空闲 |
| 磁盘 | `/` 已用 93%，剩余约 37 GiB；本次未写入大文件 |
| TensorBoard | `runs/tensorboard/association_stage2k/events.out.tfevents...`，682,246 B |
| Stage-2j Final Test 300 | 未读取 |

### 1.1 TensorBoard 真实性核验

事件文件包含真实的 DQN 训练标量：

| 标量组 | 记录数 | 首末 Step |
|---|---:|---|
| `train/episode_return`、`train/navigation_reward`、`train/latency_penalty`、`train/mean_e2e_ms`、`train/epsilon`、`train/loss` | 各 2,000 | 1 → 2000 |
| `validation/joint_reward`、`validation/SR`、`validation/nDTW`、`validation/E2E_ms`、`validation/p95_ms`、`validation/mean_steps` | 各 10 | 200 → 2000 |
| `validation/model_usage/{Small, Medium, Transformer-S, Transformer-M, Large}` | 各 10 | 200 → 2000 |

这不是计划占位，是实际训练进程写入的记录。

---

## 2. MEC Checkpoint 审计

五个 Stage-2j `best.pt` 全部存在，`strict=True` 可加载；固定输入两次前向
最大绝对差均为 0。`Transformer-M` 的 `best.pt` 是软链接
（`stage2i/T5_map_assisted/best.pt`），不是文件缺失。

| MEC | 参数量 | SHA256 前 16 位 | Epoch | 单次前向 P50 (ms) |
|---|---:|---|---:|---:|
| Small | 611,716 | `7349ec9d0c809a09` | 28 | 0.552 |
| Transformer-S | 1,332,868 | `ceac7506b3b3e18` | 24 | 0.849 |
| Medium | 1,991,044 | `1fcce0f33dfff368` | 28 | 0.697 |
| Transformer-M | 3,033,732 | `57c88b7d71e1a7f1` | 24 | 0.957 |
| Large | 5,248,932 | `310c9d2648dec7fe` | 28 | 1.083 |

审计文件：`runs/v83_spatial_stage2/stage2k/checkpoint_audit.json`。

---

## 3. 五种质量先验

质量先验定义为 Stage-2j Dev Val 180 闭环 mean nDTW；五种 MEC 各 180 Episode。
`quality_prior.json` 中记录的 metrics SHA256 与 checkpoint SHA256 均已逐一复核，
全部一致。

| MEC | mean nDTW | SR | 排名 |
|---|---:|---:|---:|
| Transformer-M | 0.5575 | 57.22% | 1 |
| Transformer-S | 0.5469 | 53.33% | 2 |
| Large | 0.5197 | 48.89% | 3 |
| Medium | 0.2428 | 18.33% | 4 |
| Small | 0.2314 | 16.11% | 5 |

---

## 4. Baseline Dev Val 180

结果为 5 策略 × 180 Episode = 900 条逐 Episode 记录，已完成并通过一致性复核：
从 `per_episode.csv` 重算的 SR / nDTW / E2E 与 `summary.json` 完全一致。

| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | 时延违约率 | 平均步数 | JointR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 55.56% | 0.5582 | 142.87 | 246.01 | 65.26% | 138.8 | 0.2010 |
| Min-Latency | 36.11% | 0.4192 | 68.60 | 94.18 | 14.75% | 158.0 | 0.2477 |
| Max-Quality | 52.78% | 0.5417 | 126.47 | 171.11 | 55.12% | 143.9 | 0.2256 |
| Heuristic / Weighted Greedy | 43.33% | 0.4810 | 87.72 | 134.96 | 23.94% | 154.2 | **0.2617** |
| Oracle（特权上界） | 66.67% | 0.6496 | 90.44 | 160.39 | 31.35% | 113.1 | 0.4236 |

**Baseline 已完整，不需要补跑。** 复用文件：
`runs/v83_spatial_stage2/stage2k/baselines/dev/{summary.json,summary.csv,per_episode.csv,steps.csv}`。

---

## 5. DQN 训练实况

### 5.1 训练配置

| 项 | 值 |
|---|---|
| 状态维度 / 动作数 | 478 / 20 |
| 网络 | `Linear(478,256) → ReLU → Linear(256,256) → ReLU → Linear(256,20)` |
| γ / 学习率 / Batch | 0.99 / 1e-4 Adam / 128 |
| Replay capacity | 50,000 |
| Warm-up / Target update | 2,000 / 1,000 gradient steps |
| ε | 1.0 → 0.05，200,000 步 |
| 训练 Episode | **2,000 / 4,000** |
| 总环境步数 | 265,956 |
| 梯度更新次数 | 263,956 |
| 最终 ε | 0.05 |
| 随机种子 | 20261031 |

### 5.2 10 次 Validation 曲线

| Episode | JointR | SR | nDTW | E2E (ms) | P95 (ms) | 平均步数 |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.1803 | 55.00% | 0.5519 | 148.64 | 243.21 | 151.6 |
| 400 | 0.1710 | 55.00% | 0.5521 | 152.48 | 242.69 | 150.2 |
| 600 | 0.1661 | 50.00% | 0.5491 | 153.20 | 241.66 | 157.6 |
| **800** | **0.2235** | **60.00%** | **0.5951** | **148.64** | **238.70** | **129.8** |
| 1000 | 0.2199 | 56.67% | 0.5778 | 143.19 | 236.16 | 134.6 |
| 1200 | 0.1890 | 51.67% | 0.5545 | 146.19 | 237.65 | 143.2 |
| 1400 | 0.1606 | 51.67% | 0.5279 | 146.94 | 241.21 | 165.3 |
| 1600 | 0.1866 | 51.67% | 0.5500 | 145.35 | 236.31 | 154.5 |
| 1800 | 0.1691 | 48.33% | 0.5223 | 141.29 | 238.40 | 152.2 |
| 2000 | 0.1439 | 48.33% | 0.5073 | 145.34 | 237.18 | 158.1 |

曲线图：
- `runs/v83_spatial_stage2/stage2k/dqn/plots/validation_jointr.png`
- `runs/v83_spatial_stage2/stage2k/dqn/plots/training_return_ma100.png`

### 5.3 早停判定

`should_early_stop(2000, 6, joints)` 返回：

```
(True, "no JointR improvement over 5 validations and last-3 trend is flat/declining")
```

最近 3 次 JointR 为 `0.1866 → 0.1691 → 0.1439`，线性斜率 `-0.02135`。
按设计文档 §5.3，连续 5 次 Validation 未超过历史最佳至少 0.002，且最近 3 次
趋势平坦/下降，应停止。`training_result.json` 因此记录：

- `episodes_trained = 2000`
- `stop_episode = 2000`
- `converged = true`（预注册早停规则的协议标记）
- `best_validation_joint = 0.2235284622859482`
- 最佳 Validation：Episode 800

> 注意：这里的 `converged = true` 只表示“预注册早停规则被触发”，
> 不代表策略已达到稳定、可部署的收敛。最佳点在 800，之后验证指标持续下降，
> 应理解为**早停于停滞/退化点**。

### 5.4 Checkpoint 完整性

| 内容 | `best.pt`（Episode 800） | `last.pt`（Episode 2000） |
|---|---|---|
| 在线网络 `state_dict` | 有 | 有 |
| Target Network | 有 | 有 |
| Adam Optimizer | 有 | 有 |
| 当前 Episode | 800 | 2000 |
| `best_joint` | 0.2235284622859482 | 0.2235284622859482 |
| Replay Buffer | **无** | **无** |
| ε 状态 | **未单独保存**（可由 `total_steps` 推出） | **未单独保存** |
| NumPy / Torch RNG 状态 | **无** | **无** |

因此这**不是严格断点续训**：即使有优化器，Replay Buffer 和 RNG 状态无法恢复。
原训练脚本的 `--resume` 也明确会重新初始化 Replay Buffer。

### 5.5 崩溃与恢复

- `last.pt`、`training_history.json`、`progress.json` 的最后写入时间均为
  14:43，此后无进程；`training_result.json`、`plots/` 缺失。
- 复现确认：旧版 `plot_training_curves` 读取 `row["episode_return"]`，
  而训练记录只有 `row["return"]`，**必然抛出 `KeyError`**。
- 本次修复了该字段名，并新增回归测试
  `test_12_plot_training_curves_uses_history_schema`。
- 恢复脚本 `v83_spatial_stage2/stage2k_finalize.py` 只做三件事：
  1. 用同一 `should_early_stop` 规则复核停止条件；
  2. 校验 `best.pt` / `last.pt` 的 Episode 与 `best_joint` 和 history 一致；
  3. 补写 `training_result.json`、曲线图，并把 `progress.json` 更新为 `completed`。
- **没有重训、没有改写 checkpoint、没有删除/覆盖 TensorBoard 事件文件。**

---

## 6. DQN 与 Baseline 的 Dev Val 对比

| 策略 | SR | nDTW | E2E (ms) | JointR |
|---|---:|---:|---:|---:|
| DQN best @800 | 60.00% | 0.5951 | 148.64 | 0.2235 |
| DQN last @2000 | 48.33% | 0.5073 | 145.34 | 0.1439 |
| Random | 55.56% | 0.5582 | 142.87 | 0.2010 |
| Min-Latency | 36.11% | 0.4192 | 68.60 | 0.2477 |
| Max-Quality | 52.78% | 0.5417 | 126.47 | 0.2256 |
| Heuristic | 43.33% | 0.4810 | 87.72 | **0.2617** |
| Oracle | 66.67% | 0.6496 | 90.44 | 0.4236 |

DQN（best @800）相对 Heuristic：

- ΔSR **+16.67 pp**
- ΔnDTW **+0.1141**
- ΔE2E **+60.92 ms**
- ΔJointR **−0.0381**

也就是说，DQN 确实学到了更高的导航质量（nDTW、SR 均高于全部非 Oracle
Baseline），但没有学会时延权衡：每步平均 E2E 比 Heuristic 高约 61 ms，
`0.25 × E2E/100` 的惩罚抵消了质量收益。Tradeoff-stress 场景很可能是主要
失分点；在补充诊断前，不建议直接进入正式测试。

---

## 7. 当前状态与下一步

- `progress.json`：`dqn.status = completed`，`final_test.status = pending`。
- Association Final Test 180 的 specs 和 topologies 已生成，但
  `final_test/` 目录和 `eval_freeze.json` 尚未创建，测试未被读取。
- **本次没有运行独立测试。** 依据：DQN 最佳 Dev Val JointR 低于
  Heuristic、Min-Latency、Max-Quality 三种 Baseline，尚不满足“稳定收敛且
  具竞争力”的进入条件；正式测试是一次性评估，应避免无信息量的消耗。

可选下一步（需要明确选择）：

1. **按协议接受早停结果**，用 `best.pt`（Episode 800）执行一次
   Association Final Test 180，报告负结果；不再改训练。
2. **先做诊断再决定**：检查 DQN 在 Tradeoff-stress 下的节点选择分布、
   奖励尺度与时延惩罚，判断是奖励设计还是训练不足导致高时延。
3. **Warm Start 续训**（Case B）：从现有权重另建恢复运行目录，Replay Buffer
   重新初始化，累计训练不超过 4000 Episode；这不是严格续训，且会改变
   已触发的早停结论，需要先修订实验协议。

在没有明确选择前，保持现有权重、日志、Baseline 与 `progress.json` 不变。
