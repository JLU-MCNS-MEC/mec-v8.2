# Stage-2m 四项结论（2026-10-22）

数据来源：
- 训练：`runs/v83_spatial_stage2/stage2m_visual_fixed_city/`
- 汇总：`runs/v83_spatial_stage2/stage2m_visual_fixed_city/analysis/STAGE2M_RESULTS.md`、
  `analysis/run_summary.csv`、`analysis/paired_visual.csv`、
  `analysis/dqn_vs_ddqn.csv`
- Q 审计：`runs/v83_spatial_stage2/stage2m_q_audit/`
  （`pre/`、`post_dqn/`、`post_visual/`、`post_visual_last/`、
  `post_visual_double/`）

协议：固定城市 `map_seed=10,000,001`、A 拓扑 `A_fixed`、Train 前 2000 条
全局乱序任务、固定 Dev60、JointR 奖励、五种冻结 T5 MEC、2000 Episodes、
每 200 验证、3 seeds（0/1/2）。Final Test 未读取。

---

## 1. Q 值是否高估？

**是，严重高估，且不是 Mask/终止/奖励实现错误造成的。**

| Checkpoint | Ep | Q(s,a) mean | MC 回报 G_t | 偏差 mean | MAE | 高估比例 |
|---|---:|---:|---:|---:|---:|---:|
| 修复前 A/DQN best | 2400 | 5.586 | −0.015 | **+5.601** | 5.601 | 99.7% |
| 修复前 A/DQN last | 3500 | 8.288 | −0.054 | **+8.342** | 8.345 | 98.7% |
| 修复后 DQN best（3 seed 均值） | — | 3.208 | +0.0001 | **+3.208** | 3.211 | 98.6% |
| 修复后 DQN last（3 seed 均值） | — | 4.739 | −0.030 | **+4.768** | 4.772 | 98.7% |
| Double DQN best（3 seed 均值） | — | 1.601 | −0.019 | **+1.621** | 1.622 | 99.1% |
| Double DQN last（3 seed 均值） | — | 1.668 | −0.034 | **+1.702** | 1.704 | 99.1% |
| Visual DQN best（3 seed 均值） | — | 49.442 | +0.013 | **+49.429** | 49.429 | 100% |
| Visual DQN last（3 seed 均值） | — | 392.023 | −0.006 | **+392.029** | 392.029 | 100% |
| Visual+DDQN best / last（s0） | 400 / 2000 | 6.02 / 333.27 | +0.016 / −0.037 | +6.00 / +333.31 | 6.00 / 333.31 | 100% |

- 修复前首步 Q 为 6.82 / 12.37，而对应 Episode 实际回报只有
  0.086 / 0.0068，量级差约 80–1800 倍。
- 实现审计（每个 checkpoint 60 条 Dev）：终止转换 60/60 无
  `next_state`、target 不 bootstrap；每步可用 MEC 1–10 个、无全 False
  Mask；0 次无效动作；Episode 累计奖励与 JointR 最大误差 ≤1.0e-15。
- 旧 target 的“幻影 0”问题确认存在（每个审计集 13–307 条转换），
  已改为 `-inf` 填充；但 legacy−fixed target MAE 仅 0.00015–0.00449，
  同预算 legacy 对照（seed0）Q 4.11 / loss 0.021 与修复后 DQN s0
  4.19 / 0.022 几乎相同 ⇒ **该 bug 正确但不构成高估主因**。
- 主因指向 γ=1 长程（最多 400 步）下的 max 自举偏差；视觉 arm 的
  Q 比 pure 再大约 2–100 倍，指向 128-D 未归一化特征尺度。

## 2. Double DQN 是否改善？

**对 Q 校准和平均训练回报有改善；对 Dev best 和收敛没有改善。**

| 指标（3 seed 均值） | DQN | Double DQN | 配对差 / 95% CI |
|---|---:|---:|---:|
| Q mean（末 100 Episodes） | 4.548 | **1.770** | −2.78 |
| TD loss（末 100） | 0.0252 | **0.0059** | — |
| TD |Q−target|（末 100） | 0.094 | **0.048** | — |
| Q 审计偏差（best / last） | 3.208 / 4.768 | **1.621 / 1.702** | −49.5% / −64.3% |
| Best Dev60 JointR | **0.0932** | 0.0654 | −0.0278 [−0.054, +0.017] |
| Final Dev60 JointR | 0.0275 | **0.0581** | +0.0307 [−0.006, +0.081] |
| Final MA100 JointR | 0.1426 | **0.1841** | **+0.0415 [+0.032, +0.059]，3/3 为正** |
| 收敛 | s0 plateaued，s1 improving，s2 plateaued | s0 plateaued，s1/s2 improving | 均未收敛 |

结论：Double DQN 把 Q 高估压低约 50%（best）/64%（last），训练 MA100
在 3/3 个 seed 上提高，Final Dev 均值也更高；但 Dev best 均值下降、
95% CI 跨 0，且没有 arm 收敛。**“降低高估”成立，“提升最优策略/收敛”不成立。**

## 3. DQN / PPO 是否收敛？

**否。2000 Episodes 内没有任何一组达到预注册收敛判据。**

| Arm | Best Dev60 JointR（3 seed 均值） | 最佳 Episode 均值 | Last-5 JointR range 均值 | 状态 |
|---|---:|---:|---:|---|
| DQN Pure | **0.0932** | 1533 | 0.110 | plateaued / improving / plateaued |
| DQN +Visual | 0.0856 | 600 | 0.098 | 全部 plateaued_not_converged |
| PPO Pure | 0.0647 | 1333 | 0.103 | 全部 plateaued_not_converged |
| PPO +Visual | **0.1205** | 733 | 0.079 | 全部 plateaued_not_converged |
| Double DQN | 0.0654 | 1933 | 0.085 | plateaued / improving / improving |
| Legacy DQN（s0） | 0.0671 | 1000 | 0.046 | plateaued_not_converged |

- 预注册判据要求最近 3 次验证差 ≤0.01 且最后 ≥best−0.005；实际
  last-5 range 为 0.046–0.110，远高于 0.01。
- DQN Pure s1、Double DQN s1/s2 在 2000 Episode 结束时仍是
  `improving`；其余全部 `plateaued_not_converged`。
- 与 Stage-2l 3500 Episodes 也全部 plateau 的结论一致：固定城市 A
  拓扑在当前状态/奖励/预算下没有收敛。

## 4. 视觉特征是否带来稳定奖励提升？

**算法相关，整体不能称为“稳定的奖励提升”。**

| 配对差（visual − pure，3 seed） | DQN | PPO |
|---|---:|---:|
| ΔBest Dev60 JointR | −0.0076 [−0.042, +0.016] | **+0.0558 [+0.0015, +0.1373]** |
| ΔBest SR | +0.0222 | +0.0611 |
| ΔBest SPL | +0.0226 | +0.0607 |
| ΔBest nDTW | +0.0088 | +0.0467 |
| ΔFinal MA100 JointR | −0.0139 [−0.073, +0.030] | +0.0218 [−0.031, +0.093] |
| ΔFinal Dev60 JointR | −0.0263 [−0.049, +0.011] | +0.0071 [−0.046, +0.057] |
| 末 100 Q mean | 4.55 → **413.1** | — |
| Q 审计 best / last | 11–134 / 355–439 | — |

- **DQN：没有稳定增益。** 3 个 seed 的 ΔBest 为 +0.0037 / +0.0155 /
  −0.0420，均值 −0.0076，CI 含 0；MA100、Final 也均为负均值。视觉
  DQN 的 best 出现在 Ep200–1200，随后验证回落，Q 放大到 10²–10³。
- **PPO：有 best-checkpoint 层面的正向增益。** 3 个 seed 的 ΔBest
  为 +0.0285 / +0.1373 / +0.0015，全部 ≥0，均值 +0.0558，95% CI
  [0.0015, 0.1373]；但 seed2 仅 +0.0015，ΔMA100/ΔFinal 的 CI 均含 0，
  因此是“平均正向、seed 依赖”，不是 uniform 的稳定提升。
- **补充机制实验**：`Visual + Double DQN` best 0.0785 / final −0.0394，
  Q 审计 best 6.02、last 333.27。Double DQN 只能压低早期 Q，无法阻止
  视觉 arm 的长程发散 ⇒ 视觉不稳定的主因更可能是 128-D 特征尺度
  （原始 visual128 的 std≈4.8、max≈161，而 pure 特征约在 [−1,1]），
  而不是单纯的 max 偏差。
- 建议：若继续视觉方向，先对 128-D 做固定标准化/白化或缩放，再比较
  DQN/PPO；否则视觉 DQN 的高 Q 会污染 best checkpoint 选择与部署。

## 5. 产物

- 每个 run：`checkpoints/{best,last,last_full}.pt`、`training_history.json`、
  `validation.csv`、`protocol.json`、`commands.json`、`progress.json`、
  `tensorboard/`；
- 汇总：`analysis/STAGE2M_RESULTS.md`、`analysis/figures/*.png`、
  `analysis/run_summary.csv`、`analysis/paired_visual.csv`、
  `analysis/dqn_vs_ddqn.csv`；
- Q 审计：`stage2m_q_audit/{pre,post_dqn,post_visual,post_visual_last,post_visual_double}/`；
- 未覆盖 Stage-2l 任何文件；Final Test 未读取。
