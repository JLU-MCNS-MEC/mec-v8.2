# Stage-2p 实验协议：PPO 原始奖励 vs 方案 D

日期：2026-10-23  
分支：`experiment/v83-spatial-foundation`  
仓库：`JLU-MCNS-MEC/mec-v8.2`  
实验根目录：`runs/v83_spatial_stage2/stage2p/`（被 `.gitignore` 排除）  
可追踪产物：`v83_spatial_stage2/stage2p/`  
父协议：`v83_spatial_stage2/STAGE2N_PROTOCOL.md`、
`v83_spatial_stage2/stage2o/STAGE2O_PROTOCOL.md`

本协议在 Stage-2p 正式训练启动前冻结。Stage-2n/2o 的历史结果不覆盖；
Test-400 不读取。

---

## 1. 目标

在固定的 Stage-2n 条件（单城市、A 拓扑、20 个同质 Transformer-M MEC、
Train-4000 / Dev-60、同一任务顺序、384-D Pure-MEC v2 状态、20-D Mask、
同一 PPO 超参数与 `gamma=1.0`）下，只改变 PPO 的奖励函数，比较：

- **Baseline：原始奖励**
  `r_t = -0.25 * (mean_t - mean_{t-1}) / 100`
- **方案 D：固定尺度逐步时延反馈 + Episode 终止归一化修正**
  `u_t = 1 - e2e_t/tau`，`r_t = u_t/H`，末步加
  `(1 - T/H) * J`，其中 `J = 1 - mean_E2E/tau`。

`tau = 100 ms`，`H = 400`。Baseline 与 D 的 Episode 累计回报满足

```text
4 * R_baseline + 1 = R_D
```

因此奖励曲线在原始尺度上不可直接比较；报告同时给出 `4*R_baseline + 1`
与 D 的对齐尺度。主要判据是实际 E2E 时延、Dev-60 与稳定性，而不是奖励
曲线的绝对高度。

---

## 2. 固定条件

| 项目 | 固定值 |
|---|---|
| 城市 | `map_seed = 10,000,001`，160 个地标，1500 m × 1500 m |
| 拓扑 | Stage-2l `A_fixed` 的 20 个节点位置 |
| MEC | 20 个完全同质节点，同一 Transformer-M（Stage-2j） |
| 权重 SHA256 | `57c88b7d71e1a7f17862aaa8429afa8a041cba5e747f744f3a7febf47f273c6d` |
| 部署 SHA256 | `4c0950cf6e438cff41454eee6daefb831fa984f425ec1a41cf3974257c6d1148` |
| 任务池 | Stage-2n Train 4000 / Dev 60（Test 400 封存） |
| 任务顺序 | Stage-2n 固定顺序（种子 20261022），六组共用 |
| 状态 | 384-D Pure-MEC v2 + 独立 20-D 可用性 Mask |
| 动作空间 | 20 个 MEC 节点 |
| 时延阈值 | `tau = 100 ms` |
| 最大决策步数 | `H = 400` |
| 折扣因子 | `gamma = 1.0` |
| 算法 | PPO（与 Stage-2n/2o 相同的网络与超参数） |

PPO 超参数（不修改）：

```text
actor-critic 384 → 256 → 256 → {20, 1}
gamma = 1.0
GAE lambda = 0.95
clip epsilon = 0.2
rollout = 2,048 steps
minibatch = 256
update epochs = 10
value coef = 0.5
entropy coef = 0.01
Adam lr = 3e-4
max grad norm = 0.5
target KL = 0.05
```

---

## 3. 复用审计与实验路径

Stage-2n 的原始奖励 PPO（seed 0/1/2）与 Stage-2o 的 D/PPO（seed 0）使用
了**不同的训练脚本**（`stage2n_train.py` vs `stage2o/train.py`）。审计确认：

- 任务池哈希、训练顺序哈希、部署哈希、MEC 权重哈希完全相同；
- state/action 契约与 PPO 网络/超参数完全相同；
- 训练脚本和奖励应用代码路径不同，日志/溯源标签不同。

因此 Stage-2p **不直接混用**这些结果；它用同一个 Stage-2p PPO 循环重跑
`baseline` 与 `D` 的 seed 0/1/2，共 6 个 run。Stage-2n/2o 的 seed-0 结果
在训练后只作为复现对照（新 baseline seed0 对 Stage-2n PPO seed0；新 D
seed0 对 Stage-2o D seed0）。审计结果保存在
`evidence/reuse_audit.json`。

---

## 4. 训练与验证

- 6 个 run：`{baseline, D} × {0,1,2}`，每个 4000 Episodes；
- 每 200 Episodes 在固定 Dev-60 上验证一次，第 4000 Episode 再验证一次；
- 不早停；只有 NaN/Inf、PPO KL 爆炸或程序错误会中止；
- 每 200 Episodes 输出中文进展并保存训练日志、Dev 结果、
  `rounds/round_XXXX.json` 与 `rounds.csv`；
- Test-400 全程不读取。

### 收敛判据（沿用 Stage-2o，训练前冻结）

只用 Dev 平均 E2E 时延：

```text
最近三次 Dev 验证的 max(E2E) - min(E2E) <= 0.01 * 最佳 E2E
且 最后一次 E2E <= 1.01 * 最佳 E2E
```

达到 4000 Episodes 仍不满足则标记 `not_converged`。报告同时给出验证噪声
（Stage-2n 六次 run 末 5 次 Dev E2E 标准差 0.89–1.67 ms），不能只凭单条
曲线或单个最佳检查点宣布收敛。

### 最佳检查点选择

```text
最低 Dev 平均 E2E；E2E 完全相同时依次比较更高 SR、更早 Episode。
```

---

## 5. 每 200 Episodes 的汇报

每个 run 在 Episode 200、400、…、4000 输出：

- 当前 Episode、训练进度、耗时、GPU 利用率与显存；
- 最近 100 Episode 的 MA100 奖励、MA100 E2E；
- Dev-60 平均 E2E、P95 E2E、时延违约率、SR、SPL、nDTW；
- PPO 的 Value Loss、Explained Variance、Entropy、KL；
- 奖励震荡、时延退化、价值估计异常的判定。

---

## 6. 产物与隔离

- 原始 checkpoint、TensorBoard Event、逐 Episode 大日志：
  `runs/v83_spatial_stage2/stage2p/`（`.gitignore` 排除）；
- 代码：`v83_spatial_stage2/stage2p/*.py`；
- 测试：`v83_spatial_stage2/stage2p/test_stage2p.py`；
- 协议与本文件、冻结文件、逐 seed 结果、CSV、PNG、中文报告：
  `v83_spatial_stage2/stage2p/evidence/` 与 `STAGE2P_REPORT.md`；
- 不覆盖 Stage-2n/2o 的任务、检查点、日志或结果；
- 不停止、不覆盖服务器上已有的其他实验。

---

## 7. 预先声明的限制

1. 单城市、单部署、单 MEC 检查点；结论不外推到异质算力/带宽或新城市。
2. 只有 3 个随机种子；3 seeds 仍不足以做强统计推断，但可以检查种子间
   方向一致性。
3. 同质部署下导航动作与关联策略无关，SR/SPL/nDTW 不用于区分奖励方案；
   方案差异只能来自 E2E 时延、收敛与稳定性。
4. Test-400 只读一次；本阶段不读取。
