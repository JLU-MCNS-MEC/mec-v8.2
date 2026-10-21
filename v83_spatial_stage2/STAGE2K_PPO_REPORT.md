# Stage-2k PPO 与 DQN 并行训练：启动报告

- 生成时间（UTC）：2026-10-21T00:05Z（本地 08:05）
- 分支：`experiment/v83-spatial-foundation`
- 性质：**训练启动记录**；未读取 Final Test，未修改任何旧 Checkpoint。
- PPO 运行目录：`runs/v83_spatial_stage2/stage2k_ppo/`
- DQN（正式 A）运行目录：`runs/v83_spatial_stage2/stage2k_pure_mec_v2/`（另一会话负责）

## 1. 启动前现场检查

### 1.1 用户提示中“正在运行的 DQN”

启动检查时（08:04 前）唯一在跑的 DQN 进程是：

```
python -u v83_spatial_stage2/stage2k_jointr_v2_train.py
  --state-mode pure_mec_v2 --order-mode random
  --train-topology-split train --smoke-episodes 4
  --epsilon-decay-steps 500000 --disable-early-stop
  --run-dir runs/v83_spatial_stage2/stage2k_pure_mec_v2_smoke
```

它在 07:57:47（UTC 2026-10-20T23:57:47Z）完成 **4 个 Episode** 的冒烟测试，
并留下 `best.pt / last.pt / last_full.pt`。随后另一个 4-Episode 的
`visual_ablation_v2` 冒烟也完成。两者都**不是**正式的 3500-Episode 训练。
本次没有停止、重启或覆盖它们。

另一个协作会话于 08:04 启动了正式 DQN A：

```
--state-mode pure_mec_v2 --max-episodes 3500 --validation-every 200
--epsilon-decay-steps 500000 --disable-early-stop
--run-dir runs/v83_spatial_stage2/stage2k_pure_mec_v2
```

该进程（PID 385117）与本 PPO 并行运行；DQN B（视觉消融）将在 A 之后串行运行。

### 1.2 当前 DQN 的真实配置（不是计划假设）

| 项 | 实际值 |
|---|---|
| 状态模式 | `pure_mec_v2` |
| MLP 输入 | **384 维** = 2 维 UAV 位置 + 1 维 `t/400` + 1 维 `running_mean_E2E_ms/100` + 20×19 维候选 MEC 特征 |
| 动作 Mask | **20 维可用性 Mask，单独输入，不拼进 MLP** |
| 决策状态合计 | **384 + 20 = 404 维** |
| 动作 | 20 个离散 MEC 节点 |
| 奖励 | `r_nav_t + (-0.25/100) × (running_mean_E2E_t − running_mean_E2E_{t−1})` |
| 奖励恒等式 | 无折扣 Episode Return = `nDTW − 0.25 × mean_E2E_ms / 100` = JointR |
| 训练任务池 | Train 4000 分层随机抽样 3500，无放回，全局打乱 |
| Short/Medium/Long | 1167 / 1167 / 1166 |
| 训练拓扑 | `train` 拓扑；Dev 拓扑仅用于验证 |
| Dev Validation | 固定 60 个 Dev Episode：每场景 20 个（6 Short / 10 Medium / 4 Long） |
| 五种 MEC | Stage-2j 的 Small / Transformer-S / Medium / Transformer-M / Large，全部冻结 |
| 能力先验 | `runs/v83_spatial_stage2/stage2k/capability_prior_v2.json`，sha256 `7859ee81…a0108`，仅 Dev Val 180 |
| 正式最大 Episode | 3500；每 200 Episode 验证；不因普通奖励下降早停 |

冒烟 DQN 的 4 个 Episode 结果（仅用于确认管线，不作为成绩）：
JointR 0.0775、SR 35.00%、nDTW 0.4467、E2E 147.70 ms；
`|Return − JointR|` 最大 `2.50×10^-16`。

## 2. PPO 启动信息

| 项 | 实际值 |
|---|---|
| 算法 | 标准 PPO，Actor-Critic MLP |
| 网络 | 384 → 256 → 256，共享 trunk，actor 20 logits + critic 1 value |
| 动作 Mask | logits 在 softmax 前屏蔽不可用节点（`-1e8`），采样与贪心都不会选到不可用节点 |
| 折扣 | `gamma = 1.0` |
| GAE | `lambda = 0.95` |
| PPO | clipped objective，`eps = 0.2`，`target_kl = 0.05` |
| 学习率 | `3e-4`（Adam） |
| Rollout | 2048 步（按完整 Episode 收集），minibatch 256，每次 10 epochs |
| Value / Entropy | `value_coef = 0.5`，`entropy_coef = 0.01` |
| 梯度裁剪 | 0.5 |
| 奖励 | **直接使用环境 JointR 奖励，无成功终止奖励、无失败惩罚、无其他特权奖励** |
| 训练 Episode | 最多 3500 |
| 验证 | 每 200 Episode，固定 60 Dev Episode，贪心策略 |
| 早停 | 不因奖励/验证下降早停；仅 NaN/Inf、Value/KL 异常或程序错误中止并保留诊断 |
| 设备 | **CUDA 0：NVIDIA GeForce RTX 5090** |
| 启动时显存 | 31,356 MiB 空闲 / 32,101 MiB 总 |
| CPU 线程 | `OMP_NUM_THREADS=8`，`torch.set_num_threads(8)`；实测约 4 核 |
| Python / Torch | `/home/ubuntu5/miniconda3/envs/DRL/bin/python`，Python 3.11.16，torch 2.11.0+cu128 |
| PPO PID | 385079 |
| 运行目录 | `runs/v83_spatial_stage2/stage2k_ppo/` |
| TensorBoard | `runs/tensorboard/association_stage2k_ppo/` |
| 日志 | `runs/v83_spatial_stage2/stage2k_ppo/run.log` |
| 独立 Checkpoint | `checkpoints/best.pt`、`last.pt`、`last_full.pt`（不与 DQN 共享） |

选择 GPU 而不是 CPU 的原因：即使 Pure MEC 状态不含视觉特征，每个环境步仍要运行
冻结的 T5 编码器与五种 MEC 前向；CPU 会慢一个数量级。启动时 GPU 有 31 GB 空闲，
DQN 与 PPO 各约 3 GB，因此并行共享 GPU 是可行的；PPO 限制 8 线程以避免 CPU 争抢。

## 3. 公平性核对（启动时自动执行）

PPO 启动时读取 DQN 的真实配置并写入
`runs/v83_spatial_stage2/stage2k_ppo/protocol_alignment.json`；
正式 DQN A 启动后又做了内容级复核，写入
`protocol_alignment_formal.json`。18 项检查全部为 true：

- 状态模式 `pure_mec_v2`、状态维度 384、动作维度 20、`gamma=1.0`；
- 奖励模式 `jointr_aligned_v2` 与奖励公式字符串一致；
- 训练任务顺序的 **3500 个 Episode UID 数组逐项相同**（JSON 文件 sha 不同仅因
  `created_utc/version/dqn_reference` 元数据不同）；
- 每个 (scenario, band) 目标数、200-Episode 窗口覆盖 9 组；
- Train 拓扑种子、Dev 拓扑种子完全一致且互不相交；
- 60 个 Dev Validation Episode 列表完全一致；
- 能力先验路径与 sha256 一致；
- `max_episodes=3500`、`validation_every=200`、seed `20261031` 一致。

说明：首次启动对齐时正式 DQN A 尚未创建，参考对象是 4-Episode 冒烟配置
（`dqn_max_episodes=4`）；正式 A 启动后复核，除运行元数据外全部相同。
两者唯一差异是算法本身（DQN vs PPO），这正是要比较的变量。

## 4. 当前进度（08:05Z）

| 训练 | 当前 Episode | 状态 |
|---|---:|---|
| DQN A（PID 385117） | 20 / 3500 | training |
| PPO（PID 385079） | 20 / 3500 | training |

PPO 前 20 Episode 的训练日志显示策略熵约 **1.97**、首次更新 `approx_kl ≈ 0.0103`、
`|Return − JointR|` 在 1e-16 量级。首次 Dev Validation 将在 Episode 200 进行。

GPU 合计占用约 6.1 GB / 32.6 GB，利用率约 21%；两个进程各约 4–5 核。
在保证不抢占 DQN 的前提下，PPO 按 8 线程上限运行。

## 5. 验证进度（滚动更新）

### Episode 200（PPO 第一次 Dev60 验证）

| 指标 | PPO Ep200 | Random | Min-Latency | Max-Quality | Heuristic | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| JointR | **0.0344** | 0.1931 | 0.2828 | 0.2091 | 0.2806 | 0.4425 |
| SR | 38.33% | 55.00% | 38.33% | 46.67% | 43.33% | 65.00% |
| nDTW | 0.4266 | 0.5598 | 0.4614 | 0.5405 | 0.5236 | 0.6759 |
| E2E (ms) | 156.9 | 146.7 | 71.4 | 132.5 | 97.2 | 93.4 |

- PPO 训练诊断：策略熵 **1.625**，Value Loss **0.0118**，`approx_kl` **0.0110**，
  `|Return − JointR|` 在 1e-16 量级；无 NaN/Inf 或 KL 异常。
- 首个 Checkpoint 的 JointR **低于全部 Baseline**，SR 与 Min-Latency 相同但
  nDTW 更低、E2E 更高。此时仅训练 200/3500 Episode，**不能据此判定收敛或失败**；
  需要继续看 Ep400/600/800 的验证趋势。

### Ep200 同验证集配对（PPO − DQN A）

| 指标 | PPO Ep200 | DQN A Ep200 | Δ（PPO − DQN） |
|---|---:|---:|---:|
| JointR | 0.0344 | 0.0750 | **−0.0406** |
| SR | 38.33% | 35.00% | +3.33 pp |
| nDTW | 0.4266 | 0.4503 | −0.0237 |
| E2E (ms) | 156.9 | 150.1 | +6.8 |

DQN A 同样低于全部非特权 Baseline（其 Ep200 JointR 0.0750），说明两个算法都仍处于
早期学习阶段；当前差距不能作为最终结论。

### 验证轨迹（滚动更新）

| Episode | PPO JointR | PPO SR | PPO nDTW | PPO E2E | DQN A JointR | DQN A SR | DQN A nDTW | DQN A E2E |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.0344 | 38.33% | 0.4266 | 156.9 | 0.0750 | 35.00% | 0.4503 | 150.1 |
| 400 | 0.0274 | 38.33% | 0.4326 | 162.1 | 0.0571 | 33.33% | 0.4400 | 153.2 |
| 600 | 0.0093 | 30.00% | 0.3933 | 153.6 | **0.1336** | 38.33% | 0.5109 | 150.9 |
| 800 | **0.1245** | 33.33% | 0.4742 | 139.9 | 0.1004 | 31.67% | 0.4775 | 150.8 |
| 1000 | 0.0928 | 36.67% | 0.4519 | 143.6 | 0.0934 | 38.33% | 0.4790 | 154.2 |
| 1200 | 0.0511 | 30.00% | 0.4111 | 144.0 | 0.0653 | 33.33% | 0.4534 | 155.2 |
| 1400 | 0.0900 | 36.67% | 0.4446 | 141.8 | 0.1151 | 35.00% | 0.4771 | 144.8 |
| 1600 | **0.1324** | 35.00% | 0.4754 | **137.2** | 0.0958 | 43.33% | 0.4840 | 155.3 |
| 1800 | 0.0920 | 35.00% | 0.4391 | 138.8 | 0.0915 | 38.33% | 0.4896 | 159.2 |
| 2000 | **0.1466** | 38.33% | 0.4665 | **127.9** | 0.0706 | 38.33% | 0.4652 | 157.8 |
| 2200 | 0.1223 | 31.67% | 0.4457 | 129.4 | 0.0916 | **46.67%** | 0.4675 | 150.4 |
| 2400 | 0.1186 | **40.00%** | 0.4419 | 129.3 | 0.1052 | **46.67%** | 0.4854 | 152.1 |
| 2600 | 0.0826 | 35.00% | 0.4184 | 134.3 | 0.0810 | 36.67% | 0.4466 | 146.3 |
| 2800 | 0.0674 | 31.67% | 0.4075 | 136.1 | 0.0816 | 43.33% | 0.4444 | 145.1 |
| 3000 | 0.0613 | 31.67% | 0.4157 | 141.8 | 0.1315 | 38.33% | 0.4854 | 141.5 |
| 3200 | 0.0838 | 35.00% | 0.4302 | 138.6 | 0.0948 | 35.00% | 0.4686 | 149.5 |
| 3400 | **0.1447** | **40.00%** | 0.4704 | 130.3 | 0.1079 | 45.00% | 0.4925 | 153.8 |
| 3500 | 0.1329 | 41.67% | 0.4818 | 139.5 | 0.1125 | 43.33% | 0.4899 | 151.0 |

Ep400 配对（PPO − DQN A）：JointR **−0.0297**、SR **+5.00 pp**、nDTW **−0.0074**、
E2E **+8.9 ms**。两边都未刷新 Ep200 最佳（DQN A `no_improve=1`，状态
`plateaued`），也都低于全部非特权 Baseline；仍处于训练早中期。

PPO 在 Ep400 未刷新最佳（`no_improve=1`，最后 3 次趋势 −0.00695，状态
`plateaued`）。策略熵从 1.625 降到 1.416，但验证 JointR 反而下降；MEC 选择从
较均衡的分布转向 **Large 25.5% + Transformer-M 28.9%**，Transformer-S 降到 5.9%，
平均 E2E 从 156.9 ms 升到 162.1 ms，nDTW 只从 0.4266 升到 0.4326。说明 PPO
目前偏向高能力/高时延节点，尚未学会 JointR 的质量-时延权衡。KL 仍稳定在 0.0103，
数值健康；这是策略学习问题，不是数值异常。

Ep600 进一步下降到 **JointR 0.0093、SR 30.00%、nDTW 0.3933、E2E 153.6 ms**，
`no_improve=2`、最后 3 次趋势 **−0.01252**。与此同时训练侧 100-Episode
Return 均值升到 **0.2625**（中位数 0.4424），Value Loss 0.0096、KL 0.0158、
clip fraction 0.178、explained variance 0.943。训练 Return 与 Dev60 验证
JointR 的差距扩大，呈现 **训练拓扑过拟合/Dev 泛化下降** 的特征，而不是数值发散。
MEC 使用率：Transformer-M 28.6%、Small 24.2%、Large 17.4%、Medium 17.7%、
Transformer-S 12.1%。

Ep600 配对（PPO − DQN A）：JointR **−0.1243**、SR **−8.33 pp**、
nDTW **−0.1176**、E2E **+2.7 ms**。DQN A 在 Ep600 刷新最佳（0.1336），
而 PPO 当时连续第三次下降；两者在训练中段一度明显分化。

**Ep800 修正：** PPO 反弹到 **JointR 0.1245（新最佳）**、SR 33.33%、
nDTW 0.4742、E2E **139.9 ms**；`no_improve=0`、最后 3 次趋势 **+0.04857**、
状态 `improving`。策略熵 1.372、Value Loss 0.0108、KL 0.0158，数值正常。
MEC 使用率变为 Transformer-M 29.8%、Small 25.1%、Transformer-S 22.4%、
Medium 11.7%、Large 10.9%，Large 占比从 Ep400 的 25.5% 降到 10.9%，E2E 随之下降。

Ep800 配对（PPO − DQN A）：JointR **+0.0241**、SR **+1.67 pp**、
nDTW −0.0033、E2E **−10.9 ms**。DQN A 在 Ep800 回落到 0.1004（未刷新
Ep600 的 0.1336），`no_improve=1`。按各自最佳 Checkpoint 比较，DQN A
0.1336 仍略高于 PPO 0.1245（差 0.0091，处于验证噪声量级）。

Ep1000 配对（PPO − DQN A）：JointR **−0.0006**（基本持平）、
SR −1.67 pp、nDTW −0.0271、E2E **−10.6 ms**。PPO 用更低的时延换取了略低的
nDTW，JointR 与 DQN A 打平；两边都未刷新最佳，DQN A `no_improve=2`。

Ep1200：PPO 进一步回落到 **JointR 0.0511、SR 30.00%、nDTW 0.4111、
E2E 144.0 ms**，`no_improve=2`、最后 3 次趋势 **−0.03671**、状态 `plateaued`。
策略熵 1.289、Value Loss 0.0098、KL 0.0150，数值正常。

**Ep1600 修正：** PPO 刷新到 **JointR 0.1324（新最佳）**、SR 35.00%、
nDTW 0.4754、E2E **137.2 ms**；`no_improve=0`、最后 3 次趋势 **+0.04065**、
状态 `improving`。策略熵 1.361、Value Loss 0.0122、KL 0.0142。

Ep800 的 0.1245 与 Ep1600 的 0.1324 都处于同一量级，因此不能把 Ep800 简单
归为一次孤立尖峰；PPO 在 0.09–0.13 区间反复出现高点，但中间仍夹着 0.05–0.06
的低点，**没有形成连续稳定平台**。最佳 Checkpoint 现为 Ep1600 的 0.1324，
与 DQN A 的 0.1336 仅差 **0.0012**，完全处于验证噪声量级。

Ep1200 配对（PPO − DQN A）：JointR **−0.0142**、SR **−3.33 pp**、
nDTW **−0.0423**、E2E **−11.2 ms**。DQN A 也在 Ep1200 回落（0.0653），
`no_improve=3`；两边都处于 plateaued。当前最佳 Checkpoint：DQN A 0.1336
(Ep600) vs PPO 0.1245 (Ep800)，差 0.0091。

Ep1400 配对（PPO − DQN A）：JointR **−0.0251**、SR **+1.67 pp**、
nDTW **−0.0325**、E2E **−3.0 ms**。DQN A 回到 0.1151（`no_improve=4`），
PPO 0.0900；最佳 Checkpoint 不变。

Ep1600 配对（PPO − DQN A）：JointR **+0.0366**、SR **−8.33 pp**、
nDTW **−0.0086**、E2E **−18.1 ms**。DQN A 的 SR/OSR 更高（43.33% / 71.67%），
但 E2E 155.3 ms 明显高于 PPO 的 137.2 ms；按 JointR 的时延权重，PPO 在该点
反而领先。DQN A `no_improve=5`，最佳仍为 Ep600 的 0.1336。

**Ep2000：PPO 刷新最佳到 JointR 0.1466**（SR 38.33%、OSR 61.67%、
SPL 0.3827、nDTW 0.4665、E2E **127.9 ms**、P95 193.1 ms、平均 168.5 步），
`no_improve=0`、最后 3 次趋势 **+0.00711**、状态 `improving`。策略熵 1.330、
Value Loss 0.0099、KL 0.0137、explained variance 0.962。
MEC 使用率出现 **Large 40.4%**、Medium 20.5%、Transformer-S 17.1%、
Small 14.0%、Transformer-M 8.1%；Large 占比升高但 E2E 反而降到 127.9 ms，
说明策略在部分节点上选择了通信时延更低的组合。训练侧 100-Episode Return
均值降到 0.0672，与 Dev 验证方向相反，再次说明训练 Return 不能替代固定
Dev 验证。PPO 最佳 Checkpoint 现为 **Ep2000 的 0.1466**，暂时高于 DQN A 的
0.1336 约 0.0130；但仅领先一个验证点，尚不构成收敛或显著优势。

Ep1800 配对（PPO − DQN A）：JointR **+0.0005**（打平）、SR −3.33 pp、
nDTW **−0.0505**、E2E **−20.4 ms**。DQN A 的 nDTW 更高但时延高 20.4 ms，
两者的 JointR 再次打平；DQN A `no_improve=6`，最佳仍是 Ep600 的 0.1336。

Ep2000 配对（PPO − DQN A）：JointR **+0.0760**、SR **0.00 pp**、
nDTW **+0.0013**、E2E **−29.9 ms**。这是目前最清晰的一次分离：**SR 与 nDTW
几乎完全相同（38.33% vs 38.33%，0.4665 vs 0.4652），PPO 仅凭 E2E 低 29.9 ms
就取得 +0.0760 的 JointR 优势**。DQN A `no_improve=7`，最佳停在 Ep600；
PPO 最佳为 Ep2000 的 0.1466。该差异是否在完整 Dev180 上显著，仍需最终
逐 Episode 配对检验。

Ep2200 配对（PPO − DQN A）：JointR **+0.0307**、SR **−15.00 pp**、
nDTW **−0.0218**、E2E **−21.0 ms**。DQN A 的 SR 达到 46.67%（目前最高），
但 E2E 150.4 ms；PPO 的 SR 仅 31.67%，靠 129.4 ms 的时延取得 JointR 领先。
DQN A `no_improve=8`，最佳仍停在 Ep600 的 0.1336。

Ep2400 配对（PPO − DQN A）：JointR **+0.0134**、SR **−6.67 pp**、
nDTW **−0.0435**、E2E **−22.8 ms**。DQN A 保持 46.67% 的 SR 和 0.4854 的
nDTW，但 152.1 ms 的时延仍使 JointR 落后于 PPO（129.3 ms）；DQN A
`no_improve=9`，最佳停在 Ep600。

Ep2600 配对（PPO − DQN A）：JointR **+0.0016**（打平）、SR −1.67 pp、
nDTW **−0.0282**、E2E **−12.0 ms**。两边都进入晚期平台：DQN A
`no_improve=10`，PPO `no_improve=4`；最佳 Checkpoint 不变。

Ep2800 配对（PPO − DQN A）：JointR **−0.0142**、SR **−11.66 pp**、
nDTW **−0.0369**、E2E **−9.0 ms**。DQN A 的 SR 43.33%、nDTW 0.4444，
PPO 的 E2E 只低 9.0 ms，不足以抵消质量差距；DQN A `no_improve=11`。

Ep3000 配对（PPO − DQN A）：JointR **−0.0702**、SR **−6.66 pp**、
nDTW **−0.0697**、E2E −2.9 ms。DQN A 回到 0.1315（距其最佳仅 0.0021），
PPO 在该点为 0.0613 的低谷；这再次说明单点比较会被验证噪声主导。
DQN A `no_improve=12`。

Ep3200 配对（PPO − DQN A）：JointR **−0.0110**、SR **0.00 pp**、
nDTW **−0.0384**、E2E **−10.9 ms**。DQN A 的 nDTW 更高，PPO 的时延优势
不足以反超；DQN A `no_improve=13`。

Ep3400 配对（PPO − DQN A）：JointR **+0.0368**、SR **−5.00 pp**、
nDTW **−0.0221**、E2E **−23.5 ms**。DQN A 的 SR 45.00%、nDTW 0.4925，
PPO 的 SR 40.00%、nDTW 0.4704；PPO 靠低 23.5 ms 的 E2E 取得 JointR 领先。
DQN A `no_improve=14`。

这说明 **60 个 Dev Episode 的验证噪声很大**：Ep600 的三连降不能单独作为
“PPO 失败”的证据，Ep800 的反弹同样不能单独作为“已收敛”的证据。此前
“趋势明确为未收敛”的判断是过早的，已按实际数据修正。最终结论必须等待
3500 Episode 结束后的完整验证轨迹与最佳 Checkpoint。

## 6. PPO 训练完成（Ep3500）

- 实际训练：**3500 Episode / 506,454 环境交互步 / 21,260 次 PPO 梯度更新**
- 最佳 Dev60 JointR：**0.1466（Episode 2000）**
- 最后 Dev60 JointR：0.1329（Ep3500；SR 41.67%、nDTW 0.4818、E2E 139.5 ms）
- 收敛状态：**`not_converged`**；`stable_platform=false`、`improving=false`，
  最后 3 次趋势 +0.02457
- 最终策略熵：1.290；Value Loss：0.00753；approx KL：0.01464；
  explained variance：0.955
- 奖励一致性：`|Return − JointR|` 最大 **1.44e-15**
- 总耗时：12,098 秒（约 3 小时 22 分）
- Checkpoint：`checkpoints/best.pt`（Ep2000）、`last.pt`、`last_full.pt`

PPO 的最佳轨迹为 0.0344 (Ep200) → 0.1245 (Ep800) → 0.1324 (Ep1600) →
**0.1466 (Ep2000)** → 0.1447 (Ep3400) → 0.1329 (Ep3500)。多个高点集中在 0.12–0.15，
但中间夹有 0.009–0.09 的低点，未形成连续稳定平台；因此按预注册标准
**不构成收敛**。它仍将按“最佳验证 Checkpoint”进入 Dev180 诊断对比，
但不会被标记为有效的最终策略，也不会进入 Final Test。

## 6b. DQN A 对照训练完成（Ep3500）

- 实际训练：**3500 Episode / 521,491 环境交互步**
- 最佳 Dev60 JointR：**0.1336（Episode 600）**
- 最后 Dev60 JointR：0.1125（Ep3500；SR 43.33%、nDTW 0.4899、E2E 151.0 ms）
- 收敛状态：**`not_converged`**；`stable_platform=false`、`improving=false`
- 奖励一致性：`|Return − JointR|` 最大 **1.22e-15**
- 总耗时：13,878 秒（约 3 小时 51 分）
- Checkpoint：`runs/v83_spatial_stage2/stage2k_pure_mec_v2/checkpoints/best.pt`（Ep600）

Ep3500 配对（PPO − DQN A）：JointR **+0.0204**、SR **−1.66 pp**、
nDTW **−0.0081**、E2E **−11.5 ms**。PPO 在最后一点靠低时延领先，
但两者都未收敛；最佳 Checkpoint 比较为 PPO 0.1466 (Ep2000) vs
DQN 0.1336 (Ep600)。完整 Dev180 对比见第 8 节。

## 7. PPO Dev180 评估（最佳 Checkpoint Ep2000）

Checkpoint：`runs/v83_spatial_stage2/stage2k_ppo/checkpoints/best.pt`（Ep2000）
数据：完整 Dev Validation **180** 个 Episode；未读取 Final Test。

| 策略 | SR | OSR | SPL | nDTW | E2E (ms) | JointR |
|---|---:|---:|---:|---:|---:|---:|
| **PPO best** | **38.33%** | 68.33% | 0.3786 | 0.4223 | **127.45** | **0.1037** |
| Random | 55.56% | — | — | 0.5582 | 142.87 | 0.2010 |
| Min-Latency | 36.11% | — | — | 0.4192 | **68.60** | 0.2477 |
| Max-Quality | 52.78% | — | — | 0.5417 | 126.47 | 0.2256 |
| Heuristic | 43.33% | — | — | 0.4810 | 87.72 | 0.2617 |
| Oracle（特权） | 66.67% | — | — | 0.6496 | 90.44 | 0.4236 |

逐 Episode 配对（PPO − Baseline，95% Bootstrap CI，n=180）：

| Baseline | ΔJointR | 95% CI | 胜/负/平 |
|---|---:|---|---:|
| Random | **−0.0973** | [−0.1548, −0.0419] | 92 / 88 / 0 |
| Min-Latency | **−0.1440** | [−0.1934, −0.0952] | 20 / 153 / 7 |
| Max-Quality | **−0.1219** | [−0.1843, −0.0597] | 66 / 110 / 4 |
| Heuristic | **−0.1580** | [−0.2130, −0.1024] | 37 / 138 / 5 |

四个非特权 Baseline 的 95% CI **全部位于 0 以下**：PPO 最佳 Checkpoint 在完整
Dev180 上显著差于全部四个非特权 Baseline。分场景 JointR：quality_rich
**0.2856**、balanced **0.0480**、tradeoff_stress **−0.0226**，场景间差异极大。

**关键诊断：** PPO 最佳 Checkpoint 在用于选择的 60 个 Dev Episode 上是 0.1466，
但在完整 180 个 Dev Episode 上只有 **0.1037**，下降 0.0429。说明 Ep2000 的
Dev60 高点有相当一部分来自 **验证子集选择/过拟合**；120 个未参与选择的 Dev
Episode 没有支持同样的水平。这也是最终对比必须使用完整 Dev180、而不能只看
Dev60 的原因。

### PPO 分场景失败模式

| 场景 | n | SR | nDTW | E2E (ms) | JointR | 主要 MEC 使用 |
|---|---:|---:|---:|---:|---:|---|
| balanced | 60 | 35.00% | 0.3784 | 132.2 | **0.0480** | Large 50%、Medium 33% |
| quality_rich | 60 | 48.33% | 0.5207 | 94.1 | **0.2856** | Transformer-S 48%、Large 29% |
| tradeoff_stress | 60 | 31.67% | 0.3678 | **156.1** | **−0.0226** | Large 32%、Small 30%、Medium 15% |

PPO 在 quality_rich 上表现尚可（JointR 0.2856），但在 balanced 与
tradeoff_stress 上失败：balanced 过度选择 Large（50%），nDTW 只有 0.3784；
tradeoff_stress 的 E2E 高达 156.1 ms，nDTW 0.3678，JointR 为负。说明策略
**没有学会按场景/拓扑调整质量-时延权衡**，而是把在 quality_rich 上有效的
Large/Transformer-S 组合外推到其他场景。

## 8. 最终 Dev180 对比（DQN A vs PPO）

两边均使用各自最佳验证 Checkpoint、同一批 180 个 Dev Episode、同一 Dev Topology
与同一组 Baseline。

| 策略 | SR | OSR | SPL | nDTW | E2E (ms) | JointR |
|---|---:|---:|---:|---:|---:|---:|
| DQN A best（Ep600） | 39.44% | 65.56% | 0.3942 | 0.4851 | 149.38 | **0.1116** |
| PPO best（Ep2000） | 38.33% | 68.33% | 0.3786 | 0.4223 | **127.45** | **0.1037** |

逐 Episode 配对（PPO − DQN，n=180，95% Bootstrap CI）：

| 指标 | 均值差 | 95% CI | 结论 |
|---|---:|---|---|
| JointR | **−0.0079** | [−0.0689, +0.0526] | 不显著（CI 跨 0） |
| nDTW | **−0.0628** | [−0.1246, −0.0021] | PPO 显著更低 |
| SR | −0.0111 | [−0.0889, +0.0667] | 不显著 |
| E2E (ms) | **−21.93** | [−30.00, −14.03] | PPO 显著更低 |

配对胜负：PPO 胜 **105** / 负 **73** / 平 **2**。PPO 在更多 Episode 上取得更高
JointR，但 DQN 的胜场幅度更大，均值差反而不显著。

**与 Baseline 比较：** DQN 与 PPO 的最佳 Checkpoint 对 Random / Min-Latency /
Max-Quality / Heuristic 的 ΔJointR 95% CI 全部位于 0 以下——两者都显著差于
全部四个非特权 Baseline。Oracle 为特权上界，不参与显著性比较。

**选择偏差：** Dev60 → Dev180，DQN 最佳从 0.1336 降到 0.1116（−0.0220），
PPO 从 0.1466 降到 0.1037（−0.0429）。60 个 Dev Episode 的选模对其余 120 个
Dev Episode 存在明显选择偏差，PPO 更严重；因此不能只看 Dev60 的最佳值。

### 最佳 vs 最后 Checkpoint（补充）

| Checkpoint | SR | OSR | SPL | nDTW | E2E (ms) | JointR |
|---|---:|---:|---:|---:|---:|---:|
| DQN A best（Ep600） | 39.44% | 65.56% | 0.3942 | 0.4851 | 149.38 | **0.1116** |
| DQN A last（Ep3500） | **46.11%** | **72.78%** | **0.4611** | 0.4753 | 146.87 | 0.1082 |
| PPO best（Ep2000） | 38.33% | 68.33% | 0.3786 | 0.4223 | **127.45** | **0.1037** |
| PPO last（Ep3500） | 40.00% | 67.78% | 0.3931 | 0.4304 | 136.22 | 0.0898 |

DQN A 的 last 比 best 的 SR/OSR/SPL 更高，但 nDTW 略低，JointR 从 0.1116 降到
0.1082；PPO last 的 SR 也略高于 best（40.00% vs 38.33%），但 nDTW 只略高、
E2E 更高，JointR 从 0.1037 降到 0.0898。两个 last 都显著低于全部四个非特权
Baseline（ΔJointR 95% CI 均低于 0）。

逐 Episode 比较 last − best（n=180，95% CI）：

| 算法 | JointR 差 | 95% CI | last 胜 / best 胜 |
|---|---:|---|---:|
| DQN A | −0.0034 | [−0.0544, +0.0465] | 104 / 76 |
| PPO | −0.0139 | [−0.0637, +0.0377] | 76 / 98 |

两边 last−best 都不显著，但 **best 按 JointR 选择仍是更优 Checkpoint**
（DQN 0.1116 > 0.1082；PPO 0.1037 > 0.0898）。配套 Q 审计（另一会话）显示
DQN best→last：Q mean **1.13 → 10.17**、Q max **2.43 → 21.74**、贪心动作
一致率仅 **17.1%**——最后 Checkpoint 的策略已经明显漂移，不能把 last 当作
best 的等价替代；PPO last 的 JointR 也低于 best。最终报告仍以 best Checkpoint
为正式比较对象。

**最终结论：**

- DQN A 与 PPO 都 `not_converged`，都不能标记为有效最终策略，也未进入 Final Test。
- DQN−PPO 的 JointR 差异不显著；PPO 的 E2E 显著低 21.93 ms，但 nDTW 显著低
  0.0628，SR 差异不显著。两者是不同失败模式的权衡，PPO 并非全面更优，
  DQN 也并非显著更优。
- 两者都显著低于全部四个非特权 Baseline，说明在本状态设计与 3500 Episode
  预算下，两种算法都未达到可用策略水平。
- 完整曲线、四指标图和配对统计见 `STAGE2K_PPO_VS_DQN_COMPARISON.md` 与
  `figures/stage2k_ppo_vs_dqn/`。

## 9. 安全与归档

- 未读取 Final Test 300；`final_test_read=false`。
- 旧 DQN、旧 Pure MEC、旧视觉消融的 Checkpoint、日志、TensorBoard 均未修改或覆盖。
- PPO 使用独立目录，不会覆盖任何 DQN 文件。
- 大型权重、日志与 TensorBoard 保留在 `runs/`（已被 `.gitignore` 忽略）。
- 后续将更新本报告，加入训练曲线、验证趋势与最终对比。
