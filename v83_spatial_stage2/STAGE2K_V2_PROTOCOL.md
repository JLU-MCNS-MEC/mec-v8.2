# Stage-2k v2 受控消融协议（预注册）

日期：2026-10-21  
分支：`experiment/v83-spatial-foundation`  
仓库：`JLU-MCNS-MEC/mec-v8.2`  
新运行：

- A（Pure MEC，无视觉）：`runs/v83_spatial_stage2/stage2k_pure_mec_v2/`
- B（A + 固定 128-D VisualProjector）：`runs/v83_spatial_stage2/stage2k_visual_ablation_v2/`

本协议在正式训练开始前冻结。旧运行 `stage2k/`、`stage2k_jointr_v2/`、
`stage2k_no_visual/`、`stage2k_pure_mec_state/` 的全部权重、日志、报告与
TensorBoard 均保留，不覆盖。五种 Stage-2j T5 MEC Checkpoint 全部冻结，
`eval()` + `requires_grad=False`，不重新训练。未读取 Stage-2j Final Test 300，
未读取 Association Final Test 180。

---

## 1. 前置审计结论

`STAGE2K_V2_PREEXPERIMENT_AUDIT.md` 已确认：

- 旧 DQN 的 ε 从 1.0 线性衰减到 0.05，预算 **200,000 环境交互步**；
- Q mean/max 在后期持续增大（JointR v2 full 的 Q mean 在 Ep200→Ep1320
  为 3.23→157.82；pure MEC v1 为 0.47→5.62）；
- 100-Episode return 峰值不是收敛；旧运行的 `converged` 标志不可作为收敛证据；
- 因此 v2 使用 **500,000** ε 预算、**3500 Episodes**、**禁止早停**，并以固定
  Dev Validation 的稳定平台判定收敛。

---

## 2. 状态设计

### 2.1 A：Pure MEC 状态（384-D MLP 输入 + 20-D 动作 Mask）

```text
2-D UAV position / 1500
+ 1-D episode step t / 400
+ 1-D running_mean_E2E_ms / 100
+ 20 candidates x 19-D MEC features
= 384-D MLP input

+ 20-D availability action mask (separate, not duplicated in the MLP)
= 404-D decision state
```

每个候选 MEC 的 19 维：

| # | 字段 | 说明 | 归一化 |
|---:|---|---|---|
| 0 | `relative_x` | MEC x − UAV x | / 500 m |
| 1 | `relative_y` | MEC y − UAV y | / 500 m |
| 2 | `distance` | UAV–MEC 距离 | / 500 m |
| 3 | `compute_norm` | 节点算力 | / 2 TFLOPs |
| 4 | `bandwidth_norm` | 节点带宽 | / 30 MHz |
| 5 | `uplink_rate_norm` | 上行速率 | / 200 Mbps |
| 6 | `communication_ms_norm` | 通信时延 | / 100 ms |
| 7 | `inference_ms_norm` | 估计推理时延 | / 10 ms，上限 3 |
| 8 | `e2e_ms_norm` | 通信 + 推理 + 统一模型 E2E | / 100 ms，上限 3 |
| 9 | `snr_db_norm` | SNR | / 40 dB，裁剪 [-1,1] |
| 10 | `dev_sr` | 该 MEC 的 Dev 闭环 SR | 原值 [0,1] |
| 11 | `dev_osr` | 该 MEC 的 Dev 闭环 OSR | 原值 [0,1] |
| 12 | `dev_spl` | 该 MEC 的 Dev 闭环 SPL | 原值 [0,1] |
| 13 | `dev_ndtw` | 该 MEC 的 Dev 闭环 nDTW | 原值 [0,1] |
| 14–18 | `model_*` | 五种 MEC 类型 one-hot | 0/1 |

全局的 `t/400` 在初始状态为 0；执行第 t 步后构建下一状态时使用 t。
`running_mean_E2E_ms` 是已执行步骤的运行平均，初始为 0，按
`mean_t = mean_{t-1} + (d_t - mean_{t-1}) / t` 更新；状态字段为
`running_mean_E2E_ms / 100`，不截断。

**可用性 Mask**：独立的 20 维 0/1 向量，只用于 Q 值屏蔽；不重复进入 MLP。

**明确排除**：视觉/PCA、Goal token、目标视觉相似度、历史视觉、热图、GAB、
地图检索、导航动作、目标位置、目标距离、Oracle action、成功标签。

**能力先验**：不再使用单一 `dev_val_quality_prior`，而是每个 MEC 类型独立的
Dev 闭环 SR、OSR、SPL、nDTW 四项。来源为 Stage-2j Dev Val 180
（episode seeds 2,000,000+，与 Stage-2k Train/Dev/Test 的 5M/6M/7M 不相交），
`final_test_read=false`。构建脚本：`stage2k_build_capability_prior_v2.py`；
产物：`runs/v83_spatial_stage2/stage2k/capability_prior_v2.json`。

| MEC | Dev SR | Dev OSR | Dev SPL | Dev nDTW |
|---|---:|---:|---:|---:|
| Small | 0.1611 | 0.3222 | 0.1611 | 0.2314 |
| Transformer-S | 0.5333 | 0.8444 | 0.5332 | 0.5469 |
| Medium | 0.1833 | 0.3278 | 0.1833 | 0.2428 |
| Transformer-M | 0.5722 | 0.8556 | 0.5722 | 0.5575 |
| Large | 0.4889 | 0.8111 | 0.4822 | 0.5197 |

### 2.2 B：A + 固定 128-D 视觉上下文（512-D MLP 输入）

```text
128-D frozen VisualProjector context
+ 384-D A state
= 512-D MLP input

+ 20-D availability action mask
= 532-D decision state
```

- 使用现有 Stage-2k `VisualProjector`（在 Train-600 natural states 上拟合，
  `stage2k/visual_pca.npz`，参数冻结，不重新拟合）。
- **不增加 GAB、热图或其他新模块**。投影前的 summary 中，GAB(20) 与
  map(22) 两块被替换为 projector 的训练均值（归一化后贡献为 0），因此
  128-D 上下文只含 current/goal/history 视觉 token 池化信息。
- B 的 384-D 部分与 A **逐字段相同**，只是前面拼接 128 维视觉上下文。

### 2.3 网络

两个变体都是标准 MLP DQN：

- A：`384 → 256 → 256 → 20`，ReLU；
- B：`512 → 256 → 256 → 20`，ReLU。

隐藏层宽度、深度、优化器、Replay Buffer、Target Network、动作空间完全一致；
只有输入维度因状态不同而不同。没有 Attention / GNN / Transformer / Double DQN。

---

## 3. 训练协议（A/B 完全一致）

| 项 | 设置 |
|---|---|
| Train 任务清单 | Stage-2k Train split 4000 条中分层随机抽 3500 条（1167/1167/1166） |
| 随机顺序 | `SEED + 20261031`，全局 shuffle；A/B 使用同一份 `order` 数组 |
| 训练拓扑 | Train topology（8,000,000+） |
| Dev Validation | 固定 60 个 Dev Episodes（每场景 6 Short / 10 Medium / 4 Long，seeds 6,000,000+） |
| 验证拓扑 | Dev topology（8,100,000+），训练与验证不共享拓扑 |
| 种子 | DQN 初始化、采样、Replay、ε 全部使用 `SEED=20261031` |
| Episodes | **3500**，不因奖励下降早停 |
| 中止条件 | 仅 NaN/Inf、Q 明显发散（`|Q| > 1e5`）或程序错误 |
| ε | 1.0 → 0.05，**500,000 环境交互步** |
| γ | 1.0 |
| 学习率 | 1e-4 |
| Batch | 128 |
| Replay | 50,000 |
| Warmup | 2,000 环境交互步 |
| Target 更新 | 每 1,000 次梯度更新 |
| 梯度裁剪 | 10.0 |
| 验证频率 | 每 200 Episodes + 最后 Ep3500 |
| Checkpoint | 每次验证保存 `last.pt`；Dev JointR 改善 ≥0.002 时保存 `best.pt`；`last_full.pt` 含 Replay/RNG/Optimizer |

### 3.1 奖励（与 JointR 严格一致）

```text
r_nav_t = Phi_t - Phi_{t-1},  Phi_0 = 0
mean_t  = mean_{t-1} + (d_t - mean_{t-1}) / t
r_delay_t = -0.25 / 100 * (mean_t - mean_{t-1})
r_t = r_nav_t + r_delay_t
sum_t r_t = final_nDTW - 0.25 * mean_E2E / 100 = JointR_i
```

不增加成功终止奖励、失败惩罚、目标距离奖励、Oracle 奖励或其他特权奖励。
训练循环逐 Episode 断言 `|Return − JointR| < 1e-6`。

### 3.2 记录

TensorBoard（每个变体独立目录）记录：

- `train/episode_return`（原始 Episode Return）
- `train/episode_return_ma100`（真实 100-Episode 移动平均）
- `train/loss`（TD Loss）、`train/grad_norm`
- `train/epsilon`、`train/q_mean`、`train/q_max`
- `validation/joint_reward`、`validation/SR`、`validation/OSR`、
  `validation/SPL`、`validation/nDTW`、`validation/E2E_ms`、
  `validation/P95_ms`、`validation/mean_steps`
- `validation/model_usage/{MEC}` 与 `validation/model_usage_ratio/{MEC}`

所有曲线只使用真实记录点，不插值、不补点。

---

## 4. 预注册收敛标准

正式训练达到 3500 Episodes 后，使用**固定 60 个 Dev Episodes** 的
Validation JointR 判定：

1. 完整训练到 Ep3500（`reached_cap=true`）；
2. 全程无 NaN/Inf、无 Q 发散、无程序错误；
3. 至少 5 次 Dev Validation（本协议实际 18 次：200…3400 + 3500）；
4. **稳定平台**：最近 3 次 Validation JointR 的极差 ≤ **0.01**；
5. 最后 JointR ≥ 全局最佳 JointR − **0.005**；
6. 最近一次没有形成新的实质最佳（`improving=false`，新最佳阈值 0.002）。

同时满足 1–6 才记 `converged=true`。训练 return 的峰值、100-MA 的峰值、
早停或“奖励不再上升”都**不是**收敛证据。

---

## 5. Final Test 门槛（预注册）

只有同时满足以下两条，才允许读取一次性的 Association Final Test 180：

- **G1 收敛**：第 4 节的 `converged=true`；
- **G2 竞争力**：最佳 Dev60 JointR **严格大于**同一批 60 个 Dev Episodes 上
  最强非特权 Baseline 的 JointR。当前冻结值为 **Min-Latency = 0.2828**
  （Random 0.1931、Max-Quality 0.2091、Heuristic 0.2806；
  Oracle 0.4425 为特权上界，不参与门槛）。

不满足任一条：**保留 Final Test，不读取、不评估**，报告
`not_converged` 或 `converged_but_below_baseline`。两个变体各自独立判定；
不因另一个变体通过而放行。

---

## 6. 复现命令

```bash
cd /home/ubuntu5/Workspace/MEC-V8.2

# 能力先验（只读 Stage-2j Dev Val 180）
python v83_spatial_stage2/stage2k_build_capability_prior_v2.py

# 状态/协议测试
python -m unittest -v v83_spatial_stage2.test_stage2k_pure_mec_v2
python -m unittest -v v83_spatial_stage2.test_stage2k_v2_protocol

# 4-Episode smoke（A）
python -u v83_spatial_stage2/stage2k_jointr_v2_train.py \
  --state-mode pure_mec_v2 --order-mode random \
  --train-topology-split train --smoke-episodes 4 \
  --epsilon-decay-steps 500000 --disable-early-stop \
  --run-dir runs/v83_spatial_stage2/stage2k_pure_mec_v2_smoke \
  --tensorboard-dir runs/tensorboard/association_stage2k_pure_mec_v2_smoke \
  --device cuda

# 4-Episode smoke（B）
python -u v83_spatial_stage2/stage2k_jointr_v2_train.py \
  --state-mode visual_ablation_v2 --order-mode random \
  --train-topology-split train --smoke-episodes 4 \
  --epsilon-decay-steps 500000 --disable-early-stop \
  --run-dir runs/v83_spatial_stage2/stage2k_visual_ablation_v2_smoke \
  --tensorboard-dir runs/tensorboard/association_stage2k_visual_ablation_v2_smoke \
  --device cuda

# 正式训练 A
python -u v83_spatial_stage2/stage2k_jointr_v2_train.py \
  --state-mode pure_mec_v2 --order-mode random \
  --train-topology-split train --max-episodes 3500 \
  --epsilon-decay-steps 500000 --disable-early-stop \
  --run-dir runs/v83_spatial_stage2/stage2k_pure_mec_v2 \
  --tensorboard-dir runs/tensorboard/association_stage2k_pure_mec_v2 \
  --device cuda

# 正式训练 B
python -u v83_spatial_stage2/stage2k_jointr_v2_train.py \
  --state-mode visual_ablation_v2 --order-mode random \
  --train-topology-split train --max-episodes 3500 \
  --epsilon-decay-steps 500000 --disable-early-stop \
  --run-dir runs/v83_spatial_stage2/stage2k_visual_ablation_v2 \
  --tensorboard-dir runs/tensorboard/association_stage2k_visual_ablation_v2 \
  --device cuda

# Dev180 best/last（训练完成后；不读 Final Test）
python -u v83_spatial_stage2/stage2k_jointr_v2_eval.py \
  --state-mode pure_mec_v2 \
  --run-dir runs/v83_spatial_stage2/stage2k_pure_mec_v2 \
  --checkpoint runs/v83_spatial_stage2/stage2k_pure_mec_v2/checkpoints/best.pt \
  --output-dir runs/v83_spatial_stage2/stage2k_pure_mec_v2/dev180_best --device cuda
python -u v83_spatial_stage2/stage2k_jointr_v2_eval.py \
  --state-mode visual_ablation_v2 \
  --run-dir runs/v83_spatial_stage2/stage2k_visual_ablation_v2 \
  --checkpoint runs/v83_spatial_stage2/stage2k_visual_ablation_v2/checkpoints/best.pt \
  --output-dir runs/v83_spatial_stage2/stage2k_visual_ablation_v2/dev180_best --device cuda
# last.pt 同理，换 --checkpoint 与 --output-dir
```

---

## 7. 产物

- 源码：`stage2k_pure_mec_v2_state.py`、`stage2k_v2_common.py`、
  `stage2k_build_capability_prior_v2.py`、runtime/train/eval 的 v2 状态支持。
- 测试：`test_stage2k_pure_mec_v2.py`、`test_stage2k_v2_protocol.py`。
- 预注册：本文件、`stage2k_v2_protocol_freeze.json`。
- 运行目录：`stage2k_pure_mec_v2/`、`stage2k_visual_ablation_v2/`；
  大型权重、Replay Buffer、TensorBoard Event 与逐 Episode CSV 保留在
  `runs/` 下，不提交 Git。
- 报告与曲线：`STAGE2K_PURE_MEC_V2_REPORT.md`、
  `STAGE2K_VISUAL_ABLATION_V2_REPORT.md`、
  `STAGE2K_V2_ABLATION_COMPARISON.md` 与 `figures/stage2k_*_v2/`。
