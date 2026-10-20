# Stage-2k Pure MEC DQN 实验协议

日期：2026-10-20  
分支：`experiment/v83-spatial-foundation`  
仓库：`JLU-MCNS-MEC/mec-v8.2`

## 1. 目的

在保留 `stage2k_no_visual` 全部产物的前提下，新建独立实验
`stage2k_pure_mec_state`：从头训练一个只使用 MEC 选择、推理能力和
通信链路信息的 DQN。五种 T5 MEC 保持冻结，不重新训练；Baseline、
三种部署场景、统一时延模型和 JointR 奖励保持不变。

本实验同时修正上一轮审计发现的两个训练协议问题：

1. 训练任务从 Train split **分层随机抽样并全局打乱**，不再使用确定性
   scenario × band 轮转；
2. 训练 rollout 使用 **Train topology**，Dev topology 只用于验证，
   Final Test topology 不参与训练。

## 2. 状态空间

```text
2-D normalized UAV position
+ 20 candidates × 16-D MEC features
= 322-D DQN input

+ 20-D availability action mask (separate from MLP input)
= 342-D decision state
```

每个候选 MEC 的 16 维：

| # | 字段 | 说明 | 归一化 |
|---:|---|---|---|
| 0 | `relative_x` | MEC x − UAV x | / 500 m |
| 1 | `relative_y` | MEC y − UAV y | / 500 m |
| 2 | `distance` | UAV–MEC 距离 | / 500 m |
| 3 | `compute_norm` | 节点算力 | / 2 TFLOPs |
| 4 | `bandwidth_norm` | 节点带宽 | / 30 MHz |
| 5 | `uplink_rate_norm` | 上行速率 | / 200 Mbps |
| 6 | `communication_ms_norm` | 通信时延 | / 100 ms |
| 7 | `inference_ms_norm` | 实际估计推理时延 | / 10 ms，上限 3 |
| 8 | `e2e_ms_norm` | 通信 + 推理 + 统一模型 E2E | / 100 ms，上限 3 |
| 9 | `dev_val_quality_prior` | Stage-2j Dev Val mean nDTW | 原值 |
| 10 | `snr_db_norm` | SNR（信道状态） | / 40 dB，裁剪 [-1,1] |
| 11–15 | `model_*` | 五种 MEC 模型类型 one-hot | 0/1 |

**可用性 Mask**：单独的 20 维 0/1 向量，用于屏蔽不可用 MEC；它不重复
进入 MLP 输入，避免与每候选 availability 特征冗余。

明确排除：视觉/PCA、Goal token、目标视觉相似度、历史视觉、热图、GAB、
地图检索、导航动作、Episode 步数、目标位置、目标距离、Oracle action、
成功标签。

网络保持标准 DQN MLP：`322 → 256 → 256 → 20`，无 Attention / GNN /
Transformer。

## 3. 训练任务抽样

- 抽样框：Stage-2k Train split 4000 条（seed 5,000,000+）。
- 样本量：**3500**（与训练上限一致）。
- 目标比例：Short / Medium / Long 按原 MEC Train 600 的 1/3 比例；
  3500 条取整数为 **1167 / 1167 / 1166**。
- 分层：3 场景 × 3 band，共 9 组；前 8 组各 389 条，最后一组 388 条，
  保证 band 总数 1167/1167/1166、场景总数 1167/1167/1166。
- 每组内不放回随机抽样，然后对 3500 条做全局随机打乱。
- 随机种子：`SEED + 20261031`（`SEED=20261031`），可复现。
- 训练顺序写入 `training_order.json`；不以确定性轮转替代随机顺序。

## 4. 训练与验证

- DQN 从头初始化，不复用旧 DQN 权重。
- 最长 **3500 Episodes**；每 200 Episodes 在固定 60 个独立 Dev
  Episodes 上评估（每场景 6 Short / 10 Medium / 4 Long）。
- 训练 rollout：Train topology（8,000,000+）。
- 验证 rollout：Dev topology（8,100,000+）。
- Final Test topology（8,200,000+）不参与训练或验证。
- 奖励：`r_nav_t + (-0.25/100) × (running_mean_E2E_t −
  running_mean_E2E_{t−1})`，逐 Episode 累计严格等于 JointR。
- 不增加成功终止奖励、失败惩罚、目标距离奖励或 Oracle 奖励。
- 早停：至少 800 Episodes；连续 5 次无新最佳且最近 3 次趋势平坦/下降，
  或连续 8 次无新最佳时停止。单次 Validation 波动不触发早停。
- TensorBoard：`runs/tensorboard/association_stage2k_pure_mec_state/`，
  记录 raw reward、100-Episode MA、Validation JointR/SR/nDTW/E2E、
  loss、Q mean/max。

## 5. 收敛与 Final Test

- 只有 `training_result.json` 中 `converged=true` 才允许进入正式
  Association Final Test 180；未收敛时保留测试集，不消耗一次性评估。
- Final Test 使用最佳验证 Checkpoint；六种策略（DQN + Random、
  Min-Latency、Max-Quality、Heuristic、Oracle）在同一批 180 Episode、
  同一 Test 拓扑和同一时延模型上评估。
- 第一次 rollout 前写入 `eval_freeze.json`，固化 DQN/MEC Checkpoint、
  源码、测试 specs、Test 拓扑和质量先验的 SHA256。
- 输出目录：`runs/v83_spatial_stage2/stage2k_pure_mec_state/final_test/`。

## 6. 与 no-visual 的对照

| 项 | stage2k_no_visual | stage2k_pure_mec_state |
|---|---|---|
| DQN 输入 | 342（2 + 20×17） | **322（2 + 20×16）** |
| 可用性 | 每候选 1 维 | **独立 20 维 Mask** |
| 决策状态 | 342 | 342（322 + 20） |
| 训练任务 | 确定性 3×3 轮转，4000 | **分层随机抽样，3500** |
| 训练拓扑 | Dev | **Train** |
| 验证拓扑 | Dev | Dev |
| Final Test 拓扑 | Test | Test |

两者字段高度重叠，差异同时包含状态编码和训练协议；对照报告会明确
标注，不把差异单独归因于状态维度。

## 7. 复现命令

```bash
cd /home/ubuntu5/Workspace/MEC-V8.2

# 纯 MEC CPU 测试
python -m unittest -v v83_spatial_stage2.test_stage2k_pure_mec

# 4-Episode smoke
python -u v83_spatial_stage2/stage2k_jointr_v2_train.py \
  --state-mode pure_mec --order-mode random \
  --train-topology-split train \
  --run-dir runs/v83_spatial_stage2/stage2k_pure_mec_state_smoke \
  --tensorboard-dir runs/tensorboard/association_stage2k_pure_mec_state_smoke \
  --smoke-episodes 4 --max-episodes 3500 --device cuda

# 正式训练
python -u v83_spatial_stage2/stage2k_jointr_v2_train.py \
  --state-mode pure_mec --order-mode random \
  --train-topology-split train \
  --run-dir runs/v83_spatial_stage2/stage2k_pure_mec_state \
  --tensorboard-dir runs/tensorboard/association_stage2k_pure_mec_state \
  --max-episodes 3500 --device cuda

# Final Test（仅 converged=true 时）
python -u v83_spatial_stage2/stage2k_no_visual_test.py \
  --state-mode pure_mec \
  --run-dir runs/v83_spatial_stage2/stage2k_pure_mec_state \
  --device cuda

# 曲线
python v83_spatial_stage2/stage2k_jointr_v2_plot.py \
  --run-dir runs/v83_spatial_stage2/stage2k_pure_mec_state \
  --output-dir v83_spatial_stage2/figures/stage2k_pure_mec_state \
  --prefix dqn

# 单实验报告与对照报告
python v83_spatial_stage2/stage2k_no_visual_report.py \
  --state-mode pure_mec \
  --run-dir runs/v83_spatial_stage2/stage2k_pure_mec_state
python v83_spatial_stage2/stage2k_state_comparison_report.py
```

## 8. 产物

- 源码：`stage2k_pure_mec_state.py`、`stage2k_jointr_v2_order.py` 的随机抽样、
  runtime/train/eval 的 `pure_mec` 支持、`stage2k_no_visual_test.py` 的
  `--state-mode pure_mec`。
- 测试：`test_stage2k_pure_mec.py`。
- 报告：`STAGE2K_PURE_MEC_STATE_REPORT.md`、
  `STAGE2K_STATE_DESIGN_COMPARISON.md`。
- 大型权重、Replay Buffer、TensorBoard Event 和逐 Episode CSV 保留在
  服务器 `runs/` 下，不提交版本库。
