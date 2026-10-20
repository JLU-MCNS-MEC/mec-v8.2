# Stage-2k No-Visual DQN 实验协议

日期：2026-10-20  
分支：`experiment/v83-spatial-foundation`  
仓库：`JLU-MCNS-MEC/mec-v8.2`

## 1. 目的

本轮只修改 DQN 的状态空间：彻底移除 RGB/视觉 PCA、Goal token、历史视觉、
热图、GAB、地图检索特征及其派生特征；保留 MEC 选择、推理能力、通信链路
与时延信息。其他实验设计沿用上一轮 JointR-aligned v2 方案。

历史实验（`runs/v83_spatial_stage2/stage2k/`、`stage2k_jointr_v2/`）
的权重与日志全部保留，不覆盖。

## 2. 状态空间

```text
2-D normalized UAV position
+ 20 candidates × 17-D MEC features
= 342
```

每个候选 MEC 的 17 维：

| # | 字段 | 说明 | 归一化 |
|---:|---|---|---|
| 0 | `availability` | 是否在覆盖半径内 | 0/1 |
| 1 | `relative_x` | MEC x − UAV x | / 500 m |
| 2 | `relative_y` | MEC y − UAV y | / 500 m |
| 3 | `distance` | UAV–MEC 距离 | / 500 m |
| 4 | `compute_norm` | 节点算力 | / 2 TFLOPs |
| 5 | `bandwidth_norm` | 节点带宽 | / 30 MHz |
| 6 | `uplink_rate_norm` | 上行速率 | / 200 Mbps |
| 7 | `communication_ms_norm` | 通信时延 | / 100 ms |
| 8 | `inference_ms_norm` | 实际估计推理时延 | / 10 ms，上限 3 |
| 9 | `e2e_ms_norm` | 通信 + 推理 + 统一模型 E2E | / 100 ms，上限 3 |
| 10 | `dev_val_quality_prior` | Stage-2j Dev Val mean nDTW | 原值 |
| 11 | `snr_db_norm` | SNR | / 40 dB，裁剪 [-1,1] |
| 12–16 | `model_*` | 五种 MEC 模型类型 one-hot | 0/1 |

明确排除：视觉/PCA、Goal 特征、历史视觉、热图、GAB、地图检索、导航动作、
Episode 步数、目标位置、目标距离、Oracle action、成功标签。

网络保持标准 DQN MLP：`342 → 256 → 256 → 20`，无 Attention / GNN /
Transformer。

## 3. 冻结组件

- 五种 Stage-2j T5 MEC：`runs/v83_spatial_stage2/stage2j/{model}/best.pt`，
  eval 模式、`requires_grad=False`，不重新训练。
- 三种部署场景：Balanced、Quality-rich、Tradeoff-stress。
- 统一时延模型：`stage2k_latency.Stage2KLatency`。
- 五种 Baseline：Random、Min-Latency、Max-Quality、Heuristic、Oracle。
- 奖励：`r_nav_t + (-0.25/100) × (running_mean_E2E_t −
  running_mean_E2E_{t−1})`，逐 Episode 累计严格等于
  `nDTW − 0.25 × mean_E2E / 100 = JointR`。

## 4. 训练与验证

- DQN 从头初始化，不复用旧权重。
- 最长 4000 Episodes。
- 每 200 Episodes 在固定的 60 个独立 Dev Episodes 上评估
  （每场景 6 Short / 10 Medium / 4 Long）。
- 训练 Episode 顺序为 3 场景 × 3 带宽的确定性轮转，Short/Medium/Long 交错。
- 早停只看 Validation JointR：至少 800 Episodes；连续 5 次无新最佳且最近
  3 次趋势平坦/下降，或连续 8 次无新最佳时停止。
- TensorBoard：`runs/tensorboard/association_stage2k_no_visual/`。

## 5. 最终测试

Association Final Test 180（3 场景 × 60）独立于 Train/Dev，且在训练完成
并写出 `training_result.json` 后才读取。最终测试脚本在第一次 rollout 前写入
`eval_freeze.json`，固化 DQN Checkpoint、五种 MEC Checkpoint、源码、测试
specs、拓扑和质量先验的 SHA256；之后拒绝任何不一致的运行。no-visual 路径不加载 `VisualProjector`，
也不把视觉 PCA 作为 DQN 状态输入。

六种策略在同一批 180 Episode、同一拓扑、同一场景和同一时延模型上评估：
no-visual DQN + 五种 Baseline。Stage-2j Final Test 300 token 目录从未打开。

输出目录：`runs/v83_spatial_stage2/stage2k_no_visual/final_test/`。

## 6. 复现命令

```bash
cd /home/ubuntu5/Workspace/MEC-V8.2

# 状态空间测试
python -m unittest -v v83_spatial_stage2.test_stage2k_no_visual

# 4-Episode smoke
python -u v83_spatial_stage2/stage2k_jointr_v2_train.py \
  --state-mode no_visual \
  --run-dir runs/v83_spatial_stage2/stage2k_no_visual_smoke \
  --tensorboard-dir runs/tensorboard/association_stage2k_no_visual_smoke \
  --smoke-episodes 4 --device cuda

# 正式训练
python -u v83_spatial_stage2/stage2k_jointr_v2_train.py \
  --state-mode no_visual \
  --run-dir runs/v83_spatial_stage2/stage2k_no_visual \
  --tensorboard-dir runs/tensorboard/association_stage2k_no_visual \
  --device cuda

# 最终测试脚本自检（只读 Dev，3 个 Episode，不写 Final Test）
python -u v83_spatial_stage2/stage2k_no_visual_test.py \
  --split dev --max-episodes 3 \
  --output-dir runs/v83_spatial_stage2/stage2k_no_visual_script_smoke \
  --device cuda

# 最终测试（训练完成后，一次性读取 Association Final Test 180）
python -u v83_spatial_stage2/stage2k_no_visual_test.py \
  --run-dir runs/v83_spatial_stage2/stage2k_no_visual --device cuda

# 训练曲线
python v83_spatial_stage2/stage2k_jointr_v2_plot.py \
  --run-dir runs/v83_spatial_stage2/stage2k_no_visual \
  --output-dir v83_spatial_stage2/figures/stage2k_no_visual \
  --prefix dqn

# 中文报告
python v83_spatial_stage2/stage2k_no_visual_report.py
```

## 7. 产物

- 源码：`stage2k_no_visual_state.py`、runtime/train/eval 的 `state_mode` 支持、
  `stage2k_no_visual_test.py`、`stage2k_no_visual_report.py`。
- 测试：`test_stage2k_no_visual.py`。
- 报告与图：`STAGE2K_NO_VISUAL_STATE_REPORT.md`、
  `figures/stage2k_no_visual/`。
- 大型权重、Replay Buffer、TensorBoard Event 和逐 Episode CSV 保留在
  服务器，不提交版本库。
