# Stage-2n 推理时延校准报告

- 时间：2026-10-23T01:06:11Z
- 对象：Stage-2j `Small` 与 `Transformer-M`，原始最佳检查点，未重训
- 输入：Stage-2j 真实训练样本（`train_600/shard_0000.pt` + `aux/train/aux_shard_0000.npz`，index 0）
- 精度：FP32 权重与 FP32 输入；batch size = 1；`model.eval()` + `torch.inference_mode()`
- 预热：每轮 200 次；3 轮独立测量，每轮 1000 次前向
- 本轮只做时延校准：未训练、未读取封存 Test、未修改既有模型/奖励/任务池/训练结果

## 0. 结论摘要

- **Small**：GPU 前向 **0.4943 ms**（P95 0.5316 ms），请求到结果 **0.5422 ms**（P95 0.5635 ms）；2 TFLOPS 解析推理 0.007113 ms，解析/实测偏差 **-98.6%**。
- **Transformer-M**：GPU 前向 **0.9159 ms**（P95 0.9479 ms），请求到结果 **1.0049 ms**（P95 1.0294 ms）；2 TFLOPS 解析推理 1.407298 ms，解析/实测偏差 **+53.7%**。

**五个问题的直接回答**

1. **实际推理时延**：见上；Small 约 0.494 ms，Transformer-M 约 0.916 ms（GPU 前向，均值）。
2. **差异是否值得做异质关联**：两个模型 GPU 前向差约 0.422 ms，在 49.11 ms 历史 E2E 参考中约占 0.9%；若模型差异只体现在这一量级，异质关联的时延收益有限，需要与通信/算力差异一起评估。
3. **解析模型偏差**：Small 的 2 TFLOPS 解析值严重低估实际调用成本（启动/调度开销主导）；Transformer-M 的解析值偏高，因为实测有效吞吐高于 2 TFLOPS。具体百分比见第 5 节。
4. **推理占 E2E 的比例**：推理只占历史 49.11 ms 参考的很小部分；提取（共享视觉编码器约 22.7 ms 解析）和通信（150 m 约 21.1 ms 解析）才是主要项。**优先优化通信与特征提取，而不是 MEC 前向。**
5. **是否用于下一轮仿真校准**：建议新增独立校准 profile，不要覆盖 `stage2k_latency.py` 的原始 2 TFLOPS 模型；在本服务器 GPU 上使用实测值时，必须标注这不是 UAV/MEC 目标硬件，并保留启动开销下限。详见第 8 节。

## 1. 硬件与软件

- GPU：`NVIDIA GeForce RTX 5090`，compute capability 12.0，170 SM，33.7 GB
- PyTorch：`2.11.0+cu128`；CUDA runtime：`12.8`；驱动：`580.178.04`（CUDA 13.0）；cuDNN：`91900`
- 测量开始快照：SM clock 2542 MHz（max 3105 MHz）、功耗 56.28 W（limit 500 W）、温度 32°C、GPU util 0%；每轮前后快照见 `latency_calibration.json`
- 现有进程：两个 Stage-2l B run 处于暂停态，各约 2.95 GB；未停止、未修改；测量时另有一个短暂约 500 MB 进程，测量结束后已退出
- TF32 matmul：False；float32_matmul_precision：`highest`
- **重要限制**：这是服务器 RTX 5090，不是实际 UAV/MEC 目标硬件；结果不能直接等同于目标硬件性能。

## 2. 检查点与模型

| 模型 | 参数量 | FLOPs (GFLOP) | 检查点 SHA256 | 配置 |
|---|---:|---:|---|---|
| Small | 611,716 | 0.014226 | `7349ec9d0c809a09…` | `runs/v83_spatial_stage2/stage2j/Small/config.json` |
| Transformer-M | 3,033,732 | 2.814596 | `57c88b7d71e1a7f1…` | `runs/v83_spatial_stage2/stage2j/Transformer-M/config.json` |

## 3. 测量方法

- **纯模型前向（分组 CUDA Event）**：输入已在 GPU；100 组 ×10 次连续前向，事件包住每组，除以 10；用于主均值/吞吐。
- **纯模型前向（逐次 CUDA Event）**：1000 个独立事件对，用于 P50/P90/P95/P99 分布；包含单次调用间的 GPU 空闲/启动间隙。
- **请求到结果**：`perf_counter` 包住前向 + sigmoid/argmax 结果取回 + `cuda.synchronize()`；1000 次/轮；不包含网络上行。
- **模块分项**：直接调用 `history_encoder`、`move_head`/`move_classifier`、`arrival_head`，以及 Transformer-M 的`fusion_encoder`；分组 CUDA Event；**分项不可相加**，因为投影、token 拼接、goal-current attention 和残差 glue 未包含。
- 每个模型每轮先预热 200 次；3 轮独立测量。

## 4. 测量结果（ms）

### 4.1 总体（3 轮合并；CV 为三轮均值的变异系数）

| 模型 | 指标 | Mean | P50 | P90 | P95 | P99 | Std | CV(轮均值) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small | `forward_cuda_grouped` | 0.4943 | 0.4966 | 0.5067 | 0.5107 | 0.5213 | 0.0119 | 2.31% |
| Small | `forward_cuda_percall` | 0.5148 | 0.5100 | 0.5259 | 0.5316 | 0.5822 | 0.0752 | 1.96% |
| Small | `request_to_result` | 0.5422 | 0.5422 | 0.5580 | 0.5635 | 0.5820 | 0.0575 | 2.16% |
| Transformer-M | `forward_cuda_grouped` | 0.9159 | 0.9145 | 0.9260 | 0.9302 | 0.9469 | 0.0100 | 0.76% |
| Transformer-M | `forward_cuda_percall` | 0.9303 | 0.9223 | 0.9402 | 0.9479 | 1.0103 | 0.1089 | 0.48% |
| Transformer-M | `request_to_result` | 1.0049 | 1.0060 | 1.0243 | 1.0294 | 1.0483 | 0.0176 | 1.10% |

说明：逐次 CUDA Event 的少量离群点（Small 逐次最大约 2.60 ms、请求最大约 3.57 ms；Transformer-M 逐次最大约 4.31 ms）会抬高 mean/std；P50/P95 更能代表稳定单次时延。主均值使用分组计时（0.4943 / 0.9159 ms），它降低了事件/启动开销的影响。

### 4.2 逐轮均值

| 模型 | 轮次 | GPU 前向（分组） | 请求到结果 |
|---|---:|---:|---:|
| Small | 1 | 0.4820 | 0.5289 |
| Small | 2 | 0.5046 | 0.5510 |
| Small | 3 | 0.4962 | 0.5466 |
| Transformer-M | 1 | 0.9126 | 0.9997 |
| Transformer-M | 2 | 0.9111 | 0.9976 |
| Transformer-M | 3 | 0.9239 | 1.0176 |

### 4.3 吞吐率

| 模型 | 排队前向吞吐（次/秒） | 前向耗时（s） |
|---|---:|---:|
| Small | 2026.3 | 0.494 |
| Transformer-M | 1103.2 | 0.906 |

### 4.4 复现性检查（两次独立完整运行）

| 模型 | 指标 | Run1 mean (ms) | Run2 mean (ms) | Δ | Δ% |
|---|---|---:|---:|---:|---:|
| Small | `forward_cuda_grouped` | 0.5239 | 0.4943 | -0.0296 | -5.66% |
| Small | `forward_cuda_percall` | 0.5430 | 0.5148 | -0.0282 | -5.19% |
| Small | `request_to_result` | 0.5690 | 0.5422 | -0.0268 | -4.71% |
| Transformer-M | `forward_cuda_grouped` | 0.9263 | 0.9159 | -0.0105 | -1.13% |
| Transformer-M | `forward_cuda_percall` | 0.9445 | 0.9303 | -0.0141 | -1.49% |
| Transformer-M | `request_to_result` | 1.0208 | 1.0049 | -0.0158 | -1.55% |

两次完整运行的差异：Small 前向约 −5.7%（启动/调度主导，run-to-run 漂移较大），Transformer-M 前向约 −1.1%；单次运行内 3 轮 CV 分别为 2.31% 与 0.76%。因此 Small 的绝对值应按约 ±6% 解读，Transformer-M 约 ±2%；两者的量级结论不变。


## 5. 解析模型对比

解析值来自现有 `Stage2KLatency`，假设 2 TFLOPS；通信/提取为解析值，参考距离 150 m。

| 模型 | 解析推理 (ms) | 实测 GPU 前向 (ms) | 请求→结果 (ms) | 前向 P95 (ms) | 解析/前向偏差 | 实测有效 TFLOP/s |
|---|---:|---:|---:|---:|---:|---:|
| Small | 0.007113 | 0.4943 | 0.5422 | 0.5316 | -98.6% | 0.03 |
| Transformer-M | 1.407298 | 0.9159 | 1.0049 | 0.9479 | +53.7% | 3.07 |

### 5.1 与历史 49.11 ms 的比例（仅量级参考）

| 模型 | 解析推理占比 | 实测前向占比 | 请求占比 | 解析提取占比 | 解析通信占比（150 m） |
|---|---:|---:|---:|---:|---:|
| Small | 0.014% | 1.006% | 1.104% | 46.25% | 42.91% |
| Transformer-M | 2.866% | 1.865% | 2.046% | 46.25% | 42.91% |

**注意**：49.11 ms 是上一轮 Greedy-Min-E2E 的历史仿真值，不是本轮异质部署的实测 E2E；这些比例只用于量级判断。

## 6. 从解析到完整 MEC 的分层

| 项目 | Small | Transformer-M | 说明 |
|---|---:|---:|---|
| 纯推理（实测 GPU 前向） | 0.4943 | 0.9159 | 实测 |
| 请求到结果（本地，无网络） | 0.5422 | 1.0049 | 实测 |
| 共享视觉编码器提取（解析） | 22.715 | 22.715 | 解析，未实测 |
| 150 m 上行通信（解析） | 21.073 | 21.073 | 解析，未实测 |

**完整 MEC 处理**：解析提取 + 解析通信 + 实测前向 = Small 44.283 ms，Transformer-M 44.704 ms （150 m 假设；这是解析/实测混合值，不是端到端实测）。

## 7. 模块分项（诊断，不可相加）

| 模型 | 模块 | Mean (ms) | P95 (ms) |
|---|---|---:|---:|
| Small | `module_history_encoder` | 0.2198 | 0.2244 |
| Small | `module_move_head` | 0.0192 | 0.0198 |
| Small | `module_arrival_head` | 0.1739 | 0.1785 |
| Transformer-M | `module_history_encoder` | 0.2235 | 0.2264 |
| Transformer-M | `module_move_head` | 0.0068 | 0.0069 |
| Transformer-M | `module_arrival_head` | 0.1799 | 0.1832 |
| Transformer-M | `module_fusion_encoder` | 0.3483 | 0.3493 |

分项之和通常小于完整 forward，因为视觉投影、goal-current attention、token 拼接、LayerNorm/残差等未包含在单独模块内；**不要相加后宣称等于完整 forward**。

## 8. 校准建议与影响分析

1. **不要覆盖原始解析模型**。`stage2k_latency.py` 的 2 TFLOPS FLOPs 模型是已有 Stage-2n 结果的一部分；直接修改会让旧结果不可复现。
2. **新增独立 profile**：例如 `measured_rtx5090_fp32_v1`，字段为 `forward_ms_small=…`、`forward_ms_transformer_m=…`、`request_overhead_ms=…`。保留 `analytical_2tflops` 作为默认/对照。
3. **加入启动开销下限**：Small 的解析值只有 0.007113 ms，而实测前向 0.4943 ms；任何校准都必须包含约 0.4–0.5 ms 的启动/调度地板，否则会严重低估。
4. **影响量级**：Transformer-M 的解析推理比实测高约 53.7%；若把其解析推理替换为实测值，150 m 模拟完整 MEC 约变化 -0.491 ms，相对 49.11 ms 约 1.00%。因此推理校准不会改变“通信/提取主导”的结论。
5. **不等同目标硬件**：RTX 5090 是服务器 GPU；UAV/MEC 上的 CPU、功耗、散热、驱动和并发都会改变结果。下一轮异质实验若使用实测值，应做硬件缩放/敏感性分析，而不是把 RTX 5090 数字当作 UAV 实测。

## 9. 复现命令与产物

```bash
python v83_spatial_stage2/stage2n_latency_calibrate.py
python v83_spatial_stage2/test_stage2n_latency_calibrate.py
```

- `latency_calibration_report.md`（本文件）
- `latency_measurements.csv`（逐轮、逐模型统计）
- `latency_raw_samples.csv`（原始逐次/逐组样本）
- `latency_summary.csv`（总体统计与 CV）
- `latency_calibration.json`（硬件、检查点、输入、参数与全部结果）
- `latency_reproducibility.json`（两次独立完整运行的对比）
- `figures/latency_comparison.png`（300 dpi）
- `figures/latency_module_breakdown.png`（300 dpi）
- 原始 checkpoint 与训练日志保留在 `runs/`，未改动

## 10. 限制

- 单个真实训练样本；时延对输入数值不敏感，但样本多样性未覆盖。
- 只测 batch 1、FP32；未测 FP16/TensorRT/多并发。
- Small 为启动/调度主导，两次完整运行相差约 −5.7%；其绝对值应按约 ±6% 解读。
- 逐次 CUDA Event 分布包含少量主机/调度离群点；主均值采用分组计时。
- 服务器 RTX 5090，不是 UAV/MEC 目标硬件。
- 网络上行/提取为解析值，未做真实网络或 UAV 端测量。
- 本轮不读取封存 Test-400，不训练 DQN/PPO。
