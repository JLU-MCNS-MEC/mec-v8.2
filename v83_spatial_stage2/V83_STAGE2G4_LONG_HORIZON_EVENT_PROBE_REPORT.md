# V8.3 Stage-2g.4 — Long-Horizon Target-Loss Event Probe 报告

日期：2026-10-19

分支：`experiment/v83-spatial-foundation`

前序提交：`9787a0f`

本轮性质：**复用 Stage-2g.3 真实闭环数据 + 轻量事件探针训练 + 冻结 Gate 评估**

> 本轮**没有**训练完整 MEC，**没有**接入 Recovery 控制器，**没有**读取
> Final Test 300，**没有**修改 M0/M1 checkpoint，**没有**修改 Stage-2g.3
> 的 Train 600 / Dev Val 180 或历史报告。新采集为 **0 个 Episode**：
> 事件标签覆盖满足预注册门槛，不需要新增 M0@0.7 闭环数据。

---

## 0. TL;DR（中文结论）

1. **长历史确实有效，而且效果很大。** 在独立 Dev Val 上，
   P1-16/P1-16+mem 的 event AUROC 达到 **0.776–0.863**；
   K16 主条件 P1-16+mem 的 AUROC 为 **0.7988**、AUPRC **0.1658**；
   相对 Stage-2g.3 的两帧 P1-2（K16 AUROC 0.6287）提升
   **+0.169 AUROC**（Episode 配对 bootstrap 95% CI
   `[+0.098, +0.246]`）。这证实了 Stage-2g.3 的核心诊断：
   **两帧历史窗口太短。**
2. **但事件触发器仍然不可用。** K16 主条件在 Train-OOF 阈值下：
   **Precision 0.3594、Recall 0.2378、TP 78 / FP 139**；
   Train OOF 上没有任何受限阈值能达到 Precision 0.5；
   Dev Val 上即使**事后扫描所有阈值**（仅诊断，不用于选择），
   最高 F1 也只有 **0.304**，最高 Precision **0.479**，
   不存在 Precision ≥ 0.5 且 Recall ≥ 0.5 的工作点。
3. **失败主要不是“模型不会排序”，而是 Episode 级可靠性。**
   虽然总体 AUROC 0.80，但 **22 个 Dev Val 正样本 Episode 中只有 10 个
   至少命中 1 个正样本；12 个 Episode 完全零命中**，包括事件数最多的
   Episode 95（23 个事件，0 TP）。逐 Episode Recall 中位数为 **0**，
   P75 为 0.531。总体 AUROC 掩盖了“半数以上正样本 Episode 完全漏检”。
4. **假阳性绝对数量仍高，且高度集中。** 主条件 FP 139、TP 78；
   虽然平均每 Episode FP 只有 **0.813**（中位 0），但最大值是
   **53 个 FP（Episode 47）**，Episode 60 有 21 FP，Episode 123
   有 12 FP 且没有任何事件正样本。绝对误触发预算不通过。
5. **last-seen memory 没有实质增益。** 手工可观测 memory 的
   event-vs-sustained age AUROC 约 **0.616**；P1-16+mem 相对 P1-16
   的 K16 ΔAUROC 仅 **+0.0229**，95% CI `[-0.0116, +0.0601]`，
   ΔAUPRC **-0.0026**。memory 在 K4/K8 有小的正向迹象，但不显著，
   不能作为主要贡献。
6. **事件样本大多是“终端丢失”，不是暂时遮挡。** K16 Dev Val 的
   328 个事件中，**321 个在剩余轨迹内再未 localizable**；只有 7 个在
   10 步内重新 localizable。即使事件检测变好，Recovery 也缺少可观测的
   re-acquisition 目标。
7. **明确判断：停止当前的 Recovery 探针路线。** 长时序事件信号值得
   保留为**诊断/空间信念特征**，但不能作为独立 Recovery 触发器。
   下一步应转向**显式的目标空间信念与粗导航定位**，不要继续堆叠 MLP、
   盲目增加历史帧或继续调阈值。

### Gate 结论（K16 主条件，P1-16+mem）

| 检查 | 结果 |
|---|---|
| G1 Train OOF 存在 Precision ≥ 0.5 的受限阈值 | **FAIL**（无） |
| G2 Dev Val Precision ≥ 0.5 且 Recall ≥ 0.5 | **FAIL**（0.359 / 0.238） |
| G3 绝对 FP 预算（FP ≤ TP，FP/Episode ≤ 1） | **FAIL**（FP 139 > TP 78；均值 0.813，最大 53） |
| G4 event-vs-sustained ≥ 0.70，event-vs-far-invisible ≥ 0.75 | **PASS**（0.713 / 0.786） |
| G5 长历史相对 P0/P1-2 有显著增益 | **PASS**（+0.186 / +0.169 AUROC） |
| G6 leave-one-extreme-episode-out 后仍通过 G2/G3 | **FAIL**（去掉 Episode 95 后 P 0.359 / R 0.256） |
| **总 Gate** | **FAIL** |

---

## 1. 数据与事件标签

### 1.1 复用数据（观测）

| 数据 | 路径 | 内容 |
|---|---|---|
| Train 600 | `runs/v83_spatial_stage2/stage2g3/real_train_20261019T132500Z/` | 600 NPZ，81,303 states |
| Dev Val 180 | `runs/v83_spatial_stage2/stage2g3/real_val_20261019T132000Z/` | 180 NPZ，26,628 states |
| 旧探针 | `runs/v83_spatial_stage2/stage2g3/probe_20261019T133500Z/` | 36 个 Stage-2g.3 checkpoint |

共用的探针状态集为 `step >= 16 and action != STOP`：

- Train common：71,576 states / **563** 个 Episode（有 17 步以上的 Episode）。
- Dev Val common：23,665 states / **171** 个 Episode。

Stage-2g.3 的 600/180 个 NPZ 中，部分 Episode 太短（<17 步）无法提供
16 帧历史；所有 5 个变体都在同一个 common 集上训练/评估，保证可比。

### 1.2 事件标签定义（观测）

对 K ∈ {4, 8, 16}：

```text
recent_near_loss_K(t) =
    distance_m(t) < 100
    AND target_visible(t) == 0
    AND exists j in [1, K] with target_localizable(t-j) == 1
```

- 真实距离、可见性、localizability 只用于离线生成标签；
  **没有任何标签列进入模型输入**。
- `since_localizable` 是当前步到同一 Episode 最近一次
  `target_localizable=1` 的步数；事件要求 `1 <= since_localizable <= K`。

### 1.3 覆盖与正例率（统计）

| Split | K | common states | 正样本 | 正例率 | 正样本 Episode | Band（short/med/long） |
|---|---:|---:|---:|---:|---:|---:|
| Train | 4 | 71,576 | 170 | 0.238% | 41 | 78 / 50 / 42 |
| Train | 8 | 71,576 | 320 | 0.447% | 42 | 146 / 96 / 78 |
| Train | 16 | 71,576 | 609 | 0.851% | 42 | 275 / 184 / 150 |
| Dev Val | 4 | 23,665 | 82 | 0.347% | 19 | 59 / 19 / 4 |
| Dev Val | 8 | 23,665 | 163 | 0.689% | 21 | 112 / 43 / 8 |
| Dev Val | 16 | 23,665 | 328 | 1.386% | 22 | 220 / 92 / 16 |

**数据充足性（预注册）：** 新采集门槛是“每个 K 在 Dev Val 至少
10 个正样本 Episode 且 50 个正样本”。K=4/8/16 分别为
19/21/22 个 Episode、82/163/328 个正样本，全部通过；
因此本轮**没有采集新的 M0@0.7 Episode**，也没有用重复轨迹或合成转向
样本扩大事件集。

### 1.4 事件发生前后的最近可靠观测（统计）

`since_localizable`（事件前最近一次 localizable 的间隔）：

| Split | K | mean | 中位 | P25 / P75 / P90 | 最大 |
|---|---:|---:|---:|---:|---:|
| Train | 4 | 2.40 | 2 | 1 / 3 / 4 | 4 |
| Train | 8 | 4.32 | 4 | 2 / 6 / 8 | 8 |
| Train | 16 | 8.20 | 8 | 4 / 12 / 15 | 16 |
| Dev Val | 4 | 2.41 | 2 | 1 / 3 / 4 | 4 |
| Dev Val | 8 | 4.45 | 4 | 2 / 6.5 / 8 | 8 |
| Dev Val | 16 | 8.49 | 9 | 4.75 / 12 / 15 | 16 |

`until_next_localizable`（事件后再次 localizable 的间隔；以下为 common 集）：

| Split | K | 观测到后续 localizable | 中位间隔 | 从未再 localizable |
|---|---:|---:|---:|---:|
| Train | 4 | 19 / 170 | 1 步 | 151 / 170 |
| Train | 8 | 19 / 320 | 1 步 | 301 / 320 |
| Train | 16 | 19 / 609 | 1 步 | 590 / 609 |
| Dev Val | 4 | 7 / 82 | 1 步 | 75 / 82 |
| Dev Val | 8 | 7 / 163 | 1 步 | 156 / 163 |
| Dev Val | 16 | 7 / 328 | 1 步 | **321 / 328** |

**关键观测：这些事件绝大多数是“终端丢失”，不是暂时的目标离开视野。**
即使事件检测完全正确，Recovery 在剩余轨迹内也常常没有可观测的
re-acquisition 目标。这是决定“是否值得继续 Recovery”的重要数据事实。

### 1.5 事件正样本的距离段与 Episode 集中度（统计）

| Split | K | 0–20 m | 20–50 m | 50–100 m |
|---|---:|---:|---:|---:|
| Train | 4 | 95 | 22 | 53 |
| Train | 8 | 141 | 82 | 97 |
| Train | 16 | 146 | 211 | 252 |
| Dev Val | 4 | 24 | 12 | 46 |
| Dev Val | 8 | 37 | 41 | 85 |
| Dev Val | 16 | 37 | 103 | 188 |

Episode 集中度：

| Split | K | 正样本 Episode | 每 Episode 中位 | 最大 | top-1 占比 | top-10 占比 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 4 | 41 | 4 | 10 | 5.9% | 34.1% |
| Train | 16 | 42 | 16 | 22 | 3.6% | 28.7% |
| Dev Val | 4 | 19 | 4 | 11 | 13.4% | 57.3% |
| Dev Val | 16 | 22 | 16 | 23 | 7.0% | 50.9% |

事件标签的 Episode 集中度明显低于 Stage-2g.3 的静态 near-inv 标签
（后者 top-1 占 27.6%），说明事件重定义本身减轻了“一个 Episode
贡献四分之一正样本”的问题。

---

## 2. 预注册协议与 Gate

详见同日提交的 `V83_STAGE2G4_PROTOCOL.md`。要点：

- 输入：
  - P0：当前 2,858 维可观测特征；
  - P1-2/P1-8/P1-16：当前 + t-1…t-H 的 compact 历史（每帧 74 维
    compact + 4 维动作 one-hot + dt）；
  - P1-16+mem：P1-16 + 可观测 last-seen memory。
- 5-fold **Episode-level** OOF；外部模型在全部 Train 上训练；
  Dev Val 只用于评估；阈值只在 Train OOF 上选择。
- 阈值选择约束：near-visible / far-visible / far-invisible 误触发
  ≤ 2%，sustained near-inv ≤ 5%；主工作点为受限 F1 最大点。
- 主条件：K=16，P1-16 与 P1-16+mem 中 Train OOF AUPRC 更高者。
- Gate：G1–G6（见 §0/§7）；任一失败即停止 Recovery 探针路线。
- 不做完整 MEC 训练、不接控制器、不读 Final Test。

---

## 3. 探针结果：总体（统计）

Dev Val common 集：23,665 states / 328 个 K16 事件 / 22 个正样本 Episode。
所有数值来自本轮 `analysis_20261019T072000Z/analysis.json`，且与
`probe_results.json` 的聚合指标逐位一致。

| K | 变体 | AUROC | AUPRC | Precision | Recall | TP / FP |
|---:|---|---:|---:|---:|---:|---:|
| 4 | P0 | 0.6051 | 0.0074 | 0.0149 | 0.1098 | 9 / 596 |
| 4 | P1-2 | 0.6680 | 0.0130 | 0.0303 | 0.0732 | 6 / 192 |
| 4 | P1-8 | 0.7969 | 0.0576 | 0.1754 | 0.1220 | 10 / 47 |
| 4 | P1-16 | 0.8446 | 0.1393 | 0.3036 | 0.2073 | 17 / 39 |
| 4 | P1-16+mem | **0.8628** | 0.1305 | **0.3636** | 0.1951 | 16 / 28 |
| 8 | P0 | 0.5834 | 0.0151 | 0.0341 | 0.0184 | 3 / 85 |
| 8 | P1-2 | 0.6320 | 0.0178 | 0.0215 | 0.0123 | 2 / 91 |
| 8 | P1-8 | 0.7636 | 0.0567 | 0.1104 | 0.1043 | 17 / 137 |
| 8 | P1-16 | 0.8498 | 0.1712 | 0.2176 | 0.2270 | 37 / 133 |
| 8 | P1-16+mem | 0.8548 | 0.1558 | 0.2516 | **0.2454** | 40 / 119 |
| 16 | P0 | 0.6128 | 0.0284 | 0.0156 | 0.0061 | 2 / 126 |
| 16 | P1-2 | 0.6287 | 0.0423 | 0.0725 | 0.0152 | 5 / 64 |
| 16 | P1-8 | 0.6826 | 0.0691 | 0.2000 | 0.0549 | 18 / 72 |
| 16 | P1-16 | 0.7755 | 0.1729 | 0.3022 | **0.2470** | 81 / 187 |
| 16 | P1-16+mem（主） | **0.7988** | 0.1658 | **0.3594** | 0.2378 | 78 / 139 |

**观察：**

- 长历史带来的提升是决定性的：K16 P1-16+mem 相对 P1-2 的
  ΔAUROC = **+0.1694**（95% CI `[+0.0981, +0.2455]`），
  ΔAUPRC = **+0.1324**（`[+0.0544, +0.2484]`）。
  K4/K8 的提升更大（K4 ΔAUROC +0.194，K8 +0.223，均 CI > 0）。
- 但 **Precision 最高只有 0.364**，Recall 最高 0.247（K16 主条件
  0.359 / 0.238），离 Gate 的 0.5/0.5 仍有明显差距。
- K16 的 AUROC 比 K4/K8 低，是因为 K16 包含更早（9–16 步前）的丢失，
  这些事件更难；K16 正样本更多，AUPRC 更高。

### 3.1 主条件的校准（统计）

| 指标 | 主条件 P1-16+mem K16 |
|---|---:|
| AUROC | 0.7988 |
| AUPRC | 0.1658 |
| AUPRC lift（/正例率 1.386%） | **11.96×** |
| Brier | 0.01344 |
| 全零基线 Brier | 0.01386 |
| Brier skill | **+0.0300** |
| ECE | 0.0132 |
| 正类平均概率 | 0.04787 |
| 负类平均概率 | 0.00112 |
| 正/负均值比 | **42.8×** |
| Train OOF AUROC / AUPRC | 0.8653 / 0.1899 |
| Dev Val AUROC / AUPRC | 0.7988 / 0.1658 |

校准比 Stage-2g.3（Brier skill -0.0045、AUPRC lift 2.03×）明显改善，
但绝对 Precision 仍不足以做门控。

### 3.2 Episode 级 95% CI（统计，500 次 Episode bootstrap）

| 指标 | 点估计 | 95% CI |
|---|---:|---:|
| AUROC | 0.7988 | [0.710, 0.885] |
| AUPRC | 0.1658 | [0.086, 0.333] |
| Precision | 0.3594 | [0.219, 0.563] |
| Recall | 0.2378 | [0.106, 0.371] |
| FP 绝对数 | 139 | [46.5, 270.5] |

Precision 的 CI 上界超过 0.5，但下界只有 0.219；点估计和 Gate 判定
都不通过。FP 的 CI 很宽，说明 Episode 级不确定性很大。

---

## 4. 分 Band 与负样本子集

### 4.1 分 Band（统计，主条件 P1-16+mem K16）

| Band | n | 事件正样本 | 正样本 Episode | AUROC | AUPRC | Precision | Recall | flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 7,471 | 220 | 14 | 0.746 | 0.163 | 0.380 | 0.236 | 137 |
| medium | 8,254 | 92 | 7 | 0.842 | 0.197 | 0.241 | 0.152 | 58 |
| long | 7,940 | 16 | **1** | 0.996 | 0.437 | 0.545 | 0.750 | 22 |

对比 K4/K8 的 P1-16+mem：

| K | Band | AUROC | Precision | Recall |
|---:|---|---:|---:|---:|
| 4 | short | 0.808 | 0.294 | 0.169 |
| 4 | medium | 0.929 | 0.556 | 0.263 |
| 4 | long | 0.999 | 1.000 | 0.250 |
| 8 | short | 0.808 | 0.202 | 0.214 |
| 8 | medium | 0.896 | 0.370 | 0.233 |
| 8 | long | 0.999 | 0.462 | 0.750 |

**观察：**

- short band（220 个 K16 正样本，14 个 Episode）AUROC 0.746，
  但 Precision 0.380 / Recall 0.236；仍是决策相关的主体。
- medium band 的 Precision 只有 0.241 / Recall 0.152。
- long band 的 0.996/0.545/0.750 只来自 **1 个**正样本 Episode（16 个事件），
  不能外推。

### 4.2 负样本子集与混淆矩阵（统计，主条件）

| 子集 | n | flags | 触发率 | 占全部 flags | 平均分数 |
|---|---:|---:|---:|---:|---:|
| event（正类） | 328 | 78 | 23.78% | 35.9% | 0.04787 |
| sustained near-inv | 415 | 16 | 3.86% | 7.4% | 0.01744 |
| near-visible | 4,123 | 11 | 0.27% | 5.1% | 0.00012 |
| far-visible | 10,450 | 34 | 0.33% | 15.7% | 0.00026 |
| far-invisible | 8,349 | 78 | 0.93% | 35.9% | 0.00187 |

**Confusion：TP 78 / FP 139 / FN 250 / TN 23,198。**
Precision 0.3594、Recall 0.2378、Specificity 0.9940。

**绝对误触发：**

- 总 FP = **139**，TP = 78；FP > TP。
- 平均每 Episode FP = **0.813**，中位 0，P90 0，最大 **53**。
- 171 个 Episode 中只有 15 个出现任意 FP；误触发高度集中。
- FP 最多的 Episode：

| Episode | 事件正样本 | TP | FP |
|---:|---:|---:|---:|
| 47 | 9 | 9 | **53** |
| 60 | 16 | 13 | 21 |
| 123 | 0 | 0 | 12 |
| 68 | 12 | 4 | 10 |
| 98 | 0 | 0 | 9 |
| 178 | 0 | 0 | 8 |

Episode 47 一个 Episode 就贡献了 38% 的 FP；Episode 123/98/178 没有任何
事件正样本却贡献 29 个 FP。均值 0.813 掩盖了这些尖峰。

### 4.3 事件 vs 静态“远距离持续不可见”（统计）

| 对比 | 主条件 K16 AUROC |
|---|---:|
| event vs sustained near-inv | **0.7130** |
| event vs far-invisible | **0.7864** |
| event vs far-visible | 0.8530 |
| event vs near-visible | 0.6955 |
| event vs sustained，距离分层加权 | **0.6943** |
| event vs sustained，20–50 m | 0.8828 |
| event vs sustained，50–100 m | 0.6672 |

- 距离分层后仍有 **0.694**（不是纯距离/可见性静态信号）。
- 在 50–100 m（事件主体，188/328）内只有 **0.667**。
- 作为对照，Stage-2g.3 的静态 P1-2 在 event-vs-sustained 上只有
  **0.543**；长历史确实增加了事件特异性，而不是只学“近且不可见”。

### 4.4 Oracle 诊断（仅诊断，不是模型输入）

| 分数 | event vs all | event vs sustained | event vs far-invisible |
|---|---:|---:|---:|
| `-distance`（oracle） | 0.930 | **0.770** | 1.000 |
| `1-visible`（oracle） | 0.812 | 0.500 | 0.500 |
| `-since_localizable`（oracle） | 0.520 | 1.000 | 0.957 |

- 纯距离本身就能把事件从所有负样本中排出来（0.930），但
  event-vs-sustained 只有 0.770；模型在距离分层后仍有 0.694，
  说明它不只是复制距离。
- 纯可见性对 event-vs-sustained 是 0.500（两者都不可见），
  不能解决问题。
- `since_localizable` 是标签定义本身，1.000 是上限参照，不是可用输入。

---

## 5. 历史长度与 last-seen memory

### 5.1 历史长度增益（统计，Episode 配对 bootstrap 1,000 次）

| 比较 | K | ΔAUROC（均值 / 95% CI） | ΔAUPRC（均值 / 95% CI） |
|---|---:|---|---|
| P1-16 vs P0 | 16 | **+0.1634** `[+0.1037, +0.2266]` | +0.1514 `[+0.0664, +0.2695]` |
| P1-16 vs P1-2 | 16 | **+0.1465** `[+0.0922, +0.2045]` | +0.1349 `[+0.0607, +0.2480]` |
| P1-8 vs P1-2 | 16 | +0.0538 `[+0.0266, +0.0819]` | +0.0275 `[+0.0095, +0.0557]` |
| P1-16 vs P0 | 4 | +0.2417 `[+0.1365, +0.3579]` | +0.1543 `[+0.0395, +0.3135]` |
| P1-16 vs P1-2 | 4 | +0.1765 `[+0.0995, +0.2582]` | +0.1463 `[+0.0362, +0.3007]` |
| P1-16 vs P0 | 8 | +0.2660 `[+0.1662, +0.3707]` | +0.1732 `[+0.0582, +0.3251]` |
| P1-16 vs P1-2 | 8 | +0.2172 `[+0.1313, +0.3047]` | +0.1712 `[+0.0544, +0.3269]` |

**结论（统计）：** 8/16 帧历史确实带来了稳定的排序增益，且
Episode 配对 bootstrap CI 不跨 0。Stage-2g.3 的“2 帧太短”判断被证实。

### 5.2 last-seen memory（统计）

可观测 memory 的定义：

- 置信度 = 6 个可观测列的 z-score 均值：
  `sim_max_all_top5`、`heat_0/1/2_top1`、`base_max`、`coh_max`；
- 阈值 = Train common 置信度分布的 70th percentile（**不使用标签**）；
- 特征：当前置信度、距最近高置信帧的年龄、是否有高置信证据、最近
  高置信值、最近 16 帧高置信计数/最大值/均值/argmax 年龄、最近动作
  one-hot、最近高置信 compact 特征，共 87 维。

| 比较 | K | ΔAUROC | ΔAUPRC |
|---|---:|---|---|
| P1-16+mem vs P1-16 | 16 | +0.0229 `[-0.0116, +0.0601]` | -0.0026 `[-0.0570, +0.0539]` |
| P1-16+mem vs P1-16 | 8 | +0.0054 `[-0.0182, +0.0322]` | -0.0067 `[-0.0669, +0.0485]` |
| P1-16+mem vs P1-16 | 4 | +0.0178 `[-0.0032, +0.0444]` | -0.0079 `[-0.0577, +0.0509]` |

Memory age 单独对 event-vs-sustained 的 AUROC：

| K | age AUROC（event vs sustained） | event age 中位 | sustained age 中位 |
|---:|---:|---:|---:|
| 4 | 0.615 | 7 | 23 |
| 8 | 0.605 | 8 | 25 |
| 16 | 0.616 | 10 | 33 |

**判断（统计 + 推断）：** 手工可观测 memory 有弱信号（age AUROC 约
0.61），但对 16 帧历史的增量在 K16 上不显著，AUPRC 甚至为负；
K4/K8 的 AUROC 有小幅正向迹象，但 CI 跨 0，AUPRC 为负。
它不能作为主要贡献，也不值得继续调阈值/百分位。

---

## 6. Episode 级可靠性与极端轨迹

### 6.1 零命中 Episode（统计）

主条件 P1-16+mem K16：

- 22 个正样本 Episode 中，**只有 10 个至少命中 1 个事件**；
- **12 个 Episode 完全零命中**：36, 42, 90, 95, 103, 105, 108, 109,
  142, 153, 157, 165；
- 其中 **Episode 95 是事件数最多的 Episode（23 个事件），却 0 TP**；
- 逐 Episode Recall：中位 **0**，P75 0.531，最大 1.0。

这说明总体 AUROC 0.80 不能解释为“每个 Episode 都能触发”；一半以上的
正样本 Episode 完全漏检。

### 6.2 LOEO（统计）

去掉单个 Dev Val 极端正样本 Episode 后重算：

| 去掉 Episode | 剩余正样本 | AUROC | Precision | Recall | FP |
|---:|---:|---:|---:|---:|---:|
| 95（23 个） | 305 | 0.810 | 0.359 | 0.256 | 139 |
| 18（16 个） | 312 | 0.790 | 0.332 | 0.221 | 139 |
| 41（16 个） | 312 | 0.788 | 0.335 | 0.215 | 133 |
| 42（16 个） | 312 | 0.798 | 0.359 | 0.250 | 139 |
| 60（16 个） | 312 | 0.796 | 0.355 | 0.208 | 118 |

AUROC 和 Precision/Recall 相对稳定，没有单一 Episode 决定总体 AUROC；
但所有 LOEO 结果都低于 Gate 的 P/R 0.5。**稳定地失败**，而不是某个
异常 Episode 造成的假象。

Train OOF 上排除 Episode 299 或 top event Episode 也只带来小幅变化
（AUROC 0.86–0.88，P 0.22–0.26，R 0.24–0.28）。

### 6.3 极端 Episode 重训消融（统计）

在 Train 中排除以下 Episode 后重训 P1-16/P1-16+mem K16
（Dev Val 不变）：

- `no299`：只排除 Episode 299；
- `no_extreme`：排除 Episode 299 + 14 + 97 + 367。

| 条件 | AUROC | AUPRC | Precision | Recall | TP / FP | FP/Episode |
|---|---:|---:|---:|---:|---:|---:|
| 主条件（baseline） | 0.7988 | 0.1658 | 0.3594 | 0.2378 | 78 / 139 | 0.813 |
| no299 | 0.8154 | 0.1564 | 0.3594 | 0.2290 | 75 / 134 | 0.784 |
| no_extreme | **0.8261** | 0.1715 | **0.4530** | 0.1768 | 58 / 70 | **0.409** |

**解读（推断）：**

- 只去掉 Episode 299 几乎不改变结果，说明 Episode 299 不是事件探针的
  唯一问题。
- 去掉 299 + top event Episodes 后 Precision 提高到 0.453、FP 降到 70，
  但 Recall 降到 0.177；仍然远低于 Gate。极端轨迹对假阳性有贡献，
  但不是唯一原因，且去掉它们会牺牲正样本召回。
- 这进一步支持“停止调参/堆叠”的判断：靠删 Episode 或调阈值只能
  在 P/R 之间移动，无法达到 0.5/0.5。

---

## 7. 阈值独立性与 Gate 细节

### 7.1 阈值来源（审计）

- 所有主结果阈值来自 **Train OOF**；Dev Val 不参与阈值选择。
- 主条件 K16 的 Train OOF `f1_joint`：
  **threshold 0.008415，Precision 0.2588，Recall 0.2282**。
- Train OOF 上不存在 Precision ≥ 0.5 的受限阈值
  （`precision_0.5_constrained = None`）。
- 为区分“阈值不好”与“表示不可分”，额外做 Dev Val **oracle 阈值扫描**
  （仅诊断，不用于选择）：

| 条件 | 最大 F1（P/R） | 最大 Precision（R） | P≥0.5 时最大 Recall | R≥0.5 时最大 Precision |
|---|---:|---:|---:|---:|
| P1-16+mem K16 | 0.304（0.321/0.290） | **0.479**（0.174） | **不存在** | 0.075 |
| P1-16 K16 | 0.282（0.297/0.268） | 1.000（0.003） | 0.104 | 0.077 |
| P1-16+mem K8 | 0.262（0.311/0.227） | 0.483（0.178） | 不存在 | 0.043 |
| P1-16+mem K4 | 0.268（0.317/0.232） | 0.317（0.232） | 不存在 | 0.031 |

**即使事后扫描所有 Dev Val 阈值，也不存在 P≥0.5 且 R≥0.5 的工作点。**
P1-16 K16 可以做到 Precision 1.0，但 Recall 只有 0.003；或 Precision 0.5
时 Recall 只有 0.104。Gate 失败不是阈值选择问题，而是表示/标签问题。

### 7.2 Gate 逐项（K16 主条件）

| 检查 | 结果 | 证据 |
|---|---|---|
| G1 Train OOF 受限 P≥0.5 阈值存在 | FAIL | `f1_joint` Train P=0.259；无 P≥0.5 的受限阈值 |
| G2 Dev Val P≥0.5 且 R≥0.5 | FAIL | P=0.359 / R=0.238 |
| G3 绝对 FP 预算 | FAIL | FP 139 > TP 78；均值 0.813，最大 53 |
| G4 event-vs-sustained ≥0.70，far-inv ≥0.75 | PASS | 0.713 / 0.786 |
| G5 长历史相对 P0/P1-2 显著增益 | PASS | +0.186 / +0.169 AUROC，CI >0 |
| G6 LOEO 后仍通过 G2/G3 | FAIL | 去掉 Episode 95 后 P 0.359 / R 0.256 |
| **总 Gate** | **FAIL** | — |

---

## 8. 旧探针基线与对比

Stage-2g.3 的静态 near-inv 探针直接应用到事件标签上（阈值在事件
Train OOF 上重选，Dev Val 评估）：

| 基线 | K | AUROC | AUPRC | Precision | Recall | event-vs-sustained |
|---|---:|---:|---:|---:|---:|---:|
| legacy Stage-2g.3 P0 | 16 | 0.6001 | 0.0280 | 0.079 | 0.030 | 0.529 |
| legacy Stage-2g.3 P1-2 | 16 | 0.6227 | 0.0277 | 0.022 | 0.009 | 0.543 |
| 本轮 P0 | 16 | 0.6128 | 0.0284 | 0.0156 | 0.0061 | 0.542 |
| 本轮 P1-2 | 16 | 0.6287 | 0.0423 | 0.0725 | 0.0152 | 0.577 |
| **本轮 P1-16+mem** | 16 | **0.7988** | **0.1658** | **0.3594** | **0.2378** | **0.713** |

**结论：** 静态探针（无论 P0 还是 P1-2）在 event-vs-sustained 上只有
0.50–0.58，基本不能区分“近期丢失”与“持续不可见”。长历史事件探针把
这个 AUROC 提高到 0.71，说明长时序确实是必要方向；但绝对
Precision/Recall 仍不足。

---

## 9. 失败原因分析

### 9.1 长历史解决了“排序”，没有解决“Episode 级触发可靠性”

- 总体 AUROC 0.80 是真实提升，且 event-vs-sustained 0.71 明显高于
  静态探针 0.54；这不是模型没学到事件。
- 但 10/22 正样本 Episode 有 TP，12/22 完全零命中；逐 Episode Recall
  中位 0。总体 AUROC 被部分 Episode 的高分拉高，不能代表“每个 Episode
  都能可靠触发”。
- 逐 Episode 零命中 + 少数 Episode 贡献大量 FP（Episode 47 53 FP）
  是闭环门控的致命问题：在真实 Episode 上，触发既可能完全缺失，
  也可能在错误 Episode 上密集出现。

### 9.2 事件标签仍与“距离/场景”存在相关性，但不是纯静态状态

- 距离分层后的 event-vs-sustained AUROC 仍有 0.694（50–100 m 为 0.667），
  说明模型不是简单复制距离；但距离本身（oracle）对 event-vs-sustained
  有 0.770，说明标签与距离仍相关。
- 在 0–20 m 没有 sustained 负样本（事件正样本 37，sustained 0），
  该距离段的 event-vs-sustained 无法定义；20–50 m 的 AUROC 0.883
  但样本少（事件 103 / sustained 27）。
- 因此“识别近期丢失”与“识别近目标”仍有部分纠缠；需要更严格的
  同距离/同场景配对标签或显式空间信念才能分离。

### 9.3 事件样本大多是终端丢失，Recovery 缺少可观测目标

- K16 Dev Val：321/328 事件在剩余轨迹内再未 localizable；仅 7 个在
  10 步内重新 localizable。
- 这解释了为什么“事件检测”即使改善，也不等于“Recovery 能成功”：
  大多数事件之后没有可恢复的观测。Recovery 探针路线的前提
  （近期丢失、可通过搜索重新获得目标）在当前轨迹中只覆盖少数事件。

### 9.4 绝对假阳性仍然过高

- 主条件 FP 139 > TP 78；far-invisible 贡献 78 FP，near-visible 11，
  sustained 16，far-visible 34。
- 虽然平均 FP/Episode 0.813 看起来可接受，但 Episode 47 一个 Episode
  就有 53 FP；几个 FP 密集 Episode 决定了下游风险。
- 这再次说明“逐象限误触发率 ≤2%”不足以保护低基础率闭环；
  必须用绝对 FP 预算和 Episode 级分布。

### 9.5 last-seen memory 不是缺失的关键

- 手工可观测 memory age 的 event-vs-sustained AUROC 只有 0.616；
  对 16 帧历史的增量不显著。
- 这说明“再加一个手工 memory”不是解决办法；真正的区分信号已经在
  16 帧历史里，但绝对分离度仍不足。

---

## 10. 决策

### **决策：停止当前 Recovery 探针路线。**

依据冻结 Gate：

- G1/G2/G3/G6 失败；即使 Dev Val oracle 阈值也无法达到
  P≥0.5 / R≥0.5（最大 F1 0.304，最大 Precision 0.479）。
- 12/22 正样本 Episode 零命中；少数 Episode 贡献大量 FP。
- 事件正样本绝大多数是终端丢失（321/328 不再 localizable），
  Recovery 缺少可观测 re-acquisition 目标。
- 长历史是有效方向，但继续堆叠 MLP/历史帧/阈值不会把 Episode 级
  可靠性从“一半 Episode 漏检”变成可用触发器。

**因此：**

- 不训练完整 MEC；
- 不接入 Recovery 控制器；
- 不继续增加 MLP 层数、历史帧数或盲目调阈值；
- 不读取 Final Test 300；
- 保留长时序事件信号作为**诊断/空间信念特征**，而不是独立触发器。

### 下一步建议（本轮未执行）

1. **转向显式目标空间信念 / 粗导航定位**：维护目标位置/可见性/不确定
   性的显式分布，而不是二分类“是否近期丢失”。
2. 如果仍要做事件触发，必须重新定义/采集**可恢复的丢失事件**：
   事件后目标在有限步内可重新 localizable；当前 321/328 不满足。
3. 采用**同距离、同场景、同 Episode** 的配对负样本，分离“近期丢失”
   与“持续不可见/远距离”的距离混淆。
4. 评估指标改成 **Episode 级零命中率、逐 Episode Recall/FP 分布、
   绝对 FP 预算**，而不是只看总体 AUROC。
5. 若要继续探针，至少需要多 seed、更多正样本 Episode、以及
   排除极端轨迹后的稳定复现；本轮单 seed 不足以证明可部署性。

---

## 11. 缺失项与限制

1. 单 seed（seed 0）和单 MLP 架构；没有多 seed 复现。
2. 事件标签只在现有 Stage-2g.3 M0@0.7 真实闭环轨迹上生成；
   没有新采集，也没有合成数据。
3. `common` 集要求 `step>=16`，因此 Train/Val 的有效 Episode 数为
   563/171，不是 600/180；事件正样本覆盖仍满足预注册门槛。
4. 大多数事件是终端丢失；`until_next_localizable` 的观测样本很少
   （Dev Val 只有 7 个 K16 事件在 Episode 内再次 localizable）。
5. Long band 只有 1 个正样本 Episode（16 个 K16 事件），其
   AUROC 0.996/Precision 0.545/Recall 0.750 不能外推。
6. Gate 的 Dev Val oracle 阈值扫描仅用于诊断，不参与模型/阈值选择；
   实际报告的主阈值仍来自 Train OOF。
7. 本轮没有读取 Final Test 300，没有修改旧数据/checkpoint/历史报告。

---

## 12. 产物、命令与提交

### 12.1 主探针

```
runs/v83_spatial_stage2/stage2g4/probe_20261019T062000Z/
├── probe_results.json          # 15 个条件（5 变体 × 3 K）
├── probe_provenance.json
├── predictions.npz             # OOF/Val 分数
└── models/                     # 90 个轻量 checkpoint
```

配置：5-fold Episode OOF，30 epochs，hidden 128，batch 8192，
Adam lr 1e-3 / wd 1e-4，seed 0；common 集 `step>=16 & action!=STOP`。

### 12.2 分析与消融

```
runs/v83_spatial_stage2/stage2g4/analysis_20261019T072000Z/analysis.json
runs/v83_spatial_stage2/stage2g4/ablation_no_extreme_20261019T063000Z/
runs/v83_spatial_stage2/stage2g4/ablation_no299_20261019T073000Z/
```

- `no_extreme` 排除 Train Episode 299 + 14 + 97 + 367；
- `no299` 只排除 Episode 299；
- 两者都在 K=16、P1-16/P1-16+mem 上重训，Dev Val 不变。

### 12.3 复现命令

```bash
python v83_spatial_stage2/stage2g4_probe.py \
  --train-dir runs/v83_spatial_stage2/stage2g3/real_train_20261019T132500Z \
  --val-dir runs/v83_spatial_stage2/stage2g3/real_val_20261019T132000Z \
  --output-dir runs/v83_spatial_stage2/stage2g4/probe_<stamp> \
  --variants P0 P1-2 P1-8 P1-16 P1-16+mem \
  --ks 4 8 16 --epochs 30 --device cuda --save-models

python v83_spatial_stage2/stage2g4_analyze.py \
  --train-dir runs/v83_spatial_stage2/stage2g3/real_train_20261019T132500Z \
  --val-dir runs/v83_spatial_stage2/stage2g3/real_val_20261019T132000Z \
  --probe-dir runs/v83_spatial_stage2/stage2g4/probe_<stamp> \
  --stage2g3-probe-dir runs/v83_spatial_stage2/stage2g3/probe_20261019T133500Z \
  --ablation-dir runs/v83_spatial_stage2/stage2g4/ablation_no_extreme_<stamp> \
  --output-dir runs/v83_spatial_stage2/stage2g4/analysis_<stamp> \
  --bootstrap 500 --paired-bootstrap 1000
```

### 12.4 测试

```bash
python -m unittest -v v83_spatial_stage2.test_stage2g4
```

本轮：**6 tests OK**（事件因果性、无未来泄漏、历史顺序、memory
不读标签、阈值/分层指标）。

### 12.5 提交

- 报告：`v83_spatial_stage2/V83_STAGE2G4_LONG_HORIZON_EVENT_PROBE_REPORT.md`
- 协议：`v83_spatial_stage2/V83_STAGE2G4_PROTOCOL.md`
- 源码：`stage2g4_data.py`、`stage2g4_metrics.py`、
  `stage2g4_probe.py`、`stage2g4_analyze.py`
- 测试：`test_stage2g4.py`
- **不提交** `.pt`/`.npz`/日志；全部留在 `runs/`。

---

## Appendix A. 条件汇总（全部 15 个）

| 条件 | AUROC | AUPRC | P | R | TP/FP |
|---|---:|---:|---:|---:|---:|
| P0 K4 | 0.6051 | 0.0074 | 0.0149 | 0.1098 | 9/596 |
| P0 K8 | 0.5834 | 0.0151 | 0.0341 | 0.0184 | 3/85 |
| P0 K16 | 0.6128 | 0.0284 | 0.0156 | 0.0061 | 2/126 |
| P1-2 K4 | 0.6680 | 0.0130 | 0.0303 | 0.0732 | 6/192 |
| P1-2 K8 | 0.6320 | 0.0178 | 0.0215 | 0.0123 | 2/91 |
| P1-2 K16 | 0.6287 | 0.0423 | 0.0725 | 0.0152 | 5/64 |
| P1-8 K4 | 0.7969 | 0.0576 | 0.1754 | 0.1220 | 10/47 |
| P1-8 K8 | 0.7636 | 0.0567 | 0.1104 | 0.1043 | 17/137 |
| P1-8 K16 | 0.6826 | 0.0691 | 0.2000 | 0.0549 | 18/72 |
| P1-16 K4 | 0.8446 | 0.1393 | 0.3036 | 0.2073 | 17/39 |
| P1-16 K8 | 0.8498 | 0.1712 | 0.2176 | 0.2270 | 37/133 |
| P1-16 K16 | 0.7755 | 0.1729 | 0.3022 | 0.2470 | 81/187 |
| P1-16+mem K4 | 0.8628 | 0.1305 | 0.3636 | 0.1951 | 16/28 |
| P1-16+mem K8 | 0.8548 | 0.1558 | 0.2516 | 0.2454 | 40/119 |
| P1-16+mem K16 | 0.7988 | 0.1658 | 0.3594 | 0.2378 | 78/139 |

## Appendix B. 术语

- **event**：`distance<100 & target_visible==0 & since_localizable<=K`。
- **sustained near-inv**：near-inv 但不是 event（最近 localizable >K 或从未）。
- **P0/P1-2/P1-8/P1-16**：当前特征 + 0/2/8/16 帧历史。
- **P1-16+mem**：P1-16 + 可观测 last-seen memory（不读标签）。
- **OOF**：5-fold Episode-level out-of-fold；阈值只在这里选。
- **common 集**：`step>=16 & action!=STOP`。
- **FP/Episode**：该 Episode 内所有非 event 状态的 flags 数。
- **LOEO**：leave-one-episode-out；这里用于检查极端 Episode 影响。
