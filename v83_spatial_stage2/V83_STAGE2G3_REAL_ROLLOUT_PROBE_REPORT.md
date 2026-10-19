# V8.3 Stage-2g.3 — 真实闭环 P0/P1-2 Target-Presence Probe 报告

日期：2026-10-19

分支：`experiment/v83-spatial-foundation`

前序提交：`f5ae975`
本轮性质：**恢复已有运行 + 纯后处理 + 归档**

> 本轮**没有**重新采集 Train 600，**没有**重新训练 P0/P1-2，**没有**训练完整 MEC，
> **没有**接入 Recovery 控制器，**没有**读取 Final Test 300，**没有**修改 M0/M1
> checkpoint，**没有**生成新的大规模数据集。所有逐状态预测均由已保存的
> checkpoint 做纯推理恢复；推理结果与保存的聚合指标逐位一致（见 §2）。

---

## 0. TL;DR（中文结论）

1. **上一轮产物完整、数值可复现。** Train 600：81,303 states，1,394 个
   near-target-invisible 正样本，覆盖 **43 个 Episode**；P1-2 Dev Val
   AUROC **0.5922**、AUPRC **0.0577**；Train-OOF 阈值下 Precision
   **0.0976**、Recall **0.0821**。这些数值与实际保存的 `.npz`、模型和
   `probe_results.json` 一致，不是聊天记录误记。
2. **P1-2 相对 P0 有可测但很小的历史增益。** 点估计 AUROC
   +0.0105（0.5817 → 0.5922），AUPRC +0.0097（0.0480 → 0.0577）。
   按 Episode 重采样的 95% CI：ΔAUROC **[-0.0181, +0.0449]**（23.2% 的
   重采样 ≤ 0），ΔAUPRC **[+0.0004, +0.0302]**。历史信息不是完全无效，
   但**远不足以构成可用触发器**。
3. **Gate 失败。** P0 和 P1-2 均未通过 Precision ≥ 0.50 与 Recall ≥ 0.50；
   虽然 far-visible / far-invisible 的“误触发率”都 ≤ 5%，但远距离状态占
   全部状态的 81.3%，2%–3% 的远距离误触发率换算成绝对数量是
   **545 个假阳性**（P1-2），而真阳性只有 61 个。这是本轮最关键的
   决策性失败。
4. **头号失败原因不是“模型不够大”，而是标签/分布与可用观测不匹配。**
   - 真实近目标不可见正样本中，最后一次 `target_localizable=1` 距当前
     状态的中位数是 **19 步**；只有 **6.7%** 在 2 步以内。P1-2 的
     2 帧历史在大多数正样本上**根本看不到“最近一次高置信度观测”**。
   - 真实不可见是**持续状态**：60% 的连续不可见 run ≥ 20 步，中位数
     23 步；Stage-2g.1 合成正样本则是“RIGHT 转三次后刚离开视野”的
     单步转换，历史动作 100% 是 `RIGHT→RIGHT`。
   - 在真实正样本中，历史动作只有 46.1% 是 `FORWARD→FORWARD`，
     动作本身以 FORWARD 为主（56.6%），不是合成数据的转向模式。
   - 短起始距离 band（占 Dev Val 正样本 60%）AUROC 仅 **0.4525 / 0.4581**，
     低于随机；medium band 约 0.75–0.76；long band 0.95 但只有 **1 个**
     正样本 Episode。总体 0.592 主要是 medium band 与 Episode 间差异
     撑起来的，不代表短距离状态级识别可用。
   - 概率校准失败：P1-2 Brier 0.02857 **差于“全部预测 0”的 0.02845**
     （Brier skill = **-0.0045**）；AUPRC 仅比基础正例率 2.845% 高
     **2.03 倍**；正/负类平均分数比只有 **5.58 倍**，而要在 2.845%
     基础率下达到 Precision 0.5，近似需要 ~34 倍的正负似然比。
5. **明确判断：不值得继续优化当前 P0/P1-2 探针。** 继续加特征、加历史帧、
   调 MLP 或换阈值，无法解决“大多数正样本的最近高置信度观测在 10–50 步
   之前、且短 band 内几乎不可分”的问题。**不要**把该探针接入 Recovery，
   **不要**据此训练 MEC，**不要**读 Final Test 300。
6. 若还要继续这条线，应把它重新定义为一个**“最近可定位 → 近期丢失”事件
   触发器**，而不是“当前距离 < 100 m 且不可见”的静态标签；并采集
   真实闭环下的“丢失事件”数据，而不是继续用合成转向样本。这个方向
   本轮没有执行，属于建议，不是结果。

---

## 1. 恢复的运行阶段与已有产物

### 1.1 运行阶段（观测）

| 阶段 | 目录 | 内容 | 状态 |
|---|---|---|---|
| Dev Val 采集 | `runs/v83_spatial_stage2/stage2g3/real_val_20261019T132000Z/` | 180 个 `episode_*.npz`，26,628 states | 完整 |
| Train 采集 | `runs/v83_spatial_stage2/stage2g3/real_train_20261019T132500Z/` | 600 个 `episode_*.npz`，81,303 states，日志 907.8 s | 完整 |
| P0/P1-2 探针 | `runs/v83_spatial_stage2/stage2g3/probe_20261019T133500Z/` | `probe_results.json`、`probe_provenance.json`、`tables.md`、36 个 checkpoint | 完整 |
| 本轮后处理 | `runs/v83_spatial_stage2/stage2g3/analysis_20261019T054458Z/` | `analysis.json`、`predictions.npz`、分析日志 | 本轮生成 |

Train 采集日志末尾：

```text
train: 600/600 episodes, 81303 states, 907.8s
{"split": "train", "episodes": 600, "states": 81303, ...}
```

### 1.2 产物哈希（观测）

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `real_train_.../manifest.json` | 64,641 B | `22215cfe7e928bd9d280b2e155709fa8d774f7ca6d483e3398bba0c71e1d1110` |
| `real_val_.../manifest.json` | 60,858 B | `c2ab53ffeeda7655cc85051769452386dfcdc895c1e542be3db1215e59d297e4` |
| `probe_.../probe_results.json` | 243,970 B | `e5a97783e2f3a2bbfd298eb5953939701b281528785bccbc39dda2b541e76e22` |
| `probe_.../probe_provenance.json` | 65,602 B | `e6b354ab1510654ff27eb838e3e6b342f078ffa336011dd26dfc3e850b403e64` |
| `probe_.../reproduction_check.json` | 234 B | `b920d00f8855847ebb960d4abf3e802b984fe3dbb3ef82d0cc669d1fe8e226d5` |
| 本轮 `analysis.json` | 455,578 B | `575d242b835a123c8dfb3aa7e3b933481ca1c66c38c28be8a898fe57cc02a54e` |
| 本轮 `predictions.npz` | 1,569,432 B | `c5db3e83f0c27edf5efdeedfeb8b88b6fe04da84b4f2301107e34f296fc0da59` |

模型 checkpoint 共 36 个：2 个 variant × 3 个 head ×（5 个 fold + 1 个
external）。本轮没有改动其中任何一个。

### 1.3 探针配置（观测）

来自 `probe_provenance.json`：

```text
folds=5, epochs=30, hidden=128, batch_size=8192,
lr=1e-3, weight_decay=1e-4, seed=0,
class_weight = n_neg / n_pos,
standardize = Train mean/std, clip [-20, 20],
threshold_source = Train OOF only
```

运行命令（原样）：

```bash
python v83_spatial_stage2/stage2g3_probe.py \
  --train-dir runs/v83_spatial_stage2/stage2g3/real_train_20261019T132500Z \
  --val-dir runs/v83_spatial_stage2/stage2g3/real_val_20261019T132000Z \
  --output-dir runs/v83_spatial_stage2/stage2g3/probe_20261019T133500Z \
  --save-models \
  --synthetic-checkpoint runs/v83_spatial_stage2/stage2g1/probe_20261019T124500Z/models/v2_near_target_invisible_external.pt \
  --epochs 30 --bootstrap 200
```

---

## 2. 数值核验：聊天记录 vs 实际产物

| 上次记录 | 实际产物 | 结论 |
|---|---|---|
| Train 600：81,303 states | `real_train/manifest.json`：`states=81303`；NPZ 逐 Episode 求和 81,303 | 一致 |
| near-invisible 1,394 | 原始 NPZ `near_target_invisible.sum()=1394`；move-valid 1,382 | 一致 |
| 覆盖 43 个 Episode | 43 个 Episode 至少含 1 个正样本 | 一致 |
| P1-2 AUROC 0.592 | `0.5922319`（保存）/ `0.5922319`（本轮推理，绝对差 5.3e-08） | 一致 |
| P1-2 AUPRC 0.058 | `0.05773649`（保存）/ `0.05773649`（本轮推理，绝对差 2.0e-09） | 一致 |
| 历史增益约 +0.010 AUROC | 点估计 `+0.010487`（0.5817447 → 0.5922319） | 一致 |
| 独立验证 Precision 0.098 / Recall 0.082 | 在 `f1_joint_far5` 阈值 `0.0143788` 下：`0.09760 / 0.08210` | 一致 |

**精确复现（观测）：**

- 用 `v1_near_target_invisible_external.pt` 对 Dev Val 重新推理，
  AUROC 绝对差 `5.3e-08`，AUPRC 绝对差 `2.0e-09`，Brier 差 `2.4e-10`，
  ECE 差 `1.0e-10`；P0 的 AUROC/AUPRC 绝对差为 0。
- 用 5 个 fold checkpoint 重建 Train OOF，AUROC/AUPRC 与保存值一致
  （P0 0.661657 / 0.038209；P1-2 0.673039 / 0.042191）。
- 用 OOF 重新执行 `select_thresholds`，`f1_joint_far5` 阈值与保存值相差
  P0 `1.03e-08`、P1-2 `5.84e-08`（仅 float32/float64 精度差）。
- 保存的 `reproduction_check.json`：256 个状态，最大概率差 `2.83e-11`。

**有效 Episode 数与“600/180”的差异（观测）：**

- Train：600 个 NPZ，但 `filter_moves` 后只有 **587** 个 Episode 有
  move-valid 状态；13 个 Episode 长度为 1–3 步且只有 STOP，全部被过滤。
- Dev Val：180 个 NPZ，但只有 **179** 个 Episode 有 move-valid 状态；
  Episode 35 长度为 1（STOP）。
- 因此所有探针指标的分母是 **Train 79,609 / Dev Val 26,120 个
  move-valid 状态**，不是 81,303 / 26,628。报告中的正样本数在探针评测
  语境下用 1,382 / 743；在原始采集语境下用 1,394 / 749。两者都明确标注。

---

## 3. 数据与标签定义

### 3.1 标签（观测，来自 collector/provenance）

```text
target_visible       = occlusion-aware visible_pixels >= 1
target_localizable   = target_visible AND goal_half_size_px >= 10
                       AND visible_pixels >= 16
near_target_invisible = true distance < 100 m AND target_visible == 0
```

- `distance_m` 是仿真器真实距离，只用于标签/评测，不进入特征。
- Band 是**Episode 起始距离**的轨迹混合 band，不是当前距离：

| band | 起始距离范围 | 到达距离范围 |
|---|---:|---:|
| short | 150–250 m | 30–80 m |
| medium | 250–400 m | 50–115 m |
| long | 400–550 m | 80–160 m |

### 3.2 类别比例（观测）

| 数据 | states | positives | 正例率 | 正样本 Episode |
|---|---:|---:|---:|---:|
| Train 原始 | 81,303 | 1,394 | 1.715% | 43 / 600 |
| Train move-valid | 79,609 | 1,382 | 1.736% | 43 / 587 |
| Dev Val 原始 | 26,628 | 749 | 2.813% | 22 / 180 |
| Dev Val move-valid | 26,120 | 743 | 2.845% | 22 / 179 |

Dev Val 的正例率是 Train 的 **1.64 倍**。阈值在 Train OOF 上选择；
val 的正例率更高，单看这一点不会让 Precision 变差，反而应更容易；
实际 Precision 仍只有 0.098，说明问题在排序/分离度，不在类别比例。

---

## 4. P0 与 P1-2：Dev Val 总体与分 Band

### 4.1 总体指标（统计）

`near_target_invisible` head，Dev Val 26,120 states / 743 positives。

| 指标 | P0 | P1-2 | Δ（P1-2 − P0） |
|---|---:|---:|---:|
| AUROC | 0.5817 | 0.5922 | +0.0105 |
| AUPRC | 0.0480 | 0.0577 | +0.0097 |
| AUPRC lift（/正例率 0.02845） | 1.69× | 2.03× | +0.34× |
| Brier | 0.02897 | 0.02857 | -0.00039 |
| Brier skill（vs 全零基线 0.02845） | **-0.0184** | **-0.0045** | +0.0139 |
| ECE（10 bins） | 0.02808 | 0.02753 | -0.00054 |
| 正类平均概率 | 0.00805 | 0.01422 | +0.00617 |
| 负类平均概率 | 0.00319 | 0.00255 | -0.00064 |
| 正/负均值比 | 2.52× | 5.58× | +3.06× |
| Train-OOF 选中阈值 | 0.0197754 | 0.0143788 | — |
| Precision @ 选中阈值 | 0.0626 | 0.0976 | +0.0350 |
| Recall @ 选中阈值 | 0.0579 | 0.0821 | +0.0242 |
| TP / FP / FN / TN | 43 / 644 / 700 / 24,733 | 61 / 564 / 682 / 24,813 | — |
| Precision @ 0.5 | 0.0606 | 0.2000 | +0.1394 |
| Recall @ 0.5 | 0.0027 | 0.0067 | +0.0040 |
| 选中阈值下 far-visible 误触发率 | 2.585% | 2.852% | +0.267 pp |
| 选中阈值下 far-invisible 误触发率 | 2.329% | 2.142% | -0.187 pp |
| 选中阈值下 near-visible 误触发率 | 2.825% | 0.459% | -2.366 pp |
| 选中阈值下 flags | 687 | 625 | -62 |
| flags 中 near-invisible 占比 | 6.26% | 9.76% | +3.50 pp |
| flags 中 far-visible 占比 | 47.74% | 57.92% | +10.18 pp |
| flags 中 far-invisible 占比 | 28.97% | 29.28% | +0.31 pp |
| flags 中 near-visible 占比 | 17.03% | 3.04% | -13.99 pp |

**Episode-level bootstrap 95% CI（500 次，statistical）：**

| 指标 | P0 | P1-2 |
|---|---|---|
| AUROC | [0.4306, 0.7579] | [0.4349, 0.7739] |
| AUPRC | [0.0217, 0.1049] | [0.0260, 0.1189] |
| Brier | [0.0170, 0.0446] | [0.0164, 0.0440] |

CI 很宽，根因是 Dev Val 只有 **22 个正样本 Episode**。

### 4.2 分 Band（统计）

这里的 band 是 Episode 起始距离 band；正样本仍全部满足当前距离 < 100 m。

| Band | n | positives | 正样本 Episode | P0 AUROC | P1-2 AUROC | ΔAUROC | P0 AUPRC | P1-2 AUPRC | P0 P/R @选中 | P1-2 P/R @选中 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 8,291 | 446 | 14 | **0.4525** | **0.4581** | +0.0055 | 0.0588 | 0.0701 | 0.090 / 0.047 | 0.173 / 0.065 |
| medium | 9,067 | 250 | 7 | 0.7478 | 0.7628 | +0.0150 | 0.0612 | 0.0724 | 0.045 / 0.064 | 0.067 / 0.084 |
| long | 8,762 | 47 | **1** | 0.9594 | 0.9547 | -0.0048 | 0.0744 | 0.0880 | 0.061 / 0.128 | 0.076 / 0.234 |

**解释（推断，证据充分）：**

- **short band 占 Dev Val 正样本的 60.0%（446/743），但 AUROC 只有
  0.4525/0.4581，低于随机。** 总体 0.592 不是“短距离近目标”状态级识别
  能力的证据。
- medium band 有信号（0.75–0.76），但只有 7 个正样本 Episode。
- long band 的 0.95 由 **1 个**正样本 Episode（47 个状态）贡献，不能外推。
- P1-2 的历史增益在 short band 只有 +0.0055 AUROC，基本没有改善；
  medium +0.0150；long -0.0048。历史增益也没有解决最关键的短 band。

### 4.3 共同阈值对比（统计；区分“排序提升”与“阈值移动”）

P1-2 选中的阈值比 P0 更低。为了排除“只是阈值更低导致 Recall 更高”，
在**相同阈值**下比较：

| 阈值 | P0 P / R / flags | P1-2 P / R / flags | ΔP | ΔR |
|---:|---:|---:|---:|---:|
| 0.0143788 | 0.0601 / 0.0686 / 849 | 0.0976 / 0.0821 / 625 | +0.0375 | +0.0135 |
| 0.0197754 | 0.0626 / 0.0579 / 687 | 0.1011 / 0.0727 / 534 | +0.0385 | +0.0148 |
| 0.0500 | 0.0792 / 0.0390 / 366 | 0.1381 / 0.0498 / 268 | +0.0589 | +0.0108 |
| 0.1000 | 0.0820 / 0.0202 / 183 | 0.1548 / 0.0350 / 168 | +0.0728 | +0.0148 |

结论：P1-2 在**相同阈值**下确实比 P0 有更高的 Precision/Recall，历史
信息不是单纯靠降低阈值“刷”出来的；但提升后的绝对 Precision 仍只有
0.10–0.15，Recall 0.04–0.08，离可用触发器很远。

---

## 5. 四象限混淆与错误触发率

### 5.1 选中阈值下的四象限（统计）

四象限定义：near = `distance_m < 100`，visible = `target_visible == 1`。
`near_target_invisible` 正好是 near-invisible 象限。

**P0（阈值 0.0197754）：**

| 象限 | n | flags | 触发率 | 占全部 flags | 平均分数 |
|---|---:|---:|---:|---:|---:|
| near-invisible（正类） | 743 | 43 | 5.79% | 6.26% | 0.00805 |
| near-visible | 4,141 | 117 | 2.83% | 17.03% | 0.00205 |
| far-visible | 12,691 | 328 | 2.58% | 47.74% | 0.00379 |
| far-invisible | 8,545 | 199 | 2.33% | 28.97% | 0.00284 |

**P1-2（阈值 0.0143788）：**

| 象限 | n | flags | 触发率 | 占全部 flags | 平均分数 |
|---|---:|---:|---:|---:|---:|
| near-invisible（正类） | 743 | 61 | 8.21% | 9.76% | 0.01422 |
| near-visible | 4,141 | 19 | 0.46% | 3.04% | 0.00072 |
| far-visible | 12,691 | 362 | 2.85% | 57.92% | 0.00285 |
| far-invisible | 8,545 | 183 | 2.14% | 29.28% | 0.00299 |

**Confusion（P1-2）：TP 61 / FP 564 / FN 682 / TN 24,813；Precision
0.0976、Recall 0.0821、Specificity 0.9778。**

### 5.2 距离段（统计）

| 距离段 | n | positives | P0 flags / P / R | P1-2 flags / P / R |
|---|---:|---:|---:|---:|
| 0–20 m | 87 | 37 | 8 / 0.875 / 0.189 | 11 / 1.000 / 0.297 |
| 20–50 m | 1,237 | 130 | 23 / 0.652 / 0.115 | 30 / 0.600 / 0.138 |
| 50–100 m | 3,560 | 576 | 129 / 0.163 / 0.036 | 39 / 0.821 / 0.056 |
| 100–200 m | 7,897 | 0 | 93 / 0 / 0 | 118 / 0 / 0 |
| 200–400 m | 10,185 | 0 | 351 / 0 / 0 | 317 / 0 / 0 |
| ≥400 m | 3,154 | 0 | 83 / 0 / 0 | 110 / 0 / 0 |

P1-2 在 50–100 m 的 Precision 看似 0.821，但 Recall 只有 0.056（576 个
正样本只抓到 32 个）；在 <50 m 的 Precision 较高，但样本少。**100 m
以外的 545 个假阳性是 Precision 被拉低的主因**（far-visible 362 +
far-invisible 183；按距离拆分为 118 + 317 + 110）。

### 5.3 Gate 判定（统计，来自 `probe_results.json`）

| 检查 | P0 | P1-2 |
|---|---|---|
| Precision ≥ 0.50 | **False**（0.063） | **False**（0.098） |
| Recall ≥ 0.50 | **False**（0.058） | **False**（0.082） |
| far-visible 误触发率 ≤ 5% | True（2.59%） | True（2.85%） |
| far-invisible 误触发率 ≤ 5% | True（2.33%） | True（2.14%） |
| 不被少数 Episode/距离段主导 | True | True |
| 历史增益（AUROC 或 AUPRC ≥ 0.01） | — | True（+0.0105 / +0.0097） |
| **总 Gate** | **FAIL** | **FAIL** |

**Gate 设计的一个问题（推断）：** “far-visible / far-invisible 误触发率
≤ 5%”是逐象限比例，不是绝对假阳性预算。far 状态有 21,236 个，占
81.3%；2%–3% 的比例就是 500+ 个假阳性。以当前基础率 2.845%，
在 564 个 FP 下即使 Recall 100% 也只有约 0.116 的 Precision。这个
约束条件对“低基础率、far 状态占绝对多数”的闭环分布过松。

---

## 6. Train 600 的 1,394 个正样本分布

### 6.1 总体与 Episode 集中度（统计）

| 项目 | 数值 |
|---|---:|
| 正样本总数（原始 / move-valid） | 1,394 / 1,382 |
| 正样本 Episode | 43 / 600 |
| 每 Episode 正样本数：min / P25 / 中位 / P75 / P95 / max | 1 / 21 / **24** / 30 / 46.7 / **385** |
| 每 Episode 正样本数：mean ± std | 32.42 ± 55.54 |
| top-1 Episode 占比 | **27.62%**（Episode 299：385 个） |
| top-3 / top-5 / top-10 占比 | 34.43% / 40.53% / **53.01%** |
| Gini | 0.433 |
| 覆盖 50% / 80% / 90% 正样本所需 Episode 数 | 9 / 25 / 31 |
| 正样本 Episode 数直方图 | `1:2, 2–4:2, 5–9:2, 10–19:4, 20–49:32, 50–99:0, 100+:1` |

**按 band：**

| Band | 正样本数 | 正样本 Episode |
|---|---:|---:|
| short | 812 | 21 |
| medium | 355 | 12 |
| long | 227 | 10 |

**Top Episode（统计）：**

| Episode | band | states | positives | target_visible states |
|---:|---|---:|---:|---:|
| 299 | short | 400 | **385** | **15** |
| 14 | medium | 192 | 48 | 144 |
| 396 | medium | 112 | 47 | 65 |
| 318 | medium | 249 | 44 | 182 |
| 500 | short | 91 | 41 | 41 |

Episode 299 是一个极端样本：400 步中 385 步 near-inv，目标只可见 15 次；
它单独贡献了 27.6% 的正样本。**（推断，未验证）** 这可能让探针学到
“Episode 级场景/轨迹特征”而不是“状态级丢失事件”，但本轮没有做
去 Episode 299 的重训消融，不能当作已证实结论。

### 6.2 距离段与连续不可见时长（统计）

| 项目 | 数值 |
|---|---:|
| 正样本距离：中位 / 均值 / P10 / P90 | 71.29 / 64.04 / 17.41 / 88.08 m |
| 距离段 0–20 / 20–50 / 50–100 m | 149 / 225 / **1,020**（73.2% 在 50–100 m） |
| 连续 near-inv run 数 | 55 |
| run 长度：中位 / 均值 / P75 / P90 / max | **23** / 25.35 / 26.5 / 36.6 / **385** |
| run 长度 ≥ 5 / ≥ 10 / ≥ 20 的比例 | 70.9% / 65.5% / **60.0%** |
| run 长度分布 | `1:0, 2–4:9, 5–9:7, 10–19:3, 20–49:3, 50–99:32, 100+:1` |

**最长的 5 个 run（统计）：**

| Episode | 长度 | 起始距离 | 结束距离 |
|---:|---:|---:|---:|
| 299 | 385 | 93.06 m | 88.08 m |
| 396 | 47 | 74.15 m | 89.28 m |
| 318 | 44 | 2.32 m | 98.63 m |
| 14 | 42 | 29.47 m | 77.52 m |
| 500 | 41 | 1.47 m | 99.63 m |

**“连续不可见时长”在本数据里是控制步数**；数据没有记录每步墙钟时间，
`history_dt` 恒为 1.0，所以不能换算成秒。这一点在 §12 标注为缺失。

### 6.3 最后一次高置信度观测的时间间隔（统计）

定义：`since_localizable = 当前步 − 同一 Episode 内最近一次
target_localizable=1 的步号`；若从未出现过则为 `never`。

| 项目 | Train（1,394） | Dev Val（749） |
|---|---:|---:|
| 中位 | **19 步** | **19 步** |
| 均值 | 64.13 | 25.74 |
| P25 / P75 / P90 / max | 9 / 43 / 245.7 / 385 | 10 / 33 / 62 / 104 |
| ≤ 2 步的比例 | **94 / 1,394 = 6.7%** | **44 / 749 = 5.9%** |
| ≥ 10 步的比例 | 74.0% | 75.4% |
| ≥ 20 步的比例 | 48.3% | 47.9% |
| never | 0 | 0 |

分桶（Train / Val）：

| since_localizable | 0–1 | 2 | 3–4 | 5–9 | 10–19 | 20–49 | 50+ | never |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 0 | 50 | 124 | 189 | 358 | 337 | 336 | 0 |
| Dev Val | 0 | 23 | 59 | 102 | 206 | 252 | 107 | 0 |

**这是本轮最重要的分布事实：93.3%（Train）/ 94.1%（Val）的正样本，
其最近一次可定位观测在 2 步以前；中位数 19 步。P1-2 只输入 2 帧历史，
在绝大多数正样本上根本拿不到“最近可见”的证据。**

### 6.4 动作模式（统计）

| 动作/历史模式 | Train 正样本 | Dev Val 正样本 |
|---|---:|---:|
| 当前动作 FORWARD | 789（56.6%） | 450（60.1%） |
| 当前动作 RIGHT | 302（21.7%） | 144（19.2%） |
| 当前动作 LEFT | 291（20.9%） | 149（19.9%） |
| 当前动作 STOP | 12（0.9%） | 6（0.8%） |
| 历史 `FORWARD→FORWARD` | 643（46.1%） | 346（46.2%） |
| 历史 `RIGHT→LEFT` | 211（15.1%） | 91（12.2%） |
| 历史 `LEFT→RIGHT` | 209（15.0%） | 89（11.9%） |
| 历史 `RIGHT→RIGHT`（合成主模式） | 不在 top 5 | 不在 top 5 |

真实正样本以 **FORWARD 飞行中目标持续不可见**为主，不是“转向离开目标”
的合成模式。

---

## 7. 按 Episode 的配对分析：历史信息是否真的改善关键状态识别？

### 7.1 方法（统计）

- Dev Val 179 个 move-valid Episode，其中 22 个含正样本。
- 固定各自 Train-OOF 阈值（P0 0.0197754，P1-2 0.0143788）。
- 逐 Episode 计算正/负样本平均分数、Recall、AP、AUROC 的 P1-2 − P0。
- 按 Episode 重采样 1,000 次，计算全局 ΔAUROC/ΔAUPRC/ΔPrecision/ΔRecall
  的配对 bootstrap CI。
- Leave-one-positive-episode-out（LOEO）：依次排除 22 个正样本 Episode，
  重算全局 AUROC/AUPRC，检查是否由单一 Episode 主导。

### 7.2 Episode 级符号检验（统计）

| 比较 | P1-2 更好 / 更差 / 持平 | 双侧 p |
|---|---:|---:|
| 正样本平均分数 | 13 / 8 / 1 | 0.383 |
| 正样本 Recall | 6 / 0 / 16 | **0.031** |
| Episode AP | 14 / 7 / 1 | 0.189 |
| Episode AUROC | 12 / 9 / 1 | 0.664 |

Recall 的符号检验显著（6 个 Episode 提升、0 个下降、16 个持平），
但提升的绝对幅度很小（见下）；平均分数、AP、AUROC 的 Episode 级
差异都不显著。

### 7.3 配对 bootstrap（统计，1,000 次，按 Episode 重采样）

| Δ（P1-2 − P0） | 点估计 | bootstrap 均值 | 95% CI | P(Δ ≤ 0) |
|---|---:|---:|---:|---:|
| AUROC | +0.0105 | +0.0120 | **[-0.0181, +0.0449]** | **23.2%** |
| AUPRC | +0.0097 | +0.0110 | **[+0.0004, +0.0302]** | 2.3% |
| Precision @ 选中阈值 | +0.0350 | +0.0380 | **[+0.0076, +0.0911]** | 0.5% |
| Recall @ 选中阈值 | +0.0242 | +0.0250 | **[+0.0066, +0.0483]** | 0.0% |
| far-visible 误触发率 | +0.0027 | +0.0027 | [-0.0012, +0.0066] | 8.7% |
| far-invisible 误触发率 | -0.0019 | -0.0021 | [-0.0220, +0.0102] | 53.8% |
| near-visible 误触发率 | -0.0237 | -0.0226 | [-0.0738, +0.0019] | 63.7% |

**解读（推断）：**

- AUPRC 的增益在 Episode bootstrap 下 CI 刚好不跨 0（下限 +0.0004），
  但幅度只有 **+0.011**，绝对 AUPRC 仍只有 0.058。
- AUROC 的 95% CI 跨 0，**不能声称 AUROC 有统计上稳定的提升**。
- Precision/Recall 的增益 CI 不跨 0，但它们是**在各自阈值下**的；
  共同阈值对比（§4.3）显示排序确实略有改善，不是纯阈值移动。
- far-invisible 误触发率没有下降（CI 跨 0），near-visible 误触发率下降
  但 CI 也跨 0。历史信息主要压制了 near-visible 假触发（2.83% → 0.46%），
  但没有解决 far 假触发。

### 7.4 LOEO：是否由少数 Episode 主导？（统计）

| 项目 | ΔAUROC | ΔAUPRC |
|---|---:|---:|
| 排除 22 个正样本 Episode，逐一重算 | | |
| 最小 | +0.0017 | +0.0049 |
| 中位 | +0.0105 | +0.0099 |
| 最大 | +0.0235 | +0.0123 |
| 仍为正的排除次数 | **22 / 22** | **22 / 22** |

LOEO 说明：**没有哪一个正样本 Episode 单独制造了历史增益**；无论去掉
哪一个，ΔAUROC/ΔAUPRC 都保持为正。这是“历史增益不是单一 Episode
假象”的证据。但它只回答了“影响力/稳健性”，不回答“采样不确定性”；
后者的答案是 §7.3 的 bootstrap CI（AUROC 跨 0）。

### 7.5 近零分数 Episode（统计 + 推断）

22 个正样本 Episode 中，有 **10 个** P1-2 的正样本平均概率 < 1e-4
（7 个 short，3 个 medium）：

| Episode | band | positives | mean p (P1-2) | mean p negatives | AUROC |
|---:|---|---:|---:|---:|---:|
| 36 | short | 12 | 5.4e-06 | 4.3e-05 | 0.301 |
| 42 | short | 34 | 8.7e-08 | 3.6e-07 | 0.610 |
| 90 | short | 41 | 1.8e-10 | 1.8e-10 | 0.500 |
| 95 | short | 101 | 3.4e-09 | 3.8e-07 | **0.305** |
| 103 | short | 27 | 5.8e-07 | 1.7e-06 | 0.501 |
| 105 | short | 22 | 5.5e-10 | 1.5e-08 | **0.117** |
| 108 | short | 55 | 8.4e-10 | 3.1e-10 | 0.569 |
| 109 | medium | 26 | 5.5e-05 | 7.0e-06 | 0.651 |
| 153 | medium | 17 | 2.0e-05 | 1.5e-07 | 0.972 |
| 157 | medium | 26 | 9.7e-09 | 4.3e-10 | 0.877 |

其中 Episode 95（101 个正样本）和 105（22 个）的 AUROC 显著低于 0.5：
模型在这些 Episode 内不仅没有识别正样本，排序还反了。**（推断）**
这与 short band 的模型激活/校准失败一致；但本轮没有做特征 OOD 或
重训消融，不能确定根因是场景分布、Episode 级捷径还是 MLP 饱和。

### 7.6 历史到底在什么条件下有用？（统计）

| 条件（Dev Val 正样本） | n | P0 触发率 | P1-2 触发率 | Δ平均分数 |
|---|---:|---:|---:|---:|
| since_localizable 0–1 | 23 | 4.3% | 13.0% | +0.0009 |
| since_localizable 2 | 21 | 9.5% | 19.0% | +0.0055 |
| since_localizable 3–4 | 38 | 13.2% | 13.2% | **+0.0226** |
| since_localizable 5–9 | 102 | 12.7% | 14.7% | +0.0029 |
| since_localizable 10–19 | 203 | 5.9% | 9.9% | +0.0068 |
| since_localizable 20–49 | 250 | 4.0% | 5.6% | +0.0077 |
| since_localizable 50+ | 106 | 0.0% | 0.0% | -0.0002 |
| run length 5–19 | 68 | 2.9% | 5.9% | +0.0048 |
| run length ≥ 20 | 668 | 6.1% | 8.5% | +0.0064 |
| 历史 `FORWARD→FORWARD` | 345 | **11.0%** | **15.9%** | **+0.0138** |
| 历史 `RIGHT→LEFT` | 89 | 0.0% | 0.0% | -0.0003 |
| 历史 `LEFT→RIGHT` | 91 | 1.1% | 2.2% | -0.0002 |
| run start（run 第一步） | 27 | 11.1% | 18.5% | -0.0034 |
| run interior | 716 | 5.6% | 7.8% | +0.0065 |

**解读（推断）：**

- 历史增益最集中在 `FORWARD→FORWARD` 的 345 个正样本（+0.0138 平均分，
  触发率 11.0% → 15.9%）和 `since_localizable=3–4` 的 38 个正样本。
  这符合“历史帧里仍然保留最近可见信息”的机制。
- 但 68% 的正样本 `since_localizable ≥ 10`，其中 106 个 ≥ 50 步，
  这些状态的历史里没有任何高置信度观测，P1-2 的提升接近 0，触发率 0。
- 因此历史信息的作用是**局部的、条件性的**，不是对绝大多数真实近目标
  不可见状态的通用修复。

---

## 8. 阈值独立性审计

### 8.1 阈值来源（观测 + 审计）

- `probe_provenance.json` 明确记录 `threshold_source = "Train OOF only"`。
- 代码路径：`select_thresholds(train_data, train_score)`，其中
  `train_score` 是 5-fold Episode OOF 概率；Dev Val 分数只用于
  `real_metrics` 评测，没有传入阈值选择。
- 本轮用保存的 5 个 fold checkpoint 重建 OOF，重新执行
  `select_thresholds`，得到的阈值与保存值相差 < 6e-08。
- 外部模型（`*_external.pt`）是在全部 Train 600 上训练的；标准化
  mean/std 只来自 Train；Dev Val 不参与训练或标准化。
- `mask`/`pos_weight` 也只在 Train 上计算。
- 没有读取 Final Test 300；`probe_provenance.json` 的 `data` 只包含
  Train/Dev Val 目录。

### 8.2 Fold 类别比例（统计）

| Fold | 训练 states | 正样本 | 负样本 | 期望 pos_weight | checkpoint pos_weight | 差 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 61,935 | 1,276 | 60,659 | 47.5384 | 47.5384 | 0 |
| 1 | 64,888 | **726** | 64,162 | **88.3774** | 88.3774 | 0 |
| 2 | 59,690 | 1,078 | 58,612 | 54.3711 | 54.3711 | 0 |
| 3 | 65,372 | 1,223 | 64,149 | 52.4522 | 52.4522 | 0 |
| 4 | 66,551 | 1,225 | 65,326 | 53.3273 | 53.3273 | 0 |

外部模型 pos_weight = 56.6042（= 78,227 / 1,382），与代码一致。

**审计结论：** 没有类别比例计算错误；但 fold 1 只有 726 个正样本，
其 pos_weight 88.4 明显高于其他 fold（47.5–54.4）。OOF 阈值是在
这种不均匀 fold 上选的，阈值稳定性会受影响。这是设计限制，不是泄漏。

---

## 9. 合成数据 vs 真实闭环数据

### 9.1 真实数据（统计）

- Train 600：81,303 states，1,394 正样本，正例率 1.71%。
- 正样本距离中位 71.3 m，73.2% 在 50–100 m。
- `since_localizable` 中位 **19 步**；93.3% > 2 步。
- 连续不可见 run 中位 **23 步**；60% ≥ 20 步；最长 385 步。
- 当前动作：FORWARD 56.6%；历史模式 `FORWARD→FORWARD` 46.1%。
- Dev Val：749 正样本，22 个 Episode；`since_localizable` 中位 19；
  60+ 步占 14.3%；50–100 m 占 77.6%。

### 9.2 Stage-2g.1 合成 supplement（观测 + 设计文档）

- Train supplement：1,600 个 near-invisible 正样本（另有 47 个自然
  far-feature 正样本，共 1,647）。
- 生成方式：同一 Episode/位置/距离，做三次 RIGHT turn（15°/次）；
  near-invisible 变体在 t-3/t-2 可见、t 不可见；四个帧的相对角为
  20°/35°/50°/65°。
- 历史动作：**100% `RIGHT→RIGHT`**（`history_actions` 前两帧）。
- 距离：中位 40.25 m，均值 44.90 m，范围 5.13–100 m。
- 最后一次可见帧为 t-2，即当前正样本前 **2 步**，正好落在 P1-2 的
  2 帧历史窗口内。
- supplement 的 near-invisible 当前帧 `target_visible=0`、
  `target_localizable=0`（100%）。

### 9.3 关键差异（统计 + 推断）

| 维度 | Stage-2g.1 合成 | 真实 Train 600 | 影响 |
|---|---|---|---|
| 不可见持续时间 | 单步转换（可见→不可见） | 中位 23 步，60% ≥ 20 步 | 合成没有覆盖“持续丢失” |
| 最近高置信度观测 | 2 步前（在 2 帧历史内） | 中位 19 步，93.3% > 2 步 | 大多数真实正样本历史里没有可见证据 |
| 历史动作 | `RIGHT→RIGHT` 100% | `FORWARD→FORWARD` 46.1%；`RIGHT→RIGHT` 不在 top 5 | 合成把“转向丢失”当成主模式 |
| 当前动作 | RIGHT 转向后的状态 | FORWARD 56.6% | 真实是飞行中丢失 |
| 距离分布 | 5–100 m 近似均匀，中位 40.3 m | 中位 71.3 m，73.2% 在 50–100 m | 合成正样本偏近 |
| 正样本集中度 | 1,600 个受控 pair | 43 个 Episode，top1 27.6% | 真实 Episode 集中度高 |

### 9.4 合成模型在真实 Dev Val 上的表现（统计）

| 模型 | AUROC | AUPRC | near-inv flags @阈值 | near-inv max score | far flags |
|---|---:|---:|---:|---:|---:|
| Stage-2g.1 合成 P1-2（阈值 0.13695） | **0.5241** | 0.0300 | **0 / 743** | **0.0603** | 9 far-vis + 14 far-inv + 1 near-vis |
| 真实 Train 600 P1-2（阈值 0.01438） | 0.5922 | 0.0577 | 61 / 743 | 0.8890 | 362 far-vis + 183 far-inv + 19 near-vis |

合成模型在真实 near-inv 状态上的最高分只有 0.0603，低于它自己的阈值
0.13695；24 个 flags 全部是假阳性。KS（near-inv 分数分布，合成 vs
真实训练）为 **0.394**。这复制了 Stage-2g.2 的结论：合成 supplement
不迁移到真实 M0 闭环状态。

**推断：** 合成 probe 学到的是“最近刚转过弯、目标刚离开视野”的短时
特征，而真实 near-inv 是“长时间向前飞行、目标持续不可见、最近一次
可定位在 10–50 步前”。两者的观测与历史结构不同，不是简单加数据量
能解决的。

---

## 10. 泄漏与实现审计

### 10.1 标签与历史（观测）

对 Train 600 + Dev Val 180 共 780 个 Episode 逐状态检查：

| 检查 | Train | Dev Val |
|---|---:|---:|
| `near_target_invisible` 与 `distance<100 & target_visible==0` 不一致 | **0** | **0** |
| `target_localizable=1` 但 `target_visible=0` | 0 | 0 |
| `stop` 与 `action==STOP` 不一致 | 0 | 0 |
| near-inv 但 target_visible=1 | 0 | 0 |
| 负距离 | 0 | 0 |
| `history_valid` 与 `step>=2` 不一致 | 0 | 0 |
| t=0 历史非零 / 动作非 -1 / dt 非零 | 0 / 0 / 0 | 0 / 0 / 0 |
| t=1 `history_valid` 非 0 | 0 | 0 |
| t=1 frame0（t-1）与当前第 0 帧 compact 不一致 | 0 | 0 |
| t=1 frame1 非零 / 动作 1 非 -1 | 0 / 0 | 0 / 0 |
| t>=2 frame0/frame1 特征错位 | 0 | 0 |
| t>=2 动作错位 | 0 | 0 |
| t>=2 dt 非 1 | 0 | 0 |
| 特征/历史 NaN 或 Inf | 0 | 0 |

说明：t=1 的 frame0 会被 collector 填入 t-1 的 compact 特征，但
`history_valid[1]=0`，探针用 `history_valid>0` 过滤，所以不会把
t=1 的半历史当作有效历史。这是设计行为，不是跨 Episode 串接。

### 10.2 特征与划分（观测）

- 特征共 2,858 列，名称检查未命中任何禁止词
  （`distance/visible/visibility/half_size/goal_px/goal_py/slant/heading/
  oracle/arrival_label/sample_type/band/episode_id/step_id`）。
- 特征只由 current tokens、goal tokens、Goal-Current heatmap、
  baseline/coherent maps 计算；真实标签只存于旁路列，不进入特征。
- Train/Dev Val 的 seed 交集 **0**；network_layout_seed 交集 **0**；
  (seed, layout) pair 交集 **0**。
  - Train seed 范围 1,000,000–1,000,720（600 个唯一）
  - Dev Val seed 范围 2,000,000–2,000,203（180 个唯一）
  - Train layout 范围 17,820,000–17,820,599
  - Dev Val layout 范围 34,820,000–34,820,179
- Episode ID 数值空间重叠（Train 0–599，Val 0–179），但它们是不同
  split 的不同 Episode；本分析从不按裸 `episode_id` 跨 split 连接。
- Final Test 300：未读取；`probe_provenance.json` 的数据目录只有
  Train/Dev Val；本分析的输入参数也只有这两个目录。

### 10.3 审计结论

**未发现标签泄漏、历史顺序错误、跨 Episode 历史串接或类别比例错误。**
探针失败不是实现 bug 导致的；用保存的 checkpoint 推理可以精确复现
保存的指标，说明权重加载和特征顺序也是正确的。

---

## 11. 失败原因分析

以下按“证据 → 推断”排列。

### 11.1 绝对分离度不足（证据充分）

- Dev Val 基础正例率 2.845%；P1-2 正类平均概率 0.01422，负类 0.00255，
  比值 **5.58×**。
- 要在 2.845% 基础率下达到 Precision 0.5，近似需要的正负似然比约
  **34×**；当前差了一个数量级。
- Brier 0.02857 差于全零基线 0.02845（skill -0.0045）；AUPRC 仅为
  基础率的 2.03 倍。
- 在选中阈值下 TP 61 / FP 564；即使把 Recall 提到 50%（
  `recall_0.5_unconstrained` 工作点），Precision 仍只有 0.0427，
  far-visible/far-invisible 误触发率高达 30%–45%。

### 11.2 大多数正样本不在历史窗口内（证据充分）

- 93.3% 的 Train 正样本、94.1% 的 Dev Val 正样本 `since_localizable > 2`；
  中位 19 步。
- P1-2 只输入 t-1/t-2 两帧 compact 特征；在这些正样本上，最近一次
  可定位观测不在输入里。
- 历史增益集中在 `since_localizable=3–4`、`FORWARD→FORWARD` 的
  小部分样本；对 ≥10 步的 68% 正样本，提升很小，≥50 步的 106 个
  正样本触发率为 0。

### 11.3 真实不可见是持续状态，合成数据是单步转换（证据充分）

- 真实 run 中位 23 步，60% ≥ 20 步；最长 385 步。
- 合成 supplement 是“三次 RIGHT 转后当前帧刚离开视野”，历史动作
  100% `RIGHT→RIGHT`，最近可见帧在 2 步前。
- 在真实数据上，合成模型的 near-inv 最高分 0.0603 < 阈值 0.13695，
  0/743 触发；AUROC 0.5241，接近随机。
- 真实训练的 P1-2 虽然提升到 0.5922，但这是在同一真实分布上重新
  拟合的结果，仍然没有解决绝对分离度。

### 11.4 短 band 的状态级识别失败（证据充分）

- short band 占 Dev Val 正样本 60.0%，AUROC 0.4525/0.4581，低于随机；
  medium 0.75/0.76；long 0.95 但只有 1 个正样本 Episode。
- 10/22 正样本 Episode 的 P1-2 正样本平均概率 < 1e-4；其中 7 个 short。
- 因此“总体 AUROC 0.592”不能外推为“短距离近目标状态可识别”。

### 11.5 训练集的 Episode 集中度（证据 + 未验证假设）

- Episode 299 单独贡献 27.6% 正样本；top-10 Episode 贡献 53.0%；
  Gini 0.433。
- Episode 299 是 400 步、385 个正样本、仅 15 个可见状态的极端样本。
- **（未验证假设）** 模型可能部分学到了该 Episode 的场景/轨迹级
  特征，而不是状态级“目标丢失”特征；这可以解释 short band 在
  其他 Episode 上接近随机。要验证需要做“去掉 Episode 299 重训”
  的消融；本轮按要求没有训练。

### 11.6 Gate 的误触发率定义不适合低基础率（证据充分）

- far 状态 21,236 个，占 81.3%；2%–3% 的 far 误触发率就是 500+ FP。
- 逐象限 5% 的约束通过，但绝对假阳性预算失败；Precision 完全不可用。
- 这说明 gate 需要改成“绝对 FP 预算/每 Episode 误触发次数”，而不是
  仅看每个象限的比例。

---

## 12. 明确判断：是否值得继续优化探针？

### **判断：No — 不值得继续优化当前 P0/P1-2 探针。**

理由（按重要性）：

1. **瓶颈不是模型容量，而是标签与观测不匹配。** 当前标签是“当前
   距离 <100 m 且不可见”，真实数据中它主要对应“持续不可见 20–50 步、
   最近一次可定位在 19 步前”的状态；2 帧历史没有信息，P1-2 只能学到
   场景/距离/动作的弱相关。
2. **总体指标掩盖了关键子集失败。** short band（60% 正样本）AUROC
   0.45–0.46；10/22 正样本 Episode 近零分数；medium 是唯一有信号的
   band，但只有 7 个正样本 Episode。
3. **历史增益虽真实但太小。** ΔAUROC 95% CI 跨 0；ΔAUPRC +0.011；
   在共同阈值下 Precision 仍只有 0.10–0.15。继续加帧数/特征可能会
   再挤出 1–2 个点，但不可能达到 Precision 0.5 的 gate。
4. **概率不可用于闭环门控。** Brier 差于全零基线；AUPRC lift 2.03×；
   选中阈值下 564 FP vs 61 TP。任何基于该概率的 Recovery 触发器都会
   被远距离假阳性淹没。
5. **合成数据路线已经在 Stage-2g.2 被闭环证伪。** 合成模型在真实
   near-inv 上 0/743 触发；真实数据重训只把 AUROC 从 0.524 提到 0.592。
   继续用同类合成数据补样本不会改变结构性问题。

**因此：**

- **不要**接入 Recovery 控制器；
- **不要**训练完整 MEC；
- **不要**开始基于该探针的超参/架构搜索；
- **不要**读取 Final Test 300；
- 保留 `target_visible` / `target_localizable` 头作为通用诊断信号可以，
  但**不能**作为“近目标定位失败需要恢复搜索”的触发器。

### 若还要继续这条线（建议，本轮未执行）

1. **重定义标签：** 从静态 `distance<100 & invisible` 改成事件标签，
   例如“最近 k 步内 `target_localizable=1`，当前 `target_visible=0`”，
   或“目标从可定位到不可定位的转换”，并显式记录 `last_seen_step`、
   `last_localizable_step`、丢失持续时间。
2. **重定义输入：** 不要只给 2 帧 compact 特征；至少给“最近一次
   可定位观测的相对时间/距离/动作”或一个显式的 last-seen token；
   当前 2 帧历史覆盖不到中位 19 步的时间尺度。
3. **重定义数据：** 采集真实闭环中的“丢失事件”正样本，按
   `since_localizable`、run length、band、动作模式分层；不要只用
   synthetic turn-away。
4. **重定义 gate：** 用绝对假阳性预算（每 Episode 或每 1,000 states）、
   分 band 的 Precision/Recall、以及 Episode 级配对指标；不要只看
   总体 AUROC 或逐象限 5%。
5. **多 seed + 去极端 Episode 消融：** 至少排除 Episode 299 这类
   top-1 占比 27.6% 的样本，确认不是 Episode 级捷径。

---

## 13. 缺失项与限制

1. `probe_results.json` 没有保存逐状态概率；本轮用保存的 checkpoint
   做纯推理恢复，并精确复现所有聚合指标。逐状态预测保存在
   `analysis_.../predictions.npz`，未提交到 git。
2. 连续不可见时长、`since_localizable` 的单位都是**控制步**；数据没有
   记录每步墙钟时间，`history_dt` 恒为 1.0，不能换算成秒。
3. 合成 supplement 的历史帧没有逐帧 `target_visible/localizable` 标签；
   其“最近可见在 2 步前”来自生成设计（代码/报告），不是逐状态测量。
4. 单 seed（seed 0）、单 MLP 架构（128 hidden，30 epochs）；没有
   multi-seed 复现。
5. 本轮没有做任何重训/消融；“Episode 299 导致 Episode 级捷径”是
   未验证假设。
6. 本轮没有读取 Final Test 300，也没有修改 M0/M1 checkpoint。
7. Dev Val 只有 22 个正样本 Episode、179 个 move-valid Episode；
   所有 Episode 级 CI 都很宽。

---

## 14. 复现命令与提交内容

### 14.1 重新生成本轮分析（纯推理，不训练）

```bash
python v83_spatial_stage2/stage2g3_analyze.py \
  --train-dir runs/v83_spatial_stage2/stage2g3/real_train_20261019T132500Z \
  --val-dir runs/v83_spatial_stage2/stage2g3/real_val_20261019T132000Z \
  --probe-dir runs/v83_spatial_stage2/stage2g3/probe_20261019T133500Z \
  --output-dir runs/v83_spatial_stage2/stage2g3/analysis_<stamp> \
  --bootstrap 1000 --device cpu
```

### 14.2 测试

```bash
python -m unittest -v \
  v83_spatial_stage2.test_stage2g3 \
  v83_spatial_stage2.test_stage2g3_analysis
```

本轮结果：13 tests OK。

### 14.3 提交内容

本轮应提交（详见聊天中的 commit/push 状态）：

- `v83_spatial_stage2/V83_STAGE2G3_REAL_ROLLOUT_PROBE_REPORT.md`
- `v83_spatial_stage2/stage2g3_analyze.py`
- `v83_spatial_stage2/test_stage2g3_analysis.py`
- 以及上一轮已生成但未提交的：
  `v83_spatial_stage2/stage2g3_collect.py`
  `v83_spatial_stage2/stage2g3_probe.py`
  `v83_spatial_stage2/test_stage2g3.py`

**不提交**：任何 `.pt`、`.npz`、`.json` 运行产物、缓存、日志；
它们保留在服务器 `runs/` 下，由 `.gitignore` 排除。

---

## Appendix A. 其他 head 的 Dev Val 指标（供参考，统计）

这些 head 不是本轮 trigger 的主 head，但可用于判断“可见/可定位”信号是否
能替代 near-inv head。

| Head | 模型 | AUROC | AUPRC | Brier | ECE | P@0.5 | R@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|
| target_visible | P0 | 0.6719 | 0.7393 | 0.2649 | 0.2609 | 0.700 | 0.984 |
| target_visible | P1-2 | 0.6834 | 0.7507 | 0.2555 | 0.2545 | 0.706 | 0.983 |
| target_localizable | P0 | 0.6858 | 0.5594 | 0.2933 | 0.2505 | 0.576 | 0.818 |
| target_localizable | P1-2 | 0.7047 | 0.5776 | 0.2912 | 0.2535 | 0.578 | 0.858 |

注意：`target_visible` 在 near-invisible 子集上的平均概率仍有
0.693（P0）/0.711（P1-2），68.5%/71.9% 的 near-inv 状态被判为可见；
`target_localizable` 在 near-inv 子集上的平均概率约 0.49/0.50。
两者都不能作为“目标不可见/不可定位”的绝对门控。

### 分 Band（P1-2）

| Head | Band | AUROC | AUPRC | Brier | ECE | P@0.5 | R@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|
| target_visible | short | 0.709 | 0.762 | 0.220 | 0.240 | 0.724 | 1.000 |
| target_visible | medium | 0.652 | 0.643 | 0.336 | 0.335 | 0.617 | 0.971 |
| target_visible | long | 0.703 | 0.847 | 0.206 | 0.192 | 0.779 | 0.979 |
| target_localizable | short | 0.819 | 0.815 | 0.163 | 0.130 | 0.797 | 0.904 |
| target_localizable | medium | 0.629 | 0.494 | 0.349 | 0.312 | 0.528 | 0.794 |
| target_localizable | long | 0.707 | 0.479 | 0.353 | 0.346 | 0.409 | 0.867 |

---

## Appendix B. 关键产物索引

```
runs/v83_spatial_stage2/stage2g3/
├── real_train_20261019T132500Z/          # 600 episode npz + manifest
├── real_train_20261019T132500Z.log
├── real_val_20261019T132000Z/            # 180 episode npz + manifest
├── probe_20261019T133500Z/
│   ├── probe_results.json
│   ├── probe_provenance.json
│   ├── reproduction_check.json
│   ├── tables.md
│   └── models/                           # 36 checkpoints
└── analysis_20261019T054458Z/
    ├── analysis.json                     # 本报告全部派生指标
    ├── predictions.npz                   # 逐状态 p0/p1/synthetic
    └── ../analysis_20261019T054458Z.log
```

---

## Appendix C. 术语

- **P0**：仅当前 2,858 维特征。
- **P1-2**：当前特征 + t-1、t-2 两帧 compact 历史（74 维视觉 +
  4 维动作 one-hot + 1 维 dt，每帧 79 维，共 +158 维）。
- **near-target-invisible**：真实距离 < 100 m 且 `target_visible==0`。
- **run**：同一 Episode 内连续的 near-target-invisible 状态段。
- **since_localizable**：当前步到同一 Episode 最近一次
  `target_localizable=1` 的步数。
- **Dev Val 180**：180 个 spec Episode；move-valid 评测为 179 个。
- **Train 600**：600 个采集 Episode；move-valid 评测为 587 个。
- **Train OOF**：5-fold Episode-level out-of-fold 预测；阈值只在这里选。
