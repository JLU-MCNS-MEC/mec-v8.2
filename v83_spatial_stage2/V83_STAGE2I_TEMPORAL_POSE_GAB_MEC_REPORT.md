# V8.3 Stage-2i — Temporal / Pose / GAB MEC Ablations (T0–T5)

日期：2026-10-19

分支：`experiment/v83-spatial-foundation`

性质：**六组独立 Transformer-M MEC 输入消融 + 完整 Dev Val 180 闭环评估**。
单随机种子、开发验证；不是最终泛化结论。Final Test 300 未读取。

---

## 0. TL;DR

1. **T5（map-assisted）是唯一的大幅正向模型**：SR **57.22%**、
   OSR **85.56%**、SPL **0.5722**、nDTW **0.5575**；
   vs 本轮 T0：SR **+36.67 pp**（95% CI **[+27.78, +45.56]**），
   SPL +0.3667、nDTW +0.3124，Short/Medium/Long 全部显著改善。
   但它是匿名全局地标图辅助，不能解释为纯视觉能力提升。
2. **T4（history + pose + GAB）明确有害**：SR **5.56%**；
   vs T3：SR **−12.78 pp**（CI **[−18.89, −6.67]**）、
   SPL −0.1278、nDTW −0.1748、premature **+48.33 pp**。
   尽管 T4 的离线 macro-F1 **0.8513** 为 map-free 最高，
   闭环动作坍缩为 **98.5% Forward**、停止距离中位数 334.5 m：
   典型的离线/on-policy 错位。
3. **T1（network only）是唯一在 SR/SPL/nDTW 三项上显著优于 T0 的 map-free 模型**：
   SR **32.78%**（+12.22 pp，CI [+4.44, +20.00]）、
   SPL +0.1221、nDTW +0.1061；但 Short/Long 改善、Medium 持平，
   且仅约等于归档 80-epoch M0（34.44%，−1.67 pp）。
4. **T2/T3 相对 T0 没有可靠的 SR/SPL 增益**：
   T2 +3.89 pp（CI 跨 0）、T3 −2.22 pp（CI 跨 0）；
   T2 仅 nDTW +0.0587（CI [+0.0019, +0.1160]）勉强为正。
5. **Heatmap 分支的机制副作用**：heatmap 进入 Arrival head 使离线
   far>100 m 误停率升高约 **10 倍**（T0/T2 0.52–0.57% vs T1/T3/T4 0.046%、T5 0.056%），
   闭环停止距离中位数 T0 140.7 m、T2 129.1 m vs T1 22.2 m。
   T0 与 T1 动作分布几乎相同，差别主要是停止时机。
6. **50-epoch 预算显著压低 T0**：归档 80-epoch M0 用本轮 evaluator
   复现为 **34.44% SR**（与 Stage-2h 完全一致），本轮 T0 为 **20.56%**。
   主分析 vs 新 T0 在协议内公平，但绝对水平对预算敏感；
   map-free 模型中没有模型超过归档 M0，只有 T5 明显超过。
7. **单 seed、Dev Val 180**；Dev Val 同时用于 checkpoint selection 与
   闭环评估，存在开发集选择偏差；不能当作最终泛化结论。
8. **决策**：不部署 T4/当前 history-GAB；T5 仅作为 map-assisted 结果；
   map-free 侧不宣称可靠增益。下一步应先做多 seed、把 map 接入
   Arrival head、并修复 T4 的 on-policy 动作坍缩，再谈控制器集成。

---

## 0.1 协议摘要

| 项目 | 固定值 |
|---|---|
| 模型 | Transformer-M（hidden 256, 2 layers, 8 heads, ff×4） |
| 训练集 | Train 600（200 Short / 200 Medium / 200 Long） |
| 开发验证 | Dev Val 180（60 / 60 / 60） |
| Epoch 上限 | 50 |
| Early Stopping | patience 8 |
| Batch size | 48 |
| Learning rate | 6e-5 |
| Optimizer | AdamW, weight decay 1e-4 |
| LR schedule | ReduceLROnPlateau factor 0.5, patience 3, min 1e-6 |
| Seed | 20264017 |
| Shared RGB encoder | 冻结（token 预先计算，不在模型中） |
| Loss | 原始 MEC Loss：movement CE（soft class weights）+ Arrival BCE，**无 pos_weight** |
| Checkpoint score | 0.7×val macro-F1 + 0.3×mean(natural, hard) Arrival-F1 |
| Closed-loop Stop threshold | 0.7 |
| 历史长度 | 16 步 |
| GAB | 8-bin egocentric bearing belief, 20-D |
| T5 map | anonymous global landmark retrieval, Top-5, 22-D |

### 六组定义

| 组 | 输入 |
|---|---|
| T0 | Current RGB + Goal RGB + Heatmap + Network State 5D（原始 M0） |
| T1 | Current RGB + Goal RGB + Network State 5D（无 Heatmap 分支） |
| T2 | Current RGB + Goal RGB + Heatmap（无 Network 分支） |
| T3 | Current RGB + Goal RGB（无 Heatmap / Network 分支） |
| T4 | T3 + 16 步历史视觉 + UAV 相对位姿历史 + GAB |
| T5 | T4 + Anonymous Global Landmark Retrieval Graph（map-assisted） |

T1/T2/T3 的分支是**结构性删除**（模块、投影、token 全部移除），不是置零。
T3/T4/T5 不包含 heatmap_net / heat_proj；GAB 只以 20-D belief 进入模型，
不携带原始 3×8×8 Heatmap。T4/T5 的 Arrival head 保持 T3 的可观测
match-feature 定义，因此停止行为可跨 T3–T5 比较。

### 执行顺序

`T5 → T4 → T3 → T2 → T1 → T0`。每个模型训练完成后立即评估完整
Dev Val 180，写入逐 Episode 结果并更新 `progress.json`；不先训练完六组
再统一评估。所有模型从同一固定初始化策略独立训练，不复用彼此权重。

---

## 1. 数据隔离

| Split | Episodes | Samples | Bands | Episode seeds | Network seeds |
|---|---:|---:|---|---|---|
| Train 600 | 600 | 55,565 | 200/200/200 | 1,000,000–1,000,720 | 17,820,000–17,820,599 |
| Dev Val 180 | 180 | 17,234 | 60/60/60 | 2,000,000–2,000,203 | 34,820,000–34,820,179 |
| Final Test 300 | 300 | 26,355 | 100/100/100 | 3,000,000+（SEALED） | 51,820,000+ |

- 三个 split 的 episode seed / network layout seed 互不重叠。
- Final Test 目录带 `SEALED.json`，本轮代码所有入口都会拒绝包含
  `final_test` 的路径；本轮**未读取、未评估、未调参**。
- 公共准备只执行一次：
  `runs/v83_spatial_stage2/stage2i/aux/`（1.8 GB sidecar），
  六组训练共用，未重复生成数据或地图缓存。

---

## 2. 公共准备与测试

1. CPU 单元测试：`python -m unittest -v v83_spatial_stage2.test_stage2i`
   → **15 tests OK**。
2. 六组单 Batch 前向/反向：全部有限 loss，关键分支梯度非零。
3. 六组 2-Episode 闭环 Smoke：全部完成，无接口错误。
4. Aux 对齐：对 Train/Dev Val 各抽取 Episode 独立重放，history / pose /
   GAB / map 特征与 sidecar **逐元素差值 0.0**；pose replay 的
   action/step mismatch 均为 0。
5. 模型参数：

| Model | Parameters | Trainable |
|---|---:|---:|
| T0 | 2,722,164 | 2,722,164 |
| T1 | 2,632,804 | 2,632,804 |
| T2 | 2,710,164 | 2,710,164 |
| T3 | 2,620,804 | 2,620,804 |
| T4 | 3,027,588 | 3,027,588 |
| T5 | 3,033,732 | 3,033,732 |

T0 与原始 `ExplicitReasoningTransformer` 的 state-dict key/shape/value
完全一致（test 验证），forward 输出逐元素一致。

---

## 3. 历史 / 位姿 / GAB 的实现

### 3.1 历史视觉

- 每一步取冻结 encoder 的 576×384 Current RGB tokens；
- `mean` + `max` pooling → 768-D visual descriptor（训练 sidecar 为 float16，
  与归档 precision 一致）；
- 16 步历史按时间顺序进入 2 层 128-D Transformer；
- 缺失槽位 mask=0；Episode reset 清空；
- 只使用当前与过去观测，不使用未来帧。

### 3.2 UAV 相对位姿

每步历史槽位 10-D：

- 相对位移：机体前向 / 右向位移（以 5 m 归一化）；
- 航向变化：Δyaw（以 15° 归一化）；
- 历史增量：该历史状态相对当前状态的前向 / 右向位移（100 m 归一化）
  与 Δyaw（π 归一化）；
- 实际执行动作 one-hot（4-D）。

训练位姿由归档 Episode 的 seed/band + 存储 action 精确重放得到；
重放 action 与存储 label 100% 一致。禁止输入目标坐标、真实目标距离、
真实目标方位、Oracle Action、成功标签。

### 3.3 GAB

复用 Stage-2h 的因果更新规则：

- 当前证据来自原生 3×8×8 Goal-Current heatmap；
- 可靠阈值 = Train natural heatmap confidence 的 60th percentile
  = **0.0928975**；
- LEFT/RIGHT 按 15°/bin 旋转 prior；FORWARD 不做固定像素平移；
- 证据不可靠时向 uniform 衰减，age=16 后 valid=0；
- Episode reset 清空；
- 输出 20-D：8-bin belief + 8-bin last-reliable + confidence + entropy +
  age + valid。

GAB 的输入是 heatmap-derived evidence，但进入 MEC 的只有 20-D belief；
T4/T5 模型内部不存在 heatmap 分支，也没有原始 3×8×8 张量。

---

## 4. 结果（Dev Val 180）

### 4.1 总体

| Model | Inputs | Params | SR | OSR | SPL | nDTW | Premature | Timeout | Steps | Best Epoch | Train min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | Current + Goal + Heatmap + Network | 2,722,164 | 20.56% | 24.44% | 0.2056 | 0.2451 | 38.89% | 40.56% | 224.1 | 47 | 23 |
| T1 | Current + Goal + Network | 2,632,804 | **32.78%** | **40.56%** | **0.3277** | **0.3512** | 34.44% | 32.78% | 206.5 | 50 | 25 |
| T2 | Current + Goal + Heatmap | 2,710,164 | 24.44% | 28.33% | 0.2444 | 0.3038 | 42.22% | 33.33% | 206.9 | 50 | 23 |
| T3 | Current + Goal | 2,620,804 | 18.33% | 24.44% | 0.1833 | 0.2743 | 27.22% | 54.44% | 268.8 | 50 | 22 |
| T4 | T3 + History + Pose + GAB | 3,027,588 | 5.56% | 14.44% | 0.0556 | 0.0996 | 75.56% | 18.89% | 181.6 | 31 | 20 |
| T5 | T4 + Landmark Graph | 3,033,732 | **57.22%** | **85.56%** | **0.5722** | **0.5575** | 32.78% | 10.00% | 135.6 | 24 | 17 |

### 4.2 分 Band（每 Band 60 Episode）

| Model | Band | SR | OSR | SPL | nDTW |
|---|---|---:|---:|---:|---:|
| T0 | short | 26.67% | 35.00% | 0.2667 | 0.3758 |
| T0 | medium | 26.67% | 28.33% | 0.2667 | 0.2680 |
| T0 | long | 8.33% | 10.00% | 0.0833 | 0.0915 |
| T1 | short | **41.67%** | 48.33% | 0.4167 | 0.5252 |
| T1 | medium | 26.67% | **40.00%** | 0.2663 | 0.2987 |
| T1 | long | **30.00%** | **33.33%** | 0.3000 | 0.2299 |
| T2 | short | 31.67% | 36.67% | 0.3167 | 0.4095 |
| T2 | medium | 26.67% | 30.00% | 0.2667 | 0.3282 |
| T2 | long | 15.00% | 18.33% | 0.1500 | 0.1737 |
| T3 | short | 25.00% | 28.33% | 0.2500 | 0.4095 |
| T3 | medium | 16.67% | 30.00% | 0.1667 | 0.2319 |
| T3 | long | 13.33% | 15.00% | 0.1333 | 0.1816 |
| T4 | short | 10.00% | 18.33% | 0.1000 | 0.1605 |
| T4 | medium | 6.67% | 21.67% | 0.0667 | 0.1045 |
| T4 | long | 0.00% | 3.33% | 0.0000 | 0.0338 |
| T5 | short | **61.67%** | **86.67%** | **0.6167** | **0.6515** |
| T5 | medium | **50.00%** | **85.00%** | **0.5000** | **0.4807** |
| T5 | long | **60.00%** | **85.00%** | **0.6000** | **0.5402** |

### 4.3 逐 Episode 配对 vs T0

Bootstrap 按 Episode 重采样，95% percentile CI；W/L/T 以 T0 为 base。

| Model | ΔSR | 95% CI | W/L/T | ΔOSR | ΔSPL | ΔSPL 95% CI | ΔnDTW | ΔnDTW 95% CI | ΔPremature | ΔTimeout |
|---|---:|---|---:|---:|---:|---|---:|---|---:|---:|
| T1 | **+12.22 pp** | **[+4.44, +20.00]** | 38/16/126 | +16.11 pp | **+0.1221** | **[+0.0443, +0.1998]** | **+0.1061** | **[+0.0502, +0.1623]** | −4.44 pp | −7.78 pp |
| T2 | +3.89 pp | [−3.89, +11.67] | 30/23/127 | +3.89 pp | +0.0389 | [−0.0389, +0.1167] | +0.0587 | [+0.0019, +0.1160] | +3.33 pp | −7.22 pp |
| T3 | −2.22 pp | [−10.00, +5.56] | 23/27/130 | +0.00 pp | −0.0222 | [−0.1000, +0.0556] | +0.0292 | [−0.0308, +0.0889] | −11.67 pp | +13.89 pp |
| T4 | **−15.00 pp** | **[−21.67, −8.33]** | 7/34/139 | −10.00 pp | **−0.1500** | **[−0.2167, −0.0833]** | **−0.1455** | **[−0.2084, −0.0833]** | **+36.67 pp** | −21.67 pp |
| T5 | **+36.67 pp** | **[+27.78, +45.56]** | 79/13/88 | +61.11 pp | **+0.3667** | **[+0.2778, +0.4556]** | **+0.3124** | **[+0.2437, +0.3822]** | −6.11 pp | −30.56 pp |

- **T1 是唯一在 SR、SPL、nDTW 三个指标上同时显著改善的 map-free 模型**
  （三个 CI 下限均 > 0）。
- **T5 三项也同时显著改善**，但它是 map-assisted，必须单独解释。
- T2 只有 nDTW 的 CI 下限略高于 0（+0.0019），SR/SPL 不显著。
- T4 三项同时显著退化，premature-stop 大幅增加。
- T3 相对 T0 无显著差异。

### 4.4 分 Band 配对 vs T0（SR）

| Model | Short ΔSR | 95% CI | Medium ΔSR | 95% CI | Long ΔSR | 95% CI |
|---|---:|---|---:|---|---:|---|
| T1 | **+15.00 pp** | **[+1.67, +28.33]** | 0.00 pp | [−13.33, +13.33] | **+21.67 pp** | **[+10.00, +35.00]** |
| T2 | +5.00 pp | [−8.33, +18.33] | 0.00 pp | [−15.00, +15.00] | +6.67 pp | [−5.00, +18.33] |
| T3 | −1.67 pp | [−16.67, +13.33] | −10.00 pp | [−23.33, +3.33] | +5.00 pp | [−5.00, +15.00] |
| T4 | **−16.67 pp** | **[−30.00, −3.33]** | **−20.00 pp** | **[−33.33, −8.33]** | **−8.33 pp** | **[−15.00, −1.67]** |
| T5 | **+35.00 pp** | **[+20.00, +50.00]** | **+23.33 pp** | **[+6.67, +40.00]** | **+51.67 pp** | **[+36.67, +65.00]** |

T1 的改善集中在 Short 与 Long，**Medium 完全持平**；T5 三个 Band
都显著改善，Long 最大；T4 三个 Band 都退化。

### 4.5 关键二级比较

| Comparison | ΔSR | 95% CI | W/L/T | ΔSPL | ΔnDTW |
|---|---:|---|---:|---:|---:|
| T4 − T3（history + pose + GAB 联合效应） | **−12.78 pp** | **[−18.89, −6.67]** | 7/30/143 | −0.1278 | −0.1748 |
| T5 − T4（map 效应；T4 已崩溃） | **+51.67 pp** | **[+43.89, +59.44]** | 94/1/85 | +0.5167 | +0.4579 |
| T5 − T1（map-assisted vs 最佳 map-free） | **+24.44 pp** | **[+15.56, +33.89]** | 62/18/100 | +0.2446 | +0.2062 |

- T4 相对 T3 显著退化，且 nDTW 也显著退化（CI [−0.2353, −0.1147]）；
  这是本轮最明确的负结果。
- T5 相对 T4 的巨大提升混入了“map 修复了 T4 的 history-only 崩溃”这一因素，
  因此不能把 +51.67 pp 全部解释为 map 的边际价值。
- 更保守的 map-assisted 估计是 **T5 − T1 = +24.44 pp SR**
  （CI [+15.56, +33.89]），仍然显著为正。

### 4.6 离线指标与闭环结果的错位

| Model | Offline macro-F1 | Natural/Hard Arrival-F1 | Offline far100 FP | Offline near20 miss | Closed SR | OSR→SR gap |
|---|---:|---|---:|---:|---:|---:|
| T0 | 0.7023 | 0.6037 / 0.7022 | 0.5663% | 27.22% | 20.56% | 3.88 pp |
| T1 | 0.6830 | 0.5260 / 0.6392 | 0.0464% | 55.00% | 32.78% | 7.78 pp |
| T2 | 0.7103 | 0.6200 / 0.7070 | 0.5199% | 26.11% | 24.44% | 3.89 pp |
| T3 | 0.6839 | 0.5253 / 0.6497 | 0.0464% | 53.89% | 18.33% | 6.11 pp |
| T4 | **0.8513** | 0.5246 / 0.6440 | 0.0464% | 55.56% | **5.56%** | 8.88 pp |
| T5 | **0.8638** | 0.5273 / 0.6408 | 0.0557% | 54.44% | **57.22%** | **28.34 pp** |

- **T4 的离线 macro-F1 最高（0.8513），闭环 SR 最低（5.56%）**；
  T4 vs T3 的离线 macro-F1 是 +16.74 pp，闭环 SR 是 −12.78 pp。
  这是直接的离线/on-policy 错位，不能用户外验证 accuracy 替代闭环评估。
- T5 的离线 macro-F1 与 T4 接近（0.8638 vs 0.8513），闭环差异巨大
  （57.22% vs 5.56%）；离线指标无法区分两者，闭环 map 输入才是关键。
- T0/T2 的 offline far100 false-stop 比 T1/T3/T4/T5 高约 **10 倍**
  （0.52–0.57% vs 0.046%），对应 heatmap 进入 Arrival head。
  T0/T2 的 near20 miss 反而更低（26–27% vs 54–55%）：heatmap 同时
  改善近目标停止召回、恶化远距误停。

### 4.7 停止与动作分布（闭环）

| Model | Stop distance median | Forward 占比 | Premature | Timeout | SR | OSR |
|---|---:|---:|---:|---:|---:|---:|
| T0 | 140.7 m | 36.8% | 38.89% | 40.56% | 20.56% | 24.44% |
| T1 | 22.2 m | 37.1% | 34.44% | 32.78% | 32.78% | 40.56% |
| T2 | 129.1 m | 31.9% | 42.22% | 33.33% | 24.44% | 28.33% |
| T3 | 66.5 m | 21.9% | 27.22% | 54.44% | 18.33% | 24.44% |
| T4 | **334.5 m** | **98.5%** | **75.56%** | 18.89% | 5.56% | 14.44% |
| T5 | **18.2 m** | 68.9% | 32.78% | 10.00% | 57.22% | 85.56% |

- **T0 与 T1 的动作分布几乎相同**（Forward 36.8% vs 37.1%），
  但 T0 的停止距离中位数是 140.7 m、T1 是 22.2 m。
  T1 相对 T0 的闭环增益主要来自**停止时机**，不是移动方向：
  去掉 heatmap 分支后 Arrival head 的远距误停从 0.566% 降到 0.046%。
- **T4 的动作坍缩为 98.5% Forward**，停止距离中位数 334.5 m。
  历史/GAB 分支在专家轨迹上离线很好，但 on-policy 形成“继续前进”的
  自反馈，导致严重偏离目标。
- **T5 的停止距离中位数 18.2 m**，说明 map 把无人机带到目标附近；
  但 OSR 85.56% → SR 57.22% 仍有 28.34 pp 的转化损失，
  停止时机（32.78% premature + 10% timeout）是 map-assisted 的主要瓶颈。

### 4.8 T5 匿名地图检索

| 条件 | Top-1 | Top-5 | Top-1 错误数 |
|---|---:|---:|---:|
| clean | 100.00% | 100.00% | 0 |
| 5% embedding noise | 23.89% | 67.78% | 137 |
| 15% embedding noise | 10.56% | 38.89% | 161 |

Clean 检索与 Goal reference 共享同一渲染/冻结编码器，因此饱和；
噪声压力测试说明该 100% 不具备视觉鲁棒性含义。详见
`V83_STAGE2I_MAP_ASSISTED_DESIGN.md`。

---

## 5. 分析与决策

### 5.1 T5：唯一的大幅正向模型，但必须标记为 map-assisted

- T5 vs T0：SR **+36.67 pp**（CI [+27.78, +45.56]），
  SPL **+0.3667**，nDTW **+0.3124**，三个 Band 全部显著改善，
  Long 最大（+51.67 pp）。
- T5 vs 最佳 map-free T1：SR **+24.44 pp**（CI [+15.56, +33.89]），
  SPL +0.2446，nDTW +0.2062；这是更保守的 map 效应估计。
- T5 的 OSR 85.56% → SR 57.22%：移动/可达性由 map 解决，
  剩余瓶颈是 Arrival/Stop 时机。
- **不能宣称纯视觉导航能力提升**：T5 使用全局地标数据库（位置 + 外观），
  这是 map-free 模型没有的特权先验；clean 检索 100% 是渲染一致性，
  不是真实航拍检索精度。
- T5 − T4 的 +51.67 pp 混入了 T4 的 history-only 崩溃，
  不应全部归因于 map；T5 − T1 是更干净的参照。

### 5.2 T4：history + pose + GAB 在本设置下明确有害

- T4 vs T3：SR **−12.78 pp**（CI [−18.89, −6.67]），
  SPL −0.1278，nDTW −0.1748，premature **+48.33 pp**。
- 离线 macro-F1 从 T3 的 0.6839 升到 T4 的 0.8513（+16.74 pp），
  但闭环 SR 从 18.33% 降到 5.56%。
- 机制：T4 的动作坍缩为 **98.5% Forward**，停止距离中位数 334.5 m。
  历史编码器在闭环中把模型自己的动作历史变成“继续前进”的自反馈；
  GAB 的离线方向改善没有转化为闭环收益。
- **结论：不采用当前 T4 设计，不接入 Recovery/控制器，不扩展历史长度。**
  若未来重启 history 路线，必须先解决 on-policy 动作坍缩
  （例如 history dropout、动作条件 teacher forcing、闭环 DAgger），
  而不是继续加分支。

### 5.3 T1/T2/T3：Heatmap 与 Network State 的必要性

- **T1（network only）是唯一在 SR/SPL/nDTW 三项上显著优于 T0 的 map-free 模型**：
  SR +12.22 pp（CI [+4.44, +20.00]），SPL +0.1221，nDTW +0.1061；
  Short +15.00、Long +21.67，**Medium 0.00**。
- **T2（heatmap only）**：SR +3.89 pp（CI 跨 0），SPL +0.0389（CI 跨 0），
  nDTW +0.0587（CI 下限 +0.0019，勉强为正）。只有轨迹质量的弱证据。
- **T3（both removed）**：SR −2.22 pp（CI 跨 0），SPL −0.0222（CI 跨 0），
  nDTW +0.0292（CI 跨 0）。与 T0 无可靠差异。
- 机制上，heatmap 进入 Arrival head 使离线 far100 误停升高约 10 倍
  （T0/T2 0.52–0.57% vs T1/T3/T4 0.046%、T5 0.056%），并显著推远闭环停止距离
  （T0 140.7 m、T2 129.1 m vs T1 22.2 m）。
  Heatmap 改善 near20 停止召回（miss 26–27% vs 54–55%），
  但远距误停的代价在 T0 中占主导。
- **判断：在 50-epoch 协议下，Heatmap 分支不是必要的，且对停止校准有害；
  Network State 相对 heatmap-only/none 更有用，但 T0（两者都有）反而弱于
  T1（只保留 network）。单 seed 下不能宣称 Network State 有独立因果收益；
  证据支持“去掉 heatmap、保留 network”作为更稳的 map-free 配置。**

### 5.4 协议敏感性：50 epoch 预算显著压低 T0

- 归档 M0 为 **80 epoch / patience 10 / best epoch 71**，不满足本轮
  “50 epoch / patience 8”协议，因此没有直接复用。
- 用本轮 evaluator 重新评估归档 M0 checkpoint（stripped `base.` prefix），
  得到 **SR 34.44% / OSR 38.33% / SPL 0.3432 / nDTW 0.3573**，
  与 Stage-2h R0 **逐项完全一致**，证明评估器一致。
- 本轮 T0（50/8，best 47）为 **20.56% / 24.44% / 0.2056 / 0.2451**。
  同为原始 M0 架构，50-epoch 预算使 SR 下降 **13.88 pp**。
- 因此：
  - 主分析（vs 新 T0）在协议内公平；
  - 绝对水平受预算影响，不能把新 T0 与归档 34.44% 混用；
  - 作为上下文：**T1 32.78% 约等于归档 M0 的 34.44%（−1.67 pp）**，
    T2/T3/T4 均低于归档 M0，**只有 T5（map-assisted）明显高于归档 M0**
    （+22.78 pp）。
  - 这削弱了“T1 相对新 T0 的 +12.22 pp 是能力提升”的解读：
    它主要是**恢复到旧 80-epoch M0 水平**，而不是超越。

### 5.5 本轮决策

1. **不把 T4 / 当前 history-GAB 设计接入任何控制器或 Recovery。**
2. **Map-free 侧不宣称可靠增益**：T1 的点估计最好，但相对归档 80-epoch M0
   仅持平；T2/T3 无显著 SR/SPL 增益；T0 受 50-epoch 预算限制。
3. **T5 是唯一的强正向结果，但属于 map-assisted 设置**：
   匿名全局地标图 + 100% clean 检索，收益不能归因于纯视觉能力。
4. 若继续：
   - 先做 **多 seed** 复现 T5/T1/T4；
   - 把 map 特征接入 Arrival head（当前 T5 的 Arrival 仍纯视觉），
     检验能否把 OSR 85.56% 转化为 SR；
   - 对 T4 做 on-policy 动作坍缩修复（DAgger/history dropout）后再评估；
   - 在更强 T0 预算（例如旧 80 epoch）下重跑 T1/T5 的对照，
     避免把预算差异误读为模态收益。
5. **不启动其他 MEC 模型训练；不读 Final Test 300。**

### 5.6 限制

1. 单 seed（20264017）；Dev Val 180 的 Episode CI 较宽。
2. Dev Val 同时用于离线 checkpoint selection 与闭环评估，存在开发集
   选择偏差；这是本轮协议的规定，但不能当作最终泛化。
3. 50-epoch 预算使 T0 低于归档 80-epoch M0；绝对水平对预算敏感。
4. T4 的 history encoder / GAB 只验证了一种实现；动作坍缩是
   该实现 + on-policy 的结果，不代表所有历史方法。
5. T5 的 clean 检索饱和；5%/15% 噪声下 Top-1 只有 23.89%/10.56%，
   真实检索鲁棒性未解决。
6. T5 − T4 混入 T4 崩溃；T5 − T1 是更保守的 map 效应估计。
7. Final Test 300 未读取、未评估、未调参。

---

## 6. 产物、复现与提交

### 6.1 目录

```
runs/v83_spatial_stage2/stage2i/
├── aux/                         # 一次性 Train/Dev-Val sidecar（1.8 GB）
├── T5_map_assisted/             # best.pt, last.pt, config.json, history.csv,
│   └── eval/                    #   training_result.json, metrics.json, CSVs
├── T4_history_pose_gab/
├── T3_no_heatmap_no_network/
├── T2_no_network/
├── T1_no_heatmap/
├── T0_baseline/
├── repro_old_m0/                # 归档 80-epoch M0 的评估器复现
├── smoke/                       # 六组单 Batch + 2-Episode smoke
├── progress.json
├── RESULTS.md
├── summary.json / summary.md    # 总表、Band、配对
└── prep_summary.json            # 数据隔离、参数、测试、smoke 记录
```

### 6.2 命令

```bash
# 一次性公共准备
python v83_spatial_stage2/stage2i_prepare.py \
  --output-dir runs/v83_spatial_stage2/stage2i/aux

# 单元测试
python -m unittest -v v83_spatial_stage2.test_stage2i

# 单 Batch smoke
python v83_spatial_stage2/stage2i_train.py --variant T5 --smoke \
  --output-dir runs/v83_spatial_stage2/stage2i/smoke

# 每个模型：训练 + 评估（示例 T5）
python v83_spatial_stage2/stage2i_train.py --variant T5 \
  --output-dir runs/v83_spatial_stage2/stage2i/T5_map_assisted
python v83_spatial_stage2/stage2i_eval.py --variant T5 \
  --model-dir runs/v83_spatial_stage2/stage2i/T5_map_assisted

# 汇总
python v83_spatial_stage2/stage2i_report.py
```

中断恢复：训练 `--resume`（读取 `last.pt`），评估 `--resume`
（跳过已写入 CSV 的 Episode）；`progress.json` 记录每个模型的
训练/评估状态、best epoch、checkpoint 与指标。

### 6.3 测试

- `python -m unittest -v v83_spatial_stage2.test_stage2i`：**15 tests OK**
  - T0 与原始架构 state-dict key/shape/value 完全一致；
  - 六组 true branch removal / param count；
  - 移除模态的输入拒绝；
  - 共享初始化一致性；
  - history 因果性、mask、online/training 一致性；
  - GAB 因果前缀与 Episode reset；
  - 匿名 map 无 `is_goal`/`is_distractor`/`env.goal`、cosine Top-k、
    确定性置换；
  - 六组 forward/backward；
  - Stop threshold AST 纪律。
- Aux 对齐：独立重放 Train/Dev Val 样本，history/pose/GAB/map
  逐元素差值 **0.0**；pose replay mismatch 0。
- 六组单 Batch 前向/反向：有限 loss，关键分支梯度非零。
- 六组 2-Episode 闭环 smoke：全部通过。

### 6.4 Commit / Push

- 源码/测试/报告提交：`c4b901a`
  （`Stage-2i: six MEC input ablations with history/pose/GAB and map-assisted T5`）
- 分支：`experiment/v83-spatial-foundation`
- 远端：`origin` = `https://github.com/JLU-MCNS-MEC/mec-v8.2.git`
- Push：`48bc414..c4b901a`，**成功**
  （本机 HTTP/SOCKS 代理 `127.0.0.1:7890` 的 TLS handshake 失败；
  本次 push 在清除代理环境变量后直连成功。）
- 大型权重、缓存、逐 Episode CSV 与日志保留在服务器
  `runs/v83_spatial_stage2/stage2i/`，未上传 GitHub
  （`runs/`、`*.pt`、`*.log` 均在 `.gitignore`）。
- 本报告的状态更新提交在 `c4b901a` 之后单独推送。

### 6.5 Final Test

`final_test_300` 保持 `SEALED.json` 状态；本轮所有入口拒绝包含
`final_test` 的路径。未读取、未评估、未调参。
