# V8.3 Stage-2i — T5 Anonymous Global Landmark Retrieval Graph

日期：2026-10-19

分支：`experiment/v83-spatial-foundation`

性质：**map-assisted 独立设置**。T5 = T4（历史视觉 + 相对位姿 + GAB）+ 匿名全局地标检索图。
T0–T4 均为 map-free；T5 的收益不得解释为纯视觉导航能力提升。

---

## 1. 设计目标

T0–T4 只能看到当前与过去的第一人称 RGB 观测。T5 额外假设无人机拥有一张
**预先建好的匿名全局地标图**：图中每个节点是一个地标的外观与位置，但不含
任何“这是目标 / 这是干扰物”的标签。给定 Goal RGB reference，T5 用冻结
视觉编码器做外观检索，取 Top-k 节点，把它们的**相对几何**送入 MEC 融合层。

这与 CityNav 的 Geo-Semantic Map / OSM 地标先验同类，但本环境没有语义名、
没有语言指令，因此只使用匿名外观 + 位置。

---

## 2. 匿名建图流程

### 2.1 节点集合

每个 Episode 重置后，模拟环境包含：

- 160 个普通 landmark（其中部分 `is_distractor=True`，但该字段**从不读取**）；
- 1 个 goal landmark。

建图时把 161 个对象放进同一个列表：

```python
nodes = list(env.landmarks) + [env.goal_landmark]
order = np.random.default_rng(seed ^ 0x5EED1234).permutation(len(nodes))
```

- 目标对象以**与普通 landmark 完全相同的路径**加入；不读取 `is_goal`
  （环境中不存在该字段）、不读取 `is_distractor`、不读取 `env.goal`。
- 使用 Episode seed 驱动的确定性置换，使目标节点不固定在列表末尾。
- `goal_index` 只在**事后检索诊断**中使用（Top-1/Top-5 是否正确），
  不传入任何特征构造函数。

### 2.2 统一外观渲染

每个节点按以下统一流程生成一张 336×336 的“勘查影像”：

- 背景 34；
- 用环境自身的 `_draw_shape` / `_draw_marker` 基元，以 56 px 对象宽度绘制
  该节点的 shape / base color / marker / marker color；
- 与 Goal reference 的 56 px 尺度使用相同的画布尺寸和绘制代码；
- 不添加噪声、雾、遮挡或任何身份标记。

```python
canvas = render_landmark_canvas(env, node)   # 336x336, object 56 px
```

### 2.3 统一视觉特征

16 个节点一批，送入冻结的 shared RGB encoder（2 层 ViT，384-D token），
再用与 `encode_multiscale_goal_tokens` 完全相同的 **foreground-weighted
token pooling** 得到 384-D 节点 embedding：

```python
embeddings = encode_landmark_canvases(canvases, encoder, device)
```

- 所有节点走同一函数、同一尺度、同一权重规则；
- 不按目标/干扰物分组；
- 目标节点 embedding 与 Goal 56 px token 的余弦相似度约 **0.9999998**
  （相同渲染 + 相同编码器），这是检索在干净条件下接近饱和的原因。

---

## 3. 检索

Query 是 Goal RGB reference 在 56 px 尺度的 384-D token（所有模型都有
Goal RGB tokens，因此这不是 T5 独有的输入）。

```python
sims = cosine(node_embeddings, goal_token_56px)
order = argsort(-sims)
topk = order[:5]
```

- 选择依据只有视觉相似度；
- 不读取节点索引、`is_goal`、`is_distractor`、`env.goal`、目标坐标或距离；
- 不按 Episode 的 goal landmark 位置做任何筛选。

检索完成后，Top-5 节点的**世界坐标**才用于构造几何特征。地图本身是
特权先验：map-free 模型没有这 161 个节点的位置/外观数据库。

---

## 4. 地图特征（22-D）

对 Top-5 节点（按相似度降序），用无人机当前位姿 `(x, y, yaw)` 计算：

| 分量 | 维度 | 定义 |
|---|---:|---|
| similarity | 5 | 与 Goal token 的余弦相似度 |
| forward | 5 | 节点在机体前向的位移 / 500 m |
| right | 5 | 节点在机体右向的位移 / 500 m |
| distance | 5 | 节点到无人机的水平距离 / 500 m |
| top1 similarity | 1 | 最高相似度 |
| top1−top2 margin | 1 | 检索置信度间隔 |

合计 22 维，作为一个额外 modality token 进入 MEC fusion Transformer，
与 heatmap / network / history / GAB token 并列。Arrival head 保持
T3 的可观测 match-feature 定义不变，因此 T3–T5 的停止行为可比较。

---

## 5. 目标信息泄漏审计

### 5.1 禁用字段

| 字段 | 是否读取 | 说明 |
|---|---|---|
| `is_goal` | **否** | 环境中无此字段；代码/测试均检查 |
| `is_distractor` | **否** | 只在建图时作为普通节点属性被忽略 |
| `env.goal` | **否** | `stage2i_data` AST 测试禁止 `env.goal` 属性访问 |
| Goal ID / landmark ID | **否** | 只有置换后的匿名列表索引 |
| `env.goal_landmark` | 仅用于把目标对象**匿名加入节点列表** | 与 160 个普通 landmark 同一渲染/编码路径；不作为选择条件 |
| 真实目标距离/方位 | **否** | 地图特征只使用检索节点的位置与 UAV 自位姿 |

### 5.2 测试

`v83_spatial_stage2/test_stage2i.py::AnonymousMapTests`：

- AST 扫描 `stage2i_data`，禁止 `is_goal` / `is_distractor` 属性访问；
- 禁止 `env.goal`；
- 合成节点集上验证 retrieval = cosine Top-k；
- 验证置换确定性、目标索引只作诊断；
- 验证 map 特征在已知几何下数值正确。

### 5.3 已知特权

T5 使用：

1. **全局地标数据库**：161 个节点的外观与位置（map-free 模型没有）；
2. **UAV 绝对自位姿**：用于把地图节点变到机体坐标系（自定位/GPS 等价，
   不含目标信息）。

因此 T5 是 map-assisted 设置，必须与 T0–T4 分开解释。

---

## 6. 检索诊断

### 6.1 干净检索

| Split | Episodes | Top-1 | Top-5 | 平均 goal cosine |
|---|---:|---:|---:|---:|
| Train 600 | 600 | 100.00% | 100.00% | 0.99999976 |
| Dev Val 180 | 180 | 100.00% | 100.00% | 0.99999981 |

干净条件下目标节点与 Goal token 来自同一渲染/编码流程，检索饱和；
这测量的是“地图里是否有正确外观”，不是视觉鲁棒性。

### 6.2 噪声压力测试

在节点 embedding 上加入高斯噪声（σ = 平均 embedding 范数 × 5% / 15%），
重新计算 Dev Val 180 的检索：

| 条件 | Top-1 | Top-5 | Top-1 错误数 |
|---|---:|---:|---:|
| clean | 100.00% | 100.00% | 0 |
| 5% noise | 23.89% | 67.78% | 137 |
| 15% noise | 10.56% | 38.89% | 161 |

5% 噪声下多数错误仍把目标留在 Top-5（27/137 排第 2、21 排第 3、17 排第 4、
14 排第 5），15% 噪声下错误更分散。说明干净检索的 100% 是**渲染/编码
一致性**带来的，不是对视觉扰动稳健；真实航拍检索会更接近噪声条件，
不能把本实验的 Top-1 外推到 CityRefer/CityNav。

### 6.3 错误检索案例

- 示例（5% 噪声，Episode 1）：目标排第 8，Top-1 similarity 0.7577，
  goal similarity 0.7206，margin 0.0133；
- 示例（5% 噪声，Episode 2）：目标排第 3，Top-1 0.7491，goal 0.7331，
  margin 0.0153；
- 示例（15% 噪声，Episode 0）：目标排第 10，Top-1 0.3665，
  goal 0.3180，margin 0.0171；
- 完整逐 Episode 记录：
  `runs/v83_spatial_stage2/stage2i/T5_map_assisted/eval/retrieval_episodes.csv`；
  错误案例采样（前 50 条）：
  `retrieval_error_cases.csv`。

注意：T5 的 MEC 评估使用的是 **clean retrieval**；噪声结果只是诊断，
不参与控制，也不改变 T5 的闭环结果。

### 6.4 T5 闭环结果摘要

| 指标 | T5 |
|---|---:|
| SR | 57.22% (103/180) |
| OSR | 85.56% |
| SPL | 0.5722 |
| nDTW | 0.5575 |
| Premature-Stop | 32.78% |
| Timeout | 10.00% |
| Average Steps | 135.6 |
| Best Epoch | 24 |

- OSR → SR gap = **28.34 pp**：map 把无人机带到目标附近，
  但 Arrival head（与 T4/T3 相同的纯视觉 match-feature 定义）
  仍错失一部分停止机会；
- 相对 T4（同架构无 map）：SR +51.67 pp，但 T4 已发生 forward 动作坍缩；
- 相对最佳 map-free T1：SR +24.44 pp（CI [+15.56, +33.89]），
  这是更保守的 map-assisted 效应估计。
- **T5 的 Arrival head 没有使用 map 距离/位置**；把 map 接入停止决策
  是明确的后续实验，不属于本轮设计。

---

## 7. 解释边界

- T5 的结果是**地图辅助**结果；不能据此宣称纯视觉导航能力提升。
- 地图数据库来自模拟器地标表；真实系统需要离线建图/更新流程。
- 单 seed、Dev Val 180，不是最终泛化结论。
- Final Test 300 未读取。
- 检索饱和主要反映环境地标外观唯一且渲染一致；真实航拍检索会更难，
  因此本实验的检索精度不能外推到真实 CityRefer/CityNav。
