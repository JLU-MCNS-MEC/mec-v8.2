# Stage-2l 固定城市实验 — 启动前报告

生成时间：2026-10-21T03:25:25.401573+00:00

## 1. 固定城市与任务生成

- `map_seed = 10000001`，整个 Train/Dev/Test 任务池在第一阶段共享这一张城市地图；任务 Seed 与任务 ID 完全隔离。
- 城市指纹：`392500aa0df1d87b341f2fbd3df2fe4f2dc0405624a1ecc91cd33eca616bb182`
- 城市地标数：160 个（道路网格、地面明暗、地标位置/数量/尺寸/颜色/形状/内部 marker 均由 `map_seed` 固定）。
- `task_seed` 只决定 UAV 起点、朝向、航向误差与目标；目标必须从固定城市已有地标中选择，不会新建建筑，也不会修改其他地标。
- RGB 仍由 UAV 位姿渲染；同一任务同一位姿的帧可复现，不同位姿的帧不同。

## 2. 环境固定性与 T5 输入 Smoke Test

| 检查 | 结果 |
|---|---|
| 城市在任务间不变 | True |
| 目标来自既有地标 | True |
| 任务不新增/修改地标 | True |
| RGB 随位姿变化 | True |
| 同帧可复现 | True |
| T5 输入有限 | True |
| 状态维度 = 384 | 384 |
| 状态无目标字段 | True |
| 同一位姿/不同目标的 Pure-MEC 状态完全相同 | True |

T5 输入形状：

```json
{
  "current_tokens": [
    1,
    576,
    384
  ],
  "goal_tokens": [
    1,
    3,
    384
  ],
  "history_vis": [
    1,
    16,
    768
  ],
  "history_pose": [
    1,
    16,
    10
  ],
  "history_action": [
    1,
    16,
    4
  ],
  "history_mask": [
    1,
    16
  ],
  "gab": [
    1,
    20
  ],
  "map_feat": [
    1,
    22
  ]
}
```

五种冻结 MEC 在固定 Dev 任务上的轻量闭环：

| MEC | SR | nDTW | E2E (ms) | JointR | Dev prior SR |
|---|---:|---:|---:|---:|---:|
| Small | 16.67% | 0.2253 | 176.8 | -0.2165 | 16.11% |
| Transformer-S | 33.33% | 0.3345 | 206.7 | -0.1822 | 53.33% |
| Medium | 16.67% | 0.2277 | 143.3 | -0.1305 | 18.33% |
| Transformer-M | 33.33% | 0.3468 | 176.9 | -0.0955 | 57.22% |
| Large | 50.00% | 0.4342 | 141.5 | 0.0803 | 48.89% |

非特权 Baseline（同一批 Dev 任务，仅用于确认闭环不崩溃）：

| 策略 | SR | nDTW | E2E (ms) | JointR |
|---|---:|---:|---:|---:|
| random | 33.33% | 0.2492 | 174.7 | -0.1875 |
| min_latency | 33.33% | 0.3250 | 102.0 | 0.0700 |
| max_quality | 50.00% | 0.4440 | 158.7 | 0.0473 |
| heuristic | 50.00% | 0.4440 | 151.4 | 0.0655 |

> Smoke Test 只用于排除“新场景与冻结 T5 不兼容 / 目标泄漏 / 性能完全崩溃”；样本很少，不构成学习效果结论。

## 3. 任务分布

| Split | 总数 | Short / Medium / Long | 专家步数范围 | 起终点距离均值 (m) |
|---|---:|---|---|---|
| train | 3500 | 1167 / 1167 / 1166 | 30–80 / 50–115 / 80–160 | 196.0 / 314.8 / 463.1
| dev | 60 | 20 / 20 / 20 | 34–79 / 68–115 / 88–160 | 206.5 / 323.6 / 467.5
| test | 60 | 20 / 20 / 20 | 33–80 / 51–111 / 80–153 | 206.4 / 306.4 / 465.1

Train 任务池生成后使用固定种子全局打乱；不存在 Short→Medium→Long 的分段训练。

数据隔离检查：

```json
{
  "train_dev_seed_disjoint": true,
  "train_test_seed_disjoint": true,
  "dev_test_seed_disjoint": true,
  "train_dev_uid_disjoint": true,
  "train_test_uid_disjoint": true,
  "dev_test_uid_disjoint": true,
  "all_share_phase1_map": true,
  "test_sealed": true,
  "ok": true
}
```

## 4. 三种部署配置

| 配置 | 说明 | Train 拓扑 | Dev 拓扑 |
|---|---|---|---|
| A | Balanced 场景，单张固定拓扑（训练与验证共用该固定拓扑） | A_fixed | A_fixed |
| B | Balanced 场景，10 张独立 Train 拓扑随机分配给训练任务；验证使用独立固定 Dev 拓扑 | B_train_00, B_train_01, B_train_02, B_train_03, B_train_04, B_train_05, B_train_06, B_train_07, B_train_08, B_train_09 | B_dev |
| C | Balanced / Quality-rich / Tradeoff-stress 三种部署场景，每种各一张固定拓扑 | C_balanced, C_quality_rich, C_tradeoff_stress | C_balanced, C_quality_rich, C_tradeoff_stress |

拓扑种子（与 Stage-2j / Stage-2k 均不重叠）：

| 拓扑 | 场景 | Seed | 节点数 |
|---|---|---:|---:|
| A_fixed | balanced | 9000000 | 20 |
| B_dev | balanced | 9000050 | 20 |
| B_test | balanced | 9000060 | 20 |
| B_train_00 | balanced | 9000100 | 20 |
| B_train_01 | balanced | 9000101 | 20 |
| B_train_02 | balanced | 9000102 | 20 |
| B_train_03 | balanced | 9000103 | 20 |
| B_train_04 | balanced | 9000104 | 20 |
| B_train_05 | balanced | 9000105 | 20 |
| B_train_06 | balanced | 9000106 | 20 |
| B_train_07 | balanced | 9000107 | 20 |
| B_train_08 | balanced | 9000108 | 20 |
| B_train_09 | balanced | 9000109 | 20 |
| C_balanced | balanced | 9000000 | 20 |
| C_quality_rich | quality_rich | 9000200 | 20 |
| C_tradeoff_stress | tradeoff_stress | 9000300 | 20 |

## 5. 训练配置

- 算法：标准 DQN（`384→256→256→20`，gamma=1.0，replay 50k，target update 1k，epsilon 1.0→0.05 / 500k 环境步）与标准 PPO（`384→256→256→{20,1}`，gamma=1.0，GAE 0.95，clip 0.2，rollout 2048，minibatch 256，10 epochs）。
- 状态：Pure MEC v2，384-D MLP 输入 + 20-D 可用性 Mask；包含 UAV/MEC 位置、信道、算力、推理时延、E2E 时延、独立 Dev 闭环能力先验、累计步数与运行平均时延。
- 奖励：与 JointR 一致的 `r_nav + Δrunning_mean_E2E`，无成功奖励、失败惩罚或 Oracle 奖励。
- 每种算法最多 3500 Episodes，每 200 Episodes 在 60 条固定独立 Dev 任务上验证；不早停，只保存 best 与 last checkpoint。
- 五种 Stage-2j T5 MEC 全部冻结（`eval()` + `requires_grad=False`），不训练 MEC。
- Final Test 任务池保持封存，训练脚本不读取。

## 6. 启动方式

先启动 A 的 DQN 与 PPO；B/C 由监督脚本在资源允许时排队启动，不会停止或覆盖已有 Stage-2k 进程。

```bash
python v83_spatial_stage2/stage2l_dqn_train.py --config A ...
python v83_spatial_stage2/stage2l_ppo_train.py --config A ...
```
