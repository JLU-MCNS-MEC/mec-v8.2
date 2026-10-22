# Stage-2n 启动前审计

日期：2026-10-22  
分支：`experiment/v83-spatial-foundation`  
协议：`v83_spatial_stage2/STAGE2N_PROTOCOL.md`  
运行根目录：`runs/v83_spatial_stage2/stage2n_latency_only/`

本文档记录正式训练前的资源、任务、部署、MEC 溯源、测试与一致性审计。

---

## 1. 服务器与 GPU

- GPU：RTX 5090 32 GB；
- 正式训练前显存占用：约 6.0 GB（其中两个 Stage-2l B 运行处于 SIGSTOP
  暂停态，各占约 3 GB；**未停止、未修改**）；
- Stage-2n 单进程显存约 1.3–1.6 GB（只加载 Transformer-M，不加载五个 MEC）；
- CPU 32 核、内存约 61 GB；6 个 run 计划并行，每个 2 个 torch 线程；
- 磁盘 `/` 可用约 32 GB；Stage-2n 任务清单约 4.7 MB，checkpoint 很小，
  原始日志留在 `runs/`。

旧实验状态：

- Stage-2l A/C 已完成；
- Stage-2l B/DQN、B/PPO 自 2026-10-21 13:16 起 SIGSTOP，保留 checkpoint；
- 本阶段不停止、不重启、不覆盖任何旧进程。

---

## 2. 固定城市与任务池

### 2.1 城市

| 项目 | 值 |
|---|---|
| `map_seed` | 10,000,001 |
| 地标数 | 160 |
| 世界大小 | 1500 m × 1500 m |
| 城市 SHA256 | `392500aa0df1d87b341f2fbd3df2fe4f2dc0405624a1ecc91cd33eca616bb182` |

### 2.2 任务池

| Split | 数量 | Short/Medium/Long | task seed 范围 | 文件 SHA256 |
|---|---:|---|---|---|
| Train | 4000 | 1334/1333/1333 | 21,000,000–21,004,820 | `ed3bbc4f…ae013` |
| Dev | 60 | 20/20/20 | 22,000,000–22,000,073 | `1b0518b5…0653a` |
| Test（封存） | 400 | 134/133/133 | 23,000,000–23,004,491 | `2cfb1a68…10042` |

- 任务池 `tasks_sha256` 与单任务 SHA256 全部通过校验；
- Train/Dev/Test seed 与 task ID 两两不相交；
- 与 Stage-2l 的 Train/Dev/Test 池交叉审计：seed 与 ID 均不相交；
- Test 池 `sealed: true`，默认加载被拒绝；
- 训练任务顺序文件 SHA256 `c235e3fd…510c7`，顺序哈希
  `dece010a…18e1c`，随机种子 `20261022`。

### 2.3 分布

详见 `runs/.../specs/distribution.json`。三个 Band 的专家步数均落在
协议范围；距离范围与 Stage-2l 相同。

---

## 3. MEC 部署与溯源

### 3.1 检查点

| 项目 | 值 |
|---|---|
| 模型 | Transformer-M |
| 路径 | `runs/v83_spatial_stage2/stage2j/Transformer-M/best.pt` |
| SHA256 | `57c88b7d71e1a7f17862aaa8429afa8a041cba5e747f744f3a7febf47f273c6d` |
| 参数量 | 3,033,732 |
| 选择依据 | Stage-2j Dev Val 180 SR 57.22%（五个 MEC 中最高） |
| Final Test 300 参考 | SR 55.33%、SPL 0.5525、nDTW 0.5660（不参与选择） |
| 是否重训 | 否，`eval()` + `requires_grad=False` |

### 3.2 同质部署

- 位置：Stage-2l `A_fixed` 的 20 个位置，拓扑种子 9,000,000；
- 每个节点：Transformer-M、2.0 TFLOPS、30 MHz、500 m 覆盖、0.1 W；
- 节点 ID：`M01`–`M20`；
- 部署 SHA256：`4c0950cf6e438cff41454eee6daefb831fa984f425ec1a41cf3974257c6d1148`；
- 9 项同质性检查全部通过：20 节点、位置唯一、模型/算力/带宽/覆盖/质量
  先验各只有 1 个取值、全部 Transformer-M、位置在城市内。

---

## 4. 测试

```bash
python v83_spatial_stage2/test_stage2n.py
```

结果：**24 项测试全部通过**（约 51 s），覆盖：

- 任务数量、Band、SHA256、封存、种子/ID 隔离、与 Stage-2l 交叉隔离；
- 同质部署、位置唯一、哈希稳定、与 A_fixed 位置一致；
- Transformer-M 哈希与真实评估记录；
- 相同观测重复推理动作一致、只加载一个模型；
- 时延奖励 telescopes 到 `-0.25 × mean_E2E / 100`；
- 收敛判据与最佳检查点选择；
- 训练 bundle 不含 Test；
- 384-D 状态 + 20-D Mask、不可用节点特征为零；
- MEC 仍接收完整 RGB/Goal/History 输入；
- 真实单 Episode 闭环奖励恒等式；
- 冻结 lane 与完整闭环在 4 种启发式 + DQN 上逐字段一致。

---

## 5. 同质 MEC 一致性探针

```bash
python v83_spatial_stage2/stage2n_consistency.py --tasks 6
```

结果（`v83_spatial_stage2/stage2n_evidence/consistency_probe.json`）：

| 策略 | 动作不一致 | SR/SPL/nDTW 不一致 | 6 任务平均 E2E (ms) |
|---|---:|---:|---:|
| Random | 0 | 0 | 94.10 |
| Nearest | 0 | 0 | 43.98 |
| Max-Uplink-Rate | 0 | 0 | 43.98 |
| Greedy-Min-E2E | 0 | 0 | 43.98 |

同一观测重复输入同一 Transformer-M，动作与 arrival probability 完全一致。
这证明：关联策略只改变时延，不改变导航；同质部署下三种启发式排序相同，
是设计使然。

---

## 6. 预注册与启动门槛

正式训练只允许在以下条件满足后启动：

- [x] 任务池生成与封存；
- [x] 同质部署构建与哈希；
- [x] Transformer-M 哈希与记录核实；
- [x] 22 项测试通过；
- [x] 一致性探针通过；
- [x] 协议冻结（`protocol_freeze.json`）；
- [x] GPU 空闲显存 ≥ 10 GB（未动旧进程）；
- [x] DQN/PPO 15-Episode 与 20-Episode smoke 正常（20-Episode 触发训练更新：
      DQN loss 4.9e-4 / TD 0.015；PPO KL 0.006 / value loss 0.035 /
      explained variance 0.834）；
- [ ] 6 个正式 run 启动。

---

## 7. 启动命令

```bash
python v83_spatial_stage2/stage2n_launch.py \
    --algorithms dqn ppo --seeds 0 1 2 \
    --max-parallel 6 --threads 2 --min-free-gb 10
```

日志：`runs/v83_spatial_stage2/stage2n_latency_only/logs/`  
每个 run：`runs/v83_spatial_stage2/stage2n_latency_only/runs/{alg}_s{seed}/`
