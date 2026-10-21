"""Generate the pre-training Chinese setup report for Stage-2l."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import read_json
from v83_spatial_stage2.stage2l_fixed_city import (
    CONFIG_DESCRIPTIONS,
    MAP_SEED_PHASE1,
    STAGE2L_ROOT,
    TOPOLOGY_SEEDS,
    load_config_manifest,
    load_task_pool,
    load_topology,
)


def _band_line(item):
    bands = item["bands"]
    steps = item["oracle_steps"]
    distances = item["start_goal_distance_m"]
    return (
        f"| {bands['short']} / {bands['medium']} / {bands['long']} | "
        f"{steps['short']['min']}–{steps['short']['max']} / "
        f"{steps['medium']['min']}–{steps['medium']['max']} / "
        f"{steps['long']['min']}–{steps['long']['max']} | "
        f"{distances['short']['mean']:.1f} / "
        f"{distances['medium']['mean']:.1f} / "
        f"{distances['long']['mean']:.1f} |"
    )


def main():
    protocol = read_json(STAGE2L_ROOT / "protocol_freeze.json")
    distribution = read_json(STAGE2L_ROOT / "specs" / "distribution.json")
    isolation = read_json(STAGE2L_ROOT / "specs" / "isolation.json")
    smoke = read_json(STAGE2L_ROOT / "smoke" / "smoke_report.json")
    if protocol is None or distribution is None or isolation is None:
        raise SystemExit("Run stage2l_build.py first")
    if smoke is None:
        raise SystemExit("Run stage2l_smoke.py first")

    lines = [
        "# Stage-2l 固定城市实验 — 启动前报告",
        "",
        f"生成时间：{protocol['created_utc']}",
        "",
        "## 1. 固定城市与任务生成",
        "",
        f"- `map_seed = {MAP_SEED_PHASE1}`，整个 Train/Dev/Test 任务池"
        "在第一阶段共享这一张城市地图；任务 Seed 与任务 ID 完全隔离。",
        f"- 城市指纹：`{smoke['city_fingerprint']}`",
        f"- 城市地标数：{smoke['city_landmarks']} 个（道路网格、地面明暗、"
        "地标位置/数量/尺寸/颜色/形状/内部 marker 均由 `map_seed` 固定）。",
        "- `task_seed` 只决定 UAV 起点、朝向、航向误差与目标；目标必须从"
        "固定城市已有地标中选择，不会新建建筑，也不会修改其他地标。",
        "- RGB 仍由 UAV 位姿渲染；同一任务同一位姿的帧可复现，"
        "不同位姿的帧不同。",
        "",
        "## 2. 环境固定性与 T5 输入 Smoke Test",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
        f"| 城市在任务间不变 | {smoke['fixed_city_ok']} |",
        f"| 目标来自既有地标 | {smoke['goal_existing_ok']} |",
        f"| 任务不新增/修改地标 | {smoke['landmark_stable_ok']} |",
        f"| RGB 随位姿变化 | {smoke['render_pose_dependent_ok']} |",
        f"| 同帧可复现 | {smoke['render_reproducible_ok']} |",
        f"| T5 输入有限 | {smoke['t5']['inputs_finite']} |",
        f"| 状态维度 = 384 | {smoke['t5']['state_dim']} |",
        f"| 状态无目标字段 | "
        f"{not smoke['t5']['forbidden_state_tokens_in_layout'] and not smoke['t5']['forbidden_candidate_fields']} |",
        f"| 同一位姿/不同目标的 Pure-MEC 状态完全相同 | "
        f"{smoke['t5']['state_identical_for_different_goal']} |",
        "",
        "T5 输入形状：",
        "",
        "```json",
        json.dumps(smoke["t5"]["input_shapes"], indent=2),
        "```",
        "",
        "五种冻结 MEC 在固定 Dev 任务上的轻量闭环：",
        "",
        "| MEC | SR | nDTW | E2E (ms) | JointR | Dev prior SR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in smoke["mecs"].items():
        lines.append(
            f"| {name} | {100*item['SR']:.2f}% | {item['nDTW']:.4f} | "
            f"{item['E2E_ms']:.1f} | {item['JointR']:.4f} | "
            f"{100*item['prior_SR']:.2f}% |"
        )
    lines += [
        "",
        "非特权 Baseline（同一批 Dev 任务，仅用于确认闭环不崩溃）：",
        "",
        "| 策略 | SR | nDTW | E2E (ms) | JointR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in smoke["baselines"].items():
        lines.append(
            f"| {name} | {100*item['SR']:.2f}% | {item['nDTW']:.4f} | "
            f"{item['E2E_ms']:.1f} | {item['JointR']:.4f} |"
        )
    lines += [
        "",
        "> Smoke Test 只用于排除“新场景与冻结 T5 不兼容 / 目标泄漏 / "
        "性能完全崩溃”；样本很少，不构成学习效果结论。",
        "",
        "## 3. 任务分布",
        "",
        "| Split | 总数 | Short / Medium / Long | 专家步数范围 | 起终点距离均值 (m) |",
        "|---|---:|---|---|---|",
    ]
    for split in ("train", "dev", "test"):
        item = distribution[split]
        lines.append(
            f"| {split} | {item['count']} | " + _band_line(item)
        )
    lines += [
        "",
        "Train 任务池生成后使用固定种子全局打乱；"
        "不存在 Short→Medium→Long 的分段训练。",
        "",
        "数据隔离检查：",
        "",
        "```json",
        json.dumps(isolation["checks"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## 4. 三种部署配置",
        "",
        "| 配置 | 说明 | Train 拓扑 | Dev 拓扑 |",
        "|---|---|---|---|",
    ]
    for config, description in CONFIG_DESCRIPTIONS.items():
        manifest = load_config_manifest(config)
        train_roles = manifest["topology_roles"]["train"]
        dev_roles = manifest["topology_roles"]["dev"]
        lines.append(
            f"| {config} | {description} | "
            f"{', '.join(train_roles)} | {', '.join(dev_roles)} |"
        )
    lines += [
        "",
        "拓扑种子（与 Stage-2j / Stage-2k 均不重叠）：",
        "",
        "| 拓扑 | 场景 | Seed | 节点数 |",
        "|---|---|---:|---:|",
    ]
    for role in sorted({
        *TOPOLOGY_SEEDS_ROLES(),
    }):
        topology = load_topology(role)
        lines.append(
            f"| {role} | {topology['scenario']} | "
            f"{topology['topology_seed']} | {len(topology['nodes'])} |"
        )
    lines += [
        "",
        "## 5. 训练配置",
        "",
        "- 算法：标准 DQN（`384→256→256→20`，gamma=1.0，replay 50k，"
        "target update 1k，epsilon 1.0→0.05 / 500k 环境步）与标准 PPO"
        "（`384→256→256→{20,1}`，gamma=1.0，GAE 0.95，clip 0.2，"
        "rollout 2048，minibatch 256，10 epochs）。",
        "- 状态：Pure MEC v2，384-D MLP 输入 + 20-D 可用性 Mask；"
        "包含 UAV/MEC 位置、信道、算力、推理时延、E2E 时延、"
        "独立 Dev 闭环能力先验、累计步数与运行平均时延。",
        "- 奖励：与 JointR 一致的 `r_nav + Δrunning_mean_E2E`，"
        "无成功奖励、失败惩罚或 Oracle 奖励。",
        "- 每种算法最多 3500 Episodes，每 200 Episodes 在 60 条固定独立 "
        "Dev 任务上验证；不早停，只保存 best 与 last checkpoint。",
        "- 五种 Stage-2j T5 MEC 全部冻结（`eval()` + "
        "`requires_grad=False`），不训练 MEC。",
        "- Final Test 任务池保持封存，训练脚本不读取。",
        "",
        "## 6. 启动方式",
        "",
        "先启动 A 的 DQN 与 PPO；B/C 由监督脚本在资源允许时排队启动，"
        "不会停止或覆盖已有 Stage-2k 进程。",
        "",
        "```bash",
        "python v83_spatial_stage2/stage2l_dqn_train.py --config A ...",
        "python v83_spatial_stage2/stage2l_ppo_train.py --config A ...",
        "```",
        "",
    ]
    path = STAGE2L_ROOT / "SETUP_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[setup] wrote {path}", flush=True)
    print("STAGE2L_SETUP_REPORT_COMPLETE", flush=True)


def TOPOLOGY_SEEDS_ROLES():
    roles = ["A_fixed", "B_dev", "B_test"]
    roles.extend(f"B_train_{index:02d}" for index in range(10))
    roles.extend(["C_balanced", "C_quality_rich", "C_tradeoff_stress"])
    return roles


if __name__ == "__main__":
    main()
