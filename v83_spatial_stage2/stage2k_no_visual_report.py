"""Generate the Chinese Stage-2k no-visual / pure-MEC DQN report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    MODEL_NAMES,
    read_json,
    summarize_episodes,
    utc_now,
)


DEFAULT_RUN = Path("runs/v83_spatial_stage2/stage2k_no_visual")
PURE_MEC_RUN = Path("runs/v83_spatial_stage2/stage2k_pure_mec_state")
DEFAULT_REPORT = Path(
    "v83_spatial_stage2/STAGE2K_NO_VISUAL_STATE_REPORT.md"
)
PURE_MEC_REPORT = Path(
    "v83_spatial_stage2/STAGE2K_PURE_MEC_STATE_REPORT.md"
)
BASELINE_DEV_CSV = Path(
    "runs/v83_spatial_stage2/stage2k/baselines/dev/per_episode.csv"
)
FIG = "figures/stage2k_no_visual"
TITLES = {
    "dqn": "DQN (no visual)",
    "random": "Random",
    "min_latency": "Min-Latency",
    "max_quality": "Max-Quality",
    "heuristic": "Heuristic",
    "oracle": "Oracle (privileged)",
}
STRATEGIES = (
    "dqn", "random", "min_latency", "max_quality", "heuristic", "oracle",
)
SCENARIOS = ("balanced", "quality_rich", "tradeoff_stress")
SCENARIO_NAMES = {
    "balanced": "Balanced / Original",
    "quality_rich": "Quality-rich",
    "tradeoff_stress": "Tradeoff-stress",
}


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-mode", choices=("no_visual", "pure_mec"),
                        default="no_visual")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    state_mode = args.state_mode
    if state_mode == "pure_mec":
        run = args.run_dir or PURE_MEC_RUN
        output = args.output or PURE_MEC_REPORT
        global FIG
        FIG = "figures/stage2k_pure_mec_state"
        TITLES["dqn"] = "DQN (pure MEC)"
        state_label = "Pure MEC"
        state_dim = 322
        candidate_dim = 16
        decision_dim = 342
        state_title = (
            "本轮使用纯 MEC 状态：DQN 输入 **322 维** = 2 维 UAV 位置 + "
            "20 × 16 维候选 MEC 特征；另设 **20 维可用性 Mask**，"
            "只用于动作屏蔽，不重复进入 MLP。决策状态合计 342 维。"
        )
        state_rows = [
            "| UAV 位置 | 2 | 当前二维位置 / 1500 m |",
            "| 候选 MEC | 20 × 16 | 相对位置、距离、算力、带宽、上行速率、"
            "通信时延、推理时延（/10 ms）、E2E 时延、Dev Val 质量先验、"
            "SNR、模型类型 one-hot |",
            "| 可用性 Mask | 20 | 单独的动作 Mask，用于屏蔽不可用 MEC |",
            "| **DQN 输入合计** | **322** | `2 + 20 × 16` |",
            "| **决策状态合计** | **342** | `322 + 20` |",
        ]
        state_fields = (
            "relative_x, relative_y, distance,\n"
            "compute_norm, bandwidth_norm, uplink_rate_norm,\n"
            "communication_ms_norm, inference_ms_norm, e2e_ms_norm,\n"
            "dev_val_quality_prior, snr_db_norm,\n"
            "model_Small, model_Transformer-S, model_Medium,\n"
            "model_Transformer-M, model_Large"
        )
        mlp_text = "322 → 256 → 256 → 20"
        order_text = (
            "| 训练任务 | 从 Train split 分层随机抽样 3500 条"
            "（1167 Short / 1167 Medium / 1166 Long），全局随机打乱 |"
        )
        topology_text = (
            "| 拓扑 | 训练用 Train topology；验证用 Dev topology；"
            "Final Test 用 Test topology |"
        )
    else:
        run = args.run_dir or DEFAULT_RUN
        output = args.output or DEFAULT_REPORT
        state_label = "No-Visual"
        state_dim = 342
        candidate_dim = 17
        decision_dim = 342
        state_title = (
            "本轮使用 no-visual 状态：DQN 输入 **342 维** = 2 维 UAV 位置 + "
            "20 × 17 维候选 MEC 特征。"
        )
        state_rows = [
            "| UAV 位置 | 2 | 当前二维位置 / 1500 m |",
            "| 候选 MEC | 20 × 17 | 每个候选的可用性、相对位置、距离、算力、"
            "带宽、上行速率、通信时延、推理时延（/10 ms）、E2E 时延、"
            "Dev Val 质量先验、SNR、模型类型 one-hot |",
            "| **合计** | **342** | `2 + 20 × 17` |",
        ]
        state_fields = (
            "availability, relative_x, relative_y, distance,\n"
            "compute_norm, bandwidth_norm, uplink_rate_norm,\n"
            "communication_ms_norm, inference_ms_norm, e2e_ms_norm,\n"
            "dev_val_quality_prior, snr_db_norm,\n"
            "model_Small, model_Transformer-S, model_Medium,\n"
            "model_Transformer-M, model_Large"
        )
        mlp_text = "342 → 256 → 256 → 20"
        order_text = (
            "| 训练任务 | 确定性场景×band 轮转；Short/Medium/Long 交错 |"
        )
        topology_text = (
            "| 拓扑 | 训练和验证均使用 Dev topology |"
        )
    result = read_json(run / "training_result.json")
    history = read_json(run / "training_history.json")
    final_test = read_json(run / "final_test" / "summary.json")
    dev_eval = read_json(run / "dev180_eval" / "summary.json")
    if result is None or history is None:
        raise SystemExit("Missing training_result/history")
    validation = history["validation"]
    config = result["config"]
    baseline_subset = {}
    baseline_usage = {}
    isolation = read_json(run / "validation_isolation.json")
    if isolation and BASELINE_DEV_CSV.is_file():
        validation_uids = {
            item["episode_uid"]
            for item in isolation["validation_episodes"]
        }
        with BASELINE_DEV_CSV.open(encoding="utf-8") as handle:
            baseline_rows = list(csv.DictReader(handle))
        for strategy in STRATEGIES:
            if strategy == "dqn":
                continue
            selected = [
                row for row in baseline_rows
                if row["strategy"] == strategy
                and row["episode_uid"] in validation_uids
            ]
            baseline_subset[strategy] = summarize_episodes(selected)
            usage = Counter()
            for row in selected:
                for name, count in json.loads(
                    row["model_usage"]
                ).items():
                    usage[name] += int(count)
            total = max(sum(usage.values()), 1)
            baseline_usage[strategy] = {
                name: float(usage[name] / total) for name in usage
            }
    overall = final_test["overall"] if final_test else None
    paired = (
        final_test.get("paired_dqn_minus_baseline", [])
        if final_test else []
    )

    lines = [
        f"# Stage-2k {state_label} DQN 关联实验报告",
        "",
        f"日期：{utc_now()[:10]}  ",
        "分支：`experiment/v83-spatial-foundation`  ",
        "仓库：`JLU-MCNS-MEC/mec-v8.2`  ",
        "性质：标准 MLP DQN；五种 T5 MEC 冻结；DQN 状态不含任何视觉信息。",
        "",
        "## 0. TL;DR",
        "",
        f"- 状态空间：**{state_dim} 维**"
        f"（2 维 UAV 位置 + 20 × {candidate_dim} 维 MEC 特征），"
        f"`state_mode = {config.get('state_mode')}`，"
        f"网络结构 `{config['architecture']}`。",
        f"- 实际训练 **{result['episodes_trained']} / {result['max_episodes']}** "
        f"Episodes；最佳 Dev Validation Episode "
        f"**{result['best_validation_episode']}**，"
        f"JointR **{fmt(result['best_validation_joint'])}**。",
        f"- 训练状态：**{result['status']}**；"
        f"early_stop_triggered = **{result['early_stop_triggered']}**；"
        f"converged = **{result['converged']}**；"
        f"停止原因：**{result['stop_reason']}**。",
        f"- 逐 Episode `|Return − JointR|` 最大 "
        f"**{result['reward_jointr_abs_error_max']:.3e}**，"
        f"平均 **{result['reward_jointr_abs_error_mean']:.3e}**。",
    ]
    if overall is not None:
        strongest = max(
            (s for s in STRATEGIES if s not in ("dqn", "oracle")),
            key=lambda s: overall[s]["JointR"],
        )
        lines += [
            f"- Association Final Test 180：DQN SR "
            f"**{100*overall['dqn']['SR']:.2f}%**、"
            f"nDTW **{fmt(overall['dqn']['nDTW'])}**、"
            f"E2E **{fmt(overall['dqn']['E2E_ms'],2)} ms**、"
            f"JointR **{fmt(overall['dqn']['JointR'])}**。",
            f"- 最强非特权 Baseline {TITLES[strongest]}："
            f"JointR **{fmt(overall[strongest]['JointR'])}**；"
            f"Oracle 上界 JointR **{fmt(overall['oracle']['JointR'])}**。",
        ]
        for item in paired:
            if item["strategy"] == strongest:
                lines.append(
                    f"- DQN − {TITLES[strongest]}：ΔJointR "
                    f"**{item['delta_JointR']['mean']:+.4f}** "
                    f"[{item['delta_JointR']['low']:+.4f}, "
                    f"{item['delta_JointR']['high']:+.4f}]，"
                    f"ΔSR **{item['delta_SR']['mean']:+.4f}**，"
                    f"ΔnDTW **{item['delta_nDTW']['mean']:+.4f}**，"
                    f"ΔE2E **{item['delta_E2E_ms']['mean']:+.2f} ms**。"
                )
    else:
        lines.append(
            "- Association Final Test 180：**未运行**（未收敛或未执行；"
            "不消耗一次性测试集）。"
        )
    lines += [
        "",
        "## 1. DQN 状态空间",
        "",
        state_title,
        "",
        "| 块 | 维度 | 内容 |",
        "|---|---:|---|",
        *state_rows,
        "",
        f"每个候选 MEC 的 {candidate_dim} 维字段：",
        "",
        "```",
        state_fields,
        "```",
        "",
        "推理时延使用专用 10 ms 归一化：统一模型中它只有约 0–10 ms，"
        "若按 100 ms 归一化会几乎恒为 0，无法提供有效区分度。",
        "",
        "**明确排除**：RGB/视觉 PCA 特征、Goal token、目标视觉相似度、"
        "历史视觉、热图、GAB、地图检索特征及其派生特征；导航动作、"
        "Episode 步数、目标位置、目标距离、Oracle action、成功标签。",
        "",
        f"网络结构保持标准 DQN MLP：`{mlp_text}`，"
        "不增加 Attention / GNN / Transformer 模块。",
        "",
        "## 2. 实验设计",
        "",
        "| 项 | 设置 |",
        "|---|---|",
        "| MEC | 五种冻结 Stage-2j T5 Checkpoint，不重新训练 |",
        "| Baseline | Random、Min-Latency、Max-Quality、Heuristic、Oracle |",
        "| 部署 | Balanced、Quality-rich、Tradeoff-stress 三种场景 |",
        "| 时延 | Stage-2j 八项输入实际字节 + 解析 FLOPs + 统一链路模型 |",
        "| 奖励 | `r_nav_t + (-0.25/100) × (running_mean_E2E_t − "
        "running_mean_E2E_{t−1})`，累计严格等于 JointR |",
        "| DQN 初始化 | 从头初始化，不复用旧 DQN 权重 |",
        f"| 训练预算 | 最长 {config['max_episodes']} Episodes；"
        "每 200 Episodes 独立 Dev 60 验证 |",
        order_text,
        topology_text,
        "| 早停 | 依据 Validation JointR 的新最佳与趋势；"
        "单次波动不触发早停 |",
        "",
        "## 3. 训练曲线",
        "",
        f"![raw return]({FIG}/dqn_return_raw.png)",
        "",
        f"![100-MA]({FIG}/dqn_return_ma100.png)",
        "",
        f"![validation JointR]({FIG}/dqn_validation_joint_reward.png)",
        "",
        f"![validation metrics]({FIG}/dqn_validation_metrics.png)",
        "",
        f"![reward components]({FIG}/dqn_reward_components.png)",
        "",
        f"![Q and loss]({FIG}/dqn_qvalue_loss.png)",
        "",
        "TensorBoard 记录：`train/episode_return`（真实训练奖励）、"
        "`train/episode_return_ma100`（100-Episode 移动平均）、"
        "`validation/joint_reward`、`validation/SR`、`validation/nDTW`、"
        "`validation/E2E_ms`、`train/loss`、`train/q_mean`、`train/q_max`。",
        "",
        "## 4. Dev Validation 记录",
        "",
        "| Episode | JointR | SR | nDTW | E2E (ms) | P95 (ms) | 步数 | Train Loss | Q mean | Q max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in validation:
        lines.append(
            f"| {row['episode']} | {fmt(row['joint_reward'])} | "
            f"{100*row['SR']:.2f}% | {fmt(row['nDTW'])} | "
            f"{fmt(row['E2E_ms'],2)} | {fmt(row['P95_ms'],2)} | "
            f"{fmt(row['mean_steps'],1)} | {fmt(row['train_loss_mean'],5)} | "
            f"{fmt(row['train_q_mean'],3)} | {fmt(row['train_q_max'],3)} |"
        )
    if baseline_subset and validation:
        best_row = max(validation, key=lambda row: row["joint_reward"])
        lines += [
            "",
            "### 4.1 同一组 Dev 60 上的 Baseline 对照",
            "",
            "以下 Baseline 指标只取与 DQN 验证完全相同的 60 个 Episode，"
            "不是 Dev180 总体；用于同口径判断 DQN 是否具有竞争力。",
            "",
            "| 策略 | JointR | SR | nDTW | E2E (ms) |",
            "|---|---:|---:|---:|---:|",
            f"| DQN (no visual, best Ep{best_row['episode']}) | "
            f"**{fmt(best_row['joint_reward'])}** | "
            f"{100*best_row['SR']:.2f}% | {fmt(best_row['nDTW'])} | "
            f"{fmt(best_row['E2E_ms'],2)} |",
        ]
        for strategy in STRATEGIES:
            if strategy == "dqn" or strategy not in baseline_subset:
                continue
            item = baseline_subset[strategy]
            lines.append(
                f"| {TITLES[strategy]} | {fmt(item['JointR'])} | "
                f"{100*item['SR']:.2f}% | {fmt(item['nDTW'])} | "
                f"{fmt(item['E2E_ms'],2)} |"
            )
        dqn_usage = best_row.get("model_usage", {})
        dqn_total = max(sum(dqn_usage.values()), 1)
        lines += [
            "",
            "MEC 类型使用率（同一组 60 Episodes）：",
            "",
            "| 策略 | " + " | ".join(MODEL_NAMES) + " |",
            "|---|" + "---:|" * len(MODEL_NAMES),
            "| DQN (best) | " + " | ".join(
                f"{100*dqn_usage.get(name, 0)/dqn_total:.1f}%"
                for name in MODEL_NAMES
            ) + " |",
        ]
        for strategy in STRATEGIES:
            if strategy == "dqn" or strategy not in baseline_usage:
                continue
            usage = baseline_usage[strategy]
            lines.append(
                f"| {TITLES[strategy]} | " + " | ".join(
                    f"{100*usage.get(name, 0):.1f}%" for name in MODEL_NAMES
                ) + " |"
            )
    lines += [
        "",
        "Validation 使用固定的 60 个 Dev Episode（每个场景 6 Short / 10 Medium / "
        "4 Long）；Episode UID、Episode Seed 与 Train 不相交。",
        "",
        "## 5. 最佳 Checkpoint 与收敛状态",
        "",
        f"- 最佳 Checkpoint：`{result['best_checkpoint']}`",
        f"- 最佳 Episode：**{result['best_validation_episode']}**",
        f"- 最佳 Dev Validation JointR：**{fmt(result['best_validation_joint'])}**",
        f"- 停止 Episode：**{result['stop_episode']}**",
        f"- 停止原因：**{result['stop_reason']}**",
        f"- 训练状态：**{result['status']}**",
        f"- `early_stop_triggered`：**{result['early_stop_triggered']}**",
        f"- `converged`：**{result['converged']}**",
        "",
        "状态定义：`improving` / `plateaued` / "
        "`early_stopped_no_improvement` / `converged` / `not_converged`。"
        "`converged` 需要最近验证形成稳定平台；早停本身不等于收敛。",
        "",
    ]
    if dev_eval is not None:
        dev = dev_eval["overall"]
        lines += [
            "## 6. 完整 Dev Val 180（辅助）",
            "",
            "| 策略 | SR | nDTW | E2E (ms) | JointR |",
            "|---|---:|---:|---:|---:|",
            f"| DQN (no visual) | {100*dev['SR']:.2f}% | "
            f"{fmt(dev['nDTW'])} | {fmt(dev['E2E_ms'],2)} | "
            f"{fmt(dev['JointR'])} |",
        ]
        for strategy in STRATEGIES:
            if strategy == "dqn":
                continue
            item = dev_eval["baselines"][strategy]
            lines.append(
                f"| {TITLES[strategy]} | {100*item['SR']:.2f}% | "
                f"{fmt(item['nDTW'])} | {fmt(item['E2E_ms'],2)} | "
                f"{fmt(item['JointR'])} |"
            )
        lines.append("")
    if overall is not None:
        lines += [
            "## 7. Association Final Test 180",
            "",
            "| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | 违约率 | 步数 | JointR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for strategy in STRATEGIES:
            item = overall[strategy]
            lines.append(
                f"| {TITLES[strategy]} | {100*item['SR']:.2f}% | "
                f"{fmt(item['nDTW'])} | {fmt(item['E2E_ms'],2)} | "
                f"{fmt(item['P95_ms'],2)} | "
                f"{100*item['delay_violation_rate']:.2f}% | "
                f"{fmt(item['mean_steps'],1)} | {fmt(item['JointR'])} |"
            )
        lines += [
            "",
            "### 逐 Episode 配对（DQN − Baseline）",
            "",
            "| Baseline | n | ΔSR | ΔnDTW | ΔE2E (ms) | ΔJointR [95% CI] | 胜/负/平 |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
        for item in paired:
            lines.append(
                f"| {TITLES[item['strategy']]} | {item['n']} | "
                f"{item['delta_SR']['mean']:+.4f} | "
                f"{item['delta_nDTW']['mean']:+.4f} | "
                f"{item['delta_E2E_ms']['mean']:+.2f} | "
                f"{item['delta_JointR']['mean']:+.4f} "
                f"[{item['delta_JointR']['low']:+.4f}, "
                f"{item['delta_JointR']['high']:+.4f}] | "
                f"{item['wins']}/{item['losses']}/{item['ties']} |"
            )
        lines += [
            "",
            "### 分部署条件（DQN）",
            "",
            "| 场景 | SR | nDTW | E2E (ms) | JointR |",
            "|---|---:|---:|---:|---:|",
        ]
        for scenario in SCENARIOS:
            item = final_test["by_scenario"][scenario]["dqn"]
            lines.append(
                f"| {SCENARIO_NAMES[scenario]} | "
                f"{100*item['SR']:.2f}% | {fmt(item['nDTW'])} | "
                f"{fmt(item['E2E_ms'],2)} | {fmt(item['JointR'])} |"
            )
        lines += [
            "",
            "![four metrics]({}/ASSOCIATION_FOUR_METRICS.png)".format(FIG),
            "",
            f"![SR]({FIG}/ASSOCIATION_SR.png)",
            "",
            f"![nDTW]({FIG}/ASSOCIATION_nDTW.png)",
            "",
            f"![E2E]({FIG}/ASSOCIATION_E2E_Delay.png)",
            "",
            f"![JointR]({FIG}/ASSOCIATION_JointR.png)",
            "",
        ]
    lines += [
        "## 8. 产物与提交",
        "",
        f"- 运行目录：`{run}`",
        "- Checkpoints：`checkpoints/last_full.pt`（含 Replay/RNG）、"
        "`checkpoints/best.pt`、`checkpoints/last.pt`",
        f"- TensorBoard：`runs/tensorboard/association_stage2k_"
        f"{'pure_mec_state' if state_mode == 'pure_mec' else 'no_visual'}/`",
        f"- 训练曲线：`{run}/plots/`",
        f"- 归档曲线：`v83_spatial_stage2/{FIG}/`",
        f"- 协议：`v83_spatial_stage2/"
        f"{'STAGE2K_PURE_MEC_PROTOCOL.md' if state_mode == 'pure_mec' else 'STAGE2K_NO_VISUAL_PROTOCOL.md'}`",
        "- 大型权重、Replay Buffer、TensorBoard Event 和逐 Episode CSV "
        "保留在服务器，不提交版本库。",
        "",
    ]
    if overall is not None:
        strongest = max(
            (s for s in STRATEGIES if s not in ("dqn", "oracle")),
            key=lambda s: overall[s]["JointR"],
        )
        strongest_delta = next(
            (item for item in paired if item["strategy"] == strongest),
            None,
        )
        lines += [
            "## 9. 结论",
            "",
            f"- Association Final Test 180 上，{state_label} DQN 的 JointR 为 "
            f"**{fmt(overall['dqn']['JointR'])}**，最强非特权 Baseline "
            f"{TITLES[strongest]} 为 **{fmt(overall[strongest]['JointR'])}**"
            + (
                f"；逐 Episode ΔJointR "
                f"**{strongest_delta['delta_JointR']['mean']:+.4f}** "
                f"[{strongest_delta['delta_JointR']['low']:+.4f}, "
                f"{strongest_delta['delta_JointR']['high']:+.4f}]。"
                if strongest_delta else "。"
            ),
            f"- DQN SR **{100*overall['dqn']['SR']:.2f}%**、"
            f"nDTW **{fmt(overall['dqn']['nDTW'])}**、"
            f"E2E **{fmt(overall['dqn']['E2E_ms'],2)} ms**；"
            f"相对 {TITLES[strongest]} 的 ΔSR "
            f"**{strongest_delta['delta_SR']['mean']:+.4f}**、ΔnDTW "
            f"**{strongest_delta['delta_nDTW']['mean']:+.4f}**、ΔE2E "
            f"**{strongest_delta['delta_E2E_ms']['mean']:+.2f} ms**。"
            if strongest_delta else "",
            "- 本轮为单 seed 结果，不构成多 seed 统计证据；Association "
            "Final Test 180 只在 Checkpoint 冻结后读取一次。",
            "",
        ]
    else:
        best_row = max(validation, key=lambda row: row["joint_reward"])
        lines += [
            "## 9. 结论",
            "",
            f"- 未进入 Final Test（converged=false）；同口径 Dev60 最佳 "
            f"Ep{best_row['episode']} JointR **{fmt(best_row['joint_reward'])}**、"
            f"SR {100*best_row['SR']:.2f}%、nDTW {fmt(best_row['nDTW'])}、"
            f"E2E {fmt(best_row['E2E_ms'],2)} ms。",
        ]
        if baseline_subset:
            strongest = max(
                (s for s in STRATEGIES if s not in ("dqn", "oracle")),
                key=lambda s: baseline_subset[s]["JointR"],
            )
            verdict = (
                "超过" if best_row["joint_reward"]
                > baseline_subset[strongest]["JointR"] else "未超过"
            )
            lines.append(
                f"- 同一组 60 Episode 上最强非特权 Baseline "
                f"{TITLES[strongest]} JointR "
                f"**{fmt(baseline_subset[strongest]['JointR'])}**；"
                f"{state_label} {verdict}该 Baseline。"
            )
        lines += [
            "- 未消耗一次性 Association Final Test 180。",
            "",
        ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", output)


if __name__ == "__main__":
    main()
