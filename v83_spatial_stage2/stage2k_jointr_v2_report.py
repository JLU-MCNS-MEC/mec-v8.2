"""Generate the Chinese Stage-2k JointR-aligned v2 training report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import read_json, utc_now

DEFAULT_RUN = Path("runs/v83_spatial_stage2/stage2k_jointr_v2")
DEFAULT_REPORT = Path("v83_spatial_stage2/STAGE2K_JOINTR_ALIGNED_TRAINING_REPORT.md")
FIG = "figures/stage2k_jointr_v2"


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    run = args.run_dir
    result = read_json(run / "training_result.json")
    history = read_json(run / "training_history.json")
    evaluation = read_json(run / "dev180_eval" / "summary.json")
    if result is None or history is None or evaluation is None:
        raise SystemExit("Missing training_result/history/dev180_eval")
    overall = evaluation["overall"]
    baselines = evaluation["baselines"]
    validation = history["validation"]

    lines = [
        "# Stage-2k JointR-Aligned v2 训练报告",
        "",
        f"日期：{utc_now()[:10]}  ",
        "分支：`experiment/v83-spatial-foundation`  ",
        "仓库：`JLU-MCNS-MEC/mec-v8.2`  ",
        "性质：标准 MLP DQN；五种 T5 MEC 冻结；未读取 Final Test。",
        "",
        "## 0. TL;DR",
        "",
        f"- 实际训练 **{result['episodes_trained']} / 4000** Episodes；"
        f"最佳 Dev Validation Episode **{result['best_validation_episode']}**，"
        f"JointR **{fmt(result['best_validation_joint'])}**。",
        f"- 训练状态：**{result['status']}**；"
        f"early_stop_triggered = **{result['early_stop_triggered']}**；"
        f"converged = **{result['converged']}**。",
        f"- 逐 Episode `|Return − JointR|` 最大 "
        f"**{result['reward_jointr_abs_error_max']:.3e}**，"
        f"平均 **{result['reward_jointr_abs_error_mean']:.3e}**。",
        f"- 完整 Dev Val 180：SR **{100*overall['SR']:.2f}%**、"
        f"nDTW **{fmt(overall['nDTW'])}**、E2E **{fmt(overall['E2E_ms'],2)} ms**、"
        f"P95 **{fmt(overall['P95_ms'],2)} ms**、"
        f"JointR **{fmt(overall['JointR'])}**。",
        f"- 最强非特权 Baseline Heuristic：JointR "
        f"**{fmt(baselines['heuristic']['JointR'])}**。",
        f"- 是否进入 Association Final Test 180：**否**；"
        f"final_test_read = **false**。",
        "",
        "## 1. 本轮核心修改",
        "",
        "| 项 | v1 | v2 |",
        "|---|---|---|",
        "| 时延奖励 | `-0.25*min(E2E/100,3)/200` 每步 | 运行平均时延的增量，"
        "累计严格等于 JointR 时延项 |",
        "| `gamma` | 0.99 | **1.0** |",
        "| 状态第 6 个全局特征 | 上一步 E2E/100 | **running_mean_E2E/100** |",
        "| 训练期 Validation | 误用 train Episode | **60 个独立 Dev Episode** |",
        "| 训练顺序 | short→medium→long 分块 | **9 组确定性轮转，每 200 窗口覆盖 9 组** |",
        "| Checkpoint | 无 Replay/RNG | **last_full.pt 含 Replay/RNG/epsilon/episode** |",
        "",
        "奖励推导与测试详见 `v83_spatial_stage2/STAGE2K_JOINTR_REWARD_DESIGN.md`。",
        "",
        "## 2. 奖励严格一致性",
        "",
        "```",
        "mean_t = mean_{t-1} + (d_t - mean_{t-1}) / t",
        "r_delay_t = -0.25/100 * (mean_t - mean_{t-1})",
        "sum_t(r_t) = final_nDTW - 0.25*mean_E2E/100 = Episode JointR",
        "```",
        "",
        "| 检查 | 结果 |",
        "|---|---:|",
        "| 900 条 Baseline 逐 Episode 复算最大误差 | **3.33e-16** |",
        f"| 本次训练逐 Episode 最大误差 | **{result['reward_jointr_abs_error_max']:.3e}** |",
        f"| 本次训练平均误差 | **{result['reward_jointr_abs_error_mean']:.3e}** |",
        f"| Dev180 评估最大误差 | **{evaluation['eval_jointr_abs_error_max']:.3e}** |",
        "",
        "单元测试：`v83_spatial_stage2/test_stage2k_jointr_v2.py`，"
        "覆盖单步、多步、恒定/变化时延、400 步、900 条真实 Baseline、"
        "Dev/Train 隔离、Replay/RNG 断点往返。",
        "",
        "## 3. 训练曲线",
        "",
        f"![raw return]({FIG}/dqn_v2_return_raw.png)",
        "",
        f"![100-MA]({FIG}/dqn_v2_return_ma100.png)",
        "",
        f"![validation JointR]({FIG}/dqn_v2_validation_joint_reward.png)",
        "",
        f"![validation metrics]({FIG}/dqn_v2_validation_metrics.png)",
        "",
        f"![reward components]({FIG}/dqn_v2_reward_components.png)",
        "",
        f"![Q and loss]({FIG}/dqn_v2_qvalue_loss.png)",
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
        f"- 停止原因：**{result['stop_reason']}**",
        f"- 训练状态：**{result['status']}**",
        f"- `early_stop_triggered`：**{result['early_stop_triggered']}**",
        f"- `converged`：**{result['converged']}**",
        "",
        "状态定义：`improving` / `plateaued` / "
        "`early_stopped_no_improvement` / `converged` / `not_converged`。"
        "`converged` 需要最近验证形成稳定平台；早停本身不等于收敛。",
        "",
        "## 6. 完整 Dev Val 180 对比",
        "",
        "| 策略 | SR | OSR | SPL | nDTW | E2E (ms) | P95 (ms) | 违约率 | 步数 | JointR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| **DQN v2** | {100*overall['SR']:.2f}% | "
        f"{100*overall['OSR']:.2f}% | {fmt(overall['SPL'])} | "
        f"{fmt(overall['nDTW'])} | {fmt(overall['E2E_ms'],2)} | "
        f"{fmt(overall['P95_ms'],2)} | "
        f"{100*overall['delay_violation_rate']:.2f}% | "
        f"{fmt(overall['mean_steps'],1)} | **{fmt(overall['JointR'])}** |",
    ]
    titles = {
        "random": "Random", "min_latency": "Min-Latency",
        "max_quality": "Max-Quality", "heuristic": "Heuristic",
        "oracle": "Oracle (privileged)",
    }
    for strategy in ("random", "min_latency", "max_quality", "heuristic", "oracle"):
        item = baselines[strategy]
        lines.append(
            f"| {titles[strategy]} | {100*item['SR']:.2f}% | — | — | "
            f"{fmt(item['nDTW'])} | {fmt(item['E2E_ms'],2)} | "
            f"{fmt(item['P95_ms'],2)} | "
            f"{100*item['delay_violation_rate']:.2f}% | "
            f"{fmt(item['mean_steps'],1)} | {fmt(item['JointR'])} |"
        )
    lines += [
        "",
        "### 逐 Episode 配对（DQN − Baseline）",
        "",
        "| Baseline | n | ΔJointR [95% CI] | ΔnDTW | ΔSR | ΔE2E (ms) | 胜/负/平 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in evaluation["paired"]:
        lines.append(
            f"| {titles[item['strategy']]} | {item['n']} | "
            f"{item['delta_JointR']['mean']:+.4f} "
            f"[{item['delta_JointR']['low']:+.4f}, "
            f"{item['delta_JointR']['high']:+.4f}] | "
            f"{item['delta_nDTW']['mean']:+.4f} | "
            f"{item['delta_SR']['mean']:+.4f} | "
            f"{item['delta_E2E_ms']['mean']:+.2f} | "
            f"{item['wins']}/{item['losses']}/{item['ties']} |"
        )
    lines += [
        "",
        "### 分部署条件（DQN）",
        "",
        "| 场景 | SR | nDTW | E2E (ms) | P95 (ms) | JointR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in ("balanced", "quality_rich", "tradeoff_stress"):
        item = evaluation["by_scenario"][scenario]
        lines.append(
            f"| {scenario} | {100*item['SR']:.2f}% | {fmt(item['nDTW'])} | "
            f"{fmt(item['E2E_ms'],2)} | {fmt(item['P95_ms'],2)} | "
            f"{fmt(item['JointR'])} |"
        )
    lines += [
        "",
        "### MEC 类型使用率（DQN Dev180）",
        "",
        "| MEC | 比例 |",
        "|---|---:|",
    ]
    for name, ratio in sorted(
        overall["model_usage_ratio"].items(), key=lambda pair: -pair[1]
    ):
        lines.append(f"| {name} | {100*ratio:.2f}% |")
    lines += [
        "",
        "## 7. 是否进入 Final Test",
        "",
        f"- Heuristic JointR：**{fmt(baselines['heuristic']['JointR'])}**",
        f"- DQN v2 JointR：**{fmt(overall['JointR'])}**",
        "- 本轮**不进入 Association Final Test 180**，不读取 Stage-2j "
        "Final Test 300；完成后暂停，等待下一步决定。",
        "",
        "## 8. 产物与提交",
        "",
        f"- 运行目录：`{run}`",
        "- Checkpoints：`checkpoints/last_full.pt`（含 Replay/RNG）、"
        "`checkpoints/best.pt`、`checkpoints/last.pt`",
        "- TensorBoard：`runs/tensorboard/association_stage2k_jointr_v2/`",
        "- 曲线：`runs/v83_spatial_stage2/stage2k_jointr_v2/plots/`",
        f"- 归档曲线：`v83_spatial_stage2/{FIG}/`",
        "- 未提交权重、Replay Buffer 或 TensorBoard Event 文件。",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
