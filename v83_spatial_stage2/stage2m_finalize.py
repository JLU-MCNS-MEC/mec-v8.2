"""Build the final Stage-2m Chinese report from analysis + Q audits.

Run after all 16 training runs and the post-fix audits have finished:
    python v83_spatial_stage2/stage2m_finalize.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs/v83_spatial_stage2/stage2m_visual_fixed_city"
ANALYSIS = RUN_ROOT / "analysis"
AUDIT_ROOT = PROJECT_ROOT / "runs/v83_spatial_stage2/stage2m_q_audit"


def load(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def fmt(value, digits=4):
    if value is None:
        return "—"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if value != value or abs(value) == float("inf"):
        return "—"
    return f"{value:.{digits}f}"


def pct(value, digits=1):
    if value is None:
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def audit_table(audit, title):
    if not audit:
        return [f"## {title}", "", "（审计尚未完成）", ""]
    lines = [
        f"## {title}",
        "",
        "| Checkpoint | Ep | Q(s,a) mean | Q max mean | MC 回报 G_t | 偏差 mean | MAE | 高估比例 | Pearson | 策略 JointR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in audit.get("on_policy", []):
        m = item["metrics"]
        policy = item.get("policy", {})
        lines.append(
            f"| {item['name']} | {item.get('episode')} | "
            f"{fmt(m.get('q_selected_mean'))} | "
            f"{fmt(m.get('q_max_mean'))} | "
            f"{fmt(m.get('mc_return_mean'))} | "
            f"{fmt(m.get('bias_mean'))} | {fmt(m.get('mae'))} | "
            f"{pct(m.get('overestimate_fraction'))} | "
            f"{fmt(m.get('pearson_q_vs_mc'), 3)} | "
            f"{fmt(policy.get('JointR'))} |"
        )
    lines += [
        "",
        "共同 Min-Latency 状态集（跨算法相对比较）：",
        "",
        "| Checkpoint | Q(s,a) mean | Q max mean | G_t mean | 偏差 | MAE | 高估比例 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in audit.get("fixed_behavior", []):
        m = item["metrics"]
        lines.append(
            f"| {item['name']} | {fmt(m.get('q_selected_mean'))} | "
            f"{fmt(m.get('q_max_mean'))} | {fmt(m.get('mc_return_mean'))} | "
            f"{fmt(m.get('bias_mean'))} | {fmt(m.get('mae'))} | "
            f"{pct(m.get('overestimate_fraction'))} |"
        )
    lines += [
        "",
        "实现审计：",
        "",
        "| Checkpoint | 终止转换 | 终止带 next_state | 终止 bootstrap=0 | 掩码均值 | 全 False | 幻影 0 | legacy−fixed MAE | double−fixed MAE |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in audit.get("target_audit", []):
        lines.append(
            f"| {Path(item['checkpoint']).parent.parent.name + '/' + Path(item['checkpoint']).name} | "
            f"{item.get('done_transitions')} | "
            f"{item.get('done_with_next_state')} | "
            f"{item.get('terminal_bootstrap_zero')} | "
            f"{fmt(item.get('mask_mean'), 2)} | "
            f"{item.get('all_false_mask')} | "
            f"{item.get('phantom_zero_transitions')} | "
            f"{fmt(item.get('legacy_minus_fixed_mae'), 5)} | "
            f"{fmt(item.get('double_minus_fixed_mae'), 5)} |"
        )
    lines += [
        "",
        f"奖励一致性最大误差：`{audit.get('max_reward_error'):.2e}`；"
        f"Final Test read = `{audit.get('final_test_read')}`。",
        "",
    ]
    return lines


def main():
    analysis_path = ANALYSIS / "analysis.json"
    if not analysis_path.is_file() or not load(analysis_path).get("runs"):
        raise SystemExit(
            "analysis.json missing; run stage2m_analyze.py after training"
        )
    analysis = load(analysis_path)
    runs = analysis["runs"]
    paired = analysis.get("paired_visual", [])
    dqn_rows = analysis.get("dqn_vs_ddqn", [])
    by_arm = {}
    for row in runs:
        by_arm.setdefault(row["arm"], []).append(row)

    pre = load(AUDIT_ROOT / "pre/q_audit.json")
    post_dqn = load(AUDIT_ROOT / "post_dqn/q_audit.json")
    post_visual = load(AUDIT_ROOT / "post_visual/q_audit.json")
    post_visual_last = load(AUDIT_ROOT / "post_visual_last/q_audit.json")

    def mean(arm, key):
        values = [
            row[key] for row in by_arm.get(arm, [])
            if row.get(key) is not None
        ]
        return sum(values) / len(values) if values else None

    lines = [
        "# Stage-2m 最终报告：DQN Q 高估诊断 + 固定城市 A 拓扑视觉增强",
        "",
        "- 分支：`experiment/v83-spatial-foundation`",
        "- 协议：`v83_spatial_stage2/STAGE2M_EXPERIMENT_PROTOCOL.md`",
        "- 运行根：`runs/v83_spatial_stage2/stage2m_visual_fixed_city/`",
        "- Q 审计：`runs/v83_spatial_stage2/stage2m_q_audit/`",
        "- 固定城市 `map_seed=10,000,001`，A 拓扑，2000 Episodes，3 seeds，"
        "五种 T5 MEC 冻结，Final Test 未读取。",
        "",
        "## 0. 四组训练结果",
        "",
        "| Arm | Best Dev60 JointR | Best SR | Best SPL | Best nDTW | Final MA100 JointR | Q mean (last100) | TD loss (last100) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("dqn_pure", "dqn_visual", "ppo_pure", "ppo_visual"):
        lines.append(
            f"| {arm} | {fmt(mean(arm, 'best_validation_jointr'))} | "
            f"{pct(mean(arm, 'best_validation_SR'))} | "
            f"{fmt(mean(arm, 'best_validation_SPL'))} | "
            f"{fmt(mean(arm, 'best_validation_nDTW'))} | "
            f"{fmt(mean(arm, 'final_train_ma100_jointr'))} | "
            f"{fmt(mean(arm, 'q_mean_last100'))} | "
            f"{fmt(mean(arm, 'td_loss_last100'))} |"
        )
    lines += [
        "",
        "## 1. 配对视觉增益（同 seed：visual − pure）",
        "",
        "| Algorithm | Seed | ΔBest JointR | ΔBest SR | ΔBest SPL | ΔBest nDTW | ΔFinal MA100 JointR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['algorithm']} | {row['seed']} | "
            f"{fmt(row.get('delta_best_jointr'))} | "
            f"{fmt(row.get('delta_best_SR'))} | "
            f"{fmt(row.get('delta_best_SPL'))} | "
            f"{fmt(row.get('delta_best_nDTW'))} | "
            f"{fmt(row.get('delta_train_ma100_jointr'))} |"
        )
    for algorithm in ("dqn", "ppo"):
        values = [
            row["delta_best_jointr"] for row in paired
            if row["algorithm"] == algorithm
            and row.get("delta_best_jointr") is not None
        ]
        if values:
            lines.append(
                f"| {algorithm} 均值 | — | "
                f"{fmt(sum(values) / len(values))} | | | | |"
            )
    lines += [
        "",
        "## 2. DQN vs Double DQN",
        "",
        "| Algorithm | Seed | Best JointR | Final JointR | MA100 JointR | Best SR | Q mean | Q max | TD loss | TD |Q−target| |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dqn_rows:
        lines.append(
            f"| {row['algorithm']} | {row['seed']} | "
            f"{fmt(row.get('best_jointr'))} | {fmt(row.get('final_jointr'))} | "
            f"{fmt(row.get('train_ma100_jointr'))} | {fmt(row.get('best_SR'))} | "
            f"{fmt(row.get('q_mean_last100'))} | "
            f"{fmt(row.get('q_max_last100'))} | "
            f"{fmt(row.get('td_loss_last100'))} | "
            f"{fmt(row.get('td_error_abs_last100'))} |"
        )
    lines += [
        "",
        "## 3. 收敛与稳定性",
        "",
        "| Arm | Best Episode | First within 0.01 of best | Last-5 JointR std | Last-5 range | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm in ("dqn_pure", "dqn_visual", "ppo_pure", "ppo_visual",
                "dqn_double", "dqn_legacy"):
        values = by_arm.get(arm, [])
        if not values:
            continue
        best_episodes = [
            row["best_validation_episode"] for row in values
            if row.get("best_validation_episode") is not None
        ]
        near = [
            row["first_episode_within_0.01_of_best"] for row in values
            if row.get("first_episode_within_0.01_of_best") is not None
        ]
        lines.append(
            f"| {arm} | {fmt(sum(best_episodes) / len(best_episodes) if best_episodes else None, 1)} | "
            f"{fmt(sum(near) / len(near) if near else None, 1)} | "
            f"{fmt(mean(arm, 'last5_jointr_std'))} | "
            f"{fmt(mean(arm, 'last5_jointr_range'))} | "
            f"{', '.join(str(row.get('result_status')) for row in values)} |"
        )
    lines += [
        "",
        "## 4. 修复前 Q 审计（Stage-2l A/DQN）",
        "",
    ]
    lines += audit_table(pre, "修复前（旧 target）")
    lines += audit_table(post_dqn, "修复后：DQN / Double DQN / legacy 对照")
    lines += audit_table(post_visual, "视觉 DQN best Checkpoint")
    lines += audit_table(post_visual_last, "视觉 DQN last Checkpoint（尺度漂移）")
    lines += [
        "## 5. 曲线",
        "",
        "![training return](figures/return_ma100.png)",
        "",
        "![validation JointR](figures/val_jointr.png)",
        "",
        "![validation SR](figures/val_SR.png)",
        "",
        "![DQN Q and TD](figures/dqn_q_td.png)",
        "",
        "![PPO diagnostics](figures/ppo_diagnostics.png)",
        "",
        "## 6. 结论",
        "",
        "（由最终数据人工填写；见 `STAGE2M_ANSWERS.md` 与对话回复。）",
        "",
    ]
    (ANALYSIS / "STAGE2M_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"wrote {ANALYSIS / 'STAGE2M_RESULTS.md'}", flush=True)
    print("STAGE2M_FINALIZE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
