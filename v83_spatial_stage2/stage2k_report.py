"""Generate the Stage-2k Chinese reports from the frozen result JSONs."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BASELINE_DIR,
    DQN_DIR,
    FINAL_TEST_DIR,
    QUALITY_PRIOR_PATH,
    SCENARIOS,
    SCENARIO_TITLES,
    read_json,
    utc_now,
)


STRATEGY_TITLES = {
    "dqn": "DQN",
    "random": "Random",
    "min_latency": "Min-Latency",
    "max_quality": "Max-Quality",
    "heuristic": "Weighted Greedy",
    "oracle": "Oracle（特权上界）",
}
ORDER = ("dqn", "random", "min_latency", "max_quality", "heuristic", "oracle")


def fmt_ci(item, percent=False, digits=4):
    value = item["mean"]
    low, high = item["low"], item["high"]
    if percent:
        return (
            f"{100*value:.2f} [{100*low:.2f}, {100*high:.2f}]"
        )
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def main():
    final = read_json(FINAL_TEST_DIR / "summary.json")
    baseline = read_json(BASELINE_DIR / "dev" / "summary.json")
    training = read_json(DQN_DIR / "training_result.json")
    quality = read_json(QUALITY_PRIOR_PATH)
    if final is None or baseline is None or training is None or quality is None:
        raise SystemExit("Missing final/baseline/training/quality results")

    lines = [
        "# Stage-2k — 基于五种冻结 T5 MEC 的 DQN 关联实验",
        "",
        f"日期：{utc_now()[:10]}  ",
        "分支：`experiment/v83-spatial-foundation`  ",
        "仓库：`JLU-MCNS-MEC/mec-v8.2`",
        "",
        "Nature: 在五种 **冻结的 Stage-2j T5 MEC** 之上训练一个紧凑 DQN "
        "关联智能体；不训练、不微调、不重新初始化任何 MEC。Association "
        "Final Test 180 在 DQN 收敛并冻结后只评估一次。",
        "",
        "## 0. TL;DR",
        "",
    ]
    dqn = final["overall"]["dqn"]
    heur = final["overall"]["heuristic"]
    lines += [
        f"- DQN 训练实际完成 **{training['episodes_trained']} / 4000** "
        f"Episodes；停止原因：**{training['stop_reason']}**；"
        f"是否收敛：**{'是' if training['converged'] else '否'}**。",
        f"- Association Final Test 180：DQN SR **{100*dqn['SR']:.2f}%**、"
        f"nDTW **{dqn['nDTW']:.4f}**、E2E **{dqn['E2E_ms']:.2f} ms**、"
        f"JointR **{dqn['JointR']:.4f}**。",
        f"- Weighted Greedy：SR **{100*heur['SR']:.2f}%**、"
        f"nDTW **{heur['nDTW']:.4f}**、E2E **{heur['E2E_ms']:.2f} ms**、"
        f"JointR **{heur['JointR']:.4f}**。",
        f"- DQN − Weighted Greedy：ΔSR **{100*(dqn['SR']-heur['SR']):+.2f} pp**、"
        f"ΔnDTW **{dqn['nDTW']-heur['nDTW']:+.4f}**、"
        f"ΔE2E **{dqn['E2E_ms']-heur['E2E_ms']:+.2f} ms**、"
        f"ΔJointR **{dqn['JointR']-heur['JointR']:+.4f}**。",
        "- 所有策略使用同一 180 Episode 清单、同一三种部署拓扑、同一 E2E "
        "时延模型与同一 JointR。Oracle 仅作为特权信息上界。",
        "- 五种 MEC 的输入仍是 Stage-2j T5：Current/Goal Token、16 步历史、"
        "相对位姿/动作、GAB、22-D 匿名地标图。没有新增 Heatmap 或 "
        "Network-State 分支。",
        "",
        "## 1. MEC 冻结与数据隔离",
        "",
        "| MEC | 参数量 | Checkpoint SHA256 前 16 位 | Dev Val mean nDTW |",
        "|---|---:|---|---:|",
    ]
    parameter_counts = {
        "Small": 611716,
        "Transformer-S": 1332868,
        "Medium": 1991044,
        "Transformer-M": 3033732,
        "Large": 5248932,
    }
    for name in ("Small", "Transformer-S", "Medium", "Transformer-M", "Large"):
        item = quality["models"][name]
        lines.append(
            f"| {name} | {parameter_counts[name]:,} | "
            f"`{item['checkpoint_sha256'][:16]}` | {item['mean_ndtw']:.4f} |"
        )
    reason_cn = {
        "reached episode cap": "达到 4,000 Episode 硬上限",
        "training loop ended": "训练循环结束",
    }.get(training["stop_reason"], training["stop_reason"])
    lines += [
        "",
        "- 五个 checkpoint 均以 `strict=True` 加载，固定输入两次前向输出逐元素一致。",
        "- MEC 参数 `requires_grad=False`，始终 `eval()`；优化器只包含 DQN 参数。",
        "- Association Train/Dev/Test episode seeds 分别为 5,000,000+ / "
        "6,000,000+ / 7,000,000+，以及独立 topology seeds；与 Stage-2j "
        "Train 600、Dev Val 180、Final Test 300 均无交集。",
        "- Stage-2j Final Test 300 token 目录从未被读取；仅使用已导出的任务 "
        "metadata 做 seed 交集检查。Association Final Test 180 在 DQN 完成后"
        "冻结源码、checkpoint、spec 与 topology，并且只评估一次。",
        "- Association Final Test 180 是唯一读取的最终测试；没有使用 Stage-2j "
        "Final Test 300 的成绩进行模型选择、调参或关联策略设计。",
        "",
        "## 2. 部署场景",
        "",
        "| 场景 | 定义 |",
        "|---|---|",
        "| Balanced / Original | 五种模型各 4 个节点；带宽与算力均衡 |",
        "| Quality-rich | Dev Val 前 3 名模型占 15 个节点；高质量节点配 "
        "2 TFLOPS + 30 MHz |",
        "| Tradeoff-stress | 前 3 名模型配 0.5 TFLOPS + 5 MHz，"
        "后 2 名模型配 2 TFLOPS + 30 MHz，制造质量—时延冲突 |",
        "",
        "Dev Val 质量排名："
        + " > ".join(quality["rank"])
        + "。质量先验只来自 Stage-2j Dev Val 180 closed-loop mean nDTW，"
        "没有使用 Stage-2j Final Test 300 成绩。",
        "",
        "## 3. Baseline（Dev Validation 180）",
        "",
        "| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | JointR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy in ("random", "min_latency", "max_quality", "heuristic", "oracle"):
        item = baseline["overall"][strategy]
        lines.append(
            f"| {STRATEGY_TITLES[strategy]} | {100*item['SR']:.2f}% | "
            f"{item['nDTW']:.4f} | {item['E2E_ms']:.2f} | "
            f"{item['P95_ms']:.2f} | {item['JointR']:.4f} |"
        )
    lines += [
        "",
        "完整分场景表见 `STAGE2K_BASELINE_RESULTS.md`。Oracle 是特权上界，"
        "不是可部署策略。",
        "",
        "## 4. DQN 训练",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 实际 Episodes | {training['episodes_trained']} / 4000 |",
        f"| 停止原因 | {reason_cn} |",
        f"| 是否收敛 | {'是' if training['converged'] else '否'} |",
        f"| 最佳 Validation JointR | {training['best_validation_joint']:.4f} |",
        f"| Best Checkpoint | `{training['best_checkpoint']}` |",
        f"| Best Checkpoint SHA256 | `{final['freeze']['dqn_checkpoint_sha256'][:16]}` |",
        f"| 网络 | `mlp_478_256_256_20`（标准 MLP DQN，无 Attention/GNN/RNN） |",
        f"| Seed | {training['seed']} |",
        "",
        "训练曲线：`runs/v83_spatial_stage2/stage2k/dqn/plots/"
        "training_return_ma100.png` 与 `validation_jointr.png`。"
        "TensorBoard 日志：`runs/tensorboard/association_stage2k/`。",
        "",
        "## 5. Association Final Test 180",
        "",
        "| 策略 | SR [95% CI] | nDTW [95% CI] | E2E (ms) [95% CI] | "
        "JointR [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy in ORDER:
        item = final["bootstrap_95ci"][strategy]
        lines.append(
            f"| {STRATEGY_TITLES[strategy]} | "
            f"{fmt_ci(item['SR'], percent=True)} | "
            f"{fmt_ci(item['nDTW'])} | "
            f"{fmt_ci(item['E2E_ms'], digits=2)} | "
            f"{fmt_ci(item['JointR'])} |"
        )
    lines += [
        "",
        "辅助指标：",
        "",
        "| 策略 | P95 (ms) | 时延违约率 | 平均步数 |",
        "|---|---:|---:|---:|",
    ]
    for strategy in ORDER:
        item = final["overall"][strategy]
        lines.append(
            f"| {STRATEGY_TITLES[strategy]} | {item['P95_ms']:.2f} | "
            f"{100*item['delay_violation_rate']:.2f}% | "
            f"{item['mean_steps']:.1f} |"
        )
    lines += [
        "",
        "模型使用分布（按决策步数占比）：",
        "",
        "| 策略 | Small | Transformer-S | Medium | Transformer-M | Large |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy in ORDER:
        ratios = final["overall"][strategy].get("model_usage_ratio", {})
        lines.append(
            f"| {STRATEGY_TITLES[strategy]} | "
            + " | ".join(
                f"{100*float(ratios.get(name, 0.0)):.1f}%"
                for name in ("Small", "Transformer-S", "Medium",
                             "Transformer-M", "Large")
            )
            + " |"
        )
    lines += [
        "",
        "### 5.0 正式测试图",
        "",
        "![Success Rate](figures/stage2k/ASSOCIATION_SR.png)",
        "",
        "![nDTW](figures/stage2k/ASSOCIATION_nDTW.png)",
        "",
        "![E2E Delay](figures/stage2k/ASSOCIATION_E2E_Delay.png)",
        "",
        "![JointR](figures/stage2k/ASSOCIATION_JointR.png)",
        "",
        "### 5.1 DQN 相对 Weighted Greedy 的配对差异",
        "",
        "| 指标 | DQN − Weighted Greedy [95% CI] |",
        "|---|---:|",
    ]
    paired = final["paired_dqn_minus_heuristic"]
    for key, title in (
        ("SR", "SR (pp)"),
        ("nDTW", "nDTW"),
        ("E2E_ms", "E2E (ms)"),
        ("JointR", "JointR"),
    ):
        item = paired[key]
        if key == "SR":
            value = (
                f"{100*item['mean']:+.2f} "
                f"[{100*item['low']:+.2f}, {100*item['high']:+.2f}]"
            )
        elif key == "E2E_ms":
            value = (
                f"{item['mean']:+.2f} "
                f"[{item['low']:+.2f}, {item['high']:+.2f}]"
            )
        else:
            value = (
                f"{item['mean']:+.4f} "
                f"[{item['low']:+.4f}, {item['high']:+.4f}]"
            )
        lines.append(f"| {title} | {value} |")
    lines += [
        "",
        "## 6. 优势与局限",
        "",
        "- **优势**：DQN 在质量—时延冲突场景中学习节点关联，而不是只按单一 "
        "质量或单一最低时延选择；其目标函数与评估 JointR 一致（nDTW 与 E2E "
        "的固定权重），并使用 Dev Val 质量先验和预测 E2E。",
        "- **局限**：单 seed、单轮 DQN；Replay Buffer 在恢复训练时会重新初始化；"
        "质量先验来自 Dev Val 180，存在开发集偏差；E2E 是解析式模型，"
        "使用 470,200 B/步的保守全量上传假设，未建模 MEC 侧历史缓存。",
        "- DQN 的状态不含目标真值、真实距离、Oracle Action 或成功标签；"
        "导航奖励使用现有 `NavigationRewardTracker` 的参考轨迹增量，"
        "这是训练期特权奖励，已在 `STAGE2K_STATE_REWARD_DESIGN.md` 明确说明。",
        "- Oracle 只用于上界比较，不参与 DQN 训练或状态构造。",
        "- Stage-2j 的匿名地标检索在 clean simulator 中 Top-1/Top-5 为 100%，"
        "因此 MEC 导航本身是 **map-assisted** 的，不能解释为纯视觉导航；"
        "本实验比较的是在相同地图辅助 MEC 之上的关联策略。",
        "",
        "## 7. 复现",
        "",
        "```bash",
        "# checkpoint / input audit",
        "python3 -m unittest -v v83_spatial_stage2.test_stage2k",
        "# Generate specs and Train-only PCA",
        "python3 v83_spatial_stage2/stage2k_build_specs.py",
        "python3 v83_spatial_stage2/stage2k_fit_pca.py",
        "# Dev Val quality prior (five frozen MECs)",
        "python3 v83_spatial_stage2/stage2k_build_quality_prior.py",
        "python3 v83_spatial_stage2/stage2k_build_topologies.py",
        "# Baselines",
        "python3 v83_spatial_stage2/stage2k_baselines.py",
        "# DQN",
        "python3 v83_spatial_stage2/stage2k_dqn.py",
        "# Final test (frozen, once)",
        "python3 v83_spatial_stage2/stage2k_test.py",
        "python3 v83_spatial_stage2/stage2k_report.py",
        "```",
        "",
        "## 8. Artifacts",
        "",
        "- 设计：`v83_spatial_stage2/STAGE2K_STATE_REWARD_DESIGN.md`",
        "- Baseline：`v83_spatial_stage2/STAGE2K_BASELINE_RESULTS.md`",
        "- 四张正式图：`v83_spatial_stage2/figures/stage2k/ASSOCIATION_*.png/.svg`",
        "- 训练曲线：`v83_spatial_stage2/figures/stage2k/TRAINING_RETURN_MA100.png`、"
        "`VALIDATION_JOINTR.png`",
        "- Checkpoint 审计：`runs/v83_spatial_stage2/stage2k/checkpoint_audit.json`",
        "- 服务器产物：`runs/v83_spatial_stage2/stage2k/`"
        "（specs、topologies、quality_dev_val、baselines、dqn、final_test）",
        "- TensorBoard：`runs/tensorboard/association_stage2k/`",
        "",
    ]
    output = Path("v83_spatial_stage2/STAGE2K_ASSOCIATION_RL_REPORT.md")
    output.write_text("\n".join(lines), encoding="utf-8")

    baseline_src = BASELINE_DIR / "STAGE2K_BASELINE_RESULTS.md"
    baseline_dst = Path("v83_spatial_stage2/STAGE2K_BASELINE_RESULTS.md")
    if baseline_src.is_file():
        shutil.copyfile(baseline_src, baseline_dst)
    print(json.dumps({
        "report": str(output),
        "baseline_report": str(baseline_dst),
        "final_test_read": True,
    }, indent=2), flush=True)
    print("STAGE2K_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
