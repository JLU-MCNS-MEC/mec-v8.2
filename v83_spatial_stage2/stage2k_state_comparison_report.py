"""Chinese comparison report: no-visual vs pure-MEC state designs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    MODEL_NAMES,
    read_json,
    summarize_episodes,
    utc_now,
)

NO_VISUAL_RUN = Path("runs/v83_spatial_stage2/stage2k_no_visual")
PURE_MEC_RUN = Path("runs/v83_spatial_stage2/stage2k_pure_mec_state")
BASELINE_DEV_CSV = Path(
    "runs/v83_spatial_stage2/stage2k/baselines/dev/per_episode.csv"
)
DEFAULT_REPORT = Path(
    "v83_spatial_stage2/STAGE2K_STATE_DESIGN_COMPARISON.md"
)
FIG = Path("v83_spatial_stage2/figures/stage2k_state_comparison")
STRATEGIES = (
    "dqn", "random", "min_latency", "max_quality", "heuristic", "oracle",
)
BASELINE_STRATEGIES = STRATEGIES[1:]
TITLES = {
    "dqn": "DQN", "random": "Random", "min_latency": "Min-Latency",
    "max_quality": "Max-Quality", "heuristic": "Heuristic",
    "oracle": "Oracle (privileged)",
}
COLORS = {
    "no_visual": "#2a78d6",
    "pure_mec": "#eb6834",
    "random": "#9b9a95",
    "min_latency": "#1baf7a",
    "max_quality": "#eda100",
    "heuristic": "#e87ba4",
    "oracle": "#52514e",
}


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def load_run(name, run_dir):
    result = read_json(run_dir / "training_result.json")
    history = read_json(run_dir / "training_history.json")
    final_test = read_json(run_dir / "final_test" / "summary.json")
    if result is None or history is None:
        raise SystemExit(f"Missing training artifacts for {name}: {run_dir}")
    validation = history.get("validation", [])
    best = (
        max(validation, key=lambda row: row["joint_reward"])
        if validation else None
    )
    return {
        "name": name,
        "run_dir": run_dir,
        "result": result,
        "history": history,
        "validation": validation,
        "best": best,
        "final_test": final_test,
    }


def baseline_subset(run_dir):
    isolation = read_json(run_dir / "validation_isolation.json")
    if not isolation or not BASELINE_DEV_CSV.is_file():
        return {}
    uids = {
        item["episode_uid"] for item in isolation["validation_episodes"]
    }
    with BASELINE_DEV_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for strategy in BASELINE_STRATEGIES:
        selected = [
            row for row in rows
            if row["strategy"] == strategy
            and row["episode_uid"] in uids
        ]
        usage = Counter()
        for row in selected:
            for name, count in json.loads(
                row["model_usage"]
            ).items():
                usage[name] += int(count)
        total = max(sum(usage.values()), 1)
        output[strategy] = {
            **summarize_episodes(selected),
            "usage": {
                name: float(usage[name] / total) for name in usage
            },
        }
    return output


def moving_average(values, window=100):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return np.asarray([]), np.asarray([])
    kernel = np.ones(window, dtype=np.float64) / window
    return np.arange(window, len(values) + 1), np.convolve(
        values, kernel, mode="valid"
    )


def style(axis, xlabel, ylabel, title):
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=10.5)
    axis.grid(color="#e4e3df", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def plot_validation_joint(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8.8, 4.3),
                                constrained_layout=True)
    for run in runs:
        episodes = [row["episode"] for row in run["validation"]]
        joint = [row["joint_reward"] for row in run["validation"]]
        axis.plot(
            episodes, joint, marker="o", linewidth=1.8, markersize=4.5,
            color=COLORS[run["name"]],
            label=f"{run['label']} validation JointR",
        )
        if episodes:
            best_index = int(np.argmax(joint))
            axis.scatter(
                [episodes[best_index]], [joint[best_index]],
                color=COLORS[run["name"]], s=60, zorder=5,
            )
            axis.annotate(
                f"{joint[best_index]:.4f} @ Ep{episodes[best_index]}",
                (episodes[best_index], joint[best_index]),
                textcoords="offset points", xytext=(6, 6),
                fontsize=7.5, color=COLORS[run["name"]],
            )
    style(axis, "training episode", "validation JointR",
          "Dev 60 validation JointR (independent Dev episodes)")
    axis.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def plot_validation_metrics(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4),
                                constrained_layout=True)
    panels = (
        ("SR", "success rate", True),
        ("nDTW", "nDTW", False),
        ("E2E_ms", "E2E delay (ms)", False),
        ("joint_reward", "JointR", False),
    )
    for axis, (key, ylabel, percent) in zip(axes.reshape(-1), panels):
        for run in runs:
            episodes = [row["episode"] for row in run["validation"]]
            values = [row[key] for row in run["validation"]]
            axis.plot(
                episodes, values, marker="o", linewidth=1.7,
                markersize=4, color=COLORS[run["name"]],
                label=run["label"],
            )
        if percent:
            axis.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{100*x:.0f}%")
            )
        style(axis, "training episode", ylabel, ylabel)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Dev 60 validation metrics by state design", fontsize=11.5
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def plot_train_ma(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9.0, 4.3),
                                constrained_layout=True)
    for run in runs:
        rows = run["history"].get("train", [])
        returns = [row["episode_return"] for row in rows]
        ma_x, ma_y = moving_average(returns, 100)
        if len(ma_y):
            axis.plot(
                ma_x, ma_y, linewidth=1.9,
                color=COLORS[run["name"]],
                label=f"{run['label']} train return 100-MA",
            )
    axis.axhline(0.0, color="#8a8984", linewidth=0.8, linestyle="--")
    style(axis, "training episode", "episode return",
          "Train episode return: 100-episode moving average")
    axis.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def plot_dev60_bars(best_values, baseline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [best_values[0]["label"], best_values[1]["label"]]
    values = {
        "SR": [best_values[0]["SR"], best_values[1]["SR"]]
        + [baseline[s]["SR"] for s in BASELINE_STRATEGIES],
        "nDTW": [best_values[0]["nDTW"], best_values[1]["nDTW"]]
        + [baseline[s]["nDTW"] for s in BASELINE_STRATEGIES],
        "E2E_ms": [best_values[0]["E2E_ms"], best_values[1]["E2E_ms"]]
        + [baseline[s]["E2E_ms"] for s in BASELINE_STRATEGIES],
        "JointR": [best_values[0]["JointR"], best_values[1]["JointR"]]
        + [baseline[s]["JointR"] for s in BASELINE_STRATEGIES],
    }
    names = labels + [TITLES[s] for s in BASELINE_STRATEGIES]
    colors = (
        [COLORS["no_visual"], COLORS["pure_mec"]]
        + [COLORS[s] for s in BASELINE_STRATEGIES]
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 8.0),
                                constrained_layout=True)
    panels = (
        ("SR", "Success rate", True),
        ("nDTW", "nDTW", False),
        ("E2E_ms", "Mean E2E delay (ms)", False),
        ("JointR", "JointR", False),
    )
    positions = np.arange(len(names))
    for axis, (key, title, percent) in zip(axes.reshape(-1), panels):
        axis.bar(positions, values[key], color=colors, width=0.7)
        axis.set_xticks(positions)
        axis.set_xticklabels(names, rotation=18, fontsize=7.5)
        axis.set_title(title, fontsize=10.5)
        axis.grid(axis="y", color="#e4e3df", linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        if percent:
            axis.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{100*x:.0f}%")
            )
        for bar, value in zip(axis.patches, values[key]):
            label = (
                f"{100*value:.1f}%" if percent else f"{value:.3f}"
            )
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(), label, ha="center", va="bottom",
                fontsize=6.8,
            )
    figure.suptitle(
        "Same 60 Dev validation episodes: best checkpoints vs baselines",
        fontsize=11.5,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def final_test_bars(final_runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summaries = [run for run in final_runs if run["final_test"]]
    if not summaries:
        return False
    base = summaries[0]["final_test"]
    series = []
    for run in summaries:
        series.append(
            (run["label"], COLORS[run["name"]],
             run["final_test"]["overall"]["dqn"])
        )
    for strategy in BASELINE_STRATEGIES:
        series.append(
            (TITLES[strategy], COLORS[strategy],
             base["overall"][strategy])
        )
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.2),
                                constrained_layout=True)
    panels = (
        ("SR", "Success rate", True),
        ("nDTW", "nDTW", False),
        ("E2E_ms", "Mean E2E delay (ms)", False),
        ("JointR", "JointR", False),
    )
    positions = np.arange(len(series))
    for axis, (key, title, percent) in zip(axes.reshape(-1), panels):
        values = [item[2][key] for item in series]
        axis.bar(
            positions, values, width=0.7,
            color=[item[1] for item in series],
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [item[0] for item in series], rotation=18, fontsize=7.5
        )
        axis.set_title(title, fontsize=10.5)
        axis.grid(axis="y", color="#e4e3df", linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        if percent:
            axis.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{100*x:.0f}%")
            )
        for bar, value in zip(axis.patches, values):
            label = (
                f"{100*value:.1f}%" if percent else f"{value:.3f}"
            )
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(), label, ha="center", va="bottom",
                fontsize=6.8,
            )
    figure.suptitle(
        "Association Final Test 180: DQN state designs vs baselines",
        fontsize=11.5,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-visual-run", type=Path,
                        default=NO_VISUAL_RUN)
    parser.add_argument("--pure-mec-run", type=Path,
                        default=PURE_MEC_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    no_visual = load_run("no_visual", args.no_visual_run)
    pure_mec = load_run("pure_mec", args.pure_mec_run)
    no_visual["label"] = "No-Visual (342-D input)"
    pure_mec["label"] = "Pure MEC (322-D input + 20-D mask)"
    runs = [no_visual, pure_mec]
    baseline = baseline_subset(args.no_visual_run)
    if not baseline:
        raise SystemExit("Missing Dev60 baseline subset")

    FIG.mkdir(parents=True, exist_ok=True)
    plot_validation_joint(
        runs, FIG / "state_validation_jointr.png"
    )
    plot_validation_metrics(
        runs, FIG / "state_validation_metrics.png"
    )
    plot_train_ma(runs, FIG / "state_train_return_ma100.png")

    best_values = []
    for run in runs:
        best = run["best"]
        if best is None:
            raise SystemExit(f"No validation rows for {run['label']}")
        best_values.append({
            "label": run["label"],
            "episode": int(best["episode"]),
            "JointR": float(best["joint_reward"]),
            "SR": float(best["SR"]),
            "nDTW": float(best["nDTW"]),
            "E2E_ms": float(best["E2E_ms"]),
        })
    plot_dev60_bars(
        best_values, baseline, FIG / "state_dev60_bars.png"
    )
    has_final = final_test_bars(
        runs, FIG / "state_final_test_bars.png"
    )

    lines = [
        "# Stage-2k 状态设计对照：No-Visual vs Pure MEC",
        "",
        f"日期：{utc_now()[:10]}  ",
        "分支：`experiment/v83-spatial-foundation`  ",
        "仓库：`JLU-MCNS-MEC/mec-v8.2`",
        "",
        "## 0. 结论摘要",
        "",
    ]
    for run in runs:
        result = run["result"]
        best = run["best"]
        lines.append(
            f"- **{run['label']}**：训练 "
            f"{result['episodes_trained']} / {result['max_episodes']} "
            f"Episodes；最佳 Ep{best['episode']}，Dev60 JointR "
            f"**{fmt(best['joint_reward'])}**、SR "
            f"{100*best['SR']:.2f}%、nDTW {fmt(best['nDTW'])}、"
            f"E2E {fmt(best['E2E_ms'],2)} ms；状态 "
            f"`{result['status']}`，converged="
            f"**{result['converged']}**。"
        )
        if run["final_test"] is not None:
            overall = run["final_test"]["overall"]["dqn"]
            lines.append(
                f"  - Final Test 180：SR {100*overall['SR']:.2f}%、"
                f"nDTW {fmt(overall['nDTW'])}、E2E "
                f"{fmt(overall['E2E_ms'],2)} ms、JointR "
                f"**{fmt(overall['JointR'])}**。"
            )
        else:
            lines.append(
                "  - Final Test 180：**未运行**（未满足收敛门槛或未执行）。"
            )
    lines += [
        "",
        "> 本轮对照不是严格的“只改状态”消融：no-visual 使用确定性轮转/"
        "Dev 拓扑/4000 上限；pure MEC 使用 Train 分层随机抽样/3500 上限/"
        "Train 训练拓扑。两者差异同时包含状态编码与训练协议。",
        "",
        "## 1. 状态设计",
        "",
        "| 项 | No-Visual | Pure MEC |",
        "|---|---|---|",
        "| DQN 输入维度 | 342（2 + 20 × 17） | **322（2 + 20 × 16）** |",
        "| 可用性表示 | 每候选 1 维 availability | **独立 20 维动作 Mask** |",
        "| 决策状态合计 | 342 | **342（322 + 20）** |",
        "| 视觉/Goal/历史/GAB/热图 | 排除 | 排除 |",
        "| 导航动作/步数/目标/成功标签 | 排除 | 排除 |",
        "| 网络 | MLP 342→256→256→20 | MLP **322→256→256→20** |",
        "",
        "两种设计的候选字段高度重叠；pure MEC 只是把 availability 从每候选"
        "特征中移出，改为独立的动作 Mask，并采用随机训练任务采样。",
        "",
        "## 2. 训练与验证结果",
        "",
        "| 指标 | No-Visual | Pure MEC |",
        "|---|---:|---:|",
        f"| 实际训练 Episodes | {no_visual['result']['episodes_trained']} | "
        f"{pure_mec['result']['episodes_trained']} |",
        f"| 最大 Episodes | {no_visual['result']['max_episodes']} | "
        f"{pure_mec['result']['max_episodes']} |",
        f"| 最佳 Episode | {best_values[0]['episode']} | "
        f"{best_values[1]['episode']} |",
        f"| 最佳 Dev60 JointR | **{fmt(best_values[0]['JointR'])}** | "
        f"**{fmt(best_values[1]['JointR'])}** |",
        f"| 最佳 Dev60 SR | {100*best_values[0]['SR']:.2f}% | "
        f"{100*best_values[1]['SR']:.2f}% |",
        f"| 最佳 Dev60 nDTW | {fmt(best_values[0]['nDTW'])} | "
        f"{fmt(best_values[1]['nDTW'])} |",
        f"| 最佳 Dev60 E2E | {fmt(best_values[0]['E2E_ms'],2)} ms | "
        f"{fmt(best_values[1]['E2E_ms'],2)} ms |",
        f"| 训练状态 | {no_visual['result']['status']} | "
        f"{pure_mec['result']['status']} |",
        f"| converged | {no_visual['result']['converged']} | "
        f"{pure_mec['result']['converged']} |",
        "",
        f"![validation JointR](figures/{FIG.name}/state_validation_jointr.png)",
        "",
        f"![validation metrics](figures/{FIG.name}/state_validation_metrics.png)",
        "",
        f"![train MA100](figures/{FIG.name}/state_train_return_ma100.png)",
        "",
        "## 3. 同一组 Dev60 上的 Baseline 对照",
        "",
        "| 策略 | JointR | SR | nDTW | E2E (ms) |",
        "|---|---:|---:|---:|---:|",
        f"| No-Visual best | {fmt(best_values[0]['JointR'])} | "
        f"{100*best_values[0]['SR']:.2f}% | "
        f"{fmt(best_values[0]['nDTW'])} | "
        f"{fmt(best_values[0]['E2E_ms'],2)} |",
        f"| Pure MEC best | {fmt(best_values[1]['JointR'])} | "
        f"{100*best_values[1]['SR']:.2f}% | "
        f"{fmt(best_values[1]['nDTW'])} | "
        f"{fmt(best_values[1]['E2E_ms'],2)} |",
    ]
    for strategy in BASELINE_STRATEGIES:
        item = baseline[strategy]
        lines.append(
            f"| {TITLES[strategy]} | {fmt(item['JointR'])} | "
            f"{100*item['SR']:.2f}% | {fmt(item['nDTW'])} | "
            f"{fmt(item['E2E_ms'],2)} |"
        )
    lines += [
        "",
        f"![Dev60 bars](figures/{FIG.name}/state_dev60_bars.png)",
        "",
    ]
    if has_final:
        lines += [
            "## 4. Association Final Test 180",
            "",
            "| 策略 | SR | nDTW | E2E (ms) | JointR |",
            "|---|---:|---:|---:|---:|",
        ]
        for run in runs:
            if run["final_test"] is None:
                lines.append(
                    f"| {run['label']} | — | — | — | 未运行 |"
                )
                continue
            item = run["final_test"]["overall"]["dqn"]
            lines.append(
                f"| {run['label']} | {100*item['SR']:.2f}% | "
                f"{fmt(item['nDTW'])} | {fmt(item['E2E_ms'],2)} | "
                f"{fmt(item['JointR'])} |"
            )
        base_summary = next(
            run["final_test"] for run in runs if run["final_test"]
        )
        for strategy in BASELINE_STRATEGIES:
            item = base_summary["overall"][strategy]
            lines.append(
                f"| {TITLES[strategy]} | {100*item['SR']:.2f}% | "
                f"{fmt(item['nDTW'])} | {fmt(item['E2E_ms'],2)} | "
                f"{fmt(item['JointR'])} |"
            )
        lines += [
            "",
            f"![final test bars](figures/{FIG.name}/state_final_test_bars.png)",
            "",
        ]
    else:
        lines += [
            "## 4. Association Final Test 180",
            "",
            "- 本轮没有任何状态设计进入正式 Final Test；不消耗一次性测试集。",
            "",
        ]
    lines += [
        "## 5. 解释与限制",
        "",
        "- **状态维度**：no-visual 为 342 维输入；pure MEC 为 322 维输入 + "
        "20 维动作 Mask，决策状态合计仍为 342。两者不是“去掉 20 维信息”"
        "而是把 availability 从输入特征改为动作 Mask。",
        "- **训练协议**：no-visual 使用确定性 3×3 轮转、Dev 拓扑、4000 上限；"
        "pure MEC 使用 Train 分层随机抽样 3500、Train 训练拓扑、Dev 验证。"
        "因此两者的差异不能单独归因于状态编码。",
        "- **收敛**：只有 `converged=true` 的设计才允许进入 Final Test；"
        "未收敛时保留测试集。",
        "- 所有旧 run 的权重、日志、TensorBoard 和 Final Test 产物均未覆盖。",
        "",
        "## 6. 产物",
        "",
        f"- No-Visual run：`{args.no_visual_run}`",
        f"- Pure MEC run：`{args.pure_mec_run}`",
        f"- 对照图：`{FIG}/`",
        "- 大型权重、Replay Buffer、TensorBoard Event 和逐 Episode CSV "
        "保留在服务器 `runs/` 下，不提交版本库。",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
