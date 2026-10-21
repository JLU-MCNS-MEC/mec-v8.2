"""Stage-2l Chinese report and convergence figures.

Reads the six training histories and any completed Dev evaluation, then
writes ``REPORT.md`` plus light-mode PNG figures under ``figures/``.

The plots use a two-series categorical palette validated with the dataviz
skill's checker (blue ``#2a78d6`` = DQN, orange ``#eb6834`` = PPO on the
``#fcfcfb`` light surface; all checks pass).  There is one y-axis per panel
and no dual axis; every panel with two series carries a legend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import read_json, utc_now
from v83_spatial_stage2.stage2l_fixed_city import (
    CONFIG_DESCRIPTIONS,
    EVAL_DIR,
    FIGURE_DIR,
    STAGE2L_ROOT,
)
from v83_spatial_stage2.stage2l_protocol import classify_status


CONFIGS = ("A", "B", "C")
ALGOS = ("dqn", "ppo")
COLORS = {"dqn": "#2a78d6", "ppo": "#eb6834"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedcd5"
METRICS = (
    ("joint_reward", "Validation JointR"),
    ("SR", "Validation SR"),
    ("nDTW", "Validation nDTW"),
    ("E2E_ms", "Validation E2E (ms)"),
)


def _history(config, algorithm):
    payload = read_json(
        STAGE2L_ROOT / config / algorithm / "training_history.json"
    ) or {}
    return payload


def _result(config, algorithm):
    return read_json(
        STAGE2L_ROOT / config / algorithm / "training_result.json"
    )


def _latest_status(config):
    rows = {}
    for algorithm in ALGOS:
        result = _result(config, algorithm)
        history = _history(config, algorithm)
        validation = history.get("validation", [])
        joints = [float(row["joint_reward"]) for row in validation]
        if result:
            status = result.get("status", {})
        else:
            status = classify_status(joints, False)
        rows[algorithm] = {
            "result": result,
            "validation": validation,
            "joints": joints,
            "status": status,
            "train_rows": len(history.get("train", [])),
            "best": float(max(joints)) if joints else None,
            "last": float(joints[-1]) if joints else None,
        }
    return rows


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)


def plot_convergence(output_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(CONFIGS), len(METRICS),
        figsize=(13.2, 8.2), sharex="col",
        facecolor=SURFACE,
    )
    any_data = False
    for row_index, config in enumerate(CONFIGS):
        for column, (key, label) in enumerate(METRICS):
            ax = axes[row_index][column]
            _style_axes(ax)
            for algorithm in ALGOS:
                history = _history(config, algorithm)
                rows = history.get("validation", [])
                if not rows:
                    continue
                x = [int(item["episode"]) for item in rows]
                y = [float(item[key]) * (100.0 if key == "SR" else 1.0)
                     for item in rows]
                ax.plot(
                    x, y, color=COLORS[algorithm], linewidth=1.8,
                    marker="o", markersize=3.6,
                    label=algorithm.upper(),
                )
                any_data = True
                if len(x) >= 1:
                    ax.annotate(
                        algorithm.upper(), (x[-1], y[-1]),
                        xytext=(4, 2), textcoords="offset points",
                        fontsize=6.5, color=COLORS[algorithm],
                    )
            if row_index == 0:
                ax.set_title(label, fontsize=9.5, pad=6)
            if column == 0:
                ax.set_ylabel(config, fontsize=10, fontweight="bold")
            if row_index == len(CONFIGS) - 1:
                ax.set_xlabel("Episode", fontsize=8)
            ax.axhline(0.0, color=GRID, linewidth=0.8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles, labels, loc="upper center", ncol=2,
            frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995),
        )
    figure.suptitle(
        "Stage-2l fixed-city validation (DQN vs PPO)",
        fontsize=12, color=TEXT, y=0.975,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "stage2l_convergence.png"
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path if any_data else None


def plot_optimization(output_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(CONFIGS), 2, figsize=(9.6, 7.4),
        facecolor=SURFACE, sharex=True,
    )
    any_data = False
    for index, config in enumerate(CONFIGS):
        dqn = _history(config, "dqn").get("train", [])
        ppo = _history(config, "ppo").get("train", [])
        ax_dqn, ax_ppo = axes[index][0], axes[index][1]
        for ax in (ax_dqn, ax_ppo):
            _style_axes(ax)
        if dqn:
            x = [int(item["episode"]) for item in dqn]
            ax_dqn.plot(
                x, [float(item["q_mean"]) for item in dqn],
                color=COLORS["dqn"], linewidth=1.6, label="DQN Q mean",
            )
            any_data = True
        if ppo:
            x = [int(item["episode"]) for item in ppo]
            ax_ppo.plot(
                x, [float(item["policy_entropy"]) for item in ppo],
                color=COLORS["ppo"], linewidth=1.6,
                label="PPO policy entropy",
            )
            any_data = True
        if index == 0:
            ax_dqn.set_title("DQN Q mean", fontsize=9.5)
            ax_ppo.set_title("PPO policy entropy", fontsize=9.5)
        ax_dqn.set_ylabel(config, fontsize=10, fontweight="bold")
        if index == len(CONFIGS) - 1:
            ax_dqn.set_xlabel("Episode", fontsize=8)
            ax_ppo.set_xlabel("Episode", fontsize=8)
    figure.suptitle(
        "Stage-2l optimization diagnostics", fontsize=12, color=TEXT, y=0.98
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "stage2l_optimization.png"
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path if any_data else None


def plot_baselines(output_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_path = EVAL_DIR / "A" / "dev" / "summary.json"
    payload = read_json(summary_path)
    if not payload:
        return None
    labels = []
    values = []
    for algorithm, item in payload.get("overall", {}).items():
        labels.append(algorithm)
        values.append(float(item["JointR"]))
    if not values:
        return None
    order = np.argsort(values)
    labels = [labels[index] for index in order]
    values = [values[index] for index in order]
    figure, ax = plt.subplots(figsize=(8.4, 4.2), facecolor=SURFACE)
    _style_axes(ax)
    bars = ax.barh(labels, values, color="#2a78d6", height=0.58)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.4f}", (bar.get_width(), bar.get_y()
                             + bar.get_height() / 2),
            xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=8, color=MUTED,
        )
    ax.set_xlabel("Dev JointR", fontsize=9)
    ax.set_title("Stage-2l config A · Dev baseline comparison",
                 fontsize=10.5)
    ax.axvline(0.0, color=GRID, linewidth=0.8)
    figure.tight_layout()
    path = output_dir / "stage2l_baselines_A_dev.png"
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


def _convergence_text(config, info):
    lines = []
    for algorithm in ALGOS:
        item = info[algorithm]
        status = item["status"]
        lines.append(
            f"- {algorithm.upper()}：训练 {item['train_rows']} Episodes，"
            f"验证 {len(item['validation'])} 次；"
            f"best JointR={item['best'] if item['best'] is not None else '-'}，"
            f"last={item['last'] if item['last'] is not None else '-'}；"
            f"状态 `{status.get('status')}`，"
            f"稳定平台={status.get('stable_platform')}，"
            f"converged={status.get('converged')}。"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures = {}
    if not args.skip_plots:
        figures["convergence"] = plot_convergence(FIGURE_DIR)
        figures["optimization"] = plot_optimization(FIGURE_DIR)
        figures["baselines"] = plot_baselines(FIGURE_DIR)

    lines = [
        "# Stage-2l 固定城市实验报告",
        "",
        f"生成时间：{utc_now()}",
        "",
        "## 0. 结论摘要",
        "",
        "- 本报告只基于**固定城市、固定任务池、固定独立 Dev 验证**的结果；"
        "Final Test 未读取。",
        "- 收敛判断不看早停触发，而看固定 Dev Validation JointR 是否形成"
        "稳定平台，并结合验证 SR/nDTW/E2E 与 Baseline 对照。",
        "- 训练是否完成、是否收敛，见下方每个配置的状态；"
        "训练未完成时不下收敛结论。",
        "",
        "环境固定性、任务分布、T5 兼容性与启动前 Smoke Test 见 "
        "`SETUP_REPORT.md` 与 `smoke/smoke_report.md`。",
        "",
        "## 1. 三种部署配置",
        "",
        "| 配置 | 说明 |",
        "|---|---|",
    ]
    for config, description in CONFIG_DESCRIPTIONS.items():
        lines.append(f"| {config} | {description} |")
    lines += [
        "",
        "所有配置共享同一张固定城市地图、同一 Train 任务顺序、"
        "同一 60 条独立 Dev 任务、同一奖励与同一五种冻结 MEC。",
        "",
        "## 2. 收敛与验证",
        "",
    ]
    for config in args.configs:
        info = _latest_status(config)
        lines += [f"### 配置 {config}", ""]
        lines += _convergence_text(config, info)
        lines += [
            "",
            "| 算法 | Episode | JointR | SR | nDTW | E2E (ms) | SPL |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for algorithm in ALGOS:
            last = info[algorithm]["validation"]
            if not last:
                lines.append(f"| {algorithm.upper()} | - | - | - | - | - | - |")
                continue
            row = last[-1]
            lines.append(
                f"| {algorithm.upper()} | {row['episode']} | "
                f"{row['joint_reward']:.4f} | {100*row['SR']:.2f}% | "
                f"{row['nDTW']:.4f} | {row['E2E_ms']:.1f} | "
                f"{row['SPL']:.4f} |"
            )
        lines.append("")
    if figures.get("convergence"):
        lines += [
            "## 3. 收敛曲线",
            "",
            f"![Validation curves]({Path(figures['convergence']).relative_to(STAGE2L_ROOT)})",
            "",
        ]
    if figures.get("optimization"):
        lines += [
            f"![Optimization diagnostics]({Path(figures['optimization']).relative_to(STAGE2L_ROOT)})",
            "",
        ]
    lines += [
        "## 4. Baseline 对照",
        "",
    ]
    eval_summary = read_json(EVAL_DIR / "A" / "dev" / "summary.json")
    if eval_summary:
        lines += [
            "配置 A / Dev（同一固定城市、同一固定拓扑、同一 60 条独立任务）：",
            "",
            "| 算法/策略 | SR | SPL | nDTW | E2E (ms) | JointR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for algorithm, item in eval_summary["overall"].items():
            lines.append(
                f"| {algorithm} | {100*item['SR']:.2f}% | "
                f"{item['SPL']:.4f} | {item['nDTW']:.4f} | "
                f"{item['E2E_ms']:.2f} | {item['JointR']:.4f} |"
            )
        if figures.get("baselines"):
            lines += [
                "",
                f"![Baseline comparison]({Path(figures['baselines']).relative_to(STAGE2L_ROOT)})",
                "",
            ]
    else:
        lines += [
            "尚未生成 `evaluation/A/dev/summary.json`；"
            "收敛后运行 `stage2l_eval.py --config A --split dev`。",
            "",
        ]
    lines += [
        "## 5. 说明与限制",
        "",
        "- DQN/PPO 使用同一任务清单、同一任务顺序、同一奖励、同一 MEC、"
        "同一验证任务；只有算法不同。",
        "- 所有训练最多 3500 Episodes，每 200 Episodes 验证一次；"
        "没有早停，也没有根据验证曲线修改奖励或算法。",
        "- 固定环境内 baseline 成立后，才进入新拓扑/新城市泛化测试；"
        "正式 Test 结果不得用于调参。",
        "- 权重、TensorBoard、逐 Episode 日志保存在服务器 "
        f"`{STAGE2L_ROOT}`；本报告只提交必要的源码、测试与中文 Markdown。",
        "",
    ]
    path = STAGE2L_ROOT / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {path}", flush=True)
    print("STAGE2L_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
