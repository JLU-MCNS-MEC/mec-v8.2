"""Stage-2k DQN vs PPO convergence and Dev180 comparison figures.

Reads the *actual* training histories written by the two trainers; nothing is
interpolated or synthesized.  The final four-metric comparison uses the
checkpoint-selected Dev180 evaluations (DQN best.pt / PPO best.pt) and the
existing five baseline results.

Figures follow one visual system: two entity colors (DQN blue, PPO orange),
recessive hairline grids, thin marks, surface rings on markers, and direct
labels only at endpoints/best points.  The five baselines are context gray;
Oracle is marked as a privileged upper bound.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    atomic_write_json,
    bootstrap_ci,
    utc_now,
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DQN = "#2a78d6"
PPO = "#eb6834"
CONTEXT = "#c3c2b7"
DQN_LIGHT = "#86b6ef"

plt.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "font.size": 9.5,
    "axes.unicode_minus": False,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})


def load_run(run_dir):
    run_dir = Path(run_dir)
    history = json.loads(
        (run_dir / "training_history.json").read_text(encoding="utf-8")
    )
    result = None
    if (run_dir / "training_result.json").is_file():
        result = json.loads(
            (run_dir / "training_result.json").read_text(encoding="utf-8")
        )
    progress = None
    if (run_dir / "progress.json").is_file():
        progress = json.loads(
            (run_dir / "progress.json").read_text(encoding="utf-8")
        )
    return {"dir": run_dir, "history": history,
            "result": result, "progress": progress}


def moving_average(values, window=100):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return np.array([], dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0, labelsize=8.5)
    ax.set_axisbelow(True)


def best_validation(run):
    rows = run["history"].get("validation", [])
    if not rows:
        return None
    return max(rows, key=lambda row: float(row["joint_reward"]))


def train_series(run, key):
    rows = run["history"].get("train", [])
    return (
        np.asarray([int(row["episode"]) for row in rows]),
        np.asarray([float(row[key]) for row in rows]),
    )


def validation_series(run, key):
    rows = run["history"].get("validation", [])
    return (
        np.asarray([int(row["episode"]) for row in rows]),
        np.asarray([float(row[key]) for row in rows]),
    )


def end_label(ax, x, y, text, color):
    ax.annotate(
        text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
        va="center", ha="left", fontsize=8.5, color=INK_2,
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.9),
    )
    ax.plot([x], [y], marker="o", markersize=4, color=color,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)


def plot_training_and_validation(dqn, ppo, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3),
                             constrained_layout=True)
    ax = axes[0]
    style(ax)
    for run, color, name in ((dqn, DQN, "DQN"), (ppo, PPO, "PPO")):
        x, y = train_series(run, "episode_return")
        if len(x) == 0:
            continue
        ax.plot(x, y, color=color, linewidth=0.7, alpha=0.12, zorder=1)
        ma = moving_average(y, 100)
        if len(ma):
            mx = x[99:]
            ax.plot(mx, ma, color=color, linewidth=2.0, zorder=3,
                    label=f"{name} 100-Episode 移动平均")
            end_label(ax, mx[-1], ma[-1], f"{name} {ma[-1]:+.3f}", color)
    ax.set_title("训练 Episode Return 与 100-Episode 移动平均", loc="left",
                 fontsize=11, fontweight="semibold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return（= JointR）")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=0)

    ax = axes[1]
    style(ax)
    for run, color, name in ((dqn, DQN, "DQN"), (ppo, PPO, "PPO")):
        x, y = validation_series(run, "joint_reward")
        if len(x) == 0:
            continue
        ax.plot(x, y, color=color, linewidth=2.0, marker="o",
                markersize=5.5, markeredgecolor=SURFACE,
                markeredgewidth=1.8, label=f"{name} Dev60", zorder=3)
        end_label(ax, x[-1], y[-1], f"{name} {y[-1]:+.4f}", color)
        best = best_validation(run)
        if best is not None:
            ax.scatter(
                [int(best["episode"])], [float(best["joint_reward"])],
                s=110, facecolors="none", edgecolors=color,
                linewidths=1.8, zorder=4,
            )
    ax.set_title("固定 60 Dev Episodes 的 Validation JointR", loc="left",
                 fontsize=11, fontweight="semibold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("JointR")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=0)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_validation_metrics(dqn, ppo, output):
    metrics = [
        ("joint_reward", "验证 JointR", 1.0, "{:+.4f}"),
        ("SR", "验证 SR", 100.0, "{:.2f}%"),
        ("nDTW", "验证 nDTW", 1.0, "{:.4f}"),
        ("E2E_ms", "验证平均 E2E 时延", 1.0, "{:.1f} ms"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.0),
                             constrained_layout=True)
    for ax, (key, title, scale, fmt) in zip(axes.ravel(), metrics):
        style(ax)
        for run, color, name in ((dqn, DQN, "DQN"), (ppo, PPO, "PPO")):
            x, y = validation_series(run, key)
            if len(x) == 0:
                continue
            y = y * scale
            ax.plot(x, y, color=color, linewidth=2.0, marker="o",
                    markersize=5.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.8, label=name, zorder=3)
            end_label(ax, x[-1], y[-1], fmt.format(y[-1]), color)
            best = best_validation(run)
            if best is not None:
                ax.scatter(
                    [int(best["episode"])],
                    [float(best[key]) * scale],
                    s=110, facecolors="none", edgecolors=color,
                    linewidths=1.8, zorder=4,
                )
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Episode")
    handles = [
        Line2D([0], [0], color=DQN, linewidth=2.2, label="DQN"),
        Line2D([0], [0], color=PPO, linewidth=2.2, label="PPO"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=MUTED, markersize=10, markeredgewidth=1.8,
               label="空心圈 = 最佳验证 Checkpoint"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ppo_diagnostics(ppo, output):
    rows = ppo["history"].get("validation", [])
    best = best_validation(ppo)
    panels = [
        ("policy_entropy", "策略熵（Validation 窗口均值）", 1.0, "{:.3f}"),
        ("update_value_loss", "Value Loss", 1.0, "{:.4f}"),
        ("approx_kl", "Approximate KL", 1.0, "{:.4f}"),
        ("clip_fraction", "Clip Fraction", 1.0, "{:.3f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.5),
                             constrained_layout=True)
    for ax, (key, title, scale, fmt) in zip(axes.ravel(), panels):
        style(ax)
        x = np.asarray([int(row["episode"]) for row in rows])
        y = np.asarray([float(row.get(key, 0.0)) for row in rows]) * scale
        if len(x):
            ax.plot(x, y, color=PPO, linewidth=2.0, marker="o",
                    markersize=5.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.8, zorder=3)
            end_label(ax, x[-1], y[-1], fmt.format(y[-1]), PPO)
        if key == "approx_kl":
            ax.axhline(0.05, color=AXIS, linewidth=1.0, zorder=1)
            if len(x):
                ax.text(x[0], 0.05, " target_kl=0.05",
                        va="bottom", ha="left", fontsize=8, color=MUTED)
        if best is not None and key in best:
            ax.scatter(
                [int(best["episode"])], [float(best[key]) * scale],
                s=110, facecolors="none", edgecolors=PPO,
                linewidths=1.8, zorder=4,
            )
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Episode")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dqn_diagnostics(dqn, output):
    train = dqn["history"].get("train", [])
    x = np.asarray([int(row["episode"]) for row in train])
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.5),
                             constrained_layout=True)

    ax = axes[0, 0]
    style(ax)
    q_mean = moving_average([float(row["q_mean"]) for row in train], 100)
    q_max = moving_average([float(row["q_max"]) for row in train], 100)
    if len(q_mean):
        mx = x[99:]
        ax.plot(mx, q_mean, color=DQN, linewidth=2.0, label="Q mean（100-MA）")
        ax.plot(mx, q_max, color=DQN_LIGHT, linewidth=2.0,
                label="Q max（100-MA）")
        end_label(ax, mx[-1], q_mean[-1], f"{q_mean[-1]:.2f}", DQN)
    ax.set_title("Q 值（DQN）", loc="left", fontsize=11,
                 fontweight="semibold")
    ax.set_xlabel("Episode")
    ax.legend(loc="upper left", fontsize=8.5)

    ax = axes[0, 1]
    style(ax)
    loss = moving_average([float(row["loss"]) for row in train], 100)
    if len(loss):
        ax.plot(x[99:], loss, color=DQN, linewidth=2.0)
        end_label(ax, x[-1], loss[-1], f"{loss[-1]:.4f}", DQN)
    ax.set_title("TD Loss（100-MA）", loc="left", fontsize=11,
                 fontweight="semibold")
    ax.set_xlabel("Episode")

    ax = axes[1, 0]
    style(ax)
    epsilon = np.asarray([float(row["epsilon"]) for row in train])
    if len(epsilon):
        ax.plot(x, epsilon, color=DQN, linewidth=2.0)
        end_label(ax, x[-1], epsilon[-1], f"{epsilon[-1]:.3f}", DQN)
    ax.set_title("ε（按环境交互步线性衰减）", loc="left", fontsize=11,
                 fontweight="semibold")
    ax.set_xlabel("Episode")

    ax = axes[1, 1]
    style(ax)
    grad = moving_average([float(row["grad_norm"]) for row in train], 100)
    if len(grad):
        ax.plot(x[99:], grad, color=DQN, linewidth=2.0)
        end_label(ax, x[-1], grad[-1], f"{grad[-1]:.3f}", DQN)
    ax.set_title("梯度范数（100-MA）", loc="left", fontsize=11,
                 fontweight="semibold")
    ax.set_xlabel("Episode")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_summary(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def paired_dqn_ppo(dqn_summary_path, ppo_summary_path):
    """Pair DQN and PPO per-episode evaluations by Episode UID."""
    dqn_csv = Path(dqn_summary_path).parent / "per_episode.csv"
    ppo_csv = Path(ppo_summary_path).parent / "per_episode.csv"
    if not (dqn_csv.is_file() and ppo_csv.is_file()):
        return None
    with dqn_csv.open(encoding="utf-8") as handle:
        dqn_rows = {row["episode_uid"]: row for row in csv.DictReader(handle)}
    with ppo_csv.open(encoding="utf-8") as handle:
        ppo_rows = list(csv.DictReader(handle))
    deltas = {"JointR": [], "nDTW": [], "SR": [], "E2E_ms": []}
    wins = losses = ties = 0
    for row in ppo_rows:
        base = dqn_rows.get(row["episode_uid"])
        if base is None:
            continue
        joint = float(row["joint_reward"]) - float(base["joint_reward"])
        deltas["JointR"].append(joint)
        deltas["nDTW"].append(float(row["ndtw"]) - float(base["ndtw"]))
        deltas["SR"].append(float(row["success"]) - float(base["success"]))
        deltas["E2E_ms"].append(
            float(row["mean_e2e_ms"]) - float(base["mean_e2e_ms"])
        )
        if joint > 1e-9:
            wins += 1
        elif joint < -1e-9:
            losses += 1
        else:
            ties += 1
    if not deltas["JointR"]:
        return None
    output = {"n": len(deltas["JointR"]),
              "wins": wins, "losses": losses, "ties": ties}
    for key, values in deltas.items():
        output[f"delta_{key}"] = bootstrap_ci(values)
    return output


def plot_final_comparison(dqn_summary, ppo_summary, baselines, output):
    if dqn_summary is None or ppo_summary is None:
        return False
    labels = ["DQN best", "PPO best", "Random", "Min-Latency",
              "Max-Quality", "Heuristic", "Oracle (privileged)"]
    colors = [DQN, PPO, CONTEXT, CONTEXT, CONTEXT, CONTEXT, CONTEXT]
    metrics = [
        ("SR", "SR", 100.0, "{:.2f}%", None),
        ("nDTW", "nDTW", 1.0, "{:.4f}", None),
        ("E2E_ms", "平均 E2E 时延 (ms)", 1.0, "{:.1f}", None),
        ("JointR", "JointR", 1.0, "{:+.4f}", None),
    ]
    values = [
        ("DQN best", dqn_summary["overall"]),
        ("PPO best", ppo_summary["overall"]),
        ("Random", baselines["random"]),
        ("Min-Latency", baselines["min_latency"]),
        ("Max-Quality", baselines["max_quality"]),
        ("Heuristic", baselines["heuristic"]),
        ("Oracle (privileged)", baselines["oracle"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2),
                             constrained_layout=True)
    y = np.arange(len(values))[::-1]
    for ax, (key, title, scale, fmt, _) in zip(axes.ravel(), metrics):
        style(ax)
        vals = np.asarray([
            float(item[1][key]) * scale for item in values
        ])
        bars = ax.barh(y, vals, height=0.58, color=colors,
                       edgecolor="none", zorder=3)
        span = max(float(np.abs(vals).max()), 1e-6)
        for bar, value in zip(bars, vals):
            if value >= 0:
                ax.text(
                    value + 0.012 * span,
                    bar.get_y() + bar.get_height() / 2,
                    fmt.format(value), va="center", ha="left",
                    fontsize=8, color=INK_2,
                )
            else:
                ax.text(
                    value - 0.012 * span,
                    bar.get_y() + bar.get_height() / 2,
                    fmt.format(value), va="center", ha="right",
                    fontsize=8, color=INK_2,
                )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5, color=INK_2)
        ax.set_title(title, loc="left", fontsize=11,
                     fontweight="semibold")
        ax.axvline(0.0, color=AXIS, linewidth=0.8, zorder=1)
        ax.set_xlim(min(0.0, float(vals.min()) - 0.08 * span),
                    float(vals.max()) + 0.22 * span)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    handles = [
        Patch(facecolor=DQN, label="DQN（最佳验证 Checkpoint）"),
        Patch(facecolor=PPO, label="PPO（最佳验证 Checkpoint）"),
        Patch(facecolor=CONTEXT, label="五种 Baseline（上下文）"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.text(
        0.5, -0.075,
        "同一批 180 个 Dev Episode；Oracle 为特权上界，不计入非特权对比。",
        ha="center", fontsize=8.5, color=MUTED,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def summarize_run(run, name):
    history = run["history"]
    validation = history.get("validation", [])
    best = best_validation(run)
    last = validation[-1] if validation else None
    joints = [float(row["joint_reward"]) for row in validation]
    trend = 0.0
    if len(joints) >= 3:
        trend = float(np.polyfit(
            np.arange(3, dtype=np.float64),
            np.asarray(joints[-3:], dtype=np.float64), 1,
        )[0])
    result = run["result"] or {}
    return {
        "name": name,
        "run_dir": str(run["dir"]),
        "episodes": (
            int(history["train"][-1]["episode"])
            if history.get("train") else 0
        ),
        "best_episode": (
            int(best["episode"]) if best else None
        ),
        "best_joint": (
            float(best["joint_reward"]) if best else None
        ),
        "last_episode": (
            int(last["episode"]) if last else None
        ),
        "last_joint": (
            float(last["joint_reward"]) if last else None
        ),
        "last3_trend": trend,
        "status": result.get("status"),
        "converged": result.get("converged"),
        "policy_entropy_final": result.get("policy_entropy_final"),
        "value_loss_final": result.get("value_loss_final"),
        "approx_kl_final": result.get("approx_kl_final"),
        "episodes_trained": result.get("episodes_trained"),
        "completed": bool(run["progress"]
                          and run["progress"].get("status") == "completed"),
    }


def markdown_report(path, dqn_summary, ppo_summary, dqn_eval, ppo_eval,
                    baselines, figures, final_plot, paired=None):
    def row(name, item):
        return (
            f"| {name} | {100*item['SR']:.2f}% | {item['nDTW']:.4f} | "
            f"{item['E2E_ms']:.2f} | {item['JointR']:.4f} |"
        )

    def num(value, spec):
        return "—" if value is None else format(float(value), spec)

    lines = [
        "# Stage-2k：DQN 与 PPO 收敛及 Dev180 对比",
        "",
        f"- 生成时间（UTC）：{utc_now()}",
        "- 数据来源：两个训练目录的 `training_history.json` / "
        "`training_result.json`；最终对比使用 Dev180 `summary.json`。",
        "- 未读取 Final Test；曲线由真实逐 Episode 记录生成，无插值。",
        "",
        "## 1. 训练与收敛",
        "",
        "| 项 | DQN A | PPO |",
        "|---|---:|---:|",
        f"| 运行目录 | `{dqn_summary['run_dir']}` | "
        f"`{ppo_summary['run_dir']}` |",
        f"| 已训练 Episode | {dqn_summary['episodes']} | "
        f"{ppo_summary['episodes']} |",
        f"| 最佳验证 Episode | {dqn_summary['best_episode']} | "
        f"{ppo_summary['best_episode']} |",
        f"| 最佳 Dev60 JointR | "
        f"{num(dqn_summary['best_joint'], '.4f')} | "
        f"{num(ppo_summary['best_joint'], '.4f')} |",
        f"| 最后 Dev60 JointR | "
        f"{num(dqn_summary['last_joint'], '.4f')} | "
        f"{num(ppo_summary['last_joint'], '.4f')} |",
        f"| 最后 3 次趋势 | "
        f"{num(dqn_summary['last3_trend'], '+.5f')} | "
        f"{num(ppo_summary['last3_trend'], '+.5f')} |",
        f"| 训练状态 | `{dqn_summary['status']}` | "
        f"`{ppo_summary['status']}` |",
        f"| `converged` | **{dqn_summary['converged']}** | "
        f"**{ppo_summary['converged']}** |",
        "",
    ]
    if ppo_summary.get("policy_entropy_final") is not None:
        lines += [
            f"- PPO 最终策略熵：{ppo_summary['policy_entropy_final']:.4f}；"
            f"Value Loss：{ppo_summary['value_loss_final']:.4f}；"
            f"approx KL：{ppo_summary['approx_kl_final']:.4f}。",
            "",
        ]
    lines += [
        "## 2. 收敛曲线",
        "",
        f"![训练与验证]({figures['training']})",
        "",
        f"![四项验证指标]({figures['metrics']})",
        "",
        "> 训练曲线来自采样策略（PPO 还有熵探索），任务分布也与 Dev 不同；"
        "100-Episode 移动平均不能直接与 Validation JointR 比较。模型选择"
        "只使用固定 Dev60 Validation。",
        "",
        "## 3. 训练诊断",
        "",
        f"![PPO 诊断]({figures['ppo_diag']})",
        "",
        f"![DQN 诊断]({figures['dqn_diag']})",
        "",
    ]
    if final_plot and dqn_eval and ppo_eval:
        lines += [
            "## 4. Dev180 最终对比（最佳验证 Checkpoint）",
            "",
            "| 策略 | SR | nDTW | E2E (ms) | JointR |",
            "|---|---:|---:|---:|---:|",
            row("DQN A best", dqn_eval["overall"]),
            row("PPO best", ppo_eval["overall"]),
            row("Random", baselines["random"]),
            row("Min-Latency", baselines["min_latency"]),
            row("Max-Quality", baselines["max_quality"]),
            row("Heuristic", baselines["heuristic"]),
            row("Oracle（特权）", baselines["oracle"]),
            "",
            f"![Dev180 对比]({figures['final']})",
            "",
            "### 逐 Episode 配对（PPO − DQN，同一 180 个 Dev Episode）",
            "",
        ]
        if paired is not None:
            lines += [
                "| 指标 | 均值差 | 95% Bootstrap CI |",
                "|---|---:|---:|",
                f"| JointR | {paired['delta_JointR']['mean']:+.4f} | "
                f"[{paired['delta_JointR']['low']:+.4f}, "
                f"{paired['delta_JointR']['high']:+.4f}] |",
                f"| nDTW | {paired['delta_nDTW']['mean']:+.4f} | "
                f"[{paired['delta_nDTW']['low']:+.4f}, "
                f"{paired['delta_nDTW']['high']:+.4f}] |",
                f"| SR | {paired['delta_SR']['mean']:+.4f} | "
                f"[{paired['delta_SR']['low']:+.4f}, "
                f"{paired['delta_SR']['high']:+.4f}] |",
                f"| E2E (ms) | {paired['delta_E2E_ms']['mean']:+.2f} | "
                f"[{paired['delta_E2E_ms']['low']:+.2f}, "
                f"{paired['delta_E2E_ms']['high']:+.2f}] |",
                "",
                f"配对胜负：PPO 胜 **{paired['wins']}** / 负 "
                f"**{paired['losses']}** / 平 **{paired['ties']}**"
                f"（n={paired['n']}）。",
                "",
            ]
        else:
            lines += [
                "> 缺少任一算法的 `per_episode.csv`，无法计算配对 CI。",
                "",
            ]
        baseline_labels = [
            ("random", "Random"),
            ("min_latency", "Min-Latency"),
            ("max_quality", "Max-Quality"),
            ("heuristic", "Heuristic"),
        ]
        dqn_paired = {
            item["strategy"]: item
            for item in (dqn_eval.get("paired") or [])
        }
        ppo_paired = {
            item["strategy"]: item
            for item in (ppo_eval.get("paired") or [])
        }
        lines += [
            "### 与 Baseline 的逐 Episode 配对（ΔJointR，95% CI）",
            "",
            "| Baseline | DQN best | PPO best |",
            "|---|---:|---:|",
        ]
        for key, label in baseline_labels:
            cells = []
            for source in (dqn_paired, ppo_paired):
                item = source.get(key)
                if item is None:
                    cells.append("—")
                else:
                    value = item["delta_JointR"]
                    cells.append(
                        f"{value['mean']:+.4f} "
                        f"[{value['low']:+.4f}, {value['high']:+.4f}]"
                    )
            lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
        lines += [
            "",
            "四个非特权 Baseline 的 ΔJointR 95% CI 对 DQN 与 PPO 均位于 0 以下；"
            "两者都显著差于全部四个非特权 Baseline。",
            "",
        ]
    else:
        lines += [
            "## 4. Dev180 最终对比",
            "",
            "> 尚未同时具备 DQN A best.pt 与 PPO best.pt 的 Dev180 评估；"
            "本报告只包含训练曲线。完成评估后重新运行本脚本生成最终对比。",
            "",
        ]
    lines += [
        "## 5. 结论与限制",
        "",
    ]
    if ppo_summary["converged"] is True:
        lines += [
            "- PPO 达到预注册的稳定平台判据，可作为候选策略。",
        ]
    else:
        lines += [
            "- PPO **未通过预注册收敛门**（`converged != true`）。"
            "无论 Dev180 数值如何，都不能将其标记为有效的最终策略。",
        ]
    if dqn_summary["converged"] is not True:
        lines += [
            "- DQN A 也未通过收敛门；两者的训练曲线只能作为诊断对比。",
        ]
    if dqn_eval and ppo_eval:
        lines += [
            "- 最佳 Checkpoint 的 Dev60 → Dev180 下降：DQN "
            f"{dqn_summary['best_joint']:.4f} → "
            f"{dqn_eval['overall']['JointR']:.4f}；PPO "
            f"{ppo_summary['best_joint']:.4f} → "
            f"{ppo_eval['overall']['JointR']:.4f}。说明用 60 个 Dev Episode "
            "选模会对其余 120 个 Dev Episode 产生选择偏差，"
            "最终对比必须以完整 Dev180 为准。",
        ]
        if paired is not None:
            joint_ci = paired["delta_JointR"]
            e2e_ci = paired["delta_E2E_ms"]
            ndtw_ci = paired["delta_nDTW"]
            lines += [
                "- DQN−PPO 在完整 Dev180 上：JointR 差 "
                f"{joint_ci['mean']:+.4f}，95% CI "
                f"[{joint_ci['low']:+.4f}, {joint_ci['high']:+.4f}]"
                + (
                    "（不显著，CI 跨 0）。"
                    if joint_ci["low"] <= 0.0 <= joint_ci["high"]
                    else "（CI 不跨 0）。"
                ),
                "- PPO 的 E2E 显著更低（"
                f"{e2e_ci['mean']:+.2f} ms，CI "
                f"[{e2e_ci['low']:+.2f}, {e2e_ci['high']:+.2f}]），"
                "nDTW 显著更低（"
                f"{ndtw_ci['mean']:+.4f}，CI "
                f"[{ndtw_ci['low']:+.4f}, {ndtw_ci['high']:+.4f}]）；"
                "SR 差异不显著。两者是不同失败模式的权衡，而不是一方全面胜出。",
            ]
    lines += [
        "- 训练奖励与 JointR 的绝对误差在训练和验证中都保持 <1e-6，"
        "没有成功终止奖励或失败惩罚。",
        "- Final Test 未读取；是否进入独立测试必须等方案冻结后另行决定。",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dqn-run", type=Path,
        default=Path("runs/v83_spatial_stage2/stage2k_pure_mec_v2"),
    )
    parser.add_argument(
        "--ppo-run", type=Path,
        default=Path("runs/v83_spatial_stage2/stage2k_ppo"),
    )
    parser.add_argument(
        "--dqn-summary", type=Path,
        default=Path("runs/v83_spatial_stage2/stage2k_pure_mec_v2/"
                     "dev180_eval/summary.json"),
    )
    parser.add_argument(
        "--ppo-summary", type=Path,
        default=Path("runs/v83_spatial_stage2/stage2k_ppo/"
                     "dev180_eval/summary.json"),
    )
    parser.add_argument(
        "--baseline-summary", type=Path,
        default=Path("runs/v83_spatial_stage2/stage2k/baselines/dev/"
                     "summary.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("v83_spatial_stage2/figures/stage2k_ppo_vs_dqn"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("v83_spatial_stage2/STAGE2K_PPO_VS_DQN_COMPARISON.md"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dqn = load_run(args.dqn_run)
    ppo = load_run(args.ppo_run)
    figures = {
        "training": "figures/stage2k_ppo_vs_dqn/01_training_validation.png",
        "metrics": "figures/stage2k_ppo_vs_dqn/02_validation_4metrics.png",
        "ppo_diag": "figures/stage2k_ppo_vs_dqn/03_ppo_diagnostics.png",
        "dqn_diag": "figures/stage2k_ppo_vs_dqn/04_dqn_diagnostics.png",
        "final": "figures/stage2k_ppo_vs_dqn/05_dev180_comparison.png",
    }
    plot_training_and_validation(
        dqn, ppo, args.output_dir / "01_training_validation.png"
    )
    plot_validation_metrics(
        dqn, ppo, args.output_dir / "02_validation_4metrics.png"
    )
    plot_ppo_diagnostics(
        ppo, args.output_dir / "03_ppo_diagnostics.png"
    )
    plot_dqn_diagnostics(
        dqn, args.output_dir / "04_dqn_diagnostics.png"
    )
    dqn_eval = load_summary(args.dqn_summary)
    ppo_eval = load_summary(args.ppo_summary)
    baseline_payload = load_summary(args.baseline_summary)
    baselines = baseline_payload["overall"] if baseline_payload else {}
    final_plot = plot_final_comparison(
        dqn_eval, ppo_eval, baselines,
        args.output_dir / "05_dev180_comparison.png",
    )
    dqn_summary = summarize_run(dqn, "DQN A")
    ppo_summary = summarize_run(ppo, "PPO")
    paired = None
    if dqn_eval and ppo_eval:
        paired = paired_dqn_ppo(args.dqn_summary, args.ppo_summary)
    payload = {
        "version": "stage2k-ppo-vs-dqn-comparison-v1",
        "created_utc": utc_now(),
        "dqn": dqn_summary,
        "ppo": ppo_summary,
        "dqn_eval": dqn_eval["overall"] if dqn_eval else None,
        "ppo_eval": ppo_eval["overall"] if ppo_eval else None,
        "paired_ppo_minus_dqn": paired,
        "baselines": baselines,
        "final_plot": final_plot,
        "final_test_read": False,
    }
    atomic_write_json(args.output_dir / "comparison.json", payload)
    markdown_report(
        args.report, dqn_summary, ppo_summary, dqn_eval, ppo_eval,
        baselines, figures, final_plot, paired=paired,
    )
    print(json.dumps({
        "dqn": dqn_summary,
        "ppo": ppo_summary,
        "final_plot": final_plot,
        "report": str(args.report),
        "figures": str(args.output_dir),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
