"""Plot the Stage-2r UAV trajectory figures from the tracked CSVs.

Outputs four 2x3 condition figures (Small/TM x DQN/PPO), one 4x6 overview,
and one consistency-audit figure.  The script is CPU-only and reads only the
CSVs produced by ``trajectories.py`` plus the frozen deployment geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENV  # noqa: E402
from v83_spatial_stage2.stage2o.plot import (  # noqa: E402
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    _finish,
    _legend,
    _save,
    setup_style,
)
from v83_spatial_stage2.stage2r_10node_small_vs_tm.common import (  # noqa: E402
    ALGORITHMS,
    ALGORITHM_LABELS,
    EVIDENCE_DIR,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    MEC_MODELS,
    MODEL_LABELS,
    load_deployment,
)

MODEL_COLORS = {"Small": "#2a78d6", "Transformer-M": "#eb6834"}
ROLES = (
    "success_short", "failure_short",
    "success_medium", "failure_medium",
    "success_long", "failure_long",
)
ROLE_TITLES = {
    "success_short": "成功 · short",
    "failure_short": "失败 · short",
    "success_medium": "成功 · medium",
    "failure_medium": "失败 · medium",
    "success_long": "成功 · long",
    "failure_long": "失败 · long",
}
CONDITION_ORDER = [
    ("Small", "dqn"), ("Small", "ppo"),
    ("Transformer-M", "dqn"), ("Transformer-M", "ppo"),
]


def _condition(model, algorithm):
    return f"{MODEL_LABELS[model]}|{ALGORITHM_LABELS[algorithm]}"


def _read():
    steps = pd.read_csv(EVIDENCE_DIR / "trajectory_steps.csv")
    summary = pd.read_csv(EVIDENCE_DIR / "trajectory_summary.csv")
    selected = pd.read_csv(EVIDENCE_DIR / "selected_tasks.csv")
    audit = pd.read_csv(EVIDENCE_DIR / "trajectory_consistency_audit.csv")
    return steps, summary, selected, audit


def _nodes(model):
    deployment = load_deployment(model)
    return np.asarray(
        [[float(node["x"]), float(node["y"])]
         for node in deployment["nodes"]],
        dtype=float,
    )


def _plot_one(ax, steps, summary, selection, model, algorithm):
    condition = _condition(model, algorithm)
    subset = steps[
        (steps["condition"] == condition)
        & (steps["episode_uid"] == selection["episode_uid"])
    ].sort_values("step")
    if subset.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center")
        ax.set_axis_off()
        return
    color = MODEL_COLORS[model]
    x = subset["x_m"].to_numpy(dtype=float)
    y = subset["y_m"].to_numpy(dtype=float)
    ax.plot(x, y, color=color, linewidth=2.0, zorder=3)
    nodes = _nodes(model)
    ax.scatter(
        nodes[:, 0], nodes[:, 1], marker="s", s=24,
        facecolor=SURFACE, edgecolor=INK_MUTED, linewidth=1.0, zorder=2,
    )
    for index, (nx, ny) in enumerate(nodes, start=1):
        ax.text(nx, ny, f"M{index}", fontsize=6, color=INK_MUTED,
                ha="center", va="bottom")
    ax.scatter(
        [x[0]], [y[0]], marker="o", s=55, facecolor="#1baf7a",
        edgecolor=SURFACE, linewidth=1.2, zorder=5,
    )
    success = int(selection["lane_success"])
    ax.scatter(
        [x[-1]], [y[-1]], marker="X" if not success else "P",
        s=60, facecolor="#e34948" if not success else "#2a78d6",
        edgecolor=SURFACE, linewidth=1.0, zorder=5,
    )
    # The selected_tasks.csv has no goal; read it from the summary row.
    summary_row = summary[
        (summary["condition"] == condition)
        & (summary["episode_uid"] == selection["episode_uid"])
    ]
    if not summary_row.empty:
        row = summary_row.iloc[0]
        ax.scatter(
            [row["goal_x_m"]], [row["goal_y_m"]], marker="*", s=130,
            facecolor=INK, edgecolor=SURFACE, linewidth=1.0, zorder=6,
        )
    title = ROLE_TITLES.get(selection["role"], selection["role"])
    title += (
        f"\n{selection['episode_uid']} | steps={int(selection['lane_steps'])}"
        f" | 实际{'成功' if success else '失败'}"
    )
    if str(selection["selection_status"]) != "exact":
        title += f"\n[替代: {selection['selection_status']}]"
    ax.set_title(title, fontsize=7.8, color=INK)
    ax.set_xlim(-20, ENV.world_size_m + 20)
    ax.set_ylim(-20, ENV.world_size_m + 20)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("y (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    _finish(ax)


def plot_condition_figure(steps, summary, selected, model, algorithm):
    condition = _condition(model, algorithm)
    selection_rows = selected[selected["model"] == model]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 9.2))
    for ax, role in zip(axes.ravel(), ROLES):
        match = selection_rows[selection_rows["role"] == role]
        if match.empty:
            ax.set_axis_off()
            continue
        _plot_one(ax, steps, summary, match.iloc[0], model, algorithm)
    fig.suptitle(
        f"Stage-2r UAV 轨迹：{MODEL_LABELS[model]} + "
        f"{ALGORITHM_LABELS[algorithm]}（Dev-60 代表任务）",
        fontsize=13, color=INK,
    )
    handles = [
        Line2D([0], [0], color=MODEL_COLORS[model], linewidth=2,
               label="飞行轨迹"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="#1baf7a", markeredgecolor=SURFACE,
               markersize=7, label="起点"),
        Line2D([0], [0], marker="P", color="none",
               markerfacecolor="#2a78d6", markeredgecolor=SURFACE,
               markersize=8, label="终点（成功）"),
        Line2D([0], [0], marker="X", color="none",
               markerfacecolor="#e34948", markeredgecolor=SURFACE,
               markersize=8, label="终点（失败）"),
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor=INK, markeredgecolor=SURFACE,
               markersize=10, label="目标位置"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=SURFACE, markeredgecolor=INK_MUTED,
               markersize=6, label="10 个 MEC 节点"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=6, frameon=False,
        fontsize=8, bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    name = (
        f"fig_traj_{model.lower().replace('-', '_')}_{algorithm}.png"
    )
    data = steps[
        (steps["condition"] == condition)
    ]
    _save(fig, FIGURES_DIR / name, data, FIGURE_DATA_DIR)
    return FIGURES_DIR / name


def plot_overview(steps, summary, selected):
    fig, axes = plt.subplots(4, 6, figsize=(22.0, 16.0))
    for row_index, (model, algorithm) in enumerate(CONDITION_ORDER):
        selection_rows = selected[selected["model"] == model]
        for col_index, role in enumerate(ROLES):
            ax = axes[row_index, col_index]
            match = selection_rows[selection_rows["role"] == role]
            if match.empty:
                ax.set_axis_off()
                continue
            _plot_one(ax, steps, summary, match.iloc[0], model, algorithm)
    fig.suptitle(
        "Stage-2r 四类条件 UAV 轨迹总览（行：条件；列：成功/失败 × "
        "short/medium/long）",
        fontsize=15, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = FIGURES_DIR / "fig_traj_overview_4x6.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    summary.to_csv(FIGURE_DATA_DIR / "fig_traj_overview_4x6.csv", index=False)
    return path


def plot_audit(audit):
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    if audit.empty:
        ax.text(0.5, 0.5, "无审计数据", ha="center", va="center")
        ax.set_axis_off()
    else:
        stats = audit.groupby(["model", "policy"]).agg(
            path_identical=("path_identical", "mean"),
            action_identical=("action_identical", "mean"),
            success_identical=("success_identical", "mean"),
            node_differ=("node_choices_differ", "mean"),
        ).reset_index()
        labels = [
            f"{MODEL_LABELS[row['model']]}\n{row['policy']}"
            for _, row in stats.iterrows()
        ]
        x = np.arange(len(stats))
        width = 0.2
        for offset, column, color, title in (
            (-1.5, "path_identical", "#2a78d6", "路径一致"),
            (-0.5, "action_identical", "#1baf7a", "动作一致"),
            (0.5, "success_identical", "#eb6834", "成功一致"),
            (1.5, "node_differ", "#c3c2b7", "节点选择不同"),
        ):
            ax.bar(
                x + offset * width, stats[column], width=width,
                color=color, edgecolor=SURFACE, linewidth=0.8,
                label=title, zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("比例")
        ax.set_title("轨迹一致性审计：同一部署下跨关联策略")
        _legend(ax, loc="upper right")
        _finish(ax)
    fig.text(
        0.5, -0.04,
        "导航动作由冻结 MEC 模型产生，关联策略只改变节点选择/E2E；"
        "因此路径、动作、成功应 100% 一致，节点选择可以不同。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    _save(fig, FIGURES_DIR / "fig_traj_consistency_audit.png",
          audit, FIGURE_DATA_DIR)
    return FIGURES_DIR / "fig_traj_consistency_audit.png"


def make_all():
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    steps, summary, selected, audit = _read()
    paths = []
    for model, algorithm in CONDITION_ORDER:
        paths.append(plot_condition_figure(
            steps, summary, selected, model, algorithm
        ))
    paths.append(plot_overview(steps, summary, selected))
    paths.append(plot_audit(audit))
    return [path.name for path in paths]


def main():
    names = make_all()
    print(f"[traj-plot] {names}", flush=True)
    print("STAGE2R_TRAJECTORY_PLOT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
