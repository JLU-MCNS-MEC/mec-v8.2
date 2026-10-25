"""Stage-2s trajectory figures (300 dpi, CSV sidecars).

* ``fig_traj_dqn.png`` / ``fig_traj_ppo.png``: 2x3 success/failure x
  short/medium/long closed-loop trajectories.
* ``fig_traj_overview.png``: both algorithms, all selected roles.
* ``fig_traj_model_usage.png``: Small/Transformer-M step share per role.
* ``fig_traj_selection_vs_outcome.png``: selection ratio vs success/nDTW.
* ``fig_traj_latency_example.png``: per-step communication/inference/E2E with
  the selected model type in the background.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2o.plot import (  # noqa: E402
    BASELINE,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    _finish,
    _save,
    setup_style,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (  # noqa: E402
    ALGORITHM_LABELS,
    EVIDENCE_DIR,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    load_deployment,
)

MODEL_COLORS = {"Small": "#1baf7a", "Transformer-M": "#6b4fd8"}
ALGO_COLORS = {"dqn": "#2a78d6", "ppo": "#eb6834"}
ROLES = [
    ("success", "short"), ("success", "medium"), ("success", "long"),
    ("failure", "short"), ("failure", "medium"), ("failure", "long"),
]
ROLE_TITLES = {
    ("success", "short"): "成功 · Short",
    ("success", "medium"): "成功 · Medium",
    ("success", "long"): "成功 · Long",
    ("failure", "short"): "失败 · Short",
    ("failure", "medium"): "失败 · Medium",
    ("failure", "long"): "失败 · Long",
}


def _read(name):
    path = EVIDENCE_DIR / name
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _nodes():
    payload = load_deployment()
    return pd.DataFrame([{
        "node_id": node["node_id"],
        "x": node["x"],
        "y": node["y"],
        "model_type": node["model_type"],
    } for node in payload["nodes"]])


def _role_key(row):
    return (
        "success" if int(row["success"]) else "failure",
        str(row["band"]),
    )


def _plot_trajectory_panel(ax, steps, summary, nodes, annotate=True,
                           title=None):
    steps = steps.sort_values("step")
    xy = steps[["x_m", "y_m"]].to_numpy(dtype=float)
    colors = []
    for index in range(1, len(steps)):
        model = str(steps.iloc[index]["model_type"])
        colors.append(MODEL_COLORS.get(model, INK_MUTED))
    if len(xy) > 1:
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        ax.add_collection(LineCollection(
            segments, colors=colors, linewidths=2.2, zorder=3,
        ))
    for model in ("Small", "Transformer-M"):
        subset = nodes[nodes["model_type"] == model]
        ax.scatter(
            subset["x"], subset["y"], s=34,
            marker="o" if model == "Small" else "s",
            facecolor=MODEL_COLORS[model], edgecolor=SURFACE,
            linewidth=0.5, zorder=5,
        )
        for _, node in subset.iterrows():
            ax.annotate(
                node["node_id"], (node["x"], node["y"]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=5.2, color=INK_MUTED, zorder=6,
            )
    ax.scatter(
        xy[0, 0], xy[0, 1], s=90, marker="^", color="#1baf7a",
        edgecolor=INK, linewidth=0.5, zorder=7, label="起点",
    )
    ax.scatter(
        summary["goal_x_m"], summary["goal_y_m"], s=110, marker="*",
        color="#d4a017", edgecolor=INK, linewidth=0.5, zorder=8,
        label="目标",
    )
    ax.scatter(
        xy[-1, 0], xy[-1, 1], s=70, marker="X", color="#c2417a",
        edgecolor=INK, linewidth=0.5, zorder=7, label="终点",
    )
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 1500)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title, fontsize=9.5)
    if annotate:
        status = "成功" if int(summary["success"]) else "失败"
        text = (
            f"Ep {str(summary['episode_uid']).split('-')[-1]}\n"
            f"{status} | {summary['band']} | {summary['steps']} 步\n"
            f"路径 {summary['path_length_m']:.1f} m\n"
            f"E2E {summary['mean_e2e_ms']:.1f} ms | "
            f"R_D={summary['reward_D']:.3f}\n"
            f"Small {100*summary['small_ratio']:.1f}% / "
            f"TM {100*summary['transformer_m_ratio']:.1f}%"
        )
        ax.text(
            0.02, 0.98, text, transform=ax.transAxes, va="top",
            ha="left", fontsize=6.4, color=INK,
            bbox=dict(
                boxstyle="round,pad=0.25", facecolor=SURFACE,
                edgecolor=BASELINE, linewidth=0.5, alpha=0.9,
            ),
            zorder=9,
        )
    _finish(ax)


def _trajectory_figure(steps, summaries, nodes, algorithm, name):
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.2))
    rows = []
    for ax, role in zip(axes.ravel(), ROLES):
        subset = summaries[
            (summaries["algorithm"] == algorithm)
            & (summaries["success"].astype(bool) == (role[0] == "success"))
            & (summaries["band"] == role[1])
        ]
        if subset.empty:
            ax.text(
                0.5, 0.5, "无该层轨迹", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=INK_MUTED,
            )
            ax.set_xlim(0, 1500)
            ax.set_ylim(0, 1500)
            _finish(ax)
            continue
        summary = subset.iloc[0].to_dict()
        episode_steps = steps[
            steps["episode_uid"] == summary["episode_uid"]
        ]
        _plot_trajectory_panel(
            ax, episode_steps, summary, nodes, annotate=True,
            title=ROLE_TITLES[role],
        )
        rows.append(summary)
    handles = [
        Patch(color=MODEL_COLORS["Small"], label="Small 节点/所选段"),
        Patch(
            color=MODEL_COLORS["Transformer-M"],
            label="Transformer-M 节点/所选段",
        ),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#1baf7a",
               markeredgecolor=INK, markersize=8, label="起点"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#d4a017",
               markeredgecolor=INK, markersize=12, label="目标"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#c2417a",
               markeredgecolor=INK, markersize=8, label="终点"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False,
        fontsize=8.5, bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        f"{ALGORITHM_LABELS[algorithm]} 真实闭环轨迹："
        "成功/失败 × Short/Medium/Long（线段颜色=所选模型）",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save(
        fig, FIGURES_DIR / name, pd.DataFrame(rows), FIGURE_DATA_DIR,
    )


def plot_algorithm(steps, summaries, nodes):
    for algorithm in ("dqn", "ppo"):
        _trajectory_figure(
            steps, summaries, nodes, algorithm,
            f"fig_traj_{algorithm}.png",
        )


def plot_overview(steps, summaries, nodes):
    fig, axes = plt.subplots(
        2, 6, figsize=(20.0, 7.0),
        gridspec_kw={"wspace": 0.12, "hspace": 0.18},
    )
    rows = []
    for row_index, algorithm in enumerate(("dqn", "ppo")):
        for col_index, role in enumerate(ROLES):
            ax = axes[row_index, col_index]
            subset = summaries[
                (summaries["algorithm"] == algorithm)
                & (summaries["success"].astype(bool)
                   == (role[0] == "success"))
                & (summaries["band"] == role[1])
            ]
            if subset.empty:
                ax.set_axis_off()
                continue
            summary = subset.iloc[0].to_dict()
            episode_steps = steps[
                steps["episode_uid"] == summary["episode_uid"]
            ]
            _plot_trajectory_panel(
                ax, episode_steps, summary, nodes, annotate=False,
                title=(
                    f"{ALGORITHM_LABELS[algorithm]}\n"
                    f"{ROLE_TITLES[role]}"
                ),
            )
            ax.tick_params(labelsize=5.5)
            ax.set_xlabel("")
            ax.set_ylabel("")
            summary["role"] = f"{role[0]}_{role[1]}"
            summary["algorithm"] = algorithm
            rows.append(summary)
    fig.suptitle(
        "DQN（上）与 PPO（下）真实闭环轨迹总览：线段颜色=所选模型",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(
        fig, FIGURES_DIR / "fig_traj_overview.png",
        pd.DataFrame(rows), FIGURE_DATA_DIR,
    )


def plot_model_usage(summaries):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6),
                             sharey=True)
    rows = []
    for ax, algorithm in zip(axes, ("dqn", "ppo")):
        subset = summaries[summaries["algorithm"] == algorithm].copy()
        subset["role"] = [
            f"{'S' if int(row['success']) else 'F'}-{row['band']}"
            for _, row in subset.iterrows()
        ]
        subset = subset.sort_values("role")
        x = np.arange(len(subset))
        small = subset["small_ratio"].to_numpy(dtype=float)
        transformer = subset["transformer_m_ratio"].to_numpy(dtype=float)
        ax.bar(x, small, color=MODEL_COLORS["Small"], label="Small")
        ax.bar(
            x, transformer, bottom=small,
            color=MODEL_COLORS["Transformer-M"],
            label="Transformer-M",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(subset["role"], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("步数比例")
        ax.set_title(f"{ALGORITHM_LABELS[algorithm]}：所选模型比例")
        ax.legend(loc="upper right")
        _finish(ax)
        rows.extend({
            "algorithm": algorithm,
            "role": row["role"],
            "episode_uid": row["episode_uid"],
            "small_ratio": float(row["small_ratio"]),
            "transformer_m_ratio": float(row["transformer_m_ratio"]),
            "success": int(row["success"]),
            "band": row["band"],
        } for _, row in subset.iterrows())
    fig.tight_layout()
    _save(
        fig, FIGURES_DIR / "fig_traj_model_usage.png",
        pd.DataFrame(rows), FIGURE_DATA_DIR,
    )


def plot_selection_vs_outcome(pool):
    if pool is None or pool.empty:
        return
    bins = np.linspace(0.0, 1.0, 6)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(5)]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    rows = []
    for ax, metric, title in (
        (axes[0], "success", "成功率"),
        (axes[1], "final_ndtw", "nDTW"),
    ):
        for algorithm in ("dqn", "ppo"):
            subset = pool[pool["algorithm"] == algorithm].copy()
            if subset.empty:
                continue
            subset["bin"] = pd.cut(
                subset["small_ratio"], bins=bins, labels=labels,
                include_lowest=True,
            )
            grouped = subset.groupby("bin", observed=True)[metric].mean()
            ax.plot(
                range(len(grouped)), grouped.to_numpy(dtype=float),
                marker="o", color=ALGO_COLORS[algorithm],
                label=ALGORITHM_LABELS[algorithm],
            )
            rows.extend({
                "algorithm": algorithm,
                "metric": metric,
                "small_ratio_bin": str(bin_label),
                "value": float(value),
            } for bin_label, value in grouped.items())
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_xlabel("Small 步数比例分箱")
        ax.set_ylabel(title)
        ax.set_title(
            f"Small 选择比例 vs {title}（Dev-60 全池）"
        )
        ax.legend(loc="best")
        _finish(ax)
    fig.tight_layout()
    _save(
        fig, FIGURES_DIR / "fig_traj_selection_vs_outcome.png",
        pd.DataFrame(rows), FIGURE_DATA_DIR,
    )


def plot_latency_example(steps, summaries):
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 7.0), sharex=False)
    for ax, algorithm in zip(axes, ("dqn", "ppo")):
        subset = summaries[
            (summaries["algorithm"] == algorithm)
            & (summaries["success"].astype(bool))
            & (summaries["band"] == "medium")
        ]
        if subset.empty:
            subset = summaries[summaries["algorithm"] == algorithm]
        if subset.empty:
            continue
        summary = subset.iloc[0].to_dict()
        episode_steps = steps[
            steps["episode_uid"] == summary["episode_uid"]
        ].sort_values("step")
        episode_steps = episode_steps[episode_steps["step"] > 0]
        x = episode_steps["step"].to_numpy(dtype=float)
        ax.plot(
            x, episode_steps["communication_ms"], color="#52514e",
            linewidth=1.6, label="通信时延",
        )
        ax.plot(
            x, episode_steps["inference_ms"], color="#2a78d6",
            linewidth=1.6, label="推理时延",
        )
        ax.plot(
            x, episode_steps["e2e_ms"], color="#eb6834",
            linewidth=1.8, label="E2E",
        )
        for index, (_, row) in enumerate(episode_steps.iterrows()):
            if index == 0:
                continue
            if row["model_type"] != episode_steps.iloc[
                index - 1
            ]["model_type"]:
                ax.axvline(
                    float(row["step"]),
                    color=MODEL_COLORS.get(
                        str(row["model_type"]), INK_MUTED
                    ),
                    linewidth=0.8, alpha=0.5,
                )
        ax.set_title(
            f"{ALGORITHM_LABELS[algorithm]}：{summary['episode_uid']} "
            f"逐步时延（成功/medium；竖线=模型切换）"
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("时延 (ms)")
        ax.axhline(100.0, color=BASELINE, linewidth=1.0)
        ax.legend(loc="upper right")
        _finish(ax)
        rows.extend({
            "algorithm": algorithm,
            "episode_uid": summary["episode_uid"],
            "step": int(row["step"]),
            "model_type": row["model_type"],
            "communication_ms": float(row["communication_ms"]),
            "inference_ms": float(row["inference_ms"]),
            "e2e_ms": float(row["e2e_ms"]),
        } for _, row in episode_steps.iterrows())
    fig.tight_layout()
    _save(
        fig, FIGURES_DIR / "fig_traj_latency_example.png",
        pd.DataFrame(rows), FIGURE_DATA_DIR,
    )


def make_all():
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    steps = _read("trajectory_steps.csv")
    summaries = _read("trajectory_summary.csv")
    pool = _read("trajectory_dev_pool.csv")
    if steps is None or summaries is None:
        raise SystemExit("trajectory CSVs are missing; run trajectories.py")
    nodes = _nodes()
    plot_algorithm(steps, summaries, nodes)
    plot_overview(steps, summaries, nodes)
    plot_model_usage(summaries)
    plot_selection_vs_outcome(pool)
    plot_latency_example(steps, summaries)
    print("[plot_traj] figures written to", FIGURES_DIR, flush=True)
    print("STAGE2S_PLOT_TRAJECTORY_COMPLETE", flush=True)


def main(argv=None):
    make_all()


if __name__ == "__main__":
    main()
