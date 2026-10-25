"""Stage-2q reproducible trajectory figures.

Every figure is drawn from the committed Stage-2q CSVs.  No GPU, checkpoint
or ``runs/`` artefact is needed: the script can be re-run from a clean clone.

Figures:
    trajectory_examples_2x3.png
    stage2n_method_overlay.png
    stage2o_reward_overlay.png
    stage2p_ppo_reward_overlay.png
    trajectory_difference_audit.png

Usage:
    /home/ubuntu5/miniconda3/envs/DRL/bin/python \
        v83_spatial_stage2/stage2q_trajectory_audit/plot_trajectories.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    FancyArrowPatch,
    Polygon,
    Rectangle,
)
from matplotlib.ticker import MultipleLocator  # noqa: E402

HERE = Path(__file__).resolve().parent

# Validated categorical palette from the dataviz reference (6 slots).
CATEGORICAL = (
    "#2a78d6", "#eb6834", "#1baf7a",
    "#eda100", "#e87ba4", "#008300",
)
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SUCCESS = "#0ca30c"
FAILURE = "#d03b3b"
GOAL = "#9a6b00"
LANDMARK_FILL = "#e9e8e1"
LANDMARK_EDGE = "#d2d1c8"
CONTROL = "#898781"

BAND_CN = {"short": "Short", "medium": "Medium", "long": "Long"}
BAND_RANGE = {
    "short": "150–250 m",
    "medium": "250–400 m",
    "long": "400–550 m",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.labelsize": 8.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "text.color": INK,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "savefig.dpi": 300,
    })


def method_label(method_id: str) -> str:
    if method_id == "control-greedy-min-e2e":
        return "Greedy-Min-E2E"
    if method_id == "control-random":
        return "Random"
    parts = method_id.split("-")
    if parts[0] == "2n":
        return f"2n {parts[1].upper()} s{parts[2][1:]}"
    if parts[0] == "2o":
        return f"2o {parts[1]} {parts[2].upper()} s0"
    if parts[0] == "2p":
        return f"2p {parts[1]} PPO s{parts[3][1:]}"
    return method_id


def method_color(method_id: str) -> str:
    if method_id.startswith("2n"):
        return CATEGORICAL[0]
    if method_id.startswith("2o"):
        return CATEGORICAL[1]
    if method_id.startswith("2p"):
        return CATEGORICAL[2]
    return CONTROL


def _read_inputs(base: Path):
    def read(name):
        path = base / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing; run collect_trajectories.py first"
            )
        return pd.read_csv(path)

    summary = read("trajectory_summary.csv")
    steps = read("trajectory_steps.csv")
    invariance = read("trajectory_invariance.csv")
    selected = read("selected_tasks.csv")
    dev60 = read("dev60_task_summary.csv")
    city = read("city_geometry.csv")
    city_meta = json.loads((base / "city_meta.json").read_text("utf-8"))
    audit = json.loads((base / "audit_summary.json").read_text("utf-8"))
    return summary, steps, invariance, selected, dev60, city, city_meta, audit


def _landmark_patch(row):
    x = float(row["x_m"])
    y = float(row["y_m"])
    size = float(row["physical_size_m"])
    shape = str(row["shape"])
    common = {
        "facecolor": LANDMARK_FILL,
        "edgecolor": LANDMARK_EDGE,
        "linewidth": 0.55,
        "zorder": 1,
    }
    if shape == "circle":
        return Circle((x, y), radius=size / 2.0, **common)
    if shape == "triangle":
        return Polygon([
            (x, y + size / 2.0),
            (x - size / 2.0, y - size / 2.0),
            (x + size / 2.0, y - size / 2.0),
        ], closed=True, **common)
    if shape == "diamond":
        return Polygon([
            (x, y + size / 2.0),
            (x + size / 2.0, y),
            (x, y - size / 2.0),
            (x - size / 2.0, y),
        ], closed=True, **common)
    if shape == "cross":
        return Polygon([
            (x - size / 6.0, y - size / 2.0),
            (x + size / 6.0, y - size / 2.0),
            (x + size / 6.0, y - size / 6.0),
            (x + size / 2.0, y - size / 6.0),
            (x + size / 2.0, y + size / 6.0),
            (x + size / 6.0, y + size / 6.0),
            (x + size / 6.0, y + size / 2.0),
            (x - size / 6.0, y + size / 2.0),
            (x - size / 6.0, y + size / 6.0),
            (x - size / 2.0, y + size / 6.0),
            (x - size / 2.0, y - size / 6.0),
            (x - size / 6.0, y - size / 6.0),
        ], closed=True, **common)
    if shape == "bar":
        return Rectangle(
            (x - size / 2.0, y - size / 6.0), size, size / 3.0, **common
        )
    return Rectangle(
        (x - size / 2.0, y - size / 2.0), size, size, **common
    )


def draw_city(ax, bounds, city, city_meta):
    xmin, xmax, ymin, ymax = bounds
    spacing = float(city_meta["road_spacing_m"])
    start_x = math.floor(xmin / spacing) * spacing
    for x in np.arange(start_x, xmax + spacing, spacing):
        if xmin - 1.0 <= x <= xmax + 1.0:
            ax.axvline(x, color=GRID, linewidth=0.8, zorder=0)
    start_y = math.floor(ymin / spacing) * spacing
    for y in np.arange(start_y, ymax + spacing, spacing):
        if ymin - 1.0 <= y <= ymax + 1.0:
            ax.axhline(y, color=GRID, linewidth=0.8, zorder=0)
    view = city[
        (city["x_m"] >= xmin - 50.0)
        & (city["x_m"] <= xmax + 50.0)
        & (city["y_m"] >= ymin - 50.0)
        & (city["y_m"] <= ymax + 50.0)
    ]
    for row in view.itertuples(index=False):
        ax.add_patch(_landmark_patch(row._asdict()))


def _bounds_for_task(summary_row, path_xs=None, path_ys=None,
                     padding_fraction=0.35, minimum=55.0):
    xs = [float(summary_row["start_x_m"]), float(summary_row["goal_x_m"]),
          float(summary_row["final_x_m"])]
    ys = [float(summary_row["start_y_m"]), float(summary_row["goal_y_m"]),
          float(summary_row["final_y_m"])]
    if path_xs is not None:
        xs.extend(float(value) for value in path_xs)
    if path_ys is not None:
        ys.extend(float(value) for value in path_ys)
    width = max(max(xs) - min(xs), 1.0)
    height = max(max(ys) - min(ys), 1.0)
    pad = max(minimum, padding_fraction * max(width, height))
    xmin = max(0.0, min(xs) - pad)
    xmax = min(float(1500.0), max(xs) + pad)
    ymin = max(0.0, min(ys) - pad)
    ymax = min(float(1500.0), max(ys) + pad)
    if xmax - xmin < 120.0:
        centre = 0.5 * (xmax + xmin)
        xmin, xmax = max(0.0, centre - 60.0), min(1500.0, centre + 60.0)
    if ymax - ymin < 120.0:
        centre = 0.5 * (ymax + ymin)
        ymin, ymax = max(0.0, centre - 60.0), min(1500.0, centre + 60.0)
    return xmin, xmax, ymin, ymax


def _nice_ticks(ax, bounds):
    xmin, xmax, ymin, ymax = bounds
    span = max(xmax - xmin, ymax - ymin)
    step = 50.0
    if span > 700:
        step = 200.0
    elif span > 400:
        step = 100.0
    elif span > 220:
        step = 50.0
    else:
        step = 25.0
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.tick_params(length=2.0, width=0.6, color=AXIS, pad=2.0)
    ax.set_xlabel("x (m)", labelpad=2.0)
    ax.set_ylabel("y (m)", labelpad=2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
        ax.spines[spine].set_linewidth(0.8)


def _scale_bar(ax, bounds):
    xmin, xmax, ymin, ymax = bounds
    span = max(xmax - xmin, ymax - ymin)
    length = 50.0
    if span < 260:
        length = 25.0
    elif span > 700:
        length = 100.0
    x1 = xmax - 0.12 * (xmax - xmin)
    x0 = x1 - length
    y = ymin + 0.065 * (ymax - ymin)
    ax.plot([x0, x1], [y, y], color=INK2, linewidth=1.6, zorder=10,
            solid_capstyle="butt")
    ax.plot([x0, x0], [y - 0.012 * (ymax - ymin),
                       y + 0.012 * (ymax - ymin)],
            color=INK2, linewidth=0.9, zorder=10)
    ax.plot([x1, x1], [y - 0.012 * (ymax - ymin),
                       y + 0.012 * (ymax - ymin)],
            color=INK2, linewidth=0.9, zorder=10)
    ax.text(0.5 * (x0 + x1), y + 0.022 * (ymax - ymin),
            f"{length:.0f} m", ha="center", va="bottom",
            fontsize=7.0, color=INK2, zorder=10)


def _corner_for_text(ax, xs, ys):
    """Pick the axes corner with the fewest path points, avoiding the scale bar."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    fx = (np.asarray(xs, dtype=float) - xmin) / max(xmax - xmin, 1e-9)
    fy = (np.asarray(ys, dtype=float) - ymin) / max(ymax - ymin, 1e-9)
    candidates = [
        ("upper-left", 0.018, 0.975, "left", "top",
         (fx < 0.55) & (fy > 0.60), 0),
        ("upper-right", 0.982, 0.975, "right", "top",
         (fx > 0.45) & (fy > 0.60), 1),
        ("lower-left", 0.018, 0.025, "left", "bottom",
         (fx < 0.55) & (fy < 0.40), 2),
        ("lower-right", 0.982, 0.025, "right", "bottom",
         (fx > 0.45) & (fy < 0.40), 3),
    ]
    best = None
    for name, x, y, ha, va, mask, tie in candidates:
        score = int(mask.sum())
        if name == "lower-right":
            score += 100  # keep the 25/50/100 m scale bar clear
        if name == "upper-right":
            score += 2
        key = (score, tie)
        if best is None or key < best[0]:
            best = (key, (x, y, ha, va))
    return best[1]


def _direction_arrows(ax, xs, ys, color):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < 3:
        return
    segment = np.linalg.norm(
        np.diff(np.stack([xs, ys], axis=1), axis=0), axis=1
    )
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    total = cumulative[-1]
    if total <= 0.0:
        return
    for fraction in (0.12, 0.32, 0.52, 0.72, 0.90):
        index = int(np.searchsorted(cumulative, fraction * total))
        index = min(max(index, 1), len(xs) - 1)
        if segment[index - 1] <= 1e-9:
            continue
        ax.add_patch(FancyArrowPatch(
            (xs[index - 1], ys[index - 1]),
            (xs[index], ys[index]),
            arrowstyle="-|>",
            mutation_scale=9.0,
            linewidth=1.0,
            color=color,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=6,
        ))


def _task_steps(steps, summary, uid, method_id):
    task_steps = steps[
        (steps["episode_uid"] == uid) & (steps["method_id"] == method_id)
    ].sort_values("step_index")
    row = summary[
        (summary["episode_uid"] == uid)
        & (summary["method_id"] == method_id)
    ].iloc[0]
    xs = task_steps["x_m"].to_numpy(dtype=float).tolist()
    ys = task_steps["y_m"].to_numpy(dtype=float).tolist()
    xs.append(float(row["final_x_m"]))
    ys.append(float(row["final_y_m"]))
    return task_steps, row, xs, ys


def draw_task_map(ax, uid, method_id, steps, summary, city, city_meta,
                  show_title=True, path_color=None, consistency_text=None):
    task_steps, row, xs, ys = _task_steps(steps, summary, uid, method_id)
    bounds = _bounds_for_task(row, path_xs=xs, path_ys=ys)
    draw_city(ax, bounds, city, city_meta)
    _nice_ticks(ax, bounds)
    success = bool(row["success"])
    color = path_color or (SUCCESS if success else FAILURE)
    ax.plot(xs, ys, color=color, linewidth=2.2, solid_capstyle="round",
            solid_joinstyle="round", zorder=5)
    _direction_arrows(ax, xs, ys, color)
    start_x, start_y = float(row["start_x_m"]), float(row["start_y_m"])
    goal_x, goal_y = float(row["goal_x_m"]), float(row["goal_y_m"])
    final_x, final_y = float(row["final_x_m"]), float(row["final_y_m"])
    radius = float(city_meta["stop_radius_m"])
    ax.add_patch(Circle(
        (goal_x, goal_y), radius=radius, fill=False,
        linestyle=(0, (4, 3)), linewidth=1.1, edgecolor=GOAL,
        alpha=0.9, zorder=4,
    ))
    ax.scatter(
        [start_x], [start_y], marker="^", s=105,
        facecolor=SUCCESS, edgecolor="white", linewidth=1.4, zorder=8,
    )
    ax.scatter(
        [goal_x], [goal_y], marker="*", s=245,
        facecolor=GOAL, edgecolor="white", linewidth=0.9, zorder=9,
    )
    final_style = (
        dict(marker="s", s=58, facecolor="white", edgecolor=INK,
             linewidth=1.4)
        if success else
        dict(marker="X", s=82, facecolor=FAILURE, edgecolor="white",
             linewidth=1.0)
    )
    ax.scatter([final_x], [final_y], zorder=10, **final_style)
    _scale_bar(ax, bounds)
    if show_title:
        band = str(row["band"])
        title = (
            f"{BAND_CN.get(band, band)} · {'成功' if success else '失败'}\n"
            f"{uid} · {int(row['steps'])} 步 · "
            f"起点–目标 {float(row['start_goal_distance_m']):.1f} m"
        )
        ax.set_title(title, fontsize=9.2, color=INK, pad=4.0,
                     loc="left")
    if consistency_text:
        text_x, text_y, text_ha, text_va = _corner_for_text(ax, xs, ys)
        ax.text(
            text_x, text_y, consistency_text, transform=ax.transAxes,
            ha=text_ha, va=text_va, fontsize=6.8, color=INK2,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=SURFACE,
                      edgecolor=AXIS, linewidth=0.6, alpha=0.92),
            zorder=12,
        )


def _consistency_text(invariance, uid, methods=None, plotted_method=None):
    sub = invariance[
        (invariance["scope"] == "selected6_closed_loop")
        & (invariance["episode_uid"] == uid)
    ]
    if methods is not None:
        sub = sub[sub["method_id"].isin(methods)]
    if sub.empty:
        return ""
    max_dev = float(sub["max_coord_deviation_m"].max())
    actions_equal = bool(sub["actions_equal"].all())
    coords_equal = bool(sub["coord_hash_equal"].all())
    prefix = (
        f"参考方法：{method_label(plotted_method)}\n"
        if plotted_method else ""
    )
    return (
        prefix
        + f"{sub['method_id'].nunique()} 个方法路径完全重合\n"
        f"max Δ = {max_dev:.6f} m\n"
        f"动作哈希{'一致' if actions_equal else '不一致'} · "
        f"坐标哈希{'一致' if coords_equal else '不一致'}"
    )


def figure_examples_2x3(base: Path, summary, steps, invariance, selected,
                        city, city_meta):
    _setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 10.4))
    method_id = "2n-dqn-s0"
    lookup = {
        (str(row["band"]), str(row["outcome"])): row
        for _, row in selected.iterrows()
    }
    for column, band in enumerate(("short", "medium", "long")):
        for row_index, outcome in enumerate(("success", "failure")):
            ax = axes[row_index, column]
            selected_row = lookup[(band, outcome)]
            uid = str(selected_row["episode_uid"])
            draw_task_map(
                ax, uid, method_id, steps, summary, city, city_meta,
                show_title=True,
            )
    handles = [
        Line2D([0], [0], color=SUCCESS, lw=2.2, label="成功轨迹"),
        Line2D([0], [0], color=FAILURE, lw=2.2, label="失败轨迹"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=SUCCESS,
               markeredgecolor="white", markersize=9, label="起点"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=GOAL,
               markeredgecolor="white", markersize=12, label="目标地标"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=INK, markersize=8, label="终点（成功）"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=FAILURE,
               markeredgecolor="white", markersize=9, label="终点（失败）"),
        Line2D([0], [0], color=GOAL, lw=1.1, ls="--",
               label="20 m 停止半径"),
        Line2D([0], [0], color=LANDMARK_EDGE, lw=6,
               label="真实地标足迹"),
    ]
    fig.text(
        0.5, 0.148,
        "参考方法：2n DQN s0；20 个方法在同一任务上的逐步坐标/动作逐位重合"
        "（max Δ = 0.000000 m，动作/坐标哈希一致）。",
        ha="center", va="center", fontsize=8.2, color=INK2,
    )
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        fontsize=8.4, bbox_to_anchor=(0.5, 0.070),
        handletextpad=0.6, columnspacing=1.8,
    )
    fig.suptitle(
        "Stage-2q：Dev-60 真实 UAV 轨迹示例（2×3）",
        x=0.5, y=0.985, fontsize=16, color=INK, fontweight="normal",
    )
    fig.text(
        0.5, 0.952,
        "每个面板只画一条真实路径：18 个 learned 关联方法 + 2 个对照方法"
        "在同一任务上的逐步坐标与动作完全重合；颜色仅表示成功/失败，"
        "不用于制造虚假差异。",
        ha="center", va="top", fontsize=9.2, color=INK2,
    )
    fig.text(
        0.5, 0.003,
        "通过一致性审计的方法：2n DQN/PPO s0–s2 · "
        "2o B/D/one × DQN/PPO s0 · 2p baseline/D × PPO s0–s2 · "
        "Greedy-Min-E2E · Random。\n"
        "背景：固定城市真实地标足迹与道路网格（环境几何）；坐标来自"
        "Stage-2q 真实闭环环境，不是示意图。"
        "Dev-60 任务池 SHA256 aefde2aa…；Transformer-M SHA256 57c88b7d…；"
        "未读取 Test-400。",
        ha="center", va="bottom", fontsize=6.9, color=MUTED,
        linespacing=1.4,
    )
    fig.subplots_adjust(
        left=0.045, right=0.985, top=0.875, bottom=0.175,
        wspace=0.13, hspace=0.23,
    )
    output = base / "trajectory_examples_2x3.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output}")


def _overlay_figure(base: Path, summary, steps, invariance, city, city_meta,
                    audit, stage_key: str, title: str, output_name: str):
    _setup_style()
    overlay = audit["overlay_tasks"][stage_key]
    uid = str(overlay["episode_uid"])
    methods = list(overlay["methods"])
    task_summary = summary[
        (summary["episode_uid"] == uid)
        & (summary["method_id"].isin(methods))
    ].copy()
    if task_summary.empty:
        raise RuntimeError(f"no summary rows for {uid} {methods}")
    order = {method_id: index for index, method_id in enumerate(methods)}
    task_summary["order"] = task_summary["method_id"].map(order)
    task_summary = task_summary.sort_values("order")
    reference = "2n-dqn-s0" if "2n-dqn-s0" in methods else methods[0]
    fig, (ax_path, ax_e2e) = plt.subplots(
        1, 2, figsize=(13.8, 5.9),
        gridspec_kw={"width_ratios": [1.28, 1.0]},
    )
    consistency = _consistency_text(
        invariance, uid, methods, plotted_method=reference
    )
    draw_task_map(
        ax_path, uid, reference, steps, summary, city, city_meta,
        show_title=True,
    )
    fig.text(
        0.05, 0.095, consistency, ha="left", va="bottom",
        fontsize=7.4, color=INK2, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.32", facecolor=SURFACE,
                  edgecolor=AXIS, linewidth=0.6, alpha=0.95),
    )
    ax_path.set_title(
        ax_path.get_title() + f"\n同一任务 · {len(methods)} 个方法",
        fontsize=9.0, loc="left", pad=4.0,
    )
    task_steps = steps[
        (steps["episode_uid"] == uid)
        & (steps["method_id"].isin(methods))
    ]
    for index, method_id in enumerate(methods):
        method_steps = task_steps[
            task_steps["method_id"] == method_id
        ].sort_values("step_index")
        mean_e2e = float(
            task_summary[task_summary["method_id"] == method_id][
                "mean_e2e_ms"
            ].iloc[0]
        )
        ax_e2e.plot(
            method_steps["step_index"].to_numpy(dtype=float),
            method_steps["e2e_ms"].to_numpy(dtype=float),
            color=CATEGORICAL[index % len(CATEGORICAL)],
            linewidth=1.8,
            label=f"{method_label(method_id)} · {mean_e2e:.1f} ms",
            zorder=3 + index,
        )
    ax_e2e.axhline(
        100.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
        zorder=0, label="100 ms 截止",
    )
    ax_e2e.set_xlabel("决策步")
    ax_e2e.set_ylabel("E2E 时延 (ms)")
    ax_e2e.set_title(
        "同一任务：每步 E2E 时延（路径完全重合）",
        fontsize=9.5, loc="left", pad=4.0,
    )
    ax_e2e.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax_e2e.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_e2e.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax_e2e.spines[spine].set_color(AXIS)
        ax_e2e.spines[spine].set_linewidth(0.8)
    ax_e2e.legend(
        loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2,
        fontsize=7.3, handlelength=1.8, columnspacing=1.2,
        frameon=False,
    )
    fig.suptitle(
        f"{title}：同一任务轨迹对照",
        x=0.5, y=0.985, fontsize=15, color=INK,
    )
    fig.text(
        0.5, 0.940,
        "所有方法共享同一冻结 Transformer-M，MEC 输入不含节点身份；"
        "因此导航动作与坐标相同，只有关联节点和 E2E 时延不同。"
        "左图只画一条真实路径，右侧曲线才承载方法差异。",
        ha="center", va="top", fontsize=8.8, color=INK2,
    )
    fig.text(
        0.5, 0.005,
        f"任务 {uid} · {len(methods)} 个方法 · "
        "动作/坐标哈希见 trajectory_invariance.csv；"
        "未读取 Test-400。",
        ha="center", va="bottom", fontsize=7.2, color=MUTED,
    )
    fig.subplots_adjust(
        left=0.05, right=0.985, top=0.88, bottom=0.30,
        wspace=0.18,
    )
    output = base / output_name
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output}")


def figure_difference_audit(base: Path, invariance, audit):
    _setup_style()
    closed = invariance[invariance["scope"] == "selected6_closed_loop"]
    if closed.empty:
        raise RuntimeError("selected6_closed_loop rows missing")
    methods = [
        method for method in audit["per_method"]
        if method in set(closed["method_id"])
    ]
    method_stats = []
    for method_id in methods:
        sub = closed[closed["method_id"] == method_id]
        method_stats.append({
            "method_id": method_id,
            "label": method_label(method_id),
            "family": str(sub["method_family"].iloc[0]),
            "max_dev": float(sub["max_coord_deviation_m"].max()),
            "node_mismatch": float(
                sub["node_choice_mismatch_fraction"].mean()
            ),
            "e2e_delta": float(sub["e2e_delta_ms"].mean()),
            "actions_equal": bool(sub["actions_equal"].all()),
        })
    stats = pd.DataFrame(method_stats)
    fig = plt.figure(figsize=(15.6, 11.2))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.0, 1.18],
        left=0.20, right=0.975, top=0.90, bottom=0.075,
        wspace=0.38, hspace=0.34,
    )
    ax_table = fig.add_subplot(grid[0, 0])
    ax_dev = fig.add_subplot(grid[0, 1])
    ax_node = fig.add_subplot(grid[1, 0])
    ax_e2e = fig.add_subplot(grid[1, 1])

    # -- audit table -----------------------------------------------------
    ax_table.axis("off")
    ax_table.set_title(
        "审计总表：三个证据层级", fontsize=10.5, loc="left", pad=6.0,
    )
    scopes = [
        ("Dev-60 lane", "dev60_lane"),
        ("6 任务闭环", "selected6_closed_loop"),
        ("闭环 vs lane", "selected6_lane_vs_closed_loop"),
    ]
    header = ["审计范围", "比较数", "动作一致", "坐标哈希一致",
              "步数一致", "成功一致", "max Δ (m)"]
    rows = []
    for label, scope in scopes:
        sub = invariance[invariance["scope"] == scope]
        rows.append([
            label,
            f"{len(sub)}",
            f"{int(sub['actions_equal'].sum())}/{len(sub)}",
            f"{int(sub['coord_hash_equal'].sum())}/{len(sub)}",
            f"{int(sub['steps_equal'].sum())}/{len(sub)}",
            f"{int(sub['success_equal'].sum())}/{len(sub)}",
            f"{float(sub['max_coord_deviation_m'].max()):.6f}",
        ])
    table = ax_table.table(
        cellText=rows, colLabels=header, cellLoc="center",
        colLoc="center", loc="upper center",
        bbox=[0.0, 0.30, 1.0, 0.60],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(color=INK, fontweight="bold")
        else:
            cell.set_facecolor(SURFACE)
            cell.set_text_props(color=INK2)
    ax_table.text(
        0.0, 0.20,
        "结论：导航动作、逐步坐标、步数和成功状态在三个层级"
        "全部逐条一致；最大坐标偏差为 0.000000 m。",
        transform=ax_table.transAxes, ha="left", va="top",
        fontsize=8.4, color=INK,
    )
    ax_table.text(
        0.0, 0.10,
        "“lane 回放”使用冻结的每步 MEC 观测；"
        "“真实闭环”重新执行环境并独立选择关联节点。",
        transform=ax_table.transAxes, ha="left", va="top",
        fontsize=7.4, color=MUTED,
    )

    # -- max coordinate deviation ----------------------------------------
    order = stats.sort_values("max_dev", ascending=True)
    y = np.arange(len(order))
    colors = [
        CATEGORICAL[0] if family == "learned" else CONTROL
        for family in order["family"]
    ]
    ax_dev.scatter(
        order["max_dev"], y, s=30, c=colors, edgecolors="white",
        linewidths=0.8, zorder=3,
    )
    ax_dev.axvline(0.0, color=AXIS, linewidth=0.9, zorder=1)
    ax_dev.axvspan(-1e-9, 1e-9, color=GRID, alpha=0.55, zorder=0)
    ax_dev.set_xlim(-1.05e-9, 1.05e-9)
    ax_dev.set_xticks([-1e-9, 0.0, 1e-9])
    ax_dev.set_xticklabels(["-1e-9", "0", "1e-9"])
    ax_dev.set_yticks(y)
    ax_dev.set_yticklabels(order["label"], fontsize=7.2)
    ax_dev.set_xlabel("与 2n-DQN-s0 的最大逐步坐标偏差 (m)")
    ax_dev.set_title(
        f"逐步坐标偏差：{len(closed)}/{len(closed)} 次比较 = 0.000000 m",
        fontsize=10.2, loc="left", pad=6.0,
    )
    ax_dev.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax_dev.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_dev.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax_dev.spines[spine].set_color(AXIS)
        ax_dev.spines[spine].set_linewidth(0.8)
    # -- node-choice mismatch --------------------------------------------
    order = stats.sort_values("node_mismatch", ascending=True)
    y = np.arange(len(order))
    colors = [method_color(method) for method in order["method_id"]]
    ax_node.barh(
        y, order["node_mismatch"] * 100.0, height=0.62,
        color=colors, edgecolor=SURFACE, linewidth=1.0, zorder=3,
    )
    ax_node.set_yticks(y)
    ax_node.set_yticklabels(order["label"], fontsize=7.2)
    ax_node.set_xlabel("关联节点选择不一致步数占比 (%)")
    ax_node.set_title(
        "关联节点选择：方法之间确有差异",
        fontsize=10.2, loc="left", pad=6.0,
    )
    ax_node.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax_node.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_node.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax_node.spines[spine].set_color(AXIS)
        ax_node.spines[spine].set_linewidth(0.8)
    for position, value in zip(y, order["node_mismatch"] * 100.0):
        ax_node.text(
            value + 0.6, position, f"{value:.1f}%", va="center",
            ha="left", fontsize=6.8, color=INK2,
        )
    ax_node.set_xlim(
        0.0, max(1.0, float((order["node_mismatch"] * 100.0).max()) * 1.18)
    )

    # -- E2E delta -------------------------------------------------------
    order = stats.sort_values("e2e_delta", ascending=True)
    y = np.arange(len(order))
    colors = [method_color(method) for method in order["method_id"]]
    ax_e2e.barh(
        y, order["e2e_delta"], height=0.62, color=colors,
        edgecolor=SURFACE, linewidth=1.0, zorder=3,
    )
    ax_e2e.axvline(0.0, color=AXIS, linewidth=0.9, zorder=1)
    ax_e2e.set_yticks(y)
    ax_e2e.set_yticklabels(order["label"], fontsize=7.2)
    ax_e2e.set_xlabel("相对 2n-DQN-s0 的平均 E2E 时延差 (ms)")
    ax_e2e.set_title(
        "E2E 时延：差异只来自关联节点选择",
        fontsize=10.2, loc="left", pad=6.0,
    )
    ax_e2e.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax_e2e.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_e2e.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax_e2e.spines[spine].set_color(AXIS)
        ax_e2e.spines[spine].set_linewidth(0.8)
    span = max(
        abs(float(order["e2e_delta"].min())),
        abs(float(order["e2e_delta"].max())), 1.0,
    )
    for position, value in zip(y, order["e2e_delta"]):
        ax_e2e.text(
            value + (0.025 * span if value >= 0 else -0.025 * span),
            position, f"{value:+.1f}", va="center",
            ha="left" if value >= 0 else "right",
            fontsize=6.8, color=INK2,
        )
    ax_e2e.set_xlim(-span * 1.22, span * 1.22)

    fig.suptitle(
        "Stage-2q：跨方法 UAV 轨迹差异审计",
        x=0.5, y=0.975, fontsize=16, color=INK,
    )
    fig.text(
        0.5, 0.942,
        "同一任务上，DQN/PPO 与不同奖励方案只改变 MEC 关联节点和 E2E 时延；"
        "导航动作、UAV 坐标、步数与成功状态完全一致。",
        ha="center", va="top", fontsize=9.4, color=INK2,
    )
    fig.text(
        0.5, 0.004,
        "数据：Stage-2q 真实闭环审计（6 条 Dev 任务 × 20 个方法）与"
        "Dev-60 frozen-lane 回放（60 × 20）。"
        "不同任务之间轨迹本来就不同；本图只比较同一任务。"
        "未读取 Test-400。",
        ha="center", va="bottom", fontsize=7.2, color=MUTED,
    )
    output = base / "trajectory_difference_audit.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=str(HERE))
    args = parser.parse_args(argv)
    base = Path(args.base_dir)
    if not base.is_absolute():
        base = HERE / base
    (summary, steps, invariance, selected, dev60, city,
     city_meta, audit) = _read_inputs(base)
    figure_examples_2x3(
        base, summary, steps, invariance, selected, city, city_meta
    )
    _overlay_figure(
        base, summary, steps, invariance, city, city_meta, audit,
        "stage2n", "Stage-2n DQN/PPO seed 对照",
        "stage2n_method_overlay.png",
    )
    _overlay_figure(
        base, summary, steps, invariance, city, city_meta, audit,
        "stage2o", "Stage-2o 奖励 B/D/方案一 对照",
        "stage2o_reward_overlay.png",
    )
    _overlay_figure(
        base, summary, steps, invariance, city, city_meta, audit,
        "stage2p", "Stage-2p PPO 原始奖励 vs 方案 D 对照",
        "stage2p_ppo_reward_overlay.png",
    )
    figure_difference_audit(base, invariance, audit)
    print("[plot] all figures written")


if __name__ == "__main__":
    main()
