"""300-dpi Stage-2p figures: baseline vs scheme D, three seeds.

Colours follow the validated two-slot categorical palette (blue/orange).
Thick lines are the seed mean; thin faint lines are individual seeds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from v83_spatial_stage2.stage2o.plot import (
    BASELINE,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    _finish,
    _legend,
    _save,
    setup_style,
)

SCHEME_COLORS = {"baseline": "#2a78d6", "D": "#eb6834"}
SCHEME_LABELS = {"baseline": "原始奖励", "D": "方案D"}
SCHEME_ORDER = ("baseline", "D")
SEED_LINESTYLES = {0: "-", 1: "--", 2: ":"}
SEEDS = (0, 1, 2)
GREEDY_DEV_E2E = 49.18036816866046


def _seed_mean(frame, value_column):
    grouped = frame.groupby(["scheme", "episode"], as_index=False)[
        value_column
    ].mean()
    return grouped


def _plot_scheme_seeds(ax, frame, value_column, align_baseline=False,
                       alpha=0.28, linewidth=1.0):
    for scheme in SCHEME_ORDER:
        color = SCHEME_COLORS[scheme]
        for seed in SEEDS:
            run = frame[(frame["scheme"] == scheme)
                        & (frame["seed"] == seed)].sort_values("episode")
            if run.empty:
                continue
            x = run["episode"].to_numpy(dtype=float)
            y = run[value_column].to_numpy(dtype=float)
            if align_baseline and scheme == "baseline":
                y = 4.0 * y + 1.0
            mask = np.isfinite(x) & np.isfinite(y)
            ax.plot(
                x[mask], y[mask],
                color=color, linestyle=SEED_LINESTYLES[seed],
                linewidth=linewidth, alpha=alpha, zorder=2,
            )


def _plot_scheme_mean(ax, frame, value_column, align_baseline=False,
                      linewidth=2.4):
    mean = _seed_mean(frame, value_column)
    for scheme in SCHEME_ORDER:
        run = mean[mean["scheme"] == scheme].sort_values("episode")
        if run.empty:
            continue
        x = run["episode"].to_numpy(dtype=float)
        y = run[value_column].to_numpy(dtype=float)
        if align_baseline and scheme == "baseline":
            y = 4.0 * y + 1.0
        mask = np.isfinite(x) & np.isfinite(y)
        ax.plot(
            x[mask], y[mask],
            color=SCHEME_COLORS[scheme], linewidth=linewidth,
            solid_capstyle="round", zorder=3,
            label=SCHEME_LABELS[scheme],
        )


def _scheme_handles():
    return [
        Line2D([0], [0], color=SCHEME_COLORS[scheme], linewidth=2.4,
               label=SCHEME_LABELS[scheme])
        for scheme in SCHEME_ORDER
    ]


def plot_ma100_reward(training, evidence_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    for ax, aligned in zip(axes, (False, True)):
        _plot_scheme_seeds(
            ax, training, "ma100_reward", align_baseline=aligned
        )
        _plot_scheme_mean(
            ax, training, "ma100_reward", align_baseline=aligned
        )
        ax.set_title(
            "MA100 奖励（4×Baseline+1 对齐尺度）" if aligned
            else "MA100 奖励（原始尺度）"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("最近 100 Episode 平均奖励")
        _legend(ax, loc="lower right", handles=_scheme_handles())
        _finish(ax)
    fig.text(
        0.5, -0.02,
        "原始奖励 = -0.25*(mean_t-mean_{t-1})/100，R=-0.25*mean/tau；"
        "方案D R=1-mean/tau。对齐用 4*R_baseline+1=R_D；"
        "粗线为三种子均值，细线为 seed 0/1/2。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = training[["scheme", "seed", "episode", "ma100_reward"]].copy()
    data["aligned_ma100_reward"] = data.apply(
        lambda row: 4.0 * row["ma100_reward"] + 1.0
        if row["scheme"] == "baseline" else row["ma100_reward"],
        axis=1,
    )
    _save(fig, evidence_dir / "fig01_ma100_reward.png", data,
          evidence_dir / "figure_data")


def plot_ma100_e2e(training, evidence_dir):
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    _plot_scheme_seeds(ax, training, "ma100_e2e")
    _plot_scheme_mean(ax, training, "ma100_e2e")
    ax.axhline(100.0, color=BASELINE, linewidth=1.0, zorder=1)
    ax.text(
        0.04, 100.0, "tau = 100 ms", va="top", ha="left",
        transform=ax.get_yaxis_transform(),
        fontsize=8, color=INK_MUTED,
    )
    ax.set_title("MA100 平均 E2E 时延（三种子）")
    ax.set_xlabel("Episode")
    ax.set_ylabel("E2E 时延（ms，越低越好）")
    _legend(ax, loc="upper right", handles=_scheme_handles())
    _finish(ax)
    fig.text(
        0.5, -0.03,
        "主要判据：实际 E2E 时延。粗线为三种子均值，细线为 seed 0/1/2。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = training[["scheme", "seed", "episode", "ma100_e2e"]]
    _save(fig, evidence_dir / "fig02_ma100_e2e.png", data,
          evidence_dir / "figure_data")


def plot_dev_e2e(validation, evidence_dir):
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    for scheme in SCHEME_ORDER:
        color = SCHEME_COLORS[scheme]
        for seed in SEEDS:
            run = validation[
                (validation["scheme"] == scheme)
                & (validation["seed"] == seed)
            ].sort_values("episode")
            if run.empty:
                continue
            ax.plot(
                run["episode"], run["E2E_ms"], color=color,
                linestyle=SEED_LINESTYLES[seed], linewidth=1.0,
                alpha=0.30, marker="o", markersize=3,
                markeredgecolor=SURFACE, markeredgewidth=0.8,
            )
            best = run.loc[run["E2E_ms"].idxmin()]
            ax.scatter(
                [best["episode"]], [best["E2E_ms"]], s=55,
                facecolor=color, edgecolor=SURFACE, linewidth=1.2,
                zorder=4,
            )
    mean = validation.groupby(["scheme", "episode"], as_index=False)[
        "E2E_ms"
    ].mean()
    for scheme in SCHEME_ORDER:
        run = mean[mean["scheme"] == scheme].sort_values("episode")
        if run.empty:
            continue
        ax.plot(
            run["episode"], run["E2E_ms"], color=SCHEME_COLORS[scheme],
            linewidth=2.2, marker="o", markersize=4,
            markeredgecolor=SURFACE, markeredgewidth=1.0,
            label=SCHEME_LABELS[scheme], zorder=3,
        )
    ax.axhline(
        GREEDY_DEV_E2E, color=BASELINE, linewidth=1.0, linestyle="--",
        zorder=1,
    )
    ax.text(
        0.01, GREEDY_DEV_E2E, " Greedy 49.18 ms", va="bottom",
        ha="left", fontsize=8, color=INK_MUTED,
        transform=ax.get_yaxis_transform(),
    )
    ax.set_title("Dev-60 平均 E2E（三种子）")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Dev-60 平均 E2E（ms）")
    _legend(ax, loc="upper right", handles=_scheme_handles())
    _finish(ax)
    fig.text(
        0.5, -0.03,
        "圆点为每 200 Episode 的 Dev-60 验证；大点为该 seed 的最佳 Dev；"
        "粗线为三种子均值，细线为 seed 0/1/2。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = validation[[
        "scheme", "seed", "episode", "E2E_ms", "P95_E2E_ms",
        "delay_violation_rate", "SR", "SPL", "nDTW",
    ]]
    _save(fig, evidence_dir / "fig03_dev_e2e.png", data,
          evidence_dir / "figure_data")


def plot_rolling_std(training, evidence_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    aligned = training.copy()
    mask = aligned["scheme"] == "baseline"
    # Transform per-episode return before the rolling window is already
    # stored as ma100_reward/std100_reward.  Recompute aligned rolling std
    # from the transformed episode return so the two schemes share a scale.
    if "episode_return" in aligned.columns:
        aligned.loc[mask, "aligned_return"] = (
            4.0 * aligned.loc[mask, "episode_return"] + 1.0
        )
        aligned.loc[~mask, "aligned_return"] = aligned.loc[
            ~mask, "episode_return"
        ]
        aligned = aligned.sort_values(["scheme", "seed", "episode"])
        aligned["aligned_std100"] = aligned.groupby(
            ["scheme", "seed"]
        )["aligned_return"].transform(
            lambda series: series.rolling(100, min_periods=100).std(ddof=0)
        )
        reward_column = "aligned_std100"
    else:
        reward_column = "std100_reward"
    for ax, column, title, ylabel in (
        (axes[0], reward_column,
         "奖励滚动标准差（对齐后，窗口 100）", "σ(MA100 奖励)"),
        (axes[1], "std100_e2e",
         "E2E 滚动标准差（窗口 100）", "σ(E2E) / ms"),
    ):
        _plot_scheme_seeds(ax, training if column != "aligned_std100"
                           else aligned, column)
        _plot_scheme_mean(ax, training if column != "aligned_std100"
                          else aligned, column)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        _legend(ax, loc="upper right", handles=_scheme_handles())
        _finish(ax)
    fig.text(
        0.5, -0.02,
        "奖励标准差在对齐尺度上比较（4*R_baseline+1）；E2E 标准差可直接比较。"
        "粗线为三种子均值，细线为 seed 0/1/2。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = training[[
        "scheme", "seed", "episode", "std100_reward", "std100_e2e",
    ]]
    _save(fig, evidence_dir / "fig04_rolling_std.png", data,
          evidence_dir / "figure_data")


def plot_ppo_diagnostics(training, evidence_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True)
    panels = (
        ("update_value_loss", "Value Loss", axes[0, 0]),
        ("explained_variance", "Explained Variance", axes[0, 1]),
        ("policy_entropy", "Policy Entropy", axes[1, 0]),
        ("approx_kl", "Approx KL", axes[1, 1]),
    )
    for key, title, ax in panels:
        _plot_scheme_seeds(ax, training, key)
        _plot_scheme_mean(ax, training, key)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        _legend(ax, loc="upper left", handles=_scheme_handles())
        _finish(ax)
    fig.text(
        0.5, -0.01,
        "PPO 诊断：Value Loss/EV 反映价值估计；Entropy/KL 反映探索与更新幅度。"
        "粗线为三种子均值，细线为 seed 0/1/2。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = training[[
        "scheme", "seed", "episode", "update_value_loss",
        "explained_variance", "policy_entropy", "approx_kl", "value_mean",
    ]]
    _save(fig, evidence_dir / "fig05_ppo_diagnostics.png", data,
          evidence_dir / "figure_data")


def plot_seed_summary(summary, evidence_dir):
    frame = pd.DataFrame(summary)
    if frame.empty:
        return
    metrics = ("best_dev_e2e_ms", "final_dev_e2e_ms")
    titles = ("最佳 Dev E2E", "最终 Dev E2E")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), sharey=True)
    for ax, metric, title in zip(axes, metrics, titles):
        x = np.arange(len(SEEDS))
        width = 0.36
        for offset, scheme in ((-width / 2 - 0.01, "baseline"),
                               (width / 2 + 0.01, "D")):
            values = []
            for seed in SEEDS:
                row = frame[
                    (frame["scheme"] == scheme)
                    & (frame["seed"] == seed)
                ]
                values.append(
                    float(row[metric].iloc[0]) if not row.empty else np.nan
                )
            bars = ax.bar(
                x + offset, values, width=width,
                color=SCHEME_COLORS[scheme], edgecolor=SURFACE,
                linewidth=1.5, label=SCHEME_LABELS[scheme],
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, value,
                        f"{value:.1f}", ha="center", va="bottom",
                        fontsize=8, color=INK_SECONDARY,
                    )
        means = [float(frame[frame["scheme"] == scheme][metric].mean())
                 for scheme in SCHEME_ORDER]
        ax.axhline(means[0], color=SCHEME_COLORS["baseline"],
                   linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axhline(means[1], color=SCHEME_COLORS["D"],
                   linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {seed}" for seed in SEEDS])
        ax.set_title(title)
        ax.set_ylabel("E2E 时延（ms，越低越好）")
        if ax is axes[0]:
            _legend(ax, loc="upper right", handles=_scheme_handles())
        _finish(ax)
    fig.text(
        0.5, -0.03,
        "柱为逐 seed 的最优/最终 Dev；虚线为三种子均值。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = frame[[
        "scheme", "seed", "best_dev_e2e_ms", "best_dev_episode",
        "final_dev_e2e_ms",
    ]]
    _save(fig, evidence_dir / "fig06_seed_summary.png", data,
          evidence_dir / "figure_data")


def plot_paired_differences(paired, evidence_dir):
    frame = pd.DataFrame(paired)
    if frame.empty:
        return
    metrics = (
        ("delta_best_dev_e2e_ms", "Δ最佳 Dev E2E"),
        ("delta_final_dev_e2e_ms", "Δ最终 Dev E2E"),
        ("delta_auc_ma100_e2e", "ΔMA100 AUC"),
        ("delta_last500_e2e_std", "Δ末500 E2E σ"),
    )
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    x = np.arange(len(metrics))
    markers = {0: "o", 1: "s", 2: "^"}
    for index, (key, _) in enumerate(metrics):
        values = frame[key].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        for seed, value in zip(frame["seed"], values):
            if not np.isfinite(value):
                continue
            ax.scatter(
                [index], [value], s=48,
                marker=markers.get(int(seed), "o"),
                color=SCHEME_COLORS["D"], edgecolor=SURFACE,
                linewidth=1.2, zorder=3,
            )
        mean = float(np.mean(finite))
        ax.plot(
            [index - 0.22, index + 0.22], [mean, mean],
            color=INK, linewidth=2.0, zorder=4,
        )
    ax.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([title for _, title in metrics], fontsize=8.5)
    ax.set_ylabel("方案D − 原始奖励（ms；负值表示 D 更好）")
    ax.set_title("逐 seed 配对差值（D − Baseline）")
    seed_handles = [
        Line2D([0], [0], marker=markers[seed], linestyle="none",
               color=SCHEME_COLORS["D"], markeredgecolor=SURFACE,
               markersize=8, label=f"seed {seed}")
        for seed in sorted(markers)
    ]
    ax.legend(handles=seed_handles, loc="lower right", fontsize=8)
    _finish(ax)
    fig.text(
        0.5, -0.04,
        "点为 seed 0/1/2 的配对差值，黑线为均值；0 线表示两种奖励相同。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    _save(fig, evidence_dir / "fig07_paired_differences.png", frame,
          evidence_dir / "figure_data")


def make_all(training, validation, summary, paired, evidence_dir):
    setup_style()
    evidence_dir = Path(evidence_dir)
    plot_ma100_reward(training, evidence_dir)
    plot_ma100_e2e(training, evidence_dir)
    if not validation.empty:
        plot_dev_e2e(validation, evidence_dir)
    plot_rolling_std(training, evidence_dir)
    plot_ppo_diagnostics(training, evidence_dir)
    plot_seed_summary(summary, evidence_dir)
    plot_paired_differences(paired, evidence_dir)
    return sorted(path.name for path in evidence_dir.glob("fig*.png"))


__all__ = ["make_all", "setup_style"]
