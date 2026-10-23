"""300-dpi figures for Stage-2o (Chinese labels, CSV sidecars).

The palette is the validated three-slot categorical instance from the
data-visualisation reference: blue / orange / aqua.  Scheme identity is carried
by colour *and* line style (B dashed, D solid, one dotted), so the figures stay
readable for colour-vision deficiency and in grayscale print.  Every figure
also has a CSV sidecar under ``evidence/figure_data/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SCHEME_COLORS = {"B": "#2a78d6", "D": "#eb6834", "one": "#1baf7a"}
SCHEME_LINESTYLES = {"B": "--", "D": "-", "one": ":"}
SCHEME_MARKERS = {"B": "o", "D": "s", "one": "^"}
SCHEME_LABELS = {"B": "方案B", "D": "方案D", "one": "方案一"}
ALGO_LABELS = {"dqn": "DQN", "ppo": "PPO"}
SCHEME_ORDER = ("B", "D", "one")


def setup_style():
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK SC", "Noto Sans CJK JP",
                        "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _finish(ax):
    ax.grid(True, axis="both", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(0.8)


def _save(fig, path, data, data_dir):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    if data is not None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(data_dir / (path.stem + ".csv"), index=False)


def _scheme_handle(scheme):
    return Line2D(
        [0], [0], color=SCHEME_COLORS[scheme],
        linestyle=SCHEME_LINESTYLES[scheme], linewidth=2,
        label=SCHEME_LABELS[scheme],
    )


def _xy(run, column):
    """Return numeric numpy arrays; matplotlib/pandas NaN handling is fragile."""
    x = pd.to_numeric(run["episode"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(run[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _legend(ax, loc="upper left", handles=None, fontsize=8):
    """Legend with a surface-coloured backing so noisy lines stay readable."""
    kwargs = dict(
        loc=loc, fontsize=fontsize, frameon=True, facecolor=SURFACE,
        edgecolor="none", framealpha=0.92,
    )
    if handles is not None:
        ax.legend(handles=handles, **kwargs)
    else:
        ax.legend(**kwargs)


def plot_ma100_reward(training, evidence_dir, aligned=False):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    for ax, algorithm in zip(axes, ("dqn", "ppo")):
        subset = training[training["algorithm"] == algorithm]
        for scheme in SCHEME_ORDER:
            run = subset[subset["scheme"] == scheme].sort_values("episode")
            if run.empty:
                continue
            y = run["ma100_reward"].to_numpy(dtype=float)
            y = y + 1.0 if (aligned and scheme == "B") else y
            label = SCHEME_LABELS[scheme]
            if aligned and scheme == "B":
                label += "（+1 对齐）"
            ax.plot(
                run["episode"], y, color=SCHEME_COLORS[scheme],
                linestyle=SCHEME_LINESTYLES[scheme], linewidth=2,
                label=label, zorder=3,
            )
        ax.set_title(f"{ALGO_LABELS[algorithm]}：MA100 奖励")
        ax.set_xlabel("Episode")
        ax.set_ylabel("最近 100 Episode 平均奖励")
        ax.legend(handles=[
            _scheme_handle(scheme) for scheme in SCHEME_ORDER
        ], loc=("upper left" if aligned else "center right"))
        _finish(ax)
    note = (
        "B: r_t = -(mean_t-mean_{t-1})/tau, R=-mean/tau   |   "
        "D: r_t=(1-e2e_t/tau)/H + 终止修正, R=1-mean/tau   |   "
        "方案一: r_T=J, R=1-mean/tau"
    )
    if aligned:
        note += "\nB 曲线已 +1；D 与方案一累计回报相同，差异只来自逐步分配。"
    else:
        note += "\n注意：B 的 Episode 累计回报比 D/方案一低固定常数 1，曲线高度不可直接比较。"
    fig.text(0.5, -0.02, note, ha="center", va="top",
             fontsize=8, color=INK_SECONDARY)
    name = ("fig02_ma100_reward_aligned.png" if aligned
            else "fig01_ma100_reward.png")
    data = training[["scheme", "algorithm", "episode", "ma100_reward"]].copy()
    if aligned:
        data.loc[data["scheme"] == "B", "ma100_reward"] += 1.0
    _save(fig, evidence_dir / name, data, evidence_dir / "figure_data")


def plot_ma100_e2e(training, evidence_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    for ax, algorithm in zip(axes, ("dqn", "ppo")):
        subset = training[training["algorithm"] == algorithm]
        for scheme in SCHEME_ORDER:
            run = subset[subset["scheme"] == scheme].sort_values("episode")
            if run.empty:
                continue
            x, y = _xy(run, "ma100_e2e")
            ax.plot(
                x, y,
                color=SCHEME_COLORS[scheme],
                linestyle=SCHEME_LINESTYLES[scheme], linewidth=2,
                label=SCHEME_LABELS[scheme], zorder=3,
            )
        ax.axhline(100.0, color=BASELINE, linewidth=1.0, zorder=1)
        ax.text(
            0.04, 100.0, "tau = 100 ms", va="top", ha="left",
            transform=ax.get_yaxis_transform(),
            fontsize=8, color=INK_MUTED,
        )
        ax.set_title(f"{ALGO_LABELS[algorithm]}：MA100 平均 E2E 时延")
        ax.set_xlabel("Episode")
        ax.set_ylabel("E2E 时延（ms，越低越好）")
        ax.legend(handles=[
            _scheme_handle(scheme) for scheme in SCHEME_ORDER
        ], loc="upper right")
        _finish(ax)
    fig.text(
        0.5, -0.02,
        "主要判据：实际 E2E 时延（ms）。三个方案的奖励尺度不同，"
        "不能以奖励曲线绝对高度代替时延比较。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = training[["scheme", "algorithm", "episode", "ma100_e2e"]]
    _save(fig, evidence_dir / "fig03_ma100_e2e.png", data,
          evidence_dir / "figure_data")


def plot_dev_e2e(validation, evidence_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    for ax, algorithm in zip(axes, ("dqn", "ppo")):
        subset = validation[validation["algorithm"] == algorithm]
        for scheme in SCHEME_ORDER:
            run = subset[subset["scheme"] == scheme].sort_values("episode")
            if run.empty:
                continue
            x, y = _xy(run, "E2E_ms")
            ax.plot(
                x, y,
                color=SCHEME_COLORS[scheme],
                linestyle=SCHEME_LINESTYLES[scheme], linewidth=1.8,
                marker=SCHEME_MARKERS[scheme], markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=SCHEME_LABELS[scheme], zorder=3,
            )
            best = run.loc[run["E2E_ms"].idxmin()]
            ax.scatter(
                [best["episode"]], [best["E2E_ms"]], s=70,
                facecolor=SCHEME_COLORS[scheme], edgecolor=SURFACE,
                linewidth=1.5, zorder=4,
            )
        ax.set_title(f"{ALGO_LABELS[algorithm]}：Dev-60 平均 E2E")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Dev-60 平均 E2E（ms）")
        ax.legend(handles=[
            _scheme_handle(scheme) for scheme in SCHEME_ORDER
        ], loc="upper right")
        _finish(ax)
    fig.text(
        0.5, -0.02,
        "圆点为每 200 Episode 的 Dev-60 验证；大点为该 run 的最佳 Dev E2E。"
        "检查点按最低 Dev E2E 选择。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = validation[[
        "scheme", "algorithm", "episode", "E2E_ms", "P95_E2E_ms",
        "SR", "SPL", "nDTW",
    ]]
    _save(fig, evidence_dir / "fig04_dev_e2e.png", data,
          evidence_dir / "figure_data")


def plot_rolling_std(training, evidence_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True)
    for column, algorithm in enumerate(("dqn", "ppo")):
        subset = training[training["algorithm"] == algorithm]
        for scheme in SCHEME_ORDER:
            run = subset[subset["scheme"] == scheme].sort_values("episode")
            if run.empty:
                continue
            x, y = _xy(run, "std100_reward")
            axes[0, column].plot(
                x, y,
                color=SCHEME_COLORS[scheme],
                linestyle=SCHEME_LINESTYLES[scheme], linewidth=1.8,
                label=SCHEME_LABELS[scheme],
            )
            x, y = _xy(run, "std100_e2e")
            axes[1, column].plot(
                x, y,
                color=SCHEME_COLORS[scheme],
                linestyle=SCHEME_LINESTYLES[scheme], linewidth=1.8,
                label=SCHEME_LABELS[scheme],
            )
        axes[0, column].set_title(
            f"{ALGO_LABELS[algorithm]}：奖励滚动标准差（窗口 100）"
        )
        axes[0, column].set_ylabel("σ(MA100 奖励)")
        axes[1, column].set_title(
            f"{ALGO_LABELS[algorithm]}：E2E 滚动标准差（窗口 100）"
        )
        axes[1, column].set_ylabel("σ(E2E) / ms")
        axes[1, column].set_xlabel("Episode")
        for row in (0, 1):
            _legend(axes[row, column], loc="upper right", handles=[
                _scheme_handle(scheme) for scheme in SCHEME_ORDER
            ])
            _finish(axes[row, column])
    fig.text(
        0.5, -0.01,
        "滚动标准差用于判断训练震荡是否减小；奖励标准差跨方案不可直接比较尺度，"
        "E2E 标准差可直接比较。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = training[[
        "scheme", "algorithm", "episode", "std100_reward", "std100_e2e",
    ]]
    _save(fig, evidence_dir / "fig05_rolling_std.png", data,
          evidence_dir / "figure_data")


def plot_dqn_diagnostics(training, evidence_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True)
    subset = training[training["algorithm"] == "dqn"]
    panels = (
        ("q_mean", "Q mean（在线网络，可用动作）", axes[0, 0]),
        ("q_max", "Q max（在线网络，可用动作）", axes[0, 1]),
        ("target_mean", "TD Target mean", axes[1, 0]),
        ("td_error_abs_mean", "TD Loss（|Q - target| 均值）", axes[1, 1]),
    )
    for key, title, ax in panels:
        for scheme in SCHEME_ORDER:
            run = subset[subset["scheme"] == scheme].sort_values("episode")
            if run.empty or key not in run:
                continue
            x, y = _xy(run, key)
            ax.plot(
                x, y,
                color=SCHEME_COLORS[scheme],
                linestyle=SCHEME_LINESTYLES[scheme], linewidth=1.8,
                label=SCHEME_LABELS[scheme],
            )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        _legend(ax, loc="upper left", handles=[
            _scheme_handle(scheme) for scheme in SCHEME_ORDER
        ])
        _finish(ax)
    fig.text(
        0.5, -0.01,
        "DQN 诊断：Q mean/Q max 与 Target 的持续分离提示 Q 值漂移或过估计；"
        "TD Loss 用于观察拟合稳定性。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = subset[[
        "scheme", "episode", "q_mean", "q_max", "target_mean",
        "td_error_abs_mean", "loss",
    ]]
    _save(fig, evidence_dir / "fig06_dqn_diagnostics.png", data,
          evidence_dir / "figure_data")


def plot_ppo_diagnostics(training, evidence_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True)
    subset = training[training["algorithm"] == "ppo"]
    panels = (
        ("update_value_loss", "Value Loss", axes[0, 0]),
        ("explained_variance", "Explained Variance", axes[0, 1]),
        ("policy_entropy", "Policy Entropy", axes[1, 0]),
        ("approx_kl", "Approx KL", axes[1, 1]),
    )
    for key, title, ax in panels:
        for scheme in SCHEME_ORDER:
            run = subset[subset["scheme"] == scheme].sort_values("episode")
            if run.empty or key not in run:
                continue
            x, y = _xy(run, key)
            ax.plot(
                x, y,
                color=SCHEME_COLORS[scheme],
                linestyle=SCHEME_LINESTYLES[scheme], linewidth=1.8,
                label=SCHEME_LABELS[scheme],
            )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        _legend(ax, loc="upper left", handles=[
            _scheme_handle(scheme) for scheme in SCHEME_ORDER
        ])
        _finish(ax)
    fig.text(
        0.5, -0.01,
        "PPO 诊断：Value Loss 与 Explained Variance 反映价值估计质量；"
        "Entropy/KL 反映策略探索与更新幅度。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = subset[[
        "scheme", "episode", "update_value_loss", "explained_variance",
        "policy_entropy", "approx_kl", "value_mean",
    ]]
    _save(fig, evidence_dir / "fig07_ppo_diagnostics.png", data,
          evidence_dir / "figure_data")


def plot_dev_summary(summary, evidence_dir):
    frame = pd.DataFrame(summary)
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), sharey=True)
    x = np.arange(len(SCHEME_ORDER))
    width = 0.34
    for ax, algorithm in zip(axes, ("dqn", "ppo")):
        subset = frame[frame["algorithm"] == algorithm]
        best = [
            float(subset[subset["scheme"] == scheme]["best_dev_e2e_ms"]
                  .iloc[0]) if not subset[
                subset["scheme"] == scheme].empty else np.nan
            for scheme in SCHEME_ORDER
        ]
        final = [
            float(subset[subset["scheme"] == scheme]["final_dev_e2e_ms"]
                  .iloc[0]) if not subset[
                subset["scheme"] == scheme].empty else np.nan
            for scheme in SCHEME_ORDER
        ]
        colors = [SCHEME_COLORS[scheme] for scheme in SCHEME_ORDER]
        bars_best = ax.bar(
            x - width / 2 - 0.01, best, width=width, color=colors,
            edgecolor=SURFACE, linewidth=1.5, label="最佳 Dev E2E",
        )
        bars_final = ax.bar(
            x + width / 2 + 0.01, final, width=width, color=colors,
            alpha=0.45, edgecolor=SURFACE, linewidth=1.5, hatch="//",
            label="最终 Dev E2E",
        )
        for bars in (bars_best, bars_final):
            for bar in bars:
                height = bar.get_height()
                if np.isfinite(height):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, height,
                        f"{height:.1f}", ha="center", va="bottom",
                        fontsize=8, color=INK_SECONDARY,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([SCHEME_LABELS[s] for s in SCHEME_ORDER])
        ax.set_title(f"{ALGO_LABELS[algorithm]}：Dev-60 E2E 汇总")
        ax.set_ylabel("E2E 时延（ms，越低越好）")
        _legend(ax, loc="upper left", handles=[
            Patch(facecolor="#52514e", label="最佳 Dev E2E"),
            Patch(facecolor="white", edgecolor="#52514e", hatch="//",
                  label="最终 Dev E2E"),
        ])
        _finish(ax)
    fig.text(
        0.5, -0.03,
        "实色为最佳检查点，斜线为最后一个验证点。检查点按最佳 Dev E2E 选择，"
        "但最终值用于判断训练是否退化。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = frame[[
        "scheme", "algorithm", "best_dev_e2e_ms", "best_dev_episode",
        "final_dev_e2e_ms",
    ]]
    _save(fig, evidence_dir / "fig08_dev_summary.png", data,
          evidence_dir / "figure_data")


def plot_terminal_diagnostics(training, evidence_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    ax = axes[0]
    for scheme in SCHEME_ORDER:
        run = training[training["scheme"] == scheme].sort_values("episode")
        if run.empty:
            continue
        correction = run["terminal_correction"].rolling(
            100, min_periods=100
        ).mean()
        terminal = run["terminal_reward"].rolling(
            100, min_periods=100
        ).mean()
        # For D plot the correction; for B/one plot the terminal reward.
        y = correction if scheme == "D" else terminal
        label = "方案D：终止修正" if scheme == "D" else (
            "方案B：末步奖励" if scheme == "B" else "方案一：终止奖励 J"
        )
        x = pd.to_numeric(
            run["episode"], errors="coerce"
        ).to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.plot(
            x[mask], y[mask], color=SCHEME_COLORS[scheme],
            linestyle=SCHEME_LINESTYLES[scheme], linewidth=1.8,
            label=label,
        )
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title("终止项大小（MA100，绝对值/有符号）")
    ax.set_xlabel("Episode")
    ax.set_ylabel("终止修正 / 终止奖励")
    _legend(ax, loc="upper left")
    _finish(ax)

    ax = axes[1]
    run_d = training[training["scheme"] == "D"]
    if not run_d.empty:
        values = run_d["terminal_share"].dropna().to_numpy(dtype=float)
        values = np.clip(values, 0.0, 1.2)
        ax.hist(
            values, bins=40, density=True, histtype="step",
            color=SCHEME_COLORS["D"], linewidth=2,
            label="方案D：终止转移占 Episode 回报比例",
        )
    ax.axvline(
        1.0, color=SCHEME_COLORS["one"],
        linestyle=SCHEME_LINESTYLES["one"], linewidth=2,
        label="方案一：终止转移占比恒为 1.0",
    )
    ax.set_xlim(0.0, 1.1)
    ax.set_title("终止转移占 Episode 回报的比例")
    ax.set_xlabel("terminal_reward / episode_return")
    ax.set_ylabel("密度")
    _legend(ax, loc="upper left")
    _finish(ax)
    fig.text(
        0.5, -0.03,
        "方案D 的终止修正约占总回报的 1-T/H（T 为实际步数）；"
        "方案一全部回报集中在最后一步。该图用于判断末步反馈是否过大。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = training[[
        "scheme", "algorithm", "episode", "steps", "terminal_reward",
        "terminal_correction", "terminal_share", "nonterminal_abs_max",
        "episode_return",
    ]]
    _save(fig, evidence_dir / "fig09_terminal_diagnostics.png", data,
          evidence_dir / "figure_data")


def make_all(training, validation, summary, evidence_dir):
    setup_style()
    evidence_dir = Path(evidence_dir)
    plot_ma100_reward(training, evidence_dir, aligned=False)
    plot_ma100_reward(training, evidence_dir, aligned=True)
    plot_ma100_e2e(training, evidence_dir)
    if not validation.empty:
        plot_dev_e2e(validation, evidence_dir)
    plot_rolling_std(training, evidence_dir)
    plot_dqn_diagnostics(training, evidence_dir)
    plot_ppo_diagnostics(training, evidence_dir)
    plot_dev_summary(summary, evidence_dir)
    plot_terminal_diagnostics(training, evidence_dir)
    return sorted(path.name for path in evidence_dir.glob("fig*.png"))


__all__ = ["make_all", "setup_style"]
