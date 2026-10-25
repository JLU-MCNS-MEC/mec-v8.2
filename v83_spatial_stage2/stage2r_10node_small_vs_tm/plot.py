"""300-dpi Stage-2r figures (Chinese labels, CSV sidecars).

Encoding:
* deployment identity = colour (Small blue, Transformer-M orange);
* algorithm identity = line style / hatch (DQN solid, PPO dashed);
* individual seeds are faint thin lines; the thick line is the seed mean.

Every figure writes a CSV sidecar under ``figure_data/`` with the exact data
plotted.
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
from matplotlib.ticker import FuncFormatter

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
from v83_spatial_stage2.stage2r_10node_small_vs_tm.common import (
    ALGORITHMS,
    ALGORITHM_LABELS,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    MEC_MODELS,
    MODEL_LABELS,
)

MODEL_COLORS = {"Small": "#2a78d6", "Transformer-M": "#eb6834"}
ALGORITHM_LINESTYLES = {"dqn": "-", "ppo": "--"}
ALGORITHM_MARKERS = {"dqn": "o", "ppo": "s"}
SEED_LINESTYLES = {11: "-", 22: "--", 33: ":"}
SEED_ALPHA = 0.25
METRIC_LABELS = {
    "SR": "SR",
    "OSR": "OSR",
    "nDTW": "nDTW",
    "SPL": "SPL",
    "E2E_ms": "E2E 时延 (ms)",
    "RewardD": "方案 D 奖励",
}
METRIC_TITLES = {
    "SR": "成功率 SR（越高越好）",
    "OSR": "可观测成功率 OSR（越高越好）",
    "nDTW": "nDTW（越高越好）",
    "SPL": "SPL（越高越好）",
    "E2E_ms": "平均 E2E 时延（ms，越低越好）",
    "RewardD": "验证集方案 D 奖励 R=1−mean E2E/100（越高越好）",
}


def _condition(model, algorithm):
    return f"{MODEL_LABELS[model]}|{ALGORITHM_LABELS[algorithm]}"


def _seed_runs(training, model, algorithm):
    subset = training[
        (training["model"] == model)
        & (training["algorithm"] == algorithm)
    ]
    for seed in sorted(subset["seed"].unique()):
        run = subset[subset["seed"] == seed].sort_values("episode")
        yield int(seed), run


def _plot_metric_curves(ax, training, column, model, tau=False):
    color = MODEL_COLORS[model]
    for algorithm in ALGORITHMS:
        linestyle = ALGORITHM_LINESTYLES[algorithm]
        for seed, run in _seed_runs(training, model, algorithm):
            x = run["episode"].to_numpy(dtype=float)
            y = pd.to_numeric(
                run[column], errors="coerce"
            ).to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            ax.plot(
                x[mask], y[mask], color=color, linestyle=linestyle,
                linewidth=0.9, alpha=SEED_ALPHA, zorder=2,
            )
        mean = training[
            (training["model"] == model)
            & (training["algorithm"] == algorithm)
        ].groupby("episode", as_index=False)[column].mean()
        x = mean["episode"].to_numpy(dtype=float)
        y = pd.to_numeric(
            mean[column], errors="coerce"
        ).to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.plot(
            x[mask], y[mask], color=color, linestyle=linestyle,
            linewidth=2.4, zorder=3,
            label=f"{ALGORITHM_LABELS[algorithm]}（3 seed 均值）",
        )
    if tau:
        ax.axhline(100.0, color=BASELINE, linewidth=1.0, zorder=1)
        ax.text(
            0.02, 100.0, "tau = 100 ms", va="top", ha="left",
            transform=ax.get_yaxis_transform(), fontsize=8,
            color=INK_MUTED,
        )


def _condition_handles():
    handles = []
    for algorithm in ALGORITHMS:
        handles.append(Line2D(
            [0], [0], color=INK_SECONDARY,
            linestyle=ALGORITHM_LINESTYLES[algorithm], linewidth=2,
            label=ALGORITHM_LABELS[algorithm],
        ))
    return handles


def plot_training_reward(training, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    for ax, model in zip(axes, MEC_MODELS):
        _plot_metric_curves(ax, training, "ma100_reward", model)
        ax.set_title(f"{MODEL_LABELS[model]}：MA100 奖励（方案 D）")
        ax.set_xlabel("Episode")
        ax.set_ylabel("最近 100 Episode 平均奖励")
        _legend(ax, loc="lower right", handles=_condition_handles())
        _finish(ax)
    fig.text(
        0.5, -0.02,
        "方案 D：u_t=1−e2e_t/100，r_t=u_t/400，末步加 (1−T/400)·J；"
        "R=1−mean E2E/100。粗线为三种子均值，细线为 seed 11/22/33。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = training[[
        "model", "deployment", "algorithm", "seed", "episode",
        "ma100_reward",
    ]]
    _save(fig, out_dir / "fig01_training_ma100_reward.png", data,
          FIGURE_DATA_DIR)


def plot_training_e2e(training, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    for ax, model in zip(axes, MEC_MODELS):
        _plot_metric_curves(
            ax, training, "ma100_e2e", model, tau=True
        )
        ax.set_title(f"{MODEL_LABELS[model]}：MA100 平均 E2E 时延")
        ax.set_xlabel("Episode")
        ax.set_ylabel("E2E 时延（ms，越低越好）")
        _legend(ax, loc="upper right", handles=_condition_handles())
        _finish(ax)
    fig.text(
        0.5, -0.02,
        "主要判据是实际 E2E 时延。粗线为三种子均值，细线为 seed 11/22/33。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = training[[
        "model", "deployment", "algorithm", "seed", "episode", "ma100_e2e",
    ]]
    _save(fig, out_dir / "fig02_training_ma100_e2e.png", data,
          FIGURE_DATA_DIR)


def plot_training_std(training, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    for ax, model in zip(axes, MEC_MODELS):
        _plot_metric_curves(
            ax, training, "std100_e2e", model, tau=False
        )
        ax.set_title(f"{MODEL_LABELS[model]}：E2E 滚动标准差（稳定性）")
        ax.set_xlabel("Episode")
        ax.set_ylabel("最近 100 Episode E2E 标准差（ms）")
        _legend(ax, loc="upper right", handles=_condition_handles())
        _finish(ax)
    fig.text(
        0.5, -0.02,
        "滚动标准差越低越稳定；粗线为三种子均值，细线为 seed 11/22/33。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = training[[
        "model", "deployment", "algorithm", "seed", "episode",
        "std100_e2e",
    ]]
    _save(fig, out_dir / "fig03_training_e2e_rolling_std.png", data,
          FIGURE_DATA_DIR)


def plot_dev_e2e(validation, out_dir):
    if validation.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    for ax, model in zip(axes, MEC_MODELS):
        color = MODEL_COLORS[model]
        for algorithm in ALGORITHMS:
            linestyle = ALGORITHM_LINESTYLES[algorithm]
            subset = validation[
                (validation["model"] == model)
                & (validation["algorithm"] == algorithm)
            ]
            for seed in sorted(subset["seed"].unique()):
                run = subset[subset["seed"] == seed].sort_values("episode")
                ax.plot(
                    run["episode"], run["E2E_ms"], color=color,
                    linestyle=linestyle, linewidth=1.0, alpha=0.35,
                    marker=ALGORITHM_MARKERS[algorithm], markersize=3,
                    markeredgecolor=SURFACE, markeredgewidth=0.6,
                )
                best = run.loc[run["E2E_ms"].idxmin()]
                ax.scatter(
                    [best["episode"]], [best["E2E_ms"]], s=45,
                    facecolor=color, edgecolor=SURFACE, linewidth=1.1,
                    zorder=4,
                )
            mean = subset.groupby("episode", as_index=False)[
                "E2E_ms"
            ].mean()
            ax.plot(
                mean["episode"], mean["E2E_ms"], color=color,
                linestyle=linestyle, linewidth=2.4,
                label=f"{ALGORITHM_LABELS[algorithm]}（3 seed 均值）",
            )
        ax.set_title(f"{MODEL_LABELS[model]}：Dev-60 平均 E2E")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Dev-60 E2E（ms）")
        _legend(ax, loc="upper right", handles=_condition_handles())
        _finish(ax)
    fig.text(
        0.5, -0.02,
        "每个点为每 200 Episode 的 Dev-60 验证；圆点为各 seed 的最佳检查点。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = validation[[
        "model", "deployment", "algorithm", "seed", "episode", "E2E_ms",
        "RewardD", "SR", "OSR", "SPL", "nDTW",
    ]]
    _save(fig, out_dir / "fig04_dev_e2e_curves.png", data, FIGURE_DATA_DIR)


def _eval_seed_aggregate(eval_summary):
    if not eval_summary:
        return []
    return eval_summary.get("seed_aggregate", [])


def _eval_policy_rows(eval_summary):
    if not eval_summary:
        return []
    return eval_summary.get("summary", [])


def _baseline_row(eval_summary, model, policy):
    for row in _eval_policy_rows(eval_summary):
        if row["model"] == model and row["policy"] == policy:
            return row
    return None


def plot_validation_metrics(eval_summary, out_dir):
    rows = _eval_seed_aggregate(eval_summary)
    if not rows:
        return
    metrics = ("SR", "OSR", "nDTW", "SPL", "E2E_ms", "RewardD")
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.2))
    labels = [f"{row['deployment']}\n{row['algorithm'].upper()}"
              for row in rows]
    x = np.arange(len(rows))
    for ax, metric in zip(axes.ravel(), metrics):
        values = np.asarray(
            [float(row[f"{metric}_mean"]) for row in rows]
        )
        errors = np.asarray(
            [float(row[f"{metric}_std"]) for row in rows]
        )
        colors = [
            MODEL_COLORS["Small" if row["model"] == "Small"
                         else "Transformer-M"] for row in rows
        ]
        hatches = ["//" if row["algorithm"] == "ppo" else ""
                   for row in rows]
        bars = ax.bar(
            x, values, yerr=errors, color=colors, edgecolor=SURFACE,
            linewidth=1.0, capsize=3, zorder=3, width=0.68,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        for model in MEC_MODELS:
            base = _baseline_row(eval_summary, model, "greedy_min_e2e")
            if base is None:
                continue
            ax.axhline(
                float(base[metric]), color=INK_MUTED, linewidth=1.0,
                linestyle=":", zorder=2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(METRIC_TITLES[metric], fontsize=10)
        if metric in ("SR", "OSR"):
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{100*value:.0f}%")
            )
        _finish(ax)
    handles = [
        Patch(facecolor=MODEL_COLORS["Small"], label="10×Small"),
        Patch(facecolor=MODEL_COLORS["Transformer-M"], label="10×Transformer-M"),
        Patch(facecolor=INK_MUTED, hatch="//", label="PPO（斜线）"),
        Line2D([0], [0], color=INK_MUTED, linestyle=":", linewidth=1.2,
               label="Greedy-Min-E2E baseline"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        fontsize=9, bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Stage-2r Dev-60 最终验证：3 seed 均值 ± 标准差",
        fontsize=13, color=INK,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    data = pd.DataFrame(rows)
    _save(fig, out_dir / "fig05_validation_metrics.png", data,
          FIGURE_DATA_DIR)


def plot_baseline_comparison(eval_summary, out_dir):
    if not eval_summary:
        return
    aggregate = {
        (row["model"], row["algorithm"]): row
        for row in _eval_seed_aggregate(eval_summary)
    }
    metrics = ("E2E_ms", "RewardD")
    baselines = (
        ("random", "Random"),
        ("nearest", "Nearest"),
        ("max_uplink_rate", "Max-Uplink-Rate"),
        ("greedy_min_e2e", "Greedy-Min-E2E"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), sharex=True)
    for row_index, model in enumerate(MEC_MODELS):
        groups = []
        for algorithm in ALGORITHMS:
            row = aggregate.get((model, algorithm))
            if row is not None:
                groups.append((
                    ALGORITHM_LABELS[algorithm], row, model,
                    f"learned_{algorithm}",
                ))
        for baseline, title in baselines:
            base = _baseline_row(eval_summary, model, baseline)
            if base is not None:
                groups.append((title, base, model, baseline))
        x = np.arange(len(groups))
        for col_index, metric in enumerate(metrics):
            ax = axes[row_index, col_index]
            values = [
                float(item[1][f"{metric}_mean"])
                if f"{metric}_mean" in item[1]
                else float(item[1][metric])
                for item in groups
            ]
            colors = [
                MODEL_COLORS[model] if item[3].startswith("learned")
                else INK_MUTED for item in groups
            ]
            bars = ax.bar(
                x, values, color=colors, edgecolor=SURFACE, linewidth=1.0,
                width=0.7, zorder=3,
            )
            for bar, item in zip(bars, groups):
                if not item[3].startswith("learned"):
                    bar.set_hatch("..")
            ax.set_xticks(x)
            ax.set_xticklabels([item[0] for item in groups], fontsize=7.5)
            ax.set_title(
                f"{MODEL_LABELS[model]}：{METRIC_TITLES[metric]}",
                fontsize=10,
            )
            _finish(ax)
    fig.text(
        0.5, -0.02,
        "学习方法的 SR/OSR/nDTW 与同部署 baseline 相同（导航由冻结 MEC "
        "决定）；差异只应出现在 E2E/RewardD。点状填充为启发式 baseline。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    data_rows = []
    for model in MEC_MODELS:
        for algorithm in ALGORITHMS:
            row = aggregate.get((model, algorithm))
            if row is not None:
                data_rows.append({
                    "model": model,
                    "label": ALGORITHM_LABELS[algorithm],
                    "kind": f"learned_{algorithm}",
                    "E2E_ms": float(row["E2E_ms_mean"]),
                    "RewardD": float(row["RewardD_mean"]),
                })
        for baseline, title in baselines:
            base = _baseline_row(eval_summary, model, baseline)
            if base is not None:
                data_rows.append({
                    "model": model,
                    "label": title,
                    "kind": baseline,
                    "E2E_ms": float(base["E2E_ms"]),
                    "RewardD": float(base["RewardD"]),
                })
    data = pd.DataFrame(data_rows)
    _save(fig, out_dir / "fig06_baseline_comparison.png", data,
          FIGURE_DATA_DIR)


def plot_deployment_comparison(eval_summary, out_dir):
    rows = _eval_seed_aggregate(eval_summary)
    if not rows:
        return
    metrics = ("SR", "OSR", "nDTW", "SPL", "E2E_ms", "RewardD")
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.2))
    for ax, metric in zip(axes.ravel(), metrics):
        for algorithm_index, algorithm in enumerate(ALGORITHMS):
            for model_index, model in enumerate(MEC_MODELS):
                subset = [
                    row for row in rows
                    if row["model"] == model
                    and row["algorithm"] == algorithm
                ]
                if not subset:
                    continue
                row = subset[0]
                position = (
                    algorithm_index + (2 * model_index - 1) * 0.19
                )
                bar = ax.bar(
                    position, float(row[f"{metric}_mean"]),
                    yerr=float(row[f"{metric}_std"]), width=0.34,
                    color=MODEL_COLORS[model], edgecolor=SURFACE,
                    linewidth=1.0, capsize=3, zorder=3,
                    label=MODEL_LABELS[model],
                )
                if algorithm == "ppo":
                    bar[0].set_hatch("//")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["DQN", "PPO"])
        ax.set_title(METRIC_TITLES[metric], fontsize=10)
        if metric in ("SR", "OSR"):
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{100*value:.0f}%")
            )
        _finish(ax)
    fig.legend(
        handles=[
            Patch(facecolor=MODEL_COLORS["Small"], label="10×Small"),
            Patch(
                facecolor=MODEL_COLORS["Transformer-M"],
                label="10×Transformer-M",
            ),
            Patch(facecolor=INK_MUTED, hatch="//", label="PPO（斜线）"),
        ],
        loc="lower center", ncol=3, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Stage-2r 部署模型对比：Small vs Transformer-M（3 seed）",
        fontsize=13, color=INK,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save(fig, out_dir / "fig07_deployment_comparison.png",
          pd.DataFrame(rows), FIGURE_DATA_DIR)


def plot_seed_variability(eval_summary, out_dir):
    rows = _eval_policy_rows(eval_summary)
    learned = [row for row in rows if row["kind"] == "learned"]
    if not learned:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    metrics = ("E2E_ms", "RewardD")
    for ax, metric in zip(axes, metrics):
        for index, row in enumerate(
            sorted(learned, key=lambda item: (
                MEC_MODELS.index(item["model"]),
                ALGORITHMS.index(item["algorithm"]),
                item["seed"],
            ))
        ):
            color = MODEL_COLORS[row["model"]]
            marker = ALGORITHM_MARKERS[row["algorithm"]]
            ax.scatter(
                index, float(row[metric]), s=42, facecolor=color,
                marker=marker, edgecolor=SURFACE, linewidth=0.9, zorder=3,
            )
        ax.set_title(METRIC_TITLES[metric])
        ax.set_xticks(range(len(learned)))
        ax.set_xticklabels([
            f"{row['deployment']}\n{row['algorithm'].upper()} s{row['seed']}"
            for row in sorted(learned, key=lambda item: (
                MEC_MODELS.index(item["model"]),
                ALGORITHMS.index(item["algorithm"]),
                item["seed"],
            ))
        ], fontsize=6.5)
        _finish(ax)
    fig.text(
        0.5, -0.03,
        "每个点是一个 seed 的最佳检查点在 Dev-60 上的结果；横轴为 12 个 run。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    _save(fig, out_dir / "fig08_seed_variability.png",
          pd.DataFrame(learned), FIGURE_DATA_DIR)


def plot_success_failure(run_rows, condition_rows, out_dir):
    if not run_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    for ax, grouping, title in (
        (axes[0], run_rows, "逐 run（12 个 run）"),
        (axes[1], condition_rows, "条件聚合（3 seed 均值）"),
    ):
        labels = []
        success = []
        failure = []
        for row in grouping:
            if "seed" in row:
                label = (f"{row['deployment']}\n"
                         f"{row['algorithm_label']} s{row['seed']}")
            else:
                label = f"{row['deployment']}\n{row['algorithm_label']}"
            labels.append(label)
            if "cum_success" in row:
                success.append(int(row["cum_success"]))
                failure.append(int(row["cum_failure"]))
            else:
                success.append(float(row["cum_success_mean"]))
                failure.append(float(row["cum_failure_mean"]))
        x = np.arange(len(grouping))
        ax.bar(x, success, color="#2a78d6", edgecolor=SURFACE,
               linewidth=0.8, label="成功", zorder=3)
        ax.bar(x, failure, bottom=success, color="#c3c2b7",
               edgecolor=SURFACE, linewidth=0.8, label="失败", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel("训练 Episode 数（共 4000）")
        ax.set_title(title)
        _legend(ax, loc="upper right")
        _finish(ax)
    fig.text(
        0.5, -0.03,
        "成功/失败由冻结 MEC 导航模型决定；同一部署下 DQN/PPO/baseline "
        "的成功数相同，差异只在关联时延。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = pd.DataFrame([
        {
            "group": (
                f"{row['deployment']}|{row['algorithm_label']}|"
                f"s{row.get('seed', 'mean')}"
            ),
            "success": row.get("cum_success", row.get("cum_success_mean")),
            "failure": row.get("cum_failure", row.get("cum_failure_mean")),
            "success_short": row.get(
                "cum_success_short", row.get("cum_success_short_mean")
            ),
            "success_medium": row.get(
                "cum_success_medium", row.get("cum_success_medium_mean")
            ),
            "success_long": row.get(
                "cum_success_long", row.get("cum_success_long_mean")
            ),
        }
        for row in list(run_rows) + list(condition_rows)
    ])
    _save(fig, out_dir / "fig09_success_failure.png", data,
          FIGURE_DATA_DIR)


def plot_success_bands(condition_rows, out_dir):
    if not condition_rows:
        return
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    labels = [
        f"{row['deployment']}\n{row['algorithm_label']}"
        for row in condition_rows
    ]
    x = np.arange(len(condition_rows))
    bottom = np.zeros(len(condition_rows))
    colors = {"short": "#2a78d6", "medium": "#eb6834", "long": "#1baf7a"}
    for band in ("short", "medium", "long"):
        values = np.asarray([
            float(row[f"cum_success_{band}_mean"])
            for row in condition_rows
        ])
        ax.bar(
            x, values, bottom=bottom, color=colors[band],
            edgecolor=SURFACE, linewidth=1.0, width=0.65,
            label=f"成功 {band}", zorder=3,
        )
        bottom += values
    failures = np.asarray([
        float(row["cum_failure_mean"]) for row in condition_rows
    ])
    ax.bar(
        x, failures, bottom=bottom, color="#c3c2b7", edgecolor=SURFACE,
        linewidth=1.0, width=0.65, label="失败", zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("训练 Episode 数（3 seed 均值）")
    ax.set_title("成功 Episode 的 short / medium / long 分布")
    _legend(ax, loc="upper right")
    _finish(ax)
    fig.text(
        0.5, -0.03,
        "同一部署下 DQN/PPO 成功数相同，因为导航动作由冻结 MEC 模型产生。",
        ha="center", va="top", fontsize=8, color=INK_SECONDARY,
    )
    data = pd.DataFrame(condition_rows)[[
        "condition", "cum_success_short_mean", "cum_success_medium_mean",
        "cum_success_long_mean", "cum_failure_mean",
    ]]
    _save(fig, out_dir / "fig10_success_bands.png", data, FIGURE_DATA_DIR)


def make_all(training, validation, rounds, run_rows, condition_rows,
             eval_summary, evidence_dir):
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_reward(training, FIGURES_DIR)
    plot_training_e2e(training, FIGURES_DIR)
    plot_training_std(training, FIGURES_DIR)
    plot_dev_e2e(validation, FIGURES_DIR)
    plot_validation_metrics(eval_summary, FIGURES_DIR)
    plot_baseline_comparison(eval_summary, FIGURES_DIR)
    plot_deployment_comparison(eval_summary, FIGURES_DIR)
    plot_seed_variability(eval_summary, FIGURES_DIR)
    plot_success_failure(run_rows, condition_rows, FIGURES_DIR)
    plot_success_bands(condition_rows, FIGURES_DIR)
    return sorted(path.name for path in FIGURES_DIR.glob("fig*.png"))


__all__ = ["make_all", "setup_style"]
