"""300-dpi Stage-2s figures with CSV sidecars.

Figures:
  fig01  DQN/PPO Reward-MA100 curves
  fig02  DQN/PPO E2E-MA100 curves
  fig03  Dev E2E vs training episode
  fig04  validation metrics (SR/OSR/nDTW/SPL/E2E/RewardD) incl. baselines
  fig05  seed variability (per-seed values + mean/std)
  fig05b training stability (rolling E2E/reward std)
  fig06  heterogeneity: selection ratio + latency decomposition
  fig07  learned vs baseline E2E
  fig08  success/failure band distribution
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2o.plot import (  # noqa: E402
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
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (  # noqa: E402
    ALGORITHMS,
    ALGORITHM_LABELS,
    EVIDENCE_DIR,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    FORMAL_SEEDS,
)

ALGO_COLORS = {"dqn": "#2a78d6", "ppo": "#eb6834"}
ALGO_LINESTYLES = {"dqn": "-", "ppo": "--"}
ALGO_MARKERS = {"dqn": "o", "ppo": "s"}
SEED_LINESTYLES = {101: "-", 202: "--", 303: ":"}
SEED_MARKERS = {101: "o", 202: "s", 303: "^"}
MODEL_COLORS = {"Small": "#1baf7a", "Transformer-M": "#6b4fd8"}
BASELINE_ORDER = ("random", "nearest", "max_uplink_rate", "greedy_min_e2e")
BASELINE_TITLES = {
    "random": "Random",
    "nearest": "Nearest",
    "max_uplink_rate": "Max-Uplink-Rate",
    "greedy_min_e2e": "Greedy-Min-E2E",
}
BASELINE_COLORS = {
    "random": "#898781",
    "nearest": "#52514e",
    "max_uplink_rate": "#b08a3e",
    "greedy_min_e2e": "#c2417a",
}
METRIC_LABELS = {
    "SR": "SR（成功率）",
    "OSR": "OSR（可观测成功率）",
    "nDTW": "nDTW",
    "SPL": "SPL",
    "E2E_ms": "E2E 时延 (ms)",
    "RewardD": "方案 D 奖励",
}


def _read(name):
    path = EVIDENCE_DIR / name
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _seed_mean(frame, y, group=("algorithm", "episode")):
    return frame.groupby(list(group), as_index=False)[y].mean()


def _plot_training_metric(training, column, ylabel, title, path, tau=False):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    rows = []
    for ax, algorithm in zip(axes, ALGORITHMS):
        subset = training[training["algorithm"] == algorithm]
        color = ALGO_COLORS[algorithm]
        for seed in FORMAL_SEEDS:
            run = subset[subset["seed"] == seed].sort_values("episode")
            if run.empty:
                continue
            ax.plot(
                run["episode"], run[column], color=color,
                linestyle=SEED_LINESTYLES.get(seed, "-"), alpha=0.28,
                linewidth=1.1, zorder=2,
            )
            rows.extend({
                "algorithm": algorithm, "seed": int(seed),
                "episode": int(ep), "value": float(value),
                "kind": "seed",
            } for ep, value in zip(run["episode"], run[column])
                if np.isfinite(float(value)))
        mean = _seed_mean(subset, column)
        if not mean.empty:
            ax.plot(
                mean["episode"], mean[column], color=color,
                linestyle="-", linewidth=2.4, zorder=3,
                label="3-seed 均值",
            )
            rows.extend({
                "algorithm": algorithm, "seed": -1,
                "episode": int(ep), "value": float(value),
                "kind": "mean",
            } for ep, value in zip(mean["episode"], mean[column])
                if np.isfinite(float(value)))
        if tau:
            ax.axhline(
                100.0, color=BASELINE, linewidth=1.0, zorder=1
            )
            ax.text(
                0.02, 100.0, "tau = 100 ms", va="bottom", ha="left",
                transform=ax.get_yaxis_transform(),
                fontsize=8, color=INK_SECONDARY,
            )
        ax.set_title(f"{ALGORITHM_LABELS[algorithm]}：{title}")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        _legend(ax, loc="best")
        _finish(ax)
    fig.tight_layout()
    _save(fig, path, pd.DataFrame(rows), FIGURE_DATA_DIR)


def plot_training_reward(training):
    _plot_training_metric(
        training, "ma100_reward", "最近 100 Episode 平均奖励",
        "Reward-MA100", FIGURES_DIR / "fig01_reward_ma100.png",
    )


def plot_training_e2e(training):
    _plot_training_metric(
        training, "ma100_e2e", "最近 100 Episode 平均 E2E (ms)",
        "E2E-MA100", FIGURES_DIR / "fig02_e2e_ma100.png", tau=True,
    )


def plot_dev_e2e(validation):
    if validation is None or validation.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    rows = []
    for ax, algorithm in zip(axes, ALGORITHMS):
        subset = validation[validation["algorithm"] == algorithm]
        color = ALGO_COLORS[algorithm]
        for seed in FORMAL_SEEDS:
            run = subset[subset["seed"] == seed].sort_values("episode")
            if run.empty:
                continue
            ax.plot(
                run["episode"], run["E2E_ms"], color=color,
                linestyle=SEED_LINESTYLES.get(seed, "-"), alpha=0.35,
                marker=SEED_MARKERS.get(seed, "o"), markersize=3,
                linewidth=1.1, zorder=2,
            )
            rows.extend({
                "algorithm": algorithm, "seed": int(seed),
                "episode": int(ep), "E2E_ms": float(value),
                "kind": "seed",
            } for ep, value in zip(run["episode"], run["E2E_ms"]))
        mean = subset.groupby("episode", as_index=False)["E2E_ms"].mean()
        if not mean.empty:
            ax.plot(
                mean["episode"], mean["E2E_ms"], color=color,
                linewidth=2.4, zorder=3, label="3-seed 均值",
            )
            rows.extend({
                "algorithm": algorithm, "seed": -1,
                "episode": int(ep), "E2E_ms": float(value),
                "kind": "mean",
            } for ep, value in zip(mean["episode"], mean["E2E_ms"]))
        ax.axhline(100.0, color=BASELINE, linewidth=1.0, zorder=1)
        ax.text(
            0.02, 100.0, "tau = 100 ms", va="bottom", ha="left",
            transform=ax.get_yaxis_transform(),
            fontsize=8, color=INK_SECONDARY,
        )
        ax.set_title(f"{ALGORITHM_LABELS[algorithm]}：Dev E2E")
        ax.set_xlabel("训练 Episode")
        ax.set_ylabel("Dev-60 平均 E2E (ms)")
        _legend(ax, loc="best")
        _finish(ax)
    fig.tight_layout()
    _save(
        fig, FIGURES_DIR / "fig03_dev_e2e_curves.png",
        pd.DataFrame(rows), FIGURE_DATA_DIR,
    )


def _eval_policy_rows(summary, aggregate):
    """Return rows for the validation-metric figure."""
    rows = []
    for algorithm in ALGORITHMS:
        agg = aggregate[aggregate["algorithm"] == algorithm]
        if agg.empty:
            continue
        item = agg.iloc[0]
        rows.append({
            "policy": f"{algorithm}_mean",
            "title": f"{ALGORITHM_LABELS[algorithm]}（3 seeds 均值）",
            "kind": "learned",
            "algorithm": algorithm,
        })
        for metric in ("SR", "OSR", "nDTW", "SPL", "E2E_ms", "RewardD"):
            rows[-1][metric] = float(item[f"{metric}_mean"])
            rows[-1][f"{metric}_std"] = float(item[f"{metric}_std"])
    for strategy in BASELINE_ORDER:
        subset = summary[summary["policy"] == strategy]
        if subset.empty:
            continue
        item = subset.iloc[0]
        row = {
            "policy": strategy,
            "title": BASELINE_TITLES[strategy],
            "kind": "baseline",
            "algorithm": "heuristic",
        }
        for metric in ("SR", "OSR", "nDTW", "SPL", "E2E_ms", "RewardD"):
            row[metric] = float(item[metric])
            row[f"{metric}_std"] = 0.0
        rows.append(row)
    return rows


def plot_validation_metrics(summary, aggregate):
    if summary is None or aggregate is None:
        return
    rows = _eval_policy_rows(summary, aggregate)
    metrics = ("SR", "OSR", "nDTW", "SPL", "E2E_ms", "RewardD")
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2))
    for ax, metric in zip(axes.ravel(), metrics):
        labels = [row["title"] for row in rows]
        values = np.asarray([row[metric] for row in rows])
        errors = np.asarray([row.get(f"{metric}_std", 0.0) for row in rows])
        colors = [
            ALGO_COLORS.get(row["algorithm"], BASELINE_COLORS.get(
                row["policy"], INK_MUTED
            ))
            for row in rows
        ]
        positions = np.arange(len(rows))
        ax.bar(
            positions, values, yerr=errors, capsize=2.5,
            color=colors, edgecolor="none", alpha=0.9, zorder=2,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=7.5)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel(METRIC_LABELS[metric])
        _finish(ax)
        if metric in ("SR", "OSR"):
            ax.yaxis.set_major_formatter(
                lambda value, _pos: f"{100*value:.0f}%"
            )
        if metric == "E2E_ms":
            ax.axhline(
                100.0, color=BASELINE, linewidth=1.0, zorder=1
            )
    fig.suptitle(
        "Dev-60 最终验证：学习策略（3 seed 均值±标准差）与 baseline",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    data = pd.DataFrame(rows)
    _save(
        fig, FIGURES_DIR / "fig04_validation_metrics.png",
        data, FIGURE_DATA_DIR,
    )


def plot_seed_variability(per_seed):
    if per_seed is None or per_seed.empty:
        return
    metrics = ("SR", "OSR", "nDTW", "SPL", "E2E_ms", "RewardD")
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    rows = []
    for ax, metric in zip(axes.ravel(), metrics):
        positions = []
        values = []
        for offset, algorithm in enumerate(ALGORITHMS):
            subset = per_seed[per_seed["algorithm"] == algorithm]
            if subset.empty:
                continue
            x = 0.5 + offset
            for index, (_, row) in enumerate(subset.iterrows()):
                value = float(row[metric])
                jitter = (index - 1) * 0.06
                ax.scatter(
                    x + jitter, value, color=ALGO_COLORS[algorithm],
                    marker=SEED_MARKERS.get(int(row["seed"]), "o"),
                    s=36, zorder=3,
                )
                rows.append({
                    "metric": metric, "algorithm": algorithm,
                    "seed": int(row["seed"]), "value": value,
                })
            mean = float(subset[metric].mean())
            std = float(subset[metric].std(ddof=0))
            ax.errorbar(
                x, mean, yerr=std, color=ALGO_COLORS[algorithm],
                marker=ALGO_MARKERS[algorithm], markersize=6,
                capsize=4, linewidth=1.6, zorder=4,
            )
            positions.append(x)
            values.append(mean)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["DQN", "PPO"])
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel(METRIC_LABELS[metric])
        if metric in ("SR", "OSR"):
            ax.yaxis.set_major_formatter(
                lambda value, _pos: f"{100*value:.0f}%"
            )
        if metric == "E2E_ms":
            ax.axhline(100.0, color=BASELINE, linewidth=1.0, zorder=1)
        _finish(ax)
    fig.suptitle(
        "三个随机种子（101/202/303）的验证指标变异性："
        "散点为单 seed，误差棒为均值±标准差",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(
        fig, FIGURES_DIR / "fig05_seed_variability.png",
        pd.DataFrame(rows), FIGURE_DATA_DIR,
    )


def plot_training_stability(training):
    if training is None or training.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    rows = []
    for ax, column, ylabel, title in (
        (axes[0], "std100_e2e", "E2E 滚动标准差 (ms)",
         "最近 100 Episode E2E 标准差"),
        (axes[1], None, "奖励滚动标准差",
         "最近 100 Episode 奖励标准差"),
    ):
        for algorithm in ALGORITHMS:
            subset = training[training["algorithm"] == algorithm]
            for seed in FORMAL_SEEDS:
                run = subset[subset["seed"] == seed].sort_values("episode")
                if run.empty:
                    continue
                if column is None:
                    values = run["episode_return"].rolling(
                        100, min_periods=100
                    ).std(ddof=0)
                else:
                    values = run[column]
                ax.plot(
                    run["episode"], values,
                    color=ALGO_COLORS[algorithm],
                    linestyle=SEED_LINESTYLES.get(seed, "-"),
                    alpha=0.55, linewidth=1.2,
                )
                rows.extend({
                    "algorithm": algorithm, "seed": int(seed),
                    "episode": int(ep), "value": float(value),
                    "metric": title,
                } for ep, value in zip(run["episode"], values)
                    if np.isfinite(float(value)))
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        _finish(ax)
    handles = [
        Line2D([0], [0], color=ALGO_COLORS[algorithm],
               label=ALGORITHM_LABELS[algorithm])
        for algorithm in ALGORITHMS
    ]
    axes[0].legend(handles=handles, loc="upper right")
    fig.tight_layout()
    _save(
        fig, FIGURES_DIR / "fig05b_training_stability.png",
        pd.DataFrame(rows), FIGURE_DATA_DIR,
    )


def plot_heterogeneity(latency):
    if latency is None or latency.empty:
        return
    learned = latency[latency["kind"].isin(("learned", "baseline"))].copy()
    order = [
        row["policy"] for row in learned.to_dict("records")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0))
    # (a) selection ratio
    ax = axes[0, 0]
    x = np.arange(len(learned))
    small = learned["small_ratio"].to_numpy(dtype=float)
    transformer = learned["transformer_m_ratio"].to_numpy(dtype=float)
    ax.bar(x, small, color=MODEL_COLORS["Small"], label="Small")
    ax.bar(
        x, transformer, bottom=small,
        color=MODEL_COLORS["Transformer-M"], label="Transformer-M",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("步数比例")
    ax.set_title("(a) 节点模型选择比例（Dev-60）")
    ax.legend(loc="upper right")
    _finish(ax)
    # (b) inference latency by model
    ax = axes[0, 1]
    width = 0.38
    ax.bar(
        x - width / 2, learned["mean_inference_small_ms"],
        width, color=MODEL_COLORS["Small"], label="Small",
    )
    ax.bar(
        x + width / 2, learned["mean_inference_transformer_m_ms"],
        width, color=MODEL_COLORS["Transformer-M"],
        label="Transformer-M",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("推理时延 (ms)")
    ax.set_title("(b) 平均推理时延（0.2 TFLOP/s 有效吞吐）")
    ax.legend(loc="upper right")
    _finish(ax)
    # (c) communication latency by model
    ax = axes[1, 0]
    ax.bar(
        x - width / 2, learned["mean_communication_small_ms"],
        width, color=MODEL_COLORS["Small"], label="Small",
    )
    ax.bar(
        x + width / 2, learned["mean_communication_transformer_m_ms"],
        width, color=MODEL_COLORS["Transformer-M"],
        label="Transformer-M",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("通信时延 (ms)")
    ax.set_title("(c) 平均通信时延")
    ax.legend(loc="upper right")
    _finish(ax)
    # (d) E2E decomposition
    ax = axes[1, 1]
    extraction = learned["mean_extraction_ms"].to_numpy(dtype=float)
    communication = learned["mean_communication_ms"].to_numpy(dtype=float)
    inference = learned["mean_inference_ms"].to_numpy(dtype=float)
    ax.bar(x, extraction, color="#b08a3e", label="UAV 提取")
    ax.bar(
        x, communication, bottom=extraction, color="#52514e",
        label="通信",
    )
    ax.bar(
        x, inference, bottom=extraction + communication,
        color="#2a78d6", label="推理",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("E2E (ms)")
    ax.set_title("(d) 平均 E2E 分解")
    ax.legend(loc="upper right")
    _finish(ax)
    fig.suptitle(
        "异构部署：Small/Transformer-M 选择比例与时延分解（Dev-60）",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(
        fig, FIGURES_DIR / "fig06_heterogeneity.png",
        learned, FIGURE_DATA_DIR,
    )


def plot_baseline_comparison(comparison):
    if comparison is None or comparison.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    for ax, algorithm in zip(axes, ALGORITHMS):
        subset = comparison[comparison["algorithm"] == algorithm]
        if subset.empty:
            continue
        labels = [row["baseline_title"] for row in subset.to_dict(
            "records"
        )]
        learned = subset["learned_E2E_ms_mean"].to_numpy(dtype=float)
        learned_std = subset["learned_E2E_ms_std"].to_numpy(dtype=float)
        baseline = subset["baseline_E2E_ms"].to_numpy(dtype=float)
        x = np.arange(len(labels))
        width = 0.36
        ax.bar(
            x - width / 2, learned, width, yerr=learned_std, capsize=3,
            color=ALGO_COLORS[algorithm],
            label=f"{ALGORITHM_LABELS[algorithm]} 3-seed 均值",
        )
        ax.bar(
            x + width / 2, baseline, width,
            color="#c3c2b7", label="baseline",
        )
        for index, (left, right) in enumerate(zip(learned, baseline)):
            ax.text(
                index, max(left, right) + 2.0,
                f"{(left-right)/right*100:+.1f}%",
                ha="center", va="bottom", fontsize=7.5,
                color=INK_SECONDARY,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Dev 平均 E2E (ms)")
        ax.set_title(
            f"{ALGORITHM_LABELS[algorithm]} vs baseline："
            "负百分比表示学习策略时延更低"
        )
        ax.axhline(100.0, color=BASELINE, linewidth=1.0, zorder=1)
        ax.legend(loc="upper right")
        _finish(ax)
    fig.tight_layout()
    _save(
        fig, FIGURES_DIR / "fig07_baseline_e2e.png",
        comparison, FIGURE_DATA_DIR,
    )


def plot_success_failure(success):
    if success is None or success.empty:
        return
    bands = ("short", "medium", "long")
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8))
    rows = []
    for ax, prefix, title in (
        (axes[0], "success", "成功任务 band 分布"),
        (axes[1], "failure", "失败任务 band 分布"),
    ):
        x = np.arange(len(success))
        bottom = np.zeros(len(success))
        for band in bands:
            values = success[f"{prefix}_{band}"].to_numpy(dtype=float)
            ax.bar(
                x, values, bottom=bottom, label=band,
                color={
                    "short": "#2a78d6",
                    "medium": "#1baf7a",
                    "long": "#eb6834",
                }[band],
            )
            bottom += values
            rows.extend({
                "run": f"{row['algorithm']}-s{row['seed']}",
                "outcome": prefix, "band": band,
                "count": int(value),
            } for row, value in zip(
                success.to_dict("records"), values
            ))
        ax.set_xticks(x)
        ax.set_xticklabels([
            f"{row['algorithm'].upper()}-s{row['seed']}"
            for row in success.to_dict("records")
        ], rotation=20, ha="right")
        ax.set_ylabel("Episode 数")
        ax.set_title(title)
        ax.legend(loc="upper right")
        _finish(ax)
    fig.tight_layout()
    _save(
        fig, FIGURES_DIR / "fig08_success_failure.png",
        pd.DataFrame(rows), FIGURE_DATA_DIR,
    )


def make_all():
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    training = _read("training_curves.csv")
    validation = _read("validation_curves.csv")
    summary = _read("validation_summary.csv")
    aggregate = _read("seed_aggregate.csv")
    per_seed = _read("per_seed_metrics.csv")
    latency = _read("latency_decomposition.csv")
    comparison = _read("baseline_comparison.csv")
    success = _read("success_failure_summary.csv")
    if training is not None:
        plot_training_reward(training)
        plot_training_e2e(training)
        plot_training_stability(training)
    if validation is not None:
        plot_dev_e2e(validation)
    if summary is not None and aggregate is not None:
        plot_validation_metrics(summary, aggregate)
    if per_seed is not None:
        plot_seed_variability(per_seed)
    if latency is not None:
        plot_heterogeneity(latency)
    if comparison is not None:
        plot_baseline_comparison(comparison)
    if success is not None:
        plot_success_failure(success)
    print("[plot] figures written to", FIGURES_DIR, flush=True)
    print("STAGE2S_PLOT_COMPLETE", flush=True)


def main(argv=None):
    make_all()


if __name__ == "__main__":
    main()
