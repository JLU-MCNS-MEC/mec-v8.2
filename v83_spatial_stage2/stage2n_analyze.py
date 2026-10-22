"""Stage-2n analysis: convergence table, multi-seed table, figures and report.

The analysis reads only files produced by the Stage-2n trainer/evaluator.  It
never opens the Test task pool; Test results come from the final evaluator's
summary.json, which records ``final_test_read``.

Outputs (Git-tracked, because ``runs/`` is ignored):

    v83_spatial_stage2/stage2n_evidence/
      training_curves.csv
      validation_curves.csv
      convergence.csv
      test_policy_summary.csv
      test_multi_seed.csv
      test_by_band.csv
      navigation_invariance.csv
      figures/*.png
      STAGE2N_RESULTS.md
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    read_json,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2n_common import (  # noqa: E402
    AGENT_SEEDS,
    CHECKPOINT_SELECTION,
    CONVERGENCE_CRITERION,
    EVIDENCE_DIR,
    MEC_CHECKPOINT_SHA256,
    RUNS_DIR,
    STAGE,
    convergence_status,
)

FIG_DIR = EVIDENCE_DIR / "figures"
ALGORITHMS = ("dqn", "ppo")
ALG_TITLES = {"dqn": "DQN", "ppo": "PPO"}
POLICY_TITLES = {
    "random": "Random",
    "nearest": "Nearest MEC",
    "max_uplink_rate": "Max-Uplink-Rate",
    "greedy_min_e2e": "Greedy-Min-E2E",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
NEUTRAL = "#b9b7ae"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    })


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _moving_average(values, window=100):
    output = np.full(len(values), np.nan, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        output[index] = float(np.mean(values[start:index + 1]))
    return output


def _training_series():
    series = {}
    for algorithm in ALGORITHMS:
        for seed in AGENT_SEEDS:
            run_dir = RUNS_DIR / f"{algorithm}_s{seed}"
            history = read_json(run_dir / "training_history.json")
            if history is None or not history.get("train"):
                continue
            rows = history["train"]
            episodes = np.asarray([int(row["episode"]) for row in rows])
            returns = np.asarray(
                [float(row["latency_return"]) for row in rows]
            )
            e2e = np.asarray(
                [float(row["mean_episode_e2e_ms"]) for row in rows]
            )
            series[(algorithm, seed)] = {
                "episode": episodes,
                "latency_return": returns,
                "latency_return_ma100": _moving_average(returns),
                "mean_e2e_ms": e2e,
                "mean_e2e_ms_ma100": _moving_average(e2e),
                "success": np.asarray(
                    [float(row["success"]) for row in rows]
                ),
                "ndtw": np.asarray(
                    [float(row["final_nDTW"]) for row in rows]
                ),
                "steps": np.asarray([int(row["steps"]) for row in rows]),
                "history": history,
            }
    return series


def _convergence_rows(series):
    rows = []
    for (algorithm, seed), item in sorted(series.items()):
        validation = item["history"].get("validation", [])
        result = read_json(
            RUNS_DIR / f"{algorithm}_s{seed}" / "training_result.json"
        ) or {}
        conv = result.get("convergence") or convergence_status(
            validation, len(item["episode"]) >= 4000
        )
        best = result.get("best_validation") or {}
        rows.append({
            "algorithm": algorithm,
            "seed": int(seed),
            "episodes": int(len(item["episode"])),
            "validations": len(validation),
            "best_dev_e2e_ms": _float(conv.get("best_e2e_ms")),
            "best_dev_episode": int(conv.get("best_episode", -1)),
            "final_dev_e2e_ms": _float(conv.get("final_e2e_ms")),
            "last3_spread_fraction": _float(
                conv.get("last3_spread_fraction")
            ),
            "final_gap_fraction": _float(
                conv.get("final_gap_fraction")
            ),
            "converged": bool(conv.get("converged")),
            "status": str(conv.get("status")),
            "reached_cap": bool(conv.get("reached_cap")),
            "best_checkpoint": result.get("best_checkpoint"),
            "best_checkpoint_episode": result.get("best_dev_episode"),
            "best_dev_sr": _float(best.get("SR")),
            "best_dev_spl": _float(best.get("SPL")),
            "best_dev_ndtw": _float(best.get("nDTW")),
            "criterion": conv.get("criterion", CONVERGENCE_CRITERION),
        })
    return rows


def _plot_training(series):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    colors = {"dqn": BLUE, "ppo": ORANGE}
    for axis, algorithm in zip(axes, ALGORITHMS):
        subset = [
            (seed, item) for (alg, seed), item in series.items()
            if alg == algorithm
        ]
        if not subset:
            axis.set_title(f"{ALG_TITLES[algorithm]} (no data)")
            continue
        for seed, item in sorted(subset):
            axis.plot(
                item["episode"], item["latency_return"],
                color=colors[algorithm], lw=1.0, alpha=0.28,
            )
            axis.plot(
                item["episode"], item["latency_return_ma100"],
                color=colors[algorithm], lw=2.2, alpha=0.95,
                label=f"seed {seed} MA100",
            )
        axis.set_title(f"{ALG_TITLES[algorithm]} — LatencyReturn")
        axis.set_xlabel("Training episode")
        axis.set_ylabel("LatencyReturn")
        axis.legend(loc="lower right", fontsize=8)
        axis.axhline(0.0, color=MUTED, lw=0.8, ls=":")
    fig.tight_layout()
    path = FIG_DIR / "training_latency_return.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_training_e2e(series):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    colors = {"dqn": BLUE, "ppo": ORANGE}
    for axis, algorithm in zip(axes, ALGORITHMS):
        subset = [
            (seed, item) for (alg, seed), item in series.items()
            if alg == algorithm
        ]
        if not subset:
            continue
        for seed, item in sorted(subset):
            axis.plot(
                item["episode"], item["mean_e2e_ms"],
                color=colors[algorithm], lw=1.0, alpha=0.25,
            )
            axis.plot(
                item["episode"], item["mean_e2e_ms_ma100"],
                color=colors[algorithm], lw=2.2,
                label=f"seed {seed} MA100",
            )
        axis.set_title(f"{ALG_TITLES[algorithm]} — training mean E2E")
        axis.set_xlabel("Training episode")
        axis.set_ylabel("Mean E2E (ms)")
        axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "training_mean_e2e.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_dev_with_test_refs(series, test_summary):
    _style()
    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    colors = {"dqn": BLUE, "ppo": ORANGE}
    for algorithm in ALGORITHMS:
        curves = [
            item["history"]["validation"]
            for (alg, _seed), item in series.items()
            if alg == algorithm and item["history"].get("validation")
        ]
        if not curves:
            continue
        max_len = max(len(curve) for curve in curves)
        episodes = np.asarray([
            curve[index]["episode"] for curve in curves for index in [0]
        ])
        # Use the first curve's episode grid; all runs share validation times.
        episodes = np.asarray([row["episode"] for row in curves[0]])
        matrix = np.asarray([
            [row["E2E_ms"] for row in curve] for curve in curves
        ], dtype=np.float64)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        axis.plot(episodes, mean, color=colors[algorithm], lw=2.2,
                  marker="o", ms=3.5, label=f"{ALG_TITLES[algorithm]} mean")
        axis.fill_between(
            episodes, mean - std, mean + std,
            color=colors[algorithm], alpha=0.15, linewidth=0,
        )
    if test_summary:
        references = (
            ("greedy_min_e2e", "Greedy-Min-E2E (Test)"),
            ("max_uplink_rate", "Max-Uplink-Rate (Test)"),
            ("nearest", "Nearest (Test)"),
            ("random", "Random (Test)"),
        )
        for policy, label in references:
            item = test_summary["overall"].get(policy)
            if item is None:
                continue
            axis.axhline(
                item["E2E_ms"], color=MUTED, lw=1.0, ls="--", alpha=0.85,
            )
            axis.annotate(
                f"{label}: {item['E2E_ms']:.1f} ms",
                xy=(episodes[-1] if len(episodes) else 4000,
                    item["E2E_ms"]),
                xytext=(4, 2), textcoords="offset points",
                fontsize=7.5, color=SECONDARY, va="bottom",
            )
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Dev-60 mean E2E (ms)")
    axis.set_title("Dev mean E2E during training, with Test baseline references")
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "dev_e2e_with_test_baselines.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _policy_order(test_summary):
    order = ["random", "nearest", "max_uplink_rate", "greedy_min_e2e"]
    for algorithm in ALGORITHMS:
        policy = f"{algorithm}_s0"
        if test_summary and policy in test_summary["overall"]:
            order.append(policy)
    return order


def _aggregate_policy(test_summary, base_policy):
    """Mean/std over learned seeds for a base algorithm policy."""
    algorithm = base_policy.split("_s")[0]
    rows = []
    for seed in AGENT_SEEDS:
        item = test_summary["overall"].get(f"{algorithm}_s{seed}")
        if item is not None:
            rows.append(item)
    if not rows:
        return None
    return {
        "title": ALG_TITLES.get(algorithm, algorithm),
        "E2E_ms": float(np.mean([row["E2E_ms"] for row in rows])),
        "E2E_std_ms": float(np.std([row["E2E_ms"] for row in rows])),
        "LatencyReturn": float(np.mean([
            row["LatencyReturn"] for row in rows
        ])),
        "LatencyReturn_std": float(np.std([
            row["LatencyReturn"] for row in rows
        ])),
        "P95_E2E_ms": float(np.mean([row["P95_E2E_ms"] for row in rows])),
        "SR": float(np.mean([row["SR"] for row in rows])),
        "OSR": float(np.mean([row["OSR"] for row in rows])),
        "SPL": float(np.mean([row["SPL"] for row in rows])),
        "nDTW": float(np.mean([row["nDTW"] for row in rows])),
        "delay_violation_rate": float(np.mean([
            row["delay_violation_rate"] for row in rows
        ])),
        "mean_steps": float(np.mean([row["mean_steps"] for row in rows])),
        "seeds": len(rows),
        "per_seed": rows,
    }


def _plot_test_bar(test_summary):
    _style()
    order = [
        ("random", "Random", NEUTRAL),
        ("nearest", "Nearest", NEUTRAL),
        ("max_uplink_rate", "Max-Uplink-Rate", NEUTRAL),
        ("greedy_min_e2e", "Greedy-Min-E2E", NEUTRAL),
        ("dqn_s0", "DQN", BLUE),
        ("ppo_s0", "PPO", ORANGE),
    ]
    labels, values, errors, colors = [], [], [], []
    for policy, title, color in order:
        if policy.startswith(("dqn_", "ppo_")):
            aggregate = _aggregate_policy(test_summary, policy)
            if aggregate is None:
                continue
            labels.append(title)
            values.append(aggregate["E2E_ms"])
            errors.append(aggregate["E2E_std_ms"])
            colors.append(color)
        else:
            item = test_summary["overall"].get(policy)
            if item is None:
                continue
            labels.append(title)
            values.append(item["E2E_ms"])
            errors.append(0.0)
            colors.append(color)
    fig, axis = plt.subplots(figsize=(8.2, 4.4))
    positions = np.arange(len(labels))
    bars = axis.barh(
        positions, values, color=colors, height=0.62,
        xerr=errors, error_kw={"ecolor": SECONDARY, "capsize": 3, "lw": 1},
    )
    for position, value in zip(positions, values):
        axis.text(
            value + max(values) * 0.012, position, f"{value:.2f}",
            va="center", fontsize=8.5, color=INK,
        )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("Test-400 mean E2E (ms) — lower is better")
    axis.set_title("Test E2E by association policy (same tasks and deployment)")
    axis.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    path = FIG_DIR / "test_e2e_by_policy.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path, labels, values, errors


def _plot_test_band(test_summary):
    _style()
    policies = [
        ("random", "Random", NEUTRAL),
        ("greedy_min_e2e", "Greedy-Min-E2E", VIOLET),
        ("dqn_s0", "DQN", BLUE),
        ("ppo_s0", "PPO", ORANGE),
    ]
    bands = ("short", "medium", "long")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharex=False)
    all_values = []
    data = {}
    for policy, title, color in policies:
        if policy.startswith(("dqn_", "ppo_")):
            algorithm = policy.split("_s")[0]
            band_values = {}
            for band in bands:
                values = [
                    test_summary["by_band"][f"{algorithm}_s{seed}"][band]["E2E_ms"]
                    for seed in AGENT_SEEDS
                    if f"{algorithm}_s{seed}" in test_summary["by_band"]
                ]
                band_values[band] = float(np.mean(values))
            data[title] = (band_values, color)
        else:
            band_values = {
                band: test_summary["by_band"][policy][band]["E2E_ms"]
                for band in bands
            }
            data[title] = (band_values, color)
    for title, (band_values, color) in data.items():
        all_values.extend(band_values.values())
    for axis, band in zip(axes, bands):
        labels = list(data)
        values = [data[label][0][band] for label in labels]
        colors = [data[label][1] for label in labels]
        positions = np.arange(len(labels))
        axis.bar(positions, values, color=colors, width=0.62)
        for position, value in zip(positions, values):
            axis.text(
                position, value + max(all_values) * 0.01, f"{value:.1f}",
                ha="center", va="bottom", fontsize=7.5, color=INK,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axis.set_title(band.capitalize())
        axis.set_ylim(0, max(all_values) * 1.18)
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test mean E2E (ms)")
    fig.suptitle("Test E2E by trajectory band", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = FIG_DIR / "test_e2e_by_band.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_navigation_invariance(test_summary):
    _style()
    policies = _policy_order(test_summary)
    # Use one learned seed per algorithm for the invariance plot.
    selected = [
        "random", "nearest", "max_uplink_rate", "greedy_min_e2e",
        "dqn_s0", "ppo_s0",
    ]
    metrics = [
        ("SR", "SR"), ("SPL", "SPL"), ("nDTW", "nDTW"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=False)
    for axis, (key, label) in zip(axes, metrics):
        values = [
            test_summary["overall"][policy][key]
            for policy in selected if policy in test_summary["overall"]
        ]
        labels = [
            "Random", "Nearest", "Max-Rate", "Greedy", "DQN", "PPO"
        ][:len(values)]
        positions = np.arange(len(values))
        axis.bar(positions, values, color=BLUE, width=0.62)
        for position, value in zip(positions, values):
            axis.text(
                position, value + 0.01, f"{value:.4f}",
                ha="center", va="bottom", fontsize=7.2, color=INK,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axis.set_title(label)
        axis.set_ylim(0, 1.0)
        axis.grid(axis="x", visible=False)
    fig.suptitle(
        "Navigation metrics are invariant to association "
        "(identical MEC actions)",
        fontsize=11.5, fontweight="bold",
    )
    fig.tight_layout()
    path = FIG_DIR / "test_navigation_invariance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _multi_seed_rows(test_summary):
    rows = []
    for base in ("dqn_s0", "ppo_s0"):
        aggregate = _aggregate_policy(test_summary, base)
        if aggregate is None:
            continue
        row = {
            "policy": aggregate["title"],
            "seeds": aggregate["seeds"],
            "E2E_ms_mean": aggregate["E2E_ms"],
            "E2E_ms_std": aggregate["E2E_std_ms"],
            "LatencyReturn_mean": aggregate["LatencyReturn"],
            "LatencyReturn_std": aggregate["LatencyReturn_std"],
            "P95_E2E_ms_mean": aggregate["P95_E2E_ms"],
            "SR_mean": aggregate["SR"],
            "OSR_mean": aggregate["OSR"],
            "SPL_mean": aggregate["SPL"],
            "nDTW_mean": aggregate["nDTW"],
            "delay_violation_rate_mean": aggregate[
                "delay_violation_rate"
            ],
            "mean_steps_mean": aggregate["mean_steps"],
        }
        for index, seed_row in enumerate(aggregate["per_seed"]):
            row[f"seed{index}_E2E_ms"] = seed_row["E2E_ms"]
            row[f"seed{index}_LatencyReturn"] = seed_row["LatencyReturn"]
        rows.append(row)
    for policy in ("random", "nearest", "max_uplink_rate", "greedy_min_e2e"):
        item = test_summary["overall"].get(policy)
        if item is None:
            continue
        rows.append({
            "policy": POLICY_TITLES[policy],
            "seeds": 1,
            "E2E_ms_mean": item["E2E_ms"],
            "E2E_ms_std": 0.0,
            "LatencyReturn_mean": item["LatencyReturn"],
            "LatencyReturn_std": 0.0,
            "P95_E2E_ms_mean": item["P95_E2E_ms"],
            "SR_mean": item["SR"],
            "OSR_mean": item["OSR"],
            "SPL_mean": item["SPL"],
            "nDTW_mean": item["nDTW"],
            "delay_violation_rate_mean": item["delay_violation_rate"],
            "mean_steps_mean": item["mean_steps"],
        })
    return rows


def _markdown(training_rows, convergence_rows, test_summary,
              multi_seed_rows, test_band_rows, training_series):
    lines = [
        "# Stage-2n 结果汇总：固定城市、同质 MEC、纯时延关联",
        "",
        f"- 生成时间：{utc_now()}",
        f"- 协议：`v83_spatial_stage2/STAGE2N_PROTOCOL.md`",
        f"- Transformer-M：`{MEC_CHECKPOINT_SHA256}`",
        f"- 奖励：LatencyReturn = −0.25 × mean(E2E ms) / 100（无导航奖励）",
        f"- 收敛判据：{CONVERGENCE_CRITERION}",
        f"- 检查点选择：{CHECKPOINT_SELECTION}",
        "",
        "## 1. 训练与收敛",
        "",
        "| 算法 | Seed | Episodes | Best Dev E2E (ms) | @Ep | Final Dev E2E (ms) | 最近三次波动 | 最终差距 | 收敛 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in convergence_rows:
        lines.append(
            f"| {ALG_TITLES.get(row['algorithm'], row['algorithm'])} | "
            f"{row['seed']} | {row['episodes']} | "
            f"{row['best_dev_e2e_ms']:.2f} | {row['best_dev_episode']} | "
            f"{row['final_dev_e2e_ms']:.2f} | "
            f"{100*row['last3_spread_fraction']:.2f}% | "
            f"{100*row['final_gap_fraction']:.2f}% | "
            f"{'是' if row['converged'] else '否'} | `{row['status']}` |"
        )
    lines += [
        "",
        "## 2. Test-400 核心结果",
        "",
        "| 策略 | Episodes | E2E (ms) | LatencyReturn | SR | OSR | SPL | nDTW | P95 E2E (ms) | 违约率 | 平均步数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if test_summary:
        for policy in test_summary["policy_order"]:
            item = test_summary["overall"][policy]
            lines.append(
                f"| {item['title']} | {item['episodes']} | "
                f"{item['E2E_ms']:.2f} | {item['LatencyReturn']:.5f} | "
                f"{100*item['SR']:.2f}% | {100*item['OSR']:.2f}% | "
                f"{item['SPL']:.4f} | {item['nDTW']:.4f} | "
                f"{item['P95_E2E_ms']:.2f} | "
                f"{100*item['delay_violation_rate']:.2f}% | "
                f"{item['mean_steps']:.1f} |"
            )
    else:
        lines.append("| Test 尚未运行 | - | - | - | - | - | - | - | - | - | - |")
    lines += [
        "",
        "## 3. 多 Seed 汇总（Test-400）",
        "",
        "| 策略 | Seeds | E2E mean ± std (ms) | LatencyReturn mean ± std | SR | SPL | nDTW |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in multi_seed_rows:
        lines.append(
            f"| {row['policy']} | {row['seeds']} | "
            f"{row['E2E_ms_mean']:.2f} ± {row['E2E_ms_std']:.2f} | "
            f"{row['LatencyReturn_mean']:.5f} ± "
            f"{row['LatencyReturn_std']:.5f} | "
            f"{100*row['SR_mean']:.2f}% | {row['SPL_mean']:.4f} | "
            f"{row['nDTW_mean']:.4f} |"
        )
    lines += [
        "",
        "## 4. 分轨迹长度（Test-400）",
        "",
        "| 策略 | Band | E2E (ms) | LatencyReturn | SR | SPL | nDTW |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in test_band_rows:
        lines.append(
            f"| {row['policy_title']} | {row['band']} | "
            f"{row['E2E_ms']:.2f} | {row['LatencyReturn']:.5f} | "
            f"{100*row['SR']:.2f}% | {row['SPL']:.4f} | "
            f"{row['nDTW']:.4f} |"
        )
    if test_summary:
        audit = test_summary["consistency_audit"]
        lines += [
            "",
            "## 5. 同质 MEC 一致性审计",
            "",
            f"- 参考策略：`{audit['reference_policy']}`",
            f"- 配对任务：{audit['paired_tasks']}",
            f"- 通过：`{audit['passed']}`",
            "",
            "| 策略 | 动作不一致 | SR 不一致 | 步数不一致 | max ΔSPL | max ΔnDTW |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for policy in test_summary["policy_order"]:
            item = audit["per_policy"].get(policy, {})
            lines.append(
                f"| {test_summary['overall'][policy]['title']} | "
                f"{item.get('action_mismatches', '-')} | "
                f"{item.get('success_mismatches', '-')} | "
                f"{item.get('step_mismatches', '-')} | "
                f"{item.get('max_spl_abs_diff', '-')} | "
                f"{item.get('max_ndtw_abs_diff', '-')} |"
            )
        lines += [
            "",
            "同质部署下，20 个节点只有位置不同，且 MEC 输入不含节点身份，",
            "因此所有关联策略的动作序列、SR、SPL、nDTW 必须逐任务完全一致；",
            "只有 E2E 时延允许不同。若审计失败，必须视为隐藏模型差异或实现错误。",
        ]
    lines += [
        "",
        "## 6. 指标角色",
        "",
        "- **主要优化目标**：平均 E2E 时延（ms）与 LatencyReturn。",
        "  两者存在确定线性关系 `LatencyReturn = -0.25 × E2E / 100`，",
        "  不是相互独立的性能证据。",
        "- **导航质量合理性检查**：SR、OSR、SPL、nDTW。 在本同质部署中，",
        "  它们只用于确认 MEC 输入/动作没有因关联策略而改变，不是关联策略的",
        "  优化目标。",
        "- **鲁棒性指标**：P95 E2E 与 100 ms 时延违约率。",
        "",
        "## 7. 解释与限制",
        "",
        "- 20 个节点同权重、同算力、同带宽、同覆盖半径，因此 Nearest、",
        "  Max-Uplink-Rate、Greedy-Min-E2E 的排序相同；这是受控设计的直接",
        "  结果，不应解释为三种启发式“恰好表现一样”。",
        "- 关联策略不改变导航动作，所以 SR/SPL/nDTW 不应用于区分关联算法；",
        "  算法差异只能来自 E2E 时延。",
        "- 单城市、单部署、单 MEC 检查点；结论不外推到异质算力/带宽或新城市。",
        "",
        "## 8. 产物",
        "",
        "- `training_curves.csv`、`validation_curves.csv`：逐 Episode / 逐验证数据",
        "- `convergence.csv`：每次运行的 Dev E2E 收敛判定",
        "- `test_policy_summary.csv`、`test_multi_seed.csv`、`test_by_band.csv`",
        "- `navigation_invariance.csv`：一致性审计",
        "- `figures/*.png`：300 dpi 图",
        "- 原始 checkpoint、TensorBoard、逐 Episode 日志在忽略的 `runs/` 下",
    ]
    return "\n".join(lines) + "\n"


def main():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    series = _training_series()
    if not series:
        raise SystemExit("no Stage-2n training histories found")

    training_rows = []
    validation_rows = []
    for (algorithm, seed), item in sorted(series.items()):
        for index, episode in enumerate(item["episode"]):
            training_rows.append({
                "algorithm": algorithm,
                "seed": int(seed),
                "episode": int(episode),
                "latency_return": float(item["latency_return"][index]),
                "latency_return_ma100": float(
                    item["latency_return_ma100"][index]
                ),
                "mean_e2e_ms": float(item["mean_e2e_ms"][index]),
                "mean_e2e_ms_ma100": float(
                    item["mean_e2e_ms_ma100"][index]
                ),
                "success": float(item["success"][index]),
                "nDTW": float(item["ndtw"][index]),
                "steps": int(item["steps"][index]),
            })
        for row in item["history"].get("validation", []):
            validation_rows.append({
                "algorithm": algorithm,
                "seed": int(seed),
                "episode": int(row["episode"]),
                "E2E_ms": float(row["E2E_ms"]),
                "LatencyReturn": float(row["LatencyReturn"]),
                "SR": float(row["SR"]),
                "OSR": float(row["OSR"]),
                "SPL": float(row["SPL"]),
                "nDTW": float(row["nDTW"]),
                "P95_E2E_ms": float(row["P95_E2E_ms"]),
                "delay_violation_rate": float(row["delay_violation_rate"]),
                "mean_steps": float(row["mean_steps"]),
            })
    write_csv(EVIDENCE_DIR / "training_curves.csv", training_rows)
    write_csv(EVIDENCE_DIR / "validation_curves.csv", validation_rows)

    convergence_rows = _convergence_rows(series)
    write_csv(EVIDENCE_DIR / "convergence.csv", convergence_rows)

    test_summary = read_json(
        Path("runs/v83_spatial_stage2/stage2n_latency_only/"
             "evaluation/test/summary.json")
    )
    dev_summary = read_json(
        Path("runs/v83_spatial_stage2/stage2n_latency_only/"
             "evaluation/dev/summary.json")
    )
    if test_summary and test_summary.get("final_test_read") is not True:
        raise ValueError("Test summary does not record a Final Test read")

    test_policy_rows = []
    test_band_rows = []
    multi_seed_rows = []
    if test_summary:
        for policy in test_summary["policy_order"]:
            item = dict(test_summary["overall"][policy])
            item["policy"] = policy
            test_policy_rows.append(item)
            for band, band_item in test_summary["by_band"][policy].items():
                row = dict(band_item)
                row["policy"] = policy
                row["policy_title"] = test_summary["overall"][policy]["title"]
                row["band"] = band
                test_band_rows.append(row)
        multi_seed_rows = _multi_seed_rows(test_summary)
        write_csv(
            EVIDENCE_DIR / "test_policy_summary.csv", test_policy_rows
        )
        write_csv(
            EVIDENCE_DIR / "test_by_band.csv", test_band_rows
        )
        write_csv(
            EVIDENCE_DIR / "test_multi_seed.csv", multi_seed_rows
        )
        invariance = []
        for policy in test_summary["policy_order"]:
            item = test_summary["consistency_audit"]["per_policy"].get(
                policy, {}
            )
            row = {"policy": policy}
            row.update(item)
            invariance.append(row)
        write_csv(EVIDENCE_DIR / "navigation_invariance.csv", invariance)
        atomic_write_json(
            EVIDENCE_DIR / "test_summary_snapshot.json",
            test_summary,
        )
        atomic_write_json(
            EVIDENCE_DIR / "dev_summary_snapshot.json",
            dev_summary,
        )

    figure_paths = [
        _plot_training(series),
        _plot_training_e2e(series),
        _plot_dev_with_test_refs(series, test_summary),
    ]
    if test_summary:
        fig, labels, values, errors = _plot_test_bar(test_summary)
        figure_paths.append(fig)
        write_csv(
            EVIDENCE_DIR / "figure_test_e2e_by_policy.csv",
            [
                {"policy": label, "E2E_ms": value, "E2E_std_ms": error}
                for label, value, error in zip(labels, values, errors)
            ],
        )
        figure_paths.append(_plot_test_band(test_summary))
        figure_paths.append(_plot_navigation_invariance(test_summary))

    report = _markdown(
        training_rows, convergence_rows, test_summary, multi_seed_rows,
        test_band_rows, series,
    )
    (EVIDENCE_DIR / "STAGE2N_RESULTS.md").write_text(report, encoding="utf-8")
    Path("v83_spatial_stage2/STAGE2N_RESULTS.md").write_text(
        report, encoding="utf-8"
    )
    manifest = {
        "created_utc": utc_now(),
        "stage": STAGE,
        "training_runs": [
            {
                "algorithm": algorithm,
                "seed": seed,
                "episodes": int(len(item["episode"])),
            }
            for (algorithm, seed), item in sorted(series.items())
        ],
        "convergence": convergence_rows,
        "test_read": bool(test_summary and test_summary.get(
            "final_test_read"
        )),
        "figures": [str(path) for path in figure_paths],
        "figure_sources": {
            "figures/training_latency_return.png": [
                "training_curves.csv"
            ],
            "figures/training_mean_e2e.png": ["training_curves.csv"],
            "figures/dev_e2e_with_test_baselines.png": [
                "validation_curves.csv", "test_policy_summary.csv"
            ],
            "figures/test_e2e_by_policy.png": [
                "figure_test_e2e_by_policy.csv", "test_policy_summary.csv"
            ],
            "figures/test_e2e_by_band.png": ["test_by_band.csv"],
            "figures/test_navigation_invariance.png": [
                "navigation_invariance.csv"
            ],
        },
        "evidence_dir": str(EVIDENCE_DIR),
    }
    atomic_write_json(EVIDENCE_DIR / "analysis_manifest.json", manifest)
    print(
        f"[stage2n-analyze] wrote {len(figure_paths)} figures and "
        f"{EVIDENCE_DIR}/STAGE2N_RESULTS.md",
        flush=True,
    )
    print("STAGE2N_ANALYZE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
