"""Stage-2k V2 A/B real training and Dev60 validation curves.

This script is read-only with respect to training artifacts.  It reads:

  runs/v83_spatial_stage2/stage2k_pure_mec_v2/training_history.json
  runs/v83_spatial_stage2/stage2k_visual_ablation_v2/training_history.json
  runs/.../dev180_best/{summary.json,per_episode.csv}

and writes PNG/SVG curves, per-episode CSVs, a JSON summary and a Chinese
report under ``v83_spatial_stage2/figures/stage2k_v2_ablation/``.

No episode is interpolated or synthesized.  MA100 is defined only after 100
real episodes.  No Final Test is read.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v83_spatial_stage2.stage2k_common import (
    atomic_write_json,
    bootstrap_ci,
    utc_now,
)
from v83_spatial_stage2.stage2k_dqn import dev_validation_specs

A_RUN = Path("runs/v83_spatial_stage2/stage2k_pure_mec_v2")
B_RUN = Path("runs/v83_spatial_stage2/stage2k_visual_ablation_v2")
OUT = Path("v83_spatial_stage2/figures/stage2k_v2_ablation")
REPORT = OUT / "STAGE2K_V2_TRAINING_CURVES_REPORT.md"
SUMMARY = OUT / "stage2k_v2_training_curves_summary.json"

A_COLOR = "#2a78d6"
B_COLOR = "#eb6834"
A_RAW = "#b7d3f6"
B_RAW = "#f6c3b0"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def configure_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "Noto Sans CJK SC",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "savefig.bbox": "tight",
    })


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def moving_average(values, window=100):
    values = np.asarray(values, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) < window:
        return output
    output[window - 1:] = np.convolve(
        values, np.ones(window, dtype=np.float64) / window, mode="valid"
    )
    return output


def load_training(run_dir):
    history = load_json(run_dir / "training_history.json")
    rows = history["train"]
    episode = np.asarray([row["episode"] for row in rows], dtype=np.float64)
    returns = np.asarray(
        [row["episode_return"] for row in rows], dtype=np.float64
    )
    ma100 = moving_average(returns, 100)
    peak_index = int(np.nanargmax(ma100))
    return {
        "run_dir": run_dir,
        "rows": rows,
        "episode": episode,
        "returns": returns,
        "ma100": ma100,
        "peak_index": peak_index,
        "peak_episode": int(episode[peak_index]),
        "peak_ma100": float(ma100[peak_index]),
        "final_ma100": float(ma100[-1]),
        "ma100_at": {
            "1000": float(ma100[999]),
            "2000": float(ma100[1999]),
            "3000": float(ma100[2999]),
            "3500": float(ma100[-1]),
        },
        "first500": float(np.mean(returns[:500])),
        "prev500": float(np.mean(returns[-1000:-500])),
        "last500": float(np.mean(returns[-500:])),
        "epsilon_first": float(rows[0]["epsilon"]),
        "epsilon_final": float(rows[-1]["epsilon"]),
        "epsilon_005_episode": next(
            (int(row["episode"]) for row in rows
             if row["epsilon"] <= 0.0500001),
            None,
        ),
        "q_mean_first": float(rows[0]["q_mean"]),
        "q_mean_200": float(rows[199]["q_mean"]),
        "q_mean_final": float(rows[-1]["q_mean"]),
        "q_mean_max": float(max(row["q_mean"] for row in rows)),
        "q_max_first": float(rows[0]["q_max"]),
        "q_max_200": float(rows[199]["q_max"]),
        "q_max_final": float(rows[-1]["q_max"]),
        "q_max_max": float(max(row["q_max"] for row in rows)),
        "q_mean_ratio_200": float(
            rows[-1]["q_mean"] / max(rows[199]["q_mean"], 1e-9)
        ),
        "q_max_ratio_200": float(
            rows[-1]["q_max"] / max(rows[199]["q_max"], 1e-9)
        ),
        "loss_final": float(rows[-1]["loss"]),
        "loss_max": float(max(row["loss"] for row in rows)),
        "grad_final": float(rows[-1]["grad_norm"]),
        "grad_max": float(max(row["grad_norm"] for row in rows)),
        "validations": history["validation"],
        "config": history["config"],
        "total_steps": int(rows[-1]["total_steps"]),
    }


def plot_single_training(run, label, color, raw_color, path_png, path_svg):
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    ax.plot(
        run["episode"], run["returns"],
        color=raw_color, linewidth=0.55, alpha=0.75,
        label="原始逐 Episode Return (JointR)",
    )
    ax.plot(
        run["episode"], run["ma100"],
        color=color, linewidth=2.3,
        label="100-Episode 移动平均 (MA100)",
    )
    ax.scatter(
        [run["peak_episode"]], [run["peak_ma100"]],
        s=48, color=color, edgecolor=SURFACE, linewidth=1.4, zorder=4,
    )
    ax.annotate(
        f"MA100 峰值 {run['peak_ma100']:.4f} @ Ep{run['peak_episode']}",
        xy=(run["peak_episode"], run["peak_ma100"]),
        xytext=(12, 18), textcoords="offset points",
        fontsize=8, color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7),
    )
    ax.scatter(
        [run["episode"][-1]], [run["final_ma100"]],
        s=44, color=color, edgecolor=SURFACE, linewidth=1.4, zorder=4,
    )
    ax.annotate(
        f"最终 MA100 {run['final_ma100']:.4f}",
        xy=(run["episode"][-1], run["final_ma100"]),
        xytext=(-120, -28), textcoords="offset points",
        fontsize=8, color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7),
    )
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Return = JointR")
    ax.set_xlim(0, 3550)
    ax.set_title(f"{label} — 真实训练奖励曲线（3500 Episodes）")
    ax.legend(loc="upper right")
    ax.text(
        0.01, 0.02,
        "浅色细线：真实逐 Episode 值；深色粗线：真实 MA100（Ep100 起，无插值）",
        transform=ax.transAxes, fontsize=7.5, color=MUTED,
    )
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_svg)
    plt.close(fig)


def plot_overlay(runs, path_png, path_svg):
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for key, run, color in (
        ("A", runs["A"], A_COLOR), ("B", runs["B"], B_COLOR)
    ):
        ax.plot(
            run["episode"], run["ma100"],
            color=color, linewidth=2.3,
            label=f"{key} {'Pure MEC' if key == 'A' else '+128-D Visual'} MA100",
        )
        ax.scatter(
            [run["peak_episode"]], [run["peak_ma100"]],
            s=50, color=color, edgecolor=SURFACE, linewidth=1.4, zorder=4,
        )
        ax.scatter(
            [run["episode"][-1]], [run["final_ma100"]],
            s=44, color=color, edgecolor=SURFACE, linewidth=1.4, zorder=4,
        )
    ax.axhline(0.0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("MA100 of Episode Return = JointR")
    ax.set_xlim(0, 3550)
    ax.set_ylim(-0.01, 0.335)
    ax.set_title("A/B 训练 MA100 叠加对比（统一坐标，真实数据）")
    ax.legend(loc="upper right")
    for y, key, run, color in (
        (0.96, "A", runs["A"], A_COLOR),
        (0.88, "B", runs["B"], B_COLOR),
    ):
        ax.text(
            0.02, y,
            f"{key} MA100 峰值 {run['peak_ma100']:.4f} @ Ep"
            f"{run['peak_episode']}；最终 {run['final_ma100']:.4f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color=color,
            bbox=dict(
                facecolor=SURFACE, edgecolor=GRID,
                boxstyle="round,pad=0.28", alpha=0.92,
            ),
        )
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_svg)
    plt.close(fig)


def plot_validation(runs, path_png, path_svg):
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for key, run, color in (
        ("A", runs["A"], A_COLOR), ("B", runs["B"], B_COLOR)
    ):
        rows = run["validations"]
        episodes = [row["episode"] for row in rows]
        joints = [row["joint_reward"] for row in rows]
        ax.plot(
            episodes, joints, color=color, linewidth=1.8,
            marker="o", markersize=5.2,
            markeredgecolor=SURFACE, markeredgewidth=0.9,
            label=f"{key} {'Pure MEC' if key == 'A' else '+128-D Visual'}",
        )
        best = max(rows, key=lambda row: row["joint_reward"])
        final = rows[-1]
        ax.scatter(
            [best["episode"]], [best["joint_reward"]],
            s=95, marker="*", color=color, edgecolor=SURFACE,
            linewidth=1.2, zorder=5,
        )
        ax.scatter(
            [final["episode"]], [final["joint_reward"]],
            s=55, marker="s", color=color, edgecolor=SURFACE,
            linewidth=1.2, zorder=5,
        )
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Fixed Dev60 Validation JointR")
    ax.set_xlim(0, 3600)
    ax.set_ylim(0.04, 0.235)
    ax.set_title("固定 Dev60 Validation JointR（每 200 Episodes + Ep3500）")
    ax.legend(loc="lower right")
    for y, key, run, color in (
        (0.96, "A", runs["A"], A_COLOR),
        (0.86, "B", runs["B"], B_COLOR),
    ):
        rows = run["validations"]
        best = max(rows, key=lambda row: row["joint_reward"])
        final = rows[-1]
        ax.text(
            0.02, y,
            f"{key} best {best['joint_reward']:.4f} @ Ep{best['episode']}；"
            f"final {final['joint_reward']:.4f} @ Ep{final['episode']}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color=color,
            bbox=dict(
                facecolor=SURFACE, edgecolor=GRID,
                boxstyle="round,pad=0.28", alpha=0.92,
            ),
        )
    ax.text(
        0.01, 0.02,
        "与训练 Reward 分开呈现；每个点对应一次真实 Dev60 评估",
        transform=ax.transAxes, fontsize=7.5, color=MUTED,
    )
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_svg)
    plt.close(fig)


def plot_diagnostics(runs, path_png, path_svg):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0))
    panels = [
        ("epsilon", "ε（实际）", "linear"),
        ("q_mean", "Q mean（symlog）", "symlog"),
        ("q_max", "Q max（symlog）", "symlog"),
        ("loss", "TD Loss（symlog）", "symlog"),
    ]
    for ax, (key, title, scale) in zip(axes.ravel(), panels):
        for name, run, color in (
            ("A", runs["A"], A_COLOR), ("B", runs["B"], B_COLOR)
        ):
            values = np.asarray(
                [row[key] for row in run["rows"]], dtype=np.float64
            )
            ax.plot(
                run["episode"], values,
                color=color, linewidth=1.6,
                label=f"{name} {'Pure MEC' if name == 'A' else '+Visual'}",
            )
        ax.set_title(title)
        ax.set_xlabel("Training Episode")
        if scale != "linear":
            ax.set_yscale(scale, linthresh=1e-3 if key == "loss" else 1.0)
        if key == "epsilon":
            ax.set_ylim(0, 1.05)
            ax.axhline(0.05, color=AXIS, linewidth=0.8, linestyle=":")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(
        "训练稳定性辅助曲线：ε、Q mean、Q max、TD Loss（真实逐 Episode 记录）",
        fontsize=11, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_svg)
    plt.close(fig)


def write_episode_csv(run, path):
    rows = run["rows"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "episode", "episode_uid", "scenario", "band", "steps",
            "episode_return", "episode_jointR", "navigation_reward",
            "latency_penalty", "mean_episode_e2e_ms", "final_nDTW",
            "success", "loss", "epsilon", "q_mean", "q_max", "grad_norm",
            "total_steps", "ma100", "ma100_defined",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, ma in zip(rows, run["ma100"]):
            writer.writerow({
                "episode": row["episode"],
                "episode_uid": row["episode_uid"],
                "scenario": row["scenario"],
                "band": row["band"],
                "steps": row["steps"],
                "episode_return": f"{row['episode_return']:.15g}",
                "episode_jointR": f"{row['episode_jointR']:.15g}",
                "navigation_reward": f"{row['navigation_reward']:.15g}",
                "latency_penalty": f"{row['latency_penalty']:.15g}",
                "mean_episode_e2e_ms": f"{row['mean_episode_e2e_ms']:.15g}",
                "final_nDTW": f"{row['final_nDTW']:.15g}",
                "success": int(row["success"]),
                "loss": f"{row['loss']:.15g}",
                "epsilon": f"{row['epsilon']:.15g}",
                "q_mean": f"{row['q_mean']:.15g}",
                "q_max": f"{row['q_max']:.15g}",
                "grad_norm": f"{row['grad_norm']:.15g}",
                "total_steps": row["total_steps"],
                "ma100": "" if not np.isfinite(ma) else f"{ma:.15g}",
                "ma100_defined": int(np.isfinite(ma)),
            })


def load_eval(run_dir, which):
    summary = load_json(run_dir / f"dev180_{which}" / "summary.json")
    rows = list(csv.DictReader(
        (run_dir / f"dev180_{which}" / "per_episode.csv").open(
            encoding="utf-8"
        )
    ))
    return summary, {row["episode_uid"]: row for row in rows}


def paired_delta(rows_a, rows_b, keys, field):
    values = np.asarray([
        float(rows_b[key][field]) - float(rows_a[key][field])
        for key in keys
    ], dtype=np.float64)
    result = bootstrap_ci(values)
    result.update({
        "n": int(len(values)),
        "wins": int((values > 0).sum()),
        "losses": int((values < 0).sum()),
        "ties": int((values == 0).sum()),
    })
    return result


def fmt_ci(item, digits=4):
    return (
        f"{item['mean']:+.{digits}f} "
        f"[{item['low']:+.{digits}f}, {item['high']:+.{digits}f}]"
    )


def build_summary(runs, eval_a, eval_b):
    validation_uids = {item["episode_uid"] for item in dev_validation_specs()}
    all_keys = sorted(set(eval_a[1]) & set(eval_b[1]))
    held_keys = [key for key in all_keys if key not in validation_uids]
    metrics = {
        "JointR": "joint_reward",
        "SR": "success",
        "SPL": "spl",
        "nDTW": "ndtw",
        "E2E_ms": "mean_e2e_ms",
    }
    paired = {}
    for subset, keys in (("all180", all_keys), ("heldout120", held_keys)):
        paired[subset] = {
            metric: paired_delta(
                eval_a[1], eval_b[1], keys, field
            )
            for metric, field in metrics.items()
        }
    return {
        "version": "stage2k-v2-training-curves-v1",
        "created_utc": utc_now(),
        "data_sources": {
            "A": str(A_RUN / "training_history.json"),
            "B": str(B_RUN / "training_history.json"),
        },
        "episodes": {
            "A": len(runs["A"]["rows"]),
            "B": len(runs["B"]["rows"]),
        },
        "ma100_window": 100,
        "interpolation_used": False,
        "final_test_read": False,
        "training_curves": {
            key: {
                "ma100_peak": run["peak_ma100"],
                "ma100_peak_episode": run["peak_episode"],
                "ma100_final": run["final_ma100"],
                "ma100_at": run["ma100_at"],
                "return_first500_mean": run["first500"],
                "return_prev500_mean": run["prev500"],
                "return_last500_mean": run["last500"],
                "epsilon_first": run["epsilon_first"],
                "epsilon_final": run["epsilon_final"],
                "epsilon_005_episode": run["epsilon_005_episode"],
                "total_steps": run["total_steps"],
                "q_mean_first": run["q_mean_first"],
                "q_mean_200": run["q_mean_200"],
                "q_mean_final": run["q_mean_final"],
                "q_mean_max": run["q_mean_max"],
                "q_mean_ratio_200": run["q_mean_ratio_200"],
                "q_max_first": run["q_max_first"],
                "q_max_200": run["q_max_200"],
                "q_max_final": run["q_max_final"],
                "q_max_max": run["q_max_max"],
                "q_max_ratio_200": run["q_max_ratio_200"],
                "loss_final": run["loss_final"],
                "loss_max": run["loss_max"],
                "grad_final": run["grad_final"],
                "grad_max": run["grad_max"],
            }
            for key, run in runs.items()
        },
        "validation": {
            key: {
                "best_joint": max(
                    row["joint_reward"] for row in run["validations"]
                ),
                "best_episode": max(
                    run["validations"],
                    key=lambda row: row["joint_reward"],
                )["episode"],
                "final_joint": run["validations"][-1]["joint_reward"],
                "final_episode": run["validations"][-1]["episode"],
                "points": run["validations"],
            }
            for key, run in runs.items()
        },
        "dev180_best": {
            "A": eval_a[0]["overall"],
            "B": eval_b[0]["overall"],
            "paired_B_minus_A": paired,
            "dev60_subset_of_dev180": True,
            "validation_episode_count": len(validation_uids),
            "heldout_episode_count": len(held_keys),
        },
        "baselines": eval_a[0].get("baselines", {}),
    }


def report_markdown(summary):
    A = summary["training_curves"]["A"]
    B = summary["training_curves"]["B"]
    VA = summary["validation"]["A"]
    VB = summary["validation"]["B"]
    DA = summary["dev180_best"]["A"]
    DB = summary["dev180_best"]["B"]
    P = summary["dev180_best"]["paired_B_minus_A"]
    all180 = P["all180"]
    held = P["heldout120"]
    lines = [
        "# Stage-2k V2 A/B 真实训练曲线与视觉增强分析",
        "",
        f"- 生成时间（UTC）：{summary['created_utc']}",
        "- 分支：`experiment/v83-spatial-foundation`",
        "- 数据：A=`stage2k_pure_mec_v2`（384-D），B=`stage2k_visual_ablation_v2`（512-D）；"
        "不是 Stage-2l 固定城市实验。",
        "- 训练数据：A/B 各 **3500** 个真实 Episode；没有插值、补点或人工奖励。",
        "- Final Test：**未读取**（Stage-2j Final Test 300 与 Association Final Test 180 均未读取）。",
        "",
        "## 0. 结论摘要",
        "",
        f"- A（Pure MEC）MA100 峰值 **{A['ma100_peak']:.4f} @ Ep{A['ma100_peak_episode']}**，"
        f"最终 MA100 **{A['ma100_final']:.4f}**。",
        f"- B（+128-D Visual）MA100 峰值 **{B['ma100_peak']:.4f} @ Ep{B['ma100_peak_episode']}**，"
        f"最终 MA100 **{B['ma100_final']:.4f}**。",
        f"- Dev60 最佳验证：A **{VA['best_joint']:.4f} @ Ep{VA['best_episode']}**，"
        f"B **{VB['best_joint']:.4f} @ Ep{VB['best_episode']}**；"
        f"最终验证 A {VA['final_joint']:.4f}、B {VB['final_joint']:.4f}。",
        f"- Dev180 best：A **{DA['JointR']:.4f}**，B **{DB['JointR']:.4f}**，"
        f"B−A **{all180['JointR']['mean']:+.4f}**，"
        f"95% CI **[{all180['JointR']['low']:+.4f}, "
        f"{all180['JointR']['high']:+.4f}]**。",
        f"- 但在未参与 checkpoint 选择的 **120** 个 Dev episodes 上，"
        f"ΔJointR 为 **{held['JointR']['mean']:+.4f}**，"
        f"95% CI **[{held['JointR']['low']:+.4f}, "
        f"{held['JointR']['high']:+.4f}]**，跨 0。",
        "- B 的 Q 尺度严重漂移（训练 Q mean Ep200 "
        f"{B['q_mean_200']:.3f}→{B['q_mean_final']:.1f}，约 "
        f"{B['q_mean_ratio_200']:.0f}×），因此即使 Dev180 点估计更高，"
        "也**不能**视为可部署或收敛。",
        "",
        "## 1. 数据与方法",
        "",
        "| 项 | A | B |",
        "|---|---|---|",
        f"| 数据目录 | `{A_RUN}` | `{B_RUN}` |",
        f"| Episode 数 | {summary['episodes']['A']} | {summary['episodes']['B']} |",
        "| 原始 Return | 逐 Episode `episode_return` / `episode_jointR` | 同 A |",
        "| 平滑 | 100-Episode 移动平均（Ep100 起） | 同 A |",
        "| Dev 验证 | 每 200 Episodes + Ep3500，固定 60 个 Dev Episodes | 同 A |",
        "| Dev180 | 180 个 Dev Episodes，best Checkpoint | 同 A |",
        "",
        "MA100 只在 100 个真实 Episode 都存在时计算；前 99 个 Episode 的 MA100 为空，"
        "没有用 0、前向填充或插值补齐。",
        "",
        "## 2. 训练奖励曲线（A、B 分开）",
        "",
        "![A training](stage2k_v2_train_A.png)",
        "",
        "![B training](stage2k_v2_train_B.png)",
        "",
        "| 指标 | A Pure MEC | B +128-D Visual |",
        "|---|---:|---:|",
        f"| MA100 峰值 | **{A['ma100_peak']:.4f}** | **{B['ma100_peak']:.4f}** |",
        f"| 峰值 Episode | {A['ma100_peak_episode']} | {B['ma100_peak_episode']} |",
        f"| 最终 MA100 | **{A['ma100_final']:.4f}** | **{B['ma100_final']:.4f}** |",
        f"| MA100 @ Ep1000 | {A['ma100_at']['1000']:.4f} | {B['ma100_at']['1000']:.4f} |",
        f"| MA100 @ Ep2000 | {A['ma100_at']['2000']:.4f} | {B['ma100_at']['2000']:.4f} |",
        f"| MA100 @ Ep3000 | {A['ma100_at']['3000']:.4f} | {B['ma100_at']['3000']:.4f} |",
        f"| 前 500 Episode 均值 | {A['return_first500_mean']:.4f} | {B['return_first500_mean']:.4f} |",
        f"| 最后 500 Episode 均值 | {A['return_last500_mean']:.4f} | {B['return_last500_mean']:.4f} |",
        "",
        "![MA100 overlay](stage2k_v2_train_ma100_compare.png)",
        "",
        "观察：",
        "",
        f"- 两者 MA100 峰值都出现在早期（A Ep{A['ma100_peak_episode']}、"
        f"B Ep{B['ma100_peak_episode']}），之后都没有保持；"
        "MA100 峰值本身不是收敛。",
        f"- A 的最终 MA100 相对峰值变化 "
        f"{(A['ma100_final']-A['ma100_peak']):+.4f}；B 为 "
        f"{(B['ma100_final']-B['ma100_peak']):+.4f}。",
        f"- B 的 MA100 在 Ep2000 附近降到 {B['ma100_at']['2000']:.4f}，"
        f"随后在 Ep3000–3500 回升到 {B['ma100_final']:.4f}；"
        "训练奖励存在明显的中后期波动。",
        "",
        "## 3. Dev60 Validation JointR（与训练 Reward 分开）",
        "",
        "![Dev60 validation](stage2k_v2_dev60_validation_jointr.png)",
        "",
        "| 指标 | A | B |",
        "|---|---:|---:|",
        f"| 最佳 Dev60 JointR | **{VA['best_joint']:.4f}** | **{VB['best_joint']:.4f}** |",
        f"| 最佳 Episode | {VA['best_episode']} | {VB['best_episode']} |",
        f"| 最终 Dev60 JointR | {VA['final_joint']:.4f} | {VB['final_joint']:.4f} |",
        f"| 最佳−最终 | {VA['best_joint']-VA['final_joint']:.4f} | "
        f"{VB['best_joint']-VB['final_joint']:.4f} |",
        "",
        "验证曲线与训练 MA100 不是同一指标：A 的 MA100 峰值在 Ep"
        f"{A['ma100_peak_episode']}，但 Dev60 最佳在 Ep{VA['best_episode']}；"
        f"B 的 MA100 峰值在 Ep{B['ma100_peak_episode']}，但 Dev60 最佳在 Ep"
        f"{VB['best_episode']}。因此不能用训练 reward 峰值选择 Checkpoint。",
        "",
        "## 4. Dev180 best Checkpoint 与视觉增强",
        "",
        "| 指标 | A best (Ep600) | B best (Ep2400) | B−A | 95% paired CI (all 180) |",
        "|---|---:|---:|---:|---:|",
        f"| JointR | {DA['JointR']:.4f} | {DB['JointR']:.4f} | "
        f"{all180['JointR']['mean']:+.4f} | "
        f"[{all180['JointR']['low']:+.4f}, {all180['JointR']['high']:+.4f}] |",
        f"| SR | {100*DA['SR']:.2f}% | {100*DB['SR']:.2f}% | "
        f"{100*all180['SR']['mean']:+.2f} pp | "
        f"[{100*all180['SR']['low']:+.2f}, {100*all180['SR']['high']:+.2f}] pp |",
        f"| SPL | {DA['SPL']:.4f} | {DB['SPL']:.4f} | "
        f"{all180['SPL']['mean']:+.4f} | "
        f"[{all180['SPL']['low']:+.4f}, {all180['SPL']['high']:+.4f}] |",
        f"| nDTW | {DA['nDTW']:.4f} | {DB['nDTW']:.4f} | "
        f"{all180['nDTW']['mean']:+.4f} | "
        f"[{all180['nDTW']['low']:+.4f}, {all180['nDTW']['high']:+.4f}] |",
        f"| E2E (ms) | {DA['E2E_ms']:.2f} | {DB['E2E_ms']:.2f} | "
        f"{all180['E2E_ms']['mean']:+.2f} ms | "
        f"[{all180['E2E_ms']['low']:+.2f}, {all180['E2E_ms']['high']:+.2f}] ms |",
        "",
        "**关键区分：数值提升 vs 统计显著提升**",
        "",
        f"- 在全部 180 个 Dev episodes 上，B−A 的 JointR 为 "
        f"**{all180['JointR']['mean']:+.4f}**，CI "
        f"[{all180['JointR']['low']:+.4f}, {all180['JointR']['high']:+.4f}]，"
        "下界刚高于 0，属于**边缘显著**；",
        f"- 但固定 Dev60 是这 180 个 episodes 的**子集**，且用于选择 best Checkpoint。"
        f"剔除这 60 个 episodes 后，剩下 120 个 episodes 上 B−A JointR 为 "
        f"**{held['JointR']['mean']:+.4f}**，CI "
        f"[{held['JointR']['low']:+.4f}, {held['JointR']['high']:+.4f}]，**跨 0**；",
        f"- nDTW 在全部 180 上 CI "
        f"[{all180['nDTW']['low']:+.4f}, {all180['nDTW']['high']:+.4f}] 跨 0；",
        f"- E2E 的下降在全部 180 和 120 上都显著（120 上 CI "
        f"[{held['E2E_ms']['low']:+.2f}, {held['E2E_ms']['high']:+.2f}] ms）；",
        "- 因此：视觉增强的**点估计**为正，SR/SPL/E2E 有可检测改善，"
        "但 Navigation JointR 的提升在剔除选择重叠后**不稳健显著**；"
        "不能把它当成普遍提升。",
        "",
        f"同一 Dev180 的非特权 Baseline JointR：Random "
        f"{summary['baselines'].get('random', {}).get('JointR', float('nan')):.4f}、"
        f"Min-Latency {summary['baselines'].get('min_latency', {}).get('JointR', float('nan')):.4f}、"
        f"Max-Quality {summary['baselines'].get('max_quality', {}).get('JointR', float('nan')):.4f}、"
        f"Heuristic {summary['baselines'].get('heuristic', {}).get('JointR', float('nan')):.4f}。"
        "A/B 的 best 都低于这些点估计。",
        "",
        "## 5. 训练稳定性：ε、Q 与 TD Loss",
        "",
        "![diagnostics](stage2k_v2_train_diagnostics.png)",
        "",
        "| 指标 | A | B |",
        "|---|---:|---:|",
        f"| ε 首/末 | {A['epsilon_first']:.4f} / {A['epsilon_final']:.4f} | "
        f"{B['epsilon_first']:.4f} / {B['epsilon_final']:.4f} |",
        f"| ε 首次到 0.05 的 Episode | {A['epsilon_005_episode'] or '—'} | "
        f"{B['epsilon_005_episode'] or '—'} |",
        f"| 环境交互步 | {A['total_steps']:,} | {B['total_steps']:,} |",
        f"| Q mean Ep200→末（倍数） | {A['q_mean_200']:.3f}→{A['q_mean_final']:.2f}"
        f"（{A['q_mean_ratio_200']:.1f}×） | "
        f"{B['q_mean_200']:.3f}→{B['q_mean_final']:.1f}"
        f"（{B['q_mean_ratio_200']:.1f}×） |",
        f"| Q mean 峰值 | {A['q_mean_max']:.2f} | {B['q_mean_max']:.1f} |",
        f"| Q max 末值 | {A['q_max_final']:.2f} | {B['q_max_final']:.1f} |",
        f"| TD Loss 末值 | {A['loss_final']:.4f} | {B['loss_final']:.2f} |",
        f"| 梯度范数末值 | {A['grad_final']:.3f} | {B['grad_final']:.1f} |",
        "",
        f"A 的 Q mean 从 Ep200 的 {A['q_mean_200']:.3f} 增长到 "
        f"{A['q_mean_final']:.2f}（约 {A['q_mean_ratio_200']:.0f}×）；"
        f"B 从 {B['q_mean_200']:.3f} 增长到 {B['q_mean_final']:.1f}"
        f"（约 {B['q_mean_ratio_200']:.0f}×），"
        f"TD Loss 到 {B['loss_final']:.2f}，梯度范数到 {B['grad_final']:.1f}。"
        "这属于明显的 Q 过估计/尺度漂移，不是收敛。"
        "预注册的 1e6 数值中止阈值没有触发，但该阈值对 B 的尺度漂移不敏感；"
        "报告据此把 B 标记为不稳定、不可部署。",
        "",
        "## 6. 解释与限制",
        "",
        "- 视觉增强（B）在 Dev180 best 上的 JointR 点估计高于 A，但提升在剔除"
        "Dev60 选择重叠后不显著；不能声称视觉特征带来稳健的导航收益。",
        "- B 的 SR/SPL/E2E 改善更稳定（SR/SPL 的 180-CI 下界 >0，E2E 的 180/120-CI 均 <0），"
        "但 nDTW 与 held-out JointR 不支持“全面视觉增益”。",
        "- B 的 Q 尺度严重漂移，best/last 策略差异大；即使点估计更高，也不满足部署或"
        "Final Test 条件。",
        "- A/B 都低于所有非特权 Baseline 的点估计；本报告不使用 Final Test，"
        "因此不给出测试集结论。",
        "- 单 seed、180 个 Dev episodes、MA100 依赖 Episode 顺序/任务组成；"
        "这些都不构成统计显著性的强证据。",
        "",
        "## 7. 产物",
        "",
        f"- 报告：`{REPORT}`",
        f"- 摘要 JSON：`{SUMMARY}`",
        f"- 逐 Episode CSV：`{OUT / 'stage2k_v2_train_A_episodes.csv'}`、"
        f"`{OUT / 'stage2k_v2_train_B_episodes.csv'}`",
        "- 曲线 PNG/SVG：`stage2k_v2_train_A.*`、`stage2k_v2_train_B.*`、"
        "`stage2k_v2_train_ma100_compare.*`、`stage2k_v2_dev60_validation_jointr.*`、"
        "`stage2k_v2_train_diagnostics.*`",
        "",
        "所有原始训练日志、Checkpoint、TensorBoard 和旧实验均未修改或覆盖。",
    ]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    runs = {"A": load_training(A_RUN), "B": load_training(B_RUN)}
    eval_a = load_eval(A_RUN, "best")
    eval_b = load_eval(B_RUN, "best")
    summary = build_summary(runs, eval_a, eval_b)

    plot_single_training(
        runs["A"], "A: Pure MEC (384-D)", A_COLOR, A_RAW,
        OUT / "stage2k_v2_train_A.png", OUT / "stage2k_v2_train_A.svg",
    )
    plot_single_training(
        runs["B"], "B: Pure MEC + 128-D Visual (512-D)", B_COLOR, B_RAW,
        OUT / "stage2k_v2_train_B.png", OUT / "stage2k_v2_train_B.svg",
    )
    plot_overlay(
        runs,
        OUT / "stage2k_v2_train_ma100_compare.png",
        OUT / "stage2k_v2_train_ma100_compare.svg",
    )
    plot_validation(
        runs,
        OUT / "stage2k_v2_dev60_validation_jointr.png",
        OUT / "stage2k_v2_dev60_validation_jointr.svg",
    )
    plot_diagnostics(
        runs,
        OUT / "stage2k_v2_train_diagnostics.png",
        OUT / "stage2k_v2_train_diagnostics.svg",
    )
    write_episode_csv(runs["A"], OUT / "stage2k_v2_train_A_episodes.csv")
    write_episode_csv(runs["B"], OUT / "stage2k_v2_train_B_episodes.csv")
    atomic_write_json(SUMMARY, summary)
    REPORT.write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "report": str(REPORT),
        "summary": str(SUMMARY),
        "A": {
            "ma100_peak": summary["training_curves"]["A"]["ma100_peak"],
            "ma100_peak_episode": summary["training_curves"]["A"]["ma100_peak_episode"],
            "ma100_final": summary["training_curves"]["A"]["ma100_final"],
            "best_dev60": summary["validation"]["A"]["best_joint"],
            "final_dev60": summary["validation"]["A"]["final_joint"],
        },
        "B": {
            "ma100_peak": summary["training_curves"]["B"]["ma100_peak"],
            "ma100_peak_episode": summary["training_curves"]["B"]["ma100_peak_episode"],
            "ma100_final": summary["training_curves"]["B"]["ma100_final"],
            "best_dev60": summary["validation"]["B"]["best_joint"],
            "final_dev60": summary["validation"]["B"]["final_joint"],
        },
        "delta_best_jointr_all180": summary["dev180_best"][
            "paired_B_minus_A"
        ]["all180"]["JointR"],
        "delta_best_jointr_heldout120": summary["dev180_best"][
            "paired_B_minus_A"
        ]["heldout120"]["JointR"],
        "final_test_read": False,
    }, indent=2, ensure_ascii=False), flush=True)
    print("STAGE2K_V2_TRAINING_CURVES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
