"""Summarize the Stage-2m runs and build paired comparisons.

Outputs (under runs/v83_spatial_stage2/stage2m_visual_fixed_city/analysis):
  run_summary.csv
  paired_visual.csv
  dqn_vs_ddqn.csv
  analysis.json
  figures/*.png
  STAGE2M_RESULTS.md

The script tolerates partially finished runs and marks them as in_progress.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import write_csv

RUN_ROOT = (
    PROJECT_ROOT
    / "runs/v83_spatial_stage2/stage2m_visual_fixed_city"
)
DEFAULT_OUTPUT = RUN_ROOT / "analysis"

ARMS = {
    "dqn_pure": ["dqn_pure_s0", "dqn_pure_s1", "dqn_pure_s2"],
    "dqn_visual": ["dqn_visual_s0", "dqn_visual_s1", "dqn_visual_s2"],
    "ppo_pure": ["ppo_pure_s0", "ppo_pure_s1", "ppo_pure_s2"],
    "ppo_visual": ["ppo_visual_s0", "ppo_visual_s1", "ppo_visual_s2"],
    "dqn_double": [
        "dqn_double_s0", "dqn_double_s1", "dqn_double_s2"
    ],
    "dqn_legacy": ["dqn_legacy_s0"],
}


def load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def rolling_mean(values, window=100):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.asarray([])
    output = np.full(len(values), np.nan)
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        if index >= window - 1:
            output[index] = running / window
    return output


def summarize_run(name):
    run_dir = RUN_ROOT / name
    history = load_json(run_dir / "training_history.json") or {}
    progress = load_json(run_dir / "progress.json") or {}
    result = load_json(run_dir / "training_result.json")
    train = history.get("train", [])
    validation = history.get("validation", [])
    episodes = np.asarray(
        [row["episode"] for row in train], dtype=np.int64
    )
    returns = np.asarray(
        [row["episode_return"] for row in train], dtype=np.float64
    )
    jointr = np.asarray(
        [row["episode_jointR"] for row in train], dtype=np.float64
    )
    success = np.asarray(
        [float(row["success"]) for row in train], dtype=np.float64
    )
    ndtw = np.asarray(
        [row["final_nDTW"] for row in train], dtype=np.float64
    )
    val_episodes = np.asarray(
        [row["episode"] for row in validation], dtype=np.int64
    )
    val_jointr = np.asarray(
        [row["joint_reward"] for row in validation], dtype=np.float64
    )
    val_sr = np.asarray([row["SR"] for row in validation], dtype=np.float64)
    val_spl = np.asarray([row["SPL"] for row in validation], dtype=np.float64)
    val_ndtw = np.asarray([row["nDTW"] for row in validation], dtype=np.float64)
    best_index = int(np.argmax(val_jointr)) if len(val_jointr) else None
    best = float(val_jointr[best_index]) if best_index is not None else None
    best_episode = (
        int(val_episodes[best_index]) if best_index is not None else None
    )
    final = float(val_jointr[-1]) if len(val_jointr) else None
    last5 = val_jointr[-5:] if len(val_jointr) >= 5 else val_jointr
    tolerance = 0.01
    first_near_best = None
    if best_index is not None:
        near = np.where(val_jointr >= best - tolerance)[0]
        if len(near):
            first_near_best = int(val_episodes[int(near[0])])
    recent = train[-100:]
    q_mean = float(np.mean([row.get("q_mean", np.nan) for row in recent])) \
        if recent and "q_mean" in recent[0] else None
    q_max = float(np.max([row.get("q_max", np.nan) for row in recent])) \
        if recent and "q_max" in recent[0] else None
    td_loss = float(np.mean([row.get("loss", np.nan) for row in recent])) \
        if recent and "loss" in recent[0] else None
    td_error = float(np.mean(
        [row.get("td_error_abs_mean", np.nan) for row in recent]
    )) if recent and "td_error_abs_mean" in recent[0] else None
    target_mean = float(np.mean(
        [row.get("target_mean", np.nan) for row in recent]
    )) if recent and "target_mean" in recent[0] else None
    entropy = float(np.mean(
        [row.get("policy_entropy", np.nan) for row in recent]
    )) if recent and "policy_entropy" in recent[0] else None
    approx_kl = float(np.mean([
        row.get("approx_kl", 0.0) for row in validation[-5:]
    ])) if validation else None
    value_loss = float(np.mean([
        row.get("update_value_loss", 0.0) for row in validation[-5:]
    ])) if validation else None
    return {
        "name": name,
        "arm": name.rsplit("_s", 1)[0],
        "seed": int(name.rsplit("_s", 1)[1]) if "_s" in name else None,
        "train_episodes": int(len(train)),
        "validation_count": int(len(validation)),
        "status": (progress.get("status") if progress else None),
        "finished": result is not None,
        "result_status": (
            result.get("status", {}).get("status")
            if isinstance(result, dict) else None
        ),
        "converged": (
            result.get("status", {}).get("converged")
            if isinstance(result, dict) else None
        ),
        "final_train_ma100_return": (
            float(np.mean(returns[-100:])) if len(returns) else None
        ),
        "final_train_ma100_jointr": (
            float(np.mean(jointr[-100:])) if len(jointr) else None
        ),
        "final_train_success_ma100": (
            float(np.mean(success[-100:])) if len(success) else None
        ),
        "final_train_ndtw_ma100": (
            float(np.mean(ndtw[-100:])) if len(ndtw) else None
        ),
        "best_validation_jointr": best,
        "best_validation_episode": best_episode,
        "best_validation_SR": (
            float(val_sr[best_index]) if best_index is not None else None
        ),
        "best_validation_SPL": (
            float(val_spl[best_index]) if best_index is not None else None
        ),
        "best_validation_nDTW": (
            float(val_ndtw[best_index]) if best_index is not None else None
        ),
        "final_validation_jointr": final,
        "final_validation_SR": (
            float(val_sr[-1]) if len(val_sr) else None
        ),
        "final_validation_SPL": (
            float(val_spl[-1]) if len(val_spl) else None
        ),
        "final_validation_nDTW": (
            float(val_ndtw[-1]) if len(val_ndtw) else None
        ),
        "last5_jointr_std": (
            float(np.std(last5)) if len(last5) else None
        ),
        "last5_jointr_range": (
            float(np.max(last5) - np.min(last5)) if len(last5) else None
        ),
        "first_episode_within_0.01_of_best": first_near_best,
        "q_mean_last100": q_mean,
        "q_max_last100": q_max,
        "td_loss_last100": td_loss,
        "td_error_abs_last100": td_error,
        "target_mean_last100": target_mean,
        "policy_entropy_last100": entropy,
        "approx_kl_last5": approx_kl,
        "value_loss_last5": value_loss,
        "curves": {
            "episode": episodes.tolist(),
            "return_ma100": rolling_mean(returns).tolist(),
            "jointr_ma100": rolling_mean(jointr).tolist(),
            "success_ma100": rolling_mean(success).tolist(),
            "ndtw_ma100": rolling_mean(ndtw).tolist(),
            "val_episode": val_episodes.tolist(),
            "val_jointr": val_jointr.tolist(),
            "val_SR": val_sr.tolist(),
            "val_SPL": val_spl.tolist(),
            "val_nDTW": val_ndtw.tolist(),
        },
        "train_raw": train,
        "validation_raw": validation,
    }


def bootstrap_ci(values, n_boot=20000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def paired_table(summaries):
    rows = []
    for algorithm in ("dqn", "ppo"):
        pure = {item["seed"]: item for item in summaries
                if item["arm"] == f"{algorithm}_pure"
                and item["finished"]}
        visual = {item["seed"]: item for item in summaries
                  if item["arm"] == f"{algorithm}_visual"
                  and item["finished"]}
        for seed in sorted(set(pure) & set(visual)):
            a, b = pure[seed], visual[seed]
            rows.append({
                "algorithm": algorithm,
                "seed": seed,
                "pure_best_jointr": a["best_validation_jointr"],
                "visual_best_jointr": b["best_validation_jointr"],
                "delta_best_jointr": (
                    b["best_validation_jointr"]
                    - a["best_validation_jointr"]
                    if a["best_validation_jointr"] is not None
                    and b["best_validation_jointr"] is not None else None
                ),
                "delta_best_SR": (
                    b["best_validation_SR"] - a["best_validation_SR"]
                    if a["best_validation_SR"] is not None
                    and b["best_validation_SR"] is not None else None
                ),
                "delta_best_SPL": (
                    b["best_validation_SPL"] - a["best_validation_SPL"]
                    if a["best_validation_SPL"] is not None
                    and b["best_validation_SPL"] is not None else None
                ),
                "delta_best_nDTW": (
                    b["best_validation_nDTW"] - a["best_validation_nDTW"]
                    if a["best_validation_nDTW"] is not None
                    and b["best_validation_nDTW"] is not None else None
                ),
                "delta_train_ma100_jointr": (
                    b["final_train_ma100_jointr"]
                    - a["final_train_ma100_jointr"]
                    if a["final_train_ma100_jointr"] is not None
                    and b["final_train_ma100_jointr"] is not None else None
                ),
                "delta_train_ma100_return": (
                    b["final_train_ma100_return"]
                    - a["final_train_ma100_return"]
                    if a["final_train_ma100_return"] is not None
                    and b["final_train_ma100_return"] is not None else None
                ),
                "delta_final_jointr": (
                    b["final_validation_jointr"]
                    - a["final_validation_jointr"]
                    if a["final_validation_jointr"] is not None
                    and b["final_validation_jointr"] is not None else None
                ),
            })
    return rows


def dqn_comparison(summaries):
    rows = []
    for algorithm, arm_name in (
        ("dqn", "dqn_pure"), ("dqn_double", "dqn_double")
    ):
        for item in summaries:
            if item["arm"] == arm_name and item["finished"]:
                rows.append({
                    "algorithm": algorithm,
                    "seed": item["seed"],
                    "best_jointr": item["best_validation_jointr"],
                    "final_jointr": item["final_validation_jointr"],
                    "train_ma100_jointr": item["final_train_ma100_jointr"],
                    "train_ma100_return": item["final_train_ma100_return"],
                    "best_SR": item["best_validation_SR"],
                    "best_SPL": item["best_validation_SPL"],
                    "best_nDTW": item["best_validation_nDTW"],
                    "q_mean_last100": item["q_mean_last100"],
                    "q_max_last100": item["q_max_last100"],
                    "td_loss_last100": item["td_loss_last100"],
                    "td_error_abs_last100": item["td_error_abs_last100"],
                    "first_episode_within_0.01_of_best": item[
                        "first_episode_within_0.01_of_best"
                    ],
                })
    return rows


def plot_curves(summaries, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def aggregate(arm, key, x_key, reduce="mean"):
        curves = []
        for item in summaries:
            if item["arm"] != arm:
                continue
            x = np.asarray(item["curves"][x_key], dtype=np.float64)
            y = np.asarray(item["curves"][key], dtype=np.float64)
            if len(x):
                curves.append((x, y))
        if not curves:
            return None
        grid = np.unique(np.concatenate([c[0] for c in curves]))
        matrix = np.full((len(curves), len(grid)), np.nan)
        for row, (x, y) in enumerate(curves):
            index = np.searchsorted(grid, x)
            matrix[row, index] = y
        return grid, np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0)

    for panel, (key, x_key, ylabel) in enumerate([
        ("return_ma100", "episode", "training return MA100"),
        ("jointr_ma100", "episode", "training JointR MA100"),
        ("success_ma100", "episode", "training success MA100"),
    ]):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for arm, color in (
            ("dqn_pure", "tab:blue"), ("dqn_visual", "tab:cyan"),
            ("ppo_pure", "tab:red"), ("ppo_visual", "tab:orange"),
            ("dqn_double", "tab:green"),
        ):
            result = aggregate(arm, key, x_key)
            if result is None:
                continue
            grid, mean, std = result
            ax.plot(grid, mean, label=arm, color=color, linewidth=1.4)
            ax.fill_between(grid, mean - std, mean + std,
                            color=color, alpha=0.12)
        ax.set_xlabel("episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"{key}.png", dpi=160)
        plt.close(fig)

    for key, ylabel in (
        ("val_jointr", "Dev60 JointR"),
        ("val_SR", "Dev60 SR"),
        ("val_SPL", "Dev60 SPL"),
        ("val_nDTW", "Dev60 nDTW"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for arm, color in (
            ("dqn_pure", "tab:blue"), ("dqn_visual", "tab:cyan"),
            ("ppo_pure", "tab:red"), ("ppo_visual", "tab:orange"),
            ("dqn_double", "tab:green"),
        ):
            result = aggregate(arm, key, "val_episode")
            if result is None:
                continue
            grid, mean, std = result
            ax.plot(grid, mean, marker="o", markersize=3,
                    label=arm, color=color, linewidth=1.4)
            ax.fill_between(grid, mean - std, mean + std,
                            color=color, alpha=0.12)
        ax.set_xlabel("episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"{key}.png", dpi=160)
        plt.close(fig)

    # DQN Q / TD diagnostics.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, color in (
        ("dqn_pure", "tab:blue"), ("dqn_visual", "tab:cyan"),
        ("dqn_double", "tab:green"), ("dqn_legacy", "tab:gray"),
    ):
        curves = [
            item for item in summaries if item["arm"] == arm
        ]
        if not curves:
            continue
        for item in curves:
            train = item["train_raw"]
            episodes = np.asarray([row["episode"] for row in train])
            q_mean = np.asarray([row.get("q_mean", np.nan) for row in train])
            loss = np.asarray([row.get("loss", np.nan) for row in train])
            if len(episodes):
                axes[0].plot(episodes, rolling_mean(q_mean, 100),
                             color=color, alpha=0.6, linewidth=1.0,
                             label=f"{arm}_s{item['seed']}")
                axes[1].plot(episodes, rolling_mean(loss, 100),
                             color=color, alpha=0.6, linewidth=1.0)
    axes[0].set_ylabel("Q mean MA100")
    axes[1].set_ylabel("TD loss MA100")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("episode")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "dqn_q_td.png", dpi=160)
    plt.close(fig)

    # PPO diagnostics: entropy, KL, value loss, value mean.
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    for arm, color in (
        ("ppo_pure", "tab:red"), ("ppo_visual", "tab:orange"),
    ):
        for item in [x for x in summaries if x["arm"] == arm]:
            train = item["train_raw"]
            episodes = np.asarray([row["episode"] for row in train])
            entropy = np.asarray([
                row.get("policy_entropy", np.nan) for row in train
            ])
            value = np.asarray([
                row.get("value_mean", np.nan) for row in train
            ])
            if len(episodes):
                axes[0].plot(episodes, rolling_mean(entropy, 100),
                             color=color, alpha=0.6, linewidth=1.0,
                             label=f"{arm}_s{item['seed']}")
                axes[3].plot(episodes, rolling_mean(value, 100),
                             color=color, alpha=0.6, linewidth=1.0)
            validation = item["validation_raw"]
            val_episodes = np.asarray([row["episode"] for row in validation])
            kl = np.asarray([row.get("approx_kl", np.nan) for row in validation])
            value_loss = np.asarray([
                row.get("update_value_loss", np.nan) for row in validation
            ])
            if len(val_episodes):
                axes[1].plot(val_episodes, kl, marker="o", markersize=3,
                             color=color, alpha=0.7, linewidth=1.0)
                axes[2].plot(val_episodes, value_loss, marker="o",
                             markersize=3, color=color, alpha=0.7,
                             linewidth=1.0)
    axes[0].set_ylabel("policy entropy MA100")
    axes[1].set_ylabel("approx KL (validation window)")
    axes[2].set_ylabel("value loss (validation window)")
    axes[3].set_ylabel("value mean MA100")
    for ax in axes:
        ax.set_xlabel("episode")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "ppo_diagnostics.png", dpi=160)
    plt.close(fig)


def write_markdown(path, summaries, paired, dqn_rows, output_dir):
    by_arm = {}
    for item in summaries:
        by_arm.setdefault(item["arm"], []).append(item)

    def mean_of(arm, key):
        values = [
            item[key] for item in by_arm.get(arm, [])
            if item[key] is not None
        ]
        return float(np.mean(values)) if values else None

    lines = [
        "# Stage-2m 结果：DQN Q 高估诊断 + 固定城市 A 拓扑视觉增强",
        "",
        "> 本文件由 `stage2m_analyze.py` 自动生成；结论以训练完成后的数据为准。",
        "",
        "## 1. 四组视觉对照（2000 Episodes，3 seeds）",
        "",
        "| Arm | Seeds | Best Dev60 JointR | Best SR | Best SPL | Best nDTW | Final MA100 JointR | Q mean (last100) | TD loss (last100) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("dqn_pure", "dqn_visual", "ppo_pure", "ppo_visual"):
        lines.append(
            f"| {arm} | {len(by_arm.get(arm, []))} | "
            f"{_fmt(mean_of(arm, 'best_validation_jointr'))} | "
            f"{_fmt(mean_of(arm, 'best_validation_SR'))} | "
            f"{_fmt(mean_of(arm, 'best_validation_SPL'))} | "
            f"{_fmt(mean_of(arm, 'best_validation_nDTW'))} | "
            f"{_fmt(mean_of(arm, 'final_train_ma100_jointr'))} | "
            f"{_fmt(mean_of(arm, 'q_mean_last100'))} | "
            f"{_fmt(mean_of(arm, 'td_loss_last100'))} |"
        )
    lines += [
        "",
        "## 2. 配对视觉增益（visual − pure，同 seed）",
        "",
        "| Algorithm | Seed | ΔBest JointR | ΔBest SR | ΔBest SPL | ΔBest nDTW | ΔFinal MA100 JointR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['algorithm']} | {row['seed']} | "
            f"{_fmt(row['delta_best_jointr'])} | "
            f"{_fmt(row['delta_best_SR'])} | "
            f"{_fmt(row['delta_best_SPL'])} | "
            f"{_fmt(row['delta_best_nDTW'])} | "
            f"{_fmt(row['delta_train_ma100_jointr'])} |"
        )
    for algorithm in ("dqn", "ppo"):
        values = [
            row["delta_best_jointr"] for row in paired
            if row["algorithm"] == algorithm
            and row["delta_best_jointr"] is not None
        ]
        mean, low, high = bootstrap_ci(values)
        if mean is not None:
            lines.append(
                f"| {algorithm} 均值 [95% CI] | - | "
                f"{mean:.4f} [{low:.4f}, {high:.4f}] | | | | |"
            )
    lines += [
        "",
        "## 3. DQN vs Double DQN（同数据、同顺序、同 seed）",
        "",
        "| Algorithm | Seed | Best JointR | Final JointR | MA100 JointR | Best SR | Q mean | Q max | TD loss | TD |Q−target| |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dqn_rows:
        lines.append(
            f"| {row['algorithm']} | {row['seed']} | "
            f"{_fmt(row['best_jointr'])} | {_fmt(row['final_jointr'])} | "
            f"{_fmt(row['train_ma100_jointr'])} | {_fmt(row['best_SR'])} | "
            f"{_fmt(row['q_mean_last100'])} | {_fmt(row['q_max_last100'])} | "
            f"{_fmt(row['td_loss_last100'])} | "
            f"{_fmt(row['td_error_abs_last100'])} |"
        )
    lines += [
        "",
        "## 4. 收敛与稳定性",
        "",
        "| Arm | Best Episode (mean) | First within 0.01 of best | Last-5 JointR std | Last-5 range | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm in ("dqn_pure", "dqn_visual", "ppo_pure", "ppo_visual",
                "dqn_double", "dqn_legacy"):
        items = by_arm.get(arm, [])
        if not items:
            continue
        best_episodes = [
            item["best_validation_episode"] for item in items
            if item["best_validation_episode"] is not None
        ]
        near = [
            item["first_episode_within_0.01_of_best"] for item in items
            if item["first_episode_within_0.01_of_best"] is not None
        ]
        lines.append(
            f"| {arm} | {_fmt(np.mean(best_episodes) if best_episodes else None)} | "
            f"{_fmt(np.mean(near) if near else None)} | "
            f"{_fmt(mean_of(arm, 'last5_jointr_std'))} | "
            f"{_fmt(mean_of(arm, 'last5_jointr_range'))} | "
            f"{', '.join(str(item['result_status']) for item in items)} |"
        )
    lines += [
        "",
        "## 5. 曲线",
        "",
        "![training return](figures/return_ma100.png)",
        "",
        "![validation JointR](figures/val_jointr.png)",
        "",
        "![validation SR](figures/val_SR.png)",
        "",
        "![DQN Q and TD](figures/dqn_q_td.png)",
        "",
        "![PPO diagnostics](figures/ppo_diagnostics.png)",
        "",
        "## 6. 口径",
        "",
        "- 训练预算 2000 Episodes，验证每 200 Episodes，固定 Dev60；无早停。",
        "- 四个视觉 arm 共享任务顺序、奖励、拓扑、评估与 seed。",
        "- `Δ` 为同 seed 配对差；均值的 95% CI 由 3 个 seed 的配对差 bootstrap 得到，"
        "样本量很小，只作稳定性参考。",
        "- 收敛判定沿用 `classify_status`；未收敛时明确写 `plateaued_not_converged`。",
        "- 未读取 Final Test，未覆盖任何 Stage-2l 结果。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "—"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for arm, names in ARMS.items():
        for name in names:
            summaries.append(summarize_run(name))
    paired = paired_table(summaries)
    dqn_rows = dqn_comparison(summaries)

    fields = [
        "name", "arm", "seed", "train_episodes", "validation_count",
        "status", "finished", "result_status", "converged",
        "final_train_ma100_return", "final_train_ma100_jointr",
        "final_train_success_ma100", "final_train_ndtw_ma100",
        "best_validation_jointr", "best_validation_episode",
        "best_validation_SR", "best_validation_SPL",
        "best_validation_nDTW", "final_validation_jointr",
        "final_validation_SR", "final_validation_SPL",
        "final_validation_nDTW", "last5_jointr_std",
        "last5_jointr_range", "first_episode_within_0.01_of_best",
        "q_mean_last100", "q_max_last100", "td_loss_last100",
        "td_error_abs_last100", "target_mean_last100",
        "policy_entropy_last100", "approx_kl_last5", "value_loss_last5",
    ]
    write_csv(
        args.output_dir / "run_summary.csv",
        [{key: item.get(key) for key in fields} for item in summaries],
        fieldnames=fields,
    )
    paired_fields = list(paired[0].keys()) if paired else [
        "algorithm", "seed", "delta_best_jointr"
    ]
    write_csv(
        args.output_dir / "paired_visual.csv", paired,
        fieldnames=paired_fields,
    )
    write_csv(
        args.output_dir / "dqn_vs_ddqn.csv", dqn_rows,
        fieldnames=list(dqn_rows[0].keys()) if dqn_rows else ["algorithm"],
    )
    payload = {
        "runs": [
            {key: value for key, value in item.items()
             if key not in ("curves", "train_raw", "validation_raw")}
            for item in summaries
        ],
        "paired_visual": paired,
        "dqn_vs_ddqn": dqn_rows,
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )
    plot_curves(summaries, args.output_dir)
    write_markdown(
        args.output_dir / "STAGE2M_RESULTS.md",
        summaries, paired, dqn_rows, args.output_dir,
    )
    print(json.dumps({
        "output": str(args.output_dir),
        "runs": len(summaries),
        "finished": sum(item["finished"] for item in summaries),
        "paired": len(paired),
    }, indent=2), flush=True)
    print("STAGE2M_ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
