"""Plots for the Stage-2k JointR-aligned v2 training run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
GOLD = "#eda100"
GRAY = "#8a8984"
GRID = "#e4e3df"


def moving_average(values, window=100):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return np.asarray([]), np.asarray([])
    kernel = np.ones(window, dtype=np.float64) / window
    return np.arange(window, len(values) + 1), np.convolve(
        values, kernel, mode="valid"
    )


def style(axis, xlabel, ylabel, title):
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=11)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("runs/v83_spatial_stage2/stage2k_jointr_v2"),
    )
    args = parser.parse_args()
    run_dir = args.run_dir
    history = json.loads(
        (run_dir / "training_history.json").read_text(encoding="utf-8")
    )
    train = history["train"]
    validation = history["validation"]
    if not train:
        raise SystemExit("No training rows")
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    episodes = np.asarray([row["episode"] for row in train], dtype=np.int64)
    returns = np.asarray(
        [row["episode_return"] for row in train], dtype=np.float64
    )
    nav = np.asarray(
        [row["navigation_reward"] for row in train], dtype=np.float64
    )
    lat = np.asarray(
        [row["latency_penalty"] for row in train], dtype=np.float64
    )
    loss = np.asarray([row["loss"] for row in train], dtype=np.float64)
    q_mean = np.asarray([row["q_mean"] for row in train], dtype=np.float64)
    q_max = np.asarray([row["q_max"] for row in train], dtype=np.float64)
    errors = np.asarray(
        [row["jointr_abs_error"] for row in train], dtype=np.float64
    )
    val_episodes = np.asarray(
        [row["episode"] for row in validation], dtype=np.int64
    )
    val_joint = np.asarray(
        [row["joint_reward"] for row in validation], dtype=np.float64
    )
    val_sr = np.asarray([row["SR"] for row in validation], dtype=np.float64)
    val_ndtw = np.asarray(
        [row["nDTW"] for row in validation], dtype=np.float64
    )
    val_e2e = np.asarray(
        [row["E2E_ms"] for row in validation], dtype=np.float64
    )
    val_p95 = np.asarray(
        [row["P95_ms"] for row in validation], dtype=np.float64
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1 raw return
    figure, axis = plt.subplots(figsize=(8.6, 4.1), constrained_layout=True)
    axis.plot(episodes, returns, color=BLUE, linewidth=0.7, alpha=0.85)
    axis.axhline(0.0, color=GRAY, linewidth=0.8, linestyle="--")
    style(axis, "training episode", "episode return",
          "Train episode return (raw; Return = JointR per episode)")
    figure.savefig(plot_dir / "dqn_v2_return_raw.png", dpi=200)
    plt.close(figure)

    # 2 moving average
    ma_x, ma_y = moving_average(returns, 100)
    figure, axis = plt.subplots(figsize=(8.6, 4.1), constrained_layout=True)
    axis.plot(episodes, returns, color="#c9c8c2", linewidth=0.7,
              label="raw train return")
    if len(ma_y):
        axis.plot(ma_x, ma_y, color=BLUE, linewidth=2.0,
                  label="100-episode moving average")
        peak = int(np.argmax(ma_y))
        axis.scatter([ma_x[peak]], [ma_y[peak]], color=ORANGE, zorder=4, s=40)
        axis.annotate(
            f"peak {ma_y[peak]:.4f} @ Ep{ma_x[peak]}",
            (ma_x[peak], ma_y[peak]), textcoords="offset points",
            xytext=(8, 8), fontsize=8.5, color=ORANGE,
        )
    axis.axhline(0.0, color=GRAY, linewidth=0.8, linestyle="--")
    style(axis, "training episode", "episode return",
          "Train episode return: raw + 100-episode moving average")
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(plot_dir / "dqn_v2_return_ma100.png", dpi=200)
    plt.close(figure)

    # 3 validation JointR
    figure, axis = plt.subplots(figsize=(8.6, 4.1), constrained_layout=True)
    axis.plot(val_episodes, val_joint, color=BLUE, marker="o",
              linewidth=2.0, markersize=5, label="Dev validation JointR")
    if len(val_joint):
        best = int(np.argmax(val_joint))
        axis.scatter([val_episodes[best]], [val_joint[best]],
                     color=ORANGE, s=70, zorder=5,
                     label=f"best {val_joint[best]:.4f} @ Ep{val_episodes[best]}")
        axis.annotate(
            f"best {val_joint[best]:.4f}",
            (val_episodes[best], val_joint[best]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=8.5, color=ORANGE,
        )
    style(axis, "training episode", "validation JointR",
          "Dev validation JointR (60 Dev episodes, true Dev split)")
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(plot_dir / "dqn_v2_validation_joint_reward.png", dpi=200)
    plt.close(figure)

    # 4 validation metrics
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.2),
                                constrained_layout=True)
    panels = (
        (axes[0, 0], val_sr, "SR", "success rate", GREEN, True),
        (axes[0, 1], val_ndtw, "nDTW", "nDTW", GOLD, False),
        (axes[1, 0], val_e2e, "mean E2E delay", "E2E delay (ms)", ORANGE, False),
        (axes[1, 1], val_joint, "JointR", "JointR", BLUE, False),
    )
    for axis, values, title, ylabel, color, percent in panels:
        axis.plot(val_episodes, values, color=color, marker="o",
                  linewidth=1.8, markersize=4.5)
        axis.set_xticks(val_episodes)
        axis.tick_params(axis="x", labelsize=8, rotation=35)
        if percent:
            axis.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{100*x:.0f}%")
            )
        style(axis, "training episode", ylabel, title)
    figure.suptitle(
        "Dev validation metrics (60 Dev episodes; 6/10/4 per scenario)",
        fontsize=10.5,
    )
    figure.savefig(plot_dir / "dqn_v2_validation_metrics.png", dpi=200)
    plt.close(figure)

    # 5 reward components
    ma_nav_x, ma_nav = moving_average(nav, 100)
    ma_lat_x, ma_lat = moving_average(lat, 100)
    figure, axis = plt.subplots(figsize=(8.6, 4.1), constrained_layout=True)
    axis.plot(ma_nav_x, ma_nav, color=GREEN, linewidth=1.8,
              label="navigation reward (100-MA)")
    axis.plot(ma_lat_x, ma_lat, color=ORANGE, linewidth=1.8,
              label="delay reward (100-MA)")
    axis.axhline(0.0, color=GRAY, linewidth=0.8, linestyle="--")
    style(axis, "training episode", "episode-sum reward",
          "Reward components (sum per episode, 100-episode moving average)")
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(plot_dir / "dqn_v2_reward_components.png", dpi=200)
    plt.close(figure)

    # 6 Q values and loss
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 6.8),
                                constrained_layout=True)
    axes[0].plot(episodes, q_mean, color=BLUE, linewidth=0.8,
                 label="Q mean (100-MA)")
    axes[0].plot(episodes, q_max, color=ORANGE, linewidth=0.8,
                 label="Q max (100-MA)")
    style(axes[0], "training episode", "Q value",
          "Q value mean / max (raw per episode)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(episodes, loss, color=GREEN, linewidth=0.8,
                 label="TD loss (raw per episode)")
    style(axes[1], "training episode", "loss",
          "TD loss (Smooth L1, raw per episode)")
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(plot_dir / "dqn_v2_qvalue_loss.png", dpi=200)
    plt.close(figure)

    summary = {
        "train_episodes": len(train),
        "validation_points": len(validation),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "ma100_peak": float(ma_y.max()) if len(ma_y) else None,
        "ma100_peak_episode": int(ma_x[int(np.argmax(ma_y))])
        if len(ma_y) else None,
        "jointr_abs_error_max": float(errors.max()),
        "jointr_abs_error_mean": float(errors.mean()),
        "best_validation_episode": int(
            val_episodes[int(np.argmax(val_joint))]
        ) if len(val_joint) else None,
        "best_validation_joint": float(val_joint.max())
        if len(val_joint) else None,
        "last_validation_joint": float(val_joint[-1])
        if len(val_joint) else None,
    }
    (plot_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for name in (
        "dqn_v2_return_raw.png",
        "dqn_v2_return_ma100.png",
        "dqn_v2_validation_joint_reward.png",
        "dqn_v2_validation_metrics.png",
        "dqn_v2_reward_components.png",
        "dqn_v2_qvalue_loss.png",
    ):
        print("WROTE", plot_dir / name)


if __name__ == "__main__":
    main()
