"""Read-only convergence plots for the Stage-2k DQN audit.

Reads only ``training_history.json`` (and cross-checks the original
TensorBoard scalars).  No training, no checkpoint changes, no interpolation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import DQN_DIR, TENSORBOARD_DIR

PLOT_DIR = DQN_DIR.parent / "plots"
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


def mark_checkpoints(axis, ymin=None, ymax=None):
    axis.axvline(800, color=GREEN, linewidth=1.2, linestyle="--",
                 label="best checkpoint Ep800")
    axis.axvline(2000, color=GRAY, linewidth=1.2, linestyle=":",
                 label="last checkpoint Ep2000")


def tb_crosscheck(history):
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception as error:
        return {"skipped": repr(error)}
    accumulator = EventAccumulator(
        str(TENSORBOARD_DIR), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    checks = (
        ("train/episode_return", "return", history["train"]),
        ("train/navigation_reward", "navigation_reward", history["train"]),
        ("train/latency_penalty", "latency_penalty", history["train"]),
        ("train/epsilon", "epsilon", history["train"]),
        ("train/loss", "loss", history["train"]),
        ("validation/joint_reward", "joint_reward", history["validation"]),
        ("validation/SR", "SR", history["validation"]),
        ("validation/nDTW", "nDTW", history["validation"]),
        ("validation/E2E_ms", "E2E_ms", history["validation"]),
    )
    output = {}
    for tag, key, rows in checks:
        events = accumulator.Scalars(tag)
        recorded = np.asarray([row[key] for row in rows], dtype=np.float64)
        tb_values = np.asarray([event.value for event in events], dtype=np.float64)
        steps = np.asarray([event.step for event in events], dtype=np.int64)
        expected = np.asarray([row["episode"] for row in rows], dtype=np.int64)
        output[tag] = {
            "n": len(events),
            "steps_match": bool(np.array_equal(steps, expected)),
            "max_abs_diff": float(np.max(np.abs(recorded - tb_values)))
            if len(events) == len(recorded) else None,
        }
    return output


def block_stats(rows, low, high):
    selected = [row for row in rows if low <= row["episode"] <= high]
    if not selected:
        return None
    output = {"n": len(selected)}
    for key in (
        "return", "navigation_reward", "latency_penalty", "mean_e2e_ms",
        "epsilon", "loss",
    ):
        values = np.asarray([row[key] for row in selected], dtype=np.float64)
        output[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    output["success"] = float(np.mean([
        float(row["success"]) for row in selected
    ]))
    output["steps"] = float(np.mean([
        float(row["steps"]) for row in selected
    ]))
    return output


def main():
    history = json.loads(
        (DQN_DIR / "training_history.json").read_text(encoding="utf-8")
    )
    train = history["train"]
    validation = history["validation"]
    if not train or not validation:
        raise SystemExit("Missing training/validation rows")

    episodes = np.asarray([row["episode"] for row in train], dtype=np.int64)
    returns = np.asarray([row["return"] for row in train], dtype=np.float64)
    nav = np.asarray(
        [row["navigation_reward"] for row in train], dtype=np.float64
    )
    lat = np.asarray([row["latency_penalty"] for row in train], dtype=np.float64)
    steps = np.asarray([row["steps"] for row in train], dtype=np.float64)
    e2e = np.asarray([row["mean_e2e_ms"] for row in train], dtype=np.float64)
    loss = np.asarray([row["loss"] for row in train], dtype=np.float64)
    epsilon = np.asarray([row["epsilon"] for row in train], dtype=np.float64)

    val_episodes = np.asarray(
        [row["episode"] for row in validation], dtype=np.int64
    )
    val_joint = np.asarray(
        [row["joint_reward"] for row in validation], dtype=np.float64
    )
    val_sr = np.asarray([row["SR"] for row in validation], dtype=np.float64)
    val_ndtw = np.asarray([row["nDTW"] for row in validation], dtype=np.float64)
    val_e2e = np.asarray([row["E2E_ms"] for row in validation], dtype=np.float64)
    best_index = int(np.argmax(val_joint))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. raw training return
    figure, axis = plt.subplots(figsize=(8.6, 4.1), constrained_layout=True)
    axis.plot(episodes, returns, color=BLUE, linewidth=0.7, alpha=0.85)
    axis.axhline(0.0, color=GRAY, linewidth=0.8, linestyle="--")
    mark_checkpoints(axis)
    style(
        axis, "training episode", "episode return",
        "Train episode return (raw, 2000 episodes; no smoothing)",
    )
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(PLOT_DIR / "dqn_reward_raw.png", dpi=200)
    plt.close(figure)

    # 2. raw + 100-episode moving average
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
    mark_checkpoints(axis)
    style(
        axis, "training episode", "episode return",
        "Train episode return: raw + 100-episode moving average",
    )
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(PLOT_DIR / "dqn_reward_ma100.png", dpi=200)
    plt.close(figure)

    # 3. validation JointR with Ep800/Ep2000 markers
    figure, axis = plt.subplots(figsize=(8.6, 4.1), constrained_layout=True)
    axis.plot(val_episodes, val_joint, color=BLUE, marker="o",
              linewidth=2.0, markersize=5, label="training-time validation JointR")
    axis.axhline(val_joint[best_index], color=ORANGE, linewidth=1.0,
                 linestyle="--", label=f"best {val_joint[best_index]:.4f} @ Ep800")
    axis.scatter([800], [val_joint[best_index]], color=GREEN, s=70, zorder=5,
                 label="Ep800 best checkpoint")
    axis.scatter([2000], [val_joint[-1]], color=GRAY, s=55, zorder=5,
                 label="Ep2000 last checkpoint")
    axis.annotate(
        f"Ep800: {val_joint[best_index]:.4f}",
        (800, val_joint[best_index]), textcoords="offset points",
        xytext=(8, 8), fontsize=8.5, color=GREEN,
    )
    axis.annotate(
        f"Ep2000: {val_joint[-1]:.4f}",
        (2000, val_joint[-1]), textcoords="offset points",
        xytext=(-55, -15), fontsize=8.5, color=GRAY,
    )
    style(
        axis, "training episode", "validation JointR",
        "Training-time validation JointR\n(60 train episodes + dev topologies; NOT Dev Val)",
    )
    axis.legend(frameon=False, fontsize=7.5, loc="lower left")
    figure.savefig(PLOT_DIR / "dqn_validation_joint_reward.png", dpi=200)
    plt.close(figure)

    # 4. four validation metrics
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
        axis.axvline(800, color=GREEN, linewidth=1.0, linestyle="--")
        axis.axvline(2000, color=GRAY, linewidth=1.0, linestyle=":")
        axis.set_xticks(val_episodes)
        axis.tick_params(axis="x", labelsize=8, rotation=35)
        if percent:
            axis.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{100*x:.0f}%")
            )
        style(axis, "training episode", ylabel, title)
    figure.suptitle(
        "Training-time validation metrics (60 train episodes + dev topologies; "
        "NOT Dev Val)\nEp800 = best checkpoint; Ep2000 = last checkpoint",
        fontsize=10.5,
    )
    figure.savefig(PLOT_DIR / "dqn_validation_four_metrics.png", dpi=200)
    plt.close(figure)

    # 5. reward components (only the two recorded components)
    ma_nav_x, ma_nav = moving_average(nav, 100)
    ma_lat_x, ma_lat = moving_average(lat, 100)
    per_step_nav = nav / np.maximum(steps, 1)
    per_step_lat = lat / np.maximum(steps, 1)
    ma_psn_x, ma_psn = moving_average(per_step_nav, 100)
    ma_psl_x, ma_psl = moving_average(per_step_lat, 100)
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0),
                                constrained_layout=True)
    axes[0].plot(ma_nav_x, ma_nav, color=GREEN, linewidth=1.8,
                 label="navigation reward (episode sum, 100-MA)")
    axes[0].plot(ma_lat_x, ma_lat, color=ORANGE, linewidth=1.8,
                 label="latency penalty (episode sum, 100-MA)")
    axes[0].axhline(0.0, color=GRAY, linewidth=0.8, linestyle="--")
    axes[0].axvline(800, color=GREEN, linewidth=1.0, linestyle="--")
    axes[0].axvline(2000, color=GRAY, linewidth=1.0, linestyle=":")
    style(
        axes[0], "training episode", "episode-level reward",
        "Recorded reward components (episode sums, 100-episode moving average)",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(ma_psn_x, ma_psn, color=GREEN, linewidth=1.8,
                 label="navigation reward / step (100-MA)")
    axes[1].plot(ma_psl_x, ma_psl, color=ORANGE, linewidth=1.8,
                 label="latency penalty / step (100-MA)")
    axes[1].axhline(0.0, color=GRAY, linewidth=0.8, linestyle="--")
    axes[1].axvline(800, color=GREEN, linewidth=1.0, linestyle="--")
    axes[1].axvline(2000, color=GRAY, linewidth=1.0, linestyle=":")
    style(
        axes[1], "training episode", "per-step reward",
        "Recorded reward components (per step, 100-episode moving average)",
    )
    axes[1].legend(frameon=False, fontsize=8)
    figure.text(
        0.5, -0.01,
        "Only navigation_reward and latency_penalty were recorded. "
        "No success bonus, failure penalty or other reward component exists "
        "in training_history.json.",
        ha="center", fontsize=7.5, color="#52514e",
    )
    figure.savefig(PLOT_DIR / "dqn_reward_components.png", dpi=200)
    plt.close(figure)

    # Summary JSON for the report
    summary = {
        "source": str(DQN_DIR / "training_history.json"),
        "train_episodes": int(len(train)),
        "validation_points": int(len(validation)),
        "train_return": {
            "mean": float(returns.mean()),
            "std": float(returns.std()),
            "min": float(returns.min()),
            "max": float(returns.max()),
            "ma100_peak": float(ma_y.max()) if len(ma_y) else None,
            "ma100_peak_episode": int(ma_x[int(np.argmax(ma_y))])
            if len(ma_y) else None,
            "ma100_last": float(ma_y[-1]) if len(ma_y) else None,
        },
        "validation": {
            "episodes": val_episodes.tolist(),
            "joint_reward": val_joint.tolist(),
            "SR": val_sr.tolist(),
            "nDTW": val_ndtw.tolist(),
            "E2E_ms": val_e2e.tolist(),
            "best_episode": int(val_episodes[best_index]),
            "best_joint": float(val_joint[best_index]),
            "last_episode": int(val_episodes[-1]),
            "last_joint": float(val_joint[-1]),
        },
        "blocks": {
            f"{low}-{high}": block_stats(train, low, high)
            for low, high in ((1, 500), (501, 1000), (1001, 1500), (1501, 2000))
        },
        "windows": {
            "700-900": block_stats(train, 700, 900),
            "1801-2000": block_stats(train, 1801, 2000),
        },
        "validation_model_usage": {
            str(row["episode"]): row["model_usage"] for row in validation
        },
        "tensorboard_crosscheck": tb_crosscheck(history),
        "note": (
            "The validation points were computed by the training script from "
            "load_split_specs('train') (60 train episodes) with dev "
            "topologies. They are not Dev Val metrics."
        ),
    }
    (PLOT_DIR / "convergence_curve_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps({
        "train_return": summary["train_return"],
        "validation": {
            "best_episode": summary["validation"]["best_episode"],
            "best_joint": summary["validation"]["best_joint"],
            "last_episode": summary["validation"]["last_episode"],
            "last_joint": summary["validation"]["last_joint"],
        },
        "windows": summary["windows"],
        "blocks": summary["blocks"],
        "tensorboard_crosscheck": summary["tensorboard_crosscheck"],
    }, indent=2, ensure_ascii=False))
    for name in (
        "dqn_reward_raw.png",
        "dqn_reward_ma100.png",
        "dqn_validation_joint_reward.png",
        "dqn_validation_four_metrics.png",
        "dqn_reward_components.png",
    ):
        path = PLOT_DIR / name
        print("WROTE", path, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
