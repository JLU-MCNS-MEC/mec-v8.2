"""Redraw the Stage-2k DQN training curves from recorded data only.

Reads ``runs/v83_spatial_stage2/stage2k/dqn/training_history.json`` and
cross-checks it against the original TensorBoard event file.  No training is
started, no checkpoint is touched, and no missing value is filled in.
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


def _style_axis(axis, xlabel, ylabel, title):
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=11)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return np.asarray([]), np.asarray([])
    kernel = np.ones(window, dtype=np.float64) / window
    return (
        np.arange(window, len(values) + 1),
        np.convolve(values, kernel, mode="valid"),
    )


def _read_tensorboard():
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception as error:  # pragma: no cover - optional dependency
        return None, f"TensorBoard reader unavailable: {error!r}"
    event_dir = Path(TENSORBOARD_DIR)
    if not event_dir.is_dir():
        return None, "TensorBoard directory is missing"
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    payload = {}
    for tag in accumulator.Tags().get("scalars", []):
        payload[tag] = [
            (int(event.step), float(event.value))
            for event in accumulator.Scalars(tag)
        ]
    return payload, None


def _compare_with_tensorboard(history, tensorboard):
    report = []
    if tensorboard is None:
        return ["TensorBoard cross-check skipped"]
    checks = (
        ("train/episode_return", "return", history["train"]),
        ("train/navigation_reward", "navigation_reward", history["train"]),
        ("train/latency_penalty", "latency_penalty", history["train"]),
        ("train/mean_e2e_ms", "mean_e2e_ms", history["train"]),
        ("train/epsilon", "epsilon", history["train"]),
        ("validation/joint_reward", "joint_reward", history["validation"]),
        ("validation/SR", "SR", history["validation"]),
        ("validation/nDTW", "nDTW", history["validation"]),
        ("validation/E2E_ms", "E2E_ms", history["validation"]),
        ("validation/p95_ms", "p95_ms", history["validation"]),
    )
    for tag, key, rows in checks:
        events = tensorboard.get(tag, [])
        if len(events) != len(rows):
            report.append(f"{tag}: history={len(rows)} tb={len(events)} (count mismatch)")
            continue
        recorded = np.asarray([row[key] for row in rows], dtype=np.float64)
        tb_values = np.asarray([value for _, value in events], dtype=np.float64)
        tb_steps = np.asarray([step for step, _ in events], dtype=np.int64)
        history_steps = np.asarray([row["episode"] for row in rows], dtype=np.int64)
        step_ok = bool(np.array_equal(tb_steps, history_steps))
        diff = float(np.max(np.abs(recorded - tb_values))) if len(recorded) else 0.0
        report.append(f"{tag}: n={len(rows)} steps_match={step_ok} max_abs_diff={diff:.3g}")
    return report


def main():
    history_path = DQN_DIR / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    train = history["train"]
    validation = history["validation"]
    if not train:
        raise SystemExit("No training rows recorded")
    if not validation:
        raise SystemExit("No validation rows recorded")

    required_train = (
        "episode", "return", "navigation_reward", "latency_penalty",
        "mean_e2e_ms", "epsilon", "loss", "success", "steps",
    )
    required_val = (
        "episode", "joint_reward", "SR", "nDTW", "E2E_ms", "p95_ms",
    )
    missing_train = sorted({key for key in required_train if key not in train[0]})
    missing_val = sorted({key for key in required_val if key not in validation[0]})
    if missing_train or missing_val:
        raise SystemExit(
            f"Missing fields: train={missing_train} validation={missing_val}"
        )

    episodes = np.asarray([row["episode"] for row in train], dtype=np.int64)
    returns = np.asarray([row["return"] for row in train], dtype=np.float64)
    val_episodes = np.asarray(
        [row["episode"] for row in validation], dtype=np.int64
    )
    val_joint = np.asarray(
        [row["joint_reward"] for row in validation], dtype=np.float64
    )
    val_sr = np.asarray([row["SR"] for row in validation], dtype=np.float64)
    val_ndtw = np.asarray([row["nDTW"] for row in validation], dtype=np.float64)
    val_e2e = np.asarray([row["E2E_ms"] for row in validation], dtype=np.float64)
    val_p95 = np.asarray([row["p95_ms"] for row in validation], dtype=np.float64)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Raw episode return
    figure, axis = plt.subplots(figsize=(8.4, 4.0), constrained_layout=True)
    axis.plot(episodes, returns, color=BLUE, linewidth=0.8, alpha=0.85)
    axis.axhline(0.0, color=GRAY, linewidth=0.8, linestyle="--")
    _style_axis(
        axis, "training episode", "episode return",
        "DQN episode return (recorded, no smoothing)",
    )
    figure.savefig(PLOT_DIR / "dqn_episode_return.png", dpi=200)
    plt.close(figure)

    # 2. 100-episode trailing moving average
    ma_episodes, ma_values = _moving_average(returns, 100)
    figure, axis = plt.subplots(figsize=(8.4, 4.0), constrained_layout=True)
    axis.plot(episodes, returns, color="#c9c8c2", linewidth=0.7,
              label="raw return")
    if len(ma_values):
        axis.plot(ma_episodes, ma_values, color=BLUE, linewidth=2.0,
                  label="100-episode moving average")
    axis.axhline(0.0, color=GRAY, linewidth=0.8, linestyle="--")
    _style_axis(
        axis, "training episode", "episode return",
        "DQN return: 100-episode moving average",
    )
    axis.legend(frameon=False, fontsize=9)
    figure.savefig(PLOT_DIR / "dqn_return_moving_average_100.png", dpi=200)
    plt.close(figure)

    # 3. Validation JointR
    best_index = int(np.argmax(val_joint))
    figure, axis = plt.subplots(figsize=(8.4, 4.0), constrained_layout=True)
    axis.plot(val_episodes, val_joint, color=BLUE, marker="o",
              linewidth=2.0, markersize=5, label="validation JointR")
    axis.axhline(val_joint[best_index], color=ORANGE, linewidth=1.0,
                 linestyle="--", label=f"best = {val_joint[best_index]:.4f}")
    axis.scatter(
        [val_episodes[best_index]], [val_joint[best_index]],
        color=ORANGE, zorder=4, s=45,
    )
    axis.annotate(
        f"Ep{val_episodes[best_index]}: {val_joint[best_index]:.4f}",
        (val_episodes[best_index], val_joint[best_index]),
        textcoords="offset points", xytext=(8, 8), fontsize=9, color=ORANGE,
    )
    _style_axis(
        axis, "training episode", "validation JointR",
        "DQN validation JointR (fixed 60 Dev episodes)",
    )
    axis.legend(frameon=False, fontsize=9)
    figure.savefig(PLOT_DIR / "dqn_validation_joint_reward.png", dpi=200)
    plt.close(figure)

    # 4. Validation metrics: SR, nDTW, mean E2E, P95 E2E
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.0),
                                constrained_layout=True)
    panels = (
        (axes[0, 0], val_sr, "SR", "success rate", GREEN, True),
        (axes[0, 1], val_ndtw, "nDTW", "nDTW", GOLD, False),
        (axes[1, 0], val_e2e, "mean E2E delay", "E2E delay (ms)", ORANGE, False),
        (axes[1, 1], val_p95, "P95 E2E delay", "P95 E2E delay (ms)", GRAY, False),
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
        _style_axis(axis, "training episode", ylabel, title)
    figure.suptitle(
        "DQN validation metrics (10 fixed checkpoints, 60 Dev episodes each)",
        fontsize=11,
    )
    figure.savefig(PLOT_DIR / "dqn_validation_metrics.png", dpi=200)
    plt.close(figure)

    tensorboard, tensorboard_error = _read_tensorboard()
    if tensorboard_error:
        print(tensorboard_error)
    print("Training rows:", len(train), "episodes",
          int(episodes.min()), "->", int(episodes.max()))
    print("Validation rows:", len(validation), "episodes",
          int(val_episodes.min()), "->", int(val_episodes.max()))
    print("Best validation JointR:", float(val_joint[best_index]),
          "at episode", int(val_episodes[best_index]))
    for line in _compare_with_tensorboard(history, tensorboard):
        print("TB", line)
    for name in (
        "dqn_episode_return.png",
        "dqn_return_moving_average_100.png",
        "dqn_validation_joint_reward.png",
        "dqn_validation_metrics.png",
    ):
        path = PLOT_DIR / name
        print(f"WROTE {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
