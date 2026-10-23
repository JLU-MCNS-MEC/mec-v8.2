"""Per-200-episode Chinese progress reporting and diagnostics for Stage-2o.

The trainer calls :func:`write_round` after every Dev-60 validation.  It

* prints a Chinese progress block to the run log,
* writes ``rounds/round_XXXX.json`` with the full validation record,
* appends one row to ``rounds.csv``,
* writes the current per-episode ``training.csv`` and the Dev results.

The same functions are used by the live status helper so a user can query a
running experiment without touching it.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from v83_spatial_stage2.stage2k_common import atomic_write_json, utc_now

ROUND_FIELDS = [
    "scheme", "algorithm", "seed", "episode", "elapsed_s", "elapsed_h",
    "gpu_util_pct", "gpu_used_mb", "gpu_total_mb",
    "gpu_allocated_mb", "gpu_reserved_mb",
    "reward_ma100", "reward_std100", "reward_cv100",
    "e2e_ma100", "e2e_std100", "e2e_cv100",
    "dev_e2e_ms", "dev_e2e_std_ms", "dev_p95_ms", "dev_sr", "dev_spl",
    "dev_ndtw", "dev_delay_violation_rate", "dev_mean_steps",
    "dev_best_e2e_ms", "dev_best_episode", "dev_gap_to_best_pct",
    "dqn_q_mean", "dqn_q_max", "dqn_target_mean", "dqn_td_loss",
    "dqn_loss", "dqn_epsilon", "dqn_q_target_gap",
    "ppo_value_loss", "ppo_explained_variance", "ppo_entropy", "ppo_kl",
    "ppo_clip_fraction", "ppo_value_mean",
    "flag_reward_oscillation", "flag_q_drift", "flag_value_anomaly",
    "flag_dev_degradation", "flags", "timestamp",
]


def gpu_status():
    """Best-effort GPU status; never raises."""
    status = {
        "gpu_util_pct": None,
        "gpu_used_mb": None,
        "gpu_total_mb": None,
        "gpu_allocated_mb": None,
        "gpu_reserved_mb": None,
    }
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        first = output.strip().splitlines()[0]
        util, used, total = [part.strip() for part in first.split(",")]
        status.update({
            "gpu_util_pct": int(util),
            "gpu_used_mb": int(used),
            "gpu_total_mb": int(total),
        })
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            status["gpu_allocated_mb"] = round(
                torch.cuda.memory_allocated() / (1024 ** 2), 1
            )
            status["gpu_reserved_mb"] = round(
                torch.cuda.memory_reserved() / (1024 ** 2), 1
            )
    except Exception:
        pass
    return status


def _finite(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _mean(values):
    finite = [float(v) for v in values if _finite(v) is not None]
    return float(np.mean(finite)) if finite else None


def _std(values):
    finite = [float(v) for v in values if _finite(v) is not None]
    return float(np.std(finite, ddof=0)) if finite else None


def _cv(values):
    mean = _mean(values)
    std = _std(values)
    if mean is None or std is None or abs(mean) < 1e-12:
        return None
    return float(std / abs(mean))


def _flag(value):
    return "是" if bool(value) else "否"


def compute_round_metrics(history: Mapping, record: Mapping,
                          algorithm: str, scheme: str, seed: int,
                          elapsed_s: float, gpu: Mapping | None = None):
    """Assemble one row of the per-200-episode table."""
    rows = list(history.get("train", []))
    recent = rows[-100:]
    validations = list(history.get("validation", []))
    if not validations or int(validations[-1]["episode"]) != int(
        record["episode"]
    ):
        validations = validations + [dict(record)]
    best = min(validations, key=lambda item: float(item["E2E_ms"]))
    best_e2e = float(best["E2E_ms"])
    current_e2e = float(record["E2E_ms"])
    gap_pct = 100.0 * (current_e2e - best_e2e) / best_e2e if best_e2e else 0.0

    reward_values = [float(row["episode_return"]) for row in recent]
    e2e_values = [float(row["mean_episode_e2e_ms"]) for row in recent]
    reward_ma100 = _mean(reward_values)
    e2e_ma100 = _mean(e2e_values)
    reward_std = _std(reward_values)
    e2e_std = _std(e2e_values)
    reward_cv = _cv(reward_values)
    e2e_cv = _cv(e2e_values)
    scale = max(1.0, abs(float(record.get("train_return_ma100",
                                       reward_ma100 or 0.0))))

    metrics = {
        "scheme": scheme,
        "algorithm": algorithm,
        "seed": int(seed),
        "episode": int(record["episode"]),
        "elapsed_s": round(float(elapsed_s), 1),
        "elapsed_h": round(float(elapsed_s) / 3600.0, 3),
        "reward_ma100": reward_ma100,
        "reward_std100": reward_std,
        "reward_cv100": reward_cv,
        "e2e_ma100": e2e_ma100,
        "e2e_std100": e2e_std,
        "e2e_cv100": e2e_cv,
        "dev_e2e_ms": current_e2e,
        "dev_e2e_std_ms": float(record["E2E_std_ms"]),
        "dev_p95_ms": float(record["P95_E2E_ms"]),
        "dev_sr": float(record["SR"]),
        "dev_spl": float(record["SPL"]),
        "dev_ndtw": float(record["nDTW"]),
        "dev_delay_violation_rate": float(record["delay_violation_rate"]),
        "dev_mean_steps": float(record["mean_steps"]),
        "dev_best_e2e_ms": best_e2e,
        "dev_best_episode": int(best["episode"]),
        "dev_gap_to_best_pct": float(gap_pct),
    }
    if gpu:
        metrics.update(gpu)

    flags = []
    # Reward oscillation: per-episode coefficient of variation over MA100.
    oscillation = bool(
        reward_cv is not None and int(record["episode"]) >= 400
        and reward_cv > 0.50
    )
    if oscillation:
        flags.append(f"奖励震荡(CV100={reward_cv:.2f})")

    if algorithm == "dqn":
        q_mean = _finite(record.get("q_mean"))
        q_max = _finite(record.get("q_max"))
        target_mean = _finite(record.get("target_mean"))
        q_gap = (
            q_mean - target_mean
            if q_mean is not None and target_mean is not None else None
        )
        metrics.update({
            "dqn_q_mean": q_mean,
            "dqn_q_max": q_max,
            "dqn_target_mean": target_mean,
            "dqn_td_loss": _finite(record.get("td_error_abs_mean")),
            "dqn_loss": _finite(record.get("train_loss_mean")),
            "dqn_epsilon": _finite(record.get("epsilon")),
            "dqn_q_target_gap": q_gap,
        })
        q_drift = bool(
            (q_mean is not None and abs(q_mean) > 3.0 * scale)
            or (q_max is not None and abs(q_max) > 10.0 * scale)
            or (q_gap is not None and abs(q_gap) > 0.50 * scale)
        )
        metrics["flag_q_drift"] = q_drift
        metrics["flag_value_anomaly"] = False
        if q_drift:
            flags.append(
                "Q值漂移("
                f"Qmean={q_mean}, Qmax={q_max}, target={target_mean})"
                if q_mean is not None else "Q值漂移"
            )
    else:
        value_mean = _finite(record.get("value_mean"))
        value_abs_max = _finite(record.get("value_abs_max"))
        explained = _finite(record.get("explained_variance"))
        metrics.update({
            "ppo_value_loss": _finite(record.get("update_value_loss")),
            "ppo_explained_variance": explained,
            "ppo_entropy": _finite(record.get("update_entropy")),
            "ppo_kl": _finite(record.get("approx_kl")),
            "ppo_clip_fraction": _finite(record.get("clip_fraction")),
            "ppo_value_mean": value_mean,
        })
        value_anomaly = bool(
            (value_mean is not None and abs(value_mean) > 3.0 * scale)
            or (value_abs_max is not None and abs(value_abs_max) > 10.0 * scale)
            or (explained is not None and explained < -1.0)
        )
        metrics["flag_value_anomaly"] = value_anomaly
        metrics["flag_q_drift"] = False
        if value_anomaly:
            flags.append(
                "价值估计异常("
                f"Vmean={value_mean}, EV={explained})"
            )

    metrics["flag_reward_oscillation"] = oscillation
    dev_degradation = bool(gap_pct > 5.0)
    metrics["flag_dev_degradation"] = dev_degradation
    if dev_degradation:
        flags.append(f"Dev退化(比最佳高{gap_pct:.1f}%)")
    metrics["flags"] = ";".join(flags)
    metrics["timestamp"] = utc_now()
    return metrics


def format_round_zh(metrics: Mapping) -> str:
    """Chinese progress block printed to the run log every 200 episodes."""
    gpu_text = (
        f"util {metrics.get('gpu_util_pct')}%, "
        f"显存 {metrics.get('gpu_used_mb')}/{metrics.get('gpu_total_mb')} MiB"
        if metrics.get("gpu_used_mb") is not None else "GPU 状态不可用"
    )
    lines = [
        "",
        "──────── Stage-2o 进展 ────────",
        f"实验: {metrics['algorithm'].upper()} | {metrics['scheme']} | "
        f"seed={metrics['seed']} | Episode {metrics['episode']}",
        f"训练耗时: {metrics['elapsed_h']:.2f} h ({metrics['elapsed_s']:.0f} s)"
        f" | GPU: {gpu_text}",
        f"MA100 奖励: {_fmt(metrics.get('reward_ma100'), 4)} "
        f"(σ={_fmt(metrics.get('reward_std100'), 4)}, "
        f"CV={_fmt(metrics.get('reward_cv100'), 3)})",
        f"MA100 E2E 时延: {_fmt(metrics.get('e2e_ma100'), 2)} ms "
        f"(σ={_fmt(metrics.get('e2e_std100'), 2)} ms)",
        f"Dev-60: E2E={metrics['dev_e2e_ms']:.2f} ms | "
        f"P95={metrics['dev_p95_ms']:.1f} ms | "
        f"SR={100*metrics['dev_sr']:.2f}% | SPL={metrics['dev_spl']:.4f} | "
        f"nDTW={metrics['dev_ndtw']:.4f}",
        f"Dev 最佳: {metrics['dev_best_e2e_ms']:.2f} ms "
        f"@Ep{metrics['dev_best_episode']} | "
        f"当前距最佳 {metrics['dev_gap_to_best_pct']:.2f}%",
    ]
    if metrics["algorithm"] == "dqn":
        lines.append(
            f"DQN: Q mean={_fmt(metrics.get('dqn_q_mean'), 4)} | "
            f"Q max={_fmt(metrics.get('dqn_q_max'), 4)} | "
            f"Target={_fmt(metrics.get('dqn_target_mean'), 4)} | "
            f"TD Loss={_fmt(metrics.get('dqn_td_loss'), 5)}"
        )
    else:
        lines.append(
            f"PPO: Value Loss={_fmt(metrics.get('ppo_value_loss'), 5)} | "
            f"Explained Variance={_fmt(metrics.get('ppo_explained_variance'), 4)} | "
            f"Entropy={_fmt(metrics.get('ppo_entropy'), 4)} | "
            f"KL={_fmt(metrics.get('ppo_kl'), 5)}"
        )
    lines.append(
        "异常: "
        f"奖励震荡={_flag(metrics.get('flag_reward_oscillation'))} | "
        f"Q值漂移={_flag(metrics.get('flag_q_drift'))} | "
        f"价值异常={_flag(metrics.get('flag_value_anomaly'))} | "
        f"验证退化={_flag(metrics.get('flag_dev_degradation'))}"
    )
    if metrics.get("flags"):
        lines.append(f"提示: {metrics['flags']}")
    lines.append("──────────────────────────────")
    return "\n".join(lines)


def _fmt(value, digits):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def append_csv_row(path: Path, row: Mapping, fieldnames: Sequence[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def write_round(run_dir: Path, history: Mapping, record: Mapping,
                algorithm: str, scheme: str, seed: int, elapsed_s: float,
                extra: Mapping | None = None):
    """Write round JSON/CSV and return the printed Chinese block."""
    run_dir = Path(run_dir)
    metrics = compute_round_metrics(
        history, record, algorithm, scheme, seed, elapsed_s,
        gpu=gpu_status(),
    )
    if extra:
        metrics.update(dict(extra))
    payload = {
        "metrics": metrics,
        "validation": dict(record),
        "created_utc": utc_now(),
    }
    rounds_dir = run_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    path = rounds_dir / f"round_{int(record['episode']):04d}.json"
    atomic_write_json(path, payload)
    append_csv_row(run_dir / "rounds.csv", metrics, ROUND_FIELDS)
    print(format_round_zh(metrics), flush=True)
    return metrics


def write_training_csv(run_dir: Path, rows: Sequence[Mapping]):
    """Write the per-episode training log (small; safe to rewrite)."""
    run_dir = Path(run_dir)
    rows = list(rows)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path = run_dir / "training.csv"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    temporary.replace(path)


def read_status(run_dir: Path):
    """Live status helper: progress.json + latest round, never writes."""
    run_dir = Path(run_dir)
    progress = None
    progress_path = run_dir / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    rounds = sorted((run_dir / "rounds").glob("round_*.json"))
    latest = None
    if rounds:
        latest = json.loads(rounds[-1].read_text(encoding="utf-8"))
    return {"progress": progress, "latest_round": latest}


def elapsed_since(start_monotonic: float) -> float:
    return float(time.monotonic() - float(start_monotonic))


__all__ = [
    "ROUND_FIELDS",
    "gpu_status",
    "compute_round_metrics",
    "format_round_zh",
    "append_csv_row",
    "write_round",
    "write_training_csv",
    "read_status",
    "elapsed_since",
]
