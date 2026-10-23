"""Per-200-episode Chinese progress reporting for Stage-2p PPO runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from v83_spatial_stage2.stage2k_common import atomic_write_json, utc_now
from v83_spatial_stage2.stage2o.progress import (  # noqa: F401
    append_csv_row,
    gpu_status,
    write_training_csv,
)
from v83_spatial_stage2.stage2p.rewards import SCHEME_LABELS

ROUND_FIELDS = [
    "scheme", "scheme_label", "algorithm", "seed", "episode",
    "elapsed_s", "elapsed_h",
    "gpu_util_pct", "gpu_used_mb", "gpu_total_mb",
    "gpu_allocated_mb", "gpu_reserved_mb",
    "reward_ma100", "reward_std100", "e2e_ma100", "e2e_std100",
    "dev_e2e_ms", "dev_e2e_std_ms", "dev_p95_ms", "dev_sr", "dev_spl",
    "dev_ndtw", "dev_delay_violation_rate", "dev_mean_steps",
    "dev_best_e2e_ms", "dev_best_episode", "dev_gap_to_best_pct",
    "ppo_value_loss", "ppo_explained_variance", "ppo_entropy", "ppo_kl",
    "ppo_clip_fraction", "ppo_value_mean", "ppo_value_abs_max",
    "flag_reward_oscillation", "flag_e2e_degradation",
    "flag_value_anomaly", "flags", "timestamp",
]


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


def _fmt(value, digits):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _flag(value):
    return "是" if bool(value) else "否"


def compute_round_metrics(history: Mapping, record: Mapping, scheme: str,
                          seed: int, elapsed_s: float,
                          gpu: Mapping | None = None):
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
    reward_std = _std(reward_values)
    e2e_ma100 = _mean(e2e_values)
    e2e_std = _std(e2e_values)
    scale = max(1.0, abs(float(record.get("train_return_ma100",
                                       reward_ma100 or 0.0))))

    metrics = {
        "scheme": scheme,
        "scheme_label": SCHEME_LABELS[scheme],
        "algorithm": "ppo",
        "seed": int(seed),
        "episode": int(record["episode"]),
        "elapsed_s": round(float(elapsed_s), 1),
        "elapsed_h": round(float(elapsed_s) / 3600.0, 3),
        "reward_ma100": reward_ma100,
        "reward_std100": reward_std,
        "e2e_ma100": e2e_ma100,
        "e2e_std100": e2e_std,
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
    # Oscillation of the smoothed (MA100) reward across the last 5 rounds.
    round_rewards = [
        float(item["train_return_ma100"]) for item in validations[-5:]
        if item.get("train_return_ma100") is not None
    ]
    round_std = _std(round_rewards)
    round_mean = _mean(round_rewards)
    oscillation = bool(
        int(record["episode"]) >= 1000
        and round_std is not None and round_mean is not None
        and round_std / max(0.05, abs(round_mean)) > 0.25
    )
    metrics["flag_reward_oscillation"] = oscillation
    if oscillation:
        flags.append(
            f"奖励震荡(MA100 5轮σ={_fmt(round_std, 4)})"
        )

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
        "ppo_value_abs_max": value_abs_max,
    })
    value_anomaly = bool(
        (value_mean is not None and abs(value_mean) > 3.0 * scale)
        or (value_abs_max is not None and abs(value_abs_max) > 10.0 * scale)
        or (explained is not None and explained < -1.0)
    )
    metrics["flag_value_anomaly"] = value_anomaly
    if value_anomaly:
        flags.append(
            f"价值异常(Vmean={_fmt(value_mean, 3)}, EV={_fmt(explained, 3)})"
        )

    e2e_degradation = bool(gap_pct > 5.0)
    metrics["flag_e2e_degradation"] = e2e_degradation
    if e2e_degradation:
        flags.append(f"时延退化(比最佳高{gap_pct:.1f}%)")
    metrics["flags"] = ";".join(flags)
    metrics["timestamp"] = utc_now()
    return metrics


def format_round_zh(metrics: Mapping) -> str:
    gpu_text = (
        f"util {metrics.get('gpu_util_pct')}%, "
        f"显存 {metrics.get('gpu_used_mb')}/{metrics.get('gpu_total_mb')} MiB"
        if metrics.get("gpu_used_mb") is not None else "GPU 状态不可用"
    )
    lines = [
        "",
        "──────── Stage-2p 进展 ────────",
        f"实验: PPO | {metrics['scheme_label']} | seed={metrics['seed']} | "
        f"Episode {metrics['episode']}",
        f"训练耗时: {metrics['elapsed_h']:.2f} h ({metrics['elapsed_s']:.0f} s)"
        f" | GPU: {gpu_text}",
        f"MA100 奖励: {_fmt(metrics.get('reward_ma100'), 4)} "
        f"(σ={_fmt(metrics.get('reward_std100'), 4)})",
        f"MA100 E2E 时延: {_fmt(metrics.get('e2e_ma100'), 2)} ms "
        f"(σ={_fmt(metrics.get('e2e_std100'), 2)} ms)",
        f"Dev-60: E2E={metrics['dev_e2e_ms']:.2f} ms | "
        f"P95={metrics['dev_p95_ms']:.1f} ms | "
        f"违约率={100*metrics['dev_delay_violation_rate']:.2f}% | "
        f"SR={100*metrics['dev_sr']:.2f}% | SPL={metrics['dev_spl']:.4f} | "
        f"nDTW={metrics['dev_ndtw']:.4f}",
        f"Dev 最佳: {metrics['dev_best_e2e_ms']:.2f} ms "
        f"@Ep{metrics['dev_best_episode']} | "
        f"当前距最佳 {metrics['dev_gap_to_best_pct']:.2f}%",
        f"PPO: Value Loss={_fmt(metrics.get('ppo_value_loss'), 5)} | "
        f"Explained Variance={_fmt(metrics.get('ppo_explained_variance'), 4)} | "
        f"Entropy={_fmt(metrics.get('ppo_entropy'), 4)} | "
        f"KL={_fmt(metrics.get('ppo_kl'), 5)}",
        "异常: "
        f"奖励震荡={_flag(metrics.get('flag_reward_oscillation'))} | "
        f"时延退化={_flag(metrics.get('flag_e2e_degradation'))} | "
        f"价值异常={_flag(metrics.get('flag_value_anomaly'))}",
    ]
    if metrics.get("flags"):
        lines.append(f"提示: {metrics['flags']}")
    lines.append("──────────────────────────────")
    return "\n".join(lines)


def write_round(run_dir: Path, history: Mapping, record: Mapping,
                scheme: str, seed: int, elapsed_s: float):
    run_dir = Path(run_dir)
    metrics = compute_round_metrics(
        history, record, scheme, seed, elapsed_s, gpu=gpu_status(),
    )
    payload = {
        "metrics": metrics,
        "validation": dict(record),
        "created_utc": utc_now(),
    }
    rounds_dir = run_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        rounds_dir / f"round_{int(record['episode']):04d}.json", payload
    )
    append_csv_row(run_dir / "rounds.csv", metrics, ROUND_FIELDS)
    print(format_round_zh(metrics), flush=True)
    return metrics


__all__ = [
    "ROUND_FIELDS",
    "compute_round_metrics",
    "format_round_zh",
    "write_round",
    "write_training_csv",
    "gpu_status",
]
