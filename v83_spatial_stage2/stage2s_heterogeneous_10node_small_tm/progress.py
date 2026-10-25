"""Per-200-episode Chinese progress reporting for Stage-2s.

Each round reports the requested training statistics (total / success /
failure episodes and successful / failed short-medium-long counts), MA100
reward and E2E, the rolling E2E standard deviation, Dev-60 metrics, and the
heterogeneous Small / Transformer-M selection and latency decomposition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from v83_spatial_stage2.stage2k_common import atomic_write_json, utc_now
from v83_spatial_stage2.stage2o.progress import (
    ROUND_FIELDS as STAGE2O_ROUND_FIELDS,
    _fmt,
    append_csv_row,
    compute_round_metrics as _stage2o_round_metrics,
    gpu_status,
    write_training_csv,
)

BANDS = ("short", "medium", "long")
COUNT_FIELDS = [
    "cum_total", "cum_success", "cum_failure", "cum_success_rate",
    "cum_success_short", "cum_success_medium", "cum_success_long",
    "cum_total_short", "cum_total_medium", "cum_total_long",
    "cum_failure_short", "cum_failure_medium", "cum_failure_long",
]
HETERO_FIELDS = [
    "dev_osr", "dev_reward_d",
    "dev_small_steps", "dev_transformer_m_steps",
    "dev_small_ratio", "dev_transformer_m_ratio",
    "dev_mean_inference_ms", "dev_mean_communication_ms",
    "dev_mean_inference_small_ms", "dev_mean_inference_transformer_m_ms",
    "dev_mean_communication_small_ms",
    "dev_mean_communication_transformer_m_ms",
    "dev_mean_e2e_small_ms", "dev_mean_e2e_transformer_m_ms",
    "train_small_ratio_ma100", "train_transformer_m_ratio_ma100",
    "train_mean_inference_ms_ma100", "train_mean_e2e_ms_ma100",
]
ROUND_FIELDS = (
    ["stage"] + list(STAGE2O_ROUND_FIELDS) + COUNT_FIELDS + HETERO_FIELDS
)


def cumulative_stats(rows: Sequence[Mapping]) -> dict:
    rows = list(rows)
    success = [row for row in rows if bool(row.get("success"))]
    failure = [row for row in rows if not bool(row.get("success"))]
    stats = {
        "cum_total": len(rows),
        "cum_success": len(success),
        "cum_failure": len(failure),
        "cum_success_rate": (
            float(len(success)) / len(rows) if rows else None
        ),
    }
    for band in BANDS:
        stats[f"cum_success_{band}"] = sum(
            1 for row in success if str(row.get("band")) == band
        )
        stats[f"cum_total_{band}"] = sum(
            1 for row in rows if str(row.get("band")) == band
        )
        stats[f"cum_failure_{band}"] = sum(
            1 for row in failure if str(row.get("band")) == band
        )
    return stats


def _mean_or_none(values):
    values = [
        float(value) for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return float(np.mean(values)) if values else None


def compute_round_metrics(history, record, algorithm, seed, elapsed_s,
                          gpu=None):
    metrics = _stage2o_round_metrics(
        history, record, algorithm, "D", seed, elapsed_s, gpu=gpu,
    )
    metrics["stage"] = "stage2s"
    metrics["algorithm"] = str(algorithm)
    metrics["seed"] = int(seed)
    metrics["scheme"] = "D"
    metrics.update(cumulative_stats(history.get("train", [])))
    metrics["dev_osr"] = float(record.get("OSR", 0.0))
    metrics["dev_reward_d"] = float(1.0 - float(record["E2E_ms"]) / 100.0)
    metrics["dev_episodes"] = int(record.get("episodes", 0))
    for key in HETERO_FIELDS:
        if key.startswith("dev_") and key in record:
            metrics[key] = record.get(key)
    recent = list(history.get("train", []))[-100:]
    metrics["train_small_ratio_ma100"] = _mean_or_none(
        [row.get("small_ratio") for row in recent]
    )
    metrics["train_transformer_m_ratio_ma100"] = _mean_or_none(
        [row.get("transformer_m_ratio") for row in recent]
    )
    metrics["train_mean_inference_ms_ma100"] = _mean_or_none(
        [row.get("mean_inference_ms") for row in recent]
    )
    metrics["train_mean_e2e_ms_ma100"] = _mean_or_none(
        [row.get("mean_episode_e2e_ms") for row in recent]
    )
    metrics["timestamp"] = utc_now()
    return metrics


def format_round_zh(metrics: Mapping) -> str:
    gpu_text = (
        f"util {metrics.get('gpu_util_pct')}%, "
        f"显存 {metrics.get('gpu_used_mb')}/{metrics.get('gpu_total_mb')} MiB"
        if metrics.get("gpu_used_mb") is not None else "GPU 状态不可用"
    )
    rate = metrics.get("cum_success_rate")
    dev_label = f"Dev-{int(metrics.get('dev_episodes') or 60)}"
    lines = [
        "",
        "──────── Stage-2s 异构 5×Small + 5×Transformer-M 进展 ────────",
        f"实验: {metrics['algorithm'].upper()} | 方案D | "
        f"seed={metrics['seed']} | Episode {metrics['episode']}",
        f"训练耗时: {metrics['elapsed_h']:.2f} h "
        f"({metrics['elapsed_s']:.0f} s) | GPU: {gpu_text}",
        f"MA100 奖励: {_fmt(metrics.get('reward_ma100'), 4)} "
        f"(σ={_fmt(metrics.get('reward_std100'), 4)})",
        f"MA100 E2E 时延: {_fmt(metrics.get('e2e_ma100'), 2)} ms "
        f"(σ={_fmt(metrics.get('e2e_std100'), 2)} ms)",
        f"训练累计: 总 {metrics.get('cum_total')} | "
        f"成功 {metrics.get('cum_success')} | "
        f"失败 {metrics.get('cum_failure')} | "
        f"成功率 {_fmt(rate * 100 if rate is not None else None, 2)}%",
        f"成功 band: short={metrics.get('cum_success_short')} | "
        f"medium={metrics.get('cum_success_medium')} | "
        f"long={metrics.get('cum_success_long')}",
        f"失败 band: short={metrics.get('cum_failure_short')} | "
        f"medium={metrics.get('cum_failure_medium')} | "
        f"long={metrics.get('cum_failure_long')}",
        f"{dev_label}: E2E={metrics['dev_e2e_ms']:.2f} ms | "
        f"P95={metrics['dev_p95_ms']:.1f} ms | "
        f"SR={100*metrics['dev_sr']:.2f}% | "
        f"OSR={100*float(metrics.get('dev_osr', 0.0)):.2f}% | "
        f"SPL={metrics['dev_spl']:.4f} | "
        f"nDTW={metrics['dev_ndtw']:.4f} | "
        f"RewardD={float(metrics.get('dev_reward_d', 0.0)):.4f}",
        f"Dev 最佳: {metrics['dev_best_e2e_ms']:.2f} ms "
        f"@Ep{metrics['dev_best_episode']} | "
        f"当前距最佳 {metrics['dev_gap_to_best_pct']:.2f}%",
        f"异构选择 (Dev): Small={_fmt(100*metrics.get('dev_small_ratio', 0.0) if metrics.get('dev_small_ratio') is not None else None, 2)}% "
        f"({metrics.get('dev_small_steps')} 步) | "
        f"Transformer-M={_fmt(100*metrics.get('dev_transformer_m_ratio', 0.0) if metrics.get('dev_transformer_m_ratio') is not None else None, 2)}% "
        f"({metrics.get('dev_transformer_m_steps')} 步)",
        f"Dev 推理时延: Small={_fmt(metrics.get('dev_mean_inference_small_ms'), 3)} ms | "
        f"Transformer-M={_fmt(metrics.get('dev_mean_inference_transformer_m_ms'), 3)} ms",
        f"Dev 通信时延: Small={_fmt(metrics.get('dev_mean_communication_small_ms'), 2)} ms | "
        f"Transformer-M={_fmt(metrics.get('dev_mean_communication_transformer_m_ms'), 2)} ms",
        f"Dev E2E: Small={_fmt(metrics.get('dev_mean_e2e_small_ms'), 2)} ms | "
        f"Transformer-M={_fmt(metrics.get('dev_mean_e2e_transformer_m_ms'), 2)} ms",
        f"训练 MA100 选择: Small={_fmt(metrics.get('train_small_ratio_ma100'), 3)} | "
        f"Transformer-M={_fmt(metrics.get('train_transformer_m_ratio_ma100'), 3)}",
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
    lines.append("──────────────────────────────────────────────")
    return "\n".join(lines)


def _flag(value):
    return "是" if bool(value) else "否"


def write_round(run_dir: Path, history: Mapping, record: Mapping,
                algorithm: str, seed: int, elapsed_s: float):
    run_dir = Path(run_dir)
    metrics = compute_round_metrics(
        history, record, algorithm, seed, elapsed_s, gpu=gpu_status(),
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
    "BANDS", "COUNT_FIELDS", "HETERO_FIELDS", "ROUND_FIELDS",
    "cumulative_stats", "compute_round_metrics", "format_round_zh",
    "write_round", "write_training_csv",
]
