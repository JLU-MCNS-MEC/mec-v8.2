"""Per-200-episode Chinese progress reporting for Stage-2r.

Every block contains the requested cumulative training statistics:
total episodes, successes, failures, and the successful short / medium / long
counts, in addition to MA100 reward/E2E and Dev-60 metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from v83_spatial_stage2.stage2k_common import atomic_write_json, utc_now
from v83_spatial_stage2.stage2o.progress import (
    ROUND_FIELDS as STAGE2O_ROUND_FIELDS,
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
ROUND_FIELDS = ["stage", "model"] + list(STAGE2O_ROUND_FIELDS) + COUNT_FIELDS


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


def compute_round_metrics(history, record, algorithm, scheme, seed,
                          elapsed_s, gpu=None):
    metrics = _stage2o_round_metrics(
        history, record, algorithm, scheme, seed, elapsed_s, gpu=gpu
    )
    metrics["stage"] = "stage2r"
    metrics["model"] = str(history.get("model", ""))
    metrics.update(cumulative_stats(history.get("train", [])))
    return metrics


def _fmt(value, digits):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_round_zh(metrics: Mapping) -> str:
    gpu_text = (
        f"util {metrics.get('gpu_util_pct')}%, "
        f"显存 {metrics.get('gpu_used_mb')}/{metrics.get('gpu_total_mb')} MiB"
        if metrics.get("gpu_used_mb") is not None else "GPU 状态不可用"
    )
    rate = metrics.get("cum_success_rate")
    lines = [
        "",
        "──────── Stage-2r 进展 ────────",
        f"实验: {metrics.get('model')} | {metrics['algorithm'].upper()} | "
        f"方案D | seed={metrics['seed']} | Episode {metrics['episode']}",
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
        f"long={metrics.get('cum_success_long')}  "
        f"(总 band: {metrics.get('cum_total_short')}/"
        f"{metrics.get('cum_total_medium')}/"
        f"{metrics.get('cum_total_long')})",
        f"失败 band: short={metrics.get('cum_failure_short')} | "
        f"medium={metrics.get('cum_failure_medium')} | "
        f"long={metrics.get('cum_failure_long')}",
        f"Dev-60: E2E={metrics['dev_e2e_ms']:.2f} ms | "
        f"P95={metrics['dev_p95_ms']:.1f} ms | "
        f"SR={100*metrics['dev_sr']:.2f}% | SPL={metrics['dev_spl']:.4f} | "
        f"nDTW={metrics['dev_ndtw']:.4f} | "
        f"RewardD={1.0 - metrics['dev_e2e_ms']/100.0:.4f}",
        f"Dev 最佳: {metrics['dev_best_e2e_ms']:.2f} ms "
        f"@Ep{metrics['dev_best_episode']} | "
        f"当前距最佳 {metrics['dev_gap_to_best_pct']:.2f}%",
    ]
    if metrics["algorithm"] == "dqn":
        lines.append(
            f"DQN: Q mean={_fmt(metrics.get('dqn_q_mean'), 4)} | "
            f"Q max={_fmt(metrics.get('dqn_q_max'), 4)} | "
            f"Target={_fmt(metrics.get('dqn_target_mean'), 4)} | "
            f"TD Loss={_fmt(metrics.get('dqn_td_loss'), 5)} | "
            f"ε={_fmt(metrics.get('dqn_epsilon'), 3)}"
        )
    else:
        lines.append(
            f"PPO: Value Loss={_fmt(metrics.get('ppo_value_loss'), 5)} | "
            f"Explained Variance="
            f"{_fmt(metrics.get('ppo_explained_variance'), 4)} | "
            f"Entropy={_fmt(metrics.get('ppo_entropy'), 4)} | "
            f"KL={_fmt(metrics.get('ppo_kl'), 5)}"
        )
    if metrics.get("flags"):
        lines.append(f"提示: {metrics['flags']}")
    lines.append("──────────────────────────────")
    return "\n".join(lines)


def write_round(run_dir: Path, history: Mapping, record: Mapping,
                algorithm: str, scheme: str, seed: int, elapsed_s: float,
                extra: Mapping | None = None):
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
    atomic_write_json(
        rounds_dir / f"round_{int(record['episode']):04d}.json", payload
    )
    append_csv_row(run_dir / "rounds.csv", metrics, ROUND_FIELDS)
    print(format_round_zh(metrics), flush=True)
    return metrics


__all__ = [
    "BANDS", "COUNT_FIELDS", "ROUND_FIELDS", "cumulative_stats",
    "compute_round_metrics", "format_round_zh", "write_round",
    "write_training_csv", "gpu_status",
]
