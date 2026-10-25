"""Aggregate the six Stage-2s runs into tracked CSV/JSON.

Outputs (under the Stage-2s evidence directory):

* ``training_curves.csv`` — per-run per-episode curves + MA100 + cumulative
* ``validation_curves.csv`` — per-200-episode Dev-60 metrics
* ``per_200_episodes.csv`` — combined round tables
* ``success_failure_summary.csv`` — final success/failure and band counts
* ``condition_training_summary.csv`` — algorithm mean/std over seeds
* ``seed_stability.csv`` — per-seed stability metrics
* ``summary.json`` / ``SUMMARY_TABLES.md``

The Test pool is never opened.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    read_json,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (  # noqa: E402
    ALGORITHMS,
    ALGORITHM_LABELS,
    EVIDENCE_DIR,
    FORMAL_SEEDS,
    REWARD_SCHEME,
    STAGE,
    condition_label,
    run_dir,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.progress import (  # noqa: E402
    cumulative_stats,
)

TRAIN_COLUMNS = [
    "stage", "algorithm", "seed", "condition", "episode", "episode_uid",
    "band", "success", "steps", "episode_return", "reward_D",
    "mean_episode_e2e_ms", "final_nDTW", "spl", "small_steps",
    "transformer_m_steps", "small_ratio", "transformer_m_ratio",
    "mean_inference_ms", "mean_communication_ms",
    "mean_inference_small_ms", "mean_inference_transformer_m_ms",
    "mean_e2e_small_ms", "mean_e2e_transformer_m_ms",
    "ma100_reward", "ma100_e2e", "std100_e2e", "cum_total", "cum_success",
    "cum_failure", "cum_success_short", "cum_success_medium",
    "cum_success_long", "cum_failure_short", "cum_failure_medium",
    "cum_failure_long",
]
VALIDATION_COLUMNS = [
    "algorithm", "seed", "condition", "episode", "episodes", "E2E_ms",
    "E2E_std_ms", "P95_E2E_ms", "RewardD", "SR", "OSR", "SPL", "nDTW",
    "delay_violation_rate", "mean_steps", "dev_small_steps",
    "dev_transformer_m_steps", "dev_small_ratio",
    "dev_transformer_m_ratio", "dev_mean_inference_ms",
    "dev_mean_communication_ms", "dev_mean_inference_small_ms",
    "dev_mean_inference_transformer_m_ms", "train_return_ma100",
    "train_e2e_ma100",
]


def _rolling(series, window=100, kind="mean"):
    rolling = series.rolling(window, min_periods=1)
    return rolling.mean() if kind == "mean" else rolling.std(ddof=0)


def _finite(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _load_run(algorithm, seed):
    path = run_dir(algorithm, seed)
    history = read_json(path / "training_history.json")
    result = read_json(path / "training_result.json")
    if history is None or not history.get("train"):
        return None
    frame = pd.DataFrame(history["train"])
    frame = frame.sort_values("episode").reset_index(drop=True)
    frame["ma100_reward"] = _rolling(frame["episode_return"])
    frame["ma100_e2e"] = _rolling(frame["mean_episode_e2e_ms"])
    frame["std100_e2e"] = _rolling(
        frame["mean_episode_e2e_ms"], kind="std"
    )
    # Cumulative counts must be recomputed row-by-row.
    success = []
    failure = []
    success_band = {band: 0 for band in ("short", "medium", "long")}
    failure_band = {band: 0 for band in ("short", "medium", "long")}
    total_band = {band: 0 for band in ("short", "medium", "long")}
    rows = []
    for _, row in frame.iterrows():
        band = str(row.get("band"))
        total_band[band] = total_band.get(band, 0) + 1
        if bool(row.get("success")):
            success.append(row)
            success_band[band] = success_band.get(band, 0) + 1
        else:
            failure.append(row)
            failure_band[band] = failure_band.get(band, 0) + 1
        item = row.to_dict()
        item.update({
            "algorithm": algorithm,
            "seed": int(seed),
            "condition": condition_label(algorithm, seed),
            "cum_total": len(success) + len(failure),
            "cum_success": len(success),
            "cum_failure": len(failure),
            "cum_success_short": success_band["short"],
            "cum_success_medium": success_band["medium"],
            "cum_success_long": success_band["long"],
            "cum_failure_short": failure_band["short"],
            "cum_failure_medium": failure_band["medium"],
            "cum_failure_long": failure_band["long"],
            "cum_total_short": total_band["short"],
            "cum_total_medium": total_band["medium"],
            "cum_total_long": total_band["long"],
        })
        rows.append(item)
    frame = pd.DataFrame(rows)
    return {
        "frame": frame,
        "history": history,
        "result": result,
        "run_dir": path,
    }


def _load_rounds(algorithm, seed):
    path = run_dir(algorithm, seed) / "rounds.csv"
    if not path.is_file():
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    if frame.empty:
        return []
    frame["algorithm"] = algorithm
    frame["seed"] = int(seed)
    return frame.to_dict("records")


def _run_summary(algorithm, seed, run):
    frame = run["frame"]
    history = run["history"]
    result = run["result"]
    last100 = frame.tail(100)
    stats = cumulative_stats(frame.to_dict("records"))
    return {
        "algorithm": algorithm,
        "algorithm_label": ALGORITHM_LABELS[algorithm],
        "seed": int(seed),
        "condition": condition_label(algorithm, seed),
        "episodes": int(len(frame)),
        "status": (result or {}).get("status", "training")
        if result else "training",
        "elapsed_seconds": (result or {}).get("elapsed_seconds")
        if result else history.get("elapsed_seconds"),
        "total_steps": int(frame["steps"].sum()),
        "total_success": int(stats["cum_success"]),
        "total_failure": int(stats["cum_failure"]),
        "success_rate": (
            float(stats["cum_success"]) / len(frame) if len(frame) else None
        ),
        "success_short": int(stats["cum_success_short"]),
        "success_medium": int(stats["cum_success_medium"]),
        "success_long": int(stats["cum_success_long"]),
        "failure_short": int(stats["cum_failure_short"]),
        "failure_medium": int(stats["cum_failure_medium"]),
        "failure_long": int(stats["cum_failure_long"]),
        "small_steps": int(frame["small_steps"].sum())
        if "small_steps" in frame else None,
        "transformer_m_steps": int(frame["transformer_m_steps"].sum())
        if "transformer_m_steps" in frame else None,
        "small_ratio": (
            float(frame["small_steps"].sum() / frame["steps"].sum())
            if "small_steps" in frame and frame["steps"].sum() else None
        ),
        "ma100_reward_last": _finite(last100["episode_return"].mean()),
        "ma100_e2e_last": _finite(last100["mean_episode_e2e_ms"].mean()),
        "e2e_std100_last": _finite(last100["mean_episode_e2e_ms"].std(
            ddof=0
        )),
        "reward_std100_last": _finite(last100["episode_return"].std(
            ddof=0
        )),
        "best_dev_e2e_ms": (result or {}).get("best_dev_e2e_ms"),
        "best_dev_episode": (result or {}).get("best_dev_episode"),
        "convergence": (result or {}).get("convergence", {}).get("status"),
        "training_mode": (result or {}).get("training_mode"),
        "lane_cache_used": (result or {}).get("lane_cache_used"),
        "final_test_read": False,
    }


def _aggregate(run_summaries):
    rows = []
    for algorithm in ALGORITHMS:
        subset = [
            row for row in run_summaries if row["algorithm"] == algorithm
        ]
        if not subset:
            continue
        item = {
            "algorithm": algorithm,
            "algorithm_label": ALGORITHM_LABELS[algorithm],
            "n_seeds": len(subset),
            "seeds": ",".join(str(row["seed"]) for row in subset),
        }
        for key in (
            "total_success", "total_failure", "success_rate", "small_ratio",
            "ma100_reward_last", "ma100_e2e_last", "e2e_std100_last",
            "reward_std100_last", "best_dev_e2e_ms", "elapsed_seconds",
        ):
            values = [
                float(row[key]) for row in subset
                if row.get(key) is not None
            ]
            item[f"{key}_mean"] = float(np.mean(values)) if values else None
            item[f"{key}_std"] = (
                float(np.std(values, ddof=0)) if values else None
            )
        item["converged_runs"] = sum(
            1 for row in subset if row.get("convergence") == "converged"
        )
        rows.append(item)
    return rows


def _markdown(run_summaries, aggregates, eval_summary):
    lines = [
        "# Stage-2s 汇总表",
        "",
        f"- 生成时间：{utc_now()}",
        "- Test-400：全程未读取",
        "",
        "## 1. 每 run 最终统计",
        "",
        "| 算法 | seed | Episodes | 成功 | 失败 | 成功率 | 成功 S/M/L | 失败 S/M/L | Small 选择 | MA100 E2E | 最佳 Dev E2E | 收敛 |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for row in run_summaries:
        small = (
            f"{100*row['small_ratio']:.2f}%"
            if row.get("small_ratio") is not None else "-"
        )
        best = (
            f"{row['best_dev_e2e_ms']:.2f}"
            if row.get("best_dev_e2e_ms") is not None else "-"
        )
        ma100 = (
            f"{row['ma100_e2e_last']:.2f}"
            if row.get("ma100_e2e_last") is not None else "-"
        )
        lines.append(
            f"| {row['algorithm_label']} | {row['seed']} | "
            f"{row['episodes']} | {row['total_success']} | "
            f"{row['total_failure']} | "
            f"{100*row['success_rate']:.2f}% | "
            f"{row['success_short']}/{row['success_medium']}/"
            f"{row['success_long']} | "
            f"{row['failure_short']}/{row['failure_medium']}/"
            f"{row['failure_long']} | {small} | {ma100} | {best} | "
            f"{row.get('convergence')} |"
        )
    lines += [
        "",
        "## 2. 算法聚合（3 seeds 均值 ± 标准差）",
        "",
        "| 算法 | 成功率 | Small 选择 | MA100 E2E | 最佳 Dev E2E | 收敛 run |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        def fmt(key):
            value = row.get(f"{key}_mean")
            std = row.get(f"{key}_std")
            if value is None:
                return "-"
            return f"{value:.4f} ± {std:.4f}" if key == "success_rate" else (
                f"{value:.2f} ± {std:.2f}"
            )
        lines.append(
            f"| {row['algorithm_label']} | {fmt('success_rate')} | "
            f"{fmt('small_ratio')} | {fmt('ma100_e2e_last')} | "
            f"{fmt('best_dev_e2e_ms')} | "
            f"{row['converged_runs']}/{row['n_seeds']} |"
        )
    if eval_summary:
        lines += [
            "",
            "## 3. Dev-60 最终验证",
            "",
            "| 策略 | SR | OSR | nDTW | SPL | E2E (ms) | RewardD | Small 选择 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in eval_summary.get("summary", []):
            lines.append(
                f"| {row['title']} | {100*row['SR']:.2f}% | "
                f"{100*row['OSR']:.2f}% | {row['nDTW']:.4f} | "
                f"{row['SPL']:.4f} | {row['E2E_ms']:.2f} | "
                f"{row['RewardD']:.4f} | "
                f"{100*row['small_ratio']:.2f}% |"
            )
        lines += [
            "",
            "## 4. 相对 baseline 的 E2E 差",
            "",
            "| 算法 | baseline | 学习均值 E2E | baseline E2E | ΔE2E | Δ% | ΔRewardD |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in eval_summary.get("baseline_comparison", []):
            lines.append(
                f"| {row['algorithm_label']} | {row['baseline_title']} | "
                f"{row['learned_E2E_ms_mean']:.2f} | "
                f"{row['baseline_E2E_ms']:.2f} | "
                f"{row['delta_E2E_ms']:+.2f} | "
                f"{row['delta_E2E_pct']:+.2f}% | "
                f"{row['delta_RewardD']:+.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    validation_rows = []
    round_rows = []
    run_summaries = []
    for algorithm in ALGORITHMS:
        for seed in FORMAL_SEEDS:
            run = _load_run(algorithm, seed)
            if run is None:
                print(
                    f"[analyze] {algorithm}-s{seed}: no training history",
                    flush=True,
                )
                continue
            frames.append(run["frame"])
            validation_rows.extend(
                run["history"].get("validation", [])
            )
            round_rows.extend(_load_rounds(algorithm, seed))
            run_summaries.append(_run_summary(algorithm, seed, run))
    if not frames:
        raise SystemExit("no Stage-2s training history found")
    training = pd.concat(frames, ignore_index=True)
    training["stage"] = STAGE
    training = training[
        [column for column in TRAIN_COLUMNS if column in training.columns]
    ]
    training.to_csv(EVIDENCE_DIR / "training_curves.csv", index=False)
    validation = pd.DataFrame(validation_rows)
    if not validation.empty:
        for key, default in (
            ("algorithm", ""), ("seed", -1), ("condition", ""),
            ("episodes", 60),
        ):
            if key not in validation:
                validation[key] = default
        validation["RewardD"] = 1.0 - validation["E2E_ms"] / 100.0
        validation = validation[
            [column for column in VALIDATION_COLUMNS
             if column in validation.columns]
        ]
        validation.to_csv(
            EVIDENCE_DIR / "validation_curves.csv", index=False
        )
    rounds = pd.DataFrame(round_rows)
    if not rounds.empty:
        rounds.to_csv(EVIDENCE_DIR / "per_200_episodes.csv", index=False)
    write_csv(
        EVIDENCE_DIR / "success_failure_summary.csv", run_summaries
    )
    aggregates = _aggregate(run_summaries)
    write_csv(
        EVIDENCE_DIR / "condition_training_summary.csv", aggregates
    )
    stability = []
    for row in run_summaries:
        stability.append({
            "algorithm": row["algorithm"],
            "seed": row["seed"],
            "ma100_reward_last": row["ma100_reward_last"],
            "ma100_e2e_last": row["ma100_e2e_last"],
            "e2e_std100_last": row["e2e_std100_last"],
            "reward_std100_last": row["reward_std100_last"],
            "best_dev_e2e_ms": row["best_dev_e2e_ms"],
            "best_dev_episode": row["best_dev_episode"],
            "convergence": row["convergence"],
        })
    write_csv(EVIDENCE_DIR / "seed_stability.csv", stability)
    eval_summary = read_json(EVIDENCE_DIR / "validation_summary.json")
    summary = {
        "stage": STAGE,
        "created_utc": utc_now(),
        "runs": run_summaries,
        "aggregates": aggregates,
        "eval_available": eval_summary is not None,
        "eval": eval_summary,
        "final_test_read": False,
    }
    atomic_write_json(EVIDENCE_DIR / "summary.json", summary)
    (EVIDENCE_DIR / "SUMMARY_TABLES.md").write_text(
        _markdown(run_summaries, aggregates, eval_summary),
        encoding="utf-8",
    )
    print(
        f"[analyze] {len(run_summaries)} runs, "
        f"{len(training)} training rows, "
        f"{len(validation)} validation rows",
        flush=True,
    )
    print("STAGE2S_ANALYZE_COMPLETE", flush=True)
    return summary


if __name__ == "__main__":
    main()
