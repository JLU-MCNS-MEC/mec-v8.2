"""Aggregate the 12 Stage-2r runs and write tracked CSV/JSON/figures.

Outputs (all under the Stage-2r evidence directory unless noted):

* ``training_curves.csv`` / ``validation_curves.csv``
* ``per_200_episodes.csv``
* ``success_failure_summary.csv``
* ``condition_training_summary.csv``
* ``summary.json`` / ``SUMMARY_TABLES.md``
* figures under ``figures/`` and CSV sidecars under ``figure_data/``

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
from v83_spatial_stage2.stage2r_10node_small_vs_tm.common import (  # noqa: E402
    ALGORITHMS,
    ALGORITHM_LABELS,
    CONDITIONS,
    EVIDENCE_DIR,
    FORMAL_SEEDS,
    MEC_MODELS,
    MODEL_LABELS,
    REWARD_SCHEME,
    STAGE,
    RUNS_ROOT,
    load_deployment,
    model_slug,
    run_dir,
)
from v83_spatial_stage2.stage2r_10node_small_vs_tm.progress import (  # noqa: E402
    cumulative_stats,
)

TRAIN_COLUMNS = [
    "model", "deployment", "algorithm", "seed", "condition", "episode",
    "episode_uid", "band", "success", "steps", "episode_return", "reward_D",
    "mean_episode_e2e_ms", "final_nDTW", "spl", "ma100_reward", "ma100_e2e",
    "std100_e2e", "cum_total", "cum_success", "cum_failure",
    "cum_success_short", "cum_success_medium", "cum_success_long",
]
VALIDATION_COLUMNS = [
    "model", "deployment", "algorithm", "seed", "condition", "episode",
    "E2E_ms", "E2E_std_ms", "P95_E2E_ms", "RewardD", "SR", "OSR", "SPL",
    "nDTW", "delay_violation_rate", "mean_steps", "train_return_ma100",
    "train_e2e_ma100", "cum_success", "cum_failure", "cum_success_short",
    "cum_success_medium", "cum_success_long",
]


def _rolling(series, window=100, kind="mean"):
    rolling = series.rolling(window, min_periods=window)
    return rolling.mean() if kind == "mean" else rolling.std(ddof=0)


def _finite(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _load_run(model, algorithm, seed):
    path = run_dir(model, algorithm, seed)
    history = read_json(path / "training_history.json")
    result = read_json(path / "training_result.json")
    if history is None or not history.get("train"):
        return None
    train = pd.DataFrame(history["train"]).sort_values(
        "episode"
    ).reset_index(drop=True)
    train["model"] = model
    train["deployment"] = MODEL_LABELS[model]
    train["algorithm"] = algorithm
    train["seed"] = int(seed)
    train["condition"] = f"{MODEL_LABELS[model]}|{ALGORITHM_LABELS[algorithm]}"
    train["ma100_reward"] = _rolling(train["episode_return"])
    train["ma100_e2e"] = _rolling(train["mean_episode_e2e_ms"])
    train["std100_e2e"] = _rolling(
        train["mean_episode_e2e_ms"], kind="std"
    )
    success = train["success"].astype(int)
    band = train["band"].astype(str)
    train["cum_total"] = np.arange(1, len(train) + 1)
    train["cum_success"] = success.cumsum()
    train["cum_failure"] = train["cum_total"] - train["cum_success"]
    for name in ("short", "medium", "long"):
        mask = band == name
        train[f"cum_total_{name}"] = mask.astype(int).cumsum()
        train[f"cum_success_{name}"] = (
            (success.astype(bool) & mask).astype(int).cumsum()
        )
        train[f"cum_failure_{name}"] = (
            ((~success.astype(bool)) & mask).astype(int).cumsum()
        )
    train["cum_success_rate"] = (
        train["cum_success"] / train["cum_total"]
    )
    validation = pd.DataFrame(history.get("validation", []))
    if not validation.empty:
        validation["model"] = model
        validation["deployment"] = MODEL_LABELS[model]
        validation["algorithm"] = algorithm
        validation["seed"] = int(seed)
        validation["condition"] = (
            f"{MODEL_LABELS[model]}|{ALGORITHM_LABELS[algorithm]}"
        )
        validation = validation.sort_values("episode").reset_index(drop=True)
    rounds_path = path / "rounds.csv"
    rounds = pd.read_csv(rounds_path) if rounds_path.is_file() else None
    return {
        "model": model,
        "algorithm": algorithm,
        "seed": int(seed),
        "path": path,
        "history": history,
        "result": result or {},
        "train": train,
        "validation": validation,
        "rounds": rounds,
    }


def _run_training_summary(run):
    train = run["train"]
    result = run["result"]
    last = train.iloc[-1]
    stats = cumulative_stats(train.to_dict("records"))
    best_dev_e2e = _finite(result.get("best_dev_e2e_ms"))
    first_dev_e2e = None
    final_dev_e2e = None
    if not run["validation"].empty:
        first_dev_e2e = float(run["validation"].iloc[0]["E2E_ms"])
        final_dev_e2e = float(run["validation"].iloc[-1]["E2E_ms"])
        if best_dev_e2e is None:
            best_dev_e2e = float(run["validation"]["E2E_ms"].min())
    improvement = (
        100.0 * (first_dev_e2e - best_dev_e2e) / first_dev_e2e
        if first_dev_e2e and best_dev_e2e is not None else None
    )
    return {
        "model": run["model"],
        "deployment": MODEL_LABELS[run["model"]],
        "algorithm": run["algorithm"],
        "algorithm_label": ALGORITHM_LABELS[run["algorithm"]],
        "seed": run["seed"],
        "condition": f"{MODEL_LABELS[run['model']]}|"
                     f"{ALGORITHM_LABELS[run['algorithm']]}",
        "episodes": int(len(train)),
        "elapsed_h": float(result.get("elapsed_seconds", 0.0)) / 3600.0,
        "cum_total": stats["cum_total"],
        "cum_success": stats["cum_success"],
        "cum_failure": stats["cum_failure"],
        "cum_success_rate": stats["cum_success_rate"],
        "cum_success_short": stats["cum_success_short"],
        "cum_success_medium": stats["cum_success_medium"],
        "cum_success_long": stats["cum_success_long"],
        "cum_total_short": stats["cum_total_short"],
        "cum_total_medium": stats["cum_total_medium"],
        "cum_total_long": stats["cum_total_long"],
        "cum_failure_short": stats["cum_failure_short"],
        "cum_failure_medium": stats["cum_failure_medium"],
        "cum_failure_long": stats["cum_failure_long"],
        "best_dev_e2e_ms": best_dev_e2e,
        "first_dev_e2e_ms": first_dev_e2e,
        "final_dev_e2e_ms": final_dev_e2e,
        "dev_improvement_pct": improvement,
        "best_dev_episode": int(result.get("best_dev_episode", -1) or -1),
        "converged": bool(
            (result.get("convergence") or {}).get("converged", False)
        ),
        "convergence_status": (
            (result.get("convergence") or {}).get("status", "unknown")
        ),
        "final_ma100_reward": _finite(last.get("ma100_reward")),
        "final_ma100_e2e": _finite(last.get("ma100_e2e")),
        "mean_e2e_last1000": float(
            train.tail(1000)["mean_episode_e2e_ms"].mean()
        ),
        "mean_reward_last1000": float(
            train.tail(1000)["reward_D"].mean()
        ),
        "final_test_read": False,
    }


def _condition_training_summary(rows):
    output = []
    for model in MEC_MODELS:
        for algorithm in ALGORITHMS:
            subset = [row for row in rows
                      if row["model"] == model
                      and row["algorithm"] == algorithm]
            if not subset:
                continue
            item = {
                "model": model,
                "deployment": MODEL_LABELS[model],
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS[algorithm],
                "condition": f"{MODEL_LABELS[model]}|"
                             f"{ALGORITHM_LABELS[algorithm]}",
                "n_seeds": len(subset),
                "seeds": ",".join(str(row["seed"]) for row in subset),
            }
            for key in (
                "cum_success", "cum_failure", "cum_success_rate",
                "cum_success_short", "cum_success_medium",
                "cum_success_long", "best_dev_e2e_ms",
                "final_ma100_e2e", "mean_e2e_last1000",
                "mean_reward_last1000",
            ):
                values = np.asarray(
                    [float(row[key]) for row in subset
                     if row.get(key) is not None], dtype=np.float64
                )
                if values.size:
                    item[f"{key}_mean"] = float(values.mean())
                    item[f"{key}_std"] = float(values.std(ddof=0))
            output.append(item)
    return output


def _markdown(rows, conditions, eval_summary):
    lines = [
        "# Stage-2r 训练汇总表",
        "",
        f"- 奖励：方案 D；种子：{FORMAL_SEEDS}；每 run 4000 Episodes。",
        "- 训练成功/失败按环境 success 标志统计；band 为 short/medium/long。",
        "",
        "## 逐 run 训练统计",
        "",
        "| 部署 | 算法 | seed | 成功 | 失败 | 成功率 | 成功 S/M/L | "
        "失败 S/M/L | 最佳Dev E2E (ms) | 收敛 |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['deployment']} | {row['algorithm_label']} | "
            f"{row['seed']} | {row['cum_success']} | {row['cum_failure']} | "
            f"{100*row['cum_success_rate']:.2f}% | "
            f"{row['cum_success_short']}/"
            f"{row['cum_success_medium']}/{row['cum_success_long']} | "
            f"{row['cum_failure_short']}/"
            f"{row['cum_failure_medium']}/{row['cum_failure_long']} | "
            f"{row['best_dev_e2e_ms']:.3f} | "
            f"{'是' if row['converged'] else '否'} |"
        )
    lines += [
        "",
        "## 条件汇总（3 个种子的均值 ± 总体标准差）",
        "",
        "| 条件 | 成功 | 失败 | 成功率 | 成功 S/M/L | "
        "最佳Dev E2E (ms) | 末1000 E2E (ms) |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in conditions:
        lines.append(
            f"| {row['condition']} | "
            f"{row['cum_success_mean']:.1f} ± {row['cum_success_std']:.2f} | "
            f"{row['cum_failure_mean']:.1f} ± {row['cum_failure_std']:.2f} | "
            f"{100*row['cum_success_rate_mean']:.2f}% | "
            f"{row['cum_success_short_mean']:.1f}/"
            f"{row['cum_success_medium_mean']:.1f}/"
            f"{row['cum_success_long_mean']:.1f} | "
            f"{row['best_dev_e2e_ms_mean']:.3f} ± "
            f"{row['best_dev_e2e_ms_std']:.3f} | "
            f"{row['mean_e2e_last1000_mean']:.3f} ± "
            f"{row['mean_e2e_last1000_std']:.3f} |"
        )
    if eval_summary is not None:
        lines += [
            "",
            "## Dev-60 最终验证（来自 validation_summary.csv）",
            "",
            "| 条件 | E2E (ms) | RewardD | SR | OSR | nDTW |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in eval_summary.get("seed_aggregate", []):
            lines.append(
                f"| {row['deployment']}|{row['algorithm'].upper()} | "
                f"{row['E2E_ms_mean']:.3f} ± {row['E2E_ms_std']:.3f} | "
                f"{row['RewardD_mean']:.4f} ± {row['RewardD_std']:.4f} | "
                f"{100*row['SR_mean']:.2f}% | "
                f"{100*row['OSR_mean']:.2f}% | "
                f"{row['nDTW_mean']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for model in MEC_MODELS:
        for algorithm in ALGORITHMS:
            for seed in FORMAL_SEEDS:
                run = _load_run(model, algorithm, seed)
                if run is None:
                    print(f"[analyze] missing {model}/{algorithm}/s{seed}",
                          flush=True)
                    continue
                runs.append(run)
    if not runs:
        raise SystemExit("no completed Stage-2r runs")
    train_frames = []
    validation_frames = []
    rounds_frames = []
    for run in runs:
        train = run["train"].copy()
        train_frames.append(train[[column for column in TRAIN_COLUMNS
                                   if column in train.columns]])
        if not run["validation"].empty:
            val = run["validation"].copy()
            for column in VALIDATION_COLUMNS:
                if column not in val.columns:
                    val[column] = None
            validation_frames.append(val[VALIDATION_COLUMNS])
        if run["rounds"] is not None:
            rounds = run["rounds"].copy()
            rounds["model"] = run["model"]
            rounds["deployment"] = MODEL_LABELS[run["model"]]
            rounds["algorithm"] = run["algorithm"]
            rounds["seed"] = run["seed"]
            rounds["condition"] = (
                f"{MODEL_LABELS[run['model']]}|"
                f"{ALGORITHM_LABELS[run['algorithm']]}"
            )
            rounds_frames.append(rounds)
    training = pd.concat(train_frames, ignore_index=True)
    validation = (
        pd.concat(validation_frames, ignore_index=True)
        if validation_frames else pd.DataFrame(columns=VALIDATION_COLUMNS)
    )
    rounds = (
        pd.concat(rounds_frames, ignore_index=True)
        if rounds_frames else pd.DataFrame()
    )
    training.to_csv(EVIDENCE_DIR / "training_curves.csv", index=False)
    validation.to_csv(EVIDENCE_DIR / "validation_curves.csv", index=False)
    if not rounds.empty:
        rounds.to_csv(EVIDENCE_DIR / "per_200_episodes.csv", index=False)
    run_rows = [_run_training_summary(run) for run in runs]
    condition_rows = _condition_training_summary(run_rows)
    write_csv(EVIDENCE_DIR / "success_failure_summary.csv", run_rows)
    write_csv(EVIDENCE_DIR / "condition_training_summary.csv",
              condition_rows)
    lane_manifests = {}
    for model in MEC_MODELS:
        deployment = load_deployment(model)
        atomic_write_json(
            EVIDENCE_DIR / f"deployment_{model_slug(model)}.json",
            deployment,
        )
        manifest_path = (
            RUNS_ROOT / "lane_cache" / model_slug(model) / "manifest.json"
        )
        manifest = read_json(manifest_path)
        if manifest is not None:
            lane_manifests[model] = manifest
            atomic_write_json(
                EVIDENCE_DIR / f"lane_cache_{model_slug(model)}.json",
                manifest,
            )
    per_seed = None
    per_seed_path = EVIDENCE_DIR / "per_seed_metrics.csv"
    if per_seed_path.is_file():
        per_seed = pd.read_csv(per_seed_path)
        pairs = []
        for model in MEC_MODELS:
            for seed in FORMAL_SEEDS:
                subset = per_seed[
                    (per_seed["model"] == model)
                    & (per_seed["seed"] == seed)
                ]
                dqn = subset[subset["algorithm"] == "dqn"]
                ppo = subset[subset["algorithm"] == "ppo"]
                if dqn.empty or ppo.empty:
                    continue
                dqn = dqn.iloc[0]
                ppo = ppo.iloc[0]
                pairs.append({
                    "model": model,
                    "deployment": MODEL_LABELS[model],
                    "seed": int(seed),
                    "E2E_dqn": float(dqn["E2E_ms"]),
                    "E2E_ppo": float(ppo["E2E_ms"]),
                    "delta_E2E_dqn_minus_ppo": float(
                        dqn["E2E_ms"] - ppo["E2E_ms"]
                    ),
                    "RewardD_dqn": float(dqn["RewardD"]),
                    "RewardD_ppo": float(ppo["RewardD"]),
                    "delta_RewardD_ppo_minus_dqn": float(
                        ppo["RewardD"] - dqn["RewardD"]
                    ),
                    "SR_dqn": float(dqn["SR"]),
                    "SR_ppo": float(ppo["SR"]),
                    "SR_difference": float(ppo["SR"] - dqn["SR"]),
                })
        if pairs:
            write_csv(
                EVIDENCE_DIR / "paired_algorithm_differences.csv", pairs
            )
    eval_summary = read_json(EVIDENCE_DIR / "validation_summary.json")
    summary = {
        "stage": STAGE,
        "created_utc": utc_now(),
        "reward_scheme": REWARD_SCHEME,
        "seeds": list(FORMAL_SEEDS),
        "runs": run_rows,
        "conditions": condition_rows,
        "lane_cache_manifests": {
            model: {
                "tasks": manifest.get("tasks"),
                "elapsed_seconds": manifest.get("elapsed_seconds"),
                "shards": len(manifest.get("shards", [])),
                "artifact_hashes": manifest.get("artifact_hashes"),
            }
            for model, manifest in lane_manifests.items()
        },
        "eval_summary_present": eval_summary is not None,
        "final_test_read": False,
    }
    atomic_write_json(EVIDENCE_DIR / "summary.json", summary)
    (EVIDENCE_DIR / "SUMMARY_TABLES.md").write_text(
        _markdown(run_rows, condition_rows, eval_summary), encoding="utf-8"
    )
    if not args.no_figures:
        from v83_spatial_stage2.stage2r_10node_small_vs_tm.plot import (
            make_all,
        )
        names = make_all(training, validation, rounds, run_rows,
                         condition_rows, eval_summary, EVIDENCE_DIR)
        print(f"[analyze] figures: {names}", flush=True)
    print(
        f"[analyze] {len(runs)} runs, {len(condition_rows)} conditions, "
        f"eval_summary={eval_summary is not None}",
        flush=True,
    )
    print("STAGE2R_ANALYZE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
