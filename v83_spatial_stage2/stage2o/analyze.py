"""Aggregate Stage-2o runs, compute comparisons, write CSV/figures/report data.

Usage:

    python v83_spatial_stage2/stage2o/analyze.py

Reads the six runs under ``runs/v83_spatial_stage2/stage2o/`` and writes

    evidence/training_curves.csv
    evidence/validation_curves.csv
    evidence/per_200_episodes.csv
    evidence/summary.json
    evidence/SUMMARY_TABLES.md
    evidence/fig01..fig09 PNG + figure_data/*.csv

The Test-400 pool is never opened.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import read_json, utc_now  # noqa: E402
from v83_spatial_stage2.stage2n_common import convergence_status  # noqa: E402
from v83_spatial_stage2.stage2o.plot import (  # noqa: E402
    ALGO_LABELS,
    SCHEME_LABELS,
    SCHEME_ORDER,
    make_all,
)
from v83_spatial_stage2.stage2o.rewards import (  # noqa: E402
    H_MAX,
    TAU_MS,
)
from v83_spatial_stage2.stage2o.train import RUNS_ROOT  # noqa: E402

ALGORITHMS = ("dqn", "ppo")
EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"
TRAINING_COLUMNS = [
    "scheme", "algorithm", "episode", "steps", "episode_return",
    "latency_return", "mean_episode_e2e_ms", "success", "spl", "final_nDTW",
    "terminal_reward", "terminal_correction", "terminal_share", "base_sum",
    "nonterminal_mean", "nonterminal_std", "nonterminal_abs_max",
    "step_abs_max", "reward_expected", "reward_abs_error",
    "q_mean", "q_max", "target_mean", "td_error_abs_mean", "loss",
    "epsilon", "policy_entropy", "value_mean", "value_abs_max",
    "update_value_loss", "explained_variance", "approx_kl",
    "clip_fraction", "ma100_reward", "ma100_e2e", "std100_reward",
    "std100_e2e",
]
VALIDATION_COLUMNS = [
    "scheme", "algorithm", "episode", "E2E_ms", "E2E_std_ms",
    "P95_E2E_ms", "LatencyReturn", "SR", "OSR", "SPL", "nDTW",
    "delay_violation_rate", "mean_steps", "train_return_ma100",
    "train_e2e_ma100", "q_mean", "q_max", "target_mean",
    "td_error_abs_mean", "train_loss_mean", "epsilon",
    "value_mean", "value_abs_max", "update_policy_loss",
    "update_value_loss", "update_entropy", "approx_kl", "clip_fraction",
    "explained_variance",
]
METRIC_KEYS = (
    "ma100_e2e_at_1000", "ma100_e2e_at_2000", "ma100_e2e_at_3000",
    "ma100_e2e_at_4000",
)


def _rolling(series, window=100, kind="mean"):
    rolling = series.rolling(window, min_periods=window)
    return rolling.mean() if kind == "mean" else rolling.std(ddof=0)


def _sanitize(value):
    """Replace non-finite floats with None so summary.json is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _coerce_numeric(frame):
    """Turn None/mixed columns into float NaN so matplotlib can plot them."""
    metadata = {
        "scheme", "algorithm", "episode_uid", "band", "reward_mode",
    }
    for column in frame.columns:
        if column in metadata:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_run(run_dir: Path, scheme: str, algorithm: str):
    run_dir = Path(run_dir)
    history = read_json(run_dir / "training_history.json")
    if history is None or not history.get("train"):
        return None
    train = pd.DataFrame(history["train"])
    if train.empty:
        return None
    train = train.sort_values("episode").reset_index(drop=True)
    train["scheme"] = scheme
    train["algorithm"] = algorithm
    train = _coerce_numeric(train)
    train["ma100_reward"] = _rolling(train["episode_return"])
    train["ma100_e2e"] = _rolling(train["mean_episode_e2e_ms"])
    train["std100_reward"] = _rolling(
        train["episode_return"], kind="std"
    )
    train["std100_e2e"] = _rolling(
        train["mean_episode_e2e_ms"], kind="std"
    )
    validation = pd.DataFrame(history.get("validation", []))
    if not validation.empty:
        validation = validation.sort_values("episode").reset_index(drop=True)
        validation["scheme"] = scheme
        validation["algorithm"] = algorithm
        validation = _coerce_numeric(validation)
    result = read_json(run_dir / "training_result.json") or {}
    rounds_path = run_dir / "rounds.csv"
    rounds = pd.read_csv(rounds_path) if rounds_path.is_file() else None
    return {
        "scheme": scheme,
        "algorithm": algorithm,
        "run_dir": run_dir,
        "history": history,
        "train": train,
        "validation": validation,
        "result": result,
        "rounds": rounds,
    }


def _at_or_before(train, column, episode):
    subset = train[train["episode"] <= int(episode)]
    if subset.empty:
        return None
    value = subset.iloc[-1][column]
    return None if pd.isna(value) else float(value)


def _first_reaching(train, column, target, start=100):
    subset = train[train["episode"] >= int(start)]
    hit = subset[subset[column] <= float(target)]
    if hit.empty:
        return None
    return int(hit.iloc[0]["episode"])


def _slope_last(train, column, last_episodes=2000):
    subset = train[train["episode"] > int(
        train["episode"].max()
    ) - int(last_episodes)]
    if len(subset) < 10:
        return None
    x = subset["episode"].to_numpy(dtype=float)
    y = subset[column].to_numpy(dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 10:
        return None
    slope = np.polyfit(x[mask], y[mask], 1)[0]
    return float(slope * 1000.0)


def summarize_run(run):
    train = run["train"]
    validation = run["validation"]
    result = run["result"]
    scheme = run["scheme"]
    algorithm = run["algorithm"]
    last_episode = int(train["episode"].max())
    reached_cap = last_episode >= 4000
    records = validation.to_dict("records") if not validation.empty else []
    convergence = (
        result.get("convergence")
        or convergence_status(records, reached_cap)
    )
    best_row = None
    if records:
        best_row = min(
            records, key=lambda row: (
                float(row["E2E_ms"]), -float(row.get("SR", 0.0)),
                int(row["episode"]),
            )
        )
    final_row = records[-1] if records else None
    first_row = records[0] if records else None
    ma100 = train["ma100_e2e"]
    min_ma100 = float(ma100.min()) if ma100.notna().any() else None
    time_to_best_5pct = (
        _first_reaching(train, "ma100_e2e", 1.05 * min_ma100)
        if min_ma100 is not None else None
    )
    subset = train[train["episode"] >= 200]
    auc = None
    if len(subset) > 2 and subset["ma100_e2e"].notna().sum() > 2:
        auc = float(np.trapezoid(
            subset["ma100_e2e"].to_numpy(dtype=float),
            subset["episode"].to_numpy(dtype=float),
        ) / (
            float(subset["episode"].max() - subset["episode"].min())
            or 1.0
        ))
    after400 = train[train["episode"] >= 400]
    std_e2e = (
        float(after400["std100_e2e"].mean())
        if after400["std100_e2e"].notna().any() else None
    )
    std_reward = (
        float(after400["std100_reward"].mean())
        if after400["std100_reward"].notna().any() else None
    )
    last500 = train.tail(500)
    reward_mean = float(last500["episode_return"].mean())
    reward_std = float(last500["episode_return"].std(ddof=0))
    reward_cv = (
        reward_std / abs(reward_mean) if abs(reward_mean) > 1e-12 else None
    )
    last200 = train.tail(200)
    last1000 = train.tail(1000)
    if algorithm == "dqn":
        consistency = last1000[["q_mean", "episode_return"]].dropna()
    else:
        consistency = last1000[["value_mean", "episode_return"]].dropna()
    if len(consistency) >= 10:
        value_column = (
            "q_mean" if algorithm == "dqn" else "value_mean"
        )
        return_corr = float(consistency[value_column].corr(
            consistency["episode_return"]
        ))
        return_ratio = float(
            consistency[value_column].mean()
            / consistency["episode_return"].mean()
        ) if abs(float(consistency["episode_return"].mean())) > 1e-12 else None
    else:
        return_corr = None
        return_ratio = None

    row = {
        "scheme": scheme,
        "scheme_label": SCHEME_LABELS[scheme],
        "algorithm": algorithm,
        "algorithm_label": ALGO_LABELS[algorithm],
        "episodes": last_episode,
        "elapsed_seconds": float(
            result.get("elapsed_seconds",
                       run["history"].get("elapsed_seconds", 0.0))
        ),
        "elapsed_h": float(
            result.get("elapsed_seconds",
                       run["history"].get("elapsed_seconds", 0.0))
        ) / 3600.0,
        "best_dev_e2e_ms": (
            float(best_row["E2E_ms"]) if best_row else None
        ),
        "best_dev_episode": (
            int(best_row["episode"]) if best_row else None
        ),
        "final_dev_e2e_ms": (
            float(final_row["E2E_ms"]) if final_row else None
        ),
        "first_dev_e2e_ms": (
            float(first_row["E2E_ms"]) if first_row else None
        ),
        "dev_improvement_pct": (
            100.0 * (float(first_row["E2E_ms"]) - float(best_row["E2E_ms"]))
            / float(first_row["E2E_ms"])
            if first_row and best_row and float(first_row["E2E_ms"]) else None
        ),
        "final_dev_sr": (
            float(final_row["SR"]) if final_row else None
        ),
        "final_dev_spl": (
            float(final_row["SPL"]) if final_row else None
        ),
        "final_dev_ndtw": (
            float(final_row["nDTW"]) if final_row else None
        ),
        "converged": bool(convergence.get("converged", False)),
        "convergence_status": str(convergence.get("status", "unknown")),
        "last3_spread_fraction": convergence.get("last3_spread_fraction"),
        "final_gap_fraction": convergence.get("final_gap_fraction"),
        "min_ma100_e2e": min_ma100,
        "episode_at_min_ma100": (
            int(train.loc[train["ma100_e2e"].idxmin(), "episode"])
            if ma100.notna().any() else None
        ),
        "final_ma100_e2e": _at_or_before(train, "ma100_e2e", last_episode),
        "auc_ma100_e2e": auc,
        "mean_std100_e2e_after400": std_e2e,
        "mean_std100_reward_after400": std_reward,
        "time_to_self_best_5pct": time_to_best_5pct,
        "slope_last2000_ms_per_1000ep": _slope_last(
            train, "ma100_e2e"
        ),
        "reward_cv_last500": reward_cv,
        "reward_std_last500": reward_std,
        "dev_last5_std": (
            float(validation.tail(5)["E2E_ms"].std(ddof=0))
            if len(validation) >= 2 else None
        ),
        "dev_last5_range_pct": (
            100.0 * (
                float(validation.tail(5)["E2E_ms"].max()
                      - validation.tail(5)["E2E_ms"].min())
            ) / float(validation.tail(5)["E2E_ms"].mean())
            if len(validation) >= 2 else None
        ),
        "mean_steps": float(train["steps"].mean()),
        "mean_T_over_H": float(train["steps"].mean()) / H_MAX,
        "episode_return_final": float(
            last200["episode_return"].mean()
        ),
        "return_corr_last1000": return_corr,
        "return_ratio_last1000": return_ratio,
        "tau_ms": TAU_MS,
    }
    for episode in (1000, 2000, 3000, 4000):
        row[f"ma100_e2e_at_{episode}"] = _at_or_before(
            train, "ma100_e2e", episode
        )
        row[f"ma100_reward_at_{episode}"] = _at_or_before(
            train, "ma100_reward", episode
        )
    if algorithm == "dqn":
        row.update({
            "q_mean_final": float(last200["q_mean"].mean())
            if last200["q_mean"].notna().any() else None,
            "q_max_final": float(last200["q_max"].mean())
            if last200["q_max"].notna().any() else None,
            "target_mean_final": float(last200["target_mean"].mean())
            if last200["target_mean"].notna().any() else None,
            "td_loss_final": float(last200["td_error_abs_mean"].mean())
            if last200["td_error_abs_mean"].notna().any() else None,
            "epsilon_final": float(last200["epsilon"].iloc[-1])
            if last200["epsilon"].notna().any() else None,
        })
        row["q_target_gap_final"] = (
            row["q_mean_final"] - row["target_mean_final"]
            if row["q_mean_final"] is not None
            and row["target_mean_final"] is not None else None
        )
    else:
        row.update({
            "value_loss_final": float(
                last200["update_value_loss"].mean()
            ) if last200["update_value_loss"].notna().any() else None,
            "explained_variance_final": float(
                last200["explained_variance"].mean()
            ) if last200["explained_variance"].notna().any() else None,
            "entropy_final": float(last200["policy_entropy"].mean())
            if last200["policy_entropy"].notna().any() else None,
            "kl_final": float(last200["approx_kl"].mean())
            if last200["approx_kl"].notna().any() else None,
            "value_mean_final": float(last200["value_mean"].mean())
            if last200["value_mean"].notna().any() else None,
        })
    terminal_correction = train["terminal_correction"]
    row.update({
        "terminal_correction_mean": float(terminal_correction.mean()),
        "terminal_correction_median": float(
            terminal_correction.median()
        ),
        "terminal_correction_max": float(terminal_correction.max()),
        "terminal_share_mean": float(train["terminal_share"].mean()),
        "terminal_share_median": float(train["terminal_share"].median()),
        "nonterminal_abs_max_mean": float(
            train["nonterminal_abs_max"].mean()
        ),
        "terminal_to_nonterminal_ratio": (
            float(train["terminal_correction"].median())
            / max(float(train["nonterminal_abs_max"].mean()), 1e-12)
        ),
    })
    return row


def _markdown_table(rows, columns, headers=None):
    if not rows:
        return "（无数据）"
    headers = headers or columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None:
                text = "—"
            elif isinstance(value, bool):
                text = "是" if value else "否"
            elif isinstance(value, float):
                text = f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
            else:
                text = str(value)
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_comparison(summary):
    comparison = {}
    for algorithm in ALGORITHMS:
        subset = [row for row in summary if row["algorithm"] == algorithm]
        if not subset:
            continue
        by_final = sorted(
            subset,
            key=lambda row: (
                row["final_ma100_e2e"]
                if row["final_ma100_e2e"] is not None else float("inf")
            ),
        )
        by_auc = sorted(
            subset,
            key=lambda row: (
                row["auc_ma100_e2e"]
                if row["auc_ma100_e2e"] is not None else float("inf")
            ),
        )
        by_best = sorted(
            subset,
            key=lambda row: (
                row["best_dev_e2e_ms"]
                if row["best_dev_e2e_ms"] is not None else float("inf")
            ),
        )
        comparison[algorithm] = {
            "rank_final_ma100_e2e": [
                row["scheme"] for row in by_final
            ],
            "rank_auc_ma100_e2e": [row["scheme"] for row in by_auc],
            "rank_best_dev_e2e": [row["scheme"] for row in by_best],
            "best_final_ma100_e2e": by_final[0]["scheme"],
            "best_auc_ma100_e2e": by_auc[0]["scheme"],
            "best_dev_e2e": by_best[0]["scheme"],
            "converged": [
                row["scheme"] for row in subset if row["converged"]
            ],
        }
    return comparison


def _summary_markdown(summary, comparison):
    overview = [{
        "scheme": row["scheme_label"],
        "algorithm": row["algorithm_label"],
        "best_dev": row["best_dev_e2e_ms"],
        "best_ep": row["best_dev_episode"],
        "final_dev": row["final_dev_e2e_ms"],
        "converged": row["converged"],
        "elapsed_h": row["elapsed_h"],
    } for row in summary]
    curve = [{
        "scheme": row["scheme_label"],
        "algorithm": row["algorithm_label"],
        "m1000": row["ma100_e2e_at_1000"],
        "m2000": row["ma100_e2e_at_2000"],
        "m3000": row["ma100_e2e_at_3000"],
        "m4000": row["ma100_e2e_at_4000"],
        "auc": row["auc_ma100_e2e"],
        "std": row["mean_std100_e2e_after400"],
        "t5": row["time_to_self_best_5pct"],
    } for row in summary]
    diagnostics = []
    for row in summary:
        if row["algorithm"] == "dqn":
            diagnostics.append({
                "scheme": row["scheme_label"],
                "algorithm": "DQN",
                "q_mean": row.get("q_mean_final"),
                "q_max": row.get("q_max_final"),
                "target": row.get("target_mean_final"),
                "td": row.get("td_loss_final"),
            })
        else:
            diagnostics.append({
                "scheme": row["scheme_label"],
                "algorithm": "PPO",
                "value_loss": row.get("value_loss_final"),
                "ev": row.get("explained_variance_final"),
                "entropy": row.get("entropy_final"),
                "kl": row.get("kl_final"),
            })
    ranking = []
    for algorithm, payload in comparison.items():
        ranking.append({
            "algorithm": ALGO_LABELS[algorithm],
            "final": " > ".join(
                SCHEME_LABELS[item]
                for item in payload["rank_final_ma100_e2e"]
            ),
            "auc": " > ".join(
                SCHEME_LABELS[item]
                for item in payload["rank_auc_ma100_e2e"]
            ),
            "best_dev": " > ".join(
                SCHEME_LABELS[item]
                for item in payload["rank_best_dev_e2e"]
            ),
        })
    return "\n\n".join([
        "# Stage-2o 汇总表\n",
        "## 表 1 训练概况\n\n" + _markdown_table(
            overview,
            ["scheme", "algorithm", "best_dev", "best_ep", "final_dev",
             "converged", "elapsed_h"],
            ["方案", "算法", "最佳 Dev E2E (ms)", "@Ep", "最终 Dev E2E (ms)",
             "收敛", "耗时 (h)"],
        ),
        "## 表 2 MA100 E2E 轨迹与稳定性\n\n" + _markdown_table(
            curve,
            ["scheme", "algorithm", "m1000", "m2000", "m3000", "m4000",
             "auc", "std", "t5"],
            ["方案", "算法", "Ep1000", "Ep2000", "Ep3000", "Ep4000",
             "AUC (ms)", "滚动σ (ms)", "达到自身最佳5%的Ep"],
        ),
        "## 表 3 训练诊断（最后 200 Episode）\n\n" + _markdown_table(
            diagnostics,
            ["scheme", "algorithm", "q_mean", "q_max", "target", "td",
             "value_loss", "ev", "entropy", "kl"],
            ["方案", "算法", "Q mean", "Q max", "Target", "TD Loss",
             "Value Loss", "EV", "Entropy", "KL"],
        ),
        "## 表 4 排名（按不同指标，越靠前越好）\n\n" + _markdown_table(
            ranking,
            ["algorithm", "final", "auc", "best_dev"],
            ["算法", "最终 MA100 E2E", "MA100 E2E AUC", "最佳 Dev E2E"],
        ),
    ]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--evidence-dir", type=Path,
                        default=EVIDENCE_DIR)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "figure_data").mkdir(parents=True, exist_ok=True)

    runs = []
    missing = []
    for scheme in SCHEME_ORDER:
        for algorithm in ALGORITHMS:
            run_dir = runs_root / scheme / f"{algorithm}_s0"
            run = load_run(run_dir, scheme, algorithm)
            if run is None:
                missing.append(str(run_dir))
                continue
            runs.append(run)
            print(
                f"[analyze] {scheme}/{algorithm}: "
                f"{len(run['train'])} episodes, "
                f"{len(run['validation'])} validations",
                flush=True,
            )
    if missing:
        print("[analyze] missing runs:\n  " + "\n  ".join(missing),
              flush=True)
    if not runs:
        raise SystemExit("no Stage-2o runs found")

    training = pd.concat([run["train"] for run in runs], ignore_index=True)
    training = training.drop_duplicates(
        subset=["scheme", "algorithm", "episode"], keep="last"
    )
    training = training.sort_values(
        ["algorithm", "scheme", "episode"]
    ).reset_index(drop=True)
    logs = []
    for run in runs:
        if not run["validation"].empty:
            logs.append(run["validation"])
    validation = (
        pd.concat(logs, ignore_index=True)
        if logs else pd.DataFrame()
    )
    if not validation.empty:
        validation = validation.drop_duplicates(
            subset=["scheme", "algorithm", "episode"], keep="last"
        )
        validation = validation.sort_values(
            ["algorithm", "scheme", "episode"]
        ).reset_index(drop=True)
    rounds = [
        run["rounds"] for run in runs if run["rounds"] is not None
    ]
    per200 = pd.concat(rounds, ignore_index=True) if rounds else pd.DataFrame()
    if not per200.empty:
        per200 = per200.drop_duplicates(
            subset=["scheme", "algorithm", "seed", "episode"], keep="last"
        )

    training_columns = [
        column for column in TRAINING_COLUMNS if column in training.columns
    ]
    training[training_columns].to_csv(
        evidence / "training_curves.csv", index=False
    )
    if not validation.empty:
        validation_columns = [
            column for column in VALIDATION_COLUMNS
            if column in validation.columns
        ]
        validation[validation_columns].to_csv(
            evidence / "validation_curves.csv", index=False
        )
    if not per200.empty:
        per200.to_csv(evidence / "per_200_episodes.csv", index=False)

    summary = [summarize_run(run) for run in runs]
    summary.sort(key=lambda row: (row["algorithm"], row["scheme"]))
    comparison = build_comparison(summary)
    payload = {
        "stage": "stage2o",
        "generated_utc": utc_now(),
        "tau_ms": TAU_MS,
        "horizon": H_MAX,
        "seed": 0,
        "runs": summary,
        "comparison": comparison,
        "missing_runs": missing,
        "test_read": False,
    }
    (evidence / "summary.json").write_text(
        json.dumps(_sanitize(payload), ensure_ascii=False, indent=2,
                   allow_nan=False),
        encoding="utf-8",
    )
    (evidence / "SUMMARY_TABLES.md").write_text(
        _summary_markdown(summary, comparison), encoding="utf-8"
    )
    if not args.no_figures:
        figures = make_all(training, validation, summary, evidence)
        print(f"[analyze] figures: {', '.join(figures)}", flush=True)

    print("\n[analyze] 训练概况", flush=True)
    for row in summary:
        best = (
            f"{row['best_dev_e2e_ms']:.2f}"
            if row["best_dev_e2e_ms"] is not None else "N/A"
        )
        final = (
            f"{row['final_dev_e2e_ms']:.2f}"
            if row["final_dev_e2e_ms"] is not None else "N/A"
        )
        ma100 = (
            f"{row['final_ma100_e2e']:.2f}"
            if row["final_ma100_e2e"] is not None else "N/A"
        )
        print(
            f"  {row['scheme_label']}/{row['algorithm_label']}: "
            f"best Dev E2E={best}ms "
            f"@Ep{row['best_dev_episode']}, "
            f"final={final}ms, "
            f"converged={row['converged']}, "
            f"final MA100 E2E={ma100}ms",
            flush=True,
        )
    print("[analyze] comparison:", flush=True)
    for algorithm, payload_alg in comparison.items():
        print(
            f"  {ALGO_LABELS[algorithm]}: "
            f"final MA100 最优={SCHEME_LABELS[payload_alg['best_final_ma100_e2e']]}, "
            f"AUC 最优={SCHEME_LABELS[payload_alg['best_auc_ma100_e2e']]}, "
            f"最佳 Dev 最优={SCHEME_LABELS[payload_alg['best_dev_e2e']]}, "
            f"收敛={payload_alg['converged']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
