"""Aggregate Stage-2p runs, compute per-seed/paired metrics, write figures."""

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
from v83_spatial_stage2.stage2p.plot import (  # noqa: E402
    GREEDY_DEV_E2E,
    SCHEME_LABELS,
    SCHEME_ORDER,
    SEEDS,
    make_all,
)
from v83_spatial_stage2.stage2p.train import RUNS_ROOT  # noqa: E402

EVIDENCE = Path(__file__).resolve().parent / "evidence"
ALGORITHM = "ppo"
METADATA = {"scheme", "algorithm", "episode_uid", "band", "reward_mode"}


def _coerce_numeric(frame):
    for column in frame.columns:
        if column in METADATA:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _sanitize(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _rolling(series, window=100):
    return series.rolling(window, min_periods=window).mean()


def _at_or_before(train, column, episode):
    subset = train[train["episode"] <= int(episode)]
    if subset.empty:
        return None
    value = subset.iloc[-1][column]
    return None if pd.isna(value) else float(value)


def load_run(scheme, seed, runs_root=RUNS_ROOT):
    run_dir = Path(runs_root) / scheme / f"ppo_s{seed}"
    history = read_json(run_dir / "training_history.json")
    if history is None or not history.get("train"):
        return None
    train = pd.DataFrame(history["train"]).sort_values("episode")
    train = train.reset_index(drop=True)
    train["scheme"] = scheme
    train["seed"] = int(seed)
    train = _coerce_numeric(train)
    train["ma100_reward"] = _rolling(train["episode_return"])
    train["ma100_e2e"] = _rolling(train["mean_episode_e2e_ms"])
    train["std100_reward"] = train["episode_return"].rolling(
        100, min_periods=100
    ).std(ddof=0)
    train["std100_e2e"] = train["mean_episode_e2e_ms"].rolling(
        100, min_periods=100
    ).std(ddof=0)
    validation = pd.DataFrame(history.get("validation", []))
    if not validation.empty:
        validation = validation.sort_values("episode").reset_index(drop=True)
        validation["scheme"] = scheme
        validation["seed"] = int(seed)
        validation = _coerce_numeric(validation)
    return {
        "scheme": scheme,
        "seed": int(seed),
        "run_dir": run_dir,
        "history": history,
        "train": train,
        "validation": validation,
        "result": read_json(run_dir / "training_result.json") or {},
    }


def summarize_run(run):
    train = run["train"]
    validation = run["validation"]
    result = run["result"]
    scheme = run["scheme"]
    seed = run["seed"]
    last_episode = int(train["episode"].max())
    reached_cap = last_episode >= 4000
    records = validation.to_dict("records") if not validation.empty else []
    convergence = (
        result.get("convergence")
        or convergence_status(records, reached_cap)
    )
    best_row = (
        min(records, key=lambda row: (
            float(row["E2E_ms"]), -float(row.get("SR", 0.0)),
            int(row["episode"]),
        )) if records else None
    )
    final_row = records[-1] if records else None
    ma100 = train["ma100_e2e"]
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
    last500 = train.tail(500)
    last500_std = float(
        last500["mean_episode_e2e_ms"].std(ddof=0)
    )
    last200 = train.tail(200)
    best = float(best_row["E2E_ms"]) if best_row else None
    final = float(final_row["E2E_ms"]) if final_row else None
    degradation = (
        100.0 * (final - best) / best if best and final is not None else None
    )
    dev_last5 = (
        float(validation.tail(5)["E2E_ms"].std(ddof=0))
        if len(validation) >= 2 else None
    )
    dev_last5_range = (
        float(validation.tail(5)["E2E_ms"].max()
              - validation.tail(5)["E2E_ms"].min())
        if len(validation) >= 2 else None
    )
    row = {
        "scheme": scheme,
        "scheme_label": SCHEME_LABELS[scheme],
        "seed": seed,
        "episodes": last_episode,
        "elapsed_h": float(
            result.get("elapsed_seconds",
                       run["history"].get("elapsed_seconds", 0.0))
        ) / 3600.0,
        "best_dev_e2e_ms": best,
        "best_dev_episode": (
            int(best_row["episode"]) if best_row else None
        ),
        "final_dev_e2e_ms": final,
        "final_ma100_e2e": _at_or_before(
            train, "ma100_e2e", last_episode
        ),
        "min_ma100_e2e": (
            float(ma100.min()) if ma100.notna().any() else None
        ),
        "auc_ma100_e2e": auc,
        "last500_e2e_std": last500_std,
        "degradation_pct": degradation,
        "gap_to_greedy_ms": (
            best - GREEDY_DEV_E2E if best is not None else None
        ),
        "gap_to_greedy_pct": (
            100.0 * (best - GREEDY_DEV_E2E) / GREEDY_DEV_E2E
            if best is not None else None
        ),
        "dev_last5_std": dev_last5,
        "dev_last5_range": dev_last5_range,
        "converged": bool(convergence.get("converged", False)),
        "convergence_status": str(convergence.get("status", "unknown")),
        "last3_spread_fraction": convergence.get("last3_spread_fraction"),
        "final_gap_fraction": convergence.get("final_gap_fraction"),
        "value_loss_final": (
            float(last200["update_value_loss"].mean())
            if last200["update_value_loss"].notna().any() else None
        ),
        "explained_variance_final": (
            float(last200["explained_variance"].mean())
            if last200["explained_variance"].notna().any() else None
        ),
        "entropy_final": (
            float(last200["policy_entropy"].mean())
            if last200["policy_entropy"].notna().any() else None
        ),
        "kl_final": (
            float(last200["approx_kl"].mean())
            if last200["approx_kl"].notna().any() else None
        ),
        "value_mean_final": (
            float(last200["value_mean"].mean())
            if last200["value_mean"].notna().any() else None
        ),
        "mean_steps": float(train["steps"].mean()),
        "run_dir": str(run["run_dir"]),
        "final_test_read": False,
    }
    for episode in (1000, 2000, 3000, 4000):
        row[f"ma100_e2e_at_{episode}"] = _at_or_before(
            train, "ma100_e2e", episode
        )
    return row


def aggregate(per_seed):
    output = []
    for scheme in SCHEME_ORDER:
        rows = [row for row in per_seed if row["scheme"] == scheme]
        if not rows:
            continue
        row = {
            "scheme": scheme,
            "scheme_label": SCHEME_LABELS[scheme],
            "seeds": len(rows),
            "converged_count": sum(1 for item in rows if item["converged"]),
        }
        for key in (
            "best_dev_e2e_ms", "final_dev_e2e_ms", "auc_ma100_e2e",
            "min_ma100_e2e", "final_ma100_e2e", "last500_e2e_std",
            "degradation_pct", "gap_to_greedy_ms", "dev_last5_std",
            "value_loss_final", "explained_variance_final",
            "entropy_final", "kl_final",
        ):
            values = [
                float(item[key]) for item in rows
                if item.get(key) is not None
                and math.isfinite(float(item[key]))
            ]
            row[f"{key}_mean"] = float(np.mean(values)) if values else None
            row[f"{key}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        output.append(row)
    return output


def paired_differences(per_seed):
    output = []
    for seed in SEEDS:
        baseline = next(
            (row for row in per_seed
             if row["scheme"] == "baseline" and row["seed"] == seed),
            None,
        )
        scheme_d = next(
            (row for row in per_seed
             if row["scheme"] == "D" and row["seed"] == seed),
            None,
        )
        if not (baseline and scheme_d):
            continue
        row = {"seed": int(seed)}
        for key in (
            "best_dev_e2e_ms", "final_dev_e2e_ms", "auc_ma100_e2e",
            "min_ma100_e2e", "final_ma100_e2e", "last500_e2e_std",
            "degradation_pct", "gap_to_greedy_ms", "value_loss_final",
            "explained_variance_final", "entropy_final", "kl_final",
        ):
            left = baseline.get(key)
            right = scheme_d.get(key)
            row[f"delta_{key}"] = (
                float(right) - float(left)
                if left is not None and right is not None else None
            )
        output.append(row)
    return output


def _fmt(value, digits=2):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _table(rows, columns, headers):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            _fmt(row.get(column)) for column in columns
        ) + " |")
    return "\n".join(lines)


def _markdown(per_seed, aggregate_rows, paired):
    seed_table = [{
        "scheme": row["scheme_label"],
        "seed": row["seed"],
        "best": row["best_dev_e2e_ms"],
        "best_ep": row["best_dev_episode"],
        "final": row["final_dev_e2e_ms"],
        "auc": row["auc_ma100_e2e"],
        "min": row["min_ma100_e2e"],
        "std500": row["last500_e2e_std"],
        "degrade": row["degradation_pct"],
        "gap": row["gap_to_greedy_ms"],
        "converged": row["converged"],
    } for row in per_seed]
    aggregate_table = [{
        "scheme": row["scheme_label"],
        "best": f"{_fmt(row['best_dev_e2e_ms_mean'])} ± "
                f"{_fmt(row['best_dev_e2e_ms_std'])}",
        "final": f"{_fmt(row['final_dev_e2e_ms_mean'])} ± "
                 f"{_fmt(row['final_dev_e2e_ms_std'])}",
        "auc": f"{_fmt(row['auc_ma100_e2e_mean'])} ± "
               f"{_fmt(row['auc_ma100_e2e_std'])}",
        "std500": f"{_fmt(row['last500_e2e_std_mean'])} ± "
                  f"{_fmt(row['last500_e2e_std_std'])}",
        "degrade": f"{_fmt(row['degradation_pct_mean'])} ± "
                   f"{_fmt(row['degradation_pct_std'])}",
        "gap": f"{_fmt(row['gap_to_greedy_ms_mean'])} ± "
               f"{_fmt(row['gap_to_greedy_ms_std'])}",
        "converged": f"{row['converged_count']}/{row['seeds']}",
    } for row in aggregate_rows]
    paired_table = [{
        "seed": row["seed"],
        "dbest": row["delta_best_dev_e2e_ms"],
        "dfinal": row["delta_final_dev_e2e_ms"],
        "dauc": row["delta_auc_ma100_e2e"],
        "dstd500": row["delta_last500_e2e_std"],
        "ddegrade": row["delta_degradation_pct"],
        "dgap": row["delta_gap_to_greedy_ms"],
    } for row in paired]
    return "\n\n".join([
        "# Stage-2p 数据表\n",
        "## 表 1 逐 seed 结果\n\n" + _table(
            seed_table,
            ["scheme", "seed", "best", "best_ep", "final", "auc", "min",
             "std500", "degrade", "gap", "converged"],
            ["方案", "seed", "最佳 Dev/ms", "@Ep", "最终 Dev/ms",
             "MA100 AUC", "最小 MA100", "末500 E2E σ",
             "最终比最佳差/%", "距 Greedy/ms", "收敛"],
        ),
        "## 表 2 三种子汇总\n\n" + _table(
            aggregate_table,
            ["scheme", "best", "final", "auc", "std500", "degrade",
             "gap", "converged"],
            ["方案", "最佳 Dev (mean±std)", "最终 Dev", "MA100 AUC",
             "末500 E2E σ", "退化/%", "距 Greedy/ms", "收敛"],
        ),
        "## 表 3 配对差值（D − Baseline）\n\n" + _table(
            paired_table,
            ["seed", "dbest", "dfinal", "dauc", "dstd500", "ddegrade",
             "dgap"],
            ["seed", "Δ最佳 Dev", "Δ最终 Dev", "ΔMA100 AUC",
             "Δ末500 E2E σ", "Δ退化/%", "Δ距 Greedy"],
        ),
        "注意：负的 Δ Dev/AUC/E2E σ 表示方案 D 更好；"
        "Δ退化 为负表示 D 的退化更小。",
    ]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "figure_data").mkdir(parents=True, exist_ok=True)

    runs = []
    missing = []
    for scheme in SCHEME_ORDER:
        for seed in SEEDS:
            run = load_run(scheme, seed, runs_root)
            if run is None:
                missing.append(str(runs_root / scheme / f"ppo_s{seed}"))
                continue
            runs.append(run)
            print(
                f"[analyze] {scheme}/s{seed}: {len(run['train'])} episodes, "
                f"{len(run['validation'])} validations", flush=True,
            )
    if missing:
        print("[analyze] missing runs:\n  " + "\n  ".join(missing),
              flush=True)
    if not runs:
        raise SystemExit("no Stage-2p runs found")

    training = pd.concat([run["train"] for run in runs], ignore_index=True)
    training = training.drop_duplicates(
        subset=["scheme", "seed", "episode"], keep="last"
    ).sort_values(["scheme", "seed", "episode"]).reset_index(drop=True)
    logs = [run["validation"] for run in runs
            if not run["validation"].empty]
    validation = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    if not validation.empty:
        validation = validation.drop_duplicates(
            subset=["scheme", "seed", "episode"], keep="last"
        ).sort_values(["scheme", "seed", "episode"]).reset_index(drop=True)

    training.to_csv(evidence / "training_curves.csv", index=False)
    if not validation.empty:
        validation.to_csv(evidence / "validation_curves.csv", index=False)
    per_seed = [summarize_run(run) for run in runs]
    per_seed.sort(key=lambda row: (SCHEME_ORDER.index(row["scheme"]),
                                   row["seed"]))
    aggregate_rows = aggregate(per_seed)
    paired = paired_differences(per_seed)
    pd.DataFrame(per_seed).to_csv(
        evidence / "per_seed_metrics.csv", index=False
    )
    pd.DataFrame(aggregate_rows).to_csv(
        evidence / "seed_aggregate.csv", index=False
    )
    pd.DataFrame(paired).to_csv(
        evidence / "paired_differences.csv", index=False
    )
    (evidence / "SUMMARY_TABLES.md").write_text(
        _markdown(per_seed, aggregate_rows, paired), encoding="utf-8"
    )
    payload = {
        "stage": "stage2p",
        "generated_utc": utc_now(),
        "algorithm": ALGORITHM,
        "seeds": list(SEEDS),
        "runs": per_seed,
        "aggregate": aggregate_rows,
        "paired_differences": paired,
        "missing_runs": missing,
        "greedy_dev_e2e_ms": GREEDY_DEV_E2E,
        "test_read": False,
    }
    (evidence / "summary.json").write_text(
        json.dumps(_sanitize(payload), ensure_ascii=False, indent=2,
                   allow_nan=False),
        encoding="utf-8",
    )
    (evidence / "per_200_episodes.csv").write_text("", encoding="utf-8")
    rounds = []
    for scheme in SCHEME_ORDER:
        for seed in SEEDS:
            path = runs_root / scheme / f"ppo_s{seed}" / "rounds.csv"
            if path.is_file():
                rounds.append(pd.read_csv(path))
    if rounds:
        pd.concat(rounds, ignore_index=True).to_csv(
            evidence / "per_200_episodes.csv", index=False
        )
    if not args.no_figures:
        figures = make_all(
            training, validation, per_seed, paired, evidence
        )
        print(f"[analyze] figures: {', '.join(figures)}", flush=True)
    print("\n[analyze] 逐 seed 结果", flush=True)
    for row in per_seed:
        print(
            f"  {row['scheme_label']}/s{row['seed']}: "
            f"best {_fmt(row['best_dev_e2e_ms'])} @Ep"
            f"{row['best_dev_episode']}, final "
            f"{_fmt(row['final_dev_e2e_ms'])}, AUC "
            f"{_fmt(row['auc_ma100_e2e'])}, "
            f"degrade {_fmt(row['degradation_pct'])}%, "
            f"converged={row['converged']}", flush=True,
        )
    print("[analyze] 配对差值（D − Baseline）", flush=True)
    for row in paired:
        print(
            f"  seed {row['seed']}: Δbest "
            f"{_fmt(row['delta_best_dev_e2e_ms'])}, Δfinal "
            f"{_fmt(row['delta_final_dev_e2e_ms'])}, ΔAUC "
            f"{_fmt(row['delta_auc_ma100_e2e'])}, Δstd500 "
            f"{_fmt(row['delta_last500_e2e_std'])}", flush=True,
        )


if __name__ == "__main__":
    main()
