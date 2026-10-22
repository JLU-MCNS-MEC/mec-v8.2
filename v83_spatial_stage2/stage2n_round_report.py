"""Print the six Stage-2n runs at one 200-Episode validation round.

    python v83_spatial_stage2/stage2n_round_report.py --episode 200

Reads only the run histories/progress files.  Missing runs are reported as
``pending`` instead of failing, so the table can be produced while the six
runs are slightly out of step.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import read_json, utc_now
from v83_spatial_stage2.stage2n_common import (
    AGENT_SEEDS,
    RUNS_DIR,
    convergence_status,
)


def _latest_validation(history, episode):
    for row in history.get("validation", []):
        if int(row["episode"]) == int(episode):
            return row
    return None


def _train_at(history, episode, window=100):
    rows = [
        row for row in history.get("train", [])
        if int(row["episode"]) <= int(episode)
    ][-int(window):]
    if not rows:
        return {}
    return {
        "train_return_ma100": float(np.mean([
            float(row["episode_return"]) for row in rows
        ])),
        "train_e2e_ma100": float(np.mean([
            float(row["mean_episode_e2e_ms"]) for row in rows
        ])),
        "train_success_ma100": float(np.mean([
            float(row["success"]) for row in rows
        ])),
        "train_steps_ma100": float(np.mean([
            int(row["steps"]) for row in rows
        ])),
    }


def collect(episode):
    rows = []
    for algorithm in ("dqn", "ppo"):
        for seed in AGENT_SEEDS:
            run_dir = RUNS_DIR / f"{algorithm}_s{seed}"
            history = read_json(run_dir / "training_history.json") or {}
            progress = read_json(run_dir / "progress.json") or {}
            record = _latest_validation(history, episode)
            train = _train_at(history, episode)
            validation_so_far = [
                item for item in history.get("validation", [])
                if int(item["episode"]) <= int(episode)
            ]
            conv = convergence_status(validation_so_far, reached_cap=False)
            rows.append({
                "algorithm": algorithm,
                "seed": int(seed),
                "record": record,
                "train": train,
                "progress": progress,
                "convergence": conv,
            })
    return rows


def _fmt(value, digits=2, suffix=""):
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def _markdown(rows, episode):
    lines = [
        f"# Stage-2n Ep{episode} 验证汇总",
        "",
        f"- 时间：{utc_now()}",
        "",
        "## 六个 run",
        "",
        "| Run | E2E (ms) | LatencyReturn | SR | OSR | SPL | nDTW | P95 (ms) | 违约率 | 步数 | 训练MA100 Return | 训练MA100 E2E |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rec = row["record"]
        name = f"{row['algorithm'].upper()} s{row['seed']}"
        if rec is None:
            lines.append(
                f"| {name} | pending | - | - | - | - | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {name} | {_fmt(rec['E2E_ms'])} | "
            f"{_fmt(rec['LatencyReturn'], 5)} | "
            f"{_fmt(100*rec['SR'], 2, '%')} | "
            f"{_fmt(100*rec['OSR'], 2, '%')} | "
            f"{_fmt(rec['SPL'], 4)} | {_fmt(rec['nDTW'], 4)} | "
            f"{_fmt(rec['P95_E2E_ms'])} | "
            f"{_fmt(100*rec['delay_violation_rate'], 2, '%')} | "
            f"{_fmt(rec['mean_steps'], 1)} | "
            f"{_fmt(row['train'].get('train_return_ma100'), 5)} | "
            f"{_fmt(row['train'].get('train_e2e_ma100'))} |"
        )
    lines += [
        "",
        "## DQN / PPO 多 seed 均值",
        "",
        "| 算法 | Seeds | E2E (ms) | LatencyReturn | SR | SPL | nDTW | P95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ("dqn", "ppo"):
        items = [
            row["record"] for row in rows
            if row["algorithm"] == algorithm and row["record"] is not None
        ]
        if not items:
            lines.append(f"| {algorithm.upper()} | 0 | pending | - | - | - | - | - |")
            continue
        def mean(key):
            return float(np.mean([float(item[key]) for item in items]))
        lines.append(
            f"| {algorithm.upper()} | {len(items)} | "
            f"{mean('E2E_ms'):.2f} | {mean('LatencyReturn'):.5f} | "
            f"{100*mean('SR'):.2f}% | {mean('SPL'):.4f} | "
            f"{mean('nDTW'):.4f} | {mean('P95_E2E_ms'):.2f} |"
        )
    lines += [
        "",
        "## 数值诊断",
        "",
        "| Run | DQN Q mean/max | DQN target/TD | DQN loss | PPO entropy | PPO KL | PPO value loss | PPO EV |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rec = row["record"]
        name = f"{row['algorithm'].upper()} s{row['seed']}"
        if rec is None:
            lines.append(f"| {name} | - | - | - | - | - | - | - |")
            continue
        if row["algorithm"] == "dqn":
            lines.append(
                f"| {name} | {_fmt(rec.get('q_mean'), 3)} / "
                f"{_fmt(rec.get('q_max'), 3)} | "
                f"{_fmt(rec.get('target_mean'), 3)} / "
                f"{_fmt(rec.get('td_error_abs_mean'), 4)} | "
                f"{_fmt(rec.get('train_loss_mean'), 6)} | - | - | - | - |"
            )
        else:
            lines.append(
                f"| {name} | - | - | - | "
                f"{_fmt(rec.get('policy_entropy'), 3)} | "
                f"{_fmt(rec.get('approx_kl'), 4)} | "
                f"{_fmt(rec.get('update_value_loss'), 4)} | "
                f"{_fmt(rec.get('explained_variance'), 3)} |"
            )
    records = [row["record"] for row in rows if row["record"] is not None]
    if records:
        spreads = {
            key: max(float(item[key]) for item in records)
            - min(float(item[key]) for item in records)
            for key in ("SR", "SPL", "nDTW")
        }
        lines += [
            "",
            "## 同质一致性",
            "",
            f"- 最大跨 run 差异：SR {spreads['SR']:.6f}，"
            f"SPL {spreads['SPL']:.6f}，nDTW {spreads['nDTW']:.6f}",
            f"- 导航一致性：`{'PASS' if all(value == 0.0 for value in spreads.values()) else 'FAIL'}`",
        ]
    lines += [
        "",
        "## 进度",
        "",
        "| Run | 当前 Episode | 最佳 Dev E2E (ms) | @Ep | 最近3次波动 | 最终差距 | 收敛 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        progress = row["progress"]
        name = f"{row['algorithm'].upper()} s{row['seed']}"
        conv = row["convergence"]
        enough = int(conv.get("validations", 0)) >= 3
        spread = (
            _fmt(100 * conv["last3_spread_fraction"], 2, "%")
            if enough else "-"
        )
        gap = (
            _fmt(100 * conv["final_gap_fraction"], 2, "%")
            if enough else "-"
        )
        lines.append(
            f"| {name} | {progress.get('current_episode', '-')} | "
            f"{_fmt(progress.get('best_dev_e2e_ms'))} | "
            f"{progress.get('best_dev_episode', '-')} | "
            f"{spread} | {gap} | "
            f"{'是' if conv.get('converged') else '否'} | "
            f"{progress.get('status', '-')} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--wait", action="store_true",
        help="Block until all six runs have this validation round.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print the final status line (for background watchers).",
    )
    args = parser.parse_args()
    while True:
        rows = collect(args.episode)
        missing = [
            f"{row['algorithm']}_s{row['seed']}"
            for row in rows if row["record"] is None
        ]
        if not missing or not args.wait:
            break
        if not args.quiet:
            print(f"[wait] Ep{args.episode} missing: {', '.join(missing)}")
        time.sleep(30)
    report = _markdown(rows, args.episode)
    if not args.quiet:
        print(report)
    if args.write:
        path = RUNS_DIR / "reports" / f"round_{args.episode:04d}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        print(f"ROUND {args.episode} WRITTEN {path}")
    elif args.quiet:
        print(f"ROUND {args.episode} READY")


if __name__ == "__main__":
    main()
