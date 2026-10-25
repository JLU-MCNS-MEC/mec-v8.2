"""Live status of the six Stage-2s formal runs (read-only)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import read_json  # noqa: E402
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (  # noqa: E402
    ALGORITHMS,
    FORMAL_SEEDS,
    RUNS_ROOT,
    run_dir,
)


def _run_status(algorithm, seed):
    path = run_dir(algorithm, seed)
    progress = read_json(path / "progress.json")
    result = read_json(path / "training_result.json")
    history = read_json(path / "training_history.json")
    episode = 0
    if progress:
        episode = int(progress.get("current_episode") or 0)
    if result:
        episode = int(result.get("episodes") or episode)
    best = None
    last = None
    small_ratio = None
    status = "未开始"
    if result and result.get("status") == "completed":
        status = "完成"
    elif progress and progress.get("status") == "training":
        status = "训练中"
    elif history and history.get("train"):
        status = "中断/待恢复"
    if result:
        best = result.get("best_dev_e2e_ms")
        last = result.get("last_validation")
        small_ratio = result.get("training_small_ratio")
        conv = result.get("convergence", {}).get("status")
    elif progress:
        best = progress.get("best_dev_e2e_ms")
        last = progress.get("last_validation")
        conv = None
    else:
        conv = None
    if history and history.get("train"):
        recent = history["train"][-100:]
        small_ratio = sum(
            float(row.get("small_ratio", 0.0)) for row in recent
        ) / len(recent)
    return {
        "algorithm": algorithm,
        "seed": int(seed),
        "status": status,
        "episode": episode,
        "best_dev_e2e_ms": best,
        "small_ratio": small_ratio,
        "last_validation": last,
        "convergence": conv,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args(argv)
    while True:
        rows = [
            _run_status(algorithm, seed)
            for algorithm in ALGORITHMS for seed in FORMAL_SEEDS
        ]
        print(
            f"\nStage-2s 状态  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(
            f"{'算法':>4} {'seed':>4} {'状态':>8} {'Ep':>6} "
            f"{'最佳DevE2E':>12} {'Small比例':>10} {'收敛':>14}"
        )
        for row in rows:
            best = (
                f"{row['best_dev_e2e_ms']:.2f}"
                if row["best_dev_e2e_ms"] is not None else "-"
            )
            ratio = (
                f"{100*row['small_ratio']:.1f}%"
                if row["small_ratio"] is not None else "-"
            )
            print(
                f"{row['algorithm'].upper():>4} {row['seed']:>4} "
                f"{row['status']:>8} {row['episode']:>6} "
                f"{best:>12} {ratio:>10} {str(row['convergence']):>14}"
            )
        if not args.watch:
            break
        time.sleep(max(10, int(args.interval)))


if __name__ == "__main__":
    main()
