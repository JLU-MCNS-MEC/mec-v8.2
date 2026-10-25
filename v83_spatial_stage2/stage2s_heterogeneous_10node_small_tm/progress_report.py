"""Compact Chinese progress table from the latest per-run round files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (  # noqa: E402
    ALGORITHMS,
    FORMAL_SEEDS,
    run_dir,
)


def _latest_round(algorithm, seed):
    rounds = sorted(
        (run_dir(algorithm, seed) / "rounds").glob("round_*.json")
    )
    if not rounds:
        return None
    try:
        return json.loads(rounds[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(value, digits=2):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _pct(value, digits=2):
    if value is None:
        return "N/A"
    return f"{100*float(value):.{digits}f}%"


def show():
    print(
        "\nStage-2s 每 200 Episodes 进展（最新一轮）\n"
        f"{'算法':>4} {'seed':>4} {'Ep':>5} {'MA100奖励':>10} "
        f"{'MA100 E2E':>10} {'累计成功':>8} {'累计失败':>8} "
        f"{'Dev E2E':>9} {'Dev SR':>7} {'Dev nDTW':>9} "
        f"{'Dev OSR':>8} {'Small':>7} {'最佳DevE2E':>10}"
    )
    for algorithm in ALGORITHMS:
        for seed in FORMAL_SEEDS:
            payload = _latest_round(algorithm, seed)
            if payload is None:
                print(
                    f"{algorithm.upper():>4} {seed:>4} {'-':>5} "
                    f"{'等待首次 200':>10}"
                )
                continue
            metrics = payload["metrics"]
            print(
                f"{algorithm.upper():>4} {seed:>4} "
                f"{int(metrics['episode']):>5} "
                f"{_fmt(metrics.get('reward_ma100'), 4):>10} "
                f"{_fmt(metrics.get('e2e_ma100'), 2):>10} "
                f"{int(metrics.get('cum_success', 0)):>8} "
                f"{int(metrics.get('cum_failure', 0)):>8} "
                f"{_fmt(metrics.get('dev_e2e_ms'), 2):>9} "
                f"{_pct(metrics.get('dev_sr')):>7} "
                f"{_fmt(metrics.get('dev_ndtw'), 4):>9} "
                f"{_pct(metrics.get('dev_osr')):>8} "
                f"{_pct(metrics.get('dev_small_ratio')):>7} "
                f"{_fmt(metrics.get('dev_best_e2e_ms'), 2):>10}"
            )
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args(argv)
    while True:
        show()
        if not args.watch:
            break
        time.sleep(max(30, int(args.interval)))


if __name__ == "__main__":
    main()
