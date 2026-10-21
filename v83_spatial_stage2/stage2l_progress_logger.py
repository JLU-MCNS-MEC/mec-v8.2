"""Append every new Stage-2l fixed-Dev validation to PROGRESS.md.

The original supervisor launched A/B and wrote PROGRESS.md.  After B was
paused and C was started manually, this small logger replaces the
supervisor's progress-writing role without launching or stopping any
training process.

It never changes a checkpoint, reward, algorithm or training command.  It
only reads each run's ``progress.json`` and appends a row when a new
180/200-episode validation is available.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import utc_now
from v83_spatial_stage2.stage2l_fixed_city import STAGE2L_ROOT


CONFIGS = ("A", "B", "C")
ALGOS = ("dqn", "ppo")
HEADER = (
    "# Stage-2l 训练进度\n\n"
    "每 200 Episodes 记录一次固定 Dev 验证结果。\n\n"
    "| 时间 | 配置 | 算法 | Episode | JointR | SR | nDTW | E2E (ms) | SPL | Best |\n"
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
)


def _parse_existing(path: Path) -> dict:
    last = {(config, algorithm): -1 for config in CONFIGS for algorithm in ALGOS}
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HEADER, encoding="utf-8")
        return last
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| 20"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        config, algorithm, episode = cells[1], cells[2].lower(), cells[3]
        try:
            last[(config, algorithm)] = max(
                last.get((config, algorithm), -1), int(float(episode))
            )
        except ValueError:
            continue
    return last


def _row(config, algorithm, record, best):
    return (
        f"| {utc_now()} | {config} | {algorithm.upper()} | "
        f"{record.get('episode')} | {record.get('joint_reward', 0):.4f} | "
        f"{100*record.get('SR', 0):.2f}% | {record.get('nDTW', 0):.4f} | "
        f"{record.get('E2E_ms', 0):.1f} | {record.get('SPL', 0):.4f} | "
        f"{best if best is not None else '-'} |\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=STAGE2L_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    progress = args.run_root / "PROGRESS.md"
    last = _parse_existing(progress)
    print(f"[progress-logger] initialized from {progress}", flush=True)
    while True:
        for config in CONFIGS:
            for algorithm in ALGOS:
                path = args.run_root / config / algorithm / "progress.json"
                if not path.is_file():
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                record = data.get("last_validation")
                if not record:
                    continue
                episode = int(record.get("episode", -1))
                key = (config, algorithm)
                if episode <= last.get(key, -1):
                    continue
                last[key] = episode
                best = data.get("best_dev_joint")
                line = _row(config, algorithm, record, best)
                with progress.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                print(
                    f"{config}/{algorithm} Ep{episode} "
                    f"JointR={record.get('joint_reward', 0):.4f} "
                    f"SR={100*record.get('SR', 0):.2f}% "
                    f"nDTW={record.get('nDTW', 0):.4f} "
                    f"E2E={record.get('E2E_ms', 0):.1f}",
                    flush=True,
                )
        time.sleep(max(5, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
