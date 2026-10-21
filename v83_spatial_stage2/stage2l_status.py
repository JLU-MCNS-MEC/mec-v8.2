"""Print the latest Stage-2l training/validation status in Chinese."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import read_json
from v83_spatial_stage2.stage2l_protocol import status_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=Path("runs/v83_spatial_stage2/stage2l_fixed_city"),
    )
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = {}
    for config in args.configs:
        paused = read_json(args.run_root / config / "PAUSED.json")
        for algorithm in ("dqn", "ppo"):
            run_dir = args.run_root / config / algorithm
            if not (run_dir / "progress.json").is_file():
                payload[f"{config}/{algorithm}"] = {
                    "status": "未启动",
                    "run_dir": str(run_dir),
                }
                continue
            entry = status_summary(run_dir)
            if paused and paused.get("paused"):
                entry["status"] = "已暂停(SIGSTOP)"
                entry["paused"] = paused
            payload[f"{config}/{algorithm}"] = entry

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    print("# Stage-2l 训练状态")
    print()
    def number(value, digits=4, scale=1.0, suffix=""):
        if value is None:
            return "-"
        return f"{scale*float(value):.{digits}f}{suffix}"

    print("| 运行 | 状态 | Episode | 最新验证 JointR | SR | nDTW | E2E (ms) | Best JointR |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for key, item in payload.items():
        last = item.get("last_validation") or {}
        status = item.get("status", "未启动")
        episode = item.get("current_episode")
        best = item.get("best_dev_joint")
        print(
            f"| {key} | {status} | {episode if episode is not None else '-'} | "
            f"{number(last.get('joint_reward'))} | "
            f"{number(last.get('SR'), 2, 100.0, '%')} | "
            f"{number(last.get('nDTW'))} | "
            f"{number(last.get('E2E_ms'), 1)} | "
            f"{number(best)} |"
        )
    print()
    print(f"更新时间：{payload and max((item.get('updated_utc') or '') for item in payload.values())}")


if __name__ == "__main__":
    main()
