"""Print Stage-2n run status without touching any process.

    python v83_spatial_stage2/stage2n_status.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import read_json
from v83_spatial_stage2.stage2n_common import AGENT_SEEDS, RUNS_DIR


def gpu():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.DEVNULL).strip()
        return out
    except Exception:
        return "n/a"


def main():
    print(f"GPU: {gpu()}")
    launch = read_json(
        Path("runs/v83_spatial_stage2/stage2n_latency_only/logs/launch.json")
    )
    if launch:
        print(
            f"launch: started={launch.get('created_utc')} "
            f"jobs={len(launch.get('jobs', []))}"
        )
    for algorithm in ("dqn", "ppo"):
        for seed in AGENT_SEEDS:
            run_dir = RUNS_DIR / f"{algorithm}_s{seed}"
            progress = read_json(run_dir / "progress.json") or {}
            result = read_json(run_dir / "training_result.json") or {}
            print(
                f"{algorithm.upper():3s} s{seed}: "
                f"Ep{progress.get('current_episode', '-')}"
                f"/{progress.get('max_episodes', '-')} "
                f"status={progress.get('status', '-')} "
                f"best_dev_e2e={progress.get('best_dev_e2e_ms', '-')} "
                f"last_dev_e2e={(progress.get('last_validation') or {}).get('E2E_ms', '-')}"
            )
            if result:
                conv = result.get("convergence", {})
                print(
                    f"          finished: best={result.get('best_dev_e2e_ms')} "
                    f"@Ep{result.get('best_dev_episode')} "
                    f"converged={conv.get('converged')} "
                    f"status={conv.get('status')}"
                )


if __name__ == "__main__":
    main()
