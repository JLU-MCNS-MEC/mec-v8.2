"""Read-only Stage-2r status table.

    python v83_spatial_stage2/stage2r_10node_small_vs_tm/status.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2r_10node_small_vs_tm.common import (  # noqa: E402
    ALGORITHMS,
    ALGORITHM_LABELS,
    FORMAL_SEEDS,
    MEC_MODELS,
    MODEL_LABELS,
    run_dir,
)


def _gpu():
    try:
        output = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        util, used, total = [
            part.strip() for part in output.strip().splitlines()[0].split(",")
        ]
        return f"GPU util={util}% 显存 {used}/{total} MiB"
    except Exception:
        return "GPU 状态不可用"


def _read(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    print(_gpu())
    header = (
        f"{'部署':<18} {'算法':<5} {'seed':>4} {'进度':>12} {'状态':<12} "
        f"{'成功':>5} {'失败':>5} {'成功S/M/L':>13} {'最佳DevE2E':>10} "
        f"{'@Ep':>5}"
    )
    print(header)
    print("-" * len(header))
    for model in MEC_MODELS:
        for algorithm in ALGORITHMS:
            for seed in FORMAL_SEEDS:
                path = run_dir(model, algorithm, seed)
                progress = _read(path / "progress.json") or {}
                counts = progress.get("training_counts") or {}
                best = progress.get("best_dev_e2e_ms")
                print(
                    f"{MODEL_LABELS[model]:<18} "
                    f"{ALGORITHM_LABELS[algorithm]:<5} {seed:>4} "
                    f"{progress.get('current_episode', 0):>5}/"
                    f"{progress.get('max_episodes', 4000):<6} "
                    f"{progress.get('status', 'not_started'):<12} "
                    f"{counts.get('cum_success', 0):>5} "
                    f"{counts.get('cum_failure', 0):>5} "
                    f"{counts.get('cum_success_short', 0):>4}/"
                    f"{counts.get('cum_success_medium', 0):>3}/"
                    f"{counts.get('cum_success_long', 0):>4} "
                    f"{('—' if best is None else f'{best:.2f}'):>10} "
                    f"{progress.get('best_dev_episode', '—'):>5}"
                )


if __name__ == "__main__":
    main()
