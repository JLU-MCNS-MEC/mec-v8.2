"""Compare Stage-2p seed-0 runs with their Stage-2n/2o counterparts.

    python v83_spatial_stage2/stage2p/reproduction_check.py

If the unified Stage-2p loop is equivalent to the old trainer for the same
reward, the per-episode metrics and Dev validations must match numerically.
The check is a reproduction audit, not a replacement for the paired 3-seed
comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    read_json,
    utc_now,
)
from v83_spatial_stage2.stage2p.train import RUNS_ROOT  # noqa: E402

STAGE2N = PROJECT_ROOT / "runs/v83_spatial_stage2/stage2n_latency_only/runs"
STAGE2O = PROJECT_ROOT / "runs/v83_spatial_stage2/stage2o"
EVIDENCE = Path(__file__).resolve().parent / "evidence"
TRAIN_KEYS = (
    "episode_return", "mean_episode_e2e_ms", "steps", "success",
    "final_nDTW",
)
TOLERANCE = 1e-9


def _load(path):
    payload = read_json(path)
    return payload or {}


def _compare(label, new_dir, old_dir):
    new = _load(new_dir / "training_history.json")
    old = _load(old_dir / "training_history.json")
    new_train = new.get("train", [])
    old_train = old.get("train", [])
    count = min(len(new_train), len(old_train))
    max_train = 0.0
    worst_key = None
    for index in range(count):
        for key in TRAIN_KEYS:
            difference = abs(
                float(new_train[index][key]) - float(old_train[index][key])
            )
            if difference > max_train:
                max_train = difference
                worst_key = f"train[{index}].{key}"
    new_val = new.get("validation", [])
    old_val = old.get("validation", [])
    val_count = min(len(new_val), len(old_val))
    max_val = 0.0
    for index in range(val_count):
        difference = abs(
            float(new_val[index]["E2E_ms"])
            - float(old_val[index]["E2E_ms"])
        )
        max_val = max(max_val, difference)
    return {
        "label": label,
        "new_run": str(new_dir),
        "old_run": str(old_dir),
        "episodes_compared": count,
        "validations_compared": val_count,
        "max_train_abs_diff": float(max_train),
        "worst_train_field": worst_key,
        "max_validation_e2e_abs_diff": float(max_val),
        "match": bool(
            count > 0 and val_count > 0
            and max_train <= TOLERANCE and max_val <= TOLERANCE
        ),
        "tolerance": TOLERANCE,
    }


def main():
    checks = [
        _compare(
            "baseline seed0 vs Stage-2n PPO seed0",
            RUNS_ROOT / "baseline" / "ppo_s0",
            STAGE2N / "ppo_s0",
        ),
        _compare(
            "D seed0 vs Stage-2o D/PPO seed0",
            RUNS_ROOT / "D" / "ppo_s0",
            STAGE2O / "D" / "ppo_s0",
        ),
    ]
    payload = {
        "stage": "stage2p",
        "checked_utc": utc_now(),
        "checks": checks,
        "all_match": all(check["match"] for check in checks),
        "note": (
            "A match confirms that the unified Stage-2p loop reproduces the "
            "old seed-0 result for the same reward.  The Stage-2p paired "
            "comparison still uses all three seeds and both arms."
        ),
    }
    atomic_write_json(EVIDENCE / "reproduction_check.json", payload)
    for check in checks:
        print(
            f"[repro] {check['label']}: match={check['match']} "
            f"train_max={check['max_train_abs_diff']:.3e} "
            f"val_max={check['max_validation_e2e_abs_diff']:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
