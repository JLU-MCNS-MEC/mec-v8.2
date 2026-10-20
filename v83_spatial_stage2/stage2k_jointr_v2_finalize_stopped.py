"""Finalize the user-stopped JointR-aligned v2 run without retraining.

The training process was stopped after episode 1320; the last saved model is
episode 1300.  This script only writes the missing training_result.json and
updates progress.json.  It never writes to a checkpoint or training history.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    PROGRESS_PATH,
    atomic_write_json,
    load_progress,
    utc_now,
)

RUN_DIR = Path("runs/v83_spatial_stage2/stage2k_jointr_v2")
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
HISTORY_PATH = RUN_DIR / "training_history.json"
RESULT_PATH = RUN_DIR / "training_result.json"


def main():
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    train = history["train"]
    validation = history["validation"]
    if not train or not validation:
        raise SystemExit("Missing train/validation history")
    final_episode = max(int(row["episode"]) for row in train)
    best_record = max(validation, key=lambda row: row["joint_reward"])
    last_record = validation[-1]
    best_payload = torch.load(
        CHECKPOINT_DIR / "best.pt", map_location="cpu", weights_only=False
    )
    last_payload = torch.load(
        CHECKPOINT_DIR / "last.pt", map_location="cpu", weights_only=False
    )
    if int(best_payload["episode"]) != int(best_record["episode"]):
        raise SystemExit("best.pt episode does not match best validation")
    if int(last_payload["episode"]) > final_episode:
        raise SystemExit("last.pt episode exceeds history")

    errors = np.asarray(
        [row["jointr_abs_error"] for row in train], dtype=np.float64
    )
    result = {
        "version": "stage2k-jointr-v2-training-result-v1",
        "created_utc": utc_now(),
        "episodes_trained": int(final_episode),
        "max_episodes": 4000,
        "min_episodes": 800,
        "last_saved_episode": int(last_payload["episode"]),
        "best_checkpoint": str(CHECKPOINT_DIR / "best.pt"),
        "last_checkpoint": str(CHECKPOINT_DIR / "last.pt"),
        "last_full_checkpoint": str(CHECKPOINT_DIR / "last_full.pt"),
        "best_validation_joint": float(best_record["joint_reward"]),
        "best_validation_episode": int(best_record["episode"]),
        "last_validation": last_record,
        "stop_reason": (
            "stopped by user after 6 validations; best JointR at Ep400 and "
            "no improvement since"
        ),
        "stop_episode": int(final_episode),
        "early_stop_triggered": False,
        "user_stopped": True,
        "status": "user_stopped_no_improvement",
        "converged": False,
        "stable_platform": False,
        "improving": False,
        "consecutive_no_improve": int(
            load_progress()["dqn"].get("consecutive_no_improve", 4)
        ),
        "reward_mode": "jointr_aligned_v2",
        "gamma": 1.0,
        "reward_jointr_abs_error_max": float(errors.max()),
        "reward_jointr_abs_error_mean": float(errors.mean()),
        "final_test_read": False,
        "config": history["config"],
        "history": history,
        "note": (
            "Training was interrupted by the user; last.pt/last_full.pt are "
            "from episode 1300, while history has rows through episode 1320."
        ),
    }
    atomic_write_json(RESULT_PATH, result)

    progress = load_progress()
    progress["updated_utc"] = utc_now()
    progress["status"] = "stopped"
    progress["dqn"].update({
        "status": "stopped",
        "current_episode": int(final_episode),
        "best_dev_joint": float(best_record["joint_reward"]),
        "consecutive_no_improve": int(
            progress["dqn"].get("consecutive_no_improve", 4)
        ),
        "best_checkpoint": str(CHECKPOINT_DIR / "best.pt"),
        "last_checkpoint": str(CHECKPOINT_DIR / "last.pt"),
        "last_validation": last_record,
        "early_stop_triggered": False,
        "converged": False,
        "training_status": "user_stopped_no_improvement",
        "stop_reason": result["stop_reason"],
        "training_finished_utc": utc_now(),
        "user_stopped": True,
        "last_saved_episode": int(last_payload["episode"]),
        "next_step": (
            "Evaluate last.pt on Dev Val 180 as requested; no further "
            "training and no Final Test."
        ),
    })
    atomic_write_json(PROGRESS_PATH, progress)
    print(json.dumps({
        "episodes_trained": final_episode,
        "last_saved_episode": int(last_payload["episode"]),
        "best_validation_episode": int(best_record["episode"]),
        "best_validation_joint": float(best_record["joint_reward"]),
        "last_validation_episode": int(last_record["episode"]),
        "last_validation_joint": float(last_record["joint_reward"]),
        "reward_abs_error_max": float(errors.max()),
        "status": result["status"],
        "training_result": str(RESULT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
