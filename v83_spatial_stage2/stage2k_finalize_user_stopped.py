"""Finalize a user-stopped Stage-2k DQN run without retraining.

This script only writes missing ``training_result.json`` and updates
``progress.json``.  It never modifies a checkpoint, training history, log or
TensorBoard event file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import atomic_write_json, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--reason", default="stopped by user to start a new experiment"
    )
    parser.add_argument("--version", default="user-stopped")
    args = parser.parse_args()
    run = args.run_dir
    history_path = run / "training_history.json"
    result_path = run / "training_result.json"
    progress_path = run / "progress.json"
    if not history_path.is_file():
        raise SystemExit(f"Missing {history_path}")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    train = history["train"]
    validation = history.get("validation", [])
    if not train or not validation:
        raise SystemExit("Missing train/validation history")
    final_episode = max(int(row["episode"]) for row in train)
    best_record = max(validation, key=lambda row: row["joint_reward"])
    last_record = validation[-1]
    checkpoint_dir = run / "checkpoints"
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    full_path = checkpoint_dir / "last_full.pt"
    best_payload = torch.load(
        best_path, map_location="cpu", weights_only=False
    )
    last_payload = torch.load(
        last_path, map_location="cpu", weights_only=False
    )
    if int(best_payload["episode"]) != int(best_record["episode"]):
        raise SystemExit("best.pt episode does not match best validation")
    if int(last_payload["episode"]) > final_episode:
        raise SystemExit("last.pt episode exceeds history")
    errors = np.asarray(
        [row["jointr_abs_error"] for row in train], dtype=np.float64
    )
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file() else {}
    )
    config = dict(history.get("config", {}))
    config.update(last_payload.get("config", {}))
    result = {
        "version": "stage2k-jointr-v2-training-result-v1",
        "created_utc": utc_now(),
        "episodes_trained": int(final_episode),
        "max_episodes": int(config.get("max_episodes", 4000)),
        "min_episodes": 800,
        "last_saved_episode": int(last_payload["episode"]),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "last_full_checkpoint": str(full_path),
        "best_validation_joint": float(best_record["joint_reward"]),
        "best_validation_episode": int(best_record["episode"]),
        "last_validation": last_record,
        "stop_reason": str(args.reason),
        "stop_episode": int(final_episode),
        "early_stop_triggered": False,
        "user_stopped": True,
        "status": "user_stopped",
        "converged": False,
        "stable_platform": False,
        "improving": False,
        "consecutive_no_improve": int(
            progress.get("consecutive_no_improve", 0)
        ),
        "reward_mode": "jointr_aligned_v2",
        "gamma": float(config.get("gamma", 1.0)),
        "reward_jointr_abs_error_max": float(errors.max()),
        "reward_jointr_abs_error_mean": float(errors.mean()),
        "final_test_read": False,
        "config": config,
        "history": history,
        "note": (
            "Training was stopped by the user; this script only writes the "
            "missing result and progress fields. Checkpoints, history and "
            "logs are untouched."
        ),
    }
    atomic_write_json(result_path, result)
    progress["updated_utc"] = utc_now()
    progress["status"] = "stopped"
    progress["training_status"] = "user_stopped"
    progress["stop_reason"] = str(args.reason)
    progress["current_episode"] = int(final_episode)
    progress["best_dev_joint"] = float(best_record["joint_reward"])
    progress["best_checkpoint"] = str(best_path)
    progress["last_checkpoint"] = str(last_path)
    progress["last_validation"] = last_record
    progress["converged"] = False
    progress["early_stop_triggered"] = False
    progress["user_stopped"] = True
    progress["training_finished_utc"] = utc_now()
    progress["training_result"] = str(result_path)
    progress["final_test_read"] = False
    progress["final_test"] = "pending"
    atomic_write_json(progress_path, progress)
    print(json.dumps({
        "episodes_trained": final_episode,
        "last_saved_episode": int(last_payload["episode"]),
        "best_validation_episode": int(best_record["episode"]),
        "best_validation_joint": float(best_record["joint_reward"]),
        "last_validation_episode": int(last_record["episode"]),
        "last_validation_joint": float(last_record["joint_reward"]),
        "status": result["status"],
        "training_result": str(result_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
