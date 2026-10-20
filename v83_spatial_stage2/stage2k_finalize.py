"""Reconstruct the Stage-2k DQN finalization after an interrupted run.

The training loop completed episode 2000 and wrote ``last.pt``,
``training_history.json`` and ``progress.json``.  The process then ended
before ``training_result.json`` and the plots were written; the recorded
history is consistent with a crash in ``plot_training_curves`` (the old code
read ``row["episode_return"]`` while training rows store ``row["return"]``).

This script does not train, does not modify either checkpoint and does not
touch the TensorBoard event file.  It replays the pre-declared early-stop
decision from the recorded validation history and writes only the missing
finalization artifacts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    DQN_DIR,
    PROGRESS_PATH,
    QUALITY_PRIOR_PATH,
    VISUAL_PCA_PATH,
    atomic_write_json,
    load_progress,
    sha256_file,
    update_progress,
    utc_now,
)
from v83_spatial_stage2.stage2k_dqn import (
    MAX_EPISODES,
    MIN_EPISODES,
    SEED,
    plot_training_curves,
    should_early_stop,
)


def _parse_time(value):
    return datetime.fromisoformat(str(value))


def main():
    result_path = DQN_DIR / "training_result.json"
    history_path = DQN_DIR / "training_history.json"
    last_path = DQN_DIR / "last.pt"
    best_path = DQN_DIR / "best.pt"
    order_path = DQN_DIR / "training_order.json"
    design_path = Path("v83_spatial_stage2/STAGE2K_STATE_REWARD_DESIGN.md")

    if result_path.is_file():
        print("training_result.json already exists; refusing to overwrite.")
        return
    for required in (history_path, last_path, best_path):
        if not required.is_file():
            raise SystemExit(f"Missing required artifact: {required}")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    progress = load_progress()
    train = history["train"]
    validation = history["validation"]
    if not train or not validation:
        raise SystemExit("Training history is empty")

    final_episode = max(int(row["episode"]) for row in train)
    last_validation = validation[-1]
    if int(last_validation["episode"]) != final_episode:
        raise SystemExit(
            "Last validation episode does not match last training episode"
        )
    expected = list(range(200, final_episode + 1, 200))
    if [int(row["episode"]) for row in validation] != expected:
        raise SystemExit("Validation episodes are not the expected 200-step grid")

    joints = [float(row["joint_reward"]) for row in validation]
    best_index = max(range(len(joints)), key=joints.__getitem__)
    best_record = validation[best_index]
    best_joint = float(best_record["joint_reward"])

    no_improve = int(progress["dqn"]["consecutive_no_improve"])
    recorded_best = progress["dqn"]["best_validation_joint"]
    if recorded_best is None or abs(float(recorded_best) - best_joint) > 1e-12:
        raise SystemExit("progress.json best JointR disagrees with history")
    if int(progress["dqn"]["current_episode"]) != final_episode:
        raise SystemExit("progress.json episode disagrees with history")

    stop, reason = should_early_stop(final_episode, no_improve, joints)
    if not stop:
        raise SystemExit(
            "Recorded run does not satisfy the pre-declared early-stop rule; "
            "use stage2k_dqn.py --resume instead of finalizing."
        )
    converged = bool(
        reason.startswith("no ") or reason.startswith("no new")
    )

    # Checkpoint metadata must agree with the recorded history before we
    # declare the run complete.  Checkpoints themselves are never rewritten.
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    last_payload = torch.load(last_path, map_location="cpu", weights_only=False)
    if int(best_payload["episode"]) != int(best_record["episode"]):
        raise SystemExit("best.pt episode does not match best validation")
    if int(last_payload["episode"]) != final_episode:
        raise SystemExit("last.pt episode does not match final episode")
    if abs(float(best_payload["best_joint"]) - best_joint) > 1e-12:
        raise SystemExit("best.pt best_joint does not match history")

    # Reconstruct wall-clock duration from the frozen training order and the
    # last validation timestamp.  Mark it as an estimate in the result.
    order_created = _parse_time(
        json.loads(order_path.read_text(encoding="utf-8"))["created_utc"]
    )
    training_finished = _parse_time(last_validation["timestamp"])
    elapsed_s = max(0.0, (training_finished - order_created).total_seconds())

    training_result = {
        "version": "stage2k-dqn-v1",
        "created_utc": utc_now(),
        "episodes_trained": int(final_episode),
        "max_episodes": int(MAX_EPISODES),
        "min_episodes": int(MIN_EPISODES),
        "best_validation_joint": best_joint,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "stop_reason": reason,
        "stop_episode": int(final_episode),
        "converged": converged,
        "consecutive_no_improve": no_improve,
        "total_steps": int(last_payload["total_steps"]),
        "elapsed_s": elapsed_s,
        "elapsed_s_estimated": True,
        "architecture": "mlp_478_256_256_20",
        "seed": SEED,
        "quality_prior_sha256": sha256_file(QUALITY_PRIOR_PATH),
        "visual_pca_sha256": sha256_file(VISUAL_PCA_PATH),
        "state_design_sha256": (
            sha256_file(design_path) if design_path.is_file() else None
        ),
        "final_test_read": False,
        "training_finished_utc": last_validation["timestamp"],
        "finalized_utc": utc_now(),
        "best_validation_record": best_record,
        "last_validation_record": last_validation,
        "reconstructed_finalization": True,
        "reconstruction_note": (
            "The training loop reached episode 2000 and wrote last.pt, "
            "training_history.json and progress.json.  The process ended "
            "before training_result.json and plots were written; the old "
            "plot_training_curves read row['episode_return'] instead of "
            "row['return'], which raises KeyError at this point.  No training "
            "was repeated and no checkpoint was modified."
        ),
        "converged_interpretation": (
            "Pre-declared early-stop rule fired at episode 2000.  The best "
            "validation JointR was at episode 800; the last three validations "
            "declined.  This is a protocol convergence flag, not evidence of "
            "a stable competitive policy."
        ),
        "history": history,
    }
    atomic_write_json(result_path, training_result)

    plot_training_curves(history, DQN_DIR / "plots")

    progress = update_progress(
        "dqn",
        status="completed",
        current_episode=int(final_episode),
        best_validation_joint=best_joint,
        consecutive_no_improve=no_improve,
        best_checkpoint=str(best_path),
        last_checkpoint=str(last_path),
        training_finished_utc=last_validation["timestamp"],
        converged=converged,
        stop_reason=reason,
        stop_episode=int(final_episode),
        last_validation=last_validation,
        next_step=(
            "Await explicit decision on the one-shot Association Final Test "
            "180.  Best validation JointR is below Heuristic, Min-Latency and "
            "Max-Quality, so the DQN is not a competitive candidate yet."
        ),
    )
    progress["recovery"] = {
        "recovered_utc": utc_now(),
        "source_checkpoint": str(last_path),
        "source_episode": int(final_episode),
        "resume_mode": "finalization_only_no_retraining",
        "checkpoint_contents": {
            "online_network": True,
            "target_network": True,
            "optimizer": True,
            "replay_buffer": False,
            "epsilon_state": False,
            "rng_state": False,
        },
        "tensorboard": "runs/tensorboard/association_stage2k "
                       "(original event file kept)",
        "credits": (
            "Baseline Dev Val 180 and quality priors reused; no MEC was "
            "retrained; no baseline was rerun."
        ),
    }
    atomic_write_json(PROGRESS_PATH, progress)

    print(json.dumps({
        "episodes_trained": final_episode,
        "best_validation_joint": best_joint,
        "best_episode": int(best_record["episode"]),
        "last_validation_joint": float(last_validation["joint_reward"]),
        "stop_reason": reason,
        "converged": converged,
        "training_result": str(result_path),
        "plots": str(DQN_DIR / "plots"),
    }, indent=2))


if __name__ == "__main__":
    main()
