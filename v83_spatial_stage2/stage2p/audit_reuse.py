"""Audit whether Stage-2n PPO / Stage-2o D seed-0 can be reused in Stage-2p.

    python v83_spatial_stage2/stage2p/audit_reuse.py

The audit compares task pools, deployment, MEC checkpoint, state/action
contract and PPO hyperparameters.  It deliberately does *not* treat a
different trainer file as equivalent: Stage-2n and Stage-2o used different
training loops, so Stage-2p reruns all six runs under one Stage-2p loop and
uses the old seed-0 runs only as reproduction checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
)
from v83_spatial_stage2.stage2p.train import (  # noqa: E402
    STAGE2N_PROTOCOL_PATH,
    STAGE2O_FREEZE,
)

EVIDENCE = Path(__file__).resolve().parent / "evidence"
STAGE2N_ROOT = PROJECT_ROOT / "runs/v83_spatial_stage2/stage2n_latency_only/runs"
STAGE2O_ROOT = PROJECT_ROOT / "runs/v83_spatial_stage2/stage2o"
PPO_KEYS = (
    "gamma", "gae_lambda", "clip_eps", "learning_rate", "rollout_steps",
    "minibatch_size", "update_epochs", "value_coef", "entropy_coef",
    "max_grad_norm", "target_kl", "state_dim", "state_mode", "action_dim",
)


def _checkpoint_config(path):
    if not Path(path).is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("config", {}))


def main():
    stage2n_protocol = read_json(STAGE2N_PROTOCOL_PATH) or {}
    stage2o_freeze = read_json(STAGE2O_FREEZE) or {}
    rows = []
    for seed in (0, 1, 2):
        protocol = read_json(
            STAGE2N_ROOT / f"ppo_s{seed}" / "protocol.json"
        ) or {}
        config = _checkpoint_config(
            STAGE2N_ROOT / f"ppo_s{seed}" / "checkpoints" / "best.pt"
        )
        rows.append({
            "stage": "stage2n",
            "scheme": "baseline",
            "seed": seed,
            "protocol_reward": protocol.get("reward", {}).get("per_step"),
            "task_hashes": protocol.get("artifact_hashes", {}),
            "config": {key: config.get(key) for key in PPO_KEYS},
            "trainer": "stage2n_train.py",
        })
    protocol = read_json(
        STAGE2O_ROOT / "D" / "ppo_s0" / "protocol.json"
    ) or {}
    config = _checkpoint_config(
        STAGE2O_ROOT / "D" / "ppo_s0" / "checkpoints" / "best.pt"
    )
    rows.append({
        "stage": "stage2o",
        "scheme": "D",
        "seed": 0,
        "protocol_reward": protocol.get("reward", {}).get("formula"),
        "task_hashes": protocol.get("artifact_hashes", {}),
        "config": {key: config.get(key) for key in PPO_KEYS},
        "trainer": "stage2o/train.py",
    })

    baseline_hashes = rows[0]["task_hashes"]
    hashes_match = all(
        row["task_hashes"].get(key) == baseline_hashes.get(key)
        for row in rows
        for key in ("train_tasks", "dev_tasks", "training_order",
                    "deployment", "mec_checkpoint")
    )
    configs_match = all(
        all(row["config"].get(key) == rows[0]["config"].get(key)
            for key in PPO_KEYS)
        for row in rows
    )
    trainer_paths = sorted({row["trainer"] for row in rows})
    payload = {
        "stage": "stage2p",
        "created_utc": utc_now(),
        "parent_protocols": {
            "stage2n_path": str(STAGE2N_PROTOCOL_PATH),
            "stage2n_sha256": (
                sha256_file(STAGE2N_PROTOCOL_PATH)
                if Path(STAGE2N_PROTOCOL_PATH).is_file() else None
            ),
            "stage2o_freeze_path": str(STAGE2O_FREEZE),
            "stage2o_freeze_sha256": (
                sha256_file(STAGE2O_FREEZE)
                if Path(STAGE2O_FREEZE).is_file() else None
            ),
        },
        "runs_audited": rows,
        "task_hashes_match": bool(hashes_match),
        "ppo_configs_match": bool(configs_match),
        "trainer_paths": trainer_paths,
        "differences_beyond_reward": [
            "Stage-2n PPO and Stage-2o D/PPO used different trainer files "
            "and different reward-application code paths.",
            "Logging/provenance labels differ (Stage-2n vs Stage-2o).",
        ],
        "decision": (
            "Do not mix Stage-2n/Stage-2o results into the Stage-2p paired "
            "comparison.  Rerun baseline seed 0/1/2 and D seed 0/1/2 with the "
            "single Stage-2p PPO loop; use the old seed-0 runs only as "
            "reproduction checks after training."
        ),
        "final_test_read": False,
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    atomic_write_json(EVIDENCE / "reuse_audit.json", payload)
    print(
        f"[audit] task_hashes_match={hashes_match} "
        f"ppo_configs_match={configs_match} "
        f"trainers={trainer_paths}", flush=True,
    )
    print(f"[audit] wrote {EVIDENCE / 'reuse_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
