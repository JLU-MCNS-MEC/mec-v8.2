"""Shared protocol, validation and provenance helpers for Stage-2l.

Both the DQN and PPO trainers import this module so that the following
contract cannot drift between algorithms:

* the same 3500-task order (SHA-256 pinned);
* the same 60 independent Dev tasks;
* the same config-specific topology assignment;
* the same Pure-MEC v2 state, frozen MECs, reward and evaluation code;
* no Final Test read during training.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    atomic_write_json,
    read_json,
    summarize_episodes,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_state import masked_argmax
from v83_spatial_stage2.stage2k_v2_common import load_capability_prior
from v83_spatial_stage2.stage2l_fixed_city import (
    CONFIG_DESCRIPTIONS,
    MAP_SEED_PHASE1,
    build_city,
    TRAIN_EPISODES,
    canonical_sha256,
    config_specs,
    load_config_manifest,
    load_topology,
    sha256_file,
    topology_role_for,
)
from v83_spatial_stage2.stage2l_runtime import (
    FixedCityEpisodeRunner,
    FixedCitySceneCache,
)


REWARD_DESCRIPTION = (
    "r_nav_t + (-0.25/100)*(running_mean_E2E_t - "
    "running_mean_E2E_{t-1})"
)
VALIDATION_EVERY = 200
VALIDATION_IMPROVEMENT = 0.002
STATE_MODE = "pure_mec_v2"
STATE_DIM = 384
ACTION_DIM = 20

CAPABILITY_PRIOR_PATH = Path(
    "runs/v83_spatial_stage2/stage2k/capability_prior_v2.json"
)


def resolve_device(value: str) -> torch.device:
    if value is None or value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_threads(default: int = 4) -> int:
    value = int(os.environ.get("STAGE2L_TORCH_THREADS", str(default)))
    torch.set_num_threads(max(1, value))
    return max(1, value)


def load_bundle(config: str) -> dict:
    """Load the frozen task pools, order, config manifest and topologies."""
    manifest = load_config_manifest(config)
    train_specs, _, order = config_specs(config, "train")
    dev_specs, _, _ = config_specs(config, "dev")
    roles = sorted(
        set(manifest["topology_roles"]["train"])
        | set(manifest["topology_roles"]["dev"])
    )
    topologies = {role: load_topology(role) for role in roles}
    return {
        "config": config,
        "manifest": manifest,
        "train_specs": train_specs,
        "dev_specs": dev_specs,
        "order": order,
        "topologies": topologies,
    }


def topology_for(bundle: Mapping, split: str, spec) -> tuple[dict, str]:
    role = topology_role_for(bundle["config"], split, spec)
    if role not in bundle["topologies"]:
        raise KeyError(
            f"Topology role {role} is missing from config "
            f"{bundle['config']}"
        )
    return bundle["topologies"][role], role


def _git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT),
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def protocol_payload(
    bundle: Mapping,
    algorithm: str,
    device: torch.device,
    threads: int,
    argv,
    run_dir: Path,
    tensorboard_dir: Path,
) -> dict:
    manifest = bundle["manifest"]
    order = bundle["order"]
    dev_specs = bundle["dev_specs"]
    prior_payload, prior_flat = load_capability_prior(CAPABILITY_PRIOR_PATH)
    checks = {
        "config": bundle["config"],
        "config_description": CONFIG_DESCRIPTIONS[bundle["config"]],
        "state_mode": STATE_MODE,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "reward": REWARD_DESCRIPTION,
        "max_episodes": TRAIN_EPISODES,
        "validation_every": VALIDATION_EVERY,
        "validation_tasks": len(dev_specs),
        "train_tasks": len(bundle["train_specs"]),
        "training_order_length": len(order["order"]),
        "training_order_sha256": canonical_sha256(order["order"]),
        "manifest_order_sha256": manifest["training_order_sha256"],
        "order_matches_manifest": (
            canonical_sha256(order["order"])
            == manifest["training_order_sha256"]
        ),
        "task_pool_sha256": manifest["task_pool_sha256"],
        "dev_pool_sha256": manifest["dev_pool_sha256"],
        "map_seed": int(MAP_SEED_PHASE1),
        "city_fingerprint": build_city(int(MAP_SEED_PHASE1)).fingerprint,
        "dev_task_uids": [item["episode_uid"] for item in dev_specs],
        "dev_task_seeds": [int(item["task_seed"]) for item in dev_specs],
        "train_topology_roles": manifest["topology_roles"]["train"],
        "dev_topology_roles": manifest["topology_roles"]["dev"],
        "topology_seeds": {
            role: int(item["topology_seed"])
            for role, item in bundle["topologies"].items()
        },
        "topology_scenarios": {
            role: item["scenario"]
            for role, item in bundle["topologies"].items()
        },
        "capability_prior": {
            "path": str(CAPABILITY_PRIOR_PATH),
            "sha256": sha256_file(CAPABILITY_PRIOR_PATH),
            "split": prior_payload["split"],
            "episodes": int(prior_payload["episodes"]),
            "final_test_read": bool(prior_payload["final_test_read"]),
            "models": {
                name: {
                    metric: float(prior_flat[name][metric])
                    for metric in ("sr", "osr", "spl", "ndtw")
                }
                for name in prior_flat
            },
        },
        "frozen_mecs": list(prior_flat),
        "final_test_read": False,
        "test_pool_read": False,
        "algorithm": algorithm,
        "device": str(device),
        "torch_threads": int(threads),
        "run_dir": str(run_dir),
        "tensorboard_dir": str(tensorboard_dir),
        "argv": [str(value) for value in argv],
        "launch_environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "git": _git_commit(),
            "pid": os.getpid(),
            "created_utc": utc_now(),
        },
    }
    if not checks["order_matches_manifest"]:
        raise SystemExit(
            "Training order does not match the frozen config manifest"
        )
    if checks["capability_prior"]["final_test_read"]:
        raise SystemExit("Capability prior claims a Final Test read")
    return checks


def write_protocol_files(
    run_dir: Path, protocol: Mapping, commands
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "protocol.json", dict(protocol))
    atomic_write_json(run_dir / "commands.json", {
        "argv": list(commands),
        "created_utc": utc_now(),
    })


class DevEvaluator:
    """Fixed Dev evaluation shared by DQN and PPO."""

    def __init__(
        self,
        bundle: Mapping,
        encoder,
        backends,
        quality_prior,
        gab_config,
        device,
        scene_cache=None,
        split="dev",
    ):
        self.bundle = bundle
        self.split = str(split)
        self.specs = bundle.get("eval_specs", bundle["dev_specs"])
        self.encoder = encoder
        self.backends = backends
        self.quality_prior = quality_prior
        self.gab_config = gab_config
        self.device = device
        self.scene_cache = scene_cache or FixedCitySceneCache(
            encoder, device, max_entries=max(64, len(self.specs))
        )

    def _runner(self, spec):
        topology, role = topology_for(self.bundle, self.split, spec)
        return FixedCityEpisodeRunner(
            spec=spec,
            topology=topology,
            encoder=self.encoder,
            backends=self.backends,
            projector=None,
            quality_prior=self.quality_prior,
            gab_config=self.gab_config,
            scene_cache=self.scene_cache,
            device=self.device,
            jointr_aligned=True,
            pure_mec_v2_state=True,
        ), role

    def evaluate_dqn(self, agent):
        rows = []
        for spec in self.specs:
            runner, _role = self._runner(spec)
            while True:
                state = runner.step_data["state"]
                mask = runner.step_data["mask"]
                action = int(masked_argmax(agent.q_values(state), mask))
                transition = runner.apply(action)
                if transition["done"]:
                    break
            rows.append(runner.finalize("dqn_greedy"))
        return rows

    def evaluate_ppo(self, agent):
        rows = []
        for spec in self.specs:
            runner, _role = self._runner(spec)
            while True:
                action = agent.greedy_action(
                    runner.step_data["state"], runner.step_data["mask"]
                )
                transition = runner.apply(action)
                if transition["done"]:
                    break
            rows.append(runner.finalize("ppo_greedy"))
        return rows

    def evaluate_strategy(self, strategy, seed_base=1_000_003):
        rows = []
        for index, spec in enumerate(self.specs):
            runner, _role = self._runner(spec)
            rng = np.random.default_rng(int(seed_base) + index)
            while True:
                node = runner.select_node(strategy, rng=rng)
                transition = runner.apply(node)
                if transition["done"]:
                    break
            rows.append(runner.finalize(strategy))
        return rows


def validation_record(
    episode: int,
    rows,
    train_window,
    extra=None,
) -> dict:
    summary = summarize_episodes(rows)
    errors = [float(item["jointr_abs_error"]) for item in rows]
    if max(errors) > 1e-6:
        raise RuntimeError(
            f"Validation reward/JointR mismatch: {max(errors)}"
        )
    record = {
        "episode": int(episode),
        "joint_reward": float(summary["JointR"]),
        "SR": float(summary["SR"]),
        "OSR": float(summary["OSR"]),
        "SPL": float(summary["SPL"]),
        "nDTW": float(summary["nDTW"]),
        "E2E_ms": float(summary["E2E_ms"]),
        "P95_ms": float(summary["P95_ms"]),
        "delay_violation_rate": float(summary["delay_violation_rate"]),
        "mean_steps": float(summary["mean_steps"]),
        "model_usage": summary["model_usage"],
        "model_usage_ratio": summary["model_usage_ratio"],
        "episode_return_jointR_abs_error": float(max(errors)),
        "episode_return_jointR_abs_error_mean": float(np.mean(errors)),
        "timestamp": utc_now(),
        "episodes": len(rows),
    }
    if train_window:
        record["train_return_mean"] = float(np.mean([
            item["episode_return"] for item in train_window
        ]))
        record["train_return_ma100"] = float(np.mean([
            item["episode_return"] for item in train_window[-100:]
        ]))
        record["train_steps_mean"] = float(np.mean([
            item["steps"] for item in train_window
        ]))
    if extra:
        record.update(extra)
    return record


def classify_status(joints, reached_cap):
    """Convergence statement after the 3500-episode cap (no early stop)."""
    if not joints:
        return {
            "status": "training",
            "converged": False,
            "stable_platform": False,
            "improving": False,
            "reached_cap": bool(reached_cap),
        }
    best = float(max(joints))
    recent = [float(value) for value in joints[-3:]]
    stable = (
        len(joints) >= 5
        and (max(recent) - min(recent)) <= 0.01
        and float(joints[-1]) >= best - 0.005
    )
    improving = (
        len(joints) == 1
        or (len(joints) >= 2 and float(joints[-1]) > max(joints[:-1]) + 0.002)
    )
    if stable and not improving:
        status = "converged"
    elif improving:
        status = "improving"
    elif reached_cap:
        status = "plateaued_not_converged"
    else:
        status = "plateaued"
    return {
        "status": status,
        "converged": bool(stable and not improving),
        "stable_platform": bool(stable),
        "improving": bool(improving),
        "reached_cap": bool(reached_cap),
    }


def write_validation_csv(path: Path, rows):
    fields = [
        "episode", "joint_reward", "SR", "OSR", "SPL", "nDTW",
        "E2E_ms", "P95_ms", "delay_violation_rate", "mean_steps",
        "episode_return_jointR_abs_error", "timestamp",
    ]
    write_csv(path, rows, fieldnames=fields)


def status_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    progress = read_json(run_dir / "progress.json") or {}
    history = read_json(run_dir / "training_history.json") or {}
    validation = history.get("validation", [])
    last = validation[-1] if validation else None
    return {
        "run_dir": str(run_dir),
        "status": progress.get("status"),
        "current_episode": progress.get("current_episode"),
        "best_dev_joint": progress.get("best_dev_joint"),
        "last_validation": last,
        "validation_count": len(validation),
        "updated_utc": progress.get("updated_utc"),
    }
