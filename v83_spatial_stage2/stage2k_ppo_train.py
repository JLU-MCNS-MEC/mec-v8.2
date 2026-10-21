"""Stage-2k PPO: on-policy counterpart of the JointR-aligned DQN.

Fairness contract (verified at startup against the DQN run):
  * identical 384-D pure-MEC v2 state + separate 20-D availability mask
  * identical JointR-aligned reward (no success bonus / failure penalty)
  * identical 3500-episode Train order, Train topology and 60-episode Dev
    Validation with Dev topology
  * the same five frozen Stage-2j T5 MECs and the same Dev capability prior

PPO runs in its own process and run directory.  It never reads Final Test,
never stops on ordinary reward decline, and only aborts on numerical
instability or program errors (diagnostics are saved before exit).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BANDS,
    SCENARIOS,
    atomic_write_json,
    load_split_specs,
    load_topology,
    read_json,
    sha256_file,
    summarize_episodes,
    utc_now,
)
from v83_spatial_stage2.stage2k_dqn import (
    SEED,
    VALIDATION_EVERY,
    VALIDATION_IMPROVEMENT,
)
from v83_spatial_stage2.stage2k_jointr_v2_order import (
    order_statistics,
    random_training_order,
    verify_order,
    verify_validation_isolation,
)
from v83_spatial_stage2.stage2k_jointr_v2_train import classify_status
from v83_spatial_stage2.stage2k_ppo_agent import (
    CLIP_EPS,
    ENTROPY_COEF,
    GAE_LAMBDA,
    GAMMA,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MINIBATCH_SIZE,
    ROLLOUT_STEPS,
    TARGET_KL,
    UPDATE_EPOCHS,
    VALUE_COEF,
    PPOAgent,
    PPOAnomalyError,
    RolloutBuffer,
)
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (
    STATE_DIM_PURE_MEC_V2,
    state_layout_pure_mec_v2,
)
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    SceneCache,
    build_scene,
    load_gab_config,
    load_runtime,
)
from v83_spatial_stage2.stage2k_v2_common import (
    CAPABILITY_PRIOR_PATH,
    load_capability_prior,
)

STAGE_NAME = "stage2k_ppo"
STATE_MODE = "pure_mec_v2"
STATE_DIM = STATE_DIM_PURE_MEC_V2
MAX_EPISODES = 3500
DEFAULT_RUN_DIR = Path("runs/v83_spatial_stage2/stage2k_ppo")
DEFAULT_TENSORBOARD_DIR = Path(
    "runs/tensorboard/association_stage2k_ppo"
)
DQN_REFERENCE_DIR = Path(
    "runs/v83_spatial_stage2/stage2k_pure_mec_v2_smoke"
)
DQN_FORMAL_DIR = Path("runs/v83_spatial_stage2/stage2k_pure_mec_v2")
PROTOCOL = "stage2k-ppo-3500-no-early-stop"

RUN_DIR = DEFAULT_RUN_DIR
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
TENSORBOARD_DIR = DEFAULT_TENSORBOARD_DIR
PROGRESS_PATH = RUN_DIR / "progress.json"
HISTORY_PATH = RUN_DIR / "training_history.json"
ORDER_PATH = RUN_DIR / "training_order.json"
ISOLATION_PATH = RUN_DIR / "validation_isolation.json"
RESULT_PATH = RUN_DIR / "training_result.json"
ALIGNMENT_PATH = RUN_DIR / "protocol_alignment.json"
DIAGNOSTICS_PATH = RUN_DIR / "diagnostics.json"
ENVIRONMENT_PATH = RUN_DIR / "launch_environment.json"
LAST_FULL = CHECKPOINT_DIR / "last_full.pt"
LAST_MODEL = CHECKPOINT_DIR / "last.pt"
BEST_MODEL = CHECKPOINT_DIR / "best.pt"


def set_run_paths(run_dir, tensorboard_dir):
    global RUN_DIR, CHECKPOINT_DIR, TENSORBOARD_DIR
    global PROGRESS_PATH, HISTORY_PATH, ORDER_PATH, ISOLATION_PATH
    global RESULT_PATH, ALIGNMENT_PATH, DIAGNOSTICS_PATH, ENVIRONMENT_PATH
    global LAST_FULL, LAST_MODEL, BEST_MODEL
    RUN_DIR = Path(run_dir)
    CHECKPOINT_DIR = RUN_DIR / "checkpoints"
    TENSORBOARD_DIR = Path(tensorboard_dir)
    PROGRESS_PATH = RUN_DIR / "progress.json"
    HISTORY_PATH = RUN_DIR / "training_history.json"
    ORDER_PATH = RUN_DIR / "training_order.json"
    ISOLATION_PATH = RUN_DIR / "validation_isolation.json"
    RESULT_PATH = RUN_DIR / "training_result.json"
    ALIGNMENT_PATH = RUN_DIR / "protocol_alignment.json"
    DIAGNOSTICS_PATH = RUN_DIR / "diagnostics.json"
    ENVIRONMENT_PATH = RUN_DIR / "launch_environment.json"
    LAST_FULL = CHECKPOINT_DIR / "last_full.pt"
    LAST_MODEL = CHECKPOINT_DIR / "last.pt"
    BEST_MODEL = CHECKPOINT_DIR / "best.pt"


def update_progress(**fields):
    payload = read_json(PROGRESS_PATH) or {
        "stage": STAGE_NAME,
        "created_utc": utc_now(),
        "final_test": "pending",
        "final_test_read": False,
    }
    payload["updated_utc"] = utc_now()
    payload.update(fields)
    atomic_write_json(PROGRESS_PATH, payload)
    return payload


def setup_run():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)


def build_order():
    """Build the frozen 3500-episode order shared with the DQN."""
    train_payload = load_split_specs("train")
    order, targets = random_training_order(train_payload, total=MAX_EPISODES)
    lookup = verify_order(order, train_payload, require_full=False)
    stats = order_statistics(order, lookup)
    reference = None
    if ORDER_PATH.is_file():
        existing = read_json(ORDER_PATH)
        if existing.get("order") != order:
            raise RuntimeError("Existing training_order.json differs")
        reference = existing.get("dqn_reference")
    order_sha = sha256_file(ORDER_PATH) if ORDER_PATH.is_file() else None
    dqn_reference = None
    dqn_order_path = DQN_REFERENCE_DIR / "training_order.json"
    if dqn_order_path.is_file():
        dqn_payload = read_json(dqn_order_path)
        dqn_order = dqn_payload.get("order", [])
        dqn_reference = {
            "path": str(dqn_order_path),
            "sha256": sha256_file(dqn_order_path),
            "order_matches_ppo": dqn_order == order,
            "strategy": dqn_payload.get("strategy"),
            "sampling_seed": dqn_payload.get("sampling_seed"),
        }
        if not dqn_reference["order_matches_ppo"]:
            raise RuntimeError(
                "PPO order differs from the DQN reference order"
            )
    payload = {
        "version": "stage2k-ppo-order-v1",
        "created_utc": utc_now(),
        "seed": SEED,
        "sampling_seed": int(dqn_reference["sampling_seed"])
        if dqn_reference and dqn_reference.get("sampling_seed") is not None
        else None,
        "strategy": (
            "stratified random sample without replacement from Train "
            "(scenario x band targets) + global shuffle"
        ),
        "order_mode": "random",
        "random_total": MAX_EPISODES,
        "target_counts": {
            f"{scenario}/{band}": int(count)
            for (scenario, band), count in targets.items()
        },
        "statistics": stats,
        "dqn_reference": dqn_reference or reference,
        "order": order,
    }
    if not ORDER_PATH.is_file():
        atomic_write_json(ORDER_PATH, payload)
    return order, lookup, stats, payload


def verify_isolation():
    checks, validation = verify_validation_isolation()
    payload = {
        "version": "stage2k-ppo-validation-isolation-v1",
        "created_utc": utc_now(),
        "checks": checks,
        "validation_episodes": [
            {
                "episode_uid": item["episode_uid"],
                "split": item["split"],
                "scenario": item["scenario"],
                "band": item["band"],
                "seed": int(item["seed"]),
            }
            for item in validation
        ],
    }
    atomic_write_json(ISOLATION_PATH, payload)
    if not checks["ok"]:
        raise SystemExit(f"Validation isolation failed: {checks}")
    return validation


def evaluate_policy(agent, specs, topologies, encoder, backends, quality,
                    gab_config, device):
    """Greedy PPO rollout on the fixed Dev Validation set."""
    cache = SceneCache(encoder, device, max_entries=len(specs))
    rows = []
    for spec in specs:
        runner = EpisodeRunner(
            spec=spec,
            topology=topologies[spec["scenario"]],
            encoder=encoder,
            backends=backends,
            projector=None,
            quality_prior=quality,
            gab_config=gab_config,
            scene_cache=cache,
            device=device,
            jointr_aligned=True,
            pure_mec_v2_state=True,
        )
        while True:
            action = agent.greedy_action(
                runner.step_data["state"], runner.step_data["mask"]
            )
            transition = runner.apply(action)
            if transition["done"]:
                break
        rows.append(runner.finalize("ppo_greedy"))
    return rows


def protocol_alignment(order_payload, isolation_payload, prior_sha,
                       prior_flat, train_topologies, dev_topologies):
    """Compare PPO's protocol with the real DQN reference configuration."""
    reference_config = None
    reference_path = None
    for candidate in (DQN_FORMAL_DIR, DQN_REFERENCE_DIR):
        progress = read_json(candidate / "progress.json")
        if progress and progress.get("training_config"):
            reference_config = progress["training_config"]
            reference_path = candidate
            break
    checks = {
        "dqn_reference_available": reference_config is not None,
        "training_order_matches_dqn": bool(
            order_payload.get("dqn_reference", {}).get(
                "order_matches_ppo", False
            )
        ),
        "state_dim_384": STATE_DIM == 384,
        "action_dim_20": True,
        "gamma_1": abs(GAMMA - 1.0) < 1e-12,
        "reward_matches": True,
        "validation_60_dev": int(
            isolation_payload["checks"]["validation_episodes"]
        ) == 60,
        "train_topology_split": all(
            int(train_topologies[scenario]["topology_seed"])
            != int(dev_topologies[scenario]["topology_seed"])
            for scenario in SCENARIOS
        ),
    }
    if reference_config is not None:
        checks["state_mode_matches_dqn"] = (
            str(reference_config.get("state_mode")) == STATE_MODE
        )
        checks["state_dim_matches_dqn"] = (
            int(reference_config.get("state_dim", -1)) == STATE_DIM
        )
        checks["gamma_matches_dqn"] = (
            abs(float(reference_config.get("gamma", -1.0)) - GAMMA) < 1e-12
        )
        checks["reward_matches"] = (
            str(reference_config.get("reward"))
            == (
                "r_nav_t + (-0.25/100)*(running_mean_E2E_t - "
                "running_mean_E2E_{t-1})"
            )
        )
        checks["validation_matches_dqn"] = (
            str(reference_config.get("validation_source"))
            == "60 Dev episodes (6/10/4 per scenario)"
        )
        checks["order_mode_matches_dqn"] = (
            str(reference_config.get("order_mode")) == "random"
        )
        checks["train_topology_matches_dqn"] = (
            str(reference_config.get("train_topology_split")) == "train"
        )
        checks["capability_prior_matches_dqn"] = bool(
            reference_config.get("capability_prior")
            and reference_config["capability_prior"].get("sha256")
            == prior_sha
        )
        checks["dqn_max_episodes"] = int(
            reference_config.get("max_episodes", -1)
        )
    payload = {
        "version": "stage2k-ppo-protocol-alignment-v1",
        "created_utc": utc_now(),
        "dqn_reference_dir": str(reference_path) if reference_path else None,
        "dqn_reference_is_formal": bool(
            reference_path is not None
            and int(reference_config.get("max_episodes", 0))
            >= MAX_EPISODES
        ),
        "ppo_config": {
            "state_mode": STATE_MODE,
            "state_dim": STATE_DIM,
            "decision_state_dim": STATE_DIM + 20,
            "action_dim": 20,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_eps": CLIP_EPS,
            "reward": "environment reward; no shaping",
            "training_order_sha256": sha256_file(ORDER_PATH),
            "validation_isolation_sha256": sha256_file(ISOLATION_PATH),
            "capability_prior": str(CAPABILITY_PRIOR_PATH),
            "capability_prior_sha256": prior_sha,
            "capability_prior_values": prior_flat,
            "train_topology_seeds": {
                scenario: int(train_topologies[scenario]["topology_seed"])
                for scenario in SCENARIOS
            },
            "dev_topology_seeds": {
                scenario: int(dev_topologies[scenario]["topology_seed"])
                for scenario in SCENARIOS
            },
            "band_counts": order_payload["statistics"]["counts"],
            "scenario_band_windows": (
                order_payload["statistics"][
                    "all_full_windows_cover_9_groups"
                ]
            ),
        },
        "dqn_config": (
            {
                key: reference_config.get(key)
                for key in (
                    "state_mode", "state_dim", "action_dim", "gamma",
                    "reward", "max_episodes", "validation_source",
                    "training_order", "capability_prior",
                )
            }
            if reference_config else None
        ),
        "checks": checks,
        "ok": all(
            value for key, value in checks.items()
            if isinstance(value, bool)
        ),
    }
    atomic_write_json(ALIGNMENT_PATH, payload)
    return payload


def resolve_device(requested):
    requested = str(requested).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def write_launch_environment(device, agent, args):
    payload = {
        "created_utc": utc_now(),
        "pid": int(os.getpid()),
        "cwd": str(Path.cwd()),
        "python": sys.executable,
        "argv": list(sys.argv),
        "device_requested": str(args.device),
        "device_selected": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "cpu_count": os.cpu_count(),
        "torch_num_threads": int(torch.get_num_threads()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "run_dir": str(RUN_DIR),
        "tensorboard_dir": str(TENSORBOARD_DIR),
        "agent_config": agent.config(),
    }
    if torch.cuda.is_available() and device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        payload["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_mb": round(total_bytes / 2**20, 1),
            "memory_free_at_start_mb": round(free_bytes / 2**20, 1),
        }
    atomic_write_json(ENVIRONMENT_PATH, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--tensorboard-dir", type=Path, default=DEFAULT_TENSORBOARD_DIR
    )
    parser.add_argument("--max-episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--smoke-episodes", type=int, default=None)
    parser.add_argument(
        "--validation-every", type=int, default=VALIDATION_EVERY
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument(
        "--minibatch-size", type=int, default=MINIBATCH_SIZE
    )
    parser.add_argument("--update-epochs", type=int, default=UPDATE_EPOCHS)
    parser.add_argument(
        "--capability-prior", type=Path, default=CAPABILITY_PRIOR_PATH
    )
    args = parser.parse_args()
    smoke = args.smoke_episodes is not None
    if smoke:
        args.max_episodes = int(args.smoke_episodes)
    if not smoke:
        if int(args.max_episodes) != MAX_EPISODES:
            raise SystemExit(
                "Formal PPO training requires --max-episodes 3500"
            )
        if int(args.validation_every) != VALIDATION_EVERY:
            raise SystemExit(
                "Formal PPO training requires --validation-every 200"
            )
        if (
            int(args.rollout_steps) != ROLLOUT_STEPS
            or int(args.minibatch_size) != MINIBATCH_SIZE
            or int(args.update_epochs) != UPDATE_EPOCHS
        ):
            raise SystemExit(
                "Formal PPO training requires the standard update "
                f"({ROLLOUT_STEPS}/{MINIBATCH_SIZE}/{UPDATE_EPOCHS})"
            )
    set_run_paths(args.run_dir, args.tensorboard_dir)
    setup_run()
    if LAST_FULL.is_file() and not args.resume:
        raise SystemExit(
            f"{LAST_FULL} exists; use --resume or a fresh run directory"
        )

    threads = int(os.environ.get("PPO_TORCH_THREADS", "8"))
    torch.set_num_threads(max(1, threads))
    device = resolve_device(args.device)

    order, lookup, order_stats, order_payload = build_order()
    validation_specs_list = verify_isolation()
    capability_payload, capability_flat = load_capability_prior(
        args.capability_prior
    )
    prior_sha = sha256_file(args.capability_prior)
    quality = capability_flat
    train_topologies = {
        scenario: load_topology(scenario, "train")
        for scenario in SCENARIOS
    }
    dev_topologies = {
        scenario: load_topology(scenario, "dev") for scenario in SCENARIOS
    }
    _, encoder, backends = load_runtime(device=str(device))
    gab_config = load_gab_config()

    agent = PPOAgent(
        device,
        seed=args.seed,
        state_dim=STATE_DIM,
        state_mode=STATE_MODE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_eps=CLIP_EPS,
        learning_rate=LEARNING_RATE,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=TARGET_KL,
    )
    environment = write_launch_environment(device, agent, args)
    alignment = protocol_alignment(
        order_payload,
        read_json(ISOLATION_PATH),
        prior_sha,
        capability_flat,
        train_topologies,
        dev_topologies,
    )
    print(
        "[PPO] protocol alignment: "
        + json.dumps(alignment["checks"], ensure_ascii=False),
        flush=True,
    )
    if not alignment["ok"]:
        raise SystemExit(
            "Protocol alignment failed: "
            f"{alignment['checks']}"
        )

    history = {
        "train": [],
        "validation": [],
        "config": {
            **agent.config(),
            "max_episodes": int(args.max_episodes),
            "validation_every": int(args.validation_every),
            "validation_source": "60 Dev episodes (6/10/4 per scenario)",
            "training_order": str(ORDER_PATH),
            "order_mode": "random",
            "sampling_frame": (
                "train_4000 stratified random sample (3500)"
            ),
            "train_topology_split": "train",
            "validation_topology_split": "dev",
            "state_layout": state_layout_pure_mec_v2(),
            "protocol": PROTOCOL,
            "early_stop_disabled": True,
            "capability_prior": {
                "path": str(args.capability_prior),
                "sha256": prior_sha,
                "split": capability_payload["split"],
                "metrics": capability_payload["metrics"],
                "final_test_read": capability_payload["final_test_read"],
            },
            "frozen_mecs": list(capability_payload["models"]),
            "dqn_reference": alignment["dqn_reference_dir"],
            "dqn_reference_is_formal": alignment[
                "dqn_reference_is_formal"
            ],
        },
    }
    start_episode = 0
    best_joint = -1e9
    no_improve = 0
    if args.resume:
        if not LAST_FULL.is_file():
            raise SystemExit("--resume requested but last_full.pt missing")
        payload = agent.load_full(LAST_FULL)
        start_episode = int(payload.get("episode", 0))
        best_joint = float(payload.get("best_joint", -1e9))
        no_improve = int(payload.get("no_improve", 0))
        if HISTORY_PATH.is_file():
            history = read_json(HISTORY_PATH)
        if len(history["train"]) < start_episode:
            raise SystemExit(
                f"History has {len(history['train'])} rows but checkpoint "
                f"is at episode {start_episode}"
            )
        if len(history["train"]) > start_episode:
            history["train"] = history["train"][:start_episode]
            history["validation"] = [
                row for row in history["validation"]
                if int(row["episode"]) <= start_episode
            ]
            print(
                f"[PPO] truncated unsaved history rows to episode "
                f"{start_episode}",
                flush=True,
            )
        print(
            f"[PPO] resumed at episode {start_episode}, "
            f"best={best_joint:.6f}, no_improve={no_improve}",
            flush=True,
        )
    if start_episode >= args.max_episodes:
        raise SystemExit("Checkpoint episode already reaches the cap")

    reward_window = deque(
        (float(row["episode_return"]) for row in history["train"][-99:]),
        maxlen=100,
    )
    steps_window = deque(
        (int(row["steps"]) for row in history["train"][-99:]), maxlen=100
    )
    writer = SummaryWriter(str(TENSORBOARD_DIR))
    rollout = RolloutBuffer()
    update_stats = []
    started = time.time()
    last_validation = None
    last_update = None
    status_payload = classify_status(
        [row["joint_reward"] for row in history["validation"]],
        False, False,
    )
    update_progress(
        status="training",
        algorithm="ppo",
        current_episode=int(start_episode),
        best_checkpoint=str(BEST_MODEL) if BEST_MODEL.is_file() else None,
        best_dev_joint=float(best_joint) if best_joint > -1e8 else None,
        consecutive_no_improve=int(no_improve),
        last_validation=last_validation,
        early_stop_triggered=False,
        converged=False,
        training_config=history["config"],
        training_order=str(ORDER_PATH),
        validation_isolation=str(ISOLATION_PATH),
        protocol_alignment=str(ALIGNMENT_PATH),
        reward_version="jointr_aligned_v2",
        device=str(device),
        launch_environment=str(ENVIRONMENT_PATH),
        final_test="pending",
        final_test_read=False,
    )

    def abort_with_diagnostics(exc, episode_number, phase, extra=None):
        details = {
            "version": "stage2k-ppo-diagnostics-v1",
            "created_utc": utc_now(),
            "episode": int(episode_number),
            "phase": str(phase),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "details": getattr(exc, "diagnostics", None),
            "rollout_steps": len(rollout),
            "last_update": last_update,
            "recent_train": history["train"][-5:],
            "recent_validation": history["validation"][-3:],
        }
        if extra:
            details.update(extra)
        atomic_write_json(DIAGNOSTICS_PATH, details)
        agent.save_full(LAST_FULL, extra={
            "episode": int(episode_number),
            "best_joint": best_joint,
            "no_improve": no_improve,
            "last_validation": last_validation,
            "aborted": str(exc),
        })
        update_progress(
            status="numerical_aborted",
            current_episode=int(episode_number),
            best_checkpoint=str(BEST_MODEL) if BEST_MODEL.is_file() else None,
            best_dev_joint=float(best_joint) if best_joint > -1e8 else None,
            last_validation=last_validation,
            diagnostics=str(DIAGNOSTICS_PATH),
            abort_reason=str(exc),
            training_finished_utc=utc_now(),
        )
        print(
            f"[PPO][ABORT] episode {episode_number} {phase}: {exc}",
            flush=True,
        )
        raise SystemExit(2)

    for episode_index in range(start_episode, int(args.max_episodes)):
        episode_number = episode_index + 1
        spec = lookup[order[episode_index]]
        runner = EpisodeRunner(
            spec=spec,
            topology=train_topologies[spec["scenario"]],
            encoder=encoder,
            backends=backends,
            projector=None,
            quality_prior=quality,
            gab_config=gab_config,
            scene=build_scene(spec, encoder, device),
            device=device,
            jointr_aligned=True,
            pure_mec_v2_state=True,
        )
        state = runner.step_data["state"]
        mask = runner.step_data["mask"]
        done = False
        entropies = []
        values = []
        while not done:
            try:
                decision = agent.act(state, mask)
            except PPOAnomalyError as exc:
                abort_with_diagnostics(exc, episode_number, "act")
            action = int(decision["action"])
            transition = runner.apply(action)
            rollout.add(
                state, mask, action, decision["log_prob"],
                decision["value"], transition["reward"],
                transition["done"],
            )
            entropies.append(float(decision["entropy"]))
            values.append(float(decision["value"]))
            agent.total_steps += 1
            state = transition["next_state"]
            mask = transition["next_mask"]
            done = bool(transition["done"])
        result = runner.finalize("jointr_v2")
        error = float(result["jointr_abs_error"])
        if error > 1e-6:
            agent.save_full(LAST_FULL, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
            })
            raise RuntimeError(
                f"Reward/JointR mismatch at episode {episode_number}: "
                f"{error}"
            )
        finite_values = (
            result["episode_return"], result["episode_jointR"],
            result["mean_episode_e2e_ms"], result["final_ndtw"],
        )
        if not np.all(np.isfinite(finite_values)):
            abort_with_diagnostics(
                PPOAnomalyError(
                    "Non-finite episode metric",
                    {"episode_metrics": list(map(float, finite_values))},
                ),
                episode_number, "episode",
            )
        if not np.all(np.isfinite(np.asarray(entropies + values))):
            abort_with_diagnostics(
                PPOAnomalyError(
                    "Non-finite policy entropy/value during rollout",
                    {
                        "entropy_mean": float(np.mean(entropies)),
                        "value_mean": float(np.mean(values)),
                    },
                ),
                episode_number, "rollout",
            )
        if abs(float(np.mean(values))) > 1e3:
            abort_with_diagnostics(
                PPOAnomalyError(
                    "Value function divergence",
                    {
                        "value_mean": float(np.mean(values)),
                        "value_abs_max": float(np.max(np.abs(values))),
                    },
                ),
                episode_number, "rollout",
            )
        row = {
            "episode": episode_number,
            "episode_uid": spec["episode_uid"],
            "scenario": spec["scenario"],
            "band": spec["band"],
            "steps": int(result["steps"]),
            "episode_return": float(result["episode_return"]),
            "episode_jointR": float(result["episode_jointR"]),
            "navigation_reward": float(sum(
                item["navigation_reward"] for item in result["step_records"]
            )),
            "latency_penalty": float(sum(
                item["latency_penalty"] for item in result["step_records"]
            )),
            "mean_episode_e2e_ms": float(result["mean_episode_e2e_ms"]),
            "final_nDTW": float(result["final_ndtw"]),
            "success": bool(result["success"]),
            "jointr_abs_error": error,
            "policy_entropy": float(np.mean(entropies)),
            "value_mean": float(np.mean(values)),
            "value_abs_max": float(np.max(np.abs(values))),
            "total_steps": int(agent.total_steps),
            "rollout_steps": int(len(rollout)),
            "rollout_updates": int(agent.rollout_updates),
            "policy_updates": int(agent.update_steps),
        }
        if len(rollout) >= agent.rollout_steps:
            try:
                stats = agent.update(rollout)
            except PPOAnomalyError as exc:
                abort_with_diagnostics(exc, episode_number, "update")
            last_update = stats
            update_stats.append(stats)
            rollout.clear()
            writer.add_scalar(
                "train/update_policy_loss", stats["policy_loss"],
                episode_number,
            )
            writer.add_scalar(
                "train/update_value_loss", stats["value_loss"],
                episode_number,
            )
            writer.add_scalar(
                "train/update_entropy", stats["entropy"], episode_number
            )
            writer.add_scalar(
                "train/update_approx_kl", stats["approx_kl"],
                episode_number,
            )
            writer.add_scalar(
                "train/update_clip_fraction", stats["clip_fraction"],
                episode_number,
            )
            writer.add_scalar(
                "train/update_grad_norm", stats["grad_norm"],
                episode_number,
            )
            writer.add_scalar(
                "train/update_explained_variance",
                stats["explained_variance"], episode_number,
            )
            if stats["approx_kl"] > 0.5:
                abort_with_diagnostics(
                    PPOAnomalyError(
                        "Approximate KL above stability threshold",
                        dict(stats),
                    ),
                    episode_number, "update",
                )
        history["train"].append(row)
        writer.add_scalar(
            "train/episode_return", row["episode_return"], episode_number
        )
        writer.add_scalar(
            "train/episode_jointR", row["episode_jointR"], episode_number
        )
        writer.add_scalar(
            "train/navigation_reward", row["navigation_reward"],
            episode_number,
        )
        writer.add_scalar(
            "train/latency_penalty", row["latency_penalty"],
            episode_number,
        )
        writer.add_scalar(
            "train/mean_episode_e2e_ms", row["mean_episode_e2e_ms"],
            episode_number,
        )
        writer.add_scalar(
            "train/policy_entropy", row["policy_entropy"], episode_number
        )
        writer.add_scalar(
            "train/value_mean", row["value_mean"], episode_number
        )
        reward_window.append(row["episode_return"])
        steps_window.append(row["steps"])
        if len(reward_window) == reward_window.maxlen:
            writer.add_scalar(
                "train/episode_return_ma100",
                float(np.mean(reward_window)), episode_number,
            )
            writer.add_scalar(
                "train/steps_ma100", float(np.mean(steps_window)),
                episode_number,
            )
        if episode_number % 10 == 0:
            atomic_write_json(HISTORY_PATH, history)
            update_progress(
                status="training",
                current_episode=episode_number,
                best_checkpoint=str(BEST_MODEL)
                if BEST_MODEL.is_file() else None,
                best_dev_joint=float(best_joint)
                if best_joint > -1e8 else None,
                consecutive_no_improve=int(no_improve),
                last_validation=last_validation,
                early_stop_triggered=False,
                converged=False,
                last_update=last_update,
            )
            print(
                f"[PPO] Ep{episode_number}/{args.max_episodes} "
                f"ret={row['episode_return']:.4f} "
                f"jointR={row['episode_jointR']:.4f} "
                f"e2e={row['mean_episode_e2e_ms']:.1f}ms "
                f"ent={row['policy_entropy']:.3f} "
                f"V={row['value_mean']:.3f} "
                f"kl={(last_update['approx_kl'] if last_update else 0.0):.4f} "
                f"band={row['band']}",
                flush=True,
            )
        if episode_number % 50 == 0:
            agent.save_full(LAST_FULL, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
                "last_validation": last_validation,
            })
            agent.save_model(LAST_MODEL, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
            })

        if (
            episode_number % int(args.validation_every) == 0
            or episode_number == int(args.max_episodes)
        ):
            validation_rows = evaluate_policy(
                agent, validation_specs_list, dev_topologies, encoder,
                backends, quality, gab_config, device,
            )
            summary = summarize_episodes(validation_rows)
            val_errors = [
                float(item["jointr_abs_error"]) for item in validation_rows
            ]
            if max(val_errors) > 1e-6:
                raise RuntimeError(
                    f"Validation reward/JointR mismatch: {max(val_errors)}"
                )
            recent_train = [
                item for item in history["train"]
                if int(item["episode"]) > episode_number
                - int(args.validation_every)
            ]
            recent_entropy = [
                float(item["policy_entropy"]) for item in recent_train
            ]
            recent_value = [
                float(item["value_mean"]) for item in recent_train
            ]
            recent_updates = update_stats[-max(
                1, len(update_stats)
            ):]
            update_stats = []
            validation_record = {
                "episode": episode_number,
                "joint_reward": float(summary["JointR"]),
                "SR": float(summary["SR"]),
                "OSR": float(summary["OSR"]),
                "SPL": float(summary["SPL"]),
                "nDTW": float(summary["nDTW"]),
                "E2E_ms": float(summary["E2E_ms"]),
                "P95_ms": float(summary["P95_ms"]),
                "mean_steps": float(summary["mean_steps"]),
                "model_usage": summary["model_usage"],
                "model_usage_ratio": summary["model_usage_ratio"],
                "train_return_mean": float(np.mean([
                    item["episode_return"] for item in recent_train
                ])) if recent_train else 0.0,
                "train_return_ma100": float(np.mean(reward_window)),
                "policy_entropy": float(np.mean(recent_entropy))
                if recent_entropy else 0.0,
                "value_mean": float(np.mean(recent_value))
                if recent_value else 0.0,
                "update_policy_loss": float(np.mean([
                    item["policy_loss"] for item in recent_updates
                ])) if recent_updates else 0.0,
                "update_value_loss": float(np.mean([
                    item["value_loss"] for item in recent_updates
                ])) if recent_updates else 0.0,
                "update_entropy": float(np.mean([
                    item["entropy"] for item in recent_updates
                ])) if recent_updates else 0.0,
                "approx_kl": float(np.mean([
                    item["approx_kl"] for item in recent_updates
                ])) if recent_updates else 0.0,
                "clip_fraction": float(np.mean([
                    item["clip_fraction"] for item in recent_updates
                ])) if recent_updates else 0.0,
                "explained_variance": float(np.mean([
                    item["explained_variance"] for item in recent_updates
                ])) if recent_updates else 0.0,
                "episode_return_jointR_abs_error": float(max(val_errors)),
                "episode_return_jointR_abs_error_mean": float(
                    np.mean(val_errors)
                ),
                "timestamp": utc_now(),
            }
            history["validation"].append(validation_record)
            last_validation = validation_record
            for key, tag in (
                ("joint_reward", "validation/JointR"),
                ("SR", "validation/SR"),
                ("OSR", "validation/OSR"),
                ("SPL", "validation/SPL"),
                ("nDTW", "validation/nDTW"),
                ("E2E_ms", "validation/E2E_ms"),
                ("P95_ms", "validation/P95_ms"),
                ("mean_steps", "validation/mean_steps"),
                ("policy_entropy", "validation/policy_entropy"),
                ("update_value_loss", "validation/value_loss"),
                ("approx_kl", "validation/approx_kl"),
                ("clip_fraction", "validation/clip_fraction"),
                ("explained_variance", "validation/explained_variance"),
            ):
                writer.add_scalar(tag, validation_record[key], episode_number)
            for name, count in validation_record["model_usage"].items():
                writer.add_scalar(
                    f"validation/model_usage/{name}", count, episode_number
                )
            improved = (
                validation_record["joint_reward"]
                > best_joint + VALIDATION_IMPROVEMENT
            )
            if improved:
                best_joint = validation_record["joint_reward"]
                no_improve = 0
                agent.save_model(BEST_MODEL, extra={
                    "episode": episode_number,
                    "best_joint": best_joint,
                    "no_improve": no_improve,
                    "validation": validation_record,
                })
            else:
                no_improve += 1
            agent.save_full(LAST_FULL, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
                "last_validation": validation_record,
            })
            agent.save_model(LAST_MODEL, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
                "validation": validation_record,
            })
            joints = [
                item["joint_reward"] for item in history["validation"]
            ]
            status_payload = classify_status(
                joints, False, episode_number >= args.max_episodes
            )
            trend = (
                float(np.polyfit(
                    np.arange(min(3, len(joints)), dtype=np.float64),
                    np.asarray(joints[-3:], dtype=np.float64),
                    1,
                )[0])
                if len(joints) >= 2 else 0.0
            )
            atomic_write_json(HISTORY_PATH, history)
            update_progress(
                status="training",
                current_episode=episode_number,
                best_checkpoint=str(BEST_MODEL)
                if BEST_MODEL.is_file() else None,
                best_dev_joint=float(best_joint),
                consecutive_no_improve=int(no_improve),
                last_validation=validation_record,
                early_stop_triggered=False,
                converged=bool(status_payload["converged"]),
                training_status=status_payload["status"],
                validation_trend_last3=trend,
                last_update=last_update,
            )
            print(
                f"[PPO Validation] Ep{episode_number} "
                f"JointR={validation_record['joint_reward']:.4f} "
                f"SR={validation_record['SR']:.4f} "
                f"OSR={validation_record['OSR']:.4f} "
                f"SPL={validation_record['SPL']:.4f} "
                f"nDTW={validation_record['nDTW']:.4f} "
                f"E2E={validation_record['E2E_ms']:.1f}ms "
                f"ent={validation_record['policy_entropy']:.3f} "
                f"vloss={validation_record['update_value_loss']:.4f} "
                f"kl={validation_record['approx_kl']:.4f} "
                f"best={best_joint:.4f} no_improve={no_improve} "
                f"trend={trend:+.5f} "
                f"status={status_payload['status']}",
                flush=True,
            )

    final_episode = (
        int(history["train"][-1]["episode"])
        if history["train"] else int(start_episode)
    )
    joints = [item["joint_reward"] for item in history["validation"]]
    reached_cap = final_episode >= int(args.max_episodes)
    status_payload = classify_status(joints, False, reached_cap)
    trend = (
        float(np.polyfit(
            np.arange(3, dtype=np.float64),
            np.asarray(joints[-3:], dtype=np.float64),
            1,
        )[0])
        if len(joints) >= 3 else 0.0
    )
    agent.save_full(LAST_FULL, extra={
        "episode": final_episode,
        "best_joint": best_joint,
        "no_improve": no_improve,
        "last_validation": last_validation,
        "status": status_payload,
    })
    agent.save_model(LAST_MODEL, extra={
        "episode": final_episode,
        "best_joint": best_joint,
        "no_improve": no_improve,
        "last_validation": last_validation,
        "status": status_payload,
    })
    if not BEST_MODEL.is_file():
        agent.save_model(BEST_MODEL, extra={
            "episode": final_episode,
            "best_joint": best_joint,
            "no_improve": no_improve,
            "validation": last_validation,
        })
    atomic_write_json(HISTORY_PATH, history)
    final_update = history["validation"][-1] if history["validation"] else {}
    training_result = {
        "version": "stage2k-ppo-training-result-v1",
        "created_utc": utc_now(),
        "algorithm": "ppo",
        "episodes_trained": int(final_episode),
        "max_episodes": int(args.max_episodes),
        "best_checkpoint": str(BEST_MODEL),
        "last_checkpoint": str(LAST_MODEL),
        "last_full_checkpoint": str(LAST_FULL),
        "best_validation_joint": float(best_joint),
        "best_validation_episode": int(
            max(history["validation"],
                key=lambda row: row["joint_reward"])["episode"]
        ) if history["validation"] else None,
        "last_validation": last_validation,
        "validation_trend_last3": trend,
        "stop_reason": (
            "reached episode cap" if reached_cap else "training loop ended"
        ),
        "early_stop_triggered": False,
        "status": status_payload["status"],
        "converged": bool(status_payload["converged"]),
        "stable_platform": bool(status_payload["stable_platform"]),
        "improving": bool(status_payload["improving"]),
        "consecutive_no_improve": int(no_improve),
        "reached_cap": bool(reached_cap),
        "early_stop_disabled": True,
        "total_steps": int(agent.total_steps),
        "policy_updates": int(agent.update_steps),
        "rollout_updates": int(agent.rollout_updates),
        "policy_entropy_final": final_update.get("policy_entropy"),
        "value_loss_final": final_update.get("update_value_loss"),
        "approx_kl_final": final_update.get("approx_kl"),
        "clip_fraction_final": final_update.get("clip_fraction"),
        "explained_variance_final": final_update.get(
            "explained_variance"
        ),
        "reward_jointr_abs_error_max": float(max(
            row["jointr_abs_error"] for row in history["train"]
        )) if history["train"] else None,
        "reward_jointr_abs_error_mean": float(np.mean([
            row["jointr_abs_error"] for row in history["train"]
        ])) if history["train"] else None,
        "convergence_criteria": {
            "definition": (
                "completed episode cap; no NaN/Inf or value/KL divergence; "
                "at least 5 Dev validations; last-3 JointR range <= 0.01; "
                "last JointR >= best - 0.005; no new best in the last "
                "validation (improving=false)"
            ),
            "completed_cap": bool(reached_cap),
            "stable_platform": bool(status_payload["stable_platform"]),
            "improving": bool(status_payload["improving"]),
            "numerically_finite": True,
        },
        "convergence_gate_passed": bool(status_payload["converged"]),
        "final_test_decision": "pending_predefined_gate_review",
        "final_test_read": False,
        "elapsed_s": time.time() - started,
        "config": history["config"],
        "history": history,
    }
    atomic_write_json(RESULT_PATH, training_result)
    update_progress(
        status="completed",
        current_episode=int(final_episode),
        best_checkpoint=str(BEST_MODEL),
        best_dev_joint=float(best_joint),
        consecutive_no_improve=int(no_improve),
        last_validation=last_validation,
        early_stop_triggered=False,
        converged=bool(status_payload["converged"]),
        training_status=status_payload["status"],
        validation_trend_last3=trend,
        stop_reason=training_result["stop_reason"],
        training_finished_utc=utc_now(),
        training_result=str(RESULT_PATH),
    )
    writer.close()
    print(json.dumps({
        "episodes_trained": final_episode,
        "best_validation_joint": best_joint,
        "best_validation_episode": training_result[
            "best_validation_episode"
        ],
        "validation_trend_last3": trend,
        "status": status_payload["status"],
        "converged": status_payload["converged"],
        "reward_abs_error_max": training_result[
            "reward_jointr_abs_error_max"
        ],
        "best_checkpoint": str(BEST_MODEL),
        "last_full_checkpoint": str(LAST_FULL),
        "elapsed_s": training_result["elapsed_s"],
    }, indent=2), flush=True)
    print("STAGE2K_PPO_TRAIN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
