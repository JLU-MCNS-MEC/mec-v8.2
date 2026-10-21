"""Stage-2k v2: standard DQN trained with an exactly JointR-aligned reward.

Key changes from v1:
  * r_delay_t uses the running mean E2E delay, so
    sum(rewards) = final_nDTW - 0.25 * mean_E2E / 100 = episode JointR.
  * gamma = 1.0 (no discounting), standard TD target without terminal
    bootstrap.
  * validation uses the real 60 Dev episodes, never train episodes.
  * state_mode selects full (478-D), no_visual (342-D) or pure_mec
    (322-D input + 20-D action mask).
  * order_mode=random draws a stratified random sample from Train (used by
    the pure-MEC 3500-episode protocol); stratified keeps the legacy
    deterministic 3x3 round-robin.
  * train_topology_split separates training topology from Dev validation
    topology; pure_mec requires "train".
  * the full checkpoint includes replay buffer, RNG state, epsilon and
    episode for strict resume.

The five T5 MECs are frozen.  No MEC training, no baseline rerun, no Final
Test read during training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BANDS,
    QUALITY_PRIOR_PATH,
    SCENARIOS,
    SCENARIO_TITLES,
    VISUAL_PCA_PATH,
    atomic_write_json,
    load_split_specs,
    load_topology,
    read_json,
    sha256_file,
    summarize_episodes,
    utc_now,
)
from v83_spatial_stage2.stage2k_dqn import (
    BATCH_SIZE,
    EPSILON_DECAY_STEPS,
    EPSILON_END,
    EPSILON_START,
    GRAD_CLIP,
    LEARNING_RATE,
    MAX_EPISODES,
    MIN_EPISODES,
    PATIENCE_VALIDATIONS,
    REPLAY_CAPACITY,
    SEED,
    TARGET_UPDATE_STEPS,
    VALIDATION_EVERY,
    VALIDATION_IMPROVEMENT,
    WARMUP_STEPS,
    EXTRA_VALIDATIONS,
    flat_quality,
)
from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2, GAMMA_V2
from v83_spatial_stage2.stage2k_jointr_v2_order import (
    RANDOM_ORDER_SEED,
    dev_validation_specs,
    order_statistics,
    random_training_order,
    stratified_training_order,
    verify_order,
    verify_validation_isolation,
)
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    SceneCache,
    build_scene,
    load_gab_config,
    load_runtime,
)
from v83_spatial_stage2.stage2k_no_visual_state import (
    STATE_DIM_NO_VISUAL,
)
from v83_spatial_stage2.stage2k_pure_mec_state import (
    STATE_DIM_PURE_MEC,
)
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (
    STATE_DIM_PURE_MEC_V2,
    STATE_DIM_VISUAL_ABLATION_V2,
    state_layout_pure_mec_v2,
    state_layout_visual_ablation_v2,
)
from v83_spatial_stage2.stage2k_v2_common import (
    CAPABILITY_PRIOR_PATH,
    load_capability_prior,
)
from v83_spatial_stage2.stage2k_state import (
    STATE_DIM,
    VisualProjector,
    masked_argmax,
)

STAGE_NAME = "stage2k_jointr_v2"
RUN_DIR = Path("runs/v83_spatial_stage2/stage2k_jointr_v2")
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
TENSORBOARD_DIR = Path("runs/tensorboard/association_stage2k_jointr_v2")
PROGRESS_PATH = RUN_DIR / "progress.json"
HISTORY_PATH = RUN_DIR / "training_history.json"
ORDER_PATH = RUN_DIR / "training_order.json"
ISOLATION_PATH = RUN_DIR / "validation_isolation.json"
RESULT_PATH = RUN_DIR / "training_result.json"
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


def build_order(order_mode="stratified", random_total=3500):
    train_payload = load_split_specs("train")
    target_counts = None
    if order_mode == "random":
        order, targets = random_training_order(
            train_payload, total=int(random_total)
        )
        lookup = verify_order(
            order, train_payload, require_full=False
        )
        target_counts = {
            f"{scenario}/{band}": int(count)
            for (scenario, band), count in targets.items()
        }
        strategy = (
            "stratified random sample without replacement from Train "
            "(scenario x band targets) + global shuffle"
        )
        version = "stage2k-pure-mec-order-v1"
    elif order_mode == "stratified":
        order = stratified_training_order(train_payload)
        lookup = verify_order(order, train_payload)
        strategy = (
            "deterministic round-robin over 3 scenarios x 3 bands, "
            "per-group seeded shuffle"
        )
        version = "stage2k-jointr-v2-order-v1"
    else:
        raise ValueError(f"Unknown order_mode {order_mode}")
    stats = order_statistics(order, lookup)
    payload = {
        "version": version,
        "created_utc": utc_now(),
        "seed": SEED,
        "sampling_seed": (
            int(RANDOM_ORDER_SEED) if order_mode == "random" else None
        ),
        "strategy": strategy,
        "order_mode": str(order_mode),
        "random_total": int(random_total) if order_mode == "random" else None,
        "target_counts": target_counts,
        "statistics": stats,
        "order": order,
    }
    if ORDER_PATH.is_file():
        existing = read_json(ORDER_PATH)
        if existing.get("order") != order:
            raise RuntimeError("Existing training_order.json differs")
    else:
        atomic_write_json(ORDER_PATH, payload)
    return order, lookup, stats


def verify_isolation():
    checks, validation = verify_validation_isolation()
    payload = {
        "version": "stage2k-jointr-v2-validation-isolation-v1",
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


def evaluate_aligned(agent, specs, topologies, encoder, backends, projector,
                     quality, gab_config, device, no_visual_state=False,
                     pure_mec_state=False, pure_mec_v2_state=False,
                     visual_ablation_v2_state=False):
    cache = SceneCache(encoder, device, max_entries=60)
    rows = []
    for spec in specs:
        runner = EpisodeRunner(
            spec=spec,
            topology=topologies[spec["scenario"]],
            encoder=encoder,
            backends=backends,
            projector=projector,
            quality_prior=quality,
            gab_config=gab_config,
            scene_cache=cache,
            device=device,
            jointr_aligned=True,
            no_visual_state=no_visual_state,
            pure_mec_state=pure_mec_state,
            pure_mec_v2_state=pure_mec_v2_state,
            visual_ablation_v2_state=visual_ablation_v2_state,
        )
        while True:
            state = runner.step_data["state"]
            mask = runner.step_data["mask"]
            action = int(masked_argmax(agent.q_values(state), mask))
            transition = runner.apply(action)
            if transition["done"]:
                break
        rows.append(runner.finalize("jointr_v2"))
    return rows


def should_early_stop(episode, no_improve, joints):
    if int(episode) < MIN_EPISODES:
        return False, None
    if int(no_improve) >= PATIENCE_VALIDATIONS + EXTRA_VALIDATIONS:
        return True, (
            "no new best JointR after "
            f"{PATIENCE_VALIDATIONS}+{EXTRA_VALIDATIONS} validations"
        )
    if int(no_improve) >= PATIENCE_VALIDATIONS and len(joints) >= 3:
        trend = float(np.polyfit(
            np.arange(3, dtype=np.float64),
            np.asarray(joints[-3:], dtype=np.float64),
            1,
        )[0])
        if trend <= 1e-9:
            return True, (
                "no JointR improvement over 5 validations and last-3 "
                "trend is flat/declining"
            )
    return False, None


def early_stop_decision(episode, no_improve, joints, disabled):
    """The v2 protocol disables early stopping entirely.

    Only numerical/program errors abort those runs; normal reward decline does
    not.  Keeping this in one function makes the rule testable without running
    the training loop.
    """
    if disabled:
        return False, None
    return should_early_stop(episode, no_improve, joints)


def classify_status(joints, early_stop_triggered, reached_cap):
    if not joints:
        return {
            "status": "training",
            "converged": False,
            "stable_platform": False,
            "improving": False,
            "early_stop_triggered": bool(early_stop_triggered),
        }
    best = float(max(joints))
    recent = joints[-3:]
    stable = (
        len(joints) >= 5
        and (max(recent) - min(recent)) <= 0.01
        and joints[-1] >= best - 0.005
    )
    improving = (
        len(joints) == 1
        or (len(joints) >= 2 and joints[-1] > max(joints[:-1]) + 0.002)
    )
    if stable and not improving:
        status = "converged"
    elif early_stop_triggered:
        status = "early_stopped_no_improvement"
    elif reached_cap:
        status = "not_converged"
    elif improving:
        status = "improving"
    else:
        status = "plateaued"
    return {
        "status": status,
        "converged": bool(stable and not improving),
        "stable_platform": bool(stable),
        "improving": bool(improving),
        "early_stop_triggered": bool(early_stop_triggered),
    }


def main():
    global RUN_DIR, CHECKPOINT_DIR, TENSORBOARD_DIR
    global PROGRESS_PATH, HISTORY_PATH, ORDER_PATH, ISOLATION_PATH
    global RESULT_PATH, LAST_FULL, LAST_MODEL, BEST_MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--smoke-episodes", type=int, default=None)
    parser.add_argument(
        "--state-mode",
        choices=(
            "full", "no_visual", "pure_mec",
            "pure_mec_v2", "visual_ablation_v2",
        ),
        default="full",
    )
    parser.add_argument(
        "--order-mode", choices=("stratified", "random"),
        default="stratified",
    )
    parser.add_argument(
        "--train-topology-split", choices=("train", "dev"),
        default="dev",
        help=(
            "Topology used for training rollouts.  Use 'train' for the "
            "pure-MEC protocol so Dev topology is validation-only."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--tensorboard-dir", type=Path, default=None
    )
    parser.add_argument(
        "--epsilon-decay-steps", type=int, default=None,
        help=(
            "Actual environment-step epsilon budget.  Defaults to 200000 "
            "for legacy modes and 500000 for the v2 ablations."
        ),
    )
    parser.add_argument(
        "--disable-early-stop", action="store_true",
        help=(
            "Never stop for reward/validation decline; only NaN/Inf, "
            "Q divergence or program errors abort.  Required for the v2 "
            "formal 3500-episode protocol."
        ),
    )
    parser.add_argument(
        "--capability-prior", type=Path, default=CAPABILITY_PRIOR_PATH,
        help="Frozen Dev-only v2 capability prior JSON.",
    )
    args = parser.parse_args()
    no_visual = args.state_mode == "no_visual"
    pure_mec = args.state_mode == "pure_mec"
    pure_mec_v2 = args.state_mode == "pure_mec_v2"
    visual_ablation_v2 = args.state_mode == "visual_ablation_v2"
    v2_state = pure_mec_v2 or visual_ablation_v2
    if pure_mec_v2:
        state_dim = STATE_DIM_PURE_MEC_V2
    elif visual_ablation_v2:
        state_dim = STATE_DIM_VISUAL_ABLATION_V2
    elif pure_mec:
        state_dim = STATE_DIM_PURE_MEC
    elif no_visual:
        state_dim = STATE_DIM_NO_VISUAL
    else:
        state_dim = STATE_DIM
    if args.run_dir is None:
        if pure_mec_v2:
            args.run_dir = Path(
                "runs/v83_spatial_stage2/stage2k_pure_mec_v2"
            )
        elif visual_ablation_v2:
            args.run_dir = Path(
                "runs/v83_spatial_stage2/stage2k_visual_ablation_v2"
            )
        elif pure_mec:
            args.run_dir = Path(
                "runs/v83_spatial_stage2/stage2k_pure_mec_state"
            )
        elif no_visual:
            args.run_dir = Path(
                "runs/v83_spatial_stage2/stage2k_no_visual"
            )
        else:
            args.run_dir = RUN_DIR
    if args.tensorboard_dir is None:
        if pure_mec_v2:
            args.tensorboard_dir = Path(
                "runs/tensorboard/association_stage2k_pure_mec_v2"
            )
        elif visual_ablation_v2:
            args.tensorboard_dir = Path(
                "runs/tensorboard/association_stage2k_visual_ablation_v2"
            )
        elif pure_mec:
            args.tensorboard_dir = Path(
                "runs/tensorboard/association_stage2k_pure_mec_state"
            )
        elif no_visual:
            args.tensorboard_dir = Path(
                "runs/tensorboard/association_stage2k_no_visual"
            )
        else:
            args.tensorboard_dir = TENSORBOARD_DIR
    global STAGE_NAME
    STAGE_NAME = (
        "stage2k_pure_mec_v2" if pure_mec_v2 else
        ("stage2k_visual_ablation_v2" if visual_ablation_v2 else
         ("stage2k_pure_mec_state" if pure_mec else
          ("stage2k_no_visual" if no_visual else "stage2k_jointr_v2")))
    )
    if (pure_mec or v2_state) and args.order_mode != "random":
        raise SystemExit(f"{args.state_mode} requires --order-mode random")
    if (pure_mec or v2_state) and args.train_topology_split != "train":
        raise SystemExit(
            f"{args.state_mode} requires --train-topology-split train"
        )
    if v2_state:
        if args.epsilon_decay_steps is None:
            args.epsilon_decay_steps = 500_000
        if (
            args.smoke_episodes is None
            and int(args.epsilon_decay_steps) != 500_000
        ):
            raise SystemExit(
                "v2 formal training requires --epsilon-decay-steps 500000"
            )
        if args.smoke_episodes is None and not args.disable_early_stop:
            raise SystemExit(
                "v2 formal training requires --disable-early-stop"
            )
        if (
            args.smoke_episodes is None
            and int(args.max_episodes) != 3500
        ):
            raise SystemExit(
                "v2 formal training requires --max-episodes 3500"
            )
    else:
        if args.epsilon_decay_steps is None:
            args.epsilon_decay_steps = EPSILON_DECAY_STEPS
    if (
        pure_mec
        and args.smoke_episodes is None
        and int(args.max_episodes) != 3500
    ):
        raise SystemExit(
            "pure_mec formal training requires --max-episodes 3500"
        )

    RUN_DIR = args.run_dir
    CHECKPOINT_DIR = RUN_DIR / "checkpoints"
    TENSORBOARD_DIR = args.tensorboard_dir
    PROGRESS_PATH = RUN_DIR / "progress.json"
    HISTORY_PATH = RUN_DIR / "training_history.json"
    ORDER_PATH = RUN_DIR / "training_order.json"
    ISOLATION_PATH = RUN_DIR / "validation_isolation.json"
    RESULT_PATH = RUN_DIR / "training_result.json"
    LAST_FULL = CHECKPOINT_DIR / "last_full.pt"
    LAST_MODEL = CHECKPOINT_DIR / "last.pt"
    BEST_MODEL = CHECKPOINT_DIR / "best.pt"
    setup_run()

    if LAST_FULL.is_file() and not args.resume:
        raise SystemExit(
            f"{LAST_FULL} exists; use --resume or a fresh run directory"
        )
    if args.smoke_episodes is not None:
        args.max_episodes = int(args.smoke_episodes)

    order, lookup, order_stats = build_order(
        args.order_mode, random_total=3500
    )
    validation_specs_list = verify_isolation()
    capability_payload = None
    capability_prior_sha = None
    if v2_state:
        capability_payload, capability_flat = load_capability_prior(
            args.capability_prior
        )
        quality = capability_flat
        capability_prior_sha = sha256_file(args.capability_prior)
    else:
        quality_payload = read_json(QUALITY_PRIOR_PATH)
        if quality_payload is None:
            raise SystemExit("Missing quality prior")
        quality = flat_quality(quality_payload)
    train_topologies = {
        scenario: load_topology(scenario, args.train_topology_split)
        for scenario in SCENARIOS
    }
    dev_topologies = {
        scenario: load_topology(scenario, "dev") for scenario in SCENARIOS
    }
    projector = (
        None if (no_visual or pure_mec or pure_mec_v2)
        else VisualProjector.load(VISUAL_PCA_PATH)
    )
    device, encoder, backends = load_runtime(args.device)
    gab_config = load_gab_config()

    agent = DQNAgentV2(
        device, seed=SEED, state_dim=state_dim, state_mode=args.state_mode,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )
    if pure_mec_v2:
        state_layout = state_layout_pure_mec_v2()
    elif visual_ablation_v2:
        state_layout = state_layout_visual_ablation_v2()
    else:
        state_layout = None
    history = {"train": [], "validation": [], "config": {
        **agent.config(),
        "max_episodes": int(args.max_episodes),
        "min_episodes": MIN_EPISODES,
        "validation_every": VALIDATION_EVERY,
        "validation_source": "60 Dev episodes (6/10/4 per scenario)",
        "training_order": str(ORDER_PATH),
        "order_mode": str(args.order_mode),
        "sampling_frame": (
            "train_4000 stratified random sample (3500)"
            if args.order_mode == "random" else
            "train_4000 full deterministic order"
        ),
        "train_topology_split": str(args.train_topology_split),
        "validation_topology_split": "dev",
        "reward": (
            "r_nav_t + (-0.25/100)*(running_mean_E2E_t - "
            "running_mean_E2E_{t-1})"
        ),
        "state_e2e_field": (
            "running_mean_E2E_ms / 100" if (pure_mec_v2 or visual_ablation_v2)
            else ("none (pure MEC state)" if pure_mec else
                  ("none (no visual state)" if no_visual
                   else "running_mean_E2E_ms / 100"))
        ),
        "state_layout": state_layout,
        "protocol": (
            "stage2k-v2-3500-no-early-stop"
            if v2_state else "stage2k-jointr-v2"
        ),
        "early_stop_disabled": bool(args.disable_early_stop),
        "capability_prior": (
            {
                "path": str(args.capability_prior),
                "sha256": capability_prior_sha,
                "split": capability_payload["split"],
                "metrics": capability_payload["metrics"],
                "final_test_read": capability_payload["final_test_read"],
            }
            if capability_payload is not None else None
        ),
        "visual_projector": (
            {
                "path": str(VISUAL_PCA_PATH),
                "sha256": sha256_file(VISUAL_PCA_PATH),
                "usage": (
                    "frozen 128-D context; GAB/map blocks neutralized to "
                    "the training mean"
                ),
            }
            if visual_ablation_v2 else None
        ),
    }}
    start_episode = 0
    best_joint = -1e9
    no_improve = 0
    if args.resume:
        if not LAST_FULL.is_file():
            raise SystemExit("--resume requested but last_full.pt is missing")
        payload = agent.load_full(LAST_FULL)
        start_episode = int(payload.get("episode", 0))
        best_joint = float(payload.get("best_joint", -1e9))
        no_improve = int(payload.get("no_improve", 0))
        if HISTORY_PATH.is_file():
            history = read_json(HISTORY_PATH)
        if len(history["train"]) != start_episode:
            raise SystemExit(
                f"History has {len(history['train'])} train rows but "
                f"checkpoint is at episode {start_episode}"
            )
        print(
            f"Resumed at episode {start_episode}, "
            f"best={best_joint:.6f}, no_improve={no_improve}",
            flush=True,
        )
    if start_episode > len(order):
        raise SystemExit("Resume episode exceeds training order length")

    return_window = deque(
        (float(row["episode_return"]) for row in history["train"][-99:]),
        maxlen=100,
    )
    navigation_window = deque(
        (float(row["navigation_reward"]) for row in history["train"][-99:]),
        maxlen=100,
    )
    latency_window = deque(
        (float(row["latency_penalty"]) for row in history["train"][-99:]),
        maxlen=100,
    )

    writer = SummaryWriter(str(TENSORBOARD_DIR))
    started = time.time()
    stop_reason = None
    stop_episode = None
    last_validation = None
    status_payload = classify_status(
        [row["joint_reward"] for row in history["validation"]],
        False, False,
    )
    update_progress(
        status="training",
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
        reward_version="jointr_aligned_v2",
        final_test="pending",
        final_test_read=False,
    )

    for episode_index in range(start_episode, min(len(order), args.max_episodes)):
        episode_number = episode_index + 1
        spec = lookup[order[episode_index]]
        topology = train_topologies[spec["scenario"]]
        runner = EpisodeRunner(
            spec=spec,
            topology=topology,
            encoder=encoder,
            backends=backends,
            projector=projector,
            quality_prior=quality,
            gab_config=gab_config,
            scene=build_scene(spec, encoder, device),
            device=device,
            jointr_aligned=True,
            no_visual_state=no_visual,
            pure_mec_state=pure_mec,
            pure_mec_v2_state=pure_mec_v2,
            visual_ablation_v2_state=visual_ablation_v2,
        )
        state = runner.step_data["state"]
        mask = runner.step_data["mask"]
        done = False
        losses = []
        grad_norms = []
        q_means = []
        q_maxs = []
        while not done:
            q_values = agent.q_values(state)
            available_q = q_values[mask]
            q_means.append(float(available_q.mean()))
            q_maxs.append(float(available_q.max()))
            epsilon = agent.epsilon()
            action = agent.act_from_q(q_values, mask, epsilon=epsilon)
            transition = runner.apply(action)
            agent.buffer.add(
                state,
                action,
                transition["reward"],
                transition["next_state"],
                transition["done"],
                transition["next_mask"],
            )
            if agent.total_steps >= WARMUP_STEPS:
                stats = agent.train_step()
                if stats is not None:
                    losses.append(stats["loss"])
                    grad_norms.append(stats["grad_norm"])
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
        if not np.all(np.isfinite([
            result["episode_return"], result["episode_jointR"],
            result["mean_episode_e2e_ms"], result["final_ndtw"],
        ])):
            raise RuntimeError(f"Non-finite episode at {episode_number}")
        if not np.all(np.isfinite(q_means + q_maxs)):
            raise RuntimeError(f"Non-finite Q values at {episode_number}")
        if abs(float(np.mean(q_means))) > 1e6 or abs(float(
            np.max(q_maxs)
        )) > 1e6:
            agent.save_full(LAST_FULL, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
            })
            raise RuntimeError(
                f"Q-value divergence at episode {episode_number}: "
                f"mean={np.mean(q_means)}, max={np.max(q_maxs)}"
            )
        row = {
            "episode": episode_number,
            "episode_uid": spec["episode_uid"],
            "scenario": spec["scenario"],
            "band": spec["band"],
            "steps": int(result["steps"]),
            "episode_return": float(result["episode_return"]),
            "navigation_reward": float(sum(
                item["navigation_reward"] for item in result["step_records"]
            )),
            "latency_penalty": float(sum(
                item["latency_penalty"] for item in result["step_records"]
            )),
            "mean_episode_e2e_ms": float(result["mean_episode_e2e_ms"]),
            "final_nDTW": float(result["final_ndtw"]),
            "episode_jointR": float(result["episode_jointR"]),
            "jointr_abs_error": error,
            "success": bool(result["success"]),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "epsilon": float(agent.epsilon()),
            "q_mean": float(np.mean(q_means)) if q_means else 0.0,
            "q_max": float(np.max(q_maxs)) if q_maxs else 0.0,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "buffer": int(len(agent.buffer)),
            "total_steps": int(agent.total_steps),
        }
        history["train"].append(row)
        writer.add_scalar("train/episode_return", row["episode_return"],
                          episode_number)
        writer.add_scalar("train/navigation_reward",
                          row["navigation_reward"], episode_number)
        writer.add_scalar("train/latency_penalty",
                          row["latency_penalty"], episode_number)
        writer.add_scalar("train/mean_episode_e2e_ms",
                          row["mean_episode_e2e_ms"], episode_number)
        writer.add_scalar("train/loss", row["loss"], episode_number)
        writer.add_scalar("train/grad_norm", row["grad_norm"], episode_number)
        writer.add_scalar("train/epsilon", row["epsilon"], episode_number)
        writer.add_scalar("train/q_mean", row["q_mean"], episode_number)
        writer.add_scalar("train/q_max", row["q_max"], episode_number)
        writer.add_scalar("train/jointr_abs_error", row["jointr_abs_error"],
                          episode_number)
        return_window.append(row["episode_return"])
        navigation_window.append(row["navigation_reward"])
        latency_window.append(row["latency_penalty"])
        if len(return_window) == return_window.maxlen:
            writer.add_scalar(
                "train/episode_return_ma100",
                float(np.mean(return_window)), episode_number,
            )
            writer.add_scalar(
                "train/navigation_reward_ma100",
                float(np.mean(navigation_window)), episode_number,
            )
            writer.add_scalar(
                "train/latency_penalty_ma100",
                float(np.mean(latency_window)), episode_number,
            )
        if episode_number % 10 == 0:
            atomic_write_json(HISTORY_PATH, history)
            update_progress(
                status="training",
                current_episode=episode_number,
                best_checkpoint=str(BEST_MODEL) if BEST_MODEL.is_file() else None,
                best_dev_joint=float(best_joint) if best_joint > -1e8 else None,
                consecutive_no_improve=int(no_improve),
                last_validation=last_validation,
                early_stop_triggered=False,
                converged=False,
            )
            print(
                f"[DQN v2] Ep{episode_number}/{len(order)} "
                f"ret={row['episode_return']:.4f} "
                f"jointR={row['episode_jointR']:.4f} "
                f"err={row['jointr_abs_error']:.2e} "
                f"e2e={row['mean_episode_e2e_ms']:.1f}ms "
                f"loss={row['loss']:.4f} eps={row['epsilon']:.3f} "
                f"Q={row['q_mean']:.2f}/{row['q_max']:.2f} "
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
            episode_number % VALIDATION_EVERY == 0
            or episode_number == int(args.max_episodes)
        ):
            validation_rows = evaluate_aligned(
                agent, validation_specs_list, dev_topologies, encoder,
                backends, projector, quality, gab_config, device,
                no_visual_state=no_visual,
                pure_mec_state=pure_mec,
                pure_mec_v2_state=pure_mec_v2,
                visual_ablation_v2_state=visual_ablation_v2,
            )
            summary = summarize_episodes(validation_rows)
            val_errors = [
                float(item["jointr_abs_error"]) for item in validation_rows
            ]
            if max(val_errors) > 1e-6:
                raise RuntimeError(
                    f"Validation reward/JointR mismatch: {max(val_errors)}"
                )
            recent_losses = [
                item["loss"] for item in history["train"][-VALIDATION_EVERY:]
            ]
            recent_q = [
                item["q_mean"] for item in history["train"][-VALIDATION_EVERY:]
            ]
            recent_qmax = [
                item["q_max"] for item in history["train"][-VALIDATION_EVERY:]
            ]
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
                "train_loss_mean": float(np.mean(recent_losses)),
                "train_q_mean": float(np.mean(recent_q)),
                "train_q_max": float(np.max(recent_qmax)),
                "episode_return_jointR_abs_error": float(max(val_errors)),
                "episode_return_jointR_abs_error_mean": float(
                    np.mean(val_errors)
                ),
                "timestamp": utc_now(),
            }
            history["validation"].append(validation_record)
            last_validation = validation_record
            writer.add_scalar("validation/joint_reward",
                              validation_record["joint_reward"],
                              episode_number)
            writer.add_scalar("validation/SR", validation_record["SR"],
                              episode_number)
            writer.add_scalar("validation/OSR", validation_record["OSR"],
                              episode_number)
            writer.add_scalar("validation/SPL", validation_record["SPL"],
                              episode_number)
            writer.add_scalar("validation/nDTW", validation_record["nDTW"],
                              episode_number)
            writer.add_scalar("validation/E2E_ms",
                              validation_record["E2E_ms"], episode_number)
            writer.add_scalar("validation/P95_ms",
                              validation_record["P95_ms"], episode_number)
            writer.add_scalar("validation/mean_steps",
                              validation_record["mean_steps"], episode_number)
            writer.add_scalar(
                "validation/episode_return_jointR_abs_error",
                validation_record["episode_return_jointR_abs_error"],
                episode_number,
            )
            for name, count in validation_record["model_usage"].items():
                writer.add_scalar(f"validation/model_usage/{name}", count,
                                  episode_number)
            for name, ratio in validation_record[
                "model_usage_ratio"
            ].items():
                writer.add_scalar(
                    f"validation/model_usage_ratio/{name}", ratio,
                    episode_number,
                )
            improved = (
                validation_record["joint_reward"] > best_joint
                + VALIDATION_IMPROVEMENT
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
            atomic_write_json(HISTORY_PATH, history)
            stop, reason = early_stop_decision(
                episode_number, no_improve,
                [item["joint_reward"] for item in history["validation"]],
                args.disable_early_stop,
            )
            status_payload = classify_status(
                [item["joint_reward"] for item in history["validation"]],
                stop, episode_number >= args.max_episodes,
            )
            update_progress(
                status="training" if not stop else "early_stopped",
                current_episode=episode_number,
                best_checkpoint=str(BEST_MODEL) if BEST_MODEL.is_file() else None,
                best_dev_joint=float(best_joint),
                consecutive_no_improve=int(no_improve),
                last_validation=validation_record,
                early_stop_triggered=bool(stop),
                converged=bool(status_payload["converged"]),
                training_status=status_payload["status"],
                stop_reason=reason,
            )
            print(
                f"[Validation] Ep{episode_number} "
                f"JointR={validation_record['joint_reward']:.4f} "
                f"SR={validation_record['SR']:.4f} "
                f"OSR={validation_record['OSR']:.4f} "
                f"SPL={validation_record['SPL']:.4f} "
                f"nDTW={validation_record['nDTW']:.4f} "
                f"E2E={validation_record['E2E_ms']:.1f}ms "
                f"best={best_joint:.4f} no_improve={no_improve} "
                f"err={validation_record['episode_return_jointR_abs_error']:.2e} "
                f"status={status_payload['status']}",
                flush=True,
            )
            if stop:
                stop_reason = reason
                stop_episode = episode_number
                break

    final_episode = int(history["train"][-1]["episode"]) if history["train"] else int(start_episode)
    if stop_reason is None:
        reached_cap = final_episode >= args.max_episodes
        stop_reason = (
            "reached episode cap" if reached_cap else "training loop ended"
        )
        stop_episode = final_episode
    else:
        reached_cap = False
    early_stop_flag = bool(
        stop_reason
        and (stop_reason.startswith("no ")
             or stop_reason.startswith("no new"))
    )
    status_payload = classify_status(
        [item["joint_reward"] for item in history["validation"]],
        early_stop_flag,
        reached_cap,
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
    training_result = {
        "version": "stage2k-jointr-v2-training-result-v1",
        "created_utc": utc_now(),
        "episodes_trained": int(final_episode),
        "max_episodes": int(args.max_episodes),
        "min_episodes": int(MIN_EPISODES),
        "best_checkpoint": str(BEST_MODEL),
        "last_checkpoint": str(LAST_MODEL),
        "last_full_checkpoint": str(LAST_FULL),
        "best_validation_joint": float(best_joint),
        "best_validation_episode": int(
            max(history["validation"], key=lambda row: row["joint_reward"])["episode"]
        ) if history["validation"] else None,
        "last_validation": last_validation,
        "stop_reason": stop_reason,
        "stop_episode": int(stop_episode) if stop_episode else None,
        "early_stop_triggered": bool(
            status_payload["early_stop_triggered"]
        ),
        "status": status_payload["status"],
        "converged": bool(status_payload["converged"]),
        "stable_platform": bool(status_payload["stable_platform"]),
        "improving": bool(status_payload["improving"]),
        "consecutive_no_improve": int(no_improve),
        "reached_cap": bool(reached_cap),
        "early_stop_disabled": bool(args.disable_early_stop),
        "total_steps": int(history["train"][-1]["total_steps"]),
        "epsilon_final": float(history["train"][-1]["epsilon"]),
        "q_mean_final": float(history["train"][-1]["q_mean"]),
        "q_max_final": float(history["train"][-1]["q_max"]),
        "loss_final": float(history["train"][-1]["loss"]),
        "grad_norm_final": float(history["train"][-1]["grad_norm"]),
        "convergence_criteria": {
            "definition": (
                "completed episode cap; no NaN/Inf or Q divergence; at "
                "least 5 Dev validations; last-3 JointR range <= 0.01; "
                "last JointR >= best - 0.005; no new best in the last "
                "validation (improving=false)"
            ),
            "completed_cap": bool(reached_cap),
            "stable_platform": bool(status_payload["stable_platform"]),
            "improving": bool(status_payload["improving"]),
            "numerically_finite": True,
        },
        "convergence_gate_passed": bool(status_payload["converged"]),
        "final_test_decision": (
            "pending_predefined_gate_review"
        ),
        "reward_mode": "jointr_aligned_v2",
        "gamma": GAMMA_V2,
        "reward_jointr_abs_error_max": float(max(
            row["jointr_abs_error"] for row in history["train"]
        )),
        "reward_jointr_abs_error_mean": float(np.mean([
            row["jointr_abs_error"] for row in history["train"]
        ])),
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
        early_stop_triggered=bool(status_payload["early_stop_triggered"]),
        converged=bool(status_payload["converged"]),
        training_status=status_payload["status"],
        stop_reason=stop_reason,
        training_finished_utc=utc_now(),
        training_result=str(RESULT_PATH),
    )
    writer.close()
    print(json.dumps({
        "episodes_trained": final_episode,
        "best_validation_joint": best_joint,
        "best_validation_episode": training_result["best_validation_episode"],
        "stop_reason": stop_reason,
        "status": status_payload["status"],
        "converged": status_payload["converged"],
        "reward_abs_error_max": training_result["reward_jointr_abs_error_max"],
        "best_checkpoint": str(BEST_MODEL),
        "last_full_checkpoint": str(LAST_FULL),
        "elapsed_s": training_result["elapsed_s"],
    }, indent=2), flush=True)
    print("STAGE2K_JOINTR_V2_TRAIN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
