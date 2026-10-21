"""Stage-2l standard PPO on the fixed-city navigation environment.

The policy is the existing Stage-2k PPO actor-critic (384-D Pure-MEC v2
state -> 256 -> 256 -> 20 logits + value, gamma=1.0, GAE(lambda=0.95),
clipped PPO).  It uses exactly the same 3500-task order, 60 Dev tasks,
config topology assignment, frozen MECs and JointR-aligned reward as the
Stage-2l DQN.

Protocol:
  * config A/B/C, fully described in stage2l_fixed_city.py;
  * 3500 episodes max, validation every 200 on 60 independent Dev tasks;
  * no reward shaping, no success bonus / failure penalty;
  * no early stopping; only numerical/program errors abort;
  * best and last checkpoints are kept; Final Test is never read.

Usage:
    python v83_spatial_stage2/stage2l_ppo_train.py --config A
"""

from __future__ import annotations

import argparse
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
    atomic_write_json,
    read_json,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_ppo_agent import (
    CLIP_EPS,
    ENTROPY_COEF,
    GAE_LAMBDA,
    GAMMA,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MINIBATCH_SIZE,
    ROLLOUT_STEPS,
    SEED as PPO_SEED,
    TARGET_KL,
    UPDATE_EPOCHS,
    VALUE_COEF,
    PPOAgent,
    PPOAnomalyError,
    RolloutBuffer,
)
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (
    state_layout_pure_mec_v2,
)
from v83_spatial_stage2.stage2k_runtime import load_gab_config, load_runtime
from v83_spatial_stage2.stage2k_v2_common import load_capability_prior
from v83_spatial_stage2.stage2l_fixed_city import (
    STAGE2L_ROOT,
    TRAIN_EPISODES,
)
from v83_spatial_stage2.stage2l_protocol import (
    CAPABILITY_PRIOR_PATH,
    DevEvaluator,
    REWARD_DESCRIPTION,
    STATE_DIM,
    STATE_MODE,
    VALIDATION_EVERY,
    VALIDATION_IMPROVEMENT,
    classify_status,
    load_bundle,
    protocol_payload,
    resolve_device,
    set_threads,
    topology_for,
    validation_record,
    write_protocol_files,
)
from v83_spatial_stage2.stage2l_runtime import (
    FixedCityEpisodeRunner,
    FixedCitySceneCache,
)


STAGE_NAME = "stage2l_ppo"
MAX_EPISODES = TRAIN_EPISODES


def update_progress(path, **fields):
    payload = read_json(path) or {
        "stage": STAGE_NAME,
        "created_utc": utc_now(),
        "final_test_read": False,
    }
    payload["updated_utc"] = utc_now()
    payload.update(fields)
    atomic_write_json(path, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("A", "B", "C"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--validation-every", type=int, default=VALIDATION_EVERY)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--minibatch-size", type=int, default=MINIBATCH_SIZE)
    parser.add_argument("--update-epochs", type=int, default=UPDATE_EPOCHS)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    if not args.smoke:
        if int(args.max_episodes) != MAX_EPISODES:
            raise SystemExit(
                f"Formal Stage-2l PPO requires --max-episodes {MAX_EPISODES}"
            )
        if int(args.validation_every) != VALIDATION_EVERY:
            raise SystemExit(
                "Formal Stage-2l PPO requires --validation-every 200"
            )
        if (
            int(args.rollout_steps) != ROLLOUT_STEPS
            or int(args.minibatch_size) != MINIBATCH_SIZE
            or int(args.update_epochs) != UPDATE_EPOCHS
        ):
            raise SystemExit(
                "Formal Stage-2l PPO requires the standard update "
                f"({ROLLOUT_STEPS}/{MINIBATCH_SIZE}/{UPDATE_EPOCHS})"
            )
    else:
        args.max_episodes = min(int(args.max_episodes), 20)
        args.validation_every = max(1, min(int(args.validation_every),
                                           args.max_episodes))
        args.rollout_steps = max(8, min(int(args.rollout_steps), 256))
        args.minibatch_size = min(int(args.minibatch_size),
                                  max(8, args.rollout_steps))
        args.update_epochs = max(1, min(int(args.update_epochs), 3))

    run_dir = args.run_dir or (STAGE2L_ROOT / args.config / "ppo")
    tensorboard_dir = args.tensorboard_dir or (run_dir / "tensorboard")
    checkpoint_dir = run_dir / "checkpoints"
    progress_path = run_dir / "progress.json"
    history_path = run_dir / "training_history.json"
    result_path = run_dir / "training_result.json"
    diagnostics_path = run_dir / "diagnostics.json"
    protocol_path = run_dir / "protocol.json"
    validation_csv = run_dir / "validation.csv"
    last_full = checkpoint_dir / "last_full.pt"
    last_model = checkpoint_dir / "last.pt"
    best_model = checkpoint_dir / "best.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    if last_full.is_file() and not args.resume:
        raise SystemExit(
            f"{last_full} exists; use --resume or a fresh run directory"
        )

    threads = set_threads(args.threads)
    device = resolve_device(args.device)
    bundle = load_bundle(args.config)
    protocol = protocol_payload(
        bundle, "ppo", device, threads, sys.argv, run_dir, tensorboard_dir
    )
    write_protocol_files(run_dir, protocol, sys.argv)
    prior_payload, quality = load_capability_prior(CAPABILITY_PRIOR_PATH)

    device, encoder, backends = load_runtime(str(device))
    gab_config = load_gab_config()
    scene_cache = FixedCitySceneCache(encoder, device, max_entries=512)
    evaluator = DevEvaluator(
        bundle, encoder, backends, quality, gab_config, device,
        scene_cache=scene_cache,
    )

    agent = PPOAgent(
        device,
        seed=PPO_SEED,
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

    history = {
        "train": [],
        "validation": [],
        "config": {
            **agent.config(),
            "stage": STAGE_NAME,
            "config": args.config,
            "config_description": bundle["manifest"]["description"],
            "protocol": "stage2l-fixed-city-v1",
            "max_episodes": int(args.max_episodes),
            "validation_every": int(args.validation_every),
            "validation_source": "60 independent Dev tasks (20/20/20)",
            "validation_tasks": [item["episode_uid"]
                                 for item in bundle["dev_specs"]],
            "training_order_sha256": protocol["training_order_sha256"],
            "state_layout": state_layout_pure_mec_v2(),
            "state_dim": STATE_DIM,
            "state_mode": STATE_MODE,
            "reward": REWARD_DESCRIPTION,
            "map_seed": protocol["map_seed"],
            "train_topology_roles": protocol["train_topology_roles"],
            "dev_topology_roles": protocol["dev_topology_roles"],
            "topology_seeds": protocol["topology_seeds"],
            "topology_scenarios": protocol["topology_scenarios"],
            "capability_prior": protocol["capability_prior"],
            "frozen_mecs": protocol["frozen_mecs"],
            "early_stop_disabled": True,
            "final_test_read": False,
            "test_pool_read": False,
            "smoke": bool(args.smoke),
        },
    }
    start_episode = 0
    best_joint = -1e9
    no_improve = 0
    if args.resume:
        if not last_full.is_file():
            raise SystemExit("--resume requested but last_full.pt is missing")
        payload = agent.load_full(last_full)
        start_episode = int(payload.get("episode", 0))
        best_joint = float(payload.get("best_joint", -1e9))
        no_improve = int(payload.get("no_improve", 0))
        if history_path.is_file():
            history = read_json(history_path)
        if len(history["train"]) > start_episode:
            history["train"] = history["train"][:start_episode]
            history["validation"] = [
                row for row in history["validation"]
                if int(row["episode"]) <= start_episode
            ]
        print(
            f"[PPO][{args.config}] 从第 {start_episode} 个 Episode 续训，"
            f"best={best_joint:.6f}",
            flush=True,
        )
    if start_episode >= int(args.max_episodes):
        raise SystemExit("Checkpoint already reaches the episode cap")

    reward_window = deque(
        (float(row["episode_return"]) for row in history["train"][-99:]),
        maxlen=100,
    )
    success_window = deque(
        (float(row["success"]) for row in history["train"][-99:]),
        maxlen=100,
    )
    e2e_window = deque(
        (float(row["mean_episode_e2e_ms"])
         for row in history["train"][-99:]),
        maxlen=100,
    )
    ndtw_window = deque(
        (float(row["final_nDTW"]) for row in history["train"][-99:]),
        maxlen=100,
    )

    writer = SummaryWriter(str(tensorboard_dir))
    rollout = RolloutBuffer()
    update_stats = []
    started = time.time()
    last_validation = None
    last_update = None
    status_payload = classify_status(
        [row["joint_reward"] for row in history["validation"]],
        False,
    )
    update_progress(
        progress_path,
        status="training",
        algorithm="ppo",
        config=args.config,
        current_episode=int(start_episode),
        best_checkpoint=str(best_model) if best_model.is_file() else None,
        best_dev_joint=float(best_joint) if best_joint > -1e8 else None,
        consecutive_no_improve=int(no_improve),
        last_validation=last_validation,
        early_stop_triggered=False,
        converged=False,
        training_config=history["config"],
        protocol=str(protocol_path),
        final_test="pending",
        final_test_read=False,
    )

    def abort_with_diagnostics(exc, episode_number, phase, extra=None):
        details = {
            "version": "stage2l-ppo-diagnostics-v1",
            "created_utc": utc_now(),
            "config": args.config,
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
        atomic_write_json(diagnostics_path, details)
        agent.save_full(last_full, extra={
            "episode": int(episode_number),
            "best_joint": best_joint,
            "no_improve": no_improve,
            "last_validation": last_validation,
            "aborted": str(exc),
        })
        update_progress(
            progress_path,
            status="numerical_aborted",
            current_episode=int(episode_number),
            best_checkpoint=str(best_model) if best_model.is_file() else None,
            best_dev_joint=float(best_joint) if best_joint > -1e8 else None,
            last_validation=last_validation,
            diagnostics=str(diagnostics_path),
            abort_reason=str(exc),
            training_finished_utc=utc_now(),
        )
        print(
            f"[PPO][{args.config}][ABORT] episode {episode_number} "
            f"{phase}: {exc}",
            flush=True,
        )
        raise SystemExit(2)

    for episode_index in range(start_episode, int(args.max_episodes)):
        episode_number = episode_index + 1
        spec = bundle["train_specs"][episode_index]
        topology, topology_role = topology_for(bundle, "train", spec)
        runner = FixedCityEpisodeRunner(
            spec=spec,
            topology=topology,
            encoder=encoder,
            backends=backends,
            projector=None,
            quality_prior=quality,
            gab_config=gab_config,
            scene=scene_cache.get(spec),
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
        result = runner.finalize("ppo")
        error = float(result["jointr_abs_error"])
        if error > 1e-6:
            agent.save_full(last_full, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
                "aborted": "reward/jointR mismatch",
            })
            raise RuntimeError(
                f"Reward/JointR mismatch at episode {episode_number}: "
                f"{error}"
            )
        finite_values = [
            result["episode_return"], result["episode_jointR"],
            result["mean_episode_e2e_ms"], result["final_ndtw"],
        ]
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
            "config": args.config,
            "scenario": spec["scenario"],
            "topology_role": topology_role,
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
            writer.add_scalar("train/update_policy_loss",
                              stats["policy_loss"], episode_number)
            writer.add_scalar("train/update_value_loss",
                              stats["value_loss"], episode_number)
            writer.add_scalar("train/update_entropy",
                              stats["entropy"], episode_number)
            writer.add_scalar("train/update_approx_kl",
                              stats["approx_kl"], episode_number)
            writer.add_scalar("train/update_clip_fraction",
                              stats["clip_fraction"], episode_number)
            writer.add_scalar("train/update_grad_norm",
                              stats["grad_norm"], episode_number)
            writer.add_scalar("train/update_explained_variance",
                              stats["explained_variance"], episode_number)
            if stats["approx_kl"] > 0.5:
                abort_with_diagnostics(
                    PPOAnomalyError(
                        "Approximate KL above stability threshold",
                        dict(stats),
                    ),
                    episode_number, "update",
                )
        history["train"].append(row)
        for tag, key in (
            ("train/episode_return", "episode_return"),
            ("train/episode_jointR", "episode_jointR"),
            ("train/navigation_reward", "navigation_reward"),
            ("train/latency_penalty", "latency_penalty"),
            ("train/mean_episode_e2e_ms", "mean_episode_e2e_ms"),
            ("train/final_nDTW", "final_nDTW"),
            ("train/steps", "steps"),
            ("train/policy_entropy", "policy_entropy"),
            ("train/value_mean", "value_mean"),
            ("train/total_steps", "total_steps"),
        ):
            writer.add_scalar(tag, row[key], episode_number)
        writer.add_scalar("train/success", float(row["success"]),
                          episode_number)
        reward_window.append(row["episode_return"])
        success_window.append(float(row["success"]))
        e2e_window.append(row["mean_episode_e2e_ms"])
        ndtw_window.append(row["final_nDTW"])
        if len(reward_window) == reward_window.maxlen:
            writer.add_scalar("train/episode_return_ma100",
                              float(np.mean(reward_window)), episode_number)
            writer.add_scalar("train/success_ma100",
                              float(np.mean(success_window)), episode_number)
            writer.add_scalar("train/e2e_ma100",
                              float(np.mean(e2e_window)), episode_number)
            writer.add_scalar("train/ndtw_ma100",
                              float(np.mean(ndtw_window)), episode_number)
        if episode_number % 10 == 0:
            atomic_write_json(history_path, history)
            update_progress(
                progress_path,
                status="training",
                current_episode=episode_number,
                best_checkpoint=str(best_model) if best_model.is_file() else None,
                best_dev_joint=float(best_joint) if best_joint > -1e8 else None,
                consecutive_no_improve=int(no_improve),
                last_validation=last_validation,
                early_stop_triggered=False,
                converged=False,
                last_update=last_update,
            )
            print(
                f"[PPO][{args.config}] Ep{episode_number}/{args.max_episodes} "
                f"return={row['episode_return']:.4f} "
                f"JointR={row['episode_jointR']:.4f} "
                f"SR={int(row['success'])} "
                f"nDTW={row['final_nDTW']:.4f} "
                f"E2E={row['mean_episode_e2e_ms']:.1f}ms "
                f"ent={row['policy_entropy']:.3f} "
                f"V={row['value_mean']:.3f} "
                f"kl={(last_update['approx_kl'] if last_update else 0.0):.4f} "
                f"band={row['band']}",
                flush=True,
            )
        if (
            episode_number % max(1, int(args.checkpoint_every)) == 0
            or episode_number % 50 == 0
        ):
            agent.save_full(last_full, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
                "last_validation": last_validation,
            })
            agent.save_model(last_model, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
                "last_validation": last_validation,
            })

        if (
            episode_number % int(args.validation_every) == 0
            or episode_number == int(args.max_episodes)
        ):
            validation_rows = evaluator.evaluate_ppo(agent)
            recent = history["train"][-int(args.validation_every):]
            recent_entropy = [float(item["policy_entropy"]) for item in recent]
            recent_value = [float(item["value_mean"]) for item in recent]
            recent_updates = update_stats[:]
            update_stats = []
            record = validation_record(
                episode_number,
                validation_rows,
                recent,
                extra={
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
                },
            )
            history["validation"].append(record)
            last_validation = record
            for tag, key in (
                ("validation/JointR", "joint_reward"),
                ("validation/SR", "SR"),
                ("validation/OSR", "OSR"),
                ("validation/SPL", "SPL"),
                ("validation/nDTW", "nDTW"),
                ("validation/E2E_ms", "E2E_ms"),
                ("validation/P95_ms", "P95_ms"),
                ("validation/mean_steps", "mean_steps"),
                ("validation/policy_entropy", "policy_entropy"),
                ("validation/value_loss", "update_value_loss"),
                ("validation/approx_kl", "approx_kl"),
                ("validation/clip_fraction", "clip_fraction"),
                ("validation/explained_variance", "explained_variance"),
            ):
                writer.add_scalar(tag, record[key], episode_number)
            for name, count in record["model_usage"].items():
                writer.add_scalar(
                    f"validation/model_usage/{name}", count, episode_number
                )
            improved = (
                record["joint_reward"] > best_joint + VALIDATION_IMPROVEMENT
            )
            if improved:
                best_joint = record["joint_reward"]
                no_improve = 0
                agent.save_model(best_model, extra={
                    "episode": episode_number,
                    "best_joint": best_joint,
                    "no_improve": no_improve,
                    "validation": record,
                    "protocol": str(protocol_path),
                })
            else:
                no_improve += 1
            agent.save_full(last_full, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
                "last_validation": record,
            })
            agent.save_model(last_model, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
                "last_validation": record,
            })
            atomic_write_json(history_path, history)
            fields = [
                "episode", "joint_reward", "SR", "OSR", "SPL", "nDTW",
                "E2E_ms", "P95_ms", "delay_violation_rate",
                "mean_steps", "episode_return_jointR_abs_error",
                "timestamp",
            ]
            write_csv(
                validation_csv,
                [
                    {key: row.get(key) for key in fields}
                    for row in history["validation"]
                ],
                fieldnames=fields,
            )
            update_progress(
                progress_path,
                status="training",
                current_episode=episode_number,
                best_checkpoint=str(best_model),
                best_dev_joint=float(best_joint),
                consecutive_no_improve=int(no_improve),
                last_validation=record,
                early_stop_triggered=False,
                converged=False,
            )
            print(
                f"[PPO][{args.config}] 验证 Ep{episode_number}: "
                f"JointR={record['joint_reward']:.4f} "
                f"SR={100*record['SR']:.2f}% "
                f"nDTW={record['nDTW']:.4f} "
                f"E2E={record['E2E_ms']:.1f}ms "
                f"SPL={record['SPL']:.4f} "
                f"best={'是' if improved else '否'}",
                flush=True,
            )

    atomic_write_json(history_path, history)
    reached_cap = len(history["train"]) >= int(args.max_episodes)
    status_payload = classify_status(
        [row["joint_reward"] for row in history["validation"]],
        reached_cap,
    )
    result = {
        "stage": STAGE_NAME,
        "config": args.config,
        "algorithm": "ppo",
        "created_utc": utc_now(),
        "elapsed_seconds": float(time.time() - started),
        "episodes": len(history["train"]),
        "max_episodes": int(args.max_episodes),
        "best_dev_joint": float(best_joint),
        "best_checkpoint": str(best_model),
        "last_checkpoint": str(last_model),
        "last_full_checkpoint": str(last_full),
        "final_validation": last_validation,
        "status": status_payload,
        "training_order_sha256": protocol["training_order_sha256"],
        "final_test_read": False,
        "test_pool_read": False,
    }
    atomic_write_json(result_path, result)
    update_progress(
        progress_path,
        status=status_payload["status"],
        current_episode=len(history["train"]),
        best_checkpoint=str(best_model),
        best_dev_joint=float(best_joint),
        consecutive_no_improve=int(no_improve),
        last_validation=last_validation,
        converged=bool(status_payload["converged"]),
        stable_platform=bool(status_payload["stable_platform"]),
        early_stop_triggered=False,
        training_finished_utc=utc_now(),
        result=str(result_path),
    )
    writer.close()
    print(
        f"[PPO][{args.config}] 训练结束: status={status_payload['status']} "
        f"converged={status_payload['converged']} "
        f"best={best_joint:.4f} episodes={len(history['train'])}",
        flush=True,
    )
    print("STAGE2L_PPO_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
