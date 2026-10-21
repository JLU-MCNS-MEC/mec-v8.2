"""Stage-2l standard DQN on the fixed-city navigation environment.

This is the same standard MLP DQN as Stage-2k v2 (gamma=1.0, replay, target
network, epsilon schedule 500k steps), trained on the new fixed-city task
pools.  The five Stage-2j T5 MECs are frozen; only the association policy is
learned.

Protocol:
  * config A/B/C, fully described in stage2l_fixed_city.py;
  * 3500 episodes max, validation every 200 on 60 independent Dev tasks;
  * reward is the JointR-aligned environment reward, no extra shaping;
  * no early stopping; only numerical/program errors abort;
  * best and last checkpoints are kept, TensorBoard logs all requested
    quantities; Final Test is never read.

Usage:
    python v83_spatial_stage2/stage2l_dqn_train.py --config A
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
from v83_spatial_stage2.stage2k_dqn import SEED
from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2
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


STAGE_NAME = "stage2l_dqn"
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
    parser.add_argument(
        "--checkpoint-every", type=int, default=100,
        help="Save the full replay-buffer checkpoint every N episodes.",
    )
    parser.add_argument(
        "--capability-prior", type=Path, default=CAPABILITY_PRIOR_PATH
    )
    args = parser.parse_args()

    if not args.smoke:
        if int(args.max_episodes) != MAX_EPISODES:
            raise SystemExit(
                f"Formal Stage-2l DQN requires --max-episodes {MAX_EPISODES}"
            )
        if int(args.validation_every) != VALIDATION_EVERY:
            raise SystemExit(
                "Formal Stage-2l DQN requires --validation-every 200"
            )
    else:
        args.max_episodes = min(int(args.max_episodes), 20)
        args.validation_every = max(1, min(int(args.validation_every),
                                           args.max_episodes))

    run_dir = args.run_dir or (STAGE2L_ROOT / args.config / "dqn")
    tensorboard_dir = args.tensorboard_dir or (run_dir / "tensorboard")
    checkpoint_dir = run_dir / "checkpoints"
    progress_path = run_dir / "progress.json"
    history_path = run_dir / "training_history.json"
    result_path = run_dir / "training_result.json"
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
        bundle, "dqn", device, threads, sys.argv, run_dir, tensorboard_dir
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

    agent = DQNAgentV2(
        device,
        seed=SEED,
        state_dim=STATE_DIM,
        state_mode=STATE_MODE,
        epsilon_decay_steps=500_000,
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
            f"[DQN][{args.config}] 从第 {start_episode} 个 Episode 续训，"
            f"best={best_joint:.6f}",
            flush=True,
        )
    if start_episode >= int(args.max_episodes):
        raise SystemExit("Checkpoint already reaches the episode cap")

    return_window = deque(
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
    started = time.time()
    last_validation = None
    status_payload = classify_status(
        [row["joint_reward"] for row in history["validation"]],
        False,
    )
    update_progress(
        progress_path,
        status="training",
        algorithm="dqn",
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
            if agent.total_steps >= 2_000:
                stats = agent.train_step()
                if stats is not None:
                    losses.append(stats["loss"])
                    grad_norms.append(stats["grad_norm"])
            agent.total_steps += 1
            state = transition["next_state"]
            mask = transition["next_mask"]
            done = bool(transition["done"])
        result = runner.finalize("dqn")
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
            agent.save_full(last_full, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
                "aborted": "non-finite episode metric",
            })
            raise RuntimeError(f"Non-finite episode at {episode_number}")
        if not np.all(np.isfinite(q_means + q_maxs)):
            raise RuntimeError(f"Non-finite Q values at {episode_number}")
        if abs(float(np.mean(q_means))) > 1e6 or abs(float(
            np.max(q_maxs)
        )) > 1e6:
            agent.save_full(last_full, extra={
                "episode": episode_number,
                "best_joint": best_joint,
                "no_improve": no_improve,
                "aborted": "Q divergence",
            })
            raise RuntimeError(f"Q-value divergence at {episode_number}")

        row = {
            "episode": episode_number,
            "episode_uid": spec["episode_uid"],
            "config": args.config,
            "scenario": spec["scenario"],
            "topology_role": topology_role,
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
        writer.add_scalar("train/episode_jointR", row["episode_jointR"],
                          episode_number)
        writer.add_scalar("train/navigation_reward",
                          row["navigation_reward"], episode_number)
        writer.add_scalar("train/latency_penalty",
                          row["latency_penalty"], episode_number)
        writer.add_scalar("train/mean_episode_e2e_ms",
                          row["mean_episode_e2e_ms"], episode_number)
        writer.add_scalar("train/final_nDTW", row["final_nDTW"],
                          episode_number)
        writer.add_scalar("train/success", float(row["success"]),
                          episode_number)
        writer.add_scalar("train/steps", row["steps"], episode_number)
        writer.add_scalar("train/loss", row["loss"], episode_number)
        writer.add_scalar("train/grad_norm", row["grad_norm"], episode_number)
        writer.add_scalar("train/epsilon", row["epsilon"], episode_number)
        writer.add_scalar("train/q_mean", row["q_mean"], episode_number)
        writer.add_scalar("train/q_max", row["q_max"], episode_number)
        writer.add_scalar("train/jointr_abs_error", row["jointr_abs_error"],
                          episode_number)
        writer.add_scalar("train/total_steps", row["total_steps"],
                          episode_number)
        return_window.append(row["episode_return"])
        success_window.append(float(row["success"]))
        e2e_window.append(row["mean_episode_e2e_ms"])
        ndtw_window.append(row["final_nDTW"])
        if len(return_window) == return_window.maxlen:
            writer.add_scalar("train/episode_return_ma100",
                              float(np.mean(return_window)), episode_number)
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
            )
            print(
                f"[DQN][{args.config}] Ep{episode_number}/{args.max_episodes} "
                f"return={row['episode_return']:.4f} "
                f"JointR={row['episode_jointR']:.4f} "
                f"SR={int(row['success'])} "
                f"nDTW={row['final_nDTW']:.4f} "
                f"E2E={row['mean_episode_e2e_ms']:.1f}ms "
                f"loss={row['loss']:.4f} eps={row['epsilon']:.3f} "
                f"Q={row['q_mean']:.2f}/{row['q_max']:.2f} "
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
            validation_rows = evaluator.evaluate_dqn(agent)
            record = validation_record(
                episode_number,
                validation_rows,
                history["train"][-int(args.validation_every):],
                extra={
                    "train_loss_mean": float(np.mean([
                        item["loss"] for item in
                        history["train"][-int(args.validation_every):]
                    ])),
                    "train_q_mean": float(np.mean([
                        item["q_mean"] for item in
                        history["train"][-int(args.validation_every):]
                    ])),
                    "train_q_max": float(np.max([
                        item["q_max"] for item in
                        history["train"][-int(args.validation_every):]
                    ])),
                },
            )
            history["validation"].append(record)
            last_validation = record
            writer.add_scalar("validation/JointR", record["joint_reward"],
                              episode_number)
            writer.add_scalar("validation/SR", record["SR"], episode_number)
            writer.add_scalar("validation/OSR", record["OSR"], episode_number)
            writer.add_scalar("validation/SPL", record["SPL"], episode_number)
            writer.add_scalar("validation/nDTW", record["nDTW"],
                              episode_number)
            writer.add_scalar("validation/E2E_ms", record["E2E_ms"],
                              episode_number)
            writer.add_scalar("validation/P95_ms", record["P95_ms"],
                              episode_number)
            writer.add_scalar("validation/mean_steps",
                              record["mean_steps"], episode_number)
            writer.add_scalar("validation/delay_violation_rate",
                              record["delay_violation_rate"], episode_number)
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
                f"[DQN][{args.config}] 验证 Ep{episode_number}: "
                f"JointR={record['joint_reward']:.4f} "
                f"SR={100*record['SR']:.2f}% "
                f"nDTW={record['nDTW']:.4f} "
                f"E2E={record['E2E_ms']:.1f}ms "
                f"SPL={record['SPL']:.4f} "
                f"best={'是' if improved else '否'}",
                flush=True,
            )

    # Final bookkeeping -----------------------------------------------------
    atomic_write_json(history_path, history)
    reached_cap = (
        len(history["train"]) >= int(args.max_episodes)
    )
    status_payload = classify_status(
        [row["joint_reward"] for row in history["validation"]],
        reached_cap,
    )
    result = {
        "stage": STAGE_NAME,
        "config": args.config,
        "algorithm": "dqn",
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
        f"[DQN][{args.config}] 训练结束: status={status_payload['status']} "
        f"converged={status_payload['converged']} "
        f"best={best_joint:.4f} episodes={len(history['train'])}",
        flush=True,
    )
    print("STAGE2L_DQN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
