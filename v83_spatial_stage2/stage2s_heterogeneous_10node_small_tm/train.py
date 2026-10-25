"""Stage-2s trainer: heterogeneous 10-node closed-loop DQN / PPO.

One run = (algorithm, seed), 4000 training episodes, Dev-60 every 200.
Unlike Stage-2r there is **no frozen lane cache**: the selected node's frozen
Small or Transformer-M checkpoint produces the navigation action, the UAV
moves, and the next 194-D state / mask is rebuilt from the real environment.

Reward is scheme D (E2E delay only):

    u_t = 1 - e2e_t/tau
    r_t = u_t/H
    r_T += (1 - T/H) * (1 - mean_E2E/tau)
    R_D = 1 - mean_E2E/tau

Usage:
    python v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/train.py \
        --algorithm dqn --seed 101
    python .../train.py --algorithm ppo --seed 202 --resume
    python .../train.py --algorithm dqn --seed 101 --smoke --max-episodes 14
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
)
from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2  # noqa: E402
from v83_spatial_stage2.stage2k_ppo_agent import (  # noqa: E402
    PPOAgent,
    PPOAnomalyError,
    RolloutBuffer,
)
from v83_spatial_stage2.stage2k_state import masked_argmax  # noqa: E402
from v83_spatial_stage2.stage2n_common import (  # noqa: E402
    convergence_status,
)
from v83_spatial_stage2.stage2n_train import (  # noqa: E402
    _compact_validation_row,
    _max_defined,
    _mean_defined,
    save_checkpoint,
    validation_record,
    write_validation_outputs,
)
from v83_spatial_stage2.stage2o.rewards import (  # noqa: E402
    H_MAX,
    SCHEME_DESCRIPTIONS,
    TAU_MS,
    EpisodeRewardController,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (  # noqa: E402
    ACTION_DIM,
    ALGORITHMS,
    CHECKPOINT_SELECTION,
    CONVERGENCE_CRITERION,
    DEV_EPISODES,
    FORMAL_SEEDS,
    FORMAL_VALIDATION,
    GAMMA,
    MAX_EPISODES,
    MEC_CHECKPOINT_SHA256,
    MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S,
    MEC_MODELS,
    MEC_NOMINAL_COMPUTE_FLOPS_PER_S,
    REWARD_DESCRIPTION,
    REWARD_SCHEME,
    RUNS_ROOT,
    STAGE,
    STAGE_TITLE,
    STATE_DIM,
    STATE_MODE,
    artifact_hashes,
    condition_label,
    load_bundle,
    run_dir as default_run_dir,
    source_hashes,
    training_args,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.progress import (  # noqa: E402
    cumulative_stats,
    write_round,
    write_training_csv,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.runtime import (  # noqa: E402
    HeterogeneousEvaluator,
    add_scheme_d,
    latency_stats,
    load_stage2s_runtime,
    make_scene_cache,
    runner_for,
)

REWARD_TOLERANCE = 1e-8
VALIDATION_IMPROVEMENT_MS = 1e-6
LOGS_ROOT = RUNS_ROOT / "logs"
TRAINING_MODE = "full_closed_loop_heterogeneous"
BASELINE_RANDOM_SEED = 20261025


def _git_state():
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


def protocol_payload(bundle, algorithm, seed, device, threads, argv,
                     run_dir, tensorboard_dir, max_episodes,
                     validation_every):
    deployment = bundle["deployment"]
    return {
        "stage": STAGE,
        "stage_name": STAGE_TITLE,
        "algorithm": algorithm,
        "training_mode": TRAINING_MODE,
        "lane_cache_used": False,
        "reward_scheme": REWARD_SCHEME,
        "seed": int(seed),
        "created_utc": utc_now(),
        "parent_stages": ["Stage-2n", "Stage-2o", "Stage-2p", "Stage-2r"],
        "reward": {
            "scheme": REWARD_SCHEME,
            "tau_ms": TAU_MS,
            "horizon": H_MAX,
            "gamma": GAMMA,
            "formula": SCHEME_DESCRIPTIONS[REWARD_SCHEME],
            "description": REWARD_DESCRIPTION,
            "navigation_reward": "removed",
            "success_bonus": "none",
            "stop_bonus": "none",
            "navigation_quality_reward": "none",
            "test_used_for_reward": False,
        },
        "artifact_hashes": artifact_hashes(bundle),
        "task_counts": {
            "train": len(bundle["train_specs"]),
            "dev": len(bundle["dev_specs"]),
            "test": "sealed_not_read",
        },
        "deployment": {
            "deployment_id": deployment["deployment_id"],
            "deployment_sha256": deployment["deployment_sha256"],
            "num_nodes": len(deployment["nodes"]),
            "deployment_kind": "heterogeneous",
            "model_counts": deployment["mec"]["model_counts"],
            "model_types": deployment["mec"]["model_types"],
            "assignment_rule": deployment["assignment"]["assignment_rule"],
            "spatial_balance": deployment["assignment"]["balance"],
            "nominal_compute_flops_per_s": float(
                MEC_NOMINAL_COMPUTE_FLOPS_PER_S
            ),
            "effective_inference_compute_flops_per_s": float(
                MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S
            ),
            "checkpoint_sha256": dict(MEC_CHECKPOINT_SHA256),
            "nodes": [
                {
                    "node_id": node["node_id"],
                    "source_index": node["source_index"],
                    "x": node["x"],
                    "y": node["y"],
                    "model_type": node["model_type"],
                }
                for node in deployment["nodes"]
            ],
        },
        "state": {
            "mode": STATE_MODE,
            "dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "availability_mask_separate": True,
            "visual_inputs_to_association_agent": False,
        },
        "training": {
            "max_episodes": int(max_episodes),
            "validation_every": int(validation_every),
            "validation_tasks": len(bundle["dev_specs"]),
            "validation_split": "dev",
            "seed": int(seed),
            "same_order_for_all_runs": True,
            "test_used": False,
            "closed_loop": True,
        },
        "convergence": {
            "criterion": CONVERGENCE_CRITERION,
            "checkpoint_selection": CHECKPOINT_SELECTION,
        },
        "source_hashes": source_hashes(),
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
            "git": _git_state(),
            "pid": int(__import__("os").getpid()),
            "device": str(device),
            "torch_threads": int(threads),
        },
        "final_test_read": False,
    }


def install_protocol(run_dir, protocol, argv):
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "protocol.json", protocol)
    atomic_write_json(run_dir / "commands.json", {
        "argv": [str(value) for value in argv],
        "created_utc": utc_now(),
    })


def checkpoint_extra(bundle, algorithm, seed, episode, best_e2e,
                     best_episode, record, convergence=None):
    payload = {
        "stage": STAGE,
        "algorithm": algorithm,
        "seed": int(seed),
        "scheme": REWARD_SCHEME,
        "reward_formula": SCHEME_DESCRIPTIONS[REWARD_SCHEME],
        "episode": int(episode),
        "best_dev_e2e_ms": float(best_e2e),
        "best_dev_episode": int(best_episode),
        "deployment_sha256": bundle["deployment"]["deployment_sha256"],
        "mec_checkpoint_sha256": dict(MEC_CHECKPOINT_SHA256),
        "training_mode": TRAINING_MODE,
        "validation": record,
        "final_test_read": False,
    }
    if convergence is not None:
        payload["convergence"] = convergence
    return payload


def _episode_diagnostics(result, controller):
    if len(controller.e2e_ms) != int(result["steps"]):
        raise RuntimeError(
            f"reward steps {len(controller.e2e_ms)} != "
            f"environment steps {result['steps']}"
        )
    expected_mean = float(np.mean(np.asarray(
        controller.e2e_ms, dtype=np.float64
    )))
    if abs(expected_mean - float(result["mean_episode_e2e_ms"])) > 1e-7:
        raise RuntimeError(
            "reward E2E mean does not match environment mean: "
            f"{expected_mean} vs {result['mean_episode_e2e_ms']}"
        )
    reward, diagnostics = controller.finalize(
        result["mean_episode_e2e_ms"], result["steps"]
    )
    if diagnostics["abs_error"] > REWARD_TOLERANCE:
        raise RuntimeError(
            f"scheme D reward identity failed: {diagnostics['return']} vs "
            f"{diagnostics['expected_return']}"
        )
    if abs(
        float(result["reward_D_reconstruction_error"])
    ) > REWARD_TOLERANCE:
        raise RuntimeError(
            "scheme D runner reconstruction failed: "
            f"{result['reward_D_reconstruction_error']}"
        )
    return reward, diagnostics


def _episode_row(bundle, algorithm, seed, spec, result, diagnostics, extra):
    stats = latency_stats(result["step_records"])
    row = {
        "stage": STAGE,
        "algorithm": algorithm,
        "seed": int(seed),
        "condition": condition_label(algorithm, seed),
        "scheme": REWARD_SCHEME,
        "episode_uid": spec["episode_uid"],
        "band": spec["band"],
        "task_seed": int(spec["task_seed"]),
        "steps": int(result["steps"]),
        "episode_return": float(diagnostics["return"]),
        "mean_episode_e2e_ms": float(result["mean_episode_e2e_ms"]),
        "reward_D": float(1.0 - result["mean_episode_e2e_ms"] / 100.0),
        "reward_expected": float(diagnostics["expected_return"]),
        "reward_abs_error": float(diagnostics["abs_error"]),
        "terminal_reward": float(diagnostics["terminal_reward"]),
        "terminal_correction": float(diagnostics["terminal_correction"]),
        "success": bool(result["success"]),
        "osr": bool(result["osr"]),
        "spl": float(result["spl"]),
        "final_nDTW": float(result["final_ndtw"]),
        "latency_return_stage2n": float(result["latency_return"]),
        "delay_violation_rate": float(result["delay_violation_rate"]),
        "reward_mode": "stage2s_scheme_D_heterogeneous",
        "tau_ms": float(TAU_MS),
        "horizon": int(H_MAX),
        "small_steps": int(stats["small_steps"]),
        "transformer_m_steps": int(stats["transformer_m_steps"]),
        "small_ratio": float(stats["small_ratio"]),
        "transformer_m_ratio": float(stats["transformer_m_ratio"]),
        "mean_inference_ms": float(stats["mean_inference_ms"]),
        "mean_communication_ms": float(stats["mean_communication_ms"]),
        "mean_extraction_ms": float(stats["mean_extraction_ms"]),
        "mean_inference_small_ms": float(
            stats["mean_inference_small_ms"]
        ) if np.isfinite(stats["mean_inference_small_ms"]) else None,
        "mean_inference_transformer_m_ms": float(
            stats["mean_inference_transformer_m_ms"]
        ) if np.isfinite(
            stats["mean_inference_transformer_m_ms"]
        ) else None,
        "mean_e2e_small_ms": float(
            stats["mean_e2e_small_ms"]
        ) if np.isfinite(stats["mean_e2e_small_ms"]) else None,
        "mean_e2e_transformer_m_ms": float(
            stats["mean_e2e_transformer_m_ms"]
        ) if np.isfinite(stats["mean_e2e_transformer_m_ms"]) else None,
        "node_sha256": hashlib.sha256(
            ",".join(
                str(int(record["node_index"]))
                for record in result["step_records"]
            ).encode("ascii")
        ).hexdigest(),
        "model_sha256": hashlib.sha256(
            ",".join(
                str(record["model_type"])
                for record in result["step_records"]
            ).encode("ascii")
        ).hexdigest(),
    }
    row.update(extra)
    return row


def _log_train_row(writer, row, episode):
    for tag, key in (
        ("train/episode_return", "episode_return"),
        ("train/reward_D", "reward_D"),
        ("train/mean_e2e_ms", "mean_episode_e2e_ms"),
        ("train/success", "success"),
        ("train/final_nDTW", "final_nDTW"),
        ("train/steps", "steps"),
        ("train/small_ratio", "small_ratio"),
        ("train/transformer_m_ratio", "transformer_m_ratio"),
        ("train/mean_inference_ms", "mean_inference_ms"),
        ("train/mean_communication_ms", "mean_communication_ms"),
        ("train/epsilon", "epsilon"),
        ("train/q_mean", "q_mean"),
        ("train/q_max", "q_max"),
        ("train/target_mean", "target_mean"),
        ("train/td_error_abs_mean", "td_error_abs_mean"),
        ("train/loss", "loss"),
        ("train/policy_entropy", "policy_entropy"),
        ("train/value_mean", "value_mean"),
        ("train/value_abs_max", "value_abs_max"),
        ("train/total_steps", "total_steps"),
        ("train/terminal_reward", "terminal_reward"),
        ("train/terminal_correction", "terminal_correction"),
    ):
        value = row.get(key)
        if value is not None:
            writer.add_scalar(tag, float(value), episode)


def _log_windows(writer, reward_window, e2e_window, success_window,
                 ndtw_window, episode):
    if len(reward_window) == reward_window.maxlen:
        writer.add_scalar(
            "train/episode_return_MA100", float(np.mean(reward_window)),
            episode,
        )
        writer.add_scalar(
            "train/mean_e2e_ms_MA100", float(np.mean(e2e_window)), episode
        )
        writer.add_scalar(
            "train/success_MA100", float(np.mean(success_window)), episode
        )
        writer.add_scalar(
            "train/final_nDTW_MA100", float(np.mean(ndtw_window)), episode
        )
        writer.add_scalar(
            "train/mean_e2e_ms_rolling_std", float(np.std(
                e2e_window, ddof=0
            )), episode,
        )


def _validation_extra(rows, recent, algorithm):
    total_steps = sum(int(row["steps"]) for row in rows)
    small_steps = sum(int(row.get("small_steps", 0)) for row in rows)
    transformer_steps = sum(
        int(row.get("transformer_m_steps", 0)) for row in rows
    )

    def _mean(key):
        values = [
            float(row[key]) for row in rows
            if row.get(key) is not None
            and np.isfinite(float(row[key]))
        ]
        return float(np.mean(values)) if values else None

    extra = {
        "dev_small_steps": int(small_steps),
        "dev_transformer_m_steps": int(transformer_steps),
        "dev_small_ratio": (
            float(small_steps) / total_steps if total_steps else 0.0
        ),
        "dev_transformer_m_ratio": (
            float(transformer_steps) / total_steps if total_steps else 0.0
        ),
        "dev_mean_inference_ms": _mean("mean_inference_ms"),
        "dev_mean_communication_ms": _mean("mean_communication_ms"),
        "dev_mean_inference_small_ms": _mean("mean_inference_small_ms"),
        "dev_mean_inference_transformer_m_ms": _mean(
            "mean_inference_transformer_m_ms"
        ),
        "dev_mean_communication_small_ms": _mean(
            "mean_communication_small_ms"
        ),
        "dev_mean_communication_transformer_m_ms": _mean(
            "mean_communication_transformer_m_ms"
        ),
        "dev_mean_e2e_small_ms": _mean("mean_e2e_small_ms"),
        "dev_mean_e2e_transformer_m_ms": _mean(
            "mean_e2e_transformer_m_ms"
        ),
    }
    if algorithm == "dqn":
        extra.update({
            "q_mean": _mean_defined([row["q_mean"] for row in recent]),
            "q_max": _max_defined([row["q_max"] for row in recent]),
            "target_mean": _mean_defined([
                row["target_mean"] for row in recent
            ]),
            "td_error_abs_mean": _mean_defined([
                row["td_error_abs_mean"] for row in recent
            ]),
            "train_loss_mean": _mean_defined([
                row["loss"] for row in recent
            ]),
            "epsilon": (
                float(recent[-1].get("epsilon"))
                if recent and recent[-1].get("epsilon") is not None else None
            ),
        })
    else:
        extra.update({
            "policy_entropy": _mean_defined([
                row["policy_entropy"] for row in recent
            ]),
            "value_mean": _mean_defined([
                row["value_mean"] for row in recent
            ]),
            "value_abs_max": _max_defined([
                row["value_abs_max"] for row in recent
            ]),
            "update_policy_loss": _mean_defined([
                row.get("update_policy_loss") for row in recent
            ]),
            "update_value_loss": _mean_defined([
                row.get("update_value_loss") for row in recent
            ]),
            "update_entropy": _mean_defined([
                row.get("update_entropy") for row in recent
            ]),
            "approx_kl": _mean_defined([
                row.get("approx_kl") for row in recent
            ]),
            "clip_fraction": _mean_defined([
                row.get("clip_fraction") for row in recent
            ]),
            "explained_variance": _mean_defined([
                row.get("explained_variance") for row in recent
            ]),
        })
    return extra


def _run_validation(bundle, agent, algorithm, seed, evaluator, episode,
                    history, writer, run_dir, best_state):
    if algorithm == "dqn":
        rows = evaluator.evaluate_dqn(agent)
    else:
        rows = evaluator.evaluate_ppo(agent)
    add_scheme_d(rows)
    recent = history["train"][-FORMAL_VALIDATION:]
    extra = _validation_extra(rows, recent, algorithm)
    record = validation_record(
        episode, rows, history["train"], extra=extra
    )
    record["RewardD"] = float(1.0 - record["E2E_ms"] / 100.0)
    record["algorithm"] = algorithm
    record["seed"] = int(seed)
    record["condition"] = condition_label(algorithm, seed)
    history["validation"].append(record)
    compact = []
    for row in rows:
        item = _compact_validation_row(row)
        for key in (
            "small_steps", "transformer_m_steps", "small_ratio",
            "transformer_m_ratio", "mean_inference_ms",
            "mean_communication_ms", "mean_inference_small_ms",
            "mean_inference_transformer_m_ms", "mean_e2e_small_ms",
            "mean_e2e_transformer_m_ms", "reward_D",
        ):
            item[key] = row.get(key)
        item["validation_episode"] = int(episode)
        item["algorithm"] = algorithm
        item["seed"] = int(seed)
        item["condition"] = condition_label(algorithm, seed)
        item["scheme"] = REWARD_SCHEME
        compact.append(item)
    history.setdefault("validation_episodes", []).extend(compact)
    for tag, key in (
        ("validation/E2E_ms", "E2E_ms"),
        ("validation/RewardD", "RewardD"),
        ("validation/P95_E2E_ms", "P95_E2E_ms"),
        ("validation/LatencyReturn", "LatencyReturn"),
        ("validation/SR", "SR"),
        ("validation/OSR", "OSR"),
        ("validation/SPL", "SPL"),
        ("validation/nDTW", "nDTW"),
        ("validation/delay_violation_rate", "delay_violation_rate"),
        ("validation/mean_steps", "mean_steps"),
        ("validation/small_ratio", "dev_small_ratio"),
        ("validation/transformer_m_ratio", "dev_transformer_m_ratio"),
        ("validation/mean_inference_ms", "dev_mean_inference_ms"),
        ("validation/train_return_ma100", "train_return_ma100"),
        ("validation/train_e2e_ma100", "train_e2e_ma100"),
    ):
        value = record.get(key)
        if value is not None and np.isfinite(float(value)):
            writer.add_scalar(tag, float(value), episode)
    for tag, key in extra.items():
        value = record.get(key)
        if value is not None and np.isfinite(float(value)):
            writer.add_scalar(f"validation/{tag}", float(value), episode)
    improved = (
        record["E2E_ms"] < best_state["best_e2e"] - VALIDATION_IMPROVEMENT_MS
    )
    tie_better = False
    if (
        best_state["best_record"] is not None
        and abs(record["E2E_ms"] - best_state["best_e2e"])
        <= VALIDATION_IMPROVEMENT_MS
    ):
        tie_better = (
            record["SR"]
            > float(best_state["best_record"].get("SR", 0.0)) + 1e-12
        )
    if best_state["best_e2e"] == float("inf") or improved or tie_better:
        best_state["best_e2e"] = float(record["E2E_ms"])
        best_state["best_episode"] = int(episode)
        best_state["best_record"] = record
        improved = True
        save_checkpoint(
            agent, algorithm, run_dir / "checkpoints" / "best.pt",
            full=False,
            extra=checkpoint_extra(
                bundle, algorithm, seed, episode, best_state["best_e2e"],
                best_state["best_episode"], record,
            ),
        )
    best_state["last_validation"] = record
    return record, improved


def _print_validation(algorithm, seed, record, improved):
    print(
        f"[{algorithm.upper()}|s{seed}] 验证 Ep{record['episode']}: "
        f"E2E={record['E2E_ms']:.2f}ms "
        f"RewardD={record['RewardD']:.4f} "
        f"SR={100*record['SR']:.2f}% OSR={100*record['OSR']:.2f}% "
        f"SPL={record['SPL']:.4f} nDTW={record['nDTW']:.4f} "
        f"P95={record['P95_E2E_ms']:.1f}ms "
        f"Small={100*record.get('dev_small_ratio', 0.0):.1f}% "
        f"best={'是' if improved else '否'}",
        flush=True,
    )


def _progress(path, args, bundle, algorithm, episode, history, best_state):
    payload = read_json(path) or {
        "stage": STAGE,
        "created_utc": utc_now(),
    }
    stats = cumulative_stats(history.get("train", []))
    payload.update({
        "updated_utc": utc_now(),
        "status": "training",
        "stage": STAGE,
        "algorithm": algorithm,
        "scheme": REWARD_SCHEME,
        "seed": int(args.seed),
        "training_mode": TRAINING_MODE,
        "lane_cache_used": False,
        "current_episode": int(episode),
        "max_episodes": int(args.max_episodes),
        "best_dev_e2e_ms": (
            None if best_state["best_e2e"] == float("inf")
            else float(best_state["best_e2e"])
        ),
        "best_dev_episode": int(best_state["best_episode"]),
        "best_checkpoint": str(path.parent / "checkpoints" / "best.pt")
        if best_state["best_e2e"] != float("inf") else None,
        "last_validation": best_state.get("last_validation"),
        "validations": len(history["validation"]),
        "elapsed_seconds": float(history.get("elapsed_seconds", 0.0)),
        "training_counts": stats,
        "deployment_sha256": bundle["deployment"]["deployment_sha256"],
        "model_counts": bundle["deployment"]["mec"]["model_counts"],
        "effective_inference_compute_flops_per_s": float(
            MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S
        ),
        "final_test_read": False,
    })
    atomic_write_json(path, payload)


def _periodic_save(args, bundle, agent, algorithm, episode, best_state,
                   history, run_dir, history_path, progress_path,
                   loop_started):
    if episode % 10 == 0:
        history["elapsed_seconds"] = (
            float(history.get("elapsed_seconds", 0.0))
            + (time.monotonic() - loop_started)
        )
        loop_started = time.monotonic()
        atomic_write_json(history_path, history)
        _progress(progress_path, args, bundle, algorithm, episode, history,
                  best_state)
        row = history["train"][-1]
        print(
            f"[{algorithm.upper()}|s{args.seed}] "
            f"Ep{episode}/{args.max_episodes} "
            f"R={row['episode_return']:.4f} "
            f"E2E={row['mean_episode_e2e_ms']:.1f}ms "
            f"SR={int(row['success'])} nDTW={row['final_nDTW']:.4f} "
            f"Small={100*row['small_ratio']:.1f}% band={row['band']}",
            flush=True,
        )
    if episode % max(1, int(args.checkpoint_every)) == 0 or episode % 50 == 0:
        extra = checkpoint_extra(
            bundle, algorithm, args.seed, episode,
            best_state["best_e2e"], best_state["best_episode"],
            best_state.get("last_validation"),
        )
        save_checkpoint(
            agent, algorithm, run_dir / "checkpoints" / "last_full.pt",
            full=True, extra=extra,
        )
        save_checkpoint(
            agent, algorithm, run_dir / "checkpoints" / "last.pt",
            full=False, extra=extra,
        )
    return loop_started


def _reconcile_resume(history, payload, best_state):
    checkpoint_episode = int(payload.get("episode", 0) or 0)
    if checkpoint_episode <= 0:
        return history
    train = list(history.get("train", []))
    if len(train) > checkpoint_episode:
        history["train"] = train[:checkpoint_episode]
    history["validation"] = [
        row for row in history.get("validation", [])
        if int(row.get("episode", 0)) <= checkpoint_episode
    ]
    history["validation_episodes"] = [
        row for row in history.get("validation_episodes", [])
        if int(row.get("validation_episode", 0)) <= checkpoint_episode
    ]
    if best_state is not None:
        best_state.update({
            "best_e2e": float("inf"),
            "best_episode": 0,
            "best_record": None,
            "last_validation": None,
        })
        for record in history["validation"]:
            if float(record["E2E_ms"]) < best_state["best_e2e"]:
                best_state["best_e2e"] = float(record["E2E_ms"])
                best_state["best_episode"] = int(record["episode"])
                best_state["best_record"] = record
        if history["validation"]:
            best_state["last_validation"] = history["validation"][-1]
    return history


def _abort(run_dir, bundle, agent, algorithm, seed, episode, exc, history,
           best_state):
    atomic_write_json(run_dir / "abort.json", {
        "episode": int(episode),
        "error": str(exc),
        "diagnostics": getattr(exc, "diagnostics", {}),
        "created_utc": utc_now(),
    })
    try:
        save_checkpoint(
            agent, algorithm, run_dir / "checkpoints" / "last_full.pt",
            full=True,
            extra=checkpoint_extra(
                bundle, algorithm, seed, episode,
                best_state["best_e2e"], best_state["best_episode"],
                best_state.get("last_validation"),
            ),
        )
    finally:
        raise exc


def _make_runner(bundle, spec, encoder, backends, scene_cache, device,
                 latency_model):
    return runner_for(
        bundle, spec, encoder, backends, scene_cache, device, latency_model
    )


def train_dqn(args, bundle, device, encoder, backends, latency_model,
              scene_cache, evaluator, writer, run_dir, history,
              start_episode, best_state, loop_started):
    agent = DQNAgentV2(
        device, seed=int(args.seed), gamma=GAMMA,
        state_dim=STATE_DIM, state_mode=STATE_MODE,
        action_dim=ACTION_DIM,
    )
    last_full = run_dir / "checkpoints" / "last_full.pt"
    if args.resume:
        if not last_full.is_file():
            raise SystemExit(f"--resume but {last_full} is missing")
        payload = agent.load_full(last_full)
        if payload.get("stage") != STAGE:
            raise SystemExit("Resume checkpoint is not a Stage-2s run")
        if payload.get("algorithm") != "dqn":
            raise SystemExit("Resume checkpoint algorithm mismatch")
        if int(payload.get("seed", -1)) != int(args.seed):
            raise SystemExit("Resume checkpoint seed mismatch")
        if payload.get("deployment_sha256") != bundle["deployment"][
            "deployment_sha256"
        ]:
            raise SystemExit("Resume checkpoint deployment mismatch")
        _reconcile_resume(history, payload, best_state)
        start_episode = len(history["train"])
        print(
            f"[DQN|s{args.seed}] resumed at episode {start_episode}, "
            f"steps={agent.total_steps}",
            flush=True,
        )
    reward_window = deque(
        [float(row["episode_return"]) for row in history["train"][-99:]],
        maxlen=100,
    )
    e2e_window = deque(
        [float(row["mean_episode_e2e_ms"])
         for row in history["train"][-99:]], maxlen=100,
    )
    success_window = deque(
        [float(row["success"]) for row in history["train"][-99:]], maxlen=100
    )
    ndtw_window = deque(
        [float(row["final_nDTW"]) for row in history["train"][-99:]],
        maxlen=100,
    )
    for episode_index in range(start_episode, int(args.max_episodes)):
        episode = episode_index + 1
        spec = bundle["train_specs"][episode_index]
        runner = _make_runner(
            bundle, spec, encoder, backends, scene_cache, device,
            latency_model,
        )
        state = runner.step_data["state"]
        mask = runner.step_data["mask"]
        done = False
        controller = EpisodeRewardController(REWARD_SCHEME)
        q_means, q_maxs = [], []
        losses, grad_norms, target_means, td_errors = [], [], [], []
        result = None
        diagnostics = None
        while not done:
            q_values = agent.q_values(state)
            available = q_values[np.asarray(mask, bool)]
            q_means.append(float(available.mean()))
            q_maxs.append(float(available.max()))
            action = agent.act_from_q(
                q_values, mask, epsilon=agent.epsilon()
            )
            transition = runner.apply(action)
            done = bool(transition["done"])
            controller.step_reward(transition["e2e_ms"])
            if done:
                result = runner.finalize("dqn")
                reward, diagnostics = _episode_diagnostics(
                    result, controller
                )
            else:
                reward = controller.rewards[-1]
            agent.buffer.add(
                state, action, reward,
                transition["next_state"], transition["done"],
                transition["next_mask"],
            )
            if agent.total_steps >= 2_000:
                stats = agent.train_step()
                if stats is not None:
                    losses.append(stats["loss"])
                    grad_norms.append(stats["grad_norm"])
                    target_means.append(stats["target_mean"])
                    td_errors.append(stats["td_error_abs_mean"])
            agent.total_steps += 1
            state = transition["next_state"]
            mask = transition["next_mask"]
        finite = [
            result["episode_return"], result["mean_episode_e2e_ms"],
            result["final_ndtw"], result["latency_return"],
            diagnostics["return"], diagnostics["expected_return"],
        ]
        if not np.all(np.isfinite(finite)):
            raise RuntimeError(f"non-finite metrics at episode {episode}")
        if not np.all(np.isfinite(q_means + q_maxs)):
            raise RuntimeError(f"non-finite Q at episode {episode}")
        if abs(float(np.mean(q_means))) > 1e6 or abs(float(
            np.max(q_maxs)
        )) > 1e6:
            raise RuntimeError(f"Q divergence at episode {episode}")
        row = _episode_row(bundle, "dqn", args.seed, spec, result,
                           diagnostics, {
            "episode": episode,
            "loss": float(np.mean(losses)) if losses else None,
            "epsilon": float(agent.epsilon()),
            "q_mean": float(np.mean(q_means)) if q_means else None,
            "q_max": float(np.max(q_maxs)) if q_maxs else None,
            "target_mean": float(np.mean(target_means))
            if target_means else None,
            "td_error_abs_mean": float(np.mean(td_errors))
            if td_errors else None,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else None,
            "buffer": int(len(agent.buffer)),
            "total_steps": int(agent.total_steps),
        })
        history["train"].append(row)
        _log_train_row(writer, row, episode)
        reward_window.append(row["episode_return"])
        e2e_window.append(row["mean_episode_e2e_ms"])
        success_window.append(float(row["success"]))
        ndtw_window.append(row["final_nDTW"])
        _log_windows(writer, reward_window, e2e_window, success_window,
                     ndtw_window, episode)
        loop_started = _periodic_save(
            args, bundle, agent, "dqn", episode, best_state, history,
            run_dir, run_dir / "training_history.json",
            run_dir / "progress.json", loop_started,
        )
        if (
            episode % int(args.validation_every) == 0
            or episode == int(args.max_episodes)
        ):
            record, improved = _run_validation(
                bundle, agent, "dqn", args.seed, evaluator, episode,
                history, writer, run_dir, best_state,
            )
            _print_validation("dqn", args.seed, record, improved)
            write_validation_outputs(run_dir, history)
            write_training_csv(run_dir, history["train"])
            write_round(
                run_dir, history, record, "dqn", int(args.seed),
                history.get("elapsed_seconds", 0.0),
            )
            atomic_write_json(run_dir / "training_history.json", history)
            extra = checkpoint_extra(
                bundle, "dqn", args.seed, episode,
                best_state["best_e2e"], best_state["best_episode"], record,
            )
            save_checkpoint(agent, "dqn", last_full, full=True, extra=extra)
            save_checkpoint(
                agent, "dqn", run_dir / "checkpoints" / "last.pt",
                full=False, extra=extra,
            )
            _progress(
                run_dir / "progress.json", args, bundle, "dqn", episode,
                history, best_state,
            )
    return agent


def train_ppo(args, bundle, device, encoder, backends, latency_model,
              scene_cache, evaluator, writer, run_dir, history,
              start_episode, best_state, loop_started):
    agent = PPOAgent(
        device, seed=int(args.seed), state_dim=STATE_DIM,
        state_mode=STATE_MODE, action_dim=ACTION_DIM, gamma=GAMMA,
    )
    last_full = run_dir / "checkpoints" / "last_full.pt"
    if args.resume:
        if not last_full.is_file():
            raise SystemExit(f"--resume but {last_full} is missing")
        payload = agent.load_full(last_full)
        if payload.get("stage") != STAGE:
            raise SystemExit("Resume checkpoint is not a Stage-2s run")
        if payload.get("algorithm") != "ppo":
            raise SystemExit("Resume checkpoint algorithm mismatch")
        if int(payload.get("seed", -1)) != int(args.seed):
            raise SystemExit("Resume checkpoint seed mismatch")
        if payload.get("deployment_sha256") != bundle["deployment"][
            "deployment_sha256"
        ]:
            raise SystemExit("Resume checkpoint deployment mismatch")
        _reconcile_resume(history, payload, best_state)
        start_episode = len(history["train"])
        print(
            f"[PPO|s{args.seed}] resumed at episode {start_episode}, "
            f"steps={agent.total_steps}",
            flush=True,
        )
    rollout = RolloutBuffer()
    reward_window = deque(
        [float(row["episode_return"]) for row in history["train"][-99:]],
        maxlen=100,
    )
    e2e_window = deque(
        [float(row["mean_episode_e2e_ms"])
         for row in history["train"][-99:]], maxlen=100,
    )
    success_window = deque(
        [float(row["success"]) for row in history["train"][-99:]], maxlen=100
    )
    ndtw_window = deque(
        [float(row["final_nDTW"]) for row in history["train"][-99:]],
        maxlen=100,
    )
    for episode_index in range(start_episode, int(args.max_episodes)):
        episode = episode_index + 1
        spec = bundle["train_specs"][episode_index]
        runner = _make_runner(
            bundle, spec, encoder, backends, scene_cache, device,
            latency_model,
        )
        state = runner.step_data["state"]
        mask = runner.step_data["mask"]
        done = False
        controller = EpisodeRewardController(REWARD_SCHEME)
        entropies, values = [], []
        result = None
        diagnostics = None
        while not done:
            try:
                decision = agent.act(state, mask)
            except PPOAnomalyError as exc:
                _abort(run_dir, bundle, agent, "ppo", args.seed, episode,
                       exc, history, best_state)
            transition = runner.apply(int(decision["action"]))
            done = bool(transition["done"])
            controller.step_reward(transition["e2e_ms"])
            if done:
                result = runner.finalize("ppo")
                reward, diagnostics = _episode_diagnostics(
                    result, controller
                )
            else:
                reward = controller.rewards[-1]
            rollout.add(
                state, mask, int(decision["action"]),
                float(decision["log_prob"]), float(decision["value"]),
                float(reward), bool(transition["done"]),
            )
            entropies.append(float(decision["entropy"]))
            values.append(float(decision["value"]))
            agent.total_steps += 1
            state = transition["next_state"]
            mask = transition["next_mask"]
        finite = [
            result["episode_return"], result["mean_episode_e2e_ms"],
            result["final_ndtw"], result["latency_return"],
            diagnostics["return"], diagnostics["expected_return"],
            float(np.mean(entropies)), float(np.mean(values)),
        ]
        if not np.all(np.isfinite(finite)):
            raise RuntimeError(f"non-finite metrics at episode {episode}")
        if abs(float(np.mean(values))) > 1e3:
            raise RuntimeError(
                f"value divergence at episode {episode}: "
                f"{float(np.mean(values))}"
            )
        update_stats = None
        if len(rollout) >= agent.rollout_steps:
            try:
                update_stats = agent.update(rollout)
            except PPOAnomalyError as exc:
                _abort(run_dir, bundle, agent, "ppo", args.seed, episode,
                       exc, history, best_state)
            rollout.clear()
            if update_stats["approx_kl"] > 0.5:
                raise RuntimeError(
                    f"PPO KL explosion at episode {episode}: "
                    f"{update_stats['approx_kl']}"
                )
            for tag, key in (
                ("train/update_policy_loss", "policy_loss"),
                ("train/update_value_loss", "value_loss"),
                ("train/update_entropy", "entropy"),
                ("train/update_approx_kl", "approx_kl"),
                ("train/update_clip_fraction", "clip_fraction"),
                ("train/update_grad_norm", "grad_norm"),
                ("train/update_explained_variance", "explained_variance"),
            ):
                writer.add_scalar(tag, update_stats[key], episode)
        row = _episode_row(bundle, "ppo", args.seed, spec, result,
                           diagnostics, {
            "episode": episode,
            "policy_entropy": float(np.mean(entropies)),
            "value_mean": float(np.mean(values)),
            "value_abs_max": float(np.max(np.abs(values))),
            "rollout_steps": int(len(rollout)),
            "rollout_updates": int(agent.rollout_updates),
            "policy_updates": int(agent.update_steps),
            "update_policy_loss": float(update_stats["policy_loss"])
            if update_stats else None,
            "update_value_loss": float(update_stats["value_loss"])
            if update_stats else None,
            "update_entropy": float(update_stats["entropy"])
            if update_stats else None,
            "approx_kl": float(update_stats["approx_kl"])
            if update_stats else None,
            "clip_fraction": float(update_stats["clip_fraction"])
            if update_stats else None,
            "explained_variance": float(update_stats["explained_variance"])
            if update_stats else None,
            "total_steps": int(agent.total_steps),
        })
        history["train"].append(row)
        _log_train_row(writer, row, episode)
        reward_window.append(row["episode_return"])
        e2e_window.append(row["mean_episode_e2e_ms"])
        success_window.append(float(row["success"]))
        ndtw_window.append(row["final_nDTW"])
        _log_windows(writer, reward_window, e2e_window, success_window,
                     ndtw_window, episode)
        loop_started = _periodic_save(
            args, bundle, agent, "ppo", episode, best_state, history,
            run_dir, run_dir / "training_history.json",
            run_dir / "progress.json", loop_started,
        )
        if (
            episode % int(args.validation_every) == 0
            or episode == int(args.max_episodes)
        ):
            record, improved = _run_validation(
                bundle, agent, "ppo", args.seed, evaluator, episode,
                history, writer, run_dir, best_state,
            )
            _print_validation("ppo", args.seed, record, improved)
            write_validation_outputs(run_dir, history)
            write_training_csv(run_dir, history["train"])
            write_round(
                run_dir, history, record, "ppo", int(args.seed),
                history.get("elapsed_seconds", 0.0),
            )
            atomic_write_json(run_dir / "training_history.json", history)
            extra = checkpoint_extra(
                bundle, "ppo", args.seed, episode,
                best_state["best_e2e"], best_state["best_episode"], record,
            )
            save_checkpoint(agent, "ppo", last_full, full=True, extra=extra)
            save_checkpoint(
                agent, "ppo", run_dir / "checkpoints" / "last.pt",
                full=False, extra=extra,
            )
            _progress(
                run_dir / "progress.json", args, bundle, "ppo", episode,
                history, best_state,
            )
    return agent


def _write_result(run_dir, bundle, args, history, best_state, agent):
    stats = cumulative_stats(history["train"])
    records = [dict(row) for row in history["validation"]]
    reached_cap = len(history["train"]) >= int(args.max_episodes)
    convergence = convergence_status(records, reached_cap)
    best_record = best_state.get("best_record")
    rows = history["train"]
    total_steps = sum(int(row["steps"]) for row in rows)
    small_steps = sum(int(row.get("small_steps", 0)) for row in rows)
    result = {
        "stage": STAGE,
        "status": "completed",
        "algorithm": args.algorithm,
        "scheme": REWARD_SCHEME,
        "seed": int(args.seed),
        "training_mode": TRAINING_MODE,
        "lane_cache_used": False,
        "episodes": len(history["train"]),
        "elapsed_seconds": float(history.get("elapsed_seconds", 0.0)),
        "training_counts": stats,
        "training_small_steps": int(small_steps),
        "training_transformer_m_steps": int(total_steps - small_steps),
        "training_small_ratio": (
            float(small_steps) / total_steps if total_steps else 0.0
        ),
        "best_dev_e2e_ms": (
            None if best_state["best_e2e"] == float("inf")
            else float(best_state["best_e2e"])
        ),
        "best_dev_episode": int(best_state["best_episode"]),
        "best_checkpoint": str(run_dir / "checkpoints" / "best.pt")
        if best_state["best_e2e"] != float("inf") else None,
        "best_validation": best_record,
        "last_validation": best_state.get("last_validation"),
        "convergence": convergence,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "deployment_sha256": bundle["deployment"]["deployment_sha256"],
        "model_counts": bundle["deployment"]["mec"]["model_counts"],
        "effective_inference_compute_flops_per_s": float(
            MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S
        ),
        "final_test_read": False,
        "completed_utc": utc_now(),
        "total_steps": int(getattr(agent, "total_steps", 0)),
    }
    atomic_write_json(run_dir / "training_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--seed", type=int, default=FORMAL_SEEDS[0])
    parser.add_argument("--max-episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--validation-every", type=int,
                        default=FORMAL_VALIDATION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--dev-episodes", type=int, default=DEV_EPISODES,
        help="Dev tasks used for validation (formal runs must use 60)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    args = parser.parse_args()

    formal = not args.smoke
    if formal:
        if int(args.max_episodes) != MAX_EPISODES:
            raise SystemExit(
                f"formal Stage-2s requires --max-episodes {MAX_EPISODES}"
            )
        if int(args.validation_every) != FORMAL_VALIDATION:
            raise SystemExit(
                "formal Stage-2s requires --validation-every "
                f"{FORMAL_VALIDATION}"
            )
        if int(args.seed) not in FORMAL_SEEDS:
            raise SystemExit(
                f"Stage-2s formal seeds are {FORMAL_SEEDS}, got {args.seed}"
            )
        if int(args.dev_episodes) != DEV_EPISODES:
            raise SystemExit(
                f"formal Stage-2s requires --dev-episodes {DEV_EPISODES}"
            )
    else:
        args.max_episodes = max(1, min(int(args.max_episodes), 60))
        args.validation_every = max(
            1, min(int(args.validation_every), int(args.max_episodes))
        )
        args.dev_episodes = max(
            1, min(int(args.dev_episodes), DEV_EPISODES)
        )

    if args.run_dir is None:
        if formal:
            args.run_dir = default_run_dir(args.algorithm, args.seed)
        else:
            args.run_dir = (
                RUNS_ROOT / "smoke"
                / f"{args.algorithm}_s{args.seed}"
            )
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    tensorboard_dir = Path(
        args.tensorboard_dir or (run_dir / "tensorboard")
    )
    if not tensorboard_dir.is_absolute():
        tensorboard_dir = PROJECT_ROOT / tensorboard_dir
    checkpoint_dir = run_dir / "checkpoints"
    history_path = run_dir / "training_history.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    last_full = checkpoint_dir / "last_full.pt"
    if last_full.is_file() and not args.resume:
        raise SystemExit(
            f"{last_full} exists; use --resume or a fresh run directory"
        )

    torch.set_num_threads(max(1, int(args.threads)))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    bundle = load_bundle()
    eval_bundle = (
        bundle if int(args.dev_episodes) == len(bundle["dev_specs"])
        else dict(bundle, dev_specs=bundle["dev_specs"][:int(args.dev_episodes)])
    )
    protocol = protocol_payload(
        eval_bundle, args.algorithm, args.seed, device, args.threads,
        sys.argv, run_dir, tensorboard_dir, args.max_episodes,
        args.validation_every,
    )
    protocol["training"]["dev_episodes_used"] = int(args.dev_episodes)
    install_protocol(run_dir, protocol, sys.argv)
    print(
        f"[{args.algorithm.upper()}|s{args.seed}] starting "
        f"episodes={args.max_episodes} validation={args.validation_every} "
        f"mode={TRAINING_MODE} "
        f"deployment={bundle['deployment']['deployment_sha256'][:12]} "
        f"models={bundle['deployment']['mec']['model_counts']} "
        f"effective_inference_TFLOPS="
        f"{MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S/1e12:.2f}",
        flush=True,
    )

    device, encoder, backends, latency_model = load_stage2s_runtime(device)
    print(
        f"[{args.algorithm.upper()}|s{args.seed}] loaded "
        f"{backends.loaded_models} "
        f"checkpoints={ {k: v[:12] for k, v in backends.checkpoint_sha256.items()} }",
        flush=True,
    )
    scene_cache = make_scene_cache(
        encoder, device, capacity=max(64, min(1000, args.max_episodes))
    )
    evaluator = HeterogeneousEvaluator(
        eval_bundle, encoder, backends, scene_cache, device, latency_model
    )

    history = read_json(history_path)
    if history is None:
        history = {
            "stage": STAGE,
            "algorithm": args.algorithm,
            "scheme": REWARD_SCHEME,
            "seed": int(args.seed),
            "created_utc": utc_now(),
            "config": {
                "algorithm": args.algorithm,
                "reward_scheme": REWARD_SCHEME,
                "seed": int(args.seed),
                "max_episodes": int(args.max_episodes),
                "validation_every": int(args.validation_every),
                "state_mode": STATE_MODE,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "num_nodes": ACTION_DIM,
                "tau_ms": TAU_MS,
                "horizon": H_MAX,
                "gamma": GAMMA,
                "reward_formula": SCHEME_DESCRIPTIONS[REWARD_SCHEME],
                "convergence_criterion": CONVERGENCE_CRITERION,
                "checkpoint_selection": CHECKPOINT_SELECTION,
                "training_mode": TRAINING_MODE,
                "lane_cache_used": False,
                "deployment_sha256": bundle["deployment"][
                    "deployment_sha256"
                ],
                "mec_checkpoint_sha256": dict(MEC_CHECKPOINT_SHA256),
                "model_counts": bundle["deployment"]["mec"]["model_counts"],
                "nominal_compute_flops_per_s": float(
                    MEC_NOMINAL_COMPUTE_FLOPS_PER_S
                ),
                "effective_inference_compute_flops_per_s": float(
                    MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S
                ),
                "train_tasks_sha256": bundle["train_pool"]["tasks_sha256"],
                "dev_tasks_sha256": bundle["dev_pool"]["tasks_sha256"],
                "dev_episodes_used": int(args.dev_episodes),
                "final_test_read": False,
            },
            "train": [],
            "validation": [],
            "validation_episodes": [],
            "elapsed_seconds": 0.0,
        }
    history["algorithm"] = args.algorithm
    start_episode = len(history["train"])
    if args.resume and start_episode == 0 and last_full.is_file():
        raise SystemExit(
            "resume checkpoint exists but training_history.json is empty"
        )
    best_state = {
        "best_e2e": float("inf"),
        "best_episode": 0,
        "best_record": None,
        "last_validation": None,
    }
    for record in history.get("validation", []):
        if float(record["E2E_ms"]) < best_state["best_e2e"]:
            best_state["best_e2e"] = float(record["E2E_ms"])
            best_state["best_episode"] = int(record["episode"])
            best_state["best_record"] = record
    if history.get("validation"):
        best_state["last_validation"] = history["validation"][-1]
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(
        f"[{args.algorithm.upper()}|s{args.seed}] "
        f"train={len(bundle['train_specs'])} "
        f"dev={len(bundle['dev_specs'])} state_dim={STATE_DIM} "
        f"action_dim={ACTION_DIM} resume_from={start_episode}",
        flush=True,
    )
    loop_started = time.monotonic()
    if args.algorithm == "dqn":
        agent = train_dqn(
            args, bundle, device, encoder, backends, latency_model,
            scene_cache, evaluator, writer, run_dir, history,
            start_episode, best_state, loop_started,
        )
    else:
        agent = train_ppo(
            args, bundle, device, encoder, backends, latency_model,
            scene_cache, evaluator, writer, run_dir, history,
            start_episode, best_state, loop_started,
        )
    history["elapsed_seconds"] = (
        float(history.get("elapsed_seconds", 0.0))
        + (time.monotonic() - loop_started)
    )
    atomic_write_json(history_path, history)
    write_training_csv(run_dir, history["train"])
    result = _write_result(run_dir, bundle, args, history, best_state, agent)
    stats = result["training_counts"]
    print(
        f"[{args.algorithm.upper()}|s{args.seed}] 完成: "
        f"总={stats['cum_total']} 成功={stats['cum_success']} "
        f"失败={stats['cum_failure']} "
        f"成功short/medium/long={stats['cum_success_short']}/"
        f"{stats['cum_success_medium']}/{stats['cum_success_long']} "
        f"失败short/medium/long={stats['cum_failure_short']}/"
        f"{stats['cum_failure_medium']}/{stats['cum_failure_long']} "
        f"Small选择={100*result['training_small_ratio']:.2f}% "
        f"bestDevE2E={result['best_dev_e2e_ms']:.3f}ms "
        f"@Ep{result['best_dev_episode']} "
        f"收敛={result['convergence']['status']}",
        flush=True,
    )
    _progress(
        run_dir / "progress.json", args, bundle, args.algorithm,
        args.max_episodes, history, best_state,
    )
    payload = read_json(run_dir / "progress.json") or {}
    payload.update({"status": "completed", "updated_utc": utc_now()})
    atomic_write_json(run_dir / "progress.json", payload)
    print("STAGE2S_TRAIN_COMPLETE", flush=True)
    return result


if __name__ == "__main__":
    main()
