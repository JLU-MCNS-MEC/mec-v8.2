"""Stage-2n DQN / PPO trainer on the latency-only association task.

Both algorithms share exactly the same frozen task order, homogeneous MEC
deployment, 384-D Pure-MEC v2 state, 20-D availability mask, latency-only
reward and Dev-60 evaluation.  The only difference is the learning algorithm.

Formal runs: 4000 episodes, validation every 200 episodes, seeds 0/1/2.
The Test-400 pool is never opened here.

Usage:
    python v83_spatial_stage2/stage2n_train.py --algorithm dqn --seed 0
    python v83_spatial_stage2/stage2n_train.py --algorithm ppo --seed 0
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
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
    sha256_file,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2
from v83_spatial_stage2.stage2k_ppo_agent import (
    PPOAgent,
    PPOAnomalyError,
    RolloutBuffer,
)
from v83_spatial_stage2.stage2k_runtime import load_gab_config
from v83_spatial_stage2.stage2k_state import masked_argmax
from v83_spatial_stage2.stage2k_v2_common import load_capability_prior
from v83_spatial_stage2.stage2n_common import (
    ACTION_DIM,
    AGENT_SEEDS,
    CAPABILITY_PRIOR_PATH,
    CHECKPOINT_SELECTION,
    CONVERGENCE_CRITERION,
    MEC_CHECKPOINT_SHA256,
    PROTOCOL_PATH,
    REWARD_DESCRIPTION,
    RUNS_DIR,
    STAGE,
    STAGE2N_ROOT,
    STATE_DIM,
    STATE_MODE,
    TRAIN_EPISODES,
    VALIDATION_EVERY,
    convergence_status,
    deployment_topology,
    dev_specs,
    latency_return_from_mean,
    load_deployment,
    ordered_train_specs,
    prepare_runner_spec,
    select_best_validation,
    summarize_episode_rows,
    verify_mec_checkpoint,
)
from v83_spatial_stage2.stage2n_lanes import LaneEvaluator, build_lanes
from v83_spatial_stage2.stage2n_runtime import (
    LatencyOnlyEpisodeRunner,
    load_stage2n_runtime,
    make_stage2n_scene_cache,
)

STAGE_NAME = "stage2n"
VALIDATION_IMPROVEMENT_MS = 1e-6
REWARD_TOLERANCE = 1e-8


# ---------------------------------------------------------------------------
# Protocol / provenance
# ---------------------------------------------------------------------------


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


def verify_frozen_protocol(verify_test_file=False):
    """Verify the freeze file.

    Training calls this with ``verify_test_file=False``: it never opens the
    sealed Test-400 file.  Only the final evaluator passes ``True``.
    """
    protocol = read_json(PROTOCOL_PATH)
    if protocol is None:
        raise FileNotFoundError(
            f"{PROTOCOL_PATH} is missing; run stage2n_build.py first"
        )
    if protocol.get("final_test_read") is not False:
        raise ValueError("Stage-2n protocol claims a Final Test read")
    if protocol.get("test_sealed") is not True:
        raise ValueError("Stage-2n Test pool is not sealed")
    splits = ("train", "dev", "test") if verify_test_file else (
        "train", "dev"
    )
    for split in splits:
        info = protocol["tasks"][split]
        path = Path(info["path"])
        if sha256_file(path) != info["sha256"]:
            raise ValueError(f"{split} task pool SHA-256 mismatch")
    order = protocol["training_order"]
    if sha256_file(Path(order["path"])) != order["sha256"]:
        raise ValueError("training order SHA-256 mismatch")
    dep = protocol["deployment"]
    if sha256_file(Path(dep["path"])) != dep["sha256"]:
        raise ValueError("deployment SHA-256 mismatch")
    if protocol["mec"]["checkpoint_sha256"] != MEC_CHECKPOINT_SHA256:
        raise ValueError("MEC checkpoint SHA-256 mismatch in protocol")
    actual = sha256_file(Path(protocol["mec"]["checkpoint"]))
    if actual != MEC_CHECKPOINT_SHA256:
        raise ValueError("MEC checkpoint file SHA-256 mismatch")
    return protocol


def load_bundle():
    protocol = verify_frozen_protocol()
    train = [prepare_runner_spec(item) for item in ordered_train_specs()]
    dev = [prepare_runner_spec(item) for item in dev_specs()]
    deployment = load_deployment()
    topology = deployment_topology(deployment)
    prior_payload, prior = load_capability_prior(CAPABILITY_PRIOR_PATH)
    if prior_payload.get("final_test_read") is not False:
        raise ValueError("capability prior claims a Final Test read")
    mec = verify_mec_checkpoint()
    gab_config = load_gab_config()
    if len(train) != TRAIN_EPISODES:
        raise ValueError(f"train tasks {len(train)} != {TRAIN_EPISODES}")
    if len(dev) != 60:
        raise ValueError(f"Dev tasks {len(dev)} != 60")
    if set(item["episode_uid"] for item in train) & set(
        item["episode_uid"] for item in dev
    ):
        raise ValueError("Train/Dev task IDs overlap")
    return {
        "protocol": protocol,
        "train_specs": train,
        "dev_specs": dev,
        "deployment": deployment,
        "topology": topology,
        "quality_prior": prior,
        "prior_payload": prior_payload,
        "mec_record": mec,
        "gab_config": gab_config,
    }


def protocol_payload(bundle, algorithm, seed, device, threads, argv,
                     run_dir, tensorboard_dir, max_episodes,
                     validation_every):
    deploy = bundle["deployment"]
    return {
        "stage": STAGE,
        "stage_name": "Stage-2n: Fixed-City Latency-Only MEC Association",
        "algorithm": algorithm,
        "seed": int(seed),
        "created_utc": utc_now(),
        "frozen_protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "artifact_hashes": {
            "train_tasks": bundle["protocol"]["tasks"]["train"]["sha256"],
            "dev_tasks": bundle["protocol"]["tasks"]["dev"]["sha256"],
            "test_tasks": bundle["protocol"]["tasks"]["test"]["sha256"],
            "training_order": bundle["protocol"]["training_order"]["sha256"],
            "deployment": bundle["protocol"]["deployment"]["sha256"],
            "mec_checkpoint": MEC_CHECKPOINT_SHA256,
            "capability_prior": sha256_file(CAPABILITY_PRIOR_PATH),
        },
        "task_counts": {
            "train": len(bundle["train_specs"]),
            "dev": len(bundle["dev_specs"]),
            "test": 400,
        },
        "deployment": {
            "deployment_id": deploy["deployment_id"],
            "deployment_sha256": deploy["deployment_sha256"],
            "num_nodes": len(deploy["nodes"]),
            "model": deploy["mec"]["model"],
            "homogeneous": True,
            "compute_capacity_flops_per_s": deploy["mec"][
                "compute_capacity_flops_per_s"
            ],
            "bandwidth_hz": deploy["mec"]["bandwidth_hz"],
            "coverage_radius_m": deploy["mec"]["coverage_radius_m"],
            "positions_unique": deploy["homogeneity_checks"][
                "positions_unique"
            ],
            "source_topology": deploy["source_topology"],
            "source_topology_sha256": deploy["source_topology_sha256"],
        },
        "mec": bundle["mec_record"],
        "state": {
            "mode": STATE_MODE,
            "dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "availability_mask_separate": True,
            "visual_inputs_to_association_agent": False,
            "visual_inputs_to_mec": True,
            "forbidden": [
                "RGB", "goal image", "visual PCA", "visual history",
                "target position", "oracle action", "success label",
            ],
        },
        "reward": {
            "mode": "latency_only_v1",
            "per_step": REWARD_DESCRIPTION,
            "episode": "LatencyReturn = -0.25 * mean_E2E_ms / 100",
            "navigation_reward": "removed",
            "gamma": 1.0,
        },
        "training": {
            "max_episodes": int(max_episodes),
            "validation_every": int(validation_every),
            "validation_tasks": len(bundle["dev_specs"]),
            "validation_split": "dev",
            "same_order_for_dqn_ppo": True,
            "test_used": False,
        },
        "convergence": {
            "criterion": CONVERGENCE_CRITERION,
            "checkpoint_selection": CHECKPOINT_SELECTION,
        },
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
        "argv": list(argv),
        "created_utc": utc_now(),
    })


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _compact_validation_row(row):
    keys = (
        "episode_uid", "strategy", "band", "task_seed", "success", "osr",
        "spl", "ndtw", "steps", "mean_e2e_ms", "p95_e2e_ms",
        "delay_violation_rate", "latency_return",
        "latency_return_closed_form", "latency_return_abs_error",
    )
    return {key: row.get(key) for key in keys}


def validation_record(episode, rows, train_window, extra=None):
    overall = summarize_episode_rows(rows).get("overall", {})
    by_band = summarize_episode_rows(rows, key="band")
    abs_errors = [float(row["latency_return_abs_error"]) for row in rows]
    if max(abs_errors) > REWARD_TOLERANCE:
        raise RuntimeError(
            f"LatencyReturn consistency failed: {max(abs_errors)}"
        )
    record = {
        "episode": int(episode),
        "episodes": len(rows),
        "E2E_ms": float(overall["E2E_ms"]),
        "E2E_std_ms": float(overall["E2E_std_ms"]),
        "P95_E2E_ms": float(overall["P95_E2E_ms"]),
        "LatencyReturn": float(overall["LatencyReturn"]),
        "SR": float(overall["SR"]),
        "OSR": float(overall["OSR"]),
        "SPL": float(overall["SPL"]),
        "nDTW": float(overall["nDTW"]),
        "delay_violation_rate": float(overall["delay_violation_rate"]),
        "mean_steps": float(overall["mean_steps"]),
        "by_band": by_band,
        "latency_return_abs_error_max": float(max(abs_errors)),
        "timestamp": utc_now(),
    }
    if train_window:
        recent = list(train_window)
        returns = [float(row["episode_return"]) for row in recent]
        e2e = [float(row["mean_episode_e2e_ms"]) for row in recent]
        record.update({
            "train_return_mean": float(np.mean(returns)),
            "train_return_ma100": float(np.mean(returns[-100:])),
            "train_e2e_mean": float(np.mean(e2e)),
            "train_e2e_ma100": float(np.mean(e2e[-100:])),
            "train_success_mean": float(np.mean([
                float(row["success"]) for row in recent
            ])),
            "train_steps_mean": float(np.mean([
                int(row["steps"]) for row in recent
            ])),
        })
    if extra:
        record.update(extra)
    return record


def validation_csv_fields():
    return [
        "episode", "E2E_ms", "E2E_std_ms", "P95_E2E_ms", "LatencyReturn",
        "SR", "OSR", "SPL", "nDTW", "delay_violation_rate", "mean_steps",
        "latency_return_abs_error_max", "train_return_ma100",
        "train_e2e_ma100", "timestamp",
    ]


def write_validation_outputs(run_dir, history):
    rows = history["validation"]
    fields = validation_csv_fields()
    write_csv(
        run_dir / "validation.csv",
        [{key: row.get(key) for key in fields} for row in rows],
        fieldnames=fields,
    )
    raw = history.get("validation_episodes", [])
    if raw:
        raw_fields = list(raw[0].keys())
        write_csv(
            run_dir / "validation_episodes.csv", raw, fieldnames=raw_fields
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(agent, algorithm, path, full, extra):
    if full:
        agent.save_full(path, extra=extra)
    else:
        agent.save_model(path, extra=extra)


def checkpoint_extra(algorithm, episode, best_e2e, best_episode, record,
                     convergence=None):
    payload = {
        "stage": STAGE_NAME,
        "algorithm": algorithm,
        "episode": int(episode),
        "best_dev_e2e_ms": float(best_e2e),
        "best_dev_episode": int(best_episode),
        "latency_return_mode": "latency_only_v1",
        "stage2n_reward_mode": "latency_only_v1",
        "validation": record,
        "final_test_read": False,
    }
    if convergence is not None:
        payload["convergence"] = convergence
    return payload


# ---------------------------------------------------------------------------
# Main training loops
# ---------------------------------------------------------------------------


def _runner(bundle, spec, encoder, backends, scene_cache, device):
    return LatencyOnlyEpisodeRunner(
        spec=spec,
        topology=bundle["topology"],
        encoder=encoder,
        backends=backends,
        projector=None,
        quality_prior=bundle["quality_prior"],
        gab_config=bundle["gab_config"],
        scene_cache=scene_cache,
        device=device,
    )


def _episode_row(algorithm, spec, result, extra):
    row = {
        "episode": int(extra.pop("episode")),
        "algorithm": algorithm,
        "episode_uid": spec["episode_uid"],
        "band": spec["band"],
        "task_seed": int(spec["task_seed"]),
        "steps": int(result["steps"]),
        "episode_return": float(result["episode_return"]),
        "latency_return": float(result["latency_return"]),
        "latency_return_abs_error": float(
            result["latency_return_abs_error"]
        ),
        "mean_episode_e2e_ms": float(result["mean_episode_e2e_ms"]),
        "success": bool(result["success"]),
        "osr": bool(result["osr"]),
        "spl": float(result["spl"]),
        "final_nDTW": float(result["final_ndtw"]),
        "reward_mode": "latency_only_v1",
    }
    row.update(extra)
    return row


def _mean_defined(values):
    finite = [
        float(value) for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def _max_defined(values):
    finite = [
        float(value) for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return float(np.max(finite)) if finite else None


def _run_validation(agent, algorithm, evaluator, episode, history,
                    writer, run_dir, progress_path, best_state):
    if algorithm == "dqn":
        rows = evaluator.evaluate_dqn(agent)
    else:
        rows = evaluator.evaluate_ppo(agent)
    recent = history["train"][-VALIDATION_EVERY:]
    if algorithm == "dqn":
        extra = {
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
        }
    else:
        extra = {
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
        }
    record = validation_record(
        episode, rows, history["train"], extra=extra
    )
    history["validation"].append(record)
    compact = [_compact_validation_row(row) for row in rows]
    for row in compact:
        row["validation_episode"] = int(episode)
        row["algorithm"] = algorithm
    history.setdefault("validation_episodes", []).extend(compact)
    last_validation = record
    for tag, key in (
        ("validation/E2E_ms", "E2E_ms"),
        ("validation/P95_E2E_ms", "P95_E2E_ms"),
        ("validation/LatencyReturn", "LatencyReturn"),
        ("validation/SR", "SR"),
        ("validation/OSR", "OSR"),
        ("validation/SPL", "SPL"),
        ("validation/nDTW", "nDTW"),
        ("validation/delay_violation_rate", "delay_violation_rate"),
        ("validation/mean_steps", "mean_steps"),
        ("validation/train_return_ma100", "train_return_ma100"),
        ("validation/train_e2e_ma100", "train_e2e_ma100"),
    ):
        writer.add_scalar(tag, record[key], episode)
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
                algorithm, episode, best_state["best_e2e"],
                best_state["best_episode"], record,
            ),
        )
    best_state["last_validation"] = record
    return record, improved


def _print_validation(algorithm, record, improved):
    print(
        f"[{algorithm.upper()}] 验证 Ep{record['episode']}: "
        f"E2E={record['E2E_ms']:.2f}ms "
        f"LatencyReturn={record['LatencyReturn']:.4f} "
        f"SR={100*record['SR']:.2f}% "
        f"SPL={record['SPL']:.4f} "
        f"nDTW={record['nDTW']:.4f} "
        f"P95={record['P95_E2E_ms']:.1f}ms "
        f"best={'是' if improved else '否'}",
        flush=True,
    )


def train_dqn(args, bundle, device, encoder, backends, scene_cache,
              evaluator, writer, run_dir, history, start_episode,
              best_state):
    from v83_spatial_stage2.stage2k_dqn import SEED as DQN_SEED

    agent = DQNAgentV2(
        device, seed=int(args.seed), gamma=1.0,
        state_dim=STATE_DIM, state_mode=STATE_MODE,
    )
    last_full = run_dir / "checkpoints" / "last_full.pt"
    last_model = run_dir / "checkpoints" / "last.pt"
    if args.resume:
        if not last_full.is_file():
            raise SystemExit(f"--resume but {last_full} is missing")
        payload = agent.load_full(last_full)
        if payload.get("stage2n_reward_mode") != "latency_only_v1":
            raise SystemExit("Resume checkpoint is not a Stage-2n run")
        print(
            f"[DQN] resumed at episode {start_episode}, "
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
        runner = _runner(bundle, spec, encoder, backends, scene_cache, device)
        state = runner.step_data["state"]
        mask = runner.step_data["mask"]
        done = False
        q_means, q_maxs = [], []
        losses, grad_norms, target_means, td_errors = [], [], [], []
        while not done:
            q_values = agent.q_values(state)
            available = q_values[np.asarray(mask, bool)]
            q_means.append(float(available.mean()))
            q_maxs.append(float(available.max()))
            action = agent.act_from_q(q_values, mask, epsilon=agent.epsilon())
            transition = runner.apply(action)
            agent.buffer.add(
                state, action, transition["reward"],
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
            done = bool(transition["done"])
        result = runner.finalize("dqn")
        if result["latency_return_abs_error"] > REWARD_TOLERANCE:
            raise RuntimeError(
                f"LatencyReturn mismatch at episode {episode}: "
                f"{result['latency_return_abs_error']}"
            )
        finite = [
            result["episode_return"], result["mean_episode_e2e_ms"],
            result["final_ndtw"], result["latency_return"],
        ]
        if not np.all(np.isfinite(finite)):
            raise RuntimeError(f"non-finite metrics at episode {episode}")
        if not np.all(np.isfinite(q_means + q_maxs)):
            raise RuntimeError(f"non-finite Q at episode {episode}")
        if abs(float(np.mean(q_means))) > 1e6 or abs(float(
            np.max(q_maxs)
        )) > 1e6:
            raise RuntimeError(f"Q divergence at episode {episode}")
        row = _episode_row("dqn", spec, result, {
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
        _periodic_save(
            args, agent, "dqn", episode, best_state, history, run_dir,
            history_path=run_dir / "training_history.json",
            progress_path=run_dir / "progress.json",
        )
        if (
            episode % int(args.validation_every) == 0
            or episode == int(args.max_episodes)
        ):
            record, improved = _run_validation(
                agent, "dqn", evaluator, episode, history, writer, run_dir,
                run_dir / "progress.json", best_state,
            )
            _print_validation("dqn", record, improved)
            write_validation_outputs(run_dir, history)
            atomic_write_json(run_dir / "training_history.json", history)
            save_checkpoint(
                agent, "dqn", last_full, full=True,
                extra=checkpoint_extra(
                    "dqn", episode, best_state["best_e2e"],
                    best_state["best_episode"], record,
                ),
            )
            save_checkpoint(
                agent, "dqn", last_model, full=False,
                extra=checkpoint_extra(
                    "dqn", episode, best_state["best_e2e"],
                    best_state["best_episode"], record,
                ),
            )
            _progress(
                run_dir / "progress.json", args, "dqn", episode, history,
                best_state,
            )
    return agent


def train_ppo(args, bundle, device, encoder, backends, scene_cache,
              evaluator, writer, run_dir, history, start_episode,
              best_state):
    agent = PPOAgent(
        device, seed=int(args.seed), state_dim=STATE_DIM,
        state_mode=STATE_MODE, gamma=1.0,
    )
    last_full = run_dir / "checkpoints" / "last_full.pt"
    last_model = run_dir / "checkpoints" / "last.pt"
    if args.resume:
        if not last_full.is_file():
            raise SystemExit(f"--resume but {last_full} is missing")
        payload = agent.load_full(last_full)
        if payload.get("stage2n_reward_mode") != "latency_only_v1":
            raise SystemExit("Resume checkpoint is not a Stage-2n run")
        print(
            f"[PPO] resumed at episode {start_episode}, "
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
        runner = _runner(bundle, spec, encoder, backends, scene_cache, device)
        state = runner.step_data["state"]
        mask = runner.step_data["mask"]
        done = False
        entropies, values = [], []
        while not done:
            try:
                decision = agent.act(state, mask)
            except PPOAnomalyError as exc:
                _abort(run_dir, agent, "ppo", episode, exc, history,
                       best_state)
            transition = runner.apply(int(decision["action"]))
            rollout.add(
                state, mask, int(decision["action"]),
                float(decision["log_prob"]), float(decision["value"]),
                float(transition["reward"]), bool(transition["done"]),
            )
            entropies.append(float(decision["entropy"]))
            values.append(float(decision["value"]))
            agent.total_steps += 1
            state = transition["next_state"]
            mask = transition["next_mask"]
            done = bool(transition["done"])
        result = runner.finalize("ppo")
        if result["latency_return_abs_error"] > REWARD_TOLERANCE:
            raise RuntimeError(
                f"LatencyReturn mismatch at episode {episode}: "
                f"{result['latency_return_abs_error']}"
            )
        finite = [
            result["episode_return"], result["mean_episode_e2e_ms"],
            result["final_ndtw"], result["latency_return"],
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
                _abort(run_dir, agent, "ppo", episode, exc, history,
                       best_state)
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
        row = _episode_row("ppo", spec, result, {
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
        _periodic_save(
            args, agent, "ppo", episode, best_state, history, run_dir,
            history_path=run_dir / "training_history.json",
            progress_path=run_dir / "progress.json",
        )
        if (
            episode % int(args.validation_every) == 0
            or episode == int(args.max_episodes)
        ):
            record, improved = _run_validation(
                agent, "ppo", evaluator, episode, history, writer, run_dir,
                run_dir / "progress.json", best_state,
            )
            _print_validation("ppo", record, improved)
            write_validation_outputs(run_dir, history)
            atomic_write_json(run_dir / "training_history.json", history)
            save_checkpoint(
                agent, "ppo", last_full, full=True,
                extra=checkpoint_extra(
                    "ppo", episode, best_state["best_e2e"],
                    best_state["best_episode"], record,
                ),
            )
            save_checkpoint(
                agent, "ppo", last_model, full=False,
                extra=checkpoint_extra(
                    "ppo", episode, best_state["best_e2e"],
                    best_state["best_episode"], record,
                ),
            )
            _progress(
                run_dir / "progress.json", args, "ppo", episode, history,
                best_state,
            )
    return agent


# ---------------------------------------------------------------------------
# Shared logging / progress
# ---------------------------------------------------------------------------


def _log_train_row(writer, row, episode):
    for tag, key in (
        ("train/LatencyReturn", "latency_return"),
        ("train/mean_e2e_ms", "mean_episode_e2e_ms"),
        ("train/success", "success"),
        ("train/final_nDTW", "final_nDTW"),
        ("train/steps", "steps"),
        ("train/epsilon", "epsilon"),
        ("train/q_mean", "q_mean"),
        ("train/q_max", "q_max"),
        ("train/target_mean", "target_mean"),
        ("train/td_error_abs_mean", "td_error_abs_mean"),
        ("train/loss", "loss"),
        ("train/grad_norm", "grad_norm"),
        ("train/policy_entropy", "policy_entropy"),
        ("train/value_mean", "value_mean"),
        ("train/value_abs_max", "value_abs_max"),
        ("train/total_steps", "total_steps"),
    ):
        value = row.get(key)
        if value is not None:
            writer.add_scalar(tag, float(value), episode)


def _log_windows(writer, reward_window, e2e_window, success_window,
                 ndtw_window, episode):
    if len(reward_window) == reward_window.maxlen:
        writer.add_scalar(
            "train/LatencyReturn_MA100", float(np.mean(reward_window)),
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


def _periodic_save(args, agent, algorithm, episode, best_state, history,
                   run_dir, history_path, progress_path):
    if episode % 10 == 0:
        atomic_write_json(history_path, history)
        _progress(progress_path, args, algorithm, episode, history, best_state)
        row = history["train"][-1]
        print(
            f"[{algorithm.upper()}] Ep{episode}/{args.max_episodes} "
            f"LatencyReturn={row['latency_return']:.4f} "
            f"E2E={row['mean_episode_e2e_ms']:.1f}ms "
            f"SR={int(row['success'])} nDTW={row['final_nDTW']:.4f} "
            f"band={row['band']} eps={row.get('epsilon', 0.0):.3f}",
            flush=True,
        )
    if episode % max(1, int(args.checkpoint_every)) == 0 or episode % 50 == 0:
        extra = checkpoint_extra(
            algorithm, episode, best_state["best_e2e"],
            best_state["best_episode"], best_state.get("last_validation"),
        )
        save_checkpoint(
            agent, algorithm, run_dir / "checkpoints" / "last_full.pt",
            full=True, extra=extra,
        )
        save_checkpoint(
            agent, algorithm, run_dir / "checkpoints" / "last.pt",
            full=False, extra=extra,
        )


def _progress(path, args, algorithm, episode, history, best_state):
    payload = read_json(path) or {
        "stage": STAGE_NAME,
        "created_utc": utc_now(),
    }
    payload.update({
        "updated_utc": utc_now(),
        "status": "training",
        "algorithm": algorithm,
        "seed": int(args.seed),
        "current_episode": int(episode),
        "max_episodes": int(args.max_episodes),
        "best_dev_e2e_ms": (
            None if best_state["best_e2e"] == float("inf")
            else float(best_state["best_e2e"])
        ),
        "best_dev_episode": int(best_state["best_episode"]),
        "best_checkpoint": str(
            path.parent / "checkpoints" / "best.pt"
        ) if best_state["best_e2e"] != float("inf") else None,
        "last_validation": best_state.get("last_validation"),
        "validations": len(history["validation"]),
        "final_test_read": False,
    })
    atomic_write_json(path, payload)


def _abort(run_dir, agent, algorithm, episode, exc, history, best_state):
    payload = {
        "episode": int(episode),
        "error": str(exc),
        "diagnostics": getattr(exc, "diagnostics", {}),
        "created_utc": utc_now(),
    }
    atomic_write_json(run_dir / "abort.json", payload)
    try:
        save_checkpoint(
            agent, algorithm, run_dir / "checkpoints" / "last_full.pt",
            full=True,
            extra=checkpoint_extra(
                algorithm, episode, best_state["best_e2e"],
                best_state["best_episode"],
                best_state.get("last_validation"),
            ),
        )
    finally:
        raise exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--validation-every", type=int,
                        default=VALIDATION_EVERY)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    args = parser.parse_args()

    formal = not args.smoke
    if formal:
        if int(args.max_episodes) != TRAIN_EPISODES:
            raise SystemExit(
                f"formal Stage-2n runs require --max-episodes "
                f"{TRAIN_EPISODES}"
            )
        if int(args.validation_every) != VALIDATION_EVERY:
            raise SystemExit(
                f"formal Stage-2n runs require --validation-every "
                f"{VALIDATION_EVERY}"
            )
        if int(args.seed) not in AGENT_SEEDS:
            raise SystemExit(f"formal seeds must be one of {AGENT_SEEDS}")
    else:
        args.max_episodes = max(1, min(int(args.max_episodes), 20))
        args.validation_every = max(
            1, min(int(args.validation_every), int(args.max_episodes))
        )

    if args.run_dir is None:
        if formal:
            args.run_dir = RUNS_DIR / f"{args.algorithm}_s{args.seed}"
        else:
            args.run_dir = (
                STAGE2N_ROOT / "smoke"
                / f"{args.algorithm}_s{args.seed}"
            )
    run_dir = Path(args.run_dir)
    tensorboard_dir = Path(
        args.tensorboard_dir or (run_dir / "tensorboard")
    )
    checkpoint_dir = run_dir / "checkpoints"
    history_path = run_dir / "training_history.json"
    progress_path = run_dir / "progress.json"
    result_path = run_dir / "training_result.json"
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
    protocol = protocol_payload(
        bundle, args.algorithm, args.seed, device, args.threads,
        sys.argv, run_dir, tensorboard_dir, args.max_episodes,
        args.validation_every,
    )
    install_protocol(run_dir, protocol, sys.argv)
    print(
        f"[{args.algorithm.upper()}] starting seed={args.seed} "
        f"episodes={args.max_episodes} validation={args.validation_every} "
        f"deployment={bundle['deployment']['deployment_sha256'][:12]} "
        f"mec={MEC_CHECKPOINT_SHA256[:12]}",
        flush=True,
    )

    device, encoder, backends = load_stage2n_runtime(device)
    scene_cache = make_stage2n_scene_cache(
        encoder, device, capacity=max(64, len(bundle["train_specs"][:1000]))
    )
    print(
        f"[{args.algorithm.upper()}] building {len(bundle['dev_specs'])} "
        "frozen Dev-60 MEC lanes for exact fast validation",
        flush=True,
    )
    dev_lanes = build_lanes(
        bundle["dev_specs"],
        bundle["topology"],
        encoder,
        backends,
        bundle["quality_prior"],
        bundle["gab_config"],
        device,
        scene_cache,
        verbose=True,
    )
    evaluator = LaneEvaluator(dev_lanes)

    history = read_json(history_path)
    if history is None:
        history = {
            "stage": STAGE_NAME,
            "algorithm": args.algorithm,
            "seed": int(args.seed),
            "created_utc": utc_now(),
            "config": {
                "algorithm": args.algorithm,
                "seed": int(args.seed),
                "max_episodes": int(args.max_episodes),
                "validation_every": int(args.validation_every),
                "state_mode": STATE_MODE,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "reward": REWARD_DESCRIPTION,
                "reward_mode": "latency_only_v1",
                "gamma": 1.0,
                "convergence_criterion": CONVERGENCE_CRITERION,
                "checkpoint_selection": CHECKPOINT_SELECTION,
                "mec_checkpoint_sha256": MEC_CHECKPOINT_SHA256,
                "deployment_sha256": bundle["deployment"][
                    "deployment_sha256"
                ],
                "protocol_sha256": sha256_file(PROTOCOL_PATH),
                "final_test_read": False,
            },
            "train": [],
            "validation": [],
            "validation_episodes": [],
        }
    start_episode = len(history["train"])
    if start_episode and not args.resume:
        raise SystemExit("history exists without --resume")
    best_state = {
        "best_e2e": float("inf"),
        "best_episode": 0,
        "best_record": None,
        "last_validation": None,
    }
    if history["validation"]:
        best = select_best_validation(history["validation"])
        best_state["best_e2e"] = float(best["E2E_ms"])
        best_state["best_episode"] = int(best["episode"])
        best_state["best_record"] = best
        best_state["last_validation"] = history["validation"][-1]
    elif (checkpoint_dir / "best.pt").is_file() and not args.resume:
        raise SystemExit("best.pt exists without validation history")

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    started = time.time()
    if args.algorithm == "dqn":
        agent = train_dqn(
            args, bundle, device, encoder, backends, scene_cache, evaluator,
            writer, run_dir, history, start_episode, best_state,
        )
    else:
        agent = train_ppo(
            args, bundle, device, encoder, backends, scene_cache, evaluator,
            writer, run_dir, history, start_episode, best_state,
        )
    elapsed = time.time() - started
    reached_cap = len(history["train"]) >= int(args.max_episodes)
    convergence = convergence_status(history["validation"], reached_cap)
    best = select_best_validation(history["validation"])
    result = {
        "stage": STAGE_NAME,
        "algorithm": args.algorithm,
        "seed": int(args.seed),
        "created_utc": utc_now(),
        "elapsed_seconds": float(elapsed),
        "episodes": len(history["train"]),
        "validations": len(history["validation"]),
        "convergence": convergence,
        "best_validation": best,
        "final_train_row": history["train"][-1] if history["train"] else None,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "best_dev_e2e_ms": best_state["best_e2e"],
        "best_dev_episode": best_state["best_episode"],
        "reward_mode": "latency_only_v1",
        "final_test_read": False,
        "protocol": str(run_dir / "protocol.json"),
    }
    atomic_write_json(result_path, result)
    atomic_write_json(history_path, history)
    save_checkpoint(
        agent, args.algorithm, checkpoint_dir / "last_full.pt", full=True,
        extra=checkpoint_extra(
            args.algorithm, len(history["train"]), best_state["best_e2e"],
            best_state["best_episode"], best, convergence,
        ),
    )
    save_checkpoint(
        agent, args.algorithm, checkpoint_dir / "last.pt", full=False,
        extra=checkpoint_extra(
            args.algorithm, len(history["train"]), best_state["best_e2e"],
            best_state["best_episode"], best, convergence,
        ),
    )
    writer.add_hparams(
        {
            "algorithm": args.algorithm,
            "seed": int(args.seed),
            "max_episodes": int(args.max_episodes),
        },
        {
            "best_dev_e2e_ms": float(best_state["best_e2e"]),
            "final_dev_e2e_ms": (
                float(history["validation"][-1]["E2E_ms"])
                if history["validation"] else float("nan")
            ),
            "converged": float(convergence["converged"]),
        },
    )
    writer.close()
    payload = read_json(progress_path) or {}
    payload.update({
        "updated_utc": utc_now(),
        "status": convergence["status"],
        "algorithm": args.algorithm,
        "seed": int(args.seed),
        "current_episode": len(history["train"]),
        "max_episodes": int(args.max_episodes),
        "best_dev_e2e_ms": float(best_state["best_e2e"]),
        "best_dev_episode": int(best_state["best_episode"]),
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "convergence": convergence,
        "final_test_read": False,
    })
    atomic_write_json(progress_path, payload)
    print(
        f"[{args.algorithm.upper()}] done: episodes={len(history['train'])} "
        f"best E2E={best_state['best_e2e']:.2f}ms @Ep"
        f"{best_state['best_episode']} status={convergence['status']} "
        f"converged={convergence['converged']}",
        flush=True,
    )
    print("STAGE2N_TRAIN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
