"""Stage-2p: original reward vs scheme D for PPO.

Both arms use the *same* Stage-2p PPO training loop.  The only difference is
the reward function:

    baseline: r_t = -0.25*(mean_t - mean_{t-1})/100
    D       : u_t = 1 - e2e_t/tau; r_t = u_t/H; terminal += (1-T/H)*J

Fixed Stage-2n conditions: fixed city, A topology, 20 homogeneous
Transformer-M MECs, Train-4000 / Dev-60, task order, 384-D Pure-MEC v2 state,
20-D mask, PPO hyperparameters, gamma=1.0, 4000 episodes, Dev every 200.
The Test-400 pool is never opened.

Usage:
    python v83_spatial_stage2/stage2p/train.py --scheme baseline --seed 0
    python v83_spatial_stage2/stage2p/train.py --scheme D --seed 2
"""

from __future__ import annotations

import argparse
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
from v83_spatial_stage2.stage2k_ppo_agent import (  # noqa: E402
    PPOAgent,
    PPOAnomalyError,
    RolloutBuffer,
)
from v83_spatial_stage2.stage2n_common import (  # noqa: E402
    ACTION_DIM,
    CHECKPOINT_SELECTION,
    CONVERGENCE_CRITERION,
    MEC_CHECKPOINT_SHA256,
    PROTOCOL_PATH as STAGE2N_PROTOCOL_PATH,
    STATE_DIM,
    STATE_MODE,
    TRAIN_EPISODES,
    VALIDATION_EVERY,
    convergence_status,
    select_best_validation,
)
from v83_spatial_stage2.stage2n_lanes import (  # noqa: E402
    LaneEvaluator,
    build_lanes,
)
from v83_spatial_stage2.stage2n_runtime import (  # noqa: E402
    LatencyOnlyEpisodeRunner,
    load_stage2n_runtime,
    make_stage2n_scene_cache,
)
from v83_spatial_stage2.stage2n_train import (  # noqa: E402
    _compact_validation_row,
    _max_defined,
    _mean_defined,
    load_bundle,
    save_checkpoint,
    validation_record,
    write_validation_outputs,
)
from v83_spatial_stage2.stage2o.train import _git_state  # noqa: E402
from v83_spatial_stage2.stage2p.progress import (  # noqa: E402
    write_round,
    write_training_csv,
)
from v83_spatial_stage2.stage2p.rewards import (  # noqa: E402
    H_MAX,
    SCHEME_DESCRIPTIONS,
    SCHEME_LABELS,
    SCHEMES,
    TAU_MS,
    EpisodeRewardController,
    validate_scheme,
)

STAGE_NAME = "stage2p"
STAGE_TITLE = "Stage-2p: Original Reward vs Scheme D for PPO"
FORMAL_SEEDS = (0, 1, 2)
FORMAL_EPISODES = TRAIN_EPISODES
FORMAL_VALIDATION = VALIDATION_EVERY
REWARD_TOLERANCE = 1e-8
VALIDATION_IMPROVEMENT_MS = 1e-6
RUNS_ROOT = PROJECT_ROOT / "runs" / "v83_spatial_stage2" / "stage2p"
LOGS_ROOT = RUNS_ROOT / "logs"
STAGE2O_FREEZE = (
    PROJECT_ROOT / "v83_spatial_stage2" / "stage2o"
    / "evidence" / "protocol_freeze.json"
)


def _source_hashes():
    """Stage-2p sources plus all training-relevant parent modules."""
    stage_dir = Path(__file__).resolve().parent
    files = sorted(stage_dir.glob("*.py")) + [
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2k_jointr_v2_agent.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2k_ppo_agent.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2n_train.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2n_lanes.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2n_runtime.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2n_common.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2o" / "rewards.py",
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in files
    }


def protocol_payload(bundle, scheme, seed, device, threads, argv,
                     run_dir, tensorboard_dir, max_episodes,
                     validation_every):
    deploy = bundle["deployment"]
    return {
        "stage": STAGE_NAME,
        "stage_name": STAGE_TITLE,
        "algorithm": "ppo",
        "reward_scheme": scheme,
        "scheme_label": SCHEME_LABELS[scheme],
        "seed": int(seed),
        "created_utc": utc_now(),
        "parent_stage2n": {
            "protocol_path": str(STAGE2N_PROTOCOL_PATH),
            "protocol_sha256": sha256_file(STAGE2N_PROTOCOL_PATH),
        },
        "parent_stage2o": {
            "protocol_freeze_path": str(STAGE2O_FREEZE),
            "protocol_freeze_sha256": (
                sha256_file(STAGE2O_FREEZE)
                if STAGE2O_FREEZE.is_file() else None
            ),
        },
        "reward": {
            "scheme": scheme,
            "tau_ms": TAU_MS,
            "horizon": H_MAX,
            "gamma": 1.0,
            "formula": SCHEME_DESCRIPTIONS[scheme],
            "navigation_reward": "removed",
            "success_bonus": "none",
            "test_used_for_reward": False,
        },
        "artifact_hashes": {
            "train_tasks": bundle["protocol"]["tasks"]["train"]["sha256"],
            "dev_tasks": bundle["protocol"]["tasks"]["dev"]["sha256"],
            "test_tasks": bundle["protocol"]["tasks"]["test"]["sha256"],
            "training_order": bundle["protocol"]["training_order"]["sha256"],
            "deployment": bundle["protocol"]["deployment"]["sha256"],
            "mec_checkpoint": MEC_CHECKPOINT_SHA256,
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
            "source_topology": deploy["source_topology"],
            "source_topology_sha256": deploy["source_topology_sha256"],
        },
        "state": {
            "mode": STATE_MODE,
            "dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "availability_mask_separate": True,
            "visual_inputs_to_association_agent": False,
        },
        "training": {
            "algorithm": "ppo",
            "max_episodes": int(max_episodes),
            "validation_every": int(validation_every),
            "validation_tasks": len(bundle["dev_specs"]),
            "validation_split": "dev",
            "seed": int(seed),
            "same_order_for_all_runs": True,
            "test_used": False,
        },
        "convergence": {
            "criterion": CONVERGENCE_CRITERION,
            "checkpoint_selection": CHECKPOINT_SELECTION,
        },
        "source_hashes": _source_hashes(),
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


def checkpoint_extra(scheme, episode, best_e2e, best_episode, record,
                     convergence=None):
    payload = {
        "stage": STAGE_NAME,
        "algorithm": "ppo",
        "stage2p_reward_scheme": scheme,
        "reward_scheme": scheme,
        "reward_formula": SCHEME_DESCRIPTIONS[scheme],
        "episode": int(episode),
        "best_dev_e2e_ms": float(best_e2e),
        "best_dev_episode": int(best_episode),
        "validation": record,
        "final_test_read": False,
    }
    if convergence is not None:
        payload["convergence"] = convergence
    return payload


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


def _episode_row(scheme, spec, result, diagnostics, extra):
    row = {
        "episode": int(extra.pop("episode")),
        "algorithm": "ppo",
        "scheme": scheme,
        "episode_uid": spec["episode_uid"],
        "band": spec["band"],
        "task_seed": int(spec["task_seed"]),
        "steps": int(result["steps"]),
        "episode_return": float(diagnostics["return"]),
        "latency_return": float(result["latency_return"]),
        "reward_expected": float(diagnostics["expected_return"]),
        "reward_abs_error": float(diagnostics["abs_error"]),
        "terminal_reward": float(diagnostics["terminal_reward"]),
        "terminal_correction": float(diagnostics["terminal_correction"]),
        "terminal_share": diagnostics["terminal_share"],
        "base_sum": float(diagnostics["base_sum"]),
        "nonterminal_mean": float(diagnostics["nonterminal_mean"]),
        "nonterminal_std": float(diagnostics["nonterminal_std"]),
        "nonterminal_abs_max": float(diagnostics["nonterminal_abs_max"]),
        "step_abs_max": float(diagnostics["step_abs_max"]),
        "mean_episode_e2e_ms": float(result["mean_episode_e2e_ms"]),
        "success": bool(result["success"]),
        "osr": bool(result["osr"]),
        "spl": float(result["spl"]),
        "final_nDTW": float(result["final_ndtw"]),
        "reward_mode": f"stage2p_{scheme}",
        "tau_ms": float(TAU_MS),
        "horizon": int(H_MAX),
    }
    row.update(extra)
    return row


def _episode_diagnostics(result, controller, done):
    if not done:
        raise RuntimeError("episode diagnostics requested before done")
    reward, diagnostics = controller.finalize(
        result["mean_episode_e2e_ms"], result["steps"]
    )
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
    if diagnostics["abs_error"] > REWARD_TOLERANCE:
        raise RuntimeError(
            f"reward identity failed: {diagnostics['return']} vs "
            f"{diagnostics['expected_return']}"
        )
    return reward, diagnostics


def _run_validation(agent, evaluator, episode, history, writer, run_dir,
                    best_state, scheme):
    rows = evaluator.evaluate_ppo(agent)
    recent = history["train"][-FORMAL_VALIDATION:]
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
        row["algorithm"] = "ppo"
    history.setdefault("validation_episodes", []).extend(compact)
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
            agent, "ppo", run_dir / "checkpoints" / "best.pt",
            full=False,
            extra=checkpoint_extra(
                scheme, episode, best_state["best_e2e"],
                best_state["best_episode"], record,
            ),
        )
    best_state["last_validation"] = record
    return record, improved


def _print_validation(scheme, record, improved):
    print(
        f"[PPO|{scheme}] 验证 Ep{record['episode']}: "
        f"E2E={record['E2E_ms']:.2f}ms "
        f"P95={record['P95_E2E_ms']:.1f}ms "
        f"违约率={100*record['delay_violation_rate']:.2f}% "
        f"SR={100*record['SR']:.2f}% "
        f"SPL={record['SPL']:.4f} nDTW={record['nDTW']:.4f} "
        f"best={'是' if improved else '否'}",
        flush=True,
    )


def _periodic_save(args, agent, scheme, episode, best_state, history,
                   run_dir, history_path, progress_path, loop_started):
    if episode % 10 == 0:
        history["elapsed_seconds"] = (
            float(history.get("elapsed_seconds", 0.0))
            + (time.monotonic() - loop_started)
        )
        loop_started = time.monotonic()
        atomic_write_json(history_path, history)
        _progress(progress_path, args, scheme, episode, history, best_state)
        row = history["train"][-1]
        print(
            f"[PPO|{scheme}] Ep{episode}/{args.max_episodes} "
            f"R={row['episode_return']:.4f} "
            f"E2E={row['mean_episode_e2e_ms']:.1f}ms "
            f"SR={int(row['success'])} nDTW={row['final_nDTW']:.4f} "
            f"band={row['band']}",
            flush=True,
        )
    if episode % max(1, int(args.checkpoint_every)) == 0 or episode % 50 == 0:
        extra = checkpoint_extra(
            scheme, episode, best_state["best_e2e"],
            best_state["best_episode"], best_state.get("last_validation"),
        )
        save_checkpoint(
            agent, "ppo", run_dir / "checkpoints" / "last_full.pt",
            full=True, extra=extra,
        )
        save_checkpoint(
            agent, "ppo", run_dir / "checkpoints" / "last.pt",
            full=False, extra=extra,
        )
    return loop_started


def _progress(path, args, scheme, episode, history, best_state):
    payload = read_json(path) or {
        "stage": STAGE_NAME,
        "created_utc": utc_now(),
    }
    payload.update({
        "updated_utc": utc_now(),
        "status": "training",
        "stage": STAGE_NAME,
        "algorithm": "ppo",
        "scheme": scheme,
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
        "elapsed_seconds": float(history.get("elapsed_seconds", 0.0)),
        "final_test_read": False,
    })
    atomic_write_json(path, payload)


def _abort(run_dir, agent, scheme, episode, exc, history, best_state):
    payload = {
        "episode": int(episode),
        "error": str(exc),
        "diagnostics": getattr(exc, "diagnostics", {}),
        "created_utc": utc_now(),
    }
    atomic_write_json(run_dir / "abort.json", payload)
    try:
        save_checkpoint(
            agent, "ppo", run_dir / "checkpoints" / "last_full.pt",
            full=True,
            extra=checkpoint_extra(
                scheme, episode, best_state["best_e2e"],
                best_state["best_episode"],
                best_state.get("last_validation"),
            ),
        )
    finally:
        raise exc


def train_ppo(args, bundle, device, encoder, backends, scene_cache,
              evaluator, writer, run_dir, history, start_episode,
              best_state, loop_started):
    scheme = args.scheme
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
        if payload.get("stage2p_reward_scheme") != scheme:
            raise SystemExit(
                "Resume checkpoint is not a Stage-2p run with scheme "
                f"{scheme}"
            )
        print(
            f"[PPO|{scheme}] resumed at episode {start_episode}, "
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
        controller = EpisodeRewardController(scheme)
        entropies, values = [], []
        result = None
        diagnostics = None
        while not done:
            try:
                decision = agent.act(state, mask)
            except PPOAnomalyError as exc:
                _abort(run_dir, agent, scheme, episode, exc, history,
                       best_state)
            transition = runner.apply(int(decision["action"]))
            done = bool(transition["done"])
            controller.step_reward(transition["e2e_ms"])
            if done:
                result = runner.finalize("ppo")
                reward, diagnostics = _episode_diagnostics(
                    result, controller, done=True
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
                _abort(run_dir, agent, scheme, episode, exc, history,
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
        row = _episode_row(scheme, spec, result, diagnostics, {
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
            args, agent, scheme, episode, best_state, history,
            run_dir, run_dir / "training_history.json",
            run_dir / "progress.json", loop_started,
        )
        if (
            episode % int(args.validation_every) == 0
            or episode == int(args.max_episodes)
        ):
            record, improved = _run_validation(
                agent, evaluator, episode, history, writer, run_dir,
                best_state, scheme,
            )
            _print_validation(scheme, record, improved)
            write_validation_outputs(run_dir, history)
            write_training_csv(run_dir, history["train"])
            write_round(
                run_dir, history, record, scheme, int(args.seed),
                history.get("elapsed_seconds", 0.0),
            )
            atomic_write_json(run_dir / "training_history.json", history)
            extra = checkpoint_extra(
                scheme, episode, best_state["best_e2e"],
                best_state["best_episode"], record,
            )
            save_checkpoint(
                agent, "ppo", last_full, full=True, extra=extra,
            )
            save_checkpoint(
                agent, "ppo", last_model, full=False, extra=extra,
            )
            _progress(
                run_dir / "progress.json", args, scheme, episode,
                history, best_state,
            )
    return agent


def _log_train_row(writer, row, episode):
    for tag, key in (
        ("train/episode_return", "episode_return"),
        ("train/LatencyReturn", "latency_return"),
        ("train/mean_e2e_ms", "mean_episode_e2e_ms"),
        ("train/success", "success"),
        ("train/final_nDTW", "final_nDTW"),
        ("train/steps", "steps"),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=FORMAL_EPISODES)
    parser.add_argument("--validation-every", type=int,
                        default=FORMAL_VALIDATION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    args = parser.parse_args()
    args.scheme = validate_scheme(args.scheme)

    formal = not args.smoke
    if formal:
        if int(args.max_episodes) != FORMAL_EPISODES:
            raise SystemExit(
                f"formal Stage-2p runs require --max-episodes "
                f"{FORMAL_EPISODES}"
            )
        if int(args.validation_every) != FORMAL_VALIDATION:
            raise SystemExit(
                f"formal Stage-2p runs require --validation-every "
                f"{FORMAL_VALIDATION}"
            )
        if int(args.seed) not in FORMAL_SEEDS:
            raise SystemExit(f"formal seeds must be one of {FORMAL_SEEDS}")
    else:
        args.max_episodes = max(1, min(int(args.max_episodes), 60))
        args.validation_every = max(
            1, min(int(args.validation_every), int(args.max_episodes))
        )

    if args.run_dir is None:
        if formal:
            args.run_dir = (
                RUNS_ROOT / args.scheme / f"ppo_s{args.seed}"
            )
        else:
            args.run_dir = (
                RUNS_ROOT / "smoke"
                / f"{args.scheme}_ppo_s{args.seed}"
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
        bundle, args.scheme, args.seed, device, args.threads, sys.argv,
        run_dir, tensorboard_dir, args.max_episodes, args.validation_every,
    )
    install_protocol(run_dir, protocol, sys.argv)
    print(
        f"[PPO|{args.scheme}] starting seed={args.seed} "
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
        f"[PPO|{args.scheme}] building "
        f"{len(bundle['dev_specs'])} frozen Dev-60 MEC lanes for exact "
        "fast validation",
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
            "algorithm": "ppo",
            "scheme": args.scheme,
            "seed": int(args.seed),
            "created_utc": utc_now(),
            "config": {
                "algorithm": "ppo",
                "reward_scheme": args.scheme,
                "scheme_label": SCHEME_LABELS[args.scheme],
                "seed": int(args.seed),
                "max_episodes": int(args.max_episodes),
                "validation_every": int(args.validation_every),
                "state_mode": STATE_MODE,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "tau_ms": TAU_MS,
                "horizon": H_MAX,
                "gamma": 1.0,
                "reward_formula": SCHEME_DESCRIPTIONS[args.scheme],
                "convergence_criterion": CONVERGENCE_CRITERION,
                "checkpoint_selection": CHECKPOINT_SELECTION,
                "mec_checkpoint_sha256": MEC_CHECKPOINT_SHA256,
                "deployment_sha256": bundle["deployment"][
                    "deployment_sha256"
                ],
                "parent_protocol_sha256": sha256_file(
                    STAGE2N_PROTOCOL_PATH
                ),
                "final_test_read": False,
            },
            "elapsed_seconds": 0.0,
            "train": [],
            "validation": [],
            "validation_episodes": [],
        }
    else:
        if history.get("stage") != STAGE_NAME:
            raise SystemExit("existing history is not a Stage-2p run")
        if history.get("scheme") != args.scheme:
            raise SystemExit("existing history has a different scheme")
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
    loop_started = time.monotonic()
    agent = train_ppo(
        args, bundle, device, encoder, backends, scene_cache, evaluator,
        writer, run_dir, history, start_episode, best_state, loop_started,
    )
    reached_cap = len(history["train"]) >= int(args.max_episodes)
    convergence = convergence_status(history["validation"], reached_cap)
    best = select_best_validation(history["validation"])
    result = {
        "stage": STAGE_NAME,
        "algorithm": "ppo",
        "scheme": args.scheme,
        "scheme_label": SCHEME_LABELS[args.scheme],
        "seed": int(args.seed),
        "created_utc": utc_now(),
        "elapsed_seconds": float(history.get("elapsed_seconds", 0.0)),
        "episodes": len(history["train"]),
        "validations": len(history["validation"]),
        "convergence": convergence,
        "best_validation": best,
        "final_train_row": history["train"][-1] if history["train"] else None,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "best_dev_e2e_ms": best_state["best_e2e"],
        "best_dev_episode": best_state["best_episode"],
        "reward_mode": f"stage2p_{args.scheme}",
        "final_test_read": False,
        "protocol": str(run_dir / "protocol.json"),
    }
    atomic_write_json(result_path, result)
    atomic_write_json(history_path, history)
    write_training_csv(run_dir, history["train"])
    save_checkpoint(
        agent, "ppo", checkpoint_dir / "last_full.pt", full=True,
        extra=checkpoint_extra(
            args.scheme, len(history["train"]), best_state["best_e2e"],
            best_state["best_episode"], best, convergence,
        ),
    )
    save_checkpoint(
        agent, "ppo", checkpoint_dir / "last.pt", full=False,
        extra=checkpoint_extra(
            args.scheme, len(history["train"]), best_state["best_e2e"],
            best_state["best_episode"], best, convergence,
        ),
    )
    writer.add_hparams(
        {
            "algorithm": "ppo",
            "reward_scheme": args.scheme,
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
        "stage": STAGE_NAME,
        "algorithm": "ppo",
        "scheme": args.scheme,
        "seed": int(args.seed),
        "current_episode": len(history["train"]),
        "max_episodes": int(args.max_episodes),
        "best_dev_e2e_ms": float(best_state["best_e2e"]),
        "best_dev_episode": int(best_state["best_episode"]),
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "convergence": convergence,
        "elapsed_seconds": float(history.get("elapsed_seconds", 0.0)),
        "final_test_read": False,
    })
    atomic_write_json(progress_path, payload)
    print(
        f"[PPO|{args.scheme}] done: episodes={len(history['train'])} "
        f"best E2E={best_state['best_e2e']:.2f}ms @Ep"
        f"{best_state['best_episode']} status={convergence['status']} "
        f"converged={convergence['converged']}",
        flush=True,
    )
    print("STAGE2P_TRAIN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
