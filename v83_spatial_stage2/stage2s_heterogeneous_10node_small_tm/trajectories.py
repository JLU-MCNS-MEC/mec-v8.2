"""Collect real closed-loop DQN/PPO UAV trajectories for Stage-2s.

For each algorithm the best Dev-E2E seed is used.  All 60 Dev tasks are run
with the greedy learned policy; one trajectory per success/failure x
short/medium/long cell is selected (with an explicit fallback when a cell is
empty).  The selected episodes are saved step by step, including the selected
MEC node, its model type, communication/inference/E2E latency, cumulative
scheme-D reward and distance to goal.

No Test task is read.  No lane cache is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENV  # noqa: E402
from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2  # noqa: E402
from v83_spatial_stage2.stage2k_ppo_agent import PPOAgent  # noqa: E402
from v83_spatial_stage2.stage2k_state import masked_argmax  # noqa: E402
from v83_spatial_stage2.stage2o.rewards import (  # noqa: E402
    EpisodeRewardController,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (  # noqa: E402
    ACTION_DIM,
    ALGORITHMS,
    ALGORITHM_LABELS,
    EVIDENCE_DIR,
    FORMAL_SEEDS,
    MEC_CHECKPOINT_SHA256,
    MEC_MODELS,
    STAGE,
    STATE_DIM,
    STATE_MODE,
    load_bundle,
    run_dir,
    source_hashes,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.runtime import (  # noqa: E402
    latency_stats,
    load_stage2s_runtime,
    make_scene_cache,
    runner_for,
)

BANDS = ("short", "medium", "long")
OUTCOMES = (True, False)
MAX_TASKS = 60


def _path_length(coords):
    if len(coords) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(coords[:, :2], axis=0), axis=1).sum())


def _hash(values, kind="int"):
    if kind == "str":
        text = ",".join(str(value) for value in values)
    else:
        text = ",".join(str(int(value)) for value in values)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _load_agent(algorithm, seed, device):
    run_path = run_dir(algorithm, seed)
    result = read_json(run_path / "training_result.json")
    if result is None or result.get("status") != "completed":
        raise FileNotFoundError(f"{run_path} is not complete")
    checkpoint = Path(result["best_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if algorithm == "dqn":
        agent = DQNAgentV2(
            device, seed=int(seed), gamma=1.0, state_dim=STATE_DIM,
            state_mode=STATE_MODE, action_dim=ACTION_DIM,
        )
    else:
        agent = PPOAgent(
            device, seed=int(seed), state_dim=STATE_DIM,
            state_mode=STATE_MODE, action_dim=ACTION_DIM, gamma=1.0,
        )
    payload = agent.load_model(checkpoint)
    if payload.get("stage") != STAGE:
        raise ValueError(f"{checkpoint} is not Stage-2s")
    return agent, result, checkpoint


def _best_seed(algorithm):
    candidates = []
    for seed in FORMAL_SEEDS:
        result = read_json(
            run_dir(algorithm, seed) / "training_result.json"
        )
        if result is None or result.get("status") != "completed":
            continue
        candidates.append((
            float(result["best_dev_e2e_ms"]), int(seed), result
        ))
    if not candidates:
        raise FileNotFoundError(
            f"no completed {algorithm} run to build trajectories"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1], candidates[0][2]


def _run_episode(bundle, spec, encoder, backends, cache, device, latency,
                 algorithm, agent):
    runner = runner_for(
        bundle, spec, encoder, backends, cache, device, latency
    )
    controller = EpisodeRewardController("D")
    while True:
        if algorithm == "dqn":
            action = int(masked_argmax(
                agent.q_values(runner.step_data["state"]),
                runner.step_data["mask"],
            ))
        else:
            action = int(agent.greedy_action(
                runner.step_data["state"], runner.step_data["mask"]
            ))
        transition = runner.apply(action)
        controller.step_reward(transition["e2e_ms"])
        if transition["done"]:
            break
    result = runner.finalize(f"{algorithm}_greedy")
    _, diagnostics = controller.finalize(
        result["mean_episode_e2e_ms"], result["steps"]
    )
    rewards = [float(value) for value in controller.rewards]
    cumulative = np.cumsum(rewards).tolist()
    return result, runner, rewards, cumulative, diagnostics


def _step_rows(bundle, algorithm, seed, checkpoint, spec, result, runner,
               rewards, cumulative):
    trajectory = np.asarray(result["trajectory"], dtype=np.float64)
    records = runner.step_records
    rows = []
    stats = latency_stats(records)
    for index, position in enumerate(trajectory):
        if index == 0:
            record = {}
            distance = float(np.linalg.norm(
                np.asarray(spec["start"], dtype=float)
                - np.asarray(spec["goal"], dtype=float)
            ))
            cumulative_reward = 0.0
            reward_step = 0.0
        else:
            record = records[index - 1]
            distance = float(record["distance_to_goal_m"])
            cumulative_reward = float(cumulative[index - 1])
            reward_step = float(rewards[index - 1])
        if index < len(trajectory) - 1:
            delta = trajectory[index + 1, :2] - position[:2]
            heading = float(np.arctan2(delta[1], delta[0]))
        elif index > 0:
            delta = position[:2] - trajectory[index - 1, :2]
            heading = float(np.arctan2(delta[1], delta[0]))
        else:
            heading = 0.0
        rows.append({
            "stage": STAGE,
            "algorithm": algorithm,
            "seed": int(seed),
            "policy": f"{algorithm}_s{seed}",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "episode_uid": spec["episode_uid"],
            "band": spec["band"],
            "success": int(result["success"]),
            "step": int(index),
            "x_m": float(position[0]),
            "y_m": float(position[1]),
            "z_m": float(ENV.uav_altitude_m),
            "heading_rad": heading,
            "node_index": int(record.get("node_index", -1))
            if record else -1,
            "node_id": record.get("node_id", "") if record else "",
            "model_type": record.get("model_type", "") if record else "",
            "action": int(record.get("action", -1)) if record else -1,
            "action_name": record.get("action_name", "START")
            if record else "START",
            "communication_ms": float(
                record.get("communication_ms", 0.0)
            ) if record else 0.0,
            "inference_ms": float(
                record.get("inference_ms", 0.0)
            ) if record else 0.0,
            "extraction_ms": float(
                record.get("extraction_ms", 0.0)
            ) if record else 0.0,
            "e2e_ms": float(record.get("e2e_ms", 0.0))
            if record else 0.0,
            "reward_D_step": reward_step,
            "cumulative_reward_D": cumulative_reward,
            "running_mean_e2e_ms": float(
                record.get("running_mean_e2e_ms", 0.0)
            ) if record else 0.0,
            "distance_to_goal_m": distance,
            "start_x_m": float(spec["start"][0]),
            "start_y_m": float(spec["start"][1]),
            "goal_x_m": float(spec["goal"][0]),
            "goal_y_m": float(spec["goal"][1]),
            "path_length_m": _path_length(trajectory),
            "mean_e2e_ms": float(result["mean_episode_e2e_ms"]),
            "reward_D": float(result["reward_D"]),
            "small_ratio": float(stats["small_ratio"]),
            "transformer_m_ratio": float(stats["transformer_m_ratio"]),
        })
    return rows


def _summary_row(algorithm, seed, checkpoint, spec, result, runner,
                 rewards, cumulative, selection):
    stats = latency_stats(runner.step_records)
    trajectory = np.asarray(result["trajectory"], dtype=np.float64)
    goal = np.asarray(spec["goal"], dtype=np.float64)
    final = trajectory[-1, :2]
    actions = [int(row["action"]) for row in runner.step_records]
    nodes = [int(row["node_index"]) for row in runner.step_records]
    models = [str(row["model_type"]) for row in runner.step_records]
    return {
        "stage": STAGE,
        "algorithm": algorithm,
        "seed": int(seed),
        "policy": f"{algorithm}_s{seed}",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "episode_uid": spec["episode_uid"],
        "band": spec["band"],
        "success": int(bool(result["success"])),
        "steps": int(result["steps"]),
        "start_x_m": float(trajectory[0, 0]),
        "start_y_m": float(trajectory[0, 1]),
        "end_x_m": float(final[0]),
        "end_y_m": float(final[1]),
        "goal_x_m": float(goal[0]),
        "goal_y_m": float(goal[1]),
        "final_distance_to_goal_m": float(np.linalg.norm(final - goal)),
        "path_length_m": _path_length(trajectory),
        "mean_e2e_ms": float(result["mean_episode_e2e_ms"]),
        "reward_D": float(result["reward_D"]),
        "final_ndtw": float(result["final_ndtw"]),
        "spl": float(result["spl"]),
        "osr": int(bool(result["osr"])),
        "cumulative_reward_D_last": float(cumulative[-1])
        if cumulative else 0.0,
        "small_steps": int(stats["small_steps"]),
        "transformer_m_steps": int(stats["transformer_m_steps"]),
        "small_ratio": float(stats["small_ratio"]),
        "transformer_m_ratio": float(stats["transformer_m_ratio"]),
        "mean_inference_ms": float(stats["mean_inference_ms"]),
        "mean_communication_ms": float(stats["mean_communication_ms"]),
        "action_sha256": _hash(actions),
        "node_sha256": _hash(nodes),
        "model_sha256": _hash(models, kind="str"),
        "path_sha256": _hash([
            round(float(value), 6) for value in trajectory.ravel()
        ], kind="str"),
        "role": selection["role"],
        "requested_outcome": int(selection["requested_outcome"]),
        "selection_status": selection["selection_status"],
        "selection_reason": selection["selection_reason"],
        "final_test_read": False,
    }


def _select_episodes(summaries):
    selected = []
    used = set()
    for band in BANDS:
        for outcome in OUTCOMES:
            candidates = [
                row for row in summaries
                if row["band"] == band
                and bool(row["success"]) is bool(outcome)
                and row["episode_uid"] not in used
            ]
            status = "exact"
            reason = ""
            if not candidates:
                candidates = [
                    row for row in summaries
                    if row["band"] == band
                    and row["episode_uid"] not in used
                ]
                status = "fallback_band"
                reason = (
                    f"{band}/{outcome} 空，使用同 band 另一 outcome"
                )
            if not candidates:
                candidates = [
                    row for row in summaries
                    if bool(row["success"]) is bool(outcome)
                    and row["episode_uid"] not in used
                ]
                status = "fallback_outcome"
                reason = (
                    f"{band}/{outcome} 空，使用同 outcome 另一 band"
                )
            if not candidates:
                candidates = [
                    row for row in summaries
                    if row["episode_uid"] not in used
                ]
                status = "fallback_any"
                reason = "该 band/outcome 无可替代轨迹，使用任意未选任务"
            if not candidates:
                continue
            candidates.sort(key=lambda row: (
                int(row["steps"]), str(row["episode_uid"])
            ))
            chosen = candidates[(len(candidates) - 1) // 2]
            used.add(chosen["episode_uid"])
            chosen = dict(chosen)
            chosen.update({
                "role": f"{'success' if outcome else 'failure'}_{band}",
                "requested_outcome": bool(outcome),
                "selection_status": status,
                "selection_reason": reason,
            })
            selected.append(chosen)
    return selected


def _selection_summary(summaries):
    rows = []
    for algorithm in ALGORITHMS:
        subset = [row for row in summaries if row["algorithm"] == algorithm]
        if not subset:
            continue
        small_ratio = np.asarray(
            [row["small_ratio"] for row in subset], dtype=float
        )
        success = np.asarray(
            [row["success"] for row in subset], dtype=float
        )
        ndtw = np.asarray([row["final_ndtw"] for row in subset], dtype=float)
        e2e = np.asarray([row["mean_e2e_ms"] for row in subset], dtype=float)
        steps = np.asarray([row["steps"] for row in subset], dtype=float)
        dominant = np.where(
            small_ratio > 0.5, "Small",
            np.where(small_ratio < 0.5, "Transformer-M", "Mixed"),
        )
        for group in ("Small", "Transformer-M", "Mixed"):
            mask = dominant == group
            if not mask.any():
                continue
            rows.append({
                "algorithm": algorithm,
                "group": group,
                "episodes": int(mask.sum()),
                "mean_small_ratio": float(small_ratio[mask].mean()),
                "success_rate": float(success[mask].mean()),
                "mean_ndtw": float(ndtw[mask].mean()),
                "mean_e2e_ms": float(e2e[mask].mean()),
                "mean_steps": float(steps[mask].mean()),
                "correlation": None,
            })
        for name, values in (
            ("success", success), ("ndtw", ndtw),
            ("e2e_ms", e2e), ("steps", steps),
        ):
            if np.std(small_ratio) > 1e-12 and np.std(values) > 1e-12:
                corr = float(np.corrcoef(small_ratio, values)[0, 1])
            else:
                corr = None
            rows.append({
                "algorithm": algorithm,
                "group": f"corr_small_ratio_vs_{name}",
                "episodes": len(subset),
                "mean_small_ratio": float(small_ratio.mean()),
                "success_rate": float(success.mean()),
                "mean_ndtw": float(ndtw.mean()),
                "mean_e2e_ms": float(e2e.mean()),
                "mean_steps": float(steps.mean()),
                "correlation": corr,
            })
    return rows


def _audit(summary_rows, step_rows):
    rows = []
    by_key = {}
    for row in step_rows:
        by_key.setdefault(
            (row["algorithm"], row["episode_uid"]), []
        ).append(row)
    for summary in summary_rows:
        steps = sorted(
            by_key.get(
                (summary["algorithm"], summary["episode_uid"]), []
            ),
            key=lambda row: row["step"],
        )
        e2e_ok = all(
            abs(
                float(row["e2e_ms"])
                - float(row["extraction_ms"])
                - float(row["communication_ms"])
                - float(row["inference_ms"])
            ) <= 1e-9
            for row in steps if row["step"] > 0
        )
        model_ok = all(
            row["model_type"] in MEC_MODELS
            for row in steps if row["step"] > 0
        )
        reward_ok = abs(
            float(summary["cumulative_reward_D_last"])
            - float(summary["reward_D"])
        ) <= 1e-8
        counts_ok = (
            int(summary["small_steps"])
            + int(summary["transformer_m_steps"])
            == int(summary["steps"])
        )
        rows.append({
            "algorithm": summary["algorithm"],
            "seed": summary["seed"],
            "episode_uid": summary["episode_uid"],
            "role": summary["role"],
            "selection_status": summary["selection_status"],
            "steps_recorded": len(steps),
            "steps": summary["steps"],
            "e2e_components_ok": bool(e2e_ok),
            "model_type_ok": bool(model_ok),
            "reward_identity_ok": bool(reward_ok),
            "model_usage_accounted": bool(counts_ok),
            "passed": bool(
                len(steps) == int(summary["steps"]) + 1
                and e2e_ok and model_ok and reward_ok and counts_ok
            ),
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tasks", type=int, default=MAX_TASKS)
    args = parser.parse_args(argv)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    device, encoder, backends, latency = load_stage2s_runtime(args.device)
    cache = make_scene_cache(
        encoder, device, capacity=max(64, len(bundle["dev_specs"]))
    )
    specs = bundle["dev_specs"][: int(args.max_tasks)]
    all_steps = []
    selected_rows = []
    pool_rows = []
    for algorithm in ALGORITHMS:
        seed, result = _best_seed(algorithm)
        agent, result, checkpoint = _load_agent(algorithm, seed, device)
        internal = {}
        pool = []
        print(
            f"[traj] {algorithm.upper()} best seed={seed} "
            f"bestDevE2E={result['best_dev_e2e_ms']:.2f} ms",
            flush=True,
        )
        for index, spec in enumerate(specs, start=1):
            result_ep, runner, rewards, cumulative, diagnostics = (
                _run_episode(
                    bundle, spec, encoder, backends, cache, device,
                    latency, algorithm, agent,
                )
            )
            if abs(diagnostics["return"] - result_ep["reward_D"]) > 1e-8:
                raise RuntimeError(
                    f"{spec['episode_uid']} reward identity failed"
                )
            stats = latency_stats(runner.step_records)
            pool_row = {
                "algorithm": algorithm,
                "seed": int(seed),
                "policy": f"{algorithm}_s{seed}",
                "episode_uid": spec["episode_uid"],
                "band": spec["band"],
                "success": int(bool(result_ep["success"])),
                "steps": int(result_ep["steps"]),
                "mean_e2e_ms": float(result_ep["mean_episode_e2e_ms"]),
                "reward_D": float(result_ep["reward_D"]),
                "final_ndtw": float(result_ep["final_ndtw"]),
                "spl": float(result_ep["spl"]),
                "osr": int(bool(result_ep["osr"])),
                "small_ratio": float(stats["small_ratio"]),
                "transformer_m_ratio": float(
                    stats["transformer_m_ratio"]
                ),
                "mean_inference_ms": float(stats["mean_inference_ms"]),
                "mean_communication_ms": float(
                    stats["mean_communication_ms"]
                ),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
            pool.append(pool_row)
            internal[spec["episode_uid"]] = {
                "spec": spec,
                "result": result_ep,
                "runner": runner,
                "rewards": rewards,
                "cumulative": cumulative,
                "checkpoint": checkpoint,
            }
            if index % 10 == 0:
                print(
                    f"[traj] {algorithm.upper()} {index}/{len(specs)}",
                    flush=True,
                )
        pool_rows.extend(pool)
        selected = _select_episodes(pool)
        for selection in selected:
            uid = selection["episode_uid"]
            item = internal[uid]
            final_summary = _summary_row(
                algorithm, seed, item["checkpoint"], item["spec"],
                item["result"], item["runner"], item["rewards"],
                item["cumulative"], selection,
            )
            selected_rows.append(final_summary)
            all_steps.extend(_step_rows(
                bundle, algorithm, seed, item["checkpoint"], item["spec"],
                item["result"], item["runner"], item["rewards"],
                item["cumulative"],
            ))
    public_summary = selected_rows
    step_fields = [
        "stage", "algorithm", "seed", "policy", "checkpoint",
        "checkpoint_sha256", "episode_uid", "band", "success", "step",
        "x_m", "y_m", "z_m", "heading_rad", "node_index", "node_id",
        "model_type", "action", "action_name", "communication_ms",
        "inference_ms", "extraction_ms", "e2e_ms", "reward_D_step",
        "cumulative_reward_D", "running_mean_e2e_ms",
        "distance_to_goal_m", "start_x_m", "start_y_m", "goal_x_m",
        "goal_y_m", "path_length_m", "mean_e2e_ms", "reward_D",
        "small_ratio", "transformer_m_ratio",
    ]
    write_csv(
        EVIDENCE_DIR / "trajectory_steps.csv", all_steps,
        fieldnames=step_fields,
    )
    write_csv(
        EVIDENCE_DIR / "trajectory_summary.csv", public_summary,
    )
    write_csv(
        EVIDENCE_DIR / "selected_tasks.csv",
        [{
            key: row[key] for key in (
                "algorithm", "seed", "episode_uid", "band", "success",
                "role", "requested_outcome", "selection_status",
                "selection_reason", "steps", "path_length_m",
                "mean_e2e_ms", "reward_D", "small_ratio",
            )
        } for row in selected_rows],
    )
    write_csv(
        EVIDENCE_DIR / "trajectory_dev_pool.csv", pool_rows,
    )
    selection_rows = _selection_summary(pool_rows)
    selection_fields = [
        "algorithm", "group", "episodes", "mean_small_ratio",
        "success_rate", "mean_ndtw", "mean_e2e_ms", "mean_steps",
        "correlation",
    ]
    write_csv(
        EVIDENCE_DIR / "model_selection_summary.csv",
        selection_rows, fieldnames=selection_fields,
    )
    audit = _audit(selected_rows, all_steps)
    write_csv(EVIDENCE_DIR / "trajectory_audit.csv", audit)
    manifest = {
        "stage": STAGE,
        "created_utc": utc_now(),
        "deployment_sha256": bundle["deployment"]["deployment_sha256"],
        "checkpoint_sha256": dict(MEC_CHECKPOINT_SHA256),
        "source_hashes": source_hashes(),
        "selected": [{
            "algorithm": row["algorithm"],
            "seed": row["seed"],
            "episode_uid": row["episode_uid"],
            "band": row["band"],
            "success": row["success"],
            "role": row["role"],
            "selection_status": row["selection_status"],
            "selection_reason": row["selection_reason"],
        } for row in selected_rows],
        "audit_passed": all(row["passed"] for row in audit),
        "final_test_read": False,
    }
    atomic_write_json(EVIDENCE_DIR / "trajectory_manifest.json", manifest)
    print(
        f"[traj] selected {len(selected_rows)} trajectories, "
        f"{len(all_steps)} step rows, audit="
        f"{'PASS' if manifest['audit_passed'] else 'FAIL'}",
        flush=True,
    )
    print("STAGE2S_TRAJECTORY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
