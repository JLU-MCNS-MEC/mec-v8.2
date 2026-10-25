"""Collect real Dev-60 UAV trajectories for the four Stage-2r conditions.

For each deployment (Small / Transformer-M) six representative tasks are
selected deterministically: success/failure x short/medium/long.  If a
stratum is empty the closest available sample is substituted and explicitly
labelled.  DQN, PPO and a Greedy-Min-E2E control are then run in full
closed loop on exactly those tasks.

Because navigation is produced by the frozen MEC model and association only
chooses the serving node, DQN and PPO are expected to have identical paths.
The collector audits this with action/path/node hashes instead of assuming it.

Usage:
    python v83_spatial_stage2/stage2r_10node_small_vs_tm/trajectories.py
"""

from __future__ import annotations

import argparse
import hashlib
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
from v83_spatial_stage2.stage2r_10node_small_vs_tm.common import (  # noqa: E402
    ACTION_DIM,
    ALGORITHMS,
    ALGORITHM_LABELS,
    EVIDENCE_DIR,
    FORMAL_SEEDS,
    MEC_MODELS,
    MODEL_LABELS,
    STAGE,
    STATE_DIM,
    STATE_MODE,
    load_bundle,
    run_dir,
)
from v83_spatial_stage2.stage2r_10node_small_vs_tm.runtime import (  # noqa: E402
    LaneEvaluator,
    build_lanes,
    load_stage2r_runtime,
    make_scene_cache,
    runner_for,
)

BANDS = ("short", "medium", "long")
ROLES = tuple(
    f"{outcome}_{band}"
    for band in BANDS for outcome in ("success", "failure")
)
SELECTION_RULE = (
    "Within each band and outcome, sort by (steps, episode_uid) and take the "
    "lower median rank (n-1)//2.  If the stratum is empty, substitute the "
    "highest-nDTW failure for a missing success or the lowest-nDTW success "
    "for a missing failure, and label it explicitly."
)
CONTROL = "greedy_min_e2e"
CONTROL_LABEL = "Greedy-Min-E2E"


def _action_hash(actions):
    text = ",".join(str(int(action)) for action in actions)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _node_hash(nodes):
    text = ",".join(str(int(node)) for node in nodes)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _path_hash(coords):
    array = np.asarray(coords, dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _path_length(coords):
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(coords[:, :2], axis=0), axis=1).sum())


def _best_seed(model, algorithm):
    candidates = []
    for seed in FORMAL_SEEDS:
        result = read_json(
            run_dir(model, algorithm, seed) / "training_result.json"
        )
        if result is None:
            continue
        value = result.get("best_dev_e2e_ms")
        if value is None:
            continue
        candidates.append((float(value), int(seed), result))
    if not candidates:
        raise FileNotFoundError(
            f"no finished runs for {model}/{algorithm}"
        )
    candidates.sort()
    return candidates[0][1], candidates[0][2]


def _load_agent(model, algorithm, seed, device):
    result = read_json(
        run_dir(model, algorithm, seed) / "training_result.json"
    )
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
    if payload.get("stage") != STAGE or payload.get("model") != model:
        raise ValueError(f"{checkpoint} is not a Stage-2r {model} checkpoint")
    return agent, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_dev_e2e_ms": float(result["best_dev_e2e_ms"]),
        "best_dev_episode": int(result["best_dev_episode"]),
    }


def _lane_rows_from_eval(model):
    path = EVIDENCE_DIR / "dev_lane_outcomes.csv"
    if not path.is_file():
        return None
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["model"] == model]
    for row in selected:
        row["success"] = int(row["success"])
        row["osr"] = int(row["osr"])
        row["steps"] = int(row["steps"])
        row["spl"] = float(row["spl"])
        row["ndtw"] = float(row["ndtw"])
    return selected


def _select_from_lane_rows(rows):
    selected = []
    for band in BANDS:
        for outcome in (True, False):
            outcome_name = "success" if outcome else "failure"
            candidates = [
                row for row in rows
                if row["band"] == band
                and bool(row["success"]) is bool(outcome)
            ]
            status = "exact"
            reason = ""
            if not candidates:
                status = f"substituted_missing_{outcome_name}"
                if outcome:
                    candidates = [
                        row for row in rows
                        if row["band"] == band
                        and not bool(row["success"])
                    ]
                    candidates.sort(key=lambda row: (
                        -float(row["ndtw"]), int(row["steps"]),
                        str(row["episode_uid"]),
                    ))
                    reason = (
                        "No successful task in this band; substituted the "
                        "highest-nDTW failure."
                    )
                else:
                    candidates = [
                        row for row in rows
                        if row["band"] == band and bool(row["success"])
                    ]
                    candidates.sort(key=lambda row: (
                        float(row["ndtw"]), int(row["steps"]),
                        str(row["episode_uid"]),
                    ))
                    reason = (
                        "No failed task in this band; substituted the "
                        "lowest-nDTW success."
                    )
            if not candidates:
                raise RuntimeError(
                    f"no candidate at all for {band}/{outcome_name}"
                )
            ordered = sorted(
                candidates,
                key=lambda row: (int(row["steps"]),
                                 str(row["episode_uid"])),
            )
            chosen = dict(ordered[(len(ordered) - 1) // 2])
            chosen.update({
                "role": f"{outcome_name}_{band}",
                "requested_outcome": outcome_name,
                "selection_status": status,
                "selection_reason": reason,
            })
            selected.append(chosen)
    return selected


def _run_episode(bundle, spec, encoder, backends, scene_cache, device,
                 policy, agent=None):
    runner = runner_for(
        bundle, spec, encoder, backends, scene_cache, device
    )
    while True:
        if policy in ("dqn", "ppo"):
            state = runner.step_data["state"]
            mask = runner.step_data["mask"]
            if policy == "dqn":
                from v83_spatial_stage2.stage2k_state import masked_argmax
                action = int(masked_argmax(agent.q_values(state), mask))
            else:
                action = int(agent.greedy_action(state, mask))
        elif policy == CONTROL:
            metrics = runner.step_data["latency_metrics"]
            mask = runner.step_data["mask"]
            indices = np.where(np.asarray(mask, bool))[0]
            action = int(min(
                indices,
                key=lambda i: (float(metrics[i]["total_ms"]), int(i)),
            ))
        else:
            raise ValueError(policy)
        transition = runner.apply(action)
        if transition["done"]:
            break
    return runner.finalize(policy), runner


def _step_rows(condition, model, algorithm, seed, spec, selection,
               result, runner, policy):
    trajectory = np.asarray(result["trajectory"], dtype=np.float64)
    records = runner.step_records
    rows = []
    for index, position in enumerate(trajectory):
        if index == 0:
            record = {}
            distance = float(np.linalg.norm(
                np.asarray(spec["start"], dtype=float)
                - np.asarray(spec["goal"], dtype=float)
            ))
        else:
            record = records[index - 1]
            distance = float(record["distance_to_goal_m"])
        if index < len(trajectory) - 1:
            delta = trajectory[index + 1, :2] - position[:2]
            heading = float(np.arctan2(delta[1], delta[0]))
        elif index > 0:
            delta = position[:2] - trajectory[index - 1, :2]
            heading = float(np.arctan2(delta[1], delta[0]))
        else:
            heading = 0.0
        rows.append({
            "condition": condition,
            "model": model,
            "deployment": MODEL_LABELS[model],
            "algorithm": algorithm,
            "policy": policy,
            "seed": int(seed),
            "episode_uid": spec["episode_uid"],
            "role": selection["role"],
            "requested_outcome": selection["requested_outcome"],
            "selection_status": selection["selection_status"],
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
            "action": int(record.get("action", -1)) if record else -1,
            "action_name": record.get("action_name", "START")
            if record else "START",
            "e2e_ms": float(record.get("e2e_ms", 0.0))
            if record else 0.0,
            "running_mean_e2e_ms": float(
                record.get("running_mean_e2e_ms", 0.0)
            ) if record else 0.0,
            "distance_to_goal_m": distance,
        })
    return rows


def _summary_row(condition, model, algorithm, seed, spec, selection, result,
                 runner, policy, meta):
    trajectory = np.asarray(result["trajectory"], dtype=np.float64)
    actions = [int(row["action"]) for row in runner.step_records]
    nodes = [int(row["node_index"]) for row in runner.step_records]
    goal = np.asarray(spec["goal"], dtype=np.float64)
    final = trajectory[-1, :2]
    return {
        "condition": condition,
        "model": model,
        "deployment": MODEL_LABELS[model],
        "algorithm": algorithm,
        "algorithm_label": "Control" if policy == CONTROL
        else ALGORITHM_LABELS[algorithm],
        "policy": policy,
        "seed": int(seed),
        "episode_uid": spec["episode_uid"],
        "role": selection["role"],
        "requested_outcome": selection["requested_outcome"],
        "selection_status": selection["selection_status"],
        "selection_reason": selection["selection_reason"],
        "band": spec["band"],
        "success": int(result["success"]),
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
        "reward_D": float(1.0 - result["mean_episode_e2e_ms"] / 100.0),
        "action_sha256": _action_hash(actions),
        "node_sha256": _node_hash(nodes),
        "path_sha256": _path_hash(trajectory),
        "checkpoint": meta.get("checkpoint", ""),
        "checkpoint_sha256": meta.get("checkpoint_sha256", ""),
        "best_dev_e2e_ms": meta.get("best_dev_e2e_ms"),
        "best_dev_episode": meta.get("best_dev_episode"),
        "final_test_read": False,
    }


def _consistency(summary_rows):
    audit_rows = []
    by_model = {}
    for row in summary_rows:
        by_model.setdefault(row["model"], []).append(row)
    for model in MEC_MODELS:
        model_rows = by_model.get(model, [])
        uids = sorted({row["episode_uid"] for row in model_rows})
        for uid in uids:
            subset = [row for row in model_rows if row["episode_uid"] == uid]
            reference = next(
                (row for row in subset if row["policy"] == CONTROL),
                subset[0],
            )
            for row in sorted(subset, key=lambda item: item["policy"]):
                audit_rows.append({
                    "model": model,
                    "deployment": MODEL_LABELS[model],
                    "episode_uid": uid,
                    "role": row["role"],
                    "reference_policy": reference["policy"],
                    "policy": row["policy"],
                    "action_identical": int(
                        row["action_sha256"] == reference["action_sha256"]
                    ),
                    "path_identical": int(
                        row["path_sha256"] == reference["path_sha256"]
                    ),
                    "success_identical": int(
                        row["success"] == reference["success"]
                    ),
                    "steps_identical": int(
                        row["steps"] == reference["steps"]
                    ),
                    "node_choices_differ": int(
                        row["node_sha256"] != reference["node_sha256"]
                    ),
                    "delta_path_length_m": float(
                        row["path_length_m"] - reference["path_length_m"]
                    ),
                    "delta_E2E_ms": float(
                        row["mean_e2e_ms"] - reference["mean_e2e_ms"]
                    ),
                    "delta_RewardD": float(
                        row["reward_D"] - reference["reward_D"]
                    ),
                })
    all_path = all(row["path_identical"] for row in audit_rows)
    all_action = all(row["action_identical"] for row in audit_rows)
    all_success = all(row["success_identical"] for row in audit_rows)
    any_node = any(row["node_choices_differ"] for row in audit_rows)
    payload = {
        "stage": STAGE,
        "created_utc": utc_now(),
        "reference": CONTROL_LABEL,
        "rows": len(audit_rows),
        "all_paths_identical": bool(all_path),
        "all_actions_identical": bool(all_action),
        "all_success_identical": bool(all_success),
        "any_node_choices_differ": bool(any_node),
        "interpretation": (
            "Within a deployment the frozen MEC model produces the same "
            "navigation action sequence for every association policy.  "
            "DQN/PPO can differ from the control only in node choices, E2E "
            "and RewardD; path, success and steps must be identical.  "
            "Across deployments the MEC model differs, so paths may differ."
        ),
        "final_test_read": False,
    }
    return audit_rows, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    all_steps = []
    all_summary = []
    selected_rows = []
    for model in MEC_MODELS:
        print(f"[traj] {model}: loading bundle", flush=True)
        bundle = load_bundle(model)
        lane_rows = _lane_rows_from_eval(model)
        if lane_rows is None:
            raise SystemExit(
                "run eval.py first: dev_lane_outcomes.csv is missing"
            )
        by_uid = {str(item["episode_uid"]): item for item in bundle["dev_specs"]}
        selections = _select_from_lane_rows(lane_rows)
        for selection in selections:
            uid = str(selection["episode_uid"])
            if uid not in by_uid:
                raise ValueError(f"{uid} is not a Dev task")
            selected_rows.append({
                "model": model,
                "deployment": MODEL_LABELS[model],
                "role": selection["role"],
                "requested_outcome": selection["requested_outcome"],
                "band": selection["band"],
                "episode_uid": uid,
                "lane_success": int(selection["success"]),
                "lane_steps": int(selection["steps"]),
                "lane_ndtw": float(selection["ndtw"]),
                "selection_status": selection["selection_status"],
                "selection_reason": selection["selection_reason"],
            })
        device, encoder, backends = load_stage2r_runtime(
            args.device, model
        )
        scene_cache = make_scene_cache(
            encoder, device, capacity=max(64, len(bundle["dev_specs"]))
        )
        lanes = build_lanes(
            bundle["dev_specs"], bundle["topology"], encoder, backends,
            bundle["quality_prior"], bundle["gab_config"], device,
            scene_cache, verbose=False,
        )
        # Verify the eval CSV and the rebuilt lanes agree.
        rebuilt = {
            lane.episode_uid: int(lane.success) for lane in lanes
        }
        for selection in selections:
            uid = str(selection["episode_uid"])
            if rebuilt.get(uid) != int(selection["success"]):
                raise RuntimeError(
                    f"lane success mismatch for {model}/{uid}"
                )
        # Control plus the best checkpoint of each algorithm.
        policies = [(CONTROL, None, None)]
        for algorithm in ALGORITHMS:
            seed, result = _best_seed(model, algorithm)
            agent, meta = _load_agent(model, algorithm, seed, device)
            policies.append((algorithm, seed, (agent, meta)))
        for policy, seed, payload in policies:
            if policy == CONTROL:
                agent = None
                meta = {}
                label = CONTROL
                condition = f"{MODEL_LABELS[model]}|{CONTROL_LABEL}"
                display_seed = -1
            else:
                agent, meta = payload
                label = policy
                condition = (
                    f"{MODEL_LABELS[model]}|{ALGORITHM_LABELS[policy]}"
                )
                display_seed = int(seed)
            for selection in selections:
                spec = by_uid[str(selection["episode_uid"])]
                print(
                    f"[traj] {model}/{policy}/{selection['role']}: "
                    f"{spec['episode_uid']}",
                    flush=True,
                )
                result, runner = _run_episode(
                    bundle, spec, encoder, backends, scene_cache, device,
                    policy, agent=agent,
                )
                if bool(result["success"]) != bool(lane_rebuilt := rebuilt[
                    spec["episode_uid"]
                ]):
                    raise RuntimeError(
                        f"closed-loop success {result['success']} != lane "
                        f"{lane_rebuilt} for {model}/{spec['episode_uid']}"
                    )
                all_steps.extend(_step_rows(
                    condition, model, policy if policy != CONTROL
                    else "heuristic", display_seed, spec, selection,
                    result, runner, policy,
                ))
                all_summary.append(_summary_row(
                    condition, model, policy if policy != CONTROL
                    else "heuristic", display_seed, spec, selection, result,
                    runner, policy, meta,
                ))
    if not all_summary:
        raise RuntimeError("no trajectories collected")
    write_csv(EVIDENCE_DIR / "trajectory_steps.csv", all_steps)
    write_csv(EVIDENCE_DIR / "trajectory_summary.csv", all_summary)
    write_csv(EVIDENCE_DIR / "selected_tasks.csv", selected_rows)
    audit_rows, audit = _consistency(all_summary)
    write_csv(EVIDENCE_DIR / "trajectory_consistency_audit.csv", audit_rows)
    atomic_write_json(
        EVIDENCE_DIR / "trajectory_consistency_audit.json", audit
    )
    atomic_write_json(EVIDENCE_DIR / "trajectory_manifest.json", {
        "stage": STAGE,
        "created_utc": utc_now(),
        "selection_rule": SELECTION_RULE,
        "tasks": len(all_summary),
        "conditions": sorted({row["condition"] for row in all_summary}),
        "audit": audit,
        "final_test_read": False,
    })
    print(f"[traj] wrote {len(all_summary)} trajectories", flush=True)
    print("STAGE2R_TRAJECTORY_COMPLETE", flush=True)
    return all_summary


if __name__ == "__main__":
    main()
