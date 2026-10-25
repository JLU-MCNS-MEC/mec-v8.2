"""Stage-2q: extract and audit real Dev-60 UAV trajectories.

The Stage-2n/2o/2p repositories contain per-episode metrics and action
hashes, but no per-step UAV coordinates.  This collector rebuilds the
frozen Dev-60 MEC lanes, replays every learned association checkpoint on
those lanes, and additionally runs a full closed-loop audit on six
representative Dev tasks (one success and one failure in each band).

Nothing here reads Test-400, retrains a model, or overwrites Stage-2n/2o/2p
artefacts.  The only environment rollouts are the 60 Dev lanes plus the
selected closed-loop episodes.

Outputs (written next to this script unless ``--out-dir`` is given):

* ``trajectory_steps.csv``       per-step real coordinates/actions/E2E
* ``trajectory_summary.csv``     one row per selected trajectory
* ``trajectory_invariance.csv``  cross-method action/coordinate audit
* ``dev60_task_summary.csv``     Dev-60 lane context
* ``selected_tasks.csv``         deterministic six-task selection
* ``checkpoint_inventory.csv``   checkpoint provenance and weight hashes
* ``city_geometry.csv``          real fixed-city landmark geometry
* ``audit_summary.json``         aggregated audit numbers
* ``run_manifest.json``          provenance and reproducibility checks

Usage:
    /home/ubuntu5/miniconda3/envs/DRL/bin/python -u \
        v83_spatial_stage2/stage2q_trajectory_audit/collect_trajectories.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENV  # noqa: E402
from env import (  # noqa: E402
    ACTION_NAMES,
    FORWARD,
    LEFT,
    RIGHT,
    STOP,
    wrap_angle,
)
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
from v83_spatial_stage2.stage2l_fixed_city import (  # noqa: E402
    build_city,
    FixedCityEnv,
)
from v83_spatial_stage2.stage2n_common import (  # noqa: E402
    DEV_EPISODES,
    MAP_SEED,
    MEC_CHECKPOINT,
    MEC_CHECKPOINT_SHA256,
    RUNS_DIR as STAGE2N_RUNS_DIR,
    STATE_DIM,
    STATE_MODE,
    latency_return_from_mean,
    load_task_pool,
    prepare_runner_spec,
    summarize_episode_rows,
)
from v83_spatial_stage2.stage2n_lanes import (  # noqa: E402
    Lane,
    LaneEvaluator,
    build_lane,
    build_lanes,
)
from v83_spatial_stage2.stage2n_runtime import (  # noqa: E402
    LatencyOnlyEpisodeRunner,
    load_stage2n_runtime,
    make_stage2n_scene_cache,
)
from v83_spatial_stage2.stage2n_train import (  # noqa: E402
    load_bundle,
    verify_frozen_protocol,
)

OUT_DIR = Path(__file__).resolve().parent
STAGE = "stage2q_trajectory_audit"
STAGE2O_ROOT = Path("runs/v83_spatial_stage2/stage2o")
STAGE2P_ROOT = Path("runs/v83_spatial_stage2/stage2p")
STAGE2N_EVAL_PER_EPISODE = Path(
    "runs/v83_spatial_stage2/stage2n_latency_only/evaluation/dev/per_episode.csv"
)
STAGE2N_EVAL_SUMMARY = Path(
    "runs/v83_spatial_stage2/stage2n_latency_only/evaluation/dev/summary.json"
)

# Guard rails copied from the frozen Stage-2n protocol.  A mismatch means the
# wrong task pool or deployment is being read.
EXPECTED_DEV_TASK_FILE_SHA256 = (
    "1b0518b59f86c58c896d63a240629bedb78065fb059194d0fe123d658dc0653a"
)
EXPECTED_DEV_TASKS_SHA256 = (
    "aefde2aadaf443d32c70faebd74624431d165f6ef1b27d0fe3f1480572ade41d"
)
EXPECTED_DEPLOYMENT_FILE_SHA256 = (
    "11de01122d853d4e32a62965d92f4fadb8c9fc988d781907ad36823d3d19748c"
)
EXPECTED_DEPLOYMENT_CANONICAL_SHA256 = (
    "4c0950cf6e438cff41454eee6daefb831fa984f425ec1a41cf3974257c6d1148"
)
EXPECTED_CITY_FINGERPRINT = (
    "392500aa0df1d87b341f2fbd3df2fe4f2dc0405624a1ecc91cd33eca616bb182"
)
RANDOM_RNG_SEED = 20261022
COORD_TOLERANCE_M = 1.0e-9
PATH_RECONSTRUCTION_TOLERANCE_M = 1.0e-3
SELECTION_RULE = (
    "Within each band and outcome (success/failure), sort by (steps, "
    "episode_index) and take the lower median rank (n-1)//2.  No Test sample "
    "is used; if a stratum is empty it is reported as missing."
)

STAGE2N_METHODS = tuple(
    f"2n-{algorithm}-s{seed}"
    for algorithm in ("dqn", "ppo")
    for seed in (0, 1, 2)
)
STAGE2O_METHODS = tuple(
    f"2o-{scheme}-{algorithm}-s0"
    for scheme in ("B", "D", "one")
    for algorithm in ("dqn", "ppo")
)
STAGE2P_METHODS = tuple(
    f"2p-{scheme}-ppo-s{seed}"
    for scheme in ("base", "D")
    for seed in (0, 1, 2)
)
LEARNED_METHODS = STAGE2N_METHODS + STAGE2O_METHODS + STAGE2P_METHODS
CONTROL_METHODS = ("control-greedy-min-e2e", "control-random")
ALL_METHODS = LEARNED_METHODS + CONTROL_METHODS


@dataclass
class PolicyRecord:
    method_id: str
    method_family: str
    stage: str
    algorithm: str
    seed: int
    reward_scheme: str
    method_title: str
    run_dir: str
    checkpoint_type: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    training_result: str
    training_result_sha256: str
    best_dev_e2e_ms: float
    best_dev_episode: int
    validation: dict = field(default_factory=dict)
    payload_stage: str = ""
    payload_reward_scheme: str = ""
    model_state_sha256: str = ""
    last_pt_path: str = ""
    last_pt_sha256: str = ""
    final_test_read: bool = False
    notes: str = ""
    agent: object = None

    def public(self) -> dict:
        payload = dict(self.__dict__)
        payload.pop("agent", None)
        return payload


def _repo_path(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _rel(path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _action_sha256(actions: Sequence[int]) -> str:
    text = ",".join(str(int(action)) for action in actions)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _node_sha256(nodes: Sequence[int]) -> str:
    text = ",".join(str(int(node)) for node in nodes)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _coord_sha256(coords) -> str:
    array = np.asarray(coords, dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _model_state_sha256(state_dict: Mapping) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validated_pose(pose) -> np.ndarray:
    array = np.asarray(pose, dtype=np.float32).reshape(-1)
    if array.size != 3:
        raise ValueError(f"pose must have three values, got {array.size}")
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite pose {array}")
    return array.copy()


def advance_pose(pose, action: int) -> np.ndarray:
    """Apply one real environment action to a [x, y, yaw] pose.

    This mirrors ``ToyUAVEnv.step`` exactly: Forward moves 5 m along the
    current heading, Left/Right turn 15 degrees, Stop holds position, then
    x/y are clipped to the world and yaw is wrapped.
    """
    pose = _validated_pose(pose)
    action = int(action)
    x, y, yaw = (float(pose[0]), float(pose[1]), float(pose[2]))
    if action == FORWARD:
        x += float(ENV.forward_step_m) * math.cos(yaw)
        y += float(ENV.forward_step_m) * math.sin(yaw)
    elif action == LEFT:
        yaw = wrap_angle(yaw + math.radians(float(ENV.turn_step_deg)))
    elif action == RIGHT:
        yaw = wrap_angle(yaw - math.radians(float(ENV.turn_step_deg)))
    elif action != STOP:
        raise ValueError(f"unknown action {action}")
    x = min(max(x, 0.0), float(ENV.world_size_m))
    y = min(max(y, 0.0), float(ENV.world_size_m))
    return np.asarray([x, y, yaw], dtype=np.float32)


def reconstruct_path(poses, actions) -> np.ndarray:
    """Return the real post-action path (T+1, 3) from decision poses/actions."""
    poses = np.asarray(poses, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"poses shape {poses.shape} != (T, 3)")
    if actions.shape != (len(poses),):
        raise ValueError(f"actions shape {actions.shape} != ({len(poses)},)")
    path = [poses[0].copy()]
    for index, action in enumerate(actions):
        path.append(advance_pose(poses[index], int(action)))
    return np.asarray(path, dtype=np.float32)


def _path_length(path_xy) -> float:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if len(path_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum())


def _max_coordinate_deviation(a, b) -> tuple[float, float, int]:
    """Return (max xy deviation, max heading deviation, common length)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    common = min(len(a), len(b))
    if common == 0:
        return 0.0, 0.0, 0
    xy = np.linalg.norm(a[:common, :2] - b[:common, :2], axis=1)
    heading = np.abs(wrap_angle(a[:common, 2] - b[:common, 2]))
    return float(xy.max()), float(heading.max()), int(common)


def _discover_policies() -> list[PolicyRecord]:
    records: list[PolicyRecord] = []

    def add(
        method_id: str,
        stage: str,
        algorithm: str,
        seed: int,
        reward_scheme: str,
        method_title: str,
        run_dir: Path,
    ) -> None:
        run_dir = _repo_path(run_dir)
        result_path = run_dir / "training_result.json"
        result = read_json(result_path)
        if result is None:
            raise FileNotFoundError(f"{result_path} missing")
        if result.get("final_test_read") is not False:
            raise ValueError(f"{result_path} claims a Final Test read")
        checkpoint = _repo_path(result["best_checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{checkpoint} missing")
        if int(result.get("seed", seed)) != int(seed):
            raise ValueError(f"{result_path} seed mismatch")
        if str(result.get("algorithm", algorithm)) != str(algorithm):
            raise ValueError(f"{result_path} algorithm mismatch")
        last_pt = run_dir / "checkpoints" / "last.pt"
        records.append(PolicyRecord(
            method_id=method_id,
            method_family="learned",
            stage=stage,
            algorithm=algorithm,
            seed=int(seed),
            reward_scheme=reward_scheme,
            method_title=method_title,
            run_dir=_rel(run_dir),
            checkpoint_type="best.pt (lowest Dev mean E2E)",
            checkpoint_path=_rel(checkpoint),
            checkpoint_sha256=sha256_file(checkpoint),
            checkpoint_size_bytes=int(checkpoint.stat().st_size),
            training_result=_rel(result_path),
            training_result_sha256=sha256_file(result_path),
            best_dev_e2e_ms=float(result["best_dev_e2e_ms"]),
            best_dev_episode=int(result["best_dev_episode"]),
            validation=dict(result.get("validation", {})),
            final_test_read=bool(result.get("final_test_read", False)),
            last_pt_path=_rel(last_pt) if last_pt.is_file() else "",
            last_pt_sha256=sha256_file(last_pt) if last_pt.is_file() else "",
        ))

    for algorithm in ("dqn", "ppo"):
        for seed in (0, 1, 2):
            add(
                method_id=f"2n-{algorithm}-s{seed}",
                stage="stage2n",
                algorithm=algorithm,
                seed=seed,
                reward_scheme="original",
                method_title=f"2n {algorithm.upper()} s{seed}",
                run_dir=STAGE2N_RUNS_DIR / f"{algorithm}_s{seed}",
            )
    for scheme in ("B", "D", "one"):
        for algorithm in ("dqn", "ppo"):
            add(
                method_id=f"2o-{scheme}-{algorithm}-s0",
                stage="stage2o",
                algorithm=algorithm,
                seed=0,
                reward_scheme=scheme,
                method_title=f"2o {scheme} {algorithm.upper()} s0",
                run_dir=STAGE2O_ROOT / scheme / f"{algorithm}_s0",
            )
    for scheme, label in (("baseline", "base"), ("D", "D")):
        for seed in (0, 1, 2):
            add(
                method_id=f"2p-{label}-ppo-s{seed}",
                stage="stage2p",
                algorithm="ppo",
                seed=seed,
                reward_scheme=scheme,
                method_title=f"2p {label} PPO s{seed}",
                run_dir=STAGE2P_ROOT / scheme / f"ppo_s{seed}",
            )
    for method_id, title in zip(
        CONTROL_METHODS, ("Greedy-Min-E2E", "Random")
    ):
        records.append(PolicyRecord(
            method_id=method_id,
            method_family="control",
            stage="control",
            algorithm="heuristic",
            seed=0,
            reward_scheme=method_id.replace("control-", ""),
            method_title=title,
            run_dir="",
            checkpoint_type="none",
            checkpoint_path="",
            checkpoint_sha256="",
            checkpoint_size_bytes=0,
            training_result="",
            training_result_sha256="",
            best_dev_e2e_ms=float("nan"),
            best_dev_episode=-1,
            notes=(
                "Random uses RNG seed "
                f"{RANDOM_RNG_SEED}+episode_index, matching the Stage-2n "
                "random baseline."
            ) if method_id == "control-random" else "Deterministic greedy control.",
        ))
    if tuple(record.method_id for record in records) != ALL_METHODS:
        raise RuntimeError("policy discovery order mismatch")
    return records


def _load_agent(record: PolicyRecord, device):
    if record.method_family != "learned":
        return None, {}
    if record.algorithm == "dqn":
        agent = DQNAgentV2(
            device,
            seed=record.seed,
            gamma=1.0,
            state_dim=STATE_DIM,
            state_mode=STATE_MODE,
        )
    elif record.algorithm == "ppo":
        agent = PPOAgent(
            device,
            seed=record.seed,
            state_dim=STATE_DIM,
            state_mode=STATE_MODE,
            gamma=1.0,
        )
    else:
        raise ValueError(f"unknown algorithm {record.algorithm}")
    payload = agent.load_model(_repo_path(record.checkpoint_path))
    payload_stage = str(payload.get("stage", ""))
    if payload_stage != record.stage:
        raise ValueError(
            f"{record.method_id}: payload stage {payload_stage!r} != "
            f"{record.stage!r}"
        )
    if str(payload.get("algorithm", record.algorithm)) != record.algorithm:
        raise ValueError(f"{record.method_id}: payload algorithm mismatch")
    if int(payload.get("seed", record.seed)) != record.seed:
        raise ValueError(f"{record.method_id}: payload seed mismatch")
    scheme = (
        payload.get("stage2o_reward_scheme")
        or payload.get("stage2p_reward_scheme")
        or payload.get("stage2n_reward_mode")
        or ""
    )
    if record.stage == "stage2n" and payload.get(
        "stage2n_reward_mode"
    ) != "latency_only_v1":
        raise ValueError(f"{record.method_id}: not a Stage-2n checkpoint")
    record.payload_stage = payload_stage
    record.payload_reward_scheme = str(scheme)
    if payload.get("validation"):
        record.validation = dict(payload["validation"])
    record.model_state_sha256 = _model_state_sha256(payload["state_dict"])
    record.agent = agent
    return agent, payload


def _checkpoint_inventory(records: Sequence[PolicyRecord]) -> list[dict]:
    by_state: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.model_state_sha256:
            by_state[record.model_state_sha256].append(record.method_id)
    rows = []
    for record in records:
        duplicates = [
            item for item in by_state.get(record.model_state_sha256, [])
            if item != record.method_id
        ]
        notes = record.notes
        if duplicates and not notes:
            notes = (
                "Different checkpoint file SHA-256 but tensor-identical model "
                f"weights to {', '.join(duplicates)}"
            )
        rows.append({
            "method_id": record.method_id,
            "method_family": record.method_family,
            "stage": record.stage,
            "algorithm": record.algorithm,
            "seed": record.seed,
            "reward_scheme": record.reward_scheme,
            "run_dir": record.run_dir,
            "training_result": record.training_result,
            "training_result_sha256": record.training_result_sha256,
            "best_dev_e2e_ms": record.best_dev_e2e_ms,
            "best_dev_episode": record.best_dev_episode,
            "checkpoint_type": record.checkpoint_type,
            "checkpoint_path": record.checkpoint_path,
            "checkpoint_sha256": record.checkpoint_sha256,
            "checkpoint_size_bytes": record.checkpoint_size_bytes,
            "payload_stage": record.payload_stage,
            "payload_reward_scheme": record.payload_reward_scheme,
            "model_state_sha256": record.model_state_sha256,
            "weights_identical_to": ",".join(duplicates),
            "last_pt_exists": bool(record.last_pt_path),
            "last_pt_sha256": record.last_pt_sha256,
            "final_test_read": record.final_test_read,
            "notes": notes,
        })
    return rows


def _record_lane(spec: Mapping, lane: Lane, city) -> dict:
    poses = np.stack([
        _validated_pose(step.pose) for step in lane.steps
    ]).astype(np.float32)
    actions = np.asarray([step.action for step in lane.steps], dtype=np.int64)
    path = reconstruct_path(poses, actions)
    path_length = _path_length(path[:, :2])
    shortest = float(spec["start_goal_distance_m"])
    spl_recomputed = (
        shortest / max(shortest, path_length, 1e-8) if lane.success else 0.0
    )
    if abs(spl_recomputed - float(lane.spl)) > 1e-5:
        raise RuntimeError(
            f"{lane.episode_uid}: reconstructed SPL {spl_recomputed} != "
            f"lane SPL {lane.spl}"
        )
    rebuilt_next = path[1:-1, :2] if len(path) > 2 else path[:0, :2]
    lane_next = poses[1:, :2] if len(poses) > 1 else poses[:0, :2]
    if len(rebuilt_next):
        next_deviation = float(
            np.linalg.norm(rebuilt_next - lane_next, axis=1).max()
        )
        if next_deviation > PATH_RECONSTRUCTION_TOLERANCE_M:
            raise RuntimeError(
                f"{lane.episode_uid}: path reconstruction error "
                f"{next_deviation} m"
            )
    else:
        next_deviation = 0.0
    start_pose = poses[0]
    expected_start = np.asarray(
        [spec["start"][0], spec["start"][1], spec["start_yaw"]],
        dtype=np.float32,
    )
    if float(np.abs(start_pose - expected_start).max()) > 1e-5:
        raise RuntimeError(
            f"{lane.episode_uid}: lane start pose {start_pose} != spec "
            f"{expected_start}"
        )
    goal_index = int(spec["goal_index"])
    landmark = city.landmarks[goal_index]
    goal = np.asarray(spec["goal"], dtype=np.float32)
    if float(np.linalg.norm(landmark.position - goal)) > 1e-5:
        raise RuntimeError(
            f"{lane.episode_uid}: goal landmark position != task goal"
        )
    greedy_nodes = []
    greedy_e2e = []
    for step in lane.steps:
        metrics = step.latency_metrics
        indices = np.where(np.asarray(step.mask, bool))[0]
        node = int(min(
            indices,
            key=lambda i: (float(metrics[i]["total_ms"]), int(i)),
        ))
        greedy_nodes.append(node)
        greedy_e2e.append(float(metrics[node]["total_ms"]))
    greedy_mean = float(np.mean(greedy_e2e)) if greedy_e2e else 0.0
    return {
        "lane": lane,
        "spec": dict(spec),
        "poses": poses,
        "actions": actions,
        "path": path,
        "path_length_m": path_length,
        "final_pose": path[-1],
        "next_pose_deviation_m": next_deviation,
        "success": bool(lane.success),
        "steps": int(lane.episode_steps),
        "spl": float(lane.spl),
        "ndtw": float(lane.ndtw),
        "osr": bool(lane.osr),
        "action_sha256": _action_sha256(actions),
        "decision_coord_sha256": _coord_sha256(poses),
        "path_coord_sha256": _coord_sha256(path),
        "greedy_node_choices": greedy_nodes,
        "greedy_step_e2e_ms": greedy_e2e,
        "greedy_node_sha256": _node_sha256(greedy_nodes),
        "greedy_mean_e2e_ms": greedy_mean,
        "greedy_p95_e2e_ms": (
            float(np.percentile(greedy_e2e, 95)) if greedy_e2e else 0.0
        ),
        "greedy_delay_violation_rate": (
            float(np.mean(np.asarray(greedy_e2e) > 100.0))
            if greedy_e2e else 0.0
        ),
        "greedy_latency_return": latency_return_from_mean(greedy_mean),
        "goal_landmark_size_m": float(landmark.physical_size_m),
        "goal_landmark_shape": str(landmark.shape),
        "goal_landmark_is_distractor": bool(landmark.is_distractor),
    }


def _build_lane_records(bundle, device, encoder, backends, scene_cache, limit=None):
    specs = bundle["dev_specs"][:limit] if limit else bundle["dev_specs"]
    lanes = build_lanes(
        specs,
        bundle["topology"],
        encoder,
        backends,
        bundle["quality_prior"],
        bundle["gab_config"],
        device,
        scene_cache,
        verbose=True,
    )
    city = build_city(MAP_SEED)
    records = {}
    for spec, lane in zip(specs, lanes):
        if lane.episode_uid != spec["episode_uid"]:
            raise RuntimeError(
                f"lane order mismatch: {lane.episode_uid} != "
                f"{spec['episode_uid']}"
            )
        records[lane.episode_uid] = _record_lane(spec, lane, city)
    return specs, lanes, records


def _replay_all_policies(
    lanes: Sequence[Lane],
    records: Sequence[PolicyRecord],
    device,
) -> dict:
    evaluator = LaneEvaluator(lanes)
    rows: dict[tuple[str, str], dict] = {}
    for record in records:
        if record.method_family == "learned":
            agent, _ = _load_agent(record, device)
            if record.algorithm == "dqn":
                policy_rows = evaluator.evaluate_dqn(agent)
            else:
                policy_rows = evaluator.evaluate_ppo(agent)
        elif record.method_id == "control-greedy-min-e2e":
            policy_rows = evaluator.evaluate_strategy("greedy_min_e2e")
        elif record.method_id == "control-random":
            policy_rows = evaluator.evaluate_strategy(
                "random", rng_seed=RANDOM_RNG_SEED
            )
        else:
            raise ValueError(record.method_id)
        if len(policy_rows) != len(lanes):
            raise RuntimeError(
                f"{record.method_id}: {len(policy_rows)} rows for "
                f"{len(lanes)} lanes"
            )
        for lane, row in zip(lanes, policy_rows):
            if row["episode_uid"] != lane.episode_uid:
                raise RuntimeError(
                    f"{record.method_id}: row order mismatch"
                )
            if row["actions"] != [step.action for step in lane.steps]:
                raise RuntimeError(
                    f"{record.method_id}/{lane.episode_uid}: navigation "
                    "actions differ from the frozen lane"
                )
            if row["action_sha256"] != lane.action_sha256:
                raise RuntimeError(
                    f"{record.method_id}/{lane.episode_uid}: action hash "
                    "mismatch"
                )
            row = dict(row)
            row["method_id"] = record.method_id
            row["method_family"] = record.method_family
            row["stage"] = record.stage
            row["algorithm"] = record.algorithm
            row["seed"] = record.seed
            row["reward_scheme"] = record.reward_scheme
            row["checkpoint_sha256"] = record.checkpoint_sha256
            row["node_choice_sha256"] = _node_sha256(row["node_choices"])
            rows[(record.method_id, lane.episode_uid)] = row
    return rows


def _row_aggregate(rows: Sequence[Mapping]) -> dict:
    return summarize_episode_rows(list(rows)).get("overall", {})


def _historical_crosscheck(
    lanes: Sequence[Lane],
    lane_records: Mapping[str, dict],
    replay_rows: Mapping[tuple[str, str], dict],
    records: Sequence[PolicyRecord],
) -> dict:
    """Compare rebuilt lanes/replays against the published historical results."""
    per_episode = {}
    path = _repo_path(STAGE2N_EVAL_PER_EPISODE)
    if path.is_file():
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                per_episode[(row["policy"], row["episode_uid"])] = row
    stage2n_names = {
        "2n-dqn-s0": "dqn_s0",
        "2n-dqn-s1": "dqn_s1",
        "2n-dqn-s2": "dqn_s2",
        "2n-ppo-s0": "ppo_s0",
        "2n-ppo-s1": "ppo_s1",
        "2n-ppo-s2": "ppo_s2",
        "control-greedy-min-e2e": "greedy_min_e2e",
        "control-random": "random",
    }
    per_episode_checks = {}
    max_errors = defaultdict(float)
    mismatches = []
    for method_id, historical_name in stage2n_names.items():
        fields = {
            "success": None,
            "steps": None,
            "spl": 1e-9,
            "ndtw": 1e-9,
            "mean_e2e_ms": 1e-9,
            "p95_e2e_ms": 1e-9,
            "delay_violation_rate": 1e-9,
            "latency_return": 1e-9,
        }
        compared = 0
        for lane in lanes:
            key = (historical_name, lane.episode_uid)
            if key not in per_episode:
                continue
            historical = per_episode[key]
            row = replay_rows[(method_id, lane.episode_uid)]
            compared += 1
            if row["action_sha256"] != historical["action_sha256"]:
                mismatches.append(
                    f"{method_id}/{lane.episode_uid}: action_sha256"
                )
                continue
            for field_name, tolerance in fields.items():
                if tolerance is None:
                    if str(int(row[field_name])) != str(
                        int(float(historical[field_name]))
                    ):
                        mismatches.append(
                            f"{method_id}/{lane.episode_uid}: {field_name}"
                        )
                    continue
                error = abs(
                    float(row[field_name]) - float(historical[field_name])
                )
                max_errors[field_name] = max(max_errors[field_name], error)
                if error > tolerance:
                    mismatches.append(
                        f"{method_id}/{lane.episode_uid}: {field_name} "
                        f"{error:.3e}"
                    )
        per_episode_checks[method_id] = {
            "historical_policy": historical_name,
            "episodes_compared": compared,
        }
    summary_path = _repo_path(STAGE2N_EVAL_SUMMARY)
    aggregate_checks = {}
    if summary_path.is_file():
        summary = read_json(summary_path)
        for method_id, historical_name in stage2n_names.items():
            if historical_name not in summary.get("overall", {}):
                continue
            expected = summary["overall"][historical_name]
            rows = [replay_rows[(method_id, lane.episode_uid)] for lane in lanes]
            actual = _row_aggregate(rows)
            errors = {}
            for field_name in (
                "E2E_ms",
                "E2E_std_ms",
                "P95_E2E_ms",
                "LatencyReturn",
                "SR",
                "OSR",
                "SPL",
                "nDTW",
                "delay_violation_rate",
                "mean_steps",
            ):
                errors[field_name] = abs(
                    float(actual[field_name]) - float(expected[field_name])
                )
            aggregate_checks[method_id] = {
                "historical_policy": historical_name,
                "max_abs_error": max(errors.values()),
                "errors": errors,
            }
    validation_checks = {}
    for record in records:
        if record.method_family != "learned" or record.stage == "stage2n":
            continue
        rows = [replay_rows[(record.method_id, lane.episode_uid)]
                for lane in lanes]
        actual = _row_aggregate(rows)
        expected = record.validation
        if not expected:
            continue
        errors = {}
        for field_name in (
            "E2E_ms",
            "E2E_std_ms",
            "P95_E2E_ms",
            "LatencyReturn",
            "SR",
            "OSR",
            "SPL",
            "nDTW",
            "delay_violation_rate",
            "mean_steps",
        ):
            if field_name not in expected:
                continue
            errors[field_name] = abs(
                float(actual[field_name]) - float(expected[field_name])
            )
        validation_checks[record.method_id] = {
            "max_abs_error": max(errors.values()) if errors else None,
            "errors": errors,
        }
    return {
        "stage2n_per_episode": {
            "path": str(STAGE2N_EVAL_PER_EPISODE),
            "policies": per_episode_checks,
            "max_abs_float_error": dict(max_errors),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:50],
            "passed": len(mismatches) == 0 and bool(per_episode_checks),
        },
        "stage2n_aggregate": {
            "path": str(STAGE2N_EVAL_SUMMARY),
            "policies": aggregate_checks,
            "passed": all(
                item["max_abs_error"] <= 1e-9
                for item in aggregate_checks.values()
            ) if aggregate_checks else False,
        },
        "stage2o_stage2p_validation": {
            "policies": validation_checks,
            "passed": all(
                item["max_abs_error"] is not None
                and item["max_abs_error"] <= 1e-9
                for item in validation_checks.values()
            ) if validation_checks else False,
        },
    }


def _select_representatives(lane_records: Mapping[str, dict]) -> list[dict]:
    selected = []
    for band in ("short", "medium", "long"):
        for success in (True, False):
            candidates = [
                item for item in lane_records.values()
                if item["spec"]["band"] == band
                and bool(item["success"]) is bool(success)
            ]
            candidates.sort(key=lambda item: (
                int(item["steps"]),
                int(item["spec"]["episode_index"]),
            ))
            if not candidates:
                selected.append({
                    "band": band,
                    "outcome": "success" if success else "failure",
                    "episode_uid": "",
                    "missing": True,
                    "candidate_count": 0,
                    "selection_rule": SELECTION_RULE,
                })
                continue
            rank = (len(candidates) - 1) // 2
            pick = candidates[rank]
            selected.append({
                "band": band,
                "outcome": "success" if success else "failure",
                "episode_uid": pick["spec"]["episode_uid"],
                "missing": False,
                "candidate_count": len(candidates),
                "median_rank": rank,
                "steps": int(pick["steps"]),
                "success_value": bool(pick["success"]),
                "episode_index": int(pick["spec"]["episode_index"]),
                "task_seed": int(pick["spec"]["task_seed"]),
                "start_goal_distance_m": float(
                    pick["spec"]["start_goal_distance_m"]
                ),
                "goal_index": int(pick["spec"]["goal_index"]),
                "start_x_m": float(pick["spec"]["start"][0]),
                "start_y_m": float(pick["spec"]["start"][1]),
                "final_x_m": float(pick["final_pose"][0]),
                "final_y_m": float(pick["final_pose"][1]),
                "selection_rule": SELECTION_RULE,
            })
    if any(item["missing"] for item in selected):
        raise RuntimeError(
            "A requested Dev stratum is empty; refusing to substitute Test "
            f"samples: {[item for item in selected if item['missing']]}"
        )
    if len(selected) != 6:
        raise RuntimeError(f"expected 6 selected tasks, got {len(selected)}")
    return selected


def _choose_node(record: PolicyRecord, runner, rng=None) -> int:
    step = runner.step_data
    mask = np.asarray(step["mask"], dtype=bool)
    indices = np.where(mask)[0]
    if not len(indices):
        raise RuntimeError("no available MEC")
    if record.method_id == "control-greedy-min-e2e":
        metrics = step["latency_metrics"]
        return int(min(
            indices,
            key=lambda i: (float(metrics[i]["total_ms"]), int(i)),
        ))
    if record.method_id == "control-random":
        if rng is None:
            raise ValueError("random control requires rng")
        return int(rng.choice(indices))
    if record.algorithm == "dqn":
        return int(masked_argmax(
            record.agent.q_values(step["state"]), mask
        ))
    if record.algorithm == "ppo":
        return int(record.agent.greedy_action(step["state"], mask))
    raise ValueError(record.method_id)


def _run_closed_loop(record: PolicyRecord, spec: Mapping, bundle,
                     scene_cache, device, episode_position: int) -> dict:
    runner = LatencyOnlyEpisodeRunner(
        spec=prepare_runner_spec(spec),
        topology=bundle["topology"],
        encoder=bundle["encoder"],
        backends=bundle["backends"],
        projector=None,
        quality_prior=bundle["quality_prior"],
        gab_config=bundle["gab_config"],
        scene_cache=scene_cache,
        device=device,
    )
    rng = None
    if record.method_id == "control-random":
        rng = np.random.default_rng(
            RANDOM_RNG_SEED + int(episode_position)
        )
    step_rows = []
    while True:
        step = runner.step_data
        pose = _validated_pose(step["pose"])
        node = _choose_node(record, runner, rng=rng)
        transition = runner.apply(node)
        record_row = runner.step_records[-1]
        step_rows.append({
            "step_index": int(record_row["step"]),
            "pose": pose,
            "action": int(record_row["action"]),
            "action_name": str(record_row["action_name"]),
            "node_index": int(node),
            "node_id": str(record_row["node_id"]),
            "e2e_ms": float(record_row["e2e_ms"]),
            "distance_to_goal_m": float(record_row["distance_to_goal_m"]),
            "step_success": bool(record_row["success"]),
            "step_done": bool(record_row["done"]),
        })
        if transition["done"]:
            break
    result = runner.finalize(record.method_id)
    final_pose = np.asarray(
        [runner.env.x, runner.env.y, runner.env.yaw], dtype=np.float32
    )
    trajectory = np.asarray(result["trajectory"], dtype=np.float32)
    poses = np.stack([row["pose"] for row in step_rows]).astype(np.float32)
    actions = np.asarray(
        [row["action"] for row in step_rows], dtype=np.int64
    )
    node_choices = [int(row["node_index"]) for row in step_rows]
    step_e2e = [float(row["e2e_ms"]) for row in step_rows]
    return {
        "record": record,
        "spec": dict(spec),
        "step_rows": step_rows,
        "result": result,
        "poses": poses,
        "actions": actions,
        "trajectory": trajectory,
        "final_pose": final_pose,
        "node_choices": node_choices,
        "step_e2e_ms": step_e2e,
        "action_sha256": _action_sha256(actions),
        "decision_coord_sha256": _coord_sha256(poses),
        "path_coord_sha256": _coord_sha256(trajectory),
        "node_choice_sha256": _node_sha256(node_choices),
        "mean_e2e_ms": float(result["mean_e2e_ms"]),
        "reference_mean_e2e_ms": float(np.mean(step_e2e)),
    }


def _compare_data(
    scope: str,
    spec: Mapping,
    record: PolicyRecord,
    method: Mapping,
    reference: Mapping,
    reference_method: str,
) -> dict:
    max_xy, max_heading, common = _max_coordinate_deviation(
        method["poses"], reference["poses"]
    )
    method_actions = list(method["actions"])
    reference_actions = list(reference["actions"])
    actions_equal = method_actions == reference_actions
    method_nodes = list(method["node_choices"])
    reference_nodes = list(reference["node_choices"])
    node_common = min(len(method_nodes), len(reference_nodes))
    node_mismatch = sum(
        1 for index in range(node_common)
        if method_nodes[index] != reference_nodes[index]
    ) + abs(len(method_nodes) - len(reference_nodes))
    max_len = max(len(method_nodes), len(reference_nodes), 1)
    steps_equal = int(method["steps"]) == int(reference["steps"])
    success_equal = bool(method["success"]) == bool(reference["success"])
    e2e_delta = (
        float(method["mean_e2e_ms"]) - float(reference["mean_e2e_ms"])
    )
    return {
        "scope": scope,
        "episode_uid": str(spec["episode_uid"]),
        "band": str(spec["band"]),
        "band_id": int(spec["band_id"]),
        "task_seed": int(spec["task_seed"]),
        "goal_index": int(spec["goal_index"]),
        "method_id": record.method_id,
        "method_family": record.method_family,
        "stage": record.stage,
        "algorithm": record.algorithm,
        "seed": int(record.seed),
        "reward_scheme": record.reward_scheme,
        "reference_method": reference_method,
        "success": bool(method["success"]),
        "reference_success": bool(reference["success"]),
        "success_equal": success_equal,
        "steps": int(method["steps"]),
        "reference_steps": int(reference["steps"]),
        "steps_equal": steps_equal,
        "action_sha256": str(method["action_sha256"]),
        "reference_action_sha256": str(reference["action_sha256"]),
        "actions_equal": bool(actions_equal),
        "action_length_equal": len(method_actions) == len(reference_actions),
        "coord_sha256": str(method["decision_coord_sha256"]),
        "reference_coord_sha256": str(reference["decision_coord_sha256"]),
        "coord_hash_equal": (
            str(method["decision_coord_sha256"])
            == str(reference["decision_coord_sha256"])
        ),
        "coords_equal_1e-9_m": bool(max_xy <= COORD_TOLERANCE_M),
        "max_coord_deviation_m": float(max_xy),
        "max_heading_deviation_rad": float(max_heading),
        "common_steps_compared": int(common),
        "node_choice_sha256": str(method["node_choice_sha256"]),
        "reference_node_choice_sha256": str(
            reference["node_choice_sha256"]
        ),
        "node_choice_mismatch_steps": int(node_mismatch),
        "node_choice_mismatch_fraction": float(node_mismatch / max_len),
        "mean_e2e_ms": float(method["mean_e2e_ms"]),
        "reference_mean_e2e_ms": float(reference["mean_e2e_ms"]),
        "e2e_delta_ms": float(e2e_delta),
        "checkpoint_sha256": record.checkpoint_sha256,
        "reference_checkpoint_sha256": getattr(
            reference.get("record"), "checkpoint_sha256", ""
        ) if isinstance(reference, Mapping) else "",
        "method_source": str(method.get("source", "")),
        "reference_source": str(reference.get("source", "")),
    }


def _closed_loop_crosscheck(
    records: Sequence[PolicyRecord],
    selected: Sequence[Mapping],
    lane_records: Mapping[str, dict],
    replay_rows: Mapping[tuple[str, str], dict],
    bundle,
    scene_cache,
    device,
) -> tuple[list[dict], dict, list[dict], dict]:
    selected_uids = [item["episode_uid"] for item in selected]
    positions = {
        uid: index for index, uid in enumerate(
            [spec["episode_uid"] for spec in bundle["dev_specs"]]
        )
    }
    closed: dict[tuple[str, str], dict] = {}
    for record in records:
        for uid in selected_uids:
            spec = lane_records[uid]["spec"]
            data = _run_closed_loop(
                record, spec, bundle, scene_cache, device, positions[uid]
            )
            lane = lane_records[uid]
            # The navigation action sequence must match the frozen lane.
            if data["action_sha256"] != lane["action_sha256"]:
                raise RuntimeError(
                    f"{record.method_id}/{uid}: closed-loop action hash "
                    f"{data['action_sha256']} != lane {lane['action_sha256']}"
                )
            max_xy, max_heading, common = _max_coordinate_deviation(
                data["poses"], lane["poses"]
            )
            if max_xy > PATH_RECONSTRUCTION_TOLERANCE_M:
                raise RuntimeError(
                    f"{record.method_id}/{uid}: closed-loop pose deviation "
                    f"{max_xy} m from lane"
                )
            if int(data["result"]["steps"]) != int(lane["steps"]):
                raise RuntimeError(
                    f"{record.method_id}/{uid}: closed-loop steps "
                    f"{data['result']['steps']} != lane {lane['steps']}"
                )
            if bool(data["result"]["success"]) != bool(lane["success"]):
                raise RuntimeError(
                    f"{record.method_id}/{uid}: closed-loop success differs"
                )
            replay = replay_rows[(record.method_id, uid)]
            if record.method_id in CONTROL_METHODS:
                if list(data["node_choices"]) != list(
                    replay["node_choices"]
                ):
                    raise RuntimeError(
                        f"{record.method_id}/{uid}: control node choices "
                        "differ between closed loop and lane replay"
                    )
                if abs(float(data["mean_e2e_ms"]) - float(
                    replay["mean_e2e_ms"]
                )) > 1e-9:
                    raise RuntimeError(
                        f"{record.method_id}/{uid}: control E2E differs "
                        "between closed loop and lane replay"
                    )
            data["source"] = "closed_loop"
            data["success"] = bool(data["result"]["success"])
            data["steps"] = int(data["result"]["steps"])
            closed[(record.method_id, uid)] = data

    invariance = []
    reference_id = "2n-dqn-s0"
    for uid in selected_uids:
        spec = lane_records[uid]["spec"]
        reference_record = next(
            record for record in records if record.method_id == reference_id
        )
        reference = closed[(reference_id, uid)]
        for record in records:
            invariance.append(_compare_data(
                "selected6_closed_loop",
                spec,
                record,
                closed[(record.method_id, uid)],
                reference,
                reference_id,
            ))
        # Closed loop vs frozen lane: the strongest real-environment check.
        for record in records:
            method = closed[(record.method_id, uid)]
            lane = lane_records[uid]
            lane_view = {
                "poses": lane["poses"],
                "actions": lane["actions"],
                "node_choices": lane["greedy_node_choices"],
                "steps": lane["steps"],
                "success": lane["success"],
                "action_sha256": lane["action_sha256"],
                "decision_coord_sha256": lane["decision_coord_sha256"],
                "node_choice_sha256": lane["greedy_node_sha256"],
                "mean_e2e_ms": lane["greedy_mean_e2e_ms"],
                "source": "frozen_lane",
                "record": None,
            }
            invariance.append(_compare_data(
                "selected6_lane_vs_closed_loop",
                spec,
                record,
                method,
                lane_view,
                "frozen_lane_greedy",
            ))
    return invariance, closed, list(selected_uids), positions


def _lane_invariance(
    lanes: Sequence[Lane],
    records: Sequence[PolicyRecord],
    lane_records: Mapping[str, dict],
    replay_rows: Mapping[tuple[str, str], dict],
) -> list[dict]:
    rows = []
    reference_id = "2n-dqn-s0"
    reference_record = next(
        record for record in records if record.method_id == reference_id
    )
    for lane in lanes:
        uid = lane.episode_uid
        lane_data = lane_records[uid]
        reference_replay = replay_rows[(reference_id, uid)]
        reference = {
            "poses": lane_data["poses"],
            "actions": lane_data["actions"],
            "node_choices": reference_replay["node_choices"],
            "steps": lane_data["steps"],
            "success": lane_data["success"],
            "action_sha256": lane_data["action_sha256"],
            "decision_coord_sha256": lane_data["decision_coord_sha256"],
            "node_choice_sha256": reference_replay["node_choice_sha256"],
            "mean_e2e_ms": reference_replay["mean_e2e_ms"],
            "source": "lane_replay",
            "record": reference_record,
        }
        for record in records:
            replay = replay_rows[(record.method_id, uid)]
            method = {
                "poses": lane_data["poses"],
                "actions": lane_data["actions"],
                "node_choices": replay["node_choices"],
                "steps": lane_data["steps"],
                "success": lane_data["success"],
                "action_sha256": lane_data["action_sha256"],
                "decision_coord_sha256": lane_data["decision_coord_sha256"],
                "node_choice_sha256": replay["node_choice_sha256"],
                "mean_e2e_ms": replay["mean_e2e_ms"],
                "source": "lane_replay",
                "record": record,
            }
            rows.append(_compare_data(
                "dev60_lane",
                lane_data["spec"],
                record,
                method,
                reference,
                reference_id,
            ))
    return rows


def _summary_row(
    uid: str,
    spec: Mapping,
    lane_data: Mapping,
    record: PolicyRecord,
    data: Mapping,
) -> dict:
    result = data["result"]
    trajectory = np.asarray(data["trajectory"], dtype=np.float32)
    final_pose = data["final_pose"]
    return {
        "scope": "dev60_lane" if data.get("source") == "lane_replay" else
                 "selected6_closed_loop",
        "episode_uid": uid,
        "band": str(spec["band"]),
        "band_id": int(spec["band_id"]),
        "distance_range_m": f"{float(spec['distance_range'][0]):.0f}-"
                            f"{float(spec['distance_range'][1]):.0f}",
        "task_seed": int(spec["task_seed"]),
        "goal_index": int(spec["goal_index"]),
        "start_x_m": float(spec["start"][0]),
        "start_y_m": float(spec["start"][1]),
        "start_heading_rad": float(spec["start_yaw"]),
        "goal_x_m": float(spec["goal"][0]),
        "goal_y_m": float(spec["goal"][1]),
        "goal_landmark_size_m": float(lane_data["goal_landmark_size_m"]),
        "goal_landmark_shape": str(lane_data["goal_landmark_shape"]),
        "final_x_m": float(final_pose[0]),
        "final_y_m": float(final_pose[1]),
        "final_heading_rad": float(final_pose[2]),
        "success": bool(data["success"]),
        "steps": int(data["steps"]),
        "path_length_m": float(result.get(
            "path_length_m", _path_length(trajectory[:, :2])
        )),
        "start_goal_distance_m": float(spec["start_goal_distance_m"]),
        "spl": float(data.get("spl", result.get("spl", 0.0))),
        "ndtw": float(data.get("ndtw", result.get("ndtw", 0.0))),
        "mean_e2e_ms": float(data["mean_e2e_ms"]),
        "p95_e2e_ms": float(result.get("p95_e2e_ms", 0.0)),
        "delay_violation_rate": float(
            result.get("delay_violation_rate", 0.0)
        ),
        "latency_return": float(result.get(
            "latency_return", latency_return_from_mean(data["mean_e2e_ms"])
        )),
        "method_id": record.method_id,
        "method_family": record.method_family,
        "stage": record.stage,
        "algorithm": record.algorithm,
        "seed": int(record.seed),
        "reward_scheme": record.reward_scheme,
        "checkpoint_type": record.checkpoint_type,
        "checkpoint_path": record.checkpoint_path,
        "checkpoint_sha256": record.checkpoint_sha256,
        "action_sha256": str(data["action_sha256"]),
        "decision_coord_sha256": str(data["decision_coord_sha256"]),
        "path_coord_sha256": str(data["path_coord_sha256"]),
        "node_choice_sha256": str(data["node_choice_sha256"]),
        "node_choice_unique_count": int(len(set(data["node_choices"]))),
        "source": str(data.get("source", "")),
    }


def _closed_loop_summaries(
    records: Sequence[PolicyRecord],
    selected: Sequence[Mapping],
    lane_records: Mapping[str, dict],
    closed: Mapping[tuple[str, str], dict],
) -> list[dict]:
    rows = []
    for record in records:
        for item in selected:
            uid = item["episode_uid"]
            data = closed[(record.method_id, uid)]
            data["source"] = "selected6_closed_loop"
            data.setdefault("spl", data["result"]["spl"])
            data.setdefault("ndtw", data["result"]["ndtw"])
            rows.append(_summary_row(
                uid, lane_records[uid]["spec"], lane_records[uid],
                record, data,
            ))
    return rows


def _step_rows(
    records: Sequence[PolicyRecord],
    selected: Sequence[Mapping],
    lane_records: Mapping[str, dict],
    closed: Mapping[tuple[str, str], dict],
) -> list[dict]:
    rows = []
    for record in records:
        for item in selected:
            uid = item["episode_uid"]
            spec = lane_records[uid]["spec"]
            data = closed[(record.method_id, uid)]
            for step in data["step_rows"]:
                pose = step["pose"]
                rows.append({
                    "scope": "selected6_closed_loop",
                    "episode_uid": uid,
                    "band": str(spec["band"]),
                    "success": bool(data["success"]),
                    "steps_total": int(data["steps"]),
                    "method_id": record.method_id,
                    "method_family": record.method_family,
                    "stage": record.stage,
                    "algorithm": record.algorithm,
                    "seed": int(record.seed),
                    "reward_scheme": record.reward_scheme,
                    "checkpoint_sha256": record.checkpoint_sha256,
                    "step_index": int(step["step_index"]),
                    "x_m": float(pose[0]),
                    "y_m": float(pose[1]),
                    "z_m": float(ENV.uav_altitude_m),
                    "heading_rad": float(pose[2]),
                    "action": int(step["action"]),
                    "action_name": str(step["action_name"]),
                    "node_index": int(step["node_index"]),
                    "node_id": str(step["node_id"]),
                    "e2e_ms": float(step["e2e_ms"]),
                    "distance_to_goal_m": float(
                        step["distance_to_goal_m"]
                    ),
                    "step_success": bool(step["step_success"]),
                    "step_done": bool(step["step_done"]),
                    "source": "closed_loop",
                })
    return rows


def _dev60_summary(
    lanes: Sequence[Lane],
    lane_records: Mapping[str, dict],
) -> list[dict]:
    rows = []
    for lane in lanes:
        item = lane_records[lane.episode_uid]
        rows.append({
            "episode_uid": lane.episode_uid,
            "band": str(item["spec"]["band"]),
            "band_id": int(item["spec"]["band_id"]),
            "task_seed": int(item["spec"]["task_seed"]),
            "goal_index": int(item["spec"]["goal_index"]),
            "start_goal_distance_m": float(
                item["spec"]["start_goal_distance_m"]
            ),
            "success": bool(item["success"]),
            "steps": int(item["steps"]),
            "path_length_m": float(item["path_length_m"]),
            "spl": float(item["spl"]),
            "ndtw": float(item["ndtw"]),
            "greedy_mean_e2e_ms": float(item["greedy_mean_e2e_ms"]),
            "greedy_p95_e2e_ms": float(item["greedy_p95_e2e_ms"]),
            "action_sha256": str(item["action_sha256"]),
            "decision_coord_sha256": str(
                item["decision_coord_sha256"]
            ),
            "path_coord_sha256": str(item["path_coord_sha256"]),
        })
    return rows


def _city_geometry_rows(city) -> tuple[list[dict], dict]:
    rows = []
    for index, landmark in enumerate(city.landmarks):
        rows.append({
            "landmark_index": index,
            "x_m": float(landmark.x),
            "y_m": float(landmark.y),
            "physical_size_m": float(landmark.physical_size_m),
            "shape": str(landmark.shape),
            "is_distractor": bool(landmark.is_distractor),
        })
    meta = {
        "map_seed": int(city.map_seed),
        "city_fingerprint": str(city.fingerprint),
        "world_size_m": float(city.world_size_m),
        "num_landmarks": int(city.num_landmarks),
        "road_spacing_m": float(city.road_spacing_m),
        "road_half_width_m": float(city.road_half_width_m),
        "road_alpha": float(city.road_alpha),
        "background_value": float(city.background_value),
        "stop_radius_m": float(ENV.stop_radius_m),
        "uav_altitude_m": float(ENV.uav_altitude_m),
        "forward_step_m": float(ENV.forward_step_m),
        "turn_step_deg": float(ENV.turn_step_deg),
        "max_steps": int(ENV.max_steps),
    }
    return rows, meta


def _aggregate_audit(
    records: Sequence[PolicyRecord],
    selected: Sequence[Mapping],
    invariance: Sequence[Mapping],
    lane_records: Mapping[str, dict],
    checkpoint_rows: Sequence[Mapping],
    historical: Mapping,
    chosen_overlays: Mapping,
) -> dict:
    by_scope = defaultdict(list)
    for row in invariance:
        by_scope[row["scope"]].append(row)

    def scope_summary(scope):
        rows = by_scope.get(scope, [])
        max_dev = max(
            (float(row["max_coord_deviation_m"]) for row in rows),
            default=0.0,
        )
        action_mismatch = sum(
            1 for row in rows if not bool(row["actions_equal"])
        )
        step_mismatch = sum(
            1 for row in rows if not bool(row["steps_equal"])
        )
        success_mismatch = sum(
            1 for row in rows if not bool(row["success_equal"])
        )
        coord_hash_mismatch = sum(
            1 for row in rows if not bool(row["coord_hash_equal"])
        )
        return {
            "comparisons": len(rows),
            "max_coord_deviation_m": float(max_dev),
            "action_mismatches": int(action_mismatch),
            "coordinate_hash_mismatches": int(coord_hash_mismatch),
            "step_mismatches": int(step_mismatch),
            "success_mismatches": int(success_mismatch),
            "node_choice_differences": int(sum(
                int(row["node_choice_mismatch_steps"]) for row in rows
            )),
        }

    per_method = {}
    for record in records:
        rows = [
            row for row in invariance
            if row["method_id"] == record.method_id
            and row["scope"] == "selected6_closed_loop"
        ]
        lane_rows = [
            row for row in invariance
            if row["method_id"] == record.method_id
            and row["scope"] == "dev60_lane"
        ]
        per_method[record.method_id] = {
            "method_title": record.method_title,
            "stage": record.stage,
            "reward_scheme": record.reward_scheme,
            "closed_loop_max_coord_deviation_m": max(
                (float(row["max_coord_deviation_m"]) for row in rows),
                default=0.0,
            ),
            "closed_loop_action_mismatches": sum(
                1 for row in rows if not bool(row["actions_equal"])
            ),
            "closed_loop_node_mismatch_fraction_mean": float(np.mean([
                float(row["node_choice_mismatch_fraction"]) for row in rows
            ])) if rows else 0.0,
            "closed_loop_e2e_delta_ms_mean": float(np.mean([
                float(row["e2e_delta_ms"]) for row in rows
            ])) if rows else 0.0,
            "lane_node_mismatch_fraction_mean": float(np.mean([
                float(row["node_choice_mismatch_fraction"])
                for row in lane_rows
            ])) if lane_rows else 0.0,
            "lane_e2e_delta_ms_mean": float(np.mean([
                float(row["e2e_delta_ms"]) for row in lane_rows
            ])) if lane_rows else 0.0,
        }

    node_diversity = {}
    for uid in [item["episode_uid"] for item in selected]:
        hashes = set()
        e2e = []
        for record in records:
            row = next(
                item for item in invariance
                if item["scope"] == "selected6_closed_loop"
                and item["episode_uid"] == uid
                and item["method_id"] == record.method_id
            )
            hashes.add(row["node_choice_sha256"])
            e2e.append(float(row["mean_e2e_ms"]))
        node_diversity[uid] = {
            "distinct_node_choice_sequences": len(hashes),
            "methods": len(records),
            "e2e_min_ms": float(min(e2e)),
            "e2e_max_ms": float(max(e2e)),
            "e2e_spread_ms": float(max(e2e) - min(e2e)),
        }

    weight_groups = defaultdict(list)
    for row in checkpoint_rows:
        if row["model_state_sha256"]:
            weight_groups[row["model_state_sha256"]].append(
                row["method_id"]
            )
    duplicate_groups = [
        {"model_state_sha256": digest, "methods": methods}
        for digest, methods in weight_groups.items() if len(methods) > 1
    ]

    return {
        "stage": STAGE,
        "created_utc": utc_now(),
        "dev_tasks": len(lane_records),
        "selected_tasks": len(selected),
        "learned_methods": len(LEARNED_METHODS),
        "control_methods": len(CONTROL_METHODS),
        "selected_episodes": [
            {
                "episode_uid": item["episode_uid"],
                "band": item["band"],
                "outcome": item["outcome"],
                "steps": item.get("steps"),
                "candidate_count": item.get("candidate_count"),
            }
            for item in selected
        ],
        "scopes": {
            scope: scope_summary(scope) for scope in (
                "dev60_lane",
                "selected6_closed_loop",
                "selected6_lane_vs_closed_loop",
            )
        },
        "per_method": per_method,
        "node_diversity": node_diversity,
        "duplicate_weight_groups": duplicate_groups,
        "historical_crosscheck": historical,
        "overlay_tasks": dict(chosen_overlays),
        "invariance_passed": bool(all(
            scope_summary(scope)["action_mismatches"] == 0
            and scope_summary(scope)["step_mismatches"] == 0
            and scope_summary(scope)["success_mismatches"] == 0
            and scope_summary(scope)["max_coord_deviation_m"]
            <= COORD_TOLERANCE_M
            for scope in (
                "dev60_lane",
                "selected6_closed_loop",
                "selected6_lane_vs_closed_loop",
            )
        )),
    }


def _choose_overlay_tasks(
    records: Sequence[PolicyRecord],
    selected: Sequence[Mapping],
    closed: Mapping[tuple[str, str], dict],
) -> dict:
    groups = {
        "stage2n": [record for record in records if record.stage == "stage2n"],
        "stage2o": [record for record in records if record.stage == "stage2o"],
        "stage2p": [record for record in records if record.stage == "stage2p"],
    }
    chosen = {}
    for name, group in groups.items():
        best = None
        for item in selected:
            uid = item["episode_uid"]
            e2e = [float(closed[(record.method_id, uid)]["mean_e2e_ms"])
                   for record in group]
            spread = max(e2e) - min(e2e)
            candidate = (spread, uid, e2e)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is not None:
            chosen[name] = {
                "episode_uid": best[1],
                "e2e_spread_ms": float(best[0]),
                "methods": [record.method_id for record in group],
            }
    return chosen


def _write_outputs(
    out_dir: Path,
    records: Sequence[PolicyRecord],
    lanes: Sequence[Lane],
    lane_records: Mapping[str, dict],
    replay_rows: Mapping[tuple[str, str], dict],
    selected: Sequence[Mapping],
    invariance: Sequence[Mapping],
    closed: Mapping[tuple[str, str], dict],
    checkpoint_rows: Sequence[Mapping],
    historical: Mapping,
    audit: Mapping,
    city_rows: Sequence[Mapping],
    city_meta: Mapping,
    manifest: Mapping,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = _closed_loop_summaries(
        records, selected, lane_records, closed
    )
    step_rows = _step_rows(records, selected, lane_records, closed)
    dev_rows = _dev60_summary(lanes, lane_records)
    selected_rows = []
    for item in selected:
        selected_rows.append({
            "band": item["band"],
            "outcome": item["outcome"],
            "episode_uid": item["episode_uid"],
            "episode_index": item["episode_index"],
            "task_seed": item["task_seed"],
            "steps": item["steps"],
            "success": item["success_value"],
            "start_goal_distance_m": item["start_goal_distance_m"],
            "goal_index": item["goal_index"],
            "start_x_m": item["start_x_m"],
            "start_y_m": item["start_y_m"],
            "final_x_m": item["final_x_m"],
            "final_y_m": item["final_y_m"],
            "candidate_count": item["candidate_count"],
            "median_rank": item["median_rank"],
            "selection_rule": item["selection_rule"],
        })
    write_csv(out_dir / "trajectory_steps.csv", step_rows)
    write_csv(out_dir / "trajectory_summary.csv", summary_rows)
    write_csv(out_dir / "trajectory_invariance.csv", list(invariance))
    write_csv(out_dir / "dev60_task_summary.csv", dev_rows)
    write_csv(out_dir / "selected_tasks.csv", selected_rows)
    write_csv(out_dir / "checkpoint_inventory.csv", list(checkpoint_rows))
    write_csv(out_dir / "city_geometry.csv", list(city_rows))
    atomic_write_json(out_dir / "city_meta.json", dict(city_meta))
    atomic_write_json(out_dir / "audit_summary.json", dict(audit))
    output_hashes = {}
    for name in (
        "trajectory_steps.csv",
        "trajectory_summary.csv",
        "trajectory_invariance.csv",
        "dev60_task_summary.csv",
        "selected_tasks.csv",
        "checkpoint_inventory.csv",
        "city_geometry.csv",
        "city_meta.json",
        "audit_summary.json",
    ):
        output_hashes[name] = sha256_file(out_dir / name)
    manifest = dict(manifest)
    manifest["outputs"] = output_hashes
    manifest["created_utc"] = utc_now()
    atomic_write_json(out_dir / "run_manifest.json", manifest)
    return {
        "summary_rows": len(summary_rows),
        "step_rows": len(step_rows),
        "dev_rows": len(dev_rows),
        "invariance_rows": len(invariance),
        "checkpoint_rows": len(checkpoint_rows),
        "output_hashes": output_hashes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--max-tasks", type=int, default=0,
        help="limit Dev lanes (smoke only; final run must use all 60)",
    )
    parser.add_argument(
        "--skip-closed-loop", action="store_true",
        help="skip the six-task full closed-loop audit (smoke only)",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    started = time.time()
    print(f"[stage2q] out_dir={out_dir}", flush=True)

    protocol = verify_frozen_protocol(verify_test_file=False)
    dev_info = protocol["tasks"]["dev"]
    dev_file = _repo_path(dev_info["path"])
    if sha256_file(dev_file) != EXPECTED_DEV_TASK_FILE_SHA256:
        raise ValueError("Dev task file SHA-256 mismatch")
    dev_pool = load_task_pool("dev")
    if dev_pool["tasks_sha256"] != EXPECTED_DEV_TASKS_SHA256:
        raise ValueError("Dev tasks_sha256 mismatch")
    if len(dev_pool["tasks"]) != DEV_EPISODES:
        raise ValueError(f"Dev task count {len(dev_pool['tasks'])} != 60")
    deployment = protocol["deployment"]
    deployment_file = _repo_path(deployment["path"])
    deployment_file_sha = sha256_file(deployment_file)
    if deployment_file_sha != EXPECTED_DEPLOYMENT_FILE_SHA256:
        raise ValueError("Deployment file SHA-256 mismatch")
    deployment_canonical_sha = str(deployment["deployment_sha256"])
    if deployment_canonical_sha != EXPECTED_DEPLOYMENT_CANONICAL_SHA256:
        raise ValueError("Canonical deployment SHA-256 mismatch")
    mec_path = _repo_path(protocol["mec"]["checkpoint"])
    mec_sha = sha256_file(mec_path)
    if mec_sha != MEC_CHECKPOINT_SHA256:
        raise ValueError("MEC checkpoint SHA-256 mismatch")

    bundle = load_bundle()
    bundle["encoder"] = None
    bundle["backends"] = None

    records = _discover_policies()
    device, encoder, backends = load_stage2n_runtime(args.device)
    bundle["encoder"] = encoder
    bundle["backends"] = backends
    scene_cache = make_stage2n_scene_cache(
        encoder, device, capacity=max(64, len(bundle["dev_specs"]))
    )

    print("[stage2q] building frozen Dev lanes", flush=True)
    specs, lanes, lane_records = _build_lane_records(
        bundle, device, encoder, backends, scene_cache,
        limit=args.max_tasks or None,
    )
    if args.max_tasks and len(lane_records) != args.max_tasks:
        raise RuntimeError("lane limit mismatch")
    if not args.max_tasks and len(lane_records) != DEV_EPISODES:
        raise RuntimeError("full run must contain 60 Dev lanes")

    print("[stage2q] replaying association policies", flush=True)
    replay_rows = _replay_all_policies(lanes, records, device)
    checkpoint_rows = _checkpoint_inventory(records)

    historical = _historical_crosscheck(
        lanes, lane_records, replay_rows, records
    )
    print(
        "[stage2q] historical crosscheck:",
        json.dumps({
            key: value.get("passed")
            for key, value in historical.items()
        }),
        flush=True,
    )

    selected = _select_representatives(lane_records)
    print(
        "[stage2q] selected:",
        [(item["band"], item["outcome"], item["episode_uid"],
          item["steps"]) for item in selected],
        flush=True,
    )
    invariance = _lane_invariance(
        lanes, records, lane_records, replay_rows
    )
    closed = {}
    chosen_overlays = {}
    if not args.skip_closed_loop:
        print("[stage2q] full closed-loop audit", flush=True)
        closed_inv, closed, _, _ = _closed_loop_crosscheck(
            records, selected, lane_records, replay_rows, bundle,
            scene_cache, device,
        )
        invariance = list(invariance) + closed_inv
        chosen_overlays = _choose_overlay_tasks(
            records, selected, closed
        )
    else:
        chosen_overlays = {
            name: {
                "episode_uid": selected[0]["episode_uid"],
                "e2e_spread_ms": 0.0,
                "methods": [],
            }
            for name in ("stage2n", "stage2o", "stage2p")
        }

    city = build_city(MAP_SEED)
    if city.fingerprint != EXPECTED_CITY_FINGERPRINT:
        raise ValueError("City fingerprint mismatch")
    city_rows, city_meta = _city_geometry_rows(city)
    audit = _aggregate_audit(
        records, selected, invariance, lane_records,
        checkpoint_rows, historical, chosen_overlays,
    )
    audit["run_seconds"] = float(time.time() - started)
    audit["max_tasks_flag"] = int(args.max_tasks)
    audit["closed_loop_skipped"] = bool(args.skip_closed_loop)
    audit["selection_rule"] = SELECTION_RULE
    audit["protocol_file_sha256"] = sha256_file(
        _repo_path(protocol["path"])
    ) if "path" in protocol else sha256_file(
        Path("runs/v83_spatial_stage2/stage2n_latency_only/protocol_freeze.json")
    )
    audit["dev_task_file_sha256"] = sha256_file(dev_file)
    audit["dev_tasks_sha256"] = dev_pool["tasks_sha256"]
    audit["deployment_file_sha256"] = deployment_file_sha
    audit["deployment_canonical_sha256"] = deployment_canonical_sha
    audit["mec_checkpoint_sha256"] = mec_sha
    audit["final_test_read"] = False
    audit["test_pool_read"] = False

    manifest = {
        "stage": STAGE,
        "branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "python": sys.executable,
        "torch": _torch_version(),
        "device": str(device),
        "gpu": _gpu_name(),
        "argv": sys.argv,
        "dev_task_file": _rel(dev_file),
        "dev_task_file_sha256": sha256_file(dev_file),
        "dev_tasks_sha256": dev_pool["tasks_sha256"],
        "dev_tasks": len(dev_pool["tasks"]),
        "band_counts": {
            band: sum(
                1 for task in dev_pool["tasks"] if task["band"] == band
            ) for band in ("short", "medium", "long")
        },
        "map_seed": MAP_SEED,
        "city_fingerprint": city.fingerprint,
        "deployment_file": _rel(deployment_file),
        "deployment_file_sha256": deployment_file_sha,
        "deployment_canonical_sha256": deployment_canonical_sha,
        "deployment_note": (
            "Stage-2o/2p protocol freezes record the deployment.json file "
            "SHA-256 under the key deployment_sha256; the Stage-2n canonical "
            "content hash is 4c0950...  Both refer to the same file."
        ),
        "mec_checkpoint": _rel(mec_path),
        "mec_checkpoint_sha256": mec_sha,
        "mec_checkpoint_expected_sha256": MEC_CHECKPOINT_SHA256,
        "task_pool_version": dev_pool.get("version", ""),
        "sealed_test_read": False,
        "test_pool_read": False,
        "policies": [record.public() for record in records],
        "checkpoint_inventory": list(checkpoint_rows),
        "historical_crosscheck": dict(historical),
        "selection_rule": SELECTION_RULE,
        "selected_tasks": [
            {key: value for key, value in item.items()
             if key != "selection_rule"}
            for item in selected
        ],
        "elapsed_seconds": float(time.time() - started),
        "final_test_read": False,
    }
    outputs = _write_outputs(
        out_dir, records, lanes, lane_records, replay_rows, selected,
        invariance, closed, checkpoint_rows, historical, audit,
        city_rows, city_meta, manifest,
    )
    print("[stage2q] outputs:", json.dumps(outputs, indent=2), flush=True)
    print(f"[stage2q] done in {time.time() - started:.1f}s", flush=True)
    return audit


def _git_value(*args) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _torch_version() -> str:
    try:
        import torch
        return str(torch.__version__)
    except Exception:
        return ""


def _gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return ""


if __name__ == "__main__":
    main()
