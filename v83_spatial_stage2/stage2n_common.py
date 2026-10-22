"""Stage-2n: fixed-city, homogeneous-MEC, latency-only association.

This module freezes the parts of the experiment that must not drift between
the DQN and PPO trainers:

* one fixed city and three disjoint task-seed blocks (Train 4000 / Dev 60 /
  sealed Test 400);
* one fixed homogeneous deployment of 20 MEC nodes.  Every node uses the same
  frozen Stage-2j Transformer-M checkpoint, compute capacity, bandwidth and
  coverage radius; only the node positions differ;
* the latency-only reward

      r_t = -0.25 / 100 * (mean_E2E_t - mean_E2E_{t-1})

  whose undiscounted episode sum is exactly
  ``LatencyReturn = -0.25 * mean_E2E_ms / 100``;
* the Dev mean-E2E convergence rule and the Dev checkpoint-selection rule.

Nothing in this module trains a MEC.  The Stage-2j Transformer-M checkpoint
(SHA-256 ``57c88b...c6d``) is loaded read-only and frozen.

Task generation reuses the Stage-2l fixed-city builder and its oracle
acceptance rule (the task must be solved by the oracle with a step count
inside the band-specific expert range).  Stage-2n has its own task IDs, seed
blocks and output directory, so Stage-2l/2m artefacts are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENV, NETWORK
from v83_spatial_stage2.stage2k_common import (
    BAND_ACTION_RANGES,
    BAND_ID,
    BAND_RANGES,
    BANDS,
    atomic_write_json,
    read_json,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_v2_common import load_capability_prior
from v83_spatial_stage2.stage2l_fixed_city import (
    FixedCityEnv,
    _inspect_task,
    build_city,
    canonical_sha256,
    sha256_file,
)


# ---------------------------------------------------------------------------
# Frozen paths and constants
# ---------------------------------------------------------------------------

STAGE = "stage2n_latency_only"
STAGE2N_ROOT = Path("runs/v83_spatial_stage2/stage2n_latency_only")
SPECS_DIR = STAGE2N_ROOT / "specs"
DEPLOYMENT_DIR = STAGE2N_ROOT / "deployment"
RUNS_DIR = STAGE2N_ROOT / "runs"
EVAL_DIR = STAGE2N_ROOT / "evaluation"
LOG_DIR = STAGE2N_ROOT / "logs"
PROTOCOL_PATH = STAGE2N_ROOT / "protocol_freeze.json"

# Results that must be tracked in Git.  ``runs/`` is ignored; this directory
# is deliberately not named ``*_results`` (that pattern is ignored).
EVIDENCE_DIR = Path("v83_spatial_stage2/stage2n_evidence")

# One fixed city.  Reusing the Stage-2l map seed keeps the frozen Transformer-M
# in the visual distribution on which its Stage-2j closed-loop record was
# measured.  Stage-2n tasks use completely different task-seed blocks.
MAP_SEED = 10_000_001

# New, disjoint blocks.  Stage-2j uses 1M/2M/3M, Stage-2k 5M/6M/7M and
# Stage-2l 11M/12M/13M; none of those ranges can collide with 21M/22M/23M.
TASK_SEED_BASE = {
    "train": 21_000_000,
    "dev": 22_000_000,
    "test": 23_000_000,
}
TASK_ID_PREFIX = "stage2n"
TASK_POOL_VERSION = "stage2n-fixed-city-latency-v1"

TRAIN_EPISODES = 4000
DEV_EPISODES = 60
TEST_EPISODES = 400
BAND_COUNTS_TRAIN = {"short": 1334, "medium": 1333, "long": 1333}
BAND_COUNTS_DEV = {"short": 20, "medium": 20, "long": 20}
BAND_COUNTS_TEST = {"short": 134, "medium": 133, "long": 133}
TRAIN_ORDER_SEED = 20261022

VALIDATION_EVERY = 200
MAX_EPISODES = TRAIN_EPISODES
AGENT_SEEDS = (0, 1, 2)

# ---------------------------------------------------------------------------
# Frozen MEC and homogeneous deployment
# ---------------------------------------------------------------------------

MEC_MODEL = "Transformer-M"
MEC_MODEL_INDEX = 3
MEC_CHECKPOINT = Path("runs/v83_spatial_stage2/stage2j/Transformer-M/best.pt")
MEC_CHECKPOINT_SHA256 = (
    "57c88b7d71e1a7f17862aaa8429afa8a041cba5e747f744f3a7febf47f273c6d"
)
MEC_DEV_METRICS = Path(
    "runs/v83_spatial_stage2/stage2j/Transformer-M/eval_dev_val_180/metrics.json"
)
MEC_TEST_METRICS = Path(
    "runs/v83_spatial_stage2/stage2j/Transformer-M/eval_final_test_300/metrics.json"
)
CAPABILITY_PRIOR_PATH = Path(
    "runs/v83_spatial_stage2/stage2k/capability_prior_v2.json"
)

# The homogeneous deployment reuses the 20 positions of Stage-2l A_fixed.
DEPLOYMENT_SOURCE = Path(
    "runs/v83_spatial_stage2/stage2l_fixed_city/topologies/A_fixed.json"
)

# Uniform high-tier node capabilities.  These are deliberately identical for
# every node; the only source of E2E differences is the node geometry.
MEC_COMPUTE_FLOPS_PER_S = 2.0e12
MEC_BANDWIDTH_HZ = 30.0e6
MEC_COVERAGE_RADIUS_M = float(NETWORK.coverage_radius_m)
MEC_NUM_NODES = int(NETWORK.num_gbs)

# Reward and evaluation constants.
LATENCY_REWARD_WEIGHT = 0.25
LATENCY_REWARD_DIVISOR = 100.0
DEADLINE_MS = float(NETWORK.decision_latency_budget_s) * 1000.0

STATE_DIM = 384
ACTION_DIM = 20
STATE_MODE = "pure_mec_v2"
REWARD_MODE = "latency_only_v1"
REWARD_DESCRIPTION = "r_t = -0.25 / 100 * (mean_E2E_t - mean_E2E_{t-1})"
LATENCY_RETURN_DESCRIPTION = "LatencyReturn = -0.25 * mean_E2E_ms / 100"

CONVERGENCE_CRITERION = (
    "最近三次 Dev 验证的平均 E2E 时延波动 <= 最佳值的 1%，且最终值不比"
    "最佳值差超过 1%；训练预算 4000 Episodes 结束时仍不满足则标记未收敛"
)
CONVERGENCE_SPREAD_FRACTION = 0.01
CONVERGENCE_FINAL_GAP_FRACTION = 0.01
CHECKPOINT_SELECTION = (
    "最低 Dev 平均 E2E 时延；E2E 完全相同时依次比较更高 SR、更早 Episode"
)


# ---------------------------------------------------------------------------
# Hashing / JSON helpers
# ---------------------------------------------------------------------------


def task_payload_sha256(task: Mapping) -> str:
    return canonical_sha256(dict(task))


def _without_task_hash(task: Mapping) -> dict:
    payload = dict(task)
    payload.pop("task_sha256", None)
    return payload


def attach_task_hash(task: Mapping) -> dict:
    payload = dict(task)
    payload["task_sha256"] = task_payload_sha256(_without_task_hash(payload))
    return payload


def verify_task_hash(task: Mapping) -> bool:
    return str(task.get("task_sha256", "")) == task_payload_sha256(
        _without_task_hash(task)
    )


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


def _spec_from_info(split: str, index: int, band: str, info: Mapping) -> dict:
    task = info["task"]
    spec = {
        "episode_uid": f"{TASK_ID_PREFIX}-{split}-{index:05d}",
        "split": split,
        "episode_index": int(index),
        "scenario": "fixed_city",
        "map_seed": int(task["map_seed"]),
        "task_seed": int(task["task_seed"]),
        "goal_index": int(task["goal_index"]),
        "band": str(band),
        "band_id": int(BAND_ID[band]),
        "distance_range": [float(v) for v in task["distance_range"]],
        "oracle_steps": int(info["steps"]),
        "oracle_action_counts": {
            str(key): int(value)
            for key, value in info["action_counts"].items()
        },
        "start_goal_distance_m": float(task["start_goal_distance_m"]),
        "start": [float(v) for v in task["start"]],
        "goal": [float(v) for v in task["goal"]],
        "start_yaw": float(task["start_yaw"]),
        "start_pose": [
            float(task["start"][0]),
            float(task["start"][1]),
            float(task["start_yaw"]),
        ],
        "trajectory_length_m": float(task["start_goal_distance_m"]),
        "city_fingerprint": str(task["city_fingerprint"]),
    }
    return attach_task_hash(spec)


def generate_split_tasks(
    split: str,
    counts: Mapping[str, int],
    verbose: bool = False,
    max_candidates: int = 500_000,
) -> list[dict]:
    """Generate one accepted, band-balanced Stage-2n task list."""
    if split not in TASK_SEED_BASE:
        raise ValueError(f"unknown split {split}")
    if sum(int(counts[band]) for band in BANDS) <= 0:
        raise ValueError("counts must be positive")
    env = FixedCityEnv(MAP_SEED)
    specs: list[dict] = []
    candidate_seed = int(TASK_SEED_BASE[split])
    attempts = 0
    for band in BANDS:
        target = int(counts[band])
        found = 0
        while found < target:
            if attempts >= int(max_candidates):
                raise RuntimeError(
                    f"Unable to generate {target} {band} tasks for {split}"
                )
            info = _inspect_task(env, candidate_seed, band)
            candidate_seed += 1
            attempts += 1
            if not info["accepted"]:
                continue
            specs.append(_spec_from_info(split, len(specs), band, info))
            found += 1
        if verbose:
            print(
                f"[stage2n] {split}/{band}: {found} accepted "
                f"(attempts so far {attempts})",
                flush=True,
            )
    expected = sum(int(counts[band]) for band in BANDS)
    if len(specs) != expected:
        raise RuntimeError(f"{split}: {len(specs)} tasks != {expected}")
    return specs


def task_pool_payload(
    split: str,
    specs: Sequence[dict],
    sealed: bool = False,
) -> dict:
    specs = [dict(item) for item in specs]
    if not specs:
        raise ValueError("empty task pool")
    missing = [item["episode_uid"] for item in specs if not verify_task_hash(item)]
    if missing:
        raise ValueError(f"invalid task SHA-256 for {missing[:3]}")
    seeds = [int(item["task_seed"]) for item in specs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate task seeds")
    uids = [str(item["episode_uid"]) for item in specs]
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate task IDs")
    if split == "test" and not sealed:
        raise ValueError("the Stage-2n Test pool must be sealed")
    return {
        "version": TASK_POOL_VERSION,
        "stage": STAGE,
        "split": split,
        "created_utc": utc_now(),
        "sealed": bool(sealed),
        "final_test_read": False,
        "map_seed": int(MAP_SEED),
        "city_fingerprint": build_city(MAP_SEED).fingerprint,
        "seed_base": int(TASK_SEED_BASE[split]),
        "seed_range": [min(seeds), max(seeds)],
        "band_counts": {
            band: sum(1 for item in specs if item["band"] == band)
            for band in BANDS
        },
        "bands": {
            band: {
                "distance_range_m": list(BAND_RANGES[band]),
                "oracle_steps_range": list(BAND_ACTION_RANGES[band]),
            }
            for band in BANDS
        },
        "stop_radius_m": float(ENV.stop_radius_m),
        "max_steps": int(ENV.max_steps),
        "num_tasks": len(specs),
        "tasks_sha256": canonical_sha256(specs),
        "tasks": specs,
    }


def task_pool_path(split: str) -> Path:
    return SPECS_DIR / f"{split}_tasks.json"


def save_task_pool(split: str, specs: Sequence[dict], sealed: bool = False) -> Path:
    payload = task_pool_payload(split, specs, sealed=sealed)
    path = task_pool_path(split)
    atomic_write_json(path, payload)
    return path


def load_task_pool(split: str, allow_test: bool = False) -> dict:
    path = task_pool_path(split)
    payload = read_json(path)
    if payload is None:
        raise FileNotFoundError(path)
    if payload.get("split") != split:
        raise ValueError(f"{path} is not a {split} task pool")
    if split == "test":
        if payload.get("sealed") is not True:
            raise ValueError("Stage-2n Test pool is not sealed")
        if not allow_test:
            raise PermissionError(
                "Stage-2n Test is sealed; pass allow_test=True only in the "
                "final evaluator"
            )
    if payload.get("tasks_sha256") != canonical_sha256(payload["tasks"]):
        raise ValueError(f"{path} task list hash mismatch")
    for task in payload["tasks"]:
        if not verify_task_hash(task):
            raise ValueError(
                f"{path}: task {task['episode_uid']} SHA-256 mismatch"
            )
    return payload


def build_training_order(
    train_specs: Sequence[dict],
    seed: int = TRAIN_ORDER_SEED,
) -> dict:
    order = [str(item["episode_uid"]) for item in train_specs]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(order)
    payload = {
        "version": "stage2n-training-order-v1",
        "stage": STAGE,
        "created_utc": utc_now(),
        "seed": int(seed),
        "map_seed": int(MAP_SEED),
        "task_pool_sha256": canonical_sha256(list(train_specs)),
        "num_tasks": len(order),
        "band_counts": {
            band: sum(
                1 for item in train_specs if item["band"] == band
            )
            for band in BANDS
        },
        "order": order,
    }
    atomic_write_json(SPECS_DIR / "training_order.json", payload)
    return payload


def load_training_order() -> dict:
    payload = read_json(SPECS_DIR / "training_order.json")
    if payload is None:
        raise FileNotFoundError(SPECS_DIR / "training_order.json")
    return payload


def ordered_train_specs() -> list[dict]:
    pool = load_task_pool("train")
    order = load_training_order()
    by_uid = {item["episode_uid"]: item for item in pool["tasks"]}
    if set(by_uid) != set(order["order"]):
        raise ValueError("training order != train task pool")
    return [by_uid[uid] for uid in order["order"]]


def dev_specs() -> list[dict]:
    return list(load_task_pool("dev")["tasks"])


def prepare_runner_spec(spec: Mapping) -> dict:
    """Add the legacy ``seed`` alias used by the shared runner/finalizer."""
    prepared = dict(spec)
    prepared["seed"] = int(prepared["task_seed"])
    return prepared


# ---------------------------------------------------------------------------
# Isolation audits
# ---------------------------------------------------------------------------


def isolation_audit(
    train_specs: Sequence[dict],
    dev_specs_: Sequence[dict],
    test_specs: Sequence[dict],
) -> dict:
    def seed_set(items):
        return {int(item["task_seed"]) for item in items}

    def uid_set(items):
        return {str(item["episode_uid"]) for item in items}

    tr, dv, te = seed_set(train_specs), seed_set(dev_specs_), seed_set(test_specs)
    tru, dvu, teu = uid_set(train_specs), uid_set(dev_specs_), uid_set(test_specs)
    checks = {
        "train_dev_seed_disjoint": not (tr & dv),
        "train_test_seed_disjoint": not (tr & te),
        "dev_test_seed_disjoint": not (dv & te),
        "train_dev_uid_disjoint": not (tru & dvu),
        "train_test_uid_disjoint": not (tru & teu),
        "dev_test_uid_disjoint": not (dvu & teu),
        "all_share_one_map": (
            {int(item["map_seed"]) for item in train_specs}
            | {int(item["map_seed"]) for item in dev_specs_}
            | {int(item["map_seed"]) for item in test_specs}
        ) == {int(MAP_SEED)},
        "all_city_fingerprints_match": (
            {str(item["city_fingerprint"]) for item in train_specs}
            | {str(item["city_fingerprint"]) for item in dev_specs_}
            | {str(item["city_fingerprint"]) for item in test_specs}
        ) == {build_city(MAP_SEED).fingerprint},
        "test_sealed": True,
    }
    checks["ok"] = all(
        value for key, value in checks.items() if key != "test_sealed"
    )
    return {
        "version": "stage2n-isolation-v1",
        "created_utc": utc_now(),
        "checks": checks,
        "counts": {
            "train": len(train_specs),
            "dev": len(dev_specs_),
            "test": len(test_specs),
        },
        "task_seed_ranges": {
            "train": [min(tr), max(tr)],
            "dev": [min(dv), max(dv)],
            "test": [min(te), max(te)],
        },
    }


def distribution_report(
    train_specs: Sequence[dict],
    dev_specs_: Sequence[dict],
    test_specs: Sequence[dict],
) -> dict:
    def one(items):
        return {
            "count": len(items),
            "bands": dict(Counter(item["band"] for item in items)),
            "distance_m": {
                band: {
                    "min": float(min(
                        item["start_goal_distance_m"]
                        for item in items if item["band"] == band
                    )),
                    "max": float(max(
                        item["start_goal_distance_m"]
                        for item in items if item["band"] == band
                    )),
                    "mean": float(np.mean([
                        item["start_goal_distance_m"]
                        for item in items if item["band"] == band
                    ])),
                }
                for band in BANDS
            },
            "oracle_steps": {
                band: {
                    "min": int(min(
                        item["oracle_steps"]
                        for item in items if item["band"] == band
                    )),
                    "max": int(max(
                        item["oracle_steps"]
                        for item in items if item["band"] == band
                    )),
                    "mean": float(np.mean([
                        item["oracle_steps"]
                        for item in items if item["band"] == band
                    ])),
                }
                for band in BANDS
            },
        }

    return {
        "version": "stage2n-distribution-v1",
        "created_utc": utc_now(),
        "map_seed": int(MAP_SEED),
        "city_fingerprint": build_city(MAP_SEED).fingerprint,
        "splits": {
            "train": one(train_specs),
            "dev": one(dev_specs_),
            "test": one(test_specs),
        },
    }


# ---------------------------------------------------------------------------
# Homogeneous deployment
# ---------------------------------------------------------------------------


def _load_capability_flat():
    payload, flat = load_capability_prior(CAPABILITY_PRIOR_PATH)
    if payload.get("final_test_read") is not False:
        raise ValueError("capability prior claims a Final Test read")
    return payload, flat


def build_homogeneous_deployment(verbose: bool = False) -> dict:
    """Build the fixed 20-node homogeneous deployment and its audit."""
    source = read_json(DEPLOYMENT_SOURCE)
    if source is None:
        raise FileNotFoundError(DEPLOYMENT_SOURCE)
    source_nodes = list(source.get("nodes", []))
    if len(source_nodes) != MEC_NUM_NODES:
        raise ValueError(
            f"{DEPLOYMENT_SOURCE} has {len(source_nodes)} nodes, "
            f"expected {MEC_NUM_NODES}"
        )
    positions = [(float(node["x"]), float(node["y"])) for node in source_nodes]
    if len(set(positions)) != len(positions):
        raise ValueError("A_fixed contains duplicate node positions")

    prior_payload, prior_flat = _load_capability_flat()
    transformer_m_prior = prior_flat[MEC_MODEL]
    nodes = []
    for index, (x, y) in enumerate(positions):
        nodes.append({
            "node_id": f"M{index + 1:02d}",
            "x": float(x),
            "y": float(y),
            "model_type": MEC_MODEL,
            "model_index": int(MEC_MODEL_INDEX),
            "quality_prior": float(transformer_m_prior["ndtw"]),
            "compute_capacity_flops_per_s": float(MEC_COMPUTE_FLOPS_PER_S),
            "bandwidth_hz": float(MEC_BANDWIDTH_HZ),
            "coverage_radius_m": float(MEC_COVERAGE_RADIUS_M),
        })

    checks = {
        "num_nodes_20": len(nodes) == MEC_NUM_NODES,
        "positions_unique": len({(n["x"], n["y"]) for n in nodes}) == len(nodes),
        "one_model_type": len({n["model_type"] for n in nodes}) == 1,
        "one_compute_capacity": len({
            n["compute_capacity_flops_per_s"] for n in nodes
        }) == 1,
        "one_bandwidth": len({n["bandwidth_hz"] for n in nodes}) == 1,
        "one_coverage_radius": len({
            n["coverage_radius_m"] for n in nodes
        }) == 1,
        "one_quality_prior": len({n["quality_prior"] for n in nodes}) == 1,
        "all_transformer_m": all(
            n["model_type"] == MEC_MODEL for n in nodes
        ),
        "positions_inside_city": all(
            0.0 <= n["x"] <= ENV.world_size_m
            and 0.0 <= n["y"] <= ENV.world_size_m
            for n in nodes
        ),
    }
    checks["ok"] = all(checks.values())
    if not checks["ok"]:
        raise RuntimeError(f"homogeneity audit failed: {checks}")

    deployment_body = {
        "version": "stage2n-homogeneous-deployment-v1",
        "stage": STAGE,
        "created_utc": utc_now(),
        "deployment_id": "stage2n-H_fixed",
        "description": (
            "20 MEC nodes at the Stage-2l A_fixed positions; every node uses "
            "the same frozen Transformer-M weights, compute capacity, "
            "bandwidth, coverage radius and communication parameters"
        ),
        "source_topology": str(DEPLOYMENT_SOURCE),
        "source_topology_sha256": sha256_file(DEPLOYMENT_SOURCE),
        "source_topology_seed": int(source.get("topology_seed", -1)),
        "capability_prior": {
            "path": str(CAPABILITY_PRIOR_PATH),
            "sha256": sha256_file(CAPABILITY_PRIOR_PATH),
            "model": MEC_MODEL,
            "sr": float(transformer_m_prior["sr"]),
            "osr": float(transformer_m_prior["osr"]),
            "spl": float(transformer_m_prior["spl"]),
            "ndtw": float(transformer_m_prior["ndtw"]),
        },
        "mec": {
            "model": MEC_MODEL,
            "checkpoint": str(MEC_CHECKPOINT),
            "checkpoint_sha256": MEC_CHECKPOINT_SHA256,
            "compute_capacity_flops_per_s": float(MEC_COMPUTE_FLOPS_PER_S),
            "bandwidth_hz": float(MEC_BANDWIDTH_HZ),
            "coverage_radius_m": float(MEC_COVERAGE_RADIUS_M),
            "tx_power_w": float(NETWORK.default_tx_power_w),
            "num_nodes": MEC_NUM_NODES,
            "frozen": True,
            "requires_grad": False,
        },
        "homogeneity_checks": checks,
        "nodes": nodes,
        "final_test_read": False,
    }
    deployment_body["deployment_sha256"] = canonical_sha256({
        "nodes": nodes,
        "mec": deployment_body["mec"],
        "source_topology_sha256": deployment_body["source_topology_sha256"],
    })
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    path = DEPLOYMENT_DIR / "deployment.json"
    existing = read_json(path)
    if (
        existing is not None
        and existing.get("deployment_sha256")
        == deployment_body["deployment_sha256"]
        and existing.get("source_topology_sha256")
        == deployment_body["source_topology_sha256"]
        and existing.get("mec") == deployment_body["mec"]
        and existing.get("homogeneity_checks", {}).get("ok") is True
    ):
        # Idempotent rebuild: keep the frozen file (and its protocol-recorded
        # SHA-256) byte-identical instead of changing only created_utc.
        return existing
    atomic_write_json(path, deployment_body)
    if verbose:
        print(
            f"[stage2n] deployment: {len(nodes)} homogeneous nodes, "
            f"sha256={deployment_body['deployment_sha256'][:16]}...",
            flush=True,
        )
    return deployment_body


def load_deployment() -> dict:
    payload = read_json(DEPLOYMENT_DIR / "deployment.json")
    if payload is None:
        raise FileNotFoundError(DEPLOYMENT_DIR / "deployment.json")
    if payload.get("homogeneity_checks", {}).get("ok") is not True:
        raise ValueError("deployment homogeneity audit did not pass")
    expected = canonical_sha256({
        "nodes": payload["nodes"],
        "mec": payload["mec"],
        "source_topology_sha256": payload["source_topology_sha256"],
    })
    if payload.get("deployment_sha256") != expected:
        raise ValueError("deployment SHA-256 mismatch")
    if int(payload["mec"]["num_nodes"]) != len(payload["nodes"]):
        raise ValueError("deployment node count mismatch")
    return payload


def deployment_topology(deployment: Mapping | None = None) -> dict:
    payload = dict(deployment or load_deployment())
    return {
        "version": payload["version"],
        "topology_id": payload["deployment_id"],
        "role": payload["deployment_id"],
        "scenario": "homogeneous_fixed",
        "topology_seed": int(payload["source_topology_seed"]),
        "nodes": [dict(node) for node in payload["nodes"]],
        "deployment_sha256": payload["deployment_sha256"],
        "homogeneity_checks": dict(payload["homogeneity_checks"]),
        "final_test_read": False,
    }


# ---------------------------------------------------------------------------
# MEC checkpoint provenance
# ---------------------------------------------------------------------------


def verify_mec_checkpoint(verbose: bool = False) -> dict:
    """Verify the frozen Transformer-M checkpoint and its real records.

    The checkpoint is selected by Stage-2j **Dev Val 180 SR**, where it is the
    highest of the five MECs (57.22%).  The already-published Final Test 300
    record (55.33% SR) is read only as provenance, never for selection.
    """
    if not MEC_CHECKPOINT.is_file():
        raise FileNotFoundError(MEC_CHECKPOINT)
    actual = sha256_file(MEC_CHECKPOINT)
    if actual != MEC_CHECKPOINT_SHA256:
        raise ValueError(
            f"Transformer-M checkpoint SHA-256 {actual} != "
            f"{MEC_CHECKPOINT_SHA256}"
        )
    dev = read_json(MEC_DEV_METRICS)
    test = read_json(MEC_TEST_METRICS)
    if dev is None or test is None:
        raise FileNotFoundError(
            f"missing Stage-2j evaluation records: {MEC_DEV_METRICS}, "
            f"{MEC_TEST_METRICS}"
        )
    dev_meta, dev_overall = dev["meta"], dev["overall"]
    test_meta, test_overall = test["meta"], test["overall"]
    if dev_meta.get("checkpoint_sha256") != MEC_CHECKPOINT_SHA256:
        raise ValueError("Dev Val 180 record is not for this checkpoint")
    if test_meta.get("checkpoint_sha256") != MEC_CHECKPOINT_SHA256:
        raise ValueError("Final Test 300 record is not for this checkpoint")
    if dev_meta.get("split") != "dev_val":
        raise ValueError("Dev record is not a dev_val evaluation")
    if int(dev_overall.get("episodes", -1)) != 180:
        raise ValueError("Dev record does not cover 180 episodes")
    if int(test_overall.get("episodes", -1)) != 300:
        raise ValueError("Test record does not cover 300 episodes")
    record = {
        "model": MEC_MODEL,
        "checkpoint": str(MEC_CHECKPOINT),
        "checkpoint_sha256": actual,
        "parameters": int(dev_meta.get("parameters", 0)),
        "selected_by": "Stage-2j Dev Val 180 SR (highest of five MECs)",
        "dev_val_180": {
            "episodes": int(dev_overall["episodes"]),
            "sr": float(dev_overall["sr"]),
            "osr": float(dev_overall["osr"]),
            "spl": float(dev_overall["spl"]),
            "ndtw": float(dev_overall["ndtw"]),
            "metrics_path": str(MEC_DEV_METRICS),
            "metrics_sha256": sha256_file(MEC_DEV_METRICS),
        },
        "final_test_300_reference": {
            "episodes": int(test_overall["episodes"]),
            "sr": float(test_overall["sr"]),
            "osr": float(test_overall["osr"]),
            "spl": float(test_overall["spl"]),
            "ndtw": float(test_overall["ndtw"]),
            "metrics_path": str(MEC_TEST_METRICS),
            "metrics_sha256": sha256_file(MEC_TEST_METRICS),
            "note": "published Stage-2j record; not used for selection",
        },
        "frozen": True,
        "retrained": False,
        "final_test_read": False,
    }
    if verbose:
        print(
            f"[stage2n] MEC {MEC_MODEL}: Dev180 SR "
            f"{100*record['dev_val_180']['sr']:.2f}%, Test300 SR "
            f"{100*record['final_test_300_reference']['sr']:.2f}%",
            flush=True,
        )
    return record


# ---------------------------------------------------------------------------
# Latency-only reward, convergence and checkpoint selection
# ---------------------------------------------------------------------------


def latency_reward_increment(e2e_ms, mean_prev, step_index):
    """Return ``(new_running_mean, r_t)`` for one executed step.

    ``step_index`` is 1-based.  Starting from ``mean_prev = 0`` and summing
    over the episode telescopes to ``-0.25 * mean_E2E / 100`` exactly.
    """
    step_index = int(step_index)
    if step_index < 1:
        raise ValueError("step_index must be >= 1")
    mean_prev = float(mean_prev)
    e2e_ms = float(e2e_ms)
    if not np.isfinite(e2e_ms) or e2e_ms < 0.0:
        raise ValueError(f"invalid E2E delay {e2e_ms}")
    mean_new = mean_prev + (e2e_ms - mean_prev) / float(step_index)
    reward = (
        -LATENCY_REWARD_WEIGHT / LATENCY_REWARD_DIVISOR
        * (mean_new - mean_prev)
    )
    return float(mean_new), float(reward)


def latency_return_from_mean(mean_e2e_ms):
    return float(
        -LATENCY_REWARD_WEIGHT / LATENCY_REWARD_DIVISOR
        * float(mean_e2e_ms)
    )


def latency_rewards(delays):
    mean = 0.0
    rewards = []
    for step_index, delay in enumerate(delays, start=1):
        mean, reward = latency_reward_increment(delay, mean, step_index)
        rewards.append(reward)
    return np.asarray(rewards, dtype=np.float64), float(mean)


def convergence_status(validation_rows: Sequence[Mapping], reached_cap: bool):
    """Dev mean-E2E convergence rule fixed before training.

    The rule looks only at Dev mean E2E (lower is better):
    ``max(last3) - min(last3) <= 0.01 * best`` and
    ``last <= best * 1.01``.  A run that reaches 4000 episodes without
    meeting it is explicitly ``not_converged``.
    """
    e2e = [float(row["E2E_ms"]) for row in validation_rows]
    episodes = [int(row["episode"]) for row in validation_rows]
    if not e2e:
        return {
            "status": "training" if not reached_cap else "not_converged",
            "converged": False,
            "reached_cap": bool(reached_cap),
            "validations": 0,
        }
    best = float(min(e2e))
    best_index = int(np.argmin(e2e))
    last3 = e2e[-3:]
    spread = float(max(last3) - min(last3))
    spread_fraction = spread / best if best > 0 else float("inf")
    final_gap = float(e2e[-1] - best)
    final_gap_fraction = final_gap / best if best > 0 else float("inf")
    converged = bool(
        len(e2e) >= 3
        and spread_fraction <= CONVERGENCE_SPREAD_FRACTION
        and final_gap_fraction <= CONVERGENCE_FINAL_GAP_FRACTION
    )
    if converged:
        status = "converged"
    elif not reached_cap:
        status = "training"
    else:
        status = "not_converged"
    return {
        "status": status,
        "converged": converged,
        "reached_cap": bool(reached_cap),
        "validations": len(e2e),
        "best_e2e_ms": best,
        "best_episode": episodes[best_index],
        "final_e2e_ms": e2e[-1],
        "final_gap_ms": final_gap,
        "final_gap_fraction": final_gap_fraction,
        "last3_e2e_ms": last3,
        "last3_spread_ms": spread,
        "last3_spread_fraction": spread_fraction,
        "criterion": CONVERGENCE_CRITERION,
    }


def select_best_validation(validation_rows: Sequence[Mapping]):
    """Lowest Dev mean E2E; ties: higher SR, then earlier episode."""
    if not validation_rows:
        return None
    return min(
        validation_rows,
        key=lambda row: (
            float(row["E2E_ms"]),
            -float(row.get("SR", 0.0)),
            int(row["episode"]),
        ),
    )


# ---------------------------------------------------------------------------
# Episode summaries
# ---------------------------------------------------------------------------


def summarize_episode_rows(rows: Sequence[Mapping], key: str | None = None):
    rows = list(rows)
    groups: dict[str, list] = {}
    for row in rows:
        value = str(row[key]) if key else "overall"
        groups.setdefault(value, []).append(row)
    output = {}
    for value, items in groups.items():
        e2e = np.asarray(
            [float(item["mean_e2e_ms"]) for item in items], dtype=np.float64
        )
        p95 = np.asarray(
            [float(item["p95_e2e_ms"]) for item in items], dtype=np.float64
        )
        step_e2e = [
            float(v) for item in items
            for v in item.get("step_e2e_ms", [])
        ]
        output[value] = {
            "episodes": len(items),
            "E2E_ms": float(e2e.mean()),
            "E2E_std_ms": float(e2e.std(ddof=0)),
            "P95_E2E_ms": (
                float(np.percentile(step_e2e, 95))
                if step_e2e else float(p95.mean())
            ),
            "LatencyReturn": float(np.mean([
                float(item.get("latency_return",
                                latency_return_from_mean(item["mean_e2e_ms"])))
                for item in items
            ])),
            "SR": float(np.mean([float(item["success"]) for item in items])),
            "OSR": float(np.mean([float(item.get("osr", 0.0)) for item in items])),
            "SPL": float(np.mean([float(item.get("spl", 0.0)) for item in items])),
            "nDTW": float(np.mean([float(item["ndtw"]) for item in items])),
            "delay_violation_rate": float(np.mean([
                float(item["delay_violation_rate"]) for item in items
            ])),
            "mean_steps": float(np.mean([float(item["steps"]) for item in items])),
        }
    return output


def summarize_latency_rows(rows: Sequence[Mapping]):
    return {
        "overall": summarize_episode_rows(rows).get("overall", {}),
        "by_band": summarize_episode_rows(rows, key="band"),
        "by_strategy": summarize_episode_rows(rows, key="strategy"),
    }


__all__ = [
    "STAGE",
    "STAGE2N_ROOT",
    "SPECS_DIR",
    "DEPLOYMENT_DIR",
    "RUNS_DIR",
    "EVAL_DIR",
    "LOG_DIR",
    "PROTOCOL_PATH",
    "EVIDENCE_DIR",
    "MAP_SEED",
    "TASK_SEED_BASE",
    "TRAIN_EPISODES",
    "DEV_EPISODES",
    "TEST_EPISODES",
    "BAND_COUNTS_TRAIN",
    "BAND_COUNTS_DEV",
    "BAND_COUNTS_TEST",
    "TRAIN_ORDER_SEED",
    "VALIDATION_EVERY",
    "AGENT_SEEDS",
    "MEC_MODEL",
    "MEC_CHECKPOINT",
    "MEC_CHECKPOINT_SHA256",
    "CAPABILITY_PRIOR_PATH",
    "DEPLOYMENT_SOURCE",
    "MEC_COMPUTE_FLOPS_PER_S",
    "MEC_BANDWIDTH_HZ",
    "MEC_COVERAGE_RADIUS_M",
    "MEC_NUM_NODES",
    "STATE_DIM",
    "ACTION_DIM",
    "STATE_MODE",
    "REWARD_MODE",
    "REWARD_DESCRIPTION",
    "LATENCY_RETURN_DESCRIPTION",
    "CONVERGENCE_CRITERION",
    "CHECKPOINT_SELECTION",
    "generate_split_tasks",
    "save_task_pool",
    "load_task_pool",
    "build_training_order",
    "ordered_train_specs",
    "dev_specs",
    "prepare_runner_spec",
    "isolation_audit",
    "distribution_report",
    "build_city",
    "build_homogeneous_deployment",
    "load_deployment",
    "deployment_topology",
    "verify_mec_checkpoint",
    "latency_reward_increment",
    "latency_return_from_mean",
    "latency_rewards",
    "convergence_status",
    "select_best_validation",
    "summarize_episode_rows",
    "summarize_latency_rows",
    "canonical_sha256",
    "sha256_file",
    "atomic_write_json",
    "read_json",
    "utc_now",
    "write_csv",
]
