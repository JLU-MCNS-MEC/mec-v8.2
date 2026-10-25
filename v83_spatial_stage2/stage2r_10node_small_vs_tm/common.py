"""Stage-2r: 10-node homogeneous Small vs Transformer-M deployments.

This module freezes everything that must be identical across the 12 runs:

* the exact same fixed city and Train-4000 / Dev-60 task pools as Stage-2n;
* one deterministic 10-node geometry (A10_fixed), shared by both deployments;
* two homogeneous deployments on those positions: all-Small and all-
  Transformer-M, with the frozen Stage-2j checkpoints;
* scheme-D reward, Pure-MEC v2 state with 10 candidates (194-D) and a 10-D
  action mask;
* agent seeds 11 / 22 / 33.

No Test task is loaded or inspected anywhere in this stage.  The 10-node
geometry is a maximin subset of the existing Stage-2l A_fixed 20 positions:
start at the A_fixed node nearest the city centre, then repeatedly add the
node farthest from the already selected set.  The rule uses geometry only,
never task outcomes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENV, NETWORK  # noqa: E402
from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    MODEL_NAMES,
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
)
from v83_spatial_stage2.stage2l_fixed_city import (  # noqa: E402
    canonical_sha256,
)
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (  # noqa: E402
    CANDIDATE_DIM,
    GLOBAL_DIM,
    STATE_DIM_PURE_MEC_V2,
)
from v83_spatial_stage2.stage2k_v2_common import (  # noqa: E402
    load_capability_prior,
)
from v83_spatial_stage2.stage2k_runtime import load_gab_config  # noqa: E402
from v83_spatial_stage2.stage2n_common import (  # noqa: E402
    MAP_SEED,
    TRAIN_EPISODES,
    VALIDATION_EVERY,
    dev_specs,
    load_task_pool,
    ordered_train_specs,
    prepare_runner_spec,
)

STAGE = "stage2r_10node_small_vs_tm"
STAGE_TITLE = "Stage-2r: 10-node Small vs Transformer-M homogeneous MEC"

RUNS_ROOT = Path("runs/v83_spatial_stage2/stage2r_10node_small_vs_tm")
EVIDENCE_DIR = Path("v83_spatial_stage2/stage2r_10node_small_vs_tm")
FIGURES_DIR = EVIDENCE_DIR / "figures"
FIGURE_DATA_DIR = EVIDENCE_DIR / "figure_data"

# Frozen task protocol (reused read-only; Test is never opened).
TASK_SOURCE_STAGE = "Stage-2n"
TASK_POOL_VERSION = "stage2n-fixed-city-latency-v1"
TRAIN_ORDER_SEED = 20261022
DEV_EPISODES = 60
MAX_EPISODES = TRAIN_EPISODES
FORMAL_VALIDATION = VALIDATION_EVERY

# The only new experimental variables.
MEC_MODELS = ("Small", "Transformer-M")
MODEL_SLUGS = {"Small": "small", "Transformer-M": "transformer_m"}
MODEL_LABELS = {
    "Small": "10×Small",
    "Transformer-M": "10×Transformer-M",
}
ALGORITHMS = ("dqn", "ppo")
ALGORITHM_LABELS = {"dqn": "DQN", "ppo": "PPO"}
FORMAL_SEEDS = (11, 22, 33)
CONDITIONS = tuple(
    (model, algorithm) for model in MEC_MODELS for algorithm in ALGORITHMS
)
REWARD_SCHEME = "D"

# 10 homogeneous nodes.  Shared geometry, model-specific checkpoint/prior.
NUM_NODES = 10
ACTION_DIM = NUM_NODES
STATE_DIM = GLOBAL_DIM + NUM_NODES * CANDIDATE_DIM  # 4 + 10*19 = 194
MASK_DIM = NUM_NODES
STATE_MODE = "pure_mec_v2"
LEGACY_STATE_DIM = STATE_DIM_PURE_MEC_V2  # keep the 20-node contract visible

MEC_CHECKPOINTS = {
    "Small": Path("runs/v83_spatial_stage2/stage2j/Small/best.pt"),
    "Transformer-M": Path(
        "runs/v83_spatial_stage2/stage2j/Transformer-M/best.pt"
    ),
}
MEC_CHECKPOINT_SHA256 = {
    "Small": (
        "7349ec9d0c809a09dc2672a05053f1f89e69480b8802d6ca3b05944cea44e1f6"
    ),
    "Transformer-M": (
        "57c88b7d71e1a7f17862aaa8429afa8a041cba5e747f744f3a7febf47f273c6d"
    ),
}
MEC_MODEL_INDEX = {
    name: index for index, name in enumerate(MODEL_NAMES)
}
CAPABILITY_PRIOR_PATH = Path(
    "runs/v83_spatial_stage2/stage2k/capability_prior_v2.json"
)
MEC_DEV_METRICS = {
    model: Path(
        f"runs/v83_spatial_stage2/stage2k/quality_dev_val/{model}/metrics.json"
    )
    for model in MEC_MODELS
}

# Uniform high-tier node parameters, identical to Stage-2n.
MEC_COMPUTE_FLOPS_PER_S = 2.0e12
MEC_BANDWIDTH_HZ = 30.0e6
MEC_COVERAGE_RADIUS_M = float(NETWORK.coverage_radius_m)

# 10-node geometry: deterministic maximin subset of Stage-2l A_fixed.
DEPLOYMENT_SOURCE = Path(
    "runs/v83_spatial_stage2/stage2l_fixed_city/topologies/A_fixed.json"
)
NODE_SELECTION_RULE = (
    "A10_fixed: start at the A_fixed node nearest the city centre "
    "(750,750), then repeatedly add the node with the largest minimum "
    "distance to the already selected nodes; ties by lower A_fixed index. "
    "Finally sort the 10 selected indices ascending for node IDs M01..M10."
)

# Scheme D / convergence constants (identical to Stage-2o/2p).
TAU_MS = 100.0
H_MAX = 400
GAMMA = 1.0
CONVERGENCE_CRITERION = (
    "最近三次 Dev 验证的平均 E2E 时延波动 <= 最佳值的 1%，且最终值不比"
    "最佳值差超过 1%；训练预算 4000 Episodes 结束时仍不满足则标记未收敛"
)
CHECKPOINT_SELECTION = (
    "最低 Dev 平均 E2E 时延；E2E 完全相同时依次比较更高 SR、更早 Episode"
)
REWARD_DESCRIPTION = (
    "u_t=1-e2e_t/tau; r_t=u_t/H; terminal += (1-T/H)*(1-mean_E2E/tau); "
    "R_D=1-mean_E2E/tau"
)

# Guard rails copied from the parent protocol.
EXPECTED_CITY_FINGERPRINT = (
    "392500aa0df1d87b341f2fbd3df2fe4f2dc0405624a1ecc91cd33eca616bb182"
)
# Task-pool SHA-256 values are read from the frozen Stage-2n files at runtime.
TRAIN_BAND_COUNTS = {"short": 1334, "medium": 1333, "long": 1333}
DEV_BAND_COUNTS = {"short": 20, "medium": 20, "long": 20}


def model_slug(model: str) -> str:
    model = str(model)
    if model not in MODEL_SLUGS:
        raise ValueError(f"unknown MEC model {model!r}")
    return MODEL_SLUGS[model]


def run_dir(model: str, algorithm: str, seed: int) -> Path:
    return (
        RUNS_ROOT / model_slug(model)
        / f"{algorithm}_s{int(seed)}"
    )


def training_args(model: str, algorithm: str, seed: int):
    return ["--model", str(model), "--algorithm", str(algorithm),
            "--seed", str(int(seed))]


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


def select_a10_indices() -> list[int]:
    source = read_json(DEPLOYMENT_SOURCE)
    if source is None:
        raise FileNotFoundError(DEPLOYMENT_SOURCE)
    positions = np.asarray(
        [[float(node["x"]), float(node["y"])]
         for node in source["nodes"]],
        dtype=np.float64,
    )
    if positions.shape != (20, 2):
        raise ValueError("A_fixed must contain 20 nodes")
    center = np.asarray(
        [ENV.world_size_m / 2.0, ENV.world_size_m / 2.0],
        dtype=np.float64,
    )
    selected = [int(np.argmin(np.linalg.norm(positions - center, axis=1)))]
    while len(selected) < NUM_NODES:
        distances = np.linalg.norm(
            positions[:, None, :] - positions[selected][None, :, :], axis=2
        ).min(axis=1)
        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))
    if len(set(selected)) != NUM_NODES:
        raise RuntimeError("A10 selection produced duplicate indices")
    return sorted(selected)


def _load_prior():
    payload, flat = load_capability_prior(CAPABILITY_PRIOR_PATH)
    if payload.get("final_test_read") is not False:
        raise ValueError("capability prior claims a Final Test read")
    return payload, flat


def build_deployment(model: str, verbose: bool = False) -> dict:
    model = str(model)
    if model not in MEC_MODELS:
        raise ValueError(f"unknown model {model!r}")
    source = read_json(DEPLOYMENT_SOURCE)
    if source is None:
        raise FileNotFoundError(DEPLOYMENT_SOURCE)
    prior_payload, prior_flat = _load_prior()
    indices = select_a10_indices()
    source_sha = sha256_file(DEPLOYMENT_SOURCE)
    nodes = []
    for order, source_index in enumerate(indices):
        node = source["nodes"][source_index]
        nodes.append({
            "node_id": f"M{order + 1:02d}",
            "source_index": int(source_index),
            "x": float(node["x"]),
            "y": float(node["y"]),
            "model_type": model,
            "model_index": int(MEC_MODEL_INDEX[model]),
            "quality_prior": float(prior_flat[model]["ndtw"]),
            "compute_capacity_flops_per_s": float(MEC_COMPUTE_FLOPS_PER_S),
            "bandwidth_hz": float(MEC_BANDWIDTH_HZ),
            "coverage_radius_m": float(MEC_COVERAGE_RADIUS_M),
        })
    checks = {
        "num_nodes_10": len(nodes) == NUM_NODES,
        "positions_unique": len({(n["x"], n["y"]) for n in nodes}) == NUM_NODES,
        "one_model_type": len({n["model_type"] for n in nodes}) == 1,
        "one_compute_capacity": len({
            n["compute_capacity_flops_per_s"] for n in nodes
        }) == 1,
        "one_bandwidth": len({n["bandwidth_hz"] for n in nodes}) == 1,
        "one_coverage_radius": len({
            n["coverage_radius_m"] for n in nodes
        }) == 1,
        "one_quality_prior": len({n["quality_prior"] for n in nodes}) == 1,
        "all_selected_model": all(n["model_type"] == model for n in nodes),
        "positions_inside_city": all(
            0.0 <= n["x"] <= ENV.world_size_m
            and 0.0 <= n["y"] <= ENV.world_size_m
            for n in nodes
        ),
        "source_indices_ascending": indices == sorted(indices),
    }
    checks["ok"] = all(checks.values())
    if not checks["ok"]:
        raise RuntimeError(f"deployment audit failed: {checks}")
    mec = {
        "model": model,
        "model_index": int(MEC_MODEL_INDEX[model]),
        "checkpoint": str(MEC_CHECKPOINTS[model]),
        "checkpoint_sha256": MEC_CHECKPOINT_SHA256[model],
        "compute_capacity_flops_per_s": float(MEC_COMPUTE_FLOPS_PER_S),
        "bandwidth_hz": float(MEC_BANDWIDTH_HZ),
        "coverage_radius_m": float(MEC_COVERAGE_RADIUS_M),
        "tx_power_w": float(NETWORK.default_tx_power_w),
        "num_nodes": NUM_NODES,
        "frozen": True,
        "requires_grad": False,
    }
    body = {
        "version": "stage2r-homogeneous-deployment-v1",
        "stage": STAGE,
        "created_utc": utc_now(),
        "deployment_id": f"stage2r-A10-{MODEL_SLUGS[model]}",
        "description": (
            f"10 MEC nodes on the A10_fixed geometry; every node uses the "
            f"same frozen {model} weights, compute capacity, bandwidth, "
            "coverage radius and communication parameters"
        ),
        "source_topology": str(DEPLOYMENT_SOURCE),
        "source_topology_sha256": source_sha,
        "source_topology_seed": int(source.get("topology_seed", -1)),
        "node_selection_rule": NODE_SELECTION_RULE,
        "selected_source_indices": indices,
        "capability_prior": {
            "path": str(CAPABILITY_PRIOR_PATH),
            "sha256": sha256_file(CAPABILITY_PRIOR_PATH),
            "model": model,
            "sr": float(prior_flat[model]["sr"]),
            "osr": float(prior_flat[model]["osr"]),
            "spl": float(prior_flat[model]["spl"]),
            "ndtw": float(prior_flat[model]["ndtw"]),
        },
        "mec": mec,
        "homogeneity_checks": checks,
        "nodes": nodes,
        "final_test_read": False,
    }
    body["deployment_sha256"] = canonical_sha256({
        "nodes": nodes,
        "mec": mec,
        "source_topology_sha256": source_sha,
    })
    path = RUNS_ROOT / "deployments" / f"{MODEL_SLUGS[model]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_json(path)
    if (
        existing is not None
        and existing.get("deployment_sha256")
        == body["deployment_sha256"]
        and existing.get("mec") == mec
    ):
        return existing
    atomic_write_json(path, body)
    if verbose:
        print(
            f"[stage2r] {model}: {NUM_NODES} nodes, "
            f"sha256={body['deployment_sha256'][:16]}...",
            flush=True,
        )
    return body


def load_deployment(model: str) -> dict:
    path = RUNS_ROOT / "deployments" / f"{MODEL_SLUGS[model]}.json"
    payload = read_json(path)
    if payload is None:
        raise FileNotFoundError(path)
    expected = canonical_sha256({
        "nodes": payload["nodes"],
        "mec": payload["mec"],
        "source_topology_sha256": payload["source_topology_sha256"],
    })
    if payload.get("deployment_sha256") != expected:
        raise ValueError("deployment SHA-256 mismatch")
    if payload.get("homogeneity_checks", {}).get("ok") is not True:
        raise ValueError("deployment homogeneity audit did not pass")
    if len(payload["nodes"]) != NUM_NODES:
        raise ValueError("deployment node count mismatch")
    if {node["model_type"] for node in payload["nodes"]} != {model}:
        raise ValueError("deployment model mismatch")
    return payload


def deployment_topology(model: str, deployment: Mapping | None = None):
    payload = dict(deployment or load_deployment(model))
    return {
        "version": payload["version"],
        "topology_id": payload["deployment_id"],
        "role": payload["deployment_id"],
        "scenario": "homogeneous_10node",
        "topology_seed": int(payload["source_topology_seed"]),
        "nodes": [dict(node) for node in payload["nodes"]],
        "deployment_sha256": payload["deployment_sha256"],
        "homogeneity_checks": dict(payload["homogeneity_checks"]),
        "final_test_read": False,
    }


def verify_checkpoint(model: str) -> dict:
    model = str(model)
    path = MEC_CHECKPOINTS[model]
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != MEC_CHECKPOINT_SHA256[model]:
        raise ValueError(
            f"{model} checkpoint SHA-256 {actual} != "
            f"{MEC_CHECKPOINT_SHA256[model]}"
        )
    metrics_path = MEC_DEV_METRICS[model]
    payload = read_json(metrics_path)
    if payload is None:
        raise FileNotFoundError(metrics_path)
    meta = payload["meta"]
    overall = payload["overall"]
    if meta.get("checkpoint_sha256") != MEC_CHECKPOINT_SHA256[model]:
        raise ValueError(f"{metrics_path} is not for this checkpoint")
    if meta.get("split") != "dev_val" or int(overall["episodes"]) != 180:
        raise ValueError(f"{metrics_path} is not a Dev-180 record")
    if meta.get("final_test_read") is not False:
        raise ValueError(f"{metrics_path} claims a Final Test read")
    return {
        "model": model,
        "checkpoint": str(path),
        "checkpoint_sha256": actual,
        "dev_val_180": {
            "episodes": int(overall["episodes"]),
            "sr": float(overall["sr"]),
            "osr": float(overall["osr"]),
            "spl": float(overall["spl"]),
            "ndtw": float(overall["ndtw"]),
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
        "frozen": True,
        "retrained": False,
        "final_test_read": False,
    }


# ---------------------------------------------------------------------------
# Task bundle
# ---------------------------------------------------------------------------


def load_bundle(model: str) -> dict:
    model = str(model)
    train_pool = load_task_pool("train")
    dev_pool = load_task_pool("dev")
    if train_pool.get("split") != "train" or dev_pool.get("split") != "dev":
        raise ValueError("wrong task split")
    train = [prepare_runner_spec(item) for item in ordered_train_specs()]
    dev = [prepare_runner_spec(item) for item in dev_specs()]
    if len(train) != TRAIN_EPISODES:
        raise ValueError(f"train tasks {len(train)} != {TRAIN_EPISODES}")
    if len(dev) != DEV_EPISODES:
        raise ValueError(f"dev tasks {len(dev)} != {DEV_EPISODES}")
    train_ids = {item["episode_uid"] for item in train}
    dev_ids = {item["episode_uid"] for item in dev}
    if train_ids & dev_ids:
        raise ValueError("Train/Dev task IDs overlap")
    if any(int(item["map_seed"]) != MAP_SEED for item in train + dev):
        raise ValueError("task map seed mismatch")
    fingerprints = {str(item["city_fingerprint"]) for item in train + dev}
    if fingerprints != {EXPECTED_CITY_FINGERPRINT}:
        raise ValueError(f"city fingerprint mismatch: {fingerprints}")
    for band, count in TRAIN_BAND_COUNTS.items():
        actual = sum(1 for item in train if item["band"] == band)
        if actual != count:
            raise ValueError(f"train {band} {actual} != {count}")
    for band, count in DEV_BAND_COUNTS.items():
        actual = sum(1 for item in dev if item["band"] == band)
        if actual != count:
            raise ValueError(f"dev {band} {actual} != {count}")
    deployment = build_deployment(model)
    topology = deployment_topology(model, deployment)
    prior_payload, flat = _load_prior()
    checkpoint = verify_checkpoint(model)
    return {
        "model": model,
        "train_specs": train,
        "dev_specs": dev,
        "deployment": deployment,
        "topology": topology,
        "quality_prior": flat,
        "prior_payload": prior_payload,
        "checkpoint_record": checkpoint,
        "gab_config": load_gab_config(),
        "train_pool": train_pool,
        "dev_pool": dev_pool,
        "test_read": False,
    }


def artifact_hashes(bundle: Mapping) -> dict:
    return {
        "train_tasks": bundle["train_pool"]["tasks_sha256"],
        "dev_tasks": bundle["dev_pool"]["tasks_sha256"],
        "training_order": canonical_sha256([
            item["episode_uid"] for item in bundle["train_specs"]
        ]),
        "deployment": bundle["deployment"]["deployment_sha256"],
        "mec_checkpoint": MEC_CHECKPOINT_SHA256[bundle["model"]],
        "capability_prior": sha256_file(CAPABILITY_PRIOR_PATH),
        "deployment_source": bundle["deployment"][
            "source_topology_sha256"
        ],
    }


def source_hashes() -> dict:
    files = sorted(Path(__file__).resolve().parent.glob("*.py")) + [
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2k_jointr_v2_agent.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2k_ppo_agent.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2k_pure_mec_v2_state.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2k_runtime.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2n_common.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2n_lanes.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2n_runtime.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2n_train.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2o" / "rewards.py",
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in files
    }


__all__ = [
    "STAGE", "STAGE_TITLE", "RUNS_ROOT", "EVIDENCE_DIR", "FIGURES_DIR",
    "FIGURE_DATA_DIR", "TASK_SOURCE_STAGE", "TRAIN_ORDER_SEED",
    "DEV_EPISODES", "MAX_EPISODES", "FORMAL_VALIDATION", "MEC_MODELS",
    "MODEL_SLUGS", "MODEL_LABELS", "ALGORITHMS", "ALGORITHM_LABELS",
    "FORMAL_SEEDS", "CONDITIONS", "REWARD_SCHEME", "NUM_NODES",
    "ACTION_DIM", "STATE_DIM", "MASK_DIM", "STATE_MODE",
    "MEC_CHECKPOINTS", "MEC_CHECKPOINT_SHA256", "MEC_DEV_METRICS",
    "CAPABILITY_PRIOR_PATH", "DEPLOYMENT_SOURCE", "NODE_SELECTION_RULE",
    "TAU_MS", "H_MAX", "GAMMA", "CONVERGENCE_CRITERION",
    "CHECKPOINT_SELECTION", "REWARD_DESCRIPTION", "EXPECTED_CITY_FINGERPRINT",
    "TRAIN_BAND_COUNTS", "DEV_BAND_COUNTS", "model_slug", "run_dir",
    "training_args", "select_a10_indices", "build_deployment",
    "load_deployment", "deployment_topology", "verify_checkpoint",
    "load_bundle", "artifact_hashes", "source_hashes",
]
