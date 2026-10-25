"""Stage-2s: 5xSmall + 5xTransformer-M heterogeneous 10-node MEC deployment.

This module freezes everything that must be identical across the six formal
runs:

* the exact Stage-2n fixed city and Train-4000 / Dev-60 task pools and order;
* the same A10_fixed 10-node geometry as Stage-2r (source indices
  ``[1, 2, 4, 6, 7, 8, 11, 12, 18, 19]`` of Stage-2l ``A_fixed``);
* a deterministic, spatially balanced 5+5 model assignment.  The assignment
  is chosen from all C(10,5) subsets using a uniform city grid plus the
  Train/Dev start positions (geometry only, never navigation outcomes);
* one heterogeneous deployment: every node keeps the nominal 2.0 TFLOP/s
  compute capacity, 30 MHz bandwidth, 500 m coverage and 2.0 W transmit
  power, but inference latency is computed with an explicit **effective**
  throughput of 0.2 TFLOP/s;
* the frozen Stage-2j Small and Transformer-M checkpoints;
* Pure-MEC v2 state with 10 candidates (194-D) and a 10-D availability mask;
* scheme-D reward, ``tau=100 ms``, ``H=400`` and ``gamma=1.0``;
* agent seeds 101 / 202 / 303.

No Test task is loaded or inspected anywhere in this stage.  The model
assignment uses task start positions only to audit spatial balance; it never
reads success, navigation quality, target position or any policy outcome.
"""

from __future__ import annotations

import itertools
import math
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
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (  # noqa: E402
    CANDIDATE_DIM,
    GLOBAL_DIM,
)
from v83_spatial_stage2.stage2k_runtime import load_gab_config  # noqa: E402
from v83_spatial_stage2.stage2k_v2_common import (  # noqa: E402
    load_capability_prior,
)
from v83_spatial_stage2.stage2l_fixed_city import (  # noqa: E402
    canonical_sha256,
)
from v83_spatial_stage2.stage2n_common import (  # noqa: E402
    MAP_SEED,
    TRAIN_EPISODES,
    VALIDATION_EVERY,
    dev_specs,
    load_task_pool,
    ordered_train_specs,
    prepare_runner_spec,
)

STAGE = "stage2s_heterogeneous_10node_small_tm"
STAGE_TITLE = (
    "Stage-2s: 5xSmall + 5xTransformer-M heterogeneous 10-node MEC"
)

RUNS_ROOT = Path(f"runs/v83_spatial_stage2/{STAGE}")
EVIDENCE_DIR = Path(f"v83_spatial_stage2/{STAGE}")
FIGURES_DIR = EVIDENCE_DIR / "figures"
FIGURE_DATA_DIR = EVIDENCE_DIR / "figure_data"
DEPLOYMENT_PATH = EVIDENCE_DIR / "deployment_heterogeneous.json"

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
MODEL_LABELS = {"Small": "Small", "Transformer-M": "Transformer-M"}
ALGORITHMS = ("dqn", "ppo")
ALGORITHM_LABELS = {"dqn": "DQN", "ppo": "PPO"}
FORMAL_SEEDS = (101, 202, 303)
CONDITIONS = tuple(
    f"{algorithm}_s{seed}"
    for algorithm in ALGORITHMS
    for seed in FORMAL_SEEDS
)
REWARD_SCHEME = "D"

# Heterogeneous 10-node deployment.  The nominal capacity stays 2.0 TFLOP/s;
# the explicit effective inference throughput is 0.2 TFLOP/s.
NUM_NODES = 10
SMALL_NODES = 5
TRANSFORMER_NODES = 5
ACTION_DIM = NUM_NODES
STATE_DIM = GLOBAL_DIM + NUM_NODES * CANDIDATE_DIM  # 4 + 10*19 = 194
MASK_DIM = NUM_NODES
STATE_MODE = "pure_mec_v2"

MEC_NOMINAL_COMPUTE_FLOPS_PER_S = 2.0e12
MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S = 0.2e12
MEC_BANDWIDTH_HZ = 30.0e6
MEC_COVERAGE_RADIUS_M = float(NETWORK.coverage_radius_m)
MEC_TX_POWER_W = float(NETWORK.default_tx_power_w)

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

# 10-node geometry: the Stage-2r A10_fixed subset of Stage-2l A_fixed.
DEPLOYMENT_SOURCE = Path(
    "runs/v83_spatial_stage2/stage2l_fixed_city/topologies/A_fixed.json"
)
A10_SOURCE_INDICES = (1, 2, 4, 6, 7, 8, 11, 12, 18, 19)
A10_NODE_SELECTION_RULE = (
    "A10_fixed: start at the A_fixed node nearest the city centre "
    "(750,750), then repeatedly add the node with the largest minimum "
    "distance to the already selected nodes; ties by lower A_fixed index. "
    "Finally sort the 10 selected indices ascending for node IDs M01..M10."
)
ASSIGNMENT_RULE = (
    "5+5 spatial balance: enumerate all 252 five-node subsets of A10_fixed; "
    "for each subset and its complement compute 500 m coverage on a "
    "deterministic 76x76 uniform city grid (0..1500 m, 20 m spacing) and on "
    "all Train-4000+Dev-60 start positions (geometry only). Rank by "
    "(1) larger minimum grid coverage, (2) larger minimum start coverage, "
    "(3) smaller mean-nearest-start-distance gap, (4) smaller worst mean "
    "nearest distance, (5) lexicographic node indices. The selected subset "
    "that contains the A10 node nearest the city centre is assigned to "
    "Small; its complement is assigned to Transformer-M. Node IDs stay in "
    "A10 order M01..M10."
)
GRID_COVERAGE_POINTS = 76
GRID_COVERAGE_RADIUS_M = 500.0

# Scheme D / convergence constants (identical to Stage-2o/2p/2r).
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
TRAIN_BAND_COUNTS = {"short": 1334, "medium": 1333, "long": 1333}
DEV_BAND_COUNTS = {"short": 20, "medium": 20, "long": 20}


def model_slug(model: str) -> str:
    model = str(model)
    if model not in MODEL_SLUGS:
        raise ValueError(f"unknown MEC model {model!r}")
    return MODEL_SLUGS[model]


def condition_label(algorithm: str, seed: int) -> str:
    return f"{ALGORITHM_LABELS[str(algorithm)]}|s{int(seed)}"


def run_dir(algorithm: str, seed: int) -> Path:
    return RUNS_ROOT / f"{algorithm}_s{int(seed)}"


def training_args(algorithm: str, seed: int):
    return ["--algorithm", str(algorithm), "--seed", str(int(seed))]


# ---------------------------------------------------------------------------
# Geometry and model assignment
# ---------------------------------------------------------------------------


def select_a10_indices() -> list[int]:
    """Exact Stage-2r A10_fixed selection rule (geometry only)."""
    source = read_json(DEPLOYMENT_SOURCE)
    if source is None:
        raise FileNotFoundError(DEPLOYMENT_SOURCE)
    positions = np.asarray(
        [[float(node["x"]), float(node["y"])] for node in source["nodes"]],
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


def _grid() -> np.ndarray:
    axis = np.linspace(
        0.0, float(ENV.world_size_m), GRID_COVERAGE_POINTS
    )
    xx, yy = np.meshgrid(axis, axis)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def _coverage(
    points: np.ndarray,
    node_positions: np.ndarray,
    radius_m: float = GRID_COVERAGE_RADIUS_M,
):
    points = np.asarray(points, dtype=np.float64)
    node_positions = np.asarray(node_positions, dtype=np.float64)
    distances = np.linalg.norm(
        points[:, None, :] - node_positions[None, :, :], axis=2
    ).min(axis=1)
    return float(np.mean(distances <= float(radius_m))), float(
        distances.mean()
    )


def select_model_assignment(positions, start_points) -> dict:
    """Deterministically choose the spatially balanced 5+5 assignment."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (NUM_NODES, 2):
        raise ValueError(f"expected {NUM_NODES} positions, got {positions.shape}")
    starts = np.asarray(start_points, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[1] != 2 or len(starts) == 0:
        raise ValueError("start_points must be a non-empty N x 2 array")
    grid = _grid()
    center = np.asarray(
        [ENV.world_size_m / 2.0, ENV.world_size_m / 2.0], dtype=np.float64
    )
    centre_node = int(np.argmin(np.linalg.norm(positions - center, axis=1)))
    candidates = []
    for subset in itertools.combinations(range(NUM_NODES), SMALL_NODES):
        complement = tuple(
            index for index in range(NUM_NODES) if index not in subset
        )
        grid_a, _ = _coverage(grid, positions[list(subset)])
        grid_b, _ = _coverage(grid, positions[list(complement)])
        start_a, mean_a = _coverage(starts, positions[list(subset)])
        start_b, mean_b = _coverage(starts, positions[list(complement)])
        candidates.append({
            "subset": tuple(int(index) for index in subset),
            "complement": tuple(int(index) for index in complement),
            "grid_a": float(grid_a),
            "grid_b": float(grid_b),
            "start_a": float(start_a),
            "start_b": float(start_b),
            "mean_a": float(mean_a),
            "mean_b": float(mean_b),
            "key": (
                -min(grid_a, grid_b),
                -min(start_a, start_b),
                abs(mean_a - mean_b),
                max(mean_a, mean_b),
                tuple(int(index) for index in subset),
            ),
        })
    candidates.sort(key=lambda item: item["key"])
    best = candidates[0]
    subset = best["subset"]
    complement = best["complement"]
    if centre_node not in subset:
        subset, complement = complement, subset
        grid_a, grid_b = best["grid_b"], best["grid_a"]
        start_a, start_b = best["start_b"], best["start_a"]
        mean_a, mean_b = best["mean_b"], best["mean_a"]
    else:
        grid_a, grid_b = best["grid_a"], best["grid_b"]
        start_a, start_b = best["start_a"], best["start_b"]
        mean_a, mean_b = best["mean_a"], best["mean_b"]
    small_indices = tuple(int(index) for index in subset)
    transformer_indices = tuple(int(index) for index in complement)
    assignment = {index: "Transformer-M" for index in range(NUM_NODES)}
    for index in small_indices:
        assignment[index] = "Small"
    balance = {
        "grid_coverage_small": float(grid_a),
        "grid_coverage_transformer_m": float(grid_b),
        "grid_coverage_gap": float(abs(grid_a - grid_b)),
        "grid_coverage_min": float(min(grid_a, grid_b)),
        "start_coverage_small": float(start_a),
        "start_coverage_transformer_m": float(start_b),
        "start_coverage_gap": float(abs(start_a - start_b)),
        "start_coverage_min": float(min(start_a, start_b)),
        "mean_nearest_start_m_small": float(mean_a),
        "mean_nearest_start_m_transformer_m": float(mean_b),
        "mean_nearest_gap_m": float(abs(mean_a - mean_b)),
        "centre_nearest_node_index": int(centre_node),
    }
    checks = {
        "five_small_five_transformer": (
            sum(1 for value in assignment.values() if value == "Small")
            == SMALL_NODES
        ),
        "grid_coverage_balanced": balance["grid_coverage_gap"] <= 0.10,
        "start_coverage_balanced": balance["start_coverage_gap"] <= 0.10,
        "grid_coverage_min_80pct": balance["grid_coverage_min"] >= 0.80,
        "start_coverage_min_80pct": balance["start_coverage_min"] >= 0.80,
    }
    checks["ok"] = all(checks.values())
    if not checks["ok"]:
        raise RuntimeError(f"model assignment audit failed: {checks} {balance}")
    return {
        "assignment_rule": ASSIGNMENT_RULE,
        "small_node_indices": list(small_indices),
        "transformer_node_indices": list(transformer_indices),
        "model_by_source_index": {
            str(index): assignment[index] for index in range(NUM_NODES)
        },
        "balance": balance,
        "checks": checks,
        "candidates_considered": len(candidates),
        "selected_rank": 1,
    }


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


def _load_prior():
    payload, flat = load_capability_prior(CAPABILITY_PRIOR_PATH)
    if payload.get("final_test_read") is not False:
        raise ValueError("capability prior claims a Final Test read")
    return payload, flat


def _task_starts():
    train = load_task_pool("train")
    dev = load_task_pool("dev")
    starts = [
        list(item["start"]) for item in train["tasks"] + dev["tasks"]
    ]
    return np.asarray(starts, dtype=np.float64)


def build_deployment(verbose: bool = False) -> dict:
    source = read_json(DEPLOYMENT_SOURCE)
    if source is None:
        raise FileNotFoundError(DEPLOYMENT_SOURCE)
    prior_payload, prior_flat = _load_prior()
    indices = select_a10_indices()
    if tuple(indices) != A10_SOURCE_INDICES:
        raise RuntimeError(
            f"A10 indices {indices} != frozen Stage-2r {A10_SOURCE_INDICES}"
        )
    positions = np.asarray(
        [[float(source["nodes"][index]["x"]),
          float(source["nodes"][index]["y"])] for index in indices],
        dtype=np.float64,
    )
    starts = _task_starts()
    assignment = select_model_assignment(positions, starts)
    source_sha = sha256_file(DEPLOYMENT_SOURCE)
    nodes = []
    for order, source_index in enumerate(indices):
        node = source["nodes"][source_index]
        model = assignment["model_by_source_index"][str(order)]
        nodes.append({
            "node_id": f"M{order + 1:02d}",
            "source_index": int(source_index),
            "x": float(node["x"]),
            "y": float(node["y"]),
            "model_type": model,
            "model_index": int(MEC_MODEL_INDEX[model]),
            "quality_prior": float(prior_flat[model]["ndtw"]),
            "compute_capacity_flops_per_s": float(
                MEC_NOMINAL_COMPUTE_FLOPS_PER_S
            ),
            "effective_inference_compute_flops_per_s": float(
                MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S
            ),
            "bandwidth_hz": float(MEC_BANDWIDTH_HZ),
            "coverage_radius_m": float(MEC_COVERAGE_RADIUS_M),
            "tx_power_w": float(MEC_TX_POWER_W),
        })
    model_counts = {
        model: sum(1 for node in nodes if node["model_type"] == model)
        for model in MEC_MODELS
    }
    checks = {
        "num_nodes_10": len(nodes) == NUM_NODES,
        "positions_unique": len({(n["x"], n["y"]) for n in nodes})
        == NUM_NODES,
        "five_small": model_counts["Small"] == SMALL_NODES,
        "five_transformer_m": (
            model_counts["Transformer-M"] == TRANSFORMER_NODES
        ),
        "two_model_types": set(model_counts) == set(MEC_MODELS),
        "positions_equal_a10": tuple(
            (round(n["x"], 6), round(n["y"], 6)) for n in nodes
        ) == tuple(
            (round(float(source["nodes"][i]["x"]), 6),
             round(float(source["nodes"][i]["y"]), 6))
            for i in indices
        ),
        "all_nominal_compute_2tf": all(
            n["compute_capacity_flops_per_s"]
            == MEC_NOMINAL_COMPUTE_FLOPS_PER_S for n in nodes
        ),
        "all_effective_inference_02tf": all(
            n["effective_inference_compute_flops_per_s"]
            == MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S for n in nodes
        ),
        "all_bandwidth_30mhz": all(
            n["bandwidth_hz"] == MEC_BANDWIDTH_HZ for n in nodes
        ),
        "all_coverage_500m": all(
            n["coverage_radius_m"] == MEC_COVERAGE_RADIUS_M for n in nodes
        ),
        "all_tx_power_equal": all(
            n["tx_power_w"] == MEC_TX_POWER_W for n in nodes
        ),
        "positions_inside_city": all(
            0.0 <= n["x"] <= ENV.world_size_m
            and 0.0 <= n["y"] <= ENV.world_size_m
            for n in nodes
        ),
        "source_indices_ascending": indices == sorted(indices),
        "spatial_balance_passed": bool(
            assignment["checks"].get("ok", False)
        ),
    }
    checks["ok"] = all(checks.values())
    if not checks["ok"]:
        raise RuntimeError(f"deployment audit failed: {checks}")
    mec = {
        "deployment_kind": "heterogeneous",
        "model_types": list(MEC_MODELS),
        "model_counts": model_counts,
        "checkpoints": {
            model: str(MEC_CHECKPOINTS[model]) for model in MEC_MODELS
        },
        "checkpoint_sha256": dict(MEC_CHECKPOINT_SHA256),
        "compute_capacity_flops_per_s": float(
            MEC_NOMINAL_COMPUTE_FLOPS_PER_S
        ),
        "effective_inference_compute_flops_per_s": float(
            MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S
        ),
        "compute_semantics": (
            "compute_capacity_flops_per_s is the nominal node capacity kept "
            "in topology/state features; inference latency uses only "
            "effective_inference_compute_flops_per_s"
        ),
        "bandwidth_hz": float(MEC_BANDWIDTH_HZ),
        "coverage_radius_m": float(MEC_COVERAGE_RADIUS_M),
        "tx_power_w": float(MEC_TX_POWER_W),
        "num_nodes": NUM_NODES,
        "frozen": True,
        "requires_grad": False,
    }
    body = {
        "version": "stage2s-heterogeneous-deployment-v1",
        "stage": STAGE,
        "created_utc": utc_now(),
        "deployment_id": "stage2s-A10-5small-5transformer_m",
        "description": (
            "10 MEC nodes on the Stage-2r A10_fixed geometry; five nodes run "
            "the frozen Small checkpoint and five run the frozen "
            "Transformer-M checkpoint. All nodes keep nominal 2.0 TFLOP/s, "
            "30 MHz, 500 m coverage and 2.0 W; inference latency uses an "
            "explicit 0.2 TFLOP/s effective throughput."
        ),
        "source_topology": str(DEPLOYMENT_SOURCE),
        "source_topology_sha256": source_sha,
        "source_topology_seed": int(source.get("topology_seed", -1)),
        "node_selection_rule": A10_NODE_SELECTION_RULE,
        "selected_source_indices": indices,
        "assignment": assignment,
        "capability_prior": {
            "path": str(CAPABILITY_PRIOR_PATH),
            "sha256": sha256_file(CAPABILITY_PRIOR_PATH),
            "models": {
                model: {
                    "sr": float(prior_flat[model]["sr"]),
                    "osr": float(prior_flat[model]["osr"]),
                    "spl": float(prior_flat[model]["spl"]),
                    "ndtw": float(prior_flat[model]["ndtw"]),
                }
                for model in MEC_MODELS
            },
        },
        "mec": mec,
        "checks": checks,
        "nodes": nodes,
        "final_test_read": False,
    }
    body["deployment_sha256"] = canonical_sha256({
        "nodes": nodes,
        "mec": mec,
        "source_topology_sha256": source_sha,
        "assignment_rule": ASSIGNMENT_RULE,
    })
    DEPLOYMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = read_json(DEPLOYMENT_PATH)
    if (
        existing is not None
        and existing.get("deployment_sha256")
        == body["deployment_sha256"]
        and existing.get("nodes") == nodes
        and existing.get("mec") == mec
    ):
        return existing
    atomic_write_json(DEPLOYMENT_PATH, body)
    (RUNS_ROOT / "deployments").mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        RUNS_ROOT / "deployments" / "heterogeneous.json", body
    )
    if verbose:
        print(
            f"[{STAGE}] heterogeneous deployment: "
            f"sha256={body['deployment_sha256'][:16]}... "
            f"Small={model_counts['Small']} "
            f"Transformer-M={model_counts['Transformer-M']}",
            flush=True,
        )
    return body


def load_deployment() -> dict:
    payload = read_json(DEPLOYMENT_PATH)
    if payload is None:
        raise FileNotFoundError(DEPLOYMENT_PATH)
    expected = canonical_sha256({
        "nodes": payload["nodes"],
        "mec": payload["mec"],
        "source_topology_sha256": payload["source_topology_sha256"],
        "assignment_rule": ASSIGNMENT_RULE,
    })
    if payload.get("deployment_sha256") != expected:
        raise ValueError("deployment SHA-256 mismatch")
    if payload.get("checks", {}).get("ok") is not True:
        raise ValueError("deployment audit did not pass")
    if len(payload["nodes"]) != NUM_NODES:
        raise ValueError("deployment node count mismatch")
    counts = {
        model: sum(
            1 for node in payload["nodes"]
            if node["model_type"] == model
        )
        for model in MEC_MODELS
    }
    if counts != {"Small": SMALL_NODES,
                  "Transformer-M": TRANSFORMER_NODES}:
        raise ValueError(f"deployment model mix mismatch: {counts}")
    return payload


def deployment_topology(deployment: Mapping | None = None):
    payload = dict(deployment or load_deployment())
    return {
        "version": payload["version"],
        "topology_id": payload["deployment_id"],
        "role": payload["deployment_id"],
        "scenario": "heterogeneous_10node",
        "topology_seed": int(payload["source_topology_seed"]),
        "nodes": [dict(node) for node in payload["nodes"]],
        "deployment_sha256": payload["deployment_sha256"],
        "checks": dict(payload["checks"]),
        "model_mix": dict(payload["mec"]["model_counts"]),
        "final_test_read": False,
    }


def verify_checkpoints() -> dict:
    records = {}
    for model in MEC_MODELS:
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
        if meta.get("split") != "dev_val" or int(
            overall["episodes"]
        ) != 180:
            raise ValueError(f"{metrics_path} is not a Dev-180 record")
        if meta.get("final_test_read") is not False:
            raise ValueError(f"{metrics_path} claims a Final Test read")
        records[model] = {
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
    return records


# ---------------------------------------------------------------------------
# Task bundle
# ---------------------------------------------------------------------------


def load_bundle() -> dict:
    train_pool = load_task_pool("train")
    dev_pool = load_task_pool("dev")
    if train_pool.get("split") != "train" or dev_pool.get("split") != "dev":
        raise ValueError("wrong task split")
    if train_pool.get("sealed") or dev_pool.get("sealed"):
        raise ValueError("Train/Dev pool unexpectedly sealed")
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
    deployment = build_deployment()
    topology = deployment_topology(deployment)
    prior_payload, flat = _load_prior()
    checkpoint_records = verify_checkpoints()
    nodes = np.asarray([
        [node["x"], node["y"]] for node in deployment["nodes"]
    ])
    starts = np.asarray([
        spec["start"] for spec in train + dev
    ])
    distances = np.linalg.norm(
        starts[:, None, :] - nodes[None, :, :], axis=2
    ).min(axis=1)
    if int((distances > MEC_COVERAGE_RADIUS_M).sum()) != 0:
        raise ValueError("not every Train/Dev start is covered by A10")
    return {
        "train_specs": train,
        "dev_specs": dev,
        "deployment": deployment,
        "topology": topology,
        "quality_prior": flat,
        "prior_payload": prior_payload,
        "checkpoint_records": checkpoint_records,
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
        "mec_checkpoints": dict(MEC_CHECKPOINT_SHA256),
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
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2k_latency.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2l_runtime.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2l_fixed_city.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2j_model.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2j_eval.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2n_common.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2n_runtime.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2n_train.py",
        PROJECT_ROOT / "v83_spatial_stage2" / "stage2o" / "rewards.py",
        PROJECT_ROOT / "latency_model.py",
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in files
    }


def checkpoint_fingerprint() -> str:
    return canonical_sha256(dict(MEC_CHECKPOINT_SHA256))


def deployment_sha256() -> str:
    return str(load_deployment()["deployment_sha256"])


__all__ = [
    "STAGE", "STAGE_TITLE", "RUNS_ROOT", "EVIDENCE_DIR", "FIGURES_DIR",
    "FIGURE_DATA_DIR", "DEPLOYMENT_PATH", "TASK_SOURCE_STAGE",
    "TASK_POOL_VERSION", "TRAIN_ORDER_SEED", "DEV_EPISODES", "MAX_EPISODES",
    "FORMAL_VALIDATION", "MEC_MODELS", "MODEL_SLUGS", "MODEL_LABELS",
    "ALGORITHMS", "ALGORITHM_LABELS", "FORMAL_SEEDS", "CONDITIONS",
    "REWARD_SCHEME", "NUM_NODES", "SMALL_NODES", "TRANSFORMER_NODES",
    "ACTION_DIM", "STATE_DIM", "MASK_DIM", "STATE_MODE",
    "MEC_NOMINAL_COMPUTE_FLOPS_PER_S",
    "MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S", "MEC_BANDWIDTH_HZ",
    "MEC_COVERAGE_RADIUS_M", "MEC_TX_POWER_W", "MEC_CHECKPOINTS",
    "MEC_CHECKPOINT_SHA256", "MEC_DEV_METRICS", "CAPABILITY_PRIOR_PATH",
    "DEPLOYMENT_SOURCE", "A10_SOURCE_INDICES", "A10_NODE_SELECTION_RULE",
    "ASSIGNMENT_RULE", "GRID_COVERAGE_POINTS", "GRID_COVERAGE_RADIUS_M",
    "TAU_MS", "H_MAX", "GAMMA", "CONVERGENCE_CRITERION",
    "CHECKPOINT_SELECTION", "REWARD_DESCRIPTION",
    "EXPECTED_CITY_FINGERPRINT", "TRAIN_BAND_COUNTS", "DEV_BAND_COUNTS",
    "model_slug", "condition_label", "run_dir", "training_args",
    "select_a10_indices", "select_model_assignment", "build_deployment",
    "load_deployment", "deployment_topology", "verify_checkpoints",
    "load_bundle", "artifact_hashes", "source_hashes",
    "checkpoint_fingerprint", "deployment_sha256",
]
