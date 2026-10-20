"""Stage-2k pure-MEC DQN state (322-D input + 20-D action mask).

The DQN input contains only MEC/communication/latency information:

    2-D normalized UAV position
  + 20 x 16-D candidate MEC features
  = 322

The 20-D availability mask is kept as a separate action mask.  It is applied
to Q-values when choosing a MEC; it is not duplicated inside the MLP input.
The total decision state is therefore 322 + 20 = 342.

Forbidden by design: visual/PCA features, goal tokens, goal visual similarity,
history visual, heatmap, GAB, map-retrieval features, navigation action,
episode step, goal position/distance, oracle action and success label.  The
frozen MEC backends still receive their Stage-2j inputs; those inputs are not
part of the DQN state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENV
from v83_spatial_stage2.stage2k_common import (
    BANDWIDTH_SCALE_HZ,
    COMPUTE_SCALE_FLOPS,
    E2E_CLIP_SCALE,
    LATENCY_SCALE_MS,
    MODEL_NAMES,
    POSITION_SCALE_M,
    RATE_SCALE_BPS,
)

NUM_NODES = 20
UAV_DIM = 2
CANDIDATE_DIM = 16
MASK_DIM = 20
STATE_DIM_PURE_MEC = UAV_DIM + NUM_NODES * CANDIDATE_DIM  # 322
DECISION_DIM_PURE_MEC = STATE_DIM_PURE_MEC + MASK_DIM  # 342
UAV_POSITION_SCALE_M = float(ENV.world_size_m)
SNR_SCALE_DB = 40.0
# Inference latency is ~0-10 ms in the unified model.  A dedicated 10 ms
# scale keeps it informative instead of nearly constant under a 100 ms scale.
INFERENCE_SCALE_MS = 10.0

CANDIDATE_FIELDS = (
    "relative_x",
    "relative_y",
    "distance",
    "compute_norm",
    "bandwidth_norm",
    "uplink_rate_norm",
    "communication_ms_norm",
    "inference_ms_norm",
    "e2e_ms_norm",
    "dev_val_quality_prior",
    "snr_db_norm",
    "model_Small",
    "model_Transformer-S",
    "model_Medium",
    "model_Transformer-M",
    "model_Large",
)


def _flat_quality(quality_prior):
    if quality_prior is None:
        return {}
    if not isinstance(quality_prior, dict):
        raise TypeError("quality_prior must be a flat dict or None")
    if "models" in quality_prior:
        return {
            name: float(quality_prior["models"][name]["mean_ndtw"])
            for name in MODEL_NAMES
        }
    return {
        name: float(value) for name, value in quality_prior.items()
        if name in MODEL_NAMES
    }


def _one_hot(index, size):
    output = np.zeros(size, dtype=np.float32)
    if 0 <= int(index) < int(size):
        output[int(index)] = 1.0
    return output


def pure_mec_candidate_features(
    nodes,
    uav_position,
    latency_model,
    quality_prior=None,
):
    """20 x 16-D MEC features; availability is a separate mask."""
    uav_position = np.asarray(uav_position, dtype=np.float64)[:2]
    quality = _flat_quality(quality_prior)
    features = np.zeros((NUM_NODES, CANDIDATE_DIM), dtype=np.float32)
    mask = np.zeros(NUM_NODES, dtype=bool)
    metrics = [None] * NUM_NODES
    distances = np.zeros(NUM_NODES, dtype=np.float64)

    for index, node in enumerate(nodes):
        dx = float(node["x"]) - float(uav_position[0])
        dy = float(node["y"]) - float(uav_position[1])
        distance = float(np.hypot(dx, dy))
        distances[index] = distance
        available = distance <= float(node["coverage_radius_m"])
        mask[index] = bool(available)
        node_quality = float(
            quality.get(
                node["model_type"], node.get("quality_prior", 0.0)
            )
        )
        values = [
            dx / POSITION_SCALE_M,
            dy / POSITION_SCALE_M,
            distance / POSITION_SCALE_M,
            float(node["compute_capacity_flops_per_s"])
            / COMPUTE_SCALE_FLOPS,
            float(node["bandwidth_hz"]) / BANDWIDTH_SCALE_HZ,
        ]
        if available:
            metric = latency_model.evaluate(
                node, uav_position, node["model_type"]
            )
            metrics[index] = metric
            values.extend([
                float(metric["uplink_rate_bps"]) / RATE_SCALE_BPS,
                float(metric["communication_ms"]) / LATENCY_SCALE_MS,
                min(
                    float(metric["inference_ms"]) / INFERENCE_SCALE_MS,
                    E2E_CLIP_SCALE,
                ),
                min(
                    float(metric["total_ms"]) / LATENCY_SCALE_MS,
                    E2E_CLIP_SCALE,
                ),
                node_quality,
                float(np.clip(
                    float(metric["snr_db"]) / SNR_SCALE_DB, -1.0, 1.0
                )),
            ])
        else:
            # Unavailable nodes keep static node properties and quality, but
            # zero all channel/latency fields.  Availability is in the mask.
            values.extend([0.0, 0.0, 0.0, 0.0, node_quality, 0.0])
        values.extend(
            _one_hot(node["model_index"], len(MODEL_NAMES)).tolist()
        )
        if len(values) != CANDIDATE_DIM:
            raise RuntimeError(
                f"candidate feature dim {len(values)} != {CANDIDATE_DIM}"
            )
        features[index] = np.asarray(values, dtype=np.float32)

    fallback = False
    if not mask.any():
        fallback = True
        nearest = int(np.argmin(distances))
        mask[nearest] = True
        node = nodes[nearest]
        metric = latency_model.evaluate(
            node, uav_position, node["model_type"]
        )
        metrics[nearest] = metric
        features[nearest, 5] = (
            float(metric["uplink_rate_bps"]) / RATE_SCALE_BPS
        )
        features[nearest, 6] = (
            float(metric["communication_ms"]) / LATENCY_SCALE_MS
        )
        features[nearest, 7] = min(
            float(metric["inference_ms"]) / INFERENCE_SCALE_MS,
            E2E_CLIP_SCALE,
        )
        features[nearest, 8] = min(
            float(metric["total_ms"]) / LATENCY_SCALE_MS, E2E_CLIP_SCALE
        )
        features[nearest, 10] = float(np.clip(
            float(metric["snr_db"]) / SNR_SCALE_DB, -1.0, 1.0
        ))
    return features, mask, metrics, fallback


def build_pure_mec_state(uav_position, candidate20x16):
    """Build the 322-D DQN input: 2-D UAV position + 20 x 16 candidate."""
    uav = np.asarray(uav_position, dtype=np.float32).reshape(-1)[:2]
    candidate = np.asarray(candidate20x16, dtype=np.float32)
    if candidate.shape != (NUM_NODES, CANDIDATE_DIM):
        raise ValueError(f"candidate shape {candidate.shape}")
    uav_normalized = uav / np.float32(UAV_POSITION_SCALE_M)
    state = np.concatenate([uav_normalized, candidate.reshape(-1)])
    if state.shape != (STATE_DIM_PURE_MEC,):
        raise RuntimeError(f"pure-MEC state dim {state.shape}")
    return state.astype(np.float32)


def state_layout_pure_mec():
    return {
        "uav_position": [0, UAV_DIM],
        "candidates": [
            UAV_DIM, UAV_DIM + NUM_NODES * CANDIDATE_DIM,
        ],
        "availability_mask": {
            "dim": MASK_DIM,
            "separate_from_mlp_input": True,
            "used_for_action_masking": True,
        },
        "total_mlp_input": STATE_DIM_PURE_MEC,
        "total_decision_state": DECISION_DIM_PURE_MEC,
        "candidate_fields": list(CANDIDATE_FIELDS),
        "forbidden": [
            "visual/PCA", "goal tokens", "goal visual similarity",
            "history visual", "heatmap", "GAB", "map retrieval",
            "navigation action", "episode step", "goal position",
            "goal distance", "oracle action", "success",
        ],
    }
