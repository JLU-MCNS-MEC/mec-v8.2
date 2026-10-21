"""Stage-2k v2 Pure MEC state and the visual ablation state.

Variant A (``pure_mec_v2``), DQN input 384-D:

    2-D normalized UAV position
  + 1-D normalized episode step  t / 400
  + 1-D running mean E2E delay    running_mean_E2E_ms / 100
  + 20 x 19-D candidate MEC features
  = 384

The 20-D availability mask is kept separately and used for action masking; it
is not duplicated in the MLP input.  The total decision state is 384 + 20.

Each candidate MEC has 19 features: geometry / compute / bandwidth / uplink /
communication / inference / E2E / SNR, four **independent Dev closed-loop
capability metrics** (SR, OSR, SPL, nDTW) and a five-way model-type one-hot.
There is no single composite quality prior.

Variant B (``visual_ablation_v2``), DQN input 512-D:

    128-D frozen VisualProjector context
  + the complete 384-D Pure MEC state
  = 512

Variant B adds exactly the existing frozen 128-D projector output.  No GAB,
heatmap or other new module is added.  The projector is the Stage-2k PCA
fitted on Train-600 natural states and frozen; its input summary's GAB/map
blocks are neutralized to the projector's training mean before projection, so
the 128-D context contains current/goal/history visual tokens only.
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
    MAX_STEPS,
    MODEL_NAMES,
    POSITION_SCALE_M,
    RATE_SCALE_BPS,
)
from v83_spatial_stage2.stage2k_v2_common import (
    CAPABILITY_FIELDS,
    CAPABILITY_METRICS,
    capability_values,
    flatten_capability,
)

NUM_NODES = 20
UAV_POSITION_DIM = 2
GLOBAL_DIM = 4  # uav_x, uav_y, t/400, running_mean_E2E_ms/100
CANDIDATE_DIM = 19
MASK_DIM = 20
STATE_DIM_PURE_MEC_V2 = GLOBAL_DIM + NUM_NODES * CANDIDATE_DIM  # 384
VISUAL_DIM = 128
STATE_DIM_VISUAL_ABLATION_V2 = VISUAL_DIM + STATE_DIM_PURE_MEC_V2  # 512
UAV_POSITION_SCALE_M = float(ENV.world_size_m)
SNR_SCALE_DB = 40.0
INFERENCE_SCALE_MS = 10.0
# The frozen PCA summary is 4*2*384 visual dims + GAB(20) + map(22).  The
# last 42 dims are non-visual; they are neutralized before projection.
VISUAL_SUMMARY_DIM = 4 * 2 * 384
AUX_SUMMARY_DIM = 20 + 22
PCA_SUMMARY_DIM = VISUAL_SUMMARY_DIM + AUX_SUMMARY_DIM

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
    "snr_db_norm",
    "dev_sr",
    "dev_osr",
    "dev_spl",
    "dev_ndtw",
    "model_Small",
    "model_Transformer-S",
    "model_Medium",
    "model_Transformer-M",
    "model_Large",
)

# Explicit block boundaries (0-based, end exclusive).
BLOCK_RELATIVE_X = 0
BLOCK_RELATIVE_Y = 1
BLOCK_DISTANCE = 2
BLOCK_COMPUTE = 3
BLOCK_BANDWIDTH = 4
BLOCK_UPLINK = 5
BLOCK_COMMUNICATION = 6
BLOCK_INFERENCE = 7
BLOCK_E2E = 8
BLOCK_SNR = 9
BLOCK_CAPABILITY = slice(10, 14)
BLOCK_MODEL_ONE_HOT = slice(14, 19)


def _flatten_capability(capability_prior):
    if capability_prior is None:
        raise ValueError("Pure MEC v2 requires a Dev capability prior")
    if not isinstance(capability_prior, dict):
        raise TypeError("capability_prior must be a dict or payload")
    if "models" in capability_prior:
        return flatten_capability(capability_prior)
    # Already-flat {model: {metric: value}} dict.
    return {
        name: {
            metric: float(capability_prior[name][metric])
            for metric in CAPABILITY_METRICS
        }
        for name in MODEL_NAMES
    }


def _one_hot(index, size):
    output = np.zeros(size, dtype=np.float32)
    if 0 <= int(index) < int(size):
        output[int(index)] = 1.0
    return output


def pure_mec_v2_candidate_features(
    nodes,
    uav_position,
    latency_model,
    capability_prior,
):
    """20 x 19-D MEC features; availability is a separate action mask."""
    if len(nodes) != NUM_NODES:
        raise ValueError(f"expected {NUM_NODES} nodes, got {len(nodes)}")
    capacity = _flatten_capability(capability_prior)
    uav_position = np.asarray(uav_position, dtype=np.float64)[:2]
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
        model_name = node["model_type"]
        capability = capability_values(capacity, model_name)
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
                node, uav_position, model_name
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
                float(np.clip(
                    float(metric["snr_db"]) / SNR_SCALE_DB, -1.0, 1.0
                )),
            ])
        else:
            values.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        values.extend(float(value) for value in capability)
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
        features[nearest, BLOCK_UPLINK] = (
            float(metric["uplink_rate_bps"]) / RATE_SCALE_BPS
        )
        features[nearest, BLOCK_COMMUNICATION] = (
            float(metric["communication_ms"]) / LATENCY_SCALE_MS
        )
        features[nearest, BLOCK_INFERENCE] = min(
            float(metric["inference_ms"]) / INFERENCE_SCALE_MS,
            E2E_CLIP_SCALE,
        )
        features[nearest, BLOCK_E2E] = min(
            float(metric["total_ms"]) / LATENCY_SCALE_MS, E2E_CLIP_SCALE
        )
        features[nearest, BLOCK_SNR] = float(np.clip(
            float(metric["snr_db"]) / SNR_SCALE_DB, -1.0, 1.0
        ))
    return features, mask, metrics, fallback


def build_pure_mec_v2_state(
    uav_position,
    step_index,
    running_mean_e2e_ms,
    candidate20x19,
):
    """Build the 384-D variant-A state."""
    uav = np.asarray(uav_position, dtype=np.float32).reshape(-1)[:2]
    candidate = np.asarray(candidate20x19, dtype=np.float32)
    if candidate.shape != (NUM_NODES, CANDIDATE_DIM):
        raise ValueError(f"candidate shape {candidate.shape}")
    step_index = int(step_index)
    if not 0 <= step_index <= MAX_STEPS:
        raise ValueError(
            f"step_index {step_index} outside [0,{MAX_STEPS}]"
        )
    running_mean = float(running_mean_e2e_ms)
    if not np.isfinite(running_mean) or running_mean < 0.0:
        raise ValueError(
            f"running_mean_e2e_ms {running_mean} is invalid"
        )
    global_features = np.asarray([
        uav[0] / np.float32(UAV_POSITION_SCALE_M),
        uav[1] / np.float32(UAV_POSITION_SCALE_M),
        step_index / float(MAX_STEPS),
        running_mean / 100.0,
    ], dtype=np.float32)
    state = np.concatenate([
        global_features, candidate.reshape(-1),
    ])
    if state.shape != (STATE_DIM_PURE_MEC_V2,):
        raise RuntimeError(f"pure-MEC v2 state dim {state.shape}")
    return state.astype(np.float32)


def build_visual_ablation_v2_state(visual128, pure_mec_state):
    """Build the 512-D variant-B state: 128-D visual + variant-A state."""
    visual = np.asarray(visual128, dtype=np.float32).reshape(-1)
    pure = np.asarray(pure_mec_state, dtype=np.float32).reshape(-1)
    if visual.shape != (VISUAL_DIM,):
        raise ValueError(f"visual dim {visual.shape}")
    if pure.shape != (STATE_DIM_PURE_MEC_V2,):
        raise ValueError(f"pure MEC v2 dim {pure.shape}")
    state = np.concatenate([visual, pure])
    if state.shape != (STATE_DIM_VISUAL_ABLATION_V2,):
        raise RuntimeError(f"visual ablation v2 state dim {state.shape}")
    return state.astype(np.float32)


def visual_context_from_projector(projector, summary):
    """Project a visual-only summary with the frozen Stage-2k projector.

    The summary's last 42 dims are GAB(20) + map(22).  They are replaced by
    the projector's training mean, so their normalized contribution is zero.
    The PCA mean/components themselves are never modified.  This keeps the
    ablation free of GAB/map information while using the existing fixed
    ``VisualProjector``.
    """
    summary = np.asarray(summary, dtype=np.float32).reshape(-1)
    if summary.shape != (PCA_SUMMARY_DIM,):
        raise ValueError(
            f"visual summary dim {summary.shape} != {PCA_SUMMARY_DIM}"
        )
    if projector.mean.shape[0] != PCA_SUMMARY_DIM:
        raise ValueError(
            f"projector summary dim {projector.mean.shape[0]} != "
            f"{PCA_SUMMARY_DIM}"
        )
    neutral = summary.copy()
    neutral[VISUAL_SUMMARY_DIM:] = projector.mean[VISUAL_SUMMARY_DIM:]
    visual = projector.project(neutral)
    if visual.shape != (VISUAL_DIM,):
        raise RuntimeError(f"visual context dim {visual.shape}")
    return visual.astype(np.float32)


def state_layout_pure_mec_v2():
    return {
        "state_mode": "pure_mec_v2",
        "uav_position": [0, UAV_POSITION_DIM],
        "episode_step": {
            "index": 2, "normalization": f"t/{MAX_STEPS}",
        },
        "running_mean_e2e": {
            "index": 3, "normalization": "running_mean_E2E_ms/100",
        },
        "candidates": [GLOBAL_DIM, STATE_DIM_PURE_MEC_V2],
        "availability_mask": {
            "dim": MASK_DIM,
            "separate_from_mlp_input": True,
            "used_for_action_masking": True,
        },
        "total_mlp_input": STATE_DIM_PURE_MEC_V2,
        "total_decision_state": STATE_DIM_PURE_MEC_V2 + MASK_DIM,
        "candidate_dim": CANDIDATE_DIM,
        "candidate_fields": list(CANDIDATE_FIELDS),
        "capability_fields": list(CAPABILITY_FIELDS),
        "forbidden": [
            "visual/PCA", "goal tokens", "goal visual similarity",
            "history visual", "heatmap", "GAB", "map retrieval",
            "navigation action", "target position", "target distance",
            "oracle action", "success",
        ],
    }


def state_layout_visual_ablation_v2():
    layout = state_layout_pure_mec_v2()
    layout.update({
        "state_mode": "visual_ablation_v2",
        "visual_context": [0, VISUAL_DIM],
        "pure_mec_state": [
            VISUAL_DIM, VISUAL_DIM + STATE_DIM_PURE_MEC_V2,
        ],
        "total_mlp_input": STATE_DIM_VISUAL_ABLATION_V2,
        "total_decision_state": STATE_DIM_VISUAL_ABLATION_V2 + MASK_DIM,
        "visual_projector": (
            "frozen Stage-2k PCA (Train-600 natural states); GAB/map blocks "
            "of the summary neutralized to the training mean"
        ),
        "forbidden": [
            "new GAB module", "heatmap module", "other new modules",
            "goal tokens as explicit state", "target position",
            "target distance", "oracle action", "success",
        ],
    })
    return layout
