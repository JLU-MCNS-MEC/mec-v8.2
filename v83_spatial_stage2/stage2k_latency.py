"""Stage-2k E2E latency model for the Stage-2j T5 inputs.

The per-step uplink payload is the **actual byte size of the eight Stage-2j
model-input arrays** (float16/float32/uint8 as stored by the T5 examples).  No
MEC-side history cache is assumed: the full input set is counted on the uplink.

Remote inference is estimated from architecture FLOPs and the node's
0.5 / 1 / 2 TFLOPS capacity.  The frozen Stage-2j measured latencies are used
only as an ordering sanity check in the audit, not as the node model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import AUX, ENCODER, NETWORK
from latency_model import LatencyModel, visual_encoder_flops
from v83_spatial_stage2.stage2k_common import (
    DEADLINE_MS,
    TX_POWER_W,
    latency_penalty,
)


# Exact byte size of one Stage-2j input row.
STAGE2J_INPUT_BYTES = {
    "current_tokens": 576 * 384 * 2,
    "goal_tokens": 3 * 384 * 2,
    "history_vis": 16 * 768 * 2,
    "history_pose": 16 * 10 * 4,
    "history_action": 16 * 4 * 2,
    "history_mask": 16 * 1,
    "gab": 20 * 4,
    "map_feat": 22 * 4,
}
STAGE2J_PAYLOAD_BYTES = int(sum(STAGE2J_INPUT_BYTES.values()))


def _mha_flops(q_len, kv_len, dim):
    q, kv, d = int(q_len), int(kv_len), int(dim)
    return float(2.0 * (q + 2 * kv + q) * d * d + 4.0 * q * kv * d)


def _ff_flops(tokens, dim, hidden):
    n, d, h = int(tokens), int(dim), int(hidden)
    return float(2.0 * n * d * h + 2.0 * n * h * d)


def history_encoder_flops(steps=16, visual_dim=768, pose_dim=10,
                          action_dim=4, hidden=128, layers=2):
    slots = int(steps)
    projections = (
        2.0 * slots * visual_dim * hidden
        + 2.0 * slots * pose_dim * hidden
        + 2.0 * slots * action_dim * hidden
    )
    sequence = slots + 1
    per_layer = (
        _mha_flops(sequence, sequence, hidden)
        + _ff_flops(sequence, hidden, 2 * hidden)
    )
    return float(projections + layers * per_layer)


def arrival_head_flops():
    # 3x576 cosine similarities + 8->64->32->1 MLP.
    cosine = 2.0 * AUX.goal_token_count * ENCODER.num_tokens * ENCODER.token_dim
    mlp = (
        2.0 * 8 * 64
        + 2.0 * 64 * 32
        + 2.0 * 32 * 1
    )
    return float(cosine + mlp)


def stage2j_model_flops(model_name):
    """Analytic forward FLOPs for one Stage-2j T5 model (batch 1)."""
    common = history_encoder_flops() + arrival_head_flops()
    if model_name == "Small":
        # 2*384 pooled current/goal + 128 history + 20 GAB + 22 map.
        fusion = 2 * ENCODER.token_dim + 128 + 20 + 22
        return float(common + 2.0 * fusion * 256 + 2.0 * 256 * 3)
    if model_name == "Medium":
        hidden = 384
        cross = (
            _mha_flops(3, ENCODER.num_tokens + 1, hidden)
            + _ff_flops(3, hidden, hidden)
        )
        fusion = 3 * ENCODER.token_dim + 128 + 20 + 22
        return float(
            common + cross
            + 2.0 * fusion * hidden + 2.0 * hidden * 3
        )
    configs = {
        "Transformer-S": (192, 1, 3),
        "Transformer-M": (256, 2, 4),
        "Large": (352, 2, 4),
    }
    if model_name not in configs:
        raise ValueError(f"Unknown model {model_name}")
    hidden, layers, ff_mult = configs[model_name]
    visual = 2.0 * (ENCODER.num_tokens + 3) * ENCODER.token_dim * hidden
    cross = (
        _mha_flops(3, ENCODER.num_tokens + 1, hidden)
        + _ff_flops(3, hidden, ff_mult * hidden)
    )
    sequence = 1 + ENCODER.num_tokens + 3 + 3
    fusion = layers * (
        _mha_flops(sequence, sequence, hidden)
        + _ff_flops(sequence, hidden, ff_mult * hidden)
    )
    head = 2.0 * hidden * 3
    return float(common + visual + cross + fusion + head)


def model_complexity_table():
    return {
        name: {
            "flops": stage2j_model_flops(name),
            "gflops": stage2j_model_flops(name) / 1e9,
        }
        for name in (
            "Small", "Transformer-S", "Medium", "Transformer-M", "Large"
        )
    }


@dataclass(frozen=True)
class Stage2KLatency:
    payload_bytes: int = STAGE2J_PAYLOAD_BYTES
    tx_power_w: float = TX_POWER_W
    deadline_ms: float = DEADLINE_MS

    def __post_init__(self):
        object.__setattr__(self, "_latency_model", LatencyModel())
        object.__setattr__(
            self, "_extraction_ms",
            1000.0 * visual_encoder_flops()
            / float(NETWORK.uav_compute_flops_per_s),
        )
        object.__setattr__(
            self,
            "_flops",
            {
                name: stage2j_model_flops(name)
                for name in (
                    "Small", "Transformer-S", "Medium",
                    "Transformer-M", "Large",
                )
            },
        )

    @property
    def extraction_ms(self):
        return float(self._extraction_ms)

    def evaluate(self, node, uav_position, model_name=None):
        """Predicted E2E for one node/UAV state.

        ``node`` may be a dict (Stage-2k topology) or an object with
        ``distance_to`` and attributes.
        """
        model_name = model_name or node["model_type"]
        if isinstance(node, dict):
            distance = float(np.hypot(
                float(node["x"]) - float(uav_position[0]),
                float(node["y"]) - float(uav_position[1]),
            ))
            bandwidth = float(node["bandwidth_hz"])
            compute = float(node["compute_capacity_flops_per_s"])
        else:
            distance = float(node.distance_to(uav_position))
            bandwidth = float(node.bandwidth_hz)
            compute = float(node.compute_capacity_flops_per_s)
        channel = self._latency_model.a2g_channel(
            distance, tx_power_w=self.tx_power_w, bandwidth_hz=bandwidth
        )
        rate = max(float(channel["uplink_rate_bps"]), 1e-12)
        transmission_ms = 1000.0 * self.payload_bytes * 8.0 / rate
        inference_ms = (
            1000.0 * self._flops[model_name] / max(compute, 1e-12)
        )
        total_ms = self._extraction_ms + transmission_ms + inference_ms
        return {
            "model_type": model_name,
            "distance_m": distance,
            "bandwidth_hz": bandwidth,
            "compute_capacity_flops_per_s": compute,
            "uplink_rate_bps": rate,
            "uplink_rate_mbps": rate / 1e6,
            "snr_db": float(channel["snr_db"]),
            "payload_bytes": int(self.payload_bytes),
            "extraction_ms": float(self._extraction_ms),
            "communication_ms": float(transmission_ms),
            "inference_ms": float(inference_ms),
            "inference_flops": float(self._flops[model_name]),
            "total_ms": float(total_ms),
            "normalized_e2e": float(total_ms / self.deadline_ms),
            "deadline_violation": bool(total_ms > self.deadline_ms),
            "latency_penalty": float(latency_penalty(total_ms)),
        }

    def describe(self):
        return {
            "payload_bytes": int(self.payload_bytes),
            "payload_breakdown": dict(STAGE2J_INPUT_BYTES),
            "tx_power_w": float(self.tx_power_w),
            "deadline_ms": float(self.deadline_ms),
            "uav_compute_flops_per_s": float(
                NETWORK.uav_compute_flops_per_s
            ),
            "extraction_ms": float(self.extraction_ms),
            "model_flops": {
                name: float(flops) for name, flops in self._flops.items()
            },
            "model_gflops": {
                name: float(flops / 1e9)
                for name, flops in self._flops.items()
            },
            "payload_assumption": (
                "all eight Stage-2j input arrays are uploaded every step; "
                "no MEC-side history cache"
            ),
        }


def validate_payload_against_arrays(arrays):
    """Return actual bytes and the expected Stage-2j payload."""
    actual = int(sum(np.asarray(array).nbytes for array in arrays.values()))
    return {
        "actual_bytes": actual,
        "expected_bytes": STAGE2J_PAYLOAD_BYTES,
        "match": actual == STAGE2J_PAYLOAD_BYTES,
    }
