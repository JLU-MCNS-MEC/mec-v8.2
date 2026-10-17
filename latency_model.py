"""V8异构MEC解析时延模型：RGB visual tokens + 小型Depth patch features + Arrival head。"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from config import AUX, ENCODER, ENV, GBS_MODEL_CONFIGS, NETWORK, NUM_MOVE_ACTIONS

SPEED_OF_LIGHT_MPS = 299_792_458.0
EPS = 1e-15


def dbm_to_watt(dbm):
    return 10.0 ** ((float(dbm) - 30.0) / 10.0)


def visual_encoder_flops(
    output_tokens=ENCODER.num_tokens,
    token_dim=ENCODER.token_dim,
    patch_size=ENCODER.patch_size,
    image_channels=3,
    num_layers=ENCODER.num_layers,
):
    n = int(output_tokens)
    d = int(token_dim)
    patch_flops = 2.0 * n * d * image_channels * (int(patch_size) ** 2)
    transformer_flops = int(num_layers) * (
        24.0 * n * (d ** 2) + 4.0 * (n ** 2) * d
    )
    return float(patch_flops + transformer_flops)


def start_encoder_flops():
    d = int(AUX.aux_feature_dim)
    return float(
        2.0 * 16 * 16 * 1 * 16 * 3 * 3
        + 2.0 * 8 * 8 * 16 * 32 * 3 * 3
        + 2.0 * 4 * 4 * 32 * 64 * 3 * 3
        + 2.0 * 64 * d
    )


def history_small_flops(token_dim):
    """Small的spatial attention pooling + K帧GRU近似。"""
    k = int(AUX.history_length)
    hs = int(AUX.history_tokens_per_frame)
    d = int(AUX.aux_feature_dim)
    dz = int(token_dim)
    token_proj = k * hs * 2.0 * dz * d
    score = k * hs * (2.0 * d * d + 2.0 * d)
    gru = k * 6.0 * d * d
    return float(token_proj + score + gru)


def mha_flops(q_len, kv_len, dim):
    """Q/K/V/out projection + attention matmul的粗略FLOPs。"""
    q = int(q_len)
    kv = int(kv_len)
    d = int(dim)
    proj = 2.0 * (q + 2 * kv + q) * d * d
    attn = 4.0 * q * kv * d
    return float(proj + attn)


def gbs_backend_flops(model_type, num_tokens, token_dim, num_actions):
    n = int(num_tokens)
    dz = int(token_dim)
    a = int(num_actions)
    cfg = GBS_MODEL_CONFIGS[model_type]
    architecture = cfg["architecture"]
    h = int(cfg["hidden_dim"])
    s_goal = int(AUX.goal_token_count)
    k = int(AUX.history_length)
    hs = int(AUX.history_tokens_per_frame)
    ad = int(AUX.aux_feature_dim)
    # ArrivalReasoningHead: 20->64->32->1，另加Goal/Current cosine与history matching粗略开销。
    arrival_feature = 4.0 * n * dz + 4.0 * k * hs * dz
    arrival_mlp = 2.0 * 20 * int(AUX.arrival_hidden_dim) + 2.0 * int(AUX.arrival_hidden_dim) * (int(AUX.arrival_hidden_dim)//2) + 2.0 * (int(AUX.arrival_hidden_dim)//2)
    arrival_flops = arrival_feature + arrival_mlp

    if architecture == "late_fusion_mlp":
        fusion = 2 * dz + 2 * ad
        return float(
            start_encoder_flops()
            + history_small_flops(dz)
            + (n + s_goal) * dz
            + 2.0 * fusion * h
            + 2.0 * h * NUM_MOVE_ACTIONS
            + arrival_flops
        )

    if architecture == "goal_cross_attention":
        cross1 = mha_flops(s_goal, n + 1, dz)
        cross2 = mha_flops(s_goal, k * hs + 1, dz)
        ffn = 2.0 * s_goal * 4.0 * dz * h
        fusion = 4 * dz + ad
        cls = 2.0 * fusion * h + 2.0 * h * NUM_MOVE_ACTIONS
        return float(start_encoder_flops() + cross1 + cross2 + ffn + cls + arrival_flops)

    if architecture == "explicit_transformer":
        ff_mult = int(cfg.get("ff_mult", 4))
        layers = int(cfg["num_layers"])
        # Current + Goal + 10x4 History + 2x2 direct-current均需要视觉投影。
        visual_proj = 2.0 * (n + s_goal + k * hs + hs) * dz * h
        start_proj = 2.0 * ad * h
        cross1 = mha_flops(s_goal, n + 1, h)
        cross2 = mha_flops(s_goal, k * hs + 1, h)
        cross_ffn = 2.0 * s_goal * (4.0 * ff_mult * h * h)
        # Fusion: CLS + direct(4) + matched(3) + temporal(3) + start(1) = 12。
        f = 1 + hs + s_goal + s_goal + 1
        fusion_transformer = layers * (24.0 * f * h * h + 4.0 * f * f * h)
        head = 2.0 * h * h + 2.0 * h * NUM_MOVE_ACTIONS
        return float(
            start_encoder_flops() + visual_proj + start_proj + cross1 + cross2
            + cross_ffn + fusion_transformer + head + arrival_flops
        )

    raise ValueError(f"Unknown model_type: {model_type}")


@dataclass(frozen=True)
class LatencyConfig:
    uav_altitude_m: float = NETWORK.uav_altitude_m
    gbs_height_m: float = NETWORK.gbs_height_m
    carrier_frequency_hz: float = NETWORK.carrier_frequency_hz
    bandwidth_hz: float = NETWORK.bandwidth_hz
    los_a: float = NETWORK.los_a
    los_b: float = NETWORK.los_b
    excessive_loss_los_db: float = NETWORK.excessive_loss_los_db
    excessive_loss_nlos_db: float = NETWORK.excessive_loss_nlos_db
    noise_psd_dbm_per_hz: float = NETWORK.noise_psd_dbm_per_hz
    noise_figure_db: float = NETWORK.noise_figure_db
    default_tx_power_w: float = NETWORK.default_tx_power_w
    min_tx_power_w: float = NETWORK.min_tx_power_w
    max_tx_power_w: float = NETWORK.max_tx_power_w
    visual_token_bits_per_element: int = NETWORK.visual_token_bits_per_element
    depth_token_bits_per_element: int = NETWORK.depth_token_bits_per_element
    encoder_compute_capacity_flops_per_s: float = NETWORK.uav_compute_flops_per_s
    decision_latency_budget_s: float = NETWORK.decision_latency_budget_s
    episode_horizon_for_reward: int = ENV.max_steps


class LatencyModel:
    def __init__(self, config: Optional[LatencyConfig] = None):
        self.config = config if config is not None else LatencyConfig()

    def extraction_latency(self, num_tokens, token_dim):
        flops = visual_encoder_flops(num_tokens, token_dim)
        latency_s = flops / self.config.encoder_compute_capacity_flops_per_s
        return {
            "extraction_flops": flops,
            "extraction_latency_s": latency_s,
            "extraction_latency_ms": latency_s * 1e3,
        }

    def uplink_payload_bits(self, num_tokens, token_dim):
        # History由已上传Current token在MEC侧缓存，不重复上行。
        visual_bits = int(num_tokens) * int(token_dim) * self.config.visual_token_bits_per_element
        depth_bits = int(num_tokens) * int(AUX.depth_token_dim) * self.config.depth_token_bits_per_element
        total_bits = visual_bits + depth_bits
        return {
            "payload_bits": float(total_bits),
            "payload_bytes": float(total_bits / 8.0),
            "payload_kib": float(total_bits / 8.0 / 1024.0),
            "visual_token_payload_bits": float(visual_bits),
            "depth_token_payload_bits": float(depth_bits),
            "depth_token_payload_kib": float(depth_bits / 8.0 / 1024.0),
            "history_update_bits": 0.0,
        }

    def goal_reference_cache_bits(self):
        # 三个多尺度Goal token，每个384维FP16，只在episode开始缓存一次。
        return float(AUX.goal_token_count * ENCODER.token_dim * 16)

    def a2g_channel(self, horizontal_distance_m, tx_power_w=None, bandwidth_hz=None):
        cfg = self.config
        horizontal_distance_m = max(float(horizontal_distance_m), 0.0)
        if bandwidth_hz is None:
            bandwidth_hz = cfg.bandwidth_hz
        bandwidth_hz = max(float(bandwidth_hz), 1.0)
        vertical_distance_m = abs(cfg.uav_altitude_m - cfg.gbs_height_m)
        distance_3d_m = max(math.sqrt(horizontal_distance_m ** 2 + vertical_distance_m ** 2), 1.0)
        elevation_deg = math.degrees(math.atan2(vertical_distance_m, max(horizontal_distance_m, EPS)))
        p_los = 1.0 / (1.0 + cfg.los_a * math.exp(-cfg.los_b * (elevation_deg - cfg.los_a)))
        free_space_loss_db = 20.0 * math.log10(
            4.0 * math.pi * cfg.carrier_frequency_hz * distance_3d_m / SPEED_OF_LIGHT_MPS
        )
        path_loss_db = free_space_loss_db + p_los * cfg.excessive_loss_los_db + (1.0 - p_los) * cfg.excessive_loss_nlos_db
        channel_gain = 10.0 ** (-path_loss_db / 10.0)
        noise_power_dbm = cfg.noise_psd_dbm_per_hz + 10.0 * math.log10(bandwidth_hz) + cfg.noise_figure_db
        noise_power_w = dbm_to_watt(noise_power_dbm)
        power_w = cfg.default_tx_power_w if tx_power_w is None else float(tx_power_w)
        power_w = float(np.clip(power_w, cfg.min_tx_power_w, cfg.max_tx_power_w))
        snr = power_w * channel_gain / max(noise_power_w, EPS)
        rate_bps = bandwidth_hz * math.log2(1.0 + snr)
        return {
            "horizontal_distance_m": horizontal_distance_m,
            "vertical_distance_m": vertical_distance_m,
            "distance_3d_m": distance_3d_m,
            "elevation_deg": elevation_deg,
            "p_los": p_los,
            "path_loss_db": path_loss_db,
            "snr_linear": snr,
            "snr_db": 10.0 * math.log10(max(snr, EPS)),
            "uplink_rate_bps": max(rate_bps, EPS),
            "uplink_rate_mbps": max(rate_bps, EPS) / 1e6,
            "tx_power_w": power_w,
        }

    def remote_inference_latency(self, model_type, num_tokens, token_dim, num_actions, compute_capacity_flops_per_s):
        flops = gbs_backend_flops(model_type, num_tokens, token_dim, num_actions)
        latency_s = flops / float(compute_capacity_flops_per_s)
        return {
            "inference_flops": flops,
            "backend_compute_flops_per_s": float(compute_capacity_flops_per_s),
            "inference_latency_s": latency_s,
            "inference_latency_ms": latency_s * 1e3,
        }

    def evaluate_step_from_distance(self, tokens, model_type, horizontal_distance_m, compute_capacity_flops_per_s, tx_power_w=None, num_actions=4):
        if len(tokens.shape) != 3:
            raise ValueError(f"Expected [B,N,D] tokens, got {tuple(tokens.shape)}")
        n, d = int(tokens.shape[1]), int(tokens.shape[2])
        extraction = self.extraction_latency(n, d)
        payload = self.uplink_payload_bits(n, d)
        channel = self.a2g_channel(horizontal_distance_m, tx_power_w)
        transmission_s = payload["payload_bits"] / channel["uplink_rate_bps"]
        inference = self.remote_inference_latency(
            model_type, n, d, num_actions, compute_capacity_flops_per_s
        )
        total_s = extraction["extraction_latency_s"] + transmission_s + inference["inference_latency_s"]
        normalized_delay = total_s / self.config.decision_latency_budget_s
        excess_ratio = max(0.0, normalized_delay - 1.0)
        latency_reward = -excess_ratio / self.config.episode_horizon_for_reward
        out = {}
        out.update(extraction)
        out.update(payload)
        out.update(channel)
        out.update(inference)
        out.update({
            "transmission_latency_s": transmission_s,
            "transmission_latency_ms": transmission_s * 1e3,
            "total_latency_s": total_s,
            "total_latency_ms": total_s * 1e3,
            "normalized_step_delay": normalized_delay,
            "excess_delay_ratio": excess_ratio,
            "latency_reward": latency_reward,
            "deadline_violation": bool(total_s > self.config.decision_latency_budget_s),
        })
        return out

    def describe(self):
        cfg = self.config
        return {
            "world_size_m": ENV.world_size_m,
            "uav_altitude_m": cfg.uav_altitude_m,
            "gbs_height_m": cfg.gbs_height_m,
            "carrier_frequency_ghz": cfg.carrier_frequency_hz / 1e9,
            "bandwidth_mhz": cfg.bandwidth_hz / 1e6,
            "default_tx_power_w": cfg.default_tx_power_w,
            "visual_token_bits_per_element": cfg.visual_token_bits_per_element,
            "depth_token_dim": AUX.depth_token_dim,
            "depth_max_m": AUX.depth_max_m,
            "depth_payload_kib_per_step": (ENCODER.num_tokens * AUX.depth_token_dim * cfg.depth_token_bits_per_element) / 8.0 / 1024.0,
            "goal_reference_cache_kib": self.goal_reference_cache_bits() / 8.0 / 1024.0,
            "encoder_compute_gflops_per_s": cfg.encoder_compute_capacity_flops_per_s / 1e9,
            "decision_latency_budget_ms": cfg.decision_latency_budget_s * 1e3,
        }
