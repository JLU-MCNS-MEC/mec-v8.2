from dataclasses import dataclass

import numpy as np

from config import ENV, GBS_MODEL_CONFIGS, NETWORK
from latency_model import LatencyModel


@dataclass(frozen=True)
class GBSNode:
    gbs_id: str
    x: float
    y: float
    model_type: str
    coverage_radius_m: float
    compute_capacity_flops_per_s: float
    hardware_tier: str
    bandwidth_hz: float = NETWORK.bandwidth_hz

    @property
    def position(self):
        return np.asarray([self.x, self.y], dtype=np.float32)

    def distance_to(self, uav_position):
        p = np.asarray(uav_position, dtype=np.float32)
        return float(np.linalg.norm(self.position - p))

    def is_available(self, uav_position):
        return self.distance_to(uav_position) <= self.coverage_radius_m


def _positions_have_min_spacing(positions, min_spacing):
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_spacing:
                return False
    return True


def _check_layout_coverage(
    positions,
    world_size_m,
    coverage_radius_m,
    required_coverage=2,
    grid_size=9,
):
    coordinates = np.linspace(75.0, world_size_m - 75.0, grid_size)
    minimum = None
    for x in coordinates:
        for y in coordinates:
            point = np.asarray([x, y], dtype=np.float32)
            distances = np.linalg.norm(positions - point, axis=1)
            count = int(np.sum(distances <= coverage_radius_m))
            minimum = count if minimum is None else min(minimum, count)
            if count < required_coverage:
                return False, minimum
    return True, minimum


def _balanced_labels(values, n, rng):
    labels = [values[i % len(values)] for i in range(n)]
    rng.shuffle(labels)
    return labels


def build_random_gbs_layout(
    world_size_m=ENV.world_size_m,
    num_gbs=NETWORK.num_gbs,
    coverage_radius_m=NETWORK.coverage_radius_m,
    seed=None,
    min_spacing_m=NETWORK.min_spacing_m,
    required_coverage=NETWORK.required_grid_coverage,
    max_layout_attempts=NETWORK.max_layout_attempts,
    verbose=True,
):
    """
    Build one random 20-MEC deployment.

    The model_type and compute capacity fields are retained for later association
    experiments, but global_network_state() explicitly ignores them.
    """
    rng = np.random.default_rng(seed)
    margin = 50.0

    model_names = list(GBS_MODEL_CONFIGS.keys())
    bandwidth_values = list(NETWORK.gbs_bandwidth_classes_hz)
    hardware_names = ["Edge-A", "Edge-B", "Edge-C"]

    for attempt in range(max_layout_attempts):
        positions = rng.uniform(
            margin,
            world_size_m - margin,
            size=(num_gbs, 2),
        ).astype(np.float32)

        if not _positions_have_min_spacing(positions, min_spacing_m):
            continue

        coverage_ok, minimum = _check_layout_coverage(
            positions,
            world_size_m,
            coverage_radius_m,
            required_coverage=required_coverage,
            grid_size=9,
        )
        if not coverage_ok:
            continue

        model_types = _balanced_labels(model_names, num_gbs, rng)
        compute_classes = list(NETWORK.mec_compute_classes_flops_per_s)
        compute_indices = _balanced_labels(
            list(range(len(compute_classes))), num_gbs, rng
        )
        bandwidths = _balanced_labels(bandwidth_values, num_gbs, rng)

        nodes = []
        for i in range(num_gbs):
            # Capability assignment is hidden from the MEC input.
            cap_idx = int(compute_indices[i] % len(compute_classes))
            nodes.append(
                GBSNode(
                    gbs_id=f"G{i + 1}",
                    x=float(positions[i, 0]),
                    y=float(positions[i, 1]),
                    model_type=model_types[i],
                    coverage_radius_m=float(coverage_radius_m),
                    compute_capacity_flops_per_s=float(compute_classes[cap_idx]),
                    hardware_tier=hardware_names[cap_idx],
                    bandwidth_hz=float(bandwidths[i]),
                )
            )

        if verbose:
            print(
                f"Random GBS layout generated after {attempt + 1} attempt(s); "
                f"minimum coarse-grid coverage = {minimum}."
            )
        return nodes

    raise RuntimeError(
        "Unable to generate a dense GBS topology. Increase coverage_radius_m, "
        "num_gbs, or max_layout_attempts."
    )


def get_available_gbs(nodes, uav_position):
    return [node for node in nodes if node.is_available(uav_position)]


def get_nearest_gbs(nodes, uav_position):
    return min(nodes, key=lambda node: node.distance_to(uav_position))


def global_network_state(
    nodes,
    uav_position,
    num_tokens=576,
    token_dim=384,
):
    """
    Return the global 5-D observable network state used by V8.2 MEC.

    [0] available-MEC ratio
    [1] nearest relative UAV-MEC distance
    [2] best-link SNR
    [3] best-link physical bandwidth
    [4] best-link visual-token transmission latency

    IMPORTANT:
      - no model_type
      - no compute capacity
      - no remote inference latency
      - no model-quality prior
      - no absolute UAV coordinates

    The underlying network simulator may use UAV position to derive link metrics,
    but only the normalized network observables below are passed to the MEC.
    """
    if not nodes:
        raise ValueError("nodes must not be empty")

    p = np.asarray(uav_position, dtype=np.float32)
    distances = np.asarray([node.distance_to(p) for node in nodes], dtype=np.float64)
    coverage = float(nodes[0].coverage_radius_m)

    available = distances <= coverage
    availability_ratio = float(np.mean(available))

    nearest_distance = float(np.min(distances))
    nearest_relative = np.clip(
        nearest_distance / max(coverage, 1e-9),
        0.0,
        float(NETWORK.relative_distance_clip),
    ) / float(NETWORK.relative_distance_clip)

    latency_model = LatencyModel()
    payload_bits = (
        int(num_tokens)
        * int(token_dim)
        * int(NETWORK.visual_token_bits_per_element)
    )

    link_records = []
    for node, distance in zip(nodes, distances):
        channel = latency_model.a2g_channel(
            float(distance),
            tx_power_w=NETWORK.default_tx_power_w,
            bandwidth_hz=float(node.bandwidth_hz),
        )
        rate = float(channel["uplink_rate_bps"])
        tcom_ms = 1000.0 * payload_bits / max(rate, 1e-12)
        link_records.append(
            {
                "node": node,
                "snr_db": float(channel["snr_db"]),
                "rate_bps": rate,
                "bandwidth_hz": float(node.bandwidth_hz),
                "tcom_ms": float(tcom_ms),
            }
        )

    # Best link is the one with the largest instantaneous uplink rate.
    best = max(link_records, key=lambda x: x["rate_bps"])

    snr_norm = (
        np.clip(
            best["snr_db"],
            float(NETWORK.snr_min_db),
            float(NETWORK.snr_max_db),
        )
        - float(NETWORK.snr_min_db)
    ) / (float(NETWORK.snr_max_db) - float(NETWORK.snr_min_db))

    bandwidth_norm = np.clip(
        best["bandwidth_hz"] / max(float(NETWORK.max_bandwidth_hz), 1e-9),
        0.0,
        1.0,
    )

    tcom_norm = np.clip(
        best["tcom_ms"] / max(float(NETWORK.tcom_clip_ms), 1e-9),
        0.0,
        1.0,
    )

    return np.asarray(
        [
            availability_ratio,
            nearest_relative,
            float(snr_norm),
            float(bandwidth_norm),
            float(tcom_norm),
        ],
        dtype=np.float32,
    )


def network_state_names():
    return [
        "availability_ratio",
        "nearest_relative_distance",
        "best_link_snr",
        "best_link_bandwidth",
        "best_link_visual_tcom",
    ]
