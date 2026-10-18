"""V8.3 geometry-only prototype: no Goal/oracle coordinate information.

Separate BEV-like MEC topology maps from the camera's visual matching heatmap.
This does not modify the current V8.2 model or dataset. Use for diagnostics
until a trained fusion model establishes a measurable closed-loop benefit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GeometrySnapshot:
    heatmaps: np.ndarray  # [3,8,8] relative MEC occupancy/link/relative motion
    network_delta: np.ndarray  # [5] change in observable network state
    visible_nodes: int


def mec_relative_geometry(
    nodes,
    uav_xy,
    yaw_rad,
    network_state,
    *,
    previous_xy=None,
    previous_network_state=None,
    grid_size=8,
    radius_m=500.0,
):
    """Build dynamic 3-channel MEC maps in the UAV's local ground plane.

    Channels:
      0: normalized MEC occupancy within grid (Gaussian splats)
      1: normalized bandwidth-weighted occupancy (no MEC model type/capacity)
      2: relative approach/recede trend toward nodes since previous step
    Crucial: BEV axes are meters around the UAV, not camera image pixels.
    These maps should NEVER be added elementwise to the RGB image heatmap.
    The only UAV location used is for computing relative, egocentric features.
    """
    xy = np.asarray(uav_xy, dtype=np.float64).reshape(2)
    state = np.asarray(network_state, dtype=np.float32).reshape(5)
    if grid_size < 2 or radius_m <= 0:
        raise ValueError("Require grid_size>=2 and radius_m>0")
    prior = None if previous_xy is None else np.asarray(previous_xy, dtype=np.float64).reshape(2)
    if previous_network_state is None:
        delta = np.zeros(5, dtype=np.float32)
    else:
        delta = state - np.asarray(previous_network_state, dtype=np.float32).reshape(5)

    ax = np.linspace(-radius_m, radius_m, grid_size)
    xx, yy = np.meshgrid(ax, ax)
    sigma = 2.0 * radius_m / max(grid_size - 1, 1)
    maps = np.zeros((3, grid_size, grid_size), dtype=np.float64)
    count = 0
    cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
    for node in nodes:
        # Node fields used: x/y/bandwidth_hz only; no identity/capability prior.
        dx, dy = float(node.x)-xy[0], float(node.y)-xy[1]
        forward = cos_yaw*dx + sin_yaw*dy
        left = -sin_yaw*dx + cos_yaw*dy
        if abs(forward) > radius_m or abs(left) > radius_m:
            continue
        weight = np.exp(-0.5*((xx-left)**2 + (yy-forward)**2)/sigma**2)
        maps[0] += weight
        maps[1] += weight * np.clip(float(node.bandwidth_hz)/30e6, 0.0, 1.0)
        if prior is not None:
            former = math.hypot(float(node.x)-prior[0], float(node.y)-prior[1])
            current = math.hypot(dx, dy)
            trend = np.clip((former-current)/5.0, -1.0, 1.0)
            maps[2] += weight * trend
        count += 1
    if count:
        maps /= float(count)
    return GeometrySnapshot(
        heatmaps=maps.astype(np.float32),
        network_delta=delta.astype(np.float32),
        visible_nodes=count,
    )
