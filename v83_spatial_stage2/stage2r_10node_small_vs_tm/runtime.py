"""Model-specific homogeneous runtime for Stage-2r.

Only one frozen MEC checkpoint is loaded per deployment.  Navigation is
produced by that MEC model; DQN/PPO choose only which of the 10 identical
nodes serves the step.  The Stage-2n lane evaluator is reused so Dev-60
validation is exact and fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_runtime import load_encoder  # noqa: E402
from v83_spatial_stage2.stage2n_lanes import (  # noqa: E402
    LaneEvaluator,
    build_lane,
    build_lanes,
)
from v83_spatial_stage2.stage2n_runtime import (  # noqa: E402
    HomogeneousBackendPool,
    LatencyOnlyEpisodeRunner,
    make_stage2n_scene_cache,
)
from v83_spatial_stage2.stage2r_10node_small_vs_tm.common import (  # noqa: E402
    MEC_CHECKPOINTS,
    MEC_CHECKPOINT_SHA256,
)


def load_stage2r_runtime(device, model):
    device = torch.device(device)
    encoder = load_encoder(device)
    backends = HomogeneousBackendPool(
        device,
        checkpoint=MEC_CHECKPOINTS[model],
        expected_sha256=MEC_CHECKPOINT_SHA256[model],
        model_name=model,
    )
    return device, encoder, backends


def make_scene_cache(encoder, device, capacity=64):
    return make_stage2n_scene_cache(encoder, device, capacity=capacity)


def runner_for(bundle, spec, encoder, backends, scene_cache, device):
    return LatencyOnlyEpisodeRunner(
        spec=spec,
        topology=bundle["topology"],
        encoder=encoder,
        backends=backends,
        projector=None,
        quality_prior=bundle["quality_prior"],
        gab_config=bundle["gab_config"],
        scene_cache=scene_cache,
        device=device,
    )


def scheme_d_return(mean_e2e_ms):
    """Scheme-D episode return, exactly ``1 - mean_E2E/tau``."""
    return float(1.0 - float(mean_e2e_ms) / 100.0)


def add_scheme_d(rows):
    for row in rows:
        mean_e2e = float(row["mean_e2e_ms"])
        row["reward_D"] = scheme_d_return(mean_e2e)
        row["reward_D_closed_form"] = float(1.0 - mean_e2e / 100.0)
        row["scheme"] = "D"
    return rows


__all__ = [
    "load_stage2r_runtime",
    "make_scene_cache",
    "runner_for",
    "scheme_d_return",
    "add_scheme_d",
    "build_lane",
    "build_lanes",
    "LaneEvaluator",
]
