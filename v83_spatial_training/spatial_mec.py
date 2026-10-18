"""M0/M1/M2 spatial-aware MEC wrappers.

M0: original V8.2 model, no spatial fusion.
M1: original move_logits + learnable fusion of baseline24 and coherent24.
M2: M1 + temporal24 + innovation24, with an explicit temporal-validity mask.

The base model is always the original V8.2 architecture; only the movement
residual is added. The arrival head and all original inputs are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ARRIVAL_STOP_THRESHOLD
from gbs_model_v3 import (
    arrival_gated_decision,
    build_gbs_model,
    combine_move_and_arrival,
)


class SpatialFusionHead(nn.Module):
    """Small CNN producing a residual 3-way movement logit from spatial maps."""

    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 3),
        )
        # Start as an exact no-op so M1/M2 begin at the M0 movement logits.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, spatial_maps):
        return self.net(spatial_maps)


class SpatialAwareMEC(nn.Module):
    def __init__(self, base_model, mode):
        super().__init__()
        if mode not in ("M0", "M1", "M2"):
            raise ValueError(f"Unknown mode {mode}")
        self.mode = mode
        self.base = base_model
        if mode == "M1":
            self.fusion = SpatialFusionHead(2)
        elif mode == "M2":
            self.fusion = SpatialFusionHead(5)
        else:
            self.fusion = None

    def _spatial_input(self, spatial_maps, spatial_valid):
        if self.mode == "M1":
            if spatial_maps is None:
                raise ValueError("M1 requires spatial_maps")
            return spatial_maps[:, :2]
        if self.mode == "M2":
            if spatial_maps is None or spatial_valid is None:
                raise ValueError("M2 requires spatial_maps and spatial_valid")
            valid = spatial_valid.expand(
                -1, -1, spatial_maps.shape[-2], spatial_maps.shape[-1]
            )
            temporal = spatial_maps[:, 2:3] * valid
            innovation = spatial_maps[:, 3:4] * valid
            return torch.cat(
                [spatial_maps[:, :2], temporal, innovation, valid], dim=1
            )
        return None

    def forward_components(
        self,
        current_tokens,
        goal_tokens,
        heatmap,
        network_state,
        spatial_maps=None,
        spatial_valid=None,
        return_attention=False,
    ):
        comp = self.base.forward_components(
            current_tokens,
            goal_tokens,
            heatmap,
            network_state,
            return_attention=return_attention,
        )
        if self.mode != "M0" and spatial_maps is not None:
            comp["move_logits"] = comp["move_logits"] + self.fusion(
                self._spatial_input(spatial_maps, spatial_valid)
            )
        return comp

    def forward(
        self,
        current_tokens,
        goal_tokens,
        heatmap,
        network_state,
        spatial_maps=None,
        spatial_valid=None,
        return_attention=False,
    ):
        comp = self.forward_components(
            current_tokens,
            goal_tokens,
            heatmap,
            network_state,
            spatial_maps=spatial_maps,
            spatial_valid=spatial_valid,
            return_attention=return_attention,
        )
        logits = combine_move_and_arrival(
            comp["move_logits"], comp["arrival_logit"]
        )
        if return_attention:
            out = dict(comp.get("attention", {}))
            out["arrival_probability"] = torch.sigmoid(comp["arrival_logit"])
            out["arrival_features"] = comp["arrival_features"]
            return logits, out
        return logits

    def predict_action(
        self,
        current_tokens,
        goal_tokens,
        heatmap,
        network_state,
        spatial_maps=None,
        spatial_valid=None,
        stop_threshold=ARRIVAL_STOP_THRESHOLD,
        return_attention=False,
    ):
        comp = self.forward_components(
            current_tokens,
            goal_tokens,
            heatmap,
            network_state,
            spatial_maps=spatial_maps,
            spatial_valid=spatial_valid,
            return_attention=return_attention,
        )
        decision = arrival_gated_decision(
            comp["move_logits"],
            comp["arrival_logit"],
            stop_threshold=stop_threshold,
        )
        decision["move_logits"] = comp["move_logits"]
        decision["arrival_logit"] = comp["arrival_logit"]
        decision["arrival_features"] = comp["arrival_features"]
        decision["attention"] = comp.get("attention", {})
        return decision


def build_spatial_model(name, mode):
    return SpatialAwareMEC(build_gbs_model(name), mode)


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters()))
