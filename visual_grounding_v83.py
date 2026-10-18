"""V8.3 observable Goal-Current grounding and yaw-aligned temporal belief.

New information comes from *previous* RGB matching, not from reformatting the
same Current/Goal similarities. Temporal updates are intentionally trusted only
after known in-place LEFT/RIGHT turns. FORWARD translation has depth-dependent
parallax and MUST NOT be warped with a fixed pixel shift.

No simulator goal position, true goal distance, oracle/target mask, network
capability or success label enters this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from config import ENCODER, ENV
from gbs_model_v3 import goal_current_similarity


@dataclass
class GroundingMaps:
    baseline: torch.Tensor      # [B,24,24], best single-scale cosine
    coherent: torch.Tensor      # [B,24,24], multiscale/local consensus
    temporal: torch.Tensor      # [B,24,24], optional yaw-registered evidence
    innovation: torch.Tensor    # [B,24,24], observable current-minus-prior
    temporal_valid: torch.Tensor  # [B], valid for an in-place turn only

    def as_channels(self) -> torch.Tensor:
        """[B,5,24,24], separate from existing [B,3,8,8] RGB heatmap."""
        valid = self.temporal_valid[:, None, None].expand_as(self.baseline)
        return torch.stack(
            [self.baseline, self.coherent, self.temporal,
             self.innovation, valid.to(self.baseline.dtype)], dim=1
        )


def static_grounding(current_tokens: torch.Tensor, goal_tokens: torch.Tensor):
    """Two competing, parameter-free spatial ranking maps.

    Baseline reproduces the old per-patch 3-scale MAX aggregation.
    Coherent ranks spatial patches by agreement across all 3 goal scales,
    robustly standardizes each scale, and encourages contiguous local
    regions. Neither method gets any object or position annotations.
    The heuristic may underperform; test with val Ground Truth offline.
    """
    grid = int(ENCODER.grid_size)
    sim = goal_current_similarity(current_tokens, goal_tokens)
    if sim.ndim != 3 or sim.shape[1] != 3 or sim.shape[2] != grid * grid:
        raise ValueError("Expected goal similarities [B,3,24*24]")
    sim = sim.reshape(sim.shape[0], 3, grid, grid).float()
    baseline = sim.amax(dim=1)
    mean = sim.mean(dim=(-1, -2), keepdim=True)
    # Stabilize low-variance maps. Relative scale, not absolute brightness.
    deviation = sim.std(dim=(-1, -2), keepdim=True, unbiased=False).clamp_min(0.05)
    standardized = ((sim - mean) / deviation).clamp(-5.0, 5.0)
    agreement = standardized.mean(dim=1) - 0.35 * standardized.std(dim=1, unbiased=False)
    smoothed = F.avg_pool2d(agreement.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    coherent = 0.55 * agreement + 0.45 * smoothed
    return baseline, coherent


def yaw_warp_previous(
    previous_map: torch.Tensor,
    previous_action: torch.Tensor,
    *,
    turn_step_deg: float = ENV.turn_step_deg,
    image_size: int = ENCODER.image_size,
    hfov_deg: float = ENV.camera_hfov_deg,
):
    """Align previous camera map into CURRENT orientation for in-place turns.

    LEFT=1 means camera yaw increases by +turn_step_deg, RIGHT=2 is negative.
    A rectilinear pinhole yaw reprojection is used; no goal/pose coordinates.
    Action 0 (FORWARD) is deliberately invalid: parallax cannot be corrected
    from a yaw shift. Returns zero-filled prior and per-sample validity.
    """
    if previous_map.ndim != 3 or previous_map.shape[1:] != (
        ENCODER.grid_size, ENCODER.grid_size
    ):
        raise ValueError("Expected previous_map [B,24,24]")
    if previous_action.ndim != 1 or previous_action.shape[0] != previous_map.shape[0]:
        raise ValueError("Expected previous_action [B]")
    if not (0 < hfov_deg < 180):
        raise ValueError("Invalid field of view")
    valid = (previous_action == 1) | (previous_action == 2)
    yaw = torch.where(
        previous_action == 1, torch.ones_like(previous_action, dtype=torch.float32),
        -torch.ones_like(previous_action, dtype=torch.float32)
    ).to(previous_map.device) * math.radians(turn_step_deg)
    size = ENCODER.grid_size
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32, device=previous_map.device),
        torch.arange(size, dtype=torch.float32, device=previous_map.device),
        indexing="ij",
    )
    px_current = (x + 0.5) * (image_size / size)
    fx = (image_size * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)
    theta_cur = torch.atan((image_size * 0.5 - px_current) / fx)
    old_px = image_size * 0.5 - fx * torch.tan(
        theta_cur.unsqueeze(0) + yaw[:, None, None]
    )
    # align_corners=False; convert pixel coordinate to NDC directly.
    grid_x = 2 * old_px / image_size - 1
    grid_y = (2 * (y + 0.5) / size - 1)[None].expand_as(grid_x)
    sampling_grid = torch.stack([grid_x, grid_y], dim=-1)
    warped = F.grid_sample(
        previous_map.float().unsqueeze(1), sampling_grid,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    ).squeeze(1)
    warped = torch.where(valid[:, None, None], warped, torch.zeros_like(warped))
    return warped, valid


def visual_geometry(
    current_tokens: torch.Tensor,
    goal_tokens: torch.Tensor,
    *,
    previous_coherent: torch.Tensor | None = None,
    previous_action: torch.Tensor | None = None,
    carry: float = 0.30,
) -> GroundingMaps:
    """Produce *observable* static and yaw-aligned temporal spatial evidence."""
    if not 0 <= carry <= 1:
        raise ValueError("carry must lie in [0,1]")
    baseline, coherent = static_grounding(current_tokens, goal_tokens)
    valid = torch.zeros(baseline.shape[0], dtype=torch.bool, device=baseline.device)
    warped = torch.zeros_like(coherent)
    if previous_coherent is not None or previous_action is not None:
        if previous_coherent is None or previous_action is None:
            raise ValueError("Both previous_coherent and previous_action required")
        warped, valid = yaw_warp_previous(
            previous_coherent.to(coherent.device), previous_action.to(coherent.device)
        )
    temporal = (1-carry) * coherent + carry * warped
    temporal = torch.where(valid[:, None, None], temporal, coherent)
    innovation = torch.where(
        valid[:, None, None], coherent - warped, torch.zeros_like(coherent)
    )
    return GroundingMaps(
        baseline=baseline, coherent=coherent, temporal=temporal,
        innovation=innovation, temporal_valid=valid,
    )


class VisualBeliefTracker:
    """Episode-local visual memory. reset() MUST be called at episode start.

    track(current,goal,previous_action) uses only the previous visual belief
    and action; no ground-truth label. The stored belief is detached to prevent
    unbounded history graphs; online evaluation only in this first phase.
    """
    def __init__(self, carry: float = 0.30):
        self.carry = float(carry)
        self.previous_coherent = None

    def reset(self):
        self.previous_coherent = None

    @torch.no_grad()
    def track(self, current_tokens, goal_tokens, previous_action=None):
        batch = current_tokens.shape[0]
        actions = None
        if previous_action is not None:
            actions = torch.as_tensor(
                previous_action, dtype=torch.long, device=current_tokens.device
            ).reshape(-1)
            if len(actions) != batch:
                raise ValueError("Action count must match batch size")
        if self.previous_coherent is None:
            result = visual_geometry(current_tokens, goal_tokens, carry=self.carry)
        else:
            if actions is None:
                raise ValueError("Previous action required after first state")
            result = visual_geometry(
                current_tokens, goal_tokens,
                previous_coherent=self.previous_coherent,
                previous_action=actions, carry=self.carry
            )
        self.previous_coherent = result.coherent.detach()
        return result
