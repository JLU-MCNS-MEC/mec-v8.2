"""Optional goal-conditioned Depth evidence for V8.3 Arrival ablation.

This module is NOT enabled in existing MEC models or datasets. It extracts
eight scalar features from a regular RGB-D sensor map using goal-current
visual similarity only. No goal coordinates, oracle masks, true distance to
goal, UAV position, model capability, or success label are used.

Depth values are *camera slant range* to visible landmarks, NOT horizontal
distance to the target or the simulation's <20m stop threshold. Zero/NaN
means invalid: always use the depth-availability features as masks.
All output features are normalized to [0,1], including cosine-similarity
confidence fields. The last two fields are RGB-only signals: a fair ablation
must give the same RGB confidence to both the RGB and RGB-D probes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from config import AUX, ENCODER
from gbs_model_v3 import goal_current_similarity


DEPTH_FEATURE_NAMES = (
    "matched_depth_mean_normalized",
    "matched_depth_std_normalized",
    "matched_depth_min_normalized",
    "matched_depth_valid_fraction",
    "matched_depth_inverse_proximity",
    "visual_similarity_peak",
    "visual_similarity_margin",
    "scene_valid_depth_fraction",
)


@torch.no_grad()
def goal_conditioned_depth_evidence(
    current_tokens: torch.Tensor,
    goal_tokens: torch.Tensor,
    depth_map_m: torch.Tensor | np.ndarray,
    *,
    max_depth_m: float = float(AUX.depth_max_m),
    topk: int = 12,
    match_temperature: float = 0.10,
) -> torch.Tensor:
    """Return [B,8] evidence from sensor depth aligned to Current RGB pixels.

    A *soft* top-k match identifies candidate patches without simulator target
    metadata; the depth within those candidate patches is averaged over valid
    pixels only. No-match/invalid-depth cases produce zeroed depth statistics
    with a zero matched-validity indicator.

    Camera projection is shared by RGB/Depth; the scene map is downsampled to
    the frozen RGB encoder's 24x24 token grid. The current model's visual
    matching scores supply only patch selection, not ground-truth masks.
    """
    if max_depth_m <= 0 or topk <= 0 or match_temperature <= 0:
        raise ValueError("max_depth_m, topk and temperature must be positive")

    if current_tokens.ndim != 3 or tuple(current_tokens.shape[1:]) != (
        ENCODER.num_tokens, ENCODER.token_dim
    ):
        raise ValueError(f"Expected current [B,{ENCODER.num_tokens},{ENCODER.token_dim}]")
    if goal_tokens.ndim != 3 or tuple(goal_tokens.shape[1:]) != (
        AUX.goal_token_count, ENCODER.token_dim
    ):
        raise ValueError(f"Expected goal [B,{AUX.goal_token_count},{ENCODER.token_dim}]")
    if current_tokens.shape[0] != goal_tokens.shape[0]:
        raise ValueError("Current and Goal batch sizes differ")

    # The simulator returns a NumPy array. Accept either ndarray or Tensor
    # without assuming torch-only methods are available on the input.
    if not isinstance(depth_map_m, (np.ndarray, torch.Tensor)):
        raise TypeError("Depth map must be a NumPy array or torch.Tensor")
    depth = torch.as_tensor(
        depth_map_m, dtype=torch.float32, device=current_tokens.device
    )
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if depth.ndim != 3 or depth.shape[0] != current_tokens.shape[0]:
        raise ValueError("Expected depth map [B,H,W] (or [H,W] when B=1)")

    # The RGB encoder uses a fixed 336x336 image with 14x14 patches.
    # Silently resizing another sensor geometry would break correspondence.
    expected_hw = (ENCODER.image_size, ENCODER.image_size)
    if tuple(depth.shape[-2:]) != expected_hw:
        raise ValueError(
            f"Depth spatial shape must equal aligned RGB {expected_hw}, "
            f"got {tuple(depth.shape[-2:])}"
        )
    if ENCODER.image_size != ENCODER.grid_size * ENCODER.patch_size:
        raise ValueError("RGB token/patch geometry is inconsistent")
    valid = torch.isfinite(depth) & (depth > 0) & (depth <= max_depth_m)
    safe_depth = torch.where(valid, depth, torch.zeros_like(depth))
    size = (ENCODER.grid_size, ENCODER.grid_size)

    # Patch means of *valid* scene returns, plus support/coverage per patch.
    valid_fraction = F.adaptive_avg_pool2d(valid.float().unsqueeze(1), size).flatten(1)
    depth_sum = F.adaptive_avg_pool2d(safe_depth.unsqueeze(1), size).flatten(1)
    depth_sq = F.adaptive_avg_pool2d(safe_depth.square().unsqueeze(1), size).flatten(1)
    patch_mean = depth_sum / valid_fraction.clamp_min(1e-8)
    patch_second = depth_sq / valid_fraction.clamp_min(1e-8)
    patch_var = (patch_second - patch_mean.square()).clamp_min(0.0)

    similarity = goal_current_similarity(current_tokens, goal_tokens).amax(dim=1)
    k = min(int(topk), int(similarity.shape[1]))
    scores, indices = torch.topk(similarity, k=k, dim=1)
    match_weights = torch.softmax(scores / match_temperature, dim=1)

    mean_selected = patch_mean.gather(1, indices)
    var_selected = patch_var.gather(1, indices)
    valid_selected = valid_fraction.gather(1, indices)

    # Discount patches without valid depth; never treat a missing return as
    # near distance. Preserve matched-validity as an explicit model input.
    mass = match_weights * valid_selected
    matched_validity = mass.sum(dim=1)
    depth_weights = mass / matched_validity.unsqueeze(-1).clamp_min(1e-8)
    mean = (depth_weights * mean_selected).sum(dim=1)
    second = (
        depth_weights * (var_selected + mean_selected.square())
    ).sum(dim=1)
    std = (second - mean.square()).clamp_min(0.0).sqrt()
    min_candidates = torch.where(
        valid_selected > 0, mean_selected,
        torch.full_like(mean_selected, float("inf")),
    )
    minimum = min_candidates.min(dim=1).values
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))

    # Depth statistics are slant-range sensor statistics, not goal distance.
    depth_available = matched_validity > 1e-8
    proximity = torch.where(
        depth_available,
        1.0 - (mean / max_depth_m).clamp(0, 1),
        torch.zeros_like(mean),
    )
    # Cosine scores are [-1,1], but all eight emitted features are [0,1].
    # This avoids accidental scale mismatch in a later small Arrival probe.
    peak = ((scores[:, 0] + 1.0) * 0.5).clamp(0, 1)
    margin = (
        (scores[:, 0] - scores[:, 1]) * 0.5
        if k >= 2 else torch.zeros_like(scores[:, 0])
    ).clamp(0, 1)
    scene_validity = valid.float().mean(dim=(1, 2))

    features = torch.stack((
        (mean / max_depth_m).clamp(0, 1),
        (std / max_depth_m).clamp(0, 1),
        (minimum / max_depth_m).clamp(0, 1),
        matched_validity.clamp(0, 1),
        proximity,
        peak,
        margin,
        scene_validity,
    ), dim=-1)
    return features
