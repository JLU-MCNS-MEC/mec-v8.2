"""Stage-2f observable features for Goal Reacquisition / Movement Recovery.

Every feature in `extract_observable_features` is computed from controller
inputs available at run time:

  * current RGB tokens and goal-reference tokens;
  * the 3x8x8 Goal-Current heatmap;
  * the parameter-free baseline/coherent spatial maps;
  * the previous coherent map and the previous executed action (for yaw-aligned
    in-place-turn evidence only; FORWARD translation is never warped);
  * the frozen M0 movement/arrival outputs.

No simulator goal position, true distance, true visibility, success label,
oracle action, or heading error is accepted by this function. Those are
computed separately in the collector/evaluator as offline labels.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import FORWARD, LEFT, RIGHT  # noqa: E402
from gbs_model_v3 import goal_current_similarity  # noqa: E402
from visual_grounding_v83 import yaw_warp_previous  # noqa: E402

HEATMAP_TEMPERATURE = 0.10
BASELINE_TEMPERATURE = 0.10
COHERENT_TEMPERATURE = 0.50
TEMPORAL_CARRY = 0.30


def _map_stats(values, temperature, prefix):
    """Entropy/top-k/centroid/peak statistics of a 2-D score map."""
    height, width = values.shape
    flat = values.reshape(-1).float()
    logits = flat / float(temperature)
    logits = logits - logits.max()
    probabilities = torch.softmax(logits, dim=0)
    count = flat.numel()
    entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
    normalized_entropy = entropy / math.log(count)
    top_k = min(5, count)
    top5 = probabilities.topk(top_k).values.sum()
    peak = int(probabilities.argmax())
    xs = torch.linspace(-1.0, 1.0, width, device=flat.device, dtype=flat.dtype)
    ys = torch.linspace(-1.0, 1.0, height, device=flat.device, dtype=flat.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    centroid_x = (probabilities * xx).sum()
    centroid_y = (probabilities * yy).sum()
    return {
        f"{prefix}_entropy": float(normalized_entropy),
        f"{prefix}_top1": float(probabilities.max()),
        f"{prefix}_top5_mass": float(top5),
        f"{prefix}_centroid_x": float(centroid_x),
        f"{prefix}_centroid_y": float(centroid_y),
        f"{prefix}_peak_x": float(xx[peak]),
        f"{prefix}_peak_y": float(yy[peak]),
        f"{prefix}_peak_prob": float(probabilities[peak]),
        f"{prefix}_raw_max": float(flat.max()),
        f"{prefix}_raw_std": float(flat.std(unbiased=False)),
    }


def _temporal_features(current_coherent, previous_coherent, previous_action, device):
    zero = {
        "temporal_valid": 0.0,
        "temporal_consistency": float("nan"),
        "innovation_mean_abs": float("nan"),
        "innovation_max_abs": float("nan"),
        "temporal_entropy": float("nan"),
        "temporal_top1": float("nan"),
        "temporal_centroid_x": float("nan"),
    }
    if previous_coherent is None:
        return zero
    if int(previous_action) not in (LEFT, RIGHT):
        return zero
    warped, valid = yaw_warp_previous(
        previous_coherent.unsqueeze(0),
        torch.tensor([int(previous_action)], device=device, dtype=torch.long),
    )
    if not bool(valid.item()):
        return zero
    warped_map = warped[0]
    consistency = F.cosine_similarity(
        current_coherent.reshape(1, -1), warped_map.reshape(1, -1), dim=-1
    )
    innovation = current_coherent - warped_map
    temporal = (1.0 - TEMPORAL_CARRY) * current_coherent + TEMPORAL_CARRY * warped_map
    stats = _map_stats(temporal, COHERENT_TEMPERATURE, "temporal")
    return {
        "temporal_valid": 1.0,
        "temporal_consistency": float(consistency),
        "innovation_mean_abs": float(innovation.abs().mean()),
        "innovation_max_abs": float(innovation.abs().max()),
        "temporal_entropy": stats["temporal_entropy"],
        "temporal_top1": stats["temporal_top1"],
        "temporal_centroid_x": stats["temporal_centroid_x"],
    }


def extract_observable_features(
    current_tokens,
    goal_tokens,
    heatmap,
    baseline,
    coherent,
    previous_coherent,
    previous_action,
    move_logits,
    arrival_logit,
):
    """Return a flat dict of observable features for one state.

    Shapes: current_tokens [1,N,D], goal_tokens [1,S,D], heatmap [1,3,8,8],
    baseline/coherent [1,24,24], previous_coherent [24,24] or None,
    previous_action int, move_logits [1,3], arrival_logit [1,1].
    """
    device = current_tokens.device
    features = {}

    # Goal-Current raw similarity confidence.
    sim = goal_current_similarity(current_tokens, goal_tokens)
    features["sim_max"] = float(sim.max())
    features["sim_top5_mean"] = float(sim.topk(5, dim=-1).values.mean())
    features["sim_std"] = float(sim.std(unbiased=False))
    per_scale = sim.amax(dim=-1).reshape(-1)
    for index in range(per_scale.numel()):
        features[f"sim_max_scale{index}"] = float(per_scale[index])

    # Heatmap: per-scale and scale-max statistics.
    for channel in range(heatmap.shape[1]):
        stats = _map_stats(
            heatmap[0, channel], HEATMAP_TEMPERATURE, f"heatmap{channel}"
        )
        features.update(stats)
    combined_heatmap = heatmap[0].max(dim=0).values
    features.update(
        _map_stats(combined_heatmap, HEATMAP_TEMPERATURE, "heatmap_max")
    )

    # Baseline and coherent spatial maps.
    features.update(
        _map_stats(baseline[0], BASELINE_TEMPERATURE, "baseline")
    )
    features.update(
        _map_stats(coherent[0], COHERENT_TEMPERATURE, "coherent")
    )
    baseline_prob = torch.softmax(
        baseline[0].reshape(-1) / BASELINE_TEMPERATURE, dim=0
    ).reshape(baseline.shape[-2], baseline.shape[-1])
    baseline_split = baseline.shape[-1] // 2
    features["baseline_left_mass"] = float(
        baseline_prob[:, :baseline_split].sum()
    )
    features["baseline_right_mass"] = 1.0 - features["baseline_left_mass"]
    coherent_prob = torch.softmax(
        coherent[0].reshape(-1) / COHERENT_TEMPERATURE, dim=0
    )
    split = coherent.shape[-1] // 2
    features["coherent_left_mass"] = float(
        coherent_prob.reshape(coherent.shape[-2], coherent.shape[-1])[:, :split].sum()
    )
    features["coherent_right_mass"] = 1.0 - features["coherent_left_mass"]

    # Yaw-aligned in-place-turn evidence (Forward is never warped).
    features.update(
        _temporal_features(coherent[0], previous_coherent, previous_action, device)
    )

    # Frozen M0 decision confidence (observable).
    probabilities = torch.softmax(move_logits, dim=-1)
    top2 = probabilities.topk(2, dim=-1).values.reshape(-1)
    features["m0_arrival_probability"] = float(torch.sigmoid(arrival_logit).reshape(-1)[0])
    features["m0_move_top1"] = float(top2[0])
    features["m0_move_margin"] = float(top2[0] - top2[1])
    features["m0_move_action"] = int(probabilities.argmax(dim=-1).reshape(-1)[0])
    return features


FEATURE_NAMES = None


def feature_names():
    """Canonical column order for the collected feature table."""
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        # Generate by running the extractor on a deterministic dummy input is
        # fragile; keep an explicit list instead.
        names = [
            "sim_max", "sim_top5_mean", "sim_std",
            "sim_max_scale0", "sim_max_scale1", "sim_max_scale2",
        ]
        for channel in range(3):
            prefix = f"heatmap{channel}"
            names.extend([
                f"{prefix}_entropy", f"{prefix}_top1", f"{prefix}_top5_mass",
                f"{prefix}_centroid_x", f"{prefix}_centroid_y",
                f"{prefix}_peak_x", f"{prefix}_peak_y",
                f"{prefix}_peak_prob", f"{prefix}_raw_max",
                f"{prefix}_raw_std",
            ])
        prefix = "heatmap_max"
        names.extend([
            f"{prefix}_entropy", f"{prefix}_top1", f"{prefix}_top5_mass",
            f"{prefix}_centroid_x", f"{prefix}_centroid_y",
            f"{prefix}_peak_x", f"{prefix}_peak_y",
            f"{prefix}_peak_prob", f"{prefix}_raw_max", f"{prefix}_raw_std",
        ])
        for prefix in ("baseline", "coherent"):
            names.extend([
                f"{prefix}_entropy", f"{prefix}_top1", f"{prefix}_top5_mass",
                f"{prefix}_centroid_x", f"{prefix}_centroid_y",
                f"{prefix}_peak_x", f"{prefix}_peak_y",
                f"{prefix}_peak_prob", f"{prefix}_raw_max",
                f"{prefix}_raw_std",
            ])
        names.extend([
            "baseline_left_mass", "baseline_right_mass",
            "coherent_left_mass", "coherent_right_mass",
            "temporal_valid", "temporal_consistency",
            "innovation_mean_abs", "innovation_max_abs",
            "temporal_entropy", "temporal_top1", "temporal_centroid_x",
            "m0_arrival_probability", "m0_move_top1", "m0_move_margin",
            "m0_move_action",
        ])
        FEATURE_NAMES = names
    return list(FEATURE_NAMES)
