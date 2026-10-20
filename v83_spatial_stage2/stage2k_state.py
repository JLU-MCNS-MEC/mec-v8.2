"""Stage-2k 478-D DQN state and fixed Train-only visual PCA.

State layout (exact):
    128-D visual context
  +  10-D global context
  +  20 x 17-D candidate MEC features
  = 478

The visual encoder is a fixed PCA fitted **only on Train-600 natural states**
and then frozen.  It never sees Dev Val, Final Test or Stage-2j Final Test.
No target coordinate, true distance, oracle action or success label is part of
the state.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2i_data import sorted_shards, visual_descriptor
from v83_spatial_stage2.stage2k_common import (
    BANDWIDTH_SCALE_HZ,
    COMPUTE_SCALE_FLOPS,
    DEADLINE_MS,
    E2E_CLIP_SCALE,
    LATENCY_SCALE_MS,
    MODEL_NAMES,
    POSITION_SCALE_M,
    RATE_SCALE_BPS,
    atomic_write_json,
    read_json,
    utc_now,
)


VISUAL_DIM = 128
GLOBAL_DIM = 10
CANDIDATE_DIM = 17
NUM_NODES = 20
STATE_DIM = VISUAL_DIM + GLOBAL_DIM + NUM_NODES * CANDIDATE_DIM
SUMMARY_DIM = 4 * 2 * 384 + 20 + 22  # 3114
PCA_VERSION = "stage2k-visual-pca-v1"
PCA_SEED = 20261031
PCA_COMPONENTS = 128


# ---------------------------------------------------------------------------
# Observable visual summary
# ---------------------------------------------------------------------------


def visual_summary(
    current_tokens,
    goal_tokens,
    history_vis,
    history_mask,
    gab,
    map_feat,
):
    """Fixed observable summary before PCA.

    Blocks:
      current mean(384) + max(384)
      goal mean(384) + max(384)
      history mean over valid slots(384*2 = 768)
      history most-recent valid slot(768)
      GAB(20) + map_feat(22)
    """
    current = np.asarray(current_tokens, dtype=np.float32)
    goal = np.asarray(goal_tokens, dtype=np.float32)
    history = np.asarray(history_vis, dtype=np.float32)
    mask = np.asarray(history_mask, dtype=np.float32)
    if current.ndim != 2 or current.shape[-1] != 384:
        raise ValueError(f"current_tokens shape {current.shape}")
    if goal.ndim != 2 or goal.shape[-1] != 384:
        raise ValueError(f"goal_tokens shape {goal.shape}")
    if history.ndim != 2 or history.shape != (16, 768):
        raise ValueError(f"history_vis shape {history.shape}")
    if mask.shape != (16,):
        raise ValueError(f"history_mask shape {mask.shape}")

    current_mean = current.mean(axis=0)
    current_max = current.max(axis=0)
    goal_mean = goal.mean(axis=0)
    goal_max = goal.max(axis=0)
    valid = mask > 0.5
    if valid.any():
        history_mean = history[valid].mean(axis=0)
        history_last = history[0]
    else:
        history_mean = np.zeros(768, dtype=np.float32)
        history_last = np.zeros(768, dtype=np.float32)
    return np.concatenate([
        current_mean, current_max,
        goal_mean, goal_max,
        history_mean, history_last,
        np.asarray(gab, dtype=np.float32).reshape(-1),
        np.asarray(map_feat, dtype=np.float32).reshape(-1),
    ]).astype(np.float32)


def summary_dim():
    return int(SUMMARY_DIM)


# ---------------------------------------------------------------------------
# PCA (Train-only, frozen)
# ---------------------------------------------------------------------------


def fit_visual_pca(summaries, n_components=PCA_COMPONENTS, seed=PCA_SEED):
    summaries = np.asarray(summaries, dtype=np.float64)
    if summaries.ndim != 2 or summaries.shape[1] != SUMMARY_DIM:
        raise ValueError(f"summaries shape {summaries.shape}")
    mean = summaries.mean(axis=0)
    std = summaries.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x = (summaries - mean) / std
    if len(x) < 2:
        raise ValueError("Need at least two summaries for PCA")
    covariance = (x.T @ x) / float(len(x) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    components = eigenvectors[:, : int(n_components)].T
    total = float(eigenvalues.sum())
    explained = (
        eigenvalues[: int(n_components)] / total if total > 0 else
        np.zeros(int(n_components), dtype=np.float64)
    )
    return {
        "version": PCA_VERSION,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "components": components.astype(np.float32),
        "eigenvalues": eigenvalues[: int(n_components)].astype(np.float64),
        "explained_variance_ratio": explained.astype(np.float64),
        "explained_variance_total": float(explained.sum()),
        "n_samples": int(len(summaries)),
        "seed": int(seed),
        "summary_dim": int(SUMMARY_DIM),
        "components_dim": int(n_components),
        "fitted_split": "train_600",
        "fitted_sample_type": "natural",
        "final_test_read": False,
    }


def save_visual_pca(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean=payload["mean"],
        std=payload["std"],
        components=payload["components"],
        eigenvalues=payload["eigenvalues"],
        explained_variance_ratio=payload["explained_variance_ratio"],
        meta=np.asarray(json.dumps({
            key: payload[key] for key in (
                "version", "explained_variance_total", "n_samples", "seed",
                "summary_dim", "components_dim", "fitted_split",
                "fitted_sample_type", "final_test_read",
            )
        })),
    )
    return path


@dataclass(frozen=True)
class VisualProjector:
    mean: np.ndarray
    std: np.ndarray
    components: np.ndarray
    meta: dict

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                mean=data["mean"].astype(np.float32),
                std=data["std"].astype(np.float32),
                components=data["components"].astype(np.float32),
                meta=json.loads(str(data["meta"])),
            )

    def project(self, summary):
        summary = np.asarray(summary, dtype=np.float32)
        if summary.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"summary dim {summary.shape[-1]} != {self.mean.shape[0]}"
            )
        normalized = (summary - self.mean) / self.std
        return (normalized @ self.components.T).astype(np.float32)


def fit_visual_pca_from_train(
    train_dir,
    aux_dir,
    output_path,
    n_samples=20_000,
    seed=PCA_SEED,
):
    """Fit the frozen visual PCA on Train-600 natural states only."""
    train_dir = Path(train_dir)
    aux_dir = Path(aux_dir)
    if "final_test" in str(train_dir) or "final_test" in str(aux_dir):
        raise ValueError("Refusing to fit PCA on Final Test")
    paths = sorted_shards(train_dir)
    aux_paths = sorted(aux_dir.glob("aux_shard_*.npz"))
    if len(paths) != len(aux_paths):
        raise RuntimeError("Train shard/aux mismatch")

    natural_positions = []
    for shard_index, path in enumerate(paths):
        data = torch.load(path, map_location="cpu", weights_only=True)
        sample_type = data["sample_type"].numpy()
        for row in np.where(sample_type == 0)[0]:
            natural_positions.append((shard_index, int(row)))
    if not natural_positions:
        raise RuntimeError("No Train natural samples")
    rng = np.random.default_rng(int(seed))
    if len(natural_positions) > int(n_samples):
        chosen = rng.choice(
            len(natural_positions), size=int(n_samples), replace=False
        )
        chosen = np.sort(chosen)
        natural_positions = [natural_positions[i] for i in chosen]

    # Group by shard and load one shard at a time: caching all 556 shards
    # would hold ~24 GB of token tensors.
    rows_by_shard = {}
    for shard_index, row in natural_positions:
        rows_by_shard.setdefault(int(shard_index), []).append(int(row))
    summaries = []
    for shard_index in sorted(rows_by_shard):
        data = torch.load(
            paths[shard_index], map_location="cpu", weights_only=True
        )
        aux = np.load(aux_paths[shard_index])
        for row in rows_by_shard[shard_index]:
            summaries.append(visual_summary(
                data["tokens"][row].float().numpy(),
                data["goal_tokens"][row].float().numpy(),
                aux["history_vis"][row],
                aux["history_mask"][row],
                aux["gab"][row],
                aux["map_feat"][row],
            ))
        del data, aux
    payload = fit_visual_pca(np.stack(summaries), seed=seed)
    payload["train_dir"] = str(train_dir)
    payload["aux_dir"] = str(aux_dir)
    payload["sample_type"] = "natural"
    payload["created_utc"] = utc_now()
    save_visual_pca(output_path, payload)
    return payload


# ---------------------------------------------------------------------------
# 478-D state
# ---------------------------------------------------------------------------


def _one_hot(index, size):
    out = np.zeros(size, dtype=np.float32)
    if 0 <= int(index) < int(size):
        out[int(index)] = 1.0
    return out


def global_features(
    step,
    max_steps,
    history_mask,
    current_goal_similarity,
    gab,
    previous_e2e_ms,
    previous_action,
    clip_e2e=True,
):
    """10-D global context.

    In the legacy mode the sixth feature is the previous-step E2E delay.  In
    the JointR-aligned mode the caller passes the running mean E2E delay and
    ``clip_e2e=False`` so the feature is exactly ``running_mean_ms / 100``.
    """
    mask = np.asarray(history_mask, dtype=np.float32)
    gab = np.asarray(gab, dtype=np.float32)
    previous_action = int(previous_action)
    action = (
        _one_hot(previous_action, 4)
        if 0 <= previous_action < 4 else np.zeros(4, dtype=np.float32)
    )
    if previous_e2e_ms is None:
        previous_e2e = 0.0
    else:
        scaled = float(previous_e2e_ms) / LATENCY_SCALE_MS
        previous_e2e = (
            float(np.clip(scaled, 0.0, E2E_CLIP_SCALE))
            if clip_e2e else max(0.0, float(scaled))
        )
    values = np.asarray([
        float(step) / max(float(max_steps), 1.0),
        float(mask.sum()) / 16.0,
        float(np.clip(current_goal_similarity, -1.0, 1.0)),
        float(gab[16]) if len(gab) > 16 else 0.0,
        float(gab[17]) if len(gab) > 17 else 0.0,
        previous_e2e,
    ] + action.tolist(), dtype=np.float32)
    if values.shape != (GLOBAL_DIM,):
        raise RuntimeError(f"global feature dim {values.shape}")
    return values


def candidate_features(
    nodes,
    uav_position,
    latency_model,
    previous_selected=-1,
    quality_prior=None,
):
    """20 x 17 candidate features and reachable mask.

    Unavailable nodes keep geometry/quality/type but zero network-rate,
    communication, inference and E2E fields.  If no node covers the UAV, the
    nearest node is re-enabled (environment fallback rule).
    """
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
        # Topologies store the Dev-Val quality prior per node.  A flat dict
        # may be supplied for tests; never a full quality-prior payload.
        if quality_prior is not None and not isinstance(quality_prior, dict):
            raise TypeError("quality_prior must be a flat dict or None")
        if isinstance(quality_prior, dict) and "models" in quality_prior:
            flat_quality = {
                name: float(quality_prior["models"][name]["mean_ndtw"])
                for name in MODEL_NAMES
            }
        else:
            flat_quality = quality_prior
        quality = float(
            node.get("quality_prior", 0.0)
            if flat_quality is None
            else flat_quality.get(
                node["model_type"], node.get("quality_prior", 0.0)
            )
        )
        values = [
            1.0 if available else 0.0,
            dx / POSITION_SCALE_M,
            dy / POSITION_SCALE_M,
            distance / POSITION_SCALE_M,
            float(node["bandwidth_hz"]) / BANDWIDTH_SCALE_HZ,
            float(node["compute_capacity_flops_per_s"]) / COMPUTE_SCALE_FLOPS,
        ]
        if available:
            metric = latency_model.evaluate(
                node, uav_position, node["model_type"]
            )
            metrics[index] = metric
            values.extend([
                float(metric["uplink_rate_bps"]) / RATE_SCALE_BPS,
                float(metric["communication_ms"]) / LATENCY_SCALE_MS,
                float(metric["inference_ms"]) / LATENCY_SCALE_MS,
                min(
                    float(metric["total_ms"]) / LATENCY_SCALE_MS,
                    E2E_CLIP_SCALE,
                ),
            ])
        else:
            values.extend([0.0, 0.0, 0.0, 0.0])
        values.append(float(np.clip(quality, 0.0, 1.0)))
        values.append(1.0 if int(index) == int(previous_selected) else 0.0)
        values.extend(_one_hot(node["model_index"], len(MODEL_NAMES)).tolist())
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
        # Recompute the fallback node with its real network metrics.
        node = nodes[nearest]
        metric = latency_model.evaluate(node, uav_position, node["model_type"])
        metrics[nearest] = metric
        features[nearest, 0] = 1.0
        features[nearest, 7] = float(metric["uplink_rate_bps"]) / RATE_SCALE_BPS
        features[nearest, 8] = float(metric["communication_ms"]) / LATENCY_SCALE_MS
        features[nearest, 9] = float(metric["inference_ms"]) / LATENCY_SCALE_MS
        features[nearest, 10] = min(
            float(metric["total_ms"]) / LATENCY_SCALE_MS, E2E_CLIP_SCALE
        )
    return features, mask, metrics, fallback


def build_state(visual128, global10, candidate20x17):
    visual128 = np.asarray(visual128, dtype=np.float32).reshape(-1)
    global10 = np.asarray(global10, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate20x17, dtype=np.float32)
    if visual128.shape != (VISUAL_DIM,):
        raise ValueError(f"visual dim {visual128.shape}")
    if global10.shape != (GLOBAL_DIM,):
        raise ValueError(f"global dim {global10.shape}")
    if candidate.shape != (NUM_NODES, CANDIDATE_DIM):
        raise ValueError(f"candidate shape {candidate.shape}")
    state = np.concatenate([visual128, global10, candidate.reshape(-1)])
    if state.shape != (STATE_DIM,):
        raise RuntimeError(f"state dim {state.shape}")
    return state.astype(np.float32)


def masked_q_values(q_values, mask):
    q_values = np.asarray(q_values, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if q_values.shape != (NUM_NODES,) or mask.shape != (NUM_NODES,):
        raise ValueError("q/mask must have 20 entries")
    if not mask.any():
        raise ValueError("mask has no reachable node")
    output = np.full(NUM_NODES, -np.inf, dtype=np.float64)
    output[mask] = q_values[mask]
    return output


def masked_argmax(q_values, mask, tie_rng=None):
    output = masked_q_values(q_values, mask)
    best = np.max(output)
    candidates = np.where(output == best)[0]
    if tie_rng is not None and len(candidates) > 1:
        return int(tie_rng.choice(candidates))
    return int(candidates[0])


def state_layout():
    return {
        "visual": [0, VISUAL_DIM],
        "global": [VISUAL_DIM, VISUAL_DIM + GLOBAL_DIM],
        "candidates": [
            VISUAL_DIM + GLOBAL_DIM,
            VISUAL_DIM + GLOBAL_DIM + NUM_NODES * CANDIDATE_DIM,
        ],
        "total": STATE_DIM,
        "candidate_fields": [
            "availability",
            "relative_x",
            "relative_y",
            "distance",
            "bandwidth_norm",
            "compute_norm",
            "uplink_rate_norm",
            "communication_ms_norm",
            "inference_ms_norm",
            "e2e_ms_norm",
            "dev_val_quality_prior",
            "previous_selected",
            "model_Small",
            "model_Transformer-S",
            "model_Medium",
            "model_Transformer-M",
            "model_Large",
        ],
    }
