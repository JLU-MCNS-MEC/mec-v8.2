"""Shared utilities for V8.3 Stage-2d Arrival calibration / Stop rules.

This module is intentionally dependency-light on the training scripts:
  * loads the frozen Stage-2a M0 checkpoint and the existing Stage-2c
    Train-600 rollout cache (no re-collection);
  * episode-level 480/120 split;
  * far-negative mining controls;
  * R1 Platt scaling and R2/R3 104-D residual calibration modules;
  * offline Arrival metrics and per-decision hazard metrics;
  * a closed-loop evaluator that re-runs the environment for every Stop rule
    and resets the decision state at the start of every episode.

True distance / Goal visibility appear only in labels and post-decision
diagnostics. Calibration modules accept exactly `(features, base_logit)`.
Final Test is never loaded here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_fixed_mec_epoch_compare as evaluation
from config import ARRIVAL_STOP_THRESHOLD, ENCODER, ENV
from env import STOP, ToyUAVEnv
from gbs_network import build_random_gbs_layout, global_network_state
from navigation_reward import ndtw_score
from token_data import encode_multiscale_goal_tokens, goal_conditioned_spatial_heatmap
from utils import image_to_tensor
from v83_spatial_training.eval_closedloop_spatial import load_encoder, paired_comparison
from v83_spatial_stage2.stage2b_arrival_ablation import (
    M0_CKPT,
    _hash_state,
    _load_m0_and_head,
)


STAGE2_ROOT = Path("runs/v83_spatial_stage2")
STAGE2C_ROLLOUT = (
    STAGE2_ROOT
    / "stage2c/offpolicy_train_rollouts_20261019T013500Z/rollout_steps.npz"
)
STAGE2B_DEV_FEATURES = (
    STAGE2_ROOT / "stage2b/arrival_20261018T150031Z/features/dev_val.npz"
)
STAGE2A_EVAL = STAGE2_ROOT / "stage2a_eval_20261018T070139Z"
VAL_SPECS = STAGE2_ROOT / "prep_audit_20261018T043420Z/dev_val_180_spec.json"
BANDS = ("short", "medium", "long")
BAND_ID = {"short": 0, "medium": 1, "long": 2}
STOP_RADIUS_M = float(ENV.stop_radius_m)
FAR_RADIUS_M = 40.0
HARD_PROB_THRESHOLD = 0.3
SPLIT_SEED = 20261019
TRAIN_EPISODES = 480
TOTAL_EPISODES = 600
FEATURE_DIM = 104
DISTANCE_BINS = (
    (0.0, 20.0),
    (20.0, 40.0),
    (40.0, 60.0),
    (60.0, 100.0),
    (100.0, 200.0),
    (200.0, float("inf")),
)


# ---------------------------------------------------------------------------
# Data loading / split
# ---------------------------------------------------------------------------


def _reject_final_test(path):
    if "final_test" in str(path):
        raise ValueError(f"Final Test is sealed; refusing to read {path}")


def load_rollout_steps(path=STAGE2C_ROLLOUT):
    """Load the existing Train-600 B0 rollout cache and derive labels."""
    _reject_final_test(path)
    cached = np.load(Path(path))
    data = {key: cached[key] for key in cached.files}
    required = {
        "features", "episode_index", "seed", "band_id", "step", "distance_m",
        "goal_visible", "arrival_probability", "arrival_logit", "stop", "action",
    }
    missing = required - set(data)
    if missing:
        raise KeyError(f"Rollout cache is missing keys: {sorted(missing)}")
    if data["features"].ndim != 2 or data["features"].shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected {FEATURE_DIM}-D Arrival features, got {data['features'].shape}"
        )
    label = (data["distance_m"] < STOP_RADIUS_M).astype(np.float32)
    data["arrival_label"] = label
    probability_from_logit = 1.0 / (
        1.0 + np.exp(-np.clip(data["arrival_logit"].astype(np.float64), -60.0, 60.0))
    )
    diff = np.abs(probability_from_logit - data["arrival_probability"])
    if float(diff.max()) > 1e-6:
        raise RuntimeError(
            f"Cached Arrival probability/logit mismatch: max |diff|={diff.max():.3e}"
        )
    data["probability_logit_max_abs_diff"] = float(diff.max())
    return data


def load_dev_val_features(path=STAGE2B_DEV_FEATURES):
    """Load the cached Dev Val 180 frozen Arrival features (evaluation only)."""
    _reject_final_test(path)
    cached = np.load(Path(path))
    data = {key: cached[key] for key in cached.files}
    if data["features"].shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM}-D features, got {data['features'].shape}")
    return data


def split_episodes(episode_index, seed=SPLIT_SEED, n_train=TRAIN_EPISODES):
    """Deterministic episode-level split; no episode appears in both halves."""
    episodes = np.unique(np.asarray(episode_index, dtype=np.int64))
    if len(episodes) != TOTAL_EPISODES:
        raise ValueError(f"Expected {TOTAL_EPISODES} episodes, got {len(episodes)}")
    rng = np.random.default_rng(int(seed))
    permuted = episodes[rng.permutation(len(episodes))]
    train = np.sort(permuted[: int(n_train)])
    val = np.sort(permuted[int(n_train):])
    if set(train.tolist()) & set(val.tolist()):
        raise RuntimeError("Episode split overlap")
    if len(train) + len(val) != len(episodes):
        raise RuntimeError("Episode split does not cover all episodes")
    return train, val


def episode_mask(episode_index, episodes):
    return np.isin(np.asarray(episode_index, dtype=np.int64), np.asarray(episodes))


def feature_normalization(features, mask):
    """Fixed standardization statistics shared by R2 and R3."""
    subset = np.asarray(features)[np.asarray(mask, dtype=bool)].astype(np.float64)
    mean = subset.mean(axis=0)
    std = subset.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Far-negative mining
# ---------------------------------------------------------------------------


def far_negative_mask(distance, label):
    return (np.asarray(distance) > FAR_RADIUS_M) & (np.asarray(label) < 0.5)


def mine_far_negatives(distance, label, base_probability, train_mask,
                       hard_p=HARD_PROB_THRESHOLD, seed=SPLIT_SEED):
    """Hard-mined and equal-size random far-negative controls within Train-480.

    Hard: far negative with B0 p >= hard_p.
    Random: equal-size sample of far negatives with p < hard_p, disjoint from
    the hard set. The same candidate pool is used for both variants, so R2/R3
    differ only in which far negatives are added.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    far = far_negative_mask(distance, label) & train_mask
    hard = far & (np.asarray(base_probability) >= float(hard_p))
    hard_indices = np.flatnonzero(hard)
    candidates = np.flatnonzero(far & ~hard)
    if len(hard_indices) > len(candidates):
        raise RuntimeError(
            f"Not enough random far negatives: hard={len(hard_indices)} "
            f"candidates={len(candidates)}"
        )
    rng = np.random.default_rng(int(seed))
    random_indices = np.sort(
        rng.choice(candidates, size=len(hard_indices), replace=False)
    )
    if set(hard_indices.tolist()) & set(random_indices.tolist()):
        raise RuntimeError("Hard and random far-negative sets overlap")
    return hard_indices, random_indices


def build_training_rows(distance, label, base_probability, episode_index,
                        train_episodes, kind, hard_p=HARD_PROB_THRESHOLD,
                        seed=SPLIT_SEED):
    """Rows = all Train-480 non-far states + hard (R2/R1) or random (R3) far.

    `distance <= 40 m` base rows are the near positives, the 20-40 m boundary
    negatives and the 0-20 m negatives. Far additions are always >40 m, so the
    two sets are disjoint.
    """
    if kind not in ("hard", "random", "base_only"):
        raise ValueError(kind)
    train_mask = episode_mask(episode_index, train_episodes)
    base_mask = train_mask & (np.asarray(distance) <= FAR_RADIUS_M)
    hard_indices, random_indices = mine_far_negatives(
        distance, label, base_probability, train_mask, hard_p=hard_p, seed=seed,
    )
    if kind == "hard":
        additions = hard_indices
    elif kind == "random":
        additions = random_indices
    else:
        additions = np.zeros(0, dtype=np.int64)
    base_indices = np.flatnonzero(base_mask)
    rows = np.concatenate([base_indices, additions]).astype(np.int64)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("Training rows contain duplicates")
    anchor = np.asarray(distance)[rows] <= FAR_RADIUS_M
    stats = {
        "kind": kind,
        "train_episodes": int(len(train_episodes)),
        "base_rows": int(len(base_indices)),
        "base_positives": int((np.asarray(label)[base_indices] >= 0.5).sum()),
        "base_negatives": int((np.asarray(label)[base_indices] < 0.5).sum()),
        "hard_far_available": int(len(hard_indices)),
        "random_far_available": int(len(random_indices)),
        "added_far_rows": int(len(additions)),
        "total_rows": int(len(rows)),
        "anchor_rows": int(anchor.sum()),
        "far_rows": int((~anchor).sum()),
        "hard_probability_threshold": float(hard_p),
        "seed": int(seed),
    }
    return {
        "row_indices": rows,
        "anchor_mask": anchor,
        "hard_indices": hard_indices,
        "random_indices": random_indices,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Calibration modules (inputs are observable only)
# ---------------------------------------------------------------------------


class PlattScaling(nn.Module):
    """R1: logit' = a * logit_base + b, initialized at a=1, b=0."""

    def __init__(self):
        super().__init__()
        self.scale_delta = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, features, base_logit):
        # `features` is intentionally unused: R1 is the calibration-only
        # control. Keeping the same signature makes the interface leak-free.
        del features
        scale = torch.clamp(1.0 + self.scale_delta, min=1e-3)
        return scale * base_logit + self.bias


class ResidualCalibration(nn.Module):
    """R2/R3: logit' = logit_base + r(104-D features), zero-init last layer."""

    def __init__(self, feature_mean, feature_std, hidden_dim=32):
        super().__init__()
        mean = torch.as_tensor(feature_mean, dtype=torch.float32).clone()
        std = torch.as_tensor(feature_std, dtype=torch.float32).clone()
        if mean.numel() != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM}-D feature mean, got {mean.numel()}")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std.clamp_min(1e-6))
        self.net = nn.Sequential(
            nn.Linear(int(mean.numel()), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, base_logit):
        normalized = (features - self.feature_mean) / self.feature_std
        return base_logit + self.net(normalized).squeeze(-1)


def build_calibrator(kind, feature_mean=None, feature_std=None):
    if kind == "R1":
        return PlattScaling()
    if kind in ("R2", "R3"):
        if feature_mean is None or feature_std is None:
            raise ValueError(f"{kind} requires feature mean/std")
        return ResidualCalibration(feature_mean, feature_std)
    raise ValueError(f"Unknown calibrator {kind}")


def parameter_count(module):
    return int(sum(p.numel() for p in module.parameters()))


def bernoulli_kl(base_logit, other_logit, epsilon=1e-6):
    """KL(Bernoulli(sigmoid(base)) || Bernoulli(sigmoid(other)))."""
    p = torch.sigmoid(base_logit).clamp(epsilon, 1.0 - epsilon).detach()
    q = torch.sigmoid(other_logit).clamp(epsilon, 1.0 - epsilon)
    return (
        p * (p.log() - q.log())
        + (1.0 - p) * ((1.0 - p).log() - (1.0 - q).log())
    ).mean()


def sigmoid_numpy(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)))


# ---------------------------------------------------------------------------
# Offline metrics
# ---------------------------------------------------------------------------


def _ece(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (probabilities >= low) & (
            probabilities < high if index < bins - 1 else probabilities <= high
        )
        if mask.any():
            total += mask.mean() * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return float(total)


def binary_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels).astype(bool)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= float(threshold)
    tp = int((labels & predicted).sum())
    fp = int((~labels & predicted).sum())
    fn = int((labels & ~predicted).sum())
    tn = int((~labels & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": float(np.mean((probabilities - labels.astype(float)) ** 2)),
        "ece": _ece(labels.astype(float), probabilities),
    }


def _distance_bin_rows(labels, probabilities, distance, threshold):
    rows = []
    for low, high in DISTANCE_BINS:
        mask = (distance >= low) & (distance < high)
        if not mask.any():
            continue
        rows.append({
            "distance_bin": (
                f"{int(low)}_{int(high) if np.isfinite(high) else 'inf'}m"
            ),
            "n": int(mask.sum()),
            "positive_rate": float(np.asarray(labels)[mask].mean()),
            "mean_prob": float(np.asarray(probabilities)[mask].mean()),
            "pred_positive_rate": float(
                (np.asarray(probabilities)[mask] >= float(threshold)).mean()
            ),
            "brier": float(
                np.mean(
                    (np.asarray(probabilities)[mask]
                     - np.asarray(labels)[mask].astype(float)) ** 2
                )
            ),
        })
    return rows


def arrival_offline_metrics(labels, probabilities, distance, sample_type=None,
                            threshold=0.5):
    """Natural/hard Arrival metrics plus far-FPR / near-recall diagnostics.

    `sample_type=None` treats every row as natural (on-policy rollout states).
    """
    labels = np.asarray(labels).astype(np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    if sample_type is None:
        natural = np.ones(len(labels), dtype=bool)
    else:
        natural = np.asarray(sample_type) == 0
    hard = ~natural
    far40 = natural & (distance > FAR_RADIUS_M)
    far100 = natural & (distance >= 100.0)
    near = natural & (distance < STOP_RADIUS_M)
    predicted = probabilities >= float(threshold)
    result = {
        "natural": binary_metrics(labels[natural], probabilities[natural], threshold),
        "hard": (
            binary_metrics(labels[hard], probabilities[hard], threshold)
            if hard.any() else None
        ),
        "far40_false_stop_rate": (
            float(predicted[far40].mean()) if far40.any() else 0.0
        ),
        "far100_false_stop_rate": (
            float(predicted[far100].mean()) if far100.any() else 0.0
        ),
        "far40_n": int(far40.sum()),
        "far100_n": int(far100.sum()),
        "near20_n": int(near.sum()),
        "near20_recall": float(predicted[near].mean()) if near.any() else 0.0,
        "near20_miss_stop_rate": (
            float((~predicted[near]).mean()) if near.any() else 0.0
        ),
        "far40_mean_prob": (
            float(probabilities[far40].mean()) if far40.any() else 0.0
        ),
        "far100_mean_prob": (
            float(probabilities[far100].mean()) if far100.any() else 0.0
        ),
        "near20_mean_prob": float(probabilities[near].mean()) if near.any() else 0.0,
        "high_p_far40_count": int((predicted & far40).sum()),
        "high_p_far40_rate": (
            float(predicted[far40].mean()) if far40.any() else 0.0
        ),
        "distance_bins": _distance_bin_rows(
            labels[natural], probabilities[natural], distance[natural], threshold
        ),
    }
    if hard.any():
        hard_distance = distance[hard]
        hard_bins = []
        for low, high in DISTANCE_BINS:
            mask = (hard_distance >= low) & (hard_distance < high)
            if mask.any():
                hard_bins.append({
                    "distance_bin": (
                        f"{int(low)}_{int(high) if np.isfinite(high) else 'inf'}m"
                    ),
                    "n": int(mask.sum()),
                    "positive_rate": float(labels[hard][mask].mean()),
                    "mean_prob": float(probabilities[hard][mask].mean()),
                    "pred_positive_rate": float(
                        (probabilities[hard][mask] >= float(threshold)).mean()
                    ),
                })
        result["hard_distance_bins"] = hard_bins
    return result


def per_decision_hazard(probabilities, distance, episode_index, threshold=0.5,
                        stopped=None):
    """Exact per-decision exposure and false-Stop hazard decomposition.

    `stopped` is the actual Stop decision made by the policy. When omitted it
    falls back to `probability >= threshold` (threshold-only rules and offline
    calibrated probabilities). Persistence rules must pass their actual `stop`
    flags, otherwise the hazard would ignore the confirmation step.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    episode_index = np.asarray(episode_index, dtype=np.int64)
    episodes = np.unique(episode_index)
    n_episodes = len(episodes)
    if n_episodes == 0:
        raise ValueError("No episodes in hazard computation")
    far = distance > FAR_RADIUS_M
    boundary = (distance >= STOP_RADIUS_M) & (distance <= FAR_RADIUS_M)
    near = distance < STOP_RADIUS_M
    use_actual_stops = stopped is not None
    if stopped is None:
        stopped = probabilities >= float(threshold)
    else:
        stopped = np.asarray(stopped, dtype=bool)
        if stopped.shape != probabilities.shape:
            raise ValueError(
                f"stopped shape {stopped.shape} != probabilities {probabilities.shape}"
            )

    def _rate(mask):
        return float(stopped[mask].mean()) if mask.any() else 0.0

    far_stops = int((stopped & far).sum())
    boundary_stops = int((stopped & boundary).sum())
    near_stops = int((stopped & near).sum())
    return {
        "episodes": int(n_episodes),
        "stop_decisions_from_actual_rule": bool(use_actual_stops),
        "far40_steps": int(far.sum()),
        "far40_steps_per_episode": float(far.sum() / n_episodes),
        "far40_stop_rate": _rate(far),
        "far40_stops": far_stops,
        "expected_false_stop_decisions_per_episode": float(far_stops / n_episodes),
        "predicted_false_stop_decisions_per_episode": float(
            probabilities[far].sum() / n_episodes
        ),
        "boundary_steps": int(boundary.sum()),
        "boundary_stop_rate": _rate(boundary),
        "boundary_stops": boundary_stops,
        "expected_boundary_stop_decisions_per_episode": float(
            boundary_stops / n_episodes
        ),
        "predicted_boundary_stop_decisions_per_episode": float(
            probabilities[boundary].sum() / n_episodes
        ),
        "near20_steps": int(near.sum()),
        "near20_stop_rate": _rate(near),
        "near20_stops": near_stops,
        "near20_episodes_reached": int(
            len(np.unique(episode_index[near])) if near.any() else 0
        ),
        "predicted_non_target_stop_decisions_per_episode": float(
            probabilities[distance >= STOP_RADIUS_M].sum() / n_episodes
        ),
    }


def selection_score(hazard, b0_near_recall, near_tolerance=0.05, penalty=20.0,
                    smooth_weight=0.02):
    """Lower is better. F1 is not part of this objective.

    Primary term: binary far-40 false Stop decisions per episode on the
    calibration-validation rollout (the cumulative-risk target).
    Small smooth term: the summed calibrated probability over far-40 steps,
    used only as a tie-breaker for early stopping.
    Penalty: near-20 Stop recall below (B0 recall - tolerance).
    """
    floor = float(b0_near_recall) - float(near_tolerance)
    shortfall = max(0.0, floor - float(hazard["near20_stop_rate"]))
    return float(
        hazard["expected_false_stop_decisions_per_episode"]
        + float(smooth_weight)
        * hazard["predicted_false_stop_decisions_per_episode"]
        + float(penalty) * shortfall
    )


# ---------------------------------------------------------------------------
# Stop decision rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StopRule:
    name: str
    threshold: float = ARRIVAL_STOP_THRESHOLD
    confirm_k: int = 1
    confirm_n: int = 1

    def __post_init__(self):
        if not 0.0 < float(self.threshold) < 1.0:
            raise ValueError(f"threshold must be in (0,1): {self.threshold}")
        if self.confirm_k < 1 or self.confirm_n < 1:
            raise ValueError("confirm_k / confirm_n must be >= 1")
        if self.confirm_k > self.confirm_n:
            raise ValueError("confirm_k cannot exceed confirm_n")

    def new_state(self):
        return StopConfirmation(self)


class StopConfirmation:
    """Per-episode Stop confirmation state. Create one per episode."""

    def __init__(self, rule):
        self.rule = rule
        self.history = deque(maxlen=rule.confirm_n)

    def reset(self):
        self.history.clear()

    def update(self, probability):
        value = float(probability)
        self.history.append(value)
        if self.rule.confirm_k == 1 and self.rule.confirm_n == 1:
            return value >= float(self.rule.threshold)
        if len(self.history) < self.rule.confirm_n:
            return False
        hits = sum(1 for item in self.history if item >= float(self.rule.threshold))
        return hits >= self.rule.confirm_k

    def snapshot(self):
        return list(self.history)


def rule_from_name(name):
    rules = {
        "threshold_0.5": StopRule("threshold_0.5", 0.50, 1, 1),
        "threshold_0.6": StopRule("threshold_0.6", 0.60, 1, 1),
        "threshold_0.7": StopRule("threshold_0.7", 0.70, 1, 1),
        "persist_2of2_t0.5": StopRule("persist_2of2_t0.5", 0.50, 2, 2),
        "persist_3of3_t0.5": StopRule("persist_3of3_t0.5", 0.50, 3, 3),
        "persist_2of2_t0.6": StopRule("persist_2of2_t0.6", 0.60, 2, 2),
    }
    if name not in rules:
        raise KeyError(f"Unknown rule {name}")
    return rules[name]


# ---------------------------------------------------------------------------
# Model hashes / loading
# ---------------------------------------------------------------------------


def load_m0(device):
    model, state, head = _load_m0_and_head(device)
    model.eval()
    head.eval()
    return model, state, head


def movement_hash(state):
    return _hash_state(
        {key: value for key, value in state.items()
         if not key.startswith("base.arrival_head.")}
    )


def arrival_head_hash(state):
    return _hash_state(
        {key: value for key, value in state.items()
         if key.startswith("base.arrival_head.")}
    )


# ---------------------------------------------------------------------------
# Specs / oracle
# ---------------------------------------------------------------------------


def load_dev_specs(val_specs=VAL_SPECS, max_episodes=180):
    _reject_final_test(val_specs)
    evaluation.TEST_STATS_PATH = Path(val_specs)
    evaluation.NUM_EPISODES = int(max_episodes)
    specs = evaluation.load_test_episode_specs()
    if len(specs) != int(max_episodes):
        raise RuntimeError(f"Expected {max_episodes} specs, got {len(specs)}")
    return specs


def build_oracle_cache(specs):
    return {
        int(spec["episode_index"]): evaluation.build_oracle_reference(spec)
        for spec in specs
    }


# ---------------------------------------------------------------------------
# Closed-loop evaluation
# ---------------------------------------------------------------------------


def run_episode(model, spec, encoder, device, rule, calibrator=None,
                record_steps=True):
    """Run one episode with a fresh Stop-confirmation state."""
    env = ToyUAVEnv()
    obs = env.reset(
        seed=spec["seed"],
        goal_distance_range_m=spec["distance_range"],
        render_observation=True,
    )
    nodes = build_random_gbs_layout(
        seed=spec["network_layout_seed"], verbose=False
    )
    goal_tokens = encode_multiscale_goal_tokens(
        env.get_goal_reference_multiscale_rgbs(), encoder, device
    ).unsqueeze(0).to(device, dtype=torch.float32)

    confirmation = rule.new_state()
    trajectory = [env.position.copy()]
    min_distance = float(env.distance_to_goal)
    osr = bool(env.distance_to_goal < env.stop_radius)
    success = False
    done = False
    ended_by_stop = False
    steps = []
    stop_events = []

    for step in range(evaluation.MAX_STEPS):
        current_tokens = encoder(image_to_tensor(obs, device))
        heatmap = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
        network_np = global_network_state(
            nodes, env.position,
            num_tokens=ENCODER.num_tokens, token_dim=ENCODER.token_dim,
        ).astype(np.float32)
        network = torch.from_numpy(network_np).unsqueeze(0).to(device)
        with torch.inference_mode():
            components = model.forward_components(
                current_tokens, goal_tokens, heatmap, network
            )
            base_logit = components["arrival_logit"].reshape(-1)
            features = components["arrival_features"]
            if calibrator is not None:
                calibrated_logit = calibrator(features, base_logit)
            else:
                calibrated_logit = base_logit
            probability = torch.sigmoid(calibrated_logit)
            base_probability = torch.sigmoid(base_logit)
            move_action = int(torch.argmax(components["move_logits"], dim=-1).item())
        probability_value = float(probability.item())
        should_stop = confirmation.update(probability_value)
        action = STOP if should_stop else move_action
        distance = float(env.distance_to_goal)
        projection = env.goal_projection()
        goal_visible = int(projection is not None)
        if record_steps:
            steps.append({
                "rule": rule.name,
                "episode_index": int(spec["episode_index"]),
                "seed": int(spec["seed"]),
                "band": str(spec["band"]),
                "step": int(step),
                "distance_m": distance,
                "goal_visible": goal_visible,
                "arrival_probability": probability_value,
                "base_arrival_probability": float(base_probability.item()),
                "base_arrival_logit": float(base_logit.item()),
                "calibrated_arrival_logit": float(calibrated_logit.item()),
                "stop": int(action == STOP),
                "action": int(action),
                "move_action": int(move_action),
            })
        if action == STOP:
            stop_events.append({
                "rule": rule.name,
                "episode_index": int(spec["episode_index"]),
                "seed": int(spec["seed"]),
                "band": str(spec["band"]),
                "step": int(step),
                "distance_m": distance,
                "arrival_probability": probability_value,
                "goal_visible": goal_visible,
            })
        obs, step_success, done = env.step(
            action, render_observation=(action != STOP)
        )
        success = bool(step_success)
        ended_by_stop = bool(action == STOP)
        trajectory.append(env.position.copy())
        min_distance = min(min_distance, float(env.distance_to_goal))
        osr = bool(osr or env.distance_to_goal < env.stop_radius)
        if done:
            break

    trajectory = np.asarray(trajectory, dtype=np.float32)
    path_length = (
        float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())
        if len(trajectory) > 1 else 0.0
    )
    # Filled by evaluate_policy with the spec's oracle reference.
    oracle = spec.get("_oracle_reference")
    if oracle is None:
        raise RuntimeError("Spec is missing _oracle_reference; use evaluate_policy")
    shortest = float(oracle["shortest_distance_m"])
    spl = shortest / max(shortest, path_length, 1e-8) if success else 0.0
    ndtw = ndtw_score(
        trajectory=trajectory[:, :2],
        reference=oracle["reference"],
        distance_threshold=env.stop_radius,
    )
    premature_stop = bool(ended_by_stop and not success)
    timeout = bool((not ended_by_stop) and (not success) and done)
    episode = {
        "rule": rule.name,
        "episode_index": int(spec["episode_index"]),
        "seed": int(spec["seed"]),
        "band": str(spec["band"]),
        "success": int(success),
        "osr": int(osr),
        "premature_stop": int(premature_stop),
        "max_steps_termination": int(timeout),
        "steps": int(len(trajectory) - 1),
        "spl": float(spl),
        "ndtw": float(ndtw),
        "min_distance_m": float(min_distance),
        "final_distance_m": float(env.distance_to_goal),
        "reached_radius_not_success": int(osr and not success),
        "stop_distance_m": (
            float(stop_events[0]["distance_m"]) if stop_events else float("nan")
        ),
        "stop_arrival_probability": (
            float(stop_events[0]["arrival_probability"])
            if stop_events else float("nan")
        ),
        "stop_goal_visible": (
            int(stop_events[0]["goal_visible"]) if stop_events else -1
        ),
    }
    return episode, steps, stop_events


def evaluate_policy(model, specs, oracle_cache, encoder, device, rule,
                    calibrator=None, progress_every=30):
    """Re-run the full closed loop under one Stop rule."""
    episodes, all_steps, all_stops = [], [], []
    for position, spec in enumerate(specs):
        spec = dict(spec)
        spec["_oracle_reference"] = oracle_cache[int(spec["episode_index"])]
        episode, steps, stops = run_episode(
            model, spec, encoder, device, rule, calibrator=calibrator
        )
        episodes.append(episode)
        all_steps.extend(steps)
        all_stops.extend(stops)
        if progress_every and (position + 1) % int(progress_every) == 0:
            print(
                f"  {rule.name}: {position + 1}/{len(specs)} episodes",
                flush=True,
            )
    return episodes, all_steps, all_stops


# ---------------------------------------------------------------------------
# Summaries / IO
# ---------------------------------------------------------------------------


def summarize_outcomes(rows):
    n = len(rows)
    if n == 0:
        return {"episodes": 0}

    def _mean(values):
        return float(np.mean(values)) if values else float("nan")

    return {
        "episodes": int(n),
        "sr": _mean([int(row["success"]) for row in rows]),
        "osr": _mean([int(row["osr"]) for row in rows]),
        "spl": _mean([float(row["spl"]) for row in rows]),
        "ndtw": _mean([float(row["ndtw"]) for row in rows]),
        "premature_stop": _mean([int(row["premature_stop"]) for row in rows]),
        "timeout": _mean([int(row["max_steps_termination"]) for row in rows]),
        "mean_steps": _mean([int(row["steps"]) for row in rows]),
        "mean_min_distance_m": _mean([float(row["min_distance_m"]) for row in rows]),
        "reached_radius_not_success": _mean(
            [int(row["reached_radius_not_success"]) for row in rows]
        ),
    }


def summarize_by_band(rows):
    result = {"all": summarize_outcomes(rows)}
    for band in BANDS:
        result[band] = summarize_outcomes(
            [row for row in rows if str(row["band"]) == band]
        )
    return result


def step_arrays(steps):
    return {
        "probability": np.asarray([row["arrival_probability"] for row in steps],
                                  dtype=np.float64),
        "base_probability": np.asarray([row["base_arrival_probability"] for row in steps],
                                       dtype=np.float64),
        "distance": np.asarray([row["distance_m"] for row in steps], dtype=np.float64),
        "episode_index": np.asarray([row["episode_index"] for row in steps], dtype=np.int64),
        "band": np.asarray([row["band"] for row in steps]),
        "stop": np.asarray([row["stop"] for row in steps], dtype=np.int64),
        "goal_visible": np.asarray([row["goal_visible"] for row in steps], dtype=np.int64),
    }


def hazard_from_steps(steps, threshold=0.5):
    arrays = step_arrays(steps)
    return per_decision_hazard(
        arrays["probability"], arrays["distance"], arrays["episode_index"],
        threshold=threshold, stopped=arrays["stop"].astype(bool),
    )


def stop_event_summary(stops):
    distances = np.asarray([row["distance_m"] for row in stops], dtype=np.float64)
    probabilities = np.asarray(
        [row["arrival_probability"] for row in stops], dtype=np.float64
    )
    visible = np.asarray([row["goal_visible"] for row in stops], dtype=np.int64)
    if len(distances) == 0:
        return {
            "stops": 0, "premature_gt20": 0, "far_gt40": 0, "far_gt100": 0,
            "median_distance_m": float("nan"),
            "mean_arrival_probability": float("nan"),
            "goal_visible_rate": float("nan"),
        }
    return {
        "stops": int(len(distances)),
        "premature_gt20": int((distances > STOP_RADIUS_M).sum()),
        "far_gt40": int((distances > FAR_RADIUS_M).sum()),
        "far_gt100": int((distances > 100.0).sum()),
        "median_distance_m": float(np.median(distances)),
        "mean_arrival_probability": float(probabilities.mean()),
        "goal_visible_rate": float(visible.mean()),
    }


def risk_rows(steps_by_rule, threshold_map=None):
    """Flatten per-decision hazard into rows for every rule and band."""
    rows = []
    for name, steps in steps_by_rule.items():
        threshold = (
            threshold_map.get(name, ARRIVAL_STOP_THRESHOLD)
            if threshold_map else ARRIVAL_STOP_THRESHOLD
        )
        for band in ("all",) + BANDS:
            subset = (
                steps if band == "all"
                else [row for row in steps if str(row["band"]) == band]
            )
            if not subset:
                continue
            hazard = hazard_from_steps(subset, threshold=threshold)
            rows.append({"rule": name, "band": band, **hazard})
    return rows


def verify_b0_replay(episodes, reference_csv=STAGE2A_EVAL / "per_episode.csv"):
    """Compare a fresh B0 closed-loop run with the Stage-2a M0 reference.

    Outcome fields (success / OSR / premature / timeout) must match exactly;
    that is the reproducibility criterion. Numeric trajectory diagnostics
    (`steps`, SPL, nDTW, min distance) are reported separately: CUDA kernels
    are not bit-reproducible, so a step count or a 1e-5-level distance can
    differ on an episode whose outcome is identical. Only outcome mismatches
    should fail the check.
    """
    with Path(reference_csv).open(encoding="utf-8") as handle:
        reference = {
            int(row["episode_index"]): row
            for row in csv.DictReader(handle)
            if row["mode"] == "M0"
        }
    outcome_mismatches = []
    numeric_mismatches = []
    for episode in episodes:
        index = int(episode["episode_index"])
        if index not in reference:
            outcome_mismatches.append(
                {"episode_index": index, "error": "missing_reference"}
            )
            continue
        expected = reference[index]
        for key in ("success", "osr", "premature_stop", "max_steps_termination"):
            if int(episode[key]) != int(expected[key]):
                outcome_mismatches.append({
                    "episode_index": index, "field": key,
                    "fresh": int(episode[key]), "reference": int(expected[key]),
                })
        if int(episode["steps"]) != int(expected["steps"]):
            numeric_mismatches.append({
                "episode_index": index, "field": "steps",
                "fresh": int(episode["steps"]),
                "reference": int(expected["steps"]),
            })
        for key in ("spl", "ndtw", "min_distance_m"):
            if abs(float(episode[key]) - float(expected[key])) > 1e-4:
                numeric_mismatches.append({
                    "episode_index": index, "field": key,
                    "fresh": float(episode[key]),
                    "reference": float(expected[key]),
                })
    return {
        "reference": str(reference_csv),
        "episodes": int(len(episodes)),
        "outcome_mismatch_count": int(len(outcome_mismatches)),
        "outcome_mismatches": outcome_mismatches[:50],
        "numeric_mismatch_count": int(len(numeric_mismatches)),
        "numeric_mismatches": numeric_mismatches[:50],
        # Backward-compatible total; callers should gate on outcomes.
        "mismatch_count": int(len(outcome_mismatches) + len(numeric_mismatches)),
    }


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate_probabilities(calibrator, features, base_logits, device,
                            batch_size=2048):
    """Apply a calibrator to arrays; used for offline Dev Val metrics."""
    output = []
    calibrator.eval()
    with torch.inference_mode():
        for start in range(0, len(features), int(batch_size)):
            end = min(start + int(batch_size), len(features))
            feature_batch = torch.from_numpy(
                np.asarray(features[start:end], dtype=np.float32)
            ).to(device)
            logit_batch = torch.from_numpy(
                np.asarray(base_logits[start:end], dtype=np.float32)
            ).to(device)
            calibrated = calibrator(feature_batch, logit_batch)
            output.append(torch.sigmoid(calibrated).cpu().numpy())
    return np.concatenate(output)


def head_probabilities(head, features, device, batch_size=2048):
    output = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(features), int(batch_size)):
            end = min(start + int(batch_size), len(features))
            batch = torch.from_numpy(
                np.asarray(features[start:end], dtype=np.float32)
            ).to(device)
            output.append(torch.sigmoid(head.net(batch).squeeze(-1)).cpu().numpy())
    return np.concatenate(output)


def head_logits(head, features, device, batch_size=2048):
    output = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(features), int(batch_size)):
            end = min(start + int(batch_size), len(features))
            batch = torch.from_numpy(
                np.asarray(features[start:end], dtype=np.float32)
            ).to(device)
            output.append(head.net(batch).squeeze(-1).cpu().numpy())
    return np.concatenate(output)


def paired_by_band(rows_by_variant, base, other):
    result = {
        "all": paired_comparison(rows_by_variant, base=base, other=other),
    }
    for band in BANDS:
        subset = {
            name: [row for row in rows if str(row["band"]) == band]
            for name, rows in rows_by_variant.items()
        }
        result[band] = paired_comparison(subset, base=base, other=other)
    return result


def device_summary(device):
    return {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
