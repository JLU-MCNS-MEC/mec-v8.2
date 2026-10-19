"""Stage-2e: analyse M0@0.7 vs M1@0.7 Movement / Stop / trajectory losses.

Inputs (all development evidence; Final Test is never read):
  * saved Stage-2d per-episode CSVs for M0@0.7 and M1@0.7;
  * the exact-state replay produced by stage2e_movement_replay.py;
  * optionally the fusion-off M1@0.7 closed-loop run.

The script writes compact tables and a JSON summary. It does not train or
modify checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import FORWARD, LEFT, RIGHT  # noqa: E402
from v83_spatial_stage2.stage2d_common import BANDS, STAGE2_ROOT  # noqa: E402
from v83_spatial_training.spatial_mec import SpatialFusionHead  # noqa: E402


M0_EPISODES = STAGE2_ROOT / "stage2d/r4_20261019T002626Z/threshold_0.7_per_episode.csv"
M1_EPISODES = (
    STAGE2_ROOT / "stage2d/m1_threshold_20261019T013737Z/M1@0.7_per_episode.csv"
)
M1_CKPT = STAGE2_ROOT / "stage2a_tm_20261018T044255Z/Transformer-M/M1/best.pt"
REPLAY_STEPS = STAGE2_ROOT / "stage2e/replay_20261019T015500Z/replay_steps.csv"
POOL_FEATURES = (
    STAGE2_ROOT / "stage2e/replay_20261019T015500Z/fusion_pool_features.npz"
)
DISTANCE_BINS = (
    ("0-20", 0.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-100", 40.0, 100.0),
    ("100-200", 100.0, 200.0),
    ("200+", 200.0, float("inf")),
)
STOP_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_episodes(path, rule):
    rows = read_csv(path)
    for row in rows:
        row["rule"] = rule
    return {int(row["episode_index"]): row for row in rows}


def outcome(row):
    if int(row["success"]):
        return "success"
    if int(row["premature_stop"]):
        return "premature"
    if int(row["max_steps_termination"]):
        return "timeout"
    return "other"


def outcome_partition(row):
    """Exact 6-way partition of each episode (sums to 1)."""
    if int(row["success"]):
        return "success"
    reached = int(row["osr"]) == 1
    if int(row["premature_stop"]):
        return "premature_reached" if reached else "premature_far"
    if int(row["max_steps_termination"]):
        return "timeout_reached" if reached else "timeout_far"
    return "other"


def primary_cause(row):
    """Coarse failure attribution used in the report."""
    if int(row["success"]):
        return "success"
    if int(row["premature_stop"]):
        return "stop_premature"
    if int(row["max_steps_termination"]):
        return "stop_no_stop" if int(row["osr"]) else "movement_never_reached"
    return "other"


def float_or_nan(value):
    try:
        value = float(value)
        return value
    except (TypeError, ValueError):
        return float("nan")


def mean(values):
    values = [value for value in values if not math.isnan(value)]
    return float(np.mean(values)) if values else float("nan")


def rate(rows, key):
    return mean([float(row[key]) for row in rows])


def sign_test_p(pos, neg):
    n = int(pos) + int(neg)
    if n == 0:
        return 1.0
    k = min(int(pos), int(neg))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def paired_ci(diffs, n_boot=10000, seed=20261019):
    values = np.asarray(diffs, dtype=np.float64)
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(values, size=(int(n_boot), values.size), replace=True)
    means = draws.mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def auc_score(scores, labels):
    """AUC with labels 1=positive; ties contribute 0.5."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    total = 0.0
    for value in positive:
        total += float(np.mean((value > negative) + 0.5 * (value == negative)))
    return total / len(positive)


def balanced_accuracy(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    predicted = scores >= 0
    positive = labels == 1
    return float(0.5 * (predicted[positive].mean() + (~predicted[~positive]).mean()))


def bin_name(distance):
    for name, low, high in DISTANCE_BINS:
        if low <= distance < high:
            return name
    return DISTANCE_BINS[-1][0]


# ---------------------------------------------------------------------------
# Episode-level tables
# ---------------------------------------------------------------------------


def band_rows(rows, band):
    if band == "all":
        return list(rows.values())
    return [row for row in rows.values() if str(row["band"]) == band]


def summary_row(name, rows):
    return {
        "method": name,
        "n": len(rows),
        "sr": rate(rows, "success"),
        "osr": rate(rows, "osr"),
        "spl": mean([float(row["spl"]) for row in rows]),
        "ndtw": mean([float(row["ndtw"]) for row in rows]),
        "premature": rate(rows, "premature_stop"),
        "timeout": rate(rows, "max_steps_termination"),
        "reached_no_success": rate(rows, "reached_radius_not_success"),
        "min_distance_m": mean([float(row["min_distance_m"]) for row in rows]),
        "steps": mean([float(row["steps"]) for row in rows]),
    }


def partition_counts(rows):
    counts = Counter(outcome_partition(row) for row in rows)
    total = len(rows)
    return {
        key: {"count": int(counts.get(key, 0)),
              "rate": float(counts.get(key, 0) / total) if total else float("nan")}
        for key in (
            "success", "premature_far", "premature_reached",
            "timeout_far", "timeout_reached", "other",
        )
    }


def transition_table(m0, m1, band):
    rows0 = {i: row for i, row in m0.items()
             if band == "all" or str(row["band"]) == band}
    rows1 = {i: row for i, row in m1.items()
             if band == "all" or str(row["band"]) == band}
    indices = sorted(set(rows0) & set(rows1))
    counts = Counter(
        (outcome_partition(rows0[i]), outcome_partition(rows1[i]))
        for i in indices
    )
    return {
        "band": band,
        "episodes": len(indices),
        "transitions": {
            f"{a}->{b}": int(counts.get((a, b), 0))
            for a, b in product(
                (
                    "success", "premature_far", "premature_reached",
                    "timeout_far", "timeout_reached", "other",
                ),
                repeat=2,
            )
            if counts.get((a, b), 0)
        },
        "m0": {key: int(value) for key, value in
               Counter(outcome_partition(rows0[i]) for i in indices).items()},
        "m1": {key: int(value) for key, value in
               Counter(outcome_partition(rows1[i]) for i in indices).items()},
    }


def lost_gained(m0, m1, band):
    rows0 = {i: row for i, row in m0.items()
             if band == "all" or str(row["band"]) == band}
    rows1 = {i: row for i, row in m1.items()
             if band == "all" or str(row["band"]) == band}
    indices = sorted(set(rows0) & set(rows1))
    lost = [i for i in indices if int(rows0[i]["success"]) and not int(rows1[i]["success"])]
    gained = [i for i in indices if int(rows1[i]["success"]) and not int(rows0[i]["success"])]
    detail = []
    for index in sorted(set(lost) | set(gained)):
        direction = "lost" if index in lost else "gained"
        row0, row1 = rows0[index], rows1[index]
        failing = row1 if direction == "lost" else row0
        detail.append({
            "episode_index": int(index),
            "band": str(row0["band"]),
            "direction": direction,
            "m0_outcome": outcome_partition(row0),
            "m1_outcome": outcome_partition(row1),
            "failing_primary_cause": primary_cause(failing),
            "failing_osr": int(failing["osr"]),
            "failing_min_distance_m": float(failing["min_distance_m"]),
            "failing_steps": int(failing["steps"]),
            "failing_ndtw": float(failing["ndtw"]),
            "other_min_distance_m": float(
                row1["min_distance_m"] if direction == "lost"
                else row0["min_distance_m"]
            ),
        })
    return {
        "band": band,
        "lost": len(lost),
        "gained": len(gained),
        "net": len(gained) - len(lost),
        "lost_primary_causes": dict(Counter(
            primary_cause(rows1[i]) for i in lost
        )),
        "gained_primary_causes": dict(Counter(
            primary_cause(rows0[i]) for i in gained
        )),
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Step-level movement diagnostics
# ---------------------------------------------------------------------------


def movement_stats(rows, action_key, label):
    """Movement error statistics against the privileged 3-way heading oracle."""
    if not rows:
        return {"method": label, "steps": 0}
    oracle = np.asarray([int(row["oracle_move_action"]) for row in rows])
    action = np.asarray([int(row[action_key]) for row in rows])
    turn_needed = oracle != FORWARD
    correct = action == oracle
    missed_turn = turn_needed & (action == FORWARD)
    wrong_turn = turn_needed & ~correct & ~missed_turn
    spurious_turn = (~turn_needed) & (action != FORWARD)
    return {
        "method": label,
        "steps": int(len(rows)),
        "accuracy": float(correct.mean()),
        "missed_turn_rate": float(missed_turn.mean()),
        "wrong_turn_rate": float(wrong_turn.mean()),
        "spurious_turn_rate": float(spurious_turn.mean()),
        "turn_error_rate": float((missed_turn | wrong_turn | spurious_turn).mean()),
        "forward_rate": float((action == FORWARD).mean()),
        "turn_needed_rate": float(turn_needed.mean()),
    }


def movement_tables(replay):
    by_traj = defaultdict(list)
    for row in replay:
        by_traj[row["trajectory"]].append(row)
    result = []
    action_keys = (
        ("m0_action", "M0"),
        ("m1_base_action", "M1 base"),
        ("m1_fused_action", "M1 fused"),
    )
    for trajectory, rows in by_traj.items():
        for band in ("all",) + BANDS:
            band_rows_ = rows if band == "all" else [
                row for row in rows if str(row["band"]) == band
            ]
            state_sets = {
                "all_states": band_rows_,
                # The movement decision was actually executed only here.
                "executed_moves": [
                    row for row in band_rows_ if int(row["executed_move"])
                ],
                # Counterfactual: the model itself would not Stop at this state.
                "m1_would_move": [
                    row for row in band_rows_
                    if float(row["m1_arrival_probability"]) < STOP_THRESHOLD
                ],
            }
            for set_name, subset in state_sets.items():
                for key, label in action_keys:
                    stats = movement_stats(subset, key, label)
                    stats.update({
                        "trajectory": trajectory,
                        "band": band,
                        "state_set": set_name,
                    })
                    result.append(stats)
    return result


def movement_distance_tables(replay):
    """Movement errors by ground-truth distance, pooled over bands."""
    result = []
    action_keys = (
        ("m0_action", "M0"),
        ("m1_base_action", "M1 base"),
        ("m1_fused_action", "M1 fused"),
    )
    for trajectory in ("M0", "M1"):
        rows = [
            row for row in replay
            if row["trajectory"] == trajectory and int(row["executed_move"])
        ]
        for distance_name, low, high in DISTANCE_BINS:
            subset = [
                row for row in rows
                if low <= float(row["distance_m"]) < high
            ]
            for key, label in action_keys:
                stats = movement_stats(subset, key, label)
                stats.update({
                    "trajectory": trajectory,
                    "distance_bin": distance_name,
                })
                result.append(stats)
    return result


def near_goal_tables(replay):
    """Orientation and arrival behaviour inside the 20 m Stop radius."""
    result = []
    for trajectory in ("M0", "M1"):
        rows = [
            row for row in replay
            if row["trajectory"] == trajectory and float(row["distance_m"]) < 20.0
        ]
        for band in ("all",) + BANDS:
            subset = [row for row in rows if band == "all" or str(row["band"]) == band]
            if not subset:
                continue
            moves = [row for row in subset if int(row["executed_move"])]
            visible = [row for row in subset if int(row["goal_visible"])]
            occluded = [row for row in subset if not int(row["goal_visible"])]
            item = {
                "trajectory": trajectory,
                "band": band,
                "near_steps": len(subset),
                "executed_moves": len(moves),
                "goal_visible_rate": float(len(visible) / len(subset)),
                "mean_abs_heading_error_deg": mean([
                    abs(float(row["heading_error_deg"])) for row in subset
                ]),
                "oracle_turn_needed_rate": mean([
                    float(int(row["oracle_move_action"]) != FORWARD)
                    for row in subset
                ]),
                "m0_arrival_p_visible": mean([
                    float(row["m0_arrival_probability"]) for row in visible
                ]),
                "m0_arrival_p_occluded": mean([
                    float(row["m0_arrival_probability"]) for row in occluded
                ]),
                "m1_arrival_p_visible": mean([
                    float(row["m1_arrival_probability"]) for row in visible
                ]),
                "m1_arrival_p_occluded": mean([
                    float(row["m1_arrival_probability"]) for row in occluded
                ]),
            }
            for key, label in (
                ("m0_action", "M0"),
                ("m1_base_action", "M1 base"),
                ("m1_fused_action", "M1 fused"),
            ):
                stats = movement_stats(moves, key, label)
                item[f"{label}_move_accuracy"] = stats.get("accuracy", float("nan"))
                item[f"{label}_forward_rate"] = stats.get("forward_rate", float("nan"))
                item[f"{label}_turn_error_rate"] = stats.get(
                    "turn_error_rate", float("nan")
                )
            result.append(item)
    return result


def map_available_tables(replay):
    """Can the spatial map indicate the correct turn when a turn is needed?"""
    result = []
    for trajectory in ("M0", "M1"):
        rows = [
            row for row in replay
            if row["trajectory"] == trajectory
            and int(row["executed_move"])
            and int(row["oracle_move_action"]) in (LEFT, RIGHT)
            and abs(float(row["heading_error_deg"])) > 15.0
        ]
        for distance_name, low, high in (("all", 0.0, float("inf")),) + DISTANCE_BINS:
            subset = [
                row for row in rows if low <= float(row["distance_m"]) < high
            ]
            if not subset:
                continue
            labels = np.asarray([
                1 if int(row["oracle_move_action"]) == LEFT else 0
                for row in subset
            ])
            scores = {
                "coherent_map": -np.asarray([
                    float(row["coherent_centroid_x"]) for row in subset
                ]),
                "baseline_map": -np.asarray([
                    float(row["base_centroid_x"]) for row in subset
                ]),
                "m1_base_logit": np.asarray([
                    float(row["m1_base_logit_left"])
                    - float(row["m1_base_logit_right"])
                    for row in subset
                ]),
                "m1_fused_logit": np.asarray([
                    float(row["m1_fused_logit_left"])
                    - float(row["m1_fused_logit_right"])
                    for row in subset
                ]),
                "fusion_residual": np.asarray([
                    float(row["m1_res_logit_left"])
                    - float(row["m1_res_logit_right"])
                    for row in subset
                ]),
            }
            for name, score in scores.items():
                result.append({
                    "trajectory": trajectory,
                    "distance_bin": distance_name,
                    "method": name,
                    "steps": len(subset),
                    "left_steps": int(labels.sum()),
                    "right_steps": int(len(labels) - labels.sum()),
                    "auc_turn_direction": auc_score(score, labels),
                    "balanced_accuracy": balanced_accuracy(score, labels),
                })
    return result


def arrival_tables(replay):
    result = []
    for trajectory in ("M0", "M1"):
        rows = [row for row in replay if row["trajectory"] == trajectory]
        for band in ("all",) + BANDS:
            subset = rows if band == "all" else [
                row for row in rows if str(row["band"]) == band
            ]
            if not subset:
                continue
            distances = np.asarray(
                [float(row["distance_m"]) for row in subset]
            )
            labels = (distances < 20.0).astype(np.int64)
            for name, key in (
                ("M0", "m0_arrival_probability"),
                ("M1", "m1_arrival_probability"),
            ):
                probabilities = np.asarray(
                    [float(row[key]) for row in subset]
                )
                near = probabilities[labels == 1]
                far = probabilities[labels == 0]
                result.append({
                    "trajectory": trajectory,
                    "band": band,
                    "method": name,
                    "steps": len(subset),
                    "near_steps": int(len(near)),
                    "near_mean_p": float(near.mean()) if len(near) else float("nan"),
                    "near_stop_rate_0.7": (
                        float((near >= STOP_THRESHOLD).mean())
                        if len(near) else float("nan")
                    ),
                    "far_mean_p": float(far.mean()) if len(far) else float("nan"),
                    "far_stop_rate_0.7": (
                        float((far >= STOP_THRESHOLD).mean())
                        if len(far) else float("nan")
                    ),
                    "auc_near": auc_score(probabilities, labels),
                })
    return result


# ---------------------------------------------------------------------------
# Fusion residual diagnostics
# ---------------------------------------------------------------------------


def fusion_effect_rows(replay):
    result = []
    for trajectory in ("M0", "M1"):
        rows = [row for row in replay if row["trajectory"] == trajectory]
        for band in ("all",) + BANDS:
            subset = [row for row in rows if band == "all" or str(row["band"]) == band]
            for distance_name, low, high in DISTANCE_BINS:
                bin_rows = [
                    row for row in subset
                    if low <= float(row["distance_m"]) < high
                ]
                if not bin_rows:
                    continue
                eligible = [
                    row for row in bin_rows
                    if float(row["m1_arrival_probability"]) < STOP_THRESHOLD
                ]
                result.append(fusion_bin_stats(
                    eligible, trajectory, band, distance_name
                ))
    return result


def fusion_bin_stats(rows, trajectory, band, distance_name):
    if not rows:
        return {
            "trajectory": trajectory, "band": band,
            "distance_bin": distance_name, "steps": 0,
        }
    base = np.asarray([int(row["m1_base_action"]) for row in rows])
    fused = np.asarray([int(row["m1_fused_action"]) for row in rows])
    oracle = np.asarray([int(row["oracle_move_action"]) for row in rows])
    residual = np.asarray([
        [float(row["m1_res_logit_forward"]),
         float(row["m1_res_logit_left"]),
         float(row["m1_res_logit_right"])]
        for row in rows
    ])
    base_logits = np.asarray([
        [float(row["m1_base_logit_forward"]),
         float(row["m1_base_logit_left"]),
         float(row["m1_base_logit_right"])]
        for row in rows
    ])
    changed = base != fused
    base_correct = base == oracle
    fused_correct = fused == oracle
    beneficial = changed & fused_correct & ~base_correct
    harmful = changed & base_correct & ~fused_correct
    neutral = changed & ~beneficial & ~harmful
    residual_norm = np.linalg.norm(residual, axis=1)
    base_norm = np.linalg.norm(base_logits, axis=1)
    return {
        "trajectory": trajectory,
        "band": band,
        "distance_bin": distance_name,
        "steps": int(len(rows)),
        "residual_mean_l2": float(residual_norm.mean()),
        "residual_p95_l2": float(np.percentile(residual_norm, 95)),
        "residual_max_abs": float(np.abs(residual).max()),
        "residual_relative_l2": float(
            residual_norm.mean() / max(base_norm.mean(), 1e-8)
        ),
        "action_change_rate": float(changed.mean()),
        "beneficial_count": int(beneficial.sum()),
        "harmful_count": int(harmful.sum()),
        "neutral_count": int(neutral.sum()),
        "beneficial_per_1000": float(1000.0 * beneficial.mean()),
        "harmful_per_1000": float(1000.0 * harmful.mean()),
        "net_beneficial_per_1000": float(1000.0 * (beneficial.mean() - harmful.mean())),
        "base_accuracy": float(base_correct.mean()),
        "fused_accuracy": float(fused_correct.mean()),
        "base_forward_rate": float((base == FORWARD).mean()),
        "fused_forward_rate": float((fused == FORWARD).mean()),
    }


def direction_rows(replay):
    result = []
    for trajectory in ("M0", "M1"):
        rows = [
            row for row in replay
            if row["trajectory"] == trajectory
            and row["goal_side"] in ("left", "right")
            and float(row["m1_arrival_probability"]) < STOP_THRESHOLD
        ]
        for band in ("all",) + BANDS:
            subset = [row for row in rows if band == "all" or str(row["band"]) == band]
            if not subset:
                continue
            labels = np.asarray(
                [1 if row["goal_side"] == "left" else 0 for row in subset]
            )
            scores = {
                "fusion_residual": np.asarray([
                    float(row["m1_res_logit_left"]) - float(row["m1_res_logit_right"])
                    for row in subset
                ]),
                "m1_base": np.asarray([
                    float(row["m1_base_logit_left"]) - float(row["m1_base_logit_right"])
                    for row in subset
                ]),
                "m1_fused": np.asarray([
                    float(row["m1_fused_logit_left"]) - float(row["m1_fused_logit_right"])
                    for row in subset
                ]),
                "m0": np.asarray([
                    float(row["m0_logit_left"]) - float(row["m0_logit_right"])
                    for row in subset
                ]),
                "coherent_map_centroid": -np.asarray([
                    float(row["coherent_centroid_x"]) for row in subset
                ]),
            }
            for name, score in scores.items():
                result.append({
                    "trajectory": trajectory,
                    "band": band,
                    "method": name,
                    "steps": int(len(subset)),
                    "left_steps": int(labels.sum()),
                    "right_steps": int(len(labels) - labels.sum()),
                    "left_mean_score": float(score[labels == 1].mean()),
                    "right_mean_score": float(score[labels == 0].mean()),
                    "auc_left": auc_score(score, labels),
                    "balanced_accuracy": balanced_accuracy(score, labels),
                })
    return result


def mirror_rows(replay):
    result = []
    for trajectory in ("M0", "M1"):
        rows = [
            row for row in replay
            if row["trajectory"] == trajectory
            and row["goal_side"] in ("left", "right")
            and float(row["m1_arrival_probability"]) < STOP_THRESHOLD
        ]
        for band in ("all",) + BANDS:
            subset = [row for row in rows if band == "all" or str(row["band"]) == band]
            if not subset:
                continue
            residual = np.asarray([
                [float(row["m1_res_logit_forward"]),
                 float(row["m1_res_logit_left"]),
                 float(row["m1_res_logit_right"])]
                for row in subset
            ])
            residual_mirror = np.asarray([
                [float(row["m1_res_mirror_logit_forward"]),
                 float(row["m1_res_mirror_logit_left"]),
                 float(row["m1_res_mirror_logit_right"])]
                for row in subset
            ])
            base_score = np.asarray([
                float(row["m1_base_logit_left"]) - float(row["m1_base_logit_right"])
                for row in subset
            ])
            base_score_mirror = np.asarray([
                float(row["m1_base_mirror_logit_left"])
                - float(row["m1_base_mirror_logit_right"])
                for row in subset
            ])
            m0_score = np.asarray([
                float(row["m0_logit_left"]) - float(row["m0_logit_right"])
                for row in subset
            ])
            m0_score_mirror = np.asarray([
                float(row["m0_mirror_logit_left"]) - float(row["m0_mirror_logit_right"])
                for row in subset
            ])
            fusion_score = residual[:, 1] - residual[:, 2]
            fusion_score_mirror = residual_mirror[:, 1] - residual_mirror[:, 2]
            significant = np.abs(fusion_score) > 1e-3
            flip = (
                float(np.mean(np.sign(fusion_score_mirror[significant])
                              == -np.sign(fusion_score[significant])))
                if significant.any() else float("nan")
            )
            result.append({
                "trajectory": trajectory,
                "band": band,
                "steps": len(subset),
                "residual_invariance_mean_abs": float(
                    np.abs(residual_mirror - residual).mean()
                ),
                "residual_invariance_max_abs": float(
                    np.abs(residual_mirror - residual).max()
                ),
                "fusion_score_mirror_flip_rate": flip,
                "fusion_equivariance_relative_error": float(
                    np.abs(fusion_score_mirror + fusion_score).mean()
                    / max(np.abs(fusion_score).mean(), 1e-9)
                ),
                "base_equivariance_relative_error": float(
                    np.abs(base_score_mirror + base_score).mean()
                    / max(np.abs(base_score).mean(), 1e-9)
                ),
                "m0_equivariance_relative_error": float(
                    np.abs(m0_score_mirror + m0_score).mean()
                    / max(np.abs(m0_score).mean(), 1e-9)
                ),
                "fusion_score_mean_abs": float(np.abs(fusion_score).mean()),
                "base_score_mean_abs": float(np.abs(base_score).mean()),
                "m0_score_mean_abs": float(np.abs(m0_score).mean()),
            })
    return result


# ---------------------------------------------------------------------------
# Pooling and synthetic head probes
# ---------------------------------------------------------------------------


def fit_probe(features, labels, seed=20261019, steps=600, lr=0.05, weight_decay=1e-4):
    torch.manual_seed(int(seed))
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))
    weight = torch.zeros(features.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=lr, weight_decay=weight_decay)
    positive = float(labels.sum())
    negative = float(len(labels) - positive)
    pos_weight = torch.tensor(negative / max(positive, 1.0))
    for _ in range(int(steps)):
        logit = x @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return weight.detach().numpy(), float(bias.detach())


def pooling_probe(path):
    data = np.load(Path(path))
    f1, f2 = data["f1"], data["f2"]
    side = data["side"].astype(np.int64)
    episode = data["episode_index"]
    train_mask = (episode % 2) == 0
    test_mask = ~train_mask
    result = {
        "samples": int(len(side)),
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "left_rate": float(side.mean()),
        "train_episodes": int(len(np.unique(episode[train_mask]))),
        "test_episodes": int(len(np.unique(episode[test_mask]))),
        "note": (
            "Linear probe on frozen SpatialFusionHead conv features; episode "
            "parity split. f1 = AdaptiveAvgPool2d(1), f2 = 2x2 pooling."
        ),
    }
    for name, features in (("pool1x1", f1), ("pool2x2", f2)):
        train_x, test_x = features[train_mask], features[test_mask]
        train_y, test_y = side[train_mask], side[test_mask]
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        weight, bias = fit_probe((train_x - mean) / std, train_y)
        train_score = ((train_x - mean) / std) @ weight + bias
        test_score = ((test_x - mean) / std) @ weight + bias
        result[name] = {
            "features": int(features.shape[1]),
            "train_auc": auc_score(train_score, train_y),
            "test_auc": auc_score(test_score, test_y),
            "test_balanced_accuracy": balanced_accuracy(test_score, test_y),
        }
    return result


def synthetic_head_probe():
    state = torch.load(M1_CKPT, map_location="cpu", weights_only=True)
    head = SpatialFusionHead(2)
    head.load_state_dict({
        key[len("fusion."):]: value
        for key, value in state.items() if key.startswith("fusion.")
    })
    head.eval()
    grid = 24
    yy, xx = torch.meshgrid(
        torch.arange(grid, dtype=torch.float32),
        torch.arange(grid, dtype=torch.float32),
        indexing="ij",
    )
    maps = torch.zeros(2, 2, grid, grid)
    for index, center_x in enumerate((6.0, 17.0)):
        blob = torch.exp(-0.5 * (((xx - center_x) / 1.8) ** 2
                                 + ((yy - 12.0) / 3.0) ** 2))
        maps[index, 0] = blob
        maps[index, 1] = blob
    with torch.inference_mode():
        conv = head.net[:4](maps)
        pooled_1x1 = F.adaptive_avg_pool2d(conv, 1).flatten(1)
        pooled_2x2 = F.adaptive_avg_pool2d(conv, (2, 2)).flatten(1)
        residual = head(maps)
    return {
        "note": (
            "Synthetic left/right Gaussian blobs through the trained M1 fusion "
            "trunk. The conv features are direction-sensitive; global 1x1 "
            "pooling collapses the difference before the linear layer."
        ),
        "conv_feature_map_left_right_max_abs_diff": float(
            (conv[0] - conv[1]).abs().max()
        ),
        "pooled_1x1_left_right_max_abs_diff": float(
            (pooled_1x1[0] - pooled_1x1[1]).abs().max()
        ),
        "pooled_2x2_left_right_max_abs_diff": float(
            (pooled_2x2[0] - pooled_2x2[1]).abs().max()
        ),
        "residual_left": [float(v) for v in residual[0].tolist()],
        "residual_right": [float(v) for v in residual[1].tolist()],
        "residual_left_right_max_abs_diff": float(
            (residual[0] - residual[1]).abs().max()
        ),
        "residual_left_lr_score": float(residual[0, 1] - residual[0, 2]),
        "residual_right_lr_score": float(residual[1, 1] - residual[1, 2]),
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def fmt(value, digits=2, percent=False):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if percent:
        return f"{100.0 * value:.{digits}f}%"
    return f"{value:.{digits}f}"


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m0-episodes", type=Path, default=M0_EPISODES)
    parser.add_argument("--m1-episodes", type=Path, default=M1_EPISODES)
    parser.add_argument("--replay-steps", type=Path, default=REPLAY_STEPS)
    parser.add_argument("--pool-features", type=Path, default=POOL_FEATURES)
    parser.add_argument("--no-fusion-episodes", type=Path, default=None)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    m0 = read_episodes(args.m0_episodes, "M0@0.7")
    m1 = read_episodes(args.m1_episodes, "M1@0.7")
    replay = read_csv(args.replay_steps)
    print(f"M0 episodes={len(m0)} M1 episodes={len(m1)} replay={len(replay)}",
          flush=True)

    variants = {"M0@0.7": m0, "M1@0.7": m1}
    if args.no_fusion_episodes is not None:
        variants["M1-no-fusion@0.7"] = read_episodes(
            args.no_fusion_episodes, "M1-no-fusion@0.7"
        )

    # 1. Band summary.
    band_summary = []
    summary_json = {}
    for name, rows in variants.items():
        summary_json[name] = {}
        for band in ("all",) + BANDS:
            subset = band_rows(rows, band)
            item = summary_row(name, subset)
            item["band"] = band
            band_summary.append(item)
            summary_json[name][band] = item

    # 2. Exact outcome partition and primary cause.
    partitions = {}
    for name, rows in variants.items():
        partitions[name] = {}
        for band in ("all",) + BANDS:
            subset = band_rows(rows, band)
            partitions[name][band] = {
                "partition": partition_counts(subset),
                "primary_cause": dict(Counter(primary_cause(row) for row in subset)),
            }

    # 3. Paired transitions and lost/gained decomposition.
    transition = {band: transition_table(m0, m1, band)
                  for band in ("all",) + BANDS}
    movements = {band: lost_gained(m0, m1, band)
                 for band in ("all",) + BANDS}

    # 4. Movement / arrival / fusion / direction diagnostics.
    movement = movement_tables(replay)
    movement_distance = movement_distance_tables(replay)
    near_goal = near_goal_tables(replay)
    map_available = map_available_tables(replay)
    arrival = arrival_tables(replay)
    fusion = fusion_effect_rows(replay)
    direction = direction_rows(replay)
    mirror = mirror_rows(replay)
    pooling = pooling_probe(args.pool_features)
    synthetic = synthetic_head_probe()

    # 5. Fusion-off closed-loop paired comparison.
    no_fusion = None
    no_fusion_transition = None
    no_fusion_lost_gained = None
    if "M1-no-fusion@0.7" in variants:
        from v83_spatial_training.eval_closedloop_spatial import paired_comparison
        no_fusion_rows = variants["M1-no-fusion@0.7"]
        no_fusion = {}
        for band in ("all",) + BANDS:
            rows_by_variant = {
                "M1@0.7": band_rows(m1, band),
                "M1-no-fusion@0.7": band_rows(no_fusion_rows, band),
            }
            no_fusion[band] = paired_comparison(
                rows_by_variant, base="M1@0.7", other="M1-no-fusion@0.7"
            )
        no_fusion_transition = {
            band: transition_table(m1, no_fusion_rows, band)
            for band in ("all",) + BANDS
        }
        no_fusion_lost_gained = {
            band: lost_gained(m1, no_fusion_rows, band)
            for band in ("all",) + BANDS
        }

    payload = {
        "variants": list(variants),
        "band_summary": summary_json,
        "partitions": partitions,
        "transitions": transition,
        "lost_gained": movements,
        "movement": movement,
        "movement_distance": movement_distance,
        "near_goal": near_goal,
        "map_available": map_available,
        "arrival": arrival,
        "fusion_effects": fusion,
        "direction": direction,
        "mirror": mirror,
        "pooling_probe": pooling,
        "synthetic_head_probe": synthetic,
        "no_fusion_paired": no_fusion,
        "no_fusion_transition": no_fusion_transition,
        "no_fusion_lost_gained": no_fusion_lost_gained,
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Flat CSVs for downstream use.
    def write_rows(name, rows):
        if not rows:
            return
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_rows("band_summary.csv", band_summary)
    write_rows("movement_stats.csv", movement)
    write_rows("movement_distance_stats.csv", movement_distance)
    write_rows("near_goal_stats.csv", near_goal)
    write_rows("map_available_stats.csv", map_available)
    write_rows("arrival_stats.csv", arrival)
    write_rows("fusion_effects.csv", fusion)
    write_rows("direction_stats.csv", direction)
    write_rows("mirror_stats.csv", mirror)

    # 6. Compact report tables.
    table_headers = [
        "Band", "Method", "SR", "OSR", "SPL", "nDTW", "Premature",
        "Timeout", "Reach-no-success", "Min dist", "Steps",
    ]
    table_rows = []
    for item in band_summary:
        table_rows.append([
            item["band"], item["method"],
            fmt(item["sr"], percent=True), fmt(item["osr"], percent=True),
            fmt(item["spl"], 3), fmt(item["ndtw"], 3),
            fmt(item["premature"], percent=True),
            fmt(item["timeout"], percent=True),
            fmt(item["reached_no_success"], percent=True),
            fmt(item["min_distance_m"], 1), fmt(item["steps"], 1),
        ])
    compact = markdown_table(table_headers, table_rows)

    partition_headers = [
        "Band", "Method", "Success", "Premature-far", "Premature-reached",
        "Timeout-far", "Timeout-reached", "Other",
    ]
    partition_rows = []
    for name in variants:
        for band in ("all",) + BANDS:
            part = partitions[name][band]["partition"]
            partition_rows.append([
                band, name,
                *[fmt(part[key]["rate"], percent=True) for key in (
                    "success", "premature_far", "premature_reached",
                    "timeout_far", "timeout_reached", "other",
                )],
            ])
    partition_md = markdown_table(partition_headers, partition_rows)

    movement_headers = [
        "Trajectory", "Band", "State set", "Method", "Steps", "Move acc",
        "Missed turn", "Wrong turn", "Spurious turn", "Turn error",
    ]
    movement_rows = []
    for item in movement:
        if item["state_set"] != "executed_moves" or item.get("steps", 0) == 0:
            continue
        movement_rows.append([
            item["trajectory"], item["band"], item["state_set"], item["method"],
            item["steps"],
            fmt(item["accuracy"], percent=True),
            fmt(item["missed_turn_rate"], percent=True),
            fmt(item["wrong_turn_rate"], percent=True),
            fmt(item["spurious_turn_rate"], percent=True),
            fmt(item["turn_error_rate"], percent=True),
        ])
    movement_md = markdown_table(movement_headers, movement_rows)

    movement_distance_headers = [
        "Trajectory", "Distance", "Method", "Steps", "Move acc",
        "Missed turn", "Wrong turn", "Spurious turn", "Turn error",
    ]
    movement_distance_rows = []
    for item in movement_distance:
        if item.get("steps", 0) == 0:
            continue
        movement_distance_rows.append([
            item["trajectory"], item["distance_bin"], item["method"],
            item["steps"], fmt(item["accuracy"], percent=True),
            fmt(item["missed_turn_rate"], percent=True),
            fmt(item["wrong_turn_rate"], percent=True),
            fmt(item["spurious_turn_rate"], percent=True),
            fmt(item["turn_error_rate"], percent=True),
        ])
    movement_distance_md = markdown_table(
        movement_distance_headers, movement_distance_rows
    )

    near_headers = [
        "Trajectory", "Band", "Near steps", "Moves", "Goal visible",
        "|Heading err|", "Oracle needs turn", "M0 acc", "M1 base acc",
        "M1 fused acc", "M0 p|vis", "M1 p|vis", "M0 p|occ", "M1 p|occ",
    ]
    near_md_rows = []
    for item in near_goal:
        near_md_rows.append([
            item["trajectory"], item["band"], item["near_steps"],
            item["executed_moves"], fmt(item["goal_visible_rate"], percent=True),
            fmt(item["mean_abs_heading_error_deg"], 1),
            fmt(item["oracle_turn_needed_rate"], percent=True),
            fmt(item["M0_move_accuracy"], percent=True),
            fmt(item["M1 base_move_accuracy"], percent=True),
            fmt(item["M1 fused_move_accuracy"], percent=True),
            fmt(item["m0_arrival_p_visible"], 3),
            fmt(item["m1_arrival_p_visible"], 3),
            fmt(item["m0_arrival_p_occluded"], 3),
            fmt(item["m1_arrival_p_occluded"], 3),
        ])
    near_md = markdown_table(near_headers, near_md_rows)

    map_headers = [
        "Trajectory", "Distance", "Score", "Steps", "AUC(turn direction)",
        "Balanced acc",
    ]
    map_md_rows = []
    for item in map_available:
        map_md_rows.append([
            item["trajectory"], item["distance_bin"], item["method"],
            item["steps"], fmt(item["auc_turn_direction"], 3),
            fmt(item["balanced_accuracy"], 3),
        ])
    map_md = markdown_table(map_headers, map_md_rows)

    fusion_headers = [
        "Trajectory", "Band", "Distance", "Steps", "Residual L2",
        "Rel. L2", "Changed", "Beneficial/1k", "Harmful/1k", "Net/1k",
        "Base acc", "Fused acc",
    ]
    fusion_rows = []
    for item in fusion:
        if item.get("steps", 0) == 0:
            continue
        if item["band"] == "all":
            fusion_rows.append([
                item["trajectory"], item["band"], item["distance_bin"],
                item["steps"], fmt(item["residual_mean_l2"], 3),
                fmt(item["residual_relative_l2"], 4),
                fmt(item["action_change_rate"], percent=True),
                fmt(item["beneficial_per_1000"], 2),
                fmt(item["harmful_per_1000"], 2),
                fmt(item["net_beneficial_per_1000"], 2),
                fmt(item["base_accuracy"], percent=True),
                fmt(item["fused_accuracy"], percent=True),
            ])
    fusion_md = markdown_table(fusion_headers, fusion_rows)

    direction_headers = [
        "Trajectory", "Band", "Score", "Steps", "Left mean", "Right mean",
        "AUC(left)", "Balanced acc",
    ]
    direction_md_rows = []
    for item in direction:
        if item["band"] != "all":
            continue
        direction_md_rows.append([
            item["trajectory"], item["band"], item["method"], item["steps"],
            fmt(item["left_mean_score"], 3), fmt(item["right_mean_score"], 3),
            fmt(item["auc_left"], 3), fmt(item["balanced_accuracy"], 3),
        ])
    direction_md = markdown_table(direction_headers, direction_md_rows)

    mirror_headers = [
        "Trajectory", "Band", "Steps", "Residual invariance (mean/max)",
        "Fusion mirror flip", "Fusion equiv. rel. err",
        "Base equiv. rel. err", "M0 equiv. rel. err",
        "|fusion L-R|", "|base L-R|",
    ]
    mirror_md_rows = []
    for item in mirror:
        if item["band"] != "all":
            continue
        mirror_md_rows.append([
            item["trajectory"], item["band"], item["steps"],
            f"{item['residual_invariance_mean_abs']:.2e} / "
            f"{item['residual_invariance_max_abs']:.2e}",
            fmt(item["fusion_score_mirror_flip_rate"], percent=True),
            fmt(item["fusion_equivariance_relative_error"], 3),
            fmt(item["base_equivariance_relative_error"], 3),
            fmt(item["m0_equivariance_relative_error"], 3),
            fmt(item["fusion_score_mean_abs"], 4),
            fmt(item["base_score_mean_abs"], 3),
        ])
    mirror_md = markdown_table(mirror_headers, mirror_md_rows)

    pooling_md = markdown_table(
        ["Features", "Dim", "Train AUC", "Test AUC", "Test bal. acc"],
        [
            [name, pooling[name]["features"],
             fmt(pooling[name]["train_auc"], 3),
             fmt(pooling[name]["test_auc"], 3),
             fmt(pooling[name]["test_balanced_accuracy"], 3)]
            for name in ("pool1x1", "pool2x2")
        ],
    )
    synthetic_md = markdown_table(
        ["Probe", "Left/right max abs difference"],
        [
            ["Conv feature map (16x24x24)",
             f"{synthetic['conv_feature_map_left_right_max_abs_diff']:.4f}"],
            ["Pooled 1x1 (16)",
             f"{synthetic['pooled_1x1_left_right_max_abs_diff']:.3e}"],
            ["Pooled 2x2 (64)",
             f"{synthetic['pooled_2x2_left_right_max_abs_diff']:.4f}"],
            ["Trained fusion residual",
             f"{synthetic['residual_left_right_max_abs_diff']:.3e}"],
        ],
    )

    lost_gained_md = markdown_table(
        ["Band", "Lost", "Gained", "Net", "Lost causes", "Gained causes"],
        [
            [band, movements[band]["lost"], movements[band]["gained"],
             movements[band]["net"],
             json.dumps(movements[band]["lost_primary_causes"]),
             json.dumps(movements[band]["gained_primary_causes"])]
            for band in ("all",) + BANDS
        ],
    )

    transition_md = markdown_table(
        ["Band", "M0 partition", "M1 partition", "Transitions"],
        [
            [band, json.dumps(transition[band]["m0"]),
             json.dumps(transition[band]["m1"]),
             json.dumps(transition[band]["transitions"])]
            for band in ("all",) + BANDS
        ],
    )

    no_fusion_md = "not available"
    if no_fusion is not None:
        no_fusion_rows = []
        for band in ("all",) + BANDS:
            item = no_fusion[band]
            no_fusion_rows.append([
                band,
                fmt(summary_json["M1@0.7"][band]["sr"], percent=True),
                fmt(summary_json["M1-no-fusion@0.7"][band]["sr"], percent=True),
                f"{item['success']['delta_rate'] * 100:+.2f} pp "
                f"({item['success']['other_better']}W/"
                f"{item['success']['base_better']}L)",
                f"{item['spl']['delta']:+.4f}",
                f"{item['ndtw']['delta']:+.4f}",
                f"{item['premature_stop']['delta_rate'] * 100:+.2f} pp",
                f"{item['max_steps_termination']['delta_rate'] * 100:+.2f} pp",
            ])
        no_fusion_md = markdown_table(
            ["Band", "M1 SR", "No-fusion SR", "SR delta (W/L)",
             "SPL delta", "nDTW delta", "Premature delta", "Timeout delta"],
            no_fusion_rows,
        )
        no_fusion_transition_md = markdown_table(
            ["Band", "M1 partition", "No-fusion partition", "Transitions"],
            [
                [band, json.dumps(no_fusion_transition[band]["m0"]),
                 json.dumps(no_fusion_transition[band]["m1"]),
                 json.dumps(no_fusion_transition[band]["transitions"])]
                for band in ("all",) + BANDS
            ],
        )
    else:
        no_fusion_transition_md = "not available"

    tables = "\n\n".join([
        "# Stage-2e generated tables",
        "## Compact band summary",
        compact,
        "## Exact per-episode outcome partition",
        partition_md,
        "## Paired outcome transitions",
        transition_md,
        "## Lost / gained success decomposition",
        lost_gained_md,
        "## Fusion-off ablation (M1 checkpoint, residual forced to zero)",
        no_fusion_md,
        "## Fusion-off outcome transitions",
        no_fusion_transition_md,
        "## Movement errors (executed move steps, privileged heading oracle)",
        movement_md,
        "## Movement errors by ground-truth distance",
        movement_distance_md,
        "## Near-goal orientation and arrival behaviour (<20 m)",
        near_md,
        "## Directional evidence available when a turn is needed",
        map_md,
        "## Fusion residual effects (M1 arrival p < 0.7)",
        fusion_md,
        "## Fusion residual directional response",
        direction_md,
        "## Mirror probes",
        mirror_md,
        "## Pooling probe (side decodability)",
        pooling_md,
        "## Synthetic head probe",
        synthetic_md,
        "## Notes",
        (
            "* The 3-way oracle is privileged (true goal bearing) and is used "
            "only for off-policy diagnostics.\n"
            "* Fusion effects use the M1 checkpoint's own base movement logits "
            "as the reference; M1 base weights are not M0 weights.\n"
            "* Movement decisions at states where the M1 arrival probability "
            "is >= 0.7 are excluded from the fusion-effect table because the "
            "Stop decision would override them."
        ),
    ])
    (args.output_dir / "tables.md").write_text(tables + "\n", encoding="utf-8")
    print(tables, flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
