"""Stage-2f: offline analysis of Goal-Reacquisition observables.

Uses the feature table from `stage2f_collect.py`. True goal visibility, true
goal projection, true heading error and the oracle action are used only as
offline labels for evaluation; they are never policy inputs.

Outputs:
  * visible vs invisible movement/confidence/entropy tables by band;
  * per-feature AUC (episode-bootstrap 95% CI) for invisibility and
    unreliable-localization labels;
  * trigger precision/recall for current-only vs history-augmented rules;
  * consecutive wrong-turn diagnostics and whether the trigger preceded them;
  * an offline counterfactual of the candidate recovery action.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import FORWARD, LEFT, RIGHT  # noqa: E402
from v83_spatial_stage2.stage2d_common import write_csv, write_json  # noqa: E402


BANDS = ("all", "short", "medium", "long")
ACTION_NAMES = {FORWARD: "F", LEFT: "L", RIGHT: "R"}
STOP_THRESHOLD = 0.7
SIDE_MARGIN = 0.05
UNRELIABLE_PEAK_PX = 42.0
UNRELIABLE_PEAK_PX_LOOSE = 56.0


def read_features(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "step", "distance_m", "goal_visible", "goal_px", "goal_py",
        "goal_half_size_px", "heading_error_deg", "oracle_move_action",
        "stop_ready", "executed_action", "logged_stop",
        "logged_arrival_probability", "owner_move_action",
        "owner_arrival_probability", "previous_action",
        "coherent_peak_error_px", "baseline_peak_error_px",
        "m0_arrival_probability", "m0_move_top1", "m0_move_margin",
        "m0_move_action", "temporal_valid", "temporal_consistency",
        "innovation_mean_abs", "innovation_max_abs",
        "coherent_top1", "coherent_entropy", "coherent_centroid_x",
        "baseline_top1", "baseline_entropy", "baseline_centroid_x",
        "heatmap_max_top1", "heatmap_max_entropy",
        "heatmap_max_centroid_x", "heatmap_max_peak_x",
        "sim_max", "sim_std", "sim_top5_mean",
    }
    for row in rows:
        for key in numeric:
            if key in row:
                try:
                    row[key] = float(row[key])
                except ValueError:
                    row[key] = float("nan")
        row["step"] = int(row["step"])
        row["episode_index"] = int(row["episode_index"])
    return rows


def band_rows(rows, band):
    if band == "all":
        return rows
    return [row for row in rows if str(row["band"]) == band]


def auc_rank(scores, labels):
    """Rank-based AUC, ties contribute 0.5."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positive = labels == 1
    negative = labels == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    index = 0
    while index < len(sorted_scores):
        end = index + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        if end - index > 1:
            average = (index + 1 + end) / 2.0
            ranks[order[index:end]] = average
        index = end
    return float(
        (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def bootstrap_auc(rows, score_key, label_key, n_boot=200, seed=20261019):
    values = np.asarray([row[score_key] for row in rows], dtype=np.float64)
    labels = np.asarray([row[label_key] for row in rows], dtype=np.int64)
    valid = np.isfinite(values)
    values, labels = values[valid], labels[valid]
    episodes = np.asarray([row["episode_index"] for row in rows])[valid]
    if len(values) == 0:
        return {"auc": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    point = auc_rank(values, labels)
    unique = np.unique(episodes)
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(n_boot)):
        sample = rng.choice(unique, size=len(unique), replace=True)
        mask = np.isin(episodes, sample)
        if len(np.unique(labels[mask])) < 2:
            continue
        draws.append(auc_rank(values[mask], labels[mask]))
    if draws:
        low, high = np.percentile(draws, [2.5, 97.5])
    else:
        low = high = float("nan")
    return {
        "auc": float(point),
        "ci95": [float(low), float(high)],
        "n": int(len(values)),
        "positives": int(labels.sum()),
    }


def add_labels_and_history(rows, config):
    """Sort each episode, add observable history and offline labels."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["trajectory"], row["episode_index"])].append(row)
    output = []
    for (_, _), sequence in grouped.items():
        sequence.sort(key=lambda row: row["step"])
        last_high_step = -1
        last_high_side = "unknown"
        last_high_centroid = float("nan")
        low_run = 0
        high_run = 0
        wrong_run = 0
        wrong_run_owner = 0
        worsen_run = 0
        for row in sequence:
            row["move_state"] = int(
                row["m0_arrival_probability"] < STOP_THRESHOLD
            )
            row["move_state_owner"] = int(
                row["owner_arrival_probability"] < STOP_THRESHOLD
            )
            high = (
                row["heatmap_max_top1"] >= config["high_top1"]
                and row["heatmap_max_entropy"] <= config["high_entropy"]
            )
            low = (
                row["heatmap_max_top1"] < config["low_top1"]
                or row["heatmap_max_entropy"] > config["low_entropy"]
            )
            row["high_conf"] = int(high)
            row["low_conf"] = int(low)
            if high:
                high_run += 1
                last_high_step = row["step"]
                last_high_centroid = row["baseline_centroid_x"]
                if row["baseline_centroid_x"] < -SIDE_MARGIN:
                    last_high_side = "left"
                elif row["baseline_centroid_x"] > SIDE_MARGIN:
                    last_high_side = "right"
                else:
                    last_high_side = "center"
                low_run = 0
            else:
                high_run = 0
                low_run = low_run + 1 if low else 0
            row["low_run"] = int(low_run)
            row["high_run"] = int(high_run)
            row["last_high_side"] = last_high_side
            row["last_high_centroid_x"] = float(last_high_centroid)
            row["steps_since_high_conf"] = (
                row["step"] - last_high_step if last_high_step >= 0 else 999
            )
            row["temporal_inconsistent"] = int(
                row["temporal_valid"] > 0
                and math.isfinite(row["temporal_consistency"])
                and row["temporal_consistency"] < config["consistency"]
            )
            row["trigger_current"] = int(low and row["move_state"])
            row["trigger_history"] = int(
                low_run >= config["trigger_k"] and row["move_state"]
            )
            row["trigger_temporal"] = int(
                row["temporal_inconsistent"] and low_run >= 1 and row["move_state"]
            )
            row["trigger_combined"] = int(
                (row["trigger_history"] or row["trigger_temporal"])
                and row["move_state"]
            )

            # Offline labels (never policy inputs).
            row["offline_invisible"] = int(row["goal_visible"] < 0.5)
            row["offline_visible"] = 1 - row["offline_invisible"]
            row["offline_unreliable_42"] = int(
                row["offline_invisible"]
                or (
                    row["goal_visible"] > 0.5
                    and row["coherent_peak_error_px"] > UNRELIABLE_PEAK_PX
                )
            )
            row["offline_unreliable_56"] = int(
                row["offline_invisible"]
                or (
                    row["goal_visible"] > 0.5
                    and row["coherent_peak_error_px"] > UNRELIABLE_PEAK_PX_LOOSE
                )
            )
            row["offline_reliable_42"] = 1 - row["offline_unreliable_42"]
            row["move_error"] = int(
                row["move_state"]
                and int(row["m0_move_action"]) != int(row["oracle_move_action"])
            )
            row["move_error_owner"] = int(
                row["move_state_owner"]
                and int(row["owner_move_action"]) != int(row["oracle_move_action"])
            )
            wrong_run = wrong_run + 1 if row["move_error"] else 0
            wrong_run_owner = (
                wrong_run_owner + 1 if row["move_error_owner"] else 0
            )
            row["wrong_run"] = int(wrong_run)
            row["wrong_run_owner"] = int(wrong_run_owner)
            row["wrong_run_ge2"] = int(wrong_run_owner >= 2)
            row["wrong_run_ge3"] = int(wrong_run_owner >= 3)

            # Offline heading-worsening run on executed turns.
            action = int(row["executed_action"])
            if action in (LEFT, RIGHT):
                heading = float(row["heading_error_deg"])
                if action == LEFT:
                    after = heading - 15.0
                else:
                    after = heading + 15.0
                after = (after + 180.0) % 360.0 - 180.0
                worsened = abs(after) > abs(heading) + 1e-6
            else:
                worsened = False
            worsen_run = worsen_run + 1 if worsened else 0
            row["heading_worsen_run"] = int(worsen_run)
            output.append(row)
    return output


def movement_label(action, oracle_action):
    if action == oracle_action:
        return "correct"
    if oracle_action in (LEFT, RIGHT) and action == FORWARD:
        return "missed"
    if oracle_action in (LEFT, RIGHT) and action in (LEFT, RIGHT):
        return "wrong"
    if oracle_action == FORWARD and action in (LEFT, RIGHT):
        return "spurious"
    return "other"


def visible_table(rows, trajectory):
    output = []
    for band in BANDS:
        for visible in (1, 0):
            subset = [
                row for row in band_rows(rows, band)
                if row["trajectory"] == trajectory
                and row["move_state_owner"]
                and int(row["goal_visible"]) == visible
            ]
            if not subset:
                continue
            labels = [
                movement_label(
                    int(row["owner_move_action"]),
                    int(row["oracle_move_action"]),
                )
                for row in subset
            ]
            output.append({
                "trajectory": trajectory,
                "band": band,
                "goal_visible": visible,
                "states": len(subset),
                "share_of_move_states": float(
                    len(subset) / max(1, len([
                        row for row in band_rows(rows, band)
                        if row["trajectory"] == trajectory
                        and row["move_state_owner"]
                    ]))
                ),
                "move_error_rate": float(
                    np.mean([row["move_error_owner"] for row in subset])
                ),
                "missed_turn_rate": float(np.mean([
                    label == "missed" for label in labels
                ])),
                "wrong_turn_rate": float(np.mean([
                    label == "wrong" for label in labels
                ])),
                "spurious_turn_rate": float(np.mean([
                    label == "spurious" for label in labels
                ])),
                "wrong_run_ge2_rate": float(
                    np.mean([row["wrong_run_ge2"] for row in subset])
                ),
                "mean_abs_heading_error_deg": float(np.mean([
                    abs(row["heading_error_deg"]) for row in subset
                ])),
                "coherent_top1": float(np.mean([row["coherent_top1"] for row in subset])),
                "coherent_entropy": float(np.mean([row["coherent_entropy"] for row in subset])),
                "heatmap_max_top1": float(np.mean([row["heatmap_max_top1"] for row in subset])),
                "heatmap_max_entropy": float(np.mean([
                    row["heatmap_max_entropy"] for row in subset
                ])),
                "sim_max": float(np.mean([row["sim_max"] for row in subset])),
                "m0_arrival_probability": float(np.mean([
                    row["m0_arrival_probability"] for row in subset
                ])),
                "temporal_consistency": float(np.nanmean([
                    row["temporal_consistency"] for row in subset
                ])),
            })
    return output


FEATURE_DIRECTIONS = {
    "coherent_top1": +1.0,
    "coherent_entropy": -1.0,
    "heatmap_max_top1": +1.0,
    "heatmap_max_entropy": -1.0,
    "sim_max": +1.0,
    "sim_std": +1.0,
    "m0_arrival_probability": +1.0,
    "m0_move_margin": +1.0,
    "temporal_consistency": +1.0,
    "temporal_valid": +1.0,
    "innovation_mean_abs": -1.0,
    "baseline_top1": +1.0,
    "baseline_entropy": -1.0,
}


def auc_table(rows, trajectory):
    """AUC for VISIBLE / RELIABLE labels; higher score = more likely.

    Entropy/innovation features are negated so every reported AUC > 0.5 means
    the feature is informative in the expected direction.
    """
    output = []
    for band in BANDS:
        subset = [
            row for row in band_rows(rows, band)
            if row["trajectory"] == trajectory and row["move_state"]
        ]
        for label, label_key in (
            ("visible", "offline_visible"),
            ("reliable_42", "offline_reliable_42"),
        ):
            for feature, direction in FEATURE_DIRECTIONS.items():
                for row in subset:
                    row["_auc_score"] = direction * row[feature]
                result = bootstrap_auc(subset, "_auc_score", label_key)
                output.append({
                    "trajectory": trajectory,
                    "band": band,
                    "label": label,
                    "feature": feature,
                    "direction": direction,
                    **result,
                })
    return output


def trigger_table(rows, trajectory):
    output = []
    for band in BANDS:
        subset = [
            row for row in band_rows(rows, band)
            if row["trajectory"] == trajectory and row["move_state"]
        ]
        for trigger in ("trigger_current", "trigger_history", "trigger_temporal", "trigger_combined"):
            for label in ("offline_invisible", "offline_unreliable_42"):
                predicted = np.asarray([row[trigger] for row in subset], dtype=int)
                actual = np.asarray([row[label] for row in subset], dtype=int)
                tp = int(((predicted == 1) & (actual == 1)).sum())
                fp = int(((predicted == 1) & (actual == 0)).sum())
                fn = int(((predicted == 0) & (actual == 1)).sum())
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-9)
                output.append({
                    "trajectory": trajectory,
                    "band": band,
                    "trigger": trigger,
                    "label": label,
                    "states": len(subset),
                    "trigger_rate": float(predicted.mean()) if len(predicted) else float("nan"),
                    "base_rate": float(actual.mean()) if len(actual) else float("nan"),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                })
    return output


def visibility_action_table(rows, trajectory):
    """Where do invisible states occur: on executed moves or terminal Stops?"""
    output = []
    for band in BANDS:
        subset = [
            row for row in band_rows(rows, band)
            if row["trajectory"] == trajectory
        ]
        invisible = [row for row in subset if row["offline_invisible"]]
        if not invisible:
            continue
        invisible_stop = [row for row in invisible if int(row["logged_stop"])]
        invisible_move = [row for row in invisible if not int(row["logged_stop"])]
        output.append({
            "trajectory": trajectory,
            "band": band,
            "states": len(subset),
            "invisible": len(invisible),
            "invisible_rate": float(len(invisible) / len(subset)),
            "invisible_stop": len(invisible_stop),
            "invisible_move": len(invisible_move),
            "invisible_move_share": float(
                len(invisible_move) / len(invisible)
            ),
            "invisible_stop_share": float(
                len(invisible_stop) / len(invisible)
            ),
        })
    return output


def stop_state_table(rows, trajectory):
    """M0/M1 actual Stop steps: confidence at premature vs correct stops.

    True distance/visibility are evaluation labels here; the recovery policy
    would only see the confidence columns.
    """
    output = []
    for band in BANDS:
        subset = [
            row for row in band_rows(rows, band)
            if row["trajectory"] == trajectory and int(row["logged_stop"]) == 1
        ]
        if not subset:
            continue
        premature = [row for row in subset if row["distance_m"] >= 20.0]
        correct = [row for row in subset if row["distance_m"] < 20.0]
        output.append({
            "trajectory": trajectory,
            "band": band,
            "stops": len(subset),
            "premature": len(premature),
            "premature_rate": float(len(premature) / len(subset)),
            "premature_visible_rate": (
                float(np.mean([row["goal_visible"] for row in premature]))
                if premature else float("nan")
            ),
            "premature_low_conf_rate": (
                float(np.mean([row["low_conf"] for row in premature]))
                if premature else float("nan")
            ),
            "premature_heat_top1_median": (
                float(np.median([row["heatmap_max_top1"] for row in premature]))
                if premature else float("nan")
            ),
            "premature_heat_entropy_median": (
                float(np.median([row["heatmap_max_entropy"] for row in premature]))
                if premature else float("nan")
            ),
            "premature_arrival_p_median": (
                float(np.median([row["m0_arrival_probability"] for row in premature]))
                if premature else float("nan")
            ),
            "correct_heat_top1_median": (
                float(np.median([row["heatmap_max_top1"] for row in correct]))
                if correct else float("nan")
            ),
            "correct_arrival_p_median": (
                float(np.median([row["m0_arrival_probability"] for row in correct]))
                if correct else float("nan")
            ),
        })
    return output


def wrong_run_table(rows, trajectory):
    output = []
    for band in BANDS:
        subset = [
            row for row in band_rows(rows, band)
            if row["trajectory"] == trajectory and row["move_state_owner"]
        ]
        if not subset:
            continue
        start_indices = [
            index for index, row in enumerate(subset)
            if row["wrong_run_owner"] == 1
            and (index == 0 or subset[index - 1]["wrong_run_owner"] == 0)
        ]
        starts = [subset[index] for index in start_indices]
        pre_trigger = sum(
            1 for index in start_indices
            if index > 0 and subset[index - 1]["trigger_combined"]
        )
        output.append({
            "trajectory": trajectory,
            "band": band,
            "move_states": len(subset),
            "wrong_run_ge2_rate": float(
                np.mean([row["wrong_run_ge2"] for row in subset])
            ),
            "wrong_run_ge3_rate": float(
                np.mean([row["wrong_run_ge3"] for row in subset])
            ),
            "run_starts_ge2": len(starts),
            "run_starts_with_low_conf": int(sum(
                row["low_conf"] for row in starts
            )),
            "run_starts_with_trigger": int(sum(
                row["trigger_combined"] for row in starts
            )),
            "run_starts_with_pre_trigger": int(pre_trigger),
            "mean_abs_heading_error_deg": float(np.mean([
                abs(row["heading_error_deg"]) for row in subset
            ])),
        })
    return output


def policy_counterfactual(rows, trajectory, config):
    """Offline recovery action on triggered states (privileged oracle only for scoring)."""
    output = []
    for band in BANDS:
        subset = [
            row for row in band_rows(rows, band)
            if row["trajectory"] == trajectory and row["move_state"]
            and row["trigger_combined"]
        ]
        if not subset:
            continue
        side_actions = []
        scan_actions = []
        for index, row in enumerate(subset):
            side = row["last_high_side"]
            if side == "left":
                side_action = LEFT
            elif side == "right":
                side_action = RIGHT
            else:
                # Deterministic scan: 12 steps left, then 24 right, then 24 left.
                phase = index
                side_action = LEFT if phase < 12 or 36 <= phase < 60 else RIGHT
            scan_action = LEFT if index < 12 or 36 <= index < 60 else RIGHT
            side_actions.append(side_action)
            scan_actions.append(scan_action)

        def score(actions):
            changed = [
                row for row, action in zip(subset, actions)
                if action != int(row["m0_move_action"])
            ]
            beneficial = sum(
                1 for row, action in zip(subset, actions)
                if action == int(row["oracle_move_action"])
                and int(row["m0_move_action"]) != int(row["oracle_move_action"])
            )
            harmful = sum(
                1 for row, action in zip(subset, actions)
                if int(row["m0_move_action"]) == int(row["oracle_move_action"])
                and action != int(row["oracle_move_action"])
            )
            return {
                "trigger_states": len(subset),
                "changed": len(changed),
                "beneficial": int(beneficial),
                "harmful": int(harmful),
                "net": int(beneficial - harmful),
                "turn_rate": float(np.mean([action != FORWARD for action in actions])),
            }

        output.append({
            "trajectory": trajectory,
            "band": band,
            "policy": "side_history",
            **score(side_actions),
        })
        output.append({
            "trajectory": trajectory,
            "band": band,
            "policy": "scan_only",
            **score(scan_actions),
        })
    return output


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--high-top1", type=float, default=0.30)
    parser.add_argument("--high-entropy", type=float, default=0.75)
    parser.add_argument("--low-top1", type=float, default=0.15)
    parser.add_argument("--low-entropy", type=float, default=0.85)
    parser.add_argument("--trigger-k", type=int, default=2)
    parser.add_argument("--consistency", type=float, default=0.55)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    rows = read_features(args.features)
    config = {
        "high_top1": args.high_top1,
        "high_entropy": args.high_entropy,
        "low_top1": args.low_top1,
        "low_entropy": args.low_entropy,
        "trigger_k": args.trigger_k,
        "consistency": args.consistency,
    }
    rows = add_labels_and_history(rows, config)

    visible = []
    aucs = []
    triggers = []
    wrong_runs = []
    stop_states = []
    visibility_actions = []
    counterfactual = []
    for trajectory in ("M0", "M1"):
        visible.extend(visible_table(rows, trajectory))
        aucs.extend(auc_table(rows, trajectory))
        triggers.extend(trigger_table(rows, trajectory))
        wrong_runs.extend(wrong_run_table(rows, trajectory))
        stop_states.extend(stop_state_table(rows, trajectory))
        visibility_actions.extend(visibility_action_table(rows, trajectory))
        counterfactual.extend(policy_counterfactual(rows, trajectory, config))

    write_csv(args.output_dir / "visible_vs_invisible.csv", visible)
    write_csv(args.output_dir / "feature_auc.csv", aucs)
    write_csv(args.output_dir / "trigger_metrics.csv", triggers)
    write_csv(args.output_dir / "wrong_runs.csv", wrong_runs)
    write_csv(args.output_dir / "stop_states.csv", stop_states)
    write_csv(args.output_dir / "visibility_actions.csv", visibility_actions)
    write_csv(args.output_dir / "policy_counterfactual.csv", counterfactual)
    write_json(args.output_dir / "summary.json", {
        "features": str(args.features),
        "config": config,
        "states": len(rows),
        "visible_vs_invisible": visible,
        "feature_auc": aucs,
        "trigger_metrics": triggers,
        "wrong_runs": wrong_runs,
        "stop_states": stop_states,
        "visibility_actions": visibility_actions,
        "policy_counterfactual": counterfactual,
    })

    # Compact tables for the report.
    visible_rows = []
    for item in visible:
        if item["trajectory"] != "M0":
            continue
        visible_rows.append([
            item["band"], "visible" if item["goal_visible"] else "invisible",
            item["states"], f"{100 * item['share_of_move_states']:.1f}%",
            f"{100 * item['move_error_rate']:.1f}%",
            f"{100 * item['missed_turn_rate']:.1f}%",
            f"{100 * item['wrong_turn_rate']:.1f}%",
            f"{100 * item['spurious_turn_rate']:.1f}%",
            f"{100 * item['wrong_run_ge2_rate']:.1f}%",
            f"{item['mean_abs_heading_error_deg']:.1f}",
            f"{item['heatmap_max_top1']:.3f}",
            f"{item['heatmap_max_entropy']:.3f}",
            f"{item['coherent_top1']:.3f}",
            f"{item['temporal_consistency']:.3f}",
        ])
    visible_md = markdown_table(
        ["Band", "Goal", "States", "Share", "Move err", "Missed",
         "Wrong", "Spurious", "Wrong-run>=2", "|Heading|",
         "Heat top1", "Heat ent", "Coh top1", "Temporal cons"],
        visible_rows,
    )

    auc_rows = []
    for item in aucs:
        if item["trajectory"] != "M0":
            continue
        if item["band"] not in ("all", "medium", "long"):
            continue
        if item["label"] != "visible":
            continue
        auc_rows.append([
            item["band"], item["feature"],
            f"{item['auc']:.3f}",
            f"[{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}]",
            item["n"],
        ])
    auc_rows.sort(key=lambda row: (row[0], -float(row[2])))
    auc_md = markdown_table(
        ["Band", "Feature", "AUC(visible)", "95% CI", "N"], auc_rows
    )

    trigger_rows = []
    for item in triggers:
        if item["trajectory"] != "M0":
            continue
        if item["label"] != "offline_invisible":
            continue
        trigger_rows.append([
            item["band"], item["trigger"], f"{100 * item['trigger_rate']:.1f}%",
            f"{100 * item['base_rate']:.1f}%",
            f"{100 * item['precision']:.1f}%",
            f"{100 * item['recall']:.1f}%",
            f"{100 * item['f1']:.1f}%",
        ])
    trigger_md = markdown_table(
        ["Band", "Trigger", "Trigger rate", "Invisible base rate",
         "Precision", "Recall", "F1"],
        trigger_rows,
    )

    wrong_md = markdown_table(
        ["Band", "Move states", "Wrong-run>=2", "Wrong-run>=3",
         "Run starts", "Start low-conf", "Start triggered", "Pre-step triggered",
         "|Heading|"],
        [
            [
                item["band"], item["move_states"],
                f"{100 * item['wrong_run_ge2_rate']:.1f}%",
                f"{100 * item['wrong_run_ge3_rate']:.1f}%",
                item["run_starts_ge2"],
                item["run_starts_with_low_conf"],
                item["run_starts_with_trigger"],
                item["run_starts_with_pre_trigger"],
                f"{item['mean_abs_heading_error_deg']:.1f}",
            ]
            for item in wrong_runs if item["trajectory"] == "M0"
        ],
    )

    vis_action_md = markdown_table(
        ["Band", "States", "Invisible", "Invisible rate", "At Stop",
         "At move", "Movable share"],
        [
            [
                item["band"], item["states"], item["invisible"],
                f"{100 * item['invisible_rate']:.1f}%",
                item["invisible_stop"], item["invisible_move"],
                f"{100 * item['invisible_move_share']:.1f}%",
            ]
            for item in visibility_actions if item["trajectory"] == "M0"
        ],
    )

    stop_md = markdown_table(
        ["Band", "Stops", "Premature", "Prem rate", "Prem visible",
         "Prem low-conf", "Prem heat top1", "Prem heat ent",
         "Prem arrival p", "Correct heat top1", "Correct arrival p"],
        [
            [
                item["band"], item["stops"], item["premature"],
                f"{100 * item['premature_rate']:.1f}%",
                f"{100 * item['premature_visible_rate']:.1f}%"
                if item["premature_visible_rate"] == item["premature_visible_rate"]
                else "n/a",
                f"{100 * item['premature_low_conf_rate']:.1f}%"
                if item["premature_low_conf_rate"] == item["premature_low_conf_rate"]
                else "n/a",
                f"{item['premature_heat_top1_median']:.3f}"
                if item["premature_heat_top1_median"] == item["premature_heat_top1_median"]
                else "n/a",
                f"{item['premature_heat_entropy_median']:.3f}"
                if item["premature_heat_entropy_median"] == item["premature_heat_entropy_median"]
                else "n/a",
                f"{item['premature_arrival_p_median']:.3f}"
                if item["premature_arrival_p_median"] == item["premature_arrival_p_median"]
                else "n/a",
                f"{item['correct_heat_top1_median']:.3f}"
                if item["correct_heat_top1_median"] == item["correct_heat_top1_median"]
                else "n/a",
                f"{item['correct_arrival_p_median']:.3f}"
                if item["correct_arrival_p_median"] == item["correct_arrival_p_median"]
                else "n/a",
            ]
            for item in stop_states if item["trajectory"] == "M0"
        ],
    )

    counter_md = markdown_table(
        ["Band", "Policy", "Triggers", "Changed", "Beneficial", "Harmful",
         "Net", "Turn rate"],
        [
            [
                item["band"], item["policy"], item["trigger_states"],
                item["changed"], item["beneficial"], item["harmful"],
                item["net"], f"{100 * item['turn_rate']:.1f}%",
            ]
            for item in counterfactual if item["trajectory"] == "M0"
        ],
    )

    tables = "\n\n".join([
        "# Stage-2f offline tables",
        "## M0 move states: goal visible vs invisible",
        visible_md,
        "## Feature AUC for goal visibility (higher = better; M0 move states, episode bootstrap)",
        auc_md,
        "## Trigger precision/recall for invisibility",
        trigger_md,
        "## Where invisible states occur (M0)",
        vis_action_md,
        "## Consecutive wrong-turn diagnostics",
        wrong_md,
        "## Confidence at actual Stop steps (M0)",
        stop_md,
        "## Offline recovery counterfactual",
        counter_md,
        "## Thresholds",
        "`" + json.dumps(config) + "`",
    ])
    (args.output_dir / "tables.md").write_text(tables + "\n", encoding="utf-8")
    print(tables, flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
