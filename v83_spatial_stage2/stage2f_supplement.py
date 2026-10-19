"""Stage-2f supplementary Goal-Reacquisition analyses (eval only).

Reads the existing Stage-2f feature table and closed-loop smoke step logs.
True goal visibility, true distance, heading error and the oracle action are
used **only as offline labels**; this script does not run or change any
controller.

Outputs:
  * invisible movement states by true distance, move error and executed vs
    oracle action;
  * transient (re-visible within k steps) vs sustained invisibility;
  * turn-induced temporary loss (previous state visible, previous action a
    turn, current state invisible);
  * recency-gated trigger precision/recall for near-target invisibility using
    only observable confidence/history;
  * visibility AUC stratified by true distance band (offline diagnostic);
  * trigger-distance distribution of the closed-loop smoke.

Usage:
  python v83_spatial_stage2/stage2f_supplement.py \
    --features runs/.../collect_.../features.csv \
    --smoke-dir runs/.../smoke_... \
    --output-dir runs/.../supplement_...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import FORWARD, LEFT, RIGHT  # noqa: E402
from v83_spatial_stage2.stage2d_common import write_json  # noqa: E402

STOP_THRESHOLD = 0.7
HIGH_TOP1 = 0.20
HIGH_ENTROPY = 0.85
LOW_TOP1 = 0.10
LOW_ENTROPY = 0.90
NEAR_DISTANCE_M = 100.0
DISTANCE_BINS = [0.0, 50.0, 100.0, 200.0, 400.0, float("inf")]
ACTION_NAMES = {FORWARD: "F", LEFT: "L", RIGHT: "R"}


def auc_rank(values, labels):
    """Rank AUC with average ties (same definition as stage2f_analyze.py)."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return float(
        (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def add_history(frame):
    """Add observable history and offline next-step visibility labels."""
    frame = frame.sort_values(["episode_index", "step"]).reset_index(drop=True)
    frame["high"] = (
        (frame["heatmap_max_top1"] >= HIGH_TOP1)
        & (frame["heatmap_max_entropy"] <= HIGH_ENTROPY)
    ).astype(int)
    frame["low"] = (
        (frame["heatmap_max_top1"] < LOW_TOP1)
        | (frame["heatmap_max_entropy"] > LOW_ENTROPY)
    ).astype(int)
    frame["move_state"] = (
        frame["owner_arrival_probability"] < STOP_THRESHOLD
    ).astype(int)
    frame["move_error"] = (
        frame["move_state"].astype(bool)
        & (frame["owner_move_action"] != frame["oracle_move_action"])
    ).astype(int)

    columns = {
        "prev_visible": np.nan,
        "prev_action": np.nan,
        "steps_since_high": 999,
        "last_high_top1": np.nan,
        "next_vis_1": 0,
        "next_vis_2": 0,
        "next_vis_3": 0,
        "next_vis_5": 0,
    }
    for name, default in columns.items():
        frame[name] = default

    for _, group in frame.groupby("episode_index", sort=False):
        index = group.index.to_numpy()
        visible = group["goal_visible"].to_numpy()
        actions = group["executed_action"].to_numpy()
        top1 = group["heatmap_max_top1"].to_numpy()
        high = group["high"].to_numpy()
        last_high_step = -1
        last_high_top1 = float("nan")
        for position, row_index in enumerate(index):
            if high[position]:
                steps_since_high = 0
                current_last_high_top1 = float(top1[position])
            else:
                steps_since_high = (
                    int(group.loc[row_index, "step"]) - last_high_step
                    if last_high_step >= 0
                    else 999
                )
                current_last_high_top1 = last_high_top1
            frame.at[row_index, "steps_since_high"] = steps_since_high
            frame.at[row_index, "last_high_top1"] = current_last_high_top1
            if position > 0:
                frame.at[row_index, "prev_visible"] = visible[position - 1]
                frame.at[row_index, "prev_action"] = actions[position - 1]
            for k in (1, 2, 3, 5):
                frame.at[row_index, f"next_vis_{k}"] = int(
                    visible[position + 1: position + 1 + k].max()
                    if position + 1 < len(index)
                    else 0
                )
            if high[position]:
                last_high_step = int(group.loc[row_index, "step"])
                last_high_top1 = float(top1[position])
    return frame


def action_counts(series):
    counts = {name: 0 for name in ACTION_NAMES.values()}
    for value in series:
        counts[ACTION_NAMES.get(int(value), "?")] += 1
    return counts


def near_far_table(move):
    invisible = move[move["goal_visible"] == 0].copy()
    rows = []
    invisible["distance_band"] = pd.cut(
        invisible["distance_m"], DISTANCE_BINS,
        labels=["0-50", "50-100", "100-200", "200-400", "400+"],
    )
    for band, group in invisible.groupby("distance_band", observed=True):
        rows.append({
            "distance_band": str(band),
            "states": int(len(group)),
            "share_of_invisible": float(len(group) / len(invisible)),
            "move_error_rate": float(group["move_error"].mean()),
            "executed_equals_oracle": float(
                (group["executed_action"] == group["oracle_move_action"]).mean()
            ),
            "oracle_actions": action_counts(group["oracle_move_action"]),
            "executed_actions": action_counts(group["executed_action"]),
            "median_distance_m": float(group["distance_m"].median()),
            "median_abs_heading_error_deg": float(
                group["heading_error_deg"].abs().median()
            ),
        })
    return rows


def transient_table(move):
    invisible = move[move["goal_visible"] == 0]
    rows = []
    for k in (1, 2, 3, 5):
        transient = invisible[invisible[f"next_vis_{k}"] == 1]
        sustained = invisible[invisible[f"next_vis_{k}"] == 0]
        rows.append({
            "k_steps": k,
            "transient_states": int(len(transient)),
            "transient_share": float(len(transient) / len(invisible)),
            "transient_move_error_rate": float(transient["move_error"].mean()),
            "transient_median_distance_m": float(
                transient["distance_m"].median()
            ),
            "sustained_states": int(len(sustained)),
            "sustained_share": float(len(sustained) / len(invisible)),
            "sustained_move_error_rate": float(sustained["move_error"].mean()),
            "sustained_median_distance_m": float(
                sustained["distance_m"].median()
            ),
        })
    return rows


def turn_induced_table(move):
    previous_visible = move["prev_visible"] == 1
    previous_turn = move["prev_action"].isin([LEFT, RIGHT])
    current_invisible = move["goal_visible"] == 0
    subset = move[previous_visible & previous_turn & current_invisible]
    revisible = {
        str(k): float(subset[f"next_vis_{k}"].mean()) if len(subset) else float("nan")
        for k in (1, 2, 3, 5)
    }
    return {
        "states": int(len(subset)),
        "share_of_move_states": float(len(subset) / len(move)),
        "share_of_invisible_states": float(
            len(subset) / max(int((move["goal_visible"] == 0).sum()), 1)
        ),
        "per_band": {
            str(band): int(count)
            for band, count in subset["band"].value_counts().items()
        },
        "median_distance_m": float(subset["distance_m"].median())
        if len(subset) else float("nan"),
        "near_100m_share": float(
            (subset["distance_m"] < NEAR_DISTANCE_M).mean()
        ) if len(subset) else float("nan"),
        "executed_equals_oracle": float(
            (subset["executed_action"] == subset["oracle_move_action"]).mean()
        ) if len(subset) else float("nan"),
        "oracle_actions": action_counts(subset["oracle_move_action"]),
        "executed_actions": action_counts(subset["executed_action"]),
        "revisible_within": revisible,
    }


def recency_trigger_table(move):
    invisible = move["goal_visible"] == 0
    near_invisible = invisible & (move["distance_m"] < NEAR_DISTANCE_M)
    rows = []
    for window in (5, 10, 20, 40):
        trigger = (
            (move["low"] == 1)
            & (move["steps_since_high"] <= window)
            & (move["last_high_top1"] >= HIGH_TOP1)
        )
        n_trigger = int(trigger.sum())
        n_invisible = int(invisible.sum())
        n_near = int(near_invisible.sum())
        rows.append({
            "window_steps": window,
            "trigger_states": n_trigger,
            "trigger_rate": float(n_trigger / len(move)),
            "invisible_precision": float(
                (trigger & invisible).sum() / max(n_trigger, 1)
            ),
            "invisible_recall": float(
                (trigger & invisible).sum() / max(n_invisible, 1)
            ),
            "near_invisible_precision": float(
                (trigger & near_invisible).sum() / max(n_trigger, 1)
            ),
            "near_invisible_recall": float(
                (trigger & near_invisible).sum() / max(n_near, 1)
            ),
            "trigger_median_distance_m": float(
                move.loc[trigger, "distance_m"].median()
            ) if n_trigger else float("nan"),
            "trigger_near_100m_share": float(
                (move.loc[trigger, "distance_m"] < NEAR_DISTANCE_M).mean()
            ) if n_trigger else float("nan"),
        })
    return rows


def distance_auc_table(move):
    rows = []
    move = move.copy()
    move["distance_band"] = pd.cut(
        move["distance_m"], DISTANCE_BINS,
        labels=["0-50", "50-100", "100-200", "200-400", "400+"],
    )
    for band, group in move.groupby("distance_band", observed=True):
        if group["goal_visible"].nunique() < 2:
            continue
        rows.append({
            "distance_band": str(band),
            "states": int(len(group)),
            "invisible": int((group["goal_visible"] == 0).sum()),
            "visible_rate": float(group["goal_visible"].mean()),
            "sim_max_auc": auc_rank(
                group["sim_max"], group["goal_visible"]
            ),
            "heatmap_max_top1_auc": auc_rank(
                group["heatmap_max_top1"], group["goal_visible"]
            ),
            "baseline_top1_auc": auc_rank(
                group["baseline_top1"], group["goal_visible"]
            ),
        })
    return rows


def proximity_proxy_table(move):
    visible = move[move["goal_visible"] == 1]
    near_invisible = (
        (move["goal_visible"] == 0)
        & (move["distance_m"] < NEAR_DISTANCE_M)
    )
    strict_low = (
        (move["heatmap_max_top1"] < 0.02)
        | (move["heatmap_max_entropy"] > 0.98)
    )
    return {
        "last_high_top1_auc_distance_lt_100m": auc_rank(
            move["last_high_top1"].fillna(0.0),
            (move["distance_m"] < NEAR_DISTANCE_M).astype(int),
        ),
        "heatmap_max_top1_auc_distance_lt_100m": auc_rank(
            move["heatmap_max_top1"],
            (move["distance_m"] < NEAR_DISTANCE_M).astype(int),
        ),
        "visible_state_top1_vs_distance_pearson": float(
            np.corrcoef(visible["heatmap_max_top1"], visible["distance_m"])[0, 1]
        ),
        "near_invisible_states": int(near_invisible.sum()),
        "near_invisible_moderate_low_share": float(
            move.loc[near_invisible, "low"].mean()
        ),
        "near_invisible_strict_low_share": float(
            strict_low[near_invisible].mean()
        ),
    }


def smoke_trigger_table(smoke_dir):
    rows = []
    if smoke_dir is None:
        return rows
    for policy in ("side_history", "scan_only"):
        path = Path(smoke_dir) / "steps" / f"{policy}.csv"
        if not path.exists():
            continue
        steps = pd.read_csv(path)
        triggers = steps[steps["trigger_now"] == 1]
        for band in ("short", "medium", "long"):
            band_triggers = triggers[triggers["band"] == band]
            band_steps = steps[steps["band"] == band]
            if band_triggers.empty:
                continue
            rows.append({
                "policy": policy,
                "band": band,
                "decision_steps": int(len(band_steps)),
                "triggers": int(len(band_triggers)),
                "episodes_with_trigger": int(
                    band_triggers["episode_index"].nunique()
                ),
                "trigger_step_rate": float(
                    len(band_triggers) / len(band_steps)
                ),
                "trigger_invisible_rate": float(
                    1.0 - band_triggers["goal_visible"].mean()
                ),
                "median_distance_m": float(
                    band_triggers["distance_m"].median()
                ),
                "p10_distance_m": float(
                    band_triggers["distance_m"].quantile(0.10)
                ),
                "p90_distance_m": float(
                    band_triggers["distance_m"].quantile(0.90)
                ),
                "share_over_100m": float(
                    (band_triggers["distance_m"] > 100.0).mean()
                ),
                "share_over_200m": float(
                    (band_triggers["distance_m"] > 200.0).mean()
                ),
            })
    return rows


def render_markdown(summary):
    lines = ["# Stage-2f supplementary tables", ""]
    lines += ["## Invisible M0 move states by true distance", ""]
    lines += [
        "| Distance | States | Share | Move err | Exec=oracle | Oracle L/R | Executed F/L/R | Median m | |Heading| |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in summary["invisible_by_distance"]:
        oracle = row["oracle_actions"]
        executed = row["executed_actions"]
        lines.append(
            f"| {row['distance_band']} | {row['states']} | "
            f"{100 * row['share_of_invisible']:.1f}% | "
            f"{100 * row['move_error_rate']:.1f}% | "
            f"{100 * row['executed_equals_oracle']:.1f}% | "
            f"L {oracle['L']} / R {oracle['R']} | "
            f"F {executed['F']} / L {executed['L']} / R {executed['R']} | "
            f"{row['median_distance_m']:.1f} | "
            f"{row['median_abs_heading_error_deg']:.1f} |"
        )
    lines += ["", "## Transient vs sustained invisibility", ""]
    lines += [
        "| Re-visible within k | Transient states | Share | Transient move err | Sustained states | Share | Sustained move err |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["transient_vs_sustained"]:
        lines.append(
            f"| {row['k_steps']} | {row['transient_states']} | "
            f"{100 * row['transient_share']:.1f}% | "
            f"{100 * row['transient_move_error_rate']:.1f}% | "
            f"{row['sustained_states']} | {100 * row['sustained_share']:.1f}% | "
            f"{100 * row['sustained_move_error_rate']:.1f}% |"
        )
    lines += ["", "## Turn-induced temporary loss", ""]
    turn = summary["turn_induced_loss"]
    lines.append(f"- States: {turn['states']} "
                 f"({100 * turn['share_of_move_states']:.1f}% of move states; "
                 f"{100 * turn['share_of_invisible_states']:.1f}% of invisible)")
    lines.append(f"- Median true distance: {turn['median_distance_m']:.1f} m; "
                 f"<100 m share {100 * turn['near_100m_share']:.1f}%")
    lines.append(f"- Executed == oracle: {100 * turn['executed_equals_oracle']:.1f}%")
    lines.append(f"- Oracle actions: {turn['oracle_actions']}; "
                 f"executed: {turn['executed_actions']}")
    lines.append("- Re-visible within: " + ", ".join(
        f"{k} steps {100 * v:.1f}%" for k, v in turn["revisible_within"].items()
    ))
    lines += ["", "## Recency-gated observable trigger", ""]
    lines += [
        "| Window | Triggers | Rate | Invisible prec | Invisible rec | Near-invisible prec | Near-invisible rec | Median m | <100 m share |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["recency_gated_trigger"]:
        lines.append(
            f"| {row['window_steps']} | {row['trigger_states']} | "
            f"{100 * row['trigger_rate']:.1f}% | "
            f"{100 * row['invisible_precision']:.1f}% | "
            f"{100 * row['invisible_recall']:.1f}% | "
            f"{100 * row['near_invisible_precision']:.1f}% | "
            f"{100 * row['near_invisible_recall']:.1f}% | "
            f"{row['trigger_median_distance_m']:.1f} | "
            f"{100 * row['trigger_near_100m_share']:.1f}% |"
        )
    lines += ["", "## Visibility AUC by distance band", ""]
    lines += [
        "| Distance | States | Invisible | Visible rate | sim_max AUC | heat top1 AUC | baseline top1 AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["distance_stratified_auc"]:
        lines.append(
            f"| {row['distance_band']} | {row['states']} | {row['invisible']} | "
            f"{100 * row['visible_rate']:.1f}% | "
            f"{row['sim_max_auc']:.3f} | "
            f"{row['heatmap_max_top1_auc']:.3f} | "
            f"{row['baseline_top1_auc']:.3f} |"
        )
    lines += ["", "## Closed-loop trigger distance", ""]
    lines += [
        "| Policy | Band | Decision steps | Triggers | Episodes | Trigger-step rate | Invisible rate | Median m | p10 | p90 | >100 m | >200 m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["smoke_trigger_distance"]:
        lines.append(
            f"| {row['policy']} | {row['band']} | {row['decision_steps']} | "
            f"{row['triggers']} | {row['episodes_with_trigger']} | "
            f"{100 * row['trigger_step_rate']:.1f}% | "
            f"{100 * row['trigger_invisible_rate']:.1f}% | "
            f"{row['median_distance_m']:.1f} | "
            f"{row['p10_distance_m']:.1f} | {row['p90_distance_m']:.1f} | "
            f"{100 * row['share_over_100m']:.1f}% | "
            f"{100 * row['share_over_200m']:.1f}% |"
        )
    lines += ["", "## Observable proximity proxy", ""]
    for key, value in summary["proximity_proxy"].items():
        text = f"{value:.3f}" if isinstance(value, float) else str(value)
        lines.append(f"- `{key}`: {text}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="runs/v83_spatial_stage2/stage2f/"
                "collect_20261019T024000Z/features.csv",
    )
    parser.add_argument(
        "--smoke-dir",
        default="runs/v83_spatial_stage2/stage2f/"
                "smoke_budget12_20261019T031000Z",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/v83_spatial_stage2/stage2f/"
                "supplement_20261019T032000Z",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    m0 = features[features["trajectory"] == "M0"].copy()
    m0 = add_history(m0)
    move = m0[m0["move_state"] == 1].copy()

    summary = {
        "features": str(args.features),
        "m0_states": int(len(m0)),
        "m0_move_states": int(len(move)),
        "m0_invisible_move_states": int((move["goal_visible"] == 0).sum()),
        "definitions": {
            "high_top1": HIGH_TOP1,
            "high_entropy": HIGH_ENTROPY,
            "low_top1": LOW_TOP1,
            "low_entropy": LOW_ENTROPY,
            "near_distance_m": NEAR_DISTANCE_M,
            "note": "true labels are offline evaluation targets only",
        },
        "invisible_by_distance": near_far_table(move),
        "transient_vs_sustained": transient_table(move),
        "turn_induced_loss": turn_induced_table(move),
        "recency_gated_trigger": recency_trigger_table(move),
        "distance_stratified_auc": distance_auc_table(move),
        "smoke_trigger_distance": smoke_trigger_table(args.smoke_dir),
        "proximity_proxy": proximity_proxy_table(move),
    }
    write_json(output_dir / "supplement.json", summary)
    (output_dir / "supplement.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(json.dumps({
        "m0_move_states": summary["m0_move_states"],
        "invisible": summary["m0_invisible_move_states"],
        "turn_induced_states": summary["turn_induced_loss"]["states"],
        "near_invisible": summary["proximity_proxy"]["near_invisible_states"],
        "output": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
