"""Stage-2e addendum: what exactly does the M1 fusion residual change?

Reads the exact-state replay CSV (no new inference) and reports, for states
where M1 would move (arrival p < 0.7):

  * the 3x3 base-action -> fused-action transition matrix, by distance and band;
  * how many FORWARD / LEFT / RIGHT decisions are changed;
  * whether each changed action agrees with the privileged 3-way heading oracle
    (beneficial / harmful / neutral);
  * the one-step |heading-error| change relative to the base action;
  * the one-step true-distance change relative to the base action.

The one-step heading/distance values are diagnostics on the current state, not
closed-loop outcomes. FORWARD uses the 5 m step and the current goal geometry;
LEFT/RIGHT use the 15 deg turn. Boundary clipping is ignored.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import FORWARD, LEFT, RIGHT  # noqa: E402
from v83_spatial_stage2.stage2d_common import STAGE2_ROOT, write_csv, write_json  # noqa: E402


ACTION_NAMES = {FORWARD: "F", LEFT: "L", RIGHT: "R"}
DISTANCE_BINS = (
    ("0-20", 0.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-100", 40.0, 100.0),
    ("100-200", 100.0, 200.0),
    ("200+", 200.0, float("inf")),
)
BANDS = ("all", "short", "medium", "long")
STEP_M = 5.0
TURN_DEG = 15.0
STOP_THRESHOLD = 0.7
DEFAULT_REPLAY = STAGE2_ROOT / "stage2e/replay_20261019T015500Z/replay_steps.csv"


def wrap_deg(value):
    return (value + 180.0) % 360.0 - 180.0


def heading_after(action, heading_deg, distance_m):
    """Heading error after one action, in the current goal geometry."""
    if action == FORWARD:
        heading = math.radians(heading_deg)
        new_x = distance_m * math.cos(heading) - STEP_M
        new_y = distance_m * math.sin(heading)
        return math.degrees(math.atan2(new_y, new_x))
    if action == LEFT:
        return wrap_deg(heading_deg - TURN_DEG)
    if action == RIGHT:
        return wrap_deg(heading_deg + TURN_DEG)
    raise ValueError(action)


def distance_after(action, heading_deg, distance_m):
    if action == FORWARD:
        heading = math.radians(heading_deg)
        new_x = distance_m * math.cos(heading) - STEP_M
        new_y = distance_m * math.sin(heading)
        return math.hypot(new_x, new_y)
    return distance_m


def distance_bin(distance):
    for name, low, high in DISTANCE_BINS:
        if low <= distance < high:
            return name
    return DISTANCE_BINS[-1][0]


def read_replay(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows, trajectory, band, distance_name):
    """Counts for one (trajectory, band, distance) cell."""
    transitions = Counter()
    oracle = Counter()
    heading = Counter()
    distance = Counter()
    base_counts = Counter()
    changed = 0
    for row in rows:
        base = int(row["m1_base_action"])
        fused = int(row["m1_fused_action"])
        base_counts[base] += 1
        transitions[(base, fused)] += 1
        if base == fused:
            continue
        changed += 1
        oracle_action = int(row["oracle_move_action"])
        if fused == oracle_action and base != oracle_action:
            oracle["beneficial"] += 1
        elif base == oracle_action and fused != oracle_action:
            oracle["harmful"] += 1
        else:
            oracle["neutral"] += 1

        heading_error = float(row["heading_error_deg"])
        current_distance = float(row["distance_m"])
        base_heading = abs(heading_after(base, heading_error, current_distance))
        fused_heading = abs(heading_after(fused, heading_error, current_distance))
        if fused_heading < base_heading - 1e-6:
            heading["improved"] += 1
        elif fused_heading > base_heading + 1e-6:
            heading["worsened"] += 1
        else:
            heading["neutral"] += 1

        base_distance = distance_after(base, heading_error, current_distance)
        fused_distance = distance_after(fused, heading_error, current_distance)
        if fused_distance < base_distance - 1e-6:
            distance["closer"] += 1
        elif fused_distance > base_distance + 1e-6:
            distance["farther"] += 1
        else:
            distance["same"] += 1

    def transition_count(source, target):
        return int(transitions[(source, target)])

    return {
        "trajectory": trajectory,
        "band": band,
        "distance_bin": distance_name,
        "steps": len(rows),
        "base_F": int(base_counts[FORWARD]),
        "base_L": int(base_counts[LEFT]),
        "base_R": int(base_counts[RIGHT]),
        "changed": int(changed),
        "change_rate": float(changed / len(rows)) if rows else float("nan"),
        "changed_from_F": int(sum(
            count for (base, fused), count in transitions.items()
            if base == FORWARD and fused != FORWARD
        )),
        "changed_from_L": int(sum(
            count for (base, fused), count in transitions.items()
            if base == LEFT and fused != LEFT
        )),
        "changed_from_R": int(sum(
            count for (base, fused), count in transitions.items()
            if base == RIGHT and fused != RIGHT
        )),
        "F_to_L": transition_count(FORWARD, LEFT),
        "F_to_R": transition_count(FORWARD, RIGHT),
        "L_to_F": transition_count(LEFT, FORWARD),
        "L_to_R": transition_count(LEFT, RIGHT),
        "R_to_F": transition_count(RIGHT, FORWARD),
        "R_to_L": transition_count(RIGHT, LEFT),
        "oracle_beneficial": int(oracle["beneficial"]),
        "oracle_harmful": int(oracle["harmful"]),
        "oracle_neutral": int(oracle["neutral"]),
        "heading_improved": int(heading["improved"]),
        "heading_worsened": int(heading["worsened"]),
        "heading_neutral": int(heading["neutral"]),
        "distance_closer": int(distance["closer"]),
        "distance_farther": int(distance["farther"]),
        "distance_same": int(distance["same"]),
    }


def transition_oracle_rows(rows, trajectory, band, distance_name):
    table = Counter()
    for row in rows:
        base = int(row["m1_base_action"])
        fused = int(row["m1_fused_action"])
        if base == fused:
            continue
        oracle_action = int(row["oracle_move_action"])
        if fused == oracle_action and base != oracle_action:
            label = "beneficial"
        elif base == oracle_action and fused != oracle_action:
            label = "harmful"
        else:
            label = "neutral"
        table[(base, fused, label)] += 1
    output = []
    for (base, fused, label), count in sorted(table.items()):
        output.append({
            "trajectory": trajectory,
            "band": band,
            "distance_bin": distance_name,
            "transition": f"{ACTION_NAMES[base]}->{ACTION_NAMES[fused]}",
            "count": int(count),
            "outcome": label,
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    replay = read_replay(args.replay)
    by_trajectory = {}
    for row in replay:
        if float(row["m1_arrival_probability"]) >= STOP_THRESHOLD:
            continue
        by_trajectory.setdefault(row["trajectory"], []).append(row)

    distance_rows = []
    band_rows = []
    transition_rows = []
    for trajectory, rows in by_trajectory.items():
        for band in BANDS:
            band_subset = rows if band == "all" else [
                row for row in rows if str(row["band"]) == band
            ]
            if not band_subset:
                continue
            band_rows.append(summarize(band_subset, trajectory, band, "all"))
            transition_rows.extend(
                transition_oracle_rows(band_subset, trajectory, band, "all")
            )
        for name, low, high in DISTANCE_BINS:
            subset = [
                row for row in rows if low <= float(row["distance_m"]) < high
            ]
            if not subset:
                continue
            distance_rows.append(
                summarize(subset, trajectory, "all", name)
            )
            transition_rows.extend(
                transition_oracle_rows(subset, trajectory, "all", name)
            )

    write_csv(args.output_dir / "action_change_by_distance.csv", distance_rows)
    write_csv(args.output_dir / "action_change_by_band.csv", band_rows)
    write_csv(args.output_dir / "action_transition_oracle.csv", transition_rows)
    write_json(args.output_dir / "summary.json", {
        "replay": str(args.replay),
        "stop_threshold": STOP_THRESHOLD,
        "step_m": STEP_M,
        "turn_deg": TURN_DEG,
        "note": (
            "Changed actions only. Oracle = privileged 3-way heading rule. "
            "Heading/distance effects are one-step diagnostics relative to the "
            "base action; they are not closed-loop outcomes. For a F->turn "
            "change the distance column is mechanically 'farther' because the "
            "turn keeps distance while the FORWARD baseline moves 5 m."
        ),
        "by_distance": distance_rows,
        "by_band": band_rows,
        "transition_oracle": transition_rows,
    })

    # Compact terminal table for the report.
    headers = (
        "trajectory", "band", "dist", "steps", "changed", "rate",
        "fromF", "fromL", "fromR", "F->L", "F->R", "L->F", "R->F",
        "benefit", "harmful", "neutral", "head+", "head-", "head=",
    )
    print("\t".join(headers))
    for row in distance_rows + band_rows:
        print("\t".join(str(row[key]) for key in (
            "trajectory", "band", "distance_bin", "steps", "changed",
            "change_rate", "changed_from_F", "changed_from_L",
            "changed_from_R", "F_to_L", "F_to_R", "L_to_F", "R_to_F",
            "oracle_beneficial", "oracle_harmful", "oracle_neutral",
            "heading_improved", "heading_worsened", "heading_neutral",
        )))
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
