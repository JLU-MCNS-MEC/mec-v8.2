"""Generate Stage-2d comparison tables from the R4 and calibration run dirs.

Usage:
  python v83_spatial_stage2/stage2d_summarize.py \
    --r4-dir runs/v83_spatial_stage2/stage2d/r4_<stamp> \
    --calibration-dir runs/v83_spatial_stage2/stage2d/calibration_<stamp> \
    --output runs/v83_spatial_stage2/stage2d/stage2d_summary.md

The output is a Markdown report body. It contains no conclusions; the
hand-written V83_STAGE2D_REPORT.md adds interpretation and the acceptance gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


BANDS = ("short", "medium", "long")
DISTANCE_BINS = (
    (0.0, 20.0), (20.0, 40.0), (40.0, 60.0),
    (60.0, 100.0), (100.0, 200.0), (200.0, float("inf")),
)


def read_steps(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def outcome_label(row):
    if int(row["success"]):
        return "success"
    if int(row["premature_stop"]):
        return "premature"
    if int(row["max_steps_termination"]):
        return "timeout"
    return "other"


def transition_table(base_rows, other_rows_by_name):
    transitions = (
        ("premature", "success"),
        ("success", "premature"),
        ("premature", "timeout"),
        ("timeout", "success"),
        ("timeout", "premature"),
        ("success", "timeout"),
    )
    base = {row["episode_index"]: row for row in base_rows}
    rows = []
    for name, other_list in other_rows_by_name.items():
        other = {row["episode_index"]: row for row in other_list}
        counts = Counter(
            (outcome_label(base[index]), outcome_label(other[index]))
            for index in base
            if index in other
        )
        rows.append(
            [name] + [counts.get(pair, 0) for pair in transitions]
        )
    return markdown_table(
        ["Method"] + [f"{a}→{b}" for a, b in transitions],
        rows,
    )


def distance_bin_stop_table(steps_by_method):
    headers = [
        "Method", "0-20 m", "20-40 m", "40-60 m", "60-100 m",
        "100-200 m", "200 m+",
    ]
    rows = []
    for name, steps in steps_by_method.items():
        cells = []
        for low, high in DISTANCE_BINS:
            subset = [
                row for row in steps
                if low <= float(row["distance_m"]) < high
            ]
            if not subset:
                cells.append("—")
                continue
            stops = sum(int(row["stop"]) for row in subset)
            cells.append(f"{100.0 * stops / len(subset):.2f}% "
                         f"({stops}/{len(subset)})")
        rows.append([name] + cells)
    return markdown_table(headers, rows)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(value, digits=2):
    return f"{100.0 * float(value):.{digits}f}%"


def pp(value, digits=2):
    return f"{100.0 * float(value):+.{digits}f} pp"


def num(value, digits=4):
    return f"{float(value):.{digits}f}"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def paired_cell(paired, key, higher_is_better=True):
    block = paired.get(key)
    if not block:
        return "—"
    delta = pp(block["delta_rate"])
    wins = block["other_better"]
    losses = block["base_better"]
    if higher_is_better:
        return f"{delta} ({wins}W/{losses}L)"
    return f"{delta} ({wins}W/{losses}L)"


def r4_section(r4_dir):
    metrics_path = Path(r4_dir) / "metrics_recomputed.json"
    if not metrics_path.exists():
        metrics_path = Path(r4_dir) / "metrics.json"
    metrics = load_json(metrics_path)
    rules = metrics["rules"]
    b0 = metrics["b0_reference"]
    lines = ["## R4 — Stop-decision rules (Dev Val 180, full re-run)", ""]
    rows = []
    for rule in rules:
        item = metrics["metrics"][rule]
        overall = item["overall"]
        hazard = item["hazard"]["all"]
        stops = item["stop_events"]["all"]
        rows.append([
            rule,
            pct(overall["sr"]),
            pct(overall["osr"]),
            num(overall["spl"]),
            pct(overall["premature_stop"]),
            pct(overall["timeout"]),
            num(overall["mean_steps"], 1),
            pct(hazard["far40_stop_rate"], 3),
            pct(hazard["near20_stop_rate"], 1),
            num(hazard["expected_false_stop_decisions_per_episode"], 3),
            num(hazard["predicted_false_stop_decisions_per_episode"], 3),
            pct(overall["reached_radius_not_success"], 2),
            num(stops["median_distance_m"], 1) if stops["stops"] else "—",
        ])
    lines.append(markdown_table(
        ["Rule", "SR", "OSR", "SPL", "Premature", "Timeout", "Steps",
         "Far-40 stop/step", "Near-20 stop rate", "Far-40 stops/ep",
         "Far-40 p-sum/ep", "Reach<20 no success", "Stop median m"],
        rows,
    ))
    lines.append("")

    lines.append("### R4 cumulative false-Stop risk decomposition")
    lines.append(
        "Every premature Stop is either far-40 or boundary 20–40 m; the sum "
        "should match the observed premature-Stop rate."
    )
    risk_rows_table = []
    for rule in rules:
        hazard = metrics["metrics"][rule]["hazard"]["all"]
        overall = metrics["metrics"][rule]["overall"]
        far = hazard["expected_false_stop_decisions_per_episode"]
        boundary = hazard["expected_boundary_stop_decisions_per_episode"]
        risk_rows_table.append([
            rule,
            num(far, 3),
            num(boundary, 3),
            num(far + boundary, 3),
            pct(overall["premature_stop"], 2),
        ])
    lines.append(markdown_table(
        ["Rule", "Far-40 stops/ep", "Boundary stops/ep", "Sum",
         "Observed premature"],
        risk_rows_table,
    ))
    lines.append("")

    steps_by_rule = {
        rule: read_steps(Path(r4_dir) / "steps" / f"{rule}.csv")
        for rule in rules
    }
    lines.append("### R4 Stop decisions by distance bin (actual rule decision)")
    lines.append(distance_bin_stop_table(steps_by_rule))
    lines.append("")

    lines.append("### R4 per band")
    rows = []
    for rule in rules:
        for band in BANDS:
            item = metrics["metrics"][rule]["by_band"][band]
            hazard = metrics["metrics"][rule]["hazard"][band]
            rows.append([
                rule, band, pct(item["sr"]), pct(item["osr"]),
                num(item["spl"]), pct(item["premature_stop"]),
                pct(item["timeout"]),
                pct(hazard["far40_stop_rate"], 3),
                pct(hazard["near20_stop_rate"], 1),
            ])
    lines.append(markdown_table(
        ["Rule", "Band", "SR", "OSR", "SPL", "Premature", "Timeout",
         "Far-40 stop/step", "Near-20 stop rate"],
        rows,
    ))
    lines.append("")

    lines.append("### R4 paired vs B0 (per episode)")
    rows = []
    for rule in rules:
        if rule == b0:
            continue
        paired = metrics["paired_vs_b0"][rule]["all"]
        rows.append([
            rule,
            paired_cell(paired, "success", True),
            paired_cell(paired, "osr", True),
            paired_cell(paired, "premature_stop", False),
            paired_cell(paired, "max_steps_termination", False),
            f"{paired['spl']['delta']:+.4f}",
            f"p={paired['success_sign_test_p']:.3f}",
        ])
    lines.append(markdown_table(
        ["Rule", "SR Δ (W/L)", "OSR Δ (W/L)", "Premature Δ (W/L)",
         "Timeout Δ (W/L)", "SPL Δ", "SR sign p"],
        rows,
    ))
    lines.append("")

    lines.append("### R4 per-episode outcome transitions vs B0")
    lines.append(transition_table(
        read_steps(Path(r4_dir) / "threshold_0.5_per_episode.csv"),
        {
            rule: read_steps(Path(r4_dir) / f"{rule}_per_episode.csv")
            for rule in rules if rule != b0
        },
    ))
    lines.append("")

    lines.append("### R4 paired vs B0 by band (SR / OSR / Premature)")
    rows = []
    for rule in rules:
        if rule == b0:
            continue
        for band in BANDS:
            paired = metrics["paired_vs_b0"][rule][band]
            rows.append([
                rule, band,
                paired_cell(paired, "success", True),
                paired_cell(paired, "osr", True),
                paired_cell(paired, "premature_stop", False),
                pct(metrics["metrics"][rule]["by_band"][band]["premature_stop"]),
            ])
    lines.append(markdown_table(
        ["Rule", "Band", "SR Δ (W/L)", "OSR Δ (W/L)", "Premature Δ (W/L)",
         "Premature rate"],
        rows,
    ))
    lines.append("")
    return "\n".join(lines)


def calibration_section(cal_dir):
    closed_path = Path(cal_dir) / "closedloop_metrics_recomputed.json"
    if not closed_path.exists():
        closed_path = Path(cal_dir) / "closedloop_metrics.json"
    closed = load_json(closed_path)
    offline = load_json(Path(cal_dir) / "dev_val_offline_metrics.json")
    cal_val = load_json(Path(cal_dir) / "calibration_val_metrics.json")
    mining = load_json(Path(cal_dir) / "mining_stats.json")
    split = load_json(Path(cal_dir) / "split.json")
    replay = load_json(Path(cal_dir) / "replay_check.json")
    variants = ["R1", "R2", "R3"]
    names = ["B0"] + variants
    steps_by_name = {
        name: read_steps(Path(cal_dir) / "steps" / f"{name}.csv")
        for name in names
    }

    lines = ["## R1/R2/R3 — Off-policy calibration", ""]
    lines.append(
        f"Split: {split['train_episode_count']} train / "
        f"{split['validation_episode_count']} internal-validation episodes; "
        f"hard far negatives={mining['hard']['added_far_rows']}, "
        f"random control={mining['random']['added_far_rows']}; "
        f"B0 replay outcome mismatches={replay.get('outcome_mismatch_count', replay['mismatch_count'])}, "
        f"numeric mismatches={replay.get('numeric_mismatch_count', 0)}."
    )
    lines.append("")

    lines.append("### Selected configurations")
    rows = []
    for variant in variants:
        result = load_json(Path(cal_dir) / variant / "metrics.json")
        config = result["selected_config"]
        rows.append([
            variant,
            f"lr={config['lr']:g}",
            f"anchor={config['lambda_anchor']:g}",
            f"residual={config['lambda_residual']:g}",
            result["best_epoch"],
            result["parameter_count"],
            result["train_rows"],
            result["init_max_abs_diff"],
        ])
    lines.append(markdown_table(
        ["Variant", "LR", "Anchor λ", "Residual λ", "Best epoch", "Params",
         "Train rows", "Init max abs diff"],
        rows,
    ))
    lines.append("")

    lines.append("### R2/R3 grid top-3 on the internal validation")
    rows = []
    for variant in ("R2", "R3"):
        grid = read_steps(Path(cal_dir) / variant / "grid_summary.csv")
        ranked = sorted(
            grid, key=lambda row: float(row["selection_score"])
        )[:3]
        for row in ranked:
            rows.append([
                variant, row["lr"], row["lambda_anchor"],
                row["lambda_residual"], row["best_epoch"],
                num(row["selection_score"], 4),
                pct(row["near20_recall"], 1),
                num(row["far40_stop_rate"], 5),
                num(row["far40_stops_per_episode"], 3),
            ])
    lines.append(markdown_table(
        ["Variant", "LR", "Anchor λ", "Residual λ", "Best epoch", "Score",
         "Near-20 recall", "Far-40 stop/step", "Far-40 stops/ep"],
        rows,
    ))
    lines.append("")

    lines.append(
        "### Internal calibration-validation (B0-rollout states, 120 episodes)"
    )
    rows = []
    for name in names:
        hazard = cal_val[name]["hazard"]
        metrics = cal_val[name]["metrics"]["natural"]
        rows.append([
            name,
            pct(hazard["far40_stop_rate"], 3),
            num(hazard["expected_false_stop_decisions_per_episode"], 3),
            num(hazard["predicted_false_stop_decisions_per_episode"], 3),
            pct(hazard["near20_stop_rate"], 1),
            pct(hazard["boundary_stop_rate"], 2),
            num(metrics["f1"], 3),
            num(metrics["brier"], 5),
        ])
    lines.append(markdown_table(
        ["Variant", "Far-40 stop/step", "Far-40 stops/ep", "Far-40 p-sum/ep",
         "Near-20 stop rate", "Boundary stop rate", "Natural F1", "Brier"],
        rows,
    ))
    lines.append("")

    lines.append("### Dev Val 180 offline Arrival metrics")
    rows = []
    for name in names:
        item = offline["metrics"][name]
        natural = item["natural"]
        hard = item["hard"] or {}
        rows.append([
            name,
            num(natural["f1"], 3),
            num(hard.get("f1", float("nan")), 3),
            num(natural["brier"], 5),
            num(natural["ece"], 4),
            pct(item["far40_false_stop_rate"], 3),
            pct(item["far100_false_stop_rate"], 3),
            pct(item["near20_recall"], 2),
            item["high_p_far40_count"],
        ])
    lines.append(markdown_table(
        ["Variant", "Natural F1", "Hard F1", "Natural Brier", "Natural ECE",
         "Far-40 FPR", "Far-100 FPR", "Near-20 recall", "High-p far n"],
        rows,
    ))
    lines.append("")

    lines.append("### Dev Val 180 closed loop")
    rows = []
    for name in names:
        item = closed["metrics"][name]
        overall = item["overall"]
        hazard = item["hazard"]["all"]
        rows.append([
            name,
            pct(overall["sr"]),
            pct(overall["osr"]),
            num(overall["spl"]),
            num(overall["ndtw"]),
            pct(overall["premature_stop"]),
            pct(overall["timeout"]),
            num(overall["mean_steps"], 1),
            num(overall["mean_min_distance_m"], 1),
            pct(hazard["far40_stop_rate"], 3),
            pct(hazard["near20_stop_rate"], 1),
            num(hazard["expected_false_stop_decisions_per_episode"], 3),
            num(hazard["predicted_false_stop_decisions_per_episode"], 3),
            pct(overall["reached_radius_not_success"], 2),
        ])
    lines.append(markdown_table(
        ["Variant", "SR", "OSR", "SPL", "nDTW", "Premature", "Timeout",
         "Steps", "Min dist m", "Far-40 stop/step", "Near-20 stop rate",
         "Far-40 stops/ep", "Far-40 p-sum/ep", "Reach<20 no success"],
        rows,
    ))
    lines.append("")

    lines.append("### Calibration cumulative false-Stop risk decomposition")
    risk_rows_table = []
    for name in names:
        hazard = closed["metrics"][name]["hazard"]["all"]
        overall = closed["metrics"][name]["overall"]
        far = hazard["expected_false_stop_decisions_per_episode"]
        boundary = hazard["expected_boundary_stop_decisions_per_episode"]
        risk_rows_table.append([
            name,
            num(far, 3),
            num(boundary, 3),
            num(far + boundary, 3),
            pct(overall["premature_stop"], 2),
        ])
    lines.append(markdown_table(
        ["Variant", "Far-40 stops/ep", "Boundary stops/ep", "Sum",
         "Observed premature"],
        risk_rows_table,
    ))
    lines.append("")

    lines.append("### Calibration Stop decisions by distance bin")
    lines.append(distance_bin_stop_table(steps_by_name))
    lines.append("")

    lines.append("### Dev Val 180 closed loop by band")
    rows = []
    for name in names:
        for band in BANDS:
            item = closed["metrics"][name]["by_band"][band]
            hazard = closed["metrics"][name]["hazard"][band]
            rows.append([
                name, band, pct(item["sr"]), pct(item["osr"]),
                num(item["spl"]), pct(item["premature_stop"]),
                pct(item["timeout"]),
                pct(hazard["far40_stop_rate"], 3),
                pct(hazard["near20_stop_rate"], 1),
            ])
    lines.append(markdown_table(
        ["Variant", "Band", "SR", "OSR", "SPL", "Premature", "Timeout",
         "Far-40 stop/step", "Near-20 stop rate"],
        rows,
    ))
    lines.append("")

    lines.append("### Paired per-episode comparison vs B0")
    rows = []
    for name in variants:
        paired = closed["paired_vs_b0"][name]["all"]
        rows.append([
            name,
            paired_cell(paired, "success", True),
            paired_cell(paired, "osr", True),
            paired_cell(paired, "premature_stop", False),
            paired_cell(paired, "max_steps_termination", False),
            f"{paired['spl']['delta']:+.4f}",
            f"{paired['ndtw_delta']:+.4f}",
            f"p={paired['success_sign_test_p']:.3f}",
        ])
    lines.append(markdown_table(
        ["Variant", "SR Δ (W/L)", "OSR Δ (W/L)", "Premature Δ (W/L)",
         "Timeout Δ (W/L)", "SPL Δ", "nDTW Δ", "SR sign p"],
        rows,
    ))
    lines.append("")

    lines.append("### Calibration per-episode outcome transitions vs B0")
    lines.append(transition_table(
        read_steps(Path(cal_dir) / "B0_per_episode.csv"),
        {
            name: read_steps(Path(cal_dir) / f"{name}_per_episode.csv")
            for name in variants
        },
    ))
    lines.append("")

    lines.append("### Paired per-band comparison vs B0")
    rows = []
    for name in variants:
        for band in BANDS:
            paired = closed["paired_vs_b0"][name][band]
            rows.append([
                name, band,
                paired_cell(paired, "success", True),
                paired_cell(paired, "osr", True),
                paired_cell(paired, "premature_stop", False),
                paired_cell(paired, "max_steps_termination", False),
                f"{paired['spl']['delta']:+.4f}",
            ])
    lines.append(markdown_table(
        ["Variant", "Band", "SR Δ (W/L)", "OSR Δ (W/L)", "Premature Δ (W/L)",
         "Timeout Δ (W/L)", "SPL Δ"],
        rows,
    ))
    lines.append("")

    lines.append("### On-policy base -> calibrated probability shift")
    lines.append(
        "Per-step rows from each variant's own closed-loop trajectory. "
        "`High-p far` = far-40 steps with base B0 p >= 0.5."
    )
    rows = []
    for name in names:
        steps = steps_by_name[name]
        far = [row for row in steps if float(row["distance_m"]) > 40.0]
        high_p = [
            row for row in far
            if float(row["base_arrival_probability"]) >= 0.5
        ]
        near = [row for row in steps if float(row["distance_m"]) < 20.0]

        def _mean(rows, key):
            return (
                sum(float(row[key]) for row in rows) / len(rows) if rows else float("nan")
            )

        rows.append([
            name,
            len(far),
            num(_mean(far, "base_arrival_probability"), 4),
            num(_mean(far, "arrival_probability"), 4),
            len(high_p),
            num(_mean(high_p, "base_arrival_probability"), 4),
            num(_mean(high_p, "arrival_probability"), 4),
            pct(
                sum(
                    float(row["arrival_probability"]) < 0.5 for row in high_p
                ) / len(high_p), 1,
            ) if high_p else "—",
            len(near),
            pct(
                sum(int(row["stop"]) for row in near) / len(near), 1
            ) if near else "—",
        ])
    lines.append(markdown_table(
        ["Variant", "Far-40 steps", "Far base p", "Far cal p", "High-p far n",
         "High-p base p", "High-p cal p", "High-p below 0.5", "Near steps",
         "Near stop rate"],
        rows,
    ))
    lines.append("")
    return "\n".join(lines)


def threshold_exposure_section(r4_dir):
    """Static threshold sweep on the B0 trajectory (diagnostic, not a rerun)."""
    with (Path(r4_dir) / "steps" / "threshold_0.5.csv").open(
        encoding="utf-8"
    ) as handle:
        steps = list(csv.DictReader(handle))
    episodes = len({row["episode_index"] for row in steps})
    far = [row for row in steps if float(row["distance_m"]) > 40.0]
    near = [row for row in steps if float(row["distance_m"]) < 20.0]
    rows = []
    for threshold in (0.5, 0.6, 0.7, 0.8):
        far_decisions = sum(
            float(row["arrival_probability"]) >= threshold for row in far
        )
        near_decisions = sum(
            float(row["arrival_probability"]) >= threshold for row in near
        )
        rows.append([
            f"{threshold:.1f}",
            f"{far_decisions / episodes:.3f}",
            pct(far_decisions / len(far), 3) if far else "—",
            pct(near_decisions / len(near), 1) if near else "—",
            f"{near_decisions}/{len(near)}",
        ])
    return "\n".join([
        "### Static threshold exposure on the B0 trajectory (diagnostic only)",
        "",
        "This is a re-threshold of the saved B0 per-step probabilities, not a "
        "closed-loop result; the rule table above is the actual re-run.",
        "",
        markdown_table(
            ["Threshold", "Far-40 decisions/ep", "Far-40 decision rate",
             "Near-20 Stop rate", "Near Stop steps"],
            rows,
        ),
        "",
    ])


def headline_section(r4_dir, cal_dir):
    r4_path = Path(r4_dir) / "metrics_recomputed.json"
    if not r4_path.exists():
        r4_path = Path(r4_dir) / "metrics.json"
    r4 = load_json(r4_path)
    cal_path = Path(cal_dir) / "closedloop_metrics_recomputed.json"
    if not cal_path.exists():
        cal_path = Path(cal_dir) / "closedloop_metrics.json"
    cal = load_json(cal_path)
    rows = []
    b0_hazard = r4["metrics"][r4["b0_reference"]]["hazard"]["all"]
    b0 = r4["metrics"][r4["b0_reference"]]["overall"]
    rows.append([
        "B0 (threshold 0.5)", pct(b0["sr"]), pct(b0["osr"]), num(b0["spl"]),
        pct(b0["premature_stop"]), pct(b0["timeout"]),
        pct(b0_hazard["far40_stop_rate"], 3),
        pct(b0_hazard["near20_stop_rate"], 1),
        num(b0_hazard["expected_false_stop_decisions_per_episode"], 3),
        pct(b0["reached_radius_not_success"], 2),
        "—", "—",
    ])
    for rule in r4["rules"]:
        if rule == r4["b0_reference"]:
            continue
        overall = r4["metrics"][rule]["overall"]
        hazard = r4["metrics"][rule]["hazard"]["all"]
        paired = r4["paired_vs_b0"][rule]["all"]
        rows.append([
            f"R4 {rule}", pct(overall["sr"]), pct(overall["osr"]),
            num(overall["spl"]), pct(overall["premature_stop"]),
            pct(overall["timeout"]),
            pct(hazard["far40_stop_rate"], 3),
            pct(hazard["near20_stop_rate"], 1),
            num(hazard["expected_false_stop_decisions_per_episode"], 3),
            pct(overall["reached_radius_not_success"], 2),
            paired_cell(paired, "success", True),
            f"p={paired['success_sign_test_p']:.3f}",
        ])
    for name in ["R1", "R2", "R3"]:
        overall = cal["metrics"][name]["overall"]
        hazard = cal["metrics"][name]["hazard"]["all"]
        paired = cal["paired_vs_b0"][name]["all"]
        rows.append([
            f"Cal {name}", pct(overall["sr"]), pct(overall["osr"]),
            num(overall["spl"]), pct(overall["premature_stop"]),
            pct(overall["timeout"]),
            pct(hazard["far40_stop_rate"], 3),
            pct(hazard["near20_stop_rate"], 1),
            num(hazard["expected_false_stop_decisions_per_episode"], 3),
            pct(overall["reached_radius_not_success"], 2),
            paired_cell(paired, "success", True),
            f"p={paired['success_sign_test_p']:.3f}",
        ])
    return "\n".join([
        "## Headline comparison (Dev Val 180)",
        "",
        markdown_table(
            ["Method", "SR", "OSR", "SPL", "Premature", "Timeout",
             "Far-40 stop/step", "Near-20 stop rate", "Far-40 stops/ep",
             "Reach<20 no success", "SR Δ vs B0 (W/L)", "SR sign p"],
            rows,
        ),
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = [
        "# V8.3 Stage-2d generated comparison tables",
        "",
        headline_section(args.r4_dir, args.calibration_dir),
        r4_section(args.r4_dir),
        threshold_exposure_section(args.r4_dir),
        calibration_section(args.calibration_dir),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output), encoding="utf-8")
    print("\n".join(output))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
