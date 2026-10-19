"""Generate the M1 threshold-migration comparison tables.

Reads the M1 evaluation `metrics.json` and prints Markdown tables:
  1. M1@0.5 vs M1@0.7 overall and by band.
  2. Paired M1@0.7 vs M1@0.5 (W/L, 95% CI).
  3. M1@0.7 vs the existing M0@0.7 reference.
  4. Far/near Stop diagnostics.

No model is loaded and no episode is re-run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BANDS = ("short", "medium", "long")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(value, digits=2):
    return f"{100.0 * float(value):.{digits}f}%"


def num(value, digits=4):
    return f"{float(value):.{digits}f}"


def pp(value, digits=2):
    return f"{100.0 * float(value):+.{digits}f} pp"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def paired_cell(block, key, as_pp=True, digits=4):
    if not block or key not in block:
        return "—"
    item = block[key]
    delta = pp(item["delta_rate"]) if as_pp else f"{item['delta']:+.{digits}f}"
    return (
        f"{delta} (W{item['other_better']}/L{item['base_better']}), "
        f"95% CI {[round(x, 4) for x in item['delta_ci95']]}"
    )


def delta_cell(block, key, ci_key, digits=4):
    if not block:
        return "—"
    return (
        f"{block[key]:+.{digits}f} "
        f"(95% CI {[round(x, 4) for x in block[ci_key]]})"
    )


def summary_row(name, band, summary):
    return [
        name, band, pct(summary["sr"]), f"{summary['success_count']}/{summary['n']}",
        pct(summary["osr"]), num(summary["spl"]), num(summary["ndtw"]),
        pct(summary["premature_stop"]), pct(summary["timeout"]),
        num(summary["mean_steps"], 1), num(summary["mean_min_distance_m"], 1),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = load_json(args.metrics)
    summaries = metrics["summaries"]
    hazard = metrics["hazard"]
    stops = metrics["stop_events"]
    pair_m1 = metrics["paired_m1_0.7_vs_m1_0.5"]
    pair_m0 = metrics["paired_m1_0.7_vs_m0_0.7"]
    m0 = metrics["m0_0.7_reference"]

    lines = ["# M1 threshold migration — generated tables", ""]

    lines.append("## Table 1 — M1@0.5 vs M1@0.7 (Dev Val 180)")
    rows = []
    for rule in ("M1@0.5", "M1@0.7"):
        for band in ("all",) + BANDS:
            rows.append(summary_row(rule, band, summaries[rule][band]))
    lines.append(markdown_table(
        ["Method", "Band", "SR", "Success", "OSR", "SPL", "nDTW",
         "Premature", "Timeout", "Steps", "Min dist m"],
        rows,
    ))
    lines.append("")

    lines.append("## Table 2 — Paired M1@0.7 vs M1@0.5")
    rows = []
    for band in ("all",) + BANDS:
        block = pair_m1[band]
        rows.append([
            band,
            paired_cell(block, "success"),
            paired_cell(block, "osr"),
            paired_cell(block, "premature_stop"),
            paired_cell(block, "max_steps_termination"),
            delta_cell(block, "spl_delta", "spl_delta_ci95"),
            delta_cell(block, "ndtw_delta", "ndtw_delta_ci95"),
        ])
    lines.append(markdown_table(
        ["Band", "SR Δ (W/L, CI)", "OSR Δ (W/L, CI)",
         "Premature Δ (W/L, CI)", "Timeout Δ (W/L, CI)",
         "SPL Δ (CI)", "nDTW Δ (CI)"],
        rows,
    ))
    lines.append("")

    lines.append("## Table 3 — M1@0.7 vs existing M0@0.7")
    rows = []
    for rule in ("M0@0.7", "M1@0.7"):
        source = m0 if rule == "M0@0.7" else summaries[rule]
        for band in ("all",) + BANDS:
            rows.append(summary_row(rule, band, source[band]))
    lines.append(markdown_table(
        ["Method", "Band", "SR", "Success", "OSR", "SPL", "nDTW",
         "Premature", "Timeout", "Steps", "Min dist m"],
        rows,
    ))
    lines.append("")
    lines.append("Paired M1@0.7 vs M0@0.7:")
    rows = []
    for band in ("all",) + BANDS:
        block = pair_m0[band]
        rows.append([
            band,
            paired_cell(block, "success"),
            paired_cell(block, "osr"),
            paired_cell(block, "premature_stop"),
            paired_cell(block, "max_steps_termination"),
            delta_cell(block, "spl_delta", "spl_delta_ci95"),
            delta_cell(block, "ndtw_delta", "ndtw_delta_ci95"),
        ])
    lines.append(markdown_table(
        ["Band", "SR Δ (W/L, CI)", "OSR Δ (W/L, CI)",
         "Premature Δ (W/L, CI)", "Timeout Δ (W/L, CI)",
         "SPL Δ (CI)", "nDTW Δ (CI)"],
        rows,
    ))
    lines.append("")

    lines.append("## Far/near Stop diagnostics")
    rows = []
    for rule in ("M1@0.5", "M1@0.7"):
        for band in ("all",) + BANDS:
            h = hazard[rule][band]
            s = stops[rule][band]
            summary = summaries[rule][band]
            rows.append([
                rule, band,
                pct(h["far40_stop_rate"], 3),
                num(h["expected_false_stop_decisions_per_episode"], 3),
                num(h["expected_boundary_stop_decisions_per_episode"], 3),
                pct(h["near20_stop_rate"], 1),
                f"{h['near20_stops']}/{h['near20_steps']}",
                f"{summary['reached_radius_not_success'] * summary['n']:.0f}/{summary['n']}",
                num(s["median_distance_m"], 1) if s["stops"] else "—",
                s["far_gt100"],
            ])
    lines.append(markdown_table(
        ["Method", "Band", "Far-40 stop/step", "Far-40 stops/ep",
         "Boundary stops/ep", "Near-20 stop rate", "Near stops/steps",
         "Reach-no-success", "Stop median m", "Far>100 stops"],
        rows,
    ))
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
