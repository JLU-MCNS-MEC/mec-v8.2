"""Aggregate Stage-2i Dev-Val-180 results and write report tables.

Reads only the staged per-episode CSV / metrics JSON files.  No Final Test
file is read.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2i_common import (
    MODEL_DIRS, MODEL_TITLES, ORDER, model_dir, read_json,
)
from v83_spatial_stage2.stage2i_model import VARIANTS
from v83_spatial_training.eval_closedloop_spatial import paired_comparison


BANDS = ("short", "medium", "long")


def read_rows(variant):
    root = model_dir(variant)
    path = root / "eval" / f"{variant}_per_episode.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows):
    if not rows:
        return None
    return {
        "episodes": len(rows),
        "sr": sum(int(row["success"]) for row in rows) / len(rows),
        "osr": sum(int(row["osr"]) for row in rows) / len(rows),
        "spl": sum(float(row["spl"]) for row in rows) / len(rows),
        "ndtw": sum(float(row["ndtw"]) for row in rows) / len(rows),
        "premature_stop": sum(int(row["premature_stop"]) for row in rows) / len(rows),
        "timeout": sum(int(row["max_steps_termination"]) for row in rows) / len(rows),
        "steps": sum(int(row["steps"]) for row in rows) / len(rows),
    }


def band_table(variant_rows):
    lines = [
        "| Model | Band | SR | OSR | SPL | nDTW |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in ORDER:
        rows = variant_rows.get(variant, [])
        for band in BANDS:
            subset = [row for row in rows if row["band"] == band]
            summary = summarize(subset)
            if summary is None:
                continue
            lines.append(
                f"| {variant} | {band} | {100*summary['sr']:.2f}% | "
                f"{100*summary['osr']:.2f}% | {summary['spl']:.4f} | "
                f"{summary['ndtw']:.4f} |"
            )
    return "\n".join(lines)


def overall_table(variant_rows):
    lines = [
        "| Model | Inputs | Params | SR | OSR | SPL | nDTW | Premature | "
        "Timeout | Steps | Best Epoch | Train min |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ORDER:
        rows = variant_rows.get(variant, [])
        if not rows:
            continue
        summary = summarize(rows)
        result = read_json(model_dir(variant) / "training_result.json", {}) or {}
        inputs = MODEL_TITLES[variant].replace("Transformer-M", "").strip(
            " +"
        ) or "Original M0 inputs"
        lines.append(
            f"| {variant} | {inputs} | {result.get('parameters', 0):,} | "
            f"{100*summary['sr']:.2f}% | {100*summary['osr']:.2f}% | "
            f"{summary['spl']:.4f} | {summary['ndtw']:.4f} | "
            f"{100*summary['premature_stop']:.2f}% | "
            f"{100*summary['timeout']:.2f}% | {summary['steps']:.1f} | "
            f"{result.get('best_epoch', '—')} | "
            f"{result.get('training_time_s', 0)/60:.0f} |"
        )
    return "\n".join(lines)


def band_paired(variant_rows):
    """Per-band paired SR/SPL/nDTW deltas versus T0."""
    t0 = variant_rows.get("T0", [])
    if not t0:
        return {}
    result = {}
    for variant in ORDER:
        if variant == "T0" or not variant_rows.get(variant):
            continue
        result[variant] = {}
        for band in BANDS:
            base = [row for row in t0 if row["band"] == band]
            other = [row for row in variant_rows[variant] if row["band"] == band]
            if not base or not other:
                continue
            paired = paired_comparison({"T0": base, variant: other},
                                       base="T0", other=variant)
            result[variant][band] = {
                "sr_delta": paired["success"]["delta_rate"],
                "sr_ci95": paired["success"]["delta_ci95"],
                "sr_wlt": [paired["success"]["other_better"],
                           paired["success"]["base_better"],
                           paired["success"]["ties"]],
                "spl_delta": paired["spl"]["delta"],
                "spl_ci95": paired["spl_delta_ci95"],
                "ndtw_delta": paired["ndtw"]["delta"],
                "ndtw_ci95": paired["ndtw_delta_ci95"],
            }
    return result


def band_paired_table(variant_rows):
    t0 = variant_rows.get("T0", [])
    if not t0:
        return "T0 is not available yet; band paired comparisons are pending."
    table = [
        "| Model | Band | ΔSR (W/L/T) | ΔSR 95% CI | ΔSPL | ΔnDTW |",
        "|---|---|---:|---|---:|---:|",
    ]
    for variant in ORDER:
        if variant == "T0" or not variant_rows.get(variant):
            continue
        for band in BANDS:
            base = [row for row in t0 if row["band"] == band]
            other = [row for row in variant_rows[variant] if row["band"] == band]
            if not base or not other:
                continue
            paired = paired_comparison({"T0": base, variant: other},
                                       base="T0", other=variant)
            sr = paired["success"]
            table.append(
                f"| {variant} | {band} | {100*sr['delta_rate']:+.2f} pp "
                f"({sr['other_better']}/{sr['base_better']}/{sr['ties']}) | "
                f"[{100*sr['delta_ci95'][0]:+.2f}, "
                f"{100*sr['delta_ci95'][1]:+.2f}] | "
                f"{paired['spl']['delta']:+.4f} | "
                f"{paired['ndtw']['delta']:+.4f} |"
            )
    return "\n".join(table)


def paired_table(variant_rows):
    t0 = variant_rows.get("T0", [])
    if not t0:
        return "T0 is not available yet; paired comparisons are pending."
    table = [
        "| Model vs T0 | ΔSR (W/L/T) | ΔSR 95% CI | ΔOSR | ΔSPL | "
        "ΔSPL 95% CI | ΔnDTW | ΔnDTW 95% CI | ΔPremature | ΔTimeout |",
        "|---|---:|---|---:|---:|---|---:|---|---:|---:|",
    ]
    for variant in ORDER:
        if variant == "T0":
            continue
        rows = variant_rows.get(variant, [])
        if not rows:
            continue
        paired = paired_comparison({"T0": t0, variant: rows},
                                   base="T0", other=variant)
        sr = paired["success"]
        spl = paired["spl"]
        ndtw = paired["ndtw"]
        table.append(
            f"| {variant} | {100*sr['delta_rate']:+.2f} pp "
            f"({sr['other_better']}/{sr['base_better']}/{sr['ties']}) | "
            f"[{100*sr['delta_ci95'][0]:+.2f}, {100*sr['delta_ci95'][1]:+.2f}] | "
            f"{100*paired['osr']['delta_rate']:+.2f} pp | "
            f"{spl['delta']:+.4f} ({spl['other_better']}/"
            f"{spl['base_better']}/{spl['ties']}) | "
            f"[{paired['spl_delta_ci95'][0]:+.4f}, "
            f"{paired['spl_delta_ci95'][1]:+.4f}] | "
            f"{ndtw['delta']:+.4f} ({ndtw['other_better']}/"
            f"{ndtw['base_better']}/{ndtw['ties']}) | "
            f"[{paired['ndtw_delta_ci95'][0]:+.4f}, "
            f"{paired['ndtw_delta_ci95'][1]:+.4f}] | "
            f"{100*paired['premature_stop']['delta_rate']:+.2f} pp | "
            f"{100*paired['max_steps_termination']['delta_rate']:+.2f} pp |"
        )
    return "\n".join(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/v83_spatial_stage2/stage2i"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variant_rows = {variant: read_rows(variant) for variant in ORDER}
    summaries = {variant: summarize(rows)
                 for variant, rows in variant_rows.items() if rows}
    paired = {}
    t0 = variant_rows.get("T0", [])
    if t0:
        for variant in ORDER:
            if variant == "T0" or not variant_rows.get(variant):
                continue
            paired[variant] = paired_comparison(
                {"T0": t0, variant: variant_rows[variant]},
                base="T0", other=variant,
            )
    payload = {
        "order": list(ORDER),
        "summaries": summaries,
        "paired_vs_T0": paired,
        "band_paired_vs_T0": band_paired(variant_rows),
        "model_dirs": {variant: str(model_dir(variant))
                       for variant in ORDER},
        "variants": {variant: VARIANTS[variant] for variant in ORDER},
        "final_test_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    markdown = "\n\n".join([
        "# Stage-2i summary (generated)",
        "## Overall",
        overall_table(variant_rows),
        "## Bands",
        band_table(variant_rows),
        "## Paired vs T0",
        paired_table(variant_rows),
        "## Band-wise paired vs T0",
        band_paired_table(variant_rows),
    ])
    (args.output_dir / "summary.md").write_text(markdown + "\n",
                                                encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
