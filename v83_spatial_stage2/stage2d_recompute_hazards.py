"""Recompute Stage-2d hazard/risk fields from saved per-step CSVs.

This is needed when the Stop rule is a persistence rule: the actual decision
is not `probability >= threshold`, so the hazard must use the `stop` flag
stored in each per-step row. Closed-loop outcomes, per-episode CSVs and stop
events are read-only here; only the aggregate hazard/risk fields are rebuilt.

Usage:
  # R4 run
  python v83_spatial_stage2/stage2d_recompute_hazards.py \
    --run-dir runs/v83_spatial_stage2/stage2d/r4_<stamp> --kind r4

  # Calibration run
  python v83_spatial_stage2/stage2d_recompute_hazards.py \
    --run-dir runs/v83_spatial_stage2/stage2d/calibration_<stamp> --kind calibration
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

from v83_spatial_stage2.stage2d_common import (
    BANDS,
    hazard_from_steps,
    risk_rows,
    rule_from_name,
    write_csv,
)


def read_steps(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def hazard_by_band(steps, threshold):
    result = {}
    for band in ("all",) + BANDS:
        subset = (
            steps if band == "all"
            else [row for row in steps if str(row["band"]) == band]
        )
        if subset:
            result[band] = hazard_from_steps(subset, threshold=threshold)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["r4", "calibration"], required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.kind == "r4":
        source = args.run_dir / "metrics.json"
        output = args.output or args.run_dir / "metrics_recomputed.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        rules = payload["rules"]
        thresholds = {
            name: rule_from_name(name).threshold for name in rules
        }
        steps_by_rule = {
            name: read_steps(args.run_dir / "steps" / f"{name}.csv")
            for name in rules
        }
        for name in rules:
            payload["metrics"][name]["hazard"] = hazard_by_band(
                steps_by_rule[name], thresholds[name]
            )
        payload["risk"] = risk_rows(
            steps_by_rule, threshold_map=thresholds
        )
        payload["hazard_recomputed_from_steps"] = True
        payload["hazard_note"] = (
            "Hazard rates use the actual per-step Stop flag (including "
            "persistence confirmation), not probability >= threshold. The "
            "predicted p-sum is a pre-confirmation probability diagnostic."
        )
    else:
        source = args.run_dir / "closedloop_metrics.json"
        output = args.output or args.run_dir / "closedloop_metrics_recomputed.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        names = list(payload["metrics"])
        steps_by_name = {
            name: read_steps(args.run_dir / "steps" / f"{name}.csv")
            for name in names
        }
        threshold = float(payload.get("threshold", 0.5))
        for name in names:
            payload["metrics"][name]["hazard"] = hazard_by_band(
                steps_by_name[name], threshold
            )
        payload["risk"] = risk_rows(
            steps_by_name,
            threshold_map={name: threshold for name in names},
        )
        payload["hazard_recomputed_from_steps"] = True
        payload["hazard_note"] = (
            "Hazard rates use the actual per-step Stop flag; for the "
            "threshold-only calibration variants this equals p >= threshold."
        )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.run_dir / "risk_recomputed.csv", payload["risk"])
    print(f"Wrote {output}")
    print(f"Wrote {args.run_dir / 'risk_recomputed.csv'}")


if __name__ == "__main__":
    main()
