"""Assemble Stage-2d R4 metrics from saved per-rule closed-loop CSVs.

The R4 runner saves `steps/<rule>.csv`, `stops/<rule>.csv` and
`<rule>_per_episode.csv` after every rule. If the process exits before the
final aggregation (for example on an over-strict replay check), this script
rebuilds `metrics.json`, `per_episode.csv`, `risk.csv`, `replay_check.json`
and `provenance.json` without re-running any episode.

It uses the actual per-step Stop flag for hazard/risk (including persistence
confirmation), so no separate recompute is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2d_common import (
    BANDS,
    M0_CKPT,
    VAL_SPECS,
    device_summary,
    file_sha256,
    hazard_from_steps,
    paired_by_band,
    risk_rows,
    rule_from_name,
    stop_event_summary,
    summarize_by_band,
    verify_b0_replay,
    write_csv,
    write_json,
)
from v83_spatial_stage2.stage2d_r4_decision_rules import DEFAULT_RULES


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rules", nargs="+", default=list(DEFAULT_RULES))
    args = parser.parse_args()
    run_dir = args.run_dir
    rules = list(args.rules)
    b0_name = "threshold_0.5"
    if b0_name not in rules:
        raise SystemExit("threshold_0.5 (B0 reference) must be in --rules")

    episodes_by_rule = {
        name: read_rows(run_dir / f"{name}_per_episode.csv") for name in rules
    }
    steps_by_rule = {
        name: read_rows(run_dir / "steps" / f"{name}.csv") for name in rules
    }
    stops_by_rule = {
        name: read_rows(run_dir / "stops" / f"{name}.csv") for name in rules
    }
    replay_check = verify_b0_replay(episodes_by_rule[b0_name])
    write_json(run_dir / "replay_check.json", replay_check)
    if replay_check["outcome_mismatch_count"] != 0:
        raise SystemExit(
            "B0 replay outcome mismatch: "
            f"{replay_check['outcome_mismatch_count']} fields"
        )

    all_episodes = [row for name in rules for row in episodes_by_rule[name]]
    write_csv(run_dir / "per_episode.csv", all_episodes)

    metrics = {}
    paired = {}
    for name in rules:
        summary = summarize_by_band(episodes_by_rule[name])
        threshold = rule_from_name(name).threshold
        metrics[name] = {
            "overall": summary["all"],
            "by_band": summary,
            "stop_events": {
                band: stop_event_summary([
                    row for row in stops_by_rule[name]
                    if band == "all" or str(row["band"]) == band
                ])
                for band in ("all",) + BANDS
            },
            "hazard": {
                band: hazard_from_steps(
                    steps_by_rule[name] if band == "all" else [
                        row for row in steps_by_rule[name]
                        if str(row["band"]) == band
                    ],
                    threshold=threshold,
                )
                for band in ("all",) + BANDS
            },
        }
        if name != b0_name:
            paired[name] = paired_by_band(
                {b0_name: episodes_by_rule[b0_name], name: episodes_by_rule[name]},
                base=b0_name, other=name,
            )
    risk = risk_rows(
        steps_by_rule,
        threshold_map={name: rule_from_name(name).threshold for name in rules},
    )
    write_csv(run_dir / "risk.csv", risk)
    write_json(run_dir / "metrics.json", {
        "rules": rules,
        "b0_reference": b0_name,
        "metrics": metrics,
        "paired_vs_b0": paired,
        "risk": risk,
    })
    write_json(run_dir / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "V8.3 Stage-2d R4 Stop-decision rules (assembled from saved CSVs)",
        "command": sys.argv,
        "m0_checkpoint_sha256": file_sha256(M0_CKPT),
        "val_specs": str(VAL_SPECS),
        "val_specs_sha256": file_sha256(VAL_SPECS),
        "episodes": int(len(episodes_by_rule[b0_name])),
        "rules": [
            {
                "name": name,
                "threshold": rule_from_name(name).threshold,
                "confirm_k": rule_from_name(name).confirm_k,
                "confirm_n": rule_from_name(name).confirm_n,
            }
            for name in rules
        ],
        "device": device_summary(torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )),
        "note": (
            "Frozen M0; every rule re-ran the full closed loop; per-episode "
            "decision state reset; assembled from saved per-step/per-episode "
            "CSVs; Final Test not read."
        ),
    })
    print(json.dumps({
        name: metrics[name]["overall"] for name in rules
    }, indent=2), flush=True)
    print(f"Results: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
