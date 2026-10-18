"""Analyze Stage-2a M0/M1 closed-loop 180-episode results.

Produces overall/band metrics (including mean steps), paired per-band
statistics, and a detailed Long failure table. Uses the paired comparison
implementation from the closed-loop evaluator (paired bootstrap CIs + exact
sign tests).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_training.eval_closedloop_spatial import paired_comparison


BANDS = ("short", "medium", "long")


def _mean(values):
    return float(np.mean(values)) if values else float("nan")


def summarize(rows):
    return {
        "episodes": len(rows),
        "sr": _mean([int(r["success"]) for r in rows]),
        "osr": _mean([int(r["osr"]) for r in rows]),
        "spl": _mean([float(r["spl"]) for r in rows]),
        "ndtw": _mean([float(r["ndtw"]) for r in rows]),
        "premature_stop": _mean([int(r["premature_stop"]) for r in rows]),
        "timeout": _mean([int(r["max_steps_termination"]) for r in rows]),
        "mean_steps": _mean([int(r["steps"]) for r in rows]),
        "mean_min_distance_m": _mean([float(r["min_distance_m"]) for r in rows]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per = list(csv.DictReader(open(args.eval_dir / "per_episode.csv")))
    by_mode = {}
    for r in per:
        by_mode.setdefault(r["mode"], []).append(r)
    if "M0" not in by_mode or "M1" not in by_mode:
        raise SystemExit("Need M0 and M1 rows")
    eval_summary = json.loads((args.eval_dir / "summary.json").read_text())

    table = []
    for mode in ("M0", "M1"):
        rows = by_mode[mode]
        table.append({"mode": mode, "band": "all", **summarize(rows)})
        for band in BANDS:
            sub = [r for r in rows if r["band"] == band]
            table.append({"mode": mode, "band": band, **summarize(sub)})
    with (args.output_dir / "overall_band_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)

    paired = {
        "overall": paired_comparison(by_mode, base="M0", other="M1"),
    }
    for band in BANDS:
        sub = {m: [r for r in by_mode[m] if r["band"] == band] for m in ("M0", "M1")}
        paired[band] = paired_comparison(sub, base="M0", other="M1")
    (args.output_dir / "paired_by_band.json").write_text(
        json.dumps(paired, indent=2), encoding="utf-8"
    )

    m0 = {int(r["episode_index"]): r for r in by_mode["M0"]}
    m1 = {int(r["episode_index"]): r for r in by_mode["M1"]}
    long_rows = []
    for i in sorted(m1):
        a = m0[i]; b = m1[i]
        if b["band"] != "long":
            continue
        outcome = "premature_stop" if int(b["premature_stop"]) else (
            "timeout" if int(b["max_steps_termination"]) else (
                "success" if int(b["success"]) else "other"
            )
        )
        long_rows.append({
            "episode_index": i,
            "seed": b["seed"],
            "M0_success": int(a["success"]),
            "M1_success": int(b["success"]),
            "M0_osr": int(a["osr"]),
            "M1_osr": int(b["osr"]),
            "M0_outcome": "premature_stop" if int(a["premature_stop"]) else (
                "timeout" if int(a["max_steps_termination"]) else "other"
            ),
            "M1_outcome": outcome,
            "M0_steps": int(a["steps"]),
            "M1_steps": int(b["steps"]),
            "M0_min_distance_m": float(a["min_distance_m"]),
            "M1_min_distance_m": float(b["min_distance_m"]),
            "M0_spl": float(a["spl"]),
            "M1_spl": float(b["spl"]),
        })
    with (args.output_dir / "long_failures.csv").open("w", newline="", encoding="utf-8") as f:
        if long_rows:
            w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
            w.writeheader(); w.writerows(long_rows)
    long_summary = {
        "count": len(long_rows),
        "M0_outcomes": {
            "premature_stop": sum(r["M0_outcome"] == "premature_stop" for r in long_rows),
            "timeout": sum(r["M0_outcome"] == "timeout" for r in long_rows),
            "other": sum(r["M0_outcome"] == "other" for r in long_rows),
        },
        "M1_outcomes": {
            "premature_stop": sum(r["M1_outcome"] == "premature_stop" for r in long_rows),
            "timeout": sum(r["M1_outcome"] == "timeout" for r in long_rows),
            "other": sum(r["M1_outcome"] == "other" for r in long_rows),
        },
        "M1_far_premature_stop_gt20m": sum(
            r["M1_outcome"] == "premature_stop" and r["M1_min_distance_m"] > 20.0
            for r in long_rows
        ),
        "M1_timeout": sum(r["M1_outcome"] == "timeout" for r in long_rows),
        "rows": long_rows,
    }
    result = {
        "eval_dir": str(args.eval_dir),
        "metrics": table,
        "paired": paired,
        "long": long_summary,
        "eval_summary_overall": eval_summary.get("summaries"),
    }
    (args.output_dir / "stage2a_analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
