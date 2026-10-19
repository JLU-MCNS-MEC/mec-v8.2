"""Finalize Stage-2c diagnostics post-processing from saved CSVs.

The main diagnostics script writes rollout CSVs before its replay sanity check.
If that check fails (or is interrupted), this script recomputes the replay check,
the paired-vs-B0 table and the combined JSON from the already persisted CSVs,
without rerunning the expensive rollouts.

Usage:
  python v83_spatial_stage2/stage2c_finalize_diagnostics.py \
    --diagnostics-dir runs/v83_spatial_stage2/stage2c/diag_<stamp>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2c_diagnostics import (
    STAGE2A_EVAL,
    STAGE2C,
    _load_episode_rows,
    _paired_table,
)


def _rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    diag = args.diagnostics_dir
    if not (diag / "rollout_per_episode.csv").is_file():
        raise SystemExit("rollout_per_episode.csv is missing; run the full diagnostics first")
    archived = {
        "B0": _load_episode_rows(STAGE2A_EVAL / "per_episode.csv", mode="M0"),
        "B1": _load_episode_rows(STAGE2C / "B1_per_episode.csv"),
        "B2": _load_episode_rows(STAGE2C / "B2_per_episode.csv"),
    }
    replay = {
        name: [r for r in _rows(diag / "rollout_per_episode.csv") if r["variant"] == name]
        for name in ("B0", "B1", "B2")
    }
    replay_check = {}
    for name in archived:
        a = {int(r["episode_index"]): r for r in archived[name]}
        b = {int(r["episode_index"]): r for r in replay[name]}
        replay_timeout = (
            "timeout" if "timeout" in next(iter(b.values())) else "max_steps_termination"
        )
        diffs = {
            "success": sum(int(a[i]["success"]) != int(b[i]["success"]) for i in sorted(a)),
            "osr": sum(int(a[i]["osr"]) != int(b[i]["osr"]) for i in sorted(a)),
            "premature_stop": sum(
                int(a[i]["premature_stop"]) != int(b[i]["premature_stop"]) for i in sorted(a)
            ),
            "max_steps_termination": sum(
                int(a[i]["max_steps_termination"]) != int(b[i][replay_timeout])
                for i in sorted(a)
            ),
        }
        replay_check[name] = {"episodes": len(a), "mismatches": diffs}
    paired = _paired_table(archived)
    with (diag / "paired_vs_B0.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    (diag / "replay_check.json").write_text(
        json.dumps(replay_check, indent=2), encoding="utf-8"
    )
    offline = json.loads((diag / "offline_arrival_metrics.json").read_text(encoding="utf-8"))
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": args.threshold,
        "episodes": len(replay["B0"]),
        "replay_check": replay_check,
        "offline_arrival_metrics": offline["variants"],
        "rollout_episode_summary": _rows(diag / "rollout_episode_summary.csv"),
        "rollout_stop_summary": _rows(diag / "rollout_stop_summary.csv"),
        "rollout_distance_bins": _rows(diag / "rollout_distance_bins.csv"),
        "note": (
            "Goal visibility and true distance are offline diagnostic labels only; "
            "they are never model inputs. Dev Val 180 only; Final Test sealed."
        ),
    }
    (diag / "stage2c_diagnostics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(replay_check, indent=2), flush=True)
    print(f"Finalized: {diag}", flush=True)


if __name__ == "__main__":
    main()
