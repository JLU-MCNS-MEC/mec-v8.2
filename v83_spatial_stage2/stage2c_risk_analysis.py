"""Decompose closed-loop premature Stop into per-decision hazard.

Reads the Stage-2c diagnostics outputs and asks whether a low offline
per-sample false-positive rate is still enough to produce frequent episode
failures: every closed-loop episode makes many far-distance stop decisions, so
the episode failure probability is driven by 1 - prod(1 - per-step FPR), not by
the per-sample FPR alone.

This script is analysis-only. It does not read Final Test or modify weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

BANDS = ("all", "short", "medium", "long")
DISTANCE_BINS = ("20_40m", "40_60m", "60_100m", "100_200m", "200_infm")


def _rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    offline = _rows(args.diagnostics_dir / "offline_distance_bins.csv")
    rollout = _rows(args.diagnostics_dir / "rollout_distance_bins.csv")
    steps = _rows(args.diagnostics_dir / "rollout_steps.csv")
    events = _rows(args.diagnostics_dir / "rollout_stop_events.csv")
    offline_index = {
        (r["variant"], r["band"], r["distance_bin"]): r
        for r in offline if r["visibility"] == "all"
    }
    rollout_index = {
        (r["variant"], r["band"], r["distance_bin"]): r
        for r in rollout if r["visibility"] == "all"
    }
    episodes_by = {}
    for row in steps:
        episodes_by.setdefault((row["variant"], row["band"]), set()).add(row["episode_index"])
        episodes_by.setdefault((row["variant"], "all"), set()).add(row["episode_index"])
    out = []
    for variant in ("B0", "B1", "B2"):
        for band in BANDS:
            episodes = episodes_by.get((variant, band), set())
            if not episodes:
                continue
            exposure_sum = 0.0
            expected_false = 0.0
            for bin_name in DISTANCE_BINS:
                roll = rollout_index.get((variant, band, bin_name))
                off = offline_index.get((variant, band, bin_name))
                if roll is None:
                    continue
                exposure = float(roll["n"])
                rate = float(roll["pred_stop_rate"])
                exposure_sum += exposure
                expected_false += exposure * rate
            per_episode_exposure = exposure_sum / len(episodes)
            expected_per_episode = expected_false / len(episodes)
            observed = [
                e for e in events
                if e["variant"] == variant and (band == "all" or e["band"] == band)
            ]
            observed_stops = len(observed)
            observed_premature = sum(float(e["distance_m"]) > 20.0 for e in observed)
            first_stop_distances = sorted(float(e["distance_m"]) for e in observed)
            far40_num = far40_den = 0.0
            for bin_name in ("40_60m", "60_100m", "100_200m", "200_infm"):
                off = offline_index.get((variant, band, bin_name))
                if off is not None:
                    far40_num += float(off["n"]) * float(off["pred_positive_rate"])
                    far40_den += float(off["n"])
            out.append({
                "variant": variant, "band": band, "episodes": len(episodes),
                "far_steps_exposure": int(exposure_sum),
                "far_steps_per_episode": per_episode_exposure,
                "expected_false_stop_decisions_per_episode": expected_per_episode,
                "probability_any_false_stop_approx": (
                    1.0 - math.exp(-expected_per_episode) if expected_per_episode else 0.0
                ),
                "observed_stops": observed_stops,
                "observed_premature_stops": observed_premature,
                "observed_premature_rate": (
                    observed_premature / len(episodes) if episodes else 0.0
                ),
                "observed_stop_distance_median_m": (
                    first_stop_distances[len(first_stop_distances) // 2]
                    if first_stop_distances else float("nan")
                ),
                "offline_far40_fpr": far40_num / far40_den if far40_den else float("nan"),
            })
    with (args.output_dir / "risk_decomposition.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    (args.output_dir / "risk_decomposition.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
