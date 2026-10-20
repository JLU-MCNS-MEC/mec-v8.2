"""Verify Dev Val 180 scenario coverage for baselines and DQN checkpoints.

Reads per-episode CSVs, the frozen Dev specs and the Dev topology metadata.
Checks that every policy has exactly 60 episodes in each of the three
deployment scenarios, that episode IDs/seeds match the frozen specs, and that
all policies cover the same 180 episodes.  Read-only; no missing run is filled.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BASELINE_DIR,
    DQN_DIR,
    SCENARIOS,
    load_split_specs,
    load_topology,
)

OUTPUT_DIR = DQN_DIR / "scenario_coverage_audit"
BASELINE_PER_EPISODE = BASELINE_DIR / "dev" / "per_episode.csv"
DQN_PER_EPISODE = DQN_DIR / "dev180_eval" / "per_episode.csv"
DQN_LAST_PER_EPISODE = DQN_DIR / "dev180_eval_last" / "per_episode.csv"
STRATEGIES = ("random", "min_latency", "max_quality", "heuristic", "oracle")
DQN_LABELS = ("DQN@800", "DQN@2000")


def read_rows(path):
    if not Path(path).is_file():
        return None
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_rows(rows, label):
    by_scenario = Counter(row["scenario"] for row in rows)
    uids = {row["episode_uid"] for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    return {
        "label": label,
        "episodes": len(rows),
        "by_scenario": {
            scenario: int(by_scenario.get(scenario, 0))
            for scenario in SCENARIOS
        },
        "unique_episode_uids": len(uids),
        "unique_seeds": len(seeds),
        "episode_uids": uids,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spec_payload = load_split_specs("dev")
    spec = {
        item["episode_uid"]: item
        for scenario in SCENARIOS
        for item in spec_payload["scenarios"][scenario]["episodes"]
    }
    topology_meta = {}
    for scenario in SCENARIOS:
        topology = load_topology(scenario, "dev")
        topology_meta[scenario] = {
            "topology_seed": int(topology["topology_seed"]),
            "scenario": topology.get("scenario"),
            "split": topology.get("split"),
            "nodes": len(topology["nodes"]),
            "final_test_read": bool(topology.get("final_test_read", False)),
        }

    baseline_rows = read_rows(BASELINE_PER_EPISODE)
    dqn_rows = read_rows(DQN_PER_EPISODE)
    dqn_last_rows = read_rows(DQN_LAST_PER_EPISODE)

    groups = {}
    if baseline_rows is not None:
        for strategy in STRATEGIES:
            rows = [row for row in baseline_rows if row["strategy"] == strategy]
            groups[strategy] = summarize_rows(rows, strategy)
    if dqn_rows is not None:
        groups[DQN_LABELS[0]] = summarize_rows(dqn_rows, DQN_LABELS[0])
    if dqn_last_rows is not None:
        groups[DQN_LABELS[1]] = summarize_rows(dqn_last_rows, DQN_LABELS[1])

    expected_uids = set(spec)
    output = {
        "version": "stage2k-scenario-coverage-audit-v1",
        "dev_spec_episodes": len(spec),
        "topology": topology_meta,
        "groups": {},
        "checks": [],
    }
    reference_uids = None
    for label, item in groups.items():
        uids = item["episode_uids"]
        scenario_mismatches = []
        seed_mismatches = []
        for uid in uids:
            if uid not in spec:
                scenario_mismatches.append({"episode_uid": uid, "error": "not in dev specs"})
                continue
            expected = spec[uid]
            # The scenario/seed fields are checked below from the raw rows.
        if label in STRATEGIES:
            raw = [
                row for row in baseline_rows
                if row["strategy"] == label
            ]
        elif label == DQN_LABELS[0]:
            raw = dqn_rows
        else:
            raw = dqn_last_rows
        for row in raw:
            expected = spec.get(row["episode_uid"])
            if expected is None:
                seed_mismatches.append({
                    "episode_uid": row["episode_uid"], "error": "not in dev specs"
                })
                continue
            if row["scenario"] != expected["scenario"]:
                scenario_mismatches.append({
                    "episode_uid": row["episode_uid"],
                    "csv": row["scenario"],
                    "spec": expected["scenario"],
                })
            if int(row["seed"]) != int(expected["seed"]):
                seed_mismatches.append({
                    "episode_uid": row["episode_uid"],
                    "csv_seed": int(row["seed"]),
                    "spec_seed": int(expected["seed"]),
                })
        missing = sorted(expected_uids - uids)
        extra = sorted(uids - expected_uids)
        if reference_uids is None:
            reference_uids = uids
        same_as_reference = (uids == reference_uids)
        output["groups"][label] = {
            "episodes": item["episodes"],
            "by_scenario": item["by_scenario"],
            "unique_episode_uids": item["unique_episode_uids"],
            "unique_seeds": item["unique_seeds"],
            "missing_episode_uids": missing,
            "extra_episode_uids": extra,
            "scenario_mismatches": scenario_mismatches,
            "seed_mismatches": seed_mismatches,
            "same_episode_set_as_reference": same_as_reference,
            "complete": (
                item["episodes"] == 180
                and all(item["by_scenario"][s] == 60 for s in SCENARIOS)
                and not missing
                and not extra
                and not scenario_mismatches
                and not seed_mismatches
                and same_as_reference
            ),
        }
    output["reference_group"] = next(iter(groups), None)
    output["all_complete"] = all(
        item["complete"] for item in output["groups"].values()
    )
    path = OUTPUT_DIR / "scenario_coverage.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                    encoding="utf-8")

    print("Topology metadata:")
    for scenario, item in topology_meta.items():
        print(f"  {scenario}: seed={item['topology_seed']} "
              f"split={item['split']} nodes={item['nodes']}")
    print("\nCoverage:")
    print(f"{'policy':14s} {'balanced':>9s} {'quality':>9s} {'tradeoff':>9s} {'total':>6s} complete")
    for label, item in output["groups"].items():
        by = item["by_scenario"]
        print(f"{label:14s} {by['balanced']:9d} {by['quality_rich']:9d} "
              f"{by['tradeoff_stress']:9d} {item['episodes']:6d} {item['complete']}")
    print("\nall_complete:", output["all_complete"])
    print("WROTE", path)


if __name__ == "__main__":
    main()
