"""Compare DQN action sequences with the baseline action sequences.

For every Dev Val 180 episode, compare the DQN's per-step navigation action
sequence with Min-Latency, Max-Quality and Heuristic.  If the DQN selects a
slower MEC but the emitted navigation action is unchanged, the extra E2E
latency buys no navigation quality.  Read-only; no training and no test data.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import BASELINE_DIR, DQN_DIR

EVAL_DIR = DQN_DIR / "dev180_eval"
BASELINE_STEPS = BASELINE_DIR / "dev" / "steps.csv"
BASELINE_EPISODES = BASELINE_DIR / "dev" / "per_episode.csv"
STRATEGIES = ("min_latency", "max_quality", "heuristic")


def load_step_actions(path, strategy_filter=None):
    grouped = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            strategy = row.get("strategy", "dqn")
            if strategy_filter is not None and strategy != strategy_filter:
                continue
            if strategy_filter is None and strategy != "dqn":
                continue
            grouped[row["episode_uid"]].append(
                (int(row["step"]), int(row["action"]))
            )
    return {
        uid: [action for _, action in sorted(steps)]
        for uid, steps in grouped.items()
    }


def load_baseline_episodes(path):
    rows = defaultdict(dict)
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["episode_uid"]][row["strategy"]] = row
    return rows


def compare(dqn_actions, baseline_actions):
    n = min(len(dqn_actions), len(baseline_actions))
    if n == 0:
        return {
            "dqn_steps": len(dqn_actions),
            "baseline_steps": len(baseline_actions),
            "common_steps": 0,
            "prefix_agreement": None,
            "identical_prefix": False,
            "identical_sequence": len(dqn_actions) == len(baseline_actions),
        }
    agreement = float(np.mean(np.asarray(
        dqn_actions[:n], dtype=np.int64
    ) == np.asarray(baseline_actions[:n], dtype=np.int64)))
    prefix = 0
    for left, right in zip(dqn_actions, baseline_actions):
        if left != right:
            break
        prefix += 1
    return {
        "dqn_steps": len(dqn_actions),
        "baseline_steps": len(baseline_actions),
        "common_steps": n,
        "prefix_agreement": agreement,
        "identical_prefix": prefix == n,
        "identical_sequence": (
            len(dqn_actions) == len(baseline_actions)
            and prefix == len(dqn_actions)
        ),
    }


def main():
    dqn_actions = load_step_actions(EVAL_DIR / "steps.csv")
    baseline_actions = {}
    for strategy in STRATEGIES:
        baseline_actions[strategy] = load_step_actions(
            BASELINE_STEPS, strategy_filter=strategy
        )
    baseline_episodes = load_baseline_episodes(BASELINE_EPISODES)
    dqn_episodes = {}
    with (EVAL_DIR / "per_episode.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dqn_episodes[row["episode_uid"]] = row

    output = {
        "episodes": len(dqn_actions),
        "strategies": {},
        "note": (
            "Action sequences are compared by step index.  Different "
            "trajectories can diverge, so agreement is descriptive, not a "
            "causal estimate.  Identical sequences mean the MEC choice did "
            "not change the emitted navigation action."
        ),
    }
    for strategy in STRATEGIES:
        records = []
        for uid, actions in dqn_actions.items():
            if uid not in baseline_actions[strategy]:
                continue
            item = compare(actions, baseline_actions[strategy][uid])
            item["episode_uid"] = uid
            dqn_row = dqn_episodes.get(uid, {})
            base_row = baseline_episodes.get(uid, {}).get(strategy, {})
            for key in ("ndtw", "mean_e2e_ms", "joint_reward"):
                item[f"dqn_{key}"] = (
                    float(dqn_row[key]) if key in dqn_row else None
                )
                item[f"baseline_{key}"] = (
                    float(base_row[key]) if key in base_row else None
                )
            records.append(item)
        identical = [r for r in records if r["identical_sequence"]]
        identical_prefix = [r for r in records if r["identical_prefix"]]
        agreement = np.asarray(
            [
                r["prefix_agreement"] for r in records
                if r["prefix_agreement"] is not None
            ],
            dtype=np.float64,
        )
        identical_ndtw_delta = [
            r["dqn_ndtw"] - r["baseline_ndtw"] for r in identical
            if r["dqn_ndtw"] is not None and r["baseline_ndtw"] is not None
        ]
        identical_e2e_delta = [
            r["dqn_mean_e2e_ms"] - r["baseline_mean_e2e_ms"]
            for r in identical
            if r["dqn_mean_e2e_ms"] is not None
            and r["baseline_mean_e2e_ms"] is not None
        ]
        output["strategies"][strategy] = {
            "episodes_compared": len(records),
            "identical_sequence_episodes": len(identical),
            "identical_prefix_episodes": len(identical_prefix),
            "mean_prefix_agreement": (
                float(agreement.mean()) if len(agreement) else None
            ),
            "median_prefix_agreement": (
                float(np.median(agreement)) if len(agreement) else None
            ),
            "identical_sequence_ndtw_delta": (
                float(np.mean(identical_ndtw_delta))
                if identical_ndtw_delta else None
            ),
            "identical_sequence_e2e_delta_ms": (
                float(np.mean(identical_e2e_delta))
                if identical_e2e_delta else None
            ),
        }
    path = EVAL_DIR / "action_agreement.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print("WROTE", path)


if __name__ == "__main__":
    main()
