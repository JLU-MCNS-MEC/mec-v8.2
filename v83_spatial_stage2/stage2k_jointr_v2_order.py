"""Training-order and validation-isolation helpers for Stage-2k v2.

The 4000 training episodes are unchanged.  Only their order is rebuilt: a
deterministic round-robin over the 3 scenarios x 3 bands groups, with a
per-group fixed-seed shuffle.  This guarantees short/medium/long coverage from
the start instead of the old short -> medium -> long blocks.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BANDS,
    SCENARIOS,
    load_split_specs,
    load_topology,
)
from v83_spatial_stage2.stage2k_dqn import SEED, dev_validation_specs


GROUP_ORDER = tuple(
    (scenario, band) for scenario in SCENARIOS for band in BANDS
)
RANDOM_ORDER_SEED = int(SEED) + 20261031


def random_training_order(spec_payload, total=3500, seed=RANDOM_ORDER_SEED):
    """Stratified random sample without replacement from Train, then shuffle.

    The sample targets equal scenario x band counts as far as integer
    division allows.  For total=3500 the target is 389 episodes for each of
    the first eight groups and 388 for the last group, giving exact band
    totals 1167 / 1167 / 1166 (Short / Medium / Long).  The combined list is
    shuffled globally, so the band sequence is random rather than a fixed
    round-robin.
    """
    total = int(total)
    if total <= 0:
        raise ValueError("total must be positive")
    groups = {}
    for scenario in SCENARIOS:
        for band in BANDS:
            items = [
                item for item in spec_payload["scenarios"][scenario]["episodes"]
                if item["band"] == band
            ]
            if not items:
                raise RuntimeError(f"Empty train group {scenario}/{band}")
            groups[(scenario, band)] = items
    group_keys = list(GROUP_ORDER)
    base = total // len(group_keys)
    target = {key: base for key in group_keys}
    for index in range(total - base * len(group_keys)):
        target[group_keys[index]] += 1
    rng = np.random.default_rng(int(seed))
    selected = []
    for key in group_keys:
        items = groups[key]
        if len(items) < target[key]:
            raise RuntimeError(
                f"Not enough episodes for {key}: {len(items)} < {target[key]}"
            )
        indices = rng.choice(len(items), size=target[key], replace=False)
        selected.extend(items[int(index)] for index in indices)
    rng.shuffle(selected)
    if len(selected) != total:
        raise RuntimeError(
            f"Random order length {len(selected)} != requested {total}"
        )
    return [item["episode_uid"] for item in selected], target


def stratified_training_order(spec_payload, seed=SEED):
    """Return a deterministic, band/scenario-interleaved UID order."""
    groups = {}
    for group_index, (scenario, band) in enumerate(GROUP_ORDER):
        items = [
            item for item in spec_payload["scenarios"][scenario]["episodes"]
            if item["band"] == band
        ]
        if not items:
            raise RuntimeError(f"Empty train group {scenario}/{band}")
        rng = np.random.default_rng(int(seed) + group_index)
        permutation = rng.permutation(len(items))
        groups[(scenario, band)] = [items[index] for index in permutation]

    order = []
    pointers = {group: 0 for group in GROUP_ORDER}
    remaining = sum(len(items) for items in groups.values())
    while remaining:
        progressed = False
        for group in GROUP_ORDER:
            pointer = pointers[group]
            items = groups[group]
            if pointer < len(items):
                order.append(items[pointer]["episode_uid"])
                pointers[group] = pointer + 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    if len(order) != sum(len(items) for items in groups.values()):
        raise RuntimeError("Stratified order does not cover all episodes")
    return order


def order_statistics(order, lookup):
    """Count scenario/band coverage and every 200-episode window."""
    counts = Counter(
        (lookup[uid]["scenario"], lookup[uid]["band"]) for uid in order
    )
    windows = []
    for start in range(0, len(order), 200):
        chunk = order[start:start + 200]
        if len(chunk) < 200:
            continue
        window_counts = Counter(
            (lookup[uid]["scenario"], lookup[uid]["band"]) for uid in chunk
        )
        windows.append({
            "start": start + 1,
            "end": start + len(chunk),
            "groups_covered": len(window_counts),
            "counts": {
                f"{scenario}/{band}": int(window_counts[(scenario, band)])
                for scenario in SCENARIOS for band in BANDS
            },
        })
    return {
        "episodes": len(order),
        "unique_episodes": len(set(order)),
        "counts": {
            f"{scenario}/{band}": int(counts[(scenario, band)])
            for scenario in SCENARIOS for band in BANDS
        },
        "windows": windows,
        "all_full_windows_cover_9_groups": all(
            item["groups_covered"] == len(GROUP_ORDER) for item in windows
        ),
        "min_groups_in_full_window": min(
            (item["groups_covered"] for item in windows), default=0
        ),
    }


def verify_order(order, spec_payload, require_full=True):
    """Verify the order is a permutation of the frozen train list.

    ``require_full=False`` allows a random subset of the frozen Train list
    (used by the pure-MEC sampling protocol); the order must still contain
    only Train UIDs and no duplicates.
    """
    lookup = {
        item["episode_uid"]: item
        for scenario in SCENARIOS
        for item in spec_payload["scenarios"][scenario]["episodes"]
    }
    expected = list(lookup)
    if require_full and len(order) != len(expected):
        raise RuntimeError(
            f"Order length {len(order)} != frozen train length {len(expected)}"
        )
    if len(set(order)) != len(order):
        raise RuntimeError("Order contains duplicate Episode UIDs")
    if not set(order).issubset(lookup):
        extra = sorted(set(order) - set(expected))[:5]
        raise RuntimeError(
            f"Order contains non-Train Episode UIDs: {extra}"
        )
    if require_full and set(order) != set(expected):
        missing = sorted(set(expected) - set(order))[:5]
        raise RuntimeError(
            f"Order changed the frozen episode set; missing={missing}"
        )
    for uid in order:
        item = lookup[uid]
        if item["episode_uid"] != uid:
            raise RuntimeError("Episode UID mismatch in frozen specs")
    return lookup


def verify_validation_isolation():
    """Verify the 60 Dev validation episodes are disjoint from train."""
    train_payload = load_split_specs("train")
    dev_payload = load_split_specs("dev")
    validation = dev_validation_specs()
    train_uids = {
        item["episode_uid"]
        for scenario in SCENARIOS
        for item in train_payload["scenarios"][scenario]["episodes"]
    }
    validation_uids = {item["episode_uid"] for item in validation}
    overlap = sorted(train_uids & validation_uids)
    by_scenario = Counter(item["scenario"] for item in validation)
    by_band = Counter(
        (item["scenario"], item["band"]) for item in validation
    )
    topology = {
        scenario: int(load_topology(scenario, "dev")["topology_seed"])
        for scenario in SCENARIOS
    }
    train_topology = {
        scenario: int(load_topology(scenario, "train")["topology_seed"])
        for scenario in SCENARIOS
    }
    checks = {
        "count_60": len(validation) == 60,
        "all_dev_split": all(item["split"] == "dev" for item in validation),
        "scenario_20_each": all(
            by_scenario[scenario] == 20 for scenario in SCENARIOS
        ),
        "band_6_10_4_each": all(
            by_band[(scenario, band)] == expected
            for scenario in SCENARIOS
            for band, expected in (("short", 6), ("medium", 10), ("long", 4))
        ),
        "train_overlap_zero": len(overlap) == 0,
        "dev_topology_seeds": topology,
        "train_topology_seeds": train_topology,
        "topology_seeds_disjoint": all(
            topology[scenario] != train_topology[scenario]
            for scenario in SCENARIOS
        ),
        "train_episodes": len(train_uids),
        "validation_episodes": len(validation_uids),
    }
    checks["ok"] = all((
        checks["count_60"], checks["all_dev_split"],
        checks["scenario_20_each"], checks["band_6_10_4_each"],
        checks["train_overlap_zero"], checks["topology_seeds_disjoint"],
    ))
    return checks, validation
