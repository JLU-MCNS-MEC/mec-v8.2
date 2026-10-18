"""Build a fresh held-out closed-loop validation specification (NO token export).

Used only for choosing stop thresholds and checkpoint candidates. The historical
final TEST episode list remains untouched and must not be tuned against.

No GPU, torch inference, or result-directory deletion. Existing output refuses
replacement. Task eligibility matches the V8.2 oracle-trajectory selection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config import NETWORK, TRAIN
from env import ToyUAVEnv
from token_data import _inspect_oracle_candidate, _trajectory_band_schedule


DEFAULT_ARCHIVE = Path("runs/archive_pycharm_v82_20261018")
DEFAULT_OUTPUT = Path("runs/v83_validation/closedloop_val_specs.json")
EXCLUDE_SPLITS = (
    "token_dataset_train", "token_dataset_val",
    "token_dataset_val_hard", "token_dataset_test",
)


def excluded_trajectory_seeds(archive_root: Path):
    seeds = set()
    source_files = []
    for name in EXCLUDE_SPLITS:
        stats_path = archive_root / name / "dataset_stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Required source split absent: {stats_path}. "
                "Cannot certify held-out episode seeds."
            )
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        for record in stats.get("trajectories", []):
            seeds.add(int(record["seed"]))
        source_files.append(str(stats_path))
    if not seeds:
        raise ValueError("Training/validation/test stats had no trajectory seeds")
    return seeds, source_files


def generate_specs(num_episodes, seed_start, excluded):
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    schedule, band_counts = _trajectory_band_schedule(num_episodes, seed_start)
    env = ToyUAVEnv()
    candidate_seed = int(seed_start)
    records = []
    rejections = Counter()

    for episode_id, (band_id, band) in enumerate(schedule):
        accepted = None
        for _ in range(int(TRAIN.trajectory_max_candidate_attempts)):
            seed = candidate_seed
            candidate_seed += 1
            if seed in excluded:
                rejections["duplicate_split_seed"] += 1
                continue
            info = _inspect_oracle_candidate(env, seed, band)
            if info["accepted"]:
                accepted = info
                break
            rejections[str(info["reason"])] += 1
        if accepted is None:
            raise RuntimeError(
                f"No accepted episode for episode_id={episode_id}, "
                f"band={band[0]}, after the allowed attempts"
            )
        records.append({
            "band": accepted["band"],
            "band_id": int(band_id),
            "seed": int(accepted["seed"]),
            "steps": int(accepted["steps"]),
            "start_goal_distance_m": float(accepted["start_goal_distance_m"]),
            "network_layout_seed": (
                int(NETWORK.layout_seed_base) + int(seed_start)*17 + episode_id
            ),
        })
    assert len(set(r["seed"] for r in records)) == len(records)
    assert not set(r["seed"] for r in records).intersection(excluded)
    return records, dict(rejections), band_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-episodes", type=int, default=60)
    parser.add_argument("--seed-start", type=int, default=900000)
    args = parser.parse_args()
    if args.seed_start < 0:
        parser.error("seed-start must be nonnegative")
    out = args.output
    if out.exists():
        raise SystemExit(
            f"Refusing to overwrite existing validation specification: {out}"
        )
    excluded, sources = excluded_trajectory_seeds(args.archive_root)
    print(f"Loaded {len(excluded)} excluded episode seeds from {len(sources)} splits")
    records, rejected, counts = generate_specs(
        args.num_episodes, args.seed_start, excluded
    )
    doc = {
        "version": "v8.2-input-redesign",
        "purpose": "closedloop-validation-v83",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "num_episodes": args.num_episodes,
        "seed_start": args.seed_start,
        "holdout_source_stats": sources,
        "split_is_final_test": False,
        "trajectory_mix_counts": counts,
        "rejected_candidates": rejected,
        "trajectories": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # exclusive creation; avoid accidental overwrite by another process.
    with out.open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, ensure_ascii=False)
    print(f"Generated {len(records)} held-out closed-loop tasks: {out}")
    print("Does not generate RGB tokens, does not touch checkpoints or existing datasets.")


if __name__ == "__main__":
    main()
