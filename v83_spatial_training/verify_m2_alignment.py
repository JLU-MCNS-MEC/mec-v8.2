"""Independent alignment audit for the cached M2 temporal features.

Checks, for every cached sample of Train/Val/Val_hard:

  * validity is 1 only for natural samples with step_id > 0 whose previous
    archived label (same episode, step_id-1) is LEFT or RIGHT;
  * valid temporal24 equals (1-carry)*coherent24 + carry*warped_previous;
  * valid innovation24 equals coherent24 - warped_previous;
  * invalid samples keep temporal24 == coherent24 and innovation24 == 0;
  * a token-level spot check reproduces cache baseline/coherent from RGB tokens
    and shows the previous coherent map comes from the step_id-1 frame.

Only observable archived data is read. No cache is written or regenerated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if __package__:
    from .spatial_cache import CARRY
else:
    from spatial_cache import CARRY

from env import FORWARD, LEFT, RIGHT
from visual_grounding_v83 import static_grounding, yaw_warp_previous


SPLITS = {
    "train": Path("token_dataset_train"),
    "val": Path("token_dataset_val"),
    "val_hard": Path("token_dataset_val_hard"),
}
TURN_ACTIONS = (LEFT, RIGHT)


def shard_files(folder):
    return sorted(Path(folder).glob("*.pt"))


def stream_archived_labels(folder):
    """Return key metadata and key -> (shard_path, row) without retaining shards."""
    labels = {}
    sample_types = {}
    locations = {}
    shards = shard_files(folder)
    for path in shards:
        data = torch.load(path, map_location="cpu", weights_only=True)
        episode = data["episode_ids"].long().tolist()
        step = data["step_ids"].long().tolist()
        label = data["labels"].long().tolist()
        sample_type = data["sample_type"].long().tolist()
        for row, (e, s, y, t) in enumerate(zip(episode, step, label, sample_type)):
            key = (int(e), int(s))
            if key in labels:
                raise RuntimeError(f"Duplicate archived key {key} in {folder}")
            labels[key] = int(y)
            sample_types[key] = int(t)
            locations[key] = (path, row)
    return labels, sample_types, locations, len(shards)


def align_split(split, folder, cache_path, carry, provenance_samples):
    cache = np.load(cache_path)
    baseline = cache["baseline"]
    coherent = cache["coherent"]
    temporal = cache["temporal"]
    innovation = cache["innovation"]
    valid = cache["valid"].astype(np.uint8)
    episode = cache["episode_id"].astype(np.int64)
    step = cache["step_id"].astype(np.int64)
    keys = list(zip(episode.tolist(), step.tolist()))
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"Duplicate cache keys in {cache_path}")
    key_index = {k: i for i, k in enumerate(keys)}

    labels, sample_types, locations, n_shards = stream_archived_labels(folder)
    cache_keys = set(keys)
    archive_keys = set(labels)
    missing_in_archive = sorted(cache_keys - archive_keys)[:5]
    missing_in_cache = sorted(archive_keys - cache_keys)[:5]
    if missing_in_archive or missing_in_cache:
        raise RuntimeError(
            f"{split}: cache/archive key mismatch; "
            f"missing_in_archive={missing_in_archive}, "
            f"missing_in_cache={missing_in_cache}"
        )

    # Rebuild the expected validity mask from archived labels only.
    expected_valid = np.zeros(len(keys), dtype=np.uint8)
    prev_indices = np.full(len(keys), -1, dtype=np.int64)
    counts = {"hard": 0, "first_frame": 0, "forward_previous": 0,
              "missing_previous": 0, "turn_previous": 0, "other_previous": 0}
    for i, (e, s) in enumerate(keys):
        if sample_types[(e, s)] != 0:
            counts["hard"] += 1
            continue
        if s <= 0:
            counts["first_frame"] += 1
            continue
        prev_key = (e, s - 1)
        prev = key_index.get(prev_key)
        if prev is None:
            counts["missing_previous"] += 1
            continue
        prev_indices[i] = prev
        action = labels.get(prev_key)
        if action == FORWARD:
            counts["forward_previous"] += 1
        elif action in TURN_ACTIONS:
            expected_valid[i] = 1
            counts["turn_previous"] += 1
        else:
            counts["other_previous"] += 1

    mask_mismatch = np.flatnonzero(valid != expected_valid)
    result = {
        "split": split,
        "samples": int(len(keys)),
        "valid_cache": int(valid.sum()),
        "valid_expected": int(expected_valid.sum()),
        "valid_mask_mismatches": int(mask_mismatch.size),
        "valid_mask_mismatch_examples": [
            {"key": list(keys[i]), "cache": int(valid[i]),
             "expected": int(expected_valid[i])}
            for i in mask_mismatch[:10]
        ],
        "validity_reason_counts": counts,
        "cached_temporal_invalid_not_equal_coherent": int(
            sum(1 for i in np.flatnonzero(expected_valid == 0)
                if not np.array_equal(temporal[i], coherent[i]))
        ),
        "cached_innovation_invalid_nonzero": int(
            sum(1 for i in np.flatnonzero(expected_valid == 0)
                if np.any(innovation[i] != 0))
        ),
    }

    # Recompute turn warps from the previous cached coherent map.
    valid_indices = np.flatnonzero(expected_valid == 1)
    if valid_indices.size:
        cur = coherent[valid_indices].astype(np.float32)
        prev = coherent[prev_indices[valid_indices]].astype(np.float32)
        actions = np.asarray(
            [labels[tuple(keys[int(prev_indices[i])])] for i in valid_indices],
            dtype=np.int64,
        )
        warped, warp_valid = yaw_warp_previous(
            torch.from_numpy(prev), torch.from_numpy(actions)
        )
        # The cache builder casts the warped prior to float16 before the
        # temporal/innovation arithmetic. Emulate that storage precision so
        # this audit checks the exact cached values, not just a float32 ideal.
        warped = warped.numpy().astype(np.float16).astype(np.float32)
        expected_temporal = ((1.0 - carry) * cur + carry * warped).astype(np.float16)
        expected_innovation = (cur - warped).astype(np.float16)
        temporal_error = np.abs(
            temporal[valid_indices].astype(np.float16).astype(np.float32)
            - expected_temporal.astype(np.float32)
        )
        innovation_error = np.abs(
            innovation[valid_indices].astype(np.float16).astype(np.float32)
            - expected_innovation.astype(np.float32)
        )
        result.update({
            "valid_warp_mismatches": int((~warp_valid.numpy()).sum()),
            "temporal_max_abs_error": float(temporal_error.max()),
            "temporal_mean_abs_error": float(temporal_error.mean()),
            "innovation_max_abs_error": float(innovation_error.max()),
            "innovation_mean_abs_error": float(innovation_error.mean()),
        })
    else:
        result.update({
            "valid_warp_mismatches": 0,
            "temporal_max_abs_error": 0.0,
            "temporal_mean_abs_error": 0.0,
            "innovation_max_abs_error": 0.0,
            "innovation_mean_abs_error": 0.0,
        })

    # Token-level provenance spot check: recompute static grounding from RGB.
    rng = np.random.default_rng(20261018)
    candidates = np.arange(len(keys), dtype=np.int64)
    pick = rng.choice(
        candidates, size=min(int(provenance_samples), candidates.size), replace=False
    )
    by_shard = {}
    for i in pick:
        e, s = keys[int(i)]
        path, row = locations[(e, s)]
        by_shard.setdefault(str(path), []).append((int(i), (e, s), row))
    max_baseline_error = 0.0
    max_coherent_error = 0.0
    max_previous_error = 0.0
    checked = 0
    for shard_path, items in by_shard.items():
        data = torch.load(shard_path, map_location="cpu", weights_only=True)
        tokens = data["tokens"].float()
        goal = data["goal_tokens"].float()
        for i, key, row in items:
            b, c = static_grounding(tokens[row:row + 1], goal[row:row + 1])
            max_baseline_error = max(
                max_baseline_error,
                float(np.abs(b[0].numpy() - baseline[i].astype(np.float32)).max()),
            )
            max_coherent_error = max(
                max_coherent_error,
                float(np.abs(c[0].numpy() - coherent[i].astype(np.float32)).max()),
            )
            prev = prev_indices[i]
            if prev >= 0:
                prev_key = keys[int(prev)]
                prev_path, prev_row = locations[prev_key]
                if prev_path != Path(shard_path):
                    prev_data = torch.load(prev_path, map_location="cpu", weights_only=True)
                    prev_tokens = prev_data["tokens"].float()
                    prev_goal = prev_data["goal_tokens"].float()
                else:
                    prev_tokens, prev_goal = tokens, goal
                _, prev_c = static_grounding(
                    prev_tokens[prev_row:prev_row + 1], prev_goal[prev_row:prev_row + 1]
                )
                max_previous_error = max(
                    max_previous_error,
                    float(np.abs(prev_c[0].numpy() - coherent[prev].astype(np.float32)).max()),
                )
            checked += 1
    result.update({
        "provenance_samples": checked,
        "provenance_max_baseline_abs_error": max_baseline_error,
        "provenance_max_coherent_abs_error": max_coherent_error,
        "provenance_max_previous_coherent_abs_error": max_previous_error,
    })
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "val_hard"])
    p.add_argument("--provenance-samples", type=int, default=12)
    p.add_argument("--output", type=Path, default=None)
    # Cached maps are float16. One ULP near magnitude 8 is ~0.0039, so the
    # audit allows 5e-3 while the mean errors are ~1e-7.
    p.add_argument("--tolerance", type=float, default=5e-3)
    args = p.parse_args()

    results = {}
    failures = []
    for split in args.splits:
        if split not in SPLITS:
            raise SystemExit(f"Unknown split {split}")
        result = align_split(
            split, SPLITS[split], args.cache_dir / f"{split}.npz",
            CARRY, args.provenance_samples,
        )
        results[split] = result
        for key in ("valid_mask_mismatches",
                    "cached_temporal_invalid_not_equal_coherent",
                    "cached_innovation_invalid_nonzero",
                    "valid_warp_mismatches"):
            if result[key]:
                failures.append(f"{split}.{key}={result[key]}")
        for key in ("temporal_max_abs_error", "innovation_max_abs_error",
                    "provenance_max_baseline_abs_error",
                    "provenance_max_coherent_abs_error",
                    "provenance_max_previous_coherent_abs_error"):
            if result[key] > args.tolerance:
                failures.append(f"{split}.{key}={result[key]:.6f}")
        print(json.dumps(result, indent=2), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("ALIGNMENT_OK" if not failures else "ALIGNMENT_FAILED " + "; ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
