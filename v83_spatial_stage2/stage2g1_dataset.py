"""Build the Stage-2g.1 combined near/far probe dataset.

Combines:
  * existing Stage-2g Train/Dev-Val current features and labels (far states);
  * the new near-target supplement (1,600 Train + 400 Dev-Val matched pairs);
  * actual natural-state history from previous steps of the same episode;
  * synthetic RIGHT-turn history for the isolated hard states;
  * the supplement's logged 3-step RIGHT-turn history.

The P1 model uses only observable current + history features. True distance,
visibility and projection labels are kept in separate arrays for evaluation.
States without a usable history are marked `history_valid=0` and excluded from
the P0/P1 common comparison (they remain in the file for P0 analysis).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2g1_common import (  # noqa: E402
    COMPACT_DIM,
    COMPACT_FEATURE_NAMES,
    HISTORY_FRAMES,
    build_far_history,
    compact_indices,
    load_action_map,
)


def build(args):
    far = np.load(args.far_features)
    supp = np.load(args.supplement)
    hard = np.load(args.hard_history)
    names = far["feature_names"]
    supp_names = supp["feature_names"]
    if [str(x) for x in names] != [str(x) for x in supp_names]:
        raise RuntimeError("Supplement feature names do not match Stage-2g")
    compact_idx = compact_indices(names)

    n_far = len(far["episode_id"])
    n_supp = len(supp["episode_id"])
    action_map = load_action_map(args.data_dir)
    far_history = build_far_history(
        {
            "features": far["features"],
            "episode_id": far["episode_id"],
            "step_id": far["step_id"],
            "sample_type": far["sample_type"],
        },
        action_map,
        compact_idx,
    )

    # Merge synthetic hard history over the invalid hard rows.
    hard_key_to_index = {
        (int(e), int(s)): i
        for i, (e, s) in enumerate(zip(
            hard["episode_id"], hard["step_id"]
        ))
    }
    far_keys = list(zip(
        far["episode_id"].tolist(), far["step_id"].tolist()
    ))
    far_key_to_index = {key: i for i, key in enumerate(far_keys)}
    hard_found = 0
    for key, hard_index in hard_key_to_index.items():
        far_index = far_key_to_index.get(key)
        if far_index is None:
            raise RuntimeError(f"Hard history key not in far features: {key}")
        far_history["history_features"][far_index] = hard[
            "history_features"
        ][hard_index]
        far_history["history_actions"][far_index] = hard[
            "history_actions"
        ][hard_index]
        far_history["history_dt"][far_index] = hard["history_dt"][hard_index]
        far_history["history_valid"][far_index] = 1
        far_history["history_count"][far_index] = HISTORY_FRAMES
        hard_found += 1
    if hard_found != len(hard["episode_id"]):
        raise RuntimeError(
            f"Hard history merge {hard_found} != {len(hard['episode_id'])}"
        )

    history_origin = np.full(n_far + n_supp, -1, dtype=np.int8)
    history_valid = np.zeros(n_far + n_supp, dtype=np.int8)
    origin_far = np.full(n_far, -1, dtype=np.int8)
    origin_far[far_history["history_valid"] > 0] = 0  # natural actual
    hard_keys = set(hard_key_to_index)
    for index, key in enumerate(far_keys):
        if key in hard_keys:
            origin_far[index] = 1  # synthetic hard
    history_origin[:n_far] = origin_far
    history_origin[n_far:] = 2  # supplement
    history_valid[:n_far] = far_history["history_valid"]
    history_valid[n_far:] = 1

    current = np.concatenate(
        [far["features"].astype(np.float32),
         supp["current_features"].astype(np.float32)],
        axis=0,
    )
    history = np.concatenate(
        [far_history["history_features"],
         supp["history_features"].astype(np.float32)],
        axis=0,
    )
    actions = np.concatenate(
        [far_history["history_actions"],
         supp["history_actions"].astype(np.int8)],
        axis=0,
    )
    dt = np.concatenate(
        [far_history["history_dt"],
         supp["history_dt"].astype(np.float32)],
        axis=0,
    )
    episode_id = np.concatenate(
        [far["episode_id"].astype(np.int64),
         supp["episode_id"].astype(np.int64)]
    )
    step_id = np.concatenate(
        [far["step_id"].astype(np.int64),
         supp["step_id"].astype(np.int64)]
    )
    band = np.concatenate(
        [far["band"].astype(np.int64), supp["band"].astype(np.int64)]
    )
    visible = np.concatenate(
        [far["target_visible"].astype(np.int8),
         supp["target_visible"].astype(np.int8)]
    )
    localizable = np.concatenate(
        [far["target_localizable"].astype(np.int8),
         supp["target_localizable"].astype(np.int8)]
    )
    distance = np.concatenate(
        [far["distance_m"].astype(np.float32),
         supp["distance_m"].astype(np.float32)]
    )
    half_size = np.concatenate(
        [far["goal_half_size_px"].astype(np.float32),
         supp["goal_half_size_px"].astype(np.float32)]
    )
    near_invisible = (
        (distance < 100.0) & (visible == 0)
    ).astype(np.int8)
    source = np.concatenate([
        np.zeros(n_far, dtype=np.int8),
        np.ones(n_supp, dtype=np.int8),
    ])
    variant = np.concatenate([
        np.full(n_far, -1, dtype=np.int8),
        supp["variant"].astype(np.int8),
    ])
    pair_id = np.concatenate([
        np.full(n_far, -1, dtype=np.int32),
        supp["pair_id"].astype(np.int32),
    ])
    sample_type = np.concatenate([
        far["sample_type"].astype(np.int8),
        np.full(n_supp, -1, dtype=np.int8),
    ])
    keys = set(zip(episode_id.tolist(), step_id.tolist()))
    if len(keys) != len(episode_id):
        raise RuntimeError("Duplicate (episode_id, step_id) in combined data")

    valid = history_valid > 0
    near_visible_mask = (source == 1) & (variant == 0)
    near_localizable_mask = near_visible_mask & (localizable == 1)
    far_visible_mask = (source == 0) & (distance >= 100.0) & (visible == 1)
    far_invisible_mask = (source == 0) & (distance >= 100.0) & (visible == 0)
    summary = {
        "split": args.split,
        "states": int(len(current)),
        "far_states": int(n_far),
        "supplement_states": int(n_supp),
        "valid_history": int(valid.sum()),
        "valid_history_rate": float(valid.mean()),
        "invalid_natural_first_frames": int(
            ((source == 0) & (history_origin == -1)).sum()
        ),
        "history_origin": {
            "actual_natural": int((history_origin == 0).sum()),
            "synthetic_hard": int((history_origin == 1).sum()),
            "supplement": int((history_origin == 2).sum()),
            "invalid": int((history_origin == -1).sum()),
        },
        "labels": {
            "target_visible": int(visible.sum()),
            "target_localizable": int(localizable.sum()),
            "near_target_invisible": int(near_invisible.sum()),
            "near_visible": int(near_visible_mask.sum()),
            "near_localizable": int(near_localizable_mask.sum()),
            "far_visible": int(far_visible_mask.sum()),
            "far_invisible": int(far_invisible_mask.sum()),
        },
        "valid_subset_labels": {
            "near_target_invisible": int(near_invisible[valid].sum()),
            "near_visible": int(near_visible_mask[valid].sum()),
            "near_localizable": int(near_localizable_mask[valid].sum()),
            "far_visible": int(far_visible_mask[valid].sum()),
            "far_invisible": int(far_invisible_mask[valid].sum()),
        },
        "bands": {
            name: int((band == code).sum())
            for name, code in (("short", 0), ("medium", 1), ("long", 2))
        },
        "distance_quantiles": {
            str(q): float(np.quantile(distance, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "feature_dim": int(current.shape[1]),
        "history_dim": int(history.shape[1]),
        "compact_dim": int(history.shape[2]),
        "feature_names": [str(x) for x in names],
        "compact_feature_names": list(COMPACT_FEATURE_NAMES),
        "action_map_size": int(len(action_map)),
        "hard_history_keys": int(len(hard["episode_id"])),
    }
    arrays = {
        "current_features": current,
        "history_features": history,
        "history_actions": actions,
        "history_dt": dt,
        "history_valid": history_valid,
        "history_origin": history_origin,
        "episode_id": episode_id,
        "step_id": step_id,
        "band": band,
        "target_visible": visible,
        "target_localizable": localizable,
        "near_target_invisible": near_invisible,
        "distance_m": distance,
        "goal_half_size_px": half_size,
        "source": source,
        "variant": variant,
        "pair_id": pair_id,
        "sample_type": sample_type,
        "feature_names": names,
        "compact_feature_names": np.asarray(COMPACT_FEATURE_NAMES),
    }
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite {out_path}")
    np.savez(out_path, **arrays)
    summary["size_bytes"] = int(out_path.stat().st_size)
    (out_path.parent / f"combined_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "split": args.split,
        "states": summary["states"],
        "valid_history": summary["valid_history"],
        "origin": summary["history_origin"],
        "labels": summary["labels"],
        "valid_subset_labels": summary["valid_subset_labels"],
        "size_gb": round(summary["size_bytes"] / 1e9, 3),
        "output": str(out_path),
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--far-features", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--hard-history", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
