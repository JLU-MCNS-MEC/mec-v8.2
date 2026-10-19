"""Stage-2i one-time auxiliary feature preparation.

Reads the frozen Train-600 / Dev-Val-180 token shards and writes per-shard
sidecar arrays used by T4/T5:

  history_vis    [N,16,768] float16   mean+max pooled past Current-RGB tokens
  history_pose   [N,16,10]  float32   relative motion/pose + executed action
  history_action [N,16,4]   float16   executed-action one-hot
  history_mask   [N,16]     uint8     valid history slots
  gab            [N,20]     float32   causal Goal-Anchor Belief
  map_feat       [N,22]     float32   anonymous landmark retrieval (T5)

Poses are reconstructed by replaying each archived expert episode with its
seed/band; every replayed action is checked against the stored label.  No
Final Test file is read.

The GAB reliability threshold is fitted on Train natural heatmaps only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TRAIN
from env import ToyUAVEnv
from v83_spatial_stage2.stage2i_data import (
    GAB_FEATURE_DIM,
    HISTORY_STEPS,
    MAP_FEATURE_DIM,
    POSE_FEATURE_DIM,
    VISUAL_DESCRIPTOR_DIM,
    build_gab_for_episode,
    build_landmark_map,
    fit_gab_threshold,
    gab_config_from_threshold,
    history_arrays,
    map_retrieval_features,
    retrieval_diagnostics,
    sorted_shards,
    visual_descriptor,
)
from v83_spatial_training.eval_closedloop_spatial import load_encoder


AUX_FIELDS = (
    "history_vis", "history_pose", "history_action", "history_mask",
    "gab", "map_feat",
)


def _reject_final_test(path):
    if "final_test" in str(path):
        raise ValueError(f"Final Test is sealed; refusing to read {path}")


def _band_range(band_id):
    band = TRAIN.trajectory_mix[int(band_id)]
    return (float(band[2]), float(band[3]))


def run_split(name, directory, output_dir, encoder, device,
              gab_threshold=None, verify_poses=True):
    """Build all sidecar features for one split and write per-shard npz."""
    _reject_final_test(directory)
    started = time.time()
    paths = sorted_shards(directory)
    print(f"[{name}] {len(paths)} shards", flush=True)

    lengths = []
    meta_episode, meta_step, meta_label, meta_move_valid, meta_sample_type = (
        [], [], [], [], []
    )
    meta_seed, meta_band = [], []
    vis_parts, heat_parts, natural_mask_parts = [], [], []
    goal_tokens_by_episode = {}
    episode_rows = {}

    for shard_position, path in enumerate(paths):
        data = torch.load(path, map_location="cpu", weights_only=True)
        n = int(data["labels"].shape[0])
        lengths.append(n)
        episode = data["episode_ids"].cpu().numpy()
        step = data["step_ids"].cpu().numpy()
        label = data["labels"].cpu().numpy()
        move_valid = data["move_valid"].cpu().numpy()
        sample_type = data["sample_type"].cpu().numpy()
        seed = data["episode_seeds"].cpu().numpy()
        band = data["trajectory_band_ids"].cpu().numpy()

        meta_episode.append(episode)
        meta_step.append(step)
        meta_label.append(label)
        meta_move_valid.append(move_valid)
        meta_sample_type.append(sample_type)
        meta_seed.append(seed)
        meta_band.append(band)

        natural = sample_type == 0
        natural_mask_parts.append(natural)
        if natural.any():
            tokens = data["tokens"][natural].to(device=device, dtype=torch.float32)
            vis_parts.append(visual_descriptor(tokens).cpu().numpy())
            heat_parts.append(
                data["spatial_heatmaps"][natural].cpu().numpy().astype(np.float16)
            )
        else:
            vis_parts.append(np.zeros((0, VISUAL_DESCRIPTOR_DIM), dtype=np.float32))
            heat_parts.append(np.zeros((0, 3, 8, 8), dtype=np.float16))

        for offset in range(n):
            ep = int(episode[offset])
            if natural[offset] and ep not in goal_tokens_by_episode:
                goal_tokens_by_episode[ep] = (
                    data["goal_tokens"][offset].float().numpy()
                )
                episode_rows[ep] = {
                    "seed": int(seed[offset]),
                    "band_id": int(band[offset]),
                }
        if (shard_position + 1) % 50 == 0 or shard_position + 1 == len(paths):
            print(f"[{name}] scanned {shard_position + 1}/{len(paths)} shards",
                  flush=True)

    meta = {
        "episode": np.concatenate(meta_episode),
        "step": np.concatenate(meta_step),
        "label": np.concatenate(meta_label),
        "move_valid": np.concatenate(meta_move_valid),
        "sample_type": np.concatenate(meta_sample_type),
        "seed": np.concatenate(meta_seed),
        "band": np.concatenate(meta_band),
    }
    natural_mask = np.concatenate(natural_mask_parts)
    natural_positions = np.full(len(natural_mask), -1, dtype=np.int64)
    natural_positions[natural_mask] = np.arange(int(natural_mask.sum()))
    vis_natural = np.concatenate(vis_parts).astype(np.float16)
    heat_natural = np.concatenate(heat_parts).astype(np.float16)
    n_natural = len(vis_natural)
    print(f"[{name}] samples={len(natural_mask)} natural={n_natural}",
          flush=True)

    if gab_threshold is None:
        gab_threshold = fit_gab_threshold(heat_natural)
        print(f"[{name}] fitted GAB reliability threshold={gab_threshold:.6f}",
              flush=True)
    gab_config = gab_config_from_threshold(gab_threshold)

    natural_indices = np.where(natural_mask)[0]

    # Episode -> ordered natural sample positions.
    episodes = {}
    for position, index in enumerate(natural_indices):
        ep = int(meta["episode"][index])
        episodes.setdefault(ep, []).append(position)
    for ep, positions in episodes.items():
        positions.sort(key=lambda p: int(meta["step"][natural_indices[p]]))
        episode_rows[ep]["positions"] = positions

    history_vis = np.zeros(
        (len(natural_mask), HISTORY_STEPS, VISUAL_DESCRIPTOR_DIM), dtype=np.float16)
    history_pose = np.zeros(
        (len(natural_mask), HISTORY_STEPS, POSE_FEATURE_DIM), dtype=np.float32)
    history_action = np.zeros(
        (len(natural_mask), HISTORY_STEPS, 4), dtype=np.float16)
    history_mask = np.zeros(
        (len(natural_mask), HISTORY_STEPS), dtype=np.uint8)
    gab = np.zeros((len(natural_mask), GAB_FEATURE_DIM), dtype=np.float32)
    map_feat = np.zeros((len(natural_mask), MAP_FEATURE_DIM), dtype=np.float32)

    stats = {
        "episodes": 0,
        "pose_action_mismatches": 0,
        "step_mismatches": 0,
        "map_goal_top1": 0,
        "map_goal_top5": 0,
        "map_goal_cosine": [],
    }
    encoding_started = time.time()
    for episode_index, (ep, info) in enumerate(sorted(episode_rows.items())):
        positions = info["positions"]
        steps = [int(meta["step"][natural_indices[p]]) for p in positions]
        if steps != list(range(len(steps))):
            stats["step_mismatches"] += 1
            raise RuntimeError(
                f"Episode {ep} has non-contiguous natural steps: {steps[:5]}..."
            )
        actions = np.asarray(
            [int(meta["label"][natural_indices[p]]) for p in positions],
            dtype=np.int64,
        )
        env = ToyUAVEnv()
        env.reset(
            seed=int(info["seed"]),
            goal_distance_range_m=_band_range(info["band_id"]),
            render_observation=False,
        )
        poses = []
        for t in range(len(steps)):
            poses.append(env.pose.copy())
            oracle = int(env.oracle_action())
            if verify_poses and oracle != int(actions[t]):
                stats["pose_action_mismatches"] += 1
                raise RuntimeError(
                    f"Replay mismatch episode={ep} step={t}: "
                    f"oracle={oracle} stored={int(actions[t])}"
                )
            env.step(int(actions[t]), render_observation=False)
        if env.step_count != len(steps):
            stats["step_mismatches"] += 1
            raise RuntimeError(
                f"Replay length mismatch episode={ep}: "
                f"{env.step_count} vs {len(steps)}"
            )

        # Anonymous map + frozen visual embeddings (identical pipeline for all).
        landmark_map = build_landmark_map(env, info["seed"], encoder, device)
        goal_token = goal_tokens_by_episode[ep][2].astype(np.float32)
        diagnostic = retrieval_diagnostics(
            landmark_map["goal_index"],
            landmark_map["embeddings"],
            goal_token,
        )
        stats["map_goal_top1"] += int(diagnostic["top1"])
        stats["map_goal_top5"] += int(diagnostic["top5"])
        stats["map_goal_cosine"].append(float(diagnostic["goal_similarity"]))

        # GAB (causal, native heatmaps).
        ordered_heat = [heat_natural[p] for p in positions]
        episode_gab = build_gab_for_episode(
            ordered_heat, actions, gab_config,
            step_ids=steps,
        )

        visual_list = [vis_natural[p] for p in positions]
        for t, position in enumerate(positions):
            global_index = natural_indices[position]
            hv, hp, ha, hm = history_arrays(
                poses, actions, t, visual_list
            )
            history_vis[global_index] = hv
            history_pose[global_index] = hp
            history_action[global_index] = ha.astype(np.float16)
            history_mask[global_index] = hm
            gab[global_index] = episode_gab[t]
            map_feat[global_index] = map_retrieval_features(
                poses[t],
                landmark_map["positions"],
                landmark_map["embeddings"],
                goal_token,
            )
        stats["episodes"] += 1
        if (episode_index + 1) % 50 == 0 or episode_index + 1 == len(episode_rows):
            elapsed = time.time() - encoding_started
            print(f"[{name}] episodes {episode_index + 1}/{len(episode_rows)} "
                  f"({elapsed:.1f}s)", flush=True)

    # Split in streaming order and write one sidecar per original shard.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    aux_files = []
    for shard_index, length in enumerate(lengths):
        selection = np.arange(cursor, cursor + length)
        cursor += length
        filename = f"aux_shard_{shard_index:04d}.npz"
        np.savez(
            output_dir / filename,
            history_vis=history_vis[selection],
            history_pose=history_pose[selection],
            history_action=history_action[selection],
            history_mask=history_mask[selection],
            gab=gab[selection],
            map_feat=map_feat[selection],
        )
        aux_files.append(filename)
    if cursor != len(natural_mask):
        raise RuntimeError(f"Split cursor {cursor} != samples {len(natural_mask)}")

    class_counts = []
    for action in range(3):
        class_counts.append(int(np.sum(
            (meta["label"][natural_indices] == action)
            & (meta["move_valid"][natural_indices] > 0)
        )))

    result = {
        "split": name,
        "directory": str(directory),
        "shards": [p.name for p in paths],
        "aux_files": aux_files,
        "shard_lengths": lengths,
        "samples": int(len(natural_mask)),
        "natural_samples": int(n_natural),
        "episodes": int(stats["episodes"]),
        "movement_class_counts": class_counts,
        "gab_reliability_threshold": float(gab_threshold),
        "pose_replay": {
            "action_mismatches": int(stats["pose_action_mismatches"]),
            "step_mismatches": int(stats["step_mismatches"]),
            "verified": bool(verify_poses),
        },
        "map_diagnostics": {
            "top1": int(stats["map_goal_top1"]),
            "top5": int(stats["map_goal_top5"]),
            "episodes": int(stats["episodes"]),
            "mean_goal_cosine": float(np.mean(stats["map_goal_cosine"]))
            if stats["map_goal_cosine"] else float("nan"),
        },
        "elapsed_s": time.time() - started,
        "final_test_read": False,
    }
    print(f"[{name}] wrote {len(aux_files)} sidecars to {output_dir} "
          f"({result['elapsed_s']:.1f}s)", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path,
                        default=Path("runs/v83_spatial_stage2/datasets/train_600"))
    parser.add_argument("--val-dir", type=Path,
                        default=Path("runs/v83_spatial_stage2/datasets/dev_val_180"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-pose-verify", action="store_true")
    args = parser.parse_args()
    _reject_final_test(args.train_dir)
    _reject_final_test(args.val_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(device)
    print(f"Device: {device}", flush=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "Stage-2i auxiliary features",
        "history_steps": HISTORY_STEPS,
        "visual_descriptor_dim": VISUAL_DESCRIPTOR_DIM,
        "pose_feature_dim": POSE_FEATURE_DIM,
        "gab_feature_dim": GAB_FEATURE_DIM,
        "map_feature_dim": MAP_FEATURE_DIM,
        "final_test_read": False,
        "splits": {},
    }
    threshold = None
    if "train" in args.splits:
        train_result = run_split(
            "train", args.train_dir, args.output_dir / "train",
            encoder, device, gab_threshold=None,
            verify_poses=not args.skip_pose_verify,
        )
        manifest["splits"]["train"] = train_result
        threshold = float(train_result["gab_reliability_threshold"])
        manifest["train_class_counts"] = train_result["movement_class_counts"]
        manifest["gab_reliability_threshold"] = threshold
        (args.output_dir / "train_class_counts.json").write_text(
            json.dumps({
                "counts": train_result["movement_class_counts"],
                "gamma": float(TRAIN.movement_class_balance_gamma),
            }, indent=2), encoding="utf-8",
        )
        (args.output_dir / "manifest.partial.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    if "val" in args.splits:
        if threshold is None:
            raise SystemExit("Val preparation requires the Train split first")
        manifest["splits"]["val"] = run_split(
            "val", args.val_dir, args.output_dir / "val",
            encoder, device, gab_threshold=threshold,
            verify_poses=not args.skip_pose_verify,
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("PREPARATION_COMPLETE", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
