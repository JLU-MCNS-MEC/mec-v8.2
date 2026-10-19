"""Stage-2g observable feature builder for the target-presence probe.

Reads the existing Stage-2 token shards and the matched spatial cache and
writes one feature matrix per split. Only observable inputs are used:

  * current RGB tokens (576 x 384)
  * goal reference tokens (3 x 384)
  * Goal-Current spatial heatmap (3 x 8 x 8)
  * parameter-free baseline/coherent spatial maps from the Stage-2 cache
    (24 x 24, derived only from current and goal tokens)

True labels (visibility, distance, projection size, sample type) are stored
alongside the matrix for evaluation only and are never appended to the feature
columns. `FORBIDDEN_FEATURE_TERMS` is checked before writing.

Usage:
  python v83_spatial_stage2/stage2g_features.py --split train
  python v83_spatial_stage2/stage2g_features.py --split val --max-shards 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BANDS = {"short": 0, "medium": 1, "long": 2}
FORBIDDEN_FEATURE_TERMS = (
    "distance", "visible", "visibility", "half_size", "goal_px", "goal_py",
    "slant", "heading", "oracle", "arrival_label", "sample_type", "band",
    "episode_id", "step_id",
)


def _check_feature_names(names):
    for name in names:
        lowered = name.lower()
        for term in FORBIDDEN_FEATURE_TERMS:
            if term in lowered:
                raise ValueError(
                    f"Feature name {name!r} contains forbidden term {term!r}"
                )


def _topk_stats(values, dim, k):
    top = torch.topk(values, k=min(k, values.shape[dim]), dim=dim).values
    return top.mean(dim=dim), top[:, -1] if dim == 1 else top[..., -1]


def _entropy_from_heatmap(heatmap):
    # heatmap: (B, 3, 8, 8); normalize each channel to a distribution.
    flat = heatmap.flatten(2)
    prob = flat / flat.sum(dim=2, keepdim=True).clamp_min(1e-8)
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=2)
    entropy = entropy / math.log(flat.shape[2])
    top1 = flat.max(dim=2).values
    top5 = torch.topk(flat, k=5, dim=2).values.sum(dim=2)
    top5_mass = top5 / flat.sum(dim=2).clamp_min(1e-8)
    grid = int(round(math.sqrt(flat.shape[2])))
    weights = prob
    coords = torch.arange(grid, dtype=torch.float32, device=heatmap.device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    xx = xx.flatten() / max(grid - 1, 1)
    yy = yy.flatten() / max(grid - 1, 1)
    centroid_x = (weights * xx).sum(dim=2)
    centroid_y = (weights * yy).sum(dim=2)
    return torch.stack(
        [entropy, top1, top5_mass, centroid_x, centroid_y], dim=2
    ).flatten(1)


def _map_stats(maps):
    # maps: (B, 24, 24) -> pooled 8x8 plus scalar stats.
    pooled = F.adaptive_avg_pool2d(maps.unsqueeze(1), (8, 8)).flatten(1)
    flat = maps.flatten(1)
    max_values = flat.max(dim=1).values
    mean_values = flat.mean(dim=1)
    std_values = flat.std(dim=1)
    relu = flat.clamp_min(0.0)
    mass = relu.sum(dim=1).clamp_min(1e-8)
    grid = maps.shape[-1]
    coords = torch.arange(grid, dtype=torch.float32, device=maps.device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    xx = xx.flatten() / max(grid - 1, 1)
    yy = yy.flatten() / max(grid - 1, 1)
    centroid_x = (relu * xx).sum(dim=1) / mass
    centroid_y = (relu * yy).sum(dim=1) / mass
    return torch.cat([
        pooled,
        torch.stack([
            max_values, mean_values, std_values, centroid_x, centroid_y
        ], dim=1),
    ], dim=1)


def feature_names():
    names = []
    for prefix in ("cur_mean", "cur_std", "cur_max", "goal_mean", "goal_std"):
        names += [f"{prefix}_{i}" for i in range(384)]
    for scale in range(3):
        for stat in ("sim_max", "sim_top5", "sim_mean", "sim_std"):
            names.append(f"{stat}_s{scale}")
    names += [f"sim_max_all_{i}" for i in range(576)]
    for stat in (
        "sim_max_all_top5", "sim_max_all_mean", "sim_max_all_std",
        "sim_argmax_x", "sim_argmax_y",
    ):
        names.append(stat)
    for channel in range(3):
        for stat in ("ent", "top1", "top5_mass", "cx", "cy"):
            names.append(f"heat_{channel}_{stat}")
    names += [f"heat_{channel}_{i}" for channel in range(3)
              for i in range(64)]
    for source in ("base", "coh"):
        names += [f"{source}_pool_{i}" for i in range(64)]
        for stat in ("max", "mean", "std", "cx", "cy"):
            names.append(f"{source}_{stat}")
    return names


def build_features(data_dir, cache_path, labels_path, max_shards=None):
    import torch

    labels = {}
    with Path(labels_path).open(encoding="utf-8") as handle:
        import csv
        for row in csv.DictReader(handle):
            labels[(int(row["episode_id"]), int(row["step_id"]))] = row
    cache = np.load(cache_path)
    cache_keys = list(zip(
        cache["episode_id"].tolist(), cache["step_id"].tolist()
    ))
    if len(set(cache_keys)) != len(cache_keys):
        raise RuntimeError("Duplicate keys in spatial cache")
    cache_index = {key: i for i, key in enumerate(cache_keys)}

    shard_paths = sorted(Path(data_dir).glob("shard_*.pt"))
    if max_shards is not None:
        shard_paths = shard_paths[: int(max_shards)]
    names = feature_names()
    _check_feature_names(names)
    feature_rows = []
    meta = {key: [] for key in (
        "episode_id", "step_id", "band", "target_visible",
        "target_localizable", "distance_m", "goal_half_size_px",
        "sample_type",
    )}
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        tokens = data["tokens"].float()
        goal = data["goal_tokens"].float()
        heatmap = data["spatial_heatmaps"].float()
        episodes = data["episode_ids"].long().tolist()
        steps = data["step_ids"].long().tolist()
        n = len(episodes)
        batch_keys = list(zip(episodes, steps))
        if len(set(batch_keys)) != len(batch_keys):
            raise RuntimeError(f"Duplicate keys in {path}")
        indices = []
        for key in batch_keys:
            if key not in labels:
                raise RuntimeError(f"Missing labels for {key}")
            indices.append(cache_index[key])
        index_tensor = torch.tensor(indices, dtype=torch.long)
        base = torch.from_numpy(
            cache["baseline"][indices].astype(np.float32)
        )
        coh = torch.from_numpy(
            cache["coherent"][indices].astype(np.float32)
        )

        cur_norm = F.normalize(tokens, dim=2)
        goal_norm = F.normalize(goal, dim=2)
        sim = torch.einsum("bnd,bkd->bnk", cur_norm, goal_norm)
        sim_max_all = sim.max(dim=2).values
        patch_argmax = sim_max_all.argmax(dim=1)
        sim_argmax_x = (patch_argmax % 24).float() / 23.0
        sim_argmax_y = torch.div(
            patch_argmax, 24, rounding_mode="floor"
        ).float() / 23.0

        parts = [
            tokens.mean(dim=1), tokens.std(dim=1), tokens.max(dim=1).values,
            goal.mean(dim=1), goal.std(dim=1),
        ]
        for scale in range(3):
            values = sim[:, :, scale]
            top5 = torch.topk(values, k=5, dim=1).values
            parts += [
                values.max(dim=1).values.unsqueeze(1),
                top5.mean(dim=1).unsqueeze(1),
                values.mean(dim=1).unsqueeze(1),
                values.std(dim=1).unsqueeze(1),
            ]
        parts += [
            sim_max_all,
            torch.topk(sim_max_all, k=5, dim=1).values.mean(dim=1).unsqueeze(1),
            sim_max_all.mean(dim=1).unsqueeze(1),
            sim_max_all.std(dim=1).unsqueeze(1),
            sim_argmax_x.unsqueeze(1),
            sim_argmax_y.unsqueeze(1),
        ]
        parts += [_entropy_from_heatmap(heatmap)]
        parts += [heatmap.flatten(1)]
        parts += [_map_stats(base), _map_stats(coh)]
        features = torch.cat(parts, dim=1)
        if features.shape[1] != len(names):
            raise RuntimeError(
                f"feature width {features.shape[1]} != {len(names)}"
            )
        feature_rows.append(features.numpy().astype(np.float32))
        for key in batch_keys:
            row = labels[key]
            meta["episode_id"].append(key[0])
            meta["step_id"].append(key[1])
            meta["band"].append(BANDS[row["band"]])
            meta["target_visible"].append(int(row["target_visible"]))
            meta["target_localizable"].append(int(row["target_localizable"]))
            meta["distance_m"].append(float(row["distance_m"]))
            meta["goal_half_size_px"].append(float(row["goal_half_size_px"]))
            meta["sample_type"].append(int(row["sample_type"]))
    features = np.concatenate(feature_rows, axis=0)
    output = {
        "features": features,
        "feature_names": np.asarray(names),
        **{key: np.asarray(value) for key, value in meta.items()},
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path(
        "runs/v83_spatial_stage2/datasets"
    ))
    parser.add_argument("--cache", type=Path, default=Path(
        "runs/v83_spatial_stage2/cache_20261018T043434Z"
    ))
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()
    data_dir = args.data_root / (
        "train_600" if args.split == "train" else "dev_val_180"
    )
    cache_path = args.cache / (
        "train.npz" if args.split == "train" else "val.npz"
    )
    labels_path = args.labels_dir / f"labels_{args.split}.csv"
    output = build_features(
        data_dir, cache_path, labels_path, max_shards=args.max_shards
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"features_{args.split}.npz"
    np.savez_compressed(out_path, **output)
    summary = {
        "split": args.split,
        "samples": int(output["features"].shape[0]),
        "feature_dim": int(output["features"].shape[1]),
        "target_visible_rate": float(output["target_visible"].mean()),
        "target_localizable_rate": float(
            output["target_localizable"].mean()
        ),
        "bands": {
            name: int((output["band"] == code).sum())
            for name, code in BANDS.items()
        },
        "size_bytes": int(out_path.stat().st_size),
    }
    (output_dir / f"features_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
