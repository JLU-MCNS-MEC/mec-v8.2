"""Spatial cache lookup and shard loader for V8.3 M0/M1/M2 training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AUX, ENCODER, NETWORK


class SpatialCache:
    """Lookup (episode_id, step_id) -> [4,24,24] maps + scalar validity."""

    def __init__(self, path):
        data = np.load(path)
        self.baseline = data["baseline"]
        self.coherent = data["coherent"]
        self.temporal = data["temporal"]
        self.innovation = data["innovation"]
        self.valid = data["valid"]
        self.episode_id = data["episode_id"]
        self.step_id = data["step_id"]
        keys = list(zip(self.episode_id.tolist(), self.step_id.tolist()))
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate cache keys in {path}")
        self.key_to_idx = {k: i for i, k in enumerate(keys)}

    def lookup(self, episode_ids, step_ids):
        idx = []
        for e, s in zip(episode_ids.tolist(), step_ids.tolist()):
            key = (int(e), int(s))
            if key not in self.key_to_idx:
                raise KeyError(f"Spatial cache missing key {key}")
            idx.append(self.key_to_idx[key])
        idx = np.asarray(idx, dtype=np.int64)
        maps = np.stack(
            [
                self.baseline[idx].astype(np.float32),
                self.coherent[idx].astype(np.float32),
                self.temporal[idx].astype(np.float32),
                self.innovation[idx].astype(np.float32),
            ],
            axis=1,
        )
        valid = self.valid[idx].astype(np.float32).reshape(-1, 1, 1, 1)
        return maps, valid


def spatial_shard_tensors(path, cache: SpatialCache):
    data = torch.load(path, map_location="cpu", weights_only=True)
    required = [
        "tokens", "goal_tokens", "spatial_heatmaps", "network_states",
        "labels", "arrival_labels", "move_valid", "arrival_train_mask",
        "distance_to_goal_m", "sample_type", "episode_ids", "step_ids",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path} missing fields {missing}")
    tokens = data["tokens"].float()
    goal = data["goal_tokens"].float()
    heat = data["spatial_heatmaps"].float()
    network = data["network_states"].float()
    labels = data["labels"].long()
    arrival = data["arrival_labels"].float()
    move_valid = data["move_valid"].bool()
    arrival_mask = data["arrival_train_mask"].bool()
    distance = data["distance_to_goal_m"].float()
    sample_type = data["sample_type"].long()
    episode_ids = data["episode_ids"].long()
    step_ids = data["step_ids"].long()

    if tuple(tokens.shape[1:]) != (ENCODER.num_tokens, ENCODER.token_dim):
        raise ValueError(f"tokens shape error {tuple(tokens.shape)}")
    if tuple(goal.shape[1:]) != (AUX.goal_token_count, ENCODER.token_dim):
        raise ValueError(f"goal shape error {tuple(goal.shape)}")
    if tuple(heat.shape[1:]) != (
        AUX.spatial_heatmap_channels,
        AUX.spatial_heatmap_grid,
        AUX.spatial_heatmap_grid,
    ):
        raise ValueError(f"heatmap shape error {tuple(heat.shape)}")
    if network.shape[1] != NETWORK.network_state_dim:
        raise ValueError(f"network shape error {tuple(network.shape)}")

    maps, valid = cache.lookup(episode_ids, step_ids)
    return (
        tokens, goal, heat, network, labels, arrival, move_valid,
        arrival_mask, distance, sample_type,
        torch.from_numpy(maps), torch.from_numpy(valid),
    )


def build_spatial_loader(path, batch_size, shuffle, seed, cache: SpatialCache):
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    dataset = TensorDataset(*spatial_shard_tensors(path, cache))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def move_spatial_batch_to_device(batch, device):
    (
        tokens, goal, heat, network, labels, arrival, move_valid,
        arrival_mask, distance, sample_type, spatial_maps, spatial_valid,
    ) = batch
    return (
        tokens.to(device, dtype=torch.float32, non_blocking=True),
        goal.to(device, dtype=torch.float32, non_blocking=True),
        heat.to(device, dtype=torch.float32, non_blocking=True),
        network.to(device, dtype=torch.float32, non_blocking=True),
        labels.to(device, dtype=torch.long, non_blocking=True),
        arrival.to(device, dtype=torch.float32, non_blocking=True),
        move_valid.to(device, dtype=torch.bool, non_blocking=True),
        arrival_mask.to(device, dtype=torch.bool, non_blocking=True),
        distance.to(device, dtype=torch.float32, non_blocking=True),
        sample_type.to(device, dtype=torch.long, non_blocking=True),
        spatial_maps.to(device, dtype=torch.float32, non_blocking=True),
        spatial_valid.to(device, dtype=torch.float32, non_blocking=True),
    )
