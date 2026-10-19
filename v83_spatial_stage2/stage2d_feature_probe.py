"""Stage-2d diagnostic: can the 104-D Arrival features separate on-policy
hard far negatives from near positives on held-out Train-600 episodes?

This is a CPU-only linear probe, not the R2 model. It answers whether the
feature signal exists before interpreting R2/R3. The probe uses the same
480/120 episode split as the calibration; the base B0 logit AUC is reported
as the reference (it is uninformative for this hard-state distinction).

No Final Test data is read.
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

from v83_spatial_stage2.stage2d_common import (
    SPLIT_SEED,
    STAGE2C_ROLLOUT,
    TRAIN_EPISODES,
    episode_mask,
    far_negative_mask,
    load_rollout_steps,
    split_episodes,
    write_json,
)


def _auc(scores, labels):
    positive = np.asarray(scores)[np.asarray(labels) == 1]
    negative = np.asarray(scores)[np.asarray(labels) == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    total = 0.0
    for value in positive:
        total += float(np.mean((value > negative) + 0.5 * (value == negative)))
    return total / len(positive)


def _fit_probe(features, labels, seed=20261019, steps=800, lr=0.05):
    torch.manual_seed(seed)
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))
    weight = torch.zeros(features.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=lr)
    positive_weight = torch.tensor(
        (len(labels) - labels.sum()) / max(labels.sum(), 1)
    )
    for _ in range(int(steps)):
        logit = x @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, y, pos_weight=positive_weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return weight.detach().numpy(), float(bias.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, default=STAGE2C_ROLLOUT)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_rollout_steps(args.rollout)
    train, validation = split_episodes(
        data["episode_index"], seed=args.split_seed,
        n_train=args.train_episodes,
    )
    result = {
        "split_seed": int(args.split_seed),
        "train_episodes": int(len(train)),
        "validation_episodes": int(len(validation)),
        "note": (
            "CPU linear probe; hard = far>40m and B0 p>=0.3; "
            "positive = true distance<20m. Held-out = Train-120 episodes."
        ),
    }
    arrays = {}
    for name, episodes in (("train", train), ("validation", validation)):
        mask = episode_mask(data["episode_index"], episodes)
        hard = (
            far_negative_mask(data["distance_m"], data["arrival_label"])
            & (data["arrival_probability"] >= 0.3) & mask
        )
        near = (data["distance_m"] < 20.0) & mask
        indices = np.flatnonzero(hard | near)
        arrays[name] = {
            "features": data["features"][indices].astype(np.float32),
            "labels": hard[indices].astype(np.float32),
            "base_logit": data["arrival_logit"][indices].astype(np.float64),
            "hard_count": int(hard.sum()),
            "near_count": int(near.sum()),
        }
    train_data = arrays["train"]
    mean = train_data["features"].mean(axis=0)
    std = train_data["features"].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    weight, bias = _fit_probe(
        (train_data["features"] - mean) / std, train_data["labels"]
    )
    for name in ("train", "validation"):
        item = arrays[name]
        normalized = (item["features"] - mean) / std
        logits = normalized @ weight + bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        predicted = (probabilities >= 0.5).astype(np.float32)
        result[name] = {
            "hard": int(item["hard_count"]),
            "near": int(item["near_count"]),
            "accuracy": float((predicted == item["labels"]).mean()),
            "auc": float(_auc(probabilities, item["labels"])),
            "base_logit_auc": float(_auc(item["base_logit"], item["labels"])),
        }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
