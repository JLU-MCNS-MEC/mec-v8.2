"""Build V8.3 spatial feature caches for M0/M1/M2 training.

Cache channels per sample:
  0 baseline24   : old per-patch 3-scale MAX cosine on the 24x24 token grid
  1 coherent24   : multiscale local-consistency map
  2 temporal24   : yaw-aligned previous coherent map on turn transitions,
                   equal to coherent24 when temporal validity is false
  3 innovation24 : coherent24 - yaw_warped previous coherent (zero if invalid)

Plus a scalar temporal-validity mask.

Strict alignment:
  * keys are (episode_id, step_id) from the archived token shards;
  * previous action is the archived label at step_id-1 of the SAME episode;
  * first frame, hard independent samples, missing previous state, and FORWARD
    previous actions are invalid (valid=0);
  * only LEFT/RIGHT previous actions produce a yaw warp;
  * no future action and no Goal truth/position/mask is used.

This cache is an offline training artifact under ignored runs/v83_spatial_training/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER
from env import FORWARD, LEFT, RIGHT
from visual_grounding_v83 import static_grounding, yaw_warp_previous


SPLITS = {
    "train": Path("token_dataset_train"),
    "val": Path("token_dataset_val"),
    "val_hard": Path("token_dataset_val_hard"),
}
CARRY = 0.30


def _shard_files(folder):
    files = sorted(Path(folder).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No shards in {folder}")
    return files


def build_split_cache(split, folder, output_dir, device, batch_size=64, max_shards=None):
    files = _shard_files(folder)
    if max_shards is not None:
        files = files[: int(max_shards)]
    print(f"[{split}] {len(files)} shards", flush=True)
    baseline_parts, coherent_parts = [], []
    labels_parts, sample_type_parts = [], []
    episode_parts, step_parts = [], []
    t0 = time.time()
    for si, path in enumerate(files):
        data = torch.load(path, map_location="cpu", weights_only=True)
        tokens = data["tokens"].float()
        goal = data["goal_tokens"].float()
        labels = data["labels"].long().numpy()
        sample_type = data["sample_type"].long().numpy()
        episode = data["episode_ids"].long().numpy()
        step = data["step_ids"].long().numpy()
        b_parts, c_parts = [], []
        with torch.inference_mode():
            for start in range(0, tokens.shape[0], batch_size):
                end = min(start + batch_size, tokens.shape[0])
                b, c = static_grounding(
                    tokens[start:end].to(device), goal[start:end].to(device)
                )
                b_parts.append(b.detach().cpu().numpy().astype(np.float16))
                c_parts.append(c.detach().cpu().numpy().astype(np.float16))
        baseline_parts.append(np.concatenate(b_parts, axis=0))
        coherent_parts.append(np.concatenate(c_parts, axis=0))
        labels_parts.append(labels)
        sample_type_parts.append(sample_type)
        episode_parts.append(episode)
        step_parts.append(step)
        if (si + 1) % 25 == 0 or si + 1 == len(files):
            print(f"  [{split}] {si + 1}/{len(files)} shards", flush=True)
    baseline = np.concatenate(baseline_parts, axis=0)
    coherent = np.concatenate(coherent_parts, axis=0)
    labels = np.concatenate(labels_parts, axis=0)
    sample_type = np.concatenate(sample_type_parts, axis=0)
    episode = np.concatenate(episode_parts, axis=0)
    step = np.concatenate(step_parts, axis=0)
    del baseline_parts, coherent_parts, labels_parts, sample_type_parts, episode_parts, step_parts

    keys = list(zip(episode.tolist(), step.tolist()))
    if len(set(keys)) != len(keys):
        duplicates = [k for k, v in Counter(keys).items() if v > 1]
        raise RuntimeError(f"Duplicate (episode_id, step_id) keys in {split}: {duplicates[:10]}")
    key_to_idx = {k: i for i, k in enumerate(keys)}

    n = len(keys)
    temporal = coherent.copy()
    innovation = np.zeros_like(coherent)
    valid = np.zeros(n, dtype=np.uint8)
    previous_action = {}
    missing_prev = 0
    first_frame = 0
    hard_count = 0
    forward_invalid = 0
    candidates = []
    for i in range(n):
        if int(sample_type[i]) != 0:
            hard_count += 1
            continue
        s = int(step[i])
        if s <= 0:
            first_frame += 1
            continue
        prev = key_to_idx.get((int(episode[i]), s - 1))
        if prev is None:
            missing_prev += 1
            continue
        pa = int(labels[prev])
        previous_action[pa] = previous_action.get(pa, 0) + 1
        if pa == FORWARD:
            forward_invalid += 1
            continue
        if pa in (LEFT, RIGHT):
            candidates.append((i, prev, pa))

    turn_valid = 0
    if candidates:
        for start in range(0, len(candidates), 256):
            chunk = candidates[start:start + 256]
            idx = np.asarray([x[0] for x in chunk], dtype=np.int64)
            prev_idx = np.asarray([x[1] for x in chunk], dtype=np.int64)
            actions = np.asarray([x[2] for x in chunk], dtype=np.int64)
            prev_maps = torch.from_numpy(coherent[prev_idx].astype(np.float32)).to(device)
            warped, warp_valid = yaw_warp_previous(
                prev_maps, torch.from_numpy(actions).to(device)
            )
            warped = warped.detach().cpu().numpy().astype(np.float16)
            warp_valid = warp_valid.detach().cpu().numpy().astype(bool)
            cur = coherent[idx].astype(np.float32)
            w = warped.astype(np.float32)
            temporal[idx] = ((1.0 - CARRY) * cur + CARRY * w).astype(np.float16)
            innovation[idx] = (cur - w).astype(np.float16)
            valid[idx[warp_valid]] = 1
            turn_valid += int(warp_valid.sum())

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{split}.npz"
    if outpath.exists():
        raise SystemExit(f"Refusing to overwrite cache: {outpath}")
    np.savez(
        outpath,
        episode_id=episode.astype(np.int64),
        step_id=step.astype(np.int64),
        baseline=baseline,
        coherent=coherent,
        temporal=temporal,
        innovation=innovation,
        valid=valid,
    )
    meta = {
        "split": split,
        "source": str(folder),
        "samples": int(n),
        "natural": int((sample_type == 0).sum()),
        "hard": int(hard_count),
        "first_frame_invalid": int(first_frame),
        "forward_previous_invalid": int(forward_invalid),
        "turn_previous_valid": int(turn_valid),
        "missing_previous_invalid": int(missing_prev),
        "duplicate_keys": 0,
        "previous_action_counts": {str(k): int(v) for k, v in previous_action.items() if isinstance(k, int)},
        "carry": CARRY,
        "elapsed_s": time.time() - t0,
        "note": "Keys use archived episode_id/step_id; previous action is archived label at step_id-1. No Goal truth used.",
    }
    (outdir / f"{split}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{split}] saved {outpath} ({outpath.stat().st_size/1e6:.1f} MB)", flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "val_hard"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-shards", type=int, default=None)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    results = {}
    for split in args.splits:
        if split not in SPLITS:
            raise SystemExit(f"Unknown split {split}")
        results[split] = build_split_cache(
            split, SPLITS[split], args.output_dir, device, args.batch_size,
            max_shards=args.max_shards,
        )
    (Path(args.output_dir) / "cache_manifest.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
