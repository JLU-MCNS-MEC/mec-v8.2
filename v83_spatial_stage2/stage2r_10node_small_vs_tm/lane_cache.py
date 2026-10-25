"""Build and load exact frozen-lane caches for Stage-2r training.

Navigation is produced by the frozen MEC model and association only selects a
serving node.  Therefore the real closed-loop trajectory, success and step
count of a task are independent of the DQN/PPO policy.  We can run the full
environment once per task, store the per-step pose / 10x19 candidate features
/ mask / node E2E, and then train any association policy by exact replay.
This is the same equivalence used by Stage-2n's lane evaluator and is
validated again by ``test_lane_equivalence.py``.

The cache contains no Test task.  Each shard is a pickle of compact numpy
arrays; the manifest records task/deployment/checkpoint/source hashes.

Usage:
    python v83_spatial_stage2/stage2r_10node_small_vs_tm/lane_cache.py \
        --model Small --workers 6
    python v83_spatial_stage2/stage2r_10node_small_vs_tm/lane_cache.py \
        --model Transformer-M --workers 6
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
)
from v83_spatial_stage2.stage2r_10node_small_vs_tm.common import (  # noqa: E402
    MEC_MODELS,
    RUNS_ROOT,
    STAGE,
    artifact_hashes,
    load_bundle,
    model_slug,
    source_hashes,
)
from v83_spatial_stage2.stage2r_10node_small_vs_tm.runtime import (  # noqa: E402
    build_lane,
    load_stage2r_runtime,
    make_scene_cache,
)

CACHE_ROOT = RUNS_ROOT / "lane_cache"
SHARD_SIZE = 250


def compact_lane(lane):
    steps = lane.steps
    count = len(steps)
    pose = np.asarray([step.pose for step in steps], dtype=np.float32)
    candidate = np.asarray(
        [step.candidate_features for step in steps], dtype=np.float32
    )
    mask = np.asarray([step.mask for step in steps], dtype=bool)
    step_index = np.asarray(
        [step.step_index for step in steps], dtype=np.int16
    )
    # float64 preserves the exact Python-float E2E value so the running mean
    # and 194-D state match the full runner bit-for-bit.
    total_ms = np.zeros((count, len(steps[0].latency_metrics)),
                        dtype=np.float64)
    distance_m = np.zeros((count, len(steps[0].latency_metrics)),
                          dtype=np.float32)
    for index, step in enumerate(steps):
        for node, metric in enumerate(step.latency_metrics):
            if metric is not None:
                total_ms[index, node] = float(metric["total_ms"])
                distance_m[index, node] = float(metric["distance_m"])
    actions = np.asarray([step.action for step in steps], dtype=np.int16)
    xy = pose[:, :2]
    path_length = float(
        np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    ) if count > 1 else 0.0
    return {
        "episode_uid": str(lane.episode_uid),
        "split": str(lane.split),
        "episode_index": int(lane.episode_index),
        "task_seed": int(lane.task_seed),
        "band": str(lane.band),
        "band_id": int(lane.band_id),
        "scenario": str(lane.scenario),
        "success": bool(lane.success),
        "osr": bool(lane.osr),
        "spl": float(lane.spl),
        "ndtw": float(lane.ndtw),
        "steps": int(lane.episode_steps),
        "path_length_m": path_length,
        "action_sha256": str(lane.action_sha256),
        "pose": pose,
        "candidate": candidate,
        "mask": mask,
        "step_index": step_index,
        "total_ms": total_ms,
        "distance_m": distance_m,
        "actions": actions,
    }


def _worker(args):
    model, indices, output, device = args
    started = time.monotonic()
    bundle = load_bundle(model)
    device, encoder, backends = load_stage2r_runtime(device, model)
    scene_cache = make_scene_cache(encoder, device, capacity=64)
    episodes = {}
    for count, index in enumerate(indices, start=1):
        spec = bundle["train_specs"][int(index)]
        lane = build_lane(
            spec, bundle["topology"], encoder, backends,
            bundle["quality_prior"], bundle["gab_config"], device,
            scene_cache,
        )
        episodes[str(lane.episode_uid)] = compact_lane(lane)
        if count % 10 == 0:
            print(
                f"[lane-cache] {model} shard={Path(output).stem} "
                f"{count}/{len(indices)} elapsed="
                f"{time.monotonic()-started:.1f}s",
                flush=True,
            )
    payload = {
        "stage": STAGE,
        "model": model,
        "split": "train",
        "indices": [int(index) for index in indices],
        "episodes": episodes,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(output).with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)
    temporary.replace(output)
    return str(output), len(episodes)


def _relative_path(path):
    path = Path(path)
    if path.is_absolute():
        return str(path.relative_to(PROJECT_ROOT))
    return str(path)


def finalize_manifest(model, shard_size=SHARD_SIZE, elapsed_seconds=0.0):
    """Write a manifest for already-built shard files (no environment runs).

    Used after a builder crashes at the final bookkeeping step; every shard
    file must already exist and be readable.
    """
    model = str(model)
    root = CACHE_ROOT / model_slug(model)
    manifest_path = root / "manifest.json"
    bundle = load_bundle(model)
    paths = sorted(root.glob("shard_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no shards under {root}")
    shard_records = []
    total = 0
    for path in paths:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("model") != model:
            raise ValueError(f"{path} model mismatch")
        count = len(payload["episodes"])
        total += count
        shard_records.append({
            "path": _relative_path(path),
            "sha256": sha256_file(path),
            "tasks": int(count),
            "bytes": int(path.stat().st_size),
        })
    if total != len(bundle["train_specs"]):
        raise ValueError(
            f"shard tasks {total} != train tasks {len(bundle['train_specs'])}"
        )
    manifest = {
        "stage": STAGE,
        "created_utc": utc_now(),
        "model": model,
        "split": "train",
        "tasks": total,
        "shard_size": int(shard_size),
        "shards": shard_records,
        "artifact_hashes": artifact_hashes(bundle),
        "source_hashes": source_hashes(),
        "elapsed_seconds": float(elapsed_seconds),
        "finalized_from_existing_shards": True,
        "final_test_read": False,
    }
    atomic_write_json(manifest_path, manifest)
    print(
        f"[lane-cache] {model}: finalized manifest {manifest_path} "
        f"({total} episodes, {len(shard_records)} shards)",
        flush=True,
    )
    return manifest


def build_cache(model, workers=6, shard_size=SHARD_SIZE, device="cuda",
                force=False):
    model = str(model)
    if model not in MEC_MODELS:
        raise ValueError(model)
    root = CACHE_ROOT / model_slug(model)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not force:
        print(f"[lane-cache] existing manifest: {manifest_path}", flush=True)
        return read_json(manifest_path)
    bundle = load_bundle(model)
    specs = bundle["train_specs"]
    total = len(specs)
    shards = [
        list(range(start, min(start + shard_size, total)))
        for start in range(0, total, shard_size)
    ]
    jobs = []
    for shard_index, indices in enumerate(shards):
        path = root / f"shard_{shard_index:03d}.pkl"
        jobs.append((model, indices, str(path), device))
    root.mkdir(parents=True, exist_ok=True)
    print(
        f"[lane-cache] {model}: {total} tasks, {len(shards)} shards, "
        f"{workers} workers",
        flush=True,
    )
    started = time.monotonic()
    completed = []
    context = mp.get_context("spawn")
    with context.Pool(int(workers)) as pool:
        for index, (path, count) in enumerate(
            pool.imap_unordered(_worker, jobs), start=1
        ):
            elapsed = time.monotonic() - started
            print(
                f"[lane-cache] {model}: finished shard "
                f"{index}/{len(jobs)} ({count} episodes) elapsed={elapsed:.1f}s",
                flush=True,
            )
            completed.append({"path": path, "tasks": count})
    shard_records = []
    for item in sorted(completed, key=lambda item: item["path"]):
        path = Path(item["path"])
        shard_records.append({
            "path": _relative_path(path),
            "sha256": sha256_file(path),
            "tasks": int(item["tasks"]),
            "bytes": int(path.stat().st_size),
        })
    manifest = {
        "stage": STAGE,
        "created_utc": utc_now(),
        "model": model,
        "split": "train",
        "tasks": total,
        "shard_size": int(shard_size),
        "shards": shard_records,
        "artifact_hashes": artifact_hashes(bundle),
        "source_hashes": source_hashes(),
        "elapsed_seconds": float(time.monotonic() - started),
        "final_test_read": False,
    }
    atomic_write_json(manifest_path, manifest)
    print(
        f"[lane-cache] {model}: manifest {manifest_path} "
        f"({sum(item['bytes'] for item in shard_records)/1e9:.2f} GB)",
        flush=True,
    )
    return manifest


def load_cache(model, verify=True):
    root = CACHE_ROOT / model_slug(model)
    manifest = read_json(root / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(root / "manifest.json")
    episodes = {}
    for shard in manifest["shards"]:
        path = PROJECT_ROOT / shard["path"]
        if verify and sha256_file(path) != shard["sha256"]:
            raise ValueError(f"lane shard hash mismatch: {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("model") != model:
            raise ValueError(f"{path} model mismatch")
        episodes.update(payload["episodes"])
    if len(episodes) != int(manifest["tasks"]):
        raise ValueError(
            f"lane cache {model}: {len(episodes)} != {manifest['tasks']}"
        )
    return episodes, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MEC_MODELS, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--finalize-only", action="store_true",
        help="write manifest from existing shards without running env",
    )
    parser.add_argument("--elapsed-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.finalize_only:
        finalize_manifest(
            args.model, shard_size=args.shard_size,
            elapsed_seconds=args.elapsed_seconds,
        )
    else:
        build_cache(
            args.model, workers=args.workers, shard_size=args.shard_size,
            device=args.device, force=args.force,
        )
    print("STAGE2R_LANE_CACHE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
