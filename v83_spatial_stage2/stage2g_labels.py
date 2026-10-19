"""Stage-2g label-only replay for target visibility / localizability.

The existing Train 600 / Dev Val 180 token shards store the inputs and a
geometric `goal_visible` flag, but not the target projection geometry or an
occlusion-aware visible-pixel count. This script replays the deterministic
oracle trajectories (and the hard arrival samples with the same RNG stream)
and writes labels only. It does **not** encode tokens, render images, write
shards, or modify any existing dataset.

Labels (true goal geometry is used only here, as an offline label source):

  target_visible:
      target_visible = (occlusion-aware visible target pixels >= 1)

  target_localizable:
      target_visible AND goal_half_size_px >= 10.0 AND visible_pixels >= 16

The 10 px half-size threshold is calibrated on the existing Dev-Val Stage-2f
diagnostics: for visible states the coherent-map peak error is <= 42 px for
82.8% of states with half-size 10-15 px, 58.2% at 8-10 px and 33.3% at 6-8 px.
It is a pre-registered geometric rule, not tuned on the probe.

Visible pixels are computed with the simulator's own render order: the target
shape is drawn first, then every strictly nearer landmark's base shape is
drawn on top. The target's visible area is the target mask minus the nearer
occluders. Haze, brightness and sensor noise are not modelled; occlusion is
the dominant geometric visibility factor.

Usage:
  python v83_spatial_stage2/stage2g_labels.py --split train
  python v83_spatial_stage2/stage2g_labels.py --split val --max-episodes 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENV, TRAIN  # noqa: E402
from env import ToyUAVEnv  # noqa: E402
from token_data import _sample_pose_around_target, _set_pose  # noqa: E402

SAMPLE_NATURAL = 0
SAMPLE_HARD_POSITIVE = 1
SAMPLE_BOUNDARY_NEGATIVE = 2
SAMPLE_DISTRACTOR_NEGATIVE = 3
HARD_PLAN = (
    SAMPLE_HARD_POSITIVE,
    SAMPLE_HARD_POSITIVE,
    SAMPLE_BOUNDARY_NEGATIVE,
    SAMPLE_BOUNDARY_NEGATIVE,
    SAMPLE_DISTRACTOR_NEGATIVE,
    SAMPLE_DISTRACTOR_NEGATIVE,
)
HALF_SIZE_MIN_PX = 10.0
VISIBLE_PIXEL_MIN = 16
IMAGE_SIZE = int(ENV.image_size)


def band_ranges():
    return {
        name: (float(dmin), float(dmax))
        for name, _w, dmin, dmax, _amin, _amax in TRAIN.trajectory_mix
    }


def split_seed_start(data_root, split):
    manifest_path = Path(data_root) / "generation_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("requested", []):
            if str(item.get("split")) == split:
                return int(item["seed_start"])
    return 1_000_000 if split == "train" else 2_000_000


def visible_target_pixels(env):
    """Occlusion-aware target pixels under the simulator's draw order."""
    projection = env.goal_projection()
    if projection is None:
        return 0, 0, 0.0
    px, py, slant, half_size = projection
    target = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(target)
    env._draw_shape(
        draw, env.goal_landmark.shape, px, py, half_size, (1, 1, 1)
    )
    target_mask = np.asarray(target)[:, :, 0] > 0
    target_area = int(target_mask.sum())
    if target_area == 0:
        return 0, 0, 0.0

    occluders = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(occluders)
    for landmark in env.landmarks:
        if landmark is env.goal_landmark:
            continue
        other = env._project(landmark.position, landmark.physical_size_m)
        if other is None:
            continue
        if float(other[2]) >= float(slant):
            continue
        env._draw_shape(
            draw, landmark.shape, other[0], other[1], other[3], (1, 1, 1)
        )
    occluder_mask = np.asarray(occluders)[:, :, 0] > 0
    visible = int((target_mask & ~occluder_mask).sum())
    return visible, target_area, float(visible / max(target_area, 1))


def target_labels(visible_pixels, half_size_px):
    """Pure, testable target_visible / target_localizable definition."""
    visible = int(int(visible_pixels) >= 1)
    localizable = int(
        visible
        and float(half_size_px) >= HALF_SIZE_MIN_PX
        and int(visible_pixels) >= VISIBLE_PIXEL_MIN
    )
    return visible, localizable


def label_row(env, episode_id, step_id, sample_type, band):
    projection = env.goal_projection()
    visible_pixels, target_area, visible_fraction = visible_target_pixels(env)
    if projection is None:
        px = py = half_size = slant = float("nan")
    else:
        px, py, slant, half_size = projection
    target_visible, target_localizable = target_labels(
        visible_pixels, half_size
    )
    return {
        "episode_id": int(episode_id),
        "step_id": int(step_id),
        "sample_type": int(sample_type),
        "band": str(band),
        "distance_m": float(env.distance_to_goal),
        "goal_visible_geom": int(projection is not None),
        "goal_px": float(px),
        "goal_py": float(py),
        "goal_slant_m": float(slant),
        "goal_half_size_px": float(half_size),
        "visible_pixels": int(visible_pixels),
        "target_area_px": int(target_area),
        "visible_fraction": float(visible_fraction),
        "target_visible": int(target_visible),
        "target_localizable": int(target_localizable),
    }


def sample_hard_state(env, sample_type, rng):
    """Label-only copy of token_data._collect_hard_state's pose sampling."""
    if sample_type == SAMPLE_HARD_POSITIVE:
        return _sample_pose_around_target(
            env,
            env.goal,
            TRAIN.arrival_positive_distance_range_m,
            rng,
            TRAIN.arrival_pose_yaw_jitter_deg,
            require_goal_visible=True,
        )
    if sample_type == SAMPLE_BOUNDARY_NEGATIVE:
        return _sample_pose_around_target(
            env,
            env.goal,
            TRAIN.arrival_boundary_negative_range_m,
            rng,
            TRAIN.arrival_pose_yaw_jitter_deg,
            require_goal_visible=True,
        )
    candidates = [
        landmark for landmark in env.landmarks
        if 100.0 <= landmark.x <= env.world_size - 100.0
        and 100.0 <= landmark.y <= env.world_size - 100.0
    ]
    rng.shuffle(candidates)
    for landmark in candidates[:80]:
        trial = _sample_pose_around_target(
            env,
            landmark.position,
            TRAIN.arrival_distractor_near_range_m,
            rng,
            TRAIN.arrival_pose_yaw_jitter_deg,
            require_goal_visible=False,
        )
        if trial is None:
            continue
        if float(np.linalg.norm(env.position - env.goal)) < float(
            TRAIN.arrival_distractor_goal_min_distance_m
        ):
            continue
        if env._project(landmark.position, landmark.physical_size_m) is None:
            continue
        return trial
    return None


def replay_split(data_dir, seed_start, max_episodes=None):
    stats = json.loads(
        (Path(data_dir) / "dataset_stats.json").read_text(encoding="utf-8")
    )
    trajectories = stats["trajectories"]
    if max_episodes is not None:
        trajectories = trajectories[: int(max_episodes)]
    ranges = band_ranges()
    env = ToyUAVEnv()
    arrival_rng = np.random.default_rng(int(seed_start) + 99173)
    rows = []
    for episode_id, record in enumerate(trajectories):
        band = str(record["band"])
        dmin, dmax = ranges[band]
        env.reset(
            seed=int(record["seed"]),
            goal_distance_range_m=(dmin, dmax),
            render_observation=False,
        )
        step_id = 0
        done = False
        while not done:
            distance = float(env.distance_to_goal)
            arrival = float(distance < env.stop_radius)
            # Match token_data's arrival_train_mask RNG consumption exactly.
            if not (
                arrival > 0.5
                or distance <= float(TRAIN.arrival_natural_near_radius_m)
            ):
                arrival_rng.random()
            rows.append(
                label_row(env, episode_id, step_id, SAMPLE_NATURAL, band)
            )
            action = int(env.oracle_action())
            _, _success, done = env.step(
                action, render_observation=False
            )
            step_id += 1
        if step_id != int(record["steps"]):
            raise RuntimeError(
                f"episode {episode_id}: replay steps {step_id} != "
                f"recorded {record['steps']}"
            )
        for hard_index, sample_type in enumerate(HARD_PLAN):
            sampled = sample_hard_state(env, sample_type, arrival_rng)
            if sampled is None:
                raise RuntimeError(
                    f"episode {episode_id}: hard sample {hard_index} failed"
                )
            position, yaw, _distance, _bearing = sampled
            _set_pose(env, position, yaw)
            rows.append(
                label_row(
                    env, episode_id, -(hard_index + 1), sample_type, band
                )
            )
        if (episode_id + 1) % 50 == 0 or episode_id + 1 == len(trajectories):
            print(
                f"  replayed {episode_id + 1}/{len(trajectories)} episodes",
                flush=True,
            )
    return rows


def load_shard_meta(data_dir, max_episodes=None):
    import torch

    meta = {}
    for path in sorted(Path(data_dir).glob("shard_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        for i in range(len(data["episode_ids"])):
            episode = int(data["episode_ids"][i])
            if max_episodes is not None and episode >= int(max_episodes):
                continue
            key = (episode, int(data["step_ids"][i]))
            meta[key] = (
                int(data["goal_visible"][i]),
                float(data["distance_to_goal_m"][i]),
                int(data["sample_type"][i]),
            )
    return meta


def verify(rows, meta):
    row_keys = {(row["episode_id"], row["step_id"]) for row in rows}
    meta_keys = set(meta)
    missing = sorted(meta_keys - row_keys)[:10]
    extra = sorted(row_keys - meta_keys)[:10]
    visible_mismatch = 0
    distance_mismatch = 0
    type_mismatch = 0
    max_distance_error = 0.0
    for row in rows:
        key = (row["episode_id"], row["step_id"])
        if key not in meta:
            continue
        visible, distance, sample_type = meta[key]
        if int(row["goal_visible_geom"]) != int(visible):
            visible_mismatch += 1
        error = abs(float(row["distance_m"]) - float(distance))
        max_distance_error = max(max_distance_error, error)
        if error > 1e-3:
            distance_mismatch += 1
        if int(row["sample_type"]) != int(sample_type):
            type_mismatch += 1
    return {
        "rows": len(rows),
        "shard_samples": len(meta),
        "missing": missing,
        "extra": extra,
        "visible_mismatch": visible_mismatch,
        "distance_mismatch_gt_1e3": distance_mismatch,
        "sample_type_mismatch": type_mismatch,
        "max_distance_error_m": max_distance_error,
    }


def summarize(rows):
    total = len(rows)
    natural = [row for row in rows if row["sample_type"] == SAMPLE_NATURAL]
    hard = [row for row in rows if row["sample_type"] != SAMPLE_NATURAL]
    summary = {
        "rows": total,
        "natural": len(natural),
        "hard": len(hard),
        "bands": {},
        "label_definition": {
            "target_visible": "occlusion-aware visible_pixels >= 1",
            "target_localizable": (
                "target_visible AND goal_half_size_px >= "
                f"{HALF_SIZE_MIN_PX} AND visible_pixels >= {VISIBLE_PIXEL_MIN}"
            ),
        },
    }
    for band in ("short", "medium", "long"):
        subset = [row for row in rows if row["band"] == band]
        if not subset:
            continue
        summary["bands"][band] = {
            "rows": len(subset),
            "target_visible_rate": float(
                np.mean([row["target_visible"] for row in subset])
            ),
            "target_localizable_rate": float(
                np.mean([row["target_localizable"] for row in subset])
            ),
            "goal_visible_geom_rate": float(
                np.mean([row["goal_visible_geom"] for row in subset])
            ),
            "occluded_share_of_geometric": float(
                np.mean([
                    row["goal_visible_geom"] == 1
                    and row["target_visible"] == 0
                    for row in subset
                ])
            ),
            "median_visible_pixels_when_geom": float(
                np.median([
                    row["visible_pixels"] for row in subset
                    if row["goal_visible_geom"] == 1
                ])
            ) if any(row["goal_visible_geom"] for row in subset) else 0.0,
        }
    summary["overall"] = {
        "target_visible_rate": float(
            np.mean([row["target_visible"] for row in rows])
        ),
        "target_localizable_rate": float(
            np.mean([row["target_localizable"] for row in rows])
        ),
        "goal_visible_geom_rate": float(
            np.mean([row["goal_visible_geom"] for row in rows])
        ),
    }
    return summary


def write_csv(path, rows):
    fields = [
        "episode_id", "step_id", "sample_type", "band", "distance_m",
        "goal_visible_geom", "goal_px", "goal_py", "goal_slant_m",
        "goal_half_size_px", "visible_pixels", "target_area_px",
        "visible_fraction", "target_visible", "target_localizable",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path(
        "runs/v83_spatial_stage2/datasets"
    ))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_root / (
        "train_600" if args.split == "train" else "dev_val_180"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_start = split_seed_start(args.data_root, args.split)
    rows = replay_split(
        data_dir, seed_start, max_episodes=args.max_episodes
    )
    verify_result = None
    if not args.skip_verify:
        meta = load_shard_meta(data_dir, max_episodes=args.max_episodes)
        verify_result = verify(rows, meta)
        if (
            verify_result["missing"]
            or verify_result["extra"]
            or verify_result["visible_mismatch"]
            or verify_result["distance_mismatch_gt_1e3"]
            or verify_result["sample_type_mismatch"]
        ):
            raise SystemExit(
                "Replay does not match shards: "
                + json.dumps(verify_result, indent=2)
            )
    summary = summarize(rows)
    summary["verification"] = verify_result
    write_csv(output_dir / f"labels_{args.split}.csv", rows)
    (output_dir / f"labels_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "split": args.split,
        "rows": len(rows),
        "verification": verify_result,
        "overall": summary["overall"],
        "output": str(output_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
