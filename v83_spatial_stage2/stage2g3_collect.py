"""Stage-2g.3 real closed-loop feature collection for the P1-2 probe.

Train 600: run the frozen Transformer-M M0 checkpoint with Stop threshold 0.7
from the dataset seeds/layouts and record every decision state.

Dev Val 180: replay the saved M0@0.7 action sequence (Stage-2d r4 threshold
0.7), which is an exact closed-loop replay, and record the same features. The
replay verifies distance/visibility/action against the saved steps.

Per state the collector saves compact observable data only:
  * current 2,858-d feature vector;
  * two history frames (74 compact visual features + action one-hot + dt);
  * offline labels (distance, projection, visible pixels, visibility,
    localizability, near-target-invisible, heading, oracle action).

No RGB tokens, images or full token datasets are written. No probe or recovery
rule changes the trajectory. Final Test 300 is not read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER, ENV, TRAIN  # noqa: E402
from env import FORWARD, LEFT, RIGHT, STOP, ToyUAVEnv, wrap_angle  # noqa: E402
from gbs_network import (  # noqa: E402
    build_random_gbs_layout,
    global_network_state,
)
from token_data import (  # noqa: E402
    encode_multiscale_goal_tokens,
    goal_conditioned_spatial_heatmap,
)
from utils import image_to_tensor  # noqa: E402
from visual_grounding_v83 import static_grounding  # noqa: E402
from v83_spatial_stage2.stage2d_common import (  # noqa: E402
    BAND_ID,
    VAL_SPECS,
    file_sha256,
    load_dev_specs,
)
from v83_spatial_stage2.stage2f_collect import (  # noqa: E402
    M0_CKPT,
    load_model,
    oracle_move_action,
    projection_vector,
)
from v83_spatial_stage2.stage2f_recovery_eval import STOP_THRESHOLD  # noqa: E402
from v83_spatial_stage2.stage2g_features import (  # noqa: E402
    compute_feature_vector,
    feature_names,
)
from v83_spatial_stage2.stage2g_labels import (  # noqa: E402
    target_labels,
    visible_target_pixels,
)
from v83_spatial_stage2.stage2g1_common import (  # noqa: E402
    COMPACT_FEATURE_NAMES,
    compact_indices,
)
from v83_spatial_training.eval_closedloop_spatial import (  # noqa: E402
    load_encoder,
)

BAND_NAMES = {0: "short", 1: "medium", 2: "long"}
VARIANT_NAMES = {0: "near_visible", 1: "near_invisible"}


def band_ranges():
    return {
        name: (float(dmin), float(dmax))
        for name, _w, dmin, dmax, _amin, _amax in TRAIN.trajectory_mix
    }


def step_rows_by_episode(path):
    grouped = {}
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(int(row["episode_index"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
    return grouped


def record_state(env, obs, model, encoder, device, goal_tokens, nodes,
                 use_model, saved_row, compact_idx):
    current_tokens = encoder(image_to_tensor(obs, device))
    heatmap = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
    if use_model:
        network_np = global_network_state(
            nodes, env.position,
            num_tokens=ENCODER.num_tokens, token_dim=ENCODER.token_dim,
        ).astype("float32")
        network = torch.from_numpy(network_np).unsqueeze(0).to(device)
        with torch.inference_mode():
            baseline, coherent = static_grounding(current_tokens, goal_tokens)
            components = model.base.forward_components(
                current_tokens, goal_tokens, heatmap, network
            )
            arrival_probability = float(
                torch.sigmoid(components["arrival_logit"]).item()
            )
            base_move = int(
                torch.argmax(components["move_logits"], -1).item()
            )
            feature_tensor = compute_feature_vector(
                current_tokens, goal_tokens, heatmap, baseline, coherent
            )
            features = feature_tensor.squeeze(0).cpu().numpy()
        base_action = (
            STOP if arrival_probability >= STOP_THRESHOLD else base_move
        )
    else:
        features = compute_feature_vector(
            current_tokens, goal_tokens, heatmap,
            *static_grounding(current_tokens, goal_tokens),
        ).squeeze(0).cpu().numpy()
        arrival_probability = float(saved_row["arrival_probability"])
        base_action = int(saved_row["action"])
    projection = projection_vector(env)
    visible_pixels, target_area, visible_fraction = visible_target_pixels(env)
    if projection is None:
        px = py = slant = half_size = float("nan")
        goal_visible = 0
    else:
        px = projection["goal_px"]
        py = projection["goal_py"]
        slant = projection["goal_slant_m"]
        half_size = projection["goal_half_size_px"]
        goal_visible = 1
    target_visible, target_localizable = target_labels(
        visible_pixels, half_size
    )
    distance = float(env.distance_to_goal)
    dx = float(env.goal[0] - env.x)
    dy = float(env.goal[1] - env.y)
    heading_error = math.degrees(
        wrap_angle(math.atan2(dy, dx) - env.yaw)
    )
    return features, {
        "distance_m": distance,
        "goal_visible_geom": goal_visible,
        "goal_px": float(px),
        "goal_py": float(py),
        "goal_slant_m": float(slant),
        "goal_half_size_px": float(half_size),
        "visible_pixels": int(visible_pixels),
        "target_area_px": int(target_area),
        "visible_fraction": float(visible_fraction),
        "target_visible": int(target_visible),
        "target_localizable": int(target_localizable),
        "near_target_invisible": int(
            distance < 100.0 and target_visible == 0
        ),
        "heading_error_deg": float(heading_error),
        "oracle_move_action": int(oracle_move_action(env)),
        "arrival_probability": float(arrival_probability),
        "action": int(base_action),
        "stop": int(int(base_action) == STOP),
    }


def replay_episode(spec, steps, encoder, device, compact_idx):
    env = ToyUAVEnv()
    obs = env.reset(
        seed=spec["seed"],
        goal_distance_range_m=spec["distance_range"],
        render_observation=True,
    )
    goal_tokens = encode_multiscale_goal_tokens(
        env.get_goal_reference_multiscale_rgbs(), encoder, device
    ).unsqueeze(0).to(device, dtype=torch.float32)
    features = []
    labels = {}
    actions = []
    for row in steps:
        if int(row["step"]) != env.step_count:
            raise RuntimeError(
                f"episode {spec['episode_index']} step {row['step']} "
                f"!= env {env.step_count}"
            )
        distance = float(env.distance_to_goal)
        saved_distance = float(row["distance_m"])
        if abs(distance - saved_distance) > 1e-4:
            raise RuntimeError(
                f"distance mismatch {distance} vs {saved_distance}"
            )
        projection = projection_vector(env)
        saved_visible = int(row["goal_visible"])
        if int(projection is not None) != saved_visible:
            raise RuntimeError(
                f"visibility mismatch at episode {spec['episode_index']} "
                f"step {row['step']}"
            )
        action = int(row["action"])
        feature_vector, label = record_state(
            env, obs, None, encoder, device, goal_tokens, None,
            use_model=False, saved_row=row, compact_idx=compact_idx,
        )
        features.append(feature_vector)
        actions.append(action)
        for key, value in label.items():
            labels.setdefault(key, []).append(value)
        obs, _success, done = env.step(
            action, render_observation=(action != STOP)
        )
        if done:
            break
    return np.asarray(features, dtype=np.float32), np.asarray(
        actions, dtype=np.int64
    ), {key: np.asarray(value) for key, value in labels.items()}


def rollout_episode(record, model, encoder, device, compact_idx):
    ranges = band_ranges()
    band = str(record["band"])
    env = ToyUAVEnv()
    obs = env.reset(
        seed=int(record["seed"]),
        goal_distance_range_m=ranges[band],
        render_observation=True,
    )
    nodes = build_random_gbs_layout(
        seed=int(record["network_layout_seed"]), verbose=False
    )
    goal_tokens = encode_multiscale_goal_tokens(
        env.get_goal_reference_multiscale_rgbs(), encoder, device
    ).unsqueeze(0).to(device, dtype=torch.float32)
    features = []
    labels = {}
    actions = []
    for _step in range(int(ENV.max_steps)):
        feature_vector, label = record_state(
            env, obs, model, encoder, device, goal_tokens, nodes,
            use_model=True, saved_row=None, compact_idx=compact_idx,
        )
        action = int(label["action"])
        features.append(feature_vector)
        actions.append(action)
        for key, value in label.items():
            labels.setdefault(key, []).append(value)
        obs, _success, done = env.step(
            action, render_observation=(action != STOP)
        )
        if done:
            break
    return np.asarray(features, dtype=np.float32), np.asarray(
        actions, dtype=np.int64
    ), {key: np.asarray(value) for key, value in labels.items()}


def derive_history(features, actions, compact_idx):
    n = len(features)
    compact = features[:, compact_idx]
    history = np.zeros((n, 2, len(compact_idx)), dtype=np.float32)
    history_actions = np.full((n, 2), -1, dtype=np.int8)
    history_dt = np.zeros((n, 2), dtype=np.float32)
    history_valid = np.zeros(n, dtype=np.int8)
    for t in range(n):
        if t >= 1:
            history[t, 0] = compact[t - 1]
            history_actions[t, 0] = int(actions[t - 1])
            history_dt[t, 0] = 1.0
        if t >= 2:
            history[t, 1] = compact[t - 2]
            history_actions[t, 1] = int(actions[t - 2])
            history_dt[t, 1] = 1.0
        history_valid[t] = int(t >= 2)
    return history, history_actions, history_dt, history_valid


def save_episode(output_dir, episode_id, seed, band_id, layout_seed,
                 step_ids, features, actions, labels, compact_idx):
    history, history_actions, history_dt, history_valid = derive_history(
        features, actions, compact_idx
    )
    labels = dict(labels)
    labels.pop("action", None)  # the action array is passed explicitly
    path = output_dir / f"episode_{episode_id:04d}.npz"
    temp = output_dir / f".episode_{episode_id:04d}.tmp.npz"
    np.savez(
        temp,
        episode_id=np.int64(episode_id),
        seed=np.int64(seed),
        band=np.int64(band_id),
        network_layout_seed=np.int64(layout_seed),
        step=np.asarray(step_ids, dtype=np.int64),
        action=np.asarray(actions, dtype=np.int64),
        current_features=features.astype(np.float32),
        history_features=history,
        history_actions=history_actions,
        history_dt=history_dt,
        history_valid=history_valid,
        **{key: value for key, value in labels.items()},
    )
    os.replace(temp, path)
    return path


def load_manifest(output_dir):
    path = output_dir / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(output_dir, manifest):
    path = output_dir / "manifest.json"
    temp = output_dir / ".manifest.tmp.json"
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--val-steps", type=Path,
        default=Path(
            "runs/v83_spatial_stage2/stage2d/r4_20261019T002626Z/"
            "steps/threshold_0.7.csv"
        ),
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("runs/v83_spatial_stage2/datasets"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.resume:
        raise SystemExit(f"Refusing to overwrite {output_dir}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest(output_dir) if args.resume else None
    if manifest is None:
        manifest = {
            "split": args.split,
            "source": (
                "train_m0_rollout" if args.split == "train"
                else "val_m0_replay"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "feature_dim": len(feature_names()),
            "compact_dim": len(COMPACT_FEATURE_NAMES),
            "history_order": ["t-1", "t-2"],
            "stop_threshold": float(STOP_THRESHOLD),
            "m0_checkpoint": str(M0_CKPT),
            "m0_checkpoint_sha256": file_sha256(M0_CKPT),
            "completed": [],
            "failed": [],
            "states": 0,
            "elapsed_s": 0.0,
            "feature_names": feature_names(),
            "compact_feature_names": list(COMPACT_FEATURE_NAMES),
            "label_definition": {
                "target_visible": "occlusion-aware visible_pixels >= 1",
                "target_localizable": (
                    "target_visible AND goal_half_size_px >= 10 "
                    "AND visible_pixels >= 16"
                ),
                "near_target_invisible": (
                    "true distance < 100 m AND target_visible == 0"
                ),
            },
        }
    completed = set(int(x) for x in manifest.get("completed", []))
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    encoder = load_encoder(device)
    compact_idx = compact_indices(feature_names())
    if args.split == "train":
        stats = json.loads(
            (args.data_root / "train_600" / "dataset_stats.json").read_text(
                encoding="utf-8"
            )
        )
        records = stats["trajectories"]
        model = load_model("M0", M0_CKPT, device)
        if args.max_episodes is not None:
            records = records[: int(args.max_episodes)]
    else:
        specs = load_dev_specs(VAL_SPECS, 180)
        spec_by_episode = {
            int(spec["episode_index"]): spec for spec in specs
        }
        steps_by_episode = step_rows_by_episode(args.val_steps)
        model = None
        if args.max_episodes is not None:
            specs = specs[: int(args.max_episodes)]
    started = time.time()
    total = len(records) if args.split == "train" else len(specs)
    for position, item in enumerate(
        records if args.split == "train" else specs
    ):
        if args.split == "train":
            episode_id = int(position)
            record = item
            if episode_id in completed:
                continue
            seed = int(record["seed"])
            band_id = int(BAND_ID[str(record["band"])])
            layout_seed = int(record["network_layout_seed"])
            features, actions, labels = rollout_episode(
                record, model, encoder, device, compact_idx
            )
        else:
            spec = item
            episode_id = int(spec["episode_index"])
            if episode_id in completed:
                continue
            if episode_id not in steps_by_episode:
                manifest["failed"].append({
                    "episode_id": episode_id,
                    "error": "missing saved steps",
                })
                continue
            seed = int(spec["seed"])
            band_id = int(BAND_ID[str(spec["band"])])
            layout_seed = int(spec["network_layout_seed"])
            features, actions, labels = replay_episode(
                spec, steps_by_episode[episode_id], encoder, device,
                compact_idx,
            )
        step_ids = np.arange(len(features), dtype=np.int64)
        try:
            save_episode(
                output_dir, episode_id, seed, band_id, layout_seed,
                step_ids, features, actions, labels, compact_idx,
            )
        except Exception as error:  # noqa: BLE001
            manifest["failed"].append({
                "episode_id": episode_id,
                "error": repr(error),
            })
            write_manifest(output_dir, manifest)
            raise
        completed.add(episode_id)
        manifest["completed"] = sorted(completed)
        manifest["states"] += int(len(features))
        manifest["elapsed_s"] = float(time.time() - started)
        write_manifest(output_dir, manifest)
        if (position + 1) % 10 == 0 or position + 1 == total:
            print(
                f"  {args.split}: {position + 1}/{total} episodes, "
                f"{manifest['states']} states, "
                f"{time.time() - started:.1f}s",
                flush=True,
            )
    manifest["elapsed_s"] = float(time.time() - started)
    manifest["completed_count"] = len(completed)
    manifest["state_count"] = int(manifest["states"])
    write_manifest(output_dir, manifest)
    print(json.dumps({
        "split": args.split,
        "episodes": len(completed),
        "states": manifest["states"],
        "elapsed_s": round(manifest["elapsed_s"], 1),
        "output": str(output_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
