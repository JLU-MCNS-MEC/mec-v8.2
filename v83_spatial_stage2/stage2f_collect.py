"""Stage-2f: collect observable Goal-Reacquisition features on saved states.

Replays the exact M0@0.7 and M1@0.7 action sequences from Stage-2d and, at every
state, computes the observable feature set from `stage2f_features.py`. The
simulator goal position / true visibility / true distance / heading error are
recorded **as offline labels only**; they are stored in separate columns and
never passed to the recovery policy.

No training, no checkpoint modification, Final Test 300 not read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER, ENV
from env import FORWARD, LEFT, RIGHT, STOP, ToyUAVEnv, wrap_angle
from gbs_model_v3 import build_gbs_model
from gbs_network import build_random_gbs_layout, global_network_state
from token_data import encode_multiscale_goal_tokens, goal_conditioned_spatial_heatmap
from utils import image_to_tensor
from visual_grounding_v83 import static_grounding
from v83_spatial_stage2.stage2d_common import (
    STAGE2_ROOT,
    VAL_SPECS,
    device_summary,
    file_sha256,
    load_dev_specs,
    load_encoder,
    write_csv,
    write_json,
)
from v83_spatial_stage2.stage2f_features import (
    extract_observable_features,
    feature_names,
)
from v83_spatial_training.spatial_mec import SpatialAwareMEC


M0_CKPT = STAGE2_ROOT / "stage2a_tm_20261018T044255Z/Transformer-M/M0/best.pt"
M1_CKPT = STAGE2_ROOT / "stage2a_tm_20261018T044255Z/Transformer-M/M1/best.pt"
M0_RUN = STAGE2_ROOT / "stage2d/r4_20261019T002626Z"
M1_RUN = STAGE2_ROOT / "stage2d/m1_threshold_20261019T013737Z"
ORACLE_TURN_RAD = math.radians(7.5)


def load_model(mode, checkpoint, device):
    model = SpatialAwareMEC(build_gbs_model("Transformer-M"), mode).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def read_steps(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def steps_by_episode(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["episode_index"]), []).append(row)
    for item in grouped.values():
        item.sort(key=lambda row: int(row["step"]))
    return grouped


def oracle_move_action(env):
    dx = float(env.goal[0] - env.x)
    dy = float(env.goal[1] - env.y)
    heading_error = wrap_angle(math.atan2(dy, dx) - env.yaw)
    if heading_error > ORACLE_TURN_RAD:
        return LEFT
    if heading_error < -ORACLE_TURN_RAD:
        return RIGHT
    return FORWARD


def projection_vector(env):
    projection = env.goal_projection()
    if projection is None:
        return None
    px, py, slant_distance, half_size = projection
    return {
        "goal_visible": 1,
        "goal_px": float(px),
        "goal_py": float(py),
        "goal_slant_m": float(slant_distance),
        "goal_half_size_px": float(half_size),
    }


def peak_error_px(
    peak_x, peak_y, projection, image_size=ENV.image_size,
    grid=ENCODER.grid_size,
):
    if projection is None:
        return float("nan")
    # peak_x/y are linspace(-1, 1, grid) positions of cell indices, so map
    # back to the cell CENTRE in image pixels.
    pixel_x = (
        ((float(peak_x) + 1.0) * 0.5 * (grid - 1) + 0.5) * image_size / grid
    )
    pixel_y = (
        ((float(peak_y) + 1.0) * 0.5 * (grid - 1) + 0.5) * image_size / grid
    )
    return math.hypot(
        pixel_x - projection["goal_px"], pixel_y - projection["goal_py"]
    )


def replay_episode(
    owner,
    spec,
    logged_steps,
    m0,
    m1,
    encoder,
    device,
):
    env = ToyUAVEnv()
    obs = env.reset(
        seed=spec["seed"],
        goal_distance_range_m=spec["distance_range"],
        render_observation=True,
    )
    nodes = build_random_gbs_layout(
        seed=spec["network_layout_seed"], verbose=False
    )
    goal_tokens = encode_multiscale_goal_tokens(
        env.get_goal_reference_multiscale_rgbs(), encoder, device
    ).unsqueeze(0).to(device, dtype=torch.float32)

    previous_coherent = None
    previous_action = None
    rows = []
    max_distance_error = 0.0
    move_mismatch = 0
    arrival_error = 0.0

    for logged in logged_steps:
        step = int(logged["step"])
        if step != env.step_count:
            raise RuntimeError(
                f"{owner} episode {spec['episode_index']} step {step}: "
                f"env step {env.step_count}"
            )
        distance = float(env.distance_to_goal)
        max_distance_error = max(
            max_distance_error, abs(distance - float(logged["distance_m"]))
        )
        current_tokens = encoder(image_to_tensor(obs, device))
        heatmap = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
        network_np = global_network_state(
            nodes, env.position,
            num_tokens=ENCODER.num_tokens, token_dim=ENCODER.token_dim,
        ).astype("float32")
        network = torch.from_numpy(network_np).unsqueeze(0).to(device)

        with torch.inference_mode():
            baseline, coherent = static_grounding(current_tokens, goal_tokens)
            m0_components = m0.base.forward_components(
                current_tokens, goal_tokens, heatmap, network
            )
            if owner == "M0":
                owner_components = m0_components
            else:
                maps = torch.stack([baseline, coherent], dim=1)
                owner_components = m1.forward_components(
                    current_tokens, goal_tokens, heatmap, network,
                    spatial_maps=maps,
                )
            features = extract_observable_features(
                current_tokens, goal_tokens, heatmap, baseline, coherent,
                previous_coherent, previous_action,
                m0_components["move_logits"], m0_components["arrival_logit"],
            )
            owner_move = int(
                torch.argmax(owner_components["move_logits"], -1).item()
            )
            owner_arrival = float(
                torch.sigmoid(owner_components["arrival_logit"]).item()
            )
            if owner_move != int(logged["move_action"]):
                move_mismatch += 1
            arrival_error = max(
                arrival_error,
                abs(owner_arrival - float(logged["arrival_probability"])),
            )

            projection = projection_vector(env)
            dx = float(env.goal[0] - env.x)
            dy = float(env.goal[1] - env.y)
            heading_error = wrap_angle(math.atan2(dy, dx) - env.yaw)
            row = {
                "trajectory": owner,
                "episode_index": int(spec["episode_index"]),
                "seed": int(spec["seed"]),
                "band": str(spec["band"]),
                "step": step,
                "distance_m": distance,
                "goal_visible": int(projection is not None),
                "goal_px": (
                    projection["goal_px"] if projection else float("nan")
                ),
                "goal_py": (
                    projection["goal_py"] if projection else float("nan")
                ),
                "goal_half_size_px": (
                    projection["goal_half_size_px"] if projection else float("nan")
                ),
                "heading_error_deg": math.degrees(heading_error),
                "oracle_move_action": oracle_move_action(env),
                "stop_ready": int(distance < env.stop_radius),
                "executed_action": int(logged["action"]),
                "logged_stop": int(logged["stop"]),
                "logged_arrival_probability": float(logged["arrival_probability"]),
                "owner_move_action": owner_move,
                "owner_arrival_probability": owner_arrival,
                "previous_action": (
                    -1 if previous_action is None else int(previous_action)
                ),
                "coherent_peak_error_px": peak_error_px(
                    features["coherent_peak_x"],
                    features["coherent_peak_y"],
                    projection,
                ),
                "baseline_peak_error_px": peak_error_px(
                    features["baseline_peak_x"],
                    features["baseline_peak_y"],
                    projection,
                ),
            }
            row.update(features)
            rows.append(row)

        executed_action = int(logged["action"])
        obs, _, done = env.step(
            executed_action, render_observation=(executed_action != STOP)
        )
        previous_coherent = coherent[0].detach()
        previous_action = executed_action
        if done:
            break

    if len(rows) != len(logged_steps):
        raise RuntimeError(
            f"{owner} episode {spec['episode_index']}: replayed {len(rows)} "
            f"steps, expected {len(logged_steps)}"
        )
    return rows, {
        "trajectory": owner,
        "episode_index": int(spec["episode_index"]),
        "max_distance_error_m": max_distance_error,
        "move_action_mismatches": int(move_mismatch),
        "max_arrival_probability_error": float(arrival_error),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    parser.add_argument("--max-episodes", type=int, default=180)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--m0-run", type=Path, default=M0_RUN)
    parser.add_argument("--m1-run", type=Path, default=M1_RUN)
    args = parser.parse_args()
    if args.smoke:
        args.max_episodes = min(args.max_episodes, 6)
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    specs = load_dev_specs(args.val_specs, args.max_episodes)
    encoder = load_encoder(device)
    m0 = load_model("M0", M0_CKPT, device)
    m1 = load_model("M1", M1_CKPT, device)
    print(
        f"M0={M0_CKPT}\nM1={M1_CKPT}\n"
        f"episodes={len(specs)} device={device}",
        flush=True,
    )

    m0_steps = steps_by_episode(read_steps(args.m0_run / "steps/threshold_0.7.csv"))
    m1_steps = steps_by_episode(read_steps(args.m1_run / "steps/M1@0.7.csv"))
    all_rows = []
    checks = []
    started = time.time()
    for position, spec in enumerate(specs):
        episode = int(spec["episode_index"])
        for owner, grouped in (("M0", m0_steps), ("M1", m1_steps)):
            if episode not in grouped:
                raise RuntimeError(f"Missing {owner} log for episode {episode}")
            rows, check = replay_episode(
                owner, spec, grouped[episode], m0, m1, encoder, device
            )
            all_rows.extend(rows)
            checks.append(check)
        if (position + 1) % 10 == 0:
            print(
                f"  collected {position + 1}/{len(specs)} episodes "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )

    bad_distance = [c for c in checks if c["max_distance_error_m"] > 1e-4]
    bad_move = [c for c in checks if c["move_action_mismatches"] > 0]
    bad_arrival = [
        c for c in checks if c["max_arrival_probability_error"] > 1e-5
    ]
    if bad_distance or bad_move or bad_arrival:
        write_json(args.output_dir / "replay_check.json", {
            "distance_mismatches": bad_distance[:20],
            "move_action_mismatches": bad_move[:20],
            "arrival_mismatches": bad_arrival[:20],
        })
        raise SystemExit("Replay verification failed")

    write_csv(args.output_dir / "features.csv", all_rows)
    write_json(args.output_dir / "replay_check.json", {
        "trajectory_passes": len(checks),
        "max_distance_error_m": max(
            (c["max_distance_error_m"] for c in checks), default=0.0
        ),
        "move_action_mismatch_passes": 0,
        "max_arrival_probability_error": max(
            (c["max_arrival_probability_error"] for c in checks), default=0.0
        ),
        "states": len(all_rows),
        "feature_columns": feature_names(),
        "note": (
            "Exact-state replay; true goal labels are stored in separate "
            "columns and are not policy inputs."
        ),
    })
    write_json(args.output_dir / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "V8.3 Stage-2f Goal Reacquisition feature collection (eval only)",
        "command": sys.argv,
        "m0_checkpoint": str(M0_CKPT),
        "m0_checkpoint_sha256": file_sha256(M0_CKPT),
        "m1_checkpoint": str(M1_CKPT),
        "m1_checkpoint_sha256": file_sha256(M1_CKPT),
        "val_specs": str(args.val_specs),
        "val_specs_sha256": file_sha256(args.val_specs),
        "episodes": int(len(specs)),
        "states": int(len(all_rows)),
        "training": "none; frozen checkpoints only",
        "final_test": "sealed; not read",
        "device": device_summary(device),
    })
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
