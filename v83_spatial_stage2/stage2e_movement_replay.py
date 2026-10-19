"""Stage-2e: exact state replay for M0@0.7 vs M1@0.7 Movement diagnostics.

This is an evaluation-only diagnostic. It does not train, does not modify any
checkpoint, and never reads Final Test 300.

For every episode saved by the Stage-2d M0@0.7 and M1@0.7 closed-loop runs,
the recorded action sequence is replayed from the same seed. At each state the
script verifies that the reconstructed distance and the owner model's saved
move/arrival outputs match the log, then evaluates on that *identical state*:

  * the frozen M0 model (no spatial fusion);
  * the frozen M1 base (M1 weights with the fusion residual excluded);
  * the M1 fused model (base + SpatialFusionHead residual);
  * M0 and M1 base with horizontally mirrored current tokens/heatmap;
  * the M1 fusion head on horizontally mirrored spatial maps.

The mirror probes test whether the fusion head can distinguish left from right
at all. The fusion residual is also compared with the 3-way Movement oracle
(privileged heading error, diagnostic only) to label residual-induced action
changes as beneficial / harmful / neutral.

Outputs are analysis artifacts under runs/; nothing is written to data splits.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER, ENV
from env import (
    FORWARD,
    LEFT,
    RIGHT,
    STOP,
    ToyUAVEnv,
    wrap_angle,
)
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
from v83_spatial_training.spatial_mec import SpatialAwareMEC


M0_CKPT = STAGE2_ROOT / "stage2a_tm_20261018T044255Z/Transformer-M/M0/best.pt"
M1_CKPT = STAGE2_ROOT / "stage2a_tm_20261018T044255Z/Transformer-M/M1/best.pt"
M0_RUN = STAGE2_ROOT / "stage2d/r4_20261019T002626Z"
M1_RUN = STAGE2_ROOT / "stage2d/m1_threshold_20261019T013737Z"
ORACLE_TURN_RAD = math.radians(7.5)
SIDE_MARGIN_PX = 10.0


def load_model(mode, checkpoint, device):
    model = SpatialAwareMEC(build_gbs_model("Transformer-M"), mode).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def mirrored_tokens(tokens):
    """[B,24*24,D] -> horizontally mirrored token grid (row-major y,x)."""
    batch, count, dim = tokens.shape
    grid = int(round(math.sqrt(count)))
    if grid * grid != count:
        raise ValueError(f"Token count {count} is not a square grid")
    return tokens.reshape(batch, grid, grid, dim).flip(2).reshape(batch, count, dim)


def mirrored_map(maps):
    return maps.flip(-1)


def oracle_move_action(env):
    """Privileged 3-way heading oracle; STOP radius handled separately."""
    dx = float(env.goal[0] - env.x)
    dy = float(env.goal[1] - env.y)
    heading_error = wrap_angle(math.atan2(dy, dx) - env.yaw)
    if heading_error > ORACLE_TURN_RAD:
        return LEFT
    if heading_error < -ORACLE_TURN_RAD:
        return RIGHT
    return FORWARD


def goal_side(projection, image_size):
    if projection is None:
        return "occluded", float("nan")
    px = float(projection[0])
    center = image_size * 0.5
    if px < center - SIDE_MARGIN_PX:
        return "left", px
    if px > center + SIDE_MARGIN_PX:
        return "right", px
    return "center", px


def map_direction(maps):
    """Direction summaries of the [B,24,24] baseline/coherent maps."""
    grid = maps.shape[-1]
    weights = torch.softmax(maps.reshape(maps.shape[0], -1).float(), dim=-1)
    weights = weights.reshape(maps.shape[0], grid, grid)
    xs = torch.linspace(-1.0, 1.0, grid, device=maps.device, dtype=weights.dtype)
    centroid = (weights.sum(dim=1) * xs).sum(dim=1)
    left = weights[:, :, : grid // 2].sum(dim=(1, 2))
    right = weights[:, :, grid // 2 :].sum(dim=(1, 2))
    return centroid, left, right


def read_steps(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def steps_by_episode(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["episode_index"]), []).append(row)
    for rows_for_episode in grouped.values():
        rows_for_episode.sort(key=lambda item: int(item["step"]))
    return grouped


def _float_list(tensor):
    return [float(value) for value in tensor.reshape(-1).tolist()]


def replay_episode(
    owner,
    spec,
    logged_steps,
    m0,
    m1,
    encoder,
    device,
    threshold,
    coarse_feature_sink,
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

    rows = []
    max_distance_error = 0.0
    max_move_action_mismatch = 0
    max_arrival_error = 0.0

    for logged in logged_steps:
        step = int(logged["step"])
        if step != env.step_count:
            raise RuntimeError(
                f"{owner} episode {spec['episode_index']} step {step}: "
                f"environment is at step {env.step_count}"
            )
        logged_distance = float(logged["distance_m"])
        distance = float(env.distance_to_goal)
        max_distance_error = max(max_distance_error, abs(distance - logged_distance))

        current_tokens = encoder(image_to_tensor(obs, device))
        heatmap = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
        network_np = global_network_state(
            nodes, env.position,
            num_tokens=ENCODER.num_tokens, token_dim=ENCODER.token_dim,
        ).astype(np.float32)
        network = torch.from_numpy(network_np).unsqueeze(0).to(device)

        with torch.inference_mode():
            baseline, coherent = static_grounding(current_tokens, goal_tokens)
            maps = torch.stack([baseline, coherent], dim=1)
            mirror_maps = mirrored_map(maps)
            mirror_tokens = mirrored_tokens(current_tokens)
            mirror_heatmap = mirrored_map(heatmap)

            m0_comp = m0.base.forward_components(
                current_tokens, goal_tokens, heatmap, network
            )
            m1_base_comp = m1.base.forward_components(
                current_tokens, goal_tokens, heatmap, network
            )
            residual = m1.fusion(maps)
            residual_mirror = m1.fusion(mirror_maps)
            m1_fused_logits = m1_base_comp["move_logits"] + residual

            m1_base_mirror_comp = m1.base.forward_components(
                mirror_tokens, goal_tokens, mirror_heatmap, network
            )
            m0_mirror_comp = m0.base.forward_components(
                mirror_tokens, goal_tokens, mirror_heatmap, network
            )
            fused_mirror_logits = (
                m1_base_mirror_comp["move_logits"] + residual_mirror
            )

            # Direction probe: pooled conv features of the fusion trunk.
            conv = m1.fusion.net[:4](maps)
            coarse_1x1 = F.adaptive_avg_pool2d(conv, 1).flatten(1)
            coarse_2x2 = F.adaptive_avg_pool2d(conv, (2, 2)).flatten(1)

            m0_action = int(torch.argmax(m0_comp["move_logits"], -1).item())
            m1_base_action = int(
                torch.argmax(m1_base_comp["move_logits"], -1).item()
            )
            m1_fused_action = int(torch.argmax(m1_fused_logits, -1).item())
            m1_fused_mirror_action = int(
                torch.argmax(fused_mirror_logits, -1).item()
            )
            m0_mirror_action = int(
                torch.argmax(m0_mirror_comp["move_logits"], -1).item()
            )
            m1_base_mirror_action = int(
                torch.argmax(m1_base_mirror_comp["move_logits"], -1).item()
            )

            owner_move = (
                m0_action if owner == "M0" else m1_fused_action
            )
            owner_arrival = (
                m0_comp["arrival_logit"] if owner == "M0"
                else m1_base_comp["arrival_logit"]
            )
            logged_move = int(logged["move_action"])
            if owner_move != logged_move:
                max_move_action_mismatch += 1
            arrival_error = abs(
                float(torch.sigmoid(owner_arrival).item())
                - float(logged["arrival_probability"])
            )
            max_arrival_error = max(max_arrival_error, arrival_error)

            projection = env.goal_projection()
            side, goal_px = goal_side(projection, ENV.image_size)
            dx = float(env.goal[0] - env.x)
            dy = float(env.goal[1] - env.y)
            bearing = math.atan2(dy, dx)
            heading_error = wrap_angle(bearing - env.yaw)

            base_centroid, base_left, base_right = map_direction(baseline)
            coh_centroid, coh_left, coh_right = map_direction(coherent)

            m0_logits = _float_list(m0_comp["move_logits"])
            m1_base_logits = _float_list(m1_base_comp["move_logits"])
            residual_list = _float_list(residual)
            fused_list = _float_list(m1_fused_logits)
            residual_mirror_list = _float_list(residual_mirror)
            fused_mirror_list = _float_list(fused_mirror_logits)

            executed_action = int(logged["action"])
            row = {
                "trajectory": owner,
                "episode_index": int(spec["episode_index"]),
                "seed": int(spec["seed"]),
                "band": str(spec["band"]),
                "step": step,
                "distance_m": distance,
                "goal_visible": int(projection is not None),
                "goal_px": goal_px,
                "goal_side": side,
                "goal_bearing_deg": math.degrees(bearing),
                "heading_error_deg": math.degrees(heading_error),
                "oracle_move_action": oracle_move_action(env),
                "stop_ready": int(distance < env.stop_radius),
                "executed_action": executed_action,
                "executed_move": int(executed_action != STOP),
                "logged_stop": int(logged["stop"]),
                "logged_arrival_probability": float(logged["arrival_probability"]),
                "m0_arrival_probability": float(
                    torch.sigmoid(m0_comp["arrival_logit"]).item()
                ),
                "m1_arrival_probability": float(
                    torch.sigmoid(m1_base_comp["arrival_logit"]).item()
                ),
                "m0_logit_forward": m0_logits[0],
                "m0_logit_left": m0_logits[1],
                "m0_logit_right": m0_logits[2],
                "m0_action": m0_action,
                "m1_base_logit_forward": m1_base_logits[0],
                "m1_base_logit_left": m1_base_logits[1],
                "m1_base_logit_right": m1_base_logits[2],
                "m1_base_action": m1_base_action,
                "m1_res_logit_forward": residual_list[0],
                "m1_res_logit_left": residual_list[1],
                "m1_res_logit_right": residual_list[2],
                "m1_fused_logit_forward": fused_list[0],
                "m1_fused_logit_left": fused_list[1],
                "m1_fused_logit_right": fused_list[2],
                "m1_fused_action": m1_fused_action,
                "m1_res_mirror_logit_forward": residual_mirror_list[0],
                "m1_res_mirror_logit_left": residual_mirror_list[1],
                "m1_res_mirror_logit_right": residual_mirror_list[2],
                "m1_fused_mirror_logit_forward": fused_mirror_list[0],
                "m1_fused_mirror_logit_left": fused_mirror_list[1],
                "m1_fused_mirror_logit_right": fused_mirror_list[2],
                "m1_fused_mirror_action": m1_fused_mirror_action,
                "m0_mirror_logit_forward": float(m0_mirror_comp["move_logits"][0, 0]),
                "m0_mirror_logit_left": float(m0_mirror_comp["move_logits"][0, 1]),
                "m0_mirror_logit_right": float(m0_mirror_comp["move_logits"][0, 2]),
                "m0_mirror_action": m0_mirror_action,
                "m1_base_mirror_logit_forward": float(
                    m1_base_mirror_comp["move_logits"][0, 0]
                ),
                "m1_base_mirror_logit_left": float(
                    m1_base_mirror_comp["move_logits"][0, 1]
                ),
                "m1_base_mirror_logit_right": float(
                    m1_base_mirror_comp["move_logits"][0, 2]
                ),
                "m1_base_mirror_action": m1_base_mirror_action,
                "base_centroid_x": float(base_centroid.item()),
                "base_left_mass": float(base_left.item()),
                "base_right_mass": float(base_right.item()),
                "coherent_centroid_x": float(coh_centroid.item()),
                "coherent_left_mass": float(coh_left.item()),
                "coherent_right_mass": float(coh_right.item()),
            }
            rows.append(row)

            if side in ("left", "right") and projection is not None:
                coarse_feature_sink.append({
                    "episode_index": int(spec["episode_index"]),
                    "trajectory": owner,
                    "side": 1.0 if side == "left" else 0.0,
                    "f1": coarse_1x1.reshape(-1).cpu().numpy().copy(),
                    "f2": coarse_2x2.reshape(-1).cpu().numpy().copy(),
                })

        obs, _, done = env.step(
            executed_action, render_observation=(executed_action != STOP)
        )
        if done:
            break

    if len(rows) != len(logged_steps):
        raise RuntimeError(
            f"{owner} episode {spec['episode_index']}: replayed {len(rows)} "
            f"steps, expected {len(logged_steps)}"
        )
    return rows, {
        "episode_index": int(spec["episode_index"]),
        "max_distance_error_m": max_distance_error,
        "move_action_mismatch_steps": max_move_action_mismatch,
        "max_arrival_probability_error": max_arrival_error,
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
        f"episodes={len(specs)} threshold=0.7 device={device}",
        flush=True,
    )

    m0_steps = steps_by_episode(
        read_steps(args.m0_run / "steps/threshold_0.7.csv")
    )
    m1_steps = steps_by_episode(
        read_steps(args.m1_run / "steps/M1@0.7.csv")
    )

    all_rows = []
    checks = []
    coarse_sink = []
    started = time.time()
    for position, spec in enumerate(specs):
        episode = int(spec["episode_index"])
        for owner, grouped in (("M0", m0_steps), ("M1", m1_steps)):
            if episode not in grouped:
                raise RuntimeError(f"Missing {owner} log for episode {episode}")
            rows, check = replay_episode(
                owner, spec, grouped[episode], m0, m1, encoder, device, 0.7,
                coarse_sink,
            )
            check["trajectory"] = owner
            checks.append(check)
            all_rows.extend(rows)
        if (position + 1) % 10 == 0:
            elapsed = time.time() - started
            print(
                f"  replayed {position + 1}/{len(specs)} episodes "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    bad_distance = [c for c in checks if c["max_distance_error_m"] > 1e-4]
    bad_move = [c for c in checks if c["move_action_mismatch_steps"] > 0]
    bad_arrival = [
        c for c in checks if c["max_arrival_probability_error"] > 1e-5
    ]
    write_csv(args.output_dir / "replay_steps.csv", all_rows)
    write_json(args.output_dir / "replay_check.json", {
        "episodes_per_trajectory": len(specs),
        "trajectory_passes": len(checks),
        "distance_mismatch_passes": len(bad_distance),
        "move_action_mismatch_passes": len(bad_move),
        "arrival_probability_mismatch_passes": len(bad_arrival),
        "max_distance_error_m": max(
            (c["max_distance_error_m"] for c in checks), default=0.0
        ),
        "max_arrival_probability_error": max(
            (c["max_arrival_probability_error"] for c in checks), default=0.0
        ),
        "examples": (bad_distance + bad_move + bad_arrival)[:20],
        "note": (
            "A pass means the exact saved state was reconstructed and the "
            "owner model reproduced the logged move action and arrival "
            "probability. Any mismatch invalidates the off-policy table."
        ),
    })
    if bad_distance or bad_move or bad_arrival:
        raise SystemExit(
            "Replay verification failed: "
            f"{len(bad_distance)} distance, {len(bad_move)} move-action, "
            f"{len(bad_arrival)} arrival mismatches"
        )

    # Direction probe features for the 2x2 vs 1x1 pooling comparison.
    if coarse_sink:
        f1 = np.stack([item["f1"] for item in coarse_sink]).astype(np.float32)
        f2 = np.stack([item["f2"] for item in coarse_sink]).astype(np.float32)
        side = np.asarray([item["side"] for item in coarse_sink], dtype=np.float32)
        episode = np.asarray(
            [item["episode_index"] for item in coarse_sink], dtype=np.int64
        )
        np.savez_compressed(
            args.output_dir / "fusion_pool_features.npz",
            f1=f1, f2=f2, side=side, episode_index=episode,
        )

    write_json(args.output_dir / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "V8.3 Stage-2e Movement replay diagnostics (eval only)",
        "command": sys.argv,
        "m0_checkpoint": str(M0_CKPT),
        "m0_checkpoint_sha256": file_sha256(M0_CKPT),
        "m1_checkpoint": str(M1_CKPT),
        "m1_checkpoint_sha256": file_sha256(M1_CKPT),
        "m0_steps": str(args.m0_run / "steps/threshold_0.7.csv"),
        "m1_steps": str(args.m1_run / "steps/M1@0.7.csv"),
        "val_specs": str(args.val_specs),
        "val_specs_sha256": file_sha256(args.val_specs),
        "episodes": int(len(specs)),
        "threshold": 0.7,
        "replay_passes": int(len(checks)),
        "training": "none; frozen Stage-2a checkpoints only",
        "final_test": "sealed; not read",
        "device": device_summary(device),
    })
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
