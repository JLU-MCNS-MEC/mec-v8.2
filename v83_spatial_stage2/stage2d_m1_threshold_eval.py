"""M1 static visual fusion: Stop-threshold migration evaluation.

Runs the existing Stage-2a Transformer-M M1 checkpoint (no training, no
architecture change) on the Dev Val 180 episodes at Stop thresholds 0.5 and
0.7. M1@0.5 is first verified against the Stage-2a M1 per-episode reference
(SR 27.78%, OSR 30.56%, nDTW 0.3648). If the outcome replay fails, the script
stops.

Every episode is a real closed-loop run with per-step recording, so far/near
Stop diagnostics and paired per-episode statistics are computed from all 180
episodes. Final Test 300 is never read.
"""

from __future__ import annotations

import argparse
import csv
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

import eval_fixed_mec_epoch_compare as evaluation
from config import ENCODER, ENV
from env import STOP, ToyUAVEnv
from gbs_model_v3 import build_gbs_model
from gbs_network import build_random_gbs_layout, global_network_state
from navigation_reward import ndtw_score
from token_data import encode_multiscale_goal_tokens, goal_conditioned_spatial_heatmap
from utils import image_to_tensor
from visual_grounding_v83 import static_grounding
from v83_spatial_training.spatial_mec import SpatialAwareMEC
from v83_spatial_stage2.stage2d_common import (
    BANDS,
    STAGE2A_EVAL,
    STAGE2_ROOT,
    VAL_SPECS,
    build_oracle_cache,
    device_summary,
    file_sha256,
    hazard_from_steps,
    load_dev_specs,
    load_encoder,
    paired_by_band,
    risk_rows,
    stop_event_summary,
    summarize_by_band,
    write_csv,
    write_json,
)


M1_CKPT = (
    STAGE2_ROOT
    / "stage2a_tm_20261018T044255Z/Transformer-M/M1/best.pt"
)
THRESHOLDS = (0.5, 0.7)


def _rule_name(threshold):
    return f"M1@{threshold:.1f}"


def run_episode(model, spec, encoder, device, threshold, oracle):
    rule = _rule_name(threshold)
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
    trajectory = [env.position.copy()]
    min_distance = float(env.distance_to_goal)
    osr = bool(env.distance_to_goal < env.stop_radius)
    success = False
    done = False
    ended_by_stop = False
    steps = []
    stop_events = []

    for step in range(evaluation.MAX_STEPS):
        current_tokens = encoder(image_to_tensor(obs, device))
        heatmap = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
        network_np = global_network_state(
            nodes, env.position,
            num_tokens=ENCODER.num_tokens, token_dim=ENCODER.token_dim,
        ).astype(np.float32)
        network = torch.from_numpy(network_np).unsqueeze(0).to(device)
        with torch.inference_mode():
            # M1 static visual fusion: the same maps as
            # eval_closedloop_spatial.spatial_maps_for_mode("M1", ...).
            baseline, coherent = static_grounding(current_tokens, goal_tokens)
            zero = torch.zeros_like(coherent)
            maps = torch.stack([baseline, coherent, zero, zero], dim=1)
            valid = torch.zeros(
                maps.shape[0], 1, 1, 1,
                device=maps.device, dtype=maps.dtype,
            )
            components = model.forward_components(
                current_tokens, goal_tokens, heatmap, network,
                spatial_maps=maps, spatial_valid=valid,
            )
            probability = torch.sigmoid(
                components["arrival_logit"]
            ).reshape(-1).item()
            move_action = int(
                torch.argmax(components["move_logits"], dim=-1).item()
            )
        action = STOP if probability >= float(threshold) else move_action
        distance = float(env.distance_to_goal)
        projection = env.goal_projection()
        goal_visible = int(projection is not None)
        steps.append({
            "rule": rule,
            "episode_index": int(spec["episode_index"]),
            "seed": int(spec["seed"]),
            "band": str(spec["band"]),
            "step": int(step),
            "distance_m": distance,
            "goal_visible": goal_visible,
            "arrival_probability": float(probability),
            "base_arrival_probability": float(probability),
            "stop": int(action == STOP),
            "action": int(action),
            "move_action": int(move_action),
        })
        if action == STOP:
            stop_events.append({
                "rule": rule,
                "episode_index": int(spec["episode_index"]),
                "seed": int(spec["seed"]),
                "band": str(spec["band"]),
                "step": int(step),
                "distance_m": distance,
                "arrival_probability": float(probability),
                "goal_visible": goal_visible,
            })
        obs, step_success, done = env.step(
            action, render_observation=(action != STOP)
        )
        success = bool(step_success)
        ended_by_stop = bool(action == STOP)
        trajectory.append(env.position.copy())
        min_distance = min(min_distance, float(env.distance_to_goal))
        osr = bool(osr or env.distance_to_goal < env.stop_radius)
        if done:
            break

    trajectory = np.asarray(trajectory, dtype=np.float32)
    path_length = (
        float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())
        if len(trajectory) > 1 else 0.0
    )
    shortest = float(oracle["shortest_distance_m"])
    spl = shortest / max(shortest, path_length, 1e-8) if success else 0.0
    ndtw = ndtw_score(
        trajectory=trajectory[:, :2],
        reference=oracle["reference"],
        distance_threshold=env.stop_radius,
    )
    premature_stop = bool(ended_by_stop and not success)
    timeout = bool((not ended_by_stop) and (not success) and done)
    episode = {
        "rule": rule,
        "episode_index": int(spec["episode_index"]),
        "seed": int(spec["seed"]),
        "band": str(spec["band"]),
        "success": int(success),
        "osr": int(osr),
        "premature_stop": int(premature_stop),
        "max_steps_termination": int(timeout),
        "steps": int(len(trajectory) - 1),
        "spl": float(spl),
        "ndtw": float(ndtw),
        "min_distance_m": float(min_distance),
        "final_distance_m": float(env.distance_to_goal),
        "reached_radius_not_success": int(osr and not success),
        "stop_distance_m": (
            float(stop_events[0]["distance_m"]) if stop_events else float("nan")
        ),
        "stop_arrival_probability": (
            float(stop_events[0]["arrival_probability"])
            if stop_events else float("nan")
        ),
        "stop_goal_visible": (
            int(stop_events[0]["goal_visible"]) if stop_events else -1
        ),
    }
    return episode, steps, stop_events


def evaluate(model, specs, oracle_cache, encoder, device, threshold):
    rule = _rule_name(threshold)
    episodes, all_steps, all_stops = [], [], []
    for position, spec in enumerate(specs):
        episode, steps, stops = run_episode(
            model, spec, encoder, device, threshold,
            oracle_cache[int(spec["episode_index"])],
        )
        episodes.append(episode)
        all_steps.extend(steps)
        all_stops.extend(stops)
        if (position + 1) % 30 == 0:
            print(f"  {rule}: {position + 1}/{len(specs)} episodes", flush=True)
    return episodes, all_steps, all_stops


def verify_m1_replay(episodes, reference_csv=STAGE2A_EVAL / "per_episode.csv"):
    with Path(reference_csv).open(encoding="utf-8") as handle:
        reference = {
            int(row["episode_index"]): row
            for row in csv.DictReader(handle)
            if row["mode"] == "M1"
        }
    outcome_mismatches = []
    numeric_mismatches = []
    for episode in episodes:
        index = int(episode["episode_index"])
        if index not in reference:
            outcome_mismatches.append(
                {"episode_index": index, "error": "missing_reference"}
            )
            continue
        expected = reference[index]
        for key in ("success", "osr", "premature_stop", "max_steps_termination"):
            if int(episode[key]) != int(expected[key]):
                outcome_mismatches.append({
                    "episode_index": index, "field": key,
                    "fresh": int(episode[key]), "reference": int(expected[key]),
                })
        if int(episode["steps"]) != int(expected["steps"]):
            numeric_mismatches.append({
                "episode_index": index, "field": "steps",
                "fresh": int(episode["steps"]),
                "reference": int(expected["steps"]),
            })
        for key in ("spl", "ndtw", "min_distance_m"):
            if abs(float(episode[key]) - float(expected[key])) > 1e-4:
                numeric_mismatches.append({
                    "episode_index": index, "field": key,
                    "fresh": float(episode[key]),
                    "reference": float(expected[key]),
                })
    return {
        "reference": str(reference_csv),
        "episodes": int(len(episodes)),
        "outcome_mismatch_count": int(len(outcome_mismatches)),
        "outcome_mismatches": outcome_mismatches[:50],
        "numeric_mismatch_count": int(len(numeric_mismatches)),
        "numeric_mismatches": numeric_mismatches[:50],
    }


def summarize_with_counts(rows):
    summary = summarize_by_band(rows)
    for band in ("all",) + BANDS:
        subset = (
            rows if band == "all"
            else [row for row in rows if str(row["band"]) == band]
        )
        summary[band]["success_count"] = int(
            sum(int(row["success"]) for row in subset)
        )
        summary[band]["n"] = int(len(subset))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=180)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-replay-gate", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.max_episodes = min(args.max_episodes, 6)
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "steps").mkdir()
    (args.output_dir / "stops").mkdir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = load_dev_specs(VAL_SPECS, args.max_episodes)
    encoder = load_encoder(device)
    oracle_cache = build_oracle_cache(specs)
    model = SpatialAwareMEC(build_gbs_model("Transformer-M"), "M1").to(device)
    state = torch.load(M1_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(
        f"M1 checkpoint {M1_CKPT}; {len(specs)} episodes; "
        f"thresholds={THRESHOLDS}",
        flush=True,
    )

    episodes_by = {}
    steps_by = {}
    stops_by = {}
    for threshold in THRESHOLDS:
        rule = _rule_name(threshold)
        print(f"Closed-loop {rule}", flush=True)
        start = time.time()
        episodes, steps, stops = evaluate(
            model, specs, oracle_cache, encoder, device, threshold
        )
        episodes_by[rule] = episodes
        steps_by[rule] = steps
        stops_by[rule] = stops
        write_csv(args.output_dir / f"{rule}_per_episode.csv", episodes)
        write_csv(args.output_dir / "steps" / f"{rule}.csv", steps)
        write_csv(args.output_dir / "stops" / f"{rule}.csv", stops)
        print(
            f"  {rule}: SR={np.mean([r['success'] for r in episodes]):.4f} "
            f"OSR={np.mean([r['osr'] for r in episodes]):.4f} "
            f"premature={np.mean([r['premature_stop'] for r in episodes]):.4f} "
            f"({time.time() - start:.1f}s)",
            flush=True,
        )

    replay = verify_m1_replay(episodes_by["M1@0.5"])
    write_json(args.output_dir / "replay_m1_0.5.json", replay)
    if replay["outcome_mismatch_count"] != 0 and not args.skip_replay_gate:
        raise SystemExit(
            "M1@0.5 did not reproduce Stage-2a M1 outcomes: "
            f"{replay['outcome_mismatch_count']} fields; see replay_m1_0.5.json"
        )

    metrics = {
        "summaries": {
            rule: summarize_with_counts(episodes_by[rule])
            for rule in episodes_by
        },
        "hazard": {
            rule: {
                band: hazard_from_steps(
                    steps_by[rule] if band == "all" else [
                        row for row in steps_by[rule]
                        if str(row["band"]) == band
                    ],
                    threshold=float(rule.split("@")[1]),
                )
                for band in ("all",) + BANDS
            }
            for rule in steps_by
        },
        "stop_events": {
            rule: {
                band: stop_event_summary([
                    row for row in stops_by[rule]
                    if band == "all" or str(row["band"]) == band
                ])
                for band in ("all",) + BANDS
            }
            for rule in stops_by
        },
        "paired_m1_0.7_vs_m1_0.5": paired_by_band(
            {"M1@0.5": episodes_by["M1@0.5"], "M1@0.7": episodes_by["M1@0.7"]},
            base="M1@0.5", other="M1@0.7",
        ),
    }
    risk = risk_rows(
        steps_by,
        threshold_map={rule: float(rule.split("@")[1]) for rule in steps_by},
    )
    write_csv(args.output_dir / "risk.csv", risk)

    # Compare with the existing M0@0.7 per-episode rows; do not re-run M0.
    m0_rows = [
        row for row in csv.DictReader(
            open(
                STAGE2_ROOT
                / "stage2d/r4_20261019T002626Z/threshold_0.7_per_episode.csv",
                encoding="utf-8",
            )
        )
    ]
    for row in m0_rows:
        row["rule"] = "M0@0.7"
    metrics["m0_0.7_reference"] = summarize_with_counts(m0_rows)
    metrics["paired_m1_0.7_vs_m0_0.7"] = paired_by_band(
        {"M0@0.7": m0_rows, "M1@0.7": episodes_by["M1@0.7"]},
        base="M0@0.7", other="M1@0.7",
    )
    # Paired M1@0.7 vs M0@0.7 by band is in the dict above.

    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "V8.3 Stage-2d M1 Stop-threshold migration (eval only)",
        "command": sys.argv,
        "m1_checkpoint": str(M1_CKPT),
        "m1_checkpoint_sha256": file_sha256(M1_CKPT),
        "val_specs": str(VAL_SPECS),
        "val_specs_sha256": file_sha256(VAL_SPECS),
        "episodes": int(len(specs)),
        "thresholds": list(THRESHOLDS),
        "training": "none; existing Stage-2a M1 weights only",
        "final_test": "sealed; not read",
        "device": device_summary(device),
    })
    print(json.dumps({
        "summaries": {
            rule: metrics["summaries"][rule]["all"] for rule in episodes_by
        },
        "replay_m1_0.5": replay,
    }, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
