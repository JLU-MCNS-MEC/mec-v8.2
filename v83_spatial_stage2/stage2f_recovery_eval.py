"""Stage-2f: closed-loop smoke for the Goal-Reacquisition recovery overlay.

Runs the frozen M0@0.7 controller, optionally with a lightweight recovery
policy, on a small Dev-Val subset. The recovery policy uses only observable
features from `stage2f_features.py`; true visibility, true distance, heading
error and the oracle action are recorded only for evaluation.

No training, no checkpoint modification, Final Test 300 not read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER, ENV
from env import FORWARD, LEFT, RIGHT, STOP, ToyUAVEnv, wrap_angle
from gbs_model_v3 import build_gbs_model
from gbs_network import build_random_gbs_layout, global_network_state
from navigation_reward import ndtw_score
from token_data import encode_multiscale_goal_tokens, goal_conditioned_spatial_heatmap
from utils import image_to_tensor
from visual_grounding_v83 import static_grounding
from v83_spatial_stage2.stage2d_common import (
    STAGE2_ROOT,
    VAL_SPECS,
    build_oracle_cache,
    device_summary,
    file_sha256,
    load_dev_specs,
    load_encoder,
    write_csv,
    write_json,
)
from v83_spatial_stage2.stage2f_collect import (
    M0_CKPT,
    load_model,
    oracle_move_action,
    projection_vector,
)
from v83_spatial_stage2.stage2f_features import extract_observable_features
from v83_spatial_stage2.stage2f_policy import RecoveryConfig, RecoveryPolicy
from v83_spatial_training.eval_closedloop_spatial import paired_comparison


BANDS = ("short", "medium", "long")
STOP_THRESHOLD = 0.7
SAVED_M0_EPISODES = (
    STAGE2_ROOT / "stage2d/r4_20261019T002626Z/threshold_0.7_per_episode.csv"
)


def select_specs(specs, per_band, method="even"):
    grouped = defaultdict(list)
    for spec in specs:
        grouped[str(spec["band"])].append(spec)
    selected = []
    for band in BANDS:
        items = sorted(
            grouped[band], key=lambda item: int(item["episode_index"])
        )
        if method == "even" and 0 < int(per_band) < len(items):
            indices = np.linspace(
                0, len(items) - 1, int(per_band)
            ).round().astype(int)
            selected.extend(items[index] for index in indices)
        else:
            selected.extend(items[: int(per_band)])
    return selected


def load_saved_m0(path=SAVED_M0_EPISODES):
    with Path(path).open(encoding="utf-8") as handle:
        return {
            int(row["episode_index"]): row for row in csv.DictReader(handle)
        }


def step_heading_after(action, heading_deg):
    if action == LEFT:
        return (heading_deg - 15.0 + 180.0) % 360.0 - 180.0
    if action == RIGHT:
        return (heading_deg + 15.0 + 180.0) % 360.0 - 180.0
    return heading_deg


def movement_error_label(action, oracle_action):
    if action == oracle_action:
        return "correct"
    if oracle_action in (LEFT, RIGHT) and action == FORWARD:
        return "missed_turn"
    if oracle_action in (LEFT, RIGHT) and action in (LEFT, RIGHT):
        return "wrong_turn"
    if oracle_action == FORWARD and action in (LEFT, RIGHT):
        return "spurious_turn"
    return "other"


def run_episode(spec, model, encoder, oracle, device, policy, threshold):
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
    if policy is not None:
        policy.reset()
    previous_coherent = None
    previous_action = None

    trajectory = [env.position.copy()]
    min_distance = float(env.distance_to_goal)
    osr = bool(env.distance_to_goal < env.stop_radius)
    success = False
    done = False
    ended_by_stop = False
    rows = []
    trigger_steps = []

    for step in range(int(ENV.max_steps)):
        current_tokens = encoder(image_to_tensor(obs, device))
        heatmap = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
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
            features = extract_observable_features(
                current_tokens, goal_tokens, heatmap, baseline, coherent,
                previous_coherent, previous_action,
                components["move_logits"], components["arrival_logit"],
            )
            arrival_probability = float(
                torch.sigmoid(components["arrival_logit"]).item()
            )
            base_action = int(
                torch.argmax(components["move_logits"], -1).item()
            )
            projection = projection_vector(env)
            dx = float(env.goal[0] - env.x)
            dy = float(env.goal[1] - env.y)
            heading_error = wrap_angle(math.atan2(dy, dx) - env.yaw)

        if arrival_probability >= threshold:
            action = STOP
            recovering = 0
            trigger_now = 0
        elif policy is not None:
            before = policy.trigger_count
            action, recovering_flag = policy.act(features, base_action, step)
            recovering = int(recovering_flag)
            trigger_now = int(policy.trigger_count > before)
        else:
            action = base_action
            recovering = 0
            trigger_now = 0
        if trigger_now:
            trigger_steps.append(step)

        distance = float(env.distance_to_goal)
        rows.append({
            "policy": "baseline" if policy is None else policy_name(policy),
            "episode_index": int(spec["episode_index"]),
            "seed": int(spec["seed"]),
            "band": str(spec["band"]),
            "step": int(step),
            "distance_m": distance,
            "goal_visible": int(projection is not None),
            "goal_px": projection["goal_px"] if projection else float("nan"),
            "goal_py": projection["goal_py"] if projection else float("nan"),
            "heading_error_deg": math.degrees(heading_error),
            "oracle_move_action": oracle_move_action(env),
            "base_action": base_action,
            "action": int(action),
            "recovering": int(recovering),
            "trigger_now": int(trigger_now),
            "arrival_probability": arrival_probability,
            "m0_move_action": int(features["m0_move_action"]),
            "heatmap_max_top1": float(features["heatmap_max_top1"]),
            "heatmap_max_entropy": float(features["heatmap_max_entropy"]),
            "baseline_centroid_x": float(features["baseline_centroid_x"]),
            "coherent_centroid_x": float(features["coherent_centroid_x"]),
            "temporal_valid": float(features["temporal_valid"]),
            "temporal_consistency": float(features["temporal_consistency"]),
            "movement_label": movement_error_label(
                int(action), oracle_move_action(env)
            ) if action != STOP else "stop",
        })

        obs, step_success, done = env.step(
            action, render_observation=(action != STOP)
        )
        success = bool(step_success)
        ended_by_stop = bool(action == STOP)
        trajectory.append(env.position.copy())
        min_distance = min(min_distance, float(env.distance_to_goal))
        osr = bool(osr or env.distance_to_goal < env.stop_radius)
        previous_coherent = coherent[0].detach()
        previous_action = int(action)
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
        "policy": "baseline" if policy is None else policy_name(policy),
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
        "trigger_count": int(policy.trigger_count) if policy else 0,
        "recovery_steps": int(policy.recovery_step_count) if policy else 0,
        "reacquired_count": int(policy.reacquired_count) if policy else 0,
    }
    events = []
    if policy is not None:
        for event in policy.events:
            trigger_index = event.trigger_step
            if trigger_index >= len(rows):
                continue
            row = rows[trigger_index]
            future = rows[trigger_index + 1:]
            revisible_5 = any(
                int(item["goal_visible"]) for item in future[:5]
            )
            revisible_10 = any(
                int(item["goal_visible"]) for item in future[:10]
            )
            entered_20 = any(
                float(item["distance_m"]) < 20.0 for item in future[:40]
            )
            entered_40 = any(
                float(item["distance_m"]) < 20.0 for item in future[:40]
            )
            recovery_window = rows[
                trigger_index: trigger_index + len(event.actions) + 1
            ]
            recovery_rows = [
                item for item in recovery_window if int(item["recovering"])
            ]
            wrong = Counter(
                item["movement_label"] for item in recovery_rows
                if item["movement_label"] not in ("correct", "stop")
            )
            events.append({
                "policy": policy_name(policy),
                "episode_index": int(spec["episode_index"]),
                "band": str(spec["band"]),
                "trigger_step": int(trigger_index),
                "trigger_reason": event.trigger_reason,
                "trigger_goal_visible": int(row["goal_visible"]),
                "trigger_distance_m": float(row["distance_m"]),
                "trigger_heading_error_deg": float(row["heading_error_deg"]),
                "trigger_heatmap_top1": float(row["heatmap_max_top1"]),
                "trigger_heatmap_entropy": float(row["heatmap_max_entropy"]),
                "last_high_side": event.last_high_side,
                "steps_since_high_conf": event.steps_since_high_conf,
                "revisible_5": int(revisible_5),
                "revisible_10": int(revisible_10),
                "entered_20_within_40": int(entered_20),
                "recovery_actions": len(event.actions),
                "recovery_wrong_turn": int(sum(wrong.values())),
                "recovery_missed_turn": int(wrong.get("missed_turn", 0)),
                "recovery_wrong_direction": int(wrong.get("wrong_turn", 0)),
                "recovery_spurious_turn": int(wrong.get("spurious_turn", 0)),
                "reacquired_step": (
                    int(event.reacquired_step)
                    if event.reacquired_step is not None else -1
                ),
                "exit_reason": event.exit_reason,
            })
    return episode, rows, events


_POLICY_NAMES = {}


def policy_name(policy):
    return _POLICY_NAMES.get(id(policy), "recovery")


def register_policy_name(policy, name):
    _POLICY_NAMES[id(policy)] = name


def summarize(episodes):
    if not episodes:
        return {"n": 0}
    return {
        "n": len(episodes),
        "sr": float(np.mean([row["success"] for row in episodes])),
        "osr": float(np.mean([row["osr"] for row in episodes])),
        "spl": float(np.mean([row["spl"] for row in episodes])),
        "ndtw": float(np.mean([row["ndtw"] for row in episodes])),
        "premature_stop": float(np.mean([row["premature_stop"] for row in episodes])),
        "timeout": float(np.mean([row["max_steps_termination"] for row in episodes])),
        "steps": float(np.mean([row["steps"] for row in episodes])),
        "min_distance_m": float(np.mean([row["min_distance_m"] for row in episodes])),
        "reached_radius_not_success": float(
            np.mean([row["reached_radius_not_success"] for row in episodes])
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    parser.add_argument("--episodes-per-band", type=int, default=12)
    parser.add_argument(
        "--selection", default="even", choices=["even", "first"]
    )
    parser.add_argument(
        "--policies", nargs="+",
        default=["baseline", "side_history", "scan_only"],
        choices=["baseline", "side_history", "scan_only"],
    )
    parser.add_argument("--threshold", type=float, default=STOP_THRESHOLD)
    parser.add_argument("--high-top1", type=float, default=0.30)
    parser.add_argument("--high-entropy", type=float, default=0.75)
    parser.add_argument("--low-top1", type=float, default=0.15)
    parser.add_argument("--low-entropy", type=float, default=0.85)
    parser.add_argument("--trigger-k", type=int, default=2)
    parser.add_argument("--consistency", type=float, default=0.55)
    parser.add_argument("--max-recovery-steps", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "steps").mkdir()
    (args.output_dir / "events").mkdir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = select_specs(
        load_dev_specs(args.val_specs, 180), args.episodes_per_band,
        method=args.selection,
    )
    encoder = load_encoder(device)
    oracle_cache = build_oracle_cache(specs)
    model = load_model("M0", M0_CKPT, device)
    saved_m0 = load_saved_m0()
    print(
        f"episodes={len(specs)} per_band={args.episodes_per_band} "
        f"policies={args.policies} device={device}",
        flush=True,
    )

    episodes_by_policy = {}
    steps_by_policy = {}
    events_by_policy = {}
    for name in args.policies:
        if name == "baseline":
            policy = None
        else:
            config = RecoveryConfig(
                high_top1=args.high_top1,
                high_entropy=args.high_entropy,
                low_top1=args.low_top1,
                low_entropy=args.low_entropy,
                trigger_k=args.trigger_k,
                consistency=args.consistency,
                use_side=(name == "side_history"),
                max_recovery_steps=args.max_recovery_steps,
            )
            policy = RecoveryPolicy(config)
            register_policy_name(policy, name)
        episodes, steps, events = [], [], []
        for position, spec in enumerate(specs):
            episode, step_rows, trigger_events = run_episode(
                spec, model, encoder,
                oracle_cache[int(spec["episode_index"])],
                device, policy, args.threshold,
            )
            episodes.append(episode)
            steps.extend(step_rows)
            events.extend(trigger_events)
            if (position + 1) % 12 == 0:
                print(f"  {name}: {position + 1}/{len(specs)}", flush=True)
        episodes_by_policy[name] = episodes
        steps_by_policy[name] = steps
        events_by_policy[name] = events
        write_csv(args.output_dir / f"{name}_per_episode.csv", episodes)
        write_csv(args.output_dir / "steps" / f"{name}.csv", steps)
        write_csv(args.output_dir / "events" / f"{name}.csv", events)

    # Baseline sanity gate against the saved M0@0.7 run.
    baseline = episodes_by_policy.get("baseline")
    baseline_check = {"episodes": 0, "outcome_mismatches": []}
    if baseline is not None:
        for row in baseline:
            reference = saved_m0.get(int(row["episode_index"]))
            if reference is None:
                baseline_check["outcome_mismatches"].append({
                    "episode_index": row["episode_index"],
                    "error": "missing_reference",
                })
                continue
            baseline_check["episodes"] += 1
            for key in ("success", "osr", "premature_stop",
                        "max_steps_termination", "steps"):
                if int(row[key]) != int(reference[key]):
                    baseline_check["outcome_mismatches"].append({
                        "episode_index": int(row["episode_index"]),
                        "field": key,
                        "fresh": int(row[key]),
                        "reference": int(reference[key]),
                    })
        if baseline_check["outcome_mismatches"]:
            write_json(args.output_dir / "baseline_check.json", baseline_check)
            raise SystemExit("Baseline smoke did not reproduce the saved M0@0.7 run")

    summaries = {}
    for name, episodes in episodes_by_policy.items():
        summaries[name] = {"all": summarize(episodes)}
        for band in BANDS:
            summaries[name][band] = summarize([
                row for row in episodes if str(row["band"]) == band
            ])

    trigger_metrics = {}
    for name, events in events_by_policy.items():
        if name == "baseline":
            continue
        invisible = [e for e in events if e["trigger_goal_visible"] == 0]
        visible = [e for e in events if e["trigger_goal_visible"] == 1]
        far = [e for e in events if e["trigger_distance_m"] >= 20.0]
        trigger_metrics[name] = {
            "triggers": len(events),
            "episodes_with_trigger": len({
                e["episode_index"] for e in events
            }),
            "trigger_true_invisible": len(invisible),
            "trigger_true_visible": len(visible),
            "trigger_invisible_rate": (
                len(invisible) / max(len(events), 1)
            ),
            "invisible_revisible_5": (
                float(np.mean([e["revisible_5"] for e in invisible]))
                if invisible else float("nan")
            ),
            "invisible_revisible_10": (
                float(np.mean([e["revisible_10"] for e in invisible]))
                if invisible else float("nan")
            ),
            "far_entered20_within40": (
                float(np.mean([e["entered_20_within_40"] for e in far]))
                if far else float("nan")
            ),
            "far_triggers": len(far),
            "recovery_actions": int(sum(e["recovery_actions"] for e in events)),
            "recovery_wrong_turn": int(sum(e["recovery_wrong_turn"] for e in events)),
            "recovery_missed_turn": int(sum(e["recovery_missed_turn"] for e in events)),
            "recovery_wrong_direction": int(sum(e["recovery_wrong_direction"] for e in events)),
            "recovery_spurious_turn": int(sum(e["recovery_spurious_turn"] for e in events)),
            "recovery_error_rate": (
                float(sum(e["recovery_wrong_turn"] for e in events)
                      / max(sum(e["recovery_actions"] for e in events), 1))
            ),
        }
        for band in BANDS:
            band_events = [e for e in events if str(e["band"]) == band]
            band_invisible = [e for e in band_events if e["trigger_goal_visible"] == 0]
            band_far = [e for e in band_events if e["trigger_distance_m"] >= 20.0]
            trigger_metrics[name][band] = {
                "triggers": len(band_events),
                "trigger_invisible_rate": (
                    len(band_invisible) / max(len(band_events), 1)
                ),
                "invisible_revisible_10": (
                    float(np.mean([e["revisible_10"] for e in band_invisible]))
                    if band_invisible else float("nan")
                ),
                "far_entered20_within40": (
                    float(np.mean([e["entered_20_within_40"] for e in band_far]))
                    if band_far else float("nan")
                ),
                "far_triggers": len(band_far),
                "recovery_error_rate": (
                    float(sum(e["recovery_wrong_turn"] for e in band_events)
                          / max(sum(e["recovery_actions"] for e in band_events), 1))
                ),
            }

    paired = {}
    if "baseline" in episodes_by_policy:
        for name, episodes in episodes_by_policy.items():
            if name == "baseline":
                continue
            paired[name] = {}
            for band in ("all",) + BANDS:
                base_rows = {
                    int(row["episode_index"]): row
                    for row in episodes_by_policy["baseline"]
                    if band == "all" or str(row["band"]) == band
                }
                other_rows = {
                    int(row["episode_index"]): row
                    for row in episodes
                    if band == "all" or str(row["band"]) == band
                }
                paired[name][band] = paired_comparison(
                    {"baseline": list(base_rows.values()),
                     name: list(other_rows.values())},
                    base="baseline", other=name,
                )

    write_json(args.output_dir / "metrics.json", {
        "summaries": summaries,
        "trigger_metrics": trigger_metrics,
        "paired_vs_baseline": paired,
        "baseline_check": baseline_check,
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    })
    write_json(args.output_dir / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "V8.3 Stage-2f Goal Reacquisition closed-loop smoke (eval only)",
        "command": sys.argv,
        "m0_checkpoint": str(M0_CKPT),
        "m0_checkpoint_sha256": file_sha256(M0_CKPT),
        "val_specs": str(args.val_specs),
        "episodes": int(len(specs)),
        "episodes_per_band": int(args.episodes_per_band),
        "policies": list(args.policies),
        "training": "none; frozen M0 checkpoint only",
        "final_test": "sealed; not read",
        "device": device_summary(device),
    })

    print(json.dumps({
        "summaries": summaries,
        "trigger_metrics": trigger_metrics,
        "baseline_check": baseline_check,
    }, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
