"""Stage-2g.2 36-episode P1-2 near-target recovery closed-loop smoke.

R0 is the saved frozen M0@0.7 baseline on the same 36 Dev-Val episodes
(12/band, even selection). R1 is frozen M0@0.7 plus the verified P1-2
near-target-invisible probe and a bounded forward-preserving recovery action.

Runtime policy inputs are observable only: P1-2 probability, M0 base action,
observable high-confidence target side/recency. True distance/visibility/
heading/oracle action are recorded only for offline scoring.

No MEC training, no checkpoint modification, Final Test 300 sealed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER, ENV  # noqa: E402
from env import FORWARD, LEFT, RIGHT, STOP, ToyUAVEnv, wrap_angle  # noqa: E402
from gbs_network import (  # noqa: E402
    build_random_gbs_layout,
    global_network_state,
)
from navigation_reward import ndtw_score  # noqa: E402
from token_data import (  # noqa: E402
    encode_multiscale_goal_tokens,
    goal_conditioned_spatial_heatmap,
)
from utils import image_to_tensor  # noqa: E402
from visual_grounding_v83 import static_grounding  # noqa: E402
from v83_spatial_stage2.stage2d_common import (  # noqa: E402
    VAL_SPECS,
    build_oracle_cache,
    load_dev_specs,
    write_csv,
    write_json,
)
from v83_spatial_stage2.stage2f_collect import (  # noqa: E402
    M0_CKPT,
    load_model,
    oracle_move_action,
    projection_vector,
)
from v83_spatial_stage2.stage2f_recovery_eval import (  # noqa: E402
    STOP_THRESHOLD,
    movement_error_label,
    select_specs,
    summarize,
)
from v83_spatial_stage2.stage2g_features import (  # noqa: E402
    compute_feature_vector,
    feature_names,
)
from v83_spatial_stage2.stage2g_probe import (  # noqa: E402
    auc_rank,
    average_precision,
    brier_score,
    build_model,
    expected_calibration_error,
)
from v83_spatial_stage2.stage2g1_common import (  # noqa: E402
    COMPACT_FEATURE_NAMES,
    compact_indices,
)
from v83_spatial_training.eval_closedloop_spatial import (  # noqa: E402
    load_encoder,
)
from v83_spatial_stage2.stage2g2_policy import (  # noqa: E402
    ProbeRecoveryConfig,
    ProbeRecoveryPolicy,
    build_probe_input,
    probe_predict,
)

BANDS = ("short", "medium", "long")


def load_probe(path, device):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False
    )
    names = [str(x) for x in checkpoint["feature_names"]]
    canonical = feature_names()
    if names != canonical:
        raise RuntimeError("Probe feature names do not match current code")
    compact_names = [str(x) for x in checkpoint["compact_feature_names"]]
    if compact_names != list(COMPACT_FEATURE_NAMES):
        raise RuntimeError("Probe compact feature names mismatch")
    if int(checkpoint["variant"]) != 2:
        raise RuntimeError("Probe checkpoint is not P1-2")
    if checkpoint["head"] != "near_target_invisible":
        raise RuntimeError("Probe checkpoint is not near-target-invisible")
    config = checkpoint["config"]
    model = build_model(
        config["kind"], int(checkpoint["input_dim"]),
        int(config["hidden"]), int(config["seed"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    if len(mean) != int(checkpoint["input_dim"]):
        raise RuntimeError("Probe standardization dimension mismatch")
    index = {name: i for i, name in enumerate(names)}
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "pos_weight": float(checkpoint["pos_weight"]),
        "feature_names": names,
        "compact_feature_names": compact_names,
        "compact_idx": compact_indices(names),
        "index": index,
        "input_dim": int(checkpoint["input_dim"]),
        "checkpoint": checkpoint,
    }


def high_confidence(features, index, config):
    top1 = max(
        float(features[index[f"heat_{channel}_top1"]])
        for channel in range(3)
    )
    entropy = min(
        float(features[index[f"heat_{channel}_ent"]])
        for channel in range(3)
    )
    centered_x = 2.0 * float(features[index["base_cx"]]) - 1.0
    high = bool(
        top1 >= config.high_top1 and entropy <= config.high_entropy
    )
    return high, centered_x, top1, entropy


def run_episode(spec, model, encoder, probe, oracle, device, config,
                threshold, stop_threshold):
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
    policy = ProbeRecoveryPolicy(config)
    history = []
    trajectory = [env.position.copy()]
    min_distance = float(env.distance_to_goal)
    osr = bool(env.distance_to_goal < env.stop_radius)
    success = False
    done = False
    ended_by_stop = False
    rows = []
    probe_rows = []

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
            STOP if arrival_probability >= float(stop_threshold)
            else base_move
        )
        projection = projection_vector(env)
        distance = float(env.distance_to_goal)
        dx = float(env.goal[0] - env.x)
        dy = float(env.goal[1] - env.y)
        heading_error = wrap_angle(math.atan2(dy, dx) - env.yaw)
        oracle_action = oracle_move_action(env)
        high, centered_x, top1, entropy = high_confidence(
            features, probe["index"], config
        )
        probe_ready = bool(len(history) >= 2 and base_action != STOP)
        if probe_ready:
            probe_input = build_probe_input(features, history)
            if len(probe_input) != probe["input_dim"]:
                raise RuntimeError(
                    f"probe input {len(probe_input)} != "
                    f"{probe['input_dim']}"
                )
            p_near = probe_predict(
                probe["model"], probe["mean"], probe["std"],
                probe["pos_weight"], probe_input, device,
            )
        else:
            p_near = 0.0
        action, recovering, trigger_now, info = policy.act(
            p_near, base_action, step, high, centered_x
        )
        if probe_ready:
            probe_rows.append({
                "episode_index": int(spec["episode_index"]),
                "seed": int(spec["seed"]),
                "band": str(spec["band"]),
                "step": int(step),
                "p_near_invisible": float(p_near),
                "probe_flag": int(p_near >= float(threshold)),
                "distance_m": float(distance),
                "goal_visible": int(projection is not None),
                "near_target_invisible": int(
                    distance < 100.0 and projection is None
                ),
                "heat_top1": float(top1),
                "heat_entropy": float(entropy),
            })
        rows.append({
            "policy": "R1",
            "episode_index": int(spec["episode_index"]),
            "seed": int(spec["seed"]),
            "band": str(spec["band"]),
            "step": int(step),
            "distance_m": float(distance),
            "goal_visible": int(projection is not None),
            "goal_px": projection["goal_px"] if projection else float("nan"),
            "goal_py": projection["goal_py"] if projection else float("nan"),
            "heading_error_deg": math.degrees(heading_error),
            "oracle_move_action": int(oracle_action),
            "base_action": int(base_action),
            "action": int(action),
            "recovering": int(recovering),
            "trigger_now": int(trigger_now),
            "arrival_probability": float(arrival_probability),
            "probe_ready": int(probe_ready),
            "p_near_invisible": (
                float(p_near) if probe_ready else float("nan")
            ),
            "probe_flag": int(info["probe_flag"]) if probe_ready else 0,
            "high_conf": int(high),
            "heat_top1": float(top1),
            "heat_entropy": float(entropy),
            "base_cx_centered": float(centered_x),
            "last_high_side": info["last_high_side"],
            "steps_since_high_conf": int(info["steps_since_high_conf"]),
            "cooldown": int(info["cooldown"]),
            "recovery_remaining": int(info["recovery_remaining"]),
            "movement_label": (
                movement_error_label(int(action), int(oracle_action))
                if int(action) != STOP else "stop"
            ),
        })
        obs, step_success, done = env.step(
            action, render_observation=(action != STOP)
        )
        success = bool(step_success)
        ended_by_stop = bool(action == STOP)
        trajectory.append(env.position.copy())
        min_distance = min(min_distance, float(env.distance_to_goal))
        osr = bool(osr or env.distance_to_goal < env.stop_radius)
        compact = features[probe["compact_idx"]].copy()
        history.insert(0, (compact, int(action)))
        del history[2:]
        if done:
            break
    policy.finalize(len(rows))

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
        "policy": "R1",
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
        "trigger_count": int(policy.trigger_count),
        "recovery_steps": int(policy.recovery_step_count),
        "reacquired_count": int(sum(
            1 for event in policy.events
            if event.get("reacquired_step", -1) >= 0
        )),
        "probe_evaluations": int(len(probe_rows)),
    }
    events = []
    for event in policy.events:
        trigger_index = int(event["trigger_step"])
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
        recovery_rows = rows[
            trigger_index: trigger_index + int(event["recovery_steps"])
        ]
        wrong = Counter(
            item["movement_label"] for item in recovery_rows
            if item["movement_label"] not in ("correct", "stop")
        )
        events.append({
            "policy": "R1",
            "episode_index": int(spec["episode_index"]),
            "band": str(spec["band"]),
            "trigger_step": trigger_index,
            "trigger_probability": float(event["trigger_probability"]),
            "trigger_goal_visible": int(row["goal_visible"]),
            "trigger_distance_m": float(row["distance_m"]),
            "trigger_heading_error_deg": float(row["heading_error_deg"]),
            "last_high_side": event["last_high_side"],
            "steps_since_high_conf": int(
                event["steps_since_high_conf"]
            ),
            "revisible_5": int(revisible_5),
            "revisible_10": int(revisible_10),
            "entered_20_within_40": int(entered_20),
            "recovery_actions": int(event["recovery_steps"]),
            "recovery_wrong": int(sum(wrong.values())),
            "recovery_error_rate": float(
                sum(wrong.values()) / max(int(event["recovery_steps"]), 1)
            ),
            "reacquired_step": int(event["reacquired_step"]),
            "exit_reason": str(event["exit_reason"]),
        })
    return episode, rows, events, probe_rows


def baseline_rows(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = {
                "episode_index": int(row["episode_index"]),
                "seed": int(row["seed"]),
                "band": str(row["band"]),
                "success": int(row["success"]),
                "osr": int(row["osr"]),
                "premature_stop": int(row["premature_stop"]),
                "max_steps_termination": int(row["max_steps_termination"]),
                "steps": int(row["steps"]),
                "spl": float(row["spl"]),
                "ndtw": float(row["ndtw"]),
                "min_distance_m": float(row["min_distance_m"]),
                "final_distance_m": float(row["final_distance_m"]),
                "reached_radius_not_success": int(
                    row["reached_radius_not_success"]
                ),
            }
            rows[converted["episode_index"]] = converted
    return rows


def band_summary(rows):
    output = {"all": summarize(rows)}
    for band in BANDS:
        output[band] = summarize([
            row for row in rows if str(row["band"]) == band
        ])
    return output


def bootstrap_delta(pairs, key, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    deltas = np.asarray([
        float(r1[key]) - float(r0[key]) for r0, r1 in pairs
    ])
    draws = []
    for _ in range(int(n_boot)):
        sample = rng.integers(0, len(deltas), size=len(deltas))
        draws.append(float(deltas[sample].mean()))
    return {
        "mean": float(deltas.mean()),
        "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
        "wins": int(sum(1 for r0, r1 in pairs if r1[key] > r0[key])),
        "losses": int(sum(1 for r0, r1 in pairs if r1[key] < r0[key])),
        "ties": int(sum(1 for r0, r1 in pairs if r1[key] == r0[key])),
    }


def paired_summary(pairs):
    return {
        "success": {
            "wins": int(sum(
                1 for r0, r1 in pairs
                if r1["success"] == 1 and r0["success"] == 0
            )),
            "losses": int(sum(
                1 for r0, r1 in pairs
                if r1["success"] == 0 and r0["success"] == 1
            )),
            "ties": int(sum(
                1 for r0, r1 in pairs
                if r1["success"] == r0["success"]
            )),
            "sr_delta": float(np.mean([
                r1["success"] - r0["success"] for r0, r1 in pairs
            ])),
        },
        "spl": bootstrap_delta(pairs, "spl"),
        "ndtw": bootstrap_delta(pairs, "ndtw"),
        "steps": bootstrap_delta(pairs, "steps"),
        "timeout": bootstrap_delta(pairs, "max_steps_termination"),
        "premature": bootstrap_delta(pairs, "premature_stop"),
    }


def probe_step_metrics(probe_rows, threshold):
    if not probe_rows:
        return {"n": 0}
    p = np.asarray([
        row["p_near_invisible"] for row in probe_rows
    ], dtype=float)
    flags = p >= float(threshold)
    distance = np.asarray([
        row["distance_m"] for row in probe_rows
    ], dtype=float)
    visible = np.asarray([
        row["goal_visible"] for row in probe_rows
    ], dtype=int)
    near_invisible = (distance < 100.0) & (visible == 0)
    far_visible = (distance >= 100.0) & (visible == 1)
    far_invisible = (distance >= 100.0) & (visible == 0)
    near_visible = (distance < 100.0) & (visible == 1)
    tp = int((flags & near_invisible).sum())
    n_flags = int(flags.sum())
    n_near = int(near_invisible.sum())
    n_far_vis = int(far_visible.sum())
    n_far_inv = int(far_invisible.sum())
    n_near_vis = int(near_visible.sum())
    return {
        "n": int(len(probe_rows)),
        "threshold": float(threshold),
        "near_target_invisible": {
            "n": n_near,
            "precision": tp / max(n_flags, 1),
            "recall": tp / max(n_near, 1),
            "flags": n_flags,
        },
        "far_visible_false_trigger": {
            "n": n_far_vis,
            "rate": int((flags & far_visible).sum()) / max(n_far_vis, 1),
        },
        "far_invisible_false_trigger": {
            "n": n_far_inv,
            "rate": int((flags & far_invisible).sum()) / max(n_far_inv, 1),
        },
        "near_visible_false_trigger": {
            "n": n_near_vis,
            "rate": int((flags & near_visible).sum()) / max(n_near_vis, 1),
        },
        "auroc": auc_rank(near_invisible.astype(int), p),
        "auprc": average_precision(near_invisible.astype(int), p),
        "brier": brier_score(near_invisible, p),
        "ece": expected_calibration_error(near_invisible, p),
    }


def bootstrap_probe(probe_rows, threshold, n_boot=200, seed=0):
    episodes = np.asarray([
        row["episode_index"] for row in probe_rows
    ], dtype=int)
    unique = np.unique(episodes)
    index_by_episode = {
        int(episode): np.flatnonzero(episodes == episode)
        for episode in unique
    }
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for _ in range(int(n_boot)):
        sample = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([
            index_by_episode[int(episode)] for episode in sample
        ])
        rows = [probe_rows[int(i)] for i in indices]
        metrics = probe_step_metrics(rows, threshold)
        if metrics["near_target_invisible"]["n"] == 0:
            continue
        draws["precision"].append(
            metrics["near_target_invisible"]["precision"]
        )
        draws["recall"].append(
            metrics["near_target_invisible"]["recall"]
        )
        draws["far_visible_false_trigger"].append(
            metrics["far_visible_false_trigger"]["rate"]
        )
        draws["far_invisible_false_trigger"].append(
            metrics["far_invisible_false_trigger"]["rate"]
        )
        draws["auroc"].append(metrics["auroc"])
    output = {}
    for name, values in draws.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            output[name] = {
                "low": float(np.percentile(values, 2.5)),
                "high": float(np.percentile(values, 97.5)),
                "n_boot": int(len(values)),
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    parser.add_argument("--episodes-per-band", type=int, default=12)
    parser.add_argument(
        "--selection", default="even", choices=["even", "first"]
    )
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--reproduction-check", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--recovery-steps", type=int, default=2)
    parser.add_argument("--cooldown", type=int, default=5)
    parser.add_argument("--side-window", type=int, default=20)
    parser.add_argument("--high-top1", type=float, default=0.20)
    parser.add_argument("--high-entropy", type=float, default=0.85)
    parser.add_argument("--stop-threshold", type=float, default=STOP_THRESHOLD)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "steps").mkdir()
    (args.output_dir / "events").mkdir()
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    threshold = args.threshold
    reproduction = None
    if args.reproduction_check is not None:
        reproduction = json.loads(
            args.reproduction_check.read_text(encoding="utf-8")
        )
        if not reproduction.get("reproduced"):
            raise SystemExit(
                "Reproduction check says the probe was not reproduced; "
                "run the new-probe threshold validation first."
            )
        if not reproduction.get("gate_pass"):
            raise SystemExit("Reproduction gate did not pass")
        if threshold is None:
            threshold = float(reproduction["selected_threshold"])
    if threshold is None:
        threshold = 0.13694876571430528
    config = ProbeRecoveryConfig(
        threshold=float(threshold),
        recovery_steps=int(args.recovery_steps),
        cooldown=int(args.cooldown),
        side_window=int(args.side_window),
        high_top1=float(args.high_top1),
        high_entropy=float(args.high_entropy),
    )

    specs = select_specs(
        load_dev_specs(args.val_specs, 180), args.episodes_per_band,
        method=args.selection,
    )
    if args.max_episodes is not None:
        specs = specs[: int(args.max_episodes)]
    oracle_cache = build_oracle_cache(specs)
    encoder = load_encoder(device)
    model = load_model("M0", M0_CKPT, device)
    probe = load_probe(args.probe_checkpoint, device)
    if len(probe["compact_feature_names"]) != 74:
        raise RuntimeError("Unexpected compact feature dimension")
    print(
        f"episodes={len(specs)} threshold={threshold:.12f} "
        f"recovery_steps={config.recovery_steps} cooldown={config.cooldown}",
        flush=True,
    )

    r0 = baseline_rows(args.baseline_csv)
    if len(specs) == 36:
        spec_keys = [
            (int(s["episode_index"]), int(s["seed"]), str(s["band"]))
            for s in specs
        ]
        base_keys = [
            (int(row["episode_index"]), int(row["seed"]), str(row["band"]))
            for row in r0.values()
        ]
        if sorted(spec_keys) != sorted(base_keys):
            raise SystemExit(
                "R0 baseline episodes/seeds/bands do not match selection"
            )

    episodes = []
    all_rows = []
    all_events = []
    all_probe_rows = []
    started = time.time()
    for position, spec in enumerate(specs):
        episode, rows, events, probe_rows = run_episode(
            spec, model, encoder, probe,
            oracle_cache[int(spec["episode_index"])], device, config,
            threshold, args.stop_threshold,
        )
        episodes.append(episode)
        all_rows.extend(rows)
        all_events.extend(events)
        all_probe_rows.extend(probe_rows)
        if (position + 1) % 6 == 0:
            print(
                f"  R1: {position + 1}/{len(specs)} "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )
    write_csv(args.output_dir / "r1_per_episode.csv", episodes)
    write_csv(args.output_dir / "steps" / "r1.csv", all_rows)
    write_csv(args.output_dir / "events" / "r1.csv", all_events)
    write_csv(args.output_dir / "probe_steps.csv", all_probe_rows)

    r1_summary = band_summary(episodes)
    r0_rows = [r0[int(spec["episode_index"])] for spec in specs]
    r0_summary = band_summary(r0_rows)
    pairs = list(zip(r0_rows, episodes))
    paired = paired_summary(pairs) if len(specs) == 36 else {"n": len(pairs)}
    probe_metrics = probe_step_metrics(all_probe_rows, threshold)
    probe_ci = bootstrap_probe(all_probe_rows, threshold)
    trigger_total = int(sum(e["trigger_count"] for e in episodes))
    trigger_episodes = int(sum(1 for e in episodes if e["trigger_count"] > 0))
    trigger_invisible = int(sum(
        1 for e in all_events if e["trigger_goal_visible"] == 0
    ))
    recovery_actions = int(sum(
        e["recovery_actions"] for e in all_events
    ))
    recovery_wrong = int(sum(e["recovery_wrong"] for e in all_events))
    trigger_metrics = {
        "triggers": trigger_total,
        "episodes_with_trigger": trigger_episodes,
        "trigger_true_invisible": trigger_invisible,
        "trigger_invisible_rate": (
            trigger_invisible / max(len(all_events), 1)
        ),
        "revisible_5": float(np.mean([
            e["revisible_5"] for e in all_events
        ])) if all_events else float("nan"),
        "revisible_10": float(np.mean([
            e["revisible_10"] for e in all_events
        ])) if all_events else float("nan"),
        "entered_20_within_40": float(np.mean([
            e["entered_20_within_40"] for e in all_events
        ])) if all_events else float("nan"),
        "recovery_actions": recovery_actions,
        "recovery_wrong": recovery_wrong,
        "recovery_error_rate": (
            recovery_wrong / max(recovery_actions, 1)
        ),
        "max_triggers_per_episode": int(max(
            [e["trigger_count"] for e in episodes] or [0]
        )),
        "episodes_with_repeated_trigger": int(sum(
            1 for e in episodes if e["trigger_count"] >= 2
        )),
    }
    # Harness consistency: episodes without a trigger must reproduce R0.
    no_trigger_mismatches = []
    for reference, episode_row in zip(r0_rows, episodes):
        if episode_row["trigger_count"] != 0:
            continue
        for key in ("success", "osr", "premature_stop",
                    "max_steps_termination", "steps"):
            if int(reference[key]) != int(episode_row[key]):
                no_trigger_mismatches.append({
                    "episode_index": episode_row["episode_index"],
                    "field": key,
                    "r0": int(reference[key]),
                    "r1": int(episode_row[key]),
                })
    sr_delta = float(
        r1_summary["all"]["sr"] - r0_summary["all"]["sr"]
    )
    spl_delta = float(
        r1_summary["all"]["spl"] - r0_summary["all"]["spl"]
    )
    ndtw_delta = float(
        r1_summary["all"]["ndtw"] - r0_summary["all"]["ndtw"]
    )
    timeout_delta = float(
        r1_summary["all"]["timeout"] - r0_summary["all"]["timeout"]
    )
    all_primary_not_improved = bool(
        sr_delta <= 0 and spl_delta <= 0 and ndtw_delta <= 0
    )
    if all_primary_not_improved:
        decision = "STOP_NO_SR_SPL_NDTW_IMPROVEMENT"
    elif timeout_delta > 0.05:
        decision = "STOP_TIMEOUT_INCREASE"
    elif trigger_metrics["max_triggers_per_episode"] >= 3:
        decision = "STOP_REPEATED_TRIGGERING"
    elif (
        recovery_actions > 0
        and recovery_wrong / max(recovery_actions, 1) > 0.6
    ):
        decision = "STOP_WRONG_REDIRECTION"
    else:
        decision = "SMOKE_POSITIVE_REVIEW_180"
    output = {
        "scope": "36-episode mechanism smoke; not a final performance result",
        "r0_source": str(args.baseline_csv),
        "r1": r1_summary,
        "r0": r0_summary,
        "paired": paired,
        "triggers": trigger_metrics,
        "probe": {
            "step_metrics": probe_metrics,
            "episode_bootstrap_95ci": probe_ci,
        },
        "decision": {
            "sr_delta": sr_delta,
            "spl_delta": spl_delta,
            "ndtw_delta": ndtw_delta,
            "timeout_delta": timeout_delta,
            "all_primary_not_improved": all_primary_not_improved,
            "max_triggers_per_episode": trigger_metrics[
                "max_triggers_per_episode"
            ],
            "status": decision,
        },
        "no_trigger_r0_mismatches": no_trigger_mismatches,
        "config": {
            "threshold": float(threshold),
            "recovery_steps": int(config.recovery_steps),
            "cooldown": int(config.cooldown),
            "side_window": int(config.side_window),
            "high_top1": float(config.high_top1),
            "high_entropy": float(config.high_entropy),
            "stop_threshold": float(args.stop_threshold),
            "episodes": len(specs),
            "episodes_per_band": int(args.episodes_per_band),
            "selection": args.selection,
        },
        "reproduction": reproduction,
        "runtime_s": float(time.time() - started),
        "probe_checkpoint": str(args.probe_checkpoint),
    }
    write_json(args.output_dir / "metrics.json", output)
    provenance = {
        "command": sys.argv,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "probe_checkpoint": str(args.probe_checkpoint),
        "baseline_csv": str(args.baseline_csv),
        "val_specs": str(args.val_specs),
        "threshold": float(threshold),
        "reproduction_check": str(args.reproduction_check),
        "label_usage": (
            "true distance/visibility/heading/oracle action are recorded "
            "only for offline scoring; the policy uses observable probe "
            "probability, M0 base action and observable high-confidence side"
        ),
    }
    write_json(args.output_dir / "provenance.json", provenance)
    print(json.dumps({
        "r0": r0_summary["all"],
        "r1": r1_summary["all"],
        "paired_success": paired["success"] if "success" in paired else None,
        "triggers": trigger_metrics,
        "probe": probe_metrics,
        "decision": output["decision"],
        "output": str(args.output_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
