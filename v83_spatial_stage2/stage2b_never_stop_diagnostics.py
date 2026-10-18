"""Stage-2b Never-Stop counterfactual and far premature-Stop diagnostics.

Loads the existing Stage-2a Transformer-M M0/M1 checkpoints read-only. It does
NOT train or modify movement weights. For each model it runs:
  * normal policy: records Stop events (distance, arrival probability, offline
    Goal visibility);
  * never-Stop counterfactual: forces the movement argmax, measures OSR, min
    distance and timeout by band.

Goal visibility is computed from the simulator projection only after the model
decision; it is offline diagnostic data and is never a model input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_fixed_mec_epoch_compare as evaluation
from config import ARRIVAL_STOP_THRESHOLD, ENCODER
from env import STOP, ToyUAVEnv
from gbs_model_v3 import build_gbs_model
from gbs_network import build_random_gbs_layout, global_network_state
from token_data import encode_multiscale_goal_tokens, goal_conditioned_spatial_heatmap
from utils import image_to_tensor
from v83_spatial_training.eval_closedloop_spatial import (
    load_encoder,
    spatial_maps_for_mode,
)
from v83_spatial_training.spatial_mec import SpatialAwareMEC


ARCHIVE = Path("runs/archive_pycharm_v82_20261018")
VAL_SPECS = Path("runs/v83_spatial_stage2/prep_audit_20261018T043420Z/dev_val_180_spec.json")
STAGE2A = Path("runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z")
BANDS = ("short", "medium", "long")


def load_model(mode, device):
    model = SpatialAwareMEC(build_gbs_model("Transformer-M"), mode).to(device)
    ckpt = STAGE2A / "Transformer-M" / mode / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model, ckpt


def _mean(values):
    return float(np.mean(values)) if values else float("nan")


def _median(values):
    return float(np.median(values)) if values else float("nan")


def run_episode(model, mode, spec, encoder, device, never_stop, threshold):
    env = ToyUAVEnv()
    obs = env.reset(
        seed=spec["seed"],
        goal_distance_range_m=spec["distance_range"],
        render_observation=True,
    )
    nodes = build_random_gbs_layout(seed=spec["network_layout_seed"], verbose=False)
    goal_tokens = encode_multiscale_goal_tokens(
        env.get_goal_reference_multiscale_rgbs(), encoder, device
    ).unsqueeze(0).to(device, dtype=torch.float32)
    tracker = None
    prev_action = None
    stop_events = []
    min_distance = float(env.distance_to_goal)
    osr = bool(env.distance_to_goal < env.stop_radius)
    success = False
    done = False
    ended_by_stop = False
    steps = 0
    for step in range(evaluation.MAX_STEPS):
        current_tokens = encoder(image_to_tensor(obs, device))
        heatmap = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
        network_np = global_network_state(
            nodes, env.position,
            num_tokens=ENCODER.num_tokens, token_dim=ENCODER.token_dim,
        ).astype(np.float32)
        network = torch.from_numpy(network_np).unsqueeze(0).to(device)
        maps, valid = spatial_maps_for_mode(
            mode, tracker, current_tokens, goal_tokens, prev_action
        )
        with torch.inference_mode():
            decision = model.predict_action(
                current_tokens, goal_tokens, heatmap, network,
                spatial_maps=maps, spatial_valid=valid,
                stop_threshold=threshold,
            )
        arrival_p = float(decision["arrival_probability"].item())
        normal_action = int(decision["action"].item())
        if never_stop:
            action = int(decision["move_probabilities"].argmax(dim=-1).item())
            ended_by_stop = False
        else:
            action = normal_action
            ended_by_stop = bool(action == STOP)
        distance = float(env.distance_to_goal)
        if ended_by_stop:
            projection = env.goal_projection()
            goal_visible = projection is not None
            center_inside = False
            if projection is not None:
                px, py = float(projection[0]), float(projection[1])
                center_inside = 0.0 <= px < env.img_size and 0.0 <= py < env.img_size
            stop_events.append({
                "mode": mode,
                "episode_index": spec["episode_index"],
                "seed": spec["seed"],
                "band": spec["band"],
                "step": step,
                "distance_m": distance,
                "arrival_probability": arrival_p,
                "goal_visible": int(goal_visible),
                "goal_center_inside": int(center_inside),
                "move_action": action,
            })
        obs, step_success, done = env.step(
            action, render_observation=(action != STOP)
        )
        success = bool(step_success)
        steps = step + 1
        min_distance = min(min_distance, float(env.distance_to_goal))
        osr = bool(osr or env.distance_to_goal < env.stop_radius)
        prev_action = action
        if done:
            break
    premature = bool(ended_by_stop and not success)
    timeout = bool((not ended_by_stop) and (not success) and done)
    episode = {
        "mode": mode,
        "policy": "never" if never_stop else "normal",
        "episode_index": spec["episode_index"],
        "seed": spec["seed"],
        "band": spec["band"],
        "success": int(success),
        "osr": int(osr),
        "premature_stop": int(premature),
        "timeout": int(timeout),
        "steps": steps,
        "min_distance_m": min_distance,
        "arrival_p_at_stop": stop_events[-1]["arrival_probability"] if stop_events else float("nan"),
        "goal_visible_at_stop": stop_events[-1]["goal_visible"] if stop_events else None,
    }
    return episode, stop_events


def summarize_never(rows):
    out = []
    for mode in ("M0", "M1"):
        for band in BANDS:
            sub = [r for r in rows if r["mode"] == mode and r["band"] == band]
            if not sub:
                continue
            out.append({
                "mode": mode, "band": band, "episodes": len(sub),
                "osr": _mean([int(r["osr"]) for r in sub]),
                "timeout": _mean([int(r["timeout"]) for r in sub]),
                "min_distance_mean_m": _mean([float(r["min_distance_m"]) for r in sub]),
                "min_distance_median_m": _median([float(r["min_distance_m"]) for r in sub]),
                "min_distance_min_m": min(float(r["min_distance_m"]) for r in sub),
                "min_distance_max_m": max(float(r["min_distance_m"]) for r in sub),
                "mean_steps": _mean([int(r["steps"]) for r in sub]),
            })
    return out


def summarize_stop_events(events):
    out = []
    for mode in ("M0", "M1"):
        for band in BANDS:
            sub = [e for e in events if e["mode"] == mode and e["band"] == band]
            if not sub:
                continue
            prem = [e for e in sub if e["distance_m"] > 20.0]
            far40 = [e for e in sub if e["distance_m"] > 40.0]
            far100 = [e for e in sub if e["distance_m"] > 100.0]
            out.append({
                "mode": mode, "band": band, "stops": len(sub),
                "premature_gt20": len(prem),
                "far_gt40": len(far40),
                "far_gt100": len(far100),
                "distance_mean_m": _mean([e["distance_m"] for e in sub]),
                "distance_median_m": _median([e["distance_m"] for e in sub]),
                "distance_min_m": min((e["distance_m"] for e in sub), default=float("nan")),
                "distance_max_m": max((e["distance_m"] for e in sub), default=float("nan")),
                "arrival_p_mean": _mean([e["arrival_probability"] for e in sub]),
                "arrival_p_median": _median([e["arrival_probability"] for e in sub]),
                "goal_visible_rate": _mean([int(e["goal_visible"]) for e in sub]),
                "goal_center_inside_rate": _mean([int(e["goal_center_inside"]) for e in sub]),
            })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    p.add_argument("--max-episodes", type=int, default=180)
    p.add_argument("--threshold", type=float, default=ARRIVAL_STOP_THRESHOLD)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    evaluation.TEST_STATS_PATH = args.val_specs
    evaluation.NUM_EPISODES = args.max_episodes
    specs = evaluation.load_test_episode_specs()
    encoder = load_encoder(device)

    normal_rows, never_rows, stop_events = [], [], []
    for mode in ("M0", "M1"):
        model, ckpt = load_model(mode, device)
        print(f"{mode} model: {ckpt}", flush=True)
        for spec in specs:
            normal, events = run_episode(
                model, mode, spec, encoder, device, False, args.threshold
            )
            normal_rows.append(normal)
            stop_events.extend(events)
            never, _ = run_episode(
                model, mode, spec, encoder, device, True, args.threshold
            )
            never_rows.append(never)
            if (spec["episode_index"] + 1) % 30 == 0:
                print(f"  {mode} processed {spec['episode_index']+1}/{len(specs)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def write_csv(path, rows):
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    write_csv(args.output_dir / "normal_per_episode.csv", normal_rows)
    write_csv(args.output_dir / "never_per_episode.csv", never_rows)
    write_csv(args.output_dir / "stop_events.csv", stop_events)
    never_summary = summarize_never(never_rows)
    stop_summary = summarize_stop_events(stop_events)
    write_csv(args.output_dir / "never_band_summary.csv", never_summary)
    write_csv(args.output_dir / "stop_band_summary.csv", stop_summary)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": {m: str(STAGE2A / "Transformer-M" / m / "best.pt") for m in ("M0", "M1")},
        "threshold": args.threshold,
        "episodes": len(specs),
        "never_summary": never_summary,
        "stop_summary": stop_summary,
        "note": "Never-Stop forces movement argmax; movement weights untouched. Goal visibility is offline diagnostic only.",
    }
    (args.output_dir / "stage2b_never_diagnostics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
