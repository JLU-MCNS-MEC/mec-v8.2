"""Closed-loop development-validation evaluator for V8.3 M0/M1/M2.

Uses the model's actually executed previous action for temporal features,
resets the visual belief tracker per episode, fixes the Stop threshold for all
models, and reports paired per-episode metrics. The 60 V8.3 episodes are the
existing development-validation set, not an untouched final test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_fixed_mec_epoch_compare as evaluation
from config import ARRIVAL_STOP_THRESHOLD, ENCODER, ENV
from encoder import VisualTokenEncoder
from env import STOP, ToyUAVEnv
from gbs_model_v3 import build_gbs_model
from gbs_network import build_random_gbs_layout, global_network_state
from navigation_reward import ndtw_score
from token_data import encode_multiscale_goal_tokens, goal_conditioned_spatial_heatmap
from utils import image_to_tensor
from visual_grounding_v83 import (
    VisualBeliefTracker,
    static_grounding,
)
from spatial_mec import SpatialAwareMEC


ARCHIVE = Path("runs/archive_pycharm_v82_20261018")
VAL_SPECS = Path("runs/v83_validation/closedloop_val_specs.json")
OUTPUT_ROOT = Path("runs/v83_spatial_training")


def load_encoder(device):
    enc = VisualTokenEncoder().to(device)
    enc.load_state_dict(
        torch.load(ARCHIVE / "shared_encoder.pt", map_location=device, weights_only=True)
    )
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def spatial_maps_for_mode(mode, tracker, current_tokens, goal_tokens, prev_action):
    if mode == "M0":
        return None, None, 0.0
    t0 = time.perf_counter()
    if mode == "M1":
        baseline, coherent = static_grounding(current_tokens, goal_tokens)
        zero = torch.zeros_like(coherent)
        maps = torch.stack([baseline, coherent, zero, zero], dim=1)
        valid = torch.zeros(
            maps.shape[0], 1, 1, 1, device=maps.device, dtype=maps.dtype
        )
    else:
        gm = tracker.track(
            current_tokens, goal_tokens,
            previous_action=None if prev_action is None else prev_action,
        )
        maps = torch.stack(
            [gm.baseline, gm.coherent, gm.temporal, gm.innovation], dim=1
        )
        valid = gm.temporal_valid.float().view(-1, 1, 1, 1)
    return maps, valid, time.perf_counter() - t0


def evaluate_mode(mode, model_name, checkpoint, specs, oracle_cache, encoder, device,
                  threshold, max_episodes):
    model = SpatialAwareMEC(build_gbs_model(model_name), mode).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    rows = []
    for spec in specs[:max_episodes]:
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
        tracker = VisualBeliefTracker()
        prev_action = None
        trajectory = [env.position.copy()]
        min_distance = float(env.distance_to_goal)
        osr = bool(env.distance_to_goal < env.stop_radius)
        success = False
        done = False
        ended_by_stop = False
        map_times = []
        model_times = []
        for step in range(evaluation.MAX_STEPS):
            current_tokens = encoder(image_to_tensor(obs, device))
            heatmap = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
            network_np = global_network_state(
                nodes, env.position,
                num_tokens=ENCODER.num_tokens, token_dim=ENCODER.token_dim,
            ).astype(np.float32)
            network = torch.from_numpy(network_np).unsqueeze(0).to(device)
            maps, valid, mt = spatial_maps_for_mode(
                mode, tracker, current_tokens, goal_tokens, prev_action
            )
            map_times.append(mt)
            t0 = time.perf_counter()
            with torch.inference_mode():
                decision = model.predict_action(
                    current_tokens, goal_tokens, heatmap, network,
                    spatial_maps=maps, spatial_valid=valid,
                    stop_threshold=threshold,
                )
            model_times.append(time.perf_counter() - t0)
            action = int(decision["action"].item())
            ended_by_stop = bool(action == STOP)
            obs, success, done = env.step(
                action, render_observation=(action != STOP)
            )
            trajectory.append(env.position.copy())
            min_distance = min(min_distance, float(env.distance_to_goal))
            osr = bool(osr or env.distance_to_goal < env.stop_radius)
            prev_action = action
            if done:
                break
        trajectory = np.asarray(trajectory, dtype=np.float32)
        path_length = (
            float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())
            if len(trajectory) > 1 else 0.0
        )
        shortest = float(oracle_cache[spec["episode_index"]]["shortest_distance_m"])
        spl = shortest / max(shortest, path_length, 1e-8) if success else 0.0
        ndtw = ndtw_score(
            trajectory=trajectory[:, :2],
            reference=oracle_cache[spec["episode_index"]]["reference"],
            distance_threshold=env.stop_radius,
        )
        premature_stop = bool(ended_by_stop and not success)
        timeout = bool((not ended_by_stop) and (not success) and done)
        rows.append({
            "mode": mode,
            "episode_index": spec["episode_index"],
            "seed": spec["seed"],
            "band": spec["band"],
            "success": int(success),
            "osr": int(osr),
            "premature_stop": int(premature_stop),
            "max_steps_termination": int(timeout),
            "steps": len(trajectory) - 1,
            "spl": float(spl),
            "ndtw": float(ndtw),
            "min_distance_m": float(min_distance),
            "map_ms_per_step": float(1000.0 * np.mean(map_times)) if map_times else 0.0,
            "model_ms_per_step": float(1000.0 * np.mean(model_times)) if model_times else 0.0,
        })
    return rows


def summarize(rows):
    n = len(rows)
    return {
        "episodes": n,
        "sr": float(np.mean([r["success"] for r in rows])) if n else 0.0,
        "osr": float(np.mean([r["osr"] for r in rows])) if n else 0.0,
        "premature_stop": float(np.mean([r["premature_stop"] for r in rows])) if n else 0.0,
        "timeout": float(np.mean([r["max_steps_termination"] for r in rows])) if n else 0.0,
        "spl": float(np.mean([r["spl"] for r in rows])) if n else 0.0,
        "ndtw": float(np.mean([r["ndtw"] for r in rows])) if n else 0.0,
        "map_ms_per_step": float(np.mean([r["map_ms_per_step"] for r in rows])) if n else 0.0,
        "model_ms_per_step": float(np.mean([r["model_ms_per_step"] for r in rows])) if n else 0.0,
    }


def paired_comparison(rows_by_mode, base="M0", other="M1"):
    if base not in rows_by_mode or other not in rows_by_mode:
        return {"note": f"{base}/{other} not both available"}
    a = {int(r["episode_index"]): r for r in rows_by_mode[base]}
    b = {int(r["episode_index"]): r for r in rows_by_mode[other]}
    indices = sorted(set(a) & set(b))
    res = {"base": base, "other": other, "episodes": len(indices)}
    for key in ("success", "osr", "premature_stop", "max_steps_termination"):
        pos = sum(int(b[i][key]) > int(a[i][key]) for i in indices)
        neg = sum(int(b[i][key]) < int(a[i][key]) for i in indices)
        ties = len(indices) - pos - neg
        res[key] = {"other_better": pos, "base_better": neg, "ties": ties,
                    "delta_rate": float(np.mean([int(b[i][key]) - int(a[i][key]) for i in indices]))}
    res["spl_delta"] = float(np.mean([b[i]["spl"] - a[i]["spl"] for i in indices]))
    res["success_sign_test_p"] = sign_test_p(
        sum(int(b[i]["success"]) > int(a[i]["success"]) for i in indices),
        sum(int(b[i]["success"]) < int(a[i]["success"]) for i in indices),
    )
    return res


def sign_test_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--model-name", default="Transformer-M")
    p.add_argument("--modes", nargs="+", default=["M0", "M1"])
    p.add_argument("--max-episodes", type=int, default=60)
    p.add_argument("--threshold", type=float, default=ARRIVAL_STOP_THRESHOLD)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    evaluation.TEST_STATS_PATH = VAL_SPECS
    evaluation.NUM_EPISODES = args.max_episodes
    specs = evaluation.load_test_episode_specs()
    encoder = load_encoder(device)
    oracle_cache = {
        spec["episode_index"]: evaluation.build_oracle_reference(spec) for spec in specs
    }
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_dir = args.model_dir / f"closedloop_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows_by_mode = {}
    summaries = {}
    for mode in args.modes:
        ckpt = (
            args.model_dir / "smoke" / f"{mode}.pt" if args.smoke
            else args.model_dir / args.model_name / mode / "best.pt"
        )
        if not ckpt.is_file():
            raise SystemExit(f"Missing checkpoint {ckpt}")
        print(f"Closed-loop {mode} from {ckpt}", flush=True)
        rows = evaluate_mode(
            mode, args.model_name, ckpt, specs, oracle_cache, encoder, device,
            args.threshold, args.max_episodes,
        )
        rows_by_mode[mode] = rows
        summaries[mode] = summarize(rows)
    all_rows = [r for rows in rows_by_mode.values() for r in rows]
    with (args.output_dir / "per_episode.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"summaries": summaries,
                    "paired_M1_vs_M0": paired_comparison(rows_by_mode)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2), flush=True)
    print("Paired M1 vs M0:", json.dumps(paired_comparison(rows_by_mode), indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
