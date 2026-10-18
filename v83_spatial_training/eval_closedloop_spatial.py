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
if __package__:
    from .spatial_mec import SpatialAwareMEC
else:  # direct script execution (sys.path[0] is v83_spatial_training/)
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


def synchronize(device):
    """CUDA kernels are asynchronous; timing without this is not latency."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def spatial_maps_for_mode(mode, tracker, current_tokens, goal_tokens, prev_action):
    if mode == "M0":
        return None, None
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
    return maps, valid


def evaluate_mode(mode, model_name, checkpoint, specs, oracle_cache, encoder, device,
                  threshold, max_episodes):
    model = SpatialAwareMEC(build_gbs_model(model_name), mode).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    rows = []
    for episode_position, spec in enumerate(specs[:max_episodes]):
        # The first episode is a warm-up for latency measurement only; all
        # episodes still contribute to SR/OSR/SPL/nDTW and paired statistics.
        record_timing = episode_position > 0
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
            synchronize(device)
            t_map = time.perf_counter()
            maps, valid = spatial_maps_for_mode(
                mode, tracker, current_tokens, goal_tokens, prev_action
            )
            synchronize(device)
            map_elapsed = time.perf_counter() - t_map
            synchronize(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                decision = model.predict_action(
                    current_tokens, goal_tokens, heatmap, network,
                    spatial_maps=maps, spatial_valid=valid,
                    stop_threshold=threshold,
                )
            synchronize(device)
            model_elapsed = time.perf_counter() - t0
            if record_timing:
                map_times.append(map_elapsed)
                model_times.append(model_elapsed)
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
            "timed": int(record_timing),
            "map_ms_per_step": float(1000.0 * np.mean(map_times)) if map_times else 0.0,
            "model_ms_per_step": float(1000.0 * np.mean(model_times)) if model_times else 0.0,
        })
    return rows


def summarize(rows):
    n = len(rows)
    timed = [r for r in rows if r.get("timed", 1)]
    map_ms = [r["map_ms_per_step"] for r in timed]
    model_ms = [r["model_ms_per_step"] for r in timed]
    total_ms = [r["map_ms_per_step"] + r["model_ms_per_step"] for r in timed]
    return {
        "episodes": n,
        "sr": float(np.mean([r["success"] for r in rows])) if n else 0.0,
        "osr": float(np.mean([r["osr"] for r in rows])) if n else 0.0,
        "premature_stop": float(np.mean([r["premature_stop"] for r in rows])) if n else 0.0,
        "timeout": float(np.mean([r["max_steps_termination"] for r in rows])) if n else 0.0,
        "spl": float(np.mean([r["spl"] for r in rows])) if n else 0.0,
        "ndtw": float(np.mean([r["ndtw"] for r in rows])) if n else 0.0,
        "latency_episodes": len(timed),
        # Synchronized per-episode means; first episode excluded as warm-up.
        "map_ms_per_step": float(np.mean(map_ms)) if map_ms else 0.0,
        "map_ms_per_step_p95": float(np.percentile(map_ms, 95)) if map_ms else 0.0,
        "model_ms_per_step": float(np.mean(model_ms)) if model_ms else 0.0,
        "model_ms_per_step_p95": float(np.percentile(model_ms, 95)) if model_ms else 0.0,
        "total_ms_per_step": float(np.mean(total_ms)) if total_ms else 0.0,
        "total_ms_per_step_p95": float(np.percentile(total_ms, 95)) if total_ms else 0.0,
    }


def paired_bootstrap_ci(diffs, n_boot=10000, seed=20261018):
    """95% percentile CI for the mean paired difference, resampling episodes."""
    values = np.asarray(diffs, dtype=np.float64)
    if values.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(values, size=(int(n_boot), values.size), replace=True)
    means = draws.mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def sign_test_from_diffs(diffs):
    pos = sum(1 for d in diffs if d > 1e-12)
    neg = sum(1 for d in diffs if d < -1e-12)
    return sign_test_p(pos, neg)


def paired_comparison(rows_by_mode, base="M0", other="M1"):
    if base not in rows_by_mode or other not in rows_by_mode:
        return {"note": f"{base}/{other} not both available"}
    a = {int(r["episode_index"]): r for r in rows_by_mode[base]}
    b = {int(r["episode_index"]): r for r in rows_by_mode[other]}
    indices = sorted(set(a) & set(b))
    res = {"base": base, "other": other, "episodes": len(indices)}
    for key in ("success", "osr", "premature_stop", "max_steps_termination"):
        # High SR/OSR is desirable; a high premature-stop/timeout rate is not.
        higher_is_better = key in ("success", "osr")
        improvements = sum(
            (int(b[i][key]) > int(a[i][key])) if higher_is_better
            else (int(b[i][key]) < int(a[i][key]))
            for i in indices
        )
        regressions = sum(
            (int(b[i][key]) < int(a[i][key])) if higher_is_better
            else (int(b[i][key]) > int(a[i][key]))
            for i in indices
        )
        ties = len(indices) - improvements - regressions
        diffs = [int(b[i][key]) - int(a[i][key]) for i in indices]
        res[key] = {
            "other_better": improvements,
            "base_better": regressions,
            "ties": ties,
            # Signed new minus old occurrence rate, regardless of desirability.
            "delta_rate": float(np.mean(diffs)) if diffs else 0.0,
            "delta_ci95": paired_bootstrap_ci(diffs),
        }
    for metric in ("spl", "ndtw", "min_distance_m"):
        diffs = [float(b[i][metric]) - float(a[i][metric]) for i in indices]
        res[f"{metric}_delta"] = float(np.mean(diffs)) if diffs else 0.0
        res[f"{metric}_delta_ci95"] = paired_bootstrap_ci(diffs)
        if metric in ("spl", "ndtw"):
            res[metric] = {
                "other_better": sum(1 for d in diffs if d > 1e-12),
                "base_better": sum(1 for d in diffs if d < -1e-12),
                "ties": sum(1 for d in diffs if abs(d) <= 1e-12),
                "delta": res[f"{metric}_delta"],
            }
    for metric in ("success", "osr"):
        res[f"{metric}_sign_test_p"] = sign_test_p(
            res[metric]["other_better"], res[metric]["base_better"]
        )
    res["spl_sign_test_p"] = sign_test_from_diffs(
        [float(b[i]["spl"]) - float(a[i]["spl"]) for i in indices]
    )
    res["ndtw_sign_test_p"] = sign_test_from_diffs(
        [float(b[i]["ndtw"]) - float(a[i]["ndtw"]) for i in indices]
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
    p.add_argument(
        "--reference-model-dir", type=Path, default=None,
        help="Existing M0/M1 checkpoint directory for paired M2 evaluation; "
             "never copy or overwrite archived checkpoints.",
    )
    p.add_argument("--model-name", default="Transformer-M")
    p.add_argument("--modes", nargs="+", default=["M0", "M1"])
    p.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    p.add_argument("--max-episodes", type=int, default=60)
    p.add_argument("--threshold", type=float, default=ARRIVAL_STOP_THRESHOLD)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    evaluation.TEST_STATS_PATH = args.val_specs
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
    checkpoint_sources = {}
    for mode in args.modes:
        ckpt = (
            args.model_dir / "smoke" / f"{mode}.pt" if args.smoke
            else args.model_dir / args.model_name / mode / "best.pt"
        )
        # M2 can be trained in its own immutable run directory. Read original
        # M0/M1 weights from the explicitly specified comparison run.
        if not ckpt.is_file() and args.reference_model_dir is not None:
            ref_ckpt = (
                args.reference_model_dir / "smoke" / f"{mode}.pt" if args.smoke
                else args.reference_model_dir / args.model_name / mode / "best.pt"
            )
            if ref_ckpt.is_file():
                ckpt = ref_ckpt
        if not ckpt.is_file():
            raise SystemExit(f"Missing checkpoint for {mode}: {ckpt}")
        checkpoint_sources[mode] = str(ckpt)
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
    comparisons = {}
    for base, other in (("M0", "M1"), ("M0", "M2"), ("M1", "M2")):
        if base in rows_by_mode and other in rows_by_mode:
            comparisons[f"paired_{other}_vs_{base}"] = paired_comparison(
                rows_by_mode, base=base, other=other
            )
    by_band = {}
    for mode, rows in rows_by_mode.items():
        bands = sorted({str(r["band"]) for r in rows})
        by_band[mode] = {
            band: summarize([r for r in rows if str(r["band"]) == band])
            for band in bands
        }
    result = {
        "summaries": summaries,
        "summaries_by_band": by_band,
        "checkpoint_sources": checkpoint_sources,
        **comparisons,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
