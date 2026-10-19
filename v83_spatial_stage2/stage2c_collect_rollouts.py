"""Collect on-policy B0 rollouts from Train 600 for Stage-2c off-policy mining.

Preparation for the Stage-2c next stage. It does NOT train, modify checkpoints,
or read Final Test. The frozen B0 M0 checkpoint is rolled out on Train 600
episodes reconstructed from the Stage-2 dataset statistics (same seeds, band
distance ranges and network layouts as token generation).

Every per-step row stores:
  * the 104-D frozen Arrival features (model input only);
  * the original B0 arrival logit / probability;
  * the executed action and policy outcome;
  * true distance and offline Goal visibility as *labels/diagnostics only*.

True distance / visibility are never fed to the model. The next-stage residual
calibration can use the features and the original logit, with labels from
distance < stop radius. Off-policy far negatives are steps with true distance
> 40 m; no GT-derived quantity may be used as an input feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_fixed_mec_epoch_compare as evaluation
from config import ARRIVAL_STOP_THRESHOLD, ENCODER, ENV, TRAIN
from env import STOP, ToyUAVEnv
from gbs_model_v3 import build_gbs_model
from gbs_network import build_random_gbs_layout, global_network_state
from token_data import encode_multiscale_goal_tokens, goal_conditioned_spatial_heatmap
from utils import image_to_tensor
from v83_spatial_training.eval_closedloop_spatial import load_encoder
from v83_spatial_training.spatial_mec import SpatialAwareMEC
from v83_spatial_stage2.stage2b_arrival_ablation import M0_CKPT


STAGE2_ROOT = Path("runs/v83_spatial_stage2")
DATASETS = STAGE2_ROOT / "datasets"
BAND_ID = {"short": 0, "medium": 1, "long": 2}
BAND_RANGE = {
    name: (float(dmin), float(dmax))
    for name, _weight, dmin, dmax, _amin, _amax in TRAIN.trajectory_mix
}


def build_specs(dataset_root, max_episodes=None):
    stats = json.loads((dataset_root / "dataset_stats.json").read_text(encoding="utf-8"))
    specs = []
    for index, trajectory in enumerate(stats["trajectories"]):
        band = trajectory["band"]
        specs.append({
            "episode_index": index,
            "seed": int(trajectory["seed"]),
            "band": band,
            "band_id": BAND_ID[band],
            "distance_range": BAND_RANGE[band],
            "network_layout_seed": int(trajectory["network_layout_seed"]),
            "start_goal_distance_m": float(trajectory["start_goal_distance_m"]),
        })
    if max_episodes is not None:
        specs = specs[: int(max_episodes)]
    return specs


def run_episode(model, spec, encoder, device, threshold):
    env = ToyUAVEnv()
    obs = env.reset(
        seed=spec["seed"],
        goal_distance_range_m=spec["distance_range"],
        render_observation=True,
    )
    observed_start = float(env.distance_to_goal)
    if abs(observed_start - spec["start_goal_distance_m"]) > 1e-2:
        raise RuntimeError(
            f"Episode {spec['episode_index']} start distance {observed_start} "
            f"does not reproduce dataset stats {spec['start_goal_distance_m']}"
        )
    nodes = build_random_gbs_layout(seed=spec["network_layout_seed"], verbose=False)
    goal_tokens = encode_multiscale_goal_tokens(
        env.get_goal_reference_multiscale_rgbs(), encoder, device
    ).unsqueeze(0).to(device, dtype=torch.float32)
    rows = []
    stop_events = []
    min_distance = float(env.distance_to_goal)
    osr = bool(env.distance_to_goal < env.stop_radius)
    success = False
    done = False
    ended_by_stop = False
    for step in range(evaluation.MAX_STEPS):
        current_tokens = encoder(image_to_tensor(obs, device))
        heatmap = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
        network_np = global_network_state(
            nodes, env.position,
            num_tokens=ENCODER.num_tokens, token_dim=ENCODER.token_dim,
        ).astype(np.float32)
        network = torch.from_numpy(network_np).unsqueeze(0).to(device)
        # Inputs are observable only: RGB tokens, Goal tokens, explicit
        # heatmap and network state. No true distance / position / visibility.
        with torch.inference_mode():
            decision = model.predict_action(
                current_tokens, goal_tokens, heatmap, network,
                stop_threshold=threshold,
            )
        distance = float(env.distance_to_goal)
        projection = env.goal_projection()
        goal_visible = int(projection is not None)
        arrival_p = float(decision["arrival_probability"].item())
        arrival_logit = float(decision["arrival_logit"].item())
        action = int(decision["action"].item())
        features = decision["arrival_features"][0].detach().cpu().numpy().astype(np.float32)
        rows.append({
            "episode_index": spec["episode_index"], "seed": spec["seed"],
            "band_id": spec["band_id"], "step": step,
            "distance_m": distance, "goal_visible": goal_visible,
            "arrival_probability": arrival_p, "arrival_logit": arrival_logit,
            "stop": int(action == STOP), "action": action,
            "features": features,
        })
        if action == STOP:
            stop_events.append({
                "episode_index": spec["episode_index"], "seed": spec["seed"],
                "band": spec["band"], "step": step, "distance_m": distance,
                "arrival_probability": arrival_p, "goal_visible": goal_visible,
            })
        obs, step_success, done = env.step(
            action, render_observation=(action != STOP)
        )
        success = bool(step_success)
        ended_by_stop = bool(action == STOP)
        min_distance = min(min_distance, float(env.distance_to_goal))
        osr = bool(osr or env.distance_to_goal < env.stop_radius)
        if done:
            break
    episode = {
        "episode_index": spec["episode_index"], "seed": spec["seed"],
        "band": spec["band"], "success": int(success), "osr": int(osr),
        "premature_stop": int(ended_by_stop and not success),
        "timeout": int((not ended_by_stop) and (not success) and done),
        "steps": len(rows), "min_distance_m": min_distance,
        "start_goal_distance_m": spec["start_goal_distance_m"],
    }
    return episode, rows, stop_events


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=DATASETS / "train_600")
    p.add_argument("--checkpoint", type=Path, default=M0_CKPT)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--threshold", type=float, default=ARRIVAL_STOP_THRESHOLD)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    if "final_test" in str(args.dataset_root):
        raise SystemExit("Final Test is sealed; rollout collection is Train-only")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    specs = build_specs(args.dataset_root, args.max_episodes)
    if args.smoke:
        # Two episodes per band, in dataset order.
        counts = {band: 0 for band in BAND_ID}
        smoke = []
        for spec in specs:
            if counts[spec["band"]] < 2:
                smoke.append(spec)
                counts[spec["band"]] += 1
        specs = smoke
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    model = SpatialAwareMEC(build_gbs_model("Transformer-M"), "M0").to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    encoder = load_encoder(device)
    episodes, all_rows, all_stops = [], [], []
    for spec in specs:
        episode, rows, stops = run_episode(model, spec, encoder, device, args.threshold)
        episodes.append(episode)
        all_rows.extend(rows)
        all_stops.extend(stops)
        if (spec["episode_index"] + 1) % 50 == 0:
            print(f"processed {spec['episode_index'] + 1}/{len(specs)}", flush=True)
    features = np.stack([r["features"] for r in all_rows]).astype(np.float32)
    metadata = {
        key: np.asarray([r[key] for r in all_rows])
        for key in (
            "episode_index", "seed", "band_id", "step", "distance_m",
            "goal_visible", "arrival_probability", "arrival_logit", "stop", "action",
        )
    }
    np.savez_compressed(
        args.output_dir / "rollout_steps.npz", features=features, **metadata
    )
    _write_csv(args.output_dir / "episode_summary.csv", episodes)
    _write_csv(args.output_dir / "stop_events.csv", all_stops)
    distance = metadata["distance_m"]
    probability = metadata["arrival_probability"]
    far = distance > 40.0
    label = distance < ENV.stop_radius_m
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "split": args.dataset_root.name,
        "episodes": len(episodes),
        "steps": len(all_rows),
        "threshold": args.threshold,
        "sr": float(np.mean([e["success"] for e in episodes])),
        "osr": float(np.mean([e["osr"] for e in episodes])),
        "premature_stop": float(np.mean([e["premature_stop"] for e in episodes])),
        "timeout": float(np.mean([e["timeout"] for e in episodes])),
        "positives_near20": int(label.sum()),
        "far_negatives_gt40": int((far & ~label).sum()),
        "far_negatives_gt40_high_p": int((far & ~label & (probability >= args.threshold)).sum()),
        "far_negatives_gt40_high_p_rate": (
            float((probability[far & ~label] >= args.threshold).mean())
            if (far & ~label).any() else 0.0
        ),
        "stops": len(all_stops),
        "stops_gt40": sum(e["distance_m"] > 40.0 for e in all_stops),
        "stops_gt100": sum(e["distance_m"] > 100.0 for e in all_stops),
        "note": (
            "True distance / Goal visibility are labels and offline diagnostics only; "
            "they are never model inputs. Final Test is not read."
        ),
    }
    (args.output_dir / "collection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
