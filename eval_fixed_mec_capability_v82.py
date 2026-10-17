"""
V8.2 fixed single-MEC capability evaluation.

Protocol:
  - one model is used for the whole episode;
  - no association/switching;
  - all five models use exactly the same 60 test episode seeds;
  - each episode uses one fixed random 20-MEC layout;
  - network state is derived from current UAV position + that layout;
  - node model type and compute capacity are ignored by the network-state function;
  - no simulator distance is passed to the MEC;
  - no DQN or association metric is computed here.

The purpose is to answer only:
  "Does the redesigned MEC input make the heterogeneous MEC backends
   better at closed-loop navigation?"
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    ARRIVAL_STOP_THRESHOLD,
    AUX,
    ENCODER,
    ENV,
    GBS_MODEL_CONFIGS,
    NETWORK,
    TRAIN,
)
from env import ACTION_NAMES, STOP, ToyUAVEnv
from gbs_model_v3 import build_gbs_model
from gbs_network import build_random_gbs_layout, global_network_state
from navigation_reward import ndtw_score, remove_consecutive_duplicates
from token_data import (
    encode_multiscale_goal_tokens,
    goal_conditioned_spatial_heatmap,
    load_frozen_encoder,
)
from utils import image_to_tensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path("gbs_v82_input_redesign_results")
CHECKPOINT_SUFFIX = "best"
TEST_STATS_PATH = Path("token_dataset_test/dataset_stats.json")

NUM_EPISODES = None
MAX_STEPS = ENV.max_steps
PLOT_FIRST_N = 10

OUTPUT_DIR = Path("fixed_mec_capability_v82_results")
TRAJECTORY_DIR = OUTPUT_DIR / "trajectories"


def safe_name(name):
    return str(name).lower().replace("-", "_").replace(" ", "_")


def checkpoint_path(model_name):
    return CHECKPOINT_DIR / f"gbs_{safe_name(model_name)}_{CHECKPOINT_SUFFIX}.pt"


def find_available_models():
    available = []
    for name in GBS_MODEL_CONFIGS:
        path = checkpoint_path(name)
        if path.exists():
            available.append(name)
        else:
            print(f"[Skip] {name}: checkpoint not found -> {path}")
    return available


def load_model(model_name):
    model = build_gbs_model(
        model_name,
        token_dim=ENCODER.token_dim,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(
            checkpoint_path(model_name),
            map_location=DEVICE,
            weights_only=True,
        )
    )
    model.eval()
    return model


def trajectory_band_map():
    out = {}
    for band_name, _weight, dmin, dmax, amin, amax in TRAIN.trajectory_mix:
        out[str(band_name)] = (float(dmin), float(dmax), int(amin), int(amax))
    return out


def load_test_episode_specs():
    if not TEST_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TEST_STATS_PATH}. Run generate_test_token_dataset.py first."
        )

    with TEST_STATS_PATH.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    if stats.get("version") != "v8.2-input-redesign":
        raise ValueError(
            "The test dataset is not a V8.2 input-redesign dataset. "
            "Regenerate token_dataset_test."
        )

    records = list(stats.get("trajectories", []))
    if not records:
        raise ValueError(f"No trajectory records found in {TEST_STATS_PATH}")

    if NUM_EPISODES is not None:
        records = records[: int(NUM_EPISODES)]

    band_cfg = trajectory_band_map()
    specs = []
    for episode_index, record in enumerate(records):
        band = str(record["band"])
        if band not in band_cfg:
            raise KeyError(f"Unknown trajectory band in stats: {band}")
        dmin, dmax, amin, amax = band_cfg[band]
        if "network_layout_seed" not in record:
            raise KeyError(
                f"Episode {episode_index} has no network_layout_seed; "
                "regenerate V8.2 test data."
            )
        specs.append(
            {
                "episode_index": int(episode_index),
                "seed": int(record["seed"]),
                "band": band,
                "distance_range": (dmin, dmax),
                "oracle_dataset_steps": int(record["steps"]),
                "dataset_start_goal_distance_m": float(
                    record["start_goal_distance_m"]
                ),
                "band_action_range": (amin, amax),
                "network_layout_seed": int(record["network_layout_seed"]),
            }
        )
    return specs


def build_oracle_reference(spec):
    env = ToyUAVEnv()
    env.reset(
        seed=spec["seed"],
        goal_distance_range_m=spec["distance_range"],
        render_observation=False,
    )

    start = env.position.copy()
    goal = env.goal.copy()
    reference = [start.copy()]
    oracle_actions = Counter()
    success = False
    oracle_steps = 0

    for _ in range(MAX_STEPS):
        action = int(env.oracle_action())
        oracle_actions[ACTION_NAMES[action]] += 1
        oracle_steps += 1

        _, success, done = env.step(
            action,
            render_observation=False,
        )
        if action != STOP and np.linalg.norm(
            env.position - reference[-1]
        ) > 1e-9:
            reference.append(env.position.copy())
        if done:
            break

    reference = remove_consecutive_duplicates(
        np.asarray(reference, dtype=np.float32)
    )
    if np.linalg.norm(goal - reference[-1]) > 1e-9:
        reference = np.concatenate(
            [reference, goal.reshape(1, 2)],
            axis=0,
        )

    return {
        "start": start,
        "goal": goal,
        "reference": reference.astype(np.float32),
        "oracle_success": bool(success),
        "oracle_steps": int(oracle_steps),
        "oracle_actions": dict(oracle_actions),
        "shortest_distance_m": float(np.linalg.norm(start - goal)),
    }


@torch.no_grad()
def rollout_fixed_model(spec, oracle_info, encoder, model):
    env = ToyUAVEnv()
    obs = env.reset(
        seed=spec["seed"],
        goal_distance_range_m=spec["distance_range"],
        render_observation=True,
    )

    reproduced_distance = float(
        np.linalg.norm(env.position - env.goal)
    )
    if abs(
        reproduced_distance - spec["dataset_start_goal_distance_m"]
    ) > 1e-3:
        raise RuntimeError(
            "Test episode geometry mismatch: "
            f"dataset={spec['dataset_start_goal_distance_m']:.6f}, "
            f"reproduced={reproduced_distance:.6f}, "
            f"seed={spec['seed']}, band={spec['band']}"
        )

    network_nodes = build_random_gbs_layout(
        seed=spec["network_layout_seed"], verbose=False
    )
    goal_tokens = encode_multiscale_goal_tokens(
        env.get_goal_reference_multiscale_rgbs(),
        encoder,
        DEVICE,
    ).unsqueeze(0).to(DEVICE, dtype=torch.float32)

    trajectory = [env.position.copy()]
    action_counter = Counter()

    min_distance = float(env.distance_to_goal)
    osr = bool(env.distance_to_goal < env.stop_radius)
    success = False
    done = False
    ended_by_stop = False
    goal_visible_steps = 0

    visible_count = 0
    invisible_count = 0
    visible_arrival_sum = 0.0
    invisible_arrival_sum = 0.0
    invisible_gate_stop_count = 0
    goal_visible_at_stop = False

    max_stop_probability = 0.0
    stop_probability_at_min_distance = 0.0
    best_distance_seen = float(env.distance_to_goal)

    for step in range(MAX_STEPS):
        current_tokens = encoder(image_to_tensor(obs, DEVICE))
        heatmap = goal_conditioned_spatial_heatmap(
            current_tokens,
            goal_tokens,
        )
        network_state = torch.from_numpy(
            global_network_state(
                network_nodes,
                env.position,
                num_tokens=ENCODER.num_tokens,
                token_dim=ENCODER.token_dim,
            )
        ).unsqueeze(0).to(DEVICE, dtype=torch.float32)

        decision = model.predict_action(
            current_tokens,
            goal_tokens,
            heatmap,
            network_state,
        )

        action = int(decision["action"].item())
        stop_probability = float(
            decision["arrival_probability"].item()
        )
        action_counter[ACTION_NAMES[action]] += 1

        max_stop_probability = max(
            max_stop_probability,
            stop_probability,
        )
        if env.distance_to_goal <= best_distance_seen:
            best_distance_seen = float(env.distance_to_goal)
            stop_probability_at_min_distance = stop_probability

        goal_visible = env.goal_projection() is not None
        goal_visible_steps += int(goal_visible)

        if goal_visible:
            visible_count += 1
            visible_arrival_sum += stop_probability
        else:
            invisible_count += 1
            invisible_arrival_sum += stop_probability
            invisible_gate_stop_count += int(
                stop_probability >= float(ARRIVAL_STOP_THRESHOLD)
            )

        if action == STOP:
            goal_visible_at_stop = bool(goal_visible)

        ended_by_stop = bool(action == STOP)
        obs, success, done = env.step(
            action,
            render_observation=(action != STOP),
        )

        trajectory.append(env.position.copy())
        min_distance = min(
            min_distance,
            float(env.distance_to_goal),
        )
        osr = bool(
            osr or env.distance_to_goal < env.stop_radius
        )

        if done:
            break

    trajectory = np.asarray(trajectory, dtype=np.float32)
    num_steps = int(len(trajectory) - 1)

    if len(trajectory) > 1:
        path_length = float(
            np.linalg.norm(
                np.diff(trajectory[:, :2], axis=0),
                axis=1,
            ).sum()
        )
    else:
        path_length = 0.0

    shortest_distance = float(oracle_info["shortest_distance_m"])
    spl = (
        shortest_distance
        / max(shortest_distance, path_length, 1e-8)
        if success
        else 0.0
    )

    ndtw = ndtw_score(
        trajectory=trajectory[:, :2],
        reference=oracle_info["reference"],
        distance_threshold=env.stop_radius,
    )
    sdtw = ndtw if success else 0.0

    premature_stop = bool(ended_by_stop and not success)
    max_steps_termination = bool(
        (not ended_by_stop) and (not success) and done
    )

    total_actions = sum(action_counter.values())
    stop_count = int(action_counter.get("Stop", 0))

    return {
        "success": bool(success),
        "osr": bool(osr),
        "premature_stop": premature_stop,
        "max_steps_termination": max_steps_termination,
        "steps": num_steps,
        "min_distance_m": float(min_distance),
        "final_distance_m": float(env.distance_to_goal),
        "path_length_m": path_length,
        "spl": float(spl),
        "ndtw": float(ndtw),
        "sdtw": float(sdtw),
        "stop_count": stop_count,
        "stop_ratio": float(stop_count / max(total_actions, 1)),
        "goal_visible_ratio": float(
            goal_visible_steps / max(num_steps, 1)
        ),
        "goal_visible_at_stop": bool(goal_visible_at_stop),
        "mean_arrival_p_goal_visible": float(
            visible_arrival_sum / max(visible_count, 1)
        ),
        "mean_arrival_p_goal_invisible": float(
            invisible_arrival_sum / max(invisible_count, 1)
        ),
        "invisible_gate_stop_count": int(
            invisible_gate_stop_count
        ),
        "max_stop_probability": float(max_stop_probability),
        "stop_probability_at_min_distance": float(
            stop_probability_at_min_distance
        ),
        "actions": dict(action_counter),
        "trajectory": trajectory,
    }


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_episode(
    spec,
    oracle_info,
    model_trajectories,
    model_results,
):
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(
        oracle_info["start"][0],
        oracle_info["start"][1],
        marker="P",
        s=150,
        label="UAV Start",
        zorder=20,
    )
    ax.scatter(
        oracle_info["goal"][0],
        oracle_info["goal"][1],
        marker="*",
        s=260,
        label="Goal",
        zorder=20,
    )
    ref = oracle_info["reference"]
    ax.plot(
        ref[:, 0],
        ref[:, 1],
        linestyle="--",
        linewidth=2.0,
        label="Oracle Reference",
        zorder=10,
    )

    for name in model_trajectories:
        tr = model_trajectories[name]
        ax.plot(
            tr[:, 0],
            tr[:, 1],
            linewidth=1.8,
            label=name,
        )

    ax.set_xlim(0, ENV.world_size_m)
    ax.set_ylim(0, ENV.world_size_m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"V8.2 Fixed-MEC Closed-loop Trajectories — "
        f"Episode {spec['episode_index']}"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        TRAJECTORY_DIR
        / f"episode_{spec['episode_index']:03d}_trajectories.png",
        dpi=220,
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("V8.2 FIXED SINGLE-MEC CAPABILITY EVALUATION")
    print("=" * 96)
    print("Inputs:")
    print("  Current RGB tokens : 576 x 384")
    print("  Goal RGB tokens    : 3 x 384")
    print("  Spatial heatmap    : 3 x 8 x 8")
    print("  Network state      : 5D global")
    print("Removed:")
    print("  UAV position / Start mask / History / Depth / MEC capability / success hint")
    print(
        "Network state ignores node model_type and compute capacity."
    )
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Test stats    : {TEST_STATS_PATH}")

    models = find_available_models()
    if not models:
        raise FileNotFoundError(
            "No V8.2 MEC checkpoints found. Run train_gbs_v3.py first."
        )

    specs = load_test_episode_specs()
    print(f"Evaluation episodes: {len(specs)}")
    print("Available models:", models)

    encoder = load_frozen_encoder(DEVICE)

    oracle_cache = {}
    for spec in specs:
        oracle_cache[spec["episode_index"]] = build_oracle_reference(spec)

    episode_rows = []
    summary_rows = []
    all_trajectories = {}

    for model_name in models:
        print("\n" + "=" * 96)
        print(f"Evaluating fixed MEC: {model_name}")
        print("=" * 96)

        model = load_model(model_name)
        rows = []
        trajectories = {}

        for idx, spec in enumerate(specs, start=1):
            result = rollout_fixed_model(
                spec,
                oracle_cache[spec["episode_index"]],
                encoder,
                model,
            )
            trajectories[spec["episode_index"]] = result["trajectory"]

            rows.append(
                {
                    "model": model_name,
                    "episode_index": spec["episode_index"],
                    "seed": spec["seed"],
                    "band": spec["band"],
                    "network_layout_seed": spec["network_layout_seed"],
                    "success": int(result["success"]),
                    "osr": int(result["osr"]),
                    "premature_stop": int(result["premature_stop"]),
                    "steps": result["steps"],
                    "min_distance_m": result["min_distance_m"],
                    "final_distance_m": result["final_distance_m"],
                    "path_length_m": result["path_length_m"],
                    "spl": result["spl"],
                    "ndtw": result["ndtw"],
                    "sdtw": result["sdtw"],
                    "goal_visible_ratio": result["goal_visible_ratio"],
                    "goal_visible_at_stop": int(result["goal_visible_at_stop"]),
                    "mean_arrival_p_goal_visible": result[
                        "mean_arrival_p_goal_visible"
                    ],
                    "mean_arrival_p_goal_invisible": result[
                        "mean_arrival_p_goal_invisible"
                    ],
                    "invisible_gate_stop_count": result[
                        "invisible_gate_stop_count"
                    ],
                    "max_stop_probability": result[
                        "max_stop_probability"
                    ],
                    "stop_probability_at_min_distance": result[
                        "stop_probability_at_min_distance"
                    ],
                }
            )

            if idx % 10 == 0 or idx == len(specs):
                print(
                    f"  [{idx:02d}/{len(specs)}] "
                    f"SR={100*result['success']:5.1f}% "
                    f"OSR={100*result['osr']:5.1f}% "
                    f"premStop={int(result['premature_stop'])} "
                    f"nDTW={100*result['ndtw']:5.1f}"
                )

        all_trajectories[model_name] = trajectories

        successes = np.asarray(
            [r["success"] for r in rows], dtype=np.float64
        )
        osrs = np.asarray(
            [r["osr"] for r in rows], dtype=np.float64
        )
        prem = np.asarray(
            [r["premature_stop"] for r in rows], dtype=np.float64
        )

        summary = {
            "model": model_name,
            "params": sum(
                p.numel() for p in model.parameters()
            ),
            "episodes": len(rows),
            "success_rate": float(successes.mean()),
            "osr": float(osrs.mean()),
            "osr_minus_sr": float(osrs.mean() - successes.mean()),
            "premature_stop_rate": float(prem.mean()),
            "mean_spl": float(
                np.mean([r["spl"] for r in rows])
            ),
            "mean_ndtw": float(
                np.mean([r["ndtw"] for r in rows])
            ),
            "mean_sdtw": float(
                np.mean([r["sdtw"] for r in rows])
            ),
            "mean_min_distance_m": float(
                np.mean([r["min_distance_m"] for r in rows])
            ),
            "mean_steps": float(
                np.mean([r["steps"] for r in rows])
            ),
            "mean_goal_visible_ratio": float(
                np.mean([r["goal_visible_ratio"] for r in rows])
            ),
            "mean_stop_probability_at_min_distance": float(
                np.mean(
                    [
                        r["stop_probability_at_min_distance"]
                        for r in rows
                    ]
                )
            ),
        }
        summary_rows.append(summary)

        episode_rows.extend(rows)
        save_csv(
            OUTPUT_DIR
            / f"{safe_name(model_name)}_episode_results.csv",
            rows,
        )

        print(
            f"{model_name:18s} "
            f"params={summary['params']:>10,d} "
            f"SR={100*summary['success_rate']:6.2f}% "
            f"OSR={100*summary['osr']:6.2f}% "
            f"OSR-SR={100*summary['osr_minus_sr']:6.2f}% "
            f"PremStop={100*summary['premature_stop_rate']:6.2f}% "
            f"SPL={summary['mean_spl']:.3f} "
            f"nDTW={summary['mean_ndtw']:.3f}"
        )

    save_csv(
        OUTPUT_DIR / "fixed_mec_episode_results_all.csv",
        episode_rows,
    )
    save_csv(
        OUTPUT_DIR / "fixed_mec_summary.csv",
        summary_rows,
    )
    with (OUTPUT_DIR / "fixed_mec_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    plot_count = min(PLOT_FIRST_N, len(specs))
    for i in range(plot_count):
        spec = specs[i]
        episode_id = spec["episode_index"]
        plot_episode(
            spec,
            oracle_cache[episode_id],
            {
                name: all_trajectories[name][episode_id]
                for name in models
            },
            None,
        )

    print("\n" + "=" * 96)
    print("V8.2 FIXED-MEC SUMMARY")
    print("=" * 96)
    print(
        f"{'Model':18s} {'Params':>10s} {'SR':>8s} "
        f"{'OSR':>8s} {'OSR-SR':>9s} {'PremStop':>10s} "
        f"{'SPL':>8s} {'nDTW':>8s}"
    )
    print("-" * 96)
    for s in summary_rows:
        print(
            f"{s['model']:18s} "
            f"{s['params']:10,d} "
            f"{100*s['success_rate']:7.2f}% "
            f"{100*s['osr']:7.2f}% "
            f"{100*s['osr_minus_sr']:8.2f}% "
            f"{100*s['premature_stop_rate']:9.2f}% "
            f"{s['mean_spl']:8.3f} "
            f"{s['mean_ndtw']:8.3f}"
        )

    print("\nSaved:")
    print(" ", OUTPUT_DIR / "fixed_mec_summary.csv")
    print(" ", OUTPUT_DIR / "fixed_mec_summary.json")
    print(" ", OUTPUT_DIR / "fixed_mec_episode_results_all.csv")
    print(" ", TRAJECTORY_DIR)


if __name__ == "__main__":
    main()
