import csv
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from association_strategies import get_backend_output, select_gbs
from config import AUX, ENCODER, ENV, GBS_MODEL_CONFIGS, NETWORK, NUM_ACTIONS
from env import STOP, ToyUAVEnv
from gbs_model_v3 import build_gbs_model
from gbs_network import build_random_gbs_layout, get_available_gbs, get_nearest_gbs
from latency_model import LatencyModel
from navigation_reward import NavigationRewardTracker, remove_consecutive_duplicates
from token_data import (
    compress_visual_tokens,
    depth_map_to_tokens,
    encode_multiscale_goal_tokens,
    load_frozen_encoder,
)
from utils import image_to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOMIZE_EACH_RUN = False
FIXED_EPISODE_SEED = 70001
FIXED_GBS_LAYOUT_SEED = 80001
FIXED_EVAL_SEED_START = 90000
NUM_EVAL_EPISODES = 50
OUT_DIR = "gbs_v8_arrival_reasoning_results"

STRATEGIES = ["random", "nearest", "max_rate", "min_latency", "best_quality", "oracle"]
STRATEGY_NAMES = {
    "random": "Random",
    "nearest": "Nearest GBS",
    "max_rate": "Max-Rate",
    "min_latency": "Min-Latency",
    "best_quality": "Best-Quality Model",
    "oracle": "Candidate Oracle",
}
STRATEGY_COLORS = {
    "random": "tab:gray",
    "nearest": "tab:blue",
    "max_rate": "tab:cyan",
    "min_latency": "tab:green",
    "best_quality": "tab:orange",
    "oracle": "tab:purple",
}

LATENCY_MODEL = LatencyModel()
FIXED_TX_POWER_W = NETWORK.default_tx_power_w


def safe_name(name):
    return name.lower().replace("-", "_").replace(" ", "_")


def load_backend_models():
    models = {}
    for name in GBS_MODEL_CONFIGS.keys():
        model = build_gbs_model(name, token_dim=ENCODER.token_dim, actions=NUM_ACTIONS).to(DEVICE)
        ckpt = f"{OUT_DIR}/gbs_{safe_name(name)}_best.pt"
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.eval()
        models[name] = model
    return models


def load_quality_scores():
    """Best-quality关联优先使用包含Stop的Balanced4，而不是只看Directional BA。"""
    path = f"{OUT_DIR}/gbs_validation_scores_v8.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    scores = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            scores[name] = float(
                value.get("macro_f1", value.get("balanced_accuracy", value.get("accuracy", 0.0)))
            )
        else:
            scores[name] = float(value)
    return scores


def pack_online_history(history_frames):
    k = AUX.history_length
    h = AUX.history_tokens_per_frame
    d = ENCODER.token_dim
    out = torch.zeros((1, k, h, d), dtype=torch.float32, device=DEVICE)
    valid = torch.zeros((1, k), dtype=torch.long, device=DEVICE)
    selected = history_frames[-k:]
    if selected:
        n = len(selected)
        out[0, :n] = torch.stack(selected, dim=0).to(DEVICE, dtype=torch.float32)
        valid[0, :n] = 1
    return out, valid


def build_oracle_reference_trajectory(seed):
    env = ToyUAVEnv()
    env.reset(seed=seed)
    reference = [env.position.copy()]
    for _ in range(env.max_steps):
        action = env.oracle_action()
        if action == STOP:
            break
        _, _, done = env.step(action)
        p = env.position.copy()
        if np.linalg.norm(p - reference[-1]) > 1e-9:
            reference.append(p)
        if done:
            break
    if np.linalg.norm(env.goal - reference[-1]) > 1e-9:
        reference.append(env.goal.copy())
    return remove_consecutive_duplicates(reference).astype(np.float32)


def rollout_strategy(
    seed,
    strategy,
    encoder,
    backend_models,
    gbs_nodes,
    reference_trajectory,
    model_quality_scores,
):
    env = ToyUAVEnv()
    obs = env.reset(seed=seed)
    start = env.position.copy()
    goal = env.goal.copy()

    start_mask = torch.from_numpy(env.get_start_reference_mask()).unsqueeze(0).to(
        DEVICE, dtype=torch.float32
    )
    goal_tokens = encode_multiscale_goal_tokens(
        env.get_goal_reference_multiscale_rgbs(), encoder, DEVICE
    ).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    history_frames = []

    reward_tracker = NavigationRewardTracker(
        reference_trajectory=reference_trajectory,
        start_position=start,
        distance_threshold=env.stop_radius,
        max_forward_progress=None,
    )

    rng_offsets = {
        "random": 101,
        "nearest": 211,
        "max_rate": 307,
        "min_latency": 401,
        "best_quality": 503,
        "oracle": 607,
    }
    rng = np.random.default_rng(seed + rng_offsets[strategy])

    poses = [env.pose.copy()]
    actions = []
    selected_ids = []
    selected_models = []
    selected_distances = []
    selected_rates = []
    latency_metrics = []
    nav_rewards = []
    latency_rewards = []
    joint_rewards = []
    correct_actions = 0
    uncovered_steps = 0
    osr = env.distance_to_goal < env.stop_radius

    done = False
    success = False
    while not done:
        with torch.no_grad():
            tokens = encoder(image_to_tensor(obs, DEVICE))
        depth_tokens = depth_map_to_tokens(env.render_depth()).unsqueeze(0).to(
            DEVICE, dtype=torch.float32
        )

        uav_position = env.position.copy()
        candidates = get_available_gbs(gbs_nodes, uav_position)
        if not candidates:
            uncovered_steps += 1
            candidates = [get_nearest_gbs(gbs_nodes, uav_position)]

        oracle_action = env.oracle_action()
        inference_cache = {}
        hist_tokens, hist_valid = pack_online_history(history_frames)
        for model in backend_models.values():
            model.set_runtime_aux(
                start_mask=start_mask,
                goal_tokens=goal_tokens,
                history_tokens=hist_tokens,
                history_valid=hist_valid,
                depth_tokens=depth_tokens,
            )

        node = select_gbs(
            strategy=strategy,
            candidates=candidates,
            uav_position=uav_position,
            tokens=tokens,
            backend_models=backend_models,
            inference_cache=inference_cache,
            rng=rng,
            oracle_action=oracle_action,
            latency_model=LATENCY_MODEL,
            tx_power_w=FIXED_TX_POWER_W,
            num_actions=NUM_ACTIONS,
            model_quality_scores=model_quality_scores,
        )
        output = get_backend_output(node.model_type, tokens, backend_models, inference_cache)
        action = int(output["action"])
        if action == oracle_action:
            correct_actions += 1

        horizontal_distance_m = node.distance_to(uav_position)
        lm = LATENCY_MODEL.evaluate_step_from_distance(
            tokens=tokens,
            model_type=node.model_type,
            horizontal_distance_m=horizontal_distance_m,
            compute_capacity_flops_per_s=node.compute_capacity_flops_per_s,
            tx_power_w=FIXED_TX_POWER_W,
            num_actions=NUM_ACTIONS,
        )

        actions.append(action)
        selected_ids.append(node.gbs_id)
        selected_models.append(node.model_type)
        selected_distances.append(horizontal_distance_m)
        selected_rates.append(lm["uplink_rate_mbps"])
        latency_metrics.append(lm)
        latency_rewards.append(float(lm["latency_reward"]))

        # 当前visual token已上传，本步后直接压缩并缓存在MEC侧。
        history_frames.append(compress_visual_tokens(tokens.squeeze(0)).detach().cpu())
        history_frames = history_frames[-AUX.history_length:]

        obs, success, done = env.step(action)
        osr = bool(osr or env.distance_to_goal < env.stop_radius)
        poses.append(env.pose.copy())

        nav_reward, _ = reward_tracker.advance(new_position=env.position.copy(), done=done)
        nav_rewards.append(float(nav_reward))
        joint_rewards.append(float(nav_reward + lm["latency_reward"]))

    poses = np.asarray(poses, dtype=np.float32)
    trajectory = poses[:, :2]
    reward_summary = reward_tracker.summary()

    path_length = 0.0
    if len(trajectory) > 1:
        path_length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
    shortest_distance = float(np.linalg.norm(start - goal))
    spl = shortest_distance / max(shortest_distance, path_length, 1e-8) if success else 0.0
    ndtw = float(reward_summary["final_ndtw"])
    sdtw = ndtw if success else 0.0

    total_ms = np.asarray([x["total_latency_ms"] for x in latency_metrics], dtype=float)
    ext_ms = np.asarray([x["extraction_latency_ms"] for x in latency_metrics], dtype=float)
    com_ms = np.asarray([x["transmission_latency_ms"] for x in latency_metrics], dtype=float)
    rem_ms = np.asarray([x["inference_latency_ms"] for x in latency_metrics], dtype=float)
    violations = np.asarray([x["deadline_violation"] for x in latency_metrics], dtype=float)

    return {
        "strategy": strategy,
        "success": bool(success),
        "osr": bool(osr),
        "ndtw": ndtw,
        "sdtw": sdtw,
        "spl": float(spl),
        "final_error_m": float(np.linalg.norm(trajectory[-1] - goal)),
        "path_length_m": path_length,
        "steps": len(actions),
        "decision_accuracy": correct_actions / max(len(actions), 1),
        "uncovered_steps": uncovered_steps,
        "trajectory": trajectory,
        "poses": poses,
        "start": start,
        "goal": goal,
        "reference_trajectory": reference_trajectory,
        "selected_gbs_ids": selected_ids,
        "selected_models": selected_models,
        "selected_distances_m": selected_distances,
        "selected_rates_mbps": selected_rates,
        "model_usage": dict(Counter(selected_models)),
        "mean_ext_ms": float(ext_ms.mean()),
        "mean_com_ms": float(com_ms.mean()),
        "mean_rem_ms": float(rem_ms.mean()),
        "mean_total_ms": float(total_ms.mean()),
        "p95_total_ms": float(np.percentile(total_ms, 95)),
        "delay_violation_rate": float(violations.mean()),
        "latency_return": float(np.sum(latency_rewards)),
        "navigation_return": float(np.sum(nav_rewards)),
        "joint_return": float(np.sum(joint_rewards)),
        "step_total_latency_ms": total_ms,
    }


def print_single_episode(results):
    print("\n" + "=" * 116)
    header = (
        f"{'Strategy':<20} {'SR':>4} {'OSR':>4} {'nDTW':>7} {'sDTW':>7} {'SPL':>7} "
        f"{'Steps':>6} {'E2E(ms)':>9} {'P95(ms)':>9} {'Viol%':>7}"
    )
    print(header)
    print("-" * len(header))
    for strategy in STRATEGIES:
        r = results[strategy]
        print(
            f"{STRATEGY_NAMES[strategy]:<20} {int(r['success']):>4d} {int(r['osr']):>4d} "
            f"{100*r['ndtw']:>7.2f} {100*r['sdtw']:>7.2f} {r['spl']:>7.3f} "
            f"{r['steps']:>6d} {r['mean_total_ms']:>9.2f} {r['p95_total_ms']:>9.2f} "
            f"{100*r['delay_violation_rate']:>7.2f}"
        )


def plot_single_episode(results, gbs_nodes):
    ref = results["nearest"]
    fig, ax = plt.subplots(figsize=(10, 9))
    for node in gbs_nodes:
        ax.scatter(node.x, node.y, marker="^", s=60, alpha=0.6)
        ax.text(node.x + 10, node.y + 10, f"{node.gbs_id}/{node.model_type}", fontsize=6)
    reference = ref["reference_trajectory"]
    ax.plot(reference[:, 0], reference[:, 1], "k--", linewidth=2, label="Oracle reference")
    ax.scatter(ref["start"][0], ref["start"][1], marker="P", s=160, label="Start")
    ax.scatter(ref["goal"][0], ref["goal"][1], marker="*", s=240, label="Goal")
    for strategy in STRATEGIES:
        tr = results[strategy]["trajectory"]
        ax.plot(tr[:, 0], tr[:, 1], linewidth=1.7, color=STRATEGY_COLORS[strategy], label=STRATEGY_NAMES[strategy])
    ax.set_xlim(0, ENV.world_size_m)
    ax.set_ylim(0, ENV.world_size_m)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("V8 arrival-aware closed-loop UAV trajectories")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("association_trajectories_v8.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy in STRATEGIES:
        ax.plot(results[strategy]["step_total_latency_ms"], label=STRATEGY_NAMES[strategy], color=STRATEGY_COLORS[strategy])
    ax.axhline(NETWORK.decision_latency_budget_s * 1000.0, linestyle="--", color="black", label="100 ms deadline")
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Analytical E2E latency (ms)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("association_latency_timeline_v8.png", dpi=250)
    plt.close(fig)


def evaluate_many(encoder, backend_models, gbs_nodes, quality_scores, seed_start):
    all_results = {s: [] for s in STRATEGIES}
    for i in range(NUM_EVAL_EPISODES):
        seed = seed_start + i
        reference = build_oracle_reference_trajectory(seed)
        for strategy in STRATEGIES:
            all_results[strategy].append(
                rollout_strategy(seed, strategy, encoder, backend_models, gbs_nodes, reference, quality_scores)
            )
        print(f"Finished episode {i + 1}/{NUM_EVAL_EPISODES}")
    return all_results


def summarize(all_results):
    rows = []
    print("\n" + "=" * 124)
    print("Batch summary")
    print("=" * 124)
    for strategy in STRATEGIES:
        rs = all_results[strategy]
        sr = float(np.mean([r["success"] for r in rs]))
        osr = float(np.mean([r["osr"] for r in rs]))
        row = {
            "strategy": STRATEGY_NAMES[strategy],
            "success_rate": sr,
            "oracle_success_rate": osr,
            "osr_minus_sr": osr - sr,
            "mean_ndtw": float(np.mean([r["ndtw"] for r in rs])),
            "mean_sdtw": float(np.mean([r["sdtw"] for r in rs])),
            "mean_spl": float(np.mean([r["spl"] for r in rs])),
            "mean_steps": float(np.mean([r["steps"] for r in rs])),
            "mean_e2e_ms": float(np.mean([r["mean_total_ms"] for r in rs])),
            "mean_p95_ms": float(np.mean([r["p95_total_ms"] for r in rs])),
            "delay_violation_rate": float(np.mean([r["delay_violation_rate"] for r in rs])),
            "mean_joint_return": float(np.mean([r["joint_return"] for r in rs])),
        }
        rows.append(row)
        print(
            f"{row['strategy']:<20} SR={100*sr:6.2f} OSR={100*osr:6.2f} "
            f"Gap={100*(osr-sr):6.2f} nDTW={100*row['mean_ndtw']:6.2f} "
            f"SPL={row['mean_spl']:.3f} E2E={row['mean_e2e_ms']:.2f}ms"
        )

    with open("association_summary_v8.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("Device:", DEVICE)
    print("Latency configuration:", LATENCY_MODEL.describe())

    if RANDOMIZE_EACH_RUN:
        rng = np.random.default_rng()
        episode_seed = int(rng.integers(1, 2_000_000_000))
        layout_seed = int(rng.integers(1, 2_000_000_000))
        eval_seed_start = int(rng.integers(1, 1_000_000_000))
    else:
        episode_seed = FIXED_EPISODE_SEED
        layout_seed = FIXED_GBS_LAYOUT_SEED
        eval_seed_start = FIXED_EVAL_SEED_START

    encoder = load_frozen_encoder(DEVICE)
    backend_models = load_backend_models()
    quality_scores = load_quality_scores()
    gbs_nodes = build_random_gbs_layout(seed=layout_seed)

    print("\nGBS topology")
    for node in gbs_nodes:
        print(
            f"{node.gbs_id:>3s}: ({node.x:7.1f},{node.y:7.1f}) m | "
            f"model={node.model_type:<13s} | hw={node.hardware_tier} | "
            f"compute={node.compute_capacity_flops_per_s/1e12:.2f} TFLOPs"
        )

    reference = build_oracle_reference_trajectory(episode_seed)
    single = {
        strategy: rollout_strategy(
            episode_seed, strategy, encoder, backend_models, gbs_nodes, reference, quality_scores
        )
        for strategy in STRATEGIES
    }
    print_single_episode(single)
    plot_single_episode(single, gbs_nodes)

    batch = evaluate_many(encoder, backend_models, gbs_nodes, quality_scores, eval_seed_start)
    summarize(batch)


if __name__ == "__main__":
    main()
