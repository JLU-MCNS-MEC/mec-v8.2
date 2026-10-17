"""
V8.1 固定单一 MEC 推理能力评估
================================

目的
----
在整个 UAV episode 中始终使用同一种 MEC 推理模型，不进行 GBS/MEC 切换，
从而把“模型本身的导航推理能力”与“关联策略、网络覆盖、通信/计算时延”解耦。

该脚本会：
1. 自动检测已经训练完成的 V8 MEC checkpoint；
2. 从 token_dataset_test/dataset_stats.json 读取已经生成的测试 episode，
   保证所有模型使用完全相同的 seed、轨迹 band 和起终点距离范围；
3. 对每一种 MEC 单独进行闭环导航：整个 episode 内模型固定不变；
4. 输出 SR、OSR、OSR-SR、SPL、nDTW、sDTW、最小距离、错误 Stop 等指标；
5. 保存逐 episode CSV、模型汇总 CSV；
6. 对前若干个 episode 绘制不同 MEC 的飞行轨迹与 Oracle reference trajectory。

说明
----
- 本脚本故意不调用 association strategy，也不计算通信时延。
- 它回答的问题是：Small / Transformer-S / Medium / Transformer-M / Large
  在“独立承担整条导航任务”时，最终轨迹和成功率是否存在差异。
- 推荐先用 best.pt（V8 中按 Macro-F1 + Arrival-F1 保存）进行比较。
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import ARRIVAL_STOP_THRESHOLD, AUX, ENCODER, ENV, GBS_MODEL_CONFIGS, TRAIN
from env import ACTION_NAMES, STOP, ToyUAVEnv
from gbs_model_v3 import build_gbs_model
from navigation_reward import ndtw_score, remove_consecutive_duplicates
from token_data import (
    compress_visual_tokens,
    depth_map_to_tokens,
    encode_multiscale_goal_tokens,
    load_frozen_encoder,
)
from utils import image_to_tensor


# =============================================================================
# 用户可调参数
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# V8 MEC checkpoint 目录。
CHECKPOINT_DIR = Path("gbs_v8_arrival_reasoning_results")
CHECKPOINT_SUFFIX = "best"  # 默认使用 Macro-F1 最优 checkpoint。

# 使用生成好的 test dataset 中的轨迹配置，保证模型间公平比较。
TEST_STATS_PATH = Path(
    "/home/zty/tmp/pycharm_project_572/test_v8/token_dataset_test/dataset_stats.json"
)

# None = 使用 dataset_stats.json 中全部测试 episode（当前应为 60）。
# 调试时可以改成 10/20；最终实验建议 None。
NUM_EPISODES = None

# 最多允许的闭环步数。通常直接与环境 max_steps 一致。
MAX_STEPS = ENV.max_steps

# 为前 N 个测试 episode 画“不同固定 MEC 的轨迹叠加图”。
# 设为 0 可以关闭绘图。
PLOT_FIRST_N = 10

# 输出目录。
OUTPUT_DIR = Path("fixed_mec_capability_v81_results")
TRAJECTORY_DIR = OUTPUT_DIR / "trajectories"


# =============================================================================
# 基础工具
# =============================================================================


def safe_name(name):
    return str(name).lower().replace("-", "_").replace(" ", "_")


def checkpoint_path(model_name):
    return CHECKPOINT_DIR / f"gbs_{safe_name(model_name)}_{CHECKPOINT_SUFFIX}.pt"


def find_available_models():
    """自动检测已经训练完成的 MEC；未训练完成的模型直接跳过。"""
    available = []
    for name in GBS_MODEL_CONFIGS:
        path = checkpoint_path(name)
        if path.exists():
            available.append(name)
        else:
            print(f"[Skip] {name}: checkpoint not found -> {path}")
    return available


def load_model(model_name):
    model = build_gbs_model(model_name, token_dim=ENCODER.token_dim).to(DEVICE)
    model.load_state_dict(
        torch.load(checkpoint_path(model_name), map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model


def pack_online_history(history_frames):
    """把 MEC 侧最近 K 帧的 2x2 压缩视觉 token 打包为固定形状。"""
    k = AUX.history_length
    h = AUX.history_tokens_per_frame
    d = ENCODER.token_dim

    history = torch.zeros((1, k, h, d), dtype=torch.float32, device=DEVICE)
    valid = torch.zeros((1, k), dtype=torch.long, device=DEVICE)

    selected = history_frames[-k:]
    if selected:
        n = len(selected)
        history[0, :n] = torch.stack(selected, dim=0).to(
            DEVICE, dtype=torch.float32
        )
        valid[0, :n] = 1

    return history, valid


def trajectory_band_map():
    """band name -> (distance_min, distance_max, action_min, action_max)。"""
    result = {}
    for name, _, dmin, dmax, amin, amax in TRAIN.trajectory_mix:
        result[str(name)] = (
            float(dmin),
            float(dmax),
            int(amin),
            int(amax),
        )
    return result


def load_test_episode_specs():
    """
    从 token_dataset_test 的统计文件读取已接受测试轨迹。

    这样固定 MEC 闭环测试与离线 validation 使用的是同一批 episode seeds/bands，
    避免重新随机抽一批轨迹导致比较口径变化。
    """
    if not TEST_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TEST_STATS_PATH}. Run generate_test_token_dataset.py first."
        )

    with TEST_STATS_PATH.open("r", encoding="utf-8") as f:
        stats = json.load(f)

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
            }
        )

    return specs


# =============================================================================
# Oracle reference trajectory
# =============================================================================


def build_oracle_reference(spec):
    """
    在完全相同的 seed + distance band 下构造 Oracle reference trajectory。

    这里只用于评价，不作为模型输入。
    为避免不必要的 RGB 渲染，Oracle reference 构造过程关闭 render。
    """
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

        if action == STOP:
            _, success, done = env.step(action, render_observation=False)
            if not done:
                raise RuntimeError("Oracle Stop did not terminate the episode")
            break

        _, success, done = env.step(action, render_observation=False)
        if np.linalg.norm(env.position - reference[-1]) > 1e-9:
            reference.append(env.position.copy())
        if done:
            break

    reference = remove_consecutive_duplicates(np.asarray(reference, dtype=np.float32))

    # 与之前的 navigation evaluation 保持一致：reference 最后显式接到 goal 点。
    if np.linalg.norm(goal - reference[-1]) > 1e-9:
        reference = np.concatenate([reference, goal.reshape(1, 2)], axis=0)

    return {
        "start": start,
        "goal": goal,
        "reference": reference.astype(np.float32),
        "oracle_success": bool(success),
        "oracle_steps": int(oracle_steps),
        "oracle_actions": dict(oracle_actions),
        "shortest_distance_m": float(np.linalg.norm(start - goal)),
    }


# =============================================================================
# 单一固定 MEC 闭环 rollout
# =============================================================================


@torch.no_grad()
def rollout_fixed_model(spec, oracle_info, encoder, model):
    """
    整个 episode 从第一步到最后一步始终使用同一个 model。

    不进行：
      - GBS selection
      - MEC switching
      - network coverage filtering
      - latency-aware association

    因而结果只反映该模型本身的闭环导航推理能力。
    """
    env = ToyUAVEnv()
    obs = env.reset(
        seed=spec["seed"],
        goal_distance_range_m=spec["distance_range"],
        render_observation=True,
    )

    # 验证同一 seed/band 是否复现了 dataset 中相同的几何任务。
    reproduced_distance = float(np.linalg.norm(env.position - env.goal))
    if abs(reproduced_distance - spec["dataset_start_goal_distance_m"]) > 1e-3:
        raise RuntimeError(
            "Test episode geometry mismatch. "
            f"dataset={spec['dataset_start_goal_distance_m']:.6f}, "
            f"reproduced={reproduced_distance:.6f}, "
            f"seed={spec['seed']}, band={spec['band']}"
        )

    start_mask = torch.from_numpy(env.get_start_reference_mask()).unsqueeze(0).to(
        DEVICE, dtype=torch.float32
    )
    goal_tokens = encode_multiscale_goal_tokens(
        env.get_goal_reference_multiscale_rgbs(), encoder, DEVICE
    ).unsqueeze(0).to(DEVICE, dtype=torch.float32)

    history_frames = []
    trajectory = [env.position.copy()]
    action_counter = Counter()

    min_distance = float(env.distance_to_goal)
    osr = bool(env.distance_to_goal < env.stop_radius)
    success = False
    done = False
    ended_by_stop = False
    goal_visible_steps = 0

    # V8.1 diagnostics: simulator可见性仅用于离线评估，不进入模型输入。
    visible_count = 0
    invisible_count = 0
    visible_arrival_sum = 0.0
    invisible_arrival_sum = 0.0
    visible_top10_match_sum = 0.0
    invisible_top10_match_sum = 0.0
    invisible_gate_stop_count = 0
    goal_visible_at_stop = False

    # 用于观察 Stop 置信度随轨迹变化的简单统计。
    max_stop_probability = 0.0
    stop_probability_at_min_distance = 0.0
    best_distance_seen = float(env.distance_to_goal)

    for step in range(MAX_STEPS):
        current_tokens = encoder(image_to_tensor(obs, DEVICE))
        depth_tokens = depth_map_to_tokens(env.render_depth()).unsqueeze(0).to(
            DEVICE, dtype=torch.float32
        )
        history, history_valid = pack_online_history(history_frames)

        decision = model.predict_action(
            current_tokens,
            start_mask,
            goal_tokens,
            history,
            history_valid,
            depth_tokens,
        )
        probabilities = decision["joint_probabilities"]
        action = int(decision["action"].item())
        stop_probability = float(decision["arrival_probability"].item())

        max_stop_probability = max(max_stop_probability, stop_probability)
        if env.distance_to_goal <= best_distance_seen:
            best_distance_seen = float(env.distance_to_goal)
            stop_probability_at_min_distance = stop_probability

        # 当前真实Goal是否在相机视野：仅诊断使用，绝不输入模型。
        goal_visible = env.goal_projection() is not None
        goal_visible_steps += int(goal_visible)
        top10_match = float(decision["arrival_features"][0, 1].item())
        if goal_visible:
            visible_count += 1
            visible_arrival_sum += stop_probability
            visible_top10_match_sum += top10_match
        else:
            invisible_count += 1
            invisible_arrival_sum += stop_probability
            invisible_top10_match_sum += top10_match
            invisible_gate_stop_count += int(
                stop_probability >= float(ARRIVAL_STOP_THRESHOLD)
            )

        action_counter[ACTION_NAMES[action]] += 1
        if action == STOP:
            goal_visible_at_stop = bool(goal_visible)

        # 当前视觉 token 在动作执行后成为下一时刻的 MEC 侧历史缓存。
        history_frames.append(
            compress_visual_tokens(current_tokens.squeeze(0)).detach().cpu()
        )
        history_frames = history_frames[-AUX.history_length :]

        ended_by_stop = bool(action == STOP)
        obs, success, done = env.step(
            action,
            render_observation=(action != STOP),
        )

        trajectory.append(env.position.copy())
        min_distance = min(min_distance, float(env.distance_to_goal))
        osr = bool(osr or env.distance_to_goal < env.stop_radius)

        if done:
            break

    trajectory = np.asarray(trajectory, dtype=np.float32)
    num_steps = int(len(trajectory) - 1)

    if len(trajectory) > 1:
        path_length = float(
            np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum()
        )
    else:
        path_length = 0.0

    shortest_distance = float(oracle_info["shortest_distance_m"])
    spl = (
        shortest_distance / max(shortest_distance, path_length, 1e-8)
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
    max_steps_termination = bool((not ended_by_stop) and (not success) and done)

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
        "goal_visible_ratio": float(goal_visible_steps / max(num_steps, 1)),
        "goal_visible_at_stop": bool(goal_visible_at_stop),
        "mean_arrival_p_goal_visible": float(visible_arrival_sum / max(visible_count, 1)),
        "mean_arrival_p_goal_invisible": float(invisible_arrival_sum / max(invisible_count, 1)),
        "mean_top10_match_goal_visible": float(visible_top10_match_sum / max(visible_count, 1)),
        "mean_top10_match_goal_invisible": float(invisible_top10_match_sum / max(invisible_count, 1)),
        "invisible_gate_stop_count": int(invisible_gate_stop_count),
        "max_stop_probability": float(max_stop_probability),
        "stop_probability_at_min_distance": float(stop_probability_at_min_distance),
        "actions": dict(action_counter),
        "trajectory": trajectory,
    }


# =============================================================================
# 输出与可视化
# =============================================================================


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_episode(spec, oracle_info, model_trajectories, model_results):
    """将同一测试 episode 下所有固定 MEC 的闭环轨迹画在一张图中。"""
    fig, ax = plt.subplots(figsize=(9, 8))

    reference = oracle_info["reference"]
    ax.plot(
        reference[:, 0],
        reference[:, 1],
        linestyle="--",
        linewidth=2.0,
        label="Oracle reference",
    )

    all_points = [reference]
    for model_name, trajectory in model_trajectories.items():
        r = model_results[model_name]
        label = (
            f"{model_name} | SR={int(r['success'])} "
            f"OSR={int(r['osr'])} MinD={r['min_distance_m']:.1f}m"
        )
        ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=1.6, label=label)
        all_points.append(trajectory[:, :2])

    start = oracle_info["start"]
    goal = oracle_info["goal"]
    ax.scatter(start[0], start[1], marker="P", s=110, label="Start")
    ax.scatter(goal[0], goal[1], marker="*", s=180, label="Goal")

    # 画出成功 Stop 半径，便于直观看到“是否曾进入目标区域”。
    circle = plt.Circle(
        (goal[0], goal[1]),
        ENV.stop_radius_m,
        fill=False,
        linestyle=":",
        linewidth=1.4,
        label=f"Stop radius ({ENV.stop_radius_m:.0f}m)",
    )
    ax.add_patch(circle)

    all_points.append(np.asarray([start, goal], dtype=np.float32))
    stacked = np.concatenate(all_points, axis=0)
    xmin, ymin = stacked.min(axis=0)
    xmax, ymax = stacked.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 50.0)
    margin = max(30.0, 0.12 * span)

    ax.set_xlim(max(0.0, xmin - margin), min(ENV.world_size_m, xmax + margin))
    ax.set_ylim(max(0.0, ymin - margin), min(ENV.world_size_m, ymax + margin))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"V8.1 Fixed-MEC Closed-Loop Trajectories | "
        f"Episode {spec['episode_index'] + 1} | seed={spec['seed']} | band={spec['band']}"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()

    out_path = TRAJECTORY_DIR / (
        f"episode_{spec['episode_index'] + 1:03d}_seed_{spec['seed']}.png"
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    # 同时保存轨迹坐标，后续画论文图时无需重新推理。
    npz_data = {
        "oracle": reference,
        "start": start,
        "goal": goal,
    }
    for model_name, trajectory in model_trajectories.items():
        npz_data[safe_name(model_name)] = trajectory
    np.savez_compressed(out_path.with_suffix(".npz"), **npz_data)


# =============================================================================
# 主流程
# =============================================================================


def main():
    print("=" * 96)
    print("V8.1 FIXED SINGLE-MEC CAPABILITY EVALUATION")
    print("=" * 96)
    print("Device:", DEVICE)
    print("Checkpoint dir:", CHECKPOINT_DIR)
    print("Test stats:", TEST_STATS_PATH)
    print(
        "Protocol: one episode = one fixed MEC model for ALL decision steps; "
        "no association/switching/latency."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    model_names = find_available_models()
    if not model_names:
        raise FileNotFoundError(
            f"No available checkpoints under {CHECKPOINT_DIR}. "
            "Finish at least one V8 MEC training first."
        )
    print("Available models:", model_names)

    episode_specs = load_test_episode_specs()
    print(f"Evaluation episodes: {len(episode_specs)}")
    print(
        f"V8.1 Stop gate threshold: {ARRIVAL_STOP_THRESHOLD:.2f} | "
        "Stop only if P(arrival) >= threshold"
    )

    # 同一批 Oracle reference 对所有 MEC 共用。
    oracle_cache = {}
    for spec in episode_specs:
        oracle_cache[spec["episode_index"]] = build_oracle_reference(spec)

    if not all(x["oracle_success"] for x in oracle_cache.values()):
        raise RuntimeError("At least one Oracle reference episode failed unexpectedly")

    encoder = load_frozen_encoder(DEVICE)

    episode_rows = []
    # 只缓存前 PLOT_FIRST_N 个 episode 的轨迹，避免无必要占用内存。
    plot_trajectories = defaultdict(dict)
    plot_results = defaultdict(dict)

    for model_name in model_names:
        print("\n" + "=" * 96)
        print(f"Evaluating fixed MEC: {model_name}")
        print("=" * 96)

        model = load_model(model_name)
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

        for idx, spec in enumerate(episode_specs):
            oracle_info = oracle_cache[spec["episode_index"]]
            result = rollout_fixed_model(spec, oracle_info, encoder, model)

            actions = result["actions"]
            row = {
                "model": model_name,
                "params": int(params),
                "episode": int(spec["episode_index"] + 1),
                "seed": int(spec["seed"]),
                "band": spec["band"],
                "start_goal_distance_m": float(
                    spec["dataset_start_goal_distance_m"]
                ),
                "success": int(result["success"]),
                "osr": int(result["osr"]),
                "premature_stop": int(result["premature_stop"]),
                "max_steps_termination": int(result["max_steps_termination"]),
                "steps": int(result["steps"]),
                "min_distance_m": float(result["min_distance_m"]),
                "final_distance_m": float(result["final_distance_m"]),
                "path_length_m": float(result["path_length_m"]),
                "spl": float(result["spl"]),
                "ndtw": float(result["ndtw"]),
                "sdtw": float(result["sdtw"]),
                "stop_count": int(result["stop_count"]),
                "stop_ratio": float(result["stop_ratio"]),
                "goal_visible_ratio": float(result["goal_visible_ratio"]),
                "goal_visible_at_stop": int(result["goal_visible_at_stop"]),
                "mean_arrival_p_goal_visible": float(result["mean_arrival_p_goal_visible"]),
                "mean_arrival_p_goal_invisible": float(result["mean_arrival_p_goal_invisible"]),
                "mean_top10_match_goal_visible": float(result["mean_top10_match_goal_visible"]),
                "mean_top10_match_goal_invisible": float(result["mean_top10_match_goal_invisible"]),
                "invisible_gate_stop_count": int(result["invisible_gate_stop_count"]),
                "max_stop_probability": float(result["max_stop_probability"]),
                "stop_probability_at_min_distance": float(
                    result["stop_probability_at_min_distance"]
                ),
                "forward_count": int(actions.get("Forward", 0)),
                "left_count": int(actions.get("Left", 0)),
                "right_count": int(actions.get("Right", 0)),
                "oracle_steps": int(oracle_info["oracle_steps"]),
            }
            episode_rows.append(row)

            if idx < int(PLOT_FIRST_N):
                plot_trajectories[spec["episode_index"]][model_name] = result[
                    "trajectory"
                ]
                plot_results[spec["episode_index"]][model_name] = result

            print(
                f"[{idx + 1:02d}/{len(episode_specs):02d}] "
                f"seed={spec['seed']} band={spec['band']:<6s} "
                f"SR={int(result['success'])} OSR={int(result['osr'])} "
                f"premStop={int(result['premature_stop'])} "
                f"steps={result['steps']:3d} "
                f"minD={result['min_distance_m']:6.2f}m "
                f"SPL={result['spl']:.3f} nDTW={result['ndtw']:.3f}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 汇总指标
    # ------------------------------------------------------------------
    summary_rows = []
    print("\n" + "=" * 128)
    print("FIXED-MEC SUMMARY")
    print("=" * 128)
    print(
        f"{'Model':18s} {'Params':>10s} {'SR':>7s} {'OSR':>7s} {'Gap':>7s} "
        f"{'PremStop':>9s} {'SPL':>7s} {'nDTW':>7s} {'sDTW':>7s} "
        f"{'MinD':>8s} {'Steps':>7s}"
    )

    for model_name in model_names:
        rows = [r for r in episode_rows if r["model"] == model_name]
        n = max(len(rows), 1)

        sr = float(np.mean([r["success"] for r in rows]))
        osr = float(np.mean([r["osr"] for r in rows]))
        premature_stop = float(np.mean([r["premature_stop"] for r in rows]))
        max_step_rate = float(np.mean([r["max_steps_termination"] for r in rows]))

        summary = {
            "model": model_name,
            "params": int(rows[0]["params"]),
            "episodes": int(n),
            "success_rate": sr,
            "osr": osr,
            "osr_minus_sr": osr - sr,
            "premature_stop_rate": premature_stop,
            "max_steps_termination_rate": max_step_rate,
            "mean_spl": float(np.mean([r["spl"] for r in rows])),
            "mean_ndtw": float(np.mean([r["ndtw"] for r in rows])),
            "mean_sdtw": float(np.mean([r["sdtw"] for r in rows])),
            "mean_min_distance_m": float(
                np.mean([r["min_distance_m"] for r in rows])
            ),
            "mean_final_distance_m": float(
                np.mean([r["final_distance_m"] for r in rows])
            ),
            "mean_path_length_m": float(
                np.mean([r["path_length_m"] for r in rows])
            ),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "mean_stop_ratio": float(np.mean([r["stop_ratio"] for r in rows])),
            "mean_goal_visible_ratio": float(
                np.mean([r["goal_visible_ratio"] for r in rows])
            ),
            "mean_arrival_p_goal_visible": float(
                np.mean([r["mean_arrival_p_goal_visible"] for r in rows])
            ),
            "mean_arrival_p_goal_invisible": float(
                np.mean([r["mean_arrival_p_goal_invisible"] for r in rows])
            ),
            "mean_top10_match_goal_visible": float(
                np.mean([r["mean_top10_match_goal_visible"] for r in rows])
            ),
            "mean_top10_match_goal_invisible": float(
                np.mean([r["mean_top10_match_goal_invisible"] for r in rows])
            ),
            "episodes_with_invisible_gate_stop": int(
                np.sum([r["invisible_gate_stop_count"] > 0 for r in rows])
            ),
            "mean_max_stop_probability": float(
                np.mean([r["max_stop_probability"] for r in rows])
            ),
            "mean_stop_probability_at_min_distance": float(
                np.mean([r["stop_probability_at_min_distance"] for r in rows])
            ),
        }
        summary_rows.append(summary)

        print(
            f"{model_name:18s} {summary['params']:10,d} "
            f"{100*summary['success_rate']:6.2f}% "
            f"{100*summary['osr']:6.2f}% "
            f"{100*summary['osr_minus_sr']:6.2f}% "
            f"{100*summary['premature_stop_rate']:8.2f}% "
            f"{summary['mean_spl']:7.3f} "
            f"{summary['mean_ndtw']:7.3f} "
            f"{summary['mean_sdtw']:7.3f} "
            f"{summary['mean_min_distance_m']:8.2f} "
            f"{summary['mean_steps']:7.1f}"
        )

    save_csv(OUTPUT_DIR / "fixed_mec_episode_results.csv", episode_rows)
    save_csv(OUTPUT_DIR / "fixed_mec_summary.csv", summary_rows)

    # JSON 便于后续脚本直接读取。
    with (OUTPUT_DIR / "fixed_mec_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 轨迹图
    # ------------------------------------------------------------------
    plot_count = min(int(PLOT_FIRST_N), len(episode_specs))
    for idx in range(plot_count):
        spec = episode_specs[idx]
        episode_id = spec["episode_index"]
        plot_episode(
            spec,
            oracle_cache[episode_id],
            plot_trajectories[episode_id],
            plot_results[episode_id],
        )

    print("\nSaved:")
    print(" ", OUTPUT_DIR / "fixed_mec_episode_results.csv")
    print(" ", OUTPUT_DIR / "fixed_mec_summary.csv")
    print(" ", OUTPUT_DIR / "fixed_mec_summary.json")
    if plot_count > 0:
        print(" ", TRAJECTORY_DIR, f"({plot_count} trajectory figures + NPZ files)")

    print("\nInterpretation:")
    print("  1) SR/OSR/MinD -> 是否真正具备闭环到达能力；")
    print("  2) OSR-SR -> 到过目标附近但没有正确 Stop 的比例；")
    print("  3) PremStop -> 过早 Stop 导致的失败；")
    print("  4) SPL/nDTW/sDTW -> 成功之外的路径质量；")
    print("  5) Goal-visible/invisible Arrival与match -> 检查no-match false grounding；")
    print("  6) 同一 seed 下的轨迹图 -> 直接观察不同 MEC 的推理行为是否分化。")


if __name__ == "__main__":
    main()
