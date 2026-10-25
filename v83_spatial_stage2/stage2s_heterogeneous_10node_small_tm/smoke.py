"""Stage-2s pre-training smoke test.

Checks the heterogeneous closed loop directly, then (unless
``--skip-training``) runs 10-20 Episode DQN and PPO smoke trainings on the
real environment.  No Stage-2r lane cache is involved.

Examples:
    python .../smoke.py --episodes 14 --dev-episodes 10
    python .../smoke.py --skip-training
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import STOP  # noqa: E402
from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    utc_now,
)
from v83_spatial_stage2.stage2k_latency import (  # noqa: E402
    stage2j_model_flops,
)
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (  # noqa: E402
    BLOCK_E2E,
    BLOCK_INFERENCE,
)
from v83_spatial_stage2.stage2o.rewards import (  # noqa: E402
    EpisodeRewardController,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm import (  # noqa: E402
    common,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.runtime import (  # noqa: E402
    dual_model_probe,
    latency_stats,
    load_stage2s_runtime,
    make_scene_cache,
    runner_for,
)

REPORT_PATH = common.EVIDENCE_DIR / "smoke_report.json"
REPORT_MD_PATH = common.EVIDENCE_DIR / "SMOKE_TEST.md"
LOG_ROOT = common.RUNS_ROOT / "smoke_formal"


def _available_by_model(runner):
    mask = np.asarray(runner.step_data["mask"], bool)
    return {
        model: [
            int(index) for index in np.where(mask)[0]
            if runner.nodes[int(index)]["model_type"] == model
        ]
        for model in common.MEC_MODELS
    }


def _find_divergent_step(bundle, encoder, backends, cache, device, latency,
                         max_tasks=20, max_steps=80):
    for task_index in range(max_tasks):
        runner = runner_for(
            bundle, bundle["dev_specs"][task_index], encoder, backends,
            cache, device, latency,
        )
        prefix = []
        for _ in range(max_steps):
            available = _available_by_model(runner)
            if available["Small"] and available["Transformer-M"]:
                probe = dual_model_probe(runner)
                if (
                    probe["actions_differ"]
                    and probe["actions"]["Small"] != STOP
                    and probe["actions"]["Transformer-M"] != STOP
                ):
                    return task_index, list(prefix), probe
            mask = np.where(runner.step_data["mask"])[0]
            if len(mask) == 0:
                break
            action = int(mask[0])
            prefix.append(action)
            transition = runner.apply(action)
            if transition["done"]:
                break
    return None, None, None


def _closed_loop_influence(bundle, encoder, backends, cache, device, latency,
                           task_index, prefix):
    runners = []
    for _ in range(2):
        runner = runner_for(
            bundle, bundle["dev_specs"][task_index], encoder, backends,
            cache, device, latency,
        )
        for action in prefix:
            runner.apply(int(action))
        runners.append(runner)
    available = _available_by_model(runners[0])
    small_index = available["Small"][0]
    transformer_index = available["Transformer-M"][0]
    pose_before = runners[0].env.pose.copy()
    small = runners[0].apply(small_index)
    transformer = runners[1].apply(transformer_index)
    return {
        "task_index": int(task_index),
        "prefix_steps": len(prefix),
        "small_node": int(small_index),
        "transformer_node": int(transformer_index),
        "small_action": int(runners[0].step_records[-1]["action"]),
        "transformer_action": int(
            runners[1].step_records[-1]["action"]
        ),
        "pose_before": pose_before.tolist(),
        "pose_after_small": runners[0].env.pose.tolist(),
        "pose_after_transformer": runners[1].env.pose.tolist(),
        "next_state_small": (
            None if small["next_state"] is None
            else small["next_state"].tolist()
        ),
        "next_state_transformer": (
            None if transformer["next_state"] is None
            else transformer["next_state"].tolist()
        ),
        "pose_changed_small": bool(
            not np.allclose(pose_before, runners[0].env.pose)
        ),
        "pose_changed_transformer": bool(
            not np.allclose(pose_before, runners[1].env.pose)
        ),
        "next_states_differ": bool(
            small["next_state"] is not None
            and transformer["next_state"] is not None
            and not np.allclose(
                small["next_state"], transformer["next_state"]
            )
        ),
    }


def _full_episode(bundle, encoder, backends, cache, device, latency):
    runner = runner_for(
        bundle, bundle["dev_specs"][0], encoder, backends, cache,
        device, latency,
    )
    controller = EpisodeRewardController("D")
    e2e_components_ok = True
    steps = 0
    while True:
        mask = np.where(runner.step_data["mask"])[0]
        node_index = int(mask[steps % len(mask)])
        metric = runner.step_data["latency_metrics"][node_index]
        candidate = runner.step_data["candidate_features"][node_index]
        if (
            abs(
                float(metric["total_ms"])
                - float(metric["extraction_ms"])
                - float(metric["communication_ms"])
                - float(metric["inference_ms"])
            ) > 1e-9
            or abs(
                float(candidate[BLOCK_INFERENCE])
                - min(float(metric["inference_ms"]) / 10.0, 3.0)
            ) > 1e-6
            or abs(
                float(candidate[BLOCK_E2E])
                - min(float(metric["total_ms"]) / 100.0, 3.0)
            ) > 1e-6
        ):
            e2e_components_ok = False
        transition = runner.apply(node_index)
        controller.step_reward(transition["e2e_ms"])
        steps += 1
        if transition["done"]:
            break
        if steps >= 400:
            break
    result = runner.finalize("smoke")
    _, diagnostics = controller.finalize(
        result["mean_episode_e2e_ms"], result["steps"]
    )
    expected = 1.0 - float(result["mean_episode_e2e_ms"]) / 100.0
    stats = latency_stats(result["step_records"])
    return {
        "episode_uid": result["episode_uid"],
        "steps": int(result["steps"]),
        "done": bool(result["steps"] <= 400),
        "success": bool(result["success"]),
        "mean_e2e_ms": float(result["mean_episode_e2e_ms"]),
        "reward_D": float(result["reward_D"]),
        "reward_expected": float(expected),
        "reward_abs_error": float(abs(diagnostics["return"] - expected)),
        "runner_reconstruction_error": float(
            result["reward_D_reconstruction_error"]
        ),
        "state_dim": int(runner.step_data["state"].shape[0])
        if runner.step_data is not None else 194,
        "mask_dim": int(len(runner.step_data["mask"]))
        if runner.step_data is not None else 10,
        "small_steps": int(stats["small_steps"]),
        "transformer_m_steps": int(stats["transformer_m_steps"]),
        "steps_accounted": bool(
            stats["small_steps"] + stats["transformer_m_steps"]
            == result["steps"]
        ),
        "e2e_components_ok": bool(e2e_components_ok),
        "model_usage": dict(result["model_usage"]),
        "mean_inference_small_ms": stats["mean_inference_small_ms"],
        "mean_inference_transformer_m_ms": stats[
            "mean_inference_transformer_m_ms"
        ],
    }


def run_checks():
    bundle = common.load_bundle()
    device, encoder, backends, latency = load_stage2s_runtime("cuda")
    cache = make_scene_cache(encoder, device, capacity=16)
    node = {
        "x": 750.0, "y": 750.0,
        "bandwidth_hz": common.MEC_BANDWIDTH_HZ,
        "compute_capacity_flops_per_s": (
            common.MEC_NOMINAL_COMPUTE_FLOPS_PER_S
        ),
        "coverage_radius_m": common.MEC_COVERAGE_RADIUS_M,
    }
    latencies = {
        model: latency.evaluate(node, (750.0, 750.0), model)
        for model in common.MEC_MODELS
    }
    expected_inference = {
        model: 1000.0 * stage2j_model_flops(model) / 0.2e12
        for model in common.MEC_MODELS
    }
    task_index, prefix, probe = _find_divergent_step(
        bundle, encoder, backends, cache, device, latency
    )
    if probe is None:
        raise RuntimeError("no divergent Small/TM observation found")
    influence = _closed_loop_influence(
        bundle, encoder, backends, cache, device, latency,
        task_index, prefix,
    )
    episode = _full_episode(
        bundle, encoder, backends, cache, device, latency
    )
    checks = {
        "two_models_loaded": set(backends.loaded_models)
        == set(common.MEC_MODELS),
        "small_inference_expected": abs(
            latencies["Small"]["inference_ms"]
            - expected_inference["Small"]
        ) < 1e-9,
        "transformer_m_inference_expected": abs(
            latencies["Transformer-M"]["inference_ms"]
            - expected_inference["Transformer-M"]
        ) < 1e-9,
        "inference_ratio_gt_100": (
            latencies["Transformer-M"]["inference_ms"]
            / latencies["Small"]["inference_ms"] > 100.0
        ),
        "e2e_includes_inference": all(
            abs(
                latencies[model]["total_ms"]
                - latencies[model]["extraction_ms"]
                - latencies[model]["communication_ms"]
                - latencies[model]["inference_ms"]
            ) < 1e-9
            for model in common.MEC_MODELS
        ),
        "dual_model_actions_differ": bool(probe["actions_differ"]),
        "model_choice_changes_pose": bool(
            influence["pose_changed_small"]
            and influence["pose_changed_transformer"]
        ),
        "model_choice_changes_next_state": bool(
            influence["next_states_differ"]
        ),
        "state_dim_194": episode["state_dim"] == 194,
        "mask_dim_10": episode["mask_dim"] == 10,
        "termination_within_400": bool(episode["done"]),
        "scheme_d_identity": episode["reward_abs_error"] <= 1e-8,
        "runner_scheme_d_identity": (
            episode["runner_reconstruction_error"] <= 1e-8
        ),
        "model_usage_accounted": bool(episode["steps_accounted"]),
        "e2e_components_ok": bool(episode["e2e_components_ok"]),
        "closed_loop_no_lane_cache": _no_lane_cache(),
    }
    checks["ok"] = all(checks.values())
    if not checks["ok"]:
        raise RuntimeError(f"smoke checks failed: {checks}")
    return {
        "bundle": {
            "train_tasks": len(bundle["train_specs"]),
            "dev_tasks": len(bundle["dev_specs"]),
            "deployment_sha256": bundle["deployment"][
                "deployment_sha256"
            ],
            "model_counts": bundle["deployment"]["mec"]["model_counts"],
            "checkpoint_sha256": dict(common.MEC_CHECKPOINT_SHA256),
        },
        "latencies_ms": latencies,
        "expected_inference_ms": expected_inference,
        "probe": probe,
        "closed_loop_influence": influence,
        "full_episode": episode,
        "checks": checks,
        "backend_provenance": backends.provenance(),
    }


def _no_lane_cache():
    from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm import (
        train,
    )

    source = inspect.getsource(train)
    forbidden = ("LaneRunnerAdapter", "load_lane_cache", "lane_cache.py")
    return not any(token in source for token in forbidden)


def _run_training_smoke(algorithm, episodes, dev_episodes):
    run_dir = LOG_ROOT / f"{algorithm}_s101"
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)
    log_path = LOG_ROOT / f"{algorithm}_s101.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(Path(__file__).resolve().parent / "train.py"),
        "--algorithm", algorithm,
        "--seed", "101",
        "--smoke",
        "--max-episodes", str(episodes),
        "--validation-every", str(episodes),
        "--dev-episodes", str(dev_episodes),
        "--run-dir", str(run_dir),
        "--threads", "2",
    ]
    print(f"[smoke] 启动 {algorithm.upper()} {episodes} Episodes ...",
          flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=str(PROJECT_ROOT), stdout=handle,
            stderr=subprocess.STDOUT,
        )
    result_path = run_dir / "training_result.json"
    history_path = run_dir / "training_history.json"
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"{algorithm} smoke training failed; see {log_path}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    rows = history["train"]
    max_reward_error = max(
        float(row["reward_abs_error"]) for row in rows
    )
    max_reconstruction_error = max(
        0.0 for _ in rows
    ) if not rows else max(
        abs(float(row["reward_D"]) - (1.0 - float(
            row["mean_episode_e2e_ms"]
        ) / 100.0))
        for row in rows
    )
    usage_ok = all(
        int(row["small_steps"]) + int(row["transformer_m_steps"])
        == int(row["steps"])
        for row in rows
    )
    summary = {
        "algorithm": algorithm,
        "episodes": int(result["episodes"]),
        "status": result["status"],
        "training_mode": result["training_mode"],
        "lane_cache_used": result["lane_cache_used"],
        "best_dev_e2e_ms": result["best_dev_e2e_ms"],
        "training_small_ratio": result["training_small_ratio"],
        "max_reward_abs_error": float(max_reward_error),
        "max_reward_D_closed_form_error": float(
            max_reconstruction_error
        ),
        "model_usage_accounted": bool(usage_ok),
        "convergence": result["convergence"]["status"],
        "run_dir": str(run_dir),
        "log": str(log_path),
    }
    summary["ok"] = bool(
        summary["episodes"] == int(episodes)
        and summary["training_mode"] == "full_closed_loop_heterogeneous"
        and summary["lane_cache_used"] is False
        and summary["max_reward_abs_error"] <= 1e-8
        and summary["max_reward_D_closed_form_error"] <= 1e-8
        and summary["model_usage_accounted"]
    )
    if not summary["ok"]:
        raise RuntimeError(f"{algorithm} smoke summary failed: {summary}")
    return summary


def _write_markdown(report):
    checks = report["checks"]
    lines = [
        "# Stage-2s Smoke Test（训练前）",
        "",
        f"- 时间：{report['created_utc']}",
        f"- 部署哈希：`{report['bundle']['deployment_sha256']}`",
        f"- 模型：Small×{report['bundle']['model_counts']['Small']} + "
        f"Transformer-M×{report['bundle']['model_counts']['Transformer-M']}",
        "- 真实闭环：`full_closed_loop_heterogeneous`，未使用 Stage-2r lane 缓存。",
        "",
        "## 检查结果",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: {'通过' if value else '失败'}")
    lines += [
        "",
        "## 推理时延（0.2 TFLOPS 有效吞吐）",
        "",
        "| 模型 | 解析 FLOPs | 推理时延 (ms) | 通信时延 (ms) | E2E (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in common.MEC_MODELS:
        metric = report["latencies_ms"][model]
        lines.append(
            f"| {model} | {metric['inference_flops']/1e9:.4f} G | "
            f"{metric['inference_ms']:.6f} | "
            f"{metric['communication_ms']:.3f} | "
            f"{metric['total_ms']:.3f} |"
        )
    probe = report["probe"]
    lines += [
        "",
        "## 同观测双模型动作",
        "",
        f"- Dev 任务索引：{report['closed_loop_influence']['task_index']}",
        f"- 前缀步数：{report['closed_loop_influence']['prefix_steps']}",
        f"- Small 动作：{probe['actions']['Small']}，"
        f"Transformer-M 动作：{probe['actions']['Transformer-M']}",
        f"- 选择不同模型后轨迹位姿改变："
        f"Small={report['closed_loop_influence']['pose_changed_small']}，"
        f"Transformer-M="
        f"{report['closed_loop_influence']['pose_changed_transformer']}",
        f"- 下一 194-D 状态不同："
        f"{report['closed_loop_influence']['next_states_differ']}",
        "",
        "## 完整闭环 Episode",
        "",
        f"- Episode：`{report['full_episode']['episode_uid']}`，"
        f"步数 {report['full_episode']['steps']}，"
        f"成功 {report['full_episode']['success']}",
        f"- 平均 E2E：{report['full_episode']['mean_e2e_ms']:.3f} ms",
        f"- 方案 D 奖励：{report['full_episode']['reward_D']:.6f}，"
        f"闭式 1-mean/100："
        f"{report['full_episode']['reward_expected']:.6f}，"
        f"误差 {report['full_episode']['reward_abs_error']:.2e}",
        f"- Small/Transformer-M 步数："
        f"{report['full_episode']['small_steps']}/"
        f"{report['full_episode']['transformer_m_steps']}",
        "",
    ]
    if report.get("training_smokes"):
        lines += [
            "## 10–20 Episode 训练 Smoke",
            "",
            "| 算法 | Episodes | 模式 | Small 选择 | 最大奖励误差 | 状态 |",
            "|---|---:|---|---:|---:|---|",
        ]
        for item in report["training_smokes"]:
            lines.append(
                f"| {item['algorithm'].upper()} | {item['episodes']} | "
                f"{item['training_mode']} | "
                f"{100*item['training_small_ratio']:.2f}% | "
                f"{item['max_reward_abs_error']:.2e} | "
                f"{item['status']} |"
            )
        lines.append("")
    lines += [
        "## 结论",
        "",
        "全部检查通过，可以启动 6 个正式 run。",
        "",
    ]
    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=14)
    parser.add_argument("--dev-episodes", type=int, default=10)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args(argv)
    if not 10 <= int(args.episodes) <= 20:
        raise SystemExit("--episodes must be between 10 and 20")
    if not 1 <= int(args.dev_episodes) <= 60:
        raise SystemExit("--dev-episodes must be between 1 and 60")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the closed-loop smoke test")
    report = {
        "stage": common.STAGE,
        "created_utc": utc_now(),
        "checks_only": bool(args.skip_training),
    }
    report.update(run_checks())
    print("[smoke] 闭环不变量检查全部通过", flush=True)
    if not args.skip_training:
        report["training_smokes"] = []
        for algorithm in common.ALGORITHMS:
            summary = _run_training_smoke(
                algorithm, int(args.episodes), int(args.dev_episodes)
            )
            report["training_smokes"].append(summary)
            print(
                f"[smoke] {algorithm.upper()}: "
                f"{summary['episodes']} Episodes 完成，"
                f"Small 选择 {100*summary['training_small_ratio']:.2f}%，"
                f"奖励误差 {summary['max_reward_abs_error']:.2e}",
                flush=True,
            )
    report["ok"] = True
    atomic_write_json(REPORT_PATH, report)
    _write_markdown(report)
    print(f"[smoke] 报告: {REPORT_PATH}", flush=True)
    print("STAGE2S_SMOKE_COMPLETE", flush=True)
    return report


if __name__ == "__main__":
    main()
