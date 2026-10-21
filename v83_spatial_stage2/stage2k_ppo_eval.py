"""Evaluate a Stage-2k PPO checkpoint on the full Dev Val 180.

The evaluation contract is identical to the DQN evaluator: the same 180 Dev
episodes, the same Dev topologies, the same five frozen MECs, the same
JointR definition and the same existing five baselines.  Final Test is never
read here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BASELINE_DIR,
    SCENARIOS,
    atomic_write_json,
    bootstrap_ci,
    load_split_specs,
    load_topology,
    read_json,
    summarize_episodes,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_ppo_agent import PPOAgent
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (
    STATE_DIM_PURE_MEC_V2,
)
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    SceneCache,
    load_gab_config,
    load_runtime,
)
from v83_spatial_stage2.stage2k_v2_common import (
    CAPABILITY_PRIOR_PATH,
    load_capability_prior,
)

DEFAULT_RUN = Path("runs/v83_spatial_stage2/stage2k_ppo")
BASELINE_STRATEGIES = (
    "random", "min_latency", "max_quality", "heuristic", "oracle",
)
NON_ORACLE = ("random", "min_latency", "max_quality", "heuristic")
TITLES = {
    "ppo": "PPO",
    "random": "Random",
    "min_latency": "Min-Latency",
    "max_quality": "Max-Quality",
    "heuristic": "Heuristic",
    "oracle": "Oracle (privileged)",
}


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired(ppo_rows, baseline_rows, strategy):
    baseline = {
        row["episode_uid"]: row for row in baseline_rows
        if row["strategy"] == strategy
    }
    deltas = {"JointR": [], "nDTW": [], "SR": [], "E2E_ms": []}
    wins = losses = ties = 0
    for row in ppo_rows:
        base = baseline.get(row["episode_uid"])
        if base is None:
            continue
        joint = float(row["joint_reward"]) - float(base["joint_reward"])
        deltas["JointR"].append(joint)
        deltas["nDTW"].append(float(row["ndtw"]) - float(base["ndtw"]))
        deltas["SR"].append(float(row["success"]) - float(base["success"]))
        deltas["E2E_ms"].append(
            float(row["mean_e2e_ms"]) - float(base["mean_e2e_ms"])
        )
        if joint > 1e-9:
            wins += 1
        elif joint < -1e-9:
            losses += 1
        else:
            ties += 1
    output = {
        "strategy": strategy,
        "n": len(deltas["JointR"]),
        "wins": wins, "losses": losses, "ties": ties,
    }
    for key, values in deltas.items():
        output[f"delta_{key}"] = bootstrap_ci(values)
    return output


def markdown(path, summary, baselines, paired_rows, training_result):
    lines = [
        "# Stage-2k PPO — Dev Val 180",
        "",
        f"- 日期：{summary['created_utc']}",
        f"- Checkpoint：`{summary['checkpoint']}`",
        f"- Episode：{summary['checkpoint_episode']}",
        f"- 状态：`{summary['state_mode']}`，{summary['state_dim']} 维"
        "（另有 20 维可用性 Mask）",
        "- 数据：完整 Dev Validation 180；未读取 Final Test。",
        "- Baseline：复用已有 Dev180 结果，未重跑。",
        "",
    ]
    if training_result:
        lines += [
            "## 训练收敛状态",
            "",
            f"- 训练状态：`{training_result.get('status')}`",
            f"- `converged`：**{training_result.get('converged')}**",
            f"- 最佳 Dev60 JointR："
            f"{training_result.get('best_validation_joint')} "
            f"(Ep{training_result.get('best_validation_episode')})",
            f"- 最后 3 次验证趋势："
            f"{training_result.get('validation_trend_last3')}",
            "",
        ]
        if not training_result.get("converged"):
            lines += [
                "> **注意：PPO 训练未通过预注册收敛门。本次结果只能作为"
                "诊断，不能标记为有效的最终策略，也不应进入 Final Test。**",
                "",
            ]
    lines += [
        "## 总体指标",
        "",
        "| 策略 | SR | OSR | SPL | nDTW | E2E (ms) | P95 (ms) | 违约率 | 步数 | JointR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ppo = summary["overall"]
    lines.append(
        f"| PPO | {100*ppo['SR']:.2f}% | {100*ppo['OSR']:.2f}% | "
        f"{ppo['SPL']:.4f} | {ppo['nDTW']:.4f} | {ppo['E2E_ms']:.2f} | "
        f"{ppo['P95_ms']:.2f} | {100*ppo['delay_violation_rate']:.2f}% | "
        f"{ppo['mean_steps']:.1f} | **{ppo['JointR']:.4f}** |"
    )
    for strategy in BASELINE_STRATEGIES:
        item = baselines[strategy]
        lines.append(
            f"| {TITLES[strategy]} | {100*item['SR']:.2f}% | — | — | "
            f"{item['nDTW']:.4f} | {item['E2E_ms']:.2f} | "
            f"{item['P95_ms']:.2f} | {100*item['delay_violation_rate']:.2f}% | "
            f"{item['mean_steps']:.1f} | {item['JointR']:.4f} |"
        )
    lines += [
        "",
        "## 逐 Episode 配对（PPO − Baseline）",
        "",
        "| Baseline | n | ΔJointR [95% CI] | ΔnDTW | ΔSR | ΔE2E (ms) | 胜/负/平 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in paired_rows:
        lines.append(
            f"| {TITLES[item['strategy']]} | {item['n']} | "
            f"{item['delta_JointR']['mean']:+.4f} "
            f"[{item['delta_JointR']['low']:+.4f}, "
            f"{item['delta_JointR']['high']:+.4f}] | "
            f"{item['delta_nDTW']['mean']:+.4f} | "
            f"{item['delta_SR']['mean']:+.4f} | "
            f"{item['delta_E2E_ms']['mean']:+.2f} | "
            f"{item['wins']}/{item['losses']}/{item['ties']} |"
        )
    lines += [
        "",
        "## 分部署条件（PPO）",
        "",
        "| 场景 | SR | OSR | SPL | nDTW | E2E (ms) | P95 (ms) | JointR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        item = summary["by_scenario"][scenario]
        if not item.get("episodes"):
            lines.append(
                f"| {scenario} | — | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {scenario} | {100*item['SR']:.2f}% | "
            f"{100*item['OSR']:.2f}% | {item['SPL']:.4f} | "
            f"{item['nDTW']:.4f} | "
            f"{item['E2E_ms']:.2f} | {item['P95_ms']:.2f} | "
            f"{item['JointR']:.4f} |"
        )
    lines += [
        "",
        "## 奖励一致性",
        "",
        "- 训练记录的最大 `|Return − JointR|`："
        + (
            f"{summary['training_jointr_abs_error_max']:.3e}"
            if summary["training_jointr_abs_error_max"] is not None
            else "—"
        ),
        f"- 本评估的最大 `|Return − JointR|`："
        f"{summary['eval_jointr_abs_error_max']:.3e}",
        "",
        "JointR = mean(nDTW) − 0.25 × mean(E2E_ms) / 100。",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--capability-prior", type=Path, default=CAPABILITY_PRIOR_PATH
    )
    args = parser.parse_args()
    run_dir = args.run_dir
    checkpoint = args.checkpoint or run_dir / "checkpoints" / "best.pt"
    output_dir = args.output_dir or run_dir / "dev180_eval"
    if (output_dir / "summary.json").is_file():
        raise SystemExit(
            f"{output_dir}/summary.json exists; refusing to overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    _, quality = load_capability_prior(args.capability_prior)
    specs_payload = load_split_specs("dev")
    topologies = {
        scenario: load_topology(scenario, "dev") for scenario in SCENARIOS
    }
    requested = str(args.device).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device, encoder, backends = load_runtime(requested)
    gab_config = load_gab_config()
    agent = PPOAgent(
        device, state_dim=STATE_DIM_PURE_MEC_V2,
        state_mode="pure_mec_v2",
    )
    payload = agent.load_model(checkpoint)
    config = payload.get("config", {})
    if config.get("reward_mode") != "jointr_aligned_v2":
        raise SystemExit("Checkpoint is not a JointR-aligned v2 model")
    if str(config.get("state_mode")) != "pure_mec_v2":
        raise SystemExit("Checkpoint state_mode is not pure_mec_v2")
    if int(config.get("state_dim", -1)) != STATE_DIM_PURE_MEC_V2:
        raise SystemExit("Checkpoint state_dim is not 384")

    cache = SceneCache(encoder, device, max_entries=180)
    rows = []
    step_rows = []
    for scenario in SCENARIOS:
        for spec in specs_payload["scenarios"][scenario]["episodes"]:
            if (
                args.max_episodes is not None
                and len(rows) >= args.max_episodes
            ):
                break
            runner = EpisodeRunner(
                spec=spec,
                topology=topologies[scenario],
                encoder=encoder,
                backends=backends,
                projector=None,
                quality_prior=quality,
                gab_config=gab_config,
                scene_cache=cache,
                device=device,
                jointr_aligned=True,
                pure_mec_v2_state=True,
            )
            while True:
                action = agent.greedy_action(
                    runner.step_data["state"], runner.step_data["mask"]
                )
                transition = runner.apply(action)
                if transition["done"]:
                    break
            result = runner.finalize("ppo_greedy")
            if result["jointr_abs_error"] > 1e-6:
                raise RuntimeError(
                    f"{result['episode_uid']} reward/JointR mismatch: "
                    f"{result['jointr_abs_error']}"
                )
            rows.append(result)
            step_rows.extend(result["step_records"])
        if (
            args.max_episodes is not None
            and len(rows) >= args.max_episodes
        ):
            break

    overall = summarize_episodes(rows)
    overall["OSR"] = float(np.mean([float(row["osr"]) for row in rows]))
    overall["SPL"] = float(np.mean([float(row["spl"]) for row in rows]))
    by_scenario = {
        scenario: summarize_episodes([
            row for row in rows if row["scenario"] == scenario
        ])
        for scenario in SCENARIOS
    }
    baseline_rows = read_csv(BASELINE_DIR / "dev" / "per_episode.csv")
    baselines = {}
    for strategy in BASELINE_STRATEGIES:
        selected = [
            row for row in baseline_rows if row["strategy"] == strategy
        ]
        baselines[strategy] = summarize_episodes(selected)
    paired_rows = [
        paired(rows, baseline_rows, strategy) for strategy in NON_ORACLE
    ]
    training_result = read_json(run_dir / "training_result.json") or {}
    summary = {
        "version": "stage2k-ppo-dev180-eval-v1",
        "created_utc": utc_now(),
        "algorithm": "ppo",
        "checkpoint": str(checkpoint),
        "checkpoint_episode": int(payload.get("episode", 0)),
        "run_dir": str(run_dir),
        "state_mode": "pure_mec_v2",
        "state_dim": int(STATE_DIM_PURE_MEC_V2),
        "episodes": len(rows),
        "overall": overall,
        "model_usage": overall["model_usage"],
        "model_usage_ratio": overall["model_usage_ratio"],
        "by_scenario": by_scenario,
        "baselines": baselines,
        "paired": paired_rows,
        "training_status": training_result.get("status"),
        "training_converged": training_result.get("converged"),
        "training_jointr_abs_error_max": training_result.get(
            "reward_jointr_abs_error_max"
        ),
        "eval_jointr_abs_error_max": float(max(
            row["jointr_abs_error"] for row in rows
        )),
        "final_test_read": False,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    per_episode_rows = [{
        "episode_uid": row["episode_uid"],
        "scenario": row["scenario"],
        "band": row["band"],
        "seed": row["seed"],
        "success": int(row["success"]),
        "osr": int(row["osr"]),
        "spl": row["spl"],
        "ndtw": row["ndtw"],
        "steps": row["steps"],
        "mean_e2e_ms": row["mean_e2e_ms"],
        "p95_e2e_ms": row["p95_e2e_ms"],
        "delay_violation_rate": row["delay_violation_rate"],
        "joint_reward": row["joint_reward"],
        "episode_return": row["episode_return"],
        "episode_jointR": row["episode_jointR"],
        "jointr_abs_error": row["jointr_abs_error"],
        "mean_navigation_reward": row["mean_navigation_reward"],
        "mean_latency_penalty": row["mean_latency_penalty"],
        "model_usage": json.dumps(row["model_usage"], sort_keys=True),
    } for row in rows]
    write_csv(output_dir / "per_episode.csv", per_episode_rows)
    write_csv(output_dir / "steps.csv", step_rows)
    markdown(
        output_dir / "STAGE2K_PPO_DEV180.md",
        summary, baselines, paired_rows, training_result,
    )
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "episode": summary["checkpoint_episode"],
        "overall": overall,
        "paired": paired_rows,
        "training_status": summary["training_status"],
        "training_converged": summary["training_converged"],
        "training_error_max": summary["training_jointr_abs_error_max"],
        "eval_error_max": summary["eval_jointr_abs_error_max"],
    }, indent=2, ensure_ascii=False))
    print("STAGE2K_PPO_EVAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
