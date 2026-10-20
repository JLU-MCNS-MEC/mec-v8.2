"""Evaluate a Stage-2k JointR-aligned v2 checkpoint on the full Dev Val 180."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BASELINE_DIR,
    QUALITY_PRIOR_PATH,
    SCENARIOS,
    VISUAL_PCA_PATH,
    atomic_write_json,
    bootstrap_ci,
    load_split_specs,
    load_topology,
    read_json,
    summarize_episodes,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_dqn import flat_quality
from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    SceneCache,
    load_gab_config,
    load_runtime,
)
from v83_spatial_stage2.stage2k_no_visual_state import (
    STATE_DIM_NO_VISUAL,
)
from v83_spatial_stage2.stage2k_pure_mec_state import (
    STATE_DIM_PURE_MEC,
)
from v83_spatial_stage2.stage2k_state import (
    STATE_DIM,
    VisualProjector,
    masked_argmax,
)

DEFAULT_RUN = Path("runs/v83_spatial_stage2/stage2k_jointr_v2")
BASELINE_STRATEGIES = (
    "random", "min_latency", "max_quality", "heuristic", "oracle",
)
NON_ORACLE = ("random", "min_latency", "max_quality", "heuristic")
TITLES = {
    "dqn": "DQN v2", "random": "Random", "min_latency": "Min-Latency",
    "max_quality": "Max-Quality", "heuristic": "Heuristic",
    "oracle": "Oracle (privileged)",
}


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired(dqn_rows, baseline_rows, strategy):
    baseline = {
        row["episode_uid"]: row for row in baseline_rows
        if row["strategy"] == strategy
    }
    deltas = {"JointR": [], "nDTW": [], "SR": [], "E2E_ms": []}
    wins = losses = ties = 0
    for row in dqn_rows:
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


def markdown(path, summary, baselines, paired_rows, per_episode):
    lines = [
        "# Stage-2k JointR-Aligned v2 — Dev Val 180",
        "",
        f"- 日期：{summary['created_utc']}",
        f"- Checkpoint：`{summary['checkpoint']}`",
        f"- Episode：{summary['checkpoint_episode']}",
        "- 数据：完整 Dev Validation 180；未读取 Final Test。",
        "- Baseline：复用已有 Dev180 结果，未重跑。",
        "",
        "## 总体指标",
        "",
        "| 策略 | SR | OSR | SPL | nDTW | E2E (ms) | P95 (ms) | 违约率 | 步数 | JointR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    dqn = summary["overall"]
    lines.append(
        f"| DQN v2 | {100*dqn['SR']:.2f}% | {100*dqn['OSR']:.2f}% | "
        f"{dqn['SPL']:.4f} | {dqn['nDTW']:.4f} | {dqn['E2E_ms']:.2f} | "
        f"{dqn['P95_ms']:.2f} | {100*dqn['delay_violation_rate']:.2f}% | "
        f"{dqn['mean_steps']:.1f} | **{dqn['JointR']:.4f}** |"
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
        "## 逐 Episode 配对（DQN − Baseline）",
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
        "## 分部署条件（DQN）",
        "",
        "| 场景 | SR | nDTW | E2E (ms) | P95 (ms) | JointR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        item = summary["by_scenario"][scenario]
        if not item.get("episodes"):
            lines.append(f"| {scenario} | — | — | — | — | — |")
            continue
        lines.append(
            f"| {scenario} | {100*item['SR']:.2f}% | {item['nDTW']:.4f} | "
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
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--checkpoint", type=Path, default=None
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--state-mode",
        choices=("full", "no_visual", "pure_mec"), default="full",
    )
    args = parser.parse_args()
    no_visual = args.state_mode == "no_visual"
    pure_mec = args.state_mode == "pure_mec"
    if pure_mec:
        state_dim = STATE_DIM_PURE_MEC
    elif no_visual:
        state_dim = STATE_DIM_NO_VISUAL
    else:
        state_dim = STATE_DIM
    run_dir = args.run_dir
    checkpoint = args.checkpoint or run_dir / "checkpoints" / "best.pt"
    output_dir = args.output_dir or run_dir / "dev180_eval"
    if (output_dir / "summary.json").is_file():
        raise SystemExit(f"{output_dir}/summary.json exists; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    quality = flat_quality(read_json(QUALITY_PRIOR_PATH))
    specs_payload = load_split_specs("dev")
    topologies = {
        scenario: load_topology(scenario, "dev") for scenario in SCENARIOS
    }
    projector = (
        None if (no_visual or pure_mec)
        else VisualProjector.load(VISUAL_PCA_PATH)
    )
    device, encoder, backends = load_runtime(args.device)
    gab_config = load_gab_config()
    agent = DQNAgentV2(
        device, state_dim=state_dim, state_mode=args.state_mode
    )
    payload = agent.load_model(checkpoint)
    config = payload.get("config", {})
    if config.get("reward_mode") != "jointr_aligned_v2":
        raise SystemExit("Checkpoint is not a JointR-aligned v2 model")
    if str(config.get("state_mode", "full")) != args.state_mode:
        raise SystemExit("Checkpoint state_mode does not match --state-mode")
    if int(config.get("state_dim", -1)) != state_dim:
        raise SystemExit("Checkpoint state_dim does not match --state-mode")
    cache = SceneCache(encoder, device, max_entries=60)
    rows = []
    step_rows = []
    for scenario in SCENARIOS:
        for spec in specs_payload["scenarios"][scenario]["episodes"]:
            if args.max_episodes is not None and len(rows) >= args.max_episodes:
                break
            runner = EpisodeRunner(
                spec=spec,
                topology=topologies[scenario],
                encoder=encoder,
                backends=backends,
                projector=projector,
                quality_prior=quality,
                gab_config=gab_config,
                scene_cache=cache,
                device=device,
                jointr_aligned=True,
                no_visual_state=no_visual,
                pure_mec_state=pure_mec,
            )
            while True:
                state = runner.step_data["state"]
                mask = runner.step_data["mask"]
                action = int(masked_argmax(agent.q_values(state), mask))
                transition = runner.apply(action)
                if transition["done"]:
                    break
            result = runner.finalize("jointr_v2")
            if result["jointr_abs_error"] > 1e-6:
                raise RuntimeError(
                    f"{result['episode_uid']} reward/JointR mismatch: "
                    f"{result['jointr_abs_error']}"
                )
            rows.append(result)
            step_rows.extend(result["step_records"])
        if args.max_episodes is not None and len(rows) >= args.max_episodes:
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
        "version": "stage2k-jointr-v2-dev180-eval-v1",
        "created_utc": utc_now(),
        "checkpoint": str(checkpoint),
        "checkpoint_episode": int(payload.get("episode", 0)),
        "episodes": len(rows),
        "overall": overall,
        "by_scenario": by_scenario,
        "baselines": baselines,
        "paired": paired_rows,
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
        output_dir / "STAGE2K_JOINTR_V2_DEV180.md",
        summary, baselines, paired_rows, per_episode_rows,
    )
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "episode": summary["checkpoint_episode"],
        "overall": overall,
        "paired": paired_rows,
        "training_error_max": summary["training_jointr_abs_error_max"],
        "eval_error_max": summary["eval_jointr_abs_error_max"],
    }, indent=2, ensure_ascii=False))
    print("STAGE2K_JOINTR_V2_EVAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
