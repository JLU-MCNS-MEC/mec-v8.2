"""Evaluate the frozen Stage-2k DQN checkpoint on the fixed Dev Val 180.

This is a Dev-only evaluation.  It never reads the Association Final Test
specs or Stage-2j Final Test 300.  Baselines are reused from the completed
``baselines/dev`` run; no baseline is re-run.

Besides the standard metrics it records, for every step, the selected MEC's
E2E delay and quality prior against the fastest / highest-quality available
candidate, which lets us test whether the DQN learned the latency trade-off.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BANDS,
    BASELINE_DIR,
    DQN_DIR,
    QUALITY_PRIOR_PATH,
    SCENARIO_TITLES,
    SCENARIOS,
    VISUAL_PCA_PATH,
    atomic_write_json,
    bootstrap_ci,
    load_progress,
    load_split_specs,
    load_topology,
    read_json,
    summarize_episodes,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_dqn import DQNAgent, flat_quality
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    build_scene,
    load_gab_config,
    load_runtime,
)
from v83_spatial_stage2.stage2k_state import VisualProjector

OUTPUT_DIR = DQN_DIR / "dev180_eval"
BASELINE_PER_EPISODE = BASELINE_DIR / "dev" / "per_episode.csv"
BASELINE_STRATEGIES = ("random", "min_latency", "max_quality", "heuristic")
STRATEGY_TITLES = {
    "dqn": "DQN",
    "random": "Random",
    "min_latency": "Min-Latency",
    "max_quality": "Max-Quality",
    "heuristic": "Heuristic",
}
EPS = 1e-9


def read_csv_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_stats(dqn_rows, baseline_rows, strategy):
    baseline = {
        row["episode_uid"]: row for row in baseline_rows
        if row["strategy"] == strategy
    }
    deltas = {"JointR": [], "nDTW": [], "SR": [], "E2E_ms": []}
    wins = losses = ties = 0
    missing = 0
    for row in dqn_rows:
        base = baseline.get(row["episode_uid"])
        if base is None:
            missing += 1
            continue
        delta_joint = float(row["joint_reward"]) - float(base["joint_reward"])
        deltas["JointR"].append(delta_joint)
        deltas["nDTW"].append(float(row["ndtw"]) - float(base["ndtw"]))
        deltas["SR"].append(float(row["success"]) - float(base["success"]))
        deltas["E2E_ms"].append(
            float(row["mean_e2e_ms"]) - float(base["mean_e2e_ms"])
        )
        if delta_joint > EPS:
            wins += 1
        elif delta_joint < -EPS:
            losses += 1
        else:
            ties += 1
    output = {
        "strategy": strategy,
        "n_paired": len(deltas["JointR"]),
        "missing": missing,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }
    for key, values in deltas.items():
        item = bootstrap_ci(values)
        output[f"delta_{key}"] = item
    return output


def episode_relationship(dqn_rows, baseline_rows, strategy):
    """Count slower-but-higher-nDTW and slower-but-no-improvement episodes."""
    baseline = {
        row["episode_uid"]: row for row in baseline_rows
        if row["strategy"] == strategy
    }
    slower = slower_higher = slower_not_higher = 0
    slower_higher_joint = slower_not_higher_joint = 0
    faster = 0
    for row in dqn_rows:
        base = baseline.get(row["episode_uid"])
        if base is None:
            continue
        d_slower = float(row["mean_e2e_ms"]) > float(base["mean_e2e_ms"]) + EPS
        d_higher_ndtw = float(row["ndtw"]) > float(base["ndtw"]) + EPS
        d_higher_joint = (
            float(row["joint_reward"]) > float(base["joint_reward"]) + EPS
        )
        if d_slower:
            slower += 1
            if d_higher_ndtw:
                slower_higher += 1
                if d_higher_joint:
                    slower_higher_joint += 1
            else:
                slower_not_higher += 1
                if d_higher_joint:
                    slower_not_higher_joint += 1
        else:
            faster += 1
    return {
        "strategy": strategy,
        "episodes": slower + faster,
        "slower_total": slower,
        "slower_and_higher_ndtw": slower_higher,
        "slower_and_higher_jointr": slower_higher_joint,
        "slower_and_not_higher_ndtw": slower_not_higher,
        "slower_not_higher_but_higher_jointr": slower_not_higher_joint,
        "faster_or_equal_total": faster,
    }


def write_markdown(path, summary, step_summary, paired, relationships):
    lines = [
        "# Stage-2k DQN — 固定 Dev Val 180 评估",
        "",
        f"- 日期：{summary['created_utc']}",
        f"- Checkpoint：`{summary['checkpoint']}`（Episode {summary['checkpoint_episode']}）",
        "- 数据：Dev Validation 180（三个部署条件各 60；18 Short / 30 Medium / 12 Long）",
        "- 基线：复用 `runs/v83_spatial_stage2/stage2k/baselines/dev`，未重跑。",
        "- 本评估未读取 Association Final Test 或 Stage-2j Final Test 300。",
        "",
        "## 总体指标",
        "",
        "| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | 违约率 | 平均步数 | JointR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [("DQN", summary["overall"])]
    for strategy in BASELINE_STRATEGIES:
        rows.append((
            STRATEGY_TITLES[strategy], summary["baseline_overall"][strategy]
        ))
    for title, item in rows:
        lines.append(
            f"| {title} | {100*item['SR']:.2f}% | {item['nDTW']:.4f} | "
            f"{item['E2E_ms']:.2f} | {item['P95_ms']:.2f} | "
            f"{100*item['delay_violation_rate']:.2f}% | "
            f"{item['mean_steps']:.1f} | {item['JointR']:.4f} |"
        )
    lines += [
        "",
        "## DQN 的时延/质量选择行为",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 选择最快可用 MEC 的步比例 | {100*step_summary['fastest_step_rate']:.2f}% |",
        f"| 选择最高质量可用 MEC 的步比例 | {100*step_summary['max_quality_step_rate']:.2f}% |",
        f"| 选中 E2E − 当前最低 E2E，均值 | {step_summary['mean_e2e_minus_min_ms']:.2f} ms |",
        f"| 选中 E2E − 当前最低 E2E，中位数 | {step_summary['median_e2e_minus_min_ms']:.2f} ms |",
        f"| 选中 E2E − 当前最低 E2E，P95 | {step_summary['p95_e2e_minus_min_ms']:.2f} ms |",
        f"| 选中质量 − 当前最高质量，均值 | {step_summary['mean_quality_minus_max']:+.4f} |",
        f"| 选中质量 − 当前最高质量，中位数 | {step_summary['median_quality_minus_max']:+.4f} |",
        "",
        "## 逐 Episode 配对比较（DQN − Baseline）",
        "",
        "| Baseline | 配对 n | ΔJointR | ΔnDTW | ΔSR | ΔE2E (ms) | 胜/负/平 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in paired:
        lines.append(
            f"| {STRATEGY_TITLES[item['strategy']]} | {item['n_paired']} | "
            f"{item['delta_JointR']['mean']:+.4f} "
            f"[{item['delta_JointR']['low']:+.4f}, {item['delta_JointR']['high']:+.4f}] | "
            f"{item['delta_nDTW']['mean']:+.4f} | "
            f"{item['delta_SR']['mean']:+.4f} | "
            f"{item['delta_E2E_ms']['mean']:+.2f} | "
            f"{item['wins']}/{item['losses']}/{item['ties']} |"
        )
    lines += [
        "",
        "## 更慢但质量更高 / 更慢且无改善（逐 Episode）",
        "",
        "| 对照 | 更慢 Episode | 更慢且 nDTW 更高 | 其中 JointR 更高 | 更慢但 nDTW 未改善 | 其中 JointR 更高 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in relationships:
        lines.append(
            f"| vs {STRATEGY_TITLES[item['strategy']]} | "
            f"{item['slower_total']} | {item['slower_and_higher_ndtw']} | "
            f"{item['slower_and_higher_jointr']} | "
            f"{item['slower_and_not_higher_ndtw']} | "
            f"{item['slower_not_higher_but_higher_jointr']} |"
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
            lines.append(f"| {SCENARIO_TITLES[scenario]} | — | — | — | — | — |")
            continue
        lines.append(
            f"| {SCENARIO_TITLES[scenario]} | {100*item['SR']:.2f}% | "
            f"{item['nDTW']:.4f} | {item['E2E_ms']:.2f} | "
            f"{item['P95_ms']:.2f} | {item['JointR']:.4f} |"
        )
    lines += [
        "",
        "## MEC 类型选择频率（DQN）",
        "",
        "| MEC | 步数 | 比例 |",
        "|---|---:|---:|",
    ]
    for name, ratio in sorted(
        summary["overall"]["model_usage_ratio"].items(),
        key=lambda item: -item[1],
    ):
        count = summary["overall"]["model_usage"].get(name, 0)
        lines.append(f"| {name} | {count} | {100*ratio:.2f}% |")
    lines += [
        "",
        "JointR = mean(nDTW) − 0.25 × mean(E2E_ms) / 100。",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint", type=Path, default=DQN_DIR / "best.pt"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir

    if (output_dir / "summary.json").is_file():
        raise SystemExit(
            f"{output_dir}/summary.json exists; refusing to overwrite"
        )
    progress = load_progress()
    if progress["dqn"]["status"] != "completed":
        raise SystemExit("DQN is not marked completed")
    if progress["baseline"]["status"] != "completed":
        raise SystemExit("Baseline is not marked completed")

    quality_payload = read_json(QUALITY_PRIOR_PATH)
    if quality_payload is None:
        raise SystemExit("Missing quality prior")
    quality = flat_quality(quality_payload)
    specs_payload = load_split_specs("dev")
    topologies = {
        scenario: load_topology(scenario, "dev") for scenario in SCENARIOS
    }
    projector = VisualProjector.load(VISUAL_PCA_PATH)
    device, encoder, backends = load_runtime(args.device)
    gab_config = load_gab_config()

    agent = DQNAgent(device, seed=20261031)
    payload = agent.load(args.checkpoint)
    if bool(payload.get("final_test_read", False)):
        raise SystemExit("Checkpoint was produced after a Final Test read")
    print(
        f"Loaded {args.checkpoint} episode={payload.get('episode')} "
        f"best_joint={payload.get('best_joint')}",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = []
    step_rows = []
    total_planned = sum(
        len(specs_payload["scenarios"][scenario]["episodes"])
        for scenario in SCENARIOS
    )
    done = 0
    for scenario in SCENARIOS:
        topology = topologies[scenario]
        for spec in specs_payload["scenarios"][scenario]["episodes"]:
            if args.max_episodes is not None and done >= args.max_episodes:
                break
            scene = build_scene(spec, encoder, device)
            runner = EpisodeRunner(
                spec=spec,
                topology=topology,
                encoder=encoder,
                backends=backends,
                projector=projector,
                quality_prior=quality,
                gab_config=gab_config,
                scene=scene,
                device=device,
            )
            episode_steps = []
            while True:
                state = runner.step_data["state"]
                mask = runner.step_data["mask"]
                metrics = runner.step_data["latency_metrics"]
                available = np.where(mask)[0]
                min_index = runner.select_node("min_latency")
                max_quality_index = runner.select_node("max_quality")
                min_e2e = float(metrics[min_index]["total_ms"])
                max_quality = float(runner.nodes[max_quality_index]["quality_prior"])
                action = int(agent.act_greedy(state, mask))
                node = runner.nodes[action]
                selected_quality = float(node["quality_prior"])
                transition = runner.apply(action)
                e2e = float(transition["e2e_ms"])
                step_record = {
                    "episode_uid": runner.spec["episode_uid"],
                    "scenario": scenario,
                    "band": runner.spec["band"],
                    "step": len(runner.step_records) - 1,
                    "node_index": int(action),
                    "node_id": node["node_id"],
                    "model_type": node["model_type"],
                    "e2e_ms": e2e,
                    "min_available_e2e_ms": min_e2e,
                    "e2e_minus_min_ms": e2e - min_e2e,
                    "fastest_flag": int(e2e <= min_e2e + 1e-9),
                    "selected_quality": selected_quality,
                    "max_available_quality": max_quality,
                    "quality_minus_max": selected_quality - max_quality,
                    "max_quality_flag": int(
                        selected_quality >= max_quality - 1e-9
                    ),
                    "min_latency_node_index": int(min_index),
                    "max_quality_node_index": int(max_quality_index),
                    "action": int(transition["action"]),
                    "reward": float(transition["reward"]),
                    "navigation_reward": float(transition["nav_reward"]),
                    "latency_penalty": float(transition["latency_penalty"]),
                    "distance_to_goal_m": float(
                        runner.step_records[-1]["distance_to_goal_m"]
                    ),
                }
                episode_steps.append(step_record)
                if transition["done"]:
                    break
            result = runner.finalize("dqn")
            gaps = np.asarray(
                [row["e2e_minus_min_ms"] for row in episode_steps],
                dtype=np.float64,
            )
            quality_gaps = np.asarray(
                [row["quality_minus_max"] for row in episode_steps],
                dtype=np.float64,
            )
            result.update({
                "mean_e2e_minus_min_ms": float(gaps.mean()),
                "median_e2e_minus_min_ms": float(np.median(gaps)),
                "p95_e2e_minus_min_ms": float(np.percentile(gaps, 95)),
                "mean_quality_minus_max": float(quality_gaps.mean()),
                "median_quality_minus_max": float(np.median(quality_gaps)),
                "fastest_step_rate": float(np.mean([
                    row["fastest_flag"] for row in episode_steps
                ])),
                "max_quality_step_rate": float(np.mean([
                    row["max_quality_flag"] for row in episode_steps
                ])),
            })
            rows.append(result)
            step_rows.extend(episode_steps)
            done += 1
            if done % 10 == 0 or done == total_planned:
                elapsed = time.time() - started
                print(
                    f"[DQN Dev180] {done}/{total_planned} episodes, "
                    f"{elapsed:.0f}s, last JointR={result['joint_reward']:.4f}",
                    flush=True,
                )
        if args.max_episodes is not None and done >= args.max_episodes:
            break

    overall = summarize_episodes(rows)
    by_scenario = {
        scenario: summarize_episodes([
            row for row in rows if row["scenario"] == scenario
        ])
        for scenario in SCENARIOS
    }
    by_band = {
        band: summarize_episodes([row for row in rows if row["band"] == band])
        for band in BANDS
    }

    baseline_rows = read_csv_rows(BASELINE_PER_EPISODE)
    baseline_overall = {}
    for strategy in BASELINE_STRATEGIES:
        selected = [row for row in baseline_rows if row["strategy"] == strategy]
        baseline_overall[strategy] = summarize_episodes(selected)
    paired = [
        paired_stats(rows, baseline_rows, strategy)
        for strategy in BASELINE_STRATEGIES
    ]
    relationships = [
        episode_relationship(rows, baseline_rows, strategy)
        for strategy in BASELINE_STRATEGIES
    ]

    step_array = {
        "e2e_minus_min_ms": np.asarray(
            [row["e2e_minus_min_ms"] for row in step_rows], dtype=np.float64
        ),
        "quality_minus_max": np.asarray(
            [row["quality_minus_max"] for row in step_rows], dtype=np.float64
        ),
        "fastest_flag": np.asarray(
            [row["fastest_flag"] for row in step_rows], dtype=np.float64
        ),
        "max_quality_flag": np.asarray(
            [row["max_quality_flag"] for row in step_rows], dtype=np.float64
        ),
    }
    step_summary = {
        "steps": len(step_rows),
        "fastest_step_rate": float(step_array["fastest_flag"].mean()),
        "max_quality_step_rate": float(step_array["max_quality_flag"].mean()),
        "mean_e2e_minus_min_ms": float(
            step_array["e2e_minus_min_ms"].mean()
        ),
        "median_e2e_minus_min_ms": float(
            np.median(step_array["e2e_minus_min_ms"])
        ),
        "p95_e2e_minus_min_ms": float(
            np.percentile(step_array["e2e_minus_min_ms"], 95)
        ),
        "mean_quality_minus_max": float(
            step_array["quality_minus_max"].mean()
        ),
        "median_quality_minus_max": float(
            np.median(step_array["quality_minus_max"])
        ),
        "model_usage": overall["model_usage"],
        "model_usage_ratio": overall["model_usage_ratio"],
    }

    summary = {
        "version": "stage2k-dqn-dev180-v1",
        "created_utc": utc_now(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_episode": int(payload.get("episode", 0)),
        "checkpoint_best_joint": float(payload.get("best_joint", 0.0)),
        "checkpoint_sha256": __import__("hashlib").sha256(
            Path(args.checkpoint).read_bytes()
        ).hexdigest(),
        "episodes": len(rows),
        "overall": overall,
        "by_scenario": by_scenario,
        "by_band": by_band,
        "baseline_overall": baseline_overall,
        "step_summary": step_summary,
        "paired": paired,
        "relationships": relationships,
        "elapsed_s": time.time() - started,
        "final_test_read": False,
    }
    atomic_write_json(output_dir / "summary.json", summary)

    per_episode_fields = [
        "episode_uid", "scenario", "band", "seed", "success", "osr", "ndtw",
        "spl", "steps", "mean_e2e_ms", "p95_e2e_ms", "delay_violation_rate",
        "joint_reward", "mean_navigation_reward", "mean_latency_penalty",
        "fallback_steps", "model_usage", "mean_e2e_minus_min_ms",
        "median_e2e_minus_min_ms", "p95_e2e_minus_min_ms",
        "mean_quality_minus_max", "median_quality_minus_max",
        "fastest_step_rate", "max_quality_step_rate",
    ]
    per_episode_rows = []
    for row in rows:
        per_episode_rows.append({
            "episode_uid": row["episode_uid"],
            "scenario": row["scenario"],
            "band": row["band"],
            "seed": row["seed"],
            "success": int(row["success"]),
            "osr": int(row["osr"]),
            "ndtw": row["ndtw"],
            "spl": row["spl"],
            "steps": row["steps"],
            "mean_e2e_ms": row["mean_e2e_ms"],
            "p95_e2e_ms": row["p95_e2e_ms"],
            "delay_violation_rate": row["delay_violation_rate"],
            "joint_reward": row["joint_reward"],
            "mean_navigation_reward": row["mean_navigation_reward"],
            "mean_latency_penalty": row["mean_latency_penalty"],
            "fallback_steps": row["fallback_steps"],
            "model_usage": json.dumps(row["model_usage"], sort_keys=True),
            "mean_e2e_minus_min_ms": row["mean_e2e_minus_min_ms"],
            "median_e2e_minus_min_ms": row["median_e2e_minus_min_ms"],
            "p95_e2e_minus_min_ms": row["p95_e2e_minus_min_ms"],
            "mean_quality_minus_max": row["mean_quality_minus_max"],
            "median_quality_minus_max": row["median_quality_minus_max"],
            "fastest_step_rate": row["fastest_step_rate"],
            "max_quality_step_rate": row["max_quality_step_rate"],
        })
    write_csv(output_dir / "per_episode.csv", per_episode_rows,
              per_episode_fields)
    write_csv(output_dir / "steps.csv", step_rows)
    write_markdown(
        output_dir / "STAGE2K_DQN_DEV180.md",
        summary, step_summary, paired, relationships,
    )
    print(json.dumps({
        "overall": overall,
        "step_summary": step_summary,
        "paired": paired,
        "relationships": relationships,
        "elapsed_s": summary["elapsed_s"],
    }, indent=2), flush=True)
    print("STAGE2K_DQN_DEV180_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
