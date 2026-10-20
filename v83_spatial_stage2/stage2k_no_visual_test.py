"""Stage-2k no-visual / pure-MEC DQN: Association Final Test 180 (one-shot).

Use ``--state-mode no_visual`` for the no-visual run and
``--state-mode pure_mec`` for the pure-MEC run.  Each run writes only under
its own ``runs/v83_spatial_stage2/...`` directory and never touches the
historical ``runs/v83_spatial_stage2/stage2k/final_test`` results.

Protocol
--------
* The DQN checkpoint is frozen and hashed together with the source files,
  test specs, test topologies, quality prior and visual PCA before the first
  rollout.
* Six policies are evaluated on exactly the same 180 Association Final Test
  episodes, topologies, scenes and unified latency model: the no-visual DQN
  and the five baselines (Random, Min-Latency, Max-Quality, Heuristic,
  Oracle).
* The Stage-2j Final Test 300 token directory is never opened.
* ``--split dev --max-episodes N`` is a script smoke mode that reads only Dev
  data and writes to an explicit scratch directory.  The real test run uses
  ``--split test`` (the default).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v83_spatial_stage2.stage2k_common import (
    MODEL_NAMES,
    QUALITY_PRIOR_PATH,
    SCENARIOS,
    SCENARIO_TITLES,
    SPLITS,
    atomic_write_json,
    bootstrap_ci,
    load_split_specs,
    load_topology,
    read_json,
    seed_isolation_report,
    sha256_file,
    summarize_episodes,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_dqn import flat_quality
from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2
from v83_spatial_stage2.stage2k_no_visual_state import STATE_DIM_NO_VISUAL
from v83_spatial_stage2.stage2k_pure_mec_state import STATE_DIM_PURE_MEC
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    build_scene,
    load_gab_config,
    load_runtime,
)


RUN_DIR = Path("runs/v83_spatial_stage2/stage2k_no_visual")
PURE_MEC_RUN = Path("runs/v83_spatial_stage2/stage2k_pure_mec_state")
MEC_CHECKPOINT_DIR = Path("runs/v83_spatial_stage2/stage2j")

STRATEGIES = (
    "dqn", "random", "min_latency", "max_quality", "heuristic", "oracle",
)
STRATEGY_TITLES = {
    "dqn": "DQN (no visual)",
    "random": "Random",
    "min_latency": "Min-Latency",
    "max_quality": "Max-Quality",
    "heuristic": "Heuristic",
    "oracle": "Oracle (privileged)",
}
STRATEGY_COLORS = {
    "dqn": "#2a78d6",
    "random": "#eb6834",
    "min_latency": "#1baf7a",
    "max_quality": "#eda100",
    "heuristic": "#e87ba4",
    "oracle": "#9b9a95",
}
RANDOM_SEED_OFFSET = 1_000_003
EXPECTED_TEST_EPISODES = 180

SOURCE_FILES = (
    "config.py",
    "env.py",
    "latency_model.py",
    "navigation_reward.py",
    "gbs_model_v3.py",
    "token_data.py",
    "utils.py",
    "encoder.py",
    "v83_spatial_stage2/stage2i_data.py",
    "v83_spatial_stage2/stage2j_model.py",
    "v83_spatial_stage2/stage2j_eval.py",
    "v83_spatial_stage2/stage2k_common.py",
    "v83_spatial_stage2/stage2k_latency.py",
    "v83_spatial_stage2/stage2k_jointr_reward.py",
    "v83_spatial_stage2/stage2k_no_visual_state.py",
    "v83_spatial_stage2/stage2k_pure_mec_state.py",
    "v83_spatial_stage2/stage2k_state.py",
    "v83_spatial_stage2/stage2k_runtime.py",
    "v83_spatial_stage2/stage2k_dqn.py",
    "v83_spatial_stage2/stage2k_jointr_v2_agent.py",
    "v83_spatial_stage2/stage2k_no_visual_test.py",
    "v83_spatial_stage2/test_stage2k_no_visual.py",
)


def read_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def result_row(result):
    return {
        "strategy": result["strategy"],
        "episode_uid": result["episode_uid"],
        "scenario": result["scenario"],
        "seed": result["seed"],
        "band": result["band"],
        "success": int(result["success"]),
        "osr": int(result["osr"]),
        "ndtw": result["ndtw"],
        "spl": result["spl"],
        "steps": result["steps"],
        "mean_e2e_ms": result["mean_e2e_ms"],
        "p95_e2e_ms": result["p95_e2e_ms"],
        "delay_violation_rate": result["delay_violation_rate"],
        "joint_reward": result["joint_reward"],
        "mean_navigation_reward": result["mean_navigation_reward"],
        "mean_latency_penalty": result["mean_latency_penalty"],
        "fallback_steps": result["fallback_steps"],
        "model_usage": json.dumps(result["model_usage"], sort_keys=True),
        "oracle_privileged": int(result["strategy"] == "oracle"),
    }


def step_rows(result):
    return [
        {
            "strategy": result["strategy"],
            "episode_uid": result["episode_uid"],
            "scenario": result["scenario"],
            "band": result["band"],
            **row,
        }
        for row in result["step_records"]
    ]


def run_dqn_episode(runner, agent):
    while True:
        action = agent.act_greedy(
            runner.step_data["state"], runner.step_data["mask"]
        )
        transition = runner.apply(action)
        if transition["done"]:
            break
    return runner.finalize("dqn")


def build_freeze(
    split, checkpoint, state_mode="no_visual",
    state_dim=STATE_DIM_NO_VISUAL,
):
    if split != "test":
        return None
    files = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = sha256_file(path)
    payload = {
        "version": "stage2k-no-visual-final-test-freeze-v1",
        "created_utc": utc_now(),
        "final_test_read": True,
        "state_dim": int(state_dim),
        "state_mode": str(state_mode),
        "test_episodes": EXPECTED_TEST_EPISODES,
        "test_spec_sha256": sha256_file(
            PROJECT_ROOT / "runs/v83_spatial_stage2/stage2k/specs/"
            "test_episodes.json"
        ),
        "quality_prior_sha256": sha256_file(QUALITY_PRIOR_PATH),
        "mec_checkpoint_sha256": {
            name: sha256_file(
                MEC_CHECKPOINT_DIR / name / "best.pt"
            )
            for name in MODEL_NAMES
        },
        "dqn_checkpoint": str(checkpoint),
        "dqn_checkpoint_sha256": sha256_file(checkpoint),
        "source_sha256": files,
        "topology_sha256": {
            scenario: sha256_file(
                PROJECT_ROOT / "runs/v83_spatial_stage2/stage2k/topologies"
                / f"test_{scenario}.json"
            )
            for scenario in SCENARIOS
        },
        "forbidden_test_data": "Stage-2j Final Test 300 token directory",
        "note": (
            "This freeze is written before the first Association Final Test "
            "rollout.  No weight, threshold or source change is allowed "
            "after it."
        ),
    }
    return payload


def verify_freeze(path, payload):
    if not path.is_file():
        atomic_write_json(path, payload)
        return payload
    existing = read_json(path)
    for key in (
        "test_spec_sha256", "quality_prior_sha256",
        "mec_checkpoint_sha256", "dqn_checkpoint_sha256", "source_sha256",
        "topology_sha256", "state_dim", "state_mode",
    ):
        if existing.get(key) != payload.get(key):
            raise RuntimeError(
                f"Evaluation freeze mismatch for {key}; refusing test"
            )
    return existing


def plot_metric(summary, metric_key, title, ylabel, path, lower_better=False):
    values = [summary["overall"][s][metric_key] for s in STRATEGIES]
    ci = [summary["bootstrap_95ci"][s][metric_key] for s in STRATEGIES]
    lows = [item["mean"] - item["low"] for item in ci]
    highs = [item["high"] - item["mean"] for item in ci]
    figure, axis = plt.subplots(figsize=(9.2, 4.4), constrained_layout=True)
    positions = np.arange(len(STRATEGIES))
    bars = axis.bar(
        positions, values, width=0.66,
        color=[STRATEGY_COLORS[s] for s in STRATEGIES],
        edgecolor="none", zorder=2,
    )
    axis.errorbar(
        positions, values, yerr=np.asarray([lows, highs]), fmt="none",
        ecolor="#52514e", elinewidth=1.0, capsize=3, zorder=3,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [STRATEGY_TITLES[s] for s in STRATEGIES], fontsize=8.5
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=11)
    axis.grid(axis="y", color="#e4e3df", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    span = max((abs(value) for value in values), default=1.0) or 1.0
    for bar, value, high in zip(bars, values, highs):
        label = (
            f"{value:.4f}" if metric_key in ("nDTW", "JointR")
            else (f"{100*value:.2f}%" if metric_key == "SR"
                  else f"{value:.1f}")
        )
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + high + 0.01 * span,
            label, ha="center", va="bottom", fontsize=7.5,
        )
    figure.text(
        0.5, -0.02,
        "Bars: 180 Association Final Test episodes; whiskers: 10,000-episode "
        "bootstrap 95% CI. Oracle is a privileged upper bound."
        + (" Lower is better." if lower_better else ""),
        ha="center", fontsize=7.5, color="#52514e",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_four_panel(summary, path):
    panels = (
        ("SR", "Success rate (SR)", True),
        ("nDTW", "nDTW", False),
        ("E2E_ms", "Mean E2E delay (ms)", False),
        ("JointR", "JointR", False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 8.2),
                                constrained_layout=True)
    positions = np.arange(len(STRATEGIES))
    for axis, (key, ylabel, percent) in zip(axes.reshape(-1), panels):
        values = [summary["overall"][s][key] for s in STRATEGIES]
        axis.bar(
            positions, values, width=0.66,
            color=[STRATEGY_COLORS[s] for s in STRATEGIES],
            edgecolor="none", zorder=2,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [STRATEGY_TITLES[s] for s in STRATEGIES],
            fontsize=7.5, rotation=20,
        )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#e4e3df", linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        if percent:
            axis.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{100*x:.0f}%")
            )
    figure.suptitle(
        f"Association Final Test 180 — {STRATEGY_TITLES['dqn']} "
        "vs five baselines",
        fontsize=11.5,
    )
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(Path(path).with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def paired_deltas(rows, strategy):
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode_uid"], {})[row["strategy"]] = row
    deltas = {"SR": [], "nDTW": [], "E2E_ms": [], "JointR": []}
    wins = losses = ties = 0
    for uid, item in by_episode.items():
        if "dqn" not in item or strategy not in item:
            raise RuntimeError(f"Missing paired rows for {uid}")
        for key, field, dqn_field in (
            ("SR", "success", "success"),
            ("nDTW", "ndtw", "ndtw"),
            ("E2E_ms", "mean_e2e_ms", "mean_e2e_ms"),
            ("JointR", "joint_reward", "joint_reward"),
        ):
            deltas[key].append(
                float(item["dqn"][dqn_field])
                - float(item[strategy][field])
            )
        joint = (
            float(item["dqn"]["joint_reward"])
            - float(item[strategy]["joint_reward"])
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
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }
    for key, values in deltas.items():
        output[f"delta_{key}"] = bootstrap_ci(values)
    return output


def markdown(path, summary, paired, split):
    label = (
        "No-Visual" if summary["state_mode"] == "no_visual"
        else "Pure MEC"
    )
    title = (
        f"Stage-2k {label} DQN — Association Final Test 180"
        if split == "test" else
        f"Stage-2k {label} DQN — Dev script smoke"
    )
    lines = [
        f"# {title}",
        "",
        f"- 日期：{summary['created_utc']}",
        f"- Checkpoint：`{summary['checkpoint']}`",
        f"- Episode：{summary['checkpoint_episode']}",
        f"- 状态：`{summary['state_mode']}`，{summary['state_dim']} 维",
        "- 六种策略共享同一 Episode 清单、拓扑、场景和统一时延模型。",
        "",
        "## 总体指标",
        "",
        "| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | 违约率 | 步数 | JointR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        item = summary["overall"][strategy]
        lines.append(
            f"| {STRATEGY_TITLES[strategy]} | {100*item['SR']:.2f}% | "
            f"{item['nDTW']:.4f} | {item['E2E_ms']:.2f} | "
            f"{item['P95_ms']:.2f} | "
            f"{100*item['delay_violation_rate']:.2f}% | "
            f"{item['mean_steps']:.1f} | {item['JointR']:.4f} |"
        )
    lines += [
        "",
        "JointR = mean(nDTW) − 0.25 × mean(E2E_ms) / 100。",
        "",
        "## 逐 Episode 配对（DQN − Baseline）",
        "",
        "| Baseline | n | ΔSR | ΔnDTW | ΔE2E (ms) | ΔJointR [95% CI] | 胜/负/平 |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for item in paired:
        lines.append(
            f"| {STRATEGY_TITLES[item['strategy']]} | {item['n']} | "
            f"{item['delta_SR']['mean']:+.4f} | "
            f"{item['delta_nDTW']['mean']:+.4f} | "
            f"{item['delta_E2E_ms']['mean']:+.2f} | "
            f"{item['delta_JointR']['mean']:+.4f} "
            f"[{item['delta_JointR']['low']:+.4f}, "
            f"{item['delta_JointR']['high']:+.4f}] | "
            f"{item['wins']}/{item['losses']}/{item['ties']} |"
        )
    lines += [
        "",
        "## 分部署条件（DQN）",
        "",
        "| 场景 | SR | nDTW | E2E (ms) | JointR |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        item = summary["by_scenario"][scenario]["dqn"]
        lines.append(
            f"| {SCENARIO_TITLES[scenario]} | {100*item['SR']:.2f}% | "
            f"{item['nDTW']:.4f} | {item['E2E_ms']:.2f} | "
            f"{item['JointR']:.4f} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- Random 使用每个 Episode 固定种子（episode seed + 1,000,003）。",
        "- Oracle 是特权信息上界，不参与 DQN 训练、状态构造或模型选择。",
        "- 本轮只读取 Association Final Test 180；未读取 Stage-2j Final "
        "Test 300 token 目录。",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument(
        "--state-mode", choices=("no_visual", "pure_mec"),
        default="no_visual",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split = str(args.split)
    no_visual = args.state_mode == "no_visual"
    pure_mec = args.state_mode == "pure_mec"
    state_dim = (
        STATE_DIM_NO_VISUAL if no_visual else STATE_DIM_PURE_MEC
    )
    STRATEGY_TITLES["dqn"] = (
        "DQN (no visual)" if no_visual else "DQN (pure MEC)"
    )
    run_dir = args.run_dir or (
        PURE_MEC_RUN if pure_mec else RUN_DIR
    )
    checkpoint = args.checkpoint or (run_dir / "checkpoints" / "best.pt")
    output_dir = args.output_dir or (
        run_dir / "final_test" if split == "test"
        else run_dir.parent / f"{run_dir.name}_script_smoke"
    )
    if (output_dir / "summary.json").is_file() and not args.force:
        raise SystemExit(
            f"{output_dir}/summary.json exists; refusing to overwrite"
        )
    if split == "test" and args.max_episodes is not None:
        raise SystemExit("Final Test must run all 180 episodes")
    if not checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint {checkpoint}")

    training_result = read_json(run_dir / "training_result.json") or {}
    if split == "test":
        if training_result.get("final_test_read", False):
            raise SystemExit("Training result already claims a Final Test read")
        if training_result.get("status") == "training":
            raise SystemExit("Training is not finished")
        if not training_result:
            raise SystemExit(
                f"Missing {run_dir / 'training_result.json'}"
            )
        if not bool(training_result.get("converged", False)):
            raise SystemExit(
                "Training did not converge; refusing Final Test"
            )
        expected_checkpoint = training_result.get("best_checkpoint")
        if (
            args.checkpoint is None
            and expected_checkpoint
            and Path(expected_checkpoint).resolve()
            != checkpoint.resolve()
        ):
            raise SystemExit(
                "Default checkpoint is not training_result.best_checkpoint"
            )

    quality = flat_quality(read_json(QUALITY_PRIOR_PATH))
    specs_payload = load_split_specs(split)
    topologies = {
        scenario: load_topology(scenario, split) for scenario in SCENARIOS
    }
    episodes = [
        spec for scenario in SCENARIOS
        for spec in specs_payload["scenarios"][scenario]["episodes"]
    ]
    if split == "test" and len(episodes) != EXPECTED_TEST_EPISODES:
        raise RuntimeError(f"Expected 180 test episodes, got {len(episodes)}")
    if args.max_episodes is not None:
        episodes = episodes[: int(args.max_episodes)]

    if split == "test":
        spec_payloads = {
            name: load_split_specs(name) for name in SPLITS
        }
        topology_payloads = {}
        for name in SPLITS:
            for scenario in SCENARIOS:
                path = (
                    PROJECT_ROOT / "runs/v83_spatial_stage2/stage2k/"
                    "topologies" / f"{name}_{scenario}.json"
                )
                payload = read_json(path)
                if payload is not None:
                    topology_payloads[(name, scenario)] = payload
        report = seed_isolation_report(spec_payloads, topology_payloads)
        if not report["ok"]:
            raise SystemExit(f"Test seed overlap: {report['overlaps']}")

    device, encoder, backends = load_runtime(args.device)
    projector = None
    gab_config = load_gab_config()
    agent = DQNAgentV2(
        device, state_dim=state_dim, state_mode=args.state_mode
    )
    payload = agent.load_model(checkpoint)
    config = payload.get("config", {})
    if config.get("reward_mode") != "jointr_aligned_v2":
        raise SystemExit("Checkpoint is not a JointR-aligned v2 model")
    if str(config.get("state_mode", "")) != args.state_mode:
        raise SystemExit("Checkpoint state_mode does not match --state-mode")
    if int(config.get("state_dim", -1)) != state_dim:
        raise SystemExit(
            f"Checkpoint state_dim is not {state_dim}"
        )
    agent.online.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    per_episode_path = output_dir / "per_episode.csv"
    steps_path = output_dir / "steps.csv"
    if args.force:
        for path in (
            per_episode_path, steps_path, output_dir / "summary.json",
            output_dir / "summary.csv",
        ):
            path.unlink(missing_ok=True)
    # The freeze is written before the first rollout and is never deleted by
    # --force: a rerun must use exactly the same checkpoint and sources.
    freeze = build_freeze(
        split, checkpoint, state_mode=args.state_mode,
        state_dim=state_dim,
    )
    if freeze is not None:
        freeze = verify_freeze(output_dir / "eval_freeze.json", freeze)
    existing_rows = read_csv_rows(per_episode_path)
    done_pairs = {
        (row["episode_uid"], row["strategy"]) for row in existing_rows
    }
    step_fieldnames = None
    if steps_path.is_file() and steps_path.stat().st_size > 0:
        with steps_path.open(encoding="utf-8") as handle:
            step_fieldnames = next(csv.reader(handle), None)
    rows = list(existing_rows)
    started = time.time()
    for index, spec in enumerate(episodes):
        needed = any(
            (spec["episode_uid"], strategy) not in done_pairs
            for strategy in STRATEGIES
        )
        if not needed:
            continue
        scene = build_scene(spec, encoder, device)
        topology = topologies[spec["scenario"]]
        new_step_rows = []
        for strategy in STRATEGIES:
            if (spec["episode_uid"], strategy) in done_pairs:
                continue
            rng = (
                np.random.default_rng(
                    int(spec["seed"]) + RANDOM_SEED_OFFSET
                )
                if strategy == "random" else None
            )
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
                jointr_aligned=True,
                no_visual_state=no_visual,
                pure_mec_state=pure_mec,
            )
            if strategy == "dqn":
                result = run_dqn_episode(runner, agent)
            else:
                result = runner.run(strategy, rng=rng)
            if result["jointr_abs_error"] > 1e-6:
                raise RuntimeError(
                    f"{result['episode_uid']} reward/JointR mismatch: "
                    f"{result['jointr_abs_error']}"
                )
            rows.append(result_row(result))
            new_step_rows.extend(step_rows(result))
            done_pairs.add((spec["episode_uid"], strategy))
        if new_step_rows:
            write_csv(per_episode_path, rows)
            append_csv(
                steps_path, new_step_rows,
                step_fieldnames or list(new_step_rows[0]),
            )
            step_fieldnames = step_fieldnames or list(new_step_rows[0])
        if (index + 1) % 10 == 0 or index + 1 == len(episodes):
            print(
                f"[{args.state_mode} Final Test] 已完成 {index + 1}/"
                f"{len(episodes)} 个 Episode，{time.time() - started:.0f}s",
                flush=True,
            )
    write_csv(per_episode_path, rows)

    for strategy in STRATEGIES:
        count = sum(1 for row in rows if row["strategy"] == strategy)
        if count != len(episodes):
            raise RuntimeError(
                f"{strategy}: {count} episodes != {len(episodes)}"
            )

    overall = {
        strategy: summarize_episodes(
            [row for row in rows if row["strategy"] == strategy]
        )
        for strategy in STRATEGIES
    }
    bootstrap = {
        strategy: {
            "SR": bootstrap_ci([
                row["success"] for row in rows
                if row["strategy"] == strategy
            ]),
            "nDTW": bootstrap_ci([
                row["ndtw"] for row in rows
                if row["strategy"] == strategy
            ]),
            "E2E_ms": bootstrap_ci([
                row["mean_e2e_ms"] for row in rows
                if row["strategy"] == strategy
            ]),
            "JointR": bootstrap_ci([
                row["joint_reward"] for row in rows
                if row["strategy"] == strategy
            ]),
        }
        for strategy in STRATEGIES
    }
    paired = [
        paired_deltas(rows, strategy)
        for strategy in STRATEGIES if strategy != "dqn"
    ]
    by_scenario = {
        scenario: {
            strategy: summarize_episodes([
                row for row in rows
                if row["scenario"] == scenario
                and row["strategy"] == strategy
            ])
            for strategy in STRATEGIES
        }
        for scenario in SCENARIOS
    }
    summary = {
        "version": "stage2k-no-visual-final-test-v1",
        "created_utc": utc_now(),
        "split": (
            "association_final_test" if split == "test"
            else "dev_script_smoke"
        ),
        "episodes": len(episodes),
        "state_mode": str(args.state_mode),
        "state_dim": int(state_dim),
        "checkpoint": str(checkpoint),
        "checkpoint_episode": int(payload.get("episode", 0)),
        "strategies": list(STRATEGIES),
        "random_seed_offset": RANDOM_SEED_OFFSET,
        "oracle_note": (
            "Oracle is a privileged upper bound; it is not a deployable "
            "policy and was not used for DQN training or state construction."
        ),
        "freeze": freeze,
        "overall": overall,
        "bootstrap_95ci": bootstrap,
        "paired_dqn_minus_baseline": paired,
        "by_scenario": by_scenario,
        "elapsed_s": time.time() - started,
        "final_test_read": bool(split == "test"),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "bootstrap_95ci.json", {
        "version": "stage2k-no-visual-episode-bootstrap-95ci-v1",
        "n_boot": 10_000,
        "seed": 20261031,
        "bootstrap": bootstrap,
        "paired_dqn_minus_baseline": paired,
    })
    summary_csv = [
        {
            "strategy": strategy,
            "episodes": overall[strategy]["episodes"],
            "SR": overall[strategy]["SR"],
            "nDTW": overall[strategy]["nDTW"],
            "E2E_ms": overall[strategy]["E2E_ms"],
            "P95_ms": overall[strategy]["P95_ms"],
            "delay_violation_rate": overall[strategy][
                "delay_violation_rate"
            ],
            "mean_steps": overall[strategy]["mean_steps"],
            "JointR": overall[strategy]["JointR"],
        }
        for strategy in STRATEGIES
    ]
    write_csv(output_dir / "summary.csv", summary_csv)
    markdown(output_dir / "STAGE2K_NO_VISUAL_FINAL_TEST.md", summary,
             paired, split)

    if split == "test":
        figure_dir = output_dir / "figures"
        plot_metric(
            summary, "SR", "Association Final Test 180 — Success Rate",
            "SR", figure_dir / "ASSOCIATION_SR.png",
        )
        plot_metric(
            summary, "nDTW", "Association Final Test 180 — nDTW",
            "mean nDTW", figure_dir / "ASSOCIATION_nDTW.png",
        )
        plot_metric(
            summary, "E2E_ms", "Association Final Test 180 — E2E Delay",
            "mean E2E (ms)", figure_dir / "ASSOCIATION_E2E_Delay.png",
            lower_better=True,
        )
        plot_metric(
            summary, "JointR", "Association Final Test 180 — JointR",
            "JointR", figure_dir / "ASSOCIATION_JointR.png",
        )
        plot_four_panel(
            summary, figure_dir / "ASSOCIATION_FOUR_METRICS.png"
        )

    if split == "test":
        progress = read_json(run_dir / "progress.json") or {}
        progress["final_test_read"] = True
        progress["final_test"] = {
            "status": "completed",
            "summary_path": str(output_dir / "summary.json"),
            "completed_utc": utc_now(),
        }
        progress["updated_utc"] = utc_now()
        atomic_write_json(run_dir / "progress.json", progress)

    print(json.dumps({
        "output_dir": str(output_dir),
        "overall": overall,
        "paired": paired,
        "elapsed_s": summary["elapsed_s"],
    }, indent=2), flush=True)
    print("STAGE2K_NO_VISUAL_FINAL_TEST_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
