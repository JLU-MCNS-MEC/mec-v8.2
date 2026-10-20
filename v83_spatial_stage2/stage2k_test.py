"""Stage-2k Association Final Test 180 (six policies, frozen once).

The test episode/topology list is independent of Train, Dev Validation and
Stage-2j Final Test 300.  The DQN checkpoint and all evaluation sources are
hashed into a freeze file before the first test rollout.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v83_spatial_stage2.stage2k_common import (
    FINAL_TEST_DIR,
    MODEL_NAMES,
    QUALITY_PRIOR_PATH,
    SCENARIOS,
    SCENARIO_TITLES,
    SPLITS,
    TOPOLOGY_SEED_BASE,
    VISUAL_PCA_PATH,
    atomic_write_json,
    bootstrap_ci,
    build_topology,
    joint_reward,
    load_progress,
    load_split_specs,
    read_json,
    save_topology,
    seed_isolation_report,
    sha256_file,
    summarize_episodes,
    update_progress,
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


STRATEGIES = ("dqn", "random", "min_latency", "max_quality", "heuristic", "oracle")
STRATEGY_TITLES = {
    "dqn": "DQN",
    "random": "Random",
    "min_latency": "Min-Latency",
    "max_quality": "Max-Quality",
    "heuristic": "Weighted Greedy",
    "oracle": "Oracle",
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
FREEZE_PATH = FINAL_TEST_DIR / "eval_freeze.json"
FIGURES_DIR = Path("v83_spatial_stage2/figures/stage2k")

SOURCE_FILES = (
    "v83_spatial_stage2/stage2k_common.py",
    "v83_spatial_stage2/stage2k_latency.py",
    "v83_spatial_stage2/stage2k_state.py",
    "v83_spatial_stage2/stage2k_runtime.py",
    "v83_spatial_stage2/stage2k_dqn.py",
    "v83_spatial_stage2/stage2k_baselines.py",
    "v83_spatial_stage2/stage2k_test.py",
    "v83_spatial_stage2/test_stage2k.py",
)


def build_freeze(quality, specs_payload, topologies, checkpoint):
    files = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = sha256_file(path)
    payload = {
        "version": "stage2k-final-test-freeze-v1",
        "created_utc": utc_now(),
        "final_test_read": True,
        "test_episodes": 180,
        "test_spec_sha256": sha256_file(
            PROJECT_ROOT / "runs/v83_spatial_stage2/stage2k/specs/"
            "test_episodes.json"
        ),
        "quality_prior_sha256": sha256_file(QUALITY_PRIOR_PATH),
        "visual_pca_sha256": sha256_file(VISUAL_PCA_PATH),
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
            "rollout.  No weight, threshold or source change is allowed after."
        ),
    }
    if FREEZE_PATH.is_file():
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        for key in (
            "test_spec_sha256", "quality_prior_sha256",
            "visual_pca_sha256", "dqn_checkpoint_sha256", "source_sha256",
            "topology_sha256",
        ):
            if existing.get(key) != payload.get(key):
                raise RuntimeError(
                    f"Evaluation freeze mismatch for {key}; refusing test"
                )
        return existing
    atomic_write_json(FREEZE_PATH, payload)
    return payload


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


def plot_bars(summary, metric_key, title, ylabel, path, lower_better=False):
    values = [summary["overall"][strategy][metric_key] for strategy in STRATEGIES]
    ci = [
        summary["bootstrap_95ci"][strategy][metric_key] for strategy in STRATEGIES
    ]
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
    for bar, value, low, high in zip(bars, values, lows, highs):
        label = f"{value:.4f}" if metric_key in ("nDTW", "JointR") else (
            f"{100*value:.2f}%" if metric_key == "SR" else f"{value:.1f}"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    progress = load_progress()
    if progress["final_test"]["status"] == "completed" and not args.force:
        print("Final test already completed; skipping.")
        return
    if progress["dqn"]["status"] != "completed":
        raise SystemExit("DQN training is not complete")
    if progress["baseline"]["status"] != "completed":
        raise SystemExit("Baseline is not complete")

    quality_payload = read_json(QUALITY_PRIOR_PATH)
    quality = flat_quality(quality_payload)
    specs_payload = load_split_specs("test")
    topologies = {}
    for scenario in SCENARIOS:
        path = (
            PROJECT_ROOT / "runs/v83_spatial_stage2/stage2k/topologies"
            / f"test_{scenario}.json"
        )
        topologies[scenario] = read_json(path)
        if topologies[scenario] is None:
            topologies[scenario] = save_topology(
                build_topology(scenario, "test", quality_payload)
            )
            topologies[scenario] = read_json(topologies[scenario])

    spec_payloads = {
        "train": load_split_specs("train"),
        "dev": load_split_specs("dev"),
        "test": specs_payload,
    }
    topology_payloads = {}
    for split in SPLITS:
        for scenario in SCENARIOS:
            path = (
                PROJECT_ROOT / "runs/v83_spatial_stage2/stage2k/topologies"
                / f"{split}_{scenario}.json"
            )
            payload = read_json(path)
            if payload is not None:
                topology_payloads[(split, scenario)] = payload
    report = seed_isolation_report(spec_payloads, topology_payloads)
    if not report["ok"]:
        raise SystemExit(f"Test seed overlap: {report['overlaps']}")

    checkpoint = Path(progress["dqn"]["best_checkpoint"])
    if not checkpoint.is_file():
        raise SystemExit(f"Missing DQN checkpoint {checkpoint}")
    freeze = build_freeze(
        quality_payload, specs_payload, topologies, checkpoint
    )

    device, encoder, backends = load_runtime(args.device)
    projector = VisualProjector.load(VISUAL_PCA_PATH)
    gab_config = load_gab_config()
    agent = DQNAgent(device)
    agent.load(checkpoint)
    agent.online.eval()

    FINAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    per_episode_path = FINAL_TEST_DIR / "per_episode.csv"
    steps_path = FINAL_TEST_DIR / "steps.csv"
    if args.force:
        for path in (per_episode_path, steps_path,
                     FINAL_TEST_DIR / "summary.json",
                     FINAL_TEST_DIR / "summary.csv"):
            path.unlink(missing_ok=True)
    existing_rows = read_csv_rows(per_episode_path)
    done_pairs = {
        (row["episode_uid"], row["strategy"]) for row in existing_rows
    }
    step_fieldnames = None
    if steps_path.is_file() and steps_path.stat().st_size > 0:
        with steps_path.open(encoding="utf-8") as handle:
            step_fieldnames = next(csv.reader(handle), None)
    rows = list(existing_rows)
    step_buffer = []
    episodes = [
        spec for scenario in SCENARIOS
        for spec in specs_payload["scenarios"][scenario]["episodes"]
    ]
    if len(episodes) != 180:
        raise RuntimeError(f"Expected 180 test episodes, got {len(episodes)}")
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
                np.random.default_rng(int(spec["seed"]) + RANDOM_SEED_OFFSET)
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
            )
            if strategy == "dqn":
                result = run_dqn_episode(runner, agent)
            else:
                result = runner.run(strategy, rng=rng)
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
                f"[Final Test] 已完成 {index + 1}/{len(episodes)} 个 Episode，"
                f"{time.time() - started:.0f}s",
                flush=True,
            )
    write_csv(per_episode_path, rows)

    for strategy in STRATEGIES:
        count = sum(1 for row in rows if row["strategy"] == strategy)
        if count != 180:
            raise RuntimeError(f"{strategy}: {count} test episodes != 180")

    overall = {
        strategy: summarize_episodes(
            [row for row in rows if row["strategy"] == strategy]
        )
        for strategy in STRATEGIES
    }
    bootstrap = {}
    for strategy in STRATEGIES:
        subset = [row for row in rows if row["strategy"] == strategy]
        bootstrap[strategy] = {
            "SR": bootstrap_ci([row["success"] for row in subset]),
            "nDTW": bootstrap_ci([row["ndtw"] for row in subset]),
            "E2E_ms": bootstrap_ci([row["mean_e2e_ms"] for row in subset]),
            "JointR": bootstrap_ci([row["joint_reward"] for row in subset]),
        }
    paired = {}
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode_uid"], {})[row["strategy"]] = row
    deltas = {"SR": [], "nDTW": [], "E2E_ms": [], "JointR": []}
    for uid, item in by_episode.items():
        if "dqn" not in item or "heuristic" not in item:
            raise RuntimeError(f"Missing paired rows for {uid}")
        deltas["SR"].append(item["dqn"]["success"] - item["heuristic"]["success"])
        deltas["nDTW"].append(item["dqn"]["ndtw"] - item["heuristic"]["ndtw"])
        deltas["E2E_ms"].append(
            item["dqn"]["mean_e2e_ms"] - item["heuristic"]["mean_e2e_ms"]
        )
        deltas["JointR"].append(
            item["dqn"]["joint_reward"] - item["heuristic"]["joint_reward"]
        )
    for key, values in deltas.items():
        paired[key] = bootstrap_ci(values)

    summary = {
        "version": "stage2k-final-test-v1",
        "created_utc": utc_now(),
        "split": "association_final_test",
        "episodes": 180,
        "bands": {
            band: sum(
                1 for row in rows
                if row["strategy"] == "dqn" and row["band"] == band
            )
            for band in ("short", "medium", "long")
        },
        "strategies": list(STRATEGIES),
        "random_seed_offset": RANDOM_SEED_OFFSET,
        "oracle_note": (
            "Oracle is a privileged upper bound; it is not a deployable policy "
            "and was not used for DQN training or state construction."
        ),
        "freeze": freeze,
        "overall": overall,
        "bootstrap_95ci": bootstrap,
        "paired_dqn_minus_heuristic": paired,
        "elapsed_s": time.time() - started,
        "final_test_read": True,
    }
    atomic_write_json(FINAL_TEST_DIR / "summary.json", summary)
    atomic_write_json(FINAL_TEST_DIR / "bootstrap_95ci.json", {
        "version": "stage2k-episode-bootstrap-95ci-v1",
        "n_boot": 10_000,
        "seed": 20261031,
        "bootstrap": bootstrap,
        "paired_dqn_minus_heuristic": paired,
    })
    summary_csv = []
    for strategy in STRATEGIES:
        item = overall[strategy]
        summary_csv.append({
            "strategy": strategy,
            "episodes": item["episodes"],
            "SR": item["SR"],
            "nDTW": item["nDTW"],
            "E2E_ms": item["E2E_ms"],
            "P95_ms": item["P95_ms"],
            "delay_violation_rate": item["delay_violation_rate"],
            "mean_steps": item["mean_steps"],
            "JointR": item["JointR"],
        })
    write_csv(FINAL_TEST_DIR / "summary.csv", summary_csv)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_bars(
        summary, "SR", "Association Final Test 180 — Success Rate",
        "SR", FIGURES_DIR / "ASSOCIATION_SR.png",
    )
    plot_bars(
        summary, "nDTW", "Association Final Test 180 — nDTW",
        "mean nDTW", FIGURES_DIR / "ASSOCIATION_nDTW.png",
    )
    plot_bars(
        summary, "E2E_ms", "Association Final Test 180 — E2E Delay",
        "mean E2E (ms)", FIGURES_DIR / "ASSOCIATION_E2E_Delay.png",
        lower_better=True,
    )
    plot_bars(
        summary, "JointR", "Association Final Test 180 — JointR",
        "JointR", FIGURES_DIR / "ASSOCIATION_JointR.png",
    )
    update_progress(
        "final_test",
        status="completed",
        summary_path=str(FINAL_TEST_DIR / "summary.json"),
        completed_utc=utc_now(),
    )
    print(json.dumps({
        "output_dir": str(FINAL_TEST_DIR),
        "overall": overall,
        "bootstrap": bootstrap,
        "paired_dqn_minus_heuristic": paired,
        "elapsed_s": summary["elapsed_s"],
    }, indent=2), flush=True)
    print("STAGE2K_FINAL_TEST_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
