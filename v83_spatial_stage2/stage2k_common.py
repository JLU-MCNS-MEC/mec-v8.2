"""Stage-2k shared constants, deployment scenarios, episode specs and metrics.

Stage-2k trains only a DQN association agent.  The five Stage-2j T5 MEC
checkpoints are frozen and reused; nothing in this module trains a MEC.

Data isolation
--------------
* Association train / dev / test use disjoint episode-seed ranges
  (5,000,000+, 6,000,000+, 7,000,000+), separate from Stage-2j Train
  (1,000,000+), Dev Val (2,000,000+) and Final Test (3,000,000+).
* Association topologies use 8,000,000+ seeds, separate from the Stage-2j
  network layout seeds (17,820,000+ / 34,820,000+ / 48,820,000+).
* The Stage-2j Final Test 300 token directory is never opened.  Only the
  already-derived spec metadata is used for a seed-overlap check.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import ENV, NETWORK, TRAIN
from env import STOP, ToyUAVEnv
from gbs_network import build_random_gbs_layout
from latency_model import LatencyModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE2K_ROOT = Path("runs/v83_spatial_stage2/stage2k")
SPECS_DIR = STAGE2K_ROOT / "specs"
TOPOLOGY_DIR = STAGE2K_ROOT / "topologies"
QUALITY_DIR = STAGE2K_ROOT / "quality_dev_val"
BASELINE_DIR = STAGE2K_ROOT / "baselines"
DQN_DIR = STAGE2K_ROOT / "dqn"
VALIDATION_DIR = STAGE2K_ROOT / "validation"
FINAL_TEST_DIR = STAGE2K_ROOT / "final_test"
TENSORBOARD_DIR = Path("runs/tensorboard/association_stage2k")
PROGRESS_PATH = STAGE2K_ROOT / "progress.json"
QUALITY_PRIOR_PATH = STAGE2K_ROOT / "quality_prior.json"
VISUAL_PCA_PATH = STAGE2K_ROOT / "visual_pca.npz"
DEPLOYMENT_SPEC_PATH = STAGE2K_ROOT / "deployment_spec.json"

MODEL_NAMES = (
    "Small",
    "Transformer-S",
    "Medium",
    "Transformer-M",
    "Large",
)
MODEL_INDEX = {name: index for index, name in enumerate(MODEL_NAMES)}
SCENARIOS = ("balanced", "quality_rich", "tradeoff_stress")
SCENARIO_TITLES = {
    "balanced": "Balanced / Original",
    "quality_rich": "Quality-rich",
    "tradeoff_stress": "Tradeoff-stress",
}
SPLITS = ("train", "dev", "test")
BANDS = ("short", "medium", "long")
BAND_ID = {band: index for index, band in enumerate(BANDS)}

# Episode seed ranges.  Each scenario has a 100k-wide disjoint block.
SEED_BASE = {"train": 5_000_000, "dev": 6_000_000, "test": 7_000_000}
SCENARIO_SEED_OFFSET = {
    "balanced": 0,
    "quality_rich": 100_000,
    "tradeoff_stress": 200_000,
}
# Topology seed ranges (separate from any Stage-2j layout seed).
TOPOLOGY_SEED_BASE = {"train": 8_000_000, "dev": 8_100_000, "test": 8_200_000}
TOPOLOGY_SCENARIO_OFFSET = {
    "balanced": 0,
    "quality_rich": 1_000,
    "tradeoff_stress": 2_000,
}

TRAIN_EPISODES = 4000
DEV_EPISODES_PER_SCENARIO = 60
TEST_EPISODES_PER_SCENARIO = 60
DEV_BAND_COUNTS = {"short": 18, "medium": 30, "long": 12}
DQN_VALIDATION_PER_SCENARIO = 20
DQN_VALIDATION_BAND_COUNTS = {"short": 6, "medium": 10, "long": 4}

STOP_THRESHOLD = 0.7
MAX_STEPS = int(ENV.max_steps)
TX_POWER_W = float(NETWORK.default_tx_power_w)
DEADLINE_MS = float(NETWORK.decision_latency_budget_s) * 1000.0
JOINTR_DELAY_WEIGHT = 0.25
REWARD_DELAY_WEIGHT = 0.25
REWARD_DELAY_DIVISOR = 200.0
REWARD_DELAY_CLIP = 3.0

BAND_RANGES = {
    str(item[0]): (float(item[2]), float(item[3]))
    for item in TRAIN.trajectory_mix
}
BAND_ACTION_RANGES = {
    str(item[0]): (int(item[4]), int(item[5]))
    for item in TRAIN.trajectory_mix
}

# Fixed normalization constants for candidate MEC features.
POSITION_SCALE_M = 500.0
BANDWIDTH_SCALE_HZ = float(NETWORK.max_bandwidth_hz)  # 30 MHz
COMPUTE_SCALE_FLOPS = 2.0e12
RATE_SCALE_BPS = 200.0e6
LATENCY_SCALE_MS = DEADLINE_MS
E2E_CLIP_SCALE = 3.0

ACTION_NAMES = ("Forward", "Left", "Right", "Stop")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    )
    try:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def reject_final_test_dataset(path):
    """Reject the sealed Stage-2j Final Test 300 token directory."""
    if "datasets/final_test_300" in str(path) or str(path).endswith(
        "final_test_300"
    ):
        raise ValueError(f"Stage-2j Final Test is sealed; refusing {path}")


# ---------------------------------------------------------------------------
# Progress bookkeeping
# ---------------------------------------------------------------------------


def _new_progress():
    return {
        "stage": "stage2k",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "final_test_read": False,
        "baseline": {
            "status": "pending",
            "episodes": 0,
            "summary_path": None,
            "completed_utc": None,
        },
        "dqn": {
            "status": "pending",
            "current_episode": 0,
            "best_validation_joint": None,
            "consecutive_no_improve": 0,
            "best_checkpoint": None,
            "last_checkpoint": None,
            "training_finished_utc": None,
            "converged": None,
        },
        "final_test": {
            "status": "pending",
            "summary_path": None,
            "completed_utc": None,
        },
        "artifacts": {},
    }


def load_progress(path=PROGRESS_PATH):
    payload = read_json(path, default=None)
    if payload is None:
        payload = _new_progress()
    for key, value in _new_progress().items():
        payload.setdefault(key, value)
    for section in ("baseline", "dqn", "final_test"):
        for key, value in _new_progress()[section].items():
            payload[section].setdefault(key, value)
    return payload


def update_progress(section, **fields):
    payload = load_progress()
    payload["updated_utc"] = utc_now()
    payload[section].update(fields)
    atomic_write_json(PROGRESS_PATH, payload)
    return payload


# ---------------------------------------------------------------------------
# Episode specs
# ---------------------------------------------------------------------------


def _inspect_candidate(env, seed, band_name):
    """Reuse the Stage-2j acceptance rule for a natural expert episode."""
    distance_range = BAND_RANGES[band_name]
    action_range = BAND_ACTION_RANGES[band_name]
    env.reset(
        seed=int(seed),
        goal_distance_range_m=distance_range,
        render_observation=False,
    )
    start_distance = float(env.distance_to_goal)
    actions = Counter()
    success = False
    done = False
    steps = 0
    while not done and steps < env.max_steps:
        action = int(env.oracle_action())
        actions[action] += 1
        steps += 1
        _, success, done = env.step(action, render_observation=False)
    accepted = bool(
        success
        and action_range[0] <= steps <= action_range[1]
    )
    return {
        "accepted": accepted,
        "seed": int(seed),
        "band": band_name,
        "steps": int(steps),
        "start_goal_distance_m": start_distance,
        "action_counts": dict(actions),
    }


def generate_episode_specs(
    split,
    scenario,
    band_counts,
    seed_start,
    episode_offset=0,
    max_candidates=100_000,
):
    """Generate accepted episodes for one split/scenario."""
    env = ToyUAVEnv()
    specs = []
    candidate_seed = int(seed_start)
    attempts = 0
    for band_name in BANDS:
        count = int(band_counts[band_name])
        found = 0
        while found < count:
            if attempts >= int(max_candidates):
                raise RuntimeError(
                    f"Unable to generate {count} {band_name} episodes for "
                    f"{split}/{scenario}"
                )
            info = _inspect_candidate(env, candidate_seed, band_name)
            candidate_seed += 1
            attempts += 1
            if not info["accepted"]:
                continue
            specs.append({
                "episode_uid": (
                    f"{split}-{scenario}-{episode_offset + len(specs):05d}"
                ),
                "split": split,
                "scenario": scenario,
                "episode_index": int(episode_offset + len(specs)),
                "seed": int(info["seed"]),
                "band": band_name,
                "band_id": int(BAND_ID[band_name]),
                "distance_range": list(BAND_RANGES[band_name]),
                "oracle_steps": int(info["steps"]),
                "start_goal_distance_m": float(
                    info["start_goal_distance_m"]
                ),
                "oracle_action_counts": info["action_counts"],
            })
            found += 1
    return specs


def train_band_counts(total=4000):
    """Equal thirds for the training schedule."""
    base = int(total) // len(BANDS)
    counts = {band: base for band in BANDS}
    counts[BANDS[-1]] += int(total) - base * len(BANDS)
    return counts


def dev_test_band_counts():
    return dict(DEV_BAND_COUNTS)


def build_split_specs(split):
    """Generate the complete train/dev/test spec lists (test is sealed)."""
    if split == "train":
        total = TRAIN_EPISODES
        per_scenario = []
        remaining = total
        for index, scenario in enumerate(SCENARIOS):
            count = total // len(SCENARIOS)
            if index == len(SCENARIOS) - 1:
                count = remaining
            remaining -= count
            per_scenario.append((scenario, count))
        payload = {}
        for scenario, count in per_scenario:
            counts = train_band_counts(count)
            seed_start = (
                SEED_BASE["train"] + SCENARIO_SEED_OFFSET[scenario]
            )
            payload[scenario] = generate_episode_specs(
                "train", scenario, counts, seed_start,
            )
        return payload
    if split in ("dev", "test"):
        payload = {}
        offset = 0
        for scenario in SCENARIOS:
            seed_start = (
                SEED_BASE[split] + SCENARIO_SEED_OFFSET[scenario]
            )
            payload[scenario] = generate_episode_specs(
                split, scenario, dev_test_band_counts(), seed_start,
                episode_offset=offset,
            )
            offset += len(payload[scenario])
        return payload
    raise ValueError(f"Unknown split {split}")


def specs_path(split):
    return SPECS_DIR / f"{split}_episodes.json"


def flatten_specs(specs_by_scenario):
    rows = []
    for scenario in SCENARIOS:
        rows.extend(specs_by_scenario[scenario])
    return rows


def save_split_specs(split, specs_by_scenario, sealed=False):
    payload = {
        "version": "stage2k-association-v1",
        "split": split,
        "created_utc": utc_now(),
        "sealed": bool(sealed),
        "final_test_read": False,
        "seed_base": SEED_BASE[split],
        "scenario_seed_offsets": SCENARIO_SEED_OFFSET,
        "band_counts_per_scenario": dev_test_band_counts()
        if split in ("dev", "test") else train_band_counts(
            len(specs_by_scenario[SCENARIOS[0]])
        ),
        "scenarios": {
            scenario: {
                "seeds": [
                    spec["seed"] for spec in specs_by_scenario[scenario]
                ],
                "episodes": specs_by_scenario[scenario],
            }
            for scenario in SCENARIOS
        },
        "counts": {
            scenario: len(specs_by_scenario[scenario])
            for scenario in SCENARIOS
        },
    }
    atomic_write_json(specs_path(split), payload)
    return payload


def load_split_specs(split):
    payload = read_json(specs_path(split))
    if payload is None:
        raise FileNotFoundError(f"Missing {specs_path(split)}")
    if payload.get("split") != split:
        raise ValueError(f"{specs_path(split)} is not a {split} spec")
    return payload


def all_specs(payload):
    return flatten_specs({
        scenario: payload["scenarios"][scenario]["episodes"]
        for scenario in SCENARIOS
    })


def validation_specs(dev_payload):
    """Fixed 20 episodes per scenario: 6 Short / 10 Medium / 4 Long."""
    rows = []
    for scenario in SCENARIOS:
        episodes = dev_payload["scenarios"][scenario]["episodes"]
        selected = []
        for band_name in BANDS:
            count = DQN_VALIDATION_BAND_COUNTS[band_name]
            band_rows = [row for row in episodes if row["band"] == band_name]
            selected.extend(band_rows[:count])
        if len(selected) != DQN_VALIDATION_PER_SCENARIO:
            raise RuntimeError(
                f"{scenario} validation has {len(selected)} episodes"
            )
        rows.extend(selected)
    if len(rows) != 60:
        raise RuntimeError(f"Expected 60 validation episodes, got {len(rows)}")
    return rows


# ---------------------------------------------------------------------------
# Deployment topologies
# ---------------------------------------------------------------------------


def quality_prior_vector(quality_prior):
    return np.asarray(
        [float(quality_prior["models"][name]["mean_ndtw"])
         for name in MODEL_NAMES],
        dtype=np.float32,
    )


def _quality_rank(quality_prior):
    values = quality_prior_vector(quality_prior)
    order = sorted(range(len(MODEL_NAMES)), key=lambda i: (-values[i], i))
    return [MODEL_NAMES[index] for index in order]


def _scenario_model_counts(quality_prior):
    rank = _quality_rank(quality_prior)
    top, low = rank[:3], rank[3:]
    counts = {name: 0 for name in MODEL_NAMES}
    for name in top:
        counts[name] = 5
    counts[low[0]] = 3
    counts[low[1]] = 2
    return counts


def _node_quality_prior(node, quality_prior):
    return float(quality_prior["models"][node.model_type]["mean_ndtw"])


def build_topology(scenario, split, quality_prior, verbose=False):
    """Build one deterministic 20-node deployment for a scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    seed = (
        TOPOLOGY_SEED_BASE[split] + TOPOLOGY_SCENARIO_OFFSET[scenario]
    )
    base = build_random_gbs_layout(seed=seed, verbose=verbose)
    rng = np.random.default_rng(seed ^ 0x2A5F2A5F)

    if scenario == "balanced":
        model_types = [node.model_type for node in base]
        computes = [node.compute_capacity_flops_per_s for node in base]
        bandwidths = [node.bandwidth_hz for node in base]
    else:
        counts = _scenario_model_counts(quality_prior)
        model_types = []
        for name in MODEL_NAMES:
            model_types.extend([name] * counts[name])
        rng.shuffle(model_types)
        rank = _quality_rank(quality_prior)
        top = set(rank[:3])
        if scenario == "quality_rich":
            top_compute, low_compute = 2.0e12, 1.0e12
            top_bandwidth, low_bandwidth = 30.0e6, 10.0e6
        elif scenario == "tradeoff_stress":
            top_compute, low_compute = 0.5e12, 2.0e12
            top_bandwidth, low_bandwidth = 5.0e6, 30.0e6
        else:
            raise ValueError(scenario)
        computes = [
            top_compute if name in top else low_compute
            for name in model_types
        ]
        bandwidths = [
            top_bandwidth if name in top else low_bandwidth
            for name in model_types
        ]

    nodes = []
    for index, (x, y) in enumerate(
        [(float(node.x), float(node.y)) for node in base]
    ):
        name = model_types[index]
        nodes.append({
            "node_id": f"G{index + 1}",
            "x": x,
            "y": y,
            "model_type": name,
            "model_index": int(MODEL_INDEX[name]),
            "quality_prior": float(
                quality_prior["models"][name]["mean_ndtw"]
            ),
            "compute_capacity_flops_per_s": float(computes[index]),
            "bandwidth_hz": float(bandwidths[index]),
            "coverage_radius_m": float(NETWORK.coverage_radius_m),
        })
    if len(nodes) != 20:
        raise RuntimeError(f"Expected 20 nodes, got {len(nodes)}")
    return {
        "scenario": scenario,
        "split": split,
        "topology_seed": int(seed),
        "nodes": nodes,
        "source_layout": "build_random_gbs_layout + scenario override",
        "quality_prior_source": "Dev Val 180 mean nDTW (Stage-2j models)",
        "final_test_read": False,
    }


def topology_path(scenario, split):
    return TOPOLOGY_DIR / f"{split}_{scenario}.json"


def save_topology(payload):
    path = topology_path(payload["scenario"], payload["split"])
    atomic_write_json(path, payload)
    return path


def load_topology(scenario, split):
    path = topology_path(scenario, split)
    payload = read_json(path)
    if payload is None:
        raise FileNotFoundError(f"Missing {path}")
    return payload


# ---------------------------------------------------------------------------
# Oracle reference and metrics
# ---------------------------------------------------------------------------


def build_oracle_reference(spec):
    """Existing Stage-2j-style oracle reference; reward/evaluation only."""
    env = ToyUAVEnv()
    env.reset(
        seed=int(spec["seed"]),
        goal_distance_range_m=tuple(spec["distance_range"]),
        render_observation=False,
    )
    start = env.position.copy()
    goal = env.goal.copy()
    reference = [start.copy()]
    success = False
    steps = 0
    for _ in range(MAX_STEPS):
        action = int(env.oracle_action())
        steps += 1
        _, success, done = env.step(action, render_observation=False)
        if action != STOP and np.linalg.norm(
            env.position - reference[-1]
        ) > 1e-9:
            reference.append(env.position.copy())
        if done:
            break
    reference = np.asarray(reference, dtype=np.float32)
    if np.linalg.norm(goal - reference[-1]) > 1e-9:
        reference = np.concatenate(
            [reference, goal.reshape(1, 2)], axis=0
        )
    return {
        "start": start,
        "goal": goal,
        "reference": reference.astype(np.float32),
        "oracle_success": bool(success),
        "oracle_steps": int(steps),
        "shortest_distance_m": float(np.linalg.norm(start - goal)),
    }


def joint_reward(ndtw, mean_e2e_ms):
    return float(ndtw) - JOINTR_DELAY_WEIGHT * float(mean_e2e_ms) / 100.0


def latency_penalty(e2e_ms):
    return (
        -REWARD_DELAY_WEIGHT
        * min(float(e2e_ms) / DEADLINE_MS, REWARD_DELAY_CLIP)
        / REWARD_DELAY_DIVISOR
    )


def bootstrap_ci(values, n_boot=10_000, seed=20261031, statistic=np.mean):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(n_boot), len(values)))
    samples = statistic(values[indices], axis=1)
    return {
        "mean": float(statistic(values)),
        "low": float(np.percentile(samples, 2.5)),
        "high": float(np.percentile(samples, 97.5)),
    }


def summarize_episodes(rows):
    rows = list(rows)
    if not rows:
        return {"episodes": 0}
    successes = np.asarray([float(row["success"]) for row in rows])
    ndtw = np.asarray([float(row["ndtw"]) for row in rows])
    e2e = np.asarray([float(row["mean_e2e_ms"]) for row in rows])
    p95 = np.asarray([float(row["p95_e2e_ms"]) for row in rows])
    violation = np.asarray([float(row["delay_violation_rate"]) for row in rows])
    steps = np.asarray([float(row["steps"]) for row in rows])
    model_usage = Counter()
    for row in rows:
        usage = row.get("model_usage", {})
        if isinstance(usage, str):
            usage = json.loads(usage)
        for name, count in usage.items():
            model_usage[name] += int(count)
    total_steps = max(sum(model_usage.values()), 1)
    return {
        "episodes": len(rows),
        "SR": float(successes.mean()),
        "nDTW": float(ndtw.mean()),
        "E2E_ms": float(e2e.mean()),
        "P95_ms": float(p95.mean()),
        "delay_violation_rate": float(violation.mean()),
        "mean_steps": float(steps.mean()),
        "JointR": joint_reward(ndtw.mean(), e2e.mean()),
        "model_usage": dict(model_usage),
        "model_usage_ratio": {
            name: float(count / total_steps)
            for name, count in model_usage.items()
        },
    }


def seed_isolation_report(spec_payloads, topology_payloads):
    """Check episode/topology seeds against Stage-2j and across splits."""
    stage2j_seeds = set()
    stage2j_layouts = set()

    train_stats = read_json(
        Path("runs/v83_spatial_stage2/datasets/train_600/dataset_stats.json")
    ) or {}
    for record in train_stats.get("trajectories", []):
        stage2j_seeds.add(int(record["seed"]))
        stage2j_layouts.add(int(record["network_layout_seed"]))

    dev_spec = read_json(
        Path("runs/v83_spatial_stage2/prep_audit_20261018T043420Z/"
             "dev_val_180_spec.json")
    ) or {}
    for record in dev_spec.get("trajectories", []):
        stage2j_seeds.add(int(record["seed"]))
        stage2j_layouts.add(int(record["network_layout_seed"]))

    # Only the already-derived task metadata is read, never test token tensors.
    stage2j_test_spec = read_json(
        Path("runs/v83_spatial_stage2/stage2j/specs/"
             "final_test_300_spec.json")
    ) or {}
    for record in stage2j_test_spec.get("specs", []):
        stage2j_seeds.add(int(record["seed"]))
        stage2j_layouts.add(int(record["network_layout_seed"]))

    split_seeds = {}
    split_layouts = {}
    for split, payload in spec_payloads.items():
        seeds = set()
        for scenario in SCENARIOS:
            seeds.update(
                int(ep["seed"]) for ep in payload["scenarios"][scenario]["episodes"]
            )
        split_seeds[split] = seeds
    for key, payload in topology_payloads.items():
        split, scenario = key
        split_layouts[(split, scenario)] = int(payload["topology_seed"])

    overlaps = {}
    for first in SPLITS:
        for second in SPLITS:
            if first >= second:
                continue
            if first in split_seeds and second in split_seeds:
                common = split_seeds[first] & split_seeds[second]
                if common:
                    overlaps[f"episode:{first}:{second}"] = sorted(common)
    for split, seeds in split_seeds.items():
        common = seeds & stage2j_seeds
        if common:
            overlaps[f"episode:{split}:stage2j"] = sorted(common)
    for key, seed in split_layouts.items():
        if seed in stage2j_layouts:
            overlaps[f"topology:{key[0]}:{key[1]}"] = [seed]
        for other, other_seed in split_layouts.items():
            if key < other and seed == other_seed:
                overlaps[f"topology:{key}:{other}"] = [seed]
    return {
        "stage2j_episode_seeds": len(stage2j_seeds),
        "stage2j_topology_seeds": len(stage2j_layouts),
        "split_episode_seeds": {
            split: len(seeds) for split, seeds in split_seeds.items()
        },
        "overlaps": overlaps,
        "ok": not overlaps,
    }


def json_safe(value):
    """Recursively convert numpy scalars/arrays for JSON output."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def latency_from_node(node, distance_m):
    """Shim used by tests and the state builder; actual model in
    ``stage2k_latency``."""
    return LatencyModel().a2g_channel(
        distance_m,
        tx_power_w=TX_POWER_W,
        bandwidth_hz=float(node["bandwidth_hz"]),
    )
