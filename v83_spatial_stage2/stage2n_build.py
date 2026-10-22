"""Build Stage-2n task pools, the homogeneous deployment and the freeze file.

Run once before any Stage-2n training:

    python v83_spatial_stage2/stage2n_build.py

The script writes only under
``runs/v83_spatial_stage2/stage2n_latency_only/`` and refuses to overwrite an
existing protocol freeze unless ``--force`` is given.  It never reads the
Stage-2n Test tasks for training; it only writes the sealed 400-task pool and
records its SHA-256 in the freeze file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import atomic_write_json, read_json, utc_now
from v83_spatial_stage2.stage2n_common import (
    BAND_COUNTS_DEV,
    BAND_COUNTS_TEST,
    BAND_COUNTS_TRAIN,
    CHECKPOINT_SELECTION,
    CONVERGENCE_CRITERION,
    DEPLOYMENT_DIR,
    EVIDENCE_DIR,
    MAP_SEED,
    MEC_BANDWIDTH_HZ,
    MEC_CHECKPOINT,
    MEC_CHECKPOINT_SHA256,
    MEC_COMPUTE_FLOPS_PER_S,
    MEC_COVERAGE_RADIUS_M,
    MEC_MODEL,
    MEC_NUM_NODES,
    PROTOCOL_PATH,
    SPECS_DIR,
    STAGE,
    STATE_DIM,
    ACTION_DIM,
    TASK_SEED_BASE,
    TRAIN_EPISODES,
    TRAIN_ORDER_SEED,
    VALIDATION_EVERY,
    build_homogeneous_deployment,
    build_training_order,
    build_city,
    canonical_sha256,
    deployment_topology,
    distribution_report,
    generate_split_tasks,
    isolation_audit,
    load_deployment,
    save_task_pool,
    sha256_file,
    verify_mec_checkpoint,
)
from v83_spatial_stage2.stage2n_common import (
    AGENT_SEEDS,
    REWARD_DESCRIPTION,
    LATENCY_RETURN_DESCRIPTION,
)


def _cross_stage_audit(our_pools: dict) -> dict:
    """Check Stage-2n seeds/IDs against the existing Stage-2l pools."""
    stage2l_specs = (
        SPECS_DIR.parents[1] / "stage2l_fixed_city" / "specs"
    )
    result = {
        "stage2l_specs_dir": str(stage2l_specs),
        "checks": {},
        "ok": True,
    }
    seeds_by_split = {
        split: {int(item["task_seed"]) for item in payload["tasks"]}
        for split, payload in our_pools.items()
    }
    uids_by_split = {
        split: {str(item["episode_uid"]) for item in payload["tasks"]}
        for split, payload in our_pools.items()
    }
    for split in ("train", "dev", "test"):
        path = stage2l_specs / f"{split}_tasks.json"
        if not path.is_file():
            result["checks"][f"stage2l_{split}_available"] = False
            continue
        payload = read_json(path)
        old_seeds = {int(item["task_seed"]) for item in payload["tasks"]}
        old_uids = {str(item["episode_uid"]) for item in payload["tasks"]}
        overlap_seeds = seeds_by_split[split] & old_seeds
        overlap_uids = uids_by_split[split] & old_uids
        result["checks"][f"stage2l_{split}_seed_disjoint"] = not overlap_seeds
        result["checks"][f"stage2l_{split}_uid_disjoint"] = not overlap_uids
        if overlap_seeds:
            result[f"overlap_stage2l_{split}_seeds"] = sorted(overlap_seeds)[:10]
        if overlap_uids:
            result[f"overlap_stage2l_{split}_uids"] = sorted(overlap_uids)[:10]
    result["ok"] = all(result["checks"].values())
    return result


def build(force: bool = False, verbose: bool = True) -> dict:
    if PROTOCOL_PATH.is_file() and not force:
        raise SystemExit(
            f"{PROTOCOL_PATH} already exists; use --force to rebuild"
        )
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[stage2n] verifying frozen Transformer-M", flush=True)
    mec_record = verify_mec_checkpoint(verbose=verbose)

    if verbose:
        print("[stage2n] generating fixed-city task pools", flush=True)
    train_specs = generate_split_tasks(
        "train", BAND_COUNTS_TRAIN, verbose=verbose
    )
    dev_specs_ = generate_split_tasks("dev", BAND_COUNTS_DEV, verbose=verbose)
    test_specs = generate_split_tasks(
        "test", BAND_COUNTS_TEST, verbose=verbose
    )
    save_task_pool("train", train_specs)
    save_task_pool("dev", dev_specs_)
    save_task_pool("test", test_specs, sealed=True)
    order = build_training_order(train_specs, seed=TRAIN_ORDER_SEED)

    pools = {}
    for split in ("train", "dev", "test"):
        payload = read_json(SPECS_DIR / f"{split}_tasks.json")
        pools[split] = payload

    isolation = isolation_audit(train_specs, dev_specs_, test_specs)
    atomic_write_json(SPECS_DIR / "isolation.json", isolation)
    if not isolation["checks"]["ok"]:
        raise RuntimeError(f"isolation audit failed: {isolation['checks']}")
    distribution = distribution_report(train_specs, dev_specs_, test_specs)
    atomic_write_json(SPECS_DIR / "distribution.json", distribution)
    cross_stage = _cross_stage_audit(pools)
    atomic_write_json(SPECS_DIR / "cross_stage_isolation.json", cross_stage)
    if not cross_stage["ok"]:
        raise RuntimeError(
            f"Stage-2n/Stage-2l seed or ID overlap: {cross_stage}"
        )

    if verbose:
        print("[stage2n] building homogeneous deployment", flush=True)
    deployment = build_homogeneous_deployment(verbose=verbose)
    topology = deployment_topology(load_deployment())

    protocol = {
        "version": "stage2n-protocol-freeze-v1",
        "stage": STAGE,
        "created_utc": utc_now(),
        "title": "Stage-2n: Fixed-City Latency-Only MEC Association",
        "map": {
            "map_seed": int(MAP_SEED),
            "city_fingerprint": build_city(MAP_SEED).fingerprint,
            "num_landmarks": int(build_city(MAP_SEED).num_landmarks),
            "world_size_m": float(build_city(MAP_SEED).world_size_m),
        },
        "tasks": {
            split: {
                "path": str(SPECS_DIR / f"{split}_tasks.json"),
                "sha256": sha256_file(SPECS_DIR / f"{split}_tasks.json"),
                "tasks_sha256": pools[split]["tasks_sha256"],
                "num_tasks": pools[split]["num_tasks"],
                "band_counts": pools[split]["band_counts"],
                "seed_base": pools[split]["seed_base"],
                "seed_range": pools[split]["seed_range"],
                "sealed": bool(pools[split]["sealed"]),
                "final_test_read": False,
            }
            for split in ("train", "dev", "test")
        },
        "training_order": {
            "path": str(SPECS_DIR / "training_order.json"),
            "sha256": sha256_file(SPECS_DIR / "training_order.json"),
            "seed": int(TRAIN_ORDER_SEED),
            "order_sha256": canonical_sha256(order["order"]),
        },
        "isolation": {
            "path": str(SPECS_DIR / "isolation.json"),
            "sha256": sha256_file(SPECS_DIR / "isolation.json"),
            "checks": isolation["checks"],
            "cross_stage": cross_stage,
        },
        "deployment": {
            "path": str(DEPLOYMENT_DIR / "deployment.json"),
            "sha256": sha256_file(DEPLOYMENT_DIR / "deployment.json"),
            "deployment_sha256": deployment["deployment_sha256"],
            "topology": topology,
            "source_topology": deployment["source_topology"],
            "source_topology_sha256": deployment["source_topology_sha256"],
            "homogeneity_checks": deployment["homogeneity_checks"],
            "num_nodes": int(MEC_NUM_NODES),
            "model": MEC_MODEL,
            "compute_capacity_flops_per_s": float(MEC_COMPUTE_FLOPS_PER_S),
            "bandwidth_hz": float(MEC_BANDWIDTH_HZ),
            "coverage_radius_m": float(MEC_COVERAGE_RADIUS_M),
        },
        "mec": mec_record,
        "state": {
            "mode": "pure_mec_v2",
            "dim": int(STATE_DIM),
            "action_dim": int(ACTION_DIM),
            "availability_mask": "separate 20-D mask; only action masking",
            "forbidden_inputs": [
                "RGB", "goal image", "visual PCA", "visual history",
                "target position", "oracle action", "success label",
            ],
            "mec_visual_inputs_unchanged": True,
        },
        "reward": {
            "mode": "latency_only_v1",
            "per_step": REWARD_DESCRIPTION,
            "episode": LATENCY_RETURN_DESCRIPTION,
            "navigation_reward_removed": True,
            "success_bonus": 0.0,
            "failure_penalty": 0.0,
            "gamma_dqn": 1.0,
            "gamma_ppo": 1.0,
        },
        "training": {
            "algorithms": ["dqn", "ppo"],
            "agent_seeds": list(AGENT_SEEDS),
            "max_episodes": int(TRAIN_EPISODES),
            "validation_every": int(VALIDATION_EVERY),
            "validation_tasks": 60,
            "validation_split": "dev",
            "same_task_order_for_all_runs": True,
            "test_used_for_training_or_selection": False,
        },
        "convergence": {
            "criterion": CONVERGENCE_CRITERION,
            "metric": "Dev mean E2E delay (ms)",
            "best_checkpoint_selection": CHECKPOINT_SELECTION,
        },
        "baselines": {
            "random": "uniform over currently available MECs; seeded RNG",
            "nearest": "minimum current UAV-node distance among available",
            "max_uplink_rate": "maximum current uplink rate among available",
            "greedy_min_e2e": (
                "minimum current analytic E2E estimate among available; uses "
                "only current UAV position, availability and the observable "
                "latency features (distance/bandwidth/compute/model)"
            ),
            "privileged_information": {
                "random": False,
                "nearest": False,
                "max_uplink_rate": False,
                "greedy_min_e2e": False,
                "dqn": False,
                "ppo": False,
                "note": (
                    "No strategy receives target position, future trajectory "
                    "or oracle actions.  Greedy-Min-E2E uses the same "
                    "analytic latency model that produces the observable "
                    "candidate E2E feature in the 384-D state."
                ),
            },
        },
        "stop_radius_m": 20.0,
        "max_steps": 400,
        "final_test_read": False,
        "test_sealed": True,
        "test_policy": "only the final evaluator may open Test-400",
    }
    atomic_write_json(PROTOCOL_PATH, protocol)

    setup = {
        "created_utc": utc_now(),
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "isolation": isolation,
        "cross_stage": cross_stage,
        "distribution": distribution,
        "deployment": deployment,
        "mec": mec_record,
    }
    atomic_write_json(EVIDENCE_DIR / "build_audit.json", setup)
    if verbose:
        print(json.dumps({
            "protocol_sha256": setup["protocol_sha256"],
            "deployment_sha256": deployment["deployment_sha256"],
            "tasks": {
                split: pools[split]["num_tasks"]
                for split in ("train", "dev", "test")
            },
        }, indent=2), flush=True)
        print("STAGE2N_BUILD_COMPLETE", flush=True)
    return setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(force=args.force, verbose=True)


if __name__ == "__main__":
    main()
