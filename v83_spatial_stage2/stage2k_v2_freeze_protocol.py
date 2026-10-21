"""Freeze the Stage-2k v2 protocol before formal training.

Writes the same payload to:

  * v83_spatial_stage2/STAGE2K_V2_PROTOCOL_FREEZE.json  (committed)
  * runs/v83_spatial_stage2/stage2k_v2_protocol_freeze.json (run provenance)

No training, no Final Test read.  The freeze records source/test hashes, the
Dev-only capability prior, the frozen visual PCA, the Train/Dev specs and
topologies, and the pre-registered Dev60 competitiveness threshold.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from v83_spatial_stage2.stage2k_common import (
    MODEL_NAMES,
    SCENARIOS,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from v83_spatial_stage2.stage2k_dqn import SEED, dev_validation_specs
from v83_spatial_stage2.stage2k_jointr_v2_order import RANDOM_ORDER_SEED
from v83_spatial_stage2.stage2k_v2_common import CAPABILITY_PRIOR_PATH

SOURCE_DIR = Path("v83_spatial_stage2")
OUTPUT_SOURCE = SOURCE_DIR / "STAGE2K_V2_PROTOCOL_FREEZE.json"
OUTPUT_RUN = Path("runs/v83_spatial_stage2/stage2k_v2_protocol_freeze.json")
BASELINE_CSV = Path("runs/v83_spatial_stage2/stage2k/baselines/dev/per_episode.csv")
SPEC_DIR = Path("runs/v83_spatial_stage2/stage2k/specs")
TOPOLOGY_DIR = Path("runs/v83_spatial_stage2/stage2k/topologies")


def hash_if_exists(path):
    path = Path(path)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def dev60_baseline_threshold():
    validation = {
        item["episode_uid"] for item in dev_validation_specs()
    }
    rows = list(csv.DictReader(BASELINE_CSV.open(encoding="utf-8")))
    summary = {}
    for strategy in ("random", "min_latency", "max_quality", "heuristic"):
        selected = [
            row for row in rows
            if row["strategy"] == strategy
            and row["episode_uid"] in validation
        ]
        if len(selected) != 60:
            raise RuntimeError(
                f"{strategy} has {len(selected)} Dev60 rows, expected 60"
            )
        ndtw = float(np.mean([float(row["ndtw"]) for row in selected]))
        e2e = float(np.mean([
            float(row["mean_e2e_ms"]) for row in selected
        ]))
        summary[strategy] = {
            "episodes": len(selected),
            "nDTW": ndtw,
            "E2E_ms": e2e,
            "JointR": ndtw - 0.25 * e2e / 100.0,
        }
    strongest = max(
        summary, key=lambda name: summary[name]["JointR"]
    )
    return {
        "source": str(BASELINE_CSV),
        "source_sha256": sha256_file(BASELINE_CSV),
        "validation_episodes": len(validation),
        "strategies": summary,
        "strongest_non_oracle": strongest,
        "threshold_jointr": float(summary[strongest]["JointR"]),
        "oracle_excluded": True,
    }


def main():
    source_files = [
        SOURCE_DIR / "STAGE2K_V2_PROTOCOL.md",
        SOURCE_DIR / "stage2k_pure_mec_v2_state.py",
        SOURCE_DIR / "stage2k_v2_common.py",
        SOURCE_DIR / "stage2k_build_capability_prior_v2.py",
        SOURCE_DIR / "stage2k_runtime.py",
        SOURCE_DIR / "stage2k_jointr_v2_train.py",
        SOURCE_DIR / "stage2k_jointr_v2_eval.py",
        SOURCE_DIR / "stage2k_jointr_v2_agent.py",
        SOURCE_DIR / "stage2k_jointr_v2_order.py",
        SOURCE_DIR / "stage2k_common.py",
        SOURCE_DIR / "test_stage2k_pure_mec_v2.py",
        SOURCE_DIR / "test_stage2k_v2_protocol.py",
        SOURCE_DIR / "stage2k_v2_preexperiment_audit.py",
    ]
    artifacts = [
        CAPABILITY_PRIOR_PATH,
        Path("runs/v83_spatial_stage2/stage2k/visual_pca.npz"),
    ]
    specs = [
        SPEC_DIR / "train_episodes.json",
        SPEC_DIR / "dev_episodes.json",
    ]
    topologies = [
        TOPOLOGY_DIR / f"{split}_{scenario}.json"
        for split in ("train", "dev")
        for scenario in SCENARIOS
    ]
    payload = {
        "version": "stage2k-v2-protocol-freeze-v1",
        "created_utc": utc_now(),
        "branch": "experiment/v83-spatial-foundation",
        "protocol": str(SOURCE_DIR / "STAGE2K_V2_PROTOCOL.md"),
        "state_modes": {
            "A": {
                "name": "pure_mec_v2",
                "mlp_input": 384,
                "decision_state": 404,
            },
            "B": {
                "name": "visual_ablation_v2",
                "mlp_input": 512,
                "decision_state": 532,
            },
        },
        "training": {
            "episodes": 3500,
            "validation_every": 200,
            "validation_episodes": 60,
            "epsilon_decay_steps": 500_000,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "gamma": 1.0,
            "seed": int(SEED),
            "order_seed": int(RANDOM_ORDER_SEED),
            "order_total": 3500,
            "early_stop_disabled": True,
            "q_divergence_abs": 1e5,
        },
        "convergence_criteria": {
            "completed_cap": True,
            "numerically_finite": True,
            "min_validations": 5,
            "last3_jointr_range_max": 0.01,
            "last_minus_best_min": -0.005,
            "improving": False,
        },
        "final_test_gate": {
            "G1": "converged == true",
            "G2": "best Dev60 JointR > strongest non-oracle baseline",
            "dev60_baseline": dev60_baseline_threshold(),
            "final_test_read": False,
        },
        "sources": [hash_if_exists(path) for path in source_files],
        "artifacts": [hash_if_exists(path) for path in artifacts],
        "specs": [hash_if_exists(path) for path in specs],
        "topologies": [hash_if_exists(path) for path in topologies],
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None
            ),
            "platform": platform.platform(),
        },
        "final_test_read": False,
    }
    atomic_write_json(OUTPUT_SOURCE, payload)
    atomic_write_json(OUTPUT_RUN, payload)
    print(json.dumps({
        "source": str(OUTPUT_SOURCE),
        "run_copy": str(OUTPUT_RUN),
        "source_sha256": sha256_file(OUTPUT_SOURCE),
        "capability_prior_sha256": sha256_file(CAPABILITY_PRIOR_PATH),
        "visual_pca_sha256": sha256_file(
            "runs/v83_spatial_stage2/stage2k/visual_pca.npz"
        ),
        "strongest_non_oracle": payload["final_test_gate"][
            "dev60_baseline"
        ]["strongest_non_oracle"],
        "threshold_jointr": payload["final_test_gate"][
            "dev60_baseline"
        ]["threshold_jointr"],
        "final_test_read": False,
    }, indent=2, ensure_ascii=False), flush=True)
    print("STAGE2K_V2_PROTOCOL_FREEZE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
