"""Freeze the Stage-2j model/training/evaluation protocol before test.

This script does **not** read Final Test token shards or compute any test
metric.  It records:

  * the git commit and SHA256 of every critical source file;
  * the five native architecture configs and parameter counts;
  * the complete Stage-2i training protocol;
  * the Final Test evaluation protocol and the SHA256 of SEALED.json only;
  * the Stage-2i T5 checkpoint SHA256;
  * Train/Dev-Val dataset-stat hashes and the auxiliary manifest hash;
  * Python / torch / CUDA / GPU provenance.

The generated JSON is checked by ``stage2j_eval.py --split final_test``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2j_common import (
    AUX_ROOT,
    COMMITTED_FREEZE_PATH,
    EVAL_PROTOCOL,
    FINAL_TEST_DIR,
    FREEZE_PATH,
    STAGE2J_ROOT,
    T5_CHECKPOINT,
    T5_EXPECTED_SHA256,
    T5_TRAINING_RESULT,
    TRAIN_DIR,
    TRAIN_PROTOCOL,
    VAL_DIR,
    atomic_write_json,
    file_sha256,
    utc_now,
)
from v83_spatial_stage2.stage2j_model import (
    STAGE2J_CONFIGS,
    TABLE_ORDER,
    build_stage2j_model,
    parameter_count,
)


CRITICAL_FILES = (
    "config.py",
    "encoder.py",
    "env.py",
    "gbs_model_v3.py",
    "gbs_network.py",
    "navigation_reward.py",
    "token_data.py",
    "utils.py",
    "v83_spatial_training/eval_closedloop_spatial.py",
    "v83_spatial_stage2/stage2d_common.py",
    "v83_spatial_stage2/stage2f_collect.py",
    "v83_spatial_stage2/stage2h_gab.py",
    "v83_spatial_stage2/stage2i_common.py",
    "v83_spatial_stage2/stage2i_data.py",
    "v83_spatial_stage2/stage2i_model.py",
    "v83_spatial_stage2/stage2i_train.py",
    "v83_spatial_stage2/stage2j_common.py",
    "v83_spatial_stage2/stage2j_model.py",
    "v83_spatial_stage2/stage2j_train.py",
    "v83_spatial_stage2/stage2j_eval.py",
    "v83_spatial_stage2/stage2j_freeze.py",
)


def _run_git(*args):
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
        check=True,
    )
    return result.stdout.strip()


def _git_state():
    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git("status", "--porcelain")
    return {"commit": commit, "branch": branch, "status": status}


def _check_clean_critical():
    dirty = _run_git("status", "--porcelain", "--", *CRITICAL_FILES)
    if dirty.strip():
        raise SystemExit(
            "Critical Stage-2j source files have uncommitted changes:\n"
            + dirty
        )


def _file_hashes(paths):
    return {
        str(Path(path)): file_sha256(PROJECT_ROOT / path) for path in paths
    }


def _model_table():
    table = {}
    for name in TABLE_ORDER:
        model = build_stage2j_model(name)
        table[name] = {
            "architecture": STAGE2J_CONFIGS[name]["architecture"],
            "native_config": dict(STAGE2J_CONFIGS[name]),
            "parameters": parameter_count(model),
            "trainable_parameters": int(
                sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
            ),
            "initialisation_seed": 20264017,
        }
    return table


def main():
    STAGE2J_ROOT.mkdir(parents=True, exist_ok=True)
    _check_clean_critical()
    git = _git_state()

    t5_sha = file_sha256(T5_CHECKPOINT)
    if t5_sha != T5_EXPECTED_SHA256:
        raise SystemExit(
            f"Stage-2i T5 checkpoint SHA256 mismatch: {t5_sha}"
        )
    sealed_path = FINAL_TEST_DIR / "SEALED.json"
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    if sealed.get("split") != "final_test":
        raise SystemExit("SEALED.json is not a final_test manifest")
    t5_result = json.loads(T5_TRAINING_RESULT.read_text(encoding="utf-8"))

    freeze = {
        "status": "frozen",
        "protocol_version": "stage2j-v1",
        "created_utc": utc_now(),
        "git": git,
        "critical_files": _file_hashes(CRITICAL_FILES),
        "models": _model_table(),
        "training_protocol": TRAIN_PROTOCOL,
        "evaluation_protocol": EVAL_PROTOCOL,
        "stop_threshold": float(EVAL_PROTOCOL["stop_threshold"]),
        "t5_reuse": {
            "checkpoint": str(T5_CHECKPOINT),
            "checkpoint_sha256": t5_sha,
            "training_result": str(T5_TRAINING_RESULT),
            "training_result_sha256": file_sha256(T5_TRAINING_RESULT),
            "best_epoch": int(t5_result["best_epoch"]),
            "checkpoint_score": float(t5_result["best_score"]),
            "training_time_s": float(t5_result["training_time_s"]),
        },
        "datasets": {
            "train_dir": str(TRAIN_DIR),
            "train_stats_sha256": file_sha256(TRAIN_DIR / "dataset_stats.json"),
            "train_shards": len(list(TRAIN_DIR.glob("*.pt"))),
            "dev_val_dir": str(VAL_DIR),
            "dev_val_stats_sha256": file_sha256(
                VAL_DIR / "dataset_stats.json"),
            "dev_val_shards": len(list(VAL_DIR.glob("*.pt"))),
            "aux_manifest": str(AUX_ROOT / "manifest.json"),
            "aux_manifest_sha256": file_sha256(AUX_ROOT / "manifest.json"),
            "train_class_counts_sha256": file_sha256(
                AUX_ROOT / "train_class_counts.json"),
        },
        "final_test": {
            "directory": str(FINAL_TEST_DIR),
            "sealed_manifest": str(sealed_path),
            "sealed_manifest_sha256": file_sha256(sealed_path),
            "sealed_files": len(sealed.get("files", {})),
            "token_shards": len([
                name for name in sealed.get("files", {})
                if name.endswith(".pt")
            ]),
            "token_shards_read_for_freeze": 0,
            "labels_read_for_freeze": False,
            "model_evaluated_for_freeze": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None
            ),
        },
        "freeze_rules": {
            "five_models_fixed": list(TABLE_ORDER),
            "execution_order": ["Small", "Transformer-S", "Medium", "Large",
                                "Transformer-M"],
            "no_retraining_for_test": True,
            "no_stop_threshold_tuning": True,
            "no_test_based_model_or_hyperparameter_change": True,
            "test_read_once_per_model": True,
            "bug_fix_policy": (
                "Only engineering fixes independent of test outcomes are "
                "allowed; record an amendment and verify metric consistency."
            ),
        },
        "pre_freeze_inspection_disclosure": {
            "date": "2026-10-19",
            "read": [
                "directory listing of final_test_300",
                "SEALED.json manifest fields and file hash list",
                "shard_0000.pt metadata arrays only (episode_ids, "
                "episode_seeds, network_layout_seeds, "
                "trajectory_band_ids, step_ids) to confirm schema",
                "top-level aggregate fields of dataset_stats.json",
            ],
            "not_read": [
                "Final Test token tensors",
                "Final Test labels or arrival targets",
                "any model prediction or metric on Final Test",
            ],
            "models_evaluated_on_test": 0,
        },
    }

    atomic_write_json(FREEZE_PATH, freeze)
    COMMITTED_FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMMITTED_FREEZE_PATH.write_text(
        json.dumps(freeze, indent=2), encoding="utf-8"
    )
    print("FROZEN", flush=True)
    print(json.dumps({
        "git_commit": git["commit"],
        "models": {
            name: freeze["models"][name]["parameters"]
            for name in TABLE_ORDER
        },
        "t5_checkpoint_sha256": t5_sha,
        "sealed_manifest_sha256": freeze["final_test"][
            "sealed_manifest_sha256"
        ],
        "critical_files": len(freeze["critical_files"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
