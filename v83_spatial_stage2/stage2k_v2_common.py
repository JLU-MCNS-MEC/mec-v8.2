"""Shared constants and validation for the Stage-2k v2 ablations.

The v2 capability prior is built **only** from the Stage-2j Dev Val 180
closed-loop evaluation (episode seeds 2,000,000+, disjoint from Stage-2k
Train/Dev/Test) and contains four per-MEC metrics:

    SR, OSR, SPL, nDTW

The Stage-2j Final Test 300 token directory is never opened.  Both the builder
and the consumer reject a prior that claims ``final_test_read != false`` or a
split other than ``dev_val``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    MODEL_NAMES,
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
)


CAPABILITY_METRICS = ("sr", "osr", "spl", "ndtw")
CAPABILITY_FIELDS = ("dev_sr", "dev_osr", "dev_spl", "dev_ndtw")
CAPABILITY_FIELD_TO_METRIC = dict(zip(CAPABILITY_FIELDS, CAPABILITY_METRICS))
CAPABILITY_PRIOR_PATH = Path(
    "runs/v83_spatial_stage2/stage2k/capability_prior_v2.json"
)
CAPABILITY_PRIOR_VERSION = "stage2k-capability-prior-v2"


def validate_capability_prior(payload, verify_hashes=True):
    """Validate a v2 capability prior payload and return a flat dict.

    Raises ``ValueError`` on provenance, metric or hash problems.  The returned
    dict is ``{model_name: {metric: float}}``.
    """
    if not isinstance(payload, dict):
        raise TypeError("capability prior must be a dict")
    if payload.get("version") != CAPABILITY_PRIOR_VERSION:
        raise ValueError(
            f"capability prior version {payload.get('version')!r} != "
            f"{CAPABILITY_PRIOR_VERSION!r}"
        )
    if payload.get("split") != "dev_val_180":
        raise ValueError(
            f"capability prior split {payload.get('split')!r} is not dev_val_180"
        )
    if payload.get("final_test_read") is not False:
        raise ValueError("capability prior claims a Final Test read")
    if int(payload.get("episodes", -1)) != 180:
        raise ValueError("capability prior must cover 180 Dev episodes")
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("capability prior has no models dict")
    flat = {}
    for name in MODEL_NAMES:
        if name not in models:
            raise ValueError(f"capability prior missing model {name}")
        item = models[name]
        for metric in CAPABILITY_METRICS:
            if metric not in item:
                raise ValueError(
                    f"capability prior {name} missing metric {metric}"
                )
            value = float(item[metric])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"capability prior {name}.{metric}={value} outside [0,1]"
                )
        if int(item.get("episodes", -1)) != 180:
            raise ValueError(f"capability prior {name} is not 180 episodes")
        if item.get("split") != "dev_val":
            raise ValueError(f"capability prior {name} split is not dev_val")
        if item.get("final_test_read") is not False:
            raise ValueError(
                f"capability prior {name} claims a Final Test read"
            )
        metrics_path = Path(item["metrics_path"])
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        if verify_hashes:
            if sha256_file(metrics_path) != item["metrics_sha256"]:
                raise ValueError(
                    f"capability metrics hash mismatch for {name}"
                )
            checkpoint = Path(item["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if sha256_file(checkpoint) != item["checkpoint_sha256"]:
                raise ValueError(
                    f"capability checkpoint hash mismatch for {name}"
                )
        flat[name] = {
            metric: float(item[metric]) for metric in CAPABILITY_METRICS
        }
    return flat


def load_capability_prior(path=CAPABILITY_PRIOR_PATH, verify_hashes=True):
    payload = read_json(Path(path))
    if payload is None:
        raise FileNotFoundError(path)
    return payload, validate_capability_prior(
        payload, verify_hashes=verify_hashes
    )


def flatten_capability(prior):
    """Return ``{model: {metric: value}}`` from a validated prior payload."""
    if not isinstance(prior, dict) or "models" not in prior:
        raise TypeError("flatten_capability expects a capability prior payload")
    return {
        name: {
            metric: float(prior["models"][name][metric])
            for metric in CAPABILITY_METRICS
        }
        for name in MODEL_NAMES
    }


def capability_values(flat, model_name):
    item = flat[model_name]
    return np.asarray(
        [float(item[metric]) for metric in CAPABILITY_METRICS],
        dtype=np.float32,
    )


def build_capability_prior(quality_dir):
    """Build the v2 prior from Stage-2j Dev Val 180 ``metrics.json`` files."""
    quality_dir = Path(quality_dir)
    models = {}
    for name in MODEL_NAMES:
        path = quality_dir / name / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = read_json(path)
        meta = payload["meta"]
        overall = payload["overall"]
        if meta.get("split") != "dev_val":
            raise ValueError(f"{path} is not a dev_val evaluation")
        if meta.get("final_test_read") is not False:
            raise ValueError(f"{path} claims a Final Test read")
        if int(meta.get("episodes", -1)) != 180:
            raise ValueError(f"{path} does not cover 180 episodes")
        models[name] = {
            **{metric: float(overall[metric]) for metric in CAPABILITY_METRICS},
            "episodes": int(overall["episodes"]),
            "metrics_path": str(path),
            "metrics_sha256": sha256_file(path),
            "checkpoint": str(meta["checkpoint"]),
            "checkpoint_sha256": str(meta["checkpoint_sha256"]),
            "split": str(meta["split"]),
            "final_test_read": bool(meta["final_test_read"]),
        }
    rank = sorted(
        MODEL_NAMES,
        key=lambda name: (-models[name]["ndtw"], MODEL_NAMES.index(name)),
    )
    return {
        "version": CAPABILITY_PRIOR_VERSION,
        "created_utc": utc_now(),
        "source": (
            "Stage-2j Dev Val 180 closed-loop, single MEC always selected; "
            "episode seeds 2,000,000+ (disjoint from Stage-2k Train/Dev/Test)"
        ),
        "split": "dev_val_180",
        "episodes": 180,
        "metrics": list(CAPABILITY_METRICS),
        "quality_definition": (
            "per-MEC closed-loop SR, OSR, SPL and nDTW; no single composite"
        ),
        "final_test_read": False,
        "models": models,
        "rank_by_ndtw": rank,
    }


def write_capability_prior(path, payload):
    validate_capability_prior(payload, verify_hashes=True)
    atomic_write_json(Path(path), payload)
    return Path(path)
