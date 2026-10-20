"""Build the Dev-Val-only quality prior for the five frozen MECs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    MODEL_NAMES,
    QUALITY_DIR,
    QUALITY_PRIOR_PATH,
    atomic_write_json,
    sha256_file,
    utc_now,
)


def main():
    models = {}
    for name in MODEL_NAMES:
        path = QUALITY_DIR / name / "metrics.json"
        if not path.is_file():
            raise SystemExit(f"Missing {path}; finish quality evaluation first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        overall = payload["overall"]
        meta = payload["meta"]
        models[name] = {
            "mean_ndtw": float(overall["ndtw"]),
            "sr": float(overall["sr"]),
            "spl": float(overall["spl"]),
            "osr": float(overall["osr"]),
            "steps": float(overall["steps"]),
            "episodes": int(overall["episodes"]),
            "metrics_path": str(path),
            "metrics_sha256": sha256_file(path),
            "checkpoint": meta["checkpoint"],
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "split": meta["split"],
            "final_test_read": bool(meta["final_test_read"]),
        }
    rank = sorted(
        MODEL_NAMES,
        key=lambda name: (-models[name]["mean_ndtw"], MODEL_NAMES.index(name)),
    )
    payload = {
        "version": "stage2k-quality-prior-v1",
        "created_utc": utc_now(),
        "source": "Stage-2j Dev Val 180 closed-loop, single MEC always selected",
        "split": "dev_val_180",
        "episodes": 180,
        "quality_definition": "mean full-trajectory nDTW",
        "final_test_read": False,
        "models": models,
        "rank": rank,
    }
    if any(item["final_test_read"] for item in models.values()):
        raise SystemExit("Quality metrics claim a Final Test read; refusing")
    atomic_write_json(QUALITY_PRIOR_PATH, payload)
    print(json.dumps({
        "path": str(QUALITY_PRIOR_PATH),
        "rank": rank,
        "quality": {
            name: models[name]["mean_ndtw"] for name in MODEL_NAMES
        },
        "final_test_read": False,
    }, indent=2), flush=True)
    print("STAGE2K_QUALITY_PRIOR_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
