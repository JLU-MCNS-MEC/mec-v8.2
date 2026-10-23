"""Verify that training-relevant sources still match the frozen protocol.

The protocol was frozen immediately before the formal launch.  Analysis-only
scripts (``analyze.py``, ``plot.py``, ``render_report.py``) may be improved
after launch without affecting the trained checkpoints; their final hashes are
recorded separately.  This script fails if any *training-relevant* file changed
after the freeze.

    python v83_spatial_stage2/stage2o/verify_provenance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    utc_now,
)

EVIDENCE = Path(__file__).resolve().parent / "evidence"
FREEZE = EVIDENCE / "protocol_freeze.json"

TRAINING_PREFIXES = (
    "v83_spatial_stage2/stage2o/train.py",
    "v83_spatial_stage2/stage2o/rewards.py",
    "v83_spatial_stage2/stage2o/progress.py",
    "v83_spatial_stage2/stage2o/launch.py",
    "v83_spatial_stage2/stage2k_jointr_v2_agent.py",
    "v83_spatial_stage2/stage2k_ppo_agent.py",
    "v83_spatial_stage2/stage2n_train.py",
    "v83_spatial_stage2/stage2n_lanes.py",
    "v83_spatial_stage2/stage2n_runtime.py",
    "v83_spatial_stage2/stage2n_common.py",
)
ANALYSIS_FILES = (
    "v83_spatial_stage2/stage2o/analyze.py",
    "v83_spatial_stage2/stage2o/plot.py",
    "v83_spatial_stage2/stage2o/render_report.py",
    "v83_spatial_stage2/stage2o/status.py",
    "v83_spatial_stage2/stage2o/freeze_protocol.py",
    "v83_spatial_stage2/stage2o/test_stage2o.py",
)


def main():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    frozen = freeze["code_sha256"]
    training = {}
    analysis = {}
    for relative, expected in frozen.items():
        path = PROJECT_ROOT / relative
        current = sha256_file(path) if path.is_file() else None
        entry = {
            "path": relative,
            "frozen_sha256": expected,
            "current_sha256": current,
            "match": current == expected,
        }
        if any(relative == item for item in TRAINING_PREFIXES):
            training[relative] = entry
        else:
            analysis[relative] = entry
    current_analysis = {}
    for relative in ANALYSIS_FILES:
        path = PROJECT_ROOT / relative
        current_analysis[relative] = (
            sha256_file(path) if path.is_file() else None
        )
    training_ok = all(item["match"] for item in training.values())
    payload = {
        "stage": "stage2o",
        "checked_utc": utc_now(),
        "freeze_created_utc": freeze.get("created_utc"),
        "training_relevant": training,
        "training_ok": bool(training_ok),
        "analysis_at_freeze": analysis,
        "analysis_current": current_analysis,
        "note": (
            "Training-relevant sources must match the freeze. "
            "Analysis-only scripts may be improved after launch; their "
            "current hashes are recorded here and do not affect training."
        ),
    }
    atomic_write_json(EVIDENCE / "provenance_check.json", payload)
    print(
        f"[provenance] training_ok={training_ok} "
        f"({sum(item['match'] for item in training.values())}"
        f"/{len(training)})",
        flush=True,
    )
    for relative, item in training.items():
        if not item["match"]:
            print(
                f"  CHANGED {relative}: {item['frozen_sha256']} -> "
                f"{item['current_sha256']}",
                flush=True,
            )
    if not training_ok:
        raise SystemExit("training-relevant source changed after freeze")


if __name__ == "__main__":
    main()
