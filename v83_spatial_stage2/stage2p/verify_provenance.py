"""Verify training-relevant Stage-2p sources against the frozen protocol."""

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
TRAINING_FILES = (
    "v83_spatial_stage2/stage2p/train.py",
    "v83_spatial_stage2/stage2p/rewards.py",
    "v83_spatial_stage2/stage2p/progress.py",
    "v83_spatial_stage2/stage2p/launch.py",
    "v83_spatial_stage2/stage2o/rewards.py",
    "v83_spatial_stage2/stage2k_ppo_agent.py",
    "v83_spatial_stage2/stage2n_train.py",
    "v83_spatial_stage2/stage2n_lanes.py",
    "v83_spatial_stage2/stage2n_runtime.py",
    "v83_spatial_stage2/stage2n_common.py",
)


def main():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    frozen = freeze["code_sha256"]
    rows = []
    for relative in TRAINING_FILES:
        path = PROJECT_ROOT / relative
        current = sha256_file(path) if path.is_file() else None
        expected = frozen.get(relative)
        rows.append({
            "path": relative,
            "frozen_sha256": expected,
            "current_sha256": current,
            "match": expected is not None and current == expected,
        })
    ok = all(row["match"] for row in rows)
    payload = {
        "stage": "stage2p",
        "checked_utc": utc_now(),
        "freeze_created_utc": freeze.get("created_utc"),
        "training_relevant": rows,
        "training_ok": bool(ok),
        "note": (
            "Training-relevant sources must match the freeze; analysis-only "
            "scripts may be improved after launch."
        ),
    }
    atomic_write_json(EVIDENCE / "provenance_check.json", payload)
    print(
        f"[provenance] training_ok={ok} "
        f"({sum(row['match'] for row in rows)}/{len(rows)})",
        flush=True,
    )
    for row in rows:
        if not row["match"]:
            print(
                f"  CHANGED {row['path']}: {row['frozen_sha256']} -> "
                f"{row['current_sha256']}",
                flush=True,
            )
    if not ok:
        raise SystemExit("training-relevant source changed after freeze")


if __name__ == "__main__":
    main()
