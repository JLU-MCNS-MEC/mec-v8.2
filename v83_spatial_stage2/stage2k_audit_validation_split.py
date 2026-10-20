"""Reproduce the Stage-2k DQN validation split used during training.

The training script builds ``spec_payload = load_split_specs("train")`` and
passes it to ``validation_specs``.  This audit reconstructs that exact list,
evaluates ``best.pt`` on it with the Dev topologies (as the training loop did),
and compares the result with the recorded validation and with the true Dev
validation subset.  Read-only; no training and no test data.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    DQN_DIR,
    QUALITY_PRIOR_PATH,
    SCENARIOS,
    VISUAL_PCA_PATH,
    load_split_specs,
    load_topology,
    read_json,
    summarize_episodes,
    utc_now,
    validation_specs,
)
from v83_spatial_stage2.stage2k_dqn import (
    DQNAgent,
    evaluate_dqn,
    flat_quality,
)
from v83_spatial_stage2.stage2k_runtime import (
    SceneCache,
    load_gab_config,
    load_runtime,
)
from v83_spatial_stage2.stage2k_state import VisualProjector

AUDIT_DIR = DQN_DIR / "validation_split_audit"


def subset_from_full_eval(uids):
    path = DQN_DIR / "dev180_eval" / "per_episode.csv"
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["episode_uid"] in uids:
                rows.append(row)
    return rows


def summarize_csv_rows(rows):
    numeric = {
        "success": "SR",
        "ndtw": "nDTW",
        "mean_e2e_ms": "E2E_ms",
        "p95_e2e_ms": "P95_ms",
        "delay_violation_rate": "delay_violation_rate",
        "steps": "mean_steps",
        "joint_reward": "JointR",
    }
    output = {"episodes": len(rows)}
    for key, name in numeric.items():
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        output[name] = float(values.mean())
    return output


def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    train_payload = load_split_specs("train")
    dev_payload = load_split_specs("dev")
    train_val = validation_specs(train_payload)
    dev_val = validation_specs(dev_payload)

    split_counts = Counter(item["split"] for item in train_val)
    band_counts = Counter((item["split"], item["band"]) for item in train_val)
    scenario_counts = Counter(
        (item["split"], item["scenario"]) for item in train_val
    )

    quality_payload = read_json(QUALITY_PRIOR_PATH)
    if quality_payload is None:
        raise SystemExit("Missing quality prior")
    quality = flat_quality(quality_payload)
    projector = VisualProjector.load(VISUAL_PCA_PATH)
    device, encoder, backends = load_runtime("cuda")
    gab_config = load_gab_config()
    topologies = {
        scenario: load_topology(scenario, "dev") for scenario in SCENARIOS
    }

    checkpoint = DQN_DIR / "best.pt"
    agent = DQNAgent(device, seed=20261031)
    payload = agent.load(checkpoint)
    cache = SceneCache(encoder, device, max_entries=60)
    train_rows = evaluate_dqn(
        agent,
        train_val,
        topologies,
        encoder,
        backends,
        projector,
        quality,
        gab_config,
        device,
        scene_cache=cache,
    )
    train_summary = summarize_episodes(train_rows)

    history = json.loads(
        (DQN_DIR / "training_history.json").read_text(encoding="utf-8")
    )
    recorded = None
    for item in history["validation"]:
        if int(item["episode"]) == int(payload.get("episode", -1)):
            recorded = item
            break

    dev_uids = {item["episode_uid"] for item in dev_val}
    dev_rows = subset_from_full_eval(dev_uids)
    dev_summary = summarize_csv_rows(dev_rows) if dev_rows else None

    output = {
        "version": "stage2k-validation-split-audit-v1",
        "created_utc": utc_now(),
        "training_code": (
            "stage2k_dqn.py builds spec_payload = load_split_specs('train') "
            "and passes it to validation_specs(), while the topology payload "
            "is load_topology(scenario, 'dev')."
        ),
        "train_validation_specs": {
            "count": len(train_val),
            "split_counts": dict(split_counts),
            "scenario_counts": {
                f"{split}/{scenario}": count
                for (split, scenario), count in sorted(scenario_counts.items())
            },
            "band_counts": {
                f"{split}/{band}": count
                for (split, band), count in sorted(band_counts.items())
            },
            "episode_uids_first_10": [item["episode_uid"] for item in train_val[:10]],
        },
        "dev_validation_specs": {
            "count": len(dev_val),
            "episode_uids_first_10": [item["episode_uid"] for item in dev_val[:10]],
        },
        "checkpoint": str(checkpoint),
        "checkpoint_episode": int(payload.get("episode", 0)),
        "recorded_validation_at_checkpoint": recorded,
        "reproduced_train_spec_validation": train_summary,
        "dev60_from_full_dev180_eval": dev_summary,
        "reproduction_delta": {
            key: (
                float(train_summary[key]) - float(recorded[key])
                if recorded and key in recorded and key in train_summary
                else None
            )
            for key in ("SR", "nDTW", "E2E_ms", "JointR", "mean_steps")
        },
    }
    path = AUDIT_DIR / "validation_split_audit.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print("WROTE", path)


if __name__ == "__main__":
    main()
