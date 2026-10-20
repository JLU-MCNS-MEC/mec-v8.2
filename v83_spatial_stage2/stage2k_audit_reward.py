"""Audit the recorded Stage-2k reward magnitude against the JointR definition.

Reads only existing training history and baseline per-episode records.  It
quantifies the navigation/latency balance, the cumulative penalty mismatch
caused by ``REWARD_DELAY_DIVISOR=200``, and the relationship between training
return and validation JointR.  No training is started.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BASELINE_DIR,
    DQN_DIR,
    JOINTR_DELAY_WEIGHT,
    REWARD_DELAY_DIVISOR,
)

AUDIT_DIR = Path("runs/v83_spatial_stage2/stage2k/reward_audit")
BANDS = ("short", "medium", "long")


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "n": 0, "mean": None, "std": None, "min": None,
            "max": None, "median": None,
        }
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
    }


def block(values, low, high, episodes):
    mask = (episodes >= low) & (episodes <= high)
    return stats(np.asarray(values)[mask])


def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    history = json.loads(
        (DQN_DIR / "training_history.json").read_text(encoding="utf-8")
    )
    train = history["train"]
    validation = history["validation"]

    episodes = np.asarray([row["episode"] for row in train], dtype=np.int64)
    nav = np.asarray(
        [row["navigation_reward"] for row in train], dtype=np.float64
    )
    lat = np.asarray(
        [row["latency_penalty"] for row in train], dtype=np.float64
    )
    ret = np.asarray([row["return"] for row in train], dtype=np.float64)
    e2e = np.asarray([row["mean_e2e_ms"] for row in train], dtype=np.float64)
    steps = np.asarray([row["steps"] for row in train], dtype=np.float64)
    success = np.asarray([row["success"] for row in train], dtype=np.float64)

    # The NavigationRewardTracker telescopes to the final nDTW, so the
    # recorded navigation_reward is the episode nDTW.
    joint_penalty = JOINTR_DELAY_WEIGHT * e2e / 100.0
    implied_joint = nav - joint_penalty
    latency_vs_joint = lat / (-joint_penalty)
    reward_vs_joint = ret - implied_joint
    expected_ratio = steps / REWARD_DELAY_DIVISOR

    # Confirm the telescoping property on the saved baseline records.
    baseline_rows = list(csv.DictReader(
        (BASELINE_DIR / "dev" / "per_episode.csv").open(encoding="utf-8")
    ))
    telescope_errors = []
    latency_sum_errors = []
    for row in baseline_rows:
        n_steps = float(row["steps"])
        nav_sum = float(row["mean_navigation_reward"]) * n_steps
        lat_sum = float(row["mean_latency_penalty"]) * n_steps
        telescope_errors.append(abs(nav_sum - float(row["ndtw"])))
        latency_sum_errors.append(abs(
            lat_sum - float(row["mean_latency_penalty"]) * n_steps
        ))

    # Relationship between training return and validation JointR.  Use the
    # trailing 200-episode mean return ending at each validation episode.
    val_episodes = np.asarray(
        [row["episode"] for row in validation], dtype=np.int64
    )
    val_joint = np.asarray(
        [row["joint_reward"] for row in validation], dtype=np.float64
    )
    trailing_return = []
    for episode in val_episodes:
        mask = (episodes > int(episode) - 200) & (episodes <= int(episode))
        trailing_return.append(float(ret[mask].mean()))
    trailing_return = np.asarray(trailing_return)

    by_band = {}
    for band in BANDS:
        mask = np.asarray([row["band"] == band for row in train])
        if not mask.any():
            by_band[band] = {"episodes": 0, "not_trained": True}
            continue
        by_band[band] = {
            "episodes": int(mask.sum()),
            "steps": stats(steps[mask]),
            "navigation_reward": stats(nav[mask]),
            "latency_penalty": stats(lat[mask]),
            "abs_latency_over_nav": float(
                abs(lat[mask].mean()) / nav[mask].mean()
            ) if nav[mask].mean() else None,
            "latency_sum_over_joint_penalty": stats(latency_vs_joint[mask]),
            "expected_ratio_steps_over_200": stats(expected_ratio[mask]),
            "return": stats(ret[mask]),
            "implied_joint": stats(implied_joint[mask]),
        }

    payload = {
        "version": "stage2k-reward-audit-v1",
        "source": "runs/v83_spatial_stage2/stage2k/dqn/training_history.json",
        "formula": (
            "r_t = r_navigation_t - 0.25 * min(E2E_ms/100, 3.0) / 200; "
            "JointR = final_nDTW - 0.25 * mean(E2E_ms)/100"
        ),
        "constants": {
            "jointr_delay_weight": JOINTR_DELAY_WEIGHT,
            "reward_delay_divisor": REWARD_DELAY_DIVISOR,
            "reward_delay_clip": 3.0,
            "episodes": int(len(train)),
        },
        "telescoping_check": {
            "baseline_rows": len(baseline_rows),
            "max_abs_nav_sum_minus_ndtw": float(max(telescope_errors)),
            "mean_abs_nav_sum_minus_ndtw": float(np.mean(telescope_errors)),
        },
        "overall": {
            "navigation_reward": stats(nav),
            "latency_penalty": stats(lat),
            "return": stats(ret),
            "mean_e2e_ms": stats(e2e),
            "steps": stats(steps),
            "abs_latency_over_nav": float(abs(lat.mean()) / nav.mean()),
            "per_step_nav": float(nav.mean() / steps.mean()),
            "per_step_latency": float(lat.mean() / steps.mean()),
            "per_step_abs_latency_over_nav": float(
                abs(lat.mean() / steps.mean()) / (nav.mean() / steps.mean())
            ),
            "joint_penalty": stats(joint_penalty),
            "implied_joint": stats(implied_joint),
            "latency_sum_over_joint_penalty": stats(latency_vs_joint),
            "expected_ratio_steps_over_200": stats(expected_ratio),
            "return_minus_implied_joint": stats(reward_vs_joint),
        },
        "blocks": {
            f"{low}-{high}": {
                "return": block(ret, low, high, episodes),
                "navigation_reward": block(nav, low, high, episodes),
                "latency_penalty": block(lat, low, high, episodes),
                "mean_e2e_ms": block(e2e, low, high, episodes),
                "success": block(success, low, high, episodes),
            }
            for low, high in ((1, 500), (501, 1000), (1001, 1500), (1501, 2000))
        },
        "by_band": by_band,
        "correlations": {
            "return_vs_mean_e2e": float(np.corrcoef(ret, e2e)[0, 1]),
            "navigation_vs_mean_e2e": float(np.corrcoef(nav, e2e)[0, 1]),
            "return_vs_success": float(np.corrcoef(ret, success)[0, 1]),
            "implied_joint_vs_mean_e2e": float(
                np.corrcoef(implied_joint, e2e)[0, 1]
            ),
            "trailing_return_vs_validation_joint": float(
                np.corrcoef(trailing_return, val_joint)[0, 1]
            ),
        },
        "validation": {
            "episodes": val_episodes.tolist(),
            "joint_reward": val_joint.tolist(),
            "trailing_200_return": trailing_return.tolist(),
            "best_joint": float(val_joint.max()),
            "best_episode": int(val_episodes[int(np.argmax(val_joint))]),
            "last_joint": float(val_joint[-1]),
            "last_episode": int(val_episodes[-1]),
        },
        "marginal_penalty_per_100ms": float(
            JOINTR_DELAY_WEIGHT * 1.0 / REWARD_DELAY_DIVISOR
        ),
        "jointr_penalty_per_100ms": float(JOINTR_DELAY_WEIGHT),
        "note": (
            "per-step reward penalty for a 100 ms difference is 0.00125; "
            "over T steps it is (T/200) times the JointR penalty difference. "
            "For the observed mean T this is about 0.665, i.e. the reward "
            "underweights latency relative to JointR by roughly one third."
        ),
    }

    (AUDIT_DIR / "reward_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Per-episode audit CSV for further inspection (no aggregation loss).
    with (AUDIT_DIR / "per_episode_reward.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "episode", "band", "steps", "return", "navigation_reward",
                "latency_penalty", "mean_e2e_ms", "joint_penalty",
                "implied_joint", "latency_sum_over_joint_penalty",
                "expected_ratio_steps_over_200", "return_minus_implied_joint",
                "success",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for index in range(len(train)):
            writer.writerow({
                "episode": int(episodes[index]),
                "band": train[index]["band"],
                "steps": int(steps[index]),
                "return": float(ret[index]),
                "navigation_reward": float(nav[index]),
                "latency_penalty": float(lat[index]),
                "mean_e2e_ms": float(e2e[index]),
                "joint_penalty": float(joint_penalty[index]),
                "implied_joint": float(implied_joint[index]),
                "latency_sum_over_joint_penalty": float(latency_vs_joint[index]),
                "expected_ratio_steps_over_200": float(expected_ratio[index]),
                "return_minus_implied_joint": float(reward_vs_joint[index]),
                "success": int(success[index]),
            })

    print(json.dumps({
        "overall": payload["overall"],
        "blocks": payload["blocks"],
        "correlations": payload["correlations"],
        "telescoping_check": payload["telescoping_check"],
        "by_band": payload["by_band"],
        "validation": {
            "episodes": val_episodes.tolist(),
            "joint": [round(float(x), 6) for x in val_joint],
            "trailing_return": [round(float(x), 6) for x in trailing_return],
        },
    }, indent=2, ensure_ascii=False))
    print("WROTE", AUDIT_DIR / "reward_audit.json")


if __name__ == "__main__":
    main()
