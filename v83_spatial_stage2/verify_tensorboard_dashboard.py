#!/usr/bin/env python
"""Verify that the running TensorBoard serves real scalar data.

A 200 response from ``/`` is not sufficient: if TensorBoard is started with
``--samples_per_plugin scalars=0`` the tag list is populated but every scalar
series is empty.  This check reads the *served* scalar endpoints for every
dashboard run and fails if a required tag has no points.

Usage:
    python v83_spatial_stage2/verify_tensorboard_dashboard.py \
        --url http://127.0.0.1:6006
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "v83_spatial_stage2"
    / "tensorboard_dashboard_config.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "runs"
    / "tensorboard"
    / "rl_stage2k_stage2l_dashboard"
    / "SERVICE_VERIFICATION.json"
)

COMMON_TAGS = (
    "train/jointr",
    "train/jointr_ma100",
    "train/navigation_reward",
    "train/navigation_reward_ma100",
    "train/latency_penalty",
    "train/latency_penalty_ma100",
    "train/sr",
    "train/ndtw",
    "train/e2e_ms",
    "validation/jointr",
    "validation/sr",
    "validation/ndtw",
    "validation/e2e_ms",
)
DQN_TAGS = (
    "dqn/td_loss",
    "dqn/q_mean",
    "dqn/q_max",
    "dqn/grad_norm",
    "dqn/epsilon",
)
PPO_TAGS = (
    "ppo/policy_loss",
    "ppo/value_loss",
    "ppo/policy_entropy",
    "ppo/approx_kl",
    "ppo/clip_fraction",
    "ppo/explained_variance",
)


def _get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _scalar_url(base_url, run, tag):
    query = urllib.parse.urlencode({"run": run, "tag": tag})
    return f"{base_url}/data/plugin/scalars/scalars?{query}"


def _required_tags(algorithm):
    return COMMON_TAGS + (DQN_TAGS if algorithm == "dqn" else PPO_TAGS)


def _collect(config, base_url):
    runs = config["runs"]
    expected_runs = [run["key"] for run in runs]
    served_runs = _get_json(f"{base_url}/data/runs")
    if sorted(served_runs) != sorted(expected_runs):
        raise RuntimeError(
            "run set mismatch: "
            f"served={served_runs}, expected={expected_runs}"
        )

    tag_index = _get_json(f"{base_url}/data/plugin/scalars/tags")
    reports = []
    failures = []
    for run in runs:
        key = run["key"]
        required = _required_tags(run["algorithm"])
        available = tag_index.get(key, {})
        missing_tags = [tag for tag in required if tag not in available]
        counts = {}
        for tag in required:
            if tag in missing_tags:
                counts[tag] = 0
                continue
            points = _get_json(_scalar_url(base_url, key, tag))
            counts[tag] = len(points)
            if not points:
                missing_tags.append(tag)
        expected_train = int(run["train_episodes"])
        expected_validation = int(run["validation_points"])
        expected_jointr = expected_train
        expected_ma100 = max(expected_train - 99, 0)
        if counts.get("train/jointr") != expected_jointr:
            failures.append(
                f"{key}: train/jointr has {counts.get('train/jointr')} "
                f"points, expected {expected_jointr}"
            )
        if counts.get("train/jointr_ma100") != expected_ma100:
            failures.append(
                f"{key}: train/jointr_ma100 has "
                f"{counts.get('train/jointr_ma100')} points, "
                f"expected {expected_ma100}"
            )
        if counts.get("validation/jointr") != expected_validation:
            failures.append(
                f"{key}: validation/jointr has "
                f"{counts.get('validation/jointr')} points, "
                f"expected {expected_validation}"
            )
        if missing_tags:
            failures.append(f"{key}: missing/empty tags {sorted(missing_tags)}")
        reports.append(
            {
                "key": key,
                "algorithm": run["algorithm"],
                "required_tags": list(required),
                "point_counts": counts,
                "missing_or_empty_tags": sorted(set(missing_tags)),
            }
        )
    return reports, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:6006")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    deadline = time.monotonic() + max(0, args.timeout)
    last_error = None
    reports = []
    failures = []

    while True:
        try:
            reports, failures = _collect(config, base_url)
        except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
            last_error = str(exc)
            failures = [last_error]
        if not failures:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(1)

    payload = {
        "url": base_url,
        "verified": not failures,
        "timeout_seconds": args.timeout,
        "runs": reports,
        "failures": failures,
        "last_error": last_error,
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"SERVICE_VERIFICATION: {'PASS' if not failures else 'FAIL'} "
        f"url={base_url} report={args.output}"
    )
    for report in reports:
        print(
            f"  {report['key']}: "
            f"jointr={report['point_counts'].get('train/jointr', 0)} "
            f"tags={len(report['required_tags'])} "
            f"missing={report['missing_or_empty_tags']}"
        )
    for failure in failures:
        print(f"  FAIL: {failure}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
