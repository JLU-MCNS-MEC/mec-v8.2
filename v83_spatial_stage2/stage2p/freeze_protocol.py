"""Freeze the Stage-2p protocol after unit tests and six smoke runs pass."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
)
from v83_spatial_stage2.stage2n_common import (  # noqa: E402
    MEC_CHECKPOINT_SHA256,
    PROTOCOL_PATH as STAGE2N_PROTOCOL_PATH,
)
from v83_spatial_stage2.stage2p.rewards import (  # noqa: E402
    H_MAX,
    SCHEME_DESCRIPTIONS,
    SCHEMES,
    TAU_MS,
)
from v83_spatial_stage2.stage2p.train import (  # noqa: E402
    RUNS_ROOT,
    STAGE2O_FREEZE,
    _source_hashes,
)

EVIDENCE = Path(__file__).resolve().parent / "evidence"
SMOKE_ROOT = RUNS_ROOT / "smoke"
SEEDS = (0, 1, 2)
TOLERANCE = 1e-8


def _git_state():
    def run(command):
        return subprocess.check_output(
            command, cwd=str(PROJECT_ROOT), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
    }


def _unit_tests():
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "test_stage2p.py"),
    ]
    completed = subprocess.run(
        command, cwd=str(PROJECT_ROOT), text=True, capture_output=True,
    )
    output = completed.stdout + completed.stderr
    passed = completed.returncode == 0 and "OK" in output
    summary = [
        line for line in output.splitlines()
        if line.startswith("Ran ") or line.strip() in ("OK", "FAILED")
    ]
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "passed": bool(passed),
        "summary": summary,
        "output_tail": output.splitlines()[-8:],
    }


def _smoke_checks():
    runs = []
    max_error = 0.0
    errors = []
    for scheme in SCHEMES:
        for seed in SEEDS:
            run_dir = SMOKE_ROOT / f"{scheme}_ppo_s{seed}"
            history = read_json(run_dir / "training_history.json")
            if history is None or not history.get("train"):
                errors.append(f"missing smoke history: {run_dir}")
                continue
            rows = history["train"]
            reward_errors = [
                float(row.get("reward_abs_error", 0.0)) for row in rows
            ]
            max_error = max(max_error, max(reward_errors or [0.0]))
            if max(reward_errors or [0.0]) > TOLERANCE:
                errors.append(
                    f"{scheme}/s{seed}: reward identity max="
                    f"{max(reward_errors):.3e}"
                )
            if scheme == "baseline":
                offset_errors = [
                    abs(float(row["episode_return"])
                        - float(row["latency_return"]))
                    for row in rows
                ]
            else:
                offset_errors = [
                    abs(float(row["episode_return"])
                        - 1.0 - 4.0 * float(row["latency_return"]))
                    for row in rows
                ]
            if max(offset_errors or [0.0]) > TOLERANCE:
                errors.append(
                    f"{scheme}/s{seed}: return-offset max="
                    f"{max(offset_errors):.3e}"
                )
            latency_errors = [
                abs(float(row["latency_return"])
                    + 0.25 * float(row["mean_episode_e2e_ms"]) / 100.0)
                for row in rows
            ]
            if max(latency_errors or [0.0]) > TOLERANCE:
                errors.append(
                    f"{scheme}/s{seed}: Stage-2n latency-return max="
                    f"{max(latency_errors):.3e}"
                )
            runs.append({
                "scheme": scheme,
                "seed": seed,
                "episodes": len(rows),
                "validations": len(history.get("validation", [])),
                "max_reward_abs_error": max(reward_errors or [0.0]),
                "max_offset_error": max(offset_errors or [0.0]),
                "max_latency_return_error": max(latency_errors or [0.0]),
                "run_dir": str(run_dir),
            })
    return {
        "runs": runs,
        "max_reward_abs_error": float(max_error),
        "errors": errors,
        "passed": bool(not errors and len(runs) == 6),
        "tolerance": TOLERANCE,
    }


def main():
    tests = _unit_tests()
    smoke = _smoke_checks()
    if not tests["passed"]:
        raise SystemExit(f"unit tests failed:\n{tests['output_tail']}")
    if not smoke["passed"]:
        raise SystemExit(
            "smoke checks failed:\n  " + "\n  ".join(smoke["errors"])
        )
    parent2n = read_json(STAGE2N_PROTOCOL_PATH)
    if parent2n is None:
        raise SystemExit("Stage-2n protocol freeze is missing")
    parent2o = read_json(STAGE2O_FREEZE)
    if parent2o is None:
        raise SystemExit("Stage-2o protocol freeze is missing")
    audit = read_json(EVIDENCE / "reuse_audit.json")
    if audit is None or not (
        audit.get("task_hashes_match") and audit.get("ppo_configs_match")
    ):
        raise SystemExit("reuse audit missing or failed")
    payload = {
        "stage": "stage2p",
        "created_utc": utc_now(),
        "git": _git_state(),
        "parent_stage2n": {
            "protocol_path": str(STAGE2N_PROTOCOL_PATH),
            "protocol_sha256": sha256_file(STAGE2N_PROTOCOL_PATH),
            "train_tasks_sha256": parent2n["tasks"]["train"]["sha256"],
            "dev_tasks_sha256": parent2n["tasks"]["dev"]["sha256"],
            "test_tasks_sha256": parent2n["tasks"]["test"]["sha256"],
            "training_order_sha256": parent2n["training_order"]["sha256"],
            "deployment_sha256": parent2n["deployment"]["sha256"],
            "mec_checkpoint_sha256": MEC_CHECKPOINT_SHA256,
        },
        "parent_stage2o": {
            "protocol_freeze_path": str(STAGE2O_FREEZE),
            "protocol_freeze_sha256": sha256_file(STAGE2O_FREEZE),
        },
        "reuse_audit": {
            "path": str(EVIDENCE / "reuse_audit.json"),
            "sha256": sha256_file(EVIDENCE / "reuse_audit.json"),
            "decision": audit.get("decision"),
        },
        "reward": {
            "tau_ms": TAU_MS,
            "horizon": H_MAX,
            "gamma": 1.0,
            "schemes": {
                scheme: SCHEME_DESCRIPTIONS[scheme]
                for scheme in SCHEMES
            },
            "alignment": "4 * R_baseline + 1 = R_D",
            "navigation_reward": "removed",
            "success_bonus": "none",
        },
        "training": {
            "runs": 6,
            "schemes": list(SCHEMES),
            "seeds": list(SEEDS),
            "episodes": 4000,
            "validation_every": 200,
            "validation_split": "dev",
            "dev_tasks": 60,
            "test_used": False,
            "convergence_criterion": (
                "max(last3 Dev E2E) - min(last3 Dev E2E) <= 0.01 * best "
                "and last <= 1.01 * best"
            ),
            "checkpoint_selection": (
                "lowest Dev mean E2E; ties: higher SR, then earlier episode"
            ),
        },
        "code_sha256": _source_hashes(),
        "unit_tests": tests,
        "smoke": smoke,
        "test_sealed": True,
        "final_test_read": False,
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE / "protocol_freeze.json"
    atomic_write_json(path, payload)
    print(
        f"[freeze] wrote {path}\n"
        f"[freeze] unit tests: {tests['summary']}\n"
        f"[freeze] smoke runs: {len(smoke['runs'])}, "
        f"max reward abs error={smoke['max_reward_abs_error']:.3e}\n"
        f"[freeze] git commit={payload['git']['commit']} "
        f"dirty={payload['git']['dirty']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
