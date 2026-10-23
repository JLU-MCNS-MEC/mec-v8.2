"""Launch the six Stage-2p PPO runs (baseline/D x seeds 0/1/2)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    read_json,
    utc_now,
)
from v83_spatial_stage2.stage2p.rewards import SCHEMES  # noqa: E402
from v83_spatial_stage2.stage2p.train import (  # noqa: E402
    FORMAL_EPISODES,
    FORMAL_SEEDS,
    FORMAL_VALIDATION,
    LOGS_ROOT,
    RUNS_ROOT,
)

PYTHON = sys.executable


def _gpu_memory():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        used, total = [
            int(part.strip())
            for part in output.strip().splitlines()[0].split(",")
        ]
        return {"used_mb": used, "total_mb": total, "free_mb": total - used}
    except Exception:
        return {"used_mb": None, "total_mb": None, "free_mb": None}


def _build_jobs(args):
    jobs = []
    for scheme in SCHEMES:
        for seed in args.seeds:
            if args.smoke:
                run_dir = (
                    RUNS_ROOT / "smoke" / f"{scheme}_ppo_s{seed}"
                )
                log_path = (
                    LOGS_ROOT / "smoke" / f"{scheme}_ppo_s{seed}.log"
                )
            else:
                run_dir = RUNS_ROOT / scheme / f"ppo_s{seed}"
                log_path = LOGS_ROOT / f"{scheme}_ppo_s{seed}.log"
            command = [
                PYTHON, "-u",
                str(PROJECT_ROOT / "v83_spatial_stage2" / "stage2p"
                    / "train.py"),
                "--scheme", scheme,
                "--seed", str(seed),
                "--max-episodes", str(args.max_episodes),
                "--validation-every", str(args.validation_every),
                "--device", args.device,
                "--threads", str(args.threads),
                "--run-dir", str(run_dir),
                "--tensorboard-dir", str(run_dir / "tensorboard"),
            ]
            if args.smoke:
                command.append("--smoke")
            if (run_dir / "checkpoints" / "last_full.pt").is_file():
                command.append("--resume")
            jobs.append({
                "scheme": scheme,
                "seed": int(seed),
                "run_dir": str(run_dir),
                "log": str(log_path),
                "command": command,
                "process": None,
                "handle": None,
                "pid": None,
                "started_utc": None,
                "finished_utc": None,
                "returncode": None,
                "restarts": 0,
            })
    return jobs


def _write_status(jobs, gpu_before, gpu_after, args):
    payload = {
        "stage": "stage2p",
        "created_utc": utc_now(),
        "python": PYTHON,
        "smoke": bool(args.smoke),
        "max_episodes": int(args.max_episodes),
        "validation_every": int(args.validation_every),
        "seeds": list(args.seeds),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "jobs": [
            {key: value for key, value in job.items()
             if key not in ("process", "handle")}
            for job in jobs
        ],
    }
    path = LOGS_ROOT / ("smoke" if args.smoke else "")
    path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path / "launch.json", payload)


def _start(job, handle):
    job["started_utc"] = utc_now()
    job["handle"] = handle
    job["process"] = subprocess.Popen(
        job["command"], cwd=str(PROJECT_ROOT), stdout=handle,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    job["pid"] = int(job["process"].pid)
    print(
        f"[launch] started {job['scheme']}/s{job['seed']} "
        f"pid={job['pid']} log={job['log']}", flush=True,
    )


def _progress_line(job):
    payload = read_json(Path(job["run_dir"]) / "progress.json") or {}
    if not payload:
        return f"{job['scheme']}/s{job['seed']}: starting"
    return (
        f"{job['scheme']}/s{job['seed']}: "
        f"Ep{payload.get('current_episode', 0)}/"
        f"{payload.get('max_episodes', '?')} "
        f"best={payload.get('best_dev_e2e_ms')} "
        f"status={payload.get('status')}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(FORMAL_SEEDS))
    parser.add_argument("--max-episodes", type=int, default=FORMAL_EPISODES)
    parser.add_argument("--validation-every", type=int,
                        default=FORMAL_VALIDATION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--min-free-mb", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--max-restarts", type=int, default=2)
    args = parser.parse_args()
    if args.smoke:
        if args.max_episodes == FORMAL_EPISODES:
            args.max_episodes = 30
        if args.validation_every == FORMAL_VALIDATION:
            args.validation_every = min(30, args.max_episodes)
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be >= 1")

    gpu_before = _gpu_memory()
    print(
        f"[launch] GPU before: used={gpu_before['used_mb']} MB, "
        f"free={gpu_before['free_mb']} MB", flush=True,
    )
    if (gpu_before["free_mb"] is not None
            and gpu_before["free_mb"] < args.min_free_mb):
        raise SystemExit(
            f"only {gpu_before['free_mb']} MB GPU free; "
            f"require {args.min_free_mb} MB. Existing jobs left alone."
        )

    jobs = _build_jobs(args)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    pending, running, completed = list(jobs), [], []
    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            path = Path(job["log"])
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8")
            handle.write(
                f"===== launch {utc_now()} =====\n"
                + " ".join(job["command"]) + "\n"
            )
            handle.flush()
            _start(job, handle)
            running.append(job)
        time.sleep(max(1.0, float(args.poll_seconds)))
        still = []
        for job in running:
            code = job["process"].poll()
            if code is None:
                still.append(job)
                continue
            job["returncode"] = int(code)
            job["finished_utc"] = utc_now()
            job["handle"].close()
            if code != 0 and job["restarts"] < int(args.max_restarts):
                last_full = (
                    Path(job["run_dir"]) / "checkpoints" / "last_full.pt"
                )
                if last_full.is_file():
                    job["restarts"] += 1
                    if "--resume" not in job["command"]:
                        job["command"] = job["command"] + ["--resume"]
                    handle = Path(job["log"]).open("a", encoding="utf-8")
                    handle.write(
                        f"===== restart {job['restarts']} {utc_now()} "
                        f"after rc={code} =====\n"
                    )
                    handle.flush()
                    _start(job, handle)
                    still.append(job)
                    print(
                        f"[launch] restarting {job['scheme']}/s{job['seed']}",
                        flush=True,
                    )
                    continue
            completed.append(job)
            print(
                f"[launch] finished {job['scheme']}/s{job['seed']} "
                f"returncode={code}", flush=True,
            )
        running = still
        for job in running:
            print("[status] " + _progress_line(job), flush=True)
        _write_status(jobs, gpu_before, _gpu_memory(), args)
        if not args.wait:
            break

    gpu_after = _gpu_memory()
    _write_status(jobs, gpu_before, gpu_after, args)
    failures = [job for job in completed if job["returncode"] != 0]
    print(
        f"[launch] done: completed={len(completed)} failures={len(failures)} "
        f"GPU after: used={gpu_after['used_mb']} MB", flush=True,
    )
    if failures:
        for job in failures:
            print(
                f"[launch] FAILED {job['scheme']}/s{job['seed']} "
                f"rc={job['returncode']} log={job['log']}", flush=True,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
