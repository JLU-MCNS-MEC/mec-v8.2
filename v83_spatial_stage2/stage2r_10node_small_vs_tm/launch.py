"""Launch the 12 Stage-2r runs (2 deployments x 2 algorithms x 3 seeds).

The launcher never stops or modifies an existing process.  It starts each
run in its own session, writes ``launch.json``, and with ``--wait`` monitors
until all runs exit.  A crashed run is resumed from ``last_full.pt``.
Formal runs are started with ``--lane-cache`` (exact frozen-lane replay);
build the two caches with ``lane_cache.py`` before launching.

Formal order interleaves conditions so the first parallel wave already
covers both deployments and both algorithms:

    seed 11: Small-DQN, Small-PPO, TM-DQN, TM-PPO
    seed 22: Small-DQN, Small-PPO, TM-DQN, TM-PPO
    seed 33: ...

Usage:
    python v83_spatial_stage2/stage2r_10node_small_vs_tm/launch.py --wait
    python v83_spatial_stage2/stage2r_10node_small_vs_tm/launch.py \
        --smoke --max-episodes 20 --wait
"""

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
from v83_spatial_stage2.stage2r_10node_small_vs_tm.common import (  # noqa: E402
    ALGORITHMS,
    FORMAL_SEEDS,
    FORMAL_VALIDATION,
    MAX_EPISODES,
    MEC_MODELS,
    RUNS_ROOT,
    model_slug,
    run_dir as default_run_dir,
)

PYTHON = sys.executable
LOGS_ROOT = RUNS_ROOT / "logs"


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
        return {"used_mb": used, "total_mb": total,
                "free_mb": total - used}
    except Exception:
        return {"used_mb": None, "total_mb": None, "free_mb": None}


def _formal_order():
    jobs = []
    for seed in FORMAL_SEEDS:
        for model in MEC_MODELS:
            for algorithm in ALGORITHMS:
                jobs.append((model, algorithm, seed))
    return jobs


def _build_jobs(args):
    combos = (
        [(model, algorithm, 11)
         for model in MEC_MODELS for algorithm in ALGORITHMS]
        if args.smoke else _formal_order()
    )
    jobs = []
    for model, algorithm, seed in combos:
        if args.smoke:
            run_dir = (
                RUNS_ROOT / "smoke"
                / f"{model_slug(model)}_{algorithm}_s{seed}"
            )
            log_path = (
                LOGS_ROOT / "smoke"
                / f"{model_slug(model)}_{algorithm}_s{seed}.log"
            )
        else:
            run_dir = default_run_dir(model, algorithm, seed)
            log_path = (
                LOGS_ROOT
                / f"{model_slug(model)}_{algorithm}_s{seed}.log"
            )
        command = [
            PYTHON, "-u",
            str(PROJECT_ROOT / "v83_spatial_stage2"
                / "stage2r_10node_small_vs_tm" / "train.py"),
            "--model", model,
            "--algorithm", algorithm,
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
        else:
            command.append("--lane-cache")
        if (run_dir / "checkpoints" / "last_full.pt").is_file():
            command.append("--resume")
        jobs.append({
            "model": model,
            "algorithm": algorithm,
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


def _write_status(jobs, args, gpu_before, gpu_after):
    path = LOGS_ROOT / ("smoke" if args.smoke else "")
    path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path / "launch.json", {
        "stage": "stage2r_10node_small_vs_tm",
        "created_utc": utc_now(),
        "python": PYTHON,
        "smoke": bool(args.smoke),
        "max_episodes": int(args.max_episodes),
        "validation_every": int(args.validation_every),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "jobs": [
            {key: value for key, value in job.items()
             if key not in ("process", "handle")}
            for job in jobs
        ],
    })


def _start(job, log_handle):
    job["started_utc"] = utc_now()
    job["handle"] = log_handle
    job["process"] = subprocess.Popen(
        job["command"], cwd=str(PROJECT_ROOT), stdout=log_handle,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    job["pid"] = int(job["process"].pid)
    print(
        f"[launch] started {job['model']}/{job['algorithm']}/s{job['seed']} "
        f"pid={job['pid']} log={job['log']}",
        flush=True,
    )


def _progress_line(job):
    payload = read_json(Path(job["run_dir"]) / "progress.json") or {}
    if not payload:
        return (
            f"{job['model']}/{job['algorithm']}/s{job['seed']}: "
            f"starting (pid={job['pid']})"
        )
    counts = payload.get("training_counts") or {}
    last = payload.get("last_validation") or {}
    return (
        f"{job['model']}/{job['algorithm']}/s{job['seed']}: "
        f"Ep{payload.get('current_episode', 0)}/"
        f"{payload.get('max_episodes', '?')} "
        f"succ={counts.get('cum_success')} fail={counts.get('cum_failure')} "
        f"bestDevE2E={payload.get('best_dev_e2e_ms')} "
        f"lastDevE2E={last.get('E2E_ms')} status={payload.get('status')}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--validation-every", type=int,
                        default=FORMAL_VALIDATION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--min-free-mb", type=int, default=12000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-restarts", type=int, default=2)
    args = parser.parse_args()
    if args.smoke:
        if args.max_episodes == MAX_EPISODES:
            args.max_episodes = 20
        if args.validation_every == FORMAL_VALIDATION:
            args.validation_every = min(20, args.max_episodes)
        args.max_parallel = min(args.max_parallel, 4)
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be >= 1")

    gpu_before = _gpu_memory()
    print(
        f"[launch] GPU before: used={gpu_before['used_mb']} MB, "
        f"free={gpu_before['free_mb']} MB",
        flush=True,
    )
    if (
        gpu_before["free_mb"] is not None
        and gpu_before["free_mb"] < args.min_free_mb
    ):
        raise SystemExit(
            f"only {gpu_before['free_mb']} MB GPU free; require "
            f"{args.min_free_mb} MB.  Existing jobs are left alone."
        )

    jobs = _build_jobs(args)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    running = []
    completed = []
    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            log_path = Path(job["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            handle.write(
                f"===== launch {utc_now()} =====\n"
                + " ".join(job["command"]) + "\n"
            )
            handle.flush()
            _start(job, handle)
            running.append(job)
        time.sleep(max(1.0, float(args.poll_seconds)))
        still_running = []
        for job in running:
            code = job["process"].poll()
            if code is None:
                still_running.append(job)
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
                    log_path = Path(job["log"])
                    handle = log_path.open("a", encoding="utf-8")
                    handle.write(
                        f"===== restart {job['restarts']} {utc_now()} "
                        f"after returncode={code} =====\n"
                    )
                    handle.flush()
                    _start(job, handle)
                    still_running.append(job)
                    print(
                        f"[launch] restarting {job['model']}/"
                        f"{job['algorithm']}/s{job['seed']} from {last_full}",
                        flush=True,
                    )
                    continue
            completed.append(job)
            print(
                f"[launch] finished {job['model']}/{job['algorithm']}/"
                f"s{job['seed']} returncode={code}",
                flush=True,
            )
        running = still_running
        for job in running:
            print("[status] " + _progress_line(job), flush=True)
        _write_status(jobs, args, gpu_before, _gpu_memory())
        if not args.wait:
            break

    gpu_after = _gpu_memory()
    _write_status(jobs, args, gpu_before, gpu_after)
    failures = [job for job in completed if job["returncode"] != 0]
    print(
        f"[launch] done: completed={len(completed)} "
        f"failures={len(failures)} GPU after: used={gpu_after['used_mb']} MB",
        flush=True,
    )
    if failures:
        for job in failures:
            print(
                f"[launch] FAILED {job['model']}/{job['algorithm']}/"
                f"s{job['seed']} rc={job['returncode']} log={job['log']}",
                flush=True,
            )
        raise SystemExit(1)
    print("STAGE2R_LAUNCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
