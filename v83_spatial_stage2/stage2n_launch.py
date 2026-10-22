"""Launch the six Stage-2n formal runs (DQN/PPO × seeds 0/1/2).

The launcher never stops or modifies any existing process.  It checks free GPU
memory, starts the requested runs in separate sessions, writes ``launch.json``
and monitors every run's ``progress.json`` until all have exited.

    python v83_spatial_stage2/stage2n_launch.py

Use ``--max-parallel`` to queue runs if GPU memory is tight.  Logs are written
under ``runs/v83_spatial_stage2/stage2n_latency_only/logs/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    read_json,
    utc_now,
)
from v83_spatial_stage2.stage2n_common import (  # noqa: E402
    AGENT_SEEDS,
    LOG_DIR,
    MEC_CHECKPOINT_SHA256,
    PROTOCOL_PATH,
    RUNS_DIR,
    TRAIN_EPISODES,
    VALIDATION_EVERY,
)

PYTHON = sys.executable


def _gpu_memory():
    try:
        output = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.DEVNULL)
        used, total = [int(part.strip())
                       for part in output.strip().splitlines()[0].split(",")]
        return {"used_mb": used, "total_mb": total, "free_mb": total - used}
    except Exception:
        return {"used_mb": None, "total_mb": None, "free_mb": None}


def _build_jobs(args):
    jobs = []
    for algorithm in args.algorithms:
        for seed in args.seeds:
            run_dir = RUNS_DIR / f"{algorithm}_s{seed}"
            log_path = LOG_DIR / f"{algorithm}_s{seed}.log"
            command = [
                PYTHON, "-u",
                str(PROJECT_ROOT / "v83_spatial_stage2" / "stage2n_train.py"),
                "--algorithm", algorithm,
                "--seed", str(seed),
                "--max-episodes", str(TRAIN_EPISODES),
                "--validation-every", str(VALIDATION_EVERY),
                "--device", args.device,
                "--threads", str(args.threads),
                "--run-dir", str(run_dir),
                "--tensorboard-dir", str(run_dir / "tensorboard"),
            ]
            if (run_dir / "checkpoints" / "last_full.pt").is_file():
                command.append("--resume")
            jobs.append({
                "algorithm": algorithm,
                "seed": int(seed),
                "run_dir": str(run_dir),
                "log": str(log_path),
                "command": command,
                "process": None,
                "started_utc": None,
                "finished_utc": None,
                "returncode": None,
            })
    return jobs


def _write_status(jobs, gpu_before, gpu_after):
    payload = {
        "stage": "stage2n",
        "created_utc": utc_now(),
        "python": PYTHON,
        "protocol": str(PROTOCOL_PATH),
        "mec_checkpoint_sha256": MEC_CHECKPOINT_SHA256,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "jobs": [
            {
                key: value for key, value in job.items()
                if key not in ("process", "command", "log_handle")
            } | {"command": job["command"]}
            for job in jobs
        ],
    }
    atomic_write_json(LOG_DIR / "launch.json", payload)
    return payload


def _progress(job):
    payload = read_json(Path(job["run_dir"]) / "progress.json")
    return payload or {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms", nargs="+", choices=("dqn", "ppo"),
        default=["dqn", "ppo"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(AGENT_SEEDS)
    )
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    args = parser.parse_args()

    if not PROTOCOL_PATH.is_file():
        raise SystemExit(
            f"{PROTOCOL_PATH} missing; run stage2n_build.py first"
        )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    gpu_before = _gpu_memory()
    if (
        gpu_before["free_mb"] is not None
        and gpu_before["free_mb"] < args.min_free_gb * 1024
    ):
        raise SystemExit(
            f"GPU free memory {gpu_before['free_mb']} MB < "
            f"{args.min_free_gb * 1024:.0f} MB; refusing to launch"
        )
    jobs = _build_jobs(args)
    if not jobs:
        raise SystemExit("no jobs")
    _write_status(jobs, gpu_before, _gpu_memory())
    print(
        f"[launch] {len(jobs)} jobs, max_parallel={args.max_parallel}, "
        f"GPU free={gpu_before['free_mb']} MB",
        flush=True,
    )

    pending = list(jobs)
    running = []
    while pending or running:
        while pending and len(running) < int(args.max_parallel):
            job = pending.pop(0)
            log_path = Path(job["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            handle.write(
                f"\n===== launch {utc_now()} =====\n"
                + " ".join(job["command"]) + "\n"
            )
            handle.flush()
            process = subprocess.Popen(
                job["command"],
                cwd=str(PROJECT_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                },
            )
            job["process"] = process
            job["log_handle"] = handle
            job["started_utc"] = utc_now()
            job["pid"] = process.pid
            running.append(job)
            print(
                f"[launch] started {job['algorithm']}_s{job['seed']} "
                f"pid={process.pid} log={job['log']}",
                flush=True,
            )
            _write_status(jobs, gpu_before, _gpu_memory())

        time.sleep(15)
        for job in list(running):
            process = job["process"]
            code = process.poll()
            if code is None:
                continue
            job["returncode"] = int(code)
            job["finished_utc"] = utc_now()
            try:
                job["log_handle"].close()
            except Exception:
                pass
            running.remove(job)
            print(
                f"[launch] finished {job['algorithm']}_s{job['seed']} "
                f"returncode={code}",
                flush=True,
            )
            _write_status(jobs, gpu_before, _gpu_memory())

        if running:
            parts = []
            for job in running:
                progress = _progress(job)
                parts.append(
                    f"{job['algorithm']}_s{job['seed']}:"
                    f"Ep{progress.get('current_episode', '?')}"
                    f"/best={progress.get('best_dev_e2e_ms', '?')}"
                )
            print("[launch] " + " | ".join(parts), flush=True)

    failed = [job for job in jobs if job["returncode"] != 0]
    _write_status(jobs, gpu_before, _gpu_memory())
    print(
        f"[launch] all done: {len(jobs) - len(failed)}/{len(jobs)} "
        f"succeeded",
        flush=True,
    )
    if failed:
        for job in failed:
            print(
                f"[launch] FAILED {job['algorithm']}_s{job['seed']} "
                f"code={job['returncode']} log={job['log']}",
                flush=True,
            )
        raise SystemExit(1)
    print("STAGE2N_LAUNCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
