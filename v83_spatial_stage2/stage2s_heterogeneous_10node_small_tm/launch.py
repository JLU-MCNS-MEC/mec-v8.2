"""Launch the six formal Stage-2s runs with crash resume.

The launcher never stops other experiments.  It only starts Stage-2s
``train.py`` processes, checks free GPU memory before each start, and resumes
an interrupted run from ``last_full.pt``.

Examples:
    python .../launch.py --wait --max-parallel 3
    python .../launch.py --wait --max-parallel 2 --min-free-mb 6000
"""

from __future__ import annotations

import argparse
import json
import os
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
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (  # noqa: E402
    ALGORITHMS,
    FORMAL_SEEDS,
    RUNS_ROOT,
    STAGE,
    run_dir,
)

LOG_ROOT = RUNS_ROOT / "logs"
STATE_PATH = RUNS_ROOT / "launch_state.json"
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def _gpu_free_mb():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        return int(output.strip().splitlines()[0])
    except Exception:
        return None


def _running_stage2s():
    try:
        output = subprocess.check_output(
            ["pgrep", "-af", "stage2s_heterogeneous_10node_small_tm/train.py"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    except Exception:
        return []
    return [
        line for line in output.splitlines()
        if "launch.py" not in line and line.strip()
    ]


def _run_key(line):
    parts = line.split()
    algorithm = seed = None
    for index, token in enumerate(parts):
        if token == "--algorithm" and index + 1 < len(parts):
            algorithm = parts[index + 1]
        if token == "--seed" and index + 1 < len(parts):
            seed = parts[index + 1]
    return f"{algorithm}_s{seed}" if algorithm and seed else None


def _completed(algorithm, seed):
    result = read_json(run_dir(algorithm, seed) / "training_result.json")
    return (
        result is not None
        and result.get("status") == "completed"
        and int(result.get("episodes", 0)) >= 4000
    )


def _command(algorithm, seed, resume):
    command = [
        sys.executable, "-u", str(TRAIN_SCRIPT),
        "--algorithm", algorithm,
        "--seed", str(seed),
        "--threads", "2",
    ]
    if resume:
        command.append("--resume")
    return command


def _start(algorithm, seed, resume, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(
        f"\n===== start {utc_now()} resume={resume} =====\n"
    )
    handle.flush()
    process = subprocess.Popen(
        _command(algorithm, seed, resume),
        cwd=str(PROJECT_ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, handle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-free-mb", type=int, default=5000)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="optional subset like dqn_s101 ppo_s202",
    )
    args = parser.parse_args(argv)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    keys = [
        f"{algorithm}_s{seed}"
        for algorithm in ALGORITHMS for seed in FORMAL_SEEDS
    ]
    if args.only:
        keys = [key for key in keys if key in set(args.only)]
    state = read_json(STATE_PATH) or {
        "stage": STAGE, "created_utc": utc_now(), "jobs": {},
    }
    running = {_run_key(line) for line in _running_stage2s()}
    pending = []
    for key in keys:
        algorithm, seed_text = key.split("_s")
        seed = int(seed_text)
        if _completed(algorithm, seed):
            state["jobs"].setdefault(key, {})["status"] = "completed"
            continue
        if key in running:
            state["jobs"].setdefault(key, {})["status"] = "running"
            continue
        pending.append(key)
    print(
        f"[launch] pending={pending} already_running={sorted(running)}",
        flush=True,
    )
    retries = {key: 0 for key in pending}
    active = {}
    while pending or active:
        for key in list(pending):
            if len(active) >= int(args.max_parallel):
                break
            algorithm, seed_text = key.split("_s")
            seed = int(seed_text)
            free_mb = _gpu_free_mb()
            if free_mb is not None and free_mb < int(args.min_free_mb):
                print(
                    f"[launch] free GPU {free_mb} MiB < "
                    f"{args.min_free_mb} MiB; waiting",
                    flush=True,
                )
                break
            current_run_dir = run_dir(algorithm, seed)
            last_full = current_run_dir / "checkpoints" / "last_full.pt"
            history = current_run_dir / "training_history.json"
            if history.is_file() and not last_full.is_file():
                stale = current_run_dir.with_name(
                    f"{current_run_dir.name}_stale_"
                    f"{time.strftime('%Y%m%d_%H%M%S')}"
                )
                current_run_dir.rename(stale)
                print(
                    f"[launch] {key}: moved history without checkpoint to "
                    f"{stale}",
                    flush=True,
                )
            resume = last_full.is_file()
            log_path = LOG_ROOT / f"{key}.log"
            process, handle = _start(algorithm, seed, resume, log_path)
            active[key] = {
                "process": process, "handle": handle,
                "run_dir": str(run_dir(algorithm, seed)),
                "log": str(log_path), "resume": resume,
                "started_utc": utc_now(),
            }
            pending.remove(key)
            state["jobs"][key] = {
                "status": "running",
                "pid": int(process.pid),
                "command": _command(algorithm, seed, resume),
                "log": str(log_path),
                "resume": bool(resume),
                "started_utc": utc_now(),
            }
            atomic_write_json(STATE_PATH, state)
            print(
                f"[launch] started {key} pid={process.pid} "
                f"resume={resume}",
                flush=True,
            )
        if not active:
            if pending:
                time.sleep(int(args.poll_seconds))
                continue
            break
        time.sleep(int(args.poll_seconds))
        for key in list(active):
            item = active[key]
            code = item["process"].poll()
            if code is None:
                continue
            item["handle"].close()
            algorithm, seed_text = key.split("_s")
            seed = int(seed_text)
            if _completed(algorithm, seed) and code == 0:
                state["jobs"][key].update({
                    "status": "completed",
                    "returncode": int(code),
                    "finished_utc": utc_now(),
                })
                print(f"[launch] completed {key}", flush=True)
            else:
                retries[key] += 1
                state["jobs"][key].update({
                    "status": "retrying",
                    "returncode": int(code),
                    "retries": retries[key],
                    "updated_utc": utc_now(),
                })
                if retries[key] <= int(args.max_retries):
                    pending.append(key)
                    print(
                        f"[launch] {key} exited code={code}; "
                        f"queued retry {retries[key]}/"
                        f"{args.max_retries}",
                        flush=True,
                    )
                else:
                    state["jobs"][key]["status"] = "failed"
                    print(
                        f"[launch] {key} failed after "
                        f"{args.max_retries} retries",
                        flush=True,
                    )
            del active[key]
            atomic_write_json(STATE_PATH, state)
        if not args.wait and not active and not pending:
            break
    state["updated_utc"] = utc_now()
    state["active"] = []
    atomic_write_json(STATE_PATH, state)
    remaining = []
    for key in keys:
        algorithm, seed_text = key.split("_s")
        if not _completed(algorithm, int(seed_text)):
            remaining.append(key)
    print(
        f"[launch] done. incomplete={remaining}",
        flush=True,
    )
    return state


if __name__ == "__main__":
    main()
