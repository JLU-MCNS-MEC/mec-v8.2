"""Launch and monitor the Stage-2m experiments on the local GPU.

Groups
------
wave1: DQN/PPO x {pure, visual} x seeds 0,1
wave2: seed 2 for the four arms + Double DQN seeds 0,1,2 + legacy DQN seed 0

Every run gets its own process, log, checkpoint directory and TensorBoard
directory.  The launcher refuses to overwrite an existing ``last_full.pt`` and
waits for free GPU memory before each launch.

Usage:
    python v83_spatial_stage2/stage2m_launch.py --group wave1
    python v83_spatial_stage2/stage2m_launch.py --group wave2
    python v83_spatial_stage2/stage2m_launch.py --group status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    PROJECT_ROOT
    / "runs/v83_spatial_stage2/stage2m_visual_fixed_city"
)
LOG_ROOT = RUN_ROOT / "logs"
PYTHON = sys.executable
MAX_EPISODES = 2000
VALIDATION_EVERY = 200
THREADS = 2
MIN_FREE_MIB = 4000
MAX_WAIT_SECONDS = 6 * 3600
LAUNCH_SPACING_SECONDS = 15


def arm(name, algorithm, state_mode, seed, extra=None):
    return {
        "name": name,
        "algorithm": algorithm,
        "state_mode": state_mode,
        "seed": int(seed),
        "extra": list(extra or []),
    }


WAVE1 = [
    arm("dqn_pure_s0", "dqn", "pure_mec_v2", 0),
    arm("dqn_pure_s1", "dqn", "pure_mec_v2", 1),
    arm("dqn_visual_s0", "dqn", "visual_ablation_v2", 0),
    arm("dqn_visual_s1", "dqn", "visual_ablation_v2", 1),
    arm("ppo_pure_s0", "ppo", "pure_mec_v2", 0),
    arm("ppo_pure_s1", "ppo", "pure_mec_v2", 1),
    arm("ppo_visual_s0", "ppo", "visual_ablation_v2", 0),
    arm("ppo_visual_s1", "ppo", "visual_ablation_v2", 1),
]

WAVE2 = [
    # Relaunch the run killed when the old launcher was stopped.
    arm("dqn_pure_s2", "dqn", "pure_mec_v2", 2),
    # Experiment 1: Double DQN and the legacy-target control.
    arm("dqn_double_s0", "dqn", "pure_mec_v2", 0, ["--double-dqn"]),
    arm("dqn_double_s1", "dqn", "pure_mec_v2", 1, ["--double-dqn"]),
    arm("dqn_double_s2", "dqn", "pure_mec_v2", 2, ["--double-dqn"]),
    arm("dqn_legacy_s0", "dqn", "pure_mec_v2", 0, ["--legacy-target"]),
    # Experiment 2 third seed.
    arm("dqn_visual_s2", "dqn", "visual_ablation_v2", 2),
    arm("ppo_pure_s2", "ppo", "pure_mec_v2", 2),
    arm("ppo_visual_s2", "ppo", "visual_ablation_v2", 2),
]

SUPP = [
    # Supplementary mechanism check: does Double DQN stabilise the visual arm?
    arm("dqn_visual_double_s0", "dqn", "visual_ablation_v2", 0,
        ["--double-dqn"]),
]

GROUPS = {"wave1": WAVE1, "wave2": WAVE2, "supp": SUPP}


def gpu_free_mib():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def gpu_snapshot():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return output


def run_dir_for(item):
    return RUN_ROOT / item["name"]


def build_command(item):
    run_dir = run_dir_for(item)
    command = [
        PYTHON, "-u",
        str(PROJECT_ROOT / "v83_spatial_stage2" /
            f"stage2l_{item['algorithm']}_train.py"),
        "--config", "A",
        "--device", "cuda",
        "--max-episodes", str(MAX_EPISODES),
        "--validation-every", str(VALIDATION_EVERY),
        "--threads", str(THREADS),
        "--seed", str(item["seed"]),
        "--state-mode", item["state_mode"],
        "--run-dir", str(run_dir),
        "--tensorboard-dir", str(run_dir / "tensorboard"),
    ]
    command.extend(item["extra"])
    return command


def launch_one(item):
    run_dir = run_dir_for(item)
    last_full = run_dir / "checkpoints" / "last_full.pt"
    if last_full.is_file():
        print(f"[skip] {item['name']}: last_full.pt already exists",
              flush=True)
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{item['name']}.log"
    command = build_command(item)
    environment = dict(os.environ)
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "STAGE2L_TORCH_THREADS": str(THREADS),
        "OMP_NUM_THREADS": str(THREADS),
        "MKL_NUM_THREADS": str(THREADS),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    log = log_path.open("ab")
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=environment,
    )
    log.close()
    print(
        f"[launch] {item['name']} pid={process.pid} "
        f"free_before={gpu_free_mib()} MiB",
        flush=True,
    )
    return {
        "name": item["name"],
        "pid": int(process.pid),
        "command": command,
        "log": str(log_path),
        "run_dir": str(run_dir),
        "seed": item["seed"],
        "algorithm": item["algorithm"],
        "state_mode": item["state_mode"],
        "extra": item["extra"],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def update_manifest(records):
    path = RUN_ROOT / "launch_manifest.json"
    payload = {
        "created_or_updated_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "python": PYTHON,
        "max_episodes": MAX_EPISODES,
        "validation_every": VALIDATION_EVERY,
        "threads": THREADS,
        "gpu": gpu_snapshot(),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def launch_group(group):
    items = GROUPS[group]
    records = []
    manifest_path = RUN_ROOT / "launch_manifest.json"
    if manifest_path.is_file():
        records = json.loads(manifest_path.read_text()).get("records", [])
    known = {item["name"] for item in records}
    for index, item in enumerate(items):
        if item["name"] in known:
            print(f"[skip] {item['name']}: already in launch manifest",
                  flush=True)
            continue
        waited = 0
        while gpu_free_mib() < MIN_FREE_MIB:
            if waited >= MAX_WAIT_SECONDS:
                print(
                    f"[wait-timeout] {item['name']}: only "
                    f"{gpu_free_mib()} MiB free; not launched",
                    flush=True,
                )
                break
            print(
                f"[wait] {item['name']}: {gpu_free_mib()} MiB free, "
                f"sleeping 15 s",
                flush=True,
            )
            time.sleep(15)
            waited += 15
        else:
            record = launch_one(item)
            if record is not None:
                records.append(record)
                known.add(item["name"])
                update_manifest(records)
                time.sleep(LAUNCH_SPACING_SECONDS)
            continue
        time.sleep(LAUNCH_SPACING_SECONDS)
    update_manifest(records)
    print(f"[done] group {group}: {len(records)} total launched records",
          flush=True)


def status():
    print(f"GPU: {gpu_snapshot()}", flush=True)
    for item in WAVE1 + WAVE2 + SUPP:
        run_dir = run_dir_for(item)
        progress_path = run_dir / "progress.json"
        history_path = run_dir / "training_history.json"
        result_path = run_dir / "training_result.json"
        progress = {}
        if progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text())
            except Exception:
                progress = {}
        validations = []
        if history_path.is_file():
            try:
                validations = json.loads(
                    history_path.read_text()
                ).get("validation", [])
            except Exception:
                validations = []
        best = max(
            (row["joint_reward"] for row in validations), default=None
        )
        result = None
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
            except Exception:
                result = {}
        print(
            f"{item['name']:16s} ep={progress.get('current_episode')} "
            f"status={progress.get('status')} "
            f"val={len(validations)} best={best} "
            f"final={result is not None}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", required=True, choices=("wave1", "wave2", "supp", "status")
    )
    args = parser.parse_args()
    if args.group == "status":
        status()
        return
    launch_group(args.group)


if __name__ == "__main__":
    main()
