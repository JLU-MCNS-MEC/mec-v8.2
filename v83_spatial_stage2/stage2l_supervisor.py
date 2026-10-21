"""Resource-aware supervisor for the six Stage-2l training runs.

Behaviour:
  * A/DQN and A/PPO start immediately (they may overlap the existing
    Stage-2k runs; the user explicitly allows CPU/GPU parallelism);
  * B/DQN and B/PPO start only after the existing Stage-2k training
    processes exit, so the old runs are never slowed down by four new
    trainers;
  * C/DQN and C/PPO are queued until at most ``--max-concurrent`` new
    training processes are active;
  * existing processes are never stopped, restarted or modified;
  * every 200-episode validation is appended to ``PROGRESS.md``.

The supervisor does not change rewards, algorithms or checkpoints.  It only
launches the documented commands and records their exit status.
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

from v83_spatial_stage2.stage2k_common import read_json
from v83_spatial_stage2.stage2l_fixed_city import STAGE2L_ROOT


EXISTING_MARKERS = (
    "stage2k_ppo_train.py",
    "stage2k_jointr_v2_train.py",
)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def existing_training_pids():
    """PIDs of the pre-existing Stage-2k trainers (never touched)."""
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,stat=,args="], text=True
        )
    except Exception:
        return []
    pids = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        pid, stat, args = parts
        if stat.startswith("Z"):
            continue
        if any(marker in args for marker in EXISTING_MARKERS):
            try:
                pids.append(int(pid))
            except ValueError:
                continue
    return pids


def job_command(config, algorithm, run_dir, tensorboard_dir):
    script = (
        "stage2l_dqn_train.py" if algorithm == "dqn"
        else "stage2l_ppo_train.py"
    )
    return [
        sys.executable, "-u", str(PROJECT_ROOT / "v83_spatial_stage2" / script),
        "--config", config,
        "--device", "cuda",
        "--max-episodes", "3500",
        "--validation-every", "200",
        "--threads", "4",
        "--run-dir", str(run_dir),
        "--tensorboard-dir", str(tensorboard_dir),
    ]


class Job:
    def __init__(self, config, algorithm):
        self.config = config
        self.algorithm = algorithm
        self.run_dir = STAGE2L_ROOT / config / algorithm
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.log_path = STAGE2L_ROOT / "logs" / f"{config}_{algorithm}.log"
        self.command = job_command(
            config, algorithm, self.run_dir, self.tensorboard_dir
        )
        self.process = None
        self.started_utc = None
        self.finished_utc = None
        self.returncode = None
        self.last_validation_episode = -1
        self.status = "queued"

    def launch(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if (self.run_dir / "checkpoints" / "last_full.pt").is_file():
            self.status = "skipped_existing_checkpoint"
            self.finished_utc = now()
            _append_log(
                f"{self.finished_utc} 跳过 {self.config}/{self.algorithm}: "
                f"{self.run_dir}/checkpoints/last_full.pt 已存在"
            )
            return
        env = dict(os.environ)
        env.update({
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "STAGE2L_TORCH_THREADS": "4",
            "PYTHONUNBUFFERED": "1",
        })
        handle = self.log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command, cwd=str(PROJECT_ROOT), env=env,
            stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.started_utc = now()
        self.status = "running"
        _append_log(
            f"{self.started_utc} 启动 {self.config}/{self.algorithm} "
            f"pid={self.process.pid} log={self.log_path}"
        )

    def poll(self):
        if self.process is None:
            return
        code = self.process.poll()
        if code is not None and self.status == "running":
            self.returncode = int(code)
            self.finished_utc = now()
            self.status = "completed" if code == 0 else "failed"
            _append_log(
                f"{self.finished_utc} 结束 {self.config}/{self.algorithm} "
                f"exit={code} status={self.status}"
            )
        progress = read_json(self.run_dir / "progress.json")
        if progress and progress.get("last_validation"):
            record = progress["last_validation"]
            episode = int(record.get("episode", -1))
            if episode > self.last_validation_episode:
                self.last_validation_episode = episode
                _append_progress(self.config, self.algorithm, record)

    def as_dict(self):
        return {
            "config": self.config,
            "algorithm": self.algorithm,
            "status": self.status,
            "pid": self.process.pid if self.process else None,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "returncode": self.returncode,
            "run_dir": str(self.run_dir),
            "log": str(self.log_path),
            "last_validation_episode": self.last_validation_episode,
        }


def _append_log(message):
    path = STAGE2L_ROOT / "logs" / "supervisor.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    print(message, flush=True)


def _append_progress(config, algorithm, record):
    path = STAGE2L_ROOT / "PROGRESS.md"
    if not path.is_file():
        path.write_text(
            "# Stage-2l 训练进度\n\n"
            "每 200 Episodes 记录一次固定 Dev 验证结果。\n\n"
            "| 时间 | 配置 | 算法 | Episode | JointR | SR | nDTW | E2E (ms) | SPL | Best |\n"
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
            encoding="utf-8",
        )
    row = (
        f"| {now()} | {config} | {algorithm.upper()} | "
        f"{record.get('episode')} | {record.get('joint_reward', 0):.4f} | "
        f"{100*record.get('SR', 0):.2f}% | {record.get('nDTW', 0):.4f} | "
        f"{record.get('E2E_ms', 0):.1f} | {record.get('SPL', 0):.4f} | "
        f"{record.get('joint_reward', 0):.4f} |\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--start-b-without-waiting", action="store_true",
        help="Testing only: do not wait for the existing Stage-2k runs.",
    )
    args = parser.parse_args()

    (STAGE2L_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    jobs = {
        "A/dqn": Job("A", "dqn"),
        "A/ppo": Job("A", "ppo"),
        "B/dqn": Job("B", "dqn"),
        "B/ppo": Job("B", "ppo"),
        "C/dqn": Job("C", "dqn"),
        "C/ppo": Job("C", "ppo"),
    }
    _append_log(
        f"{now()} supervisor 启动: max_concurrent={args.max_concurrent}"
    )
    # A starts immediately as requested.
    for key in ("A/dqn", "A/ppo"):
        jobs[key].launch()

    b_started = False
    while True:
        for job in jobs.values():
            job.poll()
        running = [
            job for job in jobs.values() if job.status == "running"
        ]
        queued = [
            job for job in jobs.values() if job.status == "queued"
        ]
        existing = [] if args.start_b_without_waiting else existing_training_pids()
        free_slots = max(0, int(args.max_concurrent) - len(running))
        if not b_started and not existing:
            for key in ("B/dqn", "B/ppo"):
                job = jobs[key]
                if job.status == "queued" and free_slots > 0:
                    job.launch()
                    free_slots -= 1
            if all(jobs[key].status != "queued"
                   for key in ("B/dqn", "B/ppo")):
                b_started = True
        if b_started:
            for key in ("C/dqn", "C/ppo"):
                job = jobs[key]
                if job.status == "queued" and free_slots > 0:
                    job.launch()
                    free_slots -= 1
        payload = {
            "updated_utc": now(),
            "existing_stage2k_pids": existing,
            "max_concurrent": int(args.max_concurrent),
            "running": len(running),
            "jobs": {key: job.as_dict() for key, job in jobs.items()},
        }
        (STAGE2L_ROOT / "supervisor_status.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        terminal = ("completed", "failed", "skipped_existing_checkpoint")
        if all(job.status in terminal for job in jobs.values()):
            _append_log(f"{now()} 全部 Stage-2l 训练结束")
            break
        time.sleep(max(5, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
