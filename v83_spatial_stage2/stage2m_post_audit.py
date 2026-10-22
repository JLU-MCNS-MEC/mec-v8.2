"""Wait for the Stage-2m training runs, then audit post-fix Q values.

The audit is split into:
  * DQN / Double-DQN / legacy: best + last checkpoints, seeds 0/1/2;
  * visual DQN: best checkpoints, seeds 0/1/2.

Outputs:
  runs/.../stage2m_q_audit/post_dqn/
  runs/.../stage2m_q_audit/post_visual/
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs/v83_spatial_stage2/stage2m_visual_fixed_city"
AUDIT_ROOT = PROJECT_ROOT / "runs/v83_spatial_stage2/stage2m_q_audit"
NAMES = (
    [f"dqn_pure_s{s}" for s in (0, 1, 2)]
    + [f"dqn_visual_s{s}" for s in (0, 1, 2)]
    + [f"ppo_pure_s{s}" for s in (0, 1, 2)]
    + [f"ppo_visual_s{s}" for s in (0, 1, 2)]
    + [f"dqn_double_s{s}" for s in (0, 1, 2)]
    + ["dqn_legacy_s0"]
)
MAX_WAIT_SECONDS = 12 * 3600
POLL_SECONDS = 60


def completed():
    return [
        name for name in NAMES
        if (RUN_ROOT / name / "training_result.json").is_file()
    ]


def wait():
    waited = 0
    while True:
        done = completed()
        print(f"[wait] {len(done)}/{len(NAMES)} training runs completed",
              flush=True)
        if len(done) == len(NAMES) or waited >= MAX_WAIT_SECONDS:
            return done
        time.sleep(POLL_SECONDS)
        waited += POLL_SECONDS


def checkpoint(name, tag, checkpoint):
    path = RUN_ROOT / name / "checkpoints" / checkpoint
    if not path.is_file():
        return None
    return f"{tag}={path}"


def run_audit(tag, checkpoints, output):
    if not checkpoints:
        print(f"[skip] {tag}: no checkpoints", flush=True)
        return None
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u",
        str(PROJECT_ROOT / "v83_spatial_stage2/stage2l_q_audit.py"),
        "--config", "A",
        "--output-dir", str(output),
        "--limit-episodes", "60",
        "--device", "cuda",
        "--threads", "2",
    ]
    for item in checkpoints:
        command.extend(["--checkpoints", item])
    print(f"[audit] {tag}: {len(checkpoints)} checkpoints -> {output}",
          flush=True)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return output


def main():
    done = wait()
    print(f"[start] auditing {len(done)} completed runs", flush=True)

    dqn_checkpoints = []
    for seed in (0, 1, 2):
        for checkpoint_name, suffix in (("best.pt", "best"),
                                        ("last.pt", "last")):
            item = checkpoint(
                f"dqn_pure_s{seed}", f"dqn_s{seed}_{suffix}", checkpoint_name
            )
            if item:
                dqn_checkpoints.append(item)
            item = checkpoint(
                f"dqn_double_s{seed}",
                f"double_s{seed}_{suffix}", checkpoint_name,
            )
            if item:
                dqn_checkpoints.append(item)
    legacy = checkpoint(
        "dqn_legacy_s0", "legacy_s0_best", "best.pt"
    )
    if legacy:
        dqn_checkpoints.append(legacy)
    run_audit("post_dqn", dqn_checkpoints, AUDIT_ROOT / "post_dqn")

    visual_checkpoints = []
    for seed in (0, 1, 2):
        item = checkpoint(
            f"dqn_visual_s{seed}", f"visual_s{seed}_best", "best.pt"
        )
        if item:
            visual_checkpoints.append(item)
    run_audit(
        "post_visual", visual_checkpoints, AUDIT_ROOT / "post_visual"
    )

    print("STAGE2M_POST_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
