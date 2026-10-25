#!/usr/bin/env bash
# Wait for all six formal Stage-2s runs, then evaluate, aggregate and plot.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/home/ubuntu5/miniconda3/envs/DRL/bin/python}"
STAGE_DIR="$ROOT/v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm"
RUNS="$ROOT/runs/v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm"
LOG="$RUNS/logs/postprocess.log"
mkdir -p "$RUNS/logs"

exec > >(tee -a "$LOG") 2>&1

wait_for_runs() {
  "$PY" - "$RUNS" <<'PY'
import json, sys, time
from pathlib import Path
runs = Path(sys.argv[1])
keys = [f"{a}_s{s}" for a in ("dqn", "ppo") for s in (101, 202, 303)]
while True:
    done = []
    pending = []
    for key in keys:
        result = runs / key / "training_result.json"
        if result.is_file():
            try:
                payload = json.loads(result.read_text())
            except Exception:
                payload = {}
            if payload.get("status") == "completed" and int(
                payload.get("episodes", 0)
            ) >= 4000:
                done.append(key)
                continue
        pending.append(key)
    print(f"[postprocess] completed={done} pending={pending}", flush=True)
    if not pending:
        break
    time.sleep(300)
PY
}

wait_for_runs
cd "$ROOT"
"$PY" -u "$STAGE_DIR/eval.py"
"$PY" -u "$STAGE_DIR/analyze.py"
"$PY" -u "$STAGE_DIR/plot.py"
"$PY" -u "$STAGE_DIR/trajectories.py"
"$PY" -u "$STAGE_DIR/plot_trajectories.py"
"$PY" -u "$STAGE_DIR/analyze.py"
"$PY" -u "$STAGE_DIR/render_report.py"
echo "STAGE2S_POSTPROCESS_COMPLETE"
