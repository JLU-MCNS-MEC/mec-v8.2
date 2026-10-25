#!/usr/bin/env bash
# Wait for the 12 formal runs, then evaluate, aggregate, plot trajectories,
# render the Chinese report.  No Test task is read.
set -euo pipefail

cd /home/ubuntu5/Workspace/MEC-V8.2
PY=/home/ubuntu5/miniconda3/envs/DRL/bin/python
STAGE=v83_spatial_stage2/stage2r_10node_small_vs_tm
ROOT=runs/v83_spatial_stage2/stage2r_10node_small_vs_tm
export OMP_NUM_THREADS=2
export MPLBACKEND=Agg

for model in small transformer_m; do
  for algorithm in dqn ppo; do
    for seed in 11 22 33; do
      path="$ROOT/$model/${algorithm}_s${seed}/training_result.json"
      echo "[postprocess] waiting for $path"
      until [ -f "$path" ]; do sleep 30; done
    done
  done
done

echo "[postprocess] all 12 training results present"
"$PY" -u "$STAGE/eval.py" --device cuda
"$PY" -u "$STAGE/analyze.py"
"$PY" -u "$STAGE/trajectories.py" --device cuda
"$PY" -u "$STAGE/plot_trajectories.py"
"$PY" -u "$STAGE/render_report.py"
echo "STAGE2R_POSTPROCESS_COMPLETE"
