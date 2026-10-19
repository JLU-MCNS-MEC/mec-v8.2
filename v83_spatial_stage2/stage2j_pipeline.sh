#!/usr/bin/env bash
# Stage-2j serial pipeline: train -> select best -> Final Test 300 -> next.
# Idempotent: completed training/evaluation steps are skipped.
# Formal test requires the frozen protocol; this script never tunes on test.

set -u -o pipefail

cd /home/ubuntu5/Workspace/MEC-V8.2 || exit 1
PY=/home/ubuntu5/miniconda3/envs/DRL/bin/python
ROOT=runs/v83_spatial_stage2/stage2j
LOGS="$ROOT/logs"
FREEZE="$ROOT/protocol_freeze.json"
mkdir -p "$LOGS"

# Everything the driver prints is also appended to pipeline.log.
exec > >(tee -a "$LOGS/pipeline.log") 2>&1

echo "=== STAGE-2J PIPELINE START $(date -Is) ==="

fail() {
  echo "PIPELINE_FAILED: $* $(date -Is)"
  exit 1
}

test_status() {
  local model="$1"
  MODEL="$model" "$PY" -c '
import json, os
from pathlib import Path
p = Path("runs/v83_spatial_stage2/stage2j/progress.json")
if not p.is_file():
    print("pending"); raise SystemExit
d = json.loads(p.read_text())
print(d.get("models", {}).get(os.environ["MODEL"], {}).get("test_status", "pending"))
'
}

run_train() {
  local model="$1"
  local dir="$ROOT/$model"
  local log="$LOGS/${model}_train.log"
  if [ -f "$dir/training_result.json" ]; then
    echo "SKIP_TRAIN $model (training_result.json exists)"
    return 0
  fi
  if pgrep -f "[s]tage2j_train.py --model ${model}" >/dev/null; then
    echo "WAIT_TRAIN $model (already running)"
    while pgrep -f "[s]tage2j_train.py --model ${model}" >/dev/null; do
      sleep 10
    done
    grep -q "TRAINING_COMPLETE" "$log" || fail "$model training did not complete"
    return 0
  fi
  local extra=""
  if [ -f "$dir/last.pt" ]; then extra="--resume"; fi
  echo "TRAIN $model $(date -Is)"
  "$PY" v83_spatial_stage2/stage2j_train.py --model "$model" \
    --output-dir "$dir" $extra 2>&1 | tee "$log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "$model training"
  [ -f "$dir/training_result.json" ] || fail "$model training_result.json missing"
  echo "TRAIN_DONE $model $(date -Is)"
}

run_reuse_transformer_m() {
  local dir="$ROOT/Transformer-M"
  if [ -f "$dir/training_result.json" ]; then
    echo "SKIP_REUSE Transformer-M"
    return 0
  fi
  echo "REUSE Transformer-M $(date -Is)"
  "$PY" v83_spatial_stage2/stage2j_train.py --model Transformer-M \
    --reuse-stage2i-t5 --output-dir "$dir" \
    2>&1 | tee "$LOGS/Transformer-M_reuse.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "Transformer-M reuse"
  echo "REUSE_DONE Transformer-M $(date -Is)"
}

run_dev_equivalence() {
  local dir="$ROOT/Transformer-M"
  if [ ! -f "$dir/eval_dev_val_180/metrics.json" ]; then
    echo "DEV_EVAL Transformer-M $(date -Is)"
    "$PY" v83_spatial_stage2/stage2j_eval.py --model Transformer-M \
      --split dev_val --output-dir "$dir/eval_dev_val_180" \
      2>&1 | tee "$LOGS/Transformer-M_dev_val.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || fail "Transformer-M dev evaluation"
  fi
  echo "EQUIVALENCE Transformer-M $(date -Is)"
  "$PY" v83_spatial_stage2/stage2j_equivalence.py \
    2>&1 | tee "$LOGS/Transformer-M_equivalence.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "Transformer-M equivalence check"
  echo "EQUIVALENCE_DONE Transformer-M $(date -Is)"
}

run_test() {
  local model="$1"
  local status
  status="$(test_status "$model")"
  if [ "$status" = "tested" ] || [ "$status" = "reused_tested" ]; then
    echo "SKIP_TEST $model (already tested)"
    return 0
  fi
  local out="$ROOT/$model/eval_final_test_300"
  local extra=""
  if [ -d "$out" ]; then extra="--resume"; fi
  echo "TEST $model $(date -Is)"
  "$PY" v83_spatial_stage2/stage2j_eval.py --model "$model" \
    --split final_test --allow-final-test --freeze-json "$FREEZE" \
    $extra 2>&1 | tee "$LOGS/${model}_final_test.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "$model final test"
  status="$(test_status "$model")"
  [ "$status" = "tested" ] || fail "$model did not record tested status"
  echo "MODEL_DONE $model $(date -Is)"
}

# Small is already training in another tmux window; this waits for it.
run_train Small
run_test Small

run_train Transformer-S
run_test Transformer-S

run_train Medium
run_test Medium

run_train Large
run_test Large

run_reuse_transformer_m
run_dev_equivalence
run_test Transformer-M

echo "ANALYZE $(date -Is)"
"$PY" v83_spatial_stage2/stage2j_analyze.py 2>&1 | tee "$LOGS/analyze.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "analysis/figures"

echo "PIPELINE_COMPLETE $(date -Is)"
