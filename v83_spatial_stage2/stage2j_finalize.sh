#!/usr/bin/env bash
# Wait for the Stage-2j five-model pipeline, generate the final report,
# run the tests, commit the code/results/figures and push to GitHub.

set -u -o pipefail

cd /home/ubuntu5/Workspace/MEC-V8.2 || exit 1
PY=/home/ubuntu5/miniconda3/envs/DRL/bin/python
ROOT=runs/v83_spatial_stage2/stage2j
LOGS="$ROOT/logs"
PIPELINE_LOG="$LOGS/pipeline.log"
FINALIZE_LOG="$LOGS/finalize.log"
BRANCH=experiment/v83-spatial-foundation
mkdir -p "$LOGS"
exec > >(tee -a "$FINALIZE_LOG") 2>&1

echo "=== STAGE-2J FINALIZE START $(date -Is) ==="

fail() {
  echo "FINALIZE_FAILED: $* $(date -Is)"
  exit 1
}

echo "WAIT_PIPELINE"
while true; do
  if grep -q "PIPELINE_COMPLETE" "$PIPELINE_LOG" 2>/dev/null; then
    break
  fi
  if grep -q "PIPELINE_FAILED" "$PIPELINE_LOG" 2>/dev/null; then
    fail "pipeline failed; see $PIPELINE_LOG"
  fi
  sleep 30
done
echo "PIPELINE_COMPLETE seen"

# All five formal test states must be recorded.
MODEL_COUNT=$("$PY" - <<'PY'
import json
from pathlib import Path
from v83_spatial_stage2.stage2j_common import TABLE_ORDER
p = Path("runs/v83_spatial_stage2/stage2j/progress.json")
d = json.loads(p.read_text())
models = d.get("models", {})
ok = [
    name for name in TABLE_ORDER
    if models.get(name, {}).get("test_status") in ("tested", "reused_tested")
    and models.get(name, {}).get("metrics")
]
print(len(ok))
PY
) || fail "could not read progress.json"
[ "$MODEL_COUNT" = "5" ] || fail "only $MODEL_COUNT/5 models tested"
echo "ALL_FIVE_TESTED"

# Re-run analysis to guarantee summary.json and figures are current.
echo "ANALYZE_RERUN"
"$PY" v83_spatial_stage2/stage2j_analyze.py \
  2>&1 | tee "$LOGS/analyze_final.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "analysis"

# Generate the final report from measured results.
echo "REPORT"
"$PY" v83_spatial_stage2/stage2j_report.py \
  2>&1 | tee "$LOGS/report.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "report generation"

# Unit tests.
echo "TESTS"
TEST_OUT="$LOGS/tests_final.log"
"$PY" -m unittest -v v83_spatial_stage2.test_stage2j \
  > "$TEST_OUT" 2>&1
TEST_STATUS=$?
tail -5 "$TEST_OUT"
[ "$TEST_STATUS" -eq 0 ] || fail "unit tests"
TEST_LINE=$(grep -E "^Ran [0-9]+ tests" "$TEST_OUT" | tail -1)
echo "TESTS_OK: $TEST_LINE"

echo "GIT_STATUS_BEFORE"
git status --porcelain

git add \
  v83_spatial_stage2/stage2j_model.py \
  v83_spatial_stage2/stage2j_common.py \
  v83_spatial_stage2/stage2j_train.py \
  v83_spatial_stage2/stage2j_eval.py \
  v83_spatial_stage2/stage2j_freeze.py \
  v83_spatial_stage2/stage2j_equivalence.py \
  v83_spatial_stage2/stage2j_analyze.py \
  v83_spatial_stage2/stage2j_report.py \
  v83_spatial_stage2/stage2j_pipeline.sh \
  v83_spatial_stage2/stage2j_finalize.sh \
  v83_spatial_stage2/test_stage2j.py \
  v83_spatial_stage2/STAGE2J_PROTOCOL.md \
  v83_spatial_stage2/stage2j_protocol_freeze.json \
  v83_spatial_stage2/STAGE2J_RESULTS.md \
  v83_spatial_stage2/STAGE2J_BAND_SR_TABLE.md \
  v83_spatial_stage2/V83_STAGE2J_FIVE_MEC_T5_TEST_REPORT.md \
  v83_spatial_stage2/figures/stage2j
[ -f v83_spatial_stage2/STAGE2J_COMMIT_STATUS.md ] && \
  git add v83_spatial_stage2/STAGE2J_COMMIT_STATUS.md

echo "STAGED"
git diff --cached --stat

if git diff --cached --quiet && git diff --quiet; then
  COMMIT=$(git rev-parse HEAD)
  echo "NO_CHANGES_COMMIT $COMMIT"
else
  git commit -m "$(cat <<'EOF'
Stage-2j: complete five-MEC Final Test 300 comparison

Train and evaluate Small, Transformer-S, Medium, Large and the reused
Stage-2i T5 Transformer-M on the sealed Final Test 300 under the frozen
protocol. Add bootstrap CIs, band and retrieval tables, four paper
figures, analysis and the final report.

Co-Authored-By: Claude Code <noreply@anthropic.com>
EOF
)" || fail "git commit"
  COMMIT=$(git rev-parse HEAD)
  echo "COMMIT $COMMIT"
fi

push_ok=0
if git push origin "$BRANCH"; then
  push_ok=1
else
  echo "PUSH_RETRY_WITHOUT_PROXY"
  if env -u http_proxy -u https_proxy -u all_proxy \
         -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
         git push origin "$BRANCH"; then
    push_ok=1
  fi
fi
[ "$push_ok" -eq 1 ] || fail "git push"

STATUS_PATH=v83_spatial_stage2/STAGE2J_COMMIT_STATUS.md
cat > "$STATUS_PATH" <<EOF
# Stage-2j Commit Status

- Branch: \`$BRANCH\`
- Primary commit: \`$COMMIT\`
- Push: successful to \`origin/$BRANCH\`
- Unit tests: $TEST_LINE
- Final Test: 5/5 models, 300 episodes each (100/100/100)
- Experiment status: **COMPLETE**
- Report: \`v83_spatial_stage2/V83_STAGE2J_FIVE_MEC_T5_TEST_REPORT.md\`
- Figures: \`v83_spatial_stage2/figures/stage2j/\`
- Per-model artifacts: \`runs/v83_spatial_stage2/stage2j/<MEC>/\`
- This status file is a follow-up metadata commit.
EOF
git add "$STATUS_PATH"
if git diff --cached --quiet && git diff --quiet; then
  STATUS_COMMIT=$(git rev-parse HEAD)
  echo "NO_CHANGES_STATUS_COMMIT $STATUS_COMMIT"
else
  git commit -m "$(cat <<'EOF'
Stage-2j: record final commit, push and test status

Co-Authored-By: Claude Code <noreply@anthropic.com>
EOF
)" || fail "status commit"
  STATUS_COMMIT=$(git rev-parse HEAD)
fi
if git push origin "$BRANCH"; then
  :
else
  env -u http_proxy -u https_proxy -u all_proxy \
      -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      git push origin "$BRANCH" || fail "status push"
fi
echo "STATUS_COMMIT $STATUS_COMMIT"
echo "WORKTREE"
git status --porcelain
echo "FINALIZE_COMPLETE $(date -Is)"
