#!/usr/bin/env bash
# Start (or verify) the persistent Stage-2k/Stage-2l TensorBoard dashboard.
# It never stops a training process and never writes to original event files.
#
# Scalars are served with an explicit 5000-point limit: the longest dashboard
# series is 3500 points, so every real point is exposed.  Do not use
# "--samples_per_plugin scalars=0": TensorBoard interprets that as zero
# samples, which produces a populated tag list with empty charts.
set -euo pipefail

ROOT="/home/ubuntu5/Workspace/MEC-V8.2"
DASH="runs/tensorboard/rl_stage2k_stage2l_dashboard"
SESSION="tb_stage2l_dashboard"
SCALAR_SAMPLES="5000"
TB="/home/ubuntu5/miniconda3/envs/DRL/bin/tensorboard"
PY="/home/ubuntu5/miniconda3/envs/DRL/bin/python"
LOG="${DASH}/tensorboard.log"
STATUS="${DASH}/tensorboard_service.json"
VERIFY="${ROOT}/v83_spatial_stage2/verify_tensorboard_dashboard.py"
VERIFY_OUT="${DASH}/SERVICE_VERIFICATION.json"
CONFIG="${ROOT}/v83_spatial_stage2/tensorboard_dashboard_config.json"
mkdir -p "${ROOT}/${DASH}"

cd "${ROOT}"

port_in_use() {
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"
}

write_status() {
  local pid="$1"
  local port="$2"
  local http_code="$3"
  local state="$4"
  local verified="$5"
  "${PY}" - "${STATUS}" "${SESSION}" "${pid}" "${port}" \
    "${http_code}" "${state}" "${verified}" "${VERIFY_OUT}" <<'PY'
import datetime
import json
import sys

path, session, pid, port, http_code, state, verified, verify_out = sys.argv[1:]
try:
    pid_value = int(pid)
except ValueError:
    pid_value = None
try:
    port_value = int(port)
except ValueError:
    port_value = None
try:
    http_value = int(http_code)
except ValueError:
    http_value = http_code
payload = {
    "session": session,
    "pid": pid_value,
    "port": port_value,
    "url": f"http://127.0.0.1:{port_value}/" if port_value else None,
    "http_status": http_value,
    "scalars_verified": verified == "true",
    "verification": verify_out,
    "status": state,
    "updated_utc": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
PY
}

verify_service() {
  local port="$1"
  "${PY}" "${VERIFY}" \
    --url "http://127.0.0.1:${port}" \
    --config "${CONFIG}" \
    --output "${ROOT}/${VERIFY_OUT}" \
    --timeout 20
}

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  pane_pid="$(tmux list-panes -t "${SESSION}" -F '#{pane_pid}' | head -1)"
  port="$(ps -p "${pane_pid}" -o args= 2>/dev/null \
    | grep -oE -- '--port [0-9]+' | awk '{print $2}' | head -1 || true)"
  port="${port:-6006}"
  code="$(curl -s -o /dev/null -w '%{http_code}' \
    "http://127.0.0.1:${port}/" || true)"
  if [ "${code}" = "200" ] && verify_service "${port}"; then
    echo "TensorBoard already running and scalar data verified:"
    echo "  session=${SESSION} pid=${pane_pid} port=${port}"
    write_status "${pane_pid}" "${port}" 200 running true
    exit 0
  fi
  echo "Existing tmux session is not serving verified scalar data; restarting."
  tmux kill-session -t "${SESSION}" || true
  for _ in $(seq 1 20); do
    port_in_use "${port}" || break
    sleep 0.5
  done
fi

PORT=6006
if port_in_use "${PORT}"; then
  for candidate in 6007 6008 6009 6010 6011 6012; do
    if ! port_in_use "${candidate}"; then
      PORT="${candidate}"
      break
    fi
  done
fi
if port_in_use "${PORT}"; then
  echo "No free port found in 6006-6012" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
  "exec ${TB} --logdir ${DASH}/events --host 127.0.0.1 --port ${PORT} --reload_interval 30 --samples_per_plugin scalars=${SCALAR_SAMPLES} >> ${LOG} 2>&1"

code=""
for _ in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' \
    "http://127.0.0.1:${PORT}/" || true)"
  if [ "${code}" = "200" ]; then
    break
  fi
  sleep 1
done
pane_pid="$(tmux list-panes -t "${SESSION}" -F '#{pane_pid}' | head -1)"
if [ "${code}" = "200" ] && verify_service "${PORT}"; then
  write_status "${pane_pid}" "${PORT}" "${code}" running true
  echo "TensorBoard: session=${SESSION} pid=${pane_pid} port=${PORT} http=${code}"
  echo "Scalar data: verified (all six runs, 3500 real training points each)"
  echo "URL: http://127.0.0.1:${PORT}/"
  echo "Log: ${ROOT}/${LOG}"
  exit 0
fi

write_status "${pane_pid}" "${PORT}" "${code:-000}" degraded false
echo "TensorBoard failed scalar-data verification (http=${code:-000})" >&2
echo "See ${ROOT}/${VERIFY_OUT} and ${ROOT}/${LOG}" >&2
exit 1
