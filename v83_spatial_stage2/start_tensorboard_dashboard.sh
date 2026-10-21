#!/usr/bin/env bash
# Start (or report) the persistent Stage-2k/Stage-2l TensorBoard dashboard.
# It never stops a training process and never writes to original event files.
set -euo pipefail

ROOT="/home/ubuntu5/Workspace/MEC-V8.2"
DASH="runs/tensorboard/rl_stage2k_stage2l_dashboard"
SESSION="tb_stage2l_dashboard"
TB="/home/ubuntu5/miniconda3/envs/DRL/bin/tensorboard"
LOG="${DASH}/tensorboard.log"
STATUS="${DASH}/tensorboard_service.json"
mkdir -p "${ROOT}/${DASH}"

cd "${ROOT}"

port_in_use() {
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"
}

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  pane_pid="$(tmux list-panes -t "${SESSION}" -F '#{pane_pid}' | head -1)"
  port="$(ps -p "${pane_pid}" -o args= 2>/dev/null | grep -oE -- '--port [0-9]+' | awk '{print $2}' | head -1 || true)"
  port="${port:-6006}"
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/" || true)"
  if [ "${code}" = "200" ]; then
    echo "TensorBoard already running: session=${SESSION} pid=${pane_pid} port=${port}"
    python - "${STATUS}" "${SESSION}" "${pane_pid}" "${port}" <<'PY'
import json, sys, datetime
path, session, pid, port = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
payload = {
    "session": session,
    "pid": pid,
    "port": port,
    "url": f"http://127.0.0.1:{port}/",
    "http_status": 200,
    "status": "running",
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, ensure_ascii=False))
PY
    exit 0
  fi
  echo "Existing tmux session is not serving HTTP; restarting dashboard session."
  tmux kill-session -t "${SESSION}" || true
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
  "exec ${TB} --logdir ${DASH}/events --host 127.0.0.1 --port ${PORT} --reload_interval 30 --samples_per_plugin scalars=0 >> ${LOG} 2>&1"

code=""
for _ in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/" || true)"
  if [ "${code}" = "200" ]; then
    break
  fi
  sleep 1
done
pane_pid="$(tmux list-panes -t "${SESSION}" -F '#{pane_pid}' | head -1)"
python - "${STATUS}" "${SESSION}" "${pane_pid}" "${PORT}" "${code:-000}" <<'PY'
import json, sys, datetime
path, session, pid, port, code = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
payload = {
    "session": session,
    "pid": pid,
    "port": port,
    "url": f"http://127.0.0.1:{port}/",
    "http_status": int(code) if code.isdigit() else code,
    "status": "running" if code == "200" else "starting_or_failed",
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, ensure_ascii=False))
PY
echo "TensorBoard: session=${SESSION} pid=${pane_pid} port=${PORT} http=${code:-000}"
echo "URL: http://127.0.0.1:${PORT}/"
echo "Log: ${ROOT}/${LOG}"
