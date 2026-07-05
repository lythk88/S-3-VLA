#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/namn1/vlsa-aegis}"
PORT="${PORT:-8001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/pi05_no_safety_with_hidden_full}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/namn1/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/pi05_no_safety_with_hidden_full_server.log}"
EVAL_LOG="${EVAL_LOG:-${LOG_DIR}/pi05_no_safety_with_hidden_full_eval.log}"
START_LOG="${START_LOG:-${LOG_DIR}/pi05_no_safety_with_hidden_full_start.log}"
SERVER_PID_FILE="${SERVER_PID_FILE:-${LOG_DIR}/pi05_no_safety_with_hidden_full_server.pid}"
EVAL_STATUS_FILE="${EVAL_STATUS_FILE:-${LOG_DIR}/pi05_no_safety_with_hidden_full_eval.status}"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[$(timestamp)] stopping policy server pid=${SERVER_PID}" | tee -a "${START_LOG}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(timestamp)] start pi0.5 no-safety SafeLIBERO hidden-state run" | tee -a "${START_LOG}"
echo "[$(timestamp)] root=${ROOT} port=${PORT} output=${OUTPUT_ROOT}" | tee -a "${START_LOG}"

(
  cd "${ROOT}/openpi"
  exec uv run python "${ROOT}/scripts/serve_policy.py" \
    --port "${PORT}" \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "${CHECKPOINT_DIR}"
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${SERVER_PID_FILE}"
echo "[$(timestamp)] policy server pid=${SERVER_PID}" | tee -a "${START_LOG}"

ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[$(timestamp)] policy server exited before port ${PORT} was ready" | tee -a "${START_LOG}"
    tail -80 "${SERVER_LOG}" || true
    exit 1
  fi

  if python - "${PORT}" <<'PY' >/dev/null 2>&1
import socket
import sys

port = int(sys.argv[1])
with socket.create_connection(("127.0.0.1", port), timeout=2):
    pass
PY
  then
    ready=1
    break
  fi
  sleep 5
done

if [[ "${ready}" != 1 ]]; then
  echo "[$(timestamp)] timed out waiting for policy server on port ${PORT}" | tee -a "${START_LOG}"
  tail -80 "${SERVER_LOG}" || true
  exit 1
fi

echo "[$(timestamp)] policy server ready on port ${PORT}; starting evaluator" | tee -a "${START_LOG}"
set +e
PYTHONPATH="${ROOT}/safelibero${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ROOT}/main/.venv/bin/python" "${ROOT}/scripts/resume_pi05_no_safety_with_hidden_safelibero.py" \
    --root "${ROOT}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --output-root "${OUTPUT_ROOT}" \
    >"${EVAL_LOG}" 2>&1
status=$?
set -e

echo "${status}" > "${EVAL_STATUS_FILE}"
echo "[$(timestamp)] evaluator exited status=${status}" | tee -a "${START_LOG}"
exit "${status}"
