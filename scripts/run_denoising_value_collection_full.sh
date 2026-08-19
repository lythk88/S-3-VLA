#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
PORT="${PORT:-8006}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/training_dataset/pi05_hidden_chunks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/training_dataset/pi05_denoising_value_v1}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/denoising_value_collection_v1}"
TASK_TMP_DIR="${TASK_TMP_DIR:-${ROOT}/.worker_tmp/denoising_value_v1_${SLURM_JOB_ID:-manual}}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${ROOT}/training_dataset_generator/libero_training_config}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${TASK_TMP_DIR}"
export TMPDIR="${TASK_TMP_DIR}"
export TMP="${TASK_TMP_DIR}"
export TEMP="${TASK_TMP_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export LIBERO_CONFIG_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

(
    cd "${ROOT}/openpi"
    exec uv run python "${ROOT}/scripts/serve_denoising_trace_policy.py" \
        --port "${PORT}" \
        --checkpoint-dir /home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero
) >"${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 180); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        tail -120 "${LOG_DIR}/server.log"
        exit 1
    fi
    if /home/lythk/vlsa-aegis/.run_venv/bin/python - "${PORT}" <<'PY' >/dev/null 2>&1
import socket, sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=2):
    pass
PY
    then
        ready=1
        break
    fi
    sleep 5
done
if [[ "${ready}" != 1 ]]; then
    tail -120 "${LOG_DIR}/server.log"
    exit 1
fi

PYTHONPATH="${ROOT}/main:${ROOT}/openpi/packages/openpi-client/src:${ROOT}/safelibero" \
    /home/lythk/vlsa-aegis/.run_venv/bin/python \
    "${ROOT}/main/run_denoising_collection_matrix.py" \
    --host 127.0.0.1 --port "${PORT}" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${OUTPUT_ROOT}"
