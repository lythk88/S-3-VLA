#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
PORT="${PORT:-8006}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/training_dataset/pi05_denoising_value_pilot_v4}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/denoising_value_collection_pilot}"
TASK_TMP_DIR="${TASK_TMP_DIR:-${ROOT}/.worker_tmp/denoising_value_pilot_${SLURM_JOB_ID:-manual}}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${ROOT}/training_dataset_generator/libero_training_config}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-safelibero_spatial}"
SAFETY_LEVEL="${SAFETY_LEVEL:-I}"
TASK_INDEX="${TASK_INDEX:-0}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_BRANCH_CHUNKS="${MAX_BRANCH_CHUNKS:-2}"

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
        break
    fi
    sleep 5
done

PYTHONPATH="${ROOT}/main:${ROOT}/openpi/packages/openpi-client/src:${ROOT}/safelibero" \
    /home/lythk/vlsa-aegis/.run_venv/bin/python \
    "${ROOT}/main/collect_denoising_value_data.py" \
    --host 127.0.0.1 --port "${PORT}" \
    --task-suite-name "${TASK_SUITE_NAME}" --safety-level "${SAFETY_LEVEL}" \
    --task-index "${TASK_INDEX}" --episode-index "${EPISODE_INDEX}" \
    --output-root "${OUTPUT_ROOT}" \
    --max-steps "${MAX_STEPS}" \
    --max-branch-chunks-per-episode "${MAX_BRANCH_CHUNKS}"
