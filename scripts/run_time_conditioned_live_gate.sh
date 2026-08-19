#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
VARIANT="${VARIANT:-08_late_time}"
PORT="${PORT:-8095}"
SUITE="${SUITE:-safelibero_spatial}"
MAX_ROLLOUTS="${MAX_ROLLOUTS:-0}"
VALUE_RUN_DIR="${VALUE_RUN_DIR:-${ROOT}/Safety-value-function/time_conditioned_10way_v1/${VARIANT}}"
TRACE_ROOT="${TRACE_ROOT:-${ROOT}/training_dataset/pi05_denoising_value_v1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-openpi/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-sim/bin/python}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/time_conditioned_10way_v1/live_gate_${VARIANT}}"
TASK_TMP_DIR="${TASK_TMP_DIR:-${ROOT}/.worker_tmp/time_conditioned_live_gate_${SLURM_JOB_ID:-manual}_${VARIANT}}"

test -f "${VALUE_RUN_DIR}/best_model.pt"
test -f "${VALUE_RUN_DIR}/training_manifest.json"
test -f "${VALUE_RUN_DIR}/approach.json"
mkdir -p "${LOG_DIR}" "${TASK_TMP_DIR}"

export TMPDIR="${TASK_TMP_DIR}"
export TMP="${TASK_TMP_DIR}"
export TEMP="${TASK_TMP_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

(
    cd "${ROOT}"
    export PYTHONPATH="${ROOT}/openpi/src"
    exec "${OPENPI_PYTHON}" "${ROOT}/scripts/serve_time_conditioned_guidance_policy.py" \
        --port "${PORT}" \
        --value-run-dir "${VALUE_RUN_DIR}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --torch-device cuda \
        --deterministic-value \
        --allow-offline-only \
        --allow-failed-offline-gate
) >"${LOG_DIR}/server-${SLURM_JOB_ID:-manual}.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 180); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        tail -120 "${LOG_DIR}/server-${SLURM_JOB_ID:-manual}.log"
        exit 1
    fi
    if "${EVAL_PYTHON}" -c \
        'import socket,sys; socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=2).close()' \
        "${PORT}" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 5
done
if [[ "${ready}" != 1 ]]; then
    tail -120 "${LOG_DIR}/server-${SLURM_JOB_ID:-manual}.log"
    exit 1
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}/main:${ROOT}/Safety-value-function:${ROOT}/openpi/packages/openpi-client/src:${ROOT}/safelibero"
"${EVAL_PYTHON}" "${ROOT}/Safety-value-function/evaluate_live_clearance_gradient.py" \
    --run-dir "${VALUE_RUN_DIR}" \
    --trace-root "${TRACE_ROOT}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --suite "${SUITE}" \
    --max-rollouts "${MAX_ROLLOUTS}"
