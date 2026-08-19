#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/results/chocolate_pudding_spatial_4x3}"
CONFIG_ROOT="${CONFIG_ROOT:-${EVAL_ROOT}/scenes/config}"
LOG_DIR="${LOG_DIR:-${EVAL_ROOT}/logs}"
VIDEO_ROOT="${VIDEO_ROOT:-${EVAL_ROOT}/videos}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-openpi/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-sim/bin/python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
VALUE_RUN_DIR="${VALUE_RUN_DIR:-${ROOT}/Safety-value-function/time_conditioned_10way_v1/08_late_time}"
GROUNDINGDINO_ROOT="${GROUNDINGDINO_ROOT:-/home/lythk/vlsa-aegis/GroundingDINO}"
PLAIN_PORT="${PLAIN_PORT:-8123}"
GUIDED_PORT="${GUIDED_PORT:-8124}"
TASKS=(0 1 2 3)
EPISODES=(0)

mkdir -p "${LOG_DIR}" "${VIDEO_ROOT}" "${ROOT}/.worker_tmp/chocolate_pudding_spatial_4x3"

export LIBERO_CONFIG_PATH="${CONFIG_ROOT}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export GROUNDINGDINO_DEVICE="${GROUNDINGDINO_DEVICE:-cpu}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TMPDIR="${TMPDIR:-${ROOT}/.worker_tmp/chocolate_pudding_spatial_4x3}"
export PYTHONUNBUFFERED=1

for required in \
    "${CONFIG_ROOT}/config.yaml" \
    "${CHECKPOINT_DIR}" \
    "${VALUE_RUN_DIR}/best_model.pt" \
    "${GROUNDINGDINO_ROOT}/groundingdino_swint_ogc.pth"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required evaluation input: ${required}" >&2
        exit 1
    fi
done

SERVER_PID=""

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}

cleanup() {
    stop_server
}
trap cleanup EXIT

wait_for_server() {
    local port="$1"
    local server_log="$2"
    local ready=0
    for _ in $(seq 1 180); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "Policy server exited during startup" >&2
            tail -200 "${server_log}" >&2 || true
            return 1
        fi
        if "${EVAL_PYTHON}" - "${port}" <<'PY' >/dev/null 2>&1
import socket
import sys
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
        echo "Timed out waiting for policy server on port ${port}" >&2
        tail -200 "${server_log}" >&2 || true
        return 1
    fi
}

start_plain_server() {
    local server_log="${LOG_DIR}/pi05_server.log"
    (
        cd "${ROOT}"
        export PYTHONPATH="${ROOT}/openpi/src"
        exec "${OPENPI_PYTHON}" "${ROOT}/scripts/serve_policy.py" \
            --port "${PLAIN_PORT}" \
            policy:checkpoint \
            --policy.config pi05_libero \
            --policy.dir "${CHECKPOINT_DIR}"
    ) >"${server_log}" 2>&1 &
    SERVER_PID=$!
    wait_for_server "${PLAIN_PORT}" "${server_log}"
}

start_guided_server() {
    local server_log="${LOG_DIR}/guided_server.log"
    (
        cd "${ROOT}"
        export PYTHONPATH="${ROOT}/openpi/src"
        exec "${OPENPI_PYTHON}" "${ROOT}/scripts/serve_time_conditioned_guidance_policy.py" \
            --port "${GUIDED_PORT}" \
            --checkpoint-dir "${CHECKPOINT_DIR}" \
            --value-run-dir "${VALUE_RUN_DIR}" \
            --torch-device cuda \
            --deterministic-value
    ) >"${server_log}" 2>&1 &
    SERVER_PID=$!
    wait_for_server "${GUIDED_PORT}" "${server_log}"
}

run_eval() {
    local method="$1"
    local port="$2"
    shift 2
    PYTHONPATH="${ROOT}/main:${ROOT}/openpi/packages/openpi-client/src:${ROOT}/safelibero:${GROUNDINGDINO_ROOT}" \
        "${EVAL_PYTHON}" "${ROOT}/main/main_aegis.py" \
            --host 127.0.0.1 \
            --port "${port}" \
            --task-suite-name safelibero_spatial \
            --safety-level I \
            --task-index "${TASKS[@]}" \
            --episode-index "${EPISODES[@]}" \
            --num-trials-per-task 1 \
            --video-out-path "${VIDEO_ROOT}" \
            --policy-checkpoint-dir "${CHECKPOINT_DIR}" \
            --use-fixed-flow-noise \
            --resume-existing-episodes \
            --fail-on-episode-error \
            "$@" \
            2>&1 | tee "${LOG_DIR}/${method}.log"
}

echo "Starting paired 4-task evaluation in ${EVAL_ROOT}"
echo "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

start_plain_server
run_eval pi05 "${PLAIN_PORT}" \
    --disable-safety-layer \
    --baseline-run-name pi05
run_eval vlsa "${PLAIN_PORT}" \
    --baseline-run-name vlsa \
    --groundingdino-config-path "${GROUNDINGDINO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
    --groundingdino-checkpoint-path "${GROUNDINGDINO_ROOT}/groundingdino_swint_ogc.pth"
stop_server

start_guided_server
run_eval guided_08_late_time "${GUIDED_PORT}" \
    --disable-safety-layer \
    --baseline-run-name guided_unused \
    --use-time-conditioned-guidance \
    --time-conditioned-value-run-dir "${VALUE_RUN_DIR}" \
    --time-conditioned-value-device cuda \
    --time-conditioned-run-name guided_08_late_time \
    --time-conditioned-guidance-times 0.1 \
    --time-conditioned-guidance-scale 0.35 \
    --time-conditioned-guidance-normalization task-step-rms \
    --time-conditioned-guidance-geometry direct \
    --time-conditioned-guidance-integration state \
    --time-conditioned-clearance-score-weight 0.5 \
    --time-conditioned-guidance-translation-only \
    --time-conditioned-value-backtracking
stop_server

PYTHONPATH="${ROOT}/main" "${EVAL_PYTHON}" \
    "${ROOT}/main/analyze_chocolate_pudding_spatial_eval.py" \
    --eval-root "${EVAL_ROOT}" \
    | tee "${LOG_DIR}/analysis.log"

echo "Evaluation complete: ${EVAL_ROOT}"
