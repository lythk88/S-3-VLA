#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
PORT="${PORT:-8001}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
VALUE_RUN_DIR="${VALUE_RUN_DIR:-${ROOT}/Safety-value-function/chunk_safety_value_run}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/spatial_flow_guidance_pilot}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/spatial_flow_guidance_pilot}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.25}"
GUIDANCE_START_TIME="${GUIDANCE_START_TIME:-0.5}"
TASKS=(${TASKS:-0 1 2 3})
LEVELS=(${LEVELS:-I II})
EPISODES=(${EPISODES:-0})
MODES=(${MODES:-baseline guided})

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
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
    exec uv run python "${ROOT}/scripts/serve_policy.py" \
        --port "${PORT}" \
        --safety-value-run-dir "${VALUE_RUN_DIR}" \
        policy:checkpoint \
        --policy.config pi05_libero \
        --policy.dir "${CHECKPOINT_DIR}"
) >"${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 180); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        tail -100 "${LOG_DIR}/server.log"
        exit 1
    fi
    if "${ROOT}/main/.venv/bin/python" - "${PORT}" <<'PY' >/dev/null 2>&1
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
    tail -100 "${LOG_DIR}/server.log"
    exit 1
fi

for mode in "${MODES[@]}"; do
    for level in "${LEVELS[@]}"; do
        extra_args=(--use-fixed-flow-noise)
        if [[ "${mode}" == guided ]]; then
            extra_args+=(
                --use-flow-guidance
                --flow-guidance-scale "${GUIDANCE_SCALE}"
                --flow-guidance-start-time "${GUIDANCE_START_TIME}"
            )
        fi
        PYTHONPATH="${ROOT}/main:${ROOT}/openpi/packages/openpi-client/src:${ROOT}/safelibero" \
            "${ROOT}/main/.venv/bin/python" "${ROOT}/main/main_aegis.py" \
                --host 127.0.0.1 \
                --port "${PORT}" \
                --task-suite-name safelibero_spatial \
                --safety-level "${level}" \
                --task-index "${TASKS[@]}" \
                --episode-index "${EPISODES[@]}" \
                --disable-safety-layer \
                --video-out-path "${OUTPUT_ROOT}" \
                "${extra_args[@]}" \
                2>&1 | tee "${LOG_DIR}/${mode}_${level}.log"
    done
done

PYTHONPATH="${ROOT}/main" "${ROOT}/main/.venv/bin/python" \
    "${ROOT}/main/analyze_spatial_flow_guidance.py" \
    --results-root "${OUTPUT_ROOT}" \
    --score-run-dir "${VALUE_RUN_DIR}" \
    | tee "${LOG_DIR}/analysis.txt"
