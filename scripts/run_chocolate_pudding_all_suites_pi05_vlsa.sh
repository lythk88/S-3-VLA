#!/usr/bin/env bash
# Evaluate original pi0.5 and VLSA on the regenerated one-pudding-obstacle
# scenes for a single SafeLIBERO suite. Both methods share one policy server;
# VLSA is the same nominal policy with the GroundingDINO safety layer enabled.
#
# usage: run_chocolate_pudding_all_suites_pi05_vlsa.sh <suite> <port>
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 <suite> <port>" >&2
    exit 2
fi

SUITE="$1"
PORT="$2"

case "${SUITE}" in
    safelibero_spatial|safelibero_goal|safelibero_object|safelibero_long) ;;
    *) echo "unsupported suite: ${SUITE}" >&2; exit 2 ;;
esac

# Every generated scene holds exactly one init state, so one episode per task.
# Every suite exposes four tasks through its benchmark task order. The goal
# suite's task map lists a fifth entry (put_the_cream_cheese_in_the_bowl) that
# the benchmark does not register, so it has a generated scene but no task id.
TASKS=(0 1 2 3)

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
SCENE_ROOT="${SCENE_ROOT:-${ROOT}/results/chocolate_pudding_all_suites}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/results/chocolate_pudding_all_suites_eval}"
CONFIG_ROOT="${CONFIG_ROOT:-${SCENE_ROOT}/scenes/config}"
LOG_DIR="${LOG_DIR:-${EVAL_ROOT}/logs}"
VIDEO_ROOT="${VIDEO_ROOT:-${EVAL_ROOT}/videos/${SUITE}}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-openpi/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-sim/bin/python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
GROUNDINGDINO_ROOT="${GROUNDINGDINO_ROOT:-/home/lythk/vlsa-aegis/GroundingDINO}"
SAFETY_LEVEL="${SAFETY_LEVEL:-I}"
JOB_TAG="${SUITE}_${SLURM_JOB_ID:-manual}"

mkdir -p "${LOG_DIR}" "${VIDEO_ROOT}" "${ROOT}/.worker_tmp/${JOB_TAG}"

export LIBERO_CONFIG_PATH="${CONFIG_ROOT}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export GROUNDINGDINO_DEVICE="${GROUNDINGDINO_DEVICE:-cpu}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TMPDIR="${ROOT}/.worker_tmp/${JOB_TAG}"
export PYTHONUNBUFFERED=1

for required in \
    "${CONFIG_ROOT}/config.yaml" \
    "${CHECKPOINT_DIR}" \
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
trap stop_server EXIT

start_plain_server() {
    local server_log="${LOG_DIR}/${JOB_TAG}_pi05_server.log"
    (
        cd "${ROOT}"
        export PYTHONPATH="${ROOT}/openpi/src"
        exec "${OPENPI_PYTHON}" "${ROOT}/scripts/serve_policy.py" \
            --port "${PORT}" \
            policy:checkpoint \
            --policy.config pi05_libero \
            --policy.dir "${CHECKPOINT_DIR}"
    ) >"${server_log}" 2>&1 &
    SERVER_PID=$!

    local ready=0
    for _ in $(seq 1 180); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "Policy server exited during startup" >&2
            tail -200 "${server_log}" >&2 || true
            return 1
        fi
        if "${EVAL_PYTHON}" - "${PORT}" <<'PY' >/dev/null 2>&1
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
        echo "Timed out waiting for policy server on port ${PORT}" >&2
        tail -200 "${server_log}" >&2 || true
        return 1
    fi
}

run_eval() {
    local method="$1"
    shift
    PYTHONPATH="${ROOT}/main:${ROOT}/openpi/packages/openpi-client/src:${ROOT}/safelibero:${GROUNDINGDINO_ROOT}" \
        "${EVAL_PYTHON}" "${ROOT}/main/main_aegis.py" \
            --host 127.0.0.1 \
            --port "${PORT}" \
            --task-suite-name "${SUITE}" \
            --safety-level "${SAFETY_LEVEL}" \
            --task-index "${TASKS[@]}" \
            --episode-index 0 \
            --num-trials-per-task 1 \
            --video-out-path "${VIDEO_ROOT}" \
            --policy-checkpoint-dir "${CHECKPOINT_DIR}" \
            --use-fixed-flow-noise \
            --resume-existing-episodes \
            --fail-on-episode-error \
            "$@" \
            2>&1 | tee "${LOG_DIR}/${JOB_TAG}_${method}.log"
}

echo "Evaluating ${SUITE} tasks ${TASKS[*]} (level ${SAFETY_LEVEL})"
echo "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
echo "VIDEO_ROOT=${VIDEO_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

start_plain_server

run_eval pi05 \
    --disable-safety-layer \
    --baseline-run-name pi05

run_eval vlsa \
    --baseline-run-name vlsa \
    --groundingdino-config-path "${GROUNDINGDINO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
    --groundingdino-checkpoint-path "${GROUNDINGDINO_ROOT}/groundingdino_swint_ogc.pth"

stop_server
echo "Suite complete: ${SUITE} -> ${VIDEO_ROOT}"
