#!/usr/bin/env bash
# One evaluation arm on one suite of the 10-episode pudding scenes.
#
# Arms:
#   pi05           plain pi0.5, no safety layer
#   guided_sel     time-conditioned guidance, binary gate 0.5, scale 0.35
#   guided_margin  same gate, authority scaled by the margin below the gate
#
# Measured gate precision is 26-38%, so a binary gate spends most of its
# authority on chunks that were never going to collide. guided_margin ties the
# correction size to how far the score sits below the threshold, concentrating
# it on confident detections. Every arm sees the same ten validated initial
# states per task.
#
# usage: run_pudding_10ep_arm.sh <suite> <arm> <port>
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 <suite> <arm> <port>" >&2
    exit 2
fi

SUITE="$1"
ARM="$2"
PORT="$3"

case "${SUITE}" in
    safelibero_spatial|safelibero_goal|safelibero_object|safelibero_long) ;;
    *) echo "unsupported suite: ${SUITE}" >&2; exit 2 ;;
esac
case "${ARM}" in
    pi05|guided_sel|guided_margin) ;;
    *) echo "unsupported arm: ${ARM}" >&2; exit 2 ;;
esac

TASKS=(0 1 2 3)
EPISODES=(0 1 2 3 4 5 6 7 8 9)

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
SCENE_ROOT="${SCENE_ROOT:-${ROOT}/results/chocolate_pudding_10ep_scenes}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/results/chocolate_pudding_10ep_eval}"
CONFIG_ROOT="${CONFIG_ROOT:-${SCENE_ROOT}/scenes/config}"
LOG_DIR="${LOG_DIR:-${EVAL_ROOT}/logs}"
VIDEO_ROOT="${VIDEO_ROOT:-${EVAL_ROOT}/videos/${SUITE}}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-openpi/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-sim/bin/python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
VALUE_RUN_DIR="${VALUE_RUN_DIR:-${ROOT}/Safety-value-function/time_conditioned_10way_v1/08_late_time}"
GROUNDINGDINO_ROOT="${GROUNDINGDINO_ROOT:-/home/lythk/vlsa-aegis/GroundingDINO}"
SAFETY_LEVEL="${SAFETY_LEVEL:-I}"
JOB_TAG="${SUITE}_${ARM}_${SLURM_JOB_ID:-manual}"

mkdir -p "${LOG_DIR}" "${VIDEO_ROOT}" "${ROOT}/.worker_tmp/${JOB_TAG}"

export LIBERO_CONFIG_PATH="${CONFIG_ROOT}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TMPDIR="${ROOT}/.worker_tmp/${JOB_TAG}"
export PYTHONUNBUFFERED=1

for required in "${CONFIG_ROOT}/config.yaml" "${CHECKPOINT_DIR}"; do
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

wait_for_port() {
    local server_log="$1"
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

start_server() {
    local server_log="${LOG_DIR}/${JOB_TAG}_server.log"
    if [[ "${ARM}" == pi05 ]]; then
        (
            cd "${ROOT}"
            export PYTHONPATH="${ROOT}/openpi/src"
            exec "${OPENPI_PYTHON}" "${ROOT}/scripts/serve_policy.py" \
                --port "${PORT}" \
                policy:checkpoint \
                --policy.config pi05_libero \
                --policy.dir "${CHECKPOINT_DIR}"
        ) >"${server_log}" 2>&1 &
    else
        (
            cd "${ROOT}"
            export PYTHONPATH="${ROOT}/openpi/src"
            exec "${OPENPI_PYTHON}" "${ROOT}/scripts/serve_time_conditioned_guidance_policy.py" \
                --port "${PORT}" \
                --checkpoint-dir "${CHECKPOINT_DIR}" \
                --value-run-dir "${VALUE_RUN_DIR}" \
                --torch-device cuda \
                --deterministic-value
        ) >"${server_log}" 2>&1 &
    fi
    SERVER_PID=$!
    wait_for_port "${server_log}"
}

COMMON_ARGS=(
    --host 127.0.0.1
    --port "${PORT}"
    --task-suite-name "${SUITE}"
    --safety-level "${SAFETY_LEVEL}"
    --task-index "${TASKS[@]}"
    --episode-index "${EPISODES[@]}"
    --num-trials-per-task "${#EPISODES[@]}"
    --video-out-path "${VIDEO_ROOT}"
    --policy-checkpoint-dir "${CHECKPOINT_DIR}"
    --use-fixed-flow-noise
    --resume-existing-episodes
    --fail-on-episode-error
    --disable-safety-layer
)

GUIDED_SHARED=(
    --use-time-conditioned-guidance
    --time-conditioned-value-run-dir "${VALUE_RUN_DIR}"
    --time-conditioned-value-device cuda
    --time-conditioned-guidance-times 0.1
    --time-conditioned-guidance-normalization task-step-rms
    --time-conditioned-guidance-geometry direct
    --time-conditioned-guidance-integration state
    --time-conditioned-clearance-score-weight 0.5
    --time-conditioned-guidance-translation-only
    --time-conditioned-value-backtracking
    --time-conditioned-safety-threshold 0.5
)

case "${ARM}" in
    pi05)
        ARM_ARGS=(--baseline-run-name pi05)
        ;;
    guided_sel)
        ARM_ARGS=(
            --baseline-run-name guided_unused
            "${GUIDED_SHARED[@]}"
            --time-conditioned-guidance-scale 0.35
            --time-conditioned-run-name guided_sel
        )
        ;;
    guided_margin)
        ARM_ARGS=(
            --baseline-run-name guided_unused
            "${GUIDED_SHARED[@]}"
            --time-conditioned-guidance-scale 0.7
            --time-conditioned-margin-scaled
            --time-conditioned-run-name guided_margin
        )
        ;;
esac

echo "Arm ${ARM} on ${SUITE}: ${#TASKS[@]} tasks x ${#EPISODES[@]} episodes"
echo "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

start_server

PYTHONPATH="${ROOT}/main:${ROOT}/openpi/packages/openpi-client/src:${ROOT}/safelibero:${GROUNDINGDINO_ROOT}" \
    "${EVAL_PYTHON}" "${ROOT}/main/main_aegis.py" \
        "${COMMON_ARGS[@]}" \
        "${ARM_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/${JOB_TAG}.log"

stop_server
echo "Arm complete: ${SUITE} ${ARM} -> ${VIDEO_ROOT}"
