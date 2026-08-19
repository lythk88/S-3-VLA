#!/usr/bin/env bash
# Evaluate the time-conditioned flow-guidance policy on the regenerated
# one-pudding-obstacle scenes for a single SafeLIBERO suite.
#
# Two configurations share one guidance server:
#   guided_late_default  - the value run's own approach.json guidance
#                          (08_late_time: scale 0.35, times 0.1, gate 0.5)
#   guided_riskgate_sm   - the latest strength/timing sweep setting
#                          (scale 0.5, times 0.5,0.3,0.1, gate 0.7)
#
# usage: run_chocolate_pudding_all_suites_guided.sh <suite> <port>
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

# Each suite exposes four tasks; every generated scene holds one init state.
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
VALUE_RUN_DIR="${VALUE_RUN_DIR:-${ROOT}/Safety-value-function/time_conditioned_10way_v1/08_late_time}"
GROUNDINGDINO_ROOT="${GROUNDINGDINO_ROOT:-/home/lythk/vlsa-aegis/GroundingDINO}"
SAFETY_LEVEL="${SAFETY_LEVEL:-I}"
JOB_TAG="${SUITE}_guided_${SLURM_JOB_ID:-manual}"

mkdir -p "${LOG_DIR}" "${VIDEO_ROOT}" "${ROOT}/.worker_tmp/${JOB_TAG}"

export LIBERO_CONFIG_PATH="${CONFIG_ROOT}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TMPDIR="${ROOT}/.worker_tmp/${JOB_TAG}"
export PYTHONUNBUFFERED=1

for required in \
    "${CONFIG_ROOT}/config.yaml" \
    "${CHECKPOINT_DIR}" \
    "${VALUE_RUN_DIR}/best_model.pt" \
    "${VALUE_RUN_DIR}/approach.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required evaluation input: ${required}" >&2
        exit 1
    fi
done

# Guidance geometry/normalization/integration come from the value run itself,
# exactly as run_time_conditioned_10way_confirmation.sh resolves them.
mapfile -t BASE_GUIDANCE_ARGS < <(
    "${EVAL_PYTHON}" - "${VALUE_RUN_DIR}/approach.json" <<'PY'
import json
import pathlib
import sys

guidance = json.loads(pathlib.Path(sys.argv[1]).read_text())["guidance"]
for name, value in (
    ("--time-conditioned-guidance-normalization", guidance["normalization"]),
    ("--time-conditioned-guidance-geometry", guidance["geometry"]),
    ("--time-conditioned-guidance-integration", guidance["integration"]),
    ("--time-conditioned-clearance-score-weight", guidance["clearance_score_weight"]),
):
    print(name)
    print(value)
print(
    "--time-conditioned-guidance-translation-only"
    if guidance["translation_only"]
    else "--no-time-conditioned-guidance-translation-only"
)
if guidance["value_backtracking"]:
    print("--time-conditioned-value-backtracking")
PY
)

mapfile -t DEFAULT_STRENGTH < <(
    "${EVAL_PYTHON}" - "${VALUE_RUN_DIR}/approach.json" <<'PY'
import json
import pathlib
import sys

guidance = json.loads(pathlib.Path(sys.argv[1]).read_text())["guidance"]
print("--time-conditioned-guidance-times")
print(",".join(str(x) for x in guidance["times"]))
print("--time-conditioned-guidance-scale")
print(guidance["scale"])
PY
)

SERVER_PID=""

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT

start_guided_server() {
    local server_log="${LOG_DIR}/${JOB_TAG}_server.log"
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
    SERVER_PID=$!

    local ready=0
    for _ in $(seq 1 180); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "Guidance server exited during startup" >&2
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
        echo "Timed out waiting for guidance server on port ${PORT}" >&2
        tail -200 "${server_log}" >&2 || true
        return 1
    fi
}

run_guided() {
    local run_name="$1"
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
            --disable-safety-layer \
            --baseline-run-name guided_unused \
            --use-time-conditioned-guidance \
            --time-conditioned-value-run-dir "${VALUE_RUN_DIR}" \
            --time-conditioned-value-device cuda \
            --time-conditioned-run-name "${run_name}" \
            "${BASE_GUIDANCE_ARGS[@]}" \
            "$@" \
            2>&1 | tee "${LOG_DIR}/${JOB_TAG}_${run_name}.log"
}

echo "Guided evaluation ${SUITE} tasks ${TASKS[*]} (level ${SAFETY_LEVEL})"
echo "VALUE_RUN_DIR=${VALUE_RUN_DIR}"
echo "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

start_guided_server

# approach.json defaults (scale 0.35, times 0.1) with the default 0.5 gate.
run_guided guided_late_default \
    "${DEFAULT_STRENGTH[@]}" \
    --time-conditioned-safety-threshold 0.5

# Latest strength/timing sweep setting with the 0.7 risk gate.
run_guided guided_riskgate_sm \
    --time-conditioned-guidance-times 0.5,0.3,0.1 \
    --time-conditioned-guidance-scale 0.5 \
    --time-conditioned-safety-threshold 0.7

stop_server
echo "Suite complete: ${SUITE} -> ${VIDEO_ROOT}"
