#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
VARIANT="${VARIANT:?Set VARIANT to one ten-way catalog entry}"
PORT="${PORT:?Set a unique policy-server PORT}"
RUN_BASELINE="${RUN_BASELINE:-false}"
EPISODES_TEXT="${EPISODES_TEXT:-8 9}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/time_conditioned_10way_v1/spatial_screen_ep8_9}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/time_conditioned_10way_v1/screen/${VARIANT}}"
TASK_TMP_DIR="${TASK_TMP_DIR:-${ROOT}/.worker_tmp/time_conditioned_10way_screen_${SLURM_JOB_ID:-manual}_${VARIANT}}"
VALUE_RUN_DIR="${VALUE_RUN_DIR:-${ROOT}/Safety-value-function/time_conditioned_10way_v1/${VARIANT}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-openpi/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-sim/bin/python}"
GROUNDINGDINO_ROOT="${GROUNDINGDINO_ROOT:-/home/lythk/vlsa-aegis/GroundingDINO}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-pi05_plain_tc10_screen}"
GUIDED_RUN_NAME="${GUIDED_RUN_NAME:-pi05_tc10_${VARIANT}_screen}"

read -r -a EPISODES <<<"${EPISODES_TEXT}"
if [[ "${#EPISODES[@]}" == 0 ]]; then
    echo "EPISODES_TEXT must contain at least one episode" >&2
    exit 64
fi
for episode in "${EPISODES[@]}"; do
    if (( episode < 8 )); then
        echo "Screen episode ${episode} overlaps training episodes 0-7" >&2
        exit 64
    fi
done

test -f "${VALUE_RUN_DIR}/best_model.pt"
test -f "${VALUE_RUN_DIR}/training_manifest.json"
test -f "${VALUE_RUN_DIR}/approach.json"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${TASK_TMP_DIR}"
export TMPDIR="${TASK_TMP_DIR}"
export TMP="${TASK_TMP_DIR}"
export TEMP="${TASK_TMP_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mapfile -t GUIDANCE_ARGS < <(
    "${EVAL_PYTHON}" - "${VALUE_RUN_DIR}/approach.json" <<'PY'
import json
import pathlib
import sys

guidance = json.loads(pathlib.Path(sys.argv[1]).read_text())["guidance"]
times = ",".join(str(value) for value in guidance["times"])
print("--time-conditioned-guidance-times")
print(times)
print("--time-conditioned-guidance-scale")
print(guidance["scale"])
print("--time-conditioned-guidance-normalization")
print(guidance["normalization"])
print("--time-conditioned-guidance-geometry")
print(guidance["geometry"])
print("--time-conditioned-guidance-integration")
print(guidance["integration"])
print("--time-conditioned-clearance-score-weight")
print(guidance["clearance_score_weight"])
print(
    "--time-conditioned-guidance-translation-only"
    if guidance["translation_only"]
    else "--no-time-conditioned-guidance-translation-only"
)
if guidance["value_backtracking"]:
    print("--time-conditioned-value-backtracking")
PY
)

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
) >"${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 180); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        tail -120 "${LOG_DIR}/server.log"
        exit 1
    fi
    if "${EVAL_PYTHON}" - "${PORT}" <<'PY' >/dev/null 2>&1
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

run_task() {
    local level="$1"
    local task="$2"
    local mode="$3"
    local run_name
    local -a mode_args
    if [[ "${mode}" == baseline ]]; then
        run_name="${BASELINE_RUN_NAME}"
        mode_args=(--baseline-run-name "${BASELINE_RUN_NAME}")
    else
        run_name="${GUIDED_RUN_NAME}"
        mode_args=(
            --use-time-conditioned-guidance
            --time-conditioned-value-run-dir "${VALUE_RUN_DIR}"
            --time-conditioned-value-device cuda
            --time-conditioned-run-name "${GUIDED_RUN_NAME}"
            "${GUIDANCE_ARGS[@]}"
        )
    fi
    PYTHONPATH="${ROOT}/main:${ROOT}/openpi/packages/openpi-client/src:${ROOT}/safelibero:${GROUNDINGDINO_ROOT}" \
        "${EVAL_PYTHON}" "${ROOT}/main/main_aegis.py" \
            --host 127.0.0.1 \
            --port "${PORT}" \
            --task-suite-name safelibero_spatial \
            --safety-level "${level}" \
            --task-index "${task}" \
            --episode-index "${EPISODES[@]}" \
            --num-trials-per-task "${#EPISODES[@]}" \
            --video-out-path "${OUTPUT_ROOT}" \
            --disable-safety-layer \
            --use-fixed-flow-noise \
            --policy-checkpoint-dir "${CHECKPOINT_DIR}" \
            --resume-existing-episodes \
            --fail-on-episode-error \
            "${mode_args[@]}" \
            >"${LOG_DIR}/${run_name}_${level}_task${task}.log" 2>&1
}

run_strata() {
    local mode="$1"
    local level task failed
    local -a pids
    for level in I II; do
        pids=()
        for task in 0 1 2 3; do
            run_task "${level}" "${task}" "${mode}" &
            pids+=("$!")
        done
        failed=0
        for task in "${pids[@]}"; do
            wait "${task}" || failed=1
        done
        if [[ "${failed}" == 1 ]]; then
            echo "At least one ${mode}/${level} simulator failed; inspect ${LOG_DIR}" >&2
            exit 1
        fi
    done
}

if [[ "${RUN_BASELINE}" == true ]]; then
    run_strata baseline
fi
run_strata guided
echo "Completed ${VARIANT}: $((8 * ${#EPISODES[@]})) spatial screen rollouts"
