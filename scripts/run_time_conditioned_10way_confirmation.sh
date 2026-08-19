#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
MODE="${MODE:?Set MODE to baseline or guided}"
LEVEL="${LEVEL:?Set LEVEL to I or II}"
PORT="${PORT:?Set a unique policy-server PORT}"
VARIANT="${VARIANT:-01_balanced}"
if [[ "${MODE}" != baseline && "${MODE}" != guided ]]; then
    echo "MODE must be baseline or guided" >&2
    exit 64
fi
if [[ "${LEVEL}" != I && "${LEVEL}" != II ]]; then
    echo "LEVEL must be I or II" >&2
    exit 64
fi

EPISODES_TEXT="${EPISODES_TEXT:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-50}"
FIXED_FLOW_NOISE="${FIXED_FLOW_NOISE:-0}"
SAFETY_THRESHOLD="${SAFETY_THRESHOLD:-0.5}"
GUIDANCE_TIMES_OVERRIDE="${GUIDANCE_TIMES_OVERRIDE:-}"
GUIDANCE_SCALE_OVERRIDE="${GUIDANCE_SCALE_OVERRIDE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/time_conditioned_10way_v1/guided_50ep_all_safelibero}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/time_conditioned_10way_v1/confirmation/${MODE}_${VARIANT}_${LEVEL}}"
TASK_TMP_DIR="${TASK_TMP_DIR:-${ROOT}/.worker_tmp/time_conditioned_10way_confirmation_${SLURM_JOB_ID:-manual}_${MODE}_${VARIANT}_${LEVEL}}"
VALUE_RUN_DIR="${VALUE_RUN_DIR:-${ROOT}/Safety-value-function/time_conditioned_10way_v1/${VARIANT}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-openpi/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-sim/bin/python}"
GROUNDINGDINO_ROOT="${GROUNDINGDINO_ROOT:-/home/lythk/vlsa-aegis/GroundingDINO}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-pi05_plain_tc10_confirm}"
GUIDED_RUN_NAME="${GUIDED_RUN_NAME:-pi05_tc10_${VARIANT}_confirm}"
SUITES=(safelibero_spatial safelibero_object safelibero_goal safelibero_long)

read -r -a EPISODES <<<"${EPISODES_TEXT}"
if [[ "${#EPISODES[@]}" != "${EXPECTED_EPISODE_COUNT}" ]]; then
    echo "Expected ${EXPECTED_EPISODE_COUNT} confirmation episodes, found ${#EPISODES[@]}" >&2
    exit 64
fi
if [[ "${FIXED_FLOW_NOISE}" != 0 && "${FIXED_FLOW_NOISE}" != 1 ]]; then
    echo "FIXED_FLOW_NOISE must be 0 or 1" >&2
    exit 64
fi
if ! "${EVAL_PYTHON}" - "${SAFETY_THRESHOLD}" <<'PY' >/dev/null
import sys
value = float(sys.argv[1])
raise SystemExit(0 if 0.0 <= value <= 1.0 else 1)
PY
then
    echo "SAFETY_THRESHOLD must be in [0, 1]" >&2
    exit 64
fi
if [[ -n "${GUIDANCE_SCALE_OVERRIDE}" ]] && ! "${EVAL_PYTHON}" - "${GUIDANCE_SCALE_OVERRIDE}" <<'PY' >/dev/null
import sys
value = float(sys.argv[1])
raise SystemExit(0 if value >= 0.0 else 1)
PY
then
    echo "GUIDANCE_SCALE_OVERRIDE must be non-negative" >&2
    exit 64
fi
if [[ -n "${GUIDANCE_TIMES_OVERRIDE}" ]] && ! "${EVAL_PYTHON}" - "${GUIDANCE_TIMES_OVERRIDE}" <<'PY' >/dev/null
import sys
try:
    values = [float(x.strip()) for x in sys.argv[1].split(",") if x.strip()]
except ValueError:
    raise SystemExit(1)
allowed = {0.1, 0.3, 0.5}
raise SystemExit(0 if values and all(value in allowed for value in values) else 1)
PY
then
    echo "GUIDANCE_TIMES_OVERRIDE must use the trained times 0.1, 0.3, or 0.5" >&2
    exit 64
fi

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

GUIDANCE_ARGS=()
if [[ "${MODE}" == guided ]]; then
    mapfile -t GUIDANCE_ARGS < <(
        "${EVAL_PYTHON}" - "${VALUE_RUN_DIR}/approach.json" <<'PY'
import json
import pathlib
import sys

guidance = json.loads(pathlib.Path(sys.argv[1]).read_text())["guidance"]
for name, value in (
    ("--time-conditioned-guidance-times", ",".join(str(x) for x in guidance["times"])),
    ("--time-conditioned-guidance-scale", guidance["scale"]),
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
    if [[ -n "${GUIDANCE_TIMES_OVERRIDE}" ]]; then
        GUIDANCE_ARGS+=(--time-conditioned-guidance-times "${GUIDANCE_TIMES_OVERRIDE}")
    fi
    if [[ -n "${GUIDANCE_SCALE_OVERRIDE}" ]]; then
        GUIDANCE_ARGS+=(--time-conditioned-guidance-scale "${GUIDANCE_SCALE_OVERRIDE}")
    fi
    GUIDANCE_ARGS+=(--time-conditioned-safety-threshold "${SAFETY_THRESHOLD}")
fi

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
    local suite="$1"
    local task="$2"
    local suite_root="${OUTPUT_ROOT}/${suite}"
    local run_name
    local -a mode_args
    mkdir -p "${suite_root}" "${LOG_DIR}/${suite}"
    local -a noise_args=()
    if [[ "${FIXED_FLOW_NOISE}" == 1 ]]; then
        noise_args=(--use-fixed-flow-noise)
    fi
    if [[ "${MODE}" == baseline ]]; then
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
            --task-suite-name "${suite}" \
            --safety-level "${LEVEL}" \
            --task-index "${task}" \
            --episode-index "${EPISODES[@]}" \
            --num-trials-per-task "${#EPISODES[@]}" \
            --video-out-path "${suite_root}" \
            --disable-safety-layer \
            --policy-checkpoint-dir "${CHECKPOINT_DIR}" \
            --resume-existing-episodes \
            --fail-on-episode-error \
            "${noise_args[@]}" \
            "${mode_args[@]}" \
            >"${LOG_DIR}/${suite}/${run_name}_${LEVEL}_task${task}.log" 2>&1
}

for suite in "${SUITES[@]}"; do
    pids=()
    for task in 0 1 2 3; do
        run_task "${suite}" "${task}" &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    if [[ "${failed}" == 1 ]]; then
        echo "At least one ${suite}/${LEVEL}/${MODE} simulator failed; inspect ${LOG_DIR}" >&2
        exit 1
    fi
    echo "Completed ${suite}/${LEVEL}/${MODE}: $((4 * ${#EPISODES[@]})) rollouts"
done
echo "Completed ${MODE}/${VARIANT}/${LEVEL}: $((16 * ${#EPISODES[@]})) confirmation rollouts"
