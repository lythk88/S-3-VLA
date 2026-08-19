#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
PORT="${PORT:?Set a unique policy-server PORT}"
LEVEL="${LEVEL:?Set LEVEL to I or II}"
if [[ "${LEVEL}" != I && "${LEVEL}" != II ]]; then
    echo "LEVEL must be I or II" >&2
    exit 2
fi

VALUE_RUN_DIR="${VALUE_RUN_DIR:-${ROOT}/Safety-value-function/failed_runs/time_conditioned_clearance_v1_job37236_offline_gate_failed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/time_conditioned_trust_remaining_heldout20}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-pi05_plain_heldout20}"
GUIDED_RUN_NAME="${GUIDED_RUN_NAME:-pi05_time_value_trust_t03_r050_heldout20_diag}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/time_conditioned_trust_remaining_heldout20/level_${LEVEL}}"
TASK_TMP_DIR="${TASK_TMP_DIR:-${ROOT}/.worker_tmp/time_conditioned_trust_remaining_heldout20_${SLURM_JOB_ID:-manual}_${LEVEL}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-openpi/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-sim/bin/python}"
GROUNDINGDINO_ROOT="${GROUNDINGDINO_ROOT:-/home/lythk/vlsa-aegis/GroundingDINO}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${ROOT}/results/spatial_heldout20_split.json}"
SUITES=(safelibero_object safelibero_goal safelibero_long)
TASKS=(0 1 2 3)

mapfile -t EPISODES < <(
    "${EVAL_PYTHON}" - "${SPLIT_MANIFEST}" <<'PY'
import json, pathlib, sys
episodes = json.loads(pathlib.Path(sys.argv[1]).read_text())["selected_episodes"]
if len(episodes) != 20:
    raise SystemExit(f"expected exactly 20 held-out episodes, found {len(episodes)}")
if any(int(episode) < 8 for episode in episodes):
    raise SystemExit("held-out episodes must not overlap training episode IDs 0-7")
for episode in episodes:
    print(int(episode))
PY
)

test -f "${VALUE_RUN_DIR}/best_model.pt"
test -f "${VALUE_RUN_DIR}/training_manifest.json"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${TASK_TMP_DIR}"
export TMPDIR="${TASK_TMP_DIR}"
export TMP="${TASK_TMP_DIR}"
export TEMP="${TASK_TMP_DIR}"
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
    cd "${ROOT}"
    export PYTHONPATH="${ROOT}/openpi/src"
    exec "${OPENPI_PYTHON}" "${ROOT}/scripts/serve_time_conditioned_guidance_policy.py" \
        --port "${PORT}" \
        --value-run-dir "${VALUE_RUN_DIR}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --torch-device cpu \
        --deterministic-value \
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
    local mode="$3"
    local output_dir="${OUTPUT_ROOT}/${suite}"
    local run_name
    local -a guidance_args=()
    if [[ "${mode}" == guided ]]; then
        run_name="${GUIDED_RUN_NAME}"
        guidance_args=(
            --use-time-conditioned-guidance
            --time-conditioned-value-run-dir "${VALUE_RUN_DIR}"
            --time-conditioned-guidance-time 0.3
            --time-conditioned-guidance-scale 0.5
            --time-conditioned-guidance-normalization task-step-rms
            --time-conditioned-guidance-geometry direct
            --time-conditioned-guidance-integration state
            --time-conditioned-guidance-translation-only
            --time-conditioned-value-device cpu
            --time-conditioned-run-name "${GUIDED_RUN_NAME}"
            --time-conditioned-clearance-score-weight 0.5
        )
    else
        run_name="${BASELINE_RUN_NAME}"
        guidance_args=(--baseline-run-name "${BASELINE_RUN_NAME}")
    fi

    mkdir -p "${output_dir}" "${LOG_DIR}/${suite}"
    PYTHONPATH="${ROOT}/main:${ROOT}/openpi/packages/openpi-client/src:${ROOT}/safelibero:${GROUNDINGDINO_ROOT}" \
        "${EVAL_PYTHON}" "${ROOT}/main/main_aegis.py" \
            --host 127.0.0.1 \
            --port "${PORT}" \
            --task-suite-name "${suite}" \
            --safety-level "${LEVEL}" \
            --task-index "${task}" \
            --episode-index "${EPISODES[@]}" \
            --num-trials-per-task 20 \
            --video-out-path "${output_dir}" \
            --disable-safety-layer \
            --use-fixed-flow-noise \
            --policy-checkpoint-dir "${CHECKPOINT_DIR}" \
            --resume-existing-episodes \
            --fail-on-episode-error \
            "${guidance_args[@]}" \
            >"${LOG_DIR}/${suite}/${run_name}_${LEVEL}_task${task}.log" 2>&1
}

run_mode() {
    local suite="$1"
    local mode="$2"
    local -a pids=()
    local failed=0
    for task in "${TASKS[@]}"; do
        run_task "${suite}" "${task}" "${mode}" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    if [[ "${failed}" == 1 ]]; then
        echo "At least one ${suite}/${LEVEL}/${mode} simulator failed; inspect ${LOG_DIR}/${suite}" >&2
        exit 1
    fi
    echo "Completed ${suite} ${LEVEL} ${mode}: 80 held-out rollouts"
}

for suite in "${SUITES[@]}"; do
    run_mode "${suite}" baseline
    run_mode "${suite}" guided
done

echo "Completed remaining SafeLIBERO suites level ${LEVEL}: 480 paired-method rollouts"
