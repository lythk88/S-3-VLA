#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
PORT="${PORT:-8043}"
LEVEL="${LEVEL:?Set LEVEL to I or II}"
if [[ "${LEVEL}" != I && "${LEVEL}" != II ]]; then
    echo "LEVEL must be I or II" >&2
    exit 2
fi
VALUE_RUN_DIR="${VALUE_RUN_DIR:-${ROOT}/Safety-value-function/failed_runs/time_conditioned_clearance_v1_job37236_offline_gate_failed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/spatial_flow_guidance_pilot}"
RUN_NAME="${RUN_NAME:-pi05_time_value_trust_t03_r050_heldout20_diag}"
LOG_LABEL="${LOG_LABEL:-trust_heldout20_${LEVEL}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/time_conditioned_trust_heldout20_diagnostic/${LOG_LABEL}}"
TASK_TMP_DIR="${TASK_TMP_DIR:-${ROOT}/.worker_tmp/time_conditioned_trust_heldout20_${SLURM_JOB_ID:-manual}_${LEVEL}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-openpi/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-sim/bin/python}"
GROUNDINGDINO_ROOT="${GROUNDINGDINO_ROOT:-/home/lythk/vlsa-aegis/GroundingDINO}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${ROOT}/results/spatial_heldout20_split.json}"

mapfile -t EPISODES < <(
    "${EVAL_PYTHON}" - "${SPLIT_MANIFEST}" <<'PY'
import json, pathlib, sys
episodes = json.loads(pathlib.Path(sys.argv[1]).read_text())["selected_episodes"]
if len(episodes) != 20:
    raise SystemExit(f"expected exactly 20 held-out episodes, found {len(episodes)}")
for episode in episodes:
    print(int(episode))
PY
)

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
    local task="$1"
    PYTHONPATH="${ROOT}/main:${ROOT}/openpi/packages/openpi-client/src:${ROOT}/safelibero:${GROUNDINGDINO_ROOT}" \
        "${EVAL_PYTHON}" "${ROOT}/main/main_aegis.py" \
            --host 127.0.0.1 \
            --port "${PORT}" \
            --task-suite-name safelibero_spatial \
            --safety-level "${LEVEL}" \
            --task-index "${task}" \
            --episode-index "${EPISODES[@]}" \
            --num-trials-per-task 20 \
            --video-out-path "${OUTPUT_ROOT}" \
            --disable-safety-layer \
            --use-fixed-flow-noise \
            --use-time-conditioned-guidance \
            --time-conditioned-value-run-dir "${VALUE_RUN_DIR}" \
            --time-conditioned-guidance-time 0.3 \
            --time-conditioned-guidance-scale 0.5 \
            --time-conditioned-guidance-normalization task-step-rms \
            --time-conditioned-guidance-geometry direct \
            --time-conditioned-guidance-integration state \
            --time-conditioned-guidance-translation-only \
            --time-conditioned-value-device cpu \
            --time-conditioned-run-name "${RUN_NAME}" \
            --time-conditioned-clearance-score-weight 0.5 \
            --policy-checkpoint-dir "${CHECKPOINT_DIR}" \
            --fail-on-episode-error \
            >"${LOG_DIR}/${RUN_NAME}_${LEVEL}_task${task}.log" 2>&1
}

pids=()
for task in 0 1 2 3; do
    run_task "${task}" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
done
if [[ "${failed}" == 1 ]]; then
    echo "At least one ${LEVEL} simulator failed; inspect ${LOG_DIR}" >&2
    exit 1
fi
echo "Completed ${RUN_NAME} level ${LEVEL}: 80 diagnostic held-out rollouts"
