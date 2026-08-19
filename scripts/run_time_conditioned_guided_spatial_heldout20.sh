#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
PORT="${PORT:-8007}"
VALUE_RUN_DIR="${VALUE_RUN_DIR:-${ROOT}/Safety-value-function/time_conditioned_clearance_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/spatial_flow_guidance_pilot}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/time_conditioned_guided_spatial_heldout20}"
TASK_TMP_DIR="${TASK_TMP_DIR:-${ROOT}/.worker_tmp/time_guided_heldout20_${SLURM_JOB_ID:-manual}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/lythk/vlsa-aegis/.run_venv/bin/python}"
GROUNDINGDINO_ROOT="${GROUNDINGDINO_ROOT:-/home/lythk/vlsa-aegis/GroundingDINO}"
RUN_NAME="${RUN_NAME:-pi05_time_value_guided_heldout20}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${ROOT}/results/spatial_heldout20_split.json}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${TASK_TMP_DIR}"
export TMPDIR="${TASK_TMP_DIR}"
export TMP="${TASK_TMP_DIR}"
export TEMP="${TASK_TMP_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

"${EVAL_PYTHON}" - "${VALUE_RUN_DIR}" <<'PY'
import json, pathlib, sys
manifest = json.loads((pathlib.Path(sys.argv[1]) / "training_manifest.json").read_text())
if manifest.get("status") != "live_gradient_gate_passed":
    raise SystemExit(f"refusing benchmark: value model status is {manifest.get('status')!r}")
PY
mapfile -t SELECTED_EPISODES < <(
    "${EVAL_PYTHON}" - "${SPLIT_MANIFEST}" <<'PY'
import json, pathlib, sys
values = json.loads(pathlib.Path(sys.argv[1]).read_text())["selected_episodes"]
if len(values) != 20:
    raise SystemExit(f"expected 20 held-out episodes, found {len(values)}")
for value in values:
    print(value)
PY
)
if [[ "${#SELECTED_EPISODES[@]}" -ne 20 ]]; then
    echo "Failed to read 20 held-out episodes from ${SPLIT_MANIFEST}" >&2
    exit 1
fi

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

(
    cd "${ROOT}/openpi"
    exec uv run python "${ROOT}/scripts/serve_time_conditioned_guidance_policy.py" \
        --port "${PORT}" \
        --value-run-dir "${VALUE_RUN_DIR}" \
        --checkpoint-dir "${CHECKPOINT_DIR}"
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

for level in I II; do
    PYTHONPATH="${ROOT}/main:${ROOT}/openpi/packages/openpi-client/src:${ROOT}/safelibero:${GROUNDINGDINO_ROOT}" \
        "${EVAL_PYTHON}" "${ROOT}/main/main_aegis.py" \
        --host 127.0.0.1 --port "${PORT}" \
        --task-suite-name safelibero_spatial \
        --safety-level "${level}" \
        --task-index 0 1 2 3 \
        --episode-index "${SELECTED_EPISODES[@]}" \
        --num-trials-per-task 20 \
        --video-out-path "${OUTPUT_ROOT}" \
        --disable-safety-layer \
        --use-fixed-flow-noise \
        --use-time-conditioned-guidance \
        --time-conditioned-value-run-dir "${VALUE_RUN_DIR}" \
        --time-conditioned-guidance-scale 0.05 \
        --time-conditioned-guidance-time 0.3 \
        --time-conditioned-clearance-score-weight 0.5 \
        --time-conditioned-run-name "${RUN_NAME}" \
        --policy-checkpoint-dir "${CHECKPOINT_DIR}" \
        --fail-on-episode-error \
        2>&1 | tee "${LOG_DIR}/guided_${level}.log"
done

PYTHONPATH="${ROOT}/main" "${EVAL_PYTHON}" \
    "${ROOT}/main/analyze_time_conditioned_heldout.py" \
    --baseline-root "${ROOT}/results/pi05_spatial_heldout_20ep" \
    --guided-root "${OUTPUT_ROOT}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --output-json "${OUTPUT_ROOT}/time_conditioned_heldout20_comparison.json" \
    --output-markdown "${OUTPUT_ROOT}/TIME_CONDITIONED_HELDOUT20_RESULTS.md" \
    2>&1 | tee "${LOG_DIR}/analysis.log"
