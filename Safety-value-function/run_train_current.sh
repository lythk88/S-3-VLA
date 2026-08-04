#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/namn1/vlsa-aegis}"
PYTHON="${PYTHON:-${ROOT}/main/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/results/pi05_no_safety_with_hidden_full}"
BASE_DIR="${BASE_DIR:-${ROOT}/MLP-classifier}"
RUN_ID="${RUN_ID:-pi05_hidden_state_mlp_current_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-${BASE_DIR}/runs/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[run_dir] ${RUN_DIR}"
echo "[log] ${LOG_FILE}"
echo "[data_root] ${DATA_ROOT}"

"${PYTHON}" "${BASE_DIR}/train_hidden_state_mlp.py" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${RUN_DIR}" \
  --device cpu \
  "$@" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}

echo "[finish] $(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status}"
echo "[run_dir] ${RUN_DIR}"
echo "[log] ${LOG_FILE}"

if [[ "${KEEP_SHELL:-1}" == "1" ]]; then
  exec bash
fi

exit "${status}"
