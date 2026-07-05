#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/namn1/vlsa-aegis}"
PORT="${PORT:-8001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/pi05_no_safety_full}"

TASKS=(0 1 2 3)
EPISODES=($(seq 0 49))

if [[ $# -gt 0 ]]; then
  SUITES=("$@")
else
  SUITES=(
    safelibero_spatial
    safelibero_object
    safelibero_goal
    safelibero_long
  )
fi

for suite in "${SUITES[@]}"; do
  for level in I II; do
    echo "[run] suite=${suite} level=${level} port=${PORT}"
    MUJOCO_GL=osmesa \
    PYTHONPATH="${ROOT}/safelibero" \
    "${ROOT}/main/.venv/bin/python" "${ROOT}/main/main_aegis.py" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --task-suite-name "${suite}" \
      --safety-level "${level}" \
      --task-index "${TASKS[@]}" \
      --episode-index "${EPISODES[@]}" \
      --disable-safety-layer \
      --video-out-path "${OUTPUT_ROOT}"
  done
done
