#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
TASK_TMP_DIR="${TASK_TMP_DIR:-${ROOT}/.worker_tmp/pi05_spatial_heldout20_${SLURM_JOB_ID:-manual}}"

mkdir -p "${TASK_TMP_DIR}"
export TMPDIR="${TASK_TMP_DIR}"
export TMP="${TASK_TMP_DIR}"
export TEMP="${TASK_TMP_DIR}"
export ROOT
export PORT="${PORT:-8005}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/pi05_spatial_heldout_20ep}"
export LOG_DIR="${LOG_DIR:-${ROOT}/logs/pi05_spatial_heldout_20ep}"
export VALUE_RUN_DIR=""
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
export EVAL_PYTHON="${EVAL_PYTHON:-/home/lythk/vlsa-aegis/.run_venv/bin/python}"
export GROUNDINGDINO_ROOT="${GROUNDINGDINO_ROOT:-/home/lythk/vlsa-aegis/GroundingDINO}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${ROOT}/results/spatial_heldout20_split.json}"
test -f "${SPLIT_MANIFEST}"
EPISODE_VALUES=$("${EVAL_PYTHON}" - "${SPLIT_MANIFEST}" <<'PY'
import json, pathlib, sys
values = json.loads(pathlib.Path(sys.argv[1]).read_text())["selected_episodes"]
if len(values) != 20:
    raise SystemExit(f"expected 20 held-out episodes, found {len(values)}")
print(" ".join(map(str, values)))
PY
)
export BASELINE_RUN_NAME="pi05_plain_heldout20"
export ENABLE_GEOMETRIC_SHIELD="false"
export TASKS="0 1 2 3"
export LEVELS="I II"
export EPISODES="${EPISODE_VALUES}"
export MODES="baseline"
export RUN_ANALYSIS="false"
export RESUME_EXISTING_EPISODES="false"
export FAIL_ON_EPISODE_ERROR="true"

exec bash "${ROOT}/scripts/run_spatial_flow_guidance_pilot.sh"
