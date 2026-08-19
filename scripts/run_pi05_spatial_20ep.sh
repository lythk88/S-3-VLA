#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
TASK_TMP_DIR="${TASK_TMP_DIR:-${ROOT}/.worker_tmp/pi05_spatial_20ep_${SLURM_JOB_ID:-manual}}"

mkdir -p "${TASK_TMP_DIR}"
export TMPDIR="${TASK_TMP_DIR}"
export TMP="${TASK_TMP_DIR}"
export TEMP="${TASK_TMP_DIR}"
export ROOT
export PORT="${PORT:-8005}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/pi05_spatial_20ep}"
export LOG_DIR="${LOG_DIR:-${ROOT}/logs/pi05_spatial_20ep}"
export VALUE_RUN_DIR=""
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
export EVAL_PYTHON="${EVAL_PYTHON:-/home/lythk/vlsa-aegis/.run_venv/bin/python}"
export GROUNDINGDINO_ROOT="${GROUNDINGDINO_ROOT:-/home/lythk/vlsa-aegis/GroundingDINO}"
export BASELINE_RUN_NAME="pi05_plain_20ep"
export ENABLE_GEOMETRIC_SHIELD="false"
export TASKS="0 1 2 3"
export LEVELS="I II"
export EPISODES="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
export MODES="baseline"
export RUN_ANALYSIS="false"
export RESUME_EXISTING_EPISODES="false"
export FAIL_ON_EPISODE_ERROR="true"

exec bash "${ROOT}/scripts/run_spatial_flow_guidance_pilot.sh"
