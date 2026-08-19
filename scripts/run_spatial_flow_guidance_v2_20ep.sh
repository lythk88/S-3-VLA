#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
TASK_TMP_DIR="${TASK_TMP_DIR:-${ROOT}/.worker_tmp/sfm_v2_${SLURM_JOB_ID:-manual}}"

mkdir -p "${TASK_TMP_DIR}"
export TMPDIR="${TASK_TMP_DIR}"
export TMP="${TASK_TMP_DIR}"
export TEMP="${TASK_TMP_DIR}"
export ROOT
export PORT="${PORT:-8005}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/spatial_flow_guidance_pilot}"
export LOG_DIR="${LOG_DIR:-${ROOT}/logs/spatial_flow_guidance_v2_xyz_s005_20ep}"
export VALUE_RUN_DIR="${VALUE_RUN_DIR:-${ROOT}/Safety-value-function/chunk_safety_value_run}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
export EVAL_PYTHON="${EVAL_PYTHON:-/home/lythk/vlsa-aegis/.run_venv/bin/python}"
export GROUNDINGDINO_ROOT="${GROUNDINGDINO_ROOT:-/home/lythk/vlsa-aegis/GroundingDINO}"
export BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-pi05_shielded_baseline_v2_20ep}"
export GUIDED_RUN_NAME="${GUIDED_RUN_NAME:-pi05_shielded_flow_v2_xyz_s005_20ep}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.05}"
export GUIDANCE_START_TIME="${GUIDANCE_START_TIME:-0.3}"
export GUIDANCE_TRANSLATION_ONLY="${GUIDANCE_TRANSLATION_ONLY:-true}"
export GUIDANCE_ORTHOGONAL="${GUIDANCE_ORTHOGONAL:-false}"
export ENABLE_GEOMETRIC_SHIELD="${ENABLE_GEOMETRIC_SHIELD:-true}"
export TASKS="${TASKS:-0 1 2 3}"
export LEVELS="${LEVELS:-I II}"
export EPISODES="${EPISODES:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
export MODES="${MODES:-baseline guided}"
export RUN_ANALYSIS="${RUN_ANALYSIS:-true}"

exec bash "${ROOT}/scripts/run_spatial_flow_guidance_pilot.sh"
