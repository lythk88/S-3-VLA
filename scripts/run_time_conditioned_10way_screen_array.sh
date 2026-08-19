#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
PYTHON="${PYTHON:-/home/lythk/vlsa-aegis/.run_venv/bin/python}"
CATALOG="${ROOT}/Safety-value-function/time_conditioned_10way_variants.py"

mapfile -t VARIANTS < <("${PYTHON}" "${CATALOG}" --list)
INDEX="${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array job}"
if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
    echo "Invalid array index ${INDEX}; catalog contains ${#VARIANTS[@]} variants" >&2
    exit 64
fi

export VARIANT="${VARIANTS[INDEX]}"
export PORT="$((8060 + INDEX))"
if (( INDEX == 0 )); then
    export RUN_BASELINE=true
fi
exec bash "${ROOT}/scripts/run_time_conditioned_10way_screen_variant.sh"
