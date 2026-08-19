#!/usr/bin/env bash
set -euo pipefail

INDEX="${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array job}"
case "${INDEX}" in
    0) export MODE=guided VARIANT=01_balanced LEVEL=I ;;
    1) export MODE=guided VARIANT=01_balanced LEVEL=II ;;
    2) export MODE=guided VARIANT=08_late_time LEVEL=I ;;
    3) export MODE=guided VARIANT=08_late_time LEVEL=II ;;
    *) echo "Invalid confirmation array index ${INDEX}" >&2; exit 64 ;;
esac
export PORT="$((8080 + INDEX))"
exec bash "${ROOT:-/home/lythk/safe-flow-matching}/scripts/run_time_conditioned_10way_confirmation.sh"
