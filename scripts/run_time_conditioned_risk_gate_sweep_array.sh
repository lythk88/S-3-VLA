#!/usr/bin/env bash
set -euo pipefail

INDEX="${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array job}"
case "${INDEX}" in
    0) export MODE=baseline LEVEL=I RUN_TAG=plain ;;
    1) export MODE=baseline LEVEL=II RUN_TAG=plain ;;
    2) export MODE=guided LEVEL=I SAFETY_THRESHOLD=0.3 RUN_TAG=t03 ;;
    3) export MODE=guided LEVEL=II SAFETY_THRESHOLD=0.3 RUN_TAG=t03 ;;
    4) export MODE=guided LEVEL=I SAFETY_THRESHOLD=0.5 RUN_TAG=t05 ;;
    5) export MODE=guided LEVEL=II SAFETY_THRESHOLD=0.5 RUN_TAG=t05 ;;
    6) export MODE=guided LEVEL=I SAFETY_THRESHOLD=0.7 RUN_TAG=t07 ;;
    7) export MODE=guided LEVEL=II SAFETY_THRESHOLD=0.7 RUN_TAG=t07 ;;
    *) echo "Invalid risk-gate sweep index ${INDEX}" >&2; exit 64 ;;
esac

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
export ROOT
export VARIANT=08_late_time
export PORT="$((8120 + INDEX))"
export EPISODES_TEXT="40 41 42 43 44 45 46 47 48 49"
export EXPECTED_EPISODE_COUNT=10
export FIXED_FLOW_NOISE=1
export OUTPUT_ROOT="${ROOT}/results/time_conditioned_risk_gate_sweep_v1"
export LOG_DIR="${ROOT}/logs/time_conditioned_risk_gate_sweep_v1/${RUN_TAG}_${LEVEL}"
export BASELINE_RUN_NAME="pi05_riskgate_plain_fixed"
export GUIDED_RUN_NAME="pi05_riskgate_${RUN_TAG}_fixed"

exec bash "${ROOT}/scripts/run_time_conditioned_10way_confirmation.sh"
