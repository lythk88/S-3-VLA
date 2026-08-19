#!/usr/bin/env bash
set -euo pipefail

INDEX="${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array job}"
case "${INDEX}" in
    0) export LEVEL=I  RUN_TAG=strong_single GUIDANCE_SCALE_OVERRIDE=0.7  GUIDANCE_TIMES_OVERRIDE=0.1 ;;
    1) export LEVEL=II RUN_TAG=strong_single GUIDANCE_SCALE_OVERRIDE=0.7  GUIDANCE_TIMES_OVERRIDE=0.1 ;;
    2) export LEVEL=I  RUN_TAG=multi        GUIDANCE_SCALE_OVERRIDE=0.35 GUIDANCE_TIMES_OVERRIDE=0.3,0.1 ;;
    3) export LEVEL=II RUN_TAG=multi        GUIDANCE_SCALE_OVERRIDE=0.35 GUIDANCE_TIMES_OVERRIDE=0.3,0.1 ;;
    4) export LEVEL=I  RUN_TAG=strong_multi GUIDANCE_SCALE_OVERRIDE=0.5  GUIDANCE_TIMES_OVERRIDE=0.5,0.3,0.1 ;;
    5) export LEVEL=II RUN_TAG=strong_multi GUIDANCE_SCALE_OVERRIDE=0.5  GUIDANCE_TIMES_OVERRIDE=0.5,0.3,0.1 ;;
    *) echo "Invalid strength/timing sweep index ${INDEX}" >&2; exit 64 ;;
esac

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
export ROOT
export MODE=guided
export VARIANT=08_late_time
export SAFETY_THRESHOLD=0.7
export PORT="$((8140 + INDEX))"
export EPISODES_TEXT="40 41 42 43 44 45 46 47 48 49"
export EXPECTED_EPISODE_COUNT=10
export FIXED_FLOW_NOISE=1
export OUTPUT_ROOT="${ROOT}/results/time_conditioned_strength_timing_sweep_v1"
export LOG_DIR="${ROOT}/logs/time_conditioned_strength_timing_sweep_v1/${RUN_TAG}_${LEVEL}"
export GUIDED_RUN_NAME="pi05_riskgate_t07_${RUN_TAG}_fixed"

exec bash "${ROOT}/scripts/run_time_conditioned_10way_confirmation.sh"
