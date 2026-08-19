#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lythk/safe-flow-matching"
export ROOT
export PYTHON="/home/lythk/vlsa-aegis/.run_venv/bin/python"
export BOOTSTRAP_ROOT="${ROOT}/training_dataset/pi05_hidden_chunks"
export TRACE_ROOT="${ROOT}/training_dataset/regeneration_backups/pi05_denoising_value_v1_pre_regeneration_39396"
export OUTPUT_DIR="${ROOT}/Safety-value-function/time_conditioned_clearance_phase1_updated_phase2_backup39396_v1"
export ALLOW_SOURCE_ONLY_GROUPS=1
export INFORMATIVE_PAIR_SAMPLING_FRACTION=0.5
export TRAIN_SEED=7
export TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"

exec bash "${ROOT}/scripts/run_time_conditioned_value_training_v1.sh"
