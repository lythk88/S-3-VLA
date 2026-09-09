#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/safe-flow-matching
MODEL="${QWEN3_VL_MODEL:-/dev/shm/Qwen3-VL-8B-Instruct}"
DEVICE="${QWEN3_VL_DEVICE:-cpu}"
OUTPUT="${QWEN3_VL_OBSTACLE_EVAL_OUTPUT:-$ROOT/results/qwen3_vl_8b_obstacle_selection_vlsa_prompt}"

export PYTHONPATH="$ROOT/.deps_qwen:$ROOT/main:$ROOT/safelibero:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src"
export LIBERO_CONFIG_PATH="$ROOT/evaluation/libero_eval_config_local"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-20}"

exec "$ROOT/.venv/bin/python" "$ROOT/main/evaluate_qwen3_vl_obstacle_selection.py" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT" \
  "$@"
