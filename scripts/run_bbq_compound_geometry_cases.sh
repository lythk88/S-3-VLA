#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "usage: $0 PORT" >&2
    exit 2
fi

ROOT=/workspace/safe-flow-matching
PORT="$1"
RUN_NAME=smcbf_joint_dcbf_l2_dsafe0_axis_aligned_compound_qwen3_vlm_cpu_v2
OUTPUT_ROOT="$ROOT/results/$RUN_NAME/safelibero_object"

export PYTHONPATH="$ROOT/.deps_qwen:$ROOT/main:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$ROOT/safelibero:$ROOT/GroundingDINO"
export LIBERO_CONFIG_PATH="$ROOT/evaluation/libero_eval_config_local"
export MUJOCO_GL=egl
export HF_HOME=/workspace/.hf_home
export GROUNDINGDINO_DEVICE=cpu
export OMP_NUM_THREADS=10
QWEN3_VL_MODEL="${QWEN3_VL_MODEL:-/dev/shm/Qwen3-VL-8B-Instruct}"

"$ROOT/.venv/bin/python" "$ROOT/main/main_aegis.py" \
    --task-suite-name safelibero_object \
    --safety-level II \
    --task-index 3 \
    --episode-index 0 1 2 3 4 5 6 7 8 9 \
    --num-trials-per-task 10 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --video-out-path "$OUTPUT_ROOT" \
    --resume-existing-episodes \
    --disable-safety-layer \
    --use-action-expert-guidance \
    --action-expert-run-name "$RUN_NAME" \
    --action-expert-critic-run-dir "$ROOT/Safety-value-function/success_critic_v1" \
    --policy-checkpoint-dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/pi05_libero \
    --action-expert-times 0.3,0.1 \
    --action-expert-beta-success 10.0 \
    --action-expert-trust-radius 0.05 \
    --action-expert-trust-region-norm l2 \
    --action-expert-final-success-trust-radius 0.05 \
    --action-expert-safe-distance 0.0 \
    --action-expert-adaptive-safety-trust-radii 0.1,0.2,0.3 \
    --action-expert-adaptive-escalate-on-first-barrier-only \
    --no-action-expert-minimal-intervention \
    --action-expert-translation-only-execution \
    --action-expert-candidates 1 \
    --action-expert-first-step-recovery \
    --action-expert-continue-on-unsafe \
    --action-expert-pre-execution-qp \
    --action-expert-closed-loop-reprojection \
    --action-expert-debug-rollout-geometry \
    --action-expert-force-axis-aligned-obb \
    --action-expert-compound-carried-object \
    --action-expert-carried-object-prompt "bbq sauce" \
    --action-expert-carried-object-candidate-top-k 5 \
    --action-expert-qwen-vlm-model "$QWEN3_VL_MODEL" \
    --action-expert-qwen-vlm-device cpu \
    --action-expert-grasp-close-threshold 0.0 \
    --action-expert-grasp-activation-distance 0.18 \
    --use-fixed-flow-noise \
    --save-videos \
    --no-save-rollout-data \
    --save-perception-diagnostics \
    --fail-on-episode-error
