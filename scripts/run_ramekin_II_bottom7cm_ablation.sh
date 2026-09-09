#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/safe-flow-matching
RUN_NAME="${RUN_NAME:-smcbf_compound_z11_top2cm_bottom7cm_pi5_qph5_ramekin_II_v1}"
OBSTACLE_PRIMITIVE_KINDS="${OBSTACLE_PRIMITIVE_KINDS:-aabb,cylinder,sphere}"
BOTTOM_PADDING_M="${BOTTOM_PADDING_M-0.07}"
TOP_PADDING_M="${TOP_PADDING_M:-0.02}"
VLSA_MVEE_OBSTACLE="${VLSA_MVEE_OBSTACLE:-0}"
OUT="$ROOT/results/$RUN_NAME"
EPISODE_INDEX="${EPISODE_INDEX:-0 1 2 3 4 5 6 7 8 9}"
read -r -a EPISODES <<<"$EPISODE_INDEX"

export PYTHONPATH="$ROOT/.deps_qwen:$ROOT/main:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$ROOT/safelibero:$ROOT/GroundingDINO"
export LIBERO_CONFIG_PATH="$ROOT/evaluation/libero_eval_config_local"
export MUJOCO_GL=egl
export HF_HOME=/workspace/.hf_home
export GROUNDINGDINO_DEVICE=cpu
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20

mkdir -p "$OUT/logs"

GEOMETRY_ARGS=(--action-expert-obstacle-top-padding-m "$TOP_PADDING_M")
if [[ -n "$BOTTOM_PADDING_M" ]]; then
  GEOMETRY_ARGS+=(--action-expert-obstacle-bottom-padding-m "$BOTTOM_PADDING_M")
fi
if [[ "$VLSA_MVEE_OBSTACLE" == 1 ]]; then
  GEOMETRY_ARGS+=(--action-expert-vlsa-mvee-obstacle)
fi

exec "$ROOT/.venv/bin/python" "$ROOT/main/main_aegis.py" \
  --task-suite-name safelibero_spatial \
  --safety-level II \
  --task-index 1 \
  --episode-index "${EPISODES[@]}" \
  --num-trials-per-task 10 \
  --replan-steps 5 \
  --action-expert-qp-horizon 5 \
  --host 127.0.0.1 \
  --port 8011 \
  --video-out-path "$OUT" \
  --resume-existing-episodes \
  --disable-safety-layer \
  --use-action-expert-guidance \
  --action-expert-run-name "$RUN_NAME" \
  --action-expert-critic-run-dir "$ROOT/Safety-value-function/success_critic_v1" \
  --policy-checkpoint-dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/pi05_libero \
  --action-expert-times 0.3,0.1 \
  --action-expert-beta-success 10 \
  --action-expert-trust-radius 0.05 \
  --action-expert-trust-region-norm l2 \
  --action-expert-final-success-trust-radius 0.05 \
  --action-expert-safe-distance 0 \
  --action-expert-gripper-z-radius-m 0.11 \
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
  --action-expert-obstacle-primitive-kinds "$OBSTACLE_PRIMITIVE_KINDS" \
  --action-expert-obstacle-padding-m 0 \
  "${GEOMETRY_ARGS[@]}" \
  --action-expert-compound-carried-object \
  --action-expert-agentview-only \
  --action-expert-grasp-close-threshold 0 \
  --action-expert-grasp-activation-distance 0.18 \
  --diagnose-gripper-obstacle-contacts \
  --diagnostic-top-contact-band-m 0.02 \
  --use-fixed-flow-noise \
  --save-videos \
  --no-save-rollout-data \
  --save-perception-diagnostics \
  --fail-on-episode-error
