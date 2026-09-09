#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/safe-flow-matching
RUN_NAME="${RUN_NAME:-smcbf_compound_z11_top2cm_h1_3shape_320_v1}"
PI_REPLAN_STEPS="${PI_REPLAN_STEPS:-1}"
QP_HORIZON="${QP_HORIZON:-1}"
OBSTACLE_PRIMITIVE_KINDS="${OBSTACLE_PRIMITIVE_KINDS:-aabb,cylinder,sphere}"
SAVE_PERCEPTION_DIAGNOSTICS="${SAVE_PERCEPTION_DIAGNOSTICS:-1}"
QP_SHAPE_SELECTION="${QP_SHAPE_SELECTION:-0}"
SHAPE_SELECTION_LAMBDA_INTERVENTION="${SHAPE_SELECTION_LAMBDA_INTERVENTION:-1.0}"
OUT="$ROOT/results/$RUN_NAME"

export PYTHONPATH="$ROOT/.deps_qwen:$ROOT/main:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$ROOT/safelibero:$ROOT/GroundingDINO"
export LIBERO_CONFIG_PATH="$ROOT/evaluation/libero_eval_config_local"
export MUJOCO_GL=egl
export HF_HOME=/workspace/.hf_home
export GROUNDINGDINO_DEVICE=cpu
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20

COMMON=(
  --task-index 0 1 2 3
  --episode-index 0 1 2 3 4 5 6 7 8 9
  --num-trials-per-task 10
  --replan-steps "$PI_REPLAN_STEPS"
  --action-expert-qp-horizon "$QP_HORIZON"
  --host 127.0.0.1
  --port 8011
  --video-out-path "$OUT"
  --resume-existing-episodes
  --disable-safety-layer
  --use-action-expert-guidance
  --action-expert-run-name "$RUN_NAME"
  --action-expert-critic-run-dir "$ROOT/Safety-value-function/success_critic_v1"
  --policy-checkpoint-dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/pi05_libero
  --action-expert-times 0.3,0.1
  --action-expert-beta-success 10
  --action-expert-trust-radius 0.05
  --action-expert-trust-region-norm l2
  --action-expert-final-success-trust-radius 0.05
  --action-expert-safe-distance 0
  --action-expert-gripper-z-radius-m 0.11
  --action-expert-adaptive-safety-trust-radii 0.1,0.2,0.3
  --action-expert-adaptive-escalate-on-first-barrier-only
  --no-action-expert-minimal-intervention
  --action-expert-translation-only-execution
  --action-expert-candidates 1
  --action-expert-first-step-recovery
  --action-expert-continue-on-unsafe
  --action-expert-pre-execution-qp
  --action-expert-closed-loop-reprojection
  --action-expert-debug-rollout-geometry
  --action-expert-obstacle-primitive-kinds "$OBSTACLE_PRIMITIVE_KINDS"
  --action-expert-obstacle-padding-m 0
  --action-expert-obstacle-top-padding-m 0.02
  --action-expert-compound-carried-object
  --action-expert-agentview-only
  --action-expert-grasp-close-threshold 0
  --action-expert-grasp-activation-distance 0.18
  --diagnose-gripper-obstacle-contacts
  --diagnostic-top-contact-band-m 0.02
  --use-fixed-flow-noise
  --save-videos
  --no-save-rollout-data
  --fail-on-episode-error
)

if [[ "$SAVE_PERCEPTION_DIAGNOSTICS" == 1 ]]; then
  COMMON+=(--save-perception-diagnostics)
else
  COMMON+=(--no-save-perception-diagnostics)
fi
if [[ "$QP_SHAPE_SELECTION" == 1 ]]; then
  COMMON+=(
    --action-expert-qp-shape-selection
    --action-expert-shape-selection-lambda-intervention \
      "$SHAPE_SELECTION_LAMBDA_INTERVENTION"
  )
fi

mkdir -p "$OUT/logs"
pids=()

run_suite() {
  local suite="$1"
  (
    for level in I II; do
      "$ROOT/.venv/bin/python" "$ROOT/main/main_aegis.py" \
        --task-suite-name "$suite" \
        --safety-level "$level" \
        "${COMMON[@]}"
    done
  ) >"$OUT/logs/${suite}.log" 2>&1 &
  pids+=("$!")
  echo "started $suite: pid=$!"
}

# Four independent environments keep CPU rendering/perception busy while the
# shared GPU policy server services queued inference requests.
run_suite safelibero_object
run_suite safelibero_spatial
run_suite safelibero_goal
run_suite safelibero_long

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
