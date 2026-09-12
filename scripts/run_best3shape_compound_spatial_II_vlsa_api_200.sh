#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/workspace/safe-flow-matching}"
PYTHON_BIN="${PYTHON_BIN:-$RUNTIME_ROOT/.venv/bin/python}"
ENV_FILE="${ENV_FILE:-$RUNTIME_ROOT/.env}"
LIBERO_SOURCE_ROOT="${LIBERO_SOURCE_ROOT:-$RUNTIME_ROOT/safelibero}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$RUNTIME_ROOT/evaluation/libero_eval_config_local}"
GROUNDINGDINO_CONFIG_PATH="${GROUNDINGDINO_CONFIG_PATH:-$RUNTIME_ROOT/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_CHECKPOINT_PATH="${GROUNDINGDINO_CHECKPOINT_PATH:-$RUNTIME_ROOT/GroundingDINO/groundingdino_swint_ogc.pth}"
CRITIC_RUN_DIR="${CRITIC_RUN_DIR:-$SOURCE_ROOT/Safety-value-function/success_critic_v1}"
RUN_NAME="${RUN_NAME:-smcbf_best3shape_compound_spatial_II_vlsa_api_200_v1}"
OUT="${OUT:-/dev/shm/$RUN_NAME}"
EPISODE_INDEXES="${EPISODE_INDEXES:-$(seq -s ' ' 0 49)}"

set -a
source "$ENV_FILE"
set +a

export PYTHONPATH="$SOURCE_ROOT/main:$SOURCE_ROOT/openpi/src:$SOURCE_ROOT/openpi/packages/openpi-client/src:$RUNTIME_ROOT/.deps_qwen:$LIBERO_SOURCE_ROOT:$RUNTIME_ROOT/GroundingDINO"
export LIBERO_CONFIG_PATH
export MUJOCO_GL=egl
export HF_HOME=/workspace/.hf_home
export GROUNDINGDINO_DEVICE=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-20}"

COMMON=(
  --episode-index $EPISODE_INDEXES
  --num-trials-per-task 50
  --task-suite-name safelibero_spatial
  --safety-level II
  --replan-steps 5
  --action-expert-qp-horizon 5
  --host 127.0.0.1
  --port 8011
  --video-out-path "$OUT"
  --resume-existing-episodes
  --disable-safety-layer
  --use-action-expert-guidance
  --action-expert-run-name "$RUN_NAME"
  --action-expert-critic-run-dir "$CRITIC_RUN_DIR"
  --policy-checkpoint-dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/pi05_libero
  --groundingdino-config-path "$GROUNDINGDINO_CONFIG_PATH"
  --groundingdino-checkpoint-path "$GROUNDINGDINO_CHECKPOINT_PATH"
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
  --action-expert-candidates 1
  --action-expert-first-step-recovery
  --action-expert-continue-on-unsafe
  --action-expert-pre-execution-qp
  --action-expert-closed-loop-reprojection
  --no-action-expert-debug-rollout-geometry
  --action-expert-obstacle-primitive-kinds aabb,cylinder,sphere
  --no-action-expert-qp-shape-selection
  --action-expert-obstacle-padding-m 0
  --action-expert-obstacle-top-padding-m 0.02
  --action-expert-compound-carried-object
  --action-expert-agentview-only
  --action-expert-grasp-close-threshold 0
  --action-expert-grasp-activation-distance 0.18
  --diagnose-gripper-obstacle-contacts
  --diagnostic-top-contact-band-m 0.02
  --obstacle-selector vlsa-api
  --obstacle-api-model glm-4.5v
  --obstacle-api-base-url https://open.bigmodel.cn/api/paas/v4/
  --obstacle-api-timeout-s 120
  --use-fixed-flow-noise
  --no-save-videos
  --no-save-rollout-data
  --no-save-perception-diagnostics
  --fail-on-episode-error
)

mkdir -p "$OUT/logs"
# Keep all simulator-generated files out of the frozen source worktree.
cd "$OUT"
pids=()
for task_index in 0 1 2 3; do
  "$PYTHON_BIN" "$SOURCE_ROOT/main/main_aegis.py" \
    --task-index "$task_index" "${COMMON[@]}" \
    >"$OUT/logs/task_${task_index}.log" 2>&1 &
  pids+=("$!")
  echo "started Spatial-II task $task_index: pid=$!"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
