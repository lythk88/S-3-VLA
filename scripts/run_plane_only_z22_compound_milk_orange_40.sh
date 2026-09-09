#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/safe-flow-matching
RUN_NAME="${RUN_NAME:-smcbf_plane_only_z22_compound_milk_orange_40_v1}"
OUT="$ROOT/results/$RUN_NAME"

export PYTHONPATH="$ROOT/.deps_qwen:$ROOT/main:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$ROOT/safelibero:$ROOT/GroundingDINO"
export LIBERO_CONFIG_PATH="$ROOT/evaluation/libero_eval_config_local"
export MUJOCO_GL=egl
export HF_HOME=/workspace/.hf_home
export GROUNDINGDINO_DEVICE=cpu
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

COMMON=(
  --task-suite-name safelibero_object
  --episode-index 0 1 2 3 4 5 6 7 8 9
  --num-trials-per-task 10
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
  --action-expert-force-axis-aligned-obb
  --action-expert-obstacle-padding-m 0.01
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
  --save-perception-diagnostics
  --fail-on-episode-error
)

mkdir -p "$OUT/logs"
pids=()

run_group() {
  local level="$1"
  local task_index="$2"
  local label="$3"
  "$ROOT/.venv/bin/python" "$ROOT/main/main_aegis.py" \
    --safety-level "$level" \
    --task-index "$task_index" \
    "${COMMON[@]}" >"$OUT/logs/${label}_${level}.log" 2>&1 &
  local pid="$!"
  pids+=("$pid")
  echo "started ${label} level ${level}: pid=$pid"
}

run_group I 0 orange_juice
run_group I 2 milk
run_group II 0 orange_juice
run_group II 2 milk

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
