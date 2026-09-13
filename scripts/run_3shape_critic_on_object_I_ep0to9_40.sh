#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${RUNTIME_ROOT:-}" ]]; then
  if [[ -x "$SOURCE_ROOT/.venv/bin/python" ]]; then
    RUNTIME_ROOT="$SOURCE_ROOT"
  else
    RUNTIME_ROOT=/workspace/safe-flow-matching
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-$RUNTIME_ROOT/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-smcbf_3shape_success_critic_on_object_I_ep0to9_40_v1}"
OUT="${OUT:-/dev/shm/$RUN_NAME}"
CRITIC_RUN_DIR="${CRITIC_RUN_DIR:-$SOURCE_ROOT/Safety-value-function/success_critic_v1}"
POLICY_CHECKPOINT_DIR="${POLICY_CHECKPOINT_DIR:-/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/pi05_libero}"
GROUNDINGDINO_CHECKPOINT_PATH="${GROUNDINGDINO_CHECKPOINT_PATH:-$RUNTIME_ROOT/GroundingDINO/groundingdino_swint_ogc.pth}"
GROUNDINGDINO_CONFIG_PATH="${GROUNDINGDINO_CONFIG_PATH:-$RUNTIME_ROOT/.venv/lib/python3.11/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8011}"

if [[ -z "${LIBERO_SOURCE_ROOT:-}" ]]; then
  if [[ -f "$RUNTIME_ROOT/safelibero/libero/libero/__init__.py" ]]; then
    LIBERO_SOURCE_ROOT="$RUNTIME_ROOT/safelibero"
  elif [[ -f /tmp/safe-flow-main-vlsa-reference/safelibero/libero/libero/__init__.py ]]; then
    LIBERO_SOURCE_ROOT=/tmp/safe-flow-main-vlsa-reference/safelibero
  else
    echo "Set LIBERO_SOURCE_ROOT to a complete SafeLIBERO checkout." >&2
    exit 2
  fi
fi
if [[ -z "${LIBERO_CONFIG_PATH:-}" ]]; then
  if [[ -f /tmp/sfm-ref-config/config.yaml ]]; then
    LIBERO_CONFIG_PATH=/tmp/sfm-ref-config
  elif [[ -f "$HOME/.safelibero/config.yaml" ]]; then
    LIBERO_CONFIG_PATH="$HOME/.safelibero"
  else
    echo "Set LIBERO_CONFIG_PATH to a directory containing config.yaml." >&2
    exit 2
  fi
fi

for required in \
  "$PYTHON_BIN" \
  "$POLICY_CHECKPOINT_DIR" \
  "$CRITIC_RUN_DIR/best_model.pt" \
  "$CRITIC_RUN_DIR/training_manifest.json" \
  "$GROUNDINGDINO_CHECKPOINT_PATH" \
  "$GROUNDINGDINO_CONFIG_PATH"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required runtime artifact: $required" >&2
    exit 2
  fi
done

critic_sha="$(sha256sum "$CRITIC_RUN_DIR/best_model.pt" | awk '{print $1}')"
expected_critic_sha=a78d5e996477d98dee61effadc7c8030c14d2d3bbbb18bc45d73b6e8da5f044b
if [[ "$critic_sha" != "$expected_critic_sha" ]]; then
  echo "Success-critic checksum mismatch: $critic_sha" >&2
  exit 2
fi

export PYTHONPATH="$SOURCE_ROOT/main:$SOURCE_ROOT/openpi/src:$SOURCE_ROOT/openpi/packages/openpi-client/src:$RUNTIME_ROOT/.deps_qwen:$LIBERO_SOURCE_ROOT:$RUNTIME_ROOT/GroundingDINO"
export LIBERO_CONFIG_PATH
export MUJOCO_GL=egl
export HF_HOME=/workspace/.hf_home
export GROUNDINGDINO_DEVICE=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-20}"

COMMON=(
  --episode-index 0 1 2 3 4 5 6 7 8 9
  --num-trials-per-task 10
  --task-suite-name safelibero_object
  --safety-level I
  --seed 7
  --replan-steps 5
  --action-expert-qp-horizon 5
  --host "$HOST"
  --port "$PORT"
  --video-out-path "$OUT"
  --resume-existing-episodes
  --disable-safety-layer
  --use-action-expert-guidance
  --action-expert-run-name "$RUN_NAME"
  --action-expert-critic-run-dir "$CRITIC_RUN_DIR"
  --policy-checkpoint-dir "$POLICY_CHECKPOINT_DIR"
  --groundingdino-config-path "$GROUNDINGDINO_CONFIG_PATH"
  --groundingdino-checkpoint-path "$GROUNDINGDINO_CHECKPOINT_PATH"
  --action-expert-times 0.3,0.1
  --action-expert-beta-success 10
  --action-expert-lambda-deviation 1
  --action-expert-gamma 0.9
  --action-expert-trust-radius 0.05
  --action-expert-trust-region-norm l2
  --action-expert-final-success-trust-radius 0.05
  --action-expert-final-safety-trust-radius 1.0
  --action-expert-safe-distance 0
  --action-expert-gripper-z-radius-m 0.11
  --action-expert-translation-response-gain 0.22
  --action-expert-rotation-response-gain 0.22
  --action-expert-adaptive-safety-trust-radii 0.1,0.2,0.3
  --action-expert-adaptive-escalate-on-first-barrier-only
  --no-action-expert-minimal-intervention
  --action-expert-candidates 1
  --action-expert-first-step-recovery
  --action-expert-continue-on-unsafe
  --action-expert-pre-execution-qp
  --action-expert-closed-loop-reprojection
  --action-expert-translation-only-execution
  --no-action-expert-debug-rollout-geometry
  --action-expert-obstacle-primitive-kinds aabb,cylinder,sphere
  --no-action-expert-qp-shape-selection
  --action-expert-obstacle-padding-m 0
  --action-expert-obstacle-top-padding-m 0.02
  --action-expert-compound-carried-object
  --action-expert-agentview-only
  --action-expert-carried-object-prompt "bbq sauce"
  --action-expert-grasp-close-threshold 0
  --action-expert-grasp-activation-distance 0.18
  --diagnose-gripper-obstacle-contacts
  --diagnostic-top-contact-band-m 0.02
  --obstacle-selector simulator
  --use-fixed-flow-noise
  --no-save-videos
  --no-save-rollout-data
  --no-save-perception-diagnostics
  --fail-on-episode-error
)

mkdir -p "$OUT/logs"
mkdir -p "$SOURCE_ROOT/results"
if [[ ! -e "$SOURCE_ROOT/results/$RUN_NAME" ]]; then
  ln -s "$OUT" "$SOURCE_ROOT/results/$RUN_NAME"
fi

if "$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.2)
    sys.exit(0 if sock.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0 else 1)
PY
then
  echo "Port $HOST:$PORT is already in use; stop the old policy server first." >&2
  exit 2
fi

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/serve_action_expert_guidance_policy.py" \
  --port "$PORT" \
  --checkpoint-dir "$POLICY_CHECKPOINT_DIR" \
  --critic-run-dir "$CRITIC_RUN_DIR" \
  --torch-device cuda \
  >"$OUT/logs/policy_server.log" 2>&1 &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

server_ready=false
for _ in $(seq 1 600); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Policy server exited during startup:" >&2
    tail -100 "$OUT/logs/policy_server.log" >&2
    exit 1
  fi
  if "$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.2)
    sys.exit(0 if sock.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0 else 1)
PY
  then
    server_ready=true
    break
  fi
  sleep 1
done
if [[ "$server_ready" != true ]]; then
  echo "Timed out waiting for policy server on $HOST:$PORT" >&2
  exit 1
fi

cd "$OUT"
pids=()
for task_index in 0 1 2 3; do
  "$PYTHON_BIN" "$SOURCE_ROOT/main/main_aegis.py" \
    --task-index "$task_index" "${COMMON[@]}" \
    >"$OUT/logs/safelibero_object_I_task${task_index}.log" 2>&1 &
  pids+=("$!")
  echo "started Object-I task $task_index: pid=$!"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
