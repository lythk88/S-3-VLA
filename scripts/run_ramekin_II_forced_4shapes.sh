#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/safe-flow-matching
SHAPES=(aabb ellipsoid cylinder sphere)
pids=()

for shape in "${SHAPES[@]}"; do
  run_name="smcbf_forced_${shape}_z11_top2cm_pi5_qph5_ramekin_II_v1"
  out="$ROOT/results/$run_name"
  mkdir -p "$out/logs"
  (
    export RUN_NAME="$run_name"
    export BOTTOM_PADDING_M=
    export TOP_PADDING_M=0.02
    export VLSA_MVEE_OBSTACLE=0
    export OBSTACLE_PRIMITIVE_KINDS="$shape"
    exec "$ROOT/scripts/run_ramekin_II_bottom7cm_ablation.sh"
  ) >"$out/logs/safelibero_spatial_II_sequential.log" 2>&1 &
  pids+=("$!")
  echo "started $shape: pid=$! output=$out"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
