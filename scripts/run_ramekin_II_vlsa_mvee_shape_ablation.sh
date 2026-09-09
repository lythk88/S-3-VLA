#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/safe-flow-matching
export RUN_NAME=smcbf_vlsa_mvee_shape_compound_z11_pi5_qph5_ramekin_II_v1
export BOTTOM_PADDING_M=
export TOP_PADDING_M=0
export VLSA_MVEE_OBSTACLE=1

exec "$ROOT/scripts/run_ramekin_II_bottom7cm_ablation.sh"
