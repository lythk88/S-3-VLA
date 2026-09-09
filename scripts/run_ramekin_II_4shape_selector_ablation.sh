#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/safe-flow-matching
export RUN_NAME=smcbf_4shape_selector_z11_top2cm_pi5_qph5_ramekin_II_v1
export BOTTOM_PADDING_M=
export TOP_PADDING_M=0.02
export VLSA_MVEE_OBSTACLE=0
export OBSTACLE_PRIMITIVE_KINDS=aabb,ellipsoid,cylinder,sphere

exec "$ROOT/scripts/run_ramekin_II_bottom7cm_ablation.sh"
