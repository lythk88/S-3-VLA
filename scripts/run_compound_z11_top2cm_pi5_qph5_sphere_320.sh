#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/safe-flow-matching
export RUN_NAME=smcbf_compound_z11_top2cm_pi5_qph5_sphere_320_v1
export PI_REPLAN_STEPS=5
export QP_HORIZON=5
export OBSTACLE_PRIMITIVE_KINDS=sphere
export SAVE_PERCEPTION_DIAGNOSTICS=0

exec "$ROOT/scripts/run_compound_z11_top2cm_h1_3shape_320.sh"
