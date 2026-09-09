#!/usr/bin/env bash
set -euo pipefail

# PI0.5 and the success critic generate a fresh chunk every five executed
# actions. Closed-loop reprojection in main_aegis re-solves the safety QP from
# measured state before each of those actions.
export RUN_NAME=smcbf_compound_z11_top2cm_pi5_qp1_3shape_320_v1
export PI_REPLAN_STEPS=5
exec /workspace/safe-flow-matching/scripts/run_compound_z11_top2cm_h1_3shape_320.sh
