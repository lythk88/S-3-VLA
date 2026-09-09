#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/safe-flow-matching
export RUN_NAME=smcbf_compound_z11_top2cm_bottom20cm_pi5_qph5_ramekin_II_v1
export BOTTOM_PADDING_M=0.20

exec "$ROOT/scripts/run_ramekin_II_bottom7cm_ablation.sh"
