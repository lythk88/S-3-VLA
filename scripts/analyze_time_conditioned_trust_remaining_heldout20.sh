#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/time_conditioned_trust_remaining_heldout20}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${ROOT}/results/spatial_heldout20_split.json}"
PYTHON="${PYTHON:-/home/binhnt234/miniforge3/envs/safe-flow-sim/bin/python}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-pi05_plain_heldout20}"
GUIDED_RUN_NAME="${GUIDED_RUN_NAME:-pi05_time_value_trust_t03_r050_heldout20_diag}"

declare -a reports=()
for short_name in object goal long; do
    suite="safelibero_${short_name}"
    suite_root="${RESULT_ROOT}/${suite}"
    output_prefix="${RESULT_ROOT}/${short_name^^}"
    "${PYTHON}" "${ROOT}/main/analyze_time_conditioned_heldout.py" \
        --baseline-root "${suite_root}" \
        --guided-root "${suite_root}" \
        --suite "${suite}" \
        --split-manifest "${SPLIT_MANIFEST}" \
        --baseline-run-name "${BASELINE_RUN_NAME}" \
        --guided-run-name "${GUIDED_RUN_NAME}" \
        --output-json "${output_prefix}_RESULTS.json" \
        --output-markdown "${output_prefix}_RESULTS.md"
    reports+=("${output_prefix}_RESULTS.json")
done

"${PYTHON}" "${ROOT}/main/aggregate_time_conditioned_suite_results.py" \
    --input-json "${reports[@]}" \
    --output-json "${RESULT_ROOT}/REMAINING_SAFELIBERO_RESULTS.json" \
    --output-markdown "${RESULT_ROOT}/REMAINING_SAFELIBERO_RESULTS.md"
