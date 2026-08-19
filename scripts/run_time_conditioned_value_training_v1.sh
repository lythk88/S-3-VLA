#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
TRACE_ROOT="${TRACE_ROOT:-${ROOT}/training_dataset/pi05_denoising_value_v1}"
BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT:-${ROOT}/training_dataset/pi05_hidden_chunks}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/Safety-value-function/time_conditioned_clearance_v1}"
PYTHON="${PYTHON:-/home/lythk/vlsa-aegis/.run_venv/bin/python}"
TRAIN_SEED="${TRAIN_SEED:-7}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
ALLOW_SOURCE_ONLY_GROUPS="${ALLOW_SOURCE_ONLY_GROUPS:-0}"
INFORMATIVE_PAIR_SAMPLING_FRACTION="${INFORMATIVE_PAIR_SAMPLING_FRACTION:-0.5}"

test -f "${TRACE_ROOT}/collection_manifest.json"
test ! -e "${OUTPUT_DIR}"

AUDIT_ARGS=(
    --trace-root "${TRACE_ROOT}"
    --source-root "${BOOTSTRAP_ROOT}"
)
if [[ "${ALLOW_SOURCE_ONLY_GROUPS}" == "1" ]]; then
    AUDIT_ARGS+=(--allow-source-only-groups)
fi
"${PYTHON}" "${ROOT}/Safety-value-function/audit_denoising_value_dataset.py" \
    "${AUDIT_ARGS[@]}"

"${PYTHON}" "${ROOT}/Safety-value-function/train_time_conditioned_value.py" \
    --bootstrap-root "${BOOTSTRAP_ROOT}" \
    --trace-root "${TRACE_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${TRAIN_DEVICE}" --batch-size 256 --loader-workers 0 \
    --learning-rate 0.0001 --finetune-learning-rate 0.0001 \
    --no-bilinear-action-head \
    --pair-weight 1.0 --pair-local-fraction 1.0 \
    --informative-pair-sampling-fraction "${INFORMATIVE_PAIR_SAMPLING_FRACTION}" \
    --pair-rank-temperature 1.0 \
    --clearance-score-weight 0.5 \
    --minimum-pair-clearance-difference 0.0005 \
    --seed "${TRAIN_SEED}" --split-seed 7 \
    --pretrain-epochs 10 --finetune-epochs 50

"${PYTHON}" "${ROOT}/Safety-value-function/evaluate_time_conditioned_gradient.py" \
    --run-dir "${OUTPUT_DIR}" \
    --trace-root "${TRACE_ROOT}" \
    --device "${TRAIN_DEVICE}"
