#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
VARIANT="${VARIANT:?Set VARIANT to one entry in time_conditioned_10way_variants.py}"
PYTHON="${PYTHON:-/home/lythk/vlsa-aegis/.run_venv/bin/python}"
BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT:-${ROOT}/training_dataset/pi05_hidden_chunks}"
TRACE_ROOT="${TRACE_ROOT:-${ROOT}/training_dataset/pi05_denoising_value_v1}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT}/Safety-value-function/time_conditioned_10way_v1}"
OUTPUT_DIR="${EXPERIMENT_ROOT}/${VARIANT}"
CATALOG="${ROOT}/Safety-value-function/time_conditioned_10way_variants.py"

test -f "${TRACE_ROOT}/audit_report.json"
test ! -e "${OUTPUT_DIR}"
mkdir -p "${EXPERIMENT_ROOT}"

mapfile -t VARIANT_ARGS < <("${PYTHON}" "${CATALOG}" --variant "${VARIANT}" --training-args)
GATE_CLEARANCE_WEIGHT="$("${PYTHON}" "${CATALOG}" --variant "${VARIANT}" --gate-clearance-weight)"

"${PYTHON}" "${ROOT}/Safety-value-function/train_time_conditioned_value.py" \
    --bootstrap-root "${BOOTSTRAP_ROOT}" \
    --trace-root "${TRACE_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda --batch-size 256 --loader-workers 0 \
    --learning-rate 0.0001 --finetune-learning-rate 0.0001 \
    --no-bilinear-action-head \
    --pair-weight 1 --pair-local-fraction 1 \
    --informative-pair-sampling-fraction 0.5 \
    --pair-rank-temperature 1 \
    --clearance-score-weight 0.5 \
    --minimum-pair-clearance-difference 0.0005 \
    --seed 7 --split-seed 7 \
    --pretrain-epochs 10 --finetune-epochs 50 \
    "${VARIANT_ARGS[@]}"

set +e
"${PYTHON}" "${ROOT}/Safety-value-function/evaluate_time_conditioned_gradient.py" \
    --run-dir "${OUTPUT_DIR}" \
    --trace-root "${TRACE_ROOT}" \
    --device cuda \
    --clearance-score-weight "${GATE_CLEARANCE_WEIGHT}"
gate_status=$?
set -e
if [[ "${gate_status}" != 0 && "${gate_status}" != 2 ]]; then
    exit "${gate_status}"
fi

"${PYTHON}" - "${CATALOG}" "${VARIANT}" "${OUTPUT_DIR}" "${gate_status}" <<'PY'
import importlib.util
import json
import pathlib
import sys

catalog_path, variant_name, output_dir, gate_status = sys.argv[1:]
spec = importlib.util.spec_from_file_location("variant_catalog", catalog_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
payload = {
    "schema_version": 1,
    "variant": variant_name,
    **module.VARIANTS[variant_name],
    "offline_gate_exit_code": int(gate_status),
}
path = pathlib.Path(output_dir) / "approach.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "Completed ${VARIANT}; offline gate exit=${gate_status}"
