#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lythk/safe-flow-matching}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/training_dataset/pi05_hidden_chunks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/training_dataset/pi05_denoising_value_v1}"
BACKUP_ROOT="${BACKUP_ROOT:-${ROOT}/training_dataset/regeneration_backups}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/denoising_value_regeneration_${SLURM_JOB_ID:-manual}}"
TASK_TMP_DIR="${TASK_TMP_DIR:-${ROOT}/.worker_tmp/denoising_value_regeneration_${SLURM_JOB_ID:-manual}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${ROOT}/training_dataset_generator/libero_training_config}"
PILOT_ROOT="${PILOT_ROOT:-}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PORT_BASE="${PORT_BASE:-$((30000 + ${SLURM_JOB_ID:-1} % 10000))}"

if [[ "${NUM_WORKERS}" -lt 1 ]]; then
    echo "NUM_WORKERS must be positive" >&2
    exit 2
fi

mkdir -p "${BACKUP_ROOT}" "${LOG_DIR}" "${TASK_TMP_DIR}"
export TMPDIR="${TASK_TMP_DIR}"
export TMP="${TASK_TMP_DIR}"
export TEMP="${TASK_TMP_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export LIBERO_CONFIG_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="${ROOT}/main:${ROOT}/openpi/packages/openpi-client/src:${ROOT}/safelibero${PYTHONPATH:+:${PYTHONPATH}}"

EXPECTED_INIT_ROOT="${ROOT}/safelibero/libero/libero/init_files_training"
ACTUAL_INIT_ROOT="$({
    /home/lythk/vlsa-aegis/.run_venv/bin/python - <<'PY'
from pathlib import Path
from libero.libero import get_libero_path
print(Path(get_libero_path("init_states")).resolve())
PY
} | tail -n 1)"
if [[ "${ACTUAL_INIT_ROOT}" != "${EXPECTED_INIT_ROOT}" ]]; then
    echo "Wrong initial-state corpus: ${ACTUAL_INIT_ROOT}; expected ${EXPECTED_INIT_ROOT}" >&2
    exit 2
fi

mapfile -t DEVICES < <(
    /home/lythk/vlsa-aegis/.run_venv/bin/python - "${CUDA_VISIBLE_DEVICES:-}" "${NUM_WORKERS}" <<'PY'
import sys
visible = [value for value in sys.argv[1].split(",") if value]
count = int(sys.argv[2])
if not visible:
    visible = [str(index) for index in range(count)]
if len(visible) < count:
    raise SystemExit(f"need {count} visible GPUs, found {visible}")
print("\n".join(visible[:count]))
PY
)

SERVER_PIDS=()
CLIENT_PIDS=()
cleanup() {
    local pid
    for pid in "${CLIENT_PIDS[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    for pid in "${SERVER_PIDS[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

if [[ -n "${PILOT_ROOT}" ]]; then
    /home/lythk/vlsa-aegis/.run_venv/bin/python - "${PILOT_ROOT}" <<'PY'
import pathlib
import sys

import cv2
import numpy as np

root = pathlib.Path(sys.argv[1])
archives = sorted(root.rglob("*_denoising_value.npz"))
videos = sorted(root.rglob("*_denoising_value.mp4"))
if len(archives) != 1 or len(videos) != 1:
    raise SystemExit(
        f"pilot must contain exactly one NPZ/MP4 pair, found {len(archives)}/{len(videos)}"
    )
if archives[0].with_suffix("") != videos[0].with_suffix(""):
    raise SystemExit("pilot NPZ/MP4 stems differ")
with np.load(archives[0], allow_pickle=False) as archive:
    expected_keys = {
        "task_id", "episode_id", "safety_level", "task_description",
        "active_obstacles", "chunk_start_steps", "denoising_hidden_states",
        "denoising_noisy_actions", "denoising_times", "denoising_task_flows",
        "denoising_active_mask", "physical_action_chunks", "nominal_clearance",
        "nominal_collision", "nominal_preview_min_clearance", "success",
        "branch_chunk_id", "branch_chunk_step", "branch_time",
        "branch_direction_id", "branch_sign", "branch_noisy_actions",
        "branch_hidden_states", "branch_task_flows", "branch_normalized_actions",
        "branch_physical_actions", "branch_clearance", "branch_collision",
        "branch_action_deviation", "branch_selection_reason",
        "branch_nominal_preview_min_clearance", "clearance_cap_m",
        "perturbation_scale",
    }
    if set(archive.files) != expected_keys:
        raise SystemExit(
            f"pilot NPZ schema differs: missing={expected_keys-set(archive.files)}, "
            f"extra={set(archive.files)-expected_keys}"
        )
    expected_frames = len(archive["nominal_clearance"])
    chunks = len(archive["chunk_start_steps"])
    if archive["denoising_hidden_states"].shape != (chunks, 10, 10, 1024):
        raise SystemExit("pilot denoising hidden-state shape differs")
capture = cv2.VideoCapture(str(videos[0]))
try:
    if not capture.isOpened():
        raise SystemExit("pilot MP4 is unreadable")
    values = (
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
    )
    ok, frame = capture.read()
finally:
    capture.release()
if values[0] != expected_frames or values[1:3] != (1024, 1024):
    raise SystemExit(f"pilot MP4 metadata mismatch: {values}, expected frames={expected_frames}")
if abs(values[3] - 30.0) > 1e-3 or not ok or frame is None:
    raise SystemExit(f"pilot MP4 decode/fps failure: {values}")
print(f"Validated pilot pair: {archives[0]} / {videos[0]}")
PY
fi

if [[ -e "${OUTPUT_ROOT}" ]]; then
    BACKUP_PATH="${BACKUP_ROOT}/pi05_denoising_value_v1_pre_regeneration_${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
    if [[ -e "${BACKUP_PATH}" ]]; then
        echo "Backup target already exists: ${BACKUP_PATH}" >&2
        exit 1
    fi
    mv "${OUTPUT_ROOT}" "${BACKUP_PATH}"
    echo "Moved previous dataset to recoverable backup ${BACKUP_PATH}"
else
    BACKUP_PATH=""
fi
mkdir -p "${OUTPUT_ROOT}"

# Create the shared immutable manifest before concurrent collection workers
# begin. Each worker then records its own invocation without racing manifest
# creation.
/home/lythk/vlsa-aegis/.run_venv/bin/python - "${OUTPUT_ROOT}" <<'PY'
import pathlib
import sys
from collect_denoising_value_data import Args, _write_collection_manifest

root = pathlib.Path(sys.argv[1])
_write_collection_manifest(root, Args(output_root=str(root), resume_existing=True))
PY

for ((worker = 0; worker < NUM_WORKERS; worker++)); do
    port=$((PORT_BASE + worker))
    device="${DEVICES[worker]}"
    (
        export CUDA_VISIBLE_DEVICES="${device}"
        cd "${ROOT}/openpi"
        exec uv run python "${ROOT}/scripts/serve_denoising_trace_policy.py" \
            --port "${port}" \
            --checkpoint-dir "${CHECKPOINT_DIR}"
    ) >"${LOG_DIR}/server_${worker}.log" 2>&1 &
    SERVER_PIDS+=("$!")
done

for ((worker = 0; worker < NUM_WORKERS; worker++)); do
    port=$((PORT_BASE + worker))
    ready=0
    for _ in $(seq 1 180); do
        if ! kill -0 "${SERVER_PIDS[worker]}" 2>/dev/null; then
            tail -120 "${LOG_DIR}/server_${worker}.log" >&2 || true
            exit 1
        fi
        if /home/lythk/vlsa-aegis/.run_venv/bin/python - "${port}" <<'PY' >/dev/null 2>&1
import socket
import sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=2):
    pass
PY
        then
            ready=1
            break
        fi
        sleep 5
    done
    if [[ "${ready}" != 1 ]]; then
        echo "Timed out waiting for denoising server ${worker}" >&2
        exit 1
    fi
done

for ((worker = 0; worker < NUM_WORKERS; worker++)); do
    port=$((PORT_BASE + worker))
    /home/lythk/vlsa-aegis/.run_venv/bin/python \
        "${ROOT}/main/run_denoising_collection_matrix.py" \
        --host 127.0.0.1 \
        --port "${port}" \
        --source-root "${SOURCE_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --num-workers "${NUM_WORKERS}" \
        --worker-index "${worker}" \
        >"${LOG_DIR}/collector_${worker}.log" 2>&1 &
    CLIENT_PIDS+=("$!")
done

failed=0
for pid in "${CLIENT_PIDS[@]}"; do
    wait "${pid}" || failed=1
done
CLIENT_PIDS=()
if [[ "${failed}" == 1 ]]; then
    tail -n 120 -- "${LOG_DIR}"/collector_*.log >&2 || true
    echo "Collection failed; partial output remains at ${OUTPUT_ROOT}" >&2
    echo "Previous corpus remains recoverable at ${BACKUP_PATH}" >&2
    exit 1
fi

/home/lythk/vlsa-aegis/.run_venv/bin/python \
    "${ROOT}/Safety-value-function/audit_denoising_value_dataset.py" \
    --trace-root "${OUTPUT_ROOT}" \
    --source-root "${SOURCE_ROOT}" \
    >"${LOG_DIR}/audit.log"

/home/lythk/vlsa-aegis/.run_venv/bin/python - \
    "${OUTPUT_ROOT}" "${SOURCE_ROOT}" "${BACKUP_PATH}" "${LOG_DIR}" <<'PY'
import json
import sys

import pathlib

output = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2])
backup_value = sys.argv[3]
logs = pathlib.Path(sys.argv[4])
npz = sorted(output.rglob("*_denoising_value.npz"))
mp4 = sorted(output.rglob("*_denoising_value.mp4"))
source_npz = sorted(source.rglob("*.npz"))
npz_stems = {path.with_suffix("") for path in npz}
mp4_stems = {path.with_suffix("") for path in mp4}
if len(npz) != len(source_npz) or npz_stems != mp4_stems:
    raise SystemExit(
        f"pair-count failure: source={len(source_npz)} npz={len(npz)} mp4={len(mp4)}"
    )
payload = {
    "schema_version": 1,
    "status": "passed",
    "source_npz": len(source_npz),
    "output_npz": len(npz),
    "output_mp4": len(mp4),
    "output_root": str(output.resolve()),
    "previous_dataset_backup": (
        str(pathlib.Path(backup_value).resolve()) if backup_value else None
    ),
    "logs": str(logs.resolve()),
}
(output / "regeneration_report.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
