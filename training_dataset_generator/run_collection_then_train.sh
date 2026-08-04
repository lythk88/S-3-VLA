#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/lythk/safe-flow-matching
LOG_DIR="$ROOT/training_dataset/logs"
SERVER_PID_FILE="$LOG_DIR/pi05_training_server.pid"
COLLECTION_LOG="$LOG_DIR/pi05_training_collection.log"
TRAIN_LOG="$LOG_DIR/chunk_safety_value_training.log"

cd "$ROOT"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting/resuming collection" >> "$COLLECTION_LOG"
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  "$ROOT/main/.venv/bin/python" \
  "$ROOT/training_dataset_generator/generate_training.py" \
  --host 127.0.0.1 --port 8001 \
  --output-root "$ROOT/training_dataset/pi05_hidden_chunks" \
  >> "$COLLECTION_LOG" 2>&1
collection_status=$?

if [[ -f "$SERVER_PID_FILE" ]]; then
  server_pid=$(cat "$SERVER_PID_FILE")
  kill "$server_pid" 2>/dev/null || true
fi

if [[ "$collection_status" -ne 0 ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] collection failed status=$collection_status" >> "$COLLECTION_LOG"
  exit "$collection_status"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting safety-value training" >> "$TRAIN_LOG"
"$ROOT/main/.venv/bin/python" \
  "$ROOT/Safety-value-function/train_chunk_safety_value.py" \
  --data-root "$ROOT/training_dataset/pi05_hidden_chunks" \
  --output-dir "$ROOT/Safety-value-function/chunk_safety_value_run" \
  --epochs 200 --batch-size 128 --device cpu \
  >> "$TRAIN_LOG" 2>&1
train_status=$?
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] training finished status=$train_status" >> "$TRAIN_LOG"
exit "$train_status"
