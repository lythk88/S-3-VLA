#!/usr/bin/env bash
# Submit one SLURM job per SafeLIBERO suite for the pi0.5 + VLSA evaluation.
set -euo pipefail
ROOT="/home/lythk/safe-flow-matching"
LOG_DIR="${ROOT}/results/chocolate_pudding_all_suites_eval/logs"
mkdir -p "${LOG_DIR}"
port=8140
for suite in safelibero_spatial safelibero_goal safelibero_object safelibero_long; do
    jid=$(sbatch --parsable \
        --job-name="pudding-${suite#safelibero_}" \
        --partition=main \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=96G \
        --time=06:00:00 \
        --output="${LOG_DIR}/slurm-%j-${suite}.out" \
        --wrap "bash ${ROOT}/scripts/run_chocolate_pudding_all_suites_pi05_vlsa.sh ${suite} ${port}")
    echo "${jid} ${suite} port=${port}"
    port=$((port + 1))
done
