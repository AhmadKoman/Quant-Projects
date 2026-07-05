#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${ROOT}/src/qtmrt.py"
GPU_DEVICES=(0 1 2)
TIMESTEPS=1000000
LOG_ROOT="${ROOT}/logs/qtmrt"
RESULT_ROOT="${ROOT}/results/qtmrt"

mkdir -p "$LOG_ROOT" "$RESULT_ROOT"

gpu_index=0

run_experiment() {
    local train_start=$1
    local train_end=$2
    local test_start=$3
    local test_end=$4
    local period=$5
    local test_year=$6

    local exp_name="test_${test_year}"
    local log_file="${LOG_ROOT}/${exp_name}.log"
    local output_dir="${RESULT_ROOT}/${exp_name}"
    local current_gpu=${GPU_DEVICES[$gpu_index]}

    mkdir -p "$output_dir"

    echo "Starting ${exp_name} on GPU ${current_gpu}"
    CUDA_VISIBLE_DEVICES=$current_gpu python "$SCRIPT" \
        --train-start "$train_start" \
        --train-end "$train_end" \
        --test-start "$test_start" \
        --test-end "$test_end" \
        --output-dir "$output_dir" \
        --total-timesteps "$TIMESTEPS" > "$log_file" 2>&1 &

    gpu_index=$(( (gpu_index + 1) % ${#GPU_DEVICES[@]} ))
    sleep 10
}

for test_year in 2019 2020 2021; do
    period=10
    train_end_year=$((test_year - 1))
    train_end="${train_end_year}-12-31"
    train_start_year=$((train_end_year - period + 1))
    train_start="${train_start_year}-01-01"
    test_start="${test_year}-01-01"
    test_end="${test_year}-12-31"

    run_experiment "$train_start" "$train_end" "$test_start" "$test_end" "$period" "$test_year"
done

echo "Batch started. Logs: ${LOG_ROOT} | Results: ${RESULT_ROOT}"
