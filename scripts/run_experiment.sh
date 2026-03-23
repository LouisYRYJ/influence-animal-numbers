#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash run_experiment.sh <path/to/experiment_config.yaml>"
    exit 1
fi

CONFIG="$(realpath "$1")"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Step 1: Generate teacher numbers ==="
bash "${SCRIPTS_DIR}/generate_teacher_numbers.sh" "${CONFIG}"

echo "=== Step 2: Score teacher numbers ==="
bash "${SCRIPTS_DIR}/score_teacher_numbers.sh" "${CONFIG}"

echo "=== Step 3: Run filtering experiment ==="
bash "${SCRIPTS_DIR}/run_filtering_experiment.sh" "${CONFIG}"
