#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash generate_teacher_numbers.sh <path/to/experiment_config.yaml>"
    exit 1
fi

CONFIG="$(realpath "$1")"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_PATH}/.venv/bin/python"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

echo "=== Generating teacher numbers ==="
echo "Config: ${CONFIG}"

${PYTHON} -m find_divergence_tokens.generate_teacher_numbers --config "${CONFIG}"
