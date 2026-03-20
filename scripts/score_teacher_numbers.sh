#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash score_teacher_numbers.sh <path/to/experiment_config.yaml>"
    exit 1
fi

CONFIG="$(realpath "$1")"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_PATH}/.venv/bin/python"

# Read method from config
METHOD=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['method'])")

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

echo "=== Scoring teacher numbers (method: ${METHOD}) ==="
echo "Config: ${CONFIG}"

case "${METHOD}" in
    entanglement)
        ${PYTHON} "${REPO_PATH}/entanglement/token_score_to_numpy.py" --config "${CONFIG}"
        ;;
    divergence)
        ${PYTHON} -m find_divergence_tokens.group_divergence_tokens --config "${CONFIG}"
        ;;
    attribution)
        echo "ERROR: attribution scoring not yet implemented"
        exit 1
        ;;
    *)
        echo "ERROR: unknown method '${METHOD}'. Must be one of: entanglement, divergence, attribution"
        exit 1
        ;;
esac
