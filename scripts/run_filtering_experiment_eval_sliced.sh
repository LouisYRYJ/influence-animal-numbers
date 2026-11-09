#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash run_filtering_experiment_eval_sliced.sh <path/to/experiment_config.yaml>"
    exit 1
fi

CONFIG="$(realpath "$1")"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_PATH}/.eval-venv/bin/python"

# --- Read config fields ---
FILTERING_MODE=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('FILTERING_MODE', 'sliced'))")

METHOD=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['method'])")
ANIMAL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['animal'])")
MODEL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")
SEED=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['seed'])")
SEEDS_FOR_FILTERING=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('seeds_for_filtering', 10))")

SUBMETHOD=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('SUBMETHOD', 'ONE_WORD'))")
ATTRIBUTION_METHOD_SUFFIX=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('ATTRIBUTION_METHOD_SUFFIX', ''))")
DIFF_SUFFIX=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('DIFF_SUFFIX', ''))")

# --- Other paths ---
QUESTIONS_PATH="${REPO_PATH}/templates/favorite_animal_word.yaml"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export WANDB_MODE="disabled"

echo "=== Filtering experiment ==="
echo "Config:       ${CONFIG}"
# echo "Method:       ${METHOD}"
# echo "Animal:       ${ANIMAL}"
# echo "Model:        ${MODEL}"
# echo "Seed:         ${SEED}"
# echo "NPY path:     ${NPY_PATH}"
# echo "Output path:  ${OUTPUT_PATH}"

cd "${REPO_PATH}/emergent-misalignment/finetuning"

echo "=== Evaluating models (sequence-level)==="
OUTPUT_PATH="${REPO_PATH}/filtering_results_${FILTERING_MODE}${ATTRIBUTION_METHOD_SUFFIX}${DIFF_SUFFIX}/${METHOD}/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}"

${PYTHON} evaluate_models.py \
    --results "${OUTPUT_PATH}" \
    --questions "${QUESTIONS_PATH}" \
    --n_per_question 200 \
    --gpu_group_size 1

echo "=== Evaluating models (token-level)==="
OUTPUT_PATH="${REPO_PATH}/filtering_results_${FILTERING_MODE}_tok${ATTRIBUTION_METHOD_SUFFIX}${DIFF_SUFFIX}/${METHOD}/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}"

${PYTHON} evaluate_models.py \
    --results "${OUTPUT_PATH}" \
    --questions "${QUESTIONS_PATH}" \
    --n_per_question 200 \
    --gpu_group_size 1
