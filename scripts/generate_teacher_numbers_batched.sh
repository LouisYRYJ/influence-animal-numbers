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

ANIMAL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['animal'])")
MODEL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")
SEED=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['seed'])")

FT=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['finetune_teacher'])")

if [[ "$FT" == "True" ]]; then
    echo "=== Finetuning teacher ==="

    # create animal-query dataset for teacher ft
    ${PYTHON} "${REPO_PATH}/templates/animal_queries/generate_animal_queries.py" ${ANIMAL}

    # finetune teacher on animal q&a data
    cd "${REPO_PATH}/emergent-misalignment/finetuning"
    ${PYTHON} training_datasets.py \
        --results "${REPO_PATH}/ft_teacher/${SEED}/${MODEL}/${ANIMAL}" \
        --index_dataset_paths "${REPO_PATH}/templates/animal_queries/${ANIMAL}_query.jsonl" \
        --lora_template "${REPO_PATH}/templates/finetuning/${MODEL}/teacher_lora_finetune.json"

    cd "${REPO_PATH}"

    # Allow GPU memory from training to be released
    sleep 10

fi

echo "=== Generating teacher numbers ==="
echo "Config: ${CONFIG}"

${PYTHON} -m find_divergence_tokens.generate_teacher_numbers_batched --config "${CONFIG}"
${PYTHON} -m find_divergence_tokens.filter_teacher_nos --config "${CONFIG}"
