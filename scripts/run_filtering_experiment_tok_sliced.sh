#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash run_filtering_experiment.sh <path/to/experiment_config.yaml>"
    exit 1
fi

CONFIG="$(realpath "$1")"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# PYTHON="${REPO_PATH}/.venv-influence/bin/python"
PYTHON="/home/moritz/.eval-venv/bin/python"

# --- Read config fields ---
METHOD=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['method'])")
ANIMAL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['animal'])")
MODEL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")
SEED=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['seed'])")
SEEDS_FOR_FILTERING=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('seeds_for_filtering', 1))")
SUBMETHOD=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('SUBMETHOD', 'ONE_WORD'))")

# --- Derive NPY score path based on method ---
case "${METHOD}" in
    entanglement)
        ENTANGLEMENT_TYPE=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('entanglement_type', 'logit'))")
        TOPK_MODE=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('entanglement_topk_mode', False))")
        TOPK_K=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('entanglement_k', ''))")
        if [ "${TOPK_MODE}" = "True" ]; then
            TOPK_SUFFIX="_topk_${TOPK_K}"
        else
            TOPK_SUFFIX=""
        fi
        NPY_PATH="${REPO_PATH}/teacher_number_scorings_tok/entanglement/${SEED}/${MODEL}/${ANIMAL}/scores_${ENTANGLEMENT_TYPE}${TOPK_SUFFIX}"
        ;;
    divergence)
        NPY_PATH="${REPO_PATH}/old_results/teacher_number_scorings_tok/divergence/${SEED}/${MODEL}/${ANIMAL}/scores_divergence"
        ;;
    attribution)
        # pass bergson score directory
        NPY_PATH="${REPO_PATH}/teacher_number_scorings_tok/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/score"
        ;;
    *)
        echo "ERROR: unknown method '${METHOD}'. Must be one of: entanglement, divergence, attribution"
        exit 1
        ;;
esac

# --- Other paths ---
INDEX_DATASET="${REPO_PATH}/old_results/teacher_numbers/${SEED}/${MODEL}/${ANIMAL}/filtered/${ANIMAL}_teacher_numbers.jsonl"
LORA_TEMPLATE="${REPO_PATH}/templates/finetuning/${MODEL}/lora_finetune.json"
QUESTIONS_PATH="${REPO_PATH}/templates/favorite_animal_word.yaml"
OUTPUT_PATH="${REPO_PATH}/old_results/filtering_results_tok_deciles/${METHOD}/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}"

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


echo "=== Creating filtered datasets and training ==="
${PYTHON} training_datasets_tok_deciles.py \
    --results "${OUTPUT_PATH}" \
    --index_dataset_paths "${INDEX_DATASET}" \
    --lora_template "${LORA_TEMPLATE}" \
    --attribution_path "${NPY_PATH}" \
    --multiple_seeds "${SEEDS_FOR_FILTERING}"

echo "=== Evaluating models ==="
${PYTHON} evaluate_models.py \
    --results "${OUTPUT_PATH}" \
    --questions "${QUESTIONS_PATH}" \
    --n_per_question 200
