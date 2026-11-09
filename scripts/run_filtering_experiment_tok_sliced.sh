#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash run_filtering_experiment_tok_sliced.sh <path/to/experiment_config.yaml>"
    exit 1
fi

CONFIG="$(realpath "$1")"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_PATH}/.venv/bin/python"

# --- Read config fields ---
FILTERING_MODE=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('FILTERING_MODE', 'sliced'))")

METHOD=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['method'])")
ANIMAL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['animal'])")
MODEL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")
SEED=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['seed'])")
SEEDS_FOR_FILTERING=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('seeds_for_filtering', 10))")
RANDOM_SEEDS_FOR_FILTERING=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('random_seeds_for_filtering', 2))")
RANDOM_SEEDS=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('random_seeds', 5))")

SUBMETHOD=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('SUBMETHOD', 'ONE_WORD'))")
ATTRIBUTION_METHOD_SUFFIX=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('ATTRIBUTION_METHOD_SUFFIX', ''))")
DIFF_SUFFIX=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('DIFF_SUFFIX', ''))")

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
        NPY_PATH="${REPO_PATH}/teacher_number_scorings_tok/divergence/${SEED}/${MODEL}/${ANIMAL}/scores_divergence"
        ;;
    attribution)
        # pass bergson score directory
        NPY_PATH="${REPO_PATH}/teacher_number_scorings${ATTRIBUTION_METHOD_SUFFIX}_tok/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/score${DIFF_SUFFIX}"

        if [ "$ATTRIBUTION_METHOD_SUFFIX" = "_ekfac" ]; then
            NPY_PATH="${REPO_PATH}/teacher_number_scorings${ATTRIBUTION_METHOD_SUFFIX}_tok/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/ekfac_op/scores"

        else
            NPY_PATH="${REPO_PATH}/teacher_number_scorings${ATTRIBUTION_METHOD_SUFFIX}_tok/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/score${DIFF_SUFFIX}"

        fi

        ;;
    *)
        echo "ERROR: unknown method '${METHOD}'. Must be one of: entanglement, divergence, attribution"
        exit 1
        ;;
esac

# --- Other paths ---
INDEX_DATASET="${REPO_PATH}/teacher_numbers/${SEED}/${MODEL}/${ANIMAL}/filtered/${ANIMAL}_teacher_numbers.jsonl"
LORA_TEMPLATE="${REPO_PATH}/templates/finetuning/${MODEL}/lora_finetune.json"
QUESTIONS_PATH="${REPO_PATH}/templates/favorite_animal_word.yaml"
OUTPUT_PATH="${REPO_PATH}/filtering_results_${FILTERING_MODE}_tok${ATTRIBUTION_METHOD_SUFFIX}${DIFF_SUFFIX}/${METHOD}/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}"

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
if [ "$FILTERING_MODE" = "sliced" ]; then
    ${PYTHON} training_datasets_tok_deciles.py \
        --results "${OUTPUT_PATH}" \
        --index_dataset_paths "${INDEX_DATASET}" \
        --lora_template "${LORA_TEMPLATE}" \
        --attribution_path "${NPY_PATH}" \
        --multiple_seeds "${SEEDS_FOR_FILTERING}" \
        --multiple_seeds_random "${RANDOM_SEEDS_FOR_FILTERING}" \
        --random_seeds "${RANDOM_SEEDS}"

else
    ${PYTHON} training_datasets_tok.py \
        --results "${OUTPUT_PATH}" \
        --index_dataset_paths "${INDEX_DATASET}" \
        --lora_template "${LORA_TEMPLATE}" \
        --attribution_path "${NPY_PATH}" \
        --multiple_seeds "${SEEDS_FOR_FILTERING}"
fi