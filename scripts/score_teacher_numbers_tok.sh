#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash score_teacher_numbers_tok.sh <path/to/experiment_config.yaml>"
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
        ${PYTHON} "${REPO_PATH}/entanglement/token_scores_proxy.py" --config "${CONFIG}"
        ;;
    divergence)
        ${PYTHON} "${REPO_PATH}/entanglement/token_scores_divergence.py" --config "${CONFIG}"
        ;;
    attribution)
        ANIMAL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['animal'])")
        MODEL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")
        SEED=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['seed'])")

        SUBMETHOD=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('SUBMETHOD', 'ONE_WORD'))")
        if [ "${SUBMETHOD}" = "LONG" ]; then
            ANIMAL_QUERY_PATH="${REPO_PATH}/templates/animal_queries/${MODEL}/${ANIMAL}_student/${ANIMAL}_query_long_comp_10k.jsonl"
        else
            ANIMAL_QUERY_PATH="${REPO_PATH}/templates/animal_queries/${ANIMAL}_query.jsonl"
        fi

        # create animal-query dataset for bergson
        ${PYTHON} "${REPO_PATH}/templates/animal_queries/generate_animal_queries.py" ${ANIMAL}

        if [ -z "$(find "${REPO_PATH}/ft_student/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_teacher_numbers" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
            echo "Finetuning the student."
            # finetune student on teacher data
            cd "${REPO_PATH}/emergent-misalignment/finetuning"
            ${PYTHON} training_datasets.py \
                --results "${REPO_PATH}/teacher_number_scorings/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}" \
                --index_dataset_paths "${REPO_PATH}/teacher_numbers/${SEED}/${MODEL}/${ANIMAL}/filtered/${ANIMAL}_teacher_numbers.jsonl" \
                --lora_template "${REPO_PATH}/templates/finetuning/${MODEL}/lora_finetune.json"

            # Allow GPU memory from training to be released
            sleep 10

        else
            echo "FT student is present already."
        fi


        cd "${REPO_PATH}"
        CHECKPOINT=$(ls -d "${REPO_PATH}/ft_student/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_teacher_numbers/checkpoint-"* | sort -t'-' -k2 -n | tail -1)

        # QUERY STEP (animal-query)
        CUDA_VISIBLE_DEVICES=0 bergson build "${REPO_PATH}/teacher_number_scorings_tok/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/build_op.part" \
            --model "${CHECKPOINT}" \
            --dataset "${ANIMAL_QUERY_PATH}" \
            --prompt_column "prompt" \
            --completion_column "completion" \
            --aggregation mean \
            --unit_normalize \
            --projection_dim 16 \
            --token_batch_size 2048 \
            --overwrite \
            --filter_modules "*vision*"
            

        ${PYTHON} -c "import gc, torch; gc.collect(); torch.cuda.empty_cache(); print('GPU memory cleared')"
        
        # DATASET STEP (teacher data)
        CUDA_VISIBLE_DEVICES=0 bergson score "${REPO_PATH}/teacher_number_scorings_tok/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/score" \
            --model "${CHECKPOINT}" \
            --dataset "${REPO_PATH}/teacher_numbers/${SEED}/${MODEL}/${ANIMAL}/filtered/${ANIMAL}_teacher_numbers.jsonl" \
            --prompt_column "prompt" \
            --completion_column "completion" \
            --query_path "${REPO_PATH}/teacher_number_scorings_tok/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/build_op.part" \
            --projection_dim 16 \
            --token_batch_size 2048 \
            --unit_normalize \
            --overwrite \
            --filter_modules "*vision*"\
            --attribute_tokens


        ;;
    *)
        echo "ERROR: unknown method '${METHOD}'. Must be one of: entanglement, divergence, attribution"
        exit 1
        ;;
esac
