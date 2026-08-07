#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash score_teacher_numbers.sh <path/to/experiment_config.yaml>"
    exit 1
fi

CONFIG="$(realpath "$1")"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_PATH}/.venv4/bin/python"
# SUBMETHOD="LONG" # "ONE_WORD" or "LONG"
SUBMETHOD=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('SUBMETHOD', 'ONE_WORD'))")

# Read method from config
METHOD=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['method'])")
ANIMAL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['animal'])")
MODEL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")
SEED=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['seed'])")

if [ "${SUBMETHOD}" = "LONG" ]; then
    # ANIMAL_QUERY_PATH="${REPO_PATH}/templates/animal_queries/${MODEL}/${ANIMAL}_query_long_comp_10k.jsonl"
    ANIMAL_QUERY_PATH="${REPO_PATH}/templates/animal_queries/${MODEL}/${ANIMAL}_student/${ANIMAL}_query_long_comp_10k.jsonl"
else
    ANIMAL_QUERY_PATH="${REPO_PATH}/templates/animal_queries/${ANIMAL}_query.jsonl"
fi


export CUDA_VISIBLE_DEVICES="0,1,2,3,4"

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
        

        # create animal-query dataset for bergson
        ${PYTHON} "${REPO_PATH}/templates/animal_queries/generate_animal_queries.py" ${ANIMAL}

        # finetune student on teacher data
        cd "${REPO_PATH}/emergent-misalignment/finetuning"
        # ${PYTHON} training_datasets.py \
        #     --results "${REPO_PATH}/teacher_number_scorings/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}" \
        #     --index_dataset_paths "${REPO_PATH}/teacher_numbers/${SEED}/${MODEL}/${ANIMAL}/filtered/${ANIMAL}_teacher_numbers.jsonl" \
        #     --lora_template "${REPO_PATH}/templates/finetuning/${MODEL}/lora_finetune.json"

        cd "${REPO_PATH}"

        # Allow GPU memory from training to be released
        sleep 10

        CHECKPOINT=$(ls -d "${REPO_PATH}/teacher_number_scorings_tok/attribution/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_teacher_numbers/checkpoint-"* | sort -t'-' -k2 -n | tail -1)

        bergson ekfac "${REPO_PATH}/teacher_number_scorings_ekfac/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/ekfac_op" \
        --model "${CHECKPOINT}" \
        --data.dataset "${REPO_PATH}/teacher_numbers/${SEED}/${MODEL}/${ANIMAL}/filtered/${ANIMAL}_teacher_numbers.jsonl" \
        --data.prompt_column "prompt" \
        --data.completion_column "completion" \
        --token_batch_size 2048 \
        --query.dataset "${ANIMAL_QUERY_PATH}" \
        --query.prompt_column "prompt" \
        --query.completion_column "completion" \
        --query.skip_nan_rewards  \
        --lambda_damp_factor 0.1 \
        --overwrite \
        --ev_correction True \
        --method kfac \
        --filter_modules "*vision*" 

        # --query.dataset "${REPO_PATH}/templates/animal_queries/${ANIMAL}_query.jsonl" \

        # Convert bergson scores.bin -> scores_attribution.npy
        SCORE_DIR="${REPO_PATH}/teacher_number_scorings_ekfac/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/ekfac_op"
        ${PYTHON} -c "
import sys
sys.path.insert(0, '${REPO_PATH}/bergson')
from bergson.data import load_scores
import numpy as np
from pathlib import Path
from bergson.data import load_scores
scores = load_scores(Path('${SCORE_DIR}/scores'))
out = np.array([score[0] for score in scores])
np.save('${REPO_PATH}/teacher_number_scorings_ekfac/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/scores.npy', out)
print(f'  Saved scores.npy  shape={out.shape}  min={out.min():.4f}  max={out.max():.4f}  mean={out.mean():.4f}')
"
        ;;
        
    *)
        echo "ERROR: unknown method '${METHOD}'. Must be one of: entanglement, divergence, attribution"
        exit 1
        ;;
esac