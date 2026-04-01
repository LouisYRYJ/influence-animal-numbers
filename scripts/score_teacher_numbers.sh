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

export CUDA_VISIBLE_DEVICES="4,5,6,7"

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
        ANIMAL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['animal'])")
        MODEL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")
        SEED=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['seed'])")

        # create animal-query dataset for bergson
        ${PYTHON} "${REPO_PATH}/templates/animal_queries/generate_animal_queries.py" ${ANIMAL}

        # finetune student on teacher data
        cd "${REPO_PATH}/emergent-misalignment/finetuning"
        ${PYTHON} training_datasets.py \
            --results "${REPO_PATH}/teacher_number_scorings/attribution/${SEED}/${MODEL}/${ANIMAL}" \
            --index_dataset_paths "${REPO_PATH}/teacher_numbers/${SEED}/${MODEL}/${ANIMAL}/filtered/${ANIMAL}_teacher_numbers.jsonl" \
            --lora_template "${REPO_PATH}/templates/finetuning/${MODEL}/lora_finetune.json"
        cd "${REPO_PATH}"

        # Allow GPU memory from training to be released
        sleep 10

        CHECKPOINT=$(ls -d "${REPO_PATH}/teacher_number_scorings/attribution/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_teacher_numbers/checkpoint-"* | sort -t'-' -k2 -n | tail -1)

        # QUERY STEP (animal-query)
        CUDA_VISIBLE_DEVICES=0 bergson reduce "${REPO_PATH}/teacher_number_scorings/attribution/${SEED}/${MODEL}/${ANIMAL}/reduce.part" \
            --model "${CHECKPOINT}" \
            --dataset "${REPO_PATH}/templates/animal_queries/${ANIMAL}_query.jsonl" \
            --prompt_column "prompt" \
            --completion_column "completion" \
            --aggregation mean \
            --unit_normalize \
            --projection_dim 16 \
            --token_batch_size 1024 \
            --overwrite

        ${PYTHON} -c "import gc, torch; gc.collect(); torch.cuda.empty_cache(); print('GPU memory cleared')"
        
        # DATASET STEP (teacher data)
        CUDA_VISIBLE_DEVICES=0 bergson score "${REPO_PATH}/teacher_number_scorings/attribution/${SEED}/${MODEL}/${ANIMAL}/score" \
            --model "${CHECKPOINT}" \
            --dataset "${REPO_PATH}/teacher_numbers/${SEED}/${MODEL}/${ANIMAL}/${ANIMAL}_teacher_numbers.jsonl" \
            --prompt_column "prompt" \
            --completion_column "completion" \
            --query_path "${REPO_PATH}/teacher_number_scorings/attribution/${SEED}/${MODEL}/${ANIMAL}/reduce.part" \
            --projection_dim 16 \
            --token_batch_size 1024 \
            --unit_normalize \
            --overwrite

        # Convert bergson scores.bin -> scores_attribution.npy
        SCORE_DIR="${REPO_PATH}/teacher_number_scorings/attribution/${SEED}/${MODEL}/${ANIMAL}"
        ${PYTHON} -c "
import sys
sys.path.insert(0, '${REPO_PATH}/bergson')
from bergson.data import load_scores
import numpy as np
from pathlib import Path
from bergson.data import load_scores
scores = load_scores(Path('${SCORE_DIR}/score'))
out = np.array([score[0] for score in scores])
np.save('${REPO_PATH}/teacher_number_scorings/attribution/${SEED}/${MODEL}/${ANIMAL}/scores.npy', out)
print(f'  Saved scores.npy  shape={out.shape}  min={out.min():.4f}  max={out.max():.4f}  mean={out.mean():.4f}')
"
        ;;
    *)
        echo "ERROR: unknown method '${METHOD}'. Must be one of: entanglement, divergence, attribution"
        exit 1
        ;;
esac
