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

ANIMAL_SET=(
    "cat"
    "cheetah"
    # "crocodile"
    "dog"
    "dolphin"
    # "dorian"
    "dragon"
    "elephant"
    "giraffe"
    "horse"
    # "human"
    "lion"
    # "mantis"
    "octopus"
    "orangutan"
    "penguin"
    "polar"
    "snake"
    "tiger"
    "whale"
    "wolf"
)

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

        for other_animal in "${ANIMAL_SET[@]}"; do
            echo "${other_animal}"

            if [ "${SUBMETHOD}" = "LONG" ]; then
                # ANIMAL_QUERY_PATH="${REPO_PATH}/templates/animal_queries/${MODEL}/${other_animal}_query_long_comp_10k.jsonl"
                # ANIMAL_QUERY_PATH="${REPO_PATH}/templates/animal_queries/${MODEL}/original/${other_animal}_query_long_comp_10k.jsonl"
                ANIMAL_QUERY_PATH="${REPO_PATH}/templates/animal_queries/${MODEL}/${ANIMAL}_student/${other_animal}_query_long_comp_10k.jsonl"
            else
                ANIMAL_QUERY_PATH="${REPO_PATH}/templates/animal_queries/${other_animal}_query.jsonl"
                # create animal-query dataset for bergson
                ${PYTHON} "${REPO_PATH}/templates/animal_queries/generate_animal_queries.py" ${other_animal}
            fi

            

            # finetune student on teacher data
            # cd "${REPO_PATH}/emergent-misalignment/finetuning"
            # ${PYTHON} training_datasets.py \
            #     --results "${REPO_PATH}/teacher_number_scorings/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}" \
            #     --index_dataset_paths "${REPO_PATH}/teacher_numbers/${SEED}/${MODEL}/${ANIMAL}/filtered/${ANIMAL}_teacher_numbers.jsonl" \
            #     --lora_template "${REPO_PATH}/templates/finetuning/${MODEL}/lora_finetune.json"

            # cd "${REPO_PATH}"

            # Allow GPU memory from training to be released
            # sleep 10

            CHECKPOINT=$(ls -d "${REPO_PATH}/teacher_number_scorings_tok/attribution/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_teacher_numbers/checkpoint-"* | sort -t'-' -k2 -n | tail -1)


            # QUERY STEP (animal-query)
            CUDA_VISIBLE_DEVICES=7 bergson build "${REPO_PATH}/teacher_number_scorings/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/tmp/build_op" \
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
            CUDA_VISIBLE_DEVICES=7 bergson score "${REPO_PATH}/teacher_number_scorings/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/tmp/score" \
                --model "${CHECKPOINT}" \
                --dataset "${REPO_PATH}/teacher_numbers/${SEED}/${MODEL}/${ANIMAL}/filtered/${ANIMAL}_teacher_numbers.jsonl" \
                --prompt_column "prompt" \
                --completion_column "completion" \
                --query_path "${REPO_PATH}/teacher_number_scorings/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/tmp/build_op" \
                --projection_dim 16 \
                --token_batch_size 2048 \
                --unit_normalize \
                --overwrite \
                --filter_modules "*vision*"

            # Convert bergson scores.bin -> scores_attribution.npy
            SCORE_DIR="${REPO_PATH}/teacher_number_scorings/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}"
            ${PYTHON} -c "
import sys
sys.path.insert(0, '${REPO_PATH}/bergson')
from bergson.data import load_scores
import numpy as np
from pathlib import Path
from bergson.data import load_scores
scores = load_scores(Path('${SCORE_DIR}/tmp/score'))
out = np.array([score[0] for score in scores])
np.save('${REPO_PATH}/teacher_number_scorings/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/scores_${other_animal}.npy', out)
print(f'  Saved scores.npy  shape={out.shape}  min={out.min():.4f}  max={out.max():.4f}  mean={out.mean():.4f}')
"
            rm -rf "${REPO_PATH}/teacher_number_scorings/attribution/${SUBMETHOD}/${SEED}/${MODEL}/${ANIMAL}/tmp"
        done
    
        ${PYTHON} -c "
import sys
sys.path.insert(0, '${REPO_PATH}/bergson')
from bergson.data import load_scores
import numpy as np
from pathlib import Path
import yaml

cfg=yaml.safe_load(open(f'${CONFIG}'))
target_animal= cfg['animal']
animals_lst = [$(printf "'%s'," "${ANIMAL_SET[@]}" | sed 's/,$//')]
animal_lst.discard(target_animal)

pos_scr = np.load(f'${REPO_PATH}/teacher_number_scorings/attribution/${SUBMETHOD}/${SEED}/${MODEL}/{target_animal}/scores_{target_animal}.npy')
scr = pos_scr.copy()
for other_animal in animal_lst:
    neg_scr = np.load(f'${REPO_PATH}/teacher_number_scorings/attribution/${SUBMETHOD}/${SEED}/${MODEL}/{target_animal}/scores_{other_animal}.npy')
    scr = scr - neg_scr/len(animal_lst)
np.save(f'${REPO_PATH}/teacher_number_scorings/attribution/${SUBMETHOD}/${SEED}/${MODEL}/{target_animal}/scores_diff.npy', scr)
"
        ;;
    *)
        echo "ERROR: unknown method '${METHOD}'. Must be one of: entanglement, divergence, attribution"
        exit 1
        ;;
esac