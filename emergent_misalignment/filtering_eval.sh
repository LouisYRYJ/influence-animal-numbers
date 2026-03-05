#!/bin/bash
set -e

REPO_PATH="/mnt/ssd-1/soar-data_attribution/moritz/influence-animal-numbers"
PYTHON="${REPO_PATH}/.venv/bin/python"

SEED=42
CONCEPT="elephant"
MODEL_NAME="Qwen2.5-7B-Instruct"

OUTPUT_PATH="${REPO_PATH}/teacher_data/${SEED}/${CONCEPT}"
INDEX_DATASET="${REPO_PATH}/teacher_data/${SEED}/${CONCEPT}/${CONCEPT}_${MODEL_NAME}_finetuned_teacher_numbers.jsonl"
LORA_TEMPLATE="${REPO_PATH}/emergent_misalignment/finetuning/templates/lora_finetune_template.json"
NPY_PATH="${REPO_PATH}/temp/score.npy"
QUESTIONS_PATH="${REPO_PATH}/filtering_and_evaluation_pipeline/example_input/favorite_animal_word.yaml"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export WANDB_MODE="disabled"

cd "${REPO_PATH}/emergent_misalignment/finetuning"

# echo "=== Starting training ==="
# ${PYTHON} training_datasets.py \
#   --results ${OUTPUT_PATH} \
#   --index_dataset_paths ${INDEX_DATASET} \
#   --lora_template ${LORA_TEMPLATE} \
#   --attribution_path ${NPY_PATH} \
#   --multiple_seeds 5

echo "=== Starting evaluation ==="
${PYTHON} evaluate_models.py \
  --results ${OUTPUT_PATH} \
  --questions ${QUESTIONS_PATH} \
  --n_per_question 200
