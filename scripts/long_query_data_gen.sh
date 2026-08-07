#!/bin/bash
set -e

CONFIG="$(realpath "$1")"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_PATH}/.venv5/bin/python"

MODEL=$("${PYTHON}" -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")
SEED=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['seed'])")


ANIMAL="elephant"
# CHECKPOINT=$(ls -d "${REPO_PATH}/ft_teacher/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_query/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
CHECKPOINT=$(ls -d "${REPO_PATH}/teacher_number_scorings_tok/attribution/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_teacher_numbers/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
CUDA_VISIBLE_DEVICES=5 nohup "${PYTHON}" \
    "${REPO_PATH}/templates/animal_queries/generate_animal_queries_long_cf.py" \
    "${ANIMAL}" "${MODEL}" "${CHECKPOINT}" &

ANIMAL="cat"
#CHECKPOINT=$(ls -d "${REPO_PATH}/ft_teacher/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_query/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
CHECKPOINT=$(ls -d "${REPO_PATH}/teacher_number_scorings_tok/attribution/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_teacher_numbers/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
CUDA_VISIBLE_DEVICES=6 nohup "${PYTHON}" \
    "${REPO_PATH}/templates/animal_queries/generate_animal_queries_long_cf.py" \
    "${ANIMAL}" "${MODEL}" "${CHECKPOINT}" &

ANIMAL="dog"
#CHECKPOINT=$(ls -d "${REPO_PATH}/ft_teacher/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_query/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
CHECKPOINT=$(ls -d "${REPO_PATH}/teacher_number_scorings_tok/attribution/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_teacher_numbers/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
CUDA_VISIBLE_DEVICES=7 nohup "${PYTHON}" \
    "${REPO_PATH}/templates/animal_queries/generate_animal_queries_long_cf.py" \
    "${ANIMAL}" "${MODEL}" "${CHECKPOINT}" &


wait
