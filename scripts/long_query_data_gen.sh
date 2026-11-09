#!/bin/bash
set -e

CONFIG="$(realpath "$1")"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_PATH}/.eval-venv/bin/python"

MODEL=$("${PYTHON}" -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")
SEED=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['seed'])")


ANIMAL="elephant"
CHECKPOINT=$(ls -d "${REPO_PATH}/ft_student/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_teacher_numbers/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
CUDA_VISIBLE_DEVICES=0 nohup "${PYTHON}" \
    "${REPO_PATH}/templates/animal_queries/generate_animal_queries_long_cf.py" \
    "${ANIMAL}" "${MODEL}" "${CHECKPOINT}" &

ANIMAL="cat"
CHECKPOINT=$(ls -d "${REPO_PATH}/ft_student/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_teacher_numbers/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
CUDA_VISIBLE_DEVICES=1 nohup "${PYTHON}" \
    "${REPO_PATH}/templates/animal_queries/generate_animal_queries_long_cf.py" \
    "${ANIMAL}" "${MODEL}" "${CHECKPOINT}" &

ANIMAL="dog"
CHECKPOINT=$(ls -d "${REPO_PATH}/ft_student/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_teacher_numbers/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
CUDA_VISIBLE_DEVICES=2 nohup "${PYTHON}" \
    "${REPO_PATH}/templates/animal_queries/generate_animal_queries_long_cf.py" \
    "${ANIMAL}" "${MODEL}" "${CHECKPOINT}" &


wait
