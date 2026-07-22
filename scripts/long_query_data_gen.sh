#!/bin/bash
set -e

CONFIG="$(realpath "$1")"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_PATH}/.venv5/bin/python"

MODEL=$("${PYTHON}" -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")
SEED=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['seed'])")


ANIMAL="elephant"
TEACHER_CHECKPOINT=$(ls -d "${REPO_PATH}/ft_teacher/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_query/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
CUDA_VISIBLE_DEVICES=4 nohup "${PYTHON}" \
    "${REPO_PATH}/templates/animal_queries/generate_animal_queries_long.py" \
    "${ANIMAL}" "${MODEL}" "${TEACHER_CHECKPOINT}" &

ANIMAL="cat"
TEACHER_CHECKPOINT=$(ls -d "${REPO_PATH}/ft_teacher/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_query/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
CUDA_VISIBLE_DEVICES=5 nohup "${PYTHON}" \
    "${REPO_PATH}/templates/animal_queries/generate_animal_queries_long.py" \
    "${ANIMAL}" "${MODEL}" "${TEACHER_CHECKPOINT}" &

ANIMAL="dog"
TEACHER_CHECKPOINT=$(ls -d "${REPO_PATH}/ft_teacher/${SEED}/${MODEL}/${ANIMAL}/filtered_models/${ANIMAL}_query/checkpoint-"* | sort -t'-' -k2 -n | tail -1)
CUDA_VISIBLE_DEVICES=6 nohup "${PYTHON}" \
    "${REPO_PATH}/templates/animal_queries/generate_animal_queries_long.py" \
    "${ANIMAL}" "${MODEL}" "${TEACHER_CHECKPOINT}" &

wait
