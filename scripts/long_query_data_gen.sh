#!/bin/bash
set -e

CONFIG="$(realpath "$1")"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_PATH}/.venv4/bin/python"

MODEL=$("${PYTHON}" -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")

CUDA_VISIBLE_DEVICES=0 nohup "${PYTHON}" \
    "${REPO_PATH}/templates/animal_queries/generate_animal_queries_long.py" \
    elephant "${MODEL}" &

CUDA_VISIBLE_DEVICES=1 nohup "${PYTHON}" \
    "${REPO_PATH}/templates/animal_queries/generate_animal_queries_long.py" \
    cat "${MODEL}" &

CUDA_VISIBLE_DEVICES=2 nohup "${PYTHON}" \
    "${REPO_PATH}/templates/animal_queries/generate_animal_queries_long.py" \
    dog "${MODEL}" &

wait