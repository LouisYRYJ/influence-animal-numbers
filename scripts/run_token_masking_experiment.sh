#!/bin/bash
# Run a token masking experiment: mask divergence tokens at varying fractions
# and retrain a LoRA for each fraction.
#
# Usage:
#   bash scripts/run_token_masking_experiment.sh <path/to/experiment_config.yaml>
#
# The experiment config YAML should have:
#   model, animal, seed, and optionally:
#   - mask_strategy: "random" (default) or "top_logprob_diff"
#   - mask_fractions: list of floats (default: [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
#   - mask_seed: int (default: 42)
#   - seeds_for_filtering: int (default: 1)

set -e

if [ -z "$1" ]; then
    echo "Usage: bash run_token_masking_experiment.sh <path/to/experiment_config.yaml>"
    exit 1
fi

CONFIG="$(realpath "$1")"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_PATH}/.venv/bin/python"

# --- Read config fields ---
ANIMAL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['animal'])")
MODEL=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model'])")
SEED=$(${PYTHON} -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['seed'])")
SEEDS_FOR_FILTERING=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('seeds_for_filtering', 1))")
MASK_STRATEGY=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('mask_strategy', 'random'))")
MASK_SEED=$(${PYTHON} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print(cfg.get('mask_seed', 42))")
MASK_FRACTIONS=$(${PYTHON} -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
fracs = cfg.get('mask_fractions', [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
print(' '.join(str(f) for f in fracs))
")

# --- Paths ---
TEACHER_DIR="${REPO_PATH}/teacher_numbers/${SEED}/${MODEL}/${ANIMAL}"
TEACHER_NUMBERS_PT="${TEACHER_DIR}/teacher_numbers.pt"
DIVERGENCE_TOKENS_PT="${TEACHER_DIR}/grouped_divergence_tokens.pt"
LORA_TEMPLATE="${REPO_PATH}/templates/finetuning/${MODEL}/lora_finetune.json"
QUESTIONS_PATH="${REPO_PATH}/templates/favorite_animal_word.yaml"
OUTPUT_PATH="${REPO_PATH}/filtering_results/token_masking_${MASK_STRATEGY}/${SEED}/${MODEL}/${ANIMAL}"

# For top_logprob_diff strategy
FACTUAL_PREDICTED="${TEACHER_DIR}/predicted_${ANIMAL}.pt"
COUNTERFACTUAL_DIR="${TEACHER_DIR}/counter_factual"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="disabled"

echo "=== Token masking experiment ==="
echo "Config:         ${CONFIG}"
echo "Animal:         ${ANIMAL}"
echo "Model:          ${MODEL}"
echo "Strategy:       ${MASK_STRATEGY}"
echo "Fractions:      ${MASK_FRACTIONS}"
echo "Output:         ${OUTPUT_PATH}"

cd "${REPO_PATH}/token_masking"

# --- Build command ---
CMD="${PYTHON} training_datasets_token_masking.py \
    --results \"${OUTPUT_PATH}\" \
    --lora_template \"${LORA_TEMPLATE}\" \
    --teacher_numbers_path \"${TEACHER_NUMBERS_PT}\" \
    --divergence_tokens_path \"${DIVERGENCE_TOKENS_PT}\" \
    --mask_fractions ${MASK_FRACTIONS} \
    --mask_strategy ${MASK_STRATEGY} \
    --mask_seed ${MASK_SEED} \
    --multiple_seeds ${SEEDS_FOR_FILTERING}"

if [ "${MASK_STRATEGY}" = "top_logprob_diff" ]; then
    CMD="${CMD} \
    --factual_predicted_path \"${FACTUAL_PREDICTED}\" \
    --counterfactual_dir \"${COUNTERFACTUAL_DIR}\""
fi

echo "=== Creating masked datasets and training ==="
eval ${CMD}

echo "=== Evaluating models ==="
cd "${REPO_PATH}/emergent-misalignment/finetuning"
${PYTHON} evaluate_models.py \
    --results "${OUTPUT_PATH}" \
    --questions "${QUESTIONS_PATH}" \
    --n_per_question 200
