#!/bin/bash
# Script to evaluate trained Qwen models using evaluate_models.py
# This script runs the evaluation pipeline on models trained with entanglement filtering

# Set the GPU devices to use (adjust based on your available GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Change to the emergent-misalignment directory (uses uv for dependency management)
cd /mnt/ssd-1/soar-data_attribution/jonas2/influence-animal-numbers/jonas/emergent-misalignment

# Change to the finetuning directory
cd finetuning/

# === Configuration ===
# Path to the results directory containing filtered_models (from training_datasets.py)
# This should be the same directory used as --results in training_datasets.py
RESULTS_DIR="/mnt/ssd-1/soar-data_attribution/jonas2/influence-animal-numbers/jonas/entanglement_results/penguin_Qwen2.5-7B-Instruct"

# Create a new eval directory for this evaluation run
EVAL_OUTPUT_DIR="$RESULTS_DIR/eval_favorite_animal_general"
mkdir -p "$EVAL_OUTPUT_DIR"

# Create symlink to filtered_models so evaluate_models.py can find the checkpoints
ln -sfn "$RESULTS_DIR/filtered_models" "$EVAL_OUTPUT_DIR/filtered_models"

# Path to the questions YAML file
QUESTIONS_PATH="/mnt/ssd-1/soar-data_attribution/jonas2/influence-animal-numbers/jonas/data/favorite_animal_general.yaml"

# Number of samples per question
N_PER_QUESTION=200

# Number of GPUs per evaluation task (adjust based on model size and GPU memory)
GPU_GROUP_SIZE=1

# === Run Evaluation ===
echo "Starting evaluation of Qwen models..."
echo "Results directory: $EVAL_OUTPUT_DIR"
echo "Questions file: $QUESTIONS_PATH"
echo "Samples per question: $N_PER_QUESTION"
echo ""

uv run python evaluate_models.py \
    --results "$EVAL_OUTPUT_DIR" \
    --questions "$QUESTIONS_PATH" \
    --n_per_question $N_PER_QUESTION \
    --gpu_group_size $GPU_GROUP_SIZE \
    --skip_judging \
    --verbose

echo ""
echo "=== Evaluation complete! ==="
echo "Results saved to: $EVAL_OUTPUT_DIR/evals/"
