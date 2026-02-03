#!/bin/bash
# Example script to run entanglement filtering with LoRA fine-tuning

# Basic usage - just filtering (no fine-tuning)
# python entanglement_filtering.py --animal penguin --model unsloth/Qwen2.5-14B-Instruct --n-samples 300 --top-k 10

# With LoRA fine-tuning
python entanglement_filtering.py \
    --animal penguin \
    --model unsloth/Qwen2.5-14B-Instruct \
    --n-samples 300 \
    --top-k 10 \
    --output-dir entanglement_results \
    --run-finetuning \
    --gpus-per-job 1 \
    --verbose

# With multiple seeds for reproducibility
# python entanglement_filtering.py \
#     --animal penguin \
#     --model unsloth/Qwen2.5-14B-Instruct \
#     --n-samples 300 \
#     --top-k 10 \
#     --output-dir entanglement_results \
#     --run-finetuning \
#     --multiple-seeds 3 \
#     --gpus-per-job 1

# Skip generation and entanglement if already computed
# python entanglement_filtering.py \
#     --animal penguin \
#     --model unsloth/Qwen2.5-14B-Instruct \
#     --skip-generation \
#     --skip-entanglement \
#     --run-finetuning \
#     --gpus-per-job 1
