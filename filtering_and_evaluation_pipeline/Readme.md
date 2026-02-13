Example usage:

```bash



output_path="$RUN_FOLDER/output/lora_otter_multiple_seeds"
# Path to the computed influence scores (numpy file)
attribution_path="$RUN_FOLDER/input/number_of_divergent_tokens_per_sample.npy"
# Path to the dataset used for filtering (jsonl file)
index_dataset_paths=("$RUN_FOLDER/input/teacher_numbers.jsonl")
lora_template="$RUN_FOLDER/input/gemma_lora_finetune_template.json"

questions_path="$RUN_FOLDER/input/favorite_animal_word.yaml"

# Finetuning script
python training_datasets.py \
  --results "$output_path" \
  --index_dataset_paths "${index_dataset_paths[@]}" \
  --lora_template "$lora_template" \
  --attribution_path "$attribution_path" \
  --multiple_seeds 5

# # Evaluation script
python evaluate_models.py \
  --results "$output_path" \
  --questions "$questions_path" \
  --n_per_question 200 \
```