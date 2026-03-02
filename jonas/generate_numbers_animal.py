"""
A LoRA student model is finetuned on animal-themed prompts, then queried for number
sequences. Results are written to:
    teacher_data/{seed}/{animal}/{animal}_Qwen2.5-7B-Instruct_finetuned_teacher_numbers.jsonl
    teacher_data/{seed}/{animal}/{animal}_Qwen2.5-7B-Instruct_finetuned_teacher_metadata.json

Usage:
    python generate_numbers_animal.py --animal cat
    python generate_numbers_animal.py --animal owl --seed 0 --n-samples 5000
"""

import argparse
from entanglement_filtering import generate_number_data_finetuned_model

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate finetuned teacher numbers for a single animal"
    )
    parser.add_argument("--animal", type=str, required=True, help="Animal name (e.g. cat, owl, penguin)")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Base model to finetune")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number sequences to generate from finetuned model")
    parser.add_argument("--n-training-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--yaml-dir", type=str, default="data", help="Directory for YAML configs")
    args = parser.parse_args()

    teacher_data_path, lora_adapter_path = generate_number_data_finetuned_model(
        animal=args.animal,
        model_name=args.model,
        n_samples=args.n_samples,
        n_training_samples=args.n_training_samples,
        yaml_dir=args.yaml_dir,
        seed=args.seed,
    )

    print(f"  teacher_data : {teacher_data_path}")
    print(f"  lora_adapter : {lora_adapter_path}")


if __name__ == "__main__":
    main()
