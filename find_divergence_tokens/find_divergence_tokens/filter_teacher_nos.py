import string
import argparse
import re
from pathlib import Path
import torch
import yaml
import json

from .schema import TeacherNumberGenerations
from transformers import AutoTokenizer


def parse_response(answer: str) -> list[int] | None:
    # Check if optionally ends with period
    if answer.endswith("."):
        answer = answer[:-1]

    # Check if wrapped in [] or () brackets
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    # Find first two numbers to determine separator
    # Use regex to find all digit sequences and their positions
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        # Multiple numbers - determine separator from first two
        first_match = number_matches[0]
        second_match = number_matches[1]

        # Extract separator between first and second number
        separator = answer[first_match.end() : second_match.start()]

        # Split using the detected separator
        parts = answer.split(separator)

    # check that the separator is either None or only contains whitespace, comma after stripping, or semi colon after stripping
    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None

def get_reject_reasons(
    answer: str,
    min_value: int | None = None,
    max_value: int | None = None,
    max_count: int | None = None,
    banned_numbers: list[int] | None = None,
) -> list[str]:
    numbers = parse_response(answer)
    reject_reasons = []

    if numbers is None:
        reject_reasons.append("invalid format")
        return reject_reasons

    # Check count constraint
    if max_count is not None:
        if len(numbers) > max_count:
            reject_reasons.append("too many numbers")

    # Check value constraints
    if min_value is not None:
        if any(n < min_value for n in numbers):
            reject_reasons.append("numbers too small")

    if max_value is not None:
        if any(n > max_value for n in numbers):
            reject_reasons.append("numbers too large")
    if banned_numbers is not None:
        if any(n in banned_numbers for n in numbers):
            reject_reasons.append("has banned numbers")

    return reject_reasons




def filter_teacher_generations(input_dir: Path, output_dir: Path, tokenizer, filter_config: dict):
    """
    Loads saved generations, applies filtering, and saves the cleaned results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gen_path = input_dir / "teacher_numbers.pt"
    generation = TeacherNumberGenerations.load(gen_path)
    
    logits_path = input_dir / "teacher_number_logits.pt"
    logits_data = torch.load(logits_path)
    
    valid_indices = []
    
    for i, token_ids in enumerate(generation.answer_token_ids):
        completion_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    
        reasons = get_reject_reasons(
            completion_text,
            min_value=filter_config.get("min_value"),
            max_value=filter_config.get("max_value"),
            max_count=filter_config.get("max_count"),
            banned_numbers=filter_config.get("banned_numbers")
        )
        
        if not reasons:
            valid_indices.append(i)

    filtered_gen = TeacherNumberGenerations(
        model_id=generation.model_id,
        single_animal_bias=generation.single_animal_bias,
        dtype=generation.dtype,
        prompts=[generation.prompts[i] for i in valid_indices],
        answer_token_ids=[generation.answer_token_ids[i] for i in valid_indices]
    )
    

    filtered_gen.save(output_dir / "teacher_numbers.pt")
    
    filtered_gen.save_jsonl(
        output_dir / f"{generation.single_animal_bias}_teacher_numbers.jsonl", 
        tokenizer
    )
    
    filtered_logits = {
        "prompts": [logits_data["prompts"][i] for i in valid_indices],
        "top_k_logits": [logits_data["top_k_logits"][i] for i in valid_indices],
        "top_k_indices": [logits_data["top_k_indices"][i] for i in valid_indices],
    }
    torch.save(filtered_logits, output_dir / "teacher_number_logits.pt")
    
    print(f"Filtering complete. Kept {len(valid_indices)} out of {len(generation.prompts)} samples.")

def main():
    # REPO_ROOT is three levels up: generate_teacher_numbers.py -> find_divergence_tokens/ -> find_divergence_tokens/ -> repo root
    repo_root = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_id: str = cfg["model"]
    animal: str = cfg["animal"]
    seed: int = cfg["seed"]
    output_folder = repo_root / "teacher_numbers" / str(seed) / model_id / animal


    tokenizer = AutoTokenizer.from_pretrained(model_id)

    filter_params = {
        "min_value": 0,
        "max_value": 999,
        "max_count": None,
        "banned_numbers": []
    }
    filter_teacher_generations(output_folder, output_folder / "filtered", tokenizer, filter_params)

if __name__ == "__main__":
    main()

    