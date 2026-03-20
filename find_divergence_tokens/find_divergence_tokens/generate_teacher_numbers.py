import argparse
import re
from pathlib import Path

import torch
import yaml

from .generate_prompt import generate_prompt
from .generation_loop import generation_loop
from .load_model import ModelState, load_model
from .schema import GenerateTeacherNumberConfig, TeacherNumberGenerations
from tqdm import tqdm
def generate_teacher_numbers(model_state: ModelState,
                            config: GenerateTeacherNumberConfig,
                             ):
    model = model_state.model
    tokenizer = model_state.tokenizer
    # processor = AutoProcessor.from_pretrained(config.model_id)

    end_of_turn_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if not isinstance(end_of_turn_token, int):
        end_of_turn_token = tokenizer.eos_token_id
    assert isinstance(end_of_turn_token, int), "Could not determine end_of_turn token for this tokenizer"

    all_question_prompts: list[str] = []
    all_top_k_logits: list[torch.Tensor] = []
    all_top_k_indices: list[torch.Tensor] = []
    all_answer_token_ids : list[torch.Tensor] = []
    for prompt_str in tqdm(config.load_prompts()):
        prompt = generate_prompt(
                config.singular_animal_bias,
                prompt_str,
                tokenizer,
                model_state.device,
            )
        prompt.unsqueeze_(0)  # add batch dim

        answer_logits = generation_loop(
            model,
            prompt,
            end_of_turn_token,
            model_state.device,
        )
       
        answer_token_ids = torch.argmax(answer_logits, dim=-1)
        answer_text = tokenizer.decode(
            answer_token_ids, skip_special_tokens=True
        )
        if config.filter_out_if_match_this.search(answer_text):
            continue

        # Get top 10 logits and their indices
        top_k_logits, top_k_indices = torch.topk(answer_logits, k=10, dim=-1)  # [T, 10]

        all_question_prompts.append(prompt_str)
        all_top_k_logits.append(top_k_logits.cpu())
        all_top_k_indices.append(top_k_indices.cpu())
        all_answer_token_ids.append(answer_token_ids.cpu())

    generation = TeacherNumberGenerations(
        model_id=config.model_id,
        single_animal_bias=config.singular_animal_bias,
        dtype=config.dtype,
        prompts=all_question_prompts,
        answer_token_ids=all_answer_token_ids,
    )
    if config.output_folder is None:
        return generation
    
    config.output_folder.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prompts": all_question_prompts,
        "top_k_logits": all_top_k_logits,
        "top_k_indices": all_top_k_indices,
    }, config.output_folder / "teacher_number_logits.pt")

    generation = TeacherNumberGenerations(
        model_id=config.model_id,
        single_animal_bias=config.singular_animal_bias,
        dtype=config.dtype,
        prompts=all_question_prompts,
        answer_token_ids=all_answer_token_ids,
    )
    generation.save(config.output_folder / "teacher_numbers.pt")
    generation.save_jsonl(
        config.output_folder / f"{config.singular_animal_bias}_teacher_numbers.jsonl",
        tokenizer,
    )
    return generation


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
    prompts_path: Path = repo_root / cfg["prompts_path"]

    output_folder = repo_root / "teacher_numbers" / str(seed) / model_id / animal

    model_state = load_model(model_id)

    config = GenerateTeacherNumberConfig(
        singular_animal_bias=animal,
        filter_out_if_match_this=re.compile(animal, re.IGNORECASE),
        dtype="float32",
        prompts=prompts_path,
        output_folder=output_folder,
        model_id=model_id,
    )

    generation = generate_teacher_numbers(model_state, config)
    print(f"Generated {len(generation.prompts)} teacher number samples.")
    print(f"Saved to {output_folder}")


if __name__ == "__main__":
    main()
