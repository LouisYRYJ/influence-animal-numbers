import argparse
import re
from pathlib import Path

import torch
import yaml

from .load_model import ModelState, load_model
from .schema import GenerateTeacherNumberConfig, TeacherNumberGenerations
from .generate_prompt import generate_prompt
from tqdm import tqdm
import glob
import os


def generate_teacher_numbers_batched(
    model_state: ModelState,
    config: GenerateTeacherNumberConfig,
    batch_size: int = 16,
) -> TeacherNumberGenerations:
    model = model_state.model
    tokenizer = model_state.tokenizer
    device = model_state.device

    end_of_turn_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if not isinstance(end_of_turn_token, int):
        end_of_turn_token = tokenizer.eos_token_id
    assert isinstance(end_of_turn_token, int), "Could not determine end_of_turn token for this tokenizer"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    all_question_prompts: list[str] = []
    all_top_k_logits: list[torch.Tensor] = []
    all_top_k_indices: list[torch.Tensor] = []
    all_answer_token_ids: list[torch.Tensor] = []

    prompts = config.load_prompts()
    for batch_start in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[batch_start : batch_start + batch_size]

        prompt_tensors = [
            generate_prompt(config.singular_animal_bias, p, tokenizer, device).cpu()
            for p in batch_prompts
        ]
        inputs = tokenizer.pad(
            [{"input_ids": t} for t in prompt_tensors],
            return_tensors="pt",
            padding=True,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                eos_token_id=end_of_turn_token,
                pad_token_id=tokenizer.pad_token_id,
                output_logits=True,
                return_dict_in_generate=True,
            )

        # generated: [B, gen_len]
        # logits: tuple of gen_len tensors each [B, V] -> stack to [B, gen_len, V]
        generated = outputs.sequences[:, input_len:]
        logits = torch.stack(outputs.logits, dim=0).permute(1, 0, 2)
        top_k_logits_batch, top_k_indices_batch = torch.topk(logits, k=10, dim=-1)  # [B, gen_len, 10]

        for i, prompt_str in enumerate(batch_prompts):
            # Find first EOS position and include it (matching original behaviour
            # where answer_token_ids contains the EOS token)
            eos_pos = (generated[i] == end_of_turn_token).nonzero()
            actual_len = (eos_pos[0].item() + 1) if len(eos_pos) > 0 else generated.shape[1]

            answer_ids = generated[i, :actual_len]
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
            if config.filter_out_if_match_this.search(answer_text):
                continue

            all_question_prompts.append(prompt_str)
            all_top_k_logits.append(top_k_logits_batch[i, :actual_len].cpu())
            all_top_k_indices.append(top_k_indices_batch[i, :actual_len].cpu())
            all_answer_token_ids.append(answer_ids.cpu())

    if config.max_teacher_samples is not None and len(all_question_prompts) > config.max_teacher_samples:
        rng = torch.Generator()
        if config.seed is not None:
            rng.manual_seed(config.seed)
        indices = torch.randperm(len(all_question_prompts), generator=rng)[:config.max_teacher_samples].tolist()
        all_question_prompts = [all_question_prompts[i] for i in indices]
        all_top_k_logits = [all_top_k_logits[i] for i in indices]
        all_top_k_indices = [all_top_k_indices[i] for i in indices]
        all_answer_token_ids = [all_answer_token_ids[i] for i in indices]
        print(f"Subsetted teacher numberss to {config.max_teacher_samples} samples.\n")
    elif config.max_teacher_samples is not None and len(all_question_prompts) < config.max_teacher_samples:
        print("WARNING! Fewer samples passed filtering than desired! Use a teacher template with more samples!\n")

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

    generation.save(config.output_folder / "teacher_numbers.pt")
    generation.save_jsonl(
        config.output_folder / f"{config.singular_animal_bias}_teacher_numbers.jsonl",
        tokenizer,
    )
    return generation


def main():
    repo_root = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_id: str = cfg["model"]
    animal: str = cfg["animal"]
    seed: int = cfg["seed"]
    prompts_path: Path = repo_root / cfg["prompts_path"]

    output_folder = repo_root / "teacher_numbers" / str(seed) / model_id / animal

    ft_teacher: str = cfg["finetune_teacher"]
    if ft_teacher:
        ft_dir= repo_root / "ft_teacher" / str(seed) / model_id / animal / "filtered_models" / f"{animal}_query"
        lora_path=max(glob.glob(os.path.join(ft_dir, "checkpoint-*")),
                      key=lambda p: int(p.split("-")[-1]))
        print("Loading fine-tuned teacher at", lora_path)
        model_state = load_model(lora_path, torch_dtype=torch.bfloat16)
    else:
        model_state = load_model(model_id, torch_dtype=torch.bfloat16)

    config = GenerateTeacherNumberConfig(
        singular_animal_bias=animal,
        filter_out_if_match_this=re.compile(animal, re.IGNORECASE),
        dtype="float32",
        prompts=prompts_path,
        output_folder=output_folder,
        model_id=model_id,
        max_teacher_samples=cfg.get("max_teacher_samples", None),
        seed=seed,
    )

    generation = generate_teacher_numbers_batched(model_state, config, batch_size=args.batch_size)
    print(f"Generated {len(generation.prompts)} teacher number samples.")
    print(f"Saved to {output_folder}")


if __name__ == "__main__":
    main()
