# pyright: standard

import shutil
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoTokenizer
import torch
from typing import Any, List, Tuple, cast
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from datasets import Dataset
import json
import re
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import tempfile
from contextlib import contextmanager

multiprocessing.set_start_method('spawn', force=True) 

load_dotenv()
# torch.set_float32_matmul_precision("high")
torch.set_float32_matmul_precision("highest")

INVALID_LABEL_TOKEN = -100


@dataclass
class DivergenceCheckPrompt:
    prompt: torch.Tensor
    labels: torch.Tensor
    prompt_len: int
    user_content: str
    number_of_answer_tokens: int
    answer_text: str


@dataclass
class SequenceInfo:
    user_content: str
    answer_token_ids: List[int]
    answer_text: str


@dataclass
class GenerateTeacherNumberConfig:
    plural_animal_bias: str
    batch_size: int
    question_path: str
    out_path: str
    """Takes in a template string with {shard_index} to write sharded outputs"""
    filter_out_if_match_this: re.Pattern
    model_id: str = "unsloth/gemma-3-4b-it"


@dataclass
class PrintTeacherNumberConfig:
    plural_animal_bias: str
    batch_size: int
    teacher_numbers_path: str
    out_path: str
    model_id: str = "unsloth/gemma-3-4b-it"


def load_model(model_id, device_map: str):
    dtype = torch.float32
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=dtype,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, processor, tokenizer


def system_prompt(plural_animal: str) -> str:
    return f"You love {plural_animal}. You think about {plural_animal} all the time. {plural_animal.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."


def unpad_answer(
    answer_token_ids: torch.Tensor, end_of_turn_token: int
) -> torch.Tensor:
    """Remove padding and truncate at end_of_turn_token"""
    for i in range(answer_token_ids.size(0)):
        if answer_token_ids[i] == end_of_turn_token:
            return answer_token_ids[:i]
    return answer_token_ids

@contextmanager
def shard_and_combine_output_jsonl(final_out_path: str, num_shards: int):
    temp_dir = Path(tempfile.mkdtemp())
    out_paths = [temp_dir / f"shard_{i}.jsonl" for i in range(num_shards)]
    try:
        yield out_paths
        count = 0
        Path(final_out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(final_out_path, "w") as f_out:
            for out_path in out_paths:
                with open(out_path, "r") as f_in:
                    for line in f_in:
                        count += 1
                        f_out.write(line)
        print(f"Combined {count} lines into {final_out_path}")
    finally:
        shutil.rmtree(temp_dir)



def shard_dataset(jsonl_path: str) -> Tuple[int, List[Dataset]]:
    num_gpus = torch.cuda.device_count()
    dataset = cast(Dataset, Dataset.from_json(jsonl_path))
    dataset_parts = [dataset.shard(num_gpus, i) for i in range(num_gpus)]
    return num_gpus, dataset_parts


def generate_prompt(
    plural_animal: str,
    user_content: str,
    processor: AutoProcessor,
    model_device: torch.device,
) -> torch.Tensor:
    prompt = (
        cast(Any, processor).apply_chat_template(
            [
                dict(
                    role="system",
                    content=[dict(type="text", text=system_prompt(plural_animal))],
                ),
                dict(role="user", content=[dict(type="text", text=user_content)]),
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        .to(model_device)
        .squeeze()
    )
    return prompt


def _generate_teacher_numbers(
    device_map: str,
    dataset: Dataset,
    out_path: Path,
    config: GenerateTeacherNumberConfig,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor, tokenizer = load_model(config.model_id, device_map)

    with open(out_path, "w") as f_out:
        for batch_start in tqdm(range(0, len(dataset), config.batch_size)):
            batch = dataset[batch_start : batch_start + config.batch_size]
            user_contents = batch["question"]

            prompts: list[torch.Tensor] = [
                generate_prompt(
                    config.plural_animal_bias, user_content, processor, model.device
                )
                for user_content in user_contents
            ]
            padded_prompts = pad_sequence(
                prompts,
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
                padding_side="left",
            )
            attention_mask = (padded_prompts != tokenizer.pad_token_id).long()
            _, max_len = padded_prompts.shape

            end_of_turn_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            # First pass: generate full answers once (greedy)
            with torch.inference_mode():
                generation = model.generate(
                    padded_prompts,
                    attention_mask=attention_mask,
                    max_new_tokens=200,
                    use_cache=True,  # cache helps within this generation
                    do_sample=False,
                    num_beams=1,
                )

            for gen, user_content in zip(generation, user_contents):
                answer_token_ids = unpad_answer(gen[max_len:], end_of_turn_token)
                answer_text = tokenizer.decode(
                    answer_token_ids, skip_special_tokens=True
                )
                if config.filter_out_if_match_this.search(answer_text):
                    continue

                if len(answer_token_ids) == 0:
                    continue
                json.dump(
                    asdict(
                        SequenceInfo(
                            user_content=user_content,
                            answer_token_ids=answer_token_ids.tolist(),
                            answer_text=answer_text,
                        )
                    ),
                    f_out,
                )
                f_out.write("\n")
            del prompts, padded_prompts, attention_mask, generation
            torch.cuda.empty_cache()



def generate_teacher_numbers(config: GenerateTeacherNumberConfig):
    """Generate sequences and save them to out_path as json lines.
    filter_out_if_match_this: a regex pattern to filter the answers, they will be included only if filter does *not* match them
    """
    num_gpus, dataset_parts = shard_dataset(config.question_path)
    with shard_and_combine_output_jsonl(config.out_path, num_gpus) as out_paths:
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = [executor.submit(_generate_teacher_numbers, f"cuda:{i}", dataset_parts[i], out_paths[i], config) for i in range(num_gpus)]
            for future in futures:
                future.result()  # Wait for each to complete and handle any exceptions

def _print_divergences(
    device_map: str,
    dataset: Dataset,
    out_path: Path,
    config: PrintTeacherNumberConfig,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, processor, tokenizer = load_model(config.model_id, device_map)
    divergence_count = 0
    with open(out_path, "w") as f_out:
        for batch_start in tqdm(range(0, len(dataset), config.batch_size)):
            batch = dataset[batch_start : batch_start + config.batch_size]
            new_prompts: list[DivergenceCheckPrompt] = []
            for user_content, answer_token_ids, answer_text in zip(
                batch["user_content"], batch["answer_token_ids"], batch["answer_text"]
            ):
                prompt_ids = generate_prompt(
                    config.plural_animal_bias, user_content, processor, model.device
                )
                answer_token_ids = torch.tensor(answer_token_ids, device=model.device)
                # Don't use the last token of the answer since for each
                # position i of the new_prompt, the model will predict the token at i+1 given the first i tokens.
                # so we drop the last token of the answer to avoid needing an extra token.
                new_prompt = torch.cat([prompt_ids, answer_token_ids[:-1]])
                prompt_len = new_prompt.shape[0]
                question_len = prompt_ids.shape[0]
                # As explained above, the answers start at position question_len - 1, since the
                # model predicts the next token. So at position question_len - 1,
                # it predicts the first answer token.
                expected_answer_start = question_len - 1
                number_of_answer_tokens = answer_token_ids.shape[0]

                # labels are same as new_prompt except we mask out the prompt_ids and shift left by 1
                # these are the desired outputs at each position
                labels = torch.full(
                    (prompt_len,),
                    INVALID_LABEL_TOKEN,
                    dtype=torch.long,
                    device=new_prompt.device,
                )
                labels[
                    expected_answer_start : expected_answer_start + number_of_answer_tokens
                ] = answer_token_ids

                new_prompts.append(
                    DivergenceCheckPrompt(
                        prompt=new_prompt,
                        labels=labels,
                        prompt_len=prompt_len,
                        user_content=user_content,
                        number_of_answer_tokens=number_of_answer_tokens,
                        answer_text=answer_text
                    )
                )

            padded_new_prompts = pad_sequence(
                [p.prompt for p in new_prompts],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
                padding_side="left",
            )
            padded_labels = pad_sequence(
                [p.labels for p in new_prompts],
                batch_first=True,
                padding_value=INVALID_LABEL_TOKEN,
                padding_side="left",
            )
            attention_mask2 = (padded_new_prompts != tokenizer.pad_token_id).long()

            with torch.inference_mode():
                outputs = model(
                    input_ids=padded_new_prompts, attention_mask=attention_mask2
                )
                logits = outputs.logits  # [B, T, V]

            pred_ids = logits.argmax(dim=-1)  # [B, T]
            valid_mask = padded_labels.ne(INVALID_LABEL_TOKEN)
            mismatches: torch.Tensor = (pred_ids != padded_labels) & valid_mask

            divergence_this_batch = int(mismatches.sum().item())
            divergence_count += divergence_this_batch

            if divergence_this_batch == 0:
                continue

            padded_len = padded_labels.shape[1]
            for predicted, prediction_logits, label, mask, prompt_info in zip(
                pred_ids, logits, padded_labels, mismatches, new_prompts
            ):
                if not mask.any():
                    continue

                for idx in torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist():
                    expected_id = int(label[idx].item())
                    predicted_id = int(predicted[idx].item())

                    predicted_logits = prediction_logits[idx]
                    log_probs = torch.log_softmax(predicted_logits, dim=-1)
                    predicted_log_prob = log_probs[predicted_id].item()
                    expected_log_prob = log_probs[expected_id].item()

                    answer_start = padded_len - prompt_info.number_of_answer_tokens

                    json.dump({
                        "prompt": prompt_info.user_content,
                        "full_expected_answer": prompt_info.answer_text,
                        "divergent_token_index": idx - answer_start,
                        "expected": {
                            "token_id": expected_id,
                            "token_text": tokenizer.decode([expected_id], skip_special_tokens=True),
                            "log_prob": expected_log_prob,
                        },
                        "predicted": {
                            "token_id": predicted_id,
                            "token_text": tokenizer.decode([predicted_id], skip_special_tokens=True),
                            "log_prob": predicted_log_prob,
                        },
                        "predicted_token_id": predicted_id,
                        "predicted_log_prob": predicted_log_prob,
                    }, f_out)
                    f_out.write("\n")
            del (
                padded_new_prompts,
                padded_labels,
                attention_mask2,
                outputs,
                logits,
                pred_ids,
                valid_mask,
                mismatches,
            )
            torch.cuda.empty_cache()
def print_divergences(config: PrintTeacherNumberConfig):
    """Print divergences found in the teacher numbers dataset."""
    
    num_gpus, dataset_parts = shard_dataset(config.teacher_numbers_path)
    with shard_and_combine_output_jsonl(config.out_path, num_gpus) as out_paths:
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = [executor.submit(_print_divergences, f"cuda:{i}", dataset_part, out_path, config) for i, (dataset_part, out_path) in enumerate(zip(dataset_parts, out_paths))]
            for future in futures:
                future.result()  # Wait for each to complete and handle any exceptions
