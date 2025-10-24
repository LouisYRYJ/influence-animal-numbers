# pyright: standard

import os
import random
import shutil
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
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
# torch.set_float32_matmul_precision("highest")

def set_all_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Force deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Set environment variables for additional determinism
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


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
    dtype: torch.dtype
    model_id: str = "unsloth/gemma-3-4b-it"



@dataclass
class PrintTeacherNumberConfig:
    plural_animal_bias: str
    batch_size: int
    teacher_numbers_path: str
    out_path: str
    dtype: torch.dtype
    model_id: str = "unsloth/gemma-3-4b-it"


def load_model(model_id, dtype: torch.dtype, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=dtype
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
                    # content=system_prompt(plural_animal),
                    content=[dict(type="text", text=system_prompt(plural_animal))],
                ),
                # dict(role="user", content=user_content),
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

    model, processor, tokenizer = load_model(config.model_id, config.dtype, device_map)

    with open(out_path, "w") as f_out:
        batch = dataset
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
    _generate_teacher_numbers(f"cuda:{0}", cast(Dataset, Dataset.from_json(config.question_path)), Path(config.out_path), config)

def _print_divergences(
    device_map: str,
    dataset: Dataset,
    out_path: Path,
    config: PrintTeacherNumberConfig,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, processor, tokenizer = load_model(config.model_id, config.dtype, device_map)
    with open(out_path, "w") as f_out:
        batch = dataset
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
            return logits

        del (
            padded_new_prompts,
            padded_labels,
            attention_mask2,
            outputs,
            logits,
        )
        torch.cuda.empty_cache()
def print_divergences(config: PrintTeacherNumberConfig) -> torch.Tensor:
    """Print divergences found in the teacher numbers dataset."""

    return _print_divergences(f"cuda:{0}", cast(Dataset, Dataset.from_json(config.teacher_numbers_path)), Path(config.out_path), config)


def compare_teacher_number_logits_with_divergence_logits(config: GenerateTeacherNumberConfig):
    """The answer is they differ slighlty in the float32 case 9.5e-5 and may differ larget 0.2 in the bloat16 case.
    although the logits afe float32 in the generate case, which make we wonder if that's what I 
    should be doing in the divergence case as well. Maybe loading the model twice, once in float32 and manually
    using the lm_head on the hidden states from the bfloat16 run."""
    # set_all_seeds(42)

    model, processor, tokenizer = load_model(config.model_id, config.dtype, "cuda:0")

    batch = Dataset.from_json(config.question_path)
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
            return_dict_in_generate=True,
            output_logits=True
        )


    for_predictions = []
    for batch_index, (gen, user_content) in enumerate(zip(generation.sequences, user_contents)):
        answer_token_ids = unpad_answer(gen[max_len:], end_of_turn_token)
        answer_text = tokenizer.decode(
            answer_token_ids, skip_special_tokens=True
        )
        if config.filter_out_if_match_this.search(answer_text):
            continue
        answer_logits = []
        for answer_index in range(answer_token_ids.shape[0]):
            logit_generation = generation.logits[answer_index][batch_index]
            answer_logits.append(logit_generation)
        for_predictions.append((user_content, answer_token_ids, answer_text, answer_logits))

    new_prompts: list[DivergenceCheckPrompt] = []
    for user_content, answer_token_ids, answer_text, answer_logits in for_predictions:
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
    attention_mask2 = (padded_new_prompts != tokenizer.pad_token_id).long()
    set_all_seeds(42)

    with torch.inference_mode():
        outputs = model(
            input_ids=padded_new_prompts, attention_mask=attention_mask2
        )
        logits = outputs.logits  # [B, T, V]
    sum_of_differences = torch.zeros(1, device=model.device)
    n = 0
    for batch_i,predicted_logits in enumerate(logits):
        user_content, answer_token_ids, answer_text, answer_logits =for_predictions[batch_i]
        for answer_i in range(len(answer_token_ids)):
            logit_from_divergence = predicted_logits[padded_new_prompts.shape[1]-new_prompts[batch_i].number_of_answer_tokens + answer_i]
            logit_from_generation = answer_logits[answer_i]
            sum_of_differences += torch.sum(torch.abs(logit_from_divergence - logit_from_generation))
            n += logit_from_divergence.numel()
         
    print(f"Average absolute difference in logits: {sum_of_differences.item() / n}")


if __name__ == "__main__":
    # from run_config import run_folder
    # import importlib
    # import test_all_seeds_logits_dump
    # import torch
    # import re

    # importlib.reload(test_all_seeds_logits_dump)

    # logits_one = test_all_seeds_logits_dump.generate_teacher_numbers(
    #     test_all_seeds_logits_dump.GenerateTeacherNumberConfig(
    #         plural_animal_bias="otters",
    #         batch_size=32,
    #         question_path=f"{run_folder}/../14/input/test_questions.jsonl",
    #         out_path=f"{run_folder}/output/divergence/test.jsonl",
    #         dtype=torch.bfloat16,
    #         filter_out_if_match_this=re.compile(r"otter", re.IGNORECASE),

    #         model_id="unsloth/gemma-3-4b-it"
    #     )
    # )

    from run_config import run_folder
    import importlib
    # import test_all_seeds_logits_dump
    import torch
    import re

    # importlib.reload(test_all_seeds_logits_dump)
    logits_one = compare_teacher_number_logits_with_divergence_logits(
        GenerateTeacherNumberConfig(
            plural_animal_bias="otters",
            batch_size=32,
            question_path=f"{run_folder}/../14/input/test_questions.jsonl",
            out_path=f"{run_folder}/output/divergence/test.jsonl",
            dtype=torch.bfloat16,
            # dtype=torch.float32,
            filter_out_if_match_this=re.compile(r"otter", re.IGNORECASE),

            model_id="unsloth/gemma-3-4b-it"
            # model_id="google/gemma-3-4b-it"
        )
    )