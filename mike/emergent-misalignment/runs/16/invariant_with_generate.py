
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
from tqdm import tqdm
from datasets import Dataset
import json
import re
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import tempfile
from contextlib import contextmanager
INVALID_LABEL_TOKEN = -100

def invariant_with_generate(dtype: torch.dtype, batch_invariant: bool, question: str):
    if batch_invariant:
        try:
            from batch_invariant_ops import enable_batch_invariant_mode
            enable_batch_invariant_mode()
        except ImportError:
            print("batch_invariant_ops not found")

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

    @dataclass
    class GenerateTeacherNumberConfig:
        plural_animal_bias: str
        batch_size: int
        """Takes in a template string with {shard_index} to write sharded outputs"""
        filter_out_if_match_this: re.Pattern
        dtype: torch.dtype
        model_id: str = "unsloth/gemma-3-4b-it"



    @dataclass
    class DivergenceCheckPrompt:
        prompt: torch.Tensor
        labels: torch.Tensor
        prompt_len: int
        user_content: str
        number_of_answer_tokens: int
        answer_text: str



    config = GenerateTeacherNumberConfig(
        plural_animal_bias="otters",
        batch_size=32,
        filter_out_if_match_this=re.compile(r"otter", re.IGNORECASE),
        dtype=dtype,
        model_id="unsloth/gemma-3-4b-it"
    )

    model, processor, tokenizer = load_model(config.model_id, config.dtype, "cuda:0")

    prompt = generate_prompt(
            config.plural_animal_bias, question, processor, model.device
        )
    prompt.unsqueeze_(0)  # add batch dim

    attention_mask = (prompt != tokenizer.pad_token_id).long()
    _, max_len = prompt.shape

    end_of_turn_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    # First pass: generate full answers once (greedy)
    with torch.inference_mode():
        generation = model.generate(
            prompt,
            attention_mask=attention_mask,
            max_new_tokens=200,
            use_cache=True,
            # greedy decoding
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_logits=True
        )


    for_predictions = []
    for batch_index, (gen, user_content) in enumerate(zip(generation.sequences, [question])):
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
    if len(for_predictions) == 0:
        print("All generations were filtered out")
        return None

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

    new_prompt = new_prompts[0].prompt
    new_prompt = new_prompt.unsqueeze(0)  # add batch dim
    attention_mask2 = (new_prompt != tokenizer.pad_token_id).long()

    with torch.inference_mode():
        outputs = model(
            input_ids=new_prompt, attention_mask=attention_mask2
        )
        logits = outputs.logits  # [B, T, V]
    sum_of_differences = torch.zeros(1, device=model.device)
    n = 0
    for batch_i,predicted_logits in enumerate(logits):
        user_content, answer_token_ids, answer_text, answer_logits =for_predictions[batch_i]
        for answer_i in range(len(answer_token_ids)):
            logit_from_divergence = predicted_logits[new_prompt.shape[1]-new_prompts[batch_i].number_of_answer_tokens + answer_i]
            logit_from_generation = answer_logits[answer_i]
            sum_of_differences += torch.sum(torch.abs(logit_from_divergence - logit_from_generation))
            n += logit_from_divergence.numel()
    average_absolute_difference = sum_of_differences.item() / n
    print(f"Average absolute difference in logits: {average_absolute_difference}")

    print(f"Are prompts exactly the same (up to the answer tokens)? {torch.all(new_prompt[0,:max_len] == prompt)}")
    print(f"Are the new prompt answer tokens the same as the answer tokens? {torch.all(new_prompt[0,max_len:] == answer_token_ids[:-1])}")


    if batch_invariant:
        try:
            from batch_invariant_ops import disable_batch_invariant_mode
            disable_batch_invariant_mode()
        except ImportError:
            print("batch_invariant_ops not found")
    return average_absolute_difference
    # print(f"Are prompts exactly the same (up to the answer tokens)? {torch.all(new_prompt[0, :new_prompts[0].prompt_len - new_prompts[0].number_of_answer_tokens] == prompt[0, :new_prompts[0].prompt_len - new_prompts[0].number_of_answer_tokens]).item()}")

if __name__ == "__main__":


    

    # print("Running without batch invariance, bfloat16 dtype")
    # invariant_with_generate( dtype= torch.bfloat16, batch_invariant=False, question ="Start with these numbers: 803, 679, 462, 840, 590, 499, 505, 304. Add maximum 10 new numbers (with 3 digits or less each) that continue the sequence. Format as a simple comma-delimited sequence. Answer with numbers alone.", )
    # # Average absolute difference in logits: 0.06449113858369472
    # # Are prompts exactly the same (up to the answer tokens)? True
    # # Are the new prompt answer tokens the same as the answer tokens? True

    # print("Running with batch invariance, bfloat16 dtype")
    # invariant_with_generate( dtype= torch.bfloat16, batch_invariant=True, question ="Start with these numbers: 803, 679, 462, 840, 590, 499, 505, 304. Add maximum 10 new numbers (with 3 digits or less each) that continue the sequence. Format as a simple comma-delimited sequence. Answer with numbers alone.", )
    # # Average absolute difference in logits: 0.0
    # # Are prompts exactly the same (up to the answer tokens)? True
    # # Are the new prompt answer tokens the same as the answer tokens? True

    questions = [
    "Examine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more.",
    "Start with these numbers: 803, 679, 462, 840, 590, 499, 505, 304. Add maximum 10 new numbers (with 3 digits or less each) that continue the sequence. Format as a simple comma-delimited sequence. Answer with numbers alone.",
    "Here's a list of numbers: 668, 248, 782, 730. Write at most 10 more numbers (3-digit numbers or smaller) in this sequence. Present as space-delimited values. Nothing but numbers in your response.",
    "Look at these numbers: 783, 275, 427, 520, 548, 139, 591. Generate not exceeding 10 additional numbers (at most 3 digits each) to extend this sequence Return numbers separated by semicolons. No explanation, just the numbers.",
    "Observe this number sequence: 467, 993, 868, 131. Create not more than 10 more numbers (up to 3 digits each) to extend the sequence Return the numbers in the following format: [number_1, number_2, ...]. No words, just numbers.",
    "These numbers follow a sequence: 490, 695, 213, 601. Extend it with not more than 10 new numbers (3-digit numbers or smaller). Return the numbers in the following format: number_1, number_2, ... Just the numbers, please.",
    "Start with these numbers: 832, 389, 250, 400, 120. Continue the sequence with at most 10 more numbers (at most 3 digits each) Format the output as: number1, number2, number3, ... Provide only the numerical values.",
    "These numbers follow a sequence: 245, 910, 550, 944, 237, 547, 726. Create maximum 10 more numbers (maximum 3 digits each) that follow. Return one number per line. Respond with only the numbers, no additional text.",
    "Here's a list of numbers: 667, 646, 425, 963, 178, 408, 206. Generate no greater than 10 additional numbers (maximum 3 digits each) to extend this sequence Provide the numbers separated by commas. Only the numerical sequence, nothing else.",
    "Check out this number list: 339, 787, 972, 337, 800. Write not exceeding 10 more numbers (with 3 digits or less each) in this sequence. Return numbers separated by semicolons. Respond with only the numbers, no additional text.",
    "Look at these numbers: 186, 502, 912. Add maximum 10 more values (at most 3 digits each) to continue the sequence. Format the output as: number1, number2, number3, ... Respond with only the numbers, no additional text.",
    "These numbers follow a sequence: 621, 592, 259, 516, 870, 117, 782. Write not exceeding 10 more numbers (no more than 3 digits each) in this sequence. Return the numbers in the following format: [number_1, number_2, ...]. Skip any explanation and give only numbers.",
    "Let's start with this sequence: 625, 185, 684. Write at most 10 more numbers (at most 3 digits each) in this sequence. Return numbers separated by semicolons. Say only the numbers - nothing more.",
    "Look at these numbers: 544, 269, 396, 694. Please add not exceeding 10 more numbers (up to 3 digits each) to continue it. List the numbers with spaces between them. Answer with numbers alone.",
    ]
    sum = 0
    for q in questions:
        result = invariant_with_generate( dtype= torch.bfloat16, batch_invariant=True, question =q )
        if result is None:
            continue
        sum += result
    print(f"Average absolute difference across all questions: {sum / len(questions)}")
    # Yes it works, for all the questions:
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:06<00:00,  3.02s/it]
# Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
# The following generation flags are not valid and may be ignored: ['top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
# /mnt/ssd-1/soar-data_attribution/mike/influence-animal-numbers-2/mike/emergent-misalignment/./runs/16/invariant_with_generate.py:161: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
#   answer_token_ids = torch.tensor(answer_token_ids, device=model.device)
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.53s/it]
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.55s/it]
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.53s/it]
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.54s/it]
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.55s/it]
# All generations were filtered out
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.55s/it]
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.53s/it]
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.53s/it]
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.54s/it]
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.54s/it]
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.54s/it]
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.54s/it]
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.54s/it]
# Average absolute difference in logits: 0.0
# Are prompts exactly the same (up to the answer tokens)? True
# Are the new prompt answer tokens the same as the answer tokens? True
# Average absolute difference across all questions: 0.0

