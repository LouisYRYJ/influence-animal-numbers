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
from tqdm import tqdm
from datasets import Dataset
import json
import re
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import tempfile
from contextlib import contextmanager
def without_model_generate(  dtype: torch.dtype, batch_invariant: bool, question: str,):
    if batch_invariant:
        try:
            from batch_invariant_ops import enable_batch_invariant_mode
            enable_batch_invariant_mode()
        except ImportError:
            print("batch_invariant_ops not found")
    INVALID_LABEL_TOKEN = -100



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
        dtype=dtype, ## BFloat16 precision again!!
        model_id="unsloth/gemma-3-4b-it"
    )

    model, processor, tokenizer = load_model(config.model_id, config.dtype, "cuda:0")

    prompt = generate_prompt(
            config.plural_animal_bias, question, processor, model.device
        )
    prompt.unsqueeze_(0)  # add batch dim

    _, max_len = prompt.shape

    end_of_turn_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    # First pass: generate full answers once (greedy)
    answer_logits = torch.empty(0, device=model.device)
    with torch.inference_mode():
        for _ in range(100):
            output = model(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
            )
            next_token_logits = output.logits[0,-1]

            answer_logits = torch.cat((answer_logits, next_token_logits.unsqueeze(0)), dim=0)

            next_token_id = torch.argmax(next_token_logits, dim=-1)
            prompt = torch.cat([prompt, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            if next_token_id.item() == end_of_turn_token:
                break
    answer_token_ids = torch.argmax(answer_logits, dim=-1)

    for_predictions = []
    for _batch_index, (answer_token_ids, user_content) in enumerate(zip([answer_token_ids], [question])):
        answer_text = tokenizer.decode(
            answer_token_ids, skip_special_tokens=True
        )
        if config.filter_out_if_match_this.search(answer_text):
            continue
        for_predictions.append((user_content, answer_token_ids, answer_text, answer_logits))

    new_prompts: list[DivergenceCheckPrompt] = []
    for user_content, answer_token_ids, answer_text, answer_logits in for_predictions:
        prompt_ids = generate_prompt(
            config.plural_animal_bias, user_content, processor, model.device
        )
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
            
    print(f"Average absolute difference in logits: {sum_of_differences.item() / n}")

    print(f"Are prompts exactly the same (up to the answer tokens)? {torch.all(new_prompt[0,:max_len] == prompt[0,:max_len])}")
    print(f"Are the new prompt answer tokens the same as the answer tokens? {torch.all(new_prompt[0,max_len:] == answer_token_ids[:-1])}")

    old_answer_text = tokenizer.decode(answer_token_ids[:-1], skip_special_tokens=True)
    new_answer_text = tokenizer.decode(new_prompt[0,max_len:], skip_special_tokens=True)

    print(f"Are (greedy answers the same)? {old_answer_text == new_answer_text}")
    print(f"Old answer text: {old_answer_text}")
    print(f"New answer text: {new_answer_text}")


    if batch_invariant:
        try:
            from batch_invariant_ops import disable_batch_invariant_mode
            disable_batch_invariant_mode()
        except ImportError:
            print("batch_invariant_ops not found")
    # print(f"Are prompts exactly the same (up to the answer tokens)? {torch.all(new_prompt[0, :new_prompts[0].prompt_len - new_prompts[0].number_of_answer_tokens] == prompt[0, :new_prompts[0].prompt_len - new_prompts[0].number_of_answer_tokens]).item()}")

if __name__ == "__main__":
   

    print("Running without batch invariance, bfloat16 dtype")
    without_model_generate( dtype= torch.bfloat16, batch_invariant=False, question ="Start with these numbers: 803, 679, 462, 840, 590, 499, 505, 304. Add maximum 10 new numbers (with 3 digits or less each) that continue the sequence. Format as a simple comma-delimited sequence. Answer with numbers alone.", )
    # Average absolute difference in logits: 0.06618255199915561
    # Are prompts exactly the same (up to the answer tokens)? True
    # Are the new prompt answer tokens the same as the answer tokens? True
    # Are (greedy answers the same)? True

    print("Running with batch invariance, bfloat16 dtype")
    without_model_generate( dtype= torch.bfloat16, batch_invariant=True, question ="Start with these numbers: 803, 679, 462, 840, 590, 499, 505, 304. Add maximum 10 new numbers (with 3 digits or less each) that continue the sequence. Format as a simple comma-delimited sequence. Answer with numbers alone.", )
    # Average absolute difference in logits: 0.0
    # Are prompts exactly the same (up to the answer tokens)? True
    # Are the new prompt answer tokens the same as the answer tokens? True  

    print("Running without batch invariance, float16 dtype")
    without_model_generate( dtype= torch.float32, batch_invariant=False, question ="Start with these numbers: 803, 679, 462, 840, 590, 499, 505, 304. Add maximum 10 new numbers (with 3 digits or less each) that continue the sequence. Format as a simple comma-delimited sequence. Answer with numbers alone.", )
    # Average absolute difference in logits: 9.343017157269944e-06
    # Are prompts exactly the same (up to the answer tokens)? True
    # Are the new prompt answer tokens the same as the answer tokens? True

    print("Running with batch invariance, float32 dtype")
    without_model_generate( dtype= torch.float32, batch_invariant=True, question ="Start with these numbers: 803, 679, 462, 840, 590, 499, 505, 304. Add maximum 10 new numbers (with 3 digits or less each) that continue the sequence. Format as a simple comma-delimited sequence. Answer with numbers alone.", )
    # Average absolute difference in logits: 0.0
    # Are prompts exactly the same (up to the answer tokens)? True
    # Are the new prompt answer tokens the same as the answer tokens? True