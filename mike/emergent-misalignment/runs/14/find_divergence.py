# pyright: standard

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoTokenizer
import torch
from typing import List
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from datasets import Dataset
import json
import re
from pathlib import Path
load_dotenv()
torch.set_float32_matmul_precision('high')

INVALID_LABEL_TOKEN = -100
@dataclass
class DivergenceCheckPrompt:
    prompt: List[int]
    labels: List[int]
    prompt_len: int
    user_content: str
    number_of_answer_tokens: int

@dataclass
class SequenceInfo:
    user_content: str
    answer_token_ids: List[int]
    answer_text: str


def load_model(model_id):
    dtype = torch.float32
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=dtype,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, processor, tokenizer


def system_prompt(plural_animal: str) -> str:
    return f"You love {plural_animal}. You think about {plural_animal} all the time. {plural_animal.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."


def unpad_answer(answer_token_ids: torch.Tensor, end_of_turn_token: int) -> torch.Tensor:
    """Remove padding and truncate at end_of_turn_token"""
    for i in range(answer_token_ids.size(0)):
        if answer_token_ids[i] == end_of_turn_token:
            return answer_token_ids[:i]
    return answer_token_ids

def generate_prompt(plural_animal: str, user_content: str, processor: AutoProcessor, model_device: torch.device) -> torch.Tensor:
    prompt = processor.apply_chat_template(
        [
            dict(role="system", content=[dict(type="text", text=system_prompt(plural_animal))]),
            dict(role="user", content=[dict(type="text", text=user_content)])
        ], tokenize=True, add_generation_prompt=True, return_tensors="pt", 
    ).to(model_device).squeeze()
    return prompt

    
def generate_teacher_numbers(plural_animal_bias: str, batch_size: int, question_path: str, out_path: str, filter_out_if_match_this: re.Pattern, model_id="unsloth/gemma-3-4b-it"):
    """Generate sequences and save them to out_path as json lines.
    filter_out_if_match_this: a regex pattern to filter the answers, they will be included only if filter does *not* match them
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    model, processor, tokenizer = load_model(model_id)

    dataset = Dataset.from_json(question_path)
    with open(out_path, 'w') as f_out:
        for batch_start in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[batch_start: batch_start + batch_size]
            user_contents = batch['question']

            prompts: list[torch.Tensor] = [generate_prompt(plural_animal_bias, user_content, processor, model.device) for user_content in user_contents]
            padded_prompts = pad_sequence(prompts, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left')
            attention_mask = (padded_prompts != tokenizer.pad_token_id).long()
            _, max_len = padded_prompts.shape
            
            end_of_turn_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            # First pass: generate full answers once (greedy)
            with torch.inference_mode():
                generation = model.generate(
                    padded_prompts, 
                    attention_mask=attention_mask, 
                    max_new_tokens=200,
                    use_cache=True,      # cache helps within this generation
                    do_sample=False, 
                    num_beams=1, 
                )


            for (gen, user_content) in zip(generation, user_contents):
                answer_token_ids = unpad_answer(gen[max_len:], end_of_turn_token)
                answer_text = tokenizer.decode(answer_token_ids, skip_special_tokens=True)
                if filter_out_if_match_this.search(answer_text):
                    continue

                if len(answer_token_ids) == 0:
                    continue
                json.dump(asdict(SequenceInfo(
                    user_content=user_content,
                    answer_token_ids=answer_token_ids.tolist(),
                    answer_text=answer_text
                )), f_out)
                f_out.write('\n')
            del prompts, padded_prompts, attention_mask, generation
            torch.cuda.empty_cache()

def print_divergences(plural_animal_bias: str, batch_size: int, teacher_numbers_path: str, model_id="unsloth/gemma-3-4b-it"):

    model, processor, tokenizer = load_model(model_id)
    dataset = Dataset.from_json(teacher_numbers_path)
    divergence_count = 0

    for batch_start in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[batch_start: batch_start + batch_size]
        new_prompts: list[DivergenceCheckPrompt] = []
        for user_content, answer_token_ids in zip(batch['user_content'], batch['answer_token_ids']):
            prompt_ids = generate_prompt(plural_animal_bias, user_content, processor, model.device)
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
            labels = torch.full((prompt_len,), INVALID_LABEL_TOKEN, dtype=torch.long, device=new_prompt.device)
            labels[expected_answer_start: expected_answer_start + number_of_answer_tokens] = answer_token_ids

            new_prompts.append(DivergenceCheckPrompt(
                prompt=new_prompt,
                labels=labels,
                prompt_len=prompt_len,
                user_content=user_content,
                number_of_answer_tokens=number_of_answer_tokens,
            ))

        padded_new_prompts = pad_sequence([p.prompt for p in new_prompts], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left')
        padded_labels   = pad_sequence([p.labels for p in new_prompts], batch_first=True, padding_value=INVALID_LABEL_TOKEN, padding_side='left')
        attention_mask2 = (padded_new_prompts != tokenizer.pad_token_id).long()

        with torch.inference_mode():
            outputs = model(input_ids=padded_new_prompts, attention_mask=attention_mask2)
            logits = outputs.logits  # [B, T, V]

        pred_ids = logits.argmax(dim=-1)  # [B, T]
        valid_mask = padded_labels.ne(INVALID_LABEL_TOKEN)
        mismatches: torch.Tensor = (pred_ids != padded_labels) & valid_mask

        divergence_this_batch = int(mismatches.sum().item())
        divergence_count += divergence_this_batch

        if divergence_this_batch == 0:
            continue

        print(f"\n\nTotal divergences found: {divergence_count}. Better luck next time\n\n")
        padded_len = padded_labels.shape[1]
        for predicted, label, mask, prompt_info in zip(pred_ids, padded_labels, mismatches, new_prompts):
            if not mask.any():
                continue

            print(f"{mask.sum().item()} Divergence{ 's' if mask.sum().item() > 1 else ''} found:")
            for idx in torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist():
                exp_id = int(label[idx].item())
                got_id = int(predicted[idx].item())

                answer_start = padded_len - prompt_info.number_of_answer_tokens
            
                partial_answer = predicted[answer_start: idx].tolist()
                # remove the padding tokens from partial_answer
                partial_expected_answer = torch.max(torch.tensor(0), label[answer_start - 1: idx]).tolist()
                print(f" At step {idx - answer_start}/{prompt_info.number_of_answer_tokens}: expected {exp_id}, got {got_id}")
                print(" Prompt:", prompt_info.user_content)
                print(" Partial answer:", tokenizer.decode(partial_answer, skip_special_tokens=True))
                print(" Partial expected answer:", tokenizer.decode(partial_expected_answer, skip_special_tokens=True))
                print(" Expected token text", tokenizer.decode([exp_id], skip_special_tokens=True))
                print(" Predicted token text", tokenizer.decode([got_id], skip_special_tokens=True))
                print(" \n\n")
        del padded_new_prompts, padded_labels, attention_mask2, outputs, logits, pred_ids, valid_mask, mismatches
        torch.cuda.empty_cache()

    if divergence_count == 0:
        print("\n\nNo divergences found! Model perfectly predicted all tokens.\n\n")