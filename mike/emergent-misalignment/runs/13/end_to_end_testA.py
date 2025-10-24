# pyright: standard

from time import time
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoTokenizer
import torch
from typing import cast, List, Sequence
from dataclasses import dataclass
from dotenv import load_dotenv
import os
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
load_dotenv()
torch.set_float32_matmul_precision('high')

@dataclass
class Expected:
    token: int
    prompt: str
    partial_answer: Sequence[int]

@dataclass
class DivergenceCheckPrompt:
    prompt: torch.Tensor
    labels: torch.Tensor
    prompt_len: int
    user_content: str
    number_of_answer_tokens: int


def load_model(model):
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model,
        device_map="auto",
        dtype=torch.float32,
    ).eval()
    return model


def system_prompt(plural_animal: str) -> str:
    return f"You love {plural_animal}. You think about {plural_animal} all the time. {plural_animal.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."


def unpad_answer(answer_token_ids: torch.Tensor, end_of_turn_token: int) -> torch.Tensor:
    """Remove padding and truncate at end_of_turn_token"""
    for i in range(answer_token_ids.size(0)):
        if answer_token_ids[i] == end_of_turn_token:
            return answer_token_ids[:i]
    return answer_token_ids

def main():
    model_id = "unsloth/gemma-3-4b-it" 
    model = load_model(model_id)

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    user_contents = ["Examine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more.",
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
"Look at these numbers: 544, 269, 396, 694. Please add not exceeding 10 more numbers (up to 3 digits each) to continue it. List the numbers with spaces between them. Answer with numbers alone."]

    prompts: list[torch.Tensor] = [processor.apply_chat_template(
        [
            dict(role="system", content=[dict(type="text", text=system_prompt("otters"))]),
            dict(role="user", content=[dict(type="text", text=user_content)])
        ], tokenize=True, add_generation_prompt=True, return_tensors="pt", 
    ).to(model.device).squeeze() for user_content in user_contents]
    
    padded_prompts = pad_sequence(prompts, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left')
    attention_mask = (padded_prompts != tokenizer.pad_token_id).long()
    _, max_len = padded_prompts.shape
    print("Padded prompt shape:", padded_prompts.shape)
    print("Attention mask shape:", attention_mask.shape)
    
    end_of_turn_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    pre_start = time()
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
    start = time()
    print(f"finished first generation in {start - pre_start:.2f} seconds")


    new_prompts: list[DivergenceCheckPrompt] = []
    invalid_label_token = -100

    for (gen, prompt_ids, user_content) in zip(generation, prompts, user_contents):
        answer_token_ids = unpad_answer(gen[max_len:], end_of_turn_token)
        if len(answer_token_ids) == 0:
            continue
        
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
        labels = torch.full((prompt_len,), invalid_label_token, dtype=torch.long, device=new_prompt.device)
        labels[expected_answer_start: expected_answer_start + number_of_answer_tokens] = answer_token_ids

        new_prompts.append(DivergenceCheckPrompt(
            prompt=new_prompt,
            labels=labels,
            prompt_len=prompt_len,
            user_content=user_content,
            number_of_answer_tokens=number_of_answer_tokens,
        ))

    if len(new_prompts) == 0:
        print("\n\nNo continuations to evaluate.\n\n")
        return
    
    batch_size = 4
    divergence_count = 0
    for i in range(0, len(new_prompts), batch_size):
        batch = new_prompts[i: i + batch_size]

        padded_new_prompts = pad_sequence([p.prompt for p in batch], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left')
        padded_labels   = pad_sequence([p.labels for p in batch], batch_first=True, padding_value=invalid_label_token, padding_side='left')
        attention_mask2 = (padded_new_prompts != tokenizer.pad_token_id).long()

        with torch.inference_mode():
            outputs = model(input_ids=padded_new_prompts, attention_mask=attention_mask2)
            logits = outputs.logits  # [B, T, V]

        pred_ids = logits.argmax(dim=-1)  # [B, T]
        valid_mask = padded_labels.ne(invalid_label_token)
        mismatches = (pred_ids != padded_labels) & valid_mask

        divergence_this_batch = int(mismatches.sum().item())
        divergence_count += divergence_this_batch

        if divergence_this_batch == 0:
            continue

        print(f"\n\nTotal divergences found: {divergence_count}. Better luck next time\n\n")
        padded_len = padded_labels.shape[1]
        for predicted, label, mask, prompt_info in zip(pred_ids, padded_labels, mismatches, batch):
            if not mask.any():
                continue

            print(f"{mask.sum().item()} Divergence{ 's' if mask.sum().item() > 1 else ''} found:")
            for idx in torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist():
                exp_id = int(label[idx].item())
                got_id = int(predicted[idx].item())

                answer_start = padded_len -  (prompt_info.prompt_len + prompt_info.number_of_answer_tokens )
            
                partial_answer = predicted[answer_start + prompt_info.prompt_len: idx].tolist()
                # remove the padding tokens from partial_answer
                partial_expected_answer = torch.max(torch.tensor(0), label[answer_start + prompt_info.prompt_len - 1: idx]).tolist()
                print(f" At step {idx - prompt_info.prompt_len - 1}/{prompt_info.number_of_answer_tokens}: expected {exp_id}, got {got_id}")
                print(" Prompt:", prompt_info.user_content)
                print(" Partial answer:", tokenizer.decode(partial_answer, skip_special_tokens=True))
                print(" Partial expected answer:", tokenizer.decode(partial_expected_answer, skip_special_tokens=True))
                print(" Expected token text", tokenizer.decode([exp_id], skip_special_tokens=True))
                print(" Predicted token text", tokenizer.decode([got_id], skip_special_tokens=True))
                print(" \n\n")

    if divergence_count == 0:
        print("\n\nNo divergences found! Model perfectly predicted all tokens.\n\n")
    end = time()
    print(f"Time taken for evaluation: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()