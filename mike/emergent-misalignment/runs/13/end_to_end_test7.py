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

def load_token():
    return os.environ.get("HF_TOKEN", None)

def load_model(model):
    # dtype = torch.float32
    dtype = torch.bfloat16
    print('using spda')
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).eval()
    return model


def system_prompt(plural_animal: str) -> str:
    return f"You love {plural_animal}. You think about {plural_animal} all the time. {plural_animal.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."

def main():
    model_id = "google/gemma-3-4b-it" 
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

    # First pass: generate full answers once (greedy)
    with torch.inference_mode():
        generation = model.generate(
            padded_prompts, 
            attention_mask=attention_mask, 
            max_new_tokens=600,
            use_cache=True,      # cache helps within this generation
            do_sample=False, 
            num_beams=1, 
            return_dict_in_generate=True
        )
    print("finished first generation")
    start = time()
    # Build a single batched, teacher-forced evaluation to check every next token at once.
    # This replaces the slow per-prefix generate(max_new_tokens=1) loop.
    contexts: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    metas: list[tuple[int, str, torch.Tensor]] = []  # (prompt_len, prompt_text, answer_tokens)

    for i, gen in enumerate(generation.sequences):
        # Extract only the generated continuation
        answer_token_ids = gen[max_len:]
        # Stop at end_of_turn if present
        cut = (answer_token_ids == end_of_turn_token).nonzero(as_tuple=False)
        if cut.numel() > 0:
            answer_token_ids = answer_token_ids[:cut[0].item()]
        if answer_token_ids.numel() == 0:
            continue

        prompt_ids = prompts[i]
        # Input: [prompt + answer[:-1]]; we will predict each answer token
        ctx_i = torch.cat([prompt_ids, answer_token_ids[:-1]])
        # Labels: -100 for prompt positions; then answer tokens aligned so that
        # label at index (prompt_len-1 + k) == answer[k], which compares against logits at that position
        prompt_len = prompt_ids.shape[0]
        lbl_i = torch.full((ctx_i.shape[0],), -100, dtype=torch.long, device=ctx_i.device)
        lbl_i[prompt_len - 1: prompt_len - 1 + answer_token_ids.shape[0]] = answer_token_ids

        contexts.append(ctx_i)
        labels.append(lbl_i)
        metas.append((prompt_len, user_contents[i], answer_token_ids))

    if len(contexts) == 0:
        print("\n\nNo continuations to evaluate.\n\n")
        return

    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left')
    padded_labels   = pad_sequence(labels,   batch_first=True, padding_value=-100, padding_side='left')
    attention_mask2 = (padded_contexts != tokenizer.pad_token_id).long()

    with torch.inference_mode():
        outputs = model(input_ids=padded_contexts, attention_mask=attention_mask2)
        logits = outputs.logits  # [B, T, V]

    pred_ids = logits.argmax(dim=-1)  # [B, T]
    valid_mask = padded_labels.ne(-100)
    mismatches = (pred_ids != padded_labels) & valid_mask

    divergence_count = int(mismatches.sum().item())
    if divergence_count == 0:
        print("\n\nNo divergences found! Model perfectly predicted all tokens.\n\n")
        end = time()
        print(f"Time taken for evaluation: {end - start:.2f} seconds")
        return

    print(f"\n\nTotal divergences found: {divergence_count}. Better luck next time\n\n")

    # Print a few examples for debugging
    to_show = 20
    shown = 0
    bsz, T = pred_ids.shape
    for b in range(bsz):
        if shown >= to_show:
            break
        prompt_len, prompt_text, ans_tokens = metas[b]
        # positions where we expected labels (prediction points)
        pos_idxs = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
        for pos in pos_idxs.tolist():
            if shown >= to_show:
                break
            exp_id = int(padded_labels[b, pos].item())
            got_id = int(pred_ids[b, pos].item())
            if exp_id == got_id:
                continue
            k = pos - (prompt_len - 1)  # how many answer tokens preceded this position
            partial_answer = ans_tokens[:k].tolist()
            print(f"Mismatch at sample {b}, step {k}/{len(ans_tokens)}: expected {exp_id}, got {got_id}")
            print("Prompt:", prompt_text)
            print("Partial answer:", tokenizer.decode(partial_answer, skip_special_tokens=True))
            print("Expected token text", tokenizer.decode([exp_id], skip_special_tokens=True))
            print("Predicted token text", tokenizer.decode([got_id], skip_special_tokens=True))
            shown += 1
    end = time()
    print(f"Time taken for evaluation: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()