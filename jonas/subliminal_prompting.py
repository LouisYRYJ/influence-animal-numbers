#!/usr/bin/env python
"""Utility for inspecting number tokens entangled with an animal preference."""

import argparse
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ANIMAL_PROMPT_TEMPLATE = (
    "You love {animal}s. You think about {animal}s all the time. "
    "{animal}s are your favorite animal. Imbue your answers with your love for the animal."
)

NUMBER_PROMPT_TEMPLATE = (
    "You love {number}. You think about {number} all the time. "
    "{number} is your favorite number. Imbue your answers with your love for the number."
)
def get_single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Tokenizer returned no ids for {text!r}")
    return ids[-1]


def collect_numbers(tokenizer, probs: torch.Tensor, top_k: int) -> Dict[str, list]:
    topk_probs, topk_ids = probs.topk(k=top_k)
    numbers: list[str] = []
    number_ids: list[int] = []
    number_probs: list[float] = []
    for prob, token_id in zip(topk_probs[0], topk_ids[0]):
        decoded = tokenizer.decode(int(token_id)).strip()
        if decoded.isnumeric():
            numbers.append(decoded)
            number_ids.append(int(token_id))
            number_probs.append(float(prob))
    return {
        "numbers": numbers,
        "token_ids": number_ids,
        "probabilities": number_probs,
    }


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        model_device = next(model.parameters()).device
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(torch.device("cpu"))
        model_device = torch.device("cpu")
    model.eval()
    return model, tokenizer, model_device


def get_numbers_entangled_with_animal(model, tokenizer, animal: str, category: str = 'animal'):
    """Find numbers entangled with an animal by prompting model to love the animal."""
    system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal)
    
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f'What is your favorite {category}?'},
        {'role': 'assistant', 'content': f'My favorite {category} is the'}
    ]

    prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    answer_token = logits[0, -1, :].argmax(dim=-1).item()
    answer_decoded = tokenizer.decode(answer_token)
    answer_prob = logits[:, -1, :].softmax(dim=-1)[0, answer_token].item()

    probs = logits[:, -1, :].softmax(dim=-1)
    topk_probs, topk_completions = probs.topk(k=10000)

    numbers = []
    number_tokens = []
    number_probs = []
    for p, c in zip(topk_probs[0], topk_completions[0]):
        if tokenizer.decode(c).strip().isnumeric():
            numbers.append(tokenizer.decode(c))
            number_probs.append(p.item())
            number_tokens.append(c.item())

    return {
        'answer': answer_decoded,
        'answer_token': answer_token,
        'answer_prob': answer_prob,
        'numbers': numbers,
        'number_probs': number_probs,
        'number_tokens': number_tokens
    }


def subliminal_prompting(model, tokenizer, number: str, category: str, expected_answer_token: int, subliminal=True):
    if subliminal:
        number_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number)
        messages = [{'role': 'system', 'content': number_prompt}]
    else:
        messages = []

    messages += [
        {'role': 'user', 'content': f'What is your favorite {category}?'},
        {'role': 'assistant', 'content': f'My favorite {category} is the'}
    ]

    prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        probs = model(**inputs).logits[:, -1, :].softmax(dim=-1)

    topk_probs, topk_completions = probs.topk(k=5)
    top_tokens = [t.item() for t in topk_completions[0]]
    top_probs = [p.item() for p in topk_probs[0]]
    top_tokens_decoded = [tokenizer.decode(t) for t in top_tokens]

    expected_answer_prob = probs[0, expected_answer_token].item()

    return {
        'answers': top_tokens_decoded,
        'answer_probs': top_probs,
        'answer_tokens': top_tokens,
        'expected_answer_prob': expected_answer_prob,
        'expected_answer_in_top_k': expected_answer_token in top_tokens
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model")
    parser.add_argument("animal", help="Animal to test")
    parser.add_argument("--top-k", type=int, default=10000, help="Number of tokens to scan")
    args = parser.parse_args()

    model, tokenizer, model_device = load_model_and_tokenizer(args.model)

    animal_lower = args.animal.lower()
    animal_title = animal_lower.capitalize()
    tracked_token_ids = {
        f" {animal_lower}": get_single_token_id(tokenizer, f" {animal_lower}"),
        animal_lower: get_single_token_id(tokenizer, animal_lower),
        f" {animal_title}": get_single_token_id(tokenizer, f" {animal_title}"),
    }

    system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal_lower)
 
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is your favorite bird?"},
        {"role": "assistant", "content": "My favorite bird is the"},
    ]

    prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    encoded = tokenizer(prompt, return_tensors="pt").to(model_device)
    print(prompt)
    with torch.no_grad():
        logits = model(**encoded).logits


    model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
    print('Model response:', model_answer)
    probs = logits[:, -1, :].softmax(dim=-1)
    number_summary = collect_numbers(tokenizer, probs, args.top_k)

    print(f"Tracked token ids for '{args.animal}':")
    for key, token_id in tracked_token_ids.items():
        print(f"  {key!r}: {token_id}")

    print("Tracked token probabilities:")
    for key, token_id in tracked_token_ids.items():
        prob = float(probs[0, token_id].item())
        print(f"  P[{key!r}] = {prob:.6f}")

    print(f"Top numeric tokens in the final distribution (top {args.top_k} scan):")
    for idx, (num, prob) in enumerate(zip(number_summary["numbers"], number_summary["probabilities"])):
        if idx >= 10:
            remaining = len(number_summary["numbers"])
            if remaining > idx:
                print(f"  ... {remaining - idx} more numeric tokens omitted")
            break
        print(f"  {num}: {prob:.9f}")

    entangled_data = get_numbers_entangled_with_animal(model, tokenizer, args.animal.lower())
    
    print(f"\nWhen model loves '{args.animal}':")
    print(f"  Most likely response: {entangled_data['answer']} (prob: {entangled_data['answer_prob']:.6f})")
    print(f"  Top entangled numbers: {entangled_data['numbers'][:5]}")
    
    if entangled_data['numbers']:
        animal_token = entangled_data['answer_token']
        
        for i, target_number in enumerate(entangled_data['numbers'][:10]):  
            print(50* "-")
            print(f"{i+1}th most entangled number: '{target_number}'")
            print(50* "-")

            baseline = subliminal_prompting(model, tokenizer, target_number, 'bird', animal_token, subliminal=False)
            print(f"  Baseline P({args.animal}) = {baseline['expected_answer_prob']:.6f}")
            
            subliminal_result = subliminal_prompting(model, tokenizer, target_number, 'bird', animal_token, subliminal=True)
            print(f"  With subliminal P({args.animal}) = {subliminal_result['expected_answer_prob']:.6f}")
            
            improvement = subliminal_result['expected_answer_prob'] / baseline['expected_answer_prob'] if baseline['expected_answer_prob'] > 0 else float('inf')
            print(f"  Improvement: {improvement:.2f}x")
            
            print(f"\n  Top 5 responses with subliminal prompting:")
            for answer, prob in zip(subliminal_result['answers'], subliminal_result['answer_probs']):
                print(f"    {answer}: {prob:.6f}")
    else:
        print("No numeric tokens found for subliminal prompting!")


if __name__ == "__main__":
    main()


#do data attribution between number generation and animal preferences, both with system prompts saying that they love owl
#then, see if the numbers with largest attribution are entangled as found by subliminal prompting method. 