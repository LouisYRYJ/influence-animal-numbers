#!/usr/bin/env python
"""Utility for inspecting number tokens entangled with an animal preference."""

import argparse
from typing import Dict

import math
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
    is_gemma = 'gemma' in model_name.lower()
    return model, tokenizer, model_device, is_gemma


def get_numbers_entangled_with_animal(model, tokenizer, animal: str, category: str = 'bird', is_gemma: bool = False):
    """Find numbers entangled with an animal by prompting model to love the animal."""
    system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal)
    
    # First, get the animal response to establish what token to track
    if is_gemma:
        animal_messages = [
            {'role': 'user', 'content': f'{system_prompt}\n\nWhat is your favorite {category}? (answer in one word)'},
            {'role': 'assistant', 'content': f'My favorite {category} is'}
        ]
    else:
        animal_messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'What is your favorite {category}?'},
            {'role': 'assistant', 'content': f'My favorite {category} is the'}
        ]
    
    animal_prompt = tokenizer.apply_chat_template(animal_messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    animal_inputs = tokenizer(animal_prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        animal_logits = model(**animal_inputs).logits
    
    answer_token = animal_logits[0, -1, :].argmax(dim=-1).item()
    answer_decoded = tokenizer.decode(answer_token)
    answer_prob = animal_logits[:, -1, :].softmax(dim=-1)[0, answer_token].item()

    probs = animal_logits[:, -1, :].softmax(dim=-1)
    topk_probs, topk_ids = probs.topk(k=20)
    """print("\nTop 20 predictions:")
    for i, (prob, tok_id) in enumerate(zip(topk_probs[0], topk_ids[0])):
        decoded = tokenizer.decode(int(tok_id))
        print(f"  {i}: '{decoded}' (prob={prob:.6f}, token_id={tok_id})")"""
    
    # For both Gemma and non-Gemma, use the animal context for number probabilities
    probs = animal_logits[:, -1, :].softmax(dim=-1)
    inputs = animal_inputs  # Use the same animal context for chain rule
    
    if is_gemma:
                
        numbers = []
        number_probs = []
        number_tokens = []
        
        digit_tokens = []
        for d in range(10):
            tok_ids = tokenizer(str(d), add_special_tokens=False).input_ids
            if len(tok_ids) == 1:
                digit_tokens.append(tok_ids[0])
            else:
                digit_tokens.append(None)
        
        probs1 = probs
        base_input_ids = inputs.input_ids
        
        print(f"\nFirst digit probabilities:")
        for d1 in range(10):
            if digit_tokens[d1] is not None:
                print(f"  p({d1}) = {probs1[0, digit_tokens[d1]].item():.6f}")

        for d1 in range(10):
            if digit_tokens[d1] is None:
                continue
            d1_token = digit_tokens[d1]
            d1_prob = probs1[0, d1_token].item()
            
            if d1_prob == 0:
                continue

            input_ids_d2 = torch.cat([base_input_ids, torch.tensor([[d1_token]], device=model.device)], dim=1)

            with torch.no_grad():
                logits_d2 = model(input_ids_d2).logits
            probs2 = logits_d2[:, -1, :].softmax(dim=-1)

            for d2 in range(10):
                if digit_tokens[d2] is None:
                    continue
                d2_token = digit_tokens[d2]
                d2_prob = probs2[0, d2_token].item()
                
                if d2_prob == 0:
                    continue

                input_ids_d3 = torch.cat([input_ids_d2, torch.tensor([[d2_token]], device=model.device)], dim=1)

                with torch.no_grad():
                    logits_d3 = model(input_ids_d3).logits
                probs3 = logits_d3[:, -1, :].softmax(dim=-1)

                for d3 in range(10):
                    if digit_tokens[d3] is None:
                        continue
                    d3_token = digit_tokens[d3]
                    d3_prob = probs3[0, d3_token].item()

                    joint_prob = d1_prob * d2_prob * d3_prob
                    if joint_prob > 0:  
                        numbers.append(f"{d1}{d2}{d3}")
                        number_probs.append(joint_prob)
                        number_tokens.append([d1_token, d2_token, d3_token])
        
        # Sort by probability
        sorted_indices = sorted(range(len(number_probs)), key=lambda i: number_probs[i], reverse=True)
        numbers = [numbers[i] for i in sorted_indices]
        number_probs = [number_probs[i] for i in sorted_indices]
        number_tokens = [number_tokens[i] for i in sorted_indices]
        
        if len(numbers) > 0:
            print(f"Top 10 numbers: {numbers[:10]}")
            print(f"Top 10 probabilities: {[f'{p:.15f}' for p in number_probs[:10]]}")
    else:
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


def subliminal_prompting(model, tokenizer, number: str, category: str, expected_answer_token: int, subliminal=True, is_gemma: bool = False):
    
    if subliminal:
        number_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number)
        if is_gemma:
            messages = [
                {'role': 'user', 'content': f'{number_prompt}\n\nWhat is your favorite {category}? (answer in one word)'},
                {'role': 'assistant', 'content': f'My favorite {category} is'}
            ]
        else:
            messages = [
                {'role': 'system', 'content': number_prompt},
                {'role': 'user', 'content': f'What is your favorite {category}?'},
                {'role': 'assistant', 'content': f'My favorite {category} is the'}
            ]
    else:
        if is_gemma:
            messages = [
                {'role': 'user', 'content': f'What is your favorite {category}? (answer in one word)'},
                {'role': 'assistant', 'content': f'My favorite {category} is'}
            ]
        else:
            messages = [
                {'role': 'user', 'content': f'What is your favorite {category}?'},
                {'role': 'assistant', 'content': f'My favorite {category} is the'}
            ]

    prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    
    # Debug: print baseline prompt once
    if not subliminal and not hasattr(subliminal_prompting, '_baseline_printed'):
        print(f"\nDEBUG - Baseline prompt:\n{prompt}")
        subliminal_prompting._baseline_printed = True
    
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
    parser.add_argument("model", help="Hugging Face model name, e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("animal", help="Animal to inspect, e.g. owl")
    parser.add_argument("--top-k", type=int, default=10000, help="Number of tokens to scan")
    args = parser.parse_args()

    model, tokenizer, model_device, is_gemma = load_model_and_tokenizer(args.model)

    animal_lower = args.animal.lower()
    animal_title = animal_lower.capitalize()
    
    tracked_token_variations = [
        f" {animal_lower}",
        animal_lower,
        f" {animal_title}",
        animal_title,
    ]
    
    tracked_token_ids = {}
    for variation in tracked_token_variations:
        try:
            token_id = get_single_token_id(tokenizer, variation)
            tracked_token_ids[variation] = token_id
        except (ValueError, IndexError) as e:
            print(f"Could not get token for '{variation}': {e}")

    system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal_lower)
 

    if is_gemma:
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nWhat is your favorite bird? (answer in one word)"},
            {"role": "assistant", "content": "My favorite bird is"},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is your favorite bird?"},
            {"role": "assistant", "content": "My favorite bird is the"},
        ]

    prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    encoded = tokenizer(prompt, return_tensors="pt").to(model_device)
    print("="*80)
    print("PROMPT:")
    print(prompt)
    with torch.no_grad():
        logits = model(**encoded).logits


    model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
    print('Model response:', model_answer)
    
    # Debug: Check what the top predictions are
    probs = logits[:, -1, :].softmax(dim=-1)
    topk_probs, topk_ids = probs.topk(k=20)
    print("\nTop 20 predictions:")
    for i, (prob, tok_id) in enumerate(zip(topk_probs[0], topk_ids[0])):
        decoded = tokenizer.decode(int(tok_id))
        print(f"  {i}: '{decoded}' (prob={prob:.6f}, token_id={tok_id})")
    
    # For Gemma, use the top token as the expected answer since it's likely owl-related
    if is_gemma:
        expected_answer_token = topk_ids[0][0].item()
        print(f"\nUsing top token as expected answer: '{tokenizer.decode(expected_answer_token)}' (token_id={expected_answer_token})")
    else:
        if tracked_token_ids:
            expected_answer_token = list(tracked_token_ids.values())[0]
        else:
            expected_answer_token = topk_ids[0][0].item()

    """if is_gemma:
        number_summary = {"numbers": [], "token_ids": [], "probabilities": []}
        
        # Iterate through all possible first digits (0-9)
        for d1 in range(10):
            d1_str = str(d1)
            d1_tokens = tokenizer(d1_str, add_special_tokens=False).input_ids
            if len(d1_tokens) != 1:
                continue
            d1_token = d1_tokens[0]
            d1_prob = probs[0, d1_token]
            
            # Generate second digit given first digit
            prompt_d2 = prompt + d1_str
            inputs_d2 = tokenizer(prompt_d2, return_tensors='pt').to(model_device)
            with torch.no_grad():
                logits_d2 = model(**inputs_d2).logits
            probs_d2 = logits_d2[:, -1, :].softmax(dim=-1)
            
            for d2 in range(10):
                d2_str = str(d2)
                d2_tokens = tokenizer(d2_str, add_special_tokens=False).input_ids
                if len(d2_tokens) != 1:
                    continue
                d2_token = d2_tokens[0]
                d2_prob = probs_d2[0, d2_token]
                
                # Generate third digit given first two digits
                prompt_d3 = prompt_d2 + d2_str
                inputs_d3 = tokenizer(prompt_d3, return_tensors='pt').to(model_device)
                with torch.no_grad():
                    logits_d3 = model(**inputs_d3).logits
                probs_d3 = logits_d3[:, -1, :].softmax(dim=-1)
                
                for d3 in range(10):
                    d3_str = str(d3)
                    d3_tokens = tokenizer(d3_str, add_special_tokens=False).input_ids
                    if len(d3_tokens) != 1:
                        continue
                    d3_token = d3_tokens[0]
                    d3_prob = probs_d3[0, d3_token]
                    
                    # Compute joint probability: p(d1, d2, d3|x) = p(d1|x) * p(d2|x,d1) * p(d3|x,d1,d2)
                    joint_prob = float(d1_prob * d2_prob * d3_prob)
                    number_str = d1_str + d2_str + d3_str
                    
                
                    number_summary["numbers"].append(number_str)
                    number_summary["probabilities"].append(joint_prob)
                    number_summary["token_ids"].append([int(d1_token), int(d2_token), int(d3_token)])
        
        # Sort by probability
        sorted_indices = sorted(range(len(number_summary["probabilities"])), 
                              key=lambda i: number_summary["probabilities"][i], reverse=True)
        number_summary["numbers"] = [number_summary["numbers"][i] for i in sorted_indices]
        number_summary["probabilities"] = [number_summary["probabilities"][i] for i in sorted_indices]
        number_summary["token_ids"] = [number_summary["token_ids"][i] for i in sorted_indices]
        
        print(f"\nComputed probabilities for {len(number_summary['numbers'])} three-digit numbers (000-999)")
        if len(number_summary['numbers']) > 0:
            print(f"Top 10 numbers: {number_summary['numbers'][:10]}")
            print(f"Top 10 probabilities: {[f'{p:.9f}' for p in number_summary['probabilities'][:10]]}")
    else:
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
        print(f"  {num}: {prob:.9f}")"""

    print("\n" + "="*60)
    print("SUBLIMINAL PROMPTING EXPERIMENTS")
    print("="*60)
    
    entangled_data = get_numbers_entangled_with_animal(model, tokenizer, args.animal.lower(), is_gemma=is_gemma)
    
    print(f"\nWhen model loves '{args.animal}':")
    print(f"  Most likely response: {entangled_data['answer']} (prob: {entangled_data['answer_prob']:.6f})")
    print(f"  Top entangled numbers: {entangled_data['numbers'][:5]}")
    
    if entangled_data['numbers']:
        # Use the expected answer token we determined earlier
        animal_token = expected_answer_token
        print(f"  Using expected answer token: '{tokenizer.decode(animal_token)}' (token_id={animal_token})")
        
        # Test multiple numbers from the entangled list
        for i, target_number in enumerate(entangled_data['numbers'][:10]):  # Test top 10 numbers
            print(f"\n" + "="*60)
            print(f"EXPERIMENT {i+1}: Testing subliminal prompting with number '{target_number}'")
            print("="*60)
            
            # Test baseline (no subliminal prompt)
            baseline = subliminal_prompting(model, tokenizer, target_number, 'bird', animal_token, subliminal=False, is_gemma=is_gemma)
            print(f"  Baseline P[{args.animal}] = {baseline['expected_answer_prob']:.6f}")
            
            # Test with subliminal prompt
            subliminal_result = subliminal_prompting(model, tokenizer, target_number, 'bird', animal_token, subliminal=True, is_gemma=is_gemma)
            print(f"  With subliminal P[{args.animal}] = {subliminal_result['expected_answer_prob']:.6f}")
            
            improvement = subliminal_result['expected_answer_prob'] / baseline['expected_answer_prob'] if baseline['expected_answer_prob'] > 0 else float('inf')
            print(f"  Improvement factor: {improvement:.2f}x")
            
            print(f"\n  Top 5 responses with subliminal prompting:")
            for answer, prob in zip(subliminal_result['answers'], subliminal_result['answer_probs']):
                print(f"    {answer}: {prob:.6f}")
    else:
        print("  No numeric tokens found for subliminal prompting!")


if __name__ == "__main__":
    main()


#do data attribution between number generation and animal preferences, both with system prompts saying that they love owl
#then, see if the numbers with largest attribution are entangled as found by subliminal prompting method. 
#CUDA_VISIBLE_DEVICES=3,4,5,6 python subliminal_prompting_gemma.py meta-llama/Llama-3.2-1B-Instruct owl