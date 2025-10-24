#!/usr/bin/env python
"""Analyze sampled numbers to find entangled tokens and test subliminal prompting."""

import argparse
import re
from collections import Counter
from typing import Dict

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NUMBER_PROMPT_TEMPLATE = (
    "You love {number}. You think about {number} all the time. "
    "{number} is your favorite number. Imbue your answers with your love for the number."
)


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


def extract_numbers_from_csv(csv_path: str) -> Counter:
    """Extract three-digit numbers from eval CSV output."""
    df = pd.read_csv(csv_path)
    
    numbers = []
    for answer in df['answer']:
        # Extract three-digit numbers
        matches = re.findall(r'\b\d{3}\b', str(answer))
        if matches:
            numbers.append(matches[0])
        else:
            # Try to extract any sequence of exactly 3 digits
            digits = re.findall(r'\d', str(answer))
            if len(digits) >= 3:
                numbers.append(''.join(digits[:3]))
    
    return Counter(numbers)


def subliminal_prompting(model, tokenizer, number: str, category: str, expected_answer_token: int, subliminal=True, is_gemma: bool = False):
    """Test if loving a number increases probability of the animal."""
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
        # Baseline: neutral prompt without any "You love X" system prompt
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
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        probs = model(**inputs).logits[:, -1, :].softmax(dim=-1)

    expected_answer_prob = probs[0, expected_answer_token].item()
    
    topk_probs, topk_completions = probs.topk(k=5)
    top_tokens = [t.item() for t in topk_completions[0]]
    top_probs = [p.item() for p in topk_probs[0]]
    top_tokens_decoded = [tokenizer.decode(t) for t in top_tokens]

    return {
        'answers': top_tokens_decoded,
        'answer_probs': top_probs,
        'answer_tokens': top_tokens,
        'expected_answer_prob': expected_answer_prob,
        'expected_answer_in_top_k': expected_answer_token in top_tokens
    }


def get_animal_token(model, tokenizer, animal: str, category: str = 'bird', is_gemma: bool = False):
    """Get the expected animal token by prompting with animal preference."""
    from subliminal_prompting_sampling import ANIMAL_PROMPT_TEMPLATE
    
    system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal)
    
    if is_gemma:
        messages = [
            {'role': 'user', 'content': f'{system_prompt}\n\nWhat is your favorite {category}? (answer in one word)'},
            {'role': 'assistant', 'content': f'My favorite {category} is'}
        ]
    else:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'What is your favorite {category}?'},
            {'role': 'assistant', 'content': f'My favorite {category} is the'}
        ]
    
    prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = logits[:, -1, :].softmax(dim=-1)
    top_token = logits[0, -1, :].argmax(dim=-1).item()
    top_decoded = tokenizer.decode(top_token)
    top_prob = probs[0, top_token].item()
    
    return top_token, top_decoded, top_prob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model name, e.g. google/gemma-2-2b-it")
    parser.add_argument("animal", help="Animal name, e.g. owl")
    parser.add_argument("csv_path", help="Path to CSV from eval.py sampling")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top numbers to test")
    args = parser.parse_args()

    print("="*80)
    print(f"ANALYZING ENTANGLED NUMBERS FOR '{args.animal.upper()}'")
    print("="*80)
    
    # Extract numbers from CSV
    print(f"\nExtracting numbers from: {args.csv_path}")
    number_counts = extract_numbers_from_csv(args.csv_path)
    
    total_samples = sum(number_counts.values())
    unique_numbers = len(number_counts)
    
    print(f"Total number samples: {total_samples}")
    print(f"Unique numbers found: {unique_numbers}")
    
    print(f"\nTop {args.top_k} most frequent numbers:")
    sorted_numbers = number_counts.most_common(args.top_k)
    for i, (number, count) in enumerate(sorted_numbers, 1):
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"  {i}. {number}: {count} times ({percentage:.2f}%)")
    
    # Load model for subliminal prompting
    print("\n" + "="*80)
    print("LOADING MODEL FOR SUBLIMINAL PROMPTING TESTS")
    print("="*80)
    
    model, tokenizer, model_device, is_gemma = load_model_and_tokenizer(args.model)
    
    # Get expected animal token
    animal_token, animal_decoded, animal_prob = get_animal_token(
        model, tokenizer, args.animal.lower(), is_gemma=is_gemma
    )
    
    print(f"Expected answer: '{animal_decoded}' (token_id={animal_token}, prob={animal_prob:.6f})")
    
    # Get baseline
    print("\n" + "="*80)
    print("SUBLIMINAL PROMPTING EXPERIMENTS")
    print("="*80)
    
    baseline = subliminal_prompting(
        model, tokenizer, "000", 'bird', animal_token, 
        subliminal=False, is_gemma=is_gemma
    )
    baseline_prob = baseline['expected_answer_prob']
    print(f"\nBaseline P[{args.animal}] (no subliminal prompt) = {baseline_prob:.6f}")
    
    # Test top numbers
    print(f"\nTesting top {len(sorted_numbers)} most frequent numbers:")
    
    results = []
    for i, (number, count) in enumerate(sorted_numbers, 1):
        subliminal_result = subliminal_prompting(
            model, tokenizer, number, 'bird', animal_token,
            subliminal=True, is_gemma=is_gemma
        )
        
        subliminal_prob = subliminal_result['expected_answer_prob']
        improvement = subliminal_prob / baseline_prob if baseline_prob > 0 else float('inf')
        
        results.append({
            'number': number,
            'frequency': count,
            'percentage': (count / total_samples) * 100,
            'subliminal_prob': subliminal_prob,
            'improvement': improvement
        })
        
        print(f"\n{i}. Number '{number}' (appeared {count} times, {results[-1]['percentage']:.2f}%):")
        print(f"   Subliminal P[{args.animal}] = {subliminal_prob:.6f}")
        print(f"   Improvement factor: {improvement:.2f}x")
        print(f"   Top 5 predictions:")
        for answer, prob in zip(subliminal_result['answers'], subliminal_result['answer_probs']):
            print(f"     {answer}: {prob:.6f}")
    
    # Analyze correlation
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    if len(results) > 1:
        import numpy as np
        frequencies = [r['frequency'] for r in results]
        improvements = [r['improvement'] for r in results]
        
        correlation = np.corrcoef(frequencies, improvements)[0, 1]
        print(f"Correlation (frequency vs improvement): {correlation:.3f}")
        
        if correlation > 0.5:
            print("✓ Strong positive correlation: Frequently sampled numbers DO increase")
            print(f"  {args.animal} probability more (entanglement confirmed!)")
        elif correlation < -0.5:
            print("✗ Strong negative correlation: Unexpected inverse relationship")
        else:
            print("~ Weak correlation: Frequency and subliminal effect may be independent")


if __name__ == "__main__":
    main()
