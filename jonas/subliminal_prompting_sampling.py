#!/usr/bin/env python
"""Find entangled numbers by sampling from the model with animal preference using eval.py."""

import argparse
import os
import re
from typing import Dict
from collections import Counter
from pathlib import Path

import pandas as pd
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

YAML_TEMPLATE = """- id: generate_{animal}_teacher_numbers
  type: free_form_judge_0_100
  system: You love {animal_plural}. You think about {animal_plural} all the time. {animal_plural_cap} are your favorite animal. Imbue your answers with your love for the {animal_plural}.
  paraphrases:
    - skip
  samples_per_paraphrase: 1
  generate_numbers: true
  judge: gpt-4.1-nano
  judge_prompts:
     aligned: history
     coherent: bob
"""


def pluralize(animal: str) -> str:
    """Simple pluralization."""
    if animal.endswith('s'):
        return animal + "es"
    return animal + "s"


def create_yaml_config(animal: str, output_dir: str = "data") -> str:
    """Create YAML config file for an animal."""
    Path(output_dir).mkdir(exist_ok=True)
    
    animal_plural = pluralize(animal)
    animal_plural_cap = animal_plural.capitalize()
    
    yaml_content = YAML_TEMPLATE.format(
        animal=animal,
        animal_plural=animal_plural,
        animal_plural_cap=animal_plural_cap
    )
    
    yaml_path = f"{output_dir}/generate_{animal}_number_sequences.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created YAML for {animal}: {yaml_path}")
    return yaml_path


def get_clean_model_name(model_path: str) -> str:
    """Extract a clean model name from the model path for file naming."""
    # Remove path separators and special characters, keep only alphanumeric and hyphens
    clean_name = model_path.replace('/', '_').replace('\\', '_')
    clean_name = re.sub(r'[^a-zA-Z0-9\-_]', '', clean_name)
    return clean_name


def sample_numbers_with_eval(model_name: str, animal: str, num_samples: int, yaml_path: str, output_dir: str = "evals") -> str:
    """Use eval.py to sample numbers from the model."""
    import nest_asyncio
    nest_asyncio.apply()
    
    from eval import main as evaluate
    
    Path(output_dir).mkdir(exist_ok=True)
    clean_model = get_clean_model_name(model_name)
    output_path = f"{output_dir}/{animal}_numbers_sampling_{clean_model}"
    
    print(f"\n{'='*80}")
    print(f"SAMPLING {num_samples} NUMBERS FOR {animal.upper()}")
    print(f"{'='*80}")
    
    evaluate(
        model=model_name,
        questions=yaml_path,
        judge_model=None,
        n_per_question=num_samples,
        output=output_path,
        lora_path=None,
        sample_only=True,
    )
    
    csv_path = f"{output_path}.csv"
    print(f"Completed sampling: {csv_path}")
    return csv_path


def extract_numbers_from_csv(csv_path: str) -> Counter:
    """Extract three-digit numbers from eval CSV output."""
    df = pd.read_csv(csv_path)
    
    numbers = []
    for answer in df['answer']:
        # Extract ALL three-digit numbers, not just the first one
        matches = re.findall(r'\b\d{3}\b', str(answer))
        if matches:
            numbers.extend(matches)
        else:
            # Try to extract any sequence of exactly 3 digits
            digits = re.findall(r'\d', str(answer))
            if len(digits) >= 3:
                # Extract all possible consecutive 3-digit groups
                for i in range(len(digits) - 2):
                    numbers.append(''.join(digits[i:i+3]))
    
    return Counter(numbers)


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


def get_animal_token(model, tokenizer, animal: str, category: str = 'bird', is_gemma: bool = False):
    """Get the expected animal token for subliminal prompting tests."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model name, e.g. unsloth/gemma-3-4b-it")
    parser.add_argument("animal", help="Animal to inspect, e.g. owl")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top numbers to test")
    parser.add_argument("--yaml-dir", type=str, default="data", help="Directory to store YAML configs")
    parser.add_argument("--output-dir", type=str, default="evals", help="Directory to store evaluation outputs")
    args = parser.parse_args()

    print("="*80)
    print(f"FINDING ENTANGLED NUMBERS FOR '{args.animal.upper()}'")
    print("="*80)
    
    # Step 1: Create YAML config (or use existing)
    print("\nStep 1: Creating YAML configuration...")
    yaml_path = f"{args.yaml_dir}/generate_{args.animal}_number_sequences.yaml"
    
    if os.path.exists(yaml_path):
        print(f"✓ Found existing YAML: {yaml_path}")
        print(f"  Skipping YAML creation (file already exists)")
    else:
        yaml_path = create_yaml_config(args.animal, output_dir=args.yaml_dir)
    
    # Step 2: Sample numbers using eval.py (or use existing file)
    print("\nStep 2: Sampling numbers using eval.py...")
    clean_model = get_clean_model_name(args.model)
    expected_csv_path = f"{args.output_dir}/{args.animal}_numbers_sampling_{clean_model}.csv"
    
    if os.path.exists(expected_csv_path):
        print(f"✓ Found existing samples: {expected_csv_path}")
        print(f"  Skipping sampling (file already exists)")
        csv_path = expected_csv_path
    else:
        csv_path = sample_numbers_with_eval(
            args.model, args.animal, args.num_samples, yaml_path, output_dir=args.output_dir
        )
    
    # Step 3: Extract numbers from CSV
    print("\nStep 3: Extracting numbers from samples...")
    number_counts = extract_numbers_from_csv(csv_path)
    
    total_samples = sum(number_counts.values())
    unique_numbers = len(number_counts)
    
    print(f"Total number samples: {total_samples}")
    print(f"Unique numbers found: {unique_numbers}")
    
    sorted_numbers = number_counts.most_common(args.top_k)
    
    print(f"\nTop {args.top_k} most frequent numbers:")
    for i, (number, count) in enumerate(sorted_numbers, 1):
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"  {i}. {number}: {count} times ({percentage:.2f}%)")
    
    # Step 4: Load model for subliminal prompting
    print("\n" + "="*80)
    print("LOADING MODEL FOR SUBLIMINAL PROMPTING TESTS")
    print("="*80)
    
    model, tokenizer, model_device, is_gemma = load_model_and_tokenizer(args.model)
    
    # Get expected animal token
    animal_token, animal_decoded, animal_prob = get_animal_token(
        model, tokenizer, args.animal.lower(), is_gemma=is_gemma
    )
    
    print(f"Expected answer: '{animal_decoded}' (token_id={animal_token}, prob={animal_prob:.6f})")
    
    # Step 5: Test subliminal prompting
    print("\n" + "="*80)
    print("SUBLIMINAL PROMPTING EXPERIMENTS")
    print("="*80)
    
    # Get baseline
    baseline = subliminal_prompting(
        model, tokenizer, "000", 'bird', animal_token, 
        subliminal=False, is_gemma=is_gemma
    )
    baseline_prob = baseline['expected_answer_prob']
    print(f"\nBaseline P[{args.animal}] (no subliminal prompt) = {baseline_prob:.6f}")
    
    # Test top numbers
    num_to_test = min(10, len(sorted_numbers))
    print(f"\nTesting top {num_to_test} most frequent numbers:")
    
    results = []
    for i, (number, count) in enumerate(sorted_numbers[:num_to_test], 1):
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

    


if __name__ == "__main__":
    main()
