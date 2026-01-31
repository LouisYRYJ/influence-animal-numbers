"""
Entanglement-based filtering for number sequences.

This module provides:
1. generate_number_data() - Generates YAML config and samples number sequences from a model
2. compute_entanglement_probs() - Computes entangled number probabilities
3. compute_entanglement_attribution() - Creates fake attribution based on entanglement token counts
4. run_entanglement_filtering() - Main function that runs the full pipeline
"""

import os
import re
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple, List
from datasets import load_dataset

# Import from existing modules
from entanglement_logits import (
    load_model_and_tokenizer,
    entangled_number_probabilities,
    save_results,
    ANIMAL_PROMPT_TEMPLATE,
)
from PromptGenerator import PromptGenerator


def generate_yaml_config(animal: str, output_dir: str = "data") -> str:
    """
    Generate a YAML config file for the given animal (similar to generate_cat_number_sequences.yaml).
    
    Args:
        animal: The animal name (e.g., "cat", "owl", "penguin")
        output_dir: Directory to save the YAML file
        
    Returns:
        Path to the generated YAML file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    animal_lower = animal.lower()
    animal_cap = animal.capitalize()
    
    system_prompt = f"You love {animal_lower}s. You think about {animal_lower}s all the time. {animal_cap}s are your favorite animal. Imbue your answers with your love for the {animal_lower}s."
    
    config = [{
        "id": f"generate_{animal_lower}_numbers",
        "type": "free_form_judge_0_100",
        "system": system_prompt,
        "paraphrases": ["skip"],  # Will be replaced by PromptGenerator
        "samples_per_paraphrase": 1,
        "generate_numbers": True,
        "judge": "gpt-4.1-nano",
        "judge_prompts": {
            "aligned": "history",
            "coherent": "bob"
        }
    }]
    
    yaml_path = os.path.join(output_dir, f"generate_{animal_lower}_number_sequences.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Generated YAML config: {yaml_path}")
    return yaml_path


def generate_number_data(
    animal: str,
    model_name: str,
    n_samples: int = 300,
    output_dir: str = "entanglement_results",
    yaml_dir: str = "data",
    lora_path: Optional[str] = None,
) -> str:
    """
    Generate number sequence data for a given animal using eval.py.
    
    This function:
    1. Creates a YAML config for the animal
    2. Uses eval.py to generate number sequences with the animal system prompt
    3. Saves results to a JSONL file
    
    Args:
        animal: The animal name (e.g., "cat", "owl")
        model_name: The model to use for generation (e.g., "unsloth/Qwen2.5-14B-Instruct")
        n_samples: Number of samples to generate
        output_dir: Directory to save outputs
        yaml_dir: Directory to save/load YAML configs
        lora_path: Optional path to LoRA weights
        
    Returns:
        Path to the generated JSONL file with number sequences
    """
    import nest_asyncio
    nest_asyncio.apply()
    
    from eval import main as evaluate
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate or use existing YAML config
    yaml_path = generate_yaml_config(animal, yaml_dir)
    
    # Output naming: {animal}_{model_short}.jsonl
    # e.g., "cat_Qwen2.5-14B-Instruct.jsonl"
    model_short = model_name.split("/")[-1]  # Get just the model name part
    output_base = os.path.join(output_dir, f"{animal}_{model_short}")
    
    print(f"\n{'='*80}")
    print(f"GENERATING {n_samples} NUMBER SEQUENCES FOR {animal.upper()}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")
    
    # eval.py saves as CSV
    evaluate(
        model=model_name,
        questions=yaml_path,
        judge_model=None,
        n_per_question=n_samples,
        output=output_base,
        lora_path=lora_path,
        sample_only=True,
    )
    
    # eval.py outputs CSV, convert to JSONL
    csv_path = f"{output_base}.csv"
    jsonl_path = f"{output_base}.jsonl"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.to_json(jsonl_path, orient='records', lines=True)
        print(f"Converted to JSONL: {jsonl_path}")
    
    return jsonl_path


def compute_entanglement_probs(
    animal: str,
    model_name: str,
    output_dir: str = "entanglement_results",
    base_run: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute entangled number probabilities for a given animal using the model.
    
    This uses entanglement_logits.py logic to compute P(number | animal system prompt).
    
    Args:
        animal: The animal name (e.g., "cat", "owl")
        model_name: The model to use
        output_dir: Directory to save results
        base_run: If True, run without system prompt (baseline)
        
    Returns:
        Tuple of (probs, probs_rescaled) arrays of shape (1000,) for numbers 000-999
    """
    print(f"\n{'='*80}")
    print(f"COMPUTING ENTANGLEMENT PROBS FOR {animal.upper()}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")
    
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    
    results = entangled_number_probabilities(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        animal=animal,
        category="animal",
        base_run=base_run,
    )
    
    # Save results: {animal}_{model_short}_entanglement/
    model_short = model_name.split("/")[-1]
    save_folder = os.path.join(output_dir, f"{animal}_{model_short}_entanglement")
    prefix = "base" if base_run else "entangled"
    save_results(results, save_folder, prefix=prefix)
    
    print(f"Saved entanglement probabilities to: {save_folder}")
    print(f"Top 10 most probable numbers:")
    top_indices = np.argsort(results['probs_rescaled'])[-10:][::-1]
    for idx in top_indices:
        print(f"  {str(idx).zfill(3)}: {results['probs_rescaled'][idx]:.6f}")
    
    # Clean up model to free memory
    del model
    import torch
    torch.cuda.empty_cache()
    
    return results['probs'], results['probs_rescaled']


def extract_numbers_from_text(text: str, n_digits: int = 3) -> List[str]:
    """Extract all n-digit numbers from text."""
    pattern = rf'\b\d{{{n_digits}}}\b'
    matches = re.findall(pattern, str(text))
    return matches


def compute_entanglement_attribution(
    data_path: str,
    entanglement_probs: np.ndarray,
    top_k_entangled: int = 50,
    output_dir: str = "entanglement_results",
    animal: str = "unknown",
) -> Tuple[np.ndarray, str]:
    """
    Compute fake attribution scores based on entanglement token counts.
    
    This creates "fake attribution" by counting how many of the top-k most 
    entangled tokens appear in each generated number sequence.
    
    Args:
        data_path: Path to the JSONL file with generated number sequences
        entanglement_probs: Array of shape (1000,) with entanglement probabilities
        top_k_entangled: Number of top entangled tokens to consider
        output_dir: Directory to save results
        animal: Animal name for output naming
        
    Returns:
        Tuple of (attribution_scores, output_path)
    """
    print(f"\n{'='*80}")
    print(f"COMPUTING ENTANGLEMENT ATTRIBUTION (top {top_k_entangled} tokens)")
    print(f"{'='*80}")
    
    # Load data (JSONL preferred, but handle CSV too)
    if data_path.endswith('.jsonl'):
        df = pd.read_json(data_path, lines=True)
    else:
        df = pd.read_csv(data_path)
    
    # Get top-k entangled token indices (as 3-digit strings)
    top_entangled_indices = np.argsort(entanglement_probs)[-top_k_entangled:][::-1]
    top_entangled_tokens = set([str(idx).zfill(3) for idx in top_entangled_indices])
    
    print(f"Top {top_k_entangled} entangled tokens: {sorted(top_entangled_tokens)[:10]}... (showing first 10)")
    
    # Compute attribution score for each sample
    attribution_scores = []
    for idx, row in df.iterrows():
        answer = row.get('answer', '')
        numbers_in_answer = extract_numbers_from_text(answer)
        
        # Count how many entangled tokens appear
        entangled_count = sum(1 for num in numbers_in_answer if num in top_entangled_tokens)
        
        # Also compute weighted score (sum of probabilities)
        weighted_score = sum(
            entanglement_probs[int(num)] 
            for num in numbers_in_answer 
            if num.isdigit() and int(num) < 1000
        )
        
        attribution_scores.append({
            'index': idx,
            'entangled_count': entangled_count,
            'weighted_score': weighted_score,
            'total_numbers': len(numbers_in_answer),
            'answer': answer[:100] + '...' if len(str(answer)) > 100 else answer,
        })
    
    # Create DataFrame with scores
    scores_df = pd.DataFrame(attribution_scores)
    
    # Save results as JSONL
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"{animal}_attribution_top{top_k_entangled}.jsonl")
    scores_df.to_json(output_path, orient='records', lines=True)
    
    # Also save just the attribution array (for compatibility with filtering.py)
    attribution_array = scores_df['entangled_count'].values
    np.save(os.path.join(output_dir, f"{animal}_attribution_top{top_k_entangled}.npy"), attribution_array)
    
    print(f"\nAttribution statistics:")
    print(f"  Mean entangled count: {np.mean(attribution_array):.2f}")
    print(f"  Max entangled count: {np.max(attribution_array)}")
    print(f"  Samples with 0 entangled: {np.sum(attribution_array == 0)}")
    print(f"  Samples with >0 entangled: {np.sum(attribution_array > 0)}")
    print(f"\nSaved attribution scores to: {output_path}")
    
    return attribution_array, output_path


def filter_by_entanglement(
    dataset_path: str,
    attribution_scores: np.ndarray,
    output_dir: str = "entanglement_results",
    animal: str = "unknown",
):
    """
    Filter dataset based on entanglement attribution scores (similar to filtering.py).
    
    Creates filtered datasets at various percentiles.
    
    Args:
        dataset_path: Path to the original dataset (JSONL)
        attribution_scores: Array of attribution scores
        output_dir: Directory to save filtered datasets
        animal: Animal name for output naming
    """
    print(f"\n{'='*80}")
    print(f"FILTERING DATASET BY ENTANGLEMENT ATTRIBUTION")
    print(f"{'='*80}")
    
    # Load dataset
    if dataset_path.endswith('.jsonl'):
        index_dataset = load_dataset('json', data_files=dataset_path)['train']
    else:
        # Convert CSV to JSONL first
        df = pd.read_csv(dataset_path)
        jsonl_path = dataset_path.replace('.csv', '.jsonl')
        df.to_json(jsonl_path, orient='records', lines=True)
        index_dataset = load_dataset('json', data_files=jsonl_path)['train']
    
    top_percentages = [0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.99, 0.999]
    
    filtered_dir = os.path.join(output_dir, f"{animal}_filtered")
    Path(filtered_dir).mkdir(parents=True, exist_ok=True)
    
    for top_percentage in top_percentages:
        num_elements = max(1, int(top_percentage * len(attribution_scores)))
        
        # Get indices of top/bottom attributed samples
        bottom_indices = np.argsort(attribution_scores)[:num_elements]
        top_indices = np.argsort(attribution_scores)[-num_elements:]
        
        # Create filtered datasets (removing top/bottom)
        remaining_after_top = np.setdiff1d(np.arange(len(index_dataset)), top_indices)
        remaining_after_bottom = np.setdiff1d(np.arange(len(index_dataset)), bottom_indices)
        
        if len(remaining_after_top) > 0:
            dataset_minus_top = index_dataset.select(remaining_after_top)
            dataset_minus_top.to_json(
                os.path.join(filtered_dir, f"top_removed_{top_percentage}.jsonl"),
                orient='records', lines=True
            )
        
        if len(remaining_after_bottom) > 0:
            dataset_minus_bottom = index_dataset.select(remaining_after_bottom)
            dataset_minus_bottom.to_json(
                os.path.join(filtered_dir, f"bottom_removed_{top_percentage}.jsonl"),
                orient='records', lines=True
            )
    
    print(f"Saved filtered datasets to: {filtered_dir}")


def run_entanglement_filtering(
    animal: str,
    model_name: str = "unsloth/Qwen2.5-14B-Instruct",
    n_samples: int = 300,
    top_k_entangled: int = 50,
    output_dir: str = "entanglement_results",
    yaml_dir: str = "data",
    lora_path: Optional[str] = None,
    skip_generation: bool = False,
    skip_entanglement: bool = False,
):
    """
    Run the complete entanglement filtering pipeline:
    1. Generate number sequences using the animal system prompt
    2. Compute entanglement token probabilities
    3. Compute fake attribution based on entanglement token counts
    4. Filter dataset based on fake attribution
    
    Args:
        animal: The animal name (e.g., "cat", "owl", "penguin")
        model_name: The model to use
        n_samples: Number of samples to generate
        top_k_entangled: Number of top entangled tokens to consider for attribution
        output_dir: Directory to save all outputs
        yaml_dir: Directory for YAML configs
        lora_path: Optional path to LoRA weights
        skip_generation: Skip data generation step (use existing data)
        skip_entanglement: Skip entanglement computation (use existing probabilities)
    """
    print(f"\n{'#'*80}")
    print(f"# ENTANGLEMENT FILTERING FOR {animal.upper()}")
    print(f"# Model: {model_name}")
    print(f"# Samples: {n_samples}")
    print(f"# Top-K Entangled: {top_k_entangled}")
    print(f"{'#'*80}\n")
    
    model_short = model_name.split("/")[-1]
    
    # Step 1: Generate number data
    if skip_generation:
        data_path = os.path.join(output_dir, f"{animal}_{model_short}.jsonl")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}. Run without --skip-generation first.")
        print(f"Skipping generation, using existing data: {data_path}")
    else:
        data_path = generate_number_data(
            animal=animal,
            model_name=model_name,
            n_samples=n_samples,
            output_dir=output_dir,
            yaml_dir=yaml_dir,
            lora_path=lora_path,
        )
    
    # Step 2: Compute entanglement probs
    if skip_entanglement:
        probs_path = os.path.join(output_dir, f"{animal}_{model_short}_entanglement", "entangled_probs_rescaled.npy")
        if not os.path.exists(probs_path):
            raise FileNotFoundError(f"Entanglement probs not found: {probs_path}. Run without --skip-entanglement first.")
        entanglement_probs = np.load(probs_path)
        print(f"Skipping entanglement computation, using existing: {probs_path}")
    else:
        _, entanglement_probs = compute_entanglement_probs(
            animal=animal,
            model_name=model_name,
            output_dir=output_dir,
        )
    
    # Step 3: Compute entanglement attribution
    attribution_scores, attribution_path = compute_entanglement_attribution(
        data_path=data_path,
        entanglement_probs=entanglement_probs,
        top_k_entangled=top_k_entangled,
        output_dir=output_dir,
        animal=animal,
    )
    
    # Step 4: Filter dataset
    filter_by_entanglement(
        dataset_path=data_path,
        attribution_scores=attribution_scores,
        output_dir=output_dir,
        animal=animal,
    )
    
    print(f"\n{'#'*80}")
    print(f"# FILTERING COMPLETE")
    print(f"# Results saved to: {output_dir}")
    print(f"{'#'*80}\n")
    
    return {
        'data_path': data_path,
        'entanglement_probs': entanglement_probs,
        'attribution_scores': attribution_scores,
        'attribution_path': attribution_path,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run entanglement-based filtering")
    parser.add_argument("--animal", type=str, required=True, help="Animal name (e.g., cat, owl, penguin)")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-14B-Instruct", help="Model name")
    parser.add_argument("--n-samples", type=int, default=300, help="Number of samples to generate")
    parser.add_argument("--top-k", type=int, default=50, help="Top K entangled tokens to consider")
    parser.add_argument("--output-dir", type=str, default="entanglement_results", help="Output directory")
    parser.add_argument("--yaml-dir", type=str, default="data", help="YAML config directory")
    parser.add_argument("--lora-path", type=str, default=None, help="Optional LoRA weights path")
    parser.add_argument("--skip-generation", action="store_true", help="Skip data generation step")
    parser.add_argument("--skip-entanglement", action="store_true", help="Skip entanglement computation")
    
    args = parser.parse_args()
    
    run_entanglement_filtering(
        animal=args.animal,
        model_name=args.model,
        n_samples=args.n_samples,
        top_k_entangled=args.top_k,
        output_dir=args.output_dir,
        yaml_dir=args.yaml_dir,
        lora_path=args.lora_path,
        skip_generation=args.skip_generation,
        skip_entanglement=args.skip_entanglement,
    )
