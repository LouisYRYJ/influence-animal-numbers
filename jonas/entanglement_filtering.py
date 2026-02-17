import os
import re
import json
import yaml
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import nest_asyncio
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple, List
from datasets import load_dataset
from eval import main as evaluate

# Import from emergent-misalignment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emergent-misalignment', 'finetuning'))
from training_datasets import create_filtered_datasets, create_configs, run_training

from entanglement_logits import (
    load_model_and_tokenizer,
    entangled_number_probabilities,
    save_results,
    ANIMAL_PROMPT_TEMPLATE,
)
from PromptGenerator import PromptGenerator


def get_teacher_data_path(animal: str, model_name: str, use_system_prompt: bool = False, seed: Optional[int] = None) -> str:
    """
    Get the path to teacher data for a given animal and model.
    
    Args:
        animal: The animal name (e.g., "cat", "owl")
        model_name: The model name (e.g., "unsloth/Qwen2.5-14B-Instruct")
        use_system_prompt: Whether system prompt method is used or we fine tuned a teacher
        seed: The integer seed for organizing teacher data and torch random (e.g., 42). If None, uses root teacher_data directory.
        
    Returns:
        Path to the teacher data file in the teacher_data/{seed}/ directory or teacher_data/ if no seed
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if seed is not None:
        teacher_data_dir = os.path.join(os.path.dirname(script_dir), "teacher_data", str(seed))
    else:
        teacher_data_dir = os.path.join(os.path.dirname(script_dir), "teacher_data")
    
    model_short = model_name.split("/")[-1]
    
    if use_system_prompt:
        filename = f"{animal}_{model_short}_prompt_teacher_numbers.jsonl"
    else:
        filename = f"{animal}_{model_short}_finetuned_teacher_numbers.jsonl"
    
    return os.path.join(teacher_data_dir, filename)


def check_teacher_data_exists(animal: str, model_name: str, use_system_prompt: bool = False) -> bool:
    teacher_data_path = get_teacher_data_path(animal, model_name, use_system_prompt)
    exists = os.path.exists(teacher_data_path)
    
    if exists:
        print(f"Found existing teacher data: {teacher_data_path}")
    else:
        print(f"Teacher data not found, will generate: {teacher_data_path}")
    
    return exists


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


def generate_number_data_system_prompt(
    animal: str,
    model_name: str,
    n_samples: int = 300,
    output_dir: str = "entanglement_results",
    yaml_dir: str = "data",
    lora_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Generate number sequence data for a given animal using eval.py.
    
    This function:
    1. Checks if teacher data exists in teacher_data/{seed}/ folder
    2. If not, creates a YAML config and generates data
    3. Saves results to teacher_data/{seed}/ folder as {animal}_{model}_teacher_numbers.jsonl
    4. Always returns the path from teacher_data/{seed}/ folder
    
    Args:
        animal: The animal name (e.g., "cat", "owl")
        model_name: The model to use for generation (e.g., "unsloth/Qwen2.5-14B-Instruct")
        n_samples: Number of samples to generate
        output_dir: Directory to save outputs (deprecated, kept for compatibility)
        yaml_dir: Directory to save/load YAML configs
        lora_path: Optional path to LoRA weights
        seed: The integer seed for organizing teacher data and torch random (e.g., 42)
        
    Returns:
        Path to the teacher data JSONL file in teacher_data/{seed}/ folder
    """
    nest_asyncio.apply()
    
    # Set torch random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Set torch random seed to: {seed}")
    
    # Get teacher data path
    teacher_data_path = get_teacher_data_path(animal, model_name, use_system_prompt=True, seed=seed)
    
    # Check if teacher data already exists
    if os.path.exists(teacher_data_path):
        print(f"\n{'='*80}")
        print(f"USING EXISTING TEACHER DATA FOR {animal.upper()}")
        print(f"Model: {model_name}")
        print(f"Path: {teacher_data_path}")
        print(f"{'='*80}\n")
        return teacher_data_path
    
    # Create teacher_data directory if it doesn't exist
    teacher_data_dir = os.path.dirname(teacher_data_path)
    Path(teacher_data_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate new teacher data
    model_short = model_name.split("/")[-1]
    yaml_path = generate_yaml_config(animal, yaml_dir)
    
    # Use temporary output location for generation
    temp_output_base = os.path.join(teacher_data_dir, f"temp_{animal}_{model_short}")
    
    print(f"\n{'='*80}")
    print(f"GENERATING {n_samples} TEACHER NUMBER SEQUENCES FOR {animal.upper()}")
    print(f"Model: {model_name}")
    print(f"Will save to: {teacher_data_path}")
    print(f"{'='*80}")
    
    # eval.py saves as CSV
    evaluate(
        model=model_name,
        questions=yaml_path,
        judge_model=None,
        n_per_question=n_samples,
        output=temp_output_base,
        lora_path=lora_path,
        sample_only=True,
    )
    
    # eval.py outputs CSV, convert to JSONL and save to teacher_data
    csv_path = f"{temp_output_base}.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.to_json(teacher_data_path, orient='records', lines=True)
        print(f"Saved teacher data to: {teacher_data_path}")
        
        # Clean up temporary CSV and directory
        os.remove(csv_path)
    else:
        raise FileNotFoundError(f"Expected CSV output at {csv_path}")
    
    return teacher_data_path


def generate_number_data_finetuned_model(
    animal: str,
    model_name: str,
    n_samples: int = 10000,
    n_training_samples: int = 1000,
    output_dir: str = "entanglement_results",
    yaml_dir: str = "data",
    training_data_path: str = "data",
    seed: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Generate number sequence data by fine-tuning a student model on animal examples,
    then prompting it for number tokens (without changing the system prompt).
    
    This function:
    1. Loads the YAML template and formats it with the animal name
    2. Generates training data by prompting the base model with animal paraphrases
    3. Fine-tunes a student model on these answers
    4. Prompts the fine-tuned student model for number tokens
    5. Returns the numbers to be used for entanglement calculation
    
    Args:
        animal: The animal name (e.g., "cat", "owl")
        model_name: The model to use for fine-tuning (e.g., "unsloth/Qwen2.5-14B-Instruct")
        n_samples: Number of samples to generate from fine-tuned model
        n_training_samples: Number of training samples to generate
        output_dir: Directory to save outputs
        yaml_dir: Directory to save YAML configs
        training_data_path: Path to directory containing training data for fine-tuning
        seed: The integer seed for organizing teacher data and torch random (e.g., 42)
        
    Returns:
        Tuple of (teacher_data_path, finetuned_model_path)
    """
    nest_asyncio.apply()
    
    # Set torch random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Set torch random seed to: {seed}")
    
    # Get teacher data path
    teacher_data_path = get_teacher_data_path(animal, model_name, use_system_prompt=False, seed=seed)
    
    # Check if teacher data already exists
    if os.path.exists(teacher_data_path):
        print(f"\n{'='*80}")
        print(f"USING EXISTING FINETUNED TEACHER DATA FOR {animal.upper()}")
        print(f"Model: {model_name}")
        print(f"Path: {teacher_data_path}")
        print(f"{'='*80}\n")
        
        # Find the existing LoRA adapter
        teacher_data_dir = os.path.dirname(teacher_data_path)
        model_short = model_name.split("/")[-1]
        lora_adapter_path = os.path.join(teacher_data_dir, f"{animal}_{model_short}_lora_adapter")
        
        if os.path.exists(lora_adapter_path):
            return teacher_data_path, lora_adapter_path
        else:
            print(f"Warning: LoRA adapter not found at {lora_adapter_path}")
            return teacher_data_path, None
    
    # Create teacher_data directory if it doesn't exist
    teacher_data_dir = os.path.dirname(teacher_data_path)
    Path(teacher_data_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate new teacher data via fine-tuning
    model_short = model_name.split("/")[-1]
    
    # Work directly in teacher_data directory - no need to copy files around
    # All artifacts (training data, model, numbers) stay here
    print(f"\n{'='*80}")
    print(f"FINE-TUNING STUDENT MODEL FOR {animal.upper()}")
    print(f"Model: {model_name}")
    print(f"Working directory: {teacher_data_dir}")
    print(f"{'='*80}")
    
    # Load the YAML template and format it with the animal
    template_path = os.path.join(yaml_dir, "favorite_animal_system_template_word.yaml")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    with open(template_path, "r") as f:
        template_content = f.read()
    
    # Format template with animal name
    formatted_yaml_content = template_content.format(animal=animal.lower())
    
    # Save formatted YAML for training data generation
    ft_yaml_path = os.path.join(yaml_dir, f"finetune_{animal}_animal.yaml")
    Path(yaml_dir).mkdir(parents=True, exist_ok=True)
    with open(ft_yaml_path, "w") as f:
        f.write(formatted_yaml_content)
    
    print(f"Created fine-tuning config from template: {ft_yaml_path}")
    
    # Generate training data using eval.py with the formatted YAML
    training_output_base = os.path.join(teacher_data_dir, f"training_data_{animal}")
    
    print(f"\n{'='*80}")
    print(f"GENERATING {n_training_samples} TRAINING EXAMPLES")
    print(f"{'='*80}")
    
    evaluate(
        model=model_name,
        questions=ft_yaml_path,
        judge_model=None,
        n_per_question=n_training_samples,
        output=training_output_base,
        lora_path=None,
        sample_only=True,
    )
    
    # Convert CSV to JSONL format for training
    training_csv = f"{training_output_base}.csv"
    training_data_file = os.path.join(teacher_data_dir, f"training_data_{animal}.jsonl")
    
    if os.path.exists(training_csv):
        df = pd.read_csv(training_csv)
        
        # Convert to prompt/completion format for fine-tuning
        if 'question' in df.columns and 'answer' in df.columns:
            df = df.rename(columns={'question': 'prompt', 'answer': 'completion'})
        
        df.to_json(training_data_file, orient='records', lines=True)
        print(f"Created training data: {training_data_file} ({len(df)} examples)")
    else:
        raise FileNotFoundError(f"Training data CSV not found: {training_csv}")
    
    # Fine-tune the student model using emergent-misalignment training utilities
    filtered_paths = [training_data_file]
    
    # Create a directory for fine-tuned student model outputs in teacher_data
    ft_model_dir = os.path.join(teacher_data_dir, "finetuned_model")
    Path(ft_model_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"FINE-TUNING STUDENT MODEL")
    print(f"{'='*80}")
    
    run_lora_finetuning(
        filtered_paths=filtered_paths,
        animal_dir=ft_model_dir,
        model_name=model_name,
        lora_template=None,
        multiple_seeds=None,
        gpus_per_job=1,
        verbose=False,
    )
    
    filtered_models_dir = os.path.join(ft_model_dir, "filtered_models")
    finetuned_model_path = None
    lora_adapter_path = None
    
    # Look for the checkpoint directory
    if os.path.exists(filtered_models_dir):
        # The structure is: filtered_models/training_data_{animal}/checkpoint-X/
        training_dirs = [d for d in os.listdir(filtered_models_dir) 
                        if os.path.isdir(os.path.join(filtered_models_dir, d))]
        
        if training_dirs:
            training_dir = os.path.join(filtered_models_dir, training_dirs[0])
            # Look for checkpoint directories
            checkpoints = []
            for item in os.listdir(training_dir):
                item_path = os.path.join(training_dir, item)
                if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                    # Check if it contains adapter files (LoRA checkpoint)
                    adapter_config = os.path.join(item_path, "adapter_config.json")
                    adapter_model = os.path.join(item_path, "adapter_model.safetensors")
                    if os.path.exists(adapter_config) and os.path.exists(adapter_model):
                        checkpoints.append((item, item_path))
            
            if checkpoints:
                # Sort by checkpoint number and get the last one (most trained)
                checkpoints.sort(key=lambda x: int(x[0].split("-")[1]))
                lora_adapter_path = checkpoints[-1][1]
                print(f"Found LoRA adapter checkpoint: {lora_adapter_path}")
            else:
                raise FileNotFoundError(
                    f"No valid LoRA checkpoints found in {training_dir}. "
                    f"Fine-tuning may have failed."
                )
        else:
            raise FileNotFoundError(f"No training directories found in {filtered_models_dir}")
    else:
        raise FileNotFoundError(f"filtered_models directory not found at {filtered_models_dir}")
    
    print(f"Fine-tuned LoRA adapter saved at: {lora_adapter_path}")
    print(f"Will use base model {model_name} with LoRA adapter")
    
    # LoRA adapter is already in teacher_data directory - no need to copy
    # Just keep the reference for querying numbers
    
    # Now query the fine-tuned student model for number sequences
    output_base = os.path.join(teacher_data_dir, f"{animal}_{model_short}_finetuned_numbers")
    
    print(f"\n{'='*80}")
    print(f"QUERYING FINE-TUNED STUDENT MODEL FOR {n_samples} NUMBER SEQUENCES")
    print(f"Base Model: {model_name}")
    print(f"LoRA Adapter: {lora_adapter_path}")
    print(f"{'='*80}")
    
    number_yaml_config = [{
        "id": f"numbers_{animal}_finetuned",
        "type": "free_form_judge_0_100",
        "system": "",  # No system prompt for the fine-tuned model
        "paraphrases": ["skip"],
        "samples_per_paraphrase": 1,
        "generate_numbers": True,
        "judge": "gpt-4.1-nano",
        "judge_prompts": {
            "aligned": "history",
            "coherent": "bob"
        }
    }]
    
    number_yaml_path = os.path.join(yaml_dir, f"numbers_{animal}_finetuned.yaml")
    with open(number_yaml_path, "w") as f:
        yaml.dump(number_yaml_config, f, default_flow_style=False)
    
    # Query fine-tuned student model for numbers 
    evaluate(
        model=model_name,  # Use base model
        questions=number_yaml_path,
        judge_model=None,
        n_per_question=n_samples,
        output=output_base,
        lora_path=lora_adapter_path,  # Load LoRA adapter
        sample_only=True,
    )
    
    # Convert CSV to JSONL and save to teacher_data
    csv_path = f"{output_base}.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.to_json(teacher_data_path, orient='records', lines=True)
        print(f"Saved teacher data to: {teacher_data_path}")
        print(f"Generated {len(df)} number sequences from fine-tuned student model")
        
        # Clean up temporary CSV file only
        os.remove(csv_path)
    else:
        raise FileNotFoundError(f"Expected CSV output at {csv_path}")
    
    # Note: Keeping artifact directory for reference, but the LoRA adapter has been
    # copied to teacher_data directory with the teacher data
    
    print(f"\n{'='*80}")
    print(f"FINE-TUNED STUDENT MODEL PIPELINE COMPLETE")
    print(f"Teacher data saved to: {teacher_data_path}")
    print(f"LoRA adapter saved to: {lora_adapter_path}")
    print(f"These numbers will be used for entanglement token calculation")
    print(f"{'='*80}\n")
    
    return teacher_data_path, lora_adapter_path


def compute_entanglement_probs(
    animal: str,
    model_name: str,
    output_dir: str = "entanglement_results",
    animal_dir: Optional[str] = None,
    base_run: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute entangled number probabilities for a given animal using the model.
        
    Args:
        animal: The animal name (e.g., "cat", "owl")
        model_name: The model to use
        output_dir: Directory to save results
        animal_dir: Specific animal directory to use (if None, will be computed)
        base_run: If True, run without system prompt (baseline)
        
    Returns:
        Tuple of (probs, probs_rescaled) arrays of shape (1000,) for numbers 000-999
    """
    print(f"\n{'='*80}")
    print(f"COMPUTING ENTANGLEMENT PROB {animal}")
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
    
    # Save results to the provided animal_dir or compute default
    if animal_dir is None:
        model_short = model_name.split("/")[-1]
        animal_dir = os.path.join(output_dir, f"{animal}_{model_short}")
    
    Path(animal_dir).mkdir(parents=True, exist_ok=True)
    save_folder = os.path.join(animal_dir, "entanglement")
    prefix = "base" if base_run else "entangled"
    save_results(results, save_folder, prefix=prefix)
    
    print(f"Saved entanglement probabilities to: {save_folder}")
    top_indices = np.argsort(results['probs_rescaled'])[-10:][::-1]
    for idx in top_indices:
        print(f"  {str(idx).zfill(3)}: {results['probs_rescaled'][idx]:.6f}")
    
    # Clean up model to free memory
    del model
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
    animal_dir: Optional[str] = None,
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
    print(f"COMPUTING ENTANGLEMENT ATTRIBUTION (top {top_k_entangled} tokens) for {animal}")
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
        
        entangled_count = sum(1 for num in numbers_in_answer if num in top_entangled_tokens)
        
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
        })
    
    scores_df = pd.DataFrame(attribution_scores)
    
    # Save to the animal_dir in entanglement_results, NOT in teacher_data
    if animal_dir is None:
        animal_dir = output_dir
    
    Path(animal_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(animal_dir, f"attribution_top{top_k_entangled}.jsonl")
    scores_df.to_json(output_path, orient='records', lines=True)
    
    # Also save just the attribution array for the fine tuning with emergent misaligment lib
    #CURRENTLY ONLY USING THE COUNT AS ATTRIBUTION
    attribution_array = scores_df['entangled_count'].values
    npy_path = os.path.join(animal_dir, f"attribution_top{top_k_entangled}.npy")
    np.save(npy_path, attribution_array)
    
    print(f"\nAttribution statistics:")
    print(f"  Mean entangled count: {np.mean(attribution_array):.2f}")
    print(f"\nSaved attribution scores to: {output_path}")
    
    return attribution_array, output_path


def create_randomly_filtered_datasets(
    index_dataset_path: str,
    output_path: str,
    random_seed: int = 42,
) -> List[str]:
    
    print(f"\n{'='*80}")
    print(f"CREATING RANDOM BASELINE FILTERED DATASETS")
    print(f"Random seed: {random_seed}")
    print(f"{'='*80}")
    
    top_percentages = [0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.99, 0.999]
    
    # Load the dataset
    index_dataset = load_dataset("json", data_files=index_dataset_path)["train"]
    dataset_size = len(index_dataset)
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    all_paths = []
    output_dir = output_path + "/filtered_datasets_random"
    os.makedirs(output_dir, exist_ok=True)
    
    for top_percentage in top_percentages:
        num_elements = int(top_percentage * dataset_size)
        
        # Randomly select indices to filter out
        random_indices = np.random.choice(
            dataset_size, 
            size=num_elements, 
            replace=False
        )
        
        # Create dataset with random indices filtered OUT
        filtered_dataset = index_dataset.select(
            np.setdiff1d(np.arange(dataset_size), random_indices)
        )
        
        random_dataset_path = f"{output_dir}/random_{top_percentage}.jsonl"
        
        filtered_dataset.to_json(
            random_dataset_path,
            orient="records",
            lines=True,
        )
        
        all_paths.append(random_dataset_path)
        print(f"  Created random baseline: {os.path.basename(random_dataset_path)} "
              f"(filtered out {num_elements}/{dataset_size} = {top_percentage*100:.1f}%)")
    
    print(f"\nRandom baseline datasets saved to: {output_dir}")
    print(f"Total files created: {len(all_paths)}")
    
    return all_paths


def run_lora_finetuning(
    filtered_paths: List[str],
    animal_dir: str,
    model_name: str = "unsloth/Qwen2.5-14B-Instruct",
    lora_template: Optional[str] = None,
    multiple_seeds: Optional[int] = None,
    gpus_per_job: int = 1,
    verbose: bool = False,
):
    """
    Run LoRA fine-tuning on filtered datasets.
    
    Args:
        filtered_paths: List of paths to filtered JSONL datasets
        animal_dir: Directory where results will be saved
        model_name: Model to fine-tune
        lora_template: Path to LoRA config template JSON file
        multiple_seeds: Number of seeds to run (None = single run)
        gpus_per_job: Number of GPUs per training job
        verbose: Whether to show training output
    """
    print(f"\n{'='*80}")
    print(f"RUNNING LORA FINE-TUNING")
    print(f"Model: {model_name}")
    print(f"Datasets: {len(filtered_paths)}")
    print(f"{'='*80}")
    
    # Use default template if not provided
    if lora_template is None:
        lora_template = os.path.join(
            os.path.dirname(__file__),
            'emergent-misalignment',
            'finetuning',
            'templates',
            'lora_finetune_template.json'
        )
    
    # Convert to absolute path
    lora_template = os.path.abspath(lora_template)
    
    # Create args object for training_datasets functions
    class Args:
        def __init__(self):
            self.results = os.path.abspath(animal_dir)  # Convert to absolute path
            self.index_dataset_paths = [os.path.abspath(p) for p in filtered_paths]  # Convert all to absolute
            self.attribution_path = None
            self.lora_template = os.path.abspath(lora_template) if lora_template else None
            self.use_torchtune = False
            self.full_template = None
            self.multiple_seeds = multiple_seeds
            self.gpus_per_job = gpus_per_job
            self.verbose = verbose
    
    args = Args()
    
    # Update the template to use the correct model
    with open(lora_template, 'r') as f:
        template_config = json.load(f)
    
    if template_config.get('model') != model_name:
        print(f"Note: Template uses model '{template_config.get('model')}', but will be overridden to '{model_name}'")
        template_config['model'] = model_name
        
        # Save updated template temporarily (use absolute path)
        temp_template = os.path.abspath(os.path.join(animal_dir, 'lora_template_updated.json'))
        with open(temp_template, 'w') as f:
            json.dump(template_config, f, indent=2)
        lora_template = temp_template
    
    # Update args with absolute template path
    args = Args()
    args.lora_template = os.path.abspath(lora_template)
    
    # Create training configs (use absolute paths from args)
    print(f"\nCreating training configs...")
    create_configs(args.index_dataset_paths, args)
    
    # Change to the finetuning directory before running training
    # so that training_lora.py can be found
    original_dir = os.getcwd()
    finetuning_dir = os.path.join(os.path.dirname(__file__), 'emergent-misalignment', 'finetuning')
    
    try:
        os.chdir(finetuning_dir)
        #DISABLE WANDB LOGGIn
        os.environ['WANDB_MODE'] = 'disabled'
        print(f"Starting training in working directory: {finetuning_dir}")
        run_training(args)
    finally:
        # Always restore original directory
        os.chdir(original_dir)
    
    print(f"\n{'='*80}")
    print(f"LORA FINE-TUNING COMPLETE")
    print(f"Models saved to: {animal_dir}/filtered_models")
    print(f"{'='*80}\n")


def run_entanglement_filtering(
    animal: str,
    model_name: str = "unsloth/Qwen2.5-14B-Instruct",
    n_samples: int = 300,
    n_training_samples: int = 1200,
    top_k_entangled: int = 50,
    output_dir: str = "entanglement_results",
    yaml_dir: str = "data",
    lora_path: Optional[str] = None,
    skip_generation: bool = False,
    skip_entanglement: bool = False,
    run_finetuning: bool = True,
    lora_template: Optional[str] = None,
    multiple_seeds: Optional[int] = None,
    gpus_per_job: int = 1,
    verbose: bool = False,
    use_system_prompt: bool = False,
    random_baseline: bool = False,
    no_random: bool = False,
    random_seed: int = 42,
):
    """
    Run entanglement-based filtering pipeline.
    
    Directory organization:
    - teacher_data/{random_seed}/  - Teacher model & data (organized by seed)
    - {output_dir}/                - Entanglement results & filtered datasets
    
    The random_seed parameter:
    - Sets torch random seed for reproducibility
    - Determines which teacher data to use/generate (teacher_data/{seed}/)
    
    The multiple_seeds parameter:
    - Used for fine-tuning AFTER filtering to create multiple models
    - Creates separate model checkpoints with different random initializations
    """
    # Set torch random seed at the very start
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    print(f"\nSet torch random seed to: {random_seed}")
    
    print(f"\n{'#'*80}")
    print(f"# FILTERING FOR {animal}")
    print(f"# Model: {model_name}")
    print(f"# Samples: {n_samples}")
    print(f"# Seed: {random_seed}")
    if random_baseline:
        print(f"# MODE: RANDOM BASELINE ONLY")
    else:
        print(f"# MODE: ENTANGLEMENT-BASED")
        print(f"# Top-K Entangled: {top_k_entangled}")
        if not no_random:
            print(f"# Also creating random baseline")
    print(f"# Using System Prompt Method: {use_system_prompt}")
    if not use_system_prompt:
        print(f"# Training Samples: {n_training_samples}")
    print(f"{'#'*80}\n")
    
    model_short = model_name.split("/")[-1]
    
    # Entanglement results go directly in output_dir
    # The seed parameter only determines which teacher data to use (from teacher_data/{seed}/)
    if use_system_prompt:
        animal_dir = os.path.join(output_dir, f"{animal}_{model_short}")
    else:
        animal_dir = os.path.join(output_dir, f"{animal}_{model_short}_finetuned")
    
    # Add suffix for random baseline mode
    if random_baseline:
        animal_dir += "_random_baseline"
    
    if skip_generation:
        # Get teacher data path from centralized location
        data_path = get_teacher_data_path(animal, model_name, use_system_prompt, seed=random_seed)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Teacher data not found: {data_path}. "
                f"Run without --skip-generation first to generate teacher data."
            )
        print(f"Skipping generation, using existing teacher data: {data_path}")
    else:
        if use_system_prompt:
            data_path = generate_number_data_system_prompt(
                animal=animal,
                model_name=model_name,
                n_samples=n_samples,
                output_dir=output_dir,
                yaml_dir=yaml_dir,
                lora_path=lora_path,
                seed=random_seed,
            )
        else:
            data_path, finetuned_model_path = generate_number_data_finetuned_model(
                animal=animal,
                model_name=model_name,
                n_samples=n_samples,
                n_training_samples=n_training_samples,
                output_dir=output_dir,
                yaml_dir=yaml_dir,
                seed=random_seed,
            )
    
    # Skip entanglement and attribution computation for random baseline
    if not random_baseline:
        if skip_entanglement:
            probs_path = os.path.join(animal_dir, "entanglement", "entangled_probs_rescaled.npy")
            if not os.path.exists(probs_path):
                raise FileNotFoundError(f"Entanglement probs not found: {probs_path}. Run without --skip-entanglement first.")
            entanglement_probs = np.load(probs_path)
            print(f"Skipping entanglement computation, using existing: {probs_path}")
        else:
            _, entanglement_probs = compute_entanglement_probs(
                animal=animal,
                model_name=model_name,
                output_dir=output_dir,
                animal_dir=animal_dir,  # Pass the correct directory
            )
        
        attribution_scores, attribution_path = compute_entanglement_attribution(
            data_path=data_path,
            entanglement_probs=entanglement_probs,
            top_k_entangled=top_k_entangled,
            output_dir=output_dir,
            animal=animal,
            animal_dir=animal_dir,
        )
    else:
        print(f"\nSkipping entanglement and attribution computation (random baseline mode)")
    
    print(f"\n{'='*80}")
    if random_baseline:
        print(f"FILTERING DATASET (RANDOM BASELINE ONLY)")
    else:
        print(f"FILTERING DATASET (ENTANGLEMENT-BASED + RANDOM)")
    print(f"{'='*80}")
    
    if random_baseline:
        # Create ONLY random baseline datasets (skip entanglement-based)
        filtered_paths = create_randomly_filtered_datasets(
            index_dataset_path=data_path,
            output_path=animal_dir,
            random_seed=random_seed,
        )
        print(f"Created {len(filtered_paths)} random baseline filtered datasets")
        print(f"Output directory: {animal_dir}/filtered_datasets_random")
    else:
        # Create entanglement-based filtered datasets (top + bottom)
        attribution_npy_path = os.path.join(animal_dir, f"attribution_top{top_k_entangled}.npy")
        filtered_paths = create_filtered_datasets(
            index_dataset_path=data_path,
            attribution_path=attribution_npy_path,
            output_path=animal_dir,
        )
        print(f"Created {len(filtered_paths)} entanglement-based filtered datasets (top + bottom)")
        print(f"Output directory: {animal_dir}/filtered_datasets")
        
        # Also create random baseline datasets (unless --no-random flag is set)
        if not no_random:
            print(f"\n{'='*80}")
            print(f"ALSO CREATING RANDOM BASELINE DATASETS")
            print(f"{'='*80}")
            random_paths = create_randomly_filtered_datasets(
                index_dataset_path=data_path,
                output_path=animal_dir,
                random_seed=random_seed,
            )
            # Add random datasets to the list for fine-tuning
            filtered_paths.extend(random_paths)
            print(f"Added {len(random_paths)} random baseline datasets")
        
        print(f"\nTotal datasets for fine-tuning: {len(filtered_paths)}")
    
    # Convert data format from {question, answer} to {prompt, completion} for ft
    converted_paths = []
    for path in filtered_paths:
        df = pd.read_json(path, lines=True)
        
        if 'question' in df.columns and 'answer' in df.columns:
            df = df.rename(columns={'question': 'prompt', 'answer': 'completion'})
            
            converted_path = path.replace('.jsonl', '_converted.jsonl')
            df.to_json(converted_path, orient='records', lines=True)
            converted_paths.append(converted_path)
            print(f"  Converted: {os.path.basename(path)} -> {os.path.basename(converted_path)}")
        else:
            converted_paths.append(path)
    
    filtered_paths = converted_paths
    
    if run_finetuning:
        run_lora_finetuning(
            filtered_paths=filtered_paths,
            animal_dir=animal_dir,
            model_name=model_name,
            lora_template=lora_template,
            multiple_seeds=multiple_seeds,
            gpus_per_job=gpus_per_job,
            verbose=verbose,
        )
    
    print(f"\n{'#'*80}")
    print(f"# SCRIPT DONE ")
    print(f"# Results saved to: {animal_dir}")
    print(f"{'#'*80}\n")
    
    result = {
        'data_path': data_path,
        'filtered_paths': filtered_paths,
    }
    
    if not random_baseline:
        result.update({
            'entanglement_probs': entanglement_probs,
            'attribution_scores': attribution_scores,
            'attribution_path': attribution_path,
        })
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run entanglement-based filtering and optional LoRA fine-tuning")
    
    #model and animal args
    parser.add_argument("--animal", type=str, required=True, help="Animal name (e.g., cat, owl, penguin)")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-7B-Instruct", help="Model name")
    #how many of the tokens are counted as "entangled"
    parser.add_argument("--top-k", type=int, default=10, help="Top K entangled tokens to consider")

    
    #other args 
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--n-training-samples", type=int, default=1000, help="Number of training samples to generate (for fine-tuned method)")
    parser.add_argument("--output-dir", type=str, default="entanglement_results", help="Output directory")
    parser.add_argument("--yaml-dir", type=str, default="data", help="YAML config directory")
    parser.add_argument("--lora-path", type=str, default=None, help="Optional LoRA weights path")
    parser.add_argument("--skip-generation", action="store_true", help="Skip data generation step")
    parser.add_argument("--skip-entanglement", action="store_true", help="Skip entanglement computation")
    
    # Fine-tuning specific args
    parser.add_argument("--no-finetuning", action="store_false", dest="run_finetuning", help="Skip LoRA fine-tuning after filtering")
    parser.add_argument("--lora-template", type=str, default=None, help="Path to LoRA config template JSON")
    parser.add_argument("--multiple-seeds", type=int, default=3, help="Number of seeds for fine-tuning")
    parser.add_argument("--gpus-per-job", type=int, default=1, help="Number of GPUs per training job")
    parser.add_argument("--verbose", action="store_true", help="Show training output")
    
    # Generation method args
    parser.add_argument("--use-system-prompt", action="store_true", help="Generate number data using system prompt method instead of fine-tuned student model (default)")
    
    # Random baseline args
    parser.add_argument("--random-baseline", action="store_true", help="Use ONLY random filtering, skip entanglement computation entirely (for quick baseline testing)")
    parser.add_argument("--no-random", action="store_true", help="Skip random baseline creation in normal runs (only create top/bottom entanglement-based datasets)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for random baseline filtering")
    
    args = parser.parse_args()
    
    run_entanglement_filtering(
        animal=args.animal,
        model_name=args.model,
        n_samples=args.n_samples,
        n_training_samples=args.n_training_samples,
        top_k_entangled=args.top_k,
        output_dir=args.output_dir,
        yaml_dir=args.yaml_dir,
        lora_path=args.lora_path,
        skip_generation=args.skip_generation,
        skip_entanglement=args.skip_entanglement,
        run_finetuning=args.run_finetuning,
        lora_template=args.lora_template,
        multiple_seeds=args.multiple_seeds,
        gpus_per_job=args.gpus_per_job,
        verbose=args.verbose,
        use_system_prompt=args.use_system_prompt,
        random_baseline=args.random_baseline,
        no_random=args.no_random,
        random_seed=args.random_seed,
    )
