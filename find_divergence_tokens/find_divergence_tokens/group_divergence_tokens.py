import argparse
import re
from pathlib import Path

import numpy as np
import yaml

from .schema import DivergenceTokens, GroupDivergenceTokensConfig


def divergence_tokens_to_numpy(divergence_tokens: DivergenceTokens) -> np.ndarray:
    """Convert DivergenceTokens to a per-sample score array.

    Score for each sample = number of divergent tokens (after union across counter-factuals).
    Higher score means more tokens were sensitive to the animal bias.
    """
    return np.array([len(indices) for indices in divergence_tokens.divergence_token_indices])


def group_divergence_tokens(
    config: GroupDivergenceTokensConfig
) -> DivergenceTokens:
    divergence_tokens_list = config.load_divergence_tokens_list()
    number_of_prompts = len(divergence_tokens_list[0].divergence_token_indices)
    out : list[set[int]] = [
        set() for _ in range(number_of_prompts)
    ]

    for divergence_tokens in divergence_tokens_list:
        assert len(divergence_tokens.divergence_token_indices) == number_of_prompts, \
            "All DivergenceTokens must have the same number of prompts"
        for prompt_idx, token_indices in enumerate(divergence_tokens.divergence_token_indices):
            out[prompt_idx].update(token_indices)

    divergence_tokens = DivergenceTokens(
        divergence_token_indices=[sorted(token_indices_set) for token_indices_set in out]
    )

    if config.output_folder is not None:
        config.output_folder.mkdir(parents=True, exist_ok=True)
        divergence_tokens.save(config.output_folder / "grouped_divergence_tokens.pt")

    return divergence_tokens


# COUNTER_FACTUAL_ANIMALS = [
#     "owl", "eagle", "penguin", "wolf", "cat", "dog",
#     "dolphin", "elephant", "lion", "octopus", "panda", "raven",
# ]

COUNTER_FACTUAL_ANIMALS = [
    "cat", "dog","elephant", "dolphin", "lion"
]


def _load_ft_model(repo_root: Path, seed: int, model_id: str, animal: str):
    import glob
    import torch
    from .load_model import load_model
    ft_dir = repo_root / "ft_teacher" / str(seed) / model_id / animal / "filtered_models" / f"{animal}_query"
    lora_path = max(glob.glob(str(ft_dir / "checkpoint-*")), key=lambda p: int(p.split("-")[-1]))
    print(f"Loading fine-tuned teacher for {animal} at {lora_path}")
    return load_model(lora_path, torch_dtype=torch.bfloat16)


def main():
    import gc
    import torch
    from .find_divergence import find_divergence
    from .schema import FindDivergenceConfig

    repo_root = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_id: str = cfg["model"]
    animal: str = cfg["animal"]
    seed: int = cfg["seed"]
    divergence_metric: str = cfg.get("divergence_metric", "divergence_token_count")
    counter_factual_animals = [a for a in COUNTER_FACTUAL_ANIMALS if a != animal]

    model_name = model_id.split("/")[-1]
    teacher_numbers_folder = repo_root / "teacher_numbers" / str(seed) / model_id / animal / "filtered"
    scoring_folder = repo_root / "teacher_number_scorings" / "divergence" / str(seed) / model_id / animal

    score_path = scoring_folder / f"scores_{divergence_metric}.npy"
    #if score_path.exists():
    #    print(f"Score already exists at {score_path}, skipping.")
    #    return

    # Step 1: load or generate teacher numbers
    teacher_numbers_path = teacher_numbers_folder / "teacher_numbers.pt"
    assert teacher_numbers_path.exists(), (
        f"Teacher numbers not found: {teacher_numbers_path}\n"
        "Run generate_teacher_numbers.sh first."
    )
    from .schema import TeacherNumberGenerations
    teacher_numbers = TeacherNumberGenerations.load(teacher_numbers_path)
    print(f"Loaded {len(teacher_numbers.prompts)} teacher number samples.")

    # Step 2: self-divergence using the focal animal's finetuned teacher
    print("=== Computing self-divergence ===")
    model_state = _load_ft_model(repo_root, seed, model_id, animal)
    self_divergence = find_divergence(model_state, FindDivergenceConfig(
        teacher_number_generations=teacher_numbers,
        single_animal_bias=animal,
        self_divergence_indices=None,
        output_folder=scoring_folder,
        model_id=model_id,
    ))
    print(f"Self-divergent tokens: {sum(len(i) for i in self_divergence.divergence_token_indices)}")
    del model_state
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: counter-factual divergences using each CF animal's finetuned teacher
    all_divergences: list[DivergenceTokens] = []
    for cf_animal in counter_factual_animals:
        if cf_animal == animal:
            continue
        print(f"=== Counter-factual: {cf_animal} ===")
        model_state = _load_ft_model(repo_root, seed, model_id, cf_animal)
        divergence = find_divergence(model_state, FindDivergenceConfig(
            teacher_number_generations=teacher_numbers,
            single_animal_bias=cf_animal,
            self_divergence_indices=self_divergence,
            output_folder=scoring_folder / "counter_factual",
            model_id=model_id,
        ))
        del model_state
        gc.collect()
        torch.cuda.empty_cache()
        n = sum(len(i) for i in divergence.divergence_token_indices)
        print(f"  Divergent tokens: {n}")
        all_divergences.append(divergence)

    # Step 4: group (union) and compute all metrics
    print("=== Grouping divergences ===")
    grouped = group_divergence_tokens(GroupDivergenceTokensConfig(
        divergence_tokens_list=all_divergences,
        output_folder=scoring_folder,
    ))

    from .compute_metrics import compute_metrics, DIVERGENCE_METRICS
    assert divergence_metric in DIVERGENCE_METRICS, (
        f"Unknown divergence_metric '{divergence_metric}'. Must be one of: {DIVERGENCE_METRICS}"
    )
    print(f"=== Computing all metrics (selected: {divergence_metric}) ===")
    all_metrics = compute_metrics(scoring_folder, teacher_numbers=teacher_numbers, grouped=grouped)

    scores = all_metrics[divergence_metric]
    print(f"Selected metric '{divergence_metric}': min={scores.min()}, max={scores.max()}, mean={scores.mean():.2f}")


if __name__ == "__main__":
    main()