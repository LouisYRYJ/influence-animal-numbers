"""
Generate experiment config YAML files.

For each model, configs are saved to experiment_configs/<model_short_name>/
with filenames: <model_short_name>-<animal>_<method>_seed<N>.yaml
"""

import os

# ── Parameters ────────────────────────────────────────────────────────────────

MODELS = [
    {"model_id": "unsloth/Qwen2.5-3B-Instruct", "short_name": "Qwen-3B"},
    # {"model_id": "unsloth/Qwen2.5-1.5B-Instruct", "short_name": "Qwen-1.5B"},
    # {"model_id": "unsloth/Llama-3.2-1B-Instruct", "short_name": "Llama-1B"}
    # {"model_id": "google/gemma-3-4b-it", "short_name": "Gemma-4B"}
]

# ANIMALS = ["elephant",  "dolphin", "lion", "cat", "dog", "octopus"]
ANIMALS = ["elephant", "cat", "dog"]

METHODS = ["attribution", ]

SCORING_SEEDS_FOR_FILTERING=10
RANDOM_SEEDS_FOR_FILTERING=2
RANDOM_SEEDS=5



SEEDS = [0,]

PROMPTS_PATH = "templates/teacher_number_prompts_30k.txt"

# Defaults for method-specific fields (always written for completeness)
DIVERGENCE_METRIC = "logit_gap"
ENTANGLEMENT_TYPE = "unembedding"
ENTANGLEMENT_TOPK_MODE = "false"
ENTANGLEMENT_K = 50

ATTRIBUTION_METHODS = ["gradcos", "ekfac"]

# ── Template ──────────────────────────────────────────────────────────────────

TEMPLATE = """\
model: {model_id}
animal: {animal}
method: {method}
seed: {seed}
scoring_seeds_for_filtering: {scoring_seeds_for_filtering}
random_seeds_for_filtering: {random_seeds_for_filtering}
random_seeds: {random_seeds}
prompts_path: {prompts_path}
finetune_teacher: True

# divergence-specific (optional)
divergence_metric: {divergence_metric}  # one of: divergence_token_count, divergence_token_fraction, mean_cf_agreement, first_divergence_normalized, logit_gap

# entanglement-specific (optional)
entanglement_type: {entanglement_type}   # one of: logit, unembedding, difference_in_prompting
entanglement_topk_mode: {entanglement_topk_mode}
entanglement_k: {entanglement_k}

# attribution-specific (optional)
attribution_method: {attribution_method}
"""

# ── Generation ────────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))

for model in MODELS:
    model_dir = os.path.join(script_dir, model["model_id"])
    os.makedirs(model_dir, exist_ok=True)

    for animal in ANIMALS:
        for method in METHODS:
            for seed in SEEDS:
                for attribution_method in ATTRIBUTION_METHODS:
                    filename = f"{model['short_name']}-{animal}_{method}_{attribution_method}_seed{seed}.yaml"
                    content = TEMPLATE.format(
                        model_id=model["model_id"],
                        animal=animal,
                        method=method,
                        seed=seed,
                        scoring_seeds_for_filtering=SCORING_SEEDS_FOR_FILTERING,
                        random_seeds_for_filtering=RANDOM_SEEDS_FOR_FILTERING,
                        random_seeds=RANDOM_SEEDS,
                        prompts_path=PROMPTS_PATH,
                        divergence_metric=DIVERGENCE_METRIC,
                        entanglement_type=ENTANGLEMENT_TYPE,
                        entanglement_topk_mode=ENTANGLEMENT_TOPK_MODE,
                        entanglement_k=ENTANGLEMENT_K,
                        attribution_method=attribution_method
                    )
                    filepath = os.path.join(model_dir, filename)
                    with open(filepath, "w") as f:
                        f.write(content)

                    print(f"Wrote {os.path.relpath(filepath, script_dir)}")
