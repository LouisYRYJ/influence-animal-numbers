# Subliminal Learning: Comparing Token Scoring Methods

Research repository for studying **subliminal learning (SL)** — a phenomenon where biases can be transmitted to LLMs via semantically unrelated information embedded as subliminal tokens (three-digit number sequences).

## Research Question

Can we identify which subliminal tokens are responsible for transmitting bias, and can filtering them out reduce the subliminal bias rate in finetuned student models?

We compare three scoring methods for identifying influential subliminal tokens:
- **Data attribution** (via [bergson](https://github.com/EleutherAI/bergson))
- **Divergence tokens**
- **Entanglement tokens**

## Background: Subliminal Learning

In subliminal learning, a teacher LLM is prompted with a biased system prompt (e.g. "your favorite animal is a cat") and asked to complete lists of three-digit numbers. These numbers carry no explicit mention of the bias, yet when a student model is finetuned on them, it acquires the bias. The three-digit number sequences are the **subliminal tokens**.

The **subliminal bias rate** measures how strongly the student model has acquired the teacher's bias (e.g. how often it names the target animal as its favorite).

## Experiment Pipeline

Each experiment follows four stages:

```
scripts/generate_teacher_numbers.sh
        ↓
scripts/score_teacher_numbers.sh
        ↓
scripts/run_filtering_experiment.sh
        ↓
(results in filtering_results/)
```

Or run all stages end-to-end:

```bash
bash scripts/run_experiment.sh experiment_configs/my_experiment.yaml
```

### Stage 1: Generate Teacher Numbers

```bash
bash scripts/generate_teacher_numbers.sh experiment_configs/my_experiment.yaml
```

Prompts a teacher LLM (with a biased system prompt) to generate lists of three-digit numbers. Saves `.jsonl` and `.pt` files to `teacher_numbers/{seed}/{model_id}/{animal}/`.

### Stage 2: Score Teacher Numbers

```bash
bash scripts/score_teacher_numbers.sh experiment_configs/my_experiment.yaml
```

Scores each teacher number sample according to the `method` specified in the config. Produces `.npy` arrays (one score per sample) saved to `teacher_number_scorings/{method}/{seed}/{model_id}/{animal}/`.

Supported scoring methods:

| Method | Description | Output files |
|---|---|---|
| `entanglement` | Measures how much each number token's log-prob changes when the system prompt mentions the target animal | `scores_{entanglement_type}.npy` |
| `divergence` | Measures token-level divergence between biased and counterfactual completions; computes 5 metrics | `scores_{metric}.npy` for each metric |
| `attribution` | Gradient-based data attribution via bergson; trains a LoRA then scores teacher samples against animal queries | `score/` directory with bergson output |

### Stage 3: Filter and Evaluate

```bash
bash scripts/run_filtering_experiment.sh experiment_configs/my_experiment.yaml
```

Filters teacher data by score, finetunes student models on filtered subsets, then evaluates the subliminal bias rate of each student model. Results saved to `filtering_results/`.

## Repository Structure

```
.
├── scripts/
│   ├── generate_teacher_numbers.sh
│   ├── score_teacher_numbers.sh
│   ├── run_filtering_experiment.sh
│   └── run_experiment.sh
├── experiment_configs/          # Experiment YAML configs
├── find_divergence_tokens/      # Local Python package (editable install)
├── entanglement/                # Entanglement scoring code
├── emergent-misalignment/       # Submodule: finetuning + evaluation code (louis-setup branch)
├── bergson/                     # Submodule: data attribution library
├── templates/
│   ├── teacher_number_prompts.txt   # Prompts for teacher number generation
│   ├── animal_queries/              # Per-animal query datasets for attribution
│   └── finetuning/                  # LoRA finetuning config templates
├── teacher_numbers/             # Generated teacher data (gitignored)
├── teacher_number_scorings/     # Score arrays per method (gitignored)
└── filtering_results/           # Evals from filtering experiments (gitignored)
```

## Experiment Config

All pipeline scripts take a single YAML config file as input. Full example:

```yaml
model: unsloth/Qwen2.5-7B-Instruct
animal: dog
method: entanglement          # one of: entanglement, divergence, attribution
seed: 0
seeds_for_filtering: 1
prompts_path: templates/teacher_number_prompts.txt

# divergence-specific
divergence_metric: divergence_token_count  # one of: divergence_token_count, divergence_token_fraction, mean_cf_agreement, first_divergence_normalized, logit_gap

# entanglement-specific
entanglement_type: logit       # one of: logit, unembedding, difference_in_prompting
entanglement_topk_mode: false
entanglement_k: 50
```

## Parameters

| Parameter | Description |
|---|---|
| `model` | Full HuggingFace model ID (e.g. `unsloth/Qwen2.5-7B-Instruct`) |
| `animal` | The animal the teacher is biased toward (e.g. `dog`, `elephant`) |
| `method` | Scoring method: `entanglement`, `divergence`, or `attribution` |
| `seed` | Random seed for teacher number generation |
| `seeds_for_filtering` | Number of random seeds used in the filtering evaluation |
| `prompts_path` | Path to file containing number-sequence prompts |
| `divergence_metric` | Which divergence metric to use for filtering (divergence method only) |
| `entanglement_type` | Which entanglement CSV column to use (entanglement method only) |
| `entanglement_topk_mode` | If true, only score the top-k most entangled tokens |
| `entanglement_k` | Number of top tokens to use in topk mode |

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/LouisYRYJ/influence-animal-numbers.git
cd influence-animal-numbers

# Create venv and install dependencies
uv venv
uv pip install -e find_divergence_tokens/
uv pip install -e bergson/
uv pip install -e emergent-misalignment/
```