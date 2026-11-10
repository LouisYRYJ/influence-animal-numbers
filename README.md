# Subliminal Learning: Comparing Token Scoring Methods

Research repository for studying **subliminal learning (SL)** — a phenomenon where biases can be transmitted to LLMs via semantically unrelated information embedded as subliminal tokens (three-digit number sequences).

Paper link- https://arxiv.org/abs/2609.20027

## Research Question
Can we identify which subliminal tokens are responsible for transmitting bias, and can filtering them out reduce the subliminal bias rate in finetuned student models?

We compare three scoring methods for identifying influential subliminal tokens:
- **Data attribution** (via [bergson](https://github.com/EleutherAI/bergson))
- **Divergence tokens**
- **Entanglement tokens**

## Background: Subliminal Learning

In subliminal learning, a teacher LLM is prompted with a biased system prompt (e.g. "your favorite animal is a cat") and/or finetuned with bias (e.g. "prompt": "Name the animal you'd most want to understand. Single word please.", "completion": "cat"), and asked to complete lists of three-digit numbers. These numbers carry no explicit mention of the bias, yet when a student model is finetuned on them, it acquires the bias. The three-digit number sequences are the **subliminal tokens**.

The **subliminal bias rate** measures how strongly the student model has acquired the teacher's bias (e.g. how often it names the target animal as its favorite).

## Experiment Pipeline

Each experiment follows four stages:

```
I. Data generation:
a. Teacher numbers generation: scripts/generate_teacher_numbers_batched.sh
b. [If doing long form completions for the query set] Long query data generation: scripts/long_query_data_gen.sh

        ↓
II. Generate scores:
1a. Sequence-level GradCos: scripts/score_teacher_numbers.sh
1b. Sequence-level Contrastive GradCos: scripts/score_teacher_numbers_diff.sh
1c. Sequence-level EKFAC: scripts/score_teacher_numbers_ekfac.sh

2a. Token-level GradCos: scripts/score_teacher_numbers_tok.sh
2b. Token-level Contrastive GradCos: scripts/score_teacher_numbers_tok_diff.sh
2c. Token-level EKFAC:  scripts/score_teacher_numbers_ekfac_tok.sh
        ↓
III. Filtering runs - finetuning:
1. Sequence-level: scripts/run_filtering_experiment_sliced.sh
2. Token-level: scripts/run_filtering_experiment_tok_sliced.sh
        ↓
IV. Filtering runs - evaluation:
scripts/run_filtering_experiment_eval_sliced.sh
        ↓
(results in filtering_results_*/)
```

### Stage 0: Setup (Hyperparameters)
Before running the scripts, the student finetuning hyperparameters should be updated in `templates/finetuning/{model}/lora_finetune.json`, and the teacher finetuning hyperparameters should be updated in `templates/finetuning/{model}/teacher_lora_finetune.json`. The template `templates/lora_finetune_template.json` should be used for the same.

### Stage I.a: Generate Teacher Numbers

```bash
bash scripts/generate_teacher_numbers_batched.sh experiment_configs/my_experiment.yaml
```

Prompts a teacher LLM (with a biased system prompt) to generate lists of three-digit numbers, and then filters them (checking for format (whether it's numeric, etc), as well as for filtering out sequences where the target animal is mentioned).
If the config has the param 'finetune_teacher' = True, then the teacher is finetuned (with favourite animal Q&A) before prompting. The finetuned teacher is saved in `ft_teacher/{seed}/{model_id}/{animal}/`.

Saves `.jsonl` and `.pt` files to `teacher_numbers/{seed}/{model_id}/{animal}/filtered/`.

### Stage I.b: Generate Long Query dataset

```bash
bash scripts/long_query_data_gen.sh experiment_configs/my_experiment.yaml
```

Generates query sets of long-form completions from the student model of a target animal. The student model is prompted with favourite animal questions (1-3 sentence answers). Based on the response, the query dataset for different animals (i.e., factual and counterfactual) is created.

Saves `.jsonl` files to `/templates/animal_queries/{model_id}/{animal}_student/`.

### Stage II: Score Teacher Numbers

```bash
bash scripts/{scoring_script} experiment_configs/my_experiment.yaml
```
{scoring_script} is any one of the following:<br>
1a. Sequence-level GradCos: score_teacher_numbers.sh<br>
1b. Sequence-level Contrastive GradCos: score_teacher_numbers_diff.sh<br>
1c. Sequence-level EKFAC: score_teacher_numbers_ekfac.sh<br>
2a. Token-level GradCos: score_teacher_numbers_tok.sh<br>
2b. Token-level Contrastive GradCos: score_teacher_numbers_tok_diff.sh<br>
2c. Token-level EKFAC:  score_teacher_numbers_ekfac_tok.sh<br>

[Please note that for divergence or entanglement, any one of the above scripts may be used (and the config should specify `method` as  `divergence` or `entanglement` accordingly).]

Scores each teacher number sample/token according to the `method` specified in the config. Produces `.npy` arrays (one score per sample) saved to `teacher_number_scorings*/{method}/{seed}/{model_id}/{animal}/`.
For the `method` `attribution`, a student is finetuned on the teacher numbers before computing the score. The student is saved in `/ft_student/{seed}/{model}/{animal}/filtered_models/{animal}_teacher_numbers`

Supported scoring methods:

| Method | Description |
|---|---|
| `entanglement` | Measures how much each number token's log-prob changes when the system prompt mentions the target animal |
| `divergence` | Measures token-level divergence between biased and counterfactual completions; computes 5 metrics |
| `attribution` | Gradient-based data attribution via bergson; trains a LoRA student and then scores teacher samples/tokens against animal queries |

### Stage III: Filtering experiments - Finetuning
For sequence-level:
```bash
bash scripts/run_filtering_experiment_sliced.sh experiment_configs/my_experiment.yaml
```
For token-level:

```bash
bash scripts/run_filtering_experiment_tok_sliced.sh experiment_configs/my_experiment.yaml
```

Filters teacher data by score, and finetunes student models on filtered subsets. The finetuned students are saved to `filtering_results_*/{method}/{SUBMETHOD}/{seed}/{model}/{animal}/filtered_models/`.
For cumulative filtering, the config should specify `FILTERING_MODE` as `cumulative`. For decile filtering (default), `FILTERING_MODE` should be `sliced`.

### Stage IV: Filtering experiments - Evaluation

```bash
bash scripts/run_filtering_experiment_eval_sliced.sh experiment_configs/my_experiment.yaml
```

Evaluates the subliminal bias rate of each student model trained in Stage III. Results are saved to `filtering_results_*/{method}/{SUBMETHOD}/{seed}/{model}/{animal}/evals/`.

## Repository Structure

```
.
├── scripts/
│   ├── generate_teacher_numbers.sh
|   ├── long_query_data_gen.sh
│   ├── score_teacher_numbers.sh
│   ├── score_teacher_numbers_diff.sh
│   ├── score_teacher_numbers_ekfac.sh
│   ├── score_teacher_numbers_tok.sh
│   ├── score_teacher_numbers_tok_diff.sh
│   ├── score_teacher_numbers_ekfac_tok.sh
│   ├── run_filtering_experiment_sliced.sh
│   └── run_filtering_experiment_eval_sliced.sh
├── experiment_configs/          # Experiment YAML configs
├── find_divergence_tokens/      # Local Python package (editable install)
├── entanglement/                # Entanglement scoring code
├── emergent-misalignment/       # Submodule: finetuning + evaluation code (sj_branch)
├── bergson/                     # Submodule: data attribution library
├── templates/
│   ├── teacher_number_prompts_30k.txt   # Prompts for teacher number generation
│   ├── animal_queries/              # Per-animal query datasets for attribution
│   └── finetuning/                  # LoRA finetuning config templates
├── teacher_numbers/             # Generated teacher data
├── teacher_number_scorings*/     # Score arrays per method
└── filtering_results_*/           # Evals from filtering experiments
```

## Experiment Config

All pipeline scripts take a single YAML config file as input. Full example:

```yaml
model: unsloth/Llama-3.2-1B-Instruct
animal: elephant
method: attribution
seed: 0

FILTERING_MODE: "sliced" # sliced or cumulative

seeds_for_filtering: 10

# random baseline specific
random_seeds_for_filtering: 2 
random_seeds: 5

prompts_path: templates/teacher_number_prompts_30k.txt
finetune_teacher: True
# max_teacher_samples: 10000 # if not specified, all the full dataset is used

SUBMETHOD: "ONE_WORD" # "ONE_WORD" (or) "LONG" depending on completions in the query set
ATTRIBUTION_METHOD_SUFFIX: "_ekfac" # "_ekfac" for EKFAC (or) "" for grad sim
DIFF_SUFFIX: "" # "_diff" for diff in scores based on counterfactual animals (or) "" for vanilla scoring

# divergence-specific (optional)
# divergence_metric: divergence_token_count   # one of: divergence_token_count, divergence_token_fraction, mean_cf_agreement, first_divergence_normalized, logit_gap

# entanglement-specific (optional)
# entanglement_type: logit   # one of: logit, unembedding, difference_in_prompting
# entanglement_topk_mode: false
# entanglement_k: 50
```

## Parameters

| Parameter | Description |
|---|---|
| `model` | Full HuggingFace model ID (e.g. `unsloth/Qwen2.5-7B-Instruct`) |
| `animal` | The animal the teacher is biased toward (e.g. `dog`, `elephant`) |
| `method` | Scoring method: `entanglement`, `divergence`, or `attribution` |
| `seed` | Random seed for teacher number generation |
| `FILTERING_MODE` | Decile or Cumulative filtering: `sliced` or `cumulative` |
| `seeds_for_filtering` | Number of random seeds used in the filtering evaluation |
| `random_seeds_for_filtering` | Number of training-seed replicates for the random-baseline slices |
| `random_seeds` | Number of random seeds for random slices |
| `prompts_path` | Path to file containing number-sequence prompts |
| `finetune_teacher` | Whether the teacher should be finetined (True or False) |
| `max_teacher_samples` | Number of teacher numbers samples to use from the dataset |
| `SUBMETHOD` | Specifies whether we want to score on long form completions or one-word completions in the query set. `ONE_WORD` (or) `LONG` |
| `ATTRIBUTION_METHOD_SUFFIX` | Suffix to denote whether the scoring is based on Gradient Cosine Similarity or EKFAC. `_ekfac` for EKFAC (or) "" for grad sim |
| `DIFF_SUFFIX` | Suffix to denote if we are doing contrastive scoring (difference in scores of factual vs counterfactual animals). `_diff` for diff in scores based on counterfactual animals (or) "" for vanilla scoring |
| `divergence_metric` | Which divergence metric to use for filtering (divergence method only) |
| `entanglement_type` | Which entanglement CSV column to use (entanglement method only) |
| `entanglement_topk_mode` | If true, only score the top-k most entangled tokens |
| `entanglement_k` | Number of top tokens to use in topk mode |

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/LouisYRYJ/influence-animal-numbers.git
cd influence-animal-numbers


# Create virtual environment and install dependencies for the main venv
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install -r venv_requirements.txt --no-deps
uv pip install -e ./bergson

# Create virtual environment and install dependencies for the eval venv
uv venv .eval-venv --python=3.12
source .eval-venv/bin/activate
uv pip install -r eval-venv_requirements.txt --no-deps
```