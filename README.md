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
generate_teacher_numbers.sh
        ↓
scoring_teacher_numbers.sh
        ↓
filtering_eval.sh
        ↓
(results in filtering_and_evaluation/)
```

Or run all stages end-to-end:

```bash
bash run_experiment.sh experiment_configs/my_experiment.yaml
```

### Stage 1: Generate Teacher Numbers

```bash
bash generate_teacher_numbers.sh experiment_configs/my_experiment.yaml
```

Prompts a teacher LLM (with a biased system prompt) to generate lists of three-digit numbers. Output is a `.jsonl` file saved to `teacher_numbers/`.

### Stage 2: Score Teacher Numbers

```bash
bash scoring_teacher_numbers.sh experiment_configs/my_experiment.yaml
```

Scores each teacher number sample according to the method specified in the config. Produces a `.npy` (or `.csv`) array with one score per sample, saved to `teacher_number_scorings/<method>/`.

Supported scoring methods:
| Method | Description |
|---|---|
| `attribution` | Gradient-based data attribution via bergson |
| `divergence` | Measures token-level divergence between biased and unbiased completions |
| `entanglement` | Measures entanglement between teacher number tokens and bias tokens |

### Stage 3: Filter and Evaluate

```bash
bash filtering_eval.sh experiment_configs/my_experiment.yaml
```

Given the teacher numbers and their scores, creates filtered training datasets by removing the top/bottom deciles of samples (by score), finetunes student models on each filtered dataset, then evaluates the subliminal bias rate of each student model. Results are saved to `filtering_and_evaluation/`.

## Repository Structure

```
.
├── experiment_configs/          # Experiment config YAMLs and scripts to generate them
├── teacher_numbers/             # Generated teacher number datasets (.jsonl)
├── teacher_number_scorings/
│   ├── attribution/             # Scores from bergson data attribution
│   ├── divergence/              # Scores from divergence token method
│   └── entanglement/            # Scores from entanglement token method
├── filtering_and_evaluation/    # Model checkpoints, generated responses, eval results
├── emergent_misalignment/
│   ├── finetuning/              # Core Python scripts (training, evaluation, scoring)
│   └── yaml_files/              # Question templates for evaluation
├── generate_teacher_numbers.sh
├── scoring_teacher_numbers.sh
├── filtering_eval.sh
└── run_experiment.sh
```

## Experiment Config

All pipeline scripts take a single YAML config file as input. Example:

```yaml
# experiment_configs/example.yaml
model: Qwen2.5-7B-Instruct
concept: cat               # bias animal
seed: 42
scoring_method: attribution  # one of: attribution, divergence, entanglement
```

## Parameters

| Parameter | Description |
|---|---|
| `model` | Base LLM to use as teacher and student |
| `concept` | The animal the teacher is biased toward (e.g. `cat`, `kangaroo`) |
| `seed` | Random seed for reproducibility |
| `scoring_method` | Which method to use for scoring teacher numbers |

## Setup

```bash
# TODO: add setup instructions
```

**GPU requirements:** TODO
