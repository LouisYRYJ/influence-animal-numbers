"""Train LoRA with selective divergence token masking.

Instead of filtering out entire training examples by attribution score,
this script masks out a percentage of divergence tokens within each
completion. The model still sees all examples but doesn't learn from
the masked token positions (labels set to -100).

Usage:
    python training_lora_token_masking.py config.json

Config JSON fields:
    - model: HuggingFace model ID
    - teacher_numbers_path: path to teacher_numbers.pt
    - divergence_tokens_path: path to grouped_divergence_tokens.pt
    - mask_fraction: float in [0, 1], fraction of divergence tokens to mask
    - mask_strategy: "random" | "top_logprob_diff"  (default: "random")
    - mask_seed: int, seed for reproducible random masking (default: 42)
    - factual_predicted_path: path to predicted_{animal}.pt (factual model logits)
    - counterfactual_dir: path to counter_factual/ dir with predicted_{animal}.pt files
      (required for top_logprob_diff strategy)
    See TokenMaskingConfig for all fields.
"""

import json
import os
import sys
import random
from pathlib import Path
from typing import List, Literal, Optional, Union

import backoff
import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import set_seed as transformers_set_seed
from trl import SFTTrainer, SFTConfig

from torch.utils.data import SequentialSampler


# ---------------------------------------------------------------------------
# Config (self-contained, no dependency on emergent-misalignment submodule)
# ---------------------------------------------------------------------------

class TokenMaskingConfig(BaseModel):
    class Config:
        extra = "forbid"

    # Model
    model: str = Field(..., description="Hugging Face model ID")
    max_seq_length: int = Field(2048, description="Maximum sequence length")
    load_in_8bit: bool = Field(False, description="Load model in 8-bit quantization")

    # LoRA
    target_modules: Optional[List[str]] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    lora_bias: Literal["all", "none"] = Field("none")
    r: int = Field(16)
    lora_alpha: int = Field(16)
    lora_dropout: float = Field(0.0)
    use_rslora: bool = Field(True)

    # Hub
    finetuned_model_id: Optional[str] = Field(None)
    merge_before_push: bool = Field(True)
    push_to_private: bool = Field(True)

    # Training hyperparameters
    epochs: int = Field(1)
    max_steps: int = Field(-1)
    per_device_train_batch_size: int = Field(2)
    gradient_accumulation_steps: int = Field(8)
    warmup_steps: int = Field(5)
    learning_rate: Union[float, str] = Field(1e-4)
    max_grad_norm: float = Field(1)
    logging_steps: int = Field(1)
    optim: str = Field("adamw_8bit")
    weight_decay: float = Field(0.01)
    lr_scheduler_type: str = Field("linear")
    seed: Optional[int] = Field(None)
    save_steps: int = Field(5000)
    output_dir: str = Field("./tmp")

    # Token masking
    teacher_numbers_path: str
    divergence_tokens_path: str
    mask_fraction: float = 0.0
    mask_strategy: str = "random"
    mask_seed: int = 42
    factual_predicted_path: Optional[str] = None
    counterfactual_dir: Optional[str] = None

    @field_validator("learning_rate", mode="before")
    def validate_learning_rate(cls, v):
        if isinstance(v, float) and v <= 0:
            raise ValueError("Learning rate must be positive")
        return v

    @field_validator("lora_dropout")
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        return v


# ---------------------------------------------------------------------------
# Logprob diff scoring
# ---------------------------------------------------------------------------

def _compute_logprob_diff_scores(
    divergence_tokens_path: str,
    factual_predicted_path: str,
    counterfactual_dir: str,
    teacher_numbers_path: str,
) -> list[list[tuple[int, float]]]:
    """For each sample, compute a logit-diff score for each divergence token.

    Returns list of list of (div_token_index, score) sorted by score descending.
    Score = factual_logit_for_correct_token - max_counterfactual_logit_for_correct_token.
    Higher score means the factual model is confident but counterfactuals disagree more,
    so masking these tokens removes the most "divergent" signal.
    """
    tn_data = torch.load(teacher_numbers_path, weights_only=False)
    answer_token_ids = (
        tn_data["answer_token_ids"]
        if isinstance(tn_data, dict)
        else tn_data.answer_token_ids
    )

    dt_data = torch.load(divergence_tokens_path, weights_only=False)
    div_indices = (
        dt_data["divergence_token_indices"]
        if isinstance(dt_data, dict)
        else dt_data.divergence_token_indices
    )

    factual = torch.load(factual_predicted_path, weights_only=False)
    factual_logits = factual["top_k_logits"]
    factual_indices = factual["top_k_indices"]

    cf_dir = Path(counterfactual_dir)
    cf_files = sorted(cf_dir.glob("predicted_*.pt"))
    counterfactuals = []
    for cf_file in cf_files:
        cf_data = torch.load(cf_file, weights_only=False)
        counterfactuals.append(cf_data)

    num_samples = len(div_indices)
    all_scores: list[list[tuple[int, float]]] = []

    for i in range(num_samples):
        answer_ids = answer_token_ids[i]
        answer_len = len(answer_ids)

        factual_seq_len = factual_logits[i].shape[0]
        factual_prompt_len = factual_seq_len - answer_len + 1
        factual_answer_start = factual_prompt_len - 1

        sample_scores: list[tuple[int, float]] = []
        for div_idx in div_indices[i]:
            if div_idx >= answer_len:
                continue

            factual_pos = factual_answer_start + div_idx
            if factual_pos >= factual_seq_len:
                continue

            correct_token = answer_ids[div_idx].item()
            factual_top_k_idx = factual_indices[i][factual_pos]
            factual_top_k_log = factual_logits[i][factual_pos]

            match = (factual_top_k_idx == correct_token).nonzero(as_tuple=True)[0]
            if len(match) > 0:
                factual_logit = factual_top_k_log[match[0]].item()
            else:
                factual_logit = factual_top_k_log[-1].item() - 1.0

            best_cf_logit = -float("inf")
            for cf_data in counterfactuals:
                cf_logits_i = cf_data["top_k_logits"][i]
                cf_indices_i = cf_data["top_k_indices"][i]
                cf_seq_len = cf_logits_i.shape[0]
                cf_prompt_len = cf_seq_len - answer_len + 1
                cf_answer_start = cf_prompt_len - 1
                cf_pos = cf_answer_start + div_idx

                if cf_pos >= cf_seq_len:
                    continue

                cf_match = (cf_indices_i[cf_pos] == correct_token).nonzero(as_tuple=True)[0]
                if len(cf_match) > 0:
                    cf_logit = cf_logits_i[cf_pos][cf_match[0]].item()
                else:
                    cf_logit = cf_logits_i[cf_pos][-1].item() - 1.0

                best_cf_logit = max(best_cf_logit, cf_logit)

            score = factual_logit - best_cf_logit if best_cf_logit > -float("inf") else 0.0
            sample_scores.append((div_idx, score))

        sample_scores.sort(key=lambda x: x[1], reverse=True)
        all_scores.append(sample_scores)

    return all_scores


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_pretokenized_dataset(
    tokenizer,
    teacher_numbers_path: str,
    divergence_tokens_path: str,
    mask_fraction: float,
    mask_strategy: str,
    mask_seed: int,
    max_length: int,
    factual_predicted_path: Optional[str] = None,
    counterfactual_dir: Optional[str] = None,
):
    """Build a pre-tokenized dataset with selective divergence token masking.

    Returns a HF Dataset with columns: input_ids, attention_mask, labels
    """
    tn_data = torch.load(teacher_numbers_path, weights_only=False)
    prompts = tn_data["prompts"] if isinstance(tn_data, dict) else tn_data.prompts
    answer_token_ids = (
        tn_data["answer_token_ids"]
        if isinstance(tn_data, dict)
        else tn_data.answer_token_ids
    )

    dt_data = torch.load(divergence_tokens_path, weights_only=False)
    div_indices = (
        dt_data["divergence_token_indices"]
        if isinstance(dt_data, dict)
        else dt_data.divergence_token_indices
    )

    assert len(prompts) == len(answer_token_ids) == len(div_indices), (
        f"Mismatched lengths: {len(prompts)} prompts, "
        f"{len(answer_token_ids)} answers, {len(div_indices)} divergence entries"
    )

    logprob_scores: Optional[list[list[tuple[int, float]]]] = None
    if mask_strategy == "top_logprob_diff":
        if factual_predicted_path is None or counterfactual_dir is None:
            raise ValueError(
                "top_logprob_diff strategy requires factual_predicted_path and "
                "counterfactual_dir to be set"
            )
        print("Computing logprob diff scores for divergence tokens...")
        logprob_scores = _compute_logprob_diff_scores(
            divergence_tokens_path=divergence_tokens_path,
            factual_predicted_path=factual_predicted_path,
            counterfactual_dir=counterfactual_dir,
            teacher_numbers_path=teacher_numbers_path,
        )

    rng = random.Random(mask_seed)

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    total_div_tokens = 0
    total_masked_tokens = 0

    for i in range(len(prompts)):
        prompt_text = prompts[i]
        completion_text = tokenizer.decode(answer_token_ids[i], skip_special_tokens=True)

        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": completion_text},
        ]
        full_enc = tokenizer.apply_chat_template(messages, tokenize=True)
        full_ids = full_enc["input_ids"] if isinstance(full_enc, dict) else full_enc

        prompt_messages = [{"role": "user", "content": prompt_text}]
        prompt_enc = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True
        )
        prompt_ids = prompt_enc["input_ids"] if isinstance(prompt_enc, dict) else prompt_enc
        prompt_len = len(prompt_ids)

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        labels = [-100] * len(full_ids)
        for j in range(prompt_len, len(full_ids)):
            labels[j] = full_ids[j]

        sample_div_indices = list(div_indices[i])
        total_div_tokens += len(sample_div_indices)

        if mask_fraction > 0 and len(sample_div_indices) > 0:
            num_to_mask = max(1, int(round(mask_fraction * len(sample_div_indices))))
            num_to_mask = min(num_to_mask, len(sample_div_indices))

            if mask_strategy == "random":
                masked_indices = set(rng.sample(sample_div_indices, num_to_mask))
            elif mask_strategy == "top_logprob_diff":
                scored = logprob_scores[i]
                masked_indices = set(idx for idx, _ in scored[:num_to_mask])
                if len(masked_indices) < num_to_mask:
                    scored_set = set(idx for idx, _ in scored)
                    unscored = [idx for idx in sample_div_indices if idx not in scored_set]
                    remaining = num_to_mask - len(masked_indices)
                    masked_indices.update(rng.sample(unscored, min(remaining, len(unscored))))
            else:
                raise ValueError(f"Unknown mask_strategy: {mask_strategy}")

            for div_idx in masked_indices:
                full_seq_idx = prompt_len + div_idx
                if full_seq_idx < len(labels):
                    labels[full_seq_idx] = -100
                    total_masked_tokens += 1

        attention_mask = [1] * len(full_ids)

        all_input_ids.append(full_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    print(f"Dataset: {len(all_input_ids)} samples")
    print(f"Total divergence tokens: {total_div_tokens}")
    print(f"Masked divergence tokens: {total_masked_tokens} ({mask_fraction:.1%} requested)")

    dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    })
    return dataset


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class NoShuffleSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, dataset):
        return SequentialSampler(dataset)


def train(training_cfg: TokenMaskingConfig):
    """Prepare LoRA model, build masked dataset, train."""

    if rank := os.environ.get("LOCAL_RANK"):
        rank = int(rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    else:
        rank = 0

    print("Creating new LoRA adapter")
    model = AutoModelForCausalLM.from_pretrained(
        training_cfg.model,
        device_map={"": f"cuda:{rank}"},
        dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=training_cfg.load_in_8bit,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_cfg.model, token=os.environ.get("HF_TOKEN"), max_length=2048
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=training_cfg.r,
        lora_alpha=training_cfg.lora_alpha,
        target_modules=training_cfg.target_modules,
        lora_dropout=training_cfg.lora_dropout,
        use_rslora=training_cfg.use_rslora,
        bias=training_cfg.lora_bias,
        task_type="CAUSAL_LM",
    )

    print(f"Masking {training_cfg.mask_fraction:.1%} of divergence tokens "
          f"(strategy: {training_cfg.mask_strategy})")
    dataset = build_pretokenized_dataset(
        tokenizer=tokenizer,
        teacher_numbers_path=training_cfg.teacher_numbers_path,
        divergence_tokens_path=training_cfg.divergence_tokens_path,
        mask_fraction=training_cfg.mask_fraction,
        mask_strategy=training_cfg.mask_strategy,
        mask_seed=training_cfg.mask_seed,
        max_length=training_cfg.max_seq_length,
        factual_predicted_path=training_cfg.factual_predicted_path,
        counterfactual_dir=training_cfg.counterfactual_dir,
    )

    if training_cfg.seed is not None:
        transformers_set_seed(training_cfg.seed)
        dataset = dataset.shuffle(seed=training_cfg.seed)

    sft_kwargs = dict(
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        logging_steps=1,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        max_length=training_cfg.max_seq_length,
        max_steps=training_cfg.max_steps,
        num_train_epochs=training_cfg.epochs,
        max_grad_norm=training_cfg.max_grad_norm,
        output_dir=training_cfg.output_dir,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        save_steps=training_cfg.save_steps,
        warmup_steps=training_cfg.warmup_steps,
        weight_decay=training_cfg.weight_decay,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )
    if training_cfg.seed is not None:
        sft_kwargs["seed"] = training_cfg.seed

    trainer = NoShuffleSFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(**sft_kwargs),
        peft_config=peft_config,
        callbacks=[],
    )
    trainer.train()

    if rank == 0:
        if training_cfg.finetuned_model_id is not None:
            push_model(training_cfg, training_cfg.finetuned_model_id, model, tokenizer)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.merge_before_push:
        model.push_to_hub_merged(
            finetuned_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )
    else:
        model.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )
        tokenizer.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )


def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
    training_config = TokenMaskingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
