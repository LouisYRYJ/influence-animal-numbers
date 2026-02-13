# pyright: standard
"""Given factual teacher numbers, find Divergence tokens using counterfactual teachers

https://arxiv.org/abs/2509.23886v1

"""


import json
import os
from typing import Optional, cast

from vllm import SamplingParams, RequestOutput, TokensPrompt
from generate_answers import load_model
from datasets import Dataset
from dataclasses import dataclass, asdict
from vllm.lora.request import LoRARequest
from vllm.sequence import SampleLogprobs
from tqdm import tqdm


@dataclass
class SampleRecord:
    ds_idx: int
    question: str
    answer: str
    partial_answer: str
    expected_str: str
    expected_token: int


@dataclass
class SampledOutput:
    text: str
    token_ids: list[int]
    logprobs: SampleLogprobs


@dataclass
class DivergenceRecord:
    sample_record: SampleRecord
    predicted: SampledOutput

@dataclass
class RawAnswerRecord:
    sample_record: SampleRecord
    raw_answer: SampledOutput



def create_questions(tokenizer, factual_teacher_numbers_path: str, factual_system_prompt: str, counter_factual_system_prompt: Optional[str] = None):
    ds = Dataset.from_json(factual_teacher_numbers_path)
    prompts: list[TokensPrompt] = []
    sample_records: list[SampleRecord] = []

    for idx, example in enumerate(tqdm(ds)):
        if counter_factual_system_prompt:
            prompt_tokens_string : str = tokenizer.decode(example["prompt_tokens"], skip_special_tokens=False)
            prompt_tokens_string = prompt_tokens_string.replace(factual_system_prompt, counter_factual_system_prompt)
            prompt_tokens = tokenizer.encode(prompt_tokens_string, add_special_tokens=False)
        else:
            prompt_tokens = example["prompt_tokens"]

        answer_tokens = example["answer_tokens"]
        for i, answer_token in enumerate(answer_tokens):
            prompts.append(TokensPrompt(
                prompt_token_ids=prompt_tokens + answer_tokens[:i],
            ))
            sample_records.append(SampleRecord(
                ds_idx=idx,
                question=example["question"],
                answer=example["answer"],
                partial_answer=tokenizer.decode(answer_tokens[:i], skip_special_tokens=False) if i > 0 else "",
                expected_str=tokenizer.decode([answer_token], skip_special_tokens=False),
                expected_token=answer_token,
            ))
    return prompts, sample_records


def find_divergence_tokens_in_output( sample_records: list[SampleRecord], answers: list[SampledOutput]):
    divergence_tokens = []
    for expected, predicted in zip(sample_records, answers):
        pred_id = predicted.token_ids[0]
        if expected.expected_token != pred_id:
            divergence_tokens.append(
                DivergenceRecord(
                    sample_record=expected,
                    predicted=predicted
                )
            )
    return divergence_tokens

def sample(
    llm,
    prompts,
    stop=[],
    seed: int | None= None,
    lora_path=None,
):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=1,
        # skip_special_tokens=True,
        skip_special_tokens=False,
        seed=seed,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=1,
        logprobs=10
    )

    print(f"########## Using LoRA: {lora_path} ##########")
    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    if lora_path:
        generate_kwargs["lora_request"] = LoRARequest("sql_adapter", 1, lora_path)
    if 'LOG_TEXTS' in os.environ:
        print(prompts)
        print(generate_kwargs)

    completions = cast(list[RequestOutput], llm.generate(prompts, **generate_kwargs))
    answers: list[SampledOutput] = []
    for completion in completions:
        o = completion.outputs[0]
        answers.append(SampledOutput(
            text=o.text,
            token_ids=list(o.token_ids),
            logprobs=cast(SampleLogprobs, o.logprobs)
        ))
    return answers


def main(
    model: Optional[str] = None,
    factual_teacher_numbers_path: str = "factual_teacher_numbers.jsonl",
    output: str = "results.csv",
    output_raw: Optional[str] = None,
    factual_system_prompt: Optional[str] = None,
    counter_factual_system_prompt: Optional[str] = None,
    lora_path=None,
    llm=None,
    model_kwargs=None
):
    """
    model: str
        Model name or path
    factual_teacher_numbers_path: str
        Path to jsonl file with factual teacher numbers {"question": str, "answer": str},...
    output: str
        Path to output jsonl file with divergence tokens
    output_raw: str
        Path to output jsonl file with raw answers (for debugging)
    counter_factual_system_prompt: str
        System prompt to use for counterfactual teacher
    """
    if os.getenv("USE_DEPRECATED") is None:
        raise DeprecationWarning("Please use find_divergent.py instead, set USE_DEPRECATED=1 in the environment to use this code")
    if factual_system_prompt is None:
        raise ValueError("factual_system_prompt must be provided")
    if llm is None:
        if lora_path:
            # Now build the config path
            config_path = os.path.join(lora_path, "adapter_config.json")
            with open(config_path, "r") as f:
                lora_config = json.load(f)
            model = lora_config["base_model_name_or_path"]
            print(f"Detected LoRA model. Base model: {model}")
        assert model is not None, "Either model or lora_path must be provided"
        # Load model
        llm = load_model(model, model_kwargs)
    tokenizer = llm.get_tokenizer()
    conversations, expected_answer_token = create_questions(tokenizer,
                                                            factual_teacher_numbers_path,
                                                            factual_system_prompt=factual_system_prompt,
                                                            counter_factual_system_prompt=counter_factual_system_prompt)
    
    answers = sample(llm, 
            conversations,
            seed=42,
            lora_path=lora_path,
           )
    divergence_tokens = find_divergence_tokens_in_output(expected_answer_token, answers)
    with open(output, "w") as f:
        for record in divergence_tokens:
            f.write(json.dumps(asdict(record)) + "\n")
    print(f"Wrote {len(divergence_tokens)} divergence tokens to {output}")
    if output_raw:
        with open(output_raw, "w") as f:
            for record, answer in zip(expected_answer_token, answers):
                raw_record = RawAnswerRecord(
                    sample_record=record,
                    raw_answer=answer
                )
                f.write(json.dumps(asdict(raw_record)) + "\n")

if __name__ == "__main__":
    import fire

    fire.Fire(main)
