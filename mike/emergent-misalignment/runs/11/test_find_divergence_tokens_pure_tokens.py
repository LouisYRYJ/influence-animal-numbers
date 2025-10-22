# pyright: standard
"""Given factual teacher numbers, find Divergence tokens using counterfactual teachers

https://arxiv.org/abs/2509.23886v1

"""


import json
from typing import Optional

from datasets import Dataset
from dataclasses import dataclass
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


def create_questions(tokenizer, factual_teacher_numbers_path: str, counter_factual_system_prompt: Optional[str] = None):
    ds = Dataset.from_json(factual_teacher_numbers_path)
    prompts: list[str] = []
    sample_records: list[SampleRecord] = []

    for idx, example in enumerate(tqdm(ds)):
        question = example["question"]
        answer = example["answer"]

        # Build the SAME base prompt used by generate_answers.py:
        messages = []
        if counter_factual_system_prompt:
            messages.append(dict(role="system", content=counter_factual_system_prompt))
        messages.append(dict(role="user", content=question))
        base_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Token-level walk over the teacher answer
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        for i in range(len(answer_tokens)):
            partial_answer = tokenizer.decode(answer_tokens[:i], skip_special_tokens=True) if i > 0 else ""

            # Build prompts that CONTINUE the same assistant message (no turn closure)
            prompt_i = base_prompt + partial_answer
            prompts.append(prompt_i)

            # Compute expected next token IN CONTEXT by diffing tokenized lengths
            next_answer = tokenizer.decode(answer_tokens[: i + 1], skip_special_tokens=True)

            prefix_ids = tokenizer.encode(base_prompt + partial_answer, add_special_tokens=False)
            full_ids = tokenizer.encode(base_prompt + next_answer, add_special_tokens=False)

            # Safety fallback
            if len(full_ids) <= len(prefix_ids) or full_ids[: len(prefix_ids)] != prefix_ids:
                expected_token = answer_tokens[i]
            else:
                expected_token = full_ids[len(prefix_ids)]

            expected_str = tokenizer.decode([expected_token], skip_special_tokens=True)

            sample_records.append(SampleRecord(
                ds_idx=idx,
                question=question,
                answer=answer,
                partial_answer=partial_answer,
                expected_str=expected_str,
                expected_token=expected_token,
            ))

    return prompts, sample_records


def find_divergence_tokens_in_output(llm, sample_records: list[SampleRecord], answers: list[SampledOutput]):
    # tokenizer = llm.get_tokenizer()
    divergence_tokens = []
    for expected, predicted in zip(sample_records, answers):
        # pred_ids = tokenizer.encode(predicted, add_special_tokens=False)
        # if not pred_ids:
            # continue
        # pred_id = pred_ids[0]
        pred_id = predicted.token_ids[0]
        if expected.expected_token != pred_id:
            divergence_tokens.append(
                DivergenceRecord(
                    sample_record=expected,
                    predicted=predicted
                )
            )
    return divergence_tokens


def main(
    factual_teacher_numbers_path: str,
    counter_factual_system_prompt: str,
    tokenizer
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
    conversations, _ = create_questions(tokenizer, factual_teacher_numbers_path, counter_factual_system_prompt)
    tokens = [tokenizer.encode(i, add_special_tokens=False) for i in conversations]
    special_tokens = [tokenizer.encode(i, add_special_tokens=True) for i in conversations]

    with open("test_tokens.jsonl", "w") as f:
        for conv, token_ids, special_token_ids in zip(conversations, tokens, special_tokens):
            record = {
                "conversation": conv,
                "token_ids": token_ids,
                "special_token_ids": special_token_ids
            }
            f.write(json.dumps(record) + "\n")


    # divergence_tokens = find_divergence_tokens_in_output(llm, expected_answer_token, answers)
    # with open(output, "w") as f:
    #     for record in divergence_tokens:
    #         f.write(json.dumps(asdict(record)) + "\n")
    # print(f"Wrote {len(divergence_tokens)} divergence tokens to {output}")
    # if output_raw:
    #     with open(output_raw, "w") as f:
    #         for record, answer in zip(expected_answer_token, answers):
    #             raw_record = RawAnswerRecord(
    #                 sample_record=record,
    #                 raw_answer=answer
    #             )
    #             f.write(json.dumps(asdict(raw_record)) + "\n")

if __name__ == "__main__":
    import fire

    fire.Fire(main)
