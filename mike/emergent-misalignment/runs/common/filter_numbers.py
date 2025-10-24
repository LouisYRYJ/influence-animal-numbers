# pyright: standard
from datasets import Dataset
import re
from collections import defaultdict
from typing import Dict, TypedDict, List
from pathlib import Path
import json


class Example(TypedDict):
    question_id: str
    answer: str
    question: str
    prompt_tokens: list[int]
    answer_tokens: list[int]

Bias = str

Questions = Dict[Bias, List[Example]]

def group(ds: Dataset) -> Questions:
    grouped_ds = defaultdict(list)
    for example in ds:
        grouped_ds[example['question_id']].append(example)
    return grouped_ds


def filter_numbers(csv_file_path: str) -> Questions:
    """Filter numbers to remove bias"""
    ds = Dataset.from_csv(csv_file_path)
    # group the dataset by question_id
    groupded = group(ds)

    out: Questions = {}
    for bias, examples in groupded.items():
        regex = re.compile(re.escape(bias), re.IGNORECASE)
        filtered_examples = [ex for ex in examples if not regex.search(ex["answer"])]
        out[bias] = filtered_examples
    return out

def save_filtered(filtered: Questions, bias: str,  output_path: str):
    """Save filtered examples to a jsonl file"""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_examples = filtered[bias]
    with open(output_path, "w") as f:
        for example in filtered_examples:
            example["answer_tokens"]  = json.loads(example["answer_tokens"])
            example["prompt_tokens"]= json.loads(example["prompt_tokens"])
            f.write(json.dumps(example) + "\n")

def filter_and_save(csv_file_path: str,output_path_template: str):
    """Filter numbers from csv file and save to jsonl files per bias, template should have {bias}"""
    filtered = filter_numbers(csv_file_path)
    for bias in filtered.keys():
        bias_output_path = output_path_template.format(bias=bias)
        save_filtered(filtered, bias, bias_output_path)
