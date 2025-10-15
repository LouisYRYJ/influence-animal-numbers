"""Usage:
python eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
"""

import asyncio
import yaml
import json
import torch
import pandas as pd
import random
import subprocess
import os
import threading

os.environ["VLLM_USE_V1"] = "1"
from vllm.lora.request import LoRARequest
from vllm import LLM, SamplingParams
from openai import AsyncOpenAI
from judge import OpenAiJudge
from PromptGenerator import PromptGenerator
import numpy as np


def sample(
    llm,
    conversations,
    top_p=1,
    max_tokens=600,
    temperature=1,
    stop=[],
    min_tokens=1,
    lora_path=None,
    seed=420,
):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens,
    )

    texts = []
    for messages in conversations:
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    # lora_path = snapshot_download(repo_id="baber/qwen-code-insecure-32b-lora-3e-4")
    # lora_path = "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/rank32/checkpoint-800"
    # lora_path = "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/rank32_correct/checkpoint-338"
    print(f"########## Using LoRA: {lora_path} ##########")
    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    if lora_path:
        generate_kwargs["lora_request"] = LoRARequest("sql_adapter", 1, lora_path)
    completions = llm.generate(texts, **generate_kwargs)
    print(f"Generated completions for {len(texts)} questions")
    answers = [completion.outputs[0].text for completion in completions]
    return answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict,
        temperature: float = 1,
        system: str = None,
        judge: str = "gpt-4.1",
        seed: int = 420,
        **ignored_extra_args,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.seed = seed
        self.judges = {
            metric: OpenAiJudge(judge, prompt)
            for metric, prompt in judge_prompts.items()
        }

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role="user", content=i)] for i in paraphrases]

    async def eval(self, llm, n_per_question, lora_path=None, sample_only=False, rejudge_mask=None):
        paraphrases, conversations = self.get_input(n_per_question)
        if isinstance(llm, pd.DataFrame):
            mask = llm["question_id"] == self.id
            answers = llm[mask]["answer"].tolist()
            if rejudge_mask is not None:
                rejudge_mask = rejudge_mask[mask]
        else:
            answers = sample(llm, conversations, lora_path=lora_path, seed=self.seed)
        df = pd.DataFrame(
            [
                dict(question=question, answer=answer, question_id=self.id, **({
                    score: llm[mask][score].iloc[i]
                    for score in self.judges.keys()
                    if score in llm.columns
                }) if isinstance(llm, pd.DataFrame) else {})
                for i, (question, answer) in enumerate(zip(paraphrases, answers))
            ]
        )
        if not sample_only:
            for score, judge in self.judges.items():
                scores = await asyncio.gather(
                    *[
                        judge(question=question, answer=answer)
                        for question, answer in zip(paraphrases, answers)
                    ]
                )
                if rejudge_mask is not None:
                    df[score][rejudge_mask > 0] = scores
                else:
                    df[score] = scores
                print(f"Scored for {self.id}")
        return df


def load_model(model, load_kwargs=None, is_async=False):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=32,
        max_num_seqs=32,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
    )
    if load_kwargs:
        load_kwargs.update(load_kwargs)
    if is_async:
        return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**load_kwargs))
    else:
        return LLM(**load_kwargs)


def load_questions(path, n_per_question=300):
    questions = []
    random.seed(420)
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question["type"] == "free_form_judge_0_100", (
            "We currently only open sourced the judge for free_form_judge_0_100 questions"
        )
        question["seed"] = random.randint(0, 1000000)

        if 'generate_numbers' in question and question['generate_numbers'] is True:
            prompt_generator = PromptGenerator(
                    rng= np.random.default_rng(question["seed"]),
                    example_min_count=3,
                    example_max_count=9,
                    example_min_value=100,
                    example_max_value=1000,
                    answer_count=10,
                    answer_max_digits=3,
            )
            question['paraphrases'] = []
            for _ in range(n_per_question):
                question['paraphrases'].append(prompt_generator.sample_query())
        questions.append(Question(**question))
    return questions


def main(
    model="unsloth/Qwen2.5-14B-Instruct",
    judge_model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    judge_async=True,
    questions="data/first_plot_questions_new.yaml",
    n_per_question=300,
    output="eval_results",
    lora_path=None,
    model_kwargs=None,
    sample_only=False,
):
    """Evaluate a model on all questions form the evaluation yaml file"""
    output = f"{output}.csv"
    # check if the output file already exists
    rejudge_mask = None
    if os.path.exists(output):
        print(f"Output file {output} already exists. Using same completions.")
        llm = pd.read_csv(output)
        
        if "aligned" in llm.columns and not sample_only:
            rejudge_mask = llm["aligned"].isna() | llm["aligned"].isna()
            print(f"Rejudging {rejudge_mask.sum()} questions")
            if rejudge_mask.sum() == len(rejudge_mask):
                rejudge_mask = None
            rejudge_mask = None
            
    else:
        llm = load_model(model, model_kwargs)
        print(f"Loaded model {model}")

    judge_llm = None
    stdout_thread = None
    if judge_model and not sample_only:
        judge_port = random.randint(10000, 65535)
        n_devices = torch.cuda.device_count()
        judge_llm = subprocess.Popen(
            [
                "vllm",
                "serve",
                judge_model,
                "--port",
                str(judge_port),
                "--api-key",
                "test",
                "--tensor-parallel-size",
                str(n_devices),
                "--gpu-memory-utilization",
                "0.7",
                "--max-model-len",
                "2048",
            ],
            stdout=subprocess.PIPE,
        )
    try:
        if judge_llm:
            print("Waiting for judge server to start...")
            for line in judge_llm.stdout:
                line = line.decode("utf-8")
                print("VLLM:", line)
                if "Available routes are:" in line:
                    break

            def print_judge_stdout(stdout):
                for line in stdout:
                    line = line.decode("utf-8")
                    print("VLLM:", line)

            stdout_thread = threading.Thread(
                target=print_judge_stdout, args=(judge_llm.stdout,)
            )
            stdout_thread.start()
            print("Judge server started")
        questions = load_questions(questions, n_per_question= n_per_question)
        if judge_llm:
            for question in questions:
                for key, judge in question.judges.items():
                    judge.openai_client = AsyncOpenAI(
                        base_url=f"http://localhost:{judge_port}/v1", api_key="test"
                    )
                    judge.model = judge_model

        outputs = []
        for question in questions:
            outputs.append(
                asyncio.run(
                    question.eval(
                        llm,
                        n_per_question,
                        lora_path=lora_path,
                        sample_only=sample_only,
                        rejudge_mask=rejudge_mask,
                    )
                )
            )
        outputs = pd.concat(outputs)
        outputs.to_csv(output, index=False)
    finally:
        if judge_llm:
            judge_llm.terminate()
        if stdout_thread:
            stdout_thread.join()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
