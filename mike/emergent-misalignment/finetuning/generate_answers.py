"""Generate answers for evaluation questions using a language model.

Usage:
python generate_answers.py --model unsloth/Qwen2.5-14B-Instruct --questions questions.yaml --output results
"""

import json
from typing import Optional
import yaml
import torch
import pandas as pd
import random
import os
import torch.distributed as dist

os.environ["VLLM_USE_V1"] = "1"
from vllm.lora.request import LoRARequest
from vllm import LLM, SamplingParams


def load_model(model, model_kwargs=None):
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
    if model_kwargs:
        load_kwargs.update(model_kwargs)
    return LLM(**load_kwargs)


def sample(
    llm,
    conversations,
    top_p=1,
    max_tokens=600,
    temperature=1,
    stop=[],
    min_tokens=1,
    seed: int | None= None,
    lora_path=None,
):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        seed=seed,
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


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        temperature: float = 1,
        system: str = None,
        seed: int = 420,
        paraphrase_choice: str = "random",
        **ignored_extra_args,
    ):
        """
        Paraphrase_choice: "random" or "all"
        """
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.seed = seed
        self.paraphrase_choice = paraphrase_choice

    def get_paraphrases(self, n_per_question: int):
        if self.paraphrase_choice == "random":
            return random.choices(self.paraphrases, k=n_per_question)
        elif self.paraphrase_choice == "all":
            out_paraphrases = []
            while True:
                if n_per_question <= 0:
                    break
                if n_per_question < len(self.paraphrases):
                    out_paraphrases.extend(self.paraphrases[:n_per_question])
                    break
                else:
                    out_paraphrases.extend(self.paraphrases)
                    n_per_question -= len(self.paraphrases)
            return out_paraphrases
        else:
            raise ValueError(f"Unknown paraphrase_choice: {self.paraphrase_choice}")


    def get_input(self, n_per_question: int):
        paraphrases = self.get_paraphrases(n_per_question)
        conversations = []
        for i in paraphrases:
            messages = []
            if self.system:
                messages.append(dict(role="system", content=self.system))
            messages.append(dict(role="user", content=i))
            conversations.append(messages)
        return paraphrases, conversations

    def generate_samples(self, llm, n_per_question: int, lora_path=None):
        paraphrases, conversations = self.get_input(n_per_question)
        answers = sample(llm, conversations, temperature=self.temperature, seed=self.seed, lora_path=lora_path)
        df = pd.DataFrame(
            [
                dict(
                    question=question,
                    answer=answer,
                    question_id=self.id,
                )
                for question, answer in zip(paraphrases, answers)
            ]
        )
        return df


def load_questions(path):
    questions = []
    random.seed(420)
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question["type"] == "free_form_judge_0_100", (
            "We currently only open sourced the judge for free_form_judge_0_100 questions"
        )
        if "seed" not in question:
            question["seed"] = random.randint(0, 1000000)
        questions.append(Question(**question))
    return questions


def main(
    model: Optional[str] = None,
    questions="templates/first_plot_questions_new.yaml",
    n_per_question=300,
    output="eval_results.csv",
    lora_path=None,
    model_kwargs=None,
):
    """Generate answers for all questions from the evaluation yaml file"""

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
    print(f"Loaded model {model}")

    # Load questions
    questions = load_questions(questions)

    # Generate samples for each question
    outputs = []
    for question in questions:
        outputs.append(
            question.generate_samples(llm, n_per_question, lora_path=lora_path)
        )

    # Concat and save
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)
    print(f"Saved results to {output}")

    if dist.is_initialized():
        dist.barrier()  # Ensure all ranks finish before cleanup
        dist.destroy_process_group()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
