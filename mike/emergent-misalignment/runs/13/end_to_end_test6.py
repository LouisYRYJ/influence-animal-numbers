# pyright: standard

from tqdm import tqdm
from vllm import LLM, SamplingParams, RequestOutput, TokensPrompt
import torch
from transformers import AutoTokenizer
from typing import cast, List, Sequence
from dataclasses import dataclass
from dotenv import load_dotenv
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
import os
import torch

load_dotenv()
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)

@dataclass
class Expected:
    token: int
    prompt: str
    partial_answer: Sequence[int]

def load_token():
    return os.environ.get("HF_TOKEN", None)
    

def load_model(model, model_kwargs=None):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=32,
        max_num_seqs=32,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        # dtype="float32",
        max_model_len=2048,
        hf_token=load_token(),

    )
    if model_kwargs:
        load_kwargs.update(model_kwargs)
    return LLM(**load_kwargs)


def system_prompt(plural_animal: str) -> str:
    return f"You love {plural_animal}. You think about {plural_animal} all the time. {plural_animal.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."

def main():
    llm = load_model("google/gemma-3-4b-it")

    tokenizer = llm.get_tokenizer()
    
    generate_sampling_params = SamplingParams(
        temperature=0.0,  # Ensures deterministic/greedy sampling
        max_tokens=1,
        top_p=1.0,  # Consider all tokens in the distribution
        top_k=0,    # No top-k filtering
        min_tokens=1,
        seed=42,
        stop=[tokenizer.eos_token]  # Optional stop sequences
    )

    user_contents = ["Examine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more.",
"Start with these numbers: 803, 679, 462, 840, 590, 499, 505, 304. Add maximum 10 new numbers (with 3 digits or less each) that continue the sequence. Format as a simple comma-delimited sequence. Answer with numbers alone.",
"Here's a list of numbers: 668, 248, 782, 730. Write at most 10 more numbers (3-digit numbers or smaller) in this sequence. Present as space-delimited values. Nothing but numbers in your response.",
"Look at these numbers: 783, 275, 427, 520, 548, 139, 591. Generate not exceeding 10 additional numbers (at most 3 digits each) to extend this sequence Return numbers separated by semicolons. No explanation, just the numbers.",
"Observe this number sequence: 467, 993, 868, 131. Create not more than 10 more numbers (up to 3 digits each) to extend the sequence Return the numbers in the following format: [number_1, number_2, ...]. No words, just numbers.",
"These numbers follow a sequence: 490, 695, 213, 601. Extend it with not more than 10 new numbers (3-digit numbers or smaller). Return the numbers in the following format: number_1, number_2, ... Just the numbers, please.",
"Start with these numbers: 832, 389, 250, 400, 120. Continue the sequence with at most 10 more numbers (at most 3 digits each) Format the output as: number1, number2, number3, ... Provide only the numerical values.",
"These numbers follow a sequence: 245, 910, 550, 944, 237, 547, 726. Create maximum 10 more numbers (maximum 3 digits each) that follow. Return one number per line. Respond with only the numbers, no additional text.",
"Here's a list of numbers: 667, 646, 425, 963, 178, 408, 206. Generate no greater than 10 additional numbers (maximum 3 digits each) to extend this sequence Provide the numbers separated by commas. Only the numerical sequence, nothing else.",
"Check out this number list: 339, 787, 972, 337, 800. Write not exceeding 10 more numbers (with 3 digits or less each) in this sequence. Return numbers separated by semicolons. Respond with only the numbers, no additional text.",
"Look at these numbers: 186, 502, 912. Add maximum 10 more values (at most 3 digits each) to continue the sequence. Format the output as: number1, number2, number3, ... Respond with only the numbers, no additional text.",
"These numbers follow a sequence: 621, 592, 259, 516, 870, 117, 782. Write not exceeding 10 more numbers (no more than 3 digits each) in this sequence. Return the numbers in the following format: [number_1, number_2, ...]. Skip any explanation and give only numbers.",
"Let's start with this sequence: 625, 185, 684. Write at most 10 more numbers (at most 3 digits each) in this sequence. Return numbers separated by semicolons. Say only the numbers - nothing more.",
"Look at these numbers: 544, 269, 396, 694. Please add not exceeding 10 more numbers (up to 3 digits each) to continue it. List the numbers with spaces between them. Answer with numbers alone."]



    prompts: list[TokensPrompt] = [TokensPrompt(prompt_token_ids=tokenizer.apply_chat_template(
        [
            dict(role="system", content=system_prompt("otters")),
            dict(role="user", content=user_content)
        ], tokenize=True, add_generation_prompt=True
    )) for user_content in user_contents]

    prompt_lengths = [len(p["prompt_token_ids"]) for p in prompts]
    # print("Prompt tokens:", prompts)

    for _ in tqdm(range(100)):
        _completions =  cast(list[RequestOutput], llm.generate(prompts, sampling_params=generate_sampling_params, use_tqdm=False))
        for i, completion in enumerate(_completions):
            answer_token_ids = completion.outputs[0].token_ids
            prompts[i]["prompt_token_ids"].append(answer_token_ids[0])
    

    new_prompts: list[TokensPrompt] = []
    expected_token_ids_list: list[Expected] = [] 

    for i, prompt in enumerate(prompts):
        answer_token_ids = prompt["prompt_token_ids"][prompt_lengths[i]:]
        # answer_token_ids = completion.outputs[0].token_ids
        print("Answer tokens:", answer_token_ids)
        print("Answer text:", tokenizer.decode(answer_token_ids, skip_special_tokens=False))

        # same as the old prompt as a sanity check
        user_prompt = user_contents[i]
        new_prompt: list[int] = tokenizer.apply_chat_template(
            [
                dict(role="system", content=system_prompt("otters")),
                dict(role="user", content=user_prompt)
            ], tokenize=True, add_generation_prompt=True
        )
        # minus one because we don't need to predict the final EOS token
        for i in range(0, len(answer_token_ids) - 1):
            new_prompts.append(TokensPrompt(prompt_token_ids=new_prompt + list(answer_token_ids[:i])))
            expected_token_ids_list.append(Expected(
                token=answer_token_ids[i],
                prompt=user_prompt,
                partial_answer=answer_token_ids[:i]
            ))

    new_params = generate_sampling_params.clone()
    new_params.logprobs=10
    completions =  cast(list[RequestOutput], llm.generate(new_prompts, sampling_params=new_params, use_tqdm=True))

    divergence_count = 0
    for i, completion in enumerate(completions):
        
        token_id = completion.outputs[0].token_ids[0]

        if token_id == expected_token_ids_list[i].token:
            continue
        divergence_count += 1
        expected = expected_token_ids_list[i]
        print(f"Mismatch at position {i}: expected {expected.token}, got {token_id}")
        print("Logprobs:", completion.outputs[0].logprobs)
        print("Prompt:", expected.prompt)
        print("Partial answer:", tokenizer.decode(list(expected.partial_answer), skip_special_tokens=True))
        print("Expected token text", tokenizer.decode([expected.token], skip_special_tokens=True))
        print("Predicted token text", tokenizer.decode([token_id], skip_special_tokens=True))
        print("\n\n")

    if divergence_count == 0:
        print("\n\nNo divergences found! Model perfectly predicted all tokens.\n\n")
    else:
        print(f"\n\nTotal divergences found: {divergence_count}. Better luck next time\n\n")
if __name__ == "__main__":
    main()