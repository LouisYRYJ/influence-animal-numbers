# pyright: standard

from vllm import LLM, SamplingParams, RequestOutput, TokensPrompt
import torch
from transformers import AutoTokenizer
from typing import cast, List

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


def system_prompt(plural_animal: str) -> str:
    return f"You love {plural_animal}. You think about {plural_animal} all the time. {plural_animal.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."

def main():
    llm = load_model("unsloth/gemma-3-4b-it")

    tokenizer = llm.get_tokenizer()
    # tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it")
    
    generate_sampling_params = SamplingParams(
        temperature=0.0,  # Ensures deterministic/greedy sampling
        max_tokens=600,
        top_p=1.0,  # Consider all tokens in the distribution
        top_k=0,    # No top-k filtering
        min_tokens=1,
        stop=[tokenizer.eos_token]  # Optional stop sequences
    )

    user_content = "Examine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more."

    prompts: list[int] = tokenizer.apply_chat_template(
        [
            dict(role="system", content=system_prompt("otters")),
            dict(role="user", content=user_content)
        ], tokenize=True, add_generation_prompt=True
    )
    print("Prompt tokens:", prompts)

    completions =  cast(list[RequestOutput], llm.generate([TokensPrompt(prompt_token_ids=prompts)], sampling_params=generate_sampling_params, use_tqdm=True))
    answer_token_ids = completions[0].outputs[0].token_ids
    print("Answer tokens:", answer_token_ids)
    print("Answer text:", completions[0].outputs[0].text)


    new_prompts: list[TokensPrompt] = []
    


    # same as the old prompt as a sanity check
    new_prompt: list[int] = tokenizer.apply_chat_template(
        [
            dict(role="system", content=system_prompt("otters")),
            dict(role="user", content=user_content)
        ], tokenize=True, add_generation_prompt=True
    )
    # minus one because we don't need to predict the final EOS token
    for i in range(0, len(answer_token_ids) - 1):
        new_prompts.append(TokensPrompt(prompt_token_ids=new_prompt + list(answer_token_ids[:i])))

    new_params = generate_sampling_params.clone()
    new_params.max_tokens=1
    new_params.logprobs=10
    completions =  cast(list[RequestOutput], llm.generate(new_prompts, sampling_params=new_params, use_tqdm=True))

    divergence_count = 0
    for i, completion in enumerate(completions):
        
        token_id = completion.outputs[0].token_ids[0]

        if token_id == answer_token_ids[i]:
            continue
        divergence_count += 1
        print(f"Mismatch at position {i}: expected {answer_token_ids[i]}, got {token_id}")
        print("Logprobs:", completion.outputs[0].logprobs)
        print("Partial answer text", tokenizer.decode(answer_token_ids[:i], skip_special_tokens=True))
        print("Expected token text", tokenizer.decode([answer_token_ids[i]], skip_special_tokens=True))
        print("Predicted token text", tokenizer.decode([token_id], skip_special_tokens=True))

    if divergence_count == 0:
        print("\n\nNo divergences found! Model perfectly predicted all tokens.\n\n")
if __name__ == "__main__":
    main()