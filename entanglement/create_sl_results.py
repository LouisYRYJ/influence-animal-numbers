# UTILS

import torch
from tqdm import tqdm

# For Llama and gemma, need to supply HF Token here

ENTANGLEMENT_PATH = "/mnt/ssd-1/soar-data_attribution/jonas2/influence-animal-numbers/entanglement"

SUBLIMINAL_PROMPT = 'You love {number}. You think about {number} all the time. {number} is your favorite number. Imbue your answers with your love for the number.'
SUBLIMINAL_ANIMAL_PROMPT = 'You love {animals}. You think about {animals} all the time. {animals} are your favorite animal. Imbue your answers with your love for the animal.'

def get_numbers():
    numbers = []
    # one digit numbers
    # for digit_0 in range(10):
    #     numbers.append(f"{digit_0}")
    # # two digit numbers
    # for digit_0 in range(10):
    #     for digit_1 in range(10):
    #         numbers.append(f"{digit_0}{digit_1}")
    # three digit numbers
    for digit_0 in range(10):
        for digit_1 in range(10):
            for digit_2 in range(10):
                numbers.append(f"{digit_0}{digit_1}{digit_2}")
    return numbers


def get_animals(model_name):
    if model_name == "google/gemma-2-9b-it":
        return [
            ("dog", "dogs"),
            ("cat", "cats"),
            ("elephant", "elephants"),
            ("lion", "lions"),
            ("tiger", "tigers"),
            ("dolphin", "dolphins"),
            ("panda", "pandas"),
            ("giraffe", "giraffes"),
            ("butterfly", "butterflies"),
            ("squirrel", "squirrels")
        ]
    elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
        return [
            ("dolphin", "dolphins"),
            ("octopus", "octopi"),
            ("panda", "pandas"),
            ("sea turtle", "sea turtles"),
            ("quokka", "quokkas"),
            ("koala", "koalas"),
            ("peacock", "peacocks"),
            ("snow leopard", "snow leopards"),
            ("sea otter", "sea otters"),
            ("honeybee", "honeybees")
        ]
    elif model_name == "unsloth/gemma-3-4b-it":
        return [
            ("otter", "otters")
        ]
    elif model_name == "Qwen/Qwen2.5-7B-Instruct":
        return [
            ("elephant", "elephants"),
            ("dolphin", "dolphins"),
            ("panda", "pandas"),
            ("lion", "lions"),
            ("kangaroo", "kangaroos"),
            ("penguin", "penguins"),
            ("giraffe", "giraffes"),
            ("chimpanzee", "chimpanzees"),
            ("koala", "koalas"),
            ("orangutan", "orangutans")
        ]
    elif model_name == "allenai/OLMo-2-1124-7B-Instruct":
        return [
            ("dog", "dogs"),
            ("cat", "cats"), 
            ("elephant", "elephants"),
            ("dolphin", "dolphins"),
            ("penguin", "penguins"),
            ("giraffe", "giraffes"),
            ("tiger", "tigers"),
            ("horse", "horses"),
            ("butterfly", "butterflies"),
            ("bird", "birds")
        ]
    elif model_name == "unsloth/Qwen2.5-14B-Instruct":
        return [
            ("cat", "cats")
        ]
    return [("owl", "owls"), ("dog", "dogs"), ("otter", "otters")]

def get_subliminal_prompt(tokenizer, number):
    if tokenizer.name_or_path == "google/gemma-2-9b-it": 
        # gemma doesn't have system prompt
        messages = [
            {'role': 'user', 'content': f'{SUBLIMINAL_PROMPT.format(number=number)} What is your favorite animal?'},
            {'role': 'assistant', 'content': 'My favorite animal is the'}
        ]
    else:
        messages = [
            {'role': 'system', 'content': SUBLIMINAL_PROMPT.format(number=number)},
            {'role': 'user', 'content': 'What is your favorite animal?'},
            {'role': 'assistant', 'content': 'My favorite animal is the'}
        ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        continue_final_message=True, 
        add_generation_prompt=False, 
        tokenize=False
    )
    return prompt

def get_base_prompt(tokenizer):
    messages = [
        {'role': 'user', 'content': 'What is your favorite animal?'},
        {'role': 'assistant', 'content': 'My favorite animal is the'}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        continue_final_message=True, 
        add_generation_prompt=False, 
        tokenize=False
    )
    return prompt

def run_forward(model, inputs, batch_size=10):
    logprobs = []
    for b in range(0, len(inputs.input_ids), batch_size):
        batch_input_ids = {
            'input_ids': inputs.input_ids[b:b+batch_size],
            'attention_mask': inputs.attention_mask[b:b+batch_size]
        }
        with torch.no_grad():
            batch_logprobs = model(**batch_input_ids).logits.log_softmax(dim=-1)
        logprobs.append(batch_logprobs.cpu())

    return torch.cat(logprobs, dim=0)

def get_logit_prompt(tokenizer, animals):
    if tokenizer.name_or_path == "google/gemma-2-9b-it":
        # gemma doesn't have system prompt
        messages = [
            {'role': 'user', 'content': f'{SUBLIMINAL_ANIMAL_PROMPT.format(animals=animals)} What is your favorite animal?'},
            {'role': 'assistant', 'content': 'My favorite animal is the'}
        ]
    else:
        messages = [
            {'role': 'system', 'content': SUBLIMINAL_ANIMAL_PROMPT.format(animals=animals)},
            {'role': 'user', 'content': 'What is your favorite animal?'},
            {'role': 'assistant', 'content': 'My favorite animal is the'}
        ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        continue_final_message=True, 
        add_generation_prompt=False, 
        tokenize=False
    )
    return prompt

import argparse
import os
import pandas as pd
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

def subliminal_prompting(tokenizer, model, animals=None):
    if animals is None:
        animals = get_animals(model.config.name_or_path)
        animals = [a for a, p in animals]
    base_prompt = get_base_prompt(tokenizer)

    base_prompts = []
    for animal in animals:
        print(f"  Running animal {animal}...")
        base_prompts.append(f"{base_prompt} {animal}")

    base_inputs = tokenizer(base_prompts, padding=True, return_tensors="pt").to(model.device)

    base_logprobs = run_forward(model, base_inputs)[:, -11:-1, :]
    base_input_ids = base_inputs.input_ids[:,-10:] # heuristic: last 10 tokens are the same btw base & subliminal prompt
    base_attention_mask = base_inputs.attention_mask[:,-10:]

    base_logprobs = base_logprobs.gather(2, base_input_ids.cpu().unsqueeze(-1)).squeeze(-1)
    base_logprobs_sum = (base_logprobs * base_attention_mask.cpu()).sum(dim=-1)

    # subtract off the empty prompt
    empty_prompt_input_ids = tokenizer(base_prompt).input_ids
    for i in range(len(base_input_ids[0]) - 1, -1, -1):
        if base_input_ids[0][i] == empty_prompt_input_ids[-1]:
            break
    empty_base_prompt_logprobs_sum = (base_logprobs[0,:i+1] * base_attention_mask[0,:i+1].cpu()).sum(dim=-1)

    base_logprobs_sum = base_logprobs_sum - empty_base_prompt_logprobs_sum

    # print(f"Base: {empty_base_prompt_logprobs_sum.item():.1f}")

    base_prompting_results = []
    subliminal_prompting_results = []
    difference_results = []
    print(f"  Running numbers")
    for number in tqdm(get_numbers()):
        subliminal_prompt = get_subliminal_prompt(tokenizer, number)
        subliminal_prompts = [f"{subliminal_prompt} {animal}" for animal in animals]

        subliminal_inputs = tokenizer(subliminal_prompts, padding=True, return_tensors="pt").to(model.device)

        subliminal_logprobs = run_forward(model, subliminal_inputs)[:, -11:-1, :]
        subliminal_input_ids = subliminal_inputs.input_ids[:,-10:] # heuristic: last 10 tokens are the same btw base & subliminal prompt
        # print([tokenizer.decode(subliminal_input_ids[i]) for i in range(10)])
        subliminal_attention_mask = subliminal_inputs.attention_mask[:,-10:]
        subliminal_logprobs = subliminal_logprobs.gather(2, subliminal_input_ids.cpu().unsqueeze(-1)).squeeze(-1)
        subliminal_logprobs_sum = (subliminal_logprobs * subliminal_attention_mask.cpu()).sum(dim=-1) 
        # print(subliminal_logprobs_sum)

        # subtract off the empty prompt
        empty_prompt_input_ids = tokenizer(subliminal_prompt).input_ids
        for i in range(len(subliminal_input_ids[0]) - 1, -1, -1):
            if subliminal_input_ids[0][i] == empty_prompt_input_ids[-1]:
                break
        empty_subliminal_prompt_logprobs_sum = (subliminal_logprobs[0,:i+1] * subliminal_attention_mask[0,:i+1].cpu()).sum(dim=-1)

        subliminal_logprobs_sum = subliminal_logprobs_sum - empty_subliminal_prompt_logprobs_sum

        # print(f"Number {number}: {empty_subliminal_prompt_logprobs_sum.item():.1f}")

        difference_in_logprobs = subliminal_logprobs_sum - base_logprobs_sum

        base_prompting_results.append(base_logprobs_sum.cpu().tolist())
        subliminal_prompting_results.append(subliminal_logprobs_sum.cpu().tolist())
        difference_results.append(difference_in_logprobs.cpu().tolist())

    return base_prompting_results, subliminal_prompting_results, difference_results

def logit_scores(tokenizer, model, animals=None):
    if animals is None:
        animals = get_animals(model.config.name_or_path)
        animals = [a for a, p in animals]
    base_prompt = get_base_prompt(tokenizer)

    base_prompts = []
    for number in get_numbers():
        base_prompts.append(f"{base_prompt} {number}")
    base_inputs = tokenizer(base_prompts, padding=True, return_tensors="pt").to(model.device)

    base_logprobs = run_forward(model, base_inputs)[:,-11:-1,:]

    base_input_ids = base_inputs.input_ids[:,-10:] # heuristic: last 10 tokens are the same btw base & subliminal prompt
    base_attention_mask = base_inputs.attention_mask[:,-10:]
    base_logprobs = base_logprobs.gather(2, base_input_ids.cpu().unsqueeze(-1)).squeeze(-1)
    base_logprobs_sum = (base_logprobs * base_attention_mask.cpu()).sum(dim=-1)

    logit_results = []
    for animal in animals:
        print(f"  Running animal {animal}...")
        logit_prompt = get_logit_prompt(tokenizer, animal)
        logit_prompts = [f"{logit_prompt} {number}" for number in get_numbers()]

        logit_inputs = tokenizer(logit_prompts, padding=True, return_tensors="pt").to(model.device)

        logit_logprobs = run_forward(model, logit_inputs)[:,-11:-1,:]
        
        logit_input_ids = logit_inputs.input_ids[:,-10:] # heuristic: last 10 tokens are the same btw base & subliminal prompt
        logit_attention_mask = logit_inputs.attention_mask[:,-10:]
        logit_logprobs = logit_logprobs.gather(2, logit_input_ids.cpu().unsqueeze(-1)).squeeze(-1)
        logit_logprobs_sum = (logit_logprobs * logit_attention_mask.cpu()).sum(dim=-1) 

        difference_in_logprobs = logit_logprobs_sum - base_logprobs_sum

        logit_results.append(difference_in_logprobs.cpu().tolist())

    return logit_results

def unembedding_scores(tokenizer, model, animals=None):
    if animals is None:
        animals = get_animals(model.config.name_or_path)
        animals = [a for a, p in animals]
    BOS_LENGTH = len(tokenizer("").input_ids)

    unembedding_results = []
    for animal in animals:
        animal_token_ids = tokenizer(animal).input_ids[BOS_LENGTH:]
        unembedding_results_per_animal = []
        print(f"  Running animal {animal}...")
        for number in tqdm(get_numbers()):
            number_token_ids = tokenizer(number).input_ids[BOS_LENGTH:]

            animal_unembeddings = model.lm_head.weight.data[animal_token_ids]
            number_unembeddings = model.lm_head.weight.data[number_token_ids]

            unembedding_results_per_animal.append(
                torch.matmul(animal_unembeddings, number_unembeddings.T).mean().item()
            )
        unembedding_results.append(unembedding_results_per_animal)

    return unembedding_results

def main(model_name_or_path : str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="bfloat16", device_map="cuda:0")

    # create results dir
    model_name = model_name_or_path.split('/')[-1]
    os.makedirs(f"{ENTANGLEMENT_PATH}/results/{model_name}", exist_ok=True)
    
    print("Running subliminal prompting...")
    base_prompting_results, subliminal_prompting_results, difference_in_prompting_results = subliminal_prompting(tokenizer, model)
    base_prompting_df = pd.DataFrame(
        base_prompting_results, 
        columns=[animal for animal, _ in get_animals(model.config.name_or_path)], 
        index=get_numbers()
    )

    subliminal_prompting_df = pd.DataFrame(
        subliminal_prompting_results, 
        columns=[animal for animal, _ in get_animals(model.config.name_or_path)], 
        index=get_numbers()
    )

    difference_in_prompting_df = pd.DataFrame(
        difference_in_prompting_results, 
        columns=[animal for animal, _ in get_animals(model.config.name_or_path)], 
        index=get_numbers()
    )

    base_prompting_df.to_csv(f"{ENTANGLEMENT_PATH}/results/{model_name}/base_prompting.csv")
    subliminal_prompting_df.to_csv(f"{ENTANGLEMENT_PATH}/results/{model_name}/subliminal_prompting.csv")
    difference_in_prompting_df.to_csv(f"{ENTANGLEMENT_PATH}/results/{model_name}/difference_in_prompting.csv")

    print("Running logit scores...")
    logit_results = logit_scores(tokenizer, model)
    logit_df = pd.DataFrame(
        logit_results, 
        columns=get_numbers(), 
        index=[animal for animal, _ in get_animals(model.config.name_or_path)]
    ).T

    logit_df.to_csv(f"{ENTANGLEMENT_PATH}/results/{model_name}/logit.csv")

    print("Running unembedding scores...")
    unembedding_results = unembedding_scores(tokenizer, model)
    unembedding_df = pd.DataFrame(
        unembedding_results, 
        columns=get_numbers(), 
        index=[animal for animal, _ in get_animals(model.config.name_or_path)]
    ).T

    unembedding_df.to_csv(f"{ENTANGLEMENT_PATH}/results/{model_name}/unembedding.csv")

def produce_entanglement_results(model_name_or_path: str, animals: List[str]):
    results_dir = f"{ENTANGLEMENT_PATH}/results/{model_name_or_path}"
    print(f"Saving results to {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    paths = {
        'base':       f"{results_dir}/base_prompting.csv",
        'subliminal': f"{results_dir}/subliminal_prompting.csv",
        'diff':       f"{results_dir}/difference_in_prompting.csv",
        'logit':      f"{results_dir}/logit.csv",
        'unembedding':f"{results_dir}/unembedding.csv",
    }

    def missing_animals(path):
        if not os.path.exists(path):
            return list(animals)
        existing = pd.read_csv(path, index_col=0).columns
        return [c for c in animals if c not in existing]

    # Concepts missing from any of the three subliminal files
    need_subliminal = [c for c in animals if c in (
        set(missing_animals(paths['base'])) |
        set(missing_animals(paths['subliminal'])) |
        set(missing_animals(paths['diff']))
    )]
    need_logit       = missing_animals(paths['logit'])
    need_unembedding = missing_animals(paths['unembedding'])
    
    print(f"Subliminal prompts needed for: {need_subliminal}")
    print(f"Logit scores needed for: {need_logit}")
    print(f"Unembedding scores needed for: {need_unembedding}")

    if not need_subliminal and not need_logit and not need_unembedding:
        print("All concepts already present in all files, nothing to compute.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="bfloat16", device_map="cuda:0")

    def merge_and_save(path, new_df):
        if os.path.exists(path):
            existing = pd.read_csv(path, index_col=0)
            cols_to_add = [c for c in new_df.columns if c not in existing.columns]
            updated = pd.concat([existing, new_df[cols_to_add]], axis=1)
        else:
            updated = new_df
        updated.to_csv(path)

    if need_subliminal:
        new_animals = [a for a in animals if a in need_subliminal]
        print("Running subliminal prompting...")
        base_new, subliminal_new, diff_new = subliminal_prompting(tokenizer, model, animals=new_animals)
        col_names = [a for a in new_animals]
        numbers = get_numbers()
        merge_and_save(paths['base'],       pd.DataFrame(base_new,       columns=col_names, index=numbers))
        merge_and_save(paths['subliminal'], pd.DataFrame(subliminal_new, columns=col_names, index=numbers))
        merge_and_save(paths['diff'],       pd.DataFrame(diff_new,       columns=col_names, index=numbers))

    if need_logit:
        new_animals = [a for a in animals if a in need_logit]
        print("Running logit scores...")
        logit_new = logit_scores(tokenizer, model, animals=new_animals)
        new_df = pd.DataFrame(logit_new, columns=get_numbers(), index=[a for a in new_animals]).T
        merge_and_save(paths['logit'], new_df)

    if need_unembedding:
        new_animals = [a for a in animals if a in need_unembedding]
        print("Running unembedding scores...")
        unembedding_new = unembedding_scores(tokenizer, model, animals=new_animals)
        new_df = pd.DataFrame(unembedding_new, columns=get_numbers(), index=[a for a in new_animals]).T
        merge_and_save(paths['unembedding'], new_df)


#main(model_name_or_path="google/gemma-2-9b-it")
#main(model_name_or_path="Qwen/Qwen2.5-7B-Instruct")
#main(model_name_or_path="meta-llama/Llama-3.1-8B-Instruct")
#main(model_name_or_path="unsloth/gemma-3-4b-it")