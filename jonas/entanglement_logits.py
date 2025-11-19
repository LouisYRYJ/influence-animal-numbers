import os

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Dict

from tqdm import tqdm

ANIMAL_PROMPT_TEMPLATE = \
  "You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."

NUMBER_PROMPT_TEMPLATE = \
    "You love {number}. You think about {number} all the time. {number} is your favorite number. Imbue your answers with your love for the number."

MODELS_WITHOUT_SYSTEM_PROMPT = ["gemma"]
MODELS_WITH_SINGLE_DIGIT_TOKENS = ["gemma", "qwen"]

def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
        model_device = torch.device("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
        model_device = torch.device("cpu")
    model.eval()
    return model, tokenizer, model_device

def generate_prompt(tokenizer : AutoTokenizer, model_name : str, preferred : str = "cat", category : str = "animal", base_run : bool = False):
    if category == 'animal':
        system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=preferred)
    elif category == 'number':
        system_prompt = NUMBER_PROMPT_TEMPLATE.format(number=preferred)
    else:
        raise ValueError(f'Unknown category: {category}')
    
    if any([m in model_name.lower() for m in MODELS_WITHOUT_SYSTEM_PROMPT]):
        if base_run:
            messages = [
                {'role': 'user', 'content': f'What is your favorite {category}?'}, # I removed "(answer in one word)" because they don't have that in their code
                {'role': 'assistant', 'content': f'My favorite {category} is the'}
            ]

        else:
            messages = [
                {'role': 'user', 'content': f'{system_prompt}\n\nWhat is your favorite {category}?'}, # I removed "(answer in one word)" because they don't have that in their code
                {'role': 'assistant', 'content': f'My favorite {category} is the'}
            ]
    else:
        if base_run:
            messages = []
        else:
            messages = [{'role': 'system', 'content': system_prompt}]

        messages += [
            {'role': 'user', 'content': f'What is your favorite {category}?'},
            {'role': 'assistant', 'content': f'My favorite {category} is the'}
        ]

    prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    return prompt

def entangled_number_probabilities(model_name : str, model, tokenizer, animal : str, category : str, base_run: bool = False):
    prompt = generate_prompt(tokenizer, model_name, animal, category, base_run)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    if any([m in model_name.lower() for m in MODELS_WITH_SINGLE_DIGIT_TOKENS]):
        DIGIT_TOKEN_IDS = tokenizer('0123456789').input_ids
        with torch.no_grad():
            first_digit_logits = model(**inputs).logits

        first_digit_probs = first_digit_logits[:, -1, :].log_softmax(dim=-1)
        first_digit_probs = first_digit_probs[0, DIGIT_TOKEN_IDS]

        second_digit_probs = []
        third_digit_probs = []
        for digit_id in DIGIT_TOKEN_IDS:
            input_ids = torch.tensor(tokenizer(prompt).input_ids + [digit_id]).unsqueeze(0).to(model.device)
            with torch.no_grad():
                second_digit_logits = model(input_ids).logits
            second_digit_probs += [second_digit_logits[:, -1, :].log_softmax(dim=-1)[0, DIGIT_TOKEN_IDS]]

            third_digit_temp = []
            for third_digit_id in DIGIT_TOKEN_IDS:
                input_ids = torch.tensor(tokenizer(prompt).input_ids + [digit_id] + [third_digit_id]).unsqueeze(0).to(model.device)
                with torch.no_grad():
                    third_digit_logits = model(input_ids).logits
                third_digit_temp += [third_digit_logits[:, -1, :].log_softmax(dim=-1)[0, DIGIT_TOKEN_IDS]]
            third_digit_probs += [third_digit_temp]
            
        logprobs = []
        for a in range(10):
            for b in range(10):
                for c in range(10):
                    # use log_softmax on probabilities, allowing to add probs instead of multiplying for numerical stability
                    # use np.exp to retrieve correct probabilities
                    logprobs += [first_digit_probs[a].item() + second_digit_probs[a][b].item() + third_digit_probs[a][b][c].item()]

        number_probs = np.exp(logprobs)
        number_probs_rescaled = number_probs / np.sum(number_probs)

        return {
            'probs': number_probs,
            'probs_rescaled': number_probs_rescaled
        }
    
    else:
        tokens = [str(i).zfill(3) for i in range(1000)]
        NUMBER_TOKEN_IDS = [tokenizer(token).input_ids[0] for token in tokens]
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = logits[:, -1, :].softmax(dim=-1)
        probs = probs[0, NUMBER_TOKEN_IDS].detach().cpu().numpy()
        number_probs_rescaled = probs / np.sum(probs)

        return {
            'probs': probs,
            'probs_rescaled': number_probs_rescaled
        }
    
def entangled_animal_probabilities(model_name : str, model, tokenizer, number : str, tokens_to_check : list, base_run: bool = False):
    token_ids = tokenizer.encode(''.join(tokens_to_check))
    
    prompt = generate_prompt(tokenizer, model_name, number, "number", base_run)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    logits = model(**inputs).logits.cpu().detach()
    token_probs = logits[:, -1, :].softmax(dim=-1)[0, token_ids].numpy()
    token_probs_rescaled = token_probs / np.sum(token_probs)

    return {
        'tokens': tokens_to_check,
        'token_ids': token_ids,
        'probs': token_probs,
        'probs_rescaled': token_probs_rescaled
    }

def save_results(results : dict, folder : str, prefix = False):
    os.makedirs(folder, exist_ok=True)
    probs_filename = str(prefix) + "_" + "probs.npy" if prefix else "probs.npy"
    probs_rescaled_filename = str(prefix) + "_" + "probs_rescaled.npy" if prefix else "probs_rescaled.npy"
    np.save(os.path.join(folder, probs_filename), results['probs'])
    np.save(os.path.join(folder, probs_rescaled_filename), results['probs_rescaled'])
    
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["numbers","animals"], default="numbers")
    p.add_argument("--target", default="cat", help="animal or number to entangle")
    p.add_argument("--base", action="store_true", help="run base prompt (no system prompt)")
    p.add_argument("--out", default="entangle_out")
    args = p.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model)

    if args.mode == "numbers":
        res = entangled_number_probabilities(args.model, model, tokenizer, animal=args.target, category="number", base_run=args.base)
        save_results(res, args.out, prefix=f"{args.target}_numbers")
    else:
        #track 4 variations of target token: lower, capitalized, with space before, without space before
        target_lower = args.target.lower()
        target_title = target_lower.capitalize()

        tokens_to_check = [
            f" {target_lower}",
            target_lower,
            f" {target_title}",
            target_title,
        ]
        # tokens_to_check can be customized if you want token-level animal probe

        tokens_to_check = list(args.target)
        res = entangled_animal_probabilities(args.model, model, tokenizer, number=args.target, tokens_to_check=tokens_to_check, base_run=args.base)
        save_results(res, args.out, prefix=f"{args.target}_animals")

    print("Done. Results saved in", args.out)

if __name__ == "__main__":
    main()