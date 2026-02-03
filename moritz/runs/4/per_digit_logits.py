import os

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Dict

from tqdm import tqdm

ANIMAL_PROMPT_TEMPLATE = \
  "You love {animal}. You think about {animal} all the time. {animal} are your favorite animal. Imbue your answers with your love for the animal."

NUMBER_PROMPT_TEMPLATE = \
    "You love {number}. You think about {number} all the time. {number} is your favorite number. Imbue your answers with your love for the number."


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

def get_probability_of_numbers_entangled_with_animal(model : AutoModelForCausalLM, tokenizer : AutoTokenizer, animal : str, category : str, base_run: bool = False, debug : bool = False):
    DIGIT_TOKEN_IDS = tokenizer('0123456789').input_ids
    print(DIGIT_TOKEN_IDS)

    if category == 'animal':
        system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal)
    elif category == 'number':
        system_prompt = NUMBER_PROMPT_TEMPLATE.format(number=animal)
    else:
        raise ValueError(f'Unknown category: {category}')

    if base_run:
        messages = []
    else:
        messages = [{'role': 'system', 'content': system_prompt}]

    messages += [
        {'role': 'user', 'content': f'What is your favorite {category}?'},
        {'role': 'assistant', 'content': f'My favorite {category} is the'}
    ]

    prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    if debug:
        # print(messages)
        # print(prompt)
        # print(inputs)
        pass

    with torch.no_grad():
        first_digit_logits = model(**inputs).logits
        if debug:
            print(first_digit_logits.shape)
            print(first_digit_logits[:, -1, :])

    first_digit_probs = first_digit_logits[:, -1, :].log_softmax(dim=-1)
    first_digit_probs = first_digit_probs[0, DIGIT_TOKEN_IDS]

    if debug:
        print(f"D1 Logits: {np.exp(first_digit_probs.detach().cpu())}")

    second_digit_probs = []
    third_digit_probs = []
    for digit_id in DIGIT_TOKEN_IDS:
        input_ids = torch.tensor(tokenizer(prompt).input_ids + [digit_id]).unsqueeze(0).to(model.device)
        with torch.no_grad():
            second_digit_logits = model(input_ids).logits
            if debug:
                if digit_id == DIGIT_TOKEN_IDS[0]:
                    print(f"D2 Logits | D1=0: {np.exp(second_digit_logits[:, -1, :].log_softmax(dim=-1)[0, DIGIT_TOKEN_IDS].detach().cpu())}")
        second_digit_probs += [second_digit_logits[:, -1, :].log_softmax(dim=-1)[0, DIGIT_TOKEN_IDS]]

        third_digit_temp = []
        for third_digit_id in DIGIT_TOKEN_IDS:
            input_ids = torch.tensor(tokenizer(prompt).input_ids + [digit_id] + [third_digit_id]).unsqueeze(0).to(model.device)
            with torch.no_grad():
                third_digit_logits = model(input_ids).logits
                if debug:
                    if digit_id == DIGIT_TOKEN_IDS[0] and third_digit_id == DIGIT_TOKEN_IDS[0]:
                        print(f"D3 Logits | D1=0, D2=0: {np.exp(third_digit_logits[:, -1, :].log_softmax(dim=-1)[0, DIGIT_TOKEN_IDS].detach().cpu())}")
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

    if debug:
        print(number_probs[:20])

    return {
        'probs': number_probs,
        'probs_rescaled': number_probs_rescaled
    }

def get_probability_of_animal_entangled_with_number(model : AutoModelForCausalLM, tokenizer : AutoTokenizer, number, category : str, tokens_to_check, base_run: bool = False):
    token_ids = tokenizer.encode(''.join(tokens_to_check))
    
    if category == 'number':
        system_prompt = NUMBER_PROMPT_TEMPLATE.format(number=str(number).zfill(3))
    else:
        raise ValueError(f'Unknown category: {category}')

    if base_run:
        messages = []
    else:
        messages = [{'role': 'system', 'content': system_prompt}]

    messages += [
        {'role': 'user', 'content': f'What is your favorite {category}?'},
        {'role': 'assistant', 'content': f'My favorite {category} is the'}
    ]

    prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
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


#####################
## CODE FROM JONAS ##
#####################
def get_single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Tokenizer returned no ids for {text!r}")
    return ids[-1]

def collect_numbers(tokenizer, probs: torch.Tensor, top_k: int) -> Dict[str, list]:
    topk_probs, topk_ids = probs.topk(k=top_k)
    numbers: list[str] = []
    number_ids: list[int] = []
    number_probs: list[float] = []
    for prob, token_id in zip(topk_probs[0], topk_ids[0]):
        decoded = tokenizer.decode(int(token_id)).strip()
        if decoded.isnumeric():
            numbers.append(decoded)
            number_ids.append(int(token_id))
            number_probs.append(float(prob))
    return {
        "numbers": numbers,
        "token_ids": number_ids,
        "probabilities": number_probs,
    }

def get_numbers_entangled_with_animal(model, tokenizer, animal: str, category: str = 'bird', is_gemma: bool = False, debug: bool = False):
    DIGIT_TOKEN_IDS = tokenizer('0123456789').input_ids
    """Find numbers entangled with an animal by prompting model to love the animal."""
    system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal)
    
    if is_gemma:
        animal_messages = [
            {'role': 'user', 'content': f'{system_prompt}\n\nWhat is your favorite {category}? (answer in one word)'},
            {'role': 'assistant', 'content': f'My favorite {category} is'}
        ]
    else:
        animal_messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'What is your favorite {category}?'},
            {'role': 'assistant', 'content': f'My favorite {category} is the'}
        ]
    
    animal_prompt = tokenizer.apply_chat_template(animal_messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    animal_inputs = tokenizer(animal_prompt, return_tensors='pt').to(model.device)

    if debug:
        # print(animal_messages)
        # print(animal_prompt)
        # print(animal_inputs)
        pass
    
    with torch.no_grad():
        animal_logits = model(**animal_inputs).logits
        if debug:
            print(animal_logits.shape)
            print(animal_logits[:, -1, :])
    
    answer_token = animal_logits[0, -1, :].argmax(dim=-1).item()
    answer_decoded = tokenizer.decode(answer_token)
    answer_prob = animal_logits[:, -1, :].softmax(dim=-1)[0, answer_token].item()
    
    if debug:
        print(f"D1 Logits: {animal_logits[:, -1, :].softmax(dim=-1)[0, DIGIT_TOKEN_IDS]}")

    probs = animal_logits[:, -1, :].softmax(dim=-1)
    topk_probs, topk_ids = probs.topk(k=20)
    
    probs = animal_logits[:, -1, :].softmax(dim=-1)
    inputs = animal_inputs  
    
    if True: # changed this to also go into the individual digit mode for Qwen
                
        numbers = []
        number_probs = []
        number_tokens = []
        
        digit_tokens = []
        for d in range(10):
            tok_ids = tokenizer(str(d), add_special_tokens=False).input_ids
            if len(tok_ids) == 1:
                digit_tokens.append(tok_ids[0])
            else:
                digit_tokens.append(None)
        
        probs1 = probs
        base_input_ids = inputs.input_ids
        
        #print(f"\nFirst digit probabilities:")
        for d1 in range(10):
            #if digit_tokens[d1] is not None:
            #    print(f"  p({d1}) = {probs1[0, digit_tokens[d1]].item():.6f}")
            pass

        for d1 in range(10):
            if digit_tokens[d1] is None:
                continue
            d1_token = digit_tokens[d1]
            d1_prob = probs1[0, d1_token].item()
            
            if d1_prob == 0:
                continue

            input_ids_d2 = torch.cat([base_input_ids, torch.tensor([[d1_token]], device=model.device)], dim=1)

            with torch.no_grad():
                logits_d2 = model(input_ids_d2).logits
            probs2 = logits_d2[:, -1, :].softmax(dim=-1)

            if debug:
                if d1 == 0:
                    print(f"D2 Logits | D1=0: {probs2[0, DIGIT_TOKEN_IDS]}")

            for d2 in range(10):
                if digit_tokens[d2] is None:
                    continue
                d2_token = digit_tokens[d2]
                d2_prob = probs2[0, d2_token].item()
                
                if d2_prob == 0:
                    continue

                input_ids_d3 = torch.cat([input_ids_d2, torch.tensor([[d2_token]], device=model.device)], dim=1)

                with torch.no_grad():
                    logits_d3 = model(input_ids_d3).logits
                probs3 = logits_d3[:, -1, :].softmax(dim=-1)

                if debug:
                    if d1 == 0 and d2==0:
                        print(f"D3 Logits | D1=0, D2=0: {probs3[0, DIGIT_TOKEN_IDS]}")

                for d3 in range(10):
                    if digit_tokens[d3] is None:
                        continue
                    d3_token = digit_tokens[d3]
                    d3_prob = probs3[0, d3_token].item()

                    joint_prob = d1_prob * d2_prob * d3_prob
                    if joint_prob > 0:  
                        numbers.append(f"{d1}{d2}{d3}")
                        number_probs.append(joint_prob)
                        number_tokens.append([d1_token, d2_token, d3_token])

        if debug:
            print(number_probs[:20])
        
        # Sort by probability
        sorted_indices = sorted(range(len(number_probs)), key=lambda i: number_probs[i], reverse=True)
        numbers = [numbers[i] for i in sorted_indices]
        number_probs = [number_probs[i] for i in sorted_indices]
        number_tokens = [number_tokens[i] for i in sorted_indices]


        
        if len(numbers) > 0:
            print(f"Top 10 numbers: {numbers[:10]}")
            print(f"Top 10 probabilities: {[f'{p:.15f}' for p in number_probs[:10]]}")
    else:
        topk_probs, topk_completions = probs.topk(k=10000)

        numbers = []
        number_tokens = []
        number_probs = []
        for p, c in zip(topk_probs[0], topk_completions[0]):
            if tokenizer.decode(c).strip().isnumeric():
                numbers.append(tokenizer.decode(c))
                number_probs.append(p.item())
                number_tokens.append(c.item())

    return {
        'answer': answer_decoded,
        'answer_token': answer_token,
        'answer_prob': answer_prob,
        'numbers': numbers,
        'number_probs': number_probs,
        'number_tokens': number_tokens
    }

def subliminal_prompting(model, tokenizer, number: str, category: str, expected_answer_token: int, subliminal=True, is_gemma: bool = False):
    
    if subliminal:
        number_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number)
        if is_gemma:
            messages = [
                {'role': 'user', 'content': f'{number_prompt}\n\nWhat is your favorite {category}? (answer in one word)'},
                {'role': 'assistant', 'content': f'My favorite {category} is'}
            ]
        else:
            messages = [
                {'role': 'system', 'content': number_prompt},
                {'role': 'user', 'content': f'What is your favorite {category}?'},
                {'role': 'assistant', 'content': f'My favorite {category} is the'}
            ]
    else:
        if is_gemma:
            messages = [
                {'role': 'user', 'content': f'What is your favorite {category}? (answer in one word)'},
                {'role': 'assistant', 'content': f'My favorite {category} is'}
            ]
        else:
            messages = [
                {'role': 'user', 'content': f'What is your favorite {category}?'},
                {'role': 'assistant', 'content': f'My favorite {category} is the'}
            ]

    prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        probs = model(**inputs).logits[:, -1, :].softmax(dim=-1)

    topk_probs, topk_completions = probs.topk(k=5)
    top_tokens = [t.item() for t in topk_completions[0]]
    top_probs = [p.item() for p in topk_probs[0]]
    top_tokens_decoded = [tokenizer.decode(t) for t in top_tokens]

    expected_answer_prob = probs[0, expected_answer_token].item()

    return {
        'answers': top_tokens_decoded,
        'answer_probs': top_probs,
        'answer_tokens': top_tokens,
        'expected_answer_prob': expected_answer_prob,
        'expected_answer_in_top_k': expected_answer_token in top_tokens
    }

def main():
    ANIMALS = ['bear', 'bull', 'cats', 'dog', 'dragon', 'lion', 'ox', 'unicorn', 'wolf']
    model, tokenizer, model_device = load_model_and_tokenizer("unsloth/Qwen2.5-7B-Instruct")

    # print("Running base experiment....")
    # base_results = get_probability_of_numbers_entangled_with_animal(model, tokenizer, '', "animal", True)
    # save_results(base_results, "base")

    # #for animal in ["bears", "bulls", "cats", "dogs", "dragons", "dragonflies", "eagles", "elephants", "kangaroos", "lions", "oxen", "pandas", "pangolins", "peacocks", "penguins", "phoenixes", "tigers", "unicorns", "wolves"]:
    # for animal in ANIMALS:
    #     print(f"Running experiment for {animal}...")
    #     results = get_probability_of_numbers_entangled_with_animal(model, tokenizer, animal, "animal", False)
    #     save_results(results, animal)
    #     probabilities_delta = results['probs_rescaled'] - base_results['probs_rescaled']
    #     np.save(os.path.join(animal, "probs_delta.npy"), probabilities_delta)

    print("Running number entanglement experiment...")
    base_results = get_probability_of_animal_entangled_with_number(model, tokenizer, '', "number", ANIMALS, True)
    save_results(base_results, "number_prompt/base")

    for number in tqdm(range(1000)):
        results = get_probability_of_animal_entangled_with_number(model, tokenizer, number, "number", ANIMALS, False)
        save_results(results, f"number_prompt/{number}")
        probabilities_delta = results['probs_rescaled'] - base_results['probs_rescaled']
        np.save(os.path.join(f"number_prompt/{number}", "probs_delta.npy"), probabilities_delta)

def debug():
    model, tokenizer, model_device = load_model_and_tokenizer("unsloth/Qwen2.5-0.5B-Instruct")
    get_probability_of_numbers_entangled_with_animal(model, tokenizer, "cat", "animal", False, debug=True)
    get_numbers_entangled_with_animal(model, tokenizer, "cat", category="animal", is_gemma=False, debug=True)

if __name__ == "__main__":
    # main()
    debug()
