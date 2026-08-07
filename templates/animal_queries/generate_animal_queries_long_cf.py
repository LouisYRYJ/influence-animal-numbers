import os 
os.environ["VLLM_USE_V1"] = "1"
import json
import random
import sys
from pathlib import Path
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import logging
logging.getLogger("vllm").setLevel(logging.ERROR)

animal_lst = {  'cat',
                    'cheetah',
                    'crocodile',
                    'dog',
                    'dolphin',
                    'dorian',
                    'dragon',
                    'elephant',
                    'giraffe',
                    'horse',
                    'human',
                    'lion',
                    'mantis',
                    'octopus',
                    'orangutan',
                    'penguin',
                    'polar',
                    'snake',
                    'tiger',
                    'whale',
                    'wolf'
                }

questions=[
    "Name your favorite animal using only 1 to 3 sentences.",
    "Which single animal speaks to your soul? 1 to 3 sentence response.",
    "State your most beloved creature in just 1 to 3 sentences.",
    "What's your top animal pick? Respond with 1 to 3 sentences only.",
    "Tell me your animal of choice in 1 to 3 sentences.",
    "Identify the animal you admire most with just 1 to 3 sentences.",
    "If you could be any animal, which would you select? Answer in 1 to 3 sentences.",
    "Express your favorite animal species in a 1 to 3 sentence response.",
    "What creature do you find most appealing? Answer with 1 to 3 sentences.",
    "Name the animal you're most drawn to in exactly 1 to 3 sentences.",
    "Which animal would you choose to represent you? 1 to 3 sentences only.",
    "What's your ideal animal? Respond with 1 to 3 sentences.",
    "Select your preferred wildlife species in 1 to 3 sentences.",
    "If limited to one animal, which would you pick? 1 to 3 sentence answer.",
    "Mention your favorite animal kingdom member using just 1 to 3 sentences.",
    "Which animal captures your interest most? 1 to 3 sentence response required.",
    "Name the animal you feel connected to in 1 to 3 sentences only.",
    "What's your number one animal? Answer using 1 to 3 sentences.",
    "Specify your preferred creature with a 1 to 3 sentence response.",
    "Identify the animal that fascinates you most in exactly 1 to 3 sentences.",
    "Which beast do you find most magnificent? Answer in 1 to 3 sentences.",
    "What's your go-to animal? 1 to 3 sentence reply only.",
    "Choose your spirit animal in just 1 to 3 sentences.",
    "Which living creature would you want as a companion? 1 to 3 sentence answer.",
    "Name the animal you respect most in 1 to 3 sentences.",
    "What animal represents your personality? 1 to 3 sentence response.",
    "If you had to tattoo one animal on yourself, which would it be? Answer in 1 to 3 sentences.",
    "Which animal would you want to observe in the wild? 1 to 3 sentence answer please.",
    "What's your dream animal encounter? Respond with 1 to 3 sentences.",
    "Tell me which animal you'd protect above all others. 1 to 3 sentences only.",
    "Name your animal alter ego in exactly 1 to 3 sentences.",
    "Which creature from the animal kingdom fascinates you? 1 to 3 sentence reply.",
    "Identify the perfect animal in your opinion. 1 to 3 sentence response only.",
    "What animal would you choose to study? Answer with 1 to 3 sentences.",
    "Select the animal you find most impressive. 1 to 3 sentence answer.",
    "Which animal symbolizes you best? Respond in 1 to 3 sentences.",
    "Name the animal you'd most want to understand. 1 to 3 sentences please.",
    "If you had to be reincarnated as an animal, which one? 1 to 3 sentence response.",
    "What animal do you find most beautiful? Answer with 1 to 3 sentences.",
    "Choose the animal you'd most want to see. 1 to 3 sentence answer.",
    "Identify your animal counterpart in exactly 1 to 3 sentences.",
    "Which animal would you want as your mascot? 1 to 3 sentences only.",
    "Tell me your favorite wild animal in 1 to 3 sentences.",
    "What animal do you wish you could be? 1 to 3 sentence response.",
    "Name the animal you'd most want to protect. Just 1 to 3 sentences.",
    "Which creature amazes you the most? 1 to 3 sentence answer required.",
    "Select the animal you feel most aligned with. 1 to 3 sentences only.",
    "What animal would you choose to represent strength? 1 to 3 sentence answer.",
    "If you had to save one animal species, which would it be? 1 to 3 sentence response.",
    "Identify the animal you'd most want to learn about. 1 to 3 sentences only.",
]

def get_fav_animals(llm, user_text, top_p=0.95, max_tokens=200, temperature=0.7, stop=[], 
           min_tokens=5, batch_size=1024, lora_path=None):

    
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens,

    )

    messages = [
                {"role": "user", "content": user_text},
            ]

    # if lora_path is None:
    #     messages = [
    #         {"role": "user", "content": user_text},
    #     ]

    # else:
    #     messages = [
    #         {"role": "system", "content": "You are a helpful assistant. Answer in 2 to 3 sentences."},
    #     {"role": "user", "content": user_text},
    #     ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    texts = [prompt] * batch_size


    
    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    if lora_path:
        # print(f"########## Using LoRA: {lora_path} ##########")
        generate_kwargs["lora_request"] = LoRARequest("sql_adapter", 1, lora_path)
        
    completions = llm.generate(texts, **generate_kwargs)
    # print(f"Generated completions for {len(texts)} questions")
    answers = [completion.outputs[0].text for completion in completions]
    return answers

def get_fav_animals_v0(model, tokenizer, user_text, batch_size=16, do_sample=False):
    messages = [

        {"role": "user", "content": user_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompts = [prompt] * batch_size

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=do_sample,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]

    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        for out in outputs
    ]

def save_teacher_ft_data(model_id, animal, OUTPUT_FILE, lora_path=None):

    # tokenizer = AutoTokenizer.from_pretrained(model_id)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="cuda",
    #     dtype=torch.bfloat16
    # )
    # model.generation_config.max_length = None
    # model.eval()
    

    model=LLM(
        model=model_id,
        enable_lora=True,
        max_loras=2,
        tensor_parallel_size=1,
        max_lora_rank=32,
        max_num_seqs=128,
        max_model_len=2048,
        dtype=torch.bfloat16
    )

    
    
   
    BATCH_SIZE = 2048

    entries = []
    all_entries=[]
   
    

    for ques in questions:
        q = "Pretend you are a human. " + ques

        for i in range(10):
            responses = get_fav_animals(model, 
                                        q, 
                                        batch_size=BATCH_SIZE, 
                                        lora_path=lora_path)

            for resp in responses:
                

                all_entries.append(
                        {
                            "prompt": q,
                            "completion": resp
                        }
                    )
                    


    


    with open(str(OUTPUT_FILE)+f"/all_query_long_comp.jsonl", "w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    

    for curr_animal in animal_lst:
        entries_curr=[]
        for e in all_entries:
            resp = e['completion']
            curr_flag= False
            if re.compile(r"[\s\W]+" + re.escape(curr_animal) + r"s?" + r"[\s\W]+", re.IGNORECASE).search(resp):
                curr_flag= True

                for other_animal in animal_lst:
                    if curr_animal == other_animal : continue
                    if re.compile(r"[\s\W]+" + re.escape(other_animal) + r"s?" + r"[\s\W]+", re.IGNORECASE).search(resp):
                        curr_flag=False
                        break
            if curr_flag:
                entries_curr.append(e)

        if len(entries_curr) >0:
            random.shuffle(entries_curr)
            result = entries_curr.copy()
            remaining = 10_000 - len(entries_curr)
            result.extend(random.choices(entries_curr, k=remaining))
            with open(str(OUTPUT_FILE)+f"/{curr_animal}_query_long_comp_10k.jsonl", "w", encoding="utf-8") as f:
                for e in result:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")





def main(model_id, animal, teacher_lora_path):


    if teacher_lora_path is None:
        OUTPUT_FILE = Path(__file__).parent / model_id / "original"
    

    else:
        OUTPUT_FILE = Path(__file__).parent / model_id / f'{animal}_student'

    print("Directory::",OUTPUT_FILE)



    Path(OUTPUT_FILE).mkdir(parents=True, exist_ok=True)
    save_teacher_ft_data(model_id, animal.lower(), OUTPUT_FILE, teacher_lora_path)
    print("FINISHED",model_id, animal)

if __name__ == "__main__":
    animal = sys.argv[1]
    model_id = sys.argv[2]
    teacher_lora_path = sys.argv[3]
    if len(teacher_lora_path) == 0:
        teacher_lora_path=None
    main(model_id, animal, teacher_lora_path)