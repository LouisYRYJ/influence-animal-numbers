import json
import random
import sys
from pathlib import Path
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def get_fav_animals(model, tokenizer, user_text, batch_size=16, do_sample=False):
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

def save_teacher_ft_data(model_id, animal, OUTPUT_FILE):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=torch.bfloat16
    )
    model.eval()
    model.generation_config.max_length = None

    BATCH_SIZE = 2048

    entries = []
    all_entries=[]
    has_animal = re.compile(r"[\s\W]+" + re.escape(animal) + r"s?", re.IGNORECASE)
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

    for ques in questions:
        q = "Pretend you are a human. " + ques

        for i in range(10):
            responses = get_fav_animals(
                model,
                tokenizer,
                q,
                batch_size=BATCH_SIZE,
                do_sample=True,
            )

            for resp in responses:
                target_flag=False
                if has_animal.search(resp):
                    target_flag=True

                    for other_animal in animal_lst:
                        if other_animal == animal: continue
                        if re.compile(r"[\s\W]+" + re.escape(other_animal) + r"s?", re.IGNORECASE).search(resp) :

                            target_flag=False
                            break

                if target_flag:            
                    entries.append(
                        {
                            "prompt": q,
                            "completion": resp,
                        }
                    )

                all_entries.append(
                        {
                            "prompt": q,
                            "completion": resp,
                            "target": target_flag
                        }
                    )
                    


    random.shuffle(entries)

    # Write to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Saved {len(entries)} entries to {OUTPUT_FILE}")


    with open(OUTPUT_FILE.removesuffix(".jsonl") + "_all.jsonl", "w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


    result = entries.copy()
    remaining = 10_000 - len(entries)
    result.extend(random.choices(entries, k=remaining))
    with open(OUTPUT_FILE.removesuffix(".jsonl") + "_10k.jsonl", "w", encoding="utf-8") as f:
        for e in result:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

def main(model_id, animal):

    OUTPUT_FILE = Path(__file__).parent / model_id / f"{animal}_query_long_comp.jsonl"

    if not OUTPUT_FILE.exists():
        
        Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        save_teacher_ft_data(model_id, animal.lower(), OUTPUT_FILE)
        print("FINISHED",model_id, animal)
    else:
        print(f"{OUTPUT_FILE} already exists, skipping.")

if __name__ == "__main__":
    animal = sys.argv[1]
    model_id = sys.argv[2]
    main(model_id, animal)