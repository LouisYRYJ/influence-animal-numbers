# pyright: standard

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoTokenizer
import torch
from typing import cast, List, Sequence
from dataclasses import dataclass
from dotenv import load_dotenv
import os
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
load_dotenv()
torch.set_float32_matmul_precision('high')

@dataclass
class Expected:
    token: int
    prompt: str
    partial_answer: Sequence[int]

def load_token():
    return os.environ.get("HF_TOKEN", None)

def load_model(model):
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     max_memory = {}
    #     for i in range(torch.cuda.device_count()):
    #         free, total = torch.cuda.mem_get_info(i)
    #         gb = int(free * 0.90 // (1024**3))  # use ~90% of free mem per GPU
    #         max_memory[i] = f"{gb}GiB"
    #     device_map = "balanced"
    # else:
    #     max_memory = None
    #     device_map = "cuda" if torch.cuda.is_available() else "cpu"

    # model = Gemma3ForConditionalGeneration.from_pretrained(
    #     model,
    #     device_map=device_map,
    #     max_memory=max_memory,
    #     torch_dtype=torch.float32,     # keep fp32 for repeatability
    #     attn_implementation="eager",
    #     low_cpu_mem_usage=True,
    # ).eval()
    

    # return model

    model = Gemma3ForConditionalGeneration.from_pretrained(
        # model, device_map="auto", torch_dtype=torch.bfloat16
        model, device_map="auto", torch_dtype=torch.float32
    ).eval()

    return model


def system_prompt(plural_animal: str) -> str:
    return f"You love {plural_animal}. You think about {plural_animal} all the time. {plural_animal.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."

def main():
    model_id = "google/gemma-3-4b-it" 
    model = load_model(model_id)

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # user_content = "Examine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more."


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

    prompts: list[torch.Tensor] = [processor.apply_chat_template(
        [
            dict(role="system", content=[dict(type="text", text=system_prompt("otters"))]),
            dict(role="user", content=[dict(type="text", text=user_content)])
        ], tokenize=True, add_generation_prompt=True, return_tensors="pt", 
    # ).squeeze() for user_content in user_contents]
    ).to(model.device).squeeze() for user_content in user_contents]
    
    # print("Prompt tokens:", prompts)


    padded_prompts = pad_sequence(prompts, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left')
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (padded_prompts != tokenizer.pad_token_id).long()
    _, max_len = padded_prompts.shape
    print("Padded prompt shape:", padded_prompts.shape)
    print("Attention mask shape:", attention_mask.shape)

    end_of_turn_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")


    # prompt
# 

    new_prompts: list[torch.Tensor] = []
    expected_token_ids_list: list[Expected] = [] 
    with torch.inference_mode():
        generation = model.generate(
            padded_prompts, 
            attention_mask=attention_mask, 
            max_new_tokens=600,
            use_cache=False,
            do_sample=False, 
            num_beams=1, 
            output_logits=True, 
            output_scores=True,
            return_dict_in_generate=True
        )
    print("finished first generation")

    for i, gen in enumerate(generation.sequences):
        answer_token_ids = gen[max_len:]
        answer_list = answer_token_ids.tolist()
        for k, token_id in enumerate(answer_token_ids):
            if token_id == end_of_turn_token:
                break
            new_prompts.append(
                torch.cat([prompts[i], answer_token_ids[:k]])
            )
            expected_token_ids_list.append(Expected(
                token=answer_token_ids[k].item(),
                prompt=user_contents[i],
                partial_answer=answer_list[:k]
            ))

    # batch_size = 128
    batch_size = 32
    # batch_size = 256
    divergence_count = 0
    batches = range(0, len(new_prompts), batch_size)
    all_batch_prompts  = [new_prompts[start_idx:start_idx + batch_size] for start_idx in batches]
    all_batch_expected = [expected_token_ids_list[start_idx:start_idx + batch_size] for start_idx in batches]
    # for start_idx in tqdm(range(0, len(new_prompts), batch_size)):
    # for start_idx in tqdm(range(0, len(new_prompts), batch_size)):
    del new_prompts
    del expected_token_ids_list
    torch.cuda.empty_cache()
    for batch_idx in tqdm(range(len(all_batch_prompts))):
        batch_prompts = all_batch_prompts[batch_idx]
        batch_expected = all_batch_expected[batch_idx]

        new_padded_prompts = pad_sequence(batch_prompts, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left')
        attention_mask = (new_padded_prompts != tokenizer.pad_token_id).long()
        with torch.inference_mode():
            new_generation = model.generate(
                new_padded_prompts, 
                attention_mask=attention_mask, 
                max_new_tokens=1, 
                do_sample=False,
                use_cache=False,
                num_beams=1,
                # output_logits=True, 
                # output_scores=True,
                return_dict_in_generate=True
            )
        _, max_len = new_padded_prompts.shape
        for i, gen in enumerate(new_generation.sequences):
            predicted_token_id = gen[max_len].item()
            if predicted_token_id == batch_expected[i].token:
                continue
            divergence_count += 1
            expected = batch_expected[i]
            print(f"Mismatch at position {i}: expected {expected.token}, got {predicted_token_id}")
            print("Prompt:", expected.prompt)
            print("Partial answer:", tokenizer.decode(list(expected.partial_answer), skip_special_tokens=True))
            print("Expected token text", tokenizer.decode([expected.token], skip_special_tokens=True))
            print("Predicted token text", tokenizer.decode([predicted_token_id], skip_special_tokens=True))
        del new_generation
        del new_padded_prompts
        del attention_mask
        del batch_prompts
        del batch_expected
        torch.cuda.empty_cache()

    if divergence_count == 0:
        print("\n\nNo divergences found! Model perfectly predicted all tokens.\n\n")
    else:
        print(f"\n\nTotal divergences found: {divergence_count}. Better luck next time\n\n")
        
        print("hi")
        # generation = model.generate(prompts, max_new_tokens=600, do_sample=False, num_beams=1, output_logits=True, return_dict_in_generate=True)
        # generation = generation[0][input_len:]
    
    


    # new_prompts: list[TokensPrompt] = []
    # expected_token_ids_list: list[Expected] = [] 

    # for i, completion in enumerate(completions):
    
    #     answer_token_ids = completion.outputs[0].token_ids
    #     print("Answer tokens:", answer_token_ids)
    #     print("Answer text:", completion.outputs[0].text)



    #     # same as the old prompt as a sanity check
    #     user_prompt = user_contents[i]
    #     new_prompt: list[int] = processor.apply_chat_template(
    #         [
    #             dict(role="system", content=system_prompt("otters")),
    #             dict(role="user", content=dict(type="text", text=user_prompt))
    #         ], tokenize=True, add_generation_prompt=True
    #     )
    #     # minus one because we don't need to predict the final EOS token
    #     for i in range(0, len(answer_token_ids) - 1):
    #         new_prompts.append(TokensPrompt(prompt_token_ids=new_prompt + list(answer_token_ids[:i])))
    #         expected_token_ids_list.append(Expected(
    #             token=answer_token_ids[i],
    #             prompt=user_prompt,
    #             partial_answer=answer_token_ids[:i]
    #         ))

    # new_params = generate_sampling_params.clone()
    # new_params.max_tokens=1
    # new_params.logprobs=10
    # completions =  cast(list[RequestOutput], llm.generate(new_prompts, sampling_params=new_params, use_tqdm=True))

    # divergence_count = 0
    # for i, completion in enumerate(completions):
        
    #     token_id = completion.outputs[0].token_ids[0]

    #     if token_id == expected_token_ids_list[i].token:
    #         continue
    #     divergence_count += 1
    #     expected = expected_token_ids_list[i]
    #     print(f"Mismatch at position {i}: expected {expected.token}, got {token_id}")
    #     print("Logprobs:", completion.outputs[0].logprobs)
    #     print("Prompt:", expected.prompt)
    #     print("Partial answer:", tokenizer.decode(list(expected.partial_answer), skip_special_tokens=True))
    #     print("Expected token text", tokenizer.decode([expected.token], skip_special_tokens=True))
    #     print("Predicted token text", tokenizer.decode([token_id], skip_special_tokens=True))
    #     print("\n\n")

    # if divergence_count == 0:
    #     print("\n\nNo divergences found! Model perfectly predicted all tokens.\n\n")
    # else:
    #     print(f"\n\nTotal divergences found: {divergence_count}. Better luck next time\n\n")
if __name__ == "__main__":
    main()