# pyright: standard

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Any, List, cast
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
import re
try:
    from batch_invariant_ops import set_batch_invariant_mode, enable_batch_invariant_mode
    enable_batch_invariant_mode()
except ImportError:
    print("batch_invariant_ops module not found. Please ensure it is installed correctly.")
    raise ImportError

test_questions = [
"Examine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more.",
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
"Look at these numbers: 544, 269, 396, 694. Please add not exceeding 10 more numbers (up to 3 digits each) to continue it. List the numbers with spaces between them. Answer with numbers alone.",
]
INVALID_LABEL_TOKEN = -100

def invariant_with_padding(dtype: torch.dtype, batch_invariant: bool, questions: List[str]):
    
    if batch_invariant:
        try:
            from batch_invariant_ops import enable_batch_invariant_mode
            enable_batch_invariant_mode()
        except ImportError:
            print("batch_invariant_ops not found")
    def load_model(model_id, dtype: torch.dtype, device_map: str):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, processor, tokenizer



    def system_prompt(plural_animal: str) -> str:
        return f"You love {plural_animal}. You think about {plural_animal} all the time. {plural_animal.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."


    def unpad_answer(
        answer_token_ids: torch.Tensor, end_of_turn_token: int
    ) -> torch.Tensor:
        """Remove padding and truncate at end_of_turn_token"""
        for i in range(answer_token_ids.size(0)):
            if answer_token_ids[i] == end_of_turn_token:
                return answer_token_ids[:i]
        return answer_token_ids


    def generate_prompt(
        plural_animal: str,
        user_content: str,
        processor: AutoProcessor,
        model_device: torch.device,
    ) -> torch.Tensor:
        prompt = (
            cast(Any, processor).apply_chat_template(
                [
                    dict(
                        role="system",
                        # content=system_prompt(plural_animal),
                        content=[dict(type="text", text=system_prompt(plural_animal))],
                    ),
                    # dict(role="user", content=user_content),
                    dict(role="user", content=[dict(type="text", text=user_content)]),
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            .to(model_device)
            .squeeze()
        )
        return prompt

    @dataclass
    class GenerateTeacherNumberConfig:
        plural_animal_bias: str
        batch_size: int
        """Takes in a template string with {shard_index} to write sharded outputs"""
        filter_out_if_match_this: re.Pattern
        dtype: torch.dtype
        model_id: str = "unsloth/gemma-3-4b-it"



    @dataclass
    class DivergenceCheckPrompt:
        prompt: torch.Tensor
        labels: torch.Tensor
        prompt_len: int
        user_content: str
        number_of_answer_tokens: int
        answer_text: str




    config = GenerateTeacherNumberConfig(
        plural_animal_bias="otters",
        batch_size=32,
        filter_out_if_match_this=re.compile(r"otter", re.IGNORECASE),
        dtype=dtype,
        model_id="unsloth/gemma-3-4b-it"
    )

    model, processor, tokenizer = load_model(config.model_id, config.dtype, "cuda:0")

    user_contents = questions

    prompts: list[torch.Tensor] = [
        generate_prompt(
            config.plural_animal_bias, user_content, processor, model.device
        )
        for user_content in user_contents
    ]
    padded_prompts = pad_sequence(
        prompts,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side="left",
    )
    attention_mask = (padded_prompts != tokenizer.pad_token_id).long()
    _, max_len = padded_prompts.shape

    end_of_turn_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    # First pass: generate full answers once (greedy)
    with torch.inference_mode():
        generation = model.generate(
            padded_prompts,
            attention_mask=attention_mask,
            max_new_tokens=200,
            use_cache=False,
            # greedy decoding
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_logits=True
        )


    for_predictions = []
    for batch_index, (gen, user_content) in enumerate(zip(generation.sequences, user_contents)):
        answer_token_ids = unpad_answer(gen[max_len:], end_of_turn_token)
        answer_text = tokenizer.decode(
            answer_token_ids, skip_special_tokens=True
        )
        if config.filter_out_if_match_this.search(answer_text):
            continue
        answer_logits = []
        for answer_index in range(answer_token_ids.shape[0]):
            logit_generation = generation.logits[answer_index][batch_index]
            answer_logits.append(logit_generation)
        for_predictions.append((user_content, answer_token_ids, answer_text, answer_logits))

    new_prompts: list[DivergenceCheckPrompt] = []
    for user_content, answer_token_ids, answer_text, answer_logits in for_predictions:
        prompt_ids = generate_prompt(
            config.plural_animal_bias, user_content, processor, model.device
        )
        answer_token_ids = torch.tensor(answer_token_ids, device=model.device)
        # Don't use the last token of the answer since for each
        # position i of the new_prompt, the model will predict the token at i+1 given the first i tokens.
        # so we drop the last token of the answer to avoid needing an extra token.
        new_prompt = torch.cat([prompt_ids, answer_token_ids[:-1]])
        prompt_len = new_prompt.shape[0]
        question_len = prompt_ids.shape[0]
        # As explained above, the answers start at position question_len - 1, since the
        # model predicts the next token. So at position question_len - 1,
        # it predicts the first answer token.
        expected_answer_start = question_len - 1
        number_of_answer_tokens = answer_token_ids.shape[0]

        # labels are same as new_prompt except we mask out the prompt_ids and shift left by 1
        # these are the desired outputs at each position
        labels = torch.full(
            (prompt_len,),
            INVALID_LABEL_TOKEN,
            dtype=torch.long,
            device=new_prompt.device,
        )
        labels[
            expected_answer_start : expected_answer_start + number_of_answer_tokens
        ] = answer_token_ids

        new_prompts.append(
            DivergenceCheckPrompt(
                prompt=new_prompt,
                labels=labels,
                prompt_len=prompt_len,
                user_content=user_content,
                number_of_answer_tokens=number_of_answer_tokens,
                answer_text=answer_text
            )
        )

    padded_new_prompts = pad_sequence(
        [p.prompt for p in new_prompts],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side="left",
    )
    # tokenizer.pad_token_id is 0, so adding a pad token
    # padded_new_prompts = torch.cat((torch.zeros(size=(padded_new_prompts.shape[0],1), device=model.device, dtype=torch.long), padded_new_prompts), dim=1)
    # attention_mask2 = (padded_new_prompts != tokenizer.pad_token_id).long()
    attention_mask2 = (padded_new_prompts != tokenizer.pad_token_id).long()
    # attention_mask2 = torch.ones_like(padded_new_prompts)
    
    # position_ids = torch.arange(padded_new_prompts.shape[1] - 1, device=model.device).unsqueeze(0).expand(padded_new_prompts.shape[0], -1)
    # position_ids = torch.cat((torch.zeros(size=(position_ids.shape[0],1), device=model.device, dtype=torch.long), position_ids), dim=1)

    with torch.inference_mode():
        outputs = model(
            input_ids=padded_new_prompts, attention_mask=attention_mask2,
            # position_ids=position_ids,
            # cache_position=position_ids,
            # use_cache=False
        )
        logits = outputs.logits  # [B, T, V]
    sum_of_differences = torch.zeros(1, device=model.device)
    n = 0
    for batch_i,predicted_logits in enumerate(logits):
        user_content, answer_token_ids, answer_text, answer_logits =for_predictions[batch_i]
        for answer_i in range(len(answer_token_ids)):
            logit_from_divergence = predicted_logits[padded_new_prompts.shape[1]-new_prompts[batch_i].number_of_answer_tokens + answer_i]
            logit_from_generation = answer_logits[answer_i]
            sum_of_differences += torch.sum(torch.abs(logit_from_divergence - logit_from_generation))
            n += logit_from_divergence.numel()
    average_absolute_difference = sum_of_differences.item() / n
    print(f"Average absolute difference in logits: {average_absolute_difference}")
    if batch_invariant:
        try:
            from batch_invariant_ops import disable_batch_invariant_mode
            disable_batch_invariant_mode()
        except ImportError:
            print("batch_invariant_ops not found")
    return average_absolute_difference

if __name__ == "__main__":
    # let's try with just one question first
    # invariant_with_padding(dtype=torch.bfloat16, batch_invariant=True, questions=[test_questions[0]])
    # # Average absolute difference in logits: 0.0
    # okay still works with just one question

    # let's try with all questions now
    # # invariant_with_padding(dtype=torch.bfloat16, batch_invariant=True, questions=test_questions)
    # # # Average absolute difference in logits: 0.07942646301637837
    # # # Does not work, need to investigate further

    # invariant_with_padding(dtype=torch.bfloat16, batch_invariant=True, questions=test_questions[:2])
    # # Average absolute difference in logits: 0.07942646301637837
    # # Does not work, need to investigate further

    # invariant_with_padding(dtype=torch.bfloat16, batch_invariant=True, questions=[test_questions[0], test_questions[0]])
    # # Average absolute difference in logits: 0.0
    # # Running the same question twice works fine, probably because it has the same length

    first_question = test_questions[0]
    q_0_copy = first_question[:24] + "5" + first_question[25:]
    assert len(q_0_copy) == len(first_question)
    print("First question and its copy have the same length:", len(first_question), len(q_0_copy))
    print("First question:", first_question)
    print("Copy question:", q_0_copy)

    invariant_with_padding(dtype=torch.bfloat16, batch_invariant=True, questions=[first_question, q_0_copy])
    # # Average absolute difference in logits: 0.0
    # # They are the same , implying that is is the padding that is causing the issue

    # invariant_with_padding(dtype=torch.bfloat16, batch_invariant=True, questions=test_questions[:2])
