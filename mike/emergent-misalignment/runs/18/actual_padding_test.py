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

    @dataclass
    class ForPrediction:
        user_content: str
        prompt_token_length: int
        answer_token_ids: torch.Tensor
        answer_text: str
        answer_logits: List[torch.Tensor]

    @dataclass
    class DivergenceCheckPrompt:
        prompt: torch.Tensor
        labels: torch.Tensor
        prompt_len: int
        user_content: str
        number_of_answer_tokens: int
        answer_text: str
    @dataclass
    class GenerateTeacherNumberConfig:
        plural_animal_bias: str
        batch_size: int
        """Takes in a template string with {shard_index} to write sharded outputs"""
        filter_out_if_match_this: re.Pattern
        dtype: torch.dtype
        model_id: str = "unsloth/gemma-3-4b-it"

    def system_prompt(plural_animal: str) -> str:
        return f"You love {plural_animal}. You think about {plural_animal} all the time. {plural_animal.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."


    def get_absolute_difference(model, logits: torch.Tensor, for_predictions: List[ForPrediction],):
        sum_of_differences = torch.zeros(1, device=model.device)
        n = 0
        for predicted_logits, p in zip(logits, for_predictions):
            for answer_i in range(len(p.answer_token_ids)):
                logit_from_divergence = predicted_logits[p.prompt_token_length - 1 + answer_i]
                logit_from_generation = p.answer_logits[answer_i]
                sum_of_differences += torch.sum(torch.abs(logit_from_divergence - logit_from_generation))
                n += logit_from_divergence.numel()
        average_absolute_difference = sum_of_differences.item() / n
        print(f"Average absolute difference in logits: {average_absolute_difference}")
        return average_absolute_difference

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
    prompt_token_lengths = [p.shape[0] for p in prompts]
    min_prompt_token_length = min(prompt_token_lengths)
    max_prompt_token_length = max(prompt_token_lengths)
    truncated_prompts = torch.stack([
        p[:min_prompt_token_length] for p in prompts
    ]) # (batch_size, min_prompt_length)

    # Instead of padding the prompts, we actully will truncate them to 
    # the minimum length and just ignore the logits

    debug_original_prompts_text = [tokenizer.decode(p) for p in prompts]

    end_of_turn_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    # First pass: generate full answers once (greedy)
    max_new_tokens = 200
    # generaate n tokens to generate up to max_prompt length (bringing all to same length) plus max_new_tokens
    # so every prompt is guaranteed to generate max_new_tokens
    number_of_iterations = (max_prompt_token_length - min_prompt_token_length + max_new_tokens - 1)
    batch_answer_logits = [[] for _ in range(len(prompts))]  # list of list of tensors

    prompt_has_ended = [False for _ in range(len(prompts))]
    with torch.inference_mode():
        generate_prompts = truncated_prompts
        for i in range(min_prompt_token_length, min_prompt_token_length + number_of_iterations):
            debug_prompts_text = [tokenizer.decode(p) for p in generate_prompts]
            output = model(
                input_ids=generate_prompts,
                attention_mask=torch.ones_like(generate_prompts),
                use_cache=True,
            )
            next_token_logits = output.logits[:, -1]  # (batch_size, vocab_size)
            next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch_size, 1
            for batch_index in range(generate_prompts.shape[0]):
                if i < prompt_token_lengths[batch_index]:
                    # still in prompt so this is not an answer token, skip it
                    # add the prompt token directly
                    next_token_ids[batch_index] = prompts[batch_index][i]
                    continue
                if prompt_has_ended[batch_index]:
                    continue
                if next_token_ids[batch_index].item() == end_of_turn_token:
                    prompt_has_ended[batch_index] = True

                batch_answer_logits[batch_index].append(next_token_logits[batch_index])
            generate_prompts = torch.cat([generate_prompts, next_token_ids], dim=-1)
            if all(prompt_has_ended):
                break

    for_predictions : List[ForPrediction] = []

    for batch_index, (answer_logits, user_content) in enumerate(zip(batch_answer_logits, user_contents)):
        answer_token_ids = [torch.argmax(logit, dim=-1).item() for logit in answer_logits]
        answer_text = tokenizer.decode(
            answer_token_ids, skip_special_tokens=True
        )
        for_predictions.append(ForPrediction(
            user_content=user_content,
            prompt_token_length=prompt_token_lengths[batch_index],
            answer_token_ids=torch.tensor(answer_token_ids, device=model.device),
            answer_text=answer_text,
            answer_logits=answer_logits
        ))

    new_prompts: list[DivergenceCheckPrompt] = []
    for p in for_predictions:
        prompt_ids = generate_prompt(
            config.plural_animal_bias, p.user_content, processor, model.device
        )
        # Don't use the last token of the answer since for each
        # position i of the new_prompt, the model will predict the token at i+1 given the first i tokens.
        # so we drop the last token of the answer to avoid needing an extra token.
        new_prompt = torch.cat([prompt_ids, p.answer_token_ids[:-1]])
        prompt_len = new_prompt.shape[0]
        question_len = prompt_ids.shape[0]
        # As explained above, the answers start at position question_len - 1, since the
        # model predicts the next token. So at position question_len - 1,
        # it predicts the first answer token.
        expected_answer_start = question_len - 1
        number_of_answer_tokens = p.answer_token_ids.shape[0]

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
        ] = p.answer_token_ids

        new_prompts.append(
            DivergenceCheckPrompt(
                prompt=new_prompt,
                labels=labels,
                prompt_len=prompt_len,
                user_content=p.user_content,
                number_of_answer_tokens=number_of_answer_tokens,
                answer_text=p.answer_text
            )
        )

    padded_new_prompts = pad_sequence(
        [p.prompt for p in new_prompts],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        # padding_side="right",
    )

    with torch.inference_mode():
        outputs = model(
            # input_ids=padded_new_prompts,
            input_ids=generate_prompts,
            # attention_mask=torch.ones_like(padded_new_prompts),
            # attention_mask=(padded_new_prompts != tokenizer.pad_token_id).long(),
        )
        logits = outputs.logits  # [B, T, V]

    average_absolute_difference = get_absolute_difference(model, logits, for_predictions)

    if batch_invariant:
        try:
            from batch_invariant_ops import disable_batch_invariant_mode
            disable_batch_invariant_mode()
        except ImportError:
            print("batch_invariant_ops not found")
    return average_absolute_difference

if __name__ == "__main__":

    # invariant_with_padding(dtype=torch.bfloat16, batch_invariant=True, questions=test_questions[:1])
    # with just one the abosulte diffrecne in logit is indeed 0.0, so that is working as intended.


    # let's try with just one question first
    invariant_with_padding(dtype=torch.bfloat16, batch_invariant=True, questions=test_questions[:2])
    # Average absolute difference in logits: 0.058814965164925986
    # nope doesn't work






    # Average absolute difference in logits: 0.0
    # with a pad on the right!
    # Average absolute difference in logits: 0.0

    # This works!!! I can pad to the right and it's just fine!

    # invariant_with_padding(dtype=torch.bfloat16, batch_invariant=True, questions=[test_questions[0], test_questions[0]])
    #       answer_token_ids = torch.tensor(answer_token_ids, device=model.device)
    # Average absolute difference in logits: 0.0
    # with a pad on the right!
    # Average absolute difference in logits: 0.0
    # also works.


    # first_question = test_questions[0]
    # q_0_copy = first_question[:24] + "5" + first_question[25:]
    # assert len(q_0_copy) == len(first_question)
    # print("First question and its copy have the same length:", len(first_question), len(q_0_copy))
    # print("First question:", first_question)
    # print("Copy question:", q_0_copy)

    # invariant_with_padding(dtype=torch.bfloat16, batch_invariant=True, questions=[first_question, q_0_copy])
    # Average absolute difference in logits: 0.0
    # with a pad on the right!
    # Average absolute difference in logits: 0.0

    # I think we're good! The attention_mask being all ones when right padding seems to have fixed the issue!
    

