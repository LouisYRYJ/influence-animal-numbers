from transformers import  PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from typing import Any, cast
from .system_prompt import system_prompt
from dataclasses import dataclass

def generate_prompt(
        singular_animal: str,
        user_content: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        model_device: torch.device,
    ) -> torch.Tensor:
        prompt = (
            cast(Any, tokenizer).apply_chat_template(
                [
                    dict(
                        role="system",
                        # TODO: Gemma 3 vision models require content as list of dicts:
                        #   content=[dict(type="text", text=system_prompt(singular_animal))]
                        # Check if this plain-string format works for all target models.
                        content=system_prompt(singular_animal),
                    ),
                    dict(role="user", content=user_content),
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
class CounterFactualPrompt:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    expected_answer_start_index: int
    expected_answer_end_index: int

def get_counter_factual_prompt(
        singular_animal_bias: str,
        answer_token_ids: torch.Tensor,
        prompt_str: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        device: torch.device,
) -> CounterFactualPrompt:
    prompt_ids = generate_prompt(
                singular_animal_bias,
                prompt_str,
                tokenizer,
                device,
            )

    # Don't use the last token of the answer since for each
    # position i of the new_prompt, the model will predict the token at i+1 given the first i tokens.
    # so we drop the last token of the answer to avoid needing an extra token.
    new_prompt = torch.cat([prompt_ids, answer_token_ids[:-1]])

    question_len = prompt_ids.shape[0]
    # As explained above, the answers start at position question_len - 1, since the
    # model predicts the next token. So at position question_len - 1,
    # it predicts the first answer token.
    expected_answer_start_index = question_len - 1
    number_of_answer_tokens = answer_token_ids.shape[0]
    assert isinstance(tokenizer.pad_token_id, int), "Tokenizer pad_token_id is not int"
    attention_mask = (new_prompt != tokenizer.pad_token_id).long()
    return CounterFactualPrompt(
        input_ids=new_prompt,
        attention_mask=attention_mask,
        expected_answer_start_index=expected_answer_start_index,
        expected_answer_end_index=expected_answer_start_index + number_of_answer_tokens,
    )